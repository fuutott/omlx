"""Portable converter checks, without importing Apple-only MLX."""
import copy
import math
import unittest

import torch

from tools import add_qwen4_mtp_q8 as mtp
from tools import quantize_qwen4_flash_next_t5 as base


class MTPConversionTests(unittest.TestCase):
    def test_exact_source_and_output_inventory(self):
        shapes = mtp.expected_source_shapes()
        self.assertEqual(len(shapes), 31)
        self.assertEqual(sum(math.prod(s) for s in shapes.values()), 2607150848)
        layout, specs = mtp.expected_output_layout()
        self.assertEqual(len(specs), 32)
        self.assertFalse(any(".experts." in k for k in layout))
        self.assertFalse(any("embed_tokens" in k or "lm_head" in k or ".ple." in k for k in layout))
        self.assertEqual(layout["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"],
                         {"shape": [512, 640, 640], "dtype": "U32"})
        self.assertEqual(layout["mtp.layers.0.mlp.switch_mlp.down_proj.scales"],
                         {"shape": [512, 2560, 10], "dtype": "BF16"})

    def test_gate_up_split_order_and_no_expert_interleave(self):
        dense = torch.arange(2 * 6 * 64).reshape(2, 6, 64).bfloat16()
        parts = mtp.split_experts("mtp.layers.0.mlp.experts.gate_up_proj", dense)
        self.assertTrue(torch.equal(parts["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"], dense[:, :3]))
        self.assertTrue(torch.equal(parts["mtp.layers.0.mlp.switch_mlp.up_proj.weight"], dense[:, 3:]))
        self.assertTrue(all(t.is_contiguous() for t in parts.values()))
        with self.assertRaises(ValueError):
            mtp.split_experts("mtp.surprise.weight", dense)

    def test_affine_packing_and_reconstruction(self):
        generator = torch.Generator().manual_seed(714)
        for shape in ((9, 128), (3, 7, 128)):
            weight = torch.randn(shape, generator=generator).bfloat16()
            name = "mtp.layers.0.mlp.switch_mlp.gate_proj.weight"
            entries, spec = mtp.quantize_tensor(name, weight, device="cpu", chunk_rows=4)
            module = name.removesuffix(".weight")
            self.assertEqual(spec, mtp.Q8)
            packed, scale, bias = (entries[module + "." + s] for s in ("weight", "scales", "biases"))
            self.assertEqual(list(packed.shape), [*shape[:-1], 32])
            codes = base.unpack_affine(packed, 128, 8).float().reshape(*shape[:-1], 2, 64)
            recovered = (codes * scale.float()[..., None] + bias.float()[..., None]).reshape(shape)
            relative_rmse = ((recovered - weight.float()).square().mean() / weight.float().square().mean()).sqrt()
            self.assertLess(relative_rmse.item(), 0.012)
            # Independent fixed byte-slot decoding, not only converter's unpacker.
            flat_words = packed.reshape(-1).to(torch.int64)
            independent = torch.stack([(flat_words >> shift) & 255 for shift in (0, 8, 16, 24)], dim=-1)
            self.assertTrue(torch.equal(independent.reshape(codes.shape), codes.to(torch.int64)))

    def test_norms_and_routers_remain_byte_exact(self):
        for name, shape in (("mtp.pre_fc_norm_hidden.weight", (128,)),
                            ("mtp.layers.0.mlp.gate.weight", (8, 64)),
                            ("mtp.layers.0.mlp.shared_expert_gate.weight", (1, 64))):
            weight = torch.randn(shape).bfloat16()
            entries, spec = mtp.quantize_tensor(name, weight, device="cpu", chunk_rows=4)
            self.assertIs(spec, False)
            self.assertTrue(torch.equal(entries[name], weight))

    def test_nonfinite_and_non_bf16_rejected(self):
        name = "mtp.fc_hidden.weight"
        for tensor in (torch.ones((2, 64)), torch.full((2, 64), float("nan")).bfloat16()):
            with self.assertRaises(ValueError):
                mtp.quantize_tensor(name, tensor, device="cpu", chunk_rows=4)

    def test_base_config_not_mutated_and_global_quant_unchanged(self):
        base_config = {"text_config": {"mtp_num_hidden_layers": 0, "ple_layer_ids": [2]},
                       "quantization": {"bits": 4, "group_size": 64, "target": {"bits": 2}},
                       "quantization_config": {"bits": 4, "group_size": 64, "target": {"bits": 2}}}
        original = copy.deepcopy(base_config)
        source = {"text_config": {"mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
                                  "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]}}}
        _, specs = mtp.expected_output_layout()
        result = mtp.augment_config(base_config, source, specs)
        self.assertEqual(base_config, original)
        self.assertEqual(result["text_config"]["mtp_num_hidden_layers"], 1)
        self.assertEqual(result["text_config"]["ple_layer_ids"], [2])
        self.assertEqual(result["quantization"]["bits"], 4)
        self.assertIs(result["quantization"]["mtp.layers.0.mlp.gate"], False)
        source["text_config"]["mtp_use_dedicated_embeddings"] = True
        with self.assertRaises(ValueError):
            mtp.augment_config(base_config, source, specs)


if __name__ == "__main__":
    unittest.main()
