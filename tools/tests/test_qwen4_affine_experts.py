"""Portable checks for the stock-oMLX affine expert format; no MLX or model download."""

import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from tools import quantize_qwen4_flash_next_t5 as q
from tools import qwen4_flash_next_imatrix as imatrix_module

GATE_UP = "model.language_model.layers.5.mlp.experts.gate_up_proj"
DOWN = "model.language_model.layers.5.mlp.experts.down_proj"
PREFIX = "language_model.model.layers.5.mlp.switch_mlp"
LAYER = "model.language_model.layers.3."


class AffineExpertTests(unittest.TestCase):
    def test_cli_defaults_and_guards(self):
        args = q.parse_args([
            "--model", "s", "--output", "o", "--expert-format", "affine", "--imatrix", "i.gguf",
        ])
        self.assertEqual(
            (args.expert_format, args.expert_gate_up_bits, args.expert_down_bits, args.imatrix_scope),
            ("affine", 2, 2, "safe"),
        )
        self.assertTrue(q.resolve_clip_search(args))
        self.assertFalse(q.resolve_clip_search(q.parse_args(["--self-test"])))
        self.assertFalse(q.resolve_clip_search(q.parse_args(["--self-test", "--no-clip-search", "--expert-format", "affine"])))
        self.assertTrue(q.resolve_clip_search(q.parse_args(["--self-test", "--clip-search"])))
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (
                ["--model", "s", "--output", "o", "--expert-format", "affine", "--imatrix", "i.gguf", "--imatrix-scope", "all"],
                ["--model", "s", "--output", "o", "--imatrix", "i.gguf"],
                ["--model", "s", "--output", "o", "--expert-gate-up-bits", "3"],
            ):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    q.parse_args(argv)
        # The T5 recipe may still raise down_proj to q3.
        self.assertEqual(q.parse_args(["--self-test", "--expert-down-bits", "3"]).expert_down_bits, 3)

    def test_transform_shapes_and_declarations(self):
        generator = torch.Generator().manual_seed(5)
        fused = torch.randn((2, 256, 256), generator=generator).bfloat16()
        down = torch.randn((2, 256, 256), generator=generator).bfloat16()
        for gate_up_bits, down_bits in ((2, 2), (3, 3), (2, 3)):
            with self.subTest(gate_up_bits=gate_up_bits, down_bits=down_bits):
                per_layer = {}
                args = SimpleNamespace(
                    device="cpu", chunk_rows=64, expert_format="affine",
                    expert_gate_up_bits=gate_up_bits, expert_down_bits=down_bits, clip_search=None,
                )
                out = q.transform(GATE_UP, fused, args, per_layer)
                out.update(q.transform(DOWN, down, args, per_layer))
                for projection, bits in (("gate_proj", gate_up_bits), ("up_proj", gate_up_bits), ("down_proj", down_bits)):
                    rows = 256 if projection == "down_proj" else 128
                    weight = out[f"{PREFIX}.{projection}.weight"]
                    self.assertEqual((tuple(weight.shape), weight.dtype), ((2, rows, 256 * bits // 32), torch.uint32))
                    self.assertEqual(tuple(out[f"{PREFIX}.{projection}.scales"].shape), (2, rows, 2))
                    self.assertEqual(per_layer[f"{PREFIX}.{projection}"], {"bits": bits, "group_size": 128, "mode": "affine"})
                config = q.normalize_config({"text_config": {}}, per_layer, expert_format="affine")
                self.assertNotIn("omlx_t5", config)
                self.assertEqual(config["text_config"]["mtp_num_hidden_layers"], 0)
                self.assertEqual(config["quantization"][f"{PREFIX}.down_proj"]["bits"], down_bits)
                self.assertIn("omlx_t5", q.normalize_config({"text_config": {}}, per_layer))

    def test_t5_path_unchanged_by_default(self):
        generator = torch.Generator().manual_seed(6)
        fused = torch.randn((2, 256, 256), generator=generator).bfloat16()
        per_layer = {}
        args = SimpleNamespace(device="cpu", chunk_rows=64, t5_fitter="prefix")
        out = q.transform(GATE_UP, fused, args, per_layer)
        self.assertEqual(out[f"{PREFIX}.gate_proj.weight"].dtype, torch.uint8)
        self.assertFalse(q.resolve_clip_search(args))

    def test_expected_layout(self):
        self.assertEqual(
            q.expected_expert_layout("gate_proj", t5=True, bits=2, group_size=128),
            ((512, 640, 520), "U8", (512, 640, 20)),
        )
        self.assertEqual(
            q.expected_expert_layout("gate_proj", t5=False, bits=2, group_size=128),
            ((512, 640, 160), "U32", (512, 640, 20)),
        )
        self.assertEqual(
            q.expected_expert_layout("down_proj", t5=False, bits=3, group_size=128),
            ((512, 2560, 60), "U32", (512, 2560, 5)),
        )
        with self.assertRaises(ValueError):
            q.expected_expert_layout("down_proj", t5=False, bits=4, group_size=128)
        with self.assertRaises(ValueError):
            q.expected_expert_layout("gate_proj", t5=True, bits=3, group_size=128)

    def test_range_search_never_worse_and_stores_bf16(self):
        generator = torch.Generator().manual_seed(9)
        weight = torch.randn((32, 256), generator=generator) * torch.exp(torch.randn((32, 1), generator=generator))
        weight[3, 7] = 9.0
        importance = torch.rand((32, 256), generator=generator) + 0.1
        ones = torch.ones_like(weight)

        def sse(result, weights):
            codes = q.unpack_affine(result[0], 256, 2).float().reshape(32, 2, 128)
            rebuilt = codes * result[1].float().reshape(32, 2, 1) + result[2].float().reshape(32, 2, 1)
            return (weights * (rebuilt.reshape(32, 256) - weight).square()).reshape(32, 2, 128).sum(-1)

        plain = q.affine_chunk(weight, 2, 128)
        searched = q.affine_chunk(weight, 2, 128, clip_search=True)
        weighted = q.affine_chunk(weight, 2, 128, importance=importance)
        self.assertTrue(torch.all(sse(searched, ones) <= sse(plain, ones) + 1e-6))
        self.assertTrue(torch.all(sse(weighted, importance) <= sse(plain, importance) + 1e-6))
        self.assertLess(sse(searched, ones).sum().item(), sse(plain, ones).sum().item() * 0.9)
        for result in (plain, searched, weighted):
            self.assertEqual(result[1].dtype, torch.bfloat16)
            self.assertEqual(result[2].dtype, torch.bfloat16)
            self.assertEqual(result[0].dtype, torch.uint32)

    def test_imatrix_scope(self):
        self.assertTrue(q._imatrix_in_scope(GATE_UP, "experts"))
        self.assertTrue(q._imatrix_in_scope(DOWN, "safe"))
        self.assertTrue(q._imatrix_in_scope(LAYER + "linear_attn.in_proj_qkv.weight", "safe"))
        self.assertTrue(q._imatrix_in_scope(LAYER + "linear_attn.out_proj.weight", "safe"))
        self.assertFalse(q._imatrix_in_scope(LAYER + "linear_attn.in_proj_qkv.weight", "experts"))
        for name in ("self_attn.o_proj.weight", "attn_hyper_connection.input_mix_weight_down.weight"):
            self.assertTrue(q._imatrix_in_scope(LAYER + name, "safe"))
            self.assertFalse(q._imatrix_in_scope(LAYER + name, "experts"))
        # Unmapped tensors never take imatrix weighting under the safe scope.
        self.assertFalse(q._imatrix_in_scope(LAYER + "self_attn.indexer.index_qk_proj.weight", "safe"))

    def test_vision_policy(self):
        two_d = SimpleNamespace(ndim=2)
        for name in ("vision_tower.blocks.3.attn.qkv.weight", "vision_tower.merger.linear_fc2.weight"):
            self.assertIsNone(q.quant_spec(name, two_d))
            self.assertEqual(q.quant_spec(name, two_d, vision_bits=8), ("affine", 8, 64))
        self.assertIsNone(q.quant_spec("vision_tower.pos_embed.weight", two_d, vision_bits=8))
        self.assertIsNone(q.quant_spec("vision_tower.patch_embed.proj.weight", SimpleNamespace(ndim=5), vision_bits=8))
        self.assertIsNone(q.quant_spec("vision_tower.blocks.3.attn.qkv.bias", SimpleNamespace(ndim=1), vision_bits=8))
        args = SimpleNamespace(device="cpu", chunk_rows=8, vision_bits=8)
        per_layer = {}
        generator = torch.Generator().manual_seed(3)
        out = q.transform(
            "model.visual.blocks.0.attn.proj.weight",
            torch.randn((8, 128), generator=generator).bfloat16(), args, per_layer,
        )
        self.assertEqual(out["vision_tower.blocks.0.attn.proj.weight"].dtype, torch.uint32)
        self.assertEqual(per_layer["vision_tower.blocks.0.attn.proj"], {"bits": 8, "group_size": 64, "mode": "affine"})
        # Widths that are not a multiple of the group stay BF16, as the 4304-wide fc2 does.
        out = q.transform(
            "model.visual.blocks.0.mlp.linear_fc2.weight",
            torch.randn((8, 96), generator=generator).bfloat16(), args, per_layer,
        )
        self.assertEqual(out["vision_tower.blocks.0.mlp.linear_fc2.weight"].dtype, torch.bfloat16)
        self.assertNotIn("vision_tower.blocks.0.mlp.linear_fc2", per_layer)
        self.assertEqual(q.parse_args(["--self-test"]).vision_bits, 0)
        self.assertEqual(q.parse_args(["--self-test", "--vision-bits", "8"]).vision_bits, 8)

    def test_out_proj_imatrix_is_unpermuted_from_tiled_v_heads(self):
        num_k, num_v, head_dim = 2, 6, 4
        per_k = num_v // num_k
        # llama.cpp stores V heads tiled: index (r, k, d); HF groups them: (k, r, d).
        hf_values = np.arange(num_v * head_dim, dtype=np.float32)
        tiled = hf_values.reshape(num_k, per_k, head_dim).transpose(1, 0, 2).reshape(-1)
        recovered = imatrix_module.unpermute_tiled_v_heads(tiled, num_k, num_v, head_dim)
        self.assertTrue(np.array_equal(recovered, hf_values))
        with self.assertRaises(ValueError):
            imatrix_module.unpermute_tiled_v_heads(tiled[:-1], num_k, num_v, head_dim)


if __name__ == "__main__":
    unittest.main()
