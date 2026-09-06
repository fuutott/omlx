"""Portable T5 fitter regressions; real packing, no MLX or model download."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tools import quantize_qwen4_flash_next_t5 as q


def unpack(packed, scales, width):
    codes = q.unpack_t5(packed, width).double().reshape(-1, 128) - 1
    return codes * scales.double().reshape(-1, 1)


def objective(weight, reconstructed):
    groups = weight.float().reshape(-1, 128)
    imp = (2 * groups.square().mean(-1, keepdim=True) + groups.square()).sqrt()
    return (imp.double() * (groups.double() - reconstructed).square()).sum(-1)


def frozen_legacy(weight, rounds=8):
    """Independent copy of the original fit, including operation order."""
    groups = weight.float().reshape(-1, weight.shape[-1] // 128, 128)
    imp = (2 * groups.square().mean(-1, keepdim=True) + groups.square()).sqrt()
    scale = groups.abs().amax(-1, keepdim=True)
    for _ in range(rounds):
        selection = torch.where(groups.abs() >= scale * .5, groups.sign(), torch.zeros_like(groups))
        denominator = (imp * selection.square()).sum(-1, keepdim=True)
        solved = (imp * selection * groups).sum(-1, keepdim=True) / denominator.clamp_min(1e-20)
        scale = torch.where(solved > 0, solved, scale)
    return (q.pack_t5((selection + 1).byte().reshape_as(weight)),
            scale.reshape(*weight.shape[:-1], weight.shape[-1] // 128).bfloat16())


class T5FittingTests(unittest.TestCase):
    def test_legacy_is_byte_identical(self):
        weight = torch.randn((3, 7, 256), generator=torch.Generator().manual_seed(14)).bfloat16()
        expected = frozen_legacy(weight)
        actual = q.weighted_ternary_chunk(weight, fitter="legacy")
        for old, new in zip(expected, actual):
            self.assertTrue(torch.equal(old, new))

    def test_outlier_local_minimum(self):
        weight = torch.full((1, 128), .2)
        weight[0, 0] = 1
        old = q.weighted_ternary_chunk(weight, fitter="legacy")
        new = q.weighted_ternary_chunk(weight)
        old_error = objective(weight, unpack(*old, 128)).item()
        new_error = objective(weight, unpack(*new, 128)).item()
        self.assertLess(new_error, old_error * .36)
        self.assertAlmostEqual(new[1].item(), .2177734375)

    def test_never_worse_per_group_in_stored_bf16(self):
        for seed in range(4):
            generator = torch.Generator().manual_seed(seed)
            weight = torch.randn((3, 17, 256), generator=generator)
            weight *= torch.logspace(-4, 2, 51).reshape(3, 17, 1)
            weight[..., 0] *= 12
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(seed=seed, dtype=dtype):
                    dense = weight.to(dtype)
                    old = q.weighted_ternary_chunk(dense, fitter="legacy")
                    new = q.weighted_ternary_chunk(dense)
                    self.assertEqual(new[0].shape, (3, 17, 52))
                    self.assertEqual(new[1].dtype, torch.bfloat16)
                    before = objective(dense, unpack(*old, 256))
                    after = objective(dense, unpack(*new, 256))
                    self.assertTrue(torch.all(after <= before))

    def test_matches_independent_prefix_oracle(self):
        generator = torch.Generator().manual_seed(33)
        weight = torch.randn((12, 128), generator=generator).bfloat16()
        result = q.weighted_ternary_chunk(weight)
        errors = objective(weight, unpack(*result, 128)).numpy()
        groups = weight.float()
        importance = (2 * groups.square().mean(-1, keepdim=True) + groups.square()).sqrt()
        for row, imp, actual in zip(groups.double().numpy(), importance.double().numpy(), errors):
            order = np.argsort(-np.abs(row), kind="stable")
            best = np.inf
            # Deliberately compute each candidate directly, not via prefix sums.
            for count in range(1, 129):
                indices = order[:count]
                solved = np.sum(imp[indices] * np.abs(row[indices])) / np.sum(imp[indices])
                scale = torch.tensor(solved).bfloat16().double().item()
                recovered = np.zeros_like(row)
                recovered[indices] = np.sign(row[indices]) * scale
                best = min(best, np.sum(imp * (row - recovered) ** 2))
            self.assertLessEqual(actual, best + 1e-7 * max(1, best))

    def test_zero_constant_ties_and_sign(self):
        weight = torch.stack((torch.zeros(128), torch.ones(128), -torch.ones(128),
                              torch.tensor([0., .25, -.25, 1., -1., 0., .25, -.25] * 16)))
        old = q.weighted_ternary_chunk(weight, fitter="legacy")
        new = q.weighted_ternary_chunk(weight)
        recovered = unpack(*new, 128)
        self.assertTrue(torch.isfinite(recovered).all())
        self.assertTrue(torch.equal(recovered[:3], weight[:3].double()))
        self.assertTrue(torch.equal(old[0][:3], new[0][:3]))
        negated = q.weighted_ternary_chunk(-weight)
        self.assertTrue(torch.equal(recovered, -unpack(*negated, 128)))

    def test_final_guard_rejects_bad_candidate_and_keeps_ties(self):
        weight = torch.ones((1, 1, 128))
        imp = torch.ones_like(weight)
        # Force the prefix ranker to select a poor one-nonzero fit.
        with patch.object(torch.Tensor, "argmax", return_value=torch.zeros((1, 1, 1), dtype=torch.long)):
            selection, scale = q._prefix_ternary_fit(weight, imp, weight, torch.ones((1, 1, 1)).bfloat16())
        self.assertTrue(torch.equal(selection, weight))
        self.assertEqual(scale.item(), 1)

    def test_chunking_and_production_fused_split(self):
        from types import SimpleNamespace
        weight = torch.randn((2, 256, 256), generator=torch.Generator().manual_seed(11)).bfloat16()
        for fitter in ("prefix", "legacy"):
            with self.subTest(fitter=fitter):
                direct = q.weighted_ternary_chunk(weight, fitter=fitter)
                chunked = q.quantize_chunked(weight, "t5", 2, 128, "cpu", 37, t5_fitter=fitter)
                self.assertTrue(torch.equal(direct[0], chunked[0]))
                self.assertTrue(torch.equal(direct[1], chunked[1]))
                self.assertTrue(torch.equal(-direct[1], chunked[2]))
                args = SimpleNamespace(device="cpu", chunk_rows=37, t5_fitter=fitter)
                out = q.transform("model.language_model.layers.0.mlp.experts.gate_up_proj", weight, args, {})
                for projection, part in (("gate", weight[:, :128]), ("up", weight[:, 128:])):
                    packed, scales = q.weighted_ternary_chunk(part, fitter=fitter)
                    base = f"language_model.model.layers.0.mlp.switch_mlp.{projection}_proj"
                    self.assertTrue(torch.equal(out[base + ".weight"], packed))
                    self.assertTrue(torch.equal(out[base + ".scales"], scales))

    def test_invalid_inputs_and_parked_imatrix(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                q.weighted_ternary_chunk(torch.full((1, 128), value))
        for width in (0, 127, 129):
            with self.assertRaises(ValueError):
                q.weighted_ternary_chunk(torch.ones((1, width)))
        with self.assertRaises(ValueError):
            q.weighted_ternary_chunk(torch.ones((1, 128)), rounds=0)
        with self.assertRaises(ValueError):
            q.weighted_ternary_chunk(torch.ones((1, 128)), fitter="typo")
        with self.assertRaises(ValueError):
            q.weighted_ternary_chunk(torch.ones((1, 128)), importance=torch.ones((1, 128)))
        command = ["--model", "source", "--output", "dest", "--imatrix", "parked.gguf", "--allow-experimental-imatrix"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            q.parse_args(command)
        self.assertEqual(q.parse_args(command + ["--t5-fitter", "legacy"]).t5_fitter, "legacy")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_legacy_identity_and_prefix_guard(self):
        weight = torch.randn((3, 37, 256), generator=torch.Generator().manual_seed(92)).bfloat16().cuda()
        weight[..., 0] *= 10
        old = q.weighted_ternary_chunk(weight, fitter="legacy")
        for original, reproduced in zip(frozen_legacy(weight), old):
            self.assertTrue(torch.equal(original, reproduced))
        new = q.weighted_ternary_chunk(weight)
        # Independently recompute the objective on CPU after actual packing.
        before = objective(weight.cpu(), unpack(*(x.cpu() for x in old), 256))
        after = objective(weight.cpu(), unpack(*(x.cpu() for x in new), 256))
        self.assertTrue(torch.all(after <= before))

    def test_artifact_metadata_names_selected_fitter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            (source / "source.safetensors").write_bytes(b"fixture")
            verification = {"bytes": 100, "ple_bytes": 60, "mmap_estimate_bytes": 40, "ple_bits": 8}
            for fitter in ("legacy", "prefix"):
                q.write_artifact_metadata(source, output, ["source.safetensors"], verification, t5_fitter=fitter)
                report = json.loads((output / "omlx_conversion.json").read_text())
                self.assertEqual(report["recipe"]["t5_fitter"], fitter)
                self.assertIn(fitter, report["recipe"]["routed_gate_up"])
                self.assertIn(f"({fitter} fitter)", (output / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
