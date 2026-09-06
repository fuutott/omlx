"""Portable checks: uv run --no-project python -m unittest discover -s tools/tests.

Kept outside tests/ because its conftest imports Apple-only MLX. These tests
exercise real Torch packing, safetensors files, and the runtime's NumPy reader.
"""

import contextlib
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file

from tools import quantize_qwen4_flash_next_t5 as converter
from omlx.patches.mlx_vlm_qwen4_exp_compat.residency import qwen4_exp_residency_estimate


ROOT = Path(__file__).resolve().parents[2]
HOST = runpy.run_path(str(ROOT / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/ple_mmap.py"))
Reader = HOST["SafeTensorMMap"]
plan_rows = HOST["plan_rows"]
assemble = HOST["assemble_affine_rows"]
SOURCE_KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
WEIGHT_KEY = converter.runtime_name(SOURCE_KEY)
BASE = converter.module_name(WEIGHT_KEY)


class ConversionTests(unittest.TestCase):
    def test_default_and_legacy_cli(self):
        self.assertEqual(converter.parse_args(["--self-test"]).ple_bits, 8)
        self.assertEqual(converter.parse_args(["--self-test", "--ple-bits", "2"]).ple_bits, 2)

    def test_imatrix_requires_explicit_opt_in(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            converter.parse_args(["--model", "source", "--output", "output", "--imatrix", "calibration.gguf"])

    def test_q8_ple_and_legacy_q2_shape_and_accuracy(self):
        generator = torch.Generator().manual_seed(19)
        dense = torch.randn((9, 160), generator=generator).bfloat16()
        errors = {}
        for bits in (2, 8):
            with self.subTest(bits=bits):
                specs = {}
                args = SimpleNamespace(device="cpu", chunk_rows=4, ple_bits=bits)
                result = converter.transform(SOURCE_KEY, dense, args, specs)
                self.assertEqual(tuple(result[WEIGHT_KEY].shape), (9, 160 * bits // 32))
                self.assertEqual(specs[BASE], {"bits": bits, "group_size": 32, "mode": "affine"})
                meta = {key: (tuple(value.shape), "U32" if value.dtype == torch.uint32 else "BF16")
                        for key, value in result.items()}
                self.assertEqual(converter.validate_ple_metadata(WEIGHT_KEY, meta, {"quantization": specs}), bits)
                codes = converter.unpack_affine(result[WEIGHT_KEY], 160, bits).float()
                recovered = (codes.reshape(9, 5, 32) * result[BASE + ".scales"].float()[..., None]
                             + result[BASE + ".biases"].float()[..., None]).reshape(9, 160)
                errors[bits] = (dense.float() - recovered).square().mean().item()
                wrong = {"quantization": {BASE: {**specs[BASE], "bits": 8 if bits == 2 else 2}}}
                with self.assertRaises(ValueError):
                    converter.validate_ple_metadata(WEIGHT_KEY, meta, wrong)
        self.assertLess(errors[8], errors[2] / 100)

    def test_non_ple_policy_unchanged(self):
        dummy = SimpleNamespace(ndim=2)
        for name, bits in (("self_attn.o_proj", 4), ("self_attn.q_proj", 5),
                           ("linear_attn.out_proj", 5), ("mlp.shared_expert.up_proj", 8)):
            self.assertEqual(converter.quant_spec(f"language_model.model.layers.3.{name}.weight", dummy)[1], bits)

    def test_q8_still_triggers_actual_ssd_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            specs = {}
            args = SimpleNamespace(device="cpu", chunk_rows=4, ple_bits=8)
            tensors = converter.transform(SOURCE_KEY, torch.ones((3, 160)).bfloat16(), args, specs)
            filename = "model.safetensors"
            save_file(tensors, str(path / filename))
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: filename for key in tensors}}))
            estimate = qwen4_exp_residency_estimate(path)
            self.assertTrue(estimate.supported)
            self.assertEqual(estimate.ple_bytes, 3 * (160 + 5 * 2 * 2))
            ceiling = (estimate.resident_bytes + estimate.mmap_bytes) // 2
            self.assertTrue(estimate.force_ssd_offload(ceiling))

    def test_resume_identity_rejects_changes_and_legacy_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "config.json").write_text("{}")
            (source / "model.safetensors.index.json").write_text("{}")
            (source / "part.safetensors").write_bytes(b"source fixture")
            args = SimpleNamespace(ple_bits=8, imatrix=None, imatrix_strict=False, device="cpu", chunk_rows=4)
            identity = converter.conversion_identity(source, ["part.safetensors"], args)
            output = Path(directory) / "output"
            fingerprint = converter.prepare_output(output, identity, resume=False)
            self.assertEqual(converter.prepare_output(output, identity, resume=True), fingerprint)
            with self.assertRaises(ValueError):
                converter.prepare_output(output, identity, resume=False)
            for field, value in (("ple_bits", 2), ("imatrix_sha256", "different"),
                                 ("converter_sha256", "different"), ("source_config_sha256", "different")):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    converter.prepare_output(output, {**identity, field: value}, resume=True)
            converter.validate_resume_metadata({"omlx_conversion_fingerprint": fingerprint}, fingerprint)
            for metadata in (None, {}, {"omlx_conversion_fingerprint": "old"}):
                with self.assertRaises(ValueError):
                    converter.validate_resume_metadata(metadata, fingerprint)
            legacy = Path(directory) / "legacy"
            legacy.mkdir()
            (legacy / "part.safetensors").write_bytes(b"old")
            with self.assertRaises(ValueError):
                converter.prepare_output(legacy, identity, resume=True)
            self.assertEqual((legacy / "part.safetensors").read_bytes(), b"old")


class HostGatherTests(unittest.TestCase):
    def test_grouping_preserves_duplicates_boundaries_and_order(self):
        offsets = (0, 4, 7, 10)
        indices = [9, 0, 4, 3, 4, 7, 6]
        plan = plan_rows(indices, offsets)
        rebuilt = [None] * len(indices)
        for shard, positions, local in plan:
            for position, row in zip(positions, local):
                rebuilt[position] = offsets[shard] + row
        self.assertEqual(rebuilt, indices)
        self.assertEqual([shard for shard, _, _ in plan], [0, 1, 2])
        self.assertEqual(plan_rows([], offsets), [])
        for invalid in (-1, 10):
            with self.assertRaises(IndexError):
                plan_rows([invalid], offsets)

    def test_mmap_rows_and_batched_assembly_bit_exact(self):
        for bits in (2, 8):
            with self.subTest(bits=bits), tempfile.TemporaryDirectory() as directory:
                generator = torch.Generator().manual_seed(29)
                tensors, specs = {}, {}
                offsets = (0, 4, 7, 10)
                for shard, (start, end) in enumerate(zip(offsets, offsets[1:])):
                    dense = torch.randn((end - start, 160), generator=generator).bfloat16()
                    quantized = converter.affine_chunk(dense, bits, 32)
                    keys = tuple(f"shard{shard}.{suffix}" for suffix in ("weight", "scales", "biases"))
                    tensors.update(zip(keys, quantized))
                    specs[shard] = (*keys, bits, 32)
                # Companions intentionally live in different files.
                readers = []
                by_key = {}
                for suffix in ("weight", "scales", "biases"):
                    path = Path(directory) / f"{suffix}.safetensors"
                    selected = {key: tensor for key, tensor in tensors.items() if key.endswith(suffix)}
                    save_file(selected, str(path))
                    reader = Reader(path)
                    readers.append(reader)
                    by_key.update({key: reader for key in selected})
                try:
                    indices = [9, 0, 4, 3, 4, 7, 6]
                    batches = assemble(plan_rows(indices, offsets), specs, by_key, len(indices))
                    for slot in range(3):
                        expected = []
                        for index in indices:
                            shard = np.searchsorted(offsets, index, side="right") - 1
                            tensor = tensors[specs[shard][slot]]
                            array = tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
                            expected.append(array[index - offsets[shard]])
                        np.testing.assert_array_equal(batches[slot], np.stack(expected))
                    owned = by_key["shard0.weight"].rows_numpy("shard0.weight", [0, 0, 3])
                    self.assertTrue(owned.flags.owndata)
                finally:
                    for reader in readers:
                        reader.close()
                # Gathered rows must remain valid after unmapping the file.
                np.testing.assert_array_equal(owned[0], owned[1])
                self.assertEqual(batches[0].shape, (7, 160 * bits // 32))


if __name__ == "__main__":
    unittest.main()
