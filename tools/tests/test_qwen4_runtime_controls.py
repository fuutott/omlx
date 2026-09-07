"""Portable control/layout tests. These do not execute MLX or Metal."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


VENDOR = Path(__file__).resolve().parents[2] / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp"


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class RuntimeSwitchTests(unittest.TestCase):
    def controls(self, value):
        keys = ("OMLX_QWEN4_EAGER_DISPATCH", "OMLX_QWEN4_FAST_RMS_NORM", "OMLX_QWEN4_HC_FUSED")
        env = {} if value is None else dict.fromkeys(keys, value)
        with patch.dict(os.environ, env, clear=True):
            language = extract(VENDOR / "language.py", {"_EAGER_DISPATCH", "_FAST_RMS_NORM"}, {"os": os})
            hc = extract(VENDOR / "hc_fused.py", {"_DISABLED"}, {"os": os})
        return language["_EAGER_DISPATCH"], language["_FAST_RMS_NORM"], not hc["_DISABLED"]

    def test_unset_preserves_baseline(self):
        self.assertEqual(self.controls(None), (False, False, False))

    def test_explicit_opt_in(self):
        for value in ("1", "true", "YES", " on "):
            with self.subTest(value=value):
                self.assertEqual(self.controls(value), (True, True, True))

    def test_invalid_and_false_values_stay_off(self):
        for value in ("0", "false", "no", "off", "", "typo"):
            with self.subTest(value=value):
                self.assertEqual(self.controls(value), (False, False, False))


class Array:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype


class QuantizedLinear(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class ProjectionLayoutTests(unittest.TestCase):
    def setUp(self):
        namespace = {
            "mx": SimpleNamespace(array=Array, uint32="u32", bfloat16="bf16"),
            "nn": SimpleNamespace(QuantizedLinear=QuantizedLinear),
        }
        self.check = extract(VENDOR / "hc_fused.py", {"_GROUP_SIZE", "_SUPPORTED_BITS", "_quantized_ok"}, namespace)["_quantized_ok"]

    def projection(self, bits=4, inputs=10240, outputs=320):
        return QuantizedLinear(
            group_size=64, bits=bits, mode="affine",
            weight=Array((outputs, inputs * bits // 32), "u32"),
            scales=Array((outputs, inputs // 64), "bf16"),
            biases=Array((outputs, inputs // 64), "bf16"),
        )

    def test_all_supported_packed_widths(self):
        for bits in (4, 5, 6, 8):
            self.assertTrue(self.check(self.projection(bits), 10240, 320))

    def test_wrong_shapes_fail_closed(self):
        for name in ("weight", "scales", "biases"):
            q = self.projection()
            q[name].shape = (1,)
            self.assertFalse(self.check(q, 10240, 320))

    def test_unsupported_layouts_fail_closed(self):
        for name, value in (("bits", 2), ("group_size", 128), ("mode", "mxfp4"), ("bias", Array((320,), "bf16"))):
            q = self.projection()
            q[name] = value
            self.assertFalse(self.check(q, 10240, 320))
        self.assertFalse(self.check(None, 10240, 320))

    def test_wrong_dtypes_fail_closed(self):
        for name in ("weight", "scales", "biases"):
            q = self.projection()
            q[name].dtype = "fp32"
            self.assertFalse(self.check(q, 10240, 320))


if __name__ == "__main__":
    unittest.main()
