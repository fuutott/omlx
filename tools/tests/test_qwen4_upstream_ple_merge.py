"""Portable checks of the actual merged PLE reader, without importing MLX."""

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import mmap
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from safetensors.numpy import save_file

from test_qwen4_runtime_controls import VENDOR


class MergedPLEReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "rows.safetensors"
        self.arrays = {
            "weight": np.arange(4096 * 83, dtype=np.uint32).reshape(4096, 83),
            "scales": np.arange(4096 * 4, dtype=np.float16).reshape(4096, 4),
            "biases": np.zeros((4096, 4), dtype=np.float16),
        }
        save_file(self.arrays, self.path)
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.addCleanup(self.pool.shutdown)
        self.reads = []
        self.short_reads = False

        def pread(fd, size, offset):
            # Independent descriptor: model POSIX pread without changing the
            # mapped reader's file offset, including on Windows.
            with self.path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(min(size, 997) if self.short_reads else size)
            self.reads.append((offset, len(data)))
            return data

        self.namespace = {
            "np": np, "Path": Path, "json": json, "math": math,
            "struct": struct, "mmap": mmap, "os": SimpleNamespace(pread=pread),
            "time": SimpleNamespace(perf_counter=lambda: 0.0, monotonic=lambda: 1000.0),
            "logger": logging.getLogger(__name__),
            "_PLE_PAGE_SIZE": 16384, "_PLE_IO_POOL": self.pool,
            "_PLE_REARM_FLOOR_SECONDS": 0.0005,
            "_PLE_REARM_PER_ROW_SECONDS": 2e-6,
            "_PLE_REARM_MIN_INTERVAL_SECONDS": 60.0,
        }
        tree = ast.parse((VENDOR / "language.py").read_text(encoding="utf-8"))
        reader = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_SafeTensorMMap")
        future = ast.parse("from __future__ import annotations").body
        module = ast.Module(body=future + [reader], type_ignores=[])
        exec(compile(module, "merged-ple-reader", "exec"), self.namespace)
        self.reader = self.namespace["_SafeTensorMMap"](self.path)
        self.addCleanup(self.reader.close)

    def indices(self):
        row_bytes = self.arrays["weight"].strides[0]
        base = self.reader._data_start + self.reader._header["weight"]["data_offsets"][0]
        crossing = next(i for i in range(4096) if (base + i * row_bytes) % 16384 + row_bytes > 16384)
        return [crossing, crossing, 0, 1, 1000, 2000, 3000, 4000, 4095]

    def test_prefetch_and_warm_rows_preserve_bits_and_file_position(self):
        indices = self.indices()
        position = self.reader._file.tell()
        actual = self.reader.rows_numpy("weight", indices)
        np.testing.assert_array_equal(actual, self.arrays["weight"][indices])
        base = self.reader._data_start + self.reader._header["weight"]["data_offsets"][0]
        width = self.arrays["weight"].strides[0]
        pages = {offset // 16384 for row in indices for offset in range(base + row * width, base + (row + 1) * width)}
        self.assertEqual({offset // 16384 for offset, _ in self.reads}, pages)
        self.assertEqual(self.reader._file.tell(), position)
        self.reads.clear()
        np.testing.assert_array_equal(self.reader.rows_numpy("weight", indices), actual)
        self.assertEqual(self.reads, [])
        actual[:] = 0
        np.testing.assert_array_equal(self.reader.rows_numpy("weight", indices), self.arrays["weight"][indices])

    def test_short_reads_cover_whole_pages(self):
        self.short_reads = True
        self.reader.rows_numpy("weight", self.indices())
        for page in {offset // 16384 for offset, _ in self.reads}:
            spans = sorted((offset, size) for offset, size in self.reads if offset // 16384 == page and size)
            cursor = page * 16384
            for offset, size in spans:
                self.assertEqual(offset, cursor)
                cursor += size
            self.assertEqual(cursor, min((page + 1) * 16384, self.path.stat().st_size))

    def test_small_gathers_and_non_posix_fallback_skip_prefetch(self):
        self.reader._prefetch_missing_pages = Mock(side_effect=AssertionError("unexpected prefetch"))
        for count in (0, 1, 8):
            np.testing.assert_array_equal(self.reader.rows_numpy("weight", list(range(count))), self.arrays["weight"][:count])
        self.namespace["os"] = SimpleNamespace()
        np.testing.assert_array_equal(self.reader.rows_numpy("weight", list(range(16))), self.arrays["weight"][:16])

    def test_batched_assembly_uses_the_same_prefetch_reader(self):
        tree = ast.parse((VENDOR / "ple_mmap.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "assemble_affine_rows")
        scope = {"np": np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "assembly", "exec"), scope)
        indices = self.indices()
        keys = ("weight", "scales", "biases")
        results = scope["assemble_affine_rows"](
            [(0, list(range(len(indices))), indices)], {0: keys},
            dict.fromkeys(keys, self.reader), len(indices),
        )
        self.assertGreater(len(self.reads), 0)
        for key, result in zip(keys, results):
            np.testing.assert_array_equal(result, self.arrays[key][indices])

    def test_reference_rows_delegate_to_numpy_reader(self):
        # Avoid MLX construction but prove the reference arm shares the reader.
        self.reader.to_mlx = lambda values, dtype: (values, dtype)
        values, dtype = self.reader.rows("weight", self.indices())
        self.assertEqual(dtype, "U32")
        self.assertTrue(self.reads)
        np.testing.assert_array_equal(values, self.arrays["weight"][self.indices()])

    def test_slow_gather_rearms_once_per_interval(self):
        self.reader.rows_numpy("weight", self.indices())
        self.reader._rearm_if_slow(0.1, 9)
        self.assertEqual(self.reader._rearm_count, 1)
        self.assertFalse(any(self.reader._seen_pages))
        self.reader.rows_numpy("weight", self.indices())
        self.reader._rearm_if_slow(0.1, 9)
        self.assertEqual(self.reader._rearm_count, 1)
        self.namespace["time"].monotonic = lambda: 1060.0
        self.reader._rearm_if_slow(0.1, 9)
        self.assertEqual(self.reader._rearm_count, 2)

    def test_invalid_rows_and_closed_mapping_fail_before_io(self):
        for index in (-1, 4096):
            with self.assertRaises(IndexError):
                self.reader.rows_numpy("weight", [index] * 9)
        self.assertEqual(self.reads, [])
        self.reader.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            self.reader.rows_numpy("weight", [0])


if __name__ == "__main__":
    unittest.main()
