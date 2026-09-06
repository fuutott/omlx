"""CPU-only PLE row reads; never materialize a complete mmap-backed table."""

from __future__ import annotations

import json
import math
import mmap
import struct
from bisect import bisect_right
from pathlib import Path

import numpy as np


class SafeTensorMMap:
    """Own a file mapping and return independent NumPy copies of selected rows."""

    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        self._mapping = None
        try:
            header_size = struct.unpack("<Q", self._file.read(8))[0]
            self._header = json.loads(self._file.read(header_size))
            self._data_start = 8 + header_size
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                self._mapping.madvise(mmap.MADV_RANDOM)
            except (AttributeError, OSError):
                pass
        except Exception:
            self.close()
            raise

    def tensor_shape(self, key: str) -> tuple[int, ...]:
        return tuple(self._header[key]["shape"])

    def tensor_dtype(self, key: str) -> str:
        return str(self._header[key]["dtype"])

    def rows_numpy(self, key: str, rows: list[int]) -> np.ndarray:
        entry = self._header[key]
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        dtype = entry["dtype"]
        np_dtype = {
            "BF16": np.dtype("<u2"),
            "F16": np.dtype("<f2"),
            "F32": np.dtype("<f4"),
            "U32": np.dtype("<u4"),
            "F8_E4M3": np.dtype("u1"),
        }.get(dtype)
        if np_dtype is None:
            raise TypeError(f"SSD-backed Qwen4 PLE does not support {dtype}")
        if len(shape) != 2 or end - start != math.prod(shape) * np_dtype.itemsize:
            raise ValueError(f"Invalid sparse PLE tensor layout for {key}")
        if self._mapping is None:
            raise ValueError("PLE mapping is closed")
        view = np.ndarray(shape, dtype=np_dtype, buffer=self._mapping,
                          offset=self._data_start + start)
        # Advanced indexing already copies. A second np.array(copy=True) here
        # duplicated every row read in the original path.
        return view[np.asarray(rows, dtype=np.intp)]

    def close(self):
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._file is not None:
            self._file.close()
            self._file = None


def plan_rows(indices: list[int], offsets: tuple[int, ...]):
    """Group once, preserving each position (including repeated row requests)."""
    groups = {}
    for position, index in enumerate(indices):
        if index < 0 or index >= offsets[-1]:
            raise IndexError("embedding index is outside the sharded vocabulary")
        shard = bisect_right(offsets, index) - 1
        positions, local = groups.setdefault(shard, ([], []))
        positions.append(position)
        local.append(index - offsets[shard])
    return [(shard, *groups[shard]) for shard in sorted(groups)]


def assemble_affine_rows(plan, specs, readers, count: int):
    """Assemble uniform packed weight/scale/bias rows in caller order on CPU.

    Callers must check uniform bits, group size and tensor dtypes first. Only
    requested rows are copied; memory scales with request size, not vocabulary.
    BF16 parameters remain raw uint16 bits until the caller creates MLX arrays.
    """
    batches = [None, None, None]
    for shard, positions, local in plan:
        for slot, key in enumerate(specs[shard][:3]):
            rows = readers[key].rows_numpy(key, local)
            if batches[slot] is None:
                batches[slot] = np.empty((count, rows.shape[1]), dtype=rows.dtype)
            batches[slot][positions] = rows
    return tuple(batches)
