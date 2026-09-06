#!/usr/bin/env python3
"""Isolated Apple Silicon PLE gather benchmark; no server or full model load.

Compare --synthetic (tiny local Q8/Q2 fixture) or --model /local/checkpoint.
Reports synchronized wall/CPU times and a separate instrumented host breakdown.
This is a row-gather microbenchmark, NOT sustained decode or cold-SSD bandwidth.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch


def synthetic_fixture(path, mx, bits):
    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
    tensors = {}
    mx.random.seed(47)
    for shard in range(128):
        dense = mx.random.normal((32, 160)).astype(mx.bfloat16)
        packed = mx.quantize(dense, group_size=32, bits=bits)
        tensors.update({f"{prefix}.shards.{shard}.{suffix}": value
                        for suffix, value in zip(("weight", "scales", "biases"), packed)})
    filename = "ple.safetensors"
    mx.save_safetensors(str(path / filename), tensors)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: filename for key in tensors}}))


def inspect_table(path, reader_type):
    index_path = path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    pattern = re.compile(r"^(.*\.ngram_embedding)\.(?:shards\.|shard_)(\d+)\.weight$")
    matches = [(match.group(1), int(match.group(2)), key)
               for key in weight_map if (match := pattern.match(key))]
    if not matches or len({prefix for prefix, _, _ in matches}) != 1:
        raise ValueError("Expected exactly one sharded PLE table")
    matches.sort(key=lambda entry: entry[1])
    if [index for _, index, _ in matches] != list(range(len(matches))):
        raise ValueError("PLE shard ids are not contiguous")
    rows = 0
    for _, _, key in matches:
        reader = reader_type(path / weight_map[key])
        try:
            rows += reader.tensor_shape(key)[0]
        finally:
            reader.close()
    return matches[0][0], rows, len(matches), hashlib.sha256(index_path.read_bytes()).hexdigest()


def resource_snapshot():
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"minor_faults": usage.ru_minflt, "major_faults": usage.ru_majflt,
            "max_rss_bytes": usage.ru_maxrss}  # macOS uses bytes.


def benchmark(embedding, inputs, mx, language, iterations, warmup, order):
    import numpy as np

    timings = {}
    for enabled in order:
        name = "batched" if enabled else "reference"
        embedding.batched_gather = enabled
        for i in range(warmup):
            mx.eval(embedding(inputs[i % len(inputs)]))
        mx.synchronize()
        before = resource_snapshot()
        wall, cpu = [], []
        for i in range(iterations):
            started_cpu, started = time.process_time_ns(), time.perf_counter_ns()
            value = embedding(inputs[i % len(inputs)])
            mx.eval(value)
            wall.append((time.perf_counter_ns() - started) / 1e6)
            cpu.append((time.process_time_ns() - started_cpu) / 1e6)
        after = resource_snapshot()
        timings[name] = {
            "mean_ms": float(np.mean(wall)), "p50_ms": float(np.median(wall)),
            "p95_ms": float(np.percentile(wall, 95)), "cpu_mean_ms": float(np.mean(cpu)),
            "minor_faults": after["minor_faults"] - before["minor_faults"],
            "major_faults": after["major_faults"] - before["major_faults"],
            "process_lifetime_max_rss_bytes": after["max_rss_bytes"],
        }

        # Separate diagnostic pass: wrappers perturb timing and are not used
        # for the headline latency. No patched functions escape this scope.
        profile = {"index_eval_ns": 0, "row_numpy_ns": 0, "row_calls": 0,
                   "dequantize_calls": 0, "requested_tensor_bytes": 0}
        original_eval, original_dequant = mx.eval, mx.dequantize
        reader_class = language._SafeTensorMMap
        original_rows = reader_class.rows_numpy

        def timed_eval(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return original_eval(*args, **kwargs)
            finally:
                profile["index_eval_ns"] += time.perf_counter_ns() - start

        def timed_rows(reader, *args, **kwargs):
            start = time.perf_counter_ns()
            result = original_rows(reader, *args, **kwargs)
            profile["row_numpy_ns"] += time.perf_counter_ns() - start
            profile["row_calls"] += 1
            profile["requested_tensor_bytes"] += result.nbytes
            return result

        def counted_dequant(*args, **kwargs):
            profile["dequantize_calls"] += 1
            return original_dequant(*args, **kwargs)

        with patch.object(mx, "eval", timed_eval), patch.object(mx, "dequantize", counted_dequant), patch.object(reader_class, "rows_numpy", timed_rows):
            start = time.perf_counter_ns()
            value = embedding(inputs[0])
            profile["host_call_ns"] = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        mx.eval(value)
        profile["completion_wait_ns"] = time.perf_counter_ns() - start
        profile["touched_shards"] = len(embedding.last_touched_shards)
        timings[name]["instrumented_one_call"] = profile
    return timings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--model", type=Path, help="Existing local checkpoint (never downloaded)")
    parser.add_argument("--bits", type=int, choices=(2, 8), default=8, help="Synthetic fixture bits only")
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 256], help="16 rows = one token's PLE heads")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reverse", action="store_true", help="Measure batched before reference to expose order bias")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(*args.rows, args.iterations) < 1 or args.warmup < 0:
        parser.error("rows/iterations must be positive, warmup non-negative")
    if args.output.exists():
        parser.error("output exists; use a new filename")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Native PLE timings require Apple Silicon; run portable tests on Windows")

    import mlx.core as mx
    import numpy as np
    from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    # A context manager owns only this tool's tiny synthetic fixture.
    with tempfile.TemporaryDirectory(prefix="qwen4-ple-bench-") as temporary:
        model = args.model.resolve() if args.model else Path(temporary)
        if args.synthetic:
            synthetic_fixture(model, mx, args.bits)
        prefix, vocab, shards, index_sha = inspect_table(model, language._SafeTensorMMap)
        embedding = language.DiskBackedShardedEmbedding(model, prefix, vocab, 160, shards)
        if embedding._uniform_affine is None:
            embedding.close()
            raise ValueError("This A/B benchmark requires uniform affine PLE shards")
        records = []
        try:
            for row_count in args.rows:
                rng = np.random.default_rng(53 + row_count)
                arrays = [mx.array(rng.integers(0, vocab, (1, row_count), dtype=np.int64))
                          for _ in range(min(args.iterations, 32))]
                mx.eval(*arrays)
                # First access is explicitly not called cold: OS cache may
                # already contain these pages, especially synthetic fixtures.
                for indices in arrays:
                    embedding.batched_gather = False
                    expected = embedding(indices)
                    mx.eval(expected)
                    embedding.batched_gather = True
                    actual = embedding(indices)
                    mx.eval(actual)
                    if not mx.array_equal(expected, actual).item():
                        raise AssertionError("Batched PLE is not bit-exact; do not enable it")
                results = benchmark(embedding, arrays, mx, language, args.iterations, args.warmup,
                                    [True, False] if args.reverse else [False, True])
                records.append({"rows": row_count, "equality": "exact", "timings": results})
        finally:
            embedding.close()
    root = Path(__file__).resolve().parents[1]
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()
    report = {
        "schema_version": 1, "platform": platform.platform(), "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"), "numpy": np.__version__,
        "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
        "source": str(args.model.resolve()) if args.model else "synthetic",
        "index_sha256": index_sha, "uniform_affine_layout": embedding._uniform_affine,
        "iterations": args.iterations, "warmup": args.warmup, "reversed_order": args.reverse,
        "notes": ["Warm/reused random rows after equality checks; not cold SSD or token throughput.",
                  "Requested bytes exclude OS page amplification and are not physical SSD bytes.",
                  "row_numpy_ns includes CPU copying and page-fault waits; not isolated SSD latency.",
                  "completion_wait_ns includes remaining MLX work, not a GPU-only timer.",
                  "Input ids are pre-evaluated; index_eval_ns excludes model ngram hash computation.",
                  "weight_scale=1 for isolated row-read measurement; full model is not loaded."],
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
