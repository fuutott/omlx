#!/usr/bin/env python3
"""Read-only real-weight comparison of legacy/prefix T5 (no checkpoint bake).

Run as ``python -m tools.qwen4_t5_fit_bench --model SNAPSHOT --output NEW.json``
inside the converter's uv environment. Never loads a full expert bank onto GPU.
Output contains measurements/identities only, not weights or credentials.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors import safe_open

from tools import quantize_qwen4_flash_next_t5 as q


def compare(weight, device, chunk_rows):
    reference = weight.float().reshape(-1, 128)
    importance = (2 * reference.square().mean(-1, keepdim=True) + reference.square()).sqrt().double()
    reference = reference.double()
    energy = (importance * reference.square()).sum().item()
    errors, records = {}, {}
    for fitter in ("legacy", "prefix"):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        packed, scales, _ = q.quantize_chunked(weight, "t5", 2, 128, device, chunk_rows, t5_fitter=fitter)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        reconstructed = (q.unpack_t5(packed, weight.shape[-1]).double().reshape(-1, 128) - 1) * scales.double().reshape(-1, 1)
        residual = reference - reconstructed
        errors[fitter] = (importance * residual.square()).sum(-1)
        if not torch.isfinite(errors[fitter]).all():
            raise AssertionError(f"Non-finite {fitter} reconstruction")
        records[fitter] = {
            "seconds_including_chunk_transfers_and_packing": elapsed,
            "weighted_sse": errors[fitter].sum().item(),
            "weighted_relative_rmse": (errors[fitter].sum().item() / energy) ** .5,
            "unweighted_rmse": residual.square().mean().sqrt().item(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
            "packed_bytes": packed.numel() * packed.element_size(),
            "scale_bytes": scales.numel() * scales.element_size(),
        }
    worse = int((errors["prefix"] > errors["legacy"]).sum())
    result = {
        "shape": list(weight.shape), "groups": reference.shape[0],
        "improved_groups": int((errors["prefix"] < errors["legacy"]).sum()),
        "worse_groups": worse, "fits": records,
        "weighted_sse_reduction_fraction": 1 - records["prefix"]["weighted_sse"] / records["legacy"]["weighted_sse"],
    }
    if worse:
        raise AssertionError(f"{worse} groups regressed in stored-scale weighted SSE")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 12, 24, 36, 47])
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 170, 340, 511])
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh JSON filename")
    if (args.chunk_rows <= 0 or any(not 0 <= x < 48 for x in args.layers)
            or any(not 0 <= x < 512 for x in args.experts)):
        parser.error("Require positive chunk size, layers 0..47 and experts 0..511")
    q.validate_config(json.loads((args.model / "config.json").read_text()))
    index_path = args.model / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_snapshot": str(args.model.resolve()),
        "source_index_sha256": q._sha256(index_path),
        "converter_sha256": q._sha256(Path(q.__file__)),
        "benchmark_sha256": q._sha256(Path(__file__)),
        "torch": str(torch.__version__), "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "experts": args.experts, "chunk_rows": args.chunk_rows,
        "notes": ["Reconstruction only, not KLD or generation quality.",
                  "Same weight-only importance, stored BF16 scales, FP64 residual scores.",
                  "Timings are single passes after tiny warmup; not a throughput benchmark.",
                  "Source paths/index hash recorded; source weight payload checksums not reverified here."],
        "results": [],
    }
    warmup = torch.ones((8, 256), device=args.device)
    for fitter in ("legacy", "prefix"):
        q.weighted_ternary_chunk(warmup, fitter=fitter)
    del warmup
    for layer in args.layers:
        key = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        source = args.model / weight_map[key]
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            tensor = handle.get_slice(key)
            if tensor.get_shape() != [512, 1280, 2560]:
                raise ValueError(f"Unexpected Qwen4 expert shape: {tensor.get_shape()}")
            for projection, start in (("gate", 0), ("up", 640)):
                sample = torch.cat([tensor[e:e + 1, start:start + 640, :] for e in args.experts])
                result = {"layer": layer, "projection": projection, "source_shard": source.name,
                          **compare(sample, args.device, args.chunk_rows)}
                report["results"].append(result)
                print(json.dumps(result), flush=True)
                del sample
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved reconstruction-only report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
