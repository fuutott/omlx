#!/usr/bin/env python3
"""Re-pack a Bonsai T5 checkpoint's routed gate/up experts as stock MLX affine Q2.

A T5 group stores trits {0, 1, 2} and one scale ``s`` and dequantizes to
``s * (trit - 1)``.  MLX affine Q2 dequantizes ``code * scale + bias``; with
``scale = s`` and ``bias = -s`` the codes 0, 1, 2 map to -s, 0, +s exactly, so
the re-packing is lossless (code 3 is simply never used).  The result loads on
stock oMLX/MLX with no Bonsai extension, at +0.375 bits per weight of storage,
and lets the T5 recipe be scored with the same tools as an affine bake.

Only shards holding T5 banks are rewritten; every other shard is hardlinked
(or copied with ``--link-mode copy``).  Config drops the ``omlx_t5`` loader
marker; the per-module ``bits: 2, group_size: 128`` declarations already match.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

try:
    from tools import quantize_qwen4_flash_next_t5 as base
except ModuleNotFoundError:  # direct script execution from tools/
    import quantize_qwen4_flash_next_t5 as base

T5_SUFFIXES = (".mlp.switch_mlp.gate_proj.weight", ".mlp.switch_mlp.up_proj.weight")


def t5_bank_to_affine(packed, scales, *, device: str, experts_per_chunk: int = 64):
    """Return (weight_u32, scales, biases) for one [E, N, packed] T5 bank."""
    torch = base._torch()
    width = scales.shape[-1] * base.T5_GROUP_SIZE
    outputs = []
    for start in range(0, packed.shape[0], experts_per_chunk):
        chunk = packed[start : start + experts_per_chunk].to(device)
        codes = base.unpack_t5(chunk, width).to(torch.int64)
        if int(codes.max()) > 2:
            raise ValueError("T5 bank contains a trit outside {0, 1, 2}")
        outputs.append(base.pack_affine_codes(codes, tuple(codes.shape), 2).cpu())
        del chunk, codes
    weight = torch.cat(outputs, dim=0)
    biases = (-scales.float()).to(torch.bfloat16)
    return weight, scales, biases


def convert(source: Path, output: Path, *, device: str, link_mode: str) -> dict:
    torch = base._torch()
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Output must not exist; never overwrite a tested artifact")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (config.get("omlx_t5") or {}).get("format") != "base3_5trits_per_byte":
        raise ValueError("Source is not a Bonsai T5 checkpoint")
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    t5_keys = {k for k in weight_map if k.endswith(T5_SUFFIXES)}
    for key in t5_keys:
        module = key[: -len(".weight")]
        for suffix in ("scales", "biases"):
            if weight_map.get(f"{module}.{suffix}") != weight_map[key]:
                raise ValueError(f"{module}: weight and {suffix} live in different shards")
    output.mkdir(parents=True)
    started = time.monotonic()
    rewritten, linked, converted = 0, 0, []
    for number, shard in enumerate(shards, 1):
        keys_here = [k for k, v in weight_map.items() if v == shard]
        t5_here = sorted(k for k in keys_here if k in t5_keys)
        if not t5_here:
            if link_mode == "hardlink":
                os.link(source / shard, output / shard)
            else:
                shutil.copy2(source / shard, output / shard)
            linked += 1
            continue
        tensors = {}
        with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
        for key in t5_here:
            module = key[: -len(".weight")]
            packed = tensors[key]
            if packed.dtype != torch.uint8:
                raise ValueError(f"{key}: expected uint8 T5 storage, got {packed.dtype}")
            weight, scales, biases = t5_bank_to_affine(packed, tensors[f"{module}.scales"], device=device)
            tensors[key] = weight
            tensors[f"{module}.biases"] = biases
            converted.append(key)
            print(f"[{number:03d}/{len(shards)}] {key} {tuple(packed.shape)} -> {tuple(weight.shape)}", flush=True)
        temporary = (output / shard).with_suffix(".safetensors.tmp")
        save_file(tensors, str(temporary), metadata={"format": "mlx", "omlx_t5_to_affine": "lossless_ternary_repack"})
        os.replace(temporary, output / shard)
        rewritten += 1
        del tensors
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    new_index = {
        "metadata": {"total_size": sum((output / s).stat().st_size for s in shards)},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (output / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2), encoding="utf-8")
    new_config = json.loads(json.dumps(config))
    t5_declaration = new_config.pop("omlx_t5")
    (output / "config.json").write_text(json.dumps(new_config, indent=2, ensure_ascii=False), encoding="utf-8")
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name not in {
            "config.json", "model.safetensors.index.json", "omlx_conversion_manifest.json",
        }:
            shutil.copy2(path, output / path.name)
    provenance = {
        "schema_version": 1,
        "source": str(source),
        "source_omlx_t5": t5_declaration,
        "converted_tensors": converted,
        "shards_rewritten": rewritten,
        "shards_linked_or_copied": linked,
        "link_mode": link_mode,
        "mapping": "affine code = trit; scale unchanged; bias = -scale (lossless)",
        "seconds": round(time.monotonic() - started, 1),
    }
    (output / "omlx_t5_to_affine.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    verification = base.verify_checkpoint(output)
    if verification.get("expert_format") != "affine":
        raise ValueError("re-packed checkpoint did not verify as affine")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Bonsai T5 checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="New affine checkpoint directory (must not exist)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    result = convert(args.source, args.output, device=args.device, link_mode=args.link_mode)
    print(json.dumps({k: v for k, v in result.items() if k != "converted_tensors"}, indent=2))


if __name__ == "__main__":
    main()
