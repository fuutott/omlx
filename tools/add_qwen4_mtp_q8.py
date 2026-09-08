"""Restore the original Qwen4 MTP head without requantizing the quantized target.

Windows/CUDA converter; no MLX import. Output is a NEW checkpoint directory.
Only immutable base safetensors are hardlinked; metadata is always independent.
Never edit shared shards in place. Use --link-mode copy if independent copies
are required. Native MTP acceptance/parity must still be checked on Apple Silicon.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct

try:
    from tools import quantize_qwen4_flash_next_t5 as base_converter
except ModuleNotFoundError:
    import quantize_qwen4_flash_next_t5 as base_converter

SOURCE_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MTP_SHARD = "mtp-q8-g64.safetensors"
Q8 = {"bits": 8, "group_size": 64, "mode": "affine"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def header(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if size > 64 * 1024 * 1024:
            raise ValueError("Unreasonably large safetensors header")
        return {k: v for k, v in json.loads(stream.read(size)).items() if k != "__metadata__"}


def expected_source_shapes():
    """Exact inventory of the pinned Qwen4 source, not a Qwen3.5 heuristic."""
    shapes = {
        "pre_fc_norm_embedding.weight": [2560], "pre_fc_norm_hidden.weight": [10240],
        "fc_embedding.weight": [2560, 2560], "fc_hidden.weight": [2560, 2560],
        "layers.0.mlp.experts.gate_up_proj": [512, 1280, 2560],
        "layers.0.mlp.experts.down_proj": [512, 2560, 640],
        "layers.0.mlp.gate.weight": [512, 2560],
        "layers.0.mlp.shared_expert_gate.weight": [1, 2560],
        "layers.0.mlp.shared_expert.gate_proj.weight": [640, 2560],
        "layers.0.mlp.shared_expert.up_proj.weight": [640, 2560],
        "layers.0.mlp.shared_expert.down_proj.weight": [2560, 640],
        "layers.0.self_attn.q_proj.weight": [12288, 2560],
        "layers.0.self_attn.k_proj.weight": [512, 2560],
        "layers.0.self_attn.v_proj.weight": [512, 2560],
        "layers.0.self_attn.o_proj.weight": [2560, 6144],
        "layers.0.self_attn.q_norm.weight": [256],
        "layers.0.self_attn.k_norm.weight": [256],
        "layers.0.self_attn.indexer.index_qk_proj.weight": [640, 2560],
        "layers.0.self_attn.indexer.q_layernorm.weight": [128],
        "layers.0.self_attn.indexer.k_layernorm.weight": [128],
    }
    for prefix in ("hyper_connection_mixer", "layers.0.attn_hyper_connection", "layers.0.mlp_hyper_connection"):
        shapes[prefix + ".hc_norm.weight"] = [10240]
        shapes[prefix + ".input_mix_weight_down.weight"] = [320, 10240]
        shapes[prefix + ".input_mix_weight_up.weight"] = [10240, 320]
        if prefix != "hyper_connection_mixer":
            shapes[prefix + ".block_inject_weight.weight"] = [4, 10240]
    return {"mtp." + k: v for k, v in shapes.items()}


def split_experts(name, tensor):
    if name == "mtp.layers.0.mlp.experts.gate_up_proj":
        if tensor.ndim != 3 or tensor.shape[-2] % 2:
            raise ValueError("Invalid fused gate/up shape")
        gate, up = tensor.chunk(2, dim=-2)
        return {"mtp.layers.0.mlp.switch_mlp.gate_proj.weight": gate.contiguous(),
                "mtp.layers.0.mlp.switch_mlp.up_proj.weight": up.contiguous()}
    if name == "mtp.layers.0.mlp.experts.down_proj":
        return {"mtp.layers.0.mlp.switch_mlp.down_proj.weight": tensor}
    if name not in expected_source_shapes():
        raise ValueError(f"Unexpected source tensor: {name}")
    return {name: tensor}


def keep_bf16(name, shape):
    return len(shape) == 1 or name in (
        "mtp.layers.0.mlp.gate.weight", "mtp.layers.0.mlp.shared_expert_gate.weight")


def expected_output_layout():
    logical = expected_source_shapes()
    logical.pop("mtp.layers.0.mlp.experts.gate_up_proj")
    logical.pop("mtp.layers.0.mlp.experts.down_proj")
    for projection, shape in (("gate_proj", [512, 640, 2560]),
                              ("up_proj", [512, 640, 2560]),
                              ("down_proj", [512, 2560, 640])):
        logical[f"mtp.layers.0.mlp.switch_mlp.{projection}.weight"] = shape
    layout, specs = {}, {}
    for name, shape in logical.items():
        module = name.removesuffix(".weight")
        if keep_bf16(name, shape):
            layout[name] = {"shape": shape, "dtype": "BF16"}
            specs[module] = False
        else:
            layout[name] = {"shape": [*shape[:-1], shape[-1] // 4], "dtype": "U32"}
            for suffix in ("scales", "biases"):
                layout[module + "." + suffix] = {"shape": [*shape[:-1], shape[-1] // 64], "dtype": "BF16"}
            specs[module] = dict(Q8)
    return layout, specs


def quantize_tensor(name, tensor, *, device, chunk_rows):
    import torch
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Expected original BF16 tensor: {name}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"Nonfinite source: {name}")
    if keep_bf16(name, tensor.shape):
        return {name: tensor.contiguous()}, False
    if tensor.shape[-1] % 64:
        raise ValueError(f"Invalid Q8 group width: {name}")
    values = base_converter.quantize_chunked(tensor, "affine", 8, 64, device, chunk_rows)
    if not all(torch.isfinite(v).all().item() for v in values[1:]):
        raise ValueError(f"Nonfinite quantization metadata: {name}")
    module = name.removesuffix(".weight")
    return {module + "." + suffix: value for suffix, value in zip(("weight", "scales", "biases"), values)}, dict(Q8)


def augment_config(base, source, specs):
    result = copy.deepcopy(base)
    tc = result["text_config"]
    stc = source["text_config"]
    if stc.get("mtp_num_hidden_layers") != 1 or stc.get("mtp_use_dedicated_embeddings") is not False:
        raise ValueError("Expected one shared-embedding Qwen4 MTP layer")
    if stc.get("mtp", {}).get("num_hidden_layers") != 1:
        raise ValueError("Inconsistent source MTP configuration")
    tc["mtp_num_hidden_layers"] = 1
    tc["mtp_use_dedicated_embeddings"] = False
    tc["mtp"] = copy.deepcopy(stc["mtp"])
    if base["quantization"] != base["quantization_config"]:
        raise ValueError("Base quantization declarations disagree")
    quant = copy.deepcopy(base["quantization"])
    if any(k.startswith("mtp.") for k in quant):
        raise ValueError("Base already has MTP quantization overrides")
    quant.update(specs)  # Explicit False prevents default Q4 from quantizing routers.
    result["quantization"] = quant
    result["quantization_config"] = copy.deepcopy(quant)
    return result


def verify(output, base):
    """Header/schema + immutable-base checks; not a native runtime load test."""
    base_config = read_json(base / "config.json")
    config = read_json(output / "config.json")
    index = read_json(output / "model.safetensors.index.json")
    base_index = read_json(base / "model.safetensors.index.json")
    manifest = read_json(output / "omlx_mtp_q8.json")
    if manifest["base_config_sha256"] != sha256(base / "config.json") or manifest["base_index_sha256"] != sha256(base / "model.safetensors.index.json"):
        raise ValueError("Base identity changed")
    if {k: v for k, v in index["weight_map"].items() if not k.startswith("mtp.")} != base_index["weight_map"]:
        raise ValueError("Target weight map changed")
    restored = copy.deepcopy(config)
    for key in ("mtp_num_hidden_layers", "mtp_use_dedicated_embeddings", "mtp"):
        if key in base_config["text_config"]:
            restored["text_config"][key] = base_config["text_config"][key]
        else:
            restored["text_config"].pop(key, None)
    for field in ("quantization", "quantization_config"):
        restored[field] = {k: v for k, v in restored[field].items() if not k.startswith("mtp.")}
    if restored != base_config:
        raise ValueError("Non-MTP configuration changed")
    tc = config["text_config"]
    if tc["mtp_num_hidden_layers"] != 1 or tc["mtp"]["num_hidden_layers"] != 1 or tc["mtp_use_dedicated_embeddings"]:
        raise ValueError("MTP declarations inconsistent")
    for name in set(base_index["weight_map"].values()):
        original, candidate = base / name, output / name
        if original.stat().st_size != candidate.stat().st_size:
            raise ValueError(f"Changed base shard size: {name}")
        if not os.path.samefile(original, candidate) and sha256(original) != sha256(candidate):
            raise ValueError(f"Changed base shard: {name}")
    observed = header(output / MTP_SHARD)
    expected, expected_specs = expected_output_layout()
    if manifest["output_tensors"] != expected or manifest["quantization"] != expected_specs:
        raise ValueError("Manifest disagrees with independent Qwen4 Q8 schema")
    if set(observed) != set(expected):
        raise ValueError("MTP tensor set mismatch")
    for name, meta in observed.items():
        if index["weight_map"].get(name) != MTP_SHARD or meta["shape"] != expected[name]["shape"] or meta["dtype"] != expected[name]["dtype"]:
            raise ValueError(f"MTP layout/index mismatch: {name}")
    if {k for k in index["weight_map"] if k.startswith("mtp.")} != set(observed):
        raise ValueError("Unexpected MTP entries in index")
    if {k: v for k, v in config["quantization"].items() if k.startswith("mtp.")} != manifest["quantization"] or config["quantization"] != config["quantization_config"]:
        raise ValueError("MTP quantization mismatch")
    if sha256(output / MTP_SHARD) != manifest["mtp_sha256"]:
        raise ValueError("MTP shard checksum mismatch")
    payload = sum(math.prod(m["shape"]) * (4 if m["dtype"] == "U32" else 2) for m in observed.values())
    if payload != manifest["mtp_payload_bytes"]:
        raise ValueError("MTP payload byte count mismatch")
    # This target converter uses full shard-file sizes, including headers.
    if index["metadata"]["total_size"] != base_index["metadata"]["total_size"] + (output / MTP_SHARD).stat().st_size:
        raise ValueError("Index byte count mismatch")
    return {"mtp_tensors": len(observed), "mtp_payload_bytes": manifest["mtp_payload_bytes"],
            "base_shards_unchanged": len(set(base_index["weight_map"].values())),
            "validation": "structural_and_checksums; native_MTP_pending"}


def audit_reconstruction(output, source):
    """Fixed evenly spaced rows per matrix; independent Q8 byte unpacking.

    This is weight reconstruction error, NOT activation KLD or MTP acceptance.
    Norms/routers are compared in full to the original BF16 weights.
    """
    import torch
    from safetensors import safe_open
    index = read_json(source / "model.safetensors.index.json")["weight_map"]
    inventory = {k: v for k, v in index.items() if k.startswith("mtp.")}
    rows = []
    with safe_open(str(output / MTP_SHARD), framework="pt", device="cpu") as target:
        for filename in sorted(set(inventory.values())):
            with safe_open(str(source / filename), framework="pt", device="cpu") as original:
                for name in sorted(k for k, v in inventory.items() if v == filename):
                    for mapped, dense in split_experts(name, original.get_tensor(name)).items():
                        if keep_bf16(mapped, dense.shape):
                            if not torch.equal(dense, target.get_tensor(mapped)):
                                raise ValueError(f"BF16 tensor changed: {mapped}")
                            rows.append({"name": mapped, "bf16_byte_exact": True})
                            continue
                        module = mapped.removesuffix(".weight")
                        width = dense.shape[-1]
                        flat = dense.reshape(-1, width)
                        ids = torch.linspace(0, flat.shape[0] - 1, min(32, flat.shape[0])).long().tolist()
                        words = target.get_tensor(mapped).reshape(-1, width // 4)
                        scales = target.get_tensor(module + ".scales").reshape(-1, width // 64)
                        biases = target.get_tensor(module + ".biases").reshape(-1, width // 64)
                        selected = torch.cat([words[i:i + 1].to(torch.int64) for i in ids])
                        codes = torch.stack([(selected >> shift) & 255 for shift in (0, 8, 16, 24)], dim=-1).float()
                        recovered = (codes.reshape(len(ids), -1, 64) * scales[ids].float()[..., None] + biases[ids].float()[..., None]).reshape(len(ids), width)
                        reference = flat[ids].float()
                        rmse = (recovered - reference).square().mean().sqrt().item()
                        relative = rmse / max(reference.square().mean().sqrt().item(), 1e-12)
                        if not math.isfinite(relative) or relative > 0.03:
                            raise ValueError(f"Unexpected Q8 reconstruction error: {mapped}: {relative}")
                        rows.append({"name": mapped, "rows": len(ids), "relative_rmse": relative,
                                     "max_absolute_error": (recovered - reference).abs().max().item()})
    return {"kind": "sampled_weight_reconstruction_not_KLD", "matrices": rows}


def build(args):
    from safetensors import safe_open
    from safetensors.torch import save_file
    source, base, output = args.source.resolve(), args.base.resolve(), args.output.resolve()
    if source.name != SOURCE_REVISION:
        raise ValueError("Use the pinned original HF snapshot directory")
    if output.exists():
        raise ValueError("Output must not exist; never overwrite a tested artifact")
    if base in output.parents or source in output.parents:
        raise ValueError("Output must be separate from source/base")
    source_config, base_config = read_json(source / "config.json"), read_json(base / "config.json")
    base_converter.validate_config(source_config)
    base_result = base_converter.verify_checkpoint(base)
    base_report = read_json(base / "omlx_conversion.json")
    recipe = base_report["recipe"]
    expert_format = recipe.get("expert_format", "t5")
    if base_report["source_revision"] != SOURCE_REVISION or base_result.get("ple_bits") != 8:
        raise ValueError("Requires a target baked from the pinned source with Q8 PLE")
    if expert_format == "t5" and recipe.get("t5_fitter") != "prefix":
        raise ValueError("T5 targets must use the prefix fitter")
    if expert_format not in ("t5", "affine"):
        raise ValueError(f"Unknown target expert format: {expert_format}")
    imatrix_file = (recipe.get("importance_matrix") or {}).get("file")
    source_index = read_json(source / "model.safetensors.index.json")["weight_map"]
    inventory = {k: v for k, v in source_index.items() if k.startswith("mtp.")}
    expected = expected_source_shapes()
    if set(inventory) != set(expected):
        raise ValueError("Source MTP inventory differs from pinned 31-tensor schema")
    for filename in set(inventory.values()):
        for name, meta in header(source / filename).items():
            if name in inventory and (meta["dtype"] != "BF16" or meta["shape"] != expected[name]):
                raise ValueError(f"Source MTP tensor differs: {name}")
    output.mkdir(parents=True)
    result, specs, tensor_shapes = {}, {}, {}
    for filename in sorted(set(inventory.values())):
        with safe_open(str(source / filename), framework="pt", device="cpu") as handle:
            for name in sorted(k for k, v in inventory.items() if v == filename):
                print(f"Quantizing {name}", flush=True)
                for target, tensor in split_experts(name, handle.get_tensor(name)).items():
                    entries, spec = quantize_tensor(target, tensor, device=args.device, chunk_rows=args.chunk_rows)
                    result.update(entries)
                    specs[target.removesuffix(".weight")] = spec
                    tensor_shapes[target] = list(tensor.shape)
    config = augment_config(base_config, source_config, specs)
    save_file(result, str(output / MTP_SHARD), metadata={"format": "mlx", "source_revision": SOURCE_REVISION, "recipe": "affine_q8_g64_bf16_router_norm"})
    payload = sum(t.numel() * t.element_size() for t in result.values())
    base_index = read_json(base / "model.safetensors.index.json")
    index = copy.deepcopy(base_index)
    index["weight_map"].update({key: MTP_SHARD for key in result})
    index["metadata"]["total_size"] += (output / MTP_SHARD).stat().st_size
    own_names = {"config.json", "model.safetensors.index.json", "omlx_conversion.json", "omlx_conversion_manifest.json", "README.md", "MAC_HANDOFF.md"}
    for path in base.iterdir():
        if not path.is_file() or path.name in own_names:
            continue
        if path.suffix == ".safetensors" and args.link_mode == "hardlink":
            os.link(path, output / path.name)
        else:
            shutil.copy2(path, output / path.name)
    write_json(output / "config.json", config)
    write_json(output / "model.safetensors.index.json", index)
    manifest = {"schema_version": 1, "source_repo": base_converter.EXPECTED_REPO,
                "source_revision": SOURCE_REVISION, "source_tensors": expected,
                "base_config_sha256": sha256(base / "config.json"),
                "base_index_sha256": sha256(base / "model.safetensors.index.json"),
                "mtp_sha256": sha256(output / MTP_SHARD), "mtp_payload_bytes": payload,
                "quantization": specs, "logical_shapes": tensor_shapes,
                "output_tensors": {k: {"shape": list(t.shape), "dtype": "U32" if str(t.dtype) == "torch.uint32" else "BF16"} for k, t in result.items()},
                "base_shard_link_mode": args.link_mode, "native_validation": "pending"}
    write_json(output / "omlx_mtp_q8.json", manifest)
    report = copy.deepcopy(base_report)
    report["recipe"]["mtp"] = "affine_q8_group64; routers_and_norms_bf16; shared_target_embeddings"
    report["converted_shard_bytes"] += (output / MTP_SHARD).stat().st_size
    report["verification"] = {"base_before_mtp": base_result, "mtp_tensors": len(result), "mtp_bytes": payload}
    report["validation_status"] = "MTP structural validation only; native MTP pending"
    write_json(output / "omlx_conversion.json", report)
    target_label = "T5 prefix" if expert_format == "t5" else "affine"
    (output / "README.md").write_text(
        f"# Experimental {target_label} / Q8 PLE / Q8 MTP candidate\n\n"
        "Original target weights unchanged. One original MTP layer restored with affine Q8/group64, "
        "BF16 norms and routers. Reuses target token embeddings and LM head. PLE must stay SSD-offloaded.\n\n"
        "Native MTP load, greedy parity, rollback, acceptance, peak memory and speed are NOT yet validated. "
        "Start depth 1 in an isolated runtime under the existing memory guard. "
        f"Target importance matrix: {imatrix_file or 'none'} (the MTP head itself is weight-only Q8).\n\n"
        "Base safetensors may be hardlinked: never edit any shard in place. Metadata files are independent. "
        "See tools/add_qwen4_mtp_q8.py and docs/experimental/qwen4_mtp_q8.md in the experimental omlx branch.\n",
        encoding="utf-8")
    checked = verify(output, base)
    write_json(output / "mtp_verification.json", checked)
    print(json.dumps(checked, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        if not args.source:
            parser.error("--source required for reconstruction audit")
        result = audit_reconstruction(args.output.resolve(), args.source.resolve())
        write_json(args.output / "mtp_reconstruction.json", result)
        print(json.dumps(result, indent=2))
    elif args.verify_only:
        print(json.dumps(verify(args.output.resolve(), args.base.resolve()), indent=2))
    else:
        if not args.source or args.chunk_rows <= 0:
            parser.error("--source and positive --chunk-rows are required for conversion")
        build(args)


if __name__ == "__main__":
    main()
