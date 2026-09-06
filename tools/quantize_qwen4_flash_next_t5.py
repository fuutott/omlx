#!/usr/bin/env python3
"""Create a sub-2-bit-expert MLX checkpoint for Qwen3.8-Flash-Next.

This is intentionally architecture-specific.  It accepts only the experimental
``qwen4_exp`` schema published as Qwen3.8-Flash-Next; it is not a Qwen3.5
converter.

The memory-critical routed experts use two formats:

* gate/up: weighted least-squares ternary, packed as Bonsai t5 (base-3)
* down: ordinary MLX affine q2

PLE n-gram embeddings default to affine q8/group-32 and remain individually sharded,
which lets OMLX mmap them from SSD.  Small/sensitive language projections use
q4-q8, vision and MoE routers stay BF16, and MTP is omitted by default.

The weighted ternary scale solve follows AngelSlim's PTQ idea.  Unlike STQ1_0,
Bonsai t5 does not impose a 3:4 sparsity constraint, because dense ternary has
the same t5 storage cost and lower reconstruction error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


EXPECTED_REPO = "Qwen/Qwen3.8-Flash-Next"
BASE_QUANT = {"bits": 4, "group_size": 64, "mode": "affine"}
T5_GROUP_SIZE = 128
PLE_GROUP_SIZE = 32
PLE_BITS = 8
T5_FITTER = "prefix"
_EXPERT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
_PLE_RE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")
_RUNTIME_EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_RUNTIME_PLE_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shards\.(\d+)\.weight$"
)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required (use the CUDA environment on the Windows host).") from exc
    return torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conversion_identity(source: Path, shards: list[str], args) -> dict[str, Any]:
    """Bind resume to this recipe and source snapshot (stat, not weight hashes)."""
    return {
        "schema_version": 1,
        "converter_sha256": _sha256(Path(__file__)),
        "source_directory": str(source.resolve()),
        "source_config_sha256": _sha256(source / "config.json"),
        "source_index_sha256": _sha256(source / "model.safetensors.index.json"),
        "source_shards_stat": {
            name: {"bytes": (source / name).stat().st_size,
                   "mtime_ns": (source / name).stat().st_mtime_ns}
            for name in shards
        },
        "ple_bits": args.ple_bits,
        "ple_group_size": PLE_GROUP_SIZE,
        "t5_fitter": getattr(args, "t5_fitter", T5_FITTER),
        "imatrix_sha256": _sha256(args.imatrix) if args.imatrix else None,
        "imatrix_importer_sha256": (
            _sha256(Path(__file__).with_name("qwen4_flash_next_imatrix.py"))
            if args.imatrix else None
        ),
        "imatrix_strict": args.imatrix_strict,
        "torch_version": str(_torch().__version__),
        "device": args.device,
        "chunk_rows": args.chunk_rows,
    }


def prepare_output(output: Path, identity: dict[str, Any], *, resume: bool) -> str:
    """Refuse stale/legacy output before touching any existing artifact."""
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
    manifest = output / "omlx_conversion_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest.is_file():
            raise ValueError("Use a fresh output directory; resume requires a matching conversion manifest")
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Resume identity mismatch (source, recipe, code, or imatrix); use a fresh output directory")
    else:
        output.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return fingerprint


def validate_resume_metadata(metadata: dict[str, str] | None, fingerprint: str) -> None:
    if (metadata or {}).get("omlx_conversion_fingerprint") != fingerprint:
        raise ValueError("Output shard has no matching conversion fingerprint; refusing to resume")


def _load_imatrix(args):
    path = getattr(args, "imatrix", None)
    if path is None:
        args.imatrix_data = None
        return None
    try:
        from tools.qwen4_flash_next_imatrix import GGUFImatrix
    except ModuleNotFoundError:
        # Direct script execution puts tools/ rather than the repository root
        # on sys.path.
        from qwen4_flash_next_imatrix import GGUFImatrix
    args.imatrix_data = GGUFImatrix(path)
    summary = args.imatrix_data.summary()
    print(
        f"loaded imatrix {summary['file']}: {summary['entries']} entries, "
        f"{summary['zero_count_expert_slots']} unobserved expert slots",
        flush=True,
    )
    return args.imatrix_data


def _importance_for(args, name: str, tensor, *, projection: str | None = None):
    matrix = getattr(args, "imatrix_data", None)
    if matrix is None:
        return None
    return matrix.importance_for_hf(
        name,
        tuple(tensor.shape),
        projection=projection,
        strict=bool(getattr(args, "imatrix_strict", False)),
    )


def validate_config(config: dict[str, Any]) -> None:
    tc = config.get("text_config") or {}
    expected = {
        "model_type": "qwen4_exp",
        "architecture": "Qwen4ExpForConditionalGeneration",
        "layers": 48,
        "experts": 512,
        "active_experts": 10,
        "hidden": 2560,
        "expert_hidden": 640,
        "ngram": 3,
        "ngram_parts": 128,
    }
    actual = {
        "model_type": config.get("model_type"),
        "architecture": (config.get("architectures") or [None])[0],
        "layers": tc.get("num_hidden_layers"),
        "experts": tc.get("num_experts"),
        "active_experts": tc.get("num_experts_per_tok"),
        "hidden": tc.get("hidden_size"),
        "expert_hidden": tc.get("moe_intermediate_size"),
        "ngram": tc.get("ngram_size"),
        "ngram_parts": tc.get("split_ngram_parts"),
    }
    if actual != expected:
        details = ", ".join(
            f"{key}={actual[key]!r} (expected {value!r})"
            for key, value in expected.items()
            if actual[key] != value
        )
        raise ValueError(f"Refusing non-Qwen4-Flash-Next architecture: {details}")


def runtime_name(name: str) -> str:
    """Map raw Hugging Face keys to the mlx-vlm Qwen4 runtime tree."""
    if name.startswith("model.language_model."):
        name = "language_model.model." + name[len("model.language_model.") :]
    elif name.startswith("model.visual."):
        name = "vision_tower." + name[len("model.visual.") :]
    elif name.startswith("lm_head."):
        name = "language_model.lm_head." + name[len("lm_head.") :]
    name = _PLE_RE.sub(lambda m: f".ple.ple_embedding.ngram_embedding.shards.{m.group(1)}.weight", name)
    return name


def module_name(weight_name: str) -> str:
    return weight_name[:-7] if weight_name.endswith(".weight") else weight_name


def pack_t5(codes, group_size: int = T5_GROUP_SIZE):
    """Pack codes in {0,1,2}, five trits per byte, preserving group boundaries."""
    torch = _torch()
    shape = tuple(codes.shape)
    rows, width = math.prod(shape[:-1]), shape[-1]
    if width % group_size:
        raise ValueError(f"t5 width {width} is not divisible by group size {group_size}")
    groups = width // group_size
    bpg = (group_size + 4) // 5
    q = codes.reshape(rows, groups, group_size).to(torch.int16)
    padding = bpg * 5 - group_size
    if padding:
        q = torch.cat(
            (q, torch.ones((rows, groups, padding), dtype=q.dtype, device=q.device)),
            dim=-1,
        )
    q = q.reshape(rows, groups, bpg, 5)
    powers = torch.tensor((1, 3, 9, 27, 81), dtype=torch.int16, device=q.device)
    packed = (q * powers).sum(dim=-1).to(torch.uint8)
    return packed.reshape(*shape[:-1], groups * bpg)


def unpack_t5(packed, width: int, group_size: int = T5_GROUP_SIZE):
    torch = _torch()
    shape = tuple(packed.shape)
    rows = math.prod(shape[:-1])
    groups = width // group_size
    bpg = (group_size + 4) // 5
    value = packed.reshape(rows, groups, bpg).to(torch.int32)
    trits = []
    for _ in range(5):
        trits.append(value.remainder(3))
        value = torch.div(value, 3, rounding_mode="floor")
    codes = torch.stack(trits, dim=-1).reshape(rows, groups, bpg * 5)[..., :group_size]
    return codes.reshape(*shape[:-1], width).to(torch.uint8)


def _legacy_ternary_fit(grouped, importance, rounds):
    """Original max-initialized alternating fit, preserved for A/B controls."""
    torch = _torch()
    scale = grouped.abs().amax(dim=-1, keepdim=True)
    selection = torch.zeros_like(grouped)
    for _ in range(rounds):
        selection = torch.where(
            grouped.abs() >= scale * 0.5,
            torch.sign(grouped),
            torch.zeros_like(grouped),
        )
        denominator = (importance * selection.square()).sum(dim=-1, keepdim=True)
        solved = (importance * selection * grouped).sum(dim=-1, keepdim=True) / denominator.clamp_min(1e-20)
        scale = torch.where(solved > 0, solved, scale)
    return selection, scale.to(torch.bfloat16)


def _prefix_ternary_fit(grouped, importance, legacy_selection, legacy_scale):
    """Search magnitude prefixes; accept only a better stored-scale result.

    For any positive scale, optimal nonzero codes have |w| >= scale/2,
    irrespective of positive importance. Sort magnitudes and solve the LS
    scale for every prefix using cumulative weighted sums. Rank candidates
    with scales rounded to the actual BF16 storage dtype, not ideal FP32
    scales. This avoids the legacy fit's max-initialization local minimum.

    Prefix ranking uses FP32 for conversion throughput. The final comparison
    uses direct FP64 residual sums (no cancellation-prone expanded SSE), and
    ties retain the legacy bytes. This is a reconstruction-error safeguard,
    not a guarantee of improved end-to-end model quality.
    """
    torch = _torch()
    magnitudes, order = grouped.abs().sort(dim=-1, descending=True, stable=True)
    sorted_importance = importance.gather(-1, order)
    numerator = (sorted_importance * magnitudes).cumsum(dim=-1)
    denominator = sorted_importance.cumsum(dim=-1)
    scales = (numerator / denominator.clamp_min(1e-20)).to(torch.bfloat16).float()
    # Maximizing reduction in SSE avoids subtracting the common total energy.
    gain = 2 * scales * numerator - scales.square() * denominator
    prefix = gain.argmax(dim=-1, keepdim=True)
    scale = scales.gather(-1, prefix)
    mask = torch.arange(T5_GROUP_SIZE, device=grouped.device) <= prefix
    selected = torch.zeros_like(mask).scatter(-1, order, mask)
    selection = grouped.sign() * selected

    reference = grouped.double()
    imp64 = importance.double()
    error = (imp64 * (reference - selection.double() * scale.double()).square()).sum(dim=-1, keepdim=True)
    legacy_error = (imp64 * (reference - legacy_selection.double() * legacy_scale.double()).square()).sum(dim=-1, keepdim=True)
    take = torch.isfinite(error) & (error < legacy_error)
    return (torch.where(take, selection, legacy_selection),
            torch.where(take, scale, legacy_scale.float()).to(torch.bfloat16))


def weighted_ternary_chunk(weight, rounds: int = 8, importance=None, *, fitter: str = T5_FITTER):
    """Weight-only prefix LS fit, or the explicitly selected historical fit."""
    torch = _torch()
    if fitter not in ("legacy", "prefix"):
        raise ValueError(f"Unknown T5 fitter: {fitter}")
    if rounds < 1:
        raise ValueError("T5 fitting requires at least one legacy round")
    original = tuple(weight.shape)
    if not original or original[-1] == 0 or original[-1] % T5_GROUP_SIZE:
        raise ValueError("T5 weight width must be a positive multiple of 128")
    grouped = weight.float().reshape(-1, original[-1] // T5_GROUP_SIZE, T5_GROUP_SIZE)
    if not torch.isfinite(grouped).all():
        raise ValueError("T5 weights must be finite")
    if importance is None:
        # Same objective as the first, weight-only T5; only the solver changes.
        sigma2 = 2.0 * grouped.square().mean(dim=-1, keepdim=True)
        importance = torch.sqrt(sigma2 + grouped.square())
    else:
        if fitter != "legacy":
            raise ValueError("Prefix T5 fitting is weight-only; parked imatrix requires fitter='legacy'")
        importance = importance.float().reshape_as(grouped).clamp_min(1e-8)
    if not torch.isfinite(importance).all():
        raise ValueError("T5 importance must be finite (check weight magnitude)")
    selection, scale = _legacy_ternary_fit(grouped, importance, rounds)
    if not torch.isfinite(scale).all():
        raise ValueError("T5 scales are not representable as finite BF16")
    if fitter == "prefix":
        selection, scale = _prefix_ternary_fit(grouped, importance, selection, scale)
    codes = (selection + 1).to(torch.uint8).reshape(original)
    scale_shape = (*original[:-1], original[-1] // T5_GROUP_SIZE)
    return pack_t5(codes), scale.reshape(scale_shape).to(torch.bfloat16)


def affine_chunk(weight, bits: int, group_size: int, importance=None):
    """Torch implementation of MLX affine quantization and bit-plane packing."""
    torch = _torch()
    original = tuple(weight.shape)
    grouped = weight.float().reshape(-1, original[-1] // group_size, group_size)
    bins = float((1 << bits) - 1)
    high = grouped.amax(dim=-1, keepdim=True)
    low = grouped.amin(dim=-1, keepdim=True)
    negative_edge = low.abs() > high.abs()
    scale = ((high - low) / bins).clamp_min(1e-7)
    scale = torch.where(negative_edge, scale, -scale)
    edge = torch.where(negative_edge, low, high)
    q0 = torch.round(edge / scale)
    bias = torch.where(q0 != 0, edge, torch.zeros_like(edge))
    scale = torch.where(q0 != 0, edge / q0, scale)
    if importance is not None:
        # Port of OMLX oQe's imatrix-weighted clipping search.  The candidates
        # preserve MLX affine semantics; only scale selection changes.
        imp = importance.float().reshape_as(grouped).clamp_min(1e-8)
        best_scale = scale
        best_bias = bias
        best_codes = torch.round((grouped - bias) / scale).clamp(0, bins)
        best_error = (imp * (grouped - (best_codes * scale + bias)).square()).sum(
            dim=-1, keepdim=True
        )
        for candidate_edge, opposite, sign in (
            (high, low, -1.0),
            (low, high, 1.0),
        ):
            raw = ((candidate_edge - opposite).abs() / bins).clamp_min(1e-7) * sign
            q0 = torch.round(candidate_edge / raw)
            scale0 = torch.where(q0 != 0, candidate_edge / q0, raw)
            bias0 = torch.where(q0 != 0, candidate_edge, torch.zeros_like(candidate_edge))
            for factor in (0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25):
                candidate_scale = scale0 * factor
                candidate_codes = torch.round(
                    (grouped - bias0) / candidate_scale
                ).clamp(0, bins)
                error = (
                    imp
                    * (grouped - (candidate_codes * candidate_scale + bias0)).square()
                ).sum(dim=-1, keepdim=True)
                take = error < best_error
                best_error = torch.where(take, error, best_error)
                best_scale = torch.where(take, candidate_scale, best_scale)
                best_bias = torch.where(take, bias0, best_bias)
        scale, bias = best_scale, best_bias
    codes = torch.round((grouped - bias) / scale).clamp(0, bins).to(torch.int64)

    packed = pack_affine_codes(codes, original, bits)
    packed_shape = (*original[:-1], original[-1] * bits // 32)
    param_shape = (*original[:-1], original[-1] // group_size)
    return (
        packed.reshape(packed_shape).to(torch.uint32),
        scale.reshape(param_shape).to(torch.bfloat16),
        bias.reshape(param_shape).to(torch.bfloat16),
    )


def pack_affine_codes(codes, original_shape: tuple[int, ...], bits: int):
    """Pack integer codes in MLX's slot/bit-plane layout."""
    torch = _torch()
    rows = codes.shape[0]
    if bits in (2, 4, 8):
        per_word = 32 // bits
        shifts = torch.arange(0, 32, bits, dtype=torch.int64, device=codes.device)
        packed = (codes.reshape(rows, -1, per_word) << shifts).sum(dim=-1)
    else:
        bit_ids = torch.arange(bits, dtype=torch.int64, device=codes.device)
        planes = ((codes.unsqueeze(-1) >> bit_ids) & 1).reshape(rows, -1, 32)
        shifts = torch.arange(32, dtype=torch.int64, device=codes.device)
        packed = (planes << shifts).sum(dim=-1)
    return packed.reshape(*original_shape[:-1], original_shape[-1] * bits // 32).to(torch.uint32)


def unpack_affine(packed, width: int, bits: int):
    """Unpack the MLX bit layout for converter verification."""
    torch = _torch()
    rows = math.prod(packed.shape[:-1])
    words = packed.reshape(rows, -1).to(torch.int64)
    if bits in (2, 4, 8):
        shifts = torch.arange(0, 32, bits, dtype=torch.int64, device=words.device)
        codes = ((words.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(rows, width)
    else:
        shifts = torch.arange(32, dtype=torch.int64, device=words.device)
        planes = ((words.unsqueeze(-1) >> shifts) & 1).reshape(rows, width, bits)
        bit_values = 1 << torch.arange(bits, dtype=torch.int64, device=words.device)
        codes = (planes * bit_values).sum(dim=-1)
    return codes


def quantize_chunked(
    weight,
    kind: str,
    bits: int,
    group_size: int,
    device: str,
    chunk_rows: int,
    importance=None,
    t5_fitter: str = T5_FITTER,
):
    torch = _torch()
    shape = tuple(weight.shape)
    flat = weight.reshape(-1, shape[-1])
    importance_tensor = None
    rows_per_expert = None
    if importance is not None:
        importance_tensor = torch.as_tensor(importance, dtype=torch.float32)
        if importance_tensor.ndim == 1 and importance_tensor.shape[0] == shape[-1]:
            pass
        elif (
            importance_tensor.ndim == 2
            and len(shape) >= 3
            and importance_tensor.shape == (shape[0], shape[-1])
        ):
            rows_per_expert = math.prod(shape[1:-1])
        else:
            raise ValueError(
                f"importance shape {tuple(importance_tensor.shape)} is incompatible "
                f"with weight shape {shape}"
            )
    outputs: list[list[Any]] = [[], [], []]
    for start in range(0, flat.shape[0], chunk_rows):
        chunk = flat[start : start + chunk_rows].to(device=device, non_blocking=True)
        chunk_importance = None
        if importance_tensor is not None:
            if importance_tensor.ndim == 1:
                chunk_importance = importance_tensor.expand(chunk.shape[0], -1)
            else:
                expert_ids = torch.arange(start, start + chunk.shape[0]) // rows_per_expert
                chunk_importance = importance_tensor[expert_ids]
            chunk_importance = chunk_importance.to(device=device, non_blocking=True)
        if kind == "t5":
            packed, scales = weighted_ternary_chunk(chunk, importance=chunk_importance, fitter=t5_fitter)
            result = (packed, scales, -scales)
        else:
            result = affine_chunk(chunk, bits, group_size, importance=chunk_importance)
        for target, value in zip(outputs, result):
            target.append(value.cpu())
        del chunk, chunk_importance, result
    joined = [torch.cat(parts, dim=0) for parts in outputs]
    packed_width = shape[-1] * bits // 32 if kind == "affine" else (shape[-1] // group_size) * ((group_size + 4) // 5)
    param_width = shape[-1] // group_size
    return (
        joined[0].reshape(*shape[:-1], packed_width),
        joined[1].reshape(*shape[:-1], param_width),
        joined[2].reshape(*shape[:-1], param_width),
    )


def quant_spec(name: str, tensor) -> tuple[str, int, int] | None:
    """Architecture-aware precision policy for non-routed tensors."""
    lower = name.lower()
    if tensor.ndim < 2 or not name.endswith(".weight"):
        return None
    if name.startswith("vision_tower."):
        return None
    if lower.endswith(".mlp.gate.weight") or lower.endswith(".router.weight"):
        return None
    if any(token in lower for token in ("a_log", "dt_bias", "conv1d.weight")):
        return None
    if "shared_expert_gate" in lower:
        return ("affine", 8, 64)
    if "shared_expert" in lower:
        return ("affine", 8, 128)
    if any(token in lower for token in ("lm_head", "embed_tokens")):
        return ("affine", 6, 64)
    attention_projections = (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
    )
    if any(token in lower for token in attention_projections):
        return ("affine", 5, 64)
    return ("affine", BASE_QUANT["bits"], BASE_QUANT["group_size"])


def _quantized_entries(
    base: str,
    weight,
    kind: str,
    bits: int,
    group_size: int,
    args,
    per_layer,
    importance=None,
):
    if weight.shape[-1] % group_size or (weight.shape[-1] * bits) % 32:
        raise ValueError(f"{base}: width {weight.shape[-1]} is incompatible with {kind} {bits}-bit/group-{group_size}")
    packed, scales, biases = quantize_chunked(
        weight,
        kind,
        bits,
        group_size,
        args.device,
        args.chunk_rows,
        importance=importance,
        t5_fitter=getattr(args, "t5_fitter", T5_FITTER),
    )
    if kind == "t5" or bits != BASE_QUANT["bits"] or group_size != BASE_QUANT["group_size"]:
        per_layer[base] = {"bits": bits, "group_size": group_size, "mode": "affine"}
    return {
        f"{base}.weight": packed,
        f"{base}.scales": scales,
        f"{base}.biases": biases,
    }


def transform(name: str, tensor, args, per_layer: dict[str, dict[str, Any]]):
    match = _EXPERT_RE.match(name)
    if match:
        layer, projection = match.groups()
        prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp"
        if projection == "gate_up_proj":
            if tensor.shape[-2] % 2:
                raise ValueError(f"{name}: fused gate/up row count is odd")
            gate, up = tensor.chunk(2, dim=-2)
            result = {}
            result.update(
                _quantized_entries(
                    f"{prefix}.gate_proj",
                    gate,
                    "t5",
                    2,
                    T5_GROUP_SIZE,
                    args,
                    per_layer,
                    importance=_importance_for(args, name, gate, projection="gate"),
                )
            )
            result.update(
                _quantized_entries(
                    f"{prefix}.up_proj",
                    up,
                    "t5",
                    2,
                    T5_GROUP_SIZE,
                    args,
                    per_layer,
                    importance=_importance_for(args, name, up, projection="up"),
                )
            )
            return result
        return _quantized_entries(
            f"{prefix}.down_proj",
            tensor,
            "affine",
            2,
            T5_GROUP_SIZE,
            args,
            per_layer,
            importance=_importance_for(args, name, tensor),
        )

    out_name = runtime_name(name)
    if _PLE_RE.search(name):
        return _quantized_entries(module_name(out_name), tensor, "affine", getattr(args, "ple_bits", PLE_BITS), PLE_GROUP_SIZE, args, per_layer)

    # qwen4_exp conv kernels use [out, kernel, in] in the MLX runtime.
    if "conv1d.weight" in out_name and tensor.shape[-1] != 1:
        tensor = tensor.movedim(2, 1).contiguous()

    spec = quant_spec(out_name, tensor)
    if spec is None:
        return {out_name: tensor}
    kind, bits, group_size = spec
    if tensor.shape[-1] % group_size or (tensor.shape[-1] * bits) % 32:
        # These are module layouts which mlx-vlm also leaves unquantized.
        return {out_name: tensor}
    return _quantized_entries(
        module_name(out_name),
        tensor,
        kind,
        bits,
        group_size,
        args,
        per_layer,
        importance=_importance_for(args, name, tensor),
    )


def normalize_config(
    config: dict[str, Any],
    per_layer: dict[str, dict[str, Any]],
    imatrix_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = json.loads(json.dumps(config))
    tc = output.get("text_config") or {}
    tc["mtp_num_hidden_layers"] = 0
    output["text_config"] = tc
    quant = dict(BASE_QUANT)
    quant.update(dict(sorted(per_layer.items())))
    output["quantization"] = quant
    output["quantization_config"] = quant
    output["omlx_t5"] = {
        "format": "base3_5trits_per_byte",
        "group_size": T5_GROUP_SIZE,
        "scope": "routed gate_proj and up_proj only",
        "calibration": (
            "llama.cpp activation-imatrix weighted least-squares, 8 rounds"
            if imatrix_summary is not None
            else "AngelSlim-inspired weighted least-squares, 8 rounds, no imatrix"
        ),
        "source": EXPECTED_REPO,
    }
    if imatrix_summary is not None:
        output["omlx_t5"]["importance_matrix"] = {
            key: imatrix_summary[key]
            for key in ("file", "bytes", "sha256", "entries")
        }
    return output


def copy_sidecars(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".safetensors" or path.name in {"model.safetensors.index.json", "config.json", "omlx_conversion_manifest.json", "omlx_conversion.json"}:
            continue
        shutil.copy2(path, output / path.name)


def write_artifact_metadata(
    source: Path,
    output: Path,
    source_shards: list[str],
    verification: dict[str, Any],
    imatrix_summary: dict[str, Any] | None = None,
    t5_fitter: str = T5_FITTER,
) -> None:
    revision = source.name if re.fullmatch(r"[0-9a-f]{40}", source.name) else None
    portable_verification = {
        key: value for key, value in verification.items() if key != "checkpoint"
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_repo": EXPECTED_REPO,
        "source_revision": revision,
        "source_shard_bytes": sum((source / name).stat().st_size for name in source_shards),
        "converted_shard_bytes": verification["bytes"],
        "verification": portable_verification,
        "recipe": {
            "routed_gate_up": f"bonsai_t5_group_128_{t5_fitter}_weighted_ls",
            "t5_fitter": t5_fitter,
            "routed_down": "affine_q2_group_128",
            "ple_ngram_embedding": f"affine_q{verification['ple_bits']}_group_32_ssd_mmap",
            "shared_experts": "affine_q8_group_128",
            "shared_expert_gate": "affine_q8_group_64",
            "attention_and_deltanet_projections": "affine_q5_group_64_except_self_attn_o_proj_q4_group_64",
            "token_embedding_and_lm_head": "affine_q6_group_64",
            "default_eligible_matrix": "affine_q4_group_64",
            "mtp": "removed",
            "importance_matrix": imatrix_summary,
        },
        "validation_status": "structural_only_windows; Apple Silicon runtime pending",
    }
    (output / "omlx_conversion.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    revision_text = revision or "local snapshot (revision not encoded in path)"
    checkpoint_gib = verification["bytes"] / 1024**3
    ple_gib = verification["ple_bytes"] / 1024**3
    mmap_gib = verification["mmap_estimate_bytes"] / 1024**3
    if imatrix_summary is None:
        importance_line = "- no importance matrix was used."
        t5_line = f"weight-only weighted least-squares scale ({t5_fitter} fitter)"
    else:
        importance_line = (
            "- activation importance: llama.cpp GGUF imatrix "
            f"`{imatrix_summary['file']}` (`{imatrix_summary['sha256']}`), with "
            f"{imatrix_summary['zero_count_experts_imputed']} unobserved routed "
            "expert slots imputed from observed experts."
        )
        t5_line = "activation-imatrix weighted least-squares scale"
    card = f"""---
base_model: {EXPECTED_REPO}
library_name: mlx
pipeline_tag: image-text-to-text
license: apache-2.0
tags:
  - mlx
  - omlx
  - qwen4_exp
  - quantized
  - experimental
---

# Qwen3.8-Flash-Next MLX T5 experiment

This is an experimental OMLX checkpoint derived from
[{EXPECTED_REPO}](https://huggingface.co/{EXPECTED_REPO}) at `{revision_text}`.
It targets a 48 GB M3 Max by keeping Qwen4 PLE n-gram embeddings SSD-mmaped and
compressing routed gate/up experts below two bits per weight. Including the
q2 down projection and scale/bias metadata, routed experts average about two
bits per weight on disk.

This artifact has passed CUDA-side packing tests and complete structural
checkpoint verification on Windows. It has **not yet been validated for model
quality, Metal numerical correctness, or 48 GB residency on Apple Silicon**.
Do not describe it as lossless or coherent until those tests are recorded.

Measured converted safetensors size is {checkpoint_gib:.2f} GiB, including
{ple_gib:.2f} GiB of PLE tensor data. OMLX's residency formula estimates
{mmap_gib:.2f} GiB with PLE offloaded and forces SSD offload at a 48 GiB memory
ceiling. This is a header-based estimate, not a measured Mac peak.

## Precision policy

- routed gate/up: Bonsai base-3 T5, group 128, {t5_line};
- routed down: affine q2/group 128;
- PLE n-gram embeddings: affine q{verification['ple_bits']}/group 32 and separately mmap-able;
- shared experts: q8; attention/DeltaNet projections: q5 except QSA o_proj (q4);
- token embeddings/LM head: q6; other eligible matrices: q4;
- vision, router/state, convolution, and norm tensors: BF16;
- MTP removed for the first memory target;
{importance_line}

The T5 representation adapts an AngelSlim-inspired weighted least-squares idea
to OMLX's existing base-3 format. It is not the exact AngelSlim STQ1_0 3:4
layout.

## Runtime requirement

This checkpoint requires the matching experimental OMLX branch with routed
rank-3 T5 support and its native Bonsai Metal kernel. A stock OMLX build cannot
run the T5 expert banks. See `omlx_conversion.json` for the conversion record;
the OMLX fork/commit will be added after publication.

## Verification

```bash
uv run python tools/quantize_qwen4_flash_next_t5.py --verify-only /path/to/checkpoint
```

The base model and tokenizer remain subject to the upstream model card and
Apache-2.0 license.
"""
    (output / "README.md").write_text(card, encoding="utf-8")


def verify_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Verify a converted checkpoint without importing torch or MLX."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors is required") from exc

    checkpoint = checkpoint.resolve()
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if (config.get("text_config") or {}).get("mtp_num_hidden_layers") != 0:
        raise ValueError("converted config must disable MTP")
    if (config.get("omlx_t5") or {}).get("format") != "base3_5trits_per_byte":
        raise ValueError("converted config is missing the OMLX T5 declaration")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = index.get("weight_map") or {}
    shard_names = sorted(set(indexed.values()))
    if not shard_names:
        raise ValueError("checkpoint index has no weight shards")
    if len(shard_names) != 131:
        raise ValueError(f"expected 131 indexed weight shards, found {len(shard_names)}")

    observed: dict[str, str] = {}
    tensor_meta: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard_name in shard_names:
        shard = checkpoint / shard_name
        if not shard.is_file():
            raise FileNotFoundError(shard)
        with safe_open(str(shard), framework="np") as handle:
            for key in handle.keys():
                if key in observed:
                    raise ValueError(
                        f"tensor {key!r} occurs in both {observed[key]!r} and {shard_name!r}"
                    )
                observed[key] = shard_name
                tensor_slice = handle.get_slice(key)
                tensor_meta[key] = (
                    tuple(tensor_slice.get_shape()),
                    tensor_slice.get_dtype(),
                )

    if observed != indexed:
        missing = sorted(set(indexed) - set(observed))[:10]
        extra = sorted(set(observed) - set(indexed))[:10]
        misplaced = sorted(
            key for key in observed.keys() & indexed.keys() if observed[key] != indexed[key]
        )[:10]
        raise ValueError(
            "checkpoint index mismatch: "
            f"missing={missing}, extra={extra}, misplaced={misplaced}"
        )
    mtp_keys = sorted(key for key in observed if key.startswith("mtp."))
    if mtp_keys:
        raise ValueError(f"MTP tensors were not removed: {mtp_keys[:10]}")

    expected_layers = set(range(48))
    expert_layers: dict[str, set[int]] = {
        "gate_proj": set(),
        "up_proj": set(),
        "down_proj": set(),
    }
    for key in observed:
        match = _RUNTIME_EXPERT_RE.match(key)
        if match is None:
            continue
        layer = int(match.group(1))
        projection = match.group(2)
        expert_layers[projection].add(layer)
        base = module_name(key)
        weight_shape, weight_dtype = tensor_meta[key]
        if projection in {"gate_proj", "up_proj"}:
            expected_weight = (512, 640, 520)
            expected_params = (512, 640, 20)
            expected_dtype = "U8"
        else:
            expected_weight = (512, 2560, 40)
            expected_params = (512, 2560, 5)
            expected_dtype = "U32"
        if (weight_shape, weight_dtype) != (expected_weight, expected_dtype):
            raise ValueError(
                f"{key}: got shape/dtype {weight_shape}/{weight_dtype}, "
                f"expected {expected_weight}/{expected_dtype}"
            )
        for suffix in ("scales", "biases"):
            companion = f"{base}.{suffix}"
            if tensor_meta.get(companion) != (expected_params, "BF16"):
                raise ValueError(
                    f"{companion}: got {tensor_meta.get(companion)!r}, "
                    f"expected {(expected_params, 'BF16')!r}"
                )
    for projection, layers in expert_layers.items():
        if layers != expected_layers:
            raise ValueError(
                f"{projection}: expected expert layers 0..47, got {sorted(layers)}"
            )

    ple_weights = [key for key in observed if _RUNTIME_PLE_RE.search(key)]
    if len(ple_weights) != 128:
        raise ValueError(f"expected 128 PLE weight shards, found {len(ple_weights)}")
    ple_bits_seen = set()
    for key in ple_weights:
        ple_bits_seen.add(validate_ple_metadata(key, tensor_meta, config))
    if len(ple_bits_seen) != 1:
        raise ValueError("Mixed PLE bit widths are not part of this conversion recipe")

    dtype_bytes = {"U32": 4, "BF16": 2}
    ple_bytes = sum(
        math.prod(shape) * dtype_bytes[dtype]
        for key, (shape, dtype) in tensor_meta.items()
        if ".ngram_embedding." in key
    )

    indexed_bytes = int((index.get("metadata") or {}).get("total_size", -1))
    actual_bytes = sum((checkpoint / shard).stat().st_size for shard in shard_names)
    if indexed_bytes != actual_bytes:
        raise ValueError(
            f"index total_size={indexed_bytes} does not match shard bytes={actual_bytes}"
        )
    resident_estimate = int(actual_bytes * 1.05)
    mmap_estimate = int((actual_bytes - ple_bytes) * 1.05)
    memory_ceiling = 48 * 1024**3
    result = {
        "checkpoint": str(checkpoint),
        "shards": len(shard_names),
        "tensors": len(observed),
        "bytes": actual_bytes,
        "expert_layers": 48,
        "ple_weight_shards": len(ple_weights),
        "ple_bytes": ple_bytes,
        "ple_bits": next(iter(ple_bits_seen)),
        "mtp_tensors": 0,
        "resident_estimate_bytes": resident_estimate,
        "mmap_estimate_bytes": mmap_estimate,
        "force_ssd_offload_at_48_gib": (
            resident_estimate > memory_ceiling and mmap_estimate <= memory_ceiling
        ),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def validate_ple_metadata(key: str, tensor_meta: dict, config: dict) -> int:
    """Check stored PLE width against its declaration, including legacy q2."""
    base = module_name(key)
    spec = config.get("quantization", {}).get(base, {})
    bits = spec.get("bits")
    if bits not in (2, 8) or spec.get("group_size") != PLE_GROUP_SIZE or spec.get("mode") != "affine":
        raise ValueError(f"{base}: expected explicit affine q2 or q8/group-32 declaration")
    weight_shape, weight_dtype = tensor_meta[key]
    if len(weight_shape) != 2 or weight_shape[-1] != 160 * bits // 32 or weight_dtype != "U32":
        raise ValueError(f"{key}: invalid PLE q{bits} weight {weight_shape}/{weight_dtype}")
    expected_params = (weight_shape[0], 5)
    for suffix in ("scales", "biases"):
        companion = f"{base}.{suffix}"
        if tensor_meta.get(companion) != (expected_params, "BF16"):
            raise ValueError(f"{companion}: expected {expected_params}/BF16")
    return bits


def convert(args) -> None:
    torch = _torch()
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise SystemExit("safetensors is required") from exc

    source = args.model.resolve()
    output = args.output.resolve()
    imatrix = _load_imatrix(args)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must be separate, not nested")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    identity = conversion_identity(source, shards, args)
    fingerprint = prepare_output(output, identity, resume=args.resume)
    per_layer: dict[str, dict[str, Any]] = {}
    weight_map: dict[str, str] = {}
    started = time.monotonic()

    for shard_number, shard_name in enumerate(shards, 1):
        source_shard = source / shard_name
        target_shard = output / shard_name
        if not source_shard.exists():
            raise FileNotFoundError(source_shard)
        if target_shard.exists() and args.resume:
            with safe_open(str(target_shard), framework="pt", device="cpu") as existing:
                validate_resume_metadata(existing.metadata(), fingerprint)
                for key in existing.keys():
                    weight_map[key] = shard_name
            print(f"[{shard_number:03d}/{len(shards)}] resume {shard_name}", flush=True)
            continue

        emitted = {}
        with safe_open(str(source_shard), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for tensor_number, name in enumerate(keys, 1):
                if name.startswith("mtp."):
                    continue
                tensor = handle.get_tensor(name)
                converted = transform(name, tensor, args, per_layer)
                overlap = emitted.keys() & converted.keys()
                if overlap:
                    raise ValueError(f"duplicate output tensors: {sorted(overlap)}")
                emitted.update(converted)
                del tensor, converted
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(
                    f"[{shard_number:03d}/{len(shards)} {tensor_number:03d}/{len(keys)}] {name}",
                    flush=True,
                )
        temporary = target_shard.with_suffix(target_shard.suffix + ".tmp")
        save_file(emitted, str(temporary), metadata={"format": "mlx", "omlx_conversion_fingerprint": fingerprint})
        os.replace(temporary, target_shard)
        for key in emitted:
            weight_map[key] = shard_name
        del emitted
        elapsed = time.monotonic() - started
        print(f"wrote {target_shard.name}; elapsed {elapsed / 60:.1f} min", flush=True)

    # Recover every per-layer declaration deterministically.  This makes
    # --resume produce the same config even when earlier shards were skipped.
    mapped_keys = set(weight_map)
    for key in mapped_keys:
        if not key.endswith(".weight") or f"{module_name(key)}.scales" not in mapped_keys:
            continue
        if key.endswith(".weight") and ".switch_mlp." in key:
            base = module_name(key)
            if base.endswith(("gate_proj", "up_proj")):
                per_layer[base] = {"bits": 2, "group_size": T5_GROUP_SIZE, "mode": "affine"}
            elif base.endswith("down_proj"):
                per_layer[base] = {"bits": 2, "group_size": T5_GROUP_SIZE, "mode": "affine"}
        if ".ngram_embedding.shards." in key and key.endswith(".weight"):
            per_layer[module_name(key)] = {"bits": args.ple_bits, "group_size": PLE_GROUP_SIZE, "mode": "affine"}
            continue
        if ".switch_mlp." in key:
            continue
        spec = quant_spec(key, SimpleNamespace(ndim=2))
        if spec is not None:
            _, bits, group_size = spec
            if bits != BASE_QUANT["bits"] or group_size != BASE_QUANT["group_size"]:
                per_layer[module_name(key)] = {
                    "bits": bits,
                    "group_size": group_size,
                    "mode": "affine",
                }

    indexed_shards = sorted(set(weight_map.values()))
    output_index = {
        "metadata": {
            "total_size": sum((output / name).stat().st_size for name in indexed_shards)
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2), encoding="utf-8"
    )
    imatrix_summary = imatrix.summary() if imatrix is not None else None
    (output / "config.json").write_text(
        json.dumps(
            normalize_config(config, per_layer, imatrix_summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    copy_sidecars(source, output)
    verification = verify_checkpoint(output)
    write_artifact_metadata(
        source, output, shards, verification, imatrix_summary=imatrix_summary,
        t5_fitter=args.t5_fitter,
    )
    print(f"complete: {output}", flush=True)


def convert_single_shard(args) -> None:
    """Run the production transform on one shard for hardware validation."""
    torch = _torch()
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = args.single_shard.resolve()
    destination = args.output.resolve()
    imatrix = _load_imatrix(args)
    if destination.exists() and destination.is_dir():
        destination = destination / source.name
    if destination.exists():
        raise ValueError("Single-shard output already exists; use a fresh destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    per_layer: dict[str, dict[str, Any]] = {}
    emitted = {}
    started = time.monotonic()
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name.startswith("mtp."):
                continue
            emitted.update(transform(name, handle.get_tensor(name), args, per_layer))
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(emitted, str(temporary), metadata={"format": "mlx"})
    os.replace(temporary, destination)
    validation = validate_expert_shard(
        source,
        destination,
        args.device,
        args.validation_experts,
        imatrix=imatrix,
    )
    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(destination),
                "source_bytes": source.stat().st_size,
                "output_bytes": destination.stat().st_size,
                "seconds": round(time.monotonic() - started, 3),
                "tensors": {key: list(value.shape) for key, value in emitted.items()},
                "per_layer": per_layer,
                "t5_fitter": args.t5_fitter,
                "imatrix": imatrix.summary() if imatrix is not None else None,
                "validation": validation,
            },
            indent=2,
        ),
        flush=True,
    )


def validate_expert_shard(
    source: Path,
    output: Path,
    device: str,
    sample_experts: int,
    imatrix=None,
):
    """Measure t5 error on real gate/up rows from a fused expert shard."""
    torch = _torch()
    from safetensors import safe_open

    # The intentionally explicit lookup keeps this validator limited to the
    # official fused Qwen4 gate/up shard layout.
    with safe_open(str(source), framework="pt", device="cpu") as raw:
        candidates = [key for key in raw.keys() if key.endswith("mlp.experts.gate_up_proj")]
        if not candidates:
            return None
        raw_key = candidates[0]
        raw_slice = raw.get_slice(raw_key)
        raw_shape = tuple(raw_slice.get_shape())
        fused = raw_slice[:sample_experts].to(device).float()
    half = fused.shape[-2] // 2
    prefix_match = _EXPERT_RE.match(raw_key)
    assert prefix_match is not None
    layer = prefix_match.group(1)
    prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp"
    metrics = {}
    with safe_open(str(output), framework="pt", device="cpu") as quantized:
        for projection, imatrix_projection, reference in (
            ("gate_proj", "gate", fused[..., :half, :]),
            ("up_proj", "up", fused[..., half:, :]),
        ):
            base = f"{prefix}.{projection}"
            packed = quantized.get_slice(f"{base}.weight")[:sample_experts].to(device)
            scales = quantized.get_slice(f"{base}.scales")[:sample_experts].to(device).float()
            codes = unpack_t5(packed, reference.shape[-1]).to(device).float()
            reconstructed = (
                (codes.reshape(*reference.shape[:-1], -1, T5_GROUP_SIZE) - 1.0)
                * scales.reshape(*reference.shape[:-1], -1, 1)
            ).reshape_as(reference)
            error = reconstructed - reference
            q2_weight, q2_scales, q2_biases = affine_chunk(
                reference, bits=2, group_size=T5_GROUP_SIZE
            )
            q2_codes = unpack_affine(q2_weight, reference.shape[-1], 2).float()
            q2_reconstructed = (
                q2_codes.reshape(*reference.shape[:-1], -1, T5_GROUP_SIZE)
                * q2_scales.float().reshape(*reference.shape[:-1], -1, 1)
                + q2_biases.float().reshape(*reference.shape[:-1], -1, 1)
            ).reshape_as(reference)
            q2_error = q2_reconstructed - reference
            metrics[projection] = {
                "sample_experts": sample_experts,
                "t5_rmse": error.square().mean().sqrt().item(),
                "t5_relative_rmse": (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference)).item(),
                "t5_cosine": torch.nn.functional.cosine_similarity(
                    reconstructed.flatten(), reference.flatten(), dim=0
                ).item(),
                "t5_max_abs_error": error.abs().amax().item(),
                "q2_rmse": q2_error.square().mean().sqrt().item(),
                "q2_relative_rmse": (torch.linalg.vector_norm(q2_error) / torch.linalg.vector_norm(reference)).item(),
                "q2_cosine": torch.nn.functional.cosine_similarity(
                    q2_reconstructed.flatten(), reference.flatten(), dim=0
                ).item(),
            }
            if imatrix is not None:
                full_half_shape = (raw_shape[0], raw_shape[-2] // 2, raw_shape[-1])
                importance = imatrix.importance_for_hf(
                    raw_key,
                    full_half_shape,
                    projection=imatrix_projection,
                    strict=True,
                )[:sample_experts]
                importance = torch.as_tensor(
                    importance, dtype=torch.float32, device=device
                )[:, None, :]
                denominator = importance.sum() * reference.shape[-2]
                weighted_error = (importance * error.square()).sum()
                weighted_reference = (importance * reference.square()).sum()
                weighted_reconstruction = (
                    importance * reconstructed.square()
                ).sum()
                weighted_dot = (importance * reconstructed * reference).sum()
                weighted_q2_error = (importance * q2_error.square()).sum()
                weighted_q2_reconstruction = (
                    importance * q2_reconstructed.square()
                ).sum()
                weighted_q2_dot = (
                    importance * q2_reconstructed * reference
                ).sum()
                metrics[projection].update(
                    {
                        "imatrix_weighted_rmse": (
                            weighted_error / denominator
                        ).sqrt().item(),
                        "imatrix_weighted_relative_rmse": (
                            weighted_error / weighted_reference.clamp_min(1e-20)
                        ).sqrt().item(),
                        "imatrix_weighted_cosine": (
                            weighted_dot
                            / (
                                weighted_reconstruction
                                * weighted_reference
                            ).clamp_min(1e-20).sqrt()
                        ).item(),
                        "q2_imatrix_weighted_rmse": (
                            weighted_q2_error / denominator
                        ).sqrt().item(),
                        "q2_imatrix_weighted_relative_rmse": (
                            weighted_q2_error
                            / weighted_reference.clamp_min(1e-20)
                        ).sqrt().item(),
                        "q2_imatrix_weighted_cosine": (
                            weighted_q2_dot
                            / (
                                weighted_q2_reconstruction
                                * weighted_reference
                            ).clamp_min(1e-20).sqrt()
                        ).item(),
                    }
                )
    return metrics


def self_test(device: str, t5_fitter: str = T5_FITTER) -> None:
    torch = _torch()
    validate_config(
        {
            "model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 48,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "hidden_size": 2560,
                "moe_intermediate_size": 640,
                "ngram_size": 3,
                "split_ngram_parts": 128,
            },
        }
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    weight = torch.randn((7, 256), generator=generator, dtype=torch.float32)
    t5, scales = weighted_ternary_chunk(weight.to(device), fitter=t5_fitter)
    codes = unpack_t5(t5, 256).float()
    if not torch.equal(unpack_t5(pack_t5(codes.to(torch.uint8)), 256), codes.to(torch.uint8)):
        raise AssertionError("t5 pack/unpack changed codes")
    recon = (codes.reshape(7, 2, 128) - 1) * scales.float().reshape(7, 2, 1)
    recon = recon.reshape_as(weight)
    if not torch.isfinite(recon).all() or int(codes.min()) < 0 or int(codes.max()) > 2:
        raise AssertionError("invalid t5 round trip")
    synthetic_importance = torch.linspace(0.1, 2.0, 256).expand(7, -1).to(device)
    imatrix_t5, imatrix_scales = weighted_ternary_chunk(
        weight.to(device), importance=synthetic_importance, fitter="legacy"
    )
    imatrix_codes = unpack_t5(imatrix_t5, 256)
    if not torch.isfinite(imatrix_scales).all() or int(imatrix_codes.max()) > 2:
        raise AssertionError("invalid imatrix-weighted t5 round trip")
    for bits, group_size in ((2, 32), (4, 64), (5, 64), (6, 64), (8, 64), (8, 32)):
        packed, scale, bias = affine_chunk(weight.to(device), bits, group_size)
        codes = unpack_affine(packed, 256, bits).float()
        known_codes = torch.randint(
            0, 1 << bits, (7, 256), generator=generator, dtype=torch.int64
        ).to(device)
        known_packed = pack_affine_codes(known_codes, (7, 256), bits)
        if not torch.equal(unpack_affine(known_packed, 256, bits), known_codes):
            raise AssertionError(f"affine q{bits} bit packing changed codes")
        reconstructed = (
            codes.reshape(7, 256 // group_size, group_size)
            * scale.float().reshape(7, 256 // group_size, 1)
            + bias.float().reshape(7, 256 // group_size, 1)
        )
        if not torch.isfinite(reconstructed).all():
            raise AssertionError(f"invalid affine q{bits} round trip")
        weighted_packed, weighted_scale, weighted_bias = affine_chunk(
            weight.to(device), bits, group_size, importance=synthetic_importance
        )
        if not all(
            torch.isfinite(value).all()
            for value in (weighted_scale.float(), weighted_bias.float())
        ) or weighted_packed.shape != packed.shape:
            raise AssertionError(f"invalid imatrix-weighted affine q{bits} result")

    # Miniature architecture-shaped conversion: fused expert split, routed
    # rank-3 t5 tensors, and the 160-wide PLE group-32 invariant.
    test_args = SimpleNamespace(device=device, chunk_rows=8, t5_fitter=t5_fitter)
    layer_config: dict[str, dict[str, Any]] = {}
    tiny_expert = torch.randn((2, 256, 256), generator=generator, dtype=torch.bfloat16)
    expert_out = transform(
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        tiny_expert,
        test_args,
        layer_config,
    )
    gate_key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    if expert_out[gate_key].shape != (2, 128, 52) or expert_out[gate_key].dtype != torch.uint8:
        raise AssertionError("routed t5 expert shape/dtype mismatch")
    tiny_ple = torch.randn((3, 160), generator=generator, dtype=torch.bfloat16)
    ple_out = transform(
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        tiny_ple,
        test_args,
        layer_config,
    )
    ple_key = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight"
    if ple_out[ple_key].shape != (3, 40) or ple_out[ple_key].dtype != torch.uint32:
        raise AssertionError("PLE q8/group-32 shape/dtype mismatch")
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = Path(temp_dir) / "sample.safetensors"
            save_file({**expert_out, **ple_out}, str(sample), metadata={"format": "mlx"})
            with safe_open(str(sample), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != set(expert_out) | set(ple_out):
                    raise AssertionError("safetensors key round trip failed")
    except ImportError as exc:
        raise AssertionError("safetensors unavailable in converter environment") from exc
    print(
        f"self-test passed on {device}; fitter={t5_fitter}; weighted t5 RMSE={torch.mean((weight - recon.cpu()) ** 2).sqrt().item():.6f}"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--model", type=Path, help="Downloaded Qwen/Qwen3.8-Flash-Next directory")
    parser.add_argument("--output", type=Path, help="Destination MLX checkpoint directory")
    parser.add_argument("--single-shard", type=Path, help="Convert one shard for validation")
    parser.add_argument("--ple-bits", type=int, choices=(2, 8), default=PLE_BITS,
                        help="SSD-backed ngram precision (default 8; 2 only for historical A/B controls)")
    parser.add_argument("--t5-fitter", choices=("legacy", "prefix"), default=T5_FITTER,
                        help="Weight-only T5 scale solver (default prefix; legacy reproduces the original fitter)")
    parser.add_argument("--allow-experimental-imatrix", action="store_true",
                        help="Explicitly opt into the parked, unvalidated imatrix experiment (known DeltaNet channel-order issue)")
    parser.add_argument(
        "--imatrix",
        type=Path,
        help="llama.cpp GGUF importance matrix used for activation-aware quantization",
    )
    parser.add_argument(
        "--imatrix-strict",
        action="store_true",
        help="Fail when a mapped Qwen4 weight has no compatible imatrix entry",
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        help="Structurally verify an already converted checkpoint",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch quantization device")
    parser.add_argument("--chunk-rows", type=int, default=4096, help="Rows quantized per CUDA chunk")
    parser.add_argument(
        "--validation-experts",
        type=int,
        default=8,
        help="Experts sampled in single-shard quality validation",
    )
    parser.add_argument("--resume", action="store_true", help="Keep completed output shards")
    parser.add_argument("--self-test", action="store_true", help="Test pack/dequant kernels without a model")
    args = parser.parse_args(argv)
    modes = sum(
        (
            bool(args.self_test),
            args.verify_only is not None,
            args.single_shard is not None,
            args.model is not None,
        )
    )
    if modes != 1:
        parser.error("select exactly one of --self-test, --verify-only, --single-shard, or --model")
    if args.single_shard is not None and args.output is None:
        parser.error("--output is required with --single-shard")
    if args.model is not None and args.output is None:
        parser.error("--output is required unless --self-test is used")
    if args.imatrix_strict and args.imatrix is None:
        parser.error("--imatrix-strict requires --imatrix")
    if args.imatrix is not None and not args.allow_experimental_imatrix:
        parser.error("Imatrix is parked: known DeltaNet channel-order issue; experimental use requires --allow-experimental-imatrix")
    if args.imatrix is not None and args.t5_fitter != "legacy":
        parser.error("Prefix fitting is weight-only; experimental imatrix also requires --t5-fitter legacy")
    if args.chunk_rows <= 0 or args.validation_experts <= 0:
        parser.error("--chunk-rows and --validation-experts must be positive")
    if args.resume and args.model is None:
        parser.error("--resume applies only to --model conversion")
    if args.imatrix is not None and (args.self_test or args.verify_only is not None):
        parser.error("--imatrix applies only to --model and --single-shard conversion")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.self_test:
        self_test(parsed.device, parsed.t5_fitter)
    elif parsed.verify_only is not None:
        verify_checkpoint(parsed.verify_only)
    elif parsed.single_shard is not None:
        convert_single_shard(parsed)
    else:
        convert(parsed)
