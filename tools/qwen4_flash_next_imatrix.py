#!/usr/bin/env python3
"""Read llama.cpp GGUF importance matrices for Qwen3.8-Flash-Next.

llama.cpp stores one ``.in_sum2`` tensor and one ``.counts`` tensor per
calibrated weight.  The former contains input-activation squared sums and the
latter contains either one dense count or one count per routed expert.  This
module keeps the 580 MB GGUF memory-mapped and materialises only the entry
currently being quantised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_HF_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_EXPERT_RE = re.compile(r"^mlp\.experts\.(gate_up_proj|down_proj)$")

# Input-channel-equivalent tensor names between the official HF checkpoint and
# llama.cpp's Qwen4-Exp conversion.  Norms, state vectors, convolutions, token
# embeddings, and PLE n-gram embeddings deliberately have no entry here.
_SUFFIX_MAP = {
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "attn_hyper_connection.input_mix_weight_down.weight": "hc_attn_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight": "hc_attn_up.weight",
    "attn_hyper_connection.block_inject_weight.weight": "hc_attn_inject.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight": "hc_ffn_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight": "hc_ffn_up.weight",
    "mlp_hyper_connection.block_inject_weight.weight": "hc_ffn_inject.weight",
}


def gguf_name_for_hf_tensor(hf_name: str, projection: str | None = None) -> str | None:
    """Return the llama.cpp base tensor with the same input channels."""
    match = _HF_LAYER_RE.match(hf_name)
    if match is None:
        return None
    layer, suffix = match.groups()
    expert = _EXPERT_RE.match(suffix)
    if expert is not None:
        fused = expert.group(1)
        if fused == "gate_up_proj":
            if projection not in {"gate", "up"}:
                raise ValueError("fused gate_up_proj requires projection='gate' or 'up'")
            mapped = f"ffn_{projection}_exps.weight"
        else:
            mapped = "ffn_down_exps.weight"
        return f"blk.{layer}.{mapped}"
    mapped = _SUFFIX_MAP.get(suffix)
    return f"blk.{layer}.{mapped}" if mapped is not None else None


def _field_scalar(reader, name: str) -> Any:
    field = reader.fields[name]
    value = field.parts[field.data[0]]
    if getattr(value, "dtype", None) == np.dtype("uint8"):
        return bytes(value).decode("utf-8")
    flat = np.asarray(value).reshape(-1)
    return flat[0].item() if flat.size == 1 else flat.tolist()


class GGUFImatrix:
    """Memory-mapped llama.cpp imatrix with Qwen4 tensor-name translation."""

    def __init__(self, path: Path | str):
        try:
            from gguf import GGUFReader
        except ImportError as exc:
            raise RuntimeError(
                "gguf is required to import a llama.cpp imatrix; install the "
                "dedicated quantization lockfile"
            ) from exc

        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.reader = GGUFReader(str(self.path), "r")
        if _field_scalar(self.reader, "general.type") != "imatrix":
            raise ValueError(f"{self.path} is a GGUF, but not an imatrix GGUF")

        tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        bases = sorted(
            name[: -len(".in_sum2")]
            for name in tensors
            if name.endswith(".in_sum2")
        )
        self.entries = {}
        for base in bases:
            counts_name = f"{base}.counts"
            if counts_name not in tensors:
                raise ValueError(f"imatrix entry {base!r} has no counts tensor")
            self.entries[base] = (tensors[f"{base}.in_sum2"], tensors[counts_name])
        orphan_counts = [
            name for name in tensors
            if name.endswith(".counts") and name[: -len(".counts")] not in self.entries
        ]
        if orphan_counts:
            raise ValueError(f"imatrix has orphan counts tensors: {orphan_counts[:5]}")

        self.applied: set[str] = set()
        self.missing: set[str] = set()
        self.mismatched: list[dict[str, Any]] = []
        self.imputed_experts: dict[str, int] = {}
        self._sha256: str | None = None

    @property
    def sha256(self) -> str:
        if self._sha256 is None:
            digest = hashlib.sha256()
            with self.path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            self._sha256 = digest.hexdigest()
        return self._sha256

    def importance_for_gguf(self, base: str, weight_shape: tuple[int, ...], *, strict: bool = False):
        """Return mean squared input activation per channel.

        Unobserved routed experts are imputed with the channel-wise mean of
        observed experts.  Treating their all-zero sums as genuinely
        unimportant would make an unseen expert effectively unprotected.
        """
        entry = self.entries.get(base)
        if entry is None:
            self.missing.add(base)
            if strict:
                raise KeyError(f"imatrix has no entry for {base}")
            return None

        sums = np.asarray(entry[0].data, dtype=np.float32)
        counts = np.asarray(entry[1].data, dtype=np.float32).reshape(-1)
        width = int(weight_shape[-1])
        if counts.size == 1:
            if sums.ndim != 1 or sums.shape[0] != width:
                return self._shape_error(base, weight_shape, sums.shape, strict)
            values = sums / counts[0] if counts[0] > 0 else np.ones_like(sums)
        else:
            if len(weight_shape) < 3 or sums.shape != (int(weight_shape[0]), width):
                return self._shape_error(base, weight_shape, sums.shape, strict)
            if counts.size != sums.shape[0]:
                return self._shape_error(base, weight_shape, sums.shape, strict, counts.shape)
            active = counts > 0
            if not np.any(active):
                values = np.ones_like(sums)
            else:
                values = np.empty_like(sums)
                values[active] = sums[active] / counts[active, None]
                missing_count = int((~active).sum())
                if missing_count:
                    values[~active] = values[active].mean(axis=0, dtype=np.float64)
                    self.imputed_experts[base] = missing_count

        values = np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=1.0)
        values = np.maximum(values, np.float32(1e-8)).astype(np.float32, copy=False)
        self.applied.add(base)
        return values

    def _shape_error(
        self,
        base: str,
        weight_shape: tuple[int, ...],
        imatrix_shape: tuple[int, ...],
        strict: bool,
        counts_shape: tuple[int, ...] | None = None,
    ):
        item = {
            "tensor": base,
            "weight_shape": list(weight_shape),
            "imatrix_shape": list(imatrix_shape),
        }
        if counts_shape is not None:
            item["counts_shape"] = list(counts_shape)
        self.mismatched.append(item)
        if strict:
            raise ValueError(f"imatrix shape mismatch: {item}")
        return None

    def importance_for_hf(
        self,
        hf_name: str,
        weight_shape: tuple[int, ...],
        *,
        projection: str | None = None,
        strict: bool = False,
    ):
        base = gguf_name_for_hf_tensor(hf_name, projection)
        if base is None:
            return None
        return self.importance_for_gguf(base, weight_shape, strict=strict)

    def summary(self, *, include_hash: bool = True) -> dict[str, Any]:
        expert_entries = 0
        total_experts = 0
        zero_experts = 0
        for _, counts_tensor in self.entries.values():
            counts = np.asarray(counts_tensor.data).reshape(-1)
            if counts.size > 1:
                expert_entries += 1
                total_experts += int(counts.size)
                zero_experts += int((counts <= 0).sum())
        result = {
            "file": self.path.name,
            "bytes": self.path.stat().st_size,
            "entries": len(self.entries),
            "chunk_count": int(_field_scalar(self.reader, "imatrix.chunk_count")),
            "chunk_size": int(_field_scalar(self.reader, "imatrix.chunk_size")),
            "expert_entries": expert_entries,
            "expert_slots": total_experts,
            "zero_count_expert_slots": zero_experts,
            "applied_entries": sorted(self.applied),
            "missing_entries": sorted(self.missing),
            "mismatched_entries": self.mismatched,
            "zero_count_experts_imputed": sum(self.imputed_experts.values()),
        }
        if include_hash:
            result["sha256"] = self.sha256
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imatrix", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    matrix = GGUFImatrix(args.imatrix)
    summary = matrix.summary()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"{summary['file']}: {summary['entries']} entries, "
            f"{summary['zero_count_expert_slots']}/{summary['expert_slots']} "
            f"expert slots unobserved, sha256={summary['sha256']}"
        )


if __name__ == "__main__":
    main()
