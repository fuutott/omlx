"""Apple Silicon equality gates for the opt-in CPU-batched PLE gather."""

import json

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.mark.parametrize("bits", [2, 8, "mixed", None])
def test_batched_ple_matches_reference_exactly(tmp_path, bits):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding

    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
    mx.random.seed(31)
    tensors = {}
    # Uneven shards and separate companion files exercise the real indexing.
    for shard, size in enumerate((4, 3, 3)):
        dense = mx.random.normal((size, 160)).astype(mx.bfloat16)
        base = f"{prefix}.shards.{shard}"
        if bits is None:
            tensors[base + ".weight"] = dense
        else:
            shard_bits = (2 if shard == 0 else 8) if bits == "mixed" else bits
            w, s, b = mx.quantize(dense, group_size=32, bits=shard_bits)
            tensors.update({base + ".weight": w, base + ".scales": s, base + ".biases": b})
    weight_map = {}
    for suffix in ("weight", "scales", "biases"):
        selected = {key: value for key, value in tensors.items() if key.endswith(suffix)}
        if selected:
            filename = f"{suffix}.safetensors"
            mx.save_safetensors(str(tmp_path / filename), selected)
            weight_map.update({key: filename for key in selected})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    embedding = DiskBackedShardedEmbedding(tmp_path, prefix, 10, 160, 3)
    try:
        embedding.weight_scale = mx.array([0.75], dtype=mx.bfloat16)
        for raw in ([[9, 0, 4, 3, 4, 7, 6]], [[4]], [[], []]):
            indices = mx.array(raw, dtype=mx.int64)
            embedding.batched_gather = False
            reference = embedding(indices)
            mx.eval(reference)
            touched = embedding.last_touched_shards
            embedding.batched_gather = True
            actual = embedding(indices)
            mx.eval(actual)
            assert mx.array_equal(reference, actual).item()
            assert embedding.rows_read == indices.size
            assert embedding.last_touched_shards == touched
        for invalid in (-1, 10):
            with pytest.raises(IndexError):
                embedding(mx.array([invalid]))
        if bits in (None, "mixed"):
            assert embedding._uniform_affine is None
        else:
            assert embedding._uniform_affine[:2] == (bits, 32)
    finally:
        embedding.close()
