from types import SimpleNamespace

import numpy as np

from tools.qwen4_flash_next_imatrix import GGUFImatrix, gguf_name_for_hf_tensor


def _matrix(entries):
    matrix = object.__new__(GGUFImatrix)
    matrix.entries = {
        key: (SimpleNamespace(data=sums), SimpleNamespace(data=counts))
        for key, (sums, counts) in entries.items()
    }
    matrix.applied = set()
    matrix.missing = set()
    matrix.mismatched = []
    matrix.imputed_experts = {}
    return matrix


def test_qwen4_expert_name_mapping_requires_fused_half():
    name = "model.language_model.layers.17.mlp.experts.gate_up_proj"
    assert gguf_name_for_hf_tensor(name, "gate") == "blk.17.ffn_gate_exps.weight"
    assert gguf_name_for_hf_tensor(name, "up") == "blk.17.ffn_up_exps.weight"


def test_qwen4_dense_and_shared_expert_name_mapping():
    assert gguf_name_for_hf_tensor(
        "model.language_model.layers.2.linear_attn.in_proj_qkv.weight"
    ) == "blk.2.attn_qkv.weight"
    assert gguf_name_for_hf_tensor(
        "model.language_model.layers.11.mlp.shared_expert.down_proj.weight"
    ) == "blk.11.ffn_down_shexp.weight"
    assert gguf_name_for_hf_tensor(
        "model.language_model.layers.3.ple.ple_embedding.ngram_embedding.shard_0.weight"
    ) is None


def test_dense_importance_divides_sums_by_count():
    matrix = _matrix(
        {"blk.0.attn_q.weight": (np.array([4.0, 10.0]), np.array([2.0]))}
    )
    values = matrix.importance_for_gguf("blk.0.attn_q.weight", (4, 2))
    np.testing.assert_allclose(values, [2.0, 5.0])


def test_unobserved_expert_uses_observed_channel_mean():
    matrix = _matrix(
        {
            "blk.0.ffn_gate_exps.weight": (
                np.array([[2.0, 4.0], [0.0, 0.0], [9.0, 3.0]], dtype=np.float32),
                np.array([[2.0], [0.0], [3.0]], dtype=np.float32),
            )
        }
    )
    values = matrix.importance_for_gguf(
        "blk.0.ffn_gate_exps.weight", (3, 4, 2)
    )
    np.testing.assert_allclose(values[0], [1.0, 2.0])
    np.testing.assert_allclose(values[2], [3.0, 1.0])
    np.testing.assert_allclose(values[1], [2.0, 1.5])
    assert matrix.imputed_experts == {"blk.0.ffn_gate_exps.weight": 1}


def test_shape_mismatch_can_be_strict():
    matrix = _matrix(
        {"blk.0.attn_q.weight": (np.ones(3, dtype=np.float32), np.array([1.0]))}
    )
    assert matrix.importance_for_gguf("blk.0.attn_q.weight", (4, 2)) is None
    try:
        matrix.importance_for_gguf("blk.0.attn_q.weight", (4, 2), strict=True)
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("strict shape mismatch did not raise")
