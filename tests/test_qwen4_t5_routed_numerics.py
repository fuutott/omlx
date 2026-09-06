"""Actual Metal routed T5 tests (no mocked kernels or checkpoint required)."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.bonsai import fast
from omlx.patches.bonsai_t5_load import _t5_gather_qmm
from tools.repack_ternary_t5 import pack_t5


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("tokens,expanded,sorted_routes", [(1, False, False), (3, False, False), (3, True, False), (7, True, True)])
def test_routed_t5_against_fp32_reference(dtype, tokens, expanded, sorted_routes):
    _check_routed(dtype, tokens, expanded, sorted_routes, n=64, k=256)


def test_qwen4_fused_gate_up_decode_shape():
    # Actual fused gate/up dimensions, top-10, with a smaller expert bank.
    _check_routed(mx.bfloat16, 1, False, False, n=1280, k=2560)


def _check_routed(dtype, tokens, expanded, sorted_routes, n, k):
    for symbol in ("bonsai_t5_gather_qmv", "bonsai_t5_qmm"):
        if not fast.has_symbol(symbol):
            pytest.skip(f"requires compiled native {symbol}; a skip is not a validation pass")
    rng = np.random.default_rng(41)
    experts, top_k, group = 12, 10, 128
    codes = rng.integers(0, 3, (experts, n, k), dtype=np.uint8)
    packed = mx.array(pack_t5(codes.reshape(-1, k), group).reshape(experts, n, -1))
    scales = mx.array(rng.uniform(0.1, 0.4, (experts, n, k // group)).astype(np.float32)).astype(dtype)
    inputs = mx.array((rng.standard_normal((tokens, k)) / np.sqrt(k)).astype(np.float32)).astype(dtype)
    # Deliberately repeated expert ids test indexing as well as input reuse.
    indices = rng.integers(0, experts, (tokens, top_k), dtype=np.int32)
    mx.eval(inputs, scales, packed)
    x_np = np.array(inputs.astype(mx.float32))
    scale_np = np.array(scales.astype(mx.float32))
    reference = np.empty((tokens, top_k, 1, n), dtype=np.float32)
    for token in range(tokens):
        for route, expert in enumerate(indices[token]):
            weight = ((codes[expert].astype(np.float32).reshape(n, -1, group) - 1)
                      * scale_np[expert, :, :, None]).reshape(n, k)
            reference[token, route, 0] = weight @ x_np[token]
    x = inputs[:, None, None, :]
    if expanded:
        x = mx.broadcast_to(x, (tokens, top_k, 1, k))
    if sorted_routes:
        order = np.argsort(indices.reshape(-1), kind="stable")
        x = x.reshape(-1, 1, k)[mx.array(order.astype(np.int32))]
        routed_indices = mx.array(indices.reshape(-1)[order])
        reference = reference.reshape(-1, 1, n)[order]
    else:
        routed_indices = mx.array(indices)
    actual = _t5_gather_qmm(x, packed, scales, None, rhs_indices=routed_indices,
                          sorted_indices=sorted_routes, bits=2, group_size=128)
    mx.eval(actual)
    assert actual.shape == reference.shape
    observed = np.array(actual.astype(mx.float32))
    assert np.isfinite(observed).all()
    # FP32 reference uses already-rounded FP16/BF16 inputs and scales.
    np.testing.assert_allclose(observed, reference, atol=1e-2, rtol=3e-2)
    relative_rmse = np.linalg.norm(observed - reference) / np.linalg.norm(reference)
    assert relative_rmse < (0.015 if dtype == mx.bfloat16 else 0.004)
