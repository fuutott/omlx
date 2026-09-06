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


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("layout", ["broadcast", "broadcast_evaluated", "contiguous_lazy", "strided_lazy"])
def test_expanded_layout_regression(dtype, layout):
    # Rebuild the graph each time: the old failure depended on allocator state.
    # In particular, NEVER evaluate the broadcast/strided-lazy inputs here.
    for seed in (41, 52, 63):
        _check_routed(dtype, 3, True, False, n=64, k=256, layout=layout, seed=seed)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_lazy_transposed_route_indices(dtype):
    _check_routed(dtype, 3, False, False, n=64, k=256, strided_indices=True)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("symbol,rows", [("bonsai_t5_qmv", 1), ("bonsai_t5_qmv_wide", 3), ("bonsai_t5_qmm", 7)])
@pytest.mark.parametrize("layout", ["broadcast", "strided", "transposed"])
def test_dense_t5_lazy_layouts(dtype, symbol, rows, layout):
    if not fast.has_symbol(symbol):
        pytest.skip(f"requires compiled native {symbol}; a skip is not a validation pass")
    rng = np.random.default_rng(91)
    n, k, group = 64, 256, 128
    codes = rng.integers(0, 3, (n, k), dtype=np.uint8)
    packed = mx.array(pack_t5(codes, group))
    scales = mx.array(rng.uniform(.1, .4, (n, k // group)).astype(np.float32)).astype(dtype)
    source_rows = 1 if layout == "broadcast" else rows
    dense = mx.array((rng.standard_normal((source_rows, k)) / np.sqrt(k)).astype(np.float32)).astype(dtype)
    mx.eval(dense, scales, packed)
    input_np = np.array(dense.astype(mx.float32))
    scale_np = np.array(scales.astype(mx.float32))
    weights = ((codes.astype(np.float32).reshape(n, -1, group) - 1) * scale_np[..., None]).reshape(n, k)
    reference = np.broadcast_to(input_np, (rows, k)) @ weights.T
    if layout == "broadcast":
        x = mx.broadcast_to(dense, (rows, k))
    elif layout == "strided":
        x = mx.stack((dense, mx.zeros_like(dense)), axis=-1).reshape(rows, 2 * k)[:, ::2]
    else:
        x = mx.array(input_np.T.copy()).astype(dtype).T
    # Keep x unevaluated through the public API; only complete the result.
    actual = getattr(fast, symbol)(x, packed, scales)
    mx.eval(actual)
    observed = np.array(actual.astype(mx.float32))
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(observed, reference, atol=1e-2, rtol=3e-2)
    relative_rmse = np.linalg.norm(observed - reference) / np.linalg.norm(reference)
    assert relative_rmse < (0.015 if dtype == mx.bfloat16 else 0.004)


def _check_routed(dtype, tokens, expanded, sorted_routes, n, k, *,
                  layout="broadcast", seed=41, strided_indices=False):
    for symbol in ("bonsai_t5_gather_qmv", "bonsai_t5_qmm"):
        if not fast.has_symbol(symbol):
            pytest.skip(f"requires compiled native {symbol}; a skip is not a validation pass")
    rng = np.random.default_rng(seed)
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
        if layout == "broadcast_evaluated":
            mx.eval(x)  # Diagnostic control only, not the production fix.
        elif layout == "contiguous_lazy":
            x = mx.contiguous(x)
        elif layout == "strided_lazy":
            # Same values with a true non-unit final stride once evaluated.
            x = mx.stack((x, mx.zeros_like(x)), axis=-1).reshape(tokens, top_k, 1, 2 * k)[..., ::2]
    if sorted_routes:
        order = np.argsort(indices.reshape(-1), kind="stable")
        x = x.reshape(-1, 1, k)[mx.array(order.astype(np.int32))]
        routed_indices = mx.array(indices.reshape(-1)[order])
        reference = reference.reshape(-1, 1, n)[order]
    else:
        routed_indices = (mx.array(indices.T.copy()).T if strided_indices else mx.array(indices))
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
