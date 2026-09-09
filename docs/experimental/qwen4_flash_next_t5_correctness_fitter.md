# T5 lazy-layout fix and weight-only fitter — 2026-09-06

We are doing quantization optimizations to fit Qwen3.8-Flash-Next (`qwen4_exp`)
on a 48 GB M3 Max, keeping PLE ngrams on SSD. The first **weight-only T5** was
reasonably coherent according to the user, with some factual errors. The later
**T5-imatrix** was unusable. These are distinct checkpoints, and neither Q2 PLE
nor the runtime defect below has been established as the cause of those quality
reports. Imatrix remains parked. Do not restore the rolled-back Mac performance
work or modify the mainline Mac installation.

## Runtime correction: implemented, native validation pending

The Mac's test of `b8b4c87d` reported 188 passes and two failures: unsorted T5
gather with three tokens, ten routes, lazily broadcast activations, in FP16 and
BF16. Materialized controls passed. The evidence is in the separate
[`954d5ddb` validation report](https://github.com/fuutott/omlx/blob/954d5ddb9fda1076c1c7079ff26b7d7221b690d4/docs/experimental/validation/2026-09-06-m3-max/README.md).

The Bonsai `ensure_row_contiguous` helper previously trusted array flags while
building the lazy graph. It now always requests `contiguous(x, false, stream)`.
This lets MLX check layout after the producer executes, before the native
kernel uses flat pointer arithmetic. No `mx.eval`, stream synchronization,
CPU activation readback, or Metal math changes were added.

This follows the actual MLX 0.32.0
[`contiguous` graph operation](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/ops.cpp#L6170)
and its [GPU implementation](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/gpu/primitives.cpp#L45).
That implementation can share an already-contiguous buffer; it may copy a
small view backed by an oversized buffer. Do not assume zero runtime overhead
without measuring it. Other native OMLX extensions already use this pattern.

`tests/test_qwen4_t5_routed_numerics.py` retains the original nine cases and
adds 28: repeated lazy/evaluated broadcast controls, explicit lazy contiguous
inputs, strided activations, lazy transposed indices, and dense T5 QMV/wide/QMM
layout tests. The original FP32-reference tolerances are unchanged. The
expanded lazy regressions must not be pre-evaluated to make them pass.

On the Mac, rebuild **only the separate experimental checkout's** extension,
preserving its pinned dependencies. An existing editable install still needs
a native rebuild; pulling Python/C++ source alone is insufficient:

```bash
# Inside the isolated experimental checkout, with HF_HOME already set.
OMLX_WITH_CUSTOM_KERNEL=1 uv pip install --python .venv/bin/python --no-deps --reinstall -e .
uv run --no-project --python .venv/bin/python python -m pytest -q \
  tests/test_bonsai_t5_load.py tests/test_qwen35_moe_gate_up.py \
  tests/test_bonsai_qmv.py tests/test_qwen4_t5_routed_numerics.py \
  tests/test_mlx_vlm_qwen4_exp_compat.py tests/test_qwen4_ple_batched.py \
  tests/test_qwen4_exp_residency.py
```

Record exact commit, MLX/macOS versions, native kernel availability and full
test output. Kernel skips are not validation passes. No model, synthetic-weight
transfer, server restart or mainline change is required. Native success and
performance are **not yet verified on Windows**.

## Weight-only fitter: implemented and tested on real source weights

The new default `--t5-fitter prefix` keeps the original weight-only objective
`importance = sqrt(2 * mean(w^2) + w^2)` and the identical base-3/group-128
packing, BF16 scales and bias metadata. All other precision assignments are
unchanged; PLE still defaults to Q8 and routed down stays Q2.

For each 128-weight group, it sorts absolute weights, considers every nonempty
prefix and solves its weighted least-squares scale. Candidate ranking uses the
actual BF16-rounded scales. A final direct FP64 weighted-residual comparison
retains the original eight-round result unless the candidate is strictly
better; ties retain the original bytes. FP32 prefix sums are used for speed,
so this is not a claim of exhaustive bit-exact global optimization.

`--t5-fitter legacy` reproduces the old solver and is explicitly tested for
byte equality on CPU and CUDA. It allows a clean Q8-PLE-only control, while
`prefix` is a separate quality candidate. The selected fitter is included in
resume identity, artifact recipe/card and single-shard reports. Never resume
across fitter choices. Parked imatrix requires both its experimental opt-in
and `--t5-fitter legacy`; the new fitter does not change calibration handling.

Real BF16 source comparison at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, RTX PRO 6000 Blackwell, Torch
2.8.0+cu128, sampled experts 0/170/340/511 in layers 0/12/24/36/47, both gate and
up projections, full 640-by-2560 matrices:

- 65,536,000 weights / 512,000 groups checked after packing and scale rounding.
- 117,615 groups improved; **zero groups worsened** in the weighted SSE metric.
- Aggregate weighted SSE decreased **0.68091%**; per-projection reduction ranged
  from 0.25359% to 1.33449%. Unweighted RMSE also decreased in all ten samples.
- Storage size is unchanged. Sample peak Torch CUDA allocation was about
  537 MiB for prefix versus 184 MiB for legacy, during conversion only.
- The adversarial `[1, 0.2 x 127]` unit fixture improves weighted SSE by about
  65%. This illustrates the local-minimum failure, not typical model gains.

These are modest real-weight reconstruction gains, **not KLD or evidence of
better generation**. No full new checkpoint has been converted or evaluated.
Single-pass timing in the JSON includes warmup effects and is not a reliable
full-conversion ETA.

The raw report (`qwen4-t5-prefix-real-weights-20260906.json`, kept outside the
repository) records the exact converter/benchmark hashes and source index identity.
Windows validation: **18 portable tests passed**, including the CUDA
legacy-identity/guard test; both CUDA self-tests passed (prefix RMSE 0.438874,
legacy 0.439448 on the same synthetic fixture). Five changed Python files
passed AST checks and `git diff --check` passed. The focused native pytest
attempt could not start because the converter venv has no pytest; MLX/Metal
execution requires the Mac regardless. No native test pass is claimed here.
Reproduce without writing any checkpoint weights:

```powershell
$env:HF_HOME='D:\hf-cache'  # your SSD-backed HF_HOME
uv --cache-dir ..\.uv-cache run --no-project --offline --python .venv\Scripts\python.exe python -B -m unittest discover -s tools/tests -v
uv --cache-dir ..\.uv-cache run --no-project --offline --python .venv\Scripts\python.exe python -B tools/quantize_qwen4_flash_next_t5.py --self-test --device cuda:0
uv --cache-dir ..\.uv-cache run --no-project --offline --python .venv\Scripts\python.exe python -B -m tools.qwen4_t5_fit_bench --model $env:HF_HOME\hub\models--Qwen--Qwen3.8-Flash-Next\snapshots\de4b8e4d43b917e7706784d8bb445c9af86a3540 --output ..\fit-report.json
```

The model download worker reported successful completion at 18:04:43 UTC on
2026-09-06. This is download completion, not an independent full SHA verification.
