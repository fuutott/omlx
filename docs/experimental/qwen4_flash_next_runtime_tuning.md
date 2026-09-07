# Qwen4 T5 runtime tuning — 2026-09-07

We are optimizing the runtime for the existing weight-only prefix-fit T5/Q8-PLE
checkpoint on a 48 GB M3 Max. Freeze the weights, tokenizer, template, expert
routing and quantization recipe. Imatrix and additional precision changes are
parked. Do not restore the previously rolled-back Mac optimizations.

This is a locally prepared experimental candidate, **not a measured speedup**.
The runtime baseline is commit `0a21f3d8b7f3406c0ccd12c66f4ebbb9a9a1b604`.
Later native validation and the new artifact supersede the older handoff's
statements that conversion and native correctness testing had not happened.

## First candidate and provenance

Adapted from upstream [PR #3469](https://github.com/jundot/omlx/pull/3469),
pinned head `b954693b9a16798db1b8bb02ae5976b8e56d81aa`:

- Per-layer asynchronous dispatch for at most 64 rows.
- Fused RMSNorm, retaining FP32 `1 + weight` and per-stream normalization.
- Three fused hyper-connection Metal kernels, at most 16 BF16 rows, affine
  group-64 projections, 4/5/6/8 bits, including partial-block handling and
  precedence over the existing exact-hybrid/compiled path.

The upstream author measured gains on an M5 Max with different weights. That
is motivation, not a performance prediction for this M3 Max/T5 checkpoint.

Local adaptations:

- All three switches default **off**, with explicit true values only.
- Added a separate RMSNorm switch and retained the exact original body.
- HC eligibility validates packed weight and scale/bias tensor shapes and
  positive geometry before launching pointer-based kernels.
- Injection work is limited to the even SIMD groups, but all groups remain
  present at its threadgroup barrier; no early odd-group return.
- Added malformed-layout, mixed-injection-bit, lazy-strided-batch and original
  RMSNorm-path tests, plus portable control/layout tests.
- Did not import the upstream global pytest TF32 setting. On M3 this is not
  applicable. Any later M5 reference run must explicitly record TF32 policy.

The HC exception fallback catches graph-construction errors only. MLX is lazy:
an asynchronous evaluation/Metal failure is a failed validation, not proof the
canonical fallback ran. Do not add per-layer synchronous waits to hide it.

Windows preparation checks: 25 portable/converter tests passed, including the
seven new runtime-control/layout tests and the existing CUDA fitting guard.
All six new/modified Python runtime/test files parsed successfully and
`git diff --check` passed. The focused native pytest command could not start
in the converter venv (`No module named pytest`); Windows cannot execute
Metal regardless. No native parity or runtime speed claim is made here.

## Independent experiment arms

Set process-local environment before importing the model. Restart **only** the
isolated experimental process when changing arms; never mutate the mainline.

| Arm | EAGER_DISPATCH | FAST_RMS_NORM | HC_FUSED | PLE_BATCHED_GATHER |
| --- | ---: | ---: | ---: | ---: |
| A: original control | 0 | 0 | 0 | 0 |
| B: dispatch | 1 | 0 | 0 | 0 |
| C: norm | 0 | 1 | 0 | 0 |
| D: HC | 0 | 0 | 1 | 0 |
| E: PLE gather | 0 | 0 | 0 | 1 |
| F: dispatch + norm | 1 | 1 | 0 | 0 |
| G: dispatch + HC | 1 | 0 | 1 | 0 |
| H: dispatch + PLE | 1 | 0 | 0 | 1 |
| I: norm + HC | 0 | 1 | 1 | 0 |
| J: norm + PLE | 0 | 1 | 0 | 1 |
| K: HC + PLE | 0 | 0 | 1 | 1 |
| L: dispatch + norm + HC | 1 | 1 | 1 | 0 |
| M: dispatch + norm + PLE | 1 | 1 | 0 | 1 |
| N: dispatch + HC + PLE | 1 | 0 | 1 | 1 |
| O: norm + HC + PLE | 0 | 1 | 1 | 1 |
| P: ALL ON | 1 | 1 | 1 | 1 |

This is the complete four-factor matrix: control, four singletons, six pairs,
four triples and all-on. Stage singletons before combinations. Exclude arms
containing any native-correctness failure; do not average failed arms or
silently replace them with fallback measurements. Report observed path use.

Prefix each column name with `OMLX_QWEN4_`. PLE SSD **offload remains enabled**;
`PLE_BATCHED_GATHER` is only the optional row-assembly/dequantization path.
Do not add either upstream SSD threading/cache PR in this first experiment.
Their cold-read gains, memory costs and interaction with our mmap reader need
a separate measured adaptation. Memory/prefill PRs are also separate work.

## Native validation before model measurements

Keep the existing experimental `.venv` and pinned dependencies. No weight,
synthetic fixture or model rebake is needed for these unit tests. The kernels
in this candidate are JIT Metal; the existing native Bonsai extension must
still be available and retain its lazy-layout correctness fix.

```bash
uv run --no-project --python .venv/bin/python python -m pytest -q \
  tests/test_qwen4_eager_dispatch.py tests/test_qwen4_rms_norm.py \
  tests/test_qwen4_hc_fused.py \
  tests/test_bonsai_t5_load.py tests/test_qwen35_moe_gate_up.py \
  tests/test_bonsai_qmv.py tests/test_qwen4_t5_routed_numerics.py \
  tests/test_mlx_vlm_qwen4_exp_compat.py tests/test_qwen4_ple_batched.py \
  tests/test_qwen4_exp_residency.py
```

The new test fixtures explicitly enable the paths they test. Record skips and
failures; Metal/kernel skips are not passes. Do not loosen numerical tolerances
to obtain a green run. Check the loaded model's actual HC routing as well:
tests of synthetic eligibility alone do not prove production use.

Preserve mainline helper publication as well as its venv/settings: `--base-path`
alone is not complete isolation. Reuse the previously validated external
bootstrap and verify the global helper hash before and after testing.

## Small evaluation loop, not the 700-case suite

The user selected the original fixed **20-question pilot**, plus:

1. `Who was Copernicus?` for interactive first-token latency and generation.
2. A **4096-token input**, including the chat wrapper and disabled-thinking
   prefix, for prompt processing. Confirm server-reported input count too.

The Windows controller has frozen these in a separate, non-Git research
manifest. Transfer that manifest privately; do not reconstruct or resample the
pilot or commit benchmark prompts/results here. No inference was launched by
the offline preparation. The 20 questions retain their original requests,
scorer, seed, max-token budgets and non-streaming mode. The two performance
extras use streaming usage metrics and a 512-token output cap, not a forced
512-token minimum. They are not part of the 20-point score.

Run the complete 22-case block **at least twice per eligible arm**, preserving
both raw runs, all answers and individual score flips. Use a first sweep A
through P followed by a reverse sweep P through A to reduce ordering bias;
give each arm a fresh isolated process and identical warmup policy in each
sweep. Record first-use/JIT separately, then perform the measured block after
the same unscored warmup. Keep OS page-cache state observational, not purged.
The minimum is 32 measured blocks / 704 requests if all arms qualify, not
704 distinct questions. Honor the user's laptop availability deadline: record
partial completion, never silently reduce repeats or declare single-run wins.

For every arm and case, show both measurements, arithmetic mean and range;
aggregate PP/TG also as total relevant tokens divided by total relevant time.
Keep per-run 20-question scores and changed case IDs rather than presenting
the repeated cases as 40 independent questions. Recheck baseline and the
provisional winner in A/B/B/A order if the apparent advantage is small or the
two sweeps disagree. Do not pick a winner from one unusually fast run.

Mark first-use separately from warmed repetitions; report
server cached-token counts. A 4096-input response with reused prefix tokens
is **not** an uncached 4096-token PP measurement. Do not purge the OS page cache
or modify VM/wiring settings. Use the isolated server's agreed cache policy.

Record client first-content latency, server PP and generation durations/rates,
actual input/output counts, total elapsed time, cache hits, completion reason,
reasoning leaks and the full response. Output tokens divided by total elapsed
time is end-to-end throughput, not decode TPS. If generation ends early, do not
present it as a sustained 512-token test. Dashboard session averages are not
per-request measurements.

Keep power mode/charger state, context/KV settings, model memory guard,
concurrency, PLE SSD location, thinking off and MTP off fixed. Capture CPU/GPU
activity, pressure/swap and SSD reads around slow probes. Investigate the
reported 4 TPS with a trace, rather than assuming throttling or a cold cache.

Twenty questions are a regression smoke, not statistical proof of unchanged
quality. Exact answer text need not survive BF16 fusion rounding; inspect
numerical parity, answer flips and truncations together. Do not run another
700-case evaluation during optimization unless the user requests it.

## Coordination

No mailbox read/post is authorized by this document. The original user-gated
HF discussion protocol remains in force. The user authorized this candidate's
commit/push and one handoff message; this is not standing permission to fetch
or post results. Do not start/reconfigure its server from Windows during
preparation. Mac-side isolated test restarts are part of the requested matrix;
preserve its mainline installation and any unrelated work.
