# Qwen3.8-Flash-Next T5 handoff

Integration update (2026-09-08): the upstream merge incorporates
`94530d8d49541ede9e99ef04a4431ee4953117a6` and includes the Mac fixes through
`0820cfc96eefcea20d156b7058ea05cda5810c11`. By user decision, fused HC now
matches upstream exactly: no local barrier correction or extra tensor-shape
checks. HC fusion and eager dispatch default on (upstream disable switches
remain); fast RMS uses upstream's unconditional implementation and the old
`OMLX_QWEN4_FAST_RMS_NORM` variable has no effect. Historical default-off
instructions below do not apply to these merged optimizations. Our separate
PLE batched-gather experiment remains opt-in. T5 loading/kernels and Q8 PLE
SSD offload remain intact. The barrier concern is not resolved by reverting
to upstream; this is not a measured correctness or performance improvement.
Native validation requires an isolated Mac environment and a fresh custom
kernel build for the merged MLX 0.32.2 dependency; do not alter mainline omlx.
Windows validation: 39 portable tests passed and all 150 changed Python files
parsed successfully. Native focused tests were not run: the Windows converter
environment has no pytest or native MLX/Metal runtime. These checks do not
establish Mac numerical parity or speed.

Latest candidate (2026-09-08): [Q8 MTP restoration](qwen4_mtp_q8.md).
Separate augmented checkpoint; the measured no-MTP baseline is unchanged.
Mac testing found deterministic greedy MTP divergence on Chinese, Copernicus
and the long-prompt probe; head-loaded MTP-off controls matched the original
baseline. MTP remains experimental and the upstream merge is not a confirmed
fix. KV-cache quantization
is parked for upstream; do not explore or change it as part of this work.

Latest work (2026-09-07): [runtime tuning candidate and small evaluation loop](qwen4_flash_next_runtime_tuning.md).
The prefix-fit/Q8-PLE artifact has since been baked and natively validated;
the older no-conversion/pending-runtime statements below are historical.
Freeze those weights for runtime tuning. New optimizations default off and
still require their own native validation. Imatrix remains parked.

Current status (2026-09-06): generated models were deleted on Windows and the
later Mac performance work was rolled back. The first weight-only T5 was
reasonably coherent with factual errors (user report); T5-imatrix was unusable.
Do not conflate them. The Mac reported still having the first weight-only
checkpoint. The agreed review baseline is
`dc7aaee37066c139d58ad5540646644d36140f39`. Imatrix is parked. PLE defaults to
Q8; weight-only fitting now defaults to a guarded prefix search, with explicit
`--t5-fitter legacy` for the original-solver control. The BF16 source download
finished at pinned revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`; no new
full conversion has run. The native lazy-layout fix still requires Mac testing.

Read [the correctness/fitter update](qwen4_flash_next_t5_correctness_fitter.md)
for the changes, real-weight measurements and required native validation.

Read [the Q8 restart and performance plan](qwen4_flash_next_q8_restart.md) for
current changes, validation gates, and outstanding recipe improvements. The
historical artifact sizes and generation results below describe Q2 PLE models,
not a Q8 model. Later quality/performance improvements remain unproven.

This branch explores whether Qwen3.8-Flash-Next can run on a 48 GB M3 Max by
combining OMLX's SSD-mmap support for Qwen4 PLE n-grams with an AngelSlim-inspired
sub-2-bit expert representation. The model must be treated as `qwen4_exp` (a
Qwen4 experimental architecture), not as Qwen3.5.

## Handoff coordinates

Published artifacts and coordination:

- OMLX fork/branch: `https://github.com/fuutott/omlx/tree/qwen4-flash-next-t5`
- Original OMLX implementation commit: `9ae674fc931070eab1c56780abaa5ccc9187273e`
- Agreed code-review baseline: `dc7aaee37066c139d58ad5540646644d36140f39`
- Private Hugging Face checkpoint: `https://huggingface.co/fuutott/Qwen3.8-Flash-Next-MLX-t5`
- Base checkpoint revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Published checkpoint revision: `7f093be9c5efbfa04f471f025c882ab0d664b42c`
- Historical `imatrix-v2` upload completed at `def75d34004cd7e229b059639209fc1bb24a792a`.
  This is a historical record, not a fresh check of remote availability.
- Canonical HF coordination thread: `https://huggingface.co/fuutott/Qwen3.8-Flash-Next-MLX-t5/discussions/1`
- Former Windows artifact (deleted): `D:\hf_models_cache\artifacts\Qwen3.8-Flash-Next-MLX-t5`
- Former Windows imatrix artifact (deleted):
  `D:\hf_models_cache\artifacts\Qwen3.8-Flash-Next-MLX-t5-imatrix`
- Windows conversion report: `omlx_conversion.json` in the artifact

The Mac needs the modified OMLX checkout and the converted MLX checkpoint. It
does not need AngelSlim, the Windows `.model-research` directory, CUDA, PyTorch,
or the converter environment merely to run the model.

The HF thread is a user-gated mailbox. Its one-message-at-a-time read/write
rules apply only to that discussion, not other communication. Read
`docs/experimental/qwen4_flash_next_t5_comms.md` before accessing it.

## What changed

The conversion recipe is architecture-specific and lives in
`tools/quantize_qwen4_flash_next_t5.py`. Its isolated Windows dependencies are
locked by `tools/qwen4_flash_next_quant.in` and
`tools/qwen4_flash_next_quant.lock`.

The current recipe is:

- routed expert gate/up: Bonsai base-3 T5, group size 128;
- routed expert down: affine 2-bit, group size 128;
- PLE n-gram embeddings: affine 8-bit by default, group size 32, retained in
  independently mmap-able shards (`--ple-bits 2` reproduces the historical control);
- shared experts: affine 8-bit;
- attention and DeltaNet projections: affine 5-bit, except QSA `o_proj`, which
  was and remains affine 4-bit in the controlled baseline;
- token embeddings and LM head: affine 6-bit;
- remaining eligible backbone matrices: affine 4-bit;
- vision, routers, recurrent state, convolutions, and norms: BF16;
- MTP weights: removed for this first 48 GB target.

Added 2026-09-08: `--expert-format affine` bakes plain MLX affine Q2/Q3 routed
experts with no T5 marker, for stock oMLX. Affine tensors then use a two-edge
importance-weighted range search, and `--imatrix-scope safe` limits imatrix
weighting to tensors whose GGUF input order was checked against llama.cpp's
`conversion/qwen4exp.py`. The DeltaNet `out_proj` permutation noted below is
llama.cpp's tiled V-head reorder; `tools/qwen4_flash_next_imatrix.py` now
undoes it for `ssm_out`, which resolves that finding. See the README "How to
bake the cake" step 2b.

The runtime additions teach the Bonsai T5 loader to accept rank-3 expert banks,
add a native routed T5 gather-QMV path for decode, and reuse dense T5 QMM over
contiguous expert runs during sorted prefill. Qwen gate/up fusion also accepts
the scalar ignored-bias placeholder left after T5 bias release.

PLE n-gram tensors remain on SSD through OMLX's existing
`DiskBackedShardedEmbedding` path. This is essential: checkpoint size is not the
same as resident unified memory.

## Historical importance-aware conversion follow-up (parked)

The review found an unresolved GGUF/HF input-channel permutation for DeltaNet
`out_proj`. Strict name/shape matching does not detect that issue, and unmapped
modules are exempt from the strict check. Do not interpret the audit below as
proof of complete/correct calibration coverage. Imatrix now requires explicit
`--allow-experimental-imatrix`; leave it out of the Q8 baseline.

The converter now accepts `--imatrix PATH` and `--imatrix-strict`. The importer
in `tools/qwen4_flash_next_imatrix.py` memory-maps a llama.cpp/Unsloth GGUF
importance matrix, validates `general.type=imatrix`, maps Qwen4 experimental
tensor names, and converts accumulated squared activations into per-input
channel energy. PLE tables are intentionally exempt because they are embedding
lookups rather than linear activations.

The tested source file has SHA-256
`a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`, 926
entries, 144 expert entries, and 73,728 expert slots. Only 24 routed-expert
slots had zero observations; those are imputed from the channel-wise mean of
observed experts rather than interpreted as zero importance.

The external activation energy now weights both T5 least-squares fitting and
affine clipping search. On real source shards:

- routed gate T5 weighted relative RMSE improved from 0.46508 to 0.45235;
- routed up T5 weighted relative RMSE improved from 0.45808 to 0.44595;
- routed down 2-bit weighted relative RMSE improved from 0.44180 to 0.38136.

These are historical reconstruction measurements, not end-to-end KLD, and the
user subsequently reported that T5-imatrix was unusable. The original
weight-only solver is now available explicitly as `--t5-fitter legacy`;
the default prefix solver is documented in the current update above.

The full strict conversion completed in 12.4 minutes on the Windows CUDA host.
Its independent `--verify-only` pass reports 131 shards, 3,671 tensors,
53,917,360,592 converted tensor bytes, 48 expert layers, and all 128 PLE weight
shards. The conversion audit records 852 applied imatrix entries, zero missing
entries, zero shape mismatches, and 24 imputed expert slots. The resulting
checkpoint was subsequently uploaded as `imatrix-v2`. End-to-end quality and
native numerical correctness were not established by structural verification.

## First Mac evidence

The original checkpoint was tested behind OMLX as
`Qwen3.8-Flash-Next-MLX-t5`. A draft 12-prompt run with thinking enabled
generated 2,989 completion tokens in 138.55 model-seconds (21.57 aggregate
output tokens/s). A corrected run with thinking disabled generated 170 tokens
in 18.8 model-seconds (9.04 aggregate output tokens/s). These short-request
aggregate rates include prompt/first-token overhead and are not a replacement
for a fixed-length sustained-decode benchmark.

The corrected no-thinking smoke scored 8/12. One genuine miss answered `Biały`
(white) to a Polish blue-sky question; `błękitny`/`blekitny` and grammatical
variants are now accepted by the scorer. Two other semantic misses answered 9
instead of 30 for modular arithmetic and Ben instead of Ava for an ordering
question. The code-only miss returned correct code inside Markdown fences.

For a base-model comparison, local LM Studio Q4_K_XL (104.53 GiB loaded) and
Q8_0 (176.14 GiB loaded) each scored 10/12 on the same corrected corpus with
thinking enabled. Both exhausted the 512-token budget reasoning about the
code-only task and both answered 9 instead of 18 for `0+3+6+9`. This small
corpus therefore does not attribute those two failures to low-bit
quantization. OpenRouter's shared Alibaba pool was too rate-limited to produce
a complete comparison run.

The T5 inference route is native Metal, not a Python dequantization loop:
routed gate/up uses `bonsai_t5_gather_qmv`, gate and up are fused into one
projection, and routed down uses MLX's native affine `gather_qmm`. A current
performance-review target is that both dense and routed T5 kernels instantiate
`qmv_fast_t5_impl` with `USE_SIGMA=false`, so activation group sums and base-4
pre-scaling are repeated across output tiles. Any change here must be
benchmarked on the M3 Max; adding separate preprocessing dispatches may cost
more than they save.

## Mac setup

Prerequisites are Apple Silicon, macOS 15 or newer, full Xcode (not only Command
Line Tools), Git, `uv`, and the Hugging Face CLI. Use an SSD-backed location with
ample free space for `HF_HOME` and the checkpoint.

```bash
export HF_HOME=/absolute/path/to/hf-cache

git clone <OMLX_FORK_URL>
cd omlx
git switch qwen4-flash-next-t5

uv venv --python 3.12 .venv
OMLX_WITH_CUSTOM_KERNEL=1 uv pip install -e .

uv run python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"
```

Stop if the `bonsai` entry reports `available: False`. Capture the complete
import error and verify that `xcrun -f metal` resolves into the full Xcode
toolchain before rebuilding.

Download the private checkpoint without putting it in the Git checkout:

```bash
hf auth whoami
hf download <HF_CHECKPOINT_REPO> --local-dir /absolute/path/to/checkpoints/Qwen3.8-Flash-Next-MLX-t5

uv run python tools/quantize_qwen4_flash_next_t5.py \
  --verify-only /absolute/path/to/checkpoints/Qwen3.8-Flash-Next-MLX-t5
```

Keep `HF_HOME` exported for OMLX as well. Do not copy the model into the fork or
commit any generated weights.

## First-load protocol

Start conservatively, with only this model visible to the server:

```bash
uv run omlx serve \
  --model-dir /absolute/path/to/checkpoints \
  --memory-guard safe \
  --memory-guard-gb 48
```

Use the admin UI to confirm that the model is detected as `qwen4_exp` and that
Qwen4 PLE SSD offload is enabled or reported as forced. Do not load a second
model during the initial test.

Validate in this order:

1. Model discovery completes without materializing all PLE shards into memory.
2. Model load succeeds and the process remains comfortably below the 48 GB
   ceiling before KV-cache growth.
3. A one-token prompt produces finite logits and one decoded token.
4. Short English and Chinese prompts remain coherent for at least 128 generated
   tokens with greedy or low-temperature sampling.
5. A 2K prompt completes prefill; record prefill and decode tokens/second.
6. Repeat with 8K only after the shorter run is stable.

Capture peak memory from Activity Monitor or `memory_pressure`, relevant OMLX
logs, SSD read activity, prompt/output pairs, and all version information:

```bash
sw_vers
xcodebuild -version
uv run python -c "import mlx; print(mlx.__version__)"
git rev-parse HEAD
```

## Focused code validation on the Mac

Run the tests that exercise the modified dispatch and fusion behavior:

```bash
uv run pytest -q \
  tests/test_bonsai_t5_load.py \
  tests/test_qwen35_moe_gate_up.py
```

The test module retains its historical Qwen3.5 name because the fusion patch is
shared; the checkpoint and architecture validation are Qwen4-specific.

## Distribution-fidelity validation

No end-to-end KL divergence has been measured yet. The Windows measurements in
this document are T5 reconstruction RMSE/cosine checks and must not be reported
as KLD.

For a meaningful result, generate teacher logits from the original checkpoint
on Windows for a fixed multilingual/code prompt corpus, then capture logits at
the same token positions from this exact checkpoint through the shipped MLX and
native Bonsai Metal path on the Mac. Compute `KL(teacher || quantized)` in FP32
and also report JS divergence, top-1 agreement, and mean/median/p95 token KLD.
If full-vocabulary teacher logits are impractical to transfer, store top-k
logits plus log-sum-exp/residual probability mass and label the result as an
approximation. Record the prompt corpus and calculation code with the metrics.

## Expected constraints

The converted safetensors measure 53,917,360,592 bytes (50.22 GiB). OMLX's
actual residency detector reports 19,200,092,160 bytes (17.88 GiB) of PLE tensor
data, `supported=True`, a 52.72 GiB fully-resident estimate, and a 33.95 GiB
estimate with PLE mmaped. `force_ssd_offload(48 GiB)` returns `True`. The Mac
still needs to measure real process peak, activations, cache headroom, and SSD
behavior.

The T5 gate/up representation is approximately 1.875 bits per weight on disk
including FP16 scale and bias metadata. Runtime bias release reduces its
resident cost. It is not AngelSlim's exact forced 3:4 STQ1_0 algorithm: the
layout is adapted to an existing OMLX base-3 kernel. The published first pass
did not use an importance matrix; the follow-up converter can use the Unsloth
matrix as described above. Accordingly, do not claim “little loss” before a
complete imatrix checkpoint has end-to-end generation and KLD evidence.

## Failure triage

- Bonsai extension unavailable: fix the Xcode/Metal source build first.
- Rank or packed-width error on expert weights: compare gate/up/down tensor
  shapes against the converted index; do not coerce them as Qwen3.5 tensors.
- Memory spike during discovery/load: verify PLE sidecar names use
  `ngram_embedding.shards.N` and that SSD offload is enabled before attempting a
  smaller context.
- Decode works but prefill fails: inspect the sorted expert-run T5 QMM route.
- Prefill works but decode fails: inspect the native T5 gather-QMV route and the
  route-to-input cardinality calculation.
- Grammatically fluent but factually or logically broken output: preserve logs
  and investigate quantization, runtime numerics, prompting, and base-model
  errors. A factual miss alone does not identify which component is responsible.
