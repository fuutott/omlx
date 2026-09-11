# Qwen4 upstream sync candidate — 2026-09-11

Baseline: c85c35ffbf24f83a085538266f3942eb68c43c1a.
Upstream: b390b31e0c6831225fed0f24d278eb1db7fcb68b, version0.7.0.dev2.
This is a separate candidate, not a validated replacement for the serving build.

## Relevant changes

- Qwen4 prefill/P​​LE batching and next-chunk prefetch: #3534, 9e71c6e0;
  follow-up970aacdc preserves prefill normalization and drains pending PLE reads.
- Qwen4 long-context QSA: #3520, bd6f5227 and535ce89f. Decode/verify gathers
  selected rows without materializing a whole token-major cache copy; text-only
  step proofs make the gathered route eligible. Wide-prefill dispatch differs.
  Upstream M5 figures are not predictions for M3/T5. MTP remains disabled here.
- MoE checkpoint-backed expert streaming:6df0d8d6 and75466d33. It fetches the
  selected experts on a cache miss, not pruning/zeroing nonresident experts.
  The qwen4_exp adapter accepts validated affine/MXFP layouts, not our uint8
  T5 banks. PLE SSD offload remains an independent feature. Do not bypass
  eligibility or turn this on for the T5 model.
- M5 INT8-activation prefill:#3548. Hardware/shape eligibility must be honored;
  this is not an M3 Max optimization.
- Other upstream changes (settings, thinking controls, usage history, cluster,
  other architectures) are included by the full upstream merge, not represented
  as independently tested features.

## Merge resolution

Only the vendored Qwen4 language module conflicted. Its PLE implementation
overlapped the fork's batched gather. Upstream `_plan`/`_assemble`, upload and
prefetch lifecycle are now authoritative. Keep the fork's owned NumPy reader,
bounds/closed-mapping checks, constructor cleanup, non-POSIX fallback and helper
names for existing tests/tools. No duplicate hot gather implementation remains.
The historical PLE_BATCHED_GATHER switch defaults on and explicit0 selects the
per-shard reference (without prefetch). Update the existing equality test to
inspect the new plan instead of the removed legacy uniform-layout attribute.
Empty calls now reset row/touched counters, avoiding stale diagnostic state.

T5 rank3 loading/native routed dispatch and the chat-render/benchmark-recursion
fixes are retained. Converter precision policy is unchanged. No model download,
new conversion, expert-offload experiment, MTP experiment or KV-quantization work.

## Validation and blocker

- Separate uv Python3.12.13 .venv,112 exact dependency versions captured from
  the previous experimental environment; MLX/mlx-metal0.32.2 unchanged.
- Seven portable owned-row/prefetch reader tests passed.
- Chat render scheduling JavaScript test passed: no idle redraws,5Hz limit,
  coalescing, cleanup and chat switching.
- All216 changed Python files parsed; diff whitespace check passed.
- Native editable build with OMLX_WITH_CUSTOM_KERNEL=1 FAILED before Bonsai
  compilation, in CMake's C++ compiler link probe. System Command Line Tools
  macOS27 SDK `.tbd` files contain arm64e.x1 entries rejected by the selected
  linker. Full Xcode26.6(17F113) and its Metal compiler are present;
  xcode-select already points at full Xcode. No global SDK/Xcode change made.
- Native focused tests, new offload/QSA tests and model validation NOT RUN.
  No native fallback, reuse of the old extension binary or speedup claim.

Next step: explicitly select a compatible full-Xcode SDK/toolchain for the
isolated build only, then verify native Bonsai and run focused tests in fresh
processes (prior tests have global-patch isolation hazards). Validate the same
checkpoint serially with SSD PLE, existing memory guard, short EN/ZH and bounded
prefill before any serving cutover. Never load both full models at once on48GB.
Retain raw old/new outputs, errors, memory/swap and timings; do not infer the
cause of another Q2 checkpoint's incoherence from these source changes alone.
