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

## Initial validation and resolved build blockers

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

September 12 retry at merge commit b3ac7d15cf4b13c12abb3bac07f82146b377d78f:
process-local DEVELOPER_DIR, SDKROOT, CC and CXX select Xcode 27 RC (27A266a).
The C++ compiler/link probe now passes. Native build still fails during Bonsai
Metal compilation: `cannot execute tool 'metal' due to missing Metal Toolchain;
use: xcodebuild -downloadComponent MetalToolchain`. A direct `xcrun metal
--version` confirms the same error. No fallback or native test success claimed;
the server remains stopped. Global toolchain selection remains unchanged.

## September 12 successful native validation

After the user installed the Metal Toolchain, the same process-local Xcode 27
build succeeded. Metal reports 32023.921 (metalfe-32023.921.6); macOS is 27.0
(26A428), Python 3.12.13, MLX 0.32.2, Apple M3 Max. Bonsai, decode_fast,
glm_moe_dsa, minimax_m3 and qwen35_prefill all report available with no import
error. No old extension reuse or runtime fallback was used to satisfy this gate.

Sixteen test modules ran serially in fresh processes: 428 passed, zero failures
and zero skips. They cover Bonsai loading (42), gate/up fusion (15), Bonsai QMV
(71), actual T5 routed/dense lazy-layout numerics (37), vendored Qwen4 compat
(66), exact PLE batch/reference equality (4), residency (3), eager dispatch (6),
RMSNorm (38), HC fusion (77), QSA row gather (20), decode gather (13), verify
gather (5), cached text positions (9), prefill memory (5), and expert-offload
eligibility/settings (17). These include both mocked and actual native tests;
the total is not 428 full-model generations. Test-session TF32 is disabled by
upstream conftest; this is not an M3 tensor-unit optimization.

Header-only inspection of the existing T5/Q3-down/Q8-PLE artifact confirms
qwen4_exp, forced PLE SSD offload at 48 GiB, and rejection of T5 gate_proj by
expert-offload shape/dtype eligibility. Estimated mmap residency is
41,432,607,132 bytes; this is a static estimate, not measured process peak.

The server remains stopped at the user's request. No full-model speed,
generation quality, peak memory or physical SSD throughput was measured for
this merge. The existing experimental checkout and global helper are unchanged;
the system-default Xcode selection is unchanged. Raw build/status/test logs
are retained outside Git in the workspace runtime results directory.

Next step: validate the same
checkpoint serially with SSD PLE, existing memory guard, short EN/ZH and bounded
prefill before any serving cutover. Never load both full models at once on48GB.
Retain raw old/new outputs, errors, memory/swap and timings; do not infer the
cause of another Q2 checkpoint's incoherence from these source changes alone.
