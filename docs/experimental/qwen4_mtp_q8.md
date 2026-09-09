# Qwen4 T5 + Q8 MTP candidate

User decision (2026-09-08): restore the original one-layer MTP head at Q8.
Keep the measured prefix-fit T5 target, Q8 SSD PLE, tokenizer and template
unchanged. No imatrix, new runtime optimization or KV-cache quantization work.
KV-cache quantization is parked for upstream; retain current cache settings.

Windows build completed: 74 MTP storage tensors, 2,771,370,496 payload bytes
(2.581 GiB), all 131 target shards unchanged. The sampled reconstruction audit
covered 21 quantized matrices (up to 32 evenly spaced rows each), with maximum
relative RMSE below 1.6%; all 11 retained BF16 tensors matched in full. These
are weight reconstruction checks, not KLD or runtime acceptance measurements.
The portable converter/runtime-control suite passed 31 tests on Windows.
The additional native pre-split-Q8 sanitizer regression remains Mac-only and
has not been executed on Windows. Full MTP load/generation validation is pending.

## Build and identity

`tools/add_qwen4_mtp_q8.py` creates a NEW checkpoint from the existing target
and original BF16 source. It does not modify the normal converter's deliberate
MTP stripping or overwrite any existing output directory.

- Exact pinned source revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Required target: Q8 PLE from the pinned source with zero existing MTP tensors;
  T5 targets must use the prefix fitter, affine (`--expert-format affine`)
  targets are accepted as-is, with or without an imatrix (2026-09-08).
- MTP matrices: MLX affine Q8, group64, BF16 scales/biases.
- MTP norms and both MoE routing gates: original BF16, explicit quantization
  exclusions. Norms are NOT recentered independently of the target.
- Raw fused expert gate/up is split along the output axis BEFORE packing;
  the three expert banks are exported at `mtp.layers.0.mlp.switch_mlp.*`.
- Token embedding/output head are shared with the target. No MTP PLE table.
- One new `mtp-q8-g64.safetensors` shard and independent config/index.
- Base safetensors are hardlinked by default. NEVER edit those shards in place;
  they are the same immutable files used by the baseline. `--link-mode copy`
  instead requires enough space for a full independent copy.

Use the existing isolated Windows CUDA environment, with explicit HF_HOME:

```powershell
$env:HF_HOME = 'D:\hf-cache'  # your SSD-backed HF_HOME
$qwenSource = Join-Path $env:HF_HOME 'hub\models--Qwen--Qwen3.8-Flash-Next\snapshots\de4b8e4d43b917e7706784d8bb445c9af86a3540'
uv run --no-project --python .venv/Scripts/python.exe python -B tools/add_qwen4_mtp_q8.py `
  --source $qwenSource `
  --base (Join-Path $env:HF_HOME 'artifacts\qwen4-t5-prefix-ple8') `
  --output (Join-Path $env:HF_HOME 'artifacts\qwen4-t5-prefix-ple8-mtp8')
```

The same script's `--verify-only --base BASE --output CANDIDATE` checks the
independent tensor schema, quantization/config/index, MTP SHA-256 and unchanged
base shards. `--audit-only --source SOURCE --base BASE --output CANDIDATE`
compares fixed sampled matrix rows and full BF16 norms/routers to the source.
The historical main converter's `--verify-only` intentionally rejects MTP;
use this candidate-specific verifier instead.

## Small transfer to the Mac

Do not overwrite the Mac's measured checkpoint. Make a NEW candidate directory,
hardlink/reflink/copy the 131 immutable baseline shards into it, and copy the
unchanged tokenizer, processor and generation sidecars. Copy independent new
files from the Windows candidate:

- `mtp-q8-g64.safetensors`
- `config.json` and `model.safetensors.index.json`
- `omlx_conversion.json` and `omlx_mtp_q8.json`
- `mtp_verification.json`, `mtp_reconstruction.json` (when present), `README.md`

Do NOT hardlink mutable metadata. Do not retain the old README/MAC_HANDOFF or
conversion manifest as descriptions of the augmented model. The MTP manifest
records the required original config/index SHA-256; compare these to the Mac
base before assembly. After assembly, rerun the candidate-specific verifier.
Only the ~2.6 GiB head and small changed sidecars need transferring when the Mac
already has the exact baseline. HF publication is a separate user decision;
do not replace the baseline HF main with an unvalidated MTP candidate.

## Native validation gates

Use the existing isolated Mac checkout/venv/bootstrap and preserve mainline
settings and global helper. Source build requires `OMLX_WITH_CUSTOM_KERNEL=1`.
Runtime optimizations are the existing opt-in flags from the bd9e051d candidate;
do not introduce any additional runtime changes as part of this head experiment.

1. Verify actual MTP weight discovery and module construction, correct Q8
   module types, BF16 routers/norms and no missing/unexpected weights. Retain
   PLE SSD offload. Cache precision stays unchanged. Capture actual flags.
2. Run focused Qwen4/Bonsai native tests, including
   `tests/test_mlx_vlm_qwen4_exp_compat.py`, and the converter-specific portable
   tests where the isolated environment includes Torch.
3. Compare the SAME augmented artifact with MTP disabled/enabled, greedy,
   depth1 initially, fixed prompts and identical optimization settings. Check
   token/logit parity and rejected-draft rollback (QSA, DeltaNet, PLE history).
   Do not assume successful load or good aggregate answers proves correctness.
4. Measure cold load, prefill, draft/verify peaks under the established 48 GB
   memory guard, then steady state, swap and pressure. Head file size alone is
   not a runtime memory budget. Stop under the established pressure criteria.
5. Only after correctness, use the frozen 20-question probe plus Copernicus
   and uncached 4096-token input. At least two runs per eligible arm, report
   acceptance, draft/verify costs, TTFT, TG and memory. Try depths2/3 only after
   depth1 passes; keep a serial control. No 700-case rerun.

MTP is a speed experiment, not a quality repair. Exact target verification is
the contract, not yet established for this particular T5/Q8-MTP combination.
No performance or memory-fit claim is implied by structural validation.

Do not read/post to the HF discussion without explicit user authorization.
