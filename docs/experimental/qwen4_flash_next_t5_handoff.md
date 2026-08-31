# Qwen3.8-Flash-Next T5 handoff

Status: converted and structurally validated on Windows; not yet validated on
Apple Silicon.

This branch explores whether Qwen3.8-Flash-Next can run on a 48 GB M3 Max by
combining OMLX's SSD-mmap support for Qwen4 PLE n-grams with an AngelSlim-inspired
sub-2-bit expert representation. The model must be treated as `qwen4_exp` (a
Qwen4 experimental architecture), not as Qwen3.5.

## Handoff coordinates

Published artifacts and coordination:

- OMLX fork/branch: `https://github.com/fuutott/omlx/tree/qwen4-flash-next-t5`
- OMLX implementation commit: `9ae674fc931070eab1c56780abaa5ccc9187273e`
- Private Hugging Face checkpoint: `https://huggingface.co/fuutott/Qwen3.8-Flash-Next-MLX-t5`
- Base checkpoint revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- Published checkpoint revision: `7f093be9c5efbfa04f471f025c882ab0d664b42c`
- Canonical HF coordination thread: `https://huggingface.co/fuutott/Qwen3.8-Flash-Next-MLX-t5/discussions/1`
- Windows artifact: `D:\hf_models_cache\artifacts\Qwen3.8-Flash-Next-MLX-t5`
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
- PLE n-gram embeddings: affine 2-bit, group size 32, retained in independently
  mmap-able shards;
- shared experts: affine 8-bit;
- attention and DeltaNet projections: affine 5-bit;
- token embeddings and LM head: affine 6-bit;
- remaining eligible backbone matrices: affine 4-bit;
- vision, routers, recurrent state, convolutions, and norms: BF16;
- MTP weights: removed for this first 48 GB target.

The runtime additions teach the Bonsai T5 loader to accept rank-3 expert banks,
add a native routed T5 gather-QMV path for decode, and reuse dense T5 QMM over
contiguous expert runs during sorted prefill. Qwen gate/up fusion also accepts
the scalar ignored-bias placeholder left after T5 bias release.

PLE n-gram tensors remain on SSD through OMLX's existing
`DiskBackedShardedEmbedding` path. This is essential: checkpoint size is not the
same as resident unified memory.

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
layout is adapted to an existing OMLX base-3 kernel, and no importance matrix is
used in this first pass. Accordingly, do not claim “little loss” before
generation and evaluation evidence exists.

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
- Grammatically fluent but factually or logically broken output: treat this as
  quantization damage. Preserve logs and outputs before changing the recipe.
