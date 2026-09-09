# Qwen48 — unofficial oMLX fork

> **Independent experiment, not an official oMLX release.** The Qwen48 work and
> reproduction recipe below belong to this fork; they are not presented as work
> endorsed or maintained by the upstream oMLX project.

## This fork: Qwen Flash Next on a 48 GB Mac

This fork runs **Qwen3.8-Flash-Next on a 48 GB M3 Max** using low-bit routed
experts and SSD-backed PLE n-gram embeddings. It is an experimental fork of oMLX,
not an upstream release or a general-purpose Qwen quantizer. The model identifies
as `qwen4_exp`: its hyper-connections, DeltaNet/QSA layers and PLE layout must not
be treated as Qwen3.5. Development happens on `main`.

**Released checkpoint:**
[fuutott/Qwen3.8-Flash-Next-MLX-t5-imatrix-q3down-ple8](https://huggingface.co/fuutott/Qwen3.8-Flash-Next-MLX-t5-imatrix-q3down-ple8)
(about 35.6 GiB resident, 38 tok/s decode, 64K-token requests on the M3 Max).
Its model card carries the measured memory, speed, KL-divergence and benchmark
numbers; the recipe below is how it was made.

### What changed against upstream oMLX, where, and why

- **Ternary (Bonsai T5) routed experts for `qwen4_exp`.** Where:
  `omlx/custom_kernels/bonsai/` (new `affine_gather_qmv_fast_t5` Metal kernel and
  `BonsaiT5GatherQmvPrimitive`), `omlx/patches/bonsai_t5_load.py`,
  `omlx/patches/qwen35_moe_gate_up.py`, `omlx/patches/m5_gather_qmm.py`,
  `omlx/utils/model_loading.py`. Why: the routed gate/up banks are stored at about
  1.875 bits/weight as rank-3 T5 tensors, which stock loaders reject and stock
  `gather_qmm` cannot read; the `omlx_t5` config marker switches the loader and
  dispatcher over, and the dispatcher chain is walked so the T5 and M5 reroutes
  cannot recurse on reload.
- **PLE n-gram table served from SSD.** Where:
  `omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/ple_mmap.py`
  (new) and `language.py`. Why: the 53.6 GiB Q8 embedding table must never be
  resident on a 48 GB machine, so rows are gathered through `mmap` per token; an
  optional batched CPU gather (`OMLX_QWEN4_PLE_BATCHED_GATHER=1`) is kept as an
  opt-in experiment.
- **Converter and bake tooling.** Where: `tools/quantize_qwen4_flash_next_t5.py`
  (T5 or affine expert formats, Q2/Q3 down, importance-weighted range search,
  verify and resume), `tools/qwen4_flash_next_imatrix.py` (llama.cpp GGUF imatrix
  importer, including the DeltaNet `out_proj` un-permute), `tools/add_qwen4_mtp_q8.py`
  (optional Q8 MTP head), `tools/t5_to_affine.py` (lossless T5 to affine re-pack so
  T5 bakes can be scored on stock kernels), `tools/qwen4_flash_next_quant.{in,lock}`.
  Why: bake and verify checkpoints on a Windows/CUDA host without touching the Mac.
- **Measurement tools.** Where: `tools/qwen4_flash_next_eval.py` and its corpus
  (12-prompt API smoke), `tools/qwen4_ple_bench.py` (PLE gather timing),
  `tools/qwen4_t5_fit_bench.py` (fitter error on real weights). Why: keep quality
  and speed claims measured rather than inferred.
- **Chat UI streaming fix.** Where: `omlx/admin/templates/chat.html`. Why: the
  streamed-answer repaint handle was reactive, so clearing it re-scheduled paints
  at display refresh rate with no new tokens; paints are now coalesced to 5 Hz by
  a timer held outside Alpine's reactivity.
- **Tests and docs.** Where: `tests/test_qwen4_*`, `tests/test_bonsai_*`,
  `tests/test_chat_render_scheduling.py`, `tools/tests/`, `docs/experimental/qwen4_*.md`.
  Why: converter tests run anywhere; native Metal numerics are tested on the Mac;
  the docs record the handoff, runtime tuning, fitter correctness and MTP notes.

There are two separate workstreams: creating better compact weights, and making
the native MLX/Metal runtime faster without changing those weights. The starting
inspiration was AngelSlim's low-bit post-training quantization work, but this is
**not Tencent's STQ1_0/GGUF recipe**. We use oMLX's Bonsai T5 format: five ternary
digits packed per byte, with per-group scaling and padding. T5 here is a packing
format, not the T5 language-model family. It is about 1.875 bits/weight including
stored scale/bias overhead for the gate/up expert matrices, **not the whole model**.
No AngelSlim checkout is needed to run this converter.

Status, **2026-09-09**: the released checkpoint (ternary + imatrix gate/up, Q3
down, Q8 PLE on SSD, no MTP) has completed its validation on the target Mac with
this fork built natively: about 35.6 GiB of resident weights, 41.5 GiB physical
peak at 8K context with 1.8-2.5 GiB of one-off startup swap, 38 tok/s decode,
64K+256-token chat requests, and a 129K-token prefill verified by oMLX's automatic
context sizing. Quality against a Q8_0 teacher on wikitext-2: mean KLD 0.489
(weight-only ternary 0.69, ternary + imatrix with Q2 down 0.57, affine Q2 +
imatrix 0.48, Unsloth UD-Q4_K_XL 0.036). On a fixed 700-case MMLU/GSM8K/TruthfulQA/CMMLU
subset it scores 587 (old weight-only ternary 578, Q4_K_XL 634); HumanEval 92.7 %,
MBPP 78.0 % (Q4_K_XL 95.7 % / 85.8 %). Known regression: Chinese instruction
following on strict-format prompts. The Unsloth importance matrix is applied
through a verified tensor mapping (DeltaNet `out_proj` un-permuted). Optional Q8
MTP remains research-only: it gave little on this hardware and cost several GiB
of swap, so keep it disabled. KV-cache quantization is left to upstream.

### How to bake the cake

These instructions create your own MLX safetensors checkpoint from the original
source model. The released checkpoint linked above was produced with exactly
these steps. Observe the source
model's license and access conditions; the runtime's license does not replace them.

#### Ingredients and current recipe

The tested conversion route is **Windows + NVIDIA CUDA**, with `uv`, Python 3.12
and the checked-in converter dependency lock. Our conversion host has 256 GB RAM
and two GPUs totalling 144 GB VRAM; that is a tested host, **not a minimum
requirement**. The converter uses one selected GPU in chunks, not pooled VRAM.
Minimum RAM/VRAM has not been established. A smaller `--chunk-rows` reduces GPU
working memory, but does not eliminate host-memory requirements.

Budget roughly 360 GB for the source plus 92 GB for one converted checkpoint,
with additional room for download caches, temporary files and optional copies;
600 GB free is a sensible planning allowance, not a measured peak requirement.
Use a fast SSD. The output is a directory of shards/config/tokenizer files,
not a GGUF or one self-contained file.

| Component | Current conversion |
| --- | --- |
| Routed expert gate/up | Bonsai T5, group 128, imatrix-weighted guarded prefix fitting |
| Routed expert down | MLX affine Q3, group 128, imatrix range search (`--expert-down-bits 2` for the smaller Q2 variant) |
| PLE n-gram embeddings | MLX affine Q8, group 32; 128 independently mmap-able shards |
| Shared experts / shared-expert gate | Affine Q8, group 128 / 64 |
| Token embeddings and LM head | Affine Q6, group 64 |
| Attention/DeltaNet projections | Affine Q5, group 64; QSA `o_proj` remains Q4/group 64 |
| Other eligible matrices | Affine Q4, group 64 |
| Vision tower Linear layers | Affine Q8, group 64 (`--vision-bits 8`; `--vision-bits 0` keeps them BF16) |
| MoE routers, norms, convolutions, recurrent parameters and the rest of the vision tower | Retain source precision (BF16 in the pinned source) |
| MTP | Omitted from the baseline; optional separate Q8 addition below |

The Q3-down output is approximately **90 GiB on disk**, including approximately
54 GiB of PLE tensors intended for SSD mmap; the Q2-down variant is about 86 GiB.
Resident weights are about 35.6 GiB (Q3 down) or 30.9 GiB (Q2 down); **that is
not total process memory**. Activations, KV
cache, mmap working pages and runtime overhead still need headroom. PLE is kept
at Q8 because its full table need not occupy unified memory. Do not reduce it to
Q2 just to shrink an SSD-resident file.

Step 2b describes the stock-oMLX variant with affine experts instead of T5.

#### 1. Prepare an isolated converter and download the pinned source

PowerShell, in a new checkout (adjust the SSD path first):

```powershell
git clone https://github.com/fuutott/omlx.git omlx-qwen48
Set-Location omlx-qwen48
git rev-parse HEAD  # Record this with your conversion results.

uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r tools/qwen4_flash_next_quant.lock --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match

# Keep downloads and generated artifacts outside the Git checkout.
$env:HF_HOME = 'D:\hf-cache'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_XET_CACHE = Join-Path $env:HF_HOME 'xet'
$qwenRevision = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
$qwenSource = Join-Path $env:HF_HUB_CACHE "models--Qwen--Qwen3.8-Flash-Next\snapshots\$qwenRevision"
$qwenOutput = Join-Path $env:HF_HOME 'artifacts\qwen4-t5-prefix-ple8'

# Install the HF CLI separately if needed; it does not belong in global Python.
uv tool install huggingface_hub
# If authentication is needed: hf auth login (never put tokens in this README).
hf download Qwen/Qwen3.8-Flash-Next --revision $qwenRevision --cache-dir $env:HF_HUB_CACHE
hf cache verify Qwen/Qwen3.8-Flash-Next --revision $qwenRevision --cache-dir $env:HF_HUB_CACHE --fail-on-missing-files

# Unsloth's importance matrix for this model; both recipes below use it.
hf download unsloth/Qwen3.8-Flash-Next-GGUF imatrix_unsloth.gguf_file --cache-dir $env:HF_HUB_CACHE
$imatrix = Get-ChildItem (Join-Path $env:HF_HUB_CACHE 'models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots') -Recurse -Filter imatrix_unsloth.gguf_file | Select-Object -First 1 -ExpandProperty FullName
```

Stop on any command failure. On Windows, enabling Developer Mode permits HF
cache symlinks and can avoid duplicate source storage. Do not install the Mac
oMLX runtime into this Windows converter environment or let `uv` sync the root
project's MLX dependencies. The commands below deliberately use `--no-project`.

#### 2. Test, convert, verify

```powershell
uv run --no-project --python .venv/Scripts/python.exe python -B -m unittest discover -s tools/tests -v
uv run --no-project --python .venv/Scripts/python.exe python -B tools/quantize_qwen4_flash_next_t5.py --self-test --device cuda:0

uv run --no-project --python .venv/Scripts/python.exe python -B tools/quantize_qwen4_flash_next_t5.py --model $qwenSource --output $qwenOutput --ple-bits 8 --t5-fitter prefix --device cuda:0 --chunk-rows 4096 --imatrix $imatrix --imatrix-strict --allow-experimental-imatrix --clip-search --vision-bits 8 --expert-down-bits 3

uv run --no-project --python .venv/Scripts/python.exe python -B tools/quantize_qwen4_flash_next_t5.py --verify-only $qwenOutput
```

Use a fresh output directory. After an interruption, rerun the **same conversion
command** with `--resume`; the manifest must match source, converter, recipe and
environment. A changed recipe/code/chunk size needs a new directory. `$imatrix` is
the Unsloth file downloaded in step 1; omit the imatrix flags to reproduce the
historical weight-only bake, and omit `--expert-down-bits 3` for the 30.9 GiB
Q2-down variant. `--t5-fitter legacy` and `--ple-bits 2` are historical
controls only.

Keep `omlx_conversion.json`, `omlx_conversion_manifest.json`, the Git SHA and
your command/environment record with the result. Verification checks schema,
packing/layout and the SSD-offload representation, **not end-to-end accuracy**.
Hash the shards before transfer and verify those hashes on the destination.
Pin the Git revision and dependencies when comparing bakes; byte-identical
output across arbitrary devices or library versions is not promised.

#### 2b. Stock-oMLX variant: affine experts, no T5

`--expert-format affine` stores every routed expert projection as plain MLX
affine Q2 (group 128) and writes no T5 loader marker, so the result loads on
stock oMLX with Qwen4-Exp PLE SSD support and needs none of this fork's kernels
or patches. Resident weights are about 36 GiB instead of 31 GiB. In this mode
every affine tensor gets an importance-weighted range search over both edges
(`--no-clip-search` disables it), and the Unsloth llama.cpp imatrix weights the
routed experts plus every mapped projection whose GGUF input-channel order was
checked against llama.cpp's `conversion/qwen4exp.py` (`--imatrix-scope safe`,
the default). llama.cpp stores DeltaNet V heads in a tiled order, which permutes
`out_proj`'s input columns; the importer undoes that permutation for `ssm_out`,
so `out_proj` is weighted correctly. A name/shape match alone cannot detect such
a permutation, so add tensors to the safe list only after reading the GGUF
converter. `--expert-down-bits 3` raises down_proj to Q3 for about 4.7 GiB more.
`--vision-bits 8` quantizes the ViT's Linear layers (about 0.25 GiB saved); the
vision tower cannot be dropped entirely because the runtime always builds it and
loads weights strictly.

```powershell
# $qwenSource and $imatrix come from step 1.
$qwenAffineOutput = Join-Path $env:HF_HOME 'artifacts\qwen4-affine-q2-imatrix-ple8'
uv run --no-project --python .venv/Scripts/python.exe python -B tools/quantize_qwen4_flash_next_t5.py --model $qwenSource --output $qwenAffineOutput --expert-format affine --imatrix $imatrix --imatrix-strict --vision-bits 8 --device cuda:0 --chunk-rows 4096
uv run --no-project --python .venv/Scripts/python.exe python -B tools/quantize_qwen4_flash_next_t5.py --verify-only $qwenAffineOutput
```

#### 3. Optional Q8 MTP head — research only

Skip this for the serial baseline. To reproduce the separate MTP experiment,
reuse the same original source and completed base, and create a **new** directory:

```powershell
$qwenMtpOutput = Join-Path $env:HF_HOME 'artifacts\qwen4-t5-prefix-ple8-mtp8'
uv run --no-project --python .venv/Scripts/python.exe python -B tools/add_qwen4_mtp_q8.py --source $qwenSource --base $qwenOutput --output $qwenMtpOutput --device cuda:0 --link-mode hardlink
uv run --no-project --python .venv/Scripts/python.exe python -B tools/add_qwen4_mtp_q8.py --verify-only --base $qwenOutput --output $qwenMtpOutput
uv run --no-project --python .venv/Scripts/python.exe python -B tools/add_qwen4_mtp_q8.py --audit-only --source $qwenSource --base $qwenOutput --output $qwenMtpOutput
```

This adds one original MTP layer: affine Q8/group 64 matrices and BF16 norms and
routers, sharing the target token embeddings/LM head. The extra shard is about
2.58 GiB, before runtime overhead. Hardlinks require the same compatible
filesystem (e.g. NTFS, not exFAT); use `--link-mode copy` otherwise and budget
another full base copy. **Never edit shared shards in place.** The augmented
directory has independent config/index files. Use its dedicated verifier, not
the base converter's no-MTP verifier. Reconstruction audit is not KLD or proof
that speculative decoding preserves target output.

#### 4. Serve your bake on the Mac

Transfer the **whole output directory**, not just the expert shards, to a fast
Mac SSD. The Mac needs this fork's native T5 support, not the Windows Python
environment or an AngelSlim checkout. Use a separate source checkout and `uv`
environment, full Xcode/Metal tools, and build with `OMLX_WITH_CUSTOM_KERNEL=1`.
Current dependencies include MLX 0.32.2; rebuild native extensions after dependency
changes and verify `native_kernel_status()` before loading. Stock oMLX release
install instructions below are not a substitute for this experimental build.

Keep an existing mainline installation untouched: use the isolated environment's
executable, separate server settings/base path and an unused port. Confirm PLE
SSD offload is active and the model is detected as `qwen4_exp`. Begin with MTP
off, one model/request at a time, a modest context (e.g. 8192), and memory guards
enabled. Check short multilingual outputs and a 4096-token prefill before trying
larger contexts. Record actual memory, swap, prefill and decode rates, not just
checkpoint size. Test MTP-off/on token parity before interpreting any MTP speedup.

The merged runtime defaults eager dispatch and fused HC on; fast RMS follows
upstream unconditionally. `OMLX_QWEN4_EAGER_DISPATCH=0` and
`OMLX_QWEN4_HC_FUSED=0` remain diagnostic switches. The old
`OMLX_QWEN4_FAST_RMS_NORM` switch is gone. Our separate
`OMLX_QWEN4_PLE_BATCHED_GATHER=1` experiment remains opt-in and is distinct from
enabling SSD offload itself.

This section is maintained with recipe/runtime changes; see the executable
source of truth in [the base converter](tools/quantize_qwen4_flash_next_t5.py)
and [the optional MTP converter](tools/add_qwen4_mtp_q8.py). The rest of this
README documents upstream oMLX.

---

## Upstream oMLX README

The original project's documentation and branding begin below.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="oMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">oMLX</h1>
<p align="center"><b>LLM inference, optimized for your Mac</b><br>Continuous batching and tiered KV caching, managed directly from your menu bar.</p>

<p align="center">
<a href="https://www.buymeacoffee.com/jundot"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="40"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.11--3.13-green" alt="Python 3.11-3.13">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
</p>

<p align="center">
  <a href="mailto:junkim.dot@gmail.com">junkim.dot@gmail.com</a> · <a href="https://omlx.ai/me">https://omlx.ai/me</a>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#features">Features</a> ·
  <a href="#models">Models</a> ·
  <a href="#cli-configuration">CLI Configuration</a> ·
  <a href="https://omlx.ai/benchmarks">Benchmarks</a> ·
  <a href="https://omlx.ai">oMLX.ai</a>
</p>

<p align="center">
  <b>English</b> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.ja.md">日本語</a>
</p>

---

<p align="center">
  <img src="docs/images/omlx_dashboard.png" alt="oMLX Admin Dashboard" width="800">
</p>

> *Every LLM server I tried made me choose between convenience and control. I wanted to pin everyday models in memory, auto-swap heavier ones on demand, set context limits - and manage it all from a menu bar.*
>
> *oMLX persists KV cache across a hot in-memory tier and cold SSD tier - even when context changes mid-conversation, all past context stays cached and reusable across requests, making local LLMs practical for real coding work with tools like Claude Code. That's why I built it.*

## Install

### macOS App

Download the `.dmg` from [Releases](https://github.com/jundot/omlx/releases), drag to Applications, done. The app includes in-app auto-update, so future upgrades are just one click. The macOS app also installs a lightweight `~/.omlx/bin/omlx` CLI shim so terminal commands and Apple Shortcuts can control the app-managed server.

### Homebrew

```bash
brew tap jundot/omlx https://github.com/jundot/omlx
brew install jundot/omlx/omlx

# Upgrade to the latest version
brew update && brew upgrade omlx

# Run as a background service (auto-restarts on crash)
omlx start

# Optional: MCP (Model Context Protocol) support
/opt/homebrew/opt/omlx/libexec/bin/pip install mcp
```

Optional GLM-5.2 / MiniMax M3 native custom kernels currently require a HEAD build:

```bash
brew install jundot/omlx/omlx --HEAD --with-custom-kernel
```

### From Source

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e .          # Core only
pip install -e ".[mcp]"   # With MCP (Model Context Protocol) support

# GLM-5.2 / MiniMax M3 / Qwen3.5 native custom kernels (strongly recommended
# if you serve those families -- see note below)
OMLX_WITH_CUSTOM_KERNEL=1 pip install -e .
```

Requires macOS 15.0+ (Sequoia), Python 3.11–3.13, and Apple Silicon (M1/M2/M3/M4/M5).

> **Note on native custom kernels:** a plain `pip install -e .` does NOT build
> them, and the affected model families then silently fall back to much slower
> generic paths -- for GLM-5.2 the fused DSA prefill is roughly 30x faster with
> the kernels (measured 845 vs ~29 tok/s on an M3 Ultra), and the fallback also
> uses more memory (#2137). Building them requires the Metal toolchain, which
> Command Line Tools alone do not provide (`xcrun: error: unable to find utility
> "metal"`): install full Xcode, or use the official DMG which ships the kernels
> precompiled. Homebrew can build them with `brew install jundot/omlx/omlx --HEAD
> --with-custom-kernel`, but that build also needs full Xcode. To verify your
> install:
>
> ```bash
> python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"
> ```

## Quickstart

### macOS App

Launch oMLX from your Applications folder. The Welcome screen guides you through three steps - model directory, server start, and first model download. That's it. To connect OpenClaw, OpenCode, Codex, Hermes Agent, or Copilot, see [Integrations](#integrations).

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.36.32.png" alt="oMLX Welcome Screen" width="360">
  <img src="docs/images/Screenshot 2026-02-10 at 00.34.30.png" alt="oMLX Menubar" width="240">
</p>

### CLI

```bash
# Managed background server (macOS app or Homebrew install)
omlx start
omlx stop
omlx restart

# Foreground server attached to this terminal
omlx serve --model-dir ~/models
```

The server discovers LLMs, VLMs, embedding models, and rerankers from subdirectories automatically. Any OpenAI-compatible client can connect to `http://localhost:8000/v1`. A built-in chat UI is also available at `http://localhost:8000/admin/chat`.

### Homebrew Service

If you installed via Homebrew, you can run oMLX as a managed background service:

```bash
omlx start                    # Start via brew services
omlx stop                     # Stop
omlx restart                  # Restart

brew services start omlx    # Start (auto-restarts on crash)
brew services stop omlx     # Stop
brew services restart omlx  # Restart
brew services info omlx     # Check status
```

The service runs `omlx serve` with zero-config defaults (`~/.omlx/models`, port 8000). `omlx start`, `omlx stop`, and `omlx restart` are the portable lifecycle commands; Homebrew installs delegate them to `brew services`. To customize, either set environment variables (`OMLX_MODEL_DIR`, `OMLX_PORT`, etc.) or run `omlx serve --model-dir /your/path` once to persist settings to `~/.omlx/settings.json`.

Logs are written to two locations:
- **Service log**: `$(brew --prefix)/var/log/omlx.log` (stdout/stderr)
- **Server log**: `~/.omlx/logs/server.log` (structured application log)

## Features

Supports text LLMs, vision-language models (VLM), OCR models, embeddings, and rerankers on Apple Silicon.

### Admin Dashboard

Web UI at `/admin` for real-time monitoring, model management, chat, benchmark, and per-model settings. Supports English, Korean, Japanese, Chinese, French, Russian, Spanish, and Brazilian Portuguese. All CDN dependencies are vendored for fully offline operation.

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.45.34.png" alt="oMLX Admin Dashboard" width="720">
</p>

### Experimental Multi-Mac Inference

Source builds can split one downloaded language model across unequal-memory Macs
using MLX pipeline ranks over Ring or Thunderbolt RDMA/JACCL. The Cluster
dashboard handles read-only peer discovery, strict SSH/runtime verification,
byte-aware unequal shard planning, measured compute/link rebalancing,
headroom-aware execution tuning, activation, and a live shard/performance map
on both Macs. Interactive, balanced, and throughput profiles expose coalesced
batching, prompt-cache affinity, rotating-KV limits, Ring connection tuning,
and a capability-gated experimental token-only output path. See
[Distributed inference across Macs](docs/distributed-cluster.md) for setup,
security boundaries, current limitations, and the physical-hardware validation
checklist.

### Vision-Language Models

Run VLMs with the same continuous batching and tiered KV cache stack as text LLMs. Supports multi-image chat, base64/URL/file image inputs, and tool calling with vision context. OCR models (DeepSeek-OCR, DOTS-OCR, GLM-OCR) are auto-detected with optimized prompts.

### Tiered KV Cache (Hot + Cold)

Block-based KV cache management inspired by vLLM, with prefix sharing and Copy-on-Write. The cache operates across two tiers:

- **Hot tier (RAM)**: Frequently accessed blocks stay in memory for fast access.
- **Cold tier (SSD)**: When the hot cache fills up, blocks are offloaded to SSD in safetensors format. On the next request with a matching prefix, they're restored from disk instead of recomputed from scratch - even after a server restart.

<p align="center">
  <img src="docs/images/omlx_hot_cold_cache.png" alt="oMLX Hot & Cold Cache" width="720">
</p>

### Continuous Batching

Handles concurrent requests through mlx-lm's BatchGenerator. Max concurrent requests is configurable via CLI or admin panel.

### Claude Code Optimization

Context scaling support for running smaller context models with Claude Code. Scales reported token counts so that auto-compact triggers at the right timing, and SSE keep-alive prevents read timeouts during long prefill.

### Multi-Model Serving

Load LLMs, VLMs, embedding models, and rerankers within the same server. Models are managed through a combination of automatic and manual controls:

- **LRU eviction**: Least-recently-used models are evicted automatically when memory runs low.
- **Manual load/unload**: Interactive status badges in the admin panel let you load or unload models on demand.
- **Model pinning**: Pin frequently used models to keep them always loaded.
- **Per-model TTL**: Set an idle timeout per model to auto-unload after a period of inactivity.
- **Process memory enforcement**: Total memory limit (default: system RAM - 8GB) prevents system-wide OOM.

### Per-Model Settings

Configure sampling parameters, chat template kwargs, TTL, model alias, model type override, and more per model directly from the admin panel. Changes apply immediately without server restart.

- **Model alias**: set a custom API-visible name. `/v1/models` returns the alias, and requests accept both the alias and directory name.
- **Model type override**: manually set a model as LLM or VLM regardless of auto-detection.
- **Profiles**: save named bundles of per-model settings and switch between them from the admin panel. A profile can optionally be exposed as its own model: `/v1/models` then also lists `<model>:<profile>` (e.g. `qwen3-8b:thinking`), which serves on the same engine as the base model with the profile's settings overlaid per request — no extra memory, no reload. When the base model has an alias, the exposed ID is advertised as `<alias>:<profile>`; the directory-name form keeps working, just like for the base model.

<p align="center">
  <img src="docs/images/omlx_ChatTemplateKwargs.png" alt="oMLX Chat Template Kwargs" width="480">
</p>

### Built-in Chat

Chat directly with any loaded model from the admin panel. Supports conversation history, model switching, dark mode, reasoning model output, and image upload for VLM/OCR models.

<p align="center">
  <img src="docs/images/ScreenShot_2026-03-14_104350_610.png" alt="oMLX Chat" width="720">
</p>


### Model Downloader

Search and download MLX models from HuggingFace directly in the admin dashboard. Browse model cards, check file sizes, and download with one click.

<p align="center">
  <img src="docs/images/downloader_omlx.png" alt="oMLX Model Downloader" width="720">
</p>

### Integrations

Set up OpenClaw, OpenCode, Codex, Hermes Agent, Copilot, and Pi directly from the admin dashboard with a single click. No manual config editing required.

<p align="center">
  <img src="docs/images/omlx_integrations.png" alt="oMLX Integrations" width="720">
</p>

### Performance Benchmark

One-click benchmarking from the admin panel. Measures prefill (PP) and text generation (TG) tokens per second, with partial prefix cache hit testing for realistic performance numbers.

<p align="center">
  <img src="docs/images/benchmark_omlx.png" alt="oMLX Benchmark Tool" width="720">
</p>

### macOS Menubar App

Native Swift / SwiftUI menubar app (not Electron). Start, stop, and monitor the server without opening a terminal. Includes persistent serving stats (survives restarts), auto-restart on crash, and built-in auto-update.

<p align="center">
  <img src="docs/images/Screenshot 2026-02-10 at 00.51.54.png" alt="oMLX Menubar Stats" width="400">
</p>

### API Compatibility

Drop-in replacement for OpenAI and Anthropic APIs. Supports streaming usage stats (`stream_options.include_usage`), Anthropic adaptive thinking, and vision inputs (base64, URL).

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming) |
| `POST /v1/completions` | Text completions (streaming) |
| `POST /v1/messages` | Anthropic Messages API |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/rerank` | Document reranking |
| `GET /v1/models` | List available models |

### Tool Calling & Structured Output

Supports all function calling formats available in mlx-lm, JSON schema validation, and MCP tool integration. Tool calling requires the model's chat template to support the `tools` parameter. The following model families are auto-detected via mlx-lm's built-in tool parsers:

| Model Family | Format |
|---|---|
| Llama, Qwen, DeepSeek, etc. | JSON `<tool_call>` |
| Qwen3.5 Series | XML `<function=...>` |
| Gemma | `<start_function_call>` |
| GLM (4.7, 5) | `<arg_key>/<arg_value>` XML |
| MiniMax | Namespaced `<minimax:tool_call>` |
| Mistral | `[TOOL_CALLS]` |
| Kimi K2 | `<\|tool_calls_section_begin\|>` |
| Longcat | `<longcat_tool_call>` |

Models not listed above may still work if their chat template accepts `tools` and their output uses a recognized `<tool_call>` XML format. For tool-enabled streaming, assistant text is emitted incrementally while known tool-call control markup is suppressed from visible content; structured tool calls are emitted after parsing the completed turn.

## Models

Point `--model-dir` at a directory containing MLX-format model subdirectories. Two-level organization folders (e.g., `mlx-community/model-name/`) are also supported.

```
~/models/
├── Step-3.5-Flash-8bit/
├── Qwen3-Coder-Next-8bit/
├── gpt-oss-120b-MXFP4-Q8/
├── Qwen3.5-122B-A10B-4bit/
└── bge-m3/
```

Models are auto-detected by type. You can also download models directly from the admin dashboard.

| Type | Models |
|------|--------|
| LLM | Any model supported by [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| VLM | Qwen3.5 Series, GLM-4V, Pixtral, and other [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) models |
| OCR | DeepSeek-OCR, DOTS-OCR, GLM-OCR |
| Embedding | BERT, BGE-M3, ModernBERT |
| Reranker | ModernBERT, XLM-RoBERTa |

## CLI Configuration

```bash
# Managed background server (macOS app or Homebrew install)
omlx start
omlx stop
omlx restart

# Start with default settings (memory guard tier = balanced, manage via admin UI)
omlx serve --model-dir ~/models

# Choose a memory guard tier at startup
omlx serve --model-dir ~/models --memory-guard safe

# Set a custom memory guard ceiling in GB
omlx serve --model-dir ~/models --memory-guard-gb 48

# Enable SSD cache for KV blocks
omlx serve --model-dir ~/models --paged-ssd-cache-dir ~/.omlx/cache

# Set in-memory hot cache size
omlx serve --model-dir ~/models --hot-cache-max-size 20%

# Adjust max concurrent requests (default: 8)
omlx serve --model-dir ~/models --max-concurrent-requests 16

# With MCP tools
omlx serve --model-dir ~/models --mcp-config mcp.json

# HuggingFace mirror endpoint (for restricted regions)
omlx serve --model-dir ~/models --hf-endpoint https://hf-mirror.com

# API key authentication
omlx serve --model-dir ~/models --api-key your-secret-key
# Localhost-only: skip verification via admin panel global settings
```

All settings can also be configured from the web admin panel at `/admin`. Settings are persisted to `~/.omlx/settings.json`, and CLI flags take precedence.

<details>
<summary>Architecture</summary>

```
FastAPI Server (OpenAI / Anthropic API)
    │
    ├── EnginePool (multi-model, LRU eviction, TTL, manual load/unload)
    │   ├── BatchedEngine (LLMs, continuous batching)
    │   ├── VLMEngine (vision-language models)
    │   ├── EmbeddingEngine
    │   └── RerankerEngine
    │
    ├── ProcessMemoryEnforcer (total memory limit, TTL checks)
    │
    ├── Scheduler (FCFS, configurable concurrency)
    │   └── mlx-lm BatchGenerator
    │
    └── Cache Stack
        ├── PagedCacheManager (GPU, block-based, CoW, prefix sharing)
        ├── Hot Cache (in-memory tier, write-back)
        └── PagedSSDCacheManager (SSD cold tier, safetensors format)
```

</details>

## Development

### CLI Server

```bash
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e ".[dev]"
pytest -m "not slow"
```

### macOS App

The native SwiftUI app lives at `apps/omlx-mac/`. Requires Xcode 26.5+ and Python 3.11+. venvstacks is declared as a dev dependency so `pip install -e ".[dev]"` (or `uv sync --dev`) brings the pinned version in. The build script also falls back to `uvx venvstacks` or `pipx run venvstacks` if you prefer a host-global tool runner.

```bash
# Stage a runnable oMLX.app (xcodebuild + venvstacks Python layers + ad-hoc sign)
apps/omlx-mac/Scripts/build.sh release

# Result lands at apps/omlx-mac/build/Stage/oMLX.app
open apps/omlx-mac/build/Stage/oMLX.app

# Force a fresh venvstacks rebuild (otherwise it's cached by fingerprint)
apps/omlx-mac/Scripts/build.sh release --rebuild-donor

# Stage with optional GLM-5.2 / MiniMax M3 native custom kernels
apps/omlx-mac/Scripts/build.sh release --with-custom-kernel
```

First cold build takes 10–20 minutes (venvstacks Python layer assembly). Subsequent builds reuse the cached `packaging/_export/` and finish in about 4 minutes. See [packaging/README.md](packaging/README.md) for the layer configuration and [apps/omlx-mac/](apps/omlx-mac/) for the Swift sources.

## Contributing

Contributions are welcome! See [Contributing Guide](docs/CONTRIBUTING.md) for details.

- Bug fixes and improvements
- Performance optimizations
- Documentation improvements

## License

[Apache 2.0](LICENSE)

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language model inference on Apple Silicon
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) - oMLX started from vllm-mlx v0.1.0 and evolved significantly with multi-model serving, tiered KV caching, VLM with full paged cache support, an admin panel, and a macOS menu bar app
- [venvstacks](https://venvstacks.lmstudio.ai) - Portable Python environment layering for the macOS app bundle
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Embedding model support for Apple Silicon
- [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) - Block diffusion speculative decoding on Apple Silicon
- [MTPLX](https://github.com/youssofal/mtplx) - Lightning MTP's verify-shape Metal kernels are powered by MTPLX by Youssof Altoukhi, which also inspired the depth-k pipeline
- [mlx-serve](https://github.com/ddalcu/mlx-serve) - The fused GDN verify prework kernel is adapted from mlx-serve's port of the mlxfast-challenge qwen35_packed_gdn_prework kernel; Qwen4 QSA's 128-bit K/V staging is adapted from mlx-serve's MIT-licensed `msv_attn_p256` kernel
- [SiliconScope](https://github.com/kennss/SiliconScope) - The menu bar statistics take their design and rendering approach from SiliconScope by Kennt Kim, which also inspired the energy-efficient re-render gating
