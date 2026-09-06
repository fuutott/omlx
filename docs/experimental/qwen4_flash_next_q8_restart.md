# Qwen3.8-Flash-Next Q8 PLE restart — 2026-09-06

We are doing quantization optimizations to run Qwen3.8-Flash-Next (`qwen4_exp`)
on a 48 GB M3 Max. Its large PLE ngram table stays mmap-backed on SSD; routed
experts use experimental T5 gate/up and affine Q2 down. The first weight-only
T5 was reasonably coherent with factual errors (user report); the later
T5-imatrix was unusable. Controlled quality and sustained decode performance
were not established.

Latest update: [runtime correctness and weight-only fitter](qwen4_flash_next_t5_correctness_fitter.md).
That update supersedes the outstanding-fitter proposal below: prefix fitting
is now implemented, with legacy available as an explicit A/B control. Native
runtime verification remains pending; source download is complete.

The user removed the generated checkpoints and rolled back the later Mac
optimizations. The agreed baseline is `dc7aaee3`. Do not reapply that later work.
The original HF discussion remains the only mailbox, under the existing
per-operation user-authorization rules. This update does not authorize a fetch,
message, upload, download, or change to the mainline Mac installation.

## Implemented locally

- PLE defaults to affine Q8/group-32. `--ple-bits 2` is an explicit historical
  control. Tensor names, 128 shards, T5 loader marker, and SSD detection stay
  unchanged. The runtime already infers Q8 from packed row width.
- Non-PLE precision assignments are unchanged, including Q2 down. T5 fitting
  now defaults to guarded prefix search; use `--t5-fitter legacy` for the
  original solver. QSA
  `self_attn.o_proj` is accurately documented as Q4 rather than the previously
  claimed Q5. This is the controlled baseline, not a finalized quality recipe.
- Imatrix is parked behind `--allow-experimental-imatrix`. The known DeltaNet
  channel permutation and incomplete strict coverage are not fixed here.
- Full conversion requires a fresh output directory or a matching resume
  manifest. Identity includes converter/config/index hashes, PLE bits, T5 fitter,
  imatrix/importer hashes, source paths/sizes/mtimes, Torch version, device and chunk size. Every resumed
  shard must also have the matching embedded fingerprint. Source weight files
  are identified by stats, not full content hashes; this is an accidental-mix
  guard, not cryptographic verification of source or output payloads. Old
  unmanifested artifacts cannot be resumed. Single-shard conversion refuses
  existing output files.
- CPU gather candidate: `OMLX_QWEN4_PLE_BATCHED_GATHER=1` groups requested rows
  once, assembles packed rows/parameters in caller order on the CPU, and issues
  one dequantization batch. Uniform affine layouts only; dense/mixed layouts
  fall back. It is **off by default**, needs native equality/timing evidence,
  and does not change hashing, expert kernels, or add an unbounded row cache.
  The shared NumPy reader also removes one redundant copy of already-copied
  advanced-index rows.

Q8/group-32 PLE occupies approximately 57.6 GB / 53.64 GiB, versus Q2's
19.2 GB / 17.88 GiB. A complete Q8 checkpoint is therefore roughly 86 GiB,
before any additional recipe changes. Non-PLE stored bytes remain unchanged;
the old ~34 GiB offloaded estimate remains only a heuristic. Larger random
rows affect SSD reads and page cache, so real process pressure still matters.

## Quality work worth doing before imatrix

1. **Weight-only T5 scale search: implemented.** Guarded prefix search fixes
   the maximum-initialization local minimum while retaining the original fit
   as a candidate after BF16 rounding. Same packed format, storage and
   inference kernel. See the current update for real-weight measurements and
   the still-required end-to-end quality comparison.
2. **Low-cost sensitive-matrix precision.** QSA `o_proj` Q4→Q5 adds ~22.5 MiB
   of packed weights across 12 layers. The 97 main hyperconnection input-mix
   down matrices Q4→Q8 add ~152 MiB. These are plausible protective changes,
   but must be tested in this architecture. Keep independent recipe switches
   and record exactly which variant is converted.
3. **Expert down Q3 as a separate experiment.** Raising all 48×512 down
   matrices Q2→Q3 costs 4.6875 GiB, before runtime headroom. Do not promise it
   fits the 48 GB machine. Selective protection needs per-layer/expert evidence;
   do not copy Hy4 layer assignments onto Qwen4.
4. **Fix evaluation before making quality claims.** The old twelve prompts
   are smoke tests, with thinking-mode/budget confounds and permissive code
   checks. Use identical prompt tokens/templates, thinking mode, sampling and
   budget; distinguish truncations/errors; add executable code checks and
   teacher-forced held-out NLL or KLD when practical. Knowledge errors do not
   isolate PLE from expert damage or runtime defects.

Items 2–4 remain outstanding; only the fitter change is implemented here.
Do not launch a full reconversion merely because the converter now defaults to
Q8. Settle fitting/precision variants first. Download the original BF16 source
again into explicit `HF_HOME` when authorized; do not requantize an old low-bit
checkpoint. Windows `HF_HOME` is `D:\hf_models_cache`; `D:\models` is GGUF-only.

## Windows validation

Use the existing repository-local `.venv` and `uv`; never global Python:

```powershell
$env:HF_HOME='D:\hf_models_cache'
uv --cache-dir ..\.uv-cache run --no-project --offline --python .venv\Scripts\python.exe python -B -m unittest discover -s tools/tests -v
uv --cache-dir ..\.uv-cache run --no-project --offline --python .venv\Scripts\python.exe python -B tools/quantize_qwen4_flash_next_t5.py --self-test --device cuda:0
```

Portable tests cover actual Q8/Q2 Torch packing, config/shape consistency,
resume rejection, and mmap row assembly with uneven shards, repeated ids and
companions in separate files. They do not execute MLX or Metal.

Initial Q8-only result: eight portable tests passed, including the real SSD residency
detector on Q8 output; CUDA self-test passed; Python AST checks and
`git diff --check` passed. The focused pytest invocation could not run in the
Windows converter venv (pytest is not installed), and MLX/Metal numerical tests
require the Mac regardless. No native-runtime speedup is claimed. The newer
correctness/fitter update records 18 passing portable/CUDA tests and the
real-weight comparison; see its validation section for current results.

## Mac validation without touching mainline

Use a separate checkout/worktree of this experimental branch and its own
`.venv`. Use the commit carrying this test bundle; `dc7aaee3` alone does not
contain it. Keep the mainline app, venv, settings, and model
directories untouched. No server is needed for the following tests.

No synthetic weights need to be transferred. The tests create seeded arrays
on the Mac, and `--synthetic` writes its own tiny temporary PLE checkpoint,
then removes that temporary fixture when finished. The Mac does not need F:,
the original source download, CUDA, or PyTorch for this stage. Only code,
the isolated Mac dependencies, and a writable results directory are needed.

```bash
# Inside the separate experimental checkout, with HF_HOME explicitly set:
uv venv --python 3.12 .venv
OMLX_WITH_CUSTOM_KERNEL=1 uv pip install --python .venv/bin/python -e . pytest
uv run --no-project --python .venv/bin/python python -m pytest -q \
  tests/test_bonsai_t5_load.py tests/test_qwen35_moe_gate_up.py \
  tests/test_bonsai_qmv.py tests/test_qwen4_t5_routed_numerics.py \
  tests/test_mlx_vlm_qwen4_exp_compat.py tests/test_qwen4_ple_batched.py \
  tests/test_qwen4_exp_residency.py

uv run --no-project --python .venv/bin/python python -m tools.qwen4_ple_bench \
  --synthetic --bits 8 --output /absolute/path/to/results/ple-q8-forward.json
uv run --no-project --python .venv/bin/python python -m tools.qwen4_ple_bench \
  --synthetic --bits 8 --reverse --output /absolute/path/to/results/ple-q8-reverse.json
```

A native-kernel test skip is not a pass. PLE tests require exact equality for
Q2/Q8 row output; routed expert tests compare rounded FP16/BF16 inputs/scales
against FP32 reference products, including top-10 input reuse, expanded
routes, multi-group scales, sorted prefill and actual fused gate/up dimensions.

When a real converted checkpoint is available, replace `--synthetic --bits 8`
with `--model /absolute/path/to/checkpoint`. The tool only mmaps PLE tables;
it does not load experts or start/stop a server. Output is create-only JSON.

The benchmark measures warm/reused random rows, not real sustained decode or
cold SSD throughput. It records CPU and synchronized wall times, page faults,
an instrumented row-read/index-eval/completion-wait breakdown, and version/git
identity. Requested row bytes are not physical SSD bytes. Inputs are already
evaluated, so hash/index synchronization in the full model still needs a trace.

For full-model testing, keep the original gather path for the quality A/B;
then enable the candidate only in the experimental server process and measure
fixed-length prefill and sustained decode separately. Record peak process and
Metal memory, swap/page pressure, GPU activity and SSD reads. Python sorted
expert routing applies at 64+ routes (7+ tokens at top-10), so optimizing that
loop alone does not explain or fix single-stream decode speed. Do not infer
compute-bound behavior or predicted TPS just from M3 Max's 400 GB/s headline.
