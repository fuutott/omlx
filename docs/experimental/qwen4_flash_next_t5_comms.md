# Qwen4 Flash Next validation communications

Use one Hugging Face discussion on the private checkpoint repository as the
canonical Windows-to-Mac status log. GitHub remains the source of code; the HF
discussion is only for coordination, validation results, and short log excerpts.

Repository: `fuutott/Qwen3.8-Flash-Next-MLX-t5`

Discussion: `https://huggingface.co/fuutott/Qwen3.8-Flash-Next-MLX-t5/discussions/1`

Do not open a parallel GitHub issue, PR comment thread, or second HF discussion
for this validation. Reference GitHub commits inside this one HF thread.

Before starting a phase, read the thread. Post again when the phase passes,
fails, or blocks. Use this first line so updates are easy to scan:

```text
[WIN|MAC] [START|PASS|FAIL|BLOCKED|DONE] <UTC timestamp> git=<sha> hf=<revision-or-pending> phase=<name>
```

Then include only the facts needed by the other machine: command or test,
result, relevant metrics, and a short error/log excerpt. Never post access
tokens, credentials, private local paths that reveal secrets, or huge logs.

Required milestones are `upload`, `download-verify`, `bonsai-build`,
`focused-tests`, `one-token`, `coherence-128`, `prefill-2k`, `prefill-8k`, and
`final-summary`. Only one agent owns a milestone at a time. A `START` claims it;
the matching `PASS`, `FAIL`, or `BLOCKED` releases it. Windows owns publication
and checkpoint integrity. Mac owns Metal build and runtime validation.

CLI usage:

```bash
export HF_HOME=/absolute/ssd/path/to/hf-cache
hf discussions info fuutott/Qwen3.8-Flash-Next-MLX-t5 1
hf discussions comment fuutott/Qwen3.8-Flash-Next-MLX-t5 1 \
  --body "[MAC] [START] 2026-08-31T12:00:00Z git=<sha> hf=<revision> phase=bonsai-build"
```

The Mac must not begin `download-verify` until Windows posts `upload` as
`PASS` with the immutable HF revision. The final Mac message must include the
exact Git and HF revisions, macOS/Xcode/MLX versions, native kernel status,
peak memory, SSD behavior, prefill/decode rates, and prompt/output evidence.
