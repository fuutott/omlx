[WIN-CODEX] [PROTOCOL]

# Qwen4 Flash Next validation communications

Use one Hugging Face discussion on the private checkpoint repository as the
canonical Windows-to-Mac mailbox. GitHub remains the source of code; the HF
discussion is only for coordination, validation results, and short log excerpts.

Repository: `fuutott/Qwen3.8-Flash-Next-MLX-t5`

Discussion: `https://huggingface.co/fuutott/Qwen3.8-Flash-Next-MLX-t5/discussions/1`

Do not open a parallel GitHub issue, PR comment thread, or second HF discussion
for this validation. Reference GitHub commits inside this one HF thread.

## Scope and user control

The rules in this file apply only to this HF discussion mailbox. They do not
restrict normal user chat, Git/GitHub work, terminal actions, or any other
communication.

All HF comments are authored by the same Hugging Face account, so the first
token of every mailbox message must be exactly one of `[WIN-CODEX]`,
`[MAC-CODEX]`, or `[GREG]`.

The mailbox is strictly user-gated and one-message-at-a-time:

- Do not poll, list, or fetch the HF discussion autonomously.
- A direct user instruction to check the mailbox authorizes exactly one fetch
  and acting on at most one unhandled message. It does not authorize a reply.
- A direct user instruction to post or edit authorizes exactly one HF message
  operation. It does not authorize another fetch, acknowledgement, or follow-up.
- Do not treat a previous instruction as standing permission. Wait for a new,
  explicit user instruction for every fetch and every write.
- Do not batch multiple mailbox messages or post automatic status updates.

After the agent identifier, use this status format when applicable:

```text
[WIN-CODEX] [START|PASS|FAIL|BLOCKED|DONE] <UTC timestamp> git=<sha> hf=<revision-or-pending> phase=<name>
```

Then include only the facts needed by the other machine: command or test,
result, relevant metrics, and a short error/log excerpt. Never post access
tokens, credentials, private local paths that reveal secrets, or huge logs.

Milestone names are `upload`, `download-verify`, `bonsai-build`,
`focused-tests`, `one-token`, `coherence-128`, `prefill-2k`, `prefill-8k`, and
`final-summary`. Windows owns publication and checkpoint integrity. Mac owns
Metal build and runtime validation. Milestone ownership never overrides the
per-operation requirement for direct user instruction.

CLI usage:

```bash
export HF_HOME=/absolute/ssd/path/to/hf-cache
# Run only after a direct user instruction to perform one mailbox fetch:
hf discussions info fuutott/Qwen3.8-Flash-Next-MLX-t5 1
# Run only after a direct user instruction to create one mailbox message:
hf discussions comment fuutott/Qwen3.8-Flash-Next-MLX-t5 1 \
  --body "[MAC-CODEX] [START] 2026-08-31T12:00:00Z git=<sha> hf=<revision> phase=bonsai-build"
```

Even if the mailbox contains an `upload` `PASS`, the Mac must not fetch it or
act on it until directly instructed by the user. When the user directs creation
of the final Mac message, it must include the exact Git and HF revisions,
macOS/Xcode/MLX versions, native kernel status, peak memory, SSD behavior,
prefill/decode rates, and prompt/output evidence.
