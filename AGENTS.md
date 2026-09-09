# Qwen3.8-Flash-Next experimental fork

Read `docs/experimental/qwen4_flash_next_t5_handoff.md` before changing or
testing the Qwen3.8-Flash-Next work in this fork.

This model is `qwen4_exp`, not Qwen3.5. Preserve its Qwen4 experimental
architecture, PLE n-gram embedding layout, DeltaNet layers, hyper-connections,
and SSD-mmap path. Do not generalize a fix from Qwen3.5 unless the Qwen4 tensor
names, shapes, and execution path have been verified independently.

Use `uv` with a repository-local `.venv` for Python work. On macOS, build the
source checkout with `OMLX_WITH_CUSTOM_KERNEL=1`; this experiment requires the
native Bonsai Metal extension. Do not install into the global Python
environment.

Never add model weights, Hugging Face credentials, `.model-research`, or local
cache directories to Git. Keep Hugging Face downloads under an explicit
`HF_HOME`. The converted checkpoint belongs in a Hugging Face model repository,
not this Git repository.

## Public recipe maintenance

Keep README.md's "This fork: Qwen Flash Next on a 48 GB Mac" and "How to bake
the cake" sections current in the same change whenever conversion policy,
fitter/defaults, source revision, dependencies, CLI flags, artifact layout,
MTP status, required runtime support or validated limitations change. Check the
commands against the actual parsers and dependency lock; update the status date
and distinguish measured results from pending validation. Keep the entire fork
overview and recipe above the oMLX logo, with a prominent unofficial-fork notice
and a clear boundary before upstream branding/documentation. Preserve upstream's
README content outside this fork-specific section. Do not add credentials or
machine-specific private paths to this public section. The original source
model ID/revision in the reproduction commands is intentional, and so is the
link to the released checkpoint. Keeping this documentation current does not
authorize background polling or downloads.

The below-q2 T5 expert format is experimental. Do not describe the model as
coherent or lossless until a real M3 Max run has completed. Record the exact
commit, checkpoint revision, macOS/MLX versions, kernel availability, peak
resident memory, prefill/decode rates, and representative generation output.

For runtime changes, run the focused Bonsai and Qwen gate/up tests as well as
syntax/static checks. Windows can validate conversion and checkpoint structure,
but native Metal compilation and numerical runtime tests must be performed on
Apple Silicon.
