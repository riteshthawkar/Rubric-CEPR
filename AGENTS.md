# ACCV v1 release copy

This is a separate editable release candidate. Do not modify the parent legacy
source, its archive, manifest, historical artifacts, or current research jobs.
Keep the original trainer, safety helpers, historical contract and manifests
byte-identical to the hashes in `provenance/IDENTITY.json`.

Use `scripts/accv_v1.py` for all model and scoring commands. In this workspace,
read the parent `docs/AGENT_GPU_ALLOCATION_AND_EXPERIMENT_RULES.md` in full before
any workload, and obey its tmux, owned RUNNING Slurm GPU step, qedit environment,
HF_HOME and source-path requirements. User authorization to use sbatch takes
precedence over the policy's preference for interactive allocation. CPU checks,
artifact installation, source packaging and metadata reads need no GPU.

Do not use CodeRabbit. Do not publish or push this candidate without explicit
user authorization. Keep historical extraction results separate from the
reference framework and any new reruns. Never introduce external reward or
benchmark judges into training while describing the method as internal-only.
