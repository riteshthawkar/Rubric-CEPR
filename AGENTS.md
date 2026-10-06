# Contributor instructions

Keep the trainer, safety helpers and training rows byte-identical to their pins
in `reproducibility/artifacts.json`. Configuration or metadata changes must
preserve scientific parameters and update the corresponding checksums.

Use `scripts/rubric_cepr.py` for model and scoring commands. Follow the hosting
workspace's GPU allocation policy before any model workload. Model commands
require an owned RUNNING Slurm GPU allocation with a numeric step, an inspected
tmux session, the activated model environment and this checkout's source path.
CPU checks, artifact installation and source packaging need no GPU.

Do not use CodeRabbit. Keep checkpoint-specific results separate from results
of the reference framework or newly trained adapters. External benchmark judges
are evaluation-only. Describe pretrained external components explicitly when a
pair miner uses them.

Run CPU tests and `tools/verify_release.py` after changes. Regenerate
`SHA256SUMS` with `tools/verify_release.py --write-checksums` when the source
payload changes. Credentials, weights, datasets and generated outputs must not
be committed. Do not modify neighboring projects or publish changes without
user authorization.
