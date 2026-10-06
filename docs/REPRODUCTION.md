# Reconstructing the surviving v1 adapter

## Inputs

- Base: `Qwen/Qwen-Image-Edit-2509`, revision
  `d3968ef930e841f4c73640fb8afa3b306a78167e`.
- Exact 64-row manifest: `reproducibility/manifests/paper_headline_extract_v1.json`.
- Separate image inventory: all 128 source/target references, with byte counts
  and SHA-256. The installer extracts only these images, never arbitrary bundle
  source files or links.
- Historical adapter SHA-256:
  `118ffe5d4a86d33a836eb07c4ce063fad7096988072067c48f86636380047942`.
- Environment: Python 3.11.15 and the exact package versions recorded in
  `configs/reproduction/paper_headline_extract_v1.yaml`.

Download/cache the pinned base model before allocating a GPU. The launcher uses
`--local_files_only`. Resolve dataset and environment prerequisites before
submitting a GPU job. The base model is not included in this release.

```bash
hf download Qwen/Qwen-Image-Edit-2509 \
  --revision d3968ef930e841f4c73640fb8afa3b306a78167e
python scripts/accv_v1.py install-artifacts \
  --bundle /path/to/paper_headline_extract_v1.tar.gz \
  --data-root /path/to/new-v1-data
python scripts/accv_v1.py check \
  --data-root /path/to/new-v1-data \
  --benchmark-json data/processed/benchmark/imgedit/basic_edit.json \
  --benchmark-images data/processed/benchmark/imgedit/original_images \
  --strict-environment
```

The overlap check requires all 737 benchmark records and source images. It
rejects basename or byte-content overlap with training sources. Missing data
does not silently disable this check.

## Allocation and execution

For this workspace, follow the parent `docs/AGENT_GPU_ALLOCATION_AND_EXPERIMENT_RULES.md`:
inspect a new tmux session, submit through `sbatch`, activate its existing
`qedit` environment **inside** the allocation, and run the command through
`srun` so that `SLURM_STEP_ID` is numeric. Set `HF_HOME` to the parent policy's
shared cache and `PYTHONPATH` to this release copy's `src`. No release command
reserves an idle GPU. A scheduler time limit ends when the workload exits.

On another Slurm cluster, supply that cluster's environment and cache paths.
The guard verifies the current user, RUNNING allocation, assigned GPU, real
numeric step, CUDA visibility and tmux context. Preserve Slurm's GPU visibility
variables. Do not force a physical GPU index.

A ready single-GPU sbatch template is `scripts/slurm/train_v1.sbatch` (4 CPUs,
96 GiB, maximum 24 hours). Set `ACCV_RELEASE_ROOT` to this checkout,
`ACCV_ENV` to the environment prefix and `ACCV_ACTIVATE_SCRIPT` to the absolute
path of `scripts/slurm/activate.example.sh`; `CONDA_EXE` may supply the conda
binary. The worker activates that environment inside its numeric `srun` step.
After the CPU preflight passes, submit from the inspected tmux session:

```bash
sbatch --export=ALL scripts/slurm/train_v1.sbatch \
  --data-root /path/to/new-v1-data --output /path/to/new-training-output
```

No job is submitted automatically by release preparation.

From the activated environment, inspect the training command without running it:

```bash
python scripts/accv_v1.py train --dry-run \
  --data-root /path/to/new-v1-data \
  --output /path/to/new-training-output
```

`--dry-run` prints the plan only; it does not claim that external inputs exist.
Run the same command without `--dry-run` inside the verified GPU step. The
release accepts no hyperparameter overrides for the historical recipe. Output
directories must be new. A reconstructed adapter may differ from the historic
adapter because of numerical execution differences; label it reconstructed
rather than substituting it for the historical bytes.

After training, the launcher validates the receipt: complete status, exactly
400 requested and completed steps, world size one, seed 123, no resumed
checkpoint, and a final 960-tensor rank-16 adapter. The frozen trainer preserves
its original save behavior. CPU receipt checks inspect file headers without
loading the base model. Keep `training_command.json`, `release_preflight.json`,
`training_completion.json` and `release_completion.json` with the checkpoint.

```bash
python scripts/accv_v1.py check --training-output /path/to/new-training-output
python scripts/accv_v1.py infer \
  --image /path/to/source.jpg --prompt 'Extract the dog.' \
  --checkpoint /path/to/pytorch_lora_weights.safetensors \
  --output /path/to/new-edit.png
```

For reconstructed weights, add `--allow-reconstructed-checkpoint` and report
their actual hash. Inference uses seed 42, 40 steps, true CFG 4, guidance scale
one and a space negative prompt. The release's inference loader supplies the
same explicit official processor components as the training shim; this
compatibility change is listed in the release provenance.
