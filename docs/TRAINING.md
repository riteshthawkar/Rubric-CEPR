# Training and inference

This page describes the fixed extraction recipe. The separate recovered
149-pair analysis recipe and guarded commands are in
[VERIFIER_BANK.md](VERIFIER_BANK.md).

## Prepare inputs

The extraction recipe uses:

- `Qwen/Qwen-Image-Edit-2509`, revision
  `d3968ef930e841f4c73640fb8afa3b306a78167e`.
- The exact 64-row manifest in `reproducibility/manifests/extraction_v1.json`.
- All 128 source and target images listed in
  `reproducibility/manifests/extraction_v1_artifacts.json`.
- The recorded environment in `configs/reproduction/extraction_v1.yaml`.

Cache the pinned base model and install the separate input bundle before
allocating a GPU. Training uses `--local_files_only`. The installer verifies the
bundle hash and extracts only the listed images, rejecting duplicate entries,
links and unsafe paths. Its destination must be new.

```bash
hf download Qwen/Qwen-Image-Edit-2509 \
  --revision d3968ef930e841f4c73640fb8afa3b306a78167e
rubric-cepr install-artifacts \
  --bundle /path/to/training-inputs.tar.gz --data-root /path/to/training-data
rubric-cepr check \
  --data-root /path/to/training-data \
  --benchmark-json data/processed/benchmark/imgedit/basic_edit.json \
  --benchmark-images data/processed/benchmark/imgedit/original_images \
  --strict-environment
```

The overlap check requires all 737 ImgEdit records and their source images. It
rejects shared source basenames or image bytes. Missing inputs do not disable it.

## Slurm execution

Follow your cluster's allocation policy. The CLI checks an owned RUNNING GPU
allocation, a numeric `srun` step, assigned GPU IDs, CUDA visibility, the compute
host, tmux context and `PYTHONPATH`. Keep Slurm's GPU visibility variables.
Activate the model environment inside the allocation and set `HF_HOME` before
running a model command. Prepare data and environments before reserving a GPU.

`scripts/slurm/train_v1.sbatch` is a single-GPU example with 4 CPUs, 96 GiB of
host memory and a 24-hour limit. Adjust scheduler resources to your cluster's
rules. From an inspected tmux session, set:

```bash
export RUBRIC_ROOT=/absolute/path/to/Rubric-CEPR
export RUBRIC_ENV_PREFIX=/absolute/path/to/model-environment
export RUBRIC_ACTIVATE_SCRIPT="$RUBRIC_ROOT/scripts/slurm/activate.example.sh"
export HF_HOME=/absolute/path/to/huggingface-cache
```

`RUBRIC_ENV_PREFIX` is a conda environment prefix; `CONDA_EXE` can select the
conda executable. The worker activates the environment inside a numeric step
and sets `PYTHONPATH` to the checkout's `src` directory.

Inspect the exact training command on CPU:

```bash
rubric-cepr train --dry-run \
  --data-root /path/to/training-data --output /path/to/new-training-output
```

After preflight passes, submit the example job:

```bash
sbatch --export=ALL scripts/slurm/train_v1.sbatch \
  --data-root /path/to/training-data --output /path/to/new-training-output
```

The launcher fixes the extraction recipe's scientific parameters and refuses
an existing output directory. The allocation ends when the workload exits.

## Completion checks

The launcher validates exactly 400 requested and completed optimizer steps,
world size one, seed 123, no resumed checkpoint and a final 960-tensor rank-16
adapter. It reads adapter headers without loading the base model. Keep the
training command, preflight, completion receipt and final adapter together.

```bash
rubric-cepr check --training-output /path/to/new-training-output
```

A retrained adapter may differ numerically from the reference adapter. Record
its actual hash and evaluate that artifact separately.

## Inference

```bash
rubric-cepr infer \
  --image /path/to/source.jpg --prompt 'Extract the dog.' \
  --checkpoint /path/to/pytorch_lora_weights.safetensors \
  --output /path/to/new-edit.png
```

Inference uses the pinned model revision, seed 42, 40 steps, true CFG 4, guidance
scale one and a space negative prompt. The default checks the reference adapter
hash in `reproducibility/artifacts.json`. Add `--allow-reconstructed-checkpoint`
for retrained weights, or `--dry-run` to inspect settings on CPU.

## Reference framework

Prepare a source manifest as described in [METHOD.md](METHOD.md), then inspect
its launch command:

```bash
rubric-cepr framework --dry-run
```

Running without `--dry-run` requires the GPU step. The supplied reference config
uses its own proposal and verification settings and emits training commands by
default; it is separate from the fixed 64-pair extraction training recipe.
