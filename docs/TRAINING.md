# Internal CEPR training and inference

This page describes `main`. The fixed detector-assisted 64-pair recipe,
`install-artifacts` command and reference extraction checkpoint belong to
[groundingdino-extraction](https://github.com/riteshthawkar/Rubric-CEPR/tree/groundingdino-extraction).
The top-level README summarizes the commands below.

## Prepare sources

Install the model environment from `requirements-model.txt` and cache the Qwen
model before requesting a GPU. Use `examples/unlabeled_manifest.jsonl` to create
your own source manifest. Paths in each row are absolute or relative to this
checkout. The Planner generates instructions from these unlabeled images.

```bash
hf download Qwen/Qwen-Image-Edit-2509 \
  --revision d3968ef930e841f4c73640fb8afa3b306a78167e
rubric-cepr check --strict-environment
rubric-cepr check \
  --manifest /path/to/sources.jsonl \
  --benchmark-json data/processed/benchmark/imgedit/basic_edit.json \
  --benchmark-images data/processed/benchmark/imgedit/original_images
```

The overlap check requires all 737 ImgEdit records and their source images. It
rejects shared source basenames or image bytes. It does not decode images or
claim a perceptual-duplicate audit. Keep all other evaluation sets out of training
and record their disjointness separately.

## Inspect the workflow on CPU

```bash
rubric-cepr framework --dry-run --manifest /path/to/sources.jsonl
rubric-cepr train --dry-run \
  --manifest /path/to/sources.jsonl --output /path/to/new-run
```

`framework` uses the configuration's training triggers, which default to
`emit_only`. `train` sets both Editor and Planner triggers to `launch`, so accepted
candidates and qualifying Planner traces lead to adapter updates. `--set` can
adjust the round and training budget. Configuration validation also applies to
overrides. External detector configurations are rejected on main.

The default preset uses one round over the supplied manifest, four candidates
per proposal, 400 Editor updates at 1e-4, and 16 Planner updates at 1e-5. It
focuses proposals on subject extraction and enables the Planner feasibility
band. Accepted Editor targets use reward weights normalized over the training
manifest. The warm-start adapter and replay ratio are explicit run settings.
Set `training.current_checkpoint_path` to your warm-start adapter; set
`training.reconstruction_replay_ratio` to add identity records.

For a smaller run, use the separate example configuration:

```bash
rubric-cepr train --config configs/examples/internal_cepr_small.yaml \
  --manifest /path/to/sources.jsonl --output /path/to/small-run --dry-run
```

The complete effective configuration and adapter identity are recorded with each
run. [RESULTS.md](RESULTS.md#release-artifacts) describes the released components
and separately supplied run assets.

## Slurm execution

Use your cluster's allocation policy and submit a batch job from an inspected
tmux session. The worker requires an owned RUNNING GPU allocation with a numeric
`srun` step, activates the environment inside that step, and sets `PYTHONPATH`.

```bash
export RUBRIC_ROOT=/absolute/path/to/Rubric-CEPR
export RUBRIC_ENV_PREFIX=/absolute/path/to/model-environment
export RUBRIC_ACTIVATE_SCRIPT="$RUBRIC_ROOT/scripts/slurm/activate.example.sh"
export HF_HOME=/absolute/path/to/huggingface-cache
sbatch --export=ALL scripts/slurm/train_cepr.sbatch \
  --manifest /path/to/sources.jsonl --output /path/to/new-run
```

The example requests one GPU, four CPUs, 96 GiB of host memory and 24 hours.
Adjust these values to your cluster. The job ends when the workload exits.
Preflight validates the source manifest, complete ImgEdit disjointness and model
environment before loading models. The internal loop records proposal, reward,
gate, target and training artifacts per round. Existing runs may be resumed by
the loop's explicit resume configuration; use a fresh output root for a new study.

## Completion and evaluation

Keep the effective configuration, round training commands, completion receipts,
source manifest, gate traces and adapter together. To inspect a completed round:

```bash
rubric-cepr check --training-output /path/to/new-run/round-directory/training_output
rubric-cepr check --checkpoint /path/to/pytorch_lora_weights.safetensors
```

The check verifies the requested step count and reports the supplied adapter's
hash. It does not identify that adapter as a released paper checkpoint.
Evaluate the exact artifact with matched protocols in [EVALUATION.md](EVALUATION.md).

## Inference

```bash
rubric-cepr infer \
  --image /path/to/source.jpg --prompt 'Make the car red.' \
  --checkpoint /path/to/pytorch_lora_weights.safetensors \
  --output /path/to/new-edit.png
```

Inference uses the pinned Qwen revision, seed 42, 40 steps, true CFG 4, guidance
scale one and a space negative prompt. It loads the supplied adapter without a
detector. `--expected-checkpoint-sha256` optionally binds the weights to a recorded
hash; `--dry-run` prints settings without loading a model.
