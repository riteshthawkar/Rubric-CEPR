# GroundingDINO-assisted Extraction Self-Distillation

This branch preserves the detector-assisted extraction baseline. The internal
Rubric-CEPR method is maintained on [main](https://github.com/riteshthawkar/Rubric-CEPR/tree/main).

GroundingDINO names a dominant source object and supplies the instruction
`Extract the <object>.` Qwen-Image-Edit generates candidate images. A reward
based on white borders, foreground area and sharpness selects a training target.
The fixed recipe trains a rank-16 LoRA on 64 generated pairs for 400 steps.

GroundingDINO is an external pretrained component used during pair construction.
This recipe does not implement the internal contrastive CEPR rubric, Planner
training, or a multi-round self-evolution run. The optimizer and inference use
Qwen and its adapter; they do not call the detector.

[Method](docs/METHOD.md) · [Training](docs/TRAINING.md) ·
[Evaluation](docs/EVALUATION.md) · [Results](docs/RESULTS.md)

## Recorded results

| Benchmark | Examples | Base | Extraction adapter | Change |
|---|---:|---:|---:|---:|
| ImgEdit Basic | 737 | 4.4406 | 4.5551 | +0.1145 |
| Complex-Edit real C4 | 531 | 8.7674 | 8.8064 | +0.0390 |

These records identify the fixed extraction adapter. Other evaluated variants,
including the negative GEdit control, are preserved in [Results](docs/RESULTS.md).
They are separate from the paper scores displayed on main.

## Setup

Use a source checkout with Python 3.11. The CLI reads the configurations and
manifests beside the source, so install it in editable mode:

```bash
git clone https://github.com/riteshthawkar/Rubric-CEPR.git
cd Rubric-CEPR
git switch groundingdino-extraction
python -m pip install -e '.[dev]'
rubric-cepr check
python -m pytest -q
```

For model execution, use the versions in
[requirements-model.txt](requirements-model.txt). Install PyTorch and torchvision
for your CUDA platform first, then install the remaining requirements. Benchmark
scoring additionally needs [requirements-eval.txt](requirements-eval.txt).

```bash
python -m pip install -r requirements-model.txt -r requirements-eval.txt
rubric-cepr check --strict-environment
hf download Qwen/Qwen-Image-Edit-2509 \
  --revision d3968ef930e841f4c73640fb8afa3b306a78167e
```

Training images, adapters and base weights are separate from this source
repository. [Artifact hashes](reproducibility/artifacts.json) identify the
training bundle and reference adapter. No public asset download is configured.
Set cache paths and evaluation credentials using [.env.example](.env.example).

Model execution uses an owned, running Slurm GPU allocation with a numeric
`srun` step. See [Training](docs/TRAINING.md) for the environment and launcher.
The `check`, `results`, installation and `--dry-run` commands run on CPU.

## Inference

Inside the configured GPU step, provide a source image, instruction and adapter:

```bash
rubric-cepr infer \
  --image /path/to/source.jpg \
  --prompt 'Extract the dog.' \
  --checkpoint /path/to/pytorch_lora_weights.safetensors \
  --output /path/to/edit.png
```

The default verifies the reference adapter's hash. For a retrained adapter, add
`--allow-reconstructed-checkpoint`. Add `--dry-run` to inspect the inference
settings without loading a model. `python scripts/rubric_cepr.py` provides the
same commands without installing the console entry point.

## Training

Install the verified input bundle into a new directory, then inspect the recipe:

```bash
rubric-cepr install-artifacts \
  --bundle /path/to/training-inputs.tar.gz --data-root /path/to/training-data
rubric-cepr train --dry-run \
  --data-root /path/to/training-data --output /path/to/training-output
```

Run without `--dry-run` inside the GPU step after the full ImgEdit source-overlap
check and environment check pass. The fixed recipe uses 400 steps, learning rate
1e-4, rank 16, batch size one and seed 123. Output directories must be new.
[Training instructions](docs/TRAINING.md) cover input preparation, Slurm launch
and completion checks.

To inspect the broader self-evolution configuration:

```bash
rubric-cepr framework --dry-run
```

Prepare your own unlabeled source manifest using
[examples/unlabeled_manifest.jsonl](examples/unlabeled_manifest.jsonl). The
reference configuration generates candidates and emits training commands by
default. See [Method](docs/METHOD.md) for the role of each component.

## Evaluation

The included protocols cover ImgEdit Basic, GEdit and Complex-Edit real C4.
Create separate Base and adapter configurations with identical generation and
judge settings, and distinct output directories:

```bash
python scripts/setup_benchmarks.py --dry-run
rubric-cepr export --benchmark imgedit --config /path/to/base.yaml --dry-run
rubric-cepr export --benchmark imgedit --config /path/to/adapter.yaml --dry-run
```

Run exports and `rubric-cepr score` inside the GPU step once data, scorer
checkouts and private API credentials are prepared. [Evaluation instructions](docs/EVALUATION.md)
give dataset versions, benchmark-specific settings and scoring commands.

## Acknowledgements

This implementation uses [Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image),
[Diffusers](https://github.com/huggingface/diffusers) and
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO). Evaluation builds
on [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit),
[Step1X-Edit / GEdit](https://github.com/stepfun-ai/Step1X-Edit) and
[Complex-Edit](https://github.com/UCSC-VLAA/Complex-Edit). Exact scorer revisions
and patches are recorded in [configs/eval/scorers.json](configs/eval/scorers.json).
