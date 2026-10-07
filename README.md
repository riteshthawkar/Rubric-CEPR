# Rubric-CEPR: Self-Evolving Image Editing via Reward-Verified Self-Distillation

Rubric-CEPR improves a pretrained image editor from its own verified edits, with
no human-edited targets and no external reward model during training. A Planner
specifies an edit, an Editor generates candidates, and a frozen Critic checks the
requested change and preservation of the source. Only candidates that pass every
applicable gate become training targets for a lightweight adapter, which is then
evaluated in the standard single-shot setting.

The Critic scores candidates with a rubric-augmented **Contrastive
Edit-Preservation Reward (CEPR)**. CEPR contrasts support for the requested edit
with incorrect alternative instructions and rewards preservation of unrelated
content; the rubric adds explicit checks for required states, removal of forbidden
old states, and preservation constraints. A candidate's reward is
`R = G * sqrt(E * P)`, and it is zero whenever a gate fails.

[Method](docs/METHOD.md) · [Training](docs/TRAINING.md) ·
[Evaluation](docs/EVALUATION.md) · [Results](docs/RESULTS.md)

## Architecture

![Rubric-CEPR architecture: the frozen Qwen-Image-Edit backbone, Planner, Editor, internal rubric Critic and best-of-N self-evolution loop](assets/architecture.png)

The framework figure from the paper. A frozen Qwen-Image-Edit backbone provides
understanding features and VAE latents for the internal Critic. The Planner
specifies edits, the Editor samples candidates, and gate-passing candidates are
selected for Planner and Editor updates across rounds.

## Overview

| Component | Role | Implementation |
|---|---|---|
| Planner | Propose an instruction and a structured edit specification | `self_evolve/edit_schema.py`, `self_evolve/backends.py` |
| Editor | Generate candidate edits and learn from accepted targets | `self_evolve/loop.py`, `train/` |
| Critic | Check semantics, preservation, validity and edit-specific rubric items | `self_evolve/backends.py` |

Implementation paths above are relative to `src/qwen_edit_project/`.
The reference framework uses the editor's pretrained components for its internal
critic. External benchmark judges are used only during evaluation.

The repository also includes a fixed **extraction-specific self-distillation
recipe** (64 generated pairs, a rank-16 LoRA, 400 training steps). Its pair miner
uses GroundingDINO for object naming and image heuristics for acceptance. The full
Planner–Editor–Critic implementation is available as a separate `framework`
command. [Results](docs/RESULTS.md) keeps the two kinds of evidence apart.

## Results

Scores from the paper. Base is Qwen-Image-Edit-2509 under the same protocol; the
adapted ImgEdit score is the mean over three training seeds (4.58, 4.60, 4.62).
Rel. gain is `100 * (adapted - base) / base`.

| Benchmark | Qwen-Image-Edit-2509 | Rubric-CEPR | Rel. gain |
|---|---:|---:|---:|
| ImgEdit (overall, 0–5) | 4.36 | **4.60** ± 0.02 SD | +5.5% |
| ImgEdit, Extract family | 3.41 | **4.26** | +24.9% |
| GEdit-Bench (overall, 0–10) | 7.39 | **8.31** | +12.4% |
| Complex-Edit (overall, 0–10) | 8.77 | **8.97** | +2.3% |

The same procedure applied to Step1X-Edit, using its own VLM and VAE, improves the
overall GEdit-Bench score from 6.69 to 7.24 (+8.2%) and the overall ImgEdit score
from 3.86 to 4.16 (+7.8%).

The per-family and per-metric tables, the judges used, and the earlier recorded
evaluations of the extraction adapter shipped here are in
[docs/RESULTS.md](docs/RESULTS.md). The machine-readable records under
`reproducibility/results/` and their hash-pinned summaries come from that earlier
evaluation and are unchanged; refreshed records for the paper's evaluation will be
added.

## Repository layout

```text
src/
  rubric_cepr/                 # Public CLI, artifact checks and GPU guard
  qwen_edit_project/
    self_evolve/                # Planner, Editor, Critic and round orchestration
    train/                      # Editor and Planner LoRA training
    eval/                       # Benchmark exports, scoring and result parsing
    utils/                      # Model loading and configuration helpers
configs/
  self_evolve/                 # Reference framework configuration
  train/                      # Extraction training command
  reproduction/               # Model revision, recipe and environment
  eval/                       # Matched benchmark protocols
scripts/                      # CLI launcher, data preparation and Slurm example
reproducibility/               # Training manifests, hashes and score records
docs/                         # Method, training, evaluation and results
tests/                        # CPU tests for inputs, commands and boundaries
assets/                       # Framework figure from the paper
```

## Setup

Use a source checkout with Python 3.11. The CLI reads the configurations and
manifests beside the source, so install it in editable mode:

```bash
git clone https://github.com/riteshthawkar/Rubric-CEPR.git
cd Rubric-CEPR
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

## Citation

```bibtex
@misc{thawkar2026rubriccepr,
  title={Rubric-CEPR: Self-Evolving Image Editing via Reward-Verified Self-Distillation},
  author={Thawkar, Ritesh and Patle, Shubham and Venkatraman, Shravan and Anwer, Rao Muhammad},
  year={2026},
  note={Preprint}
}
```

## Acknowledgements

This implementation uses [Qwen-Image-Edit](https://github.com/QwenLM/Qwen-Image),
[Diffusers](https://github.com/huggingface/diffusers) and
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO). Evaluation builds
on [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit),
[Step1X-Edit / GEdit](https://github.com/stepfun-ai/Step1X-Edit) and
[Complex-Edit](https://github.com/UCSC-VLAA/Complex-Edit). Exact scorer revisions
and patches are recorded in [configs/eval/scorers.json](configs/eval/scorers.json).
