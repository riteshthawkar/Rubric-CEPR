# Rubric-CEPR

Code and artifact records for **Rubric-CEPR: Self-Evolving Image Editing via
Reward-Verified Self-Distillation**. Version: **1.0.0-rc1**.

Source repository: [riteshthawkar/Rubric-CEPR](https://github.com/riteshthawkar/Rubric-CEPR).

This release exposes the surviving **64-pair, extraction-only rank-16 LoRA,
400-step** recipe and the broader Planner–Editor–Critic implementation as
separate entry points. The surviving checkpoint is not evidence that the full
multi-round framework produced the submitted headline scores. See
[results and evidence boundaries](docs/RESULTS.md) before comparing numbers.

## Start here

Use this source checkout with Python 3.11. Keep the checkout available: configs
and provenance are intentionally inspectable files beside the code. Install
editable rather than installing only a wheel.

```bash
python -m pip install -e '.[dev]'
python scripts/accv_v1.py check
python -m pytest -q
```

For model execution, install the recorded environment in
[requirements-model.txt](requirements-model.txt) and
[requirements-eval.txt](requirements-eval.txt). Those versions describe the
reconstruction environment; other environments are not certified for exact
reproduction. Do not install packages into the shared experiment environment
without checking it first. This release preparation does not change that
environment.

```bash
python -m pip install -r requirements-model.txt -r requirements-eval.txt
python scripts/accv_v1.py check --strict-environment
```

The base model and benchmark images are downloaded separately. Training needs
a GPU with enough memory for Qwen-Image-Edit-2509; the recorded environment used
an H200. CPU checks do not initialize CUDA, load a model, or call a judge.

## Commands

| Command | Purpose | Runs a model or judge? |
|---|---|---|
| `check` | Verify code, data, checkpoint, environment and optional benchmark overlap | No |
| `install-artifacts` | Safely install the exact 128 images from the historical bundle | No |
| `train --dry-run` | Print the original training arguments with portable paths | No |
| `train` | Reconstruct the surviving extraction-only adapter | Yes |
| `infer` | Edit one image with the historical or explicitly labeled reconstructed adapter | Yes |
| `export` / `score` | Run ImgEdit, GEdit or Complex-Edit under a recorded rerun protocol | Yes |
| `framework --dry-run` | Print the broader framework launch command | No |
| `framework` | Run the Planner–Editor–Critic reference configuration | Yes |
| `results` | Display the preserved historical metrics | No |

All model/judge commands require a verified owned Slurm GPU allocation, a
numeric `srun` step, tmux context and this checkout's `PYTHONPATH`. They fail
before model imports on the login node. The original low-level modules are
implementation details; use these guarded release commands. Follow the parent
workspace's GPU policy while this copy is hosted there.

## Reconstruct or use the historical artifact

The training-image bundle and 91 MB adapter are companion assets, separate
from the source archive. Their hashes are in
[provenance/IDENTITY.json](provenance/IDENTITY.json). There is no invented public
download URL. The local prepared companion directory is `../accv-v1-assets/`.
Data and base models retain their original distribution terms.

```bash
python scripts/accv_v1.py install-artifacts \
  --bundle /path/to/paper_headline_extract_v1.tar.gz \
  --data-root /path/to/new-v1-data
python scripts/accv_v1.py check \
  --data-root /path/to/new-v1-data \
  --checkpoint /path/to/pytorch_lora_weights.safetensors
```

[Reproduction instructions](docs/REPRODUCTION.md) cover the frozen model
revision, full benchmark overlap check, allocation environment, exact training
command, completion checks and inference. [Benchmark instructions](docs/BENCHMARKS.md)
cover setup and matched Base/LoRA reruns. Benchmarks use external API judges
**only for evaluation**; their scores are not training targets or acceptance
feedback in the released recipe.

## Code map

| Location | Role |
|---|---|
| `src/accv_v1/` | Portable launch commands, artifact checks and GPU guards |
| `src/qwen_edit_project/train/` | Original hash-pinned trainer and processor compatibility shim |
| `src/qwen_edit_project/self_evolve/` | Broader framework implementation and internal critic |
| `src/qwen_edit_project/eval/` | Export, strict score parsing and provenance |
| `configs/train/extraction_v1.json` | Exact command template derived from the frozen launcher |
| `configs/reproduction/` | Unchanged historical training contract |
| `configs/eval/` | Release-specific rerun protocols, not original score receipts |
| `reproducibility/` | Exact manifest, image hashes and surviving score records |
| `provenance/` | Source identities and original-vs-release file mapping |
| `tools/` | Pinned benchmark setup, verification and source-only packaging |

The historical extraction miner is retained in `scripts/build_extract_selfdistill.py`
for inspection. It uses an external GroundingDINO detector to name source
objects and a white-border/object-presence/sharpness heuristic to select editor
outputs. It does not implement the full internal rubric. Re-mining is not exact
reconstruction of the 64-row snapshot. See [method scope](docs/METHOD_SCOPE.md).

This is a **source release candidate**. CPU preparation
and artifact checks do not establish GPU reproducibility or certify every
submitted score. [Release readiness](docs/RELEASE_READINESS.md) records those
limits and the remaining license/citation metadata. Companion assets are not
included in this source repository.
