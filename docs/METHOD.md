# Rubric-CEPR method

CEPR means **Contrastive Edit-Preservation Reward**. This branch provides the
general internal Planner–Editor–Critic workflow on unlabeled source images:

1. The editor-side Qwen Planner proposes an instruction and structured rubric.
2. The Qwen Editor generates several candidates for the same proposal.
3. A fixed Critic uses Qwen understanding features, text anchors and VAE latents
   to check the requested change, preservation and validity.
4. The highest-reward feasible candidate becomes an Editor SFT target.
   Planner traces support its separate adapter update. Updated adapters can be
   promoted into subsequent rounds.

The trainer supports per-sample weights. In the demonstration configuration,
accepted targets have unit weight and rejected candidates are excluded; this
is selected-target SFT rather than reward-proportional weighting. Match the
final run's configuration before attributing a different weighting policy to it.

The final paper's primary instantiation is object isolation. Its programmatic
background-purity and object-completeness checks are not present in this general
reward path. The final paper also specifies replay and a strict Planner
band-pass admission rule that the demonstration configuration does not enable.
[RELEASE_ALIGNMENT.md](RELEASE_ALIGNMENT.md) lists these differences and the
settings available from the final manuscript.

The edit term contrasts the true instruction with incorrect alternatives. The
rubric checks source grounding, required after-states, forbidden old states and
preservation constraints. The selection reward is `R = G * sqrt(E * P)`; a failed
applicable gate makes the candidate infeasible. The Critic receives no gradients.

| Component | Implementation |
|---|---|
| Structured proposals | `src/qwen_edit_project/self_evolve/edit_schema.py` |
| Internal CEPR and rubric checks | `src/qwen_edit_project/self_evolve/backends.py` |
| Candidate selection and round orchestration | `src/qwen_edit_project/self_evolve/loop.py` |
| Planner training traces | `src/qwen_edit_project/self_evolve/proposer_training.py` |
| Editor adapter trainer | `src/qwen_edit_project/train/diffusers_qwen_edit_lora.py` |
| Planner adapter trainer | `src/qwen_edit_project/train/train_proposer_lora.py` |

The public CLI checks the effective configuration, including overrides, before
model imports. It requires an internal CEPR evaluator and the same Qwen backbone
for planning and editing. External detector loading is absent from this branch.
Optional grounding uses the editor-side Qwen component only; it is disabled in
the reference configuration. API benchmark judges are evaluation-only.

## Source inputs

Create a JSONL manifest using [the example](../examples/unlabeled_manifest.jsonl).
Each row has a unique `key`, a source `image`, and optional `metadata`. Relative
image paths are resolved against the checkout. Supply unlabeled images without
captions, edited targets or preference labels. Keep benchmark sources out of the
pool. The public workflow validates source existence, duplicate bytes, and overlap
with the complete ImgEdit source set before execution.

## Branch separation

The [groundingdino-extraction branch](https://github.com/riteshthawkar/Rubric-CEPR/tree/groundingdino-extraction)
contains the detector-assisted extraction miner, fixed 64-pair training manifest,
reference adapter identity and recorded evaluations. These are historical
experiments with a separate recipe and must not replace the final paper results.

The top-level README summarizes the main workflow. [TRAINING.md](TRAINING.md) gives
its commands; [RESULTS.md](RESULTS.md) distinguishes reported values from available
reproduction artifacts.
