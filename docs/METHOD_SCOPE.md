# What each release component establishes

## Artifact-backed extraction recipe

The exact training manifest contains 64 extraction pairs: 29 rows from the
first shard and the first 35 rows of a second shard that was still growing.
The final shards must not be concatenated to reconstruct this training set.
`image` is the generated target; `edit_image` is the unlabeled source.
All sample weights are one.

The surviving recipe trains a fresh rank-16, alpha-16 LoRA for 400 optimizer
steps at learning rate 1e-4, batch size one, accumulation one, seed 123. There
is no warm start, reference penalty, projection cap, replay, negative-pair
training, Planner update, or multi-round adaptation in this recipe.

The exact trainer was recovered from the July artifact bundle, rather than
the later source snapshot. Its SHA-256 is
`0d033ef25ec85d4b80fdd00f594badb4223ec7ff481d9cd32c1109d1f6783b84`.
The processor compatibility entry point constructs Qwen2-VL's tokenizer,
image processor and video processor from the same official snapshot's
`processor/` folder. It changes component loading, not the training objective.

The historical miner uses GroundingDINO for object naming and geometric image
statistics for extraction acceptance. The generated targets come from the
editor. GroundingDINO is an **external pretrained component**. This artifact
therefore cannot justify a claim that every stage uses exclusively the editor's
internal features. External benchmark judges are separate, evaluation-only
components.

## Broader paper framework

The reference implementation contains a Planner that proposes structured edits,
an Editor that samples candidates, and a frozen internal Critic that combines
semantic, preservation, validity and rubric checks. Relevant files:

| Component | File / configuration |
|---|---|
| Orchestration and accepted-target manifests | `self_evolve/loop.py` |
| Internal CEPR and rubric evaluators | `self_evolve/backends.py` |
| Structured edit specification | `self_evolve/edit_schema.py` |
| Planner training traces | `self_evolve/proposer_training.py` |
| Planner adapter trainer | `train/train_proposer_lora.py` |
| Reference configuration | `configs/self_evolve/qwen_edit_2509_internal_cepr_rubric_trainable_proposer.yaml` |

Paths above are relative to `src/qwen_edit_project/`, except configurations.
The source snapshot contains compatibility and exploratory branches added
after the original recipe. They remain labeled reference implementation; their
presence is not evidence that a branch generated any reported score. The
example disables later preference and replay experiments and emits proposed
training commands by default. It is not a certified exact paper experiment.

For source format, see `examples/unlabeled_manifest.jsonl`. Supply actual
unlabeled images and copy the manifest to `data/unlabeled/manifest.jsonl`.
The source image path is relative to the release checkout. Do not put benchmark
examples in this source pool. Use `framework --dry-run` to inspect a launch
plan; a real framework run requires the GPU guard even when its training
configuration emits commands rather than executing them.

New CAPE, native-consistency, counterfactual and revisable-editing experiments
are excluded from the v1 release. They remain in the separate research project.
