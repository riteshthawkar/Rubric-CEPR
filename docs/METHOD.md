# GroundingDINO extraction baseline and checkpoint scope

This branch preserves the detector-assisted baseline and its original artifacts.
The internal Rubric-CEPR workflow is maintained on `main`.

## Self-evolution framework

The reference implementation starts from unlabeled source images and separates
proposal, generation, verification and learning:

1. The Planner produces an instruction and structured edit specification.
2. The Editor generates multiple candidates for that specification.
3. A frozen Critic checks semantics, preservation, validity and rubric items.
4. Accepted candidates become supervised targets for an editor update. Planner
   training traces and its adapter trainer are provided separately.

| Component | Implementation |
|---|---|
| Round orchestration and accepted-target manifests | `src/qwen_edit_project/self_evolve/loop.py` |
| Internal CEPR and rubric checks | `src/qwen_edit_project/self_evolve/backends.py` |
| Structured edit specification | `src/qwen_edit_project/self_evolve/edit_schema.py` |
| Planner training traces | `src/qwen_edit_project/self_evolve/proposer_training.py` |
| Planner adapter trainer | `src/qwen_edit_project/train/train_proposer_lora.py` |
| Reference configuration | `configs/self_evolve/qwen_edit_2509_internal_cepr_rubric_trainable_proposer.yaml` |

The reference configuration uses internal model components for verification and
keeps preference and replay branches disabled. It emits proposed training
commands by default. Running it still loads generation and verification models;
only the CLI's `--dry-run` is a CPU-only command.

Use `examples/unlabeled_manifest.jsonl` as the source format and create
`data/unlabeled/manifest.jsonl`. Source image paths are relative to the checkout.
Keep benchmark images out of the source pool.

## Extraction self-distillation recipe

The verified training manifest contains 64 extraction pairs: 29 rows from one
shard and the first 35 rows of a second shard. Use these exact rows; rebuilding
from the final shards changes the training set. `image` is the generated target
and `edit_image` is the source. Every sample has unit weight.

The pair miner in `scripts/build_extract_selfdistill.py` uses GroundingDINO for
source-object naming. A white-border, object-presence and sharpness heuristic
selects editor outputs. GroundingDINO is an external pretrained component;
this extraction recipe does not implement the full internal rubric.

The recipe trains a fresh rank-16, alpha-16 LoRA for 400 optimizer steps, with
learning rate 1e-4, batch size one, accumulation one and seed 123. It performs
no replay, negative-pair training, Planner update or multi-round adaptation.
The trainer and safety helper retain their original bytes and pinned hashes.
The processor compatibility loader constructs the tokenizer, image processor
and video processor from the same official model revision's `processor/`
folder; it leaves the training objective unchanged.

The measured extraction-adapter results in [RESULTS.md](RESULTS.md) describe
this fixed recipe. They do not establish improvements for every reference
framework option. Re-mining or retraining produces a new artifact whose hash
and evaluation must be recorded separately.

## Evaluation separation

External API judges are used by benchmark scorers only. Their judgments are
not training targets, pair-acceptance signals or candidate-selection feedback
in either documented configuration. Evaluation inputs must remain separate
from the unlabeled training pool.
