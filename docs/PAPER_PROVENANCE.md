# Paper claims and surviving evidence

The internal Planner–Editor–Critic implementation on `main` has not been
authenticated as the pipeline that produced the paper's headline scores.
The historical extraction adapter uses a different recipe: GroundingDINO names
and filters source objects; Qwen generates candidates; border whiteness,
nonwhite occupancy and sharpness select self-distillation targets.
Moving its code to a separate branch does not change its training provenance.

The machine-readable inventory is
[`paper_provenance.json`](../reproducibility/paper_provenance.json).
Its recorded measurements and supporting files live on
[`groundingdino-extraction`](https://github.com/riteshthawkar/Rubric-CEPR/tree/groundingdino-extraction/docs/PAPER_PROVENANCE.md).
This documentation distinguishes recovered measurements from claims whose
checkpoint and evaluation provenance remains unresolved. The audit covers
headline benchmark claims and the multi-family analysis; it does not certify
every manuscript figure or ablation.

| Evidence | Base | Candidate | Interpretation |
|---|---:|---:|---|
| Historical extraction, ImgEdit-737 | 4.440638 | 4.555088 | 64 training pairs; recorded before content-hashed evaluation contracts |
| Recovered extraction recipe, strict three-seed ImgEdit-737 control | 4.545002 | 4.593246 | Separate adapters and judge protocol; not the original paper result |
| Historical four-family bank, ImgEdit-737 | 4.440638 | 4.536988 | Separate 149-pair manifest and adapter |
| Historical extraction, Complex-Edit real C4, 531 records | 8.7674 | 8.8064 | Recorded evaluation; not 8.91 or 8.97 |
| Historical extraction-labeled GEdit subset | 8.192449 | 8.312768 | 330 records: eleven tasks, 30 Chinese and zero English records per task |

The historical ImgEdit aggregate was recomputed from the 737 per-item scores.
The strict control's training-seed results are 4.584351, 4.608774 and 4.586612;
they do not authenticate the README's seed list of 4.58, 4.60 and 4.62 or its
4.36 base. The strict mean gain is +0.048244, with a paired item-bootstrap
95% interval of [0.012362, 0.085029]. This interval measures held-out item
uncertainty, not all sources of training or judging uncertainty.

The old GEdit subset numerically explains the rounded 8.19 to 8.31 values,
but its generation records lack an authenticated checkpoint contract. It does
not establish the current README's 7.39 baseline, full-benchmark performance,
English-only performance or balanced-language performance.

The available three-seed Step1X control is narrower than the broad transfer
claim: 57 English removal records. Its mean overall delta is +0.190577 with
95% interval [-0.339216, 0.778560], while perceptual quality decreases by
0.584795. It does not verify the reported broad 6.69 to 7.24 GEdit or 3.86 to
4.16 ImgEdit improvements.

## Code and supervision boundaries

The detector branch now includes the recovered multi-family miner, its
yes/no-token verifier, the ranker-validation script and its reference config.
The miner and validation files are byte-identical to a hash-verified legacy
snapshot. That snapshot contains later recovery work and is not certified as
the exact submitted source tree or exact code at the original execution.

For extraction, GroundingDINO supplies object labels for task construction;
candidate reward is programmatic. An accurate description is “no human-edited
targets or external reward judge; GroundingDINO supplies source-object labels.”
The stronger assertion of no external model or semantic supervision anywhere
in training-data construction is unsupported for this recipe.

For the recovered bank, removal and replacement can also use detector output
inside candidate rewards. Replacement and action records include editor-side
yes/no probabilities. “Detector-free” reward mode in the archived code still
uses GroundingDINO for source naming and localization, so it does not describe
a wholly internal training pipeline.

Nonwhite occupancy checks rough object presence, not semantic completeness,
instance identity or correspondence to the source. Those stronger checks must
not be credited to the historical extraction heuristic.

## What remains unresolved

Each primary-method claim still needs an actual checkpoint, mining manifest,
reward implementation, generation configuration, dataset identities and raw
score receipts that agree with the claimed value. The missing links are
explicitly represented in `paper_provenance.json`; code recovery alone does
not fill them. The recovered historical records and prospective protocols
must retain separate identities.
