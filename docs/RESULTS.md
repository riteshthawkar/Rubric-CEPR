# Results and artifact availability

The tables below contain the final results reported by the author in the paper
and README. Historical detector-assisted runs are recorded on their separate
branch and retain their own experiment identities.

## Final paper results

Base is Qwen-Image-Edit-2509 evaluated under the same protocol as the adapted
model. ImgEdit is scored by GPT-4o on a 0–5 scale; GEdit-Bench uses all 11 tasks
with GPT-4.1 VIEScore (0–10); Complex-Edit uses its 0–10 metrics. The adapted
ImgEdit overall score is the mean over three training seeds (4.58, 4.60, 4.62;
4.60 ± 0.02 SD; gain +0.24 ± 0.02 SD). Δ Improvement is the relative
improvement over the base, `100 * (adapted - base) / base`.

### ImgEdit, per edit family

| Model | Add | Remove | Replace | Adjust | Action | Style | Bg. | Extract | Compose | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen-Image-Edit-2509 | 4.51 | 4.36 | 4.72 | 4.40 | 4.69 | 4.70 | 4.40 | 3.41 | 4.05 | 4.36 |
| Rubric-CEPR | 4.65 | 4.50 | 4.86 | 4.54 | 4.82 | 4.83 | 4.59 | 4.26 | 4.39 | 4.60 |
| Δ Improvement | +3.1% | +3.2% | +3.0% | +3.2% | +2.8% | +2.8% | +4.3% | +24.9% | +8.4% | +5.5% |

### GEdit-Bench and Complex-Edit

| Model | GEdit SC | GEdit PQ | GEdit O | CE IF | CE ID | CE PQ | CE O |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen-Image-Edit-2509 | 8.11 | 7.10 | 7.39 | 9.69 | 9.02 | 7.60 | 8.77 |
| Rubric-CEPR | 9.05 | 7.92 | 8.31 | 9.75 | 9.18 | 7.97 | 8.97 |
| Δ Improvement | +11.6% | +11.5% | +12.4% | +0.7% | +1.8% | +4.9% | +2.3% |

SC: Semantic Consistency; PQ: Perceptual Quality; O: Overall; CE: Complex-Edit;
IF: Instruction Following; ID: Identity Preservation. Complex-Edit's overall score
is the mean of its three metrics.

### Second editor: Step1X-Edit

| Model | GEdit SC | GEdit PQ | GEdit O | ImgEdit O |
|---|---:|---:|---:|---:|
| Step1X-Edit | 7.07 | 7.58 | 6.69 | 3.86 |
| Rubric-CEPR | 7.84 | 8.05 | 7.24 | 4.16 |
| Δ Improvement | +10.9% | +6.2% | +8.2% | +7.8% |

The overall gains are +0.55 ± 0.05 SD over three training seeds (GEdit-Bench) and
+0.30 ± 0.06 s.e. (ImgEdit).

## Detector-assisted baseline

The earlier 64-pair extraction adapter, its hash-pinned records, replay variant
and negative GEdit control are preserved on
[groundingdino-extraction](https://github.com/riteshthawkar/Rubric-CEPR/tree/groundingdino-extraction/docs/RESULTS.md).
They are separate evidence and are not substituted for the internal-method
results above. The original numbers and provenance remain unchanged on that branch.

## New evaluation runs

Use the matched protocols in [EVALUATION.md](EVALUATION.md), recording the source
selection, model and adapter hashes, generation settings, judge configuration and
completed example counts. The configs are prospective protocols, not completion
receipts for the reported values.

## Release artifacts

This checkout provides the Qwen internal CEPR/rubric workflow, separate Planner
and Editor trainers, replay and adapter-anchoring support, and benchmark tools.
The final paper's extraction-specific pixel checks and Step1X training pipeline
are separate components whose final source is not included in this checkout.
Checkpoints, source manifests, effective run configurations and raw score
receipts are distributed separately from the implementation.

CPU tests cover data and execution boundaries, target admission, record weights
and replay. New model runs retain their own effective configuration, adapter hash
and evaluation receipts. The final paper's reported values and the historical
records are indexed in `reproducibility/paper_provenance.json`.
