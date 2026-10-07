# Results

This page separates two kinds of evidence: the scores reported in the paper, and
the earlier recorded evaluations of the extraction adapter shipped in this
repository.

## Paper results

Base is Qwen-Image-Edit-2509 evaluated under the same protocol as the adapted
model. ImgEdit is scored by GPT-4o on a 0–5 scale; GEdit-Bench uses all 11 tasks
with GPT-4.1 VIEScore (0–10); Complex-Edit uses its 0–10 metrics. The adapted
ImgEdit overall score is the mean over three training seeds (4.58, 4.60, 4.62;
4.60 ± 0.02 SD; gain +0.24 ± 0.02 SD). Rel. gain is
`100 * (adapted - base) / base`.

### ImgEdit, per edit family

| Model | Add | Remove | Replace | Adjust | Action | Style | Bg. | Extract | Compose | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen-Image-Edit-2509 | 4.51 | 4.36 | 4.72 | 4.40 | 4.69 | 4.70 | 4.40 | 3.41 | 4.05 | 4.36 |
| Rubric-CEPR | 4.65 | 4.50 | 4.86 | 4.54 | 4.82 | 4.83 | 4.59 | 4.26 | 4.39 | 4.60 |
| Rel. gain | +3.1% | +3.2% | +3.0% | +3.2% | +2.8% | +2.8% | +4.3% | +24.9% | +8.4% | +5.5% |

### GEdit-Bench and Complex-Edit

| Model | GEdit SC | GEdit PQ | GEdit O | CE IF | CE ID | CE PQ | CE O |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen-Image-Edit-2509 | 8.11 | 7.10 | 7.39 | 9.69 | 9.02 | 7.60 | 8.77 |
| Rubric-CEPR | 9.05 | 7.92 | 8.31 | 9.75 | 9.18 | 7.97 | 8.97 |
| Rel. gain | +11.6% | +11.5% | +12.4% | +0.7% | +1.8% | +4.9% | +2.3% |

SC: Semantic Consistency; PQ: Perceptual Quality; O: Overall; CE: Complex-Edit;
IF: Instruction Following; ID: Identity Preservation. Complex-Edit's overall score
is the mean of its three metrics.

### Second editor: Step1X-Edit

| Model | GEdit SC | GEdit PQ | GEdit O | ImgEdit O |
|---|---:|---:|---:|---:|
| Step1X-Edit | 7.07 | 7.58 | 6.69 | 3.86 |
| Rubric-CEPR | 7.84 | 8.05 | 7.24 | 4.16 |
| Rel. gain | +10.9% | +6.2% | +8.2% | +7.8% |

The overall gains are +0.55 ± 0.05 SD over three training seeds (GEdit-Bench) and
+0.30 ± 0.06 s.e. (ImgEdit).

## Earlier recorded evaluations of the extraction adapter

The rows below, and the files under `reproducibility/results/`, record an earlier
evaluation of the fixed extraction adapter. They use their own base score (ImgEdit
4.4406) and a single adapter, so they are not the paper's numbers above. The
records are kept unchanged because their hashes and the repository tests refer to
them.


| Benchmark | Examples | Base | Extraction adapter | Change |
|---|---:|---:|---:|---:|
| ImgEdit Basic | 737 | 4.440638 | 4.555088 | +0.114450 |
| Complex-Edit real C4 | 531 | 8.7674 | 8.8064 | +0.0390 |

Both candidate rows correspond to the extraction adapter identified by
`checkpoint_sha256` in `reproducibility/artifacts.json`. Its training uses 64
pairs and 400 steps. ImgEdit's extraction subscore is **3.51 → 4.26 (+0.75)**;
the overall gain does not establish consistent gains across edit categories.

## Other evaluated variants

| Benchmark / variant | Base | Candidate | Change | Scope |
|---|---:|---:|---:|---|
| ImgEdit Basic, extraction with replay fix | 4.440638 | 4.564084 | +0.123446 | Separate adapter; not the fixed extraction checkpoint |
| GEdit full-1212, naive round-loop | 8.123821 | 8.101060 | −0.022761 | Negative control; not the fixed extraction checkpoint |

The negative GEdit result remains part of the record. Neither it nor the replay
variant can be substituted for an evaluation of the extraction adapter. The
included records do not establish English-only GEdit performance or broad
transfer across editing tasks.

## Protocol scope

These measurements predate the content-hashed contracts in `configs/eval/`.
The configurations specify matched evaluations with explicit dataset, generation,
judge and output identities. They are protocols for new runs, not receipts for
the recorded results. Use [EVALUATION.md](EVALUATION.md) to run those protocols
and report the actual checkpoint hash, selection and completed example count.
