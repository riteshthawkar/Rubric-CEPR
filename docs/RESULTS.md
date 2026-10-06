# Results and evidence boundaries

The machine-readable source is
[`recovered_results_v1.json`](../reproducibility/evidence/recovered_results_v1.json).
It was copied unchanged from the frozen evidence snapshot. Its score-summary
hashes identify historical artifacts, not new release evaluations.

All seven referenced summaries were recovered and matched to those original
hashes. Public copies under `reproducibility/evidence/summaries/` redact local
filesystem paths, retaining all metrics. `provenance/SCORE_SUMMARY_MAP.json`
records both original and release-copy hashes and the number of path redactions.
Byte-identical originals remain in the private companion assets; the redacted
files are not represented as byte-identical original receipts.

| Benchmark / artifact | Surviving Base | Surviving candidate | Delta | Submitted text | Status |
|---|---:|---:|---:|---|---|
| ImgEdit-737, extraction-only v1 | 4.440638 | 4.555088 | +0.114450 | 4.44 → 4.60 | Submitted candidate number does not match the surviving summary |
| ImgEdit-737, replay-fix follow-up | 4.440638 | 4.564084 | +0.123446 | — | A different follow-up artifact; not v1 |
| Complex-Edit real C4, 531 items, v1 | 8.7674 | 8.8064 | +0.0390 | 8.77 → 8.91 | Submitted candidate number does not match the surviving summary |
| GEdit full-1212, naive round-loop | 8.123821 | 8.101060 | −0.022761 | Reported separately | Negative historical control; not the extraction-only artifact |
| GEdit 11-task transfer slices | 8.19 | 8.31 | Display-rounded +0.12 | 8.19 → 8.31 | Historical CN-only slices; not English-only or mixed-language evidence |

V1's historical ImgEdit extraction subscore is **3.51 → 4.26 (+0.75)**.
The overall gain is therefore not evidence of consistent improvement across
all edit types.

Historical summaries predate content-hashed evaluation contracts. The release
does not claim fresh matched judge runs, full regeneration of those numbers,
or agreement with the submitted rounded headline. All available measurements,
including the negative control, remain visible. Never relabel a fresh rerun,
another adapter, a different judge, or a later recovery result as a historical
paper result.

The release benchmark configurations implement later strict rerun protocols:
content-hashed inputs, output manifests, strict response parsing, and fresh
score receipts. GEdit's mixed-language selection is 15 CN plus 15 EN per task;
that differs from the historical all-CN slices. Different judge settings can
also change the score. Report those configurations as **release reruns** with
their exact contract IDs and sample counts.
