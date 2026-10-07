# Results

The recovered four-family verifier bank and its separate 149-pair manifest are
documented in [VERIFIER_BANK.md](VERIFIER_BANK.md).
[PAPER_PROVENANCE.md](PAPER_PROVENANCE.md) maps headline claims to surviving
records and unresolved checkpoint/evaluation links.

The machine-readable measurements are in
[`results.json`](../reproducibility/results/results.json) and
[`scores.csv`](../reproducibility/results/scores.csv). Seven score summaries are
included in `reproducibility/results/summaries/`. Local filesystem paths have
been redacted without changing their metrics. `reproducibility/score_sources.json`
records original summary hashes, public-copy hashes and redaction counts.

## Extraction adapter

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
