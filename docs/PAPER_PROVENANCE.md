# Final results and historical runs

The author confirmed on 8 October 2026 that the results in the final paper and
README are the final results obtained. They are reported in
[RESULTS.md](RESULTS.md). The recovered historical extraction runs and later
controls are different experiments; their values must not replace the final
results or be used as evidence that the final results are incorrect.

## Final reported results

| Benchmark | Base | Rubric-CEPR |
|---|---:|---:|
| Qwen-Image-Edit, ImgEdit overall | 4.36 | 4.60 ± 0.02 SD |
| Qwen-Image-Edit, ImgEdit Extract family | 3.41 | 4.26 |
| Qwen-Image-Edit, GEdit-Bench overall | 7.39 | 8.31 |
| Qwen-Image-Edit, Complex-Edit overall | 8.77 | 8.97 |
| Step1X-Edit, GEdit-Bench overall | 6.69 | 7.24 |
| Step1X-Edit, ImgEdit overall | 3.86 | 4.16 |

The final ImgEdit seed scores reported in the README are 4.58, 4.60 and 4.62.
These are the author's final reported measurements. The repository's earlier
audit examined available historical files rather than the final run bundle;
that audit did not establish the provenance of the final runs.

The supplied final Overleaf manuscript was cloned and inspected at commit
`b64d77ad0d38f29b1aded73247dcd6f7fc172ee8`. Its active `main.tex` includes
`tables/main_results.tex`, `tables/transfer_benchmark_results.tex` and
`tables/step1x_transfer.tex`; those tables agree with the README values above.
`sections/5_results_and_analysis.tex` gives the same final ImgEdit seed list.

## Release implementation and configuration

`main` contains the internal Planner–Editor–Critic implementation. Its supplied
configuration is a demonstration configuration with 16 Editor updates at
learning rate 1e-6; it is not the final-paper training recipe. The final run's
effective configuration, source manifest, checkpoint and evaluation records
are needed to associate the release with that exact experiment.

The final manuscript specifies an internal-only primary loop and separates
the broader yes/no-token verifier bank and detector-assisted addition analysis.
The recovered historical miners have their own detector usage, including
source naming and localization. Their supervision boundary must not be
attributed to the final primary experiment merely because their code survives.

[RELEASE_ALIGNMENT.md](RELEASE_ALIGNMENT.md) records the concrete implementation
and configuration differences identified so far. Changing a demonstration
budget alone does not identify the final experiment or resolve missing reward
components. CPU tests establish software checks, not benchmark performance.

## Historical detector-assisted experiments

The following records are retained on
[`groundingdino-extraction`](https://github.com/riteshthawkar/Rubric-CEPR/tree/groundingdino-extraction):

| Historical experiment | Base | Candidate | Scope |
|---|---:|---:|---|
| 64-pair extraction, ImgEdit-737 | 4.440638 | 4.555088 | Earlier extraction-only adapter |
| Recovered extraction recipe, strict three-seed ImgEdit-737 control | 4.545002 | 4.593246 | Separate adapters and evaluation protocol |
| Four-family bank, ImgEdit-737 | 4.440638 | 4.536988 | Separate 149-pair manifest and adapter |
| Extraction, Complex-Edit real C4 | 8.7674 | 8.8064 | Earlier 531-record evaluation |
| Extraction-labeled GEdit subset | 8.192449 | 8.312768 | Eleven tasks, 30 Chinese and zero English records per task |

The narrow Step1X removal control is also retained with its original scope and
records. It is not the author's final broad-transfer experiment. Comparisons
between these historical controls and final results require matching the actual
run identities, dataset selections, generation settings and judging protocols.

The historical miners used GroundingDINO for source naming and localization,
and some multi-family rewards also used candidate detection. The extraction
heuristic measured border whiteness, nonwhite occupancy and sharpness; occupancy
is a presence proxy rather than a semantic completeness check. These details
describe those historical experiments and must not be attributed to the final
primary method without its actual run configuration.

## Result and artifact records

[`paper_provenance.json`](../reproducibility/paper_provenance.json) separates the
author's final reported values from recovered historical measurements. Recorded
scores, hashes and original source identities for the historical experiments
remain available. The documentation will identify the final run's artifact
paths when they are provided; author-reported results and independently inspected
artifact bundles are distinct sources of evidence.
