# Benchmark setup and evaluation

The implementation includes ImgEdit, GEdit and Complex-Edit exports and scorers.
Recorded metrics and their scope are in [RESULTS.md](RESULTS.md).
The configs below are matched-run protocols; recorded score summaries use their own settings.

## Scorer sources

Run `python tools/bootstrap_benchmarks.py --dry-run` to inspect the exact
repositories and commit IDs in `third_party/SOURCES.json`. Running without
`--dry-run` creates fresh checkouts and applies hash-pinned scorer patches.
Existing checkouts are never updated or cleaned. No credentials, `.git`
histories or third-party datasets are bundled in the source repository.

| Benchmark | Upstream source | Required data |
|---|---|---|
| ImgEdit Basic | PKU-YuanGroup/ImgEdit, pinned in `SOURCES.json` | Full 737-entry `basic_edit.json`, original images and Basic `prompts.json` |
| GEdit | stepfun-ai/Step1X-Edit, pinned scorer patch | Hugging Face `stepfun-ai/GEdit-Bench`, revision `50766778e2a737474c7e9bdf84cdce82c3ea3f4f` |
| Complex-Edit | UCSC-VLAA/Complex-Edit, pinned scorer patch | Hugging Face `UCSC-VLAA/Complex-Edit`, revision `b0b8a81d740ae413d52572281a23dc975c4b4b91`, real split |

For ImgEdit, materialize the Basic original images under
`data/processed/benchmark/imgedit/original_images/` and copy the upstream
`Benchmark/Basic/basic_edit.json` to
`data/processed/benchmark/imgedit/basic_edit.json`. `prompts.json` remains in
the pinned upstream checkout. Download benchmark data from its upstream
distribution; no benchmark images are redistributed here.

GEdit's exporter loads the pinned dataset directly through `datasets`.
The default selection is 15 CN plus 15 EN per task, seed 42, and uses
task + language + key identities to prevent CN/EN collisions. This selection
is different from the recorded CN-only transfer slices.

For Complex-Edit, use `scripts/prepare_complex_edit_data.py --splits real`.
The preparation script pins the dataset revision and emits the same full
531-record real split. The hardened C4 validator checks the manifest, selection
and source-image hashes; it refuses substituted inputs. Preparation is
CPU/filesystem work and can run before allocating a GPU.

## Base and adapter runs

Create separate, explicit config files for Base and LoRA from the appropriate
benchmark config:

- `configs/eval/imgedit_hardened.yaml`
- `configs/eval/gedit_hardened.yaml`
- `configs/eval/complex_edit_c4_hardened.yaml`

For LoRA set `model.model_type: lora`, `model.checkpoint_path` to the verified
adapter and a distinct `model.model_name`. Set distinct
`scoring.score_run_id` values and output paths. Preserve all dataset, generation
and judge settings between the two roles. Keep each resolved config with its
outputs; the contracts hash those config bytes and implementation artifacts.
Use the activated environment's Python (the wrapper records its actual path).

Inside a verified GPU step:

```bash
rubric-cepr export --benchmark imgedit --config /path/to/base.yaml
rubric-cepr export --benchmark imgedit --config /path/to/lora.yaml
rubric-cepr score --benchmark imgedit --config /path/to/base.yaml
rubric-cepr score --benchmark imgedit --config /path/to/lora.yaml
```

Use `gedit` or `complex_edit` with its corresponding configs for the other
benchmarks. `--dry-run` prints the launch plan without loading a model or making
API calls. Full contracted exports reject limit/offset shortcuts.

ImgEdit uses `gpt-4o-2024-11-20`; GEdit and Complex-Edit use the recorded
`gpt-4.1` judge setting. Record the judge version used in each run. A
substitute judge is a new protocol and must be reported as such. Strict
response parsing rejects malformed responses; failed or missing items must
not be reported as successful complete evaluation.

Export `OPENAI_API_KEY` privately for ImgEdit and Complex-Edit. GEdit's legacy
VIEScore reader additionally requires `GEDIT_SECRET_ENV_PATH` pointing to a
private file containing the raw key on one line, not a dotenv assignment.
Set permissions to `0600`. The legacy wrapper temporarily supplies that key
file and removes/restores it afterward; do not run concurrent GEdit scorers
from one checkout. Never add credentials or raw key files to the release.

Judge scores must remain evaluation-only; do not use benchmark examples or judgments for
pair acceptance, training or candidate selection.
