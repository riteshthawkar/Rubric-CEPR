# Recovered multi-family verifier bank

This analysis recipe is separate from the fixed 64-pair extraction adapter and
the internal workflow on `main`. It uses GroundingDINO for source naming and
localization, programmatic region statistics, and optionally the editor's own
VLM yes/no token probabilities. It is not a detector-free primary method.

## Recovered files and records

- `scripts/build_family_selfdistill.py`: candidate generation, reward computation
  and target selection.
- `scripts/validate_judge_ranker.py`: ranker and yes/no-verifier analysis over two
  existing benchmark candidate sets; this is evaluation code, not training data.
- `configs/self_evolve/qwen_edit_2509_v45_ranked_preference_caption_extract_keeponly.yaml`:
  the ranker script's archived reference config.
- `reproducibility/verifier_bank/`: the original training manifest, mining
  metadata, recorded training arguments, input hashes, adapter identity and
  recorded evaluation.

The recovered source/config files match their frozen-snapshot hashes. Their
historical comments are retained verbatim and should be read alongside
[PAPER_PROVENANCE.md](PAPER_PROVENANCE.md), which corrects the supervision and
result-attribution boundaries. The exact source version at original execution
has not been certified.

## What was actually recorded

The combined manifest contains **149 pairs**: 29 extraction, 50 removal,
35 replacement and 35 action. Every row matches a surviving mining record.
All **298 source/target references** were checked for existence and content
hashes during recovery. The data images and weights are not included in Git.

| Family | Recorded reward metadata | External detector involvement |
|---|---|---|
| Extraction | Border whiteness, occupancy, sharpness | Source naming and filtering |
| Removal | Outside preservation, inside change, object-gone score | Source localization and candidate detection |
| Replacement | Background preservation, old-gone/new-present detection, VLM probability | Source localization and candidate detection |
| Action | VLM probability and background preservation | Source naming and localization |

The recorded recipe requests 500 SFT updates, seed 123, rank/alpha 16 and
learning rate 1e-4. A final adapter survives, but an exact-step completion
receipt does not. Its SHA-256 is in `identity.json`; requested steps must not
be confused with independently authenticated completed steps.

The surviving full-737 ImgEdit evaluation records **4.536988**, compared with
the historical base **4.440638**, a gain of **0.096350**. It does not authenticate
the manuscript's four-type +0.15 claim. Per-item scores and category regressions
are retained in `results/`; no favorable categories or seeds were selected.

## Inspect or run

Inspect the recovered files and optional local inputs on CPU:

```bash
python scripts/recovered_verifiers.py check
python scripts/recovered_verifiers.py check --data-root /path/to/historical-inputs
python scripts/recovered_verifiers.py mine --dry-run \
  --family replace --use-vlm --sources /path/to/unlabeled-images \
  --out /path/to/new-mining-run --n-samples 4 --steps 28 --reward-gate 0.12
python scripts/recovered_verifiers.py train --dry-run \
  --data-root /path/to/historical-inputs --output /path/to/new-training-run
```

`--data-root` must contain the exact relative source/target paths listed in the
recovered manifest. It is not the fixed extraction bundle's root unless the
additional multi-family inputs have also been installed. Training refuses
recipe overrides and existing output directories.

Model commands require the owned RUNNING Slurm GPU allocation, numeric step,
activated environment and inspected tmux context described in
[TRAINING.md](TRAINING.md). Prepare and verify inputs first. From inside that
prepared GPU step, remove `--dry-run` to mine or train. The wrapper checks
scheduler ownership before importing model code and applies the existing loader
compatibility shim. No model execution was performed as part of code recovery.

For the old ranker analysis, preserve its expected benchmark data and both
candidate-image/score sets before running:

```bash
python scripts/recovered_verifiers.py rank --dry-run --verify --logprob
```

Its default paths refer to historical candidate sets. The source config and
command are available; those image sets are not bundled in Git. The ranker
script's scored benchmark data must remain evaluation-only.

New runs must record actual model revisions, input hashes, completion receipts,
adapter hashes and matched evaluation identities. The old training record has
no pinned model revision, and a rebuild is not guaranteed to reproduce its
weights or paper numbers.
