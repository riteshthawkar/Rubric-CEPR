#!/usr/bin/env python
"""Prepare the Complex-Edit benchmark locally.

Downloads ``UCSC-VLAA/Complex-Edit`` (cached under ``HF_HOME``) and materialises,
for each split (``test_real`` / ``test_syn``):

    data/processed/benchmark/complex_edit/<split>/images/<idx:04d>.png
    data/processed/benchmark/complex_edit/<split>/instructions.jsonl

Each ``instructions.jsonl`` row carries the 8 compound instructions (complexity
``C1``..``C8``) and the atomic ``sequence`` for one input image. Image files are
named by zero-padded dataset index so that ``sorted(glob("*.png"))`` re-aligns
with the HuggingFace dataset order used by the upstream ``eval.py`` scorer.

CPU-only; safe to run on a login node. Idempotent (existing images are skipped).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_SPLITS = {"real": "test_real", "syn": "test_syn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialise the Complex-Edit benchmark locally.")
    parser.add_argument("--repo-id", default="UCSC-VLAA/Complex-Edit")
    parser.add_argument("--revision", default="b0b8a81d740ae413d52572281a23dc975c4b4b91")
    parser.add_argument(
        "--output-root",
        default="data/processed/benchmark/complex_edit",
        help="Destination root for images + instruction manifests.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["real", "syn"],
        choices=sorted(REPO_SPLITS),
        help="Which input-image splits to materialise.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples per split (debug).")
    parser.add_argument("--overwrite-images", action="store_true", help="Re-save images even if they already exist.")
    return parser.parse_args()


def materialise_split(dataset, split: str, out_dir: Path, limit: int | None, overwrite: bool) -> dict:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "instructions.jsonl"

    rows = dataset[REPO_SPLITS[split]]
    total = len(rows) if limit is None else min(limit, len(rows))
    saved = 0
    skipped = 0
    written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for idx in range(total):
            row = rows[idx]
            key = f"{idx:04d}"
            image_path = images_dir / f"{key}.png"
            if image_path.exists() and not overwrite:
                skipped += 1
            else:
                image = row["image"].convert("RGB")
                tmp_path = image_path.with_suffix(".png.tmp")
                image.save(tmp_path, format="PNG")
                tmp_path.replace(image_path)
                saved += 1

            edit = row["edit"]
            compound = [step["compound_instruction"] for step in edit["compound"]]
            if len(compound) != 8:
                raise ValueError(f"{split}[{idx}] expected 8 compound instructions, got {len(compound)}")
            sequence = [step.get("instruction", "") for step in edit.get("sequence", [])]
            manifest.write(
                json.dumps(
                    {"idx": idx, "key": key, "image": f"{key}.png", "compound": compound, "sequence": sequence},
                    ensure_ascii=True,
                )
                + "\n"
            )
            written += 1
            if written % 100 == 0:
                print(f"[{split}] processed {written}/{total} (saved={saved} skipped={skipped})", flush=True)

    return {
        "split": split,
        "repo_split": REPO_SPLITS[split],
        "num_samples": total,
        "images_saved": saved,
        "images_skipped_existing": skipped,
        "images_dir": str(images_dir),
        "instructions_jsonl": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    print(f"Loading {args.repo_id} (this uses the HF cache under HF_HOME) ...", flush=True)
    dataset = load_dataset(args.repo_id, revision=args.revision)
    available = set(dataset.keys())
    for split in args.splits:
        if REPO_SPLITS[split] not in available:
            raise SystemExit(f"Split {REPO_SPLITS[split]!r} not found in dataset; available: {sorted(available)}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in args.splits:
        summary = materialise_split(dataset, split, output_root / split, args.limit, args.overwrite_images)
        print(f"[{split}] done: {summary}", flush=True)
        summaries.append(summary)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps({"repo_id": args.repo_id, "splits": summaries}, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
