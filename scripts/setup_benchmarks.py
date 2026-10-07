#!/usr/bin/env python3
"""Fetch pinned scorer repositories into fresh directories; never update existing ones."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    lock = json.loads((ROOT / "configs/eval/scorers.json").read_text())
    for item in lock["repositories"]:
        target = ROOT / "data/benchmark_tools" / item["name"]
        if target.exists():
            raise FileExistsError(f"Refusing to change an existing scorer checkout: {target}")
        if item.get("patch"):
            patch = ROOT / item["patch"]
            if hashlib.sha256(patch.read_bytes()).hexdigest() != item["patch_sha256"]:
                raise ValueError(f"Scorer patch hash mismatch: {patch.name}")
        if args.dry_run:
            print(f"{item['name']}: {item['url']} @ {item['revision']}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", item["url"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", item["revision"]], check=True)
        if item.get("patch"):
            subprocess.run(["git", "-C", str(target), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(target), "apply", str(patch)], check=True)


if __name__ == "__main__":
    main()
