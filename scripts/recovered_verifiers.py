#!/usr/bin/env python3
"""Inspect recovered evidence on CPU; guard model execution behind Slurm."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
PREFIX = ROOT / "reproducibility/verifier_bank"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_evidence() -> dict:
    record = json.loads((PREFIX / "source_recovery.json").read_text())
    for entry in record["files"]:
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Recovered source changed: {entry['path']}")
    for entry in record["record_copies"]:
        if digest(ROOT / entry["public_path"]) != entry["public_sha256"]:
            raise ValueError(f"Recovered record changed: {entry['public_path']}")
    for name, expected in record["derived_file_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Derived evidence changed: {name}")
    rows = json.loads((PREFIX / "training_manifest.json").read_text())
    identity = json.loads((PREFIX / "identity.json").read_text())
    if len(rows) != identity["training_rows"] or len(rows) != 149:
        raise ValueError("The recovered multi-family manifest must contain 149 rows")
    return {"recovered_files": len(record["files"]), "record_copies": len(record["record_copies"]),
            "training_rows": len(rows), "file_integrity_verified": True,
            "documentation": "docs/VERIFIER_BANK.md"}


def verify_inputs(data_root: Path) -> None:
    from rubric_cepr.checks import safe_relative_path
    record = json.loads((PREFIX / "training_artifacts.json").read_text())
    rows = json.loads((PREFIX / "training_manifest.json").read_text())
    expected = {(r["family"], str(r["record_key"]), field): r[field]
                for r in rows for field in ("image", "edit_image")}
    seen = set()
    for item in record["artifacts"]:
        key = (item["family"], str(item["record_key"]), item["field"])
        if key in seen or expected.get(key) != item["path"]:
            raise ValueError("Duplicate or unexpected input reference")
        seen.add(key)
        path = safe_relative_path(data_root.resolve(), item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Missing or changed training input: {item['path']}")
    if seen != expected.keys() or len(seen) != 298:
        raise ValueError("Expected all 298 source/target references")


def training_command(data_root: Path, output: Path) -> list[str]:
    args = json.loads((PREFIX / "training_args.json").read_text())
    args.update(dataset_base_path=str(data_root.resolve()),
                dataset_metadata_path=str(PREFIX / "training_manifest.json"),
                output_dir=str(output.resolve()))
    command = ["-m", "qwen_edit_project.train.diffusers_qwen_edit_lora_reproduction_v1"]
    for name, value in args.items():
        # Accelerator discovers process rank from its execution context.
        if name == "local_rank" or value is None or value is False:
            continue
        if value is True:
            command.append(f"--{name}")
        else:
            command.extend((f"--{name}", str(value)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("check", "mine", "rank", "train"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    args, forwarded = parser.parse_known_args()
    integrity = verify_evidence()
    if args.mode == "check":
        if forwarded or args.output:
            parser.error("Unexpected arguments for check")
        if args.data_root:
            verify_inputs(args.data_root)
            integrity["input_references_verified"] = 298
        print(json.dumps(integrity, indent=2))
        return
    if args.mode == "train":
        if not args.data_root or not args.output or forwarded:
            parser.error("train requires --data-root and --output, with no recipe overrides")
        if args.output.exists():
            parser.error("Use a new output directory; historical outputs must remain untouched")
        command = training_command(args.data_root, args.output)
    else:
        if args.data_root or args.output:
            parser.error("Use the original script's --sources and --out arguments")
        script = "build_family_selfdistill.py" if args.mode == "mine" else "validate_judge_ranker.py"
        command = [str(ROOT / "scripts" / script), *forwarded]
        if args.mode == "mine":
            mining = argparse.ArgumentParser(add_help=False)
            mining.add_argument("--out", type=Path, required=True)
            options, _ = mining.parse_known_args(forwarded)
            destination = options.out if options.out.is_absolute() else ROOT / options.out
            if destination.exists():
                parser.error("Mining requires a new --out directory")
    if args.dry_run:
        print(json.dumps({**integrity, "command": [sys.executable, *command], "model_loaded": False}, indent=2))
        return

    # Imports of model code occur only after scheduler/environment verification.
    from rubric_cepr.runtime import require_gpu_step
    require_gpu_step()
    if args.mode == "train":
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("SLURM_NTASKS", "1")) != 1:
            raise RuntimeError("The recovered recipe requires one process")
        verify_inputs(args.data_root)
        args.output.mkdir(parents=True, exist_ok=False)
        sys.argv = [command[1], *command[2:]]
        runpy.run_module(command[1], run_name="__main__")
        completion = json.loads((args.output / "training_completion.json").read_text())
        if completion.get("status") != "complete" or completion.get("global_step") != 500:
            raise RuntimeError("Recovered training did not complete all 500 updates")
    else:
        from rubric_cepr.cli import patch_inference_processor
        patch_inference_processor()
        os.chdir(ROOT)
        sys.argv = command
        runpy.run_path(command[0], run_name="__main__")


if __name__ == "__main__":
    main()
