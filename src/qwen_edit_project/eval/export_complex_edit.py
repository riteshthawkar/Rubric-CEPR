from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from PIL import Image

from qwen_edit_project.eval.complex_edit_contract import (
    ComplexEditContractError,
    load_hardened_c4_dataset,
    require_hardened_experiment_runtime,
    require_fresh_export_namespace,
    validate_contract_still_current,
    validate_exact_export,
    validate_hardened_config,
    write_new_json,
)
from qwen_edit_project.eval.evaluation_contract import (
    build_evaluation_contract,
    contracted_model_name,
    contracted_scores_dir,
    evaluation_contract_enabled,
    sha256_file,
    write_or_validate_contract,
    write_or_validate_output_manifest,
)
from qwen_edit_project.eval.export_provenance import (
    build_edit_export_provenance,
    export_provenance_path,
    validate_resume_provenance,
    write_export_provenance,
)
from qwen_edit_project.utils.config import (
    load_yaml_config,
    merge_override,
    parse_override,
    save_json,
)
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.prompting import polish_prompt
from qwen_edit_project.utils.qwen_pipeline import load_qwen_edit_pipeline, render_edit
from qwen_edit_project.utils.run_metadata import base_run_metadata


def load_manifest(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _save_png_no_replace(image: Image.Image, output_path: Path) -> None:
    """Publish one generated PNG atomically without replacing a raced/stale file."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
    except FileExistsError as exc:
        raise ComplexEditContractError(
            f"Refusing to replace concurrently created export: {output_path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Complex-Edit benchmark images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N samples per complexity."
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--complexity",
        type=int,
        action="append",
        default=None,
        help="Override the complexity levels to export (repeatable). Defaults to dataset.complexities.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Regenerate images even when output files already exist.",
    )
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    hardening_enabled = evaluation_contract_enabled(config)
    if hardening_enabled:
        validate_hardened_config(config)
        require_hardened_experiment_runtime(
            config,
            workload="hardened Complex-Edit real/C4/531 export",
        )
        if args.device != config["runtime"]["generation_device"]:
            raise ValueError(
                "Hardened Complex-Edit --device must exactly match "
                f"runtime.generation_device={config['runtime']['generation_device']}"
            )
        if args.offset or args.limit is not None:
            raise ValueError(
                "Hardened Complex-Edit requires the complete real/C4/531 dataset; "
                "do not use --offset or --limit."
            )
        if args.no_resume:
            raise ValueError(
                "Hardened Complex-Edit never overwrites or resumes generated images; "
                "use a fresh model name/run namespace."
            )
    split = str(dataset_cfg["split"])
    manifest_path = resolve_path(dataset_cfg["manifest"])
    images_root = resolve_path(dataset_cfg["images_root"])
    if manifest_path is None or images_root is None:
        raise ValueError("Complex-Edit dataset.manifest and dataset.images_root must resolve")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Complex-Edit manifest not found: {manifest_path}. "
            "Run: python scripts/prepare_complex_edit_data.py --splits real syn"
        )

    complexities = (
        args.complexity if args.complexity else list(dataset_cfg.get("complexities", [1, 4, 8]))
    )
    complexities = [int(c) for c in complexities]
    for c in complexities:
        if not 1 <= c <= 8:
            raise ValueError(f"complexity must be in 1..8, got {c}")
    if hardening_enabled and complexities != [4]:
        raise ValueError("Hardened Complex-Edit accepts exactly one --complexity 4")

    contract_records = None
    if hardening_enabled:
        records, contract_records = load_hardened_c4_dataset(
            manifest_path,
            images_root,
        )
    else:
        records = load_manifest(manifest_path)
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    output_base = ensure_dir(resolve_path(config["output"]["edited_images_dir"]))
    summary_path = resolve_path(config["output"]["summary_path"])
    if summary_path is None:
        raise ValueError("output.summary_path must resolve")
    contract = None
    run_model_name = model_cfg["model_name"]
    if hardening_enabled:
        contract = build_evaluation_contract(
            config,
            benchmark="complex_edit",
            records=contract_records,
        )
        run_model_name = contracted_model_name(
            model_cfg["model_name"],
            contract,
            length=int(config["evaluation_contract"].get("output_id_length", 16)),
        )
        summary_dir = contracted_scores_dir(summary_path.parent, contract)
        write_or_validate_contract(summary_dir, contract)
        actual_summary_path = summary_dir / f"{run_model_name}_export_summary.json"
        if actual_summary_path.exists() or actual_summary_path.is_symlink():
            raise ComplexEditContractError(
                "Hardened Complex-Edit export never overwrites a prior summary: "
                f"{actual_summary_path}"
            )
    else:
        actual_summary_path = (
            summary_path.parent / f"{model_cfg['model_name']}_complex_edit_summary.json"
        )
    output_root = output_base / run_model_name
    if contract is not None:
        write_or_validate_contract(output_root, contract)
        require_fresh_export_namespace(output_root)

    export_provenance = build_edit_export_provenance(config)
    export_provenance["complex_edit"] = {"split": split, "complexities": complexities}
    validate_resume_provenance(
        benchmark="Complex-Edit",
        output_root=output_root,
        summary_path=actual_summary_path,
        expected=export_provenance,
        no_resume=args.no_resume,
        allow_mismatch=bool(config["output"].get("allow_resume_mismatch", False)),
    )
    ensure_dir(output_root)
    if hardening_enabled:
        write_new_json(export_provenance_path(output_root), export_provenance)
    else:
        write_export_provenance(output_root, export_provenance)

    pipe = load_qwen_edit_pipeline(
        model_id_with_origin_paths=model_cfg["model_id_with_origin_paths"],
        checkpoint_path=model_cfg.get("checkpoint_path"),
        model_type=model_cfg.get("model_type", "base"),
        device=args.device,
        processor_model_id=model_cfg.get("processor_model_id", "Qwen/Qwen-Image-Edit"),
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        backend=model_cfg.get("backend", "diffsynth"),
        base_model=model_cfg.get("base_model"),
        revision=model_cfg.get("revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        lora_scale=model_cfg.get("lora_scale"),
    )

    generation_base = dict(config["generation"])
    preserve_input_resolution = bool(generation_base.get("preserve_input_resolution", False))
    use_prompt_polish = config.get("prompting", {}).get("use_prompt_polish", False)
    progress_every = int(config["output"].get("progress_every", 25))

    written = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []
    per_complexity: dict[str, dict[str, int]] = {}

    for complexity in complexities:
        out_dir = ensure_dir(output_root / f"{split}_c{complexity}")
        c_written = c_skipped = c_failed = 0
        for record in records:
            key = record["key"]
            out_path = out_dir / f"{key}.png"
            if hardening_enabled and (out_path.exists() or out_path.is_symlink()):
                raise ComplexEditContractError(
                    f"Hardened Complex-Edit refuses pre-existing output: {out_path}"
                )
            if not hardening_enabled and out_path.exists() and not args.no_resume:
                skipped += 1
                c_skipped += 1
                continue
            try:
                instruction = record["compound"][complexity - 1]
                prompt = polish_prompt(instruction, use_prompt_polish=use_prompt_polish)
                input_image_path = images_root / record["image"]
                generation = dict(generation_base)
                with Image.open(input_image_path) as image:
                    if preserve_input_resolution:
                        generation["width"], generation["height"] = image.size
                    else:
                        generation.pop("width", None)
                        generation.pop("height", None)
                output = render_edit(pipe, prompt, [input_image_path], generation)
                result_image = output.images[0] if hasattr(output, "images") else output
                if hardening_enabled:
                    _save_png_no_replace(result_image, out_path)
                else:
                    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                    result_image.save(tmp_path, format="PNG")
                    tmp_path.replace(out_path)
            except Exception as exc:  # noqa: BLE001 - record and continue
                failed += 1
                c_failed += 1
                failures.append(
                    {"complexity": str(complexity), "key": str(key), "error": repr(exc)}
                )
                print(f"Failed Complex-Edit export c{complexity} {key}: {exc}", flush=True)
                continue
            written += 1
            c_written += 1
            done = written + skipped + failed
            if done % progress_every == 0:
                print(
                    f"Complex-Edit export progress: processed={done} "
                    f"written={written} skipped={skipped} failed={failed}",
                    flush=True,
                )
        per_complexity[f"{split}_c{complexity}"] = {
            "written": c_written,
            "skipped_existing": c_skipped,
            "failed": c_failed,
            "output_dir": str(out_dir),
        }
        print(
            f"Complex-Edit c{complexity} done: written={c_written} skipped={c_skipped} failed={c_failed}",
            flush=True,
        )

    output_manifest = None
    if hardening_enabled:
        if failures:
            raise ComplexEditContractError(
                f"Hardened Complex-Edit export failed for {len(failures)} record(s)"
            )
        _, final_contract_records = load_hardened_c4_dataset(
            manifest_path,
            images_root,
        )
        validate_contract_still_current(config, contract, final_contract_records)
        expected_images = validate_exact_export(output_root, records)
        output_manifest_path = write_or_validate_output_manifest(
            output_root,
            expected_images,
            contract_id=str(contract["contract_id"]),
        )
        output_manifest = {
            "path": str(output_manifest_path),
            "sha256": sha256_file(output_manifest_path),
            "count": len(expected_images),
        }

    summary = base_run_metadata()
    summary.update(
        {
            "benchmark": "complex_edit",
            "config_path": config["_config_path"],
            "model_name": model_cfg["model_name"],
            "run_model_name": run_model_name,
            "evaluation_contract": contract,
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "split": split,
            "complexities": complexities,
            "records_per_complexity": len(records),
            "records_exported": written,
            "records_skipped_existing": skipped,
            "records_failed": failed,
            "per_complexity": per_complexity,
            "output_root": str(output_root),
            "failures": failures,
            "export_provenance": export_provenance,
            "export_outputs": output_manifest,
        }
    )
    if hardening_enabled:
        write_new_json(actual_summary_path, summary)
    else:
        save_json(summary, actual_summary_path)
    if failures:
        failure_path = (
            summary_path.parent / f"{model_cfg['model_name']}_complex_edit_export_failures.jsonl"
        )
        with failure_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=True) + "\n")
    print(
        f"Exported {written} Complex-Edit images to {output_root} (complexities={complexities}, split={split})"
    )


if __name__ == "__main__":
    main()
