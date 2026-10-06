from __future__ import annotations

import argparse
import json
from typing import Any

from qwen_edit_project.eval.evaluation_contract import (
    build_evaluation_contract,
    contracted_model_name,
    contracted_scores_dir,
    evaluation_contract_enabled,
    write_or_validate_contract,
    write_or_validate_output_manifest,
)
from qwen_edit_project.eval.gedit_selection import (
    GEditSelectionError,
    apply_gedit_selection,
    decoded_source_image_fingerprint,
    gedit_judge_jpeg_fingerprint,
    selection_manifest,
    write_selection_manifest,
)
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.eval.export_provenance import (
    build_edit_export_provenance,
    validate_resume_provenance,
    write_export_provenance,
)
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.prompting import polish_prompt
from qwen_edit_project.utils.qwen_pipeline import load_qwen_edit_pipeline, render_edit
from qwen_edit_project.utils.run_metadata import base_run_metadata


def _load_gedit_dataset(config: dict[str, Any]):
    dataset_cfg = config["dataset"]
    source = dataset_cfg.get("source", "huggingface")
    if source == "huggingface":
        from datasets import load_dataset

        dataset = load_dataset(
            dataset_cfg["dataset_name"],
            split=dataset_cfg.get("split", "train"),
            revision=dataset_cfg.get("revision"),
        )
    elif source == "disk":
        from datasets import load_from_disk

        local_path = resolve_path(dataset_cfg["local_path"])
        if local_path is None:
            raise ValueError("dataset.local_path is required when source=disk")
        dataset = load_from_disk(str(local_path))
    else:
        raise ValueError(f"Unsupported GEdit dataset source: {source}")
    return dataset


def _gedit_record_allowed(item: dict[str, Any], dataset_cfg: dict[str, Any]) -> bool:
    language_filter = dataset_cfg.get("instruction_language", "all")
    task_filter = dataset_cfg.get("task_type", "all")
    if language_filter != "all" and item["instruction_language"] != language_filter:
        return False
    if task_filter != "all" and item["task_type"] != task_filter:
        return False
    return True


def load_gedit_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_cfg = config["dataset"]
    dataset = _load_gedit_dataset(config)

    records: list[dict[str, Any]] = []
    for item in dataset:
        if not _gedit_record_allowed(item, dataset_cfg):
            continue
        records.append(item)
    return records


def load_selected_gedit_records(
    config: dict[str, Any],
    *,
    decode_images: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the exact selection while decoding only images that will be exported."""

    dataset_cfg = config["dataset"]
    dataset = _load_gedit_dataset(config)
    metadata_columns = ["key", "task_type", "instruction_language", "instruction"]
    missing_columns = [column for column in metadata_columns if column not in dataset.column_names]
    if missing_columns:
        raise KeyError(f"GEdit dataset is missing required columns: {missing_columns}")
    metadata_dataset = dataset.select_columns(metadata_columns)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(metadata_dataset):
        if not _gedit_record_allowed(item, dataset_cfg):
            continue
        records.append({**item, "__dataset_index": index})
    selection_cfg = config["dataset"].get("selection", {})
    selected, manifest = apply_gedit_selection(records, selection_cfg)
    if not decode_images:
        return selected, manifest
    bind_decoded_sources = evaluation_contract_enabled(config)
    decoded: list[dict[str, Any]] = []
    for record in selected:
        item = dict(dataset[int(record["__dataset_index"])])
        for field in metadata_columns:
            if str(item.get(field, "")) != str(record[field]):
                raise GEditSelectionError(
                    "GEdit dataset row changed between metadata selection and image decode: "
                    f"index={record['__dataset_index']}, field={field}"
                )
        if bind_decoded_sources:
            if "input_image_raw" not in item:
                raise GEditSelectionError(
                    f"GEdit dataset row {record['__dataset_index']} has no input_image_raw"
                )
            item["__source_image_fingerprint"] = decoded_source_image_fingerprint(
                item["input_image_raw"]
            )
            item["__source_judge_image_fingerprint"] = gedit_judge_jpeg_fingerprint(
                item["input_image_raw"]
            )
        decoded.append(item)
    if bind_decoded_sources:
        manifest = selection_manifest(
            decoded,
            mode=str(manifest["mode"]),
            seed=manifest.get("seed"),
            per_language_per_task=manifest.get("per_language_per_task"),
            languages=manifest.get("languages"),
        )
    return decoded, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GEdit benchmark images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true", help="Regenerate images even when output files already exist.")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    hardening_enabled = evaluation_contract_enabled(config)
    if hardening_enabled:
        contracted_device = config.get("runtime", {}).get("generation_device")
        if contracted_device is None:
            raise ValueError(
                "Hardened GEdit evaluation requires runtime.generation_device"
            )
        if str(args.device) != str(contracted_device):
            raise ValueError(
                "GEdit --device must match the contracted runtime.generation_device: "
                f"{args.device!r} != {contracted_device!r}"
            )
    selection_mode = str(config["dataset"].get("selection", {}).get("mode", "all"))
    if selection_mode not in {"all", "none"} and not hardening_enabled:
        raise ValueError(
            "A GEdit subset must use evaluation_contract.enabled=true so it cannot be mixed "
            "with a full or differently sampled run."
        )
    if hardening_enabled and config["dataset"].get("source", "huggingface") == "huggingface":
        if not config["dataset"].get("revision"):
            raise ValueError("Hardened Hugging Face GEdit evaluation requires dataset.revision")
    if hardening_enabled and (args.offset or args.limit is not None):
        raise ValueError(
            "Hardened GEdit exports use the exact contracted selection; "
            "do not combine them with --offset or --limit."
        )
    records, selection = load_selected_gedit_records(config)
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    model_cfg = config["model"]
    model_name = model_cfg["model_name"]
    output_base = ensure_dir(resolve_path(config["output"]["edited_images_dir"]))
    summary_path = resolve_path(config["output"]["summary_path"])
    if summary_path is None:
        raise ValueError("output.summary_path must resolve")
    contract = None
    run_model_name = model_name
    if hardening_enabled:
        contract = build_evaluation_contract(
            config,
            benchmark="gedit",
            records=selection["records"],
        )
        run_model_name = contracted_model_name(
            model_name,
            contract,
            length=int(config["evaluation_contract"].get("output_id_length", 16)),
        )
        summary_dir = contracted_scores_dir(summary_path.parent, contract)
        write_or_validate_contract(summary_dir, contract)
        actual_summary_path = summary_dir / f"{run_model_name}_export_summary.json"
    else:
        actual_summary_path = summary_path.parent / f"{model_name}_summary.json"
    output_root = output_base / run_model_name / "fullset"
    if contract is not None:
        write_or_validate_contract(output_root, contract)
    write_selection_manifest(output_root / ".gedit_selection_manifest.json", selection)
    export_provenance = build_edit_export_provenance(config)
    export_provenance["dataset_selection"] = {
        key: value for key, value in selection.items() if key != "records"
    }
    validate_resume_provenance(
        benchmark="GEdit",
        output_root=output_root,
        summary_path=actual_summary_path,
        expected=export_provenance,
        no_resume=args.no_resume,
        allow_mismatch=bool(config["output"].get("allow_resume_mismatch", False)),
    )
    ensure_dir(output_root)
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

    generation = dict(config["generation"])
    preserve_input_resolution = bool(generation.get("preserve_input_resolution", True))
    written = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []
    for item in records:
        out_dir = ensure_dir(output_root / item["task_type"] / item["instruction_language"])
        out_path = out_dir / f"{item['key']}.png"
        if out_path.exists() and not args.no_resume:
            skipped += 1
            continue
        prompt = polish_prompt(
            item["instruction"],
            use_prompt_polish=config.get("prompting", {}).get("use_prompt_polish", False),
            image_context=item["input_image_raw"],
        )
        if preserve_input_resolution:
            generation["width"], generation["height"] = item["input_image_raw"].size
        else:
            generation.pop("width", None)
            generation.pop("height", None)
        try:
            output = render_edit(pipe, prompt, [item["input_image_raw"]], generation)
            image = output.images[0] if hasattr(output, "images") else output
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            image.save(tmp_path, format="PNG")
            tmp_path.replace(out_path)
        except Exception as exc:
            failed += 1
            failures.append({"key": str(item["key"]), "error": repr(exc)})
            print(f"Failed GEdit export for {item['key']}: {exc}", flush=True)
            continue
        written += 1
        done = written + skipped + failed
        if done % int(config["output"].get("progress_every", 25)) == 0:
            print(
                f"GEdit export progress: processed={done}/{len(records)} "
                f"written={written} skipped={skipped} failed={failed}",
                flush=True,
            )

    summary = base_run_metadata()
    summary.update(
        {
            "benchmark": "gedit",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "run_model_name": run_model_name,
            "evaluation_contract": contract,
            "dataset_selection": selection,
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "records_exported": written,
            "records_skipped_existing": skipped,
            "records_failed": failed,
            "records_requested": len(records),
            "output_root": str(output_root),
            "failures": failures,
            "export_provenance": export_provenance,
        }
    )
    save_json(summary, actual_summary_path)
    if failures:
        failure_path = actual_summary_path.parent / f"{run_model_name}_export_failures.jsonl"
        with failure_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=True) + "\n")
    if hardening_enabled:
        expected_images = {
            f"{item['task_type']}/{item['instruction_language']}/{item['key']}.png"
            for item in records
        }
        actual_images = {
            path.relative_to(output_root).as_posix()
            for path in output_root.rglob("*.png")
        }
        missing_images = sorted(expected_images - actual_images)
        unexpected_images = sorted(actual_images - expected_images)
        if failures or missing_images or unexpected_images:
            raise RuntimeError(
                "Hardened GEdit export is incomplete or contaminated: "
                f"failures={len(failures)}, missing={missing_images[:20]}, "
                f"unexpected={unexpected_images[:20]}"
            )
        write_or_validate_output_manifest(
            output_root,
            sorted(expected_images),
            contract_id=str(contract["contract_id"]),
        )
    print(f"Exported {written} GEdit images to {output_root}")


if __name__ == "__main__":
    main()
