from __future__ import annotations

import argparse
import json

from PIL import Image

from qwen_edit_project.eval.evaluation_contract import (
    build_evaluation_contract,
    contracted_model_name,
    contracted_scores_dir,
    evaluation_contract_enabled,
    write_or_validate_contract,
    write_or_validate_output_manifest,
)
from qwen_edit_project.eval.export_provenance import (
    build_edit_export_provenance,
    validate_resume_provenance,
    write_export_provenance,
)
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.prompting import polish_prompt
from qwen_edit_project.utils.qwen_pipeline import load_qwen_edit_pipeline, render_edit, render_edit_batch
from qwen_edit_project.utils.run_metadata import base_run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ImgEdit benchmark images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of ImgEdit prompts to generate per pipeline call.")
    parser.add_argument("--no-resume", action="store_true", help="Regenerate images even when output files already exist.")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    batch_size = max(1, int(args.batch_size))

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    edit_json_path = resolve_path(dataset_cfg["edit_json"])
    origin_root = resolve_path(dataset_cfg["origin_img_root"])
    if edit_json_path is None or origin_root is None:
        raise ValueError("ImgEdit dataset paths must resolve")
    hardening_enabled = evaluation_contract_enabled(config)
    if hardening_enabled and (args.offset or args.limit is not None):
        raise ValueError(
            "Hardened ImgEdit exports require the complete contracted dataset; "
            "do not use --offset or --limit."
        )
    if hardening_enabled:
        contracted_device = config.get("runtime", {}).get("generation_device")
        if contracted_device is None:
            raise ValueError(
                "Hardened ImgEdit evaluation requires runtime.generation_device"
            )
        if str(args.device) != str(contracted_device):
            raise ValueError(
                "ImgEdit --device must match the contracted runtime.generation_device: "
                f"{args.device!r} != {contracted_device!r}"
            )
        contracted_batch_size = config.get("generation", {}).get("export_batch_size")
        if contracted_batch_size is None:
            raise ValueError(
                "Hardened ImgEdit evaluation requires generation.export_batch_size"
            )
        if batch_size != int(contracted_batch_size):
            raise ValueError(
                "ImgEdit --batch-size must match the contracted generation.export_batch_size: "
                f"{batch_size} != {contracted_batch_size}"
            )
    with edit_json_path.open("r", encoding="utf-8") as handle:
        if hardening_enabled:
            def reject_duplicate_pairs(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError(f"Duplicate JSON key {key!r} in {edit_json_path}")
                    value[key] = item
                return value

            edit_specs = json.load(handle, object_pairs_hook=reject_duplicate_pairs)
        else:
            edit_specs = json.load(handle)

    items = list(edit_specs.items())
    if args.offset:
        items = items[args.offset :]
    if args.limit is not None:
        items = items[: args.limit]

    output_base = ensure_dir(resolve_path(config["output"]["edited_images_dir"]))
    summary_path = resolve_path(config["output"]["summary_path"])
    if summary_path is None:
        raise ValueError("output.summary_path must resolve")
    contract = None
    run_model_name = model_cfg["model_name"]
    if hardening_enabled:
        records = [
            {
                "identity": str(key),
                "key": str(key),
                "edit_type": str(item.get("edit_type", "")),
                "source_id": str(item.get("id", "")),
            }
            for key, item in edit_specs.items()
        ]
        contract = build_evaluation_contract(config, benchmark="imgedit", records=records)
        run_model_name = contracted_model_name(
            model_cfg["model_name"],
            contract,
            length=int(config["evaluation_contract"].get("output_id_length", 16)),
        )
        summary_dir = contracted_scores_dir(summary_path.parent, contract)
        write_or_validate_contract(summary_dir, contract)
        actual_summary_path = summary_dir / f"{run_model_name}_export_summary.json"
    else:
        actual_summary_path = summary_path.parent / f"{model_cfg['model_name']}_summary.json"
    output_root = output_base / run_model_name
    if contract is not None:
        write_or_validate_contract(output_root, contract)
    export_provenance = build_edit_export_provenance(config)
    validate_resume_provenance(
        benchmark="ImgEdit",
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
    if preserve_input_resolution and batch_size > 1:
        raise ValueError("ImgEdit batched export requires generation.preserve_input_resolution=false")
    written = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []

    def save_one(key: str, item: dict[str, str]) -> None:
        out_path = output_root / f"{key}.png"
        input_image_path = origin_root / item["id"]
        prompt = polish_prompt(
            item["prompt"],
            use_prompt_polish=config.get("prompting", {}).get("use_prompt_polish", False),
        )
        with Image.open(input_image_path) as image:
            if preserve_input_resolution:
                generation["width"], generation["height"] = image.size
            else:
                generation.pop("width", None)
                generation.pop("height", None)
        output = render_edit(pipe, prompt, [input_image_path], generation)
        image = output.images[0] if hasattr(output, "images") else output
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        image.save(tmp_path, format="PNG")
        tmp_path.replace(out_path)

    def maybe_log_progress() -> None:
        done = written + skipped + failed
        if done % int(config["output"].get("progress_every", 25)) == 0:
            print(
                f"ImgEdit export progress: processed={done}/{len(items)} "
                f"written={written} skipped={skipped} failed={failed}",
                flush=True,
            )

    pending: list[tuple[str, dict[str, str]]] = []
    for key, item in items:
        out_path = output_root / f"{key}.png"
        if out_path.exists() and not args.no_resume:
            skipped += 1
        else:
            pending.append((str(key), item))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        if batch_size == 1 or len(batch) == 1:
            for key, item in batch:
                try:
                    save_one(key, item)
                except Exception as exc:
                    failed += 1
                    failures.append({"key": str(key), "error": repr(exc)})
                    print(f"Failed ImgEdit export for {key}: {exc}", flush=True)
                    continue
                written += 1
                maybe_log_progress()
            continue

        keys = [key for key, _ in batch]
        prompts = [
            polish_prompt(
                item["prompt"],
                use_prompt_polish=config.get("prompting", {}).get("use_prompt_polish", False),
            )
            for _, item in batch
        ]
        input_image_paths = [origin_root / item["id"] for _, item in batch]
        try:
            images = render_edit_batch(pipe, prompts, [[path] for path in input_image_paths], generation)
            for key, image in zip(keys, images):
                out_path = output_root / f"{key}.png"
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                image.save(tmp_path, format="PNG")
                tmp_path.replace(out_path)
                written += 1
                maybe_log_progress()
        except Exception as exc:
            print(f"Failed ImgEdit batch {keys}: {exc}; retrying individually.", flush=True)
            for key, item in batch:
                try:
                    save_one(key, item)
                except Exception as item_exc:
                    failed += 1
                    failures.append({"key": str(key), "error": repr(item_exc)})
                    print(f"Failed ImgEdit export for {key}: {item_exc}", flush=True)
                    maybe_log_progress()
                    continue
                written += 1
                maybe_log_progress()

    summary = base_run_metadata()
    summary.update(
        {
            "benchmark": "imgedit",
            "config_path": config["_config_path"],
            "model_name": model_cfg["model_name"],
            "run_model_name": run_model_name,
            "evaluation_contract": contract,
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "records_exported": written,
            "records_skipped_existing": skipped,
            "records_failed": failed,
            "records_requested": len(items),
            "export_batch_size": batch_size,
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
        expected_images = {f"{key}.png" for key, _ in items}
        actual_images = {path.name for path in output_root.glob("*.png")}
        missing_images = sorted(expected_images - actual_images)
        unexpected_images = sorted(actual_images - expected_images)
        if failures or missing_images or unexpected_images:
            raise RuntimeError(
                "Hardened ImgEdit export is incomplete or contaminated: "
                f"failures={len(failures)}, missing={missing_images[:20]}, "
                f"unexpected={unexpected_images[:20]}"
            )
        write_or_validate_output_manifest(
            output_root,
            sorted(expected_images),
            contract_id=str(contract["contract_id"]),
        )
    print(f"Exported {written} ImgEdit images to {output_root}")


if __name__ == "__main__":
    main()
