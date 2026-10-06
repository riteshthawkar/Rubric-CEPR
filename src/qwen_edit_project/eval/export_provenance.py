from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_edit_project.utils.config import save_json


def build_edit_export_provenance(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    generation_cfg = config["generation"]
    scoring_cfg = config.get("scoring", {})
    return {
        "model_backend": model_cfg.get("backend", "diffsynth"),
        "base_model": model_cfg.get("base_model"),
        "model_type": model_cfg.get("model_type", "base"),
        "checkpoint_path": model_cfg.get("checkpoint_path"),
        "lora_scale": model_cfg.get("lora_scale"),
        "torch_dtype": model_cfg.get("torch_dtype", "auto"),
        "generation": {
            "num_inference_steps": generation_cfg.get("num_inference_steps"),
            "true_cfg_scale": generation_cfg.get("true_cfg_scale"),
            "guidance_scale": generation_cfg.get("guidance_scale"),
            "negative_prompt": generation_cfg.get("negative_prompt"),
            "num_images_per_prompt": generation_cfg.get("num_images_per_prompt"),
            "preserve_input_resolution": generation_cfg.get("preserve_input_resolution", True),
        },
        "scoring": {
            "backbone": scoring_cfg.get("backbone"),
            "expected_openai_model": scoring_cfg.get("expected_openai_model"),
        },
    }


def build_generation_export_provenance(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    generation_cfg = config["generation"]
    return {
        "base_model": model_cfg.get("base_model"),
        "model_type": model_cfg.get("model_type", "base"),
        "checkpoint_path": model_cfg.get("checkpoint_path"),
        "torch_dtype": model_cfg.get("torch_dtype", "auto"),
        "generation": {
            "width": generation_cfg.get("width"),
            "height": generation_cfg.get("height"),
            "seed": generation_cfg.get("seed"),
            "seed_stride": generation_cfg.get("seed_stride"),
            "samples_per_prompt": generation_cfg.get("samples_per_prompt"),
            "grid_rows": generation_cfg.get("grid_rows"),
            "grid_cols": generation_cfg.get("grid_cols"),
            "num_inference_steps": generation_cfg.get("num_inference_steps"),
            "true_cfg_scale": generation_cfg.get("true_cfg_scale"),
            "guidance_scale": generation_cfg.get("guidance_scale"),
            "negative_prompt": generation_cfg.get("negative_prompt"),
            "fill_with_black_on_error": generation_cfg.get("fill_with_black_on_error"),
        },
    }


def export_provenance_path(output_root: Path) -> Path:
    return output_root / ".export_provenance.json"


def write_export_provenance(output_root: Path, provenance: dict[str, Any]) -> None:
    save_json(provenance, export_provenance_path(output_root))


def _existing_output_count(output_root: Path, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns for _ in output_root.rglob(pattern))


def _load_previous_provenance(summary_path: Path, output_root: Path) -> dict[str, Any] | None:
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        return previous.get("export_provenance")

    sidecar_path = export_provenance_path(output_root)
    if sidecar_path.exists():
        with sidecar_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def validate_resume_provenance(
    *,
    benchmark: str,
    output_root: Path,
    summary_path: Path,
    expected: dict[str, Any],
    no_resume: bool,
    allow_mismatch: bool = False,
    existing_file_patterns: tuple[str, ...] = ("*.png",),
) -> None:
    if no_resume or allow_mismatch or not output_root.exists():
        return
    existing_count = _existing_output_count(output_root, existing_file_patterns)
    if existing_count == 0:
        return

    previous_provenance = _load_previous_provenance(summary_path, output_root)
    if previous_provenance is None:
        raise RuntimeError(
            f"{benchmark} output directory already contains {existing_count} generated file(s), but no "
            f"provenance summary exists at {summary_path} and no sidecar exists at "
            f"{export_provenance_path(output_root)}. Use a new model.model_name, delete the old output "
            "directory, or rerun export with --no-resume to regenerate with the current settings."
        )
    if previous_provenance != expected:
        save_json(
            {"expected": expected, "previous": previous_provenance},
            summary_path.with_name(summary_path.stem + "_provenance_mismatch.json"),
        )
        raise RuntimeError(
            f"{benchmark} output directory already contains {existing_count} generated file(s) from "
            "different or unknown settings. Refusing to silently mix paper-matched outputs with older "
            f"images. Inspect {summary_path.with_name(summary_path.stem + '_provenance_mismatch.json')}, "
            "then use a new model.model_name, delete the old output directory, or pass --no-resume."
        )
