from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from qwen_edit_project.utils.config import save_json
from qwen_edit_project.utils.paths import resolve_path

GEDIT_GROUPS = [
    "background_change",
    "color_alter",
    "material_alter",
    "motion_change",
    "ps_human",
    "style_change",
    "subject-add",
    "subject-remove",
    "subject-replace",
    "text_change",
    "tone_transfer",
]


def summarize_gedit(score_root: Path, model_name: str, backbone: str) -> dict:
    backbone_root = score_root / model_name / backbone
    summary = {"groups": {}, "backbone": backbone}
    semantics = []
    quality = []
    overall = []
    for group in GEDIT_GROUPS:
        candidate_paths = [
            backbone_root / f"{model_name}_{group}_all_vie_score.csv",
            backbone_root / f"{model_name}_{group}_en_vie_score.csv",
            backbone_root / f"{model_name}_{group}_cn_vie_score.csv",
        ]
        csv_path = next((path for path in candidate_paths if path.exists()), None)
        if csv_path is None:
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        rows.sort(
            key=lambda row: (
                str(row.get("instruction_language", "")),
                str(row.get("key", "")),
            )
        )
        group_semantics = [float(row["sementics_score"]) for row in rows]
        group_quality = [float(row["quality_score"]) for row in rows]
        group_overall = [math.sqrt(s * q) for s, q in zip(group_semantics, group_quality)]
        summary["groups"][group] = {
            "semantics": math.fsum(group_semantics) / len(group_semantics),
            "quality": math.fsum(group_quality) / len(group_quality),
            "overall": math.fsum(group_overall) / len(group_overall),
            "count": len(rows),
        }
        semantics.extend(group_semantics)
        quality.extend(group_quality)
        overall.extend(group_overall)
    if semantics:
        summary["count"] = len(overall)
        summary["average"] = {
            "semantics": math.fsum(semantics) / len(semantics),
            "quality": math.fsum(quality) / len(quality),
            "overall": math.fsum(overall) / len(overall),
        }
    return summary


def summarize_imgedit(score_root: Path, model_name: str) -> dict:
    average_path = score_root / f"{model_name}_average_score.json"
    types_path = score_root / f"{model_name}_typescore.json"
    summary = {}
    if average_path.exists():
        with average_path.open("r", encoding="utf-8") as handle:
            avg_scores = json.load(handle)
        summary["count"] = len(avg_scores)
        if avg_scores:
            summary["overall_average"] = math.fsum(avg_scores.values()) / len(avg_scores)
    if types_path.exists():
        with types_path.open("r", encoding="utf-8") as handle:
            summary["type_scores"] = json.load(handle)
    return summary


def summarize_geneval(result_path: Path) -> dict:
    records = []
    with result_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    summary: dict[str, object] = {
        "image_count": len(records),
        "prompt_count": 0,
        "tasks": {},
    }
    if not records:
        return summary

    prompt_groups: dict[str, list[bool]] = defaultdict(list)
    task_groups: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        correct = bool(record.get("correct", False))
        prompt_key = str(record.get("metadata", record.get("prompt", "")))
        prompt_groups[prompt_key].append(correct)
        task_groups[str(record.get("tag", "unknown"))].append(correct)

    task_scores = []
    for task, values in task_groups.items():
        rate = sum(values) / len(values)
        task_scores.append(rate)
        summary["tasks"][task] = {"correct_rate": rate, "count": len(values)}

    summary["prompt_count"] = len(prompt_groups)
    summary["image_correct_rate"] = sum(bool(item.get("correct", False)) for item in records) / len(records)
    summary["prompt_correct_rate"] = (
        sum(any(values) for values in prompt_groups.values()) / len(prompt_groups)
    )
    summary["task_average"] = sum(task_scores) / len(task_scores)
    return summary


def summarize_dpgbench(result_path: Path) -> dict:
    summary: dict[str, object] = {"image_count": 0, "l1_categories": {}, "l2_categories": {}}
    image_scores: list[float] = []
    section: str | None = None
    overall_pattern = re.compile(r"DPG-Bench score:\s*([0-9.]+)")

    with result_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            match = overall_pattern.search(line)
            if match:
                summary["overall_score"] = float(match.group(1))
                section = None
                continue
            if line == "L1 category scores:":
                section = "l1"
                continue
            if line == "L2 category scores:":
                section = "l2"
                continue
            if line.startswith("Image path:") or line.startswith("Save results to:"):
                section = None
                continue
            if line.startswith("Model:"):
                summary["model_label"] = line.split(":", 1)[1].strip()
                section = None
                continue
            if section in {"l1", "l2"} and ":" in line:
                key, value = line.split(":", 1)
                target = summary["l1_categories"] if section == "l1" else summary["l2_categories"]
                target[key.strip()] = float(value.strip())
                continue
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 2:
                continue
            try:
                image_scores.append(float(parts[-1]))
            except ValueError:
                continue

    summary["image_count"] = len(image_scores)
    if image_scores:
        summary["image_average"] = sum(image_scores) / len(image_scores)
    return summary


def _latest_matching_file(score_root: Path, pattern: str) -> Path | None:
    matches = sorted(score_root.glob(pattern), key=lambda path: path.stat().st_mtime_ns)
    return matches[-1] if matches else None


def _coerce_scalar(value: str | None) -> object:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "" or stripped.lower() == "nan":
        return None
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _read_model_row(csv_path: Path, model_name: str) -> dict[str, object]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    target_row = None
    for row in rows:
        row_name = row.get("") or row.get("Unnamed: 0")
        if row_name == model_name:
            target_row = row
            break
    if target_row is None:
        if len(rows) == 1:
            target_row = rows[0]
        else:
            raise KeyError(f"Model {model_name} not found in {csv_path}")

    payload: dict[str, object] = {}
    for key, value in target_row.items():
        if key in {"", "Unnamed: 0"}:
            continue
        payload[key] = _coerce_scalar(value)
    return payload


def summarize_oneig(score_root: Path, model_name: str) -> dict:
    summary: dict[str, object] = {"files": {}}
    file_specs = {
        "alignment": "alignment_score_*.csv",
        "text": "text_score_*.csv",
        "diversity": "diversity_score_*.csv",
        "style": "style_score_*.csv",
        "reasoning": "reasoning_score_*.csv",
    }

    metric_values: list[float] = []
    for metric, pattern in file_specs.items():
        csv_path = _latest_matching_file(score_root, pattern)
        if csv_path is None:
            continue
        summary["files"][metric] = str(csv_path)
        row = _read_model_row(csv_path, model_name)
        summary[metric] = row
        if metric == "text" and isinstance(row.get("text score"), float):
            metric_values.append(row["text score"])
        elif metric == "diversity" and isinstance(row.get("total average"), float):
            metric_values.append(row["total average"])
        else:
            first_value = next((value for value in row.values() if isinstance(value, float)), None)
            if first_value is not None:
                metric_values.append(first_value)

    if metric_values:
        summary["core_average"] = sum(metric_values) / len(metric_values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GEdit or ImgEdit score outputs.")
    parser.add_argument(
        "--benchmark",
        choices=["gedit", "imgedit", "geneval", "dpgbench", "oneig"],
        required=True,
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--backbone", default="gpt4o")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    score_root = resolve_path(args.score_root)
    if score_root is None:
        raise ValueError("score root could not be resolved")
    if args.benchmark == "gedit":
        summary = summarize_gedit(score_root, args.model_name, args.backbone)
    elif args.benchmark == "imgedit":
        summary = summarize_imgedit(score_root, args.model_name)
    elif args.benchmark == "geneval":
        result_path = score_root if score_root.is_file() else score_root / f"{args.model_name}_results.jsonl"
        summary = summarize_geneval(result_path)
    elif args.benchmark == "dpgbench":
        result_path = score_root if score_root.is_file() else score_root / f"{args.model_name}_results.txt"
        summary = summarize_dpgbench(result_path)
    else:
        summary = summarize_oneig(score_root, args.model_name)
    summary["benchmark"] = args.benchmark
    summary["model_name"] = args.model_name
    save_json(summary, args.output)


if __name__ == "__main__":
    main()
