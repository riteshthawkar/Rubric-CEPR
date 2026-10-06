from __future__ import annotations

import re
import math
from typing import Any, Mapping

from qwen_edit_project.eval.evaluation_contract import canonical_json_bytes, sha256_bytes


IMGEDIT_SCORE_LABELS: dict[str, tuple[str, str, str]] = {
    "replace": ("Prompt Compliance", "Visual Naturalness", "Physical & Detail Integrity"),
    "add": ("Prompt Compliance", "Visual Naturalness", "Physical & Detail Coherence"),
    "adjust": ("Prompt Compliance", "Visual Seamlessness", "Physical & Detail Fidelity"),
    "remove": ("Prompt Compliance", "Visual Naturalness", "Physical & Detail Integrity"),
    "style": ("Style Fidelity", "Content Preservation", "Rendering Quality"),
    "action": ("Action Fidelity", "Identity Preservation", "Visual & Anatomical Coherence"),
    "extract": ("Object Identity", "Mask Precision", "Visual Quality"),
    "background": ("Instruction Compliance", "Visual Seamlessness", "Physical Consistency"),
    "compose": (
        "Instruction Compliance",
        "Visual Naturalness",
        "Physical Consistency & Fine Detail",
    ),
}


class ImgEditScoreParseError(ValueError):
    pass


def parse_imgedit_scores(entry: object, edit_type: str) -> dict[str, int]:
    """Parse exactly the three official labels and reject malformed/duplicate scores."""

    if not isinstance(entry, str):
        raise ImgEditScoreParseError("Judge response is not text")
    try:
        labels = IMGEDIT_SCORE_LABELS[edit_type]
    except KeyError as exc:
        raise ImgEditScoreParseError(f"Unknown ImgEdit edit_type: {edit_type!r}") from exc

    parsed: dict[str, int] = {}
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    pattern = re.compile(rf"^\s*({label_pattern})\s*:\s*([+-]?\d+(?:\.\d+)?)\s*(?:/\s*5)?\s*\.?\s*$")
    generic_score_pattern = re.compile(
        r"^\s*([^:]+)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*(?:/\s*5)?\s*\.?\s*$"
    )
    for line in entry.splitlines():
        match = pattern.fullmatch(line)
        if match is None:
            unexpected = generic_score_pattern.fullmatch(line)
            if unexpected is not None:
                raise ImgEditScoreParseError(
                    f"Unexpected numeric score label: {unexpected.group(1).strip()}"
                )
            continue
        label, raw_score = match.groups()
        if label in parsed:
            raise ImgEditScoreParseError(f"Duplicate score label: {label}")
        if not re.fullmatch(r"\d+", raw_score):
            raise ImgEditScoreParseError(f"Score for {label} must be an integer, got {raw_score!r}")
        score = int(raw_score)
        if not 1 <= score <= 5:
            raise ImgEditScoreParseError(f"Score for {label} is outside [1, 5]: {score}")
        parsed[label] = score

    missing = [label for label in labels if label not in parsed]
    if missing:
        raise ImgEditScoreParseError(f"Missing exact score label(s): {missing}")
    if len(parsed) != len(labels):
        raise ImgEditScoreParseError(f"Expected exactly {len(labels)} scores, found {len(parsed)}")
    return {label: parsed[label] for label in labels}


def strict_imgedit_average(entry: object, edit_type: str) -> float:
    scores = parse_imgedit_scores(entry, edit_type)
    return sum(scores.values()) / len(scores)


def legacy_imgedit_average(entry: object) -> float | None:
    if not isinstance(entry, str):
        return None
    scores = []
    for line in entry.splitlines():
        parts = line.strip().split(": ")
        if len(parts) == 2 and parts[1].isdigit():
            scores.append(int(parts[1]))
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def extract_imgedit_average(entry: Any, *, edit_type: str | None = None, strict: bool = False) -> float | None:
    if not strict:
        return legacy_imgedit_average(entry)
    if edit_type is None:
        raise ValueError("edit_type is required for strict ImgEdit score parsing")
    try:
        return strict_imgedit_average(entry, edit_type)
    except ImgEditScoreParseError:
        return None


def build_imgedit_record_score_manifest(
    edit_specs: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, float],
) -> dict[str, Any]:
    """Build exact per-record evidence for paired comparison and audit."""

    expected = {str(key) for key in edit_specs}
    actual = {str(key) for key in scores}
    if expected != actual:
        raise ValueError(
            "ImgEdit record-score key set differs from edit specs: "
            f"missing={sorted(expected - actual)[:20]}, unexpected={sorted(actual - expected)[:20]}"
        )
    rows: list[dict[str, Any]] = []
    for raw_identity, spec in edit_specs.items():
        identity = str(raw_identity)
        family = str(spec.get("edit_type", "")).strip()
        if family not in IMGEDIT_SCORE_LABELS:
            raise ValueError(f"Unknown ImgEdit family for {identity}: {family!r}")
        score = float(scores[identity])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite ImgEdit score for {identity}: {score!r}")
        if not 1.0 <= score <= 5.0:
            raise ValueError(f"ImgEdit score for {identity} is outside [1, 5]: {score!r}")
        rows.append({"identity": identity, "family": family, "score": score})
    rows.sort(key=lambda row: row["identity"])
    return {
        "record_scores": rows,
        "record_scores_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }
