from __future__ import annotations

import hashlib
import io
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps

from qwen_edit_project.eval.evaluation_contract import canonical_json_bytes, sha256_bytes


class GEditSelectionError(ValueError):
    pass


GEDIT_JUDGE_JPEG_REPRESENTATION = "viescore-resized-jpeg-512-area-v1"


def _validated_source_fingerprint(
    raw: object,
    *,
    label: str,
    representation: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "representation",
        "width",
        "height",
        "bytes",
        "sha256",
    }:
        raise GEditSelectionError(f"{label} fingerprint has the wrong schema")
    width = raw.get("width")
    height = raw.get("height")
    byte_count = raw.get("bytes")
    digest = raw.get("sha256")
    if (
        raw.get("representation") != representation
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise GEditSelectionError(f"{label} fingerprint is invalid")
    if representation == "decoded_rgb8" and byte_count != width * height * 3:
        raise GEditSelectionError(f"{label} decoded RGB byte count is invalid")
    return dict(raw)


def _identity(record: Mapping[str, Any]) -> str:
    return f"{record.get('task_type', '')}::{record.get('instruction_language', '')}::{record.get('key', '')}"


def decoded_source_image_fingerprint(image: object) -> dict[str, Any]:
    """Identify the exact RGB value presented to the GEdit editor/judge.

    Hashing decoded RGB bytes (and separately recording shape) avoids depending on
    container encoding or Hugging Face cache paths while still detecting a dataset
    revision/cache that decodes to different pixels.
    """

    if not isinstance(image, Image.Image):
        raise GEditSelectionError(
            f"GEdit input_image_raw must decode to a Pillow image, got {type(image).__name__}"
        )
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise GEditSelectionError(f"GEdit source image has invalid dimensions: {rgb.size}")
    raw = rgb.tobytes("raw", "RGB")
    expected_bytes = width * height * 3
    if len(raw) != expected_bytes:
        raise GEditSelectionError(
            f"Decoded GEdit RGB byte count differs: expected {expected_bytes}, found {len(raw)}"
        )
    return {
        "representation": "decoded_rgb8",
        "width": width,
        "height": height,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def gedit_judge_jpeg_bytes(image: object) -> bytes:
    """Materialize the exact JPEG sent by the pinned GEdit VIEScore wrapper."""

    if not isinstance(image, Image.Image):
        raise GEditSelectionError(
            f"GEdit judge source must be a Pillow image, got {type(image).__name__}"
        )
    rgb = image.convert("RGB")
    if rgb.width <= 0 or rgb.height <= 0:
        raise GEditSelectionError(f"GEdit judge source has invalid dimensions: {rgb.size}")
    ratio = rgb.width / rgb.height
    width = math.sqrt(512 * 512 * ratio)
    height = width / ratio
    resized = rgb.resize((int(width), int(height)))
    resized = ImageOps.exif_transpose(resized).convert("RGB")
    encoded = io.BytesIO()
    resized.save(encoded, format="JPEG")
    return encoded.getvalue()


def gedit_judge_jpeg_fingerprint(image: object) -> dict[str, Any]:
    raw = gedit_judge_jpeg_bytes(image)
    with Image.open(io.BytesIO(raw)) as decoded:
        width, height = decoded.size
    return {
        "representation": GEDIT_JUDGE_JPEG_REPRESENTATION,
        "width": width,
        "height": height,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(record) for record in records]
    identities: set[str] = set()
    for record in normalized:
        for field in ("key", "task_type", "instruction_language"):
            if not str(record.get(field, "")).strip():
                raise GEditSelectionError(f"GEdit record is missing {field}: {record!r}")
        identity = _identity(record)
        if identity in identities:
            raise GEditSelectionError(f"Duplicate GEdit record identity: {identity}")
        identities.add(identity)
    return normalized


def select_stratified_gedit_records(
    records: Iterable[Mapping[str, Any]],
    *,
    per_language_per_task: int = 15,
    languages: tuple[str, ...] = ("cn", "en"),
    seed: int = 42,
    task_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Select an order-independent deterministic language-balanced task subset."""

    if per_language_per_task <= 0:
        raise GEditSelectionError("per_language_per_task must be positive")
    if not languages or len(set(languages)) != len(languages):
        raise GEditSelectionError("languages must be a non-empty list without duplicates")
    normalized = _validate_records(records)
    available_tasks = sorted({str(record["task_type"]) for record in normalized})
    tasks = list(task_types) if task_types is not None else available_tasks
    if not tasks or len(set(tasks)) != len(tasks):
        raise GEditSelectionError("task_types must be non-empty and contain no duplicates")

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in normalized:
        task = str(record["task_type"])
        language = str(record["instruction_language"])
        if task in tasks and language in languages:
            buckets[(task, language)].append(record)

    selected: list[dict[str, Any]] = []
    deficits: list[str] = []
    for task in tasks:
        for language in languages:
            bucket = buckets[(task, language)]
            if len(bucket) < per_language_per_task:
                deficits.append(
                    f"{task}/{language}: need {per_language_per_task}, found {len(bucket)}"
                )
                continue
            ranked = sorted(
                bucket,
                key=lambda record: (
                    hashlib.sha256(
                        f"{seed}\0{task}\0{language}\0{record['key']}".encode("utf-8")
                    ).hexdigest(),
                    str(record["key"]),
                ),
            )
            selected.extend(ranked[:per_language_per_task])
    if deficits:
        raise GEditSelectionError("Insufficient records for stratified selection: " + "; ".join(deficits))

    selected.sort(key=lambda record: (str(record["task_type"]), str(record["instruction_language"]), str(record["key"])))
    return selected


def selection_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    seed: int | None = None,
    per_language_per_task: int | None = None,
    languages: Iterable[str] | None = None,
) -> dict[str, Any]:
    normalized = _validate_records(records)
    rows = []
    source_bound = True
    for record in normalized:
        row = {
            "identity": _identity(record),
            "key": str(record["key"]),
            "task_type": str(record["task_type"]),
            "instruction_language": str(record["instruction_language"]),
            "instruction_sha256": hashlib.sha256(
                str(record.get("instruction", "")).encode("utf-8")
            ).hexdigest(),
        }
        source_image = record.get("__source_image_fingerprint")
        source_judge_image = record.get("__source_judge_image_fingerprint")
        if (source_image is None) != (source_judge_image is None):
            raise GEditSelectionError(
                "GEdit source RGB and judge-JPEG bindings must be present together"
            )
        if source_image is None:
            source_bound = False
        else:
            row["source_image"] = _validated_source_fingerprint(
                source_image,
                label="GEdit source image",
                representation="decoded_rgb8",
            )
            row["source_judge_image"] = _validated_source_fingerprint(
                source_judge_image,
                label="GEdit source judge JPEG",
                representation=GEDIT_JUDGE_JPEG_REPRESENTATION,
            )
        rows.append(row)
    if any("source_image" in row for row in rows) and not source_bound:
        raise GEditSelectionError(
            "GEdit selection cannot mix source-bound and unbound records"
        )
    rows.sort(key=lambda row: row["identity"])
    counts = Counter((row["task_type"], row["instruction_language"]) for row in rows)
    manifest = {
        "schema": (
            "gedit-selection-manifest/v3" if source_bound and rows else "gedit-selection-manifest/v1"
        ),
        "mode": mode,
        "seed": seed,
        "per_language_per_task": per_language_per_task,
        "languages": list(languages) if languages is not None else None,
        "count": len(rows),
        "counts": {
            f"{task}/{language}": count
            for (task, language), count in sorted(counts.items())
        },
        "records": rows,
    }
    if source_bound and rows:
        manifest["source_image_binding"] = "decoded_rgb8"
        manifest["source_judge_image_binding"] = GEDIT_JUDGE_JPEG_REPRESENTATION
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def write_selection_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(manifest) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise GEditSelectionError(f"Existing GEdit selection manifest differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def apply_gedit_selection(
    records: Iterable[Mapping[str, Any]],
    selection_config: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection_config = selection_config or {}
    mode = str(selection_config.get("mode", "all"))
    normalized = _validate_records(records)
    if mode in {"all", "none"}:
        selected = normalized
        return selected, selection_manifest(selected, mode="all")
    if mode != "stratified_language_task":
        raise GEditSelectionError(f"Unsupported GEdit selection mode: {mode}")
    languages = tuple(str(value) for value in selection_config.get("languages", ["cn", "en"]))
    per_language = int(selection_config.get("per_language_per_task", 15))
    seed = int(selection_config.get("seed", 42))
    raw_tasks = selection_config.get("task_types")
    tasks = tuple(str(value) for value in raw_tasks) if raw_tasks else None
    selected = select_stratified_gedit_records(
        normalized,
        per_language_per_task=per_language,
        languages=languages,
        seed=seed,
        task_types=tasks,
    )
    return selected, selection_manifest(
        selected,
        mode=mode,
        seed=seed,
        per_language_per_task=per_language,
        languages=languages,
    )
