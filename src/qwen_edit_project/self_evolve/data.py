from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_edit_project.self_evolve.types import UnlabeledImageRecord
from qwen_edit_project.utils.paths import resolve_path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _load_metadata_by_key(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = str(item["key"])
            metadata[key] = item
    return metadata


def load_unlabeled_records(dataset_cfg: dict[str, Any], limit: int | None = None) -> list[UnlabeledImageRecord]:
    source = dataset_cfg.get("source", "directory")
    if source == "directory":
        images_dir = resolve_path(dataset_cfg["images_dir"])
        if images_dir is None or not images_dir.exists():
            raise FileNotFoundError(f"images_dir not found: {dataset_cfg['images_dir']}")
        metadata_path = resolve_path(dataset_cfg.get("metadata_jsonl"))
        metadata_by_key = _load_metadata_by_key(metadata_path)
        image_paths = sorted(
            path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        records: list[UnlabeledImageRecord] = []
        for image_path in image_paths:
            rel = image_path.relative_to(images_dir)
            key = str(rel.with_suffix("")).replace("/", "__")
            sidecar = metadata_by_key.get(key, {})
            records.append(
                UnlabeledImageRecord(
                    key=key,
                    image_path=image_path,
                    caption=sidecar.get("caption"),
                    metadata={k: v for k, v in sidecar.items() if k not in {"key", "caption"}},
                )
            )
        return records[:limit] if limit is not None else records

    if source == "jsonl":
        manifest_path = resolve_path(dataset_cfg["manifest_jsonl"])
        if manifest_path is None or not manifest_path.exists():
            raise FileNotFoundError(f"manifest_jsonl not found: {dataset_cfg['manifest_jsonl']}")
        records = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                image_path = resolve_path(item["image"])
                if image_path is None:
                    raise ValueError(f"Could not resolve image path for {item}")
                records.append(
                    UnlabeledImageRecord(
                        key=str(item["key"]),
                        image_path=image_path,
                        caption=item.get("caption"),
                        metadata=item.get("metadata", {}),
                    )
                )
        return records[:limit] if limit is not None else records

    raise ValueError(f"Unsupported self-evolve dataset source: {source}")
