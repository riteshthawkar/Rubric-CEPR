from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import resolve_path


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(str(path))
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must load to a dictionary: {path}")
    data["_config_path"] = str(config_path)
    return data


def save_json(data: Any, path: str | Path) -> Path:
    out_path = resolve_path(str(path))
    if out_path is None:
        raise ValueError("Output path cannot be null")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
    return out_path


def merge_override(config: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    merged = deepcopy(config)
    current: dict[str, Any] = merged
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return merged


def parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must be key=value, got: {raw}")
    key, value = raw.split("=", 1)
    lowered = value.lower()
    if lowered in {"null", "none"}:
        parsed: Any = None
    elif lowered in {"true", "false"}:
        parsed = lowered == "true"
    else:
        for caster in (int, float):
            try:
                parsed = caster(value)
                break
            except ValueError:
                parsed = value
        if isinstance(parsed, str) and (
            (parsed.startswith("[") and parsed.endswith("]"))
            or (parsed.startswith("{") and parsed.endswith("}"))
        ):
            parsed = json.loads(parsed)
    return key, parsed
