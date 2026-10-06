from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from .paths import resolve_path


def _looks_like_path(value: str) -> bool:
    return "/" in value or value.startswith(".") or value.startswith("~")


def resolve_executable(value: str | None, fallback: str | None = None) -> str:
    for candidate in (value, fallback):
        if not candidate:
            continue
        if _looks_like_path(candidate):
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                resolved = resolve_path(candidate)
                path = resolved if resolved is not None else path
            if path.exists():
                return str(path)
            continue
        resolved_binary = shutil.which(candidate)
        if resolved_binary:
            return resolved_binary
    raise FileNotFoundError(f"Could not resolve executable from value={value!r} fallback={fallback!r}")


def get_config_executable(
    config: dict[str, Any],
    section: str,
    key: str,
    fallback: str | None,
) -> str:
    section_cfg = config.get(section, {})
    if not isinstance(section_cfg, dict):
        section_cfg = {}
    return resolve_executable(section_cfg.get(key), fallback)


def get_python_executable(config: dict[str, Any]) -> str:
    return get_config_executable(config, "runtime", "python_executable", sys.executable)
