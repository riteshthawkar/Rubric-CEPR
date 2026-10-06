from __future__ import annotations

import getpass
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import ensure_dir, repo_root


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def describe_git_state() -> dict[str, Any]:
    root = repo_root()
    if not (root / ".git").exists():
        return {"available": False, "commit": None, "dirty": None}
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        )
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        )
        return {"available": True, "commit": commit, "dirty": dirty}
    except Exception:
        return {"available": True, "commit": None, "dirty": None}


def write_run_metadata(path: Path, payload: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def base_run_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": utc_timestamp(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "cwd": os.getcwd(),
        "repo_root": str(repo_root()),
        "git": describe_git_state(),
    }

