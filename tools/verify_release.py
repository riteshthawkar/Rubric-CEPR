#!/usr/bin/env python3
"""Verify allowlisted payload identities and report secrets without printing values."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git", "build", "dist"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI token": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"),
    "GitHub token": re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}\b"),
    "credential in URL": re.compile(r"https?://[^/\s:]+:[^/\s@]+@"),
}


def payload_paths(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and not any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in p.relative_to(root).parts)
        and p.relative_to(root).parts[0] not in {"data", "outputs"}
        and not (p.relative_to(root).parts[0] == "third_party" and len(p.relative_to(root).parts) > 2)
    )


def audit(root: Path = ROOT) -> dict:
    findings = []
    count = 0
    for path in payload_paths(root):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append({"path": relative, "kind": "payload symlink"})
        if path.name in {".env", "secret.env"} or path.suffix in {".safetensors", ".pyc"}:
            findings.append({"path": relative, "kind": "private or generated artifact"})
        body = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(body):
                findings.append({"path": relative, "kind": label})
        if path.suffix in {".yaml", ".json", ".toml"} and re.search(r"/share_[0-9]+/|/home/", body):
            findings.append({"path": relative, "kind": "host-specific configuration path"})
        if path.suffix == ".py":
            ast.parse(body, filename=relative)
        count += 1
    if findings:
        raise ValueError(json.dumps({"release_audit_failed": findings}, indent=2))
    return {"payload_files_scanned": count, "secret_findings": 0, "host_specific_config_paths": 0}


def verify(root: Path = ROOT) -> dict:
    report = audit(root)
    checksums = root / "SHA256SUMS"
    count = 0
    expected_paths = set()
    for line in checksums.read_text().splitlines():
        expected, name = line.split("  ", 1)
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or name in expected_paths:
            raise ValueError("Invalid or duplicate checksum path")
        expected_paths.add(name)
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release payload changed: {name}")
        count += 1
    actual_paths = {p.relative_to(root).as_posix() for p in payload_paths(root) if p.name != "SHA256SUMS"}
    if expected_paths != actual_paths:
        raise ValueError("Release payload contains missing or unexpected files")
    report["payload_files_verified"] = count
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit() if args.audit_only else verify(), indent=2))
