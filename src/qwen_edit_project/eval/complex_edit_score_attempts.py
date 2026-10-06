"""Fresh operational attempt namespaces for paid Complex-Edit C4 scoring."""

from __future__ import annotations

import atexit
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from qwen_edit_project.eval.evaluation_contract import canonical_json_bytes
from qwen_edit_project.eval.score_receipts import (
    SCORE_ATTEMPT_FAILURE_FILENAME,
    SCORE_ATTEMPT_ID_PATTERN,
    SCORE_ATTEMPT_MANIFEST_FILENAME,
    SCORE_ATTEMPT_SCHEMA,
    SCORE_ATTEMPTS_DIRECTORY,
    ScoreBundleError,
    write_json_exclusive,
)


def _attempt_manifest(*, attempt_id: str, contract_id: str) -> dict[str, str]:
    return {
        "schema": SCORE_ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "benchmark": "complex_edit",
        "contract_id": contract_id,
    }


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ScoreBundleError(f"Complex-Edit score-attempt path traverses a symlink: {current}")


def _manifest_is_exact(path: Path, expected: Mapping[str, str]) -> bool:
    _reject_symlink_components(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        return False
    expected_bytes = canonical_json_bytes(dict(expected)) + b"\n"
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return raw == expected_bytes and parsed == dict(expected)


@dataclass
class ComplexEditScoreAttempt:
    """One paid scorer execution, isolated until exact validation succeeds."""

    contract_root: Path
    attempt_id: str
    contract_id: str
    current_dir: Path
    completed_dir: Path
    quarantine_dir: Path
    _committed: bool = False
    _atexit_callback: Any = field(default=None, repr=False)

    def publish_validated(self, validator: Callable[[Path], None]) -> Path:
        """Validate active bytes, then atomically rename them under ``complete``."""

        if self._committed:
            raise ScoreBundleError("Complex-Edit score attempt is already complete")
        if self.current_dir.parent.name != "active":
            raise ScoreBundleError("Complex-Edit score attempt is not active")
        if not callable(validator):
            raise ScoreBundleError("Complex-Edit score-attempt validator must be callable")
        validator(self.current_dir)
        expected_manifest = _attempt_manifest(
            attempt_id=self.attempt_id,
            contract_id=self.contract_id,
        )
        if not _manifest_is_exact(
            self.current_dir / SCORE_ATTEMPT_MANIFEST_FILENAME,
            expected_manifest,
        ):
            raise ScoreBundleError("Complex-Edit score-attempt manifest changed before promotion")
        if self.completed_dir.exists() or self.completed_dir.is_symlink():
            raise ScoreBundleError(
                f"Completed Complex-Edit score attempt already exists: {self.completed_dir}"
            )
        os.replace(self.current_dir, self.completed_dir)
        self.current_dir = self.completed_dir
        self._committed = True
        if self._atexit_callback is not None:
            atexit.unregister(self._atexit_callback)
        return self.completed_dir

    def quarantine(self, reason: str) -> Path | None:
        """Retain a failed active attempt outside the promotion-eligible tree."""

        if self.current_dir == self.quarantine_dir:
            return self.quarantine_dir
        if self._committed or not self.current_dir.exists():
            return None
        if self.current_dir.is_symlink() or not self.current_dir.is_dir():
            raise ScoreBundleError("Unsafe Complex-Edit score attempt cannot be quarantined")
        if self.quarantine_dir.exists() or self.quarantine_dir.is_symlink():
            raise ScoreBundleError(
                f"Quarantined Complex-Edit attempt already exists: {self.quarantine_dir}"
            )
        failure_path = self.current_dir / SCORE_ATTEMPT_FAILURE_FILENAME
        if not failure_path.exists():
            write_json_exclusive(
                failure_path,
                {
                    **_attempt_manifest(
                        attempt_id=self.attempt_id,
                        contract_id=self.contract_id,
                    ),
                    "status": "quarantined",
                    "reason": str(reason)[:500],
                },
            )
        os.replace(self.current_dir, self.quarantine_dir)
        self.current_dir = self.quarantine_dir
        if self._atexit_callback is not None:
            atexit.unregister(self._atexit_callback)
        return self.quarantine_dir


def begin_complex_edit_score_attempt(
    contract_root: Path,
    *,
    attempt_id: str,
    contract_id: str,
) -> ComplexEditScoreAttempt:
    """Exclusively claim a never-before-used C4 score-attempt identity."""

    attempt_id = str(attempt_id).strip()
    contract_id = str(contract_id).strip()
    if SCORE_ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None or ".." in attempt_id:
        raise ScoreBundleError(
            "Complex-Edit attempt ID must be 1-64 safe alphanumeric/dot/underscore/hyphen "
            "characters"
        )
    if (
        len(contract_id) != 64
        or contract_id.lower() != contract_id
        or any(character not in "0123456789abcdef" for character in contract_id)
    ):
        raise ScoreBundleError("Complex-Edit score attempt requires a SHA-256 contract ID")

    contract_root = Path(contract_root)
    _reject_symlink_components(contract_root)
    contract_root.mkdir(parents=True, exist_ok=True)
    attempts_root = contract_root / SCORE_ATTEMPTS_DIRECTORY
    active_root = attempts_root / "active"
    completed_root = attempts_root / "complete"
    quarantine_root = attempts_root / "quarantine"
    for directory in (attempts_root, active_root, completed_root, quarantine_root):
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ScoreBundleError(f"Unsafe Complex-Edit score-attempt namespace: {directory}")

    active_dir = active_root / attempt_id
    completed_dir = completed_root / attempt_id
    quarantine_dir = quarantine_root / attempt_id
    collisions = [
        path
        for path in (active_dir, completed_dir, quarantine_dir)
        if path.exists() or path.is_symlink()
    ]
    if collisions:
        raise ScoreBundleError(
            "Complex-Edit score attempt ID is not fresh; choose a new --attempt-id. "
            f"Existing namespaces: {[str(path) for path in collisions]}"
        )
    active_dir.mkdir(mode=0o700)
    write_json_exclusive(
        active_dir / SCORE_ATTEMPT_MANIFEST_FILENAME,
        _attempt_manifest(attempt_id=attempt_id, contract_id=contract_id),
    )
    attempt = ComplexEditScoreAttempt(
        contract_root=contract_root,
        attempt_id=attempt_id,
        contract_id=contract_id,
        current_dir=active_dir,
        completed_dir=completed_dir,
        quarantine_dir=quarantine_dir,
    )

    def quarantine_uncommitted() -> None:
        try:
            attempt.quarantine("process exited before exact C4 validation and publication")
        except Exception:
            pass

    attempt._atexit_callback = quarantine_uncommitted
    atexit.register(quarantine_uncommitted)
    return attempt
