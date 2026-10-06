from __future__ import annotations

import atexit
import base64
import hashlib
import csv
import io
import json
import math
import os
import re
import runpy
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from functools import lru_cache
from typing import Any, Mapping

from qwen_edit_project.eval.evaluation_contract import (
    OUTPUT_MANIFEST_FILENAME,
    OUTPUT_MANIFEST_SCHEMA,
    canonical_json_bytes,
    sha256_bytes,
)


SCORE_BUNDLE_SCHEMA = "qwen-edit-score-bundle/v3"
SCORE_BUNDLE_RECEIPT_FILENAME = ".score_bundle_receipt.json"
SCORE_BUNDLE_CLAIM_SCHEMA = "qwen-edit-score-bundle-claim/v1"
SCORE_BUNDLE_CLAIM_FILENAME = ".score_bundle_in_progress.json"
SCORE_ATTEMPT_SCHEMA = "qwen-edit-score-attempt/v1"
SCORE_ATTEMPT_MANIFEST_FILENAME = ".score_attempt.json"
SCORE_ATTEMPT_FAILURE_FILENAME = ".score_attempt_failure.json"
SCORE_ATTEMPTS_DIRECTORY = ".score_attempts"
SCORE_ATTEMPT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class ScoreBundleError(RuntimeError):
    """Raised when immutable scorer evidence is incomplete or has changed."""


def _score_attempt_manifest(
    *,
    attempt_id: str,
    benchmark: str,
    contract_id: str,
) -> dict[str, Any]:
    return {
        "schema": SCORE_ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "benchmark": benchmark,
        "contract_id": contract_id,
    }


@dataclass
class ScoreAttemptNamespace:
    """One exclusive operational attempt under an immutable scoring contract."""

    contract_root: Path
    attempt_id: str
    benchmark: str
    contract_id: str
    current_dir: Path
    completed_dir: Path
    quarantine_dir: Path
    _committed: bool = False
    _atexit_callback: Any = field(default=None, repr=False)

    def publish(self) -> Path:
        """Atomically move fully produced raw artifacts into the complete namespace."""

        if self._committed:
            raise ScoreBundleError("Score attempt is already committed")
        if self.current_dir.parent.name != "active":
            raise ScoreBundleError("Score attempt is not in the active namespace")
        manifest_path = self.current_dir / SCORE_ATTEMPT_MANIFEST_FILENAME
        manifest_raw = _stable_regular_file_bytes(manifest_path, label="score attempt manifest")
        manifest = _load_json_bytes(manifest_raw, label="score attempt manifest")
        expected = _score_attempt_manifest(
            attempt_id=self.attempt_id,
            benchmark=self.benchmark,
            contract_id=self.contract_id,
        )
        if manifest != expected:
            raise ScoreBundleError("Score attempt manifest differs from the active attempt")
        if self.completed_dir.exists() or self.completed_dir.is_symlink():
            raise ScoreBundleError(
                f"Completed score attempt already exists: {self.completed_dir}"
            )
        os.replace(self.current_dir, self.completed_dir)
        self.current_dir = self.completed_dir
        return self.current_dir

    def summary_record(self) -> dict[str, Any]:
        if self.current_dir != self.completed_dir or not self.completed_dir.is_dir():
            raise ScoreBundleError("Score attempt must be published before writing its summary")
        return {
            **_score_attempt_manifest(
                attempt_id=self.attempt_id,
                benchmark=self.benchmark,
                contract_id=self.contract_id,
            ),
            "namespace": f"complete/{self.attempt_id}",
        }

    def commit(self, summary_path: Path) -> None:
        """Mark an attempt complete only after its immutable bundle validates."""

        summary_path = Path(summary_path)
        if summary_path.parent != self.completed_dir:
            raise ScoreBundleError("Committed score summary is outside its attempt namespace")
        verified = validate_score_summary_bundle(
            summary_path,
            expected_benchmark=self.benchmark,
        )
        if verified["receipt"].get("contract_id") != self.contract_id:
            raise ScoreBundleError("Committed score attempt has the wrong contract ID")
        self._committed = True
        if self._atexit_callback is not None:
            atexit.unregister(self._atexit_callback)

    def quarantine(self, reason: str) -> Path | None:
        """Preserve an incomplete attempt outside the promotion-eligible namespace."""

        if self.current_dir == self.quarantine_dir:
            return self.quarantine_dir
        if self._committed or not self.current_dir.exists():
            return None
        if self.current_dir.is_symlink() or not self.current_dir.is_dir():
            raise ScoreBundleError("Unsafe score-attempt directory cannot be quarantined")
        if self.quarantine_dir.exists() or self.quarantine_dir.is_symlink():
            raise ScoreBundleError(
                f"Quarantined score attempt already exists: {self.quarantine_dir}"
            )
        failure_path = self.current_dir / SCORE_ATTEMPT_FAILURE_FILENAME
        if not failure_path.exists():
            write_json_exclusive(
                failure_path,
                {
                    "schema": SCORE_ATTEMPT_SCHEMA,
                    "attempt_id": self.attempt_id,
                    "benchmark": self.benchmark,
                    "contract_id": self.contract_id,
                    "status": "quarantined",
                    "reason": str(reason)[:500],
                },
            )
        os.replace(self.current_dir, self.quarantine_dir)
        self.current_dir = self.quarantine_dir
        if self._atexit_callback is not None:
            atexit.unregister(self._atexit_callback)
        return self.current_dir


def begin_score_attempt(
    contract_root: Path,
    *,
    attempt_id: str,
    benchmark: str,
    contract_id: str,
) -> ScoreAttemptNamespace:
    """Create a fresh unique attempt without changing the scientific contract ID."""

    attempt_id = str(attempt_id).strip()
    if SCORE_ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None or ".." in attempt_id:
        raise ScoreBundleError(
            "Score attempt ID must be 1-64 safe alphanumeric/dot/underscore/hyphen characters"
        )
    if benchmark not in {"imgedit", "gedit"}:
        raise ScoreBundleError(f"Unsupported score-attempt benchmark: {benchmark!r}")
    if re.fullmatch(r"[0-9a-f]{64}", str(contract_id)) is None:
        raise ScoreBundleError("Score attempt requires a lowercase SHA-256 contract ID")
    contract_root = Path(contract_root)
    _reject_symlink_path_components(contract_root, label="score contract root")
    contract_root.mkdir(parents=True, exist_ok=True)
    attempts_root = contract_root / SCORE_ATTEMPTS_DIRECTORY
    active_root = attempts_root / "active"
    completed_root = attempts_root / "complete"
    quarantine_root = attempts_root / "quarantine"
    for directory in (attempts_root, active_root, completed_root, quarantine_root):
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ScoreBundleError(f"Unsafe score-attempt namespace: {directory}")
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
            "Score attempt ID is not fresh; choose a new --attempt-id. "
            f"Existing namespaces: {[str(path) for path in collisions]}"
        )
    active_dir.mkdir(mode=0o700)
    write_json_exclusive(
        active_dir / SCORE_ATTEMPT_MANIFEST_FILENAME,
        _score_attempt_manifest(
            attempt_id=attempt_id,
            benchmark=benchmark,
            contract_id=contract_id,
        ),
    )
    attempt = ScoreAttemptNamespace(
        contract_root=contract_root,
        attempt_id=attempt_id,
        benchmark=benchmark,
        contract_id=contract_id,
        current_dir=active_dir,
        completed_dir=completed_dir,
        quarantine_dir=quarantine_dir,
    )

    def quarantine_uncommitted() -> None:
        try:
            attempt.quarantine("process exited before an immutable score bundle was committed")
        except Exception:
            # Never mask the original scorer failure during interpreter shutdown.
            pass

    attempt._atexit_callback = quarantine_uncommitted
    atexit.register(quarantine_uncommitted)
    return attempt


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_symlink_path_components(path: Path, *, label: str) -> None:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ScoreBundleError(f"{label} path traverses a symlink: {current}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScoreBundleError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _stable_regular_file_bytes(path: Path, *, label: str) -> bytes:
    path = Path(path)
    _reject_symlink_path_components(path, label=label)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ScoreBundleError(f"{label} is missing: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise ScoreBundleError(f"{label} must be a regular non-symlink file: {path}")
    if before.st_nlink != 1:
        raise ScoreBundleError(f"{label} must have exactly one hard link: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(opened) or _stat_identity(opened) != _stat_identity(
        after
    ):
        raise ScoreBundleError(f"{label} changed while it was read: {path}")
    return b"".join(chunks)


def _load_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ScoreBundleError(f"{label} is not canonical JSON") from exc
    if not isinstance(payload, dict):
        raise ScoreBundleError(f"{label} must contain a JSON object")
    return payload


def _file_record(path: Path, *, root: Path, role: str) -> tuple[dict[str, Any], bytes]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ScoreBundleError(f"Score artifact {path} is outside score root {root}") from exc
    raw = _stable_regular_file_bytes(path, label=f"score artifact {role}")
    return (
        {
            "role": role,
            "path": relative,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        raw,
    )


def _safe_relative(root: Path, value: object, *, label: str) -> Path:
    relative = Path(str(value))
    if not str(value) or relative.is_absolute() or ".." in relative.parts:
        raise ScoreBundleError(f"Unsafe {label} path: {value!r}")
    candidate = root / relative
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ScoreBundleError(f"Unsafe {label} path: {value!r}") from exc
    return candidate


def _summary_evidence(
    summary: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any], list[Any], str]:
    envelope = summary.get("evaluation_contract")
    if not isinstance(envelope, Mapping) or not isinstance(envelope.get("contract"), Mapping):
        raise ScoreBundleError("Score summary has no evaluation contract")
    contract = dict(envelope["contract"])
    contract_id = str(envelope.get("contract_id", ""))
    if len(contract_id) != 64 or contract_id != sha256_bytes(canonical_json_bytes(contract)):
        raise ScoreBundleError("Score summary evaluation contract ID is invalid")
    benchmark = str(summary.get("benchmark", ""))
    if benchmark not in {"imgedit", "gedit"} or contract.get("benchmark") != benchmark:
        raise ScoreBundleError("Score summary benchmark differs from its evaluation contract")
    expected_canonical_protocol = f"{benchmark}-openai-fresh-score/v1"
    if contract.get("canonical_protocol") != expected_canonical_protocol:
        raise ScoreBundleError(
            "Score summary evaluation contract has no supported canonical scoring protocol"
        )

    scoring = summary.get("scoring")
    if not isinstance(scoring, Mapping) or not isinstance(scoring.get("judge_protocol"), Mapping):
        raise ScoreBundleError("Score summary has no canonical scoring.judge_protocol")
    judge_protocol = dict(scoring["judge_protocol"])
    contract_judge = contract.get("judge")
    if (
        not isinstance(contract_judge, Mapping)
        or not isinstance(contract_judge.get("protocol"), Mapping)
        or judge_protocol != contract_judge.get("protocol")
    ):
        raise ScoreBundleError("Score summary judge protocol differs from its evaluation contract")

    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get("record_scores"), list):
        raise ScoreBundleError("Score summary has no identity-bound metrics.record_scores")
    records = list(metrics["record_scores"])
    record_digest = str(metrics.get("record_scores_sha256", ""))
    actual_record_digest = sha256_bytes(canonical_json_bytes(records))
    if record_digest != actual_record_digest:
        raise ScoreBundleError("Score summary record evidence digest is invalid")
    contract_dataset = contract.get("dataset")
    selection = contract_dataset.get("selection") if isinstance(contract_dataset, Mapping) else None
    if not isinstance(selection, Mapping) or not isinstance(selection.get("records"), list):
        raise ScoreBundleError("Score summary evaluation contract has no record selection")
    selected_records = selection["records"]
    selected_identities = [
        str(item.get("identity", "")) if isinstance(item, Mapping) else ""
        for item in selected_records
    ]
    score_identities = [
        str(item.get("identity", "")) if isinstance(item, Mapping) else "" for item in records
    ]
    if (
        not selected_identities
        or any(not identity for identity in selected_identities)
        or len(set(selected_identities)) != len(selected_identities)
        or any(not identity for identity in score_identities)
        or len(set(score_identities)) != len(score_identities)
        or set(score_identities) != set(selected_identities)
        or selection.get("count") != len(selected_records)
    ):
        raise ScoreBundleError("Score summary record evidence differs from contracted selection")
    return contract_id, contract, judge_protocol, records, record_digest


def _expected_output_paths(contract: Mapping[str, Any], *, benchmark: str) -> list[str]:
    selection = contract["dataset"]["selection"]["records"]
    if benchmark == "imgedit":
        paths = []
        for item in selection:
            if not isinstance(item, Mapping) or not str(item.get("identity", "")):
                raise ScoreBundleError("ImgEdit contracted output identity is invalid")
            paths.append(f"{item['identity']}.png")
    else:
        paths = []
        for item in selection:
            if not isinstance(item, Mapping):
                raise ScoreBundleError("GEdit contracted output identity is invalid")
            task = str(item.get("task_type", ""))
            language = str(item.get("instruction_language", ""))
            key = str(item.get("key", ""))
            if not task or not language or not key:
                raise ScoreBundleError("GEdit contracted output identity is invalid")
            paths.append(f"{task}/{language}/{key}.png")
    return sorted(paths)


def _validate_generated_output_manifest(
    *,
    result_dir: Path,
    manifest_raw: bytes,
    contract_id: str,
    contract: Mapping[str, Any],
    benchmark: str,
) -> None:
    manifest = _load_json_bytes(manifest_raw, label="generated output manifest")
    rows = manifest.get("files")
    if (
        manifest.get("schema") != OUTPUT_MANIFEST_SCHEMA
        or manifest.get("contract_id") != contract_id
        or not isinstance(rows, list)
        or manifest.get("count") != len(rows)
    ):
        raise ScoreBundleError("Generated-output manifest envelope is invalid")
    expected_paths = _expected_output_paths(contract, benchmark=benchmark)
    observed_paths: list[str] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise ScoreBundleError(f"Generated-output manifest row {index} is not a mapping")
        output_path = _safe_relative(
            result_dir,
            raw_row.get("path"),
            label=f"generated output {index}",
        )
        output_raw = _stable_regular_file_bytes(
            output_path,
            label=f"generated output {index}",
        )
        expected_row = {
            "path": output_path.relative_to(result_dir).as_posix(),
            "bytes": len(output_raw),
            "sha256": hashlib.sha256(output_raw).hexdigest(),
        }
        if dict(raw_row) != expected_row:
            raise ScoreBundleError(
                f"Generated output {expected_row['path']} differs from its manifest"
            )
        observed_paths.append(expected_row["path"])
    if observed_paths != expected_paths:
        raise ScoreBundleError("Generated-output manifest differs from contracted selection")
    final_manifest_raw = _stable_regular_file_bytes(
        result_dir / OUTPUT_MANIFEST_FILENAME,
        label="generated output manifest",
    )
    if final_manifest_raw != manifest_raw:
        raise ScoreBundleError("Generated-output manifest changed during bundle validation")


def _validate_artifact_roles(benchmark: str, roles: set[str]) -> None:
    if benchmark == "imgedit" and roles != {
        "average_scores",
        "raw_scores",
        "type_scores",
    }:
        raise ScoreBundleError("ImgEdit score bundle does not contain the exact required artifacts")
    if benchmark == "gedit":
        csv_roles = {role for role in roles if role.startswith("score_csv:")}
        raw_roles = {role for role in roles if role.startswith("raw_response:")}
        expected_csv_roles = {
            f"score_csv:{index:03d}" for index in range(len(csv_roles))
        }
        expected_raw_roles = {
            f"raw_response:{index:03d}" for index in range(len(raw_roles))
        }
        if (
            not csv_roles
            or not raw_roles
            or roles != csv_roles | raw_roles
            or csv_roles != expected_csv_roles
            or raw_roles != expected_raw_roles
        ):
            raise ScoreBundleError(
                "GEdit score bundle requires exact consecutive CSV and raw-response roles"
            )


def derive_imgedit_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the complete canonical ImgEdit metric block from identity scores."""

    ordered = sorted(records, key=lambda item: str(item["identity"]))
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in ordered:
        grouped[str(record["family"])].append(float(record["score"]))
    return {
        "count": len(ordered),
        "overall_average": math.fsum(float(item["score"]) for item in ordered)
        / len(ordered),
        "type_scores": {
            family: math.fsum(values) / len(values)
            for family, values in sorted(grouped.items())
        },
        "record_scores": ordered,
        "record_scores_sha256": sha256_bytes(canonical_json_bytes(ordered)),
    }


def derive_gedit_metrics(
    records: list[dict[str, Any]],
    *,
    backbone: str,
) -> dict[str, Any]:
    """Derive deterministic GEdit counts and means from canonical record evidence."""

    ordered = sorted(records, key=lambda item: str(item["identity"]))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        grouped[str(record["task_type"])].append(record)

    def means(values: list[dict[str, Any]]) -> dict[str, float]:
        return {
            field: math.fsum(float(item[field]) for item in values) / len(values)
            for field in ("semantics", "quality", "overall")
        }

    return {
        "groups": {
            group: {**means(values), "count": len(values)}
            for group, values in sorted(grouped.items())
        },
        "backbone": backbone,
        "count": len(ordered),
        "average": means(ordered),
        "record_scores": ordered,
        "record_scores_sha256": sha256_bytes(canonical_json_bytes(ordered)),
    }


def _validate_imgedit_score_evidence(
    *,
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    artifact_payloads: Mapping[str, bytes],
) -> None:
    from qwen_edit_project.eval.imgedit_scores import strict_imgedit_average

    selection = contract["dataset"]["selection"]["records"]
    families: dict[str, str] = {}
    for item in selection:
        identity = str(item.get("identity", ""))
        family = str(item.get("edit_type", ""))
        if not identity or not family or identity in families:
            raise ScoreBundleError("ImgEdit contracted selection has invalid score identities")
        families[identity] = family

    raw_scores = _load_json_bytes(
        artifact_payloads["raw_scores"],
        label="ImgEdit raw scores",
    )
    if set(raw_scores) != set(families):
        raise ScoreBundleError("ImgEdit raw scores differ from contracted selection")
    averages: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for identity, family in families.items():
        try:
            score = float(strict_imgedit_average(raw_scores[identity], family))
        except (TypeError, ValueError) as exc:
            raise ScoreBundleError(
                f"ImgEdit raw judge response is invalid for {identity}"
            ) from exc
        if not math.isfinite(score):
            raise ScoreBundleError(f"ImgEdit score is non-finite for {identity}")
        averages[identity] = score
        grouped[family].append(score)
        records.append({"identity": identity, "family": family, "score": score})
    records.sort(key=lambda item: str(item["identity"]))
    type_scores = {
        family: math.fsum(values) / len(values) for family, values in sorted(grouped.items())
    }
    published_averages = _load_json_bytes(
        artifact_payloads["average_scores"],
        label="ImgEdit average scores",
    )
    published_type_scores = _load_json_bytes(
        artifact_payloads["type_scores"],
        label="ImgEdit type scores",
    )
    expected_metrics = derive_imgedit_metrics(records)
    if (
        published_averages != averages
        or published_type_scores != type_scores
        or summary.get("metrics") != expected_metrics
    ):
        raise ScoreBundleError(
            "ImgEdit summary metrics are not an exact derivation of raw judge responses"
        )


def _strict_gedit_provider_score(raw_value: object, *, label: str) -> float:
    """Parse the exact successful provider text under the pinned VIEScore grammar."""

    if not isinstance(raw_value, str):
        raise ScoreBundleError(f"GEdit provider {label} content is not text")
    text = raw_value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"}:
            raise ScoreBundleError(f"GEdit provider {label} has an invalid JSON code fence")
        text = "\n".join(lines[1:-1]).strip()
    try:
        value, end = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs).raw_decode(text)
    except (json.JSONDecodeError, TypeError, ScoreBundleError) as exc:
        raise ScoreBundleError(f"GEdit provider {label} content is malformed JSON") from exc
    if text[end:].strip():
        raise ScoreBundleError(f"GEdit provider {label} content has trailing text")
    if not isinstance(value, dict) or set(value) != {"score", "reasoning"}:
        raise ScoreBundleError(f"GEdit provider {label} content has the wrong schema")
    if not isinstance(value["reasoning"], str):
        raise ScoreBundleError(f"GEdit provider {label} reasoning is not text")
    scores = value["score"]
    if not isinstance(scores, list) or not scores:
        raise ScoreBundleError(f"GEdit provider {label} has no scores")
    normalized: list[float] = []
    for score in scores:
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 10.0
        ):
            raise ScoreBundleError(f"GEdit provider {label} has an invalid score")
        normalized.append(float(score))
    return min(normalized)


def _require_sha256_value(value: object, *, label: str) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ScoreBundleError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _gedit_encoded_edited_image_bytes(image_raw: bytes) -> bytes:
    """Reproduce the exact resized JPEG bytes sent by the pinned GEdit runner."""

    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(image_raw)) as opened:
            image = opened.convert("RGB")
        ratio = image.width / image.height
        width = math.sqrt(512 * 512 * ratio)
        height = width / ratio
        image = image.resize((int(width), int(height)))
        image = ImageOps.exif_transpose(image).convert("RGB")
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG")
        return encoded.getvalue()
    except Exception as exc:
        raise ScoreBundleError("GEdit edited image cannot be reproduced as judge input") from exc


def _gedit_encoded_image_record(payload: bytes, *, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "media_type": "image/jpeg",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@lru_cache(maxsize=1)
def _gedit_prompt_templates() -> tuple[str, str, str, str]:
    prompt_path = (
        Path(__file__).resolve().parents[3]
        / "third_party/step1x-edit/GEdit-Bench/viescore/vie_prompts.py"
    )
    if prompt_path.is_symlink() or not prompt_path.is_file():
        raise ScoreBundleError("Pinned GEdit VIEScore prompts are missing or unsafe")
    try:
        namespace = runpy.run_path(str(prompt_path))
        values = tuple(
            namespace[name]
            for name in (
                "_context_no_delimit",
                "_prompts_0shot_two_image_edit_rule",
                "_prompts_0shot_tie_rule_SC",
                "_prompts_0shot_rule_PQ",
            )
        )
    except Exception as exc:
        raise ScoreBundleError("Pinned GEdit VIEScore prompts could not be loaded") from exc
    if not all(isinstance(value, str) for value in values):
        raise ScoreBundleError("Pinned GEdit VIEScore prompt constants are invalid")
    return values  # type: ignore[return-value]


def _gedit_expected_request_evidence(
    *,
    instruction: str,
    source_jpeg: bytes,
    edited_jpeg: bytes,
) -> dict[str, dict[str, Any]]:
    context, edit_rule, sc_rule, pq_rule = _gedit_prompt_templates()
    sc_prompt = "\n".join((context, edit_rule, sc_rule)).replace(
        "<instruction>", instruction
    )
    pq_prompt = "\n".join((context, pq_rule))

    def evidence(prompt: str, images: list[bytes]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(image).decode("ascii")
                    },
                }
            )
        messages = [{"role": "user", "content": content}]
        prompt_raw = prompt.encode("utf-8")
        return {
            "messages_sha256": sha256_bytes(canonical_json_bytes(messages)),
            "text_parts": [
                {
                    "index": 0,
                    "bytes": len(prompt_raw),
                    "sha256": hashlib.sha256(prompt_raw).hexdigest(),
                }
            ],
            "image_parts": [
                _gedit_encoded_image_record(image, index=index + 1)
                for index, image in enumerate(images)
            ],
        }

    return {
        "semantic_consistency": evidence(sc_prompt, [source_jpeg, edited_jpeg]),
        "perceptual_quality": evidence(pq_prompt, [edited_jpeg]),
    }


def _validate_request_part_digest(
    raw_part: object,
    *,
    label: str,
    expected_index: int,
    require_media_type: bool,
) -> dict[str, Any]:
    if not isinstance(raw_part, Mapping):
        raise ScoreBundleError(f"{label} is not a mapping")
    expected_keys = {"index", "bytes", "sha256"}
    if require_media_type:
        expected_keys.add("media_type")
    if set(raw_part) != expected_keys:
        raise ScoreBundleError(f"{label} has the wrong schema")
    byte_count = raw_part.get("bytes")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
        or raw_part.get("index") != expected_index
    ):
        raise ScoreBundleError(f"{label} has invalid index or byte count")
    if require_media_type and raw_part.get("media_type") != "image/jpeg":
        raise ScoreBundleError(f"{label} has an unexpected media type")
    _require_sha256_value(raw_part.get("sha256"), label=f"{label} digest")
    return dict(raw_part)


def _validate_gedit_provider_transaction(
    raw_transaction: object,
    *,
    label: str,
    judge_protocol: Mapping[str, Any],
    expected_request_evidence: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    if not isinstance(raw_transaction, Mapping) or set(raw_transaction) != {
        "schema",
        "request",
        "response",
    }:
        raise ScoreBundleError(f"GEdit {label} provider transaction has the wrong schema")
    if raw_transaction.get("schema") != "gedit-openai-response-evidence/v1":
        raise ScoreBundleError(f"GEdit {label} provider transaction has an unsupported schema")

    request = raw_transaction.get("request")
    if not isinstance(request, Mapping) or set(request) != {
        "endpoint",
        "model",
        "max_tokens",
        "stream",
        "timeout_seconds",
        "messages_sha256",
        "text_parts",
        "image_parts",
    }:
        raise ScoreBundleError(f"GEdit {label} sanitized request has the wrong schema")
    parameters = judge_protocol.get("request_parameters")
    if not isinstance(parameters, Mapping):
        raise ScoreBundleError("GEdit judge protocol has no request parameters")
    if (
        request.get("endpoint") != judge_protocol.get("endpoint")
        or request.get("model") != parameters.get("model")
        or request.get("max_tokens") != parameters.get("max_tokens")
        or request.get("stream") is not False
        or request.get("timeout_seconds") != parameters.get("timeout_seconds")
    ):
        raise ScoreBundleError(f"GEdit {label} request differs from the contracted judge")
    messages_sha256 = _require_sha256_value(
        request.get("messages_sha256"),
        label=f"GEdit {label} messages digest",
    )
    if messages_sha256 != expected_request_evidence.get("messages_sha256"):
        raise ScoreBundleError(f"GEdit {label} full messages digest is invalid")
    text_parts = request.get("text_parts")
    image_parts = request.get("image_parts")
    if not isinstance(text_parts, list) or len(text_parts) != 1:
        raise ScoreBundleError(f"GEdit {label} request must contain one text part")
    normalized_text = _validate_request_part_digest(
        text_parts[0],
        label=f"GEdit {label} text part",
        expected_index=0,
        require_media_type=False,
    )
    if [normalized_text] != expected_request_evidence.get("text_parts"):
        raise ScoreBundleError(f"GEdit {label} request prompt digest is invalid")
    expected_image_parts = expected_request_evidence.get("image_parts")
    if not isinstance(expected_image_parts, list):
        raise ScoreBundleError(f"GEdit {label} expected request evidence is invalid")
    if not isinstance(image_parts, list) or len(image_parts) != len(expected_image_parts):
        raise ScoreBundleError(f"GEdit {label} request has the wrong image cardinality")
    normalized_images = [
        _validate_request_part_digest(
            raw_part,
            label=f"GEdit {label} image part {index}",
            expected_index=index + 1,
            require_media_type=True,
        )
        for index, raw_part in enumerate(image_parts)
    ]
    if normalized_images != expected_image_parts:
        raise ScoreBundleError(f"GEdit {label} request image evidence is invalid")

    response = raw_transaction.get("response")
    if not isinstance(response, Mapping) or set(response) != {
        "status_code",
        "headers",
        "request_id",
        "json",
    }:
        raise ScoreBundleError(f"GEdit {label} provider response has the wrong schema")
    if response.get("status_code") != 200:
        raise ScoreBundleError(f"GEdit {label} provider response was not successful")
    headers = response.get("headers")
    if not isinstance(headers, Mapping) or not headers:
        raise ScoreBundleError(f"GEdit {label} provider response has no headers")
    sensitive_header_names = {"authorization", "api-key", "openai-api-key", "x-api-key"}
    if any(
        not isinstance(key, str)
        or key != key.lower()
        or not isinstance(value, str)
        or key in sensitive_header_names
        for key, value in headers.items()
    ):
        raise ScoreBundleError(f"GEdit {label} provider response headers are unsafe")
    raw_request_id = response.get("request_id")
    request_id = raw_request_id if isinstance(raw_request_id, str) else ""
    if (
        not request_id
        or request_id != request_id.strip()
        or headers.get("x-request-id") != request_id
    ):
        raise ScoreBundleError(f"GEdit {label} request ID differs from its response header")

    response_json = response.get("json")
    if not isinstance(response_json, Mapping):
        raise ScoreBundleError(f"GEdit {label} provider response JSON is not a mapping")
    raw_response_id = response_json.get("id")
    response_id = raw_response_id if isinstance(raw_response_id, str) else ""
    raw_returned_model = response_json.get("model")
    returned_model = raw_returned_model if isinstance(raw_returned_model, str) else ""
    requested_model = str(request.get("model", ""))
    dated_model = re.fullmatch(r"gpt-4\.1-(\d{4}-\d{2}-\d{2})", returned_model)
    dated_model_valid = False
    if dated_model is not None:
        try:
            date.fromisoformat(dated_model.group(1))
        except ValueError:
            dated_model_valid = False
        else:
            dated_model_valid = True
    usage = response_json.get("usage")
    if (
        not response_id
        or response_id != response_id.strip()
        or not returned_model
        or returned_model != returned_model.strip()
        or requested_model != "gpt-4.1"
        or not (returned_model == "gpt-4.1" or dated_model_valid)
        or not isinstance(usage, Mapping)
    ):
        raise ScoreBundleError(f"GEdit {label} provider identity or usage is invalid")
    token_counts: dict[str, int] = {}
    for token_field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(token_field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ScoreBundleError(
                f"GEdit {label} provider usage {token_field} is invalid"
            )
        token_counts[token_field] = value
    if (
        token_counts["prompt_tokens"] <= 0
        or token_counts["completion_tokens"] <= 0
        or token_counts["total_tokens"]
        != token_counts["prompt_tokens"] + token_counts["completion_tokens"]
    ):
        raise ScoreBundleError(f"GEdit {label} provider token usage arithmetic is invalid")
    choices = response_json.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ScoreBundleError(f"GEdit {label} provider response has invalid choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ScoreBundleError(f"GEdit {label} provider response has no message")
    content = message.get("content")
    score = _strict_gedit_provider_score(content, label=label)
    fingerprint = response_json.get("system_fingerprint")
    service_tier = response_json.get("service_tier")
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise ScoreBundleError(f"GEdit {label} system fingerprint is invalid")
    if service_tier is not None and not isinstance(service_tier, str):
        raise ScoreBundleError(f"GEdit {label} service tier is invalid")
    provenance = {
        "stage": label,
        "messages_sha256": messages_sha256,
        "image_parts_sha256": sha256_bytes(canonical_json_bytes(normalized_images)),
        "request_id": request_id,
        "response_id": response_id,
        "returned_model": returned_model,
        "system_fingerprint": fingerprint,
        "service_tier": service_tier,
        "usage": token_counts,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    return score, provenance


def _validate_gedit_score_evidence(
    *,
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    artifact_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    selection = contract["dataset"]["selection"]["records"]
    expected: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for item in selection:
        if not isinstance(item, Mapping):
            raise ScoreBundleError("GEdit contracted selection has invalid score identities")
        signature = (
            str(item.get("task_type", "")),
            str(item.get("instruction_language", "")),
            str(item.get("key", "")),
            str(item.get("instruction_sha256", "")),
        )
        if (
            any(not value for value in signature)
            or not str(item.get("identity", ""))
            or not str(item.get("task_type", ""))
            or signature in expected
        ):
            raise ScoreBundleError("GEdit contracted selection has invalid score identities")
        expected[signature] = item

    raw_payloads = {
        role: payload
        for role, payload in artifact_payloads.items()
        if role.startswith("raw_response:")
    }
    csv_payloads = {
        role: payload
        for role, payload in artifact_payloads.items()
        if role.startswith("score_csv:")
    }
    if len(raw_payloads) != len(expected):
        raise ScoreBundleError(
            "GEdit raw-response envelope cardinality differs from contracted selection"
        )
    judge_protocol = contract.get("judge", {}).get("protocol")
    if not isinstance(judge_protocol, Mapping):
        raise ScoreBundleError("GEdit contract has no judge protocol")
    result_dir = Path(str(summary.get("result_dir", "")))
    if not result_dir.is_absolute():
        raise ScoreBundleError("GEdit summary result_dir must be absolute")

    raw_scores: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    provenance_rows: list[dict[str, Any]] = []
    for role in sorted(raw_payloads):
        envelope = _load_json_bytes(raw_payloads[role], label=f"GEdit raw response {role}")
        if set(envelope) != {
            "schema",
            "identity",
            "instruction",
            "source_image",
            "source_request_jpeg",
            "edited_image",
            "provider_transactions",
            "derived_scores",
        } or envelope.get("schema") != "gedit-viescore-item-response/v2":
            raise ScoreBundleError(f"GEdit raw response {role} has the wrong schema")
        identity = envelope.get("identity")
        if not isinstance(identity, Mapping) or set(identity) != {
            "task_type",
            "instruction_language",
            "key",
            "instruction_sha256",
        }:
            raise ScoreBundleError(f"GEdit raw response {role} has an invalid identity")
        signature = (
            str(identity.get("task_type", "")),
            str(identity.get("instruction_language", "")),
            str(identity.get("key", "")),
            str(identity.get("instruction_sha256", "")),
        )
        item = expected.get(signature)
        instruction = envelope.get("instruction")
        if (
            item is None
            or signature in raw_scores
            or not isinstance(instruction, str)
            or hashlib.sha256(instruction.encode("utf-8")).hexdigest() != signature[3]
            or str(item.get("identity")) != f"{signature[0]}::{signature[1]}::{signature[2]}"
        ):
            raise ScoreBundleError(f"GEdit raw response {role} identity is unexpected or duplicate")
        contracted_source = item.get("source_image")
        contracted_source_jpeg = item.get("source_judge_image")
        source_image = envelope.get("source_image")
        source_request = envelope.get("source_request_jpeg")
        if (
            not isinstance(contracted_source, Mapping)
            or set(contracted_source)
            != {"representation", "width", "height", "bytes", "sha256"}
            or dict(contracted_source) != source_image
        ):
            raise ScoreBundleError(
                f"GEdit raw response {role} decoded source differs from its contract"
            )
        if (
            not isinstance(contracted_source_jpeg, Mapping)
            or set(contracted_source_jpeg)
            != {"representation", "width", "height", "bytes", "sha256"}
            or not isinstance(source_request, Mapping)
            or set(source_request)
            != {
                "representation",
                "width",
                "height",
                "bytes",
                "sha256",
                "data_base64",
            }
        ):
            raise ScoreBundleError(f"GEdit raw response {role} source JPEG schema is invalid")
        encoded_source = source_request.get("data_base64")
        if not isinstance(encoded_source, str) or not encoded_source:
            raise ScoreBundleError(f"GEdit raw response {role} has no source JPEG bytes")
        try:
            source_jpeg = base64.b64decode(encoded_source, validate=True)
            from PIL import Image

            with Image.open(io.BytesIO(source_jpeg)) as decoded_source:
                decoded_source.load()
                source_size = decoded_source.size
                source_format = decoded_source.format
        except Exception as exc:
            raise ScoreBundleError(
                f"GEdit raw response {role} source JPEG bytes are invalid"
            ) from exc
        source_metadata = {
            key: value for key, value in source_request.items() if key != "data_base64"
        }
        if (
            source_format != "JPEG"
            or source_size
            != (contracted_source_jpeg.get("width"), contracted_source_jpeg.get("height"))
            or source_metadata != dict(contracted_source_jpeg)
            or source_metadata.get("bytes") != len(source_jpeg)
            or source_metadata.get("sha256") != hashlib.sha256(source_jpeg).hexdigest()
        ):
            raise ScoreBundleError(
                f"GEdit raw response {role} source JPEG differs from its contract"
            )
        edited_image = envelope.get("edited_image")
        expected_image_path = result_dir / signature[0] / signature[1] / f"{signature[2]}.png"
        if not isinstance(edited_image, Mapping) or set(edited_image) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ScoreBundleError(f"GEdit raw response {role} has invalid edited-image evidence")
        image_raw = _stable_regular_file_bytes(
            expected_image_path,
            label=f"GEdit edited image {item['identity']}",
        )
        expected_image = {
            "path": str(expected_image_path),
            "bytes": len(image_raw),
            "sha256": hashlib.sha256(image_raw).hexdigest(),
        }
        if dict(edited_image) != expected_image:
            raise ScoreBundleError(f"GEdit raw response {role} edited image differs")
        transactions = envelope.get("provider_transactions")
        if not isinstance(transactions, Mapping) or set(transactions) != {
            "semantic_consistency",
            "perceptual_quality",
        }:
            raise ScoreBundleError(f"GEdit raw response {role} requires exactly SC and PQ")
        edited_jpeg = _gedit_encoded_edited_image_bytes(image_raw)
        expected_requests = _gedit_expected_request_evidence(
            instruction=instruction,
            source_jpeg=source_jpeg,
            edited_jpeg=edited_jpeg,
        )
        semantics, sc_provenance = _validate_gedit_provider_transaction(
            transactions["semantic_consistency"],
            label="semantic_consistency",
            judge_protocol=judge_protocol,
            expected_request_evidence=expected_requests["semantic_consistency"],
        )
        quality, pq_provenance = _validate_gedit_provider_transaction(
            transactions["perceptual_quality"],
            label="perceptual_quality",
            judge_protocol=judge_protocol,
            expected_request_evidence=expected_requests["perceptual_quality"],
        )
        derived = envelope.get("derived_scores")
        overall = math.sqrt(semantics * quality)
        if not isinstance(derived, Mapping) or set(derived) != {
            "semantics",
            "quality",
            "overall",
        }:
            raise ScoreBundleError(f"GEdit raw response {role} has invalid derived scores")
        raw_derived_values = tuple(
            derived[key] for key in ("semantics", "quality", "overall")
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_derived_values
        ):
            raise ScoreBundleError(
                f"GEdit raw response {role} has non-numeric derived scores"
            )
        derived_values = tuple(float(value) for value in raw_derived_values)
        if (
            any(not math.isfinite(value) for value in derived_values)
            or derived_values != (semantics, quality, overall)
        ):
            raise ScoreBundleError(
                f"GEdit raw response {role} derived scores differ from provider content"
            )
        raw_scores[signature] = (semantics, quality)
        for transaction in (sc_provenance, pq_provenance):
            provenance_rows.append({"identity": str(item["identity"]), **transaction})

    if set(raw_scores) != set(expected):
        raise ScoreBundleError("GEdit raw responses differ from contracted selection")
    request_ids = [str(row["request_id"]) for row in provenance_rows]
    response_ids = [str(row["response_id"]) for row in provenance_rows]
    if len(set(request_ids)) != len(request_ids) or len(set(response_ids)) != len(response_ids):
        raise ScoreBundleError("GEdit successful provider request/response IDs are not unique")
    returned_models = sorted({str(row["returned_model"]) for row in provenance_rows})
    if len(returned_models) != 1:
        raise ScoreBundleError("GEdit score bundle used inconsistent returned model identities")

    required_fields = {
        "key",
        "instruction",
        "sementics_score",
        "quality_score",
        "instruction_language",
    }
    records: list[dict[str, Any]] = []
    observed: set[tuple[str, str, str, str]] = set()
    observed_csv_groups: set[str] = set()
    for role in sorted(csv_payloads):
        try:
            text = csv_payloads[role].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ScoreBundleError(f"GEdit score artifact {role} is not UTF-8 CSV") from exc
        reader = csv.DictReader(io.StringIO(text, newline=""))
        fields = reader.fieldnames or []
        duplicate_fields = [field for field, count in Counter(fields).items() if count > 1]
        if duplicate_fields or not required_fields.issubset(fields):
            raise ScoreBundleError(f"GEdit score artifact {role} has an invalid CSV header")
        role_order: list[tuple[str, str]] = []
        role_groups: set[str] = set()
        for row in reader:
            if None in row or any(row.get(field) is None for field in required_fields):
                raise ScoreBundleError(f"GEdit score artifact {role} has a malformed CSV row")
            row_signature = (
                str(row["instruction_language"]),
                str(row["key"]),
                hashlib.sha256(str(row["instruction"]).encode("utf-8")).hexdigest(),
            )
            candidates = [
                candidate_signature
                for candidate_signature in expected
                if candidate_signature[1:] == row_signature
            ]
            if len(candidates) != 1:
                raise ScoreBundleError(
                    f"GEdit score artifact {role} has an ambiguous or unexpected row"
                )
            signature = candidates[0]
            item = expected.get(signature)
            if item is None or signature in observed:
                raise ScoreBundleError(
                    f"GEdit score artifact {role} has an unexpected or duplicate row"
                )
            observed.add(signature)
            role_order.append((signature[1], signature[2]))
            role_groups.add(signature[0])
            try:
                semantics = float(row["sementics_score"])
                quality = float(row["quality_score"])
            except (TypeError, ValueError) as exc:
                raise ScoreBundleError(f"GEdit score artifact {role} has a non-numeric score") from exc
            if (
                not math.isfinite(semantics)
                or not math.isfinite(quality)
                or not 0.0 <= semantics <= 10.0
                or not 0.0 <= quality <= 10.0
            ):
                raise ScoreBundleError(f"GEdit score artifact {role} has an invalid score")
            raw_pair = raw_scores.get(signature)
            if (
                raw_pair is None
                or semantics != raw_pair[0]
                or quality != raw_pair[1]
            ):
                raise ScoreBundleError(
                    f"GEdit score artifact {role} differs from raw provider responses"
                )
            records.append(
                {
                    "identity": str(item["identity"]),
                    "task_type": str(item["task_type"]),
                    "instruction_language": str(item["instruction_language"]),
                    "semantics": semantics,
                    "quality": quality,
                    "overall": math.sqrt(semantics * quality),
                }
            )
        if not role_order:
            raise ScoreBundleError(f"GEdit score artifact {role} is empty")
        if role_order != sorted(role_order):
            raise ScoreBundleError(
                f"GEdit score artifact {role} is not sorted by (instruction_language, key)"
            )
        if len(role_groups) != 1:
            raise ScoreBundleError(f"GEdit score artifact {role} mixes task groups")
        role_group = next(iter(role_groups))
        if role_group in observed_csv_groups:
            raise ScoreBundleError(f"GEdit score artifacts duplicate task group {role_group}")
        observed_csv_groups.add(role_group)
    if observed != set(expected):
        raise ScoreBundleError("GEdit raw score rows differ from contracted selection")
    expected_csv_groups = {signature[0] for signature in expected}
    if observed_csv_groups != expected_csv_groups or len(csv_payloads) != len(
        expected_csv_groups
    ):
        raise ScoreBundleError("GEdit score CSVs do not map one-to-one to task groups")
    records.sort(key=lambda item: str(item["identity"]))
    scoring = summary.get("scoring")
    backbone = scoring.get("backbone") if isinstance(scoring, Mapping) else None
    contract_judge = contract.get("judge")
    contracted_backbone = (
        contract_judge.get("backbone") if isinstance(contract_judge, Mapping) else None
    )
    if (
        not isinstance(backbone, str)
        or not backbone
        or not isinstance(contracted_backbone, str)
        or backbone != contracted_backbone
    ):
        raise ScoreBundleError("GEdit summary scoring backbone differs from its contract")
    expected_metrics = derive_gedit_metrics(records, backbone=backbone)
    if summary.get("metrics") != expected_metrics:
        raise ScoreBundleError(
            "GEdit summary metrics are not an exact derivation of raw provider responses"
        )
    provenance_rows.sort(key=lambda row: (str(row["identity"]), str(row["stage"])))
    fingerprints = sorted(
        {str(row["system_fingerprint"]) for row in provenance_rows if row["system_fingerprint"] is not None}
    )
    service_tiers = sorted(
        {str(row["service_tier"]) for row in provenance_rows if row["service_tier"] is not None}
    )
    return {
        "schema": "gedit-provider-evidence/v2",
        "envelope_count": len(raw_scores),
        "transaction_count": len(provenance_rows),
        "returned_models": returned_models,
        "request_ids": sorted(request_ids),
        "request_ids_sha256": sha256_bytes(canonical_json_bytes(sorted(request_ids))),
        "response_ids": sorted(response_ids),
        "response_ids_sha256": sha256_bytes(canonical_json_bytes(sorted(response_ids))),
        "system_fingerprints": fingerprints,
        "system_fingerprint_null_count": sum(
            row["system_fingerprint"] is None for row in provenance_rows
        ),
        "service_tiers": service_tiers,
        "service_tier_null_count": sum(row["service_tier"] is None for row in provenance_rows),
        "evidence_sha256": sha256_bytes(canonical_json_bytes(provenance_rows)),
    }


def _validate_score_artifact_evidence(
    *,
    benchmark: str,
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    artifact_payloads: Mapping[str, bytes],
) -> dict[str, Any] | None:
    _validate_artifact_roles(benchmark, set(artifact_payloads))
    if benchmark == "imgedit":
        _validate_imgedit_score_evidence(
            summary=summary,
            contract=contract,
            artifact_payloads=artifact_payloads,
        )
        return None
    return _validate_gedit_score_evidence(
        summary=summary,
        contract=contract,
        artifact_payloads=artifact_payloads,
    )


def _receipt_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return result


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> Path:
    """Publish canonical JSON once; never overwrite an existing score artifact."""

    path = Path(path)
    _reject_symlink_path_components(path.parent, label="score artifact parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path_components(path.parent, label="score artifact parent")
    raw = canonical_json_bytes(dict(payload)) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ScoreBundleError(
            f"Refusing to overwrite immutable score artifact {path}; use a new --attempt-id"
        ) from exc
    return path


def acquire_score_bundle_claim(
    score_root: Path,
    *,
    benchmark: str,
    contract_id: str,
) -> Path:
    """Atomically claim a fresh score namespace before any judge request is issued."""

    return write_json_exclusive(
        Path(score_root) / SCORE_BUNDLE_CLAIM_FILENAME,
        {
            "schema": SCORE_BUNDLE_CLAIM_SCHEMA,
            "benchmark": benchmark,
            "contract_id": contract_id,
        },
    )


def release_score_bundle_claim(
    claim_path: Path,
    *,
    benchmark: str,
    contract_id: str,
) -> None:
    """Remove only the exact claim created for a now-complete immutable bundle."""

    expected = canonical_json_bytes(
        {
            "schema": SCORE_BUNDLE_CLAIM_SCHEMA,
            "benchmark": benchmark,
            "contract_id": contract_id,
        }
    ) + b"\n"
    actual = _stable_regular_file_bytes(Path(claim_path), label="score bundle claim")
    if actual != expected:
        raise ScoreBundleError("Score bundle claim differs from the active contract")
    Path(claim_path).unlink()


def _score_attempt_evidence(
    *,
    summary_path: Path,
    summary: Mapping[str, Any],
    benchmark: str,
    contract_id: str,
) -> dict[str, Any]:
    attempt = summary.get("score_attempt")
    if not isinstance(attempt, Mapping):
        raise ScoreBundleError("Score summary has no complete attempt namespace binding")
    attempt_id = str(attempt.get("attempt_id", ""))
    expected = {
        **_score_attempt_manifest(
            attempt_id=attempt_id,
            benchmark=benchmark,
            contract_id=contract_id,
        ),
        "namespace": f"complete/{attempt_id}",
    }
    if dict(attempt) != expected:
        raise ScoreBundleError("Score summary attempt binding is invalid")
    score_root = Path(summary_path).parent
    if (
        SCORE_ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None
        or ".." in attempt_id
        or score_root.name != attempt_id
        or score_root.parent.name != "complete"
        or score_root.parent.parent.name != SCORE_ATTEMPTS_DIRECTORY
    ):
        raise ScoreBundleError(
            "Score summary is outside its declared complete attempt namespace"
        )
    manifest_raw = _stable_regular_file_bytes(
        score_root / SCORE_ATTEMPT_MANIFEST_FILENAME,
        label="score attempt manifest",
    )
    manifest = _load_json_bytes(manifest_raw, label="score attempt manifest")
    expected_manifest = _score_attempt_manifest(
        attempt_id=attempt_id,
        benchmark=benchmark,
        contract_id=contract_id,
    )
    if manifest != expected_manifest:
        raise ScoreBundleError("Completed score attempt manifest differs from its summary")
    return expected


def build_score_bundle_receipt(
    *,
    summary_path: Path,
    result_dir: Path,
    artifacts: Mapping[str, Path],
    benchmark: str,
) -> dict[str, Any]:
    summary_path = Path(summary_path)
    score_root = summary_path.parent
    summary_raw = _stable_regular_file_bytes(summary_path, label="score summary")
    summary = _load_json_bytes(summary_raw, label="score summary")
    contract_id, contract, judge_protocol, records, record_digest = _summary_evidence(summary)
    if str(summary.get("benchmark", "")) != benchmark:
        raise ScoreBundleError(
            f"Score summary benchmark differs: expected {benchmark}, found {summary.get('benchmark')!r}"
        )
    receipt_ref = summary.get("score_receipt")
    if receipt_ref != {"schema": SCORE_BUNDLE_SCHEMA, "path": SCORE_BUNDLE_RECEIPT_FILENAME}:
        raise ScoreBundleError("Score summary does not declare the canonical score receipt")
    score_attempt = _score_attempt_evidence(
        summary_path=summary_path,
        summary=summary,
        benchmark=benchmark,
        contract_id=contract_id,
    )

    result_dir = Path(result_dir)
    declared_result_dir = Path(str(summary.get("result_dir", "")))
    if (
        not declared_result_dir.is_absolute()
        or declared_result_dir.absolute() != result_dir.absolute()
    ):
        raise ScoreBundleError("Score summary result_dir differs from the scored output directory")
    output_manifest = result_dir / OUTPUT_MANIFEST_FILENAME
    output_raw = _stable_regular_file_bytes(
        output_manifest,
        label="generated output manifest",
    )
    _validate_generated_output_manifest(
        result_dir=result_dir,
        manifest_raw=output_raw,
        contract_id=contract_id,
        contract=contract,
        benchmark=benchmark,
    )
    artifact_items = [
        _file_record(Path(path), root=score_root, role=str(role))
        for role, path in sorted(artifacts.items())
    ]
    artifact_records = [record for record, _ in artifact_items]
    artifact_payloads = {
        str(record["role"]): raw for record, raw in artifact_items
    }
    if not artifact_records or len({item["role"] for item in artifact_records}) != len(
        artifact_records
    ):
        raise ScoreBundleError("Score bundle artifact roles must be non-empty and unique")
    declared_artifacts = summary.get("score_artifacts")
    expected_declaration = {item["role"]: item["path"] for item in artifact_records}
    if declared_artifacts != expected_declaration:
        raise ScoreBundleError("Score summary score_artifacts differ from published artifacts")
    provider_evidence = _validate_score_artifact_evidence(
        benchmark=benchmark,
        summary=summary,
        contract=contract,
        artifact_payloads=artifact_payloads,
    )

    payload = {
        "schema": SCORE_BUNDLE_SCHEMA,
        "benchmark": benchmark,
        "contract_id": contract_id,
        "score_attempt": score_attempt,
        "generated_output_manifest": {
            "path": OUTPUT_MANIFEST_FILENAME,
            "bytes": len(output_raw),
            "sha256": hashlib.sha256(output_raw).hexdigest(),
        },
        "judge_protocol": {
            "value": judge_protocol,
            "sha256": sha256_bytes(canonical_json_bytes(judge_protocol)),
        },
        "record_evidence": {
            "count": len(records),
            "sha256": record_digest,
        },
        "provider_evidence": provider_evidence,
        "summary": {
            "path": summary_path.name,
            "bytes": len(summary_raw),
            "sha256": hashlib.sha256(summary_raw).hexdigest(),
        },
        "artifacts": artifact_records,
    }
    return _receipt_payload(payload)


def write_score_bundle_receipt(
    *,
    summary_path: Path,
    result_dir: Path,
    artifacts: Mapping[str, Path],
    benchmark: str,
) -> Path:
    payload = build_score_bundle_receipt(
        summary_path=summary_path,
        result_dir=result_dir,
        artifacts=artifacts,
        benchmark=benchmark,
    )
    return write_json_exclusive(Path(summary_path).parent / SCORE_BUNDLE_RECEIPT_FILENAME, payload)


def validate_score_summary_bundle(
    summary_path: Path,
    *,
    expected_benchmark: str | None = None,
) -> dict[str, Any]:
    """Reopen and authenticate a raw-score bundle before promotion consumes it.

    The returned summary payload is parsed from the exact bytes whose digest was
    checked against the receipt, avoiding a second unverified summary read.
    """

    summary_path = Path(summary_path)
    summary_raw = _stable_regular_file_bytes(summary_path, label="score summary")
    summary = _load_json_bytes(summary_raw, label="score summary")
    benchmark = str(summary.get("benchmark", ""))
    if expected_benchmark is not None and benchmark != expected_benchmark:
        raise ScoreBundleError(
            f"Score summary benchmark differs: expected {expected_benchmark}, found {benchmark!r}"
        )
    contract_id, contract, judge_protocol, records, record_digest = _summary_evidence(summary)
    score_attempt = _score_attempt_evidence(
        summary_path=summary_path,
        summary=summary,
        benchmark=benchmark,
        contract_id=contract_id,
    )
    receipt_ref = summary.get("score_receipt")
    if not isinstance(receipt_ref, Mapping) or receipt_ref.get("schema") != SCORE_BUNDLE_SCHEMA:
        raise ScoreBundleError("Score summary has no supported score receipt")
    receipt_path = _safe_relative(
        summary_path.parent,
        receipt_ref.get("path"),
        label="score receipt",
    )
    if receipt_path.name != SCORE_BUNDLE_RECEIPT_FILENAME:
        raise ScoreBundleError("Score summary points to a non-canonical score receipt name")
    receipt_raw = _stable_regular_file_bytes(receipt_path, label="score receipt")
    receipt = _load_json_bytes(receipt_raw, label="score receipt")
    payload_digest = str(receipt.get("payload_sha256", ""))
    unsigned = dict(receipt)
    unsigned.pop("payload_sha256", None)
    if payload_digest != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ScoreBundleError("Score receipt payload digest is invalid")
    if receipt.get("schema") != SCORE_BUNDLE_SCHEMA:
        raise ScoreBundleError(f"Unsupported score receipt schema: {receipt.get('schema')!r}")
    if receipt.get("benchmark") != benchmark or receipt.get("contract_id") != contract_id:
        raise ScoreBundleError("Score receipt benchmark or contract ID differs from summary")
    if receipt.get("score_attempt") != score_attempt:
        raise ScoreBundleError("Score receipt attempt binding differs from summary")

    summary_record = receipt.get("summary")
    expected_summary = {
        "path": summary_path.name,
        "bytes": len(summary_raw),
        "sha256": hashlib.sha256(summary_raw).hexdigest(),
    }
    if summary_record != expected_summary:
        raise ScoreBundleError("Score receipt summary bytes no longer match")
    expected_judge = {
        "value": judge_protocol,
        "sha256": sha256_bytes(canonical_json_bytes(judge_protocol)),
    }
    if receipt.get("judge_protocol") != expected_judge:
        raise ScoreBundleError("Score receipt judge protocol differs from summary")
    if receipt.get("record_evidence") != {
        "count": len(records),
        "sha256": record_digest,
    }:
        raise ScoreBundleError("Score receipt record evidence differs from summary")

    result_dir = Path(str(summary.get("result_dir", "")))
    if not result_dir.is_absolute():
        raise ScoreBundleError("Score summary result_dir must be absolute")
    output_record = receipt.get("generated_output_manifest")
    if not isinstance(output_record, Mapping):
        raise ScoreBundleError("Score receipt has no generated-output manifest binding")
    output_path = _safe_relative(
        result_dir,
        output_record.get("path"),
        label="generated output manifest",
    )
    output_raw = _stable_regular_file_bytes(output_path, label="generated output manifest")
    if output_record != {
        "path": OUTPUT_MANIFEST_FILENAME,
        "bytes": len(output_raw),
        "sha256": hashlib.sha256(output_raw).hexdigest(),
    }:
        raise ScoreBundleError("Generated-output manifest differs from score receipt")
    _validate_generated_output_manifest(
        result_dir=result_dir,
        manifest_raw=output_raw,
        contract_id=contract_id,
        contract=contract,
        benchmark=benchmark,
    )

    artifact_rows = receipt.get("artifacts")
    if not isinstance(artifact_rows, list) or not artifact_rows:
        raise ScoreBundleError("Score receipt has no raw score artifacts")
    seen_roles: set[str] = set()
    declared: dict[str, str] = {}
    artifact_payloads: dict[str, bytes] = {}
    for index, raw_record in enumerate(artifact_rows):
        if not isinstance(raw_record, Mapping):
            raise ScoreBundleError(f"Score artifact receipt row {index} is not a mapping")
        role = str(raw_record.get("role", ""))
        if not role or role in seen_roles:
            raise ScoreBundleError(f"Duplicate or empty score artifact role: {role!r}")
        seen_roles.add(role)
        artifact_path = _safe_relative(
            summary_path.parent,
            raw_record.get("path"),
            label=f"score artifact {role}",
        )
        artifact_raw = _stable_regular_file_bytes(artifact_path, label=f"score artifact {role}")
        expected_record = {
            "role": role,
            "path": artifact_path.relative_to(summary_path.parent).as_posix(),
            "bytes": len(artifact_raw),
            "sha256": hashlib.sha256(artifact_raw).hexdigest(),
        }
        if dict(raw_record) != expected_record:
            raise ScoreBundleError(f"Score artifact {role} differs from its receipt")
        declared[role] = expected_record["path"]
        artifact_payloads[role] = artifact_raw
    if summary.get("score_artifacts") != declared:
        raise ScoreBundleError("Score summary score_artifacts differ from receipt")
    provider_evidence = _validate_score_artifact_evidence(
        benchmark=benchmark,
        summary=summary,
        contract=contract,
        artifact_payloads=artifact_payloads,
    )
    if receipt.get("provider_evidence") != provider_evidence:
        raise ScoreBundleError("Score receipt provider evidence differs from raw artifacts")

    # Recheck the summary identity after all dependent files were read. The caller
    # consumes the already parsed payload, not a later unverified filesystem read.
    final_summary_raw = _stable_regular_file_bytes(summary_path, label="score summary")
    if final_summary_raw != summary_raw:
        raise ScoreBundleError("Score summary changed during bundle validation")
    return {
        "summary_payload": summary,
        "summary_path": str(summary_path),
        "summary_sha256": expected_summary["sha256"],
        "receipt": receipt,
    }
