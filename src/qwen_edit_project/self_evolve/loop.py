from __future__ import annotations

import json
import logging
import math
import gc
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from qwen_edit_project.self_evolve.backends import (
    DifficultyController,
    QwenEditEditor,
    build_editor,
    build_evaluator,
    build_proposer,
)
from qwen_edit_project.self_evolve.data import load_unlabeled_records
from qwen_edit_project.self_evolve.proposer_training import (
    build_proposer_training_records,
    write_proposer_training_jsonl,
)
from qwen_edit_project.self_evolve.types import (
    AcceptedSample,
    EditProposal,
    EvaluationResult,
    ProposalDefinition,
    UnlabeledImageRecord,
)
from qwen_edit_project.train.launch_train import build_accelerate_command, build_train_command
from qwen_edit_project.utils.commands import run_and_tee, shell_join
from qwen_edit_project.utils.config import load_yaml_config, save_json
from qwen_edit_project.utils.paths import ensure_dir, relative_to_repo, resolve_path
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


OBJECT_CONTRACT_EDIT_TYPES = {"object_removal", "object_replacement"}
OBJECT_SLOT_ACTION_PHRASES = (
    "fill the area",
    "area naturally",
    "cleanly filled",
    "after removing",
    "remains visible",
    "remain visible",
    "still visible",
    "remove ",
    "delete ",
    "erase ",
    "replace ",
    "inpaint",
    "preserve ",
    "unchanged",
    "surrounding",
)
GENERIC_OBJECT_SLOT_TERMS = {
    "object",
    "item",
    "thing",
    "area",
    "region",
    "target",
    "main visible target",
    "original location",
    "same location",
}

EDIT_FAMILY_TO_SCHEDULE_TYPES = {
    "object": (
        "object_removal",
        "object_replacement",
        "object_addition",
        "subject_extraction",
        "spatial_move",
        "attribute_change",
        "color_change",
        "material_change",
        "local_enhancement",
        "compose",
        "action_change",
    ),
    "background": (
        "background_change",
        "subject_extraction",
        "style_transfer",
        "color_change",
        "local_enhancement",
        "object_addition",
        "compose",
    ),
    "color": (
        "color_change",
        "attribute_change",
        "local_enhancement",
        "style_transfer",
    ),
    "contrast": (
        "local_enhancement",
        "color_change",
    ),
    "exposure": (
        "local_enhancement",
        "color_change",
    ),
    "local": (
        "local_enhancement",
        "attribute_change",
        "color_change",
        "material_change",
    ),
    "style": (
        "style_transfer",
        "background_change",
        "attribute_change",
    ),
    "spatial": (
        "spatial_move",
        "object_replacement",
        "object_addition",
        "action_change",
    ),
    "material": (
        "material_change",
        "attribute_change",
        "local_enhancement",
    ),
}


def write_jsonl(items: list[dict[str, Any]], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def append_jsonl(item: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    if line_number == len(lines):
                        break
                    raise ValueError(f"Invalid JSONL row in {path} at line {line_number}") from exc
    return rows


def discover_latest_checkpoint(directory: Path) -> Path | None:
    candidates = sorted(directory.rglob("*.safetensors"), key=lambda path: path.stat().st_mtime_ns)
    if not candidates:
        return None
    return candidates[-1]


def discover_latest_step_checkpoint(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    checkpoints = []
    for path in directory.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda item: item[0])[-1][1]


def training_checkpoint_backend(training_cfg: dict[str, Any], train_config: dict[str, Any]) -> str:
    return str(
        train_config.get("output", {}).get("checkpoint_backend")
        or training_cfg.get("trained_checkpoint_backend", "diffsynth")
    )


def _append_flag(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a shallow or dotted metadata key without forcing a schema migration."""
    current: Any = metadata
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "null", "disabled"}:
        return False
    return default


def _coerce_int_map(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            parsed_items = []
            for part in stripped.split(","):
                if not part.strip() or ":" not in part:
                    continue
                key, raw_count = part.split(":", 1)
                parsed_items.append((key.strip(), raw_count.strip()))
            items = parsed_items
        else:
            if not isinstance(decoded, dict):
                return {}
            items = decoded.items()
    else:
        return {}

    output: dict[str, int] = {}
    for key, raw_count in items:
        name = str(key).strip()
        if not name:
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if count > 0:
            output[name] = count
    return output


def _coerce_bool_map(value: Any) -> dict[str, bool]:
    if value is None:
        return {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            parsed_items = []
            for part in stripped.split(","):
                if not part.strip() or ":" not in part:
                    continue
                key, raw_value = part.split(":", 1)
                parsed_items.append((key.strip(), raw_value.strip()))
            items = parsed_items
        else:
            if not isinstance(decoded, dict):
                return {}
            items = decoded.items()
    else:
        return {}

    output: dict[str, bool] = {}
    for key, raw_value in items:
        name = str(key).strip()
        if not name:
            continue
        output[name] = _coerce_bool(raw_value)
    return output


def _coerce_float_map(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            parsed_items = []
            for part in stripped.split(","):
                if not part.strip() or ":" not in part:
                    continue
                key, raw_count = part.split(":", 1)
                parsed_items.append((key.strip(), raw_count.strip()))
            items = parsed_items
        else:
            if not isinstance(decoded, dict):
                return {}
            items = decoded.items()
    else:
        return {}

    output: dict[str, float] = {}
    for key, raw_value in items:
        name = str(key).strip()
        if not name:
            continue
        value = _finite_float(raw_value, math.nan)
        if math.isfinite(value) and value >= 0.0:
            output[name] = value
    return output


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = [value]
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _resolve_num_rounds(value: Any, record_count: int, max_records_per_round: int) -> int:
    if isinstance(value, str) and value.lower() in {"auto", "all", "cover_dataset"}:
        if record_count <= 0:
            return 0
        return max(1, math.ceil(record_count / max(1, max_records_per_round)))
    return int(value)


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _process_memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"pid": os.getpid()}
    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmPeak:")):
                    key, value = line.split(":", 1)
                    parts = value.strip().split()
                    if parts:
                        snapshot[f"{key.lower()}_kb"] = int(parts[0])
        except OSError:
            pass
    try:
        import resource

        snapshot["ru_maxrss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            snapshot.update(
                {
                    "cuda_free_mb": round(free_bytes / (1024**2), 1),
                    "cuda_total_mb": round(total_bytes / (1024**2), 1),
                    "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
                    "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
                    "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
                    "cuda_max_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
                }
            )
    except Exception:
        pass
    return snapshot


EDITOR_TRAINING_OVERRIDE_KEYS = {
    "learning_rate",
    "num_epochs",
    "max_train_steps",
    "train_batch_size",
    "gradient_accumulation_steps",
    "checkpointing_steps",
    "checkpoints_total_limit",
    "resume_from_checkpoint",
    "mixed_precision",
    "use_8bit_adam",
    "offload",
    "training_objective",
    "preference_beta",
    "preference_margin",
    "preference_sft_weight",
    "preference_sdpo_epsilon",
    "preference_reference_mode",
}


class SelfEvolveRunner:
    def __init__(self, config: dict[str, Any], dry_run: bool = False, limit: int | None = None):
        self.config = config
        self.dry_run = dry_run or bool(config.get("runtime", {}).get("dry_run", False))
        self._enforce_slurm_policy()
        dataset_limit = config.get("dataset", {}).get("limit")
        self.limit = limit if limit is not None else dataset_limit
        self.records = load_unlabeled_records(config["dataset"], limit=self.limit)
        self.output_root = ensure_dir(resolve_path(config["output"]["root_dir"]))
        self.logger = self._build_logger()
        self.proposer = build_proposer(config["proposer"])
        self.editor = build_editor(config["editor"])
        evaluator_config = config.get("evaluator", config.get("solver"))
        if evaluator_config is None:
            raise ValueError("self-evolve config requires an evaluator: section; solver: is still accepted as an alias")
        self.evaluator = build_evaluator(evaluator_config)
        self.solver = self.evaluator  # Backward-compatible alias for older code paths.
        self.initial_editor_state = self.editor.model_state() if isinstance(self.editor, QwenEditEditor) else None
        self.previous_editor_state: dict[str, Any] | None = None
        curriculum = config["curriculum"]
        self.difficulty_controller = DifficultyController(
            initial_level=int(curriculum.get("initial_level", 1)),
            min_level=int(curriculum.get("min_level", 1)),
            max_level=int(curriculum.get("max_level", 3)),
            promote_at=float(curriculum.get("promote_at", 0.75)),
            demote_at=float(curriculum.get("demote_at", 0.45)),
        )
        self.record_by_key = {record.key: record for record in self.records}

    def _enforce_slurm_policy(self) -> None:
        if self.dry_run or os.environ.get("ALLOW_LOGIN_NODE") == "1" or os.environ.get("SLURM_JOB_ID"):
            return
        editor_cfg = self.config.get("editor", {})
        proposer_cfg = self.config.get("proposer", {})
        training_cfg = self.config.get("training", {})
        proposer_training_cfg = proposer_cfg.get("training", {}) if isinstance(proposer_cfg, dict) else {}
        requires_gpu = (
            editor_cfg.get("backend") == "qwen_edit"
            or str(proposer_cfg.get("backend", "")).startswith("trainable_qwen")
            or training_cfg.get("trigger") == "launch"
            or proposer_training_cfg.get("trigger") == "launch"
        )
        if requires_gpu:
            raise SystemExit(
                "Refusing to run Qwen self-evolve/training outside a Slurm allocation. "
                "Start a tmux session, request one GPU with srun, activate qedit, then run this command."
            )

    def _build_logger(self) -> logging.Logger:
        ensure_dir(self.output_root)
        logger = logging.getLogger(f"qwen_edit_project.self_evolve.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(self.output_root / "self_evolve.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.handlers = [stream_handler, file_handler]
        return logger

    def _release_resident_models(self, reason: str) -> None:
        released: list[str] = []
        for name in ("editor", "proposer", "evaluator"):
            component = getattr(self, name, None)
            release = getattr(component, "release_memory", None)
            if callable(release):
                release()
                released.append(name)
        if released:
            self.logger.info("Released resident model memory before %s: %s.", reason, ", ".join(released))

    def _release_component_model(self, component_name: str, reason: str) -> None:
        component = getattr(self, component_name, None)
        release = getattr(component, "release_memory", None)
        if callable(release):
            release()
            self.logger.info("Released %s model memory before %s.", component_name, reason)

    def _log_memory(self, event: str, **details: Any) -> None:
        runtime_cfg = self.config.get("runtime", {})
        if not bool(runtime_cfg.get("memory_log_enabled", True)):
            return
        payload = {"event": event, **details, **_process_memory_snapshot()}
        self.logger.info("MEMORY %s", json.dumps(payload, ensure_ascii=True, sort_keys=True))

    def _candidate_payload(
        self,
        record: UnlabeledImageRecord,
        proposal: EditProposal,
        evaluation_result: EvaluationResult | None,
        image_path: Path | None,
        status: str,
        candidate_index: int = 0,
        group_id: str | None = None,
        distractors: list[dict[str, Any]] | None = None,
        candidate_role: str = "policy",
        candidate_model_state: dict[str, Any] | None = None,
        candidate_seed: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "group_id": group_id,
            "candidate_index": candidate_index,
            "candidate_role": candidate_role,
            "candidate_model_state": candidate_model_state,
            "candidate_seed": candidate_seed,
            "record_key": record.key,
            "image_path": relative_to_repo(record.image_path),
            "caption": record.caption,
            "proposal": {
                "round_index": proposal.round_index,
                "proposal_index": proposal.proposal_index,
                "operation_id": proposal.definition.operation_id,
                "family": proposal.definition.family,
                "difficulty": proposal.definition.difficulty,
                "difficulty_level": proposal.difficulty_level,
                "instruction": proposal.instruction,
                "scope": proposal.definition.scope,
                "metric": proposal.definition.metric,
                "direction": proposal.definition.direction,
                "target": proposal.definition.target,
                "expected_changed_fraction": list(proposal.definition.expected_changed_fraction),
                "verifier": proposal.definition.verifier,
                "inverse_operation_id": proposal.definition.inverse_operation_id,
                "structured_edit": proposal.structured_edit,
            },
            "distractors": distractors or [],
            "status": status,
            "edited_image_path": relative_to_repo(image_path) if image_path is not None else None,
        }
        if evaluation_result is not None:
            evaluator_payload = {
                "global_score": evaluation_result.global_score,
                "local_score": evaluation_result.local_score,
                "total_score": evaluation_result.total_score,
                "accepted": evaluation_result.accepted,
                "component_scores": evaluation_result.component_scores,
                "signals": evaluation_result.signals,
            }
            payload["evaluator"] = evaluator_payload
            payload["solver"] = evaluator_payload
        return payload

    @staticmethod
    def _candidate_key(payload: dict[str, Any]) -> tuple[str, int]:
        return str(payload.get("group_id", "")), int(payload.get("candidate_index", -1))

    @staticmethod
    def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group_id = str(row.get("group_id") or "")
            if group_id:
                grouped[group_id].append(row)
        for values in grouped.values():
            values.sort(key=lambda row: int(row.get("candidate_index", 0)))
        return dict(grouped)

    @staticmethod
    def _self_play_cfg_from_candidate_generation(candidate_cfg: dict[str, Any]) -> dict[str, Any]:
        raw_cfg = candidate_cfg.get("self_play") or candidate_cfg.get("opponent_self_play") or {}
        return dict(raw_cfg) if isinstance(raw_cfg, dict) else {}

    @staticmethod
    def _reference_cfg_from_candidate_generation(candidate_cfg: dict[str, Any]) -> dict[str, Any]:
        raw_cfg = (
            candidate_cfg.get("reference_candidates")
            or candidate_cfg.get("base_reference")
            or candidate_cfg.get("no_harm_reference")
            or {}
        )
        return dict(raw_cfg) if isinstance(raw_cfg, dict) else {}

    @staticmethod
    def _self_play_enabled_for_round(self_play_cfg: dict[str, Any], round_index: int) -> bool:
        if not bool(self_play_cfg.get("enabled", False)):
            return False
        start_round = int(self_play_cfg.get("start_round", 2))
        end_round = self_play_cfg.get("end_round")
        if round_index < start_round:
            return False
        if end_round is not None and round_index > int(end_round):
            return False
        return True

    def _self_play_samples_for_round(self, self_play_cfg: dict[str, Any], round_index: int) -> int:
        if not self._self_play_enabled_for_round(self_play_cfg, round_index):
            return 0
        return max(0, int(self_play_cfg.get("samples_per_proposal", 1)))

    @staticmethod
    def _reference_candidates_enabled_for_round(reference_cfg: dict[str, Any], round_index: int) -> bool:
        if not bool(reference_cfg.get("enabled", False)):
            return False
        start_round = int(reference_cfg.get("start_round", 1))
        end_round = reference_cfg.get("end_round")
        if round_index < start_round:
            return False
        if end_round is not None and round_index > int(end_round):
            return False
        return True

    def _reference_samples_for_round(self, reference_cfg: dict[str, Any], round_index: int) -> int:
        if not self._reference_candidates_enabled_for_round(reference_cfg, round_index):
            return 0
        return max(0, int(reference_cfg.get("samples_per_proposal", 1)))

    @staticmethod
    def _proposal_edit_type(proposal: EditProposal) -> str:
        structured = proposal.structured_edit or {}
        return str(structured.get("edit_type") or proposal.definition.family or "").strip()

    @staticmethod
    def _payload_edit_type(payload: dict[str, Any]) -> str:
        proposal = payload.get("proposal", {})
        if not isinstance(proposal, dict):
            return "unknown"
        structured_edit = proposal.get("structured_edit", {})
        if isinstance(structured_edit, dict):
            edit_type = structured_edit.get("edit_type")
            if edit_type:
                return str(edit_type)
        return str(proposal.get("family") or "unknown")

    def _dominant_payload_edit_type(self, rows: list[dict[str, Any]]) -> str:
        counts = Counter(self._payload_edit_type(row) for row in rows)
        if not counts:
            return "unknown"
        return counts.most_common(1)[0][0]

    @staticmethod
    def _near_miss_positive_anchor_reason(
        accepted_rows: list[dict[str, Any]],
        edit_type: str,
        anchor_cfg: dict[str, Any],
    ) -> str | None:
        if not isinstance(anchor_cfg, dict) or not bool(anchor_cfg.get("enabled", False)):
            return None
        edit_types = set(_coerce_str_list(anchor_cfg.get("edit_types", sorted(OBJECT_CONTRACT_EDIT_TYPES))))
        if edit_types and edit_type not in edit_types:
            return None
        min_positive_count = max(1, int(anchor_cfg.get("min_positive_count", 1)))
        if len(accepted_rows) < min_positive_count:
            return "near_miss_positive_anchor_missing"
        return None

    def _verified_rejected_near_miss_reason(
        self,
        payload: dict[str, Any],
        edit_type: str,
        cfg: dict[str, Any],
    ) -> str | None:
        """Return None only for rejected candidates that are safe positive near misses.

        This path is intentionally narrower than ordinary near-miss mining. It is
        for object edits where detector/string gates can be brittle, but the
        internal VLM judge gives a strong, reliable signal that the edit mostly
        happened and the image stayed clean.
        """

        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return "verified_rejected_near_miss_disabled"
        allowed_statuses = set(_coerce_str_list(cfg.get("statuses", ["rejected"])))
        status = str(payload.get("status") or "")
        if allowed_statuses and status not in allowed_statuses:
            return "verified_rejected_status_filtered"
        edit_types = set(_coerce_str_list(cfg.get("edit_types", ["object_removal", "object_replacement"])))
        if edit_types and edit_type not in edit_types:
            return "verified_rejected_edit_type_filtered"

        evaluation = payload.get("evaluator") or payload.get("solver") or {}
        signals = evaluation.get("signals", {}) if isinstance(evaluation, dict) else {}
        if not isinstance(signals, dict):
            signals = {}
        reject_reason = str(signals.get("rubric_reject_reason") or "").strip()
        allowed_reasons = set(
            _coerce_str_list(
                cfg.get(
                    "allowed_reject_reasons",
                    ["object_detector_contract", "rubric_forbidden_gate"],
                )
            )
        )
        if allowed_reasons and reject_reason not in allowed_reasons:
            return "verified_rejected_reason_filtered"

        if bool(cfg.get("require_internal_vlm_judge", True)):
            judge_supported = self._payload_component_score(
                payload,
                "internal_vlm_judge_supported",
                default=0.0,
            )
            if judge_supported < 0.5:
                return "verified_rejected_missing_judge"
        if bool(cfg.get("require_reliable_judge", True)):
            judge_reliable = self._payload_component_score(
                payload,
                "internal_vlm_judge_reliable",
                default=0.0,
            )
            if judge_reliable < 0.5:
                return "verified_rejected_unreliable_judge"

        min_confidence = _finite_float(cfg.get("min_judge_confidence"), math.nan)
        if math.isfinite(min_confidence):
            confidence = self._payload_component_score(
                payload,
                "internal_vlm_judge_confidence",
                default=0.0,
            )
            if confidence < min_confidence:
                return "verified_rejected_low_confidence"

        component_floors = self._merge_edit_type_overrides(
            dict(cfg.get("min_component_scores", {})),
            cfg.get("min_component_scores_by_edit_type"),
            edit_type,
        )
        for name, raw_floor in component_floors.items():
            floor = _finite_float(raw_floor, math.nan)
            if not math.isfinite(floor):
                continue
            score = self._payload_named_component_score(payload, str(name), default=0.0)
            if score < floor:
                return f"verified_rejected_floor_{name}"

        return None

    def _policy_samples_for_proposal(
        self,
        candidate_cfg: dict[str, Any],
        proposal: EditProposal,
        default_samples: int,
    ) -> int:
        samples = max(0, int(default_samples))
        overrides = (
            candidate_cfg.get("samples_per_proposal_by_edit_type")
            or candidate_cfg.get("samples_per_proposal_overrides")
            or {}
        )
        if isinstance(overrides, dict):
            edit_type = self._proposal_edit_type(proposal)
            if edit_type in overrides:
                samples = max(0, int(overrides[edit_type]))
        return samples

    def _expected_candidates_for_proposal(
        self,
        candidate_cfg: dict[str, Any],
        proposal: EditProposal,
        default_samples: int,
        self_play_samples: int,
        reference_samples: int,
    ) -> int:
        return (
            self._policy_samples_for_proposal(candidate_cfg, proposal, default_samples)
            + max(0, int(self_play_samples))
            + max(0, int(reference_samples))
        )

    def _editor_state_for_self_play_opponent(self, self_play_cfg: dict[str, Any]) -> dict[str, Any] | None:
        strategy = str(self_play_cfg.get("opponent", self_play_cfg.get("opponent_state", "previous_round")))
        if strategy in {"initial", "base", "base_model"}:
            return dict(self.initial_editor_state) if isinstance(self.initial_editor_state, dict) else None
        if strategy in {"previous_round", "previous", "last_round"}:
            if isinstance(self.previous_editor_state, dict):
                return dict(self.previous_editor_state)
            return dict(self.initial_editor_state) if isinstance(self.initial_editor_state, dict) else None
        if strategy in {"fixed_checkpoint", "checkpoint"}:
            checkpoint_path = self_play_cfg.get("checkpoint_path")
            if not checkpoint_path:
                return None
            current_state = self.editor.model_state() if isinstance(self.editor, QwenEditEditor) else {}
            return {
                "backend": self_play_cfg.get("backend", current_state.get("backend", "diffsynth")),
                "model_type": self_play_cfg.get("model_type", current_state.get("model_type", "lora")),
                "checkpoint_path": str(checkpoint_path),
                "base_model": current_state.get("base_model"),
            }
        return None

    def _editor_state_for_reference_candidate(self, reference_cfg: dict[str, Any]) -> dict[str, Any] | None:
        strategy = str(reference_cfg.get("source", reference_cfg.get("state", "initial")))
        if strategy in {"initial", "base", "base_model", "reference"}:
            return dict(self.initial_editor_state) if isinstance(self.initial_editor_state, dict) else None
        if strategy in {"previous_round", "previous", "last_round"}:
            if isinstance(self.previous_editor_state, dict):
                return dict(self.previous_editor_state)
            return dict(self.initial_editor_state) if isinstance(self.initial_editor_state, dict) else None
        if strategy in {"fixed_checkpoint", "checkpoint"}:
            checkpoint_path = reference_cfg.get("checkpoint_path")
            if not checkpoint_path:
                return None
            current_state = self.editor.model_state() if isinstance(self.editor, QwenEditEditor) else {}
            return {
                "backend": reference_cfg.get("backend", current_state.get("backend", "official_diffusers")),
                "model_type": reference_cfg.get("model_type", "lora"),
                "checkpoint_path": str(checkpoint_path),
                "base_model": current_state.get("base_model"),
            }
        return None

    def _set_editor_model_state(self, state: dict[str, Any] | None) -> None:
        if not isinstance(self.editor, QwenEditEditor) or not isinstance(state, dict):
            return
        self.editor.set_model_checkpoint(
            state.get("checkpoint_path"),
            model_type=str(state.get("model_type", self.editor.config.get("model", {}).get("model_type", "base"))),
            backend=str(state.get("backend", self.editor.config.get("model", {}).get("backend", "diffsynth"))),
        )

    @staticmethod
    def _record_stratify_label(record: UnlabeledImageRecord, key: str) -> str:
        metadata = record.metadata or {}
        value = _metadata_value(metadata, key)
        if value is None and key in {"primary_family", "family"}:
            value = metadata.get("primary_family") or metadata.get("family")
        if isinstance(value, list):
            value = value[0] if value else None
        label = str(value).strip() if value is not None else ""
        return label or "unknown"

    @staticmethod
    def _largest_remainder_quotas(labels: list[str], weights: dict[str, float], count: int) -> dict[str, int]:
        if count <= 0 or not labels:
            return {}
        positive_weights = {label: max(0.0, float(weights.get(label, 0.0))) for label in labels}
        total = sum(positive_weights.values())
        if total <= 0.0:
            base = count // len(labels)
            quotas = {label: base for label in labels}
            for label in labels[: count - base * len(labels)]:
                quotas[label] += 1
            return quotas
        raw = {label: count * positive_weights[label] / total for label in labels}
        quotas = {label: int(math.floor(raw[label])) for label in labels}
        remaining = count - sum(quotas.values())
        order = sorted(labels, key=lambda label: (raw[label] - quotas[label], positive_weights[label]), reverse=True)
        for label in order[:remaining]:
            quotas[label] += 1
        return quotas

    def _stratified_record_indices(self, round_index: int, count: int, curriculum: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
        key = str(curriculum.get("stratify_metadata_key", "primary_family"))
        target_raw = curriculum.get("stratify_targets") or curriculum.get("stratify_weights") or {}
        labels_by_index = [self._record_stratify_label(record, key) for record in self.records]
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels_by_index):
            buckets[label].append(index)
        configured_labels = [str(label) for label in target_raw.keys()] if isinstance(target_raw, dict) else []
        labels = [label for label in configured_labels if label in buckets]
        labels.extend(label for label in sorted(buckets) if label not in labels)
        weights = {label: 1.0 for label in labels}
        if isinstance(target_raw, dict):
            for label, value in target_raw.items():
                try:
                    weights[str(label)] = float(value)
                except (TypeError, ValueError):
                    continue
        quotas = self._largest_remainder_quotas(labels, weights, count)
        seed = int(curriculum.get("stratify_shuffle_seed", self.config.get("runtime", {}).get("seed", 0)))
        label_candidates: dict[str, list[int]] = {}
        cursor_salt = round_index * 1_000_003 + seed
        for label in labels:
            bucket = list(buckets.get(label, []))
            if not bucket or quotas.get(label, 0) <= 0:
                continue
            rng = random.Random(cursor_salt + sum(ord(char) for char in label))
            rng.shuffle(bucket)
            start = ((round_index - 1) * quotas[label]) % len(bucket)
            label_candidates[label] = [
                bucket[(start + offset) % len(bucket)]
                for offset in range(quotas[label])
            ]

        selected: list[int] = []
        selected_set: set[int] = set()
        per_label: Counter[str] = Counter()

        if bool(curriculum.get("stratify_interleave", False)):
            max_quota = max((len(indices) for indices in label_candidates.values()), default=0)
            for offset in range(max_quota):
                for label in labels:
                    candidates = label_candidates.get(label, [])
                    if offset >= len(candidates):
                        continue
                    index = candidates[offset]
                    if index in selected_set:
                        continue
                    selected.append(index)
                    selected_set.add(index)
                    per_label[label] += 1
                    if len(selected) >= count:
                        break
                if len(selected) >= count:
                    break
        else:
            for label in labels:
                candidates = label_candidates.get(label, [])
                if not candidates:
                    continue
                bucket_size = len(buckets.get(label, []))
                quota = int(quotas.get(label, 0))
                if quota <= 0:
                    continue
                added_for_label = 0
                for index in candidates:
                    if index in selected_set and bucket_size >= quota:
                        continue
                    selected.append(index)
                    selected_set.add(index)
                    per_label[label] += 1
                    added_for_label += 1
                    if len(selected) >= count or added_for_label >= quota:
                        break
                if len(selected) >= count:
                    break
        if len(selected) < count:
            selected_set = set(selected)
            fallback = [index for index in range(len(self.records)) if index not in selected_set]
            rng = random.Random(cursor_salt + 17)
            rng.shuffle(fallback)
            for index in fallback[: count - len(selected)]:
                selected.append(index)
                per_label[labels_by_index[index]] += 1
        return selected[:count], {
            "stratify_metadata_key": key,
            "stratify_targets": target_raw,
            "stratified_label_counts": dict(sorted(per_label.items())),
            "stratified_available_counts": {label: len(buckets[label]) for label in sorted(buckets)},
        }

    def _round_records_for_round(
        self,
        round_index: int,
        max_records_per_round: int,
    ) -> tuple[list[UnlabeledImageRecord], dict[str, Any]]:
        if not self.records:
            return [], {
                "record_schedule": "empty",
                "record_start_index": 0,
                "record_count": 0,
                "record_indices": [],
                "record_wraparound": False,
            }
        count = min(max(1, max_records_per_round), len(self.records))
        curriculum = self.config.get("curriculum", {})
        schedule = str(curriculum.get("record_schedule", "sequential_shards"))
        schedule_metadata: dict[str, Any] = {}
        if schedule in {"fixed_first_slice", "first_slice", "fixed"}:
            start_offset = int(curriculum.get("record_start_offset", 0))
            start = start_offset % len(self.records)
            indices = [(start + offset) % len(self.records) for offset in range(count)]
        elif schedule in {"sequential_shards", "sequential", "sharded"}:
            start_offset = int(curriculum.get("record_start_offset", 0))
            start = (start_offset + (round_index - 1) * count) % len(self.records)
            indices = [(start + offset) % len(self.records) for offset in range(count)]
        elif schedule in {"stratified_metadata", "stratified", "balanced_metadata"}:
            indices, schedule_metadata = self._stratified_record_indices(round_index, count, curriculum)
        else:
            raise ValueError(
                "Unsupported curriculum.record_schedule. Use 'sequential_shards', "
                "'fixed_first_slice', or 'stratified_metadata'."
            )
        round_records: list[UnlabeledImageRecord] = []
        scheduled_edit_type_counts: Counter[str] = Counter()
        for local_index, record_index in enumerate(indices):
            record, scheduled_edit_type = self._record_with_curriculum_edit_type(
                self.records[record_index],
                round_index=round_index,
                local_index=local_index,
                record_index=record_index,
            )
            round_records.append(record)
            if scheduled_edit_type:
                scheduled_edit_type_counts[scheduled_edit_type] += 1
        metadata = {
            "record_schedule": schedule,
            "record_start_index": indices[0] if indices else 0,
            "record_count": len(indices),
            "record_indices": indices,
            "record_wraparound": bool(indices and indices[-1] < indices[0]),
            "scheduled_edit_type_counts": dict(sorted(scheduled_edit_type_counts.items())),
        }
        metadata.update(schedule_metadata)
        return round_records, metadata

    @staticmethod
    def _metadata_float(metadata: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(metadata.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _metadata_edit_families(metadata: dict[str, Any]) -> list[str]:
        families = []
        primary = str(metadata.get("primary_family") or "").strip()
        if primary:
            families.append(primary)
        families.extend(_coerce_str_list(metadata.get("edit_families")))
        output = []
        seen = set()
        for family in families:
            normalized = str(family).strip().lower().replace(" ", "_").replace("-", "_")
            if normalized and normalized not in seen:
                seen.add(normalized)
                output.append(normalized)
        return output

    @staticmethod
    def _metadata_object_hint_is_text_like(value: Any) -> bool:
        text = str(value or "").strip().lower()
        if not text:
            return False
        text_like_terms = {
            "text",
            "word",
            "words",
            "letter",
            "letters",
            "writing",
            "caption",
            "label",
            "tag",
            "sticker",
            "barcode",
            "logo",
            "poster",
            "sign",
            "book",
            "comic",
            "price",
            "menu",
            "license plate",
        }
        return any(term in text for term in text_like_terms) or any(quote in text for quote in ("'", "\""))

    def _record_feasible_schedule(
        self,
        record: UnlabeledImageRecord,
        schedule: list[str],
        curriculum: dict[str, Any],
    ) -> tuple[list[str], dict[str, Any]]:
        if not bool(curriculum.get("respect_record_edit_families", False)):
            return list(schedule), {"coverage_schedule_respects_record": False}

        metadata = dict(record.metadata or {})
        removable_hint = str(metadata.get("removable_object_description") or "").strip()
        removable_hint_text_like = self._metadata_object_hint_is_text_like(removable_hint)
        explicit_targets = _coerce_str_list(metadata.get("target_edit_types"))
        if explicit_targets and not metadata.get("coverage_schedule_injected"):
            feasible = [edit_type for edit_type in schedule if edit_type in set(explicit_targets)]
            if removable_hint_text_like:
                feasible = [
                    edit_type
                    for edit_type in feasible
                    if edit_type not in {"object_removal", "object_replacement"}
                ]
            return (
                feasible or list(schedule),
                {
                    "coverage_schedule_respects_record": True,
                    "coverage_feasible_source": "metadata_target_edit_types",
                    "coverage_feasible_edit_types": feasible or list(schedule),
                    "coverage_removable_hint_text_like": removable_hint_text_like,
                },
            )

        families = self._metadata_edit_families(metadata)
        feasible_set: set[str] = set()
        for family in families:
            feasible_set.update(EDIT_FAMILY_TO_SCHEDULE_TYPES.get(family, ()))

        removable_score = self._metadata_float(metadata, "removable_object_score")
        small_object_separability = self._metadata_float(metadata, "small_object_separability")
        min_removable_score = float(curriculum.get("min_removable_object_score_for_object_schedule", 0.55))
        min_separability_score = float(curriculum.get("min_small_object_separability_for_object_schedule", 0.45))
        object_contract_feasible = bool(
            not removable_hint_text_like
            and (
                removable_hint
                or removable_score >= min_removable_score
                or small_object_separability >= min_separability_score
            )
        )
        if object_contract_feasible:
            feasible_set.update({"object_removal", "object_replacement"})
        else:
            feasible_set.discard("object_removal")
            feasible_set.discard("object_replacement")

        if not feasible_set:
            return (
                list(schedule),
                {
                    "coverage_schedule_respects_record": True,
                    "coverage_feasible_source": "unrestricted_no_metadata",
                    "coverage_feasible_edit_types": list(schedule),
                },
            )

        feasible = [edit_type for edit_type in schedule if edit_type in feasible_set]
        if not feasible:
            feasible = list(schedule)
        return (
            feasible,
            {
                "coverage_schedule_respects_record": True,
                "coverage_feasible_source": "metadata_edit_families",
                "coverage_metadata_families": families,
                "coverage_object_contract_feasible": object_contract_feasible,
                "coverage_removable_object_score": removable_score,
                "coverage_small_object_separability": small_object_separability,
                "coverage_has_removable_object_description": bool(removable_hint),
                "coverage_removable_hint_text_like": removable_hint_text_like,
                "coverage_feasible_edit_types": feasible,
            },
        )

    def _record_with_curriculum_edit_type(
        self,
        record: UnlabeledImageRecord,
        *,
        round_index: int,
        local_index: int,
        record_index: int,
    ) -> tuple[UnlabeledImageRecord, str | None]:
        curriculum = self.config.get("curriculum", {})
        schedule = _coerce_str_list(
            curriculum.get("coverage_edit_type_schedule")
            or curriculum.get("scheduled_edit_types")
            or curriculum.get("target_edit_type_schedule")
        )
        if not schedule or not bool(curriculum.get("inject_scheduled_edit_type", True)):
            return record, None

        metadata = dict(record.metadata or {})
        existing = metadata.get("scheduled_edit_type") or metadata.get("target_edit_type")
        if existing and not bool(curriculum.get("override_scheduled_edit_type", False)):
            return record, str(existing)

        mode = str(curriculum.get("coverage_schedule_mode", "global_cycle"))
        offset = int(curriculum.get("coverage_schedule_offset", 0))
        if mode in {"round_cycle", "round"}:
            schedule_index = (local_index + round_index - 1 + offset) % len(schedule)
        elif mode in {"record_index", "dataset_index"}:
            schedule_index = (record_index + offset) % len(schedule)
        else:
            schedule_index = ((round_index - 1) * max(1, int(curriculum.get("max_records_per_round", 1))) + local_index + offset) % len(schedule)
        requested_edit_type = schedule[schedule_index]
        feasible_schedule, feasible_metadata = self._record_feasible_schedule(record, schedule, curriculum)
        feasible_set = set(feasible_schedule)
        if requested_edit_type in feasible_set:
            scheduled_edit_type = requested_edit_type
        else:
            scheduled_edit_type = next(
                (
                    schedule[(schedule_index + offset_index) % len(schedule)]
                    for offset_index in range(1, len(schedule) + 1)
                    if schedule[(schedule_index + offset_index) % len(schedule)] in feasible_set
                ),
                feasible_schedule[0],
            )
        metadata["scheduled_edit_type"] = scheduled_edit_type
        metadata["target_edit_types"] = feasible_schedule
        metadata["coverage_schedule_injected"] = True
        metadata["coverage_schedule_mode"] = mode
        metadata["coverage_requested_edit_type"] = requested_edit_type
        metadata.update(feasible_metadata)
        return (
            UnlabeledImageRecord(
                key=record.key,
                image_path=record.image_path,
                caption=record.caption,
                metadata=metadata,
            ),
            scheduled_edit_type,
        )

    @staticmethod
    def _is_completed_group(rows: list[dict[str, Any]], samples_per_proposal: int) -> bool:
        if len(rows) < samples_per_proposal:
            return False
        statuses = {str(row.get("status", "")) for row in rows[:samples_per_proposal]}
        return statuses.issubset({"accepted", "rejected"})

    @staticmethod
    def _evaluation_from_payload(payload: dict[str, Any]) -> EvaluationResult | None:
        data = payload.get("evaluator") or payload.get("solver")
        if not isinstance(data, dict):
            return None
        return EvaluationResult(
            global_score=float(data.get("global_score", 0.0)),
            local_score=float(data.get("local_score", 0.0)),
            total_score=float(data.get("total_score", 0.0)),
            accepted=bool(data.get("accepted", False)),
            signals=dict(data.get("signals", {})),
            component_scores=dict(data.get("component_scores", {})),
        )

    @staticmethod
    def _proposal_from_payload(payload: dict[str, Any]) -> EditProposal:
        proposal = payload.get("proposal", {})
        expected_range = proposal.get("expected_changed_fraction") or (0.0, 1.0)
        if len(expected_range) != 2:
            expected_range = (0.0, 1.0)
        definition = ProposalDefinition(
            operation_id=str(proposal.get("operation_id", "unknown")),
            instruction=str(proposal.get("instruction", "")),
            family=str(proposal.get("family", "unknown")),
            difficulty=int(proposal.get("difficulty", 1)),
            scope=str(proposal.get("scope", "global")),
            metric=str(proposal.get("metric", "internal_prompt_gain")),
            direction=str(proposal.get("direction", "increase")),
            target=float(proposal.get("target", 0.0)),
            expected_changed_fraction=(float(expected_range[0]), float(expected_range[1])),
            verifier=str(proposal.get("verifier", "internal")),
            inverse_operation_id=proposal.get("inverse_operation_id"),
        )
        return EditProposal(
            record_key=str(payload.get("record_key", proposal.get("record_key", ""))),
            round_index=int(proposal.get("round_index", 0)),
            proposal_index=int(proposal.get("proposal_index", 0)),
            definition=definition,
            difficulty_level=int(proposal.get("difficulty_level", definition.difficulty)),
            instruction=definition.instruction,
            structured_edit=dict(proposal.get("structured_edit", {})),
        )

    def _accepted_sample_from_payload(self, payload: dict[str, Any]) -> AcceptedSample | None:
        if payload.get("status") != "accepted":
            return None
        evaluation = self._evaluation_from_payload(payload)
        edited_path_raw = payload.get("edited_image_path")
        if evaluation is None or not edited_path_raw:
            return None
        edited_path = resolve_path(str(edited_path_raw))
        if edited_path is None or not edited_path.exists():
            return None
        record_key = str(payload.get("record_key", ""))
        record = self.record_by_key.get(record_key)
        if record is None:
            image_path = resolve_path(str(payload.get("image_path", "")))
            if image_path is None:
                return None
            record = UnlabeledImageRecord(
                key=record_key,
                image_path=image_path,
                caption=payload.get("caption"),
                metadata={},
            )
        return AcceptedSample(
            record=record,
            proposal=self._proposal_from_payload(payload),
            edited_image_path=edited_path,
            evaluation_result=evaluation,
            candidate_index=int(payload.get("candidate_index", 0)),
        )

    def _accepted_samples_from_payloads(self, rows: list[dict[str, Any]]) -> list[AcceptedSample]:
        samples = []
        seen = set()
        for row in rows:
            key = self._candidate_key(row)
            if key in seen:
                continue
            seen.add(key)
            sample = self._accepted_sample_from_payload(row)
            if sample is not None:
                samples.append(sample)
        return samples

    @staticmethod
    def _proposal_plan_payload(
        record_index: int,
        record: UnlabeledImageRecord,
        proposal: EditProposal,
        group_id: str,
    ) -> dict[str, Any]:
        return {
            "group_id": group_id,
            "record_index": record_index,
            "record_key": record.key,
            "image_path": relative_to_repo(record.image_path),
            "caption": record.caption,
            "proposal": {
                "round_index": proposal.round_index,
                "proposal_index": proposal.proposal_index,
                "operation_id": proposal.definition.operation_id,
                "family": proposal.definition.family,
                "difficulty": proposal.definition.difficulty,
                "difficulty_level": proposal.difficulty_level,
                "instruction": proposal.instruction,
                "scope": proposal.definition.scope,
                "metric": proposal.definition.metric,
                "direction": proposal.definition.direction,
                "target": proposal.definition.target,
                "expected_changed_fraction": list(proposal.definition.expected_changed_fraction),
                "verifier": proposal.definition.verifier,
                "inverse_operation_id": proposal.definition.inverse_operation_id,
                "structured_edit": proposal.structured_edit,
            },
        }

    def _build_evaluator_exports(
        self,
        candidate_payloads: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        evaluator_records: list[dict[str, Any]] = []
        preference_records: list[dict[str, Any]] = []
        grouped = self._group_rows(candidate_payloads)
        for group_id, rows in grouped.items():
            accepted_payloads = [row for row in rows if row.get("status") == "accepted"]
            rejected_payloads = [row for row in rows if row.get("status") == "rejected"]
            for row in rows:
                evaluation = row.get("evaluator") or row.get("solver")
                if not isinstance(evaluation, dict):
                    continue
                proposal = row.get("proposal", {})
                signals = evaluation.get("signals", {})
                evaluator_records.append(
                    {
                        "type": "candidate",
                        "group_id": group_id,
                        "candidate_index": row.get("candidate_index"),
                        "record_key": row.get("record_key"),
                        "source_image": row.get("image_path"),
                        "edited_image": row.get("edited_image_path"),
                        "instruction": proposal.get("instruction"),
                        "operation_id": proposal.get("operation_id"),
                        "family": proposal.get("family"),
                        "verifier": proposal.get("verifier"),
                        "structured_edit": proposal.get("structured_edit", {}),
                        "distractors": row.get("distractors", []),
                        "accepted": evaluation.get("accepted", row.get("status") == "accepted"),
                        "feasible": bool(signals.get("feasible", float(row.get("status") == "accepted"))),
                        "rank": int(signals.get("feasible_rank", 0.0)),
                        "scores": {
                            "total_score": evaluation.get("total_score"),
                            "global_score": evaluation.get("global_score"),
                            "local_score": evaluation.get("local_score"),
                            "component_scores": evaluation.get("component_scores", {}),
                        },
                    }
                )
            for winner in accepted_payloads:
                for loser in rejected_payloads:
                    winner_evaluation = winner.get("evaluator", winner.get("solver", {}))
                    loser_evaluation = loser.get("evaluator", loser.get("solver", {}))
                    proposal = winner.get("proposal", {})
                    preference_records.append(
                        {
                            "type": "preference",
                            "group_id": group_id,
                            "winner_candidate_index": winner.get("candidate_index"),
                            "loser_candidate_index": loser.get("candidate_index"),
                            "record_key": winner.get("record_key"),
                            "source_image": winner.get("image_path"),
                            "instruction": proposal.get("instruction"),
                            "operation_id": proposal.get("operation_id"),
                            "family": proposal.get("family"),
                            "verifier": proposal.get("verifier"),
                            "structured_edit": proposal.get("structured_edit", {}),
                            "distractors": winner.get("distractors", []),
                            "winner_image": winner.get("edited_image_path"),
                            "loser_image": loser.get("edited_image_path"),
                            "winner_score": winner_evaluation.get("total_score"),
                            "loser_score": loser_evaluation.get("total_score"),
                        }
                    )
        return evaluator_records, preference_records

    def _write_progress(
        self,
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        save_json({**base_run_metadata(), **payload}, path)

    def _restore_completed_round_state(self, round_summary: dict[str, Any]) -> None:
        if "next_difficulty_level" in round_summary:
            self.difficulty_controller.level = int(round_summary["next_difficulty_level"])

        editor_state_before_round = round_summary.get("editor_state_before_round")
        if isinstance(editor_state_before_round, dict):
            self.previous_editor_state = dict(editor_state_before_round)

        training_result = round_summary.get("training")
        if isinstance(training_result, dict):
            latest_checkpoint = training_result.get("latest_checkpoint")
            continue_with_checkpoint = bool(training_result.get("continue_with_trained_checkpoint", True))
            if latest_checkpoint and continue_with_checkpoint:
                self.config.setdefault("training", {})["current_checkpoint_path"] = latest_checkpoint
                if isinstance(self.editor, QwenEditEditor):
                    self.editor.set_model_checkpoint(
                        str(latest_checkpoint),
                        model_type=str(training_result.get("trained_model_type", "lora")),
                        backend=str(
                            training_result.get(
                                "trained_checkpoint_backend",
                                self.config.get("training", {}).get("trained_checkpoint_backend", "diffsynth"),
                            )
                        ),
                    )

        proposer_training_result = round_summary.get("proposer_training")
        if (
            isinstance(proposer_training_result, dict)
            and proposer_training_result.get("status") == "completed"
            and proposer_training_result.get("output_dir")
        ):
            checkpoint_path = str(proposer_training_result["output_dir"])
            self.config.setdefault("proposer", {})["checkpoint_path"] = checkpoint_path
            if hasattr(self.proposer, "set_checkpoint_path"):
                self.proposer.set_checkpoint_path(checkpoint_path)

    def _training_contract_filter_reason(
        self,
        payload: dict[str, Any],
        weighted_cfg: dict[str, Any],
        edit_type: str | None,
    ) -> str | None:
        contract_cfg = weighted_cfg.get("contract_filter", {})
        if not isinstance(contract_cfg, dict) or not bool(contract_cfg.get("enabled", False)):
            return None

        raw_edit_types = contract_cfg.get("edit_types", contract_cfg.get("strict_edit_types", []))
        if isinstance(raw_edit_types, str):
            edit_type_set = {item.strip() for item in raw_edit_types.split(",") if item.strip()}
        else:
            edit_type_set = {str(item) for item in (raw_edit_types or [])}
        if edit_type_set and edit_type not in edit_type_set:
            return None

        structured_reason = self._structured_edit_contract_filter_reason(payload, contract_cfg, edit_type)
        if structured_reason is not None:
            return structured_reason

        evaluation = payload.get("evaluator") or payload.get("solver") or {}
        component_scores = evaluation.get("component_scores", {}) if isinstance(evaluation, dict) else {}
        signals = evaluation.get("signals", {}) if isinstance(evaluation, dict) else {}

        require_strict_forbidden = _coerce_bool(contract_cfg.get("require_strict_forbidden_gate", False), False)
        require_strict_by_edit_type = contract_cfg.get("require_strict_forbidden_gate_by_edit_type", {})
        if isinstance(require_strict_by_edit_type, dict) and edit_type in require_strict_by_edit_type:
            require_strict_forbidden = _coerce_bool(
                require_strict_by_edit_type[edit_type],
                require_strict_forbidden,
            )
        if require_strict_forbidden:
            strict_pass = _finite_float(signals.get("rubric_forbidden_gate_strict_pass"), 0.0)
            if strict_pass < 0.5:
                return "contract_forbidden_gate_not_strict"

        component_mins = dict(contract_cfg.get("min_component_scores", {}))
        component_mins_by_edit_type = contract_cfg.get("min_component_scores_by_edit_type", {})
        if isinstance(component_mins_by_edit_type, dict):
            edit_type_component_mins = component_mins_by_edit_type.get(edit_type, {})
            if isinstance(edit_type_component_mins, dict):
                component_mins.update(edit_type_component_mins)
        disabled_components_by_edit_type = contract_cfg.get("disabled_component_scores_by_edit_type", {})
        disabled_components = set()
        if isinstance(disabled_components_by_edit_type, dict):
            raw_disabled = disabled_components_by_edit_type.get(edit_type, [])
            if isinstance(raw_disabled, str):
                disabled_components = {item.strip() for item in raw_disabled.split(",") if item.strip()}
            else:
                disabled_components = {str(item) for item in (raw_disabled or [])}
        for component_name in disabled_components:
            component_mins.pop(component_name, None)
        for component_name in (
            "rubric_source_grounded",
            "rubric_required_after",
            "rubric_forbidden_after_absent",
            "rubric_edit_success",
            "rubric_preservation",
            "rubric_validity",
            "rubric_cepr_raw_reward",
            "cepr_raw_reward",
            "cepr_preservation",
            "cepr_validity",
            "cepr_taxonomy",
        ):
            if component_name in disabled_components:
                continue
            shorthand_key = f"min_{component_name}"
            if shorthand_key in contract_cfg and component_name not in component_mins:
                component_mins[component_name] = contract_cfg[shorthand_key]

        for component_name, min_value_raw in component_mins.items():
            min_value = _finite_float(min_value_raw, math.nan)
            if not math.isfinite(min_value):
                continue
            score = _finite_float(component_scores.get(component_name), math.nan)
            if not math.isfinite(score):
                return f"contract_{component_name}_missing"
            if score < min_value:
                return f"contract_{component_name}_below_threshold"
        return None

    @staticmethod
    def _clean_contract_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @staticmethod
    def _token_count(text: str) -> int:
        tokens = re.findall(r"[\w-]+", text, flags=re.UNICODE)
        return len(tokens)

    @staticmethod
    def _object_slots_overlap(first: Any, second: Any) -> bool:
        first_text = re.sub(r"\s+", " ", str(first or "").strip().lower())
        second_text = re.sub(r"\s+", " ", str(second or "").strip().lower())
        if not first_text or not second_text:
            return False
        first_terms = {term for term in re.findall(r"[a-z0-9]+", first_text) if len(term) > 2}
        second_terms = {term for term in re.findall(r"[a-z0-9]+", second_text) if len(term) > 2}
        if not first_terms or not second_terms:
            return False
        return bool(first_terms & second_terms) or first_text in second_text or second_text in first_text

    def _has_spatial_anchor(self, text: str) -> bool:
        lowered = f" {text.lower()} "
        anchor_markers = (
            " on ",
            " above ",
            " below ",
            " under ",
            " beneath ",
            " beside ",
            " near ",
            " next to ",
            " left of ",
            " right of ",
            " in front of ",
            " behind ",
            " between ",
            " attached to ",
            " held by ",
            " worn by ",
            " around ",
            " at ",
            " in ",
        )
        coordinate_markers = (
            "upper ",
            "lower ",
            "top ",
            "bottom ",
            "left ",
            "right ",
            "center",
            "centre",
            "middle",
            "corner",
            "edge",
            "foreground",
            "background",
        )
        return any(marker in lowered for marker in anchor_markers) or any(
            marker in lowered for marker in coordinate_markers
        )

    def _object_slot_sanity_reason(
        self,
        value: Any,
        *,
        field_name: str,
        contract_cfg: dict[str, Any],
    ) -> str | None:
        text = self._clean_contract_text(value)
        if not text:
            return f"contract_{field_name}_missing"
        lowered = text.lower()
        generic_terms = set(GENERIC_OBJECT_SLOT_TERMS)
        generic_terms.update(item.lower() for item in _coerce_str_list(contract_cfg.get("generic_object_terms", [])))
        if lowered in generic_terms:
            return f"contract_{field_name}_generic"
        if (
            any(quote in text for quote in ('"', "\u201c", "\u201d", "\u2018", "\u2019"))
            or re.search(r"(^|\s)'|'($|\s)", text)
        ):
            return f"contract_{field_name}_contains_quoted_text"
        action_phrases = list(OBJECT_SLOT_ACTION_PHRASES)
        action_phrases.extend(
            item.lower() for item in _coerce_str_list(contract_cfg.get("object_slot_action_phrases", []))
        )
        if any(phrase and phrase in lowered for phrase in action_phrases):
            return f"contract_{field_name}_action_phrase"
        max_words = int(contract_cfg.get("max_object_slot_words", 8))
        if max_words > 0 and self._token_count(text) > max_words:
            return f"contract_{field_name}_too_long"
        return None

    def _structured_edit_contract_filter_reason(
        self,
        payload: dict[str, Any],
        contract_cfg: dict[str, Any],
        edit_type: str | None,
    ) -> str | None:
        if not bool(
            contract_cfg.get(
                "require_valid_structured_edit",
                contract_cfg.get("validate_structured_edit", False),
            )
        ):
            return None
        proposal = payload.get("proposal", {})
        if not isinstance(proposal, dict):
            return "contract_proposal_missing"
        structured_edit = proposal.get("structured_edit", {})
        if not isinstance(structured_edit, dict):
            return "contract_structured_edit_missing"

        edit_type = str(edit_type or structured_edit.get("edit_type") or proposal.get("family") or "unknown")
        if edit_type not in OBJECT_CONTRACT_EDIT_TYPES:
            return None

        source_object = structured_edit.get("source_object") or structured_edit.get("target")
        source_reason = self._object_slot_sanity_reason(
            source_object,
            field_name="source_object",
            contract_cfg=contract_cfg,
        )
        if source_reason is not None:
            return source_reason

        if edit_type == "object_replacement":
            target_object = structured_edit.get("target_object") or structured_edit.get("replacement")
            target_reason = self._object_slot_sanity_reason(
                target_object,
                field_name="target_object",
                contract_cfg=contract_cfg,
            )
            if target_reason is not None:
                return target_reason
            if self._clean_contract_text(source_object).lower() == self._clean_contract_text(target_object).lower():
                return "contract_replacement_same_as_source"
            if self._object_slots_overlap(target_object, source_object):
                return "contract_replacement_overlaps_source"

        target_region = self._clean_contract_text(structured_edit.get("target_region"))
        if target_region:
            lowered_region = target_region.lower()
            if "fill the area" in lowered_region or "area naturally" in lowered_region:
                return "contract_target_region_action_phrase"
            require_anchor = bool(contract_cfg.get("require_spatial_target_region", False))
            require_anchor_by_type = contract_cfg.get("require_spatial_target_region_by_edit_type", {})
            if isinstance(require_anchor_by_type, dict) and edit_type in require_anchor_by_type:
                require_anchor = _coerce_bool(require_anchor_by_type[edit_type], require_anchor)
            if require_anchor and not self._has_spatial_anchor(target_region):
                return "contract_target_region_missing_anchor"
        elif bool(contract_cfg.get("require_target_region", False)):
            return "contract_target_region_missing"

        return None

    def _candidate_training_weight(self, payload: dict[str, Any]) -> tuple[float, str]:
        allowed_verifiers = self.config.get("output", {}).get("train_verifiers")
        allowed_verifier_set = set(allowed_verifiers) if allowed_verifiers else None
        proposal = payload.get("proposal", {})
        if allowed_verifier_set is not None and proposal.get("verifier") not in allowed_verifier_set:
            return 0.0, "verifier_filtered"
        allowed_families = self.config.get("output", {}).get("train_families")
        allowed_family_set = set(allowed_families) if allowed_families else None
        if allowed_family_set is not None and proposal.get("family") not in allowed_family_set:
            return 0.0, "family_filtered"
        allowed_edit_types = self.config.get("output", {}).get("train_edit_types")
        allowed_edit_type_set = set(allowed_edit_types) if allowed_edit_types else None
        edit_type = self._payload_edit_type(payload)
        if allowed_edit_type_set is not None and edit_type not in allowed_edit_type_set:
            return 0.0, "edit_type_filtered"
        if not payload.get("edited_image_path"):
            return 0.0, "missing_edited_image"

        training_cfg = self.config.get("training", {})
        weighted_cfg = dict(training_cfg.get("weighted_sft", {}))
        weighted_enabled = bool(weighted_cfg.get("enabled", False))
        allowed_candidate_roles = _coerce_str_list(
            weighted_cfg.get("allowed_candidate_roles", weighted_cfg.get("candidate_roles"))
        )
        if allowed_candidate_roles and not self._role_matches(
            self._candidate_role(payload),
            allowed_candidate_roles,
        ):
            return 0.0, "candidate_role_filtered"
        status = str(payload.get("status", ""))
        contract_reason = self._training_contract_filter_reason(payload, weighted_cfg, edit_type)
        if status == "accepted":
            if contract_reason is not None:
                return 0.0, contract_reason
            return float(weighted_cfg.get("accepted_weight", 1.0)), "accepted"

        evaluation = payload.get("evaluator") or payload.get("solver") or {}
        component_scores = evaluation.get("component_scores", {}) if isinstance(evaluation, dict) else {}
        signals = evaluation.get("signals", {}) if isinstance(evaluation, dict) else {}
        if status == "rejected" and bool(weighted_cfg.get("include_feasible_ranked_positives", False)):
            feasible_rank = int(_finite_float(signals.get("feasible_rank"), 0.0))
            max_feasible_rank = max(1, int(weighted_cfg.get("max_feasible_rank", 2)))
            if bool(_finite_float(signals.get("feasible"), 0.0) >= 0.5) and 0 < feasible_rank <= max_feasible_rank:
                if contract_reason is not None:
                    return 0.0, contract_reason
                accepted_weight = float(weighted_cfg.get("accepted_weight", 1.0))
                feasible_weight = float(weighted_cfg.get("feasible_positive_weight", 0.75))
                return accepted_weight * feasible_weight, "feasible_ranked_positive"

        if not weighted_enabled:
            return 0.0, "not_accepted"
        if not bool(weighted_cfg.get("include_rejected", True)):
            return 0.0, "rejected_disabled"
        if status != "rejected":
            return 0.0, f"status_{status}"

        evaluator_cfg = self.config.get("evaluator", self.config.get("solver", {})) or {}

        raw_reward = _finite_float(
            component_scores.get("cepr_raw_reward"),
            _finite_float(evaluation.get("total_score"), _finite_float(payload.get("total_score"))),
        )
        semantic_edit = max(
            _finite_float(component_scores.get("cepr_semantic_edit")),
            _finite_float(component_scores.get("cepr_edit_specificity")),
            _finite_float(component_scores.get("cepr_edit")),
        )
        preservation = _finite_float(component_scores.get("cepr_preservation"), _finite_float(signals.get("preservation")))
        validity = _finite_float(component_scores.get("cepr_validity"), _finite_float(signals.get("validity")))
        taxonomy = _finite_float(component_scores.get("cepr_taxonomy"), 1.0)

        reward_threshold = float(weighted_cfg.get("target_reward", evaluator_cfg.get("reward_threshold", 0.30)))
        min_raw_reward = float(weighted_cfg.get("min_raw_reward", reward_threshold * 0.5))
        min_semantic_edit = float(
            weighted_cfg.get("min_semantic_edit", float(evaluator_cfg.get("edit_threshold", 0.45)) * 0.5)
        )
        min_preservation = float(
            weighted_cfg.get("min_preservation", evaluator_cfg.get("preservation_threshold", 0.20))
        )
        min_validity = float(weighted_cfg.get("min_validity", evaluator_cfg.get("validity_threshold", 0.50)))
        min_taxonomy = float(weighted_cfg.get("min_taxonomy", 0.0))

        if validity < min_validity:
            return 0.0, "validity_below_threshold"
        if preservation < min_preservation:
            return 0.0, "preservation_below_threshold"
        if semantic_edit < min_semantic_edit:
            return 0.0, "semantic_edit_below_threshold"
        if taxonomy < min_taxonomy:
            return 0.0, "taxonomy_below_threshold"
        if raw_reward < min_raw_reward:
            return 0.0, "reward_below_threshold"

        min_weight = float(weighted_cfg.get("min_rejected_weight", 0.05))
        max_weight = float(weighted_cfg.get("max_rejected_weight", 0.50))
        score_power = float(weighted_cfg.get("score_power", 1.0))
        normalized = _clamp((raw_reward - min_raw_reward) / max(reward_threshold - min_raw_reward, 1e-6))
        normalized = normalized**max(score_power, 1e-6)
        return min_weight + (max_weight - min_weight) * normalized, "cepr_weighted_rejected"

    @staticmethod
    def _record_balance_key(record: dict[str, Any], key_axis: str) -> str:
        key_axis = str(key_axis or "edit_type")
        structured_edit = record.get("structured_edit", {})
        if not isinstance(structured_edit, dict):
            structured_edit = {}
        if key_axis in {"edit_type", "type"}:
            value = structured_edit.get("edit_type")
        elif key_axis == "family":
            value = record.get("family")
        elif key_axis == "operation_id":
            value = record.get("operation_id")
        else:
            value = structured_edit.get(key_axis, record.get(key_axis))
        return str(value or record.get("family") or "unknown")

    @staticmethod
    def _record_identity(record: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(record.get("group_id") or ""),
            str(record.get("candidate_index") if record.get("candidate_index") is not None else ""),
            str(record.get("candidate_status") or record.get("chosen_candidate_index") or ""),
        )

    @staticmethod
    def _record_balance_score(record: dict[str, Any]) -> tuple[float, ...]:
        scores = record.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}
        component_scores = scores.get("component_scores", {})
        if not isinstance(component_scores, dict):
            component_scores = {}
        return (
            _finite_float(record.get("sample_weight"), 0.0),
            _finite_float(record.get("score_margin"), 0.0),
            _finite_float(record.get("chosen_raw_reward"), 0.0),
            _finite_float(component_scores.get("rubric_cepr_raw_reward"), 0.0),
            _finite_float(component_scores.get("cepr_raw_reward"), 0.0),
            _finite_float(scores.get("total_score"), 0.0),
        )

    def _apply_record_family_balance(
        self,
        records: list[dict[str, Any]],
        balance_cfg: dict[str, Any],
        *,
        item_name: str,
        max_count_keys: tuple[str, ...],
        min_count_keys: tuple[str, ...],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not isinstance(balance_cfg, dict) or not bool(balance_cfg.get("enabled", False)):
            return records, {"enabled": False}
        if not records:
            return records, {"enabled": True, "before_counts": {}, "after_counts": {}, "dropped": 0}

        key_axis = str(balance_cfg.get("key", balance_cfg.get("axis", "edit_type")))
        target_families = _coerce_str_list(balance_cfg.get("target_families", []))
        target_only = bool(balance_cfg.get("target_only", False))
        max_count_maps: list[dict[str, int]] = []
        default_max_counts: list[int] = []
        for key in max_count_keys:
            if key not in balance_cfg:
                continue
            value = balance_cfg.get(key)
            if isinstance(value, dict):
                max_count_maps.append(_coerce_int_map(value))
            else:
                try:
                    parsed_value = int(value)
                except (TypeError, ValueError):
                    parsed_value = 0
                if parsed_value > 0:
                    default_max_counts.append(parsed_value)
        max_fraction = float(balance_cfg.get("max_fraction_per_family", 0.0))
        if max_fraction > 0:
            default_max_counts.append(max(1, math.ceil(len(records) * max_fraction)))

        min_count_map: dict[str, int] = {}
        default_min_count = 0
        for key in min_count_keys:
            if key not in balance_cfg:
                continue
            value = balance_cfg.get(key)
            if isinstance(value, dict):
                min_count_map.update(_coerce_int_map(value))
            else:
                try:
                    default_min_count = max(default_min_count, int(value))
                except (TypeError, ValueError):
                    continue

        indexed_records = list(enumerate(records))
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, record in indexed_records:
            grouped[self._record_balance_key(record, key_axis)].append((index, record))
        before_counts = {key: len(value) for key, value in sorted(grouped.items())}

        def max_count_for(family_key: str) -> int | None:
            if target_only and target_families and family_key not in target_families:
                return None
            limits = list(default_max_counts)
            for max_count_map in max_count_maps:
                value = int(max_count_map.get(family_key, 0))
                if value > 0:
                    limits.append(value)
            return min(limits) if limits else None

        kept_indices: set[int] = set()
        for family_key, family_records in grouped.items():
            limit = max_count_for(family_key)
            if limit is None:
                kept_indices.update(index for index, _ in family_records)
                continue
            ranked = sorted(
                family_records,
                key=lambda item: (self._record_balance_score(item[1]), -item[0]),
                reverse=True,
            )
            kept_indices.update(index for index, _ in ranked[: max(0, limit)])

        balanced_records = [record for index, record in indexed_records if index in kept_indices]
        after_counts = Counter(self._record_balance_key(record, key_axis) for record in balanced_records)
        required_family_keys = target_families or sorted(before_counts)
        missing_min_counts = []
        for family_key in required_family_keys:
            minimum = int(min_count_map.get(family_key, default_min_count))
            if minimum <= 0:
                continue
            count = int(after_counts.get(family_key, 0))
            if count < minimum:
                missing_min_counts.append(
                    {
                        "axis": key_axis,
                        "name": family_key,
                        "count": count,
                        "minimum": minimum,
                    }
                )

        return balanced_records, {
            "enabled": True,
            "axis": key_axis,
            "item_name": item_name,
            "target_families": target_families,
            "before_counts": before_counts,
            "after_counts": dict(sorted(after_counts.items())),
            "dropped": len(records) - len(balanced_records),
            "max_fraction_per_family": max_fraction,
            "default_max_counts": default_max_counts,
            "missing_min_counts": missing_min_counts,
        }

    def _apply_training_family_balance(
        self,
        records: list[dict[str, Any]],
        audit: list[dict[str, Any]],
        weighted_cfg: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        balanced_records, summary = self._apply_record_family_balance(
            records,
            dict(weighted_cfg.get("family_balance", {})),
            item_name="samples",
            max_count_keys=("max_samples_per_family", "max_per_family"),
            min_count_keys=("min_samples_per_family", "min_per_family"),
        )
        if not summary.get("enabled"):
            return balanced_records, summary

        kept_identities = {self._record_identity(record) for record in balanced_records}
        for audit_row in audit:
            if _finite_float(audit_row.get("sample_weight"), 0.0) <= 0:
                continue
            audit_identity = (
                str(audit_row.get("group_id") or ""),
                str(audit_row.get("candidate_index") if audit_row.get("candidate_index") is not None else ""),
                str(audit_row.get("training_status") or ""),
            )
            if audit_identity in kept_identities:
                continue
            audit_row["family_balance_dropped"] = True
            audit_row["family_balance_original_weight"] = audit_row.get("sample_weight")
            audit_row["sample_weight"] = 0.0
            audit_row["weight_reason"] = "family_balance_cap"
        return balanced_records, summary

    def _apply_preference_family_balance(
        self,
        records: list[dict[str, Any]],
        preference_cfg: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._apply_record_family_balance(
            records,
            dict(preference_cfg.get("family_balance", {})),
            item_name="pairs",
            max_count_keys=("max_pairs_per_family", "max_per_family"),
            min_count_keys=("min_pairs_per_family", "min_per_family"),
        )

    def _training_records_from_payloads(
        self,
        candidate_payloads: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        records: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        for payload in candidate_payloads:
            weight, reason = self._candidate_training_weight(payload)
            evaluation = payload.get("evaluator") or payload.get("solver") or {}
            proposal = payload.get("proposal", {})
            candidate_status = (
                "feasible_ranked_positive"
                if reason == "feasible_ranked_positive"
                else payload.get("status")
            )
            audit_row = {
                "group_id": payload.get("group_id"),
                "candidate_index": payload.get("candidate_index"),
                "record_key": payload.get("record_key"),
                "status": payload.get("status"),
                "training_status": candidate_status,
                "instruction": proposal.get("instruction"),
                "sample_weight": weight,
                "weight_reason": reason,
                "edited_image": payload.get("edited_image_path"),
                "scores": {
                    "total_score": evaluation.get("total_score") if isinstance(evaluation, dict) else None,
                    "component_scores": evaluation.get("component_scores", {}) if isinstance(evaluation, dict) else {},
                },
            }
            audit.append(audit_row)
            if weight <= 0:
                continue
            edited_path = resolve_path(str(payload.get("edited_image_path", "")))
            source_path = resolve_path(str(payload.get("image_path", "")))
            if edited_path is None or source_path is None or not edited_path.exists() or not source_path.exists():
                audit_row["sample_weight"] = 0.0
                audit_row["weight_reason"] = "missing_training_image"
                continue
            records.append(
                {
                    "prompt": str(proposal.get("instruction", "")),
                    "image": relative_to_repo(edited_path),
                    "edit_image": relative_to_repo(source_path),
                    "sample_weight": round(float(weight), 6),
                    "record_key": payload.get("record_key"),
                    "group_id": payload.get("group_id"),
                    "candidate_index": payload.get("candidate_index"),
                    "candidate_status": candidate_status,
                    "source_candidate_status": payload.get("status"),
                    "weight_reason": reason,
                    "operation_id": proposal.get("operation_id"),
                    "family": proposal.get("family"),
                    "structured_edit": proposal.get("structured_edit", {}),
                    "scores": audit_row["scores"],
                }
            )
        training_cfg = self.config.get("training", {})
        weighted_cfg = dict(training_cfg.get("weighted_sft", {}))
        records, family_balance_summary = self._apply_training_family_balance(records, audit, weighted_cfg)
        included = len(records)
        weight_sum = sum(float(record.get("sample_weight", 1.0)) for record in records)
        accepted_included = sum(1 for record in records if record.get("candidate_status") == "accepted")
        rejected_included = sum(1 for record in records if record.get("candidate_status") == "rejected")
        feasible_ranked_positive_included = sum(
            1 for record in records if record.get("candidate_status") == "feasible_ranked_positive"
        )
        summary = {
            "candidates": len(candidate_payloads),
            "included": included,
            "accepted_included": accepted_included,
            "rejected_included": rejected_included,
            "feasible_ranked_positive_included": feasible_ranked_positive_included,
            "weight_sum": weight_sum,
            "avg_weight": weight_sum / max(included, 1),
            "per_family": dict(sorted(Counter(str(record.get("family") or "unknown") for record in records).items())),
            "per_edit_type": dict(
                sorted(Counter(self._record_balance_key(record, "edit_type") for record in records).items())
            ),
            "family_balance": family_balance_summary,
        }
        return records, audit, summary

    @staticmethod
    def _payload_effective_reward(payload: dict[str, Any]) -> float:
        evaluation = payload.get("evaluator") or payload.get("solver") or {}
        if not isinstance(evaluation, dict):
            return 0.0
        component_scores = evaluation.get("component_scores", {})
        if not isinstance(component_scores, dict):
            component_scores = {}
        return _finite_float(
            component_scores.get("cepr_reward"),
            _finite_float(
                evaluation.get("total_score"),
                _finite_float(component_scores.get("relative_quality_score")),
            ),
        )

    @staticmethod
    def _payload_raw_reward(payload: dict[str, Any]) -> float:
        evaluation = payload.get("evaluator") or payload.get("solver") or {}
        if not isinstance(evaluation, dict):
            return 0.0
        component_scores = evaluation.get("component_scores", {})
        if not isinstance(component_scores, dict):
            component_scores = {}
        return _finite_float(
            component_scores.get("cepr_raw_reward"),
            _finite_float(
                component_scores.get("relative_quality_score"),
                _finite_float(evaluation.get("total_score")),
            ),
        )

    @staticmethod
    def _payload_component_score(payload: dict[str, Any], *names: str, default: float = 0.0) -> float:
        evaluation = payload.get("evaluator") or payload.get("solver") or {}
        if not isinstance(evaluation, dict):
            return default
        component_scores = evaluation.get("component_scores", {})
        if not isinstance(component_scores, dict):
            component_scores = {}
        signals = evaluation.get("signals", {})
        if not isinstance(signals, dict):
            signals = {}
        for name in names:
            value = _finite_float(component_scores.get(name), math.nan)
            if math.isfinite(value):
                return value
            value = _finite_float(signals.get(name), math.nan)
            if math.isfinite(value):
                return value
        return default

    @staticmethod
    def _payload_named_component_score(payload: dict[str, Any], name: str, default: float = 0.0) -> float:
        aliases = {
            "semantic_edit": (
                "cepr_semantic_edit",
                "cepr_edit_specificity",
                "rubric_edit_success",
                "rubric_required_after",
                "internal_vlm_judge_semantic",
            ),
            "preservation": (
                "cepr_preservation",
                "conservative_region_reward",
                "conservative_outside_preservation",
                "rubric_preservation",
                "cepr_preservation_score",
                "spatial_outside_preservation",
                "internal_vlm_judge_preservation",
            ),
            "validity": (
                "subject_extraction_validity_effective",
                "cepr_validity_effective",
                "cepr_validity",
                "rubric_validity",
                "cepr_validity_score",
                "internal_vlm_judge_artifact_free",
            ),
            "artifact_free": (
                "internal_vlm_judge_artifact_free",
                "cepr_validity",
                "rubric_validity",
                "cepr_validity_score",
            ),
            "quality": (
                "internal_vlm_judge_artifact_free",
                "internal_vlm_judge_score",
                "cepr_validity",
                "rubric_validity",
                "cepr_raw_reward",
                "rubric_cepr_raw_reward",
            ),
            "taxonomy": ("cepr_taxonomy",),
            "reward": (
                "cepr_raw_reward",
                "rubric_cepr_raw_reward",
                "cepr_reward",
                "rubric_cepr_reward",
                "conservative_region_reward",
                "rubric_reward",
                "internal_vlm_judge_combined_raw_reward",
                "internal_vlm_judge_score",
                "hybrid_total_score",
            ),
            "required": ("rubric_required_after",),
            "source": ("rubric_source_grounded", "rubric_source_grounding", "rubric_source_present"),
            "judge": ("internal_vlm_judge_score", "internal_vlm_judge_combined_raw_reward"),
            "target_object_visible_after": (
                "internal_vlm_judge_target_object_visible_after",
                "internal_vlm_judge_target_correctness",
            ),
            "internal_vlm_judge_target_object_visible_after": (
                "internal_vlm_judge_target_object_visible_after",
                "internal_vlm_judge_target_correctness",
            ),
            "internal": (
                "internal_vlm_judge_supported",
                "internal_qwen_score",
                "cepr_internal_supported",
                "internal_supported",
            ),
        }
        component_names = aliases.get(str(name), (str(name),))
        return SelfEvolveRunner._payload_component_score(payload, *component_names, default=default)

    @staticmethod
    def _candidate_role(payload: dict[str, Any]) -> str:
        return str(payload.get("candidate_role") or "policy")

    @staticmethod
    def _role_matches(role: str, patterns: list[str]) -> bool:
        if not patterns:
            return False
        role = str(role)
        for pattern in patterns:
            pattern = str(pattern).strip()
            if not pattern:
                continue
            if pattern.endswith("*") and role.startswith(pattern[:-1]):
                return True
            if role == pattern or role.startswith(f"{pattern}:"):
                return True
        return False

    def _payload_conservative_pair_score(self, payload: dict[str, Any]) -> float:
        """Score a candidate for pair selection with constraints as first-class terms."""

        semantic = max(
            self._payload_component_score(payload, "cepr_semantic_edit", default=0.0),
            self._payload_component_score(payload, "cepr_edit_specificity", default=0.0),
            self._payload_component_score(payload, "rubric_edit_success", default=0.0),
            self._payload_component_score(payload, "rubric_required_after", default=0.0),
        )
        preservation_terms = [
            self._payload_component_score(payload, "cepr_preservation", default=math.nan),
            self._payload_component_score(payload, "conservative_region_reward", default=math.nan),
            self._payload_component_score(payload, "conservative_outside_preservation", default=math.nan),
            self._payload_component_score(payload, "rubric_preservation", default=math.nan),
        ]
        validity_terms = [
            self._payload_component_score(payload, "cepr_validity", default=math.nan),
            self._payload_component_score(payload, "rubric_validity", default=math.nan),
        ]
        preservation = min([value for value in preservation_terms if math.isfinite(value)] or [0.0])
        validity = min([value for value in validity_terms if math.isfinite(value)] or [0.0])
        components = [
            self._payload_raw_reward(payload),
            semantic,
            preservation,
            validity,
            self._payload_component_score(payload, "cepr_taxonomy", default=1.0),
        ]
        judge_reliable = self._payload_component_score(payload, "internal_vlm_judge_reliable", default=0.0) >= 0.5
        judge_supported = self._payload_component_score(payload, "internal_vlm_judge_supported", default=0.0) >= 0.5
        judge_score = self._payload_component_score(payload, "internal_vlm_judge_score", default=0.0)
        judge_low_score = self._payload_component_score(payload, "internal_vlm_judge_low_score", default=0.0) >= 0.5
        if judge_supported and (judge_reliable or judge_low_score or judge_score < 0.35):
            components.extend(
                [
                    judge_score,
                    self._payload_component_score(payload, "internal_vlm_judge_semantic", default=0.0),
                    self._payload_component_score(payload, "internal_vlm_judge_preservation", default=0.0),
                    self._payload_component_score(payload, "internal_vlm_judge_artifact_free", default=0.0),
                ]
            )
        product = 1.0
        valid_count = 0
        for component in components:
            if not math.isfinite(float(component)):
                continue
            product *= max(_clamp(float(component)), 1.0e-6)
            valid_count += 1
        if valid_count <= 0:
            return 0.0
        return _clamp(product ** (1.0 / valid_count))

    def _payload_passes_strict_success_filter(
        self,
        payload: dict[str, Any],
        success_cfg: dict[str, Any],
    ) -> bool:
        edit_type = self._payload_edit_type(payload)
        edit_type_cfg: dict[str, Any] = {}
        for override_key in ("by_edit_type", "overrides_by_edit_type"):
            raw_overrides = success_cfg.get(override_key, {})
            if isinstance(raw_overrides, dict) and isinstance(raw_overrides.get(edit_type), dict):
                edit_type_cfg.update(raw_overrides[edit_type])

        def cfg_get(key: str, default: Any = None) -> Any:
            return edit_type_cfg.get(key, success_cfg.get(key, default))

        require_judge = bool(cfg_get("require_internal_vlm_judge", True))
        require_reliable = bool(cfg_get("require_reliable_judge", False))
        judge_supported = self._payload_component_score(payload, "internal_vlm_judge_supported", default=0.0) >= 0.5
        judge_reliable = self._payload_component_score(payload, "internal_vlm_judge_reliable", default=0.0) >= 0.5
        if require_judge and not judge_supported:
            return False
        if require_reliable and not judge_reliable:
            return False
        if judge_supported:
            judge_floors = {
                "internal_vlm_judge_score": cfg_get("min_judge_score", 0.55),
                "internal_vlm_judge_semantic": cfg_get("min_judge_semantic", 0.55),
                "internal_vlm_judge_preservation": cfg_get("min_judge_preservation", 0.55),
                "internal_vlm_judge_artifact_free": cfg_get("min_judge_artifact_free", 0.55),
            }
            for name, raw_floor in judge_floors.items():
                floor = _finite_float(raw_floor, math.nan)
                if math.isfinite(floor) and self._payload_component_score(payload, name, default=0.0) < floor:
                    return False

        raw_min_components = success_cfg.get("min_component_scores", {})
        min_components = dict(raw_min_components) if isinstance(raw_min_components, dict) else {}
        if isinstance(success_cfg.get("min_component_scores_by_edit_type"), dict):
            edit_type_components = success_cfg["min_component_scores_by_edit_type"].get(edit_type, {})
            if isinstance(edit_type_components, dict):
                min_components.update(edit_type_components)
        if isinstance(edit_type_cfg.get("min_component_scores"), dict):
            min_components.update(edit_type_cfg["min_component_scores"])
        if isinstance(min_components, dict):
            for name, raw_floor in min_components.items():
                floor = _finite_float(raw_floor, math.nan)
                if math.isfinite(floor) and self._payload_named_component_score(payload, str(name), default=0.0) < floor:
                    return False

        if bool(cfg_get("require_accepted_status", False)) and payload.get("status") != "accepted":
            return False
        min_effective_reward = _finite_float(cfg_get("min_effective_reward"), math.nan)
        if math.isfinite(min_effective_reward) and self._payload_effective_reward(payload) < min_effective_reward:
            return False
        return True

    def _preference_group_productivity_decision(
        self,
        image_rows: list[dict[str, Any]],
        filter_cfg: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str | None]:
        if not isinstance(filter_cfg, dict) or not bool(filter_cfg.get("enabled", False)):
            return True, {"enabled": False}, None

        scope = str(filter_cfg.get("success_scope", "policy")).strip().lower()
        policy_roles = _coerce_str_list(filter_cfg.get("policy_roles", ["policy"]))
        if scope in {"policy", "policy_only"}:
            scored_rows = [
                row
                for row in image_rows
                if self._role_matches(self._candidate_role(row), policy_roles)
            ]
        else:
            scored_rows = list(image_rows)

        min_candidates = int(filter_cfg.get("min_candidates", 2))
        if len(scored_rows) < min_candidates:
            telemetry = {
                "enabled": True,
                "scope": scope,
                "candidate_count": len(scored_rows),
                "min_candidates": min_candidates,
            }
            return False, telemetry, "productive_group_too_few_candidates"

        edit_type = self._dominant_payload_edit_type(scored_rows)
        pass_count = sum(
            1 for row in scored_rows if self._payload_passes_strict_success_filter(row, filter_cfg)
        )
        success_rate = pass_count / max(len(scored_rows), 1)
        min_rate_by_edit_type = _coerce_float_map(filter_cfg.get("min_success_rate_by_edit_type"))
        max_rate_by_edit_type = _coerce_float_map(filter_cfg.get("max_success_rate_by_edit_type"))
        min_rate = float(min_rate_by_edit_type.get(edit_type, filter_cfg.get("min_success_rate", 0.25)))
        max_rate = float(max_rate_by_edit_type.get(edit_type, filter_cfg.get("max_success_rate", 0.75)))
        min_success_count = 0
        raw_min_count_overrides = filter_cfg.get("min_success_count_by_edit_type", {})
        if isinstance(raw_min_count_overrides, dict) and edit_type in raw_min_count_overrides:
            min_success_count = int(raw_min_count_overrides.get(edit_type) or 0)
        elif filter_cfg.get("min_success_count") is not None:
            min_success_count = int(filter_cfg.get("min_success_count") or 0)
        telemetry = {
            "enabled": True,
            "scope": scope,
            "edit_type": edit_type,
            "candidate_count": len(scored_rows),
            "pass_count": pass_count,
            "success_rate": round(float(success_rate), 6),
            "min_success_count": min_success_count,
            "min_success_rate": min_rate,
            "max_success_rate": max_rate,
        }
        if min_success_count > 0 and pass_count < min_success_count:
            return False, telemetry, "productive_group_min_success_count"
        if success_rate < min_rate:
            return False, telemetry, "productive_group_all_or_most_fail"
        if success_rate > max_rate:
            return False, telemetry, "productive_group_all_or_most_pass"
        return True, telemetry, None

    def _payload_pair_score(self, payload: dict[str, Any], score_mode: str) -> float:
        mode = str(score_mode or "effective").strip().lower()
        if mode in {"raw", "raw_reward"}:
            return self._payload_raw_reward(payload)
        if mode in {"conservative", "constraint", "constraint_aware"}:
            return self._payload_conservative_pair_score(payload)
        if mode in {"judge", "vlm"}:
            return self._payload_named_component_score(payload, "judge", default=0.0)
        return self._payload_effective_reward(payload)

    @staticmethod
    def _is_rank_only_feasible_reject(payload: dict[str, Any]) -> bool:
        """True when a candidate passed the edit contract but lost top-k selection."""

        if payload.get("status") != "rejected":
            return False
        evaluation = payload.get("evaluator") or payload.get("solver") or {}
        if not isinstance(evaluation, dict):
            return False
        signals = evaluation.get("signals", {})
        if not isinstance(signals, dict):
            return False
        feasible = _finite_float(signals.get("feasible"), 0.0) >= 0.5
        rank = _finite_float(signals.get("feasible_rank"), 0.0)
        reject_reason = str(signals.get("rubric_reject_reason") or "").strip()
        # Fallback acceptors can annotate feasible-but-not-top-k rows with
        # reasons such as "accepted_vlm_replacement_detector_fallback". These
        # are rank-only alternatives, not true failures, and should not become
        # negative preference examples.
        return feasible and rank > 0.0 and reject_reason.startswith("accepted")

    def _payload_passes_preference_floors(
        self,
        payload: dict[str, Any],
        min_components: dict[str, Any],
    ) -> tuple[bool, dict[str, float], str | None]:
        observed: dict[str, float] = {}
        for name, raw_floor in min_components.items():
            floor = _finite_float(raw_floor, math.nan)
            if not math.isfinite(floor):
                continue
            score = self._payload_named_component_score(payload, str(name), default=0.0)
            observed[str(name)] = score
            if score < floor:
                return False, observed, f"floor_{name}"
        return True, observed, None

    def _preference_vlm_pair_guard(
        self,
        chosen: dict[str, Any],
        rejected: dict[str, Any],
        guard_cfg: dict[str, Any],
    ) -> tuple[float, dict[str, Any], str | None]:
        if not isinstance(guard_cfg, dict) or not bool(guard_cfg.get("enabled", False)):
            return 1.0, {"enabled": False}, None

        chosen_judge = self._payload_named_component_score(chosen, "judge", default=0.0)
        rejected_judge = self._payload_named_component_score(rejected, "judge", default=0.0)
        chosen_semantic = self._payload_component_score(chosen, "internal_vlm_judge_semantic", default=0.0)
        chosen_preservation = self._payload_component_score(chosen, "internal_vlm_judge_preservation", default=0.0)
        chosen_artifact_free = self._payload_component_score(chosen, "internal_vlm_judge_artifact_free", default=0.0)
        chosen_reliable = self._payload_component_score(chosen, "internal_vlm_judge_reliable", default=0.0) >= 0.5
        rejected_reliable = self._payload_component_score(rejected, "internal_vlm_judge_reliable", default=0.0) >= 0.5
        chosen_conf = self._payload_component_score(chosen, "internal_vlm_judge_confidence", default=0.0)
        rejected_conf = self._payload_component_score(rejected, "internal_vlm_judge_confidence", default=0.0)
        chosen_supported = self._payload_component_score(chosen, "internal_vlm_judge_supported", default=0.0) >= 0.5
        rejected_supported = self._payload_component_score(rejected, "internal_vlm_judge_supported", default=0.0) >= 0.5
        telemetry = {
            "enabled": True,
            "chosen_judge": round(float(chosen_judge), 6),
            "rejected_judge": round(float(rejected_judge), 6),
            "judge_margin": round(float(chosen_judge - rejected_judge), 6),
            "chosen_semantic": round(float(chosen_semantic), 6),
            "chosen_preservation": round(float(chosen_preservation), 6),
            "chosen_artifact_free": round(float(chosen_artifact_free), 6),
            "chosen_supported": bool(chosen_supported),
            "rejected_supported": bool(rejected_supported),
            "chosen_reliable": bool(chosen_reliable),
            "rejected_reliable": bool(rejected_reliable),
            "chosen_confidence": round(float(chosen_conf), 6),
            "rejected_confidence": round(float(rejected_conf), 6),
        }

        if bool(guard_cfg.get("reject_explicit_vlm_fail", True)) and chosen_supported:
            fail_floor = float(guard_cfg.get("explicit_fail_floor", 0.05))
            if chosen_judge <= fail_floor:
                return 0.0, telemetry, "vlm_chosen_explicit_fail"
            if chosen_semantic <= fail_floor:
                return 0.0, telemetry, "vlm_chosen_semantic_explicit_fail"

        require_reliable = bool(guard_cfg.get("require_reliable_chosen", False))
        if require_reliable and not chosen_reliable:
            return 0.0, telemetry, "vlm_chosen_unreliable"
        require_both = bool(guard_cfg.get("require_reliable_pair", False))
        if require_both and not (chosen_reliable and rejected_reliable):
            return 0.0, telemetry, "vlm_pair_unreliable"

        chosen_floors = {
            "judge": (chosen_judge, guard_cfg.get("min_chosen_judge")),
            "semantic": (chosen_semantic, guard_cfg.get("min_chosen_semantic")),
            "preservation": (chosen_preservation, guard_cfg.get("min_chosen_preservation")),
            "artifact_free": (chosen_artifact_free, guard_cfg.get("min_chosen_artifact_free")),
        }
        for name, (score, raw_floor) in chosen_floors.items():
            floor = _finite_float(raw_floor, math.nan)
            if math.isfinite(floor) and score < floor:
                return 0.0, telemetry, f"vlm_chosen_{name}_below_floor"

        if chosen_reliable and rejected_reliable:
            min_margin = float(guard_cfg.get("min_judge_margin", -0.05))
            if chosen_judge - rejected_judge < min_margin:
                return 0.0, telemetry, "vlm_judge_margin_too_small"

        min_multiplier = float(guard_cfg.get("min_weight_multiplier", 0.35))
        confidence = min(chosen_conf if chosen_reliable else 0.5, rejected_conf if rejected_reliable else 0.5)
        multiplier = min_multiplier + (1.0 - min_multiplier) * _clamp(confidence)
        telemetry["weight_multiplier"] = round(float(multiplier), 6)
        return _clamp(multiplier), telemetry, None

    def _payload_axis_score(
        self,
        payload: dict[str, Any],
        axis_cfg: Any,
        *,
        default_names: list[str],
        default_reducer: str = "max",
        default: float = 0.0,
    ) -> tuple[float, dict[str, float]]:
        if isinstance(axis_cfg, dict):
            names = _coerce_str_list(axis_cfg.get("metrics", axis_cfg.get("names", default_names)))
            reducer = str(axis_cfg.get("reducer", default_reducer)).strip().lower()
        else:
            names = _coerce_str_list(axis_cfg) or list(default_names)
            reducer = default_reducer
        values: dict[str, float] = {}
        for name in names:
            score = self._payload_named_component_score(payload, name, default=math.nan)
            if math.isfinite(score):
                values[str(name)] = float(score)
        if not values:
            return default, values
        if reducer in {"min", "worst", "floor"}:
            return min(values.values()), values
        if reducer in {"mean", "avg", "average"}:
            return sum(values.values()) / max(len(values), 1), values
        return max(values.values()), values

    def _preference_pareto_pair_guard(
        self,
        chosen: dict[str, Any],
        rejected: dict[str, Any],
        *,
        edit_type: str,
        guard_cfg: dict[str, Any],
    ) -> tuple[float, dict[str, Any], str | None]:
        """Constrained pair gate for conservative editing.

        Normal candidate pairs are only useful when the chosen edit improves the
        target edit axis without degrading preservation/quality axes. Source-noop
        negatives are a special under-edit contrast: they can be worse on edit
        success by construction, so they use absolute chosen floors instead of a
        no-degradation comparison against the unedited source.
        """

        if not isinstance(guard_cfg, dict) or not bool(guard_cfg.get("enabled", False)):
            return 1.0, {"enabled": False}, None

        rejected_role = self._candidate_role(rejected)
        is_source_noop = (
            bool(rejected.get("source_noop_negative"))
            or str(rejected.get("status") or "") == "source_noop"
            or rejected_role.startswith("reference:source_noop")
        )
        edit_axis_cfg = guard_cfg.get(
            "edit_success_axis",
            {
                "metrics": [
                    "semantic_edit",
                    "internal_vlm_judge_edit_success",
                    "internal_vlm_judge_target_correctness",
                    "internal_vlm_judge_object_absence",
                    "judge",
                ],
                "reducer": "max",
            },
        )
        chosen_edit, chosen_edit_components = self._payload_axis_score(
            chosen,
            edit_axis_cfg,
            default_names=["semantic_edit", "judge"],
            default_reducer="max",
            default=0.0,
        )
        rejected_edit, rejected_edit_components = self._payload_axis_score(
            rejected,
            edit_axis_cfg,
            default_names=["semantic_edit", "judge"],
            default_reducer="max",
            default=0.0,
        )
        min_edit_delta_by_type = _coerce_float_map(guard_cfg.get("min_edit_delta_by_edit_type"))
        source_noop_min_edit_delta_by_type = _coerce_float_map(
            guard_cfg.get("source_noop_min_edit_delta_by_edit_type")
        )
        min_edit_delta = float(
            source_noop_min_edit_delta_by_type.get(
                edit_type,
                guard_cfg.get("source_noop_min_edit_delta", guard_cfg.get("min_edit_delta", 0.02)),
            )
            if is_source_noop
            else min_edit_delta_by_type.get(edit_type, guard_cfg.get("min_edit_delta", 0.02))
        )
        edit_delta = chosen_edit - rejected_edit
        telemetry: dict[str, Any] = {
            "enabled": True,
            "source_noop_negative": bool(is_source_noop),
            "edit_type": edit_type,
            "chosen_edit_success": round(float(chosen_edit), 6),
            "rejected_edit_success": round(float(rejected_edit), 6),
            "edit_delta": round(float(edit_delta), 6),
            "min_edit_delta": round(float(min_edit_delta), 6),
            "chosen_edit_components": {
                key: round(float(value), 6) for key, value in sorted(chosen_edit_components.items())
            },
            "rejected_edit_components": {
                key: round(float(value), 6) for key, value in sorted(rejected_edit_components.items())
            },
        }
        if edit_delta < min_edit_delta:
            return 0.0, telemetry, "pareto_edit_delta_too_small"

        min_chosen_scores = self._merge_edit_type_overrides(
            dict(guard_cfg.get("min_chosen_scores", {})),
            guard_cfg.get("min_chosen_scores_by_edit_type"),
            edit_type,
        )
        floor_scores: dict[str, float] = {}
        for axis_name, raw_floor in min_chosen_scores.items():
            floor = _finite_float(raw_floor, math.nan)
            if not math.isfinite(floor):
                continue
            score = self._payload_named_component_score(chosen, str(axis_name), default=0.0)
            floor_scores[str(axis_name)] = score
            if score < floor:
                telemetry["chosen_floor_scores"] = {
                    key: round(float(value), 6) for key, value in sorted(floor_scores.items())
                }
                telemetry["failed_floor"] = str(axis_name)
                telemetry["failed_floor_value"] = round(float(floor), 6)
                return 0.0, telemetry, f"pareto_chosen_{axis_name}_below_floor"
        telemetry["chosen_floor_scores"] = {
            key: round(float(value), 6) for key, value in sorted(floor_scores.items())
        }

        no_degradation_axes = guard_cfg.get("no_degradation_axes", {})
        if not isinstance(no_degradation_axes, dict):
            no_degradation_axes = {}
        axis_telemetry: dict[str, Any] = {}
        if not is_source_noop or bool(guard_cfg.get("compare_source_noop_no_degradation", False)):
            for axis_name, raw_axis_cfg in no_degradation_axes.items():
                axis_cfg = raw_axis_cfg if isinstance(raw_axis_cfg, dict) else {}
                chosen_score, chosen_components = self._payload_axis_score(
                    chosen,
                    axis_cfg,
                    default_names=[str(axis_name)],
                    default_reducer=str(axis_cfg.get("reducer", "min")) if isinstance(axis_cfg, dict) else "min",
                    default=0.0,
                )
                rejected_score, rejected_components = self._payload_axis_score(
                    rejected,
                    axis_cfg,
                    default_names=[str(axis_name)],
                    default_reducer=str(axis_cfg.get("reducer", "min")) if isinstance(axis_cfg, dict) else "min",
                    default=0.0,
                )
                allowed_drop_by_type = _coerce_float_map(axis_cfg.get("allowed_drop_by_edit_type"))
                allowed_drop = float(
                    allowed_drop_by_type.get(
                        edit_type,
                        axis_cfg.get("allowed_drop", guard_cfg.get("allowed_quality_drop", 0.02)),
                    )
                )
                axis_delta = chosen_score - rejected_score
                axis_telemetry[str(axis_name)] = {
                    "chosen": round(float(chosen_score), 6),
                    "rejected": round(float(rejected_score), 6),
                    "delta": round(float(axis_delta), 6),
                    "allowed_drop": round(float(allowed_drop), 6),
                    "chosen_components": {
                        key: round(float(value), 6) for key, value in sorted(chosen_components.items())
                    },
                    "rejected_components": {
                        key: round(float(value), 6) for key, value in sorted(rejected_components.items())
                    },
                }
                if chosen_score + allowed_drop < rejected_score:
                    telemetry["no_degradation_axes"] = axis_telemetry
                    telemetry["failed_axis"] = str(axis_name)
                    return 0.0, telemetry, f"pareto_{axis_name}_degraded"
        telemetry["no_degradation_axes"] = axis_telemetry

        min_multiplier = float(guard_cfg.get("min_weight_multiplier", 0.75))
        margin_scale = float(guard_cfg.get("edit_margin_weight_scale", 0.25))
        margin_clip = float(guard_cfg.get("edit_margin_clip", 0.50))
        multiplier = min(1.0, min_multiplier + margin_scale * min(max(edit_delta, 0.0), margin_clip))
        telemetry["weight_multiplier"] = round(float(multiplier), 6)
        return _clamp(multiplier, 0.0, 1.0), telemetry, None

    @staticmethod
    def _merge_edit_type_overrides(base: dict[str, Any], overrides: Any, edit_type: str) -> dict[str, Any]:
        merged = dict(base)
        if isinstance(overrides, dict):
            raw_override = overrides.get(edit_type)
            if isinstance(raw_override, dict):
                merged.update(raw_override)
        return merged

    def _preference_generalization_calibration(
        self,
        payload: dict[str, Any],
        edit_type: str,
        calibration_cfg: dict[str, Any],
    ) -> tuple[float, float, dict[str, float], str | None]:
        if not bool(calibration_cfg.get("enabled", False)):
            return 1.0, 1.0, {}, None

        min_components = self._merge_edit_type_overrides(
            dict(calibration_cfg.get("min_component_scores", {})),
            calibration_cfg.get("min_component_scores_by_edit_type"),
            edit_type,
        )
        observed: dict[str, float] = {}
        for name, raw_floor in min_components.items():
            floor = _finite_float(raw_floor, math.nan)
            if not math.isfinite(floor):
                continue
            score = self._payload_named_component_score(payload, str(name), default=0.0)
            observed[str(name)] = score
            if bool(calibration_cfg.get("reject_below_component_floor", True)) and score < floor:
                return 0.0, 0.0, observed, f"generalization_floor_{name}"

        component_weights = dict(calibration_cfg.get("component_weights", {}))
        weighted_total = 0.0
        total_weight = 0.0
        for name, raw_weight in component_weights.items():
            weight = _finite_float(raw_weight, math.nan)
            if not math.isfinite(weight) or weight <= 0.0:
                continue
            score = self._payload_named_component_score(payload, str(name), default=0.5)
            observed[str(name)] = score
            weighted_total += weight * _clamp(score)
            total_weight += weight

        confidence = weighted_total / total_weight if total_weight > 0.0 else 1.0
        min_confidence = float(calibration_cfg.get("min_confidence", 0.0))
        if (
            bool(calibration_cfg.get("reject_below_min_confidence", False))
            and confidence < min_confidence
        ):
            return 0.0, confidence, observed, "generalization_confidence"

        min_multiplier = float(calibration_cfg.get("min_weight_multiplier", 0.25))
        power = max(float(calibration_cfg.get("confidence_power", 1.0)), 1.0e-6)
        multiplier = min_multiplier + (1.0 - min_multiplier) * (_clamp(confidence) ** power)
        return _clamp(multiplier, 0.0, 1.0), confidence, observed, None

    def _payload_failure_tags(self, payload: dict[str, Any], failure_cfg: dict[str, Any]) -> list[str]:
        thresholds = dict(failure_cfg.get("thresholds", {}))
        semantic_min = float(thresholds.get("semantic_edit", 0.35))
        preservation_min = float(thresholds.get("preservation", 0.60))
        validity_min = float(thresholds.get("validity", 0.70))
        taxonomy_min = float(thresholds.get("taxonomy", 0.25))
        reward_min = float(thresholds.get("reward", 0.35))
        judge_min = float(thresholds.get("judge", 0.0))

        tags = _coerce_str_list(payload.get("failure_tags"))
        if judge_min > 0.0 and self._payload_named_component_score(payload, "judge", default=1.0) < judge_min:
            tags.append("judge_low_score")
        if self._payload_named_component_score(payload, "semantic_edit", default=0.0) < semantic_min:
            tags.append("under_edit")
        if self._payload_named_component_score(payload, "preservation", default=0.0) < preservation_min:
            tags.append("preservation_drift")
        if self._payload_named_component_score(payload, "validity", default=0.0) < validity_min:
            tags.append("invalid_or_artifact")
        if self._payload_named_component_score(payload, "taxonomy", default=0.0) < taxonomy_min:
            tags.append("taxonomy_mismatch")
        if self._payload_raw_reward(payload) < reward_min:
            tags.append("weak_reward")
        if not tags:
            tags.append("hard_near_miss")
        return list(dict.fromkeys(tags))

    def _order_rejected_rows_for_hard_negative_mining(
        self,
        rejected_rows: list[dict[str, Any]],
        failure_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not bool(failure_cfg.get("enabled", False)) or len(rejected_rows) <= 1:
            for row in rejected_rows:
                row.setdefault("failure_tags", self._payload_failure_tags(row, failure_cfg))
            return rejected_rows

        priority = _coerce_str_list(
            failure_cfg.get(
                "priority",
                [
                    "hard_near_miss",
                    "preservation_drift",
                    "under_edit",
                    "invalid_or_artifact",
                    "taxonomy_mismatch",
                    "weak_reward",
                ],
            )
        )
        max_per_tag = int(failure_cfg.get("max_per_tag_first_pass", 1))
        tagged_rows = []
        for row in rejected_rows:
            tags = self._payload_failure_tags(row, failure_cfg)
            row["failure_tags"] = tags
            tagged_rows.append((row, tags))

        selected: list[dict[str, Any]] = []
        selected_ids: set[int] = set()
        tag_counts: Counter[str] = Counter()
        for tag in priority:
            for row, tags in tagged_rows:
                row_id = id(row)
                if row_id in selected_ids or tag not in tags or tag_counts[tag] >= max_per_tag:
                    continue
                selected.append(row)
                selected_ids.add(row_id)
                tag_counts[tag] += 1
                break
        for row, _tags in tagged_rows:
            if id(row) not in selected_ids:
                selected.append(row)
        return selected

    @staticmethod
    def _has_evaluation(payload: dict[str, Any]) -> bool:
        evaluation = payload.get("evaluator") or payload.get("solver")
        return isinstance(evaluation, dict)

    def _build_source_noop_negative(
        self,
        chosen: dict[str, Any],
        *,
        edit_type: str,
        cfg: dict[str, Any],
    ) -> dict[str, Any] | None:
        source_image = chosen.get("image_path")
        if not source_image:
            return None

        edit_types = set(_coerce_str_list(cfg.get("edit_types", [])))
        if edit_types and edit_type not in edit_types:
            return None

        allowed_statuses = set(_coerce_str_list(cfg.get("chosen_statuses", ["accepted"])))
        if allowed_statuses and str(chosen.get("status") or "") not in allowed_statuses:
            return None

        pair_score = float(cfg.get("pair_score", 0.0))
        preservation_score = float(cfg.get("preservation_score", 1.0))
        validity_score = float(cfg.get("validity_score", 1.0))
        taxonomy_score = float(cfg.get("taxonomy_score", 1.0))
        weight_scale_by_edit_type = _coerce_float_map(cfg.get("weight_scale_by_edit_type"))
        negative_weight_scale = float(weight_scale_by_edit_type.get(edit_type, cfg.get("weight_scale", 0.55)))
        failure_tags = _coerce_str_list(
            cfg.get(
                "failure_tags",
                ["source_noop", "under_edit", "weak_reward", "judge_low_score"],
            )
        )
        candidate_index_offset = int(cfg.get("candidate_index_offset", -100000))
        chosen_index = int(chosen.get("candidate_index", 0))
        component_scores = {
            "cepr_reward": pair_score,
            "cepr_raw_reward": pair_score,
            "cepr_semantic_edit": 0.0,
            "cepr_edit_specificity": 0.0,
            "rubric_edit_success": 0.0,
            "rubric_required_after": 0.0,
            "cepr_preservation": preservation_score,
            "rubric_preservation": preservation_score,
            "conservative_region_reward": preservation_score,
            "conservative_outside_preservation": preservation_score,
            "cepr_validity": validity_score,
            "rubric_validity": validity_score,
            "cepr_taxonomy": taxonomy_score,
            "internal_vlm_judge_score": 0.0,
            "internal_vlm_judge_semantic": 0.0,
            "internal_vlm_judge_preservation": preservation_score,
            "internal_vlm_judge_artifact_free": validity_score,
            "internal_vlm_judge_confidence": 1.0,
            "internal_vlm_judge_supported": 1.0,
            "internal_vlm_judge_reliable": 1.0,
            "internal_vlm_judge_combined_raw_reward": pair_score,
        }
        signals = {
            "source_noop_negative": 1.0,
            "internal_vlm_judge_supported": 1.0,
            "internal_vlm_judge_reliable": 1.0,
            "internal_vlm_judge_low_score": 1.0,
            "rubric_reject_reason": "source_noop_under_edit",
        }
        return {
            "group_id": chosen.get("group_id"),
            "candidate_index": candidate_index_offset - chosen_index,
            "candidate_role": str(cfg.get("candidate_role", "reference:source_noop")),
            "candidate_model_state": chosen.get("candidate_model_state"),
            "record_key": chosen.get("record_key"),
            "image_path": source_image,
            "caption": chosen.get("caption"),
            "proposal": chosen.get("proposal", {}),
            "status": "source_noop",
            "edited_image_path": source_image,
            "evaluator": {
                "global_score": pair_score,
                "local_score": preservation_score,
                "total_score": pair_score,
                "accepted": False,
                "component_scores": component_scores,
                "signals": signals,
            },
            "solver": {
                "global_score": pair_score,
                "local_score": preservation_score,
                "total_score": pair_score,
                "accepted": False,
                "component_scores": component_scores,
                "signals": signals,
            },
            "failure_tags": failure_tags,
            "negative_weight_scale": negative_weight_scale,
            "source_noop_negative": True,
        }

    def _preference_records_from_payloads(
        self,
        candidate_payloads: list[dict[str, Any]],
        round_index: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        training_cfg = self.config.get("training", {})
        preference_cfg = dict(training_cfg.get("preference", {}))
        if not bool(preference_cfg.get("enabled", False)):
            return [], {"enabled": False, "pairs": 0}

        max_pairs_per_group = int(preference_cfg.get("max_pairs_per_group", 3))
        pair_score_mode = str(preference_cfg.get("score_mode", "effective"))
        min_score_margin = float(preference_cfg.get("min_score_margin", 0.05))
        min_score_margin_by_edit_type = _coerce_float_map(preference_cfg.get("min_score_margin_by_edit_type"))
        min_chosen_reward = float(
            preference_cfg.get(
                "min_chosen_reward",
                self.config.get("evaluator", self.config.get("solver", {})).get("reward_threshold", 0.30),
            )
        )
        base_weight = float(preference_cfg.get("base_weight", 1.0))
        margin_weight_scale = float(preference_cfg.get("margin_weight_scale", 0.50))
        margin_clip = float(preference_cfg.get("margin_clip", 0.50))
        max_sample_weight = float(preference_cfg.get("max_sample_weight", 1.50))
        accepted_weight_scale_by_edit_type = _coerce_float_map(
            preference_cfg.get("accepted_weight_scale_by_edit_type")
        )
        accepted_preference_sft_weight = float(
            preference_cfg.get("accepted_preference_sft_weight", preference_cfg.get("preference_sft_weight", 0.10))
        )
        ranked_accepted_cfg = preference_cfg.get("ranked_accepted_pairs", {})
        if not isinstance(ranked_accepted_cfg, dict):
            ranked_accepted_cfg = {}
        include_ranked_accepted_pairs = bool(
            ranked_accepted_cfg.get(
                "enabled",
                preference_cfg.get("include_ranked_accepted_pairs", False),
            )
        )
        ranked_accepted_source_suffix = str(
            ranked_accepted_cfg.get("source_suffix", "ranked_accepted")
        ).strip() or "ranked_accepted"
        include_near_miss = bool(
            preference_cfg.get("include_near_miss_pairs", preference_cfg.get("near_miss_enabled", False))
        )
        near_miss_min_raw_reward = float(preference_cfg.get("near_miss_min_raw_reward", 0.48))
        near_miss_min_semantic_edit = float(preference_cfg.get("near_miss_min_semantic_edit", 0.30))
        near_miss_min_preservation = float(preference_cfg.get("near_miss_min_preservation", 0.50))
        near_miss_min_validity = float(preference_cfg.get("near_miss_min_validity", 0.50))
        near_miss_min_score_margin = float(preference_cfg.get("near_miss_min_score_margin", min_score_margin))
        near_miss_min_score_margin_by_edit_type = _coerce_float_map(
            preference_cfg.get("near_miss_min_score_margin_by_edit_type")
        )
        near_miss_weight_scale = float(preference_cfg.get("near_miss_weight_scale", 0.50))
        near_miss_preference_sft_weight = float(preference_cfg.get("near_miss_preference_sft_weight", 0.0))
        near_miss_allowed_statuses = set(
            _coerce_str_list(preference_cfg.get("near_miss_allowed_statuses", ["accepted"]))
        )
        near_miss_min_effective_reward = _finite_float(
            preference_cfg.get("near_miss_min_effective_reward"),
            math.nan,
        )
        near_miss_contract_cfg = preference_cfg.get("near_miss_contract_filter", {})
        if not isinstance(near_miss_contract_cfg, dict):
            near_miss_contract_cfg = {}
        near_miss_positive_anchor_cfg = preference_cfg.get("near_miss_positive_anchor_filter", {})
        if not isinstance(near_miss_positive_anchor_cfg, dict):
            near_miss_positive_anchor_cfg = {}
        verified_rejected_near_miss_cfg = preference_cfg.get("verified_rejected_near_miss", {})
        if not isinstance(verified_rejected_near_miss_cfg, dict):
            verified_rejected_near_miss_cfg = {}
        verified_rejected_near_miss_enabled = bool(verified_rejected_near_miss_cfg.get("enabled", False))
        verified_rejected_anchor_enabled = bool(
            verified_rejected_near_miss_cfg.get("allow_as_positive_anchor", True)
        )
        verified_rejected_compete_with_accepted = bool(
            verified_rejected_near_miss_cfg.get("compete_with_accepted", False)
        )
        verified_rejected_prefer_margin = float(
            verified_rejected_near_miss_cfg.get("prefer_over_accepted_margin", 0.0)
        )
        verified_rejected_unproductive_reasons = set(
            _coerce_str_list(
                verified_rejected_near_miss_cfg.get(
                    "unproductive_group_reasons",
                    ["productive_group_all_or_most_fail"],
                )
            )
        )
        calibration_cfg = preference_cfg.get("generalization_calibration", {})
        if not isinstance(calibration_cfg, dict):
            calibration_cfg = {}
        hard_negative_cfg = preference_cfg.get("hard_negative_mining", {})
        if not isinstance(hard_negative_cfg, dict):
            hard_negative_cfg = {}
        hard_negative_allow_unproductive = bool(
            hard_negative_cfg.get("allow_unproductive_strict_positive_pairs", False)
        )
        hard_negative_unproductive_reasons = set(
            _coerce_str_list(
                hard_negative_cfg.get(
                    "unproductive_group_reasons",
                    ["productive_group_all_or_most_pass"],
                )
            )
        )
        hard_negative_explicit_tags = set(
            _coerce_str_list(
                hard_negative_cfg.get(
                    "explicit_failure_tags",
                    [
                        "judge_low_score",
                        "preservation_drift",
                        "under_edit",
                        "invalid_or_artifact",
                        "taxonomy_mismatch",
                        "weak_reward",
                    ],
                )
            )
        )
        hard_negative_require_explicit = bool(hard_negative_cfg.get("require_explicit_failure_tag", True))
        source_noop_negative_cfg = preference_cfg.get("source_noop_negative", {})
        if not isinstance(source_noop_negative_cfg, dict):
            source_noop_negative_cfg = {}
        source_noop_negative_enabled = bool(source_noop_negative_cfg.get("enabled", False))
        source_noop_allow_unproductive = bool(source_noop_negative_cfg.get("allow_unproductive_groups", True))
        source_noop_unproductive_reasons = set(
            _coerce_str_list(
                source_noop_negative_cfg.get(
                    "unproductive_group_reasons",
                    ["productive_group_all_or_most_pass"],
                )
            )
        )
        source_noop_only_when_no_candidate_negatives = bool(
            source_noop_negative_cfg.get("only_when_no_candidate_negatives", False)
        )
        source_noop_allow_ambiguous_positive = bool(
            source_noop_negative_cfg.get("allow_ambiguous_base_relative_positive", False)
        )
        source_noop_allow_ambiguous_positive_by_edit_type = _coerce_bool_map(
            source_noop_negative_cfg.get("allow_ambiguous_base_relative_positive_by_edit_type")
        )
        source_noop_source_suffix = str(source_noop_negative_cfg.get("source_suffix", "source_noop")).strip()
        if not source_noop_source_suffix:
            source_noop_source_suffix = "source_noop"
        exclude_rank_only_feasible_rejects = bool(
            preference_cfg.get("exclude_rank_only_feasible_rejects", True)
        )
        base_relative_cfg = preference_cfg.get("base_relative", {})
        if not isinstance(base_relative_cfg, dict):
            base_relative_cfg = {}
        base_relative_enabled = bool(base_relative_cfg.get("enabled", False))
        if base_relative_enabled and round_index is not None:
            start_round = int(base_relative_cfg.get("start_round", 1))
            end_round_raw = base_relative_cfg.get("end_round")
            end_round = int(end_round_raw) if end_round_raw is not None else None
            if round_index < start_round or (end_round is not None and round_index > end_round):
                base_relative_enabled = False
        reference_roles = _coerce_str_list(
            base_relative_cfg.get("reference_roles", ["reference:base", "opponent:base", "opponent:initial"])
        )
        policy_roles = _coerce_str_list(base_relative_cfg.get("policy_roles", ["policy"]))
        base_pair_score_mode = str(base_relative_cfg.get("score_mode", "conservative"))
        min_policy_over_reference_margin = float(
            base_relative_cfg.get("min_policy_over_reference_margin", min_score_margin)
        )
        min_reference_over_policy_margin = float(
            base_relative_cfg.get("min_reference_over_policy_margin", min_score_margin)
        )
        min_policy_over_reference_margin_by_edit_type = _coerce_float_map(
            base_relative_cfg.get("min_policy_over_reference_margin_by_edit_type")
        )
        min_reference_over_policy_margin_by_edit_type = _coerce_float_map(
            base_relative_cfg.get("min_reference_over_policy_margin_by_edit_type")
        )
        min_reference_score = float(base_relative_cfg.get("min_reference_score", min_chosen_reward))
        train_reference_wins = bool(base_relative_cfg.get("train_reference_wins", True))
        skip_ambiguous_base_pairs = bool(base_relative_cfg.get("skip_ambiguous", True))
        fallback_to_self_pairs = bool(base_relative_cfg.get("fallback_to_self_pairs", False))
        fallback_to_self_pairs_by_edit_type = _coerce_bool_map(
            base_relative_cfg.get("fallback_to_self_pairs_by_edit_type")
        )
        skip_ambiguous_by_edit_type = _coerce_bool_map(base_relative_cfg.get("skip_ambiguous_by_edit_type"))
        reference_win_weight_scale = float(base_relative_cfg.get("reference_win_weight_scale", 0.75))
        reference_win_weight_scale_by_edit_type = _coerce_float_map(
            base_relative_cfg.get("reference_win_weight_scale_by_edit_type")
        )
        reference_win_preference_sft_weight = float(
            base_relative_cfg.get("reference_win_preference_sft_weight", 0.0)
        )
        reference_min_components = base_relative_cfg.get("min_reference_component_scores", {})
        if not isinstance(reference_min_components, dict):
            reference_min_components = {}
        vlm_pair_guard_cfg = preference_cfg.get("vlm_pair_guard", {})
        if not isinstance(vlm_pair_guard_cfg, dict):
            vlm_pair_guard_cfg = {}
        pareto_pair_guard_cfg = preference_cfg.get("pareto_pair_guard", {})
        if not isinstance(pareto_pair_guard_cfg, dict):
            pareto_pair_guard_cfg = {}
        productive_group_filter_cfg = preference_cfg.get("productive_group_filter", {})
        if not isinstance(productive_group_filter_cfg, dict):
            productive_group_filter_cfg = {}
        positive_success_filter_cfg = preference_cfg.get("positive_success_filter", productive_group_filter_cfg)
        if not isinstance(positive_success_filter_cfg, dict):
            positive_success_filter_cfg = productive_group_filter_cfg
        success_consistency_cfg = preference_cfg.get("success_consistency_filter", {})
        if not isinstance(success_consistency_cfg, dict):
            success_consistency_cfg = {}
        accept_strict_vlm_success_as_positive = bool(
            preference_cfg.get("accept_strict_vlm_success_as_positive", True)
        )
        strict_vlm_positive_statuses = set(
            _coerce_str_list(preference_cfg.get("strict_vlm_positive_statuses", ["accepted"]))
        )
        strict_vlm_positive_min_effective_reward = _finite_float(
            preference_cfg.get("strict_vlm_positive_min_effective_reward", min_chosen_reward),
            math.nan,
        )
        strict_vlm_positive_min_raw_reward = _finite_float(
            preference_cfg.get("strict_vlm_positive_min_raw_reward"),
            math.nan,
        )
        require_success_filter_for_accepted = bool(
            preference_cfg.get(
                "require_success_filter_for_accepted",
                bool(positive_success_filter_cfg.get("enabled", False)),
            )
        )

        def near_miss_candidate_filter_reason(row: dict[str, Any]) -> str | None:
            if (
                math.isfinite(near_miss_min_effective_reward)
                and self._payload_effective_reward(row) < near_miss_min_effective_reward
            ):
                return "near_miss_effective_reward_below_threshold"
            row_edit_type = self._payload_edit_type(row)
            contract_reason = self._training_contract_filter_reason(
                row,
                {"contract_filter": near_miss_contract_cfg},
                row_edit_type,
            )
            if contract_reason is not None:
                return f"near_miss_{contract_reason}"
            raw_reward = self._payload_raw_reward(row)
            semantic_edit = self._payload_component_score(
                row,
                "cepr_semantic_edit",
                "cepr_edit_specificity",
                "rubric_edit_success",
            )
            preservation_terms = [
                self._payload_component_score(row, "cepr_preservation", default=math.nan),
                self._payload_component_score(row, "rubric_preservation", default=math.nan),
            ]
            validity_terms = [
                self._payload_component_score(row, "cepr_validity", default=math.nan),
                self._payload_component_score(row, "rubric_validity", default=math.nan),
            ]
            preservation = min([value for value in preservation_terms if math.isfinite(value)] or [0.0])
            validity = min([value for value in validity_terms if math.isfinite(value)] or [0.0])
            if raw_reward < near_miss_min_raw_reward:
                return "near_miss_raw_reward_below_threshold"
            if semantic_edit < near_miss_min_semantic_edit:
                return "near_miss_semantic_edit_below_threshold"
            if preservation < near_miss_min_preservation:
                return "near_miss_preservation_below_threshold"
            if validity < near_miss_min_validity:
                return "near_miss_validity_below_threshold"
            return None

        records: list[dict[str, Any]] = []
        skipped: Counter[str] = Counter()
        per_family: Counter[str] = Counter()
        per_failure_tag: Counter[str] = Counter()
        grouped = self._group_rows(candidate_payloads)
        for group_id, rows in grouped.items():
            image_rows = [
                row
                for row in rows
                if row.get("edited_image_path") and row.get("image_path") and self._has_evaluation(row)
            ]
            if len(image_rows) < 2:
                skipped["too_few_images"] += 1
                continue

            (
                productive_group_ok,
                productive_group_telemetry,
                productive_group_skip_reason,
            ) = self._preference_group_productivity_decision(image_rows, productive_group_filter_cfg)
            productive_group_hard_filter = bool(productive_group_filter_cfg.get("hard_filter", True))
            productive_group_hard_filter_self_pairs_only = bool(
                productive_group_filter_cfg.get("hard_filter_self_pairs_only", False)
            )
            productive_group_reason = str(productive_group_skip_reason or "productive_group_filtered")
            productive_group_source_noop_bypass = (
                source_noop_negative_enabled
                and source_noop_allow_unproductive
                and productive_group_reason in source_noop_unproductive_reasons
            )
            productive_group_hard_negative_bypass = (
                hard_negative_allow_unproductive
                and productive_group_reason in hard_negative_unproductive_reasons
            )
            if (
                not productive_group_ok
                and productive_group_hard_filter
                and not productive_group_hard_filter_self_pairs_only
            ):
                skipped[str(productive_group_skip_reason or "productive_group_filtered")] += 1
                continue
            if not productive_group_ok and not productive_group_hard_filter:
                skipped[f"soft_{productive_group_skip_reason or 'productive_group_filtered'}"] += 1

            reference_rows: list[dict[str, Any]] = []
            policy_rows: list[dict[str, Any]] = []
            chosen_pool_rows: list[dict[str, Any]]
            group_edit_type = self._dominant_payload_edit_type(image_rows)
            positive_roles = _coerce_str_list(base_relative_cfg.get("positive_roles"))
            group_min_policy_over_reference_margin = min_policy_over_reference_margin_by_edit_type.get(
                group_edit_type,
                min_policy_over_reference_margin,
            )
            group_min_reference_over_policy_margin = min_reference_over_policy_margin_by_edit_type.get(
                group_edit_type,
                min_reference_over_policy_margin,
            )
            group_reference_win_weight_scale = reference_win_weight_scale_by_edit_type.get(
                group_edit_type,
                reference_win_weight_scale,
            )
            if base_relative_enabled:
                group_fallback_to_self_pairs = (
                    fallback_to_self_pairs or fallback_to_self_pairs_by_edit_type.get(group_edit_type, False)
                )
                group_skip_ambiguous_base_pairs = skip_ambiguous_by_edit_type.get(
                    group_edit_type,
                    skip_ambiguous_base_pairs,
                )
                if not positive_roles:
                    positive_roles = list(policy_roles)
                positive_pool_rows: list[dict[str, Any]] = []
                for row in image_rows:
                    role = self._candidate_role(row)
                    if self._role_matches(role, reference_roles):
                        reference_rows.append(row)
                        continue
                    if self._role_matches(role, policy_roles):
                        policy_rows.append(row)
                    if self._role_matches(role, positive_roles):
                        positive_pool_rows.append(row)
                if not policy_rows:
                    skipped["base_relative_no_policy_candidate"] += 1
                    if not group_fallback_to_self_pairs:
                        continue
                chosen_pool_rows = positive_pool_rows if positive_pool_rows else (policy_rows if policy_rows else image_rows)
            else:
                chosen_pool_rows = image_rows
            group_edit_type = self._dominant_payload_edit_type(chosen_pool_rows) or group_edit_type
            group_min_policy_over_reference_margin = min_policy_over_reference_margin_by_edit_type.get(
                group_edit_type,
                min_policy_over_reference_margin,
            )
            group_min_reference_over_policy_margin = min_reference_over_policy_margin_by_edit_type.get(
                group_edit_type,
                min_reference_over_policy_margin,
            )
            group_reference_win_weight_scale = reference_win_weight_scale_by_edit_type.get(
                group_edit_type,
                reference_win_weight_scale,
            )

            accepted_rows = []
            verified_rejected_rows = []
            verified_rejected_ids: set[int] = set()
            if include_near_miss and verified_rejected_near_miss_enabled:
                for row in chosen_pool_rows:
                    row_index = int(row.get("candidate_index", -1))
                    verified_reason = self._verified_rejected_near_miss_reason(
                        row,
                        group_edit_type,
                        verified_rejected_near_miss_cfg,
                    )
                    if verified_reason is None:
                        verified_rejected_rows.append(row)
                        verified_rejected_ids.add(row_index)
                    elif str(row.get("status") or "") == "rejected":
                        skipped[verified_reason] += 1
            for row in chosen_pool_rows:
                ranker_positive = (
                    row.get("status") == "accepted"
                    and self._payload_effective_reward(row) >= min_chosen_reward
                )
                row_strict_success = self._payload_passes_strict_success_filter(
                    row,
                    positive_success_filter_cfg,
                )
                if ranker_positive and require_success_filter_for_accepted and not row_strict_success:
                    skipped["accepted_positive_success_filter_failed"] += 1
                    ranker_positive = False
                strict_vlm_positive = False
                if accept_strict_vlm_success_as_positive and row_strict_success:
                    row_status = str(row.get("status") or "")
                    if strict_vlm_positive_statuses and row_status not in strict_vlm_positive_statuses:
                        skipped["strict_vlm_positive_status_filtered"] += 1
                    elif (
                        math.isfinite(strict_vlm_positive_min_effective_reward)
                        and self._payload_effective_reward(row) < strict_vlm_positive_min_effective_reward
                    ):
                        skipped["strict_vlm_positive_effective_reward_below_threshold"] += 1
                    elif (
                        math.isfinite(strict_vlm_positive_min_raw_reward)
                        and self._payload_raw_reward(row) < strict_vlm_positive_min_raw_reward
                    ):
                        skipped["strict_vlm_positive_raw_reward_below_threshold"] += 1
                    else:
                        strict_vlm_positive = True
                if ranker_positive or strict_vlm_positive:
                    accepted_rows.append(row)
            chosen_is_near_miss = False
            provisional_no_positive_policy = False
            verified_rejected_positive_rows: list[dict[str, Any]] = []
            if (
                include_near_miss
                and verified_rejected_compete_with_accepted
                and verified_rejected_rows
            ):
                anchor_rows = accepted_rows
                if verified_rejected_anchor_enabled:
                    anchor_rows = accepted_rows + verified_rejected_rows
                anchor_reason = self._near_miss_positive_anchor_reason(
                    anchor_rows,
                    group_edit_type,
                    near_miss_positive_anchor_cfg,
                )
                if anchor_reason is not None:
                    skipped[anchor_reason] += 1
                else:
                    for row in verified_rejected_rows:
                        filter_reason = near_miss_candidate_filter_reason(row)
                        if filter_reason is not None:
                            skipped[filter_reason] += 1
                            continue
                        verified_rejected_positive_rows.append(row)
            if accepted_rows or verified_rejected_positive_rows:
                accepted_rows.sort(
                    key=lambda row: (
                        self._payload_pair_score(row, pair_score_mode),
                        self._payload_effective_reward(row),
                        self._payload_raw_reward(row),
                    ),
                    reverse=True,
                )
                verified_rejected_positive_rows.sort(
                    key=lambda row: (
                        self._payload_pair_score(row, pair_score_mode),
                        self._payload_raw_reward(row),
                        self._payload_effective_reward(row),
                    ),
                    reverse=True,
                )
                best_accepted = accepted_rows[0] if accepted_rows else None
                best_verified_rejected = (
                    verified_rejected_positive_rows[0] if verified_rejected_positive_rows else None
                )
                choose_verified_rejected = False
                if best_verified_rejected is not None and best_accepted is None:
                    choose_verified_rejected = True
                elif best_verified_rejected is not None and best_accepted is not None:
                    verified_score = self._payload_pair_score(best_verified_rejected, pair_score_mode)
                    accepted_score = self._payload_pair_score(best_accepted, pair_score_mode)
                    choose_verified_rejected = (
                        verified_score >= accepted_score + verified_rejected_prefer_margin
                    )

                if choose_verified_rejected:
                    chosen = best_verified_rejected
                    chosen["verified_rejected_near_miss"] = True
                    chosen_is_near_miss = True
                    chosen_pair_score = self._payload_pair_score(chosen, pair_score_mode)
                    pair_min_margin = near_miss_min_score_margin
                    pair_weight_scale = near_miss_weight_scale
                    pair_preference_sft_weight = near_miss_preference_sft_weight
                    preference_source = "self_evolve_internal_vlm_verified_near_miss"
                else:
                    chosen = best_accepted
                    chosen_pair_score = self._payload_pair_score(chosen, pair_score_mode)
                    pair_min_margin = min_score_margin
                    pair_weight_scale = 1.0
                    pair_preference_sft_weight = accepted_preference_sft_weight
                    preference_source = (
                        "self_evolve_strict_vlm_positive"
                        if chosen.get("status") != "accepted"
                        else "self_evolve_internal_cepr"
                    )
            elif include_near_miss:
                near_miss_rows = []
                anchor_rows = accepted_rows
                if verified_rejected_anchor_enabled and verified_rejected_rows:
                    anchor_rows = accepted_rows + verified_rejected_rows
                anchor_reason = self._near_miss_positive_anchor_reason(
                    anchor_rows,
                    group_edit_type,
                    near_miss_positive_anchor_cfg,
                )
                if anchor_reason is not None:
                    skipped[anchor_reason] += 1
                else:
                    for row in chosen_pool_rows:
                        row_status = str(row.get("status") or "")
                        row_is_verified_rejected = (
                            int(row.get("candidate_index", -1)) in verified_rejected_ids
                        )
                        if (
                            near_miss_allowed_statuses
                            and row_status not in near_miss_allowed_statuses
                            and not row_is_verified_rejected
                        ):
                            skipped["near_miss_status_filtered"] += 1
                            continue
                        if row_status == "rejected" and not row_is_verified_rejected:
                            skipped["near_miss_rejected_not_verified"] += 1
                            continue
                        filter_reason = near_miss_candidate_filter_reason(row)
                        if filter_reason is not None:
                            skipped[filter_reason] += 1
                            continue
                        near_miss_rows.append(row)
                if near_miss_rows:
                    near_miss_rows.sort(
                        key=lambda row: (
                            self._payload_pair_score(row, pair_score_mode),
                            self._payload_raw_reward(row),
                            self._payload_effective_reward(row),
                        ),
                        reverse=True,
                    )
                    chosen = near_miss_rows[0]
                    chosen_is_near_miss = True
                    chosen_pair_score = self._payload_pair_score(chosen, pair_score_mode)
                    pair_min_margin = near_miss_min_score_margin
                    pair_weight_scale = near_miss_weight_scale
                    pair_preference_sft_weight = near_miss_preference_sft_weight
                    preference_source = "self_evolve_internal_cepr_near_miss"
                    if int(chosen.get("candidate_index", -1)) in verified_rejected_ids:
                        chosen["verified_rejected_near_miss"] = True
                        preference_source = "self_evolve_internal_vlm_verified_near_miss"
                elif base_relative_enabled and train_reference_wins and chosen_pool_rows and reference_rows:
                    chosen = max(
                        chosen_pool_rows,
                        key=lambda row: self._payload_pair_score(row, base_pair_score_mode),
                    )
                    provisional_no_positive_policy = True
                    chosen_is_near_miss = True
                    chosen_pair_score = self._payload_pair_score(chosen, base_pair_score_mode)
                    pair_min_margin = min_reference_over_policy_margin
                    pair_weight_scale = reference_win_weight_scale
                    pair_preference_sft_weight = 0.0
                    preference_source = "base_reference_no_harm_probe"
                else:
                    skipped["no_near_miss_chosen"] += 1
                    continue
            else:
                if base_relative_enabled and train_reference_wins and chosen_pool_rows and reference_rows:
                    chosen = max(
                        chosen_pool_rows,
                        key=lambda row: self._payload_pair_score(row, base_pair_score_mode),
                    )
                    provisional_no_positive_policy = True
                    chosen_is_near_miss = True
                    chosen_pair_score = self._payload_pair_score(chosen, base_pair_score_mode)
                    pair_min_margin = min_reference_over_policy_margin
                    pair_weight_scale = reference_win_weight_scale
                    pair_preference_sft_weight = 0.0
                    preference_source = "base_reference_no_harm_probe"
                else:
                    skipped["no_accepted_chosen"] += 1
                    continue

            if provisional_no_positive_policy and not reference_rows:
                skipped["no_accepted_chosen"] += 1
                continue

            base_relative_decision: dict[str, Any] = {"enabled": base_relative_enabled, "decision": "self_pair"}
            forced_rejected_rows: list[dict[str, Any]] = []
            source_noop_only_rejected_rows = False
            if base_relative_enabled:
                def allow_ambiguous_positive_source_noop() -> bool:
                    if not source_noop_negative_enabled or provisional_no_positive_policy:
                        return False
                    allowed = (
                        source_noop_allow_ambiguous_positive
                        or source_noop_allow_ambiguous_positive_by_edit_type.get(group_edit_type, False)
                    )
                    if not allowed:
                        return False
                    if not self._role_matches(self._candidate_role(chosen), policy_roles):
                        return False
                    return self._payload_passes_strict_success_filter(chosen, positive_success_filter_cfg)

                eligible_reference_rows = []
                reference_floor_observed: dict[str, dict[str, float]] = {}
                for reference_row in reference_rows:
                    reference_score = self._payload_pair_score(reference_row, base_pair_score_mode)
                    if reference_score < min_reference_score:
                        continue
                    passes_floor, observed, floor_reason = self._payload_passes_preference_floors(
                        reference_row,
                        reference_min_components,
                    )
                    reference_floor_observed[str(reference_row.get("candidate_index"))] = observed
                    if not passes_floor:
                        skipped[f"base_reference_{floor_reason}"] += 1
                        continue
                    eligible_reference_rows.append(reference_row)

                if eligible_reference_rows:
                    eligible_reference_rows.sort(
                        key=lambda row: self._payload_pair_score(row, base_pair_score_mode),
                        reverse=True,
                    )
                    best_reference = eligible_reference_rows[0]
                    policy_score = self._payload_pair_score(chosen, base_pair_score_mode)
                    reference_score = self._payload_pair_score(best_reference, base_pair_score_mode)
                    policy_margin = policy_score - reference_score
                    base_relative_decision = {
                        "enabled": True,
                        "score_mode": base_pair_score_mode,
                        "decision": "ambiguous",
                        "policy_candidate_index": int(chosen.get("candidate_index", 0)),
                        "reference_candidate_index": int(best_reference.get("candidate_index", 0)),
                        "policy_role": self._candidate_role(chosen),
                        "reference_role": self._candidate_role(best_reference),
                        "policy_pair_score": round(float(policy_score), 6),
                        "reference_pair_score": round(float(reference_score), 6),
                        "policy_over_reference_margin": round(float(policy_margin), 6),
                        "min_policy_over_reference_margin": group_min_policy_over_reference_margin,
                        "min_reference_over_policy_margin": group_min_reference_over_policy_margin,
                        "reference_win_weight_scale": group_reference_win_weight_scale,
                        "reference_floor_observed": reference_floor_observed.get(
                            str(best_reference.get("candidate_index")),
                            {},
                        ),
                    }
                    if policy_margin >= group_min_policy_over_reference_margin and not provisional_no_positive_policy:
                        chosen_pair_score = policy_score
                        forced_rejected_rows = [best_reference]
                        preference_source = (
                            "self_evolve_policy_beats_base_reference"
                            if self._role_matches(self._candidate_role(chosen), policy_roles)
                            else "self_evolve_self_play_beats_base_reference"
                        )
                        pair_min_margin = group_min_policy_over_reference_margin
                        base_relative_decision["decision"] = "policy_beats_reference"
                    elif policy_margin >= group_min_policy_over_reference_margin and provisional_no_positive_policy:
                        skipped["base_relative_policy_not_positive"] += 1
                        continue
                    elif (
                        train_reference_wins
                        and -policy_margin >= group_min_reference_over_policy_margin
                        and group_fallback_to_self_pairs
                        and not provisional_no_positive_policy
                    ):
                        base_relative_decision["decision"] = "reference_beats_policy_self_pair_fallback"
                        base_relative_decision["self_pair_fallback_reason"] = (
                            "accepted_policy_candidate_kept_for_self_evolution"
                        )
                    elif train_reference_wins and -policy_margin >= group_min_reference_over_policy_margin:
                        forced_rejected_rows = [chosen]
                        chosen = best_reference
                        chosen_is_near_miss = False
                        chosen_pair_score = reference_score
                        pair_min_margin = group_min_reference_over_policy_margin
                        pair_weight_scale = group_reference_win_weight_scale
                        pair_preference_sft_weight = reference_win_preference_sft_weight
                        preference_source = "base_reference_no_harm"
                        base_relative_decision["decision"] = "reference_beats_policy"
                    elif group_skip_ambiguous_base_pairs and not group_fallback_to_self_pairs:
                        if allow_ambiguous_positive_source_noop():
                            base_relative_decision["decision"] = "ambiguous_positive_source_noop"
                            base_relative_decision["fallback_reason"] = (
                                "strict_positive_policy_kept_with_source_noop_negative_only"
                            )
                            source_noop_only_rejected_rows = True
                            preference_source = "self_evolve_ambiguous_positive"
                        else:
                            skipped["base_relative_ambiguous"] += 1
                            continue
                elif not group_fallback_to_self_pairs:
                    base_relative_decision = {
                        "enabled": True,
                        "score_mode": base_pair_score_mode,
                        "decision": "no_eligible_reference",
                        "reference_floor_observed": reference_floor_observed,
                    }
                    if allow_ambiguous_positive_source_noop():
                        base_relative_decision["decision"] = "no_eligible_reference_positive_source_noop"
                        base_relative_decision["fallback_reason"] = (
                            "strict_positive_policy_kept_with_source_noop_negative_only"
                        )
                        source_noop_only_rejected_rows = True
                        preference_source = "self_evolve_ambiguous_positive"
                    else:
                        skipped["base_relative_no_reference_candidate"] += 1
                        continue

            if (
                not productive_group_ok
                and productive_group_hard_filter
                and productive_group_hard_filter_self_pairs_only
            ):
                base_decision = str(base_relative_decision.get("decision") or "")
                if base_decision not in {"policy_beats_reference", "reference_beats_policy"}:
                    productive_group_verified_near_miss_bypass = (
                        bool(chosen.get("verified_rejected_near_miss"))
                        and productive_group_reason in verified_rejected_unproductive_reasons
                    )
                    if (
                        productive_group_source_noop_bypass
                        or productive_group_hard_negative_bypass
                        or productive_group_verified_near_miss_bypass
                    ):
                        productive_group_telemetry = dict(productive_group_telemetry)
                        productive_group_telemetry["hard_filter_bypassed"] = True
                        productive_group_telemetry["hard_filter_bypass_reason"] = (
                            "source_noop_negative"
                            if productive_group_source_noop_bypass
                            else (
                                "verified_rejected_near_miss"
                                if productive_group_verified_near_miss_bypass
                                else "explicit_hard_negative"
                            )
                        )
                        skipped[f"bypassed_{productive_group_reason}"] += 1
                    else:
                        skipped[productive_group_reason] += 1
                        continue

            proposal = chosen.get("proposal", {})
            family = str(proposal.get("family") or "unknown")
            structured_edit = proposal.get("structured_edit", {})
            edit_type = family
            if isinstance(structured_edit, dict):
                edit_type = str(structured_edit.get("edit_type") or family)

            success_consistency_weight_multiplier = 1.0
            success_consistency_telemetry: dict[str, Any] = {"enabled": False}
            if bool(success_consistency_cfg.get("enabled", False)) and not self._role_matches(
                self._candidate_role(chosen),
                reference_roles,
            ):
                consistency_scope = str(success_consistency_cfg.get("scope", "positive")).strip().lower()
                if consistency_scope in {"policy", "policy_only"}:
                    consistency_rows = [
                        row
                        for row in image_rows
                        if self._role_matches(self._candidate_role(row), policy_roles)
                    ]
                elif consistency_scope in {"positive", "positive_roles"}:
                    scoped_positive_roles = positive_roles or policy_roles
                    consistency_rows = [
                        row
                        for row in image_rows
                        if self._role_matches(self._candidate_role(row), scoped_positive_roles)
                    ]
                else:
                    consistency_rows = list(chosen_pool_rows)
                consistency_rows = [
                    row
                    for row in consistency_rows
                    if row.get("edited_image_path") and row.get("image_path") and self._has_evaluation(row)
                ]
                success_rows = [
                    row
                    for row in consistency_rows
                    if self._payload_passes_strict_success_filter(row, positive_success_filter_cfg)
                ]
                min_count_cfg = dict(success_consistency_cfg.get("min_success_count_by_edit_type", {}))
                min_success_count = int(
                    min_count_cfg.get(edit_type, success_consistency_cfg.get("min_success_count", 1))
                )
                min_rate_cfg = dict(success_consistency_cfg.get("min_success_rate_by_edit_type", {}))
                min_success_rate = _finite_float(
                    min_rate_cfg.get(edit_type, success_consistency_cfg.get("min_success_rate")),
                    math.nan,
                )
                candidate_count = len(consistency_rows)
                success_count = len(success_rows)
                success_rate = success_count / max(candidate_count, 1)
                chosen_strict_success = self._payload_passes_strict_success_filter(
                    chosen,
                    positive_success_filter_cfg,
                )
                reject_below_min_count = bool(success_consistency_cfg.get("reject_below_min_count", True))
                reject_below_min_rate = bool(success_consistency_cfg.get("reject_below_min_rate", False))
                weight_power = max(float(success_consistency_cfg.get("weight_power", 1.0)), 1.0e-6)
                min_weight_multiplier = float(success_consistency_cfg.get("min_weight_multiplier", 0.50))
                if candidate_count > 0:
                    support_fraction = min(1.0, success_count / max(min_success_count, 1))
                else:
                    support_fraction = 0.0
                success_consistency_weight_multiplier = _clamp(
                    min_weight_multiplier + (1.0 - min_weight_multiplier) * (support_fraction**weight_power)
                )
                success_consistency_telemetry = {
                    "enabled": True,
                    "scope": consistency_scope,
                    "candidate_count": candidate_count,
                    "success_count": success_count,
                    "success_rate": round(float(success_rate), 6),
                    "min_success_count": min_success_count,
                    "min_success_rate": None if not math.isfinite(min_success_rate) else min_success_rate,
                    "chosen_strict_success": bool(chosen_strict_success),
                    "weight_multiplier": round(float(success_consistency_weight_multiplier), 6),
                }
                if not chosen_strict_success:
                    skipped["success_consistency_chosen_not_strict_success"] += 1
                    continue
                if reject_below_min_count and success_count < min_success_count:
                    skipped["success_consistency_min_count"] += 1
                    continue
                if (
                    reject_below_min_rate
                    and math.isfinite(min_success_rate)
                    and success_rate < min_success_rate
                ):
                    skipped["success_consistency_min_rate"] += 1
                    continue

            (
                generalization_weight_multiplier,
                generalization_confidence,
                generalization_components,
                generalization_skip_reason,
            ) = self._preference_generalization_calibration(chosen, edit_type, calibration_cfg)
            if generalization_skip_reason is not None:
                skipped[generalization_skip_reason] += 1
                continue

            chosen_index = int(chosen.get("candidate_index", 0))
            chosen_reward = self._payload_effective_reward(chosen)
            chosen_raw_reward = self._payload_raw_reward(chosen)
            if source_noop_only_rejected_rows:
                rejected_rows = []
            elif base_relative_enabled and forced_rejected_rows:
                forced_ids = {int(row.get("candidate_index", -1)) for row in forced_rejected_rows}
                rejected_rows = list(forced_rejected_rows)
                rejected_rows.extend(
                    row
                    for row in image_rows
                    if int(row.get("candidate_index", -1)) != chosen_index
                    and int(row.get("candidate_index", -1)) not in forced_ids
                    and row.get("status") != "accepted"
                )
            else:
                rejected_rows = [
                    row
                    for row in image_rows
                    if int(row.get("candidate_index", -1)) != chosen_index and row.get("status") != "accepted"
                ]
            if exclude_rank_only_feasible_rejects:
                forced_ids_for_rank_filter = {
                    int(row.get("candidate_index", -1)) for row in forced_rejected_rows
                }
                filtered_rejected_rows = []
                for row in rejected_rows:
                    if int(row.get("candidate_index", -1)) in forced_ids_for_rank_filter:
                        filtered_rejected_rows.append(row)
                        continue
                    if self._is_rank_only_feasible_reject(row):
                        skipped["rank_only_feasible_reject_filtered"] += 1
                        continue
                    filtered_rejected_rows.append(row)
                rejected_rows = filtered_rejected_rows
            if not rejected_rows and include_ranked_accepted_pairs:
                ranked_accepted_pool = [
                    row
                    for row in chosen_pool_rows
                    if int(row.get("candidate_index", -1)) != chosen_index
                    and row.get("edited_image_path")
                    and self._has_evaluation(row)
                    and row.get("status") == "accepted"
                ]
                ranked_accepted_pool.sort(
                    key=lambda row: (
                        self._payload_pair_score(row, pair_score_mode),
                        self._payload_effective_reward(row),
                        self._payload_raw_reward(row),
                    ),
                    reverse=True,
                )
                rejected_rows = ranked_accepted_pool
                if rejected_rows:
                    preference_source = f"{preference_source}_{ranked_accepted_source_suffix}"
            if source_noop_negative_enabled and (
                not source_noop_only_when_no_candidate_negatives or not rejected_rows
            ):
                productive_group_verified_near_miss_bypass = (
                    bool(chosen.get("verified_rejected_near_miss"))
                    and productive_group_reason in verified_rejected_unproductive_reasons
                )
                if (
                    productive_group_ok
                    or productive_group_source_noop_bypass
                    or productive_group_verified_near_miss_bypass
                    or forced_rejected_rows
                    or source_noop_only_rejected_rows
                ):
                    noop_negative = self._build_source_noop_negative(
                        chosen,
                        edit_type=edit_type,
                        cfg=source_noop_negative_cfg,
                    )
                    if noop_negative is not None:
                        rejected_rows.append(noop_negative)
            rejected_rows.sort(
                key=lambda row: (self._payload_effective_reward(row), self._payload_raw_reward(row)),
                reverse=True,
            )
            if forced_rejected_rows:
                forced_ids = {int(row.get("candidate_index", -1)) for row in forced_rejected_rows}
                forced = [row for row in rejected_rows if int(row.get("candidate_index", -1)) in forced_ids]
                rest = [row for row in rejected_rows if int(row.get("candidate_index", -1)) not in forced_ids]
                rejected_rows = forced + rest
            rejected_rows = self._order_rejected_rows_for_hard_negative_mining(rejected_rows, hard_negative_cfg)
            if productive_group_hard_negative_bypass and not productive_group_ok:
                forced_ids_for_hard_negative = {
                    int(row.get("candidate_index", -1)) for row in forced_rejected_rows
                }
                filtered_hard_negatives = []
                for row in rejected_rows:
                    if int(row.get("candidate_index", -1)) in forced_ids_for_hard_negative:
                        filtered_hard_negatives.append(row)
                        continue
                    if row.get("source_noop_negative"):
                        filtered_hard_negatives.append(row)
                        continue
                    tags = list(row.get("failure_tags") or self._payload_failure_tags(row, hard_negative_cfg))
                    if not hard_negative_require_explicit or hard_negative_explicit_tags.intersection(tags):
                        filtered_hard_negatives.append(row)
                    else:
                        skipped["unproductive_group_nonexplicit_negative_filtered"] += 1
                rejected_rows = filtered_hard_negatives
            if (
                source_noop_negative_enabled
                and source_noop_only_when_no_candidate_negatives
                and not rejected_rows
            ):
                productive_group_verified_near_miss_bypass = (
                    bool(chosen.get("verified_rejected_near_miss"))
                    and productive_group_reason in verified_rejected_unproductive_reasons
                )
                if (
                    productive_group_ok
                    or productive_group_source_noop_bypass
                    or productive_group_verified_near_miss_bypass
                    or forced_rejected_rows
                ):
                    noop_negative = self._build_source_noop_negative(
                        chosen,
                        edit_type=edit_type,
                        cfg=source_noop_negative_cfg,
                    )
                    if noop_negative is not None:
                        rejected_rows.append(noop_negative)
            if forced_rejected_rows:
                forced_ids = {int(row.get("candidate_index", -1)) for row in forced_rejected_rows}
                forced = [row for row in rejected_rows if int(row.get("candidate_index", -1)) in forced_ids]
                rest = [row for row in rejected_rows if int(row.get("candidate_index", -1)) not in forced_ids]
                rejected_rows = forced + rest

            pairs_added = 0
            pair_min_margin_for_group = pair_min_margin
            pair_weight_scale_for_group = pair_weight_scale
            if chosen_is_near_miss and edit_type in near_miss_min_score_margin_by_edit_type:
                pair_min_margin_for_group = near_miss_min_score_margin_by_edit_type[edit_type]
            elif not chosen_is_near_miss and edit_type in min_score_margin_by_edit_type:
                pair_min_margin_for_group = min_score_margin_by_edit_type[edit_type]
            if (
                not chosen_is_near_miss
                and edit_type in accepted_weight_scale_by_edit_type
                and base_relative_decision.get("decision") != "reference_beats_policy"
            ):
                pair_weight_scale_for_group = accepted_weight_scale_by_edit_type[edit_type]
            for rejected in rejected_rows:
                if pairs_added >= max_pairs_per_group:
                    break
                rejected_reward = self._payload_effective_reward(rejected)
                rejected_raw_reward = self._payload_raw_reward(rejected)
                if base_relative_enabled and forced_rejected_rows:
                    rejected_pair_score = self._payload_pair_score(rejected, base_pair_score_mode)
                else:
                    rejected_pair_score = self._payload_pair_score(rejected, pair_score_mode)
                margin = chosen_pair_score - rejected_pair_score
                if margin < pair_min_margin_for_group:
                    skipped["margin_too_small"] += 1
                    continue
                source_image = chosen.get("image_path")
                chosen_image = chosen.get("edited_image_path")
                rejected_image = rejected.get("edited_image_path")
                if not source_image or not chosen_image or not rejected_image:
                    skipped["missing_paths"] += 1
                    continue
                vlm_multiplier, vlm_pair_telemetry, vlm_skip_reason = self._preference_vlm_pair_guard(
                    chosen,
                    rejected,
                    vlm_pair_guard_cfg,
                )
                if vlm_skip_reason is not None:
                    skipped[vlm_skip_reason] += 1
                    continue
                pareto_multiplier, pareto_pair_telemetry, pareto_skip_reason = (
                    self._preference_pareto_pair_guard(
                        chosen,
                        rejected,
                        edit_type=edit_type,
                        guard_cfg=pareto_pair_guard_cfg,
                    )
                )
                if pareto_skip_reason is not None:
                    skipped[pareto_skip_reason] += 1
                    continue
                negative_weight_scale = _finite_float(rejected.get("negative_weight_scale"), 1.0)
                sample_weight = min(
                    max_sample_weight,
                    pair_weight_scale_for_group
                    * generalization_weight_multiplier
                    * success_consistency_weight_multiplier
                    * vlm_multiplier
                    * pareto_multiplier
                    * negative_weight_scale
                    * base_weight
                    * (1.0 + margin_weight_scale * min(max(margin, 0.0), margin_clip)),
                )
                rejected_failure_tags = list(rejected.get("failure_tags") or [])
                record_preference_source = preference_source
                if rejected.get("source_noop_negative"):
                    ranked_suffix = f"_{ranked_accepted_source_suffix}"
                    if record_preference_source.endswith(ranked_suffix):
                        record_preference_source = record_preference_source[: -len(ranked_suffix)]
                    record_preference_source = f"{record_preference_source}_{source_noop_source_suffix}"
                records.append(
                    {
                        "prompt": str(proposal.get("instruction", "")),
                        "chosen_image": chosen_image,
                        "rejected_image": rejected_image,
                        "edit_image": source_image,
                        "sample_weight": round(float(sample_weight), 6),
                        "record_key": chosen.get("record_key"),
                        "group_id": group_id,
                        "family": family,
                        "operation_id": proposal.get("operation_id"),
                        "structured_edit": structured_edit if isinstance(structured_edit, dict) else {},
                        "preference_source": record_preference_source,
                        "preference_sft_weight": round(float(pair_preference_sft_weight), 6),
                        "chosen_is_near_miss": chosen_is_near_miss,
                        "chosen_status": chosen.get("status"),
                        "rejected_status": rejected.get("status"),
                        "chosen_candidate_role": chosen.get("candidate_role", "policy"),
                        "rejected_candidate_role": rejected.get("candidate_role", "policy"),
                        "chosen_model_state": chosen.get("candidate_model_state"),
                        "rejected_model_state": rejected.get("candidate_model_state"),
                        "chosen_candidate_index": chosen_index,
                        "rejected_candidate_index": int(rejected.get("candidate_index", 0)),
                        "chosen_effective_reward": chosen_reward,
                        "rejected_effective_reward": rejected_reward,
                        "chosen_raw_reward": chosen_raw_reward,
                        "rejected_raw_reward": rejected_raw_reward,
                        "chosen_pair_score": chosen_pair_score,
                        "rejected_pair_score": rejected_pair_score,
                        "score_margin": margin,
                        "min_score_margin_used": pair_min_margin_for_group,
                        "pair_weight_scale_used": pair_weight_scale_for_group,
                        "negative_weight_scale": round(float(negative_weight_scale), 6),
                        "generalization_weight_multiplier": round(
                            float(generalization_weight_multiplier),
                            6,
                        ),
                        "generalization_confidence": round(float(generalization_confidence), 6),
                        "generalization_components": generalization_components,
                        "success_consistency": success_consistency_telemetry,
                        "vlm_pair_guard": vlm_pair_telemetry,
                        "pareto_pair_guard": pareto_pair_telemetry,
                        "productive_group": productive_group_telemetry,
                        "base_relative": base_relative_decision,
                        "rejected_failure_tags": rejected_failure_tags,
                    }
                )
                per_family[family] += 1
                for tag in rejected_failure_tags:
                    per_failure_tag[tag] += 1
                pairs_added += 1
            if pairs_added <= 0:
                skipped["no_pairs_added"] += 1

        records, family_balance_summary = self._apply_preference_family_balance(records, preference_cfg)
        per_family = Counter(str(record.get("family") or "unknown") for record in records)
        per_edit_type = Counter(self._record_balance_key(record, "edit_type") for record in records)
        per_failure_tag = Counter(
            tag
            for record in records
            for tag in (record.get("rejected_failure_tags") or [])
        )

        summary = {
            "enabled": True,
            "candidate_groups": len(grouped),
            "pairs": len(records),
            "per_family": dict(per_family),
            "per_edit_type": dict(per_edit_type),
            "per_failure_tag": dict(per_failure_tag),
            "family_balance": family_balance_summary,
            "skipped": dict(skipped),
            "max_pairs_per_group": max_pairs_per_group,
            "score_mode": pair_score_mode,
            "min_score_margin": min_score_margin,
            "min_score_margin_by_edit_type": min_score_margin_by_edit_type,
            "min_chosen_reward": min_chosen_reward,
            "accepted_weight_scale_by_edit_type": accepted_weight_scale_by_edit_type,
            "near_miss_min_score_margin": near_miss_min_score_margin,
            "near_miss_min_score_margin_by_edit_type": near_miss_min_score_margin_by_edit_type,
            "near_miss_allowed_statuses": sorted(near_miss_allowed_statuses),
            "near_miss_min_effective_reward": near_miss_min_effective_reward,
            "near_miss_contract_filter": near_miss_contract_cfg,
            "near_miss_positive_anchor_filter": near_miss_positive_anchor_cfg,
            "verified_rejected_near_miss": verified_rejected_near_miss_cfg,
            "include_ranked_accepted_pairs": include_ranked_accepted_pairs,
            "ranked_accepted_pairs": ranked_accepted_cfg,
            "generalization_calibration": calibration_cfg,
            "hard_negative_mining": hard_negative_cfg,
            "source_noop_negative": source_noop_negative_cfg,
            "exclude_rank_only_feasible_rejects": exclude_rank_only_feasible_rejects,
            "base_relative": base_relative_cfg,
            "vlm_pair_guard": vlm_pair_guard_cfg,
            "pareto_pair_guard": pareto_pair_guard_cfg,
            "productive_group_filter": productive_group_filter_cfg,
            "positive_success_filter": positive_success_filter_cfg,
            "success_consistency_filter": success_consistency_cfg,
            "accept_strict_vlm_success_as_positive": accept_strict_vlm_success_as_positive,
            "strict_vlm_positive_statuses": sorted(strict_vlm_positive_statuses),
            "strict_vlm_positive_min_effective_reward": strict_vlm_positive_min_effective_reward,
            "strict_vlm_positive_min_raw_reward": strict_vlm_positive_min_raw_reward,
        }
        return records, summary

    def _write_manifest_records(
        self,
        training_records: list[dict[str, Any]],
        manifest_path: Path,
    ) -> tuple[Path, int, float]:
        training_cfg = self.config.get("training", {})
        replay_ratio = float(training_cfg.get("reconstruction_replay_ratio", 0.0))
        replay_weight = float(training_cfg.get("reconstruction_replay_weight", 0.50))
        replay_prompt = str(
            training_cfg.get(
                "reconstruction_replay_prompt",
                "Reconstruct the input image exactly. Preserve all content, layout, colors, and text.",
            )
        )
        manifest_records = list(training_records)
        replay_source_paths: list[Path] = []
        for record in training_records:
            source_path = resolve_path(str(record.get("edit_image", "")))
            if source_path is not None:
                replay_source_paths.append(source_path)
        if replay_ratio > 0 and manifest_records and replay_source_paths:
            replay_count = max(1, round(len(manifest_records) * replay_ratio))
            unique_sources = list(dict.fromkeys(replay_source_paths))
            for index in range(replay_count):
                source_path = unique_sources[index % len(unique_sources)]
                manifest_records.append(
                    {
                        "prompt": replay_prompt,
                        "image": relative_to_repo(source_path),
                        "edit_image": relative_to_repo(source_path),
                        "sample_weight": replay_weight,
                        "candidate_status": "reconstruction_replay",
                    }
                )
        save_json(manifest_records, manifest_path)
        write_jsonl(manifest_records, manifest_path.with_suffix(".jsonl"))
        weight_sum = sum(float(record.get("sample_weight", 1.0)) for record in manifest_records)
        return manifest_path, len(manifest_records), weight_sum

    def _build_preference_anchor_replay_records(
        self,
        preference_records: list[dict[str, Any]],
        preference_cfg: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        anchor_cfg = preference_cfg.get("anchor_replay", {})
        if not isinstance(anchor_cfg, dict):
            anchor_cfg = {}
        if not bool(anchor_cfg.get("enabled", False)):
            return [], {"enabled": False, "records": 0}

        ratio = float(anchor_cfg.get("ratio", 0.0))
        if ratio <= 0.0 or not preference_records:
            return [], {"enabled": True, "records": 0, "reason": "empty_or_zero_ratio"}

        max_records = int(anchor_cfg.get("max_records", 0))
        prompt = str(
            anchor_cfg.get(
                "prompt",
                "Reconstruct the input image exactly. Preserve all content, layout, colors, and text.",
            )
        )
        sample_weight = float(anchor_cfg.get("sample_weight", 0.25))
        preference_sft_weight = float(anchor_cfg.get("preference_sft_weight", 0.0))
        max_weight_fraction = float(anchor_cfg.get("max_weight_fraction", 1.0))
        rejected_image_keys = _coerce_str_list(anchor_cfg.get("rejected_image_keys", ["chosen_image", "rejected_image"]))
        if not rejected_image_keys:
            rejected_image_keys = ["chosen_image"]

        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for record in preference_records:
            source_image = str(record.get("edit_image") or "")
            if not source_image:
                continue
            rejected_image = ""
            for key in rejected_image_keys:
                value = str(record.get(key) or "")
                if value and value != source_image:
                    rejected_image = value
                    break
            if not rejected_image:
                continue
            dedupe_key = (source_image, rejected_image)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(
                {
                    "prompt": prompt,
                    "chosen_image": source_image,
                    "rejected_image": rejected_image,
                    "edit_image": source_image,
                    "sample_weight": round(float(sample_weight), 6),
                    "record_key": record.get("record_key"),
                    "group_id": record.get("group_id"),
                    "family": "preservation_anchor",
                    "operation_id": "preservation_anchor_replay",
                    "structured_edit": {
                        "edit_type": "preservation_anchor",
                        "scope": "identity",
                    },
                    "preference_source": "general_preservation_anchor_replay",
                    "preference_sft_weight": round(float(preference_sft_weight), 6),
                    "anchor_replay": True,
                    "anchor_replay_source_family": record.get("family"),
                    "anchor_replay_source_operation_id": record.get("operation_id"),
                }
            )

        replay_count = max(1, round(len(preference_records) * ratio))
        if max_records > 0:
            replay_count = min(replay_count, max_records)
        cap_applied = False
        max_anchor_weight = None
        if 0.0 <= max_weight_fraction < 1.0 and sample_weight > 0.0:
            base_weight = sum(float(record.get("sample_weight", 1.0)) for record in preference_records)
            max_anchor_weight = (max_weight_fraction / max(1.0 - max_weight_fraction, 1.0e-6)) * base_weight
            replay_count = min(replay_count, int(max_anchor_weight // sample_weight))
            cap_applied = True
        replay_count = min(replay_count, len(candidates))
        records = candidates[:replay_count]
        return records, {
            "enabled": True,
            "records": len(records),
            "candidate_records": len(candidates),
            "ratio": ratio,
            "sample_weight": sample_weight,
            "max_weight_fraction": max_weight_fraction,
            "max_anchor_weight": max_anchor_weight,
            "weight_fraction_cap_applied": cap_applied,
            "preference_sft_weight": preference_sft_weight,
            "prompt": prompt,
            "rejected_image_keys": rejected_image_keys,
        }

    def _write_preference_manifest_records(
        self,
        preference_records: list[dict[str, Any]],
        manifest_path: Path,
    ) -> tuple[Path, int, float, dict[str, Any]]:
        preference_cfg = dict(self.config.get("training", {}).get("preference", {}))
        anchor_records, anchor_summary = self._build_preference_anchor_replay_records(
            preference_records,
            preference_cfg,
        )
        manifest_records = list(preference_records) + anchor_records
        save_json(manifest_records, manifest_path)
        write_jsonl(manifest_records, manifest_path.with_suffix(".jsonl"))
        weight_sum = sum(float(record.get("sample_weight", 1.0)) for record in manifest_records)
        return manifest_path, len(manifest_records), weight_sum, anchor_summary

    def _run_training_round(
        self,
        round_index: int,
        round_dir: Path,
        manifest_path: Path,
        *,
        training_mode: str = "sft",
    ) -> dict[str, Any] | None:
        training_cfg = self.config.get("training", {})
        if training_cfg.get("trigger", "emit_only") != "launch":
            return None
        if not manifest_path.exists():
            return None

        base_config_path = training_cfg.get("base_train_config")
        if not base_config_path:
            raise ValueError("training.base_train_config is required when trigger=launch")

        editor_state_before = self.editor.model_state() if isinstance(self.editor, QwenEditEditor) else None
        train_config = load_yaml_config(base_config_path)
        train_overrides = training_cfg.get("train_config_overrides") or training_cfg.get("editor_train_overrides")
        if isinstance(train_overrides, dict):
            train_config = _deep_update(train_config, train_overrides)
        train_training_config = train_config.setdefault("training", {})
        for key in EDITOR_TRAINING_OVERRIDE_KEYS:
            if key in training_cfg:
                train_training_config[key] = training_cfg[key]
        train_config["name"] = f"{train_config['name']}_self_evolve_r{round_index:02d}"
        train_config["dataset"]["dataset_base_path"] = "."
        train_config["dataset"]["dataset_metadata_path"] = relative_to_repo(manifest_path)
        if training_mode == "preference":
            preference_cfg = dict(training_cfg.get("preference", {}))
            train_config["dataset"]["chosen_image_key"] = preference_cfg.get("chosen_image_key", "chosen_image")
            train_config["dataset"]["rejected_image_key"] = preference_cfg.get("rejected_image_key", "rejected_image")
            train_config["dataset"]["condition_image_key"] = preference_cfg.get("condition_image_key", "edit_image")
            train_config["dataset"]["prompt_key"] = preference_cfg.get("prompt_key", "prompt")
            train_training_config["training_objective"] = preference_cfg.get(
                "training_objective",
                training_cfg.get("training_objective", "pairwise_linear_sdpo"),
            )
            train_training_config["preference_beta"] = preference_cfg.get(
                "preference_beta",
                training_cfg.get("preference_beta", 2.0),
            )
            train_training_config["preference_margin"] = preference_cfg.get(
                "preference_margin",
                training_cfg.get("preference_margin", 0.0),
            )
            train_training_config["preference_sft_weight"] = preference_cfg.get(
                "preference_sft_weight",
                training_cfg.get("preference_sft_weight", 0.10),
            )
            train_training_config["preference_reference_mode"] = preference_cfg.get(
                "preference_reference_mode",
                training_cfg.get("preference_reference_mode", "none"),
            )
            train_training_config["preference_sdpo_epsilon"] = preference_cfg.get(
                "preference_sdpo_epsilon",
                training_cfg.get("preference_sdpo_epsilon", 1.0e-12),
            )
        train_config["output"]["output_path"] = relative_to_repo(round_dir / "training_output")
        train_config["output"]["command_file"] = relative_to_repo(round_dir / "training_command.txt")
        train_config["output"]["log_dir"] = relative_to_repo(round_dir / "training_logs")
        output_dir = resolve_path(train_config["output"]["output_path"])
        if output_dir is None:
            raise ValueError("Could not resolve training output path")
        if (
            bool(training_cfg.get("resume_from_latest", True))
            and not train_config.get("training", {}).get("resume_from_checkpoint")
            and discover_latest_step_checkpoint(output_dir) is not None
        ):
            train_config["training"]["resume_from_checkpoint"] = "latest"

        current_checkpoint = training_cfg.get("current_checkpoint_path")
        if current_checkpoint and train_config.get("mode") == "lora":
            lora_config = train_config.setdefault("lora", {})
            if "lora_checkpoint" in lora_config:
                lora_config["lora_checkpoint"] = current_checkpoint
            else:
                lora_config["checkpoint_path"] = current_checkpoint

        command, working_dir = build_train_command(train_config)
        command_path = resolve_path(train_config["output"]["command_file"])
        log_dir = ensure_dir(resolve_path(train_config["output"]["log_dir"]))
        if command_path is None:
            raise ValueError("Could not resolve training command file path")
        ensure_dir(command_path.parent)
        command_path.write_text(shell_join(command) + "\n", encoding="utf-8")
        log_path = log_dir / f"{train_config['name']}_{utc_timestamp()}.log"

        metadata = {
            **base_run_metadata(),
            "type": "self_evolve_training_round",
            "round_index": round_index,
            "training_mode": training_mode,
            "config_path": train_config["_config_path"],
            "command": command,
            "working_dir": str(working_dir),
            "log_path": str(log_path),
            "dry_run": self.dry_run,
            "resume_from_checkpoint": train_config.get("training", {}).get("resume_from_checkpoint"),
        }
        save_json(metadata, round_dir / "training_metadata.json")

        if self.dry_run:
            return {
                "status": "planned",
                "training_mode": training_mode,
                "command_path": str(command_path),
                "log_path": str(log_path),
                "editor_state_before_training": editor_state_before,
                "editor_state_after_training": editor_state_before,
                "continue_with_trained_checkpoint": bool(training_cfg.get("continue_with_trained_checkpoint", True)),
                "trained_model_type": str(train_config.get("mode", "lora")),
                "trained_checkpoint_backend": training_checkpoint_backend(training_cfg, train_config),
            }

        self._release_resident_models("editor training subprocess")
        return_code = run_and_tee(command, cwd=working_dir, log_path=log_path)
        if return_code != 0:
            raise SystemExit(return_code)

        latest_checkpoint = discover_latest_checkpoint(output_dir)
        editor_state_after = editor_state_before
        if latest_checkpoint is not None:
            training_cfg["current_checkpoint_path"] = str(latest_checkpoint)
            if isinstance(self.editor, QwenEditEditor):
                continue_with_checkpoint = bool(training_cfg.get("continue_with_trained_checkpoint", True))
                trained_checkpoint_backend = training_checkpoint_backend(training_cfg, train_config)
                trained_model_type = str(train_config.get("mode", "lora"))
                if continue_with_checkpoint:
                    self.editor.set_model_checkpoint(
                        str(latest_checkpoint),
                        model_type=trained_model_type,
                        backend=trained_checkpoint_backend,
                    )
                editor_state_after = self.editor.model_state()
        return {
            "status": "completed",
            "training_mode": training_mode,
            "command_path": str(command_path),
            "log_path": str(log_path),
            "output_dir": str(output_dir),
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
            "editor_state_before_training": editor_state_before,
            "editor_state_after_training": editor_state_after,
            "continue_with_trained_checkpoint": bool(training_cfg.get("continue_with_trained_checkpoint", True)),
            "trained_model_type": str(train_config.get("mode", "lora")),
            "trained_checkpoint_backend": training_checkpoint_backend(training_cfg, train_config),
        }

    def _run_proposer_training_round(
        self,
        round_index: int,
        round_dir: Path,
        candidate_payloads: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        proposer_cfg = self.config.get("proposer", {})
        training_cfg = proposer_cfg.get("training", {})
        raw_trigger = training_cfg.get("trigger", "none")
        trigger = "none" if raw_trigger is None else str(raw_trigger).strip().lower()
        if trigger in {"none", "disabled", "false"}:
            return None

        data_cfg = training_cfg.get("data", {})
        records, data_summary = build_proposer_training_records(
            candidate_payloads,
            reward_center=float(data_cfg.get("reward_center", 0.50)),
            reward_sigma=float(data_cfg.get("reward_sigma", 0.25)),
            min_reward=float(data_cfg.get("min_reward", 0.35)),
            min_quality=float(data_cfg.get("min_quality", 0.30)),
            allowed_edit_types=set(
                data_cfg.get("allowed_edit_types")
                or proposer_cfg.get("allowed_edit_types")
                or proposer_cfg.get("focus_edit_types")
                or []
            ),
            disallowed_edit_types=set(
                data_cfg.get("disallowed_edit_types")
                or proposer_cfg.get("disallowed_edit_types")
                or proposer_cfg.get("avoid_edit_types")
                or []
            ),
            reference_roles=set(data_cfg.get("reference_roles") or ["reference:base"]),
            policy_roles=set(data_cfg.get("policy_roles") or ["policy"]),
            base_improvement_weight=float(data_cfg.get("base_improvement_weight", 0.0)),
            base_harm_penalty=float(data_cfg.get("base_harm_penalty", 0.0)),
            base_margin_center=float(data_cfg.get("base_margin_center", 0.03)),
            min_judge_score=float(data_cfg.get("min_judge_score", 0.55)),
            min_judge_semantic=float(data_cfg.get("min_judge_semantic", 0.55)),
            min_judge_preservation=float(data_cfg.get("min_judge_preservation", 0.55)),
            min_judge_artifact_free=float(data_cfg.get("min_judge_artifact_free", 0.55)),
            require_judge_for_success=bool(data_cfg.get("require_judge_for_success", True)),
            all_fail_penalty=float(data_cfg.get("all_fail_penalty", 0.35)),
            all_pass_penalty=float(data_cfg.get("all_pass_penalty", 0.75)),
            all_pass_threshold=float(data_cfg.get("all_pass_threshold", 0.95)),
            require_productive_band_for_sft=bool(data_cfg.get("require_productive_band_for_sft", False)),
            min_success_rate_for_sft=float(data_cfg.get("min_success_rate_for_sft", 0.25)),
            max_success_rate_for_sft=float(data_cfg.get("max_success_rate_for_sft", 0.75)),
            require_accepted_or_improved_for_sft=bool(
                data_cfg.get("require_accepted_or_improved_for_sft", False)
            ),
            min_policy_over_reference_margin_for_sft=float(
                data_cfg.get("min_policy_over_reference_margin_for_sft", 0.04)
            ),
        )
        train_jsonl = write_proposer_training_jsonl(records, round_dir / "proposer_training.jsonl")
        effective_train_jsonl = train_jsonl
        effective_records = records
        selected_for_training = int(data_summary["selected_for_sft"])
        if bool(data_cfg.get("use_cumulative_manifest", False)):
            cumulative_records: list[dict[str, Any]] = []
            selected_only = bool(data_cfg.get("cumulative_selected_only", True))
            for previous_path in sorted(round_dir.parent.glob("round_*/proposer_training.jsonl")):
                if previous_path.resolve() == train_jsonl.resolve():
                    continue
                with previous_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if selected_only and not bool(record.get("use_for_sft", False)):
                            continue
                        cumulative_records.append(record)
            cumulative_records.extend(records)
            max_cumulative_records = int(data_cfg.get("max_cumulative_records", 0) or 0)
            if max_cumulative_records > 0 and len(cumulative_records) > max_cumulative_records:
                cumulative_records = cumulative_records[-max_cumulative_records:]
            effective_train_jsonl = write_proposer_training_jsonl(
                cumulative_records,
                round_dir / "proposer_training_cumulative.jsonl",
            )
            effective_records = cumulative_records
            selected_for_training = sum(1 for record in cumulative_records if bool(record.get("use_for_sft", False)))
            data_summary["cumulative_records"] = len(cumulative_records)
            data_summary["cumulative_selected_for_sft"] = selected_for_training
            data_summary["effective_train_jsonl"] = str(effective_train_jsonl)
        selected_records = [record for record in effective_records if bool(record.get("use_for_sft", False))]
        selected_edit_type_counts = Counter(
            str((record.get("target") or {}).get("edit_type") or "unknown")
            for record in selected_records
        )
        min_selected_records = int(
            data_cfg.get(
                "min_selected_records_for_training",
                data_cfg.get("min_cumulative_selected_records_for_training", 1),
            )
        )
        min_selected_edit_types = int(data_cfg.get("min_selected_edit_types_for_training", 1))
        data_summary["selected_edit_type_counts"] = dict(selected_edit_type_counts)
        data_summary["selected_edit_type_count"] = len(selected_edit_type_counts)
        data_summary["min_selected_records_for_training"] = min_selected_records
        data_summary["min_selected_edit_types_for_training"] = min_selected_edit_types
        save_json(data_summary, round_dir / "proposer_training_summary.json")
        if selected_for_training <= 0:
            return {
                "status": "skipped_no_positive_records",
                "train_jsonl": str(effective_train_jsonl),
                "summary": data_summary,
            }
        if selected_for_training < min_selected_records:
            return {
                "status": "skipped_min_selected_records",
                "train_jsonl": str(effective_train_jsonl),
                "selected_for_training": selected_for_training,
                "min_selected_records": min_selected_records,
                "summary": data_summary,
            }
        if len(selected_edit_type_counts) < min_selected_edit_types:
            return {
                "status": "skipped_min_selected_edit_types",
                "train_jsonl": str(effective_train_jsonl),
                "selected_edit_type_counts": dict(selected_edit_type_counts),
                "min_selected_edit_types": min_selected_edit_types,
                "summary": data_summary,
            }

        output_dir = resolve_path(
            str(training_cfg.get("output_dir") or round_dir / "proposer_training_output")
        )
        if output_dir is None:
            raise ValueError("proposer.training.output_dir could not be resolved")
        command_path = resolve_path(
            str(training_cfg.get("command_file") or round_dir / "proposer_training_command.txt")
        )
        if command_path is None:
            raise ValueError("proposer.training.command_file could not be resolved")
        log_dir = ensure_dir(resolve_path(str(training_cfg.get("log_dir") or round_dir / "proposer_training_logs")))
        train_script = resolve_path(
            str(training_cfg.get("train_script", "src/qwen_edit_project/train/train_proposer_lora.py"))
        )
        if train_script is None:
            raise ValueError("proposer.training.train_script could not be resolved")

        command = [*build_accelerate_command(training_cfg.get("accelerate_executable")), "launch"]
        accelerate_config = training_cfg.get("accelerate_config_file")
        if accelerate_config:
            command.extend(["--config_file", str(resolve_path(str(accelerate_config)) or accelerate_config)])
        command.append(str(train_script))
        _append_flag(command, "--model_name_or_path", proposer_cfg.get("model_name_or_path"))
        _append_flag(command, "--model_subfolder", proposer_cfg.get("model_subfolder"))
        _append_flag(command, "--processor_subfolder", proposer_cfg.get("processor_subfolder"))
        _append_flag(command, "--model_class", proposer_cfg.get("model_class"))
        _append_flag(command, "--train_jsonl", effective_train_jsonl)
        _append_flag(command, "--dataset_base_path", ".")
        _append_flag(command, "--output_dir", output_dir)
        _append_flag(command, "--checkpoint_path", proposer_cfg.get("checkpoint_path"))
        _append_flag(command, "--local_files_only", proposer_cfg.get("local_files_only", False))
        _append_flag(command, "--torch_dtype", proposer_cfg.get("torch_dtype", "auto"))
        _append_flag(command, "--mixed_precision", training_cfg.get("mixed_precision", "bf16"))
        _append_flag(command, "--learning_rate", training_cfg.get("learning_rate", 1e-5))
        _append_flag(command, "--num_train_epochs", training_cfg.get("num_train_epochs", 1))
        _append_flag(command, "--max_train_steps", training_cfg.get("max_train_steps"))
        _append_flag(command, "--train_batch_size", training_cfg.get("train_batch_size", 1))
        _append_flag(command, "--gradient_accumulation_steps", training_cfg.get("gradient_accumulation_steps", 4))
        _append_flag(command, "--max_grad_norm", training_cfg.get("max_grad_norm", 1.0))
        _append_flag(command, "--weight_decay", training_cfg.get("weight_decay", 0.01))
        _append_flag(command, "--seed", training_cfg.get("seed", self.config.get("runtime", {}).get("seed", 123)))
        _append_flag(command, "--min_reward", data_cfg.get("min_reward", 0.35))
        _append_flag(command, "--lora_rank", training_cfg.get("lora_rank", 16))
        _append_flag(command, "--lora_alpha", training_cfg.get("lora_alpha", 32))
        _append_flag(command, "--lora_dropout", training_cfg.get("lora_dropout", 0.05))
        _append_flag(command, "--lora_target_modules", training_cfg.get("lora_target_modules"))
        _append_flag(command, "--gradient_checkpointing", training_cfg.get("gradient_checkpointing", True))
        _append_flag(command, "--checkpointing_steps", training_cfg.get("checkpointing_steps", 0))
        _append_flag(command, "--checkpoints_total_limit", training_cfg.get("checkpoints_total_limit", 5))
        _append_flag(command, "--logging_steps", training_cfg.get("logging_steps", 10))
        _append_flag(command, "--reward_weighting", training_cfg.get("reward_weighting", False))
        _append_flag(command, "--min_reward_weight", training_cfg.get("min_reward_weight", 0.25))
        _append_flag(command, "--max_reward_weight", training_cfg.get("max_reward_weight", 1.50))
        if bool(training_cfg.get("resume_from_latest", True)) and discover_latest_step_checkpoint(output_dir) is not None:
            _append_flag(command, "--resume_from_checkpoint", "latest")

        ensure_dir(command_path.parent)
        command_path.write_text(shell_join(command) + "\n", encoding="utf-8")
        log_path = log_dir / f"proposer_r{round_index:02d}_{utc_timestamp()}.log"
        if self.dry_run or trigger == "emit_only":
            return {
                "status": "planned",
                "train_jsonl": str(effective_train_jsonl),
                "command_path": str(command_path),
                "log_path": str(log_path),
                "output_dir": str(output_dir),
                "summary": data_summary,
            }
        if trigger != "launch":
            raise ValueError(f"Unsupported proposer.training.trigger: {trigger}")
        self._release_resident_models("proposer training subprocess")
        return_code = run_and_tee(command, cwd=resolve_path(".") or Path("."), log_path=log_path)
        if return_code != 0:
            raise SystemExit(return_code)
        proposer_cfg["checkpoint_path"] = str(output_dir)
        if hasattr(self.proposer, "set_checkpoint_path"):
            self.proposer.set_checkpoint_path(str(output_dir))
        return {
            "status": "completed",
            "train_jsonl": str(train_jsonl),
            "command_path": str(command_path),
            "log_path": str(log_path),
            "output_dir": str(output_dir),
            "summary": data_summary,
        }

    def run(self) -> dict[str, Any]:
        runtime = self.config.get("runtime", {})
        seed = int(runtime.get("seed", 123))
        curriculum = self.config["curriculum"]
        proposals_per_image = int(curriculum.get("proposals_per_image", 1))
        candidate_generation = self.config.get("candidate_generation", {})
        samples_per_proposal = int(candidate_generation.get("samples_per_proposal", 1))
        candidate_seed_stride = int(candidate_generation.get("seed_stride", 7919))
        self_play_cfg = self._self_play_cfg_from_candidate_generation(candidate_generation)
        reference_cfg = self._reference_cfg_from_candidate_generation(candidate_generation)
        max_records_per_round = int(curriculum.get("max_records_per_round", len(self.records)))
        num_rounds = _resolve_num_rounds(curriculum.get("num_rounds", 3), len(self.records), max_records_per_round)
        output_cfg = self.config["output"]
        resume_cfg = output_cfg.get("resume", {})
        resume_enabled = bool(resume_cfg.get("enabled", True))
        progress_log_every = max(1, int(resume_cfg.get("progress_log_every", output_cfg.get("progress_log_every", 10))))
        save_all_candidates = bool(output_cfg.get("save_all_candidates", False))
        use_cumulative_manifest = bool(output_cfg.get("use_cumulative_manifest", True))
        write_evaluator_training = bool(output_cfg.get("write_evaluator_training", True))

        overall_summary = {
            **base_run_metadata(),
            "type": "self_evolve_run",
            "config_path": self.config["_config_path"],
            "dry_run": self.dry_run,
            "resume_enabled": resume_enabled,
            "records_available": len(self.records),
            "output_root": str(self.output_root),
            "rounds": [],
        }
        save_json({**base_run_metadata(), "config": self.config}, self.output_root / "run_config_resolved.json")
        self.logger.info(
            "Self-evolve run started: rounds=%s records=%s max_records_per_round=%s samples_per_proposal=%s resume=%s",
            num_rounds,
            len(self.records),
            max_records_per_round,
            samples_per_proposal,
            resume_enabled,
        )
        self._log_memory("run_started", rounds=num_rounds, records=len(self.records))
        cumulative_accepted: list[AcceptedSample] = []
        cumulative_training_records: list[dict[str, Any]] = []
        cumulative_preference_records: list[dict[str, Any]] = []

        for round_index in range(1, num_rounds + 1):
            round_started_at = time.time()
            round_dir = ensure_dir(self.output_root / f"round_{round_index:02d}")
            candidates_dir = ensure_dir(round_dir / "candidates")
            accepted_dir = ensure_dir(round_dir / "accepted" / "images")
            candidate_image_dir = ensure_dir(candidates_dir / "images") if save_all_candidates else None
            summary_path = round_dir / "summary.json"
            progress_path = round_dir / "progress.json"
            proposals_path = round_dir / "proposals.jsonl"
            proposal_plan_path = round_dir / "proposal_plan.jsonl"
            proposal_failure_path = round_dir / "proposal_failures.jsonl"
            evaluator_training_path = round_dir / "evaluator_training.jsonl"
            preference_path = round_dir / "evaluator_preferences.jsonl"
            editor_preference_manifest_path = round_dir / "preference_manifest.json"

            if resume_enabled and summary_path.exists():
                round_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if str(round_summary.get("status", "completed")) == "completed":
                    candidate_rows = read_jsonl(proposals_path)
                    accepted_from_round = self._accepted_samples_from_payloads(candidate_rows)
                    cumulative_accepted.extend(accepted_from_round)
                    training_records_from_round, _, _ = self._training_records_from_payloads(candidate_rows)
                    cumulative_training_records.extend(training_records_from_round)
                    preference_records_from_round, _ = self._preference_records_from_payloads(
                        candidate_rows,
                        round_index=round_index,
                    )
                    cumulative_preference_records.extend(preference_records_from_round)
                    self._restore_completed_round_state(round_summary)
                    overall_summary["rounds"].append(round_summary)
                    self.logger.info(
                        "Round %02d already completed; restored %s accepted samples, %s training records, and skipped generation/training.",
                        round_index,
                        len(accepted_from_round),
                        len(training_records_from_round),
                    )
                    continue

            if not resume_enabled:
                for stale_path in [proposals_path, proposal_plan_path, proposal_failure_path, progress_path]:
                    if stale_path.exists():
                        stale_path.unlink()

            difficulty_level = self.difficulty_controller.level
            editor_state_before_round = self.editor.model_state() if isinstance(self.editor, QwenEditEditor) else None
            proposer_state_before_round = (
                self.proposer.model_state() if hasattr(self.proposer, "model_state") else None
            )
            round_records, round_record_info = self._round_records_for_round(round_index, max_records_per_round)
            self_play_samples_per_proposal = self._self_play_samples_for_round(self_play_cfg, round_index)
            reference_samples_per_proposal = self._reference_samples_for_round(reference_cfg, round_index)
            self_play_opponent_state_for_round = (
                self._editor_state_for_self_play_opponent(self_play_cfg)
                if self_play_samples_per_proposal > 0
                else None
            )
            reference_state_for_round = (
                self._editor_state_for_reference_candidate(reference_cfg)
                if reference_samples_per_proposal > 0
                else None
            )
            if self_play_samples_per_proposal > 0 and not isinstance(self_play_opponent_state_for_round, dict):
                self.logger.warning(
                    "Round %02d requested self-play opponent candidates but no opponent editor state is available; disabling self-play for this round.",
                    round_index,
                )
                self_play_samples_per_proposal = 0
                self_play_opponent_state_for_round = None
            if reference_samples_per_proposal > 0 and not isinstance(reference_state_for_round, dict):
                self.logger.warning(
                    "Round %02d requested reference candidates but no reference editor state is available; disabling references for this round.",
                    round_index,
                )
                reference_samples_per_proposal = 0
                reference_state_for_round = None
            expected_candidates_per_group = (
                samples_per_proposal + self_play_samples_per_proposal + reference_samples_per_proposal
            )
            existing_candidate_rows = read_jsonl(proposals_path) if resume_enabled else []
            candidate_payload_by_key = {
                self._candidate_key(payload): payload
                for payload in existing_candidate_rows
                if payload.get("group_id") is not None
            }
            existing_groups = self._group_rows(list(candidate_payload_by_key.values()))
            skippable_groups = {
                group_id
                for group_id, rows in existing_groups.items()
                if self._is_completed_group(rows, expected_candidates_per_group)
            }
            if self.dry_run:
                skippable_groups.update(
                    group_id
                    for group_id, rows in existing_groups.items()
                    if len(rows) >= expected_candidates_per_group
                )

            proposal_plan_rows = read_jsonl(proposal_plan_path) if resume_enabled else []
            proposal_failure_rows = (
                read_jsonl(proposal_failure_path)
                if resume_enabled and proposal_failure_path.exists()
                else []
            )
            completed_failure_record_keys = {
                str(row.get("record_key", ""))
                for row in proposal_failure_rows
                if int(row.get("round_index", -1)) == round_index
                and str(row.get("record_key", ""))
            }
            plan_rows_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
            planned_proposal_by_group: dict[str, EditProposal] = {}
            for plan_row in proposal_plan_rows:
                plan_rows_by_record[str(plan_row.get("record_key", ""))].append(plan_row)
                group_id = str(plan_row.get("group_id") or "")
                if group_id:
                    planned_proposal_by_group[group_id] = self._proposal_from_payload(plan_row)
            for rows in plan_rows_by_record.values():
                rows.sort(key=lambda row: int(row.get("proposal", {}).get("proposal_index", 0)))
            if planned_proposal_by_group:
                skippable_groups = {
                    group_id
                    for group_id, rows in existing_groups.items()
                    if self._is_completed_group(
                        rows,
                        self._expected_candidates_for_proposal(
                            candidate_generation,
                            planned_proposal_by_group[group_id],
                            samples_per_proposal,
                            self_play_samples_per_proposal,
                            reference_samples_per_proposal,
                        )
                        if group_id in planned_proposal_by_group
                        else expected_candidates_per_group,
                    )
                }

            self._write_progress(
                progress_path,
                {
                    "status": "running",
                    "round_index": round_index,
                    "difficulty_level": difficulty_level,
                    **round_record_info,
                    "records_seen": 0,
                    "records_total": len(round_records),
                    "groups_completed": len(skippable_groups),
                    "groups_total_estimate": len(round_records) * proposals_per_image,
                    "samples_per_proposal": samples_per_proposal,
                    "self_play_samples_per_proposal": self_play_samples_per_proposal,
                    "reference_samples_per_proposal": reference_samples_per_proposal,
                    "expected_candidates_per_group": expected_candidates_per_group,
                    "candidate_rows_loaded": len(existing_candidate_rows),
                    "candidate_rows_written": len(candidate_payload_by_key),
                    "accepted": len(self._accepted_samples_from_payloads(list(candidate_payload_by_key.values()))),
                    "elapsed_seconds": 0.0,
                    "proposals_path": str(proposals_path),
                    "proposal_plan_path": str(proposal_plan_path),
                    "proposal_failure_path": str(proposal_failure_path),
                },
            )
            if existing_candidate_rows or proposal_plan_rows:
                self.logger.info(
                    "Round %02d resume state loaded: proposal_plans=%s candidate_rows=%s skippable_groups=%s.",
                    round_index,
                    len(proposal_plan_rows),
                    len(existing_candidate_rows),
                    len(skippable_groups),
                )

            groups_seen = 0
            groups_completed_this_run = 0
            no_proposal_records: list[str] = []
            for record_index, record in enumerate(round_records):
                planned_rows = plan_rows_by_record.get(record.key, [])
                if planned_rows:
                    planned_groups = [
                        (str(plan_row["group_id"]), self._proposal_from_payload(plan_row))
                        for plan_row in planned_rows
                    ]
                elif record.key in completed_failure_record_keys:
                    no_proposal_records.append(record.key)
                    continue
                else:
                    proposed = self.proposer.propose(
                        record=record,
                        round_index=round_index,
                        difficulty_level=difficulty_level,
                        proposals_per_image=proposals_per_image,
                        seed=seed + record_index,
                    )
                    planned_groups = []
                    for proposal in proposed:
                        group_id = (
                            f"{record.key}__r{round_index:02d}__p{proposal.proposal_index:02d}"
                            f"__{proposal.definition.operation_id}"
                        )
                        plan_payload = self._proposal_plan_payload(record_index, record, proposal, group_id)
                        append_jsonl(plan_payload, proposal_plan_path)
                        proposal_plan_rows.append(plan_payload)
                        planned_proposal_by_group[group_id] = proposal
                        planned_groups.append((group_id, proposal))

                if not planned_groups:
                    no_proposal_records.append(record.key)
                    diagnostics = (
                        self.proposer.proposal_diagnostics()
                        if hasattr(self.proposer, "proposal_diagnostics")
                        else []
                    )
                    append_jsonl(
                        {
                            "round_index": round_index,
                            "difficulty_level": difficulty_level,
                            "record_index": record_index,
                            "record_key": record.key,
                            "image_path": str(record.image_path),
                            "caption": record.caption,
                            "scheduled_edit_type": (record.metadata or {}).get("scheduled_edit_type"),
                            "coverage_requested_edit_type": (record.metadata or {}).get(
                                "coverage_requested_edit_type"
                            ),
                            "target_edit_types": (record.metadata or {}).get("target_edit_types"),
                            "metadata": record.metadata,
                            "diagnostics": diagnostics,
                        },
                        proposal_failure_path,
                    )
                    self.logger.warning(
                        "Round %02d record %s produced no proposal after proposer filtering; skipping.",
                        round_index,
                        record.key,
                    )
                    self._write_progress(
                        progress_path,
                        {
                            "status": "running",
                            "round_index": round_index,
                            "difficulty_level": difficulty_level,
                            **round_record_info,
                            "records_seen": record_index + 1,
                            "records_total": len(round_records),
                            "groups_seen": groups_seen,
                            "groups_completed": len(skippable_groups),
                            "groups_completed_this_run": groups_completed_this_run,
                            "groups_total_estimate": len(round_records) * proposals_per_image,
                            "candidate_rows_written": len(candidate_payload_by_key),
                            "accepted": len(
                                self._accepted_samples_from_payloads(list(candidate_payload_by_key.values()))
                            ),
                            "no_proposal_records": list(no_proposal_records),
                            "elapsed_seconds": round(time.time() - round_started_at, 3),
                            "proposals_path": str(proposals_path),
                            "proposal_plan_path": str(proposal_plan_path),
                            "proposal_failure_path": str(proposal_failure_path),
                        },
                    )
                    continue

                for group_id, proposal in planned_groups:
                    groups_seen += 1
                    policy_samples_per_proposal = self._policy_samples_for_proposal(
                        candidate_generation,
                        proposal,
                        samples_per_proposal,
                    )
                    expected_candidates_for_group = (
                        policy_samples_per_proposal
                        + self_play_samples_per_proposal
                        + reference_samples_per_proposal
                    )
                    if group_id in skippable_groups:
                        continue
                    distractors = (
                        self.evaluator.describe_distractors(proposal)
                        if hasattr(self.evaluator, "describe_distractors")
                        else []
                    )
                    if self.dry_run:
                        for candidate_index in range(policy_samples_per_proposal):
                            payload = self._candidate_payload(
                                record,
                                proposal,
                                evaluation_result=None,
                                image_path=None,
                                status="planned",
                                candidate_index=candidate_index,
                                group_id=group_id,
                                distractors=distractors,
                            )
                            candidate_payload_by_key[self._candidate_key(payload)] = payload
                            append_jsonl(payload, proposals_path)
                        reference_index_offset = int(reference_cfg.get("candidate_index_offset", 2000))
                        reference_role = str(reference_cfg.get("candidate_role", "reference:base"))
                        for reference_local_index in range(reference_samples_per_proposal):
                            candidate_index = reference_index_offset + reference_local_index
                            payload = self._candidate_payload(
                                record,
                                proposal,
                                evaluation_result=None,
                                image_path=None,
                                status="planned",
                                candidate_index=candidate_index,
                                group_id=group_id,
                                distractors=distractors,
                                candidate_role=reference_role,
                                candidate_model_state=reference_state_for_round,
                            )
                            candidate_payload_by_key[self._candidate_key(payload)] = payload
                            append_jsonl(payload, proposals_path)
                        skippable_groups.add(group_id)
                        groups_completed_this_run += 1
                        continue

                    if bool(self.config.get("runtime", {}).get("release_proposer_before_generation", True)):
                        self._release_component_model("proposer", "editor candidate generation")
                    self.logger.info(
                        "Round %02d group %s: generating %s policy candidate(s), %s self-play candidate(s), and %s reference candidate(s) for record %s.",
                        round_index,
                        group_id,
                        policy_samples_per_proposal,
                        self_play_samples_per_proposal,
                        reference_samples_per_proposal,
                        record.key,
                    )
                    self._log_memory(
                        "before_group_generation",
                        round_index=round_index,
                        group_id=group_id,
                        record_key=record.key,
                    )
                    with Image.open(record.image_path) as original_image_handle:
                        original_image = original_image_handle.convert("RGB")
                    generated_image_dir = candidate_image_dir or ensure_dir(candidates_dir / "images")
                    existing_group_rows = self._group_rows(list(candidate_payload_by_key.values())).get(group_id, [])
                    generated_rows = {
                        int(row.get("candidate_index", -1)): row
                        for row in existing_group_rows
                        if str(row.get("status")) in {"generated", "accepted", "rejected"}
                        and row.get("edited_image_path")
                    }
                    edited_images_by_index: dict[int, Image.Image] = {}
                    candidate_metadata_by_index: dict[int, dict[str, Any]] = {}
                    policy_editor_state = (
                        self.editor.model_state() if isinstance(self.editor, QwenEditEditor) else None
                    )
                    for candidate_index in range(policy_samples_per_proposal):
                        generated_row = generated_rows.get(candidate_index)
                        generated_path = resolve_path(str(generated_row.get("edited_image_path"))) if generated_row else None
                        if generated_path is not None and generated_path.exists():
                            edited_images_by_index[candidate_index] = Image.open(generated_path).convert("RGB")
                            candidate_metadata_by_index[candidate_index] = {
                                "candidate_role": generated_row.get("candidate_role", "policy"),
                                "candidate_model_state": generated_row.get("candidate_model_state", policy_editor_state),
                                "candidate_seed": generated_row.get("candidate_seed"),
                            }
                            self._log_memory(
                                "loaded_generated_candidate",
                                round_index=round_index,
                                group_id=group_id,
                                candidate_index=candidate_index,
                                image_path=str(generated_path),
                            )
                            continue
                        candidate_seed = (
                            seed
                            + round_index * 1_000_003
                            + record_index * candidate_seed_stride
                            + proposal.proposal_index * 101
                            + candidate_index
                        )
                        candidate_role = "policy"
                        candidate_model_state = policy_editor_state
                        self._log_memory(
                            "before_candidate_generation",
                            round_index=round_index,
                            group_id=group_id,
                            candidate_index=candidate_index,
                            seed=candidate_seed,
                            candidate_role=candidate_role,
                        )
                        if hasattr(self.editor, "edit_candidate"):
                            edited_image = self.editor.edit_candidate(record, proposal, candidate_index, candidate_seed)
                        else:
                            edited_image = self.editor.edit(record, proposal)
                        output_name = (
                            f"{record.key}__r{round_index:02d}__p{proposal.proposal_index:02d}"
                            f"__c{candidate_index:02d}__{proposal.definition.operation_id}.png"
                        )
                        generated_path = generated_image_dir / output_name
                        edited_image.save(generated_path)
                        payload = self._candidate_payload(
                            record,
                            proposal,
                            evaluation_result=None,
                            image_path=generated_path,
                            status="generated",
                            candidate_index=candidate_index,
                            group_id=group_id,
                            distractors=distractors,
                            candidate_role=candidate_role,
                            candidate_model_state=candidate_model_state,
                            candidate_seed=candidate_seed,
                        )
                        candidate_payload_by_key[self._candidate_key(payload)] = payload
                        append_jsonl(payload, proposals_path)
                        edited_images_by_index[candidate_index] = edited_image
                        candidate_metadata_by_index[candidate_index] = {
                            "candidate_role": candidate_role,
                            "candidate_model_state": candidate_model_state,
                            "candidate_seed": candidate_seed,
                        }
                        if isinstance(self.editor, QwenEditEditor):
                            QwenEditEditor._empty_cuda_cache()
                        self._log_memory(
                            "after_candidate_generated",
                            round_index=round_index,
                            group_id=group_id,
                            candidate_index=candidate_index,
                            image_path=str(generated_path),
                            candidate_role=candidate_role,
                        )
                    if (
                        self_play_samples_per_proposal > 0
                        and isinstance(self_play_opponent_state_for_round, dict)
                    ):
                        opponent_index_offset = int(self_play_cfg.get("candidate_index_offset", 1000))
                        opponent_seed_offset = int(self_play_cfg.get("seed_offset", 900_000_000))
                        opponent_role = str(
                            self_play_cfg.get(
                                "candidate_role",
                                f"opponent:{self_play_cfg.get('opponent', 'previous_round')}",
                            )
                        )
                        try:
                            self._set_editor_model_state(self_play_opponent_state_for_round)
                            for opponent_local_index in range(self_play_samples_per_proposal):
                                candidate_index = opponent_index_offset + opponent_local_index
                                generated_row = generated_rows.get(candidate_index)
                                generated_path = (
                                    resolve_path(str(generated_row.get("edited_image_path")))
                                    if generated_row
                                    else None
                                )
                                if generated_path is not None and generated_path.exists():
                                    edited_images_by_index[candidate_index] = Image.open(generated_path).convert("RGB")
                                    candidate_metadata_by_index[candidate_index] = {
                                        "candidate_role": generated_row.get("candidate_role", opponent_role),
                                        "candidate_model_state": generated_row.get(
                                            "candidate_model_state",
                                            self_play_opponent_state_for_round,
                                        ),
                                        "candidate_seed": generated_row.get("candidate_seed"),
                                    }
                                    self._log_memory(
                                        "loaded_generated_candidate",
                                        round_index=round_index,
                                        group_id=group_id,
                                        candidate_index=candidate_index,
                                        image_path=str(generated_path),
                                        candidate_role=opponent_role,
                                    )
                                    continue
                                candidate_seed = (
                                    seed
                                    + opponent_seed_offset
                                    + round_index * 1_000_003
                                    + record_index * candidate_seed_stride
                                    + proposal.proposal_index * 101
                                    + opponent_local_index
                                )
                                self._log_memory(
                                    "before_candidate_generation",
                                    round_index=round_index,
                                    group_id=group_id,
                                    candidate_index=candidate_index,
                                    seed=candidate_seed,
                                    candidate_role=opponent_role,
                                )
                                if hasattr(self.editor, "edit_candidate"):
                                    edited_image = self.editor.edit_candidate(
                                        record,
                                        proposal,
                                        opponent_local_index,
                                        candidate_seed,
                                    )
                                else:
                                    edited_image = self.editor.edit(record, proposal)
                                output_name = (
                                    f"{record.key}__r{round_index:02d}__p{proposal.proposal_index:02d}"
                                    f"__c{candidate_index:02d}__{proposal.definition.operation_id}.png"
                                )
                                generated_path = generated_image_dir / output_name
                                edited_image.save(generated_path)
                                payload = self._candidate_payload(
                                    record,
                                    proposal,
                                    evaluation_result=None,
                                    image_path=generated_path,
                                    status="generated",
                                    candidate_index=candidate_index,
                                    group_id=group_id,
                                    distractors=distractors,
                                    candidate_role=opponent_role,
                                    candidate_model_state=self_play_opponent_state_for_round,
                                    candidate_seed=candidate_seed,
                                )
                                candidate_payload_by_key[self._candidate_key(payload)] = payload
                                append_jsonl(payload, proposals_path)
                                edited_images_by_index[candidate_index] = edited_image
                                candidate_metadata_by_index[candidate_index] = {
                                    "candidate_role": opponent_role,
                                    "candidate_model_state": self_play_opponent_state_for_round,
                                    "candidate_seed": candidate_seed,
                                }
                                if isinstance(self.editor, QwenEditEditor):
                                    QwenEditEditor._empty_cuda_cache()
                                self._log_memory(
                                    "after_candidate_generated",
                                    round_index=round_index,
                                    group_id=group_id,
                                    candidate_index=candidate_index,
                                    image_path=str(generated_path),
                                    candidate_role=opponent_role,
                                )
                        finally:
                            self._set_editor_model_state(policy_editor_state)

                    if (
                        reference_samples_per_proposal > 0
                        and isinstance(reference_state_for_round, dict)
                    ):
                        reference_index_offset = int(reference_cfg.get("candidate_index_offset", 2000))
                        reference_seed_offset = int(reference_cfg.get("seed_offset", 1_200_000_000))
                        reference_role = str(reference_cfg.get("candidate_role", "reference:base"))
                        try:
                            self._set_editor_model_state(reference_state_for_round)
                            for reference_local_index in range(reference_samples_per_proposal):
                                candidate_index = reference_index_offset + reference_local_index
                                generated_row = generated_rows.get(candidate_index)
                                generated_path = (
                                    resolve_path(str(generated_row.get("edited_image_path")))
                                    if generated_row
                                    else None
                                )
                                if generated_path is not None and generated_path.exists():
                                    edited_images_by_index[candidate_index] = Image.open(generated_path).convert("RGB")
                                    candidate_metadata_by_index[candidate_index] = {
                                        "candidate_role": generated_row.get("candidate_role", reference_role),
                                        "candidate_model_state": generated_row.get(
                                            "candidate_model_state",
                                            reference_state_for_round,
                                        ),
                                        "candidate_seed": generated_row.get("candidate_seed"),
                                    }
                                    self._log_memory(
                                        "loaded_generated_candidate",
                                        round_index=round_index,
                                        group_id=group_id,
                                        candidate_index=candidate_index,
                                        image_path=str(generated_path),
                                        candidate_role=reference_role,
                                    )
                                    continue
                                candidate_seed = (
                                    seed
                                    + reference_seed_offset
                                    + round_index * 1_000_003
                                    + record_index * candidate_seed_stride
                                    + proposal.proposal_index * 101
                                    + reference_local_index
                                )
                                self._log_memory(
                                    "before_candidate_generation",
                                    round_index=round_index,
                                    group_id=group_id,
                                    candidate_index=candidate_index,
                                    seed=candidate_seed,
                                    candidate_role=reference_role,
                                )
                                if hasattr(self.editor, "edit_candidate"):
                                    edited_image = self.editor.edit_candidate(
                                        record,
                                        proposal,
                                        reference_local_index,
                                        candidate_seed,
                                    )
                                else:
                                    edited_image = self.editor.edit(record, proposal)
                                output_name = (
                                    f"{record.key}__r{round_index:02d}__p{proposal.proposal_index:02d}"
                                    f"__c{candidate_index:02d}__{proposal.definition.operation_id}.png"
                                )
                                generated_path = generated_image_dir / output_name
                                edited_image.save(generated_path)
                                payload = self._candidate_payload(
                                    record,
                                    proposal,
                                    evaluation_result=None,
                                    image_path=generated_path,
                                    status="generated",
                                    candidate_index=candidate_index,
                                    group_id=group_id,
                                    distractors=distractors,
                                    candidate_role=reference_role,
                                    candidate_model_state=reference_state_for_round,
                                    candidate_seed=candidate_seed,
                                )
                                candidate_payload_by_key[self._candidate_key(payload)] = payload
                                append_jsonl(payload, proposals_path)
                                edited_images_by_index[candidate_index] = edited_image
                                candidate_metadata_by_index[candidate_index] = {
                                    "candidate_role": reference_role,
                                    "candidate_model_state": reference_state_for_round,
                                    "candidate_seed": candidate_seed,
                                }
                                if isinstance(self.editor, QwenEditEditor):
                                    QwenEditEditor._empty_cuda_cache()
                                self._log_memory(
                                    "after_candidate_generated",
                                    round_index=round_index,
                                    group_id=group_id,
                                    candidate_index=candidate_index,
                                    image_path=str(generated_path),
                                    candidate_role=reference_role,
                                )
                        finally:
                            self._set_editor_model_state(policy_editor_state)

                    candidate_indices_for_scoring = sorted(edited_images_by_index)
                    edited_images = [edited_images_by_index[index] for index in candidate_indices_for_scoring]

                    self._log_memory(
                        "before_cepr_scoring",
                        round_index=round_index,
                        group_id=group_id,
                        candidates=len(edited_images),
                    )
                    if hasattr(self.evaluator, "score_group"):
                        evaluation_results = self.evaluator.score_group(
                            proposal, original_image, edited_images, editor=self.editor
                        )
                    else:
                        evaluation_results = [
                            self.evaluator.score(proposal, original_image, edited_image, editor=self.editor)
                            for edited_image in edited_images
                        ]
                    self._log_memory(
                        "after_cepr_scoring",
                        round_index=round_index,
                        group_id=group_id,
                        candidates=len(evaluation_results),
                    )

                    for candidate_index, edited_image, evaluation_result in zip(
                        candidate_indices_for_scoring,
                        edited_images,
                        evaluation_results,
                    ):
                        output_name = (
                            f"{record.key}__r{round_index:02d}__p{proposal.proposal_index:02d}"
                            f"__c{candidate_index:02d}__{proposal.definition.operation_id}.png"
                        )
                        image_path = None
                        if evaluation_result.accepted:
                            image_path = accepted_dir / output_name
                            edited_image.save(image_path)
                        else:
                            image_path = generated_image_dir / output_name
                            if not image_path.exists():
                                edited_image.save(image_path)

                        payload = self._candidate_payload(
                            record,
                            proposal,
                            evaluation_result=evaluation_result,
                            image_path=image_path,
                            status="accepted" if evaluation_result.accepted else "rejected",
                            candidate_index=candidate_index,
                            group_id=group_id,
                            distractors=distractors,
                            candidate_role=candidate_metadata_by_index.get(candidate_index, {}).get(
                                "candidate_role",
                                "policy",
                            ),
                            candidate_model_state=candidate_metadata_by_index.get(candidate_index, {}).get(
                                "candidate_model_state"
                            ),
                            candidate_seed=candidate_metadata_by_index.get(candidate_index, {}).get("candidate_seed"),
                        )
                        candidate_payload_by_key[self._candidate_key(payload)] = payload
                        append_jsonl(payload, proposals_path)

                    del edited_images
                    del evaluation_results
                    del original_image
                    gc.collect()
                    if isinstance(self.editor, QwenEditEditor):
                        QwenEditEditor._empty_cuda_cache()

                    skippable_groups.add(group_id)
                    groups_completed_this_run += 1
                    candidate_payloads_for_progress = list(candidate_payload_by_key.values())
                    accepted_for_progress = self._accepted_samples_from_payloads(candidate_payloads_for_progress)
                    self._write_progress(
                        progress_path,
                        {
                            "status": "running",
                            "round_index": round_index,
                            "difficulty_level": difficulty_level,
                            **round_record_info,
                            "records_seen": record_index + 1,
                            "records_total": len(round_records),
                            "groups_seen": groups_seen,
                            "groups_completed": len(skippable_groups),
                            "groups_completed_this_run": groups_completed_this_run,
                            "groups_total_estimate": len(round_records) * proposals_per_image,
                            "policy_samples_per_current_proposal": policy_samples_per_proposal,
                            "expected_candidates_current_group": expected_candidates_for_group,
                            "candidate_rows_written": len(candidate_payload_by_key),
                            "accepted": len(accepted_for_progress),
                            "no_proposal_records": list(no_proposal_records),
                            "current_record_key": record.key,
                            "current_group_id": group_id,
                            "elapsed_seconds": round(time.time() - round_started_at, 3),
                            "proposals_path": str(proposals_path),
                            "proposal_plan_path": str(proposal_plan_path),
                            "proposal_failure_path": str(proposal_failure_path),
                        },
                    )
                    if groups_completed_this_run % progress_log_every == 0:
                        self.logger.info(
                            "Round %02d progress: completed_groups=%s candidates=%s accepted=%s elapsed=%.1fs.",
                            round_index,
                            len(skippable_groups),
                            len(candidate_payload_by_key),
                            len(accepted_for_progress),
                            time.time() - round_started_at,
                        )

            candidate_payloads = sorted(
                candidate_payload_by_key.values(),
                key=lambda payload: (str(payload.get("group_id", "")), int(payload.get("candidate_index", 0))),
            )
            proposals_path = write_jsonl(candidate_payloads, proposals_path)
            accepted = self._accepted_samples_from_payloads(candidate_payloads)
            persisted_evaluator_training_path = None
            persisted_preference_path = None
            if write_evaluator_training:
                evaluator_training_records, preference_records = self._build_evaluator_exports(candidate_payloads)
                persisted_evaluator_training_path = write_jsonl(evaluator_training_records, evaluator_training_path)
                persisted_preference_path = write_jsonl(preference_records, preference_path)
            manifest_path = round_dir / "train_manifest.json"
            cumulative_accepted.extend(accepted)
            round_training_records, training_weight_audit, training_weight_summary = self._training_records_from_payloads(
                candidate_payloads
            )
            cumulative_training_records.extend(round_training_records)
            manifest_samples = cumulative_training_records if use_cumulative_manifest else round_training_records
            edit_manifest_sample_count = len(manifest_samples)
            train_manifest_family_counts = Counter(
                str(record.get("family") or "unknown") for record in manifest_samples
            )
            train_manifest_edit_type_counts = Counter()
            for record in manifest_samples:
                structured_edit = record.get("structured_edit", {})
                edit_type = None
                if isinstance(structured_edit, dict):
                    edit_type = structured_edit.get("edit_type")
                train_manifest_edit_type_counts[str(edit_type or record.get("family") or "unknown")] += 1
            train_weight_audit_path = write_jsonl(training_weight_audit, round_dir / "train_weights.jsonl")
            save_json(training_weight_summary, round_dir / "train_weight_summary.json")
            _, train_manifest_sample_count, train_manifest_weight_sum = self._write_manifest_records(
                manifest_samples,
                manifest_path,
            )
            self_preference_records, self_preference_summary = self._preference_records_from_payloads(
                candidate_payloads,
                round_index=round_index,
            )
            cumulative_preference_records.extend(self_preference_records)
            use_cumulative_preference_manifest = bool(
                output_cfg.get(
                    "use_cumulative_preference_manifest",
                    self.config.get("training", {}).get("preference", {}).get("use_cumulative_manifest", False),
                )
            )
            editor_preference_records = (
                cumulative_preference_records if use_cumulative_preference_manifest else self_preference_records
            )
            preference_pair_count = len(editor_preference_records)
            (
                _,
                preference_manifest_sample_count,
                preference_manifest_weight_sum,
                preference_anchor_summary,
            ) = self._write_preference_manifest_records(
                editor_preference_records,
                editor_preference_manifest_path,
            )
            self_preference_summary["anchor_replay"] = preference_anchor_summary
            self_preference_summary["round_pairs_without_anchors"] = len(self_preference_records)
            self_preference_summary["pairs_without_anchors"] = preference_pair_count
            self_preference_summary["use_cumulative_preference_manifest"] = use_cumulative_preference_manifest
            self_preference_summary["manifest_samples_with_anchors"] = preference_manifest_sample_count

            accepted_scores = [sample.evaluation_result.total_score for sample in accepted]
            global_scores = [sample.evaluation_result.global_score for sample in accepted]
            local_scores = [sample.evaluation_result.local_score for sample in accepted]
            component_score_totals: dict[str, list[float]] = {}
            for sample in accepted:
                for name, value in sample.evaluation_result.component_scores.items():
                    component_score_totals.setdefault(name, []).append(value)
            total_candidates = len(candidate_payloads)
            grouped_payloads = self._group_rows(candidate_payloads)
            total_groups = len(grouped_payloads)
            accepted_groups = sum(
                1 for rows in grouped_payloads.values() if any(row.get("status") == "accepted" for row in rows)
            )
            candidate_acceptance_rate = (len(accepted) / total_candidates) if total_candidates else 0.0
            group_acceptance_rate = (accepted_groups / total_groups) if total_groups else 0.0
            # Difficulty should track solved edit tasks, not accepted candidate images.
            # With top_m=1 and K candidates, candidate acceptance is capped at 1/K
            # even if every proposal has a valid edit.
            acceptance_rate = group_acceptance_rate
            next_level = self.difficulty_controller.update(acceptance_rate)
            self._write_progress(
                progress_path,
                {
                    "status": "training",
                    "round_index": round_index,
                    "difficulty_level": difficulty_level,
                    "next_difficulty_level": next_level,
                    **round_record_info,
                    "records_seen": len(round_records),
                    "records_total": len(round_records),
                    "groups_completed": len(skippable_groups),
                    "groups_total_estimate": len(round_records) * proposals_per_image,
                    "candidate_rows_written": total_candidates,
                    "accepted": len(accepted),
                    "accepted_groups": accepted_groups,
                    "acceptance_rate": acceptance_rate,
                    "group_acceptance_rate": group_acceptance_rate,
                    "candidate_acceptance_rate": candidate_acceptance_rate,
                    "train_manifest_edit_samples": edit_manifest_sample_count,
                    "train_manifest_family_counts": dict(train_manifest_family_counts),
                    "train_manifest_edit_type_counts": dict(train_manifest_edit_type_counts),
                    "train_manifest_samples": train_manifest_sample_count,
                    "train_manifest_weight_sum": train_manifest_weight_sum,
                    "preference_manifest_samples": preference_manifest_sample_count,
                    "preference_manifest_weight_sum": preference_manifest_weight_sum,
                    "elapsed_seconds": round(time.time() - round_started_at, 3),
                    "proposals_path": str(proposals_path),
                    "proposal_failure_path": str(proposal_failure_path),
                    "manifest_path": str(manifest_path),
                    "preference_manifest_path": str(editor_preference_manifest_path),
                },
            )
            training_cfg = self.config.get("training", {})
            preference_cfg = dict(training_cfg.get("preference", {}))
            preference_training_enabled = bool(preference_cfg.get("enabled", False))
            preference_fallback_to_sft = bool(preference_cfg.get("fallback_to_weighted_sft", False))
            preference_fallback_reason = None
            min_edit_train_samples = int(training_cfg.get("min_edit_train_samples", 1))
            min_preference_pairs = int(preference_cfg.get("min_pairs", training_cfg.get("min_preference_pairs", 1)))
            min_by_family = _coerce_int_map(training_cfg.get("min_edit_train_samples_by_family"))
            min_by_edit_type = _coerce_int_map(training_cfg.get("min_edit_train_samples_by_edit_type"))
            missing_min_counts = []
            preference_min_by_family = _coerce_int_map(preference_cfg.get("min_pairs_by_family"))
            preference_min_by_edit_type = _coerce_int_map(preference_cfg.get("min_pairs_by_edit_type"))
            missing_min_preference_counts = []
            for family, minimum in sorted(min_by_family.items()):
                count = int(train_manifest_family_counts.get(family, 0))
                if count < minimum:
                    missing_min_counts.append(
                        {"axis": "family", "name": family, "count": count, "minimum": minimum}
                    )
            for edit_type, minimum in sorted(min_by_edit_type.items()):
                count = int(train_manifest_edit_type_counts.get(edit_type, 0))
                if count < minimum:
                    missing_min_counts.append(
                        {"axis": "edit_type", "name": edit_type, "count": count, "minimum": minimum}
                    )
            preference_family_counts = Counter(
                str(record.get("family") or "unknown") for record in editor_preference_records
            )
            preference_edit_type_counts = Counter(
                self._record_balance_key(record, "edit_type") for record in editor_preference_records
            )
            for family, minimum in sorted(preference_min_by_family.items()):
                count = int(preference_family_counts.get(family, 0))
                if count < minimum:
                    missing_min_preference_counts.append(
                        {"axis": "family", "name": family, "count": count, "minimum": minimum}
                    )
            for edit_type, minimum in sorted(preference_min_by_edit_type.items()):
                count = int(preference_edit_type_counts.get(edit_type, 0))
                if count < minimum:
                    missing_min_preference_counts.append(
                        {"axis": "edit_type", "name": edit_type, "count": count, "minimum": minimum}
                    )
            if preference_training_enabled:
                editor_training_manifest_path = editor_preference_manifest_path
                editor_training_mode = "preference"
                editor_train_sample_count = preference_manifest_sample_count
            else:
                editor_training_manifest_path = manifest_path
                editor_training_mode = "sft"
                editor_train_sample_count = train_manifest_sample_count

            if (
                preference_training_enabled
                and preference_fallback_to_sft
                and train_manifest_sample_count > 0
                and (
                    preference_manifest_sample_count <= 0
                    or preference_pair_count < min_preference_pairs
                    or bool(missing_min_preference_counts)
                )
            ):
                if preference_manifest_sample_count <= 0:
                    preference_fallback_reason = "empty_preference_manifest"
                elif preference_pair_count < min_preference_pairs:
                    preference_fallback_reason = "below_min_preference_pairs"
                else:
                    preference_fallback_reason = "below_min_preference_family_or_edit_type_pairs"
                editor_training_manifest_path = manifest_path
                editor_training_mode = "sft"
                editor_train_sample_count = train_manifest_sample_count

            if editor_train_sample_count <= 0:
                training_result = None
            elif editor_training_mode == "preference" and preference_pair_count < min_preference_pairs:
                training_result = {
                    "status": "skipped_min_preference_pairs",
                    "preference_pairs": preference_pair_count,
                    "preference_manifest_samples": preference_manifest_sample_count,
                    "min_preference_pairs": min_preference_pairs,
                    "manifest_path": str(editor_preference_manifest_path),
                    "preference_summary": self_preference_summary,
                }
            elif editor_training_mode == "preference" and missing_min_preference_counts:
                training_result = {
                    "status": "skipped_min_preference_family_edit_type_pairs",
                    "missing_min_preference_counts": missing_min_preference_counts,
                    "preference_family_counts": dict(preference_family_counts),
                    "preference_edit_type_counts": dict(preference_edit_type_counts),
                    "manifest_path": str(editor_preference_manifest_path),
                    "preference_summary": self_preference_summary,
                }
            elif editor_training_mode == "sft" and edit_manifest_sample_count < min_edit_train_samples:
                training_result = {
                    "status": "skipped_min_edit_train_samples",
                    "edit_train_samples": edit_manifest_sample_count,
                    "min_edit_train_samples": min_edit_train_samples,
                    "manifest_path": str(editor_training_manifest_path),
                }
            elif missing_min_counts:
                training_result = {
                    "status": "skipped_min_family_edit_train_samples",
                    "missing_min_counts": missing_min_counts,
                    "family_counts": dict(train_manifest_family_counts),
                    "edit_type_counts": dict(train_manifest_edit_type_counts),
                    "manifest_path": str(editor_training_manifest_path),
                }
            else:
                training_result = self._run_training_round(
                    round_index,
                    round_dir,
                    editor_training_manifest_path,
                    training_mode=editor_training_mode,
                )
            if preference_fallback_reason and isinstance(training_result, dict):
                training_result["preference_fallback_to_weighted_sft"] = {
                    "enabled": True,
                    "reason": preference_fallback_reason,
                    "preference_pairs": preference_pair_count,
                    "preference_manifest_samples": preference_manifest_sample_count,
                    "train_manifest_samples": train_manifest_sample_count,
                    "train_manifest_path": str(manifest_path),
                }
            proposer_training_result = self._run_proposer_training_round(round_index, round_dir, candidate_payloads)

            round_summary = {
                "status": "completed",
                "round_index": round_index,
                "difficulty_level": difficulty_level,
                "next_difficulty_level": next_level,
                **round_record_info,
                "records_seen": len(round_records),
                "proposal_groups": total_groups,
                "candidates": total_candidates,
                "accepted": len(accepted),
                "accepted_groups": accepted_groups,
                "cumulative_accepted": len(cumulative_accepted),
                "no_proposal_records": list(no_proposal_records),
                "no_proposal_record_count": len(no_proposal_records),
                "round_training_samples": len(round_training_records),
                "round_training_weight_sum": training_weight_summary["weight_sum"],
                "train_manifest_edit_samples": edit_manifest_sample_count,
                "train_manifest_family_counts": dict(train_manifest_family_counts),
                "train_manifest_edit_type_counts": dict(train_manifest_edit_type_counts),
                "train_manifest_samples": train_manifest_sample_count,
                "train_manifest_weight_sum": train_manifest_weight_sum,
                "preference_manifest_samples": preference_manifest_sample_count,
                "preference_manifest_weight_sum": preference_manifest_weight_sum,
                "round_preference_pairs": len(self_preference_records),
                "cumulative_preference_pairs": len(cumulative_preference_records),
                "preference_summary": self_preference_summary,
                "acceptance_rate": acceptance_rate,
                "group_acceptance_rate": group_acceptance_rate,
                "candidate_acceptance_rate": candidate_acceptance_rate,
                "avg_total_score": (sum(accepted_scores) / len(accepted_scores)) if accepted_scores else 0.0,
                "avg_global_score": (sum(global_scores) / len(global_scores)) if global_scores else 0.0,
                "avg_local_score": (sum(local_scores) / len(local_scores)) if local_scores else 0.0,
                "avg_component_scores": {
                    name: (sum(values) / len(values)) for name, values in sorted(component_score_totals.items())
                },
                "resume": {
                    "enabled": resume_enabled,
                    "candidate_rows_loaded": len(existing_candidate_rows),
                    "proposal_plan_rows_loaded": len(proposal_plan_rows),
                    "skippable_groups_loaded": len(skippable_groups) - groups_completed_this_run,
                    "groups_completed_this_run": groups_completed_this_run,
                    "progress_path": str(progress_path),
                    "proposal_plan_path": str(proposal_plan_path),
                },
                "elapsed_seconds": round(time.time() - round_started_at, 3),
                "proposals_path": str(proposals_path),
                "proposal_failure_path": str(proposal_failure_path),
                "train_weights_path": str(train_weight_audit_path),
                "train_weight_summary": training_weight_summary,
                "evaluator_training_path": (
                    str(persisted_evaluator_training_path) if persisted_evaluator_training_path is not None else None
                ),
                "evaluator_preferences_path": (
                    str(persisted_preference_path) if persisted_preference_path is not None else None
                ),
                "manifest_path": str(manifest_path),
                "preference_manifest_path": str(editor_preference_manifest_path),
                "training": training_result,
                "proposer_training": proposer_training_result,
                "proposer_state_before_round": proposer_state_before_round,
                "proposer_state_after_round": (
                    self.proposer.model_state() if hasattr(self.proposer, "model_state") else None
                ),
                "editor_state_before_round": editor_state_before_round,
                "editor_state_after_round": self.editor.model_state() if isinstance(self.editor, QwenEditEditor) else None,
            }
            save_json(round_summary, summary_path)
            self._write_progress(
                progress_path,
                {
                    "status": "completed",
                    "round_index": round_index,
                    "difficulty_level": difficulty_level,
                    "next_difficulty_level": next_level,
                    **round_record_info,
                    "records_seen": len(round_records),
                    "records_total": len(round_records),
                    "groups_completed": len(skippable_groups),
                    "groups_total_estimate": len(round_records) * proposals_per_image,
                    "candidate_rows_written": total_candidates,
                    "accepted": len(accepted),
                    "accepted_groups": accepted_groups,
                    "acceptance_rate": acceptance_rate,
                    "group_acceptance_rate": group_acceptance_rate,
                    "candidate_acceptance_rate": candidate_acceptance_rate,
                    "train_manifest_edit_samples": edit_manifest_sample_count,
                    "train_manifest_family_counts": dict(train_manifest_family_counts),
                    "train_manifest_edit_type_counts": dict(train_manifest_edit_type_counts),
                    "train_manifest_samples": train_manifest_sample_count,
                    "train_manifest_weight_sum": train_manifest_weight_sum,
                    "preference_manifest_samples": preference_manifest_sample_count,
                    "preference_manifest_weight_sum": preference_manifest_weight_sum,
                    "elapsed_seconds": round(time.time() - round_started_at, 3),
                    "summary_path": str(summary_path),
                    "proposals_path": str(proposals_path),
                    "proposal_failure_path": str(proposal_failure_path),
                    "manifest_path": str(manifest_path),
                    "preference_manifest_path": str(editor_preference_manifest_path),
                },
            )
            overall_summary["rounds"].append(round_summary)
            self.logger.info(
                "Round %02d completed: groups=%s accepted_groups=%s candidates=%s accepted=%s "
                "group_acceptance_rate=%.4f candidate_acceptance_rate=%.4f next_difficulty=%s elapsed=%.1fs.",
                round_index,
                total_groups,
                accepted_groups,
                total_candidates,
                len(accepted),
                group_acceptance_rate,
                candidate_acceptance_rate,
                next_level,
                time.time() - round_started_at,
            )
            if isinstance(editor_state_before_round, dict):
                self.previous_editor_state = dict(editor_state_before_round)

        save_json(overall_summary, self.output_root / "summary.json")
        self.logger.info("Self-evolve run finished. Summary written to %s", self.output_root / "summary.json")
        return overall_summary
