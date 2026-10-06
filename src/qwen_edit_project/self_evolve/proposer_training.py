from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen_edit_project.self_evolve.edit_schema import normalize_structured_edit
from qwen_edit_project.utils.paths import ensure_dir


def _evaluation(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("evaluator") or row.get("solver") or {}


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _component(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    evaluation = _evaluation(row)
    containers = [
        evaluation.get("component_scores", {}),
        evaluation.get("signals", {}),
        evaluation,
    ]
    for name in names:
        for container in containers:
            if isinstance(container, dict) and name in container:
                return _finite_float(container.get(name), default)
    return default


def _explicit_feasibility(row: dict[str, Any]) -> tuple[bool, bool]:
    """Return ``(feasible, present)`` without confusing rank selection with feasibility."""
    evaluation = _evaluation(row)
    containers = [row, evaluation, evaluation.get("signals", {})]
    for container in containers:
        if not isinstance(container, dict) or "feasible" not in container:
            continue
        value = container.get("feasible")
        if isinstance(value, bool):
            return value, True
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value) >= 0.5, True
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True, True
            if normalized in {"false", "no", "0"}:
                return False, True
        # A present but malformed field is still metadata. Fail closed instead
        # of silently falling back to a top-m-derived status for this candidate.
        return False, True
    return False, False


def _difficulty_band(acceptance_rate: float, center: float, sigma: float) -> float:
    return math.exp(-((acceptance_rate - center) ** 2) / (2.0 * max(sigma, 1e-6) ** 2))


def _proposal_target(proposal: dict[str, Any]) -> dict[str, Any]:
    instruction = str(proposal.get("instruction", "")).strip()
    structured_edit = proposal.get("structured_edit") if isinstance(proposal.get("structured_edit"), dict) else {}
    normalized = normalize_structured_edit(
        structured_edit,
        instruction=instruction,
        family=proposal.get("family"),
    )
    normalized["instruction"] = instruction
    return normalized


def _candidate_role(row: dict[str, Any]) -> str:
    return str(row.get("candidate_role") or "policy")


def _role_matches(role: str, patterns: set[str] | list[str] | tuple[str, ...]) -> bool:
    role = str(role)
    for pattern in patterns or []:
        pattern = str(pattern).strip()
        if not pattern:
            continue
        if pattern.endswith("*") and role.startswith(pattern[:-1]):
            return True
        if role == pattern or role.startswith(f"{pattern}:"):
            return True
    return False


def _record_prompt(row: dict[str, Any]) -> str:
    caption = row.get("caption")
    proposal = row.get("proposal", {})
    parts = [
        "Generate one feasible, image-grounded edit instruction for self-training.",
        "Return only the structured edit JSON.",
    ]
    if caption:
        parts.append(f"Image caption: {caption}")
    if proposal.get("difficulty_level") is not None:
        parts.append(f"Target difficulty level: {proposal['difficulty_level']}")
    return "\n".join(parts)


def _strict_internal_success(
    row: dict[str, Any],
    *,
    min_judge_score: float,
    min_judge_semantic: float,
    min_judge_preservation: float,
    min_judge_artifact_free: float,
    require_judge: bool,
) -> bool:
    supported = _component(row, "internal_vlm_judge_supported") >= 0.5
    if require_judge and not supported:
        return False
    if supported:
        return (
            _component(row, "internal_vlm_judge_score") >= min_judge_score
            and _component(row, "internal_vlm_judge_semantic") >= min_judge_semantic
            and _component(row, "internal_vlm_judge_preservation") >= min_judge_preservation
            and _component(row, "internal_vlm_judge_artifact_free") >= min_judge_artifact_free
        )
    return row.get("status") == "accepted"


def build_proposer_training_records(
    candidate_payloads: list[dict[str, Any]],
    *,
    reward_center: float = 0.50,
    reward_sigma: float = 0.25,
    min_reward: float = 0.35,
    min_quality: float = 0.30,
    allowed_edit_types: set[str] | None = None,
    disallowed_edit_types: set[str] | None = None,
    reference_roles: set[str] | None = None,
    policy_roles: set[str] | None = None,
    base_improvement_weight: float = 0.0,
    base_harm_penalty: float = 0.0,
    base_margin_center: float = 0.03,
    min_judge_score: float = 0.55,
    min_judge_semantic: float = 0.55,
    min_judge_preservation: float = 0.55,
    min_judge_artifact_free: float = 0.55,
    require_judge_for_success: bool = True,
    all_fail_penalty: float = 0.35,
    all_pass_penalty: float = 0.75,
    all_pass_threshold: float = 0.95,
    require_productive_band_for_sft: bool = False,
    min_success_rate_for_sft: float = 0.25,
    max_success_rate_for_sft: float = 0.75,
    require_accepted_or_improved_for_sft: bool = False,
    min_policy_over_reference_margin_for_sft: float = 0.04,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_payloads:
        group_id = str(row.get("group_id") or "")
        if group_id:
            groups[group_id].append(row)

    records: list[dict[str, Any]] = []
    all_rewards = []
    selected = 0
    for group_id, rows in groups.items():
        if not rows:
            continue
        first = rows[0]
        proposal = first.get("proposal", {})
        proposal_target = _proposal_target(proposal)
        edit_type = str(proposal_target.get("edit_type", ""))
        if allowed_edit_types and edit_type not in allowed_edit_types:
            continue
        if disallowed_edit_types and edit_type in disallowed_edit_types:
            continue
        evaluations = [_evaluation(row) for row in rows]
        accepted_rows = [row for row in rows if row.get("status") == "accepted"]
        explicit_feasibility = [_explicit_feasibility(row) for row in rows]
        explicit_feasibility_count = sum(1 for _, present in explicit_feasibility if present)
        if explicit_feasibility_count > 0:
            # Mixed explicit/missing metadata is unsafe: treating the missing
            # rows as ranker rejects recreates the top_m leakage this metric is
            # meant to avoid. Missing rows therefore fail closed.
            candidate_feasibility = [value if present else False for value, present in explicit_feasibility]
            feasibility_source = (
                "explicit_feasible_group"
                if explicit_feasibility_count == len(rows)
                else "explicit_feasible_group_missing_fail_closed"
            )
        else:
            # Backward compatibility for old artifacts only. This fallback is
            # permitted at whole-group granularity, never candidate-by-candidate.
            candidate_feasibility = [row.get("status") == "accepted" for row in rows]
            feasibility_source = "legacy_accepted_group_fallback"
        feasible_count = sum(candidate_feasibility)
        feasibility_rate = feasible_count / max(len(rows), 1)
        feasibility_metadata_complete = explicit_feasibility_count == len(rows)
        feasibility_group_eligible_for_sft = (
            feasibility_metadata_complete or explicit_feasibility_count == 0
        )
        # Keep the historical key as a compatibility alias, but its semantics
        # are now genuine feasible/K rather than selected(top_m)/K.
        acceptance_rate = feasibility_rate
        rank_selection_rate = len(accepted_rows) / max(len(rows), 1)
        totals = [_finite_float(evaluation.get("total_score")) for evaluation in evaluations]
        raw_rewards = [
            _finite_float(evaluation.get("component_scores", {}).get("cepr_raw_reward"))
            for evaluation in evaluations
        ]
        edit_scores = [
            _finite_float(evaluation.get("component_scores", {}).get("cepr_semantic_edit"))
            or _finite_float(evaluation.get("component_scores", {}).get("cepr_edit_specificity"))
            for evaluation in evaluations
        ]
        preservation_scores = [
            _finite_float(evaluation.get("component_scores", {}).get("cepr_preservation"))
            for evaluation in evaluations
        ]
        validity_scores = [
            _finite_float(evaluation.get("component_scores", {}).get("cepr_validity"))
            for evaluation in evaluations
        ]
        best_total = max(totals) if totals else 0.0
        best_raw = max(raw_rewards) if raw_rewards else best_total
        reference_rows = [
            row for row in rows if reference_roles and _role_matches(_candidate_role(row), reference_roles)
        ]
        policy_rows = [
            row
            for row in rows
            if not reference_roles or not _role_matches(_candidate_role(row), reference_roles)
            if not policy_roles or _role_matches(_candidate_role(row), policy_roles)
        ]
        policy_row_ids = {id(row) for row in policy_rows}
        policy_feasibility = [
            feasible
            for row, feasible in zip(rows, candidate_feasibility, strict=True)
            if id(row) in policy_row_ids
        ]
        policy_feasible_count = sum(policy_feasibility)
        reference_rewards = [
            max(
                _finite_float(_evaluation(row).get("total_score")),
                _finite_float(_evaluation(row).get("component_scores", {}).get("cepr_raw_reward")),
            )
            for row in reference_rows
        ]
        policy_rewards = [
            max(
                _finite_float(_evaluation(row).get("total_score")),
                _finite_float(_evaluation(row).get("component_scores", {}).get("cepr_raw_reward")),
            )
            for row in policy_rows
        ]
        policy_success_rows = [
            row
            for row in policy_rows
            if _strict_internal_success(
                row,
                min_judge_score=min_judge_score,
                min_judge_semantic=min_judge_semantic,
                min_judge_preservation=min_judge_preservation,
                min_judge_artifact_free=min_judge_artifact_free,
                require_judge=require_judge_for_success,
            )
        ]
        policy_judge_supported_count = sum(
            1 for row in policy_rows if _component(row, "internal_vlm_judge_supported") >= 0.5
        )
        if policy_rows:
            policy_success_rate = policy_feasible_count / max(len(policy_rows), 1)
            success_rate_source = f"{feasibility_source}:policy"
        else:
            policy_success_rate = feasibility_rate
            success_rate_source = f"{feasibility_source}:all_candidates"
        policy_judge_scores = [
            _component(row, "internal_vlm_judge_score")
            for row in policy_rows
            if _component(row, "internal_vlm_judge_supported") >= 0.5
        ]
        best_reference_reward = max(reference_rewards) if reference_rewards else 0.0
        best_policy_reward = max(policy_rewards) if policy_rewards else best_raw
        base_margin = best_policy_reward - best_reference_reward if reference_rewards else 0.0
        base_improvement_score = 0.0
        base_harm_score = 0.0
        if reference_rewards:
            base_improvement_score = max(0.0, min(1.0, base_margin / max(base_margin_center, 1e-6)))
            base_harm_score = max(0.0, min(1.0, -base_margin / max(base_margin_center, 1e-6)))
        best_judge_score = max(policy_judge_scores) if policy_judge_scores else 0.0
        quality = max(best_total, best_raw, best_judge_score)
        band = _difficulty_band(policy_success_rate, reward_center, reward_sigma)
        edit = max(edit_scores) if edit_scores else 0.0
        preservation = max(preservation_scores) if preservation_scores else 0.0
        validity = max(validity_scores) if validity_scores else 0.0
        proposal_reward = 0.40 * band + 0.35 * quality + 0.15 * edit + 0.05 * preservation + 0.05 * validity
        if base_improvement_weight > 0.0 and reference_rewards:
            proposal_reward += float(base_improvement_weight) * base_improvement_score
        if base_harm_penalty > 0.0 and reference_rewards:
            proposal_reward *= max(0.0, 1.0 - float(base_harm_penalty) * base_harm_score)
        if policy_success_rate <= 0.0:
            proposal_reward *= max(0.0, min(1.0, all_fail_penalty))
        elif policy_success_rate >= all_pass_threshold:
            proposal_reward *= max(0.0, min(1.0, all_pass_penalty))
        proposal_reward = max(0.0, min(1.0, proposal_reward))
        all_rewards.append(proposal_reward)
        rate_in_productive_band = (
            min_success_rate_for_sft <= policy_success_rate <= max_success_rate_for_sft
        )
        productive_for_sft = rate_in_productive_band and feasibility_group_eligible_for_sft
        has_accepted_candidate = len(accepted_rows) > 0
        clearly_improves_reference = bool(reference_rewards) and base_margin >= min_policy_over_reference_margin_for_sft
        outcome_supported_for_sft = has_accepted_candidate or clearly_improves_reference
        use_for_sft = (
            proposal_reward >= min_reward
            and quality >= min_quality
            and bool(proposal.get("instruction"))
            and feasibility_group_eligible_for_sft
            and (productive_for_sft or not require_productive_band_for_sft)
            and (outcome_supported_for_sft or not require_accepted_or_improved_for_sft)
        )
        selected += int(use_for_sft)
        records.append(
            {
                "type": "proposer_sft",
                "group_id": group_id,
                "record_key": first.get("record_key"),
                "source_image": first.get("image_path"),
                "prompt": _record_prompt(first),
                "target": proposal_target,
                "target_json": json.dumps(proposal_target, ensure_ascii=True, sort_keys=True),
                "reward": proposal_reward,
                "use_for_sft": use_for_sft,
                "metrics": {
                    "acceptance_rate": acceptance_rate,
                    "feasibility_rate": feasibility_rate,
                    "feasibility_source": feasibility_source,
                    "feasible_count": feasible_count,
                    "explicit_feasibility_count": explicit_feasibility_count,
                    "missing_feasibility_count": len(rows) - explicit_feasibility_count,
                    "feasibility_metadata_complete": feasibility_metadata_complete,
                    "feasibility_group_eligible_for_sft": feasibility_group_eligible_for_sft,
                    "rank_selection_rate": rank_selection_rate,
                    "policy_success_rate": policy_success_rate,
                    "policy_success_rate_source": success_rate_source,
                    "rate_in_productive_band": rate_in_productive_band,
                    "productive_for_sft": productive_for_sft,
                    "require_productive_band_for_sft": require_productive_band_for_sft,
                    "best_total_score": best_total,
                    "best_raw_reward": best_raw,
                    "best_judge_score": best_judge_score,
                    "best_policy_reward": best_policy_reward,
                    "best_reference_reward": best_reference_reward,
                    "policy_over_reference_margin": base_margin,
                    "has_accepted_candidate": has_accepted_candidate,
                    "clearly_improves_reference": clearly_improves_reference,
                    "outcome_supported_for_sft": outcome_supported_for_sft,
                    "base_improvement_score": base_improvement_score,
                    "base_harm_score": base_harm_score,
                    "policy_candidate_count": len(policy_rows),
                    "policy_feasible_count": policy_feasible_count,
                    "policy_strict_success_count": len(policy_success_rows),
                    "policy_judge_supported_count": policy_judge_supported_count,
                    "reference_candidate_count": len(reference_rows),
                    "best_semantic_edit": edit,
                    "best_preservation": preservation,
                    "best_validity": validity,
                    "difficulty_band": band,
                    "candidate_count": len(rows),
                    "accepted_count": len(accepted_rows),
                },
            }
        )

    summary = {
        "groups": len(groups),
        "records": len(records),
        "selected_for_sft": selected,
        "avg_proposal_reward": sum(all_rewards) / max(len(all_rewards), 1),
        "max_proposal_reward": max(all_rewards) if all_rewards else 0.0,
        "min_reward": min_reward,
        "min_quality": min_quality,
        "reward_center": reward_center,
        "reward_sigma": reward_sigma,
        "reference_roles": sorted(reference_roles or []),
        "policy_roles": sorted(policy_roles or []),
        "base_improvement_weight": base_improvement_weight,
        "base_harm_penalty": base_harm_penalty,
        "base_margin_center": base_margin_center,
        "strict_success": {
            "min_judge_score": min_judge_score,
            "min_judge_semantic": min_judge_semantic,
            "min_judge_preservation": min_judge_preservation,
            "min_judge_artifact_free": min_judge_artifact_free,
            "require_judge_for_success": require_judge_for_success,
            "all_fail_penalty": all_fail_penalty,
            "all_pass_penalty": all_pass_penalty,
            "all_pass_threshold": all_pass_threshold,
            "require_productive_band_for_sft": require_productive_band_for_sft,
            "min_success_rate_for_sft": min_success_rate_for_sft,
            "max_success_rate_for_sft": max_success_rate_for_sft,
            "require_accepted_or_improved_for_sft": require_accepted_or_improved_for_sft,
            "min_policy_over_reference_margin_for_sft": min_policy_over_reference_margin_for_sft,
        },
    }
    return records, summary


def write_proposer_training_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path
