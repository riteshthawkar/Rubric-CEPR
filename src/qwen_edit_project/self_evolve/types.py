from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UnlabeledImageRecord:
    key: str
    image_path: Path
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposalDefinition:
    operation_id: str
    instruction: str
    family: str
    difficulty: int
    scope: str
    metric: str
    direction: str
    target: float
    expected_changed_fraction: tuple[float, float]
    verifier: str = "proxy"
    inverse_operation_id: str | None = None


@dataclass
class EditProposal:
    record_key: str
    round_index: int
    proposal_index: int
    definition: ProposalDefinition
    difficulty_level: int
    instruction: str
    structured_edit: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    global_score: float
    local_score: float
    total_score: float
    accepted: bool
    signals: dict[str, float]
    component_scores: dict[str, float] = field(default_factory=dict)


# Backward-compatible alias for older configs, manifests, and helper scripts.
SolverResult = EvaluationResult


@dataclass
class AcceptedSample:
    record: UnlabeledImageRecord
    proposal: EditProposal
    edited_image_path: Path
    evaluation_result: EvaluationResult
    candidate_index: int = 0

    @property
    def solver_result(self) -> EvaluationResult:
        """Backward-compatible name used by older manifests/helpers."""
        return self.evaluation_result
