"""CPU checks for target admission and the weighted-mean SFT objective."""
import json
import math

import pytest

from qwen_edit_project.self_evolve.training_weights import accepted_target_weight, clipped_reward_weight_mean, normalize_record_weights


@pytest.mark.parametrize("reward", [None, 0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_reward_cannot_become_a_positive_training_weight(reward):
    with pytest.raises(ValueError, match="reward"):
        accepted_target_weight(reward, mode="reward")


def test_uniform_mode_keeps_existing_manifest_semantics():
    assert accepted_target_weight(None, scale=0.5) == 0.5


def test_planner_reward_weighting_survives_batch_size_one():
    normalizer = clipped_reward_weight_mean([0.4, 0.8], minimum=0.25, maximum=1.5)
    multipliers = [reward / normalizer for reward in [0.4, 0.8]]
    assert multipliers == pytest.approx([2 / 3, 4 / 3])
    assert sum(multipliers) / 2 == pytest.approx(1.0)


@pytest.mark.parametrize("minimum, maximum", [(0, 1), (1, 0.5), (0.25, float("inf"))])
def test_invalid_planner_weight_bounds_fail_closed(minimum, maximum):
    with pytest.raises(ValueError, match="bounds"):
        clipped_reward_weight_mean([0.5], minimum=minimum, maximum=maximum)


def test_normalization_matches_weighted_loss_gradient_with_replay():
    records = [
        {"sample_weight": 0.8, "kind": "accepted"},
        {"sample_weight": 0.4, "kind": "accepted"},
        {"sample_weight": 0.5, "kind": "replay"},
    ]
    gradients = [2.0, -3.0, 1.0]
    normalized = normalize_record_weights(records)
    sampled_gradient = sum(r["sample_weight"] * g for r, g in zip(normalized, gradients)) / len(records)
    expected_gradient = sum(r["sample_weight"] * g for r, g in zip(records, gradients)) / sum(r["sample_weight"] for r in records)
    assert sampled_gradient == pytest.approx(expected_gradient)
    assert sum(r["sample_weight"] for r in normalized) == pytest.approx(len(records))
    assert normalized[0]["sample_weight"] / normalized[1]["sample_weight"] == pytest.approx(2.0)
    assert records[0] == {"sample_weight": 0.8, "kind": "accepted"}


@pytest.mark.parametrize("weights", [[0, 0], [-1, 1], [math.inf, 1], [math.nan, 1]])
def test_invalid_manifest_weights_fail_closed(weights):
    with pytest.raises(ValueError, match="weights"):
        normalize_record_weights([{"sample_weight": w} for w in weights])


def test_loop_weights_only_selected_targets_and_normalizes_replay(tmp_path):
    from qwen_edit_project.self_evolve.loop import SelfEvolveRunner
    runner = SelfEvolveRunner.__new__(SelfEvolveRunner)
    runner.config = {"training": {
        "weighted_sft": {"enabled": True, "include_rejected": False, "accepted_weight_mode": "reward"},
        "normalize_sample_weights": True,
        "reconstruction_replay_ratio": 0.5,
        "reconstruction_replay_weight": 0.5,
    }}
    runner._training_contract_filter_reason = lambda *args: None
    proposal = {"instruction": "Extract the cup.", "structured_edit": {"edit_type": "subject_extraction"}}
    payload = {"proposal": proposal, "edited_image_path": str(tmp_path / "target.png"),
               "status": "accepted", "evaluator": {"total_score": 0.8}}
    assert runner._candidate_training_weight(payload) == (0.8, "accepted")
    rejected = {**payload, "status": "rejected"}
    assert runner._candidate_training_weight(rejected) == (0.0, "rejected_disabled")
    records = [{"prompt": "Extract the cup.", "image": str(tmp_path / "target.png"),
                "edit_image": str(tmp_path / "source.png"), "sample_weight": 0.8}]
    path, count, total = runner._write_manifest_records(records, tmp_path / "manifest.json")
    written = json.loads(path.read_text())
    assert count == 2
    assert total == pytest.approx(2.0)
    assert written[1]["candidate_status"] == "reconstruction_replay"
    assert written[1]["image"] == written[1]["edit_image"]
    assert written[0]["sample_weight"] / written[1]["sample_weight"] == pytest.approx(0.8 / 0.5)


@pytest.mark.parametrize("feasible_count, admitted", [(0, False), (1, True), (2, True), (3, True), (4, False)])
def test_planner_admission_uses_all_candidate_gates_not_top_one_selection(feasible_count, admitted):
    from qwen_edit_project.self_evolve.proposer_training import build_proposer_training_records
    rows = [
        {"group_id": "one", "image_path": "source.png", "candidate_index": index,
         "status": "accepted" if index == 0 and feasible_count else "rejected",
         "proposal": {"instruction": "Extract the cup.", "family": "object",
                      "structured_edit": {"edit_type": "subject_extraction", "source_object": "cup"}},
         "evaluator": {"total_score": 0.8 if index < feasible_count else 0.0,
                       "signals": {"feasible": index < feasible_count},
                       "component_scores": {"cepr_raw_reward": 0.8, "cepr_semantic_edit": 0.8,
                                            "cepr_preservation": 0.8, "cepr_validity": 0.8}}}
        for index in range(4)
    ]
    records, _ = build_proposer_training_records(rows, require_productive_band_for_sft=True)
    assert records[0]["use_for_sft"] is admitted
    assert records[0]["metrics"]["feasibility_rate"] == feasible_count / 4
