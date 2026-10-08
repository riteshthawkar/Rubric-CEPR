"""Record weights for accepted-target SFT and reconstruction replay."""
from __future__ import annotations

import math
from typing import Any


def accepted_target_weight(reward: Any, *, scale: float = 1.0, mode: str = "uniform") -> float:
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Accepted-target weight scale must be finite and positive")
    if mode == "uniform":
        return scale
    if mode != "reward":
        raise ValueError("accepted_weight_mode must be 'uniform' or 'reward'")
    try:
        value = float(reward)
    except (TypeError, ValueError) as exc:
        raise ValueError("Reward-weighted accepted targets require a finite positive reward") from exc
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Reward-weighted accepted targets require a reward in (0, 1]")
    return scale * value


def normalize_record_weights(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make the mean record weight one without changing relative weights.

    With uniform record sampling, this gives an unbiased gradient of the
    weighted-mean data loss. Normalize after adding replay so both record types
    share the same denominator. Leave the caller's records unchanged.
    """
    if not records:
        return []
    weights = [float(record.get("sample_weight", 1.0)) for record in records]
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("SFT record weights must be finite and non-negative")
    mean_weight = math.fsum(weight / len(weights) for weight in weights)
    if not math.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("SFT record weights must have a finite positive mean")
    return [
        {**record, "unnormalized_sample_weight": weight, "sample_weight": weight / mean_weight}
        for record, weight in zip(records, weights, strict=True)
    ]


def clipped_reward_weight_mean(rewards: list[float], *, minimum: float, maximum: float) -> float:
    """Use one dataset-wide denominator for Planner reward weights."""
    if not math.isfinite(minimum) or not math.isfinite(maximum) or not 0 < minimum <= maximum:
        raise ValueError("Planner reward weight bounds must be finite, positive and ordered")
    if not rewards or any(not math.isfinite(float(reward)) for reward in rewards):
        raise ValueError("Planner rewards must be nonempty and finite")
    return math.fsum(min(max(float(reward), minimum), maximum) / len(rewards) for reward in rewards)
