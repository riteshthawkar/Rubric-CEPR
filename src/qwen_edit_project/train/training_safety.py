from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TrainingSchedule:
    """Resolved optimizer-step schedule for one prepared per-process dataloader."""

    batches_per_epoch: int
    gradient_accumulation_steps: int
    update_steps_per_epoch: int
    max_train_steps: int
    num_train_epochs: int


def resolve_training_schedule(
    *,
    batches_per_epoch: int,
    gradient_accumulation_steps: int,
    requested_num_train_epochs: int,
    requested_max_train_steps: int | None,
) -> TrainingSchedule:
    """Resolve epochs and updates after distributed dataloader sharding.

    ``batches_per_epoch`` must be the length of the dataloader *after* it has
    been prepared by Accelerate.  Computing it before preparation silently
    under-runs explicit step budgets when the dataloader is sharded across
    multiple processes.
    """

    if batches_per_epoch <= 0:
        raise ValueError("The prepared training dataloader must contain at least one batch per process.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if requested_num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive.")
    if requested_max_train_steps is not None and requested_max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive when provided.")

    update_steps_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    if requested_max_train_steps is None:
        max_train_steps = requested_num_train_epochs * update_steps_per_epoch
        num_train_epochs = requested_num_train_epochs
    else:
        max_train_steps = requested_max_train_steps
        # An explicit optimizer-step budget is authoritative.  Allocate enough
        # epochs to reach it, then let the step guard stop the final epoch.
        num_train_epochs = math.ceil(max_train_steps / update_steps_per_epoch)

    return TrainingSchedule(
        batches_per_epoch=batches_per_epoch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        update_steps_per_epoch=update_steps_per_epoch,
        max_train_steps=max_train_steps,
        num_train_epochs=num_train_epochs,
    )


@dataclass(frozen=True)
class ResumePosition:
    first_epoch: int
    batches_to_skip: int


def resume_position(
    *,
    global_step: int,
    batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> ResumePosition:
    """Map completed optimizer steps to an epoch and prepared-loader offset.

    The last optimizer update in an epoch can contain fewer accumulation
    batches.  Therefore multiplying every completed step by the accumulation
    factor is incorrect at epoch boundaries.
    """

    if global_step < 0:
        raise ValueError("global_step cannot be negative.")
    schedule = resolve_training_schedule(
        batches_per_epoch=batches_per_epoch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        requested_num_train_epochs=1,
        requested_max_train_steps=None,
    )
    first_epoch, updates_into_epoch = divmod(global_step, schedule.update_steps_per_epoch)
    batches_to_skip = min(updates_into_epoch * gradient_accumulation_steps, batches_per_epoch)
    return ResumePosition(first_epoch=first_epoch, batches_to_skip=batches_to_skip)


def require_completed_steps(*, global_step: int, requested_steps: int, trainer_name: str) -> None:
    """Fail instead of publishing a silently under-trained final adapter."""

    if global_step != requested_steps:
        raise RuntimeError(
            f"{trainer_name} finished at global_step={global_step}, but exactly "
            f"{requested_steps} optimizer steps were requested. Final weights were not saved."
        )


def seed_for_process(base_seed: int, process_index: int) -> int:
    """Return a stable, distinct PyTorch seed for a distributed process."""

    if process_index < 0:
        raise ValueError("process_index cannot be negative.")
    # torch.Generator.manual_seed accepts values in the unsigned 64-bit range.
    return (int(base_seed) + int(process_index)) % (2**64)


def canonical_lora_key(key: str) -> str:
    """Normalize common Diffusers/PEFT prefixes and adapter-name segments."""

    normalized = str(key)
    for prefix in ("base_model.model.", "transformer."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    normalized = normalized.replace(".lora_A.default.", ".lora_A.")
    normalized = normalized.replace(".lora_B.default.", ".lora_B.")
    return normalized


@dataclass(frozen=True)
class LoraKeyCoverage:
    expected_count: int
    provided_count: int
    matched_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    @property
    def fraction(self) -> float:
        return self.matched_count / self.expected_count if self.expected_count else 0.0


def lora_key_coverage(expected_keys: Iterable[str], provided_keys: Iterable[str]) -> LoraKeyCoverage:
    expected = {canonical_lora_key(key) for key in expected_keys}
    provided = {canonical_lora_key(key) for key in provided_keys}
    matched = expected & provided
    return LoraKeyCoverage(
        expected_count=len(expected),
        provided_count=len(provided),
        matched_count=len(matched),
        missing_keys=tuple(sorted(expected - provided)),
        unexpected_keys=tuple(sorted(provided - expected)),
    )


def require_full_lora_key_coverage(
    expected_keys: Iterable[str],
    provided_keys: Iterable[str],
    *,
    checkpoint: str,
) -> LoraKeyCoverage:
    """Require a warm start to initialize every configured adapter tensor."""

    coverage = lora_key_coverage(expected_keys, provided_keys)
    if coverage.expected_count == 0:
        raise ValueError("The configured model exposes no LoRA state keys to initialize.")
    if coverage.missing_keys or coverage.unexpected_keys:
        missing_preview = ", ".join(coverage.missing_keys[:5]) or "none"
        unexpected_preview = ", ".join(coverage.unexpected_keys[:5]) or "none"
        raise ValueError(
            f"LoRA checkpoint '{checkpoint}' is incompatible with the configured adapter: "
            f"matched {coverage.matched_count}/{coverage.expected_count} expected keys; "
            f"missing={len(coverage.missing_keys)} [{missing_preview}]; "
            f"unexpected={len(coverage.unexpected_keys)} [{unexpected_preview}]."
        )
    return coverage
