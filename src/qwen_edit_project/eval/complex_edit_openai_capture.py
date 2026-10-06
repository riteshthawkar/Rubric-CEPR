"""Bounded in-memory provenance adapter for the vendored Complex-Edit scorer."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any


def capture_response_provenance(
    response: Any,
    requested_model: str,
    choice_count: int,
) -> dict[str, Any]:
    """Capture fields exposed by the pinned OpenAI parsed-response object."""

    usage = response.usage.to_dict() if response.usage is not None else {}
    return {
        "requested_model": requested_model,
        "provider_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        # openai-python 2.38 attaches the x-request-id header to parsed
        # BaseModel responses as `_request_id`.
        "request_id": getattr(response, "_request_id", None),
        "created": getattr(response, "created", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "service_tier": getattr(response, "service_tier", None),
        "choice_count": choice_count,
        "usage": usage,
    }


def _atomic_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Publish one raw score atomically and never replace a prior artifact."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise RuntimeError(f"Refusing to overwrite Complex-Edit score artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def eval_one_alignment_with_provenance(
    args: tuple[Any, Any, str, Path],
    *,
    response_format: Any,
    if_resume: bool,
    n: int,
    m: int,
    system_prompt: str,
    prompt_template: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Vendored alignment semantics plus atomic provider provenance capture."""

    if if_resume:
        raise RuntimeError("Hardened Complex-Edit provenance capture forbids resume")
    input_image, output_image, instruction, save_path = args
    from complex_edit.eval.alignment import build_msgs
    from complex_edit.utils import (
        CLIENT_OPENAI,
        completion_retry,
        compute_usage,
        dict_mean,
        dict_sum,
    )

    messages = build_msgs(
        input_image,
        output_image,
        instruction,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
    )
    requested_model = os.environ.get("OPENAI_EVAL_MODEL", "")
    response_counts = [m for _ in range(n // m)]
    if n % m:
        response_counts.append(n % m)
    responses = [
        completion_retry(
            client=CLIENT_OPENAI,
            model_name=requested_model,
            msgs=messages,
            n=count,
            response_format=response_format,
        )
        for count in response_counts
    ]
    token_usages, cost_usages = zip(
        *(compute_usage(response) for response in responses), strict=True
    )
    results: list[dict[str, Any]] = []
    for choice in [choice for response in responses for choice in response.choices]:
        result = choice.message.parsed.model_dump()
        result["instruction_following"] = int(result["instruction_following"])
        result["identity_preservation"] = int(result["identity_preservation"])
        if not 0 <= result["instruction_following"] <= 10:
            raise ValueError(f"Invalid instruction-following score: {result}")
        if not 0 <= result["identity_preservation"] <= 10:
            raise ValueError(f"Invalid identity-preservation score: {result}")
        results.append(result)
    if len(results) != n:
        raise ValueError(f"Expected {n} alignment choices, received {len(results)}")
    payload = {
        "instruction": instruction,
        "runs": deepcopy(results),
        "provider_responses": [
            capture_response_provenance(
                response,
                requested_model=requested_model,
                choice_count=len(response.choices),
            )
            for response in responses
        ],
    }
    for result in results:
        result.pop("reasoning", None)
    average = dict_mean(results)
    payload.update(average)
    _atomic_json_new(Path(save_path), payload)
    return average, dict_sum(token_usages), dict_sum(cost_usages)


def eval_one_quality_with_provenance(
    args: tuple[Any, str, Path],
    *,
    response_format: Any,
    if_resume: bool,
    n: int,
    m: int,
    system_prompt: str,
    prompt_template: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Vendored quality semantics plus atomic provider provenance capture."""

    if if_resume:
        raise RuntimeError("Hardened Complex-Edit provenance capture forbids resume")
    output_image, instruction, save_path = args
    from complex_edit.eval.quality import build_msgs
    from complex_edit.utils import (
        CLIENT_OPENAI,
        completion_retry,
        compute_usage,
        dict_mean,
        dict_sum,
    )

    messages = build_msgs(
        output_image,
        instruction,
        system_prompt=system_prompt,
        prompt_template=prompt_template,
    )
    requested_model = os.environ.get("OPENAI_EVAL_MODEL", "")
    response_counts = [m for _ in range(n // m)]
    if n % m:
        response_counts.append(n % m)
    responses = [
        completion_retry(
            client=CLIENT_OPENAI,
            model_name=requested_model,
            msgs=messages,
            n=count,
            response_format=response_format,
        )
        for count in response_counts
    ]
    token_usages, cost_usages = zip(
        *(compute_usage(response) for response in responses), strict=True
    )
    results: list[dict[str, Any]] = []
    for choice in [choice for response in responses for choice in response.choices]:
        result = choice.message.parsed.model_dump()
        result["perceptual_quality"] = int(result["perceptual_quality"])
        if not 0 <= result["perceptual_quality"] <= 10:
            raise ValueError(f"Invalid perceptual-quality score: {result}")
        results.append(result)
    if len(results) != n:
        raise ValueError(f"Expected {n} quality choices, received {len(results)}")
    payload = {
        "instruction": instruction,
        "runs": deepcopy(results),
        "provider_responses": [
            capture_response_provenance(
                response,
                requested_model=requested_model,
                choice_count=len(response.choices),
            )
            for response in responses
        ],
    }
    for result in results:
        result.pop("reasoning", None)
    average = dict_mean(results)
    payload.update(average)
    _atomic_json_new(Path(save_path), payload)
    return average, dict_sum(token_usages), dict_sum(cost_usages)


def install_provenance_capture() -> None:
    """Replace only the two per-item callables used by the vendored evaluators."""

    import complex_edit.eval.alignment as alignment_module
    import complex_edit.eval.quality as quality_module

    alignment_module.eval_one_alignment = eval_one_alignment_with_provenance
    quality_module.eval_one_quality = eval_one_quality_with_provenance
