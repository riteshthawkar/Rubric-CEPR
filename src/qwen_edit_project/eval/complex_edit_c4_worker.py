"""Fresh-only Complex-Edit real/C4 scorer over contracted local source bytes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from qwen_edit_project.eval.complex_edit_contract import (
    ALIGNMENT_DIR,
    HARDENED_COMPLEXITY,
    HARDENED_EVAL_MODEL,
    HARDENED_NUM_PROCESSES,
    HARDENED_PYTHON_EXECUTABLE,
    HARDENED_REQUEST_MAX_ATTEMPTS,
    HARDENED_REQUEST_TIMEOUT_SECONDS,
    HARDENED_RETRY_INITIAL_BACKOFF_SECONDS,
    HARDENED_RETRY_MAX_BACKOFF_SECONDS,
    HARDENED_SCORE_LOG_FILENAME,
    HARDENED_SCORING_M,
    HARDENED_SCORING_N,
    HARDENED_SDK_MAX_RETRIES,
    METRIC_KEYS,
    OVERALL_DIR,
    QUALITY_DIR,
    load_hardened_c4_dataset,
    load_json_strict,
    require_hardened_experiment_runtime,
    require_fresh_score_attempt_namespace,
    require_fresh_score_namespace,
    validate_score_outputs,
)
from qwen_edit_project.eval.complex_edit_openai_capture import install_provenance_capture


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise RuntimeError(f"Refusing to overwrite score artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _mean_dict(rows: Sequence[Mapping[str, float]], keys: Sequence[str]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot average an empty Complex-Edit score collection")
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def reconstruct_keyed_overall(
    *,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Join evaluator outputs by key, never by unordered multiprocessing return order."""

    overall_dir = output_dir / OVERALL_DIR
    overall_dir.mkdir(parents=True, exist_ok=False)
    alignment_values: list[dict[str, float]] = []
    quality_values: list[dict[str, float]] = []
    overall_values: list[dict[str, float]] = []
    for row in rows:
        key = str(row["key"])
        instruction = str(row["compound"][HARDENED_COMPLEXITY - 1])
        alignment = load_json_strict(output_dir / ALIGNMENT_DIR / f"{key}.json")
        quality = load_json_strict(output_dir / QUALITY_DIR / f"{key}.json")
        joined = {
            "instruction_following": float(alignment["instruction_following"]),
            "identity_preservation": float(alignment["identity_preservation"]),
            "perceptual_quality": float(quality["perceptual_quality"]),
        }
        joined["overall"] = sum(joined.values()) / len(joined)
        _atomic_json_new(
            overall_dir / f"{key}.json",
            {"instruction": instruction, **joined},
        )
        alignment_values.append(
            {
                "instruction_following": joined["instruction_following"],
                "identity_preservation": joined["identity_preservation"],
            }
        )
        quality_values.append({"perceptual_quality": joined["perceptual_quality"]})
        overall_values.append(joined)

    _atomic_json_new(
        output_dir / ALIGNMENT_DIR / "final_result.json",
        _mean_dict(
            alignment_values,
            ("instruction_following", "identity_preservation"),
        ),
    )
    _atomic_json_new(
        output_dir / QUALITY_DIR / "final_result.json",
        _mean_dict(quality_values, ("perceptual_quality",)),
    )
    _atomic_json_new(
        overall_dir / "final_result.json",
        _mean_dict(overall_values, METRIC_KEYS),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edited-images-dir", type=Path, required=True)
    parser.add_argument("--score-output-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--contract-id", required=True)
    parser.add_argument("--log-filename", default=HARDENED_SCORE_LOG_FILENAME)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-images-root", type=Path, required=True)
    parser.add_argument("--complexity", type=int, required=True)
    parser.add_argument("--image-type", required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-m", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--request-timeout-seconds", type=float, required=True)
    parser.add_argument("--sdk-max-retries", type=int, required=True)
    parser.add_argument("--request-max-attempts", type=int, required=True)
    parser.add_argument("--retry-initial-backoff-seconds", type=float, required=True)
    parser.add_argument("--retry-max-backoff-seconds", type=float, required=True)
    args = parser.parse_args()

    require_hardened_experiment_runtime(
        {"runtime": {"python_executable": HARDENED_PYTHON_EXECUTABLE}},
        workload="hardened Complex-Edit real/C4/531 score worker",
    )

    if args.complexity != HARDENED_COMPLEXITY or args.image_type != "real":
        parser.error("the hardened worker accepts only real/C4")
    if args.n != HARDENED_SCORING_N or args.m != HARDENED_SCORING_M:
        parser.error("the hardened worker requires the pinned n=8, m=5 protocol")
    if args.num_processes != HARDENED_NUM_PROCESSES:
        parser.error(f"the hardened worker requires num_processes={HARDENED_NUM_PROCESSES}")
    retry_policy = {
        "request_timeout_seconds": (
            args.request_timeout_seconds,
            HARDENED_REQUEST_TIMEOUT_SECONDS,
        ),
        "sdk_max_retries": (args.sdk_max_retries, HARDENED_SDK_MAX_RETRIES),
        "request_max_attempts": (
            args.request_max_attempts,
            HARDENED_REQUEST_MAX_ATTEMPTS,
        ),
        "retry_initial_backoff_seconds": (
            args.retry_initial_backoff_seconds,
            HARDENED_RETRY_INITIAL_BACKOFF_SECONDS,
        ),
        "retry_max_backoff_seconds": (
            args.retry_max_backoff_seconds,
            HARDENED_RETRY_MAX_BACKOFF_SECONDS,
        ),
    }
    for name, (observed, expected) in retry_policy.items():
        if observed != expected:
            parser.error(f"the hardened worker requires {name}={expected}")
    if os.environ.get("OPENAI_EVAL_MODEL") != HARDENED_EVAL_MODEL:
        parser.error(f"OPENAI_EVAL_MODEL must equal {HARDENED_EVAL_MODEL}")
    if os.environ.get("OPENAI_BASE_URL"):
        parser.error("the hardened worker requires the official OpenAI default endpoint")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required")

    os.environ["COMPLEX_EDIT_REQUEST_TIMEOUT_SECONDS"] = str(
        HARDENED_REQUEST_TIMEOUT_SECONDS
    )
    os.environ["COMPLEX_EDIT_SDK_MAX_RETRIES"] = str(HARDENED_SDK_MAX_RETRIES)
    os.environ["COMPLEX_EDIT_REQUEST_MAX_ATTEMPTS"] = str(
        HARDENED_REQUEST_MAX_ATTEMPTS
    )
    os.environ["COMPLEX_EDIT_RETRY_INITIAL_BACKOFF_SECONDS"] = str(
        HARDENED_RETRY_INITIAL_BACKOFF_SECONDS
    )
    os.environ["COMPLEX_EDIT_RETRY_MAX_BACKOFF_SECONDS"] = str(
        HARDENED_RETRY_MAX_BACKOFF_SECONDS
    )

    rows, _ = load_hardened_c4_dataset(args.manifest, args.input_images_root)
    edited_images_dir = args.edited_images_dir.absolute()
    score_output_dir = args.score_output_dir.absolute()
    if edited_images_dir == score_output_dir:
        parser.error("edited-image input and score output directories must be distinct")
    require_fresh_score_namespace(edited_images_dir, rows)
    require_fresh_score_attempt_namespace(
        score_output_dir,
        attempt_id=args.attempt_id,
        contract_id=args.contract_id,
        log_filename=args.log_filename,
    )

    # Import only after the environment and all local identities are validated;
    # importing the vendored package constructs its OpenAI client.
    from complex_edit.eval import AlignmentEvaluator, QualityEvaluator
    from complex_edit.utils import setup_logger

    install_provenance_capture()

    alignment = AlignmentEvaluator(
        if_rubric=True,
        if_cot=True,
        if_resume=False,
        n=args.n,
        m=args.m,
        num_processes=args.num_processes,
    )
    quality = QualityEvaluator(
        if_rubric=True,
        if_cot=False,
        if_resume=False,
        n=args.n,
        m=args.m,
        num_processes=args.num_processes,
    )
    if alignment.result_folder_name != ALIGNMENT_DIR:
        raise RuntimeError("Vendored alignment result-folder contract changed")
    if quality.result_folder_name != QUALITY_DIR:
        raise RuntimeError("Vendored quality result-folder contract changed")

    input_images = [args.input_images_root / str(row["image"]) for row in rows]
    output_images = [edited_images_dir / f"{row['key']}.png" for row in rows]
    instructions = [str(row["compound"][HARDENED_COMPLEXITY - 1]) for row in rows]
    alignment_dir = score_output_dir / ALIGNMENT_DIR
    quality_dir = score_output_dir / QUALITY_DIR
    alignment_dir.mkdir(exist_ok=False)
    quality_dir.mkdir(exist_ok=False)
    alignment_log = alignment_dir / "log.txt"
    setup_logger(output=str(alignment_log))
    alignment_log.touch(exist_ok=True)
    alignment.eval(
        input_images=input_images,
        output_images=output_images,
        instructions=instructions,
        save_paths=[alignment_dir / f"{row['key']}.json" for row in rows],
    )
    quality_log = quality_dir / "log.txt"
    setup_logger(output=str(quality_log))
    quality_log.touch(exist_ok=True)
    quality.eval(
        output_images=output_images,
        instructions=instructions,
        save_paths=[quality_dir / f"{row['key']}.json" for row in rows],
    )

    # The vendored evaluator returns multiprocessing results in completion order.
    # Re-read each key's saved results to prevent cross-item alignment corruption.
    reconstruct_keyed_overall(output_dir=score_output_dir, rows=rows)
    overall_log = score_output_dir / OVERALL_DIR / "log.txt"
    setup_logger(output=str(overall_log))
    overall_log.touch(exist_ok=True)
    metrics = validate_score_outputs(score_output_dir, rows)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
