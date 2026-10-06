from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import time

from qwen_edit_project.eval.evaluation_contract import (
    OFFICIAL_OPENAI_BASE_URL,
    build_judge_protocol,
    build_evaluation_contract,
    contracted_model_name,
    contracted_scores_dir,
    evaluation_contract_enabled,
    enforce_official_openai_endpoint,
    validate_output_manifest,
    write_or_validate_contract,
)
from qwen_edit_project.eval.imgedit_scores import (
    build_imgedit_record_score_manifest,
    extract_imgedit_average,
)
from qwen_edit_project.eval.summarize_scores import summarize_imgedit
from qwen_edit_project.eval.score_receipts import (
    SCORE_BUNDLE_RECEIPT_FILENAME,
    SCORE_BUNDLE_SCHEMA,
    ScoreBundleError,
    begin_score_attempt,
    derive_imgedit_metrics,
    write_json_exclusive,
    write_score_bundle_receipt,
)
from qwen_edit_project.utils.commands import run_and_tee
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.runtime import get_python_executable
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


def _reject_duplicate_object_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json_dict(path: Path, *, reject_duplicates: bool = False) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=_reject_duplicate_object_pairs if reject_duplicates else None)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def invalid_imgedit_keys(edit_specs: dict, results: dict, *, strict: bool = False) -> list[str]:
    invalid = []
    for key, item in edit_specs.items():
        if extract_imgedit_average(
            results.get(key),
            edit_type=str(item.get("edit_type", "")),
            strict=strict,
        ) is None:
            invalid.append(str(key))
    return invalid


def validate_exact_keys(expected: dict, actual: dict, *, label: str) -> None:
    expected_keys = {str(key) for key in expected}
    actual_keys = {str(key) for key in actual}
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"{label} key set does not exactly match the benchmark: "
            f"missing={missing[:20]} ({len(missing)} total), "
            f"unexpected={unexpected[:20]} ({len(unexpected)} total)"
        )


def save_filtered_edit_json(edit_specs: dict, keys: list[str], path: Path) -> None:
    selected = {key: edit_specs[key] for key in keys}
    save_json(selected, path)


def run_basic_scorer(
    *,
    python_executable: str,
    repo_root: Path,
    result_dir: Path,
    edit_json_path: Path,
    origin_img_root: Path,
    prompts_json_path: Path,
    num_processes: int,
    timestamp: str,
    label: str,
    model: str,
    base_url: str,
    result_json_path: Path,
    temperature: float | None,
    timeout: float,
    max_retries: int,
    force: bool = False,
    strict_response_parser: bool = False,
    require_official_endpoint: bool = False,
) -> Path:
    basic_log = resolve_path(f"outputs/logs/imgedit_score_{timestamp}_{label}.log")
    command = [
        python_executable,
        "-m",
        "qwen_edit_project.eval.imgedit_basic_bench",
        "--result_img_folder",
        str(result_dir),
        "--edit_json",
        str(edit_json_path),
        "--origin_img_root",
        str(origin_img_root),
        "--num_processes",
        str(num_processes),
        "--prompts_json",
        str(prompts_json_path),
        "--model",
        model,
        "--base-url",
        base_url,
        "--result-json",
        str(result_json_path),
    ]
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    command.extend(
        [
            "--timeout",
            str(timeout),
            "--max_retries",
            str(max_retries),
        ]
    )
    if force:
        command.append("--force")
    if strict_response_parser:
        command.append("--strict-response-parser")
    if require_official_endpoint:
        command.append("--require-official-endpoint")
    return_code = run_and_tee(command, cwd=repo_root, log_path=basic_log)
    if return_code != 0:
        raise SystemExit(return_code)
    return basic_log


def require_fresh_imgedit_score_namespace(
    *,
    scores_dir: Path,
    artifact_paths: list[Path],
) -> None:
    receipt = scores_dir / SCORE_BUNDLE_RECEIPT_FILENAME
    existing = [path for path in [receipt, *artifact_paths] if path.exists() or path.is_symlink()]
    if existing:
        raise ScoreBundleError(
            "Refusing to overwrite or resume an ImgEdit score bundle under the same evaluation "
            f"contract; existing={[str(path) for path in existing[:10]]}. Set a new "
            "scoring.score_run_id to create a fresh contract before rescoring."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ImgEdit public scorer.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--attempt-id",
        help="Fresh operational attempt ID; required for hardened scoring.",
    )
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument(
        "--force-rescore",
        action="store_true",
        help="Ignore every existing response and issue fresh judge requests for all benchmark keys.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    model_name = config["model"]["model_name"]
    edit_json_path = resolve_path(config["dataset"]["edit_json"])
    origin_img_root = resolve_path(config["dataset"]["origin_img_root"])
    prompts_json_path = resolve_path(config["dataset"]["prompts_json"])
    edited_images_dir = resolve_path(config["output"]["edited_images_dir"])
    scores_dir = resolve_path(config["output"]["scores_dir"])
    if edit_json_path is None or origin_img_root is None or prompts_json_path is None:
        raise ValueError("ImgEdit dataset paths must resolve")
    if edited_images_dir is None or scores_dir is None:
        raise ValueError("edited_images_dir and scores_dir must resolve")
    hardening_enabled = evaluation_contract_enabled(config)
    strict_response_parser = bool(config["scoring"].get("strict_response_parser", False))
    if hardening_enabled and bool(config["scoring"].get("allow_partial", False)):
        raise ValueError("Hardened ImgEdit evaluation is fail-closed and cannot use scoring.allow_partial=true")
    if hardening_enabled:
        openai_base_url = enforce_official_openai_endpoint(config)
        judge_protocol = build_judge_protocol(config, benchmark="imgedit")
        if judge_protocol is None:
            raise RuntimeError("Hardened ImgEdit evaluation has no contracted judge protocol")
    else:
        openai_base_url = str(
            config["scoring"].get("openai_base_url")
            or os.environ.get("OPENAI_BASE_URL")
            or OFFICIAL_OPENAI_BASE_URL
        ).rstrip("/")
        judge_protocol = None
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set before scoring ImgEdit")

    edit_specs = load_json_dict(edit_json_path, reject_duplicates=hardening_enabled)
    contract = None
    score_attempt = None
    run_model_name = model_name
    if hardening_enabled:
        records = [
            {
                "identity": str(key),
                "key": str(key),
                "edit_type": str(item.get("edit_type", "")),
                "source_id": str(item.get("id", "")),
            }
            for key, item in edit_specs.items()
        ]
        contract = build_evaluation_contract(config, benchmark="imgedit", records=records)
        run_model_name = contracted_model_name(
            model_name,
            contract,
            length=int(config["evaluation_contract"].get("output_id_length", 16)),
        )
        scores_dir = contracted_scores_dir(scores_dir, contract)
        write_or_validate_contract(scores_dir, contract)
        if not args.attempt_id:
            raise ValueError("Hardened ImgEdit scoring requires an explicit fresh --attempt-id")
        score_attempt = begin_score_attempt(
            scores_dir,
            attempt_id=args.attempt_id,
            benchmark="imgedit",
            contract_id=str(contract["contract_id"]),
        )
        scores_dir = score_attempt.current_dir

    result_dir = edited_images_dir / run_model_name
    if not result_dir.exists():
        raise FileNotFoundError(f"ImgEdit output directory not found: {result_dir}")
    ensure_dir(scores_dir)
    if contract is not None:
        write_or_validate_contract(result_dir, contract)
        write_or_validate_contract(scores_dir, contract)

    average_score_json = scores_dir / f"{run_model_name}_average_score.json"
    typescore_json = scores_dir / f"{run_model_name}_typescore.json"
    summary_path = scores_dir / f"{run_model_name}_summary.json"
    raw_score_json = scores_dir / f"{run_model_name}_raw_scores.json"
    failure_path = scores_dir / f"{run_model_name}_unscored_keys.json"
    staging_dir = scores_dir / f".{run_model_name}.score_staging"
    if hardening_enabled:
        require_fresh_imgedit_score_namespace(
            scores_dir=scores_dir,
            artifact_paths=[
                raw_score_json,
                average_score_json,
                typescore_json,
                summary_path,
                failure_path,
                staging_dir,
            ],
        )

    missing_images = [key for key in edit_specs if not (result_dir / f"{key}.png").exists()]
    if missing_images:
        raise FileNotFoundError(
            "ImgEdit export is incomplete; refusing to score a partial image set. "
            f"Missing {len(missing_images)} generated PNG(s), first missing keys: {missing_images[:20]}"
        )
    if hardening_enabled:
        expected_images = {f"{key}.png" for key in edit_specs}
        actual_images = {path.name for path in result_dir.glob("*.png")}
        unexpected_images = sorted(actual_images - expected_images)
        if unexpected_images:
            raise RuntimeError(
                "Hardened ImgEdit output contains images outside the contracted key set: "
                f"{unexpected_images[:20]}"
            )
        validate_output_manifest(
            result_dir,
            sorted(expected_images),
            contract_id=str(contract["contract_id"]),
        )

    if hardening_enabled:
        staging_dir.mkdir(mode=0o700)

    timestamp = utc_timestamp()
    repo_root = resolve_path(".")
    python_executable = get_python_executable(config)
    result_json = (
        staging_dir / "result.json" if hardening_enabled else result_dir / "result.json"
    )
    logs = []
    results = load_json_dict(result_json, reject_duplicates=hardening_enabled)
    force_rescore = args.force_rescore or bool(config["scoring"].get("force_rescore", False))
    invalid_keys = list(edit_specs) if force_rescore else invalid_imgedit_keys(
        edit_specs,
        results,
        strict=strict_response_parser,
    )
    openai_model = str(config["scoring"].get("openai_model", "gpt-4o"))
    raw_temperature = config["scoring"].get("temperature")
    openai_temperature = float(raw_temperature) if raw_temperature is not None else None
    openai_timeout = float(config["scoring"].get("openai_timeout_seconds", 60))
    openai_max_retries = int(config["scoring"].get("openai_max_retries", 2))
    if invalid_keys and len(invalid_keys) == len(edit_specs):
        logs.append(
            str(
                run_basic_scorer(
                    python_executable=python_executable,
                    repo_root=repo_root,
                    result_dir=result_dir,
                    edit_json_path=edit_json_path,
                    origin_img_root=origin_img_root,
                    prompts_json_path=prompts_json_path,
                    num_processes=int(config["scoring"].get("num_processes", 4)),
                    timestamp=timestamp,
                    label="initial",
                    model=openai_model,
                    base_url=openai_base_url,
                    result_json_path=result_json,
                    temperature=openai_temperature,
                    timeout=openai_timeout,
                    max_retries=openai_max_retries,
                    force=force_rescore,
                    strict_response_parser=strict_response_parser,
                    require_official_endpoint=hardening_enabled,
                )
            )
        )
        results = load_json_dict(result_json, reject_duplicates=hardening_enabled)
        invalid_keys = invalid_imgedit_keys(edit_specs, results, strict=strict_response_parser)

    max_retry_rounds = int(config["scoring"].get("max_retry_rounds", 5))
    retry_num_processes = int(config["scoring"].get("retry_num_processes", 1))
    retry_sleep_seconds = float(config["scoring"].get("retry_sleep_seconds", 10))
    for retry_index in range(1, max_retry_rounds + 1):
        if not invalid_keys:
            break
        retry_json = (
            staging_dir if hardening_enabled else scores_dir
        ) / f"{model_name}_retry_{retry_index:02d}_edit.json"
        save_filtered_edit_json(edit_specs, invalid_keys, retry_json)
        full_results = dict(results)
        if retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)
        logs.append(
            str(
                run_basic_scorer(
                    python_executable=python_executable,
                    repo_root=repo_root,
                    result_dir=result_dir,
                    edit_json_path=retry_json,
                    origin_img_root=origin_img_root,
                    prompts_json_path=prompts_json_path,
                    num_processes=max(1, retry_num_processes),
                    timestamp=timestamp,
                    label=f"retry_{retry_index:02d}",
                    model=openai_model,
                    base_url=openai_base_url,
                    result_json_path=result_json,
                    temperature=openai_temperature,
                    timeout=openai_timeout,
                    max_retries=openai_max_retries,
                    strict_response_parser=strict_response_parser,
                    require_official_endpoint=hardening_enabled,
                )
            )
        )
        retry_results = load_json_dict(result_json, reject_duplicates=hardening_enabled)
        full_results.update(retry_results)
        save_json(full_results, result_json)
        results = full_results
        invalid_keys = invalid_imgedit_keys(edit_specs, results, strict=strict_response_parser)

    if invalid_keys and not bool(config["scoring"].get("allow_partial", False)):
        failure_payload = {"unscored_keys": invalid_keys}
        if hardening_enabled:
            write_json_exclusive(failure_path, failure_payload)
        else:
            save_json(failure_payload, failure_path)
        raise RuntimeError(
            f"ImgEdit scoring still has {len(invalid_keys)} unscored key(s) after "
            f"{max_retry_rounds} retry round(s). Wrote {failure_path}. Rerun scoring to retry, "
            "or set scoring.allow_partial=true only for debugging."
        )

    if hardening_enabled or strict_response_parser:
        validate_exact_keys(edit_specs, results, label="ImgEdit result.json")
    if hardening_enabled:
        write_json_exclusive(raw_score_json, results)

    avg_log = None
    type_log = None
    record_score_manifest: dict = {}
    if strict_response_parser:
        averages = {
            str(key): extract_imgedit_average(
                results[key],
                edit_type=str(item["edit_type"]),
                strict=True,
            )
            for key, item in edit_specs.items()
        }
        if any(score is None for score in averages.values()):
            raise RuntimeError("Strict ImgEdit metric construction encountered an invalid response")
        grouped: dict[str, list[float]] = defaultdict(list)
        for key, item in edit_specs.items():
            grouped[str(item["edit_type"])].append(float(averages[str(key)]))
        type_scores = {
            edit_type: math.fsum(values) / len(values)
            for edit_type, values in sorted(grouped.items())
        }
        if hardening_enabled:
            record_score_manifest = build_imgedit_record_score_manifest(
                edit_specs,
                {key: float(value) for key, value in averages.items()},
            )
        if hardening_enabled:
            write_json_exclusive(average_score_json, averages)
            write_json_exclusive(typescore_json, type_scores)
        else:
            save_json(averages, average_score_json)
            save_json(type_scores, typescore_json)
    else:
        avg_log = resolve_path(f"outputs/logs/imgedit_avg_{timestamp}.log")
        command = [
            python_executable,
            str(resolve_path("third_party/imgedit/Benchmark/Basic/step1_get_avgscore.py")),
            "--result_json",
            str(result_dir / "result.json"),
            "--average_score_json",
            str(average_score_json),
        ]
        return_code = run_and_tee(command, cwd=repo_root, log_path=avg_log)
        if return_code != 0:
            raise SystemExit(return_code)
        logs.append(str(avg_log))

        type_log = resolve_path(f"outputs/logs/imgedit_types_{timestamp}.log")
        command = [
            python_executable,
            str(resolve_path("third_party/imgedit/Benchmark/Basic/step2_typescore.py")),
            "--average_score_json",
            str(average_score_json),
            "--typescore_json",
            str(typescore_json),
            "--basic_edit",
            str(resolve_path(config["dataset"]["edit_json"])),
        ]
        return_code = run_and_tee(command, cwd=repo_root, log_path=type_log)
        if return_code != 0:
            raise SystemExit(return_code)
        logs.append(str(type_log))

    if hardening_enabled:
        if score_attempt is None:
            raise RuntimeError("Hardened ImgEdit scoring has no active attempt namespace")
        for child in staging_dir.iterdir():
            if child.is_symlink() or not child.is_file():
                raise ScoreBundleError(f"Unsafe ImgEdit score staging entry: {child}")
            child.unlink()
        staging_dir.rmdir()
        active_dir = scores_dir
        completed_dir = score_attempt.publish()

        def completed_path(path: Path) -> Path:
            return completed_dir / path.relative_to(active_dir)

        average_score_json = completed_path(average_score_json)
        typescore_json = completed_path(typescore_json)
        summary_path = completed_path(summary_path)
        raw_score_json = completed_path(raw_score_json)
        failure_path = completed_path(failure_path)
        scores_dir = completed_dir

    if hardening_enabled:
        metrics = derive_imgedit_metrics(record_score_manifest["record_scores"])
    else:
        metrics = summarize_imgedit(scores_dir, run_model_name)
        metrics.update(record_score_manifest)
    summary = {
        **base_run_metadata(),
        "benchmark": "imgedit",
        "config_path": config["_config_path"],
        "model_name": model_name,
        "run_model_name": run_model_name,
        "evaluation_contract": contract,
        "result_dir": str(result_dir),
        "score_attempt": score_attempt.summary_record() if hardening_enabled else None,
        "average_score_json": str(average_score_json),
        "typescore_json": str(typescore_json),
        "score_receipt": (
            {"schema": SCORE_BUNDLE_SCHEMA, "path": SCORE_BUNDLE_RECEIPT_FILENAME}
            if hardening_enabled
            else None
        ),
        "score_artifacts": (
            {
                "average_scores": average_score_json.name,
                "raw_scores": raw_score_json.name,
                "type_scores": typescore_json.name,
            }
            if hardening_enabled
            else {}
        ),
        "logs": logs,
        "unscored_keys": invalid_keys,
        "scoring": {
            "force_rescore": force_rescore,
            "strict_response_parser": strict_response_parser,
            "openai_model": openai_model,
            "openai_base_url": openai_base_url,
            "temperature": openai_temperature,
            "judge_protocol": judge_protocol,
        },
        "metrics": metrics,
    }
    if hardening_enabled:
        write_json_exclusive(summary_path, summary)
        write_score_bundle_receipt(
            summary_path=summary_path,
            result_dir=result_dir,
            artifacts={
                "average_scores": average_score_json,
                "raw_scores": raw_score_json,
                "type_scores": typescore_json,
            },
            benchmark="imgedit",
        )
        score_attempt.commit(summary_path)
    else:
        save_json(summary, summary_path)


if __name__ == "__main__":
    main()
