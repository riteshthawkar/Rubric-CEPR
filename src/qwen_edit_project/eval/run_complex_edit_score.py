from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from qwen_edit_project.eval.complex_edit_contract import (
    HARDENED_COMPLEXITY,
    HARDENED_SCORE_LOG_FILENAME,
    build_score_receipt,
    expected_export_paths,
    load_hardened_c4_dataset,
    output_manifest_path,
    require_hardened_experiment_runtime,
    require_fresh_score_namespace,
    validate_contract_still_current,
    validate_exact_export,
    validate_hardened_config,
    validate_score_attempt_for_promotion,
    validate_score_outputs,
    write_new_json,
)
from qwen_edit_project.eval.complex_edit_score_attempts import (
    begin_complex_edit_score_attempt,
)
from qwen_edit_project.eval.evaluation_contract import (
    build_evaluation_contract,
    contracted_model_name,
    contracted_scores_dir,
    evaluation_contract_enabled,
    sha256_file,
    validate_output_manifest,
    write_or_validate_contract,
)
from qwen_edit_project.utils.commands import run_and_tee
from qwen_edit_project.utils.config import (
    load_yaml_config,
    merge_override,
    parse_override,
    save_json,
)
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.runtime import get_python_executable
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp

# The upstream Complex-Edit dataset ships 531 samples per split; the vendored
# eval.py asserts the output directory holds exactly this many images.
SPLIT_SIZE = {"real": 531, "syn": 531}
METRIC_KEYS = ("instruction_following", "identity_preservation", "perceptual_quality", "overall")


def aggregate_overall(overall_dir: Path) -> dict[str, float]:
    """Mean each metric across the per-image ``overall/<key>.json`` files.

    We aggregate here rather than trusting ``overall/final_result.json`` because
    upstream eval.py writes the quality dict there by mistake; the per-image
    files are correct.
    """
    sums: dict[str, float] = {key: 0.0 for key in METRIC_KEYS}
    count = 0
    for path in sorted(overall_dir.glob("*.json")):
        if path.name == "final_result.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in METRIC_KEYS:
            if key in data:
                sums[key] += float(data[key])
        count += 1
    if count == 0:
        return {"num_scored": 0}
    result = {key: round(sums[key] / count, 4) for key in METRIC_KEYS}
    result["num_scored"] = count
    return result


def _run_hardened(
    config: dict[str, Any],
    *,
    attempt_id: str,
    scorer_repo: Path,
    python_executable: str,
) -> None:
    validate_hardened_config(config)
    model_name = str(config["model"]["model_name"])
    dataset = config["dataset"]
    scoring = config["scoring"]
    output = config["output"]
    manifest_path = resolve_path(dataset["manifest"])
    images_root = resolve_path(dataset["images_root"])
    edited_images_dir = resolve_path(output["edited_images_dir"])
    scores_root = resolve_path(output["scores_dir"])
    if None in (manifest_path, images_root, edited_images_dir, scores_root):
        raise ValueError("Hardened Complex-Edit paths must resolve")
    rows, contract_records = load_hardened_c4_dataset(manifest_path, images_root)
    contract = build_evaluation_contract(
        config,
        benchmark="complex_edit",
        records=contract_records,
    )
    run_model_name = contracted_model_name(
        model_name,
        contract,
        length=int(config["evaluation_contract"].get("output_id_length", 16)),
    )
    result_root = edited_images_dir / run_model_name
    output_dir = result_root / f"real_c{HARDENED_COMPLEXITY}"
    score_dir = contracted_scores_dir(scores_root, contract)
    write_or_validate_contract(result_root, contract)
    write_or_validate_contract(score_dir, contract)
    validate_exact_export(result_root, rows)
    expected_paths = expected_export_paths(rows)
    validate_output_manifest(
        result_root,
        expected_paths,
        contract_id=str(contract["contract_id"]),
    )
    export_manifest = output_manifest_path(result_root)
    export_manifest_sha256 = sha256_file(export_manifest)
    require_fresh_score_namespace(output_dir, rows)

    score_attempt = begin_complex_edit_score_attempt(
        score_dir,
        attempt_id=attempt_id,
        contract_id=str(contract["contract_id"]),
    )
    active_dir = score_attempt.current_dir
    completed_dir = score_attempt.completed_dir
    receipt_filename = f"{run_model_name}_score_receipt.json"
    summary_filename = f"{run_model_name}_summary.json"
    active_receipt_path = active_dir / receipt_filename
    active_summary_path = active_dir / summary_filename
    completed_receipt_path = completed_dir / receipt_filename
    completed_summary_path = completed_dir / summary_filename
    log_filename = str(output["score_log_filename"])
    if log_filename != HARDENED_SCORE_LOG_FILENAME:
        raise RuntimeError("Complex-Edit score log filename changed after validation")
    log_path = active_dir / log_filename

    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(scorer_repo), child_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    child_env["OPENAI_EVAL_MODEL"] = str(scoring["eval_model"])
    command = [
        python_executable,
        "-m",
        str(scoring["worker_module"]),
        "--edited-images-dir",
        str(output_dir),
        "--score-output-dir",
        str(active_dir),
        "--attempt-id",
        attempt_id,
        "--contract-id",
        str(contract["contract_id"]),
        "--log-filename",
        log_filename,
        "--manifest",
        str(manifest_path),
        "--input-images-root",
        str(images_root),
        "--complexity",
        str(HARDENED_COMPLEXITY),
        "--image-type",
        "real",
        "-n",
        str(scoring["n"]),
        "-m",
        str(scoring["m"]),
        "--num-processes",
        str(scoring["num_processes"]),
        "--request-timeout-seconds",
        str(scoring["request_timeout_seconds"]),
        "--sdk-max-retries",
        str(scoring["sdk_max_retries"]),
        "--request-max-attempts",
        str(scoring["request_max_attempts"]),
        "--retry-initial-backoff-seconds",
        str(scoring["retry_initial_backoff_seconds"]),
        "--retry-max-backoff-seconds",
        str(scoring["retry_max_backoff_seconds"]),
    ]
    repo_root = resolve_path(".")
    try:
        return_code = run_and_tee(
            command,
            cwd=repo_root,
            log_path=log_path,
            env=child_env,
        )
        if return_code != 0:
            raise SystemExit(return_code)

        # Recheck all durable inputs/code and immutable exported PNGs after the
        # judge has consumed them.  Raw judge artifacts exist only in the active
        # attempt directory and can never contaminate the export namespace.
        _, final_contract_records = load_hardened_c4_dataset(manifest_path, images_root)
        validate_contract_still_current(config, contract, final_contract_records)
        validate_exact_export(result_root, rows)
        validate_output_manifest(
            result_root,
            expected_paths,
            contract_id=str(contract["contract_id"]),
        )
        require_fresh_score_namespace(output_dir, rows)
        if sha256_file(export_manifest) != export_manifest_sha256:
            raise RuntimeError("Complex-Edit export manifest changed during scoring")

        metrics = validate_score_outputs(active_dir, rows)
        score_receipt = build_score_receipt(
            active_dir,
            contract_id=str(contract["contract_id"]),
            export_manifest_sha256=export_manifest_sha256,
            metrics=metrics,
        )
        write_new_json(active_receipt_path, score_receipt)
        score_receipt_sha256 = sha256_file(active_receipt_path)
        group = metrics["groups"][f"real_c{HARDENED_COMPLEXITY}"]
        score_attempt_record = {
            "schema": "qwen-edit-score-attempt/v1",
            "attempt_id": attempt_id,
            "benchmark": "complex_edit",
            "contract_id": str(contract["contract_id"]),
            "namespace": f"complete/{attempt_id}",
        }
        summary = {
            **base_run_metadata(),
            "benchmark": "complex_edit",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "run_model_name": run_model_name,
            "evaluation_contract": contract,
            "score_attempt": score_attempt_record,
            "path_binding": {
                "export_images": str(output_dir),
                "score_attempt": str(completed_dir),
                "scorer_repo": str(scorer_repo),
                "score_worker_module": str(scoring["worker_module"]),
            },
            "split": "real",
            "image_type": "real",
            "complexities": [HARDENED_COMPLEXITY],
            "eval_model": child_env["OPENAI_EVAL_MODEL"],
            "result_root": str(result_root),
            "metrics": metrics,
            "per_complexity": {f"real_c{HARDENED_COMPLEXITY}": group},
            "macro_over_complexity": metrics["average"],
            "export_outputs": {
                "manifest_path": str(export_manifest),
                "manifest_sha256": export_manifest_sha256,
                "count": len(expected_paths),
            },
            "score_receipt": {
                "path": str(completed_receipt_path),
                "sha256": score_receipt_sha256,
                "contract_id": score_receipt["contract_id"],
                "export_output_manifest_sha256": score_receipt[
                    "export_output_manifest_sha256"
                ],
                "file_count": score_receipt["file_count"],
                "files_sha256": score_receipt["files_sha256"],
                "judge_provenance": score_receipt["judge_provenance"],
            },
            "logs": [str(completed_dir / log_filename)],
        }
        write_new_json(active_summary_path, summary)

        def validate_before_publication(candidate_dir: Path) -> None:
            if candidate_dir != active_dir:
                raise RuntimeError("Complex-Edit active attempt changed before publication")
            validate_score_attempt_for_promotion(
                candidate_dir,
                completed_dir=completed_dir,
                rows=rows,
                attempt_id=attempt_id,
                contract=contract,
                run_model_name=run_model_name,
                export_output_dir=output_dir,
                export_manifest_sha256=export_manifest_sha256,
                scorer_repo=scorer_repo,
                receipt_filename=receipt_filename,
                summary_filename=summary_filename,
                expected_summary=summary,
                log_filename=log_filename,
            )
            # Make contract/export authentication the final pre-rename action,
            # after the potentially longer 1,596-file score validation above.
            _, promotion_contract_records = load_hardened_c4_dataset(
                manifest_path,
                images_root,
            )
            validate_contract_still_current(
                config,
                contract,
                promotion_contract_records,
            )
            validate_exact_export(result_root, rows)
            validate_output_manifest(
                result_root,
                expected_paths,
                contract_id=str(contract["contract_id"]),
            )
            require_fresh_score_namespace(output_dir, rows)
            if sha256_file(export_manifest) != export_manifest_sha256:
                raise RuntimeError(
                    "Complex-Edit export manifest changed immediately before publication"
                )

        published_dir = score_attempt.publish_validated(validate_before_publication)
        if published_dir != completed_dir or not completed_summary_path.is_file():
            raise RuntimeError("Complex-Edit score attempt publication failed")
    except BaseException as exc:
        score_attempt.quarantine(f"{type(exc).__name__}: {exc}")
        raise
    print(
        "Hardened Complex-Edit real/C4/531 scoring complete: "
        f"{metrics['average']} summary={completed_summary_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Complex-Edit VLM scorer.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--attempt-id",
        help="Fresh operational attempt ID; required for hardened scoring.",
    )
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    hardening_enabled = evaluation_contract_enabled(config)
    if hardening_enabled:
        validate_hardened_config(config)
        if not args.attempt_id:
            raise ValueError(
                "Hardened Complex-Edit scoring requires an explicit fresh --attempt-id"
            )
        require_hardened_experiment_runtime(
            config,
            workload="hardened Complex-Edit real/C4/531 scoring",
        )

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set before scoring Complex-Edit")

    model_name = config["model"]["model_name"]
    dataset_cfg = config["dataset"]
    scoring_cfg = config["scoring"]
    split = str(dataset_cfg["split"])
    image_type = str(scoring_cfg.get("image_type", split))
    if image_type != split:
        raise ValueError(f"scoring.image_type ({image_type}) must match dataset.split ({split})")
    complexities = [int(c) for c in dataset_cfg.get("complexities", [1, 4, 8])]

    edited_images_dir = resolve_path(config["output"]["edited_images_dir"])
    scores_dir = ensure_dir(resolve_path(config["output"]["scores_dir"]))
    if edited_images_dir is None:
        raise ValueError("output.edited_images_dir must resolve")
    result_root = edited_images_dir / model_name

    scorer_repo = resolve_path(str(scoring_cfg.get("scorer_repo", "data/benchmark_tools/complex-edit")))
    if scorer_repo is None or not (scorer_repo / "eval.py").exists():
        raise FileNotFoundError(
            "Complex-Edit scorer not found. Run python scripts/setup_benchmarks.py first."
        )
    python_executable = get_python_executable(config)
    if hardening_enabled:
        _run_hardened(
            config,
            attempt_id=str(args.attempt_id),
            scorer_repo=scorer_repo,
            python_executable=python_executable,
        )
        return

    expected = SPLIT_SIZE.get(split)
    # Verify every requested complexity has a complete image set before scoring.
    for complexity in complexities:
        out_dir = result_root / f"{split}_c{complexity}"
        if not out_dir.exists():
            raise FileNotFoundError(
                f"Missing Complex-Edit output dir: {out_dir}. Run scripts/export_complex_edit.sh first."
            )
        n_png = len(list(out_dir.glob("*.png")))
        if expected is not None and n_png != expected:
            raise FileNotFoundError(
                f"Complex-Edit export for c{complexity} is incomplete: found {n_png} PNG(s), expected {expected}. "
                "Refusing to score a partial set."
            )

    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = f"{scorer_repo}{os.pathsep}{child_env.get('PYTHONPATH', '')}".rstrip(
        os.pathsep
    )
    child_env["OPENAI_EVAL_MODEL"] = str(scoring_cfg.get("eval_model", "gpt-4.1"))

    timestamp = utc_timestamp()
    per_complexity: dict[str, dict] = {}
    logs: list[str] = []
    for complexity in complexities:
        out_dir = result_root / f"{split}_c{complexity}"
        command = [
            python_executable,
            "eval.py",
            "--path",
            str(out_dir),
            "--complexity",
            str(complexity),
            "--image-type",
            image_type,
            "-n",
            str(int(scoring_cfg.get("n", 20))),
            "-m",
            str(int(scoring_cfg.get("m", 5))),
            "--num-processes",
            str(int(scoring_cfg.get("num_processes", 16))),
        ]
        if bool(scoring_cfg.get("resume", True)):
            command.append("--resume")
        log_path = resolve_path(
            f"outputs/logs/complex_edit_score_{timestamp}_{split}_c{complexity}.log"
        )
        return_code = run_and_tee(command, cwd=scorer_repo, log_path=log_path, env=child_env)
        logs.append(str(log_path))
        if return_code != 0:
            raise SystemExit(return_code)
        per_complexity[f"{split}_c{complexity}"] = aggregate_overall(out_dir / "overall")

    # Mean of the per-complexity overall scores (macro over complexity levels).
    macro = {}
    scored = [m for m in per_complexity.values() if m.get("num_scored")]
    if scored:
        for key in METRIC_KEYS:
            vals = [m[key] for m in scored if key in m]
            if vals:
                macro[key] = round(sum(vals) / len(vals), 4)

    save_json(
        {
            **base_run_metadata(),
            "benchmark": "complex_edit",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "split": split,
            "image_type": image_type,
            "complexities": complexities,
            "eval_model": child_env["OPENAI_EVAL_MODEL"],
            "result_root": str(result_root),
            "per_complexity": per_complexity,
            "macro_over_complexity": macro,
            "logs": logs,
        },
        scores_dir / f"{model_name}_complex_edit_summary.json",
    )
    print(f"Complex-Edit scoring complete. per_complexity={per_complexity}")


if __name__ == "__main__":
    main()
