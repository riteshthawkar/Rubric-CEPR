from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path

from qwen_edit_project.eval.evaluation_contract import (
    build_evaluation_contract,
    build_judge_protocol,
    canonical_json_bytes,
    contracted_model_name,
    contracted_scores_dir,
    evaluation_contract_enabled,
    enforce_official_openai_endpoint,
    sha256_bytes,
    validate_output_manifest,
    write_or_validate_contract,
)
from qwen_edit_project.eval.gedit_selection import write_selection_manifest
from qwen_edit_project.eval.summarize_scores import summarize_gedit
from qwen_edit_project.eval.score_receipts import (
    SCORE_BUNDLE_CLAIM_FILENAME,
    SCORE_BUNDLE_RECEIPT_FILENAME,
    SCORE_BUNDLE_SCHEMA,
    ScoreBundleError,
    acquire_score_bundle_claim,
    begin_score_attempt,
    derive_gedit_metrics,
    release_score_bundle_claim,
    validate_score_summary_bundle,
    write_json_exclusive,
    write_score_bundle_receipt,
)
from qwen_edit_project.utils.commands import run_and_tee
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import resolve_path
from qwen_edit_project.utils.runtime import get_python_executable
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


GEDIT_SCORE_RECEIPT_SCHEMA = SCORE_BUNDLE_SCHEMA
GEDIT_SCORE_RECEIPT_FILENAME = SCORE_BUNDLE_RECEIPT_FILENAME


def ensure_secret_env(secret_path: Path, target_path: Path) -> None:
    secret_text = secret_path.read_text(encoding="utf-8").strip()
    if "OPENAI_API_KEY=" in secret_text:
        raise ValueError(
            "GEdit's upstream VIEScore reader expects secret.env to contain only the raw OpenAI API key, "
            "not a dotenv assignment. Put the key as a single line like: sk-proj-..."
        )
    target_path.write_text(secret_text + "\n", encoding="utf-8")


def validate_expected_openai_model(config: dict, gedit_root: Path) -> None:
    expected_model = config["scoring"].get("expected_openai_model")
    if not expected_model:
        return
    if config["scoring"].get("backbone", "gpt4o") != "gpt4o":
        return
    viescore_init = gedit_root / "viescore" / "__init__.py"
    if not viescore_init.exists():
        raise FileNotFoundError(f"Cannot validate GEdit VIEScore model alias: {viescore_init} is missing")
    source = viescore_init.read_text(encoding="utf-8")
    expected_fragment = f'model_name="{expected_model}"'
    if expected_fragment not in source and f"model_name='{expected_model}'" not in source:
        raise RuntimeError(
            "GEdit scorer does not match the configured judge: config expects backbone=gpt4o to instantiate "
            f"{expected_model}, but {viescore_init} does not contain {expected_fragment}. "
            "Update data/benchmark_tools/step1x-edit or override scoring.expected_openai_model only for separately reported protocols."
        )


def validate_gedit_judge_protocol(config: dict, gedit_root: Path) -> dict:
    protocol = build_judge_protocol(config, benchmark="gedit")
    if protocol is None:
        raise RuntimeError("Hardened GEdit evaluation has no contracted judge protocol")
    expected_request = {
        "model": "gpt-4.1",
        "temperature": None,
        "stream": False,
        "max_tokens": 1400,
        "timeout_seconds": 60,
    }
    expected_execution = {
        "thread_workers": 6,
        "item_max_retries": 3,
        "retry_initial_backoff_seconds": 2.0,
        "retry_max_backoff_seconds": 30.0,
        "strict_response_parser": True,
        "allow_partial": False,
        "instruction_language": config["dataset"].get("instruction_language"),
    }
    if protocol.get("request_parameters") != expected_request:
        raise RuntimeError("GEdit contracted request parameters differ from the pinned scorer")
    if protocol.get("execution_parameters") != expected_execution:
        raise RuntimeError("GEdit contracted execution parameters differ from the pinned scorer")

    openai_source = (gedit_root / "viescore" / "mllm_tools" / "openai.py").read_text(
        encoding="utf-8"
    )
    viescore_source = (gedit_root / "viescore" / "__init__.py").read_text(
        encoding="utf-8"
    )
    viescore_utils_source = (gedit_root / "viescore" / "utils.py").read_text(
        encoding="utf-8"
    )
    runner_source = (gedit_root / "run_gedit_score.py").read_text(encoding="utf-8")
    required_openai_fragments = (
        'self.url = "https://api.openai.com/v1/chat/completions"',
        '"max_tokens": 1400',
        "timeout=self.request_timeout_seconds",
        "response.raise_for_status()",
        "_sanitized_request_evidence(",
        'response_headers.get("x-request-id"',
        '"json": response_payload',
    )
    required_runner_fragments = (
        "max_retries=3",
        "retry_initial_backoff_seconds * (2 ** retry)",
        "ThreadPoolExecutor(max_workers=args.thread_workers)",
        'RAW_RESPONSE_SCHEMA = "gedit-viescore-item-response/v2"',
        'SOURCE_JUDGE_REPRESENTATION = "viescore-resized-jpeg-512-area-v1"',
        "group_csv_list.sort(",
        "Duplicate GEdit score identity before CSV publication",
        "os.link(temporary, final_path)",
        '"provider_transactions": provider_evidence',
    )
    required_viescore_fragments = (
        "json.JSONDecoder().raw_decode(text)",
        "Strict VIEScore parser rejected",
        "if text[end:].strip()",
        "math.isfinite(float(score))",
        "return_evidence=True",
        '"semantic_consistency": evidence_SC',
        '"perceptual_quality": evidence_PQ',
    )
    if any(fragment not in openai_source for fragment in required_openai_fragments):
        raise RuntimeError("Pinned GEdit OpenAI endpoint/request implementation differs")
    if any(fragment not in runner_source for fragment in required_runner_fragments):
        raise RuntimeError("Pinned GEdit retry/concurrency implementation differs")
    if (
        any(fragment not in viescore_source for fragment in required_viescore_fragments)
        or "guess_if_cannot_parse" in viescore_source
    ):
        raise RuntimeError("Pinned GEdit strict response parser implementation differs")
    if (
        "score guessing is disabled" not in viescore_utils_source
        or "import random" in viescore_utils_source
        or "random." in viescore_utils_source
        or "guess_if_cannot_parse" in viescore_utils_source
    ):
        raise RuntimeError("Pinned GEdit parser utilities permit fabricated scores")
    return protocol


def validate_gedit_export_complete(
    config: dict,
    model_dir: Path,
    *,
    records: list[dict] | None = None,
    exact: bool = False,
) -> None:
    if bool(config["scoring"].get("allow_partial", False)):
        return
    from qwen_edit_project.eval.export_gedit import load_selected_gedit_records

    if records is None:
        records, _ = load_selected_gedit_records(config, decode_images=False)
    expected_paths: list[Path] = []
    missing: list[Path] = []
    for item in records:
        path = model_dir / item["task_type"] / item["instruction_language"] / f"{item['key']}.png"
        expected_paths.append(path)
        if not path.exists():
            missing.append(path)
            if len(missing) >= 20:
                break
    if missing:
        total_existing = sum(1 for _ in model_dir.rglob("*.png"))
        raise FileNotFoundError(
            "GEdit export is incomplete. The official scorer expects the full configured "
            f"benchmark split before scoring. Found {total_existing} generated PNGs under "
            f"{model_dir}; first missing files:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nRerun export without --limit, or set scoring.allow_partial=true only for debugging."
        )
    if exact:
        expected_relative = {path.relative_to(model_dir).as_posix() for path in expected_paths}
        if len(expected_relative) != len(expected_paths):
            raise RuntimeError("Contracted GEdit selection maps multiple records to the same output path")
        actual_relative = {path.relative_to(model_dir).as_posix() for path in model_dir.rglob("*.png")}
        unexpected = sorted(actual_relative - expected_relative)
        if unexpected:
            raise RuntimeError(
                "Hardened GEdit output contains PNGs outside the contracted selection: "
                f"{unexpected[:20]} ({len(unexpected)} total)"
            )


def validate_gedit_score_csvs(
    *,
    score_root: Path,
    model_name: str,
    backbone: str,
    records: list[dict],
    instruction_language: str = "all",
    model_dir: Path | None = None,
) -> None:
    expected_by_group: dict[str, dict[tuple[str, str], str]] = {}
    for record in records:
        group = str(record["task_type"])
        identity = (str(record["key"]), str(record["instruction_language"]))
        group_records = expected_by_group.setdefault(group, {})
        if identity in group_records:
            raise RuntimeError(f"Duplicate contracted GEdit score identity: {group}/{identity}")
        group_records[identity] = str(record["instruction"])

    csv_root = score_root / model_name / backbone
    expected_paths = {
        group: csv_root / f"{model_name}_{group}_{instruction_language}_vie_score.csv"
        for group in expected_by_group
    }
    errors: list[str] = []
    for group, expected_records in sorted(expected_by_group.items()):
        path = expected_paths[group]
        if path.is_symlink():
            errors.append(f"symlink CSV {path}")
            continue
        if not path.is_file():
            errors.append(f"missing CSV {path}")
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        required_fields = {
            "key",
            "edited_image",
            "instruction",
            "sementics_score",
            "quality_score",
            "instruction_language",
        }
        duplicate_fields = sorted(
            field for field, count in Counter(fieldnames).items() if count > 1
        )
        missing_fields = sorted(required_fields - set(fieldnames))
        if duplicate_fields or missing_fields:
            errors.append(
                f"{group}: duplicate_fields={duplicate_fields}, missing_fields={missing_fields}"
            )
            continue
        actual_rows = [
            (str(row.get("key", "")), str(row.get("instruction_language", "")))
            for row in rows
        ]
        counts = Counter(actual_rows)
        actual = set(actual_rows)
        expected = set(expected_records)
        duplicates = sorted(identity for identity, count in counts.items() if count > 1)
        canonical_rows = sorted(actual_rows, key=lambda identity: (identity[1], identity[0]))
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        invalid_rows: list[str] = []
        for index, row in enumerate(rows):
            identity = actual_rows[index]
            if None in row:
                invalid_rows.append(f"{identity}: extra unnamed CSV field(s)")
            missing_row_fields = sorted(
                field for field in required_fields if row.get(field) is None
            )
            if missing_row_fields:
                invalid_rows.append(
                    f"{identity}: missing row field(s) {missing_row_fields}"
                )
            expected_instruction = expected_records.get(identity)
            if expected_instruction is not None and str(row.get("instruction", "")) != expected_instruction:
                invalid_rows.append(f"{identity}: instruction mismatch")
            if expected_instruction is not None and model_dir is not None:
                expected_image = (
                    model_dir / group / identity[1] / f"{identity[0]}.png"
                ).resolve()
                raw_image = str(row.get("edited_image", "")).strip()
                if not raw_image:
                    invalid_rows.append(f"{identity}: missing edited_image")
                else:
                    actual_image = Path(raw_image)
                    if not actual_image.is_absolute():
                        actual_image = (Path.cwd() / actual_image).resolve()
                    else:
                        actual_image = actual_image.resolve()
                    if actual_image != expected_image:
                        invalid_rows.append(f"{identity}: edited_image mismatch")
            for field in ("sementics_score", "quality_score"):
                try:
                    score = float(row.get(field, ""))
                except (TypeError, ValueError):
                    invalid_rows.append(f"{identity}: non-numeric {field}")
                    continue
                if not math.isfinite(score) or not 0.0 <= score <= 10.0:
                    invalid_rows.append(f"{identity}: out-of-range {field}={score!r}")
        if actual_rows != canonical_rows:
            invalid_rows.append("CSV rows are not sorted by (instruction_language, key)")
        if duplicates or missing or unexpected or invalid_rows:
            errors.append(
                f"{group}: duplicates={duplicates[:10]}, missing={missing[:10]}, "
                f"unexpected={unexpected[:10]}, invalid={invalid_rows[:10]}"
            )
    if csv_root.exists():
        expected_csv_paths = {path.resolve() for path in expected_paths.values()}
        unexpected_csvs = sorted(
            str(path)
            for path in csv_root.glob(f"{model_name}_*_vie_score.csv")
            if path.resolve() not in expected_csv_paths
        )
        if unexpected_csvs:
            errors.append(f"unexpected score CSVs: {unexpected_csvs[:10]}")
    if errors:
        raise RuntimeError("GEdit score CSV validation failed:\n- " + "\n- ".join(errors))


def _gedit_score_csv_paths(
    *,
    score_root: Path,
    model_name: str,
    backbone: str,
    records: list[dict],
    instruction_language: str,
) -> list[Path]:
    groups = sorted({str(record["task_type"]) for record in records})
    csv_root = score_root / model_name / backbone
    return [
        csv_root / f"{model_name}_{group}_{instruction_language}_vie_score.csv"
        for group in groups
    ]


def _gedit_raw_response_paths(
    *,
    score_root: Path,
    model_name: str,
    backbone: str,
    expected_count: int,
) -> list[Path]:
    raw_root = score_root / model_name / backbone / "raw_responses"
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ScoreBundleError(f"GEdit raw-response directory is missing or unsafe: {raw_root}")
    paths: list[Path] = []
    unexpected: list[str] = []
    for path in sorted(raw_root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            unexpected.append(str(path))
        else:
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                identity = envelope["identity"]
                expected_name = sha256_bytes(canonical_json_bytes(identity)) + ".json"
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
                unexpected.append(str(path))
                continue
            if path.name != expected_name:
                unexpected.append(str(path))
            else:
                paths.append(path)
    if unexpected or len(paths) != expected_count:
        raise ScoreBundleError(
            "GEdit raw-response envelopes differ from contracted selection: "
            f"expected={expected_count}, found={len(paths)}, unexpected={unexpected[:10]}"
        )
    return paths


def build_gedit_record_score_manifest(
    *,
    score_root: Path,
    model_name: str,
    backbone: str,
    records: list[dict],
    instruction_language: str,
) -> dict[str, object]:
    expected = {
        (str(record["task_type"]), str(record["instruction_language"]), str(record["key"]))
        for record in records
    }
    rows: list[dict[str, object]] = []
    for path in _gedit_score_csv_paths(
        score_root=score_root,
        model_name=model_name,
        backbone=backbone,
        records=records,
        instruction_language=instruction_language,
    ):
        task_type = path.name[len(f"{model_name}_") : -len(f"_{instruction_language}_vie_score.csv")]
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                language = str(row["instruction_language"])
                key = str(row["key"])
                semantics = float(row["sementics_score"])
                quality = float(row["quality_score"])
                rows.append(
                    {
                        "identity": f"{task_type}::{language}::{key}",
                        "task_type": task_type,
                        "instruction_language": language,
                        "semantics": semantics,
                        "quality": quality,
                        "overall": math.sqrt(semantics * quality),
                    }
                )
    observed = {
        (str(row["task_type"]), str(row["instruction_language"]), str(row["identity"]).rsplit("::", 1)[1])
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("GEdit record score evidence does not cover the exact contracted rows")
    rows.sort(key=lambda row: str(row["identity"]))
    return {
        "record_scores": rows,
        "record_scores_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


def validate_cached_gedit_scores(
    *,
    score_root: Path,
    model_name: str,
    backbone: str,
    records: list[dict],
    instruction_language: str,
    model_dir: Path,
    contract_id: str,
    summary_path: Path,
) -> bool:
    """Accept only a finalized immutable bundle; never resume or overwrite raw CSVs."""

    expected_paths = _gedit_score_csv_paths(
        score_root=score_root,
        model_name=model_name,
        backbone=backbone,
        records=records,
        instruction_language=instruction_language,
    )
    receipt_path = score_root / GEDIT_SCORE_RECEIPT_FILENAME
    claim_path = score_root / SCORE_BUNDLE_CLAIM_FILENAME
    existing = [path for path in expected_paths if path.exists() or path.is_symlink()]
    csv_root = score_root / model_name / backbone
    unexpected = (
        [
            path
            for path in csv_root.glob(f"{model_name}_*_vie_score.csv")
            if path not in expected_paths
        ]
        if csv_root.exists()
        else []
    )
    bundle_markers = [receipt_path, summary_path, claim_path]
    if not existing and not unexpected and not any(
        path.exists() or path.is_symlink() for path in bundle_markers
    ):
        return False
    if (
        len(existing) != len(expected_paths)
        or unexpected
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or summary_path.is_symlink()
        or not summary_path.is_file()
        or claim_path.exists()
        or claim_path.is_symlink()
    ):
        missing = [str(path) for path in expected_paths if path not in existing]
        raise ScoreBundleError(
            "Refusing incomplete, unreceipted, or contaminated cached GEdit scores: "
            f"missing={missing[:10]}, unexpected={[str(path) for path in unexpected[:10]]}. "
            "Set a new scoring.score_run_id and score under a fresh contract."
        )
    validate_gedit_score_csvs(
        score_root=score_root,
        model_name=model_name,
        backbone=backbone,
        records=records,
        instruction_language=instruction_language,
        model_dir=model_dir,
    )
    verified = validate_score_summary_bundle(summary_path, expected_benchmark="gedit")
    if verified["receipt"].get("contract_id") != contract_id:
        raise ScoreBundleError("Cached GEdit score bundle contract ID differs")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEdit public scorer.")
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

    model_name = config["model"]["model_name"]
    task_type = str(config["dataset"].get("task_type", "all"))
    instruction_language = str(config["dataset"].get("instruction_language", "all"))
    edited_images_dir = resolve_path(config["output"]["edited_images_dir"])
    save_dir = resolve_path(config["scoring"]["save_dir"])
    if edited_images_dir is None or save_dir is None:
        raise ValueError("edited_images_dir and save_dir must resolve")
    from qwen_edit_project.eval.export_gedit import load_selected_gedit_records

    hardening_enabled = evaluation_contract_enabled(config)
    if hardening_enabled and bool(config["scoring"].get("allow_partial", False)):
        raise ValueError("Hardened GEdit evaluation is fail-closed and cannot use scoring.allow_partial=true")
    if hardening_enabled and config["dataset"].get("source", "huggingface") == "huggingface":
        if not config["dataset"].get("revision"):
            raise ValueError("Hardened Hugging Face GEdit evaluation requires dataset.revision")
    if hardening_enabled:
        openai_base_url = enforce_official_openai_endpoint(config)
        judge_protocol = build_judge_protocol(config, benchmark="gedit")
        if judge_protocol is None:
            raise RuntimeError("Hardened GEdit evaluation has no contracted judge protocol")
    else:
        openai_base_url = None
        judge_protocol = None
    records, selection = load_selected_gedit_records(
        config,
        decode_images=hardening_enabled,
    )
    contract = None
    score_attempt = None
    run_model_name = model_name
    if hardening_enabled:
        contract = build_evaluation_contract(config, benchmark="gedit", records=selection["records"])
        run_model_name = contracted_model_name(
            model_name,
            contract,
            length=int(config["evaluation_contract"].get("output_id_length", 16)),
        )
        save_dir = contracted_scores_dir(save_dir, contract)
        write_or_validate_contract(save_dir, contract)
        if not args.attempt_id:
            raise ValueError("Hardened GEdit scoring requires an explicit fresh --attempt-id")
        score_attempt = begin_score_attempt(
            save_dir,
            attempt_id=args.attempt_id,
            benchmark="gedit",
            contract_id=str(contract["contract_id"]),
        )
        save_dir = score_attempt.current_dir
    model_dir = edited_images_dir / run_model_name / "fullset"
    summary_path = save_dir / f"{run_model_name}_summary.json"
    if not model_dir.exists():
        raise FileNotFoundError(f"GEdit output directory not found: {model_dir}")
    if contract is not None:
        write_or_validate_contract(model_dir, contract)
        write_or_validate_contract(save_dir, contract)
    write_selection_manifest(model_dir / ".gedit_selection_manifest.json", selection)
    validate_gedit_export_complete(
        config,
        model_dir,
        records=records,
        exact=hardening_enabled,
    )
    if hardening_enabled:
        expected_images = sorted(
            f"{item['task_type']}/{item['instruction_language']}/{item['key']}.png"
            for item in records
        )
        validate_output_manifest(
            model_dir,
            expected_images,
            contract_id=str(contract["contract_id"]),
        )

    backbone = config["scoring"].get("backbone", "gpt4o")
    cache_reused = False

    repo_root = resolve_path(".")
    gedit_root = resolve_path("data/benchmark_tools/step1x-edit/GEdit-Bench")
    if gedit_root is None or not gedit_root.exists():
        raise FileNotFoundError("GEdit scorer repo is missing. Run python scripts/setup_benchmarks.py first.")
    validate_expected_openai_model(config, gedit_root)
    if hardening_enabled:
        actual_protocol = validate_gedit_judge_protocol(config, gedit_root)
        if actual_protocol != judge_protocol:
            raise RuntimeError("GEdit judge protocol changed after contract construction")

    secret_env = os.environ.get("GEDIT_SECRET_ENV_PATH") or config["scoring"].get("scorer_secret_env_path")
    secret_path = resolve_path(secret_env) if secret_env else None
    if secret_path is None or not secret_path.exists():
        raise FileNotFoundError("GEdit scorer secret env file is missing")
    claim_path = None
    if hardening_enabled:
        claim_path = acquire_score_bundle_claim(
            save_dir,
            benchmark="gedit",
            contract_id=str(contract["contract_id"]),
        )
    local_secret = repo_root / "secret.env"
    python_executable = get_python_executable(config)
    had_existing_secret = local_secret.exists()
    original_secret = local_secret.read_text(encoding="utf-8") if had_existing_secret else None
    ensure_secret_env(secret_path, local_secret)
    scorer_pythonpath = os.pathsep.join(
        [
            str(gedit_root / "viescore"),
            str(gedit_root),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    scorer_env = {"PYTHONPATH": scorer_pythonpath}

    timestamp = utc_timestamp()
    log_path = resolve_path(f"outputs/logs/gedit_score_{timestamp}.log")
    try:
        command = [
            python_executable,
            str(gedit_root / "run_gedit_score.py"),
            "--model_name",
            run_model_name,
            "--edited_images_dir",
            str(edited_images_dir),
            "--save_dir",
            str(save_dir),
            "--backbone",
            backbone,
            "--task_type",
            task_type,
            "--instruction_language",
            instruction_language,
        ]
        dataset_name = config["dataset"].get("dataset_name")
        dataset_revision = config["dataset"].get("revision")
        dataset_split = config["dataset"].get("split")
        if dataset_name:
            command.extend(["--dataset_name", str(dataset_name)])
        if dataset_revision:
            command.extend(["--dataset_revision", str(dataset_revision)])
        if dataset_split:
            command.extend(["--dataset_split", str(dataset_split)])
        command.extend(
            [
                "--request_timeout_seconds",
                str(config["scoring"]["request_timeout_seconds"]),
                "--thread_workers",
                str(config["scoring"]["thread_workers"]),
                "--item_max_retries",
                str(config["scoring"]["item_max_retries"]),
                "--retry_initial_backoff_seconds",
                str(config["scoring"]["retry_initial_backoff_seconds"]),
                "--retry_max_backoff_seconds",
                str(config["scoring"]["retry_max_backoff_seconds"]),
            ]
        )
        return_code = run_and_tee(command, cwd=repo_root, log_path=log_path, env=scorer_env)
        if return_code != 0:
            raise SystemExit(return_code)

        stats_log = None
        if not bool(config["scoring"].get("allow_partial", False)):
            stats_log = resolve_path(f"outputs/logs/gedit_stats_{timestamp}.log")
            stats_command = [
                python_executable,
                str(gedit_root / "calculate_statistics.py"),
                "--model_name",
                run_model_name,
                "--backbone",
                backbone,
                "--save_path",
                str(save_dir),
                "--language",
                instruction_language,
            ]
            return_code = run_and_tee(stats_command, cwd=repo_root, log_path=stats_log, env=scorer_env)
            if return_code != 0:
                raise SystemExit(return_code)
    finally:
        if had_existing_secret and original_secret is not None:
            local_secret.write_text(original_secret, encoding="utf-8")
        elif local_secret.exists():
            local_secret.unlink()

    score_artifacts: dict[str, Path] = {}
    record_score_manifest: dict[str, object] = {}
    if hardening_enabled:
        validate_gedit_score_csvs(
            score_root=save_dir,
            model_name=run_model_name,
            backbone=backbone,
            records=records,
            instruction_language=instruction_language,
            model_dir=model_dir,
        )
        record_score_manifest = build_gedit_record_score_manifest(
            score_root=save_dir,
            model_name=run_model_name,
            backbone=backbone,
            records=records,
            instruction_language=instruction_language,
        )
        csv_paths = _gedit_score_csv_paths(
            score_root=save_dir,
            model_name=run_model_name,
            backbone=backbone,
            records=records,
            instruction_language=instruction_language,
        )
        score_artifacts = {
            f"score_csv:{index:03d}": path for index, path in enumerate(csv_paths)
        }
        raw_response_paths = _gedit_raw_response_paths(
            score_root=save_dir,
            model_name=run_model_name,
            backbone=backbone,
            expected_count=len(records),
        )
        score_artifacts.update(
            {
                f"raw_response:{index:03d}": path
                for index, path in enumerate(raw_response_paths)
            }
        )

    if hardening_enabled:
        metrics = derive_gedit_metrics(
            record_score_manifest["record_scores"],
            backbone=str(backbone),
        )
    else:
        metrics = summarize_gedit(save_dir, run_model_name, backbone)
        metrics.update(record_score_manifest)
    if hardening_enabled:
        if claim_path is None or score_attempt is None:
            raise RuntimeError("Hardened GEdit scoring has no active attempt claim")
        release_score_bundle_claim(
            claim_path,
            benchmark="gedit",
            contract_id=str(contract["contract_id"]),
        )
        active_dir = save_dir
        completed_dir = score_attempt.publish()
        score_artifacts = {
            role: completed_dir / path.relative_to(active_dir)
            for role, path in score_artifacts.items()
        }
        save_dir = completed_dir
        summary_path = save_dir / f"{run_model_name}_summary.json"
    summary = {
        **base_run_metadata(),
        "benchmark": "gedit",
        "config_path": config["_config_path"],
        "model_name": model_name,
        "run_model_name": run_model_name,
        "evaluation_contract": contract,
        "dataset_selection": selection,
        "result_dir": str(model_dir),
        "score_dir": str(save_dir),
        "score_attempt": score_attempt.summary_record() if hardening_enabled else None,
        "score_log": str(log_path),
        "stats_log": str(stats_log) if stats_log is not None else None,
        "score_receipt": (
            {"schema": SCORE_BUNDLE_SCHEMA, "path": SCORE_BUNDLE_RECEIPT_FILENAME}
            if hardening_enabled
            else None
        ),
        "score_artifacts": {
            role: path.relative_to(save_dir).as_posix()
            for role, path in score_artifacts.items()
        },
        "scoring": {
            "backbone": backbone,
            "expected_openai_model": config["scoring"].get("expected_openai_model"),
            "openai_base_url": openai_base_url,
            "judge_protocol": judge_protocol,
            "validated_cache_reused": cache_reused,
        },
        "metrics": metrics,
    }
    if hardening_enabled:
        write_json_exclusive(summary_path, summary)
        write_score_bundle_receipt(
            summary_path=summary_path,
            result_dir=model_dir,
            artifacts=score_artifacts,
            benchmark="gedit",
        )
        score_attempt.commit(summary_path)
    else:
        save_json(summary, summary_path)


if __name__ == "__main__":
    main()
