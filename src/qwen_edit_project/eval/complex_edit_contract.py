from __future__ import annotations

import hashlib
import json
import math
import os
import pwd
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from qwen_edit_project.eval.evaluation_contract import (
    CONTRACT_FILENAME,
    OUTPUT_MANIFEST_FILENAME,
    build_evaluation_contract,
    canonical_json_bytes,
    sha256_file,
)
from qwen_edit_project.eval.score_receipts import (
    SCORE_ATTEMPT_ID_PATTERN,
    SCORE_ATTEMPT_MANIFEST_FILENAME,
    SCORE_ATTEMPT_SCHEMA,
    SCORE_ATTEMPTS_DIRECTORY,
)


HARDENED_SPLIT = "real"
HARDENED_COMPLEXITY = 4
HARDENED_SAMPLE_COUNT = 531
HARDENED_DATASET_NAME = "UCSC-VLAA/Complex-Edit"
HARDENED_DATASET_REVISION = "b0b8a81d740ae413d52572281a23dc975c4b4b91"
HARDENED_MANIFEST_SHA256 = "54409f861af493c7c4858d4f719af80c23f323a08a24ba91136975016b232849"
HARDENED_SELECTION_SHA256 = "35330c722420527a22a2ea46bf7d84b601252d61e6c600d36c9ffbd7a50a5f42"
HARDENED_IMAGES_TREE_SHA256 = "b39f6b85add389a85edb6397c673012e77d208a8cb943f1b37bc50f50efafb63"
HARDENED_BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
HARDENED_BASE_MODEL_REVISION = "d3968ef930e841f4c73640fb8afa3b306a78167e"
HARDENED_MODEL_BACKEND = "official_diffusers"
HARDENED_PYTHON_EXECUTABLE = sys.executable  # Release runtime is recorded in each contract.
HARDENED_CONDA_PREFIX = sys.prefix
HARDENED_SCORING_N = 8
HARDENED_SCORING_M = 5
HARDENED_NUM_PROCESSES = 16
HARDENED_EVAL_MODEL = "gpt-4.1"
HARDENED_REQUEST_TIMEOUT_SECONDS = 60.0
HARDENED_SDK_MAX_RETRIES = 0
HARDENED_REQUEST_MAX_ATTEMPTS = 3
HARDENED_RETRY_INITIAL_BACKOFF_SECONDS = 2.0
HARDENED_RETRY_MAX_BACKOFF_SECONDS = 30.0
HARDENED_SCORER_REPO = "third_party/complex-edit"
HARDENED_SCORE_WORKER_MODULE = "qwen_edit_project.eval.complex_edit_c4_worker"
HARDENED_SCORE_ATTEMPTS_SUBDIR = SCORE_ATTEMPTS_DIRECTORY
HARDENED_SCORE_LOG_FILENAME = "scorer.log"
METRIC_KEYS = (
    "instruction_following",
    "identity_preservation",
    "perceptual_quality",
    "overall",
)
ALIGNMENT_DIR = "alignment_rubric_cot"
QUALITY_DIR = "quality_rubric"
OVERALL_DIR = "overall"
SCORE_DIRS = (ALIGNMENT_DIR, QUALITY_DIR, OVERALL_DIR)
SCORE_RECEIPT_SCHEMA = "qwen-edit-complex-edit-c4-scores/v2"
JUDGE_PROVENANCE_SCHEMA = "qwen-edit-complex-edit-openai-responses/v1"

BENCHMARK_PROTOCOL = {
    "name": "complex-edit-real-c4-531-bounded-retry-v2",
    "split": HARDENED_SPLIT,
    "complexity": HARDENED_COMPLEXITY,
    "exact_count": HARDENED_SAMPLE_COUNT,
    "scoring_n": HARDENED_SCORING_N,
    "scoring_m": HARDENED_SCORING_M,
    "scoring_num_processes": HARDENED_NUM_PROCESSES,
    "scoring_temperature": 1.0,
    "request_timeout_seconds": HARDENED_REQUEST_TIMEOUT_SECONDS,
    "sdk_max_retries": HARDENED_SDK_MAX_RETRIES,
    "request_max_attempts": HARDENED_REQUEST_MAX_ATTEMPTS,
    "retry_initial_backoff_seconds": HARDENED_RETRY_INITIAL_BACKOFF_SECONDS,
    "retry_max_backoff_seconds": HARDENED_RETRY_MAX_BACKOFF_SECONDS,
    "judge_endpoint": "official_openai_default",
    "scoring_resume": False,
    "scoring_fresh_only": True,
    "scorer_dataset_source": "contracted_local_materialization",
    "scorer_repo": HARDENED_SCORER_REPO,
    "score_worker_module": HARDENED_SCORE_WORKER_MODULE,
    "export_group_subdir": f"{HARDENED_SPLIT}_c{HARDENED_COMPLEXITY}",
    "score_attempts_subdir": HARDENED_SCORE_ATTEMPTS_SUBDIR,
    "export_output_reuse": "forbidden",
    "score_output_reuse": "forbidden",
}

HARDENED_GENERATION = {
    "seed": 42,
    "seed_strategy": "constant_per_record",
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "negative_prompt": " ",
    "num_images_per_prompt": 1,
    "preserve_input_resolution": False,
}


class ComplexEditContractError(RuntimeError):
    """The hardened real/C4/531 protocol cannot be established exactly."""


def _positive_gpu_count(value: str) -> bool:
    for token in str(value).replace(",", " ").split():
        token = re.sub(r"^(?:AllocTRES|TRES|Gres)=", "", token)
        matches = (
            re.fullmatch(r"gres/gpu(?::[^=:\s]+)?=(\d+)", token),
            re.fullmatch(r"gres/gpu(?::[^:\s]+)?:(\d+)", token),
            re.fullmatch(r"gpu(?::[^:\s]+)?:(\d+)", token),
            re.fullmatch(r"(\d+)", token),
        )
        for match in matches:
            if match is not None and int(match.group(1)) >= 1:
                return True
    return False


def _step_has_gpu_evidence(environment: Mapping[str, str]) -> bool:
    step_gpus = str(environment.get("SLURM_STEP_GPUS", "")).strip().lower()
    if step_gpus not in {"", "(null)", "n/a", "none", "nodevfiles"}:
        return True
    return any(
        _positive_gpu_count(str(environment.get(name, "")))
        for name in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK", "SLURM_TRES_PER_TASK")
    )


def _default_scheduler_query(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _scheduler_confirms_gpu_job(
    job_id: str,
    current_user: str,
    *,
    query: Any,
) -> bool:
    try:
        result = query(["scontrol", "show", "job", "-o", job_id])
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0 and str(result.stdout).strip():
        fields: dict[str, str] = {}
        for token in str(result.stdout).split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        scheduler_user = fields.get("UserId", "").split("(", 1)[0]
        return (
            scheduler_user == current_user
            and fields.get("JobState") == "RUNNING"
            and _positive_gpu_count(fields.get("AllocTRES", ""))
        )

    try:
        result = query(["squeue", "-h", "-j", job_id, "-o", "%u|%T|%b"])
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    record = next((line.strip() for line in str(result.stdout).splitlines() if line.strip()), "")
    parts = record.split("|")
    return (
        len(parts) == 3
        and parts[0] == current_user
        and parts[1] == "RUNNING"
        and _positive_gpu_count(parts[2])
    )


def require_hardened_experiment_runtime(
    config: Mapping[str, Any],
    *,
    workload: str,
    environment: Mapping[str, str] | None = None,
    executable: str | Path | None = None,
    runtime_prefix: str | Path | None = None,
    current_user: str | None = None,
    scheduler_query: Any | None = None,
) -> None:
    """Fail closed unless a hardened run is in qedit and an owned GPU srun step."""

    if environment is None:
        from accv_v1.runtime import require_gpu_step
        require_gpu_step()
    environment = os.environ if environment is None else environment
    executable = sys.executable if executable is None else str(executable)
    runtime_prefix = sys.prefix if runtime_prefix is None else str(runtime_prefix)
    current_user = pwd.getpwuid(os.geteuid()).pw_name if current_user is None else current_user
    scheduler_query = _default_scheduler_query if scheduler_query is None else scheduler_query
    configured_python = config.get("runtime", {}).get("python_executable")
    if configured_python != HARDENED_PYTHON_EXECUTABLE:
        raise ComplexEditContractError(
            f"Refusing {workload}: runtime.python_executable must equal "
            f"{HARDENED_PYTHON_EXECUTABLE}"
        )
    if Path(executable).resolve() != Path(HARDENED_PYTHON_EXECUTABLE).resolve() or (
        Path(runtime_prefix).resolve() != Path(HARDENED_CONDA_PREFIX).resolve()
    ):
        raise ComplexEditContractError(
            f"Refusing {workload}: use the pinned qedit runtime {HARDENED_CONDA_PREFIX}"
        )
    if current_user != pwd.getpwuid(os.geteuid()).pw_name:
        raise ComplexEditContractError(
            f"Refusing {workload}: project experiments must run as ritesh_thawkar"
        )
    job_id = str(environment.get("SLURM_JOB_ID", "")).strip()
    if not job_id:
        raise ComplexEditContractError(f"Refusing {workload} outside an active Slurm allocation")
    step_id = str(environment.get("SLURM_STEP_ID", "")).strip()
    if not step_id.isdigit():
        raise ComplexEditContractError(
            f"Refusing {workload}: the current shell is not inside a numeric srun step"
        )
    if not _step_has_gpu_evidence(environment):
        raise ComplexEditContractError(
            f"Refusing {workload}: srun step {job_id}.{step_id} has no step-local GPU evidence"
        )
    if not _scheduler_confirms_gpu_job(
        job_id,
        current_user,
        query=scheduler_query,
    ):
        raise ComplexEditContractError(
            f"Refusing {workload}: job {job_id} is not an owned RUNNING GPU allocation"
        )


def validate_hardened_config(config: Mapping[str, Any]) -> None:
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    scoring = config.get("scoring", {})
    output = config.get("output", {})
    contract = config.get("evaluation_contract", {})
    runtime = config.get("runtime", {})
    generation = config.get("generation", {})
    prompting = config.get("prompting", {})
    errors: list[str] = []
    if model.get("backend") != HARDENED_MODEL_BACKEND:
        errors.append(f"model.backend must equal {HARDENED_MODEL_BACKEND}")
    if model.get("base_model") != HARDENED_BASE_MODEL:
        errors.append(f"model.base_model must equal {HARDENED_BASE_MODEL}")
    if model.get("revision") != HARDENED_BASE_MODEL_REVISION:
        errors.append(
            f"model.revision must equal the recovered paper snapshot {HARDENED_BASE_MODEL_REVISION}"
        )
    if model.get("model_type") not in {"base", "lora"}:
        errors.append("model.model_type must equal base or lora")
    if model.get("torch_dtype") != "bfloat16":
        errors.append("model.torch_dtype must equal bfloat16")
    model_name = model.get("model_name")
    if (
        not isinstance(model_name, str)
        or not model_name
        or Path(model_name).name != model_name
        or model_name in {".", ".."}
    ):
        errors.append("model.model_name must be a non-empty path-safe name")
    if model.get("model_type") == "base" and model.get("checkpoint_path") is not None:
        errors.append("base evaluation must not configure model.checkpoint_path")
    if model.get("model_type") == "base" and model.get("lora_scale") is not None:
        errors.append("base evaluation must not configure model.lora_scale")
    if model.get("model_type") == "lora" and not model.get("checkpoint_path"):
        errors.append("lora evaluation requires model.checkpoint_path")
    lora_scale = model.get("lora_scale")
    if (
        model.get("model_type") == "lora"
        and lora_scale is not None
        and (
            isinstance(lora_scale, bool)
            or not isinstance(lora_scale, (int, float))
            or not math.isfinite(float(lora_scale))
            or float(lora_scale) <= 0.0
        )
    ):
        errors.append("model.lora_scale must be a finite positive number when configured")
    if dataset.get("source") != "huggingface":
        errors.append("dataset.source must equal huggingface")
    if dataset.get("dataset_name") != HARDENED_DATASET_NAME:
        errors.append(f"dataset.dataset_name must equal {HARDENED_DATASET_NAME}")
    if dataset.get("revision") != HARDENED_DATASET_REVISION:
        errors.append(
            "dataset.revision must equal the recovered benchmark snapshot "
            f"{HARDENED_DATASET_REVISION}"
        )
    if dataset.get("expected_manifest_sha256") != HARDENED_MANIFEST_SHA256:
        errors.append("dataset.expected_manifest_sha256 differs from the recovered snapshot")
    if dataset.get("expected_selection_sha256") != HARDENED_SELECTION_SHA256:
        errors.append("dataset.expected_selection_sha256 differs from the recovered snapshot")
    if dataset.get("expected_images_tree_sha256") != HARDENED_IMAGES_TREE_SHA256:
        errors.append("dataset.expected_images_tree_sha256 differs from the recovered snapshot")
    if str(dataset.get("split")) != HARDENED_SPLIT:
        errors.append(f"dataset.split must equal {HARDENED_SPLIT}")
    complexities = dataset.get("complexities")
    if complexities != [HARDENED_COMPLEXITY]:
        errors.append(f"dataset.complexities must equal [{HARDENED_COMPLEXITY}]")
    if str(scoring.get("image_type")) != HARDENED_SPLIT:
        errors.append(f"scoring.image_type must equal {HARDENED_SPLIT}")
    if scoring.get("n") != HARDENED_SCORING_N:
        errors.append(f"scoring.n must equal {HARDENED_SCORING_N}")
    if scoring.get("m") != HARDENED_SCORING_M:
        errors.append(f"scoring.m must equal {HARDENED_SCORING_M}")
    if scoring.get("num_processes") != HARDENED_NUM_PROCESSES:
        errors.append(f"scoring.num_processes must equal {HARDENED_NUM_PROCESSES}")
    if scoring.get("resume") is not False:
        errors.append("scoring.resume must be false")
    if scoring.get("fresh_only") is not True:
        errors.append("scoring.fresh_only must be true")
    if scoring.get("scorer_repo") != HARDENED_SCORER_REPO:
        errors.append(f"scoring.scorer_repo must equal {HARDENED_SCORER_REPO}")
    if scoring.get("worker_module") != HARDENED_SCORE_WORKER_MODULE:
        errors.append(
            f"scoring.worker_module must equal {HARDENED_SCORE_WORKER_MODULE}"
        )
    if scoring.get("eval_model") != HARDENED_EVAL_MODEL:
        errors.append(f"scoring.eval_model must equal {HARDENED_EVAL_MODEL}")
    if scoring.get("openai_model") != HARDENED_EVAL_MODEL:
        errors.append(f"scoring.openai_model must equal {HARDENED_EVAL_MODEL}")
    if scoring.get("expected_openai_model") != HARDENED_EVAL_MODEL:
        errors.append(f"scoring.expected_openai_model must equal {HARDENED_EVAL_MODEL}")
    if isinstance(scoring.get("temperature"), bool) or scoring.get("temperature") != 1.0:
        errors.append("scoring.temperature must equal the scorer's fixed value 1.0")
    retry_policy = {
        "request_timeout_seconds": HARDENED_REQUEST_TIMEOUT_SECONDS,
        "sdk_max_retries": HARDENED_SDK_MAX_RETRIES,
        "request_max_attempts": HARDENED_REQUEST_MAX_ATTEMPTS,
        "retry_initial_backoff_seconds": HARDENED_RETRY_INITIAL_BACKOFF_SECONDS,
        "retry_max_backoff_seconds": HARDENED_RETRY_MAX_BACKOFF_SECONDS,
    }
    for field, expected in retry_policy.items():
        value = scoring.get(field)
        if isinstance(expected, int):
            matches = type(value) is int and value == expected
        else:
            matches = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) == expected
            )
        if not matches:
            errors.append(f"scoring.{field} must equal {expected!r}")
    if scoring.get("backbone") != "openai_structured":
        errors.append("scoring.backbone must equal openai_structured")
    if scoring.get("strict_response_parser") is not True:
        errors.append("scoring.strict_response_parser must be true")
    if scoring.get("allow_custom_openai_base_url") is not False:
        errors.append("scoring.allow_custom_openai_base_url must be false")
    if output.get("allow_resume_mismatch") is not False:
        errors.append("output.allow_resume_mismatch must be false")
    if output.get("score_attempts_subdir") != HARDENED_SCORE_ATTEMPTS_SUBDIR:
        errors.append(
            f"output.score_attempts_subdir must equal {HARDENED_SCORE_ATTEMPTS_SUBDIR}"
        )
    if output.get("score_log_filename") != HARDENED_SCORE_LOG_FILENAME:
        errors.append(f"output.score_log_filename must equal {HARDENED_SCORE_LOG_FILENAME}")
    configured_scores_dir = output.get("scores_dir")
    configured_summary_path = output.get("summary_path")
    if not isinstance(configured_scores_dir, str) or not configured_scores_dir.strip():
        errors.append("output.scores_dir must be an explicit non-empty path")
    if not isinstance(configured_summary_path, str) or not configured_summary_path.strip():
        errors.append("output.summary_path must be an explicit non-empty path")
    elif isinstance(configured_scores_dir, str) and (
        Path(configured_summary_path).parent != Path(configured_scores_dir)
    ):
        errors.append("output.summary_path parent must equal output.scores_dir")
    if not isinstance(generation, Mapping) or canonical_json_bytes(dict(generation)) != (
        canonical_json_bytes(HARDENED_GENERATION)
    ):
        errors.append("generation must exactly match the recovered paper protocol")
    if prompting != {"use_prompt_polish": False}:
        errors.append("prompting must exactly disable prompt polishing")
    if runtime.get("generation_device") != "cuda":
        errors.append("runtime.generation_device must equal cuda")
    if runtime.get("python_executable") != HARDENED_PYTHON_EXECUTABLE:
        errors.append(
            "runtime.python_executable must equal the pinned qedit interpreter "
            f"{HARDENED_PYTHON_EXECUTABLE}"
        )
    runtime_provenance = contract.get("runtime_provenance", {})
    if (
        not isinstance(runtime_provenance, Mapping)
        or runtime_provenance.get("require_accelerator") is not True
    ):
        errors.append("evaluation_contract.runtime_provenance must require an accelerator")
    configured_protocol = contract.get("benchmark_protocol")
    if not isinstance(configured_protocol, Mapping) or canonical_json_bytes(
        dict(configured_protocol)
    ) != canonical_json_bytes(BENCHMARK_PROTOCOL):
        errors.append("evaluation_contract.benchmark_protocol differs from the C4/531 contract")
    if errors:
        raise ComplexEditContractError(
            "Invalid hardened Complex-Edit configuration:\n- " + "\n- ".join(errors)
        )


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ComplexEditContractError(f"Duplicate JSON key {key!r}")
        value[key] = item
    return value


def load_json_strict(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ComplexEditContractError(f"JSON artifact must not be a symlink: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ComplexEditContractError(f"Non-finite JSON number {token!r}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComplexEditContractError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComplexEditContractError(f"JSON artifact must be an object: {path}")
    return value


def _instruction_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_hardened_c4_dataset(
    manifest_path: Path,
    images_root: Path,
    *,
    enforce_recovered_snapshot: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and authenticate the exact local real/C4/531 dataset materialization."""

    manifest_path = Path(manifest_path)
    images_root = Path(images_root)
    if manifest_path.is_symlink() or images_root.is_symlink():
        raise ComplexEditContractError("Complex-Edit manifest/images_root must not be symlinks")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Complex-Edit manifest is missing: {manifest_path}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Complex-Edit images root is missing: {images_root}")

    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ComplexEditContractError(f"Complex-Edit manifest line {line_number} is blank")
            try:
                row = json.loads(line, object_pairs_hook=_duplicate_rejecting_object)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ComplexEditContractError(
                    f"Complex-Edit manifest line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, dict) or set(row) != {
                "idx",
                "key",
                "image",
                "compound",
                "sequence",
            }:
                raise ComplexEditContractError(
                    f"Complex-Edit manifest line {line_number} has an unknown schema"
                )
            expected_idx = line_number - 1
            expected_key = f"{expected_idx:04d}"
            if isinstance(row["idx"], bool) or row["idx"] != expected_idx:
                raise ComplexEditContractError(
                    f"Complex-Edit manifest line {line_number} has non-canonical idx"
                )
            if row["key"] != expected_key or row["image"] != f"{expected_key}.png":
                raise ComplexEditContractError(
                    f"Complex-Edit manifest line {line_number} has non-canonical key/image"
                )
            compound = row["compound"]
            sequence = row["sequence"]
            if (
                not isinstance(compound, list)
                or len(compound) != 8
                or any(not isinstance(item, str) or not item.strip() for item in compound)
            ):
                raise ComplexEditContractError(
                    f"Complex-Edit manifest line {line_number} requires 8 instructions"
                )
            if not isinstance(sequence, list) or any(
                not isinstance(item, str) for item in sequence
            ):
                raise ComplexEditContractError(
                    f"Complex-Edit manifest line {line_number} has invalid sequence"
                )
            rows.append(row)
    if len(rows) != HARDENED_SAMPLE_COUNT:
        raise ComplexEditContractError(
            f"Complex-Edit real/C4 contract requires {HARDENED_SAMPLE_COUNT} rows, "
            f"observed {len(rows)}"
        )

    expected_names = {str(row["image"]) for row in rows}
    actual_names: set[str] = set()
    for entry in images_root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ComplexEditContractError(
                f"Complex-Edit source tree contains a symlink or non-file: {entry}"
            )
        actual_names.add(entry.name)
    if actual_names != expected_names:
        raise ComplexEditContractError(
            "Complex-Edit source-image identities differ from the exact manifest: "
            f"missing={sorted(expected_names - actual_names)[:10]}, "
            f"unexpected={sorted(actual_names - expected_names)[:10]}"
        )

    contract_records: list[dict[str, Any]] = []
    for row in rows:
        key = str(row["key"])
        image_path = images_root / str(row["image"])
        instruction = str(row["compound"][HARDENED_COMPLEXITY - 1])
        contract_records.append(
            {
                "identity": f"{HARDENED_SPLIT}:c{HARDENED_COMPLEXITY}:{key}",
                "idx": int(row["idx"]),
                "key": key,
                "image": str(row["image"]),
                "complexity": HARDENED_COMPLEXITY,
                "instruction_sha256": _instruction_sha256(instruction),
                "source_image_sha256": sha256_file(image_path),
            }
        )
    if enforce_recovered_snapshot:
        manifest_sha256 = sha256_file(manifest_path)
        if manifest_sha256 != HARDENED_MANIFEST_SHA256:
            raise ComplexEditContractError(
                "Complex-Edit manifest bytes differ from the recovered pinned snapshot: "
                f"observed={manifest_sha256}, expected={HARDENED_MANIFEST_SHA256}"
            )
        selection_sha256 = hashlib.sha256(canonical_json_bytes(contract_records)).hexdigest()
        if selection_sha256 != HARDENED_SELECTION_SHA256:
            raise ComplexEditContractError(
                "Complex-Edit C4 instructions or source-image bytes differ from the "
                "recovered pinned snapshot: "
                f"observed={selection_sha256}, expected={HARDENED_SELECTION_SHA256}"
            )
    return rows, contract_records


def expected_export_paths(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [f"{HARDENED_SPLIT}_c{HARDENED_COMPLEXITY}/{row['key']}.png" for row in rows]


def validate_contract_still_current(
    config: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Re-fingerprint durable inputs/code while retaining the launch runtime receipt."""

    body = expected_contract.get("contract")
    if not isinstance(body, Mapping) or not isinstance(body.get("runtime_provenance"), Mapping):
        raise ComplexEditContractError("Expected evaluation contract envelope is invalid")
    rebuilt = build_evaluation_contract(
        config,
        benchmark="complex_edit",
        records=records,
        runtime_provenance=body["runtime_provenance"],
    )
    if rebuilt != expected_contract:
        raise ComplexEditContractError(
            "Hardened Complex-Edit inputs, checkpoint, or implementation changed "
            "while the evaluation stage was running"
        )


def require_fresh_export_namespace(output_root: Path) -> None:
    allowed = {CONTRACT_FILENAME}
    output_root = Path(output_root)
    if output_root.is_symlink() or not output_root.is_dir():
        raise ComplexEditContractError(
            f"Hardened export root must be a regular directory: {output_root}"
        )
    entries = list(output_root.iterdir())
    existing = {item.name for item in entries}
    unexpected = sorted(existing - allowed)
    if unexpected:
        raise ComplexEditContractError(
            "Hardened Complex-Edit export never resumes or adopts existing artifacts; "
            f"unexpected entries in {output_root}: {unexpected[:10]}"
        )
    for item in entries:
        if item.is_symlink() or not item.is_file():
            raise ComplexEditContractError(
                f"Hardened export contract entry must be a regular file: {item}"
            )


def validate_exact_export(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    output_root = Path(output_root)
    expected = set(expected_export_paths(rows))
    complexity_dir_name = f"{HARDENED_SPLIT}_c{HARDENED_COMPLEXITY}"
    allowed_root_entries = {
        CONTRACT_FILENAME,
        ".export_provenance.json",
        OUTPUT_MANIFEST_FILENAME,
        complexity_dir_name,
    }
    if output_root.is_symlink() or not output_root.is_dir():
        raise ComplexEditContractError(
            f"Hardened export root must be a regular directory: {output_root}"
        )
    root_items = list(output_root.iterdir())
    root_entries = {item.name for item in root_items}
    required_root_entries = {
        CONTRACT_FILENAME,
        ".export_provenance.json",
        complexity_dir_name,
    }
    missing_root = sorted(required_root_entries - root_entries)
    unexpected_root = sorted(root_entries - allowed_root_entries)
    if missing_root or unexpected_root:
        raise ComplexEditContractError(
            "Hardened Complex-Edit export root layout differs: "
            f"missing={missing_root[:10]}, unexpected={unexpected_root[:10]}"
        )
    for item in root_items:
        if item.name == complexity_dir_name:
            valid = not item.is_symlink() and item.is_dir()
        else:
            valid = not item.is_symlink() and item.is_file()
        if not valid:
            raise ComplexEditContractError(f"Hardened export root contains an unsafe entry: {item}")
    output_dir = output_root / complexity_dir_name
    actual: set[str] = set()
    for path in output_dir.iterdir() if output_dir.is_dir() else ():
        if path.is_symlink() or not path.is_file():
            raise ComplexEditContractError(f"Exported image must be a regular file: {path}")
        actual.add(path.relative_to(output_root).as_posix())
    if actual != expected:
        raise ComplexEditContractError(
            "Hardened Complex-Edit export is incomplete or contaminated: "
            f"missing={sorted(expected - actual)[:10]}, "
            f"unexpected={sorted(actual - expected)[:10]}"
        )
    return sorted(expected)


def require_fresh_score_namespace(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require the immutable export group to contain only contracted PNGs."""

    expected_pngs = {f"{row['key']}.png" for row in rows}
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ComplexEditContractError(
            f"Hardened score image root must be a regular directory: {output_dir}"
        )
    entries = list(output_dir.iterdir())
    actual = {item.name for item in entries}
    if actual != expected_pngs:
        raise ComplexEditContractError(
            "Hardened Complex-Edit scoring never resumes or reuses score artifacts; "
            f"missing_images={sorted(expected_pngs - actual)[:10]}, "
            f"unexpected_entries={sorted(actual - expected_pngs)[:10]}"
        )
    for item in entries:
        if item.is_symlink() or not item.is_file():
            raise ComplexEditContractError(f"Hardened score input must be a regular file: {item}")


def require_fresh_score_attempt_namespace(
    score_output_dir: Path,
    *,
    attempt_id: str,
    contract_id: str,
    log_filename: str = HARDENED_SCORE_LOG_FILENAME,
) -> None:
    """Authenticate a newly claimed active score-output directory.

    The runner opens its attempt-local tee log immediately before starting the
    worker, so that one optional empty/partial log is the only entry allowed in
    addition to the immutable attempt manifest.  Generated PNGs are deliberately
    absent: the worker reads them from the separate export namespace.
    """

    attempt_id = str(attempt_id).strip()
    contract_id = str(contract_id).strip()
    if SCORE_ATTEMPT_ID_PATTERN.fullmatch(attempt_id) is None or ".." in attempt_id:
        raise ComplexEditContractError("Invalid Complex-Edit score attempt ID")
    if re.fullmatch(r"[0-9a-f]{64}", contract_id) is None:
        raise ComplexEditContractError("Invalid Complex-Edit score contract ID")
    if log_filename != HARDENED_SCORE_LOG_FILENAME:
        raise ComplexEditContractError("Complex-Edit score log filename differs from contract")

    score_output_dir = Path(score_output_dir)
    expected_parent = score_output_dir.parent
    if (
        score_output_dir.name != attempt_id
        or expected_parent.name != "active"
        or expected_parent.parent.name != HARDENED_SCORE_ATTEMPTS_SUBDIR
    ):
        raise ComplexEditContractError(
            "Complex-Edit score output is outside its contracted active attempt namespace"
        )
    if score_output_dir.is_symlink() or not score_output_dir.is_dir():
        raise ComplexEditContractError(
            f"Complex-Edit active score attempt must be a regular directory: {score_output_dir}"
        )
    allowed = {SCORE_ATTEMPT_MANIFEST_FILENAME, log_filename}
    entries = list(score_output_dir.iterdir())
    unexpected = sorted(item.name for item in entries if item.name not in allowed)
    if unexpected:
        raise ComplexEditContractError(
            "Complex-Edit scoring never resumes an active attempt; "
            f"unexpected entries={unexpected[:10]}"
        )
    for item in entries:
        if item.is_symlink() or not item.is_file():
            raise ComplexEditContractError(
                f"Complex-Edit active score-attempt entry must be a regular file: {item}"
            )
    manifest = load_json_strict(score_output_dir / SCORE_ATTEMPT_MANIFEST_FILENAME)
    expected_manifest = {
        "schema": SCORE_ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "benchmark": "complex_edit",
        "contract_id": contract_id,
    }
    if manifest != expected_manifest:
        raise ComplexEditContractError(
            "Complex-Edit active score-attempt manifest differs from its contract"
        )


def _finite_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComplexEditContractError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 10.0:
        raise ComplexEditContractError(f"{label} must be finite and in [0, 10]")
    return number


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComplexEditContractError(f"{label} must be a non-negative integer")
    return value


def _validate_provider_responses(
    value: object,
    *,
    label: str,
) -> list[dict[str, Any]]:
    expected_choice_counts = [
        min(HARDENED_SCORING_M, HARDENED_SCORING_N),
        HARDENED_SCORING_N - min(HARDENED_SCORING_M, HARDENED_SCORING_N),
    ]
    expected_choice_counts = [count for count in expected_choice_counts if count]
    if not isinstance(value, list) or len(value) != len(expected_choice_counts):
        raise ComplexEditContractError(
            f"{label} must preserve {len(expected_choice_counts)} provider responses"
        )
    records: list[dict[str, Any]] = []
    expected_keys = {
        "requested_model",
        "provider_model",
        "response_id",
        "request_id",
        "created",
        "system_fingerprint",
        "service_tier",
        "choice_count",
        "usage",
    }
    for index, (record, choice_count) in enumerate(zip(value, expected_choice_counts, strict=True)):
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise ComplexEditContractError(f"{label} response {index} schema differs")
        for key in ("provider_model", "response_id", "request_id"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise ComplexEditContractError(f"{label} response {index} has no provider {key}")
        if record["requested_model"] != HARDENED_EVAL_MODEL:
            raise ComplexEditContractError(f"{label} response {index} requested model differs")
        if (
            isinstance(record["created"], bool)
            or not isinstance(record["created"], int)
            or record["created"] <= 0
        ):
            raise ComplexEditContractError(f"{label} response {index} has invalid creation time")
        for optional_key in ("system_fingerprint", "service_tier"):
            optional_value = record[optional_key]
            if optional_value is not None and (
                not isinstance(optional_value, str) or not optional_value.strip()
            ):
                raise ComplexEditContractError(
                    f"{label} response {index} has invalid {optional_key}"
                )
        if record["choice_count"] != choice_count:
            raise ComplexEditContractError(f"{label} response {index} choice count differs")
        usage = record["usage"]
        if not isinstance(usage, dict):
            raise ComplexEditContractError(f"{label} response {index} has no usage object")
        prompt_tokens = _nonnegative_integer(
            usage.get("prompt_tokens"), f"{label} response {index} prompt_tokens"
        )
        completion_tokens = _nonnegative_integer(
            usage.get("completion_tokens"),
            f"{label} response {index} completion_tokens",
        )
        total_tokens = _nonnegative_integer(
            usage.get("total_tokens"), f"{label} response {index} total_tokens"
        )
        if total_tokens != prompt_tokens + completion_tokens:
            raise ComplexEditContractError(
                f"{label} response {index} usage totals are inconsistent"
            )
        records.append(dict(record))
    return records


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9):
        raise ComplexEditContractError(f"{label} differs: observed={observed}, expected={expected}")


def _validate_result_directory(directory: Path, keys: Sequence[str]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ComplexEditContractError(
            f"Complex-Edit score directory must be a regular directory: {directory}"
        )
    expected = {f"{key}.json" for key in keys} | {"final_result.json", "log.txt"}
    actual = {item.name for item in directory.iterdir()}
    if actual != expected:
        raise ComplexEditContractError(
            f"Complex-Edit score directory {directory} is incomplete or contaminated: "
            f"missing={sorted(expected - actual)[:10]}, "
            f"unexpected={sorted(actual - expected)[:10]}"
        )
    for item in directory.iterdir():
        if item.is_symlink() or not item.is_file():
            raise ComplexEditContractError(f"Score artifact must be a regular file: {item}")


def validate_score_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact per-key scorer outputs and recompute all reported metrics."""

    output_dir = Path(output_dir)
    keys = [str(row["key"]) for row in rows]
    instructions = {str(row["key"]): str(row["compound"][HARDENED_COMPLEXITY - 1]) for row in rows}
    for name in SCORE_DIRS:
        _validate_result_directory(output_dir / name, keys)

    aggregate: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    provider_responses: list[dict[str, Any]] = []
    for key in keys:
        alignment = load_json_strict(output_dir / ALIGNMENT_DIR / f"{key}.json")
        quality = load_json_strict(output_dir / QUALITY_DIR / f"{key}.json")
        overall = load_json_strict(output_dir / OVERALL_DIR / f"{key}.json")
        if set(alignment) != {
            "instruction",
            "runs",
            "provider_responses",
            "instruction_following",
            "identity_preservation",
        }:
            raise ComplexEditContractError(f"Alignment score schema differs for key {key}")
        if set(quality) != {
            "instruction",
            "runs",
            "provider_responses",
            "perceptual_quality",
        }:
            raise ComplexEditContractError(f"Quality score schema differs for key {key}")
        if set(overall) != {"instruction", *METRIC_KEYS}:
            raise ComplexEditContractError(f"Overall score schema differs for key {key}")
        instruction = instructions[key]
        if (
            alignment["instruction"] != instruction
            or quality["instruction"] != instruction
            or overall["instruction"] != instruction
        ):
            raise ComplexEditContractError(f"Instruction identity differs for key {key}")

        alignment_runs = alignment["runs"]
        quality_runs = quality["runs"]
        provider_responses.extend(
            _validate_provider_responses(
                alignment["provider_responses"],
                label=f"{key} alignment",
            )
        )
        provider_responses.extend(
            _validate_provider_responses(
                quality["provider_responses"],
                label=f"{key} quality",
            )
        )
        if not isinstance(alignment_runs, list) or len(alignment_runs) != HARDENED_SCORING_N:
            raise ComplexEditContractError(f"Alignment run count differs for key {key}")
        if not isinstance(quality_runs, list) or len(quality_runs) != HARDENED_SCORING_N:
            raise ComplexEditContractError(f"Quality run count differs for key {key}")
        for index, run in enumerate(alignment_runs):
            if (
                not isinstance(run, dict)
                or set(run)
                != {
                    "reasoning",
                    "instruction_following",
                    "identity_preservation",
                }
                or not isinstance(run["reasoning"], str)
            ):
                raise ComplexEditContractError(
                    f"Alignment run schema differs for key {key}, run {index}"
                )
        for index, run in enumerate(quality_runs):
            if not isinstance(run, dict) or set(run) != {"perceptual_quality"}:
                raise ComplexEditContractError(
                    f"Quality run schema differs for key {key}, run {index}"
                )

        instruction_following = _finite_score(
            alignment["instruction_following"], f"{key} instruction_following"
        )
        identity_preservation = _finite_score(
            alignment["identity_preservation"], f"{key} identity_preservation"
        )
        perceptual_quality = _finite_score(
            quality["perceptual_quality"], f"{key} perceptual_quality"
        )
        _assert_close(
            instruction_following,
            _mean(
                [
                    _finite_score(run["instruction_following"], f"{key} alignment run")
                    for run in alignment_runs
                ]
            ),
            f"{key} instruction_following average",
        )
        _assert_close(
            identity_preservation,
            _mean(
                [
                    _finite_score(run["identity_preservation"], f"{key} alignment run")
                    for run in alignment_runs
                ]
            ),
            f"{key} identity_preservation average",
        )
        _assert_close(
            perceptual_quality,
            _mean(
                [
                    _finite_score(run["perceptual_quality"], f"{key} quality run")
                    for run in quality_runs
                ]
            ),
            f"{key} perceptual_quality average",
        )
        expected_values = {
            "instruction_following": instruction_following,
            "identity_preservation": identity_preservation,
            "perceptual_quality": perceptual_quality,
        }
        for metric, expected in expected_values.items():
            _assert_close(
                _finite_score(overall[metric], f"{key} overall {metric}"),
                expected,
                f"{key} overall {metric}",
            )
        overall_value = _finite_score(overall["overall"], f"{key} overall")
        _assert_close(
            overall_value,
            _mean(list(expected_values.values())),
            f"{key} overall mean",
        )
        for metric, value in {**expected_values, "overall": overall_value}.items():
            aggregate[metric].append(value)

    average = {metric: round(_mean(values), 4) for metric, values in aggregate.items()}
    final_alignment = load_json_strict(output_dir / ALIGNMENT_DIR / "final_result.json")
    final_quality = load_json_strict(output_dir / QUALITY_DIR / "final_result.json")
    final_overall = load_json_strict(output_dir / OVERALL_DIR / "final_result.json")
    if set(final_alignment) != {"instruction_following", "identity_preservation"}:
        raise ComplexEditContractError("Final alignment score schema differs")
    if set(final_quality) != {"perceptual_quality"}:
        raise ComplexEditContractError("Final quality score schema differs")
    if set(final_overall) != set(METRIC_KEYS):
        raise ComplexEditContractError("Final overall score schema differs")
    for metric in ("instruction_following", "identity_preservation"):
        _assert_close(
            _finite_score(final_alignment[metric], f"final alignment {metric}"),
            _mean(aggregate[metric]),
            f"final alignment {metric}",
        )
    _assert_close(
        _finite_score(final_quality["perceptual_quality"], "final quality"),
        _mean(aggregate["perceptual_quality"]),
        "final perceptual_quality",
    )
    for metric in METRIC_KEYS:
        _assert_close(
            _finite_score(final_overall[metric], f"final overall {metric}"),
            _mean(aggregate[metric]),
            f"final overall {metric}",
        )
    expected_provider_responses = (
        HARDENED_SAMPLE_COUNT * 2 * math.ceil(HARDENED_SCORING_N / HARDENED_SCORING_M)
    )
    if len(keys) != HARDENED_SAMPLE_COUNT:
        expected_provider_responses = (
            len(keys) * 2 * math.ceil(HARDENED_SCORING_N / HARDENED_SCORING_M)
        )
    if len(provider_responses) != expected_provider_responses:
        raise ComplexEditContractError(
            "Complex-Edit provider response count differs: "
            f"observed={len(provider_responses)}, expected={expected_provider_responses}"
        )
    provider_models = sorted({str(record["provider_model"]) for record in provider_responses})
    if len(provider_models) != 1:
        raise ComplexEditContractError(
            f"Complex-Edit provider model identity changed within the score run: {provider_models}"
        )
    response_ids = [str(record["response_id"]) for record in provider_responses]
    request_ids = [str(record["request_id"]) for record in provider_responses]
    if len(set(response_ids)) != len(response_ids):
        raise ComplexEditContractError("Complex-Edit provider response IDs are not unique")
    if len(set(request_ids)) != len(request_ids):
        raise ComplexEditContractError("Complex-Edit provider request IDs are not unique")
    usage_totals = {
        field: sum(int(record["usage"][field]) for record in provider_responses)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if usage_totals["total_tokens"] != (
        usage_totals["prompt_tokens"] + usage_totals["completion_tokens"]
    ):
        raise ComplexEditContractError("Complex-Edit aggregate provider usage is inconsistent")
    system_fingerprints = sorted(
        {
            str(record["system_fingerprint"])
            for record in provider_responses
            if record["system_fingerprint"] is not None
        }
    )
    service_tiers = sorted(
        {
            str(record["service_tier"])
            for record in provider_responses
            if record["service_tier"] is not None
        }
    )
    judge_provenance = {
        "schema": JUDGE_PROVENANCE_SCHEMA,
        "requested_model": HARDENED_EVAL_MODEL,
        "provider_models": provider_models,
        "response_count": len(provider_responses),
        "response_ids_sha256": hashlib.sha256(canonical_json_bytes(response_ids)).hexdigest(),
        "request_ids_sha256": hashlib.sha256(canonical_json_bytes(request_ids)).hexdigest(),
        "provider_responses_sha256": hashlib.sha256(
            canonical_json_bytes(provider_responses)
        ).hexdigest(),
        "created_min": min(int(record["created"]) for record in provider_responses),
        "created_max": max(int(record["created"]) for record in provider_responses),
        "system_fingerprints": system_fingerprints,
        "service_tiers": service_tiers,
        "usage_totals": usage_totals,
    }
    return {
        "count": len(keys),
        "average": average,
        "groups": {
            f"{HARDENED_SPLIT}_c{HARDENED_COMPLEXITY}": {
                **average,
                "count": len(keys),
            }
        },
        "judge_provenance": judge_provenance,
    }


def build_score_receipt(
    output_dir: Path,
    *,
    contract_id: str,
    export_manifest_sha256: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for directory_name in SCORE_DIRS:
        directory = Path(output_dir) / directory_name
        for path in sorted(directory.glob("*.json")):
            relative = path.relative_to(output_dir).as_posix()
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema": SCORE_RECEIPT_SCHEMA,
        "contract_id": contract_id,
        "export_output_manifest_sha256": export_manifest_sha256,
        "file_count": len(files),
        "files": files,
        "files_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
        "metrics": dict(metrics),
        "judge_provenance": dict(metrics["judge_provenance"]),
    }


def validate_score_attempt_for_promotion(
    score_output_dir: Path,
    *,
    completed_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    attempt_id: str,
    contract: Mapping[str, Any],
    run_model_name: str,
    export_output_dir: Path,
    export_manifest_sha256: str,
    scorer_repo: Path,
    receipt_filename: str,
    summary_filename: str,
    expected_summary: Mapping[str, Any],
    log_filename: str = HARDENED_SCORE_LOG_FILENAME,
) -> dict[str, Any]:
    """Revalidate a complete active C4 attempt immediately before publication."""

    score_output_dir = Path(score_output_dir)
    completed_dir = Path(completed_dir)
    export_output_dir = Path(export_output_dir)
    scorer_repo = Path(scorer_repo)
    contract_id = str(contract.get("contract_id", ""))
    if re.fullmatch(r"[0-9a-f]{64}", contract_id) is None:
        raise ComplexEditContractError("Promotion contract ID is invalid")
    if (
        score_output_dir.name != attempt_id
        or score_output_dir.parent.name != "active"
        or score_output_dir.parent.parent.name != HARDENED_SCORE_ATTEMPTS_SUBDIR
        or completed_dir.name != attempt_id
        or completed_dir.parent.name != "complete"
        or completed_dir.parent.parent != score_output_dir.parent.parent
    ):
        raise ComplexEditContractError("Score attempt promotion paths differ from the contract")
    if completed_dir.exists() or completed_dir.is_symlink():
        raise ComplexEditContractError(
            f"Completed Complex-Edit attempt already exists: {completed_dir}"
        )
    if score_output_dir.is_symlink() or not score_output_dir.is_dir():
        raise ComplexEditContractError("Active Complex-Edit score attempt is unsafe")

    expected_root_entries = {
        SCORE_ATTEMPT_MANIFEST_FILENAME,
        log_filename,
        receipt_filename,
        summary_filename,
        *SCORE_DIRS,
    }
    root_entries = list(score_output_dir.iterdir())
    observed_root_entries = {item.name for item in root_entries}
    if observed_root_entries != expected_root_entries:
        raise ComplexEditContractError(
            "Complex-Edit score attempt is incomplete or contaminated before promotion: "
            f"missing={sorted(expected_root_entries - observed_root_entries)[:10]}, "
            f"unexpected={sorted(observed_root_entries - expected_root_entries)[:10]}"
        )
    for item in root_entries:
        if item.name in SCORE_DIRS:
            valid = not item.is_symlink() and item.is_dir()
        else:
            valid = not item.is_symlink() and item.is_file()
        if not valid:
            raise ComplexEditContractError(
                f"Complex-Edit score attempt contains an unsafe entry: {item}"
            )

    attempt_manifest = load_json_strict(
        score_output_dir / SCORE_ATTEMPT_MANIFEST_FILENAME
    )
    expected_attempt_manifest = {
        "schema": SCORE_ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "benchmark": "complex_edit",
        "contract_id": contract_id,
    }
    if attempt_manifest != expected_attempt_manifest:
        raise ComplexEditContractError("Score attempt manifest differs before promotion")

    metrics = validate_score_outputs(score_output_dir, rows)
    receipt_path = score_output_dir / receipt_filename
    receipt = load_json_strict(receipt_path)
    expected_receipt = build_score_receipt(
        score_output_dir,
        contract_id=contract_id,
        export_manifest_sha256=export_manifest_sha256,
        metrics=metrics,
    )
    if receipt != expected_receipt:
        raise ComplexEditContractError(
            "Complex-Edit score receipt is not an exact derivation of the active attempt"
        )

    summary_path = score_output_dir / summary_filename
    summary = load_json_strict(summary_path)
    if summary != dict(expected_summary):
        raise ComplexEditContractError("Complex-Edit score summary bytes differ before promotion")
    expected_attempt_record = {
        **expected_attempt_manifest,
        "namespace": f"complete/{attempt_id}",
    }
    expected_path_binding = {
        "export_images": str(export_output_dir),
        "score_attempt": str(completed_dir),
        "scorer_repo": str(scorer_repo),
        "score_worker_module": HARDENED_SCORE_WORKER_MODULE,
    }
    if summary.get("evaluation_contract") != contract:
        raise ComplexEditContractError("Score summary evaluation contract differs")
    if summary.get("score_attempt") != expected_attempt_record:
        raise ComplexEditContractError("Score summary attempt binding differs")
    if summary.get("path_binding") != expected_path_binding:
        raise ComplexEditContractError("Score summary scorer/output path binding differs")
    if summary.get("run_model_name") != run_model_name or summary.get("metrics") != metrics:
        raise ComplexEditContractError("Score summary identity or metrics differ")
    if summary.get("logs") != [str(completed_dir / log_filename)]:
        raise ComplexEditContractError("Score summary log path differs")
    score_receipt = summary.get("score_receipt")
    expected_receipt_path = completed_dir / receipt_filename
    if not isinstance(score_receipt, Mapping) or (
        score_receipt.get("path") != str(expected_receipt_path)
        or score_receipt.get("sha256") != sha256_file(receipt_path)
        or score_receipt.get("contract_id") != contract_id
        or score_receipt.get("export_output_manifest_sha256")
        != export_manifest_sha256
        or score_receipt.get("file_count") != 3 * (len(rows) + 1)
        or score_receipt.get("files_sha256") != receipt["files_sha256"]
        or score_receipt.get("judge_provenance") != metrics["judge_provenance"]
    ):
        raise ComplexEditContractError("Score summary receipt binding differs")
    return metrics


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish canonical JSON without overwriting a prior evaluation artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ComplexEditContractError(f"Refusing to overwrite evaluation artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ComplexEditContractError(f"Refusing concurrent overwrite: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def output_manifest_path(output_root: Path) -> Path:
    return Path(output_root) / OUTPUT_MANIFEST_FILENAME
