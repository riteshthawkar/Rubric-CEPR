from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from qwen_edit_project.utils.paths import relative_to_repo, resolve_path


CONTRACT_SCHEMA = "qwen-edit-evaluation-contract/v2"
CONTRACT_FILENAME = ".evaluation_contract.json"
OUTPUT_MANIFEST_SCHEMA = "qwen-edit-evaluation-outputs/v1"
OUTPUT_MANIFEST_FILENAME = ".evaluation_outputs.json"
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"

_COMMON_RUNTIME_PACKAGES = (
    "accelerate",
    "diffusers",
    "huggingface-hub",
    "numpy",
    "Pillow",
    "safetensors",
    "torch",
    "transformers",
)
_BENCHMARK_RUNTIME_PACKAGES = {
    "gedit": ("datasets", "megfile", "openai", "requests"),
    "imgedit": ("openai",),
}


class EvaluationContractError(RuntimeError):
    """Raised when an immutable evaluation run contract cannot be honored."""


def normalize_openai_base_url(value: object) -> str:
    normalized = str(value).strip().rstrip("/")
    if not normalized:
        raise EvaluationContractError("OpenAI base URL must be non-empty")
    return normalized


def enforce_official_openai_endpoint(
    config: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    scoring = config.get("scoring", {})
    if not isinstance(scoring, Mapping):
        raise EvaluationContractError("scoring must be a mapping")
    configured = normalize_openai_base_url(scoring.get("openai_base_url", ""))
    if configured != OFFICIAL_OPENAI_BASE_URL:
        raise EvaluationContractError(
            "Hardened OpenAI judging requires the official endpoint "
            f"{OFFICIAL_OPENAI_BASE_URL!r}, found {configured!r}"
        )
    environment = os.environ if environment is None else environment
    raw_environment = environment.get("OPENAI_BASE_URL")
    if raw_environment is not None and str(raw_environment).strip():
        environment_url = normalize_openai_base_url(raw_environment)
        if environment_url != configured:
            raise EvaluationContractError(
                "OPENAI_BASE_URL differs from the contracted official endpoint: "
                f"{environment_url!r} != {configured!r}"
            )
    return configured


def build_judge_protocol(
    config: Mapping[str, Any],
    *,
    benchmark: str,
) -> dict[str, Any] | None:
    scoring = config.get("scoring", {})
    if not isinstance(scoring, Mapping):
        raise EvaluationContractError("scoring must be a mapping")
    provider = scoring.get("provider")
    if provider is None:
        return None
    if str(provider) != "openai":
        raise EvaluationContractError(f"Unsupported hardened judge provider: {provider!r}")
    base_url = normalize_openai_base_url(scoring.get("openai_base_url", ""))
    if base_url != OFFICIAL_OPENAI_BASE_URL:
        raise EvaluationContractError(
            f"Hardened judge protocol must use {OFFICIAL_OPENAI_BASE_URL!r}"
        )
    score_run_id = str(scoring.get("score_run_id", "")).strip()
    if not score_run_id:
        raise EvaluationContractError("Hardened judge protocol requires scoring.score_run_id")
    if not bool(scoring.get("fresh_only", False)):
        raise EvaluationContractError("Hardened judge protocol requires scoring.fresh_only=true")

    if benchmark == "imgedit":
        if not bool(scoring.get("force_rescore", False)):
            raise EvaluationContractError(
                "Hardened fresh-only ImgEdit judging requires scoring.force_rescore=true"
            )
        openai_model = str(scoring.get("openai_model", "")).strip()
        expected_model = str(scoring.get("expected_openai_model", "")).strip()
        if not openai_model or openai_model != expected_model:
            raise EvaluationContractError(
                "Hardened ImgEdit judging requires matching non-empty openai_model and "
                "expected_openai_model"
            )
        if not bool(scoring.get("strict_response_parser", False)):
            raise EvaluationContractError(
                "Hardened ImgEdit judging requires scoring.strict_response_parser=true"
            )
        if bool(scoring.get("allow_partial", False)):
            raise EvaluationContractError(
                "Hardened ImgEdit judging requires scoring.allow_partial=false"
            )
        positive_integer_fields = ("num_processes", "retry_num_processes")
        nonnegative_integer_fields = ("max_retry_rounds", "openai_max_retries")
        integer_fields = positive_integer_fields + nonnegative_integer_fields
        raw_integers = [scoring.get(field) for field in integer_fields]
        try:
            positive_integers = [int(scoring.get(field, 0)) for field in positive_integer_fields]
            nonnegative_integers = [
                int(scoring.get(field, -1)) for field in nonnegative_integer_fields
            ]
            timeout = float(scoring.get("openai_timeout_seconds", 0))
            retry_sleep = float(scoring.get("retry_sleep_seconds", -1))
            temperature = scoring.get("temperature")
            numeric_temperature = float(temperature) if temperature is not None else None
        except (TypeError, ValueError) as exc:
            raise EvaluationContractError(
                "Hardened ImgEdit numeric judge parameters are invalid"
            ) from exc
        if any(
            isinstance(value, bool) or float(value) != int(value)
            for value in raw_integers
        ):
            raise EvaluationContractError(
                "Hardened ImgEdit worker and retry counts must be integers"
            )
        if (
            any(value <= 0 for value in positive_integers)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise EvaluationContractError(
                "Hardened ImgEdit worker counts and request timeout must be positive"
            )
        if (
            any(value < 0 for value in nonnegative_integers)
            or not math.isfinite(retry_sleep)
            or retry_sleep < 0
        ):
            raise EvaluationContractError(
                "Hardened ImgEdit retry counts and sleep must be non-negative"
            )
        if numeric_temperature is not None and not math.isfinite(numeric_temperature):
            raise EvaluationContractError("Hardened ImgEdit temperature must be finite")
        protocol = {
            "schema": "qwen-edit-imgedit-openai-judge/v1",
            "provider": "openai",
            "base_url": base_url,
            "endpoint": f"{base_url}/chat/completions",
            "score_run_id": score_run_id,
            "fresh_only": True,
            "request_parameters": {
                "model": openai_model,
                "expected_model": expected_model,
                "temperature": scoring.get("temperature"),
                "stream": False,
                "timeout_seconds": scoring.get("openai_timeout_seconds"),
                "sdk_max_retries": scoring.get("openai_max_retries"),
            },
            "execution_parameters": {
                "initial_workers": scoring.get("num_processes"),
                "retry_workers": scoring.get("retry_num_processes"),
                "max_retry_rounds": scoring.get("max_retry_rounds"),
                "retry_sleep_seconds": scoring.get("retry_sleep_seconds"),
                "strict_response_parser": bool(scoring.get("strict_response_parser", False)),
                "allow_partial": bool(scoring.get("allow_partial", False)),
                "force_all_records": True,
            },
        }
    elif benchmark == "gedit":
        raw_integer_fields = {
            "thread_workers": scoring.get("thread_workers"),
            "item_max_retries": scoring.get("item_max_retries"),
            "request_max_tokens": scoring.get("request_max_tokens"),
        }
        try:
            integer_fields = {
                key: int(value) for key, value in raw_integer_fields.items()
            }
            timeout = float(scoring.get("request_timeout_seconds"))
            initial_backoff = float(scoring.get("retry_initial_backoff_seconds"))
            max_backoff = float(scoring.get("retry_max_backoff_seconds"))
        except (TypeError, ValueError) as exc:
            raise EvaluationContractError(
                "Hardened GEdit timeout/retry parameters are invalid"
            ) from exc
        if any(
            isinstance(value, bool) or float(value) != integer_fields[key]
            for key, value in raw_integer_fields.items()
        ):
            raise EvaluationContractError(
                "Hardened GEdit worker, retry, and token counts must be integers"
            )
        if any(value <= 0 for value in integer_fields.values()):
            raise EvaluationContractError(
                "Hardened GEdit worker, retry, and token counts must be positive"
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise EvaluationContractError(
                "Hardened GEdit request timeout must be finite and positive"
            )
        if (
            not math.isfinite(initial_backoff)
            or not math.isfinite(max_backoff)
            or initial_backoff < 0
            or max_backoff < initial_backoff
        ):
            raise EvaluationContractError(
                "Hardened GEdit retry backoff must be finite, non-negative, and capped"
            )
        if bool(scoring.get("allow_partial", False)):
            raise EvaluationContractError(
                "Hardened GEdit judging requires scoring.allow_partial=false"
            )
        if not bool(scoring.get("strict_response_parser", False)):
            raise EvaluationContractError(
                "Hardened GEdit judging requires scoring.strict_response_parser=true"
            )
        expected_model = str(scoring.get("expected_openai_model", "")).strip()
        if not expected_model:
            raise EvaluationContractError(
                "Hardened GEdit judging requires expected_openai_model"
            )
        protocol = {
            "schema": "qwen-edit-gedit-openai-judge/v1",
            "provider": "openai",
            "base_url": base_url,
            "endpoint": f"{base_url}/chat/completions",
            "score_run_id": score_run_id,
            "fresh_only": True,
            "request_parameters": {
                "model": expected_model,
                "temperature": None,
                "stream": False,
                "max_tokens": scoring.get("request_max_tokens"),
                "timeout_seconds": scoring.get("request_timeout_seconds"),
            },
            "execution_parameters": {
                "thread_workers": scoring.get("thread_workers"),
                "item_max_retries": scoring.get("item_max_retries"),
                "retry_initial_backoff_seconds": scoring.get(
                    "retry_initial_backoff_seconds"
                ),
                "retry_max_backoff_seconds": scoring.get(
                    "retry_max_backoff_seconds"
                ),
                "strict_response_parser": True,
                "allow_partial": bool(scoring.get("allow_partial", False)),
                "instruction_language": config.get("dataset", {}).get(
                    "instruction_language"
                ),
            },
        }
    else:
        raise EvaluationContractError(
            f"No canonical OpenAI judge protocol is defined for benchmark {benchmark!r}"
        )
    return protocol


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing and byte-for-byte comparison."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return filesystem identity fields that must stay stable across a read."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_package_versions(package_names: Iterable[str]) -> dict[str, dict[str, Any]]:
    versions: dict[str, dict[str, Any]] = {}
    normalized_names = sorted({str(name).strip() for name in package_names})
    if not normalized_names or any(not name for name in normalized_names):
        raise EvaluationContractError("runtime package names must be non-empty")
    for name in normalized_names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = {"installed": False, "version": None}
        else:
            versions[name] = {"installed": True, "version": version}
    return versions


def _torch_runtime_settings(
    *,
    requested_device: str | None,
    require_accelerator: bool,
) -> dict[str, Any]:
    """Capture stable PyTorch numerical settings without volatile job/device IDs."""

    try:
        import torch
    except ImportError as exc:
        if require_accelerator:
            raise EvaluationContractError(
                "PyTorch is required to capture hardened evaluation runtime provenance"
            ) from exc
        return {"available": False}

    def optional_attr(owner: Any, name: str) -> Any:
        return getattr(owner, name, None)

    def optional_call(owner: Any, name: str) -> Any:
        value = getattr(owner, name, None)
        return value() if callable(value) else None

    def nvidia_driver_versions() -> list[str] | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
        return versions or None

    settings: dict[str, Any] = {
        "available": True,
        "torch_version": str(torch.__version__),
        "compiled_cuda_version": optional_attr(torch.version, "cuda"),
        "compiled_hip_version": optional_attr(torch.version, "hip"),
        "default_dtype": str(torch.get_default_dtype()),
        "num_threads": int(torch.get_num_threads()),
        "num_interop_threads": int(torch.get_num_interop_threads()),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
            if hasattr(torch, "get_float32_matmul_precision")
            else None
        ),
        "deterministic_algorithms_enabled": (
            bool(torch.are_deterministic_algorithms_enabled())
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else None
        ),
        "deterministic_algorithms_warn_only": (
            bool(torch.is_deterministic_algorithms_warn_only_enabled())
            if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
            else None
        ),
        "cudnn": {
            "version": torch.backends.cudnn.version(),
            "benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "cuda_matmul": {
            "allow_tf32": optional_attr(torch.backends.cuda.matmul, "allow_tf32"),
            "allow_fp16_reduced_precision_reduction": optional_attr(
                torch.backends.cuda.matmul,
                "allow_fp16_reduced_precision_reduction",
            ),
            "allow_bf16_reduced_precision_reduction": optional_attr(
                torch.backends.cuda.matmul,
                "allow_bf16_reduced_precision_reduction",
            ),
        },
        "scaled_dot_product_attention": {
            "flash_enabled": optional_call(torch.backends.cuda, "flash_sdp_enabled"),
            "memory_efficient_enabled": optional_call(
                torch.backends.cuda, "mem_efficient_sdp_enabled"
            ),
            "math_enabled": optional_call(torch.backends.cuda, "math_sdp_enabled"),
            "cudnn_enabled": optional_call(torch.backends.cuda, "cudnn_sdp_enabled"),
        },
        "numerical_environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            )
        },
    }

    requested = str(requested_device).strip().lower() if requested_device is not None else None
    expects_cuda = bool(requested and (requested == "cuda" or requested.startswith("cuda:")))
    if not expects_cuda:
        settings["accelerator"] = {
            "requested_device": requested_device,
            "kind": "cpu_or_unspecified",
        }
        if require_accelerator:
            raise EvaluationContractError(
                "runtime_provenance.require_accelerator=true requires runtime.generation_device=cuda"
            )
        return settings

    if not torch.cuda.is_available():
        raise EvaluationContractError(
            f"Hardened evaluation requested {requested_device!r}, but CUDA is unavailable"
        )
    try:
        device_index = int(requested.split(":", 1)[1]) if ":" in requested else int(
            torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(device_index)
    except (ValueError, RuntimeError, AssertionError) as exc:
        raise EvaluationContractError(
            f"Cannot inspect requested CUDA device {requested_device!r}"
        ) from exc
    settings["accelerator"] = {
        "requested_device": requested_device,
        "kind": "cuda",
        "name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "nvidia_driver_versions": nvidia_driver_versions(),
    }
    return settings


def collect_runtime_provenance(
    config: Mapping[str, Any],
    *,
    benchmark: str,
) -> dict[str, Any]:
    """Build the deterministic runtime portion of a hardened evaluation contract.

    Volatile allocation details (hostname, Slurm job ID, GPU UUID, visible ordinal) are
    intentionally excluded: they do not define numerical behavior and would prevent a
    separately launched scoring stage from validating an otherwise identical export.
    """

    contract_cfg = config.get("evaluation_contract", {})
    provenance_cfg = contract_cfg.get("runtime_provenance", {})
    if provenance_cfg is None:
        provenance_cfg = {}
    if not isinstance(provenance_cfg, Mapping):
        raise EvaluationContractError("evaluation_contract.runtime_provenance must be a mapping")
    configured_names = provenance_cfg.get("package_names")
    if configured_names is None:
        package_names = (*_COMMON_RUNTIME_PACKAGES, *_BENCHMARK_RUNTIME_PACKAGES.get(benchmark, ()))
    elif isinstance(configured_names, (str, Path)):
        raise EvaluationContractError("runtime_provenance.package_names must be a list")
    else:
        package_names = tuple(str(name) for name in configured_names)
    packages = _runtime_package_versions(package_names)

    required_names = provenance_cfg.get("required_packages", [])
    if isinstance(required_names, (str, Path)):
        raise EvaluationContractError("runtime_provenance.required_packages must be a list")
    required = sorted({str(name) for name in required_names})
    missing_required = [
        name for name in required if name not in packages or not packages[name]["installed"]
    ]
    if missing_required:
        raise EvaluationContractError(
            "Required hardened-evaluation runtime packages are missing: "
            + ", ".join(missing_required)
        )

    runtime_cfg = config.get("runtime", {})
    if not isinstance(runtime_cfg, Mapping):
        raise EvaluationContractError("runtime must be a mapping")
    requested_device = runtime_cfg.get("generation_device")
    return {
        "schema": "qwen-edit-evaluation-runtime/v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "build": list(platform.python_build()),
            "compiler": platform.python_compiler(),
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "packages": packages,
        "numerical_runtime": _torch_runtime_settings(
            requested_device=str(requested_device) if requested_device is not None else None,
            require_accelerator=bool(provenance_cfg.get("require_accelerator", False)),
        ),
    }


def _reject_symlink_path_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise EvaluationContractError(
                f"Evaluation artifact path traverses a symlink: {current}"
            )


def _stable_artifact_file_record(
    path: Path,
    *,
    aggregate_digest: Any | None = None,
) -> tuple[int, str]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise EvaluationContractError(
            f"Evaluation artifact must be a regular non-symlink file: {path}"
        )
    if before.st_nlink != 1:
        raise EvaluationContractError(
            f"Evaluation artifact must have exactly one hard link: {path}"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if aggregate_digest is not None:
                aggregate_digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    if _stat_identity(before) != _stat_identity(opened) or _stat_identity(opened) != _stat_identity(
        after
    ):
        raise EvaluationContractError(f"Evaluation artifact changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def _directory_fingerprint(path: Path) -> dict[str, Any]:
    files: list[Path] = []
    directory_identities: dict[Path, tuple[int, int, int, int, int, int, int]] = {}
    entries: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        before = directory.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise EvaluationContractError(
                f"Evaluation artifact tree contains an unsafe directory: {directory}"
            )
        directory_identities[directory] = _stat_identity(before)
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
        for child_entry in children:
            child = directory / child_entry.name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise EvaluationContractError(
                    f"Evaluation artifact tree contains a symlink: {child}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                visit(child)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(child)
            else:
                raise EvaluationContractError(
                    f"Evaluation artifact tree contains a special file: {child}"
                )

    visit(path)
    files.sort(key=lambda item: item.relative_to(path).as_posix())
    payload_digest = hashlib.sha256()
    for child in files:
        _reject_symlink_path_components(child)
        size, digest = _stable_artifact_file_record(
            child,
            aggregate_digest=payload_digest,
        )
        entries.append(
            {
                "path": child.relative_to(path).as_posix(),
                "bytes": size,
                "sha256": digest,
            }
        )
    for directory, before_identity in directory_identities.items():
        after = directory.lstat()
        if before_identity != _stat_identity(after):
            raise EvaluationContractError(
                f"Evaluation artifact directory changed while hashing: {directory}"
            )

    return {
        "kind": "directory",
        "file_count": len(entries),
        "bytes": sum(int(item["bytes"]) for item in entries),
        "tree_sha256": sha256_bytes(canonical_json_bytes(entries)),
        "payload_sha256": payload_digest.hexdigest(),
    }


def fingerprint_artifact(
    configured_path: str | Path | None,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    """Return a portable content fingerprint for a file or directory.

    The absolute machine path is deliberately excluded from the contract identity. A
    repository-relative path is recorded when possible, while content hashes provide
    the immutable identity.
    """

    if configured_path is None:
        return None
    resolved = resolve_path(str(configured_path))
    if resolved is None:
        if required:
            raise FileNotFoundError(f"Required evaluation artifact is missing: {configured_path}")
        return {"path": str(configured_path), "exists": False}
    _reject_symlink_path_components(resolved)
    if resolved.is_symlink():
        raise EvaluationContractError(f"Evaluation artifact must not be a symlink: {resolved}")
    if not resolved.exists():
        if required:
            raise FileNotFoundError(f"Required evaluation artifact is missing: {configured_path}")
        return {"path": str(configured_path), "exists": False}
    common: dict[str, Any] = {"path": relative_to_repo(resolved), "exists": True}
    if resolved.is_file():
        size, digest = _stable_artifact_file_record(resolved)
        common.update(
            {
                "kind": "file",
                "bytes": size,
                "sha256": digest,
            }
        )
        return common
    if resolved.is_dir():
        common.update(_directory_fingerprint(resolved))
        return common
    raise EvaluationContractError(f"Unsupported artifact type: {resolved}")


def evaluation_contract_enabled(config: Mapping[str, Any]) -> bool:
    section = config.get("evaluation_contract", {})
    return isinstance(section, Mapping) and bool(section.get("enabled", False))


def _record_manifest(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    identities: set[str] = set()
    for record in records:
        item = {str(key): value for key, value in record.items()}
        identity = str(item.get("identity", item.get("key", ""))).strip()
        if not identity:
            raise EvaluationContractError("Every evaluation record must have a non-empty key or identity")
        if identity in identities:
            raise EvaluationContractError(f"Duplicate evaluation record identity: {identity}")
        identities.add(identity)
        normalized.append(item)
    normalized.sort(key=lambda item: str(item.get("identity", item.get("key"))))
    return {
        "count": len(normalized),
        "records": normalized,
        "records_sha256": sha256_bytes(canonical_json_bytes(normalized)),
    }


def _dataset_artifacts(config: Mapping[str, Any], *, required: bool) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    artifacts: dict[str, Any] = {}
    for key in (
        "edit_json",
        "prompts_json",
        "origin_img_root",
        "local_path",
        "manifest",
        "images_root",
        "source_inventory",
    ):
        if dataset.get(key) is not None:
            artifacts[key] = fingerprint_artifact(dataset[key], required=required)
    return artifacts


def _imgedit_source_inventory_binding(
    dataset: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any] | None:
    configured = dataset.get("source_inventory")
    if configured is None:
        return None
    inventory_path = resolve_path(str(configured))
    if inventory_path is None:
        raise EvaluationContractError("ImgEdit source inventory path does not resolve")
    before = inventory_path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise EvaluationContractError("ImgEdit source inventory must be a one-link regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(inventory_path, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > 10 * 1024 * 1024:
                raise EvaluationContractError("ImgEdit source inventory exceeds 10 MiB")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(opened) or _stat_identity(opened) != _stat_identity(
        after
    ):
        raise EvaluationContractError("ImgEdit source inventory changed while it was read")
    raw = b"".join(chunks)

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationContractError(
                    f"Duplicate key {key!r} in ImgEdit source inventory"
                )
            result[key] = value
        return result

    try:
        inventory = json.loads(raw, object_pairs_hook=reject_duplicate_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvaluationContractError("ImgEdit source inventory is invalid JSON") from exc
    if not isinstance(inventory, dict):
        raise EvaluationContractError("ImgEdit source inventory must contain an object")
    payload_digest = str(inventory.get("inventory_payload_sha256", ""))
    unsigned = dict(inventory)
    unsigned.pop("inventory_payload_sha256", None)
    if payload_digest != sha256_bytes(canonical_json_bytes(unsigned)):
        raise EvaluationContractError("ImgEdit source inventory payload hash is invalid")
    if inventory.get("schema") != "qwen-edit-imgedit-source-snapshot/v1":
        raise EvaluationContractError("Unsupported ImgEdit source inventory schema")
    if inventory.get("complete") is not True:
        raise EvaluationContractError("ImgEdit source inventory is not complete")
    origin = artifacts.get("origin_img_root")
    edit_json = artifacts.get("edit_json")
    if not isinstance(origin, Mapping) or origin.get("kind") != "directory":
        raise EvaluationContractError("ImgEdit source inventory has no bound source directory")
    if inventory.get("tree_sha256") != origin.get("tree_sha256"):
        raise EvaluationContractError("ImgEdit source inventory tree hash differs from source directory")
    if inventory.get("file_count") != origin.get("file_count"):
        raise EvaluationContractError("ImgEdit source inventory file count differs")
    if inventory.get("payload_bytes") != origin.get("bytes"):
        raise EvaluationContractError("ImgEdit source inventory payload byte count differs")
    if inventory.get("payload_sha256") != origin.get("payload_sha256"):
        raise EvaluationContractError("ImgEdit source inventory payload hash differs")
    inventory_edit_json = inventory.get("edit_json")
    if (
        not isinstance(edit_json, Mapping)
        or not isinstance(inventory_edit_json, Mapping)
        or inventory_edit_json.get("sha256") != edit_json.get("sha256")
    ):
        raise EvaluationContractError("ImgEdit source inventory edit JSON hash differs")
    root_path = resolve_path(str(dataset.get("origin_img_root")))
    if root_path is None or inventory.get("snapshot_root") != root_path.name:
        raise EvaluationContractError("ImgEdit source inventory snapshot root differs")
    return {
        "schema": inventory["schema"],
        "inventory_payload_sha256": payload_digest,
        "inventory_file_sha256": hashlib.sha256(raw).hexdigest(),
        "tree_sha256": inventory["tree_sha256"],
        "payload_sha256": inventory.get("payload_sha256"),
        "payload_bytes": inventory["payload_bytes"],
        "file_count": inventory["file_count"],
        "record_count": inventory.get("record_count"),
        "record_manifest_sha256": inventory.get("record_manifest_sha256"),
    }


def _judge_implementation_artifacts(
    section: Mapping[str, Any],
    *,
    required: bool,
) -> list[dict[str, Any]]:
    configured = section.get("implementation_artifacts", [])
    if configured is None:
        return []
    if isinstance(configured, (str, Path)):
        configured = [configured]
    if not isinstance(configured, (list, tuple)):
        raise EvaluationContractError("implementation_artifacts must be a list")
    artifacts: list[dict[str, Any]] = []
    for value in configured:
        fingerprint = fingerprint_artifact(value, required=required)
        if fingerprint is not None:
            artifacts.append(fingerprint)
    artifacts.sort(key=lambda item: str(item.get("path", "")))
    return artifacts


def build_evaluation_contract(
    config: Mapping[str, Any],
    *,
    benchmark: str,
    records: Iterable[Mapping[str, Any]],
    runtime_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical, self-identifying manifest for an evaluation run."""

    if not evaluation_contract_enabled(config):
        raise EvaluationContractError("evaluation_contract.enabled must be true to build a run contract")
    contract_cfg = config.get("evaluation_contract", {})
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    scoring = config.get("scoring", {})
    checkpoint_path = model.get("checkpoint_path")
    checkpoint_required = checkpoint_path is not None
    prompt_artifact = (
        fingerprint_artifact(dataset.get("prompts_json"), required=True)
        if dataset.get("prompts_json") is not None
        else None
    )
    config_path = config.get("_config_path")
    if bool(contract_cfg.get("require_config_artifact", False)) and not config_path:
        raise EvaluationContractError(
            "evaluation_contract.require_config_artifact=true requires a loaded config path"
        )
    dataset_artifacts = _dataset_artifacts(
        config,
        required=bool(contract_cfg.get("require_dataset_artifacts", True)),
    )
    source_inventory_binding = (
        _imgedit_source_inventory_binding(dataset, dataset_artifacts)
        if benchmark == "imgedit"
        else None
    )

    body: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "benchmark": benchmark,
        "model": {
            "backend": model.get("backend"),
            "base_model": model.get("base_model"),
            "base_model_revision": model.get("revision"),
            "model_type": model.get("model_type", "base"),
            "checkpoint": fingerprint_artifact(checkpoint_path, required=checkpoint_required),
            "lora_scale": model.get("lora_scale"),
            "torch_dtype": model.get("torch_dtype"),
            "processor_model_id": model.get("processor_model_id"),
            "model_id_with_origin_paths": model.get("model_id_with_origin_paths"),
            "local_files_only": bool(model.get("local_files_only", False)),
        },
        "dataset": {
            "source": dataset.get("source", "local"),
            "dataset_name": dataset.get("dataset_name"),
            "revision": dataset.get("revision"),
            "split": dataset.get("split"),
            "instruction_language": dataset.get("instruction_language"),
            "task_type": dataset.get("task_type"),
            "artifacts": dataset_artifacts,
            "source_inventory_binding": source_inventory_binding,
            "selection": _record_manifest(records),
        },
        "generation": dict(config.get("generation", {})),
        "prompting": dict(config.get("prompting", {})),
        "configuration_artifact": fingerprint_artifact(
            config_path,
            required=bool(contract_cfg.get("require_config_artifact", False)),
        ),
        "runtime_provenance": dict(runtime_provenance)
        if runtime_provenance is not None
        else collect_runtime_provenance(config, benchmark=benchmark),
        "implementation_artifacts": _judge_implementation_artifacts(
            contract_cfg,
            required=bool(contract_cfg.get("require_implementation_artifacts", True)),
        ),
        "seeds": {
            "base_seed": config.get("generation", {}).get("seed"),
            "seed_stride": config.get("generation", {}).get("seed_stride"),
            "strategy": config.get("generation", {}).get("seed_strategy", "constant_per_record"),
        },
        "judge": {
            "provider": scoring.get("provider"),
            "base_url": scoring.get("openai_base_url"),
            "backbone": scoring.get("backbone"),
            "model": scoring.get("openai_model", scoring.get("expected_openai_model")),
            "expected_model": scoring.get("expected_openai_model"),
            "temperature": scoring.get("temperature"),
            "prompt_artifact": prompt_artifact,
            "strict_response_parser": bool(scoring.get("strict_response_parser", False)),
            "implementation_artifacts": _judge_implementation_artifacts(
                scoring,
                required=bool(contract_cfg.get("require_judge_artifacts", True)),
            ),
            "protocol": build_judge_protocol(config, benchmark=benchmark),
        },
    }
    if "canonical_protocol" in contract_cfg:
        canonical_protocol = str(contract_cfg["canonical_protocol"]).strip()
        if not canonical_protocol:
            raise EvaluationContractError(
                "evaluation_contract.canonical_protocol must be non-empty"
            )
        if body["judge"]["protocol"] is not None:
            expected_protocol = f"{benchmark}-openai-fresh-score/v1"
            if canonical_protocol != expected_protocol:
                raise EvaluationContractError(
                    "evaluation_contract.canonical_protocol differs from the hardened judge "
                    f"protocol: expected {expected_protocol!r}"
                )
        body["canonical_protocol"] = canonical_protocol
    if "benchmark_protocol" in contract_cfg:
        benchmark_protocol = contract_cfg["benchmark_protocol"]
        if not isinstance(benchmark_protocol, Mapping):
            raise EvaluationContractError(
                "evaluation_contract.benchmark_protocol must be a mapping"
            )
        body["benchmark_protocol"] = dict(benchmark_protocol)
    contract_id = sha256_bytes(canonical_json_bytes(body))
    return {"contract_id": contract_id, "contract": body}


def contracted_model_name(model_name: str, contract: Mapping[str, Any], *, length: int = 16) -> str:
    contract_id = str(contract["contract_id"])
    if length < 12 or length > len(contract_id):
        raise ValueError("evaluation_contract.output_id_length must be between 12 and 64")
    return f"{model_name}__eval_{contract_id[:length]}"


def contracted_scores_dir(base_dir: Path, contract: Mapping[str, Any]) -> Path:
    return base_dir / "runs" / str(contract["contract_id"])


def contract_path(output_dir: Path) -> Path:
    return output_dir / CONTRACT_FILENAME


def write_or_validate_contract(output_dir: Path, contract: Mapping[str, Any]) -> Path:
    """Atomically establish a contract, or reject any mismatch/legacy contents."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = contract_path(output_dir)
    expected_bytes = canonical_json_bytes(contract) + b"\n"
    if path.exists():
        if path.read_bytes() != expected_bytes:
            raise EvaluationContractError(
                f"Evaluation contract mismatch at {path}; use the contract-keyed output directory "
                "for the current settings instead of mixing runs."
            )
        return path

    existing = [item.name for item in output_dir.iterdir() if item.name != CONTRACT_FILENAME]
    if existing:
        raise EvaluationContractError(
            f"Refusing to attach a new contract to non-empty output directory {output_dir}; "
            f"first entries: {sorted(existing)[:10]}"
        )
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(expected_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _safe_output_path(output_dir: Path, relative_value: str | Path) -> tuple[Path, str]:
    relative = Path(relative_value)
    normalized = relative.as_posix()
    if not normalized or relative.is_absolute() or ".." in relative.parts:
        raise EvaluationContractError(f"Unsafe evaluation output path: {relative_value!r}")
    path = output_dir / relative
    if path.is_symlink():
        raise EvaluationContractError(f"Evaluation output must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation output is missing: {path}")
    return path, normalized


def build_output_manifest(
    output_dir: Path,
    relative_paths: Iterable[str | Path],
    *,
    contract_id: str,
) -> dict[str, Any]:
    """Hash every generated image consumed by a hardened scorer.

    The evaluation contract fixes inputs and settings; this second receipt fixes the
    actual generated bytes.  Scoring validates both so a PNG cannot be replaced under
    an otherwise unchanged contract identity.
    """

    output_dir = Path(output_dir)
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_relative in sorted(relative_paths, key=lambda value: Path(value).as_posix()):
        path, relative = _safe_output_path(output_dir, raw_relative)
        if relative in seen:
            raise EvaluationContractError(f"Duplicate evaluation output path: {relative}")
        seen.add(relative)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise EvaluationContractError("A hardened evaluation output manifest cannot be empty")
    return {
        "schema": OUTPUT_MANIFEST_SCHEMA,
        "contract_id": str(contract_id),
        "count": len(files),
        "files": files,
    }


def write_or_validate_output_manifest(
    output_dir: Path,
    relative_paths: Iterable[str | Path],
    *,
    contract_id: str,
) -> Path:
    """Write the generated-byte receipt once, or require an exact resume match."""

    output_dir = Path(output_dir)
    payload = build_output_manifest(
        output_dir,
        relative_paths,
        contract_id=contract_id,
    )
    expected_bytes = canonical_json_bytes(payload) + b"\n"
    path = output_dir / OUTPUT_MANIFEST_FILENAME
    if path.exists():
        if path.is_symlink() or path.read_bytes() != expected_bytes:
            raise EvaluationContractError(
                f"Evaluation output-byte manifest mismatch at {path}; generated images "
                "changed under an existing contract. Use a new run identity."
            )
        return path
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(expected_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def validate_output_manifest(
    output_dir: Path,
    relative_paths: Iterable[str | Path],
    *,
    contract_id: str,
) -> dict[str, Any]:
    """Verify the immutable generated-byte receipt before benchmark scoring."""

    output_dir = Path(output_dir)
    path = output_dir / OUTPUT_MANIFEST_FILENAME
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Evaluation output-byte manifest is missing: {path}")
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvaluationContractError(f"Invalid evaluation output manifest: {path}") from exc
    expected = build_output_manifest(
        output_dir,
        relative_paths,
        contract_id=contract_id,
    )
    if actual != expected:
        raise EvaluationContractError(
            f"Generated evaluation outputs no longer match their receipt: {path}"
        )
    return expected
