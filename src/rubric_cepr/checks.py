"""Filesystem-only validation. Never imports a model or initializes CUDA."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity() -> dict:
    return json.loads((ROOT / "reproducibility/artifacts.json").read_text())


def check_code() -> dict:
    cfg = identity()
    for name, expected in cfg["pinned_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Pinned v1 file differs: {name}")
    rows = json.loads((ROOT / "reproducibility/manifests/extraction_v1.json").read_text())
    if len(rows) != 64:
        raise ValueError("The historical manifest must have exactly 64 rows")
    for field in ("record_key", "image", "edit_image"):
        if len({row[field] for row in rows}) != 64:
            raise ValueError(f"Duplicate manifest {field}")
    if any(row["family"] != "extract" or row["sample_weight"] != 1 for row in rows):
        raise ValueError("V1 requires extraction-only rows with unit weights")
    release_files = 0
    if (ROOT / "SHA256SUMS").is_file():
        for line in (ROOT / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            path = safe_relative_path(ROOT, name)
            if digest(path) != expected:
                raise ValueError(f"Release file differs: {name}")
            release_files += 1
    return {"pinned_files": len(cfg["pinned_files"]), "release_files": release_files, "rows": len(rows), "code_verified": True}


def safe_relative_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe artifact path: {name}")
    candidate = root / relative
    # Refuse even an ancestor symlink; no writes through external aliases.
    if any(p.is_symlink() for p in [candidate, *candidate.parents] if p != root.parent):
        raise ValueError(f"Artifact path has a symlink: {name}")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes data root: {name}")
    return candidate


def check_data(data_root: Path) -> dict:
    check_code()
    inventory = json.loads((ROOT / "reproducibility/manifests/extraction_v1_artifacts.json").read_text())
    artifacts = inventory["artifacts"]
    rows = json.loads((ROOT / inventory["training_manifest"]).read_text())
    refs = {(str(row["record_key"]), field): row[field] for row in rows for field in ("image", "edit_image")}
    seen = set()
    paths = set()
    for item in artifacts:
        ref = (str(item["record_key"]), item["field"])
        if ref in seen or refs.get(ref) != item["path"]:
            raise ValueError(f"Unexpected or duplicate training reference: {ref}")
        seen.add(ref)
        paths.add(item["path"])
        path = safe_relative_path(data_root, item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Missing or changed training artifact: {item['path']}")
    if len(seen) != 128 or len(paths) != 128 or seen != refs.keys():
        raise ValueError("Expected all 128 unique source/target references")
    return {"artifact_references": 128, "data_verified": True}


def check_disjointness(data_root: Path, benchmark_json: Path, image_root: Path) -> dict:
    rows = json.loads((ROOT / "reproducibility/manifests/extraction_v1.json").read_text())
    def no_duplicate_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate benchmark JSON key: {key}")
            result[key] = value
        return result
    records = json.loads(benchmark_json.read_text(), object_pairs_hook=no_duplicate_keys)
    if len(records) != 737:
        raise ValueError(f"Expected complete 737-row ImgEdit benchmark, found {len(records)}")
    sources = [safe_relative_path(data_root, row["edit_image"]) for row in rows]
    images = [safe_relative_path(image_root, row["id"]) for row in records.values()]
    if not all(path.is_file() for path in images):
        raise ValueError("Missing ImgEdit source images")
    if {path.name for path in sources} & {path.name for path in images}:
        raise ValueError("Training and benchmark source basenames overlap")
    if {digest(path) for path in sources} & {digest(path) for path in images}:
        raise ValueError("Training and benchmark source content overlaps")
    return {"benchmark_rows": len(records), "basename_overlap": 0, "content_overlap": 0}


def environment_report(strict: bool = False) -> dict:
    import yaml
    contract = yaml.safe_load((ROOT / "configs/reproduction/extraction_v1.yaml").read_text())
    expected = contract["environment"]
    found = {}
    mismatches = []
    for name, target in expected["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        found[name] = actual
        if actual is None or actual.split("+")[0] != str(target):
            mismatches.append(name)
    if platform.python_version() != expected["python"]:
        mismatches.append("python")
    report = {"python": platform.python_version(), "packages": found, "historical_environment_mismatches": mismatches}
    if strict and mismatches:
        raise ValueError(f"Historical environment mismatch: {', '.join(mismatches)}")
    return report


def check_completion(output: Path) -> dict:
    """Verify completion and adapter structure without loading the base model."""
    receipt = json.loads((output / "training_completion.json").read_text())
    if receipt.get("status") != "complete" or any(
        receipt.get(field) != 400 for field in ("global_step", "max_train_steps", "requested_max_train_steps")
    ):
        raise ValueError("V1 did not complete exactly 400 optimizer steps")
    if receipt.get("world_size") != 1 or receipt.get("seed") != 123:
        raise ValueError("Unexpected v1 training world size or seed")
    if receipt.get("resume_from_checkpoint") is not None or receipt.get("lora_tensor_count") != 960:
        raise ValueError("V1 must be a fresh complete adapter with 960 tensors")
    weights = output / "pytorch_lora_weights.safetensors"
    # Read the safetensors header, not tensor data. Validate offsets and finite file bounds.
    with weights.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        if length <= 0 or length > 16 * 1024 * 1024:
            raise ValueError("Invalid adapter header length")
        header = json.loads(handle.read(length))
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if len(tensors) != 960:
        raise ValueError(f"Expected 960 adapter tensors, found {len(tensors)}")
    size = weights.stat().st_size - 8 - length
    for name, tensor in tensors.items():
        start, end = tensor["data_offsets"]
        if not 0 <= start < end <= size or not any(marker in name for marker in ("lora_A", "lora_B")):
            raise ValueError(f"Invalid adapter tensor: {name}")
        if tensor["dtype"] not in {"BF16", "F32"} or 16 not in tensor["shape"]:
            raise ValueError(f"Unexpected v1 rank or dtype: {name}")
    return {"complete": True, "optimizer_steps": 400, "adapter_tensors": len(tensors), "sha256": digest(weights)}
