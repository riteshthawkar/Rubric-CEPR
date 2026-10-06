#!/usr/bin/env python3
"""Materialize the exact ImgEdit source files as an immutable byte snapshot.

This utility is deliberately model-free.  It reads the ImgEdit edit JSON, copies
each uniquely referenced source image into a new tree, and emits a canonical JSON
inventory beside that tree.  Referenced symlinks, hardlinks, and special files are
rejected.  Published files are independent regular files with one link and mode
``0444``; published directories use mode ``0555``.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_SCHEMA = "qwen-edit-imgedit-source-snapshot/v1"
RENAME_NOREPLACE = 1
AT_FDCWD = -100
CHUNK_SIZE = 1024 * 1024


class ImgEditSnapshotError(RuntimeError):
    """Raised when an exact, independent ImgEdit snapshot cannot be built."""


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    path: Path
    identity: tuple[int, int, int, int, int, int]
    references: tuple[tuple[str, str], ...]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON encoding used by every inventory hash."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    absolute = path.expanduser().absolute()
    try:
        return absolute.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(absolute)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_single_link_file(path: Path, *, label: str) -> tuple[bytes, tuple[int, ...]]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ImgEditSnapshotError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        kind = "symlink" if stat.S_ISLNK(before.st_mode) else "special/non-regular file"
        raise ImgEditSnapshotError(f"{label} must be a regular file, not a {kind}: {path}")
    if before.st_nlink != 1:
        raise ImgEditSnapshotError(f"{label} must have exactly one hard link: {path}")

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened_before = os.fstat(descriptor)
        if _stat_identity(opened_before) != _stat_identity(before):
            raise ImgEditSnapshotError(f"{label} changed before it could be read: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, CHUNK_SIZE):
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
        if _stat_identity(opened_after) != _stat_identity(before):
            raise ImgEditSnapshotError(f"{label} changed while it was read: {path}")
    finally:
        os.close(descriptor)
    return b"".join(chunks), _stat_identity(before)


def _load_edit_json(edit_json: Path) -> tuple[bytes, tuple[int, ...], dict[str, dict[str, Any]]]:
    raw, identity = _read_regular_single_link_file(edit_json, label="ImgEdit edit JSON")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ImgEditSnapshotError(
                    f"duplicate JSON key {key!r} in ImgEdit edit JSON: {edit_json}"
                )
            value[key] = item
        return value

    try:
        loaded = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImgEditSnapshotError(f"invalid ImgEdit edit JSON: {edit_json}") from exc
    if not isinstance(loaded, dict):
        raise ImgEditSnapshotError("ImgEdit edit JSON must be a top-level object")

    records: dict[str, dict[str, Any]] = {}
    for raw_record_id, raw_record in loaded.items():
        record_id = str(raw_record_id)
        if not record_id:
            raise ImgEditSnapshotError("ImgEdit record IDs must be non-empty")
        if not isinstance(raw_record, dict):
            raise ImgEditSnapshotError(f"ImgEdit record {record_id!r} must be an object")
        source_id = raw_record.get("id")
        edit_type = raw_record.get("edit_type")
        if not isinstance(source_id, str) or not source_id:
            raise ImgEditSnapshotError(
                f"ImgEdit record {record_id!r} has no non-empty string id"
            )
        if not isinstance(edit_type, str) or not edit_type:
            raise ImgEditSnapshotError(
                f"ImgEdit record {record_id!r} has no non-empty string edit_type"
            )
        records[record_id] = raw_record
    return raw, identity, records


def _canonical_relative_path(raw: str, *, record_id: str) -> str:
    if "\\" in raw:
        raise ImgEditSnapshotError(
            f"ImgEdit record {record_id!r} uses a non-POSIX source path: {raw!r}"
        )
    pure = PurePosixPath(raw)
    if pure.is_absolute() or raw != pure.as_posix() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ImgEditSnapshotError(
            f"ImgEdit record {record_id!r} uses an unsafe source path: {raw!r}"
        )
    return pure.as_posix()


def _scan_source_files(
    source_root: Path,
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, tuple[SourceFile, ...]]:
    try:
        resolved_root = source_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ImgEditSnapshotError(f"ImgEdit source root is missing: {source_root}") from exc
    if not resolved_root.is_dir():
        raise ImgEditSnapshotError(f"ImgEdit source root is not a directory: {source_root}")

    references: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record_id, item in records.items():
        relative = _canonical_relative_path(str(item["id"]), record_id=record_id)
        references[relative].append((record_id, str(item["edit_type"])))

    files: list[SourceFile] = []
    for relative in sorted(references):
        current = resolved_root
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            current /= part
            try:
                value = current.lstat()
            except OSError as exc:
                raise ImgEditSnapshotError(
                    f"referenced ImgEdit source parent is missing: {current}"
                ) from exc
            if stat.S_ISLNK(value.st_mode):
                raise ImgEditSnapshotError(
                    f"referenced ImgEdit source path contains a directory symlink: {current}"
                )
            if not stat.S_ISDIR(value.st_mode):
                raise ImgEditSnapshotError(
                    f"referenced ImgEdit source parent is not a directory: {current}"
                )

        source = resolved_root.joinpath(*parts)
        try:
            value = source.lstat()
        except OSError as exc:
            raise ImgEditSnapshotError(f"referenced ImgEdit source is missing: {relative}") from exc
        if not stat.S_ISREG(value.st_mode):
            kind = "symlink" if stat.S_ISLNK(value.st_mode) else "special/non-regular file"
            raise ImgEditSnapshotError(
                f"referenced ImgEdit source must be a regular file, not a {kind}: {relative}"
            )
        if value.st_nlink != 1:
            raise ImgEditSnapshotError(
                f"referenced ImgEdit source has a forbidden hard-link alias: {relative}"
            )
        files.append(
            SourceFile(
                relative_path=relative,
                path=source,
                identity=_stat_identity(value),
                references=tuple(sorted(references[relative])),
            )
        )
    return resolved_root, tuple(files)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:  # pragma: no cover - defensive kernel failure path
            raise OSError("short write while materializing ImgEdit source snapshot")
        offset += written


def _copy_source_file(source: SourceFile, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    source_descriptor = os.open(
        source.path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    destination_descriptor: int | None = None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        if _stat_identity(os.fstat(source_descriptor)) != source.identity:
            raise ImgEditSnapshotError(f"source changed before copying: {source.relative_path}")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        while chunk := os.read(source_descriptor, CHUNK_SIZE):
            digest.update(chunk)
            byte_count += len(chunk)
            _write_all(destination_descriptor, chunk)
        os.fsync(destination_descriptor)
        if _stat_identity(os.fstat(source_descriptor)) != source.identity:
            raise ImgEditSnapshotError(f"source changed while copying: {source.relative_path}")
    finally:
        os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)

    copied = destination.lstat()
    if not stat.S_ISREG(copied.st_mode) or copied.st_nlink != 1:
        raise ImgEditSnapshotError(
            f"copied ImgEdit source is not an independent one-link regular file: {destination}"
        )
    if (copied.st_dev, copied.st_ino) == (source.identity[0], source.identity[1]):
        raise ImgEditSnapshotError(f"copied ImgEdit source aliases its source inode: {destination}")
    if byte_count != source.identity[3]:
        raise ImgEditSnapshotError(f"source byte count changed while copying: {source.relative_path}")
    os.chmod(destination, 0o444, follow_symlinks=False)
    return {
        "path": source.relative_path,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "reference_count": len(source.references),
        "references": [
            {"record_id": record_id, "edit_type": edit_type}
            for record_id, edit_type in source.references
        ],
    }


def _payload_sha256(root: Path, entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        with (root / str(entry["path"])).open("rb") as handle:
            while chunk := handle.read(CHUNK_SIZE):
                digest.update(chunk)
    return digest.hexdigest()


def _build_inventory(
    *,
    edit_json: Path,
    edit_json_raw: bytes,
    source_root: Path,
    destination: Path,
    records: Mapping[str, Mapping[str, Any]],
    copied_files: list[dict[str, Any]],
) -> dict[str, Any]:
    tree_entries = [
        {"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]}
        for item in copied_files
    ]
    record_manifest = sorted(
        (
            {
                "record_id": str(record_id),
                "source_path": str(item["id"]),
                "edit_type": str(item["edit_type"]),
            }
            for record_id, item in records.items()
        ),
        key=lambda item: item["record_id"],
    )
    source_counts = Counter(item["source_path"] for item in record_manifest)
    content_groups: dict[str, list[str]] = defaultdict(list)
    for item in copied_files:
        content_groups[str(item["sha256"])].append(str(item["path"]))
    duplicate_content_groups = [
        {"sha256": digest, "paths": sorted(paths)}
        for digest, paths in sorted(content_groups.items())
        if len(paths) > 1
    ]
    inventory: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "copy_policy": "independent_regular_files_no_symlinks_no_hardlinks_read_only",
        "hash_definitions": {
            "canonical_json": (
                "UTF-8 JSON; keys sorted; no insignificant whitespace; ensure_ascii=false; "
                "allow_nan=false"
            ),
            "tree_sha256": "sha256(canonical_json([{path,bytes,sha256}, ... sorted by path]))",
            "payload_sha256": (
                "sha256(concatenated file bytes in files.path order; boundaries are files.bytes)"
            ),
            "inventory_payload_sha256": (
                "sha256(canonical_json(inventory with inventory_payload_sha256 omitted))"
            ),
        },
        "edit_json": {
            "path": _portable_path(edit_json),
            "bytes": len(edit_json_raw),
            "sha256": sha256_bytes(edit_json_raw),
        },
        "source": {
            "configured_root": _portable_path(source_root),
            "configured_root_was_symlink": source_root.expanduser().absolute().is_symlink(),
        },
        "snapshot_root": destination.name,
        "complete": True,
        "record_count": len(records),
        "file_count": len(copied_files),
        "payload_bytes": sum(int(item["bytes"]) for item in copied_files),
        "record_manifest_sha256": sha256_bytes(canonical_json_bytes(record_manifest)),
        "tree_sha256": sha256_bytes(canonical_json_bytes(tree_entries)),
        "payload_sha256": _payload_sha256(destination, copied_files),
        "duplicate_source_ids": {
            "group_count": sum(count > 1 for count in source_counts.values()),
            "excess_reference_count": sum(count - 1 for count in source_counts.values()),
            "maximum_reference_count": max(source_counts.values(), default=0),
        },
        "duplicate_content": {
            "unique_sha256_count": len(content_groups),
            "group_count": len(duplicate_content_groups),
            "excess_file_count": sum(
                len(item["paths"]) - 1 for item in duplicate_content_groups
            ),
            "groups": duplicate_content_groups,
        },
        "files": copied_files,
    }
    inventory["inventory_payload_sha256"] = sha256_bytes(canonical_json_bytes(inventory))
    return inventory


def _make_tree_writable(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    for current, directory_names, file_names in os.walk(path, topdown=False):
        current_path = Path(current)
        for name in file_names:
            try:
                os.chmod(current_path / name, 0o600, follow_symlinks=False)
            except OSError:
                pass
        for name in directory_names:
            try:
                os.chmod(current_path / name, 0o700, follow_symlinks=False)
            except OSError:
                pass
        try:
            os.chmod(current_path, 0o700, follow_symlinks=False)
        except OSError:
            pass


def _remove_tree(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    _make_tree_writable(path)
    shutil.rmtree(path)


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ImgEditSnapshotError(
            "renameat2(RENAME_NOREPLACE) is required for atomic snapshot publication"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"refusing to replace existing path: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_inventory_staging(inventory: Mapping[str, Any], final_path: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{final_path.name}.staging-",
        dir=final_path.parent,
    )
    staging = Path(raw_path)
    try:
        _write_all(descriptor, canonical_json_bytes(inventory))
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    except Exception:
        os.close(descriptor)
        staging.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    return staging


def _validate_staged_tree(
    staging: Path,
    copied_files: list[dict[str, Any]],
    source_files: tuple[SourceFile, ...],
) -> None:
    expected = {str(item["path"]): item for item in copied_files}
    observed: set[str] = set()
    inode_ids: set[tuple[int, int]] = set()
    source_by_path = {item.relative_path: item for item in source_files}
    for current, directory_names, file_names in os.walk(staging, followlinks=False):
        current_path = Path(current)
        directory_stat = current_path.lstat()
        if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(directory_stat.st_mode):
            raise ImgEditSnapshotError(f"snapshot contains a non-directory entry: {current_path}")
        for name in directory_names:
            path = current_path / name
            value = path.lstat()
            if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
                raise ImgEditSnapshotError(f"snapshot contains an unsafe directory: {path}")
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(staging).as_posix()
            value = path.lstat()
            if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                raise ImgEditSnapshotError(
                    f"snapshot contains a non-regular or multiply-linked file: {relative}"
                )
            if stat.S_IMODE(value.st_mode) != 0o444:
                raise ImgEditSnapshotError(f"snapshot file is not read-only: {relative}")
            inode = (value.st_dev, value.st_ino)
            if inode in inode_ids:
                raise ImgEditSnapshotError(f"snapshot files share an inode: {relative}")
            inode_ids.add(inode)
            source = source_by_path.get(relative)
            if source is None:
                raise ImgEditSnapshotError(f"snapshot contains an unexpected file: {relative}")
            if inode == (source.identity[0], source.identity[1]):
                raise ImgEditSnapshotError(f"snapshot file aliases its source inode: {relative}")
            entry = expected[relative]
            if value.st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise ImgEditSnapshotError(f"snapshot file content mismatch: {relative}")
            observed.add(relative)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        raise ImgEditSnapshotError(f"snapshot is missing expected files: {missing}")


def materialize_snapshot(
    edit_json: Path,
    source_root: Path,
    destination: Path,
    inventory_path: Path,
    *,
    expected_record_count: int | None = 737,
) -> dict[str, Any]:
    """Build and atomically publish one exact ImgEdit source snapshot."""

    destination = destination.expanduser().absolute()
    inventory_path = inventory_path.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing existing snapshot destination: {destination}")
    if inventory_path.exists() or inventory_path.is_symlink():
        raise FileExistsError(f"refusing existing inventory destination: {inventory_path}")
    if destination.parent.resolve(strict=True) != inventory_path.parent.resolve(strict=True):
        raise ImgEditSnapshotError("snapshot and inventory must use the same existing parent")
    if inventory_path.parent == destination or destination in inventory_path.parents:
        raise ImgEditSnapshotError("inventory must be outside the snapshot tree")

    edit_json_raw, edit_json_identity, records = _load_edit_json(edit_json)
    if expected_record_count is not None and len(records) != expected_record_count:
        raise ImgEditSnapshotError(
            f"expected {expected_record_count} ImgEdit records, found {len(records)}"
        )
    _, source_files = _scan_source_files(source_root, records)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    inventory_staging: Path | None = None
    snapshot_published = False
    inventory_published = False
    try:
        copied_files = [
            _copy_source_file(source, staging / source.relative_path)
            for source in source_files
        ]
        copied_files.sort(key=lambda item: str(item["path"]))

        # Fail if either the edit JSON or any referenced source changed during copying.
        if _stat_identity(edit_json.lstat()) != edit_json_identity:
            raise ImgEditSnapshotError("ImgEdit edit JSON changed during snapshot construction")
        _, source_files_after = _scan_source_files(source_root, records)
        if source_files_after != source_files:
            raise ImgEditSnapshotError("referenced ImgEdit sources changed during snapshot construction")

        inventory = _build_inventory(
            edit_json=edit_json,
            edit_json_raw=edit_json_raw,
            source_root=source_root,
            destination=staging,
            records=records,
            copied_files=copied_files,
        )
        # The staged directory name is random; inventory identity binds the final root name.
        inventory["snapshot_root"] = destination.name
        inventory.pop("inventory_payload_sha256")
        inventory["inventory_payload_sha256"] = sha256_bytes(canonical_json_bytes(inventory))

        _validate_staged_tree(staging, copied_files, source_files)
        for current, directory_names, _ in os.walk(staging, topdown=False):
            current_path = Path(current)
            for name in directory_names:
                os.chmod(current_path / name, 0o555, follow_symlinks=False)
            os.chmod(current_path, 0o555, follow_symlinks=False)
        inventory_staging = _write_inventory_staging(inventory, inventory_path)

        _atomic_rename_noreplace(staging, destination)
        snapshot_published = True
        _atomic_rename_noreplace(inventory_staging, inventory_path)
        inventory_published = True
        _fsync_directory(destination.parent)
        return inventory
    except Exception:
        if inventory_staging is not None and not inventory_published:
            inventory_staging.unlink(missing_ok=True)
        if snapshot_published and not inventory_published:
            _remove_tree(destination)
        raise
    finally:
        if not snapshot_published:
            _remove_tree(staging)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-symlink, no-hardlink immutable ImgEdit source snapshot."
    )
    parser.add_argument(
        "--edit-json",
        type=Path,
        default=Path("data/processed/benchmark/imgedit/basic_edit.json"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/processed/benchmark/imgedit/original_images"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/processed/benchmark/imgedit/original_images_snapshot_v1"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "data/processed/benchmark/imgedit/original_images_snapshot_v1.inventory.json"
        ),
    )
    parser.add_argument("--expected-record-count", type=int, default=737)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = materialize_snapshot(
        args.edit_json,
        args.source_root,
        args.destination,
        args.inventory,
        expected_record_count=args.expected_record_count,
    )
    report = {
        "snapshot_root": str(args.destination),
        "inventory_path": str(args.inventory),
        "record_count": inventory["record_count"],
        "file_count": inventory["file_count"],
        "payload_bytes": inventory["payload_bytes"],
        "tree_sha256": inventory["tree_sha256"],
        "payload_sha256": inventory["payload_sha256"],
        "inventory_payload_sha256": inventory["inventory_payload_sha256"],
        "inventory_file_sha256": sha256_file(args.inventory),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
