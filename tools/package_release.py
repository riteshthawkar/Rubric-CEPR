#!/usr/bin/env python3
"""Build a deterministic source-only release archive from the verified file list."""

import argparse
import gzip
import hashlib
import io
import tarfile
from pathlib import Path

from verify_release import ROOT, verify


def package(output: Path) -> str:
    verify()
    if output.exists() or Path(str(output) + ".sha256").exists():
        raise FileExistsError("Refusing to replace an existing release archive or sidecar")
    names = [line.split("  ", 1)[1] for line in (ROOT / "SHA256SUMS").read_text().splitlines()] + ["SHA256SUMS"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination, gzip.GzipFile(fileobj=destination, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in sorted(names):
                path = ROOT / name
                raw = path.read_bytes()
                member = tarfile.TarInfo("accv-v1/" + name)
                member.size = len(raw)
                member.mtime = 0
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                member.mode = 0o755 if name.endswith(".sh") else 0o644
                archive.addfile(member, io.BytesIO(raw))
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    Path(str(output) + ".sha256").write_text(f"{checksum}  {output.name}\n")
    return checksum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(package(args.output.resolve()))
