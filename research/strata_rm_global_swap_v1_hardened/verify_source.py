#!/usr/bin/env python3
"""Verify the exact regular-file closure of this source package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_hash(rows) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    if not package.is_dir() or package.is_symlink():
        raise ValueError("package real directory")
    manifest_path = package / "source_manifest.json"
    manifest_stat = manifest_path.lstat()
    if not stat.S_ISREG(manifest_stat.st_mode) or manifest_path.is_symlink() or \
            sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise ValueError("external manifest pin")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "strata-rm-global-swap-v1-hardened-source-manifest":
        raise ValueError("manifest schema")
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest members")
    names = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "bytes", "sha256"}:
            raise ValueError("member schema")
        name = row["name"]
        if not isinstance(name, str) or not name or Path(name).name != name or \
                name in names or name == "source_manifest.json":
            raise ValueError("flat unique member")
        names.append(name)
    entries = list(os.scandir(package))
    if {entry.name for entry in entries} != set(names) | {"source_manifest.json"} or \
            not all(entry.is_file(follow_symlinks=False) and
                    not entry.is_dir(follow_symlinks=False) for entry in entries):
        raise ValueError("unexpected file, directory, or symlink")
    observed = []
    for row in rows:
        path = package / row["name"]
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.is_symlink():
            raise ValueError(f"regular member: {row['name']}")
        item = {"name": row["name"], "bytes": path.stat().st_size,
                "sha256": sha256_file(path)}
        if item != row:
            raise ValueError(f"member pin: {row['name']}")
        observed.append(item)
    if root_hash(observed) != manifest.get("source_root_sha256"):
        raise ValueError("source root")
    print(json.dumps({"passed": True, "members": len(rows),
                      "source_root_sha256": manifest["source_root_sha256"],
                      "status": manifest["status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

