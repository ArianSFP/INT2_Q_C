#!/usr/bin/env python3
"""Verify the frozen exact regular-file closure of this review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    original = args.package
    before = original.lstat()
    if not stat.S_ISDIR(before.st_mode) or original.is_symlink():
        raise ValueError("review root real directory")
    root = original.resolve(strict=True)
    manifest_path = root / "source_manifest.json"
    manifest_raw = manifest_path.read_bytes()
    if manifest_path.is_symlink() or \
            sha256(manifest_raw) != args.expected_manifest_sha256:
        raise ValueError("review manifest external pin")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    if manifest.get("schema") != \
            "strata-rm-global-swap-v1-hardened-reproducibility-review-manifest":
        raise ValueError("review manifest schema")
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("review manifest members")
    names = []
    observed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "bytes", "sha256"}:
            raise ValueError("review member schema")
        name = row["name"]
        if not isinstance(name, str) or not name or Path(name).name != name or \
                name in names or name == "source_manifest.json":
            raise ValueError("review flat unique member")
        path = root / name
        member_before = path.lstat()
        if not stat.S_ISREG(member_before.st_mode) or path.is_symlink():
            raise ValueError(f"review regular member {name}")
        payload = path.read_bytes()
        member_after = path.lstat()
        if (member_before.st_dev, member_before.st_ino, member_before.st_mode,
                member_before.st_size, member_before.st_mtime_ns) != \
                (member_after.st_dev, member_after.st_ino, member_after.st_mode,
                 member_after.st_size, member_after.st_mtime_ns):
            raise ValueError(f"review member changed while read {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        if item != row:
            raise ValueError(f"review member pin {name}")
        names.append(name)
        observed.append(item)
    entries = list(os.scandir(root))
    if {entry.name for entry in entries} != set(names) | {"source_manifest.json"} or \
            not all(entry.is_file(follow_symlinks=False) for entry in entries):
        raise ValueError("review exact regular closure")
    actual_root = sha256(canonical(observed))
    if actual_root != manifest.get("source_root_sha256"):
        raise ValueError("review source root")
    print(json.dumps({
        "passed": True, "members": len(observed),
        "source_root_sha256": actual_root,
        "producer_source_root_sha256": manifest["producer_source_root_sha256"],
        "status": manifest["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
