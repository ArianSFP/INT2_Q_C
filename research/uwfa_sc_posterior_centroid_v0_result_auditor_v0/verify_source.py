#!/usr/bin/env python3
"""Verify the frozen source manifest without importing runtime modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


SCHEMA = "uwfa-sc-posterior-centroid-v0-result-audit-source-manifest-v0"
STATUS = "SEALED_SOURCE_ONLY_AWAITING_EXTERNAL_RESULT_PINS"
SOURCE_ROOT_DOMAIN = b"UWFA-SC-POSTERIOR-CENTROID-V0-RESULT-AUDITOR-SOURCE-ROOT-V0\x00"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_regular(path: Path) -> bytes:
    metadata = os.lstat(path)
    require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"regular nofollow: {path.name}")
    with path.open("rb") as handle:
        payload = handle.read()
    require(len(payload) == metadata.st_size, f"short read: {path.name}")
    return payload


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    require(package.is_absolute(), "absolute package path")
    metadata = os.lstat(package)
    require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), "package directory")
    manifest_payload = read_regular(package / "SOURCE_MANIFEST.json")
    manifest_hash = sha256(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(manifest_hash == expected_manifest_sha256, "external manifest hash")
    manifest = json.loads(manifest_payload.decode("utf-8"))
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == STATUS, "manifest schema/status")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "manifest rows")
    expected_names = sorted(row["name"] for row in rows) + ["SOURCE_MANIFEST.json"]
    require(sorted(item.name for item in package.iterdir()) == sorted(expected_names), "exact package member set")
    observed = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "manifest row")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name != "SOURCE_MANIFEST.json", "member name")
        payload = read_regular(package / name)
        require(len(payload) == row["bytes"] and sha256(payload) == row["sha256"], f"member binding: {name}")
        observed.append({"name": name, "bytes": len(payload), "sha256": sha256(payload)})
    observed.sort(key=lambda row: row["name"].encode("ascii"))
    root = sha256(SOURCE_ROOT_DOMAIN + canonical_json(observed))
    require(root == manifest.get("source_snapshot_root_sha256"), "source snapshot root")
    return {"schema": "uwfa-sc-posterior-centroid-v0-result-audit-source-verification-v0", "status": "PASS", "manifest_sha256": manifest_hash, "source_snapshot_root_sha256": root, "members": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--expected-manifest-sha256")
    arguments = parser.parse_args()
    print(json.dumps(verify(Path(arguments.package), arguments.expected_manifest_sha256), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
