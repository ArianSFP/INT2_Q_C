#!/usr/bin/env python3
"""Verify the frozen result-auditor source closure without importing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


SCHEMA = "tactic-actual-coarse-n18-v6-result-auditor-source-manifest-v1"
STATUS = "SEALED_POST_FAILURE_SCHEMA_REPAIR_AWAITING_RERUN"
ROOT_DOMAIN = b"TACTIC-ACTUAL-COARSE-N18-V6-RESULT-AUDITOR-SOURCE-ROOT-V1\0"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in rows:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(RuntimeError(f"{label}: nonfinite {item}")),
    )
    require(isinstance(value, dict), f"{label}: object")
    return value


def reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode), f"{label}: symlink {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def read_regular(path: Path, label: str) -> bytes:
    reject_symlink_chain(path, label)
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 4 * (1 << 20), f"{label}: regular byte bound")
        chunks = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
            require(bool(chunk), f"{label}: short read")
            chunks.append(chunk)
            offset += len(chunk)
        require(os.pread(descriptor, 1, before.st_size) == b"", f"{label}: trailing")
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_size, row.st_mtime_ns, row.st_ctime_ns, row.st_nlink)
        require(identity(before) == identity(after) == identity(named), f"{label}: identity drift")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    require(package.is_absolute(), "absolute package")
    reject_symlink_chain(package, "package")
    require(stat.S_ISDIR(os.lstat(package).st_mode), "package directory")
    manifest_payload = read_regular(package / "SOURCE_MANIFEST.json", "manifest")
    manifest_sha = sha256(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(manifest_sha == expected_manifest_sha256, "external manifest pin")
    manifest = strict_json(manifest_payload, "manifest")
    require(set(manifest) == {"schema", "status", "source_snapshot_root_sha256", "members", "access_attestation", "claim_boundary"}, "manifest exact fields")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS, "manifest schema/status")
    require(manifest["access_attestation"] == {
        "completed_v6_result_accessed": True,
        "cuda_or_cupy_initialized": False,
        "network_accessed": True,
        "numerical_result_metadata_accessed": True,
        "qwen_or_model_payload_accessed": False,
        "runpod_access_scope": "post-failure diagnosis opened the failed log plus RESULT/DECODER_RECEIPT score metadata; no COARSE.bin or BF16 payload was opened while repairing and freezing v1",
        "runpod_accessed": True,
    }, "post-failure source-build access attestation")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "manifest rows")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "manifest row")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name != "SOURCE_MANIFEST.json" and name not in names, "manifest member name")
        payload = read_regular(package / name, f"member {name}")
        require(len(payload) == row["bytes"] and sha256(payload) == row["sha256"], f"member binding {name}")
        observed.append({"name": name, "bytes": len(payload), "sha256": sha256(payload)})
        names.append(name)
    require(names == sorted(names, key=lambda value: value.encode("ascii")), "canonical row order")
    root = sha256(ROOT_DOMAIN + canonical_json(observed))
    require(root == manifest["source_snapshot_root_sha256"], "source snapshot root")
    entries = list(os.scandir(package))
    require({entry.name for entry in entries} == set(names) | {"SOURCE_MANIFEST.json"}, "exact package closure")
    require(all(entry.is_file(follow_symlinks=False) for entry in entries), "regular package closure")
    return {
        "schema": "tactic-actual-coarse-n18-v6-result-auditor-source-verification-v1",
        "status": "PASS",
        "manifest_sha256": manifest_sha,
        "source_snapshot_root_sha256": root,
        "members": len(rows),
        "positive_claim_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--expected-manifest-sha256")
    arguments = parser.parse_args()
    print(json.dumps(verify(Path(arguments.package), arguments.expected_manifest_sha256), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
