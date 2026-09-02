#!/usr/bin/env python3
"""Verify the immutable source-only UWFA-SC v9 deferred-control package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


REQUIRED = (
    "README.md",
    "design_lock.json",
    "BLOCK.json",
    "control_core.py",
    "deferred_controls.py",
    "test_source_only.py",
    "verify_source.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def strict_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise RuntimeError(f"nonfinite JSON: {value}")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), "JSON root")
    return value


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    args = parser.parse_args()
    package = Path(args.package)
    require(package.is_absolute(), "package must be absolute")
    require(package.is_dir() and not package.is_symlink(), "package directory")
    actual = {entry.name for entry in os.scandir(package)}
    require(actual == set(REQUIRED) | {"SOURCE_MANIFEST.json"}, "member closure")
    manifest = strict_json((package / "SOURCE_MANIFEST.json").read_bytes())
    require(
        set(manifest) == {"schema", "status", "members", "access_attestation", "claim_boundary"},
        "manifest fields",
    )
    require(manifest["schema"] == "uwfa-sc-v9-deferred-controls-source-manifest-v0", "manifest schema")
    require(manifest["status"] == "SEALED_SOURCE_ONLY_BOUNDED_BLOCK", "manifest status")
    rows = manifest["members"]
    require(isinstance(rows, list) and [row.get("name") for row in rows] == list(REQUIRED), "manifest order")
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "member row")
        path = package / row["name"]
        info = os.lstat(path)
        require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), "regular member")
        payload = path.read_bytes()
        require(type(row["bytes"]) is int and row["bytes"] == len(payload), "member bytes")
        require(row["sha256"] == sha256(payload), "member digest")
    design = strict_json((package / "design_lock.json").read_bytes())
    block = strict_json((package / "BLOCK.json").read_bytes())
    require(design["status"].startswith("SOURCE_ONLY_BLOCKED"), "design block status")
    require(design["opening_gate"]["v0_payload_access_authority"] is False, "payload authority")
    require(block["status"] == "BLOCK_MISSING_DECODER_CLOSED_MATCHED_CONTROL_PRODUCER_AND_AUDIT_PINS", "block status")
    require(block["payload_access_authority"] is False, "block payload authority")
    attestation = manifest["access_attestation"]
    require(isinstance(attestation, dict) and all(value is False for value in attestation.values()), "access attestation")
    print(json.dumps({
        "schema": "uwfa-sc-v9-deferred-controls-source-verification-v0",
        "status": "PASS_SOURCE_ONLY_BOUNDED_BLOCK_VERIFIED",
        "members": len(rows),
        "payload_access_authority": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
