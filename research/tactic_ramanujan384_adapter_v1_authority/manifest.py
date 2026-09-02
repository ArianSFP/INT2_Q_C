#!/usr/bin/env python3
"""One canonical source-manifest algorithm for both freezing and verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SOURCE_MEMBERS = (
    "AUDIT_REPAIR_MAP.json",
    "README.md",
    "adapter.py",
    "authenticated_io.py",
    "codec_authority.py",
    "contract.py",
    "dependency_lock.json",
    "design_lock.json",
    "freeze_source.py",
    "manifest.py",
    "read_trace.py",
    "run_source_free_cupy_smoke.py",
    "run_source_free_fixture.py",
    "stable_controls.py",
    "test_source_only.py",
    "verify_source.py",
)


class ManifestError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def member_record(path: Path, name: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "name": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def canonical_member_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "manifest member schema")
        name = row["name"]
        size = row["bytes"]
        digest = row["sha256"]
        require(isinstance(name, str) and name and "/" not in name and "\\" not in name,
                "manifest member name")
        require(name not in seen, "duplicate manifest member")
        require(type(size) is int and size > 0, "manifest member size")
        require(isinstance(digest, str) and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                "manifest member digest")
        seen.add(name)
        output.append({"name": name, "bytes": size, "sha256": digest})
    output.sort(key=lambda row: row["name"])
    return output


def source_root(rows: Iterable[dict[str, Any]]) -> str:
    """Hash sorted rows with sorted object keys; this is the sole root algorithm."""

    return hashlib.sha256(canonical_json(canonical_member_rows(rows))).hexdigest()


def collect(root: Path) -> list[dict[str, Any]]:
    rows = []
    for name in SOURCE_MEMBERS:
        path = root / name
        require(path.is_file(), f"missing source member {name}")
        rows.append(member_record(path, name))
    return canonical_member_rows(rows)
