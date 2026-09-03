#!/usr/bin/env python3
"""External bootstrap verifier: trusts no code inside the v2 package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "tactic-ramanujan384-scalable-source-manifest-v2"


def fail(message: str) -> None:
    raise SystemExit(message)


def strict_json(payload: bytes) -> dict[str, Any]:
    def pairs(rows):
        output = {}
        for key, value in rows:
            if key in output:
                fail("duplicate manifest key")
            output[key] = value
        return output
    result = json.loads(payload.decode("ascii"), object_pairs_hook=pairs,
                        parse_constant=lambda value: fail(f"nonfinite {value}"))
    if not isinstance(result, dict):
        fail("manifest object")
    return result


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--bootstrap-sha256", required=True)
    arguments = parser.parse_args()
    own = Path(__file__).resolve(strict=True)
    if hashlib.sha256(own.read_bytes()).hexdigest() != arguments.bootstrap_sha256:
        fail("external bootstrap verifier SHA256")
    package = arguments.package.resolve(strict=True)
    if package != arguments.package.absolute() or not package.is_dir():
        fail("canonical package directory")
    manifest_path = package / "SOURCE_MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        fail("manifest must be a regular nonsymlink file")
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != arguments.manifest_sha256:
        fail("manifest SHA256")
    document = strict_json(payload)
    if document.get("schema") != SCHEMA:
        fail("manifest schema")
    rows = document.get("members")
    if not isinstance(rows, list) or not rows:
        fail("manifest members")
    canonical_rows = []
    names = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "bytes", "sha256"}:
            fail("member schema")
        name, size, digest = row["name"], row["bytes"], row["sha256"]
        if (not isinstance(name, str) or not name or "/" in name or "\\" in name
                or name in names):
            fail("member name")
        if type(size) is not int or size <= 0 or not isinstance(digest, str) or len(digest) != 64:
            fail("member fields")
        names.append(name)
        path = package / name
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            fail(f"member type {name}")
        member_payload = path.read_bytes()
        if len(member_payload) != size or hashlib.sha256(member_payload).hexdigest() != digest:
            fail(f"member drift {name}")
        canonical_rows.append({"name": name, "bytes": size, "sha256": digest})
    canonical_rows.sort(key=lambda row: row["name"])
    if rows != canonical_rows:
        fail("members not canonical sorted order")
    actual_entries = sorted(path.name for path in package.iterdir())
    expected_entries = sorted(names + ["SOURCE_MANIFEST.json"])
    if actual_entries != expected_entries:
        fail("extra, missing, or nested package entry")
    root = hashlib.sha256(canonical_json(canonical_rows)).hexdigest()
    if root != document.get("source_root_sha256"):
        fail("source root")
    dependencies = document.get("dependency_pins", {})
    for key, relative in (
        ("v1", "tactic_ramanujan384_adapter_v1_authority/SOURCE_MANIFEST.json"),
        ("v1_review", "tactic_ramanujan384_adapter_v1_authority_independent_review_20260902/SOURCE_MANIFEST.json"),
    ):
        path = package.parent / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != dependencies.get(key, {}).get(
                "source_manifest_sha256"):
            fail(f"dependency drift {key}")
    print(json.dumps({
        "schema": "tactic-ramanujan384-v2-external-bootstrap-receipt",
        "status": "PASS_EXACT_CLOSURE_NO_EXTRA_ENTRIES",
        "source_root_sha256": root, "members": len(rows),
        "qwen_payload_accessed": False, "network_accessed": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
