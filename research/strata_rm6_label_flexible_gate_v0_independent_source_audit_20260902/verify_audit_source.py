#!/usr/bin/env python3
"""Verify the frozen independent STRATA-RM6 audit source package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "strata-rm6-label-flexible-gate-v0-independent-audit-source-manifest-20260902"
STATUS = "FROZEN_AUDIT_SOURCE_PENDING_EXECUTION_ARTIFACT"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def root_hash(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["name"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(row["sha256"])))
    return digest.hexdigest()


def verify(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    raw = (root / "AUDIT_SOURCE_MANIFEST.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS:
        raise RuntimeError("manifest schema/status")
    rows = manifest.get("members")
    if not isinstance(rows, list):
        raise RuntimeError("manifest members")
    observed = []
    for row in rows:
        member = (root / str(row["name"]))
        member_raw = member.read_bytes()
        actual = {"name": row["name"], "bytes": len(member_raw),
                  "sha256": sha(member_raw)}
        if actual != row:
            raise RuntimeError(f"member mismatch: {row['name']}")
        observed.append(actual)
    if [row["name"] for row in observed] != sorted(
            (row["name"] for row in observed), key=lambda value: value.encode("utf-8")):
        raise RuntimeError("member order")
    if root_hash(observed) != manifest.get("audit_source_root_sha256"):
        raise RuntimeError("audit source root")
    expected = {row["name"] for row in observed} | {"AUDIT_SOURCE_MANIFEST.json"}
    if {path.name for path in root.iterdir()} != expected:
        raise RuntimeError("audit source closure")
    return {"schema": "strata-rm6-independent-audit-source-verification-20260902",
            "status": "PASS", "manifest_sha256": sha(raw),
            "audit_source_root_sha256": manifest["audit_source_root_sha256"],
            "members": len(observed), "payload_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    print(json.dumps(verify(Path(parser.parse_args().package)), sort_keys=True,
                     separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
