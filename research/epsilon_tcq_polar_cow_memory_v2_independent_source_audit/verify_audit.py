#!/usr/bin/env python3
"""Verify the frozen independent v2 source-audit package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "epsilon-tcq-polar-cow-memory-v2-independent-audit-manifest"
STATUS = "FROZEN_PASS_GO_MEMORY_CAPACITY_HOLD_COMPUTE_AND_PAYLOAD"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def verify(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    manifest_raw = (root / "AUDIT_MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS:
        raise RuntimeError("audit manifest schema/status")
    observed = []
    for row in manifest["members"]:
        raw = (root / row["name"]).read_bytes()
        if len(raw) != row["bytes"] or sha(raw) != row["sha256"]:
            raise RuntimeError(f"audit member mismatch: {row['name']}")
        observed.append({"name": row["name"], "bytes": len(raw), "sha256": sha(raw)})
    if [row["name"] for row in observed] != sorted(
            (row["name"] for row in observed), key=lambda value: value.encode("utf-8")):
        raise RuntimeError("audit member ordering")
    if sha(canonical(observed)) != manifest["audit_root_sha256"]:
        raise RuntimeError("audit root")
    if ({path.name for path in root.iterdir()} !=
            ({row["name"] for row in observed} | {"AUDIT_MANIFEST.json"})):
        raise RuntimeError("undeclared audit member")
    receipt = json.loads((root / "AUDIT_RECEIPT.json").read_text(encoding="ascii"))
    if receipt["verdicts"] != {"memory": "GO_MEMORY_CAPACITY",
                                "compute": "HOLD_COMPUTE_AND_DEVICE_COW_IMPLEMENTATION",
                                "payload": "HOLD_PAYLOAD"}:
        raise RuntimeError("receipt split verdict")
    if receipt["qwen_payload_accessed"] or receipt["current_codec_payload_accessed"]:
        raise RuntimeError("payload access")
    if (receipt["cupy"]["sha256"] !=
            "083979b2531066e0a81f4bec3a9afa5dd027d4cd934b6fb9ce240491fa099c14"):
        raise RuntimeError("CuPy receipt pin")
    return {"schema": "epsilon-tcq-polar-cow-v2-audit-verification",
            "status": "PASS", "manifest_sha256": sha(manifest_raw),
            "audit_root_sha256": manifest["audit_root_sha256"],
            "members": len(observed), "verdicts": receipt["verdicts"],
            "qwen_payload_accessed": False, "current_codec_payload_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    print(json.dumps(verify(Path(parser.parse_args().package)), sort_keys=True,
                     separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
