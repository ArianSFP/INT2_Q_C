#!/usr/bin/env python3
"""Verify the frozen independent-audit closure and typed-HOLD receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def verify(root):
    root = root.resolve(strict=True)
    manifest_raw = (root / "AUDIT_MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest["schema"] == "epsilon-tcq-strata-bound-v1-independent-source-audit-manifest"
    assert manifest["status"] == "PASS_SOURCE_AUDIT_TYPED_HOLD_NO_PAYLOAD_AUTHORITY"
    observed = []
    for row in manifest["members"]:
        raw = (root / row["name"]).read_bytes()
        assert len(raw) == row["bytes"] and sha(raw) == row["sha256"]
        observed.append({"name": row["name"], "bytes": len(raw),
                         "sha256": sha(raw)})
    assert [row["name"] for row in observed] == sorted(
        (row["name"] for row in observed), key=lambda value: value.encode("utf-8"))
    assert sha(canonical(observed)) == manifest["source_root_sha256"]
    assert {path.name for path in root.iterdir()} == {
        row["name"] for row in observed} | {"AUDIT_MANIFEST.json"}
    receipt = json.loads((root / "RUNPOD_AUDIT_RECEIPT.json").read_bytes())
    assert receipt["status"] == "PASS_SOURCE_AUDIT_TYPED_HOLD_NO_PAYLOAD_AUTHORITY"
    assert receipt["resource_bound"]["total_peak_bytes"] == 7147102208
    assert receipt["resource_bound"]["status"] == "HOLD_PRODUCTION_POLAR_LIST_SCALABILITY"
    assert receipt["qwen_payload_accessed"] is False
    assert receipt["current_codec_payload_accessed"] is False
    assert receipt["matched_control_payload_accessed"] is False
    return {"schema": "epsilon-tcq-strata-bound-v1-audit-verification",
            "status": "PASS", "audit_manifest_sha256": sha(manifest_raw),
            "audit_source_root_sha256": manifest["source_root_sha256"],
            "members": len(observed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    print(json.dumps(verify(Path(parser.parse_args().audit)), sort_keys=True,
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
