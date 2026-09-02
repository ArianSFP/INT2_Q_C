#!/usr/bin/env python3
"""Verify exact closure and the frozen disposition of this audit package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def regular(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        fail(f"regular non-link required: {path.name}")
    payload = path.read_bytes()
    after = path.lstat()
    if ((info.st_size, info.st_mtime_ns, info.st_mode, info.st_ino) !=
            (after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino)):
        fail(f"changed during read: {path.name}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    root = args.package.resolve(strict=True)
    manifest_payload = regular(root / "AUDIT_MANIFEST.json")
    if sha256(manifest_payload) != args.expected_manifest_sha256:
        fail("audit manifest external pin")
    manifest = json.loads(manifest_payload.decode("utf-8"))
    if manifest.get("schema") != "logic-q-v1-capped-adapter-independent-audit-manifest":
        fail("audit manifest schema")
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        fail("audit members")
    observed = []
    names = []
    for row in rows:
        if set(row) != {"name", "bytes", "sha256"}:
            fail("audit member schema")
        name = row["name"]
        if (not isinstance(name, str) or not name or "/" in name or
                "\\" in name or name == "AUDIT_MANIFEST.json" or name in names):
            fail("safe unique audit member")
        payload = regular(root / name)
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        if item != row:
            fail(f"audit member pin: {name}")
        names.append(name)
        observed.append(item)
    root_payload = json.dumps(
        observed, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")
    if sha256(root_payload) != manifest.get("audit_root_sha256"):
        fail("audit observed root")
    if {entry.name for entry in os.scandir(root)} != set(names) | {"AUDIT_MANIFEST.json"}:
        fail("exact audit package closure")

    receipt = json.loads(regular(root / "RUNPOD_AUDIT_RECEIPT.json"))
    if receipt.get("status") != (
            "MECHANISM_VALID__HOLD_BOUND_SELECTOR_SCORER_AND_LIVE_BACKEND"):
        fail("receipt disposition")
    if receipt.get("v1", {}).get("manifest_sha256") != (
            "9bfd3d1225fb45a0518d2d4d6a4035262e87dc62563222e42e69665358b9aac5"):
        fail("receipt v1 pin")
    findings = receipt.get("findings", {})
    required = {
        "selection_receipt_can_be_resealed_with_different_config",
        "pooled_score_trusts_encoder_metric_objects",
        "live_cupy_guard_accepts_name_only_object",
    }
    if set(findings) != required or not all(findings.values()):
        fail("receipt adversarial findings")
    if not receipt.get("real_cupy", {}).get("executed"):
        fail("real CuPy audit path")
    if any(receipt.get(name) for name in (
            "model_or_qwen_payload_accessed",
            "current_codec_or_coarse_payload_accessed",
            "prebuilt_matched_control_accessed")):
        fail("payload-free audit boundary")

    print(json.dumps({
        "schema": "logic-q-v1-capped-adapter-independent-audit-verification",
        "status": "PASS_EXACT_AUDIT_CLOSURE_WITH_PRODUCTION_HOLD",
        "audit_manifest_sha256": args.expected_manifest_sha256,
        "audit_root_sha256": manifest["audit_root_sha256"],
        "receipt_sha256": sha256(regular(root / "RUNPOD_AUDIT_RECEIPT.json")),
        "member_count": len(rows),
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
