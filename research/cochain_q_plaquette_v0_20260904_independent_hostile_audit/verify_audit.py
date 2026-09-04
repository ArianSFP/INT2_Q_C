#!/usr/bin/env python3
"""Fail-closed verifier for the independent COCHAIN-Q hostile audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TARGET = REPO / "research" / "cochain_q_plaquette_v0_20260904"
TARGET_MANIFEST_SHA256 = "ef12407301265d8e04da9f1ed5afaadff69f0d864c31ef1be4868279506a68b3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(path: Path) -> dict:
    raw = path.read_bytes()
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise RuntimeError(f"noncanonical newline: {path.name}")
    value = json.loads(raw.decode("utf-8"))
    if raw != (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode():
        raise RuntimeError(f"noncanonical JSON: {path.name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=HERE)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    manifest_path = package / "AUDIT_MANIFEST.json"
    if sha256(manifest_path) != args.manifest_sha256.lower():
        raise RuntimeError("audit manifest pin")
    manifest = canonical_json(manifest_path)
    if manifest.get("schema") != "cochain_q_plaquette_hostile_audit_manifest_v0":
        raise RuntimeError("manifest schema")
    rows = manifest.get("files")
    names = [row.get("name") for row in rows]
    if names != sorted(names) or len(names) != len(set(names)):
        raise RuntimeError("member order")
    if sorted(p.name for p in package.iterdir()) != sorted(names + ["AUDIT_MANIFEST.json"]):
        raise RuntimeError("audit package closure")
    canonical_rows = []
    for row in rows:
        path = package / row["name"]
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError("non-regular audit member")
        if set(row) != {"bytes", "name", "sha256"} or \
                path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"audit member mismatch: {row['name']}")
        canonical_rows.append(row)
    root = hashlib.sha256(json.dumps(canonical_rows, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
    if root != manifest.get("audit_root_sha256"):
        raise RuntimeError("audit root")
    if sha256(TARGET / "SOURCE_MANIFEST.json") != TARGET_MANIFEST_SHA256:
        raise RuntimeError("target drift")
    receipt = canonical_json(package / "HOSTILE_AUDIT_RECEIPT.json")
    if receipt.get("status") != "PASS_MECHANISM__BLOCK_QWEN_CAPABILITY_PENDING_REPAIRS":
        raise RuntimeError("receipt verdict")
    if not all(receipt["blockers"][key] for key in
               ("manifest_verifier_accepted_unlisted_member",
                "manifest_verifier_accepted_missing_external_pin",
                "encoder_accepted_noninteger_labels")):
        raise RuntimeError("blocker reproduction absent")
    result = {
        "schema": "cochain_q_plaquette_hostile_audit_verification_v0",
        "status": "PASS_AUDIT_CLOSURE__QWEN_REMAINS_BLOCKED",
        "audit_manifest_sha256": args.manifest_sha256.lower(),
        "audit_root_sha256": root,
        "target_manifest_sha256": TARGET_MANIFEST_SHA256,
        "receipt_sha256": sha256(package / "HOSTILE_AUDIT_RECEIPT.json"),
    }
    if args.rerun:
        process = subprocess.run([sys.executable, "-I", "-B",
                                  str(package / "hostile_audit.py")],
                                 cwd=str(REPO), text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, check=False)
        if process.returncode != 0 or json.loads(process.stdout) != receipt:
            raise RuntimeError("audit rerun mismatch: " + process.stderr)
        result["rerun_passed"] = True
        result["rerun_stdout_sha256"] = hashlib.sha256(process.stdout.encode()).hexdigest()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
