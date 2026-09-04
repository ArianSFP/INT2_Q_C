#!/usr/bin/env python3
"""Fail-closed verifier for the independent PAIRPATH-P2 r2 hostile audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys


PACKAGE = Path(__file__).resolve().parent
TARGET_MANIFEST_SHA256 = "21983efff5ac5c0593a655cae4136d35ca24400fd807f9fe4be458a34b18e622"
TARGET_ROOT_SHA256 = "7ffb0b9c92861c7171a3b89f47d6fa03caac963322d772fb8c0b020ce501cf96"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    expected = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if raw != expected:
        raise RuntimeError(f"noncanonical JSON: {path.name}")
    return value


def closure(package: Path, expected_manifest_sha256: str) -> tuple[dict, str]:
    manifest_path = package / "SOURCE_MANIFEST.json"
    if sha256(manifest_path) != expected_manifest_sha256.lower():
        raise RuntimeError("audit manifest digest")
    manifest = canonical(manifest_path)
    if manifest.get("schema") != "pairpath_p2_r2_independent_audit_manifest_v1":
        raise RuntimeError("audit manifest schema")
    rows = manifest.get("files")
    if not isinstance(rows, list) or [r.get("name") for r in rows] != sorted(
            r.get("name") for r in rows):
        raise RuntimeError("audit member order")
    names = [r["name"] for r in rows]
    if len(names) != len(set(names)) or sorted(path.name for path in package.iterdir()) != \
            sorted(names + ["SOURCE_MANIFEST.json"]):
        raise RuntimeError("audit closure")
    canonical_rows = []
    for row in rows:
        if set(row) != {"bytes", "name", "sha256"}:
            raise RuntimeError("audit member schema")
        path = package / row["name"]
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError("audit member type")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"audit member mismatch: {row['name']}")
        canonical_rows.append({"bytes": row["bytes"], "name": row["name"],
                               "sha256": row["sha256"]})
    root = hashlib.sha256(json.dumps(
        canonical_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if root != manifest.get("source_root_sha256"):
        raise RuntimeError("audit root")
    return manifest, root


def target_closure(package: Path) -> str:
    target = package.parent / "pairpath_fl_same_layer_microcodec_v0_20260903_r2"
    manifest_path = target / "SOURCE_MANIFEST.json"
    if sha256(manifest_path) != TARGET_MANIFEST_SHA256:
        raise RuntimeError("target manifest drift")
    manifest = canonical(manifest_path)
    rows = manifest["files"]
    if sorted(path.name for path in target.iterdir()) != sorted(
            [r["name"] for r in rows] + ["SOURCE_MANIFEST.json"]):
        raise RuntimeError("target closure drift")
    canonical_rows = []
    for row in rows:
        path = target / row["name"]
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) or \
                path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"target member drift: {row['name']}")
        canonical_rows.append({"bytes": row["bytes"], "name": row["name"],
                               "sha256": row["sha256"]})
    root = hashlib.sha256(json.dumps(
        canonical_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if root != TARGET_ROOT_SHA256 or manifest["source_root_sha256"] != root:
        raise RuntimeError("target root drift")
    return root


def verify(package: Path, expected_manifest_sha256: str, self_test: bool) -> dict:
    package = package.resolve(strict=True)
    manifest, root = closure(package, expected_manifest_sha256)
    target_root = target_closure(package)
    report = canonical(package / "AUDIT_REPORT.json")
    if report.get("schema") != "pairpath_p2_r2_independent_hostile_audit_v1" or \
            report.get("verdict") != "BLOCK_R2_NO_PAYLOAD_CAPABILITY_OR_HARD_KILL_AUTHORITY":
        raise RuntimeError("audit verdict")
    expected_findings = {
        "BLOCK_FINITE_ENCODER_USES_ROLE_LOCAL_MULTIPLIERS",
        "BLOCK_KILL_ORACLE_HAS_NO_DOMINANCE_OR_GLOBAL_OPTIMALITY_CERTIFICATE",
        "BLOCK_DECODER_ACCEPTS_INVALID_UNREPLAYED_TREE_DESCRIPTOR",
    }
    actual_findings = {report["global_bit_weight"]["finding"],
                       report["joint_solver_dominance"]["finding"],
                       report["literal_packet"]["finding"]}
    if actual_findings != expected_findings or len(report.get("blockers", [])) != 3:
        raise RuntimeError("audit blocker set")
    if any(report.get(k) for k in ("qwen_payload_opened", "gpu_accessed",
                                   "network_accessed", "runpod_accessed")):
        raise RuntimeError("audit boundary")
    if report["target_closure"]["manifest_sha256"] != TARGET_MANIFEST_SHA256 or \
            report["target_closure"]["source_root_sha256"] != TARGET_ROOT_SHA256:
        raise RuntimeError("reported target closure")
    result = {"schema": "pairpath_p2_r2_independent_audit_verification_v1",
              "status": "PASS_SEALED_AUDIT__BLOCK_TARGET_R2",
              "manifest_sha256": expected_manifest_sha256.lower(),
              "source_root_sha256": root, "member_count": len(manifest["files"]),
              "target_source_root_sha256": target_root, "self_test_passed": False}
    if self_test:
        process = subprocess.run(
            [sys.executable, "-I", "-B", str(package / "hostile_audit.py")],
            cwd=str(package.parent.parent), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if process.returncode != 0:
            raise RuntimeError("hostile audit replay failed:\n" + process.stdout)
        replay = json.loads(process.stdout.strip().splitlines()[-1])
        if replay != report:
            raise RuntimeError("hostile audit replay/report mismatch")
        result["self_test_passed"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.manifest_sha256, args.self_test),
                     sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
