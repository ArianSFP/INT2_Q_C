#!/usr/bin/env python3
"""Fail-closed verifier for the PAIRPATH-P2 r3 source-only closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import pairpath_r3_core as core
import run_gate
from source_free_fixtures import iid_fixture, unequal_role_energy_fixture


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(path: Path) -> dict:
    raw = path.read_bytes()
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise RuntimeError(f"noncanonical newline: {path.name}")
    value = json.loads(raw.decode("utf-8"))
    expected = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if raw != expected:
        raise RuntimeError(f"noncanonical JSON: {path.name}")
    return value


def verify(package: Path, expected_manifest_sha256: str) -> dict:
    package = package.resolve(strict=True)
    manifest_path = package / "SOURCE_MANIFEST.json"
    if sha256(manifest_path) != expected_manifest_sha256.lower():
        raise RuntimeError("manifest digest mismatch")
    manifest = canonical_json(manifest_path)
    if manifest.get("schema") != "pairpath_p2_r3_source_manifest_v1":
        raise RuntimeError("manifest schema")
    rows = manifest.get("files")
    if not isinstance(rows, list) or [r.get("name") for r in rows] != sorted(
            r.get("name") for r in rows):
        raise RuntimeError("manifest rows/order")
    actual = sorted(path.name for path in package.iterdir())
    if actual != sorted([r["name"] for r in rows] + ["SOURCE_MANIFEST.json"]):
        raise RuntimeError("package closure has missing or extra member")
    canonical_rows = []
    for row in rows:
        if set(row) != {"bytes", "name", "sha256"}:
            raise RuntimeError("manifest member schema")
        path = package / row["name"]
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError(f"non-regular member: {row['name']}")
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError(f"member mismatch: {row['name']}")
        canonical_rows.append({"bytes": int(row["bytes"]), "name": row["name"],
                               "sha256": row["sha256"]})
    root = hashlib.sha256(json.dumps(
        canonical_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if root != manifest.get("source_root_sha256"):
        raise RuntimeError("source root mismatch")

    dependency = manifest.get("dependency")
    dep_path = (package / dependency["path"]).resolve(strict=True)
    if dep_path != core.R2_CORE_PATH.resolve(strict=True) or \
            sha256(dep_path) != dependency["sha256"] or \
            dependency["sha256"] != core.R2_CORE_SHA256:
        raise RuntimeError("pinned dependency mismatch")

    lock = canonical_json(package / "design_lock.json")
    receipt = canonical_json(package / "SOURCE_ONLY_TEST_RECEIPT.json")
    if lock.get("schema") != "pairpath_p2_r3_source_design_lock_v1" or \
            lock.get("status") != "SEALED_SOURCE_ONLY_HOLD_PENDING_INDEPENDENT_HOSTILE_AUDIT":
        raise RuntimeError("design lock")
    if receipt.get("status") != "PASS_9_TESTS" or receipt.get("tests_run") != 9:
        raise RuntimeError("test receipt")
    if any(lock.get(key) for key in ("payload_execution_enabled", "qwen_aperture_authorized",
                                     "qwen_payload_opened", "runpod_accessed",
                                     "runpod_execution_enabled")):
        raise RuntimeError("source-only boundary")
    if run_gate.PAYLOAD_EXECUTION_ENABLED or run_gate.LOCAL_GPU_EXECUTION_ENABLED or \
            run_gate.QWEN_APERTURE_AUTHORIZED:
        raise RuntimeError("enabled execution gate")
    if core.optimistic_single_letter_joint_gate(iid_fixture(),
            (core.LAMBDA_GRID[0],))["hard_kill_authority"]:
        raise RuntimeError("heuristic gained hard-kill authority")

    source = unequal_role_energy_fixture()
    lagrange = core.LAMBDA_GRID[4]
    plan = core._make_plan(source, "pair_k2_fixed", lagrange)
    expected = core.global_updown_bit_weight(source, lagrange).hex()
    role_weights = plan["r3_encoder_certificate"]["optimized_role_bit_weight_hex"]
    if set(role_weights.values()) != {expected}:
        raise RuntimeError("global multiplier repair")
    if not math.isfinite(float.fromhex(expected)):
        raise RuntimeError("global multiplier finite")

    return {
        "schema": "pairpath_p2_r3_source_verification_v1",
        "status": "PASS_SOURCE_ONLY_HOLD_PENDING_INDEPENDENT_HOSTILE_AUDIT",
        "manifest_sha256": expected_manifest_sha256.lower(),
        "source_root_sha256": root,
        "dependency_sha256": dependency["sha256"],
        "member_count": len(rows),
        "global_multiplier_hex": expected,
        "hard_kill_authority": False,
        "qwen_payload_opened": False,
        "gpu_accessed": False,
        "network_accessed": False,
        "runpod_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = verify(args.package, args.manifest_sha256)
    if args.self_test:
        process = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(args.package) / "test_source_only.py")],
            cwd=str(Path(args.package).parent.parent), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        if process.returncode != 0:
            raise RuntimeError("source tests failed:\n" + process.stdout)
        result["self_test_output_sha256"] = hashlib.sha256(
            process.stdout.encode("utf-8")).hexdigest()
        result["self_test_passed"] = True
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
