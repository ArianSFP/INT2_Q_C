#!/usr/bin/env python3
"""Fail-closed verifier for the local-3060 CuPy preflight audit."""

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
TARGET_RECEIPT_SHA256 = "a6c1fd514ddafa5a3225a4c70b030cf80df75a41f127f829f8ccd4b92cbe53ab"
EXPECTED_UUID = "GPU-458a424a-76e3-65e5-0470-803e0ed131ca"


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


def verify_closure(package: Path, manifest_sha256: str) -> tuple[dict, str]:
    path = package / "SOURCE_MANIFEST.json"
    if sha256(path) != manifest_sha256.lower():
        raise RuntimeError("manifest SHA")
    manifest = canonical(path)
    if manifest.get("schema") != "pairpath_p2_local3060_preflight_audit_manifest_v1":
        raise RuntimeError("manifest schema")
    rows = manifest.get("files")
    if not isinstance(rows, list) or [r.get("name") for r in rows] != sorted(
            r.get("name") for r in rows):
        raise RuntimeError("member order")
    names = [r["name"] for r in rows]
    if len(names) != len(set(names)) or sorted(path.name for path in package.iterdir()) != \
            sorted(names + ["SOURCE_MANIFEST.json"]):
        raise RuntimeError("package closure")
    canonical_rows = []
    for row in rows:
        if set(row) != {"bytes", "name", "sha256"}:
            raise RuntimeError("member schema")
        member = package / row["name"]
        if member.is_symlink() or not stat.S_ISREG(member.lstat().st_mode) or \
                member.stat().st_size != row["bytes"] or sha256(member) != row["sha256"]:
            raise RuntimeError(f"member mismatch: {row['name']}")
        canonical_rows.append({"bytes": row["bytes"], "name": row["name"],
                               "sha256": row["sha256"]})
    root = hashlib.sha256(json.dumps(
        canonical_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if root != manifest.get("source_root_sha256"):
        raise RuntimeError("source root")
    return manifest, root


def verify_target(package: Path, report: dict) -> None:
    target = package.parent / "pairpath_p2_local3060_cupy_preflight_v0"
    if sha256(target / "SOURCE_FREE_PREFLIGHT_RECEIPT.json") != TARGET_RECEIPT_SHA256:
        raise RuntimeError("target receipt drift")
    names = sorted(path.name for path in target.iterdir())
    if names != sorted(report["target_source_hashes"]):
        raise RuntimeError("target closure drift")
    for name, digest in report["target_source_hashes"].items():
        path = target / name
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) or sha256(path) != digest:
            raise RuntimeError(f"target source drift: {name}")


def semantic_check(report: dict) -> None:
    if report.get("schema") != "pairpath_p2_local3060_cupy_preflight_v0_independent_audit_v1" or \
            report.get("verdict") != "PASS_RUNTIME_PARITY__BLOCK_PAYLOAD_AND_HARD_KILL_AUTHORITY":
        raise RuntimeError("verdict")
    if report["runtime"]["device_uuid"] != EXPECTED_UUID or \
            not report["parity"]["complete_oracle_cpu_cupy_exact"] or \
            not report["parity"]["fresh_cpu_matches_frozen_receipt"] or \
            not report["parity"]["fresh_cupy_matches_frozen_receipt"]:
        raise RuntimeError("runtime parity")
    blocker = report["inherited_joint_solver_blocker"]
    if blocker["finding"] != "BLOCK_GPU_PARITY_FAITHFULLY_REPRODUCES_UNCERTIFIED_JOINT_SOLVER" or \
            not blocker["cpu_gpu_labels_exact"] or blocker["suboptimality_gap"] <= 0:
        raise RuntimeError("inherited solver blocker")
    if report["inherited_r2_findings"] != {
            "finite_r2_role_local_multiplier": "NOT_EXERCISED_AND_REMAINS_BLOCKED",
            "finite_r2_tree_descriptor_validation": "NOT_EXERCISED_AND_REMAINS_BLOCKED",
            "joint_oracle_dominance_certificate": "BLOCK_REPRODUCED_ON_CPU_AND_CUPY",
            "oracle_global_multiplier": "PASS_IN_PREFLIGHT_BACKEND"}:
        raise RuntimeError("r2 blocker inheritance")
    if any(report.get(key) for key in ("qwen_payload_opened", "network_accessed",
                                       "runpod_accessed", "target_modified")):
        raise RuntimeError("audit boundary")
    perf = report["performance_and_memory"]
    if not perf["analytic_memory_ledgers_exactly_recomputed"] or \
            perf["one_eighth_analytic_explicit_bytes"] != 17568128:
        raise RuntimeError("memory evidence")
    for value in perf["fresh_timings"].values():
        if not math.isfinite(value) or not 0 < value < 1:
            raise RuntimeError("timing evidence")


def compare_replay(report: dict, replay: dict) -> None:
    for key in ("schema", "target", "target_receipt_sha256", "target_source_hashes",
                "source_and_receipt_closure_passed", "parity",
                "inherited_joint_solver_blocker", "inherited_r2_findings", "verdict",
                "qwen_payload_opened", "network_accessed", "runpod_accessed",
                "target_modified"):
        if replay.get(key) != report.get(key):
            raise RuntimeError(f"runtime replay drift: {key}")
    semantic_check(replay)
    # Timings and allocator pooling are intentionally empirical. Validate their
    # scope/invariants rather than demanding byte-identical reruns.
    for aperture, estimate in replay["performance_and_memory"][
            "fresh_aperture_linear_estimates_seconds"].items():
        if not math.isfinite(estimate) or estimate <= 0 or aperture not in ("1/64", "1/8"):
            raise RuntimeError("replay timing projection")


def verify(package: Path, manifest_sha256: str, self_test: bool) -> dict:
    package = package.resolve(strict=True)
    manifest, root = verify_closure(package, manifest_sha256)
    report = canonical(package / "AUDIT_REPORT.json")
    semantic_check(report)
    verify_target(package, report)
    result = {"schema": "pairpath_p2_local3060_preflight_audit_verification_v1",
              "status": "PASS_SEALED_RUNTIME_AUDIT__BLOCK_PAYLOAD_AND_HARD_KILL",
              "manifest_sha256": manifest_sha256.lower(),
              "source_root_sha256": root, "member_count": len(manifest["files"]),
              "target_receipt_sha256": TARGET_RECEIPT_SHA256,
              "device_uuid": report["runtime"]["device_uuid"],
              "self_test_passed": False}
    if self_test:
        process = subprocess.run(
            [sys.executable, "-I", "-B", str(package / "runtime_audit.py")],
            cwd=str(package.parent.parent), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        if process.returncode != 0:
            raise RuntimeError("runtime audit replay failed:\n" + process.stdout)
        replay = json.loads(process.stdout.strip().splitlines()[-1])
        compare_replay(report, replay)
        verify_target(package, report)
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
