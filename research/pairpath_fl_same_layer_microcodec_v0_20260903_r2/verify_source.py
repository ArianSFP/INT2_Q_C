#!/usr/bin/env python3
"""Fail-closed verifier for the PAIRPATH-P2 r2 source-only closure."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import stat
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import numpy as np

import pairpath_r2_core as core
import run_gate
from source_free_fixtures import aligned_fixture, iid_fixture


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


def verify_closure(package: Path, expected_manifest_sha256: str) -> tuple[dict, str]:
    manifest_path = package / "SOURCE_MANIFEST.json"
    if sha256(manifest_path) != expected_manifest_sha256.lower():
        raise RuntimeError("manifest digest mismatch")
    manifest = canonical_json(manifest_path)
    if manifest.get("schema") != "pairpath_p2_source_manifest_v2":
        raise RuntimeError("manifest schema")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise RuntimeError("manifest rows")
    names = [row.get("name") for row in rows]
    if names != sorted(names) or len(names) != len(set(names)):
        raise RuntimeError("manifest member order")
    actual = sorted(path.name for path in package.iterdir())
    if actual != sorted(names + ["SOURCE_MANIFEST.json"]):
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
    return manifest, root


def verify(package: Path, expected_manifest_sha256: str) -> dict:
    package = package.resolve(strict=True)
    manifest, source_root = verify_closure(package, expected_manifest_sha256)
    lock = json.loads((package / "design_lock.json").read_text(encoding="utf-8"))
    receipt = canonical_json(package / "SOURCE_ONLY_TEST_RECEIPT.json")
    if lock.get("schema") != "pairpath_p2_executable_source_design_lock_v2" or \
            lock.get("status") != "EXECUTABLE_SOURCE_ONLY_HOLD_PENDING_INDEPENDENT_HOSTILE_AUDIT":
        raise RuntimeError("design lock")
    if receipt.get("status") != "PASS_10_TESTS" or receipt.get("tests_run") != 10:
        raise RuntimeError("test receipt")
    if any(lock.get(key) for key in ("payload_execution_enabled", "qwen_aperture_authorized",
                                     "qwen_payload_opened", "runpod_accessed",
                                     "runpod_execution_enabled")):
        raise RuntimeError("source-only boundary")
    if run_gate.PAYLOAD_EXECUTION_ENABLED or run_gate.LOCAL_GPU_EXECUTION_ENABLED or \
            run_gate.QWEN_APERTURE_AUTHORIZED:
        raise RuntimeError("enabled execution gate")
    if lock["architecture"]["candidate_bank"] != list(core.CANDIDATES) or \
            lock["architecture"]["lambda_grid"] != [str(v) for v in core.LAMBDA_GRID]:
        raise RuntimeError("architecture drift")
    if lock["packet"]["rate_interval"] != [str(core.RATE_MIN), str(core.RATE_MAX)]:
        raise RuntimeError("rate interval")
    if not math.isclose(core.REQUIRED_GAIN_BPW,
                        0.5 * math.log2(core.F0 / core.TARGET_F), rel_tol=0, abs_tol=1e-15):
        raise RuntimeError("gain threshold")

    forbidden = ("paramiko", "requests.", "urllib.request", "socket.",
                 "subprocess.popen", ".safetensors", "huggingface", "cupy", "runpod")
    for name in ("pairpath_r2_core.py", "run_gate.py", "source_free_fixtures.py"):
        text = (package / name).read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in text:
                raise RuntimeError(f"forbidden executable token {token!r} in {name}")

    source = iid_fixture(16384)
    scales = core.estimate_scale_bits(source)
    levels = np.stack([core.levels_per_coordinate(
        scales[e, core.OPTIMIZED_ROLES[0]], source.shape[2]) for e in range(2)])
    starts = core._ideal_initializations(source[:, core.OPTIMIZED_ROLES[0]], levels)
    constants = {(int(q[0, 0]), int(q[1, 0])) for q in starts
                 if np.all(q[0] == q[0, 0]) and np.all(q[1] == q[1, 0])}
    if constants != set(itertools.product(range(core.ALPHABET), repeat=2)):
        raise RuntimeError("multistart constants")
    mi = core.fixed_assignment_mi_ceiling(source)
    if mi.get("conditioning") != "decoder-visible role" or \
            [row.get("role") for row in mi.get("role_rows", [])] != ["up", "down"]:
        raise RuntimeError("role-conditioned MI")
    iid = core.optimistic_single_letter_joint_gate(
        source, (core.LAMBDA_GRID[0], core.LAMBDA_GRID[-1]))
    aligned = core.optimistic_single_letter_joint_gate(
        aligned_fixture(), (core.LAMBDA_GRID[0], core.LAMBDA_GRID[-1]))
    if iid["status"] != "HARD_KILL_OPTIMISTIC_JOINT_GATE_BELOW_0P045" or \
            aligned["status"] != "SURVIVE_OPTIMISTIC_GATE_WITH_PHYSICAL_MARGIN":
        raise RuntimeError("oracle controls")
    return {
        "schema": "pairpath_p2_source_verification_v2",
        "status": "PASS_SOURCE_ONLY_HOLD_PENDING_INDEPENDENT_HOSTILE_AUDIT",
        "manifest_sha256": expected_manifest_sha256.lower(),
        "source_root_sha256": source_root,
        "member_count": len(manifest["files"]),
        "multistart_count": len(starts),
        "iid_gate_status": iid["status"],
        "aligned_gate_status": aligned["status"],
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
