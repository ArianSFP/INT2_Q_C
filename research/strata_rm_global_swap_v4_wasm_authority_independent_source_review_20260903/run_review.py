#!/usr/bin/env python3
"""Run the independent source-only review; never accepts payload paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


EXPECTED_MANIFEST = "62bf04cd413317e2e8b98635713419c84394db7b7d2bd4567afddf56957a5e2f"
EXPECTED_ROOT = "f535699c4828a02e5769b916b1207309768f7381db5f92a0fb58e10915ae8a25"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def regular(path: Path, label: str) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            f"{label}: regular non-link")
    payload = path.read_bytes()
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"{label}: changed")
    return payload


def authenticate(producer: Path) -> dict:
    before = producer.lstat()
    require(stat.S_ISDIR(before.st_mode) and not producer.is_symlink(),
            "producer real directory")
    root = producer.resolve(strict=True)
    manifest_payload = regular(root / "source_manifest.json", "producer manifest")
    require(sha(manifest_payload) == EXPECTED_MANIFEST, "producer manifest pin")
    manifest = json.loads(manifest_payload)
    require(manifest["source_root_sha256"] == EXPECTED_ROOT, "producer root pin")
    observed = []
    for row in manifest["members"]:
        payload = regular(root / row["name"], f"member {row['name']}")
        item = {"name": row["name"], "bytes": len(payload), "sha256": sha(payload)}
        require(item == row, f"member pin {row['name']}")
        observed.append(item)
    require(len(observed) == 9 and sha(canonical(observed)) == EXPECTED_ROOT,
            "nine-member canonical root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} ==
            {row["name"] for row in observed} | {"source_manifest.json"} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "producer exact flat closure")
    return {"root": root, "manifest": manifest}


def run_tests(path: Path, cwd: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(path)], cwd=cwd,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=600, check=False)
    require(completed.returncode == 0,
            f"tests failed: {completed.stderr.decode('utf-8', errors='replace')[-4000:]}")
    return {"returncode": completed.returncode,
            "stdout_tail": completed.stdout.decode("utf-8", errors="replace")[-2000:],
            "stderr_tail": completed.stderr.decode("utf-8", errors="replace")[-4000:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reviewed = authenticate(args.producer)
    review_root = Path(__file__).resolve().parent
    producer_tests = run_tests(reviewed["root"] / "test_source_only.py",
                               reviewed["root"])
    independent_tests = run_tests(review_root / "test_benign_review.py", review_root)
    receipt = {
        "schema": "strata-rm-global-swap-v4-independent-source-review-receipt",
        "producer_manifest_sha256": EXPECTED_MANIFEST,
        "producer_source_root_sha256": EXPECTED_ROOT,
        "producer_members": 9,
        "producer_source_tests": producer_tests,
        "independent_source_tests": independent_tests,
        "producer_tests_declared": 20,
        "independent_tests_declared": 16,
        "wasmtime_imported": False,
        "wasm_guest_executed": False,
        "runtime_or_semantic_audit_package_opened": False,
        "model_or_packet_payload_opened": False,
        "network_used": False,
        "disposition": (
            "PASS_V3_IMMUTABILITY_LIMITS_SEMANTIC_AND_DELEGATED_GATE_REPAIRS__"
            "CONDITIONAL_ON_EXTERNALLY_TRUSTED_EXECUTED_RUNTIME_AND_SEMANTIC_AUDITS__"
            "HOLD_EXECUTING_HOST_AND_TRANSITIVE_NATIVE_PROVENANCE__"
            "HOLD_TOTAL_READ_BANDWIDTH_RUNTIME_PAYLOAD_AND_RD"),
    }
    args.output.write_bytes(canonical(receipt) + b"\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

