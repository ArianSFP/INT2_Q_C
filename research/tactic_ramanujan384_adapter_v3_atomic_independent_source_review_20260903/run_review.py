#!/usr/bin/env python3
"""Offline authenticated runner for the independent atomic-v3 source review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PRODUCER_MANIFEST_SHA = (
    "97fb4cba64ff884615810fc8fc835c12ce98bf3e9db37b8a77be93d0d5372be1"
)
PRODUCER_ROOT = (
    "5f86d9a1b48f7769867c828322132be303617d0444d50b5439f7b9d0074ab674"
)
BOOTSTRAP_SHA = (
    "f7e8cd469b0ff9dd9ef09b400c63ec9f91e067f849d6b009588ea94ad6494375"
)
DISPOSITION = (
    "PASS_V2_ATOMIC_SNAPSHOT_BYTE_WORKER_AND_SCIENTIFIC_SEMANTICS_REPAIRS__"
    "CONDITIONAL_ON_PREVERIFIED_ISOLATED_BOOTSTRAP_AND_EXTERNALLY_"
    "INDEPENDENT_NONALIASED_WORKER_AUDIT__"
    "HOLD_REAL_COARSE_DECODER_PYTHON_CUPY_PAYLOAD_AND_RD"
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def authenticate_producer(directory: Path) -> dict:
    manifest_path = directory / "SOURCE_MANIFEST.json"
    payload = manifest_path.read_bytes()
    if sha256(payload) != PRODUCER_MANIFEST_SHA:
        raise RuntimeError("producer manifest digest mismatch")
    manifest = json.loads(payload)
    rows = []
    expected_names = set()
    for expected in manifest["members"]:
        name = expected["name"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in expected_names
        ):
            raise RuntimeError("producer manifest has unsafe/duplicate member")
        member = (directory / name).read_bytes()
        observed = {
            "name": name,
            "bytes": len(member),
            "sha256": sha256(member),
        }
        if observed != expected:
            raise RuntimeError(f"producer member mismatch: {name}")
        rows.append(observed)
        expected_names.add(name)
    if sha256(canonical(rows)) != PRODUCER_ROOT:
        raise RuntimeError("producer canonical source root mismatch")
    actual_names = {item.name for item in directory.iterdir()}
    if actual_names != expected_names | {"SOURCE_MANIFEST.json"}:
        raise RuntimeError("producer directory is not an exact flat closure")
    return manifest


def run_test(path: Path) -> dict:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
    }
    command = [sys.executable, "-I", "-B", str(path)]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    producer = arguments.producer.resolve(strict=True)
    bootstrap = arguments.bootstrap.resolve(strict=True)
    review = Path(__file__).resolve().parent
    manifest = authenticate_producer(producer)
    if sha256(bootstrap.read_bytes()) != BOOTSTRAP_SHA:
        raise RuntimeError("external bootstrap digest mismatch")

    producer_tests = run_test(producer / "test_source_only.py")
    independent_tests = run_test(review / "test_benign_review.py")
    receipt = {
        "schema": "tactic-ramanujan384-atomic-v3-independent-runtime-review-v1",
        "disposition": DISPOSITION,
        "producer_manifest_sha256": PRODUCER_MANIFEST_SHA,
        "producer_source_root_sha256": PRODUCER_ROOT,
        "external_bootstrap_sha256": BOOTSTRAP_SHA,
        "producer_declared_execution": manifest["execution"],
        "network_or_payload_access": False,
        "tests": {
            "producer_source_only": producer_tests,
            "independent_benign": independent_tests,
        },
        "pass": (
            producer_tests["returncode"] == 0
            and independent_tests["returncode"] == 0
        ),
    }
    arguments.output.write_bytes(canonical(receipt) + b"\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
