#!/usr/bin/env python3
"""Fail-closed verifier for the RENORM-Q source-only closure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np


PACKAGE = Path(__file__).resolve().parent
MODULE_PATH = PACKAGE / "renorm_q_oracle.py"
SPEC = importlib.util.spec_from_file_location("renorm_q_oracle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
rq = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rq
SPEC.loader.exec_module(rq)


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
    if manifest.get("schema") != "renorm_q_smallblock_source_manifest_v0":
        raise RuntimeError("manifest schema")
    rows = manifest.get("files")
    names = [row.get("name") for row in rows]
    if names != sorted(names) or len(names) != len(set(names)):
        raise RuntimeError("manifest member order")
    actual = sorted(p.name for p in package.iterdir())
    if actual != sorted(names + ["SOURCE_MANIFEST.json"]):
        raise RuntimeError("package closure")
    canonical_rows = []
    for row in rows:
        if set(row) != {"bytes", "name", "sha256"}:
            raise RuntimeError("member schema")
        path = package / row["name"]
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError(f"non-regular member: {row['name']}")
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError(f"member mismatch: {row['name']}")
        canonical_rows.append({"bytes": int(row["bytes"]), "name": row["name"],
                               "sha256": row["sha256"]})
    root = hashlib.sha256(json.dumps(canonical_rows, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
    if root != manifest.get("source_root_sha256"):
        raise RuntimeError("source root mismatch")
    lock = json.loads((package / "DESIGN_LOCK.json").read_text(encoding="utf-8"))
    receipt = canonical_json(package / "SOURCE_ONLY_TEST_RECEIPT.json")
    if lock.get("status") != "SEALED_SOURCE_ONLY_HOLD_PENDING_INDEPENDENT_REVIEW":
        raise RuntimeError("design lock status")
    if receipt.get("status") != "PASS_8_TESTS" or receipt.get("tests_run") != 8:
        raise RuntimeError("test receipt")
    if any(lock.get(key) for key in ("qwen_authority", "gpu_authority",
                                     "network_authority", "payload_authority",
                                     "deployment_authority", "finite_codec_claim")):
        raise RuntimeError("authority boundary")
    source = (package / "renorm_q_oracle.py").read_text(encoding="utf-8").lower()
    for token in ("huggingface", "requests", "subprocess", "socket", "cupy",
                  "torch", "ssh", "runpod"):
        if token in source:
            raise RuntimeError(f"forbidden executable token: {token}")
    blocks = np.asarray(list(itertools.product(range(2), repeat=4)), dtype=np.uint8)
    env = np.bitwise_xor.reduce(blocks, axis=1)[:, None]
    xor = rq.collective_variable_census(blocks, env, 2, beta=0.0,
                                        charge_descriptor=False)
    if abs(xor[0]["mutual_information_bits_per_cell"] - 1.0) > 1e-12:
        raise RuntimeError("XOR control")
    if abs(rq.logical_common_private_read_amplification(2, 1, 1) - 4 / 3) > 1e-12:
        raise RuntimeError("read projection")
    return {
        "schema": "renorm_q_smallblock_source_verification_v0",
        "status": "PASS_SOURCE_ONLY_HOLD_PENDING_INDEPENDENT_REVIEW",
        "manifest_sha256": expected_manifest_sha256.lower(),
        "source_root_sha256": root,
        "member_count": len(rows),
        "xor_collective_mi_bits_per_cell": xor[0]["mutual_information_bits_per_cell"],
        "qwen_payload_opened": False,
        "gpu_accessed": False,
        "network_accessed": False,
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
            [sys.executable, "-I", "-B", str(Path(args.package) / "test_source.py")],
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
