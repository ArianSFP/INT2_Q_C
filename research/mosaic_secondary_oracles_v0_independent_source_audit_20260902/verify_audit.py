#!/usr/bin/env python3
"""Standard-library closure verifier for the independent audit package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


EXPECTED = {
    "AUDIT_RESULT.json",
    "CUPY_AUDIT_RECEIPT.json",
    "README.md",
    "audit_source.py",
    "run_cupy_backend_audit.py",
    "test_audit.py",
    "verify_audit.py",
}
UPSTREAM_MANIFEST_SHA256 = "4259e8e8dc87b4c25301ca89ade7dbd63c1e0c9e3415fdaa4d7881d7d10ccc06"
UPSTREAM_ROOT_SHA256 = "60bf8cb7575c165c1e8e648360b9d81f39c092070a9489684904bcf06d0bd820"


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, f"{label} duplicate key")
            output[key] = value
        return output

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(VerifyError(f"{label} nonfinite {token}")),
    )
    require(isinstance(value, dict), f"{label} object")
    return value


def read_regular(path: Path, maximum: int = 4 * (1 << 20)) -> bytes:
    descriptor = os.open(
        os.fspath(path.resolve(strict=True)),
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= maximum, "regular file")
        output = bytearray()
        while len(output) < before.st_size:
            chunk = os.read(descriptor, before.st_size - len(output))
            require(bool(chunk), "short read")
            output.extend(chunk)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
            "identity drift",
        )
        return bytes(output)
    finally:
        os.close(descriptor)


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "real audit directory")
    entries = list(os.scandir(root))
    require(all(entry.is_file(follow_symlinks=False) and not entry.is_symlink() for entry in entries), "audit files only")
    require({entry.name for entry in entries} == EXPECTED | {"SOURCE_MANIFEST.json"}, "exact audit member set")
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json")
    manifest_sha = digest(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(manifest_sha == expected_manifest_sha256, "manifest digest")
    manifest = strict_json(manifest_payload, "manifest")
    require(manifest.get("schema") == "mosaic-secondary-oracles-independent-source-audit-manifest-v1", "schema")
    require(manifest.get("status") == "MECHANISMS_VALID__HOLD_PRODUCTION_ADAPTER_SCORER_BACKEND_AND_IO_BINDING", "status")
    rows = manifest.get("members")
    require(isinstance(rows, list) and [row.get("name") for row in rows] == sorted(EXPECTED), "member rows")
    observed = []
    payloads = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "member row schema")
        payload = read_regular(root / row["name"])
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"], f"member {row['name']}")
        observed.append({"name": row["name"], "bytes": len(payload), "sha256": digest(payload)})
        payloads[row["name"]] = payload
    root_sha = digest(canonical_json(observed))
    require(root_sha == manifest.get("source_root_sha256"), "root digest")
    require(manifest.get("upstream_manifest_sha256") == UPSTREAM_MANIFEST_SHA256, "upstream manifest pin")
    require(manifest.get("upstream_source_root_sha256") == UPSTREAM_ROOT_SHA256, "upstream root pin")

    result = strict_json(payloads["AUDIT_RESULT.json"], "audit result")
    require(result["status"] == manifest["status"], "result status")
    require(result["recurrence"]["exhaustive_binary_sequences_n1_through_n10"] == 2046, "BM cases")
    require(result["recurrence"]["large_synthetic_expert_physical_bytes"] == 61440, "literal bytes")
    require(result["recurrence"]["large_synthetic_expert_physical_rate_bpw"] == 2.5, "literal rate")
    require(result["gate_and_traffic"]["gate_recomputes_source_sse_from_reconstruction"] is False, "scorer hold")
    require(result["gate_and_traffic"]["cold_read_claim_is_observed_runtime_IO"] is False, "IO hold")
    require(result["production_binding"]["direct_alias_is_valid"] is False, "STRATA binding")
    require(result["production_launch_authorized"] is False, "launch hold")
    require(result["qwen_payload_accessed"] is False and result["coarse_payload_accessed"] is False, "payload boundary")

    cupy = strict_json(payloads["CUPY_AUDIT_RECEIPT.json"], "CuPy receipt")
    require(cupy["status"] == "PASS_MECHANICS__BACKEND_NOT_BIT_IDENTICAL", "CuPy status")
    require(cupy["cupy_version"] == "14.2.0" and cupy["device_name"] == "NVIDIA GeForce RTX 5090", "CuPy runtime")
    require(cupy["cupy_array_type"] == "cupy.ndarray", "CuPy arrays")
    require(cupy["basis_bit_identical"] is False and cupy["gaussian_control_bit_identical"] is False, "backend distinction")
    require(cupy["qwen_payload_accessed"] is False and cupy["coarse_payload_accessed"] is False, "CuPy payload boundary")
    return {
        "schema": "mosaic-secondary-oracles-independent-source-audit-verifier-v1",
        "status": "PASS_AUDIT_CLOSURE",
        "manifest_sha256": manifest_sha,
        "source_root_sha256": root_sha,
        "members": len(rows),
        "upstream_manifest_sha256": UPSTREAM_MANIFEST_SHA256,
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "production_launch_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--package", type=Path, required=True)
    result.add_argument("--manifest-sha256")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(verify(args.package, args.manifest_sha256), sort_keys=True, separators=(",", ":")))
