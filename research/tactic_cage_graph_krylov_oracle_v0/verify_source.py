#!/usr/bin/env python3
"""Independent standard-library verifier for the frozen source package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


SCHEMA = "tactic-cage-graph-krylov-oracle-v0-source-manifest"
STATUS = "SEALED_SOURCE_ONLY_AWAITING_EXPLICIT_QWEN_PILOT"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"duplicate key {key!r}")
            result[key] = value
        return result
    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            VerifyError(f"nonfinite {item}")))
    require(isinstance(value, dict), "manifest JSON object")
    return value


def read_regular(path: Path, label: str) -> bytes:
    metadata = os.lstat(path)
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
            f"{label}: sole-link regular")
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label}: short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", f"{label}: trailing read")
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_nlink), f"{label}: identity drift")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir(), "package directory")
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json", "manifest")
    manifest_sha = sha256(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(HEX64.fullmatch(expected_manifest_sha256) is not None and
                manifest_sha == expected_manifest_sha256,
                "external manifest SHA-256")
    manifest = strict_json(manifest_payload)
    require(set(manifest) == {
        "schema", "status", "source_root_sha256", "members",
        "access_attestation", "claim_boundary",
    }, "manifest exact schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "manifest schema/status")
    require(manifest["access_attestation"] == {
        "qwen_or_model_payload_opened_statted_hashed_or_enumerated": False,
        "completed_v6_result_opened_statted_hashed_or_enumerated": False,
        "bf16_source_opened_statted_hashed_or_enumerated": False,
        "matched_control_opened_statted_hashed_or_enumerated": False,
        "cuda_or_cupy_initialized_during_source_build_or_tests": False,
        "source_or_test_process_accessed_network": False,
        "isolated_source_only_tests_passed": True,
    }, "source access attestation")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "member rows")
    names = []
    observed = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "member row schema")
        name = row["name"]
        require(isinstance(name, str) and name and name not in names and
                name != "SOURCE_MANIFEST.json" and "/" not in name and "\\" not in name,
                "safe unique member")
        payload = read_regular(root / name, f"member {name}")
        observed.append({"name": name, "bytes": len(payload),
                         "sha256": sha256(payload)})
        names.append(name)
    require(names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical member order")
    require(rows == observed, "literal member rows")
    require(manifest["source_root_sha256"] == sha256(canonical_json(observed)),
            "source snapshot root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {"SOURCE_MANIFEST.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "exact regular-file package closure")
    required_names = {
        "README.md", "SECONDARY_SCREENS.md", "design_lock.json",
        "oracle_core.py", "run_oracle.py", "secondary_hooks.py",
        "test_source_only.py", "verify_source.py",
    }
    require(set(names) == required_names, "complete frozen source set")
    require(not any(name == "cupy" or name.startswith("cupy.") for name in sys.modules),
            "verifier initialized CuPy")
    return {
        "schema": "tactic-cage-graph-krylov-oracle-v0-source-verification",
        "status": "PASS_INDEPENDENT_SOURCE_VERIFICATION",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": observed,
        "cupy_imported": False,
        "payload_accessed": False,
        "result_accessed": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--package", type=Path,
                       default=Path(__file__).resolve().parent)
    value.add_argument("--expected-manifest-sha256")
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(verify(arguments.package, arguments.expected_manifest_sha256),
                     sort_keys=True, separators=(",", ":")))
