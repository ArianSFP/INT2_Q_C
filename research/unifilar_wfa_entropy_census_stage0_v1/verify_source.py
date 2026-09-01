#!/usr/bin/env python3
"""Verify the sealed UWFA v1 source package without payload or CUDA access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def regular_leaf_bytes(path: Path) -> bytes:
    require(path.is_absolute(), f"absolute leaf required: {path}")
    require(os.path.lexists(path), f"missing leaf: {path.name}")
    info = os.lstat(path)
    require(not stat.S_ISLNK(info.st_mode), f"symlink leaf forbidden: {path.name}")
    require(stat.S_ISREG(info.st_mode), f"not a regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags)
    try:
        observed = os.fstat(fd)
        require(stat.S_ISREG(observed.st_mode), f"descriptor not regular: {path.name}")
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise VerificationError(f"nonfinite JSON: {value}")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(f"invalid JSON: {exc}") from exc
    require(isinstance(value, dict), "JSON root object")
    return value


def verify_package(package: Path) -> dict[str, Any]:
    package = package.absolute()
    require(package.is_dir(), "package directory")
    manifest_data = regular_leaf_bytes(package / "SOURCE_MANIFEST.json")
    manifest = load_json(manifest_data)
    require(manifest.get("schema") == "unifilar-wfa-source-manifest-v1", "manifest schema")
    require(manifest.get("status") == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "manifest status")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "manifest members")
    names: set[str] = set()
    observed = []
    for row in rows:
        require(isinstance(row, dict), "manifest row")
        name = row.get("name")
        require(isinstance(name, str) and name == Path(name).name and name != "SOURCE_MANIFEST.json", "manifest name")
        require(name not in names, f"duplicate manifest member: {name}")
        names.add(name)
        data = regular_leaf_bytes(package / name)
        require(len(data) == row.get("bytes"), f"member bytes: {name}")
        require(sha256(data) == row.get("sha256"), f"member hash: {name}")
        observed.append({"name": name, "bytes": len(data), "sha256": sha256(data)})
    disk_names = {
        child.name for child in package.iterdir()
        if child.name != "__pycache__"
    }
    require(disk_names == names | {"SOURCE_MANIFEST.json"}, "undeclared or missing package member")
    required = {
        "README.md", "design_lock.json", "uwfa_common.py", "cupy_backend.py",
        "stage0_census.py", "fixture_long_memory.py", "test_source_only.py", "verify_source.py",
    }
    require(names == required, "frozen member set")
    design = load_json(regular_leaf_bytes(package / "design_lock.json"))
    require(design.get("schema") == "unifilar-wfa-entropy-census-design-v1", "design schema")
    require(design.get("status") == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "design status")
    attestation = design.get("access_attestation", {})
    for field in (
        "model_or_qwen_payload_opened_statted_hashed_or_enumerated",
        "current_finite_artifact_or_selected_stream_opened",
        "gaussian_control_opened",
        "numpy_imported",
        "cupy_imported",
        "cuda_initialized",
        "gpu_job_launched",
    ):
        require(attestation.get(field) is False, f"access attestation: {field}")
    common_path = package / "uwfa_common.py"
    spec = importlib.util.spec_from_file_location("uwfa_verify_common", common_path)
    require(spec is not None and spec.loader is not None, "common module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules["uwfa_verify_common"] = module
    spec.loader.exec_module(module)
    require(len(module.candidate_bank()) == 150, "candidate bank size")
    require(module.STATE_SIZES == (2, 4, 8, 16, 32, 64), "state sizes")
    require(module.RESET_LENGTHS == (32, 128, 512, 2048, 4096), "reset lengths")
    require(abs(module.STANDALONE_REQUIRED_SAVING_BPW - 0.15288996696291447) < 1e-15, "physical threshold")
    source_text = regular_leaf_bytes(package / "stage0_census.py").decode("utf-8")
    cupy_at = source_text.index("import cupy as cp")
    require(source_text.index("bootstrap_source(Path(args.review_receipt))") < cupy_at, "manifest before CuPy")
    require(source_text.index("source_panel = load_panel") < cupy_at, "baseline before CuPy")
    receipt = {
        "schema": "unifilar-wfa-source-verification-v1",
        "status": "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY",
        "source_manifest_sha256": sha256(manifest_data),
        "members": observed,
        "candidate_cells": 150,
        "payload_authority_granted": False,
        "access_attestation_replayed": attestation,
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    receipt = verify_package(Path(args.package))
    if args.compact:
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_SOURCE_VERIFICATION: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
