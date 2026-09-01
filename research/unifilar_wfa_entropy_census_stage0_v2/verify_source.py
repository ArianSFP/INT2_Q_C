#!/usr/bin/env python3
"""Native verifier for the sealed source-only UWFA-SC v2 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any


EXPECTED_MEMBERS = {
    "README.md",
    "INDEPENDENT_BOOTSTRAP_ABI.md",
    "design_lock.json",
    "uwfa_common.py",
    "container_codec.py",
    "protocol.py",
    "strata_sc_adapter.py",
    "stage0_census.py",
    "cupy_backend.py",
    "dispatcher_contract.py",
    "result_envelope.py",
    "fixture_long_memory.py",
    "test_source_only.py",
    "verify_source.py",
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def regular_bytes(path: Path) -> bytes:
    require(path.is_absolute(), "absolute source path")
    # Reject every lexical component before opening the leaf.  This verifier
    # is not the authority root; the independent dispatcher later uses held
    # openat descriptors as well.
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor = cursor / component
        require(os.path.lexists(cursor), f"source component absent: {cursor}")
        info = os.lstat(cursor)
        require(not stat.S_ISLNK(info.st_mode), f"source symlink component: {cursor}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode), "source member not regular")
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def strict_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
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


def module_from_snapshot(name: str, source: bytes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = f"<verified-snapshot:{name}>"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec", dont_inherit=True), module.__dict__)
    return module


def verify_package(package: Path) -> dict[str, Any]:
    package = package.absolute()
    require(package.is_dir(), "package directory")
    manifest_bytes = regular_bytes(package / "SOURCE_MANIFEST.json")
    manifest = strict_json(manifest_bytes)
    require(set(manifest) == {"schema", "status", "members", "access_attestation", "post_seal_requirements"}, "manifest fields")
    require(manifest["schema"] == "unifilar-wfa-source-manifest-v2", "manifest schema")
    require(manifest["status"] == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "manifest status")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "manifest members")
    names = set()
    observed = []
    snapshots = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "manifest row fields")
        name = row["name"]
        require(isinstance(name, str) and name == Path(name).name and name not in names and name != "SOURCE_MANIFEST.json", "manifest name")
        data = regular_bytes(package / name)
        require(type(row["bytes"]) is int and len(data) == row["bytes"], f"member bytes: {name}")
        require(isinstance(row["sha256"], str) and sha256(data) == row["sha256"], f"member digest: {name}")
        names.add(name)
        snapshots[name] = data
        observed.append({"name": name, "bytes": len(data), "sha256": sha256(data)})
    require(names == EXPECTED_MEMBERS, "frozen member set")
    actual = {entry.name for entry in os.scandir(package)}
    require(actual == names | {"SOURCE_MANIFEST.json"}, "undeclared or missing source member")
    design = strict_json(snapshots["design_lock.json"])
    require(design.get("schema") == "unifilar-wfa-entropy-census-design-v2", "design schema")
    require(design.get("status") == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "design status")
    attestation = manifest["access_attestation"]
    require(attestation == design.get("access_attestation"), "manifest/design attestation")
    required_false = (
        "model_or_qwen_payload_opened_statted_hashed_or_enumerated",
        "current_finite_artifact_or_selected_stream_opened_statted_hashed_or_enumerated",
        "gaussian_control_opened_statted_hashed_or_enumerated",
        "numpy_imported_by_builder",
        "cupy_imported_by_builder",
        "cuda_initialized_by_builder",
        "gpu_job_launched_by_builder",
    )
    require(set(attestation) == set(required_false), "attestation fields")
    require(all(attestation[name] is False for name in required_false), "access attestation must be all false")
    common = module_from_snapshot("uwfa_verify_common_v2", snapshots["uwfa_common.py"])
    require(len(common.candidate_bank()) == 150, "candidate bank")
    require(common.STATE_SIZES == (2, 4, 8, 16, 32, 64), "state sizes")
    require(common.RESET_LENGTHS == (32, 128, 512, 2048, 4096), "reset lengths")
    require(abs(common.STANDALONE_REQUIRED_SAVING_BPW - 0.15288996696291447) < 1e-15, "standalone threshold")
    stage_text = snapshots["stage0_census.py"].decode("utf-8")
    require("BLOCK_DIRECT_EXECUTION_REQUIRES_EXTERNALLY_PINNED_DISPATCHER" in stage_text, "direct-launch block")
    require("import cupy" not in stage_text and "spec_from_file_location" not in stage_text, "producer path/dynamic import")
    adapter_text = snapshots["strata_sc_adapter.py"].decode("utf-8")
    require("sys.path" not in adapter_text and "strata_expert_local_codec" not in adapter_text, "repository-relative decoder import")
    post = manifest["post_seal_requirements"]
    require(post == [
        "EXTERNAL_PINNED_DISPATCHER_INDEPENDENT_AUDIT",
        "ALL_150_CPU_CUPY_RUNPOD_PREFLIGHT",
        "NO_PAYLOAD_LAUNCH_BEFORE_BOTH_PASS",
        "FRESH_PROCESS_INDEPENDENT_RESULT_AUDIT_AFTER_ANY_NUMERIC_RUN",
    ], "post-seal requirement list")
    return {
        "schema": "unifilar-wfa-source-verification-v2",
        "status": "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY",
        "source_manifest_sha256": sha256(manifest_bytes),
        "members": observed,
        "candidate_cells": 150,
        "payload_authority_granted": False,
        "post_seal_requirements": post,
        "access_attestation_replayed": attestation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = verify_package(Path(args.package))
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) if args.compact else json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_SOURCE_VERIFICATION: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
