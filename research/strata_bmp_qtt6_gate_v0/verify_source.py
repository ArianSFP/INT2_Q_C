#!/usr/bin/env python3
"""Fail-closed exact source verifier for STRATA-BMP/OBDD/QTT6 v0."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


SCHEMA = "strata-bmp-obdd-qtt6-source-manifest-v0"
STATUS = "FROZEN_SOURCE_ONLY_UNEXECUTED__HOLD_PAYLOAD_PENDING_INDEPENDENT_AUDIT"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED = {
    "README.md", "THREAT_MODEL.md", "codec.py", "cupy_backend.py", "design_lock.json",
    "run_cupy_smoke.py", "run_source_free_fixture.py", "search.py",
    "test_source_only.py", "verify_source.py",
}
FORBIDDEN_IMPORTS = {
    "requests", "socket", "urllib", "http", "ftplib", "paramiko",
    "subprocess", "safetensors", "transformers", "huggingface_hub", "torch",
}


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


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerifyError(f"{label} nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifyError(f"{label} strict JSON") from exc
    require(isinstance(value, dict), f"{label} object")
    return value


def read_regular(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                f"{label} regular non-link")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise VerifyError(f"{label} read") from exc
    require((before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino),
            f"{label} changed during read")
    return payload


def import_roots(source: str, filename: str) -> set[str]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise VerifyError(f"{filename} syntax") from exc
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".", 1)[0])
    return result


def verify(package: Path, expected_manifest_sha256: str | None) -> dict:
    root = package.resolve(strict=True)
    require(root.is_dir(), "package directory")
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json", "manifest")
    manifest_sha = sha256(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(HEX64.fullmatch(expected_manifest_sha256) is not None and
                expected_manifest_sha256 == manifest_sha,
                "external manifest SHA-256")
    manifest = strict_json(manifest_payload, "manifest")
    require(set(manifest) == {
        "schema", "status", "source_root_sha256", "members",
        "access_attestation", "test_attestation", "semantic_attestation",
        "caps", "claim_boundary",
    }, "manifest exact schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "manifest schema/status")
    require(manifest["access_attestation"] == {
        "model_checkpoint_or_qwen_payload_opened_statted_hashed_or_enumerated": False,
        "current_strata_or_coarse_payload_opened_statted_hashed_or_enumerated": False,
        "matched_control_artifact_opened_statted_hashed_or_enumerated": False,
        "network_used_by_tests_or_fixtures": False,
        "ordinary_tests_imported_or_initialized_cupy": False,
        "live_payload_authority": False,
    }, "access attestation")
    require(manifest["test_attestation"] == {
        "source_only_hostile_tests_executed": False,
        "source_only_test_count_declared": 17,
        "n4096_fixture_executed": False,
        "runpod_cupy_smoke_executed": False,
        "static_source_review_completed": True,
        "independent_source_audit_passed": False,
        "qwen_payload_run": False,
    }, "test attestation")
    require(manifest["semantic_attestation"] == {
        "exact_distortion_columns": 64,
        "completed_planes": 6,
        "decoded_index_range": [0, 63],
        "four_level_adapter": False,
        "mixed_radix_rows": "3*2^k",
        "hidden_width": "2^h",
        "roles": 3,
        "packet_is_complete_production_strata_packet": False,
    }, "semantic attestation")
    require(manifest["caps"] == {
        "weights": 4096, "active_features": 12, "bmp_rank": 4,
        "qtt_rank": 2, "obdd_nodes": 240, "exceptions": 64,
        "family_candidates": 32, "search_evaluations": 1000000,
        "cpu_workspace_bytes": 67108864,
        "gpu_workspace_bytes": 134217728,
    }, "manifest caps")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "manifest members")
    observed = []
    names = []
    imports = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "member schema")
        name = row["name"]
        require(isinstance(name, str) and name in REQUIRED and name not in names and
                "/" not in name and "\\" not in name, "member name")
        payload = read_regular(root / name, f"member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(row == item, f"member pin {name}")
        observed.append(item)
        names.append(name)
        if name.endswith(".py"):
            source = payload.decode("utf-8")
            roots = import_roots(source, name)
            require(not roots & FORBIDDEN_IMPORTS, f"forbidden import {name}")
            if name not in {"verify_source.py", "test_source_only.py"}:
                require("root@" not in source and ".safetensors" not in source,
                        f"payload locator {name}")
            if "cupy" in roots:
                require(name == "cupy_backend.py", "CuPy import boundary")
            imports[name] = sorted(roots)
    require(set(names) == REQUIRED and
            names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical complete members")
    require(manifest["source_root_sha256"] == sha256(canonical_json(observed)),
            "source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == REQUIRED | {"SOURCE_MANIFEST.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "exact regular package closure")
    design = strict_json(read_regular(root / "design_lock.json", "design"),
                         "design")
    require(design["status"] == STATUS and
            design["semantic_contract"]["completed_planes"] == 6 and
            design["semantic_contract"]["distortion_columns"] == 64 and
            design["semantic_contract"]["four_level_abi_accepted"] is False and
            design["family_bank"][
                "families_are_label_functions_not_value_tensor_decompositions"]
            is True and
            design["family_bank"]["families_are_not_probability_mps_models"]
            is True and design["payload_authority"]["qwen"] is False and
            design["execution_attestation"]["runtime_pass_claimed"] is False,
            "design lock")
    readme = read_regular(root / "README.md", "README").decode("utf-8")
    threat = read_regular(root / "THREAT_MODEL.md", "threat model").decode("utf-8")
    require("D[i,k]" in readme and "x_0[i] + 2*x_1[i]" in readme and
            "https://arxiv.org/abs/2505.01930" in readme and
            "https://arxiv.org/abs/2606.04506" in readme and
            "not a value tensor train" in readme and
            "not an MPS probability law" in readme and
            "HOLD all model/Qwen" in readme and
            "unexecuted freeze" in threat and "0.078125" in threat and
            "2.83203125" in threat,
            "README semantic boundary")
    return {
        "schema": "strata-bmp-obdd-qtt6-source-verification-v0",
        "status": "PASS_EXACT_SOURCE_ONLY_VERIFICATION__HOLD_PAYLOAD",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": observed,
        "python_import_roots": imports,
        "payload_accessed": False,
        "network_accessed": False,
        "independent_source_audit_passed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-manifest-sha256")
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.package,
                            arguments.expected_manifest_sha256),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
