#!/usr/bin/env python3
"""Fail-closed exact source verifier for hardened BMP/OBDD/QTT6 v1."""

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


SCHEMA = "strata-bmp-obdd-qtt6-hardened-source-manifest-v1"
STATUS = "FROZEN_SOURCE_ONLY__EXECUTION_PENDING__NO_PAYLOAD_AUTHORITY"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED = {
    "README.md", "THREAT_MODEL.md", "codec.py", "cupy_backend.py",
    "cupy_worker.py", "design_lock.json", "production_hooks.py",
    "run_cupy_smoke.py", "run_source_free_fixture.py", "search.py",
    "test_source_only.py", "verify_source.py",
}
FORBIDDEN_IMPORTS = {
    "requests", "socket", "urllib", "http", "ftplib", "paramiko",
    "safetensors", "transformers", "huggingface_hub", "torch",
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
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
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
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


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
        "predecessor_pins", "access_attestation", "test_attestation",
        "semantic_attestation", "rate_attestation", "workspace_attestation",
        "caps", "claim_boundary",
    }, "manifest exact schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "manifest schema/status")
    require(manifest["predecessor_pins"] == {
        "v0_source_manifest_sha256": "a7778080a00d5d2967636ac8d60dd31698401c4dcf8da160c9451c92dc5f6b18",
        "v0_source_root_sha256": "6b7baf9706349d10108121d4dcb03661b2378dc436303bbfe1bbccd38a0c8914",
        "v0_independent_audit_manifest_sha256": "6038fea16ba29fad6c8b351bc0968fd00f94f007f1e113a4209775974cc33df1",
        "v0_independent_audit_source_root_sha256": "e905324af56f544b27423390e22c97de5c4b15696c621ab133c2da5533e9f4a9",
    }, "predecessor pins")
    require(manifest["access_attestation"] == {
        "model_checkpoint_or_qwen_payload_opened_statted_hashed_or_enumerated": False,
        "current_strata_or_coarse_payload_opened_statted_hashed_or_enumerated": False,
        "matched_control_artifact_opened_statted_hashed_or_enumerated": False,
        "network_used_by_tests_or_fixtures": False,
        "live_payload_authority": False,
    }, "access attestation")
    require(manifest["test_attestation"] == {
        "source_only_hostile_tests_executed": False,
        "source_only_test_count_declared": 22,
        "n4096_fixture_executed": False,
        "fresh_cupy_search_executed": False,
        "static_source_review_completed": True,
        "independent_source_audit_passed": False,
        "qwen_payload_run": False,
    }, "test attestation")
    require(manifest["semantic_attestation"] == {
        "exact_distortion_columns": 64,
        "completed_planes": 6,
        "decoded_index_range": [0, 63],
        "four_level_adapter": False,
        "variable_arbitrary_uint16_rows_and_columns": True,
        "roles": 3,
        "bmp_semantically_canonical": True,
        "qtt_semantically_canonical": True,
        "packet_is_complete_production_strata_packet": False,
    }, "semantic attestation")
    require(manifest["rate_attestation"] == {
        "explicit_cap_argument_required": True,
        "integer_minimum": "ceil(43*N/20)",
        "integer_maximum": "floor(5*N/2)",
        "source_fixture_is_complete_codec": False,
    }, "rate attestation")
    require(manifest["workspace_attestation"] == {
        "cpu_named_logical_buffers_exact": True,
        "cpu_allocator_peak_claimed": False,
        "gpu_fresh_dedicated_pool_receipt_source_present": True,
        "gpu_receipt_executed": False,
    }, "workspace attestation")
    require(manifest["caps"] == {
        "weights": 4096, "active_features": 12, "bmp_rank": 4,
        "qtt_rank": 2, "obdd_nodes": 240, "exceptions": 64,
        "family_candidates": 32, "search_evaluations": 1000000,
        "cpu_workspace_bytes": 67108864,
        "gpu_workspace_bytes": 134217728,
    }, "manifest caps")

    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "manifest members")
    observed, names, imports = [], [], {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "member schema")
        name = row["name"]
        require(isinstance(name, str) and name in REQUIRED and name not in names and
                "/" not in name and "\\" not in name, "member name")
        payload = read_regular(root / name, f"member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(row == item, f"member pin {name}")
        observed.append(item); names.append(name)
        if name.endswith(".py"):
            source = payload.decode("utf-8")
            roots = import_roots(source, name)
            require(not roots & FORBIDDEN_IMPORTS, f"forbidden import {name}")
            if "subprocess" in roots:
                require(name == "run_cupy_smoke.py", "subprocess boundary")
            if "cupy" in roots:
                require(name == "cupy_backend.py", "CuPy import boundary")
            if name not in {"verify_source.py", "test_source_only.py"}:
                require("root@" not in source and ".safetensors" not in source,
                        f"payload locator {name}")
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

    design = strict_json(read_regular(root / "design_lock.json", "design"), "design")
    require(design["status"] == STATUS and
            design["semantic_contract"]["completed_planes"] == 6 and
            design["semantic_contract"]["gf2_factor_canonical_minimum_rank"] is True and
            design["semantic_contract"]["qtt_canonical_minimum_cut_ranks"] is True and
            design["payload_authority"]["qwen"] is False and
            design["execution_attestation"]["runtime_pass_claimed"] is False,
            "design lock")
    readme = read_regular(root / "README.md", "README").decode("utf-8")
    threat = read_regular(root / "THREAT_MODEL.md", "threat").decode("utf-8")
    for phrase in ("Semantic canonicality", "CompleteRateCap",
                   "arbitrary positive SwiGLU dimensions", "actual device-backed",
                   "Production holds", "not evidence of Qwen gain"):
        require(phrase in readme, f"README boundary {phrase}")
    for phrase in ("Rank inflation", "Header overflow", "Fake CuPy facade",
                   "source verifier and real CuPy worker are unexecuted"):
        require(phrase in threat, f"threat boundary {phrase}")
    return {
        "schema": "strata-bmp-obdd-qtt6-hardened-source-verification-v1",
        "status": "PASS_EXACT_SOURCE_INVENTORY__EXECUTION_PENDING__HOLD_PAYLOAD",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": observed,
        "python_import_roots": imports,
        "payload_accessed": False,
        "network_accessed": False,
        "runtime_pass_claimed": False,
        "independent_source_audit_passed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.expected_manifest_sha256),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
