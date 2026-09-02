#!/usr/bin/env python3
"""Fail-closed exact source verifier for replay-safe BMP/OBDD/QTT6 v2."""

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


SCHEMA = "strata-bmp-obdd-qtt6-replay-source-manifest-v2"
STATUS = "FROZEN_SOURCE_ONLY__RUNTIME_REPLAY_PENDING__NO_PAYLOAD_AUTHORITY"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED = {
    "README.md", "STATIC_REVIEW.json", "THREAT_MODEL.md", "codec.py", "cupy_backend.py",
    "cupy_worker.py", "design_lock.json", "production_hooks.py",
    "run_cupy_smoke.py", "run_source_free_fixture.py", "search.py",
    "test_source_only.py", "verify_source.py",
}
FORBIDDEN_IMPORTS = {
    "requests", "socket", "urllib", "http", "ftplib", "paramiko",
    "safetensors", "transformers", "huggingface_hub", "torch",
}
PREDECESSOR_PINS = {
    "v1_source_root_sha256":
        "369e01b30173977a5d8227e71104c8515f1b68ef440198dccd1488050e865203",
    "v1_audited_producer_manifest_sha256":
        "916aaca15620e3bf033e849b74a73604015fab280dfe8953683d6cbe04e0d2e4",
    "v1_independent_audit_source_root_sha256":
        "db0dea7fe3f52e88c8ab59af75eb7ceef71610c07f214fc5b2783f77dd98b56c",
    "v1_independent_audit_manifest_sha256":
        "c0c23ea892ed8066c9af78c15491049400a7f93f7f8c7d39b61b67442d10b0ed",
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
        "production_attestation", "caps", "claim_boundary",
    }, "manifest exact schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "manifest schema/status")
    require(manifest["predecessor_pins"] == PREDECESSOR_PINS,
            "predecessor pins")
    require(manifest["access_attestation"] == {
        "model_checkpoint_or_qwen_payload_opened_statted_hashed_or_enumerated": False,
        "current_strata_or_coarse_payload_opened_statted_hashed_or_enumerated": False,
        "matched_control_artifact_opened_statted_hashed_or_enumerated": False,
        "network_used_by_tests_or_fixtures": False,
        "live_payload_authority": False,
    }, "access attestation")
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
        "logical_capacity_separate_from_runtime_ownership": True,
        "argsort_index_dtype": "numpy.intp",
        "skew_candidate_maxima_derived_from_serialized_geometry": True,
        "python_numpy_allocator_peak_claimed": False,
        "gpu_pool_measurement_separate_from_logical_capacity": True,
        "gpu_receipt_executed": False,
    }, "workspace attestation")
    require(manifest["production_attestation"] == {
        "digest_syntax_alone_authorizes": False,
        "regular_nonlink_objects_authenticated": True,
        "control_read_and_independent_audit_receipts_authenticated": True,
        "production_launch_authorized": False,
    }, "production attestation")
    test = manifest["test_attestation"]
    require(isinstance(test, dict) and set(test) == {
        "source_only_test_count_declared", "manifest_self_replay_declared",
        "source_only_tests_executed", "manifest_self_replay_executed",
        "n4096_fixture_executed", "fresh_cupy_search_executed",
        "independent_source_audit_passed", "qwen_payload_run",
    } and isinstance(test["source_only_test_count_declared"], int) and
            test["source_only_test_count_declared"] >= 25 and
            test["manifest_self_replay_declared"] is True and
            test["independent_source_audit_passed"] is False and
            test["qwen_payload_run"] is False,
            "test attestation")
    require(manifest["caps"] == {
        "weights": 4096, "active_features": 12, "bmp_rank": 4,
        "qtt_rank": 2, "obdd_nodes": 240, "exceptions": 64,
        "family_candidates": 32, "actual_frozen_family_bank_candidates": 16,
        "search_evaluations": 1000000,
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
        observed.append(item)
        names.append(name)
        if name.endswith(".py"):
            source = payload.decode("utf-8")
            roots = import_roots(source, name)
            require(not roots & FORBIDDEN_IMPORTS, f"forbidden import {name}")
            if "subprocess" in roots:
                require(name in {"run_cupy_smoke.py", "test_source_only.py"},
                        "subprocess boundary")
            if "cupy" in roots:
                require(name == "cupy_backend.py", "CuPy import boundary")
            if name not in {"verify_source.py", "test_source_only.py"}:
                require("root@" not in source and ".safetensors" not in source,
                        f"payload locator {name}")
            imports[name] = sorted(roots)
    require(set(names) == REQUIRED and
            names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical UTF-8 complete members")
    require(manifest["source_root_sha256"] == sha256(canonical_json(observed)),
            "source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == REQUIRED | {"SOURCE_MANIFEST.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "exact regular package closure")

    design = strict_json(read_regular(root / "design_lock.json", "design"), "design")
    require(design["status"] == STATUS and
            design["predecessor_pins"] == PREDECESSOR_PINS and
            design["workspace_contract"]
                  ["logical_capacity_separate_from_runtime_ownership"] is True and
            design["production_hooks"]
                  ["digest_syntax_alone_authorizes"] is False and
            design["payload_authority"]["qwen"] is False,
            "design lock")
    static = strict_json(read_regular(root / "STATIC_REVIEW.json", "static"),
                         "static")
    require(static["status"] ==
            "PASS_STATIC_REPAIRS__RUNTIME_REPLAY_PENDING__HOLD_PAYLOAD" and
            static["predecessor_pins"] == PREDECESSOR_PINS and
            len(static["repairs"]) == 4 and
            all(row["static_verified"] is True for row in static["repairs"]) and
            static["access_attestation"]["payload_authority"] is False,
            "static review")
    readme = read_regular(root / "README.md", "README").decode("utf-8")
    exact_cli_lines = (
        "verify_source.py \\",
        "--package research/strata_bmp_qtt6_gate_v2_replay \\",
        "--expected-manifest-sha256",
    )
    for phrase in ("Canonical UTF-8 replay", "numpy.intp",
                   "serialized capacities", "measured CuPy pool",
                   "Object-authenticated production boundary",
                   "not evidence of Qwen gain"):
        require(phrase in readme, f"README boundary {phrase}")
    for phrase in exact_cli_lines:
        require(phrase in readme, f"README replay CLI {phrase}")
    production = read_regular(root / "production_hooks.py", "production").decode(
        "utf-8")
    require("ArtifactBinding" in production and
            "digest_syntax_only_authority" in production and
            "lstat()" in production, "object-authenticated launch source")
    search = read_regular(root / "search.py", "search").decode("utf-8")
    require("stable_order_intp_capacity" in search and
            "candidate_serialized_capacity" in search and
            "capacity_separated_from_runtime_ownership" in search,
            "workspace repairs")
    return {
        "schema": "strata-bmp-obdd-qtt6-replay-source-verification-v2",
        "status": "PASS_EXACT_SOURCE_INVENTORY__RUNTIME_PENDING__HOLD_PAYLOAD",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": observed,
        "python_import_roots": imports,
        "canonical_utf8_member_order": True,
        "payload_accessed": False,
        "network_accessed": False,
        "production_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.expected_manifest_sha256),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
