#!/usr/bin/env python3
"""Independent stdlib-only exact-closure verifier for the v3 source freeze."""

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


SCHEMA = "strata-bmp-obdd-qtt6-v3-authority-source-manifest-v1"
STATUS = "FROZEN_SOURCE_ONLY_AUTHORITY__COMPILED_LAUNCH_PIN_ABSENT__HOLD_RUNTIME_PAYLOAD_AND_RESULTS"
V2_MANIFEST_SHA256 = "84df0d32a55682f6565ac9d144f7de850acf77cde27bffdefa77a151211906f8"
V2_SOURCE_ROOT_SHA256 = "b518b203c43fd401c94e1bfcf67e029a85a95f1f7ce244fcd864a96d0780da47"
V2_AUDIT_MANIFEST_SHA256 = "324e9a6d7d16be7b57b4ae33599cce2e4b324848e279b59268826b5dcaaebd12"
V2_AUDIT_SOURCE_ROOT_SHA256 = "c817b1f1c3c270cb1f0e332262dc46df4fe9eb39c4b4fafe70a23536203572d3"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MEMBERS = {
    "README.md", "REPAIR_MAP.json", "SOURCE_ONLY_TEST_RESULT.json",
    "THREAT_MODEL.md", "authority.py", "design_lock.json",
    "test_source_only.py", "verify_source.py",
}
FORBIDDEN_IMPORTS = {
    "cupy", "torch", "safetensors", "transformers", "huggingface_hub",
    "requests", "socket", "urllib", "http", "ftplib", "paramiko",
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


def strict_json(payload: bytes, label: str, *, canonical: bool = False) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerifyError(f"{label}: nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifyError(f"{label}: strict JSON") from exc
    require(isinstance(value, dict), f"{label}: object")
    if canonical:
        require(payload == canonical_json(value) + b"\n", f"{label}: canonical JSON")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                f"{label}: regular non-link")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise VerifyError(f"{label}: read") from exc
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_mode) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
             after.st_mode), f"{label}: changed during read")
    return payload


def import_roots(source: str, filename: str) -> set[str]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise VerifyError(f"{filename}: syntax") from exc
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def verify_dependency(package: Path, *, manifest_name: str, manifest_sha: str,
                      source_root: str, root_field: str, schema: str) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir() and not package.is_symlink(), "dependency real directory")
    manifest_payload = regular_bytes(root / manifest_name, "dependency manifest")
    require(sha256(manifest_payload) == manifest_sha, "dependency manifest pin")
    manifest = strict_json(manifest_payload, "dependency manifest")
    require(manifest.get("schema") == schema and manifest.get(root_field) == source_root,
            "dependency schema/root pin")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "dependency members")
    names = []
    observed = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "dependency member row")
        name = row["name"]
        require(isinstance(name, str) and name not in names and "/" not in name and
                "\\" not in name, "dependency member name")
        payload = regular_bytes(root / name, f"dependency {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"dependency member pin {name}")
        names.append(name)
        observed.append(item)
    require(names == sorted(names, key=lambda value: value.encode("utf-8")) and
            sha256(canonical_json(observed)) == source_root,
            "dependency canonical source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {manifest_name} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "dependency exact regular closure")
    return {"manifest_sha256": manifest_sha, "source_root_sha256": source_root,
            "members": len(names)}


def verify(package: Path, expected_manifest_sha256: str,
           v2_package: Path, v2_audit_package: Path) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir() and not package.is_symlink(), "v3 real package")
    manifest_payload = regular_bytes(root / "SOURCE_MANIFEST.json", "v3 manifest")
    require(HEX64.fullmatch(expected_manifest_sha256) is not None and
            sha256(manifest_payload) == expected_manifest_sha256,
            "v3 external manifest SHA-256")
    manifest = strict_json(manifest_payload, "v3 manifest", canonical=True)
    required_manifest = {"schema", "status", "source_root_sha256", "members",
                         "lineage", "access_attestation", "test_attestation",
                         "authority_attestation", "claim_boundary"}
    require(set(manifest) == required_manifest and manifest["schema"] == SCHEMA and
            manifest["status"] == STATUS, "v3 manifest schema/status")
    expected_lineage = {
        "v2_manifest_sha256": V2_MANIFEST_SHA256,
        "v2_source_root_sha256": V2_SOURCE_ROOT_SHA256,
        "v2_audit_manifest_sha256": V2_AUDIT_MANIFEST_SHA256,
        "v2_audit_source_root_sha256": V2_AUDIT_SOURCE_ROOT_SHA256,
        "v2_or_audit_modified": False,
    }
    require(manifest["lineage"] == expected_lineage, "v3 frozen lineage")
    require(manifest["access_attestation"] == {
        "model_qwen_strata_or_control_payload_opened_statted_hashed_or_enumerated": False,
        "network_used": False,
        "cupy_imported_or_initialized": False,
        "runtime_authority": False,
    }, "honest access attestation")
    require(manifest["test_attestation"] == {
        "source_only_tests_executed": True,
        "source_only_test_count": 14,
        "failures": 0,
        "errors": 0,
        "fixtures_production_authorized": False,
        "independent_v3_audit_executed": False,
    }, "test attestation")
    require(manifest["authority_attestation"] == {
        "compiled_launch_manifest_sha256": None,
        "production_authorized": False,
        "executed_capability_kinds_required": 6,
        "literal_current_strata_adapter_required": True,
        "independent_bf16_fp64_scorer_required": True,
        "per_routed_expert_instrumented_page_trace_required": True,
        "self_authored_dummy_or_fixture_production_receipts_accepted": False,
        "model_control_path_inode_or_byte_alias_accepted": False,
    }, "authority attestation")
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(MEMBERS), "eight members")
    names = []
    observed = []
    imports = {}
    sources = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "v3 member row")
        name = row["name"]
        require(name in MEMBERS and name not in names, "v3 member name")
        payload = regular_bytes(root / name, f"v3 member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"v3 member pin {name}")
        names.append(name)
        observed.append(item)
        if name.endswith(".py"):
            source = payload.decode("utf-8")
            roots = import_roots(source, name)
            require(not roots & FORBIDDEN_IMPORTS, f"{name}: forbidden import")
            if name != "verify_source.py":
                require("root@" not in source and ".safetensors" not in source,
                        f"{name}: payload locator")
            imports[name] = sorted(roots)
            sources[name] = source
    require(set(names) == MEMBERS and
            names == sorted(names, key=lambda value: value.encode("utf-8")),
            "v3 canonical member order")
    require(manifest["source_root_sha256"] == sha256(canonical_json(observed)),
            "v3 canonical source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == MEMBERS | {"SOURCE_MANIFEST.json"} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "v3 exact regular closure")

    authority = sources["authority.py"]
    require("TRUSTED_LAUNCH_MANIFEST_SHA256: str | None = None" in authority and
            "production capability cannot be fixture, dummy, or self-authored" in authority and
            "source-byte alias across model/control routes" in authority and
            "instrumented read trace, not layout assertion" in authority and
            "CURRENT_STRATA_SIX_PLANE_INDEX64_V1" in authority and
            "BF16_LE" in authority and "FP64" in authority,
            "v3 authority repair predicates")
    readme = regular_bytes(root / "README.md", "README").decode("utf-8")
    for phrase in ("thirteen non-manifest members", "eight non-manifest members",
                   "Literal current-STRATA contract", "Independent original-BF16 score",
                   "A read trace is not a layout ratio", "Model/control non-aliasing"):
        require(phrase in readme, f"README repair boundary: {phrase}")
    design = strict_json(regular_bytes(root / "design_lock.json", "design"), "design")
    require(design["status"] == STATUS and
            design["production_authority"]["compiled_launch_manifest_sha256"] is None and
            design["payload_authority"]["qwen"] is False,
            "design lock hold")
    repairs = strict_json(regular_bytes(root / "REPAIR_MAP.json", "repair map"),
                          "repair map")
    require(len(repairs["repairs"]) == 7 and repairs["runtime_authorized"] is False,
            "complete repair map")
    tests = strict_json(regular_bytes(root / "SOURCE_ONLY_TEST_RESULT.json", "test result"),
                        "test result")
    require(tests["test_count"] == tests["tests_run"] == 14 and
            tests["failures"] == tests["errors"] == 0 and
            tests["production_fixture_authorized"] is False and
            tests["model_qwen_strata_or_control_payload_opened"] is False,
            "source test result")

    producer = verify_dependency(
        v2_package, manifest_name="SOURCE_MANIFEST.json",
        manifest_sha=V2_MANIFEST_SHA256, source_root=V2_SOURCE_ROOT_SHA256,
        root_field="source_root_sha256",
        schema="strata-bmp-obdd-qtt6-replay-source-manifest-v2")
    audit = verify_dependency(
        v2_audit_package, manifest_name="AUDIT_SOURCE_MANIFEST.json",
        manifest_sha=V2_AUDIT_MANIFEST_SHA256,
        source_root=V2_AUDIT_SOURCE_ROOT_SHA256,
        root_field="audit_source_root_sha256",
        schema="strata-bmp-qtt6-v2-independent-audit-source-manifest-v1")
    audit_manifest = strict_json(
        regular_bytes(v2_audit_package.resolve(strict=True) / "AUDIT_SOURCE_MANIFEST.json",
                      "audit manifest binding"), "audit manifest binding")
    require(audit_manifest["producer_pins"] == {
        "source_manifest_sha256": V2_MANIFEST_SHA256,
        "source_root_sha256": V2_SOURCE_ROOT_SHA256,
    }, "actual auditor manifest binds actual producer")
    require("cupy" not in __import__("sys").modules, "source verifier initialized CuPy")
    return {
        "schema": "strata-bmp-qtt6-v3-authority-source-verification-v1",
        "status": "PASS_EXACT_SOURCE_CLOSURE_AND_PINNED_PREDECESSORS__HOLD_RUNTIME_PAYLOAD",
        "source_manifest_sha256": expected_manifest_sha256,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": observed,
        "python_import_roots": imports,
        "v2": producer,
        "v2_independent_audit": audit,
        "production_authorized": False,
        "payload_accessed": False,
        "network_accessed": False,
        "cupy_imported": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--v2-package", type=Path, required=True)
    parser.add_argument("--v2-audit-package", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.package, args.expected_manifest_sha256,
                    args.v2_package, args.v2_audit_package)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
