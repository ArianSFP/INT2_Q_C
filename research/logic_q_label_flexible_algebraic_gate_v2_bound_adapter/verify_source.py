#!/usr/bin/env python3
"""Standard-library exact-closure verifier for the LOGIC-Q v2 bound adapter."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


SCHEMA = "logic-q-label-flexible-algebraic-gate-v2-bound-adapter-source-manifest"
STATUS = "SEALED_SOURCE_ONLY_AWAITING_INDEPENDENT_V2_AUDIT"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_NAMES = {
    "PREFLIGHT_HISTORY.json",
    "README.md",
    "STRATA_RM6_ADAPTER_PLAN.md",
    "bound_adapter.py",
    "design_lock.json",
    "independent_scorer.py",
    "run_source_free_fixture.py",
    "test_source_only.py",
    "verify_source.py",
}
FORBIDDEN_IMPORT_ROOTS = {
    "cupy", "torch", "safetensors", "transformers", "huggingface_hub",
    "requests", "socket", "urllib", "http", "ftplib", "paramiko",
    "subprocess",
}
V1_MANIFEST_SHA256 = "9bfd3d1225fb45a0518d2d4d6a4035262e87dc62563222e42e69665358b9aac5"
V1_SOURCE_ROOT_SHA256 = "5d145d89a20d2ae256ea60f569fab97cd6372cde66f7df75f3e86b08b3a88560"
V0_MANIFEST_SHA256 = "31edbc3325dfdae2b3f43cce4afb360062d5c70583b57dd1e6530835a178cced"
V0_SOURCE_ROOT_SHA256 = "2177f2aec39a65afddbbded9b6b3cd2c2a33118c060a41e070102f9fb6c95d4a"
V1_AUDIT_MANIFEST_SHA256 = "6a0e97d987a3288126632db29756681c2ee7c16e809d2a8466db16b22a78dfe1"
V1_AUDIT_ROOT_SHA256 = "d56f36015413694c45dd81b571b05974ef5541cf66e6099c4d2b518c75f1c63b"


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


def imported_roots(source: str, filename: str) -> set[str]:
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


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir(), "package directory")
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json", "manifest")
    manifest_sha = sha256(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(HEX64.fullmatch(expected_manifest_sha256) is not None and
                manifest_sha == expected_manifest_sha256,
                "external source manifest SHA-256")
    manifest = strict_json(manifest_payload, "manifest")
    require(set(manifest) == {
        "schema", "status", "source_root_sha256", "members", "dependencies",
        "access_attestation", "test_attestation", "semantic_attestation",
        "claim_boundary",
    }, "manifest exact schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "manifest schema/status")
    require(manifest["dependencies"] == {
        "v1_manifest_sha256": V1_MANIFEST_SHA256,
        "v1_source_root_sha256": V1_SOURCE_ROOT_SHA256,
        "v0_manifest_sha256": V0_MANIFEST_SHA256,
        "v0_source_root_sha256": V0_SOURCE_ROOT_SHA256,
        "v1_audit_manifest_sha256": V1_AUDIT_MANIFEST_SHA256,
        "v1_audit_source_root_sha256": V1_AUDIT_ROOT_SHA256,
        "v1_and_v0_modified": False,
    }, "pinned source dependencies")
    require(manifest["access_attestation"] == {
        "model_checkpoint_or_qwen_payload_opened_statted_hashed_or_enumerated": False,
        "current_strata_or_coarse_payload_opened_statted_hashed_or_enumerated": False,
        "matched_control_artifact_opened_statted_hashed_or_enumerated": False,
        "network_used_by_source_fixture_or_tests": False,
        "cupy_imported_or_initialized_by_source_fixture_or_tests": False,
        "live_qwen_authority": False,
    }, "source access attestation")
    require(manifest["test_attestation"] == {
        "local_source_only_tests_passed": True,
        "local_source_only_test_count": 27,
        "local_source_free_fixture_passed": True,
        "independent_v2_source_audit_passed": False,
        "independent_real_cupy_launch_receipt_passed": False,
    }, "test attestation")
    require(manifest["semantic_attestation"] == {
        "abstract_four_level_v1_mechanism": True,
        "current_strata_six_level_64_index_semantics_bound": False,
        "four_level_result_transfers_to_strata": False,
        "strata_rm6_is_design_plan_only": True,
    }, "STRATA semantic boundary")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "member rows")
    observed = []
    names = []
    imports = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "member row schema")
        name = row["name"]
        require(isinstance(name, str) and name in REQUIRED_NAMES and
                name not in names and "/" not in name and "\\" not in name,
                "safe unique member")
        payload = read_regular(root / name, f"member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"member pin {name}")
        observed.append(item)
        names.append(name)
        if name.endswith(".py"):
            source = payload.decode("utf-8")
            roots = imported_roots(source, name)
            require(not roots & FORBIDDEN_IMPORT_ROOTS,
                    f"{name} forbidden import")
            if name != "verify_source.py":
                require(".safetensors" not in source and "root@" not in source and
                        "model.safetensors.index" not in source,
                        f"{name} payload or remote locator")
            imports[name] = sorted(roots)
    require(set(names) == REQUIRED_NAMES and
            names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical complete members")
    require(manifest["source_root_sha256"] == sha256(canonical_json(observed)),
            "source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == REQUIRED_NAMES | {"SOURCE_MANIFEST.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "exact regular package closure")

    design = strict_json(read_regular(root / "design_lock.json", "design"), "design")
    require(design["status"] == STATUS and
            design["dependencies"]["v1_source_root_sha256"] ==
            V1_SOURCE_ROOT_SHA256 and
            design["selector"]["selected_config_recomputed_from_validation_metrics"]
            is True and
            design["independent_scorer"]["encoder_weighted_sse_accepted"] is False and
            design["live_backend"]["name_only_object_accepted"] is False and
            design["strata_semantic_boundary"][
                "v2_bound_to_current_strata_semantics"] is False and
            design["payload_authority"]["qwen"] is False,
            "design lock bindings")
    history = strict_json(read_regular(root / "PREFLIGHT_HISTORY.json", "history"),
                          "history")
    require(history["status"] ==
            "EXACT_VERIFY_FAILURE_PRESERVED__FINAL_CLEAN_RETRY_PASSED" and
            history["failure_count"] == 1 and
            [attempt["ordinal"] for attempt in history["attempts"]] ==
            list(range(len(history["attempts"]))) and
            history["attempts"][2]["result"] == "FAIL_CLOSED" and
            history["attempts"][2]["error"] ==
            "VerifyError: design lock bindings" and
            history["attempts"][2]["failure_ignored"] is False and
            history["attempts"][-1]["result"] == "PASS",
            "preflight history")
    readme = read_regular(root / "README.md", "README").decode("utf-8")
    plan = read_regular(root / "STRATA_RM6_ADAPTER_PLAN.md", "STRATA plan").decode(
        "utf-8")
    require("not bound to the current STRATA" in readme and
            "No result from that four-level" in plan and
            "D[i,k]" in plan and "2.3232421875" in plan,
            "literal STRATA non-transfer plan")
    require("cupy" not in sys.modules, "verifier initialized CuPy")
    return {
        "schema": "logic-q-v2-bound-adapter-source-verification-v1",
        "status": "PASS_EXACT_SOURCE_ONLY_VERIFICATION",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": observed, "python_import_roots": imports,
        "dependencies": manifest["dependencies"],
        "payload_accessed": False, "network_accessed": False,
        "cupy_imported": False,
        "strata_six_level_semantics_bound": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--package", type=Path,
                       default=Path(__file__).resolve().parent)
    value.add_argument("--expected-manifest-sha256")
    return value


def main() -> None:
    arguments = parser().parse_args()
    print(json.dumps(verify(arguments.package,
                            arguments.expected_manifest_sha256),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
