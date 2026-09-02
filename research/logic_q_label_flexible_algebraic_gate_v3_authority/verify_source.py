#!/usr/bin/env python3
"""Standard-library exact-closure verifier for LOGIC-Q v3 authority."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


SCHEMA = "logic-q-label-flexible-algebraic-gate-v3-authority-source-manifest"
STATUS = "SEALED_SOURCE_ONLY_AWAITING_INDEPENDENT_V3_AUDIT"
V2_MANIFEST_SHA256 = (
    "e97041b2debdd1a85ce32305f43aae1f76cf4ca937b52e275bdd246ae1b1b980")
V2_SOURCE_ROOT_SHA256 = (
    "080de7a63e596ae34f9da90941d7fd9d07b70dfb2afad97103aa5ab5943d3776")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MEMBERS = {
    "README.md", "SOURCE_ONLY_TEST_RESULT.json", "authority.py",
    "design_lock.json", "gpu_worker.py", "test_source_only.py",
    "verify_source.py",
}
FORBIDDEN_STATIC_IMPORTS = {
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


def regular_bytes(path: Path, label: str) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            f"{label} regular non-link")
    payload = path.read_bytes()
    after = path.lstat()
    require((before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino),
            f"{label} changed during read")
    return payload


def imported_roots(source: str, filename: str) -> set[str]:
    tree = ast.parse(source, filename=filename)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def verify_dependency(package: Path) -> dict[str, Any]:
    root = package.resolve(strict=True)
    manifest_payload = regular_bytes(root / "SOURCE_MANIFEST.json", "v2 manifest")
    require(sha256(manifest_payload) == V2_MANIFEST_SHA256,
            "v2 external manifest pin")
    manifest = strict_json(manifest_payload, "v2 manifest")
    require(manifest.get("source_root_sha256") == V2_SOURCE_ROOT_SHA256,
            "v2 source-root pin")
    observed = []
    names = []
    for row in manifest.get("members", []):
        require(isinstance(row, dict) and
                set(row) == {"name", "bytes", "sha256"}, "v2 member row")
        name = row["name"]
        require(isinstance(name, str) and name not in names and "/" not in name
                and "\\" not in name, "v2 member name")
        payload = regular_bytes(root / name, f"v2 member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"v2 member pin {name}")
        observed.append(item)
        names.append(name)
    require(sha256(canonical_json(observed)) == V2_SOURCE_ROOT_SHA256,
            "v2 observed source root")
    require({entry.name for entry in os.scandir(root)} ==
            set(names) | {"SOURCE_MANIFEST.json"}, "v2 exact closure")
    return {"manifest_sha256": V2_MANIFEST_SHA256,
            "source_root_sha256": V2_SOURCE_ROOT_SHA256}


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    root = package.resolve(strict=True)
    manifest_payload = regular_bytes(root / "SOURCE_MANIFEST.json", "manifest")
    manifest_sha = sha256(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(HEX64.fullmatch(expected_manifest_sha256) is not None and
                manifest_sha == expected_manifest_sha256,
                "external source manifest SHA-256")
    manifest = strict_json(manifest_payload, "manifest")
    require(set(manifest) == {"schema", "status", "source_root_sha256", "members",
                              "dependencies", "access_attestation",
                              "test_attestation", "semantic_attestation",
                              "claim_boundary"}, "manifest exact schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "manifest schema/status")
    require(manifest["dependencies"]["v2_manifest_sha256"] ==
            V2_MANIFEST_SHA256 and
            manifest["dependencies"]["v2_source_root_sha256"] ==
            V2_SOURCE_ROOT_SHA256 and
            manifest["dependencies"]["v2_v1_v0_modified"] is False,
            "manifest dependency pins")
    require(manifest["access_attestation"] == {
        "model_qwen_strata_coarse_or_control_payload_opened_statted_hashed_or_enumerated": False,
        "network_used_by_source_build_or_tests": False,
        "cupy_imported_or_initialized_by_source_build_or_tests": False,
        "live_payload_authority": False,
    }, "access attestation")
    require(manifest["test_attestation"] == {
        "source_only_hostile_tests_authored": True,
        "source_only_hostile_tests_executed": True,
        "source_only_hostile_test_count": 15,
        "independent_v3_source_audit_passed": False,
        "independent_real_fresh_cupy_worker_passed": False,
    }, "honest test attestation")
    require(manifest["semantic_attestation"] == {
        "abstract_four_level_mechanism": True,
        "current_strata_six_level_64_index_semantics_bound": False,
        "four_level_result_transfers_to_strata": False,
        "strata_rm6_implemented": False,
    }, "semantic attestation")
    observed = []
    names = []
    imports = {}
    sources = {}
    for row in manifest["members"]:
        require(isinstance(row, dict) and
                set(row) == {"name", "bytes", "sha256"}, "member row")
        name = row["name"]
        require(name in MEMBERS and name not in names, "member name")
        payload = regular_bytes(root / name, f"member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"member pin {name}")
        observed.append(item)
        names.append(name)
        if name.endswith(".py"):
            source = payload.decode("utf-8")
            roots = imported_roots(source, name)
            require(not roots & FORBIDDEN_STATIC_IMPORTS,
                    f"{name} forbidden static import")
            if name != "verify_source.py":
                require("root@" not in source and ".safetensors" not in source and
                        "model.safetensors.index" not in source,
                        f"{name} payload locator")
            imports[name] = sorted(roots)
            sources[name] = source
    require(set(names) == MEMBERS and
            names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical complete members")
    require(manifest["source_root_sha256"] == sha256(canonical_json(observed)),
            "source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == MEMBERS | {"SOURCE_MANIFEST.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "exact regular package closure")

    authority = sources["authority.py"]
    worker = sources["gpu_worker.py"]
    require("rows_by_config" not in authority and
            "packet_receipts_by_sha256" not in authority and
            "metrics_accepted_from_encoder\": False" in authority and
            "production authorization requires fresh worker replay" in authority and
            "replay_packet == parsed[\"inner_packet\"]" in authority and
            "caller_selected_config_accepted\": False" in authority,
            "authority non-injection and replay bindings")
    require('require("cupy" not in sys.modules' in worker and
            'cp = importlib.import_module("cupy")' in worker and
            worker.index('require("cupy" not in sys.modules') <
            worker.index('cp = importlib.import_module("cupy")') <
            worker.index("authority = load_authority"),
            "fresh CuPy import order")
    design = strict_json(regular_bytes(root / "design_lock.json", "design"),
                         "design")
    require(design["status"] == STATUS and
            design["authority_inputs"]["encoder_scored_rows"] is False and
            design["packet"]["complete_inner_packet_embedded"] is True and
            design["fresh_backend"]["fresh_exact_replay_before_authorization"]
            is True and
            design["selection"][
                "launched_config_derived_only_from_authorized_artifact"] is True and
            design["payload_authority"]["qwen"] is False,
            "design lock")
    test_result = strict_json(
        regular_bytes(root / "SOURCE_ONLY_TEST_RESULT.json", "test result"),
        "test result")
    require(test_result["status"] ==
            "PASS_SOURCE_ONLY_AUTHORITY_MECHANICS__NO_PAYLOAD_AUTHORITY" and
            test_result["test_count"] == 15 and
            test_result["cupy_imported_or_initialized"] is False and
            test_result["model_qwen_strata_control_payload_accessed"] is False,
            "source-only hostile test result")
    dependency = verify_dependency(
        root.parent / "logic_q_label_flexible_algebraic_gate_v2_bound_adapter")
    require("cupy" not in __import__("sys").modules,
            "verifier initialized CuPy")
    return {
        "schema": "logic-q-v3-authority-source-verification-v1",
        "status": "PASS_EXACT_SOURCE_ONLY_CLOSURE__AWAITING_EXECUTION",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": observed, "python_import_roots": imports,
        "v2_dependency": dependency,
        "payload_accessed": False, "network_accessed": False,
        "cupy_imported": False, "live_authority": False,
        "strata_semantics_bound": False,
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
