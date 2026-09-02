#!/usr/bin/env python3
"""Dependency-free verifier for the external dispatcher source package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


EXPECTED_MEMBERS = {
    "README.md",
    "bootstrap.py",
    "decoder_bundle.json",
    "design_lock.json",
    "runtime_lock.json",
    "strata_ordinal_bridge.py",
    "test_source_only.py",
    "verify_output.py",
    "verify_source.py",
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise VerificationError(f"{label} nonfinite: {value}")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), f"{label} root")
    return value


def regular_bytes(path: Path, cap: int = 256 << 20) -> bytes:
    info = os.lstat(path)
    require(stat.S_ISREG(info.st_mode), f"not regular: {path.name}")
    require(0 <= info.st_size <= cap, f"oversized: {path.name}")
    with path.open("rb") as handle:
        data = handle.read(cap + 1)
    require(len(data) == info.st_size and len(data) <= cap, f"unstable/oversized: {path.name}")
    after = os.lstat(path)
    require(
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        f"changed while reading: {path.name}",
    )
    return data


def verify(package: Path) -> dict[str, Any]:
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode, "verify_source requires CPython -I -B")
    package = package.absolute()
    require(package.is_dir(), "package directory")
    actual = {entry.name for entry in os.scandir(package)}
    require(actual == EXPECTED_MEMBERS | {"SOURCE_MANIFEST.json"}, f"exact package member set: {sorted(actual ^ (EXPECTED_MEMBERS | {'SOURCE_MANIFEST.json'}))}")
    manifest_bytes = regular_bytes(package / "SOURCE_MANIFEST.json", 4 << 20)
    manifest = strict_json(manifest_bytes, "source manifest")
    require(set(manifest) == {"schema", "status", "members", "access_attestation", "remaining_authority_gates"}, "manifest fields")
    require(manifest["schema"] == "uwfa-sc-v8-external-dispatcher-source-manifest-v2", "manifest schema")
    require(
        manifest["status"] in {
            "SOURCE_ONLY_REVIEW_CANDIDATE_NO_PAYLOAD_AUTHORITY",
            "SEALED_EXTERNAL_DISPATCHER_SOURCE_NO_PAYLOAD_AUTHORITY",
        },
        "manifest status",
    )
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_MEMBERS), "manifest row count")
    require([row.get("name") for row in rows] == sorted(EXPECTED_MEMBERS, key=lambda name: name.encode("utf-8")), "manifest UTF-8 order")
    snapshots: dict[str, bytes] = {}
    observed = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "manifest row fields")
        name = row["name"]
        require(name in EXPECTED_MEMBERS, "manifest member name")
        data = regular_bytes(package / name)
        require(type(row["bytes"]) is int and row["bytes"] == len(data), f"manifest member bytes: {name}")
        require(isinstance(row["sha256"], str) and row["sha256"] == sha256(data), f"manifest member digest: {name}")
        snapshots[name] = data
        observed.append(dict(row))
        if name.endswith(".py"):
            compile(data, f"<dispatcher-source-verifier:{name}>", "exec", dont_inherit=True, optimize=0)
    attestation = manifest["access_attestation"]
    require(
        attestation
        == {
            "qwen_or_model_payload_opened_statted_hashed_or_enumerated": False,
            "current_finite_artifact_opened_statted_hashed_or_enumerated": False,
            "gaussian_control_opened_statted_hashed_or_enumerated": False,
            "runpod_payload_execution_by_builder": False,
            "source_free_tests_are_payload_claim_evidence": False,
        },
        "source-only access attestation",
    )
    remaining = manifest["remaining_authority_gates"]
    require(
        remaining
        == [
            "EXTERNAL_PINNED_DISPATCHER_AUDIT",
            "EXTERNAL_EXACT_REQUEST_AND_REVIEW_INPUT_PINS",
            "SEALED_EXHAUSTIVE_RUNTIME_TREE",
            "PROCESS_START_IMPORT_NATIVE_EVENT_LEDGER_CLOSURE",
            "SEALED_AUTHENTICATED_DECODER_CLOSURE",
            "PUBLIC_COMMIT_FRESH_RTX5090_TYPED_PREFLIGHT",
            "FRESH_PROCESS_INDEPENDENT_RESULT_AUDIT",
            "NO_PAYLOAD_BEFORE_ALL_EXTERNAL_AUTHORITY",
            "PRIMARY_POSIX_SOURCE_TESTS_BEFORE_FINAL_SEAL",
        ],
        "remaining authority gates",
    )
    design = strict_json(snapshots["design_lock.json"], "design lock")
    require(design.get("schema") == "uwfa-sc-v8-external-dispatcher-design-v2", "design schema")
    require(design.get("status") == manifest["status"], "design/manifest lifecycle status")
    pins = design.get("embedded_authority_pins")
    require(isinstance(pins, dict), "design pins")
    require(pins.get("producer_source_manifest_sha256") == "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6", "producer manifest pin")
    require(pins.get("producer_final_review_sha256") == "57e19b93f9f771381945a42060e9b15b71962f8af4b7800fd71be0c1949c2cce", "producer review pin")
    require(pins.get("producer_public_git_commit") == "d563c4ac1e78a6b6e7f0722291211d1209f775af", "producer commit pin")
    require(set(pins) == {"producer_source_manifest_sha256", "producer_final_review_sha256", "producer_public_git_commit", "runtime_lock_sha256", "decoder_bundle_sha256"}, "noncircular embedded pin set")
    require(str(pins.get("runtime_lock_sha256", "")).startswith("__UNRESOLVED_"), "runtime pin remains fail-closed")
    require(str(pins.get("decoder_bundle_sha256", "")).startswith("__UNRESOLVED_"), "decoder pin remains fail-closed")
    launcher = design.get("external_launcher_authority")
    require(isinstance(launcher, dict) and launcher.get("status") == "UNRESOLVED_OUT_OF_TREE_BY_DESIGN" and launcher.get("typed_abi") == "ExternalLaunchAuthority", "noncircular external launcher authority")
    v2_repairs = design.get("v2_repairs")
    require(
        isinstance(v2_repairs, dict)
        and v2_repairs
        == {
            "authenticated_import_enforcement_active_before_numeric_import_and_preflight": True,
            "ambient_import_hooks_and_preloaded_numeric_modules_fail_closed": True,
            "path_substitution_cannot_change_executed_held_bytes": True,
            "transient_native_load_unload_events_cannot_evade_final_snapshots": True,
            "sys_modules_removal_or_replacement_is_ledgered_and_rejected": True,
            "strata_ordinal_rows_are_list_of_lists_for_one_axis_numpy_advanced_indexing": True,
        },
        "v2 repair declaration",
    )
    runtime = strict_json(snapshots["runtime_lock.json"], "runtime lock")
    require(runtime.get("schema") == "uwfa-sc-v8-external-runtime-lock-v2", "runtime lock schema")
    require(runtime.get("status") == "BLOCKED_UNTIL_EXHAUSTIVE_RUNTIME_TREE_MANIFEST_IS_PINNED", "runtime lock fail-closed status")
    require(runtime.get("python_import_roots", [None])[0] == runtime.get("site_packages"), "runtime import roots/site binding")
    require(
        runtime.get("native_load_enforcement")
        == {
            "mechanism": "LINUX_RTLD_AUDIT_APPEND_ONLY_NONBLOCKING_PIPE_V1",
            "event_schema": "uwfa-native-loader-event-v1",
            "auditor_role": "native_loader_auditor",
            "active_from_process_start": True,
            "records_load_and_unload": True,
            "ready_binds_held_auditor_identity_hash": True,
        },
        "runtime native load enforcement",
    )
    require("native_loader_auditor" in runtime.get("required_runtime_roles", []), "runtime native auditor role")
    decoder = strict_json(snapshots["decoder_bundle.json"], "decoder bundle")
    require(decoder.get("schema") == "uwfa-sc-v8-external-decoder-bundle-v2", "decoder bundle schema")
    require(decoder.get("status") == "BLOCKED_UNTIL_RUNPOD_PATHS_BYTES_AND_BUNDLE_PIN_ARE_REVIEWED", "decoder bundle fail-closed status")
    bootstrap = snapshots["bootstrap.py"].decode("utf-8")
    required_tokens = (
        "BLOCK_UNRESOLVED_EMBEDDED_PRODUCTION_PINS",
        "BLOCK_DIRECT_PRODUCTION_LAUNCH_REQUIRES_OUT_OF_TREE_PINNED_AUTHORITY",
        "ExternalLaunchAuthority",
        "require_isolated_cpython",
        "HeldDirectory",
        "HeldRegular",
        "authenticate_producer",
        "authenticate_dispatcher",
        "authenticate_runtime",
        "authenticate_decoder_bundle",
        "validate_logical_to_producer_member_map",
        "numpy_strata_ordinal_bridge_sha256",
        "reject_preloaded_snapshot_modules",
        "reject_preloaded_numeric_modules",
        "compile_producer_snapshots",
        "AppendOnlyImportNativeLedger",
        "AuthenticatedManifestFinder",
        "reject_ambient_import_hooks",
        "activate_import_native_enforcement",
        "PROCESS_START_RTLD_AUDIT",
        "BEFORE_AUTHORITATIVE_PREFLIGHT",
        "IMPORT_NATIVE_EVENT_CLOSURE_FINAL",
        "run_fresh_typed_source_free_preflight",
        "open_source_inputs_after_preflight",
        "validate_baseline_plan",
        "construct_bound_baseline_score",
        "validate_external_bandwidth_gate",
        "derive_authenticated_container_framing",
        "requested_bytes_with_repetition",
        "coalesced_unique_requested_bytes",
        "passes_all_bandwidth_gates",
        "CompletionLastOutput",
        "verify_completed_under_parent",
        "reject_authority_request_output_inode_aliasing",
        "PASS_MATCHED_NULL_SPECIFICITY_AWAITING_EXTERNAL_RESULT_AUDIT",
    )
    for token in required_tokens:
        require(token in bootstrap, f"bootstrap contract token: {token}")
    require("import cupy" not in bootstrap and "import numpy" not in bootstrap, "no unauthenticated direct numeric import")
    tests = snapshots["test_source_only.py"].decode("utf-8")
    for token in (
        "test_request_is_inaccessible_before_preflight_then_held",
        "test_numpy_scalar_ordinal_bridge_preserves_value_and_source_order",
        "test_numpy_2d_advanced_index_bridge_requires_list_rows",
        "test_dispatch_has_no_caller_production_pins_parameter",
        "test_runtime_loaded_image_closure_rejects_unheld_python_or_native_image",
        "test_authenticated_loader_rejects_path_substitution_between_resolution_and_exec",
        "test_hostile_meta_path_and_path_hook_are_rejected",
        "test_preloaded_numeric_module_is_rejected_before_loader_install",
        "test_import_then_delete_sys_modules_remains_in_ledger_and_fails",
        "test_transient_native_load_unload_events_are_append_only_and_unmanifested_load_fails",
        "test_import_native_enforcement_order_precedes_snapshots_numeric_import_and_preflight",
        "test_authority_exact_request_pin_rejects_before_path_access",
        "test_baseline_plan_is_parsed_sealed_and_cross_bound",
        "test_bandwidth_gate_rejects_literal_length_and_authenticated_denominator_substitution",
        "test_output_verifier_retains_fd_and_finally_rebinds_name_inode_bytes",
        "test_preloaded_snapshot_rejected",
        "test_score_is_derived_from_adapter_geometry_not_request",
        "test_repeated_request_gate_catches_unique_page_pass",
        "test_bandwidth_gate_rejects_resealed_summary_and_fraction_tamper",
        "test_direct_execution_blocks_before_argument_or_payload_handling",
    ):
        require(token in tests, f"hostile test token: {token}")
    return {
        "schema": "uwfa-sc-v8-external-dispatcher-source-verification-v2",
        "status": "PASS_SOURCE_ONLY_REVIEW_CANDIDATE_NO_PAYLOAD_AUTHORITY",
        "source_manifest_sha256": sha256(manifest_bytes),
        "members": observed,
        "producer_manifest_pin": pins["producer_source_manifest_sha256"],
        "producer_review_pin": pins["producer_final_review_sha256"],
        "producer_public_commit_pin": pins["producer_public_git_commit"],
        "production_launch_enabled": False,
        "payload_authority_granted": False,
        "access_attestation_replayed": attestation,
        "remaining_authority_gates": remaining,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = verify(Path(args.package))
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) if args.compact else json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_EXTERNAL_DISPATCHER_SOURCE_VERIFICATION: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
