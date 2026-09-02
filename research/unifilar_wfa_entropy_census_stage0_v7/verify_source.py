#!/usr/bin/env python3
"""Native verifier for a post-review sealed UWFA-SC v7 source package."""

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
    "protocol.py",
    "universal_adapter.py",
    "container_codec.py",
    "strata_sc_adapter.py",
    "stage0_census.py",
    "cupy_backend.py",
    "dispatcher_contract.py",
    "result_envelope.py",
    "fixture_long_memory.py",
    "fixture_portability.py",
    "test_source_only.py",
    "verify_source.py",
    "run_source_free_gpu_dev.py",
}

POST_FREEZE_REQUIREMENTS = [
    "EXTERNAL_PINNED_V7_SOURCE_AUDIT",
    "EXTERNAL_PINNED_DISPATCHER_AUDIT",
    "PUBLIC_GITHUB_COMMIT_FRESH_RTX5090_ALL150_AND_REPRESENTATIVE_REPLAY",
    "FRESH_PROCESS_INDEPENDENT_RESULT_AUDIT",
    "NO_PAYLOAD_BEFORE_EXTERNAL_AUTHORITY",
]


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def regular_bytes(path: Path) -> bytes:
    require(path.is_absolute(), "absolute source path")
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor = cursor / component
        require(os.path.lexists(cursor), f"source component absent: {cursor}")
        info = os.lstat(cursor)
        require(not stat.S_ISLNK(info.st_mode), f"source symlink component: {cursor}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), "source member not regular")
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                "source member changed through held descriptor")
        require(len(data) == before.st_size, "source fstat size")
        return data
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
    exec(compile(source, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


def verify_package(package: Path) -> dict[str, Any]:
    package = package.absolute()
    require(package.is_dir(), "package directory")
    manifest_bytes = regular_bytes(package / "SOURCE_MANIFEST.json")
    manifest = strict_json(manifest_bytes)
    require(set(manifest) == {"schema", "status", "members", "access_attestation", "post_freeze_requirements"}, "manifest fields")
    require(manifest["schema"] == "unifilar-wfa-source-manifest-v7", "manifest schema")
    require(manifest["status"] == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "manifest status")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "manifest members")
    names: set[str] = set()
    snapshots: dict[str, bytes] = {}
    observed = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "manifest row fields")
        name = row["name"]
        require(isinstance(name, str) and name == Path(name).name and name not in names and name != "SOURCE_MANIFEST.json", "manifest member name")
        data = regular_bytes(package / name)
        require(type(row["bytes"]) is int and row["bytes"] == len(data), f"member bytes: {name}")
        require(isinstance(row["sha256"], str) and row["sha256"] == sha256(data), f"member digest: {name}")
        names.add(name)
        snapshots[name] = data
        observed.append({"name": name, "bytes": len(data), "sha256": sha256(data)})
    require(names == EXPECTED_MEMBERS, "frozen member set")
    actual = {entry.name for entry in os.scandir(package)}
    require(actual == names | {"SOURCE_MANIFEST.json"}, "undeclared or missing source member")

    for name, data in snapshots.items():
        if name.endswith(".py"):
            compile(data, f"<syntax:{name}>", "exec", dont_inherit=True, optimize=0)

    design = strict_json(snapshots["design_lock.json"])
    require(design.get("schema") == "unifilar-wfa-entropy-census-design-v7", "design schema")
    require(design.get("status") == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "design sealed status")
    require(design.get("post_freeze_requirements") == POST_FREEZE_REQUIREMENTS, "design post-freeze requirements")
    attestation = {
        "model_or_qwen_payload_opened_statted_hashed_or_enumerated": False,
        "current_finite_artifact_or_selected_stream_opened_statted_hashed_or_enumerated": False,
        "gaussian_control_opened_statted_hashed_or_enumerated": False,
        "numpy_directly_imported_by_builder": False,
        "source_free_cupy_development_run_launched_by_builder": True,
        "source_free_cuda_development_run_launched_by_builder": True,
        "development_run_is_claim_evidence": False,
    }
    require(design.get("access_attestation") == attestation, "design access attestation")
    require(manifest["access_attestation"] == attestation, "manifest access attestation")
    require(manifest["post_freeze_requirements"] == POST_FREEZE_REQUIREMENTS, "manifest post-freeze requirements")

    common = module_from_snapshot("uwfa_verify_common_v7", snapshots["uwfa_common.py"])
    require(len(common.candidate_bank()) == 150, "candidate bank")
    require(common.STATE_SIZES == (2, 4, 8, 16, 32, 64), "state sizes")
    require(common.RESET_LENGTHS == (32, 128, 512, 2048, 4096), "reset lengths")
    require(abs(common.STANDALONE_REQUIRED_SAVING_BPW - 0.15288996696291447) < 1e-15, "physical threshold")
    model = common.serialize_model(common.candidate_bank()[0], [32768] * common.model_frequency_count(common.candidate_bank()[0]))
    require(model.startswith(b"UWFAV7\x00\x00"), "v7 serialized model")
    require(common.serialize_model(*common.deserialize_model(model)) == model, "model canonical round trip")

    protocol_text = snapshots["protocol.py"].decode("utf-8")
    require("MAX_EXPERTS = 256" in protocol_text and "OWNER_SET_BYTES = 32" in protocol_text, "universal owner ABI")
    legacy_owner_token = "owner_" + "mask"
    require(legacy_owner_token not in protocol_text, "legacy owner integer alias")
    adapter_text = snapshots["universal_adapter.py"].decode("utf-8")
    require("def validate_stream_coverage" in adapter_text and "def decode_with_callbacks" in adapter_text, "generic adapter protocol")
    container_text = snapshots["container_codec.py"].decode("utf-8")
    for token in ("UWFCV7", "UWFDIR4", "UWFREG4", "UWFFRM4", "VERSION = 4", "DIRECTORY_RECORD_BYTES = 256", "REGION_HEADER_BYTES = 256", "FRAME_HEADER_BYTES = 256", "def measure_literal_container_layout", "serialized container differs from shared literal measurement", "def routed_read_expert", "def physical_metrics", "def posterior_diagnostic_handoff", "decoded_selected_decision_triplet_sha256", "modeled_symbol_density", "routed_read_request_aggregates", "frozen_cold_gate_uses_unique_touched_page_bytes_only"):
        require(token in container_text, f"container contract: {token}")
    require("decoded_symbol_bits_sha256" not in container_text, "ambiguous posterior bit-only label")
    legacy_float_floor_token = "minimum_rate_" + "bpw"
    require(legacy_float_floor_token not in container_text, "floating rate-floor inference")
    stage_text = snapshots["stage0_census.py"].decode("utf-8")
    require("BLOCK_DIRECT_EXECUTION_REQUIRES_EXTERNALLY_PINNED_DISPATCHER" in stage_text, "direct-launch block")
    require("def representative_outer_fold_benchmark" in stage_text, "representative benchmark")
    for token in (
        "class SourcePreflightEvidence", "def validate_source_preflight",
        "source_full_geometry_sha256", "source_structural_geometry_sha256",
        "passes_pre_fit_resource_budget", "policy=\"coordinate_disjoint\"",
        "def _validate_all150_receipt", "selectors != list(range(150))",
        "candidate_selector_sha256", "cell_results_sha256",
        "def _validate_representative_receipt", "candidate_scores_sha256",
        "canonical_rebuild_sha256", "decoded_triplet_commitment_sha256",
        "def _validate_telemetry_statistics", "def _validate_resource_plan",
        "def _expected_all150_workload", "representative telemetry/workload mismatch",
        "def literal_validation_score", "def _dependence_components",
        "HOLD_NONESTIMABLE_DEPENDENCE_COMPONENTS", "no iid confidence interval",
        "authenticated_source_artifact_sha256",
        "caller source artifact digest differs from authenticated source state",
        "source_bindings.symmetric_control_closure()",
    ):
        require(token in stage_text, f"v7 source gate: {token}")
    require("import cupy" not in stage_text and "spec_from_file_location" not in stage_text, "producer path/dynamic import")
    plugin_text = snapshots["strata_sc_adapter.py"].decode("utf-8")
    require("EVALUATION_PLUGIN_FIXED_STRATA15_NOT_UNIVERSAL" in plugin_text, "plugin claim scope")
    require("sys.path" not in plugin_text and "strata_expert_local_codec" not in plugin_text, "repository-relative plugin import")
    cuda_text = snapshots["cupy_backend.py"].decode("utf-8")
    for token in ("h2d_model_table_bytes", "h2d_launch_descriptor_bytes", "h2d_kernel_scalar_bytes", "d2h_bytes", "peak_process_tree_rss_bytes", "peak_vram_incremental_bytes", "peak_pinned_pool_free_blocks", "def pack_resource_plan", "device_uuid", "pci_bus_id", "fatal_telemetry_sampling"):
        require(token in cuda_text, f"telemetry contract: {token}")
    common_text = snapshots["uwfa_common.py"].decode("utf-8")
    for token in (
        "class RetainedOutputParent", "class CompletionLastOutput", "renameat2",
        "RENAME_NOREPLACE", "def _verify_exact_staging_members",
        "with os.scandir(self.dir_fd)",
        "os.open(name, flags, dir_fd=self.dir_fd)",
        "digest.hexdigest() == row[\"sha256\"]",
        "def _verify_final_name_binding", "def _verify_named_commit_binding",
        "def _open_held_commit_authority_file", "O_TMPFILE",
        "def _link_held_unnamed_file_noreplace", "AT_EMPTY_PATH",
        "/proc/self/fd/", "def _linkat_syscall", "def _verify_held_commit_bytes",
        "held commit marker link count changed", "def committed_directory_root",
        "UWFA-V7-HELD-DIRECTORY-ROOT", "unifilar-wfa-parent-commit-v7",
    ):
        require(token in common_text, f"descriptor-relative publication contract: {token}")
    require(common_text.count("self._verify_exact_staging_members(") == 5, "held-directory verification at every V7 publication boundary")

    envelope_text = snapshots["result_envelope.py"].decode("utf-8")
    for token in (
        "class VerifiedOutputBundle", "def read_member_bytes",
        "MAX_BUFFERED_MEMBER_BYTES", "MAX_BUFFERED_AGGREGATE_BYTES",
        "def verify_completed_under_parent", "parent_commit_marker_name",
        "parent marker final-directory inode mismatch", "committed directory root mismatch",
        "completion/parent-marker member mismatch", "final directory name substituted during verification",
        "parent marker name substituted during verification",
    ):
        require(token in envelope_text, f"independent parent-marker verifier: {token}")

    protocol_text = snapshots["protocol.py"].decode("utf-8")
    for token in (
        "symmetric_closure", "baseline_plan_sha256",
        "universal_decoder_sha256", "producer_manifest_sha256",
        "audit_bootstrap_sha256", "extraction_program_sha256",
        "universal_adapter_sha256", "source_snapshot_root_sha256",
        "source_preflight_receipt_sha256",
        "def canonical_gpu_uuid", "def canonical_pci_bus_id",
    ):
        require(token in protocol_text, f"symmetric source/control closure: {token}")

    tests_text = snapshots["test_source_only.py"].decode("utf-8")
    for token in (
        "test_completion_rehash_rejects_mutated_declared_member_before_publication",
        "test_completion_reenumeration_rejects_undeclared_member_before_publication",
        "test_preflight_rejects_honestly_resealed_duplicate_cells_and_sparse_representative",
        "test_controls_reject_wrong_caller_artifact_and_foreign_symmetric_closure_before_fit",
        "test_parent_marker_rejects_staging_name_substitution_before_named_move",
        "test_parent_marker_rejects_final_substitution_after_move_before_marker",
        "test_parent_marker_makes_after_link_directory_substitution_unverifiable",
        "test_complete_json_without_parent_marker_is_never_complete",
        "test_verified_bundle_consumes_held_bytes_after_final_and_marker_path_replacement",
        "test_verified_bundle_fails_if_held_member_mutates_after_verification",
        "test_literal_alignment_can_reverse_raw_payload_candidate_order",
        "test_all_owner_shared_stream_collapses_to_one_component_and_holds_without_fit",
        "test_preflight_rejects_resealed_telemetry_not_bound_to_measured_workload",
        "test_marker_postlink_mutation_replacement_and_extra_hardlink_all_reject",
        "test_forced_otmpfile_and_both_descriptor_link_branches",
    ):
        require(token in tests_text, f"V7 hostile regression: {token}")

    return {
        "schema": "unifilar-wfa-source-verification-v7",
        "status": "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY",
        "source_manifest_sha256": sha256(manifest_bytes),
        "members": observed,
        "candidate_cells": 150,
        "expert_count_range_inclusive": [1, 256],
        "owner_set_bytes": 32,
        "payload_authority_granted": False,
        "post_freeze_requirements": POST_FREEZE_REQUIREMENTS,
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
