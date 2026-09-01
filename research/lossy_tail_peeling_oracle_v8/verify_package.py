#!/usr/bin/env python3
"""Fail-closed CPU/source-only verifier for frozen lossy-tail v8."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).absolute().parent
LOCAL_RESEARCH = HERE.parent
PINNED_RESEARCH = Path("/workspace/INT2__compression/INT2_Q_C/research")
RESEARCH = LOCAL_RESEARCH if (LOCAL_RESEARCH / "lossy_tail_peeling_oracle_v7").is_dir() else PINNED_RESEARCH
V7 = RESEARCH / "lossy_tail_peeling_oracle_v7"
V7_AUDIT = RESEARCH / "lossy_tail_peeling_oracle_v7_fresh_source_audit"
STAGE = {
    "authorization_contract.json", "audit_lock_entrypoint.py",
    "launch_manifest.json", "lossy_tail_core.py", "lossy_tail_oracle.py",
    "preflight_launch.py", "protocol_lock.json", "repair_lock.json",
    "runtime_calibrate.py", "runtime_contract.json", "source_bindings.json",
}
PACKAGE = STAGE | {
    "ARTIFACT_HASHES.json", "CPU_TEST_RECEIPT.json", "README.md",
    "test_lossy_tail_core.py", "test_release_security.py", "verify_package.py",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            value.update(block)
    return value.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def strict_pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in rows:
        check(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)
    check(isinstance(value, dict), f"JSON object required: {path.name}")
    return value


def verify_seal(path: Path, field: str) -> dict[str, Any]:
    value = load_json(path)
    claimed = value.get(field)
    check(
        isinstance(claimed, str) and len(claimed) == 64
        and all(character in "0123456789abcdef" for character in claimed),
        f"invalid internal seal: {path.name}",
    )
    copy = dict(value)
    copy.pop(field)
    check(hashlib.sha256(canonical(copy)).hexdigest() == claimed, f"internal seal mismatch: {path.name}")
    return value


def verify_zero_slot_seal(path: Path, field: str) -> dict[str, Any]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=strict_pairs)
    check(isinstance(value, dict), f"JSON object required: {path.name}")
    claimed = value.get(field)
    check(isinstance(claimed, str) and len(claimed) == 64, f"invalid zero-slot seal: {path.name}")
    claimed_bytes = claimed.encode("ascii")
    check(payload.count(claimed_bytes) == 1, f"zero-slot seal occurrence drift: {path.name}")
    zeroed = payload.replace(claimed_bytes, b"0" * 64, 1)
    check(hashlib.sha256(zeroed).hexdigest() == claimed, f"zero-slot seal mismatch: {path.name}")
    return value


def top_level_closure() -> None:
    observed: set[str] = set()
    with os.scandir(HERE) as entries:
        for entry in entries:
            check(not entry.is_symlink(), f"symlink package entry: {entry.name}")
            check(stat.S_ISREG(entry.stat(follow_symlinks=False).st_mode), f"non-regular package entry: {entry.name}")
            observed.add(entry.name)
    check(observed == PACKAGE, f"package closure mismatch: {sorted(observed ^ PACKAGE)}")


def exact_rows(rows: Any, expected: set[str], label: str) -> None:
    check(isinstance(rows, list) and len(rows) == len(expected), f"{label} cardinality")
    paths = [row.get("path") for row in rows if isinstance(row, dict)]
    check(len(paths) == len(rows) and len(set(paths)) == len(paths) and set(paths) == expected, f"{label} closure")
    for row in rows:
        check(set(row) == {"path", "bytes", "sha256"}, f"{label} keys: {row.get('path')}")
        path = HERE / row["path"]
        check(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], f"{label} identity: {row['path']}")


def imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def deleted_names(node: ast.Delete) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def main() -> None:
    check(sys.flags.optimize == 0, "optimized verifier execution is forbidden")
    check(not (HERE / "__pycache__").exists(), "__pycache__ forbidden")
    top_level_closure()

    artifact = verify_seal(HERE / "ARTIFACT_HASHES.json", "artifact_manifest_sha256")
    check(artifact.get("schema") == "lossy-tail-v8-artifact-manifest-v1", "artifact schema")
    check(artifact.get("status") == "FROZEN_SOURCE_ONLY_CANDIDATE_REQUIRES_FRESH_INDEPENDENT_AUDIT", "artifact status")
    exact_rows(artifact.get("members"), PACKAGE - {"ARTIFACT_HASHES.json"}, "artifact rows")

    manifest_path = HERE / "launch_manifest.json"
    manifest = load_json(manifest_path)
    check(manifest.get("schema") == "lossy-tail-v8-launch-manifest-v1", "launch schema")
    check(manifest.get("status") == "FROZEN_V8_SOURCE_STAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION", "launch status")
    allowed = manifest.get("allowed_members")
    check(
        isinstance(allowed, list) and len(allowed) == len(STAGE)
        and len(set(allowed)) == len(allowed) and set(allowed) == STAGE,
        "launch allowed-member closure",
    )
    exact_rows(manifest.get("members"), STAGE - {"launch_manifest.json"}, "launch rows")
    check(manifest["authorization"].startswith("NONE_EXISTS"), "manifest unexpectedly authorizes execution")
    check("env CUDA_VISIBLE_DEVICES=0" in manifest["runtime_calibration_invocation_after_independent_source_pass_only"], "runtime CUDA binding")
    check("env CUDA_VISIBLE_DEVICES=0" in manifest["production_invocation_after_independent_runtime_receipt_audit_and_separate_authorization_only"], "production CUDA binding")

    repair = verify_seal(HERE / "repair_lock.json", "repair_lock_sha256")
    check(repair.get("schema") == "lossy-tail-release-repair-lock-v8", "repair schema")
    check(repair.get("status") == "FROZEN_V8_SOURCE_PACKAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION", "repair status")
    identities = {
        "scientific_protocol_sha256": digest(HERE / "protocol_lock.json"),
        "source_bindings_sha256": digest(HERE / "source_bindings.json"),
        "runtime_contract_sha256": digest(HERE / "runtime_contract.json"),
        "authorization_contract_sha256": digest(HERE / "authorization_contract.json"),
        "oracle_bootstrap_sha256": digest(HERE / "lossy_tail_oracle.py"),
        "scientific_core_sha256": digest(HERE / "lossy_tail_core.py"),
        "preflight_sha256": digest(HERE / "preflight_launch.py"),
        "audit_entrypoint_sha256": digest(HERE / "audit_lock_entrypoint.py"),
        "runtime_calibrate_sha256": digest(HERE / "runtime_calibrate.py"),
    }
    check(repair.get("authenticated_identities") == identities, "repair authenticated identities")
    check(len(repair.get("repairs", [])) == 5, "five-blocker repair cardinality")
    check(repair["runtime_release_sequence"] == {
        "fresh_independent_v8_source_audit_required": True,
        "runtime_calibration_authorized_now": False,
        "runtime_receipt_exists": False,
        "production_authorization_exists": False,
        "production_run_authorized": False,
    }, "repair release authority drift")

    predecessor = repair["preserved_v7"]
    check(digest(V7 / "launch_manifest.json") == predecessor["launch_manifest_sha256"] == "3d5bc5ed95071cc45406d0d2906b54f40d32adad0dffc6323b8fa80ca491ed63", "v7 manifest drift")
    check(digest(V7 / "lossy_tail_core.py") == predecessor["scientific_core_sha256"] == "d1393a80b4e2b48a30c61ec0b519db9fe35b26c0aed12ca4287023f49d88fd79", "v7 scientific core drift")
    check(digest(V7 / "repair_lock.json") == predecessor["repair_lock_file_sha256"] == "6f879caed79aed81f824968a5188ef809ee1a8a1184ed5262b3e23633e18c781", "v7 repair lock drift")
    audit = repair["fresh_independent_v7_block_audit"]
    check(digest(V7_AUDIT / "audit_manifest.json") == audit["manifest_file_sha256"] == "120b616c726253a82850e93b720e48c56c2aa7af59f1c2b7ec288bec215e4621", "v7 fresh audit manifest drift")
    check(digest(V7_AUDIT / "audit_receipt.json") == audit["receipt_file_sha256"] == "b82146a04188b74a3213fd54db2ee1bb34c7d132dd685b8169b7f8ce36a78dff", "v7 fresh audit receipt drift")
    v7_manifest = verify_zero_slot_seal(V7_AUDIT / "audit_manifest.json", "audit_manifest_internal_sha256")
    check(v7_manifest["audit_manifest_internal_sha256"] == audit["manifest_internal_sha256"] == "ff8fa4ccb2ded027d4bd22b00b8a00d1014b92f9888ef55be1c68377d496e0f9", "v7 fresh audit manifest internal drift")
    v7_receipt = verify_zero_slot_seal(V7_AUDIT / "audit_receipt.json", "audit_receipt_internal_sha256")
    check(v7_receipt["audit_receipt_internal_sha256"] == audit["receipt_internal_sha256"] == "8c66b7bfe3f076fa00c993ac3862069268dbad9dbc4572f833d66401967e1ec4", "v7 fresh audit receipt internal drift")
    check(v7_receipt.get("status") == audit["status"] == "BLOCKED_SOURCE_ONLY_RELEASE_CONFORMANCE", "v7 fresh audit status drift")

    protocol = load_json(HERE / "protocol_lock.json")
    check(protocol.get("schema") == "qwen-lossy-tail-peeling-oracle-protocol-v8", "protocol schema")
    check(protocol.get("status") == "FROZEN_V8_BEFORE_ANY_RUNTIME_CALIBRATION_PAYLOAD_OR_GPU_EXECUTION", "protocol status")
    check(protocol["target"]["required_s_bpw"] == 0.16096404744368115, "target score")
    check(protocol["target"]["maximum_cold_and_page_read_amplification_exclusive"] == 2.0, "read limit")
    runtime = load_json(HERE / "runtime_contract.json")
    check(runtime.get("schema") == "lossy-tail-v8-runtime-calibration-contract-v1", "runtime schema")
    check(runtime.get("status") == "FROZEN_SOURCE_FREE_BEFORE_RUNTIME_CALIBRATION", "runtime status")
    check(runtime["probe"]["replicas"] == [0, 1, 2, 3] and runtime["probe"]["ordinals"] == list(range(12)), "48-cell closure")
    check(runtime["probe"]["memory_evidence_required_values"] == {
        "used_bytes_before_free": 0, "used_bytes_after_free": 0,
        "total_bytes_after_free": 0, "all_per_cell_gpu_arrays_deleted_before_free": True,
    }, "memory evidence contract")
    authorization = load_json(HERE / "authorization_contract.json")
    check(authorization.get("status") == "FROZEN_TEMPLATE_ONLY_NO_AUTHORIZATION_EXISTS", "authorization template status")

    entrypoints = [
        HERE / "audit_lock_entrypoint.py", HERE / "preflight_launch.py",
        HERE / "runtime_calibrate.py", HERE / "lossy_tail_oracle.py",
        HERE / "lossy_tail_core.py",
    ]
    for path in entrypoints:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        check(not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), f"assert in release entrypoint: {path.name}")
        check("AssertionError" not in source, f"AssertionError in release entrypoint: {path.name}")

    bootstrap_source = (HERE / "lossy_tail_oracle.py").read_text(encoding="utf-8")
    bootstrap_tree = ast.parse(bootstrap_source)
    check(imported_roots(bootstrap_tree).isdisjoint({"numpy", "cupy", "torch"}), "bootstrap third-party import")
    for token in (
        "SO_PEERCRED", "SOCK_SEQPACKET", "verify_inherited_preflight",
        "payload != expected_payload", "preflight_memfd_seals",
        "capability channel contains more than one record",
        "CONSUMED_ONCE_BEFORE_THIRD_PARTY_IMPORT",
    ):
        check(token in bootstrap_source, f"capability proof missing: {token}")
    preflight_source = (HERE / "preflight_launch.py").read_text(encoding="utf-8")
    check("more than one acknowledgement record" in preflight_source, "acknowledgement EOF proof missing")
    for token in ("os.memfd_create", "F_ADD_SEALS", "F_GET_SEALS", "os.execve", "validate_sealed_preflight_descriptor"):
        check(token in preflight_source, f"sealed preflight proof missing: {token}")
    check("/proc/self/cmdline" not in preflight_source + bootstrap_source, "mutable cmdline provenance returned")

    core_source = (HERE / "lossy_tail_core.py").read_text(encoding="utf-8")
    core_tree = ast.parse(core_source)
    numpy_index = next(index for index, node in enumerate(core_tree.body) if isinstance(node, ast.Import) and any(alias.name == "numpy" for alias in node.names))
    firewall_index = next(index for index, node in enumerate(core_tree.body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "") == "_v8_preimport_production_firewall")
    check(firewall_index < numpy_index, "production firewall does not precede NumPy")
    runtime_probe = next(node for node in core_tree.body if isinstance(node, ast.FunctionDef) and node.name == "runtime_probe")
    required_deletes = {"raw", "bits", "rounding", "words", "zbf", "table_words", "table", "affine_table", "gathered", "rng"}
    deletion_lines = [node.lineno for node in ast.walk(runtime_probe) if isinstance(node, ast.Delete) and deleted_names(node) >= required_deletes]
    free_lines = [node.lineno for node in ast.walk(runtime_probe) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "free_all_blocks"]
    check(deletion_lines and free_lines and min(deletion_lines) < min(free_lines), "runtime consumer deletion barrier")
    memory_fields = {
        "stream_synchronized", "used_bytes_before_free", "total_bytes_before_free",
        "used_bytes_after_free", "total_bytes_after_free",
        "all_per_cell_gpu_arrays_deleted_before_free",
    }
    memory_ledgers = 0
    for node in ast.walk(runtime_probe):
        if isinstance(node, ast.Dict):
            keys = {key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
            memory_ledgers += int(memory_fields <= keys)
    check(memory_ledgers == 2, "48-cell/five-adversary six-field memory shape")
    panel = next(node for node in core_tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_panel")
    panel_deleted = {
        item.id for node in ast.walk(panel) if isinstance(node, ast.Delete)
        for item in ast.walk(node) if isinstance(item, ast.Name)
    }
    check({"masks", "x", "words", "pair_masks", "pair_x", "pair_words"} <= panel_deleted, "panel live alias deletion")
    panel_source = ast.get_source_segment(core_source, panel) or ""
    for token in (
        "cp.cuda.get_current_stream().synchronize()", "if used_before_free != 0:",
        "pool.free_all_blocks()", "if used_after_free != 0 or total_after_free != 0:",
    ):
        check(token in panel_source, f"panel memory closure missing: {token}")
    writer = next(node for node in core_tree.body if isinstance(node, ast.FunctionDef) and node.name == "write_sealed_json_descriptor")
    writer_source = ast.get_source_segment(core_source, writer) or ""
    check(writer_source.count("dir_fd=output_parent_descriptor") >= 3, "output parent dirfd closure")
    for token in ("dir_fd=run_descriptor", "src_dir_fd=run_descriptor", "dst_dir_fd=run_descriptor", "os.O_EXCL", "O_NOFOLLOW", "os.fsync(output_parent_descriptor)"):
        check(token in writer_source, f"descriptor-relative output proof missing: {token}")
    frozen_statuses = {
        "SOURCE_AUDIT_MANIFEST_SCHEMA": "lossy-tail-v8-independent-source-audit-manifest-v1",
        "SOURCE_AUDIT_MANIFEST_STATUS": "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET",
        "SOURCE_AUDIT_RECEIPT_SCHEMA": "lossy-tail-v8-independent-source-audit-receipt-v1",
        "SOURCE_AUDIT_PASS_STATUS": "PASS_V8_INDEPENDENT_SOURCE_AUDIT",
        "RUNTIME_RECEIPT_STATUS": "UNTRUSTED_UNTIL_INDEPENDENT_RUNTIME_AUDIT",
        "RUNTIME_AUDIT_MANIFEST_SCHEMA": "lossy-tail-v8-independent-runtime-audit-manifest-v1",
        "RUNTIME_AUDIT_MANIFEST_STATUS": "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET",
        "RUNTIME_AUDIT_RECEIPT_SCHEMA": "lossy-tail-v8-independent-runtime-audit-receipt-v1",
        "RUNTIME_AUDIT_PASS_STATUS": "PASS_V8_INDEPENDENT_RUNTIME_AUDIT",
    }
    for name, value in frozen_statuses.items():
        token = f'{name} = "{value}"'
        check(token in preflight_source and token in core_source, f"frozen audit contract drift: {name}")
    check('"required_status"' not in preflight_source + core_source, "authorization-chosen required_status returned")
    for token in ("best_scored", "require_read_valid=True", "every_uniform_profile_evaluated", "coordinate candidate coverage mismatch", "require_finite_tree", "math.isfinite"):
        check(token in core_source, f"scientific repair token missing: {token}")

    receipt = verify_seal(HERE / "CPU_TEST_RECEIPT.json", "cpu_test_receipt_sha256")
    check(receipt.get("schema") == "lossy-tail-v8-source-only-cpu-test-receipt-v1", "CPU receipt schema")
    check(receipt.get("status") == "PASS_PRODUCER_CPU_AND_ADVERSARIAL_SOURCE_TESTS_LINUX_ONLY_TESTS_DEFERRED", "CPU receipt status")
    check(
        receipt.get("tests_run") == 33 and receipt.get("tests_passed") == 30
        and receipt.get("failures") == 0 and receipt.get("errors") == 0
        and receipt.get("skipped") == 3,
        "CPU test counts",
    )
    check(receipt["stage_identities"]["launch_manifest_sha256"] == digest(manifest_path), "CPU receipt manifest binding")
    check(receipt["stage_identities"]["repair_lock_internal_sha256"] == repair["repair_lock_sha256"], "CPU receipt repair binding")
    check(receipt["access_ledger"] == {
        "qwen_or_model_paths_traversed": 0, "model_payload_files_opened": 0,
        "production_result_files_opened": 0, "torch_imports": 0, "cupy_imports": 0,
        "cuda_initializations": 0, "gpu_jobs": 0, "runtime_calibrations": 0,
        "production_runs": 0, "production_outputs_created": 0,
        "source_free_capability_stub_boundaries_reached": 0,
        "network_calls": 0,
    }, "CPU receipt access ledger")
    check(receipt["authorization"] == {
        "fresh_independent_source_audit_required": True,
        "runtime_calibration_authorized": False,
        "production_authorization_exists": False,
        "production_run_authorized": False,
    }, "CPU receipt authority")

    print(json.dumps({
        "lossy_tail_v8_source_candidate": "PASS_PRODUCER_VERIFICATION_ONLY",
        "artifact_file_sha256": digest(HERE / "ARTIFACT_HASHES.json"),
        "artifact_internal_sha256": artifact["artifact_manifest_sha256"],
        "launch_manifest_sha256": digest(manifest_path),
        "repair_lock_file_sha256": digest(HERE / "repair_lock.json"),
        "repair_lock_internal_sha256": repair["repair_lock_sha256"],
        "tests": receipt["tests_run"],
        "payload_files_opened": 0,
        "fresh_independent_audit_required": True,
        "runtime_or_production_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
