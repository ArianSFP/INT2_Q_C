#!/usr/bin/env python3
"""Standard-library source/design-lock verifier; imports no producer or CuPy."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_LINEAGE = {
    "source_only_manifest_sha256": "23a7566cf9ead5191c778a9dda30e880646a32d81357ff940182dc74e11bfe99",
    "parent_normal_result_sha256": "e4fecac5f676d84739972bbf0e04467027aeae1356e62e1dc3cd2b84bff67026",
    "parent_producer_sha256": "04d033319b4bbab037b48355e5f296274ae77b1c787ddcd2508e9b58948d265e",
    "composite_result_sha256": "565e1eb2122f2e476c5bd81e4205eeb3e4cede6e6a51149e95944355199eb41c",
    "source_lock_sha256": "bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, label: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(label)
    checks.append(label)


def imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def main() -> None:
    package = Path(__file__).resolve().parent
    checks: list[str] = []
    lock_path = package / "DESIGN_LOCK.json"
    protocol_path = package / "gate_protocol.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(lock["schema"] == "polar_sparse_normal_qwen_gate_design_lock_v1", "lock_schema", checks)
    require(lock["status"] == "LOCKED_READY_NOT_EXECUTED", "ready_not_executed", checks)
    require(lock["immutable_lineage"] == EXPECTED_LINEAGE, "immutable_lineage", checks)
    require(lock["scope"]["authenticated_matrices"] == 18, "matrix_count_18", checks)
    require(lock["scope"]["fresh_validation_access"] is False, "no_fresh_validation", checks)
    require(lock["scope"]["router_access"] is False, "no_router", checks)
    require(lock["scope"]["gpu_job_active_at_lock"] is False, "no_active_gpu_job", checks)

    for row in lock["source_artifacts"]:
        path = package / row["path"]
        require(path.is_file() and not path.is_symlink(), f"regular:{row['path']}", checks)
        require(path.stat().st_size == int(row["bytes"]), f"bytes:{row['path']}", checks)
        require(sha256_file(path) == row["sha256"], f"sha256:{row['path']}", checks)

    require(protocol["objective"]["target_F_maximum"] == 0.8, "target_F", checks)
    require(
        protocol["objective"]["hard_value_ledger_bits_per_selected_coordinate"] == 1,
        "onebit_hard_ledger",
        checks,
    )
    require(
        protocol["objective"]["selected_values_reconstructed_continuously_without_error"] is True,
        "continuous_exact_values",
        checks,
    )
    require(
        protocol["families"]["arbitrary_coordinate"]["decision_eligible"] is True,
        "coordinate_decision",
        checks,
    )
    require(
        protocol["families"]["fixed_triangular_tiles"]["broad_family_kill_only_from_gross_relaxation"]
        is True,
        "tile_gross_only",
        checks,
    )
    require(
        protocol["families"]["fixed_offset_segments"]["broad_family_kill_only_from_gross_relaxation"]
        is True,
        "offset_gross_only",
        checks,
    )
    require(protocol["controls"]["decision_eligible"] is False, "controls_diagnostic", checks)
    require(protocol["read_contract"]["logical_read_amplification"] == 1.0, "logical_read_1x", checks)

    run_text = (package / "run_gate.py").read_text(encoding="utf-8")
    run_tree = ast.parse(run_text, filename="run_gate.py")
    verifier_tree = ast.parse(
        (package / "verify_result.py").read_text(encoding="utf-8"), filename="verify_result.py"
    )
    run_imports = imported_modules(run_tree)
    verifier_imports = imported_modules(verifier_tree)
    require("torch" not in run_imports, "runner_no_torch", checks)
    require("transformers" not in run_imports, "runner_no_transformers", checks)
    require("cupy" not in verifier_imports, "verifier_no_cupy", checks)
    require("numpy" not in verifier_imports, "verifier_no_numpy", checks)
    require("torch" not in verifier_imports, "verifier_no_torch", checks)
    require("fresh" not in {path.name.lower() for path in package.iterdir()}, "no_fresh_artifact", checks)
    require("gross_group_curve" in run_text, "gross_relaxation_implemented", checks)
    require("exact_choose_support" in run_text, "exact_support_implemented", checks)
    require("NUMERIC_ALLOWANCE_DISTORTION = 1e-9" in run_text, "numeric_allowance_locked", checks)
    require("control_summary" in run_text, "matched_controls_implemented", checks)
    require("source_receipts != parent_result" in run_text, "exact_source_receipt_binding", checks)
    require("normal_sha != prior" in run_text, "normal_byte_hash_binding", checks)
    require("fresh_validation_access\": False" in run_text, "result_no_fresh_claim", checks)

    print("PASS_POLAR_SPARSE_NORMAL_QWEN_GATE_SOURCE")
    print(f"checks {len(checks)}")
    print(f"design_lock_sha256 {sha256_file(lock_path)}")


if __name__ == "__main__":
    main()
