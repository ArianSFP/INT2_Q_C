#!/usr/bin/env python3
"""Pure-standard-library integrity and contract verifier for FOSP-v2."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "ARTIFACT_SHA256SUMS.txt"
EXPECTED_FILES = {
    "README.md",
    "calibrate_runtime.py",
    "create_authorization.py",
    "free_order_oracle_v2.py",
    "protocol_lock.json",
    "source_bindings.json",
    "source_only_receipt.json",
    "test_source_only.py",
    "verify_package.py",
}
RECEIPT_HASHED_FILES = EXPECTED_FILES - {"source_only_receipt.json"}
SHA_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)
        self.count += 1

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        self.require(actual == expected, f"{message}: {actual!r} != {expected!r}")

    def close(
        self,
        actual: float,
        expected: float,
        message: str,
        tolerance: float = 1e-15,
    ) -> None:
        self.require(abs(actual - expected) <= tolerance, f"{message}: {actual!r} != {expected!r}")


def regular_bytes(path: Path, checks: Checks) -> bytes:
    checks.require(path.exists(), f"missing {path.name}")
    checks.require(path.is_file(), f"not a regular file {path.name}")
    checks.require(not path.is_symlink(), f"symlink forbidden {path.name}")
    return path.read_bytes()


def sha256(path: Path, checks: Checks) -> str:
    return hashlib.sha256(regular_bytes(path, checks)).hexdigest()


def load_json(path: Path, checks: Checks) -> dict[str, Any]:
    value = json.loads(regular_bytes(path, checks).decode("utf-8"))
    checks.require(isinstance(value, dict), f"{path.name} root must be object")
    return value


def parse_artifact_manifest(checks: Checks) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(regular_bytes(MANIFEST, checks).decode("ascii").splitlines(), 1):
        match = SHA_RE.fullmatch(line)
        checks.require(match is not None, f"malformed artifact manifest line {line_number}")
        assert match is not None
        digest, name = match.groups()
        checks.require(name not in rows, f"duplicate artifact name {name}")
        rows[name] = digest
    checks.equal(set(rows), EXPECTED_FILES, "artifact manifest file closure")
    observed_files = {path.name for path in ROOT.iterdir() if path.is_file() and path.name != MANIFEST.name}
    observed_directories = [path.name for path in ROOT.iterdir() if path.is_dir()]
    checks.equal(observed_files, EXPECTED_FILES, "package regular-file closure")
    checks.equal(observed_directories, [], "package directory closure")
    for name, expected in sorted(rows.items()):
        checks.equal(sha256(ROOT / name, checks), expected, f"SHA-256 {name}")
    return rows


def top_imports(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def string_literals(tree: ast.Module) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def verify() -> dict[str, Any]:
    checks = Checks()
    artifacts = parse_artifact_manifest(checks)
    lock = load_json(ROOT / "protocol_lock.json", checks)
    bindings = load_json(ROOT / "source_bindings.json", checks)
    receipt = load_json(ROOT / "source_only_receipt.json", checks)
    oracle_raw = regular_bytes(ROOT / "free_order_oracle_v2.py", checks)
    calibration_raw = regular_bytes(ROOT / "calibrate_runtime.py", checks)
    builder_raw = regular_bytes(ROOT / "create_authorization.py", checks)
    tests_raw = regular_bytes(ROOT / "test_source_only.py", checks)
    oracle_tree = ast.parse(oracle_raw.decode("utf-8"), filename="free_order_oracle_v2.py")
    calibration_tree = ast.parse(calibration_raw.decode("utf-8"), filename="calibrate_runtime.py")
    builder_tree = ast.parse(builder_raw.decode("utf-8"), filename="create_authorization.py")
    ast.parse(tests_raw.decode("utf-8"), filename="test_source_only.py")

    checks.equal(lock["schema"], "free_order_swiglu_path_protocol_v2", "protocol schema")
    checks.equal(
        lock["status"],
        "SOURCE_ONLY_DEPLOYMENT_BLOCKED_PENDING_INDEPENDENT_AUDITS_AND_ONE_SHOT_AUTHORIZATION",
        "source-only status",
    )
    checks.require(lock["supersedes_for_future_execution_only"]["v1_remains_immutable"] is True, "v1 immutable")
    checks.equal(
        lock["supersedes_for_future_execution_only"]["v1_audit_status"],
        "BLOCK_STAGE0_DOES_NOT_CONTAIN_CROSS_ROLE_3X3_PATH_FAMILY",
        "v1 block lineage",
    )
    objective = lock["objective"]
    checks.close(float(objective["required_net_s_bpw"]), -0.5 * math.log2(0.8), "net target s")
    checks.equal(objective["physical_rate_interval_bpw"], [2.15, 2.5], "physical rate interval")
    checks.equal(lock["geometry"]["weights_per_expert"], 3 * 768 * 2048, "expert geometry")

    metric = lock["metric_compatibility"]
    checks.equal(metric["zero_bit_deployed_gauge"], "INELIGIBLE_IMMEDIATE_KILL", "zero-bit verdict")
    checks.equal(metric["information_lower_bound_bits"], (math.factorial(768) - 1).bit_length(), "factorial bits")
    checks.equal(metric["information_lower_bound_bits"], 6260, "frozen factorial bits")
    checks.equal(metric["physical_factoradic_bytes"], math.ceil(6260 / 8), "factoradic bytes")
    checks.equal(metric["physical_factoradic_bits"], 6264, "factoradic physical bits")

    bridge = lock["eligible_physical_bridge"]
    coefficient_bits = 767 * 9 * 16
    total_side_bits = 64 * 8 + 783 * 8 + coefficient_bits
    weights = 3 * 768 * 2048
    checks.equal(bridge["coefficients_per_edge"], 9, "coefficients per edge")
    checks.equal(bridge["coefficient_count"], 767 * 9, "coefficient count")
    checks.equal(bridge["coefficient_bits"], coefficient_bits, "coefficient bits")
    checks.equal(bridge["total_side_bits"], total_side_bits, "total side bits")
    checks.close(float(bridge["total_side_bpw"]), total_side_bits / weights, "side bpw")
    required_gross = -0.5 * math.log2(0.8) + total_side_bits / weights
    checks.close(float(objective["required_gross_s_after_side_bpw"]), required_gross, "gross target s")

    gate = lock["scientific_gate"]
    checks.equal(gate["marginal_klt_stage"], "REMOVED_NONCONTAINING", "marginal KLT removal")
    checks.equal(gate["joint_dense_klt_stage"], "NOT_USED_AND_RECEIVES_NO_CREDIT", "joint KLT no credit")
    checks.equal(gate["stage_order"][0], "direct_qwen_full3x3_pair_panel", "first scientific gate")
    checks.equal(gate["pair_model"]["pair_count_per_expert"], 768 * 767, "ordered pair count")
    checks.close(
        float(gate["relaxed_containing_upper_bound"]["hard_gross_kill_if_s_below"]),
        required_gross,
        "gross hard kill",
    )
    checks.require(gate["controls"]["identical_pair_search"] is True, "identical control pair search")
    checks.require(gate["controls"]["identical_cycle_cover_path"] is True, "identical control path")
    checks.require(gate["controls"]["identical_fp16_rounding_and_replay"] is True, "identical control FP16")
    checks.equal(gate["controls"]["replicates"], 8, "control replicate count")
    checks.equal(len(set(gate["controls"]["seeds"])), 8, "distinct control seeds")

    for row in lock["rate_and_read"]["rows"]:
        rate = float(row["requested_rate_bpw"])
        frame = math.floor(weights * rate / 8)
        actual = frame * 8 / weights
        payload = (frame * 8 - total_side_bits) / weights
        cold_bytes = (math.ceil(frame / 4096) + 1) * 4096
        amplification = cold_bytes / frame
        checks.equal(row["frame_bytes"], frame, f"frame bytes {rate}")
        checks.close(float(row["actual_rate_bpw"]), actual, f"actual rate {rate}")
        checks.close(float(row["residual_payload_bpw"]), payload, f"payload rate {rate}")
        checks.equal(row["cold_page_bytes"], cold_bytes, f"cold bytes {rate}")
        checks.close(float(row["cold_page_amplification"]), amplification, f"cold amplification {rate}")
        checks.require(amplification < 2.0, f"cold read cap {rate}")

    checks.equal(bindings["schema"], "free_order_swiglu_path_auxiliary_bindings_v1", "bindings schema")
    checks.equal(len(bindings["experts"]), 2, "bound expert count")
    checks.equal(
        [(int(row["layer"]), int(row["expert"])) for row in bindings["experts"]],
        [(3, 57), (3, 121)],
        "fixed auxiliary identities",
    )
    for expert in bindings["experts"]:
        checks.equal([role["role"] for role in expert["roles"]], ["gate", "up", "down"], "joint roles")
        for role in expert["roles"]:
            checks.require(re.fullmatch(r"[0-9a-f]{64}", role["sha256"]) is not None, "source digest")
            relative = Path(role["relative_path"])
            checks.require(not relative.is_absolute() and ".." not in relative.parts, "safe fixed path")
    bindings_sha = sha256(ROOT / "source_bindings.json", checks)
    checks.equal(bindings_sha, "3454b718a65efc02c32463f955c10ff393f4218fac04f358107960ff3735990d", "binding hash")
    checks.equal(
        lock["execution_firewalls"]["source"]["source_bindings_sha256"],
        bindings_sha,
        "lock-to-binding hash",
    )

    checks.require(not ({"cupy", "numpy", "scipy", "torch"} & top_imports(oracle_tree)), "oracle heavy imports deferred")
    checks.require(not ({"cupy", "numpy", "scipy", "torch"} & top_imports(calibration_tree)), "calibration heavy imports deferred")
    checks.require(not ({"cupy", "numpy", "scipy", "torch"} & top_imports(builder_tree)), "builder heavy imports absent")
    oracle_literals = string_literals(oracle_tree)
    for required in ("--workspace-root", "--output", "--authorization", "--authorization-sha256"):
        checks.require(required in oracle_literals, f"required runner CLI {required}")
    for forbidden in ("--source", "--manifest", "--panel", "--validation", "--target"):
        checks.require(forbidden not in oracle_literals, f"forbidden runner CLI {forbidden}")
    calibration_literals = string_literals(calibration_tree)
    checks.require("--output" in calibration_literals, "calibration output CLI")
    for forbidden in ("--workspace-root", "--source", "--manifest", "--panel", "--validation"):
        checks.require(forbidden not in calibration_literals, f"forbidden calibration CLI {forbidden}")

    oracle_source = oracle_raw.decode("utf-8")
    checks.require("_dense_stage" not in oracle_source, "noncontaining dense stage absent")
    checks.require("eigvalsh" not in oracle_source, "KLT eigensolver absent")
    checks.require("reverse_waterfill" not in oracle_source, "waterfill gate absent")
    checks.require("_regular_bytes_at(root_descriptor" in oracle_source, "descriptor-relative source read")
    checks.require("dir_fd=parent_descriptor" in oracle_source, "descriptor-relative output commit")
    checks.require("output parent identity changed before commit" in oracle_source, "output parent identity check")
    checks.require("_verify_canonical_seal" in oracle_source, "canonical external seals")
    checks.require("_audit_manifest_binds" in oracle_source, "external audit manifest verification")
    for literal in (
        "free-order-swiglu-path-v2-independent-source-audit-receipt-v1",
        "PASS_V2_INDEPENDENT_SOURCE_AUDIT",
        "free-order-swiglu-path-v2-independent-runtime-audit-receipt-v1",
        "PASS_V2_INDEPENDENT_RUNTIME_AUDIT",
        "PASS_COUNTEREXAMPLE_REACHES_DIRECT_STAGE",
    ):
        checks.require(literal in oracle_literals, f"strict audit literal {literal}")
    checks.require("FORGED_PASS" in tests_raw.decode("utf-8"), "forged receipt negative test")
    checks.require("identity changed" in tests_raw.decode("utf-8"), "TOCTOU parent replacement test")
    checks.require("symlinked component" in tests_raw.decode("utf-8"), "symlink evidence test")

    checks.equal(receipt["schema"], "free_order_swiglu_path_v2_source_only_receipt_v1", "receipt schema")
    checks.equal(receipt["status"], "PASS_SOURCE_ONLY_PACKAGE_DEPLOYMENT_BLOCKED", "receipt status")
    checks.equal(receipt["artifact_set_status"], "IMMUTABLE_SOURCE_ONLY_ARTIFACT_SET", "receipt artifact status")
    checks.equal(set(receipt["package_files"]), RECEIPT_HASHED_FILES, "receipt file closure")
    for name in sorted(RECEIPT_HASHED_FILES):
        checks.equal(receipt["package_files"][name]["sha256"], artifacts[name], f"receipt hash {name}")
        checks.equal(receipt["package_files"][name]["bytes"], (ROOT / name).stat().st_size, f"receipt bytes {name}")
    zero = receipt["zero_access_ledger"]
    for key in (
        "qwen_or_model_payload_files_opened",
        "qwen_or_model_payload_bytes_read",
        "binding_paths_followed",
        "pinned_panel_files_opened",
        "validation_files_opened",
        "cupy_imports",
        "cuda_api_calls",
        "gpu_device_calls",
        "external_data_fetches",
        "runpod_or_remote_hosts_contacted",
    ):
        checks.equal(int(zero[key]), 0, f"source-only zero access {key}")
    unsigned_receipt = dict(receipt)
    internal_receipt_sha = str(unsigned_receipt.pop("canonical_unsigned_sha256"))
    checks.equal(internal_receipt_sha, canonical_sha256(unsigned_receipt), "receipt canonical seal")
    checks.require(receipt["deployment"]["authorized"] is False, "deployment remains blocked")
    checks.require(receipt["scientific_decision"]["cheap_auxiliary_run_justified_after_audits"] is True, "auxiliary rationale")

    return {
        "status": "PASS",
        "checks": checks.count,
        "artifact_manifest_sha256": hashlib.sha256(regular_bytes(MANIFEST, checks)).hexdigest(),
        "source_only_receipt_sha256": hashlib.sha256(regular_bytes(ROOT / "source_only_receipt.json", checks)).hexdigest(),
        "source_only_receipt_internal_sha256": internal_receipt_sha,
        "artifacts": artifacts,
        "deployment_authorized": False,
        "first_scientific_gate": gate["stage_order"][0],
        "required_gross_s_bpw": required_gross,
        "maximum_cold_read_amplification": max(float(row["cold_page_amplification"]) for row in lock["rate_and_read"]["rows"]),
    }


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
