#!/usr/bin/env python3
"""Pure-stdlib verifier for the independent BLOCK audit artifact."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parent
PRODUCER = ROOT.parent / "free_order_swiglu_path_oracle_v1"
AUDIT_MANIFEST = ROOT / "AUDIT_SHA256SUMS.txt"
AUDIT_FILES = {
    "README.md",
    "audit_receipt.json",
    "replay_receipt.json",
    "verify_audit.py",
}
PRODUCER_FILES = {
    "ARTIFACT_SHA256SUMS.txt": (505, "e19acb3c8e888dddb8ae296f05b7541ba9db47e017619aea0c9dd341f0c5b3f4"),
    "README.md": (9820, "95824df46b8650f74d003fef76877005b6f4e32ab99fafbfe4aa9749f5d3741e"),
    "free_order_oracle.py": (28692, "7329ee7cd6838e21db7ae81b9ee16843548c1f99297e6d007487c450ceaff820"),
    "protocol_lock.json": (8690, "71e45e62fa86238e89424f24ecf346cb2cd49715e569534c2ff817011015c66a"),
    "source_bindings.json": (2878, "3454b718a65efc02c32463f955c10ff393f4218fac04f358107960ff3735990d"),
    "test_source_only.py": (7549, "83424cff84195118975e31aabf725478cfab3c57b40f4f20fd702dd809c02288"),
    "verify_package.py": (8893, "5e1940b3c626706815c10febe6dc6e35a47d4e28548df9379915e5025c5c5499"),
}
SHA_LINE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)
        self.count += 1

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        self.require(actual == expected, f"{message}: {actual!r} != {expected!r}")

    def close(self, actual: float, expected: float, message: str, tolerance: float = 1e-15) -> None:
        self.require(abs(actual - expected) <= tolerance, f"{message}: {actual!r} != {expected!r}")


def regular_bytes(path: Path, checks: Checks, expected_bytes: int | None = None) -> bytes:
    checks.require(path.exists(), f"missing {path}")
    checks.require(path.is_file(), f"not a regular file {path}")
    checks.require(not path.is_symlink(), f"symlink forbidden {path}")
    raw = path.read_bytes()
    if expected_bytes is not None:
        checks.equal(len(raw), expected_bytes, f"byte count {path.name}")
    return raw


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path, checks: Checks) -> dict[str, Any]:
    value = json.loads(regular_bytes(path, checks).decode("utf-8"))
    checks.require(isinstance(value, dict), f"JSON root must be object: {path.name}")
    return value


def parse_audit_manifest(checks: Checks) -> dict[str, str]:
    rows: dict[str, str] = {}
    raw = regular_bytes(AUDIT_MANIFEST, checks)
    for line_number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        match = SHA_LINE.fullmatch(line)
        checks.require(match is not None, f"bad audit manifest line {line_number}")
        assert match is not None
        sha, name = match.groups()
        checks.require(name not in rows, f"duplicate audit artifact {name}")
        rows[name] = sha
    checks.equal(set(rows), AUDIT_FILES, "audit manifest closure")
    checks.equal({p.name for p in ROOT.iterdir()}, AUDIT_FILES | {AUDIT_MANIFEST.name}, "audit directory closure")
    for name, expected in sorted(rows.items()):
        checks.equal(digest(regular_bytes(ROOT / name, checks)), expected, f"audit SHA-256 {name}")
    return rows


def producer_closure(checks: Checks) -> None:
    checks.equal({p.name for p in PRODUCER.iterdir()}, set(PRODUCER_FILES), "producer closure")
    for name, (size, expected) in sorted(PRODUCER_FILES.items()):
        raw = regular_bytes(PRODUCER / name, checks, size)
        checks.equal(digest(raw), expected, f"producer SHA-256 {name}")
    manifest_rows: dict[str, str] = {}
    manifest_raw = regular_bytes(PRODUCER / "ARTIFACT_SHA256SUMS.txt", checks).decode("ascii")
    for line in manifest_raw.splitlines():
        match = SHA_LINE.fullmatch(line)
        checks.require(match is not None, "producer manifest syntax")
        assert match is not None
        sha, name = match.groups()
        checks.require(name not in manifest_rows, f"producer duplicate {name}")
        manifest_rows[name] = sha
    expected = {name: row[1] for name, row in PRODUCER_FILES.items() if name != "ARTIFACT_SHA256SUMS.txt"}
    checks.equal(manifest_rows, expected, "producer manifest contents")


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def gram(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[dot(left, right) for right in rows] for left in rows]


def role_marginals(panel: Sequence[Sequence[Sequence[float]]]) -> list[list[list[float]]]:
    return [gram([neuron[role] for neuron in panel]) for role in range(3)]


def full_pair_capture(target: Sequence[Sequence[float]], predecessor: Sequence[Sequence[float]]) -> float:
    predecessor_gram = gram(predecessor)
    if predecessor_gram != [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]:
        raise AssertionError("fixture predecessor Gram is not identity")
    cross = [[dot(t, p) for p in predecessor] for t in target]
    return math.fsum(value * value for row in cross for value in row)


def verify() -> dict[str, Any]:
    checks = Checks()
    audit_hashes = parse_audit_manifest(checks)
    producer_closure(checks)
    receipt = load_json(ROOT / "audit_receipt.json", checks)
    replay = load_json(ROOT / "replay_receipt.json", checks)
    lock = load_json(PRODUCER / "protocol_lock.json", checks)
    bindings = load_json(PRODUCER / "source_bindings.json", checks)
    oracle_raw = regular_bytes(PRODUCER / "free_order_oracle.py", checks)
    oracle_source = oracle_raw.decode("utf-8")
    oracle_tree = ast.parse(oracle_source, filename="free_order_oracle.py")

    checks.equal(receipt["schema"], "free-order-swiglu-path-v1-independent-source-audit-receipt-v1", "receipt schema")
    checks.equal(receipt["status"], "BLOCK_STAGE0_DOES_NOT_CONTAIN_CROSS_ROLE_3X3_PATH_FAMILY", "receipt status")
    checks.equal(receipt["artifact_set_status"], "IMMUTABLE_BLOCK_AUDIT_ARTIFACT_SET", "artifact status")
    checks.equal(receipt["verdict"], "BLOCK", "verdict")
    checks.require(receipt["producer_modified"] is False, "producer preservation")
    checks.equal(receipt["producer_closure"]["regular_files"], 7, "producer file count receipt")
    checks.equal(receipt["producer_closure"]["total_bytes"], sum(row[0] for row in PRODUCER_FILES.values()), "producer total bytes")
    for name, (size, sha) in PRODUCER_FILES.items():
        row = receipt["producer_closure"]["files"][name]
        checks.equal(row["bytes"], size, f"receipt bytes {name}")
        checks.equal(row["sha256"], sha, f"receipt SHA {name}")

    checks.equal(replay["status"], "REPLAYED_SOURCE_ONLY_BLOCK_CONFIRMED", "replay status")
    checks.equal(replay["producer_source_tests"]["tests_run"], 11, "replayed tests")
    checks.equal(replay["producer_source_tests"]["passed"], 11, "replayed passes")
    checks.equal(replay["producer_source_verifier"]["checks"], 128, "producer verifier checks")
    checks.equal(replay["producer_source_verifier"]["status"], "PASS", "producer verifier status")
    checks.equal(replay["producer_source_verifier"]["manifest_sha256"], PRODUCER_FILES["ARTIFACT_SHA256SUMS.txt"][1], "replayed producer manifest")
    checks.equal(replay["independent_audit_verifier"]["status"], "BLOCK_CONFIRMED", "independent verifier status")
    checks.equal(replay["independent_audit_verifier"]["checks"], 254, "independent verifier checks")
    checks.require(replay["remote_producer_hashes_match_local"] is True, "cross-platform hash equality")
    checks.require(replay["producer_tree_unchanged_after_replay"] is True, "post-replay preservation")

    checks.equal(lock["schema"], "free_order_swiglu_path_protocol_v1", "protocol schema")
    checks.equal(lock["status"], "SOURCE_ONLY_NOT_AUTHORIZED_FOR_QWEN_EXECUTION", "protocol status")
    checks.require(lock["execution_firewall"]["package_authorizes_Qwen_run"] is False, "no Qwen authority")
    checks.require(lock["execution_firewall"]["package_requires_independent_source_audit_before_any_Qwen_run"] is True, "audit prerequisite")
    checks.equal(lock["geometry"]["weights_per_expert"], 3 * 768 * 2048, "expert geometry")

    required_s = -0.5 * math.log2(0.8)
    checks.close(required_s, 0.16096404744368115, "required s")
    checks.close(float(lock["objective"]["required_s_bpw"]), required_s, "frozen required s")
    factorial_bits = (math.factorial(768) - 1).bit_length()
    checks.equal(factorial_bits, 6260, "ceil(log2(768!))")
    checks.equal((factorial_bits + 7) // 8, 783, "factoradic physical bytes")
    checks.equal(783 * 8, 6264, "factoradic physical bits")
    checks.close(6264 / (3 * 768 * 2048), 0.0013275146484375, "factoradic physical bpw")
    metric = lock["metric_compatibility"]
    checks.equal(metric["information_lower_bound_bits"], factorial_bits, "locked factoradic information bits")
    checks.equal(metric["physical_factoradic_bytes"], 783, "locked factoradic bytes")
    checks.require(lock["eligible_codec"]["permutation_is_inline_side_information"] is True, "inline factoradic charge")

    original = (10.0, 20.0, 30.0, 40.0)
    permutation = (2, 0, 3, 1)
    encoded = tuple(original[index] for index in permutation)
    wrong_mse = math.fsum((a - b) ** 2 for a, b in zip(encoded, original)) / len(original)
    restored = [0.0] * len(original)
    for encoded_index, original_index in enumerate(permutation):
        restored[original_index] = encoded[encoded_index]
    right_mse = math.fsum((a - b) ** 2 for a, b in zip(restored, original)) / len(original)
    checks.require(wrong_mse > 0.0, "canonical order violates original-coordinate MSE")
    checks.equal(right_mse, 0.0, "inverse scatter restores original-coordinate MSE")
    checks.require("scatter_to_original_coordinates_before_score" in oracle_source, "runtime scatter contract emitted")

    weights = 3 * 768 * 2048
    rate_receipt = {float(row["requested_rate_bpw"]): row for row in receipt["independent_recomputations"]["rate_rows"]}
    for locked in lock["rate_and_read"]["rows"]:
        rate = float(locked["requested_rate_bpw"])
        frame = math.floor(weights * rate / 8.0)
        actual = 8.0 * frame / weights
        cold_bytes = (math.ceil(frame / 4096) + 1) * 4096
        amplification = cold_bytes / frame
        checks.equal(int(locked["frame_bytes"]), frame, f"locked frame {rate}")
        checks.close(float(locked["actual_rate_bpw"]), actual, f"locked actual rate {rate}")
        checks.close(float(locked["cold_page_amplification"]), amplification, f"locked cold amplification {rate}")
        row = rate_receipt[rate]
        checks.equal(row["frame_bytes"], frame, f"receipt frame {rate}")
        checks.close(row["actual_rate_bpw"], actual, f"receipt actual rate {rate}")
        checks.equal(row["cold_page_bytes"], cold_bytes, f"receipt cold bytes {rate}")
        checks.close(row["cold_page_amplification"], amplification, f"receipt cold amplification {rate}")
        checks.require(amplification < 2.0, f"read bound {rate}")

    mode_bits = {
        "diag3_fp16_oracle_bridge": 767 * 3 * 16,
        "full3x3_fp16_oracle_bridge": 767 * 9 * 16,
        "diag3_fixed_nibble": math.ceil(767 * 3 * 4 / 8) * 8,
        "full3x3_fixed_nibble": math.ceil(767 * 9 * 4 / 8) * 8,
    }
    locked_modes = {row["name"]: row for row in lock["eligible_codec"]["coefficient_modes"]}
    receipt_names = {
        "diag3_fp16": "diag3_fp16_oracle_bridge",
        "full3x3_fp16": "full3x3_fp16_oracle_bridge",
        "diag3_fixed_nibble": "diag3_fixed_nibble",
        "full3x3_fixed_nibble": "full3x3_fixed_nibble",
    }
    receipt_modes = {row["mode"]: row for row in receipt["independent_recomputations"]["coefficient_and_side_ledgers"]}
    for receipt_name, locked_name in receipt_names.items():
        bits = mode_bits[locked_name]
        checks.equal(locked_modes[locked_name]["coefficient_bits"], bits, f"locked coefficient bits {locked_name}")
        checks.close(float(locked_modes[locked_name]["coefficient_bpw"]), bits / weights, f"locked coefficient bpw {locked_name}")
        row = receipt_modes[receipt_name]
        total_side = 64 * 8 + 783 * 8 + bits
        checks.equal(row["coefficient_bits"], bits, f"receipt coefficient bits {receipt_name}")
        checks.close(row["coefficient_bpw"], bits / weights, f"receipt coefficient bpw {receipt_name}")
        checks.equal(row["total_side_bits_including_header_and_permutation"], total_side, f"total side {receipt_name}")
        checks.close(row["side_bpw"], total_side / weights, f"side bpw {receipt_name}")
        checks.close(row["required_gross_s_bpw"], required_s + total_side / weights, f"gross target {receipt_name}")

    controls = lock["oracle"]["controls"]
    checks.equal(controls["replicates"], 8, "control count")
    checks.equal(len(controls["seeds"]), 8, "control seed count")
    checks.equal(len(set(controls["seeds"])), 8, "distinct controls")
    checks.require(controls["identical_optimization"] is True, "identical control optimization")
    checks.equal([(int(row["layer"]), int(row["expert"])) for row in bindings["experts"]], [(3, 57), (3, 121)], "fixed auxiliary split")
    checks.require(bindings["forbidden_runtime_inputs"]["pinned_panel_path_argument"] is False, "no pinned input")
    checks.require(bindings["forbidden_runtime_inputs"]["validation_path_argument"] is False, "no validation input")
    for fragment in (
        'source_mean = cp.mean(source, axis=2)',
        'source_gram = cp.einsum("nrd,nsd->nrs", source_centered, source_centered)',
        '/ (len(control_s) * (len(control_s) - 1))',
        'combined = math.hypot(control_mc_se, jackknife)',
        'optimistic = excess + 3.0 * combined',
        'specific_upper = qwen_s - control_mean + 3.0 * control_se',
    ):
        checks.require(fragment in oracle_source, f"statistical implementation fragment: {fragment}")

    top_imports: set[str] = set()
    for node in oracle_tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".")[0])
    checks.require(not ({"cupy", "numpy", "scipy", "torch"} & top_imports), "source-only import closure")
    literals = {node.value for node in ast.walk(oracle_tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    checks.require("--workspace-root" in literals and "--output" in literals, "fixed runtime CLI")
    checks.require(not ({"--plan", "--manifest", "--source", "--target"} & literals), "no alternate runtime input")

    checks.require('matrix = expert[:, role, :]' in oracle_source, "stage0 separates roles")
    checks.require('covariance = (matrix @ matrix.T) / float(COLS)' in oracle_source, "stage0 marginal covariance")
    checks.require('for target_role in range(ROLES):' in oracle_source, "stage1 target roles")
    checks.require('for predecessor_role in range(ROLES):' in oracle_source, "stage1 predecessor roles")
    checks.require('full = cp.einsum("ijab,jbc,ijac->ij", cross, inverse, cross)' in oracle_source, "stage1 full cross-role capture")

    basis = [[1.0 if column == row else 0.0 for column in range(6)] for row in range(6)]
    predecessor = [basis[0], basis[1], basis[2]]
    cross_target = [basis[1], basis[3], basis[4]]
    null_target = [basis[3], basis[4], basis[5]]
    cross_panel = [predecessor, cross_target]
    null_panel = [predecessor, null_target]
    checks.equal(role_marginals(cross_panel), role_marginals(null_panel), "same three stage0 marginal covariances")
    checks.equal(gram(cross_target), gram(null_target), "same target within-neuron 3x3 Gram")
    checks.equal(full_pair_capture(cross_target, predecessor), 1.0, "cross-role capture exists")
    checks.equal(full_pair_capture(null_target, predecessor), 0.0, "null capture absent")

    neurons = 768
    checks.require(2048 - 1 >= neurons, "zero-mean orthonormal construction fits geometry")
    for rate in (2.15, 2.3, 2.5):
        normalized_distortion = 2.0 ** (-2.0 * rate)
        f_value = normalized_distortion * 2.0 ** (2.0 * rate)
        dense_s = -0.5 * math.log2(f_value)
        checks.close(f_value, 1.0, f"flat separate-role KLT F {rate}")
        checks.close(dense_s, 0.0, f"flat separate-role KLT s {rate}")
    total_energy = 3 * neurons
    legal_capture = 2 * (neurons - 1)
    residual = 1.0 - legal_capture / total_energy
    path_s = -0.5 * math.log2(residual)
    checks.equal(total_energy, 2304, "counterexample energy")
    checks.equal(legal_capture, 1534, "counterexample capture")
    checks.close(residual, 770 / 2304, "counterexample residual")
    checks.close(path_s, 0.7906051829300244, "counterexample path s")
    checks.require(path_s > required_s, "counterexample exceeds target")
    full_fp16_side = (64 * 8 + 783 * 8 + 767 * 9 * 16) / weights
    checks.require(path_s > required_s + full_fp16_side, "counterexample exceeds full3x3 FP16 side-adjusted target")
    counterexample = receipt["containment_counterexample"]
    checks.equal(counterexample["total_capture"], legal_capture, "receipt counterexample capture")
    checks.close(counterexample["residual_ratio"], residual, "receipt counterexample residual")
    checks.close(counterexample["stage1_s_bpw"], path_s, "receipt counterexample s")
    checks.equal(receipt["blocking_decision"]["payload_execution"], "BLOCKED", "payload block")
    checks.equal(receipt["blocking_decision"]["reason_code"], "STAGE0_SEPARATE_ROLE_KLT_NONCONTAINMENT", "block reason")
    checks.require("joint 2304-axis" in receipt["blocking_decision"]["minimal_distinct_successor_repair"], "successor repair")

    zero = receipt["zero_access_ledger"]
    for key in (
        "qwen_or_model_payload_files_opened",
        "qwen_or_model_payload_bytes_read",
        "binding_relative_paths_followed",
        "pinned_panel_files_opened",
        "validation_files_opened",
        "cupy_imports",
        "cuda_api_calls",
        "gpu_device_calls",
        "external_data_fetches",
        "producer_files_modified",
    ):
        checks.equal(zero[key], 0, f"zero-access ledger {key}")

    return {
        "status": "BLOCK_CONFIRMED",
        "verdict": "BLOCK",
        "checks": checks.count,
        "audit_manifest_sha256": digest(AUDIT_MANIFEST.read_bytes()),
        "audit_artifacts": audit_hashes,
        "producer_manifest_sha256": PRODUCER_FILES["ARTIFACT_SHA256SUMS.txt"][1],
        "producer_tests": 11,
        "producer_verifier_checks": 128,
        "counterexample_stage1_s_bpw": path_s,
        "required_s_bpw": required_s,
        "zero_payload_access": True,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
