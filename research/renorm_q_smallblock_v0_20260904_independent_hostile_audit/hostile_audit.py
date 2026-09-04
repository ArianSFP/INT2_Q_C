"""Independent hostile audit of the sealed RENORM-Q source-only package.

This script never modifies the target and has no model, GPU, network, or
payload access.  It authenticates the pinned closure, repeats its source tests,
checks the DP against exhaustive search, and proves expected adversarial flaws.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np


AUDIT = Path(__file__).resolve().parent
TARGET = AUDIT.parent / "renorm_q_smallblock_v0_20260904"
EXPECTED_MANIFEST_SHA256 = "340ba1f1c435be9cbfc58c75607cc5c5e07e6bb10692265b5938ebf530d926b9"
EXPECTED_SOURCE_ROOT_SHA256 = "8c1682ea514e067c4ba10b1e010abf1766cb842ea525233f7eb854896da6cac4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def authenticate_target() -> dict:
    manifest_path = TARGET / "SOURCE_MANIFEST.json"
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("target manifest pin mismatch")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    canonical = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if raw != canonical:
        raise RuntimeError("target manifest is not canonical JSON")
    rows = manifest["files"]
    names = [row["name"] for row in rows]
    if names != sorted(names) or len(names) != len(set(names)):
        raise RuntimeError("target member ordering")
    if sorted(path.name for path in TARGET.iterdir()) != sorted(names + ["SOURCE_MANIFEST.json"]):
        raise RuntimeError("target closure mismatch")
    canonical_rows = []
    for row in rows:
        path = TARGET / row["name"]
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError(f"non-regular target member: {row['name']}")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"target member mismatch: {row['name']}")
        canonical_rows.append(row)
    root = hashlib.sha256(json.dumps(canonical_rows, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
    if root != EXPECTED_SOURCE_ROOT_SHA256 or root != manifest["source_root_sha256"]:
        raise RuntimeError("target source-root mismatch")
    return {"manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "source_root_sha256": root, "members": len(rows)}


def load_target():
    path = TARGET / "renorm_q_oracle.py"
    spec = importlib.util.spec_from_file_location("renorm_q_hostile_target", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def repeat_target_verifier() -> dict:
    process = subprocess.run(
        [sys.executable, "-I", "-B", str(TARGET / "verify_source.py"),
         "--package", str(TARGET), "--manifest-sha256", EXPECTED_MANIFEST_SHA256],
        check=True, capture_output=True, text=True,
    )
    return json.loads(process.stdout)


def repeat_target_tests() -> dict:
    process = subprocess.run(
        [sys.executable, "-I", "-B", str(TARGET / "test_source.py")],
        check=True, capture_output=True, text=True,
    )
    combined = process.stdout + process.stderr
    if "Ran 8 tests" not in combined or "OK" not in combined:
        raise RuntimeError("target test summary mismatch")
    return {"tests_run": 8, "status": "PASS"}


def normalized_nll(probabilities: np.ndarray) -> np.ndarray:
    return -np.log2(np.asarray(probabilities, dtype=np.float64))


def randomized_dp_exactness(rq) -> dict:
    """Test depth-two backtracking and accounting, beyond the target KAT."""
    spec = rq.MapSpec("identity", np.asarray([0, 1], dtype=np.uint8), 2, 0,
                      "audit identity")
    max_objective_error = 0.0
    max_distortion_error = 0.0
    max_bits_error = 0.0
    trials = 12
    for seed in range(trials):
        rng = np.random.default_rng(0x52474D + seed)
        costs = rng.uniform(0.001, 2.0, size=(4, 1, 2)).astype(np.float64)
        root_p = rng.dirichlet(np.ones(2))
        root = normalized_nll(root_p)
        transition = np.empty((2, 2, 2), dtype=np.float64)
        for level in range(2):
            for parent in range(2):
                transition[level, parent] = normalized_nll(rng.dirichlet(np.ones(2)))
        leaf = rq.uniform_fiber_leaf_nll(spec, 2)
        dp = rq.exact_tree_min_sum(costs, 2, spec, root, transition, leaf, 0.371)
        brute = rq.brute_force_tree(costs, 2, spec, root, transition, leaf, 0.371)
        max_objective_error = max(max_objective_error, abs(dp.objective - brute.objective))
        max_distortion_error = max(max_distortion_error, abs(dp.distortion - brute.distortion))
        max_bits_error = max(max_bits_error, abs(dp.modeled_bits - brute.modeled_bits))
        if not np.array_equal(dp.tuple_ids, brute.tuple_ids):
            raise RuntimeError(f"DP assignment mismatch at seed {seed}")
    return {
        "trials": trials,
        "max_objective_abs_error": max_objective_error,
        "max_distortion_abs_error": max_distortion_error,
        "max_modeled_bits_abs_error": max_bits_error,
        "status": "PASS_EXACT_FOR_VALID_NORMALIZED_TEST_MODELS",
    }


def invalid_probability_counterexample(rq) -> dict:
    """Show that arbitrary nonnegative NLL arrays are not necessarily code lengths."""
    spec = rq.MapSpec("identity", np.asarray([0, 1], dtype=np.uint8), 2, 0,
                      "audit identity")
    costs = np.zeros((4, 1, 2), dtype=np.float64)
    root = np.zeros(2, dtype=np.float64)
    transition = np.zeros((2, 2, 2), dtype=np.float64)
    leaf = rq.uniform_fiber_leaf_nll(spec, 2)
    result = rq.exact_tree_min_sum(costs, 2, spec, root, transition, leaf, 1.0)
    root_kraft_sum = float(np.sum(np.exp2(-root)))
    transition_row_kraft_sums = np.sum(np.exp2(-transition), axis=2).tolist()
    # Identity fibres cost zero, so all 2^4 leaf strings receive modeled length zero.
    sequence_kraft_sum = 16.0
    if result.modeled_bits != 0.0 or root_kraft_sum <= 1.0 or sequence_kraft_sum <= 1.0:
        raise RuntimeError("invalid-NLL counterexample no longer reproduces")
    return {
        "accepted_by_target": True,
        "reported_modeled_bits_for_selected_four_symbol_path": result.modeled_bits,
        "root_kraft_sum": root_kraft_sum,
        "transition_row_kraft_sums": transition_row_kraft_sums,
        "full_sequence_kraft_sum": sequence_kraft_sum,
        "severity": "BLOCKER",
        "required_fix": (
            "validate normalized/subnormalized root, every transition row, and every "
            "leaf conditional, then test operational integer codelength"),
    }


def custom_map_bypass(rq) -> dict:
    """Prove callers can supply a source-derived truth table with descriptor zero."""
    blocks = rq.tuple_table(4, 2)
    ids = rq.tuple_ids(blocks, 2)
    rng = np.random.default_rng(0x4D4150)
    truth = np.zeros(16, dtype=np.uint8)
    truth[rng.choice(16, size=8, replace=False)] = 1
    environment = truth[ids, None]
    custom = rq.MapSpec("source_derived_zero_cost", truth, 2, 0,
                        "hostile source-derived lookup")
    custom_row = rq.collective_variable_census(
        blocks, environment, 2, beta=0.0, map_bank=[custom], charge_descriptor=True)[0]
    default_best = rq.collective_variable_census(
        blocks, environment, 2, beta=0.0, charge_descriptor=True)[0]
    if abs(custom_row["mutual_information_bits_per_cell"] - 1.0) > 1e-12:
        raise RuntimeError("custom-map exploit did not recover one bit")
    return {
        "accepted_by_target": True,
        "custom_map_mi_bits_per_cell": custom_row["mutual_information_bits_per_cell"],
        "custom_map_descriptor_bpw": custom_row["descriptor_bpw"],
        "custom_map_rsmi_score_bpw": custom_row["rsmi_score_bpw"],
        "best_frozen_bank_rsmi_score_bpw": default_best["rsmi_score_bpw"],
        "severity": "BLOCKER_FOR_CALLER_SUPPLIED_BANKS",
        "required_fix": (
            "promotion API must authenticate the frozen bank or charge the complete map truth "
            "table plus selection/search; self-declared descriptor_bits is not authoritative"),
    }


def binary_bank_totality_defect(rq) -> dict:
    bank = rq.public_map_bank(4, 2)
    defects = []
    for spec in bank:
        observed = np.unique(spec.outputs)
        if len(observed) != spec.cardinality:
            error = None
            try:
                rq.uniform_fiber_leaf_nll(spec, 16)
            except Exception as exc:  # exact hostile observation
                error = f"{type(exc).__name__}: {exc}"
            defects.append({"map": spec.name, "declared_cardinality": spec.cardinality,
                            "observed_states": observed.tolist(), "leaf_model_error": error})
    if not defects:
        raise RuntimeError("expected binary-bank totality defect did not reproduce")
    return {
        "defects": defects,
        "severity": "CORRECTNESS_BLOCKER_FOR_BINARY_TREE_BACKEND",
        "required_fix": "derive cardinality/remap outputs per alphabet or exclude empty-state maps",
    }


def decision_rule_counterexamples(rq) -> dict:
    cases = [
        {"qwen": 0.04, "control": 0.0, "lcb": 0.01,
         "actual": rq.kill_decision(0.04, 0.0, 0.01), "required": "HARD_KILL"},
        {"qwen": 0.06, "control": 0.0, "lcb": 0.01,
         "actual": rq.kill_decision(0.06, 0.0, 0.01), "required": "HARD_KILL"},
    ]
    if not any(case["actual"] != case["required"] for case in cases):
        raise RuntimeError("decision-rule mismatch no longer reproduces")
    return {
        "cases": cases,
        "severity": "PROMOTION_GATE_BLOCKER",
        "reason": (
            "README freezes hard kill when the control-corrected lower-confidence gain is "
            "below 0.03 bpw, but code compares the LCB only with zero"),
    }


def read_and_authority_audit(rq) -> dict:
    formula = rq.logical_common_private_read_amplification(16, 1.0, 3.0)
    expected = 4.0 / 3.0625
    if abs(formula - expected) > 1e-15:
        raise RuntimeError("read formula mismatch")
    lock = json.loads((TARGET / "DESIGN_LOCK.json").read_text(encoding="utf-8"))
    disabled = (TARGET / "RUN_DISABLED.txt").read_text(encoding="utf-8")
    false_flags = {key: lock[key] for key in
                   ("qwen_authority", "gpu_authority", "network_authority",
                    "payload_authority", "deployment_authority", "finite_codec_claim")}
    if any(false_flags.values()) or "NO PAYLOAD EXECUTION AUTHORITY" not in disabled:
        raise RuntimeError("authority boundary")
    return {
        "common_private_formula_16_experts_C1_P3": formula,
        "formula_independently_expected": expected,
        "formula_status": "PASS_IDEAL_LOGICAL_ONLY",
        "physical_page_ledger_present": False,
        "expert_local_packet_present": False,
        "assessment": (
            "not a source-package blocker because limitations are disclosed; no finite/read "
            "claim may be promoted until model tables, headers, pages, and packet bytes exist"),
        "authority_flags": false_flags,
        "authority_status": "PASS_FAIL_CLOSED_SOURCE_ONLY",
    }


def main() -> int:
    closure = authenticate_target()
    rq = load_target()
    target_verifier = repeat_target_verifier()
    report = {
        "schema": "renorm_q_smallblock_independent_hostile_audit.v0",
        "verdict": "BLOCKED_FROM_PAYLOAD_CAPABILITY_PENDING_SOURCE_FIXES",
        "target": str(TARGET.name),
        "target_closure": closure,
        "target_verifier": target_verifier,
        "target_tests": repeat_target_tests(),
        "exact_min_sum": randomized_dp_exactness(rq),
        "blockers": {
            "invalid_probability_nll": invalid_probability_counterexample(rq),
            "caller_supplied_map_bypass": custom_map_bypass(rq),
            "binary_map_bank_totality": binary_bank_totality_defect(rq),
            "frozen_decision_rule_mismatch": decision_rule_counterexamples(rq),
        },
        "read_and_authority": read_and_authority_audit(rq),
        "rsmi_normalization_assessment": {
            "mi_and_entropy_divisor": "cell_sites",
            "descriptor_divisor": "number_of_sampled_cell_sites",
            "default_bank_selector_bits": 4,
            "status": "ARITHMETIC_NORMALIZATION_CORRECT_FOR_DISJOINT_CELLS",
            "limitations": (
                "overlapping cells, tiling/traversal/beta search, table parameters, and repeated "
                "model selection are not charged; RSMI is in-sample and diagnostic only"),
        },
        "iid_xor_fixture_assessment": {
            "status": "PASS_MECHANISM_ONLY",
            "xor_mi_bits_per_cell": target_verifier["xor_collective_mi_bits_per_cell"],
            "balanced_iid_test_present": True,
            "limitation": "fixtures do not validate held-out Qwen structure or operational rate",
        },
        "required_repairs": [
            "Kraft-normalize/authenticate every NLL table and add invalid-table rejection KATs.",
            "Disallow unauthenticated custom maps or charge their complete serialized description.",
            "Remove/remap empty-state binary maps before using the tree backend.",
            "Apply the 0.03-bpw threshold to the control-corrected lower confidence bound.",
            "After repair, reseal under new hashes and obtain another independent audit.",
        ],
        "qwen_payload_opened": False,
        "gpu_accessed": False,
        "network_accessed": False,
        "target_modified": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
