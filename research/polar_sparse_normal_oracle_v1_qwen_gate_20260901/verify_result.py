#!/usr/bin/env python3
"""Standard-library verifier for the authenticated PSNO-v1 Qwen gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROWS = 768
COLS = 2048
MATRICES = 18
PANEL_VALUES = MATRICES * ROWS * COLS
UNIQUE = ROWS * (ROWS + 1) // 2
RATE = 2.5
TARGET_F = 0.8
BASE_SIDE_BPW = 0.0011701230649594908
ALLOWANCE = 1e-9
EXPECTED = {
    "parent_script": "04d033319b4bbab037b48355e5f296274ae77b1c787ddcd2508e9b58948d265e",
    "parent_result": "e4fecac5f676d84739972bbf0e04467027aeae1356e62e1dc3cd2b84bff67026",
    "composite_result": "565e1eb2122f2e476c5bd81e4205eeb3e4cede6e6a51149e95944355199eb41c",
    "source_manifest": "23a7566cf9ead5191c778a9dda30e880646a32d81357ff940182dc74e11bfe99",
    "source_protocol": "d519f217263823085e39b3b291863f123603fcb540ba912ffe25c0d13bfacf43",
    "cupy_primitives": "80b9608aaadaddc34c98fa70485b3e1de25074bdd423fdb211fa4d9d8d22bbee",
    "source_lock": "bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def verify_seal(report: dict[str, Any], field: str = "result_lock_sha256") -> str:
    clean = dict(report)
    claimed = clean.pop(field, None)
    observed = hashlib.sha256(canonical(clean)).hexdigest()
    if claimed != observed:
        raise AssertionError((field, claimed, observed))
    return observed


def write_sealed(path: Path, report: dict[str, Any]) -> None:
    clean = dict(report)
    clean.pop("receipt_lock_sha256", None)
    clean["receipt_lock_sha256"] = hashlib.sha256(canonical(clean)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def exact_support(n: int, k: int) -> int:
    if not 0 <= k <= n:
        raise AssertionError((n, k))
    return (math.comb(n, k) - 1).bit_length()


def group_count(family: str) -> int:
    if family.startswith("upper_triangular_tiles_"):
        block = int(family.split("_")[3])
        tiles = (ROWS + block - 1) // block
        return tiles * (tiles + 1) // 2
    if family.startswith("diagonal_offset_segments_"):
        segment = int(family.split("_")[3])
        return sum((ROWS - offset + segment - 1) // segment for offset in range(ROWS))
    raise AssertionError(("unknown family", family))


def phi(dimension: float, energy: float, multiplier: float) -> tuple[float, float]:
    if energy <= 0.0:
        return 0.0, 0.0
    critical = 2.0 * math.log(2.0) * energy / dimension
    if multiplier >= critical:
        return energy, 0.0
    rate = dimension / (2.0 * math.log(2.0)) * math.log(critical / multiplier)
    return multiplier * dimension / (2.0 * math.log(2.0)) + multiplier * rate, rate


def verify_scan(
    scan: dict[str, Any], records: list[dict[str, Any]], total_energy: float, checks: list[str]
) -> None:
    if not scan["continuous_selected_values_exact"] or not scan["support_bits_exact"]:
        raise AssertionError((scan["family"], "ledger weakening"))
    best = scan["best_evaluated_dual"]
    multiplier = float(best["multiplier"])
    value_bits = int(scan["value_bits_per_selected_coordinate"])
    value = -multiplier * (RATE - BASE_SIDE_BPW)
    payload = 0.0
    side = 0.0
    for record in records:
        objective, rate = phi(
            int(record["model_dof"]) / PANEL_VALUES,
            float(record["model_energy"]) / total_energy,
            multiplier,
        )
        value += objective
        payload += rate

    selected = best["selected_options"]
    if len(selected) != MATRICES:
        raise AssertionError((scan["family"], "selection count"))
    n = UNIQUE if scan["family"] == "arbitrary_symmetric_coordinate_topk" else group_count(scan["family"])
    for ordinal, (option, record) in enumerate(zip(selected, records, strict=True)):
        if int(option["matrix_ordinal"]) != ordinal:
            raise AssertionError((scan["family"], ordinal, "order"))
        k = int(option["k"] if "k" in option else option["selected_groups"])
        support = exact_support(n, k)
        if support != int(option["support_bits_exact"]):
            raise AssertionError((scan["family"], ordinal, "support", support))
        symbols = int(option["value_symbols"])
        removed = int(option["removed_dof"])
        if symbols < 0 or not 0 <= removed < int(record["normal_dof"]):
            raise AssertionError((scan["family"], ordinal, "symbols/dof"))
        capture = float(option["captured_normal_energy_fraction"])
        if not -2e-13 <= capture <= 1.0 + 2e-13:
            raise AssertionError((scan["family"], ordinal, "capture"))
        residual = max(0.0, float(record["normal_energy"]) * (1.0 - capture)) / total_energy
        dimension = max(1, int(record["normal_dof"]) - removed) / PANEL_VALUES
        objective, rate = phi(dimension, residual, multiplier)
        option_side = (support + value_bits * symbols) / PANEL_VALUES
        if not math.isclose(option_side, float(option["side_rate_bpw"]), rel_tol=0.0, abs_tol=2e-16):
            raise AssertionError((scan["family"], ordinal, "side arithmetic"))
        value += objective + multiplier * option_side
        payload += rate
        side += option_side
    if not math.isclose(value, float(best["raw_dual_distortion"]), rel_tol=0.0, abs_tol=2e-10):
        raise AssertionError((scan["family"], "dual replay", value, best["raw_dual_distortion"]))
    conservative = max(0.0, value - ALLOWANCE)
    f_value = conservative * 32.0
    if not math.isclose(
        f_value, float(best["dual_F_lower_bound"]), rel_tol=0.0, abs_tol=7e-9
    ):
        raise AssertionError((scan["family"], "F replay"))
    expected_kill = f_value > TARGET_F
    if expected_kill != bool(scan["hard_kill"]) or expected_kill != bool(best["hard_kill_at_multiplier"]):
        raise AssertionError((scan["family"], "kill direction"))
    implied = BASE_SIDE_BPW + side + payload
    if not math.isclose(
        implied, float(best["implied_total_rate_bpw"]), rel_tol=0.0, abs_tol=2e-10
    ):
        raise AssertionError((scan["family"], "implied rate replay"))
    checks.extend(
        (
            f"dual_replay:{scan['family']}:b{value_bits}",
            f"exact_support:{scan['family']}:b{value_bits}",
            f"kill_direction:{scan['family']}:b{value_bits}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    source_root = args.source_root.resolve()
    result_path = args.result.resolve()
    package = Path(__file__).resolve().parent
    checks: list[str] = []

    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_lock = verify_seal(result)
    checks.append("result_canonical_lock")
    if result["schema"] != "qwen_polar_sparse_normal_oracle_gate_v1":
        raise AssertionError("schema")
    scope = result["scope"]
    if (
        int(scope["matrices"]) != MATRICES
        or int(scope["panel_values"]) != PANEL_VALUES
        or float(scope["physical_rate_bpw"]) != RATE
        or float(scope["target_F"]) != TARGET_F
        or bool(scope["fresh_validation_access"])
        or bool(scope["router_access"])
    ):
        raise AssertionError("scope")
    checks.append("frozen_no_fresh_scope")

    paths = {
        "parent_script": repo_root / "research/polar_normal_predictor/polar_normal_predictor.py",
        "parent_result": repo_root / "research/polar_normal_predictor/result.json",
        "composite_result": repo_root / "research/composite_superoracle/result.json",
        "source_manifest": repo_root / "research/polar_sparse_normal_oracle_v1/PACKAGE_MANIFEST.json",
        "source_protocol": repo_root / "research/polar_sparse_normal_oracle_v1/protocol.json",
        "cupy_primitives": repo_root / "research/polar_sparse_normal_oracle_v1/cupy_gate_proposal.py",
        "source_lock": source_root / "blind_protocol_v2/unblinded/source_hashes.lock.json",
    }
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != EXPECTED[label]:
            raise AssertionError((label, "immutable hash"))
        checks.append(f"immutable:{label}")

    parent_result = json.loads(paths["parent_result"].read_text(encoding="utf-8"))
    verify_seal(parent_result)
    prior_records = parent_result["source_normal_records"]
    records = result["normal_records"]
    if len(records) != MATRICES or records != prior_records:
        raise AssertionError("prior normal record byte-science preservation")
    checks.append("exact_prior_normal_records")

    source_lock = json.loads(paths["source_lock"].read_text(encoding="utf-8"))
    receipts = result["audit"]["source_receipts"]
    if len(receipts) != MATRICES or len(source_lock["matrices"]) != MATRICES:
        raise AssertionError("source plan count")
    for ordinal, (receipt, row) in enumerate(zip(receipts, source_lock["matrices"], strict=True)):
        if int(receipt["matrix_ordinal"]) != ordinal or int(row["matrix_ordinal"]) != ordinal:
            raise AssertionError((ordinal, "source order"))
        source_path = paths["source_lock"].parent / row["output_relpath"]
        observed = sha256_file(source_path)
        if observed != row["source_bf16_sha256"] or observed != receipt["observed_sha256"]:
            raise AssertionError((ordinal, "source bytes"))
        checks.append(f"source:{ordinal:02d}")

    total_energy = math.fsum(float(record["source_energy"]) for record in records)
    coordinate = result["duals"]["coordinate_value_ledgers"]
    actual = result["duals"]["fixed_group_energy_prefix_onebit"]
    gross = result["duals"]["broad_group_containing_relaxation_onebit"]
    if len(coordinate) != 3 or len(actual) != 8 or len(gross) != 8:
        raise AssertionError("dual family count")
    for scan in coordinate + actual + gross:
        verify_scan(scan, records, total_energy, checks)

    onebit = next(scan for scan in coordinate if int(scan["value_bits_per_selected_coordinate"]) == 1)
    fixed_killed = all(bool(scan["hard_kill"]) for scan in actual)
    gross_killed = all(bool(scan["hard_kill"]) for scan in gross)
    decision = result["decision"]
    if bool(decision["coordinate_onebit_hard_kill"]) != bool(onebit["hard_kill"]):
        raise AssertionError("coordinate decision")
    if bool(decision["all_fixed_group_energy_prefixes_hard_killed"]) != fixed_killed:
        raise AssertionError("fixed group decision")
    if bool(decision["all_broad_group_containing_relaxations_hard_killed"]) != gross_killed:
        raise AssertionError("gross group decision")
    if bool(result["heavy_tail_matched_gaussian_controls"]["decision_eligible"]):
        raise AssertionError("control entered decision")
    if float(result["ledger"]["read_amplification_logical"]) != 1.0:
        raise AssertionError("read contract")
    checks.extend(("decision_replay", "controls_diagnostic_only", "logical_read_1x"))

    receipt = {
        "schema": "qwen_polar_sparse_normal_oracle_gate_verification_receipt_v1",
        "verdict": "PASS",
        "result_path": str(result_path),
        "result_file_sha256": sha256_file(result_path),
        "result_lock_sha256": result_lock,
        "checks": checks,
        "check_count": len(checks),
        "verifier_path": str(Path(__file__).resolve()),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "source_payloads_hashed": MATRICES,
        "fresh_validation_access": False,
        "gpu_execution": False,
    }
    if args.receipt is not None:
        write_sealed(args.receipt.resolve(), receipt)
    print("PASS_QWEN_POLAR_SPARSE_NORMAL_GATE")
    print(f"checks {len(checks)}")
    print(f"result_lock_sha256 {result_lock}")


if __name__ == "__main__":
    main()
