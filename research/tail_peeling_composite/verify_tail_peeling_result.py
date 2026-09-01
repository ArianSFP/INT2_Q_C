#!/usr/bin/env python3
"""Independent structural verifier for the sparse-tail oracle result."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROWS = 768
COLS = 2048
MATRICES = 18
EXPERTS = 6
VALUES_PER_MATRIX = ROWS * COLS
PANEL_VALUES = MATRICES * VALUES_PER_MATRIX
TARGET_F = 0.8


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def require_close(left: float, right: float, label: str, tolerance: float = 2e-11) -> None:
    if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: {left!r} != {right!r}")


@functools.lru_cache(maxsize=None)
def ceil_log2_binomial(n: int, k: int) -> int:
    count = math.comb(n, min(k, n - k))
    return 0 if count <= 1 else (count - 1).bit_length()


def verify_seal(report: dict[str, Any]) -> str:
    declared = str(report["result_lock_sha256"])
    payload = dict(report)
    payload.pop("result_lock_sha256")
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if observed != declared:
        raise AssertionError("result lock mismatch")
    return observed


def verify_candidate_ledgers(report: dict[str, Any], tail_counts: list[int]) -> None:
    matrices = report["qwen_matrix_candidates"]
    if len(matrices) != MATRICES:
        raise AssertionError("Qwen candidate ledger does not cover 18 matrices")
    for ordinal, matrix in enumerate(matrices):
        if int(matrix["matrix_ordinal"]) != ordinal:
            raise AssertionError("candidate matrix order changed")
        energy = float(matrix["energy"])
        candidates = matrix["candidates"]
        if len(candidates) != len(tail_counts):
            raise AssertionError("tail grid length changed")
        prior_tail = -1.0
        for index, (candidate, expected_k) in enumerate(zip(candidates, tail_counts, strict=True)):
            if int(candidate["candidate_index"]) != index or int(candidate["k"]) != expected_k:
                raise AssertionError("tail grid mismatch")
            mask_bits = ceil_log2_binomial(VALUES_PER_MATRIX, expected_k)
            if int(candidate["mask_code"]["bits"]) != mask_bits:
                raise AssertionError("enumerative mask length mismatch")
            tail = float(candidate["tail_energy"])
            residual = float(candidate["residual_energy"])
            if tail + 1e-12 < prior_tail:
                raise AssertionError("tail energies are not nested")
            require_close(tail + residual, energy, "matrix energy closure", 5e-12)
            value = candidate["value_code"]
            if int(value["total_bits"]) > 16 * expected_k:
                raise AssertionError("selected value mode is longer than literal BF16")
            if int(candidate["variable_side_bits"]) != mask_bits + int(value["total_bits"]):
                raise AssertionError("tail side length mismatch")
            prior_tail = tail


def verify_score(row: dict[str, Any], matrix_ledgers: list[dict[str, Any]]) -> None:
    if not row.get("valid", False):
        raise AssertionError("published selected row is invalid")
    physical_bits = int(row["physical_bits"])
    if physical_bits != int(row["capacity_bytes"]) * 8:
        raise AssertionError("capacity is not byte exact")
    require_close(
        row["physical_rate_bpw"], physical_bits / PANEL_VALUES, "physical rate closure"
    )
    if int(row["side_bits"]) + int(row["payload_bits"]) != physical_bits:
        raise AssertionError("side/payload rate does not close")
    if float(row["physical_rate_bpw"]) > 2.5 + 1e-15:
        raise AssertionError("physical rate exceeds cap")
    mse = float(row["ideal_relative_mse"])
    f_value = mse * 2.0 ** (2.0 * float(row["physical_rate_bpw"]))
    require_close(row["F"], f_value, "F identity")
    require_close(row["s_bpw"], -0.5 * math.log2(f_value), "s identity")
    require_close(
        row["target_mse"],
        TARGET_F * 2.0 ** (-2.0 * float(row["physical_rate_bpw"])),
        "target identity",
    )
    if bool(row["passes_F_le_0p8"]) != (f_value <= TARGET_F):
        raise AssertionError("target decision mismatch")
    if len(matrix_ledgers) != MATRICES or len(row["choices"]) != MATRICES:
        raise AssertionError("selected row does not cover all matrices")
    expected_variable = []
    expected_peeled = 0
    expected_tail_energy = 0.0
    total_energy = sum(float(matrix["energy"]) for matrix in matrix_ledgers)
    for ordinal, choice in enumerate(row["choices"]):
        candidate = matrix_ledgers[ordinal]["candidates"][int(choice)]
        expected_peeled += int(candidate["k"])
        expected_tail_energy += float(candidate["tail_energy"])
        if row["side_mode"] == "charged":
            bits = int(candidate["variable_side_bits"])
        elif row["side_mode"] == "free_values":
            bits = int(candidate["mask_code"]["bits"])
        elif row["side_mode"] == "free_mask_values":
            bits = 0
        else:
            raise AssertionError("unknown side mode")
        expected_variable.append(bits)
    if expected_variable != [int(value) for value in row["tail_variable_bits_by_matrix"]]:
        raise AssertionError("selected tail side vector mismatch")
    if int(row["tail_variable_bits"]) != sum(expected_variable):
        raise AssertionError("selected tail side sum mismatch")
    if int(row["peeled_weights"]) != expected_peeled:
        raise AssertionError("peeled-weight count mismatch")
    require_close(
        row["peeled_energy_fraction"],
        expected_tail_energy / total_energy,
        "peeled energy fraction",
        5e-12,
    )
    fixed = dict(row["fixed_side_bits"])
    fixed_total = int(fixed.pop("total"))
    if sum(int(value) for value in fixed.values()) != fixed_total:
        raise AssertionError("fixed side ledger does not close")
    if int(fixed["residual_directories"]) != int(row["component_count"]) * 64:
        raise AssertionError("residual directory count is not component exact")
    expected_angle_bits = (
        sum(int(value) for value in row["support_xklt_angle_counts_by_expert"]) * 16
        if bool(row["basis_charged"])
        else 0
    )
    if int(row["support_xklt_angle_bits"]) != expected_angle_bits:
        raise AssertionError("XKLT angle ledger mismatch")
    if int(row["side_bits"]) != fixed_total + sum(expected_variable) + expected_angle_bits:
        raise AssertionError("complete side ledger mismatch")
    allocations = row.get("allocations")
    if allocations is None:
        raise AssertionError("selected row omits integer allocations")
    if sum(int(item["payload_bits"]) for item in allocations) != int(row["payload_bits"]):
        raise AssertionError("allocation payload does not close")
    distortion = sum(float(item["distortion_sse"]) for item in allocations)
    require_close(distortion, row["distortion_sse"], "distortion allocation closure", 5e-12)
    for item in allocations:
        expected_error = float(item["energy"]) * math.exp(
            -2.0
            * int(item["payload_bits"])
            / int(item["dimension"])
            * math.log(2.0)
        )
        require_close(item["distortion_sse"], expected_error, "component RD identity", 5e-12)
    ledger = row["read_ledger"]
    if int(ledger["bit_closure"]) != physical_bits:
        raise AssertionError("expert frame ledger does not close")
    reference = float(ledger["reference_one_sixth_container_bytes"])
    maximum = 0.0
    for expert, item in enumerate(ledger["experts"]):
        if int(item["expert_ordinal"]) != expert:
            raise AssertionError("expert read order changed")
        amp = float(item["cold_bytes"]) / reference
        require_close(item["cold_amplification"], amp, "cold read amplification")
        maximum = max(maximum, amp)
    require_close(ledger["maximum_cold_amplification"], maximum, "maximum cold read")
    if bool(ledger["below_2x"]) != (maximum < 2.0):
        raise AssertionError("read decision mismatch")


def all_selected_scores(
    report: dict[str, Any]
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    qwen_ledgers = report["qwen_matrix_candidates"]
    for bundle in report["qwen"]["rates"].values():
        rows.extend((row, qwen_ledgers) for row in bundle.values())
    controls = report["matched_gaussian_controls"]
    if len(controls["results"]) != len(controls["candidate_ledgers"]):
        raise AssertionError("control results/candidate-ledger count mismatch")
    for control, ledger in zip(
        controls["results"], controls["candidate_ledgers"], strict=True
    ):
        rows.extend((row, ledger["matrix_candidates"]) for row in control["rates"].values())
    return rows


def verify_source_files(report: dict[str, Any], source_root: Path) -> None:
    root = source_root.resolve(strict=True)
    receipts = report["source_audit"]["receipts"]
    if len(receipts) != MATRICES:
        raise AssertionError("source receipt count changed")
    for ordinal, receipt in enumerate(receipts):
        if int(receipt["matrix_ordinal"]) != ordinal:
            raise AssertionError("source receipt order changed")
        path = (root / receipt["relative_path"]).resolve(strict=True)
        if root != path and root not in path.parents:
            raise AssertionError("source path escaped root")
        if path.stat().st_size != int(receipt["bytes"]):
            raise AssertionError("source byte length mismatch")
        if sha256_file(path) != receipt["declared_sha256"]:
            raise AssertionError("source hash mismatch")


def verify_dual_certificates(report: dict[str, Any]) -> tuple[bool, float]:
    certificates = report["qwen"].get("dual_certificates", {})
    if set(certificates) != {"raw", "support_xklt"}:
        raise AssertionError("dual certificate does not cover both charged geometries")
    source_energy = float(report["source_audit"]["panel_source_energy"])
    passed = True
    lower_values = []
    for geometry, rows in certificates.items():
        if set(rows) != {"2.15", "2.30", "2.50"}:
            raise AssertionError("dual certificate rate coverage changed")
        variant = "charged_raw_bulk" if geometry == "raw" else "charged_support_xklt_bulk"
        for rate, certificate in rows.items():
            if int(certificate["expert_option_count_each"]) != 20**3:
                raise AssertionError("dual certificate did not enumerate 20^3 local choices")
            if int(certificate["expert_count"]) != EXPERTS:
                raise AssertionError("dual certificate expert count changed")
            if not certificate["complete_expert_local_grid_enumerated"]:
                raise AssertionError("dual enumeration is incomplete")
            lower_sse = float(certificate["certified_lower_bound_sse"])
            lower_mse = lower_sse / source_energy
            require_close(
                certificate["certified_lower_bound_relative_mse"],
                lower_mse,
                "dual relative-MSE identity",
                5e-12,
            )
            lower_f = lower_mse * 2.0 ** (
                2.0 * float(certificate["physical_rate_bpw"])
            )
            require_close(
                certificate["certified_lower_bound_F"], lower_f, "dual F identity", 5e-12
            )
            retained = report["qwen"]["rates"][rate][variant]
            if lower_sse > float(retained["distortion_sse"]) + 2e-9:
                raise AssertionError("dual lower bound exceeds exhibited primal distortion")
            if lower_f > float(retained["F"]) + 2e-10:
                raise AssertionError("dual lower F exceeds exhibited primal F")
            row_passed = lower_f > TARGET_F
            if bool(certificate["certifies_F_gt_0p8_for_complete_grid"]) != row_passed:
                raise AssertionError("dual certificate decision mismatch")
            passed &= row_passed
            lower_values.append(lower_f)
    return passed, min(lower_values)


def write_receipt(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol_lock.json"))
    parser.add_argument("--oracle", type=Path, default=Path(__file__).with_name("tail_peeling_composite.py"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result_path = args.result.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    oracle_path = args.oracle.resolve(strict=True)
    report = json.loads(result_path.read_text(encoding="utf-8"))
    if report.get("schema") != "qwen_sparse_tail_peeling_composite_oracle_v1":
        raise AssertionError("unexpected result schema")
    seal = verify_seal(report)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256_file(protocol_path) != report["protocol"]["sha256"]:
        raise AssertionError("protocol file hash mismatch")
    if protocol != report["protocol"]["contents"]:
        raise AssertionError("embedded protocol differs from file")
    if sha256_file(oracle_path) != report["audit"]["script_sha256"]:
        raise AssertionError("oracle script hash mismatch")
    scope = report["scope"]
    if int(scope["panel_values"]) != PANEL_VALUES or int(scope["matrix_count"]) != MATRICES:
        raise AssertionError("scope geometry changed")
    if float(scope["target_F"]) != TARGET_F:
        raise AssertionError("target F changed")
    verify_candidate_ledgers(report, [int(x) for x in protocol["tail_counts"]])
    rows = all_selected_scores(report)
    for row, matrix_ledgers in rows:
        verify_score(row, matrix_ledgers)
    dual_passed, weakest_dual_f = verify_dual_certificates(report)

    charged = []
    for rate, bundle in report["qwen"]["rates"].items():
        for variant in ("charged_raw_bulk", "charged_support_xklt_bulk"):
            charged.append((float(bundle[variant]["F"]), rate, variant, bundle[variant]))
    best_f, best_rate, best_variant, best_row = min(charged, key=lambda item: item[:3])
    decision = report["decision"]
    require_close(decision["best_charged"]["F"], best_f, "best charged selection")
    if decision["best_charged"]["rate"] != best_rate:
        raise AssertionError("best charged rate mismatch")
    if decision["best_charged"]["variant"] != best_variant:
        raise AssertionError("best charged variant mismatch")
    if bool(decision["charged_target_reached"]) != (best_f <= TARGET_F):
        raise AssertionError("aggregate target decision mismatch")
    if bool(decision["complete_grid_dual_certificate_passed"]) != dual_passed:
        raise AssertionError("aggregate dual decision mismatch")
    require_close(
        decision["weakest_certified_lower_bound_F_across_rates_and_geometries"],
        weakest_dual_f,
        "weakest dual bound selection",
    )
    expected_hard_kill = best_f > TARGET_F and dual_passed
    if bool(decision["hard_kill_tested_charged_family"]) != expected_hard_kill:
        raise AssertionError("hard-kill decision mismatch")
    expected_gpu = best_f <= TARGET_F or not dual_passed
    if bool(decision["gpu_finite_codec_followup_warranted"]) != expected_gpu:
        raise AssertionError("GPU promotion decision mismatch")
    if not best_row["read_ledger"]["below_2x"]:
        raise AssertionError("best charged row violates the expert-read gate")

    if args.source_root is not None:
        verify_source_files(report, args.source_root)
    receipt = {
        "schema": "qwen_sparse_tail_peeling_verification_v1",
        "passed": True,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "result_lock_sha256": seal,
        "protocol_sha256": sha256_file(protocol_path),
        "oracle_sha256": sha256_file(oracle_path),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "selected_rows_verified": len(rows),
        "sources_rehashed": MATRICES if args.source_root is not None else 0,
        "best_charged_F": best_f,
        "best_charged_rate": best_rate,
        "best_charged_variant": best_variant,
        "complete_grid_dual_certificate_passed": dual_passed,
        "weakest_certified_lower_bound_F": weakest_dual_f,
        "maximum_cold_expert_read_amplification": best_row["read_ledger"][
            "maximum_cold_amplification"
        ],
    }
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
        write_receipt(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
