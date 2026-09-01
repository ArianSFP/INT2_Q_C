#!/usr/bin/env python3
"""Independent binding and arithmetic verifier for the Haar/ACG gate.

This verifier intentionally does not import the experiment.  It rehashes every
allowed source and parent artifact, validates the canonical result digest,
rebuilds all held-out statistics, threshold decisions, dimension identities,
FP16 table identities, and byte/page read ledgers.  Recomputing QR features is
covered by rerunning the source-locked experiment and by its inversion tests;
the receipt states this boundary explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROWS = 768
COLS = 2048
EXPERTS = 6
ROLES = 3
PREFIX = 4096
FRAME_OVERHEAD = 160
MODEL_BYTES = 512
RATES = (2.15, 2.25, 2.5)
REQUIRED_S = -0.5 * math.log2(0.8)
SE_MULTIPLIER = 3.0


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def close(left: float, right: float, tolerance: float = 2e-14) -> None:
    if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError((left, right))


def summary(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    se = std / math.sqrt(len(x))
    return {
        "whole_expert_groups": len(x),
        "mean_s_bpw": mean,
        "sample_std_bpw": std,
        "standard_error_bpw": se,
        "confidence_se_multiplier": SE_MULTIPLIER,
        "lower_bpw": mean - SE_MULTIPLIER * se,
        "upper_bpw": mean + SE_MULTIPLIER * se,
    }


def check_summary(stored: dict[str, Any], values: list[float]) -> None:
    rebuilt = summary(values)
    for key, value in rebuilt.items():
        if isinstance(value, int):
            if int(stored[key]) != value:
                raise AssertionError((key, stored[key], value))
        else:
            close(stored[key], value)


def rebuild_ledgers() -> list[dict[str, Any]]:
    values_per_expert = ROLES * ROWS * COLS
    total_values = EXPERTS * values_per_expert
    rows = []
    for requested in RATES:
        container = math.floor(requested * total_values / 8.0)
        available = container - PREFIX
        base, remainder = divmod(available, EXPERTS)
        frames = [base + (index < remainder) for index in range(EXPERTS)]
        offsets = []
        cursor = PREFIX
        for frame in frames:
            offsets.append(cursor)
            cursor += frame
        attribution = container / EXPERTS
        exact = [PREFIX + frame for frame in frames]
        pages = [PREFIX + 4096 * (((offset + frame - 1) // 4096) - offset // 4096 + 1) for offset, frame in zip(offsets, frames, strict=True)]
        rows.append(
            {
                "requested_rate_bpw": requested,
                "physical_rate_bpw": 8.0 * container / total_values,
                "container_bytes": container,
                "global_prefix_bytes": PREFIX,
                "frame_bytes": frames,
                "frame_offsets": offsets,
                "expert_frame_overhead_bytes_inside_frame": FRAME_OVERHEAD,
                "expert_payload_bytes": [frame - FRAME_OVERHEAD for frame in frames],
                "physical_attribution_bytes_per_expert": attribution,
                "max_cold_exact_bytes": max(exact),
                "max_cold_exact_amplification": max(exact) / attribution,
                "max_cold_4k_bytes": max(pages),
                "max_cold_4k_amplification": max(pages) / attribution,
                "below_2x": max(pages) / attribution < 2.0,
            }
        )
    return rows


def compare_tree(stored: Any, rebuilt: Any, path: str = "") -> None:
    if isinstance(rebuilt, dict):
        if set(stored) != set(rebuilt):
            raise AssertionError(f"key mismatch at {path}: {set(stored) ^ set(rebuilt)}")
        for key in rebuilt:
            compare_tree(stored[key], rebuilt[key], f"{path}.{key}")
    elif isinstance(rebuilt, list):
        if len(stored) != len(rebuilt):
            raise AssertionError(f"length mismatch at {path}")
        for index, value in enumerate(rebuilt):
            compare_tree(stored[index], value, f"{path}[{index}]")
    elif isinstance(rebuilt, bool):
        if stored is not rebuilt:
            raise AssertionError((path, stored, rebuilt))
    elif isinstance(rebuilt, (float, int)):
        close(stored, rebuilt)
    elif stored != rebuilt:
        raise AssertionError((path, stored, rebuilt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--aux-dir", type=Path, required=True)
    parser.add_argument("--target-lock", type=Path, required=True)
    parser.add_argument("--composite-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    checks: list[str] = []

    if result["schema"] != "qwen_haar_manifold_entropy_gate_v1":
        raise AssertionError(result["schema"])
    checks.append("schema")
    content = dict(result)
    expected_content_hash = content.pop("result_content_sha256")
    if canonical_hash(content) != expected_content_hash:
        raise AssertionError("canonical content hash mismatch")
    checks.append("canonical_result_content_hash")

    script = args.result.parent / "haar_manifold_entropy.py"
    if sha256_file(script) != result["binding"]["script_sha256"]:
        raise AssertionError("experiment script hash mismatch")
    if sha256_file(args.target_lock) != result["binding"]["target_lock_sha256"]:
        raise AssertionError("target lock hash mismatch")
    if sha256_file(args.composite_result) != result["thresholds"]["nested_composite"]["artifact_sha256"]:
        raise AssertionError("composite parent hash mismatch")
    checks.extend(("experiment_script_hash", "target_lock_hash", "composite_parent_hash"))

    target_lock = json.loads(args.target_lock.read_text(encoding="utf-8"))
    target_layers = {int(row["layer"]) for row in target_lock["matrices"]}
    target_experts = {int(row["expert"]) for row in target_lock["matrices"]}
    if len(target_lock["matrices"]) != 18 or result["binding"]["pinned_source_payloads_opened"] is not False:
        raise AssertionError("pinned firewall declaration invalid")
    source_hashes = result["binding"]["auxiliary_source_sha256"]
    observed = {path.name for path in args.aux_dir.glob("*.bf16.bin") if path.name in source_hashes}
    if observed != set(source_hashes):
        raise AssertionError("auxiliary source set mismatch")
    for name, digest in source_hashes.items():
        if sha256_file(args.aux_dir / name) != digest:
            raise AssertionError(f"auxiliary hash mismatch: {name}")
    for expert in result["auxiliary_metadata"]:
        if int(expert["layer"]) in target_layers or int(expert["expert"]) in target_experts:
            raise AssertionError("held-out leakage into auxiliary set")
        if len(expert["roles"]) != 2:
            raise AssertionError("whole-expert role count")
        for role in expert["roles"]:
            for diagnostics in role["diagnostics"].values():
                if float(diagnostics["max_sphere_norm_abs_error"]) > 2e-4:
                    raise AssertionError("invalid Householder sphere norm")
            moments = role["control_moments"]
            if float(moments["relative_centered_energy_error"]) > 2e-6:
                raise AssertionError("moment control drift")
    checks.extend(("pinned_source_firewall", "all_auxiliary_source_hashes", "whole_expert_nonoverlap", "qr_and_moment_diagnostics"))

    matrix_values = ROWS * COLS
    orientation = matrix_values - ROWS * (ROWS + 1) // 2
    triangle = ROWS * (ROWS + 1) // 2
    dimensions = result["canonicalization"]["dimensions"]
    if dimensions["orientation_dof"] != orientation or dimensions["upper_triangular_dof"] != triangle or dimensions["sum_dof"] != matrix_values:
        raise AssertionError("coordinate DOF mismatch")
    close(dimensions["orientation_fraction"], orientation / matrix_values)
    if "cancels exactly" not in dimensions["jacobian_handling"]:
        raise AssertionError("Jacobian accounting missing")
    checks.append("coordinate_dof_and_jacobians")

    for population in ("qwen", "moment_gaussian", "haar"):
        row = result["crossfit"][population]
        groups = row["groups"]
        if [int(group["heldout_index"]) for group in groups] != list(range(len(groups))):
            raise AssertionError("heldout index sequence")
        gains = []
        for group in groups:
            rebuilt = sum(float(value) for value in group["role_gain_bits"]) / (2 * matrix_values)
            close(group["gain_bpw"], rebuilt)
            gains.append(rebuilt)
        check_summary(row["summary"], gains)
        table = np.asarray(row["final_fp16_shape_table"], dtype=np.float16)
        if table.shape != (2, 8, 16) or not np.all(np.isfinite(table)):
            raise AssertionError(f"{population} table schema")
        if hashlib.sha256(table.tobytes(order="C")).hexdigest() != row["final_fp16_shape_table_sha256"]:
            raise AssertionError(f"{population} table hash")
    checks.extend(("all_crossfit_group_arithmetic", "all_crossfit_summaries", "all_fp16_table_hashes"))

    qwen = result["crossfit"]["qwen"]["groups"]
    paired_specs = (
        ("qwen_minus_haar", result["crossfit"]["haar"]["groups"]),
        ("qwen_minus_moment_gaussian", result["crossfit"]["moment_gaussian"]["groups"]),
    )
    paired_rows = []
    for name, control in paired_specs:
        values = [float(a["gain_bpw"]) - float(b["gain_bpw"]) for a, b in zip(qwen, control, strict=True)]
        stored = result["comparisons"][name]
        np.testing.assert_allclose(stored["group_difference_bpw"], values, rtol=2e-14, atol=2e-14)
        check_summary(stored, values)
        paired_rows.append(stored)
    checks.append("paired_control_statistics")

    total_values = EXPERTS * ROLES * matrix_values
    model_rate = 8.0 * MODEL_BYTES / total_values
    close(result["density"]["serialized_model_bpw_on_pinned_panel"], model_rate)
    if result["density"]["serialized_model_bytes"] != MODEL_BYTES:
        raise AssertionError("model byte ledger")
    lower = min(float(row["lower_bpw"]) for row in paired_rows) - model_rate
    upper = min(float(row["upper_bpw"]) for row in paired_rows) - model_rate
    close(result["comparisons"]["conservative_lower_after_model_rate_bpw"], lower)
    close(result["comparisons"]["optimistic_upper_after_model_rate_bpw"], upper)
    checks.append("charged_control_bounds")

    composite = json.loads(args.composite_result.read_text(encoding="utf-8"))
    base = composite["variants"]["role_gauge+polar"]["rates"]["2.50"]
    base_s = float(base["s_bpw"])
    base_f = float(base["F"])
    nested = result["thresholds"]["nested_composite"]
    close(nested["base_s_bpw"], base_s)
    close(nested["base_F"], base_f)
    close(nested["incremental_s_required_bpw"], REQUIRED_S - base_s)
    close(nested["identity_check_incremental_s"], -0.5 * math.log2(0.8 / base_f))
    needed = REQUIRED_S - base_s
    if lower >= REQUIRED_S:
        decision = "PROMOTE_STANDALONE_FINITE_RATE_CODEC"
    elif lower >= needed:
        decision = "PROMOTE_DIRECT_NESTED_COMPOSITE_TEST"
    elif upper < needed:
        decision = "HARD_KILL_HAAR_ACG_ORIENTATION_ENTROPY"
    else:
        decision = "INCONCLUSIVE_EXPAND_AUXILIARY_PANEL"
    if result["decision"] != decision:
        raise AssertionError((result["decision"], decision))
    checks.extend(("composite_threshold_binding", "decision_reselection"))

    free_upper = min(float(row["upper_bpw"]) for row in paired_rows)
    free_multiplier = 2.0 ** (-2.0 * free_upper)
    charged_multiplier = 2.0 ** (-2.0 * upper)
    rd = result["high_rate_rd_envelope"]
    close(rd["qwen_mean_F_multiplier"], 2.0 ** (-2.0 * float(result["crossfit"]["qwen"]["summary"]["mean_s_bpw"])))
    close(rd["moment_gaussian_mean_F_multiplier"], 2.0 ** (-2.0 * float(result["crossfit"]["moment_gaussian"]["summary"]["mean_s_bpw"])))
    close(rd["haar_mean_F_multiplier"], 2.0 ** (-2.0 * float(result["crossfit"]["haar"]["summary"]["mean_s_bpw"])))
    close(rd["qwen_specific_free_table_three_se_upper_s_bpw"], free_upper)
    close(rd["qwen_specific_free_table_F_multiplier"], free_multiplier)
    close(rd["qwen_specific_charged_three_se_upper_s_bpw"], upper)
    close(rd["qwen_specific_charged_F_multiplier"], charged_multiplier)
    close(rd["nested_composite_free_table_optimistic_F"], base_f * free_multiplier)
    if len(rd["rates"]) != len(RATES):
        raise AssertionError("RD rate count")
    for stored, rate in zip(rd["rates"], RATES, strict=True):
        gaussian_mse = 2.0 ** (-2.0 * rate)
        close(stored["rate_bpw"], rate)
        close(stored["gaussian_reference_mse"], gaussian_mse)
        close(stored["standalone_free_table_optimistic_mse"], gaussian_mse * free_multiplier)
        close(stored["standalone_charged_optimistic_mse"], gaussian_mse * charged_multiplier)
        close(stored["nested_composite_free_table_optimistic_mse"], gaussian_mse * base_f * free_multiplier)
        close(stored["target_mse"], 0.8 * gaussian_mse)
    checks.append("high_rate_rd_envelope")

    rebuilt_ledgers = rebuild_ledgers()
    compare_tree(result["serialized_layout"]["rate_read_ledgers"], rebuilt_ledgers, "rate_read_ledgers")
    if not all(row["physical_rate_bpw"] <= row["requested_rate_bpw"] and row["max_cold_4k_amplification"] < 2.0 for row in rebuilt_ledgers):
        raise AssertionError("physical rate/read cap")
    checks.extend(("byte_exact_rate_ledgers", "page_exact_read_ledgers"))

    receipt = {
        "schema": "qwen_haar_manifold_entropy_verification_v1",
        "passed": True,
        "checks": checks,
        "check_count": len(checks),
        "result_sha256": sha256_file(args.result),
        "result_content_sha256": expected_content_hash,
        "decision": decision,
        "qwen_mean_s_bpw": result["crossfit"]["qwen"]["summary"]["mean_s_bpw"],
        "conservative_lower_after_model_rate_bpw": lower,
        "optimistic_upper_after_model_rate_bpw": upper,
        "nested_required_s_bpw": needed,
        "verification_boundary": "binding and independent arithmetic/layout verification; canonical QR/ACG feature values are reproduced by rerunning the source-locked experiment and covered by the raw-QR inversion tests",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
