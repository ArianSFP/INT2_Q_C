#!/usr/bin/env python3
"""Audit and supersede the free-SVD-tail ``F`` interpretation.

The spectral screen was useful as a hypothesis generator, but its best number
is a ratio of continuous rank-k residuals, not a rate-feasible codec score.
This verifier binds that screen to the independently rate-accounted
Stiefel/Gram oracle and (optionally) the discrete binary-factor pilot, then
emits a deterministic, machine-checkable supersession receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROWS = 768
COLS = 2048
VALUES = ROWS * COLS
RANK = 764
TARGET_F = 0.8
EXPECTED_SPECTRAL_SCHEMA = "qwen-nanoquant-binary-factor-spectral-gate-v1"
EXPECTED_STIEFEL_SCHEMA = "qwen-stiefel-gram-oracle-v1"
EXPECTED_BINARY_SCHEMA = "qwen-nanoquant-discrete-tile-pilot-v1"


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_embedded_lock(value: dict[str, Any], field: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str):
        raise ValueError(f"missing {field}")
    clean = dict(value)
    clean.pop(field)
    actual = hashlib.sha256(canonical_json(clean)).hexdigest()
    if actual != claimed:
        raise ValueError(f"{field} mismatch: {actual} != {claimed}")
    return actual


def close(actual: float, expected: float, *, atol: float = 2e-14) -> None:
    if not math.isclose(actual, expected, rel_tol=2e-13, abs_tol=atol):
        raise ValueError(f"numeric identity failed: {actual} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-result", required=True, type=Path)
    parser.add_argument("--stiefel-result", required=True, type=Path)
    parser.add_argument("--binary-result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    spectral_path = args.spectral_result.resolve(strict=True)
    stiefel_path = args.stiefel_result.resolve(strict=True)
    spectral = load(spectral_path)
    stiefel = load(stiefel_path)
    if spectral.get("schema") != EXPECTED_SPECTRAL_SCHEMA:
        raise ValueError("unexpected spectral schema")
    if stiefel.get("schema") != EXPECTED_STIEFEL_SCHEMA:
        raise ValueError("unexpected Stiefel schema")
    spectral_lock = verify_embedded_lock(spectral, "result_lock_sha256")

    candidates = spectral.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("spectral candidates missing")
    best = min(candidates, key=lambda row: float(row["source_over_matched_gaussian"]))
    if best.get("candidate_id") != "xklt:plain:rank764":
        raise ValueError("unexpected best spectral candidate")
    if int(best["continuous_rank"]) != RANK:
        raise ValueError("unexpected best rank")
    source_tail = float(best["source_relative_residual"])
    gaussian_tail = float(best["gaussian_relative_residual"])
    tail_ratio = source_tail / gaussian_tail
    tail_s = -0.5 * math.log2(tail_ratio)
    close(tail_ratio, float(best["source_over_matched_gaussian"]))
    close(tail_s, float(best["structural_advantage_bpw"]))

    # Embedded rank-k matrix-manifold dimensions.
    u_stiefel_dof = ROWS * RANK - RANK * (RANK + 1) // 2
    v_stiefel_dof = COLS * RANK - RANK * (RANK + 1) // 2
    singular_value_dof = RANK
    orientation_dof = u_stiefel_dof + v_stiefel_dof
    manifold_dof = orientation_dof + singular_value_dof
    normal_dof = (ROWS - RANK) * (COLS - RANK)
    if manifold_dof != RANK * (ROWS + COLS - RANK):
        raise AssertionError("rank-manifold dimension identity failed")
    if manifold_dof + normal_dof != VALUES:
        raise AssertionError("tangent/normal dimension identity failed")

    rate_ledgers: list[dict[str, Any]] = []
    for rate in (2.15, 2.5):
        budget_bits = rate * VALUES
        rate_ledgers.append(
            {
                "physical_rate_bpw": rate,
                "bits_per_matrix": budget_bits,
                "bits_per_rank_manifold_dof_if_all_bits_assigned_there": (
                    budget_bits / manifold_dof
                ),
                "bits_per_orientation_dof_if_all_bits_assigned_there": (
                    budget_bits / orientation_dof
                ),
            }
        )

    naive_fp16_scalars = RANK * (ROWS + COLS + 1)
    naive_fp16_bits = 16 * naive_fp16_scalars

    decision = stiefel["decision"]
    methodology = stiefel["methodology"]
    adaptive = stiefel["structured_gram_rank_models"]["adaptive_rank"]
    stiefel_f = float(decision["best_optimistic_F_at_2p5"])
    stiefel_s = float(decision["best_optimistic_rate_equivalent_s_bpw"])
    close(stiefel_s, -0.5 * math.log2(stiefel_f))
    q_stiefel_dof = VALUES - ROWS * (ROWS + 1) // 2
    if int(methodology["stiefel_dof"]) != q_stiefel_dof:
        raise ValueError("Stiefel dimension mismatch")
    average_model_dof = float(adaptive["average_model_dof"])

    binary_receipt: dict[str, Any] | None = None
    if args.binary_result is not None:
        binary_path = args.binary_result.resolve(strict=True)
        binary = load(binary_path)
        if binary.get("schema") != EXPECTED_BINARY_SCHEMA:
            raise ValueError("unexpected binary-factor schema")
        binary_lock = verify_embedded_lock(binary, "result_lock_sha256")
        rows = binary.get("aggregate")
        if not isinstance(rows, list) or not rows:
            raise ValueError("binary aggregate missing")
        row = min(
            rows,
            key=lambda item: abs(float(item["ledger"]["physical_rate_bpw"]) - 2.5),
        )
        binary_rate = float(row["ledger"]["physical_rate_bpw"])
        binary_d = float(row["source_relative_mse"])
        binary_f = binary_d * 2.0 ** (2.0 * binary_rate)
        close(binary_f, float(row["source_F_equals_D_times_2pow2R"]))
        binary_ratio = float(row["source_over_matched_gaussian"])
        close(
            binary_ratio,
            binary_d / float(row["gaussian_relative_mse"]),
        )
        binary_receipt = {
            "path_name": binary_path.name,
            "file_sha256": sha256_file(binary_path),
            "result_lock_sha256": binary_lock,
            "physical_rate_bpw": binary_rate,
            "tile_shape": row["ledger"]["tile_shape"],
            "rank": int(row["ledger"]["rank"]),
            "source_relative_mse": binary_d,
            "matched_gaussian_relative_mse": float(row["gaussian_relative_mse"]),
            "source_over_matched_gaussian": binary_ratio,
            "structural_advantage_bpw": float(row["structural_advantage_bpw"]),
            "source_codec_F": binary_f,
            "cold_read_amplification": float(
                row["ledger"]["cold_read_amplification"]
            ),
            "passes_target_F_le_0p8": bool(binary_f <= TARGET_F),
            "claim_boundary": binary["claim_boundary"],
        }

    audit: dict[str, Any] = {
        "schema": "svd-tail-survivor-supersession-v1",
        "status": "INVALID_AS_RATE_FEASIBLE_CODEC_SURVIVOR",
        "inputs": {
            "spectral": {
                "path_name": spectral_path.name,
                "file_sha256": sha256_file(spectral_path),
                "result_lock_sha256": spectral_lock,
            },
            "stiefel_gram": {
                "path_name": stiefel_path.name,
                "file_sha256": sha256_file(stiefel_path),
                "script_sha256": stiefel["audit"]["script_sha256"],
                "source_lock_internal_sha256": stiefel["audit"][
                    "source_lock_internal_sha256"
                ],
            },
            "binary_factor": binary_receipt,
        },
        "apparent_survivor": {
            "candidate_id": best["candidate_id"],
            "source_discarded_tail_relative_energy": source_tail,
            "matched_gaussian_discarded_tail_relative_energy": gaussian_tail,
            "tail_energy_ratio": tail_ratio,
            "tail_only_rate_equivalent_s_bpw": tail_s,
            "percent_tail_ratio_below_gaussian": 100.0 * (1.0 - tail_ratio),
            "not_codec_F": True,
            "if_tail_were_incorrectly_treated_as_full_D_at_R2p5": {
                "source_D_times_2pow2R": source_tail * 32.0,
                "gaussian_D_times_2pow2R": gaussian_tail * 32.0,
                "note": "These impossible-looking values expose the uncharged retained reconstruction.",
            },
        },
        "degrees_of_freedom_audit": {
            "matrix_shape": [ROWS, COLS],
            "ambient_values": VALUES,
            "rank": RANK,
            "U_stiefel_dof": u_stiefel_dof,
            "V_stiefel_dof": v_stiefel_dof,
            "source_specific_orientation_dof": orientation_dof,
            "singular_value_dof": singular_value_dof,
            "rank_manifold_dof": manifold_dof,
            "rank_manifold_fraction": manifold_dof / VALUES,
            "discarded_normal_dof": normal_dof,
            "discarded_normal_fraction": normal_dof / VALUES,
            "retained_source_energy_fraction": 1.0 - source_tail,
            "free_objects_in_spectral_screen": [
                "source-specific U basis",
                "source-specific V basis",
                "all retained singular values",
                "exact reconstruction of the retained rank-764 component",
            ],
            "rate_ledgers": rate_ledgers,
            "naive_fp16_U_V_sigma": {
                "scalars_per_matrix": naive_fp16_scalars,
                "bits_per_matrix": naive_fp16_bits,
                "bpw": naive_fp16_bits / VALUES,
                "multiple_of_2p5_bpw_budget": naive_fp16_bits / (2.5 * VALUES),
            },
        },
        "metric_and_jacobian_audit": {
            "differential": (
                "dW = U dSigma V^T + dU Sigma V^T + U Sigma dV^T, "
                "after quotienting the joint signed/rotational gauge"
            ),
            "frobenius_metric": (
                "basis perturbations are multiplied by Sigma; within-subspace "
                "rotations couple singular directions, so equal parameter error "
                "does not imply equal W-space MSE"
            ),
            "embedded_volume_element": (
                "for distinct nonzero singular values, the SVD Jacobian contains "
                "products of powers of sigma_i and |sigma_i^2-sigma_j^2|; the "
                "tail-only score charges neither this volume nor chart precision"
            ),
            "omitted_coordinate_distortion": True,
        },
        "why_leave_one_out_does_not_repair_it": (
            "LOO cross-fitting protects selection of representation/scaling/rank, "
            "but each held-out matrix is still decomposed into its own exact U, V, "
            "and singular values at zero transmitted rate."
        ),
        "rate_accounted_comparator": {
            "model": stiefel["structured_gram_rank_models"]["model"],
            "scoring": methodology["scoring"],
            "optimism": methodology["optimism"],
            "Q_stiefel_dof": q_stiefel_dof,
            "adaptive_gram_rank_min_mean_max": [
                int(adaptive["minimum_rank"]),
                float(adaptive["mean_rank"]),
                int(adaptive["maximum_rank"]),
            ],
            "adaptive_average_model_dof": average_model_dof,
            "adaptive_average_normal_dof": VALUES - average_model_dof,
            "adaptive_aggregate_residual_energy_ratio": float(
                adaptive["aggregate_residual_energy_ratio"]
            ),
            "best_optimistic_F_at_2p5": stiefel_f,
            "best_optimistic_s_bpw": stiefel_s,
            "required_F": float(decision["required_F"]),
            "required_s_bpw": float(decision["required_rate_equivalent_s_bpw"]),
            "shortfall_s_bpw": float(decision["shortfall_s_bpw"]),
            "decision": decision["status"],
        },
        "followup_decision": {
            "test_implicit_Householder_Givens_or_sparse_tangent_codec": False,
            "reason": (
                "The only apparent >=target signal was the uncharged tail ratio. "
                "The more favorable rate-accounted continuous Stiefel/Gram oracle "
                "already falls to F=0.95049, and the charged discrete binary-factor "
                "test (when supplied) is much worse."
            ),
            "claim_boundary": (
                "This invalidates promotion of this SVD-tail screen and rejects the "
                "tested NanoQuant-style factorization; it is not an impossibility "
                "proof for every learned or procedural basis codec."
            ),
        },
    }
    audit["audit_lock_sha256"] = hashlib.sha256(canonical_json(audit)).hexdigest()
    serialized = json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
