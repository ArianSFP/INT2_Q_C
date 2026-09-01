#!/usr/bin/env python3
"""Verify the source-free same-layer expert-template dominance gate.

This verifier reads only compact JSON/Markdown evidence. It deliberately has
no model-weight loader, network client, CuPy import, or path-discovery logic.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
N_MATRIX = 768 * 2048
N_EXPERT = 3 * N_MATRIX
TOTAL_EXPERTS = 128
TARGET_RATE = 2.15
FULL_NEED = 0.16096404744368115
NESTED_NEED = 0.11356

INPUTS = {
    "basis": (
        HERE.parent / "neural_flow_oracle" / "shared_expert_basis_oracle.json",
        52033,
        "6ac8014943d55d243c1467f8ae0acb992da066a7e8b88962b2f3cd9da513c45e",
    ),
    "alignment": (
        HERE.parents[2] / "agent_rd_structure_diag_cross_expert_result.json",
        517531,
        "e60797e1845ed0f4fd7b6d6373e2668d3cec8d7cd6a2d549a457584f7d4f3b74",
    ),
    "alignment_report": (
        HERE.parents[2] / "agent_rd_structure_diag_cross_expert_report.md",
        3276,
        "a079a3066a5649e2ab2b4ab1fa4c6823cf8138918a2524e58f1e2aca95eeba83",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def close(actual: float, expected: float, label: str, tol: float = 1e-15) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def s_from_f(value: float) -> float:
    return -0.5 * math.log2(value)


def main() -> None:
    for label, (path, size, digest) in INPUTS.items():
        exact(path.stat().st_size, size, f"{label} byte count")
        exact(sha256(path), digest, f"{label} SHA-256")

    basis = json.loads(INPUTS["basis"][0].read_text(encoding="utf-8"))
    aligned = json.loads(INPUTS["alignment"][0].read_text(encoding="utf-8"))
    protocol_path = HERE / "protocol.lock.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result_path = HERE / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    exact(protocol_path.stat().st_size, result["protocol"]["bytes"], "protocol byte count")
    exact(sha256(protocol_path), result["protocol"]["sha256"], "protocol SHA-256")
    exact(sha256(Path(__file__).resolve()), result["verifier"]["sha256"], "verifier SHA-256")
    exact(protocol["strict_access_contract"]["new_qwen_payload_reads"], False, "protocol payload access")
    exact(protocol["strict_access_contract"]["network_access"], False, "protocol network access")
    exact(protocol["strict_access_contract"]["gpu_access"], False, "protocol GPU access")

    pooled = basis["pooled"]
    exact(basis["decision"], "HARD_KILL_SHARED_EXPERT_BASIS", "prior basis decision")
    q_mean = float(pooled["template"]["residual_energy_ratio_q"])
    q_rank1 = float(pooled["pca_rank_1"]["residual_energy_ratio_q"])
    q_rank15 = float(pooled["pca_rank_15"]["residual_energy_ratio_q"])
    s_rank15 = s_from_f(q_rank15)

    gaussian_q_mean = 16.0 / 15.0
    gaussian_q_rank1 = 1.0 - 1.0 / N_MATRIX
    gaussian_q_rank15 = 1.0 - 15.0 / N_MATRIX
    gaussian_s_rank15 = s_from_f(gaussian_q_rank15)

    variant = aligned["variant_results"]["ref_e0_concat_cosine"]
    crossfit = variant["crossfit_alignment_fit_then_eval_columns"]
    full = variant["full_columns_optimistic"]
    crossfit_gain = float(
        crossfit["rate_distortion"]["role_specific"]["fp64_klt_oracle"]
        ["side_free_gain_vs_same_total_rate_diagonal_percent"]
    )
    leaky_gain = float(
        full["rate_distortion"]["role_specific"]["fp16_decodable_gaussian_proxy"]
        ["side_free_gain_vs_same_total_rate_diagonal_percent"]
    )
    crossfit_f = 1.0 - crossfit_gain / 100.0
    leaky_f = 1.0 - leaky_gain / 100.0
    crossfit_s = s_from_f(crossfit_f)
    leaky_s = s_from_f(leaky_f)

    permutation_bits = math.ceil(math.lgamma(769.0) / math.log(2.0))
    gain_bits = 3 * 16
    cluster_id_bits = 0  # K=1 in the most favorable physically charged row.
    template_bytes = math.floor(N_EXPERT * TARGET_RATE / 8.0)
    template_bpw = template_bytes * 8.0 / N_EXPERT
    header_bpw = 512.0 / (TOTAL_EXPERTS * N_EXPERT)
    permutation_bpw = permutation_bits / N_EXPERT
    gain_bpw = gain_bits / N_EXPERT
    cluster_id_bpw = cluster_id_bits / N_EXPERT
    two_bank_template_bpw = 2.0 * template_bpw / TOTAL_EXPERTS
    side_bpw = (
        two_bank_template_bpw
        + permutation_bpw
        + gain_bpw
        + cluster_id_bpw
        + header_bpw
    )
    residual_budget_bpw = TARGET_RATE - side_bpw
    cold_read_amplification = (
        residual_budget_bpw
        + template_bpw
        + permutation_bpw
        + gain_bpw
        + cluster_id_bpw
        + header_bpw
    ) / TARGET_RATE
    hot_cached_read_amplification = (
        residual_budget_bpw
        + permutation_bpw
        + gain_bpw
        + cluster_id_bpw
        + header_bpw
    ) / TARGET_RATE

    close(q_mean, 1.066793197267109, "Qwen mean q")
    close(q_rank1, 0.9999993076481116, "Qwen rank-1 q")
    close(q_rank15, 0.9999871875561103, "Qwen rank-15 q")
    close(crossfit_gain, 0.00927946412057068, "crossfit aligned gain")
    close(leaky_gain, 0.2478521499169828, "leaky aligned gain")
    exact(permutation_bits, 6260, "enumerative permutation bits")

    computed = {
        "qwen": {
            "mean_q": q_mean,
            "rank1_q": q_rank1,
            "rank15_q": q_rank15,
            "rank15_s_bpw": s_rank15,
            "rank15_fraction_of_nested_need": s_rank15 / NESTED_NEED,
        },
        "gaussian_control": {
            "mean_q": gaussian_q_mean,
            "rank1_q": gaussian_q_rank1,
            "rank15_q": gaussian_q_rank15,
            "rank15_s_bpw": gaussian_s_rank15,
            "qwen_minus_control_rank15_s_bpw": s_rank15 - gaussian_s_rank15,
        },
        "bounded_alignment": {
            "crossfit_gain_percent": crossfit_gain,
            "crossfit_F": crossfit_f,
            "crossfit_s_bpw": crossfit_s,
            "leaky_full_gain_percent": leaky_gain,
            "leaky_full_F": leaky_f,
            "leaky_full_s_bpw": leaky_s,
            "crossfit_charged_s_bpw": crossfit_s - side_bpw,
            "crossfit_charged_F": 2.0 ** (-2.0 * (crossfit_s - side_bpw)),
            "leaky_full_charged_s_bpw": leaky_s - side_bpw,
            "leaky_full_charged_F": 2.0 ** (-2.0 * (leaky_s - side_bpw)),
        },
        "physical_one_prototype_crossfit": {
            "template_bytes_each": template_bytes,
            "template_bpw_each": template_bpw,
            "template_bank_count": 2,
            "two_bank_amortized_bpw": two_bank_template_bpw,
            "permutation_bits_per_expert": permutation_bits,
            "permutation_bpw": permutation_bpw,
            "fp16_gain_bits_per_expert": gain_bits,
            "fp16_gain_bpw": gain_bpw,
            "cluster_id_bits_per_expert": cluster_id_bits,
            "cluster_id_bpw": cluster_id_bpw,
            "header_bits": 512,
            "side_bpw": side_bpw,
            "residual_budget_bpw": residual_budget_bpw,
            "cold_read_amplification": cold_read_amplification,
            "hot_cached_read_amplification": hot_cached_read_amplification,
        },
    }

    for section, rows in computed.items():
        for key, value in rows.items():
            actual = result["computed"][section][key]
            if isinstance(value, float):
                close(float(actual), value, f"result {section}.{key}", tol=2e-15)
            else:
                exact(actual, value, f"result {section}.{key}")

    exact(result["decision"], "EARLY_KILL_NO_NEW_PAYLOAD_OR_GPU", "decision")
    if not s_rank15 < NESTED_NEED:
        raise AssertionError("unaligned dominance gate unexpectedly passed")
    if not crossfit_s < NESTED_NEED or not leaky_s < NESTED_NEED:
        raise AssertionError("bounded alignment plausibility gate unexpectedly passed")
    if not cold_read_amplification < 2.0:
        raise AssertionError("hypothetical cold-read ledger exceeds 2x")
    close(float(result["thresholds"]["full_required_s_bpw"]), FULL_NEED, "full threshold")
    close(float(result["thresholds"]["nested_remaining_required_s_bpw"]), NESTED_NEED, "nested threshold")
    exact(result["access_receipt"]["new_qwen_payload_reads"], 0, "payload reads")
    exact(result["access_receipt"]["network_calls"], 0, "network calls")
    exact(result["access_receipt"]["gpu_jobs"], 0, "GPU jobs")
    print("PASS: source-free same-layer expert-template dominance gate")


if __name__ == "__main__":
    main()
