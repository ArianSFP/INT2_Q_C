#!/usr/bin/env python3
"""CPU-only source oracle for exact SwiGLU gauge symmetry and joint polar geometry.

For one expert, write the three semantic matrices as G, U, D in R^(768x2048),
where D is down_proj.T.  The expert function is invariant to

    U <- A U,  D <- A^-1 D

for any nonsingular diagonal A (and to a shared permutation of the rows of
G/U/D).  We choose the exact equal-norm gauge, concatenate [G, AU, A^-1 D],
and fit a *joint* polar-normal manifold

    X = H Q,  Q Q^T = I,  H ~= c I + A_k.

The gauge is retained, so source-domain reconstruction remains the metric.
Every rank is evaluated with the exact inverse-gauge quadratic form.  The
reported score grants continuous coordinates, ideal Gaussian reverse
waterfilling, free charts, and lossless gauge values.  It is therefore an
optimistic early-kill oracle, not a codec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CHANNELS = 768
WIDTH = 2048
ROLES = ("gate", "up", "down")
VALUES_PER_ROLE = CHANNELS * WIDTH
VALUES_PER_EXPERT = 3 * VALUES_PER_ROLE
EXPERTS = 6
PANEL_VALUES = EXPERTS * VALUES_PER_EXPERT
JOINT_COLS = 3 * WIDTH
SYMMETRIC_DOF = CHANNELS * (CHANNELS + 1) // 2
STIEFEL_DOF = VALUES_PER_EXPERT - SYMMETRIC_DOF
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
RATES = (2.15, 2.25, 2.5)
MIN_RATE = 2.15
MAX_RATE = 2.5


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def load_bf16(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    if words.size != math.prod(shape):
        raise AssertionError((path, words.size, shape))
    return (words.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def load_panel(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    lock_path = root / "blind_protocol_v2/unblinded/source_hashes.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if len(lock["matrices"]) != 18:
        raise AssertionError("the pinned panel must contain eighteen matrices")
    source_root = lock_path.parent
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for row in lock["matrices"]:
        path = source_root / row["output_relpath"]
        observed = sha256_file(path)
        declared = str(row["source_bf16_sha256"])
        if observed != declared:
            raise AssertionError(f"source hash mismatch: {path}")
        shape = tuple(int(x) for x in row["shape"])
        matrix = load_bf16(path, shape)
        role = str(row["role"])
        if role == "down":
            matrix = matrix.T
        matrix64 = np.ascontiguousarray(matrix, dtype=np.float64)
        if matrix64.shape != (CHANNELS, WIDTH):
            raise AssertionError((path, matrix64.shape))
        gram = matrix64 @ matrix64.T
        total = float(np.sum(matrix64, dtype=np.float64))
        energy = float(np.trace(gram))
        key = (int(row["layer"]), int(row["expert"]))
        entry = grouped.setdefault(
            key,
            {
                "layer": key[0],
                "expert": key[1],
                "role_grams": {},
                "moments": {},
            },
        )
        entry["role_grams"][role] = gram
        mean = total / VALUES_PER_ROLE
        variance = max(0.0, energy / VALUES_PER_ROLE - mean * mean)
        entry["moments"][role] = {
            "count": VALUES_PER_ROLE,
            "sum": total,
            "sum_sq": energy,
            "mean": mean,
            "variance": variance,
        }
        receipts.append(
            {
                "matrix_ordinal": int(row["matrix_ordinal"]),
                "layer": key[0],
                "expert": key[1],
                "role": role,
                "shape": list(shape),
                "nbytes": int(path.stat().st_size),
                "relative_path": str(path.relative_to(source_root)).replace("\\", "/"),
                "declared_sha256": declared,
                "observed_sha256": observed,
            }
        )
        del matrix, matrix64
    experts = [grouped[key] for key in sorted(grouped)]
    if len(experts) != EXPERTS or any(set(x["role_grams"]) != set(ROLES) for x in experts):
        raise AssertionError("incomplete expert triplets")
    return lock_path, lock, experts, receipts


def block_prefix(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros((matrix.shape[0] + 1, matrix.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = np.cumsum(np.cumsum(matrix, axis=0), axis=1)
    return out


def square_block_sum(prefix: np.ndarray, start: int, stop: int) -> float:
    return float(prefix[stop, stop] - prefix[start, stop] - prefix[stop, start] + prefix[start, start])


def rank_curve_from_metric(singular: np.ndarray, metric: np.ndarray) -> list[dict[str, Any]]:
    """Target-metric optimum over every contiguous unmodelled eigenspectrum window.

    For a window W and common retained singular value c, the inverse-gauge
    residual is (1-c/s)^T K (1-c/s).  Three 2-D prefix sums make the exact
    c and residual O(1) per window.  All O(n^2/2) windows are exhausted.
    """
    inv = 1.0 / singular
    p0 = block_prefix(metric)
    p1 = block_prefix(inv[:, None] * metric)
    p2 = block_prefix((inv[:, None] * metric) * inv[None, :])
    curve: list[dict[str, Any]] = []
    for rank in range(CHANNELS - 1):
        length = CHANNELS - rank
        best: tuple[float, int, float] | None = None
        for start in range(rank + 1):
            stop = start + length
            a0 = square_block_sum(p0, start, stop)
            a1 = square_block_sum(p1, start, stop)
            a2 = square_block_sum(p2, start, stop)
            if not a2 > 0.0:
                raise AssertionError((rank, start, a2))
            common = max(0.0, a1 / a2)
            residual = max(0.0, a0 - 2.0 * common * a1 + common * common * a2)
            candidate = (residual, start, common)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        residual, start, common = best
        h_dof = 1 + CHANNELS * rank - rank * (rank - 1) // 2
        model_dof = STIEFEL_DOF + h_dof
        normal_dof = VALUES_PER_EXPERT - model_dof
        if normal_dof <= 0:
            break
        curve.append(
            {
                "rank": rank,
                "model_dof": model_dof,
                "normal_dof": normal_dof,
                "unmodelled_window_start": start,
                "unmodelled_window_stop": start + length,
                "common_singular_value": common,
                "source_residual_energy": residual,
            }
        )
    return curve


def analyze_grams(
    role_grams: dict[str, np.ndarray],
    *,
    identity: str,
    include_curve: bool,
) -> dict[str, Any]:
    gate = np.asarray(role_grams["gate"], dtype=np.float64)
    up = np.asarray(role_grams["up"], dtype=np.float64)
    down = np.asarray(role_grams["down"], dtype=np.float64)
    up_norm = np.sqrt(np.maximum(np.diag(up), np.finfo(np.float64).tiny))
    down_norm = np.sqrt(np.maximum(np.diag(down), np.finfo(np.float64).tiny))
    gauge = np.sqrt(down_norm / up_norm)
    up_c = (gauge[:, None] * up) * gauge[None, :]
    inv_gauge = 1.0 / gauge
    down_c = (inv_gauge[:, None] * down) * inv_gauge[None, :]
    joint = gate + up_c + down_c
    eigenvalues, vectors = np.linalg.eigh(joint)
    eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).tiny)
    singular = np.sqrt(eigenvalues)

    # Source metric after undoing U_c=A U and D_c=A^-1 D.
    c_gate = vectors.T @ gate @ vectors
    c_up = vectors.T @ up_c @ vectors
    c_down = vectors.T @ down_c @ vectors
    b_up = vectors.T @ ((inv_gauge * inv_gauge)[:, None] * vectors)
    b_down = vectors.T @ ((gauge * gauge)[:, None] * vectors)
    metric = c_gate * np.eye(CHANNELS) + b_up * c_up.T + b_down * c_down.T
    metric = 0.5 * (metric + metric.T)
    curve = rank_curve_from_metric(singular, metric)
    source_energy = float(np.trace(gate) + np.trace(up) + np.trace(down))
    for row in curve:
        row["source_residual_ratio"] = float(row["source_residual_energy"] / source_energy)

    output: dict[str, Any] = {
        "identity": identity,
        "source_energy": source_energy,
        "canonical_energy": float(np.sum(eigenvalues)),
        "gauge": {
            "definition": "a_j=sqrt(||down_j||_2/||up_j||_2)",
            "minimum": float(np.min(gauge)),
            "maximum": float(np.max(gauge)),
            "mean_log": float(np.mean(np.log(gauge))),
            "std_log": float(np.std(np.log(gauge))),
            "max_equal_norm_relative_error": float(
                np.max(np.abs(gauge * up_norm - inv_gauge * down_norm))
                / max(float(np.max(gauge * up_norm)), np.finfo(np.float64).tiny)
            ),
        },
        "singular_summary": {
            "minimum": float(singular[0]),
            "maximum": float(singular[-1]),
            "mean": float(np.mean(singular)),
            "std": float(np.std(singular)),
        },
        "curve": curve if include_curve else None,
    }
    return output


def reverse_waterfill(components: Iterable[dict[str, float]], rate: float) -> dict[str, Any]:
    rows = list(components)
    dimensions = np.asarray([float(x["dimension_fraction"]) for x in rows], dtype=np.float64)
    energies = np.asarray([float(x["energy_fraction"]) for x in rows], dtype=np.float64)
    if not math.isclose(float(np.sum(dimensions)), 1.0, rel_tol=2e-10, abs_tol=2e-10):
        raise AssertionError(("dimension sum", float(np.sum(dimensions))))
    if not math.isclose(float(np.sum(energies)), 1.0, rel_tol=2e-10, abs_tol=2e-10):
        raise AssertionError(("energy sum", float(np.sum(energies))))
    variances = energies / dimensions
    logv = np.log2(variances)

    def used(log_level: float) -> float:
        return 0.5 * float(np.sum(dimensions * np.maximum(0.0, logv - log_level)))

    lo = float(np.min(logv) - 2.0 * rate / np.min(dimensions) - 16.0)
    hi = float(np.max(logv))
    for _ in range(180):
        mid = 0.5 * (lo + hi)
        if used(mid) > rate:
            lo = mid
        else:
            hi = mid
    water = 2.0**hi
    distortion = float(np.sum(dimensions * np.minimum(variances, water)))
    f_value = distortion * 2.0 ** (2.0 * rate)
    allocations = []
    for spec, dim, energy, variance in zip(rows, dimensions, energies, variances):
        active = bool(variance > water)
        local_rate = 0.5 * max(0.0, math.log2(float(variance / water)))
        allocations.append(
            {
                "component": spec["component"],
                "dimension_fraction": float(dim),
                "energy_fraction": float(energy),
                "variance_per_dimension": float(variance),
                "bits_per_component_dimension": local_rate,
                "rate_bpw_contribution": float(dim) * local_rate,
                "active": active,
            }
        )
    return {
        "physical_rate_bpw": rate,
        "ideal_relative_mse": distortion,
        "gaussian_reference_relative_mse": 2.0 ** (-2.0 * rate),
        "F": f_value,
        "s_bpw": -0.5 * math.log2(f_value),
        "passes_target": f_value <= TARGET_F,
        "water_level": water,
        "allocations": allocations,
    }


def score_rank_vector(analyses: list[dict[str, Any]], ranks: list[int], rate: float) -> dict[str, Any]:
    total_energy = sum(float(x["source_energy"]) for x in analyses)
    components: list[dict[str, float]] = []
    selected = []
    for analysis, rank in zip(analyses, ranks):
        row = analysis["curve"][rank]
        residual = min(float(row["source_residual_energy"]), float(analysis["source_energy"]) * (1.0 - 1e-15))
        energy = float(analysis["source_energy"])
        components.extend(
            [
                {
                    "component": f"{analysis['identity']}:joint_manifold",
                    "dimension_fraction": float(row["model_dof"]) / PANEL_VALUES,
                    "energy_fraction": (energy - residual) / total_energy,
                },
                {
                    "component": f"{analysis['identity']}:polar_normal",
                    "dimension_fraction": float(row["normal_dof"]) / PANEL_VALUES,
                    "energy_fraction": residual / total_energy,
                },
            ]
        )
        selected.append(
            {
                "identity": analysis["identity"],
                **{key: row[key] for key in row if key != "source_residual_energy"},
                "source_residual_energy": residual,
            }
        )
    score = reverse_waterfill(components, rate)
    score["ranks"] = list(ranks)
    score["selected"] = selected
    return score


def optimize_ranks(analyses: list[dict[str, Any]], rate: float) -> dict[str, Any]:
    common_scores = [score_rank_vector(analyses, [rank] * EXPERTS, rate) for rank in range(CHANNELS - 1)]
    common = min(common_scores, key=lambda x: (x["F"], x["ranks"]))
    ranks = list(common["ranks"])
    # Exact coordinate minimization over all ranks.  Repeated sweeps account
    # for inactive reverse-waterfill components; no all-active assumption.
    for _ in range(12):
        changed = False
        for expert_index in range(EXPERTS):
            candidates = []
            for rank in range(CHANNELS - 1):
                trial = list(ranks)
                trial[expert_index] = rank
                candidates.append(score_rank_vector(analyses, trial, rate))
            winner = min(candidates, key=lambda x: (x["F"], x["ranks"]))
            if winner["ranks"][expert_index] != ranks[expert_index]:
                ranks = list(winner["ranks"])
                changed = True
        if not changed:
            break
    adaptive = score_rank_vector(analyses, ranks, rate)
    return {
        "best_common": common,
        "adaptive": adaptive,
        "adaptive_not_worse_than_common": adaptive["F"] <= common["F"] + 1e-14,
    }


def role_moments(experts: list[dict[str, Any]], heldout_index: int | None) -> dict[str, tuple[float, float]]:
    selected = [x for i, x in enumerate(experts) if heldout_index is None or i != heldout_index]
    output: dict[str, tuple[float, float]] = {}
    for role in ROLES:
        count = sum(int(x["moments"][role]["count"]) for x in selected)
        total = sum(float(x["moments"][role]["sum"]) for x in selected)
        total_sq = sum(float(x["moments"][role]["sum_sq"]) for x in selected)
        mean = total / count
        variance = max(0.0, total_sq / count - mean * mean)
        output[role] = (mean, variance)
    return output


def gaussian_role_gram(rng: np.random.Generator, mean: float, variance: float) -> np.ndarray:
    matrix = rng.standard_normal((CHANNELS, WIDTH), dtype=np.float64)
    # Make the finite control match the requested first two moments exactly,
    # rather than only in expectation.
    matrix -= float(np.mean(matrix, dtype=np.float64))
    sample_variance = float(np.mean(matrix * matrix, dtype=np.float64))
    matrix *= math.sqrt(variance / sample_variance)
    matrix += mean
    gram = matrix @ matrix.T
    del matrix
    return gram


def gaussian_panel(
    experts: list[dict[str, Any]],
    *,
    seed: int,
    mode: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    analyses = []
    for index, expert in enumerate(experts):
        if mode == "exact_role_moment_matched":
            moments = {
                role: (float(expert["moments"][role]["mean"]), float(expert["moments"][role]["variance"]))
                for role in ROLES
            }
        elif mode == "leave_one_expert_out_role_moments":
            moments = role_moments(experts, index)
        else:
            raise ValueError(mode)
        grams = {
            role: gaussian_role_gram(rng, moments[role][0], moments[role][1])
            for role in ROLES
        }
        analyses.append(analyze_grams(grams, identity=f"gaussian_{index}", include_curve=True))
    return analyses


def permutation_bits() -> int:
    return int(math.ceil(math.lgamma(CHANNELS + 1) / math.log(2.0)))


def side_read_ledger(rate: float) -> dict[str, Any]:
    ideal_bytes = rate * PANEL_VALUES / 8.0
    if math.isclose(rate, MIN_RATE):
        container_bytes = math.ceil(ideal_bytes)
    elif math.isclose(rate, MAX_RATE):
        container_bytes = math.floor(ideal_bytes)
    else:
        container_bytes = round(ideal_bytes)
    # Keep six equal local frames and absorb the 0..5 remainder in a roughly
    # 4 KiB global manifest/directory.
    global_bytes = 4092 + ((container_bytes - 4092) % EXPERTS)
    frame_bytes = (container_bytes - global_bytes) // EXPERTS
    perm_bits = permutation_bits()
    local_objects = {
        "frame_header": 64,
        "gauge_fp32_768": 4 * CHANNELS,
        "canonical_permutation_enumerative": math.ceil(perm_bits / 8),
        "rank_label_u16": 2,
        "payload_directory": 32,
        "crc32": 4,
    }
    side_bytes = sum(local_objects.values())
    payload_bytes = frame_bytes - side_bytes
    if payload_bytes <= 0:
        raise AssertionError("side ledger exceeds frame")
    actual_rate = container_bytes * 8.0 / PANEL_VALUES
    equal_share = container_bytes / EXPERTS
    cold_bytes = global_bytes + frame_bytes
    return {
        "requested_rate_bpw": rate,
        "actual_byte_derived_rate_bpw": actual_rate,
        "container_bytes": container_bytes,
        "global_manifest_directory_bytes": global_bytes,
        "equal_expert_frame_bytes": frame_bytes,
        "per_expert_side_objects_bytes": local_objects,
        "permutation_information_bits": perm_bits,
        "per_expert_side_bytes": side_bytes,
        "per_expert_payload_bytes": payload_bytes,
        "equal_physical_share_bytes": equal_share,
        "cold_requested_expert_read_bytes": cold_bytes,
        "cold_read_amplification": cold_bytes / equal_share,
        "cached_global_header_read_bytes": frame_bytes,
        "cached_read_amplification": frame_bytes / equal_share,
        "below_strict_2x_cold": cold_bytes / equal_share < 2.0,
        "scope_note": (
            "Exact byte layout proposal only; FP32 gauge distortion and a finite manifold code are "
            "not implemented. The information oracle gives the gauge and charts losslessly."
        ),
    }


def compact_control(score: dict[str, Any]) -> dict[str, Any]:
    adaptive = score["adaptive"]
    common = score["best_common"]
    return {
        "adaptive": {
            "F": adaptive["F"],
            "s_bpw": adaptive["s_bpw"],
            "ideal_relative_mse": adaptive["ideal_relative_mse"],
            "ranks": adaptive["ranks"],
        },
        "best_common": {
            "F": common["F"],
            "s_bpw": common["s_bpw"],
            "ideal_relative_mse": common["ideal_relative_mse"],
            "rank": common["ranks"][0],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="workspace root containing blind_protocol_v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaussian-replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0x51A17E)
    args = parser.parse_args()
    if args.gaussian_replicates < 1:
        raise ValueError("at least one matched Gaussian replicate is required")
    started = time.time()
    lock_path, lock, experts, receipts = load_panel(args.root.resolve())
    source_analyses = [
        analyze_grams(
            expert["role_grams"],
            identity=f"L{expert['layer']}:E{expert['expert']}",
            include_curve=True,
        )
        for expert in experts
    ]
    source_scores = {str(rate): optimize_ranks(source_analyses, rate) for rate in RATES}

    controls: dict[str, list[dict[str, Any]]] = {
        "exact_role_moment_matched": [],
        "leave_one_expert_out_role_moments": [],
    }
    for mode_index, mode in enumerate(controls):
        for replicate in range(args.gaussian_replicates):
            seed = args.seed + 1000003 * mode_index + 7919 * replicate
            panel = gaussian_panel(experts, seed=seed, mode=mode)
            rate_scores = {str(rate): compact_control(optimize_ranks(panel, rate)) for rate in RATES}
            controls[mode].append({"replicate": replicate, "seed": seed, "scores": rate_scores})

    source_s = max(float(source_scores[str(rate)]["adaptive"]["s_bpw"]) for rate in RATES)
    control_s_values = [
        float(rep["scores"][str(rate)]["adaptive"]["s_bpw"])
        for rows in controls.values()
        for rep in rows
        for rate in RATES
    ]
    matched_mean_s = float(np.mean(control_s_values))
    calibrated_excess = source_s - matched_mean_s
    decision = "SURVIVE_INFORMATION_GATE" if source_s >= TARGET_S else "KILL_INVARIANT_MANIFOLD_BRANCH"

    # Remove heavy ndarray state and retain JSON-native source moments.
    expert_moments = [
        {
            "layer": int(x["layer"]),
            "expert": int(x["expert"]),
            "moments": x["moments"],
        }
        for x in experts
    ]
    output = {
        "schema": "qwen_swiglu_gauge_coupled_polar_oracle_v1",
        "status": "complete_cpu_only_information_oracle",
        "decision": decision,
        "target": {
            "rate_interval_bpw": [MIN_RATE, MAX_RATE],
            "F_max": TARGET_F,
            "required_s_bpw": TARGET_S,
            "required_mse_at_2p15": TARGET_F * 2.0 ** (-2.0 * MIN_RATE),
            "required_mse_at_2p5": TARGET_F * 2.0 ** (-2.0 * MAX_RATE),
            "weights": PANEL_VALUES,
            "experts": EXPERTS,
        },
        "protocol": {
            "exact_function_symmetry": "U'=A U, D'^T=A^-1 D^T plus shared neuron permutation",
            "canonical_gauge": "a_j=sqrt(||down_j||/||up_j||), so ||a_j up_j||=||down_j/a_j||",
            "joint_matrix": "X=[gate, A up, A^-1 down.T] in R^(768x6144)",
            "manifold": "X=H Q, QQ^T=I, H=cI+A_k with symmetric rank-k A_k",
            "source_metric": "exact inverse-gauge quadratic error against all original BF16 weights",
            "rank_search": (
                "all ranks 0..766; all contiguous unmodelled eigenspectrum windows; common c solved "
                "analytically in the inverse-gauge source metric; adaptive ranks coordinate-minimized"
            ),
            "favourable_grants": [
                "continuous manifold coordinates",
                "ideal asymptotic Gaussian reverse-waterfilling",
                "lossless free gauge values in the information gate",
                "free charts/eigenpair order and exact source-specific ranks",
                "orthogonal component accounting at the measured source residual",
            ],
            "nonduplication": (
                "Unlike the prior per-matrix Stiefel screen, one polar factor couples Gate, Up and "
                "Down after quotienting the exact SwiGLU scaling action. Unlike the permutation "
                "predictor, no other expert's weights are used as a reference."
            ),
            "gaussian_controls": {
                "exact_role_moment_matched": "independent Gaussian roles with each target role's mean/variance",
                "leave_one_expert_out_role_moments": "each held-out control uses role moments from the other five experts",
            },
        },
        "dimension_ledger": {
            "ambient_values_per_expert": VALUES_PER_EXPERT,
            "joint_row_stiefel_dof": STIEFEL_DOF,
            "symmetric_polar_dof": SYMMETRIC_DOF,
            "gauge_constraints": CHANNELS,
            "retained_gauge_coordinates": CHANNELS,
            "net_gauge_dimension_change": 0,
            "explanation": (
                "Equal-norm canonicalization removes 768 continuous dimensions, but retaining the "
                "768 gauge coordinates needed for source reconstruction adds them back."
            ),
        },
        "source_binding": {
            "lock_path": str(lock_path),
            "lock_file_sha256": sha256_file(lock_path),
            "lock_internal_sha256": lock.get("lock_sha256"),
            "checkpoint": lock.get("checkpoint"),
            "all_source_hashes_match": True,
            "receipts": receipts,
            "expert_role_moments": expert_moments,
        },
        "source_analyses": source_analyses,
        "source_scores": source_scores,
        "gaussian_controls": controls,
        "calibration_summary": {
            "maximum_source_s_over_rate_grid_bpw": source_s,
            "matched_control_mean_s_over_families_replicates_rates_bpw": matched_mean_s,
            "source_minus_matched_control_s_bpw": calibrated_excess,
            "fraction_of_required_s_raw": source_s / TARGET_S,
            "fraction_of_required_s_after_control_subtraction": calibrated_excess / TARGET_S,
            "early_kill_margin_bpw": TARGET_S - source_s,
        },
        "side_and_read_ledger": {str(rate): side_read_ledger(rate) for rate in RATES},
        "claim_boundary": (
            "No compressed artifact is emitted and no achieved MSE/rate claim is made. Failure of "
            "the more favourable continuous oracle stops finite coding and GPU work. The byte/read "
            "ledger is a reproducible expert-local layout proposal, not an encoded container."
        ),
        "audit": {
            "elapsed_seconds": time.time() - started,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pid": os.getpid(),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "source_max_s_bpw": source_s,
        "required_s_bpw": TARGET_S,
        "calibrated_excess_s_bpw": calibrated_excess,
        "output": str(args.output),
        "elapsed_seconds": output["audit"]["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
