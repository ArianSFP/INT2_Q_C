#!/usr/bin/env python3
"""Source-locked CPU oracle for Stiefel/Gram structure in the Qwen panel.

This is deliberately not a codec.  It asks whether the polar factors of each
768 x 2048 expert matrix have enough predictable normal (Gram) structure to
remove the 0.160964 bpw of entropy-equivalent rate required by the experiment.

Every scored model is split into its exact manifold dimension and orthogonal
normal dimension.  Both parts receive an *ideal* two-component Gaussian
reverse-waterfill at the requested physical rate.  This is substantially more
favourable than any finite-block PTQ implementation and therefore serves as an
early-kill oracle.  Shared tables are scored both for free and with explicit
FP16 serialization costs.
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
from typing import Any

import numpy as np


ROWS = 768
COLS = 2048
VALUES = ROWS * COLS
PANEL_MATRICES = 18
QWEN_LAYERS = 48
QWEN_EXPERTS = 128
QWEN_ROLES = 3
TARGET_RATE = 2.5
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
STIEFEL_DOF = ROWS * COLS - ROWS * (ROWS + 1) // 2


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def load_lock(root: Path) -> tuple[Path, dict[str, Any]]:
    lock_path = root / "blind_protocol_v2/unblinded/source_hashes.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if len(lock["matrices"]) != PANEL_MATRICES:
        raise AssertionError(f"expected {PANEL_MATRICES} matrices")
    return lock_path, lock


def load_canonical(source_root: Path, row: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    path = source_root / row["output_relpath"]
    actual_hash = sha256_file(path)
    declared_hash = str(row["source_bf16_sha256"])
    if actual_hash != declared_hash:
        raise AssertionError(f"source hash mismatch: {path}")
    words = np.fromfile(path, dtype="<u2")
    shape = tuple(int(v) for v in row["shape"])
    if words.size != math.prod(shape):
        raise AssertionError((path, words.size, shape))
    matrix = (words.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if row["role"] == "down":
        matrix = matrix.T
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    if matrix.shape != (ROWS, COLS):
        raise AssertionError((path, matrix.shape))
    return matrix, {
        "path_relative_to_source_lock_parent": str(path.relative_to(source_root)).replace("\\", "/"),
        "declared_sha256": declared_hash,
        "observed_sha256": actual_hash,
        "nbytes": path.stat().st_size,
    }


def reverse_waterfill_components(
    components: list[dict[str, float]],
    *,
    rate: float = TARGET_RATE,
    shared_side_bpw: float = 0.0,
) -> dict[str, Any]:
    """Ideal Gaussian reverse-waterfill for arbitrary orthogonal components."""
    payload_rate = rate - shared_side_bpw
    if payload_rate <= 0.0:
        return {
            "physical_rate_bpw": rate,
            "shared_side_bpw": shared_side_bpw,
            "payload_rate_bpw": payload_rate,
            "valid": False,
            "reason": "shared table consumes the rate budget",
        }
    dimensions = [float(x["dimension_fraction"]) for x in components]
    energies = [float(x["energy_fraction"]) for x in components]
    names = [str(x.get("component", f"component_{i}")) for i, x in enumerate(components)]
    if any(x <= 0.0 for x in dimensions) or any(x <= 0.0 for x in energies):
        raise ValueError("component dimensions and energies must be positive")
    if not math.isclose(sum(dimensions), 1.0, rel_tol=2e-12, abs_tol=2e-12):
        raise ValueError(("dimension sum", sum(dimensions)))
    if not math.isclose(sum(energies), 1.0, rel_tol=2e-12, abs_tol=2e-12):
        raise ValueError(("energy sum", sum(energies)))
    variances = [e / d for d, e in zip(dimensions, energies)]
    pairs = list(zip(dimensions, variances))
    log_variances = [math.log2(x) for x in variances]

    def used_rate_at_log2_level(log_level: float) -> float:
        return 0.5 * sum(
            d * max(0.0, log_variance - log_level)
            for (d, _), log_variance in zip(pairs, log_variances)
        )

    lo = min(log_variances) - 2.0 * payload_rate / min(dimensions) - 16.0
    hi = max(log_variances)
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if used_rate_at_log2_level(mid) > payload_rate:
            lo = mid
        else:
            hi = mid
    water_level = 2.0**hi
    allocations = []
    distortion = 0.0
    for name, dimension, energy, variance in zip(names, dimensions, energies, variances):
        bits_per_active_dimension = 0.5 * max(0.0, math.log2(variance / water_level))
        contribution = dimension * min(variance, water_level)
        distortion += contribution
        allocations.append(
            {
                "component": name,
                "dimension_fraction": dimension,
                "energy_fraction": energy,
                "variance_per_dimension": variance,
                "bits_per_component_dimension": bits_per_active_dimension,
                "rate_bpw_contribution": dimension * bits_per_active_dimension,
                "distortion_contribution": contribution,
                "active": variance > water_level,
            }
        )
    f_value = distortion * 2.0 ** (2.0 * rate)
    s_value = -0.5 * math.log2(f_value)
    return {
        "physical_rate_bpw": rate,
        "shared_side_bpw": shared_side_bpw,
        "payload_rate_bpw": payload_rate,
        "valid": True,
        "component_count": len(components),
        "water_level_relative_mse": water_level,
        "ideal_relative_mse": distortion,
        "gaussian_reference_relative_mse": 2.0 ** (-2.0 * rate),
        "F": f_value,
        "rate_equivalent_s_bpw": s_value,
        "target_F": TARGET_F,
        "target_s_bpw": TARGET_S,
        "passes_target": f_value <= TARGET_F,
        "allocations": allocations,
    }


def reverse_waterfill_score(
    residual_ratio: float,
    model_dof: float,
    *,
    rate: float = TARGET_RATE,
    shared_side_bpw: float = 0.0,
) -> dict[str, Any]:
    """Two-component wrapper used for structural-only comparisons."""
    if not 0.0 < residual_ratio < 1.0:
        raise ValueError(residual_ratio)
    if not 0 < model_dof < VALUES:
        raise ValueError(model_dof)
    d_model = model_dof / VALUES
    d_normal = 1.0 - d_model
    score = reverse_waterfill_components(
        [
            {
                "component": "manifold",
                "dimension_fraction": d_model,
                "energy_fraction": 1.0 - residual_ratio,
            },
            {
                "component": "normal",
                "dimension_fraction": d_normal,
                "energy_fraction": residual_ratio,
            },
        ],
        rate=rate,
        shared_side_bpw=shared_side_bpw,
    )
    score.update(
        {
            "model_dof": model_dof,
            "normal_dof": VALUES - model_dof,
            "model_dimension_fraction": d_model,
            "normal_dimension_fraction": d_normal,
            "residual_energy_ratio": residual_ratio,
        }
    )
    return score


def polar_left_diagonal_optimum(
    gram: np.ndarray,
    polar_h: np.ndarray,
    energy: float,
    max_iterations: int,
    tolerance: float,
) -> dict[str, Any]:
    """Globally optimize D and Procrustes Q for W ~= DQ.

    The iterations use only G=W W^T.  For fixed D, Q is the exact polar factor
    of D W.  For fixed Q, each d_i=<w_i,q_i> is exact.  This is a monotone
    fixed-point solver initialized by the polar-diagonal projection.  In the
    squared scales x_i=d_i^2, the eliminated objective is

        ||W||_F^2 + sum(x) - 2 tr(sqrt(sqrt(diag(x)) G sqrt(diag(x)))).

    Matrix fidelity is concave in diag(x), so this objective is convex.  A
    positive fixed point is therefore a global optimum, not merely a local
    alternating minimum.
    """
    scales = np.diag(polar_h).copy()
    initial_residual = float((energy - np.dot(scales, scales)) / energy)
    best = initial_residual
    history = [initial_residual]
    converged = False
    for iteration in range(1, max_iterations + 1):
        scaled_gram = (scales[:, None] * gram) * scales[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(scaled_gram)
        floor = max(float(eigenvalues[-1]) * 1e-15, np.finfo(np.float64).tiny)
        invsqrt = (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, floor)))) @ eigenvectors.T
        q_times_wt = invsqrt @ (scales[:, None] * gram)
        updated = np.diag(q_times_wt).copy()
        residual = float(max(0.0, energy - np.dot(updated, updated)) / energy)
        if residual > history[-1] + 2e-11:
            raise AssertionError(("non-monotone DQ alternating solver", history[-1], residual))
        history.append(residual)
        best = min(best, residual)
        relative_change = float(np.max(np.abs(updated - scales)) / max(np.mean(np.abs(scales)), 1e-30))
        scales = updated
        if relative_change <= tolerance:
            converged = True
            break
    return {
        "initial_polar_diagonal_residual_ratio": initial_residual,
        "optimized_residual_ratio": best,
        "iterations": iteration,
        "converged": converged,
        "kkt_relative_fixed_point_residual": relative_change,
        "global_optimality_basis": (
            "convexity in squared scales from concavity of matrix fidelity; "
            "positive fixed-point KKT residual reported above"
        ),
        "monotone_history": history,
    }


def matrix_analysis(matrix: np.ndarray, metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if float(eigenvalues[0]) <= 0.0:
        raise AssertionError("Gram is not positive definite")
    singular = np.sqrt(eigenvalues)
    energy = float(np.sum(eigenvalues))
    polar_h = (eigenvectors * singular) @ eigenvectors.T
    reconstruction_error = float(np.linalg.norm(polar_h @ polar_h - gram, ord="fro") / np.linalg.norm(gram, ord="fro"))
    if reconstruction_error > 2e-12:
        raise AssertionError(reconstruction_error)

    tight_scale = float(np.mean(singular))
    tight_residual = float(np.sum(np.square(singular - tight_scale)) / energy)
    left = polar_left_diagonal_optimum(
        gram, polar_h, energy, args.max_iterations, args.tolerance
    )

    # A second diagonal screen: keep the exact polar Q and independently scale
    # its 2048 columns.  This is the exact projection for that fixed Q, though
    # unlike the left-DQ result it is not advertised as a joint global optimum.
    inv_h = (eigenvectors * (1.0 / singular)) @ eigenvectors.T
    polar_q = inv_h @ matrix
    column_q_energy = np.sum(np.square(polar_q), axis=0)
    column_dot = np.sum(matrix * polar_q, axis=0)
    right_residual = float(
        max(0.0, energy - np.sum(np.square(column_dot) / column_q_energy)) / energy
    )
    row_norms = np.sqrt(np.diag(gram))
    gram_sq = float(np.sum(np.square(gram)))
    stable_rank = energy * energy / gram_sq
    probabilities = eigenvalues / energy
    spectral_entropy_rank = float(math.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "metadata": metadata,
        "energy": energy,
        "gram_method": "exact FP64 W@W.T followed by exact symmetric eigh",
        "gram_sqrt_relative_reconstruction_error": reconstruction_error,
        "singular_values_ascending": singular.tolist(),
        "tight_frame": {
            "scale": tight_scale,
            "residual_energy_ratio": tight_residual,
        },
        "left_diagonal_scaled_frame": left,
        "polar_fixed_right_diagonal_scaled_frame": {
            "residual_energy_ratio": right_residual,
            "scope": "exact column-scale projection for the source polar Q; no joint Q reoptimization",
        },
        "gram_diagnostics": {
            "minimum_singular_value": float(singular[0]),
            "maximum_singular_value": float(singular[-1]),
            "singular_condition_number": float(singular[-1] / singular[0]),
            "stable_rank": float(stable_rank),
            "stable_rank_fraction": float(stable_rank / ROWS),
            "spectral_entropy_rank": spectral_entropy_rank,
            "spectral_entropy_rank_fraction": spectral_entropy_rank / ROWS,
            "gram_sphericity": float(stable_rank / ROWS),
            "gram_offdiagonal_energy_fraction": float(
                (gram_sq - np.dot(np.diag(gram), np.diag(gram))) / gram_sq
            ),
            "row_norm_coefficient_of_variation": float(np.std(row_norms) / np.mean(row_norms)),
        },
        # Retained only in memory by the caller; removed before JSON emission.
        "_arrays": {
            "H": polar_h,
            "U": eigenvectors,
            "s": singular,
        },
    }


def normalized_mean(arrays: list[np.ndarray]) -> np.ndarray:
    mean = np.mean([x / np.linalg.norm(x) for x in arrays], axis=0)
    return mean / np.linalg.norm(mean)


def scaled_template_residual(target: np.ndarray, template: np.ndarray, energy: float) -> tuple[float, float]:
    scale = float(np.sum(target * template))
    residual = float(max(0.0, energy - scale * scale) / energy)
    return residual, scale


def diagonal_basis_residual(h: np.ndarray, basis: np.ndarray, energy: float) -> float:
    diagonal = np.sum(basis * (h @ basis), axis=0)
    return float(max(0.0, energy - np.dot(diagonal, diagonal)) / energy)


def weighted_aggregate(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(x["energy"] * x[field] for x in rows) / sum(x["energy"] for x in rows))


def table_bpw(entries: int, bits: int, denominator_values: int) -> float:
    return bits * entries / denominator_values


def side_cost_ledger() -> dict[str, Any]:
    panel_values = PANEL_MATRICES * VALUES
    model_values = QWEN_LAYERS * QWEN_EXPERTS * QWEN_ROLES * VALUES
    symmetric = ROWS * (ROWS + 1) // 2
    orthogonal_angles = ROWS * (ROWS - 1) // 2
    tables = {
        "one_global_spectrum": ROWS,
        "three_role_spectra": QWEN_ROLES * ROWS,
        "one_global_symmetric_H_template": symmetric,
        "three_role_symmetric_H_templates": QWEN_ROLES * symmetric,
        "one_global_eigenbasis_minimal_angles": orthogonal_angles,
        "three_role_eigenbases_minimal_angles": QWEN_ROLES * orthogonal_angles,
        "one_global_eigenbasis_dense": ROWS * ROWS,
        "three_role_eigenbases_dense": QWEN_ROLES * ROWS * ROWS,
    }
    return {
        "bits_per_table_scalar": 16,
        "panel_denominator_values": panel_values,
        "full_qwen_moe_denominator_values": model_values,
        "full_qwen_geometry_assumption": {
            "layers": QWEN_LAYERS,
            "experts_per_layer": QWEN_EXPERTS,
            "roles": QWEN_ROLES,
            "values_per_matrix": VALUES,
        },
        "tables": {
            name: {
                "serialized_fp16_scalars": count,
                "serialized_bits": 16 * count,
                "panel_local_bpw": table_bpw(count, 16, panel_values),
                "full_model_amortized_bpw": table_bpw(count, 16, model_values),
            }
            for name, count in tables.items()
        },
        "note": (
            "Minimal-angle basis costs are optimistic degrees-of-freedom lower bounds. "
            "Dense-basis costs are the straightforward physical representation."
        ),
    }


def panel_component_score(
    records: list[dict[str, Any]],
    residual_ratios: list[float],
    model_dofs: list[float],
    *,
    shared_side_bpw: float = 0.0,
) -> dict[str, Any]:
    """Give every matrix manifold/normal pair its own ideal rate allocation."""
    if not (len(records) == len(residual_ratios) == len(model_dofs) == PANEL_MATRICES):
        raise AssertionError((len(records), len(residual_ratios), len(model_dofs)))
    total_energy = sum(float(x["energy"]) for x in records)
    components: list[dict[str, float]] = []
    for row, residual, dof in zip(records, residual_ratios, model_dofs):
        energy_fraction = float(row["energy"]) / total_energy
        ordinal = int(row["matrix_ordinal"])
        components.extend(
            [
                {
                    "component": f"matrix_{ordinal:02d}_manifold",
                    "dimension_fraction": float(dof) / (PANEL_MATRICES * VALUES),
                    "energy_fraction": energy_fraction * (1.0 - residual),
                },
                {
                    "component": f"matrix_{ordinal:02d}_normal",
                    "dimension_fraction": (VALUES - float(dof)) / (PANEL_MATRICES * VALUES),
                    "energy_fraction": energy_fraction * residual,
                },
            ]
        )
    return reverse_waterfill_components(
        components, shared_side_bpw=shared_side_bpw
    )


def isotropic_plus_symmetric_rank_curve(singular: np.ndarray) -> list[dict[str, Any]]:
    """Exact best H ~= cI + A_k curve for every feasible symmetric rank k.

    For fixed k, the k explicitly represented eigenpairs may be arbitrary.  The
    other ROWS-k singular values share one scalar c.  In one dimension the
    minimum-variance retained set is a contiguous sorted window, so exhaustive
    prefix-sum windows give the global optimum without approximation.
    """
    prefix = np.concatenate(([0.0], np.cumsum(singular, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(np.square(singular), dtype=np.float64)))
    curve: list[dict[str, Any]] = []
    for rank in range(ROWS - 1):
        unmodeled = ROWS - rank
        sums = prefix[unmodeled:] - prefix[:-unmodeled]
        sums_sq = prefix_sq[unmodeled:] - prefix_sq[:-unmodeled]
        errors = np.maximum(0.0, sums_sq - np.square(sums) / unmodeled)
        start = int(np.argmin(errors))
        symmetric_rank_dof = ROWS * rank - rank * (rank - 1) // 2
        model_dof = STIEFEL_DOF + 1 + symmetric_rank_dof
        if model_dof >= VALUES:
            break
        curve.append(
            {
                "rank": rank,
                "model_dof": model_dof,
                "normal_dof": VALUES - model_dof,
                "unmodeled_window_start": start,
                "unmodeled_window_stop": start + unmodeled,
                "unmodeled_common_scale": float(sums[start] / unmodeled),
                "residual_energy": float(errors[start]),
            }
        )
    return curve


def structured_gram_rank_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Screen source-specific low-rank polar-normal manifolds exactly."""
    curves = [isotropic_plus_symmetric_rank_curve(x["s"]) for x in records]
    common_length = min(len(x) for x in curves)
    total_energy = sum(float(x["energy"]) for x in records)
    common_curve: list[dict[str, Any]] = []
    for rank in range(common_length):
        residuals = [curves[i][rank]["residual_energy"] / records[i]["energy"] for i in range(PANEL_MATRICES)]
        dof = int(curves[0][rank]["model_dof"])
        aggregate_residual = sum(curves[i][rank]["residual_energy"] for i in range(PANEL_MATRICES)) / total_energy
        structural_score = reverse_waterfill_score(aggregate_residual, dof)
        panel_score = panel_component_score(records, residuals, [dof] * PANEL_MATRICES)
        common_curve.append(
            {
                "rank": rank,
                "model_dof": dof,
                "normal_dof": VALUES - dof,
                "residual_energy_ratio": aggregate_residual,
                "structural_only_F": structural_score["F"],
                "structural_only_s_bpw": structural_score["rate_equivalent_s_bpw"],
                "panel_component_F": panel_score["F"],
                "panel_component_s_bpw": panel_score["rate_equivalent_s_bpw"],
            }
        )
    best_common = min(common_curve, key=lambda x: x["panel_component_F"])
    common_rank = int(best_common["rank"])
    common_residuals = [curves[i][common_rank]["residual_energy"] / records[i]["energy"] for i in range(PANEL_MATRICES)]
    common_dof = int(curves[0][common_rank]["model_dof"])

    # When every component is active, log2(F) is a sum of independent component
    # contributions.  Choose each per-matrix rank by its exact contribution,
    # then confirm the resulting allocation with the general waterfiller.
    selections: list[dict[str, Any]] = []
    total_values = PANEL_MATRICES * VALUES
    for matrix_index, (record, curve) in enumerate(zip(records, curves)):
        candidates = []
        for row in curve:
            dof = float(row["model_dof"])
            residual_energy = float(row["residual_energy"])
            dm = dof / total_values
            dn = (VALUES - dof) / total_values
            em = (float(record["energy"]) - residual_energy) / total_energy
            en = residual_energy / total_energy
            contribution = dm * math.log2(em / dm) + dn * math.log2(en / dn)
            candidates.append((contribution, row))
        contribution, chosen = min(candidates, key=lambda x: x[0])
        selections.append(
            {
                "matrix_ordinal": int(record["matrix_ordinal"]),
                "layer": int(record["layer"]),
                "expert": int(record["expert"]),
                "role": str(record["role"]),
                "rank": int(chosen["rank"]),
                "model_dof": int(chosen["model_dof"]),
                "normal_dof": int(chosen["normal_dof"]),
                "residual_energy": float(chosen["residual_energy"]),
                "residual_energy_ratio": float(chosen["residual_energy"] / record["energy"]),
                "unmodeled_window_start": int(chosen["unmodeled_window_start"]),
                "unmodeled_window_stop": int(chosen["unmodeled_window_stop"]),
                "log2_F_contribution_if_all_components_active": contribution,
            }
        )
    adaptive_residuals = [x["residual_energy_ratio"] for x in selections]
    adaptive_dofs = [x["model_dof"] for x in selections]
    adaptive_panel_score = panel_component_score(records, adaptive_residuals, adaptive_dofs)
    if not all(x["active"] for x in adaptive_panel_score["allocations"]):
        raise AssertionError("adaptive closed-form selection encountered an inactive component")
    adaptive_residual = sum(x["residual_energy"] for x in selections) / total_energy
    adaptive_average_dof = sum(adaptive_dofs) / PANEL_MATRICES
    adaptive_structural_score = reverse_waterfill_score(adaptive_residual, adaptive_average_dof)
    rank_label_side_bpw = math.ceil(math.log2(ROWS)) / VALUES
    adaptive_labeled_score = panel_component_score(
        records,
        adaptive_residuals,
        adaptive_dofs,
        shared_side_bpw=rank_label_side_bpw,
    )
    return {
        "model": "H = c I + A_k, with source-specific symmetric rank-k A_k, followed by exact polar Q",
        "global_optimality": (
            "For every k and matrix, exhaustive contiguous-window variance gives the exact best "
            "spectrum: k eigenpairs are explicit and all remaining eigenvalues share c."
        ),
        "common_rank_curve": common_curve,
        "best_common_rank": {
            **best_common,
            "structural_only_score": reverse_waterfill_score(best_common["residual_energy_ratio"], common_dof),
            "panel_component_score": panel_component_score(
                records, common_residuals, [common_dof] * PANEL_MATRICES
            ),
        },
        "adaptive_rank": {
            "selections": selections,
            "minimum_rank": min(x["rank"] for x in selections),
            "maximum_rank": max(x["rank"] for x in selections),
            "mean_rank": sum(x["rank"] for x in selections) / PANEL_MATRICES,
            "average_model_dof": adaptive_average_dof,
            "aggregate_residual_energy_ratio": adaptive_residual,
            "structural_only_score": adaptive_structural_score,
            "panel_component_score_rank_labels_free": adaptive_panel_score,
            "rank_label_bits_per_matrix": math.ceil(math.log2(ROWS)),
            "rank_label_side_bpw": rank_label_side_bpw,
            "panel_component_score_rank_labels_charged": adaptive_labeled_score,
        },
    }


def heldout_analysis(records: list[dict[str, Any]], ledger: dict[str, Any]) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    identities = sorted({(x["layer"], x["expert"]) for x in records})
    for identity in identities:
        training = [x for x in records if (x["layer"], x["expert"]) != identity]
        targets = [x for x in records if (x["layer"], x["expert"]) == identity]
        if len(targets) != QWEN_ROLES or len(training) != PANEL_MATRICES - QWEN_ROLES:
            raise AssertionError((identity, len(training), len(targets)))
        for target in targets:
            same_role = [x for x in training if x["role"] == target["role"]]
            if len(same_role) != 5:
                raise AssertionError((identity, target["role"], len(same_role)))
            target_h = target["H"]
            target_s = target["s"]
            energy = target["energy"]

            role_spectrum = normalized_mean([x["s"] for x in same_role])
            global_spectrum = normalized_mean([x["s"] for x in training])
            role_spec_residual, role_spec_scale = scaled_template_residual(target_s, role_spectrum, energy)
            global_spec_residual, global_spec_scale = scaled_template_residual(target_s, global_spectrum, energy)

            role_h = normalized_mean([x["H"] for x in same_role])
            global_h = normalized_mean([x["H"] for x in training])
            role_h_residual, role_h_scale = scaled_template_residual(target_h, role_h, energy)
            global_h_residual, global_h_scale = scaled_template_residual(target_h, global_h, energy)

            _, role_basis = np.linalg.eigh(role_h)
            _, global_basis = np.linalg.eigh(global_h)
            role_basis_residual = diagonal_basis_residual(target_h, role_basis, energy)
            global_basis_residual = diagonal_basis_residual(target_h, global_basis, energy)

            candidate_role = [
                (diagonal_basis_residual(target_h, x["U"], energy), x)
                for x in same_role
            ]
            candidate_global = [
                (diagonal_basis_residual(target_h, x["U"], energy), x)
                for x in training
            ]
            best_role_residual, best_role_source = min(candidate_role, key=lambda x: x[0])
            best_global_residual, best_global_source = min(candidate_global, key=lambda x: x[0])
            outputs.append(
                {
                    "matrix_ordinal": target["matrix_ordinal"],
                    "layer": target["layer"],
                    "expert": target["expert"],
                    "role": target["role"],
                    "energy": energy,
                    "holdout_unit": {
                        "layer": identity[0],
                        "expert": identity[1],
                        "excluded_roles": ["gate", "up", "down"],
                    },
                    "training_matrix_count": len(training),
                    "same_role_training_matrix_count": len(same_role),
                    "role_spectral_template": {
                        "residual_energy_ratio": role_spec_residual,
                        "fitted_target_scale": role_spec_scale,
                    },
                    "global_spectral_template": {
                        "residual_energy_ratio": global_spec_residual,
                        "fitted_target_scale": global_spec_scale,
                    },
                    "role_H_template": {
                        "residual_energy_ratio": role_h_residual,
                        "fitted_target_scale": role_h_scale,
                    },
                    "global_H_template": {
                        "residual_energy_ratio": global_h_residual,
                        "fitted_target_scale": global_h_scale,
                    },
                    "role_mean_eigenbasis": {"residual_energy_ratio": role_basis_residual},
                    "global_mean_eigenbasis": {"residual_energy_ratio": global_basis_residual},
                    "best_training_role_eigenbasis_target_aware_oracle": {
                        "residual_energy_ratio": best_role_residual,
                        "selected_source_matrix_ordinal": best_role_source["matrix_ordinal"],
                        "target_selection_is_not_decoder_causal": True,
                    },
                    "best_training_global_eigenbasis_target_aware_oracle": {
                        "residual_energy_ratio": best_global_residual,
                        "selected_source_matrix_ordinal": best_global_source["matrix_ordinal"],
                        "target_selection_is_not_decoder_causal": True,
                    },
                }
            )

    methods = {
        "role_spectral_template": {
            "model_dof": VALUES - ROWS + 1,
            "table": "three_role_spectra",
        },
        "global_spectral_template": {
            "model_dof": VALUES - ROWS + 1,
            "table": "one_global_spectrum",
        },
        "role_H_template": {
            "model_dof": STIEFEL_DOF + 1,
            "table": "three_role_symmetric_H_templates",
        },
        "global_H_template": {
            "model_dof": STIEFEL_DOF + 1,
            "table": "one_global_symmetric_H_template",
        },
        "role_mean_eigenbasis": {
            "model_dof": STIEFEL_DOF + ROWS,
            "table": "three_role_eigenbases_minimal_angles",
        },
        "global_mean_eigenbasis": {
            "model_dof": STIEFEL_DOF + ROWS,
            "table": "one_global_eigenbasis_minimal_angles",
        },
        "best_training_role_eigenbasis_target_aware_oracle": {
            "model_dof": STIEFEL_DOF + ROWS,
            "table": None,
            "selection_oracle": True,
        },
        "best_training_global_eigenbasis_target_aware_oracle": {
            "model_dof": STIEFEL_DOF + ROWS,
            "table": None,
            "selection_oracle": True,
        },
    }
    aggregates: dict[str, Any] = {}
    total_energy = sum(x["energy"] for x in outputs)
    output_by_ordinal = {int(x["matrix_ordinal"]): x for x in outputs}
    for method, spec in methods.items():
        residual = sum(
            x["energy"] * x[method]["residual_energy_ratio"] for x in outputs
        ) / total_energy
        uncharged = reverse_waterfill_score(residual, spec["model_dof"])
        ordered_residuals = [
            float(output_by_ordinal[int(x["matrix_ordinal"])][method]["residual_energy_ratio"])
            for x in records
        ]
        panel_uncharged = panel_component_score(
            records,
            ordered_residuals,
            [spec["model_dof"]] * PANEL_MATRICES,
        )
        item: dict[str, Any] = {
            "residual_energy_ratio": residual,
            "model_dof": spec["model_dof"],
            "normal_dof": VALUES - spec["model_dof"],
            "shared_table": spec["table"],
            "optimistic_shared_table_free_score": uncharged,
            "optimistic_panel_component_free_score": panel_uncharged,
        }
        if spec["table"] is not None:
            costs = ledger["tables"][spec["table"]]
            item["panel_local_fp16_table_score"] = reverse_waterfill_score(
                residual,
                spec["model_dof"],
                shared_side_bpw=costs["panel_local_bpw"],
            )
            item["full_model_amortized_fp16_table_score"] = reverse_waterfill_score(
                residual,
                spec["model_dof"],
                shared_side_bpw=costs["full_model_amortized_bpw"],
            )
            item["panel_component_full_model_amortized_fp16_table_score"] = panel_component_score(
                records,
                ordered_residuals,
                [spec["model_dof"]] * PANEL_MATRICES,
                shared_side_bpw=costs["full_model_amortized_bpw"],
            )
        else:
            item["selection_oracle"] = True
        aggregates[method] = item
    return {
        "protocol": {
            "holdout": "leave one whole layer/expert triplet out; all gate/up/down matrices excluded together",
            "target_specific_scale": "one continuous scale counted in each local model dimension",
            "role_templates": "fit only the five same-role matrices from other held-out units",
            "global_templates": "fit the fifteen matrices from other held-out units",
            "target_aware_basis_oracles": "illegal selection ceilings, reported only to strengthen a negative result",
        },
        "matrices": outputs,
        "aggregates": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    started = time.time()
    root = args.root.resolve()
    lock_path, lock = load_lock(root)
    source_root = lock_path.parent

    internal: list[dict[str, Any]] = []
    public_matrices: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    for position, row in enumerate(lock["matrices"]):
        matrix, receipt = load_canonical(source_root, row)
        metadata = {
            "matrix_ordinal": int(row["matrix_ordinal"]),
            "layer": int(row["layer"]),
            "expert": int(row["expert"]),
            "role": str(row["role"]),
            "tensor": str(row["tensor"]),
        }
        analysis = matrix_analysis(matrix, metadata, args)
        arrays = analysis.pop("_arrays")
        internal.append({**metadata, "energy": analysis["energy"], **arrays})
        public_matrices.append(analysis)
        source_receipts.append({**metadata, **receipt})
        print(
            f"[{position + 1:02d}/{PANEL_MATRICES}] ordinal={metadata['matrix_ordinal']:02d} "
            f"{metadata['role']:4s} tight={analysis['tight_frame']['residual_energy_ratio']:.9f} "
            f"left-DQ={analysis['left_diagonal_scaled_frame']['optimized_residual_ratio']:.9f}",
            flush=True,
        )

    ledger = side_cost_ledger()
    heldout = heldout_analysis(internal, ledger)
    total_energy = sum(x["energy"] for x in public_matrices)
    tight_residual = sum(
        x["energy"] * x["tight_frame"]["residual_energy_ratio"] for x in public_matrices
    ) / total_energy
    left_residual = sum(
        x["energy"] * x["left_diagonal_scaled_frame"]["optimized_residual_ratio"]
        for x in public_matrices
    ) / total_energy
    right_residual = sum(
        x["energy"] * x["polar_fixed_right_diagonal_scaled_frame"]["residual_energy_ratio"]
        for x in public_matrices
    ) / total_energy
    tight_ratios = [float(x["tight_frame"]["residual_energy_ratio"]) for x in public_matrices]
    left_ratios = [float(x["left_diagonal_scaled_frame"]["optimized_residual_ratio"]) for x in public_matrices]
    right_ratios = [float(x["polar_fixed_right_diagonal_scaled_frame"]["residual_energy_ratio"]) for x in public_matrices]
    direct_models = {
        "nearest_scaled_tight_frame": {
            "residual_energy_ratio": tight_residual,
            "model_dof": STIEFEL_DOF + 1,
            "score": reverse_waterfill_score(tight_residual, STIEFEL_DOF + 1),
            "panel_component_score": panel_component_score(
                internal, tight_ratios, [STIEFEL_DOF + 1] * PANEL_MATRICES
            ),
        },
        "alternating_nearest_left_diagonal_scaled_frame": {
            "residual_energy_ratio": left_residual,
            "model_dof": STIEFEL_DOF + ROWS,
            "score": reverse_waterfill_score(left_residual, STIEFEL_DOF + ROWS),
            "panel_component_score": panel_component_score(
                internal, left_ratios, [STIEFEL_DOF + ROWS] * PANEL_MATRICES
            ),
        },
        "polar_fixed_right_diagonal_scaled_frame": {
            "residual_energy_ratio": right_residual,
            "model_dof": STIEFEL_DOF + COLS,
            "score": reverse_waterfill_score(right_residual, STIEFEL_DOF + COLS),
            "panel_component_score": panel_component_score(
                internal, right_ratios, [STIEFEL_DOF + COLS] * PANEL_MATRICES
            ),
            "scope": "fixed-polar exact column projection; not a jointly optimized QD model",
        },
    }
    structured_rank = structured_gram_rank_analysis(internal)
    energy_only_score = reverse_waterfill_components(
        [
            {
                "component": f"matrix_{int(x['matrix_ordinal']):02d}",
                "dimension_fraction": 1.0 / PANEL_MATRICES,
                "energy_fraction": float(x["energy"]) / sum(float(y["energy"]) for y in internal),
            }
            for x in internal
        ]
    )
    candidates = [
        (f"direct.{name}", item["panel_component_score"])
        for name, item in direct_models.items()
    ] + [
        (f"heldout.{name}", item["optimistic_panel_component_free_score"])
        for name, item in heldout["aggregates"].items()
    ] + [
        ("structured_gram.best_common_rank", structured_rank["best_common_rank"]["panel_component_score"]),
        (
            "structured_gram.adaptive_rank",
            structured_rank["adaptive_rank"]["panel_component_score_rank_labels_free"],
        ),
    ]
    best_name, best_score = min(candidates, key=lambda x: x[1]["F"])
    margin = TARGET_S - best_score["rate_equivalent_s_bpw"]

    script_path = Path(__file__).resolve()
    report = {
        "schema": "qwen-stiefel-gram-oracle-v1",
        "decision": {
            "status": "hard_kill" if margin > 0.0 else "survives",
            "best_optimistic_free_table_candidate": best_name,
            "best_optimistic_F_at_2p5": best_score["F"],
            "best_optimistic_rate_equivalent_s_bpw": best_score["rate_equivalent_s_bpw"],
            "required_F": TARGET_F,
            "required_rate_equivalent_s_bpw": TARGET_S,
            "shortfall_s_bpw": margin,
            "gpu_followup_warranted": margin <= 0.0,
            "rule": "hard-kill unless a source-locked optimistic oracle reaches s >= -0.5 log2(0.8)",
        },
        "scope": {
            "checkpoint": lock["checkpoint"],
            "matrix_count": PANEL_MATRICES,
            "canonical_shape": [ROWS, COLS],
            "values_per_matrix": VALUES,
            "physical_rate_bpw": TARGET_RATE,
            "gaussian_reference_relative_mse": 2.0 ** (-2.0 * TARGET_RATE),
            "target_relative_mse": TARGET_F * 2.0 ** (-2.0 * TARGET_RATE),
            "cpu_only": True,
            "cupy_imported": False,
        },
        "audit": {
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "source_lock_path": str(lock_path),
            "source_lock_file_sha256": sha256_file(lock_path),
            "source_lock_internal_sha256": lock.get("lock_sha256"),
            "all_source_hashes_matched": all(
                x["declared_sha256"] == x["observed_sha256"] for x in source_receipts
            ),
            "source_receipts": source_receipts,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "elapsed_seconds": time.time() - started,
        },
        "methodology": {
            "polar_identity": "W = H Q, H=(W W^T)^(1/2), Q Q^T=I",
            "stiefel_dof": STIEFEL_DOF,
            "tight_frame_model_dof": STIEFEL_DOF + 1,
            "left_diagonal_frame_model_dof": STIEFEL_DOF + ROWS,
            "fixed_spectrum_orbit_model_dof": VALUES - ROWS,
            "scoring": (
                "Exact orthogonal model/normal energies are assigned to exact dimensions, then both receive "
                "an ideal Gaussian reverse-waterfill. F=D/2^(-2R), s=-0.5log2(F)."
            ),
            "optimism": [
                "continuous manifold coordinates",
                "ideal asymptotic Gaussian RD for both components",
                "no finite-block, curvature, chart, index, scale, alignment, or decoder penalty",
                "the primary cross-method decision gives shared tables for free",
            ],
        },
        "side_cost_ledger": ledger,
        "panel_energy_only_gaussian_allocation_baseline": energy_only_score,
        "direct_frame_models": direct_models,
        "structured_gram_rank_models": structured_rank,
        "heldout_shared_models": heldout,
        "matrices": public_matrices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
