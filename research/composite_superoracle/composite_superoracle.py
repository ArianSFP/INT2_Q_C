#!/usr/bin/env python3
"""CPU-only nested super-oracle for the pinned 18-matrix Qwen panel.

This experiment composes three source structures without adding rate gains:

* an expert-local, source-metric-orthogonal Gate/Up/Down role KLT;
* equipopulous STRATA partitions of natural 2,048-value rows; and
* an exact polar/Stiefel manifold-versus-normal decomposition inside each
  current unit.

The partition is genuinely nested.  Role KLT is orthogonal in the requested
source Frobenius metric.  STRATA units have disjoint row support.  For a unit
``A`` the polar model ``H_hat Q`` and normal ``(H-H_hat)Q`` are orthogonal
because the shared singular value is the exact mean over its unmodelled
window.  Component energies and manifold dimensions are therefore passed to
one ideal Gaussian reverse-waterfiller; previously reported ``s`` values are
never summed.

This remains an optimistic information oracle, not a codec.  Continuous
manifold coordinates, charts, and ideal asymptotic Gaussian coding are free of
finite precision and curvature loss, although their dimensions are charged.
Explicit source-derived labels and selection indices are charged to the
physical rate.  Two clearly labelled source-leaky envelopes are also emitted
and kept separate from the charged decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROWS = 768
COLS = 2048
ROLES = 3
EXPERTS = 6
MATRICES = 18
STRATA = 8
ROWS_PER_STRATUM = ROWS // STRATA
VALUES_PER_MATRIX = ROWS * COLS
VALUES_PER_EXPERT = ROLES * VALUES_PER_MATRIX
PANEL_VALUES = MATRICES * VALUES_PER_MATRIX
RATES = (2.15, 2.30, 2.50)
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
GLOBAL_HEADER_BITS = 4096 * 8
ROLE_KLT_BITS_PER_EXPERT = 3 * 16
STRATA_LABEL_BITS_PER_MATRIX = ROWS * 3
FP16_BITS = 16


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def sha256_f64(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def write_sealed_json(path: Path, report: dict[str, Any]) -> None:
    clean = dict(report)
    clean.pop("result_lock_sha256", None)
    clean["result_lock_sha256"] = hashlib.sha256(canonical_json_bytes(clean)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def bf16_file(path: Path, shape: tuple[int, int]) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    if words.size != math.prod(shape):
        raise AssertionError((path, words.size, shape))
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not np.all(np.isfinite(values)):
        raise AssertionError(f"non-finite source: {path}")
    return values.reshape(shape)


def load_panel(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    lock_path = root / "blind_protocol_v2/unblinded/source_hashes.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if len(lock.get("matrices", [])) != MATRICES:
        raise AssertionError("source lock is not the pinned 18-matrix panel")
    source_root = lock_path.parent
    receipts: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    for expected_ordinal, row in enumerate(lock["matrices"]):
        if int(row["matrix_ordinal"]) != expected_ordinal:
            raise AssertionError("non-canonical source order")
        path = source_root / row["output_relpath"]
        observed = sha256_file(path)
        declared = str(row["source_bf16_sha256"])
        if observed != declared:
            raise AssertionError(f"source hash mismatch: {path}")
        shape = tuple(int(x) for x in row["shape"])
        matrix = bf16_file(path, shape)
        if str(row["role"]) == "down":
            matrix = matrix.T
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        if matrix.shape != (ROWS, COLS):
            raise AssertionError((path, matrix.shape))
        matrices.append(matrix)
        receipts.append(
            {
                "matrix_ordinal": expected_ordinal,
                "layer": int(row["layer"]),
                "expert": int(row["expert"]),
                "role": str(row["role"]),
                "tensor": str(row["tensor"]),
                "relative_path": str(path.relative_to(source_root)).replace("\\", "/"),
                "declared_sha256": declared,
                "observed_sha256": observed,
                "nbytes": path.stat().st_size,
            }
        )
    return lock_path, lock, receipts, matrices


def expert_triplets(matrices: list[np.ndarray]) -> list[np.ndarray]:
    if len(matrices) != MATRICES:
        raise AssertionError(len(matrices))
    return [np.stack(matrices[3 * i : 3 * i + 3], axis=0) for i in range(EXPERTS)]


def role_klt(triplet: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an exact orthogonal role innovation basis and its covariance."""
    flat = triplet.reshape(ROLES, -1)
    covariance = flat @ flat.T
    eigenvalues, basis = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = basis[:, order]
    # Resolve eigenvector sign without affecting the objective or spectra.
    for column in range(ROLES):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    transformed = (basis.T @ flat).reshape(triplet.shape)
    source_energy = float(np.sum(np.square(flat), dtype=np.float64))
    transformed_energy = float(np.sum(np.square(transformed), dtype=np.float64))
    if not math.isclose(source_energy, transformed_energy, rel_tol=2e-13, abs_tol=2e-10):
        raise AssertionError((source_energy, transformed_energy))
    return transformed, basis, eigenvalues


def gauge_audit(triplet: np.ndarray) -> dict[str, Any]:
    """Audit the exact SwiGLU Up/Down gauge without double-counting it.

    If Uc=aU and Dc=D/a, the inverse source metric is
    diag(1,a^-2,a^2).  Its square root maps (G,Uc,Dc) exactly back to
    (G,U,D), so an exact source-metric orthogonal role transform is simply the
    source-space KLT above.  The gauge is therefore recorded but not claimed
    as an extra gain.
    """
    up_norm = np.linalg.norm(triplet[1], axis=1)
    down_norm = np.linalg.norm(triplet[2], axis=1)
    tiny = np.finfo(np.float64).tiny
    gauge = np.sqrt(np.maximum(down_norm, tiny) / np.maximum(up_norm, tiny))
    cancellation = np.maximum(
        np.abs((gauge * gauge ** -1) - 1.0),
        np.abs((gauge ** -1 * gauge) - 1.0),
    )
    return {
        "minimum": float(np.min(gauge)),
        "maximum": float(np.max(gauge)),
        "mean_log": float(np.mean(np.log(gauge))),
        "std_log": float(np.std(np.log(gauge))),
        "maximum_source_metric_cancellation_error": float(np.max(cancellation)),
        "coordinates": ROWS,
        "counted_as_additional_gain": False,
    }


def stratum_indices(matrix: np.ndarray) -> list[np.ndarray]:
    energies = np.sum(np.square(matrix), axis=1, dtype=np.float64)
    order = np.lexsort((np.arange(ROWS, dtype=np.int64), energies))
    return [
        np.ascontiguousarray(order[i * ROWS_PER_STRATUM : (i + 1) * ROWS_PER_STRATUM])
        for i in range(STRATA)
    ]


@dataclass(frozen=True)
class Unit:
    name: str
    expert_ordinal: int
    channel: int
    stratum: int | None
    matrix: np.ndarray


def build_units(
    triplets: list[np.ndarray], *, use_role: bool, use_strata: bool
) -> tuple[list[Unit], list[dict[str, Any]], list[dict[str, Any]]]:
    units: list[Unit] = []
    role_records: list[dict[str, Any]] = []
    gauge_records: list[dict[str, Any]] = []
    for expert_ordinal, source in enumerate(triplets):
        gauge_records.append({"expert_ordinal": expert_ordinal, **gauge_audit(source)})
        if use_role:
            channels, basis, eigenvalues = role_klt(source)
            role_records.append(
                {
                    "expert_ordinal": expert_ordinal,
                    "basis": basis.tolist(),
                    "role_energy_eigenvalues": eigenvalues.tolist(),
                    "orthogonality_max_abs_error": float(
                        np.max(np.abs(basis.T @ basis - np.eye(ROLES)))
                    ),
                }
            )
        else:
            channels = source
        for channel in range(ROLES):
            matrix = channels[channel]
            if use_strata:
                for stratum, indices in enumerate(stratum_indices(matrix)):
                    units.append(
                        Unit(
                            name=(
                                f"expert_{expert_ordinal:02d}_channel_{channel}_"
                                f"stratum_{stratum}"
                            ),
                            expert_ordinal=expert_ordinal,
                            channel=channel,
                            stratum=stratum,
                            matrix=np.ascontiguousarray(matrix[indices], dtype=np.float64),
                        )
                    )
            else:
                units.append(
                    Unit(
                        name=f"expert_{expert_ordinal:02d}_channel_{channel}",
                        expert_ordinal=expert_ordinal,
                        channel=channel,
                        stratum=None,
                        matrix=matrix,
                    )
                )
    expected = MATRICES * (STRATA if use_strata else 1)
    if len(units) != expected:
        raise AssertionError((len(units), expected))
    return units, role_records, gauge_records


def polar_rank_curve(matrix: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    n, m = matrix.shape
    if not (1 < n <= ROWS and m == COLS):
        raise AssertionError(matrix.shape)
    singular = np.sort(np.linalg.svd(matrix, compute_uv=False).astype(np.float64))
    energy = float(np.sum(np.square(matrix), dtype=np.float64))
    spectral_energy = float(np.dot(singular, singular))
    if not math.isclose(energy, spectral_energy, rel_tol=2e-11, abs_tol=2e-9):
        raise AssertionError((energy, spectral_energy))
    prefix = np.concatenate(([0.0], np.cumsum(singular, dtype=np.float64)))
    prefix2 = np.concatenate(([0.0], np.cumsum(np.square(singular), dtype=np.float64)))
    stiefel_dof = n * m - n * (n + 1) // 2
    values = n * m
    curve: list[dict[str, Any]] = []
    for rank in range(n - 1):
        unmodelled = n - rank
        sums = prefix[unmodelled:] - prefix[:-unmodelled]
        sums2 = prefix2[unmodelled:] - prefix2[:-unmodelled]
        errors = np.maximum(0.0, sums2 - np.square(sums) / unmodelled)
        start = int(np.argmin(errors))
        rank_dof = n * rank - rank * (rank - 1) // 2
        model_dof = stiefel_dof + 1 + rank_dof
        normal_dof = values - model_dof
        if normal_dof <= 0:
            break
        residual_energy = float(errors[start])
        model_energy = energy - residual_energy
        if not (residual_energy > 0.0 and model_energy > 0.0):
            continue
        curve.append(
            {
                "rank": rank,
                "window_start": start,
                "window_stop": start + unmodelled,
                "common_scale": float(sums[start] / unmodelled),
                "model_dof": model_dof,
                "normal_dof": normal_dof,
                "model_energy": model_energy,
                "normal_energy": residual_energy,
            }
        )
    if not curve:
        raise AssertionError("empty polar curve")
    return curve, singular


def waterfill(
    dimensions: np.ndarray,
    energies: np.ndarray,
    rate_bpw: float,
) -> dict[str, Any]:
    """Exact finite-component Gaussian reverse-waterfill.

    Dimensions and energies are fractions of the *original panel*, and may
    sum below one for the explicitly source-leaky envelopes.
    """
    dimensions = np.asarray(dimensions, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    if dimensions.ndim != 1 or energies.shape != dimensions.shape:
        raise ValueError("component geometry mismatch")
    if np.any(dimensions <= 0.0) or np.any(energies <= 0.0):
        raise ValueError("non-positive component")
    if not rate_bpw >= 0.0:
        raise ValueError(rate_bpw)
    logv = np.log2(energies / dimensions)
    order = np.argsort(logv)[::-1]
    sorted_logv = logv[order]
    sorted_d = dimensions[order]
    cum_d = np.cumsum(sorted_d)
    cum_dlogv = np.cumsum(sorted_d * sorted_logv)
    levels = (cum_dlogv - 2.0 * rate_bpw) / cum_d
    active_count = len(dimensions)
    for k in range(1, len(dimensions) + 1):
        level = levels[k - 1]
        active_ok = level <= sorted_logv[k - 1] + 2e-14
        inactive_ok = k == len(dimensions) or level >= sorted_logv[k] - 2e-14
        if active_ok and inactive_ok:
            active_count = k
            break
    log_level = float(levels[active_count - 1])
    level = 2.0**log_level
    active = logv > log_level
    bits_per_dimension = 0.5 * np.maximum(0.0, logv - log_level)
    rate_contributions = dimensions * bits_per_dimension
    distortion_contributions = np.where(active, dimensions * level, energies)
    distortion = float(np.sum(distortion_contributions, dtype=np.float64))
    used_rate = float(np.sum(rate_contributions, dtype=np.float64))
    if not math.isclose(used_rate, rate_bpw, rel_tol=2e-11, abs_tol=2e-11):
        raise AssertionError((used_rate, rate_bpw))
    physical_f = distortion * 2.0 ** (2.0 * rate_bpw)
    return {
        "payload_rate_bpw": rate_bpw,
        "distortion": distortion,
        "F_at_payload_rate": physical_f,
        "s_at_payload_rate_bpw": -0.5 * math.log2(physical_f),
        "water_level": level,
        "active_components": int(np.count_nonzero(active)),
        "component_count": len(dimensions),
        "dimension_fraction_sum": float(np.sum(dimensions)),
        "energy_fraction_sum": float(np.sum(energies)),
        "rate_sum_check": used_rate,
        "allocations": [
            {
                "dimension_fraction": float(dimensions[i]),
                "energy_fraction": float(energies[i]),
                "variance": float(energies[i] / dimensions[i]),
                "active": bool(active[i]),
                "bits_per_component_dimension": float(bits_per_dimension[i]),
                "rate_bpw": float(rate_contributions[i]),
                "distortion": float(distortion_contributions[i]),
            }
            for i in range(len(dimensions))
        ],
    }


def module_side_bits(*, role: bool, strata: bool, polar: bool, units: list[Unit]) -> dict[str, int]:
    row = {
        "global_header": GLOBAL_HEADER_BITS,
        "role_klt_q15_angles": EXPERTS * ROLE_KLT_BITS_PER_EXPERT if role else 0,
        "strata_uint3_labels": MATRICES * STRATA_LABEL_BITS_PER_MATRIX if strata else 0,
        "polar_rank_and_window_labels": 0,
    }
    if polar:
        row["polar_rank_and_window_labels"] = sum(
            2 * math.ceil(math.log2(unit.matrix.shape[0])) for unit in units
        )
    row["total"] = sum(row.values())
    return row


def component_score(
    names: list[str],
    dimensions_abs: list[float],
    energies_abs: list[float],
    *,
    physical_rate: float,
    side_bits: int,
    include_allocations: bool = False,
) -> dict[str, Any]:
    side_bpw = side_bits / PANEL_VALUES
    payload_rate = physical_rate - side_bpw
    if payload_rate <= 0.0:
        return {
            "valid": False,
            "physical_rate_bpw": physical_rate,
            "side_bpw": side_bpw,
            "payload_rate_bpw": payload_rate,
        }
    d = np.asarray(dimensions_abs, dtype=np.float64) / PANEL_VALUES
    e = np.asarray(energies_abs, dtype=np.float64)
    total_source_energy = float(np.sum(e))
    # In an ordinary complete decomposition energies sum to source energy.  A
    # leaky subset passes a separate explicit total_source_energy argument by
    # pre-normalising energies before this wrapper and does not use this path.
    e = e / total_source_energy
    score = waterfill(d, e, payload_rate)
    distortion = float(score["distortion"])
    physical_f = distortion * 2.0 ** (2.0 * physical_rate)
    result: dict[str, Any] = {
        "valid": True,
        "physical_rate_bpw": physical_rate,
        "side_bits": side_bits,
        "side_bpw": side_bpw,
        "payload_rate_bpw": payload_rate,
        "ideal_relative_mse": distortion,
        "gaussian_reference_mse": 2.0 ** (-2.0 * physical_rate),
        "target_mse": TARGET_F * 2.0 ** (-2.0 * physical_rate),
        "F": physical_f,
        "s_bpw": -0.5 * math.log2(physical_f),
        "passes_F_le_0p8": bool(physical_f <= TARGET_F),
        "active_components": score["active_components"],
        "component_count": score["component_count"],
        "dimension_sum": float(np.sum(d)),
        "energy_sum": float(np.sum(e)),
        "water_level": score["water_level"],
    }
    if include_allocations:
        result["allocations"] = [
            {"name": name, **allocation}
            for name, allocation in zip(names, score["allocations"], strict=True)
        ]
    return result


def plain_components(units: list[Unit]) -> tuple[list[str], list[float], list[float]]:
    return (
        [unit.name for unit in units],
        [float(unit.matrix.size) for unit in units],
        [float(np.sum(np.square(unit.matrix), dtype=np.float64)) for unit in units],
    )


def curve_local_objective(row: dict[str, Any], panel_energy: float) -> float:
    dm = float(row["model_dof"]) / PANEL_VALUES
    dn = float(row["normal_dof"]) / PANEL_VALUES
    em = float(row["model_energy"]) / panel_energy
    en = float(row["normal_energy"]) / panel_energy
    return dm * math.log2(em / dm) + dn * math.log2(en / dn)


def selected_component_arrays(
    units: list[Unit], curves: list[list[dict[str, Any]]], selection: list[int]
) -> tuple[list[str], list[float], list[float]]:
    names: list[str] = []
    dimensions: list[float] = []
    energies: list[float] = []
    for unit, curve, rank_index in zip(units, curves, selection, strict=True):
        row = curve[rank_index]
        names.extend((f"{unit.name}.manifold", f"{unit.name}.normal"))
        dimensions.extend((float(row["model_dof"]), float(row["normal_dof"])))
        energies.extend((float(row["model_energy"]), float(row["normal_energy"])))
    return names, dimensions, energies


def select_polar_ranks(
    units: list[Unit],
    curves: list[list[dict[str, Any]]],
    *,
    physical_rate: float,
    side_bits: int,
    panel_energy: float,
) -> tuple[list[int], dict[str, Any]]:
    selection = [
        min(range(len(curve)), key=lambda i: curve_local_objective(curve[i], panel_energy))
        for curve in curves
    ]
    # Coordinate descent makes no all-components-active assumption.  Each
    # update is judged by the exact reverse-waterfill at the requested rate.
    passes = 0
    total_evaluations = 0
    while passes < 6:
        changed = False
        for unit_index, curve in enumerate(curves):
            current = selection[unit_index]
            best = current
            names, dims, energies = selected_component_arrays(units, curves, selection)
            best_f = component_score(
                names,
                dims,
                energies,
                physical_rate=physical_rate,
                side_bits=side_bits,
            )["F"]
            for candidate in range(len(curve)):
                if candidate == current:
                    continue
                selection[unit_index] = candidate
                names, dims, energies = selected_component_arrays(units, curves, selection)
                candidate_f = component_score(
                    names,
                    dims,
                    energies,
                    physical_rate=physical_rate,
                    side_bits=side_bits,
                )["F"]
                total_evaluations += 1
                if candidate_f < best_f - 2e-14:
                    best_f = candidate_f
                    best = candidate
            selection[unit_index] = best
            changed |= best != current
        passes += 1
        if not changed:
            break
    names, dims, energies = selected_component_arrays(units, curves, selection)
    score = component_score(
        names,
        dims,
        energies,
        physical_rate=physical_rate,
        side_bits=side_bits,
        include_allocations=True,
    )
    score["coordinate_descent_passes"] = passes
    score["rank_candidate_evaluations"] = total_evaluations
    return selection, score


def leaky_scores(
    units: list[Unit],
    curves: list[list[dict[str, Any]]],
    selection: list[int],
    *,
    physical_rate: float,
    panel_energy: float,
) -> dict[str, Any]:
    model_d: list[float] = []
    model_e: list[float] = []
    normal_d: list[float] = []
    normal_e: list[float] = []
    for curve, chosen in zip(curves, selection, strict=True):
        row = curve[chosen]
        model_d.append(float(row["model_dof"]) / PANEL_VALUES)
        model_e.append(float(row["model_energy"]) / panel_energy)
        normal_d.append(float(row["normal_dof"]) / PANEL_VALUES)
        normal_e.append(float(row["normal_energy"]) / panel_energy)

    def score_subset(d: list[float], e: list[float]) -> dict[str, Any]:
        row = waterfill(np.asarray(d), np.asarray(e), physical_rate)
        distortion = float(row["distortion"])
        f_value = distortion * 2.0 ** (2.0 * physical_rate)
        return {
            "physical_rate_bpw": physical_rate,
            "ideal_relative_mse": distortion,
            "F": f_value,
            "s_bpw": -0.5 * math.log2(f_value),
            "passes_F_le_0p8": bool(f_value <= TARGET_F),
            "coded_dimension_fraction": float(sum(d)),
            "coded_energy_fraction": float(sum(e)),
            "active_components": row["active_components"],
        }

    free_model = score_subset(normal_d, normal_e)
    free_model.update(
        {
            "free_side": "complete source-specific polar manifold",
            "free_side_dof_fraction": float(sum(model_d)),
            "fp16_side_rate_bpw": FP16_BITS * float(sum(model_d)),
            "operationally_feasible_under_2p5_bpw": False,
        }
    )
    free_normal = score_subset(model_d, model_e)
    fp16_normal_rate = FP16_BITS * float(sum(normal_d))
    free_normal.update(
        {
            "free_side": "complete source-specific polar normal correction",
            "free_side_dof_fraction": float(sum(normal_d)),
            "fp16_side_rate_bpw": fp16_normal_rate,
            "operationally_feasible_under_2p5_bpw": bool(
                fp16_normal_rate + physical_rate <= 2.5
            ),
        }
    )
    return {
        "free_manifold_predictor_encode_normal_only": free_model,
        "free_normal_correction_encode_manifold_only": free_normal,
        "claim_boundary": (
            "These envelopes omit a source-specific component from the physical stream. "
            "They are leakage ceilings, never charged codec results."
        ),
    }


def geometry_record(
    key: str,
    units: list[Unit],
    *,
    role: bool,
    strata: bool,
    role_records: list[dict[str, Any]],
    gauge_records: list[dict[str, Any]],
    run_polar: bool,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]] | None]:
    names, dims, energies = plain_components(units)
    energy = float(sum(energies))
    if not math.isclose(sum(dims), PANEL_VALUES, rel_tol=0.0, abs_tol=0.0):
        raise AssertionError((key, sum(dims), PANEL_VALUES))
    spectra: list[dict[str, Any]] = []
    curves: list[list[dict[str, Any]]] | None = None
    if run_polar:
        curves = []
        for ordinal, unit in enumerate(units):
            curve, singular = polar_rank_curve(unit.matrix)
            curves.append(curve)
            spectra.append(
                {
                    "unit_ordinal": ordinal,
                    "name": unit.name,
                    "shape": list(unit.matrix.shape),
                    "energy": energies[ordinal],
                    "singular_values_sha256": sha256_f64(singular),
                    "singular_values": singular.tolist(),
                    "rank_curve": curve,
                }
            )
            if (ordinal + 1) % max(1, len(units) // 8) == 0:
                print(f"[{key}] polar {ordinal + 1}/{len(units)}", flush=True)
    return (
        {
            "key": key,
            "modules": {"role_gauge": role, "strata": strata, "polar": run_polar},
            "unit_count": len(units),
            "panel_dimension": int(sum(dims)),
            "panel_energy": energy,
            "plain_components": [
                {"name": name, "dimension": int(dim), "energy": e}
                for name, dim, e in zip(names, dims, energies, strict=True)
            ],
            "role_records": role_records,
            "gauge_records": gauge_records,
            "polar_units": spectra,
        },
        curves,
    )


def read_ledger(rate: float, side_bits: int, leaky_side_rate: float = 0.0) -> dict[str, Any]:
    total_bits = PANEL_VALUES * (rate + leaky_side_rate)
    frame_bytes = math.ceil(total_bits / 8 / EXPERTS)
    equal_share = total_bits / 8 / EXPERTS
    cold = frame_bytes + 4096
    return {
        "expert_local": True,
        "physical_rate_bpw_including_optional_leaky_side": rate + leaky_side_rate,
        "charged_explicit_side_bpw_within_rate": side_bits / PANEL_VALUES,
        "equal_expert_frame_bytes": frame_bytes,
        "cold_expert_bytes_with_4KiB_manifest": cold,
        "cold_read_amplification": cold / equal_share,
        "below_2x": cold / equal_share < 2.0,
        "scope": (
            "Compressed-object read only. Dense polar/role reconstruction must be fused "
            "with GEMM to avoid materialized-weight HBM traffic."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-geometries",
        nargs="*",
        default=[],
        choices=("raw", "role", "strata", "role_strata"),
        help="development-only; final artifact must not skip a geometry",
    )
    args = parser.parse_args()
    started = time.time()
    root = args.root.resolve()
    lock_path, lock, receipts, matrices = load_panel(root)
    triplets = expert_triplets(matrices)
    source_energy = float(
        sum(np.sum(np.square(matrix), dtype=np.float64) for matrix in matrices)
    )
    print(f"source energy {source_energy:.17g}", flush=True)

    geometry_specs = (
        ("raw", False, False),
        ("role", True, False),
        ("strata", False, True),
        ("role_strata", True, True),
    )
    geometry_reports: dict[str, dict[str, Any]] = {}
    geometry_units: dict[str, list[Unit]] = {}
    geometry_curves: dict[str, list[list[dict[str, Any]]]] = {}
    for key, use_role, use_strata in geometry_specs:
        if key in args.skip_geometries:
            continue
        units, role_records, gauge_records = build_units(
            triplets, use_role=use_role, use_strata=use_strata
        )
        report, curves = geometry_record(
            key,
            units,
            role=use_role,
            strata=use_strata,
            role_records=role_records,
            gauge_records=gauge_records,
            run_polar=True,
        )
        if not math.isclose(report["panel_energy"], source_energy, rel_tol=3e-13, abs_tol=2e-9):
            raise AssertionError((key, report["panel_energy"], source_energy))
        geometry_reports[key] = report
        geometry_units[key] = units
        assert curves is not None
        geometry_curves[key] = curves

    variants = {
        "baseline": ("raw", False, False, False),
        "role_gauge": ("role", True, False, False),
        "strata": ("strata", False, True, False),
        "role_gauge+strata": ("role_strata", True, True, False),
        "polar": ("raw", False, False, True),
        "role_gauge+polar": ("role", True, False, True),
        "strata+polar": ("strata", False, True, True),
        "role_gauge+strata+polar": ("role_strata", True, True, True),
    }
    variant_reports: dict[str, Any] = {}
    for variant, (geometry, use_role, use_strata, use_polar) in variants.items():
        if geometry not in geometry_reports:
            continue
        units = geometry_units[geometry]
        side = module_side_bits(
            role=use_role, strata=use_strata, polar=use_polar, units=units
        )
        names, dims, energies = plain_components(units)
        rates: dict[str, Any] = {}
        for rate in RATES:
            if use_polar:
                selection, score = select_polar_ranks(
                    units,
                    geometry_curves[geometry],
                    physical_rate=rate,
                    side_bits=side["total"],
                    panel_energy=source_energy,
                )
                score["selections"] = [
                    {
                        "unit_ordinal": i,
                        "rank_curve_index": chosen,
                        **geometry_curves[geometry][i][chosen],
                    }
                    for i, chosen in enumerate(selection)
                ]
                score["source_leaky_envelopes"] = leaky_scores(
                    units,
                    geometry_curves[geometry],
                    selection,
                    physical_rate=rate,
                    panel_energy=source_energy,
                )
            else:
                score = component_score(
                    names,
                    dims,
                    energies,
                    physical_rate=rate,
                    side_bits=side["total"],
                    include_allocations=True,
                )
            score["read_ledger"] = read_ledger(rate, side["total"])
            rates[f"{rate:.2f}"] = score
        variant_reports[variant] = {
            "modules": {
                "role_gauge": use_role,
                "strata": use_strata,
                "polar": use_polar,
            },
            "geometry": geometry,
            "explicit_side_bits": side,
            "rates": rates,
        }
        best = min(rates.values(), key=lambda row: row["F"])
        print(
            f"[{variant}] best F={best['F']:.9f} s={best['s_bpw']:.9f} "
            f"pass={best['passes_F_le_0p8']}",
            flush=True,
        )

    charged_passes: list[dict[str, Any]] = []
    for variant, row in variant_reports.items():
        module_count = sum(bool(x) for x in row["modules"].values())
        for rate_text, score in row["rates"].items():
            if score["passes_F_le_0p8"]:
                charged_passes.append(
                    {
                        "variant": variant,
                        "rate": float(rate_text),
                        "F": score["F"],
                        "s_bpw": score["s_bpw"],
                        "module_count": module_count,
                        "side_bpw": score["side_bpw"],
                    }
                )
    charged_passes.sort(key=lambda x: (x["module_count"], x["F"], x["side_bpw"]))
    leaky_passes: list[dict[str, Any]] = []
    for variant, row in variant_reports.items():
        if not row["modules"]["polar"]:
            continue
        module_count = sum(bool(x) for x in row["modules"].values())
        for rate_text, score in row["rates"].items():
            for envelope, item in score["source_leaky_envelopes"].items():
                if not isinstance(item, dict) or not item.get("passes_F_le_0p8", False):
                    continue
                leaky_passes.append(
                    {
                        "variant": variant,
                        "envelope": envelope,
                        "rate": float(rate_text),
                        "F": item["F"],
                        "module_count": module_count,
                        "free_side_dof_fraction": item["free_side_dof_fraction"],
                        "fp16_side_rate_bpw": item["fp16_side_rate_bpw"],
                    }
                )
    leaky_passes.sort(
        key=lambda x: (x["module_count"], x["fp16_side_rate_bpw"], x["F"])
    )

    script_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema": "qwen_nested_composite_superoracle_v1",
        "scope": {
            "checkpoint": lock["checkpoint"],
            "matrix_count": MATRICES,
            "expert_count": EXPERTS,
            "canonical_shape": [ROWS, COLS],
            "panel_values": PANEL_VALUES,
            "source_energy": source_energy,
            "rates_bpw": list(RATES),
            "target_F": TARGET_F,
            "target_s_bpw": TARGET_S,
            "cpu_only": True,
            "cupy_imported": False,
            "scale_field_module": "omitted: owned by concurrent spectral_scale_field experiment",
        },
        "method": {
            "nested_order": [
                "source-metric orthogonal expert-local role innovation KLT",
                "disjoint equipopulous STRATA row partition",
                "exact polar H=cI+A_k model/normal split inside each current unit",
                "one Gaussian reverse-waterfill over measured energies and exact dimensions",
            ],
            "non_additivity_rule": (
                "No prior F or s is imported or summed. Every variant is rebuilt from the "
                "authenticated source and scored by one joint waterfill."
            ),
            "polar_optimism": [
                "continuous Stiefel/manifold coordinates",
                "free exact charts and infinite precision",
                "source-specific adaptive ranks and windows",
                "ideal asymptotic Gaussian rate-distortion coding",
            ],
            "gauge_rule": (
                "The exact Up/Down equal-norm gauge is audited but contributes no separate "
                "gain: its inverse source metric cancels the coordinate scaling exactly."
            ),
        },
        "decision": {
            "charged_target_reached": bool(charged_passes),
            "minimal_charged_pass": charged_passes[0] if charged_passes else None,
            "all_charged_passes": charged_passes,
            "source_leaky_target_reached": bool(leaky_passes),
            "minimal_source_leaky_pass": leaky_passes[0] if leaky_passes else None,
            "all_source_leaky_passes": leaky_passes,
            "gpu_followup_warranted": bool(charged_passes),
            "rule": (
                "Only a charged F<=0.8 result with <2x expert-local read is an operational "
                "architecture survivor. A free-component envelope is diagnostic only."
            ),
        },
        "audit": {
            "source_lock_path": str(lock_path),
            "source_lock_file_sha256": sha256_file(lock_path),
            "source_lock_internal_sha256": lock.get("lock_sha256"),
            "all_source_hashes_matched": all(
                row["declared_sha256"] == row["observed_sha256"] for row in receipts
            ),
            "source_receipts": receipts,
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "elapsed_seconds": time.time() - started,
        },
        "geometries": geometry_reports,
        "variants": variant_reports,
        "claim_boundary": (
            "This is a source-locked ideal-RD oracle, not emitted weights and not achieved "
            "finite-code MSE. Charged manifold dimensions are honest, but charts, curvature, "
            "finite precision, and quantizer losses are favourable omissions. Source-leaky "
            "envelopes omit an entire source-specific component and cannot support a codec claim."
        ),
    }
    write_sealed_json(args.output, report)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
