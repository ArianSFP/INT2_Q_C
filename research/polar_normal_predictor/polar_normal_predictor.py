#!/usr/bin/env python3
"""Held-out CPU oracle for predicting the Qwen polar normal field.

The nested composite oracle found one deliberately source-leaky pass: reveal
the exact polar normal correction and encode only the manifold.  This script
asks whether that omitted correction can be generated or represented from
shared/decoder-visible structure within a strict private field budget.

All predictors are orthogonal projections.  Their residual energies are
combined with the exact per-matrix polar manifolds and passed to one Gaussian
reverse-waterfiller.  Predictor coefficients are first granted exact for a
free-predictor gate, then charged at one bit (an optimistic lower ledger) and
FP16 (a finite storage ledger).  No CUDA/CuPy import is made.
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
from typing import Any

import numpy as np


ROWS = 768
COLS = 2048
ROLES = 3
EXPERTS = 6
MATRICES = 18
VALUES = ROWS * COLS
PANEL_VALUES = MATRICES * VALUES
FULL_MODEL_VALUES = 48 * 128 * 3 * VALUES
RATE = 2.5
TARGET_F = 0.8
STRICT_FIELD_BPW = 0.11786907284019277  # prerequested conservative gate
EXACT_COMPONENT_ALLOWANCE_BPW = 0.12024848608281823
FP16_BITS = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def write_sealed(path: Path, report: dict[str, Any]) -> None:
    clean = dict(report)
    clean.pop("result_lock_sha256", None)
    clean["result_lock_sha256"] = hashlib.sha256(canonical(clean)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    if words.size != math.prod(shape):
        raise AssertionError((path, words.size, shape))
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not np.all(np.isfinite(values)):
        raise AssertionError(path)
    return values.reshape(shape)


def load_sources(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    lock_path = root / "blind_protocol_v2/unblinded/source_hashes.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    matrices: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    for ordinal, row in enumerate(lock["matrices"]):
        if int(row["matrix_ordinal"]) != ordinal:
            raise AssertionError("source order")
        path = lock_path.parent / row["output_relpath"]
        observed = sha256_file(path)
        if observed != row["source_bf16_sha256"]:
            raise AssertionError((path, "hash"))
        matrix = read_bf16(path, tuple(int(x) for x in row["shape"]))
        if row["role"] == "down":
            matrix = matrix.T
        matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        matrices.append(matrix)
        receipts.append(
            {
                "matrix_ordinal": ordinal,
                "layer": int(row["layer"]),
                "expert": int(row["expert"]),
                "role": str(row["role"]),
                "relative_path": str(path.relative_to(lock_path.parent)).replace("\\", "/"),
                "declared_sha256": str(row["source_bf16_sha256"]),
                "observed_sha256": observed,
            }
        )
    if len(matrices) != MATRICES:
        raise AssertionError(len(matrices))
    return lock_path, lock, receipts, matrices


@dataclass
class NormalRecord:
    ordinal: int
    expert_ordinal: int
    role_ordinal: int
    layer: int
    expert: int
    role: str
    source_energy: float
    model_energy: float
    normal_energy: float
    model_dof: int
    normal_dof: int
    rank: int
    window_start: int
    window_stop: int
    common_scale: float
    normal: np.ndarray
    error: np.ndarray


def best_window(singular: np.ndarray, rank: int) -> tuple[int, int, float, float]:
    width = len(singular) - rank
    prefix = np.concatenate(([0.0], np.cumsum(singular, dtype=np.float64)))
    prefix2 = np.concatenate(([0.0], np.cumsum(np.square(singular), dtype=np.float64)))
    sums = prefix[width:] - prefix[:-width]
    sums2 = prefix2[width:] - prefix2[:-width]
    residuals = np.maximum(0.0, sums2 - np.square(sums) / width)
    start = int(np.argmin(residuals))
    return start, start + width, float(sums[start] / width), float(residuals[start])


def normal_record(
    matrix: np.ndarray,
    metadata: dict[str, Any],
    rank: int,
) -> NormalRecord:
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    # NumPy returns descending order; all composite windows use ascending order.
    u = np.ascontiguousarray(u[:, ::-1])
    singular = np.ascontiguousarray(singular[::-1], dtype=np.float64)
    vt = np.ascontiguousarray(vt[::-1])
    start, stop, common, residual = best_window(singular, rank)
    delta = np.zeros_like(singular)
    delta[start:stop] = singular[start:stop] - common
    normal = (u * delta[None, :]) @ u.T
    error = (u * delta[None, :]) @ vt
    normal_energy = float(np.sum(np.square(normal), dtype=np.float64))
    error_energy = float(np.sum(np.square(error), dtype=np.float64))
    if not math.isclose(normal_energy, residual, rel_tol=3e-10, abs_tol=3e-9):
        raise AssertionError((normal_energy, residual))
    if not math.isclose(error_energy, residual, rel_tol=3e-10, abs_tol=3e-9):
        raise AssertionError((error_energy, residual))
    n = matrix.shape[0]
    stiefel = n * matrix.shape[1] - n * (n + 1) // 2
    model_dof = stiefel + 1 + n * rank - rank * (rank - 1) // 2
    normal_dof = matrix.size - model_dof
    source_energy = float(np.sum(np.square(matrix), dtype=np.float64))
    return NormalRecord(
        ordinal=int(metadata["matrix_ordinal"]),
        expert_ordinal=int(metadata["matrix_ordinal"]) // ROLES,
        role_ordinal=int(metadata["matrix_ordinal"]) % ROLES,
        layer=int(metadata["layer"]),
        expert=int(metadata["expert"]),
        role=str(metadata["role"]),
        source_energy=source_energy,
        model_energy=source_energy - residual,
        normal_energy=residual,
        model_dof=model_dof,
        normal_dof=normal_dof,
        rank=rank,
        window_start=start,
        window_stop=stop,
        common_scale=common,
        normal=normal,
        error=error,
    )


def load_base_selections(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    raw = path.read_bytes()
    report = json.loads(raw)
    claimed = report.get("result_lock_sha256")
    clean = dict(report)
    clean.pop("result_lock_sha256", None)
    if hashlib.sha256(canonical(clean)).hexdigest() != claimed:
        raise AssertionError("base composite result seal")
    variant = report["variants"]["polar"]
    score = variant["rates"]["2.50"]
    selections = score["selections"]
    if len(selections) != MATRICES:
        raise AssertionError(len(selections))
    return report, selections, int(variant["explicit_side_bits"]["total"])


def gaussian_controls(
    matrices: list[np.ndarray],
    metadata: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    seed: int,
) -> list[NormalRecord]:
    rng = np.random.default_rng(seed)
    controls: list[NormalRecord] = []
    for ordinal, (source, meta, selection) in enumerate(
        zip(matrices, metadata, selections, strict=True)
    ):
        source_mean = float(np.mean(source, dtype=np.float64))
        centered_energy = float(np.sum(np.square(source - source_mean), dtype=np.float64))
        gaussian = rng.standard_normal(source.shape, dtype=np.float64)
        gaussian -= float(np.mean(gaussian, dtype=np.float64))
        gaussian *= math.sqrt(centered_energy / float(np.sum(np.square(gaussian))))
        gaussian += source_mean
        record = normal_record(gaussian, meta, int(selection["rank"]))
        if not math.isclose(record.source_energy, float(np.sum(np.square(source))), rel_tol=3e-13, abs_tol=3e-9):
            raise AssertionError((ordinal, "control moment"))
        controls.append(record)
    return controls


def dct_basis(n: int) -> np.ndarray:
    positions = np.arange(n, dtype=np.float64)[:, None] + 0.5
    frequencies = np.arange(n, dtype=np.float64)[None, :]
    basis = math.sqrt(2.0 / n) * np.cos(math.pi * positions * frequencies / n)
    basis[:, 0] = 1.0 / math.sqrt(n)
    if np.max(np.abs(basis.T @ basis - np.eye(n))) > 2e-12:
        raise AssertionError("DCT orthogonality")
    return basis


def waterfill(d: np.ndarray, e: np.ndarray, rate: float) -> dict[str, Any]:
    d = np.asarray(d, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    keep = (d > 0.0) & (e > 0.0)
    d = d[keep]
    e = e[keep]
    if rate <= 0.0 or not len(d):
        return {"valid": False}
    logv = np.log2(e / d)
    order = np.argsort(logv)[::-1]
    lv = logv[order]
    ds = d[order]
    cd = np.cumsum(ds)
    cdlv = np.cumsum(ds * lv)
    levels = (cdlv - 2.0 * rate) / cd
    active_count = len(d)
    for k in range(1, len(d) + 1):
        level = levels[k - 1]
        if level <= lv[k - 1] + 2e-14 and (k == len(d) or level >= lv[k] - 2e-14):
            active_count = k
            break
    log_level = float(levels[active_count - 1])
    level = 2.0**log_level
    active = logv > log_level
    used = float(np.sum(d * 0.5 * np.maximum(0.0, logv - log_level)))
    if not math.isclose(used, rate, rel_tol=3e-10, abs_tol=3e-10):
        raise AssertionError((used, rate))
    distortion = float(np.sum(np.where(active, d * level, e), dtype=np.float64))
    return {
        "valid": True,
        "payload_rate_bpw": rate,
        "distortion": distortion,
        "active_components": int(np.count_nonzero(active)),
        "dimension_sum": float(np.sum(d)),
        "energy_sum": float(np.sum(e)),
    }


def score_candidate(
    records: list[NormalRecord],
    residual_energies: list[float],
    removed_dofs: list[int],
    *,
    side_bpw: float,
) -> dict[str, Any]:
    total_energy = float(sum(row.source_energy for row in records))
    dimensions: list[float] = []
    energies: list[float] = []
    for row, residual, removed in zip(records, residual_energies, removed_dofs, strict=True):
        dimensions.append(row.model_dof / PANEL_VALUES)
        energies.append(row.model_energy / total_energy)
        if residual > 0.0:
            dimensions.append(max(1, row.normal_dof - int(removed)) / PANEL_VALUES)
            energies.append(float(residual) / total_energy)
    payload = RATE - side_bpw
    score = waterfill(np.asarray(dimensions), np.asarray(energies), payload)
    if not score["valid"]:
        return {"valid": False, "side_bpw": side_bpw, "payload_rate_bpw": payload}
    f_value = float(score["distortion"]) * 2.0 ** (2.0 * RATE)
    return {
        **score,
        "side_bpw": side_bpw,
        "physical_rate_bpw": RATE,
        "F": f_value,
        "s_bpw": -0.5 * math.log2(f_value),
        "passes_F_le_0p8": bool(f_value <= TARGET_F),
        "ideal_relative_mse": score["distortion"],
    }


@dataclass
class Candidate:
    name: str
    family: str
    parameter: Any
    residuals: list[float]
    coefficients: list[int]
    shared_parameters: int = 0
    shared_basis_bytes_cold: int = 0
    notes: str = ""


def projection_span(target: np.ndarray, templates: list[np.ndarray]) -> tuple[float, int]:
    if not templates:
        return float(np.sum(np.square(target))), 0
    gram = np.empty((len(templates), len(templates)), dtype=np.float64)
    rhs = np.empty(len(templates), dtype=np.float64)
    for i, left in enumerate(templates):
        rhs[i] = float(np.sum(left * target, dtype=np.float64))
        for j in range(i + 1):
            value = float(np.sum(left * templates[j], dtype=np.float64))
            gram[i, j] = gram[j, i] = value
    coefficient = np.linalg.pinv(gram, rcond=1e-12) @ rhs
    captured = float(rhs @ coefficient)
    energy = float(np.sum(np.square(target), dtype=np.float64))
    residual = max(0.0, energy - captured)
    return residual, len(templates)


def band_projection(normal: np.ndarray, basis: np.ndarray, bandwidth: int) -> tuple[float, int]:
    coordinates = basis.T @ normal @ basis
    return band_projection_coordinates(coordinates, bandwidth)


def band_projection_coordinates(coordinates: np.ndarray, bandwidth: int) -> tuple[float, int]:
    n = len(coordinates)
    row, col = np.indices((n, n))
    mask = np.abs(row - col) <= bandwidth
    captured = float(np.sum(np.square(coordinates[mask]), dtype=np.float64))
    energy = float(np.sum(np.square(coordinates), dtype=np.float64))
    coefficients = n * (bandwidth + 1) - bandwidth * (bandwidth + 1) // 2
    return max(0.0, energy - captured), coefficients


def lowfreq_projection(normal: np.ndarray, basis: np.ndarray, cutoff: int) -> tuple[float, int]:
    coordinates = basis.T @ normal @ basis
    return lowfreq_projection_coordinates(coordinates, cutoff)


def lowfreq_projection_coordinates(coordinates: np.ndarray, cutoff: int) -> tuple[float, int]:
    row, col = np.indices(coordinates.shape)
    mask = (row + col < cutoff)
    captured = float(np.sum(np.square(coordinates[mask]), dtype=np.float64))
    energy = float(np.sum(np.square(coordinates), dtype=np.float64))
    # Unique symmetric coefficients in the procedural triangular mask.
    unique = (row <= col) & mask
    return max(0.0, energy - captured), int(np.count_nonzero(unique))


def radial_template(training: list[np.ndarray]) -> np.ndarray:
    n = training[0].shape[0]
    values = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    for matrix in training:
        for offset in range(n):
            diagonal = np.diagonal(matrix, offset=offset)
            multiplier = 1.0 if offset == 0 else 2.0
            values[offset] += multiplier * float(np.sum(diagonal, dtype=np.float64))
            counts[offset] += multiplier * len(diagonal)
    values /= counts
    index = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return values[index]


def build_shared_candidates(records: list[NormalRecord]) -> list[Candidate]:
    identity = np.eye(ROWS, dtype=np.float64)
    dct = dct_basis(ROWS)
    folds = [[i for i in range(MATRICES) if i // ROLES != target // ROLES] for target in range(MATRICES)]
    role_training = [
        [i for i in folds[target] if i % ROLES == target % ROLES]
        for target in range(MATRICES)
    ]
    global_energy_bases: list[np.ndarray] = []
    role_energy_bases: list[np.ndarray] = []
    role_radial: list[np.ndarray] = []
    global_basis_by_expert: dict[int, np.ndarray] = {}
    for target in range(MATRICES):
        expert = target // ROLES
        if expert not in global_basis_by_expert:
            global_covariance = sum(
                records[i].normal @ records[i].normal for i in folds[target]
            )
            _, global_basis = np.linalg.eigh(global_covariance)
            global_basis_by_expert[expert] = global_basis[:, ::-1]
        role_covariance = sum(
            records[i].normal @ records[i].normal for i in role_training[target]
        )
        _, role_basis = np.linalg.eigh(role_covariance)
        global_energy_bases.append(global_basis_by_expert[expert])
        role_energy_bases.append(role_basis[:, ::-1])
        role_radial.append(radial_template([records[i].normal for i in role_training[target]]))

    candidates: list[Candidate] = []

    def add_bands(label: str, family: str, bases: list[np.ndarray] | np.ndarray, shared: int, cold: int) -> None:
        coordinates = [
            (bases[i] if isinstance(bases, list) else bases).T
            @ record.normal
            @ (bases[i] if isinstance(bases, list) else bases)
            for i, record in enumerate(records)
        ]
        for bandwidth in (0, 1, 2, 4, 8, 12, 16, 24, 32, 64, 128):
            residuals: list[float] = []
            coefficients: list[int] = []
            for coordinate in coordinates:
                residual, count = band_projection_coordinates(coordinate, bandwidth)
                residuals.append(residual)
                coefficients.append(count)
            candidates.append(
                Candidate(
                    name=f"{label}_band_{bandwidth}",
                    family=family,
                    parameter={"bandwidth": bandwidth},
                    residuals=residuals,
                    coefficients=coefficients,
                    shared_parameters=shared,
                    shared_basis_bytes_cold=cold,
                    notes="exact source-specific band coefficients; coefficient quantisation granted lossless",
                )
            )

    add_bands("identity", "analytic_identity_band", identity, 0, 0)
    add_bands("dct", "analytic_dct_band", dct, 0, 0)
    dense_basis_parameters_role = ROLES * ROWS * ROWS
    dense_basis_parameters_global = ROWS * ROWS
    add_bands(
        "role_shared_energy_basis",
        "heldout_role_shared_eigenbasis_band",
        role_energy_bases,
        dense_basis_parameters_role,
        dense_basis_parameters_role * 2,
    )
    add_bands(
        "global_shared_energy_basis",
        "heldout_global_shared_eigenbasis_band",
        global_energy_bases,
        dense_basis_parameters_global,
        dense_basis_parameters_global * 2,
    )

    dct_coordinates = [dct.T @ record.normal @ dct for record in records]
    for cutoff in (8, 16, 32, 64, 96, 128, 192, 256):
        residuals = []
        coefficients = []
        for coordinate in dct_coordinates:
            residual, count = lowfreq_projection_coordinates(coordinate, cutoff)
            residuals.append(residual)
            coefficients.append(count)
        candidates.append(
            Candidate(
                name=f"dct_lowfreq_triangle_{cutoff}",
                family="analytic_2d_scale_field",
                parameter={"cutoff": cutoff},
                residuals=residuals,
                coefficients=coefficients,
                notes="procedural 2-D DCT support; exact FP16-ledger coefficients granted lossless",
            )
        )

    for label, training_lists in (("role", role_training), ("global", folds)):
        residuals = []
        coefficients = []
        for target, training in enumerate(training_lists):
            residual, count = projection_span(
                records[target].normal, [records[i].normal for i in training]
            )
            residuals.append(residual)
            coefficients.append(count)
        template_count = (ROLES * 5 if label == "role" else 15) * ROWS * ROWS
        candidates.append(
            Candidate(
                name=f"heldout_{label}_normal_span",
                family=f"heldout_{label}_template_span",
                parameter={"templates_per_target": len(training_lists[0])},
                residuals=residuals,
                coefficients=coefficients,
                shared_parameters=template_count,
                shared_basis_bytes_cold=template_count * 2,
                notes="target coefficients fitted exactly; templates exclude the complete target expert",
            )
        )

    radial_residuals = []
    for target, template in enumerate(role_radial):
        residual, _ = projection_span(records[target].normal, [template])
        radial_residuals.append(residual)
    candidates.append(
        Candidate(
            name="heldout_role_radial_implicit",
            family="tiny_coordinate_conditioned_implicit",
            parameter={"distance_parameters_per_role": ROWS},
            residuals=radial_residuals,
            coefficients=[1] * MATRICES,
            shared_parameters=ROLES * ROWS,
            shared_basis_bytes_cold=ROLES * ROWS * 2,
            notes="N_ij=f_role(|i-j|), trained without the target expert; one exact target scale",
        )
    )
    return candidates


def load_router_bases(
    router_dir: Path, records: list[NormalRecord]
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    pca: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        if record.layer not in cache:
            path = router_dir / f"model.layers.{record.layer}.mlp.gate.weight.block0.bf16.bin"
            router = np.ascontiguousarray(read_bf16(path, (128, COLS)), dtype=np.float64)
            _, _, vt = np.linalg.svd(router, full_matrices=False)
            cache[record.layer] = (router, vt.T)
            receipts.append(
                {
                    "layer": record.layer,
                    "relative_path": path.name,
                    "sha256": sha256_file(path),
                    "nbytes": path.stat().st_size,
                }
            )
        router, basis = cache[record.layer]
        row = router[record.expert]
        norm = float(np.linalg.norm(row))
        if not norm > 0.0:
            raise AssertionError("zero router row")
        pca.append(basis)
        target_rows.append((row / norm)[:, None])
    return pca, target_rows, receipts


def build_router_candidates(
    records: list[NormalRecord], router_bases: list[np.ndarray], target_rows: list[np.ndarray]
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for rank in (1, 2, 4, 8, 16, 32, 64, 128):
        residuals = []
        coefficients = []
        for record, basis in zip(records, router_bases, strict=True):
            projection = record.error @ basis[:, :rank]
            captured = float(np.sum(np.square(projection), dtype=np.float64))
            residuals.append(max(0.0, record.normal_energy - captured))
            coefficients.append(ROWS * rank)
        candidates.append(
            Candidate(
                name=f"router_pca_right_rank_{rank}",
                family="decoder_visible_router_right_basis",
                parameter={"rank": rank},
                residuals=residuals,
                coefficients=coefficients,
                notes="same-layer router PCA is decoder-visible; exact 768xk coefficients granted lossless",
            )
        )
    residuals = []
    for record, basis in zip(records, target_rows, strict=True):
        projection = record.error @ basis
        captured = float(np.sum(np.square(projection), dtype=np.float64))
        residuals.append(max(0.0, record.normal_energy - captured))
    candidates.append(
        Candidate(
            name="routed_expert_row_right_rank_1",
            family="decoder_visible_routed_row",
            parameter={"rank": 1},
            residuals=residuals,
            coefficients=[ROWS] * MATRICES,
            notes="the exact routed expert row is decoder-visible before expert fetch",
        )
    )
    return candidates


def summarize_candidate(
    candidate: Candidate,
    records: list[NormalRecord],
    base_side_bpw: float,
) -> dict[str, Any]:
    normal_energy = float(sum(row.normal_energy for row in records))
    residual_energy = float(sum(candidate.residuals))
    captured = max(0.0, normal_energy - residual_energy)
    coefficients = int(sum(candidate.coefficients))
    field_bpw_fp16 = coefficients * FP16_BITS / PANEL_VALUES
    field_bpw_onebit = coefficients / PANEL_VALUES
    shared_full_model_bpw = candidate.shared_parameters * FP16_BITS / FULL_MODEL_VALUES
    removed = [min(count, row.normal_dof - 1) for count, row in zip(candidate.coefficients, records, strict=True)]
    free_score = score_candidate(records, candidate.residuals, removed, side_bpw=base_side_bpw)
    onebit_score = score_candidate(
        records,
        candidate.residuals,
        removed,
        side_bpw=base_side_bpw + field_bpw_onebit + shared_full_model_bpw,
    )
    fp16_score = score_candidate(
        records,
        candidate.residuals,
        removed,
        side_bpw=base_side_bpw + field_bpw_fp16 + shared_full_model_bpw,
    )
    return {
        "name": candidate.name,
        "family": candidate.family,
        "parameter": candidate.parameter,
        "notes": candidate.notes,
        "normal_energy": normal_energy,
        "residual_normal_energy": residual_energy,
        "captured_normal_energy_fraction": captured / normal_energy,
        "private_coefficient_count": coefficients,
        "private_coefficient_counts_per_matrix": candidate.coefficients,
        "private_field_bpw_one_bit_lower_bound": field_bpw_onebit,
        "private_field_bpw_fp16": field_bpw_fp16,
        "within_strict_field_gate_at_fp16": field_bpw_fp16 <= STRICT_FIELD_BPW,
        "within_exact_component_allowance_at_fp16": field_bpw_fp16 <= EXACT_COMPONENT_ALLOWANCE_BPW,
        "shared_parameters_fp16": candidate.shared_parameters,
        "shared_table_full_model_amortized_bpw": shared_full_model_bpw,
        "shared_table_cold_bytes": candidate.shared_basis_bytes_cold,
        "shared_table_cold_read_below_2x_for_2p5_payload": (
            candidate.shared_basis_bytes_cold + math.ceil(PANEL_VALUES * RATE / 8 / EXPERTS)
        )
        / math.ceil(PANEL_VALUES * RATE / 8 / EXPERTS)
        < 2.0,
        "free_exact_coefficient_score": free_score,
        "one_bit_coefficient_lower_bound_score": onebit_score,
        "fp16_exact_coefficient_optimistic_score": fp16_score,
        "residual_normal_energy_per_matrix": candidate.residuals,
    }


def adaptive_union(
    candidates: list[Candidate], records: list[NormalRecord], base_side_bpw: float
) -> dict[str, Any]:
    eligible = [
        candidate
        for candidate in candidates
        if sum(candidate.coefficients) * FP16_BITS / PANEL_VALUES <= STRICT_FIELD_BPW
    ]
    if not eligible:
        raise AssertionError("no eligible predictor")
    selected: list[Candidate] = []
    residuals: list[float] = []
    coefficients: list[int] = []
    for matrix in range(MATRICES):
        best = min(eligible, key=lambda row: row.residuals[matrix])
        selected.append(best)
        residuals.append(best.residuals[matrix])
        coefficients.append(best.coefficients[matrix])
    field_bpw = sum(coefficients) * FP16_BITS / PANEL_VALUES
    removed = [min(count, row.normal_dof - 1) for count, row in zip(coefficients, records, strict=True)]
    return {
        "selection_names": [row.name for row in selected],
        "private_coefficient_count": sum(coefficients),
        "private_field_bpw_fp16": field_bpw,
        "within_strict_field_gate": field_bpw <= STRICT_FIELD_BPW,
        "captured_normal_energy_fraction": 1.0
        - sum(residuals) / sum(row.normal_energy for row in records),
        "free_score": score_candidate(records, residuals, removed, side_bpw=base_side_bpw),
        "fp16_score": score_candidate(
            records, residuals, removed, side_bpw=base_side_bpw + field_bpw
        ),
        "claim_boundary": (
            "Source-aware per-matrix family selection and free mode labels; this is an upper bound, "
            "not an operational format."
        ),
    }


def strip_records(records: list[NormalRecord]) -> list[dict[str, Any]]:
    return [
        {
            "matrix_ordinal": row.ordinal,
            "expert_ordinal": row.expert_ordinal,
            "role_ordinal": row.role_ordinal,
            "layer": row.layer,
            "expert": row.expert,
            "role": row.role,
            "source_energy": row.source_energy,
            "model_energy": row.model_energy,
            "normal_energy": row.normal_energy,
            "model_dof": row.model_dof,
            "normal_dof": row.normal_dof,
            "rank": row.rank,
            "window_start": row.window_start,
            "window_stop": row.window_stop,
            "common_scale": row.common_scale,
            "normal_sha256_f64": hashlib.sha256(
                np.ascontiguousarray(row.normal, dtype="<f8").tobytes()
            ).hexdigest(),
        }
        for row in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaussian-seed", type=int, default=0x504F4C4152)
    args = parser.parse_args()
    started = time.time()
    root = args.root.resolve()
    lock_path, lock, receipts, matrices = load_sources(root)
    base, selections, base_side_bits = load_base_selections(args.base_result)
    metadata = lock["matrices"]
    source_records: list[NormalRecord] = []
    for ordinal, (matrix, meta, selection) in enumerate(
        zip(matrices, metadata, selections, strict=True)
    ):
        record = normal_record(matrix, meta, int(selection["rank"]))
        if record.window_start != int(selection["window_start"]):
            raise AssertionError((ordinal, "base window"))
        source_records.append(record)
        print(
            f"[source {ordinal + 1:02d}/{MATRICES}] rank={record.rank} "
            f"normal={record.normal_energy / record.source_energy:.6f}",
            flush=True,
        )
    control_records = gaussian_controls(
        matrices, metadata, selections, int(args.gaussian_seed)
    )
    print("matched Gaussian polar normals complete", flush=True)
    source_router, source_target_rows, router_receipts = load_router_bases(
        args.router_dir, source_records
    )
    control_router = source_router
    control_target_rows = source_target_rows

    source_candidates = build_shared_candidates(source_records) + build_router_candidates(
        source_records, source_router, source_target_rows
    )
    control_candidates = build_shared_candidates(control_records) + build_router_candidates(
        control_records, control_router, control_target_rows
    )
    if [x.name for x in source_candidates] != [x.name for x in control_candidates]:
        raise AssertionError("control candidate order")
    base_side_bpw = base_side_bits / PANEL_VALUES
    source_summaries = [
        summarize_candidate(candidate, source_records, base_side_bpw)
        for candidate in source_candidates
    ]
    control_summaries = [
        summarize_candidate(candidate, control_records, base_side_bpw)
        for candidate in control_candidates
    ]
    controls_by_name = {row["name"]: row for row in control_summaries}
    for row in source_summaries:
        control = controls_by_name[row["name"]]
        row["matched_gaussian"] = {
            "captured_normal_energy_fraction": control[
                "captured_normal_energy_fraction"
            ],
            "free_F": control["free_exact_coefficient_score"]["F"],
            "fp16_F": control["fp16_exact_coefficient_optimistic_score"].get("F"),
            "qwen_minus_gaussian_captured_fraction": row[
                "captured_normal_energy_fraction"
            ]
            - control["captured_normal_energy_fraction"],
        }
        print(
            f"[{row['name']}] cap={row['captured_normal_energy_fraction']:.5f} "
            f"freeF={row['free_exact_coefficient_score']['F']:.6f} "
            f"fp16bpw={row['private_field_bpw_fp16']:.5f}",
            flush=True,
        )

    best_free = min(source_summaries, key=lambda row: row["free_exact_coefficient_score"]["F"])
    eligible = [
        row
        for row in source_summaries
        if row["within_strict_field_gate_at_fp16"]
        and row["fp16_exact_coefficient_optimistic_score"]["valid"]
    ]
    best_fp16 = min(
        eligible,
        key=lambda row: row["fp16_exact_coefficient_optimistic_score"]["F"],
    )
    best_onebit = min(
        source_summaries,
        key=lambda row: row["one_bit_coefficient_lower_bound_score"]["F"],
    )
    best_budgeted_free = min(
        eligible,
        key=lambda row: row["free_exact_coefficient_score"]["F"],
    )
    source_union = adaptive_union(source_candidates, source_records, base_side_bpw)
    control_union = adaptive_union(control_candidates, control_records, base_side_bpw)

    report = {
        "schema": "qwen_polar_normal_predictor_oracle_v1",
        "scope": {
            "checkpoint": lock["checkpoint"],
            "physical_rate_bpw": RATE,
            "target_F": TARGET_F,
            "strict_private_field_gate_bpw": STRICT_FIELD_BPW,
            "corrected_exact_component_allowance_after_base_side_bpw": EXACT_COMPONENT_ALLOWANCE_BPW,
            "base_polar_explicit_side_bpw": base_side_bpw,
            "panel_values": PANEL_VALUES,
            "full_model_values_for_shared_table_amortization": FULL_MODEL_VALUES,
            "cpu_only": True,
            "cupy_imported": False,
        },
        "decision": {
            "status": (
                "SURVIVES_BUDGETED_FREE_ORACLE"
                if best_budgeted_free["free_exact_coefficient_score"][
                    "passes_F_le_0p8"
                ]
                else "KILL_POLAR_NORMAL_PREDICTOR_BRANCH"
            ),
            "best_free_predictor": {
                "name": best_free["name"],
                "F": best_free["free_exact_coefficient_score"]["F"],
                "field_bpw_fp16": best_free["private_field_bpw_fp16"],
                "within_strict_field_gate": best_free[
                    "within_strict_field_gate_at_fp16"
                ],
            },
            "best_free_predictor_within_strict_fp16_field_gate": {
                "name": best_budgeted_free["name"],
                "F": best_budgeted_free["free_exact_coefficient_score"]["F"],
                "field_bpw_fp16": best_budgeted_free["private_field_bpw_fp16"],
                "passes": best_budgeted_free["free_exact_coefficient_score"][
                    "passes_F_le_0p8"
                ],
            },
            "best_fp16_predictor_within_strict_field_gate": {
                "name": best_fp16["name"],
                "F": best_fp16["fp16_exact_coefficient_optimistic_score"]["F"],
                "field_bpw_fp16": best_fp16["private_field_bpw_fp16"],
                "passes": best_fp16["fp16_exact_coefficient_optimistic_score"][
                    "passes_F_le_0p8"
                ],
            },
            "best_one_bit_coefficient_lower_bound": {
                "name": best_onebit["name"],
                "F": best_onebit["one_bit_coefficient_lower_bound_score"]["F"],
                "field_bpw": best_onebit["private_field_bpw_one_bit_lower_bound"],
                "passes": best_onebit["one_bit_coefficient_lower_bound_score"][
                    "passes_F_le_0p8"
                ],
            },
            "source_adaptive_union": source_union,
            "matched_gaussian_adaptive_union": control_union,
            "gpu_followup_warranted": bool(
                best_fp16["fp16_exact_coefficient_optimistic_score"][
                    "passes_F_le_0p8"
                ]
                or source_union["fp16_score"]["passes_F_le_0p8"]
            ),
            "rule": (
                "Early-kill unless a held-out/free predictor preserves F<=0.8 and its exact "
                "private field fits the strict 0.117869-bpw gate."
            ),
        },
        "protocol": {
            "heldout_unit": "one complete expert triplet; no target Gate/Up/Down normal enters shared fits",
            "predictor_families": [
                "identity/DCT diagonal and banded normal matrices",
                "held-out global/role energy eigenbases",
                "held-out role/global normal-template spans",
                "procedural 2-D DCT low-frequency fields",
                "tiny held-out coordinate model N_ij=f_role(|i-j|)",
                "decoder-visible same-layer router PCA and routed-expert row bases",
            ],
            "optimism": [
                "source-specific coefficients are exact despite one-bit/FP16 ledgers",
                "coefficient subspace DOF is removed from the coded normal dimension",
                "ideal asymptotic Gaussian coding for remaining manifold/normal components",
                "shared tables use full-model amortized rate",
                "source-adaptive family union gets free mode labels",
            ],
            "matched_gaussian": (
                "independent deterministic controls match every target matrix's exact mean and "
                "centered energy, reuse its selected polar rank, and reselect the best window"
            ),
        },
        "source_normal_records": strip_records(source_records),
        "matched_gaussian_normal_records": strip_records(control_records),
        "candidates": source_summaries,
        "audit": {
            "source_lock_path": str(lock_path),
            "source_lock_file_sha256": sha256_file(lock_path),
            "source_lock_internal_sha256": lock.get("lock_sha256"),
            "source_receipts": receipts,
            "router_receipts": router_receipts,
            "base_composite_result_path": str(args.base_result.resolve()),
            "base_composite_result_sha256": sha256_file(args.base_result),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "gaussian_seed": int(args.gaussian_seed),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "elapsed_seconds": time.time() - started,
        },
        "claim_boundary": (
            "No emitted codec or achieved finite MSE is claimed. Exact source-specific coefficient "
            "values are granted at nominal one-bit/FP16 storage with zero quantisation loss, so a "
            "failure is an early-kill for these families; a pass would require finite integration."
        ),
    }
    write_sealed(args.output, report)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
