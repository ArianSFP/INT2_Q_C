#!/usr/bin/env python3
"""Favorable 2-D variance-field oracle for the pinned Qwen MoE panel.

This is deliberately an information/RD ceiling, not an operational weight
codec.  Each tested field is fitted to the very matrix it scores.  The first
screen gives that source-leaky field to the decoder for free and then applies
ideal scalar-Gaussian reverse waterfilling.  A second screen charges a concrete
FP16 representation of the field inside the requested physical rate.

Moment-matched iid Gaussian matrices pass through the identical fitting and
selection path.  This prevents finite-sample scale overfit from being called
Qwen structure.  All work is NumPy/CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROWS = 768
COLS = 2048
VALUES_PER_MATRIX = ROWS * COLS
MATRICES_PER_EXPERT = 3
EXPERTS = 6
TOTAL_VALUES = VALUES_PER_MATRIX * MATRICES_PER_EXPERT * EXPERTS
TILE_ROWS = 16
TILE_COLS = 32
GRID_ROWS = ROWS // TILE_ROWS
GRID_COLS = COLS // TILE_COLS
TILE_VALUES = TILE_ROWS * TILE_COLS
RATES = (2.15, 2.30, 2.50)
TARGET_F = 0.8
TARGET_S = -0.5 * math.log2(TARGET_F)
RESULT_SCHEMA = "qwen-spectral-scale-field-oracle-v1"

# Physical side ledger.  All model data are expert-private and colocated.
EXPERT_HEADER_BITS = 512
MATRIX_HEADER_BITS = 64
MEAN_BITS = 16
GLOBAL_SCALE_BITS = 16
MODEL_ID_BITS = 16
KLT_ANGLE_BITS = 16

LOW_RANKS = (1, 2, 4, 8, 16, 32, 48)
DCT_RECTS = (
    (1, 2),
    (2, 2),
    (2, 4),
    (4, 4),
    (4, 8),
    (8, 8),
    (8, 16),
    (16, 16),
    (16, 32),
    (32, 32),
    (48, 64),
)
HAAR_GRIDS = ((1, 1), (3, 4), (6, 8), (12, 16), (24, 32), (48, 64))


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_internal_lock(value: dict[str, Any], label: str) -> str:
    clean = dict(value)
    declared = clean.pop("lock_sha256", None)
    actual = hashlib.sha256(canonical_bytes(clean)).hexdigest()
    if declared != actual:
        raise ValueError(f"{label}: internal lock mismatch {declared} != {actual}")
    return actual


def seal(value: dict[str, Any]) -> dict[str, Any]:
    if "result_lock_sha256" in value:
        raise ValueError("object is already sealed")
    result = dict(value)
    result["result_lock_sha256"] = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return result


def read_bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    expected = int(np.prod(shape)) * 2
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes")
    words = np.memmap(path, dtype="<u2", mode="r", shape=shape)
    values = (np.asarray(words, dtype=np.uint32) << np.uint32(16)).view(np.float32)
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    parameters: tuple[int, ...] = ()


@dataclass
class MatrixField:
    values: np.ndarray
    counts: np.ndarray


@dataclass
class PreparedWaterfill:
    values: np.ndarray
    counts: np.ndarray
    matrix_ids: np.ndarray
    logs_descending: np.ndarray
    cumulative_weight: np.ndarray
    cumulative_weighted_log: np.ndarray
    total_weight: float
    total_energy: float


def model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec("global", "global"),
        ModelSpec("rowcol_separable", "rowcol"),
        ModelSpec("rowcol_tile_ipf_16x32", "rowcol_tile_ipf"),
        ModelSpec("tile_exact_16x32", "tile_exact"),
    ]
    specs.extend(ModelSpec(f"lowrank_log_k{k}", "lowrank_log", (k,)) for k in LOW_RANKS)
    specs.extend(
        ModelSpec(f"dct_log_{kr}x{kc}", "dct_log", (kr, kc))
        for kr, kc in DCT_RECTS
    )
    specs.extend(
        ModelSpec(f"haar_log_{gr}x{gc}", "haar_log", (gr, gc))
        for gr, gc in HAAR_GRIDS
    )
    return specs


def dct_basis(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)[:, None]
    frequencies = np.arange(size, dtype=np.float64)[None, :]
    basis = np.cos(math.pi * (positions + 0.5) * frequencies / size)
    basis[:, 0] *= math.sqrt(1.0 / size)
    if size > 1:
        basis[:, 1:] *= math.sqrt(2.0 / size)
    return basis


DCT_ROW = dct_basis(GRID_ROWS)
DCT_COL = dct_basis(GRID_COLS)


def centered(matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(matrix, dtype=np.float64))
    z = np.asarray(matrix, dtype=np.float64) - mean
    energy = float(np.sum(z * z, dtype=np.float64))
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("non-positive centered energy")
    return z, mean, energy


def normalize_field(values: np.ndarray, counts: np.ndarray, energy: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result = np.maximum(result, np.finfo(np.float64).tiny)
    modeled = float(np.sum(result * counts, dtype=np.float64))
    if modeled <= 0.0 or not math.isfinite(modeled):
        raise ValueError("invalid modeled field energy")
    return result * (energy / modeled)


def tile_energy_map(z2: np.ndarray) -> np.ndarray:
    return np.mean(
        z2.reshape(GRID_ROWS, TILE_ROWS, GRID_COLS, TILE_COLS),
        axis=(1, 3),
        dtype=np.float64,
    )


def rowcol_tile_ipf(z2: np.ndarray, iterations: int = 16) -> np.ndarray:
    """Maximum-entropy multiplicative field matching row/column/tile energies."""
    target_rows = np.sum(z2, axis=1, dtype=np.float64)
    target_cols = np.sum(z2, axis=0, dtype=np.float64)
    target_tiles = np.sum(
        z2.reshape(GRID_ROWS, TILE_ROWS, GRID_COLS, TILE_COLS),
        axis=(1, 3),
        dtype=np.float64,
    )
    field = np.full(z2.shape, float(np.mean(z2, dtype=np.float64)), dtype=np.float64)
    floor = np.finfo(np.float64).tiny
    for _ in range(iterations):
        row_sum = np.sum(field, axis=1, dtype=np.float64)
        field *= (target_rows / np.maximum(row_sum, floor))[:, None]
        col_sum = np.sum(field, axis=0, dtype=np.float64)
        field *= (target_cols / np.maximum(col_sum, floor))[None, :]
        tile_sum = np.sum(
            field.reshape(GRID_ROWS, TILE_ROWS, GRID_COLS, TILE_COLS),
            axis=(1, 3),
            dtype=np.float64,
        )
        correction = target_tiles / np.maximum(tile_sum, floor)
        field *= np.repeat(
            np.repeat(correction, TILE_ROWS, axis=0), TILE_COLS, axis=1
        )
    return field


def expand_piecewise_grid(log_grid: np.ndarray, gr: int, gc: int) -> np.ndarray:
    if GRID_ROWS % gr or GRID_COLS % gc:
        raise ValueError("Haar grid must divide the base grid")
    fr = GRID_ROWS // gr
    fc = GRID_COLS // gc
    coarse = np.mean(
        log_grid.reshape(gr, fr, gc, fc), axis=(1, 3), dtype=np.float64
    )
    return np.repeat(np.repeat(coarse, fr, axis=0), fc, axis=1)


def matrix_fields(matrix: np.ndarray, specs: list[ModelSpec]) -> tuple[dict[str, MatrixField], dict[str, float]]:
    z, mean, energy = centered(matrix)
    z2 = z * z
    tile = tile_energy_map(z2)
    log_tile = np.log(np.maximum(tile, energy / VALUES_PER_MATRIX * 1e-12))
    log_mean = float(np.mean(log_tile, dtype=np.float64))
    log_centered = log_tile - log_mean
    u, singular, vh = np.linalg.svd(log_centered, full_matrices=False)
    dct_coeff = DCT_ROW.T @ log_tile @ DCT_COL

    result: dict[str, MatrixField] = {}
    for spec in specs:
        if spec.family == "global":
            values = np.array([energy / VALUES_PER_MATRIX], dtype=np.float64)
            counts = np.array([VALUES_PER_MATRIX], dtype=np.float64)
        elif spec.family == "rowcol":
            row_mean = np.mean(z2, axis=1, dtype=np.float64)
            col_mean = np.mean(z2, axis=0, dtype=np.float64)
            global_mean = energy / VALUES_PER_MATRIX
            values = (row_mean[:, None] * col_mean[None, :] / global_mean).reshape(-1)
            counts = np.ones(values.size, dtype=np.float64)
        elif spec.family == "rowcol_tile_ipf":
            values = rowcol_tile_ipf(z2).reshape(-1)
            counts = np.ones(values.size, dtype=np.float64)
        else:
            counts = np.full(GRID_ROWS * GRID_COLS, TILE_VALUES, dtype=np.float64)
            if spec.family == "tile_exact":
                grid = tile
            elif spec.family == "lowrank_log":
                rank = spec.parameters[0]
                reconstructed = (u[:, :rank] * singular[:rank]) @ vh[:rank, :]
                grid = np.exp(np.clip(log_mean + reconstructed, -80.0, 80.0))
            elif spec.family == "dct_log":
                kr, kc = spec.parameters
                reconstructed = (
                    DCT_ROW[:, :kr]
                    @ dct_coeff[:kr, :kc]
                    @ DCT_COL[:, :kc].T
                )
                grid = np.exp(np.clip(reconstructed, -80.0, 80.0))
            elif spec.family == "haar_log":
                gr, gc = spec.parameters
                reconstructed = expand_piecewise_grid(log_tile, gr, gc)
                grid = np.exp(np.clip(reconstructed, -80.0, 80.0))
            else:
                raise AssertionError(spec.family)
            values = grid.reshape(-1)
        values = normalize_field(values, counts, energy)
        result[spec.model_id] = MatrixField(values, counts)
    return result, {"mean": mean, "centered_energy": energy}


def side_bits_per_matrix(spec: ModelSpec) -> int:
    fixed = MATRIX_HEADER_BITS + MEAN_BITS + GLOBAL_SCALE_BITS + MODEL_ID_BITS
    if spec.family == "global":
        payload = 0
    elif spec.family == "rowcol":
        payload = 16 * (ROWS + COLS)
    elif spec.family == "rowcol_tile_ipf":
        # FP16 row, column, and 48x64 multiplicative tile factors.
        payload = 16 * (ROWS + COLS + GRID_ROWS * GRID_COLS)
    elif spec.family == "tile_exact":
        payload = 16 * GRID_ROWS * GRID_COLS
    elif spec.family == "lowrank_log":
        rank = spec.parameters[0]
        # log mean plus U, singular values, and V in FP16.
        payload = 16 * (1 + rank * (GRID_ROWS + 1 + GRID_COLS))
    elif spec.family == "dct_log":
        kr, kc = spec.parameters
        payload = 16 * kr * kc
    elif spec.family == "haar_log":
        gr, gc = spec.parameters
        payload = 16 * gr * gc
    else:
        raise AssertionError(spec.family)
    return fixed + payload


def source_seed(source_sha256: str, representation: str, replicate: int) -> int:
    material = f"spectral-scale-field-v1:{source_sha256}:{representation}:{replicate}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def matched_gaussian(shape: tuple[int, int], energy: float, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    values = rng.standard_normal(shape, dtype=np.float64)
    values -= float(np.mean(values, dtype=np.float64))
    observed = float(np.sum(values * values, dtype=np.float64))
    values *= math.sqrt(energy / observed)
    return np.asarray(values, dtype=np.float32)


def load_sources(lock_path: Path, source_root: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    lock_payload = lock_path.read_bytes()
    lock = json.loads(lock_payload)
    internal = verify_internal_lock(lock, "source lock")
    matrices_meta = sorted(lock["matrices"], key=lambda row: int(row["matrix_ordinal"]))
    if len(matrices_meta) != 18:
        raise ValueError("expected exactly 18 pinned matrices")
    matrices: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    for expected_ordinal, meta in enumerate(matrices_meta):
        if int(meta["matrix_ordinal"]) != expected_ordinal:
            raise ValueError("matrix ordinals are not contiguous")
        path = source_root / meta["output_relpath"]
        actual_sha = sha256_file(path)
        if actual_sha != meta["source_bf16_sha256"]:
            raise ValueError(f"{path}: source hash mismatch")
        native_shape = tuple(int(v) for v in meta["shape"])
        matrix = read_bf16(path, native_shape)
        if meta["role"] == "down":
            matrix = matrix.T.copy()
        if matrix.shape != (ROWS, COLS):
            raise ValueError(f"{path}: canonical shape {matrix.shape}")
        matrices.append(matrix)
        receipts.append(
            {
                "matrix_ordinal": expected_ordinal,
                "layer": int(meta["layer"]),
                "expert": int(meta["expert"]),
                "role": str(meta["role"]),
                "tensor": str(meta["tensor"]),
                "bytes": int(meta["nbytes"]),
                "sha256": actual_sha,
                "canonical_shape": [ROWS, COLS],
            }
        )
    provenance = {
        "source_lock_path": str(lock_path.resolve()),
        "source_lock_file_sha256": hashlib.sha256(lock_payload).hexdigest(),
        "source_lock_internal_sha256": internal,
        "pinned_repository": lock["checkpoint"]["repo"],
        "pinned_revision": lock["checkpoint"]["revision"],
        "sources": receipts,
    }
    return matrices, provenance


def xklt_representation(raw: list[np.ndarray]) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    transformed: list[np.ndarray] = []
    receipts: list[dict[str, float]] = []
    for expert in range(EXPERTS):
        gate, up, down = raw[3 * expert : 3 * expert + 3]
        up64 = np.asarray(up, dtype=np.float64)
        down64 = np.asarray(down, dtype=np.float64)
        up64 -= float(np.mean(up64, dtype=np.float64))
        down64 -= float(np.mean(down64, dtype=np.float64))
        a = float(np.sum(up64 * up64, dtype=np.float64))
        b = float(np.sum(down64 * down64, dtype=np.float64))
        c = float(np.sum(up64 * down64, dtype=np.float64))
        theta = 0.5 * math.atan2(2.0 * c, a - b)
        co, si = math.cos(theta), math.sin(theta)
        first = np.asarray(co * up + si * down, dtype=np.float32)
        second = np.asarray(-si * up + co * down, dtype=np.float32)
        transformed.extend([gate, first, second])
        receipts.append({"expert_ordinal": expert, "theta_radians": theta, "cos": co, "sin": si})
    return transformed, receipts


def build_dataset(matrices: list[np.ndarray], specs: list[ModelSpec]) -> tuple[dict[str, list[MatrixField]], list[dict[str, float]]]:
    fields = {spec.model_id: [] for spec in specs}
    moments: list[dict[str, float]] = []
    for matrix in matrices:
        per_matrix, receipt = matrix_fields(matrix, specs)
        moments.append(receipt)
        for spec in specs:
            fields[spec.model_id].append(per_matrix[spec.model_id])
    return fields, moments


def aggregate_fields(per_matrix: list[MatrixField]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.concatenate([field.values for field in per_matrix])
    counts = np.concatenate([field.counts for field in per_matrix])
    matrix_ids = np.concatenate(
        [np.full(field.values.size, ordinal, dtype=np.int16) for ordinal, field in enumerate(per_matrix)]
    )
    return values, counts, matrix_ids


def prepare_waterfill(per_matrix: list[MatrixField]) -> PreparedWaterfill:
    values, counts, matrix_ids = aggregate_fields(per_matrix)
    log_values = np.log2(values)
    order = np.argsort(log_values)[::-1]
    logs = log_values[order]
    weights = counts[order]
    cumulative_weight = np.cumsum(weights, dtype=np.float64)
    cumulative_log = np.cumsum(weights * logs, dtype=np.float64)
    return PreparedWaterfill(
        values=values,
        counts=counts,
        matrix_ids=matrix_ids,
        logs_descending=logs,
        cumulative_weight=cumulative_weight,
        cumulative_weighted_log=cumulative_log,
        total_weight=float(cumulative_weight[-1]),
        total_energy=float(np.sum(values * counts, dtype=np.float64)),
    )


def solve_waterfill(prepared: PreparedWaterfill, rate_bpw: float) -> tuple[float, float, np.ndarray]:
    if rate_bpw < 0.0:
        return math.inf, math.nan, np.zeros(prepared.values.size, dtype=np.float64)
    logs = prepared.logs_descending
    log_theta = (
        prepared.cumulative_weighted_log - 2.0 * rate_bpw * prepared.total_weight
    ) / prepared.cumulative_weight
    below_current = log_theta < logs + 1e-14
    above_next = np.ones(logs.size, dtype=bool)
    above_next[:-1] = log_theta[:-1] >= logs[1:] - 1e-14
    valid = np.flatnonzero(below_current & above_next)
    index = int(valid[0]) if valid.size else logs.size - 1
    selected_log_theta = float(log_theta[index])
    theta = 2.0 ** selected_log_theta
    active = prepared.values > theta
    rate_by_cell = np.zeros(prepared.values.size, dtype=np.float64)
    rate_by_cell[active] = (
        0.5
        * np.log2(prepared.values[active] / theta)
        * prepared.counts[active]
    )
    distortion_energy = float(
        np.sum(np.minimum(prepared.values, theta) * prepared.counts, dtype=np.float64)
    )
    relative_mse = distortion_energy / prepared.total_energy
    return relative_mse, theta, rate_by_cell


def evaluate_one(
    prepared: PreparedWaterfill,
    spec: ModelSpec,
    representation: str,
    rate: float,
    charged: bool,
) -> dict[str, Any]:
    side_matrix = side_bits_per_matrix(spec) if charged else 0
    side_total = side_matrix * 18 + (EXPERT_HEADER_BITS * EXPERTS if charged else 0)
    if charged and representation == "xklt":
        side_total += KLT_ANGLE_BITS * EXPERTS
    coefficient_rate = rate - side_total / TOTAL_VALUES
    mse, theta, rate_by_cell = solve_waterfill(prepared, coefficient_rate)
    coefficient_bits_matrix = np.bincount(
        prepared.matrix_ids.astype(np.int64), weights=rate_by_cell, minlength=18
    )
    expert_bits: list[float] = []
    for expert in range(EXPERTS):
        bits = float(np.sum(coefficient_bits_matrix[3 * expert : 3 * expert + 3]))
        if charged:
            bits += 3 * side_matrix + EXPERT_HEADER_BITS
            if representation == "xklt":
                bits += KLT_ANGLE_BITS
        expert_bits.append(bits)
    total_bits = float(sum(expert_bits))
    expected_total_bits = rate * TOTAL_VALUES
    rate_error_bits = total_bits - expected_total_bits
    mean_expert_bits = total_bits / EXPERTS
    read_amp = max(expert_bits) / mean_expert_bits
    f_value = mse * (2.0 ** (2.0 * rate))
    s_value = -0.5 * math.log2(f_value)
    return {
        "representation": representation,
        "model_id": spec.model_id,
        "family": spec.family,
        "parameters": list(spec.parameters),
        "accounting": "charged" if charged else "free_source_leaky_side",
        "physical_rate_bpw": rate,
        "coefficient_rate_bpw": coefficient_rate,
        "relative_mse": mse,
        "F": f_value,
        "s_bpw": s_value,
        "passes_F_le_0p8": bool(f_value <= TARGET_F),
        "water_level": theta,
        "side_bits_total": side_total,
        "side_bits_per_matrix": side_matrix,
        "field_parameter_count_per_matrix": max(0, (side_matrix - (MATRIX_HEADER_BITS + MEAN_BITS + GLOBAL_SCALE_BITS + MODEL_ID_BITS)) // 16) if charged else 0,
        "expert_physical_bits": expert_bits,
        "cold_expert_read_amplification": read_amp,
        "cold_read_strictly_below_2x": bool(read_amp < 2.0),
        "physical_total_bits": total_bits,
        "target_total_bits": expected_total_bits,
        "rate_closure_error_bits": rate_error_bits,
    }


def evaluate_dataset(
    dataset: dict[str, list[MatrixField]],
    specs: list[ModelSpec],
    representation: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        prepared = prepare_waterfill(dataset[spec.model_id])
        for rate in RATES:
            rows.append(evaluate_one(prepared, spec, representation, rate, False))
            rows.append(evaluate_one(prepared, spec, representation, rate, True))
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, float, str]:
    return (
        str(row["representation"]),
        str(row["model_id"]),
        float(row["physical_rate_bpw"]),
        str(row["accounting"]),
    )


def summarize_controls(
    source_rows: list[dict[str, Any]], control_runs: list[list[dict[str, Any]]]
) -> None:
    controls = [{row_key(row): row for row in run} for run in control_runs]
    for source in source_rows:
        key = row_key(source)
        control_mse = [float(run[key]["relative_mse"]) for run in controls]
        mean = float(np.mean(control_mse, dtype=np.float64))
        std = float(np.std(control_mse, ddof=1, dtype=np.float64)) if len(control_mse) > 1 else 0.0
        ratio = float(source["relative_mse"]) / mean
        source["matched_gaussian_control"] = {
            "replicate_relative_mse": control_mse,
            "mean_relative_mse": mean,
            "sample_std_relative_mse": std,
            "source_over_control_mean": ratio,
            "matched_structural_s_bpw": -0.5 * math.log2(ratio),
        }


def make_controls(
    representation_matrices: list[np.ndarray],
    provenance: dict[str, Any],
    representation: str,
    replicate: int,
) -> list[np.ndarray]:
    controls: list[np.ndarray] = []
    for ordinal, matrix in enumerate(representation_matrices):
        _, _, energy = centered(matrix)
        seed = source_seed(provenance["sources"][ordinal]["sha256"], representation, replicate)
        controls.append(matched_gaussian((ROWS, COLS), energy, seed))
    return controls


def best_row(rows: list[dict[str, Any]], accounting: str) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["accounting"] == accounting
        and row["cold_read_strictly_below_2x"]
        and 2.15 <= row["physical_rate_bpw"] <= 2.5
    ]
    return min(eligible, key=lambda row: (row["F"], -row["matched_gaussian_control"]["matched_structural_s_bpw"], row["model_id"]))


def compact_best(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "representation",
            "model_id",
            "family",
            "parameters",
            "accounting",
            "physical_rate_bpw",
            "coefficient_rate_bpw",
            "relative_mse",
            "F",
            "s_bpw",
            "passes_F_le_0p8",
            "side_bits_total",
            "cold_expert_read_amplification",
            "cold_read_strictly_below_2x",
            "matched_gaussian_control",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-replicates", type=int, default=4)
    args = parser.parse_args()
    if args.control_replicates < 2:
        raise ValueError("at least two matched-Gaussian replicates are required")

    specs = model_specs()
    raw, provenance = load_sources(args.source_lock, args.source_root)
    xklt, klt_receipts = xklt_representation(raw)
    representations = {"raw": raw, "xklt": xklt}
    all_rows: list[dict[str, Any]] = []
    moment_receipts: dict[str, list[dict[str, float]]] = {}

    for representation, matrices in representations.items():
        source_dataset, moments = build_dataset(matrices, specs)
        source_rows = evaluate_dataset(source_dataset, specs, representation)
        control_runs: list[list[dict[str, Any]]] = []
        for replicate in range(args.control_replicates):
            controls = make_controls(matrices, provenance, representation, replicate)
            control_dataset, _ = build_dataset(controls, specs)
            control_runs.append(evaluate_dataset(control_dataset, specs, representation))
        summarize_controls(source_rows, control_runs)
        all_rows.extend(source_rows)
        moment_receipts[representation] = moments

    best_free = best_row(all_rows, "free_source_leaky_side")
    best_charged = best_row(all_rows, "charged")
    passes = bool(best_charged["passes_F_le_0p8"] and best_charged["matched_gaussian_control"]["matched_structural_s_bpw"] >= TARGET_S)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "PROMOTE_TO_OPERATIONAL_CODEC" if passes else "EARLY_KILL_VARIANCE_FIELD_BRANCH",
        "decision": {
            "hard_gate": "charged F <= 0.8 and matched-Gaussian s >= 0.16096404744368115 at physical R in [2.15,2.5], with cold expert read <2x",
            "passes_hard_gate": passes,
            "best_free_source_leaky": compact_best(best_free),
            "best_physically_charged": compact_best(best_charged),
            "free_oracle_shortfall_s_bpw": max(0.0, TARGET_S - float(best_free["s_bpw"])),
            "charged_shortfall_s_bpw": max(0.0, TARGET_S - float(best_charged["s_bpw"])),
            "early_stop_reason": None if passes else "Even source-specific variance fields and ideal reverse waterfilling do not satisfy the same-rate promotion gate after matched-Gaussian calibration.",
        },
        "target": {
            "physical_rate_interval_bpw": [2.15, 2.5],
            "tested_rates_bpw": list(RATES),
            "required_F": TARGET_F,
            "required_s_bpw": TARGET_S,
            "required_cold_expert_read_amplification_strict_upper_bound": 2.0,
        },
        "protocol": {
            "cpu_only": True,
            "imports_torch": False,
            "imports_cupy": False,
            "invokes_cuda": False,
            "matrix_coverage": "all 18 pinned BF16 matrices, no sampling",
            "canonical_matrix_shape": [ROWS, COLS],
            "down_projection_operation": "transpose to 768x2048",
            "representations": ["raw", "source-derived two-channel Up/Down KLT"],
            "base_variance_tile": [TILE_ROWS, TILE_COLS],
            "base_variance_grid": [GRID_ROWS, GRID_COLS],
            "field_families": {
                "rowcol_separable": "exact outer product of source row and column centered-energy marginals",
                "rowcol_tile_ipf": "source-specific multiplicative maximum-entropy field matching all row, column, and 16x32 tile energy marginals",
                "lowrank_log": {"ranks": list(LOW_RANKS), "fit": "source-specific SVD of the 48x64 log tile-energy map"},
                "dct_log": {"rectangles": [list(pair) for pair in DCT_RECTS], "basis": "fixed analytic orthonormal DCT-II"},
                "haar_log": {"grids": [list(pair) for pair in HAAR_GRIDS], "basis": "fixed dyadic/piecewise-constant Haar approximation"},
                "tile_exact": "free full 48x64 log-variance-map ceiling",
            },
            "distortion_model": "independent Gaussian components with fitted variances, ideal continuous reverse waterfilling",
            "source_leakage": "Every field and KLT angle is fitted on the scored source. Free-side rows are intentionally impossible favorable ceilings.",
            "matched_gaussian": {
                "replicates": args.control_replicates,
                "generator": "NumPy PCG64DXSM; deterministic source-hash/representation/replicate seed",
                "moment_match": "exact zero mean and per-representation-matrix centered FP64 energy",
                "identical_path": "same source-leaky field fit, candidate grid, reverse waterfill, side charge, and selection",
            },
        },
        "physical_ledger": {
            "all_field_side_information_is_expert_private": True,
            "analytic_DCT_and_Haar_bases_shared_bytes": 0,
            "expert_header_bits": EXPERT_HEADER_BITS,
            "matrix_header_bits": MATRIX_HEADER_BITS,
            "matrix_mean_bits": MEAN_BITS,
            "matrix_global_scale_bits": GLOBAL_SCALE_BITS,
            "matrix_model_id_bits": MODEL_ID_BITS,
            "xklt_angle_bits_per_expert": KLT_ANGLE_BITS,
            "field_values_and_factors": "FP16",
            "coefficient_bits": "total physical budget minus every charged side bit",
            "cold_read_definition": "maximum private expert bits divided by one-sixth of total artifact bits; rate allocation is derived from the global water level",
        },
        "overlap_boundaries": {
            "conditional_hyperprior": "That audit entropy-coded short 2048-value tile-scale residuals and measured held-out Gaussian NLL gain. This oracle instead tests source-leaky long-range 2-D variance geometry and ideal same-rate MSE reverse waterfilling. The full tile ceiling overlaps only as a deliberately favorable stop bound.",
            "nanoquant_binary_factor": "NanoQuant used free iterative row/column equilibration before an SVD-tail or discrete binary-factor test. Here row/column scales are themselves the variance source for rate allocation; no low-rank weight reconstruction or binary factor is credited.",
            "claim_boundary": "A kill rejects the tested separable, low-rank-log, low-frequency DCT, and Haar variance allocation models. It is not a lower bound for arbitrary nonlinear or activation-aware codecs.",
        },
        "provenance": {
            **provenance,
            "algorithm_path": str(Path(__file__).resolve()),
            "algorithm_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "xklt": klt_receipts,
            "matrix_moments": moment_receipts,
        },
        "candidate_rows": all_rows,
    }
    sealed = seal(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": sealed["status"], "best_free": compact_best(best_free), "best_charged": compact_best(best_charged), "result_lock_sha256": sealed["result_lock_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
