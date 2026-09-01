#!/usr/bin/env python3
"""Leakage-controlled Haar/Stiefel orientation-entropy gate.

The experiment never needs a source-specific basis.  For a canonical matrix
``W`` with shape ``rows x cols`` it applies raw Householder QR to ``W.T``.
Reflector ``j`` is represented by the direction of the residual column on
``S^(cols-j-1)``.  For an iid Gaussian matrix these directions are independent
Haar spheres.  A small held-out angular-central-Gaussian (ACG) model is tested
against that exact intrinsic reference measure.

The default command is deliberately CPU-only.  ``--backend cupy`` exists for
production-scale math after the auxiliary promotion gate; it is never selected
automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROWS = 768
COLS = 2048
COORD_BINS = 16
REFLECTOR_BANDS = 8
ROLE_NAMES = ("input", "down")
REQUIRED_S = -0.5 * math.log2(0.8)
COMPOSITE_VARIANT = "role_gauge+polar"
COMPOSITE_RATE_KEY = "2.50"
CONFIDENCE_SE_MULTIPLIER = 3.0
MODEL_FP_BYTES = 2
GLOBAL_PREFIX_BYTES = 4096
EXPERTS_IN_PINNED_PANEL = 6
ROLES_PER_EXPERT = 3
EXPERT_FRAME_OVERHEAD_BYTES = 160
DEFAULT_RATES = (2.15, 2.25, 2.5)
SOURCE_RE = re.compile(r"l(?P<layer>\d+)e(?P<expert>\d+)_(?P<role>up|down)\.bf16\.bin$")


@dataclass(frozen=True)
class ExpertPair:
    layer: int
    expert: int
    up: Path
    down: Path


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk_bytes):
            digest.update(data)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_bf16(path: Path, role: str, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != rows * cols:
        raise ValueError(f"{path}: {raw.size} values, expected {rows * cols}")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if role == "down":
        return np.asarray(values.reshape(cols, rows).T, dtype=np.float32)
    return np.asarray(values.reshape(rows, cols), dtype=np.float32)


def discover_pairs(aux_dir: Path, target_lock: Path) -> tuple[list[ExpertPair], dict[str, Any]]:
    lock = json.loads(target_lock.read_text(encoding="utf-8"))
    matrices = lock.get("matrices")
    if not isinstance(matrices, list) or len(matrices) != 18:
        raise RuntimeError("target lock must bind the pinned 18-matrix panel")
    target_layers = {int(row["layer"]) for row in matrices}
    target_experts = {int(row["expert"]) for row in matrices}
    target_hashes = {
        f"L{int(row['layer'])}_E{int(row['expert'])}_{row['role']}": row["source_bf16_sha256"]
        for row in matrices
    }

    found: dict[tuple[int, int], dict[str, Path]] = {}
    for path in sorted(aux_dir.glob("*.bf16.bin")):
        match = SOURCE_RE.fullmatch(path.name)
        if match is None:
            continue
        key = (int(match.group("layer")), int(match.group("expert")))
        found.setdefault(key, {})[match.group("role")] = path

    pairs: list[ExpertPair] = []
    excluded: list[dict[str, int]] = []
    for (layer, expert), roles in sorted(found.items()):
        if set(roles) != {"up", "down"}:
            raise RuntimeError(f"incomplete auxiliary pair L{layer} E{expert}: {sorted(roles)}")
        if layer in target_layers or expert in target_experts:
            excluded.append({"layer": layer, "expert": expert})
            continue
        pairs.append(ExpertPair(layer, expert, roles["up"], roles["down"]))
    if len(pairs) < 12:
        raise RuntimeError(f"need at least 12 leakage-clean whole experts, found {len(pairs)}")
    return pairs, {
        "target_lock_path": str(target_lock.resolve()),
        "target_lock_sha256": sha256_file(target_lock),
        "target_layers": sorted(target_layers),
        "target_experts": sorted(target_experts),
        "target_matrix_sha256": target_hashes,
        "excluded_auxiliary_pairs": excluded,
        "pinned_source_payloads_opened": False,
    }


def orientation_dof(rows: int = ROWS, cols: int = COLS) -> int:
    return rows * cols - rows * (rows + 1) // 2


def triangular_dof(rows: int = ROWS) -> int:
    return rows * (rows + 1) // 2


def bin_edges_and_counts(rows: int, cols: int, coord_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.empty((rows, coord_bins), dtype=np.int64)
    ends = np.empty_like(starts)
    for j in range(rows):
        dimension = cols - j
        relative = (np.arange(coord_bins + 1, dtype=np.int64) * dimension) // coord_bins
        starts[j] = j + relative[:-1]
        ends[j] = j + relative[1:]
    counts = ends - starts
    if np.any(counts <= 0) or np.any(np.sum(counts, axis=1) != cols - np.arange(rows)):
        raise AssertionError("invalid coordinate bins")
    return starts, ends, counts


def _array_backend(name: str):
    if name == "numpy":
        return np
    if name != "cupy":
        raise ValueError(name)
    import cupy as cp  # imported only after an explicit --backend cupy request

    return cp


def _to_numpy(value: Any, backend: str) -> np.ndarray:
    if backend == "numpy":
        return np.asarray(value)
    import cupy as cp

    return cp.asnumpy(value)


def reconstruct_from_raw_qr(h: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Reconstruct the tall QR input from NumPy/LAPACK raw coordinates.

    This small, deliberately direct inverse is used by the unit tests to prove
    that the chart plus upper triangle is source-decodable.  Production code
    would apply the reflectors tilewise rather than materialize ``Q``.
    """

    h = np.asarray(h, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    rows, cols = h.shape
    if tau.shape != (rows,) or rows >= cols:
        raise ValueError((h.shape, tau.shape))
    upper = np.triu(h.T[:rows, :])
    q = np.eye(cols, rows, dtype=np.float64)
    for j in range(rows - 1, -1, -1):
        vector = np.concatenate((np.ones(1), h[j, j + 1 :]))
        q[j:] -= tau[j] * np.outer(vector, vector @ q[j:])
    return q @ upper


def householder_sphere_bins(
    matrix: np.ndarray,
    *,
    coord_bins: int = COORD_BINS,
    backend: str = "numpy",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return exact binned squared coordinates of canonical QR spheres.

    NumPy/CuPy raw QR stores the LAPACK reflector tails in row ``j`` of the
    transposed raw array.  LAPACK chooses

      beta = -copysign(||x||, x0), tau = 1 + |x0| / ||x||.

    Therefore the residual unit direction has squared coordinates
    ``(tau-1)^2`` at ``j`` and ``tau^2 * v_tail^2`` afterwards.  Squared
    coordinates are sufficient for the antipodally symmetric ACG density.
    """

    rows, cols = matrix.shape
    if rows >= cols:
        raise ValueError("expected a wide canonical matrix")
    xp = _array_backend(backend)
    source = xp.asarray(matrix.T, dtype=xp.float32)
    raw = xp.linalg.qr(source, mode="raw")
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise RuntimeError("backend raw QR did not return (reflectors, tau)")
    h, tau = raw
    if tuple(h.shape) != (rows, cols) or tuple(tau.shape) != (rows,):
        raise RuntimeError(f"unexpected raw QR shapes {h.shape}, {tau.shape}")

    rr = xp.arange(rows, dtype=xp.int64)[:, None]
    cc = xp.arange(cols, dtype=xp.int64)[None, :]
    q2 = (tau[:, None] * h) ** 2
    q2 *= cc > rr
    q2[xp.arange(rows), xp.arange(rows)] = (tau - 1.0) ** 2
    norms = xp.sum(q2, axis=1, dtype=xp.float64)

    starts_np, ends_np, counts = bin_edges_and_counts(rows, cols, coord_bins)
    starts = xp.asarray(starts_np)
    ends = xp.asarray(ends_np)
    prefix = xp.cumsum(q2, axis=1, dtype=xp.float64)
    row_index = xp.arange(rows, dtype=xp.int64)[:, None]
    energy = prefix[row_index, ends - 1]
    previous = xp.maximum(starts - 1, 0)
    energy -= xp.where(starts > 0, prefix[row_index, previous], 0.0)
    energy_np = _to_numpy(energy, backend).astype(np.float64, copy=False)
    norms_np = _to_numpy(norms, backend).astype(np.float64, copy=False)
    diagnostics = {
        "max_sphere_norm_abs_error": float(np.max(np.abs(norms_np - 1.0))),
        "max_binned_norm_abs_error": float(np.max(np.abs(np.sum(energy_np, axis=1) - 1.0))),
        "min_binned_energy": float(np.min(energy_np)),
    }
    if diagnostics["max_sphere_norm_abs_error"] > 2e-4:
        raise RuntimeError(f"raw Householder sphere reconstruction drift: {diagnostics}")
    return energy_np, counts, diagnostics


def acg_log_ratio_nats(energy: np.ndarray, counts: np.ndarray, log_shape: np.ndarray) -> np.ndarray:
    """Intrinsic ACG log-density ratio to normalized Haar measure.

    ``log_shape[b]`` is repeated for ``counts[..., b]`` diagonal entries.
    The expression includes the determinant term and is invariant to adding a
    constant to every log shape.  At zero it is exactly zero.
    """

    energy = np.asarray(energy, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    log_shape = np.asarray(log_shape, dtype=np.float64)
    dimension = np.sum(counts, axis=-1)
    quadratic = np.sum(energy * np.exp(-log_shape), axis=-1)
    if np.any(quadratic <= 0.0):
        raise FloatingPointError("non-positive ACG quadratic form")
    logdet = np.sum(counts * log_shape, axis=-1)
    return -0.5 * logdet - 0.5 * dimension * np.log(quadratic)


def _fit_one_band(
    energy: np.ndarray,
    counts: np.ndarray,
    *,
    ridge: float,
    iterations: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    energy = np.asarray(energy, dtype=np.float64).reshape(-1, energy.shape[-1])
    counts = np.asarray(counts, dtype=np.float64).reshape(-1, counts.shape[-1])
    if energy.shape != counts.shape:
        raise ValueError((energy.shape, counts.shape))
    bins = energy.shape[1]
    projection = np.eye(bins) - np.ones((bins, bins), dtype=np.float64) / bins
    g = np.zeros(bins, dtype=np.float64)

    def value_gradient_information(point: np.ndarray):
        centered = projection @ point
        weighted = energy * np.exp(-centered)[None, :]
        normalizer = np.sum(weighted, axis=1)
        probability = weighted / normalizer[:, None]
        dimension = np.sum(counts, axis=1)
        value = float(np.sum(-0.5 * np.sum(counts * centered[None, :], axis=1) - 0.5 * dimension * np.log(normalizer)))
        value -= 0.5 * ridge * float(np.dot(centered, centered))
        gradient = -0.5 * np.sum(counts, axis=0) + 0.5 * np.sum(dimension[:, None] * probability, axis=0)
        gradient = projection @ gradient - ridge * centered
        diag = np.sum(dimension[:, None] * probability, axis=0)
        outer = np.einsum("n,ni,nj->ij", dimension, probability, probability, optimize=True)
        information = 0.5 * (np.diag(diag) - outer) + ridge * projection + 1e-9 * np.eye(bins)
        return value, gradient, information

    value, gradient, information = value_gradient_information(g)
    initial_value = value
    accepted = 0
    for _ in range(iterations):
        if float(np.max(np.abs(gradient))) < 1e-9:
            break
        delta = np.linalg.solve(information, gradient)
        delta -= np.mean(delta)
        step = 1.0
        improved = False
        for _ in range(24):
            candidate = g + step * delta
            candidate -= np.mean(candidate)
            candidate_value, candidate_gradient, candidate_information = value_gradient_information(candidate)
            if candidate_value >= value - 1e-12:
                g = candidate
                value, gradient, information = candidate_value, candidate_gradient, candidate_information
                accepted += 1
                improved = True
                break
            step *= 0.5
        if not improved or step * float(np.linalg.norm(delta)) < 1e-10:
            break
    return g, {
        "initial_log_likelihood_nats": initial_value,
        "final_log_likelihood_nats": value,
        "accepted_newton_steps": accepted,
        "max_abs_gradient": float(np.max(np.abs(gradient))),
    }


def fit_shape_table(
    features: np.ndarray,
    counts: np.ndarray,
    *,
    reflector_bands: int = REFLECTOR_BANDS,
    ridge: float = 1.0,
    iterations: int = 24,
    quantize_fp16: bool = True,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Fit one tiny determinant-normalized diagonal ACG per reflector band."""

    features = np.asarray(features, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError("features must be [experts, reflectors, coordinate_bins]")
    experts, rows, bins = features.shape
    if counts.shape != (rows, bins):
        raise ValueError((counts.shape, (rows, bins)))
    table = np.zeros((reflector_bands, bins), dtype=np.float64)
    diagnostics = []
    for band in range(reflector_bands):
        lo = band * rows // reflector_bands
        hi = (band + 1) * rows // reflector_bands
        band_energy = features[:, lo:hi].reshape(experts * (hi - lo), bins)
        band_counts = np.broadcast_to(counts[None, lo:hi], (experts, hi - lo, bins)).reshape(-1, bins)
        table[band], row = _fit_one_band(band_energy, band_counts, ridge=ridge, iterations=iterations)
        diagnostics.append({"band": band, "reflector_start": lo, "reflector_stop": hi, **row})
    if quantize_fp16:
        table = table.astype(np.float16).astype(np.float64)
    return table, diagnostics


def score_shape_table(features: np.ndarray, counts: np.ndarray, table: np.ndarray) -> tuple[float, list[float]]:
    features = np.asarray(features, dtype=np.float64)
    experts, rows, _ = features.shape
    bands = table.shape[0]
    by_band = []
    total = 0.0
    for band in range(bands):
        lo = band * rows // bands
        hi = (band + 1) * rows // bands
        local_counts = np.broadcast_to(counts[None, lo:hi], features[:, lo:hi].shape)
        value = float(np.sum(acg_log_ratio_nats(features[:, lo:hi], local_counts, table[band])))
        by_band.append(value / math.log(2.0))
        total += value
    return total / math.log(2.0), by_band


def leave_one_expert_out(
    features: np.ndarray,
    counts: np.ndarray,
    *,
    ridge: float,
    iterations: int,
) -> dict[str, Any]:
    """Cross-fit two role classes while excluding each whole expert pair."""

    experts, roles, rows, bins = features.shape
    cols = int(np.sum(counts[0]))
    if roles != len(ROLE_NAMES):
        raise ValueError(features.shape)
    groups = []
    model_hashes = []
    for heldout in range(experts):
        train = np.arange(experts) != heldout
        role_bits = []
        role_band_bits = []
        fold_tables = []
        for role in range(roles):
            table, _ = fit_shape_table(features[train, role], counts, ridge=ridge, iterations=iterations)
            bits, bands = score_shape_table(features[heldout : heldout + 1, role], counts, table)
            role_bits.append(bits)
            role_band_bits.append(bands)
            fold_tables.append(table)
        model_blob = np.asarray(fold_tables, dtype=np.float16).tobytes(order="C")
        model_hashes.append(hashlib.sha256(model_blob).hexdigest())
        groups.append(
            {
                "heldout_index": heldout,
                "role_gain_bits": role_bits,
                "role_band_gain_bits": role_band_bits,
                "gain_bpw": float(sum(role_bits) / (roles * rows * cols)),
            }
        )

    final_tables = []
    final_fit = []
    for role in range(roles):
        table, diagnostics = fit_shape_table(features[:, role], counts, ridge=ridge, iterations=iterations)
        final_tables.append(table)
        final_fit.append(diagnostics)
    final_array = np.asarray(final_tables, dtype=np.float16)
    return {
        "groups": groups,
        "fold_model_sha256": model_hashes,
        "final_fp16_shape_table": final_array.astype(np.float64).tolist(),
        "final_fp16_shape_table_sha256": hashlib.sha256(final_array.tobytes(order="C")).hexdigest(),
        "final_fit_diagnostics": final_fit,
    }


def summarize_groups(values: list[float], multiplier: float = CONFIDENCE_SE_MULTIPLIER) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        raise ValueError("at least two whole-expert groups are required")
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    se = std / math.sqrt(x.size)
    return {
        "whole_expert_groups": int(x.size),
        "mean_s_bpw": mean,
        "sample_std_bpw": std,
        "standard_error_bpw": se,
        "confidence_se_multiplier": multiplier,
        "lower_bpw": mean - multiplier * se,
        "upper_bpw": mean + multiplier * se,
    }


def paired_summary(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_values = [float(row["gain_bpw"]) for row in left["groups"]]
    right_values = [float(row["gain_bpw"]) for row in right["groups"]]
    differences = [a - b for a, b in zip(left_values, right_values, strict=True)]
    return {"group_difference_bpw": differences, **summarize_groups(differences)}


def exact_rate_read_ledgers(rows: int = ROWS, cols: int = COLS) -> list[dict[str, Any]]:
    values_per_expert = ROLES_PER_EXPERT * rows * cols
    total_values = EXPERTS_IN_PINNED_PANEL * values_per_expert
    ledgers = []
    for requested_rate in DEFAULT_RATES:
        container_bytes = math.floor(requested_rate * total_values / 8.0)
        actual_rate = 8.0 * container_bytes / total_values
        available = container_bytes - GLOBAL_PREFIX_BYTES
        if available <= 0:
            raise AssertionError("prefix exceeds container")
        base, remainder = divmod(available, EXPERTS_IN_PINNED_PANEL)
        frame_bytes = [base + (index < remainder) for index in range(EXPERTS_IN_PINNED_PANEL)]
        offsets = []
        cursor = GLOBAL_PREFIX_BYTES
        for size in frame_bytes:
            offsets.append(cursor)
            cursor += size
        if cursor != container_bytes:
            raise AssertionError("frame packing mismatch")
        attribution = container_bytes / EXPERTS_IN_PINNED_PANEL
        exact_reads = [GLOBAL_PREFIX_BYTES + size for size in frame_bytes]
        page_reads = []
        for offset, size in zip(offsets, frame_bytes, strict=True):
            first_page = offset // 4096
            last_page = (offset + size - 1) // 4096
            frame_pages = last_page - first_page + 1
            page_reads.append(GLOBAL_PREFIX_BYTES + 4096 * frame_pages)
        ledgers.append(
            {
                "requested_rate_bpw": requested_rate,
                "physical_rate_bpw": actual_rate,
                "container_bytes": container_bytes,
                "global_prefix_bytes": GLOBAL_PREFIX_BYTES,
                "frame_bytes": frame_bytes,
                "frame_offsets": offsets,
                "expert_frame_overhead_bytes_inside_frame": EXPERT_FRAME_OVERHEAD_BYTES,
                "expert_payload_bytes": [size - EXPERT_FRAME_OVERHEAD_BYTES for size in frame_bytes],
                "physical_attribution_bytes_per_expert": attribution,
                "max_cold_exact_bytes": max(exact_reads),
                "max_cold_exact_amplification": max(exact_reads) / attribution,
                "max_cold_4k_bytes": max(page_reads),
                "max_cold_4k_amplification": max(page_reads) / attribution,
                "below_2x": max(page_reads) / attribution < 2.0,
            }
        )
    return ledgers


def deterministic_seed(base_seed: int, population: str, layer: int, expert: int, role: str) -> int:
    digest = hashlib.sha256(f"haar-acg-v1:{base_seed}:{population}:{layer}:{expert}:{role}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def controls_for(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(matrix.shape, dtype=np.float32)
    haar = z.copy()
    source64 = matrix.astype(np.float64)
    mean = float(np.mean(source64))
    centered_energy = float(np.sum((source64 - mean) ** 2, dtype=np.float64))
    z64 = z.astype(np.float64)
    z64 -= float(np.mean(z64))
    z_energy = float(np.sum(z64 * z64, dtype=np.float64))
    gaussian = mean + z64 * math.sqrt(centered_energy / z_energy)
    gaussian = gaussian.astype(np.float32)
    g64 = gaussian.astype(np.float64)
    achieved_mean = float(np.mean(g64))
    achieved_energy = float(np.sum((g64 - achieved_mean) ** 2, dtype=np.float64))
    return gaussian, haar, {
        "source_mean": mean,
        "source_centered_energy": centered_energy,
        "gaussian_mean": achieved_mean,
        "gaussian_centered_energy": achieved_energy,
        "relative_mean_error": abs(achieved_mean - mean) / max(abs(mean), 1e-30),
        "relative_centered_energy_error": abs(achieved_energy - centered_energy) / centered_energy,
    }


def load_composite_threshold(composite_result: Path) -> dict[str, Any]:
    result = json.loads(composite_result.read_text(encoding="utf-8"))
    row = result["variants"][COMPOSITE_VARIANT]["rates"][COMPOSITE_RATE_KEY]
    base_s = float(row["s_bpw"])
    base_f = float(row["F"])
    return {
        "artifact_path": str(composite_result.resolve()),
        "artifact_sha256": sha256_file(composite_result),
        "variant": COMPOSITE_VARIANT,
        "rate": float(row["physical_rate_bpw"]),
        "base_s_bpw": base_s,
        "base_F": base_f,
        "incremental_s_required_bpw": REQUIRED_S - base_s,
        "identity_check_incremental_s": -0.5 * math.log2(0.8 / base_f),
        "qualification": "optimistic auxiliary promotion threshold only; raw-role ACG gain is not added to the pinned role-KLT composite without a direct nested recomputation",
    }


def build_features(
    pairs: list[ExpertPair],
    *,
    backend: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]], dict[str, str]]:
    populations: dict[str, list[np.ndarray]] = {"qwen": [], "moment_gaussian": [], "haar": []}
    metadata = []
    source_hashes: dict[str, str] = {}
    expected_counts = None
    for index, pair in enumerate(pairs):
        expert_features = {name: [] for name in populations}
        expert_meta = {"index": index, "layer": pair.layer, "expert": pair.expert, "roles": []}
        for role_index, (role, path) in enumerate((("up", pair.up), ("down", pair.down))):
            matrix = load_bf16(path, role)
            source_hashes[path.name] = sha256_file(path)
            gaussian, haar, moments = controls_for(matrix, deterministic_seed(seed, "controls", pair.layer, pair.expert, role))
            role_meta = {"role": role, "path": str(path.resolve()), "sha256": source_hashes[path.name], "control_moments": moments, "diagnostics": {}}
            for name, candidate in (("qwen", matrix), ("moment_gaussian", gaussian), ("haar", haar)):
                values, counts, diagnostics = householder_sphere_bins(candidate, backend=backend)
                if expected_counts is None:
                    expected_counts = counts
                elif not np.array_equal(expected_counts, counts):
                    raise AssertionError("coordinate count schema changed")
                expert_features[name].append(values)
                role_meta["diagnostics"][name] = diagnostics
            expert_meta["roles"].append(role_meta)
        for name in populations:
            populations[name].append(np.stack(expert_features[name]))
        metadata.append(expert_meta)
        print(f"canonicalized {index + 1}/{len(pairs)} whole experts: L{pair.layer} E{pair.expert}", flush=True)
    assert expected_counts is not None
    return {name: np.stack(values) for name, values in populations.items()}, expected_counts, metadata, source_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aux-dir", type=Path, required=True)
    parser.add_argument("--target-lock", type=Path, required=True)
    parser.add_argument("--composite-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--seed", type=int, default=26090117)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--newton-iterations", type=int, default=24)
    args = parser.parse_args()
    started = time.time()

    pairs, target_binding = discover_pairs(args.aux_dir, args.target_lock)
    populations, counts, metadata, source_hashes = build_features(pairs, backend=args.backend, seed=args.seed)
    crossfit = {}
    for name, features in populations.items():
        print(f"cross-fitting {name}", flush=True)
        crossfit[name] = leave_one_expert_out(features, counts, ridge=args.ridge, iterations=args.newton_iterations)
        crossfit[name]["summary"] = summarize_groups([float(row["gain_bpw"]) for row in crossfit[name]["groups"]])

    qwen_minus_haar = paired_summary(crossfit["qwen"], crossfit["haar"])
    qwen_minus_gaussian = paired_summary(crossfit["qwen"], crossfit["moment_gaussian"])
    total_panel_values = EXPERTS_IN_PINNED_PANEL * ROLES_PER_EXPERT * ROWS * COLS
    model_bytes = len(ROLE_NAMES) * REFLECTOR_BANDS * COORD_BINS * MODEL_FP_BYTES
    model_rate_bpw = 8.0 * model_bytes / total_panel_values
    conservative_lower = min(float(qwen_minus_haar["lower_bpw"]), float(qwen_minus_gaussian["lower_bpw"])) - model_rate_bpw
    optimistic_upper = min(float(qwen_minus_haar["upper_bpw"]), float(qwen_minus_gaussian["upper_bpw"])) - model_rate_bpw
    composite = load_composite_threshold(args.composite_result)
    nested_needed = float(composite["incremental_s_required_bpw"])
    if conservative_lower >= REQUIRED_S:
        decision = "PROMOTE_STANDALONE_FINITE_RATE_CODEC"
    elif conservative_lower >= nested_needed:
        decision = "PROMOTE_DIRECT_NESTED_COMPOSITE_TEST"
    elif optimistic_upper < nested_needed:
        decision = "HARD_KILL_HAAR_ACG_ORIENTATION_ENTROPY"
    else:
        decision = "INCONCLUSIVE_EXPAND_AUXILIARY_PANEL"

    free_table_upper = min(float(qwen_minus_haar["upper_bpw"]), float(qwen_minus_gaussian["upper_bpw"]))
    free_table_multiplier = 2.0 ** (-2.0 * free_table_upper)
    charged_upper_multiplier = 2.0 ** (-2.0 * optimistic_upper)
    rd_rates = []
    for rate in DEFAULT_RATES:
        gaussian_mse = 2.0 ** (-2.0 * rate)
        rd_rates.append(
            {
                "rate_bpw": rate,
                "gaussian_reference_mse": gaussian_mse,
                "standalone_free_table_optimistic_mse": gaussian_mse * free_table_multiplier,
                "standalone_charged_optimistic_mse": gaussian_mse * charged_upper_multiplier,
                "nested_composite_free_table_optimistic_mse": gaussian_mse * float(composite["base_F"]) * free_table_multiplier,
                "target_mse": 0.8 * gaussian_mse,
            }
        )

    dimensions = {
        "matrix_values": ROWS * COLS,
        "householder_spheres": ROWS,
        "orientation_dof": orientation_dof(),
        "upper_triangular_dof": triangular_dof(),
        "sum_dof": orientation_dof() + triangular_dof(),
        "orientation_fraction": orientation_dof() / (ROWS * COLS),
        "formula_orientation": "sum_{j=0}^{rows-1}(cols-j-1) = rows*cols - rows*(rows+1)/2",
        "qr_volume_jacobian": "dW = constant * product_j |R_jj|^(cols-j-1) dmu_Stiefel(Q) dR",
        "stereographic_sphere_jacobian": "dmu(q) = Area(S^p)^-1 * (2/(1+||y||^2))^p dy, p=cols-j-1",
        "jacobian_handling": "ACG is scored as an intrinsic density ratio with respect to normalized Haar measure; the common sphere/chart Jacobian cancels exactly. No Euclidean chart NLL is treated as a coding gain.",
    }
    result = {
        "schema": "qwen_haar_manifold_entropy_gate_v1",
        "decision": decision,
        "claim_boundary": "source-blind auxiliary high-rate entropy promotion gate for one fixed binned diagonal ACG family; not an emitted codec, finite-rate MSE, or universal Stiefel converse",
        "protocol": {
            "strict_ptq": True,
            "backend": args.backend,
            "pinned_sources_opened": False,
            "heldout_unit": "one complete auxiliary expert; Up and Down are removed together",
            "eligible_auxiliary_experts": len(pairs),
            "roles": {"input": "auxiliary Up; a promoted target model would share this with Gate and Up", "down": "canonical transposed Down"},
            "fixed_model": f"{REFLECTOR_BANDS} reflector bands x {COORD_BINS} relative-coordinate bins, determinant-normalized diagonal ACG, role-conditioned",
            "model_selection": "none; resolution, ridge, FP16 table, confidence multiplier, and controls are fixed before the pinned panel is opened",
            "ridge": args.ridge,
            "newton_iterations": args.newton_iterations,
            "confidence_se_multiplier": CONFIDENCE_SE_MULTIPLIER,
            "control_protocol": "each auxiliary matrix gets both an identically QR-processed exact-moment Gaussian control and an identically QR-processed iid Gaussian/Haar control",
            "promotion_rule": "promote only if the three-SE Qwen-minus-both-controls lower bound, after the FP16 shared-table charge, reaches standalone required s or the optimistic incremental composite threshold",
            "kill_rule": "hard kill when even the three-SE Qwen-minus-both-controls upper bound misses the optimistic incremental composite threshold",
            "seed": args.seed,
        },
        "canonicalization": {
            "factorization": "W.T = Q R using deterministic raw Householder QR",
            "lapack_sign": "beta=-copysign(||x||,x0), tau=1+|x0|/||x||",
            "sphere_coordinates": "q0=-sign(beta)*(tau-1), q_tail=-sign(beta)*tau*v_tail; ACG uses their squares",
            "inverse": "the ordered reflectors reconstruct Q and the stored upper triangle reconstructs W.T=Q R",
            "measure_zero_ties": "zero leading residual uses the documented LAPACK copysign convention",
            "source_specific_basis_or_chart": False,
            "dimensions": dimensions,
        },
        "density": {
            "formula": "log(p_ACG/p_Haar) = -0.5 log|A| - 0.5 d log(q.T A^-1 q)",
            "shape": "A is diagonal and constant inside fixed (reflector-band, relative-coordinate-bin) cells",
            "shape_scale_invariance": "adding a constant to every log diagonal leaves the likelihood unchanged; fitting pins arithmetic mean(log A)=0",
            "serialized_model_bytes": model_bytes,
            "serialized_model_bpw_on_pinned_panel": model_rate_bpw,
            "final_qwen_table_sha256": crossfit["qwen"]["final_fp16_shape_table_sha256"],
        },
        "binding": {
            **target_binding,
            "auxiliary_source_sha256": source_hashes,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "auxiliary_metadata": metadata,
        "crossfit": crossfit,
        "comparisons": {
            "qwen_minus_haar": qwen_minus_haar,
            "qwen_minus_moment_gaussian": qwen_minus_gaussian,
            "conservative_lower_after_model_rate_bpw": conservative_lower,
            "optimistic_upper_after_model_rate_bpw": optimistic_upper,
        },
        "high_rate_rd_envelope": {
            "assumption": "ideal high-resolution entropy-constrained orientation quantization with zero chart-curvature, finite-cell, arithmetic-coder, or reconstruction penalty; a promotion oracle, not achieved finite-rate MSE",
            "identity": "an intrinsic entropy advantage s bpw can at best multiply the all-active Gaussian RD envelope by 2^(-2s)",
            "qwen_mean_F_multiplier": 2.0 ** (-2.0 * float(crossfit["qwen"]["summary"]["mean_s_bpw"])),
            "moment_gaussian_mean_F_multiplier": 2.0 ** (-2.0 * float(crossfit["moment_gaussian"]["summary"]["mean_s_bpw"])),
            "haar_mean_F_multiplier": 2.0 ** (-2.0 * float(crossfit["haar"]["summary"]["mean_s_bpw"])),
            "qwen_specific_free_table_three_se_upper_s_bpw": free_table_upper,
            "qwen_specific_free_table_F_multiplier": free_table_multiplier,
            "qwen_specific_charged_three_se_upper_s_bpw": optimistic_upper,
            "qwen_specific_charged_F_multiplier": charged_upper_multiplier,
            "nested_composite_free_table_optimistic_F": float(composite["base_F"]) * free_table_multiplier,
            "rates": rd_rates,
        },
        "thresholds": {
            "standalone_required_s_bpw": REQUIRED_S,
            "nested_composite": composite,
            "fraction_of_standalone_required_lower": conservative_lower / REQUIRED_S,
            "fraction_of_nested_required_upper": optimistic_upper / nested_needed,
        },
        "serialized_layout": {
            "global_prefix_contents": {
                "total_bytes": GLOBAL_PREFIX_BYTES,
                "fp16_acg_table_bytes": model_bytes,
                "fixed_chart_and_codec_spec_bytes": 0,
                "remaining_manifest_directory_hash_crc_and_reserve_bytes": GLOBAL_PREFIX_BYTES - model_bytes,
            },
            "expert_frame_contents": "160-byte header/directory/CRC inside each frame followed by independently entropy-coded Householder-orientation and upper-triangular index streams",
            "rate_read_ledgers": exact_rate_read_ledgers(),
            "decoder_locality": "one contiguous expert frame plus one 4 KiB global prefix; no other expert payload is read",
            "reconstruction_scope": "compressed-object read only; a survivor would need reflector reconstruction fused with GEMM to avoid materialized BF16 weight traffic",
        },
        "overlap_audit": {
            "stiefel_gram_oracle": "already tests polar/Gram energy and continuous-DOF waterfilling; this gate does not credit that split again",
            "composite_superoracle": "already nests role KLT and polar energy; its s is used only to define an optimistic promotion threshold, never added to this auxiliary score as a result",
            "polar_normal_predictor": "tests prediction of the symmetric polar normal field, whereas this gate models only the Stiefel orientation measure",
            "neural_flow_oracle": "tests scalar weight contexts and affine flows, not intrinsic QR orientation density",
            "nonduplication": "the only claimed opportunity is held-out Qwen-specific ACG entropy deficit relative to identically processed Gaussian/Haar orientations",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "runtime_seconds": time.time() - started,
    }
    result["result_content_sha256"] = canonical_json_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "qwen_mean_s_bpw": crossfit["qwen"]["summary"]["mean_s_bpw"],
                "conservative_lower_after_model_rate_bpw": conservative_lower,
                "optimistic_upper_after_model_rate_bpw": optimistic_upper,
                "nested_required_s_bpw": nested_needed,
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
