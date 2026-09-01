#!/usr/bin/env python3
"""CPU-only discrete NanoQuant-style binary-factor pilot on pinned Qwen tiles.

This ports the essential LB-ADMM/SVID initialization to NumPy, extracts the
actual binary signs, refits the two row/column scale vectors by alternating
least squares, rounds scales to FP16, and compares against an identically
optimized exact-moment Gaussian control.  It is a sampled architecture pilot,
not a full-matrix codec result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from nanoquant_binary_factor_oracle import (
    EXPERTS,
    ROLES,
    REQUIRED_USER_S,
    TARGET_RATES,
    canonical_json,
    read_panel,
    sha256_file,
    stable_seed,
)


RESULT_SCHEMA = "qwen-nanoquant-discrete-tile-pilot-v1"
HEADER_BITS_PER_EXPERT = 512


def deterministic_indices(total: int, count: int, seed: int) -> np.ndarray:
    count = min(total, count)
    stride = int(seed % total) | 1
    while math.gcd(stride, total) != 1:
        stride += 2
        if stride >= total:
            stride = 1
    start = int((seed >> 19) % total)
    return ((start + stride * np.arange(count, dtype=np.uint64)) % total).astype(np.int64)


def tiles(matrix: np.ndarray, tile_rows: int, tile_cols: int) -> np.ndarray:
    rows, cols = matrix.shape
    if rows % tile_rows or cols % tile_cols:
        raise ValueError("tile does not exactly partition matrix")
    return (
        matrix.reshape(rows // tile_rows, tile_rows, cols // tile_cols, tile_cols)
        .transpose(0, 2, 1, 3)
        .reshape(-1, tile_rows, tile_cols)
    )


def rank1_svid(matrix: np.ndarray, rng: np.random.Generator, inner_iters: int) -> np.ndarray:
    signs = np.where(matrix >= 0.0, 1.0, -1.0)
    magnitude = np.abs(matrix)
    v = rng.standard_normal(magnitude.shape[1], dtype=np.float64)
    v /= max(float(np.linalg.norm(v)), 1e-300)
    for _ in range(inner_iters):
        u = magnitude @ v
        u /= max(float(np.linalg.norm(u)), 1e-300)
        v = magnitude.T @ u
        v /= max(float(np.linalg.norm(v)), 1e-300)
    u_raw = magnitude @ v
    sigma = max(float(np.linalg.norm(u_raw)), 1e-300)
    u = u_raw / sigma
    return (np.outer(u * sigma, v) * signs).astype(np.float64)


def solve_step(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dual: np.ndarray,
    rho: float,
    reg: float,
) -> np.ndarray:
    system = x.T @ x
    system = 0.5 * (system + system.T)
    diag_mean = abs(float(np.mean(np.diag(system))))
    stabilizer = max(rho * diag_mean + reg, 1e-12)
    system.flat[:: system.shape[0] + 1] += stabilizer
    rhs = x.T @ y + rho * (z - dual)
    try:
        chol = np.linalg.cholesky(system)
        return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    except np.linalg.LinAlgError:
        return np.linalg.solve(system, rhs)


def fit_scales(
    target: np.ndarray,
    left_sign: np.ndarray,
    right_sign: np.ndarray,
    iterations: int,
    rank_scale_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = target.shape
    b = np.ones(cols, dtype=np.float64)
    a = np.ones(rows, dtype=np.float64)
    c = np.ones(left_sign.shape[1], dtype=np.float64)
    for _ in range(iterations):
        if rank_scale_bits:
            left_weighted = a[:, None] * left_sign
            right_weighted = right_sign * b[None, :]
            gram = (left_weighted.T @ left_weighted) * (right_weighted @ right_weighted.T)
            rhs = np.diag(left_weighted.T @ target @ right_weighted.T)
            ridge = max(float(np.mean(np.diag(gram))) * 1e-9, 1e-12)
            gram.flat[:: gram.shape[0] + 1] += ridge
            c = np.linalg.solve(gram, rhs)
        product = (left_sign * c[None, :]) @ right_sign
        pb = product * b[None, :]
        denominator = np.sum(pb * pb, axis=1, dtype=np.float64)
        a = np.sum(target * pb, axis=1, dtype=np.float64) / np.maximum(denominator, 1e-300)
        ap = a[:, None] * product
        denominator = np.sum(ap * ap, axis=0, dtype=np.float64)
        b = np.sum(target * ap, axis=0, dtype=np.float64) / np.maximum(denominator, 1e-300)
    # Balance the scale ambiguity before FP16 conversion.
    a_rms = max(float(np.sqrt(np.mean(a * a))), 1e-300)
    b_rms = max(float(np.sqrt(np.mean(b * b))), 1e-300)
    balance = math.sqrt(b_rms / a_rms)
    a *= balance
    b /= balance
    product = (left_sign * c[None, :]) @ right_sign
    reconstruction = a[:, None] * product * b[None, :]
    return a, b, c, reconstruction


def fit_one(
    tile: np.ndarray,
    rank: int,
    outer_iters: int,
    inner_iters: int,
    reg: float,
    scale_iters: int,
    rank_scale_bits: int,
    seed: int,
) -> dict[str, Any]:
    target = np.asarray(tile, dtype=np.float64)
    energy = float(np.sum(target * target, dtype=np.float64))
    rms = math.sqrt(energy / target.size)
    normalized = target / max(rms, 1e-300)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    rows, cols = target.shape
    a_ls = rng.standard_normal((rows, rank), dtype=np.float64)
    b_ls = rng.standard_normal((rank, cols), dtype=np.float64)
    a_z = rank1_svid(a_ls, rng, inner_iters)
    b_z = rank1_svid(b_ls, rng, inner_iters)
    a_u = a_ls - a_z
    b_u = b_ls - b_z
    history: list[dict[str, float]] = []
    for iteration in range(outer_iters):
        rho = iteration / max(outer_iters, 1)
        norm_b = np.maximum(np.linalg.norm(b_z, axis=1), 1e-12)
        x_a = b_z.T / norm_b[None, :]
        a_ls = solve_step(x_a, normalized.T, a_z.T, a_u.T, rho, reg).T
        norm_a = np.maximum(np.linalg.norm(a_z, axis=0), 1e-12)
        x_b = a_z / norm_a[None, :]
        b_ls = solve_step(x_b, normalized, b_z, b_u, rho, reg)
        a_z = rank1_svid(a_ls + a_u, rng, inner_iters)
        b_z = rank1_svid(b_ls + b_u, rng, inner_iters)
        a_u += a_ls - a_z
        b_u += b_ls - b_z
        if iteration in (0, outer_iters // 2, outer_iters - 1):
            a_sign = np.where(a_z >= 0.0, 1.0, -1.0)
            b_sign = np.where(b_z >= 0.0, 1.0, -1.0)
            _, _, _, recon = fit_scales(
                target, a_sign, b_sign, min(scale_iters, 4), rank_scale_bits
            )
            history.append(
                {
                    "iteration": float(iteration + 1),
                    "relative_mse": float(np.sum((target - recon) ** 2) / energy),
                }
            )
    a_sign = np.where(a_z >= 0.0, 1.0, -1.0)
    b_sign = np.where(b_z >= 0.0, 1.0, -1.0)
    row_scale, col_scale, rank_scale, reconstruction = fit_scales(
        target, a_sign, b_sign, scale_iters, rank_scale_bits
    )
    fp64_sse = float(np.sum((target - reconstruction) ** 2, dtype=np.float64))
    row_fp16 = row_scale.astype(np.float16).astype(np.float64)
    col_fp16 = col_scale.astype(np.float16).astype(np.float64)
    if rank_scale_bits == 16:
        rank_stored = rank_scale.astype(np.float16).astype(np.float64)
    elif rank_scale_bits == 0:
        rank_stored = rank_scale
    else:
        raise ValueError("rank-scale bits must be 0 or 16")
    product = (a_sign * rank_stored[None, :]) @ b_sign
    fp16_reconstruction = row_fp16[:, None] * product * col_fp16[None, :]
    fp16_sse = float(np.sum((target - fp16_reconstruction) ** 2, dtype=np.float64))
    return {
        "energy_fp64": energy,
        "fp64_scale_sse": fp64_sse,
        "fp16_scale_sse": fp16_sse,
        "fp64_scale_relative_mse": fp64_sse / energy,
        "fp16_scale_relative_mse": fp16_sse / energy,
        "binary_left_positive_fraction": float(np.mean(a_sign > 0.0)),
        "binary_right_positive_fraction": float(np.mean(b_sign > 0.0)),
        "rank_scale_bits": rank_scale_bits,
        "history": history,
    }


def rank_and_ledger(
    tile_rows: int, tile_cols: int, rate: float, rank_scale_bits: int
) -> dict[str, Any]:
    values = tile_rows * tile_cols
    axis = tile_rows + tile_cols
    rank = int(math.floor((rate * values - 16 * axis) / (axis + rank_scale_bits)))
    rank = max(rank, 1)
    useful_bits = rank * (axis + rank_scale_bits) + 16 * axis
    physical_bytes = math.ceil(rate * values / 8.0)
    physical_bits = 8 * physical_bytes
    if useful_bits > physical_bits:
        raise ValueError("rank does not fit tile physical budget")
    return {
        "requested_rate_bpw": rate,
        "physical_rate_bpw": physical_bits / values,
        "tile_shape": [tile_rows, tile_cols],
        "rank": rank,
        "rank_exceeds_min_dimension": rank > min(tile_rows, tile_cols),
        "binary_factor_bits": rank * axis,
        "rank_scale_bits": rank * rank_scale_bits,
        "rank_scale_storage_bits_each": rank_scale_bits,
        "fp16_scale_bits": 16 * axis,
        "useful_bits": useful_bits,
        "physical_bits": physical_bits,
        "padding_bits": physical_bits - useful_bits,
        "useful_bpw": useful_bits / values,
        "shared_table_bits": 0,
        "cold_read_amplification": 1.0,
    }


def parse_csv(text: str, cast: Any) -> list[Any]:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--representation", choices=("raw", "xklt"), default="raw")
    parser.add_argument("--tile-rows", type=int, default=48)
    parser.add_argument("--tile-cols", type=int, default=128)
    parser.add_argument("--tiles-per-matrix", type=int, default=1)
    parser.add_argument("--rates", default="2.5")
    parser.add_argument("--outer-iters", type=int, default=40)
    parser.add_argument("--inner-iters", type=int, default=5)
    parser.add_argument("--scale-iters", type=int, default=12)
    parser.add_argument("--reg", type=float, default=0.03)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--rank-scale-bits", type=int, choices=(0, 16), default=0)
    args = parser.parse_args()

    rates = parse_csv(args.rates, float)
    started = time.time()
    panel, provenance = read_panel(args.plan.resolve(strict=True))
    ledgers = [
        rank_and_ledger(args.tile_rows, args.tile_cols, rate, args.rank_scale_bits)
        for rate in rates
    ]
    observations: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for ledger in ledgers:
        rate = float(ledger["requested_rate_bpw"])
        rank = int(ledger["rank"])
        src_sse = np.zeros(EXPERTS, dtype=np.float64)
        gau_sse = np.zeros(EXPERTS, dtype=np.float64)
        energy = np.zeros(EXPERTS, dtype=np.float64)
        for expert in range(EXPERTS):
            for role in range(ROLES):
                bank = tiles(panel[args.representation][expert][role], args.tile_rows, args.tile_cols)
                index_seed = stable_seed(
                    "NQ-DISCRETE-TILE-SAMPLE-v1",
                    provenance["plan_lock_sha256"],
                    args.representation,
                    expert,
                    role,
                    args.tile_rows,
                    args.tile_cols,
                )
                indices = deterministic_indices(bank.shape[0], args.tiles_per_matrix, index_seed)
                for sample_ordinal, tile_index in enumerate(indices.tolist()):
                    source = np.asarray(bank[tile_index], dtype=np.float64)
                    source_mean = float(np.mean(source, dtype=np.float64))
                    source_centered = source - source_mean
                    source_centered_energy = float(
                        np.sum(source_centered * source_centered, dtype=np.float64)
                    )
                    source_energy = float(np.sum(source * source, dtype=np.float64))
                    control_seed = stable_seed(
                        "NQ-DISCRETE-CONTROL-v1",
                        provenance["plan_lock_sha256"],
                        args.representation,
                        expert,
                        role,
                        tile_index,
                    )
                    control_rng = np.random.Generator(np.random.PCG64DXSM(control_seed))
                    gaussian = control_rng.standard_normal(source.shape, dtype=np.float64)
                    gaussian -= float(np.mean(gaussian, dtype=np.float64))
                    gaussian *= math.sqrt(
                        source_centered_energy
                        / float(np.sum(gaussian * gaussian, dtype=np.float64))
                    )
                    gaussian += source_mean
                    # The centered Gaussian now has the source centered energy and
                    # adding the source mean also reproduces its total energy (up
                    # to FP64 summation roundoff).  Do not rescale after adding the
                    # mean: that would silently change the matched first moment.
                    init_seed = stable_seed(
                        "NQ-DISCRETE-INIT-v1",
                        args.tile_rows,
                        args.tile_cols,
                        rank,
                        sample_ordinal,
                    )
                    source_fits = []
                    gaussian_fits = []
                    for restart in range(args.restarts):
                        seed = (init_seed + restart * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
                        source_fits.append(
                            fit_one(
                                source,
                                rank,
                                args.outer_iters,
                                args.inner_iters,
                                args.reg,
                                args.scale_iters,
                                args.rank_scale_bits,
                                seed,
                            )
                        )
                        gaussian_fits.append(
                            fit_one(
                                gaussian,
                                rank,
                                args.outer_iters,
                                args.inner_iters,
                                args.reg,
                                args.scale_iters,
                                args.rank_scale_bits,
                                seed,
                            )
                        )
                    source_fit = min(source_fits, key=lambda row: row["fp16_scale_sse"])
                    gaussian_fit = min(gaussian_fits, key=lambda row: row["fp16_scale_sse"])
                    src_sse[expert] += source_fit["fp16_scale_sse"]
                    gau_sse[expert] += gaussian_fit["fp16_scale_sse"]
                    energy[expert] += source_energy
                    observations.append(
                        {
                            "requested_rate_bpw": rate,
                            "expert": expert,
                            "role": role,
                            "tile_index": tile_index,
                            "source": source_fit,
                            "gaussian": gaussian_fit,
                        }
                    )
                    print(
                        f"rate={rate} expert={expert} role={role} tile={tile_index} "
                        f"Dsrc={source_fit['fp16_scale_relative_mse']:.6f} "
                        f"Dg={gaussian_fit['fp16_scale_relative_mse']:.6f}",
                        flush=True,
                    )
        source_d = float(np.sum(src_sse) / np.sum(energy))
        gaussian_d = float(np.sum(gau_sse) / np.sum(energy))
        ratio = source_d / gaussian_d
        s_bpw = -0.5 * math.log2(ratio)
        physical_rate = float(ledger["physical_rate_bpw"])
        f_source = source_d * 2.0 ** (2.0 * physical_rate)
        aggregate.append(
            {
                "ledger": ledger,
                "source_sse_by_expert": src_sse.tolist(),
                "gaussian_sse_by_expert": gau_sse.tolist(),
                "energy_by_expert": energy.tolist(),
                "source_relative_mse": source_d,
                "gaussian_relative_mse": gaussian_d,
                "source_over_matched_gaussian": ratio,
                "structural_advantage_bpw": s_bpw,
                "F_ratio_identity": 2.0 ** (-2.0 * s_bpw),
                "source_F_equals_D_times_2pow2R": f_source,
                "passes_user_target": bool(f_source <= 0.8),
            }
        )

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "SAMPLED_DISCRETE_PILOT",
        "claim_boundary": (
            "Sampled tiles and NumPy LB-ADMM/SVID port; not full-panel encoded MSE and not a "
            "bitstream. FP16 scales and binary factor bits are physically ledgered."
        ),
        "gpu_policy": {
            "cpu_only": True,
            "imports_torch": False,
            "imports_cupy": False,
            "invokes_cuda": False,
        },
        "protocol": {
            "representation": args.representation,
            "tile_shape": [args.tile_rows, args.tile_cols],
            "tiles_per_matrix": args.tiles_per_matrix,
            "matrix_count": EXPERTS * ROLES,
            "outer_iters": args.outer_iters,
            "inner_iters": args.inner_iters,
            "scale_iters": args.scale_iters,
            "reg": args.reg,
            "restarts": args.restarts,
            "optimizer": "NumPy LB-ADMM/SVID then exact-sign row/column ALS",
            "scale_storage": "FP16",
            "rank_scale_storage_bits_each": args.rank_scale_bits,
            "binary_factor_storage": "1 bit per sign",
            "control": "same optimizer/init on Gaussian tile with exact source energy and mean",
        },
        "target": {
            "required_F": 0.8,
            "required_s_bpw": REQUIRED_USER_S,
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
                for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
            },
        },
        "ledgers": ledgers,
        "observations": observations,
        "aggregate": aggregate,
        "elapsed_seconds": time.time() - started,
    }
    result["result_lock_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "output": str(args.output),
                "result_lock_sha256": result["result_lock_sha256"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
