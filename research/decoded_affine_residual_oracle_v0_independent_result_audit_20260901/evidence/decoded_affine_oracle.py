#!/usr/bin/env python3
"""Favourable CuPy envelope for affine corrections to decoded STRATA weights.

This is an early-kill oracle, not a finite codec.  It gives each tested cell
exact FP64 source-fitted coefficients while charging only their nominal FP16
payload.  A cell that cannot reach F <= 0.8 under that advantage does not
justify a finite implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path


EXPERTS = 6
ROLES = 3
ROWS = 768
COLS = 2048
GROUPS_PER_EXPERT = 2304
GROUPS = EXPERTS * GROUPS_PER_EXPERT
VALUES = GROUPS * COLS
BASELINE_SSE = 500.39553685426534
BASELINE_ENERGY = 16192.89450885593
BASELINE_F = 0.9888693569009007
TARGET_F = 0.8
WIDTHS = (2048, 512, 128, 32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_to_f64(np, path: Path, shape: tuple[int, int]):
    raw = path.read_bytes()
    words = np.frombuffer(raw, dtype="<u2")
    if words.size != shape[0] * shape[1]:
        raise ValueError(f"source geometry mismatch: {path}")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return values.astype(np.float64).reshape(shape)


def affine_sse(cp, x, y, width: int, mode: str) -> float:
    xx = x.reshape(-1, width)
    yy = y.reshape(-1, width)
    if mode == "scale":
        denominator = cp.sum(xx * xx, axis=1, keepdims=True, dtype=cp.float64)
        slope = cp.where(denominator > 0.0,
                         cp.sum(xx * yy, axis=1, keepdims=True, dtype=cp.float64) / denominator,
                         0.0)
        residual = yy - slope * xx
    elif mode == "bias":
        bias = cp.mean(yy - xx, axis=1, keepdims=True, dtype=cp.float64)
        residual = yy - (xx + bias)
    elif mode == "affine":
        mx = cp.mean(xx, axis=1, keepdims=True, dtype=cp.float64)
        my = cp.mean(yy, axis=1, keepdims=True, dtype=cp.float64)
        centered_x = xx - mx
        denominator = cp.sum(centered_x * centered_x, axis=1, keepdims=True,
                             dtype=cp.float64)
        slope = cp.where(
            denominator > 0.0,
            cp.sum(centered_x * (yy - my), axis=1, keepdims=True,
                   dtype=cp.float64) / denominator,
            0.0,
        )
        residual = yy - (slope * xx + (my - slope * mx))
    else:
        raise ValueError(mode)
    return float(cp.sum(residual * residual, dtype=cp.float64).item())


def two_way_bias_sse(cp, x, y) -> float:
    residual = y - x
    row = cp.mean(residual, axis=1, keepdims=True, dtype=cp.float64)
    col = cp.mean(residual, axis=0, keepdims=True, dtype=cp.float64)
    grand = cp.mean(residual, dtype=cp.float64)
    corrected = residual - row - col + grand
    return float(cp.sum(corrected * corrected, dtype=cp.float64).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    plan_dir = args.plan_dir.resolve(strict=True)

    import cupy as cp
    import numpy as np

    started = time.time()
    plan_path = plan_dir / "plan.lock.json"
    header_path = plan_dir / "header.bin"
    post_path = plan_dir / "independent_audit" / "post_klt_canonical_groups.f64.bin"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if len(plan.get("sources", [])) != 18:
        raise ValueError("expected 18 authenticated sources")
    header = header_path.read_bytes()
    coefficients = struct.unpack_from("<12f", header, 32)
    post = np.memmap(post_path, dtype="<f8", mode="r", shape=(GROUPS, COLS))
    source_root = Path(plan["source_root"])

    modes = ("scale", "bias", "affine")
    totals = {(mode, width): 0.0 for mode in modes for width in WIDTHS}
    two_way_total = 0.0
    baseline_sse = 0.0
    source_energy = 0.0
    matrix_rows = []

    for expert in range(EXPERTS):
        base = expert * GROUPS_PER_EXPERT
        gate_hat = np.asarray(post[base:base + ROWS])
        z0 = np.asarray(post[base + ROWS:base + 2 * ROWS])
        z1 = np.asarray(post[base + 2 * ROWS:base + 3 * ROWS])
        cosine = float(coefficients[2 * expert])
        sine = float(coefficients[2 * expert + 1])
        norm2 = cosine * cosine + sine * sine
        up_hat = (cosine * z0 - sine * z1) / norm2
        down_hat = (sine * z0 + cosine * z1) / norm2
        for local_role, reconstruction in enumerate((gate_hat, up_hat, down_hat)):
            ordinal = 3 * expert + local_role
            row = plan["sources"][ordinal]
            path = source_root / row["source_relpath"]
            if sha256_file(path) != row["source_bf16_sha256"]:
                raise ValueError(f"source hash mismatch at ordinal {ordinal}")
            shape = tuple(row["shape"])
            source = bf16_to_f64(np, path, shape)
            natural = source.T.copy() if row["role"] == "down" else source.copy()
            if natural.shape != (ROWS, COLS):
                raise ValueError(f"natural geometry at ordinal {ordinal}")
            x = cp.asarray(reconstruction, dtype=cp.float64)
            y = cp.asarray(natural, dtype=cp.float64)
            error = y - x
            raw_sse = float(cp.sum(error * error, dtype=cp.float64).item())
            energy = float(cp.sum(y * y, dtype=cp.float64).item())
            baseline_sse += raw_sse
            source_energy += energy
            cell_sse = {}
            for mode in modes:
                for width in WIDTHS:
                    value = affine_sse(cp, x, y, width, mode)
                    totals[(mode, width)] += value
                    cell_sse[f"{mode}_w{width}"] = value
            tw = two_way_bias_sse(cp, x, y)
            two_way_total += tw
            matrix_rows.append({
                "matrix_ordinal": ordinal,
                "role": row["role"],
                "baseline_sse": raw_sse,
                "source_energy": energy,
                "oracle_sse": cell_sse,
                "two_way_bias_sse": tw,
            })
            del x, y, error

    if abs(baseline_sse - BASELINE_SSE) > 2e-9 or abs(source_energy - BASELINE_ENERGY) > 2e-9:
        raise RuntimeError(f"baseline replay mismatch: {baseline_sse}, {source_energy}")

    cells = []
    for mode in modes:
        coefficient_count = 1 if mode in ("scale", "bias") else 2
        for width in WIDTHS:
            side_bpw = 16.0 * coefficient_count / width
            ratio = totals[(mode, width)] / baseline_sse
            favourable_net_f = BASELINE_F * ratio * 2.0 ** (2.0 * side_bpw)
            cells.append({
                "family": mode,
                "width": width,
                "exact_fp64_oracle_sse": totals[(mode, width)],
                "fraction_of_baseline_sse": ratio,
                "nominal_fp16_coefficient_bpw": side_bpw,
                "favourable_transfer_F": favourable_net_f,
                "passes_target": favourable_net_f <= TARGET_F,
            })
    two_way_side_bpw = 16.0 * (ROWS + COLS - 1) / (ROWS * COLS)
    two_way_ratio = two_way_total / baseline_sse
    two_way_f = BASELINE_F * two_way_ratio * 2.0 ** (2.0 * two_way_side_bpw)
    cells.append({
        "family": "row_plus_column_bias",
        "width": None,
        "exact_fp64_oracle_sse": two_way_total,
        "fraction_of_baseline_sse": two_way_ratio,
        "nominal_fp16_coefficient_bpw": two_way_side_bpw,
        "favourable_transfer_F": two_way_f,
        "passes_target": two_way_f <= TARGET_F,
    })
    cells.sort(key=lambda row: (row["favourable_transfer_F"], row["family"],
                                -1 if row["width"] is None else -row["width"]))
    result = {
        "schema": "decoded-affine-residual-oracle-v0",
        "status": ("PROMOTE_FINITE_FOLLOWUP" if any(row["passes_target"] for row in cells)
                   else "HARD_KILL_AFFINE_CORRECTION_FAMILY"),
        "claim_boundary": (
            "Early-kill envelope on the authenticated decoded STRATA checkpoint. Exact FP64 "
            "source-fitted coefficients are granted while only nominal FP16 coefficient bits are "
            "charged; transfer of correction fraction to a lower-rate coarse stream is also granted."
        ),
        "baseline": {"sse": baseline_sse, "source_energy": source_energy,
                     "relative_mse": baseline_sse / source_energy, "F": BASELINE_F},
        "target_F": TARGET_F,
        "cells": cells,
        "best": cells[0],
        "matrices": matrix_rows,
        "bindings": {
            "plan_sha256": sha256_file(plan_path),
            "header_sha256": sha256_file(header_path),
            "decoded_post_klt_sha256": sha256_file(post_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime": {"elapsed_seconds": time.time() - started, "python": os.sys.version,
                    "numpy": np.__version__, "cupy": cp.__version__,
                    "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"])},
    }
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": result["status"], "best": result["best"],
                      "result_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
