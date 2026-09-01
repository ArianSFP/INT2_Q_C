#!/usr/bin/env python3
"""FP64 CuPy oracle for multi-band polar spectral flags."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np


N = 768
M = 2048
EXPERTS = 6
ROLES = 3
MATRICES = EXPERTS * ROLES
PANEL_VALUES = MATRICES * N * M
RATE = 2.5
TARGET_F = 0.8
GLOBAL_HEADER_BITS = 32_768
BOUNDARY_BITS_PER_MATRIX = N - 1
SIDE_BITS = GLOBAL_HEADER_BITS + MATRICES * BOUNDARY_BITS_PER_MATRIX
SIDE_BPW = SIDE_BITS / PANEL_VALUES
PAYLOAD_BPW = RATE - SIDE_BPW
STIEFEL_DOF = N * M - N * (N + 1) // 2
EQUAL_BANDS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
GREEDY_MAX_BANDS = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def check_plan(plan: dict[str, Any]) -> None:
    claimed = plan.get("lock_sha256")
    clean = dict(plan)
    clean.pop("lock_sha256", None)
    if hashlib.sha256(canonical(clean)).hexdigest() != claimed:
        raise ValueError("plan internal seal mismatch")
    if len(plan.get("sources", [])) != MATRICES:
        raise ValueError("plan source count mismatch")


def bf16_matrix(path: Path, shape: tuple[int, int], expected_hash: str) -> np.ndarray:
    if path.stat().st_size != 2 * math.prod(shape):
        raise ValueError(f"source byte count mismatch: {path}")
    observed = sha256_file(path)
    if observed != expected_hash:
        raise ValueError(f"source hash mismatch: {path}")
    words = np.fromfile(path, dtype="<u2")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"nonfinite source: {path}")
    return values.astype(np.float64).reshape(shape)


def prefix_statistics(singular: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate(([0.0], np.cumsum(singular, dtype=np.float64))),
        np.concatenate(([0.0], np.cumsum(np.square(singular), dtype=np.float64))),
    )


def segment_sse(prefix: np.ndarray, prefix2: np.ndarray, begin: int, stop: int) -> float:
    count = stop - begin
    total = float(prefix[stop] - prefix[begin])
    total2 = float(prefix2[stop] - prefix2[begin])
    return max(0.0, total2 - total * total / count)


def flag_row(
    singular: np.ndarray,
    boundaries: tuple[int, ...],
    family: str,
    parameter: Any,
) -> dict[str, Any]:
    edges = (0, *boundaries, N)
    sizes = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    if any(size <= 0 for size in sizes) or sum(sizes) != N:
        raise ValueError("invalid flag partition")
    prefix, prefix2 = prefix_statistics(singular)
    residual = float(
        sum(segment_sse(prefix, prefix2, edges[i], edges[i + 1]) for i in range(len(sizes)))
    )
    bands = len(sizes)
    flag_dof = (N * N - sum(size * size for size in sizes)) // 2 + bands
    model_dof = STIEFEL_DOF + flag_dof
    normal_dof = N * M - model_dof
    if not (0 < model_dof < N * M and normal_dof > 0 and residual > 0.0):
        raise ValueError("degenerate flag row")
    energy = float(np.dot(singular, singular))
    return {
        "family": family,
        "parameter": parameter,
        "bands": bands,
        "boundaries": list(boundaries),
        "multiplicities": sizes,
        "model_dof": model_dof,
        "normal_dof": normal_dof,
        "model_energy": energy - residual,
        "normal_energy": residual,
    }


def rank_curve(singular: np.ndarray) -> list[dict[str, Any]]:
    prefix, prefix2 = prefix_statistics(singular)
    energy = float(prefix2[-1])
    rows: list[dict[str, Any]] = []
    for rank in range(N - 1):
        width = N - rank
        sums = prefix[width:] - prefix[:-width]
        sums2 = prefix2[width:] - prefix2[:-width]
        errors = np.maximum(0.0, sums2 - np.square(sums) / width)
        start = int(np.argmin(errors))
        residual = float(errors[start])
        rank_dof = N * rank - rank * (rank - 1) // 2
        model_dof = STIEFEL_DOF + 1 + rank_dof
        normal_dof = N * M - model_dof
        if residual <= 0.0 or normal_dof <= 0:
            continue
        rows.append(
            {
                "family": "rank_window_baseline",
                "parameter": rank,
                "bands": rank + 1,
                "boundaries": [start, start + width],
                "multiplicities": [1] * rank + [width],
                "model_dof": model_dof,
                "normal_dof": normal_dof,
                "model_energy": energy - residual,
                "normal_energy": residual,
                "window_start": start,
                "window_stop": start + width,
            }
        )
    return rows


def equal_boundaries(bands: int) -> tuple[int, ...]:
    raw = [round(i * N / bands) for i in range(1, bands)]
    if len(set(raw)) != bands - 1:
        raise AssertionError("equal partition collision")
    return tuple(raw)


def greedy_flag_curve(singular: np.ndarray) -> list[dict[str, Any]]:
    prefix, prefix2 = prefix_statistics(singular)
    segments: list[tuple[int, int]] = [(0, N)]
    rows: list[dict[str, Any]] = []
    for bands in range(2, GREEDY_MAX_BANDS + 1):
        best: tuple[float, float, int, int] | None = None
        for segment_index, (begin, stop) in enumerate(segments):
            parent = segment_sse(prefix, prefix2, begin, stop)
            for split in range(begin + 1, stop):
                left = split - begin
                right = stop - split
                child = segment_sse(prefix, prefix2, begin, split) + segment_sse(
                    prefix, prefix2, split, stop
                )
                reduction = max(0.0, parent - child)
                added_dof = left * right + 1
                ratio = reduction / added_dof
                candidate = (ratio, reduction, -split, -segment_index)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            break
        split = -best[2]
        segment_index = -best[3]
        begin, stop = segments.pop(segment_index)
        segments.extend(((begin, split), (split, stop)))
        segments.sort()
        boundaries = tuple(stop for _, stop in segments[:-1])
        rows.append(flag_row(singular, boundaries, "greedy_flag", bands))
    return rows


def candidate_curve(singular: np.ndarray) -> list[dict[str, Any]]:
    rows = rank_curve(singular)
    rows.extend(
        flag_row(singular, (split,), "all_two_band_splits", split)
        for split in range(1, N)
    )
    rows.extend(
        flag_row(singular, equal_boundaries(bands), "equal_width_flag", bands)
        for bands in EQUAL_BANDS
    )
    rows.extend(greedy_flag_curve(singular))
    return rows


def waterfill(dimensions: np.ndarray, energies: np.ndarray, rate_bpw: float) -> dict[str, Any]:
    dimensions = np.asarray(dimensions, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    logv = np.log2(energies / dimensions)
    order = np.argsort(logv)[::-1]
    lv = logv[order]
    ds = dimensions[order]
    cd = np.cumsum(ds)
    cdlv = np.cumsum(ds * lv)
    levels = (cdlv - 2.0 * rate_bpw) / cd
    active_count = len(dimensions)
    for k in range(1, len(dimensions) + 1):
        level = levels[k - 1]
        if level <= lv[k - 1] + 2e-14 and (k == len(dimensions) or level >= lv[k] - 2e-14):
            active_count = k
            break
    log_level = float(levels[active_count - 1])
    level = 2.0**log_level
    active = logv > log_level
    bits = 0.5 * np.maximum(0.0, logv - log_level)
    used = float(np.sum(dimensions * bits, dtype=np.float64))
    if not math.isclose(used, rate_bpw, rel_tol=3e-10, abs_tol=3e-10):
        raise AssertionError((used, rate_bpw))
    distortion = float(np.sum(np.where(active, dimensions * level, energies), dtype=np.float64))
    return {
        "distortion": distortion,
        "water_level": level,
        "active_components": int(np.count_nonzero(active)),
        "used_rate_bpw": used,
    }


def score(curves: list[list[dict[str, Any]]], selection: list[int], total_energy: float) -> dict[str, Any]:
    dimensions: list[float] = []
    energies: list[float] = []
    for curve, index in zip(curves, selection, strict=True):
        row = curve[index]
        dimensions.extend((row["model_dof"] / PANEL_VALUES, row["normal_dof"] / PANEL_VALUES))
        energies.extend((row["model_energy"] / total_energy, row["normal_energy"] / total_energy))
    wf = waterfill(np.asarray(dimensions), np.asarray(energies), PAYLOAD_BPW)
    f_value = wf["distortion"] * 2.0 ** (2.0 * RATE)
    return {
        **wf,
        "F": f_value,
        "s_bpw": -0.5 * math.log2(f_value),
        "ideal_relative_mse": wf["distortion"],
        "passes_F_le_0p8": f_value <= TARGET_F,
    }


def local_objective(row: dict[str, Any], total_energy: float) -> float:
    dm = row["model_dof"] / PANEL_VALUES
    dn = row["normal_dof"] / PANEL_VALUES
    em = row["model_energy"] / total_energy
    en = row["normal_energy"] / total_energy
    return dm * math.log2(em / dm) + dn * math.log2(en / dn)


def select_curves(curves: list[list[dict[str, Any]]], total_energy: float) -> tuple[list[int], dict[str, Any]]:
    selection = [
        min(range(len(curve)), key=lambda i: local_objective(curve[i], total_energy))
        for curve in curves
    ]
    evaluations = 0
    passes = 0
    while passes < 6:
        changed = False
        for matrix_index, curve in enumerate(curves):
            current = selection[matrix_index]
            best = current
            best_f = score(curves, selection, total_energy)["F"]
            for candidate in range(len(curve)):
                if candidate == current:
                    continue
                selection[matrix_index] = candidate
                candidate_f = score(curves, selection, total_energy)["F"]
                evaluations += 1
                if candidate_f < best_f - 2e-14:
                    best_f = candidate_f
                    best = candidate
            selection[matrix_index] = best
            changed |= best != current
        passes += 1
        if not changed:
            break
    result = score(curves, selection, total_energy)
    result["coordinate_descent_passes"] = passes
    result["candidate_evaluations"] = evaluations
    return selection, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    plan_dir = args.plan_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    plan_path = plan_dir / "plan.lock.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    check_plan(plan)
    source_root = Path(plan["source_root"])

    spectra: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    energies: list[float] = []
    for ordinal, row in enumerate(plan["sources"]):
        role = str(row["role"])
        shape = (M, N) if role == "down" else (N, M)
        path = source_root / row["source_relpath"]
        matrix = bf16_matrix(path, shape, str(row["source_bf16_sha256"]))
        if role == "down":
            matrix = np.ascontiguousarray(matrix.T)
        energy = float(np.sum(matrix * matrix, dtype=np.float64))
        device = cp.asarray(matrix, dtype=cp.float64)
        singular = cp.asnumpy(cp.linalg.svd(device, compute_uv=False))
        singular.sort()
        spectral_energy = float(np.dot(singular, singular))
        if not math.isclose(energy, spectral_energy, rel_tol=3e-10, abs_tol=3e-8):
            raise FloatingPointError((ordinal, energy, spectral_energy))
        spectra.append(singular)
        energies.append(energy)
        receipts.append(
            {
                "matrix_ordinal": ordinal,
                "role": role,
                "source_relpath": row["source_relpath"],
                "source_bf16_sha256": row["source_bf16_sha256"],
                "source_energy_fp64": energy,
                "singular_values_sha256_f64": hashlib.sha256(
                    np.ascontiguousarray(singular, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        )
        print(f"spectrum {ordinal + 1:02d}/{MATRICES} role={role} energy={energy:.9f}", flush=True)
        del device, matrix

    total_energy = float(sum(energies))
    curves = [candidate_curve(singular) for singular in spectra]
    selection, best = select_curves(curves, total_energy)
    selected_rows = []
    for ordinal, (curve, index) in enumerate(zip(curves, selection, strict=True)):
        row = curve[index]
        selected_rows.append(
            {
                "matrix_ordinal": ordinal,
                "curve_index": index,
                "curve_candidates": len(curve),
                **row,
            }
        )
    family_histogram: dict[str, int] = {}
    for row in selected_rows:
        family_histogram[row["family"]] = family_histogram.get(row["family"], 0) + 1

    # Recompute the best rank-only baseline under the same conservative side debit.
    rank_curves = [[row for row in curve if row["family"] == "rank_window_baseline"] for curve in curves]
    rank_selection, rank_score = select_curves(rank_curves, total_energy)
    improvement_s = best["s_bpw"] - rank_score["s_bpw"]
    decision = (
        "GROSS_FLAG_SURVIVOR_REQUIRES_MATCHED_GAUSSIAN_CONTROLS"
        if best["F"] <= 0.85
        else "EARLY_KILL_FLAG_FAMILY_RAW_F_FAR_ABOVE_TARGET"
    )
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    gpu_name = properties["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")
    result = {
        "schema": "spectraflag_multi_band_polar_oracle_result_v0",
        "status": "complete",
        "decision": decision,
        "objective": {
            "physical_rate_bpw": RATE,
            "target_F": TARGET_F,
            "target_relative_mse": TARGET_F * 2.0 ** (-2.0 * RATE),
        },
        "ledger": {
            "global_header_bits": GLOBAL_HEADER_BITS,
            "boundary_bits_per_matrix": BOUNDARY_BITS_PER_MATRIX,
            "boundary_bits_total": MATRICES * BOUNDARY_BITS_PER_MATRIX,
            "side_bits_total": SIDE_BITS,
            "side_bpw": SIDE_BPW,
            "payload_bpw": PAYLOAD_BPW,
            "cold_read_claim": "planning-only expert-local payload plus small global header; below 1.02x",
        },
        "search": {
            "families": [
                "all prior rank/window rows",
                "all contiguous two-band splits",
                "equal-width 2/3/4/6/8/12/16/24/32/48/64-band flags",
                "greedy exact SSE-reduction-per-added-DOF path through 256 bands",
            ],
            "candidate_count_per_matrix": [len(curve) for curve in curves],
            "selected_family_histogram": family_histogram,
        },
        "best_flag_union": best,
        "rank_only_same_ledger": rank_score,
        "increment_over_rank": {
            "delta_s_bpw": improvement_s,
            "F_ratio_flag_over_rank": best["F"] / rank_score["F"],
        },
        "selected_rows": selected_rows,
        "rank_selection_indices": rank_selection,
        "bindings": {
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "plan_internal_lock_sha256": plan["lock_sha256"],
            "sources": receipts,
            "total_source_energy_fp64": total_energy,
        },
        "execution": {
            "backend": "cupy-fp64-svd plus numpy-fp64 curve search",
            "cupy_version": cp.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_id": int(cp.cuda.Device().id),
            "device_name": gpu_name,
            "elapsed_seconds": time.time() - started,
        },
        "claim_boundary": (
            "Exploratory ideal component oracle for the frozen partition search. "
            "No finite coordinates are encoded; a gross survivor requires matched "
            "Gaussian controls, Jacobian/metric correction, and a finite codec."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "F": best["F"],
                "s_bpw": best["s_bpw"],
                "rank_only_F": rank_score["F"],
                "delta_s_bpw": improvement_s,
                "selected_family_histogram": family_histogram,
                "elapsed_seconds": result["execution"]["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
