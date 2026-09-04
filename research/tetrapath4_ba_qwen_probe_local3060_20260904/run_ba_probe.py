"""Batched CuPy Blahut--Arimoto upper-bound probe for TETRAPATH-4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
SITE = WORKSPACE / ".venv-cupy/Lib/site-packages"
_DLL_HANDLES = []
for _directory in [WORKSPACE / ".tools/cuda_dlls_3060",
                   *sorted((SITE / "nvidia").glob("*/bin"))]:
    if not _directory.is_dir():
        raise RuntimeError(f"missing process-local CUDA DLL directory: {_directory}")
    _DLL_HANDLES.append(os.add_dll_directory(str(_directory)))
os.environ["CUDA_PATH"] = str(SITE / "nvidia/cuda_runtime")
os.environ["CUPY_CACHE_DIR"] = str(WORKSPACE / "tmp/tetrapath4_ba_qwen_cupy_cache_v0")

import cupy as cp


EXPECTED_UUID = "GPU-458a424a-76e3-65e5-0470-803e0ed131ca"
PAIR_BACKEND_SHA = "e16e657604be8f5ddd2858c6b8c49a8d548072afdbcef866e3895d366a45251c"
TUPLES = np.asarray([[a, b, c, d] for a in range(4) for b in range(4)
                     for c in range(4) for d in range(4)], dtype=np.int32)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bf16(path: Path, shape: tuple[int, int], transpose: bool) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    require(raw.size == math.prod(shape), f"BF16 geometry: {path}")
    value = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)
    if transpose:
        value = value.T
    return np.ascontiguousarray(value, dtype=np.float64).reshape(-1)


def ba_batch(cost: cp.ndarray, beta: float, max_iterations: int,
             tolerance: float) -> tuple[float, float, int, float]:
    """Return mean distortion, mutual information bits/sample, iterations."""
    blocks, samples, symbols = map(int, cost.shape)
    p = cp.full((blocks, symbols), 1.0 / symbols, dtype=cp.float64)
    iterations = 0
    delta = math.inf
    for iterations in range(1, max_iterations + 1):
        logits = cp.log(cp.maximum(p[:, None, :], 1e-300)) - beta * cost
        peak = cp.max(logits, axis=2, keepdims=True)
        log_z = peak + cp.log(cp.sum(cp.exp(logits - peak), axis=2, keepdims=True))
        posterior = cp.exp(logits - log_z)
        updated = cp.mean(posterior, axis=1)
        delta = float(cp.max(cp.abs(updated - p)).get())
        p = updated
        if delta <= tolerance:
            break
    logits = cp.log(cp.maximum(p[:, None, :], 1e-300)) - beta * cost
    peak = cp.max(logits, axis=2, keepdims=True)
    log_z = peak + cp.log(cp.sum(cp.exp(logits - peak), axis=2, keepdims=True))
    posterior = cp.exp(logits - log_z)
    distortion_by_block = cp.mean(cp.sum(posterior * cost, axis=2), axis=1)
    # At the BA stationary channel I = -beta E[d] - E[log Z] in nats.
    information_by_block = (-beta * distortion_by_block -
                            cp.mean(log_z[:, :, 0], axis=1)) / math.log(2.0)
    distortion = float(cp.mean(distortion_by_block).get())
    information = max(0.0, float(cp.mean(information_by_block).get()))
    return distortion, information, iterations, delta


def lower_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    by_rate: dict[float, float] = {}
    for rate, distortion in points:
        by_rate[rate] = min(distortion, by_rate.get(rate, math.inf))
    nondominated = []
    best = math.inf
    for point in sorted(by_rate.items()):
        if point[1] < best - 1e-14:
            nondominated.append(point)
            best = point[1]
    hull: list[tuple[float, float]] = []
    for point in nondominated:
        while len(hull) >= 2:
            slope_a = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
            slope_b = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
            if slope_a < slope_b - 1e-14:
                break
            hull.pop()
        hull.append(point)
    return hull


def interpolate_d(hull: list[tuple[float, float]], rate: float) -> float:
    for (r0, d0), (r1, d1) in zip(hull, hull[1:]):
        if rate <= r1 + 1e-12:
            fraction = (rate - r0) / (r1 - r0)
            return d0 + fraction * (d1 - d0)
    return hull[-1][1]


def compare_equal_rate(base: list[tuple[float, float]], challenger: list[tuple[float, float]]) -> dict:
    a, b = lower_hull(base), lower_hull(challenger)
    low, high = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    candidates = sorted({low, high, *(r for r, _ in a if low <= r <= high),
                         *(r for r, _ in b if low <= r <= high)})
    best = (-math.inf, None)
    for rate in candidates:
        da, db = interpolate_d(a, rate), interpolate_d(b, rate)
        gain = 0.5 * math.log2(da / db)
        best = max(best, (gain, [rate, da, db]), key=lambda x: x[0])
    return {"best_equivalent_gain_bpw": best[0],
            "witness_R_Dproduct_Dfull": best[1],
            "base_hull": a, "challenger_hull": b}


PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def curves(var_cost: cp.ndarray, betas: list[float], max_iterations: int,
           tolerance: float) -> tuple[list[dict], dict[str, list[dict]], list[dict]]:
    blocks, samples, variables, labels = map(int, var_cost.shape)
    require(variables == 4 and labels == 4, "four by four cost field")
    tuple_gpu = cp.asarray(TUPLES)
    pair_label_gpu = cp.asarray([[x, y] for x in range(4) for y in range(4)],
                               dtype=cp.int32)
    joint_cost = sum(var_cost[:, :, v, tuple_gpu[:, v]] for v in range(4))
    product_rows, full_rows = [], []
    pair_rows = {f"{a}{b}_{c}{d}": [] for (a, b), (c, d) in PAIRINGS}
    # Free block-conditioned zero-rate reproduction. The product and joint
    # constants are identical because distortion is additive.
    zero_d = float(cp.mean(sum(cp.min(cp.mean(var_cost[:, :, v, :], axis=1), axis=1)
                               for v in range(4))).get())
    product_rows.append({"beta": 0.0, "rate_bpw": 0.0, "relative_D": zero_d,
                         "iterations": 0})
    full_rows.append(dict(product_rows[0]))
    for rows in pair_rows.values():
        rows.append(dict(product_rows[0]))
    for beta in betas:
        product_d = 0.0
        product_i = 0.0
        product_iterations = 0
        product_delta = 0.0
        for variable in range(4):
            d, information, iterations, delta = ba_batch(
                var_cost[:, :, variable, :], beta, max_iterations, tolerance)
            product_d += d
            product_i += information
            product_iterations = max(product_iterations, iterations)
            product_delta = max(product_delta, delta)
        full_d, full_i, full_iterations, full_delta = ba_batch(
            joint_cost, beta, max_iterations, tolerance)
        product_rows.append({"beta": beta, "rate_bpw": product_i / 4.0,
                             "relative_D": product_d, "iterations": product_iterations,
                             "maximum_probability_delta": product_delta})
        full_rows.append({"beta": beta, "rate_bpw": full_i / 4.0,
                          "relative_D": full_d, "iterations": full_iterations,
                          "maximum_probability_delta": full_delta})
        for (a, b), (c, d) in PAIRINGS:
            name = f"{a}{b}_{c}{d}"
            pair_d = pair_i = 0.0
            pair_iterations = 0
            pair_delta = 0.0
            for first, second in ((a, b), (c, d)):
                local_cost = (var_cost[:, :, first, pair_label_gpu[:, 0]] +
                              var_cost[:, :, second, pair_label_gpu[:, 1]])
                d, information, iterations, delta = ba_batch(
                    local_cost, beta, max_iterations, tolerance)
                pair_d += d
                pair_i += information
                pair_iterations = max(pair_iterations, iterations)
                pair_delta = max(pair_delta, delta)
            pair_rows[name].append({"beta": beta, "rate_bpw": pair_i / 4.0,
                                    "relative_D": pair_d, "iterations": pair_iterations,
                                    "maximum_probability_delta": pair_delta})
    return product_rows, pair_rows, full_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-stride", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=80)
    args = parser.parse_args()
    require(args.block_stride > 0 and args.max_pairs > 0 and args.pair_index >= 0,
            "positive aperture")
    repo, source_root = args.repo.resolve(), args.source_root.resolve()
    backend_path = repo / "research/pairpath_p2_local3060_cupy_preflight_v0/pairpath_cupy_backend.py"
    require(sha256(backend_path) == PAIR_BACKEND_SHA, "pinned scale backend")
    backend = load_module("tetrapath_ba_scale_backend", backend_path)
    panel_path = repo / "research/same_layer_clustered_ib_entropy_gate_v0_qwen_deployment_20260903_r3/panel_lock.json"
    panel_bytes = panel_path.read_bytes()
    panel = json.loads(panel_bytes)

    props = cp.cuda.runtime.getDeviceProperties(0)
    device_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    uuid_hex = bytes(props["uuid"][:16]).hex()
    require(device_name == "NVIDIA GeForce RTX 3060" and
            uuid_hex == "458a424a76e365e50470803e0ed131ca", "pinned local GPU")

    entries: dict[int, dict[str, dict]] = {}
    for item in panel["files"]:
        path = source_root / item["relative_path"]
        require(path.is_file() and path.stat().st_size == item["bytes"] and
                sha256(path) == item["sha256"], f"payload binding {path}")
        entries.setdefault(int(item["expert"]), {})[item["role"]] = item
    experts = list(map(int, panel["experts"]))
    all_pairings = list(zip(experts[::2], experts[1::2]))
    pairings = all_pairings[args.pair_index:args.pair_index + args.max_pairs]
    require(pairings, "pair aperture")
    betas = [float(2.0 ** exponent) for exponent in range(-4, 19, 2)]
    results = []
    started = time.perf_counter()
    for pair_index, (expert_e, expert_f) in enumerate(pairings):
        rows = []
        for expert in (expert_e, expert_f):
            row = []
            for role in ("up", "down"):
                item = entries[expert][role]
                row.append(bf16(source_root / item["relative_path"],
                                tuple(item["raw_shape"]), bool(item["down_transposed"])))
            rows.append(row)
        values = np.ascontiguousarray(np.asarray(rows, dtype=np.float64))
        scale_bits = backend.estimate_scale_bits(values)
        block_ids = np.arange(0, values.shape[2] // backend.BLOCK_VALUES,
                              args.block_stride, dtype=np.int64)
        selected = values.reshape(2, 2, -1, backend.BLOCK_VALUES)[:, :, block_ids, :]
        scales = scale_bits.view(np.float16).astype(np.float64)[:, :, block_ids]
        variable_values = np.stack((selected[0, 0], selected[0, 1],
                                    selected[1, 0], selected[1, 1]), axis=2)
        variable_scales = np.stack((scales[0, 0], scales[0, 1],
                                    scales[1, 0], scales[1, 1]), axis=1)
        levels = variable_scales[:, None, :, None] * backend.LEVELS_RMS[None, None, None, :]
        mean_vector_energy = float(np.mean(np.sum(variable_values * variable_values, axis=2)))
        require(mean_vector_energy > 0 and math.isfinite(mean_vector_energy), "source energy")
        cost = ((cp.asarray(variable_values)[..., None] - cp.asarray(levels)) ** 2 /
                mean_vector_energy)

        product, source_pairs, source_full = curves(cost, betas, args.max_iterations, 1e-8)
        rng = np.random.Generator(np.random.PCG64(0x4241544554524100 + pair_index))
        control_cost = cp.empty_like(cost)
        for block in range(cost.shape[0]):
            for variable in range(4):
                permutation = cp.asarray(rng.permutation(cost.shape[1]))
                control_cost[block, :, variable, :] = cost[block, permutation, variable, :]
        _, control_pairs, control_full = curves(control_cost, betas, args.max_iterations, 1e-8)
        product_points = [(x["rate_bpw"], x["relative_D"]) for x in product]
        source_points = [(x["rate_bpw"], x["relative_D"]) for x in source_full]
        control_points = [(x["rate_bpw"], x["relative_D"]) for x in control_full]
        source_pair_points = [
            (x["rate_bpw"], x["relative_D"])
            for rows in source_pairs.values() for x in rows]
        control_pair_points = [
            (x["rate_bpw"], x["relative_D"])
            for rows in control_pairs.values() for x in rows]
        source_comparison = compare_equal_rate(product_points, source_points)
        control_comparison = compare_equal_rate(product_points, control_points)
        source_irreducible = compare_equal_rate(source_pair_points, source_points)
        control_irreducible = compare_equal_rate(control_pair_points, control_points)
        results.append({
            "experts": [expert_e, expert_f], "block_ids": block_ids.tolist(),
            "weights": int(4 * block_ids.size * backend.BLOCK_VALUES),
            "mean_vector_energy": mean_vector_energy,
            "product": product, "source_pairs": source_pairs, "source_full": source_full,
            "control_pairs": control_pairs, "control_full": control_full,
            "source_comparison": source_comparison,
            "control_comparison": control_comparison,
            "control_corrected_gain_bpw": (source_comparison["best_equivalent_gain_bpw"] -
                                            control_comparison["best_equivalent_gain_bpw"]),
            "source_full_over_best_2plus2": source_irreducible,
            "control_full_over_best_2plus2": control_irreducible,
            "control_corrected_irreducible_gain_bpw": (
                source_irreducible["best_equivalent_gain_bpw"] -
                control_irreducible["best_equivalent_gain_bpw"]),
        })
        del cost, control_cost
        cp.get_default_memory_pool().free_all_blocks()

    gains = np.asarray([x["source_comparison"]["best_equivalent_gain_bpw"] for x in results])
    corrected = np.asarray([x["control_corrected_gain_bpw"] for x in results])
    irreducible = np.asarray([x["control_corrected_irreducible_gain_bpw"] for x in results])
    report = {
        "schema": "tetrapath4.ba_label_flexible_qwen_local3060_probe.v0",
        "status": ("HARD_KILL_APERTURE_IRREDUCIBLE_FOURWAY_BELOW_0P045_BPW" if
                   float(irreducible.max()) < 0.045 else
                   "SURVIVES_IRREDUCIBLE_FOURWAY_APERTURE_REQUIRES_EXPANSION"),
        "pair_or_higher_status": (
            "BELOW_0P045_BPW" if float(corrected.max()) < 0.045 else
            "SURVIVES_APERTURE_REQUIRES_PAIR_VS_HIGHER_ATTRIBUTION"),
        "scientific_boundary": (
            "Stochastic BA relaxation with free block-conditioned distributions and free "
            "time sharing over fixed four-level reconstructions; not a finite codec or STRATA-RM6."),
        "aperture": {"block_stride": args.block_stride, "pair_index": args.pair_index,
                     "max_pairs": args.max_pairs,
                     "blocks_per_pair": len(results[0]["block_ids"]) if results else 0,
                     "betas": betas, "max_iterations": args.max_iterations},
        "threshold_bpw": 0.045,
        "aggregate": {"maximum_source_gain_bpw": float(gains.max()),
                      "mean_source_gain_bpw": float(gains.mean()),
                      "maximum_control_corrected_gain_bpw": float(corrected.max()),
                      "mean_control_corrected_gain_bpw": float(corrected.mean()),
                      "maximum_control_corrected_irreducible_gain_bpw": float(irreducible.max()),
                      "mean_control_corrected_irreducible_gain_bpw": float(irreducible.mean())},
        "panel": {"model": panel["model"], "revision": panel["revision"],
                  "layer": panel["layer"], "panel_lock_sha256": hashlib.sha256(panel_bytes).hexdigest()},
        "runtime": {"hostname": platform.node(), "python": list(sys.version_info[:3]),
                    "numpy": np.__version__, "cupy": cp.__version__,
                    "device_name": device_name, "device_uuid": EXPECTED_UUID,
                    "elapsed_seconds": time.perf_counter() - started},
        "pairs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"],
                      "output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
