#!/usr/bin/env python3
"""CuPy upper aperture for compact mixed-matrix STRATA residual predictors."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
SITE = WORKSPACE / ".venv-cupy/Lib/site-packages"
_DLL_HANDLES = []
for _directory in [WORKSPACE / ".tools/cuda_dlls_3060",
                   *sorted((SITE / "nvidia").glob("*/bin"))]:
    if not _directory.is_dir():
        raise RuntimeError(f"missing CUDA DLL directory: {_directory}")
    _DLL_HANDLES.append(os.add_dll_directory(str(_directory)))
os.environ["CUDA_PATH"] = str(SITE / "nvidia/cuda_runtime")
os.environ["CUPY_CACHE_DIR"] = str(WORKSPACE / "tmp/operator_innovation_cupy_cache_v0")

import cupy as cp

sys.path.insert(0, str(REPO))
from strata_expert_local_codec import common
from strata_expert_local_codec import independent_audit as baseline_decoder


EXPECTED_DEVICE = "NVIDIA GeForce RTX 3060"
EXPECTED_UUID_HEX = "458a424a76e365e50470803e0ed131ca"
EXPECTED_CONTAINER_SHA = "4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b"
EXPECTED_POST_SHA = "af801b41a37774d3f0ea65a00d929ff0004122caf4a5632457dbbe232e3f84d0"
EXPECTED_BASELINE_SSE = 500.39553685426534
EXPECTED_SOURCE_ENERGY = 16192.89450885593
BASELINE_F = 0.9888693569009007
TARGET_F = 0.8
ROLES = ("gate", "up", "down")
ROWS = 768
COLS = 2048


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def bf16(path: Path, shape: tuple[int, int], transpose: bool) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    require(words.size == math.prod(shape), f"BF16 geometry: {path}")
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)
    if transpose:
        values = values.T
    return np.ascontiguousarray(values, dtype=np.float32)


def decode_worker(arguments: tuple[str, str, dict[str, Any], int]) -> dict[str, Any]:
    # This wrapper makes the spawned Windows process import this module first,
    # including the process-local CUDA DLL setup above.
    return baseline_decoder.decode_block_worker(arguments)


def regenerate_post(release: Path, scratch: Path, workers: int) -> Path:
    """Causally decode the literal checkpoint and restore canonical group order."""
    existing = scratch / "post_klt_canonical_groups.f64.bin"
    if existing.is_file():
        require(existing.stat().st_size == common.GROUPS * common.GROUP_VALUES * 8,
                "existing decoded reconstruction size")
        require(sha256(existing) == EXPECTED_POST_SHA,
                "existing historical decoded reconstruction hash")
        return existing
    plan = json.loads((release / "plan.lock.json").read_text(encoding="utf-8"))
    summary = json.loads((release / "summary.json").read_text(encoding="utf-8"))
    container = release / summary["artifact"]["relpath"]
    require(sha256(container) == EXPECTED_CONTAINER_SHA, "container binding")
    parsed = baseline_decoder.parse_container(container, plan)
    scratch.mkdir(parents=True, exist_ok=False)
    decoded_dir = scratch / "decoded"
    decoded_dir.mkdir()
    tasks = []
    for ordinal, row in enumerate(parsed["directory"]):
        output = decoded_dir / f"block_{ordinal:02d}.f64.bin"
        tasks.append((str(container), str(output), row, common.BLOCK_LOG2[ordinal]))
    decoded = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(decode_worker, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            decoded.append(row)
            print(json.dumps({"decoded_block": row["block_ordinal"],
                              "of": common.BLOCKS}), flush=True)
    decoded.sort(key=lambda row: int(row["block_ordinal"]))
    labels = parsed["labels"]
    ordinals_by_block = common.expected_block_group_ordinals(labels)
    post_path = scratch / "post_klt_canonical_groups.f64.bin"
    post = np.memmap(post_path, dtype="<f8", mode="w+",
                     shape=(common.GROUPS, common.GROUP_VALUES))
    coverage = np.zeros(common.GROUPS, dtype=np.uint8)
    for row, ordinals in zip(decoded, ordinals_by_block, strict=True):
        values = np.memmap(row["output_path"], dtype="<f8", mode="r",
                           shape=(int(row["values"]),)).reshape(-1, common.GROUP_VALUES)
        post[ordinals] = values
        coverage[ordinals] += 1
    post.flush()
    require(bool(np.all(coverage == 1)), "canonical group coverage")
    require(sha256(post_path) == EXPECTED_POST_SHA, "historical decoded reconstruction hash")
    return post_path


def solve_psd(gram: np.ndarray, cross: np.ndarray) -> tuple[np.ndarray, int, float]:
    symmetric = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(symmetric)
    peak = max(float(values[-1]), 0.0)
    keep = values > max(peak * 2e-6, 1e-12)
    require(bool(np.any(keep)), "empty feature span")
    coefficients = vectors[:, keep] @ ((vectors[:, keep].T @ cross) / values[keep])
    condition = float(values[keep][-1] / values[keep][0])
    return coefficients, int(np.count_nonzero(keep)), condition


def normalized_stack(features: list[tuple[str, cp.ndarray]]) -> tuple[list[str], cp.ndarray, list[float]]:
    names: list[str] = []
    rows: list[cp.ndarray] = []
    scales: list[float] = []
    for name, value in features:
        flat = cp.asarray(value, dtype=cp.float32).reshape(-1)
        rms = math.sqrt(float(cp.mean(flat.astype(cp.float64) ** 2).get()))
        require(math.isfinite(rms) and rms > 1e-18, f"degenerate feature {name}")
        names.append(name)
        rows.append(flat / np.float32(rms))
        scales.append(rms)
    return names, cp.stack(rows, axis=0), scales


def feature_statistics(stack: cp.ndarray, residual: cp.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    residual64 = residual.astype(cp.float64)
    sse = float(cp.sum(residual64 * residual64, dtype=cp.float64).get())
    # FP32 GEMM is the optimized path; the selected reconstruction is rescored in FP64.
    gram = cp.asnumpy(stack @ stack.T).astype(np.float64)
    cross = cp.asnumpy(stack @ residual.reshape(-1)).astype(np.float64)
    return gram, cross, sse


def select_stats(names: list[str], gram: np.ndarray, cross: np.ndarray,
                 selected: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.asarray([names.index(name) for name in selected], dtype=np.int64)
    return index, gram[np.ix_(index, index)], cross[index]


def fit_and_rescore(stack: cp.ndarray, residual: cp.ndarray, index: np.ndarray,
                    gram: np.ndarray, cross: np.ndarray) -> dict[str, Any]:
    coefficients, rank, condition = solve_psd(gram, cross)
    prediction = cp.asarray(coefficients.astype(np.float32)) @ stack[index]
    delta = residual.reshape(-1).astype(cp.float64) - prediction.astype(cp.float64)
    sse = float(cp.sum(delta * delta, dtype=cp.float64).get())
    return {
        "sse": sse,
        "effective_rank": rank,
        "retained_condition_number": condition,
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "coefficients": coefficients.tolist(),
    }


def operator_basis(matrices: list[cp.ndarray]) -> tuple[
        dict[tuple[int, int], cp.ndarray], list[tuple[str, cp.ndarray]]]:
    grams: dict[tuple[int, int], cp.ndarray] = {}
    for a in range(3):
        for b in range(3):
            grams[(a, b)] = (matrices[a] @ matrices[b].T) / np.float32(COLS)
    cubics: list[tuple[str, cp.ndarray]] = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                cubics.append((f"cubic_{a}{b}{c}", grams[(a, b)] @ matrices[c]))
    return grams, cubics


def operator_features(matrices: list[cp.ndarray], target: int,
                      grams: dict[tuple[int, int], cp.ndarray],
                      cubics: list[tuple[str, cp.ndarray]]) -> tuple[list[str], cp.ndarray, list[float]]:
    features: list[tuple[str, cp.ndarray]] = [
        ("constant", cp.ones_like(matrices[target])),
        ("identity", matrices[target]),
        *cubics,
    ]
    if target == 0:
        left, right = 1, 2
    elif target == 1:
        left, right = 0, 2
    else:
        left, right = 0, 1
    safe5 = (((grams[(left, left)] * grams[(right, right)]) @ matrices[target]) /
             np.float32(ROWS))
    features.append(("safe_gram_hadamard_degree5", safe5))
    names, stack, scales = normalized_stack(features)
    del features, safe5
    return names, stack, scales


def bank_names(all_names: list[str], target: int) -> dict[str, list[str]]:
    cubic = [name for name in all_names if name.startswith("cubic_")]
    unary = f"cubic_{target}{target}{target}"
    safe = ["constant", "identity", unary, "safe_gram_hadamard_degree5"]
    if target == 1:
        safe.append("cubic_221")  # K_V U
    elif target == 2:
        safe.append("cubic_112")  # K_U V
    return {
        "scalar": ["constant", "identity"],
        "unary_cubic": ["constant", "identity", unary],
        "symmetry_aware_mixed": safe,
        "all_27_cubic": ["constant", "identity", *cubic],
        "all_27_cubic_plus_safe5": ["constant", "identity", *cubic,
                                             "safe_gram_hadamard_degree5"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    scratch = args.scratch.resolve()
    output = args.output.resolve()
    require(not output.exists(), "output must not exist")
    require(1 <= args.workers <= common.BLOCKS, "worker count")
    props = cp.cuda.runtime.getDeviceProperties(0)
    device = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    uuid_hex = bytes(props["uuid"][:16]).hex()
    require(device == EXPECTED_DEVICE and uuid_hex == EXPECTED_UUID_HEX, "pinned local RTX 3060")
    started = time.perf_counter()
    post_path = regenerate_post(release, scratch, args.workers)
    plan_path = release / "plan.lock.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    header = (release / "assets/header.bin").read_bytes()
    coefficients = struct.unpack_from("<12f", header, 32)
    post = np.memmap(post_path, dtype="<f8", mode="r",
                     shape=(common.GROUPS, common.GROUP_VALUES))

    matrix_rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, list[dict[str, Any]]]] = {
        role: {} for role in ROLES}
    total_baseline_sse = 0.0
    total_source_energy = 0.0
    for expert in range(common.EXPERTS):
        base = expert * common.GROUPS_PER_EXPERT
        gate = np.asarray(post[base:base + ROWS], dtype=np.float32)
        z0 = np.asarray(post[base + ROWS:base + 2 * ROWS], dtype=np.float32)
        z1 = np.asarray(post[base + 2 * ROWS:base + 3 * ROWS], dtype=np.float32)
        cosine = float(coefficients[2 * expert])
        sine = float(coefficients[2 * expert + 1])
        norm2 = cosine * cosine + sine * sine
        recon_np = [gate, (cosine * z0 - sine * z1) / norm2,
                     (sine * z0 + cosine * z1) / norm2]
        source_np: list[np.ndarray] = []
        for target, row in enumerate(plan["sources"][3 * expert:3 * expert + 3]):
            path = source_root / row["source_relpath"]
            require(path.is_file() and path.stat().st_size == int(row["bytes"]),
                    f"source presence {path}")
            require(sha256(path) == row["source_bf16_sha256"], f"source hash {path}")
            source_np.append(bf16(path, tuple(row["shape"]), target == 2))
        matrices = [cp.asarray(value, dtype=cp.float32) for value in recon_np]
        grams, cubics = operator_basis(matrices)
        for target, role in enumerate(ROLES):
            source = cp.asarray(source_np[target], dtype=cp.float32)
            residual = source - matrices[target]
            baseline_sse = float(cp.sum(residual.astype(cp.float64) ** 2,
                                        dtype=cp.float64).get())
            source_energy = float(cp.sum(source.astype(cp.float64) ** 2,
                                         dtype=cp.float64).get())
            total_baseline_sse += baseline_sse
            total_source_energy += source_energy
            names, stack, scales = operator_features(matrices, target, grams, cubics)
            gram, cross, replay_sse = feature_statistics(stack, residual)
            require(abs(replay_sse - baseline_sse) <= 5e-8, "residual SSE replay")
            banks = bank_names(names, target)
            fitted = {}
            for bank, selected in banks.items():
                index, selected_gram, selected_cross = select_stats(
                    names, gram, cross, selected)
                fit = fit_and_rescore(stack, residual, index, selected_gram, selected_cross)
                fit["feature_count"] = len(selected)
                fit["capture_fraction"] = 1.0 - fit["sse"] / baseline_sse
                fitted[bank] = fit
                stats[role].setdefault(bank, []).append({
                    "expert": expert,
                    "gram": selected_gram,
                    "cross": selected_cross,
                    "baseline_sse": baseline_sse,
                    "source_fitted_sse": fit["sse"],
                    "feature_count": len(selected),
                })
            matrix_rows.append({
                "expert_ordinal": expert,
                "layer": int(plan["sources"][3 * expert]["tensor"].split(".")[2]),
                "expert": int(plan["sources"][3 * expert]["tensor"].split(".")[5]),
                "role": role,
                "baseline_sse": baseline_sse,
                "source_energy": source_energy,
                "feature_normalizers": dict(zip(names, scales, strict=True)),
                "source_fitted": fitted,
            })
            del source, residual, stack
            cp.get_default_memory_pool().free_all_blocks()
        del matrices, grams, cubics
        cp.get_default_memory_pool().free_all_blocks()

    require(abs(total_baseline_sse - EXPECTED_BASELINE_SSE) <= 3e-5,
            f"baseline SSE identity: {total_baseline_sse}")
    require(abs(total_source_energy - EXPECTED_SOURCE_ENERGY) <= 3e-4,
            f"source energy identity: {total_source_energy}")

    aggregates = {}
    all_banks = tuple(stats[ROLES[0]])
    for bank in all_banks:
        source_sse = sum(row["source_fitted"][bank]["sse"] for row in matrix_rows)
        feature_count = int(stats[ROLES[0]][bank][0]["feature_count"])
        coefficient_count = common.EXPERTS * len(ROLES) * feature_count
        side_bpw = coefficient_count * 16.0 / common.WEIGHTS
        ratio = source_sse / total_baseline_sse
        aggregate = {
            "source_fitted_sse": source_sse,
            "source_fitted_capture_fraction": 1.0 - ratio,
            "source_fitted_coefficient_count": coefficient_count,
            "nominal_private_fp16_coefficient_bpw": side_bpw,
            "favourable_transfer_F": BASELINE_F * ratio * 2.0 ** (2.0 * side_bpw),
            "leave_one_expert_out_sse": 0.0,
        }
        for role in ROLES:
            rows = stats[role][bank]
            for heldout in range(common.EXPERTS):
                train = [row for row in rows if row["expert"] != heldout]
                test = rows[heldout]
                gram = sum((row["gram"] for row in train), np.zeros_like(train[0]["gram"]))
                cross = sum((row["cross"] for row in train), np.zeros_like(train[0]["cross"]))
                coeff, _, _ = solve_psd(gram, cross)
                test_sse = (float(test["baseline_sse"]) -
                            2.0 * float(coeff @ test["cross"]) +
                            float(coeff @ test["gram"] @ coeff))
                aggregate["leave_one_expert_out_sse"] += test_sse
        aggregate["leave_one_expert_out_capture_fraction"] = (
            1.0 - aggregate["leave_one_expert_out_sse"] / total_baseline_sse)
        aggregates[bank] = aggregate

    strongest = min(aggregates, key=lambda key: aggregates[key]["source_fitted_sse"])
    best = aggregates[strongest]
    if best["favourable_transfer_F"] <= TARGET_F:
        status = "SURVIVES_DIRECT_TARGET_APERTURE_REQUIRES_CONTROLS"
    elif best["source_fitted_capture_fraction"] >= 0.10:
        status = "SURVIVES_10_PERCENT_APERTURE_REQUIRES_CONTROLS"
    else:
        status = "HARD_KILL_COMPACT_MIXED_OPERATOR_SPAN_BELOW_10_PERCENT_CAPTURE"
    report: dict[str, Any] = {
        "schema": "operator-innovation-qwen-local3060-v0",
        "status": status,
        "claim_boundary": (
            "Actual audited STRATA residual; exact source-fitted scalar projection and "
            "leave-one-expert-out regression over a fixed compact operator bank. This is "
            "not a finite packet, not an exhaustive nonlinear converse, and has no "
            "matched-Gaussian authority unless the 10-percent aperture survives."),
        "baseline": {"physical_bpw": 2.5, "sse": total_baseline_sse,
                     "source_energy": total_source_energy,
                     "relative_mse": total_baseline_sse / total_source_energy,
                     "F": BASELINE_F, "target_F": TARGET_F},
        "gate": {"control_launch_capture_fraction": 0.10,
                 "direct_target_F": TARGET_F,
                 "strongest_bank": strongest},
        "aggregate": aggregates,
        "matrices": matrix_rows,
        "bindings": {"plan_sha256": sha256(plan_path),
                     "container_sha256": EXPECTED_CONTAINER_SHA,
                     "decoded_post_sha256": sha256(post_path),
                     "source_root": str(source_root)},
        "runtime": {"host": platform.node(), "python": list(sys.version_info[:3]),
                    "numpy": np.__version__, "cupy": cp.__version__,
                    "device": device, "device_uuid_hex": uuid_hex,
                    "elapsed_seconds": time.perf_counter() - started},
    }
    report["result_sha256_excluding_self"] = hashlib.sha256(canonical(report)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps({"status": status, "strongest_bank": strongest,
                      "best": best, "output": str(output),
                      "sha256": sha256(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
