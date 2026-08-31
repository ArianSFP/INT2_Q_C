#!/usr/bin/env python3
"""CuPy distribution audit for the exact-v2 Qwen failure and RHT repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cupy as cp
import numpy as np


N = 1 << 18


def splitmix64_signs(n: int, seed: int) -> cp.ndarray:
    with np.errstate(over="ignore"):
        values = np.arange(n, dtype=np.uint64) + np.uint64(seed)
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    return cp.asarray(np.where((values & np.uint64(1)) == 0, 1.0, -1.0))


def fht(values: cp.ndarray) -> cp.ndarray:
    out = values.astype(cp.float64, copy=True)
    width = 1
    while width < out.size:
        view = out.reshape(-1, 2, width)
        left = view[:, 0, :].copy()
        right = view[:, 1, :].copy()
        view[:, 0, :] = left + right
        view[:, 1, :] = left - right
        width *= 2
    return out / math.sqrt(out.size)


def load_bf16(path: Path, local_index: int) -> tuple[cp.ndarray, str]:
    raw = np.memmap(path, dtype="<u2", mode="r")
    block = np.asarray(raw[local_index * N:(local_index + 1) * N]).copy()
    if block.size != N:
        raise ValueError(path)
    digest = hashlib.sha256(block.tobytes()).hexdigest()
    values = (block.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return cp.asarray(values, dtype=cp.float64), digest


def stats(values: cp.ndarray) -> dict[str, float]:
    mean = cp.mean(values)
    centered = values - mean
    variance = cp.mean(centered * centered)
    rms = cp.sqrt(cp.mean(values * values))
    squares = values * values
    tail_count = max(1, values.size // 1000)
    tail_energy = cp.partition(squares, values.size - tail_count)[-tail_count:].sum()
    total_energy = squares.sum()
    return {
        "mean_over_rms": float((mean / rms).get()),
        "pearson_kurtosis": float((cp.mean(centered ** 4) / (variance * variance)).get()),
        "absolute_skewness": float(cp.abs(cp.mean(centered ** 3) / (variance ** 1.5)).get()),
        "zero_fraction": float(cp.mean(values == 0).get()),
        "max_abs_over_rms": float((cp.max(cp.abs(values)) / rms).get()),
        "top_0p1pct_energy_fraction": float((tail_energy / total_energy).get()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exact-summary", type=Path, required=True)
    parser.add_argument("--rht-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    exact = json.loads(args.exact_summary.read_text(encoding="utf-8"))["blocks_detail"]
    rht = json.loads(args.rht_summary.read_text(encoding="utf-8"))["blocks_detail"]
    rows = []
    for index, block in enumerate(manifest["blocks"]):
        path = Path(block["source_path"])
        if not path.is_absolute():
            path = args.workspace / path
        source, digest = load_bf16(path, int(block["source_local_block_index"]))
        if digest != block["source_bf16_sha256"]:
            raise AssertionError(block["id"])
        transformed = fht(source * splitmix64_signs(N, int(block["rht_seed_u64"])))
        raw_stats = stats(source)
        rht_stats = stats(transformed)
        rows.append({
            "id": block["id"],
            "role": block["role"],
            "exact_relative_mse": float(exact[index]["fp16_relative_mse"]),
            "rht_relative_mse": float(rht[index]["fp16_relative_mse"]),
            "raw": raw_stats,
            "rht": rht_stats,
        })
        print(f"{index + 1}/{len(manifest['blocks'])} {block['id']}", flush=True)
    raw_kurtosis = np.asarray([row["raw"]["pearson_kurtosis"] for row in rows])
    rht_kurtosis = np.asarray([row["rht"]["pearson_kurtosis"] for row in rows])
    exact_mse = np.asarray([row["exact_relative_mse"] for row in rows])
    rht_mse = np.asarray([row["rht_relative_mse"] for row in rows])
    result = {
        "status": "complete",
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cupy_version": cp.__version__,
        "blocks": len(rows),
        "summary": {
            "raw_kurtosis_median": float(np.median(raw_kurtosis)),
            "raw_kurtosis_max": float(np.max(raw_kurtosis)),
            "rht_kurtosis_median": float(np.median(rht_kurtosis)),
            "rht_kurtosis_max": float(np.max(rht_kurtosis)),
            "raw_kurtosis_distance_to_gaussian_median": float(np.median(np.abs(raw_kurtosis - 3.0))),
            "rht_kurtosis_distance_to_gaussian_median": float(np.median(np.abs(rht_kurtosis - 3.0))),
            "correlation_exact_mse_vs_raw_kurtosis": float(np.corrcoef(exact_mse, raw_kurtosis)[0, 1]),
            "correlation_rht_mse_vs_rht_kurtosis": float(np.corrcoef(rht_mse, rht_kurtosis)[0, 1]),
        },
        "worst_exact_blocks": sorted(rows, key=lambda row: row["exact_relative_mse"], reverse=True)[:8],
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
