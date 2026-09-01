#!/usr/bin/env python3
"""Source-free FP32-screen/FP64-refine feasibility probe for FUSEED-U32."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import cupy as cp
import numpy as np


EXPECTED_EXACT_SOURCE_SHA256 = (
    "6d3cdac6ab4f1a1fcbe742f43a2fd817c8bf56d1bfeff9e38c2432fe6848149a"
)
FULL_U32 = 1 << 32
ABIS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_exact_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "fuseed_u32_33domain_calibration_v0"
        / "calibrate.py"
    )
    actual = sha256_file(path)
    if actual != EXPECTED_EXACT_SOURCE_SHA256:
        raise RuntimeError(f"exact calibration source hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("fuseed_exact_calibration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load exact calibration source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def derive_fast_source(exact_source: str) -> tuple[str, dict[str, int]]:
    source = exact_source
    counts: dict[str, int] = {}

    marker = "#define WARPS_PER_BLOCK 8\n"
    counts["type_insertion"] = source.count(marker)
    if counts["type_insertion"] != 1:
        raise RuntimeError("unexpected fast-source type insertion cardinality")
    source = source.replace(marker, marker + "typedef float xw_t;\n", 1)

    replacements = (
        ("double sum_xw[4]", "xw_t sum_xw[4]", "function_parameters", 2),
        ("double sum_xw0[4]", "xw_t sum_xw0[4]", "domain0_accumulator", 1),
        ("double sum_xw1[4]", "xw_t sum_xw1[4]", "domain1_accumulator", 1),
        (
            "sum_xw0[category] += x * (double)__ldg(",
            "sum_xw0[category] += (float)x * __ldg(",
            "domain0_fma",
            1,
        ),
        (
            "sum_xw1[category] += x * (double)__ldg(",
            "sum_xw1[category] += (float)x * __ldg(",
            "domain1_fma",
            1,
        ),
    )
    for old, new, label, expected in replacements:
        counts[label] = source.count(old)
        if counts[label] != expected:
            raise RuntimeError(f"unexpected {label} replacement cardinality")
        source = source.replace(old, new)
    if "double sum_xw[4]" in source or "double sum_xw0[4]" in source:
        raise RuntimeError("unconverted FP64 cross accumulator")
    return source, counts


def synchronize() -> None:
    cp.cuda.runtime.deviceSynchronize()


def timed_launch(kernel, grid, block, call_args) -> float:
    synchronize()
    start = time.perf_counter()
    kernel((grid,), (block,), call_args)
    synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=1 << 22)
    parser.add_argument("--exact-top-k", type=int, default=8192)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.exists():
        raise RuntimeError("output and its parent must both be absent")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not (1 << 18) <= args.candidates <= (1 << 23):
        raise RuntimeError("candidate count outside probe bound")
    if args.exact_top_k != 8192:
        raise RuntimeError("exact Top-K is frozen at 8192")
    if not 2 <= args.repetitions <= 5:
        raise RuntimeError("invalid repetition count")

    exact_path, exact = load_exact_module()
    fast_source, replacement_counts = derive_fast_source(exact.CUDA_SOURCE)
    props = cp.cuda.runtime.getDeviceProperties(0)
    targets_host, stats_host = exact.make_targets()
    targets = cp.asarray(targets_host)
    target_stats = cp.asarray(stats_host)
    ordinal = np.arange(exact.BUNDLES, dtype=np.uint64)
    seed_deltas = cp.asarray(1024 + 100 * (ordinal % 4), dtype=cp.uint64)
    sequences = cp.asarray((ordinal * 7919 + 17) % 1048576, dtype=cp.uint64)
    offsets = cp.asarray(4 * ((ordinal * 13 + 5) % 257), dtype=cp.uint64)
    q_exact = cp.empty((exact.DOMAINS, args.candidates), dtype=cp.float32)
    q_fast = cp.empty_like(q_exact)

    exact_kernel = cp.RawKernel(
        exact.CUDA_SOURCE, "fuseed_33domain_scores", options=("--std=c++17",)
    )
    fast_kernel = cp.RawKernel(
        fast_source, "fuseed_33domain_scores", options=("--std=c++17",)
    )
    exact_kernel.compile()
    fast_kernel.compile()
    block = 256
    warps_per_block = 8
    grid = min(65535, (args.candidates + warps_per_block - 1) // warps_per_block)

    common_args = (
        np.uint64(0), np.uint64(args.candidates), seed_deltas, sequences,
        offsets, targets, target_stats,
    )
    exact_args = common_args + (q_exact,)
    fast_args = common_args + (q_fast,)
    warm_count = min(args.candidates, 1 << 14)
    warm_common = (
        np.uint64(0), np.uint64(warm_count), seed_deltas, sequences,
        offsets, targets, target_stats,
    )
    warm_grid = min(grid, (warm_count + warps_per_block - 1) // warps_per_block)
    exact_kernel((warm_grid,), (block,), warm_common + (q_exact,))
    fast_kernel((warm_grid,), (block,), warm_common + (q_fast,))
    synchronize()

    exact_times = []
    fast_times = []
    for _ in range(args.repetitions):
        exact_times.append(timed_launch(exact_kernel, grid, block, exact_args))
        fast_times.append(timed_launch(fast_kernel, grid, block, fast_args))

    if not bool(cp.asnumpy(cp.isfinite(q_exact).all())):
        raise RuntimeError("exact q contains nonfinite values")
    if not bool(cp.asnumpy(cp.isfinite(q_fast).all())):
        raise RuntimeError("fast q contains nonfinite values")

    shortlist_grid = [
        value for value in (8192, 16384, 32768, 65536, 131072, 262144)
        if value < args.candidates
    ]
    domain_rows = []
    exact_seed_rows = []
    exact_value_rows = []
    fast_seed_rows = {width: [] for width in shortlist_grid}
    fast_value_rows = {width: [] for width in shortlist_grid}
    for domain_index in range(exact.DOMAINS):
        exact_seeds, exact_values, _, _ = exact.exact_topk(
            q_exact[domain_index], 0, args.exact_top_k
        )
        exact_seed_rows.append(exact_seeds)
        exact_value_rows.append(exact_values)
        diff = q_fast[domain_index] - q_exact[domain_index]
        max_abs_error = float(cp.asnumpy(cp.max(cp.abs(diff))))
        rms_error = float(cp.asnumpy(cp.sqrt(cp.mean(diff * diff, dtype=cp.float64))))
        retention = {}
        for width in shortlist_grid:
            fast_seeds, fast_values, _, _ = exact.exact_topk(
                q_fast[domain_index], 0, width
            )
            fast_seed_rows[width].append(fast_seeds)
            fast_value_rows[width].append(fast_values)
            kept = np.intersect1d(exact_seeds, fast_seeds, assume_unique=True).size
            retention[str(width)] = {
                "exact_topk_retained": int(kept),
                "exact_topk_total": args.exact_top_k,
                "retention": kept / args.exact_top_k,
            }
        domain_rows.append({
            "domain": domain_index,
            "max_abs_q_error": max_abs_error,
            "rms_q_error": rms_error,
            "shortlist_retention": retention,
        })

    exact_seed_array = np.stack(exact_seed_rows).astype("<u8", copy=False)
    exact_value_array = np.stack(exact_value_rows).astype("<f4", copy=False)
    shortlist_aggregate = {}
    for width in shortlist_grid:
        seed_array = np.stack(fast_seed_rows[width]).astype("<u8", copy=False)
        value_array = np.stack(fast_value_rows[width]).astype("<f4", copy=False)
        retained = [
            row["shortlist_retention"][str(width)]["exact_topk_retained"]
            for row in domain_rows
        ]
        shortlist_aggregate[str(width)] = {
            "minimum_domain_retention": min(retained) / args.exact_top_k,
            "mean_domain_retention": statistics.mean(retained) / args.exact_top_k,
            "all_domain_exact_topk_retained": all(value == args.exact_top_k for value in retained),
            "seed_sha256": hashlib.sha256(seed_array.tobytes()).hexdigest(),
            "value_sha256": hashlib.sha256(value_array.tobytes()).hexdigest(),
            "maximum_union_candidates_per_abi": exact.DOMAINS * width,
        }

    exact_median = statistics.median(exact_times)
    fast_median = statistics.median(fast_times)
    scale = FULL_U32 / args.candidates * ABIS
    result = {
        "schema": "fuseed_u32_source_free_numeric_cascade_probe_v0",
        "claim_boundary": (
            "Unsealed synthetic numerical-screen probe only; empirical shortlist retention "
            "is not an interval proof, modeled-retention certificate, Qwen result, or authorization."
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "exact_source": {"path": str(exact_path), "sha256": EXPECTED_EXACT_SOURCE_SHA256},
        "fast_source_derivation": {
            "operation": "replace only per-domain x*w accumulators/FMAs with FP32; retain common moments, affine solve, FP16 round-trip, and q evaluation in FP64",
            "replacement_counts": replacement_counts,
            "derived_cuda_sha256": hashlib.sha256(fast_source.encode()).hexdigest(),
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "device": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "shape": {
            "candidates": args.candidates,
            "domains": exact.DOMAINS,
            "anchors_per_candidate": exact.VALUES,
            "exact_top_k": args.exact_top_k,
            "shortlist_grid": shortlist_grid,
            "repetitions": args.repetitions,
            "q_bytes_each": int(q_exact.nbytes),
        },
        "kernels": {
            "exact_attributes": {key: int(value) for key, value in exact_kernel.attributes.items()},
            "fast_attributes": {key: int(value) for key, value in fast_kernel.attributes.items()},
            "exact_seconds": exact_times,
            "fast_seconds": fast_times,
            "median_exact_seconds": exact_median,
            "median_fast_seconds": fast_median,
            "speedup": exact_median / fast_median,
            "projected_three_abi_exact_seconds": exact_median * scale,
            "projected_three_abi_fast_screen_seconds": fast_median * scale,
        },
        "numeric_error": {
            "maximum_domain_max_abs_q_error": max(row["max_abs_q_error"] for row in domain_rows),
            "maximum_domain_rms_q_error": max(row["rms_q_error"] for row in domain_rows),
            "per_domain": domain_rows,
        },
        "retention": {
            "exact_topk_seed_sha256": hashlib.sha256(exact_seed_array.tobytes()).hexdigest(),
            "exact_topk_value_sha256": hashlib.sha256(exact_value_array.tobytes()).hexdigest(),
            "shortlists": shortlist_aggregate,
        },
        "decision_rule": {
            "promote_only_if": (
                "projected FP32 full screen is below 900 s with margin and a distinct v2 "
                "establishes conservative numerical plus modeled retention before payload access"
            ),
            "no_qwen_access_authorized": True,
        },
        "access": {
            "model_or_qwen_path_arguments": 0,
            "payload_files_opened": 0,
            "network_operations": 0,
        },
    }
    args.output.parent.mkdir(parents=False, exist_ok=False)
    with args.output.open("xb") as handle:
        handle.write((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({
        "speedup": result["kernels"]["speedup"],
        "projected_three_abi_fast_screen_seconds": result["kernels"]["projected_three_abi_fast_screen_seconds"],
        "maximum_domain_max_abs_q_error": result["numeric_error"]["maximum_domain_max_abs_q_error"],
        "shortlists": result["retention"]["shortlists"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
