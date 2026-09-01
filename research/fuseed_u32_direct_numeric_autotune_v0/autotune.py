#!/usr/bin/env python3
"""Source-free launch-shape autotune for direct-Philox FP32 FUSEED screen."""

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


EXPECTED_DIRECT_SCRIPT_SHA256 = (
    "f5a7c8b9a525e02d469ca974f9a6607030b2ca2822b66d4bce31604251516ed5"
)
FULL_U32 = 1 << 32
ABIS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_direct_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "fuseed_u32_direct_counter_calibration_v0"
        / "calibrate_direct.py"
    )
    actual = sha256_file(path)
    if actual != EXPECTED_DIRECT_SCRIPT_SHA256:
        raise RuntimeError(f"direct calibration script hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("fuseed_direct_calibration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load direct calibration module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def derive_fp32_cross_source(source: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    marker = "#define WARPS_PER_BLOCK 8\n"
    counts["type_insertion"] = source.count(marker)
    if counts["type_insertion"] != 1:
        raise RuntimeError("unexpected type insertion cardinality")
    source = source.replace(marker, marker + "typedef float xw_t;\n", 1)
    replacements = (
        ("double sum_xw[4]", "xw_t sum_xw[4]", "parameters", 2),
        ("double sum_xw0[4]", "xw_t sum_xw0[4]", "accumulator0", 1),
        ("double sum_xw1[4]", "xw_t sum_xw1[4]", "accumulator1", 1),
        ("sum_xw0[category] += x * (double)__ldg(", "sum_xw0[category] += (float)x * __ldg(", "fma0", 1),
        ("sum_xw1[category] += x * (double)__ldg(", "sum_xw1[category] += (float)x * __ldg(", "fma1", 1),
    )
    for old, new, label, expected in replacements:
        counts[label] = source.count(old)
        if counts[label] != expected:
            raise RuntimeError(f"unexpected {label} cardinality")
        source = source.replace(old, new)
    return source, counts


def synchronize() -> None:
    cp.cuda.runtime.deviceSynchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=1 << 20)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.exists():
        raise RuntimeError("output and its parent must both be absent")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not (1 << 18) <= args.candidates <= (1 << 22):
        raise RuntimeError("candidate count outside autotune bound")
    if args.repetitions != 5:
        raise RuntimeError("repetitions must be exactly five")

    direct_path, direct = load_direct_module()
    shape_path, shape = direct.load_shape_module()
    header_hashes = direct.bind_headers()
    direct_source, direct_counts = direct.derive_direct_source(shape.CUDA_SOURCE)
    fp32_source, fp32_counts = derive_fp32_cross_source(direct_source)
    parity = direct.run_parity()
    targets_host, stats_host = shape.make_targets()
    targets = cp.asarray(targets_host)
    target_stats = cp.asarray(stats_host)
    plan_host = direct.make_bundle_plan(shape)
    addends, sequences, offset_quads, normal4_indices = [cp.asarray(value) for value in plan_host]
    q = cp.empty((shape.DOMAINS, args.candidates), dtype=cp.float32)
    props = cp.cuda.runtime.getDeviceProperties(0)

    configs = (
        ("warp4_block128", 4, 128, ("--std=c++17",)),
        ("warp8_block256", 8, 256, ("--std=c++17",)),
        ("warp16_block512", 16, 512, ("--std=c++17",)),
        ("warp4_block128_r80", 4, 128, ("--std=c++17", "--maxrregcount=80")),
        ("warp8_block256_r80", 8, 256, ("--std=c++17", "--maxrregcount=80")),
    )
    rows = []
    sentinel_hashes = set()
    for name, warps, block, options in configs:
        marker = "#define WARPS_PER_BLOCK 8\n"
        if fp32_source.count(marker) != 1:
            raise RuntimeError("warp marker cardinality changed")
        configured_source = fp32_source.replace(
            marker, f"#define WARPS_PER_BLOCK {warps}\n", 1
        )
        kernel = cp.RawKernel(
            configured_source, "fuseed_33domain_scores", options=options
        )
        kernel.compile()
        grid = min(65535, (args.candidates + warps - 1) // warps)
        call_args = (
            np.uint64(0), np.uint64(args.candidates), addends, sequences,
            offset_quads, normal4_indices, targets, target_stats, q,
        )
        warm_count = min(args.candidates, 1 << 14)
        warm_grid = min(grid, (warm_count + warps - 1) // warps)
        warm_args = (
            np.uint64(0), np.uint64(warm_count), addends, sequences,
            offset_quads, normal4_indices, targets, target_stats, q,
        )
        kernel((warm_grid,), (block,), warm_args)
        synchronize()
        times = []
        for _ in range(args.repetitions):
            synchronize()
            start = time.perf_counter()
            kernel((grid,), (block,), call_args)
            synchronize()
            times.append(time.perf_counter() - start)
        if not bool(cp.asnumpy(cp.isfinite(q).all())):
            raise RuntimeError(f"{name} produced nonfinite q")
        sample_indices = cp.linspace(0, q.size - 1, 32768, dtype=cp.int64)
        sample = cp.asnumpy(q.reshape(-1)[sample_indices]).astype("<f4", copy=False)
        sentinel = hashlib.sha256(sample.tobytes()).hexdigest()
        sentinel_hashes.add(sentinel)
        median_seconds = statistics.median(times)
        rows.append({
            "name": name,
            "warps_per_block": warps,
            "block_threads": block,
            "options": list(options),
            "grid_blocks": grid,
            "times_seconds": times,
            "median_seconds": median_seconds,
            "projected_three_abi_full_u32_seconds": (
                median_seconds * (FULL_U32 / args.candidates) * ABIS
            ),
            "attributes": {key: int(value) for key, value in kernel.attributes.items()},
            "configured_cuda_sha256": hashlib.sha256(configured_source.encode()).hexdigest(),
            "q_sentinel_sha256": sentinel,
        })

    if len(sentinel_hashes) != 1:
        raise RuntimeError("launch variants changed q bytes")
    winner = min(rows, key=lambda row: (row["median_seconds"], row["name"]))
    result = {
        "schema": "fuseed_u32_source_free_direct_numeric_autotune_v0",
        "claim_boundary": (
            "Unsealed source-free launch autotune only; FP32 cross-moment screen lacks "
            "numerical/retention authorization and no model/Qwen data is accessed."
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "direct_script": {"path": str(direct_path), "sha256": EXPECTED_DIRECT_SCRIPT_SHA256},
        "shape_source": {"path": str(shape_path), "sha256": direct.EXPECTED_SHAPE_SOURCE_SHA256},
        "cuda_headers": header_hashes,
        "derivation": {
            "direct_replacement_counts": direct_counts,
            "fp32_replacement_counts": fp32_counts,
            "base_fp32_cuda_sha256": hashlib.sha256(fp32_source.encode()).hexdigest(),
            "per_domain_xw_only_is_fp32": True,
            "common_moments_affine_half_reload_and_q_remain_fp64": True,
        },
        "parity": parity,
        "runtime": {
            "python": os.sys.version.split()[0], "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "device": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "shape": {
            "candidates": args.candidates, "domains": shape.DOMAINS,
            "bundles_per_candidate": shape.BUNDLES,
            "values_per_candidate": shape.VALUES,
            "q_bytes": int(q.nbytes), "repetitions": args.repetitions,
        },
        "rows": rows,
        "winner": winner,
        "decision": {
            "promotion_margin_gate_seconds": 800.0,
            "winner_projection_below_margin_gate": (
                winner["projected_three_abi_full_u32_seconds"] < 800.0
            ),
            "next_step_if_pass": (
                "distinct direct-counter FP32 shortlist plus exact FP64 refinement calibration"
            ),
            "next_step_if_fail": "do not authorize payload; redesign screen statistic",
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
    print(json.dumps({"winner": winner, "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
