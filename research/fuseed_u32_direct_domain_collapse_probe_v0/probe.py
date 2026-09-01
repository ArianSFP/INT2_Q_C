#!/usr/bin/env python3
"""Source-free exact-FP64 active-domain timing probe for FUSEED-U32 v2."""

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
DOMAIN_COUNTS = (1, 2, 4, 8, 16, 33)
LAUNCHES = (
    ("block128", 4, 128, ("--std=c++17",)),
    ("block256", 8, 256, ("--std=c++17",)),
    ("block256_r96", 8, 256, ("--std=c++17", "--maxrregcount=96")),
    ("block256_r80", 8, 256, ("--std=c++17", "--maxrregcount=80")),
)


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


def derive_active_domain_source(source: str, active_domains: int) -> tuple[str, dict[str, int]]:
    if active_domains not in DOMAIN_COUNTS:
        raise RuntimeError("unfrozen active-domain count")
    counts: dict[str, int] = {}
    marker = "#define WARPS_PER_BLOCK 8\n"
    counts["active_define_insertion"] = source.count(marker)
    if counts["active_define_insertion"] != 1:
        raise RuntimeError("unexpected active-define marker cardinality")
    source = source.replace(
        marker, marker + f"#define ACTIVE_DOMAIN_COUNT {active_domains}\n", 1
    )

    old_cross = '''          sum_xw0[category] += x * (double)__ldg(
              &targets[domain0 * VALUE_COUNT + coordinate]);
          if (lane == 0) {
            sum_xw1[category] += x * (double)__ldg(
                &targets[domain1 * VALUE_COUNT + coordinate]);
          }'''
    new_cross = '''          if (domain0 < ACTIVE_DOMAIN_COUNT) {
            sum_xw0[category] += x * (double)__ldg(
                &targets[domain0 * VALUE_COUNT + coordinate]);
          }
          if (lane == 0 && ACTIVE_DOMAIN_COUNT > 1) {
            sum_xw1[category] += x * (double)__ldg(
                &targets[domain1 * VALUE_COUNT + coordinate]);
          }'''
    counts["cross_moment_guard"] = source.count(old_cross)
    if counts["cross_moment_guard"] != 1:
        raise RuntimeError("unexpected cross-moment block cardinality")
    source = source.replace(old_cross, new_cross, 1)

    old_output = '''    q[(unsigned long long)domain0 * candidate_count + candidate] =
        domain_q(domain0, local_sum_x, local_sum_x2, sum_xw0, target_stats);
    if (lane == 0) {
      q[(unsigned long long)domain1 * candidate_count + candidate] =
          domain_q(domain1, local_sum_x, local_sum_x2, sum_xw1, target_stats);
    }'''
    new_output = '''    if (domain0 < ACTIVE_DOMAIN_COUNT) {
      q[(unsigned long long)domain0 * candidate_count + candidate] =
          domain_q(domain0, local_sum_x, local_sum_x2, sum_xw0, target_stats);
    }
    if (lane == 0 && ACTIVE_DOMAIN_COUNT > 1) {
      q[(unsigned long long)domain1 * candidate_count + candidate] =
          domain_q(domain1, local_sum_x, local_sum_x2, sum_xw1, target_stats);
    }'''
    counts["output_guard"] = source.count(old_output)
    if counts["output_guard"] != 1:
        raise RuntimeError("unexpected output block cardinality")
    source = source.replace(old_output, new_output, 1)
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
        raise RuntimeError("candidate count outside probe bound")
    if args.repetitions != 5:
        raise RuntimeError("repetitions must be exactly five")

    direct_path, direct = load_direct_module()
    shape_path, shape = direct.load_shape_module()
    header_hashes = direct.bind_headers()
    direct_source, direct_counts = direct.derive_direct_source(shape.CUDA_SOURCE)
    parity = direct.run_parity()
    targets_host, stats_host = shape.make_targets()
    targets = cp.asarray(targets_host)
    target_stats = cp.asarray(stats_host)
    plan_host = direct.make_bundle_plan(shape)
    addends, sequences, offset_quads, normal4_indices = [cp.asarray(value) for value in plan_host]
    q = cp.empty((shape.DOMAINS, args.candidates), dtype=cp.float32)
    props = cp.cuda.runtime.getDeviceProperties(0)

    rows = []
    domain_hash_observations: dict[int, set[str]] = {
        domain: set() for domain in range(shape.DOMAINS)
    }
    sample_indices = cp.linspace(0, args.candidates - 1, 32768, dtype=cp.int64)
    for active_domains in DOMAIN_COUNTS:
        active_source, guard_counts = derive_active_domain_source(
            direct_source, active_domains
        )
        for launch_name, warps, block, options in LAUNCHES:
            marker = "#define WARPS_PER_BLOCK 8\n"
            if active_source.count(marker) != 1:
                raise RuntimeError("warp marker cardinality changed")
            configured_source = active_source.replace(
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
            q.fill(cp.nan)
            synchronize()
            times = []
            for _ in range(args.repetitions):
                synchronize()
                start = time.perf_counter()
                kernel((grid,), (block,), call_args)
                synchronize()
                times.append(time.perf_counter() - start)
            if not bool(cp.asnumpy(cp.isfinite(q[:active_domains]).all())):
                raise RuntimeError(f"{active_domains}/{launch_name} produced nonfinite active q")
            if active_domains < shape.DOMAINS:
                if not bool(cp.asnumpy(cp.isnan(q[active_domains:]).all())):
                    raise RuntimeError(f"{active_domains}/{launch_name} wrote an inactive q row")
            domain_hashes = []
            for domain in range(active_domains):
                sample = cp.asnumpy(q[domain, sample_indices]).astype("<f4", copy=False)
                digest = hashlib.sha256(sample.tobytes()).hexdigest()
                domain_hashes.append(digest)
                domain_hash_observations[domain].add(digest)
            median_seconds = statistics.median(times)
            rows.append({
                "active_domains": active_domains,
                "launch_name": launch_name,
                "warps_per_block": warps,
                "block_threads": block,
                "options": list(options),
                "grid_blocks": grid,
                "times_seconds": times,
                "median_seconds": median_seconds,
                "projected_three_abi_full_u32_kernel_seconds": (
                    median_seconds * (FULL_U32 / args.candidates) * ABIS
                ),
                "attributes": {key: int(value) for key, value in kernel.attributes.items()},
                "guard_replacement_counts": guard_counts,
                "configured_cuda_sha256": hashlib.sha256(configured_source.encode()).hexdigest(),
                "active_domain_q_sentinel_sha256": domain_hashes,
            })

    for domain, observations in domain_hash_observations.items():
        if len(observations) != 1:
            raise RuntimeError(f"domain {domain} q bytes changed across containing variants")
    winners = {}
    for active_domains in DOMAIN_COUNTS:
        eligible = [row for row in rows if row["active_domains"] == active_domains]
        winners[str(active_domains)] = min(
            eligible, key=lambda row: (row["median_seconds"], row["launch_name"])
        )
    source_winner = winners["1"]
    result = {
        "schema": "fuseed_u32_source_free_direct_domain_collapse_probe_v0",
        "claim_boundary": (
            "Source-free exact-FP64 timing probe only. Collapsing descriptive controls "
            "is a distinct v2 architecture and is not a retention, Qwen, significance, "
            "or compression result."
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "direct_script": {"path": str(direct_path), "sha256": EXPECTED_DIRECT_SCRIPT_SHA256},
        "shape_source": {"path": str(shape_path), "sha256": direct.EXPECTED_SHAPE_SOURCE_SHA256},
        "cuda_headers": header_hashes,
        "derivation": {
            "direct_replacement_counts": direct_counts,
            "active_domain_counts": list(DOMAIN_COUNTS),
            "cross_moments_affine_half_reload_and_q_remain_fp64": True,
            "generator_counter_box_muller_scale_and_bf16_path_unchanged": True,
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
            "candidates": args.candidates, "frozen_full_domains": shape.DOMAINS,
            "bundles_per_candidate": shape.BUNDLES,
            "values_per_candidate": shape.VALUES,
            "allocated_q_bytes": int(q.nbytes), "repetitions": args.repetitions,
        },
        "rows": rows,
        "winners_by_active_domain_count": winners,
        "decision": {
            "prospective_source_only_kernel_margin_gate_seconds": 800.0,
            "source_only_winner": source_winner,
            "source_only_kernel_projection_below_margin_gate": (
                source_winner["projected_three_abi_full_u32_kernel_seconds"] < 800.0
            ),
            "promotion_if_pass": (
                "distinct v2 full-shard exact source-only screen calibration including "
                "Top-K, journal, compiler/runtime bindings, and an independent audit"
            ),
            "promotion_if_fail": "kill exhaustive u32 FUSEED family before Qwen access",
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
        "winners": {
            key: {
                "launch": row["launch_name"],
                "median_seconds": row["median_seconds"],
                "projection_seconds": row["projected_three_abi_full_u32_kernel_seconds"],
                "regs": row["attributes"]["num_regs"],
                "local_bytes": row["attributes"]["local_size_bytes"],
            }
            for key, row in winners.items()
        },
        "decision": result["decision"]["source_only_kernel_projection_below_margin_gate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
