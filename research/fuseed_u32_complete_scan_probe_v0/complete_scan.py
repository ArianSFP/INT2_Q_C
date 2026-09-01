#!/usr/bin/env python3
"""Source-free complete-u32 fused-screen traversal and exact Top-K probe."""

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


EXPECTED_KERNEL_SOURCE_SHA256 = (
    "001a3d08902441ee47501ff5a99bb0ce5159bff35b23b5cd11491539a564f401"
)
FULL_U32 = 1 << 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_kernel_source():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "fuseed_u32_throughput_probe_v0"
        / "benchmark.py"
    )
    actual = sha256_file(source_path)
    if actual != EXPECTED_KERNEL_SOURCE_SHA256:
        raise RuntimeError(f"kernel source hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("fuseed_kernel_source", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load kernel source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return source_path, module


def synchronize() -> None:
    cp.cuda.runtime.deviceSynchronize()


def exact_shard_topk(scores, shard_base: int, top_k: int):
    """Return exact score-descending/seed-ascending Top-K, including ties."""
    finite = bool(cp.asnumpy(cp.isfinite(scores).all()))
    if not finite:
        raise RuntimeError(f"nonfinite score in shard {shard_base}")
    provisional = cp.argpartition(scores, scores.size - top_k)[-top_k:]
    threshold = float(cp.asnumpy(cp.min(scores[provisional])))
    greater = cp.asnumpy(cp.nonzero(scores > threshold)[0]).astype(np.uint64)
    equal = cp.asnumpy(cp.nonzero(scores == threshold)[0]).astype(np.uint64)
    need = top_k - greater.size
    if need < 0 or equal.size < need:
        raise RuntimeError("Top-K threshold cardinality invariant failed")
    equal.sort()
    chosen = np.concatenate((greater, equal[:need]))
    chosen_values = cp.asnumpy(scores[cp.asarray(chosen)]).astype("<f4", copy=False)
    chosen_seeds = (np.uint64(shard_base) + chosen).astype("<u8", copy=False)
    order = np.lexsort((chosen_seeds, -chosen_values.astype(np.float64)))
    chosen_seeds = chosen_seeds[order]
    chosen_values = chosen_values[order]
    if chosen_seeds.size != top_k:
        raise RuntimeError("wrong shard Top-K cardinality")
    return chosen_seeds, chosen_values, threshold, int(equal.size)


def merge_topk(old_seeds, old_values, new_seeds, new_values, top_k: int):
    seeds = np.concatenate((old_seeds, new_seeds))
    values = np.concatenate((old_values, new_values))
    order = np.lexsort((seeds, -values.astype(np.float64)))[:top_k]
    return seeds[order].astype("<u8", copy=False), values[order].astype("<f4", copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-candidates", type=int, default=1 << 24)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.exists():
        raise RuntimeError("output and its parent must both be absent")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not (1 << 20) <= args.shard_candidates <= (1 << 27):
        raise RuntimeError("shard candidate count outside bound")
    if FULL_U32 % args.shard_candidates:
        raise RuntimeError("shard candidate count must divide 2^32")
    if not 1 <= args.top_k < args.shard_candidates:
        raise RuntimeError("invalid Top-K")
    if not 2 <= args.repetitions <= 3:
        raise RuntimeError("complete scan requires two or three repetitions")

    source_path, source = load_kernel_source()
    props = cp.cuda.runtime.getDeviceProperties(0)
    kernel = cp.RawKernel(source.CUDA_SOURCE, "fused_u32_scores", options=("--std=c++17",))
    kernel.compile()

    bundles = int(source.BUNDLES)
    values_per_candidate = int(source.VALUES_PER_CANDIDATE)
    ordinal = np.arange(bundles, dtype=np.uint64)
    seed_deltas = cp.asarray(1024 + 100 * (ordinal % 4), dtype=cp.uint64)
    sequences = cp.asarray((ordinal * 7919 + 17) % 261120, dtype=cp.uint64)
    offsets = cp.asarray(4 * ((ordinal * 13 + 5) % 97), dtype=cp.uint64)
    roles_host = (ordinal >= (bundles // 2)).astype(np.int32)
    roles = cp.asarray(roles_host)
    coefficient_index = np.arange(values_per_candidate, dtype=np.float64)
    coefficients_host = np.sin(0.17 + coefficient_index * 0.6180339887498949)
    for role in (0, 1):
        mask = np.repeat(roles_host == role, 4)
        coefficients_host[mask] /= np.linalg.norm(coefficients_host[mask])
    coefficients = cp.asarray(coefficients_host.astype(np.float32))
    scores = cp.empty(args.shard_candidates, dtype=cp.float32)

    block = 256
    grid = min(65535, (args.shard_candidates + block - 1) // block)
    shard_count = FULL_U32 // args.shard_candidates
    warm_count = min(args.shard_candidates, 1 << 18)
    kernel(
        (min(65535, (warm_count + block - 1) // block),),
        (block,),
        (
            np.uint64(0), np.uint64(warm_count), seed_deltas, sequences,
            offsets, roles, coefficients, np.int32(bundles), scores,
        ),
    )
    synchronize()

    rows = []
    for repetition in range(args.repetitions):
        global_seeds = np.empty(0, dtype="<u8")
        global_values = np.empty(0, dtype="<f4")
        shard_digest = hashlib.sha256()
        kernel_times = []
        select_times = []
        merge_times = []
        tie_sizes = []
        wall_start = time.perf_counter()
        for shard in range(shard_count):
            shard_base = shard * args.shard_candidates
            call_args = (
                np.uint64(shard_base), np.uint64(args.shard_candidates),
                seed_deltas, sequences, offsets, roles, coefficients,
                np.int32(bundles), scores,
            )
            synchronize()
            start = time.perf_counter()
            kernel((grid,), (block,), call_args)
            synchronize()
            kernel_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            local_seeds, local_values, threshold, tie_size = exact_shard_topk(
                scores, shard_base, args.top_k
            )
            synchronize()
            select_times.append(time.perf_counter() - start)
            tie_sizes.append(tie_size)
            shard_digest.update(np.asarray([shard_base], dtype="<u8").tobytes())
            shard_digest.update(local_seeds.tobytes())
            shard_digest.update(local_values.tobytes())
            shard_digest.update(np.asarray([threshold], dtype="<f4").tobytes())

            start = time.perf_counter()
            global_seeds, global_values = merge_topk(
                global_seeds, global_values, local_seeds, local_values, args.top_k
            )
            merge_times.append(time.perf_counter() - start)

        wall_seconds = time.perf_counter() - wall_start
        rows.append({
            "repetition": repetition,
            "full_domain_first_seed": 0,
            "full_domain_last_seed": FULL_U32 - 1,
            "full_domain_candidate_count": FULL_U32,
            "shard_count": shard_count,
            "kernel_seconds_sum": sum(kernel_times),
            "selection_seconds_sum": sum(select_times),
            "host_merge_seconds_sum": sum(merge_times),
            "wall_seconds": wall_seconds,
            "candidate_rate_per_second_wall": FULL_U32 / wall_seconds,
            "normal_value_rate_per_second_kernel": FULL_U32 * values_per_candidate / sum(kernel_times),
            "kernel_shard_seconds_min": min(kernel_times),
            "kernel_shard_seconds_median": statistics.median(kernel_times),
            "kernel_shard_seconds_max": max(kernel_times),
            "max_boundary_tie_cardinality": max(tie_sizes),
            "per_shard_topk_sha256": shard_digest.hexdigest(),
            "global_topk_seed_sha256": hashlib.sha256(global_seeds.tobytes()).hexdigest(),
            "global_topk_value_sha256": hashlib.sha256(global_values.tobytes()).hexdigest(),
            "best_seed": int(global_seeds[0]),
            "best_score": float(global_values[0]),
        })

    seed_hashes = {row["global_topk_seed_sha256"] for row in rows}
    value_hashes = {row["global_topk_value_sha256"] for row in rows}
    shard_hashes = {row["per_shard_topk_sha256"] for row in rows}
    if len(seed_hashes) != 1 or len(value_hashes) != 1 or len(shard_hashes) != 1:
        raise RuntimeError("complete replay is nondeterministic")

    result = {
        "schema": "fuseed_u32_source_free_complete_scan_probe_v0",
        "claim_boundary": (
            "Unsealed source-free engineering traversal only; synthetic two-role score, "
            "not exact FUSEED domains/ABI, Qwen access, retention, or authorization."
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "kernel_source": {
            "path": str(source_path),
            "sha256": EXPECTED_KERNEL_SOURCE_SHA256,
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
            "candidate_domain": "inclusive uint32 0..4294967295",
            "candidate_count": FULL_U32,
            "shard_candidates": args.shard_candidates,
            "shard_count": shard_count,
            "normal4_bundles_per_candidate": bundles,
            "normal_values_per_candidate": values_per_candidate,
            "top_k": args.top_k,
            "repetitions": args.repetitions,
            "score_buffer_bytes": int(scores.nbytes),
            "tie_order": "score descending, uint32 seed ascending",
        },
        "rows": rows,
        "aggregate": {
            "median_full_scan_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
            "median_kernel_seconds_sum": statistics.median(row["kernel_seconds_sum"] for row in rows),
            "median_selection_seconds_sum": statistics.median(row["selection_seconds_sum"] for row in rows),
            "median_host_merge_seconds_sum": statistics.median(row["host_merge_seconds_sum"] for row in rows),
            "complete_replay_deterministic": True,
            "projection_used": False,
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
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
