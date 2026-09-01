#!/usr/bin/env python3
"""Source-free CuPy throughput probe for fused complete-u32 seed screening.

This deliberately has no model/source argument. It measures only the shape of
an 80-normal4-bundle fingerprint kernel plus deterministic shard-local Top-K.
It is an unsealed engineering calibration, not a scientific retention result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import cupy as cp
import numpy as np


BUNDLES = 80
VALUES_PER_CANDIDATE = 4 * BUNDLES
FULL_U32 = 1 << 32


CUDA_SOURCE = r'''
#include <curand_kernel.h>

extern "C" __global__ void fused_u32_scores(
    unsigned long long shard_base,
    unsigned long long candidate_count,
    const unsigned long long* seed_deltas,
    const unsigned long long* sequences,
    const unsigned long long* offsets,
    const int* roles,
    const float* coefficients,
    int bundle_count,
    float* scores) {
  unsigned long long start =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long step =
      (unsigned long long)blockDim.x * gridDim.x;
  for (unsigned long long candidate = start; candidate < candidate_count;
       candidate += step) {
    unsigned long long base_seed = shard_base + candidate;
    float z0 = 0.0f;
    float z1 = 0.0f;
    for (int bundle = 0; bundle < bundle_count; ++bundle) {
      curandStatePhilox4_32_10_t state;
      curand_init(base_seed + seed_deltas[bundle], sequences[bundle],
                  offsets[bundle], &state);
      float4 value = curand_normal4(&state);
      int k = 4 * bundle;
      float dot = coefficients[k + 0] * value.x +
                  coefficients[k + 1] * value.y +
                  coefficients[k + 2] * value.z +
                  coefficients[k + 3] * value.w;
      if (roles[bundle] == 0) z0 += dot;
      else z1 += dot;
    }
    scores[candidate] = z0 * z0 + z1 * z1;
  }
}
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize() -> None:
    cp.cuda.runtime.deviceSynchronize()


def timed(callable_):
    synchronize()
    start = time.perf_counter()
    value = callable_()
    synchronize()
    return value, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=1 << 24)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.exists():
        raise RuntimeError("output and its parent must both be absent")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not (1 << 18) <= args.candidates <= (1 << 28):
        raise RuntimeError("candidate calibration count outside bound")
    if not 1 <= args.repetitions <= 7 or not 1 <= args.top_k < args.candidates:
        raise RuntimeError("invalid repetition/top-k configuration")

    props = cp.cuda.runtime.getDeviceProperties(0)
    kernel = cp.RawKernel(CUDA_SOURCE, "fused_u32_scores", options=("--std=c++17",))
    kernel.compile()

    # Fixed synthetic cells resembling four EP-rank seed deltas and ordinary
    # normal4 bundles. They are source-free and carry no retention meaning.
    ordinal = np.arange(BUNDLES, dtype=np.uint64)
    seed_deltas = cp.asarray(1024 + 100 * (ordinal % 4), dtype=cp.uint64)
    sequences = cp.asarray((ordinal * 7919 + 17) % 261120, dtype=cp.uint64)
    offsets = cp.asarray(4 * ((ordinal * 13 + 5) % 97), dtype=cp.uint64)
    roles = cp.asarray((ordinal >= (BUNDLES // 2)).astype(np.int32))
    coefficient_index = np.arange(VALUES_PER_CANDIDATE, dtype=np.float64)
    coefficients_np = np.sin(0.17 + coefficient_index * 0.6180339887498949)
    for role in (0, 1):
        mask = np.repeat(np.asarray(roles.get()) == role, 4)
        coefficients_np[mask] /= np.linalg.norm(coefficients_np[mask])
    coefficients = cp.asarray(coefficients_np.astype(np.float32))
    scores = cp.empty(args.candidates, dtype=cp.float32)

    block = 256
    grid = min(65535, (args.candidates + block - 1) // block)
    call_args = (
        np.uint64(0), np.uint64(args.candidates), seed_deltas, sequences,
        offsets, roles, coefficients, np.int32(BUNDLES), scores,
    )

    warm_count = min(args.candidates, 1 << 18)
    kernel(
        (min(65535, (warm_count + block - 1) // block),), (block,),
        (
            np.uint64(0), np.uint64(warm_count), seed_deltas, sequences,
            offsets, roles, coefficients, np.int32(BUNDLES), scores,
        ),
    )
    synchronize()

    rows = []
    for repetition in range(args.repetitions):
        _, kernel_seconds = timed(lambda: kernel((grid,), (block,), call_args))

        def select_topk():
            selected = cp.argpartition(scores, args.candidates - args.top_k)[-args.top_k:]
            selected_values = scores[selected]
            # CuPy 14.2 accepts a 2-D key array here, not NumPy's tuple form.
            # The last row is the primary key: descending score, then seed.
            keys = cp.stack((selected.astype(cp.float64), -selected_values.astype(cp.float64)))
            order = cp.lexsort(keys)
            return selected[order], selected_values[order]

        (selected, selected_values), topk_seconds = timed(select_topk)
        selected_host = cp.asnumpy(selected).astype("<u8", copy=False)
        values_host = cp.asnumpy(selected_values).astype("<f4", copy=False)
        rows.append({
            "repetition": repetition,
            "kernel_seconds": kernel_seconds,
            "topk_seconds": topk_seconds,
            "end_to_end_seconds": kernel_seconds + topk_seconds,
            "candidate_rate_per_second": args.candidates / kernel_seconds,
            "normal_value_rate_per_second": args.candidates * VALUES_PER_CANDIDATE / kernel_seconds,
            "topk_seed_sha256": hashlib.sha256(selected_host.tobytes()).hexdigest(),
            "topk_value_sha256": hashlib.sha256(values_host.tobytes()).hexdigest(),
            "best_seed": int(selected_host[0]),
            "best_score": float(values_host[0]),
        })

    kernel_median = statistics.median(row["kernel_seconds"] for row in rows)
    end_median = statistics.median(row["end_to_end_seconds"] for row in rows)
    scale = FULL_U32 / args.candidates
    pool = cp.get_default_memory_pool()
    result = {
        "schema": "fuseed_u32_source_free_throughput_probe_v0",
        "claim_boundary": "Unsealed source-free engineering calibration only; no Qwen/model path interface, retention claim, initializer result, or execution authorization.",
        "script_sha256": sha256_file(Path(__file__)),
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
            "normal4_bundles_per_candidate": BUNDLES,
            "normal_values_per_candidate": VALUES_PER_CANDIDATE,
            "top_k": args.top_k,
            "repetitions": args.repetitions,
            "score_buffer_bytes": int(scores.nbytes),
        },
        "rows": rows,
        "aggregate": {
            "median_kernel_seconds": kernel_median,
            "median_end_to_end_seconds": end_median,
            "projected_full_u32_kernel_seconds_linear": kernel_median * scale,
            "projected_full_u32_end_to_end_seconds_linear": end_median * scale,
            "median_kernel_normal_values_per_second": args.candidates * VALUES_PER_CANDIDATE / kernel_median,
            "memory_pool_total_bytes": int(pool.total_bytes()),
            "projection_warning": "Linear single-shard projection only; full implementation must measure exact sharding, journal, global Top-K merge, all ABI maps, and retention path.",
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
