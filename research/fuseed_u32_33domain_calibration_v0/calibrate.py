#!/usr/bin/env python3
"""Source-free CuPy calibration of the frozen FUSEED 33-domain shard shape."""

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


DOMAINS = 33
BUNDLES = 256
VALUES = 1024
UP_FIT = 244
DOWN_FIT = 268
UP_SCORE = 244
DOWN_SCORE = 268
FULL_U32 = 1 << 32
ABIS = 3


CUDA_SOURCE = r'''
#include <curand_kernel.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define DOMAIN_COUNT 33
#define BUNDLE_COUNT 256
#define VALUE_COUNT 1024
#define WARPS_PER_BLOCK 8

__device__ __forceinline__ int category_of(int coordinate) {
  if (coordinate < 244) return 0;
  if (coordinate < 512) return 1;
  if (coordinate < 756) return 2;
  return 3;
}

__device__ __forceinline__ float bf16_anchor(float value) {
  return __bfloat162float(__float2bfloat16_rn(value * 0.02f));
}

__device__ __forceinline__ double decoded_half(double value) {
  return (double)__half2float(__double2half(value));
}

__device__ __forceinline__ double role_sse(
    int fit_cat, int score_cat, double sum_x[4], double sum_x2[4],
    double sum_xw[4], const double* target_stats, int domain) {
  const int fit_n = fit_cat == 0 ? 244 : 268;
  const int score_n = score_cat == 2 ? 244 : 268;
  const double sw_fit = target_stats[(domain * 4 + fit_cat) * 2 + 0];
  const double sw_score = target_stats[(domain * 4 + score_cat) * 2 + 0];
  const double sw2_score = target_stats[(domain * 4 + score_cat) * 2 + 1];
  const double mean_w = sw_fit / (double)fit_n;
  const double centered_x2 =
      sum_x2[fit_cat] - sum_x[fit_cat] * sum_x[fit_cat] / (double)fit_n;
  const double centered_wx =
      sum_xw[fit_cat] - sum_x[fit_cat] * sw_fit / (double)fit_n;
  const double alpha_raw = centered_x2 > 0.0 ? centered_wx / centered_x2 : 0.0;
  const double mu_raw = mean_w - alpha_raw * sum_x[fit_cat] / (double)fit_n;
  const double alpha = decoded_half(alpha_raw);
  const double mu = decoded_half(mu_raw);
  double sse = sw2_score + (double)score_n * mu * mu
      + alpha * alpha * sum_x2[score_cat]
      + 2.0 * mu * alpha * sum_x[score_cat]
      - 2.0 * mu * sw_score - 2.0 * alpha * sum_xw[score_cat];
  return sse > 0.0 ? sse : 0.0;
}

__device__ __forceinline__ double role_baseline(
    int fit_cat, int score_cat, const double* target_stats, int domain) {
  const int fit_n = fit_cat == 0 ? 244 : 268;
  const int score_n = score_cat == 2 ? 244 : 268;
  const double sw_fit = target_stats[(domain * 4 + fit_cat) * 2 + 0];
  const double sw_score = target_stats[(domain * 4 + score_cat) * 2 + 0];
  const double sw2_score = target_stats[(domain * 4 + score_cat) * 2 + 1];
  const double mean_w = sw_fit / (double)fit_n;
  return sw2_score - 2.0 * mean_w * sw_score
      + (double)score_n * mean_w * mean_w;
}

__device__ __forceinline__ float domain_q(
    int domain, double sum_x[4], double sum_x2[4], double sum_xw[4],
    const double* target_stats) {
  const double sse = role_sse(0, 2, sum_x, sum_x2, sum_xw, target_stats, domain)
      + role_sse(1, 3, sum_x, sum_x2, sum_xw, target_stats, domain);
  const double baseline = role_baseline(0, 2, target_stats, domain)
      + role_baseline(1, 3, target_stats, domain);
  return (float)(sse / baseline);
}

extern "C" __global__ void fuseed_33domain_scores(
    unsigned long long shard_base,
    unsigned long long candidate_count,
    const unsigned long long* seed_deltas,
    const unsigned long long* sequences,
    const unsigned long long* offsets,
    const float* targets,
    const double* target_stats,
    float* q) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const unsigned long long warp_start =
      (unsigned long long)blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
  const unsigned long long warp_step =
      (unsigned long long)gridDim.x * WARPS_PER_BLOCK;
  const unsigned mask = 0xffffffffu;
  const int domain0 = lane == 0 ? 0 : lane + 1;
  const int domain1 = 1;

  for (unsigned long long candidate = warp_start; candidate < candidate_count;
       candidate += warp_step) {
    double local_sum_x[4] = {0.0, 0.0, 0.0, 0.0};
    double local_sum_x2[4] = {0.0, 0.0, 0.0, 0.0};
    double sum_xw0[4] = {0.0, 0.0, 0.0, 0.0};
    double sum_xw1[4] = {0.0, 0.0, 0.0, 0.0};
    const unsigned long long base_seed = shard_base + candidate;

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int bundle = 32 * t + lane;
      curandStatePhilox4_32_10_t state;
      curand_init(base_seed + seed_deltas[bundle], sequences[bundle],
                  offsets[bundle], &state);
      float4 generated = curand_normal4(&state);
      generated.x = bf16_anchor(generated.x);
      generated.y = bf16_anchor(generated.y);
      generated.z = bf16_anchor(generated.z);
      generated.w = bf16_anchor(generated.w);

      const int own_base = 4 * bundle;
      const float own_values[4] = {generated.x, generated.y, generated.z, generated.w};
      #pragma unroll
      for (int component = 0; component < 4; ++component) {
        const int category = category_of(own_base + component);
        const double x = (double)own_values[component];
        local_sum_x[category] += x;
        local_sum_x2[category] += x * x;
      }

      #pragma unroll
      for (int source_lane = 0; source_lane < 32; ++source_lane) {
        const int coordinate_base = 4 * (32 * t + source_lane);
        const float received[4] = {
            __shfl_sync(mask, generated.x, source_lane),
            __shfl_sync(mask, generated.y, source_lane),
            __shfl_sync(mask, generated.z, source_lane),
            __shfl_sync(mask, generated.w, source_lane)};
        #pragma unroll
        for (int component = 0; component < 4; ++component) {
          const int coordinate = coordinate_base + component;
          const int category = category_of(coordinate);
          const double x = (double)received[component];
          sum_xw0[category] += x * (double)__ldg(
              &targets[domain0 * VALUE_COUNT + coordinate]);
          if (lane == 0) {
            sum_xw1[category] += x * (double)__ldg(
                &targets[domain1 * VALUE_COUNT + coordinate]);
          }
        }
      }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      #pragma unroll
      for (int category = 0; category < 4; ++category) {
        local_sum_x[category] += __shfl_down_sync(mask, local_sum_x[category], offset);
        local_sum_x2[category] += __shfl_down_sync(mask, local_sum_x2[category], offset);
      }
    }
    #pragma unroll
    for (int category = 0; category < 4; ++category) {
      local_sum_x[category] = __shfl_sync(mask, local_sum_x[category], 0);
      local_sum_x2[category] = __shfl_sync(mask, local_sum_x2[category], 0);
    }

    q[(unsigned long long)domain0 * candidate_count + candidate] =
        domain_q(domain0, local_sum_x, local_sum_x2, sum_xw0, target_stats);
    if (lane == 0) {
      q[(unsigned long long)domain1 * candidate_count + candidate] =
          domain_q(domain1, local_sum_x, local_sum_x2, sum_xw1, target_stats);
    }
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


def make_targets() -> tuple[np.ndarray, np.ndarray]:
    domain = np.arange(DOMAINS, dtype=np.float64)[:, None]
    coordinate = np.arange(VALUES, dtype=np.float64)[None, :]
    targets = (
        np.sin(0.173 + 0.019 * domain + coordinate * 0.6180339887498949)
        + 0.37 * np.cos(0.311 + 0.071 * domain + coordinate * 0.4142135623730950)
    ).astype(np.float32)
    # Give every domain/role a distinct finite mean and variance without any
    # model-derived statistics.
    targets += ((domain % 5) - 2.0).astype(np.float32) * np.float32(0.003)
    boundaries = ((0, 244), (244, 512), (512, 756), (756, 1024))
    stats = np.empty((DOMAINS, 4, 2), dtype=np.float64)
    for category, (start, stop) in enumerate(boundaries):
        section = targets[:, start:stop].astype(np.float64)
        stats[:, category, 0] = np.sum(section, axis=1, dtype=np.float64)
        stats[:, category, 1] = np.sum(section * section, axis=1, dtype=np.float64)
    return targets, stats


def exact_topk(row, shard_base: int, top_k: int):
    provisional = cp.argpartition(row, top_k - 1)[:top_k]
    threshold = float(cp.asnumpy(cp.max(row[provisional])))
    # q is an error ratio, so lower is better. Strictly smaller values precede
    # the threshold; ties use the implicit u32 seed.
    better = cp.asnumpy(cp.nonzero(row < threshold)[0]).astype(np.uint64)
    equal = cp.asnumpy(cp.nonzero(row == threshold)[0]).astype(np.uint64)
    need = top_k - better.size
    if need < 0 or equal.size < need:
        raise RuntimeError("Top-K threshold cardinality invariant failed")
    equal.sort()
    chosen = np.concatenate((better, equal[:need]))
    values = cp.asnumpy(row[cp.asarray(chosen)]).astype("<f4", copy=False)
    seeds = (np.uint64(shard_base) + chosen).astype("<u8", copy=False)
    order = np.lexsort((seeds, values.astype(np.float64)))
    return seeds[order], values[order], threshold, int(equal.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=1 << 24)
    parser.add_argument("--top-k", type=int, default=8192)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.exists():
        raise RuntimeError("output and its parent must both be absent")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not (1 << 18) <= args.candidates <= (1 << 24):
        raise RuntimeError("candidate count outside calibration bound")
    if not 1 <= args.top_k < args.candidates:
        raise RuntimeError("invalid Top-K")
    if not 2 <= args.repetitions <= 3:
        raise RuntimeError("two or three repetitions required")

    free_before, total_memory = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(0)
    targets_host, stats_host = make_targets()
    targets = cp.asarray(targets_host)
    target_stats = cp.asarray(stats_host)
    ordinal = np.arange(BUNDLES, dtype=np.uint64)
    seed_deltas = cp.asarray(1024 + 100 * (ordinal % 4), dtype=cp.uint64)
    sequences = cp.asarray((ordinal * 7919 + 17) % 1048576, dtype=cp.uint64)
    offsets = cp.asarray(4 * ((ordinal * 13 + 5) % 257), dtype=cp.uint64)
    q = cp.empty((DOMAINS, args.candidates), dtype=cp.float32)
    free_after_allocation, _ = cp.cuda.runtime.memGetInfo()

    kernel = cp.RawKernel(CUDA_SOURCE, "fuseed_33domain_scores", options=("--std=c++17",))
    kernel.compile()
    block = 256
    warps_per_block = block // 32
    grid = min(65535, (args.candidates + warps_per_block - 1) // warps_per_block)

    def launch(count: int) -> None:
        kernel(
            (min(grid, (count + warps_per_block - 1) // warps_per_block),),
            (block,),
            (
                np.uint64(0), np.uint64(count), seed_deltas, sequences,
                offsets, targets, target_stats, q,
            ),
        )

    launch(min(args.candidates, 1 << 14))
    synchronize()
    attributes = {key: int(value) for key, value in kernel.attributes.items()}

    rows = []
    for repetition in range(args.repetitions):
        synchronize()
        start = time.perf_counter()
        launch(args.candidates)
        synchronize()
        kernel_seconds = time.perf_counter() - start

        start = time.perf_counter()
        finite = bool(cp.asnumpy(cp.isfinite(q).all()))
        synchronize()
        finite_seconds = time.perf_counter() - start
        if not finite:
            raise RuntimeError("full q contains a nonfinite value")

        domain_seeds = []
        domain_values = []
        topk_seconds = []
        tie_sizes = []
        for domain_index in range(DOMAINS):
            synchronize()
            start = time.perf_counter()
            seeds, values, _, ties = exact_topk(q[domain_index], 0, args.top_k)
            synchronize()
            topk_seconds.append(time.perf_counter() - start)
            domain_seeds.append(seeds)
            domain_values.append(values)
            tie_sizes.append(ties)
        seeds_host = np.stack(domain_seeds).astype("<u8", copy=False)
        values_host = np.stack(domain_values).astype("<f4", copy=False)

        sample_indices = cp.linspace(0, q.size - 1, 16384, dtype=cp.int64)
        sample = cp.asnumpy(q.reshape(-1)[sample_indices]).astype("<f4", copy=False)
        end_to_end = kernel_seconds + finite_seconds + sum(topk_seconds)
        rows.append({
            "repetition": repetition,
            "kernel_seconds": kernel_seconds,
            "finite_validation_seconds": finite_seconds,
            "cold_first_domain_topk_seconds": topk_seconds[0],
            "remaining_domain_topk_seconds_sum": sum(topk_seconds[1:]),
            "median_domain_topk_seconds": statistics.median(topk_seconds),
            "end_to_end_seconds_excluding_journal": end_to_end,
            "candidate_rate_per_second_kernel": args.candidates / kernel_seconds,
            "anchor_value_rate_per_second_kernel": args.candidates * VALUES / kernel_seconds,
            "domain_cross_moments_per_second_kernel": args.candidates * DOMAINS * VALUES / kernel_seconds,
            "max_boundary_tie_cardinality": max(tie_sizes),
            "domain_topk_seed_sha256": hashlib.sha256(seeds_host.tobytes()).hexdigest(),
            "domain_topk_value_sha256": hashlib.sha256(values_host.tobytes()).hexdigest(),
            "q_sentinel_sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
            "best_seed_domain_0": int(seeds_host[0, 0]),
            "best_q_domain_0": float(values_host[0, 0]),
        })

    if len({row["domain_topk_seed_sha256"] for row in rows}) != 1:
        raise RuntimeError("Top-K seed replay differs")
    if len({row["domain_topk_value_sha256"] for row in rows}) != 1:
        raise RuntimeError("Top-K value replay differs")
    if len({row["q_sentinel_sha256"] for row in rows}) != 1:
        raise RuntimeError("q replay differs")

    kernel_median = statistics.median(row["kernel_seconds"] for row in rows)
    e2e_median = statistics.median(row["end_to_end_seconds_excluding_journal"] for row in rows)
    shards_per_abi = FULL_U32 // args.candidates
    result = {
        "schema": "fuseed_u32_source_free_33domain_calibration_v0",
        "claim_boundary": (
            "Unsealed synthetic runtime calibration only; no exact producer ABI parity, "
            "model/Qwen access, scientific retention, journal, or run authorization."
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "runtime": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "device": props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "frozen_shape_emulated": {
            "domains": DOMAINS,
            "normal4_bundles_per_candidate": BUNDLES,
            "widened_bf16_anchor_values_per_candidate": VALUES,
            "fit_counts": {"up": UP_FIT, "down": DOWN_FIT},
            "score_counts": {"up": UP_SCORE, "down": DOWN_SCORE},
            "common_fp64_moments": 8,
            "domain_fp64_cross_moments": 4,
            "affine_storage": "alpha and mu independently FP16-round/reload",
            "domain_ownership": "lane0 source+control0; lanes1..31 controls1..31",
            "candidate_count": args.candidates,
            "q_shape": [DOMAINS, args.candidates],
            "q_bytes": int(q.nbytes),
            "top_k_per_domain": args.top_k,
            "repetitions": args.repetitions,
        },
        "kernel": {
            "attributes": attributes,
            "register_spill_proxy_local_size_bytes": attributes.get("local_size_bytes"),
            "block_threads": block,
            "warps_per_block": warps_per_block,
            "grid_blocks": grid,
        },
        "memory": {
            "device_total_bytes": int(total_memory),
            "device_free_before_bytes": int(free_before),
            "device_free_after_q_and_inputs_bytes": int(free_after_allocation),
            "measured_allocation_delta_bytes": int(free_before - free_after_allocation),
            "cupy_pool_total_bytes": int(cp.get_default_memory_pool().total_bytes()),
            "cupy_pool_used_bytes": int(cp.get_default_memory_pool().used_bytes()),
        },
        "rows": rows,
        "aggregate": {
            "median_kernel_seconds_per_shard": kernel_median,
            "median_end_to_end_seconds_per_shard_excluding_journal": e2e_median,
            "shards_per_abi": shards_per_abi,
            "abis": ABIS,
            "projected_three_abi_kernel_seconds": kernel_median * shards_per_abi * ABIS,
            "projected_three_abi_end_to_end_seconds_excluding_journal": e2e_median * shards_per_abi * ABIS,
            "projection_warning": (
                "Linear planning projection from one source-free shard; excludes exact ABI parity, "
                "compile, journals, global merges, controls construction, stage1/2, and validation."
            ),
            "replay_deterministic": True,
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
