#!/usr/bin/env python3
"""Source-free exact direct-Philox FUSEED shard calibration and parity gate."""

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


EXPECTED_SHAPE_SOURCE_SHA256 = (
    "6d3cdac6ab4f1a1fcbe742f43a2fd817c8bf56d1bfeff9e38c2432fe6848149a"
)
CUDA_HEADERS = {
    "/usr/local/cuda/include/curand_normal.h": "967998564d9f9f4a045563b2b5d2a15eb1cbdfa18b0a332707c3a765e09a61c0",
    "/usr/local/cuda/include/curand_kernel.h": "4a37c07a1d77c9b5c8c627a4720733cee6b4da4200a844e8a49291858bc26adf",
    "/usr/local/cuda/include/curand_philox4x32_x.h": "4f6d483fe45d837fed49553d25ee1d2cabb012a138a2f7f08bbaf584e63dd83c",
    "/usr/local/cuda/include/cuda_bf16.h": "4d5a2ad88adb17983aef0505ed6ed2a0603497c79c103bba82c928301ea12310",
    "/usr/local/cuda/include/cuda_fp16.h": "8eb1600a8e2e33d40572bffe7001a27f6046a74949d80353fe02cf88b2563dda",
}
T = 261120
FULL_U32 = 1 << 32
ABIS = 3
UP_SCALE_BITS = 0x3C03126F
DOWN_SCALE_BITS = 0x3A560A28


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shape_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "fuseed_u32_33domain_calibration_v0"
        / "calibrate.py"
    )
    actual = sha256_file(path)
    if actual != EXPECTED_SHAPE_SOURCE_SHA256:
        raise RuntimeError(f"shape source hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("fuseed_shape_calibration", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shape calibration")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def bind_headers() -> dict[str, str]:
    observed = {}
    for raw_path, expected in CUDA_HEADERS.items():
        path = Path(raw_path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"CUDA header hash mismatch for {raw_path}: {actual}")
        observed[raw_path] = actual
    return observed


def derive_direct_source(source: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    old_scale = '''__device__ __forceinline__ float bf16_anchor(float value) {
  return __bfloat162float(__float2bfloat16_rn(value * 0.02f));
}'''
    new_scale = '''__device__ __forceinline__ float bf16_anchor(float value, int category) {
  const float scale = (category == 0 || category == 2)
      ? __uint_as_float(0x3c03126fU) : __uint_as_float(0x3a560a28U);
  return __bfloat162float(__float2bfloat16_rn(value * scale));
}'''
    counts["scale_function"] = source.count(old_scale)
    if counts["scale_function"] != 1:
        raise RuntimeError("unexpected BF16 scale function cardinality")
    source = source.replace(old_scale, new_scale, 1)

    old_signature = '''    const unsigned long long* offsets,
    const float* targets,'''
    new_signature = '''    const unsigned long long* offset_quads,
    const unsigned long long* normal4_indices,
    const float* targets,'''
    counts["signature"] = source.count(old_signature)
    if counts["signature"] != 1:
        raise RuntimeError("unexpected kernel signature cardinality")
    source = source.replace(old_signature, new_signature, 1)

    old_generation = '''      curandStatePhilox4_32_10_t state;
      curand_init(base_seed + seed_deltas[bundle], sequences[bundle],
                  offsets[bundle], &state);
      float4 generated = curand_normal4(&state);
      generated.x = bf16_anchor(generated.x);
      generated.y = bf16_anchor(generated.y);
      generated.z = bf16_anchor(generated.z);
      generated.w = bf16_anchor(generated.w);'''
    new_generation = '''      const unsigned long long seed64 = base_seed + seed_deltas[bundle];
      const unsigned long long offset_base = offset_quads[bundle];
      const unsigned long long normal4_index = normal4_indices[bundle];
      const unsigned long long counter_low = offset_base + normal4_index;
      const unsigned long long carry = counter_low < offset_base ? 1ULL : 0ULL;
      const unsigned long long counter_high = sequences[bundle] + carry;
      const uint4 counter = make_uint4(
          (unsigned int)counter_low, (unsigned int)(counter_low >> 32),
          (unsigned int)counter_high, (unsigned int)(counter_high >> 32));
      const uint2 key = make_uint2(
          (unsigned int)seed64, (unsigned int)(seed64 >> 32));
      const uint4 raw = curand_Philox4x32_10(counter, key);
      const float2 pair0 = _curand_box_muller(raw.x, raw.y);
      const float2 pair1 = _curand_box_muller(raw.z, raw.w);
      float4 generated = make_float4(pair0.x, pair0.y, pair1.x, pair1.y);
      const int bundle_category = category_of(4 * bundle);
      generated.x = bf16_anchor(generated.x, bundle_category);
      generated.y = bf16_anchor(generated.y, bundle_category);
      generated.z = bf16_anchor(generated.z, bundle_category);
      generated.w = bf16_anchor(generated.w, bundle_category);'''
    counts["generator"] = source.count(old_generation)
    if counts["generator"] != 1:
        raise RuntimeError("unexpected generator replacement cardinality")
    source = source.replace(old_generation, new_generation, 1)
    if "curand_init(" in source or "curand_normal4(" in source:
        raise RuntimeError("stateful CURAND call survived direct-source derivation")
    return source, counts


PARITY_SOURCE = r'''
#include <curand_kernel.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float4 direct_normal4(
    unsigned long long seed64, unsigned long long sequence,
    unsigned long long offset_values, unsigned long long normal4_index,
    uint4* direct_counter) {
  const unsigned long long offset_quads = offset_values >> 2;
  const unsigned long long counter_low = offset_quads + normal4_index;
  const unsigned long long carry = counter_low < offset_quads ? 1ULL : 0ULL;
  const unsigned long long counter_high = sequence + carry;
  const uint4 counter = make_uint4(
      (unsigned int)counter_low, (unsigned int)(counter_low >> 32),
      (unsigned int)counter_high, (unsigned int)(counter_high >> 32));
  const uint2 key = make_uint2((unsigned int)seed64, (unsigned int)(seed64 >> 32));
  const uint4 raw = curand_Philox4x32_10(counter, key);
  const float2 pair0 = _curand_box_muller(raw.x, raw.y);
  const float2 pair1 = _curand_box_muller(raw.z, raw.w);
  *direct_counter = counter;
  return make_float4(pair0.x, pair0.y, pair1.x, pair1.y);
}

extern "C" __global__ void direct_reference_parity(
    const unsigned long long* bases,
    const unsigned long long* addends,
    const unsigned long long* sequences,
    const unsigned long long* offsets,
    const unsigned long long* normal4_indices,
    int row_count,
    float* direct_raw,
    float* reference_raw,
    float* direct_scaled,
    float* reference_scaled,
    unsigned int* direct_counter_words,
    unsigned int* terminal_counter_words) {
  const int row = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) return;
  const unsigned long long seed64 = bases[row] + addends[row];
  uint4 direct_counter;
  const float4 direct = direct_normal4(
      seed64, sequences[row], offsets[row], normal4_indices[row], &direct_counter);
  curandStatePhilox4_32_10_t state;
  curand_init(seed64, sequences[row], offsets[row] + 4ULL * normal4_indices[row], &state);
  const float4 reference = curand_normal4(&state);
  const float direct_values[4] = {direct.x, direct.y, direct.z, direct.w};
  const float reference_values[4] = {reference.x, reference.y, reference.z, reference.w};
  #pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    direct_raw[4 * row + lane] = direct_values[lane];
    reference_raw[4 * row + lane] = reference_values[lane];
    const float up_scale = __uint_as_float(0x3c03126fU);
    const float down_scale = __uint_as_float(0x3a560a28U);
    direct_scaled[8 * row + lane] = __bfloat162float(
        __float2bfloat16_rn(direct_values[lane] * up_scale));
    reference_scaled[8 * row + lane] = __bfloat162float(
        __float2bfloat16_rn(reference_values[lane] * up_scale));
    direct_scaled[8 * row + 4 + lane] = __bfloat162float(
        __float2bfloat16_rn(direct_values[lane] * down_scale));
    reference_scaled[8 * row + 4 + lane] = __bfloat162float(
        __float2bfloat16_rn(reference_values[lane] * down_scale));
  }
  direct_counter_words[4 * row + 0] = direct_counter.x;
  direct_counter_words[4 * row + 1] = direct_counter.y;
  direct_counter_words[4 * row + 2] = direct_counter.z;
  direct_counter_words[4 * row + 3] = direct_counter.w;
  terminal_counter_words[4 * row + 0] = state.ctr.x;
  terminal_counter_words[4 * row + 1] = state.ctr.y;
  terminal_counter_words[4 * row + 2] = state.ctr.z;
  terminal_counter_words[4 * row + 3] = state.ctr.w;
}

extern "C" __global__ void philox_zero_kat(unsigned int* words) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const uint4 raw = curand_Philox4x32_10(
        make_uint4(0, 0, 0, 0), make_uint2(0, 0));
    words[0] = raw.x; words[1] = raw.y; words[2] = raw.z; words[3] = raw.w;
  }
}
'''


def max_full_bundle_index(call_size: int, sequence: int = 0) -> int:
    maximum_q = (call_size - 1 - sequence) // T
    return max(0, (maximum_q - 3) // 4)


def make_parity_vectors() -> dict[str, np.ndarray]:
    addends = (1024, 1124, 1224, 1324)
    offsets = (0, 388, 8760, 9148, (1 << 34) - 4, 1 << 34, (1 << 34) + 4)
    call_sizes = (100663296, 50331648, 3145728, 1572864)
    max_indices = tuple(max_full_bundle_index(size) for size in call_sizes)
    rows: set[tuple[int, int, int, int, int]] = set()
    for base in (0, 1, 2358, (1 << 32) - 1):
        for addend in addends:
            rows.add((base, addend, 0, 0, 0))
    for addend in addends:
        boundary = (1 << 32) - addend
        for delta in (-1, 0, 1):
            rows.add((boundary + delta, addend, T - 1, 8760, 1))
    for sequence in (0, 1, T - 1):
        for normal4_index in (0, 1) + max_indices:
            for offset in offsets:
                rows.add((2358, 1024, sequence, offset, normal4_index))
    ordered = sorted(rows)
    columns = tuple(np.asarray([row[index] for row in ordered], dtype=np.uint64) for index in range(5))
    return {
        "bases": columns[0], "addends": columns[1], "sequences": columns[2],
        "offsets": columns[3], "normal4_indices": columns[4],
        "call_sizes": np.asarray(call_sizes, dtype=np.uint64),
        "max_indices": np.asarray(max_indices, dtype=np.uint64),
    }


def increment_counter_words(counter_rows: np.ndarray) -> np.ndarray:
    result = counter_rows.copy()
    for row in range(result.shape[0]):
        carry = 1
        for word in range(4):
            if not carry:
                break
            value = int(result[row, word]) + carry
            result[row, word] = np.uint32(value & 0xFFFFFFFF)
            carry = value >> 32
    return result


def run_parity() -> dict:
    vectors = make_parity_vectors()
    row_count = len(vectors["bases"])
    device_inputs = [cp.asarray(vectors[name]) for name in (
        "bases", "addends", "sequences", "offsets", "normal4_indices"
    )]
    direct_raw = cp.empty((row_count, 4), dtype=cp.float32)
    reference_raw = cp.empty_like(direct_raw)
    direct_scaled = cp.empty((row_count, 8), dtype=cp.float32)
    reference_scaled = cp.empty_like(direct_scaled)
    direct_counter = cp.empty((row_count, 4), dtype=cp.uint32)
    terminal_counter = cp.empty_like(direct_counter)
    parity = cp.RawKernel(PARITY_SOURCE, "direct_reference_parity", options=("--std=c++17",))
    parity(
        ((row_count + 127) // 128,), (128,),
        tuple(device_inputs) + (
            np.int32(row_count), direct_raw, reference_raw, direct_scaled,
            reference_scaled, direct_counter, terminal_counter,
        ),
    )
    zero_words = cp.empty(4, dtype=cp.uint32)
    zero = cp.RawKernel(PARITY_SOURCE, "philox_zero_kat", options=("--std=c++17",))
    zero((1,), (1,), (zero_words,))
    cp.cuda.runtime.deviceSynchronize()

    direct_raw_host = cp.asnumpy(direct_raw).astype("<f4", copy=False)
    reference_raw_host = cp.asnumpy(reference_raw).astype("<f4", copy=False)
    direct_scaled_host = cp.asnumpy(direct_scaled).astype("<f4", copy=False)
    reference_scaled_host = cp.asnumpy(reference_scaled).astype("<f4", copy=False)
    counter_host = cp.asnumpy(direct_counter).astype("<u4", copy=False)
    terminal_host = cp.asnumpy(terminal_counter).astype("<u4", copy=False)
    if not np.array_equal(direct_raw_host.view("<u4"), reference_raw_host.view("<u4")):
        raise RuntimeError("direct/reference raw float32 parity failed")
    if not np.array_equal(direct_scaled_host.view("<u4"), reference_scaled_host.view("<u4")):
        raise RuntimeError("direct/reference scaled BF16 parity failed")
    expected_terminal = increment_counter_words(counter_host)
    if not np.array_equal(terminal_host, expected_terminal):
        raise RuntimeError("direct/reference terminal counter parity failed")
    kat = cp.asnumpy(zero_words).astype("<u4", copy=False)
    expected_kat = np.asarray(
        [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8], dtype="<u4"
    )
    if not np.array_equal(kat, expected_kat):
        raise RuntimeError("Philox zero KAT failed")
    vector_bytes = b"".join(
        np.asarray(vectors[name], dtype="<u8").tobytes()
        for name in ("bases", "addends", "sequences", "offsets", "normal4_indices")
    )
    return {
        "rows": row_count,
        "vector_sha256": hashlib.sha256(vector_bytes).hexdigest(),
        "raw_float32_sha256": hashlib.sha256(direct_raw_host.tobytes()).hexdigest(),
        "scaled_widened_bf16_sha256": hashlib.sha256(direct_scaled_host.tobytes()).hexdigest(),
        "direct_counter_sha256": hashlib.sha256(counter_host.tobytes()).hexdigest(),
        "terminal_counter_sha256": hashlib.sha256(terminal_host.tobytes()).hexdigest(),
        "raw_bitwise_equal": True,
        "scaled_bf16_bitwise_equal": True,
        "terminal_counter_equal": True,
        "zero_kat_words": [f"{int(value):08x}" for value in kat],
        "offset_values": [0, 388, 8760, 9148, (1 << 34) - 4, 1 << 34, (1 << 34) + 4],
        "max_normal4_indices_by_call_size": [int(value) for value in vectors["max_indices"]],
    }


def make_bundle_plan(shape) -> tuple[np.ndarray, ...]:
    ordinal = np.arange(shape.BUNDLES, dtype=np.uint64)
    addends = 1024 + 100 * (ordinal % 4)
    sequences = (ordinal * 7919 + 17) % T
    category = np.where(ordinal < 61, 0, np.where(ordinal < 128, 1, np.where(ordinal < 189, 2, 3)))
    offset_values = np.asarray((8760, 9148, 0, 388), dtype=np.uint64)[category]
    offset_quads = offset_values // 4
    normal4_indices = (ordinal * 13 + 5) % 97
    return addends, sequences, offset_quads, normal4_indices


def synchronize() -> None:
    cp.cuda.runtime.deviceSynchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=1 << 24)
    parser.add_argument("--top-k", type=int, default=8192)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.exists():
        raise RuntimeError("output and its parent must both be absent")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not (1 << 18) <= args.candidates <= (1 << 24):
        raise RuntimeError("candidate count outside calibration bound")
    if args.top_k != 8192 or args.repetitions != 3:
        raise RuntimeError("Top-K/repetitions must be exactly 8192/3")

    shape_path, shape = load_shape_module()
    header_hashes = bind_headers()
    direct_source, replacement_counts = derive_direct_source(shape.CUDA_SOURCE)
    parity_receipt = run_parity()
    props = cp.cuda.runtime.getDeviceProperties(0)
    free_before, total_memory = cp.cuda.runtime.memGetInfo()
    targets_host, stats_host = shape.make_targets()
    targets = cp.asarray(targets_host)
    target_stats = cp.asarray(stats_host)
    plan_host = make_bundle_plan(shape)
    addends, sequences, offset_quads, normal4_indices = [cp.asarray(value) for value in plan_host]
    q = cp.empty((shape.DOMAINS, args.candidates), dtype=cp.float32)
    free_after_allocation, _ = cp.cuda.runtime.memGetInfo()

    kernel = cp.RawKernel(direct_source, "fuseed_33domain_scores", options=("--std=c++17",))
    kernel.compile()
    block = 256
    warps_per_block = 8
    grid = min(65535, (args.candidates + warps_per_block - 1) // warps_per_block)

    def launch(count: int) -> None:
        kernel(
            (min(grid, (count + warps_per_block - 1) // warps_per_block),),
            (block,),
            (
                np.uint64(0), np.uint64(count), addends, sequences, offset_quads,
                normal4_indices, targets, target_stats, q,
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
        for domain_index in range(shape.DOMAINS):
            synchronize()
            start = time.perf_counter()
            seeds, values, _, ties = shape.exact_topk(q[domain_index], 0, args.top_k)
            synchronize()
            topk_seconds.append(time.perf_counter() - start)
            domain_seeds.append(seeds)
            domain_values.append(values)
            tie_sizes.append(ties)
        seeds_host = np.stack(domain_seeds).astype("<u8", copy=False)
        values_host = np.stack(domain_values).astype("<f4", copy=False)
        sample_indices = cp.linspace(0, q.size - 1, 16384, dtype=cp.int64)
        sample = cp.asnumpy(q.reshape(-1)[sample_indices]).astype("<f4", copy=False)
        selection_seconds = finite_seconds + sum(topk_seconds)
        rows.append({
            "repetition": repetition,
            "kernel_seconds": kernel_seconds,
            "finite_validation_seconds": finite_seconds,
            "cold_or_warm_first_domain_topk_seconds": topk_seconds[0],
            "remaining_domain_topk_seconds_sum": sum(topk_seconds[1:]),
            "selection_seconds_total": selection_seconds,
            "end_to_end_seconds_excluding_journal": kernel_seconds + selection_seconds,
            "candidate_rate_per_second_kernel": args.candidates / kernel_seconds,
            "normal4_bundle_rate_per_second_kernel": args.candidates * shape.BUNDLES / kernel_seconds,
            "normal_value_rate_per_second_kernel": args.candidates * shape.VALUES / kernel_seconds,
            "max_boundary_tie_cardinality": max(tie_sizes),
            "domain_topk_seed_sha256": hashlib.sha256(seeds_host.tobytes()).hexdigest(),
            "domain_topk_value_sha256": hashlib.sha256(values_host.tobytes()).hexdigest(),
            "q_sentinel_sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
            "best_seed_domain_0": int(seeds_host[0, 0]),
            "best_q_domain_0": float(values_host[0, 0]),
        })

    for field in ("domain_topk_seed_sha256", "domain_topk_value_sha256", "q_sentinel_sha256"):
        if len({row[field] for row in rows}) != 1:
            raise RuntimeError(f"replay differs for {field}")
    kernel_median = statistics.median(row["kernel_seconds"] for row in rows)
    selection_median = statistics.median(row["selection_seconds_total"] for row in rows)
    cold_excess = max(0.0, rows[0]["selection_seconds_total"] - selection_median)
    shards_per_abi = FULL_U32 // args.candidates
    total_shards = shards_per_abi * ABIS
    projected_kernel = kernel_median * total_shards
    projected_warm_e2e = (kernel_median + selection_median) * total_shards + cold_excess

    result = {
        "schema": "fuseed_u32_source_free_direct_counter_calibration_v0",
        "claim_boundary": (
            "Source-free exact direct-counter runtime calibration; no exact frozen bundle-plan "
            "digest, journal write, stage1/2, model/Qwen access, retention, or run authorization."
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "shape_source": {"path": str(shape_path), "sha256": EXPECTED_SHAPE_SOURCE_SHA256},
        "cuda_headers": header_hashes,
        "direct_source": {
            "derived_cuda_sha256": hashlib.sha256(direct_source.encode()).hexdigest(),
            "replacement_counts": replacement_counts,
            "curand_init_calls_in_performance_kernel": direct_source.count("curand_init("),
            "curand_normal4_calls_in_performance_kernel": direct_source.count("curand_normal4("),
            "direct_philox_calls_in_performance_kernel": direct_source.count("curand_Philox4x32_10("),
            "box_muller_pair_calls_in_performance_kernel": direct_source.count("_curand_box_muller("),
        },
        "parity": parity_receipt,
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
            "domains": shape.DOMAINS,
            "normal4_bundles_per_candidate": shape.BUNDLES,
            "normal_values_per_candidate": shape.VALUES,
            "fit_counts": {"up": 244, "down": 268},
            "score_counts": {"up": 244, "down": 268},
            "q_shape": [shape.DOMAINS, args.candidates],
            "q_bytes": int(q.nbytes),
            "top_k_per_domain": args.top_k,
            "repetitions": args.repetitions,
            "scale_bits": {"up": f"{UP_SCALE_BITS:08x}", "down": f"{DOWN_SCALE_BITS:08x}"},
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
            "median_warm_selection_seconds_per_shard": selection_median,
            "one_time_cold_selection_excess_seconds": cold_excess,
            "shards_per_abi": shards_per_abi,
            "abis": ABIS,
            "projected_three_abi_kernel_seconds": projected_kernel,
            "projected_three_abi_warm_end_to_end_seconds_excluding_journal": projected_warm_e2e,
            "prospective_runtime_gate_seconds": 900.0,
            "kernel_projection_below_gate": projected_kernel < 900.0,
            "warm_e2e_projection_below_gate": projected_warm_e2e < 900.0,
            "projection_warning": (
                "Exact source-free performance shape and direct generator, but linear full-search "
                "projection excludes frozen bundle-plan binding, journal/global merge, stage1/2 and validation."
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
