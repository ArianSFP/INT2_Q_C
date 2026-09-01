"""Exact random access for the grouped-v5 layout-overlay initializer.

Importing this module is CUDA-free.  CuPy and NVRTC are reached only when
``PhiloxRandomAccess`` is explicitly constructed after the launch gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import common


CUDA_SOURCE = r'''
#include <curand_kernel.h>
#include <cuda_bf16.h>

__device__ __forceinline__ unsigned long long policy_increment(
    unsigned long long numel, int sm_count, int max_threads_per_sm) {
  const unsigned long long block_size = 256ULL;
  unsigned long long grid = (numel + block_size - 1ULL) / block_size;
  unsigned long long cap = (unsigned long long)sm_count *
                           (unsigned long long)(max_threads_per_sm / 256);
  if (grid > cap) grid = cap;
  return (((numel - 1ULL) / (block_size * grid * 4ULL)) + 1ULL) * 4ULL;
}

__device__ __forceinline__ float philox_normal_at(
    unsigned long long seed,
    unsigned long long numel,
    unsigned long long base_offset,
    unsigned long long native_index,
    int sm_count,
    int max_threads_per_sm) {
  const unsigned long long block_size = 256ULL;
  unsigned long long grid = (numel + block_size - 1ULL) / block_size;
  unsigned long long cap = (unsigned long long)sm_count *
                           (unsigned long long)(max_threads_per_sm / 256);
  if (grid > cap) grid = cap;
  unsigned long long stride = block_size * grid;
  unsigned long long sequence = native_index % stride;
  unsigned long long quotient = native_index / stride;
  int lane = (int)(quotient & 3ULL);
  unsigned long long calls_before = quotient >> 2;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, sequence, base_offset, &state);
  float4 values;
  for (unsigned long long call = 0; call <= calls_before; ++call) {
    values = curand_normal4(&state);
  }
  if (lane == 0) return values.x;
  if (lane == 1) return values.y;
  if (lane == 2) return values.z;
  return values.w;
}

extern "C" __global__ void descriptor_probe(
    const unsigned long long* seeds,
    const unsigned long long* numels,
    const unsigned long long* offsets,
    const unsigned long long* native_indices,
    unsigned long long count,
    int sm_count,
    int max_threads_per_sm,
    float* standard_out,
    float* scaled_out,
    float* bf16_widened_out) {
  unsigned long long start = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long step = (unsigned long long)blockDim.x * gridDim.x;
  for (unsigned long long index = start; index < count; index += step) {
    float normal = philox_normal_at(
        seeds[index], numels[index], offsets[index], native_indices[index],
        sm_count, max_threads_per_sm);
    float scaled = normal * 0.02f;
    standard_out[index] = normal;
    scaled_out[index] = scaled;
    __nv_bfloat16 rounded = __float2bfloat16_rn(scaled);
    bf16_widened_out[index] = __bfloat162float(rounded);
  }
}

extern "C" __global__ void generate_candidate_anchors(
    const unsigned long long* ordinals,
    unsigned long long candidate_count,
    const int* experts,
    const int* roles,
    const unsigned long long* canonical_coordinates,
    unsigned long long coordinate_count,
    int sm_count,
    int max_threads_per_sm,
    float* output) {
  unsigned long long total = candidate_count * coordinate_count;
  unsigned long long start = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long step = (unsigned long long)blockDim.x * gridDim.x;
  for (unsigned long long flat = start; flat < total; flat += step) {
    unsigned long long candidate_index = flat / coordinate_count;
    unsigned long long coordinate_index = flat - candidate_index * coordinate_count;
    unsigned long long value = ordinals[candidate_index];
    int abi_index = (int)(value % 2ULL); value /= 2ULL;
    int half_index = (int)(value % 2ULL); value /= 2ULL;
    int assignment_index = (int)(value % 2ULL); value /= 2ULL;
    int etp_index = (int)(value % 4ULL); value /= 4ULL;
    int ep_index = (int)(value % 8ULL); value /= 8ULL;
    int pp_index = (int)(value % 10ULL); value /= 10ULL;
    unsigned long long stored_seed_u16 = value;
    unsigned long long cli_base_seed = stored_seed_u16 + 1ULL;
    (void)abi_index;  // numbered and copy-packed ABIs are exact value equivalents.

    const int pp_sizes[10] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 48};
    int pp_size = pp_sizes[pp_index];
    int layers_per_stage = 48 / pp_size;
    int pp_rank = 15 / layers_per_stage;
    int local_layer = 15 - pp_rank * layers_per_stage;
    int ep_size = 1 << ep_index;
    int local_experts = 128 / ep_size;
    int expert = experts[coordinate_index];
    int ep_rank;
    int local_expert;
    if (assignment_index == 0) {
      ep_rank = expert / local_experts;
      local_expert = expert - ep_rank * local_experts;
    } else {
      ep_rank = expert % ep_size;
      local_expert = expert / ep_size;
    }

    int etp_size = 1 << etp_index;
    int local_rows = 768 / etp_size;
    unsigned long long coordinate = canonical_coordinates[coordinate_index];
    int canonical_row = (int)(coordinate / 2048ULL);
    int canonical_column = (int)(coordinate % 2048ULL);
    int etp_rank = canonical_row / local_rows;
    int local_row = canonical_row - etp_rank * local_rows;
    unsigned long long seed = cli_base_seed +
        (unsigned long long)(100 * pp_rank + 1024 + 100 * ep_rank + etp_rank);

    unsigned long long n = (unsigned long long)local_rows * 2048ULL;
    unsigned long long fused_n = 2ULL * n;
    unsigned long long inc_n = policy_increment(n, sm_count, max_threads_per_sm);
    unsigned long long inc_fused = policy_increment(fused_n, sm_count, max_threads_per_sm);
    unsigned long long layer_stride =
        (unsigned long long)local_experts * (inc_fused + inc_n);
    unsigned long long layer_base = (unsigned long long)local_layer * layer_stride;

    unsigned long long target_numel;
    unsigned long long target_offset;
    unsigned long long native_index;
    int role = roles[coordinate_index];  // 0=up, 1=down
    if (role == 0) {
      target_numel = fused_n;
      target_offset = layer_base + (unsigned long long)local_expert * inc_fused;
      // gate_then_up => canonical up occupies the second fused half.
      int fused_row = local_row + ((half_index == 0) ? local_rows : 0);
      native_index = (unsigned long long)fused_row * 2048ULL + canonical_column;
    } else {
      target_numel = n;
      target_offset = layer_base +
          (unsigned long long)local_experts * inc_fused +
          (unsigned long long)local_expert * inc_n;
      native_index = (unsigned long long)canonical_column * local_rows + local_row;
    }
    output[flat] = philox_normal_at(
        seed, target_numel, target_offset, native_index,
        sm_count, max_threads_per_sm);
  }
}
'''


def policy_increment(numel: int, sm_count: int, max_threads_per_sm: int) -> int:
    if numel <= 0 or sm_count <= 0 or max_threads_per_sm < 256:
        raise common.ProtocolError("invalid Philox execution-policy input")
    grid = min((numel + 255) // 256, sm_count * (max_threads_per_sm // 256))
    return ((numel - 1) // (256 * grid * 4) + 1) * 4


@dataclass(frozen=True)
class Descriptor:
    seed: int
    target_numel: int
    target_offset: int
    native_index: int
    pp_rank: int
    local_layer: int
    ep_rank: int
    local_expert: int
    etp_rank: int
    fc1_offset: int
    fc2_offset: int
    layer_stride: int


def coordinate_descriptor(
    candidate: common.CandidateKey,
    expert: int,
    role: str,
    canonical_coordinate: int,
    sm_count: int,
    max_threads_per_sm: int,
) -> Descriptor:
    if not 0 <= int(expert) < 128:
        raise common.ProtocolError("expert index outside Qwen geometry")
    if not 0 <= int(canonical_coordinate) < common.WEIGHTS_PER_MATRIX:
        raise common.ProtocolError("canonical coordinate outside matrix")
    pp = candidate.pipeline_parallel_size
    layers_per_stage = 48 // pp
    pp_rank, local_layer = divmod(15, layers_per_stage)
    ep = candidate.expert_parallel_size
    local_experts = 128 // ep
    if candidate.expert_assignment == "contiguous":
        ep_rank, local_expert = divmod(int(expert), local_experts)
    else:
        ep_rank = int(expert) % ep
        local_expert = int(expert) // ep
    etp = candidate.expert_tensor_parallel_size
    local_rows = common.ROWS // etp
    canonical_row, canonical_column = divmod(int(canonical_coordinate), common.COLUMNS)
    etp_rank, local_row = divmod(canonical_row, local_rows)
    seed = candidate.base_seed + 100 * pp_rank + 1024 + 100 * ep_rank + etp_rank
    n = local_rows * common.COLUMNS
    fused_n = 2 * n
    inc_n = policy_increment(n, sm_count, max_threads_per_sm)
    inc_fused = policy_increment(fused_n, sm_count, max_threads_per_sm)
    layer_stride = local_experts * (inc_fused + inc_n)
    layer_base = local_layer * layer_stride
    fc1_offset = layer_base + local_expert * inc_fused
    fc2_offset = layer_base + local_experts * inc_fused + local_expert * inc_n
    if role == "up":
        target_numel = fused_n
        target_offset = fc1_offset
        fused_row = local_row
        if candidate.canonical_fc1_half_assignment == "gate_then_up":
            fused_row += local_rows
        native = fused_row * common.COLUMNS + canonical_column
    elif role == "down":
        target_numel = n
        target_offset = fc2_offset
        native = canonical_column * local_rows + local_row
    else:
        raise common.ProtocolError("unsupported role")
    return Descriptor(
        seed,
        target_numel,
        target_offset,
        native,
        pp_rank,
        local_layer,
        ep_rank,
        local_expert,
        etp_rank,
        fc1_offset,
        fc2_offset,
        layer_stride,
    )


def emulated_call_trace(
    *,
    local_experts: int,
    local_layers: int,
    etp: int,
    sm_count: int,
    max_threads_per_sm: int,
) -> tuple[dict[str, int | str | list[int]], ...]:
    """Return the source-backed FC1-all-then-FC2 call trace without CUDA."""
    if local_experts <= 0 or local_layers <= 0 or etp not in common.ETP_SIZES:
        raise common.ProtocolError("invalid grouped trace geometry")
    local_rows = common.ROWS // etp
    n = local_rows * common.COLUMNS
    inc_fused = policy_increment(2 * n, sm_count, max_threads_per_sm)
    inc_n = policy_increment(n, sm_count, max_threads_per_sm)
    stride = local_experts * (inc_fused + inc_n)
    rows: list[dict[str, int | str | list[int]]] = []
    for layer in range(local_layers):
        base = layer * stride
        for expert in range(local_experts):
            rows.append(
                {
                    "local_layer": layer,
                    "projection": "fc1",
                    "local_expert": expert,
                    "shape": [2 * local_rows, common.COLUMNS],
                    "numel": 2 * n,
                    "offset": base + expert * inc_fused,
                    "increment": inc_fused,
                }
            )
        for expert in range(local_experts):
            rows.append(
                {
                    "local_layer": layer,
                    "projection": "fc2",
                    "local_expert": expert,
                    "shape": [common.COLUMNS, local_rows],
                    "numel": n,
                    "offset": base + local_experts * inc_fused + expert * inc_n,
                    "increment": inc_n,
                }
            )
    return tuple(rows)


class PhiloxRandomAccess:
    def __init__(self, device_index: int = 0):
        try:
            import cupy as cp
        except Exception as error:  # pragma: no cover - authorized GPU runtime only
            raise common.ProtocolError(f"CuPy unavailable: {error}") from error
        self.cp = cp
        self.device_index = int(device_index)
        cp.cuda.Device(device_index).use()
        properties = cp.cuda.runtime.getDeviceProperties(device_index)
        self.sm_count = int(properties["multiProcessorCount"])
        self.max_threads_per_sm = int(properties["maxThreadsPerMultiProcessor"])
        name = properties["name"]
        self.device_name = name.decode() if isinstance(name, bytes) else str(name)
        options = ("--std=c++17",)
        try:
            self.descriptor_kernel = cp.RawKernel(CUDA_SOURCE, "descriptor_probe", options=options)
            self.anchor_kernel = cp.RawKernel(
                CUDA_SOURCE, "generate_candidate_anchors", options=options
            )
            self.descriptor_kernel.compile()
            self.anchor_kernel.compile()
        except Exception as error:
            raise common.ProtocolError(f"Tier-C RawKernel compile failed: {error}") from error

    def descriptor_probe(
        self,
        seeds: Sequence[int],
        numels: Sequence[int],
        offsets: Sequence[int],
        native_indices: Sequence[int],
    ):
        cp = self.cp
        arrays = [
            cp.asarray(value, dtype=cp.uint64)
            for value in (seeds, numels, offsets, native_indices)
        ]
        count = len(arrays[0])
        if any(len(array) != count for array in arrays):
            raise common.ProtocolError("descriptor probe array-length mismatch")
        standard = cp.empty(count, dtype=cp.float32)
        scaled = cp.empty(count, dtype=cp.float32)
        bf16 = cp.empty(count, dtype=cp.float32)
        blocks = min(65_535, max(1, (count + 255) // 256))
        self.descriptor_kernel(
            (blocks,),
            (256,),
            (
                *arrays,
                np.uint64(count),
                np.int32(self.sm_count),
                np.int32(self.max_threads_per_sm),
                standard,
                scaled,
                bf16,
            ),
        )
        return standard, scaled, bf16

    def generate(
        self,
        ordinals,
        experts,
        roles,
        canonical_coordinates,
        *,
        output=None,
    ):
        cp = self.cp
        ordinals_gpu = cp.asarray(ordinals, dtype=cp.uint64)
        experts_gpu = cp.asarray(experts, dtype=cp.int32)
        roles_gpu = cp.asarray(roles, dtype=cp.int32)
        coordinates_gpu = cp.asarray(canonical_coordinates, dtype=cp.uint64)
        candidate_count = int(ordinals_gpu.size)
        coordinate_count = int(coordinates_gpu.size)
        if experts_gpu.size != coordinate_count or roles_gpu.size != coordinate_count:
            raise common.ProtocolError("coordinate metadata length mismatch")
        if output is None:
            output = cp.empty((candidate_count, coordinate_count), dtype=cp.float32)
        elif output.shape != (candidate_count, coordinate_count) or output.dtype != cp.float32:
            raise common.ProtocolError("anchor output shape/dtype mismatch")
        total = candidate_count * coordinate_count
        blocks = min(65_535, max(1, (total + 255) // 256))
        self.anchor_kernel(
            (blocks,),
            (256,),
            (
                ordinals_gpu,
                np.uint64(candidate_count),
                experts_gpu,
                roles_gpu,
                coordinates_gpu,
                np.uint64(coordinate_count),
                np.int32(self.sm_count),
                np.int32(self.max_threads_per_sm),
                output,
            ),
        )
        return output
