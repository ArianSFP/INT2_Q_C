#!/usr/bin/env python3
"""Exact CuPy regeneration primitive for the five frozen PMG1 anchors.

This module has no CLI and performs no action on import.  A future audited
dispatcher must provide CuPy, NumPy, authenticated coordinate descriptors,
an approved device, and all payload/output authority.  The only operation
defined here is deterministic procedural anchor regeneration.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

try:
    from . import contract
except ImportError:  # Direct source-tree import for tests.
    import contract  # type: ignore[no-redef]


T = 261120

CUDA_SOURCE = r'''
#include <curand_kernel.h>
#include <cuda_bf16.h>

extern "C" __global__ void pmg_anchor_coords(
    const unsigned long long* bases,
    const unsigned long long* addends,
    const unsigned long long* sequences,
    const unsigned long long* offset_quads,
    const unsigned long long* normal4_indices,
    const unsigned char* lanes,
    const unsigned char* role_codes,
    unsigned long long coordinate_count,
    unsigned long long seed_count,
    float* output) {
  const unsigned long long linear =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long total = coordinate_count * seed_count;
  if (linear >= total) return;
  const unsigned long long seed_index = linear / coordinate_count;
  const unsigned long long coordinate = linear - seed_index * coordinate_count;
  const unsigned long long seed64 = bases[seed_index] + addends[coordinate];
  const unsigned long long offset_base = offset_quads[coordinate];
  const unsigned long long counter_low = offset_base + normal4_indices[coordinate];
  const unsigned long long carry = counter_low < offset_base ? 1ULL : 0ULL;
  const unsigned long long counter_high = sequences[coordinate] + carry;
  const uint4 counter = make_uint4(
      (unsigned int)counter_low, (unsigned int)(counter_low >> 32),
      (unsigned int)counter_high, (unsigned int)(counter_high >> 32));
  const uint2 key = make_uint2(
      (unsigned int)seed64, (unsigned int)(seed64 >> 32));
  const uint4 raw = curand_Philox4x32_10(counter, key);
  const float2 pair0 = _curand_box_muller(raw.x, raw.y);
  const float2 pair1 = _curand_box_muller(raw.z, raw.w);
  const float values[4] = {pair0.x, pair0.y, pair1.x, pair1.y};
  const float scale = role_codes[coordinate] == 0
      ? __uint_as_float(0x3c03126fU) : __uint_as_float(0x3a560a28U);
  output[linear] = __bfloat162float(
      __float2bfloat16_rn(values[(int)lanes[coordinate]] * scale));
}
'''

CUDA_SOURCE_SHA256 = hashlib.sha256(CUDA_SOURCE.encode("utf-8")).hexdigest()


def coordinate_arrays(numpy_module: Any, descriptors: Sequence[tuple[int, str, int, int]]):
    """Map canonical Up/Down coordinates to the exact frozen PMG ABI counters."""
    addends, sequences, offsets, normal4, lanes, roles = [], [], [], [], [], []
    for expert_value, role_value, row_value, column_value in descriptors:
        expert = int(expert_value)
        role = str(role_value)
        row = int(row_value)
        column = int(column_value)
        contract.require((expert, role) in contract.identity_set(), "anchor identity")
        contract.require(0 <= row < 768 and 0 <= column < 2048, "anchor coordinate")
        if role == "up":
            native = (row + 768) * 2048 + column
            offset_values = 11520 + 16 * (expert % 32)
            role_code = 0
        else:
            native = column * 768 + row
            offset_values = 12032 + 8 * (expert % 32)
            role_code = 1
        sequence = native % T
        quotient = native // T
        lane = quotient & 3
        normal4_index = quotient >> 2
        contract.require(
            sequence + T * (4 * normal4_index + lane) == native,
            "native coordinate inversion",
        )
        contract.require(offset_values % 4 == 0, "offset quad alignment")
        addends.append(1024 + 100 * (expert // 32))
        sequences.append(sequence)
        offsets.append(offset_values // 4)
        normal4.append(normal4_index)
        lanes.append(lane)
        roles.append(role_code)
    contract.require(len(addends) == len(descriptors), "coordinate array length")
    return (
        numpy_module.asarray(addends, dtype=numpy_module.uint64),
        numpy_module.asarray(sequences, dtype=numpy_module.uint64),
        numpy_module.asarray(offsets, dtype=numpy_module.uint64),
        numpy_module.asarray(normal4, dtype=numpy_module.uint64),
        numpy_module.asarray(lanes, dtype=numpy_module.uint8),
        numpy_module.asarray(roles, dtype=numpy_module.uint8),
    )


def compile_kernel(cupy_module: Any):
    """Compile only after a future dispatcher has authenticated the runtime."""
    module = cupy_module.RawModule(
        code=CUDA_SOURCE,
        options=("--std=c++17", "-I/usr/local/cuda/include"),
        backend="nvrtc",
        name_expressions=("pmg_anchor_coords",),
    )
    return module.get_function("pmg_anchor_coords")


def generate_anchors(
    cupy_module: Any,
    numpy_module: Any,
    kernel: Any,
    descriptors: Sequence[tuple[int, str, int, int]],
):
    """Return canonical coordinate-major, seed-ordinal float64 anchor rows."""
    arrays = coordinate_arrays(numpy_module, descriptors)
    device_arrays = [cupy_module.asarray(value) for value in arrays]
    bases = cupy_module.asarray(
        numpy_module.asarray(contract.SEEDS_U32, dtype=numpy_module.uint64)
    )
    count = len(descriptors)
    contract.require(count > 0, "nonempty anchor request")
    output = cupy_module.empty((len(contract.SEEDS_U32), count), dtype=cupy_module.float32)
    total = int(output.size)
    block = 256
    kernel(
        ((total + block - 1) // block,),
        (block,),
        (
            bases,
            *device_arrays,
            numpy_module.uint64(count),
            numpy_module.uint64(len(contract.SEEDS_U32)),
            output,
        ),
    )
    cupy_module.cuda.runtime.deviceSynchronize()
    contract.require(bool(cupy_module.isfinite(output).all().item()), "finite generated anchors")
    return cupy_module.asnumpy(output.T).astype(numpy_module.float64, copy=False)


__all__ = (
    "CUDA_SOURCE",
    "CUDA_SOURCE_SHA256",
    "compile_kernel",
    "coordinate_arrays",
    "generate_anchors",
)
