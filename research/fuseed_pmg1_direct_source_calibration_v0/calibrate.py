#!/usr/bin/env python3
"""Hardened source-free full-shard calibration for one-ABI exact FUSEED-PMG1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import time


EXPECTED_PLAN_SHA256 = "adcf1d8153c2a8a5048153edfa90f8f12d959d1d09e1cf7524359a532da950d1"
EXPECTED_DIRECT_SHA256 = "f5a7c8b9a525e02d469ca974f9a6607030b2ca2822b66d4bce31604251516ed5"
EXPECTED_DOMAIN_PROBE_SHA256 = "d3274d82e5321a33f0850ca02251635d41fefc804dd3d5960bac1c01cbab971a"
EXPECTED_PYTHONPATH = "/usr/local/lib/python3.12/dist-packages"
EXPECTED_RUNTIME_FILES = {
    "/usr/bin/python3.12": "1d3cf64f97cadc79fdc6fe2496a21b7b456cb94211978cfef5a65f616af74fd5",
    "/workspace/int2-cupy-venv/lib/python3.12/site-packages/numpy/__init__.py": "09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11",
    "/workspace/int2-cupy-venv/lib/python3.12/site-packages/cupy/__init__.py": "8c4724758587dea5f1c1d7c217c74a9fa0e4ed7f9d76a2b86fa001117cf3c718",
    "/workspace/int2-cupy-venv/lib/python3.12/site-packages/cupy/cuda/compiler.py": "09226d26ab41bf6e7b5b6e57b59187b4c3a5637690747af9a83d288a87d0fb6e",
    "/workspace/int2-cupy-venv/lib/python3.12/site-packages/cupy_backends/cuda/libs/nvrtc.cpython-312-x86_64-linux-gnu.so": "a3e9213226fa693231cab5e873aa1de8d31f7c6d82d9c56716c326ce438af373",
    "/workspace/int2-cupy-venv/lib/python3.12/site-packages/cupy_backends/cuda/api/runtime.cpython-312-x86_64-linux-gnu.so": "65a5c75db5e05c9bd35132b7f41631cadff6a6a6300acd85b273db3ba7ce28de",
    "/usr/local/lib/python3.12/dist-packages/torch/__init__.py": "2f0deb66d5dff6b9c02a62832c3bf3824c2ee031c462a3afeb9ca170466da5bf",
    "/usr/local/lib/python3.12/dist-packages/torch/_C.cpython-312-x86_64-linux-gnu.so": "db1e4f96208c6b297186585a04acee533035705706254ebdd4953fbae6b90224",
    "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12": "43731e24cd89e3749826304f304e8aa11fbecf1188715271b1f5018d6212b5e6",
    "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12": "c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.580.126.09": "e8e541166449da5a1278f40b27a28d072174b31f2941b101a9609b6d1d3aed32",
}
FULL_U32 = 1 << 32
SHARD_SIZE = 1 << 24
SHARDS = 256
TOP_K = 8192
REPETITIONS = 3
T = 261120
UP_SCALE_BITS = 0x3C03126F
DOWN_SCALE_BITS = 0x3A560A28
COMPILE_OPTIONS = ("--std=c++17", "-I/usr/local/cuda/include")
COMPILE_ARCH = "120"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, expected: str, name: str):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} hash mismatch: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_runtime_files() -> dict[str, dict[str, str]]:
    observed = {}
    for raw_path, expected in EXPECTED_RUNTIME_FILES.items():
        path = Path(raw_path)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"runtime binding is absent/nonregular/symlink: {raw_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"runtime file hash mismatch: {raw_path}: {actual}")
        observed[raw_path] = {"sha256": actual, "resolved": str(path.resolve())}
    return observed


def loaded_cuda_libraries() -> list[str]:
    paths = set()
    with Path("/proc/self/maps").open("r", encoding="utf-8") as handle:
        for line in handle:
            path = line.split()[-1]
            if path.startswith("/") and any(
                token in Path(path).name for token in ("libcuda.so", "libcudart.so", "libnvrtc.so")
            ):
                paths.add(str(Path(path).resolve()))
    return sorted(paths)


def synchronize(cp) -> None:
    cp.cuda.runtime.deviceSynchronize()


def compile_exact(cp, source: str, filename: str):
    from cupy.cuda import compiler

    blob, mapping = compiler.compile_using_nvrtc(
        source,
        options=COMPILE_OPTIONS,
        arch=COMPILE_ARCH,
        filename=filename,
    )
    if not isinstance(blob, bytes) or not blob.startswith(b"\x7fELF"):
        raise RuntimeError("NVRTC did not emit a loadable ELF cubin")
    if mapping is not None:
        raise RuntimeError("unexpected NVRTC name-expression mapping")
    module = cp.cuda.function.Module()
    module.load(blob)
    return module, {
        "filename": filename,
        "options": list(COMPILE_OPTIONS),
        "arch": COMPILE_ARCH,
        "cubin_bytes": len(blob),
        "cubin_sha256": hashlib.sha256(blob).hexdigest(),
        "cubin_magic_hex": blob[:8].hex(),
    }


PARITY_SOURCE = r'''
#include <curand_kernel.h>
#include <cuda_bf16.h>

#define STRIDE 261120ULL

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

__device__ __forceinline__ float scaled_bf16(float value, int role) {
  const float scale = role == 0 ? __uint_as_float(0x3c03126fU)
                                : __uint_as_float(0x3a560a28U);
  return __bfloat162float(__float2bfloat16_rn(value * scale));
}

extern "C" __global__ void direct_shifted_sequential_parity(
    const unsigned long long* bases,
    const unsigned long long* addends,
    const unsigned long long* sequences,
    const unsigned long long* offsets,
    const unsigned long long* normal4_indices,
    int row_count,
    float* direct_raw,
    float* shifted_raw,
    float* sequential_raw,
    float* direct_scaled,
    float* shifted_scaled,
    float* sequential_scaled,
    unsigned int* direct_counter_words,
    unsigned int* shifted_terminal_words,
    unsigned int* sequential_terminal_words) {
  const int row = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) return;
  const unsigned long long seed64 = bases[row] + addends[row];
  const unsigned long long j = normal4_indices[row];
  uint4 direct_counter;
  const float4 direct = direct_normal4(
      seed64, sequences[row], offsets[row], j, &direct_counter);
  curandStatePhilox4_32_10_t shifted_state;
  curand_init(seed64, sequences[row], offsets[row] + 4ULL * j, &shifted_state);
  const float4 shifted = curand_normal4(&shifted_state);
  curandStatePhilox4_32_10_t sequential_state;
  curand_init(seed64, sequences[row], offsets[row], &sequential_state);
  float4 sequential;
  for (unsigned long long call = 0; call <= j; ++call) {
    sequential = curand_normal4(&sequential_state);
  }
  const float direct_values[4] = {direct.x, direct.y, direct.z, direct.w};
  const float shifted_values[4] = {shifted.x, shifted.y, shifted.z, shifted.w};
  const float sequential_values[4] = {
      sequential.x, sequential.y, sequential.z, sequential.w};
  #pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    direct_raw[4 * row + lane] = direct_values[lane];
    shifted_raw[4 * row + lane] = shifted_values[lane];
    sequential_raw[4 * row + lane] = sequential_values[lane];
    #pragma unroll
    for (int role = 0; role < 2; ++role) {
      direct_scaled[8 * row + 4 * role + lane] = scaled_bf16(direct_values[lane], role);
      shifted_scaled[8 * row + 4 * role + lane] = scaled_bf16(shifted_values[lane], role);
      sequential_scaled[8 * row + 4 * role + lane] = scaled_bf16(sequential_values[lane], role);
    }
  }
  direct_counter_words[4 * row + 0] = direct_counter.x;
  direct_counter_words[4 * row + 1] = direct_counter.y;
  direct_counter_words[4 * row + 2] = direct_counter.z;
  direct_counter_words[4 * row + 3] = direct_counter.w;
  shifted_terminal_words[4 * row + 0] = shifted_state.ctr.x;
  shifted_terminal_words[4 * row + 1] = shifted_state.ctr.y;
  shifted_terminal_words[4 * row + 2] = shifted_state.ctr.z;
  shifted_terminal_words[4 * row + 3] = shifted_state.ctr.w;
  sequential_terminal_words[4 * row + 0] = sequential_state.ctr.x;
  sequential_terminal_words[4 * row + 1] = sequential_state.ctr.y;
  sequential_terminal_words[4 * row + 2] = sequential_state.ctr.z;
  sequential_terminal_words[4 * row + 3] = sequential_state.ctr.w;
}

extern "C" __global__ void torch_descriptor_probe(
    const unsigned long long* seeds,
    const unsigned long long* offsets,
    const unsigned long long* native_indices,
    const unsigned int* scale_bits,
    unsigned long long count,
    float* bf16_widened) {
  const unsigned long long index =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const unsigned long long native = native_indices[index];
  const unsigned long long sequence = native % STRIDE;
  const unsigned long long quotient = native / STRIDE;
  const int lane = (int)(quotient & 3ULL);
  const unsigned long long j = quotient >> 2;
  uint4 counter;
  const float4 values = direct_normal4(
      seeds[index], sequence, offsets[index], j, &counter);
  const float raw = lane == 0 ? values.x : lane == 1 ? values.y
      : lane == 2 ? values.z : values.w;
  bf16_widened[index] = __bfloat162float(
      __float2bfloat16_rn(raw * __uint_as_float(scale_bits[index])));
}
'''


def run_three_direct_replays(direct) -> dict:
    rows = [direct.run_parity() for _ in range(3)]
    encoded = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    if len(set(encoded)) != 1:
        raise RuntimeError("three direct parity replays differ")
    return {"repetitions": 3, "identical": True, "receipt": rows[0]}


def run_sequential_replays(cp, np, direct, module) -> dict:
    function = module.get_function("direct_shifted_sequential_parity")
    vectors = direct.make_parity_vectors()
    row_count = len(vectors["bases"])
    inputs = [cp.asarray(vectors[name]) for name in (
        "bases", "addends", "sequences", "offsets", "normal4_indices"
    )]
    receipts = []
    for _ in range(3):
        direct_raw = cp.empty((row_count, 4), dtype=cp.float32)
        shifted_raw = cp.empty_like(direct_raw)
        sequential_raw = cp.empty_like(direct_raw)
        direct_scaled = cp.empty((row_count, 8), dtype=cp.float32)
        shifted_scaled = cp.empty_like(direct_scaled)
        sequential_scaled = cp.empty_like(direct_scaled)
        direct_counter = cp.empty((row_count, 4), dtype=cp.uint32)
        shifted_terminal = cp.empty_like(direct_counter)
        sequential_terminal = cp.empty_like(direct_counter)
        function(
            ((row_count + 127) // 128,), (128,),
            tuple(inputs) + (
                np.int32(row_count), direct_raw, shifted_raw, sequential_raw,
                direct_scaled, shifted_scaled, sequential_scaled,
                direct_counter, shifted_terminal, sequential_terminal,
            ),
        )
        synchronize(cp)
        arrays = [cp.asnumpy(value) for value in (
            direct_raw, shifted_raw, sequential_raw,
            direct_scaled, shifted_scaled, sequential_scaled,
            direct_counter, shifted_terminal, sequential_terminal,
        )]
        d_raw, h_raw, s_raw, d_scaled, h_scaled, s_scaled, counter, h_terminal, s_terminal = arrays
        if not (
            np.array_equal(d_raw.view(np.uint32), h_raw.view(np.uint32))
            and np.array_equal(d_raw.view(np.uint32), s_raw.view(np.uint32))
            and np.array_equal(d_scaled.view(np.uint32), h_scaled.view(np.uint32))
            and np.array_equal(d_scaled.view(np.uint32), s_scaled.view(np.uint32))
        ):
            raise RuntimeError("direct/shifted/original-offset sequential parity failed")
        expected_terminal = direct.increment_counter_words(counter)
        if not np.array_equal(h_terminal, expected_terminal):
            raise RuntimeError("shifted terminal counter mismatch")
        if not np.array_equal(s_terminal, expected_terminal):
            raise RuntimeError("sequential terminal counter mismatch")
        receipts.append({
            "rows": row_count,
            "raw_float32_sha256": hashlib.sha256(d_raw.astype("<f4", copy=False).tobytes()).hexdigest(),
            "scaled_bf16_sha256": hashlib.sha256(d_scaled.astype("<f4", copy=False).tobytes()).hexdigest(),
            "counter_sha256": hashlib.sha256(counter.astype("<u4", copy=False).tobytes()).hexdigest(),
            "terminal_sha256": hashlib.sha256(h_terminal.astype("<u4", copy=False).tobytes()).hexdigest(),
            "direct_equals_shifted": True,
            "direct_equals_original_offset_sequential_j_plus_1": True,
            "terminal_counters_equal": True,
        })
    if len({json.dumps(row, sort_keys=True) for row in receipts}) != 1:
        raise RuntimeError("three sequential parity replays differ")
    return {"repetitions": 3, "identical": True, "receipt": receipts[0]}


def float_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def torch_cases():
    boundary = {
        0: (1 << 32) - 1024,
        1: (1 << 32) - 1124,
        2: (1 << 32) - 1224,
        3: (1 << 32) - 1324,
    }
    specifications = (
        (boundary[0], 0, "up"), (2358, 31, "down"),
        (boundary[1], 32, "up"), ((1 << 32) - 1, 63, "down"),
        (boundary[2], 64, "up"), (1, 95, "down"),
        (boundary[3], 96, "up"), ((1 << 32) - 2, 127, "down"),
    )
    cases = []
    for base, expert, role in specifications:
        ep_rank, local_expert = divmod(expert, 32)
        seed = base + 1024 + 100 * ep_rank
        if role == "up":
            numel = 3145728
            offset = 11520 + 16 * local_expert
            scale_bits = UP_SCALE_BITS
        else:
            numel = 1572864
            offset = 12032 + 8 * local_expert
            scale_bits = DOWN_SCALE_BITS
        natives = sorted({0, 1, T - 1, T, 4 * T - 1, 4 * T, numel - 1})
        cases.append({
            "base_seed": base, "expert": expert, "role": role,
            "effective_seed": seed, "numel": numel, "offset": offset,
            "scale_bits": scale_bits, "native_indices": natives,
        })
    return cases


def state_hash(generator, np) -> str:
    raw = generator.get_state().cpu().numpy().astype(np.uint8, copy=False)
    return hashlib.sha256(raw.tobytes()).hexdigest()


def run_torch_replays(cp, np, torch, module, props) -> dict:
    function = module.get_function("torch_descriptor_probe")
    cases = torch_cases()
    sm_count = int(props["multiProcessorCount"])
    max_threads = int(props["maxThreadsPerMultiProcessor"])
    for case in cases:
        grid = min((case["numel"] + 255) // 256, sm_count * (max_threads // 256))
        stride = 256 * grid
        if stride != T:
            raise RuntimeError(f"Torch policy stride mismatch: {stride}")
        case["expected_increment"] = ((case["numel"] - 1) // (stride * 4) + 1) * 4

    flat = [
        (case_index, native)
        for case_index, case in enumerate(cases)
        for native in case["native_indices"]
    ]
    seeds = cp.asarray(np.asarray([cases[i]["effective_seed"] for i, _ in flat], dtype=np.uint64))
    offsets = cp.asarray(np.asarray([cases[i]["offset"] for i, _ in flat], dtype=np.uint64))
    natives = cp.asarray(np.asarray([native for _, native in flat], dtype=np.uint64))
    scales = cp.asarray(np.asarray([cases[i]["scale_bits"] for i, _ in flat], dtype=np.uint32))
    output = cp.empty(len(flat), dtype=cp.float32)
    function(((len(flat) + 127) // 128,), (128,), (
        seeds, offsets, natives, scales, np.uint64(len(flat)), output,
    ))
    synchronize(cp)
    direct = cp.asnumpy(output).astype("<f4", copy=False)

    replay_receipts = []
    for _ in range(3):
        rows = []
        cursor = 0
        for case in cases:
            generator = torch.Generator(device="cuda:0")
            generator.manual_seed(int(case["effective_seed"]))
            if int(generator.initial_seed()) != case["effective_seed"]:
                raise RuntimeError("Torch initial seed mismatch")
            generator.set_offset(int(case["offset"]))
            if int(generator.get_offset()) != case["offset"]:
                raise RuntimeError("Torch initial offset mismatch")
            initial_state_sha256 = state_hash(generator, np)
            tensor = torch.empty(case["numel"], dtype=torch.bfloat16, device="cuda:0")
            tensor.normal_(
                0.0, float_from_bits(case["scale_bits"]), generator=generator
            )
            torch.cuda.synchronize()
            terminal_offset = int(generator.get_offset())
            expected_terminal = case["offset"] + case["expected_increment"]
            if terminal_offset != expected_terminal:
                raise RuntimeError("Torch terminal offset mismatch")
            terminal_state_sha256 = state_hash(generator, np)
            indices = torch.as_tensor(case["native_indices"], dtype=torch.int64, device="cuda:0")
            expected = tensor.index_select(0, indices).float().cpu().numpy().astype("<f4", copy=False)
            observed = direct[cursor : cursor + len(case["native_indices"])]
            if not np.array_equal(expected.view("<u4"), observed.view("<u4")):
                raise RuntimeError(
                    f"Torch/direct BF16 parity failed: e{case['expert']}/{case['role']}"
                )
            rows.append({
                **{key: case[key] for key in (
                    "base_seed", "expert", "role", "effective_seed", "numel",
                    "offset", "scale_bits", "native_indices", "expected_increment",
                )},
                "initial_seed": int(generator.initial_seed()),
                "initial_offset": case["offset"],
                "terminal_offset": terminal_offset,
                "initial_state_sha256": initial_state_sha256,
                "terminal_state_sha256": terminal_state_sha256,
                "bf16_widened_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
            })
            cursor += len(case["native_indices"])
            del tensor, indices
        torch.cuda.empty_cache()
        replay_receipts.append(rows)
    encoded = [json.dumps(rows, sort_keys=True, separators=(",", ":")) for rows in replay_receipts]
    if len(set(encoded)) != 1:
        raise RuntimeError("three Torch parity replays differ")
    return {
        "repetitions": 3,
        "identical": True,
        "case_count": len(cases),
        "coordinate_count": len(flat),
        "stride": T,
        "rows": replay_receipts[0],
        "direct_output_sha256": hashlib.sha256(direct.tobytes()).hexdigest(),
    }


def plan_arrays(np, plan_result):
    wire = plan_result["wire"]
    return tuple(
        np.asarray(values, dtype=np.uint64)
        for values in (
            [row["seed_addend"] for row in wire],
            [row["sequence"] for row in wire],
            [row["offset_quads"] for row in wire],
            [row["normal4_index"] for row in wire],
        )
    )


def canonical_topk(np, shape, q, top_k: int):
    seeds, values, threshold, ties = shape.exact_topk(q, 0, top_k)
    seeds = np.asarray(seeds, dtype=np.uint64)
    values = np.asarray(values, dtype=np.float32)
    order = np.lexsort((seeds, values))
    seeds = seeds[order]
    values = values[order]
    if len(seeds) != top_k or not np.isfinite(values).all():
        raise RuntimeError("canonical Top-K invariant failed")
    return seeds, values, float(threshold), int(ties)


def write_journal(path: Path, header: dict, seeds, values) -> tuple[str, int, float]:
    started = time.perf_counter()
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = (
        struct.pack("<I", len(header_bytes))
        + header_bytes
        + seeds.astype("<u4", copy=False).tobytes()
        + values.astype("<f4", copy=False).tobytes()
    )
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    elapsed = time.perf_counter() - started
    return hashlib.sha256(payload).hexdigest(), len(payload), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=SHARD_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.exists():
        raise RuntimeError("output and its parent must both be absent")
    if args.candidates != SHARD_SIZE or args.top_k != TOP_K or args.repetitions != REPETITIONS:
        raise RuntimeError("calibration shape must be exactly 2^24/8192/3")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if os.environ.get("PYTHONPATH") != EXPECTED_PYTHONPATH:
        raise RuntimeError("PYTHONPATH runtime binding mismatch")
    if Path(sys.executable).resolve() != Path("/usr/bin/python3.12"):
        raise RuntimeError("Python executable resolution mismatch")
    runtime_files = bind_runtime_files()

    # Python -I deliberately ignores PYTHONPATH.  Re-introduce the one exact,
    # hash-bound system package root only after the raw environment and every
    # file it can supply to this calibration have been authenticated above.
    if EXPECTED_PYTHONPATH in sys.path:
        raise RuntimeError("isolated launch unexpectedly preloaded the bound system package root")
    # Append, rather than prepend: the authenticated venv must continue to
    # supply NumPy/CuPy 2.5.2/14.2.0, while this root supplies Torch only.
    sys.path.append(EXPECTED_PYTHONPATH)

    import cupy as cp
    import numpy as np
    import torch
    from cupy.cuda import nvrtc

    if (np.__version__, cp.__version__, torch.__version__) != (
        "2.5.2", "14.2.0", "2.8.0+cu128"
    ):
        raise RuntimeError("Python package version mismatch")
    if torch.version.cuda != "12.8":
        raise RuntimeError("Torch CUDA version mismatch")
    if cp.cuda.runtime.runtimeGetVersion() != 12090:
        raise RuntimeError("CUDA runtime version mismatch")
    if cp.cuda.runtime.driverGetVersion() != 13000:
        raise RuntimeError("CUDA driver API version mismatch")
    if nvrtc.getVersion() != (12, 8):
        raise RuntimeError("NVRTC version mismatch")

    root = Path(__file__).resolve().parent
    plan_path = root / "plan.py"
    research = root.parent
    direct_path = research / "fuseed_u32_direct_counter_calibration_v0" / "calibrate_direct.py"
    domain_path = research / "fuseed_u32_direct_domain_collapse_probe_v0" / "probe.py"
    plan = load_module(plan_path, EXPECTED_PLAN_SHA256, "fuseed_pmg1_plan")
    direct = load_module(direct_path, EXPECTED_DIRECT_SHA256, "fuseed_direct")
    domain = load_module(domain_path, EXPECTED_DOMAIN_PROBE_SHA256, "fuseed_domain")
    plan_result = plan.reconstruct_plan()
    shape_path, shape = direct.load_shape_module()
    header_hashes = direct.bind_headers()
    direct_source, direct_counts = direct.derive_direct_source(shape.CUDA_SOURCE)
    active_source, active_counts = domain.derive_active_domain_source(direct_source, 1)
    if active_source.count("curand_init(") or active_source.count("curand_normal4("):
        raise RuntimeError("stateful generator call survived performance source derivation")

    props = cp.cuda.runtime.getDeviceProperties(0)
    device_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    if device_name != "NVIDIA GeForce RTX 5090" or cp.cuda.Device().compute_capability != "120":
        raise RuntimeError("GPU/compute-capability mismatch")

    performance_module, performance_compile = compile_exact(
        cp, active_source, "fuseed_pmg1_stage0.cu"
    )
    parity_module, parity_compile = compile_exact(
        cp, PARITY_SOURCE, "fuseed_pmg1_parity.cu"
    )
    performance = performance_module.get_function("fuseed_33domain_scores")
    direct_replays = run_three_direct_replays(direct)
    sequential_replays = run_sequential_replays(cp, np, direct, parity_module)
    torch_replays = run_torch_replays(cp, np, torch, parity_module, props)

    loaded_libraries = loaded_cuda_libraries()
    expected_loaded = sorted(
        path for path in EXPECTED_RUNTIME_FILES
        if Path(path).name.startswith(("libcuda.so", "libcudart.so", "libnvrtc.so"))
    )
    if loaded_libraries != expected_loaded:
        raise RuntimeError(
            f"loaded CUDA library set mismatch: {loaded_libraries} != {expected_loaded}"
        )

    targets_host, stats_host = shape.make_targets()
    targets = cp.asarray(targets_host)
    target_stats = cp.asarray(stats_host)
    plan_host = plan_arrays(np, plan_result)
    addends, sequences, offset_quads, normal4_indices = [cp.asarray(value) for value in plan_host]
    q = cp.empty(args.candidates, dtype=cp.float32)
    block = 256
    warps = 8
    grid = min(65535, (args.candidates + warps - 1) // warps)

    def launch(count: int) -> None:
        performance(
            (min(grid, (count + warps - 1) // warps),), (block,),
            (
                np.uint64(0), np.uint64(count), addends, sequences,
                offset_quads, normal4_indices, targets, target_stats, q,
            ),
        )

    launch(1 << 14)
    synchronize(cp)
    args.output.parent.mkdir(parents=False, exist_ok=False)
    rows = []
    retained_seeds = None
    retained_values = None
    sample_indices = cp.linspace(0, args.candidates - 1, 32768, dtype=cp.int64)
    for repetition in range(args.repetitions):
        q.fill(cp.nan)
        synchronize(cp)
        start = time.perf_counter()
        launch(args.candidates)
        synchronize(cp)
        kernel_seconds = time.perf_counter() - start

        start = time.perf_counter()
        finite = bool(cp.asnumpy(cp.isfinite(q).all()))
        synchronize(cp)
        finite_seconds = time.perf_counter() - start
        if not finite:
            raise RuntimeError("source q contains a nonfinite value")

        start = time.perf_counter()
        seeds, values, threshold, ties = canonical_topk(np, shape, q, args.top_k)
        synchronize(cp)
        topk_seconds = time.perf_counter() - start
        sample = cp.asnumpy(q[sample_indices]).astype("<f4", copy=False)
        seed_hash = hashlib.sha256(seeds.astype("<u4", copy=False).tobytes()).hexdigest()
        value_hash = hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()
        q_hash = hashlib.sha256(sample.tobytes()).hexdigest()
        journal_header = {
            "schema": "fuseed_pmg1_stage0_shard_journal_v0",
            "repetition": repetition,
            "shard_base_u32": 0,
            "candidate_count": args.candidates,
            "top_k": args.top_k,
            "metric_order": "q ascending then base_seed_u32 ascending",
            "performance_cubin_sha256": performance_compile["cubin_sha256"],
            "plan_bundle_sha256": plan_result["facts"]["abi1_category_ordered_bundle_sha256"],
            "seed_sha256": seed_hash,
            "value_sha256": value_hash,
        }
        journal_hash, journal_bytes, journal_seconds = write_journal(
            args.output.parent / f"shard_replay_{repetition}.bin",
            journal_header, seeds, values,
        )
        rows.append({
            "repetition": repetition,
            "kernel_seconds": kernel_seconds,
            "finite_validation_seconds": finite_seconds,
            "topk_seconds": topk_seconds,
            "journal_fsync_seconds": journal_seconds,
            "shard_end_to_end_seconds": kernel_seconds + finite_seconds + topk_seconds + journal_seconds,
            "threshold_q": threshold,
            "boundary_tie_cardinality": ties,
            "best_seed_u32": int(seeds[0]),
            "best_q": float(values[0]),
            "topk_seed_sha256": seed_hash,
            "topk_value_sha256": value_hash,
            "q_sentinel_sha256": q_hash,
            "journal_sha256": journal_hash,
            "journal_bytes": journal_bytes,
        })
        retained_seeds, retained_values = seeds, values

    for field in ("topk_seed_sha256", "topk_value_sha256", "q_sentinel_sha256"):
        if len({row[field] for row in rows}) != 1:
            raise RuntimeError(f"three shard replays differ: {field}")
    if retained_seeds is None or retained_values is None:
        raise RuntimeError("missing retained Top-K")

    start = time.perf_counter()
    merge_seeds = np.concatenate(
        [retained_seeds + np.uint64(shard * SHARD_SIZE) for shard in range(SHARDS)]
    )
    merge_values = np.tile(retained_values, SHARDS)
    merge_order = np.lexsort((merge_seeds, merge_values))[:TOP_K]
    global_seeds = merge_seeds[merge_order].astype("<u4", copy=False)
    global_values = merge_values[merge_order].astype("<f4", copy=False)
    global_merge_seconds = time.perf_counter() - start
    global_seed_hash = hashlib.sha256(global_seeds.tobytes()).hexdigest()
    global_value_hash = hashlib.sha256(global_values.tobytes()).hexdigest()

    shard_median = statistics.median(row["shard_end_to_end_seconds"] for row in rows)
    cold_excess = max(0.0, rows[0]["shard_end_to_end_seconds"] - shard_median)
    projection = shard_median * SHARDS + cold_excess + global_merge_seconds
    result = {
        "schema": "fuseed_pmg1_source_free_hardened_calibration_v0",
        "status": (
            "SOURCE_FREE_RUNTIME_GATE_PASS_PENDING_INDEPENDENT_AUDIT"
            if projection < 800.0 else "EARLY_KILL_RUNTIME_NO_QWEN"
        ),
        "claim_boundary": (
            "Exact source-free PMG ABI1 runtime/parity calibration only; synthetic targets "
            "do not test initializer capture, Qwen weights, compression MSE, or significance."
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "source_bindings": {
            "plan": {"path": str(plan_path), "sha256": EXPECTED_PLAN_SHA256},
            "direct": {"path": str(direct_path), "sha256": EXPECTED_DIRECT_SHA256},
            "domain_probe": {"path": str(domain_path), "sha256": EXPECTED_DOMAIN_PROBE_SHA256},
            "shape": {"path": str(shape_path), "sha256": direct.EXPECTED_SHAPE_SOURCE_SHA256},
            "cuda_headers": header_hashes,
        },
        "runtime": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "cupy": cp.__version__, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_api": int(cp.cuda.runtime.driverGetVersion()),
            "nvrtc": list(nvrtc.getVersion()),
            "device": device_name,
            "compute_capability": cp.cuda.Device().compute_capability,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "pythonpath": os.environ["PYTHONPATH"],
            "file_bindings": runtime_files,
            "loaded_cuda_libraries": loaded_libraries,
        },
        "plan": plan_result["facts"],
        "derivation": {
            "direct_replacement_counts": direct_counts,
            "active_domain_replacement_counts": active_counts,
            "active_domains": 1,
            "abi_id": "CURRENT_PMG_GATE_UP_DIRECT_BF16",
            "generator_and_exact_fp64_score_unchanged": True,
            "performance_cuda_sha256": hashlib.sha256(active_source.encode()).hexdigest(),
        },
        "compiled_kernels": {
            "performance": performance_compile,
            "parity": parity_compile,
            "parity_source_sha256": hashlib.sha256(PARITY_SOURCE.encode()).hexdigest(),
        },
        "parity": {
            "direct_shifted_reference_three_replays": direct_replays,
            "direct_shifted_and_original_offset_sequential_three_replays": sequential_replays,
            "torch_initial_terminal_and_bf16_three_replays": torch_replays,
        },
        "shape": {
            "candidates_per_shard": args.candidates,
            "shards": SHARDS,
            "complete_candidate_count": FULL_U32,
            "abi_count": 1,
            "active_domains": 1,
            "normal4_bundles_per_candidate": 256,
            "normal_values_per_candidate": 1024,
            "top_k": args.top_k,
            "repetitions": args.repetitions,
            "q_bytes": int(q.nbytes),
            "block_threads": block,
            "warps_per_block": warps,
            "grid_blocks": grid,
        },
        "rows": rows,
        "global_merge_shape_probe": {
            "input_records": SHARDS * TOP_K,
            "output_records": TOP_K,
            "seconds": global_merge_seconds,
            "seed_sha256": global_seed_hash,
            "value_sha256": global_value_hash,
            "uses_synthetic_repeated_shard_metrics": True,
        },
        "aggregate": {
            "median_complete_shard_seconds": shard_median,
            "one_time_cold_excess_seconds": cold_excess,
            "projected_complete_u32_seconds_including_finite_topk_journal_and_global_merge": projection,
            "prospective_margin_gate_seconds": 800.0,
            "projection_below_gate": projection < 800.0,
            "replay_deterministic": True,
        },
        "access": {
            "model_or_qwen_path_arguments": 0,
            "payload_files_opened": 0,
            "network_operations": 0,
        },
    }
    with args.output.open("xb") as handle:
        payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "status": result["status"],
        "projection_seconds": projection,
        "performance_cubin_sha256": performance_compile["cubin_sha256"],
        "parity_cubin_sha256": parity_compile["cubin_sha256"],
        "topk_seed_sha256": rows[0]["topk_seed_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
