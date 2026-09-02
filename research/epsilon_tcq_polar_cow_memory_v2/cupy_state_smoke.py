#!/usr/bin/env python3
"""Source-free CuPy allocation and primitive-kernel smoke for the v2 state plan.

This is deliberately not a production polar list decoder.  It proves that the
literal frozen buffers can coexist on one CUDA device and that compact f/g and
stable survivor-selection primitives can execute there.  It cannot promote the
separate compute gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Any

from memory_plan import memory_plan


CAP_BYTES = 4 * (1 << 30)


TOUCH_SOURCE = r'''
extern "C" __global__
void touch_edges(unsigned char* x, unsigned long long n, unsigned char tag) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && n > 0) {
        x[0] = tag;
        x[n - 1] = (unsigned char)(tag ^ 0x5a);
    }
}
'''


SC_SOURCE = r'''
extern "C" __global__
void compact_fg(const double* left, const double* right,
                const unsigned char* bit, double* f_out, double* g_out,
                unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double l = left[i];
    double r = right[i];
    double f = (l * r + 1.0) / (l + r);
    double g = bit[i] ? (r / l) : (l * r);
    f_out[i] = fmin(1.0e30, fmax(1.0e-30, f));
    g_out[i] = fmin(1.0e30, fmax(1.0e-30, g));
}

extern "C" __global__
void stable_topk_serial(const double* metric, int* selected, int count, int keep) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int j = 0; j < keep; ++j) {
        int best = -1;
        for (int i = 0; i < count; ++i) {
            bool used = false;
            for (int k = 0; k < j; ++k) used = used || selected[k] == i;
            if (used) continue;
            if (best < 0 || metric[i] < metric[best] ||
                (metric[i] == metric[best] && i < best)) best = i;
        }
        selected[j] = best;
    }
}
'''


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def primitive_smoke(cp: Any) -> dict[str, Any]:
    import numpy as np

    left_np = np.asarray([0.125, 0.5, 1.0, 2.0, 8.0, 17.0, 0.03125, 32.0],
                         dtype=np.float64)
    right_np = np.asarray([8.0, 3.0, 0.25, 4.0, 0.5, 0.0625, 16.0, 2.0],
                          dtype=np.float64)
    bit_np = np.asarray([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    left, right, bit = cp.asarray(left_np), cp.asarray(right_np), cp.asarray(bit_np)
    f_out, g_out = cp.empty_like(left), cp.empty_like(left)
    fg = cp.RawKernel(SC_SOURCE, "compact_fg", options=("--std=c++11",))
    fg((1,), (32,), (left, right, bit, f_out, g_out, left_np.size))
    cp.cuda.runtime.deviceSynchronize()
    f_expected = np.clip((left_np * right_np + 1.0) / (left_np + right_np),
                         1e-30, 1e30)
    g_expected = np.clip(np.where(bit_np != 0, right_np / left_np,
                                  left_np * right_np), 1e-30, 1e30)
    f_got, g_got = cp.asnumpy(f_out), cp.asnumpy(g_out)
    if not np.allclose(f_got, f_expected, rtol=2e-15, atol=0.0):
        raise RuntimeError("CuPy compact f primitive mismatch")
    if not np.allclose(g_got, g_expected, rtol=2e-15, atol=0.0):
        raise RuntimeError("CuPy compact g primitive mismatch")

    metrics_np = np.asarray([3.0, 1.0, 1.0, 4.0, -2.0, -2.0, 0.0, 7.0,
                             0.5, 0.5, 5.0, 8.0, 2.0, 6.0, 9.0, 10.0],
                            dtype=np.float64)
    metrics = cp.asarray(metrics_np)
    selected = cp.full(8, -1, dtype=cp.int32)
    topk = cp.RawKernel(SC_SOURCE, "stable_topk_serial", options=("--std=c++11",))
    topk((1,), (1,), (metrics, selected, metrics_np.size, selected.size))
    cp.cuda.runtime.deviceSynchronize()
    got = cp.asnumpy(selected)
    expected = np.lexsort((np.arange(metrics_np.size), metrics_np))[:selected.size]
    if not np.array_equal(got, expected):
        raise RuntimeError("CuPy stable top-k tie rule mismatch")
    semantic = (f_got.astype("<f8").tobytes() + g_got.astype("<f8").tobytes() +
                got.astype("<i4").tobytes())
    return {
        "status": "PASS_SOURCE_FREE_PRIMITIVES_ONLY",
        "f_max_abs_error": float(np.max(np.abs(f_got - f_expected))),
        "g_max_abs_error": float(np.max(np.abs(g_got - g_expected))),
        "stable_topk_indices": got.astype(int).tolist(),
        "semantic_receipt_sha256": sha256_bytes(semantic),
        "production_persistent_kernel_demonstrated": False,
        "q0_16_frequency_rounding_equivalence_demonstrated": False,
        "primitive_float_semantics": (
            "allclose smoke only; CUDA contraction/division is not authenticated "
            "NumPy frequency-boundary equivalence"
        ),
    }


def allocation_smoke(cp: Any, block_values: int, beam: int) -> dict[str, Any]:
    plan = memory_plan(block_values, beam)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before_used, before_total = int(pool.used_bytes()), int(pool.total_bytes())
    allocations = []
    touch = cp.RawKernel(TOUCH_SOURCE, "touch_edges", options=("--std=c++11",))
    started = time.perf_counter()
    try:
        for ordinal, row in enumerate(plan["buffers"]):
            allocated = int(row["allocated_bytes"])
            array = cp.empty(allocated, dtype=cp.uint8)
            touch((1,), (1,), (array, allocated, (ordinal * 17 + beam) & 0xff))
            allocations.append(array)
        cp.cuda.runtime.deviceSynchronize()
        used = int(pool.used_bytes()) - before_used
        total = int(pool.total_bytes()) - before_total
        literal = sum(int(array.nbytes) for array in allocations)
        if literal != int(plan["aligned_peak_bytes"]):
            raise RuntimeError("literal allocation != frozen aligned ledger")
        if used < literal:
            raise RuntimeError("CuPy pool used bytes below live literal allocations")
        if used >= CAP_BYTES or total >= CAP_BYTES:
            raise RuntimeError("actual CuPy allocation breached 4 GiB cap")
        edge_receipt = bytes(
            [int(cp.asnumpy(array[:1])[0]) for array in allocations] +
            [int(cp.asnumpy(array[-1:])[0]) for array in allocations]
        )
        return {
            "beam_width": beam,
            "block_values": block_values,
            "logical_peak_bytes": int(plan["logical_peak_bytes"]),
            "aligned_peak_bytes": int(plan["aligned_peak_bytes"]),
            "literal_live_array_bytes": literal,
            "cupy_pool_used_delta_bytes": used,
            "cupy_pool_total_delta_bytes": total,
            "passes_4gib_actual_pool": used < CAP_BYTES and total < CAP_BYTES,
            "buffers_allocated_simultaneously": len(allocations),
            "edge_touch_sha256": sha256_bytes(edge_receipt),
            "seconds": time.perf_counter() - started,
        }
    finally:
        allocations.clear()
        pool.free_all_blocks()


def run(block_values: int, beams: list[int]) -> dict[str, Any]:
    import cupy as cp

    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", "strict")
    primitive = primitive_smoke(cp)
    rows = [allocation_smoke(cp, block_values, beam) for beam in beams]
    return {
        "schema": "epsilon-tcq-polar-cow-cupy-smoke-v2",
        "status": "PASS_GO_MEMORY_CAPACITY_ONLY_HOLD_COMPUTE_AND_DEVICE_COW",
        "device": {"id": int(cp.cuda.Device().id), "name": str(name),
                   "cupy_version": cp.__version__,
                   "runtime_version": int(cp.cuda.runtime.runtimeGetVersion())},
        "block_values": block_values,
        "beams": rows,
        "primitive_kernel": primitive,
        "qwen_payload_accessed": False,
        "current_codec_payload_accessed": False,
        "compute_gate": "HOLD_COMPUTE_AND_DEVICE_COW_IMPLEMENTATION",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-values", type=int, default=1 << 21)
    parser.add_argument("--beams", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--output")
    args = parser.parse_args()
    receipt = run(args.block_values, args.beams)
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
