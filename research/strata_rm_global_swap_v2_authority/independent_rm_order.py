#!/usr/bin/env python3
"""Independent RM row-order constructions for N=2**20 and N=2**21.

The CPU implementation enumerates fixed-Hamming-weight integer combinations
with Gosper's successor and never uses the producer's popcount/sort formula.
The GPU implementation uses a byte lookup table.  Their only shared contract
is the mathematical order: descending row degree, ascending phase tie-break.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterator


TARGET_N = (1 << 20, 1 << 21)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def log2_exact(n: int) -> int:
    require(isinstance(n, int) and n > 0 and n & (n - 1) == 0,
            "N must be a positive power of two")
    return n.bit_length() - 1


def fixed_weight_integers(width: int, weight: int) -> Iterator[int]:
    """Yield width-bit integers of one Hamming weight in numeric order."""
    require(0 <= weight <= width, "fixed-weight domain")
    if weight == 0:
        yield 0
        return
    limit = 1 << width
    value = (1 << weight) - 1
    while value < limit:
        yield value
        low = value & -value
        ripple = value + low
        value = ripple | (((ripple ^ value) >> 2) // low)


def independent_cpu_order(n: int, np: Any):
    """Pure combinatorial RM order; no argsort and no bit-twiddle popcount."""
    width = log2_exact(n)
    output = np.empty(n, dtype=np.int64)
    cursor = 0
    for weight in range(width, -1, -1):
        for phase in fixed_weight_integers(width, weight):
            output[cursor] = phase
            cursor += 1
    require(cursor == n, "CPU order cardinality")
    return output


def independent_gpu_order(n: int, cp: Any):
    """GPU byte-LUT popcount order, independent of the producer formula."""
    width = log2_exact(n)
    phase = cp.arange(n, dtype=cp.uint32)
    table = cp.asarray([int(value).bit_count() for value in range(256)],
                       dtype=cp.uint8)
    byte0 = table[(phase & cp.uint32(255)).astype(cp.int32)]
    byte1 = table[((phase >> cp.uint32(8)) & cp.uint32(255)).astype(cp.int32)]
    byte2 = table[((phase >> cp.uint32(16)) & cp.uint32(255)).astype(cp.int32)]
    byte3 = table[((phase >> cp.uint32(24)) & cp.uint32(255)).astype(cp.int32)]
    population = (byte0.astype(cp.uint64) + byte1.astype(cp.uint64) +
                  byte2.astype(cp.uint64) + byte3.astype(cp.uint64))
    key = ((cp.uint64(width) - population) * cp.uint64(n) +
           phase.astype(cp.uint64))
    return cp.argsort(key).astype(cp.int64, copy=False)


def little_i64_sha256(values: Any) -> str:
    payload = values.astype("<i8", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()


def validate_small_orders(np: Any) -> None:
    for width in range(1, 13):
        n = 1 << width
        order = independent_cpu_order(n, np)
        expected = sorted(range(n), key=lambda phase: (-phase.bit_count(), phase))
        require(order.tolist() == expected, f"independent CPU semantics N={n}")
