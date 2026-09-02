#!/usr/bin/env python3
"""Pure ordering primitives for the count-preserving global STRATA swap."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


TARGET_N = (1 << 20, 1 << 21)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def log2_exact(n: int) -> int:
    require(isinstance(n, int) and n > 0 and n & (n - 1) == 0,
            "N must be a positive power of two")
    return n.bit_length() - 1


def bit_reverse_indices(n: int) -> np.ndarray:
    m = log2_exact(n)
    source = np.arange(n, dtype=np.uint32)
    work = source.copy()
    reverse = np.zeros(n, dtype=np.uint32)
    for _ in range(m):
        reverse = (reverse << np.uint32(1)) | (work & np.uint32(1))
        work >>= np.uint32(1)
    return reverse.astype(np.int64)


def polar_transform(bits: Any) -> np.ndarray:
    out = np.asarray(bits, dtype=np.uint8).copy()
    n = int(out.size)
    log2_exact(n)
    require(out.ndim == 1 and not np.any(out > 1), "binary polar input")
    step = 1
    while step < n:
        view = out.reshape(-1, 2 * step)
        view[:, :step] ^= view[:, step:]
        step *= 2
    return out


def generated_row(internal_phase: int, n: int) -> np.ndarray:
    require(0 <= internal_phase < n, "internal phase")
    internal = np.zeros(n, dtype=np.uint8)
    internal[internal_phase] = 1
    return polar_transform(internal[bit_reverse_indices(n)])


def generated_row_weight(internal_phase: int, n: int) -> int:
    return int(np.count_nonzero(generated_row(internal_phase, n)))


def _numpy_popcount_u32(indices: np.ndarray) -> np.ndarray:
    x = np.asarray(indices, dtype=np.uint32).copy()
    x = x - ((x >> np.uint32(1)) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> np.uint32(2)) & np.uint32(0x33333333))
    x = (x + (x >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    x = x + (x >> np.uint32(8))
    x = x + (x >> np.uint32(16))
    return (x & np.uint32(0x3F)).astype(np.uint8)


def rm_full_order_numpy(n: int) -> np.ndarray:
    """Normative order: descending row weight, ascending phase tie break."""

    m = log2_exact(n)
    phases = np.arange(n, dtype=np.uint32)
    popcount = _numpy_popcount_u32(phases).astype(np.uint64)
    # The composite key makes the tie rule explicit and does not rely on sort
    # stability: high popcount has a lower bucket, then phase breaks the tie.
    key = (np.uint64(m) - popcount) * np.uint64(n) + phases.astype(np.uint64)
    return np.argsort(key).astype(np.int64, copy=False)


def rm_full_order_cupy(n: int):
    """CuPy construction with the same integer composite key as NumPy."""

    import cupy as cp

    m = log2_exact(n)
    phases = cp.arange(n, dtype=cp.uint32)
    x = phases.copy()
    x = x - ((x >> cp.uint32(1)) & cp.uint32(0x55555555))
    x = (x & cp.uint32(0x33333333)) + ((x >> cp.uint32(2)) & cp.uint32(0x33333333))
    x = (x + (x >> cp.uint32(4))) & cp.uint32(0x0F0F0F0F)
    x = x + (x >> cp.uint32(8))
    x = x + (x >> cp.uint32(16))
    popcount = (x & cp.uint32(0x3F)).astype(cp.uint64)
    key = (cp.uint64(m) - popcount) * cp.uint64(n) + phases.astype(cp.uint64)
    return cp.argsort(key).astype(cp.int64, copy=False)


def rm_information_positions(n: int, selected: int, *, backend: str = "numpy"):
    require(0 <= selected <= n, "selected count")
    if backend == "numpy":
        return rm_full_order_numpy(n)[:selected].copy()
    if backend == "cupy":
        return rm_full_order_cupy(n)[:selected].copy()
    raise ValueError("backend must be numpy or cupy")


def rm_dimension(order: int, variables: int) -> int:
    require(0 <= order <= variables, "RM order")
    return sum(math.comb(variables, degree) for degree in range(order + 1))


def classify_selected_count(n: int, selected: int) -> dict[str, Any]:
    m = log2_exact(n)
    require(0 <= selected <= n, "selected count")
    exact = next((r for r in range(m + 1) if rm_dimension(r, m) == selected), None)
    return {
        "n": n,
        "variables": m,
        "selected": selected,
        "exact_rm": exact is not None,
        "exact_rm_order": exact,
        "name": f"RM({exact},{m})" if exact is not None else "RM-ordered truncated polar",
    }


def swap_one_reference_flag(reference_flag: Any, *, backend: str = "numpy"):
    reference = np.asarray(reference_flag, dtype=np.uint8)
    require(reference.ndim == 1 and reference.size > 0 and
            reference.size & (reference.size - 1) == 0 and
            not np.any(reference > 1), "binary reference flag")
    n = int(reference.size)
    selected = int(np.count_nonzero(reference == 0))
    if backend == "numpy":
        result = np.ones(n, dtype=np.uint8)
        result[rm_information_positions(n, selected, backend="numpy")] = 0
        require(int(np.count_nonzero(result == 0)) == selected, "selected-count equality")
        return result
    if backend == "cupy":
        import cupy as cp

        result = cp.ones(n, dtype=cp.uint8)
        result[rm_information_positions(n, selected, backend="cupy")] = cp.uint8(0)
        require(int(cp.count_nonzero(result == 0).get()) == selected,
                "selected-count equality")
        return result
    raise ValueError("backend must be numpy or cupy")


def swap_reference_flags(reference_flags: Iterable[Any], *, backend: str = "numpy"):
    rows = list(reference_flags)
    require(len(rows) == 6, "STRATA requires six level-major flags")
    swapped = [swap_one_reference_flag(row, backend=backend) for row in rows]
    for old, new in zip(rows, swapped, strict=True):
        old_count = int(np.count_nonzero(np.asarray(old) == 0))
        if backend == "cupy":
            import cupy as cp
            new_count = int(cp.count_nonzero(new == 0).get())
        else:
            new_count = int(np.count_nonzero(new == 0))
        require(old_count == new_count, "per-level K changed")
    return swapped

