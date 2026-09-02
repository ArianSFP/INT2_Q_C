#!/usr/bin/env python3
"""Normative source-free RM ordering used by the hardened hook worker."""

from __future__ import annotations

from typing import Any


TARGET_N = (1 << 20, 1 << 21)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def log2_exact(n: int) -> int:
    require(isinstance(n, int) and n > 0 and n & (n - 1) == 0,
            "N must be a positive power of two")
    return n.bit_length() - 1


def phase_key(phase: int, n: int) -> tuple[int, int]:
    log2_exact(n)
    require(isinstance(phase, int) and 0 <= phase < n, "internal phase")
    return -phase.bit_count(), phase


def rm_full_order_numpy(n: int, np: Any):
    """Descending generator-row weight; ascending phase is the exact tie rule."""
    m = log2_exact(n)
    phases = np.arange(n, dtype=np.uint32)
    x = phases.copy()
    x = x - ((x >> np.uint32(1)) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> np.uint32(2)) & np.uint32(0x33333333))
    x = (x + (x >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    x = x + (x >> np.uint32(8))
    x = x + (x >> np.uint32(16))
    popcount = (x & np.uint32(0x3F)).astype(np.uint64)
    key = (np.uint64(m) - popcount) * np.uint64(n) + phases.astype(np.uint64)
    return np.argsort(key).astype(np.int64, copy=False)


def rm_full_order_cupy(n: int, cp: Any):
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


def replacement_from_authenticated_reference(reference_flags: list[Any], np: Any,
                                             *, backend: str = "numpy",
                                             cp: Any = None) -> list[Any]:
    require(len(reference_flags) == 6, "six current STRATA levels")
    rows = []
    for reference in reference_flags:
        row = np.asarray(reference, dtype=np.uint8)
        require(row.ndim == 1 and int(row.size) in TARGET_N and
                not np.any(row > 1), "authenticated reference flag")
        selected = int(np.count_nonzero(row == 0))
        if backend == "numpy":
            out = np.ones(row.size, dtype=np.uint8)
            out[rm_full_order_numpy(int(row.size), np)[:selected]] = 0
        elif backend == "cupy":
            require(cp is not None, "CuPy backend object")
            out = cp.ones(row.size, dtype=cp.uint8)
            out[rm_full_order_cupy(int(row.size), cp)[:selected]] = cp.uint8(0)
            out = cp.asnumpy(out)
        else:
            raise ValueError("backend")
        require(int(np.count_nonzero(out == 0)) == selected,
                "per-level selected count changed")
        rows.append(out)
    return rows

