#!/usr/bin/env python3
"""POLARIS encoder entry point with deterministic BEC-surrogate construction."""

from __future__ import annotations

import numpy as np

import agent_polaris_qwen_rht_encoder as base


def bec_synthesized_z(capacity: float, n: int) -> np.ndarray:
    full = 1 << 31
    capacity_q31 = min(full, max(0, int(round(float(capacity) * full))))
    z = np.full(n, full - capacity_q31, dtype=np.uint64)
    step = 1
    while step < n:
        view = z.reshape(-1, 2 * step)
        left = view[:, :step].copy()
        right = view[:, step:].copy()
        product = (left * right + np.uint64(1 << 30)) >> np.uint64(31)
        view[:, :step] = left + right - product
        view[:, step:] = product
        step *= 2
    return z


def bec_flags(_repo, n: int, capacities):
    reverse = base.bit_reverse_indices(n)
    flags = []
    for capacity in capacities:
        keep = min(n, max(0, int(np.ceil(n * float(capacity)))))
        scores = bec_synthesized_z(float(capacity), n)
        order = np.lexsort((np.arange(n, dtype=np.int64), scores))
        external = np.ones(n, dtype=np.uint8)
        external[order[:keep]] = 0
        flags.append(external[reverse].copy())
    return flags


base.reliability_freeze_flags = bec_flags

if __name__ == "__main__":
    base.main()
