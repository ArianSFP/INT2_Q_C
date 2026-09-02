#!/usr/bin/env python3
"""Fail-closed distinction between current and zero frozen-value cosets."""

from __future__ import annotations

import numpy as np


class HeldCosetFork(RuntimeError):
    pass


CURRENT_RANDOM = "current_random"
ZERO = "zero"


def current_random_frozen_external(n: int, sc_seed: int, level: int) -> np.ndarray:
    if n <= 0 or n & (n - 1) or not (0 <= sc_seed <= 0xFFFFFFFF) or not (1 <= level <= 6):
        raise ValueError("current-random frozen-value parameters")
    rng = np.random.default_rng(sc_seed + 1_000_003 * level)
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def frozen_external(n: int, sc_seed: int, level: int, mode: str) -> np.ndarray:
    if mode == CURRENT_RANDOM:
        return current_random_frozen_external(n, sc_seed, level)
    if mode == ZERO:
        raise HeldCosetFork(
            "zero frozen values require a separately versioned, charged packet fork; "
            "the pinned current upstream API exposes no coset selector"
        )
    raise ValueError("unknown coset mode")


def describe() -> dict[str, object]:
    return {
        "implemented": CURRENT_RANDOM,
        "held": ZERO,
        "held_status": "HOLD_SEPARATE_FORMAT_FORK_NOT_IMPLEMENTED",
        "pool_results": False,
    }

