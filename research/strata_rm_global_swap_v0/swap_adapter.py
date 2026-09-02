#!/usr/bin/env python3
"""Count-preserving hook installer for an authenticated current encoder.

This module never opens a source payload. A future independently audited
launcher must call :func:`install` at the reliability-hook boundary of the
current global STRATA encoder.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from rm_order import TARGET_N, swap_reference_flags


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def make_replacement(reference_bec_flags: Callable[..., list[np.ndarray]], *,
                     backend: str = "numpy", production_lengths_only: bool = True):
    require(callable(reference_bec_flags), "reference BEC hook")

    def rm_ordered_flags(repo: Any, n: int, capacities: Iterable[float]):
        if production_lengths_only:
            require(n in TARGET_N, "global swap permits only N=2**20 or 2**21")
        reference = reference_bec_flags(repo, n, list(capacities))
        require(len(reference) == 6, "reference must return six STRATA levels")
        swapped = swap_reference_flags(reference, backend=backend)
        if backend == "cupy":
            import cupy as cp
            swapped = [cp.asnumpy(row) for row in swapped]
        for old, new in zip(reference, swapped, strict=True):
            require(np.asarray(old).shape == np.asarray(new).shape, "flag shape")
            require(int(np.count_nonzero(np.asarray(old) == 0)) ==
                    int(np.count_nonzero(np.asarray(new) == 0)), "per-level K equality")
        return swapped

    rm_ordered_flags.__name__ = "rm_ordered_truncated_polar_flags"
    return rm_ordered_flags


def install(base_module: Any, reference_bec_flags: Callable[..., list[np.ndarray]], *,
            backend: str = "numpy") -> dict[str, Any]:
    require(hasattr(base_module, "reliability_freeze_flags"), "base reliability hook")
    replacement = make_replacement(reference_bec_flags, backend=backend,
                                   production_lengths_only=True)
    base_module.reliability_freeze_flags = replacement
    return {
        "installed": True,
        "hook": "reliability_freeze_flags",
        "candidate": "RM-ordered truncated polar",
        "retained_quantity": "exact per-block per-level selected count K",
        "block_lengths": list(TARGET_N),
        "coset": "current_random",
        "rate_truth": "actual literal full packet bytes only",
        "status": "HOLD_PAYLOAD_PENDING_INDEPENDENT_AUDIT",
    }

