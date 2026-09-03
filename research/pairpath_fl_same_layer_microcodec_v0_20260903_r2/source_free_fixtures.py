"""Deterministic synthetic sources; no model data, locator, or payload hash."""

from __future__ import annotations

import numpy as np

import pairpath_r2_core as core


def iid_fixture(coordinates: int = 32768, seed: int = 0x49494450) -> np.ndarray:
    if coordinates < core.FOLD_COUNT * core.BLOCK_VALUES or coordinates % core.BLOCK_VALUES:
        raise ValueError("fixture block geometry")
    rng = np.random.default_rng(seed)
    return rng.standard_normal((2, 3, coordinates), dtype=np.float64)


def aligned_fixture(coordinates: int = 16384, seed: int = 0x414C4947) -> np.ndarray:
    """Strong positive control for the joint-entropy oracle."""
    x = iid_fixture(coordinates, seed)
    x[1, 1:] = x[0, 1:]
    return x


def boundary_fixture(coordinates: int = 65536, seed: int = 0x424F554E) -> np.ndarray:
    """Near-boundary labels: flexible pair coding beats fixed pair coding."""
    x = iid_fixture(coordinates, seed)
    rng = np.random.default_rng(seed ^ 0xD1B54A32D192ED03)
    for role in core.OPTIMIZED_ROLES:
        sign = np.repeat(np.where(rng.random((1, coordinates)) < 0.5, -1.0, 1.0), 2, 0)
        magnitude = 0.9816 + 0.05 * np.where(
            rng.random((2, coordinates)) < 0.5, -1.0, 1.0)
        x[:, role] = sign * magnitude
    return x
