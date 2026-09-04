"""Deterministic synthetic sources only; no payload or model access."""

from __future__ import annotations

import numpy as np

import pairpath_r3_core as core


def iid_fixture(coordinates: int = 16384, seed: int = 0x4949445033) -> np.ndarray:
    if coordinates < core.FOLD_COUNT * core.BLOCK_VALUES or coordinates % core.BLOCK_VALUES:
        raise ValueError("fixture block geometry")
    return np.random.default_rng(seed).standard_normal(
        (2, 3, coordinates), dtype=np.float64)


def aligned_fixture(coordinates: int = 16384, seed: int = 0x414C494733) -> np.ndarray:
    x = iid_fixture(coordinates, seed)
    x[1, 1:] = x[0, 1:]
    return x


def unequal_role_energy_fixture(coordinates: int = 16384) -> np.ndarray:
    x = iid_fixture(coordinates, 0x454E45524759)
    x[:, 1] *= 0.03125
    x[:, 2] *= 16.0
    return x


def adversarial_solver_fixture(coordinates: int = 16384) -> np.ndarray:
    """Asymmetric mixture intended to stress local alternating assignments."""
    rng = np.random.default_rng(0x41445645525345)
    x = rng.normal(0.0, 0.17, (2, 3, coordinates)).astype(np.float64)
    phase = np.arange(coordinates) % 11
    x[0, 1] += np.where(phase < 7, -0.72, 1.31)
    x[1, 1] += np.where((phase * 3) % 11 < 6, 0.48, -1.44)
    x[0, 2] += np.where(phase < 4, 1.63, -0.39)
    x[1, 2] += np.where((phase * 5) % 11 < 5, -1.11, 0.57)
    return x
