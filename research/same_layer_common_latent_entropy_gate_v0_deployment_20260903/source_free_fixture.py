"""Deterministic synthetic fixtures; contains no model locator or model data."""

from __future__ import annotations

import numpy as np


def make_common_label_fixture(
    expert_count: int = 8,
    coordinates_per_role: int = 4096,
    seed: int = 0x51A7E,
) -> np.ndarray:
    if not 2 <= expert_count <= 256 or coordinates_per_role <= 0:
        raise ValueError("fixture geometry")
    rng = np.random.default_rng(seed)
    labels = np.empty((expert_count, 2, coordinates_per_role), dtype=np.uint8)
    for role in range(2):
        common = rng.integers(0, 4, size=coordinates_per_role, dtype=np.uint8)
        for expert in range(expert_count):
            noise = rng.random(coordinates_per_role) < (0.08 + 0.01 * (expert % 4))
            replacement = rng.integers(0, 4, size=coordinates_per_role, dtype=np.uint8)
            labels[expert, role] = np.where(noise, replacement, common)
    return labels


def binary_objective_counterexample() -> np.ndarray:
    """Planes minimizing conditional bits and charged MDL intentionally differ."""
    return np.asarray([
        [[2,0,1,2,0,2,2,1,1,3,2], [1,0,0,3,2,1,3,3,1,2,3]],
        [[2,3,3,0,2,1,1,3,2,0,2], [1,2,2,1,3,2,0,1,1,0,0]],
        [[1,0,3,1,2,0,0,2,2,1,1], [2,2,1,1,1,0,2,3,3,2,2]],
    ], dtype=np.uint8)


def make_quantizer_fixture(block_values: int = 32) -> np.ndarray:
    """Float32 blocks spanning scale midpoints and decision boundaries."""
    if block_values < 16:
        raise ValueError("block_values")
    rng = np.random.default_rng(0xB16F16)
    blocks = rng.normal(size=(4, block_values)).astype(np.float32)
    # Scale stress: values on both sides of adjacent binary16 representables.
    blocks[1] *= np.float32(np.nextafter(np.float16(0.75), np.float16(1.0)))
    blocks[2] *= np.float32(np.nextafter(np.float16(1.5), np.float16(0.0)))
    blocks[3, ::2] *= np.float32(1e-3)
    return blocks
