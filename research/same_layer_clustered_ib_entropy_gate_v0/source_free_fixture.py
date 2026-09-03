"""Deterministic synthetic inputs containing no model data or locator."""

from __future__ import annotations

import numpy as np


def make_clustered_nonmodal_fixture(expert_count: int = 16,
                                    coordinates: int = 4096,
                                    seed: int = 0xCB1B2026) -> np.ndarray:
    """Four true groups whose binary regimes are not modal label identities."""
    if expert_count != 16 or coordinates < 256:
        raise ValueError("frozen fixture geometry")
    rng = np.random.default_rng(seed)
    labels = np.empty((expert_count, 2, coordinates), dtype=np.uint8)
    offsets = np.asarray((0, 1, 2, 3), dtype=np.uint8)
    for role in range(2):
        for group_start in range(0, expert_count, 4):
            latent = rng.integers(0, 2, size=coordinates, dtype=np.uint8)
            # Each expert uses a different symbol convention.  A modal label is
            # deliberately uninformative while the arbitrary binary regime is strong.
            for local in range(4):
                base = (offsets[local] + latent * np.uint8(2) + np.uint8(role)) % 4
                corrupt = rng.random(coordinates) < 0.035
                noise = rng.integers(0, 4, size=coordinates, dtype=np.uint8)
                labels[group_start + local, role] = np.where(corrupt, noise, base)
    return labels


def make_gpu_model_fixture(seed: int = 0xC0FFEE) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = make_clustered_nonmodal_fixture(coordinates=2048, seed=seed)[:4, 0]
    assignments = rng.integers(0, 2, size=labels.shape[1], dtype=np.uint8)
    return labels, assignments

