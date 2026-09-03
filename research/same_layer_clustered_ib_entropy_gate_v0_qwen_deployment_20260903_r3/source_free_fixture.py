"""Balanced, deterministic payload-free fixture for CBIB-1 r3."""

from __future__ import annotations

import numpy as np
import hashlib


EXPERT_COUNT = 16
ROLES = 2
COORDINATES = 131_072
FOLD_COUNT = 8
SUPERBLOCK_VALUES = 2_048
SCALE_BYTES_PER_VALUE = 2
BLOCKS_PER_ROLE = COORDINATES // SUPERBLOCK_VALUES
SCALE_BYTES_PER_EXPERT = ROLES * BLOCKS_PER_ROLE * SCALE_BYTES_PER_VALUE
LATENT_PROBABILITY = 0.5
SIGN_FLIP_PROBABILITY = 0.105
BINARY_ENTROPY_SIGN_NOISE = 0.48464773973144537
SEED = 0xCB1B2026


def make_production_geometry_survivor_fixture(
    expert_count: int = EXPERT_COUNT,
    coordinates: int = COORDINATES,
    seed: int = SEED,
) -> np.ndarray:
    """Return paired balanced regimes with reflected role ownership.

    The balanced 0.5 latent removes the r2 underfill.  A frozen 0.105 sign-noise
    probability is the first deterministic grid point below the exact private
    11-page boundary after model and scale charges.  Role 1 is the exact
    expert-reflected copy of role 0, so hard-EM latent ownership cannot starve
    one member of a pair merely through its deterministic tie break.  The
    reflection is only a source-free mechanism fixture; the universal core
    never observes or assumes it.
    """
    if expert_count != EXPERT_COUNT or coordinates != COORDINATES:
        raise ValueError("frozen production-geometry fixture")
    rng = np.random.default_rng(seed)
    labels = np.empty((expert_count, ROLES, coordinates), dtype=np.uint8)
    for pair_start in range(0, expert_count, 2):
        latent = (rng.random(coordinates) < LATENT_PROBABILITY).astype(np.uint8)
        for local in range(2):
            sign = latent ^ (
                rng.random(coordinates) < SIGN_FLIP_PROBABILITY
            ).astype(np.uint8)
            magnitude = rng.integers(0, 2, size=coordinates, dtype=np.uint8)
            labels[pair_start + local, 0] = np.where(
                sign == 0, magnitude, 3 - magnitude
            ).astype(np.uint8)
        labels[pair_start, 1] = labels[pair_start + 1, 0]
        labels[pair_start + 1, 1] = labels[pair_start, 0]
    return labels


def make_quantizer_fixture(seed: int = 0xCB1B5090) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(16, 2_048)).astype(np.float32)


def fixture_sha256() -> str:
    return hashlib.sha256(
        make_production_geometry_survivor_fixture().tobytes(order="C")
    ).hexdigest()
