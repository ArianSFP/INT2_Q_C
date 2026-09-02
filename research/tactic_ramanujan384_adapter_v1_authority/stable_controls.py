#!/usr/bin/env python3
"""Frozen integer-PRNG controls, generated once on host and copied byte-exactly."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


MASK64 = (1 << 64) - 1
UINT16_DENOMINATOR = 65536.0
CLT_TERMS = 12


class ControlError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlError(message)


def splitmix64_words(seed: int, count: int) -> np.ndarray:
    """Reference SplitMix64 using specified unsigned-64 modular arithmetic."""

    require(type(seed) is int and 0 <= seed <= MASK64, "uint64 seed")
    require(type(count) is int and count >= 0, "word count")
    output = np.empty(count, dtype="<u8")
    state = seed
    for index in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        output[index] = (value ^ (value >> 31)) & MASK64
    return output


def frozen_normal_like(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Deterministic bell-shaped binary64 control without a backend RNG/libm.

    Twelve fixed 16-bit uniforms are summed and centred.  This Irwin--Hall
    approximation has exact integer provenance; it is deliberately called a
    normal-like control rather than claiming an implementation-defined normal
    sampler.
    """

    require(len(shape) == 2 and all(type(value) is int and value > 0 for value in shape),
            "control shape")
    values = shape[0] * shape[1]
    words = splitmix64_words(seed, values * 3)
    lanes = np.empty((values, CLT_TERMS), dtype=np.uint16)
    for lane in range(CLT_TERMS):
        word = words[(lane // 4) * values:(lane // 4 + 1) * values]
        lanes[:, lane] = ((word >> (16 * (lane % 4))) & 0xFFFF).astype(np.uint16)
    centred_integer = lanes.astype(np.int64).sum(axis=1) - CLT_TERMS * 32767
    return np.ascontiguousarray(
        (centred_integer.astype(np.float64) / UINT16_DENOMINATOR).reshape(shape),
        dtype="<f8",
    )


def moment_matched_blocks(xp: Any, reference: Any, seed: int,
                          valid_counts: tuple[int, ...] | None = None) -> Any:
    """Match each block's FP64 mean and centred energy on the host exactly once."""

    host_reference = (
        np.asarray(xp.asnumpy(reference), dtype="<f8")
        if hasattr(xp, "asnumpy") else np.asarray(reference, dtype="<f8")
    )
    require(host_reference.ndim == 2 and host_reference.shape[1] > 1,
            "reference block geometry")
    raw = frozen_normal_like(tuple(host_reference.shape), seed)
    output = np.zeros_like(raw)
    if valid_counts is None:
        valid_counts = (host_reference.shape[1],) * host_reference.shape[0]
    require(len(valid_counts) == host_reference.shape[0]
            and all(type(value) is int and 1 < value <= host_reference.shape[1]
                    for value in valid_counts), "valid control counts")
    for block in range(raw.shape[0]):
        valid = valid_counts[block]
        reference_row = host_reference[block, :valid]
        mean = float(np.sum(reference_row, dtype=np.float64) / reference_row.size)
        centred = reference_row - mean
        energy = float(np.sum(centred * centred, dtype=np.float64))
        row = raw[block, :valid]
        row_mean = float(np.sum(row, dtype=np.float64) / row.size)
        row_centred = row - row_mean
        row_energy = float(np.sum(row_centred * row_centred, dtype=np.float64))
        require(math.isfinite(energy) and math.isfinite(row_energy) and row_energy > 0.0,
                "finite control moments")
        output[block, :valid] = mean + row_centred * math.sqrt(energy / row_energy)
    output = np.ascontiguousarray(output, dtype="<f8")
    return xp.asarray(output, dtype=xp.float64)


def host_bytes(xp: Any, values: Any) -> bytes:
    host = xp.asnumpy(values) if hasattr(xp, "asnumpy") else values
    return np.ascontiguousarray(host, dtype="<f8").tobytes(order="C")
