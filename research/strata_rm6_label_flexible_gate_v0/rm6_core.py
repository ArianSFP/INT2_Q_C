#!/usr/bin/env python3
"""Exact RM-frozen-set primitives for six level-major STRATA bitplanes."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


ETA = 0.25
PLANES = 6
LOCAL_LOG2 = 12
LOCAL_N = 1 << LOCAL_LOG2
ALIGNMENT_BYTES = 128
PHYSICAL_CAP_BYTES = 1280
TARGET_MIN_BPW = 2.15
TARGET_MAX_BPW = 2.5

# Monotone LSB->MSB banks.  Each entry is six exact RM orders; lower-order RM
# is a strict subcode of RM(5,12), not an arbitrary truncated monomial prefix.
ORDER_BANK: dict[int, tuple[int, ...]] = {
    0: (5, 5, 5, 5, 5, 5),
    1: (4, 5, 5, 5, 5, 5),
    2: (4, 4, 5, 5, 5, 5),
    3: (4, 4, 4, 5, 5, 5),
    4: (3, 4, 4, 5, 5, 5),
    5: (4, 4, 4, 4, 5, 5),
    6: (4, 4, 4, 4, 4, 5),
    7: (4, 4, 4, 4, 4, 4),
}


class RM6Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RM6Error(message)


def align(value: int, alignment: int = ALIGNMENT_BYTES) -> int:
    require(type(value) is int and value >= 0 and alignment > 0 and
            not (alignment & (alignment - 1)), "alignment")
    return (value + alignment - 1) // alignment * alignment


def rm_dimension(order: int, variables: int) -> int:
    require(type(order) is int and type(variables) is int and
            0 <= order <= variables <= 31, "RM geometry")
    return sum(math.comb(variables, degree) for degree in range(order + 1))


def bit_reverse_indices(n: int) -> np.ndarray:
    depth = int(math.log2(n))
    require(n >= 2 and 1 << depth == n, "bit reverse geometry")
    source = np.arange(n, dtype=np.uint32)
    result = np.zeros(n, dtype=np.uint32)
    for _ in range(depth):
        result = (result << np.uint32(1)) | (source & np.uint32(1))
        source >>= np.uint32(1)
    return result.astype(np.int64)


def polar_transform(bits: Any) -> np.ndarray:
    result = np.asarray(bits, dtype=np.uint8).copy()
    require(result.ndim == 1 and result.size >= 2 and
            not (result.size & (result.size - 1)) and not np.any(result > 1),
            "polar bits")
    stride = 1
    while stride < result.size:
        rows = result.reshape(-1, 2 * stride)
        rows[:, :stride] ^= rows[:, stride:]
        stride *= 2
    return result


def rm_information_positions(variables: int, order: int) -> np.ndarray:
    """Internal SC phases selecting exact RM(order, variables) rows.

    The authenticated STRATA output is polar_transform(internal[bit_reverse]).
    A row has weight 2**popcount(phase), so RM(r,m) selects phases whose
    Hamming weight is at least m-r.  Bit reversal preserves popcount.
    """

    n = 1 << variables
    require(0 <= order <= variables, "RM information order")
    threshold = variables - order
    positions = np.fromiter((index for index in range(n)
                             if index.bit_count() >= threshold),
                            dtype=np.int32)
    require(positions.size == rm_dimension(order, variables), "RM dimension drift")
    return positions


def rm_freeze_flag(variables: int, order: int) -> np.ndarray:
    flag = np.ones(1 << variables, dtype=np.uint8)
    flag[rm_information_positions(variables, order)] = 0
    return flag


def frozen_external_from_seed(n: int, sc_seed: int, level: int) -> np.ndarray:
    require(0 <= sc_seed <= 0xFFFFFFFF and 1 <= level <= PLANES, "SC seed/level")
    rng = np.random.default_rng(sc_seed + 1_000_003 * level)
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def plane_from_information(info_bits: Any, variables: int, order: int,
                           frozen_external: Any) -> np.ndarray:
    """Complete one affine RM-coset plane with exact STRATA transform order."""

    n = 1 << variables
    positions = rm_information_positions(variables, order)
    info = np.asarray(info_bits, dtype=np.uint8)
    frozen = np.asarray(frozen_external, dtype=np.uint8)
    require(info.shape == (positions.size,) and frozen.shape == (n,) and
            not np.any(info > 1) and not np.any(frozen > 1), "plane information")
    reverse = bit_reverse_indices(n)
    internal = frozen[reverse].copy()
    internal[positions] = info
    return polar_transform(internal[reverse])


def generator_matrix(variables: int, order: int) -> np.ndarray:
    """Actual STRATA-domain generator rows for the selected internal phases."""

    n = 1 << variables
    positions = rm_information_positions(variables, order)
    reverse = bit_reverse_indices(n)
    matrix = np.zeros((positions.size, n), dtype=np.uint8)
    matrix[np.arange(positions.size), reverse[positions]] = 1
    stride = 1
    while stride < n:
        view = matrix.reshape(matrix.shape[0], -1, 2 * stride)
        view[:, :, :stride] ^= view[:, :, stride:]
        stride *= 2
    return matrix


def assemble_indices(planes: Sequence[Any]) -> np.ndarray:
    require(len(planes) == PLANES, "six completed planes")
    arrays = [np.asarray(plane, dtype=np.uint8) for plane in planes]
    n = arrays[0].size
    require(n > 0 and all(array.shape == (n,) and not np.any(array > 1)
                          for array in arrays), "plane geometry")
    indices = np.zeros(n, dtype=np.uint8)
    for level0, plane in enumerate(arrays):
        indices |= plane << np.uint8(level0)
    require(np.all(indices < 64), "64-way index range")
    return indices


def split_indices(indices: Any) -> tuple[np.ndarray, ...]:
    source = np.asarray(indices, dtype=np.uint8)
    require(source.ndim == 1 and np.all(source < 64), "index split")
    return tuple(((source >> np.uint8(level0)) & np.uint8(1)).astype(np.uint8)
                 for level0 in range(PLANES))


def fp16_from_bits(bits: int) -> float:
    require(0 <= int(bits) <= 0xFFFF, "FP16 bits")
    value = float(np.asarray([int(bits)], dtype="<u2").view("<f2")[0])
    require(math.isfinite(value) and value > 0.0, "positive finite decoder scale")
    return value


def reconstruction_levels(scale_fp16_bits: int) -> np.ndarray:
    scale = fp16_from_bits(scale_fp16_bits)
    return (ETA * np.arange(-31, 33, dtype=np.float64)) * scale


def exact_distortion_costs(target: Any, scale_fp16_bits: int) -> np.ndarray:
    """Literal float64 per-coordinate cost for every one of 64 indices."""

    source = np.asarray(target, dtype=np.float64)
    require(source.ndim == 1 and np.all(np.isfinite(source)), "distortion target")
    levels = reconstruction_levels(scale_fp16_bits)
    return (source[:, None] - levels[None, :]) ** 2


def selected_distortion(costs: Any, indices: Any) -> float:
    table = np.asarray(costs, dtype=np.float64)
    source = np.asarray(indices, dtype=np.uint8)
    require(table.shape == (source.size, 64) and np.all(source < 64),
            "selected distortion")
    return float(np.sum(table[np.arange(source.size), source], dtype=np.float64))


def bank_dimensions(bank_id: int, variables: int = LOCAL_LOG2) -> tuple[int, ...]:
    require(bank_id in ORDER_BANK, "order bank id")
    return tuple(rm_dimension(order, variables) for order in ORDER_BANK[bank_id])


def dimension_ledger(bank_id: int, *, header_bytes: int = 40,
                     crc_bytes: int = 4) -> dict[str, Any]:
    """Screen literal K information bits plus metadata; not arithmetic worst-case."""

    dimensions = bank_dimensions(bank_id)
    information_bits = sum(dimensions)
    raw_bytes = header_bytes + (information_bits + 7) // 8 + crc_bytes
    packet_bytes = align(raw_bytes)
    physical_bpw = packet_bytes * 8.0 / LOCAL_N
    return {"bank_id": bank_id, "orders": list(ORDER_BANK[bank_id]),
            "dimensions": list(dimensions), "information_bits": information_bits,
            "selected_information_bits": information_bits,
            "header_bytes": header_bytes, "crc_bytes": crc_bytes,
            "raw_dimension_ledger_bytes": raw_bytes,
            "alignment_bytes": ALIGNMENT_BYTES, "packet_bytes": packet_bytes,
            "physical_bpw": physical_bpw,
            "passes_2_5_bpw_dimension_ledger": packet_bytes <= PHYSICAL_CAP_BYTES,
            "dimension_screen_target_rate_eligible":
                TARGET_MIN_BPW <= physical_bpw <= TARGET_MAX_BPW,
            "dimension_screen_not_emitted_arithmetic_bits": True,
            "is_exact_rm_not_dimension_truncated": True,
            "arithmetic_codelength_is_data_dependent": True}
