#!/usr/bin/env python3
"""Bounded block-level polar list oracle and production resource gate."""

from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Sequence


class OracleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def polar_transform(bits: Sequence[int]) -> tuple[int, ...]:
    result = [int(value) for value in bits]
    require(result and not (len(result) & (len(result) - 1)) and
            all(value in (0, 1) for value in result), "polar bits")
    stride = 1
    while stride < len(result):
        for base in range(0, len(result), 2 * stride):
            for offset in range(stride):
                result[base + offset] ^= result[base + stride + offset]
        stride *= 2
    return tuple(result)


def tiny_six_level_oracle(
    target_indices: Sequence[int],
    information_masks: Sequence[Sequence[int]],
    frozen_internal_bits: Sequence[Sequence[int]],
    *,
    maximum_information_bits: int = 12,
) -> dict[str, Any]:
    """Exact source-free oracle over legal block-level polar decisions.

    This is intentionally tiny.  It demonstrates the legal search unit and
    coordinate coupling; it is not a production implementation.
    """

    n = len(target_indices)
    require(n >= 2 and not (n & (n - 1)) and
            all(type(value) is int and 0 <= value < 64 for value in target_indices),
            "tiny target indices")
    require(len(information_masks) == len(frozen_internal_bits) == 6,
            "six polar levels")
    locations = []
    for level, (mask, frozen) in enumerate(zip(
            information_masks, frozen_internal_bits, strict=True)):
        require(len(mask) == len(frozen) == n and
                all(value in (0, 1) for value in mask) and
                all(value in (0, 1) for value in frozen), "level specification")
        locations.extend((level, position) for position, value in enumerate(mask) if value)
    require(len(locations) <= maximum_information_bits, "tiny exhaustive decision cap")
    winner = None
    candidates = 0
    for decisions in itertools.product((0, 1), repeat=len(locations)):
        internal = [list(row) for row in frozen_internal_bits]
        for (level, position), bit in zip(locations, decisions, strict=True):
            internal[level][position] = bit
        planes = [polar_transform(row) for row in internal]
        indices = tuple(sum(planes[level][position] << level for level in range(6))
                        for position in range(n))
        distortion = sum((left - right) ** 2
                         for left, right in zip(indices, target_indices, strict=True))
        row = (distortion, decisions, indices)
        if winner is None or row < winner:
            winner = row
        candidates += 1
    require(winner is not None, "nonempty tiny oracle")
    return {
        "status": "PASS_SOURCE_FREE_TINY_BLOCK_ORACLE",
        "block_values": n,
        "information_bits": len(locations),
        "candidate_blocks": candidates,
        "minimum_index_squared_error": winner[0],
        "selected_information_bits": list(winner[1]),
        "selected_indices": list(winner[2]),
        "search_unit": "whole_six_level_polar_block",
        "coordinate_local_search": False,
    }


def resource_estimate(block_values: int, beam_width: int) -> dict[str, int]:
    require(type(block_values) is int and block_values >= 2 and
            not (block_values & (block_values - 1)), "resource polar length")
    require(type(beam_width) is int and beam_width >= 1, "resource beam")
    depth = int(math.log2(block_values))
    # Exact v1 bound for a straightforward resumable implementation:
    # LR register f64, partial-sum register u8, six completed planes u8,
    # previous index i16, arithmetic/WFA state, and u32 backpointers.
    lr_per_path = (block_values // 2) * depth * 8
    mu_per_path = (block_values // 2) * depth
    planes_per_path = 6 * block_values
    previous_per_path = 2 * block_values
    scalar_per_path = 256
    path_state = lr_per_path + mu_per_path + planes_per_path + previous_per_path + scalar_per_path
    backpointers = 4 * block_values * beam_width
    frontier = path_state * beam_width
    return {
        "block_values": block_values,
        "depth": depth,
        "beam_width": beam_width,
        "bytes_per_path": path_state,
        "frontier_bytes": frontier,
        "backpointer_bytes": backpointers,
        "total_peak_bytes_lower_bound": frontier + backpointers,
    }


def production_gate(
    block_values: int,
    beam_width: int,
    *,
    memory_cap_bytes: int,
    cupy_topk_wired: bool,
    device_resident_polar_state: bool,
    bounded_prefix_storage: bool,
) -> dict[str, Any]:
    estimate = resource_estimate(block_values, beam_width)
    require(type(memory_cap_bytes) is int and memory_cap_bytes > 0,
            "memory cap")
    checks = {
        "peak_memory_within_cap": estimate["total_peak_bytes_lower_bound"] <= memory_cap_bytes,
        "cupy_topk_wired": cupy_topk_wired is True,
        "device_resident_polar_state": device_resident_polar_state is True,
        "bounded_prefix_storage": bounded_prefix_storage is True,
    }
    passed = all(checks.values())
    return {
        "status": ("ELIGIBLE_FOR_SEPARATE_BLOCK_LIST_SOURCE_AUDIT" if passed else
                   "HOLD_PRODUCTION_POLAR_LIST_SCALABILITY"),
        "estimate": estimate,
        "memory_cap_bytes": memory_cap_bytes,
        "checks": checks,
        "qwen_payload_may_open": False,
    }


def cupy_topk(costs: Sequence[float], count: int, backend: Any) -> tuple[int, ...]:
    require(type(count) is int and 0 < count <= len(costs), "top-k count")
    require(getattr(backend, "is_cupy", False) is True and
            getattr(backend, "device_resident", False) is True,
            "CuPy device-resident backend required")
    indices = tuple(int(value) for value in backend.topk(costs, count))
    require(len(indices) == count and len(set(indices)) == count and
            all(0 <= value < len(costs) for value in indices), "CuPy top-k result")
    expected = tuple(sorted(range(len(costs)), key=lambda index: (float(costs[index]), index))[:count])
    require(indices == expected, "CuPy top-k deterministic dominance")
    return indices
