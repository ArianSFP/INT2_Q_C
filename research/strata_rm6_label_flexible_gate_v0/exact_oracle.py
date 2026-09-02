#!/usr/bin/env python3
"""Bounded exact joint six-plane oracle and favorable unconstrained bound."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from rm6_core import (PLANES, assemble_indices, bit_reverse_indices,
                      frozen_external_from_seed, plane_from_information, require,
                      rm_dimension, selected_distortion)


def unconstrained_64way_bound(costs: Any) -> dict[str, Any]:
    table = np.asarray(costs, dtype=np.float64)
    require(table.ndim == 2 and table.shape[1] == 64 and
            np.all(np.isfinite(table)), "unconstrained costs")
    indices = np.argmin(table, axis=1).astype(np.uint8)
    return {"indices": indices, "distortion": selected_distortion(table, indices),
            "legal_rm_codeword": False,
            "scope": "favorable coordinatewise lower bound only"}


def exact_joint_oracle(costs: Any, variables: int, orders: Sequence[int], *,
                       sc_seed: int, coset_mode: str,
                       maximum_information_bits: int = 18) -> dict[str, Any]:
    """Enumerate every joint legal affine-RM message for deliberately tiny N."""

    table = np.asarray(costs, dtype=np.float64)
    n = 1 << variables
    require(len(orders) == PLANES and table.shape == (n, 64) and
            coset_mode in ("zero", "current_random"), "exact oracle geometry")
    dimensions = tuple(rm_dimension(int(order), variables) for order in orders)
    total = sum(dimensions)
    require(total <= maximum_information_bits, "exact oracle enumeration cap")
    frozen_vectors = []
    for level0 in range(PLANES):
        if coset_mode == "zero":
            frozen_vectors.append(np.zeros(n, dtype=np.uint8))
        else:
            frozen_vectors.append(frozen_external_from_seed(n, sc_seed, level0 + 1))
    best_distortion, best_message, best_indices = float("inf"), -1, None
    for message in range(1 << total):
        cursor, planes = 0, []
        for level0, (order, dimension) in enumerate(zip(orders, dimensions, strict=True)):
            info = np.fromiter(((message >> (cursor + bit)) & 1
                                for bit in range(dimension)), dtype=np.uint8,
                               count=dimension)
            cursor += dimension
            planes.append(plane_from_information(info, variables, int(order),
                                                  frozen_vectors[level0]))
        indices = assemble_indices(planes)
        distortion = selected_distortion(table, indices)
        if distortion < best_distortion:
            best_distortion, best_message, best_indices = distortion, message, indices.copy()
    require(best_indices is not None, "exact oracle result")
    return {"variables": variables, "block_values": n, "orders": list(orders),
            "dimensions": list(dimensions), "information_bits": total,
            "candidate_messages": 1 << total, "best_message": best_message,
            "indices": best_indices, "distortion": best_distortion,
            "coset_mode": coset_mode, "legal_joint_rm_codeword": True,
            "production_rm5_12_decoder": False}
