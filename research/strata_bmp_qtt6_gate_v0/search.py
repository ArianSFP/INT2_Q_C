#!/usr/bin/env python3
"""Bounded label-flexible D+lambda*R searches for BMP, ROBDD and GF(2)-QTT."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Callable

import numpy as np

from codec import (
    FAMILY_BMP, FAMILY_OBDD, FAMILY_QTT, MAX_BMP_RANK, MAX_EXCEPTIONS,
    MAX_OBDD_NODES, MAX_QTT_RANK, CodecError, Geometry, active_features,
    base_planes, build_robdd, decode_packet, encode_packet, indices_to_planes,
    planes_to_indices, qtt_core_bit_count, qtt_plane,
    validate_distortion_table,
)


MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_FAMILY_CANDIDATES = 32
MAX_BMP_ALTERNATIONS = 8
MAX_QTT_TOGGLES_PER_PLANE = 8
MAX_SEARCH_EVALUATIONS = 1_000_000
BMP_RANK_BANK = (0, 1, 2, 4)
QTT_RANK_BANK = (1, 2)
ORDER_BANK = (0, 1, 2, 3)


@dataclass
class Counter:
    evaluations: int = 0

    def add(self, count: int = 1) -> None:
        self.evaluations += count
        if self.evaluations > MAX_SEARCH_EVALUATIONS:
            raise CodecError("search evaluation cap")


def exact_sse(distortion: np.ndarray, indices: np.ndarray) -> float:
    return float(np.asarray(distortion, dtype=np.float64)[
        np.arange(indices.size), indices.astype(np.int64)].sum(dtype=np.float64))


def objective(distortion: np.ndarray, packet: bytes, lambda_bit: float) -> tuple[float, dict]:
    decoded = decode_packet(packet, allow_small=True)
    sse = exact_sse(distortion, decoded["indices"])
    return sse + lambda_bit * decoded["physical_bits"], {
        "sse": sse,
        "physical_bits": decoded["physical_bits"],
        "objective": sse + lambda_bit * decoded["physical_bits"],
    }


def conditional_plane_costs(distortion: np.ndarray, indices: np.ndarray,
                            level: int) -> tuple[np.ndarray, np.ndarray]:
    clear = indices.astype(np.uint8) & np.uint8(63 ^ (1 << level))
    set_one = clear | np.uint8(1 << level)
    rows = np.arange(indices.size)
    return distortion[rows, clear], distortion[rows, set_one]


def add_joint_exceptions(distortion: np.ndarray, base: np.ndarray,
                         lambda_bit: float, cap: int = MAX_EXCEPTIONS
                         ) -> list[tuple[int, int]]:
    nearest = np.argmin(distortion, axis=1).astype(np.uint8)
    rows = np.arange(base.size)
    gain = distortion[rows, base] - distortion[rows, nearest]
    candidates = [int(i) for i in np.argsort(-gain, kind="stable")
                  if gain[i] > 24.0 * lambda_bit and nearest[i] != base[i]]
    chosen = sorted(candidates[:cap])
    return [(position, int(nearest[position])) for position in chosen]


def _fit_rank_one(cost0: np.ndarray, cost1: np.ndarray, base: np.ndarray,
                  counter: Counter) -> tuple[np.ndarray, np.ndarray, float]:
    nr, nc = base.shape
    preferred = (cost1 < cost0).astype(np.uint8)
    seeds = [preferred[row].copy() for row in
             np.linspace(0, nr - 1, min(nr, 8), dtype=int)]
    columns = np.arange(nc, dtype=np.uint32)
    seeds.extend([((columns >> bit) & 1).astype(np.uint8)
                  for bit in range(min(8, int(np.log2(nc))))])
    best = None
    for seed in seeds:
        v = seed.copy()
        u = np.zeros(nr, dtype=np.uint8)
        for _ in range(MAX_BMP_ALTERNATIONS):
            previous_u = u.copy()
            previous_v = v.copy()
            for row in range(nr):
                out0 = base[row]
                out1 = base[row] ^ v
                score0 = np.where(out0, cost1[row], cost0[row]).sum()
                score1 = np.where(out1, cost1[row], cost0[row]).sum()
                u[row] = score1 < score0
            for col in range(nc):
                out0 = base[:, col]
                out1 = base[:, col] ^ u
                score0 = np.where(out0, cost1[:, col], cost0[:, col]).sum()
                score1 = np.where(out1, cost1[:, col], cost0[:, col]).sum()
                v[col] = score1 < score0
            counter.add(2 * (nr + nc))
            if np.array_equal(u, previous_u) and np.array_equal(v, previous_v):
                break
        output = base ^ np.outer(u, v).astype(np.uint8)
        score = float(np.where(output, cost1, cost0).sum(dtype=np.float64))
        if best is None or score < best[2]:
            best = (u.copy(), v.copy(), score)
    assert best is not None
    return best


def fit_bmp(cost0: np.ndarray, cost1: np.ndarray, nr: int, nc: int,
            rank: int, counter: Counter) -> tuple[np.ndarray, np.ndarray]:
    require_shape = cost0.shape == cost1.shape == (nr * nc,)
    if not require_shape:
        raise CodecError("BMP cost geometry")
    if not 0 <= rank <= MAX_BMP_RANK:
        raise CodecError("BMP rank bank")
    if rank == 0:
        return (np.zeros((nr, 0), dtype=np.uint8),
                np.zeros((nc, 0), dtype=np.uint8))
    c0 = cost0.reshape(nr, nc)
    c1 = cost1.reshape(nr, nc)
    U = np.zeros((nr, rank), dtype=np.uint8)
    V = np.zeros((nc, rank), dtype=np.uint8)
    base = np.zeros((nr, nc), dtype=np.uint8)
    for component in range(rank):
        u, v, _ = _fit_rank_one(c0, c1, base, counter)
        U[:, component] = u
        V[:, component] = v
        base ^= np.outer(u, v).astype(np.uint8)
    return U, V


def search_bmp(distortion: np.ndarray, geometry: Geometry, lambda_bit: float,
               counter: Counter) -> list[dict]:
    nearest = np.argmin(distortion, axis=1).astype(np.uint8)
    candidates = []
    for rank in BMP_RANK_BANK:
        indices = nearest.copy()
        factors = []
        for level in range(6):
            c0, c1 = conditional_plane_costs(distortion, indices, level)
            U, V = fit_bmp(c0, c1, geometry.row_count, geometry.col_count,
                           rank, counter)
            factors.append((U, V))
            plane = ((U.astype(np.uint16) @ V.astype(np.uint16).T) & 1).astype(
                np.uint8).reshape(-1) if rank else np.zeros(geometry.count,
                                                             dtype=np.uint8)
            indices = (indices & np.uint8(63 ^ (1 << level))) | (plane << level)
        model = {"ranks": [rank] * 6, "factors": factors}
        base = planes_to_indices(base_planes(FAMILY_BMP, model, geometry, 0))
        exceptions = add_joint_exceptions(distortion, base, lambda_bit)
        packet = encode_packet(FAMILY_BMP, 0, geometry, model, exceptions)
        score, metrics = objective(distortion, packet, lambda_bit)
        candidates.append({"family": "GF2_MATRIX_FACTOR", "rank": rank,
                           "order_id": 0, "packet": packet,
                           "score": score, **metrics})
    return candidates


def _tree_plane(cost0: np.ndarray, cost1: np.ndarray,
                features: np.ndarray, lambda_bit: float,
                counter: Counter) -> np.ndarray:
    """Exact DP for the pruned ordered-tree subset, then ROBDD reduction."""
    output = np.zeros(cost0.size, dtype=np.uint8)

    def solve(indices: np.ndarray, depth: int) -> tuple[float, np.ndarray]:
        counter.add()
        zero = float(cost0[indices].sum(dtype=np.float64))
        one = float(cost1[indices].sum(dtype=np.float64))
        if depth == features.shape[1]:
            bit = np.uint8(one < zero)
            return min(zero, one), np.full(indices.size, bit, dtype=np.uint8)
        mask = features[indices, depth] == 0
        low_cost, low_bits = solve(indices[mask], depth + 1)
        high_cost, high_bits = solve(indices[~mask], depth + 1)
        split = low_cost + high_cost + 40.0 * lambda_bit
        if zero <= one and zero <= split:
            return zero, np.zeros(indices.size, dtype=np.uint8)
        if one <= split:
            return one, np.ones(indices.size, dtype=np.uint8)
        result = np.empty(indices.size, dtype=np.uint8)
        result[mask] = low_bits
        result[~mask] = high_bits
        return split, result

    all_indices = np.arange(cost0.size, dtype=np.int32)
    _, bits = solve(all_indices, 0)
    output[all_indices] = bits
    return output


def search_obdd(distortion: np.ndarray, geometry: Geometry, lambda_bit: float,
                counter: Counter) -> list[dict]:
    nearest = np.argmin(distortion, axis=1).astype(np.uint8)
    candidates = []
    for order_id in ORDER_BANK:
        _, features = active_features(geometry, order_id)
        indices = nearest.copy()
        roots = []
        diagrams = []
        for level in range(6):
            c0, c1 = conditional_plane_costs(distortion, indices, level)
            bits = _tree_plane(c0, c1, features, lambda_bit, counter)
            root, nodes = build_robdd(bits, features)
            roots.append(root)
            diagrams.append(nodes)
            indices = (indices & np.uint8(63 ^ (1 << level))) | (bits << level)
        if sum(map(len, diagrams)) > MAX_OBDD_NODES:
            continue
        model = {"roots": roots, "nodes": diagrams}
        base = planes_to_indices(base_planes(FAMILY_OBDD, model, geometry,
                                             order_id))
        exceptions = add_joint_exceptions(distortion, base, lambda_bit)
        packet = encode_packet(FAMILY_OBDD, order_id, geometry, model, exceptions)
        score, metrics = objective(distortion, packet, lambda_bit)
        candidates.append({"family": "ROBDD", "order_id": order_id,
                           "nodes": sum(map(len, diagrams)), "packet": packet,
                           "score": score, **metrics})
    return candidates


def _qtt_constant(d: int, rank: int, value: int) -> np.ndarray:
    count = qtt_core_bit_count(d, rank)
    bits = np.zeros(count, dtype=np.uint8)
    # A single all-zero-state path evaluates to one for every coordinate.
    offset = 0
    shapes = ([(1, 2, 1)] if d == 1 else
              [(1, 2, rank)] + [(rank, 2, rank)] * (d - 2) +
              [(rank, 2, 1)])
    if value:
        for shape in shapes:
            core = bits[offset:offset + int(np.prod(shape))].reshape(shape)
            core[0, :, 0] = 1
            offset += core.size
    return bits


def _fit_qtt_plane(cost0: np.ndarray, cost1: np.ndarray,
                   features: np.ndarray, rank: int,
                   counter: Counter) -> np.ndarray:
    candidates = [_qtt_constant(features.shape[1], rank, 0),
                  _qtt_constant(features.shape[1], rank, 1)]
    best_bits = None
    best_score = float("inf")
    for initial in candidates:
        bits = initial.copy()
        plane = qtt_plane(bits, features, rank)
        score = float(np.where(plane, cost1, cost0).sum(dtype=np.float64))
        for _ in range(MAX_QTT_TOGGLES_PER_PLANE):
            improvement = 0.0
            selected = None
            selected_plane = None
            for index in range(bits.size):
                proposal = bits.copy()
                proposal[index] ^= 1
                candidate_plane = qtt_plane(proposal, features, rank)
                candidate_score = float(np.where(candidate_plane, cost1, cost0).sum(
                    dtype=np.float64))
                counter.add()
                gain = score - candidate_score
                if gain > improvement:
                    improvement = gain
                    selected = index
                    selected_plane = candidate_plane
            if selected is None:
                break
            bits[selected] ^= 1
            plane = selected_plane
            score -= improvement
        if score < best_score:
            best_score = score
            best_bits = bits.copy()
    assert best_bits is not None
    return best_bits


def search_qtt(distortion: np.ndarray, geometry: Geometry, lambda_bit: float,
               counter: Counter) -> list[dict]:
    nearest = np.argmin(distortion, axis=1).astype(np.uint8)
    candidates = []
    for order_id in ORDER_BANK:
        _, features = active_features(geometry, order_id)
        for rank in QTT_RANK_BANK:
            indices = nearest.copy()
            cores = []
            for level in range(6):
                c0, c1 = conditional_plane_costs(distortion, indices, level)
                bits = _fit_qtt_plane(c0, c1, features, rank, counter)
                cores.append(bits)
                plane = qtt_plane(bits, features, rank)
                indices = ((indices & np.uint8(63 ^ (1 << level))) |
                           (plane << level))
            model = {"ranks": [rank] * 6, "cores": cores}
            base = planes_to_indices(base_planes(FAMILY_QTT, model, geometry,
                                                 order_id))
            exceptions = add_joint_exceptions(distortion, base, lambda_bit)
            packet = encode_packet(FAMILY_QTT, order_id, geometry, model,
                                   exceptions)
            score, metrics = objective(distortion, packet, lambda_bit)
            candidates.append({"family": "BMP_QTT_GF2", "rank": rank,
                               "order_id": order_id, "packet": packet,
                               "score": score, **metrics})
    return candidates


def search_bank(distortion: np.ndarray, geometry: Geometry,
                lambda_bit: float) -> dict:
    geometry.validate()
    table = validate_distortion_table(distortion, geometry.count)
    if table.nbytes + geometry.count * 64 > MAX_WORKSPACE_BYTES:
        raise CodecError("workspace cap")
    if not np.isfinite(lambda_bit) or lambda_bit < 0:
        raise CodecError("lambda")
    counter = Counter()
    candidates = []
    candidates.extend(search_bmp(table, geometry, lambda_bit, counter))
    candidates.extend(search_obdd(table, geometry, lambda_bit, counter))
    candidates.extend(search_qtt(table, geometry, lambda_bit, counter))
    if not candidates or len(candidates) > MAX_FAMILY_CANDIDATES:
        raise CodecError("candidate bank closure")
    winner = min(candidates, key=lambda row: (row["score"], row["packet"]))
    return {
        "winner": winner,
        "candidates": candidates,
        "search_evaluations": counter.evaluations,
        "caps": {
            "workspace_bytes": MAX_WORKSPACE_BYTES,
            "family_candidates": MAX_FAMILY_CANDIDATES,
            "search_evaluations": MAX_SEARCH_EVALUATIONS,
            "bmp_alternations": MAX_BMP_ALTERNATIONS,
            "qtt_toggles_per_plane": MAX_QTT_TOGGLES_PER_PLANE,
            "exceptions": MAX_EXCEPTIONS,
            "obdd_nodes": MAX_OBDD_NODES,
        },
    }


def exhaustive_small_indices(distortion: np.ndarray, lambda_bit: float,
                             descriptor_bits: Callable[[np.ndarray], int]
                             ) -> dict:
    """Exact 64^N label search used only for N<=3 mechanism tests."""
    table = np.asarray(distortion, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 64 or not 1 <= table.shape[0] <= 3:
        raise CodecError("small exhaustive shape")
    best = None
    rows = np.arange(table.shape[0])
    for values in itertools.product(range(64), repeat=table.shape[0]):
        indices = np.asarray(values, dtype=np.uint8)
        bits = int(descriptor_bits(indices))
        score = float(table[rows, indices].sum(dtype=np.float64) +
                      lambda_bit * bits)
        if best is None or (score, values) < (best[0], best[1]):
            best = (score, values, bits)
    assert best is not None
    return {"objective": best[0], "indices": list(best[1]),
            "descriptor_bits": best[2],
            "evaluated": 64 ** table.shape[0]}
