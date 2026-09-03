#!/usr/bin/env python3
"""Literal-rank source-first aperture and deterministic owner bootstrap."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np


BLOCK_VALUES = 4096
MAX_RANK = 14
DICTIONARY_COLUMNS = 384
REQUIRED_CAPTURE = 0.32387022205373717
COARSE_RELATIVE_MSE = 0.036975150060595235
TARGET_D = 0.025
BOOTSTRAP_REPLICATES = 4096
BOOTSTRAP_ALPHA = 0.05
ROLE_ORDER = ("gate", "up", "down_transposed")


class ApertureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ApertureError(message)


def _host(xp: Any, value: Any) -> np.ndarray:
    return np.asarray(xp.asnumpy(value) if hasattr(xp, "asnumpy") else value)


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    z = (value + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def bf16_le_to_f64(payload: bytes, expected_values: int) -> np.ndarray:
    require(type(payload) is bytes and len(payload) == 2 * expected_values,
            "literal BF16 byte count")
    bits = np.frombuffer(payload, dtype="<u2").astype("<u4") << 16
    values = bits.view("<f4").astype("<f8")
    require(values.size == expected_values and np.all(np.isfinite(values)),
            "finite BF16 source")
    return np.ascontiguousarray(values, dtype="<f8")


def coarse_f32_le_to_f64(payload: bytes, expected_values: int) -> np.ndarray:
    require(type(payload) is bytes and len(payload) == 4 * expected_values,
            "literal coarse reconstruction byte count")
    values = np.frombuffer(payload, dtype="<f4").astype("<f8")
    require(values.size == expected_values and np.all(np.isfinite(values)),
            "finite independently decoded coarse reconstruction")
    return np.ascontiguousarray(values, dtype="<f8")


def fixed_sample_blocks(values: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    require(flat.size == 384 * BLOCK_VALUES and len(indices) == 16 and
            len(set(indices)) == 16 and all(0 <= index < 384 for index in indices),
            "fixed full-block sample geometry")
    blocks = flat.reshape(384, BLOCK_VALUES)
    return np.ascontiguousarray(blocks[np.asarray(indices, dtype=np.int64)], dtype="<f8")


def score_literal_rank_packets(xp: Any, core: Any, source_blocks: Any,
                               coarse_blocks: Any, role: str,
                               prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Score every representable rank 0..14 after literal packet replay.

    All linear systems, reconstructions, and SSE reductions are batched.  One
    bulk metadata transfer is needed to emit canonical packets; one bulk SSE
    transfer returns the complete candidate matrix.  There are no per-candidate
    solves, matmuls, or scalar synchronizations.
    """
    require(role in ROLE_ORDER, "role")
    source = xp.asarray(source_blocks, dtype=xp.float64)
    coarse = xp.asarray(coarse_blocks, dtype=xp.float64)
    require(source.ndim == 2 and source.shape == coarse.shape and
            source.shape[1] == BLOCK_VALUES and source.shape[0] > 0,
            "sample block arrays")
    blocks = int(source.shape[0])
    residual = source - coarse
    dictionary = xp.asarray(prepared["dictionary"], dtype=xp.float64)
    dictionary_squared = xp.asarray(prepared["dictionary_squared"], dtype=xp.float64)
    require(dictionary.shape == (BLOCK_VALUES, DICTIONARY_COLUMNS),
            "frozen dictionary shape")

    correlations = residual @ dictionary
    block_norms = xp.broadcast_to(
        xp.sum(dictionary_squared, axis=0, dtype=xp.float64)[None, :],
        correlations.shape)
    scores = xp.abs(correlations) / xp.sqrt(block_norms)
    atom_indices = xp.broadcast_to(
        xp.arange(DICTIONARY_COLUMNS, dtype=xp.int64)[None, :], scores.shape)
    support_order = xp.lexsort((atom_indices, -scores), axis=1)[:, :MAX_RANK]
    selected_correlations = xp.take_along_axis(correlations, support_order, axis=1)
    atoms = xp.transpose(xp.take(dictionary, support_order, axis=1), (1, 0, 2))
    gram = xp.einsum("bnk,bnl->bkl", atoms, atoms)

    ranks = xp.arange(1, MAX_RANK + 1, dtype=xp.int64)
    coefficient = xp.arange(MAX_RANK, dtype=xp.int64)
    active = coefficient[None, :] < ranks[:, None]
    active64 = active.astype(xp.float64)
    outer = active64[:, :, None] * active64[:, None, :]
    matrices = gram[:, None, :, :] * outer[None, :, :, :]
    diagonal = xp.diagonal(gram, axis1=1, axis2=2)
    diagonal_mean = xp.cumsum(diagonal, axis=1) / ranks[None, :]
    ridge = xp.maximum(diagonal_mean, xp.float64(1.0)) * xp.float64(2.0 ** -40)
    identity = xp.eye(MAX_RANK, dtype=xp.float64)
    matrices = matrices + (
        ridge[:, :, None, None] * identity[None, None, :, :]
        * active64[None, :, :, None]
        + identity[None, None, :, :] * (xp.float64(1.0) - active64)[None, :, :, None]
    )
    rhs = selected_correlations[:, None, :] * active64[None, :, :]
    solved = xp.linalg.solve(
        matrices.reshape(blocks * MAX_RANK, MAX_RANK, MAX_RANK),
        rhs.reshape(blocks * MAX_RANK, MAX_RANK, 1),
    ).reshape(blocks, MAX_RANK, MAX_RANK)
    maximum = xp.max(xp.abs(solved) * active64[None, :, :], axis=2)
    scales = (maximum / core.COEFFICIENT_MAX).astype(xp.float16).astype(xp.float64)
    safe_scales = xp.where(scales > 0.0, scales, xp.float64(1.0))
    quantized = xp.rint(solved / safe_scales[:, :, None])
    quantized = xp.clip(quantized, core.COEFFICIENT_MIN,
                        core.COEFFICIENT_MAX).astype(xp.int64)
    quantized = xp.where(active[None, :, :], quantized, xp.int64(0))
    valid = (scales > 0.0) & xp.all(
        (~active[None, :, :]) | (quantized != 0), axis=2)

    # One bulk host transfer for the complete candidate state.
    metadata = _host(xp, xp.concatenate((
        support_order[:, None, :].repeat(MAX_RANK, axis=1).astype(xp.float64),
        scales[:, :, None], quantized.astype(xp.float64),
        valid[:, :, None].astype(xp.float64),
    ), axis=2))
    literal_coefficients = np.zeros(
        (blocks, MAX_RANK + 1, DICTIONARY_COLUMNS), dtype=np.float64)
    packet_hashes: list[list[str | None]] = []
    packet_bytes = bytearray()
    representable = np.zeros((blocks, MAX_RANK + 1), dtype=bool)
    representable[:, 0] = True
    for block in range(blocks):
        row_hashes: list[str | None] = []
        zero = core.encode_packet(role, (), (), 0.0)
        decoded_zero = core.decode_packet(zero)
        require(decoded_zero["rank"] == 0 and core.encode_packet(
            role, decoded_zero["support"], decoded_zero["coefficients"],
            decoded_zero["scale"]) == zero, "literal rank-zero replay")
        packet_bytes.extend(zero)
        row_hashes.append(hashlib.sha256(zero).hexdigest())
        for rank in range(1, MAX_RANK + 1):
            fields = metadata[block, rank - 1]
            support = fields[:MAX_RANK].astype(np.int64)
            scale = float(fields[MAX_RANK])
            q = fields[MAX_RANK + 1:2 * MAX_RANK + 1].astype(np.int64)
            candidate_valid = bool(fields[-1])
            if not candidate_valid:
                row_hashes.append(None)
                continue
            pairs = sorted((int(support[index]), int(q[index]))
                           for index in range(rank))
            packet = core.encode_packet(
                role, tuple(atom for atom, _ in pairs),
                tuple(value for _, value in pairs), scale)
            decoded = core.decode_packet(packet)
            require(decoded["rank"] == rank and core.encode_packet(
                decoded["role"], decoded["support"], decoded["coefficients"],
                decoded["scale"]) == packet, "literal nonzero-rank replay")
            for atom, value in zip(decoded["support"], decoded["coefficients"], strict=True):
                literal_coefficients[block, rank, int(atom)] = (
                    int(value) * float(decoded["scale"]))
            representable[block, rank] = True
            packet_bytes.extend(packet)
            row_hashes.append(hashlib.sha256(packet).hexdigest())
        packet_hashes.append(row_hashes)

    literal_device = xp.asarray(literal_coefficients, dtype=xp.float64)
    corrections = xp.einsum("nk,brk->brn", dictionary, literal_device)
    errors = residual[:, None, :] - corrections
    candidate_sse_device = xp.sum(errors * errors, axis=2, dtype=xp.float64)
    valid_device = xp.asarray(representable)
    candidate_sse_device = xp.where(valid_device, candidate_sse_device,
                                    xp.float64("inf"))
    # One bulk transfer contains all literal rank scores.
    candidate_sse = np.ascontiguousarray(_host(xp, candidate_sse_device), dtype="<f8")
    winners = np.argmin(candidate_sse, axis=1).astype(np.int64)
    winner_sse = candidate_sse[np.arange(blocks), winners]
    input_sse = candidate_sse[:, 0]
    require(np.all(np.isfinite(input_sse)) and np.all(np.isfinite(winner_sse)) and
            np.all(winner_sse <= input_sse + 1e-12), "literal candidate SSE")
    return {
        "role": role,
        "blocks": blocks,
        "candidate_ranks": tuple(range(MAX_RANK + 1)),
        "candidate_sse": candidate_sse,
        "input_sse_by_block": input_sse,
        "remaining_sse_by_block": winner_sse,
        "winner_rank_by_block": winners,
        "packet_sha256_by_block_rank": packet_hashes,
        "literal_packet_set_sha256": hashlib.sha256(bytes(packet_bytes)).hexdigest(),
        "representable_candidates": int(np.sum(representable)),
        "candidate_states": blocks * (MAX_RANK + 1),
        "batched_solve_calls": 1,
        "batched_literal_reconstruction_einsums": 1,
        "per_candidate_solve_calls": 0,
        "per_candidate_matmul_calls": 0,
        "per_candidate_host_scalar_syncs": 0,
        "bulk_host_transfers": 2 if hasattr(xp, "asnumpy") else 0,
        "all_scored_candidates_literal_packet_replayed": True,
    }


def _lower_order_statistic(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    index = int(math.floor(alpha * (ordered.size - 1)))
    return float(ordered[index])


def bootstrap_capture_gate(rows: Mapping[str, Mapping[str, Any]], *,
                           replicates: int = BOOTSTRAP_REPLICATES,
                           alpha: float = BOOTSTRAP_ALPHA) -> dict[str, Any]:
    """Deterministic role-owner and stratified-block capture lower bounds."""
    require(set(rows) == set(ROLE_ORDER) and type(replicates) is int and
            replicates == BOOTSTRAP_REPLICATES and alpha == BOOTSTRAP_ALPHA,
            "frozen bootstrap contract")
    input_by_role = {}
    captured_by_role = {}
    for role in ROLE_ORDER:
        input_sse = np.asarray(rows[role]["input_sse_by_block"], dtype=np.float64)
        remaining = np.asarray(rows[role]["remaining_sse_by_block"], dtype=np.float64)
        require(input_sse.ndim == remaining.ndim == 1 and input_sse.size == 16 and
                remaining.size == 16 and np.all(np.isfinite(input_sse)) and
                np.all(np.isfinite(remaining)) and np.all(input_sse > 0.0) and
                np.all(remaining >= 0.0), "owner block SSE rows")
        input_by_role[role] = input_sse
        captured_by_role[role] = input_sse - remaining

    owner_bootstrap = {role: np.empty(replicates, dtype=np.float64) for role in ROLE_ORDER}
    pooled_bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        pooled_input = 0.0
        pooled_capture = 0.0
        for role_index, role in enumerate(ROLE_ORDER):
            selected = np.empty(16, dtype=np.int64)
            for draw in range(16):
                counter = ((role_index + 1) << 56) ^ (replicate << 16) ^ draw
                selected[draw] = _splitmix64(counter) % 16
            sampled_input = float(np.sum(input_by_role[role][selected], dtype=np.float64))
            sampled_capture = float(np.sum(captured_by_role[role][selected],
                                           dtype=np.float64))
            owner_bootstrap[role][replicate] = sampled_capture / sampled_input
            pooled_input += sampled_input
            pooled_capture += sampled_capture
        pooled_bootstrap[replicate] = pooled_capture / pooled_input

    owner_point = {role: float(np.sum(captured_by_role[role]) /
                               np.sum(input_by_role[role])) for role in ROLE_ORDER}
    owner_lcb = {role: _lower_order_statistic(owner_bootstrap[role], alpha)
                 for role in ROLE_ORDER}
    total_input = sum(float(np.sum(input_by_role[role])) for role in ROLE_ORDER)
    total_capture = sum(float(np.sum(captured_by_role[role])) for role in ROLE_ORDER)
    pooled_point = total_capture / total_input
    pooled_lcb = _lower_order_statistic(pooled_bootstrap, alpha)
    conservative_d = COARSE_RELATIVE_MSE * (1.0 - pooled_lcb)
    survives = (pooled_lcb >= REQUIRED_CAPTURE and
                min(owner_lcb.values()) >= REQUIRED_CAPTURE and
                conservative_d <= TARGET_D)
    return {
        "schema": "tactic-ramanujan384-qwen-source-first-aperture-v0",
        "bootstrap_prng": "SplitMix64 absolute role/replicate/draw counter",
        "bootstrap_replicates": replicates,
        "bootstrap_alpha": alpha,
        "owner_point_capture": owner_point,
        "owner_lcb_capture": owner_lcb,
        "pooled_point_capture": pooled_point,
        "pooled_lcb_capture": pooled_lcb,
        "required_capture": REQUIRED_CAPTURE,
        "coarse_relative_mse": COARSE_RELATIVE_MSE,
        "conservative_relative_mse_from_lcb": conservative_d,
        "target_d": TARGET_D,
        "survives_to_full_expert": survives,
        "controls_permitted": False,
        "full_expert_permitted": survives,
    }

