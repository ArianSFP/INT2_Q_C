#!/usr/bin/env python3
"""Static, batched CuPy/NumPy Ramanujan-384 codec core.

The hot path contains no dynamic imports, per-candidate Python solves or
matmuls, and no scalar device-to-host synchronizations.  All ranks are one
batched linear solve and all candidate reconstructions are one batched einsum.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BLOCK_VALUES = 4096
DICTIONARY_COLUMNS = 384
MAX_RANK = 14
ROLE_ORDER = ("gate", "up", "down_transposed")
TARGET_D = 0.025
MIN_CONTROL_EXCESS_BPW = 0.03
GAUSSIAN_SEEDS = (
    10619863, 10619881, 10619909, 10619927,
    10619953, 10619971, 10619999, 10620017,
)
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

PACKET_MAGIC = 0x4652
PACKET_VERSION = 2
PACKET_BYTES = 48
PACKET_BODY_BYTES = 44
PACKET_BODY_BITS = 352
COEFFICIENT_MIN = -1023
COEFFICIENT_MAX = 1023

CONTAINER_MAGIC = b"TRM384S2"
CONTAINER_VERSION = 2
HEADER_BYTES = 512
PAGE_BYTES = 4096
HEADER_PREFIX = struct.Struct("<8sIIIIIIQQ32s32s32s")
HEADER_CRC = struct.Struct("<I")


class ScalableCodecError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScalableCodecError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ShapeContract:
    intermediate: int
    hidden: int
    role_values: int
    total_values: int
    blocks_per_role: int
    last_block_valid_values: int
    tail_values_per_role: int
    coarse_bytes: int
    fine_bytes: int

    @property
    def tail_free(self) -> bool:
        return self.tail_values_per_role == 0

    def valid_values_for_block(self, block: int) -> int:
        require(type(block) is int and 0 <= block < self.blocks_per_role, "block index")
        return BLOCK_VALUES if block + 1 < self.blocks_per_role else self.last_block_valid_values


def define_shape(intermediate: int, hidden: int) -> ShapeContract:
    require(type(intermediate) is int and 0 < intermediate <= UINT32_MAX,
            "positive uint32 intermediate")
    require(type(hidden) is int and 0 < hidden <= UINT32_MAX,
            "positive uint32 hidden")
    role_values = intermediate * hidden
    blocks = (role_values + BLOCK_VALUES - 1) // BLOCK_VALUES
    require(blocks <= UINT32_MAX, "role block count exceeds inherited uint32 header")
    total_values = 3 * role_values
    coarse_numerator = 307 * total_values
    require(coarse_numerator % 1024 == 0,
            "shape has nonintegral exact 307/128-bpw coarse byte length")
    coarse_bytes = coarse_numerator // 1024
    fine_bytes = 3 * blocks * PACKET_BYTES
    require(coarse_bytes <= UINT64_MAX and fine_bytes <= UINT64_MAX,
            "payload length exceeds inherited uint64 header")
    last_valid = role_values - (blocks - 1) * BLOCK_VALUES
    require(1 <= last_valid <= BLOCK_VALUES, "last-block valid values")
    tail = 0 if last_valid == BLOCK_VALUES else BLOCK_VALUES - last_valid
    return ShapeContract(
        intermediate=intermediate,
        hidden=hidden,
        role_values=role_values,
        total_values=total_values,
        blocks_per_role=blocks,
        last_block_valid_values=last_valid,
        tail_values_per_role=tail,
        coarse_bytes=coarse_bytes,
        fine_bytes=fine_bytes,
    )


def physical_ledger(shape: ShapeContract) -> dict[str, Any]:
    unpadded = HEADER_BYTES + shape.coarse_bytes + shape.fine_bytes
    require(unpadded <= UINT64_MAX - (PAGE_BYTES - 1), "page rounding overflow")
    physical = ((unpadded + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES
    rate = 8.0 * physical / shape.total_values
    return {
        "header_bytes": HEADER_BYTES,
        "coarse_bytes": shape.coarse_bytes,
        "fine_bytes": shape.fine_bytes,
        "page_padding_bytes": physical - unpadded,
        "physical_bytes": physical,
        "physical_rate_bpw": rate,
        "target_rate_eligible": 2.15 <= rate <= 2.5,
        "blocks_fit_uint32": shape.blocks_per_role <= UINT32_MAX,
        "integer_only_page_rounding": True,
        "tail_values_per_role": shape.tail_values_per_role,
    }


def totient(value: int) -> int:
    require(type(value) is int and value >= 1, "totient input")
    result = value
    remainder = value
    factor = 2
    while factor * factor <= remainder:
        if remainder % factor == 0:
            while remainder % factor == 0:
                remainder //= factor
            result -= result // factor
        factor += 1
    if remainder > 1:
        result -= result // remainder
    return result


def mobius(value: int) -> int:
    require(type(value) is int and value >= 1, "mobius input")
    remainder = value
    factors = 0
    factor = 2
    while factor * factor <= remainder:
        if remainder % factor == 0:
            remainder //= factor
            factors += 1
            if remainder % factor == 0:
                return 0
            while remainder % factor == 0:
                remainder //= factor
        factor += 1
    if remainder > 1:
        factors += 1
    return -1 if factors & 1 else 1


def ramanujan_sum(period: int, coordinate: int) -> int:
    quotient = period // math.gcd(period, coordinate)
    numerator = mobius(quotient) * totient(period)
    denominator = totient(quotient)
    require(numerator % denominator == 0, "integer Ramanujan sum")
    return numerator // denominator


def period_bank_labels() -> tuple[tuple[int, int], ...]:
    periods = tuple(period for period in range(3, 128) if period & (period - 1))
    require(len(periods) == 120, "non-dyadic period bank")
    labels = []
    shift = 0
    while len(labels) < DICTIONARY_COLUMNS:
        before = len(labels)
        for period in periods:
            if shift < totient(period):
                labels.append((period, shift))
                if len(labels) == DICTIONARY_COLUMNS:
                    break
        require(len(labels) > before, "period bank exhaustion")
        shift += 1
    require(len(set(labels)) == DICTIONARY_COLUMNS
            and {period for period, _ in labels} == set(periods), "period bank identity")
    return tuple(labels)


def prepare_dictionary(xp: Any) -> dict[str, Any]:
    coordinate = xp.arange(BLOCK_VALUES, dtype=xp.int64)
    labels = period_bank_labels()
    atoms = []
    for period, shift in labels:
        lookup = xp.asarray(
            [ramanujan_sum(period, index) for index in range(period)], dtype=xp.float64
        )
        atoms.append(lookup[(coordinate - shift) % period])
    dictionary = xp.stack(atoms, axis=1).astype(xp.float64, copy=False)
    return {
        "dictionary": dictionary,
        "dictionary_squared": dictionary * dictionary,
        "labels": labels,
        "backend": "cupy" if hasattr(xp, "asnumpy") else "numpy",
    }


class BitWriter:
    def __init__(self) -> None:
        self.value = 0
        self.bits = 0

    def write(self, value: int, width: int) -> None:
        require(type(value) is int and type(width) is int and width >= 0, "bit field type")
        require(0 <= value < (1 << width), "bit field range")
        self.value |= value << self.bits
        self.bits += width

    def finish(self, size: int) -> bytes:
        require(self.bits <= 8 * size, "bit writer overflow")
        return self.value.to_bytes(size, "little")


class BitReader:
    def __init__(self, payload: bytes) -> None:
        self.value = int.from_bytes(payload, "little")
        self.bits = 0
        self.limit = 8 * len(payload)

    def read(self, width: int) -> int:
        require(type(width) is int and width >= 0 and self.bits + width <= self.limit,
                "bit read")
        output = (self.value >> self.bits) & ((1 << width) - 1)
        self.bits += width
        return output


def _f16_bits(value: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise ScalableCodecError("binary16 scale") from exc


def _from_f16_bits(value: int) -> float:
    return float(struct.unpack("<e", struct.pack("<H", value))[0])


def encode_packet(role: str, support: Sequence[int], coefficients: Sequence[int],
                  scale: float) -> bytes:
    require(role in ROLE_ORDER, "packet role")
    support = tuple(int(value) for value in support)
    coefficients = tuple(int(value) for value in coefficients)
    require(len(support) == len(coefficients) <= MAX_RANK, "packet rank")
    require(support == tuple(sorted(support)) and len(set(support)) == len(support),
            "canonical support")
    require(all(0 <= value < DICTIONARY_COLUMNS for value in support), "support range")
    rank = len(support)
    scale_bits = _f16_bits(scale)
    canonical_scale = _from_f16_bits(scale_bits)
    if rank == 0:
        require(scale_bits == 0 and not coefficients, "zero-rank scale")
    else:
        require(math.isfinite(canonical_scale) and canonical_scale > 0.0,
                "positive finite binary16 scale")
        require(all(COEFFICIENT_MIN <= value <= COEFFICIENT_MAX and value != 0
                    for value in coefficients), "signed coefficients")
    writer = BitWriter()
    writer.write(PACKET_MAGIC, 16)
    writer.write(PACKET_VERSION, 4)
    writer.write(ROLE_ORDER.index(role), 2)
    writer.write(rank, 4)
    writer.write(scale_bits, 16)
    for atom, coefficient in zip(support, coefficients, strict=True):
        writer.write(atom, 9)
        writer.write(coefficient & 0x7FF, 11)
    body = writer.finish(PACKET_BODY_BYTES)
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def decode_packet(payload: bytes) -> dict[str, Any]:
    require(type(payload) is bytes and len(payload) == PACKET_BYTES, "packet size")
    body = payload[:PACKET_BODY_BYTES]
    require((zlib.crc32(body) & 0xFFFFFFFF)
            == struct.unpack("<I", payload[PACKET_BODY_BYTES:])[0], "packet CRC")
    reader = BitReader(body)
    require(reader.read(16) == PACKET_MAGIC and reader.read(4) == PACKET_VERSION,
            "packet identity")
    role_index = reader.read(2)
    require(role_index < len(ROLE_ORDER), "packet role ordinal")
    rank = reader.read(4)
    require(rank <= MAX_RANK, "packet rank")
    scale_bits = reader.read(16)
    scale = _from_f16_bits(scale_bits)
    support = []
    coefficients = []
    for _ in range(rank):
        support.append(reader.read(9))
        field = reader.read(11)
        coefficient = field - 2048 if field & 1024 else field
        require(COEFFICIENT_MIN <= coefficient <= COEFFICIENT_MAX and coefficient != 0,
                "canonical coefficient")
        coefficients.append(coefficient)
    require(tuple(support) == tuple(sorted(support)) and len(set(support)) == rank
            and all(value < DICTIONARY_COLUMNS for value in support), "canonical support")
    if rank == 0:
        require(scale_bits == 0, "canonical zero-rank scale")
    else:
        require(math.isfinite(scale) and scale > 0.0, "canonical scale")
    require(reader.read(PACKET_BODY_BITS - reader.bits) == 0, "canonical packet padding")
    result = {
        "role": ROLE_ORDER[role_index], "rank": rank, "support": tuple(support),
        "coefficients": tuple(coefficients), "scale": scale,
    }
    require(encode_packet(result["role"], result["support"], result["coefficients"],
                          result["scale"]) == payload, "canonical packet reencode")
    return result


def split_packets(payload: bytes) -> tuple[bytes, ...]:
    require(type(payload) is bytes and len(payload) % PACKET_BYTES == 0, "packet stream")
    packets = tuple(payload[offset:offset + PACKET_BYTES]
                    for offset in range(0, len(payload), PACKET_BYTES))
    for packet in packets:
        decode_packet(packet)
    return packets


def _host_array(xp: Any, value: Any) -> np.ndarray:
    return np.asarray(xp.asnumpy(value) if hasattr(xp, "asnumpy") else value)


def _role_blocks(xp: Any, values: Any, shape: ShapeContract) -> Any:
    flat = xp.asarray(values, dtype=xp.float64).reshape(-1)
    require(int(flat.size) == shape.role_values, "exact role value count")
    padded = xp.zeros(shape.blocks_per_role * BLOCK_VALUES, dtype=xp.float64)
    padded[:shape.role_values] = flat
    return padded.reshape(shape.blocks_per_role, BLOCK_VALUES)


def encode_role_batched(
    xp: Any,
    source_values: Any,
    coarse_values: Any,
    shape: ShapeContract,
    role: str,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    """Search all ranks in one device batch and perform two bulk host syncs.

    Rank prefixes 1..14 become a `(blocks*14,14,14)` solve.  Canonicalized
    packet coefficients reconstruct all candidates in one einsum.  Therefore
    the winner metric is the direct valid-coordinate FP64 SSE of the literal
    packet state without a projection shortcut or floating selection proxy.
    """

    require(role in ROLE_ORDER, "role")
    dictionary = xp.asarray(prepared["dictionary"], dtype=xp.float64)
    dictionary_squared = xp.asarray(prepared["dictionary_squared"], dtype=xp.float64)
    source = _role_blocks(xp, source_values, shape)
    coarse = _role_blocks(xp, coarse_values, shape)
    residual = source - coarse
    blocks = shape.blocks_per_role
    valid_counts_host = np.asarray(
        [shape.valid_values_for_block(block) for block in range(blocks)], dtype=np.int64
    )
    valid_counts = xp.asarray(valid_counts_host, dtype=xp.int64)
    coordinate = xp.arange(BLOCK_VALUES, dtype=xp.int64)[None, :]
    valid_mask = coordinate < valid_counts[:, None]
    mask64 = valid_mask.astype(xp.float64)

    correlations = residual @ dictionary
    block_norms = mask64 @ dictionary_squared
    scores = xp.abs(correlations) / xp.sqrt(block_norms)
    indices = xp.broadcast_to(
        xp.arange(DICTIONARY_COLUMNS, dtype=xp.int64)[None, :], scores.shape
    )
    support_order = xp.lexsort((indices, -scores), axis=1)[:, :MAX_RANK]
    selected_correlations = xp.take_along_axis(correlations, support_order, axis=1)
    atoms = xp.transpose(xp.take(dictionary, support_order, axis=1), (1, 0, 2))
    masked_atoms = atoms * mask64[:, :, None]
    selected_gram = xp.einsum("bnk,bnl->bkl", masked_atoms, masked_atoms)

    ranks = xp.arange(1, MAX_RANK + 1, dtype=xp.int64)
    coefficient_index = xp.arange(MAX_RANK, dtype=xp.int64)
    active = coefficient_index[None, :] < ranks[:, None]
    active64 = active.astype(xp.float64)
    active_outer = active64[:, :, None] * active64[:, None, :]
    matrices = selected_gram[:, None, :, :] * active_outer[None, :, :, :]
    diagonal = xp.diagonal(selected_gram, axis1=1, axis2=2)
    diagonal_prefix = xp.cumsum(diagonal, axis=1)
    diagonal_mean = diagonal_prefix / ranks[None, :]
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
    scales = (maximum / COEFFICIENT_MAX).astype(xp.float16).astype(xp.float64)
    safe_scales = xp.where(scales > 0.0, scales, xp.float64(1.0))
    quantized = xp.rint(solved / safe_scales[:, :, None])
    quantized = xp.clip(quantized, COEFFICIENT_MIN, COEFFICIENT_MAX).astype(xp.int64)
    quantized = xp.where(active[None, :, :], quantized, xp.int64(0))
    candidate_valid = (scales > 0.0) & xp.all(
        (~active[None, :, :]) | (quantized != 0), axis=2
    )
    dequantized = quantized.astype(xp.float64) * scales[:, :, None]

    # One batched matmul/einsum materializes direct literal-state corrections
    # for every rank; this replaces all per-candidate decode matmuls.
    candidate_corrections = xp.einsum("bnk,brk->brn", atoms, dequantized)
    candidate_errors = (
        residual[:, None, :] - candidate_corrections
    ) * mask64[:, None, :]
    candidate_sse = xp.sum(
        candidate_errors * candidate_errors, axis=2, dtype=xp.float64
    )
    candidate_sse = xp.where(candidate_valid, candidate_sse, xp.float64("inf"))
    rank_zero_sse = xp.sum(
        residual * residual * mask64, axis=1, dtype=xp.float64
    )[:, None]
    all_sse = xp.concatenate((rank_zero_sse, candidate_sse), axis=1)
    winner_rank = xp.argmin(all_sse, axis=1).astype(xp.int64)
    nonzero_index = xp.maximum(winner_rank - 1, 0)
    winning_scale = xp.take_along_axis(scales, nonzero_index[:, None], axis=1)[:, 0]
    winning_q = xp.take_along_axis(
        quantized, nonzero_index[:, None, None], axis=1
    )[:, 0, :]
    winning_q = xp.where(
        coefficient_index[None, :] < winner_rank[:, None], winning_q, xp.int64(0)
    )
    winning_scale = xp.where(winner_rank > 0, winning_scale, xp.float64(0.0))

    # One bulk metadata transfer replaces every rank/block scalar sync.
    metadata_device = xp.concatenate((
        winner_rank[:, None].astype(xp.float64),
        winning_scale[:, None],
        support_order.astype(xp.float64),
        winning_q.astype(xp.float64),
    ), axis=1)
    metadata = _host_array(xp, metadata_device)
    packets = []
    replay_coefficients = np.zeros((blocks, MAX_RANK), dtype=np.float64)
    expected_coefficients = np.zeros((blocks, MAX_RANK), dtype=np.float64)
    for block in range(blocks):
        rank = int(metadata[block, 0])
        scale = float(metadata[block, 1])
        support = metadata[block, 2:2 + MAX_RANK].astype(np.int64)
        coefficients = metadata[block, 2 + MAX_RANK:].astype(np.int64)
        expected_coefficients[block, :rank] = coefficients[:rank].astype(np.float64) * scale
        pairs = sorted((int(support[index]), int(coefficients[index]))
                       for index in range(rank))
        packet = encode_packet(
            role,
            tuple(pair[0] for pair in pairs),
            tuple(pair[1] for pair in pairs),
            0.0 if rank == 0 else scale,
        )
        decoded = decode_packet(packet)
        packets.append(packet)
        by_atom = {int(atom): int(value) * float(decoded["scale"])
                   for atom, value in zip(decoded["support"], decoded["coefficients"], strict=True)}
        for index, atom in enumerate(support):
            replay_coefficients[block, index] = by_atom.get(int(atom), 0.0)

    require(np.array_equal(replay_coefficients, expected_coefficients),
            "decoded packets equal canonicalized batched candidate state")

    replay_coefficients_device = xp.asarray(replay_coefficients, dtype=xp.float64)
    replay_correction = xp.einsum(
        "bnk,brk->brn", atoms, replay_coefficients_device[:, None, :]
    )[:, 0, :]
    replay_error = (residual - replay_correction) * mask64
    replay_sse_rows = xp.sum(replay_error * replay_error, axis=1, dtype=xp.float64)
    input_sse_rows = xp.sum(residual * residual * mask64, axis=1, dtype=xp.float64)
    chosen_sse = xp.take_along_axis(all_sse, winner_rank[:, None], axis=1)[:, 0]
    maximum_replay_sse_delta = xp.max(xp.abs(replay_sse_rows - chosen_sse))
    summary_device = xp.concatenate((
        xp.stack((
            xp.sum(input_sse_rows, dtype=xp.float64),
            xp.sum(replay_sse_rows, dtype=xp.float64),
            maximum_replay_sse_delta,
        )).astype(xp.float64),
        xp.bincount(winner_rank, minlength=MAX_RANK + 1).astype(xp.float64),
    ))
    summary = _host_array(xp, summary_device)
    stream = b"".join(packets)
    return {
        "stream": stream,
        "stream_sha256": sha256(stream),
        "stream_bytes": len(stream),
        "correction": replay_correction,
        "input_sse": float(summary[0]),
        "remaining_sse": float(summary[1]),
        "maximum_equivalent_kernel_sse_delta": float(summary[2]),
        "rank_histogram": tuple(int(value) for value in summary[3:]),
        "candidate_ranks_batched": MAX_RANK + 1,
        "batched_linear_solve_calls": 1,
        "batched_candidate_einsum_calls": 1,
        "bulk_device_to_host_transfers": 2 if hasattr(xp, "asnumpy") else 0,
        "per_candidate_host_scalar_syncs": 0,
        "per_candidate_solves": 0,
        "per_candidate_matmuls": 0,
        "dynamic_imports_in_hot_path": 0,
        "direct_valid_coordinate_candidate_sse": True,
        "decoded_packet_state_equals_selected_candidate_state": True,
        "tail_padding_in_candidate_generation_or_score": False,
    }


def decode_fine_role(xp: Any, stream: bytes, shape: ShapeContract, role: str,
                     prepared: Mapping[str, Any]) -> Any:
    packets = split_packets(stream)
    require(len(packets) == shape.blocks_per_role, "fine role packet count")
    supports = np.zeros((shape.blocks_per_role, MAX_RANK), dtype=np.int64)
    coefficients = np.zeros((shape.blocks_per_role, MAX_RANK), dtype=np.float64)
    for block, payload in enumerate(packets):
        row = decode_packet(payload)
        require(row["role"] == role, "fine role order")
        for index, (atom, value) in enumerate(zip(
                row["support"], row["coefficients"], strict=True)):
            supports[block, index] = int(atom)
            coefficients[block, index] = int(value) * float(row["scale"])
    dictionary = xp.asarray(prepared["dictionary"], dtype=xp.float64)
    support_device = xp.asarray(supports, dtype=xp.int64)
    atoms = xp.transpose(xp.take(dictionary, support_device, axis=1), (1, 0, 2))
    coefficients_device = xp.asarray(coefficients, dtype=xp.float64)
    return xp.einsum(
        "bnk,brk->brn", atoms, coefficients_device[:, None, :]
    )[:, 0, :]


def _splitmix64_numpy(seed: int, count: int) -> np.ndarray:
    require(type(seed) is int and 0 <= seed <= UINT64_MAX, "uint64 Gaussian seed")
    require(type(count) is int and count >= 0, "Gaussian word count")
    with np.errstate(over="ignore"):
        index = np.arange(count, dtype=np.uint64)
        value = index + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def canonical_gaussian_f64(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Canonical Box--Muller Gaussian samples from integer counters.

    Uniforms use the exact midpoint of a 53-bit integer cell.  Box--Muller is
    evaluated once on the host, rounded once to IEEE binary32, then widened
    exactly to binary64.  Both NumPy and CuPy consume these same serialized
    binary64 bytes; no backend RNG or backend transcendental participates.
    The finite counter sample is a Monte-Carlo realization of the mathematical
    Gaussian law, not a claim that a finite PRNG has a continuous support.
    """

    require(len(shape) == 2 and all(type(value) is int and value > 0 for value in shape),
            "Gaussian shape")
    count = shape[0] * shape[1]
    pairs = (count + 1) // 2
    words = _splitmix64_numpy(seed, 2 * pairs)
    scale = np.float64(1.0 / float(1 << 53))
    u1 = ((words[:pairs] >> np.uint64(11)).astype(np.float64) + np.float64(0.5)) * scale
    u2 = ((words[pairs:] >> np.uint64(11)).astype(np.float64) + np.float64(0.5)) * scale
    radius = np.sqrt(np.float64(-2.0) * np.log(u1))
    angle = np.float64(2.0 * math.pi) * u2
    output = np.empty(2 * pairs, dtype=np.float64)
    output[0::2] = radius * np.cos(angle)
    output[1::2] = radius * np.sin(angle)
    # The binary32 canonicalization is part of the frozen integer-to-normal
    # ABI and makes replay bytes insensitive to a last-bit host libm drift.
    canonical = output[:count].astype("<f4").astype("<f8")
    return np.ascontiguousarray(canonical.reshape(shape), dtype="<f8")


def moment_matched_gaussian(xp: Any, reference_blocks: Any, seed: int,
                            valid_counts: Sequence[int]) -> tuple[Any, dict[str, Any]]:
    host_reference = _host_array(xp, reference_blocks).astype("<f8", copy=False)
    require(host_reference.ndim == 2 and host_reference.shape[1] == BLOCK_VALUES,
            "Gaussian reference blocks")
    require(len(valid_counts) == host_reference.shape[0], "Gaussian valid counts")
    raw = canonical_gaussian_f64(tuple(host_reference.shape), seed)
    output = np.zeros_like(raw)
    for block, valid in enumerate(valid_counts):
        require(type(valid) is int and 1 < valid <= BLOCK_VALUES, "Gaussian valid count")
        reference = host_reference[block, :valid]
        source_mean = float(np.sum(reference, dtype=np.float64) / valid)
        centered_source = reference - source_mean
        source_energy = float(np.sum(centered_source * centered_source, dtype=np.float64))
        gaussian = raw[block, :valid]
        gaussian_mean = float(np.sum(gaussian, dtype=np.float64) / valid)
        centered_gaussian = gaussian - gaussian_mean
        gaussian_energy = float(np.sum(centered_gaussian * centered_gaussian, dtype=np.float64))
        require(math.isfinite(source_energy) and gaussian_energy > 0.0,
                "Gaussian moment energies")
        output[block, :valid] = (
            source_mean + centered_gaussian * math.sqrt(source_energy / gaussian_energy)
        )
    output = np.ascontiguousarray(output, dtype="<f8")
    return xp.asarray(output, dtype=xp.float64), {
        "schema": "tactic-ramanujan384-canonical-box-muller-control-v2",
        "seed": seed,
        "f64_sha256": sha256(output.tobytes(order="C")),
        "counter": "SplitMix64 absolute integer coordinate",
        "integer_to_normal": "53-bit midpoint Box-Muller; canonical binary32 then exact binary64 widen",
        "backend_rng_calls": 0,
        "backend_transcendental_calls": 0,
        "cpu_cupy_input_bytes_identical_by_construction": True,
    }


def encode_header(shape: ShapeContract, coarse: bytes, fine: bytes,
                  source_binding_sha256: str) -> bytes:
    require(type(coarse) is bytes and len(coarse) == shape.coarse_bytes, "coarse bytes")
    require(type(fine) is bytes and len(fine) == shape.fine_bytes, "fine bytes")
    require(isinstance(source_binding_sha256, str) and len(source_binding_sha256) == 64,
            "source binding digest")
    try:
        binding = bytes.fromhex(source_binding_sha256)
    except ValueError as exc:
        raise ScalableCodecError("source binding digest") from exc
    prefix = HEADER_PREFIX.pack(
        CONTAINER_MAGIC, CONTAINER_VERSION, shape.intermediate, shape.hidden,
        shape.blocks_per_role, shape.blocks_per_role, shape.blocks_per_role,
        len(coarse), len(fine), hashlib.sha256(coarse).digest(),
        hashlib.sha256(fine).digest(), binding,
    )
    body = prefix + bytes(HEADER_BYTES - HEADER_CRC.size - len(prefix))
    return body + HEADER_CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode_header(payload: bytes) -> dict[str, Any]:
    require(type(payload) is bytes and len(payload) == HEADER_BYTES, "header bytes")
    body = payload[:-HEADER_CRC.size]
    require((zlib.crc32(body) & 0xFFFFFFFF) == HEADER_CRC.unpack(payload[-4:])[0],
            "header CRC")
    fields = HEADER_PREFIX.unpack_from(body, 0)
    (magic, version, intermediate, hidden, gate_blocks, up_blocks, down_blocks,
     coarse_bytes, fine_bytes, coarse_sha, fine_sha, binding) = fields
    require(magic == CONTAINER_MAGIC and version == CONTAINER_VERSION, "header identity")
    shape = define_shape(intermediate, hidden)
    require((gate_blocks, up_blocks, down_blocks) == (shape.blocks_per_role,) * 3,
            "uint32 block counts")
    require(coarse_bytes == shape.coarse_bytes and fine_bytes == shape.fine_bytes,
            "header payload lengths")
    require(body[HEADER_PREFIX.size:] == bytes(len(body) - HEADER_PREFIX.size),
            "header zero padding")
    return {
        "shape": shape, "coarse_bytes": coarse_bytes, "fine_bytes": fine_bytes,
        "coarse_sha256": coarse_sha.hex(), "fine_sha256": fine_sha.hex(),
        "source_binding_sha256": binding.hex(),
    }


def _validate_fine(fine: bytes, shape: ShapeContract) -> None:
    packets = split_packets(fine)
    require(len(packets) == 3 * shape.blocks_per_role, "fine packet count")
    cursor = 0
    for role in ROLE_ORDER:
        for payload in packets[cursor:cursor + shape.blocks_per_role]:
            require(decode_packet(payload)["role"] == role, "canonical fine role order")
        cursor += shape.blocks_per_role


def encode_composite(shape: ShapeContract, coarse: bytes,
                     role_fine_streams: Sequence[bytes], source_binding_sha256: str) -> bytes:
    require(len(role_fine_streams) == 3, "three fine streams")
    span = shape.blocks_per_role * PACKET_BYTES
    require(all(type(stream) is bytes and len(stream) == span for stream in role_fine_streams),
            "fine stream lengths")
    fine = b"".join(role_fine_streams)
    _validate_fine(fine, shape)
    header = encode_header(shape, coarse, fine, source_binding_sha256)
    unpadded = header + coarse + fine
    physical = ((len(unpadded) + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES
    return unpadded + bytes(physical - len(unpadded))


def decode_composite(payload: bytes) -> dict[str, Any]:
    require(type(payload) is bytes and len(payload) >= HEADER_BYTES
            and len(payload) % PAGE_BYTES == 0, "page-aligned composite")
    header = decode_header(payload[:HEADER_BYTES])
    coarse_end = HEADER_BYTES + header["coarse_bytes"]
    fine_end = coarse_end + header["fine_bytes"]
    expected = ((fine_end + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES
    require(len(payload) == expected and payload[fine_end:] == bytes(expected - fine_end),
            "minimal canonical page padding")
    coarse = payload[HEADER_BYTES:coarse_end]
    fine = payload[coarse_end:fine_end]
    require(sha256(coarse) == header["coarse_sha256"]
            and sha256(fine) == header["fine_sha256"], "payload hashes")
    _validate_fine(fine, header["shape"])
    require(encode_header(header["shape"], coarse, fine, header["source_binding_sha256"])
            == payload[:HEADER_BYTES], "canonical header reencode")
    return {"header": header, "coarse": coarse, "fine": fine}


def read_composite_once(path: Path, expected_bytes: int) -> tuple[bytes, dict[str, Any]]:
    absolute = path.resolve(strict=True)
    require(absolute == path.absolute(), "canonical composite path")
    descriptor = os.open(os.fspath(absolute), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
    events = []
    output = bytearray()
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and before.st_size == expected_bytes, "regular single-link composite")
        while len(output) < expected_bytes:
            requested = min(8 << 20, expected_bytes - len(output))
            row = os.read(descriptor, requested)
            require(bool(row), "short composite read")
            events.append({"offset": len(output), "requested": requested, "returned": len(row)})
            output.extend(row)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
                "composite identity drift")
    finally:
        os.close(descriptor)
    return bytes(output), {
        "events": events, "returned_bytes": len(output), "object_bytes": expected_bytes,
        "instrumented_file_read_amplification": len(output) / expected_bytes,
        "layout_bound_1x_is_not_measurement": True,
        "physical_storage_or_hbm_telemetry": False,
    }
