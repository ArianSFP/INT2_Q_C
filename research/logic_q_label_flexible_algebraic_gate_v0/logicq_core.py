#!/usr/bin/env python3
"""Finite source-free LOGIC-Q mechanisms and exact physical accounting.

This module has no payload entrypoint and imports no accelerator library.
Numerical search functions accept an explicitly supplied NumPy module so that
ordinary imports and the standard-library verifier remain inert.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Any, Iterable, Sequence


TARGET_F = 0.8
RATE_MIN = 2.15
RATE_MAX = 2.5
REFERENCE_RATE = 2.5
REFERENCE_RELATIVE_MSE = 0.030902167403153148
REFERENCE_F = REFERENCE_RELATIVE_MSE * 2.0 ** (2.0 * REFERENCE_RATE)
REQUIRED_SAVING_BPW = -0.5 * math.log2(TARGET_F / REFERENCE_F)
REQUIRED_MSE_REDUCTION = 1.0 - 0.025 / REFERENCE_RELATIVE_MSE

FAMILY_LITERAL = 0
FAMILY_RM1 = 1
FAMILY_GF2 = 2
FAMILY_ROMDD = 3
FAMILY_NAMES = {
    FAMILY_LITERAL: "literal4",
    FAMILY_RM1: "rm1_plus_exceptions",
    FAMILY_GF2: "gf2_rank_plus_exceptions",
    FAMILY_ROMDD: "romdd_plus_exceptions",
}
ROLE_IDS = {"gate": 0, "up": 1, "down_transposed": 2}
ROLE_NAMES = {value: key for key, value in ROLE_IDS.items()}
GRAY_BITS = ((0, 0), (0, 1), (1, 1), (1, 0))
BITS_TO_LABEL = (0, 1, 3, 2)

COMPONENT_MAGIC = b"LOGICQ0\0"
EXPERT_MAGIC = b"LQEXPERT"
COMPONENT_HEADER = struct.Struct(">8sBBBBIIIIIIQQ12s")
EXPERT_HEADER = struct.Struct(">8sBBHQQQ28s")
COMPONENT_HEADER_BYTES = COMPONENT_HEADER.size
EXPERT_HEADER_BYTES = EXPERT_HEADER.size
assert COMPONENT_HEADER_BYTES == 64
assert EXPERT_HEADER_BYTES == 64

COMPONENT_ALIGNMENT = 64
EXPERT_PAGE = 4096
CONTROL_SEEDS = (
    10619863,
    10619881,
    10619909,
    10619927,
    10619953,
    10619971,
    10619999,
    10620017,
)

# Public symmetric reproduction shapes. The complete profile selector is paid
# in the component header. Scale is transmitted separately per public block.
PROFILE_RATIOS = (
    (-1.510417608, -0.452780039, 0.452780039, 1.510417608),
    (-2.0, -0.5, 0.5, 2.0),
    (-1.75, -0.5833333333333334, 0.5833333333333334, 1.75),
    (-1.35, -0.35, 0.35, 1.35),
)


class LogicQError(RuntimeError):
    """Fail-closed packet, search, or protocol error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LogicQError(message)


def ceil_div(numerator: int, denominator: int) -> int:
    require(isinstance(numerator, int) and isinstance(denominator, int),
            "ceil_div integer operands")
    require(numerator >= 0 and denominator > 0, "ceil_div domain")
    return (numerator + denominator - 1) // denominator


def align_up(value: int, alignment: int) -> int:
    require(value >= 0 and alignment > 0, "alignment domain")
    return ceil_div(value, alignment) * alignment


def ceil_log2_count(count: int) -> int:
    require(isinstance(count, int) and count >= 1, "positive count")
    return (count - 1).bit_length()


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def canonical_role_shape(role: str, rows: int, cols: int) -> tuple[int, int]:
    require(role in ROLE_IDS, "known semantic role")
    require(rows > 0 and cols > 0, "positive role shape")
    return rows, cols


class BitWriter:
    """MSB-first finite bit writer with literal zero byte padding."""

    def __init__(self) -> None:
        self._bytes = bytearray()
        self._current = 0
        self._used = 0
        self.bit_length = 0

    def write(self, value: int, width: int) -> None:
        require(isinstance(value, int) and isinstance(width, int),
                "bit write integer")
        require(width >= 0 and value >= 0 and value < (1 << width if width else 1),
                "bit write range")
        for shift in range(width - 1, -1, -1):
            self._current = (self._current << 1) | ((value >> shift) & 1)
            self._used += 1
            self.bit_length += 1
            if self._used == 8:
                self._bytes.append(self._current)
                self._current = 0
                self._used = 0

    def finish(self) -> bytes:
        if self._used:
            self._bytes.append(self._current << (8 - self._used))
            self._current = 0
            self._used = 0
        return bytes(self._bytes)

    def pad_to_byte(self) -> int:
        padding = (-self.bit_length) & 7
        self.write(0, padding)
        return padding


class BitReader:
    """MSB-first reader bounded by the authenticated literal bit length."""

    def __init__(self, payload: bytes, bit_length: int) -> None:
        require(bit_length >= 0 and len(payload) == ceil_div(bit_length, 8),
                "bit reader length")
        if bit_length & 7 and payload:
            unused = 8 - (bit_length & 7)
            require(payload[-1] & ((1 << unused) - 1) == 0,
                    "nonzero final byte padding")
        self.payload = payload
        self.bit_length = bit_length
        self.position = 0

    def read(self, width: int) -> int:
        require(width >= 0 and self.position + width <= self.bit_length,
                "bit reader overrun")
        result = 0
        for _ in range(width):
            byte = self.payload[self.position >> 3]
            bit = (byte >> (7 - (self.position & 7))) & 1
            self.position += 1
            result = (result << 1) | bit
        return result

    def finish(self) -> None:
        require(self.position == self.bit_length, "unconsumed packet bits")

    def align_to_byte(self) -> int:
        padding = (-self.position) & 7
        require(self.read(padding) == 0, "nonzero internal byte padding")
        return padding


def subset_rank_lex(positions: Sequence[int], n: int) -> int:
    """Lexicographic rank of a strictly increasing e-subset of range(n)."""
    values = tuple(int(value) for value in positions)
    e = len(values)
    require(n >= 0 and all(0 <= value < n for value in values),
            "subset position range")
    require(all(values[index] < values[index + 1]
                for index in range(max(0, e - 1))), "strict subset order")
    rank = 0
    previous = -1
    for index, value in enumerate(values):
        remaining = e - index - 1
        for skipped in range(previous + 1, value):
            rank += math.comb(n - skipped - 1, remaining)
        previous = value
    require(rank < math.comb(n, e), "subset rank closure")
    return rank


def subset_unrank_lex(rank: int, n: int, e: int) -> tuple[int, ...]:
    count = math.comb(n, e)
    require(0 <= e <= n and 0 <= rank < count, "subset unrank domain")
    result: list[int] = []
    next_value = 0
    residual = rank
    for index in range(e):
        remaining = e - index - 1
        maximum = n - remaining
        selected = None
        for candidate in range(next_value, maximum):
            span = math.comb(n - candidate - 1, remaining)
            if residual < span:
                selected = candidate
                break
            residual -= span
        require(selected is not None, "subset unrank exhausted")
        result.append(selected)
        next_value = selected + 1
    require(residual == 0, "subset unrank residual")
    return tuple(result)


def exception_bits(n: int, e: int) -> dict[str, int]:
    require(n >= 1 and 0 <= e <= n, "exception count domain")
    count_bits = ceil_log2_count(n + 1)
    subset_count = math.comb(n, e)
    subset_bits = ceil_log2_count(subset_count)
    replacement_count = 3 ** e
    replacement_bits = ceil_log2_count(replacement_count)
    return {
        "count_bits": count_bits,
        "subset_bits": subset_bits,
        "replacement_bits": replacement_bits,
        "total_bits": count_bits + subset_bits + replacement_bits,
    }


@dataclass(frozen=True)
class ExceptionPlan:
    positions: tuple[int, ...]
    labels: tuple[int, ...]
    distortion: float
    bits: int
    charged_total_bits: int
    objective: float

    @property
    def count(self) -> int:
        return len(self.positions)


def optimize_exceptions(np: Any, costs: Any, base_labels: Any,
                        lambda_per_bit: float, maximum: int | None = None,
                        *, fixed_prefix_bits: int = 0,
                        byte_align_total: bool = False) -> ExceptionPlan:
    """Exact exception overlay for one fixed base-label object."""
    matrix = np.asarray(costs, dtype=np.float64)
    base = np.asarray(base_labels, dtype=np.uint8).reshape(-1)
    require(matrix.shape == (base.size, 4), "exception cost shape")
    require(np.all(np.isfinite(matrix)) and np.all(matrix >= 0.0),
            "finite nonnegative exception costs")
    require(np.all(base < 4), "base labels")
    n = int(base.size)
    limit = n if maximum is None else min(n, int(maximum))
    require(limit >= 0 and fixed_prefix_bits >= 0 and
            math.isfinite(lambda_per_bit) and lambda_per_bit >= 0.0,
            "exception optimization domain")

    indices = np.arange(n, dtype=np.int64)
    base_cost = matrix[indices, base.astype(np.int64)]
    alternatives = np.empty(n, dtype=np.uint8)
    alternative_cost = np.empty(n, dtype=np.float64)
    for position in range(n):
        label = int(base[position])
        choices = [candidate for candidate in range(4) if candidate != label]
        winner = min(choices, key=lambda candidate: (float(matrix[position, candidate]),
                                                       candidate))
        alternatives[position] = winner
        alternative_cost[position] = matrix[position, winner]
    deltas = alternative_cost - base_cost
    order = np.asarray(sorted(range(n), key=lambda position: (float(deltas[position]),
                                                               position)),
                       dtype=np.int64)
    prefix = np.concatenate((np.zeros(1, dtype=np.float64),
                             np.cumsum(deltas[order], dtype=np.float64)))
    original = float(np.sum(base_cost, dtype=np.float64))
    best: ExceptionPlan | None = None
    for count in range(limit + 1):
        chosen = tuple(sorted(int(value) for value in order[:count]))
        labels = tuple(int(alternatives[position]) for position in chosen)
        distortion = original + float(prefix[count])
        bits = exception_bits(n, count)["total_bits"]
        unaligned_total = fixed_prefix_bits + bits
        charged_total = (align_up(unaligned_total, 8)
                         if byte_align_total else unaligned_total)
        objective = distortion + lambda_per_bit * charged_total
        candidate = ExceptionPlan(
            chosen, labels, distortion, bits, charged_total, objective)
        key = (candidate.objective, candidate.charged_total_bits,
               candidate.bits, candidate.distortion,
               candidate.positions, candidate.labels)
        if best is None or key < (
                best.objective, best.charged_total_bits, best.bits,
                best.distortion, best.positions, best.labels):
            best = candidate
    require(best is not None, "exception plan exists")
    return best


def write_exceptions(writer: BitWriter, n: int, base_labels: Sequence[int],
                     plan: ExceptionPlan) -> None:
    base = tuple(int(value) for value in base_labels)
    require(len(base) == n and len(plan.positions) == len(plan.labels),
            "exception write shape")
    widths = exception_bits(n, plan.count)
    writer.write(plan.count, widths["count_bits"])
    subset_rank = subset_rank_lex(plan.positions, n)
    writer.write(subset_rank, widths["subset_bits"])
    trit_rank = 0
    multiplier = 1
    for position, label in zip(plan.positions, plan.labels):
        choices = tuple(candidate for candidate in range(4)
                        if candidate != base[position])
        require(label in choices, "exception replacement differs")
        trit_rank += choices.index(label) * multiplier
        multiplier *= 3
    writer.write(trit_rank, widths["replacement_bits"])


def read_exceptions(reader: BitReader, n: int,
                    base_labels: Sequence[int]) -> tuple[int, ...]:
    base = [int(value) for value in base_labels]
    require(len(base) == n and all(0 <= value < 4 for value in base),
            "exception base decode")
    count_bits = ceil_log2_count(n + 1)
    count = reader.read(count_bits)
    require(count <= n, "exception count unused rank")
    widths = exception_bits(n, count)
    subset_rank = reader.read(widths["subset_bits"])
    require(subset_rank < math.comb(n, count), "exception subset unused rank")
    positions = subset_unrank_lex(subset_rank, n, count)
    trit_rank = reader.read(widths["replacement_bits"])
    require(trit_rank < 3 ** count, "exception trit unused rank")
    residual = trit_rank
    for position in positions:
        choices = tuple(candidate for candidate in range(4)
                        if candidate != base[position])
        base[position] = choices[residual % 3]
        residual //= 3
    require(residual == 0, "exception trit residual")
    return tuple(base)


def fp32_to_bf16_bits(value: float) -> int:
    require(math.isfinite(value) and value > 0.0, "positive finite BF16 scale")
    raw = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    upper = raw >> 16
    lower = raw & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    upper &= 0xFFFF
    decoded = bf16_bits_to_float(upper)
    require(math.isfinite(decoded) and decoded > 0.0, "representable BF16 scale")
    return upper


def bf16_bits_to_float(bits: int) -> float:
    require(isinstance(bits, int) and 0 <= bits <= 0xFFFF, "BF16 word")
    return struct.unpack(">f", struct.pack(">I", bits << 16))[0]


def scale_payload(scales: Sequence[int]) -> bytes:
    require(scales and all(isinstance(value, int) and 0 < value <= 0x7F7F
                           for value in scales), "scale words")
    return b"".join(struct.pack(">H", value) for value in scales)


def parse_scale_payload(payload: bytes, blocks: int) -> tuple[int, ...]:
    require(blocks >= 1 and len(payload) == blocks * 2, "scale payload length")
    values = tuple(struct.unpack_from(">H", payload, 2 * index)[0]
                   for index in range(blocks))
    for value in values:
        require(math.isfinite(bf16_bits_to_float(value)) and
                bf16_bits_to_float(value) > 0.0, "decoded positive scale")
    return values


def fit_scales(np: Any, values: Any, block_size: int,
               profile: int) -> tuple[int, ...]:
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    require(source.size > 0 and source.size % block_size == 0 and
            is_power_of_two(block_size), "scale block geometry")
    require(0 <= profile < len(PROFILE_RATIOS), "profile index")
    ratios = np.asarray(PROFILE_RATIOS[profile], dtype=np.float64)
    result: list[int] = []
    for start in range(0, source.size, block_size):
        block = source[start:start + block_size]
        rms = math.sqrt(max(float(np.mean(block * block, dtype=np.float64)),
                            2.0 ** -126))
        best: tuple[float, int] | None = None
        # Fixed public grid: no hidden optimizer state is needed by the decoder.
        for exponent in range(-12, 13):
            candidate = rms * 2.0 ** (exponent / 8.0)
            word = fp32_to_bf16_bits(candidate)
            decoded = bf16_bits_to_float(word)
            levels = decoded * ratios
            costs = (block[:, None] - levels[None, :]) ** 2
            distortion = float(np.sum(np.min(costs, axis=1), dtype=np.float64))
            key = (distortion, word)
            if best is None or key < best:
                best = key
        require(best is not None, "scale candidate")
        result.append(best[1])
    return tuple(result)


def distortion_costs(np: Any, values: Any, weights: Any, block_size: int,
                     profile: int, scales: Sequence[int]) -> tuple[Any, Any]:
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    importance = np.asarray(weights, dtype=np.float64).reshape(-1)
    require(source.shape == importance.shape and source.size > 0 and
            source.size % block_size == 0, "cost input geometry")
    require(np.all(np.isfinite(source)) and np.all(np.isfinite(importance)) and
            np.all(importance > 0.0), "finite positive weighted source")
    require(len(scales) == source.size // block_size and
            0 <= profile < len(PROFILE_RATIOS), "cost quantizer descriptor")
    costs = np.empty((source.size, 4), dtype=np.float64)
    reconstruction = np.empty((source.size, 4), dtype=np.float64)
    ratios = np.asarray(PROFILE_RATIOS[profile], dtype=np.float64)
    for block_index, start in enumerate(range(0, source.size, block_size)):
        levels = bf16_bits_to_float(int(scales[block_index])) * ratios
        reconstruction[start:start + block_size, :] = levels[None, :]
        residual = source[start:start + block_size, None] - levels[None, :]
        costs[start:start + block_size, :] = (
            importance[start:start + block_size, None] * residual * residual)
    return costs, reconstruction


@dataclass(frozen=True)
class ComponentHeaderRecord:
    family: int
    role: str
    profile: int
    rows: int
    cols: int
    block_size: int
    parameter: int
    blocks: int
    scale_bytes: int
    payload_bits: int
    source_count: int


def encode_component_header(record: ComponentHeaderRecord) -> bytes:
    require(record.family in FAMILY_NAMES and record.role in ROLE_IDS,
            "component selector")
    require(0 <= record.profile < len(PROFILE_RATIOS), "component profile")
    require(record.rows > 0 and record.cols > 0 and
            record.source_count == record.rows * record.cols,
            "component shape closure")
    require(record.blocks >= 1 and record.scale_bytes == 2 * record.blocks and
            record.source_count == record.blocks * record.block_size,
            "component block closure")
    require(record.parameter >= 0 and record.payload_bits >= 0,
            "component fields")
    return COMPONENT_HEADER.pack(
        COMPONENT_MAGIC,
        0,
        record.family,
        ROLE_IDS[record.role],
        record.profile,
        record.rows,
        record.cols,
        record.block_size,
        record.parameter,
        record.blocks,
        record.scale_bytes,
        record.payload_bits,
        record.source_count,
        b"\0" * 12,
    )


def decode_component_header(payload: bytes) -> ComponentHeaderRecord:
    require(len(payload) == COMPONENT_HEADER_BYTES, "component header bytes")
    (magic, version, family, role_id, profile, rows, cols, block_size,
     parameter, blocks, scale_bytes, payload_bits, source_count,
     reserved) = COMPONENT_HEADER.unpack(payload)
    require(magic == COMPONENT_MAGIC and version == 0 and reserved == b"\0" * 12,
            "component header magic/version/reserved")
    require(family in FAMILY_NAMES and role_id in ROLE_NAMES,
            "component decoded selector")
    record = ComponentHeaderRecord(
        family, ROLE_NAMES[role_id], profile, rows, cols, block_size, parameter,
        blocks, scale_bytes, payload_bits, source_count)
    encode_component_header(record)
    return record


@dataclass(frozen=True)
class EncodedComponent:
    packet: bytes
    labels: tuple[int, ...]
    reconstruction: tuple[float, ...]
    weighted_sse: float
    source_energy: float
    family: str
    exact_search: bool
    diagnostics: dict[str, Any]

    @property
    def physical_bits(self) -> int:
        return len(self.packet) * 8

    @property
    def rate_bpw(self) -> float:
        return self.physical_bits / len(self.labels)

    @property
    def relative_mse(self) -> float:
        return self.weighted_sse / self.source_energy

    @property
    def F(self) -> float:
        return self.relative_mse * 2.0 ** (2.0 * self.rate_bpw)


def numeric_reconstruction(np: Any, labels: Sequence[int], reconstruction: Any) -> tuple[float, ...]:
    index = np.arange(len(labels), dtype=np.int64)
    values = np.asarray(reconstruction, dtype=np.float64)[index,
                                                         np.asarray(labels, dtype=np.int64)]
    return tuple(float(value) for value in values)


def component_packet(record: ComponentHeaderRecord, scales: Sequence[int],
                     family_payload: bytes) -> bytes:
    require(len(family_payload) == ceil_div(record.payload_bits, 8),
            "family payload bytes")
    if record.payload_bits & 7 and family_payload:
        require(family_payload[-1] & ((1 << (8 - (record.payload_bits & 7))) - 1) == 0,
                "family payload padding")
    scale_bytes = scale_payload(scales)
    require(len(scale_bytes) == record.scale_bytes, "packet scale bytes")
    return encode_component_header(record) + scale_bytes + family_payload


def parse_component_envelope(packet: bytes) -> tuple[ComponentHeaderRecord,
                                                     tuple[int, ...], bytes]:
    require(len(packet) >= COMPONENT_HEADER_BYTES, "component packet minimum")
    record = decode_component_header(packet[:COMPONENT_HEADER_BYTES])
    payload_bytes = ceil_div(record.payload_bits, 8)
    expected = COMPONENT_HEADER_BYTES + record.scale_bytes + payload_bytes
    require(len(packet) == expected, "component packet exact closure")
    scales = parse_scale_payload(
        packet[COMPONENT_HEADER_BYTES:COMPONENT_HEADER_BYTES + record.scale_bytes],
        record.blocks)
    family_payload = packet[COMPONENT_HEADER_BYTES + record.scale_bytes:]
    return record, scales, family_payload


def gray_labels_from_planes(np: Any, plane0: Any, plane1: Any) -> Any:
    first = np.asarray(plane0, dtype=np.uint8)
    second = np.asarray(plane1, dtype=np.uint8)
    require(first.shape == second.shape and np.all(first < 2) and np.all(second < 2),
            "Gray plane geometry")
    table = np.asarray(BITS_TO_LABEL, dtype=np.uint8)
    return table[(first << 1) | second]


def label_planes(np: Any, labels: Any) -> tuple[Any, Any]:
    values = np.asarray(labels, dtype=np.uint8)
    require(np.all(values < 4), "ordered labels")
    mapping = np.asarray(GRAY_BITS, dtype=np.uint8)
    mapped = mapping[values]
    return mapped[..., 0], mapped[..., 1]


def rm1_codewords(np: Any, n: int) -> Any:
    """All RM(1,m) words; message bit zero is the affine constant."""
    require(is_power_of_two(n), "RM block power of two")
    m = n.bit_length() - 1
    messages = 1 << (m + 1)
    coordinates = np.arange(n, dtype=np.uint64)
    result = np.empty((messages, n), dtype=np.uint8)
    for message in range(messages):
        word = np.full(n, message & 1, dtype=np.uint8)
        for variable in range(m):
            if (message >> (variable + 1)) & 1:
                word ^= ((coordinates >> variable) & 1).astype(np.uint8)
        result[message, :] = word
    return result


def _rm_base_cost_matrix(np: Any, costs: Any, codewords: Any) -> Any:
    """Exact joint four-level distortion for every pair of affine words."""
    matrix = np.asarray(costs, dtype=np.float64)
    words = np.asarray(codewords, dtype=np.uint8)
    require(matrix.shape == (words.shape[1], 4), "RM block cost shape")
    # Any Boolean-pair cost is c00 + a*x + b*y + g*x*y.
    c00 = matrix[:, 0]
    c01 = matrix[:, 1]
    c10 = matrix[:, 3]
    c11 = matrix[:, 2]
    alpha = c10 - c00
    beta = c01 - c00
    gamma = c11 - c10 - c01 + c00
    first = words.astype(np.float64)
    result = (
        float(np.sum(c00, dtype=np.float64))
        + np.sum(first * alpha[None, :], axis=1, dtype=np.float64)[:, None]
        + np.sum(first * beta[None, :], axis=1, dtype=np.float64)[None, :]
        + (first * gamma[None, :]) @ first.T
    )
    require(result.shape == (words.shape[0], words.shape[0]) and
            np.all(np.isfinite(result)), "RM base cost matrix")
    return result


@dataclass(frozen=True)
class RMBlockPlan:
    first_message: int
    second_message: int
    exceptions: ExceptionPlan
    labels: tuple[int, ...]
    base_bits: int
    total_bits: int
    distortion: float
    objective: float
    exact_pair_search: bool
    pair_candidates_evaluated: int


def search_rm1_block(np: Any, costs: Any, lambda_per_bit: float,
                     exception_limit: int | None = None,
                     exact_pair_max: int = 4096,
                     list_pairs: int = 256) -> RMBlockPlan:
    matrix = np.asarray(costs, dtype=np.float64)
    require(matrix.ndim == 2 and matrix.shape[1] == 4 and
            is_power_of_two(int(matrix.shape[0])), "RM search cost geometry")
    n = int(matrix.shape[0])
    require(n >= 2 and lambda_per_bit >= 0.0 and
            exact_pair_max >= 1 and list_pairs >= 1, "RM search parameters")
    words = rm1_codewords(np, n)
    pair_costs = _rm_base_cost_matrix(np, matrix, words)
    pair_count = int(pair_costs.size)
    exact = pair_count <= exact_pair_max
    if exact:
        flat_candidates = np.arange(pair_count, dtype=np.int64)
    else:
        retain = min(list_pairs, pair_count)
        # argpartition is followed by a total deterministic order.
        selected = np.argpartition(pair_costs.reshape(-1), retain - 1)[:retain]
        flat_candidates = np.asarray(sorted(
            (int(value) for value in selected),
            key=lambda value: (float(pair_costs.reshape(-1)[value]), value)),
            dtype=np.int64)
    messages = int(words.shape[0])
    m = n.bit_length() - 1
    base_bits = 2 * (m + 1)
    best: RMBlockPlan | None = None
    for flat in flat_candidates:
        first_message = int(flat) // messages
        second_message = int(flat) % messages
        base = gray_labels_from_planes(
            np, words[first_message], words[second_message]).reshape(-1)
        exceptions = optimize_exceptions(
            np, matrix, base, lambda_per_bit, exception_limit,
            fixed_prefix_bits=base_bits, byte_align_total=True)
        labels = list(int(value) for value in base)
        for position, label in zip(exceptions.positions, exceptions.labels):
            labels[position] = label
        total_bits = exceptions.charged_total_bits
        distortion = exceptions.distortion
        objective = exceptions.objective
        candidate = RMBlockPlan(
            first_message, second_message, exceptions, tuple(labels), base_bits,
            total_bits, distortion, objective, exact, len(flat_candidates))
        key = (candidate.objective, candidate.total_bits, candidate.distortion,
               candidate.first_message, candidate.second_message,
               candidate.exceptions.positions, candidate.exceptions.labels)
        if best is None or key < (
                best.objective, best.total_bits, best.distortion,
                best.first_message, best.second_message,
                best.exceptions.positions, best.exceptions.labels):
            best = candidate
    require(best is not None, "RM block candidate")
    return best


def write_rm1_block(writer: BitWriter, np: Any, n: int,
                    plan: RMBlockPlan) -> None:
    m = n.bit_length() - 1
    width = m + 1
    writer.write(plan.first_message, width)
    writer.write(plan.second_message, width)
    words = rm1_codewords(np, n)
    base = gray_labels_from_planes(
        np, words[plan.first_message], words[plan.second_message]).reshape(-1)
    write_exceptions(writer, n, tuple(int(value) for value in base),
                     plan.exceptions)
    writer.pad_to_byte()


def read_rm1_block(reader: BitReader, np: Any, n: int) -> tuple[int, ...]:
    require(is_power_of_two(n), "RM decode block")
    width = n.bit_length()
    messages = 1 << width
    first_message = reader.read(width)
    second_message = reader.read(width)
    require(first_message < messages and second_message < messages,
            "RM message bounds")
    words = rm1_codewords(np, n)
    base = gray_labels_from_planes(
        np, words[first_message], words[second_message]).reshape(-1)
    labels = read_exceptions(reader, n, tuple(int(value) for value in base))
    reader.align_to_byte()
    return labels


def source_energy_value(np: Any, values: Any, weights: Any) -> float:
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    importance = np.asarray(weights, dtype=np.float64).reshape(-1)
    require(source.shape == importance.shape and np.all(importance > 0.0),
            "source energy geometry")
    energy = float(np.sum(importance * source * source, dtype=np.float64))
    require(math.isfinite(energy) and energy > 0.0, "positive source energy")
    return energy


def weighted_sse_from_labels(np: Any, costs: Any, labels: Sequence[int]) -> float:
    matrix = np.asarray(costs, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    require(matrix.shape == (label_array.size, 4) and
            np.all((0 <= label_array) & (label_array < 4)), "label SSE geometry")
    return float(np.sum(matrix[np.arange(label_array.size), label_array],
                        dtype=np.float64))


def _build_component(np: Any, *, family: int, role: str, rows: int, cols: int,
                     block_size: int, parameter: int, profile: int,
                     scales: Sequence[int], writer: BitWriter,
                     labels: Sequence[int], values: Any, weights: Any,
                     costs: Any, reconstruction: Any, exact_search: bool,
                     diagnostics: dict[str, Any]) -> EncodedComponent:
    payload = writer.finish()
    record = ComponentHeaderRecord(
        family=family, role=role, profile=profile, rows=rows, cols=cols,
        block_size=block_size, parameter=parameter,
        blocks=rows * cols // block_size, scale_bytes=2 * len(scales),
        payload_bits=writer.bit_length, source_count=rows * cols)
    packet = component_packet(record, scales, payload)
    decoded_labels, decoded_values, decoded_record = decode_component(np, packet)
    require(tuple(labels) == decoded_labels and
            decoded_record == record, "independent component label replay")
    expected_values = numeric_reconstruction(np, labels, reconstruction)
    require(all(struct.pack(">d", left) == struct.pack(">d", right)
                for left, right in zip(expected_values, decoded_values)),
            "independent numeric reconstruction replay")
    sse = weighted_sse_from_labels(np, costs, labels)
    energy = source_energy_value(np, values, weights)
    return EncodedComponent(
        packet=packet, labels=tuple(int(value) for value in labels),
        reconstruction=decoded_values, weighted_sse=sse,
        source_energy=energy, family=FAMILY_NAMES[family],
        exact_search=exact_search, diagnostics=diagnostics)


def encode_literal_component(np: Any, values: Any, weights: Any, *, role: str,
                             rows: int, cols: int, block_size: int,
                             profile: int | None = None) -> EncodedComponent:
    canonical_role_shape(role, rows, cols)
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    require(source.size == rows * cols and source.size % block_size == 0,
            "literal component geometry")
    candidates = range(len(PROFILE_RATIOS)) if profile is None else (profile,)
    best_inputs = None
    for selected_profile in candidates:
        scales = fit_scales(np, source, block_size, int(selected_profile))
        costs, reconstruction = distortion_costs(
            np, source, weights, block_size, int(selected_profile), scales)
        labels = np.argmin(costs, axis=1).astype(np.uint8)
        key = (weighted_sse_from_labels(np, costs, labels), int(selected_profile))
        if best_inputs is None or key < best_inputs[0]:
            best_inputs = (key, int(selected_profile), scales, costs,
                           reconstruction, labels)
    require(best_inputs is not None, "literal profile")
    _, selected_profile, scales, costs, reconstruction, labels = best_inputs
    writer = BitWriter()
    for label in labels:
        writer.write(int(label), 2)
    return _build_component(
        np, family=FAMILY_LITERAL, role=role, rows=rows, cols=cols,
        block_size=block_size, parameter=0, profile=selected_profile,
        scales=scales, writer=writer, labels=tuple(int(value) for value in labels),
        values=source, weights=weights, costs=costs,
        reconstruction=reconstruction, exact_search=True,
        diagnostics={"label_search": "exact nearest legal level",
                     "model_selector_charged_in_header": True})


def encode_rm1_component(np: Any, values: Any, weights: Any, *, role: str,
                         rows: int, cols: int, block_size: int,
                         lambda_per_bit: float, profile: int | None = None,
                         exception_limit: int | None = None,
                         exact_pair_max: int = 4096,
                         list_pairs: int = 256) -> EncodedComponent:
    canonical_role_shape(role, rows, cols)
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    require(source.size == rows * cols and source.size % block_size == 0 and
            is_power_of_two(block_size), "RM component geometry")
    candidates = range(len(PROFILE_RATIOS)) if profile is None else (profile,)
    best = None
    for selected_profile in candidates:
        scales = fit_scales(np, source, block_size, int(selected_profile))
        costs, reconstruction = distortion_costs(
            np, source, weights, block_size, int(selected_profile), scales)
        plans = []
        for start in range(0, source.size, block_size):
            plans.append(search_rm1_block(
                np, costs[start:start + block_size], lambda_per_bit,
                exception_limit, exact_pair_max, list_pairs))
        bits = sum(plan.total_bits for plan in plans)
        distortion = sum(plan.distortion for plan in plans)
        key = (distortion + lambda_per_bit * bits, bits, distortion,
               int(selected_profile))
        if best is None or key < best[0]:
            best = (key, int(selected_profile), scales, costs,
                    reconstruction, tuple(plans))
    require(best is not None, "RM component profile")
    _, selected_profile, scales, costs, reconstruction, plans = best
    writer = BitWriter()
    labels: list[int] = []
    for plan in plans:
        write_rm1_block(writer, np, block_size, plan)
        labels.extend(plan.labels)
    exact = all(plan.exact_pair_search for plan in plans)
    return _build_component(
        np, family=FAMILY_RM1, role=role, rows=rows, cols=cols,
        block_size=block_size, parameter=1, profile=selected_profile,
        scales=scales, writer=writer, labels=tuple(labels), values=source,
        weights=weights, costs=costs, reconstruction=reconstruction,
        exact_search=exact,
        diagnostics={
            "rm_order": 1,
            "blocks": len(plans),
            "exact_pair_search": exact,
            "pair_candidates_evaluated_per_block":
                [plan.pair_candidates_evaluated for plan in plans],
            "exception_counts": [plan.exceptions.count for plan in plans],
            "label_flexible_objective": True,
            "model_selector_charged_in_header": True,
        })


def gf2_product(np: Any, left: Any, right: Any) -> Any:
    u = np.asarray(left, dtype=np.uint8)
    v = np.asarray(right, dtype=np.uint8)
    require(u.ndim == 2 and v.ndim == 2 and u.shape[1] == v.shape[0] and
            np.all(u < 2) and np.all(v < 2), "GF2 factor geometry")
    return ((u.astype(np.uint64) @ v.astype(np.uint64)) & 1).astype(np.uint8)


def gf2_rank(np: Any, matrix: Any) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy()
    require(work.ndim == 2 and np.all(work < 2), "GF2 rank matrix")
    rows, cols = work.shape
    pivot_row = 0
    for column in range(cols):
        pivot = next((row for row in range(pivot_row, rows)
                      if int(work[row, column]) == 1), None)
        if pivot is None:
            continue
        if pivot != pivot_row:
            work[[pivot_row, pivot], :] = work[[pivot, pivot_row], :]
        for row in range(rows):
            if row != pivot_row and int(work[row, column]):
                work[row, :] ^= work[pivot_row, :]
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def raw_gf2_factor_bits(rows: int, cols: int, rank: int,
                        bitplanes: int = 2) -> int:
    require(rows > 0 and cols > 0 and 0 <= rank <= min(rows, cols) and
            bitplanes >= 1, "raw factor bit domain")
    return bitplanes * rank * (rows + cols)


def log2_rank_r_binary_matrix_count(rows: int, cols: int, rank: int) -> float:
    """Information-theoretic count; not an implemented serializer."""
    require(rows > 0 and cols > 0 and 0 <= rank <= min(rows, cols),
            "rank count domain")
    result = 0.0
    for index in range(rank):
        result += rows + math.log2(1.0 - 2.0 ** (index - rows))
        result += cols + math.log2(1.0 - 2.0 ** (index - cols))
        result -= rank + math.log2(1.0 - 2.0 ** (index - rank))
    return result


def ideal_rank_matrix_bits(rows: int, cols: int, rank: int,
                           bitplanes: int = 2) -> int:
    require(bitplanes >= 1, "rank count bitplanes")
    return math.ceil(bitplanes * log2_rank_r_binary_matrix_count(
        rows, cols, rank))


def rank680_accounting() -> dict[str, Any]:
    rows, cols, rank = 768, 2048, 680
    weights_per_role = rows * cols
    raw = raw_gf2_factor_bits(rows, cols, rank)
    ideal_asymptotic = 2 * rank * (rows + cols - rank)
    exact_count = ideal_rank_matrix_bits(rows, cols, rank)
    return {
        "rows": rows,
        "cols": cols,
        "roles_per_expert": 3,
        "rank_per_bitplane": rank,
        "bitplanes": 2,
        "weights_per_role": weights_per_role,
        "weights_per_expert": 3 * weights_per_role,
        "implemented_raw_factor_bits_per_role": raw,
        "implemented_raw_factor_bits_all_three_roles": 3 * raw,
        "implemented_raw_factor_bpw_per_role_and_expert": raw / weights_per_role,
        "ideal_asymptotic_factor_bits_per_role": ideal_asymptotic,
        "ideal_asymptotic_bpw_per_role_and_expert":
            ideal_asymptotic / weights_per_role,
        "ideal_exact_rank_matrix_count_bits_per_role": exact_count,
        "ideal_exact_rank_matrix_count_bpw_per_role_and_expert":
            exact_count / weights_per_role,
        "ideal_counting_serializer_implemented": False,
        "headers_scales_exceptions_alignment_included_above": False,
    }


@dataclass(frozen=True)
class GF2Plan:
    rank: int
    u0: tuple[tuple[int, ...], ...]
    v0: tuple[tuple[int, ...], ...]
    u1: tuple[tuple[int, ...], ...]
    v1: tuple[tuple[int, ...], ...]
    exceptions: ExceptionPlan
    labels: tuple[int, ...]
    factor_bits: int
    total_bits: int
    distortion: float
    objective: float
    exact_search: bool
    candidates_evaluated: int


def _tuple_matrix(array: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in array.tolist())


def _factor_base_labels(np: Any, factors: Sequence[Any]) -> Any:
    u0, v0, u1, v1 = factors
    return gray_labels_from_planes(np, gf2_product(np, u0, v0),
                                   gf2_product(np, u1, v1)).reshape(-1)


def _evaluate_gf2_factors(np: Any, costs: Any, factors: Sequence[Any],
                          rank: int, lambda_per_bit: float,
                          exception_limit: int | None,
                          exact: bool, candidates: int) -> GF2Plan:
    matrix = np.asarray(costs, dtype=np.float64)
    rows, cols = np.asarray(factors[0]).shape[0], np.asarray(factors[1]).shape[1]
    base = _factor_base_labels(np, factors)
    factor_bits = raw_gf2_factor_bits(rows, cols, rank)
    exceptions = optimize_exceptions(
        np, matrix, base, lambda_per_bit, exception_limit,
        fixed_prefix_bits=factor_bits, byte_align_total=True)
    labels = [int(value) for value in base]
    for position, label in zip(exceptions.positions, exceptions.labels):
        labels[position] = label
    total_bits = exceptions.charged_total_bits
    distortion = exceptions.distortion
    objective = exceptions.objective
    u0, v0, u1, v1 = factors
    return GF2Plan(
        rank, _tuple_matrix(u0), _tuple_matrix(v0), _tuple_matrix(u1),
        _tuple_matrix(v1), exceptions, tuple(labels), factor_bits, total_bits,
        distortion, objective, exact, candidates)


def _binary_array_from_integer(np: Any, value: int,
                               shape: tuple[int, int]) -> Any:
    result = np.empty(shape[0] * shape[1], dtype=np.uint8)
    for index in range(result.size):
        result[index] = (value >> index) & 1
    return result.reshape(shape)


def _exact_factor_candidates(np: Any, rows: int, cols: int,
                             rank: int) -> list[tuple[int, Any, Any, bytes]]:
    per_plane_bits = rank * (rows + cols)
    require(per_plane_bits <= 12, "exact factor enumeration cap")
    unique: dict[bytes, tuple[int, Any, Any, bytes]] = {}
    u_bits = rows * rank
    for code in range(1 << per_plane_bits):
        u = _binary_array_from_integer(np, code & ((1 << u_bits) - 1),
                                       (rows, rank))
        v = _binary_array_from_integer(np, code >> u_bits, (rank, cols))
        base = gf2_product(np, u, v)
        key = bytes(base.reshape(-1).tolist())
        candidate = (code, u, v, key)
        if key not in unique or code < unique[key][0]:
            unique[key] = candidate
    return sorted(unique.values(), key=lambda row: row[0])


def _initial_row_factor(np: Any, target: Any, rank: int) -> tuple[Any, Any]:
    matrix = np.asarray(target, dtype=np.uint8)
    rows, cols = matrix.shape
    require(0 <= rank <= min(rows, cols) and rank <= 12,
            "bounded row-factor initialization")
    if rank == 0:
        return (np.zeros((rows, 0), dtype=np.uint8),
                np.zeros((0, cols), dtype=np.uint8))
    basis: list[Any] = []
    current_rank = 0
    for row in matrix:
        trial = np.stack(basis + [row.copy()], axis=0)
        trial_rank = gf2_rank(np, trial)
        if trial_rank > current_rank:
            basis.append(row.copy())
            current_rank = trial_rank
            if len(basis) == rank:
                break
    while len(basis) < rank:
        vector = np.zeros(cols, dtype=np.uint8)
        vector[len(basis) % cols] = 1
        basis.append(vector)
    v = np.stack(basis, axis=0).astype(np.uint8)
    combinations = np.empty((1 << rank, cols), dtype=np.uint8)
    for code in range(1 << rank):
        word = np.zeros(cols, dtype=np.uint8)
        for component in range(rank):
            if (code >> component) & 1:
                word ^= v[component]
        combinations[code] = word
    u = np.empty((rows, rank), dtype=np.uint8)
    for row_index, row in enumerate(matrix):
        winner = min(range(1 << rank), key=lambda code: (
            int(np.count_nonzero(combinations[code] ^ row)), code))
        for component in range(rank):
            u[row_index, component] = (winner >> component) & 1
    return u, v


def search_gf2(np: Any, costs: Any, rows: int, cols: int, rank: int,
               lambda_per_bit: float, exception_limit: int | None = None,
               exact_factor_pair_max: int = 65536,
               heuristic_sweeps: int = 2) -> GF2Plan:
    matrix = np.asarray(costs, dtype=np.float64)
    require(matrix.shape == (rows * cols, 4) and
            0 <= rank <= min(rows, cols), "GF2 search geometry")
    require(lambda_per_bit >= 0.0 and exact_factor_pair_max >= 1 and
            heuristic_sweeps >= 0, "GF2 search parameters")
    per_plane_bits = rank * (rows + cols)
    if per_plane_bits <= 12:
        plane_candidates = _exact_factor_candidates(np, rows, cols, rank)
        pair_count = len(plane_candidates) ** 2
        if pair_count <= exact_factor_pair_max:
            best: GF2Plan | None = None
            best_key = None
            evaluated = 0
            for first in plane_candidates:
                for second in plane_candidates:
                    evaluated += 1
                    candidate = _evaluate_gf2_factors(
                        np, matrix, (first[1], first[2], second[1], second[2]),
                        rank, lambda_per_bit, exception_limit, True, pair_count)
                    key = (candidate.objective, candidate.total_bits,
                           candidate.distortion, first[0], second[0])
                    if best is None or best_key is None or key < best_key:
                        best = candidate
                        best_key = key
            require(best is not None and evaluated == pair_count,
                    "exact GF2 enumeration")
            return best

    require(rank <= 12, "bounded GF2 heuristic rank cap")
    nearest = np.argmin(matrix, axis=1).astype(np.uint8).reshape(rows, cols)
    plane0, plane1 = label_planes(np, nearest)
    factors = [*_initial_row_factor(np, plane0, rank),
               *_initial_row_factor(np, plane1, rank)]
    current = _evaluate_gf2_factors(
        np, matrix, factors, rank, lambda_per_bit, exception_limit, False, 1)
    evaluated = 1
    for _ in range(heuristic_sweeps):
        changed = False
        for factor_index, factor in enumerate(factors):
            for flat in range(int(factor.size)):
                row, column = divmod(flat, int(factor.shape[1]))
                factor[row, column] ^= np.uint8(1)
                candidate = _evaluate_gf2_factors(
                    np, matrix, factors, rank, lambda_per_bit,
                    exception_limit, False, evaluated + 1)
                evaluated += 1
                if (candidate.objective, candidate.total_bits,
                    candidate.distortion) < (
                        current.objective, current.total_bits,
                        current.distortion):
                    current = candidate
                    changed = True
                else:
                    factor[row, column] ^= np.uint8(1)
        if not changed:
            break
    return GF2Plan(
        current.rank, current.u0, current.v0, current.u1, current.v1,
        current.exceptions, current.labels, current.factor_bits,
        current.total_bits, current.distortion, current.objective, False,
        evaluated)


def _matrix_array(np: Any, value: Sequence[Sequence[int]],
                  shape: tuple[int, int]) -> Any:
    result = np.asarray(value, dtype=np.uint8)
    require(result.size == shape[0] * shape[1], "serialized GF2 matrix size")
    result = result.reshape(shape)
    require(np.all(result < 2), "serialized GF2 matrix")
    return result


def write_gf2(writer: BitWriter, np: Any, rows: int, cols: int,
              plan: GF2Plan) -> None:
    rank = plan.rank
    matrices = (
        _matrix_array(np, plan.u0, (rows, rank)),
        _matrix_array(np, plan.v0, (rank, cols)),
        _matrix_array(np, plan.u1, (rows, rank)),
        _matrix_array(np, plan.v1, (rank, cols)),
    )
    require(matrices[0].shape == (rows, rank) and
            matrices[1].shape == (rank, cols) and
            matrices[2].shape == (rows, rank) and
            matrices[3].shape == (rank, cols), "GF2 write factor shapes")
    for matrix in matrices:
        for value in matrix.reshape(-1):
            writer.write(int(value), 1)
    base = _factor_base_labels(np, matrices)
    write_exceptions(writer, rows * cols,
                     tuple(int(value) for value in base), plan.exceptions)
    writer.pad_to_byte()


def read_gf2(reader: BitReader, np: Any, rows: int, cols: int,
             rank: int) -> tuple[int, ...]:
    require(0 <= rank <= min(rows, cols), "GF2 decoded rank")
    shapes = ((rows, rank), (rank, cols), (rows, rank), (rank, cols))
    matrices = []
    for shape in shapes:
        values = [reader.read(1) for _ in range(shape[0] * shape[1])]
        matrices.append(np.asarray(values, dtype=np.uint8).reshape(shape))
    base = _factor_base_labels(np, matrices)
    labels = read_exceptions(reader, rows * cols,
                             tuple(int(value) for value in base))
    reader.align_to_byte()
    return labels


def encode_gf2_component(np: Any, values: Any, weights: Any, *, role: str,
                         rows: int, cols: int, block_size: int,
                         lambda_per_bit: float, ranks: Sequence[int],
                         profile: int | None = None,
                         exception_limit: int | None = None,
                         exact_factor_pair_max: int = 65536,
                         heuristic_sweeps: int = 2) -> EncodedComponent:
    canonical_role_shape(role, rows, cols)
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    rank_bank = tuple(sorted(set(int(value) for value in ranks)))
    require(source.size == rows * cols and source.size % block_size == 0 and
            rank_bank and all(0 <= value <= min(rows, cols) for value in rank_bank),
            "GF2 component geometry")
    candidates = range(len(PROFILE_RATIOS)) if profile is None else (profile,)
    best = None
    for selected_profile in candidates:
        scales = fit_scales(np, source, block_size, int(selected_profile))
        costs, reconstruction = distortion_costs(
            np, source, weights, block_size, int(selected_profile), scales)
        for rank in rank_bank:
            plan = search_gf2(
                np, costs, rows, cols, rank, lambda_per_bit, exception_limit,
                exact_factor_pair_max, heuristic_sweeps)
            key = (plan.objective, plan.total_bits, plan.distortion,
                   int(selected_profile), rank)
            if best is None or key < best[0]:
                best = (key, int(selected_profile), scales, costs,
                        reconstruction, plan)
    require(best is not None, "GF2 component candidate")
    _, selected_profile, scales, costs, reconstruction, plan = best
    writer = BitWriter()
    write_gf2(writer, np, rows, cols, plan)
    return _build_component(
        np, family=FAMILY_GF2, role=role, rows=rows, cols=cols,
        block_size=block_size, parameter=plan.rank, profile=selected_profile,
        scales=scales, writer=writer, labels=plan.labels, values=source,
        weights=weights, costs=costs, reconstruction=reconstruction,
        exact_search=plan.exact_search,
        diagnostics={
            "rank": plan.rank,
            "factor_bits": plan.factor_bits,
            "factor_packet": "literal raw U,V per Gray bitplane",
            "exceptions": plan.exceptions.count,
            "candidates_evaluated": plan.candidates_evaluated,
            "global_negative_authority": plan.exact_search,
            "ideal_rank_matrix_serializer_used": False,
            "model_selector_charged_in_header": True,
        })


def _odd_prime_factors(value: int) -> list[int]:
    require(value >= 1, "factor value")
    result: list[int] = []
    candidate = 3
    residual = value
    while candidate * candidate <= residual:
        while residual % candidate == 0:
            result.append(candidate)
            residual //= candidate
        candidate += 2
    if residual > 1:
        result.append(residual)
    return result


def dimension_radices(value: int) -> tuple[int, ...]:
    """Exact mixed-radix domain: odd factors first, then binary digits."""
    require(value >= 1, "dimension radix value")
    twos = 0
    odd = value
    while odd % 2 == 0:
        twos += 1
        odd //= 2
    radices = _odd_prime_factors(odd) + [2] * twos
    if not radices:
        radices = [1]
    require(math.prod(radices) == value, "dimension radix product")
    return tuple(radices)


def coordinate_radices(rows: int, cols: int) -> tuple[int, ...]:
    require(rows > 0 and cols > 0, "coordinate dimensions")
    row_digits = tuple(value for value in dimension_radices(rows) if value > 1)
    col_digits = tuple(value for value in dimension_radices(cols) if value > 1)
    result = row_digits + col_digits
    if not result:
        result = (1,)
    require(math.prod(result) == rows * cols, "coordinate exact domain")
    return result


def qwen_coordinate_domain_record() -> dict[str, Any]:
    rows, cols, roles = 768, 2048, 3
    radices = coordinate_radices(rows, cols)
    valid = roles * rows * cols
    naive = 1 << 23
    return {
        "rows": rows,
        "cols": cols,
        "roles": roles,
        "role_radices": list(radices),
        "digits_per_role": len(radices),
        "sites_per_role": math.prod(radices),
        "valid_expert_sites": valid,
        "naive_23bit_sites": naive,
        "invalid_or_unused_naive_sites": naive - valid,
        "valid_fraction_of_naive_domain": valid / naive,
        "invalid_mask_transmitted_or_assumed_free": False,
        "roles_are_separate_component_packets": True,
    }


@dataclass(frozen=True)
class ROMDDNode:
    level: int
    children: tuple[int, ...]


@dataclass(frozen=True)
class ROMDDPlan:
    depth: int
    radices: tuple[int, ...]
    nodes: tuple[ROMDDNode, ...]
    root: int
    exceptions: ExceptionPlan
    labels: tuple[int, ...]
    diagram_bits: int
    total_bits: int
    distortion: float
    objective: float


def _romdd_base(np: Any, costs: Any, rows: int, cols: int,
                depth: int) -> tuple[tuple[int, ...], tuple[ROMDDNode, ...], int]:
    matrix = np.asarray(costs, dtype=np.float64)
    radices = coordinate_radices(rows, cols)
    require(matrix.shape == (rows * cols, 4) and 0 <= depth <= len(radices),
            "ROMDD base geometry")
    tensor = matrix.reshape(radices + (4,))
    nodes: list[ROMDDNode] = []
    cache: dict[tuple[int, tuple[int, ...]], int] = {}

    def build(subtensor: Any, level: int) -> int:
        if level == depth or level == len(radices):
            axes = tuple(range(subtensor.ndim - 1))
            totals = np.sum(subtensor, axis=axes, dtype=np.float64)
            return min(range(4), key=lambda label: (float(totals[label]), label))
        children = tuple(build(np.take(subtensor, digit, axis=0), level + 1)
                         for digit in range(radices[level]))
        if len(set(children)) == 1:
            return children[0]
        key = (level, children)
        if key in cache:
            return cache[key]
        node_id = 4 + len(nodes)
        nodes.append(ROMDDNode(level, children))
        cache[key] = node_id
        return node_id

    root = build(tensor, 0)
    decoded = romdd_labels(radices, tuple(nodes), root)
    require(len(decoded) == rows * cols, "ROMDD base decode size")
    return decoded, tuple(nodes), root


def romdd_diagram_bits(radices: Sequence[int], nodes: Sequence[ROMDDNode],
                       root: int) -> int:
    levels = len(radices)
    node_count = len(nodes)
    reference_width = ceil_log2_count(4 + node_count)
    level_width = ceil_log2_count(max(1, levels))
    require(0 <= root < 4 + node_count, "ROMDD root")
    bits = 32 + reference_width
    for node in nodes:
        require(0 <= node.level < levels and
                len(node.children) == radices[node.level], "ROMDD node shape")
        bits += level_width + len(node.children) * reference_width
    return bits


def romdd_labels(radices: Sequence[int], nodes: Sequence[ROMDDNode],
                 root: int) -> tuple[int, ...]:
    radices_tuple = tuple(int(value) for value in radices)
    require(radices_tuple and all(value >= 1 for value in radices_tuple),
            "ROMDD radices")
    node_tuple = tuple(nodes)
    require(0 <= root < 4 + len(node_tuple), "ROMDD root bounds")
    for index, node in enumerate(node_tuple):
        node_id = 4 + index
        require(0 <= node.level < len(radices_tuple) and
                len(node.children) == radices_tuple[node.level] and
                all(0 <= child < node_id for child in node.children),
                "ROMDD topological node")
        for child in node.children:
            if child >= 4:
                require(node_tuple[child - 4].level > node.level,
                        "ROMDD ordered levels")
    labels: list[int] = []
    # Mixed-radix odometer in C-order; no padded truth-table sites exist.
    count = math.prod(radices_tuple)
    for flat in range(count):
        residual = flat
        digits = [0] * len(radices_tuple)
        for level in range(len(radices_tuple) - 1, -1, -1):
            digits[level] = residual % radices_tuple[level]
            residual //= radices_tuple[level]
        reference = root
        while reference >= 4:
            node = node_tuple[reference - 4]
            reference = node.children[digits[node.level]]
        labels.append(reference)
    return tuple(labels)


def search_romdd(np: Any, costs: Any, rows: int, cols: int,
                 depths: Sequence[int], lambda_per_bit: float,
                 exception_limit: int | None = None) -> ROMDDPlan:
    radices = coordinate_radices(rows, cols)
    depth_bank = tuple(sorted(set(int(value) for value in depths)))
    require(depth_bank and all(0 <= value <= len(radices) for value in depth_bank)
            and lambda_per_bit >= 0.0, "ROMDD search parameters")
    best: ROMDDPlan | None = None
    for depth in depth_bank:
        base, nodes, root = _romdd_base(np, costs, rows, cols, depth)
        exceptions = optimize_exceptions(
            np, costs, base, lambda_per_bit, exception_limit,
            fixed_prefix_bits=romdd_diagram_bits(radices, nodes, root),
            byte_align_total=True)
        labels = list(base)
        for position, label in zip(exceptions.positions, exceptions.labels):
            labels[position] = label
        diagram_bits = romdd_diagram_bits(radices, nodes, root)
        total_bits = exceptions.charged_total_bits
        distortion = exceptions.distortion
        objective = exceptions.objective
        candidate = ROMDDPlan(
            depth, radices, nodes, root, exceptions, tuple(labels),
            diagram_bits, total_bits, distortion, objective)
        key = (candidate.objective, candidate.total_bits,
               candidate.distortion, candidate.depth, len(candidate.nodes),
               candidate.root)
        if best is None or key < (best.objective, best.total_bits,
                                  best.distortion, best.depth,
                                  len(best.nodes), best.root):
            best = candidate
    require(best is not None, "ROMDD candidate")
    return best


def write_romdd(writer: BitWriter, plan: ROMDDPlan) -> None:
    node_count = len(plan.nodes)
    reference_width = ceil_log2_count(4 + node_count)
    level_width = ceil_log2_count(max(1, len(plan.radices)))
    writer.write(node_count, 32)
    writer.write(plan.root, reference_width)
    for index, node in enumerate(plan.nodes):
        writer.write(node.level, level_width)
        for child in node.children:
            require(child < 4 + index, "ROMDD write topological reference")
            writer.write(child, reference_width)
    base = romdd_labels(plan.radices, plan.nodes, plan.root)
    write_exceptions(writer, len(base), base, plan.exceptions)
    writer.pad_to_byte()


def read_romdd(reader: BitReader, rows: int, cols: int) -> tuple[int, ...]:
    radices = coordinate_radices(rows, cols)
    node_count = reader.read(32)
    # Defensive finite decoder bound: a packet cannot contain more nodes than
    # its source sites times its digit count.
    require(node_count <= rows * cols * len(radices), "ROMDD node count bound")
    reference_width = ceil_log2_count(4 + node_count)
    level_width = ceil_log2_count(max(1, len(radices)))
    root = reader.read(reference_width)
    require(root < 4 + node_count, "ROMDD decoded root")
    nodes: list[ROMDDNode] = []
    for index in range(node_count):
        level = reader.read(level_width)
        require(level < len(radices), "ROMDD decoded level")
        children = tuple(reader.read(reference_width)
                         for _ in range(radices[level]))
        require(all(child < 4 + index for child in children),
                "ROMDD decoded topological reference")
        nodes.append(ROMDDNode(level, children))
    base = romdd_labels(radices, tuple(nodes), root)
    labels = read_exceptions(reader, rows * cols, base)
    reader.align_to_byte()
    return labels


def encode_romdd_component(np: Any, values: Any, weights: Any, *, role: str,
                           rows: int, cols: int, block_size: int,
                           lambda_per_bit: float, depths: Sequence[int],
                           profile: int | None = None,
                           exception_limit: int | None = None) -> EncodedComponent:
    canonical_role_shape(role, rows, cols)
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    require(source.size == rows * cols and source.size % block_size == 0,
            "ROMDD component geometry")
    candidates = range(len(PROFILE_RATIOS)) if profile is None else (profile,)
    best = None
    for selected_profile in candidates:
        scales = fit_scales(np, source, block_size, int(selected_profile))
        costs, reconstruction = distortion_costs(
            np, source, weights, block_size, int(selected_profile), scales)
        plan = search_romdd(
            np, costs, rows, cols, depths, lambda_per_bit, exception_limit)
        key = (plan.objective, plan.total_bits, plan.distortion,
               int(selected_profile), plan.depth)
        if best is None or key < best[0]:
            best = (key, int(selected_profile), scales, costs,
                    reconstruction, plan)
    require(best is not None, "ROMDD component profile")
    _, selected_profile, scales, costs, reconstruction, plan = best
    writer = BitWriter()
    write_romdd(writer, plan)
    return _build_component(
        np, family=FAMILY_ROMDD, role=role, rows=rows, cols=cols,
        block_size=block_size, parameter=plan.depth,
        profile=selected_profile, scales=scales, writer=writer,
        labels=plan.labels, values=source, weights=weights, costs=costs,
        reconstruction=reconstruction, exact_search=False,
        diagnostics={
            "mixed_radix_domain": list(plan.radices),
            "exact_domain_sites": math.prod(plan.radices),
            "depth": plan.depth,
            "nodes": len(plan.nodes),
            "diagram_bits": plan.diagram_bits,
            "exceptions": plan.exceptions.count,
            "invalid_padded_coordinate_mask_used": False,
            "global_negative_authority": False,
            "model_selector_charged_in_header": True,
        })


def decode_component(np: Any, packet: bytes) -> tuple[tuple[int, ...],
                                                       tuple[float, ...],
                                                       ComponentHeaderRecord]:
    record, scales, payload = parse_component_envelope(packet)
    reader = BitReader(payload, record.payload_bits)
    if record.family == FAMILY_LITERAL:
        require(record.parameter == 0 and record.payload_bits == 2 * record.source_count,
                "literal payload contract")
        labels = tuple(reader.read(2) for _ in range(record.source_count))
    elif record.family == FAMILY_RM1:
        require(record.parameter == 1 and is_power_of_two(record.block_size),
                "RM payload contract")
        decoded: list[int] = []
        for _ in range(record.blocks):
            decoded.extend(read_rm1_block(reader, np, record.block_size))
        labels = tuple(decoded)
    elif record.family == FAMILY_GF2:
        labels = read_gf2(reader, np, record.rows, record.cols, record.parameter)
    elif record.family == FAMILY_ROMDD:
        require(record.parameter <= len(coordinate_radices(record.rows, record.cols)),
                "ROMDD depth parameter")
        labels = read_romdd(reader, record.rows, record.cols)
    else:  # pragma: no cover - protected by header parser
        raise LogicQError("unknown component family")
    reader.finish()
    require(len(labels) == record.source_count and all(0 <= value < 4 for value in labels),
            "decoded label closure")
    ratios = PROFILE_RATIOS[record.profile]
    values: list[float] = []
    for block_index in range(record.blocks):
        scale = bf16_bits_to_float(scales[block_index])
        levels = tuple(scale * ratio for ratio in ratios)
        start = block_index * record.block_size
        for label in labels[start:start + record.block_size]:
            values.append(levels[label])
    require(len(values) == record.source_count, "decoded numeric closure")
    return labels, tuple(values), record


def pack_expert(components: dict[str, bytes]) -> bytes:
    require(set(components) == set(ROLE_IDS), "exact expert roles")
    packets = []
    lengths = []
    for role in ("gate", "up", "down_transposed"):
        packet = bytes(components[role])
        record, _, _ = parse_component_envelope(packet)
        require(record.role == role, "expert role/header binding")
        packets.append(packet)
        lengths.append(len(packet))
    header = EXPERT_HEADER.pack(
        EXPERT_MAGIC, 0, 3, 0, lengths[0], lengths[1], lengths[2], b"\0" * 28)
    output = bytearray(header)
    for packet in packets:
        output.extend(packet)
        output.extend(b"\0" * (align_up(len(output), COMPONENT_ALIGNMENT) - len(output)))
    output.extend(b"\0" * (align_up(len(output), EXPERT_PAGE) - len(output)))
    return bytes(output)


def unpack_expert(np: Any, packet: bytes) -> dict[str, tuple[tuple[int, ...],
                                                              tuple[float, ...],
                                                              ComponentHeaderRecord]]:
    require(len(packet) >= EXPERT_PAGE and len(packet) % EXPERT_PAGE == 0,
            "expert page closure")
    (magic, version, count, reserved16, gate_length, up_length, down_length,
     reserved) = EXPERT_HEADER.unpack(packet[:EXPERT_HEADER_BYTES])
    require(magic == EXPERT_MAGIC and version == 0 and count == 3 and
            reserved16 == 0 and reserved == b"\0" * 28,
            "expert header")
    lengths = (gate_length, up_length, down_length)
    cursor = EXPERT_HEADER_BYTES
    result = {}
    for role, length in zip(("gate", "up", "down_transposed"), lengths):
        require(length >= COMPONENT_HEADER_BYTES and cursor + length <= len(packet),
                "expert component span")
        component = packet[cursor:cursor + length]
        decoded = decode_component(np, component)
        require(decoded[2].role == role, "expert decoded role")
        result[role] = decoded
        cursor = align_up(cursor + length, COMPONENT_ALIGNMENT)
    require(packet[cursor:] == b"\0" * (len(packet) - cursor),
            "expert final zero padding")
    return result


def expert_ledger(components: dict[str, bytes]) -> dict[str, Any]:
    packet = pack_expert(components)
    spans = {}
    cursor = EXPERT_HEADER_BYTES
    for role in ("gate", "up", "down_transposed"):
        length = len(components[role])
        spans[role] = {
            "payload_bytes": length,
            "start_byte": cursor,
            "end_byte_exclusive": cursor + length,
            "aligned_end_byte_exclusive": align_up(cursor + length,
                                                     COMPONENT_ALIGNMENT),
        }
        cursor = spans[role]["aligned_end_byte_exclusive"]
    return {
        "expert_header_bytes": EXPERT_HEADER_BYTES,
        "component_alignment_bytes": COMPONENT_ALIGNMENT,
        "final_page_bytes": EXPERT_PAGE,
        "components": spans,
        "pre_page_bytes": cursor,
        "physical_expert_bytes": len(packet),
        "final_page_padding_bytes": len(packet) - cursor,
        "all_bytes_charged": True,
    }


def family_minimum_payload_bits(family: int, rows: int, cols: int,
                                block_size: int, parameter: int) -> int:
    n = rows * cols
    require(n > 0 and n % block_size == 0, "minimum payload geometry")
    blocks = n // block_size
    if family == FAMILY_LITERAL:
        require(parameter == 0, "literal minimum parameter")
        return 2 * n
    if family == FAMILY_RM1:
        require(parameter == 1 and is_power_of_two(block_size),
                "RM minimum parameter")
        message_bits = 2 * block_size.bit_length()
        zero_exception_bits = exception_bits(block_size, 0)["total_bits"]
        return blocks * align_up(message_bits + zero_exception_bits, 8)
    if family == FAMILY_GF2:
        require(0 <= parameter <= min(rows, cols), "GF2 minimum rank")
        return (raw_gf2_factor_bits(rows, cols, parameter)
                + exception_bits(n, 0)["total_bits"])
    if family == FAMILY_ROMDD:
        require(0 <= parameter <= len(coordinate_radices(rows, cols)),
                "ROMDD minimum depth")
        # One terminal root: uint32 node count, 2-bit terminal reference.
        return 32 + 2 + exception_bits(n, 0)["total_bits"]
    raise LogicQError("minimum unknown family")


def family_minimum_component_bits(family: int, rows: int, cols: int,
                                  block_size: int, parameter: int) -> int:
    payload_bits = family_minimum_payload_bits(
        family, rows, cols, block_size, parameter)
    scale_bytes = 2 * (rows * cols // block_size)
    return 8 * (COMPONENT_HEADER_BYTES + scale_bytes + ceil_div(payload_bits, 8))


def optimistic_family_bound(*, family: int, rows: int, cols: int,
                            block_size: int, parameter: int,
                            nearest_weighted_sse: float,
                            source_energy: float,
                            baseline_physical_bits: int | None = None) -> dict[str, Any]:
    n = rows * cols
    require(nearest_weighted_sse >= 0.0 and source_energy > 0.0 and
            math.isfinite(nearest_weighted_sse) and math.isfinite(source_energy),
            "optimistic bound distortion")
    minimum_bits = family_minimum_component_bits(
        family, rows, cols, block_size, parameter)
    minimum_rate = minimum_bits / n
    padded_rate = max(RATE_MIN, minimum_rate)
    lower_F = (nearest_weighted_sse / source_energy
               * 2.0 ** (2.0 * padded_rate))
    result = {
        "family": FAMILY_NAMES[family],
        "weights": n,
        "minimum_component_physical_bits": minimum_bits,
        "minimum_rate_bpw": minimum_rate,
        "evaluation_rate_after_required_2p15_padding_bpw": padded_rate,
        "nearest_label_relative_mse": nearest_weighted_sse / source_energy,
        "optimistic_F_lower_bound": lower_F,
        "mandatory_packet_exceeds_2p5_bpw": minimum_rate > RATE_MAX,
        "optimistic_F_still_exceeds_target": lower_F > TARGET_F,
        "bound_assumes_incompatible_best_rate_and_nearest_distortion": True,
    }
    if baseline_physical_bits is not None:
        require(baseline_physical_bits >= 0, "nonnegative baseline bits")
        saving = (baseline_physical_bits - minimum_bits) / n
        result.update({
            "baseline_physical_bits": baseline_physical_bits,
            "unchanged_mse_max_saving_bpw": saving,
            "unchanged_mse_required_saving_bpw": REQUIRED_SAVING_BPW,
            "unchanged_mse_saving_bound_misses_required":
                saving < REQUIRED_SAVING_BPW,
        })
    if minimum_rate > RATE_MAX:
        status = "HARD_KILL_MANDATORY_PACKET_EXCEEDS_2P5_BPW"
    elif (baseline_physical_bits is not None and
          result["unchanged_mse_saving_bound_misses_required"]):
        status = "RATE_ONLY_HARD_KILL_UNCHANGED_MSE_SAVING_BOUND"
    elif lower_F > TARGET_F:
        status = "HARD_KILL_OPTIMISTIC_F_LOWER_BOUND_EXCEEDS_0P8"
    else:
        status = "SURVIVES_DESCRIPTOR_AND_OPTIMISTIC_DISTORTION_BOUND"
    result["status"] = status
    return result


def maximum_raw_gf2_rank_under_rate(rows: int, cols: int, block_size: int,
                                    rate_bpw: float) -> int:
    require(rate_bpw >= 0.0 and rows > 0 and cols > 0 and
            rows * cols % block_size == 0, "GF2 rank budget")
    n = rows * cols
    cap = math.floor(rate_bpw * n)
    feasible = [rank for rank in range(min(rows, cols) + 1)
                if family_minimum_component_bits(
                    FAMILY_GF2, rows, cols, block_size, rank) <= cap]
    return max(feasible) if feasible else -1


def rm_dimension(order: int, variables: int) -> int:
    require(0 <= order <= variables, "RM dimension parameters")
    return sum(math.comb(variables, degree) for degree in range(order + 1))


def rm_descriptor_controls() -> dict[str, Any]:
    valid_weights = 3 * 768 * 2048
    global_k = rm_dimension(3, 23)
    block_rows = []
    for block_size in (128, 256, 512, 1024, 2048):
        variables = block_size.bit_length() - 1
        blocks = valid_weights // block_size
        for order in (1, 2, 3):
            k = rm_dimension(order, variables)
            bits = blocks * 2 * k
            block_rows.append({
                "block_size": block_size,
                "order": order,
                "variables": variables,
                "dimension_per_bitplane": k,
                "blocks_all_three_roles": blocks,
                "descriptor_bits_both_planes": bits,
                "descriptor_bpw_before_headers_exceptions": bits / valid_weights,
            })
    return {
        "global_rm3_23_dimension_per_bitplane": global_k,
        "global_rm3_23_descriptor_bits_both_planes": 2 * global_k,
        "global_descriptor_bpw_over_valid_expert_weights":
            2 * global_k / valid_weights,
        "global_domain": qwen_coordinate_domain_record(),
        "global_code_is_punctured_or_mixed_radix_on_valid_sites": True,
        "dense_23bit_truth_table_mask_assumed_free": False,
        "random_plane_expected_to_be_near_half_hamming_distance": True,
        "matched_gaussian_control_required": True,
        "blockwise_descriptor_rows": block_rows,
    }


def exact_budget_identities() -> dict[str, Any]:
    return {
        "reference_F": REFERENCE_F,
        "required_saving_bpw": REQUIRED_SAVING_BPW,
        "required_mse_reduction_fraction": REQUIRED_MSE_REDUCTION,
        "target_F": TARGET_F,
        "rate_interval": [RATE_MIN, RATE_MAX],
        "required_saving_fraction_exact_expression":
            "-0.5*log2(0.8/(0.030902167403153148*2^5))",
    }
