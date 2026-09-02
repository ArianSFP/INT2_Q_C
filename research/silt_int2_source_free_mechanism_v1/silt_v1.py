#!/usr/bin/env python3
"""SILT source-free finite mechanism v1.

This module accepts in-memory finite label arrays only.  It has no filesystem
input path and no model/payload authority.  All hostile scalar and product
bounds are checked before factorials, loops, slices, conversions, or decoded
allocations.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Sequence

import numpy as np


SCHEMA = "silt-int2-source-free-mechanism-v1"
VERSION = 1
PAGE_BYTES = 4096
GLOBAL_HEADER_BYTES = 4096
DIRECTORY_HEADER_BYTES = 256
DIRECTORY_ENTRY_BYTES = 32
MODEL_HEADER_BYTES = 256
FRAME_HEADER_BYTES = 256
Q16_TOTAL = 1 << 16
STATE_COUNT = 64
DETAIL_BLOCK = 32
ARITHMETIC_GUARD_BITS = 30

MAX_EXPERTS = 256
MAX_LANES = 2048
MAX_VECTORS = 1 << 20
MAX_SYMBOLS_PER_EXPERT = 1 << 24
MAX_TOTAL_SYMBOLS = 1 << 28
MAX_MODEL_PACKET_BYTES = 1 << 20
MAX_DIRECTORY_PACKET_BYTES = 1 << 16
MAX_FRAME_LOGICAL_BYTES = 1 << 26
MAX_FRAME_PADDED_BYTES = MAX_FRAME_LOGICAL_BYTES + PAGE_BYTES
MAX_CONTAINER_BYTES = 1 << 28

GLOBAL_MAGIC = b"SILTV1G\0"
DIRECTORY_MAGIC = b"SILTV1D\0"
MODEL_MAGIC = b"SILTV1M\0"
FRAME_MAGIC = b"SILTV1F\0"

GLOBAL_STRUCT = struct.Struct("<8sHHIIIIQQQQQQQQQ32s32sI")
DIRECTORY_HEADER_STRUCT = struct.Struct("<8sHHIII32sI")
DIRECTORY_ENTRY_STRUCT = struct.Struct("<QQQQ")
MODEL_STRUCT = struct.Struct("<8sHHHHII")
FRAME_STRUCT = struct.Struct("<8sHHIIIIIIQQQQQ32sI")


class FormatError(RuntimeError):
    """A v1 format, resource, or canonicality invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FormatError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checked_scalar(value: int, minimum: int, maximum: int, name: str) -> int:
    require(isinstance(value, (int, np.integer)), f"{name} integer")
    result = int(value)
    require(minimum <= result <= maximum, f"{name} range")
    return result


def checked_add(left: int, right: int, maximum: int, name: str) -> int:
    left = checked_scalar(left, 0, maximum, f"{name} left")
    right = checked_scalar(right, 0, maximum, f"{name} right")
    require(left <= maximum - right, f"{name} overflow")
    return left + right


def checked_mul(left: int, right: int, maximum: int, name: str) -> int:
    left = checked_scalar(left, 0, maximum, f"{name} left")
    right = checked_scalar(right, 0, maximum, f"{name} right")
    require(left == 0 or right <= maximum // left, f"{name} overflow")
    return left * right


def checked_slice(total: int, offset: int, length: int, maximum: int, name: str) -> tuple[int, int]:
    total = checked_scalar(total, 0, maximum, f"{name} total")
    offset = checked_scalar(offset, 0, maximum, f"{name} offset")
    length = checked_scalar(length, 0, maximum, f"{name} length")
    end = checked_add(offset, length, maximum, f"{name} end")
    require(end <= total, f"{name} bounds")
    return offset, end


def page_ceil(value: int) -> int:
    value = checked_scalar(value, 0, MAX_CONTAINER_BYTES, "page value")
    if value == 0:
        return 0
    result = checked_mul((value + PAGE_BYTES - 1) // PAGE_BYTES, PAGE_BYTES, MAX_CONTAINER_BYTES, "page ceil")
    return result


def validate_alphabet(alphabet: int) -> int:
    alphabet = int(alphabet)
    require(alphabet in (2, 4), "alphabet must be GF2 or Z4")
    return alphabet


def validate_expert_count(experts: int) -> int:
    return checked_scalar(experts, 1, MAX_EXPERTS, "expert count")


def validate_lanes(lanes: int) -> int:
    return checked_scalar(lanes, 1, MAX_LANES, "lanes")


def validate_vectors(vectors: int) -> int:
    return checked_scalar(vectors, 1, MAX_VECTORS, "vectors")


def validate_geometry(lanes: int, vectors: int) -> int:
    lanes = validate_lanes(lanes)
    vectors = validate_vectors(vectors)
    return checked_mul(lanes, vectors, MAX_SYMBOLS_PER_EXPERT, "expert symbols")


def checks_for_alphabet(alphabet: int) -> int:
    return 6 if validate_alphabet(alphabet) == 2 else 3


def context_count(alphabet: int) -> int:
    return 1 + 2 * checks_for_alphabet(alphabet)


def canonical_selector_count(alphabet: int) -> int:
    return 6 if validate_alphabet(alphabet) == 2 else 8


def validate_selector(value: int, alphabet: int) -> int:
    value = int(value)
    require(0 <= value < canonical_selector_count(alphabet), "canonical selector ID")
    return value


def level_sizes(lanes: int) -> list[int]:
    lanes = validate_lanes(lanes)
    sizes = [lanes]
    while sizes[-1] > 1:
        sizes.append((sizes[-1] + 1) // 2)
    return sizes


def level_pair_counts(lanes: int) -> list[int]:
    sizes = level_sizes(lanes)
    counts = [value // 2 for value in sizes[:-1]]
    require(sum(counts) == sizes[0] - 1, "tree node count")
    return counts


@lru_cache(maxsize=MAX_LANES)
def permutation_byte_count(lanes: int) -> int:
    lanes = validate_lanes(lanes)  # cap is mandatory before factorial
    states = math.factorial(lanes)
    count = ((states - 1).bit_length() + 7) // 8
    require(count <= MAX_FRAME_LOGICAL_BYTES, "permutation byte cap")
    return count


def validate_permutation(permutation: Sequence[int], lanes: int | None = None) -> None:
    n = validate_lanes(len(permutation) if lanes is None else lanes)
    require(len(permutation) == n, "permutation length")
    require(sorted(int(value) for value in permutation) == list(range(n)), "permutation domain")


def rank_permutation(permutation: Sequence[int]) -> int:
    validate_permutation(permutation)
    n = len(permutation)
    available = list(range(n))
    rank = 0
    for position, value in enumerate(permutation):
        index = available.index(int(value))
        rank += index * math.factorial(n - position - 1)
        available.pop(index)
    require(0 <= rank < math.factorial(n), "factoradic rank")
    return rank


def unrank_permutation(lanes: int, rank: int) -> list[int]:
    lanes = validate_lanes(lanes)
    states = math.factorial(lanes)
    rank = checked_scalar(rank, 0, states - 1, "factoradic rank")
    available = list(range(lanes))
    output: list[int] = []
    for width in range(lanes, 0, -1):
        factorial = math.factorial(width - 1)
        index, rank = divmod(rank, factorial)
        output.append(available.pop(index))
    validate_permutation(output, lanes)
    return output


def serialize_permutation(permutation: Sequence[int]) -> bytes:
    validate_permutation(permutation)
    count = permutation_byte_count(len(permutation))
    rank = rank_permutation(permutation)
    return rank.to_bytes(count, "big") if count else b""


def deserialize_permutation(lanes: int, packet: bytes) -> list[int]:
    lanes = validate_lanes(lanes)
    expected = permutation_byte_count(lanes)
    require(len(packet) == expected, "factoradic packet length")
    rank = int.from_bytes(packet, "big") if packet else 0
    return unrank_permutation(lanes, rank)


def pack_selectors(selectors: Sequence[int], alphabet: int) -> bytes:
    alphabet = validate_alphabet(alphabet)
    values = [validate_selector(value, alphabet) for value in selectors]
    bit_count = checked_mul(3, len(values), MAX_FRAME_LOGICAL_BYTES * 8, "selector bits")
    packet = bytearray((bit_count + 7) // 8)
    for index, value in enumerate(values):
        for offset in range(3):
            bit_index = 3 * index + offset
            packet[bit_index // 8] |= ((value >> (2 - offset)) & 1) << (7 - (bit_index & 7))
    return bytes(packet)


def unpack_selectors(packet: bytes, count: int, alphabet: int) -> list[int]:
    alphabet = validate_alphabet(alphabet)
    count = checked_scalar(count, 0, MAX_LANES - 1, "selector count")
    require(len(packet) == (3 * count + 7) // 8, "selector packet length")
    values: list[int] = []
    for index in range(count):
        value = 0
        for offset in range(3):
            bit_index = 3 * index + offset
            value = (value << 1) | ((packet[bit_index // 8] >> (7 - (bit_index & 7))) & 1)
        values.append(validate_selector(value, alphabet))
    for bit_index in range(3 * count, 8 * len(packet)):
        require(((packet[bit_index // 8] >> (7 - (bit_index & 7))) & 1) == 0, "selector zero tail")
    return values


def deterministic_permutation(lanes: int, seed: int) -> list[int]:
    lanes = validate_lanes(lanes)
    return [int(value) for value in np.random.default_rng(seed).permutation(lanes)]


def deterministic_selectors(lanes: int, alphabet: int, seed: int) -> list[int]:
    lanes = validate_lanes(lanes)
    high = canonical_selector_count(alphabet)
    return [int(value) for value in np.random.default_rng(seed).integers(0, high, size=lanes - 1, dtype=np.uint8)]


@dataclass(frozen=True)
class LiftedTensor:
    roots: np.ndarray
    detail_levels: tuple[np.ndarray, ...]

    def validate(self, vectors: int, lanes: int, alphabet: int) -> None:
        validate_geometry(lanes, vectors)
        alphabet = validate_alphabet(alphabet)
        require(self.roots.shape == (vectors,) and self.roots.dtype == np.uint8, "root geometry/dtype")
        require(len(self.detail_levels) == len(level_pair_counts(lanes)), "detail level count")
        for level, pairs in zip(self.detail_levels, level_pair_counts(lanes), strict=True):
            require(level.shape == (vectors, pairs) and level.dtype == np.uint8, "detail geometry/dtype")
            require(bool(np.all(level < alphabet)), "detail alphabet")
        require(bool(np.all(self.roots < alphabet)), "root alphabet")


def lift_forward(
    leaves: np.ndarray,
    alphabet: int,
    permutation: Sequence[int],
    selectors: Sequence[int],
) -> LiftedTensor:
    alphabet = validate_alphabet(alphabet)
    require(isinstance(leaves, np.ndarray) and leaves.ndim == 2, "leaf matrix")
    vectors, lanes = (int(value) for value in leaves.shape)
    validate_geometry(lanes, vectors)
    require(leaves.dtype == np.uint8 and bool(np.all(leaves < alphabet)), "leaf dtype/alphabet")
    validate_permutation(permutation, lanes)
    require(len(selectors) == lanes - 1, "selector count")
    for selector in selectors:
        validate_selector(selector, alphabet)
    current = np.ascontiguousarray(leaves[:, np.asarray(permutation, dtype=np.int64)])
    levels: list[np.ndarray] = []
    selector_offset = 0
    for pairs in level_pair_counts(lanes):
        codes = np.asarray(selectors[selector_offset : selector_offset + pairs], dtype=np.uint8)
        selector_offset += pairs
        left = current[:, 0 : 2 * pairs : 2].astype(np.int16)
        right = current[:, 1 : 2 * pairs : 2].astype(np.int16)
        swap = (((codes >> 2) & 1).astype(bool))[None, :]
        p = (((codes >> 1) & 1).astype(np.int16))[None, :]
        u = ((codes & 1).astype(np.int16))[None, :]
        x = np.where(swap, right, left)
        y = np.where(swap, left, right)
        detail = np.mod(y - p * x, alphabet).astype(np.uint8)
        coarse = np.mod(x + u * detail.astype(np.int16), alphabet).astype(np.uint8)
        current = np.concatenate((coarse, current[:, -1:]), axis=1) if current.shape[1] & 1 else coarse
        levels.append(np.ascontiguousarray(detail))
    require(selector_offset == lanes - 1 and current.shape == (vectors, 1), "tree coverage")
    result = LiftedTensor(np.ascontiguousarray(current[:, 0]), tuple(levels))
    result.validate(vectors, lanes, alphabet)
    return result


def lift_inverse(
    lifted: LiftedTensor,
    lanes: int,
    alphabet: int,
    permutation: Sequence[int],
    selectors: Sequence[int],
) -> np.ndarray:
    lanes = validate_lanes(lanes)
    alphabet = validate_alphabet(alphabet)
    vectors = validate_vectors(int(lifted.roots.shape[0]))
    lifted.validate(vectors, lanes, alphabet)
    validate_permutation(permutation, lanes)
    require(len(selectors) == lanes - 1, "inverse selector count")
    for selector in selectors:
        validate_selector(selector, alphabet)
    counts = level_pair_counts(lanes)
    offsets = np.cumsum([0] + counts).tolist()
    widths = level_sizes(lanes)
    current = np.ascontiguousarray(lifted.roots[:, None])
    for depth in range(len(counts) - 1, -1, -1):
        pairs = counts[depth]
        codes = np.asarray(selectors[offsets[depth] : offsets[depth + 1]], dtype=np.uint8)
        detail = lifted.detail_levels[depth].astype(np.int16)
        coarse = current[:, :pairs].astype(np.int16)
        swap = (((codes >> 2) & 1).astype(bool))[None, :]
        p = (((codes >> 1) & 1).astype(np.int16))[None, :]
        u = ((codes & 1).astype(np.int16))[None, :]
        x = np.mod(coarse - u * detail, alphabet)
        y = np.mod(detail + p * x, alphabet)
        left = np.where(swap, y, x).astype(np.uint8)
        right = np.where(swap, x, y).astype(np.uint8)
        previous = np.empty((vectors, widths[depth]), dtype=np.uint8)
        previous[:, 0 : 2 * pairs : 2] = left
        previous[:, 1 : 2 * pairs : 2] = right
        if widths[depth] & 1:
            require(current.shape[1] == pairs + 1, "odd carry geometry")
            previous[:, -1] = current[:, -1]
        else:
            require(current.shape[1] == pairs, "even geometry")
        current = previous
    output = np.empty_like(current)
    output[:, np.asarray(permutation, dtype=np.int64)] = current
    require(output.shape == (vectors, lanes) and bool(np.all(output < alphabet)), "inverse output")
    return output


def flatten_details(lifted: LiftedTensor) -> np.ndarray:
    if not lifted.detail_levels:
        return np.empty(0, dtype=np.uint8)
    return np.ascontiguousarray(
        np.concatenate([np.ascontiguousarray(level).reshape(-1) for level in reversed(lifted.detail_levels)])
    ).astype(np.uint8, copy=False)


def unflatten_details(flat: np.ndarray, vectors: int, lanes: int) -> tuple[np.ndarray, ...]:
    symbols = validate_geometry(lanes, vectors)
    expected = symbols - vectors
    require(flat.ndim == 1 and flat.dtype == np.uint8 and flat.size == expected, "flat detail geometry")
    levels: list[np.ndarray | None] = [None] * len(level_pair_counts(lanes))
    offset = 0
    counts = level_pair_counts(lanes)
    for depth in range(len(counts) - 1, -1, -1):
        size = checked_mul(vectors, counts[depth], MAX_SYMBOLS_PER_EXPERT, "detail level size")
        levels[depth] = np.ascontiguousarray(flat[offset : offset + size]).reshape(vectors, counts[depth])
        offset += size
    require(offset == flat.size, "flat detail coverage")
    return tuple(level for level in levels if level is not None)


def detail_context(alphabet: int, detail_index: int) -> int:
    checks = checks_for_alphabet(alphabet)
    detail_index = checked_scalar(detail_index, 0, MAX_SYMBOLS_PER_EXPERT, "detail index")
    position = detail_index % DETAIL_BLOCK
    body = DETAIL_BLOCK - checks
    return 1 + (position % checks) if position < body else 1 + checks + (position - body)


def state_successor(alphabet: int, state: int, detail_index: int, symbol: int) -> int:
    alphabet = validate_alphabet(alphabet)
    state = checked_scalar(state, 0, STATE_COUNT - 1, "state")
    symbol = checked_scalar(symbol, 0, alphabet - 1, "state symbol")
    position = int(detail_index) % DETAIL_BLOCK
    checks = checks_for_alphabet(alphabet)
    if position >= DETAIL_BLOCK - checks:
        return state
    group = position % checks
    base = alphabet**group
    old = (state // base) % alphabet
    new = (old + symbol) % alphabet
    result = state + (new - old) * base
    require(0 <= result < STATE_COUNT, "successor state")
    return result


def _q16_row(counts: np.ndarray) -> np.ndarray:
    require(counts.ndim == 1 and counts.size in (2, 4), "Q16 row geometry")
    require(bool(np.all(counts >= 0)), "Q16 nonnegative counts")
    adjusted = counts.astype(object) + 1
    remaining = Q16_TOTAL - counts.size
    denominator = int(sum(adjusted))
    quotient = np.asarray([int(value * remaining // denominator) for value in adjusted], dtype=np.int64)
    remainder = [int(value * remaining % denominator) for value in adjusted]
    frequencies = quotient + 1
    missing = Q16_TOTAL - int(frequencies.sum())
    order = sorted(range(counts.size), key=lambda index: (-remainder[index], index))
    for index in order[:missing]:
        frequencies[index] += 1
    require(int(frequencies.sum()) == Q16_TOTAL, "Q16 exact sum")
    require(bool(np.all((frequencies >= 1) & (frequencies <= 65535))), "Q16 representable")
    return frequencies.astype(np.uint16)


@dataclass(frozen=True)
class Q16TreeModel:
    alphabet: int
    frequencies: np.ndarray

    @property
    def contexts(self) -> int:
        return context_count(self.alphabet)

    def validate(self) -> None:
        alphabet = validate_alphabet(self.alphabet)
        require(
            self.frequencies.shape == (context_count(alphabet), STATE_COUNT, alphabet),
            "model geometry",
        )
        require(self.frequencies.dtype == np.uint16, "model dtype")
        require(bool(np.all(self.frequencies >= 1)), "model positive")
        require(bool(np.all(self.frequencies.sum(axis=2, dtype=np.uint64) == Q16_TOTAL)), "model normalization")

    def serialize(self) -> bytes:
        self.validate()
        header = MODEL_STRUCT.pack(
            MODEL_MAGIC,
            VERSION,
            self.alphabet,
            checks_for_alphabet(self.alphabet),
            self.contexts,
            STATE_COUNT,
            Q16_TOTAL,
        )
        require(len(header) <= MODEL_HEADER_BYTES, "model header fit")
        packet = header + bytes(MODEL_HEADER_BYTES - len(header)) + self.frequencies.astype("<u2", copy=False).tobytes(order="C")
        require(len(packet) <= MAX_MODEL_PACKET_BYTES, "model packet cap")
        return packet

    @classmethod
    def deserialize(cls, packet: bytes) -> "Q16TreeModel":
        require(isinstance(packet, bytes), "model bytes")
        checked_scalar(len(packet), MODEL_HEADER_BYTES, MAX_MODEL_PACKET_BYTES, "model packet length")
        magic, version, alphabet, checks, contexts, states, total = MODEL_STRUCT.unpack(packet[: MODEL_STRUCT.size])
        alphabet = validate_alphabet(alphabet)
        require(magic == MODEL_MAGIC and version == VERSION, "model magic/version")
        require(checks == checks_for_alphabet(alphabet) and contexts == context_count(alphabet), "model context constants")
        require(states == STATE_COUNT and total == Q16_TOTAL, "model state constants")
        require(not any(packet[MODEL_STRUCT.size : MODEL_HEADER_BYTES]), "model header zero tail")
        expected_values = checked_mul(contexts, states, 1 << 20, "model contexts*states")
        expected_values = checked_mul(expected_values, alphabet, 1 << 20, "model value count")
        expected = checked_add(MODEL_HEADER_BYTES, checked_mul(2, expected_values, MAX_MODEL_PACKET_BYTES, "model table bytes"), MAX_MODEL_PACKET_BYTES, "model length")
        require(len(packet) == expected, "model exact packet length")
        frequencies = np.frombuffer(packet[MODEL_HEADER_BYTES:], dtype="<u2", count=expected_values).copy().reshape(contexts, states, alphabet)
        result = cls(alphabet, frequencies)
        result.validate()
        return result


def fit_model(alphabet: int, roots: np.ndarray, details: np.ndarray) -> Q16TreeModel:
    alphabet = validate_alphabet(alphabet)
    require(roots.ndim == details.ndim == 1, "training stream dimensions")
    require(roots.dtype == details.dtype == np.uint8, "training stream dtype")
    checked_scalar(int(roots.size), 1, MAX_TOTAL_SYMBOLS, "training roots")
    checked_scalar(int(details.size), 0, MAX_TOTAL_SYMBOLS, "training details")
    require(bool(np.all(roots < alphabet)) and bool(np.all(details < alphabet)), "training alphabet")
    counts = np.zeros((context_count(alphabet), STATE_COUNT, alphabet), dtype=np.int64)
    for value in roots:
        counts[0, 0, int(value)] += 1
    state = 0
    for index, value in enumerate(details):
        if index % DETAIL_BLOCK == 0:
            state = 0
        symbol = int(value)
        context = detail_context(alphabet, index)
        counts[context, state, symbol] += 1
        state = state_successor(alphabet, state, index, symbol)
    frequencies = np.empty_like(counts, dtype=np.uint16)
    for context in range(counts.shape[0]):
        for state_index in range(STATE_COUNT):
            frequencies[context, state_index, :] = _q16_row(counts[context, state_index, :])
    result = Q16TreeModel(alphabet, frequencies)
    result.validate()
    return result


def generate_transformed_source(
    alphabet: int,
    vectors: int,
    lanes: int,
    seed: int,
    structured: bool,
) -> tuple[np.ndarray, np.ndarray]:
    alphabet = validate_alphabet(alphabet)
    symbols = validate_geometry(lanes, vectors)
    rng = np.random.default_rng(seed)
    roots = rng.integers(0, alphabet, size=vectors, dtype=np.uint8)
    detail_count = symbols - vectors
    if detail_count == 0:
        return roots, np.empty(0, dtype=np.uint8)
    checks = checks_for_alphabet(alphabet)
    blocks = (detail_count + DETAIL_BLOCK - 1) // DETAIL_BLOCK
    body_count = DETAIL_BLOCK - checks
    body = rng.integers(0, alphabet, size=(blocks, body_count), dtype=np.uint8)
    if structured:
        tail = np.empty((blocks, checks), dtype=np.uint8)
        for group in range(checks):
            tail[:, group] = np.mod(body[:, group::checks].sum(axis=1, dtype=np.uint64), alphabet).astype(np.uint8)
    else:
        tail = rng.integers(0, alphabet, size=(blocks, checks), dtype=np.uint8)
    details = np.concatenate((body, tail), axis=1).reshape(-1)[:detail_count]
    return np.ascontiguousarray(roots), np.ascontiguousarray(details, dtype=np.uint8)


def synthesize_leaves(
    alphabet: int,
    vectors: int,
    lanes: int,
    seed: int,
    structured: bool,
    permutation: Sequence[int],
    selectors: Sequence[int],
) -> np.ndarray:
    roots, flat = generate_transformed_source(alphabet, vectors, lanes, seed, structured)
    lifted = LiftedTensor(roots, unflatten_details(flat, vectors, lanes))
    return lift_inverse(lifted, lanes, alphabet, permutation, selectors)


class ArithmeticEncoder:
    FULL = 1 << 32
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREE_QUARTER = 3 << 30

    def __init__(self) -> None:
        self.low = 0
        self.high = self.FULL - 1
        self.pending = 0
        self.bits: list[int] = []

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)
        self.pending = 0

    def write(self, symbol: int, frequencies: Sequence[int]) -> None:
        row = [int(value) for value in frequencies]
        require(0 <= symbol < len(row), "arithmetic symbol")
        require(all(value > 0 for value in row) and sum(row) == Q16_TOTAL, "arithmetic row")
        lower_count = sum(row[:symbol])
        upper_count = lower_count + row[symbol]
        width = self.high - self.low + 1
        self.high = self.low + width * upper_count // Q16_TOTAL - 1
        self.low = self.low + width * lower_count // Q16_TOTAL
        while True:
            if self.high < self.HALF:
                self._emit(0)
            elif self.low >= self.HALF:
                self._emit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.pending += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> tuple[bytes, int]:
        self.pending += 1
        self._emit(0 if self.low < self.QUARTER else 1)
        # Guard bits are physical, included in meaningful_bits, and validated.
        self.bits.extend([0] * ARITHMETIC_GUARD_BITS)
        meaningful_bits = len(self.bits)
        while len(self.bits) % 8:
            self.bits.append(0)
        packet = bytearray(len(self.bits) // 8)
        for index, bit in enumerate(self.bits):
            packet[index // 8] |= bit << (7 - (index & 7))
        return bytes(packet), meaningful_bits


class BitLimitedReader:
    def __init__(self, packet: bytes, meaningful_bits: int) -> None:
        require(isinstance(packet, bytes), "bit packet")
        total_bits = checked_mul(8, len(packet), MAX_FRAME_LOGICAL_BYTES * 8, "payload bits")
        self.meaningful_bits = checked_scalar(meaningful_bits, 32, total_bits, "meaningful bits")
        self.packet = packet
        self.position = 0
        for bit_index in range(self.meaningful_bits, total_bits):
            require(self._raw(bit_index) == 0, "arithmetic byte zero tail")
        for bit_index in range(self.meaningful_bits - ARITHMETIC_GUARD_BITS, self.meaningful_bits):
            require(self._raw(bit_index) == 0, "arithmetic explicit zero guard")

    def _raw(self, bit_index: int) -> int:
        return (self.packet[bit_index // 8] >> (7 - (bit_index & 7))) & 1

    def read(self) -> int:
        require(self.position < self.meaningful_bits, "arithmetic meaningful EOF")
        value = self._raw(self.position)
        self.position += 1
        return value


class ArithmeticDecoder:
    FULL = ArithmeticEncoder.FULL
    HALF = ArithmeticEncoder.HALF
    QUARTER = ArithmeticEncoder.QUARTER
    THREE_QUARTER = ArithmeticEncoder.THREE_QUARTER

    def __init__(self, packet: bytes, meaningful_bits: int) -> None:
        self.reader = BitLimitedReader(packet, meaningful_bits)
        self.low = 0
        self.high = self.FULL - 1
        self.code = 0
        for _ in range(32):
            self.code = (self.code << 1) | self.reader.read()

    def read(self, frequencies: Sequence[int]) -> int:
        row = [int(value) for value in frequencies]
        require(all(value > 0 for value in row) and sum(row) == Q16_TOTAL, "decode row")
        width = self.high - self.low + 1
        scaled = ((self.code - self.low + 1) * Q16_TOTAL - 1) // width
        lower_count = 0
        selected = -1
        upper_count = 0
        for index, frequency in enumerate(row):
            upper_count = lower_count + frequency
            if scaled < upper_count:
                selected = index
                break
            lower_count = upper_count
        require(selected >= 0, "decode scaled symbol")
        self.high = self.low + width * upper_count // Q16_TOTAL - 1
        self.low = self.low + width * lower_count // Q16_TOTAL
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.low -= self.HALF
                self.high -= self.HALF
                self.code -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.low -= self.QUARTER
                self.high -= self.QUARTER
                self.code -= self.QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = ((self.code << 1) | self.reader.read()) & (self.FULL - 1)
        return selected


def encode_coefficients(model: Q16TreeModel, roots: np.ndarray, details: np.ndarray) -> tuple[bytes, int]:
    model.validate()
    require(roots.ndim == details.ndim == 1 and roots.dtype == details.dtype == np.uint8, "coefficient streams")
    require(roots.size > 0 and roots.size + details.size <= MAX_SYMBOLS_PER_EXPERT, "coefficient count")
    encoder = ArithmeticEncoder()
    for value in roots:
        encoder.write(int(value), model.frequencies[0, 0, :])
    state = 0
    for index, value in enumerate(details):
        if index % DETAIL_BLOCK == 0:
            state = 0
        context = detail_context(model.alphabet, index)
        symbol = int(value)
        encoder.write(symbol, model.frequencies[context, state, :])
        state = state_successor(model.alphabet, state, index, symbol)
    return encoder.finish()


def decode_coefficients(
    model: Q16TreeModel,
    packet: bytes,
    meaningful_bits: int,
    root_count: int,
    detail_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.validate()
    root_count = checked_scalar(root_count, 1, MAX_VECTORS, "decode roots")
    detail_count = checked_scalar(detail_count, 0, MAX_SYMBOLS_PER_EXPERT, "decode details")
    checked_add(root_count, detail_count, MAX_SYMBOLS_PER_EXPERT, "decode coefficient total")
    decoder = ArithmeticDecoder(packet, meaningful_bits)
    roots = np.empty(root_count, dtype=np.uint8)
    for index in range(root_count):
        roots[index] = decoder.read(model.frequencies[0, 0, :])
    details = np.empty(detail_count, dtype=np.uint8)
    state = 0
    for index in range(detail_count):
        if index % DETAIL_BLOCK == 0:
            state = 0
        context = detail_context(model.alphabet, index)
        symbol = decoder.read(model.frequencies[context, state, :])
        details[index] = symbol
        state = state_successor(model.alphabet, state, index, symbol)
    require(decoder.reader.position == meaningful_bits, "arithmetic exact meaningful-bit exhaustion")
    # Re-encoding is mandatory at the ordinary frame decode layer, where the
    # exact payload and declared meaningful length are available.
    return roots, details


def _frame_header(
    expert_index: int,
    alphabet: int,
    lanes: int,
    vectors: int,
    permutation_bytes: int,
    selector_bytes: int,
    payload_bytes: int,
    meaningful_bits: int,
    logical_bytes: int,
    padded_bytes: int,
    leaf_symbols: int,
    body_sha256: bytes,
    header_crc: int,
) -> bytes:
    require(len(body_sha256) == 32, "frame body hash width")
    raw = FRAME_STRUCT.pack(
        FRAME_MAGIC,
        VERSION,
        alphabet,
        expert_index,
        0,
        lanes,
        vectors,
        permutation_bytes,
        selector_bytes,
        payload_bytes,
        meaningful_bits,
        logical_bytes,
        padded_bytes,
        leaf_symbols,
        body_sha256,
        header_crc,
    )
    require(len(raw) <= FRAME_HEADER_BYTES, "frame header fit")
    return raw + bytes(FRAME_HEADER_BYTES - len(raw))


@dataclass(frozen=True)
class FrameInfo:
    expert_index: int
    alphabet: int
    lanes: int
    vectors: int
    permutation_bytes: int
    selector_bytes: int
    payload_bytes: int
    meaningful_bits: int
    logical_bytes: int
    padded_bytes: int
    leaf_symbols: int


def build_frame(
    expert_index: int,
    model: Q16TreeModel,
    leaves: np.ndarray,
    permutation: Sequence[int],
    selectors: Sequence[int],
) -> bytes:
    expert_index = checked_scalar(expert_index, 0, MAX_EXPERTS - 1, "frame expert index")
    model.validate()
    require(leaves.ndim == 2 and leaves.dtype == np.uint8, "frame leaf matrix")
    vectors, lanes = (int(value) for value in leaves.shape)
    leaf_symbols = validate_geometry(lanes, vectors)
    lifted = lift_forward(leaves, model.alphabet, permutation, selectors)
    details = flatten_details(lifted)
    payload, meaningful_bits = encode_coefficients(model, lifted.roots, details)
    permutation_packet = serialize_permutation(permutation)
    selector_packet = pack_selectors(selectors, model.alphabet)
    body = permutation_packet + selector_packet + payload
    logical = checked_add(FRAME_HEADER_BYTES, len(body), MAX_FRAME_LOGICAL_BYTES, "frame logical bytes")
    padded = page_ceil(logical)
    require(padded <= MAX_FRAME_PADDED_BYTES, "frame padded cap")
    body_hash = hashlib.sha256(body).digest()
    zero_header = _frame_header(
        expert_index,
        model.alphabet,
        lanes,
        vectors,
        len(permutation_packet),
        len(selector_packet),
        len(payload),
        meaningful_bits,
        logical,
        padded,
        leaf_symbols,
        body_hash,
        0,
    )
    header_crc = zlib.crc32(zero_header) & 0xFFFFFFFF
    header = _frame_header(
        expert_index,
        model.alphabet,
        lanes,
        vectors,
        len(permutation_packet),
        len(selector_packet),
        len(payload),
        meaningful_bits,
        logical,
        padded,
        leaf_symbols,
        body_hash,
        header_crc,
    )
    return header + body + bytes(padded - logical)


def parse_frame_header(frame: bytes | memoryview) -> FrameInfo:
    total = checked_scalar(len(frame), FRAME_HEADER_BYTES, MAX_FRAME_PADDED_BYTES, "frame packet length")
    require(total >= FRAME_STRUCT.size, "frame fixed header")
    fields = FRAME_STRUCT.unpack(bytes(frame[: FRAME_STRUCT.size]))
    (
        magic,
        version,
        alphabet,
        expert_index,
        reserved,
        lanes,
        vectors,
        permutation_bytes,
        selector_bytes,
        payload_bytes,
        meaningful_bits,
        logical_bytes,
        padded_bytes,
        leaf_symbols,
        body_hash,
        header_crc,
    ) = fields
    require(magic == FRAME_MAGIC and version == VERSION and reserved == 0, "frame magic/version")
    alphabet = validate_alphabet(alphabet)
    expert_index = checked_scalar(expert_index, 0, MAX_EXPERTS - 1, "frame expert index")
    lanes = validate_lanes(lanes)
    vectors = validate_vectors(vectors)
    expected_symbols = validate_geometry(lanes, vectors)
    require(leaf_symbols == expected_symbols, "frame leaf symbol product")
    # Factorial happens only after the lane cap above.
    expected_permutation_bytes = permutation_byte_count(lanes)
    require(permutation_bytes == expected_permutation_bytes, "frame permutation bytes")
    expected_selector_bytes = (3 * (lanes - 1) + 7) // 8
    require(selector_bytes == expected_selector_bytes, "frame selector bytes")
    payload_bytes = checked_scalar(payload_bytes, 1, MAX_FRAME_LOGICAL_BYTES, "frame payload bytes")
    payload_bits = checked_mul(payload_bytes, 8, MAX_FRAME_LOGICAL_BYTES * 8, "frame payload bits")
    meaningful_bits = checked_scalar(meaningful_bits, 32, payload_bits, "frame meaningful bits")
    # Canonical packets use the minimum whole-byte payload for meaningful bits.
    require(payload_bytes == (meaningful_bits + 7) // 8, "frame canonical payload byte length")
    body_bytes = checked_add(permutation_bytes, selector_bytes, MAX_FRAME_LOGICAL_BYTES, "frame metadata bytes")
    body_bytes = checked_add(body_bytes, payload_bytes, MAX_FRAME_LOGICAL_BYTES, "frame body bytes")
    expected_logical = checked_add(FRAME_HEADER_BYTES, body_bytes, MAX_FRAME_LOGICAL_BYTES, "frame logical bytes")
    require(logical_bytes == expected_logical, "frame exact logical bytes")
    require(padded_bytes == page_ceil(logical_bytes), "frame page rounding")
    require(padded_bytes <= MAX_FRAME_PADDED_BYTES and total == padded_bytes, "frame padded length")
    require(not any(frame[FRAME_STRUCT.size : FRAME_HEADER_BYTES]), "frame header zero tail")
    zero = _frame_header(
        expert_index,
        alphabet,
        lanes,
        vectors,
        permutation_bytes,
        selector_bytes,
        payload_bytes,
        meaningful_bits,
        logical_bytes,
        padded_bytes,
        leaf_symbols,
        body_hash,
        0,
    )
    require((zlib.crc32(zero) & 0xFFFFFFFF) == header_crc, "frame header CRC")
    body_start, body_end = checked_slice(total, FRAME_HEADER_BYTES, body_bytes, MAX_FRAME_PADDED_BYTES, "frame body slice")
    body = bytes(frame[body_start:body_end])
    require(hashlib.sha256(body).digest() == body_hash, "frame body SHA-256")
    require(not any(frame[logical_bytes:padded_bytes]), "frame page zero tail")
    payload_offset = FRAME_HEADER_BYTES + permutation_bytes + selector_bytes
    payload = bytes(frame[payload_offset : payload_offset + payload_bytes])
    BitLimitedReader(payload, meaningful_bits)  # validates byte tail and 30-bit guard
    return FrameInfo(
        expert_index,
        alphabet,
        lanes,
        vectors,
        permutation_bytes,
        selector_bytes,
        payload_bytes,
        meaningful_bits,
        logical_bytes,
        padded_bytes,
        leaf_symbols,
    )


def decode_frame(
    model: Q16TreeModel,
    frame: bytes | memoryview,
) -> tuple[np.ndarray, list[int], list[int]]:
    info = parse_frame_header(frame)
    require(info.alphabet == model.alphabet, "frame/model alphabet")
    offset = FRAME_HEADER_BYTES
    permutation_packet = bytes(frame[offset : offset + info.permutation_bytes])
    offset += info.permutation_bytes
    selector_packet = bytes(frame[offset : offset + info.selector_bytes])
    offset += info.selector_bytes
    payload = bytes(frame[offset : offset + info.payload_bytes])
    permutation = deserialize_permutation(info.lanes, permutation_packet)
    selectors = unpack_selectors(selector_packet, info.lanes - 1, info.alphabet)
    detail_count = info.leaf_symbols - info.vectors
    roots, details = decode_coefficients(
        model,
        payload,
        info.meaningful_bits,
        info.vectors,
        detail_count,
    )
    canonical_payload, canonical_bits = encode_coefficients(model, roots, details)
    require(canonical_bits == info.meaningful_bits, "canonical arithmetic meaningful length")
    require(canonical_payload == payload, "canonical arithmetic byte stream")
    lifted = LiftedTensor(roots, unflatten_details(details, info.vectors, info.lanes))
    leaves = lift_inverse(lifted, info.lanes, info.alphabet, permutation, selectors)
    return leaves, permutation, selectors


@dataclass(frozen=True)
class DirectoryEntry:
    offset: int
    padded_bytes: int
    logical_bytes: int
    leaf_symbols: int


def _directory_header(
    expert_count: int,
    entries_bytes: int,
    packet_bytes: int,
    entries_hash: bytes,
    header_crc: int,
) -> bytes:
    require(len(entries_hash) == 32, "directory hash width")
    raw = DIRECTORY_HEADER_STRUCT.pack(
        DIRECTORY_MAGIC,
        VERSION,
        DIRECTORY_ENTRY_BYTES,
        expert_count,
        entries_bytes,
        packet_bytes,
        entries_hash,
        header_crc,
    )
    require(len(raw) <= DIRECTORY_HEADER_BYTES, "directory header fit")
    return raw + bytes(DIRECTORY_HEADER_BYTES - len(raw))


def serialize_directory(entries: Sequence[DirectoryEntry]) -> bytes:
    expert_count = validate_expert_count(len(entries))
    rows = b"".join(
        DIRECTORY_ENTRY_STRUCT.pack(entry.offset, entry.padded_bytes, entry.logical_bytes, entry.leaf_symbols)
        for entry in entries
    )
    entries_bytes = checked_mul(expert_count, DIRECTORY_ENTRY_BYTES, MAX_DIRECTORY_PACKET_BYTES, "directory entries bytes")
    require(len(rows) == entries_bytes, "directory rows length")
    packet_bytes = checked_add(DIRECTORY_HEADER_BYTES, entries_bytes, MAX_DIRECTORY_PACKET_BYTES, "directory packet bytes")
    digest = hashlib.sha256(rows).digest()
    zero = _directory_header(expert_count, entries_bytes, packet_bytes, digest, 0)
    crc = zlib.crc32(zero) & 0xFFFFFFFF
    return _directory_header(expert_count, entries_bytes, packet_bytes, digest, crc) + rows


def deserialize_directory(packet: bytes, expected_experts: int) -> tuple[DirectoryEntry, ...]:
    expected_experts = validate_expert_count(expected_experts)
    checked_scalar(len(packet), DIRECTORY_HEADER_BYTES, MAX_DIRECTORY_PACKET_BYTES, "directory packet length")
    require(len(packet) >= DIRECTORY_HEADER_STRUCT.size, "directory fixed header")
    magic, version, entry_bytes, experts, entries_bytes, packet_bytes, entries_hash, header_crc = DIRECTORY_HEADER_STRUCT.unpack(
        packet[: DIRECTORY_HEADER_STRUCT.size]
    )
    require(magic == DIRECTORY_MAGIC and version == VERSION, "directory magic/version")
    require(entry_bytes == DIRECTORY_ENTRY_BYTES, "directory entry width")
    experts = validate_expert_count(experts)  # before product or loop
    require(experts == expected_experts, "directory/global expert count")
    expected_entries_bytes = checked_mul(experts, DIRECTORY_ENTRY_BYTES, MAX_DIRECTORY_PACKET_BYTES, "directory entry product")
    require(entries_bytes == expected_entries_bytes, "directory entries length")
    expected_packet_bytes = checked_add(DIRECTORY_HEADER_BYTES, entries_bytes, MAX_DIRECTORY_PACKET_BYTES, "directory exact packet")
    require(packet_bytes == expected_packet_bytes == len(packet), "directory packet coverage")
    require(not any(packet[DIRECTORY_HEADER_STRUCT.size : DIRECTORY_HEADER_BYTES]), "directory header zero tail")
    rows = packet[DIRECTORY_HEADER_BYTES:packet_bytes]
    require(hashlib.sha256(rows).digest() == entries_hash, "directory entries SHA-256")
    zero = _directory_header(experts, entries_bytes, packet_bytes, entries_hash, 0)
    require((zlib.crc32(zero) & 0xFFFFFFFF) == header_crc, "directory header CRC")
    output: list[DirectoryEntry] = []
    for index in range(experts):
        start = DIRECTORY_HEADER_BYTES + index * DIRECTORY_ENTRY_BYTES
        offset, padded, logical, symbols = DIRECTORY_ENTRY_STRUCT.unpack(packet[start : start + DIRECTORY_ENTRY_BYTES])
        output.append(DirectoryEntry(int(offset), int(padded), int(logical), int(symbols)))
    return tuple(output)


def _global_header(
    alphabet: int,
    expert_count: int,
    directory_offset: int,
    directory_packet_bytes: int,
    directory_page_bytes: int,
    model_offset: int,
    model_packet_bytes: int,
    model_page_bytes: int,
    frames_offset: int,
    total_bytes: int,
    total_leaf_symbols: int,
    directory_sha256: bytes,
    model_sha256: bytes,
    header_crc: int,
) -> bytes:
    require(len(directory_sha256) == len(model_sha256) == 32, "global hash widths")
    raw = GLOBAL_STRUCT.pack(
        GLOBAL_MAGIC,
        VERSION,
        alphabet,
        expert_count,
        GLOBAL_HEADER_BYTES,
        DIRECTORY_ENTRY_BYTES,
        0,
        directory_offset,
        directory_packet_bytes,
        directory_page_bytes,
        model_offset,
        model_packet_bytes,
        model_page_bytes,
        frames_offset,
        total_bytes,
        total_leaf_symbols,
        directory_sha256,
        model_sha256,
        header_crc,
    )
    require(len(raw) <= GLOBAL_HEADER_BYTES, "global header fit")
    return raw + bytes(GLOBAL_HEADER_BYTES - len(raw))


@dataclass(frozen=True)
class ExpertInput:
    leaves: np.ndarray
    permutation: tuple[int, ...]
    selectors: tuple[int, ...]

    @classmethod
    def create(
        cls,
        leaves: np.ndarray,
        permutation: Sequence[int],
        selectors: Sequence[int],
    ) -> "ExpertInput":
        return cls(leaves, tuple(int(value) for value in permutation), tuple(int(value) for value in selectors))


def build_container(model: Q16TreeModel, experts: Sequence[ExpertInput]) -> bytes:
    model.validate()
    expert_count = validate_expert_count(len(experts))
    frames: list[bytes] = []
    total_symbols = 0
    for expert_index, source in enumerate(experts):
        require(isinstance(source, ExpertInput), "expert input type")
        frame = build_frame(expert_index, model, source.leaves, source.permutation, source.selectors)
        info = parse_frame_header(frame)
        total_symbols = checked_add(total_symbols, info.leaf_symbols, MAX_TOTAL_SYMBOLS, "container total symbols")
        frames.append(frame)

    model_packet = model.serialize()
    model_page_bytes = page_ceil(len(model_packet))
    directory_packet_bytes = DIRECTORY_HEADER_BYTES + expert_count * DIRECTORY_ENTRY_BYTES
    require(directory_packet_bytes <= MAX_DIRECTORY_PACKET_BYTES, "directory packet cap")
    directory_page_bytes = page_ceil(directory_packet_bytes)
    directory_offset = GLOBAL_HEADER_BYTES
    model_offset = checked_add(directory_offset, directory_page_bytes, MAX_CONTAINER_BYTES, "model offset")
    frames_offset = checked_add(model_offset, model_page_bytes, MAX_CONTAINER_BYTES, "frames offset")
    entries: list[DirectoryEntry] = []
    offset = frames_offset
    for frame in frames:
        info = parse_frame_header(frame)
        entries.append(DirectoryEntry(offset, len(frame), info.logical_bytes, info.leaf_symbols))
        offset = checked_add(offset, len(frame), MAX_CONTAINER_BYTES, "frame prefix sum")
    total_bytes = offset
    directory_packet = serialize_directory(entries)
    require(len(directory_packet) == directory_packet_bytes, "directory serialized length")
    directory_hash = hashlib.sha256(directory_packet).digest()
    model_hash = hashlib.sha256(model_packet).digest()
    zero = _global_header(
        model.alphabet,
        expert_count,
        directory_offset,
        len(directory_packet),
        directory_page_bytes,
        model_offset,
        len(model_packet),
        model_page_bytes,
        frames_offset,
        total_bytes,
        total_symbols,
        directory_hash,
        model_hash,
        0,
    )
    crc = zlib.crc32(zero) & 0xFFFFFFFF
    header = _global_header(
        model.alphabet,
        expert_count,
        directory_offset,
        len(directory_packet),
        directory_page_bytes,
        model_offset,
        len(model_packet),
        model_page_bytes,
        frames_offset,
        total_bytes,
        total_symbols,
        directory_hash,
        model_hash,
        crc,
    )
    packet = (
        header
        + directory_packet
        + bytes(directory_page_bytes - len(directory_packet))
        + model_packet
        + bytes(model_page_bytes - len(model_packet))
        + b"".join(frames)
    )
    require(len(packet) == total_bytes <= MAX_CONTAINER_BYTES, "container literal total")
    return packet


@dataclass(frozen=True)
class ParsedContainer:
    packet: bytes
    alphabet: int
    expert_count: int
    model: Q16TreeModel
    entries: tuple[DirectoryEntry, ...]
    directory_packet_bytes: int
    directory_page_bytes: int
    model_packet_bytes: int
    model_page_bytes: int
    frames_offset: int
    total_leaf_symbols: int

    def frame_view(self, expert_index: int) -> memoryview:
        expert_index = checked_scalar(expert_index, 0, self.expert_count - 1, "frame lookup expert")
        entry = self.entries[expert_index]
        _, end = checked_slice(len(self.packet), entry.offset, entry.padded_bytes, MAX_CONTAINER_BYTES, "frame view")
        return memoryview(self.packet)[entry.offset:end]


def parse_container(packet: bytes) -> ParsedContainer:
    require(isinstance(packet, bytes), "container bytes")
    total_length = checked_scalar(len(packet), GLOBAL_HEADER_BYTES, MAX_CONTAINER_BYTES, "container literal size")
    require(total_length >= GLOBAL_STRUCT.size, "global fixed header")
    fields = GLOBAL_STRUCT.unpack(packet[: GLOBAL_STRUCT.size])
    (
        magic,
        version,
        alphabet,
        expert_count,
        header_bytes,
        directory_entry_bytes,
        reserved,
        directory_offset,
        directory_packet_bytes,
        directory_page_bytes,
        model_offset,
        model_packet_bytes,
        model_page_bytes,
        frames_offset,
        total_bytes,
        total_leaf_symbols,
        directory_hash,
        model_hash,
        header_crc,
    ) = fields
    require(magic == GLOBAL_MAGIC and version == VERSION, "global magic/version")
    alphabet = validate_alphabet(alphabet)
    expert_count = validate_expert_count(expert_count)  # before directory loop
    require(header_bytes == GLOBAL_HEADER_BYTES and directory_entry_bytes == DIRECTORY_ENTRY_BYTES and reserved == 0, "global constants")
    require(not any(packet[GLOBAL_STRUCT.size : GLOBAL_HEADER_BYTES]), "global header zero tail")
    directory_offset = checked_scalar(directory_offset, GLOBAL_HEADER_BYTES, MAX_CONTAINER_BYTES, "directory offset")
    require(directory_offset == GLOBAL_HEADER_BYTES and directory_offset % PAGE_BYTES == 0, "directory canonical offset")
    expected_directory_packet = checked_add(
        DIRECTORY_HEADER_BYTES,
        checked_mul(expert_count, DIRECTORY_ENTRY_BYTES, MAX_DIRECTORY_PACKET_BYTES, "directory product"),
        MAX_DIRECTORY_PACKET_BYTES,
        "directory packet expected",
    )
    require(directory_packet_bytes == expected_directory_packet, "directory exact bytes")
    require(directory_page_bytes == page_ceil(directory_packet_bytes), "directory page bytes")
    require(directory_page_bytes <= MAX_DIRECTORY_PACKET_BYTES + PAGE_BYTES, "directory page cap")
    expected_model_offset = checked_add(directory_offset, directory_page_bytes, MAX_CONTAINER_BYTES, "expected model offset")
    require(model_offset == expected_model_offset and model_offset % PAGE_BYTES == 0, "model offset")
    model_packet_bytes = checked_scalar(model_packet_bytes, MODEL_HEADER_BYTES, MAX_MODEL_PACKET_BYTES, "model packet bytes")
    require(model_page_bytes == page_ceil(model_packet_bytes), "model page bytes")
    expected_frames_offset = checked_add(model_offset, model_page_bytes, MAX_CONTAINER_BYTES, "expected frames offset")
    require(frames_offset == expected_frames_offset and frames_offset % PAGE_BYTES == 0, "frames offset")
    require(total_bytes == total_length, "global literal total bytes")
    total_leaf_symbols = checked_scalar(total_leaf_symbols, 1, MAX_TOTAL_SYMBOLS, "global total symbols")
    zero = _global_header(
        alphabet,
        expert_count,
        directory_offset,
        directory_packet_bytes,
        directory_page_bytes,
        model_offset,
        model_packet_bytes,
        model_page_bytes,
        frames_offset,
        total_bytes,
        total_leaf_symbols,
        directory_hash,
        model_hash,
        0,
    )
    require((zlib.crc32(zero) & 0xFFFFFFFF) == header_crc, "global header CRC")
    directory_start, directory_end = checked_slice(total_length, directory_offset, directory_packet_bytes, MAX_CONTAINER_BYTES, "directory slice")
    directory_packet = packet[directory_start:directory_end]
    require(hashlib.sha256(directory_packet).digest() == directory_hash, "directory packet SHA-256")
    directory_page_end = checked_add(directory_offset, directory_page_bytes, MAX_CONTAINER_BYTES, "directory page end")
    require(not any(packet[directory_end:directory_page_end]), "directory page zero tail")
    entries = deserialize_directory(directory_packet, expert_count)
    model_start, model_end = checked_slice(total_length, model_offset, model_packet_bytes, MAX_CONTAINER_BYTES, "model slice")
    model_packet = packet[model_start:model_end]
    require(hashlib.sha256(model_packet).digest() == model_hash, "model packet SHA-256")
    require(not any(packet[model_end:frames_offset]), "model page zero tail")
    model = Q16TreeModel.deserialize(model_packet)
    require(model.alphabet == alphabet, "global/model alphabet")

    expected_offset = frames_offset
    observed_symbols = 0
    packet_view = memoryview(packet)
    for expert_index, entry in enumerate(entries):
        offset = checked_scalar(entry.offset, frames_offset, MAX_CONTAINER_BYTES, "entry offset")
        padded = checked_scalar(entry.padded_bytes, PAGE_BYTES, MAX_FRAME_PADDED_BYTES, "entry padded bytes")
        logical = checked_scalar(entry.logical_bytes, FRAME_HEADER_BYTES, MAX_FRAME_LOGICAL_BYTES, "entry logical bytes")
        symbols = checked_scalar(entry.leaf_symbols, 1, MAX_SYMBOLS_PER_EXPERT, "entry symbols")
        require(offset == expected_offset and offset % PAGE_BYTES == 0, "entry contiguous aligned offset")
        require(padded % PAGE_BYTES == 0 and logical <= padded, "entry page geometry")
        _, frame_end = checked_slice(total_length, offset, padded, MAX_CONTAINER_BYTES, "entry frame bounds")
        frame_info = parse_frame_header(packet_view[offset:frame_end])
        require(frame_info.expert_index == expert_index and frame_info.alphabet == alphabet, "entry frame identity")
        require(frame_info.padded_bytes == padded and frame_info.logical_bytes == logical and frame_info.leaf_symbols == symbols, "entry/frame agreement")
        expected_offset = checked_add(expected_offset, padded, MAX_CONTAINER_BYTES, "entry prefix sum")
        observed_symbols = checked_add(observed_symbols, symbols, MAX_TOTAL_SYMBOLS, "entry total symbols")
    require(expected_offset == total_length, "frame coverage")
    require(observed_symbols == total_leaf_symbols, "symbol coverage")
    return ParsedContainer(
        packet,
        alphabet,
        expert_count,
        model,
        entries,
        directory_packet_bytes,
        directory_page_bytes,
        model_packet_bytes,
        model_page_bytes,
        frames_offset,
        total_leaf_symbols,
    )


def decode_expert(packet: bytes, expert_index: int) -> np.ndarray:
    parsed = parse_container(packet)
    return decode_frame(parsed.model, parsed.frame_view(expert_index))[0]


def decode_container(packet: bytes) -> list[np.ndarray]:
    parsed = parse_container(packet)
    return [decode_frame(parsed.model, parsed.frame_view(index))[0] for index in range(parsed.expert_count)]


def reencode_container(packet: bytes) -> bytes:
    parsed = parse_container(packet)
    sources: list[ExpertInput] = []
    for index in range(parsed.expert_count):
        leaves, permutation, selectors = decode_frame(parsed.model, parsed.frame_view(index))
        sources.append(ExpertInput.create(leaves, permutation, selectors))
    rebuilt = build_container(parsed.model, sources)
    require(rebuilt == packet, "canonical container byte re-encode")
    return rebuilt


def leaf_digest(leaves: np.ndarray) -> str:
    require(isinstance(leaves, np.ndarray) and leaves.ndim == 2 and leaves.dtype == np.uint8, "leaf digest array")
    return sha256_bytes(np.ascontiguousarray(leaves).tobytes(order="C"))


class InstrumentedPageReader:
    """Random-access reader that records the exact physical page union."""

    def __init__(self, packet: bytes) -> None:
        require(isinstance(packet, bytes), "instrumented packet")
        checked_scalar(len(packet), 1, MAX_CONTAINER_BYTES, "instrumented packet length")
        self.packet = packet
        self.pages: set[int] = set()
        self.ranges: list[dict[str, int | str]] = []

    def read_pages(self, offset: int, length: int, owner: str) -> memoryview:
        offset, end = checked_slice(len(self.packet), offset, length, MAX_CONTAINER_BYTES, "instrumented read")
        require(offset % PAGE_BYTES == 0 and length % PAGE_BYTES == 0, "instrumented whole-page read")
        for page in range(offset // PAGE_BYTES, end // PAGE_BYTES):
            self.pages.add(page)
        self.ranges.append({"owner": owner, "offset": offset, "length": length})
        return memoryview(self.packet)[offset:end]

    @property
    def union_bytes(self) -> int:
        return len(self.pages) * PAGE_BYTES


def _trace_expert_cold_pages_parsed(parsed: ParsedContainer, expert_index: int) -> dict[str, object]:
    expert_index = checked_scalar(expert_index, 0, parsed.expert_count - 1, "trace expert")
    reader = InstrumentedPageReader(parsed.packet)
    reader.read_pages(0, GLOBAL_HEADER_BYTES, "global_header")
    reader.read_pages(GLOBAL_HEADER_BYTES, parsed.directory_page_bytes, "directory")
    model_offset = GLOBAL_HEADER_BYTES + parsed.directory_page_bytes
    reader.read_pages(model_offset, parsed.model_page_bytes, "model")
    entry = parsed.entries[expert_index]
    reader.read_pages(entry.offset, entry.padded_bytes, f"expert_{expert_index}")
    expected = parsed.frames_offset + entry.padded_bytes
    require(reader.union_bytes == expected, "instrumented cold page union")
    return {
        "expert_index": expert_index,
        "page_indices": sorted(reader.pages),
        "page_count": len(reader.pages),
        "union_bytes": reader.union_bytes,
        "ranges": reader.ranges,
        "expected_global_plus_local_bytes": expected,
        "union_matches_expected": True,
    }


def trace_expert_cold_pages(packet: bytes, expert_index: int) -> dict[str, object]:
    return _trace_expert_cold_pages_parsed(parse_container(packet), expert_index)


def layout_cold_ledger(global_bytes: int, frame_bytes: Sequence[int]) -> dict[str, object]:
    expert_count = validate_expert_count(len(frame_bytes))
    global_bytes = checked_scalar(global_bytes, PAGE_BYTES, MAX_CONTAINER_BYTES, "ledger global bytes")
    require(global_bytes % PAGE_BYTES == 0, "ledger global page alignment")
    frames = [checked_scalar(value, PAGE_BYTES, MAX_FRAME_PADDED_BYTES, "ledger frame bytes") for value in frame_bytes]
    require(all(value % PAGE_BYTES == 0 for value in frames), "ledger frame page alignment")
    frame_total = 0
    for value in frames:
        frame_total = checked_add(frame_total, value, MAX_CONTAINER_BYTES, "ledger frame total")
    total_bytes = checked_add(global_bytes, frame_total, MAX_CONTAINER_BYTES, "ledger literal total")
    rows: list[dict[str, object]] = []
    owner_numerator_sum = 0
    all_below_two = True
    for expert_index, local_bytes in enumerate(frames):
        # owner_share = (G + E*F_e) / E
        owner_numerator = checked_add(
            global_bytes,
            checked_mul(expert_count, local_bytes, MAX_EXPERTS * MAX_FRAME_PADDED_BYTES, "owner local product"),
            MAX_EXPERTS * MAX_CONTAINER_BYTES,
            "owner numerator",
        )
        owner_numerator_sum += owner_numerator
        cold_bytes = checked_add(global_bytes, local_bytes, MAX_CONTAINER_BYTES, "cold bytes")
        amplification_numerator = expert_count * cold_bytes
        amplification_denominator = owner_numerator
        below_two = amplification_numerator < 2 * amplification_denominator
        all_below_two = all_below_two and below_two
        owner_fraction = Fraction(owner_numerator, expert_count)
        amplification = Fraction(amplification_numerator, amplification_denominator)
        rows.append(
            {
                "expert_index": expert_index,
                "global_bytes": global_bytes,
                "local_frame_bytes": local_bytes,
                "cold_bytes": cold_bytes,
                "owner_share_numerator": owner_fraction.numerator,
                "owner_share_denominator": owner_fraction.denominator,
                "owner_share_float_bytes": float(owner_fraction),
                "cold_amplification_numerator": amplification.numerator,
                "cold_amplification_denominator": amplification.denominator,
                "cold_amplification_float": float(amplification),
                "cold_below_two_by_integer_cross_multiplication": below_two,
            }
        )
    require(owner_numerator_sum == expert_count * total_bytes, "owner byte conservation")
    return {
        "expert_count": expert_count,
        "global_bytes": global_bytes,
        "frame_bytes": frames,
        "container_bytes": total_bytes,
        "owner_numerator_sum": owner_numerator_sum,
        "owner_denominator": expert_count,
        "owner_sum_bytes": owner_numerator_sum // expert_count,
        "owner_sum_equals_container": owner_numerator_sum == expert_count * total_bytes,
        "cold": rows,
        "all_cold_below_two": all_below_two,
        "decision_math": "exact integers/rationals; strict <2 by cross multiplication",
    }


def physical_ledger(packet: bytes) -> dict[str, object]:
    parsed = parse_container(packet)
    layout = layout_cold_ledger(
        parsed.frames_offset,
        [entry.padded_bytes for entry in parsed.entries],
    )
    traces = [_trace_expert_cold_pages_parsed(parsed, index) for index in range(parsed.expert_count)]
    for row, trace in zip(layout["cold"], traces, strict=True):
        require(row["cold_bytes"] == trace["union_bytes"], "cold ledger/instrument union")
    return {
        "schema": "silt-v1-owner-aware-physical-ledger",
        "container_bytes": len(packet),
        "container_sha256": sha256_bytes(packet),
        "alphabet": parsed.alphabet,
        "expert_count": parsed.expert_count,
        "total_leaf_symbols": parsed.total_leaf_symbols,
        "physical_bits_per_leaf_symbol": 8.0 * len(packet) / parsed.total_leaf_symbols,
        "global_header_bytes": GLOBAL_HEADER_BYTES,
        "directory_packet_bytes": parsed.directory_packet_bytes,
        "directory_page_bytes": parsed.directory_page_bytes,
        "model_packet_bytes": parsed.model_packet_bytes,
        "model_page_bytes": parsed.model_page_bytes,
        "global_owned_bytes": parsed.frames_offset,
        "layout": layout,
        "instrumented_page_unions": traces,
        "cold_below_two": bool(layout["all_cold_below_two"]),
        "source_gain_claim": False,
        "scope": "source-free finite label mechanism; not model bpw or MSE evidence",
    }


def audit_unequal_frame_counterexample() -> dict[str, object]:
    ledger = layout_cold_ledger(8192, [4096] + [8192] * 7)
    expert_zero = ledger["cold"][0]
    require(expert_zero["cold_amplification_numerator"] == 12, "counterexample numerator")
    require(expert_zero["cold_amplification_denominator"] == 5, "counterexample denominator")
    require(not expert_zero["cold_below_two_by_integer_cross_multiplication"], "counterexample must fail")
    return ledger
