#!/usr/bin/env python3
"""Source-free SILT-INT2 mechanism reference.

This module has no external input path.  It implements exact modular lifting,
canonical charged tree metadata, a serialized Q16 unifilar model, a finite
reference arithmetic stream, and a literal expert-local container.  Synthetic
success is only a mechanism unit test and is never evidence about model
weights.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


SCHEMA = "silt-int2-source-free-mechanism-v0"
VERSION = 0
PAGE_BYTES = 4096
GLOBAL_HEADER_BYTES = 4096
MODEL_HEADER_BYTES = 256
FRAME_HEADER_BYTES = 256
Q16_TOTAL = 1 << 16
STATE_COUNT = 64
DETAIL_BLOCK = 32
GLOBAL_MAGIC = b"SILTSF0\0"
MODEL_MAGIC = b"SILTMOD0"
FRAME_MAGIC = b"SILTFR0\0"
GLOBAL_STRUCT = struct.Struct("<8sHHIIIIQIIQQQ32sI")
FRAME_STRUCT = struct.Struct("<8sHHHHIIIIIQIIQII")
MODEL_STRUCT = struct.Struct("<8sHHHHII")
DIRECTORY_ENTRY = struct.Struct("<QQ")


class ContractError(RuntimeError):
    """A source-free format or mechanism invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def page_ceil(value: int) -> int:
    require(value >= 0, "nonnegative byte count")
    return ((value + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES


def checks_for_alphabet(alphabet: int) -> int:
    require(alphabet in (2, 4), "alphabet must be GF2 or Z4")
    return 6 if alphabet == 2 else 3


def context_count(alphabet: int) -> int:
    return 1 + 2 * checks_for_alphabet(alphabet)


def level_sizes(lanes: int) -> list[int]:
    require(lanes > 0, "positive lane count")
    sizes = [lanes]
    while sizes[-1] > 1:
        current = sizes[-1]
        sizes.append(current // 2 + current % 2)
    return sizes


def level_pair_counts(lanes: int) -> list[int]:
    sizes = level_sizes(lanes)
    result = [value // 2 for value in sizes[:-1]]
    require(sum(result) == lanes - 1, "balanced tree internal-node count")
    return result


def permutation_byte_count(lanes: int) -> int:
    require(lanes > 0, "positive permutation size")
    states = math.factorial(lanes)
    return ((states - 1).bit_length() + 7) // 8


def validate_permutation(permutation: Sequence[int], lanes: int | None = None) -> None:
    n = len(permutation) if lanes is None else lanes
    require(len(permutation) == n, "permutation length")
    require(sorted(int(x) for x in permutation) == list(range(n)), "permutation domain")


def rank_permutation(permutation: Sequence[int]) -> int:
    validate_permutation(permutation)
    available = list(range(len(permutation)))
    rank = 0
    for position, value in enumerate(permutation):
        index = available.index(int(value))
        rank += index * math.factorial(len(permutation) - position - 1)
        available.pop(index)
    require(0 <= rank < math.factorial(len(permutation)), "factoradic rank")
    return rank


def unrank_permutation(lanes: int, rank: int) -> list[int]:
    require(lanes > 0, "positive permutation size")
    require(0 <= rank < math.factorial(lanes), "factoradic rank range")
    available = list(range(lanes))
    result: list[int] = []
    remainder = int(rank)
    for width in range(lanes, 0, -1):
        factorial = math.factorial(width - 1)
        index, remainder = divmod(remainder, factorial)
        result.append(available.pop(index))
    validate_permutation(result, lanes)
    return result


def serialize_permutation(permutation: Sequence[int]) -> bytes:
    count = permutation_byte_count(len(permutation))
    rank = rank_permutation(permutation)
    return rank.to_bytes(count, "big") if count else b""


def deserialize_permutation(lanes: int, payload: bytes) -> list[int]:
    require(len(payload) == permutation_byte_count(lanes), "factoradic byte length")
    rank = int.from_bytes(payload, "big") if payload else 0
    return unrank_permutation(lanes, rank)


def pack_selectors(selectors: Sequence[int]) -> bytes:
    for value in selectors:
        require(0 <= int(value) < 8, "three-bit selector")
    bit_count = 3 * len(selectors)
    result = bytearray((bit_count + 7) // 8)
    for index, value in enumerate(selectors):
        value = int(value)
        for offset in range(3):
            bit = (value >> (2 - offset)) & 1
            bit_index = 3 * index + offset
            result[bit_index // 8] |= bit << (7 - (bit_index & 7))
    return bytes(result)


def unpack_selectors(payload: bytes, count: int) -> list[int]:
    require(count >= 0, "selector count")
    require(len(payload) == (3 * count + 7) // 8, "selector byte length")
    result: list[int] = []
    for index in range(count):
        value = 0
        for offset in range(3):
            bit_index = 3 * index + offset
            bit = (payload[bit_index // 8] >> (7 - (bit_index & 7))) & 1
            value = (value << 1) | bit
        result.append(value)
    used = 3 * count
    for bit_index in range(used, 8 * len(payload)):
        require(
            ((payload[bit_index // 8] >> (7 - (bit_index & 7))) & 1) == 0,
            "nonzero selector tail",
        )
    return result


def deterministic_permutation(lanes: int, seed: int) -> list[int]:
    require(lanes > 0, "positive lanes")
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.permutation(lanes)]


def deterministic_selectors(lanes: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 8, size=lanes - 1, dtype=np.uint8)]


@dataclass(frozen=True)
class LiftedTensor:
    roots: np.ndarray
    detail_levels: tuple[np.ndarray, ...]  # finest to coarsest

    def validate(self, vectors: int, lanes: int, alphabet: int) -> None:
        require(self.roots.shape == (vectors,), "root geometry")
        require(self.roots.dtype == np.uint8, "root dtype")
        require(len(self.detail_levels) == len(level_pair_counts(lanes)), "detail levels")
        for values, pairs in zip(self.detail_levels, level_pair_counts(lanes), strict=True):
            require(values.shape == (vectors, pairs), "detail geometry")
            require(values.dtype == np.uint8, "detail dtype")
            require(bool(np.all(values < alphabet)), "detail alphabet")
        require(bool(np.all(self.roots < alphabet)), "root alphabet")


def lift_forward(
    leaves: np.ndarray,
    alphabet: int,
    permutation: Sequence[int],
    selectors: Sequence[int],
) -> LiftedTensor:
    checks_for_alphabet(alphabet)
    require(leaves.ndim == 2, "leaf matrix")
    vectors, lanes = leaves.shape
    require(vectors > 0 and lanes > 0, "positive leaf geometry")
    require(leaves.dtype == np.uint8, "leaf dtype")
    require(bool(np.all(leaves < alphabet)), "leaf alphabet")
    validate_permutation(permutation, lanes)
    require(len(selectors) == lanes - 1, "selector count")
    current = np.ascontiguousarray(leaves[:, np.asarray(permutation, dtype=np.int64)])
    levels: list[np.ndarray] = []
    selector_offset = 0
    for pairs in level_pair_counts(lanes):
        codes = np.asarray(selectors[selector_offset : selector_offset + pairs], dtype=np.uint8)
        selector_offset += pairs
        left = current[:, 0 : 2 * pairs : 2].astype(np.int16)
        right = current[:, 1 : 2 * pairs : 2].astype(np.int16)
        swap = ((codes >> 2) & 1).astype(bool)[None, :]
        p = ((codes >> 1) & 1).astype(np.int16)[None, :]
        u = (codes & 1).astype(np.int16)[None, :]
        x = np.where(swap, right, left)
        y = np.where(swap, left, right)
        detail = np.mod(y - p * x, alphabet).astype(np.uint8)
        coarse = np.mod(x + u * detail.astype(np.int16), alphabet).astype(np.uint8)
        if current.shape[1] & 1:
            current = np.concatenate((coarse, current[:, -1:]), axis=1)
        else:
            current = coarse
        levels.append(np.ascontiguousarray(detail))
    require(selector_offset == len(selectors), "all selectors consumed")
    require(current.shape == (vectors, 1), "single root")
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
    checks_for_alphabet(alphabet)
    vectors = int(lifted.roots.shape[0])
    lifted.validate(vectors, lanes, alphabet)
    validate_permutation(permutation, lanes)
    require(len(selectors) == lanes - 1, "selector count")
    counts = level_pair_counts(lanes)
    offsets = np.cumsum([0] + counts).tolist()
    sizes = level_sizes(lanes)
    current = np.ascontiguousarray(lifted.roots[:, None])
    for depth in range(len(counts) - 1, -1, -1):
        pairs = counts[depth]
        previous_width = sizes[depth]
        codes = np.asarray(selectors[offsets[depth] : offsets[depth + 1]], dtype=np.uint8)
        detail = lifted.detail_levels[depth].astype(np.int16)
        coarse = current[:, :pairs].astype(np.int16)
        swap = ((codes >> 2) & 1).astype(bool)[None, :]
        p = ((codes >> 1) & 1).astype(np.int16)[None, :]
        u = (codes & 1).astype(np.int16)[None, :]
        x = np.mod(coarse - u * detail, alphabet)
        y = np.mod(detail + p * x, alphabet)
        left = np.where(swap, y, x).astype(np.uint8)
        right = np.where(swap, x, y).astype(np.uint8)
        previous = np.empty((vectors, previous_width), dtype=np.uint8)
        previous[:, 0 : 2 * pairs : 2] = left
        previous[:, 1 : 2 * pairs : 2] = right
        if previous_width & 1:
            require(current.shape[1] == pairs + 1, "odd carry geometry")
            previous[:, -1] = current[:, -1]
        else:
            require(current.shape[1] == pairs, "even coarse geometry")
        current = previous
    require(current.shape == (vectors, lanes), "inverse leaf geometry")
    result = np.empty_like(current)
    result[:, np.asarray(permutation, dtype=np.int64)] = current
    require(bool(np.all(result < alphabet)), "inverse alphabet")
    return result


def flatten_details(lifted: LiftedTensor) -> np.ndarray:
    pieces = [np.ascontiguousarray(level).reshape(-1) for level in reversed(lifted.detail_levels)]
    if not pieces:
        return np.empty(0, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate(pieces)).astype(np.uint8, copy=False)


def unflatten_details(flat: np.ndarray, vectors: int, lanes: int) -> tuple[np.ndarray, ...]:
    require(flat.ndim == 1 and flat.dtype == np.uint8, "flat detail stream")
    require(flat.size == vectors * (lanes - 1), "flat detail count")
    counts = level_pair_counts(lanes)
    levels: list[np.ndarray | None] = [None] * len(counts)
    offset = 0
    for depth in range(len(counts) - 1, -1, -1):
        size = vectors * counts[depth]
        levels[depth] = np.ascontiguousarray(flat[offset : offset + size]).reshape(
            vectors, counts[depth]
        )
        offset += size
    require(offset == flat.size, "all flat details consumed")
    return tuple(level for level in levels if level is not None)


def detail_context(alphabet: int, detail_index: int) -> int:
    checks = checks_for_alphabet(alphabet)
    body = DETAIL_BLOCK - checks
    position = detail_index % DETAIL_BLOCK
    if position < body:
        return 1 + (position % checks)
    return 1 + checks + (position - body)


def state_successor(alphabet: int, state: int, detail_index: int, symbol: int) -> int:
    require(0 <= state < STATE_COUNT, "state range")
    require(0 <= symbol < alphabet, "state symbol")
    checks = checks_for_alphabet(alphabet)
    position = detail_index % DETAIL_BLOCK
    body = DETAIL_BLOCK - checks
    if position >= body:
        return state
    group = position % checks
    base = alphabet**group
    digit = (state // base) % alphabet
    updated = (digit + symbol) % alphabet
    result = state + (updated - digit) * base
    require(0 <= result < STATE_COUNT, "successor state range")
    return result


def _q16_row(counts: np.ndarray) -> np.ndarray:
    require(counts.ndim == 1 and counts.size in (2, 4), "frequency row geometry")
    require(bool(np.all(counts >= 0)), "nonnegative counts")
    adjusted = counts.astype(object) + 1
    remaining = Q16_TOTAL - counts.size
    denominator = int(sum(adjusted))
    quotients = np.asarray([int(value * remaining // denominator) for value in adjusted], dtype=np.int64)
    remainders = [int(value * remaining % denominator) for value in adjusted]
    frequencies = quotients + 1
    missing = Q16_TOTAL - int(frequencies.sum())
    order = sorted(range(counts.size), key=lambda index: (-remainders[index], index))
    for index in order[:missing]:
        frequencies[index] += 1
    require(int(frequencies.sum()) == Q16_TOTAL, "exact Q16 row sum")
    require(bool(np.all(frequencies >= 1)) and bool(np.all(frequencies <= 65535)), "Q16 range")
    return frequencies.astype(np.uint16)


@dataclass(frozen=True)
class Q16TreeModel:
    alphabet: int
    frequencies: np.ndarray  # uint16 [context,state,symbol]

    @property
    def contexts(self) -> int:
        return context_count(self.alphabet)

    def validate(self) -> None:
        checks_for_alphabet(self.alphabet)
        require(
            self.frequencies.shape == (self.contexts, STATE_COUNT, self.alphabet),
            "model frequency geometry",
        )
        require(self.frequencies.dtype == np.uint16, "model frequency dtype")
        require(bool(np.all(self.frequencies >= 1)), "positive Q16 frequency")
        sums = self.frequencies.sum(axis=2, dtype=np.uint64)
        require(bool(np.all(sums == Q16_TOTAL)), "exact Q16 normalization")

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
        return (
            header
            + bytes(MODEL_HEADER_BYTES - len(header))
            + self.frequencies.astype("<u2", copy=False).tobytes(order="C")
        )

    @classmethod
    def deserialize(cls, packet: bytes) -> "Q16TreeModel":
        require(len(packet) >= MODEL_HEADER_BYTES, "model packet header")
        magic, version, alphabet, checks, contexts, states, total = MODEL_STRUCT.unpack(
            packet[: MODEL_STRUCT.size]
        )
        require(magic == MODEL_MAGIC and version == VERSION, "model magic/version")
        require(checks == checks_for_alphabet(alphabet), "model checks")
        require(contexts == context_count(alphabet), "model contexts")
        require(states == STATE_COUNT and total == Q16_TOTAL, "model constants")
        require(all(value == 0 for value in packet[MODEL_STRUCT.size : MODEL_HEADER_BYTES]), "model header tail")
        expected = MODEL_HEADER_BYTES + 2 * contexts * states * alphabet
        require(len(packet) == expected, "model packet length")
        frequencies = np.frombuffer(packet[MODEL_HEADER_BYTES:], dtype="<u2").copy().reshape(
            contexts, states, alphabet
        )
        model = cls(int(alphabet), frequencies)
        model.validate()
        return model


def fit_model(alphabet: int, roots: np.ndarray, details: np.ndarray) -> Q16TreeModel:
    checks_for_alphabet(alphabet)
    require(roots.ndim == 1 and details.ndim == 1, "training stream geometry")
    require(roots.dtype == np.uint8 and details.dtype == np.uint8, "training stream dtype")
    require(bool(np.all(roots < alphabet)) and bool(np.all(details < alphabet)), "training alphabet")
    counts = np.zeros((context_count(alphabet), STATE_COUNT, alphabet), dtype=np.int64)
    for symbol in roots:
        counts[0, 0, int(symbol)] += 1
    state = 0
    for index, symbol_value in enumerate(details):
        if index % DETAIL_BLOCK == 0:
            state = 0
        symbol = int(symbol_value)
        context = detail_context(alphabet, index)
        counts[context, state, symbol] += 1
        state = state_successor(alphabet, state, index, symbol)
    frequencies = np.empty_like(counts, dtype=np.uint16)
    for context in range(counts.shape[0]):
        for state in range(STATE_COUNT):
            frequencies[context, state, :] = _q16_row(counts[context, state, :])
    model = Q16TreeModel(alphabet, frequencies)
    model.validate()
    return model


def generate_transformed_source(
    alphabet: int,
    vectors: int,
    lanes: int,
    seed: int,
    structured: bool,
) -> tuple[np.ndarray, np.ndarray]:
    checks = checks_for_alphabet(alphabet)
    require(vectors > 0 and lanes > 0, "positive source geometry")
    rng = np.random.default_rng(seed)
    roots = rng.integers(0, alphabet, size=vectors, dtype=np.uint8)
    detail_count = vectors * (lanes - 1)
    if detail_count == 0:
        return roots, np.empty(0, dtype=np.uint8)
    block_count = (detail_count + DETAIL_BLOCK - 1) // DETAIL_BLOCK
    body_count = DETAIL_BLOCK - checks
    body = rng.integers(0, alphabet, size=(block_count, body_count), dtype=np.uint8)
    if structured:
        tail = np.empty((block_count, checks), dtype=np.uint8)
        for group in range(checks):
            tail[:, group] = np.mod(
                body[:, group::checks].sum(axis=1, dtype=np.uint64), alphabet
            ).astype(np.uint8)
    else:
        tail = rng.integers(0, alphabet, size=(block_count, checks), dtype=np.uint8)
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
        require(0 <= symbol < len(frequencies), "arithmetic symbol")
        row = [int(value) for value in frequencies]
        require(all(value > 0 for value in row) and sum(row) == Q16_TOTAL, "arithmetic frequencies")
        cumulative_low = sum(row[:symbol])
        cumulative_high = cumulative_low + row[symbol]
        width = self.high - self.low + 1
        self.high = self.low + (width * cumulative_high // Q16_TOTAL) - 1
        self.low = self.low + (width * cumulative_low // Q16_TOTAL)
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
        meaningful = len(self.bits)
        while len(self.bits) % 8:
            self.bits.append(0)
        payload = bytearray(len(self.bits) // 8)
        for index, bit in enumerate(self.bits):
            payload[index // 8] |= bit << (7 - (index & 7))
        return bytes(payload), meaningful


class ArithmeticDecoder:
    FULL = ArithmeticEncoder.FULL
    HALF = ArithmeticEncoder.HALF
    QUARTER = ArithmeticEncoder.QUARTER
    THREE_QUARTER = ArithmeticEncoder.THREE_QUARTER

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.bit_index = 0
        self.low = 0
        self.high = self.FULL - 1
        self.code = 0
        for _ in range(32):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self) -> int:
        if self.bit_index >= 8 * len(self.payload):
            self.bit_index += 1
            return 0
        value = (self.payload[self.bit_index // 8] >> (7 - (self.bit_index & 7))) & 1
        self.bit_index += 1
        return value

    def read(self, frequencies: Sequence[int]) -> int:
        row = [int(value) for value in frequencies]
        require(all(value > 0 for value in row) and sum(row) == Q16_TOTAL, "decode frequencies")
        width = self.high - self.low + 1
        scaled = ((self.code - self.low + 1) * Q16_TOTAL - 1) // width
        cumulative_low = 0
        symbol = -1
        cumulative_high = 0
        for index, frequency in enumerate(row):
            cumulative_high = cumulative_low + frequency
            if scaled < cumulative_high:
                symbol = index
                break
            cumulative_low = cumulative_high
        require(symbol >= 0, "arithmetic scaled symbol")
        self.high = self.low + (width * cumulative_high // Q16_TOTAL) - 1
        self.low = self.low + (width * cumulative_low // Q16_TOTAL)
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
            self.code = ((self.code << 1) | self._read_bit()) & (self.FULL - 1)
        return symbol


def encode_coefficients(
    model: Q16TreeModel, roots: np.ndarray, details: np.ndarray
) -> tuple[bytes, int]:
    model.validate()
    require(roots.ndim == 1 and details.ndim == 1, "coefficient stream geometry")
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
    payload: bytes,
    root_count: int,
    detail_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.validate()
    require(root_count > 0 and detail_count >= 0, "decode coefficient counts")
    decoder = ArithmeticDecoder(payload)
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
    body_crc: int,
    header_crc: int,
) -> bytes:
    packed = FRAME_STRUCT.pack(
        FRAME_MAGIC,
        VERSION,
        expert_index,
        alphabet,
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
        body_crc,
        header_crc,
    )
    require(len(packed) <= FRAME_HEADER_BYTES, "frame header fit")
    return packed + bytes(FRAME_HEADER_BYTES - len(packed))


def build_frame(
    expert_index: int,
    model: Q16TreeModel,
    leaves: np.ndarray,
    permutation: Sequence[int],
    selectors: Sequence[int],
) -> bytes:
    require(leaves.ndim == 2, "frame leaves")
    vectors, lanes = leaves.shape
    lifted = lift_forward(leaves, model.alphabet, permutation, selectors)
    details = flatten_details(lifted)
    payload, meaningful_bits = encode_coefficients(model, lifted.roots, details)
    permutation_packet = serialize_permutation(permutation)
    selector_packet = pack_selectors(selectors)
    require(meaningful_bits <= 8 * len(payload), "meaningful arithmetic bits")
    for bit_index in range(meaningful_bits, 8 * len(payload)):
        require(
            ((payload[bit_index // 8] >> (7 - (bit_index & 7))) & 1) == 0,
            "nonzero arithmetic byte tail",
        )
    body = permutation_packet + selector_packet + payload
    logical = FRAME_HEADER_BYTES + len(body)
    padded = page_ceil(logical)
    body_crc = zlib.crc32(body) & 0xFFFFFFFF
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
        leaves.size,
        body_crc,
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
        leaves.size,
        body_crc,
        header_crc,
    )
    return header + body + bytes(padded - logical)


def parse_frame_header(frame: bytes) -> dict[str, int]:
    require(len(frame) >= FRAME_HEADER_BYTES, "frame header")
    fields = FRAME_STRUCT.unpack(frame[: FRAME_STRUCT.size])
    (
        magic,
        version,
        expert_index,
        alphabet,
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
        body_crc,
        header_crc,
    ) = fields
    require(magic == FRAME_MAGIC and version == VERSION and reserved == 0, "frame magic/version")
    checks_for_alphabet(alphabet)
    require(lanes > 0 and vectors > 0 and leaf_symbols == lanes * vectors, "frame geometry")
    require(permutation_bytes == permutation_byte_count(lanes), "frame permutation bytes")
    require(selector_bytes == (3 * (lanes - 1) + 7) // 8, "frame selector bytes")
    require(payload_bytes > 0 and meaningful_bits <= 8 * payload_bytes, "frame payload")
    require(
        logical_bytes == FRAME_HEADER_BYTES + permutation_bytes + selector_bytes + payload_bytes,
        "frame logical bytes",
    )
    require(padded_bytes == page_ceil(logical_bytes) and len(frame) == padded_bytes, "frame padded bytes")
    require(all(value == 0 for value in frame[FRAME_STRUCT.size : FRAME_HEADER_BYTES]), "frame header tail")
    zero_header = _frame_header(
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
        body_crc,
        0,
    )
    require((zlib.crc32(zero_header) & 0xFFFFFFFF) == header_crc, "frame header CRC")
    body = frame[FRAME_HEADER_BYTES:logical_bytes]
    require((zlib.crc32(body) & 0xFFFFFFFF) == body_crc, "frame body CRC")
    require(all(value == 0 for value in frame[logical_bytes:padded_bytes]), "frame page tail")
    payload_offset = FRAME_HEADER_BYTES + permutation_bytes + selector_bytes
    arithmetic = frame[payload_offset : payload_offset + payload_bytes]
    for bit_index in range(meaningful_bits, 8 * len(arithmetic)):
        require(
            ((arithmetic[bit_index // 8] >> (7 - (bit_index & 7))) & 1) == 0,
            "frame arithmetic tail",
        )
    return {
        "expert_index": int(expert_index),
        "alphabet": int(alphabet),
        "lanes": int(lanes),
        "vectors": int(vectors),
        "permutation_bytes": int(permutation_bytes),
        "selector_bytes": int(selector_bytes),
        "payload_bytes": int(payload_bytes),
        "meaningful_bits": int(meaningful_bits),
        "logical_bytes": int(logical_bytes),
        "padded_bytes": int(padded_bytes),
        "leaf_symbols": int(leaf_symbols),
    }


def decode_frame(model: Q16TreeModel, frame: bytes) -> tuple[np.ndarray, list[int], list[int]]:
    row = parse_frame_header(frame)
    require(row["alphabet"] == model.alphabet, "frame/model alphabet")
    offset = FRAME_HEADER_BYTES
    permutation_packet = frame[offset : offset + row["permutation_bytes"]]
    offset += row["permutation_bytes"]
    selector_packet = frame[offset : offset + row["selector_bytes"]]
    offset += row["selector_bytes"]
    payload = frame[offset : offset + row["payload_bytes"]]
    permutation = deserialize_permutation(row["lanes"], permutation_packet)
    selectors = unpack_selectors(selector_packet, row["lanes"] - 1)
    roots, flat = decode_coefficients(
        model,
        payload,
        row["vectors"],
        row["vectors"] * (row["lanes"] - 1),
    )
    lifted = LiftedTensor(roots, unflatten_details(flat, row["vectors"], row["lanes"]))
    leaves = lift_inverse(lifted, row["lanes"], model.alphabet, permutation, selectors)
    return leaves, permutation, selectors


def _global_header(
    alphabet: int,
    expert_count: int,
    lanes: int,
    vectors: int,
    model_offset: int,
    model_packet_bytes: int,
    model_page_bytes: int,
    frames_offset: int,
    total_bytes: int,
    total_leaf_symbols: int,
    model_sha256: bytes,
    directory: Sequence[tuple[int, int]],
    header_crc: int,
) -> bytes:
    require(len(directory) == expert_count, "directory count")
    packed = GLOBAL_STRUCT.pack(
        GLOBAL_MAGIC,
        VERSION,
        alphabet,
        expert_count,
        lanes,
        vectors,
        GLOBAL_HEADER_BYTES,
        model_offset,
        model_packet_bytes,
        model_page_bytes,
        frames_offset,
        total_bytes,
        total_leaf_symbols,
        model_sha256,
        header_crc,
    )
    entries = b"".join(DIRECTORY_ENTRY.pack(offset, length) for offset, length in directory)
    require(len(packed) + len(entries) <= GLOBAL_HEADER_BYTES, "global directory fit")
    return packed + entries + bytes(GLOBAL_HEADER_BYTES - len(packed) - len(entries))


def build_container(
    model: Q16TreeModel,
    leaves_by_expert: Sequence[np.ndarray],
    permutations: Sequence[Sequence[int]],
    selectors_by_expert: Sequence[Sequence[int]],
) -> bytes:
    model.validate()
    expert_count = len(leaves_by_expert)
    require(expert_count > 0, "positive expert count")
    require(len(permutations) == expert_count and len(selectors_by_expert) == expert_count, "metadata count")
    vectors, lanes = leaves_by_expert[0].shape
    require(vectors > 0 and lanes > 0, "container geometry")
    frames: list[bytes] = []
    for expert_index, (leaves, permutation, selectors) in enumerate(
        zip(leaves_by_expert, permutations, selectors_by_expert, strict=True)
    ):
        require(leaves.shape == (vectors, lanes), "equal expert geometry")
        frames.append(build_frame(expert_index, model, leaves, permutation, selectors))
    model_packet = model.serialize()
    model_page_bytes = page_ceil(len(model_packet))
    model_offset = GLOBAL_HEADER_BYTES
    frames_offset = model_offset + model_page_bytes
    directory: list[tuple[int, int]] = []
    offset = frames_offset
    for frame in frames:
        directory.append((offset, len(frame)))
        offset += len(frame)
    total_bytes = offset
    model_sha = hashlib.sha256(model_packet).digest()
    zero_header = _global_header(
        model.alphabet,
        expert_count,
        lanes,
        vectors,
        model_offset,
        len(model_packet),
        model_page_bytes,
        frames_offset,
        total_bytes,
        expert_count * vectors * lanes,
        model_sha,
        directory,
        0,
    )
    header_crc = zlib.crc32(zero_header) & 0xFFFFFFFF
    header = _global_header(
        model.alphabet,
        expert_count,
        lanes,
        vectors,
        model_offset,
        len(model_packet),
        model_page_bytes,
        frames_offset,
        total_bytes,
        expert_count * vectors * lanes,
        model_sha,
        directory,
        header_crc,
    )
    return header + model_packet + bytes(model_page_bytes - len(model_packet)) + b"".join(frames)


@dataclass(frozen=True)
class ParsedContainer:
    model: Q16TreeModel
    frames: tuple[bytes, ...]
    directory: tuple[tuple[int, int], ...]
    alphabet: int
    lanes: int
    vectors: int
    total_leaf_symbols: int
    model_packet_bytes: int
    model_page_bytes: int


def parse_container(packet: bytes) -> ParsedContainer:
    require(len(packet) >= GLOBAL_HEADER_BYTES, "global container header")
    fields = GLOBAL_STRUCT.unpack(packet[: GLOBAL_STRUCT.size])
    (
        magic,
        version,
        alphabet,
        expert_count,
        lanes,
        vectors,
        header_bytes,
        model_offset,
        model_packet_bytes,
        model_page_bytes,
        frames_offset,
        total_bytes,
        total_leaf_symbols,
        model_sha,
        header_crc,
    ) = fields
    require(magic == GLOBAL_MAGIC and version == VERSION, "global magic/version")
    checks_for_alphabet(alphabet)
    require(expert_count > 0 and lanes > 0 and vectors > 0, "global geometry")
    require(header_bytes == GLOBAL_HEADER_BYTES and model_offset == GLOBAL_HEADER_BYTES, "global offsets")
    require(model_page_bytes == page_ceil(model_packet_bytes), "model page length")
    require(frames_offset == model_offset + model_page_bytes, "frame start")
    require(total_bytes == len(packet), "container total bytes")
    require(total_leaf_symbols == expert_count * lanes * vectors, "global symbol count")
    directory_end = GLOBAL_STRUCT.size + expert_count * DIRECTORY_ENTRY.size
    require(directory_end <= GLOBAL_HEADER_BYTES, "directory bounds")
    directory = tuple(
        DIRECTORY_ENTRY.unpack(
            packet[
                GLOBAL_STRUCT.size + index * DIRECTORY_ENTRY.size :
                GLOBAL_STRUCT.size + (index + 1) * DIRECTORY_ENTRY.size
            ]
        )
        for index in range(expert_count)
    )
    require(all(value == 0 for value in packet[directory_end:GLOBAL_HEADER_BYTES]), "global header tail")
    zero_header = _global_header(
        alphabet,
        expert_count,
        lanes,
        vectors,
        model_offset,
        model_packet_bytes,
        model_page_bytes,
        frames_offset,
        total_bytes,
        total_leaf_symbols,
        model_sha,
        directory,
        0,
    )
    require((zlib.crc32(zero_header) & 0xFFFFFFFF) == header_crc, "global header CRC")
    model_packet = packet[model_offset : model_offset + model_packet_bytes]
    require(hashlib.sha256(model_packet).digest() == model_sha, "model packet hash")
    require(
        all(value == 0 for value in packet[model_offset + model_packet_bytes : frames_offset]),
        "model page tail",
    )
    model = Q16TreeModel.deserialize(model_packet)
    require(model.alphabet == alphabet, "global/model alphabet")
    frames: list[bytes] = []
    expected_offset = frames_offset
    for expert_index, (offset, length) in enumerate(directory):
        require(offset == expected_offset and offset % PAGE_BYTES == 0, "contiguous aligned frame")
        require(length > 0 and length % PAGE_BYTES == 0 and offset + length <= len(packet), "frame directory")
        frame = packet[offset : offset + length]
        row = parse_frame_header(frame)
        require(row["expert_index"] == expert_index, "frame expert index")
        require(row["alphabet"] == alphabet and row["lanes"] == lanes and row["vectors"] == vectors, "frame/global geometry")
        frames.append(frame)
        expected_offset = offset + length
    require(expected_offset == len(packet), "container frame coverage")
    return ParsedContainer(
        model,
        tuple(frames),
        directory,
        int(alphabet),
        int(lanes),
        int(vectors),
        int(total_leaf_symbols),
        int(model_packet_bytes),
        int(model_page_bytes),
    )


def decode_container(packet: bytes) -> list[np.ndarray]:
    parsed = parse_container(packet)
    return [decode_frame(parsed.model, frame)[0] for frame in parsed.frames]


def reencode_container(packet: bytes) -> bytes:
    parsed = parse_container(packet)
    leaves: list[np.ndarray] = []
    permutations: list[list[int]] = []
    selectors: list[list[int]] = []
    for frame in parsed.frames:
        decoded, permutation, selector = decode_frame(parsed.model, frame)
        leaves.append(decoded)
        permutations.append(permutation)
        selectors.append(selector)
    return build_container(parsed.model, leaves, permutations, selectors)


def physical_ledger(packet: bytes) -> dict[str, object]:
    parsed = parse_container(packet)
    expert_count = len(parsed.frames)
    fair_share = len(packet) / expert_count
    global_bytes = GLOBAL_HEADER_BYTES + parsed.model_page_bytes
    cold_rows = []
    for expert_index, frame in enumerate(parsed.frames):
        cold = global_bytes + len(frame)
        cold_rows.append(
            {
                "expert_index": expert_index,
                "global_bytes": global_bytes,
                "local_frame_bytes": len(frame),
                "cold_bytes": cold,
                "amplification": cold / fair_share,
            }
        )
    return {
        "container_bytes": len(packet),
        "container_sha256": sha256_bytes(packet),
        "alphabet": parsed.alphabet,
        "expert_count": expert_count,
        "lanes": parsed.lanes,
        "vectors": parsed.vectors,
        "leaf_symbols": parsed.total_leaf_symbols,
        "physical_bits_per_leaf_symbol": 8.0 * len(packet) / parsed.total_leaf_symbols,
        "global_header_bytes": GLOBAL_HEADER_BYTES,
        "model_packet_bytes": parsed.model_packet_bytes,
        "model_page_bytes": parsed.model_page_bytes,
        "fair_share_bytes": fair_share,
        "cold": cold_rows,
        "max_cold_amplification": max(float(row["amplification"]) for row in cold_rows),
        "cold_below_two": max(float(row["amplification"]) for row in cold_rows) < 2.0,
        "scope": "source-free synthetic leaf-symbol stream; not a model-weight bpw or read claim",
    }


def leaf_digest(leaves: np.ndarray) -> str:
    require(leaves.ndim == 2 and leaves.dtype == np.uint8, "leaf digest geometry")
    return sha256_bytes(np.ascontiguousarray(leaves).tobytes(order="C"))


def population_suffix_limit(alphabet: int) -> dict[str, int | float | str]:
    checks = checks_for_alphabet(alphabet)
    maximum_suffix = DETAIL_BLOCK - checks - 1 if alphabet == 2 else 28
    # Explicit values are easier to audit than an over-general statement: for
    # every check, the earliest contributing residue is outside these suffixes.
    maximum_suffix = 25 if alphabet == 2 else 28
    saving = checks * math.log2(alphabet) / DETAIL_BLOCK
    return {
        "alphabet": alphabet,
        "checks_per_32": checks,
        "maximum_suffix_depth_with_uniform_population_check": maximum_suffix,
        "ideal_structured_saving_bits_per_symbol": saving,
        "explanation": "each check omits at least one independent uniform residue-class input from such a suffix",
    }
