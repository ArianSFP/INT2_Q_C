#!/usr/bin/env python3
"""Independent parser/decoder/re-encoder for SILT source-free v1.

This file deliberately does not import ``silt_v1``.  It duplicates format,
resource, arithmetic, tree, canonicality, and owner-ledger logic so common code
cannot make the audit roundtrip vacuous.
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


PAGE = 4096
GHEADER = 4096
DHEADER = 256
DHENTRY = 32
MHEADER = 256
FHEADER = 256
TOTAL = 65536
STATES = 64
BLOCK = 32
GUARD = 30
MAX_EXPERTS = 256
MAX_LANES = 2048
MAX_VECTORS = 1 << 20
MAX_FRAME_SYMBOLS = 1 << 24
MAX_TOTAL_SYMBOLS = 1 << 28
MAX_MODEL_BYTES = 1 << 20
MAX_DIRECTORY_BYTES = 1 << 16
MAX_FRAME_LOGICAL = 1 << 26
MAX_FRAME_PADDED = MAX_FRAME_LOGICAL + PAGE
MAX_CONTAINER = 1 << 28

GMAGIC = b"SILTV1G\0"
DMAGIC = b"SILTV1D\0"
MMAGIC = b"SILTV1M\0"
FMAGIC = b"SILTV1F\0"
GSTRUCT = struct.Struct("<8sHHIIIIQQQQQQQQQ32s32sI")
DHSTRUCT = struct.Struct("<8sHHIII32sI")
DESTRUCT = struct.Struct("<QQQQ")
MSTRUCT = struct.Struct("<8sHHHHII")
FSTRUCT = struct.Struct("<8sHHIIIIIIQQQQQ32sI")


class IndependentFormatError(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentFormatError(message)


def scalar(value: int, low: int, high: int, name: str) -> int:
    check(isinstance(value, (int, np.integer)), f"{name} integer")
    output = int(value)
    check(low <= output <= high, f"{name} range")
    return output


def add(left: int, right: int, maximum: int, name: str) -> int:
    left = scalar(left, 0, maximum, f"{name} left")
    right = scalar(right, 0, maximum, f"{name} right")
    check(left <= maximum - right, f"{name} overflow")
    return left + right


def mul(left: int, right: int, maximum: int, name: str) -> int:
    left = scalar(left, 0, maximum, f"{name} left")
    right = scalar(right, 0, maximum, f"{name} right")
    check(left == 0 or right <= maximum // left, f"{name} overflow")
    return left * right


def align(value: int) -> int:
    value = scalar(value, 0, MAX_CONTAINER, "alignment")
    return 0 if value == 0 else mul((value + PAGE - 1) // PAGE, PAGE, MAX_CONTAINER, "aligned")


def alphabet_check(alphabet: int) -> int:
    alphabet = int(alphabet)
    check(alphabet in (2, 4), "alphabet")
    return alphabet


def geometry(lanes: int, vectors: int) -> int:
    lanes = scalar(lanes, 1, MAX_LANES, "lanes")
    vectors = scalar(vectors, 1, MAX_VECTORS, "vectors")
    return mul(lanes, vectors, MAX_FRAME_SYMBOLS, "frame symbols")


def checks(alphabet: int) -> int:
    return 6 if alphabet_check(alphabet) == 2 else 3


def contexts(alphabet: int) -> int:
    return 1 + 2 * checks(alphabet)


def pairs(lanes: int) -> list[int]:
    lanes = scalar(lanes, 1, MAX_LANES, "pair lanes")
    output: list[int] = []
    width = lanes
    while width > 1:
        output.append(width // 2)
        width = (width + 1) // 2
    check(sum(output) == lanes - 1, "pair coverage")
    return output


def widths(lanes: int) -> list[int]:
    output = [scalar(lanes, 1, MAX_LANES, "width lanes")]
    while output[-1] > 1:
        output.append((output[-1] + 1) // 2)
    return output


@lru_cache(maxsize=MAX_LANES)
def permutation_bytes(lanes: int) -> int:
    lanes = scalar(lanes, 1, MAX_LANES, "factorial lanes")
    return ((math.factorial(lanes) - 1).bit_length() + 7) // 8


def decode_permutation(lanes: int, packet: bytes) -> list[int]:
    lanes = scalar(lanes, 1, MAX_LANES, "decode permutation lanes")
    check(len(packet) == permutation_bytes(lanes), "permutation length")
    rank = int.from_bytes(packet, "big") if packet else 0
    check(rank < math.factorial(lanes), "permutation canonical rank")
    pool = list(range(lanes))
    output: list[int] = []
    for width in range(lanes, 0, -1):
        factorial = math.factorial(width - 1)
        index, rank = divmod(rank, factorial)
        output.append(pool.pop(index))
    check(sorted(output) == list(range(lanes)), "permutation domain")
    return output


def encode_permutation(permutation: Sequence[int]) -> bytes:
    lanes = scalar(len(permutation), 1, MAX_LANES, "encode permutation lanes")
    pool = list(range(lanes))
    rank = 0
    for position, value in enumerate(permutation):
        check(int(value) in pool, "permutation duplicate")
        index = pool.index(int(value))
        rank += index * math.factorial(lanes - position - 1)
        pool.pop(index)
    count = permutation_bytes(lanes)
    return rank.to_bytes(count, "big") if count else b""


def selector_limit(alphabet: int) -> int:
    return 6 if alphabet_check(alphabet) == 2 else 8


def decode_selectors(packet: bytes, count: int, alphabet: int) -> list[int]:
    count = scalar(count, 0, MAX_LANES - 1, "selector count")
    check(len(packet) == (3 * count + 7) // 8, "selector bytes")
    limit = selector_limit(alphabet)
    output: list[int] = []
    for index in range(count):
        value = 0
        for offset in range(3):
            bit_index = 3 * index + offset
            value = (value << 1) | ((packet[bit_index // 8] >> (7 - (bit_index & 7))) & 1)
        check(value < limit, "canonical selector")
        output.append(value)
    for bit_index in range(3 * count, 8 * len(packet)):
        check(((packet[bit_index // 8] >> (7 - (bit_index & 7))) & 1) == 0, "selector tail")
    return output


def encode_selectors(values: Sequence[int], alphabet: int) -> bytes:
    limit = selector_limit(alphabet)
    packet = bytearray((3 * len(values) + 7) // 8)
    for index, value in enumerate(values):
        value = int(value)
        check(0 <= value < limit, "selector value")
        for offset in range(3):
            bit_index = 3 * index + offset
            packet[bit_index // 8] |= ((value >> (2 - offset)) & 1) << (7 - (bit_index & 7))
    return bytes(packet)


def detail_context(alphabet: int, index: int) -> int:
    count = checks(alphabet)
    position = index % BLOCK
    body = BLOCK - count
    return 1 + position % count if position < body else 1 + count + position - body


def successor(alphabet: int, state: int, index: int, symbol: int) -> int:
    count = checks(alphabet)
    position = index % BLOCK
    if position >= BLOCK - count:
        return state
    group = position % count
    base = alphabet**group
    old = (state // base) % alphabet
    new = (old + symbol) % alphabet
    output = state + (new - old) * base
    check(0 <= output < STATES, "state")
    return output


@dataclass(frozen=True)
class Model:
    alphabet: int
    frequencies: np.ndarray
    packet: bytes


def parse_model(packet: bytes) -> Model:
    scalar(len(packet), MHEADER, MAX_MODEL_BYTES, "model bytes")
    magic, version, alphabet, check_count, context_count, states, total = MSTRUCT.unpack(packet[: MSTRUCT.size])
    alphabet = alphabet_check(alphabet)
    check(magic == MMAGIC and version == 1, "model magic")
    check(check_count == checks(alphabet) and context_count == contexts(alphabet), "model contexts")
    check(states == STATES and total == TOTAL, "model constants")
    check(not any(packet[MSTRUCT.size:MHEADER]), "model header tail")
    values = mul(mul(context_count, states, 1 << 20, "model states"), alphabet, 1 << 20, "model values")
    expected = add(MHEADER, 2 * values, MAX_MODEL_BYTES, "model length")
    check(len(packet) == expected, "model packet length")
    frequencies = np.frombuffer(packet[MHEADER:], dtype="<u2", count=values).copy().reshape(context_count, states, alphabet)
    check(bool(np.all(frequencies >= 1)), "model positive")
    check(bool(np.all(frequencies.sum(axis=2, dtype=np.uint64) == TOTAL)), "model sums")
    return Model(alphabet, frequencies, packet)


class Reader:
    def __init__(self, packet: bytes, meaningful: int) -> None:
        self.packet = packet
        total = mul(8, len(packet), MAX_FRAME_LOGICAL * 8, "reader bits")
        self.meaningful = scalar(meaningful, 32, total, "meaningful")
        self.position = 0
        for bit in range(self.meaningful, total):
            check(self.raw(bit) == 0, "reader byte tail")
        for bit in range(self.meaningful - GUARD, self.meaningful):
            check(self.raw(bit) == 0, "reader guard")

    def raw(self, index: int) -> int:
        return (self.packet[index // 8] >> (7 - (index & 7))) & 1

    def bit(self) -> int:
        check(self.position < self.meaningful, "reader EOF")
        output = self.raw(self.position)
        self.position += 1
        return output


class Decoder:
    FULL = 1 << 32
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREE = 3 << 30

    def __init__(self, packet: bytes, meaningful: int) -> None:
        self.reader = Reader(packet, meaningful)
        self.low = 0
        self.high = self.FULL - 1
        self.code = 0
        for _ in range(32):
            self.code = (self.code << 1) | self.reader.bit()

    def symbol(self, frequencies: Sequence[int]) -> int:
        row = [int(value) for value in frequencies]
        check(all(value > 0 for value in row) and sum(row) == TOTAL, "decode row")
        width = self.high - self.low + 1
        target = ((self.code - self.low + 1) * TOTAL - 1) // width
        lower = 0
        selected = -1
        upper = 0
        for index, frequency in enumerate(row):
            upper = lower + frequency
            if target < upper:
                selected = index
                break
            lower = upper
        check(selected >= 0, "decode target")
        self.high = self.low + width * upper // TOTAL - 1
        self.low = self.low + width * lower // TOTAL
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.low -= self.HALF
                self.high -= self.HALF
                self.code -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE:
                self.low -= self.QUARTER
                self.high -= self.QUARTER
                self.code -= self.QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = ((self.code << 1) | self.reader.bit()) & (self.FULL - 1)
        return selected


class Encoder:
    FULL = 1 << 32
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREE = 3 << 30

    def __init__(self) -> None:
        self.low = 0
        self.high = self.FULL - 1
        self.pending = 0
        self.bits: list[int] = []

    def emit(self, bit: int) -> None:
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)
        self.pending = 0

    def symbol(self, selected: int, frequencies: Sequence[int]) -> None:
        row = [int(value) for value in frequencies]
        check(0 <= selected < len(row) and all(value > 0 for value in row) and sum(row) == TOTAL, "encode row")
        lower = sum(row[:selected])
        upper = lower + row[selected]
        width = self.high - self.low + 1
        self.high = self.low + width * upper // TOTAL - 1
        self.low = self.low + width * lower // TOTAL
        while True:
            if self.high < self.HALF:
                self.emit(0)
            elif self.low >= self.HALF:
                self.emit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE:
                self.pending += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> tuple[bytes, int]:
        self.pending += 1
        self.emit(0 if self.low < self.QUARTER else 1)
        self.bits.extend([0] * GUARD)
        meaningful = len(self.bits)
        while len(self.bits) % 8:
            self.bits.append(0)
        packet = bytearray(len(self.bits) // 8)
        for index, bit in enumerate(self.bits):
            packet[index // 8] |= bit << (7 - (index & 7))
        return bytes(packet), meaningful


def decode_coefficients(model: Model, packet: bytes, meaningful: int, roots_count: int, details_count: int) -> tuple[np.ndarray, np.ndarray]:
    scalar(roots_count, 1, MAX_VECTORS, "root count")
    scalar(details_count, 0, MAX_FRAME_SYMBOLS, "detail count")
    add(roots_count, details_count, MAX_FRAME_SYMBOLS, "coefficient total")
    decoder = Decoder(packet, meaningful)
    roots = np.empty(roots_count, dtype=np.uint8)
    for index in range(roots_count):
        roots[index] = decoder.symbol(model.frequencies[0, 0, :])
    details = np.empty(details_count, dtype=np.uint8)
    state = 0
    for index in range(details_count):
        if index % BLOCK == 0:
            state = 0
        context = detail_context(model.alphabet, index)
        value = decoder.symbol(model.frequencies[context, state, :])
        details[index] = value
        state = successor(model.alphabet, state, index, value)
    check(decoder.reader.position == meaningful, "exact meaningful exhaustion")
    return roots, details


def encode_coefficients(model: Model, roots: np.ndarray, details: np.ndarray) -> tuple[bytes, int]:
    encoder = Encoder()
    for value in roots:
        encoder.symbol(int(value), model.frequencies[0, 0, :])
    state = 0
    for index, value in enumerate(details):
        if index % BLOCK == 0:
            state = 0
        context = detail_context(model.alphabet, index)
        encoder.symbol(int(value), model.frequencies[context, state, :])
        state = successor(model.alphabet, state, index, int(value))
    return encoder.finish()


def split_details(flat: np.ndarray, vectors: int, lanes: int) -> list[np.ndarray]:
    count_rows = pairs(lanes)
    output: list[np.ndarray | None] = [None] * len(count_rows)
    offset = 0
    for depth in range(len(count_rows) - 1, -1, -1):
        count = mul(vectors, count_rows[depth], MAX_FRAME_SYMBOLS, "detail split")
        output[depth] = flat[offset : offset + count].reshape(vectors, count_rows[depth]).copy()
        offset += count
    check(offset == flat.size, "detail coverage")
    return [value for value in output if value is not None]


def join_details(levels: Sequence[np.ndarray]) -> np.ndarray:
    if not levels:
        return np.empty(0, dtype=np.uint8)
    return np.concatenate([np.ascontiguousarray(level).reshape(-1) for level in reversed(levels)]).astype(np.uint8, copy=False)


def invert_tree(
    roots: np.ndarray,
    levels: Sequence[np.ndarray],
    alphabet: int,
    lanes: int,
    permutation: Sequence[int],
    selectors: Sequence[int],
) -> np.ndarray:
    count_rows = pairs(lanes)
    width_rows = widths(lanes)
    offsets = np.cumsum([0] + count_rows).tolist()
    current = roots[:, None].copy()
    for depth in range(len(count_rows) - 1, -1, -1):
        pair_count = count_rows[depth]
        previous = np.empty((roots.size, width_rows[depth]), dtype=np.uint8)
        detail = levels[depth].astype(np.int16)
        coarse = current[:, :pair_count].astype(np.int16)
        for pair in range(pair_count):
            code = int(selectors[offsets[depth] + pair])
            swap, p, u = (code >> 2) & 1, (code >> 1) & 1, code & 1
            x = np.mod(coarse[:, pair] - u * detail[:, pair], alphabet)
            y = np.mod(detail[:, pair] + p * x, alphabet)
            previous[:, 2 * pair] = y if swap else x
            previous[:, 2 * pair + 1] = x if swap else y
        if width_rows[depth] & 1:
            previous[:, -1] = current[:, -1]
        current = previous
    output = np.empty_like(current)
    output[:, np.asarray(permutation, dtype=np.int64)] = current
    return output


def forward_tree(
    leaves: np.ndarray,
    alphabet: int,
    permutation: Sequence[int],
    selectors: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    current = leaves[:, np.asarray(permutation, dtype=np.int64)].copy()
    output: list[np.ndarray] = []
    offset = 0
    for pair_count in pairs(leaves.shape[1]):
        next_values = np.empty((leaves.shape[0], pair_count + current.shape[1] % 2), dtype=np.uint8)
        details = np.empty((leaves.shape[0], pair_count), dtype=np.uint8)
        for pair in range(pair_count):
            code = int(selectors[offset + pair])
            swap, p, u = (code >> 2) & 1, (code >> 1) & 1, code & 1
            left = current[:, 2 * pair].astype(np.int16)
            right = current[:, 2 * pair + 1].astype(np.int16)
            x, y = (right, left) if swap else (left, right)
            detail = np.mod(y - p * x, alphabet)
            details[:, pair] = detail
            next_values[:, pair] = np.mod(x + u * detail, alphabet)
        if current.shape[1] & 1:
            next_values[:, -1] = current[:, -1]
        output.append(details)
        current = next_values
        offset += pair_count
    check(current.shape[1] == 1 and offset == len(selectors), "forward coverage")
    return current[:, 0].copy(), output


def frame_header(
    expert: int,
    alphabet: int,
    lanes: int,
    vectors: int,
    pbytes: int,
    sbytes: int,
    payload_bytes: int,
    meaningful: int,
    logical: int,
    padded: int,
    symbols: int,
    body_hash: bytes,
    crc: int,
) -> bytes:
    raw = FSTRUCT.pack(
        FMAGIC,
        1,
        alphabet,
        expert,
        0,
        lanes,
        vectors,
        pbytes,
        sbytes,
        payload_bytes,
        meaningful,
        logical,
        padded,
        symbols,
        body_hash,
        crc,
    )
    return raw + bytes(FHEADER - len(raw))


@dataclass(frozen=True)
class Frame:
    expert: int
    alphabet: int
    lanes: int
    vectors: int
    permutation: list[int]
    selectors: list[int]
    payload: bytes
    meaningful: int
    logical: int
    padded: int
    symbols: int
    packet: bytes


def parse_frame(packet_view: bytes | memoryview) -> Frame:
    packet_length = scalar(len(packet_view), FHEADER, MAX_FRAME_PADDED, "frame length")
    fields = FSTRUCT.unpack(bytes(packet_view[: FSTRUCT.size]))
    (
        magic,
        version,
        alphabet,
        expert,
        reserved,
        lanes,
        vectors,
        pbytes,
        sbytes,
        payload_bytes,
        meaningful,
        logical,
        padded,
        symbols,
        body_hash,
        crc,
    ) = fields
    check(magic == FMAGIC and version == 1 and reserved == 0, "frame magic")
    alphabet = alphabet_check(alphabet)
    expert = scalar(expert, 0, MAX_EXPERTS - 1, "frame expert")
    expected_symbols = geometry(lanes, vectors)
    check(symbols == expected_symbols, "frame symbols")
    check(pbytes == permutation_bytes(lanes), "frame permutation bytes")
    check(sbytes == (3 * (lanes - 1) + 7) // 8, "frame selector bytes")
    payload_bytes = scalar(payload_bytes, 1, MAX_FRAME_LOGICAL, "payload bytes")
    meaningful = scalar(meaningful, 32, payload_bytes * 8, "meaningful bits")
    check(payload_bytes == (meaningful + 7) // 8, "canonical payload bytes")
    expected_logical = add(FHEADER, add(add(pbytes, sbytes, MAX_FRAME_LOGICAL, "metadata"), payload_bytes, MAX_FRAME_LOGICAL, "body"), MAX_FRAME_LOGICAL, "logical")
    check(logical == expected_logical and padded == align(logical) == packet_length, "frame lengths")
    check(padded <= MAX_FRAME_PADDED, "frame padded cap")
    check(not any(packet_view[FSTRUCT.size:FHEADER]), "frame header tail")
    zero = frame_header(expert, alphabet, lanes, vectors, pbytes, sbytes, payload_bytes, meaningful, logical, padded, symbols, body_hash, 0)
    check(zlib.crc32(zero) & 0xFFFFFFFF == crc, "frame CRC")
    body = bytes(packet_view[FHEADER:logical])
    check(hashlib.sha256(body).digest() == body_hash, "frame body hash")
    check(not any(packet_view[logical:padded]), "frame page tail")
    offset = FHEADER
    permutation_packet = bytes(packet_view[offset : offset + pbytes])
    offset += pbytes
    selector_packet = bytes(packet_view[offset : offset + sbytes])
    offset += sbytes
    payload = bytes(packet_view[offset : offset + payload_bytes])
    Reader(payload, meaningful)
    return Frame(
        expert,
        alphabet,
        lanes,
        vectors,
        decode_permutation(lanes, permutation_packet),
        decode_selectors(selector_packet, lanes - 1, alphabet),
        payload,
        meaningful,
        logical,
        padded,
        symbols,
        bytes(packet_view),
    )


def decode_frame(model: Model, frame: Frame) -> np.ndarray:
    check(model.alphabet == frame.alphabet, "frame/model alphabet")
    roots, details = decode_coefficients(
        model,
        frame.payload,
        frame.meaningful,
        frame.vectors,
        frame.symbols - frame.vectors,
    )
    canonical, meaningful = encode_coefficients(model, roots, details)
    check(meaningful == frame.meaningful and canonical == frame.payload, "ordinary canonical arithmetic")
    levels = split_details(details, frame.vectors, frame.lanes)
    return invert_tree(roots, levels, frame.alphabet, frame.lanes, frame.permutation, frame.selectors)


def rebuild_frame(model: Model, frame: Frame, leaves: np.ndarray) -> bytes:
    roots, levels = forward_tree(leaves, frame.alphabet, frame.permutation, frame.selectors)
    payload, meaningful = encode_coefficients(model, roots, join_details(levels))
    check(payload == frame.payload and meaningful == frame.meaningful, "frame canonical rebuild payload")
    permutation_packet = encode_permutation(frame.permutation)
    selector_packet = encode_selectors(frame.selectors, frame.alphabet)
    body = permutation_packet + selector_packet + payload
    logical = FHEADER + len(body)
    padded = align(logical)
    body_hash = hashlib.sha256(body).digest()
    zero = frame_header(frame.expert, frame.alphabet, frame.lanes, frame.vectors, len(permutation_packet), len(selector_packet), len(payload), meaningful, logical, padded, leaves.size, body_hash, 0)
    crc = zlib.crc32(zero) & 0xFFFFFFFF
    packet = frame_header(frame.expert, frame.alphabet, frame.lanes, frame.vectors, len(permutation_packet), len(selector_packet), len(payload), meaningful, logical, padded, leaves.size, body_hash, crc) + body + bytes(padded - logical)
    check(packet == frame.packet, "frame byte rebuild")
    return packet


def directory_header(experts: int, entries_bytes: int, packet_bytes: int, digest: bytes, crc: int) -> bytes:
    raw = DHSTRUCT.pack(DMAGIC, 1, DHENTRY, experts, entries_bytes, packet_bytes, digest, crc)
    return raw + bytes(DHEADER - len(raw))


def global_header(
    alphabet: int,
    experts: int,
    directory_offset: int,
    directory_bytes: int,
    directory_pages: int,
    model_offset: int,
    model_bytes: int,
    model_pages: int,
    frames_offset: int,
    total_bytes: int,
    symbols: int,
    directory_hash: bytes,
    model_hash: bytes,
    crc: int,
) -> bytes:
    raw = GSTRUCT.pack(
        GMAGIC,
        1,
        alphabet,
        experts,
        GHEADER,
        DHENTRY,
        0,
        directory_offset,
        directory_bytes,
        directory_pages,
        model_offset,
        model_bytes,
        model_pages,
        frames_offset,
        total_bytes,
        symbols,
        directory_hash,
        model_hash,
        crc,
    )
    return raw + bytes(GHEADER - len(raw))


@dataclass(frozen=True)
class Entry:
    offset: int
    padded: int
    logical: int
    symbols: int


@dataclass(frozen=True)
class Container:
    packet: bytes
    alphabet: int
    experts: int
    model: Model
    entries: tuple[Entry, ...]
    frames: tuple[Frame, ...]
    directory_packet: bytes
    directory_pages: int
    model_pages: int
    frames_offset: int
    total_symbols: int


def parse_directory(packet: bytes, expected_experts: int) -> tuple[Entry, ...]:
    scalar(len(packet), DHEADER, MAX_DIRECTORY_BYTES, "directory length")
    magic, version, entry_bytes, experts, entries_bytes, packet_bytes, digest, crc = DHSTRUCT.unpack(packet[: DHSTRUCT.size])
    check(magic == DMAGIC and version == 1 and entry_bytes == DHENTRY, "directory constants")
    experts = scalar(experts, 1, MAX_EXPERTS, "directory experts")
    check(experts == expected_experts, "directory expert agreement")
    expected_entries = mul(experts, DHENTRY, MAX_DIRECTORY_BYTES, "directory entries")
    check(entries_bytes == expected_entries and packet_bytes == DHEADER + entries_bytes == len(packet), "directory lengths")
    check(not any(packet[DHSTRUCT.size:DHEADER]), "directory header tail")
    rows = packet[DHEADER:]
    check(hashlib.sha256(rows).digest() == digest, "directory rows hash")
    zero = directory_header(experts, entries_bytes, packet_bytes, digest, 0)
    check(zlib.crc32(zero) & 0xFFFFFFFF == crc, "directory CRC")
    output: list[Entry] = []
    for index in range(experts):
        start = DHEADER + index * DHENTRY
        output.append(Entry(*(int(value) for value in DESTRUCT.unpack(packet[start : start + DHENTRY]))))
    return tuple(output)


def serialize_directory(entries: Sequence[Entry]) -> bytes:
    experts = scalar(len(entries), 1, MAX_EXPERTS, "serialize directory experts")
    rows = b"".join(DESTRUCT.pack(entry.offset, entry.padded, entry.logical, entry.symbols) for entry in entries)
    entries_bytes = experts * DHENTRY
    digest = hashlib.sha256(rows).digest()
    packet_bytes = DHEADER + entries_bytes
    zero = directory_header(experts, entries_bytes, packet_bytes, digest, 0)
    crc = zlib.crc32(zero) & 0xFFFFFFFF
    return directory_header(experts, entries_bytes, packet_bytes, digest, crc) + rows


def parse_container(packet: bytes) -> Container:
    total_length = scalar(len(packet), GHEADER, MAX_CONTAINER, "container length")
    fields = GSTRUCT.unpack(packet[: GSTRUCT.size])
    (
        magic,
        version,
        alphabet,
        experts,
        header_bytes,
        entry_bytes,
        reserved,
        directory_offset,
        directory_bytes,
        directory_pages,
        model_offset,
        model_bytes,
        model_pages,
        frames_offset,
        total_bytes,
        symbols,
        directory_hash,
        model_hash,
        crc,
    ) = fields
    check(magic == GMAGIC and version == 1, "global magic")
    alphabet = alphabet_check(alphabet)
    experts = scalar(experts, 1, MAX_EXPERTS, "global experts")
    check(header_bytes == directory_offset == GHEADER and entry_bytes == DHENTRY and reserved == 0, "global constants")
    check(not any(packet[GSTRUCT.size:GHEADER]), "global header tail")
    expected_directory = DHEADER + mul(experts, DHENTRY, MAX_DIRECTORY_BYTES, "global directory product")
    check(directory_bytes == expected_directory and directory_pages == align(directory_bytes), "global directory bytes")
    check(model_offset == directory_offset + directory_pages and model_offset <= MAX_CONTAINER, "global model offset")
    model_bytes = scalar(model_bytes, MHEADER, MAX_MODEL_BYTES, "global model bytes")
    check(model_pages == align(model_bytes), "global model pages")
    check(frames_offset == model_offset + model_pages and frames_offset <= MAX_CONTAINER, "global frames offset")
    check(total_bytes == total_length and 1 <= symbols <= MAX_TOTAL_SYMBOLS, "global totals")
    zero = global_header(alphabet, experts, directory_offset, directory_bytes, directory_pages, model_offset, model_bytes, model_pages, frames_offset, total_bytes, symbols, directory_hash, model_hash, 0)
    check(zlib.crc32(zero) & 0xFFFFFFFF == crc, "global CRC")
    check(directory_offset + directory_bytes <= total_length, "directory bounds")
    directory_packet = packet[directory_offset : directory_offset + directory_bytes]
    check(hashlib.sha256(directory_packet).digest() == directory_hash, "directory packet hash")
    check(not any(packet[directory_offset + directory_bytes : model_offset]), "directory page tail")
    entries = parse_directory(directory_packet, experts)
    check(model_offset + model_bytes <= total_length, "model bounds")
    model_packet = packet[model_offset : model_offset + model_bytes]
    check(hashlib.sha256(model_packet).digest() == model_hash, "model hash")
    check(not any(packet[model_offset + model_bytes : frames_offset]), "model page tail")
    model = parse_model(model_packet)
    check(model.alphabet == alphabet, "model alphabet")
    expected_offset = frames_offset
    observed_symbols = 0
    frames: list[Frame] = []
    view = memoryview(packet)
    for index, entry in enumerate(entries):
        offset = scalar(entry.offset, frames_offset, MAX_CONTAINER, "entry offset")
        padded = scalar(entry.padded, PAGE, MAX_FRAME_PADDED, "entry padded")
        logical = scalar(entry.logical, FHEADER, MAX_FRAME_LOGICAL, "entry logical")
        frame_symbols = scalar(entry.symbols, 1, MAX_FRAME_SYMBOLS, "entry symbols")
        check(offset == expected_offset and offset % PAGE == 0, "entry offset")
        check(padded % PAGE == 0 and logical <= padded, "entry pages")
        end = add(offset, padded, MAX_CONTAINER, "entry end")
        check(end <= total_length, "entry bounds")
        frame = parse_frame(view[offset:end])
        check(frame.expert == index and frame.alphabet == alphabet, "frame identity")
        check(frame.padded == padded and frame.logical == logical and frame.symbols == frame_symbols, "entry frame agreement")
        frames.append(frame)
        expected_offset = end
        observed_symbols = add(observed_symbols, frame_symbols, MAX_TOTAL_SYMBOLS, "symbol total")
    check(expected_offset == total_length and observed_symbols == symbols, "container coverage")
    return Container(packet, alphabet, experts, model, entries, tuple(frames), directory_packet, directory_pages, model_pages, frames_offset, symbols)


def verify_decode_reencode(packet: bytes, expected_leaf_sha256: Sequence[str] | None = None) -> tuple[dict[str, object], list[np.ndarray], bytes]:
    container = parse_container(packet)
    leaves: list[np.ndarray] = []
    rebuilt_frames: list[bytes] = []
    for frame in container.frames:
        decoded = decode_frame(container.model, frame)
        leaves.append(decoded)
        rebuilt_frames.append(rebuild_frame(container.model, frame, decoded))
    if expected_leaf_sha256 is not None:
        check(len(expected_leaf_sha256) == len(leaves), "digest count")
        for expected, values in zip(expected_leaf_sha256, leaves, strict=True):
            observed = hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()
            check(observed == expected, "leaf digest")
    directory_packet = serialize_directory(container.entries)
    check(directory_packet == container.directory_packet, "directory byte rebuild")
    directory_hash = hashlib.sha256(directory_packet).digest()
    model_hash = hashlib.sha256(container.model.packet).digest()
    total = container.frames_offset + sum(len(frame) for frame in rebuilt_frames)
    zero = global_header(container.alphabet, container.experts, GHEADER, len(directory_packet), container.directory_pages, GHEADER + container.directory_pages, len(container.model.packet), container.model_pages, container.frames_offset, total, container.total_symbols, directory_hash, model_hash, 0)
    crc = zlib.crc32(zero) & 0xFFFFFFFF
    header = global_header(container.alphabet, container.experts, GHEADER, len(directory_packet), container.directory_pages, GHEADER + container.directory_pages, len(container.model.packet), container.model_pages, container.frames_offset, total, container.total_symbols, directory_hash, model_hash, crc)
    rebuilt = header + directory_packet + bytes(container.directory_pages - len(directory_packet)) + container.model.packet + bytes(container.model_pages - len(container.model.packet)) + b"".join(rebuilt_frames)
    check(rebuilt == packet, "container byte rebuild")
    owner_sum = 0
    cold_rows: list[dict[str, object]] = []
    for index, frame in enumerate(rebuilt_frames):
        owner_numerator = container.frames_offset + container.experts * len(frame)
        owner_sum += owner_numerator
        numerator = container.experts * (container.frames_offset + len(frame))
        amplification = Fraction(numerator, owner_numerator)
        cold_rows.append(
            {
                "expert_index": index,
                "owner_numerator": owner_numerator,
                "owner_denominator": container.experts,
                "amplification_numerator": amplification.numerator,
                "amplification_denominator": amplification.denominator,
                "below_two": numerator < 2 * owner_numerator,
            }
        )
    check(owner_sum == container.experts * len(packet), "independent owner conservation")
    return (
        {
            "status": "PASS_INDEPENDENT_V1_CANONICAL_DECODE_REENCODE",
            "container_sha256": hashlib.sha256(packet).hexdigest(),
            "experts": container.experts,
            "owner_sum_equals_container": True,
            "cold": cold_rows,
            "source_gain_claim": False,
        },
        leaves,
        rebuilt,
    )
