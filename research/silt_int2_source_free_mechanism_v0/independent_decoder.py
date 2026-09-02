#!/usr/bin/env python3
"""Independent SILT synthetic-container decoder and byte re-encoder.

This file intentionally does not import ``silt_mechanism``.  Format constants,
tree inversion, arithmetic decoding/encoding, CRC checks and container rebuild
are implemented again so a common helper cannot make the roundtrip vacuous.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np


PAGE = 4096
GLOBAL_HEADER = 4096
MODEL_HEADER = 256
FRAME_HEADER = 256
TOTAL = 65536
STATES = 64
BLOCK = 32
GMAGIC = b"SILTSF0\0"
MMAGIC = b"SILTMOD0"
FMAGIC = b"SILTFR0\0"
GSTRUCT = struct.Struct("<8sHHIIIIQIIQQQ32sI")
FSTRUCT = struct.Struct("<8sHHHHIIIIIQIIQII")
MSTRUCT = struct.Struct("<8sHHHHII")
DIRENT = struct.Struct("<QQ")


class IndependentError(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentError(message)


def align(value: int) -> int:
    check(value >= 0, "negative length")
    return ((value + PAGE - 1) // PAGE) * PAGE


def check_count(alphabet: int) -> int:
    check(alphabet in (2, 4), "alphabet")
    return 6 if alphabet == 2 else 3


def contexts(alphabet: int) -> int:
    return 1 + 2 * check_count(alphabet)


def pair_counts(lanes: int) -> list[int]:
    check(lanes > 0, "lanes")
    values: list[int] = []
    width = lanes
    while width > 1:
        values.append(width // 2)
        width = width // 2 + width % 2
    check(sum(values) == lanes - 1, "tree node count")
    return values


def sizes(lanes: int) -> list[int]:
    result = [lanes]
    for pairs in pair_counts(lanes):
        result.append(pairs + result[-1] % 2)
    return result


def perm_bytes(lanes: int) -> int:
    return ((math.factorial(lanes) - 1).bit_length() + 7) // 8


def decode_permutation(lanes: int, packet: bytes) -> list[int]:
    check(len(packet) == perm_bytes(lanes), "permutation bytes")
    rank = int.from_bytes(packet, "big") if packet else 0
    check(rank < math.factorial(lanes), "permutation rank")
    pool = list(range(lanes))
    output: list[int] = []
    for width in range(lanes, 0, -1):
        factorial = math.factorial(width - 1)
        index, rank = divmod(rank, factorial)
        output.append(pool.pop(index))
    check(sorted(output) == list(range(lanes)), "permutation domain")
    return output


def encode_permutation(permutation: Sequence[int]) -> bytes:
    pool = list(range(len(permutation)))
    rank = 0
    for position, value in enumerate(permutation):
        check(int(value) in pool, "permutation duplicate")
        index = pool.index(int(value))
        rank += index * math.factorial(len(permutation) - position - 1)
        pool.pop(index)
    count = perm_bytes(len(permutation))
    return rank.to_bytes(count, "big") if count else b""


def decode_selectors(packet: bytes, count: int) -> list[int]:
    check(len(packet) == (3 * count + 7) // 8, "selector bytes")
    result: list[int] = []
    for index in range(count):
        value = 0
        for offset in range(3):
            bit_index = 3 * index + offset
            value = (value << 1) | (
                (packet[bit_index // 8] >> (7 - (bit_index & 7))) & 1
            )
        result.append(value)
    for bit_index in range(3 * count, 8 * len(packet)):
        check(
            ((packet[bit_index // 8] >> (7 - (bit_index & 7))) & 1) == 0,
            "selector tail",
        )
    return result


def encode_selectors(values: Sequence[int]) -> bytes:
    packet = bytearray((3 * len(values) + 7) // 8)
    for index, value in enumerate(values):
        check(0 <= int(value) < 8, "selector value")
        for offset in range(3):
            bit = (int(value) >> (2 - offset)) & 1
            bit_index = 3 * index + offset
            packet[bit_index // 8] |= bit << (7 - (bit_index & 7))
    return bytes(packet)


def detail_ctx(alphabet: int, index: int) -> int:
    count = check_count(alphabet)
    body = BLOCK - count
    position = index % BLOCK
    return 1 + position % count if position < body else 1 + count + position - body


def advance(alphabet: int, state: int, index: int, symbol: int) -> int:
    count = check_count(alphabet)
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
class IndependentModel:
    alphabet: int
    frequencies: np.ndarray
    packet: bytes


def parse_model(packet: bytes) -> IndependentModel:
    check(len(packet) >= MODEL_HEADER, "model header")
    magic, version, alphabet, count, context_count, state_count, total = MSTRUCT.unpack(
        packet[: MSTRUCT.size]
    )
    check(magic == MMAGIC and version == 0, "model magic")
    check(count == check_count(alphabet), "model checks")
    check(context_count == contexts(alphabet), "model contexts")
    check(state_count == STATES and total == TOTAL, "model constants")
    check(not any(packet[MSTRUCT.size:MODEL_HEADER]), "model header tail")
    expected = MODEL_HEADER + 2 * context_count * state_count * alphabet
    check(len(packet) == expected, "model length")
    frequencies = np.frombuffer(packet[MODEL_HEADER:], dtype="<u2").copy().reshape(
        context_count, state_count, alphabet
    )
    check(bool(np.all(frequencies >= 1)), "model positive")
    check(bool(np.all(frequencies.sum(axis=2, dtype=np.uint64) == TOTAL)), "model normalized")
    return IndependentModel(int(alphabet), frequencies, packet)


class Decoder:
    FULL = 1 << 32
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREE = 3 << 30

    def __init__(self, packet: bytes) -> None:
        self.packet = packet
        self.position = 0
        self.low = 0
        self.high = self.FULL - 1
        self.code = 0
        for _ in range(32):
            self.code = (self.code << 1) | self.bit()

    def bit(self) -> int:
        if self.position >= 8 * len(self.packet):
            self.position += 1
            return 0
        output = (
            self.packet[self.position // 8] >> (7 - (self.position & 7))
        ) & 1
        self.position += 1
        return output

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
            self.code = ((self.code << 1) | self.bit()) & (self.FULL - 1)
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
        self.output: list[int] = []

    def emit(self, bit: int) -> None:
        self.output.append(bit)
        self.output.extend([1 - bit] * self.pending)
        self.pending = 0

    def symbol(self, selected: int, frequencies: Sequence[int]) -> None:
        row = [int(value) for value in frequencies]
        check(0 <= selected < len(row) and all(value > 0 for value in row), "encode row")
        check(sum(row) == TOTAL, "encode total")
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
        meaningful = len(self.output)
        while len(self.output) % 8:
            self.output.append(0)
        packet = bytearray(len(self.output) // 8)
        for index, bit in enumerate(self.output):
            packet[index // 8] |= bit << (7 - (index & 7))
        return bytes(packet), meaningful


def decode_coefficients(
    model: IndependentModel, packet: bytes, roots_count: int, details_count: int
) -> tuple[np.ndarray, np.ndarray]:
    decoder = Decoder(packet)
    roots = np.empty(roots_count, dtype=np.uint8)
    for index in range(roots_count):
        roots[index] = decoder.symbol(model.frequencies[0, 0, :])
    details = np.empty(details_count, dtype=np.uint8)
    state = 0
    for index in range(details_count):
        if index % BLOCK == 0:
            state = 0
        context = detail_ctx(model.alphabet, index)
        value = decoder.symbol(model.frequencies[context, state, :])
        details[index] = value
        state = advance(model.alphabet, state, index, value)
    return roots, details


def encode_coefficients(
    model: IndependentModel, roots: np.ndarray, details: np.ndarray
) -> tuple[bytes, int]:
    encoder = Encoder()
    for value in roots:
        encoder.symbol(int(value), model.frequencies[0, 0, :])
    state = 0
    for index, value in enumerate(details):
        if index % BLOCK == 0:
            state = 0
        context = detail_ctx(model.alphabet, index)
        encoder.symbol(int(value), model.frequencies[context, state, :])
        state = advance(model.alphabet, state, index, int(value))
    return encoder.finish()


def split_details(flat: np.ndarray, vectors: int, lanes: int) -> list[np.ndarray]:
    counts = pair_counts(lanes)
    levels: list[np.ndarray | None] = [None] * len(counts)
    offset = 0
    for depth in range(len(counts) - 1, -1, -1):
        count = vectors * counts[depth]
        levels[depth] = flat[offset : offset + count].reshape(vectors, counts[depth]).copy()
        offset += count
    check(offset == flat.size, "detail split")
    return [value for value in levels if value is not None]


def join_details(levels: Sequence[np.ndarray]) -> np.ndarray:
    if not levels:
        return np.empty(0, dtype=np.uint8)
    return np.concatenate([np.ascontiguousarray(value).reshape(-1) for value in reversed(levels)]).astype(
        np.uint8, copy=False
    )


def invert_tree(
    roots: np.ndarray,
    levels: Sequence[np.ndarray],
    alphabet: int,
    lanes: int,
    permutation: Sequence[int],
    selectors: Sequence[int],
) -> np.ndarray:
    counts = pair_counts(lanes)
    widths = sizes(lanes)
    offsets = np.cumsum([0] + counts).tolist()
    current = roots[:, None].copy()
    for depth in range(len(counts) - 1, -1, -1):
        pairs = counts[depth]
        previous = np.empty((roots.size, widths[depth]), dtype=np.uint8)
        detail = levels[depth].astype(np.int16)
        coarse = current[:, :pairs].astype(np.int16)
        for pair in range(pairs):
            code = int(selectors[offsets[depth] + pair])
            swap, p, u = (code >> 2) & 1, (code >> 1) & 1, code & 1
            x = np.mod(coarse[:, pair] - u * detail[:, pair], alphabet)
            y = np.mod(detail[:, pair] + p * x, alphabet)
            previous[:, 2 * pair] = y if swap else x
            previous[:, 2 * pair + 1] = x if swap else y
        if widths[depth] & 1:
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
    levels: list[np.ndarray] = []
    offset = 0
    for pairs in pair_counts(leaves.shape[1]):
        next_values = np.empty((leaves.shape[0], pairs + current.shape[1] % 2), dtype=np.uint8)
        details = np.empty((leaves.shape[0], pairs), dtype=np.uint8)
        for pair in range(pairs):
            code = int(selectors[offset + pair])
            swap, p, u = (code >> 2) & 1, (code >> 1) & 1, code & 1
            left = current[:, 2 * pair].astype(np.int16)
            right = current[:, 2 * pair + 1].astype(np.int16)
            x, y = (right, left) if swap else (left, right)
            detail = np.mod(y - p * x, alphabet)
            coarse = np.mod(x + u * detail, alphabet)
            details[:, pair] = detail
            next_values[:, pair] = coarse
        if current.shape[1] & 1:
            next_values[:, -1] = current[:, -1]
        levels.append(details)
        current = next_values
        offset += pairs
    check(current.shape[1] == 1 and offset == len(selectors), "forward tree")
    return current[:, 0].copy(), levels


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
    body_crc: int,
    header_crc: int,
) -> bytes:
    raw = FSTRUCT.pack(
        FMAGIC,
        0,
        expert,
        alphabet,
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
        body_crc,
        header_crc,
    )
    return raw + bytes(FRAME_HEADER - len(raw))


@dataclass(frozen=True)
class IndependentFrame:
    expert: int
    alphabet: int
    lanes: int
    vectors: int
    permutation: list[int]
    selectors: list[int]
    payload: bytes
    meaningful: int
    packet: bytes


def parse_frame(packet: bytes) -> IndependentFrame:
    check(len(packet) >= FRAME_HEADER, "frame header")
    fields = FSTRUCT.unpack(packet[: FSTRUCT.size])
    (
        magic,
        version,
        expert,
        alphabet,
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
        body_crc,
        header_crc,
    ) = fields
    check(magic == FMAGIC and version == 0 and reserved == 0, "frame magic")
    check_count(alphabet)
    check(symbols == lanes * vectors and lanes > 0 and vectors > 0, "frame shape")
    check(pbytes == perm_bytes(lanes), "frame permutation size")
    check(sbytes == (3 * (lanes - 1) + 7) // 8, "frame selector size")
    check(logical == FRAME_HEADER + pbytes + sbytes + payload_bytes, "frame logical")
    check(padded == align(logical) == len(packet), "frame padding")
    check(not any(packet[FSTRUCT.size:FRAME_HEADER]), "frame header tail")
    zero = frame_header(
        expert,
        alphabet,
        lanes,
        vectors,
        pbytes,
        sbytes,
        payload_bytes,
        meaningful,
        logical,
        padded,
        symbols,
        body_crc,
        0,
    )
    check(zlib.crc32(zero) & 0xFFFFFFFF == header_crc, "frame header crc")
    body = packet[FRAME_HEADER:logical]
    check(zlib.crc32(body) & 0xFFFFFFFF == body_crc, "frame body crc")
    check(not any(packet[logical:]), "frame page tail")
    offset = FRAME_HEADER
    permutation_packet = packet[offset : offset + pbytes]
    offset += pbytes
    selector_packet = packet[offset : offset + sbytes]
    offset += sbytes
    payload = packet[offset : offset + payload_bytes]
    check(meaningful <= 8 * len(payload), "frame meaningful bits")
    for bit_index in range(meaningful, 8 * len(payload)):
        check(
            ((payload[bit_index // 8] >> (7 - (bit_index & 7))) & 1) == 0,
            "arithmetic tail",
        )
    return IndependentFrame(
        int(expert),
        int(alphabet),
        int(lanes),
        int(vectors),
        decode_permutation(int(lanes), permutation_packet),
        decode_selectors(selector_packet, int(lanes) - 1),
        payload,
        int(meaningful),
        packet,
    )


def rebuild_frame(frame: IndependentFrame, model: IndependentModel, leaves: np.ndarray) -> bytes:
    roots, levels = forward_tree(leaves, frame.alphabet, frame.permutation, frame.selectors)
    details = join_details(levels)
    payload, meaningful = encode_coefficients(model, roots, details)
    check(meaningful == frame.meaningful, "reencoded meaningful length")
    permutation_packet = encode_permutation(frame.permutation)
    selector_packet = encode_selectors(frame.selectors)
    body = permutation_packet + selector_packet + payload
    logical = FRAME_HEADER + len(body)
    padded = align(logical)
    body_crc = zlib.crc32(body) & 0xFFFFFFFF
    zero = frame_header(
        frame.expert,
        frame.alphabet,
        frame.lanes,
        frame.vectors,
        len(permutation_packet),
        len(selector_packet),
        len(payload),
        meaningful,
        logical,
        padded,
        leaves.size,
        body_crc,
        0,
    )
    header_crc = zlib.crc32(zero) & 0xFFFFFFFF
    header = frame_header(
        frame.expert,
        frame.alphabet,
        frame.lanes,
        frame.vectors,
        len(permutation_packet),
        len(selector_packet),
        len(payload),
        meaningful,
        logical,
        padded,
        leaves.size,
        body_crc,
        header_crc,
    )
    return header + body + bytes(padded - logical)


def global_header(
    alphabet: int,
    experts: int,
    lanes: int,
    vectors: int,
    model_offset: int,
    model_length: int,
    model_pages: int,
    frames_offset: int,
    total_bytes: int,
    symbols: int,
    model_hash: bytes,
    directory: Sequence[tuple[int, int]],
    crc: int,
) -> bytes:
    raw = GSTRUCT.pack(
        GMAGIC,
        0,
        alphabet,
        experts,
        lanes,
        vectors,
        GLOBAL_HEADER,
        model_offset,
        model_length,
        model_pages,
        frames_offset,
        total_bytes,
        symbols,
        model_hash,
        crc,
    )
    entries = b"".join(DIRENT.pack(offset, length) for offset, length in directory)
    check(len(raw) + len(entries) <= GLOBAL_HEADER, "directory fit")
    return raw + entries + bytes(GLOBAL_HEADER - len(raw) - len(entries))


@dataclass(frozen=True)
class IndependentContainer:
    model: IndependentModel
    frames: tuple[IndependentFrame, ...]
    directory: tuple[tuple[int, int], ...]
    packet: bytes
    model_pages: int


def parse_container(packet: bytes) -> IndependentContainer:
    check(len(packet) >= GLOBAL_HEADER, "container header")
    fields = GSTRUCT.unpack(packet[: GSTRUCT.size])
    (
        magic,
        version,
        alphabet,
        experts,
        lanes,
        vectors,
        header_bytes,
        model_offset,
        model_length,
        model_pages,
        frames_offset,
        total_bytes,
        symbols,
        model_hash,
        crc,
    ) = fields
    check(magic == GMAGIC and version == 0, "container magic")
    check_count(alphabet)
    check(experts > 0 and lanes > 0 and vectors > 0, "container geometry")
    check(header_bytes == model_offset == GLOBAL_HEADER, "container model offset")
    check(model_pages == align(model_length), "container model pages")
    check(frames_offset == model_offset + model_pages, "container frame offset")
    check(total_bytes == len(packet) and symbols == experts * lanes * vectors, "container totals")
    directory = tuple(
        DIRENT.unpack(
            packet[
                GSTRUCT.size + index * DIRENT.size :
                GSTRUCT.size + (index + 1) * DIRENT.size
            ]
        )
        for index in range(experts)
    )
    end = GSTRUCT.size + experts * DIRENT.size
    check(not any(packet[end:GLOBAL_HEADER]), "container header tail")
    zero = global_header(
        alphabet,
        experts,
        lanes,
        vectors,
        model_offset,
        model_length,
        model_pages,
        frames_offset,
        total_bytes,
        symbols,
        model_hash,
        directory,
        0,
    )
    check(zlib.crc32(zero) & 0xFFFFFFFF == crc, "container header crc")
    model_packet = packet[model_offset : model_offset + model_length]
    check(hashlib.sha256(model_packet).digest() == model_hash, "container model hash")
    check(not any(packet[model_offset + model_length : frames_offset]), "container model tail")
    model = parse_model(model_packet)
    check(model.alphabet == alphabet, "container model alphabet")
    frame_rows: list[IndependentFrame] = []
    expected = frames_offset
    for index, (offset, length) in enumerate(directory):
        check(offset == expected and offset % PAGE == 0, "frame offset")
        check(length > 0 and length % PAGE == 0 and offset + length <= len(packet), "frame length")
        frame = parse_frame(packet[offset : offset + length])
        check(frame.expert == index, "frame index")
        check(frame.alphabet == alphabet and frame.lanes == lanes and frame.vectors == vectors, "frame shape")
        frame_rows.append(frame)
        expected = offset + length
    check(expected == len(packet), "frame coverage")
    return IndependentContainer(model, tuple(frame_rows), directory, packet, int(model_pages))


def verify_decode_reencode(
    packet: bytes, expected_leaf_sha256: Sequence[str] | None = None
) -> tuple[dict[str, object], list[np.ndarray], bytes]:
    container = parse_container(packet)
    decoded: list[np.ndarray] = []
    rebuilt_frames: list[bytes] = []
    for frame in container.frames:
        roots, details = decode_coefficients(
            container.model,
            frame.payload,
            frame.vectors,
            frame.vectors * (frame.lanes - 1),
        )
        levels = split_details(details, frame.vectors, frame.lanes)
        leaves = invert_tree(
            roots,
            levels,
            frame.alphabet,
            frame.lanes,
            frame.permutation,
            frame.selectors,
        )
        decoded.append(leaves)
        rebuilt = rebuild_frame(frame, container.model, leaves)
        check(rebuilt == frame.packet, "independent frame reencode mismatch")
        rebuilt_frames.append(rebuilt)
    if expected_leaf_sha256 is not None:
        check(len(expected_leaf_sha256) == len(decoded), "expected digest count")
        for expected, leaves in zip(expected_leaf_sha256, decoded, strict=True):
            observed = hashlib.sha256(np.ascontiguousarray(leaves).tobytes(order="C")).hexdigest()
            check(observed == expected, "independent leaf digest")
    frames_offset = GLOBAL_HEADER + container.model_pages
    directory: list[tuple[int, int]] = []
    offset = frames_offset
    for frame in rebuilt_frames:
        directory.append((offset, len(frame)))
        offset += len(frame)
    model_packet = container.model.packet
    model_hash = hashlib.sha256(model_packet).digest()
    first = container.frames[0]
    zero = global_header(
        first.alphabet,
        len(rebuilt_frames),
        first.lanes,
        first.vectors,
        GLOBAL_HEADER,
        len(model_packet),
        container.model_pages,
        frames_offset,
        offset,
        len(rebuilt_frames) * first.lanes * first.vectors,
        model_hash,
        directory,
        0,
    )
    crc = zlib.crc32(zero) & 0xFFFFFFFF
    header = global_header(
        first.alphabet,
        len(rebuilt_frames),
        first.lanes,
        first.vectors,
        GLOBAL_HEADER,
        len(model_packet),
        container.model_pages,
        frames_offset,
        offset,
        len(rebuilt_frames) * first.lanes * first.vectors,
        model_hash,
        directory,
        crc,
    )
    rebuilt_container = (
        header
        + model_packet
        + bytes(container.model_pages - len(model_packet))
        + b"".join(rebuilt_frames)
    )
    check(rebuilt_container == packet, "independent container reencode mismatch")
    fair = len(packet) / len(rebuilt_frames)
    global_bytes = GLOBAL_HEADER + container.model_pages
    amplifications = [(global_bytes + len(frame)) / fair for frame in rebuilt_frames]
    receipt = {
        "status": "PASS_INDEPENDENT_DECODE_REENCODE",
        "container_sha256": hashlib.sha256(packet).hexdigest(),
        "container_bytes": len(packet),
        "experts": len(rebuilt_frames),
        "alphabet": first.alphabet,
        "lanes": first.lanes,
        "vectors": first.vectors,
        "leaf_sha256": [
            hashlib.sha256(np.ascontiguousarray(leaves).tobytes(order="C")).hexdigest()
            for leaves in decoded
        ],
        "max_cold_amplification": max(amplifications),
        "cold_below_two": max(amplifications) < 2.0,
        "source_gain_claim": False,
    }
    return receipt, decoded, rebuilt_container
