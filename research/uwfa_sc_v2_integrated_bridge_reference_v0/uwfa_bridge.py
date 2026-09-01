"""Independent, source-only UWFA-SC v2 integrated-container reference.

This module deliberately has no model/checkpoint loader.  It implements the
finite byte ABI, a serialized-model-only causal arithmetic adapter, exact
physical accounting, and an owner-aware cold-page ledger.  A real STRATA
decoder supplies the semantic metadata parser and reconstruction callback;
the companion synthetic fixture exercises the same interfaces without any
Qwen or baseline payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import zlib
from typing import Callable, Iterable, Sequence


PAGE_SIZE = 4096
FRAME_ALIGNMENT = 64
GLOBAL_HEADER_SIZE = 4096
DIRECTORY_RECORD_SIZE = 256
BLOCK_COUNT = 15
ARITH_TOTAL = 65536

CONTAINER_MAGIC = b"UWFASC2\x00"
MODEL_MAGIC = b"UWFAM2\x00\x00"
DIRECTORY_MAGIC = b"UWFADIR2"
FRAME_MAGIC = b"UWFABLK\x00"

CONTAINER_MAJOR = 2
CONTAINER_MINOR = 0
MODEL_MAJOR = 1
MODEL_MINOR = 0
MODEL_HEADER_SIZE = 64
MODEL_ROW_SIZE = 8
FRAME_HEADER_SIZE = 64

HEADER_CRC_OFFSET = 16
ROOT_HASH_OFFSET = 400
ROOT_HASH_END = 432
HEADER_RESERVED_OFFSET = 560


class FormatError(ValueError):
    """The byte stream is not the unique canonical representation."""


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("invalid alignment request")
    return (value + alignment - 1) & -alignment


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def require_zero(data: bytes, what: str) -> None:
    if any(data):
        raise FormatError(f"nonzero reserved/padding bytes in {what}")


def checked_range(offset: int, length: int, total: int, what: str) -> tuple[int, int]:
    if offset < 0 or length < 0:
        raise FormatError(f"negative {what} range")
    end = offset + length
    if end > 0xFFFFFFFFFFFFFFFF or end > total:
        raise FormatError(f"overflow/out-of-bounds {what} range")
    return offset, end


def pack_bits(bits: Sequence[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        if bit not in (0, 1):
            raise ValueError("decision is not binary")
        out[index >> 3] |= bit << (7 - (index & 7))
    return bytes(out)


def decision_hash(bits: Sequence[int]) -> bytes:
    return sha256(struct.pack("<Q", len(bits)) + pack_bits(bits))


@dataclass(frozen=True)
class UWFAModel:
    """Dense, lexicographically serialized nonnegative unifilar model."""

    total: int
    state_count: int
    reset_length: int
    level_count: int
    prior_bin_count: int
    topology_id: int
    frequencies: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.total != ARITH_TOTAL:
            raise ValueError("v0 reference fixes arithmetic total at 65536")
        if not (1 <= self.state_count <= 65535):
            raise ValueError("state_count out of range")
        if not (1 <= self.reset_length <= 65535):
            raise ValueError("reset_length out of range")
        if not (1 <= self.level_count <= 255):
            raise ValueError("level_count out of range")
        if not (1 <= self.prior_bin_count <= 255):
            raise ValueError("prior_bin_count out of range")
        if self.topology_id != 1:
            raise ValueError("unknown frozen transition topology")
        expected = (
            self.level_count
            * self.prior_bin_count
            * self.reset_length
            * self.state_count
        )
        if len(self.frequencies) != expected:
            raise ValueError("frequency table has wrong size")
        if any(not 1 <= value < self.total for value in self.frequencies):
            raise ValueError("frequency must be strictly internal")

    def index(self, level: int, prior_bin: int, position: int, state: int) -> int:
        if not 0 <= level < self.level_count:
            raise ValueError("level out of range")
        if not 0 <= prior_bin < self.prior_bin_count:
            raise ValueError("prior bin out of range")
        if not 0 <= position < self.reset_length:
            raise ValueError("position in reset out of range")
        if not 0 <= state < self.state_count:
            raise ValueError("state out of range")
        return (
            (((level * self.prior_bin_count + prior_bin) * self.reset_length + position)
             * self.state_count)
            + state
        )

    def frequency(self, level: int, prior_bin: int, position: int, state: int) -> int:
        return self.frequencies[self.index(level, prior_bin, position, state)]

    def prior_bin(self, original_freq1: int) -> int:
        if not 1 <= original_freq1 < self.total:
            raise ValueError("original SC frequency is not strictly internal")
        result = original_freq1 * self.prior_bin_count // self.total
        return min(result, self.prior_bin_count - 1)

    def transition(
        self,
        state: int,
        bit: int,
        level: int,
        prior_bin: int,
        position: int,
    ) -> int:
        if bit not in (0, 1):
            raise ValueError("transition decision is not binary")
        # Topology 1 is frozen by the universal decoder.  It is intentionally
        # not serialized per model row, avoiding an uncharged transition law.
        return (
            state * 5
            + bit
            + level * 3
            + prior_bin * 7
            + position * 11
            + 1
        ) % self.state_count


def serialize_model(model: UWFAModel) -> bytes:
    row_count = len(model.frequencies)
    out = bytearray(MODEL_HEADER_SIZE + row_count * MODEL_ROW_SIZE)
    out[0:8] = MODEL_MAGIC
    struct.pack_into("<HHI", out, 8, MODEL_MAJOR, MODEL_MINOR, MODEL_HEADER_SIZE)
    struct.pack_into("<I", out, 16, model.total)
    struct.pack_into(
        "<HHHHHHQ",
        out,
        20,
        model.state_count,
        model.reset_length,
        model.level_count,
        model.prior_bin_count,
        model.topology_id,
        MODEL_ROW_SIZE,
        row_count,
    )
    # CRC lives at 40; bytes 44..63 are canonical zero.
    cursor = MODEL_HEADER_SIZE
    frequency_index = 0
    for level in range(model.level_count):
        for prior_bin in range(model.prior_bin_count):
            for position in range(model.reset_length):
                for state in range(model.state_count):
                    struct.pack_into(
                        "<BBHHH",
                        out,
                        cursor,
                        level,
                        prior_bin,
                        position,
                        state,
                        model.frequencies[frequency_index],
                    )
                    frequency_index += 1
                    cursor += MODEL_ROW_SIZE
    struct.pack_into("<I", out, 40, 0)
    struct.pack_into("<I", out, 40, crc32(bytes(out)))
    return bytes(out)


def deserialize_model(data: bytes) -> UWFAModel:
    if len(data) < MODEL_HEADER_SIZE:
        raise FormatError("truncated model header")
    if data[0:8] != MODEL_MAGIC:
        raise FormatError("bad model magic")
    major, minor, header_size = struct.unpack_from("<HHI", data, 8)
    if (major, minor, header_size) != (MODEL_MAJOR, MODEL_MINOR, MODEL_HEADER_SIZE):
        raise FormatError("unsupported/noncanonical model version")
    total = struct.unpack_from("<I", data, 16)[0]
    (
        state_count,
        reset_length,
        level_count,
        prior_bin_count,
        topology_id,
        row_size,
        row_count,
    ) = struct.unpack_from("<HHHHHHQ", data, 20)
    if row_size != MODEL_ROW_SIZE:
        raise FormatError("noncanonical model row size")
    expected_rows = level_count * prior_bin_count * reset_length * state_count
    if row_count != expected_rows:
        raise FormatError("model is not a complete dense canonical table")
    expected_length = MODEL_HEADER_SIZE + row_count * MODEL_ROW_SIZE
    if len(data) != expected_length:
        raise FormatError("model length does not match row count")
    require_zero(data[44:MODEL_HEADER_SIZE], "model header")
    expected_crc = struct.unpack_from("<I", data, 40)[0]
    crc_image = bytearray(data)
    struct.pack_into("<I", crc_image, 40, 0)
    if crc32(bytes(crc_image)) != expected_crc:
        raise FormatError("model CRC mismatch")

    frequencies: list[int] = []
    cursor = MODEL_HEADER_SIZE
    for level in range(level_count):
        for prior_bin in range(prior_bin_count):
            for position in range(reset_length):
                for state in range(state_count):
                    row = struct.unpack_from("<BBHHH", data, cursor)
                    if row[0:4] != (level, prior_bin, position, state):
                        raise FormatError("model rows are missing, duplicate, or out of order")
                    frequency = row[4]
                    if not 1 <= frequency < total:
                        raise FormatError("noncanonical model frequency")
                    frequencies.append(frequency)
                    cursor += MODEL_ROW_SIZE
    try:
        return UWFAModel(
            total=total,
            state_count=state_count,
            reset_length=reset_length,
            level_count=level_count,
            prior_bin_count=prior_bin_count,
            topology_id=topology_id,
            frequencies=tuple(frequencies),
        )
    except ValueError as exc:
        raise FormatError(str(exc)) from exc


class _BitWriter:
    def __init__(self) -> None:
        self._bytes = bytearray()
        self._accumulator = 0
        self._used = 0
        self.bit_length = 0

    def write(self, bit: int) -> None:
        if bit not in (0, 1):
            raise ValueError("bit writer received non-bit")
        self._accumulator = (self._accumulator << 1) | bit
        self._used += 1
        self.bit_length += 1
        if self._used == 8:
            self._bytes.append(self._accumulator)
            self._accumulator = 0
            self._used = 0

    def finish(self) -> tuple[bytes, int]:
        if self._used:
            self._bytes.append(self._accumulator << (8 - self._used))
        return bytes(self._bytes), self.bit_length


class _BitReader:
    def __init__(self, data: bytes, bit_length: int) -> None:
        if bit_length < 0 or bit_length > len(data) * 8:
            raise FormatError("logical arithmetic length exceeds payload")
        if len(data) != (bit_length + 7) // 8:
            raise FormatError("arithmetic payload has noncanonical zero-byte extension")
        if bit_length and bit_length & 7:
            unused_mask = (1 << (8 - (bit_length & 7))) - 1
            if data[(bit_length - 1) >> 3] & unused_mask:
                raise FormatError("nonzero unused arithmetic tail bits")
        self.data = data
        self.bit_length = bit_length
        self.position = 0

    def read(self) -> int:
        # Arithmetic decoding conventionally extends a finite code by zeros.
        if self.position >= self.bit_length:
            self.position += 1
            return 0
        result = (self.data[self.position >> 3] >> (7 - (self.position & 7))) & 1
        self.position += 1
        return result


class BinaryArithmeticEncoder:
    """Canonical 32-bit binary arithmetic encoder."""

    _FULL = 1 << 32
    _HALF = 1 << 31
    _QUARTER = 1 << 30
    _THREE_QUARTERS = 3 << 30

    def __init__(self, total: int = ARITH_TOTAL) -> None:
        self.total = total
        self.low = 0
        self.high = self._FULL - 1
        self.pending = 0
        self.writer = _BitWriter()
        self.finished = False

    def _emit_with_pending(self, bit: int) -> None:
        self.writer.write(bit)
        inverse = 1 - bit
        for _ in range(self.pending):
            self.writer.write(inverse)
        self.pending = 0

    def encode(self, bit: int, freq1: int) -> None:
        if self.finished:
            raise RuntimeError("arithmetic encoder already finalized")
        if bit not in (0, 1) or not 1 <= freq1 < self.total:
            raise ValueError("invalid arithmetic event")
        frequency0 = self.total - freq1
        width = self.high - self.low + 1
        cut = self.low + width * frequency0 // self.total
        if bit == 0:
            self.high = cut - 1
        else:
            self.low = cut
        while True:
            if self.high < self._HALF:
                self._emit_with_pending(0)
            elif self.low >= self._HALF:
                self._emit_with_pending(1)
                self.low -= self._HALF
                self.high -= self._HALF
            elif self.low >= self._QUARTER and self.high < self._THREE_QUARTERS:
                self.pending += 1
                self.low -= self._QUARTER
                self.high -= self._QUARTER
            else:
                break
            self.low = (self.low << 1) & (self._FULL - 1)
            self.high = ((self.high << 1) | 1) & (self._FULL - 1)

    def finish(self) -> tuple[bytes, int]:
        if self.finished:
            raise RuntimeError("arithmetic encoder finalized twice")
        self.finished = True
        self.pending += 1
        if self.low < self._QUARTER:
            self._emit_with_pending(0)
        else:
            self._emit_with_pending(1)
        return self.writer.finish()


class BinaryArithmeticDecoder:
    """Exact inverse of :class:`BinaryArithmeticEncoder`."""

    _FULL = 1 << 32
    _HALF = 1 << 31
    _QUARTER = 1 << 30
    _THREE_QUARTERS = 3 << 30

    def __init__(self, data: bytes, bit_length: int, total: int = ARITH_TOTAL) -> None:
        self.total = total
        self.reader = _BitReader(data, bit_length)
        self.low = 0
        self.high = self._FULL - 1
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) | self.reader.read()) & (self._FULL - 1)

    def decode(self, freq1: int) -> int:
        if not 1 <= freq1 < self.total:
            raise ValueError("invalid arithmetic frequency")
        frequency0 = self.total - freq1
        width = self.high - self.low + 1
        cut = self.low + width * frequency0 // self.total
        if self.code < cut:
            bit = 0
            self.high = cut - 1
        else:
            bit = 1
            self.low = cut
        while True:
            if self.high < self._HALF:
                pass
            elif self.low >= self._HALF:
                self.low -= self._HALF
                self.high -= self._HALF
                self.code -= self._HALF
            elif self.low >= self._QUARTER and self.high < self._THREE_QUARTERS:
                self.low -= self._QUARTER
                self.high -= self._QUARTER
                self.code -= self._QUARTER
            else:
                break
            self.low = (self.low << 1) & (self._FULL - 1)
            self.high = ((self.high << 1) | 1) & (self._FULL - 1)
            self.code = ((self.code << 1) | self.reader.read()) & (self._FULL - 1)
        return bit


class UWFAEncoderAdapter:
    """Arithmetic-like encoder using only model and regenerated SC frequency."""

    def __init__(self, model: UWFAModel, arithmetic: BinaryArithmeticEncoder) -> None:
        self.model = model
        self.arithmetic = arithmetic
        self.level: int | None = None
        self.position = 0
        self.state = 0

    def set_level(self, polar_level: int) -> None:
        if not 0 <= polar_level < self.model.level_count:
            raise ValueError("polar level out of model range")
        self.level = polar_level

    def encode(self, bit: int, original_freq1: int) -> None:
        if self.level is None:
            raise RuntimeError("set_level must precede arithmetic operation")
        position_in_reset = self.position % self.model.reset_length
        if position_in_reset == 0:
            self.state = 0
        prior_bin = self.model.prior_bin(original_freq1)
        frequency = self.model.frequency(
            self.level, prior_bin, position_in_reset, self.state
        )
        self.arithmetic.encode(bit, frequency)
        self.state = self.model.transition(
            self.state, bit, self.level, prior_bin, position_in_reset
        )
        self.position += 1


class UWFADecoderAdapter:
    """Required ``decode(original_freq1)`` UWFA-SC replacement adapter."""

    def __init__(self, model: UWFAModel, arithmetic: BinaryArithmeticDecoder) -> None:
        self.model = model
        self.arithmetic = arithmetic
        self.level: int | None = None
        self.position = 0
        self.state = 0

    def set_level(self, polar_level: int) -> None:
        if not 0 <= polar_level < self.model.level_count:
            raise ValueError("polar level out of model range")
        self.level = polar_level

    def decode(self, original_freq1: int) -> int:
        if self.level is None:
            raise RuntimeError("set_level must precede arithmetic operation")
        position_in_reset = self.position % self.model.reset_length
        if position_in_reset == 0:
            self.state = 0
        prior_bin = self.model.prior_bin(original_freq1)
        frequency = self.model.frequency(
            self.level, prior_bin, position_in_reset, self.state
        )
        bit = self.arithmetic.decode(frequency)
        self.state = self.model.transition(
            self.state, bit, self.level, prior_bin, position_in_reset
        )
        self.position += 1
        return bit


@dataclass(frozen=True)
class Frame:
    ordinal: int
    decision_count: int
    encoded_bytes: bytes
    logical_bits: int
    physical_bytes: bytes


def build_frame(
    ordinal: int,
    decision_count: int,
    encoded_bytes: bytes,
    logical_bits: int,
) -> Frame:
    if not 0 <= ordinal < BLOCK_COUNT or decision_count < 0:
        raise ValueError("invalid block frame identity")
    if logical_bits < 0 or logical_bits > len(encoded_bytes) * 8:
        raise ValueError("invalid logical arithmetic length")
    # Reuse the strict reader to validate canonical zero tail bits/bytes.
    _BitReader(encoded_bytes, logical_bits)
    header = bytearray(FRAME_HEADER_SIZE)
    header[0:8] = FRAME_MAGIC
    struct.pack_into("<HHI", header, 8, 1, ordinal, FRAME_HEADER_SIZE)
    struct.pack_into("<QQQ", header, 16, decision_count, len(encoded_bytes), logical_bits)
    struct.pack_into("<I", header, 40, crc32(encoded_bytes))
    struct.pack_into("<I", header, 44, 0)
    struct.pack_into("<I", header, 44, crc32(bytes(header)))
    meaningful = bytes(header) + encoded_bytes
    physical = meaningful + bytes(align_up(len(meaningful), FRAME_ALIGNMENT) - len(meaningful))
    return Frame(ordinal, decision_count, encoded_bytes, logical_bits, physical)


def parse_frame(data: bytes, expected_ordinal: int | None = None) -> Frame:
    if len(data) < FRAME_HEADER_SIZE or len(data) % FRAME_ALIGNMENT:
        raise FormatError("frame length is not canonical 64-byte multiple")
    if data[0:8] != FRAME_MAGIC:
        raise FormatError("bad frame magic")
    version, ordinal, header_size = struct.unpack_from("<HHI", data, 8)
    if version != 1 or header_size != FRAME_HEADER_SIZE:
        raise FormatError("unsupported/noncanonical frame version")
    if expected_ordinal is not None and ordinal != expected_ordinal:
        raise FormatError("frame ordinal does not match directory")
    decision_count, encoded_length, logical_bits = struct.unpack_from("<QQQ", data, 16)
    stream_crc = struct.unpack_from("<I", data, 40)[0]
    header_crc = struct.unpack_from("<I", data, 44)[0]
    require_zero(data[48:FRAME_HEADER_SIZE], "frame header")
    header_image = bytearray(data[:FRAME_HEADER_SIZE])
    struct.pack_into("<I", header_image, 44, 0)
    if crc32(bytes(header_image)) != header_crc:
        raise FormatError("frame header CRC mismatch")
    expected_physical = align_up(FRAME_HEADER_SIZE + encoded_length, FRAME_ALIGNMENT)
    if len(data) != expected_physical:
        raise FormatError("frame has noncanonical physical length")
    stream_end = FRAME_HEADER_SIZE + encoded_length
    stream = data[FRAME_HEADER_SIZE:stream_end]
    require_zero(data[stream_end:], "frame tail")
    if crc32(stream) != stream_crc:
        raise FormatError("frame arithmetic CRC mismatch")
    _BitReader(stream, logical_bits)
    return Frame(ordinal, decision_count, stream, logical_bits, data)


@dataclass(frozen=True)
class DirectoryRecord:
    ordinal: int
    log2n: int
    role: int
    owner_mask: int
    profile_id: int
    scale_bits: int
    payload_offset: int
    physical_length: int
    logical_bits: int
    decision_count: int
    encoded_length: int
    payload_hash: bytes
    decisions_hash: bytes


def serialize_directory_record(record: DirectoryRecord) -> bytes:
    if len(record.payload_hash) != 32 or len(record.decisions_hash) != 32:
        raise ValueError("directory hashes must be SHA-256")
    if not (0 <= record.ordinal < BLOCK_COUNT and 1 <= record.log2n <= 63):
        raise ValueError("invalid directory block geometry")
    if not (0 <= record.role <= 255 and 0 < record.owner_mask < (1 << 64)):
        raise ValueError("invalid directory ownership")
    if not (0 <= record.profile_id <= 255 and 0 <= record.scale_bits <= 65535):
        raise ValueError("invalid directory profile")
    out = bytearray(DIRECTORY_RECORD_SIZE)
    out[0:8] = DIRECTORY_MAGIC
    struct.pack_into("<HBBQ", out, 8, record.ordinal, record.log2n, record.role, record.owner_mask)
    struct.pack_into("<B", out, 20, record.profile_id)
    struct.pack_into("<H", out, 22, record.scale_bits)
    struct.pack_into(
        "<QQQQQ",
        out,
        24,
        record.payload_offset,
        record.physical_length,
        record.logical_bits,
        record.decision_count,
        record.encoded_length,
    )
    out[64:96] = record.payload_hash
    out[96:128] = record.decisions_hash
    return bytes(out)


def parse_directory_record(data: bytes, expected_ordinal: int) -> DirectoryRecord:
    if len(data) != DIRECTORY_RECORD_SIZE or data[0:8] != DIRECTORY_MAGIC:
        raise FormatError("bad directory record")
    ordinal, log2n, role, owner_mask = struct.unpack_from("<HBBQ", data, 8)
    if ordinal != expected_ordinal:
        raise FormatError("directory is not in canonical block order")
    profile_id = data[20]
    if data[21] != 0:
        raise FormatError("nonzero directory reserved byte")
    scale_bits = struct.unpack_from("<H", data, 22)[0]
    payload_offset, physical_length, logical_bits, decision_count, encoded_length = (
        struct.unpack_from("<QQQQQ", data, 24)
    )
    require_zero(data[128:], "directory record")
    if not 1 <= log2n <= 63 or owner_mask == 0:
        raise FormatError("invalid directory geometry/ownership")
    if physical_length % FRAME_ALIGNMENT:
        raise FormatError("directory frame length is not aligned")
    return DirectoryRecord(
        ordinal=ordinal,
        log2n=log2n,
        role=role,
        owner_mask=owner_mask,
        profile_id=profile_id,
        scale_bits=scale_bits,
        payload_offset=payload_offset,
        physical_length=physical_length,
        logical_bits=logical_bits,
        decision_count=decision_count,
        encoded_length=encoded_length,
        payload_hash=data[64:96],
        decisions_hash=data[96:128],
    )


@dataclass(frozen=True)
class HeaderBindings:
    baseline_container_bytes: int
    baseline_relative_mse: float
    energy_convention: int
    baseline_container_hash: bytes
    baseline_plan_lock_hash: bytes
    baseline_audit_hash: bytes
    universal_decoder_hash: bytes
    source_manifest_hash: bytes
    audit_bootstrap_hash: bytes
    decoded_reconstruction_hash: bytes

    def __post_init__(self) -> None:
        hashes = (
            self.baseline_container_hash,
            self.baseline_plan_lock_hash,
            self.baseline_audit_hash,
            self.universal_decoder_hash,
            self.source_manifest_hash,
            self.audit_bootstrap_hash,
            self.decoded_reconstruction_hash,
        )
        if any(len(value) != 32 for value in hashes):
            raise ValueError("all binding hashes must be SHA-256")
        if any(value == bytes(32) for value in hashes):
            raise ValueError("binding hashes must not be the all-zero sentinel")
        if self.baseline_container_bytes < 0 or not math.isfinite(self.baseline_relative_mse):
            raise ValueError("invalid baseline evidence binding")
        if self.baseline_relative_mse < 0 or self.energy_convention != 1:
            raise ValueError("unsupported score convention")


@dataclass(frozen=True)
class BlockSpec:
    log2n: int
    role: int
    owner_mask: int
    profile_id: int
    scale_bits: int
    frame: Frame
    decisions_hash: bytes


@dataclass(frozen=True)
class ParsedHeader:
    source_weights: int
    expert_count: int
    block_count: int
    role_count: int
    state_count: int
    reset_length: int
    prior_bin_count: int
    level_count: int
    topology_id: int
    container_bytes: int
    metadata_offset: int
    metadata_actual: int
    metadata_region: int
    model_offset: int
    model_actual: int
    model_region: int
    directory_offset: int
    directory_actual: int
    directory_region: int
    frames_offset: int
    frames_end: int
    final_padding_offset: int
    final_padding_length: int
    bindings: HeaderBindings
    metadata_hash: bytes
    model_hash: bytes
    directory_hash: bytes
    root_hash: bytes


@dataclass(frozen=True)
class ParsedContainer:
    raw: bytes
    header: ParsedHeader
    metadata: bytes
    model_bytes: bytes
    model: UWFAModel
    records: tuple[DirectoryRecord, ...]
    frames: tuple[Frame, ...]


def _write_header(
    *,
    source_weights: int,
    expert_count: int,
    role_count: int,
    model: UWFAModel,
    container_bytes: int,
    metadata_offset: int,
    metadata_actual: int,
    metadata_region: int,
    model_offset: int,
    model_actual: int,
    model_region: int,
    directory_offset: int,
    directory_actual: int,
    directory_region: int,
    frames_offset: int,
    frames_end: int,
    final_padding_offset: int,
    final_padding_length: int,
    bindings: HeaderBindings,
    metadata_hash: bytes,
    model_hash: bytes,
    directory_hash: bytes,
    root_hash: bytes,
) -> bytearray:
    header = bytearray(GLOBAL_HEADER_SIZE)
    header[0:8] = CONTAINER_MAGIC
    struct.pack_into("<HHI", header, 8, CONTAINER_MAJOR, CONTAINER_MINOR, GLOBAL_HEADER_SIZE)
    struct.pack_into("<I", header, HEADER_CRC_OFFSET, 0)
    struct.pack_into("<III", header, 20, 0, PAGE_SIZE, FRAME_ALIGNMENT)
    struct.pack_into("<Q", header, 32, source_weights)
    struct.pack_into(
        "<IIIIIIII",
        header,
        40,
        expert_count,
        BLOCK_COUNT,
        role_count,
        model.state_count,
        model.reset_length,
        model.prior_bin_count,
        model.level_count,
        model.topology_id,
    )
    struct.pack_into(
        "<" + "Q" * 14,
        header,
        72,
        container_bytes,
        metadata_offset,
        metadata_actual,
        metadata_region,
        model_offset,
        model_actual,
        model_region,
        directory_offset,
        directory_actual,
        directory_region,
        frames_offset,
        frames_end,
        final_padding_offset,
        final_padding_length,
    )
    struct.pack_into("<QdI", header, 184, bindings.baseline_container_bytes, bindings.baseline_relative_mse, bindings.energy_convention)
    header[208:240] = bindings.baseline_container_hash
    header[240:272] = bindings.baseline_plan_lock_hash
    header[272:304] = bindings.baseline_audit_hash
    header[304:336] = bindings.universal_decoder_hash
    header[336:368] = bindings.source_manifest_hash
    header[368:400] = bindings.audit_bootstrap_hash
    header[400:432] = root_hash
    header[432:464] = bindings.decoded_reconstruction_hash
    header[464:496] = metadata_hash
    header[496:528] = model_hash
    header[528:560] = directory_hash
    return header


def _normalized_header_for_root(header: bytes) -> bytes:
    image = bytearray(header)
    struct.pack_into("<I", image, HEADER_CRC_OFFSET, 0)
    image[ROOT_HASH_OFFSET:ROOT_HASH_END] = bytes(ROOT_HASH_END - ROOT_HASH_OFFSET)
    return bytes(image)


def _root_digest(
    header: bytes,
    metadata: bytes,
    model_bytes: bytes,
    directory_bytes: bytes,
    frames: Sequence[Frame],
) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(_normalized_header_for_root(header))
    hasher.update(metadata)
    hasher.update(model_bytes)
    hasher.update(directory_bytes)
    for frame in frames:
        hasher.update(frame.physical_bytes[: FRAME_HEADER_SIZE + len(frame.encoded_bytes)])
    return hasher.digest()


def build_container(
    *,
    metadata: bytes,
    model: UWFAModel,
    blocks: Sequence[BlockSpec],
    source_weights: int,
    expert_count: int,
    role_count: int,
    bindings: HeaderBindings,
    enforce_rate_cap: bool = True,
) -> bytes:
    if len(blocks) != BLOCK_COUNT or source_weights <= 0:
        raise ValueError("integrated format requires fifteen blocks and positive weight count")
    if not (1 <= expert_count <= 64 and 1 <= role_count <= 255):
        raise ValueError("invalid expert/role count")
    model_bytes = serialize_model(model)
    _validate_owner_topology(tuple(block.owner_mask for block in blocks), expert_count)

    metadata_offset = GLOBAL_HEADER_SIZE
    metadata_region = align_up(len(metadata), PAGE_SIZE)
    model_offset = metadata_offset + metadata_region
    model_region = align_up(len(model_bytes), PAGE_SIZE)
    directory_offset = model_offset + model_region
    directory_actual = BLOCK_COUNT * DIRECTORY_RECORD_SIZE
    directory_region = align_up(directory_actual, PAGE_SIZE)
    frames_offset = directory_offset + directory_region

    cursor = frames_offset
    records: list[DirectoryRecord] = []
    frames: list[Frame] = []
    for ordinal, block in enumerate(blocks):
        if block.frame.ordinal != ordinal:
            raise ValueError("block/frame order is not canonical")
        if block.owner_mask >> expert_count:
            raise ValueError("block owner outside expert count")
        if not 0 <= block.role < role_count:
            raise ValueError("block role outside role count")
        if len(block.decisions_hash) != 32:
            raise ValueError("decision hash is not SHA-256")
        frame = parse_frame(block.frame.physical_bytes, ordinal)
        if frame.decision_count != block.frame.decision_count:
            raise ValueError("frame decision count mismatch")
        record = DirectoryRecord(
            ordinal=ordinal,
            log2n=block.log2n,
            role=block.role,
            owner_mask=block.owner_mask,
            profile_id=block.profile_id,
            scale_bits=block.scale_bits,
            payload_offset=cursor,
            physical_length=len(frame.physical_bytes),
            logical_bits=frame.logical_bits,
            decision_count=frame.decision_count,
            encoded_length=len(frame.encoded_bytes),
            payload_hash=sha256(frame.physical_bytes),
            decisions_hash=block.decisions_hash,
        )
        records.append(record)
        frames.append(frame)
        cursor += len(frame.physical_bytes)
    frames_end = cursor
    directory_bytes = b"".join(serialize_directory_record(record) for record in records)

    minimum_floor_bytes = (215 * source_weights + 799) // 800  # ceil(2.15*w/8)
    container_length = align_up(max(frames_end, minimum_floor_bytes), PAGE_SIZE)
    if enforce_rate_cap and Fraction(8 * container_length, source_weights) > Fraction(5, 2):
        raise ValueError("literal container exceeds 2.5 physical bpw")
    final_padding_length = container_length - frames_end

    metadata_hash = sha256(metadata)
    model_hash = sha256(model_bytes)
    directory_hash = sha256(directory_bytes)
    header = _write_header(
        source_weights=source_weights,
        expert_count=expert_count,
        role_count=role_count,
        model=model,
        container_bytes=container_length,
        metadata_offset=metadata_offset,
        metadata_actual=len(metadata),
        metadata_region=metadata_region,
        model_offset=model_offset,
        model_actual=len(model_bytes),
        model_region=model_region,
        directory_offset=directory_offset,
        directory_actual=directory_actual,
        directory_region=directory_region,
        frames_offset=frames_offset,
        frames_end=frames_end,
        final_padding_offset=frames_end,
        final_padding_length=final_padding_length,
        bindings=bindings,
        metadata_hash=metadata_hash,
        model_hash=model_hash,
        directory_hash=directory_hash,
        root_hash=bytes(32),
    )
    root_hash = _root_digest(header, metadata, model_bytes, directory_bytes, frames)
    header[ROOT_HASH_OFFSET:ROOT_HASH_END] = root_hash
    struct.pack_into("<I", header, HEADER_CRC_OFFSET, 0)
    struct.pack_into("<I", header, HEADER_CRC_OFFSET, crc32(bytes(header)))

    output = bytearray(container_length)
    output[:GLOBAL_HEADER_SIZE] = header
    output[metadata_offset : metadata_offset + len(metadata)] = metadata
    output[model_offset : model_offset + len(model_bytes)] = model_bytes
    output[directory_offset : directory_offset + len(directory_bytes)] = directory_bytes
    for record, frame in zip(records, frames):
        output[record.payload_offset : record.payload_offset + record.physical_length] = frame.physical_bytes
    return bytes(output)


def _parse_header(raw: bytes) -> ParsedHeader:
    if len(raw) < GLOBAL_HEADER_SIZE or raw[0:8] != CONTAINER_MAGIC:
        raise FormatError("truncated container or bad magic")
    major, minor, header_size = struct.unpack_from("<HHI", raw, 8)
    if (major, minor, header_size) != (CONTAINER_MAJOR, CONTAINER_MINOR, GLOBAL_HEADER_SIZE):
        raise FormatError("unsupported/noncanonical container version")
    expected_crc = struct.unpack_from("<I", raw, HEADER_CRC_OFFSET)[0]
    header_image = bytearray(raw[:GLOBAL_HEADER_SIZE])
    struct.pack_into("<I", header_image, HEADER_CRC_OFFSET, 0)
    if crc32(bytes(header_image)) != expected_crc:
        raise FormatError("global header CRC mismatch")
    flags, page_size, alignment = struct.unpack_from("<III", raw, 20)
    if (flags, page_size, alignment) != (0, PAGE_SIZE, FRAME_ALIGNMENT):
        raise FormatError("noncanonical global layout constants")
    source_weights = struct.unpack_from("<Q", raw, 32)[0]
    (
        expert_count,
        block_count,
        role_count,
        state_count,
        reset_length,
        prior_bin_count,
        level_count,
        topology_id,
    ) = struct.unpack_from("<IIIIIIII", raw, 40)
    if block_count != BLOCK_COUNT:
        raise FormatError("noncanonical block count/topology descriptor")
    ranges = struct.unpack_from("<" + "Q" * 14, raw, 72)
    (
        container_bytes,
        metadata_offset,
        metadata_actual,
        metadata_region,
        model_offset,
        model_actual,
        model_region,
        directory_offset,
        directory_actual,
        directory_region,
        frames_offset,
        frames_end,
        final_padding_offset,
        final_padding_length,
    ) = ranges
    baseline_container_bytes, baseline_relative_mse, energy_convention = struct.unpack_from("<QdI", raw, 184)
    if raw[204:208] != bytes(4):
        raise FormatError("nonzero score-header reserved bytes")
    require_zero(raw[HEADER_RESERVED_OFFSET:GLOBAL_HEADER_SIZE], "global header")
    try:
        bindings = HeaderBindings(
            baseline_container_bytes=baseline_container_bytes,
            baseline_relative_mse=baseline_relative_mse,
            energy_convention=energy_convention,
            baseline_container_hash=raw[208:240],
            baseline_plan_lock_hash=raw[240:272],
            baseline_audit_hash=raw[272:304],
            universal_decoder_hash=raw[304:336],
            source_manifest_hash=raw[336:368],
            audit_bootstrap_hash=raw[368:400],
            decoded_reconstruction_hash=raw[432:464],
        )
    except ValueError as exc:
        raise FormatError(str(exc)) from exc
    return ParsedHeader(
        source_weights=source_weights,
        expert_count=expert_count,
        block_count=block_count,
        role_count=role_count,
        state_count=state_count,
        reset_length=reset_length,
        prior_bin_count=prior_bin_count,
        level_count=level_count,
        topology_id=topology_id,
        container_bytes=container_bytes,
        metadata_offset=metadata_offset,
        metadata_actual=metadata_actual,
        metadata_region=metadata_region,
        model_offset=model_offset,
        model_actual=model_actual,
        model_region=model_region,
        directory_offset=directory_offset,
        directory_actual=directory_actual,
        directory_region=directory_region,
        frames_offset=frames_offset,
        frames_end=frames_end,
        final_padding_offset=final_padding_offset,
        final_padding_length=final_padding_length,
        bindings=bindings,
        metadata_hash=raw[464:496],
        model_hash=raw[496:528],
        directory_hash=raw[528:560],
        root_hash=raw[400:432],
    )


def parse_container(raw: bytes) -> ParsedContainer:
    header = _parse_header(raw)
    if header.container_bytes != len(raw) or len(raw) % PAGE_SIZE:
        raise FormatError("complete byte count/page rounding mismatch")
    if not (header.source_weights > 0 and 1 <= header.expert_count <= 64 and header.role_count > 0):
        raise FormatError("invalid global counts")

    expected_metadata_offset = GLOBAL_HEADER_SIZE
    expected_metadata_region = align_up(header.metadata_actual, PAGE_SIZE)
    expected_model_offset = expected_metadata_offset + expected_metadata_region
    expected_model_region = align_up(header.model_actual, PAGE_SIZE)
    expected_directory_offset = expected_model_offset + expected_model_region
    expected_directory_actual = BLOCK_COUNT * DIRECTORY_RECORD_SIZE
    expected_directory_region = align_up(expected_directory_actual, PAGE_SIZE)
    expected_frames_offset = expected_directory_offset + expected_directory_region
    if (
        header.metadata_offset != expected_metadata_offset
        or header.metadata_region != expected_metadata_region
        or header.model_offset != expected_model_offset
        or header.model_region != expected_model_region
        or header.directory_offset != expected_directory_offset
        or header.directory_actual != expected_directory_actual
        or header.directory_region != expected_directory_region
        or header.frames_offset != expected_frames_offset
    ):
        raise FormatError("ranges are not in the unique canonical layout")
    checked_range(header.metadata_offset, header.metadata_region, len(raw), "metadata")
    checked_range(header.model_offset, header.model_region, len(raw), "model")
    checked_range(header.directory_offset, header.directory_region, len(raw), "directory")
    if header.final_padding_offset != header.frames_end:
        raise FormatError("final padding does not begin at frame end")
    if header.frames_end + header.final_padding_length != len(raw):
        raise FormatError("final padding length mismatch")
    minimum_floor_bytes = (215 * header.source_weights + 799) // 800
    expected_complete = align_up(max(header.frames_end, minimum_floor_bytes), PAGE_SIZE)
    if len(raw) != expected_complete:
        raise FormatError("nonminimal final padding")

    metadata_end = header.metadata_offset + header.metadata_actual
    model_end = header.model_offset + header.model_actual
    directory_end = header.directory_offset + header.directory_actual
    metadata = raw[header.metadata_offset:metadata_end]
    model_bytes = raw[header.model_offset:model_end]
    directory_bytes = raw[header.directory_offset:directory_end]
    require_zero(raw[metadata_end : header.metadata_offset + header.metadata_region], "metadata page tail")
    require_zero(raw[model_end : header.model_offset + header.model_region], "model page tail")
    require_zero(raw[directory_end : header.directory_offset + header.directory_region], "directory page tail")
    require_zero(raw[header.frames_end:], "final padding")
    if sha256(metadata) != header.metadata_hash:
        raise FormatError("metadata hash mismatch")
    if sha256(model_bytes) != header.model_hash:
        raise FormatError("model hash mismatch")
    if sha256(directory_bytes) != header.directory_hash:
        raise FormatError("directory hash mismatch")

    model = deserialize_model(model_bytes)
    if (
        model.state_count != header.state_count
        or model.reset_length != header.reset_length
        or model.prior_bin_count != header.prior_bin_count
        or model.level_count != header.level_count
        or model.topology_id != header.topology_id
    ):
        raise FormatError("header/model topology mismatch")

    records: list[DirectoryRecord] = []
    frames: list[Frame] = []
    cursor = header.frames_offset
    for ordinal in range(BLOCK_COUNT):
        start = ordinal * DIRECTORY_RECORD_SIZE
        record = parse_directory_record(
            directory_bytes[start : start + DIRECTORY_RECORD_SIZE], ordinal
        )
        if record.owner_mask >> header.expert_count:
            raise FormatError("directory owner outside expert count")
        if record.role >= header.role_count:
            raise FormatError("directory role outside role count")
        if record.payload_offset != cursor:
            raise FormatError("frame offsets overlap, gap, or are out of order")
        _, frame_end = checked_range(record.payload_offset, record.physical_length, len(raw), "frame")
        if frame_end > header.frames_end:
            raise FormatError("frame extends into final padding")
        frame_bytes = raw[record.payload_offset:frame_end]
        if sha256(frame_bytes) != record.payload_hash:
            raise FormatError("frame SHA-256 mismatch")
        frame = parse_frame(frame_bytes, ordinal)
        if (
            frame.logical_bits != record.logical_bits
            or frame.decision_count != record.decision_count
            or len(frame.encoded_bytes) != record.encoded_length
        ):
            raise FormatError("frame header/directory mismatch")
        records.append(record)
        frames.append(frame)
        cursor = frame_end
    if cursor != header.frames_end:
        raise FormatError("declared frame end does not match records")
    _validate_owner_topology(tuple(record.owner_mask for record in records), header.expert_count)
    if _root_digest(raw[:GLOBAL_HEADER_SIZE], metadata, model_bytes, directory_bytes, frames) != header.root_hash:
        raise FormatError("container root hash mismatch")
    return ParsedContainer(raw, header, metadata, model_bytes, model, tuple(records), tuple(frames))


def _validate_owner_topology(owner_masks: Sequence[int], expert_count: int) -> None:
    """Enforce the v0 fifteen-frame route: 2 private + 1 shared/expert."""

    if expert_count != 6 or len(owner_masks) != BLOCK_COUNT:
        raise FormatError("v0 owner topology requires six experts and fifteen blocks")
    if any(mask <= 0 or mask >> expert_count for mask in owner_masks):
        raise FormatError("owner mask outside v0 expert universe")
    if any(mask.bit_count() != 1 for mask in owner_masks[:12]):
        raise FormatError("first twelve v0 frames must be expert-private")
    if any(mask.bit_count() != 2 for mask in owner_masks[12:]):
        raise FormatError("last three v0 frames must be pair-shared tails")
    for expert in range(expert_count):
        private = sum(mask == (1 << expert) for mask in owner_masks[:12])
        shared = sum(bool(mask & (1 << expert)) for mask in owner_masks[12:])
        if private != 2 or shared != 1:
            raise FormatError("each v0 expert must own two private and one shared frame")


@dataclass(frozen=True)
class PhysicalScore:
    rate: Fraction
    relative_mse: float
    f_actual: float
    rate_in_range: bool
    f_pass: bool


def physical_score(container_bytes: int, source_weights: int, relative_mse: float) -> PhysicalScore:
    if container_bytes < 0 or source_weights <= 0:
        raise ValueError("invalid physical score dimensions")
    if not math.isfinite(relative_mse) or relative_mse < 0:
        raise ValueError("invalid relative MSE")
    rate = Fraction(8 * container_bytes, source_weights)
    f_actual = relative_mse * (2.0 ** (2.0 * float(rate)))
    return PhysicalScore(
        rate=rate,
        relative_mse=relative_mse,
        f_actual=f_actual,
        rate_in_range=Fraction(43, 20) <= rate <= Fraction(5, 2),
        f_pass=f_actual <= 0.8,
    )


def _pages_for_range(offset: int, length: int) -> set[int]:
    if length == 0:
        return set()
    return set(range(offset // PAGE_SIZE, (offset + length - 1) // PAGE_SIZE + 1))


@dataclass(frozen=True)
class ExpertRead:
    expert: int
    touched_pages: tuple[int, ...]
    touched_bytes: int
    storage_share: Fraction
    amplification: float


@dataclass(frozen=True)
class ReadLedger:
    global_bytes: int
    experts: tuple[ExpertRead, ...]
    maximum_amplification: float
    storage_conservation: Fraction


def routed_read_ledger(container: ParsedContainer) -> ReadLedger:
    header = container.header
    frame_bytes = sum(record.physical_length for record in container.records)
    global_bytes = len(container.raw) - frame_bytes
    if global_bytes < 0:
        raise FormatError("negative global byte attribution")
    results: list[ExpertRead] = []
    storage_conservation = Fraction(global_bytes, 1)
    for record in container.records:
        storage_conservation += Fraction(record.physical_length, 1)

    for expert in range(header.expert_count):
        pages = set()
        pages |= _pages_for_range(0, GLOBAL_HEADER_SIZE)
        pages |= _pages_for_range(header.metadata_offset, header.metadata_actual)
        pages |= _pages_for_range(header.model_offset, header.model_actual)
        directory_record_offset = header.directory_offset
        # An expert addresses every owned record.  In v0 all 15 records occupy
        # one page, but use exact record pages so the formula remains general.
        owned_records = [record for record in container.records if record.owner_mask & (1 << expert)]
        for record in owned_records:
            pages |= _pages_for_range(
                directory_record_offset + record.ordinal * DIRECTORY_RECORD_SIZE,
                DIRECTORY_RECORD_SIZE,
            )
            pages |= _pages_for_range(record.payload_offset, record.physical_length)
        share = Fraction(global_bytes, header.expert_count)
        for record in owned_records:
            share += Fraction(record.physical_length, record.owner_mask.bit_count())
        touched_bytes = len(pages) * PAGE_SIZE
        amplification = float(Fraction(touched_bytes, 1) / share)
        results.append(
            ExpertRead(expert, tuple(sorted(pages)), touched_bytes, share, amplification)
        )
    attributed_total = sum((result.storage_share for result in results), Fraction(0, 1))
    if attributed_total != storage_conservation:
        raise FormatError("owner-aware storage shares do not conserve bytes")
    return ReadLedger(
        global_bytes=global_bytes,
        experts=tuple(results),
        maximum_amplification=max(result.amplification for result in results),
        storage_conservation=storage_conservation,
    )


class CompletionLastCapsule:
    """Small lifecycle primitive: COMPLETE.json is emitted exclusively last."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError("capsule output must be absent")
        self.path.mkdir(parents=False, exist_ok=False)
        self._complete = False

    @staticmethod
    def _safe_leaf(name: str) -> str:
        if not name or name in (".", "..") or Path(name).name != name:
            raise ValueError("artifact name must be one plain leaf")
        if name == "COMPLETE.json":
            raise ValueError("COMPLETE.json is reserved for complete()")
        return name

    def write_bytes(self, name: str, data: bytes) -> Path:
        if self._complete:
            raise RuntimeError("all capsule writes are disabled after completion")
        leaf = self._safe_leaf(name)
        target = self.path / leaf
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(target, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        return target

    def complete(self, summary: dict[str, object]) -> Path:
        if self._complete:
            raise RuntimeError("capsule already complete")
        if not isinstance(summary, dict):
            raise TypeError("completion summary must be an object")
        payload = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        target = self.path / "COMPLETE.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(target, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        self._complete = True
        return target
