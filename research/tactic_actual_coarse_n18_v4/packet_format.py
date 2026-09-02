#!/usr/bin/env python3
"""Finite expert-local UNIPOLAR-N18-307 v4 packet grammar.

This module is deliberately standard-library only.  It defines the bytes
that a numerical encoder must emit and that an independent decoder must
consume.  It does not open model files or import a numerical runtime.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Iterable


MAGIC = b"TACN18C4"
VERSION = 4
HEADER_BYTES = 128
RESERVOIR_BYTES = 78_592
PAYLOAD_BYTES = RESERVOIR_BYTES - HEADER_BYTES
PAYLOAD_BITS = PAYLOAD_BYTES * 8
N = 1 << 18
SQRT_N = 512
PROFILE_Q = 164
PROFILE_RATE_NUMERATOR = 307
PROFILE_RATE_DENOMINATOR = 128
TEST_CHANNEL_RATE_NUMERATOR = 153
TEST_CHANNEL_RATE_DENOMINATOR = 64
NOMINAL_BITS = N * TEST_CHANNEL_RATE_NUMERATOR // TEST_CHANNEL_RATE_DENOMINATOR
ETA = 0.25
ROLES = ("gate", "up", "down_transposed")
ROLE_TO_ORDINAL = {role: ordinal for ordinal, role in enumerate(ROLES)}

# The arithmetic payload has a canonical logical EOF.  The arithmetic
# decoder may use virtual zero bits after that EOF, as required by the normal
# 32-bit finalization convention; physical reservoir fill is never logical.
FLAG_ARITHMETIC = 1 << 0
FLAG_SIGNED_RHT = 1 << 1
FLAG_MAP_SC = 1 << 2
FLAG_ZERO_FILL = 1 << 3
FLAG_CANONICAL_REENCODE = 1 << 4
FLAG_ZERO_TILE = 1 << 5
FLAG_PADDED_TAIL = 1 << 6
BASE_FLAGS = (
    FLAG_ARITHMETIC
    | FLAG_SIGNED_RHT
    | FLAG_MAP_SC
    | FLAG_ZERO_FILL
    | FLAG_CANONICAL_REENCODE
)
SOURCE_ORDER = 1  # canonical row-major Gate, Up, then transposed Down
ALGORITHM_ID = hashlib.sha256(
    b"UNIPOLAR-N18-307-v4-Q31-BEC-MAP-SC-RHT-FP32"
).digest()[:16]
SC_DOMAIN = b"UNIPOLAR-N18-307-SC-v4\0"
RHT_DOMAIN = b"UNIPOLAR-N18-307-RHT-v4\0"

MAX_DIMENSION = 1 << 24
MAX_MATRIX_VALUES = 1 << 34
MAX_STREAMS_PER_MATRIX = 1 << 16
MAX_FRAME_BYTES = 1 << 40
PAGE_BYTES = 4096


class ContractError(RuntimeError):
    """The finite format or its accounting contract was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _positive_int(value: Any, maximum: int, label: str) -> int:
    require(type(value) is int and 0 < value <= maximum, label)
    return value


def _nonnegative_int(value: Any, maximum: int, label: str) -> int:
    require(type(value) is int and 0 <= value <= maximum, label)
    return value


def _stored_fp32(value: float) -> float:
    require(type(value) in (float, int) and math.isfinite(float(value)), "finite scale")
    try:
        stored = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise ContractError("FP32 scale range") from exc
    require(math.isfinite(stored) and stored > 0.0, "positive FP32 scale")
    return stored


def sha256_bytes(value: bytes) -> str:
    require(type(value) is bytes, "hash input bytes")
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class ExpertGeometry:
    """Canonical SwiGLU expert geometry.

    Gate and Up are ``[intermediate, hidden]``.  Down must be transposed to
    the same canonical coordinates before encoding.
    """

    intermediate: int
    hidden: int

    def __post_init__(self) -> None:
        _positive_int(self.intermediate, MAX_DIMENSION, "intermediate dimension")
        _positive_int(self.hidden, MAX_DIMENSION, "hidden dimension")
        require(
            self.intermediate <= MAX_MATRIX_VALUES // self.hidden,
            "matrix value product",
        )
        require(self.streams_per_role <= MAX_STREAMS_PER_MATRIX, "stream count cap")

    @property
    def role_values(self) -> int:
        return self.intermediate * self.hidden

    @property
    def values(self) -> int:
        return 3 * self.role_values

    @property
    def streams_per_role(self) -> int:
        return (self.role_values + N - 1) // N

    @property
    def records(self) -> int:
        return 3 * self.streams_per_role

    @property
    def frame_bytes(self) -> int:
        value = self.records * RESERVOIR_BYTES
        require(value <= MAX_FRAME_BYTES, "frame byte cap")
        return value

    @property
    def target_eligible(self) -> bool:
        # V4 deliberately preserves one exact N18 language.  Tail records are
        # compatibility packets and are never silently promoted into the
        # 307/128 target cell.
        return self.role_values % N == 0

    def valid_values(self, tile_ordinal: int) -> int:
        _nonnegative_int(tile_ordinal, self.streams_per_role - 1, "tile ordinal")
        return min(N, self.role_values - tile_ordinal * N)


@dataclass(frozen=True)
class ParsedReservoir:
    geometry: ExpertGeometry
    role_ordinal: int
    tile_ordinal: int
    valid_values: int
    zero_tile: bool
    padded_tail: bool
    sc_seed_u32: int
    rht_seed_u64: int
    decoder_scale_fp32: float
    logical_bits: int
    payload: bytes


@dataclass(frozen=True)
class ParsedExpertFrame:
    geometry: ExpertGeometry
    records: tuple[ParsedReservoir, ...]
    frame_sha256: str
    target_eligible: bool


def seed_pair(
    role_ordinal: int,
    intermediate: int,
    hidden: int,
    tile_ordinal: int,
) -> tuple[int, int]:
    _nonnegative_int(role_ordinal, len(ROLES) - 1, "seed role")
    geometry = ExpertGeometry(intermediate, hidden)
    _nonnegative_int(tile_ordinal, geometry.streams_per_role - 1, "seed tile")
    suffix = struct.pack(
        "<BIII", role_ordinal, geometry.intermediate, geometry.hidden, tile_ordinal
    )
    sc_seed = int.from_bytes(hashlib.sha256(SC_DOMAIN + suffix).digest()[:4], "little") or 1
    rht_seed = int.from_bytes(hashlib.sha256(RHT_DOMAIN + suffix).digest()[:8], "little")
    return sc_seed, rht_seed


def bits_to_payload(bits: Iterable[int]) -> tuple[bytes, int]:
    output = bytearray()
    current = 0
    count = 0
    for value in bits:
        require(type(value) is int and value in (0, 1), "binary logical symbol")
        count += 1
        require(count <= PAYLOAD_BITS, "logical payload overflow")
        current = (current << 1) | value
        if count % 8 == 0:
            output.append(current)
            current = 0
    if count % 8:
        output.append(current << (8 - count % 8))
    return bytes(output), count


def payload_to_bits(payload: bytes, logical_bits: int) -> tuple[int, ...]:
    require(type(payload) is bytes, "logical payload bytes")
    _nonnegative_int(logical_bits, PAYLOAD_BITS, "logical bit count")
    require(len(payload) == (logical_bits + 7) // 8, "logical payload byte length")
    if logical_bits % 8 and payload:
        require(
            payload[-1] & ((1 << (8 - logical_bits % 8)) - 1) == 0,
            "nonzero terminal padding bits",
        )
    return tuple(
        (payload[index // 8] >> (7 - index % 8)) & 1
        for index in range(logical_bits)
    )


def _header(
    *,
    geometry: ExpertGeometry,
    role_ordinal: int,
    tile_ordinal: int,
    scale: float,
    logical_bits: int,
    payload: bytes,
    zero_tile: bool,
) -> bytes:
    _nonnegative_int(role_ordinal, len(ROLES) - 1, "header role")
    _nonnegative_int(tile_ordinal, geometry.streams_per_role - 1, "header tile")
    _nonnegative_int(logical_bits, PAYLOAD_BITS, "header logical bits")
    valid_values = geometry.valid_values(tile_ordinal)
    padded_tail = valid_values != N
    if zero_tile:
        require(logical_bits == 0 and payload == b"", "zero tile has no payload")
        stored_scale = _stored_fp32(1.0)
    else:
        require(logical_bits > 0, "nonzero tile logical payload")
        stored_scale = _stored_fp32(scale)
    flags = BASE_FLAGS
    if zero_tile:
        flags |= FLAG_ZERO_TILE
    if padded_tail:
        flags |= FLAG_PADDED_TAIL
    sc_seed, rht_seed = seed_pair(
        role_ordinal, geometry.intermediate, geometry.hidden, tile_ordinal
    )
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<8sHHI", header, 0, MAGIC, VERSION, HEADER_BYTES, N)
    struct.pack_into("<BBH", header, 16, PROFILE_Q, role_ordinal, flags)
    struct.pack_into(
        "<IIII",
        header,
        20,
        geometry.intermediate,
        geometry.hidden,
        tile_ordinal,
        valid_values,
    )
    struct.pack_into("<IQfI", header, 36, sc_seed, rht_seed, stored_scale, logical_bits)
    header[56:88] = hashlib.sha256(payload).digest()
    header[88:104] = ALGORITHM_ID
    struct.pack_into("<IIII", header, 104, N - valid_values, int(padded_tail), SOURCE_ORDER, 0)
    struct.pack_into("<I", header, 120, zlib.crc32(payload) & 0xFFFFFFFF)
    struct.pack_into("<I", header, 124, zlib.crc32(header[:124]) & 0xFFFFFFFF)
    return bytes(header)


def pack_reservoir(
    payload: bytes,
    logical_bits: int,
    scale: float,
    geometry: ExpertGeometry,
    role_ordinal: int,
    tile_ordinal: int,
    *,
    zero_tile: bool = False,
) -> bytes:
    require(type(payload) is bytes, "payload bytes")
    _nonnegative_int(logical_bits, PAYLOAD_BITS, "logical bits")
    require(len(payload) == (logical_bits + 7) // 8, "logical payload byte count")
    payload_to_bits(payload, logical_bits)
    header = _header(
        geometry=geometry,
        role_ordinal=role_ordinal,
        tile_ordinal=tile_ordinal,
        scale=scale,
        logical_bits=logical_bits,
        payload=payload,
        zero_tile=zero_tile,
    )
    packet = header + payload + bytes(PAYLOAD_BYTES - len(payload))
    require(len(packet) == RESERVOIR_BYTES, "fixed reservoir byte count")
    return packet


def parse_reservoir(packet: bytes) -> ParsedReservoir:
    require(type(packet) is bytes and len(packet) == RESERVOIR_BYTES, "reservoir byte count")
    magic, version, header_bytes, n = struct.unpack_from("<8sHHI", packet, 0)
    require(
        (magic, version, header_bytes, n) == (MAGIC, VERSION, HEADER_BYTES, N),
        "header constants",
    )
    profile_q, role_ordinal, flags = struct.unpack_from("<BBH", packet, 16)
    require(profile_q == PROFILE_Q and 0 <= role_ordinal < len(ROLES), "profile/role")
    intermediate, hidden, tile_ordinal, valid_values = struct.unpack_from("<IIII", packet, 20)
    geometry = ExpertGeometry(intermediate, hidden)
    require(tile_ordinal < geometry.streams_per_role, "tile ordinal")
    require(valid_values == geometry.valid_values(tile_ordinal), "valid-values field")
    padded_tail = valid_values != N
    zero_tile = bool(flags & FLAG_ZERO_TILE)
    expected_flags = BASE_FLAGS
    if zero_tile:
        expected_flags |= FLAG_ZERO_TILE
    if padded_tail:
        expected_flags |= FLAG_PADDED_TAIL
    require(flags == expected_flags, "packet flags")
    sc_seed, rht_seed, scale, logical_bits = struct.unpack_from("<IQfI", packet, 36)
    require(
        (sc_seed, rht_seed)
        == seed_pair(role_ordinal, intermediate, hidden, tile_ordinal),
        "universal seeds",
    )
    require(math.isfinite(scale) and scale > 0.0, "stored scale")
    require(logical_bits <= PAYLOAD_BITS, "logical payload capacity")
    if zero_tile:
        require(logical_bits == 0 and scale == 1.0, "zero-tile fields")
    else:
        require(logical_bits > 0, "nonzero-tile logical length")
    require(packet[88:104] == ALGORITHM_ID, "algorithm identifier")
    padding_values, tail_marker, source_order, reserved = struct.unpack_from("<IIII", packet, 104)
    require(padding_values == N - valid_values, "padding-value count")
    require(tail_marker == int(padded_tail), "tail marker")
    require(source_order == SOURCE_ORDER and reserved == 0, "source-order/reserved fields")
    header_crc = struct.unpack_from("<I", packet, 124)[0]
    require(header_crc == zlib.crc32(packet[:124]) & 0xFFFFFFFF, "header CRC")
    used = (logical_bits + 7) // 8
    payload = packet[HEADER_BYTES : HEADER_BYTES + used]
    require(hashlib.sha256(payload).digest() == packet[56:88], "payload SHA-256")
    require(
        struct.unpack_from("<I", packet, 120)[0] == zlib.crc32(payload) & 0xFFFFFFFF,
        "payload CRC32",
    )
    payload_to_bits(payload, logical_bits)
    require(
        packet[HEADER_BYTES + used :] == bytes(PAYLOAD_BYTES - used),
        "nonzero physical reservoir fill",
    )
    return ParsedReservoir(
        geometry=geometry,
        role_ordinal=role_ordinal,
        tile_ordinal=tile_ordinal,
        valid_values=valid_values,
        zero_tile=zero_tile,
        padded_tail=padded_tail,
        sc_seed_u32=sc_seed,
        rht_seed_u64=rht_seed,
        decoder_scale_fp32=float(scale),
        logical_bits=logical_bits,
        payload=payload,
    )


def canonical_packet_reencode(packet: bytes, decoded_bits: Iterable[int]) -> bytes:
    parsed = parse_reservoir(packet)
    payload, logical_bits = bits_to_payload(decoded_bits)
    require(
        payload == parsed.payload and logical_bits == parsed.logical_bits,
        "decoded logical symbols differ from packet",
    )
    return pack_reservoir(
        payload,
        logical_bits,
        parsed.decoder_scale_fp32,
        parsed.geometry,
        parsed.role_ordinal,
        parsed.tile_ordinal,
        zero_tile=parsed.zero_tile,
    )


def parse_expert_frame(frame: bytes) -> ParsedExpertFrame:
    require(type(frame) is bytes and len(frame) >= 3 * RESERVOIR_BYTES, "expert frame size")
    require(len(frame) % RESERVOIR_BYTES == 0, "expert frame record alignment")
    require(len(frame) <= MAX_FRAME_BYTES, "expert frame cap")
    first = parse_reservoir(frame[:RESERVOIR_BYTES])
    require(first.role_ordinal == 0 and first.tile_ordinal == 0, "canonical first record")
    geometry = first.geometry
    expected_records = geometry.records
    require(len(frame) == expected_records * RESERVOIR_BYTES, "expert frame record count")
    records: list[ParsedReservoir] = []
    cursor = 0
    for role_ordinal in range(len(ROLES)):
        for tile_ordinal in range(geometry.streams_per_role):
            packet = frame[cursor : cursor + RESERVOIR_BYTES]
            row = parse_reservoir(packet)
            require(row.geometry == geometry, "frame geometry drift")
            require(
                (row.role_ordinal, row.tile_ordinal) == (role_ordinal, tile_ordinal),
                "noncanonical role/tile order",
            )
            records.append(row)
            cursor += RESERVOIR_BYTES
    require(cursor == len(frame), "expert frame exact exhaustion")
    return ParsedExpertFrame(
        geometry=geometry,
        records=tuple(records),
        frame_sha256=hashlib.sha256(frame).hexdigest(),
        target_eligible=geometry.target_eligible,
    )


def frame_ledger(
    geometry: ExpertGeometry,
    *,
    start_offset_mod_page: int = 0,
    compressed_passes: int = 1,
) -> dict[str, Any]:
    _nonnegative_int(start_offset_mod_page, PAGE_BYTES - 1, "page offset")
    _positive_int(compressed_passes, 16, "compressed pass count")
    frame_bytes = geometry.frame_bytes
    touched_pages = (start_offset_mod_page + frame_bytes + PAGE_BYTES - 1) // PAGE_BYTES
    unique_page_bytes = touched_pages * PAGE_BYTES
    worst_pages = (PAGE_BYTES - 1 + frame_bytes + PAGE_BYTES - 1) // PAGE_BYTES
    worst_page_bytes = worst_pages * PAGE_BYTES
    return {
        "intermediate": geometry.intermediate,
        "hidden": geometry.hidden,
        "weights": geometry.values,
        "records": geometry.records,
        "frame_bytes": frame_bytes,
        "physical_bpw": 8.0 * frame_bytes / geometry.values,
        "target_eligible_exact_307_over_128": geometry.target_eligible,
        "tail_values_per_role": geometry.role_values % N,
        "one_pass_schedule": compressed_passes == 1,
        "compressed_passes": compressed_passes,
        "repeated_compressed_bytes": compressed_passes * frame_bytes,
        "repeated_byte_amplification": float(compressed_passes),
        "start_offset_mod_page": start_offset_mod_page,
        "unique_pages": touched_pages,
        "unique_page_bytes": unique_page_bytes,
        "unique_page_amplification": unique_page_bytes / frame_bytes,
        "worst_alignment_unique_page_bytes": worst_page_bytes,
        "worst_alignment_unique_page_amplification": worst_page_bytes / frame_bytes,
    }


def qwen_frozen_ledgers() -> dict[str, Any]:
    geometry = ExpertGeometry(768, 2048)
    require(geometry.role_values == 6 * N and geometry.records == 18, "Qwen N18 geometry")
    coarse_expert = geometry.frame_bytes
    coarse_panel = 6 * coarse_expert
    weights_panel = 6 * geometry.values
    require(coarse_expert == 1_414_656 and coarse_panel == 8_487_936, "Qwen coarse bytes")
    require(8 * coarse_panel * 128 == weights_panel * 307, "Qwen exact 307/128")

    # This is the already-frozen TACTIC planning topology.  V4 implements only
    # its coarse prefix; the fine and common packets remain separate work.
    fine_panel = 331_776
    expert_headers_panel = 3_072
    global_packet = 24_576
    final_panel = coarse_panel + fine_panel + expert_headers_panel + global_packet
    require(final_panel == 8_847_360, "frozen 2.5-bpw panel")
    private_expert = coarse_expert + fine_panel // 6 + expert_headers_panel // 6
    routed_expert = private_expert + global_packet
    equal_share = final_panel // 6
    require(private_expert == 359 * PAGE_BYTES, "private 359-page frame")
    require(routed_expert == 365 * PAGE_BYTES and equal_share == 360 * PAGE_BYTES, "73/72 pages")
    second_private_pass_pages = 6 + 2 * 359
    require(second_private_pass_pages == 724, "second-pass page identity")
    return {
        "coarse": {
            "weights": weights_panel,
            "panel_bytes": coarse_panel,
            "expert_bytes": coarse_expert,
            "physical_bpw": 307 / 128,
            "model_bytes": 0,
            "expert_records": 18,
            "decoder_compressed_passes": 1,
        },
        "frozen_final_planning_topology_not_implemented_here": {
            "fine_panel_bytes": fine_panel,
            "expert_headers_panel_bytes": expert_headers_panel,
            "global_packet_bytes": global_packet,
            "panel_bytes": final_panel,
            "physical_bpw": 2.5,
            "selected_expert_private_pages": 359,
            "selected_global_pages": 6,
            "selected_pages": 365,
            "equal_share_pages": 360,
            "cold_page_amplification": 73 / 72,
            "forbidden_second_private_frame_pass_pages": second_private_pass_pages,
            "forbidden_second_private_frame_pass_amplification": 724 / 360,
            "second_private_frame_pass_satisfies_strict_below_2x": False,
        },
    }


def validate_constants() -> None:
    require(PAYLOAD_BITS == 627_712, "payload capacity")
    require(NOMINAL_BITS == 626_688, "nominal test-channel bits")
    require(PAYLOAD_BITS - NOMINAL_BITS == 1_024, "logical reserve")
    require(8 * RESERVOIR_BYTES * 128 == N * 307, "307/128 reservoir identity")
    qwen_frozen_ledgers()


validate_constants()
