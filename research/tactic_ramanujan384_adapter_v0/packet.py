#!/usr/bin/env python3
"""Canonical literal 384-bit Ramanujan/polyphase refinement record."""

from __future__ import annotations

import math
import struct
import zlib
from typing import Iterable, Sequence


MAGIC = 0x4652  # bytes b"RF" under the little-endian bit writer
VERSION = 1
PACKET_BYTES = 48
BODY_BYTES = 44
BODY_BITS = 352
MAX_RANK = 14
ATOM_BITS = 9
ATOM_COUNT = 384
COEFFICIENT_BITS = 11
COEFFICIENT_MIN = -1023
COEFFICIENT_MAX = 1023
ROLES = ("gate", "up", "down_transposed")


class PacketError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


class BitWriter:
    def __init__(self) -> None:
        self._value = 0
        self._bits = 0

    def write(self, value: int, width: int) -> None:
        require(type(value) is int and type(width) is int and width >= 0, "bit field type")
        require(0 <= value < (1 << width), "bit field range")
        self._value |= value << self._bits
        self._bits += width

    def finish(self, bytes_: int) -> bytes:
        require(self._bits <= 8 * bytes_, "bit writer overflow")
        return self._value.to_bytes(bytes_, "little")


class BitReader:
    def __init__(self, payload: bytes) -> None:
        self._value = int.from_bytes(payload, "little")
        self._bits = 0
        self._limit = 8 * len(payload)

    @property
    def bits(self) -> int:
        return self._bits

    def read(self, width: int) -> int:
        require(type(width) is int and 0 <= width and self._bits + width <= self._limit, "bit read")
        result = (self._value >> self._bits) & ((1 << width) - 1)
        self._bits += width
        return result


def _f16_bits(value: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise PacketError("binary16 scale") from exc


def _from_f16_bits(bits: int) -> float:
    return float(struct.unpack("<e", struct.pack("<H", bits))[0])


def _signed_to_field(value: int) -> int:
    require(type(value) is int and COEFFICIENT_MIN <= value <= COEFFICIENT_MAX and value != 0,
            "signed coefficient")
    return value & ((1 << COEFFICIENT_BITS) - 1)


def _field_to_signed(value: int) -> int:
    sign = 1 << (COEFFICIENT_BITS - 1)
    result = value - (1 << COEFFICIENT_BITS) if value & sign else value
    require(COEFFICIENT_MIN <= result <= COEFFICIENT_MAX and result != 0,
            "canonical signed coefficient")
    return result


def encode_packet(role: str, support: Sequence[int], coefficients: Sequence[int], scale: float) -> bytes:
    require(role in ROLES, "role")
    support = tuple(int(value) for value in support)
    coefficients = tuple(int(value) for value in coefficients)
    require(len(support) == len(coefficients) <= MAX_RANK, "rank")
    require(tuple(sorted(support)) == support and len(set(support)) == len(support),
            "canonical support order")
    require(all(0 <= value < ATOM_COUNT for value in support), "atom index")
    rank = len(support)
    scale_bits = _f16_bits(scale)
    canonical_scale = _from_f16_bits(scale_bits)
    if rank == 0:
        require(scale_bits == 0 and not coefficients, "zero-rank scale")
    else:
        require(math.isfinite(canonical_scale) and canonical_scale > 0.0,
                "positive finite binary16 scale")
    writer = BitWriter()
    writer.write(MAGIC, 16)
    writer.write(VERSION, 4)
    writer.write(ROLES.index(role), 2)
    writer.write(rank, 4)
    writer.write(scale_bits, 16)
    for atom, coefficient in zip(support, coefficients, strict=True):
        writer.write(atom, ATOM_BITS)
        writer.write(_signed_to_field(coefficient), COEFFICIENT_BITS)
    body = writer.finish(BODY_BYTES)
    packet = body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    require(len(packet) == PACKET_BYTES, "packet size")
    return packet


def decode_packet(payload: bytes) -> dict[str, object]:
    require(type(payload) is bytes and len(payload) == PACKET_BYTES, "literal packet size")
    body = payload[:BODY_BYTES]
    expected_crc = struct.unpack("<I", payload[BODY_BYTES:])[0]
    require((zlib.crc32(body) & 0xFFFFFFFF) == expected_crc, "packet CRC32")
    reader = BitReader(body)
    require(reader.read(16) == MAGIC, "packet magic")
    require(reader.read(4) == VERSION, "packet version")
    role_ordinal = reader.read(2)
    require(role_ordinal < len(ROLES), "packet role")
    rank = reader.read(4)
    require(rank <= MAX_RANK, "packet rank")
    scale_bits = reader.read(16)
    scale = _from_f16_bits(scale_bits)
    support = []
    coefficients = []
    for _ in range(rank):
        support.append(reader.read(ATOM_BITS))
        coefficients.append(_field_to_signed(reader.read(COEFFICIENT_BITS)))
    require(all(value < ATOM_COUNT for value in support), "atom index")
    require(tuple(sorted(support)) == tuple(support) and len(set(support)) == rank,
            "canonical support order")
    if rank == 0:
        require(scale_bits == 0, "canonical zero-rank scale")
    else:
        require(math.isfinite(scale) and scale > 0.0, "positive finite binary16 scale")
    remaining = BODY_BITS - reader.bits
    require(reader.read(remaining) == 0, "canonical zero padding")
    result = {
        "role": ROLES[role_ordinal],
        "support": tuple(support),
        "coefficients": tuple(coefficients),
        "scale": scale,
        "rank": rank,
        "padding_bits": BODY_BITS - (42 + rank * 20),
    }
    require(encode_packet(result["role"], result["support"], result["coefficients"], result["scale"]) == payload,
            "canonical packet reencode")
    return result


def concatenate_packets(packets: Iterable[bytes]) -> bytes:
    rows = tuple(packets)
    require(all(type(row) is bytes and len(row) == PACKET_BYTES for row in rows), "packet stream")
    return b"".join(rows)


def split_packets(payload: bytes) -> tuple[bytes, ...]:
    require(type(payload) is bytes and len(payload) % PACKET_BYTES == 0, "fine stream size")
    rows = tuple(payload[index:index + PACKET_BYTES] for index in range(0, len(payload), PACKET_BYTES))
    for row in rows:
        decode_packet(row)
    require(concatenate_packets(rows) == payload, "fine stream canonical reencode")
    return rows
