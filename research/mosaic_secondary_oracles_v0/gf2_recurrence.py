#!/usr/bin/env python3
"""Exact GF(2) Berlekamp-Massey and canonical four-level block packets."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


MAGIC = b"LRB0"
VERSION = 1
MODE_RAW = 0
MODE_LFSR = 1
HEADER = struct.Struct("<4sHBB")
PLANE = struct.Struct("<BHB")
CRC = struct.Struct("<I")
COMPONENT_MAGIC = b"LRC0"
COMPONENT_HEADER = struct.Struct("<4sHHIIQQQQI12s")
EXPERT_MAGIC = b"LRE0"
EXPERT_HEADER = struct.Struct("<4sHHQQ4QI4s")
ROLE_IDS = {"gate": 0, "up": 1, "down_transposed": 2}
ROLE_NAMES = {value: key for key, value in ROLE_IDS.items()}
PAGE_BYTES = 4096


class RecurrenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecurrenceError(message)


def _bit(value: Any) -> int:
    require(type(value) is not bool or value in (False, True), "bit type")
    result = int(value)
    require(result in (0, 1) and value == result, "binary value")
    return result


def berlekamp_massey_gf2(values: Sequence[int]) -> tuple[int, int]:
    """Return exact linear complexity and connection polynomial as an int.

    Polynomial bit zero is the current-symbol coefficient and is always one;
    bit ``i`` is the lag-``i`` coefficient. Python big integers make each
    discrepancy a word-parallel parity operation.
    """

    bits = tuple(_bit(value) for value in values)
    require(bits, "nonempty GF2 sequence")
    connection = 1
    previous = 1
    complexity = 0
    shift = 1
    history = 0
    for index, bit in enumerate(bits):
        history = (history << 1) | bit
        discrepancy = (connection & history).bit_count() & 1
        if discrepancy:
            held = connection
            connection ^= previous << shift
            if 2 * complexity <= index:
                complexity = index + 1 - complexity
                previous = held
                shift = 1
            else:
                shift += 1
        else:
            shift += 1
    connection &= (1 << (complexity + 1)) - 1
    require(connection & 1 == 1, "connection constant")
    require(connection.bit_length() <= complexity + 1, "connection degree")
    return complexity, connection


def generate_lfsr(initial: Sequence[int], connection: int, length: int) -> tuple[int, ...]:
    initial_bits = tuple(_bit(value) for value in initial)
    complexity = len(initial_bits)
    require(type(connection) is int and connection >= 1 and connection & 1, "connection")
    require(connection.bit_length() <= complexity + 1, "connection degree")
    require(type(length) is int and length >= complexity, "generated length")
    output = list(initial_bits)
    for index in range(complexity, length):
        value = 0
        for lag in range(1, complexity + 1):
            value ^= ((connection >> lag) & 1) & output[index - lag]
        output.append(value)
    return tuple(output)


class BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def write(self, values: Iterable[int]) -> None:
        self.bits.extend(_bit(value) for value in values)

    def payload(self) -> bytes:
        output = bytearray((len(self.bits) + 7) // 8)
        for index, value in enumerate(self.bits):
            output[index >> 3] |= value << (7 - (index & 7))
        return bytes(output)


class BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.position = 0

    def read(self, count: int) -> tuple[int, ...]:
        require(type(count) is int and count >= 0, "bit count")
        require(self.position + count <= 8 * len(self.payload), "bitstream underflow")
        output = []
        for _ in range(count):
            byte = self.payload[self.position >> 3]
            output.append((byte >> (7 - (self.position & 7))) & 1)
            self.position += 1
        return tuple(output)


@dataclass(frozen=True)
class PlanePlan:
    mode: int
    complexity: int
    connection: int
    initial: tuple[int, ...]
    raw: tuple[int, ...]

    @property
    def payload_bits(self) -> int:
        return len(self.raw) if self.mode == MODE_RAW else 2 * self.complexity


def plan_plane(values: Sequence[int]) -> PlanePlan:
    raw = tuple(_bit(value) for value in values)
    complexity, connection = berlekamp_massey_gf2(raw)
    # Directory length is paid in both cases. Ties canonically choose raw.
    if 2 * complexity < len(raw):
        initial = raw[:complexity]
        require(generate_lfsr(initial, connection, len(raw)) == raw, "BM exact replay")
        return PlanePlan(MODE_LFSR, complexity, connection, initial, raw)
    return PlanePlan(MODE_RAW, 0, 1, (), raw)


def gray_planes(labels: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    mapping = ((0, 0), (0, 1), (1, 1), (1, 0))
    first = []
    second = []
    for value in labels:
        label = int(value)
        require(type(value) is not bool and 0 <= label < 4 and value == label, "four-level label")
        a, b = mapping[label]
        first.append(a)
        second.append(b)
    require(bool(first), "nonempty label block")
    return tuple(first), tuple(second)


def labels_from_gray(first: Sequence[int], second: Sequence[int]) -> tuple[int, ...]:
    require(len(first) == len(second) and len(first) > 0, "Gray plane geometry")
    inverse = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
    return tuple(inverse[(_bit(a), _bit(b))] for a, b in zip(first, second, strict=True))


def encode_block(labels: Sequence[int]) -> tuple[bytes, dict[str, Any]]:
    planes = gray_planes(labels)
    length = len(labels)
    require(length <= 65535, "block length")
    plans = tuple(plan_plane(plane) for plane in planes)
    writer = BitWriter()
    directory = bytearray()
    for plan in plans:
        directory.extend(PLANE.pack(plan.mode, plan.complexity, 0))
        if plan.mode == MODE_RAW:
            writer.write(plan.raw)
        else:
            writer.write((plan.connection >> lag) & 1 for lag in range(1, plan.complexity + 1))
            writer.write(plan.initial)
    payload_bits = sum(plan.payload_bits for plan in plans)
    payload = writer.payload()
    body = HEADER.pack(MAGIC, length, len(plans), VERSION) + bytes(directory) + payload
    packet = body + CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)
    decoded = decode_block(packet)
    require(decoded == tuple(int(value) for value in labels), "block canonical decode")
    require(encode_block_no_check(decoded) == packet, "block canonical re-encode")
    return packet, {
        "values": length,
        "planes": [
            {
                "mode": "lfsr" if plan.mode == MODE_LFSR else "raw",
                "linear_complexity": plan.complexity,
                "payload_bits": plan.payload_bits,
            }
            for plan in plans
        ],
        "logical_payload_bits": payload_bits,
        "physical_bytes": len(packet),
        "physical_bits": 8 * len(packet),
        "raw_label_bits": 2 * length,
        "saving_bits_before_outer_headers": 2 * length - 8 * len(packet),
    }


def encode_block_no_check(labels: Sequence[int]) -> bytes:
    planes = gray_planes(labels)
    plans = tuple(plan_plane(plane) for plane in planes)
    writer = BitWriter()
    directory = bytearray()
    for plan in plans:
        directory.extend(PLANE.pack(plan.mode, plan.complexity, 0))
        if plan.mode == MODE_RAW:
            writer.write(plan.raw)
        else:
            writer.write((plan.connection >> lag) & 1 for lag in range(1, plan.complexity + 1))
            writer.write(plan.initial)
    body = HEADER.pack(MAGIC, len(labels), len(plans), VERSION) + bytes(directory) + writer.payload()
    return body + CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode_block(packet: bytes) -> tuple[int, ...]:
    require(len(packet) >= HEADER.size + 2 * PLANE.size + CRC.size, "short recurrence packet")
    body = packet[:-CRC.size]
    claimed_crc, = CRC.unpack(packet[-CRC.size:])
    require(zlib.crc32(body) & 0xFFFFFFFF == claimed_crc, "recurrence CRC")
    magic, length, planes, version = HEADER.unpack_from(body, 0)
    require((magic, planes, version) == (MAGIC, 2, VERSION) and length > 0, "recurrence header")
    cursor = HEADER.size
    rows = []
    total_bits = 0
    for _ in range(planes):
        mode, complexity, reserved = PLANE.unpack_from(body, cursor)
        cursor += PLANE.size
        require(reserved == 0 and mode in (MODE_RAW, MODE_LFSR), "recurrence plane record")
        require((mode == MODE_RAW and complexity == 0) or (mode == MODE_LFSR and complexity <= length), "recurrence complexity")
        bits = length if mode == MODE_RAW else 2 * complexity
        rows.append((mode, complexity))
        total_bits += bits
    payload = body[cursor:]
    require(len(payload) == (total_bits + 7) // 8, "recurrence payload length")
    if total_bits & 7:
        require(payload[-1] & ((1 << (8 - (total_bits & 7))) - 1) == 0, "recurrence terminal padding")
    reader = BitReader(payload)
    decoded_planes = []
    for mode, complexity in rows:
        if mode == MODE_RAW:
            decoded_planes.append(reader.read(length))
        else:
            coefficients = reader.read(complexity)
            connection = 1
            for lag, value in enumerate(coefficients, start=1):
                connection |= value << lag
            initial = reader.read(complexity)
            decoded_planes.append(generate_lfsr(initial, connection, length))
    require(reader.position == total_bits, "recurrence payload consumption")
    labels = labels_from_gray(*decoded_planes)
    require(encode_block_no_check(labels) == packet, "recurrence canonical encoding")
    return labels


def component_packet_bytes(block_packets: Sequence[bytes], *, scale_bytes_per_block: int = 2) -> int:
    require(bool(block_packets) and all(isinstance(packet, bytes) and packet for packet in block_packets), "block packets")
    require(type(scale_bytes_per_block) is int and scale_bytes_per_block >= 0, "scale bytes")
    # 64-byte component header, 4-byte offsets per block plus terminal offset,
    # BF16 scales, literal block packets, and component-level 64-byte padding.
    raw = 64 + 4 * (len(block_packets) + 1) + scale_bytes_per_block * len(block_packets)
    raw += sum(len(packet) for packet in block_packets)
    return ((raw + 63) // 64) * 64


def encode_component(
    role: str,
    label_blocks: Sequence[Sequence[int]],
    scale_f16le: Sequence[bytes],
) -> bytes:
    require(role in ROLE_IDS, "component role")
    require(bool(label_blocks) and len(label_blocks) == len(scale_f16le), "component blocks/scales")
    block_values = len(label_blocks[0])
    require(block_values > 0 and all(0 < len(block) <= block_values for block in label_blocks), "component block lengths")
    require(all(len(block) == block_values for block in label_blocks[:-1]), "only final block may be short")
    require(all(isinstance(value, bytes) and len(value) == 2 for value in scale_f16le), "component BF16 scales")
    packets = [encode_block_no_check(block) for block in label_blocks]
    offsets = [0]
    payload = bytearray()
    for packet in packets:
        payload.extend(packet)
        offsets.append(len(payload))
    directory = struct.pack(f"<{len(offsets)}I", *offsets)
    scales = b"".join(scale_f16le)
    directory_offset = COMPONENT_HEADER.size
    scales_offset = directory_offset + len(directory)
    payload_offset = scales_offset + len(scales)
    total = ((payload_offset + len(payload) + 63) // 64) * 64
    body = directory + scales + bytes(payload) + bytes(total - payload_offset - len(payload))
    zero_header = COMPONENT_HEADER.pack(
        COMPONENT_MAGIC,
        VERSION,
        ROLE_IDS[role],
        len(label_blocks),
        block_values,
        directory_offset,
        scales_offset,
        payload_offset,
        total,
        0,
        bytes(12),
    )
    crc = zlib.crc32(zero_header + body) & 0xFFFFFFFF
    header = COMPONENT_HEADER.pack(
        COMPONENT_MAGIC,
        VERSION,
        ROLE_IDS[role],
        len(label_blocks),
        block_values,
        directory_offset,
        scales_offset,
        payload_offset,
        total,
        crc,
        bytes(12),
    )
    packet = header + body
    decoded = decode_component(packet)
    require(decoded["role"] == role and decoded["label_blocks"] == tuple(tuple(int(v) for v in row) for row in label_blocks), "component canonical decode")
    return packet


def decode_component(packet: bytes) -> dict[str, Any]:
    require(len(packet) >= COMPONENT_HEADER.size and len(packet) % 64 == 0, "component byte geometry")
    fields = COMPONENT_HEADER.unpack_from(packet, 0)
    (
        magic,
        version,
        role_id,
        blocks,
        block_values,
        directory_offset,
        scales_offset,
        payload_offset,
        total,
        claimed_crc,
        reserved,
    ) = fields
    require(magic == COMPONENT_MAGIC and version == VERSION and role_id in ROLE_NAMES, "component header")
    require(blocks > 0 and block_values > 0 and total == len(packet) and reserved == bytes(12), "component constants")
    expected_scales = directory_offset + 4 * (blocks + 1)
    expected_payload = expected_scales + 2 * blocks
    require(
        directory_offset == COMPONENT_HEADER.size
        and scales_offset == expected_scales
        and payload_offset == expected_payload
        and payload_offset <= total,
        "component offsets",
    )
    zero_header = COMPONENT_HEADER.pack(
        magic, version, role_id, blocks, block_values,
        directory_offset, scales_offset, payload_offset, total, 0, bytes(12),
    )
    require(zlib.crc32(zero_header + packet[COMPONENT_HEADER.size:]) & 0xFFFFFFFF == claimed_crc, "component CRC")
    offsets = struct.unpack_from(f"<{blocks + 1}I", packet, directory_offset)
    require(offsets[0] == 0 and all(a < b for a, b in zip(offsets, offsets[1:])), "component block offsets")
    payload_end = payload_offset + offsets[-1]
    canonical_total = ((payload_end + 63) // 64) * 64
    require(
        payload_end <= total
        and total == canonical_total
        and packet[payload_end:] == bytes(total - payload_end),
        "component canonical padding",
    )
    scales = tuple(packet[scales_offset + 2 * index:scales_offset + 2 * index + 2] for index in range(blocks))
    labels = []
    for begin, end in zip(offsets, offsets[1:]):
        labels.append(decode_block(packet[payload_offset + begin:payload_offset + end]))
    require(all(len(row) == block_values for row in labels[:-1]) and 0 < len(labels[-1]) <= block_values, "component decoded block geometry")
    return {
        "role": ROLE_NAMES[role_id],
        "block_values": block_values,
        "label_blocks": tuple(labels),
        "scale_f16le": scales,
        "physical_bytes": total,
        "crc32": claimed_crc,
    }


def encode_expert(
    role_components: Sequence[bytes],
    *,
    weights: int,
) -> bytes:
    require(len(role_components) == 3 and type(weights) is int and weights > 0, "expert geometry")
    parsed = [decode_component(packet) for packet in role_components]
    require([row["role"] for row in parsed] == ["gate", "up", "down_transposed"], "expert role order")
    require(sum(sum(len(block) for block in row["label_blocks"]) for row in parsed) == weights, "expert weight coverage")
    offsets = [EXPERT_HEADER.size]
    body = bytearray()
    for packet in role_components:
        target = ((EXPERT_HEADER.size + len(body) + 63) // 64) * 64
        body.extend(bytes(target - EXPERT_HEADER.size - len(body)))
        require(EXPERT_HEADER.size + len(body) == offsets[-1], "expert component alignment")
        body.extend(packet)
        offsets.append(EXPERT_HEADER.size + len(body))
    unpadded = EXPERT_HEADER.size + len(body)
    minimum_bytes = (43 * weights + 159) // 160
    total = ((max(unpadded, minimum_bytes) + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES
    body.extend(bytes(total - unpadded))
    zero_header = EXPERT_HEADER.pack(
        EXPERT_MAGIC, VERSION, 3, weights, total, *offsets, 0, bytes(4)
    )
    crc = zlib.crc32(zero_header + bytes(body)) & 0xFFFFFFFF
    header = EXPERT_HEADER.pack(
        EXPERT_MAGIC, VERSION, 3, weights, total, *offsets, crc, bytes(4)
    )
    packet = header + bytes(body)
    decoded = decode_expert(packet)
    require(decoded["weights"] == weights and decoded["component_packets"] == tuple(role_components), "expert canonical decode")
    return packet


def decode_expert(packet: bytes) -> dict[str, Any]:
    require(len(packet) >= EXPERT_HEADER.size and len(packet) % PAGE_BYTES == 0, "expert byte geometry")
    fields = EXPERT_HEADER.unpack_from(packet, 0)
    magic, version, roles, weights, total = fields[:5]
    offsets = tuple(int(value) for value in fields[5:9])
    claimed_crc = fields[9]
    reserved = fields[10]
    require((magic, version, roles) == (EXPERT_MAGIC, VERSION, 3), "expert header")
    require(weights > 0 and total == len(packet) and reserved == bytes(4), "expert constants")
    require(offsets[0] == EXPERT_HEADER.size and all(a < b for a, b in zip(offsets, offsets[1:])), "expert offsets")
    minimum_bytes = (43 * weights + 159) // 160
    canonical_total = ((max(offsets[-1], minimum_bytes) + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES
    require(
        offsets[-1] <= total
        and total == canonical_total
        and packet[offsets[-1]:] == bytes(total - offsets[-1]),
        "expert canonical padding",
    )
    zero_header = EXPERT_HEADER.pack(magic, version, roles, weights, total, *offsets, 0, bytes(4))
    require(zlib.crc32(zero_header + packet[EXPERT_HEADER.size:]) & 0xFFFFFFFF == claimed_crc, "expert CRC")
    components = tuple(packet[begin:end] for begin, end in zip(offsets, offsets[1:]))
    parsed = tuple(decode_component(component) for component in components)
    require([row["role"] for row in parsed] == ["gate", "up", "down_transposed"], "expert decoded roles")
    require(sum(sum(len(block) for block in row["label_blocks"]) for row in parsed) == weights, "expert decoded coverage")
    return {
        "weights": weights,
        "physical_bytes": total,
        "physical_rate_bpw": 8.0 * total / weights,
        "component_packets": components,
        "components": parsed,
        "crc32": claimed_crc,
    }
