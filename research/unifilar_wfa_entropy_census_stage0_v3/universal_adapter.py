#!/usr/bin/env python3
"""Shape-driven universal SwiGLU-MoE semantic protocol for UWFA-SC v3.

The codec never keys probabilities by model, layer, expert, or stream identity.
This packet carries only charged decode geometry: one (h,m) pair per expert and
an opaque, charged evaluation-plugin extension. Stream contribution intervals
live in their literal frame and are checked here for an exact nonoverlapping
partition of every Gate/Up/Down matrix.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


MAGIC = b"UWFSEM3\x00"
VERSION = 3
HEADER_BYTES = 128
RECORD_BYTES = 32
OWNER_SET_BYTES = 32
MAX_EXPERTS = 256
MAX_DIMENSION = 1 << 24
MAX_WEIGHTS = 1 << 50
MAX_EXTENSION_BYTES = 1 << 30
ROLES = ("gate", "up", "down")


def _sha(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} outside exact integer bound")
    return value


def _checked_matrix_weights(h: int, m: int) -> int:
    h = _integer(h, "hidden width", 1, MAX_DIMENSION)
    m = _integer(m, "intermediate width", 1, MAX_DIMENSION)
    if h > MAX_WEIGHTS // m:
        raise ValueError("matrix shape product overflow")
    matrix = h * m
    if matrix > MAX_WEIGHTS // 3:
        raise ValueError("expert SwiGLU weight product overflow")
    return matrix


@dataclass(frozen=True)
class ExpertShape:
    ordinal: int
    hidden: int
    intermediate: int

    @property
    def matrix_weights(self) -> int:
        return _checked_matrix_weights(self.hidden, self.intermediate)

    @property
    def expert_weights(self) -> int:
        return 3 * self.matrix_weights


def build_semantic_packet(shapes: Sequence[ExpertShape], extension: bytes = b"") -> bytes:
    if not isinstance(shapes, (tuple, list)) or not 1 <= len(shapes) <= MAX_EXPERTS:
        raise ValueError("semantic expert count")
    if not isinstance(extension, bytes) or len(extension) > MAX_EXTENSION_BYTES:
        raise ValueError("semantic extension bound")
    records = bytearray()
    weight_sum = 0
    for expected, shape in enumerate(shapes):
        if not isinstance(shape, ExpertShape) or shape.ordinal != expected:
            raise ValueError("canonical semantic expert ordinals")
        matrix = shape.matrix_weights
        if weight_sum > MAX_WEIGHTS - 3 * matrix:
            raise ValueError("semantic source-weight overflow")
        weight_sum += 3 * matrix
        records.extend(struct.pack("<IIQQQ", expected, 0, shape.hidden, shape.intermediate, 3 * matrix))
    total = HEADER_BYTES + len(records) + len(extension)
    header = bytearray(HEADER_BYTES)
    struct.pack_into(
        "<8sHHHHIIQQQQ",
        header,
        0,
        MAGIC,
        VERSION,
        HEADER_BYTES,
        OWNER_SET_BYTES,
        RECORD_BYTES,
        len(shapes),
        0,
        len(records),
        len(extension),
        total,
        weight_sum,
    )
    header[64:96] = _sha(bytes(records))
    header[96:128] = _sha(extension)
    return bytes(header) + bytes(records) + extension


def parse_semantic_packet(packet: bytes) -> dict[str, Any]:
    if not isinstance(packet, bytes) or not HEADER_BYTES + RECORD_BYTES <= len(packet) <= HEADER_BYTES + MAX_EXPERTS * RECORD_BYTES + MAX_EXTENSION_BYTES:
        raise ValueError("semantic packet byte bound")
    fields = struct.unpack_from("<8sHHHHIIQQQQ", packet, 0)
    magic, version, header_bytes, owner_bytes, record_bytes, experts, reserved, records_bytes, extension_bytes, total_bytes, weight_sum = fields
    if (magic, version, header_bytes, owner_bytes, record_bytes, reserved) != (MAGIC, VERSION, HEADER_BYTES, OWNER_SET_BYTES, RECORD_BYTES, 0):
        raise ValueError("semantic header constants")
    experts = _integer(experts, "semantic experts", 1, MAX_EXPERTS)
    if records_bytes != experts * RECORD_BYTES or total_bytes != len(packet) or HEADER_BYTES + records_bytes + extension_bytes != total_bytes:
        raise ValueError("semantic section geometry")
    records = packet[HEADER_BYTES:HEADER_BYTES + records_bytes]
    extension = packet[HEADER_BYTES + records_bytes:]
    if _sha(records) != packet[64:96] or _sha(extension) != packet[96:128]:
        raise ValueError("semantic packet digest")
    shapes: list[ExpertShape] = []
    calculated = 0
    for ordinal in range(experts):
        observed, reserved_r, hidden, intermediate, expert_weights = struct.unpack_from("<IIQQQ", records, ordinal * RECORD_BYTES)
        if observed != ordinal or reserved_r != 0:
            raise ValueError("semantic record ordinal/reserved")
        shape = ExpertShape(ordinal, _integer(hidden, "hidden", 1, MAX_DIMENSION), _integer(intermediate, "intermediate", 1, MAX_DIMENSION))
        if expert_weights != shape.expert_weights:
            raise ValueError("semantic expert shape/weight mismatch")
        if calculated > MAX_WEIGHTS - expert_weights:
            raise ValueError("semantic total-weight overflow")
        calculated += expert_weights
        shapes.append(shape)
    if calculated != weight_sum:
        raise ValueError("semantic total-weight mismatch")
    rebuilt = build_semantic_packet(shapes, extension)
    if rebuilt != packet:
        raise ValueError("noncanonical semantic packet")
    return {
        "experts": experts,
        "shapes": tuple(shapes),
        "source_weights": calculated,
        "extension": extension,
        "packet": packet,
    }


def validate_stream_coverage(semantics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    experts = int(semantics["experts"])
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError("nonempty semantic stream rows")
    expected_weights = int(semantics["source_weights"])
    intervals: dict[tuple[int, str], list[tuple[int, int, int]]] = {
        (expert, role): [] for expert in range(experts) for role in ROLES
    }
    stream_weight_sum = 0
    seen_ordinals: set[int] = set()
    for row in rows:
        ordinal = int(row["ordinal"])
        if ordinal in seen_ordinals:
            raise ValueError("duplicate stream ordinal during coverage")
        seen_ordinals.add(ordinal)
        stream_role = row["role"]
        if not isinstance(stream_role, str) or not stream_role:
            raise ValueError("empty stream coordinate role")
        contributions = row["owner_contributions"]
        if not isinstance(contributions, (tuple, list)) or not contributions:
            raise ValueError("empty stream owner contributions")
        local_sum = 0
        previous_key: tuple[int, int, int] | None = None
        contribution_owners: list[int] = []
        for contribution in contributions:
            if set(contribution) != {"expert", "role", "source_offset", "weight_count"}:
                raise ValueError("owner contribution schema")
            expert = _integer(contribution["expert"], "contribution expert", 0, experts - 1)
            role = contribution["role"]
            if role not in ROLES:
                raise ValueError("unknown contribution SwiGLU role")
            begin = _integer(contribution["source_offset"], "contribution source offset", 0, MAX_WEIGHTS - 1)
            count = _integer(contribution["weight_count"], "contribution weight count", 1, MAX_WEIGHTS)
            key = (expert, ROLES.index(role), begin)
            if previous_key is not None and key <= previous_key:
                raise ValueError("contributions must use canonical expert/role/offset order")
            previous_key = key
            if begin > MAX_WEIGHTS - count:
                raise ValueError("contribution interval overflow")
            local_sum += count
            if local_sum > MAX_WEIGHTS:
                raise ValueError("stream contribution sum overflow")
            intervals[(expert, role)].append((begin, begin + count, ordinal))
            contribution_owners.append(expert)
        if tuple(sorted(set(contribution_owners))) != tuple(row["owners"]):
            raise ValueError("contribution owners do not equal literal owner set")
        source_weights = _integer(row["source_weights"], "stream source weights", 1, MAX_WEIGHTS)
        if local_sum != source_weights:
            raise ValueError("stream contribution conservation")
        group_rows = _integer(row["group_rows"], "group rows", 1, MAX_DIMENSION)
        group_cols = _integer(row["group_cols"], "group cols", 1, MAX_DIMENSION)
        if group_rows > MAX_WEIGHTS // group_cols or group_rows * group_cols != source_weights:
            raise ValueError("group shape/source-weight mismatch")
        if stream_weight_sum > MAX_WEIGHTS - source_weights:
            raise ValueError("stream source-weight sum overflow")
        stream_weight_sum += source_weights
    if sorted(seen_ordinals) != list(range(len(rows))):
        raise ValueError("stream ordinal coverage")
    if stream_weight_sum != expected_weights:
        raise ValueError("all stream weights do not conserve semantic shapes")
    per_expert_role: list[dict[str, Any]] = []
    for shape in semantics["shapes"]:
        target = shape.matrix_weights
        for role in ROLES:
            ordered = sorted(intervals[(shape.ordinal, role)])
            cursor = 0
            for begin, end, _ordinal in ordered:
                if begin != cursor:
                    raise ValueError("source scalar interval overlap or hole")
                cursor = end
            if cursor != target:
                raise ValueError("source scalar role coverage mismatch")
            per_expert_role.append({"expert": shape.ordinal, "role": role, "weight_count": cursor, "stream_count": len(ordered)})
    return {
        "source_weights": stream_weight_sum,
        "experts": experts,
        "all_roles_exactly_covered": True,
        "per_expert_role": per_expert_role,
    }


def decode_with_callbacks(
    semantics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    decode_stream: Any,
    place_interval: Any,
) -> dict[str, Any]:
    """Drive authenticated decoder callbacks without identity probability keys."""
    coverage = validate_stream_coverage(semantics, rows)
    decoded = 0
    placed = 0
    for row in rows:
        values = decode_stream(row)
        if isinstance(values, (bytes, bytearray, memoryview)):
            value_count = len(values)
        elif isinstance(values, Sequence):
            value_count = len(values)
        else:
            raise ValueError("decoded stream must expose exact scalar sequence")
        expected_values = sum(int(item["weight_count"]) for item in row["owner_contributions"])
        if value_count != expected_values:
            raise ValueError("decoded stream scalar count differs from semantic contributions")
        decoded += 1
        cursor = 0
        for contribution in row["owner_contributions"]:
            count = int(contribution["weight_count"])
            place_interval(
                int(contribution["expert"]),
                str(contribution["role"]),
                int(contribution["source_offset"]),
                count,
                values[cursor:cursor + count],
            )
            cursor += count
            placed += 1
        if cursor != value_count:
            raise ValueError("decoded stream slicing conservation")
    return {"decoded_streams": decoded, "placed_intervals": placed, "coverage": coverage}
