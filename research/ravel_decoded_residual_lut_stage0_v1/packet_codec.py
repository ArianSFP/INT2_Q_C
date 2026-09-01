#!/usr/bin/env python3
"""Pure-standard-library RAVEL6144-v1 packet builder, parser, and scalar feature reference."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Iterable, Sequence


FORMAT = "RAVEL6144-v1"
VERSION = 1
HEADER_BYTES = 4096
ENTRIES = 6144
TABLE_BYTES = ENTRIES * 2
PACKET_BYTES = HEADER_BYTES + TABLE_BYTES
FEATURES = [3, 4, 32, 4, 4]
ROLE_ORDER = ["gate", "up", "down"]
SEMANTICS = {
    "amplitude": "floor((decoded/row_scale + 4)*4), clipped to [0,31]; lower edges inclusive, upper edges exclusive before saturation",
    "boundary": "noncyclic self-clamp; a missing horizontal neighbor equals the center",
    "edge_state": "2*(neighbor >= 0) + 1*(abs(neighbor) > abs(center)); zero is nonnegative; magnitude ties are false",
    "flatten": "((((role*4 + row_class)*32 + amplitude)*4 + left_state)*4 + right_state); right_state fastest",
    "matrix_scale": "max(sqrt(mean(decoded_matrix^2) in FP64), 1e-30)",
    "role_order": ROLE_ORDER,
    "row_class": "count(log2(row_scale/matrix_scale) > threshold for threshold in [-0.25,0,0.25]); equality stays lower",
    "row_scale": "max(sqrt(mean(decoded_row^2) in FP64), 1e-30)",
}
HEADER_KEYS = {
    "dtype", "entries", "features", "format", "header_bytes", "packet_bytes",
    "semantics", "semantics_sha256", "shared_table_count", "table_bytes",
    "table_offset", "table_sha256", "version",
}


class PacketError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def strict_json(raw: bytes) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in rows:
            if key in out:
                raise PacketError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def finite(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise PacketError("nonfinite JSON number")
        return result

    def bad_constant(value: str) -> None:
        raise PacketError(f"nonfinite JSON constant: {value}")

    return json.loads(raw.decode("ascii"), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=bad_constant)


def semantics_sha256() -> str:
    return hashlib.sha256(canonical_json(SEMANTICS)).hexdigest()


def _pack_table(values: Iterable[float]) -> bytes:
    rows = tuple(float(value) for value in values)
    if len(rows) != ENTRIES:
        raise PacketError(f"expected {ENTRIES} entries, got {len(rows)}")
    if not all(math.isfinite(value) for value in rows):
        raise PacketError("nonfinite pre-FP16 table entry")
    try:
        payload = struct.pack("<" + "e" * ENTRIES, *rows)
    except (OverflowError, struct.error) as exc:
        raise PacketError(f"FP16 table conversion failed: {exc}") from exc
    decoded = struct.unpack("<" + "e" * ENTRIES, payload)
    if not all(math.isfinite(value) for value in decoded):
        raise PacketError("nonfinite post-FP16 table entry")
    return payload


def build_packet(values: Iterable[float]) -> bytes:
    table = _pack_table(values)
    header = {
        "dtype": "<f2-finite",
        "entries": ENTRIES,
        "features": FEATURES,
        "format": FORMAT,
        "header_bytes": HEADER_BYTES,
        "packet_bytes": PACKET_BYTES,
        "semantics": SEMANTICS,
        "semantics_sha256": semantics_sha256(),
        "shared_table_count": 1,
        "table_bytes": TABLE_BYTES,
        "table_offset": HEADER_BYTES,
        "table_sha256": hashlib.sha256(table).hexdigest(),
        "version": VERSION,
    }
    encoded = canonical_json(header) + b"\n"
    if len(encoded) > HEADER_BYTES:
        raise PacketError("header exceeds fixed 4096-byte region")
    packet = encoded + bytes(HEADER_BYTES - len(encoded)) + table
    parse_packet(packet)
    return packet


def parse_packet(packet: bytes) -> dict[str, Any]:
    if len(packet) != PACKET_BYTES:
        raise PacketError(f"packet length {len(packet)} != {PACKET_BYTES}")
    newline = packet.find(b"\n", 0, HEADER_BYTES)
    if newline <= 0:
        raise PacketError("missing header terminator")
    if packet[newline + 1:HEADER_BYTES] != bytes(HEADER_BYTES - newline - 1):
        raise PacketError("nonzero header padding")
    header = strict_json(packet[:newline])
    if not isinstance(header, dict) or set(header) != HEADER_KEYS:
        raise PacketError("header key set changed")
    expected = {
        "dtype": "<f2-finite", "entries": ENTRIES, "features": FEATURES,
        "format": FORMAT, "header_bytes": HEADER_BYTES, "packet_bytes": PACKET_BYTES,
        "semantics": SEMANTICS, "semantics_sha256": semantics_sha256(),
        "shared_table_count": 1, "table_bytes": TABLE_BYTES,
        "table_offset": HEADER_BYTES, "version": VERSION,
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise PacketError(f"header field changed: {key}")
    table = packet[HEADER_BYTES:]
    if hashlib.sha256(table).hexdigest() != header["table_sha256"]:
        raise PacketError("table hash mismatch")
    values = struct.unpack("<" + "e" * ENTRIES, table)
    if not all(math.isfinite(value) for value in values):
        raise PacketError("nonfinite decoded FP16 entry")
    return {"header": header, "header_json_bytes": newline,
            "table_bytes": table, "values": values}


def reference_scalar_index(role: int, decoded_row: Sequence[float], column: int,
                           matrix_rms: float) -> int:
    if role not in (0, 1, 2) or not decoded_row or not 0 <= column < len(decoded_row):
        raise ValueError("invalid role/row/column")
    row = tuple(float(value) for value in decoded_row)
    if not all(math.isfinite(value) for value in row) or not math.isfinite(matrix_rms):
        raise ValueError("nonfinite feature input")
    row_scale = max(math.sqrt(math.fsum(value * value for value in row) / len(row)), 1e-30)
    matrix_scale = max(float(matrix_rms), 1e-30)
    center = row[column]
    normalized = center / row_scale
    amplitude = max(0, min(31, math.floor((normalized + 4.0) * 4.0)))
    ratio = math.log2(row_scale / matrix_scale)
    row_class = sum(ratio > threshold for threshold in (-0.25, 0.0, 0.25))
    left = row[column - 1] if column > 0 else center
    right = row[column + 1] if column + 1 < len(row) else center
    left_state = (2 if left >= 0.0 else 0) | (1 if abs(left) > abs(center) else 0)
    right_state = (2 if right >= 0.0 else 0) | (1 if abs(right) > abs(center) else 0)
    result = ((((role * 4 + row_class) * 32 + amplitude) * 4 + left_state) * 4 + right_state)
    if not 0 <= result < ENTRIES:
        raise AssertionError("feature index out of range")
    return result
