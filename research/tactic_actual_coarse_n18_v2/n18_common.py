#!/usr/bin/env python3
"""Standard-library invariants for the universal UNIPOLAR-N18-307 v2 fork."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Iterable


DESIGN_SCHEMA = "tactic_actual_coarse_n18_source_design_v2"
SOURCE_PLAN_SCHEMA = "tactic_actual_coarse_n18_source_plan_v2"
MANIFEST_SCHEMA = "tactic_actual_coarse_n18_source_manifest_v2"
REVIEW_SCHEMA = "tactic_actual_coarse_n18_independent_review_v2"
ENVIRONMENT_SCHEMA = "tactic_actual_coarse_n18_runtime_environment_v2"

MAGIC = b"TACN18C2"
VERSION = 2
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
ETA_NUMERATOR = 1
ETA_DENOMINATOR = 4
ROLES = ("gate", "up", "down_transposed")
ROLE_TO_ORDINAL = {role: ordinal for ordinal, role in enumerate(ROLES)}
FLAGS = 0x000F
ALGORITHM_ID = hashlib.sha256(b"UNIPOLAR-N18-307-v2-exact-packet").digest()[:16]
SC_DOMAIN = b"UNIPOLAR-N18-307-SC-v2\0"
RHT_DOMAIN = b"UNIPOLAR-N18-307-RHT-v2\0"
MAX_EXPERTS = 256
MAX_DIMENSION = 1 << 24
MAX_MATRIX_VALUES = 1 << 34
MAX_STREAMS_PER_MATRIX = 1 << 16
MAX_PLAN_BYTES = 16 << 20
MAX_REVIEW_BYTES = 1 << 20
MAX_ENVIRONMENT_LOCK_BYTES = 1 << 20

SYNTHETIC_AUTHORIZATION = "SYNTHETIC_ONLY_UNIPOLAR_N18_307_V2"
PILOT_AUTHORIZATION = "OPEN_AUTHENTICATED_UNIPOLAR_N18_307_PILOT_V2"
FULL_AUTHORIZATION = "OPEN_AUTHENTICATED_UNIPOLAR_N18_307_FULL_V2"


class ContractError(RuntimeError):
    """A finite format or source-closure invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_walk(value: Any) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite JSON number")
    elif isinstance(value, list):
        for child in value:
            _finite_walk(child)
    elif isinstance(value, dict):
        for child in value.values():
            _finite_walk(child)


def strict_json_loads(raw: bytes | str) -> Any:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError) as exc:
        raise ContractError(f"invalid strict JSON: {exc}") from exc
    _finite_walk(value)
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return value == value.lower()


def checked_positive_int(value: Any, maximum: int, label: str) -> int:
    require(type(value) is int and 0 < value <= maximum, label)
    return value


def checked_nonnegative_int(value: Any, maximum: int, label: str) -> int:
    require(type(value) is int and 0 <= value <= maximum, label)
    return value


def _checked_product(left: int, right: int, maximum: int, label: str) -> int:
    require(left <= maximum // right, label)
    return left * right


@dataclass(frozen=True)
class MatrixGeometry:
    """One canonical role matrix after Down has been transposed."""

    role: str
    rows: int
    columns: int

    def __post_init__(self) -> None:
        require(self.role in ROLE_TO_ORDINAL, "canonical matrix role")
        checked_positive_int(self.rows, MAX_DIMENSION, "matrix rows")
        checked_positive_int(self.columns, MAX_DIMENSION, "matrix columns")
        _checked_product(self.rows, self.columns, MAX_MATRIX_VALUES, "matrix values")
        require(self.streams <= MAX_STREAMS_PER_MATRIX, "matrix stream cap")

    @property
    def values(self) -> int:
        return self.rows * self.columns

    @property
    def streams(self) -> int:
        return (self.values + N - 1) // N

    def valid_values(self, tile_ordinal: int) -> int:
        checked_nonnegative_int(tile_ordinal, self.streams - 1, "tile ordinal")
        return min(N, self.values - tile_ordinal * N)

    @property
    def reservoir_bytes(self) -> int:
        return self.streams * RESERVOIR_BYTES


@dataclass(frozen=True)
class ExpertGeometry:
    """A shape-consistent Gate/Up/Down SwiGLU expert."""

    intermediate: int
    hidden: int

    def __post_init__(self) -> None:
        checked_positive_int(self.intermediate, MAX_DIMENSION, "intermediate dimension")
        checked_positive_int(self.hidden, MAX_DIMENSION, "hidden dimension")
        _checked_product(
            self.intermediate,
            self.hidden,
            MAX_MATRIX_VALUES,
            "expert role matrix values",
        )

    @property
    def matrices(self) -> tuple[MatrixGeometry, ...]:
        return tuple(
            MatrixGeometry(role, self.intermediate, self.hidden) for role in ROLES
        )

    @property
    def values(self) -> int:
        return 3 * self.intermediate * self.hidden

    @property
    def reservoir_bytes(self) -> int:
        return sum(matrix.reservoir_bytes for matrix in self.matrices)


def panel_ledger(experts: Iterable[ExpertGeometry]) -> dict[str, Any]:
    rows = tuple(experts)
    require(1 <= len(rows) <= MAX_EXPERTS, "expert count")
    total_values = sum(row.values for row in rows)
    total_bytes = sum(row.reservoir_bytes for row in rows)
    require(total_values > 0 and total_bytes > 0, "panel totals")
    owner_rows = []
    for ordinal, expert in enumerate(rows):
        proportional_share_num = total_bytes * expert.values
        proportional_share_den = total_values
        amplification_num = expert.reservoir_bytes * proportional_share_den
        amplification_den = proportional_share_num
        owner_rows.append(
            {
                "expert_ordinal": ordinal,
                "values": expert.values,
                "reservoir_bytes": expert.reservoir_bytes,
                "proportional_share_numerator": proportional_share_num,
                "proportional_share_denominator": proportional_share_den,
                "cold_amplification_numerator": amplification_num,
                "cold_amplification_denominator": amplification_den,
                "cold_amplification": amplification_num / amplification_den,
            }
        )
    return {
        "experts": len(rows),
        "values": total_values,
        "reservoir_bytes": total_bytes,
        "physical_bpw": 8 * total_bytes / total_values,
        "owners": owner_rows,
    }


def seed_pair(role: int, rows: int, columns: int, tile_ordinal: int) -> tuple[int, int]:
    checked_nonnegative_int(role, len(ROLES) - 1, "seed role")
    checked_positive_int(rows, MAX_DIMENSION, "seed rows")
    checked_positive_int(columns, MAX_DIMENSION, "seed columns")
    checked_nonnegative_int(tile_ordinal, MAX_STREAMS_PER_MATRIX - 1, "seed tile")
    suffix = struct.pack("<BIII", role, rows, columns, tile_ordinal)
    sc = int.from_bytes(hashlib.sha256(SC_DOMAIN + suffix).digest()[:4], "little") or 1
    rht = int.from_bytes(hashlib.sha256(RHT_DOMAIN + suffix).digest()[:8], "little")
    return sc, rht


def bits_to_payload(bits: Iterable[int]) -> tuple[bytes, int]:
    output = bytearray()
    current = 0
    count = 0
    for value in bits:
        require(type(value) is int and value in (0, 1), "binary logical symbol")
        current = (current << 1) | value
        count += 1
        require(count <= PAYLOAD_BITS, "logical payload overflow")
        if count % 8 == 0:
            output.append(current)
            current = 0
    if count % 8:
        output.append(current << (8 - count % 8))
    return bytes(output), count


class LogicalBitReader:
    """Hard-EOF, MSB-first reader; zero extension is forbidden."""

    def __init__(self, payload: bytes, logical_bits: int) -> None:
        require(type(payload) is bytes, "logical payload bytes")
        checked_nonnegative_int(logical_bits, PAYLOAD_BITS, "logical bit count")
        require(len(payload) == (logical_bits + 7) // 8, "logical payload length")
        if logical_bits % 8 and payload:
            require(
                payload[-1] & ((1 << (8 - logical_bits % 8)) - 1) == 0,
                "nonzero terminal padding bits",
            )
        self.payload = payload
        self.logical_bits = logical_bits
        self.cursor = 0

    def read_bit(self) -> int:
        require(self.cursor < self.logical_bits, "logical bitstream hard EOF")
        value = (self.payload[self.cursor // 8] >> (7 - self.cursor % 8)) & 1
        self.cursor += 1
        return value

    def finish(self) -> None:
        require(self.cursor == self.logical_bits, "logical bitstream not exactly exhausted")


def _stored_fp32(value: float) -> float:
    require(isinstance(value, (float, int)) and math.isfinite(float(value)), "finite scale")
    try:
        stored = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise ContractError("FP32 scale range") from exc
    require(math.isfinite(stored) and stored > 0.0, "positive finite stored FP32 scale")
    return stored


def _header(
    *,
    geometry: MatrixGeometry,
    tile_ordinal: int,
    scale: float,
    logical_bits: int,
    payload_sha256: str,
) -> bytes:
    checked_nonnegative_int(tile_ordinal, geometry.streams - 1, "header tile")
    checked_nonnegative_int(logical_bits, PAYLOAD_BITS, "header logical bits")
    require(is_sha256(payload_sha256), "payload digest")
    role = ROLE_TO_ORDINAL[geometry.role]
    sc_seed, rht_seed = seed_pair(role, geometry.rows, geometry.columns, tile_ordinal)
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<8sHHI", header, 0, MAGIC, VERSION, HEADER_BYTES, N)
    struct.pack_into("<BBH", header, 16, PROFILE_Q, role, FLAGS)
    struct.pack_into(
        "<IIII",
        header,
        20,
        geometry.rows,
        geometry.columns,
        tile_ordinal,
        geometry.valid_values(tile_ordinal),
    )
    struct.pack_into("<IQfI", header, 36, sc_seed, rht_seed, _stored_fp32(scale), logical_bits)
    header[56:88] = bytes.fromhex(payload_sha256)
    header[88:104] = ALGORITHM_ID
    struct.pack_into("<I", header, 124, zlib.crc32(header[:124]) & 0xFFFFFFFF)
    return bytes(header)


def pack_reservoir(
    payload: bytes,
    logical_bits: int,
    scale: float,
    geometry: MatrixGeometry,
    tile_ordinal: int,
) -> bytes:
    require(type(payload) is bytes, "payload bytes")
    checked_nonnegative_int(logical_bits, PAYLOAD_BITS, "logical bits")
    require(len(payload) == (logical_bits + 7) // 8, "logical payload byte count")
    reader = LogicalBitReader(payload, logical_bits)
    while reader.cursor < logical_bits:
        reader.read_bit()
    reader.finish()
    header = _header(
        geometry=geometry,
        tile_ordinal=tile_ordinal,
        scale=scale,
        logical_bits=logical_bits,
        payload_sha256=sha256_bytes(payload),
    )
    packet = header + payload + bytes(PAYLOAD_BYTES - len(payload))
    require(len(packet) == RESERVOIR_BYTES, "fixed reservoir bytes")
    return packet


def parse_reservoir(packet: bytes) -> dict[str, Any]:
    require(type(packet) is bytes and len(packet) == RESERVOIR_BYTES, "reservoir byte count")
    magic, version, header_bytes, n = struct.unpack_from("<8sHHI", packet, 0)
    require((magic, version, header_bytes, n) == (MAGIC, VERSION, HEADER_BYTES, N), "header constants")
    profile_q, role, flags = struct.unpack_from("<BBH", packet, 16)
    require(profile_q == PROFILE_Q and flags == FLAGS, "profile/flags")
    require(0 <= role < len(ROLES), "role ordinal")
    rows, columns, tile_ordinal, valid_values = struct.unpack_from("<IIII", packet, 20)
    geometry = MatrixGeometry(ROLES[role], rows, columns)
    require(tile_ordinal < geometry.streams, "tile ordinal")
    require(valid_values == geometry.valid_values(tile_ordinal), "valid-values field")
    sc_seed, rht_seed, scale, logical_bits = struct.unpack_from("<IQfI", packet, 36)
    require((sc_seed, rht_seed) == seed_pair(role, rows, columns, tile_ordinal), "universal seeds")
    require(math.isfinite(scale) and scale > 0.0, "stored scale")
    require(logical_bits <= PAYLOAD_BITS, "logical length")
    require(packet[88:104] == ALGORITHM_ID, "algorithm identifier")
    require(packet[104:124] == bytes(20), "reserved header bytes")
    (crc,) = struct.unpack_from("<I", packet, 124)
    require(crc == zlib.crc32(packet[:124]) & 0xFFFFFFFF, "header CRC")
    used = (logical_bits + 7) // 8
    payload = packet[HEADER_BYTES : HEADER_BYTES + used]
    require(sha256_bytes(payload) == packet[56:88].hex(), "payload digest")
    reader = LogicalBitReader(payload, logical_bits)
    require(packet[HEADER_BYTES + used :] == bytes(PAYLOAD_BYTES - used), "nonzero reservoir fill")
    return {
        "geometry": geometry,
        "role_ordinal": role,
        "tile_ordinal": tile_ordinal,
        "valid_values": valid_values,
        "sc_seed_u32": sc_seed,
        "rht_seed_u64": rht_seed,
        "decoder_scale_fp32": float(scale),
        "logical_bits": logical_bits,
        "payload": payload,
        "reader": reader,
    }


def canonical_bit_reencode(packet: bytes, decoded_bits: Iterable[int]) -> bytes:
    """Require exact hard-EOF consumption and byte-identical bit re-encoding."""
    parsed = parse_reservoir(packet)
    reader: LogicalBitReader = parsed["reader"]
    observed_payload_bits: list[int] = []
    while reader.cursor < reader.logical_bits:
        observed_payload_bits.append(reader.read_bit())
    reader.finish()
    observed_payload, observed_bits = bits_to_payload(observed_payload_bits)
    require(
        observed_bits == parsed["logical_bits"]
        and observed_payload == parsed["payload"],
        "parsed logical language",
    )
    # bits_to_payload enforces the cap while consuming, so a hostile/infinite
    # iterable cannot first create an unbounded list.
    payload, logical_bits = bits_to_payload(decoded_bits)
    require(logical_bits == parsed["logical_bits"] and payload == parsed["payload"], "canonical bit re-encode")
    return pack_reservoir(
        payload,
        logical_bits,
        parsed["decoder_scale_fp32"],
        parsed["geometry"],
        parsed["tile_ordinal"],
    )


def validate_fixed_ledger() -> dict[str, Any]:
    require(PAYLOAD_BITS == 627_712, "payload capacity")
    require(NOMINAL_BITS == 626_688, "nominal test-channel bits")
    require(PAYLOAD_BITS - NOMINAL_BITS == 1_024, "logical reserve")
    require(8 * RESERVOIR_BYTES * PROFILE_RATE_DENOMINATOR == N * PROFILE_RATE_NUMERATOR, "physical 307/128 rate")
    qwen = ExpertGeometry(768, 2048)
    require(qwen.values == 4_718_592, "Qwen triplet values")
    require(qwen.reservoir_bytes == 1_414_656, "Qwen triplet bytes")
    panel = panel_ledger([qwen] * 6)
    require(panel["values"] == 28_311_552 and panel["reservoir_bytes"] == 8_487_936, "Qwen evaluation ledger")
    require(panel["physical_bpw"] == PROFILE_RATE_NUMERATOR / PROFILE_RATE_DENOMINATOR, "Qwen exact coarse rate")
    require(all(row["cold_amplification_numerator"] == row["cold_amplification_denominator"] for row in panel["owners"]), "equal-geometry 1x read")
    return panel


validate_fixed_ledger()
