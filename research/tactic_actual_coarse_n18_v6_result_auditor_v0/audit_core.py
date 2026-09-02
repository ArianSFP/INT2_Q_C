#!/usr/bin/env python3
"""Independent byte grammar and CPU decoder for N18-v6 result audits.

The module is import inert and imports no numerical package.  NumPy and the
separately authenticated polar/arithmetic implementation are injected only
after the result auditor has retained every source, input, smoke, and result
byte.  No v4/v6 packet, decoder, encoder, receipt, or scoring helper is
imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence


MAGIC = b"TACN18C4"
VERSION = 4
HEADER_BYTES = 128
RESERVOIR_BYTES = 78_592
PAYLOAD_BYTES = RESERVOIR_BYTES - HEADER_BYTES
PAYLOAD_BITS = PAYLOAD_BYTES * 8
N = 1 << 18
SQRT_N = 512
PROFILE_Q = 164
ETA = 0.25
ROLES = ("gate", "up", "down_transposed")
SOURCE_ORDER = 1
PAGE_BYTES = 4096

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
ALGORITHM_ID = hashlib.sha256(
    b"UNIPOLAR-N18-307-v4-Q31-BEC-MAP-SC-RHT-FP32"
).digest()[:16]
SC_DOMAIN = b"UNIPOLAR-N18-307-SC-v4\0"
RHT_DOMAIN = b"UNIPOLAR-N18-307-RHT-v4\0"


class AuditError(RuntimeError):
    """A fail-closed result-audit contract failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(payload: bytes) -> str:
    require(type(payload) is bytes, "SHA-256 bytes")
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise AuditError(f"canonical JSON: {error}") from error


def pretty_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise AuditError(f"pretty JSON: {error}") from error


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in rows:
            require(key not in output, f"{label}: duplicate JSON key")
            output[key] = value
        return output

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AuditError(f"{label}: nonfinite {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label}: JSON: {error}") from error
    require(isinstance(value, dict), f"{label}: JSON object")
    _walk_json(value, label, 0)
    return value


def _walk_json(value: Any, label: str, depth: int) -> None:
    require(depth <= 64, f"{label}: JSON depth")
    if value is None or isinstance(value, (bool, str)) or type(value) is int:
        return
    if type(value) is float:
        require(math.isfinite(value), f"{label}: finite JSON float")
        return
    if isinstance(value, list):
        for item in value:
            _walk_json(item, label, depth + 1)
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{label}: string keys")
        for item in value.values():
            _walk_json(item, label, depth + 1)
        return
    raise AuditError(f"{label}: unsupported JSON node")


def digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label}: lowercase SHA-256",
    )
    return value


def exact_int(
    value: Any, label: str, minimum: int = 0, maximum: int = (1 << 63) - 1
) -> int:
    require(type(value) is int and minimum <= value <= maximum, f"{label}: integer")
    return value


def finite_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    require(type(value) in (int, float), f"{label}: number")
    output = float(value)
    require(math.isfinite(output), f"{label}: finite")
    if minimum is not None:
        require(output >= minimum, f"{label}: minimum")
    return output


def require_float_close(
    observed: Any,
    expected: Any,
    label: str,
    *,
    rel: float = 2.0 ** -42,
    abs_: float = 1e-13,
) -> None:
    left = finite_float(observed, f"{label} observed")
    right = finite_float(expected, f"{label} expected")
    require(math.isclose(left, right, rel_tol=rel, abs_tol=abs_), label)


def require_deep_close(observed: Any, expected: Any, label: str) -> None:
    if expected is None or isinstance(expected, (bool, str)):
        require(type(observed) is type(expected) and observed == expected, label)
    elif type(expected) is int:
        require(type(observed) is int and observed == expected, label)
    elif type(expected) is float:
        require_float_close(observed, expected, label)
    elif isinstance(expected, Mapping):
        require(
            isinstance(observed, Mapping) and set(observed) == set(expected),
            f"{label}: mapping fields",
        )
        for key in sorted(expected):
            require_deep_close(observed[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        require(
            isinstance(observed, (list, tuple)) and len(observed) == len(expected),
            f"{label}: sequence geometry",
        )
        for ordinal, (left, right) in enumerate(zip(observed, expected, strict=True)):
            require_deep_close(left, right, f"{label}[{ordinal}]")
    else:
        require(observed == expected, label)


@dataclass(frozen=True)
class Geometry:
    intermediate: int
    hidden: int

    def __post_init__(self) -> None:
        require(type(self.intermediate) is int and 0 < self.intermediate <= 1 << 24, "intermediate")
        require(type(self.hidden) is int and 0 < self.hidden <= 1 << 24, "hidden")
        require(self.intermediate * self.hidden <= 1 << 34, "geometry product")

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
        return len(ROLES) * self.streams_per_role

    @property
    def frame_bytes(self) -> int:
        return self.records * RESERVOIR_BYTES

    @property
    def target_eligible(self) -> bool:
        return self.role_values % N == 0

    def valid_values(self, tile_ordinal: int) -> int:
        require(0 <= tile_ordinal < self.streams_per_role, "tile ordinal")
        return min(N, self.role_values - tile_ordinal * N)


@dataclass(frozen=True)
class ParsedReservoir:
    geometry: Geometry
    role_ordinal: int
    tile_ordinal: int
    valid_values: int
    zero_tile: bool
    padded_tail: bool
    sc_seed_u32: int
    rht_seed_u64: int
    scale_fp32: float
    logical_bits: int
    payload: bytes


def seed_pair(
    role_ordinal: int, intermediate: int, hidden: int, tile_ordinal: int
) -> tuple[int, int]:
    require(0 <= role_ordinal < len(ROLES), "seed role")
    geometry = Geometry(intermediate, hidden)
    require(0 <= tile_ordinal < geometry.streams_per_role, "seed tile")
    suffix = struct.pack("<BIII", role_ordinal, intermediate, hidden, tile_ordinal)
    sc_seed = int.from_bytes(hashlib.sha256(SC_DOMAIN + suffix).digest()[:4], "little") or 1
    rht_seed = int.from_bytes(hashlib.sha256(RHT_DOMAIN + suffix).digest()[:8], "little")
    return sc_seed, rht_seed


def payload_bits(payload: bytes, logical_bits: int) -> tuple[int, ...]:
    require(type(payload) is bytes and 0 <= logical_bits <= PAYLOAD_BITS, "payload geometry")
    require(len(payload) == (logical_bits + 7) // 8, "payload byte length")
    if logical_bits % 8 and payload:
        require(
            payload[-1] & ((1 << (8 - logical_bits % 8)) - 1) == 0,
            "terminal logical padding",
        )
    return tuple(
        (payload[index // 8] >> (7 - index % 8)) & 1
        for index in range(logical_bits)
    )


def _stored_fp32(value: float) -> float:
    try:
        stored = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error) as error:
        raise AuditError("scale FP32") from error
    require(math.isfinite(stored) and stored > 0.0, "positive scale FP32")
    return stored


def build_header(parsed: ParsedReservoir) -> bytes:
    flags = BASE_FLAGS
    if parsed.zero_tile:
        flags |= FLAG_ZERO_TILE
    if parsed.padded_tail:
        flags |= FLAG_PADDED_TAIL
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<8sHHI", header, 0, MAGIC, VERSION, HEADER_BYTES, N)
    struct.pack_into("<BBH", header, 16, PROFILE_Q, parsed.role_ordinal, flags)
    struct.pack_into(
        "<IIII",
        header,
        20,
        parsed.geometry.intermediate,
        parsed.geometry.hidden,
        parsed.tile_ordinal,
        parsed.valid_values,
    )
    struct.pack_into(
        "<IQfI",
        header,
        36,
        parsed.sc_seed_u32,
        parsed.rht_seed_u64,
        _stored_fp32(parsed.scale_fp32),
        parsed.logical_bits,
    )
    header[56:88] = hashlib.sha256(parsed.payload).digest()
    header[88:104] = ALGORITHM_ID
    struct.pack_into(
        "<IIII",
        header,
        104,
        N - parsed.valid_values,
        int(parsed.padded_tail),
        SOURCE_ORDER,
        0,
    )
    struct.pack_into("<I", header, 120, zlib.crc32(parsed.payload) & 0xFFFFFFFF)
    struct.pack_into("<I", header, 124, zlib.crc32(header[:124]) & 0xFFFFFFFF)
    return bytes(header)


def canonical_packet(parsed: ParsedReservoir) -> bytes:
    payload_bits(parsed.payload, parsed.logical_bits)
    header = build_header(parsed)
    return header + parsed.payload + bytes(PAYLOAD_BYTES - len(parsed.payload))


def parse_reservoir(packet: bytes) -> ParsedReservoir:
    require(type(packet) is bytes and len(packet) == RESERVOIR_BYTES, "reservoir bytes")
    magic, version, header_bytes, n = struct.unpack_from("<8sHHI", packet, 0)
    require((magic, version, header_bytes, n) == (MAGIC, VERSION, HEADER_BYTES, N), "header constants")
    profile_q, role_ordinal, flags = struct.unpack_from("<BBH", packet, 16)
    require(profile_q == PROFILE_Q and 0 <= role_ordinal < len(ROLES), "profile/role")
    intermediate, hidden, tile_ordinal, valid_values = struct.unpack_from("<IIII", packet, 20)
    geometry = Geometry(intermediate, hidden)
    require(tile_ordinal < geometry.streams_per_role, "packet tile")
    require(valid_values == geometry.valid_values(tile_ordinal), "packet valid values")
    padded = valid_values != N
    zero = bool(flags & FLAG_ZERO_TILE)
    expected_flags = BASE_FLAGS | (FLAG_ZERO_TILE if zero else 0) | (FLAG_PADDED_TAIL if padded else 0)
    require(flags == expected_flags, "packet flags")
    sc_seed, rht_seed, scale, logical_bits = struct.unpack_from("<IQfI", packet, 36)
    require((sc_seed, rht_seed) == seed_pair(role_ordinal, intermediate, hidden, tile_ordinal), "packet seeds")
    require(math.isfinite(scale) and scale > 0.0 and logical_bits <= PAYLOAD_BITS, "packet scale/bits")
    if zero:
        require(logical_bits == 0 and scale == 1.0, "zero packet fields")
    else:
        require(logical_bits > 0, "nonzero packet bits")
    require(packet[88:104] == ALGORITHM_ID, "algorithm identifier")
    padding, tail_marker, source_order, reserved = struct.unpack_from("<IIII", packet, 104)
    require((padding, tail_marker, source_order, reserved) == (N - valid_values, int(padded), SOURCE_ORDER, 0), "source/tail fields")
    require(struct.unpack_from("<I", packet, 124)[0] == zlib.crc32(packet[:124]) & 0xFFFFFFFF, "header CRC")
    used = (logical_bits + 7) // 8
    payload = packet[HEADER_BYTES : HEADER_BYTES + used]
    require(hashlib.sha256(payload).digest() == packet[56:88], "payload SHA-256")
    require(struct.unpack_from("<I", packet, 120)[0] == zlib.crc32(payload) & 0xFFFFFFFF, "payload CRC")
    payload_bits(payload, logical_bits)
    require(packet[HEADER_BYTES + used :] == bytes(PAYLOAD_BYTES - used), "reservoir zero fill")
    parsed = ParsedReservoir(
        geometry,
        role_ordinal,
        tile_ordinal,
        valid_values,
        zero,
        padded,
        sc_seed,
        rht_seed,
        float(scale),
        logical_bits,
        payload,
    )
    require(canonical_packet(parsed) == packet, "packet canonical rebuild")
    return parsed


def make_zero_packet(geometry: Geometry, role_ordinal: int, tile_ordinal: int) -> bytes:
    valid = geometry.valid_values(tile_ordinal)
    seeds = seed_pair(role_ordinal, geometry.intermediate, geometry.hidden, tile_ordinal)
    return canonical_packet(
        ParsedReservoir(
            geometry,
            role_ordinal,
            tile_ordinal,
            valid,
            True,
            valid != N,
            seeds[0],
            seeds[1],
            1.0,
            0,
            b"",
        )
    )


def i32_inverse_contract_max(index: int = 63) -> int:
    require(type(index) is int and 0 <= index <= 63, "inverse index")
    return abs(index - 31) * N


def inverse_symbols_i32(np: Any, indices: Any, rht_seed: int) -> Any:
    values = np.asarray(indices).astype(np.int32) - np.int32(31)
    require(values.shape == (N,), "inverse index geometry")
    width = 1
    while width < N:
        view = values.reshape(-1, 2, width)
        left = view[:, 0, :].copy()
        right = view[:, 1, :].copy()
        view[:, 0, :] = left + right
        view[:, 1, :] = left - right
        width *= 2
    with np.errstate(over="ignore"):
        z = np.arange(N, dtype=np.uint64) + np.uint64(rht_seed)
        z += np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    signs = np.where((z & np.uint64(1)) == 0, 1, -1).astype(np.int32)
    values *= signs
    maximum = int(np.max(np.abs(values.astype(np.int64))))
    require(maximum <= 32 * N and 32 * N < 2**31, "I32 inverse bound")
    output = values.astype("<i4", copy=False)
    require(output.dtype.str == "<i4", "I32 inverse dtype")
    return output


def decode_reservoir(packet: bytes, np: Any, decoder: Any) -> dict[str, Any]:
    parsed = parse_reservoir(packet)
    if parsed.zero_tile:
        symbols = np.zeros(N, dtype="<i4")
        reconstruction = np.zeros(parsed.valid_values, dtype="<f4")
        selected_bits = b""
        levels = []
        inverse_difference = 0.0
    else:
        profile = decoder.profile_parameters(PROFILE_Q, ETA)
        require(float(profile["rate_bpw"]) == 1.75 + PROFILE_Q / 256.0, "profile rate")
        reverse = decoder.bit_reverse_indices(N)
        layers = decoder.sc_layers(N)
        flags = decoder.bec_freeze_flags(N, profile["capacities"], reverse)
        require(len(flags) == 6 and all(row.shape == (N,) for row in flags), "freeze flags")
        arithmetic = decoder.ArithmeticBinaryDecoder(parsed.payload, 0, parsed.logical_bits)
        alphabet = ETA * np.arange(-31, 33, dtype=np.float64)
        weights = np.exp(-0.5 * (alphabet / float(profile["sigma_reconstruction"])) ** 2)
        previous = np.zeros(N, dtype=np.int16)
        selected_rows = []
        frequency_rows = []
        levels = []
        for level_index, flag in enumerate(flags):
            level = level_index + 1
            frozen_rng = np.random.default_rng(parsed.sc_seed_u32 + 1_000_003 * level)
            frozen_external = frozen_rng.integers(0, 2, size=N, dtype=np.uint8)
            prior = decoder.leaf_prior_ratios(weights, previous, level)
            x_bit, frequencies, selected = decoder.decode_sc_level(
                prior, flag, frozen_external, reverse, layers, arithmetic
            )
            previous += (1 << level_index) * x_bit.astype(np.int16)
            selected_rows.append(selected)
            frequency_rows.append(frequencies)
            levels.append(
                {
                    "level": level,
                    "selected": int(frequencies.size),
                    "capacity": float(profile["capacities"][level_index]),
                }
            )
        selected_all = np.concatenate(selected_rows)
        frequency_all = np.concatenate(frequency_rows)
        canonical_payload, canonical_bits = decoder.arithmetic_encode_binary(selected_all, frequency_all)
        require(canonical_bits == parsed.logical_bits and canonical_payload == parsed.payload, "arithmetic canonical rebuild")
        selected_bits = selected_all.astype(np.uint8, copy=False).tobytes()
        symbols = inverse_symbols_i32(np, previous, parsed.rht_seed_u64)
        reconstruction64 = symbols.astype(np.float64) * (ETA * parsed.scale_fp32 / SQRT_N)
        floating_reference = decoder.inverse_signed_rht(
            alphabet[previous] * parsed.scale_fp32, parsed.rht_seed_u64, "numpy"
        )
        inverse_difference = float(np.max(np.abs(reconstruction64 - floating_reference)))
        require(inverse_difference <= 2e-14, "integer/float inverse parity")
        reconstruction = reconstruction64[: parsed.valid_values].astype("<f4")
    require(canonical_packet(parsed) == packet, "literal canonical packet reencode")
    return {
        "parsed": parsed,
        "symbols": symbols,
        "reconstruction": reconstruction,
        "report": {
            "role_ordinal": parsed.role_ordinal,
            "tile_ordinal": parsed.tile_ordinal,
            "zero_tile": parsed.zero_tile,
            "logical_bits": parsed.logical_bits,
            "valid_values": parsed.valid_values,
            "padded_values": N - parsed.valid_values,
            "canonical_reencode_matches": True,
            "packet_sha256": sha256(packet),
            "inverse_transient_dtype": "<i4",
            "inverse_i32_dtype_verified": True,
            "canonical_symbol_abs_max": int(np.max(np.abs(symbols.astype(np.int64)))),
            "canonical_symbols_i32_sha256": sha256(symbols.astype("<i4", copy=False).tobytes()),
            "reconstruction_f32_sha256": sha256(reconstruction.astype("<f4", copy=False).tobytes()),
            "selected_decisions_sha256": sha256(selected_bits),
            "integer_float_inverse_max_abs": inverse_difference,
            "levels": levels,
        },
    }


def bf16_f64(np: Any, raw: bytes, expected_values: int) -> Any:
    require(type(raw) is bytes and len(raw) == 2 * expected_values, "BF16 source bytes")
    words = np.frombuffer(raw, dtype="<u2")
    require(bool(np.all((words & np.uint16(0x7F80)) != np.uint16(0x7F80))), "finite BF16 source")
    output = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
    require(output.shape == (expected_values,), "BF16 source geometry")
    return output


def exact_rate(frame_bytes: int, weights: int) -> dict[str, Any]:
    value = Fraction(8 * frame_bytes, weights)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": f"{value.numerator}/{value.denominator}",
        "float": float(value),
        "equals_307_over_128": value == Fraction(307, 128),
    }


def traffic_ledgers(geometry: Geometry, *, start_offset_mod_page: int = 0) -> dict[str, Any]:
    require(0 <= start_offset_mod_page < PAGE_BYTES, "page offset")
    frame_bytes = geometry.frame_bytes
    aligned_pages = (frame_bytes + PAGE_BYTES - 1) // PAGE_BYTES
    offset_pages = (start_offset_mod_page + frame_bytes + PAGE_BYTES - 1) // PAGE_BYTES
    worst_pages = (PAGE_BYTES - 1 + frame_bytes + PAGE_BYTES - 1) // PAGE_BYTES
    canonical_symbols = geometry.records * N * 4
    reconstruction = geometry.values * 4
    return {
        "external_compressed_file_read_executed_by_auditor": {
            "passes": 1,
            "first_pass_bytes": frame_bytes,
            "total_read_bytes": frame_bytes,
            "reread_bytes": 0,
            "total_read_amplification": 1.0,
            "reread_amplification": 0.0,
            "this_is_audit_file_IO_not_inference_HBM": True,
        },
        "page_projection_not_an_executed_inference_trace": {
            "start_offset_mod_page": start_offset_mod_page,
            "unique_page_bytes_at_offset": offset_pages * PAGE_BYTES,
            "unique_page_amplification_at_offset": offset_pages * PAGE_BYTES / frame_bytes,
            "aligned_unique_page_bytes": aligned_pages * PAGE_BYTES,
            "aligned_unique_page_amplification": aligned_pages * PAGE_BYTES / frame_bytes,
            "worst_alignment_unique_page_bytes": worst_pages * PAGE_BYTES,
            "worst_alignment_unique_page_amplification": worst_pages * PAGE_BYTES / frame_bytes,
        },
        "host_memory_lower_bound": {
            "causal_decode_input_scan_bytes": frame_bytes,
            "aggregate_reencode_comparison_bytes": frame_bytes,
            "frame_hash_bytes": frame_bytes,
            "minimum_full_scan_equivalents": 3,
            "minimum_bytes_touched": 3 * frame_bytes,
            "cpu_cache_or_dram_traffic_measured": False,
        },
        "scratch_lower_bound": {
            "canonical_packet_buffers_bytes": frame_bytes,
            "aggregate_reencode_buffer_bytes": frame_bytes,
            "canonical_symbols_i32_bytes": canonical_symbols,
            "reconstruction_f32_bytes": reconstruction,
            "minimum_numeric_and_packet_scratch_bytes": 2 * frame_bytes + canonical_symbols + reconstruction,
            "python_and_decoder_internal_scratch_measured": False,
        },
        "accelerator_hbm": {
            "measured": False,
            "read_bytes": None,
            "read_amplification": None,
            "below_2x_claim_authority": False,
        },
        "inference_ready_routed_decoder_executed": False,
        "strict_below_2x_inference_HBM_authority": False,
    }


def decode_frame(
    frame: bytes,
    np: Any,
    decoder: Any,
    source_role_bf16: Mapping[str, bytes],
) -> dict[str, Any]:
    require(type(frame) is bytes and len(frame) >= 3 * RESERVOIR_BYTES, "frame lower bound")
    require(len(frame) % RESERVOIR_BYTES == 0, "frame alignment")
    first = parse_reservoir(frame[:RESERVOIR_BYTES])
    require((first.role_ordinal, first.tile_ordinal) == (0, 0), "frame first record")
    geometry = first.geometry
    require(len(frame) == geometry.frame_bytes, "frame geometry bytes")
    require(set(source_role_bf16) == set(ROLES), "source role set")
    decoded = []
    reencoded = bytearray()
    for ordinal in range(geometry.records):
        begin = ordinal * RESERVOIR_BYTES
        packet = frame[begin : begin + RESERVOIR_BYTES]
        row = decode_reservoir(packet, np, decoder)
        expected_role, expected_tile = divmod(ordinal, geometry.streams_per_role)
        parsed = row["parsed"]
        require(parsed.geometry == geometry, "frame geometry drift")
        require((parsed.role_ordinal, parsed.tile_ordinal) == (expected_role, expected_tile), "frame canonical order")
        decoded.append(row)
        reencoded.extend(canonical_packet(parsed))
    require(bytes(reencoded) == frame, "aggregate literal canonical reencode")

    role_rows = []
    total_sse = 0.0
    total_energy = 0.0
    reconstruction_hashes: dict[str, str] = {}
    symbol_hashes: dict[str, str] = {}
    for role_ordinal, role in enumerate(ROLES):
        rows = [row for row in decoded if row["parsed"].role_ordinal == role_ordinal]
        reconstruction = np.concatenate([row["reconstruction"] for row in rows]).astype("<f4", copy=False)
        require(reconstruction.shape == (geometry.role_values,), "role reconstruction geometry")
        source = bf16_f64(np, source_role_bf16[role], geometry.role_values)
        residual = source - reconstruction.astype(np.float64)
        sse = float(np.dot(residual, residual))
        energy = float(np.dot(source, source))
        require(math.isfinite(sse) and math.isfinite(energy) and sse >= 0.0 and energy >= 0.0, "score finite")
        require(energy > 0.0 or sse == 0.0, "zero-energy reconstruction")
        reconstruction_hashes[role] = sha256(reconstruction.tobytes())
        symbol_bytes = b"".join(row["symbols"].astype("<i4", copy=False).tobytes() for row in rows)
        symbol_hashes[role] = sha256(symbol_bytes)
        total_sse += sse
        total_energy += energy
        role_rows.append(
            {
                "role": role,
                "weights": geometry.role_values,
                "source_bf16_sha256": sha256(source_role_bf16[role]),
                "reconstruction_f32_sha256": reconstruction_hashes[role],
                "residual_f64_sha256": sha256(residual.astype("<f8", copy=False).tobytes()),
                "sse_fp64": sse,
                "source_energy_fp64": energy,
                "raw_mse_fp64": sse / geometry.role_values,
                "relative_mse": sse / energy if energy > 0.0 else None,
                "zero_source_energy": energy == 0.0,
            }
        )
    score = {
        "domain": "original canonical BF16 Gate/Up/DownT source coordinates",
        "roles": role_rows,
        "pooled_sse_fp64": total_sse,
        "pooled_source_energy_fp64": total_energy,
        "pooled_raw_mse_fp64": total_sse / geometry.values,
        "pooled_relative_mse": total_sse / total_energy if total_energy > 0.0 else None,
        "zero_pooled_source_energy": total_energy == 0.0,
    }
    rate = exact_rate(len(frame), geometry.values)
    f_value = score["pooled_relative_mse"] * 2.0 ** (2.0 * rate["float"]) if score["pooled_relative_mse"] is not None else None
    return {
        "geometry": {
            "intermediate": geometry.intermediate,
            "hidden": geometry.hidden,
            "weights": geometry.values,
            "role_values": geometry.role_values,
            "records": geometry.records,
            "streams_per_role": geometry.streams_per_role,
        },
        "target_eligible_exact_307_over_128": geometry.target_eligible,
        "frame_bytes": len(frame),
        "frame_sha256": sha256(frame),
        "literal_aggregate_reencode_matches": True,
        "aggregate_reencoded_frame_sha256": sha256(bytes(reencoded)),
        "inverse_transient_dtype": "<i4",
        "all_records_I32_verified": True,
        "rate": rate,
        "original_domain_score": score,
        "coarse_only_F_diagnostic": f_value,
        "reconstruction_f32_sha256": reconstruction_hashes,
        "canonical_symbols_i32_sha256": symbol_hashes,
        "records": [row["report"] for row in decoded],
        "traffic": traffic_ledgers(geometry),
    }


def verify_completion(
    complete: Mapping[str, Any], data_members: Mapping[str, Mapping[str, Any]], expected_status: str,
    *, source_root_sha256: str, smoke_sha256: str, frame_sha256: str,
) -> None:
    require(
        set(complete) == {
            "schema", "status", "positive_claim_authority", "source_root_sha256",
            "source_free_smoke_file_sha256", "frame_sha256", "members",
            "members_root_sha256", "completion_claim_sha256",
        },
        "completion exact fields",
    )
    require(complete["schema"] == "tactic-actual-coarse-n18-v6-completion-v1", "completion schema")
    require(complete["status"] == expected_status and complete["positive_claim_authority"] is False, "completion status/nonpromotion")
    require(complete["source_root_sha256"] == source_root_sha256, "completion source root")
    require(complete["source_free_smoke_file_sha256"] == smoke_sha256, "completion smoke")
    require(complete["frame_sha256"] == frame_sha256, "completion frame")
    rows = complete["members"]
    require(isinstance(rows, list) and len(rows) == len(data_members), "completion rows")
    expected_names = sorted(data_members, key=lambda value: value.encode("utf-8"))
    require([row.get("name") for row in rows] == expected_names, "completion canonical member order")
    clean_rows = []
    for row in rows:
        require(isinstance(row, Mapping) and set(row) == {"name", "bytes", "sha256"}, "completion member row")
        name = row["name"]
        observed = data_members[name]
        require(row["bytes"] == observed["bytes"] and row["sha256"] == observed["sha256"], f"completion member binding {name}")
        clean_rows.append(dict(row))
    require(complete["members_root_sha256"] == sha256(canonical_json(clean_rows)), "completion member root")
    clean = dict(complete)
    claimed = clean.pop("completion_claim_sha256")
    digest(claimed, "completion claim")
    require(claimed == sha256(canonical_json(clean)), "completion internal claim seal")


def validate_result_material(
    producer: Mapping[str, Any], recomputed: Mapping[str, Any], input_binding: Mapping[str, Any],
    *, expected_status: str, source_root_sha256: str, smoke_binding: Mapping[str, Any], runtime_receipt: Mapping[str, Any],
) -> None:
    required = {
        "schema", "status", "positive_claim_authority", "source_closure",
        "source_free_smoke_binding", "source_free_smoke_file_sha256",
        "input_manifest_sha256", "input_bindings", "identity_fields_available_to_codec",
        "runtime_closure", "frame_sha256", "frame_bytes", "physical_bpw_exact",
        "target_eligible_exact_307_over_128", "matches_qwen_pilot_shape_only",
        "qwen_or_model_identity_used_by_codec", "universal_arbitrary_shape_below_2_5_bpw_claim",
        "encoder_all_self_checks_pass", "literal_aggregate_reencode_matches",
        "actual_prebuffered_decode_traffic", "modeled_one_external_file_read_not_executed_here",
        "original_domain_score", "claim_boundary",
    }
    require(set(producer) == required, "RESULT exact fields")
    require(producer["schema"] == "tactic-actual-coarse-n18-v6-bound-result-v1", "RESULT schema")
    require(producer["status"] == expected_status and producer["positive_claim_authority"] is False, "RESULT status/nonpromotion")
    require(producer["source_closure"].get("source_root_sha256") == source_root_sha256, "RESULT source root")
    expected_result_smoke = dict(smoke_binding)
    smoke_file_sha256 = expected_result_smoke.pop("receipt_file_sha256")
    require_deep_close(producer["source_free_smoke_binding"], expected_result_smoke, "RESULT smoke binding")
    require(producer["source_free_smoke_file_sha256"] == smoke_file_sha256, "RESULT smoke file")
    require(producer["input_manifest_sha256"] == input_binding["manifest_sha256"], "RESULT input manifest")
    require_deep_close(producer["input_bindings"], input_binding["roles"], "RESULT input roles")
    require(producer["identity_fields_available_to_codec"] is False and producer["qwen_or_model_identity_used_by_codec"] is False, "RESULT identity-free")
    require_deep_close(producer["runtime_closure"], runtime_receipt, "RESULT runtime")
    require(producer["frame_sha256"] == recomputed["frame_sha256"] and producer["frame_bytes"] == recomputed["frame_bytes"], "RESULT frame")
    require_deep_close(producer["physical_bpw_exact"], recomputed["rate"], "RESULT rate")
    require(producer["target_eligible_exact_307_over_128"] is True, "RESULT target rate")
    require(producer["matches_qwen_pilot_shape_only"] is True, "RESULT Qwen shape only")
    require(producer["universal_arbitrary_shape_below_2_5_bpw_claim"] is False, "RESULT no universal tail claim")
    require(producer["encoder_all_self_checks_pass"] is True and producer["literal_aggregate_reencode_matches"] is True, "RESULT integrity counters")
    require(producer["actual_prebuffered_decode_traffic"]["accelerator_hbm"]["below_2x_claim_authority"] is False, "RESULT HBM nonclaim")
    one = producer["modeled_one_external_file_read_not_executed_here"]
    require(one.get("passes") == 1 and one.get("first_pass_bytes") == recomputed["frame_bytes"] and one.get("reread_bytes") == 0, "RESULT one-pass projection")
    # The producer did not include raw-MSE fields; compare the shared exact
    # domain without mutating the independent recomputation.
    expected_score = {
        key: value for key, value in recomputed["original_domain_score"].items()
        if key != "pooled_raw_mse_fp64"
    }
    expected_score["roles"] = [
        {key: value for key, value in role.items() if key not in {"weights", "raw_mse_fp64"}}
        for role in recomputed["original_domain_score"]["roles"]
    ]
    require_deep_close(producer["original_domain_score"], expected_score, "RESULT original score")


def validate_producer_receipts(
    encoder: Mapping[str, Any], decoder_receipt: Mapping[str, Any], recomputed: Mapping[str, Any],
) -> None:
    require(encoder.get("schema") == "tactic-actual-coarse-n18-v6-frame-encode-receipt-v1", "encoder receipt schema")
    require(encoder.get("all_encoder_self_checks_required_and_passed") is True, "encoder checks")
    require(encoder.get("frame_sha256") == recomputed["frame_sha256"] and encoder.get("frame_bytes") == recomputed["frame_bytes"], "encoder frame")
    require_deep_close(encoder.get("physical_bpw_exact"), recomputed["rate"], "encoder rate")
    require(encoder.get("records") == 18 and len(encoder.get("tiles", [])) == 18, "encoder record count")
    for ordinal, (claimed, actual) in enumerate(zip(encoder["tiles"], recomputed["records"], strict=True)):
        require(claimed.get("packet_sha256") == actual["packet_sha256"], f"encoder record {ordinal} packet hash")
        require(claimed.get("all_encoder_self_checks_required_and_passed") is True, f"encoder record {ordinal} self checks")
    require(decoder_receipt.get("schema") == "tactic-actual-coarse-n18-v6-frame-decode-receipt-v1", "decoder receipt schema")
    require(decoder_receipt.get("frame_sha256") == recomputed["frame_sha256"] and decoder_receipt.get("frame_bytes") == recomputed["frame_bytes"], "decoder frame")
    require(decoder_receipt.get("aggregate_reencoded_frame_sha256") == recomputed["frame_sha256"] and decoder_receipt.get("literal_aggregate_reencode_matches") is True, "decoder canonical frame")
    require(decoder_receipt.get("inverse_transient_dtype") == "<i4", "decoder I32")
    require_deep_close(decoder_receipt.get("canonical_symbols_i32_sha256"), recomputed["canonical_symbols_i32_sha256"], "decoder symbol hashes")
    expected_score = {
        key: value for key, value in recomputed["original_domain_score"].items()
        if key != "pooled_raw_mse_fp64"
    }
    expected_score["roles"] = [
        {key: value for key, value in role.items() if key not in {"weights", "raw_mse_fp64"}}
        for role in recomputed["original_domain_score"]["roles"]
    ]
    require_deep_close(decoder_receipt.get("original_domain_score"), expected_score, "decoder original score")
    require(len(decoder_receipt.get("records", [])) == 18, "decoder records")
    for ordinal, (claimed, actual) in enumerate(zip(decoder_receipt["records"], recomputed["records"], strict=True)):
        require(claimed.get("canonical_reencode_matches") is True, f"decoder record {ordinal} canonical")
        require(claimed.get("inverse_transient_dtype") == "<i4", f"decoder record {ordinal} I32")
        require(claimed.get("canonical_symbols_i32_sha256") == actual["canonical_symbols_i32_sha256"], f"decoder record {ordinal} symbol hash")
        require(claimed.get("reconstruction_f32_sha256") == actual["reconstruction_f32_sha256"], f"decoder record {ordinal} reconstruction hash")
        require(claimed.get("canonical_symbol_abs_max") == actual["canonical_symbol_abs_max"], f"decoder record {ordinal} abs max")


def qwen_geometry_gate(recomputed: Mapping[str, Any]) -> str:
    geometry = recomputed["geometry"]
    require(geometry == {
        "intermediate": 768, "hidden": 2048, "weights": 4_718_592,
        "role_values": 1_572_864, "records": 18, "streams_per_role": 6,
    }, "exact Qwen pilot geometry")
    require(recomputed["frame_bytes"] == 1_414_656, "exact Qwen coarse bytes")
    require(recomputed["rate"]["exact"] == "307/128" and recomputed["rate"]["equals_307_over_128"] is True, "exact 307/128")
    return "PASS_V6_BOUND_TARGET_ELIGIBLE_FRAME_NONPROMOTING_INDEPENDENT_RESULT_AUDIT_REQUIRED"
