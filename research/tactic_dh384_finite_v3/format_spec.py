#!/usr/bin/env python3
"""Source-independent finite grammar for TACTIC-DH384 v3.

The module is deliberately standard-library only.  It defines every legal
48-byte refinement record, the exact decoder-conditioned dyadic transform,
the scale law, the single-expert container layout, and CPU reference
encode/decode operations.  No source/model identity participates in the
codebook.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from fractions import Fraction
from typing import Any, Iterable, Sequence


class FormatError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FormatError(message)


ROLES = ("gate", "up", "down_transposed")
INTERMEDIATE = 768
HIDDEN = 2048
ROLE_VALUES = INTERMEDIATE * HIDDEN
EXPERT_VALUES = len(ROLES) * ROLE_VALUES
COARSE_TILE_VALUES = 1 << 18
COARSE_TILES_PER_ROLE = ROLE_VALUES // COARSE_TILE_VALUES
COARSE_RECORD_BYTES = 78_592
COARSE_BYTES = len(ROLES) * COARSE_TILES_PER_ROLE * COARSE_RECORD_BYTES

BLOCK_VALUES = 4096
STAGES = 12
AUDITED_PARENT_RANK = 384
# One byte is a charged scale index; the remaining 376 bits are signs for a
# strict subset of the independently audited rank-384 conditional span.
ACTIVE_RANK = 376
FINE_RECORD_BYTES = 48
FINE_RECORD_BITS = 384
FINE_BLOCKS_PER_ROLE = ROLE_VALUES // BLOCK_VALUES
FINE_RECORDS = len(ROLES) * FINE_BLOCKS_PER_ROLE
FINE_BYTES = FINE_RECORDS * FINE_RECORD_BYTES

# This is a literal one-expert header, not an amortized six-expert common
# packet.  It consumes the complete 1/128-bpw difference between 319/128 and
# 320/128.  There is no inferred global packet and no uncharged metadata.
PILOT_HEADER_BYTES = 4_608
COMPOSITE_BYTES = PILOT_HEADER_BYTES + COARSE_BYTES + FINE_BYTES
COMPOSITE_BPW = Fraction(8 * COMPOSITE_BYTES, EXPERT_VALUES)
COARSE_BPW = Fraction(8 * COARSE_BYTES, EXPERT_VALUES)
FINE_BPW = Fraction(8 * FINE_BYTES, EXPERT_VALUES)
HEADER_BPW = Fraction(8 * PILOT_HEADER_BYTES, EXPERT_VALUES)

require(COARSE_BPW == Fraction(307, 128), "coarse rate identity")
require(FINE_BPW == Fraction(12, 128), "fine rate identity")
require(HEADER_BPW == Fraction(1, 128), "header rate identity")
require(COMPOSITE_BPW == Fraction(5, 2), "single-expert 2.5-bpw identity")
require(COMPOSITE_BYTES % 4096 == 0, "aligned pilot page count")

HEADER_MAGIC = b"TDH3FINE"
HEADER_SCHEMA = "tactic-dh384-finite-v3-single-expert-header-v1"
COMPOSITE_SCHEMA = "tactic-dh384-finite-v3-single-expert-composite-v1"

SPLITMIX_DOMAIN = 0x5441435449434448
UNIVERSAL_SELECTOR_ORDINAL = 17
MASK64 = (1 << 64) - 1
SELECTOR_PACKET_SHA256 = (
    "0946880088b766265a29d7d84ef4165a92a636eba0877dee9ce8b5b43dac56ad"
)

SCALE_DENOMINATOR = 1 << 18
SCALE_LAW_ID = "alpha=maxabs_f32(coarse_block)*uint8_scale^2/2^18"
SIGN_LAW_ID = "bit=1:+alpha;bit=0:-alpha;coefficient-order-0-through-375"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                FormatError(f"{label}: nonfinite {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FormatError(f"{label}: JSON: {error}") from error
    require(isinstance(value, dict), f"{label}: object")
    return value


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    word = state
    word = ((word ^ (word >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    word = ((word ^ (word >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (word ^ (word >> 31)) & MASK64


def universal_selector_table() -> bytes:
    state = (
        SPLITMIX_DOMAIN
        ^ ((UNIVERSAL_SELECTOR_ORDINAL + 1) * 0xD1B54A32D192ED03)
    ) & MASK64
    output = bytearray(STAGES * 256)
    for index in range(len(output)):
        state, word = splitmix64(state)
        output[index] = word & 7
    return bytes(output)


def universal_selector_packet() -> bytes:
    table = universal_selector_table()
    header = canonical_json({
        "format": "TACTIC-DH384-UNIVERSAL-SELECTOR-v2",
        "generator": "SplitMix64",
        "generator_domain_u64_hex": f"{SPLITMIX_DOMAIN:016x}",
        "ops": "swap/sign0/sign1",
        "stages": STAGES,
        "states": 256,
        "universal_ordinal": UNIVERSAL_SELECTOR_ORDINAL,
    }) + b"\n"
    require(len(header) + len(table) <= 16_384, "selector packet overflow")
    packet = header + table + bytes(16_384 - len(header) - len(table))
    require(sha256(packet) == SELECTOR_PACKET_SHA256,
            "audited selector packet identity")
    return packet


def pair_indices(length: int, stage: int) -> Iterable[tuple[int, int]]:
    stride = 1 << stage
    for base in range(0, length, 2 * stride):
        for offset in range(stride):
            yield base + offset, base + stride + offset


def feature_state(role_ordinal: int, left: int, right: int,
                  mean_abs: int) -> int:
    require(0 <= role_ordinal < len(ROLES), "role ordinal")
    absolute_left, absolute_right = abs(int(left)), abs(int(right))
    threshold = max(0, int(mean_abs))
    feature = (
        (role_ordinal << 6)
        | ((int(left) < 0) << 5)
        | ((int(right) < 0) << 4)
        | ((absolute_left > absolute_right) << 3)
        | (((absolute_left + absolute_right) > 2 * threshold) << 2)
        | ((absolute_left > threshold) << 1)
        | (absolute_right > threshold)
    )
    require(0 <= feature < 256, "feature overflow")
    return feature


def conditional_schedule(
    symbols: Sequence[int], role_ordinal: int,
    table: bytes | None = None,
) -> tuple[tuple[int, ...], ...]:
    require(len(symbols) == BLOCK_VALUES, "symbol block geometry")
    selected = universal_selector_table() if table is None else table
    require(selected == universal_selector_table(), "only frozen selector admitted")
    mean_abs = sum(abs(int(value)) for value in symbols) // BLOCK_VALUES
    shadow = [int(value) for value in symbols]
    schedule: list[tuple[int, ...]] = []
    for stage in range(STAGES):
        next_shadow = shadow.copy()
        operations: list[int] = []
        for left_index, right_index in pair_indices(BLOCK_VALUES, stage):
            left, right = shadow[left_index], shadow[right_index]
            feature = feature_state(role_ordinal, left, right, mean_abs)
            operation = selected[stage * 256 + feature] & 7
            a, b = (right, left) if operation & 1 else (left, right)
            if operation & 2:
                a = -a
            if operation & 4:
                b = -b
            next_shadow[left_index] = a + b
            next_shadow[right_index] = a - b
            operations.append(operation)
        shadow = next_shadow
        schedule.append(tuple(operations))
    return tuple(schedule)


def analysis_cpu(
    values: Sequence[float], schedule: Sequence[Sequence[int]],
) -> list[float]:
    """Apply the audited orthogonal B^T transform in FP64 semantics."""
    require(len(values) == BLOCK_VALUES and len(schedule) == STAGES,
            "analysis geometry")
    transformed = [float(value) for value in values]
    for stage in reversed(range(STAGES)):
        next_values = transformed.copy()
        pairs = tuple(pair_indices(BLOCK_VALUES, stage))
        require(len(schedule[stage]) == len(pairs), "analysis schedule count")
        for ordinal, (left_index, right_index) in enumerate(pairs):
            operation = int(schedule[stage][ordinal])
            left, right = transformed[left_index], transformed[right_index]
            x0, x1 = left + right, left - right
            if operation & 2:
                x0 = -x0
            if operation & 4:
                x1 = -x1
            if operation & 1:
                x0, x1 = x1, x0
            next_values[left_index] = x0
            next_values[right_index] = x1
        transformed = next_values
    return [value / 64.0 for value in transformed]


def synthesis_cpu(
    coefficients: Sequence[float], schedule: Sequence[Sequence[int]],
) -> list[float]:
    """Apply B, the exact inverse of :func:`analysis_cpu`."""
    require(len(coefficients) == BLOCK_VALUES and len(schedule) == STAGES,
            "synthesis geometry")
    values = [float(value) for value in coefficients]
    for stage in range(STAGES):
        next_values = values.copy()
        pairs = tuple(pair_indices(BLOCK_VALUES, stage))
        require(len(schedule[stage]) == len(pairs), "synthesis schedule count")
        for ordinal, (left_index, right_index) in enumerate(pairs):
            operation = int(schedule[stage][ordinal])
            left, right = values[left_index], values[right_index]
            a, b = (right, left) if operation & 1 else (left, right)
            if operation & 2:
                a = -a
            if operation & 4:
                b = -b
            next_values[left_index] = a + b
            next_values[right_index] = a - b
        values = next_values
    return [value / 64.0 for value in values]


def scale_alpha(scale_code: int, coarse_max_abs_f32: float) -> float:
    require(type(scale_code) is int and 0 <= scale_code <= 255,
            "uint8 scale code")
    base = float(coarse_max_abs_f32)
    require(math.isfinite(base) and base >= 0.0, "finite max-abs base")
    # Coarse reconstruction values are binary32, hence base is an exact dyadic
    # rational when widened to binary64.  Multiplication by code^2/2^18 keeps
    # the law fully decoder reproducible without source-fitted side data.
    return base * float(scale_code * scale_code) / float(SCALE_DENOMINATOR)


def pack_record(scale_code: int, positive_signs: Sequence[bool]) -> bytes:
    require(type(scale_code) is int and 0 <= scale_code <= 255,
            "record scale code")
    require(len(positive_signs) == ACTIVE_RANK, "record sign count")
    if scale_code == 0:
        require(not any(bool(value) for value in positive_signs),
                "zero scale canonical all-zero sign field")
    packed = bytearray(47)
    for ordinal, positive in enumerate(positive_signs):
        if bool(positive):
            packed[ordinal >> 3] |= 1 << (ordinal & 7)
    payload = bytes([scale_code]) + bytes(packed)
    require(len(payload) == FINE_RECORD_BYTES, "fine record bytes")
    return payload


def unpack_record(payload: bytes) -> tuple[int, tuple[bool, ...]]:
    require(type(payload) is bytes and len(payload) == FINE_RECORD_BYTES,
            "literal 48-byte record")
    scale_code = payload[0]
    signs = tuple(
        bool(payload[1 + (ordinal >> 3)] & (1 << (ordinal & 7)))
        for ordinal in range(ACTIVE_RANK)
    )
    if scale_code == 0:
        require(not any(signs), "noncanonical zero-scale record")
    require(pack_record(scale_code, signs) == payload,
            "record independent canonical reencode")
    return scale_code, signs


def select_record_cpu(
    analysed_residual: Sequence[float], coarse_reconstruction: Sequence[float],
) -> bytes:
    require(len(analysed_residual) == len(coarse_reconstruction) == BLOCK_VALUES,
            "record-selection geometry")
    coarse_max = max((abs(float(value)) for value in coarse_reconstruction),
                     default=0.0)
    require(math.isfinite(coarse_max), "finite coarse max")
    coefficients = [float(value) for value in analysed_residual[:ACTIVE_RANK]]
    require(all(math.isfinite(value) for value in coefficients),
            "finite analysed residual")
    absolute_sum = math.fsum(abs(value) for value in coefficients)
    best_code = 0
    best_objective = 0.0
    for code in range(1, 256):
        alpha = scale_alpha(code, coarse_max)
        objective = ACTIVE_RANK * alpha * alpha - 2.0 * alpha * absolute_sum
        if objective < best_objective:
            best_objective = objective
            best_code = code
    if best_code == 0:
        signs = (False,) * ACTIVE_RANK
    else:
        # Positive is the canonical tie decision at an exact zero coefficient.
        signs = tuple(value >= 0.0 for value in coefficients)
    return pack_record(best_code, signs)


def coefficient_vector_from_record(
    payload: bytes, coarse_reconstruction: Sequence[float],
) -> list[float]:
    require(len(coarse_reconstruction) == BLOCK_VALUES,
            "coarse reconstruction block")
    scale_code, signs = unpack_record(payload)
    coarse_max = max((abs(float(value)) for value in coarse_reconstruction),
                     default=0.0)
    alpha = scale_alpha(scale_code, coarse_max)
    result = [0.0] * BLOCK_VALUES
    if scale_code:
        for ordinal, positive in enumerate(signs):
            result[ordinal] = alpha if positive else -alpha
    return result


def correction_cpu(
    payload: bytes,
    symbols: Sequence[int],
    coarse_reconstruction: Sequence[float],
    role_ordinal: int,
) -> list[float]:
    schedule = conditional_schedule(symbols, role_ordinal)
    coefficients = coefficient_vector_from_record(payload, coarse_reconstruction)
    correction = synthesis_cpu(coefficients, schedule)
    # A decoder-visible executable assertion of membership in the audited
    # dyadic span.  Numerical tolerance covers only the 12 FP64 butterfly
    # stages; all transmitted coefficients beyond ACTIVE_RANK are literal 0.
    recovered = analysis_cpu(correction, schedule)
    scale = max(1.0, max((abs(value) for value in coefficients), default=0.0))
    require(max(abs(recovered[index]) for index in range(ACTIVE_RANK, BLOCK_VALUES))
            <= 2e-12 * scale, "correction escaped deterministic dyadic span")
    require(max(abs(recovered[index] - coefficients[index])
                for index in range(ACTIVE_RANK)) <= 2e-12 * scale,
            "correction coefficient roundtrip")
    return correction


def record_ordinal(role_ordinal: int, block_ordinal_within_role: int) -> int:
    require(0 <= role_ordinal < len(ROLES), "record role ordinal")
    require(0 <= block_ordinal_within_role < FINE_BLOCKS_PER_ROLE,
            "record block ordinal")
    return role_ordinal * FINE_BLOCKS_PER_ROLE + block_ordinal_within_role


def split_fine_stream(payload: bytes) -> tuple[bytes, ...]:
    require(type(payload) is bytes and len(payload) == FINE_BYTES,
            "fine stream exact bytes")
    records = tuple(
        payload[offset:offset + FINE_RECORD_BYTES]
        for offset in range(0, len(payload), FINE_RECORD_BYTES)
    )
    require(len(records) == FINE_RECORDS, "fine stream record count")
    for record in records:
        unpack_record(record)
    return records


def make_header(bindings: dict[str, Any]) -> bytes:
    required = {
        "coarse_sha256", "fine_sha256", "input_manifest_sha256",
        "v6_complete_sha256", "producer_source_manifest_sha256",
        "producer_source_root_sha256",
    }
    require(set(bindings) == required, "header binding schema")
    for name, digest in bindings.items():
        require(isinstance(digest, str) and len(digest) == 64 and
                all(character in "0123456789abcdef" for character in digest),
                f"header digest {name}")
    record = {
        "schema": HEADER_SCHEMA,
        "container_schema": COMPOSITE_SCHEMA,
        "geometry": {
            "intermediate": INTERMEDIATE,
            "hidden": HIDDEN,
            "roles": list(ROLES),
            "values": EXPERT_VALUES,
        },
        "layout": {
            "header_bytes": PILOT_HEADER_BYTES,
            "coarse_bytes": COARSE_BYTES,
            "fine_bytes": FINE_BYTES,
            "composite_bytes": COMPOSITE_BYTES,
            "fine_records": FINE_RECORDS,
            "fine_record_bytes": FINE_RECORD_BYTES,
            "global_packet_bytes_emitted": 0,
        },
        "physical_rate": {
            "coarse": "307/128",
            "fine": "12/128",
            "single_expert_header": "1/128",
            "literal_single_expert_total": "320/128",
        },
        "codebook": {
            "selector_ordinal": UNIVERSAL_SELECTOR_ORDINAL,
            "selector_packet_sha256": SELECTOR_PACKET_SHA256,
            "audited_parent_rank": AUDITED_PARENT_RANK,
            "active_sign_rank": ACTIVE_RANK,
            "scale_bits": 8,
            "sign_bits": ACTIVE_RANK,
            "record_bits": FINE_RECORD_BITS,
            "scale_law": SCALE_LAW_ID,
            "scale_denominator": SCALE_DENOMINATOR,
            "sign_law": SIGN_LAW_ID,
            "zero_scale_canonical_sign_field": "all-zero",
        },
        "bindings": bindings,
        "claim_boundary": {
            "one_qwen_geometry_expert_pilot_only": True,
            "six_expert_global_packet_emitted_or_parsed": False,
            "seventy_three_over_seventy_two_claim": False,
            "universal_tails_resolved": False,
            "non_qwen_portability_resolved": False,
        },
    }
    body = canonical_json(record)
    prefix = struct.pack("<8sI", HEADER_MAGIC, len(body))
    require(len(prefix) + len(body) <= PILOT_HEADER_BYTES,
            "single-expert header overflow")
    return prefix + body + bytes(PILOT_HEADER_BYTES - len(prefix) - len(body))


def parse_header(payload: bytes) -> dict[str, Any]:
    require(type(payload) is bytes and len(payload) == PILOT_HEADER_BYTES,
            "literal pilot header bytes")
    magic, length = struct.unpack("<8sI", payload[:12])
    require(magic == HEADER_MAGIC and 0 < length <= PILOT_HEADER_BYTES - 12,
            "pilot header prefix")
    body = payload[12:12 + length]
    require(payload[12 + length:] == bytes(PILOT_HEADER_BYTES - 12 - length),
            "pilot header canonical zero padding")
    record = strict_json(body, "pilot header")
    require(canonical_json(record) == body, "pilot header canonical JSON")
    require(record.get("schema") == HEADER_SCHEMA and
            record.get("container_schema") == COMPOSITE_SCHEMA,
            "pilot header schema")
    require(record.get("geometry") == {
        "intermediate": INTERMEDIATE,
        "hidden": HIDDEN,
        "roles": list(ROLES),
        "values": EXPERT_VALUES,
    }, "pilot header geometry")
    layout = record.get("layout")
    require(layout == {
        "coarse_bytes": COARSE_BYTES,
        "composite_bytes": COMPOSITE_BYTES,
        "fine_bytes": FINE_BYTES,
        "fine_record_bytes": FINE_RECORD_BYTES,
        "fine_records": FINE_RECORDS,
        "global_packet_bytes_emitted": 0,
        "header_bytes": PILOT_HEADER_BYTES,
    }, "pilot header layout")
    require(record.get("physical_rate") == {
        "coarse": "307/128",
        "fine": "12/128",
        "literal_single_expert_total": "320/128",
        "single_expert_header": "1/128",
    }, "pilot header exact rate")
    expected_codebook = {
        "active_sign_rank": ACTIVE_RANK,
        "audited_parent_rank": AUDITED_PARENT_RANK,
        "record_bits": FINE_RECORD_BITS,
        "scale_bits": 8,
        "scale_denominator": SCALE_DENOMINATOR,
        "scale_law": SCALE_LAW_ID,
        "selector_ordinal": UNIVERSAL_SELECTOR_ORDINAL,
        "selector_packet_sha256": SELECTOR_PACKET_SHA256,
        "sign_bits": ACTIVE_RANK,
        "sign_law": SIGN_LAW_ID,
        "zero_scale_canonical_sign_field": "all-zero",
    }
    require(record.get("codebook") == expected_codebook,
            "pilot header codebook")
    boundary = record.get("claim_boundary")
    require(boundary == {
        "non_qwen_portability_resolved": False,
        "one_qwen_geometry_expert_pilot_only": True,
        "seventy_three_over_seventy_two_claim": False,
        "six_expert_global_packet_emitted_or_parsed": False,
        "universal_tails_resolved": False,
    }, "pilot header claim boundary")
    bindings = record.get("bindings")
    require(isinstance(bindings, dict), "pilot header bindings")
    # Rebuilding proves exact schema, digest syntax and canonical padding.
    require(make_header(bindings) == payload, "pilot header canonical reencode")
    return record


def split_composite(payload: bytes) -> tuple[dict[str, Any], bytes, bytes]:
    require(type(payload) is bytes and len(payload) == COMPOSITE_BYTES,
            "literal single-expert composite bytes")
    header_bytes = payload[:PILOT_HEADER_BYTES]
    coarse_begin = PILOT_HEADER_BYTES
    fine_begin = coarse_begin + COARSE_BYTES
    coarse = payload[coarse_begin:fine_begin]
    fine = payload[fine_begin:]
    header = parse_header(header_bytes)
    require(sha256(coarse) == header["bindings"]["coarse_sha256"],
            "header/coarse binding")
    require(sha256(fine) == header["bindings"]["fine_sha256"],
            "header/fine binding")
    split_fine_stream(fine)
    return header, coarse, fine


def single_expert_traffic(start_offset_mod_page: int = 0,
                          external_passes: int = 1) -> dict[str, Any]:
    require(type(start_offset_mod_page) is int and
            0 <= start_offset_mod_page < 4096, "traffic page offset")
    require(type(external_passes) is int and 0 <= external_passes <= 16,
            "traffic pass count")
    unique_pages = (start_offset_mod_page + COMPOSITE_BYTES + 4095) // 4096
    first_pass = COMPOSITE_BYTES if external_passes else 0
    total = external_passes * COMPOSITE_BYTES
    reread = max(0, external_passes - 1) * COMPOSITE_BYTES
    require(total == first_pass + reread, "traffic identity")
    return {
        "schema": "tactic-dh384-finite-v3-single-expert-traffic-v1",
        "literal_pilot": {
            "header_bytes": PILOT_HEADER_BYTES,
            "coarse_bytes": COARSE_BYTES,
            "fine_bytes": FINE_BYTES,
            "total_bytes": COMPOSITE_BYTES,
            "external_passes": external_passes,
            "first_pass_bytes": first_pass,
            "reread_bytes": reread,
            "total_read_bytes": total,
            "unique_pages": unique_pages if external_passes else 0,
            "unique_page_bytes": unique_pages * 4096 if external_passes else 0,
            "amplification_over_literal_pilot_bytes":
                total / COMPOSITE_BYTES,
        },
        "six_expert_amortized_tactic_layout": {
            "emitted": False,
            "parsed": False,
            "global_packet_bytes": None,
            "read_amplification": None,
            "seventy_three_over_seventy_two_claim": False,
        },
        "accelerator_hbm": {
            "measured": False,
            "read_bytes": None,
            "read_amplification": None,
            "below_2x_claim_authority": False,
        },
    }
