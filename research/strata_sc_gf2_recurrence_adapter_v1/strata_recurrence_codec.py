#!/usr/bin/env python3
"""Canonical exact-GF(2) packets for current six-level STRATA SC decisions."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence


LEVELS = 6
CHUNK_DECISIONS = 4096
OWNER_DESCRIPTOR_BITS = 128
FORMAT_SCOPE = "qwen-128-expert-pilot-not-universal-swiglu-moe"
RECURRENCE_SCOPE = "independent-level-chunks-max-4096-decisions"
PAGE_BYTES = 4096
EXPERT_HEADER_BYTES = 256
CHUNK_HEADER = struct.Struct("<4sBBBBIIQIIII32sI4s")
EXPERT_CORE = struct.Struct("<8sHHIQQIIIIII32s32s32s32s32sI")
CHUNK_MAGIC = b"SGC1"
EXPERT_MAGIC = b"STRGF201"
VERSION = 1
MODE_RAW = 0
MODE_LFSR = 1
ROLE_IDS = {"gate": 0, "up": 1, "down_transposed": 2}


class CodecError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CodecError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json(payload: bytes, label: str) -> Any:
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, f"{label} duplicate key")
            output[key] = value
        return output

    return json.loads(
        payload.decode("ascii"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            CodecError(f"{label} nonfinite {token}")
        ),
    )


def align_up(value: int, alignment: int) -> int:
    require(type(value) is int and value >= 0, "alignment value")
    require(type(alignment) is int and alignment > 0, "alignment")
    return ((value + alignment - 1) // alignment) * alignment


def _digest(value: str, label: str) -> str:
    require(
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        label,
    )
    return value


def _bits(values: Any, label: str) -> bytes:
    require(isinstance(values, bytes) and len(values) > 0, label)
    require(all(value in (0, 1) for value in values), label)
    return values


def pack_bits(values: Iterable[int]) -> bytes:
    bits = tuple(int(value) for value in values)
    require(all(value in (0, 1) for value in bits), "packed bit")
    output = bytearray((len(bits) + 7) // 8)
    for index, value in enumerate(bits):
        output[index >> 3] |= value << (7 - (index & 7))
    return bytes(output)


def unpack_bits(payload: bytes, count: int) -> bytes:
    require(type(count) is int and count >= 0 and len(payload) == (count + 7) // 8, "bit payload geometry")
    if count & 7:
        require(payload[-1] & ((1 << (8 - (count & 7))) - 1) == 0, "terminal bit padding")
    return bytes((payload[index >> 3] >> (7 - (index & 7))) & 1 for index in range(count))


def berlekamp_massey(values: bytes) -> tuple[int, int]:
    sequence = _bits(values, "BM sequence")
    connection = 1
    previous = 1
    complexity = 0
    shift = 1
    history = 0
    for index, bit in enumerate(sequence):
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
    return complexity, connection


def generate_lfsr(initial: bytes, connection: int, length: int) -> bytes:
    require(isinstance(initial, bytes) and all(value in (0, 1) for value in initial), "LFSR initial")
    order = len(initial)
    require(type(connection) is int and connection >= 1 and connection & 1, "LFSR connection")
    require(connection.bit_length() <= order + 1 and type(length) is int and length >= order, "LFSR geometry")
    output = bytearray(initial)
    for index in range(order, length):
        value = 0
        for lag in range(1, order + 1):
            value ^= ((connection >> lag) & 1) & output[index - lag]
        output.append(value)
    return bytes(output)


@dataclass(frozen=True)
class StreamSource:
    ordinal: int
    role: str
    owner_set_hex: str
    owner_contributions: tuple[Mapping[str, Any], ...]
    local_source_offset: int
    local_weight_count: int
    global_source_weights: int
    profile_q: int
    decoder_scale_f16le: bytes
    logn: int
    sc_seed_u32: int
    rht_seed_u64: int
    state: bytes
    selected_bits: bytes
    levels: bytes
    selected_sha256: str
    levels_sha256: str
    base_frequencies_u16le_sha256: str
    decoded_triplet_sha256: str
    baseline_payload_bytes: int
    baseline_logical_bits: int
    baseline_payload_sha256: str
    candidate_payload_bytes: int
    candidate_logical_bits: int
    candidate_payload_sha256: str


def _contributions(value: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows = []
    for row in value:
        require(isinstance(row, Mapping) and set(row) == {"expert", "role", "source_offset", "weight_count"}, "owner contribution")
        expert = row["expert"]
        role = row["role"]
        source_offset = row["source_offset"]
        count = row["weight_count"]
        require(type(expert) is int and expert >= 0 and role in ROLE_IDS, "contribution owner/role")
        require(type(source_offset) is int and source_offset >= 0 and type(count) is int and count > 0, "contribution interval")
        rows.append({"expert": expert, "role": role, "source_offset": source_offset, "weight_count": count})
    require(bool(rows) and rows == sorted(rows, key=lambda row: (row["expert"], ROLE_IDS[row["role"]], row["source_offset"])), "canonical contributions")
    return tuple(rows)


def validate_stream(source: StreamSource, expert_ordinal: int) -> dict[str, Any]:
    require(type(source.ordinal) is int and 0 <= source.ordinal < 2**32, "stream ordinal")
    require(source.role in ROLE_IDS, "stream role")
    require(isinstance(source.owner_set_hex, str) and len(source.owner_set_hex) == 32, "owner set")
    try:
        owner_set = bytes.fromhex(source.owner_set_hex)
    except ValueError as error:
        raise CodecError("owner set hex") from error
    require(len(owner_set) == 16 and owner_set.hex() == source.owner_set_hex, "canonical owner set bytes")
    contributions = _contributions(source.owner_contributions)
    require(all(row["role"] == source.role for row in contributions), "contribution role binding")
    expected_owner_set = bytearray(OWNER_DESCRIPTOR_BITS // 8)
    for row in contributions:
        require(row["expert"] < OWNER_DESCRIPTOR_BITS, "owner outside Qwen-shaped 128-expert pilot descriptor")
        expected_owner_set[row["expert"] >> 3] |= 1 << (row["expert"] & 7)
    require(owner_set == bytes(expected_owner_set), "owner set/contribution identity")
    local = [row for row in contributions if row["expert"] == expert_ordinal]
    require(bool(local), "expert must own stream")
    require(type(source.local_source_offset) is int and source.local_source_offset >= 0, "local offset")
    require(type(source.local_weight_count) is int and source.local_weight_count > 0, "local weights")
    require(source.local_source_offset == local[0]["source_offset"], "local source offset identity")
    local_cursor = source.local_source_offset
    for row in local:
        require(row["source_offset"] == local_cursor, "local contribution contiguity")
        local_cursor += row["weight_count"]
    require(local_cursor - source.local_source_offset == source.local_weight_count, "local contribution count")
    require(type(source.global_source_weights) is int and source.global_source_weights == sum(row["weight_count"] for row in contributions), "global contribution count")
    require(type(source.profile_q) is int and 0 <= source.profile_q <= 65535, "profile")
    require(isinstance(source.decoder_scale_f16le, bytes) and len(source.decoder_scale_f16le) == 2, "scale bits")
    scale = struct.unpack("<e", source.decoder_scale_f16le)[0]
    require(math.isfinite(scale) and scale > 0.0, "finite positive decoder scale")
    require(type(source.logn) is int and 1 <= source.logn <= 31, "logn")
    require(type(source.sc_seed_u32) is int and 0 <= source.sc_seed_u32 < 2**32, "SC seed")
    require(type(source.rht_seed_u64) is int and 0 <= source.rht_seed_u64 < 2**64, "RHT seed")
    require(isinstance(source.state, bytes), "state bytes")
    bits = _bits(source.selected_bits, "selected bits")
    require(isinstance(source.levels, bytes) and len(source.levels) == len(bits), "level geometry")
    require(all(value < LEVELS for value in source.levels), "level value")
    require(list(source.levels) == sorted(source.levels), "six level-major ordering")
    counts = tuple(source.levels.count(level) for level in range(LEVELS))
    require(all(count > 0 for count in counts), "all six level boundaries")
    require(sha256(bits) == _digest(source.selected_sha256, "selected digest"), "selected digest match")
    require(sha256(source.levels) == _digest(source.levels_sha256, "levels digest"), "levels digest match")
    _digest(source.base_frequencies_u16le_sha256, "base frequency digest")
    _digest(source.decoded_triplet_sha256, "triplet digest")
    _digest(source.baseline_payload_sha256, "baseline payload digest")
    _digest(source.candidate_payload_sha256, "candidate payload digest")
    require(type(source.baseline_payload_bytes) is int and source.baseline_payload_bytes > 0, "baseline payload bytes")
    require(type(source.baseline_logical_bits) is int and 0 < source.baseline_logical_bits <= 8 * source.baseline_payload_bytes, "baseline logical bits")
    require(type(source.candidate_payload_bytes) is int and source.candidate_payload_bytes > 0, "candidate payload bytes")
    require(type(source.candidate_logical_bits) is int and 0 < source.candidate_logical_bits <= 8 * source.candidate_payload_bytes, "candidate logical bits")
    return {"contributions": contributions, "level_counts": counts, "scale": scale}


def _validate_expert_role_coverage(
    streams: Sequence[StreamSource], expert_ordinal: int, source_weights: int,
) -> None:
    """Require exact Gate/Up/Down scalar coverage for one SwiGLU expert."""

    require(source_weights % 3 == 0, "SwiGLU expert weight geometry")
    matrix_weights = source_weights // 3
    intervals: dict[str, list[tuple[int, int]]] = {role: [] for role in ROLE_IDS}
    for source in streams:
        for row in _contributions(source.owner_contributions):
            if row["expert"] == expert_ordinal:
                begin = row["source_offset"]
                end = begin + row["weight_count"]
                require(end <= matrix_weights, "expert role interval bound")
                intervals[row["role"]].append((begin, end))
    for role in ROLE_IDS:
        cursor = 0
        for begin, end in sorted(intervals[role]):
            require(begin == cursor and end > begin, f"{role} exact scalar coverage")
            cursor = end
        require(cursor == matrix_weights, f"{role} complete scalar coverage")


def _chunk_payload(bits: bytes, mode: int, complexity: int, connection: int) -> tuple[bytes, int]:
    if mode == MODE_RAW:
        return pack_bits(bits), len(bits)
    coefficients = bytes((connection >> lag) & 1 for lag in range(1, complexity + 1))
    values = coefficients + bits[:complexity]
    return pack_bits(values), 2 * complexity


def encode_chunk(
    *, stream_ordinal: int, level: int, chunk_ordinal: int,
    stream_begin: int, bits: bytes, force_raw: bool = False,
    _validate_roundtrip: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    sequence = _bits(bits, "chunk bits")
    require(0 <= level < LEVELS and 0 <= stream_ordinal < 2**32 and 0 <= chunk_ordinal < 2**32, "chunk identity")
    require(type(stream_begin) is int and stream_begin >= 0 and len(sequence) <= CHUNK_DECISIONS, "chunk geometry")
    complexity, connection = berlekamp_massey(sequence)
    mode = MODE_RAW if force_raw or 2 * complexity >= len(sequence) else MODE_LFSR
    chosen_complexity = 0 if mode == MODE_RAW else complexity
    payload, payload_bits = _chunk_payload(sequence, mode, chosen_complexity, connection)
    zero = CHUNK_HEADER.pack(
        CHUNK_MAGIC, VERSION, level, mode, 0, stream_ordinal, chunk_ordinal,
        stream_begin, len(sequence), chosen_complexity, payload_bits, len(payload),
        bytes.fromhex(sha256(payload)), 0, bytes(4),
    )
    crc = zlib.crc32(zero + payload) & 0xFFFFFFFF
    header = CHUNK_HEADER.pack(
        CHUNK_MAGIC, VERSION, level, mode, 0, stream_ordinal, chunk_ordinal,
        stream_begin, len(sequence), chosen_complexity, payload_bits, len(payload),
        bytes.fromhex(sha256(payload)), crc, bytes(4),
    )
    packet = header + payload
    if _validate_roundtrip:
        decoded = decode_chunk(packet)
        require(decoded["bits"] == sequence, "chunk replay")
    return packet, {
        "stream_ordinal": stream_ordinal, "level": level,
        "chunk_ordinal": chunk_ordinal, "stream_begin": stream_begin,
        "decisions": len(sequence), "mode": "raw" if mode == MODE_RAW else "lfsr",
        "linear_complexity": chosen_complexity, "payload_bits": payload_bits,
        "packet_bytes": len(packet),
    }


def decode_chunk(packet: bytes) -> dict[str, Any]:
    require(isinstance(packet, bytes) and len(packet) >= CHUNK_HEADER.size, "chunk packet")
    fields = CHUNK_HEADER.unpack_from(packet, 0)
    (magic, version, level, mode, reserved, stream, ordinal, begin, count,
     complexity, payload_bits, payload_bytes, payload_sha, claimed_crc,
     tail) = fields
    require((magic, version, reserved, tail) == (CHUNK_MAGIC, VERSION, 0, bytes(4)), "chunk header")
    require(0 <= level < LEVELS and mode in (MODE_RAW, MODE_LFSR), "chunk level/mode")
    require(0 < count <= CHUNK_DECISIONS and len(packet) == CHUNK_HEADER.size + payload_bytes, "chunk size")
    require((mode == MODE_RAW and complexity == 0 and payload_bits == count) or (mode == MODE_LFSR and 2 * complexity == payload_bits and 2 * complexity < count), "chunk canonical mode")
    require(payload_bytes == (payload_bits + 7) // 8, "chunk payload bytes")
    payload = packet[CHUNK_HEADER.size:]
    require(bytes.fromhex(sha256(payload)) == payload_sha, "chunk payload SHA")
    zero = CHUNK_HEADER.pack(
        magic, version, level, mode, reserved, stream, ordinal, begin, count,
        complexity, payload_bits, payload_bytes, payload_sha, 0, tail,
    )
    require(zlib.crc32(zero + payload) & 0xFFFFFFFF == claimed_crc, "chunk CRC")
    values = unpack_bits(payload, payload_bits)
    if mode == MODE_RAW:
        bits = values
    else:
        coefficients = values[:complexity]
        initial = values[complexity:]
        connection = 1
        for lag, value in enumerate(coefficients, start=1):
            connection |= value << lag
        bits = generate_lfsr(initial, connection, count)
    canonical, _row = encode_chunk(
        stream_ordinal=stream, level=level, chunk_ordinal=ordinal,
        stream_begin=begin, bits=bits, _validate_roundtrip=False,
    )
    require(canonical == packet, "chunk canonical re-encode")
    return {
        "stream_ordinal": stream, "level": level, "chunk_ordinal": ordinal,
        "stream_begin": begin, "decisions": count, "complexity": complexity,
        "mode": mode, "payload_bits": payload_bits, "bits": bits,
    }


def _stream_metadata(source: StreamSource, validated: dict[str, Any]) -> dict[str, Any]:
    return {
        "ordinal": source.ordinal,
        "role": source.role,
        "owner_set_hex": source.owner_set_hex,
        "owner_contributions": list(validated["contributions"]),
        "local_source_offset": source.local_source_offset,
        "local_weight_count": source.local_weight_count,
        "global_source_weights": source.global_source_weights,
        "profile_q": source.profile_q,
        "decoder_scale_f16le_hex": source.decoder_scale_f16le.hex(),
        "logn": source.logn,
        "sc_seed_u32": source.sc_seed_u32,
        "rht_seed_u64": source.rht_seed_u64,
        "state_hex": source.state.hex(),
        "selected_decisions": len(source.selected_bits),
        "level_counts": list(validated["level_counts"]),
        "selected_sha256": source.selected_sha256,
        "levels_sha256": source.levels_sha256,
        "base_frequencies_u16le_sha256": source.base_frequencies_u16le_sha256,
        "decoded_selected_decision_triplet_sha256": source.decoded_triplet_sha256,
        "baseline_payload_bytes": source.baseline_payload_bytes,
        "baseline_logical_bits": source.baseline_logical_bits,
        "baseline_payload_sha256": source.baseline_payload_sha256,
        "candidate_payload_bytes": source.candidate_payload_bytes,
        "candidate_logical_bits": source.candidate_logical_bits,
        "candidate_payload_sha256": source.candidate_payload_sha256,
    }


def encode_expert(
    *, expert_ordinal: int, source_weights: int, semantic_state: bytes,
    streams: Sequence[StreamSource], audit_receipt_sha256: str,
    candidate_sha256: str, reconstruction_sha256: str,
    force_raw: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    require(type(expert_ordinal) is int and 0 <= expert_ordinal < 2**32, "expert ordinal")
    require(type(source_weights) is int and source_weights > 0, "expert weights")
    require(isinstance(semantic_state, bytes), "semantic state")
    receipt_digest = _digest(audit_receipt_sha256, "audit receipt digest")
    candidate_digest = _digest(candidate_sha256, "candidate digest")
    reconstruction_digest = _digest(reconstruction_sha256, "reconstruction digest")
    require(bool(streams) and [row.ordinal for row in streams] == sorted({row.ordinal for row in streams}), "stream canonical order")
    metadata_rows = []
    chunks = []
    chunk_rows = []
    local_coverage = 0
    global_chunk_ordinal = 0
    for source in streams:
        validated = validate_stream(source, expert_ordinal)
        metadata_rows.append(_stream_metadata(source, validated))
        local_coverage += source.local_weight_count
        cursor = 0
        for level, count in enumerate(validated["level_counts"]):
            level_bits = source.selected_bits[cursor:cursor + count]
            level_begin = cursor
            for offset in range(0, count, CHUNK_DECISIONS):
                part = level_bits[offset:offset + CHUNK_DECISIONS]
                packet, row = encode_chunk(
                    stream_ordinal=source.ordinal, level=level,
                    chunk_ordinal=global_chunk_ordinal,
                    stream_begin=level_begin + offset, bits=part,
                    force_raw=force_raw,
                )
                chunks.append(packet)
                chunk_rows.append(row)
                global_chunk_ordinal += 1
            cursor += count
        require(cursor == len(source.selected_bits), "stream chunk coverage")
    require(local_coverage == source_weights, "expert weight coverage")
    _validate_expert_role_coverage(streams, expert_ordinal, source_weights)
    metadata_record = {
        "schema": "strata-sc-gf2-expert-metadata-v1",
        "format_scope": FORMAT_SCOPE,
        "owner_descriptor_bits": OWNER_DESCRIPTOR_BITS,
        "recurrence_scope": RECURRENCE_SCOPE,
        "expert_ordinal": expert_ordinal,
        "source_weights": source_weights,
        "chunk_decisions": CHUNK_DECISIONS,
        "six_level_major": True,
        "semantic_state_hex": semantic_state.hex(),
        "semantic_state_sha256": sha256(semantic_state),
        "streams": metadata_rows,
    }
    metadata = canonical_json(metadata_record)
    payload = b"".join(chunks)
    metadata_offset = EXPERT_HEADER_BYTES
    payload_offset = align_up(metadata_offset + len(metadata), 64)
    body = metadata + bytes(payload_offset - metadata_offset - len(metadata)) + payload
    unpadded_total = EXPERT_HEADER_BYTES + len(body)
    total = align_up(unpadded_total, PAGE_BYTES)
    body += bytes(total - unpadded_total)
    body_sha = sha256(body)
    zero_core = EXPERT_CORE.pack(
        EXPERT_MAGIC, VERSION, EXPERT_HEADER_BYTES, expert_ordinal,
        source_weights, total, len(streams), len(chunks), metadata_offset,
        len(metadata), payload_offset, len(payload),
        bytes.fromhex(receipt_digest), bytes.fromhex(candidate_digest),
        bytes.fromhex(reconstruction_digest), bytes.fromhex(sha256(metadata)),
        bytes.fromhex(body_sha), 0,
    )
    zero_header = zero_core + bytes(EXPERT_HEADER_BYTES - len(zero_core))
    crc = zlib.crc32(zero_header + body) & 0xFFFFFFFF
    core = EXPERT_CORE.pack(
        EXPERT_MAGIC, VERSION, EXPERT_HEADER_BYTES, expert_ordinal,
        source_weights, total, len(streams), len(chunks), metadata_offset,
        len(metadata), payload_offset, len(payload),
        bytes.fromhex(receipt_digest), bytes.fromhex(candidate_digest),
        bytes.fromhex(reconstruction_digest), bytes.fromhex(sha256(metadata)),
        bytes.fromhex(body_sha), crc,
    )
    header = core + bytes(EXPERT_HEADER_BYTES - len(core))
    packet = header + body
    decoded = decode_expert(packet)
    require(decoded["expert_ordinal"] == expert_ordinal and decoded["source_weights"] == source_weights, "expert replay")
    return packet, {
        "physical_bytes": len(packet),
        "physical_rate_bpw": float(Fraction(8 * len(packet), source_weights)),
        "selected_decisions": sum(len(source.selected_bits) for source in streams),
        "chunk_count": len(chunks),
        "raw_chunks": sum(row["mode"] == "raw" for row in chunk_rows),
        "lfsr_chunks": sum(row["mode"] == "lfsr" for row in chunk_rows),
        "sum_linear_complexity": sum(row["linear_complexity"] for row in chunk_rows),
        "chunk_packet_bytes": sum(row["packet_bytes"] for row in chunk_rows),
        "model_bytes": 0,
        "exception_bytes": 0,
    }


def _metadata_stream_to_source(row: Mapping[str, Any], bits: bytes, levels: bytes) -> StreamSource:
    required = {
        "ordinal", "role", "owner_set_hex", "owner_contributions",
        "local_source_offset", "local_weight_count", "global_source_weights",
        "profile_q", "decoder_scale_f16le_hex", "logn", "sc_seed_u32",
        "rht_seed_u64", "state_hex", "selected_decisions", "level_counts",
        "selected_sha256", "levels_sha256", "base_frequencies_u16le_sha256",
        "decoded_selected_decision_triplet_sha256", "baseline_payload_bytes",
        "baseline_logical_bits", "baseline_payload_sha256",
        "candidate_payload_bytes", "candidate_logical_bits",
        "candidate_payload_sha256",
    }
    require(isinstance(row, Mapping) and set(row) == required, "stream metadata schema")
    try:
        scale = bytes.fromhex(row["decoder_scale_f16le_hex"])
        state = bytes.fromhex(row["state_hex"])
    except (TypeError, ValueError) as error:
        raise CodecError("stream hex field") from error
    require(
        scale.hex() == row["decoder_scale_f16le_hex"]
        and state.hex() == row["state_hex"],
        "canonical stream hex field",
    )
    return StreamSource(
        ordinal=row["ordinal"], role=row["role"], owner_set_hex=row["owner_set_hex"],
        owner_contributions=tuple(row["owner_contributions"]),
        local_source_offset=row["local_source_offset"],
        local_weight_count=row["local_weight_count"],
        global_source_weights=row["global_source_weights"],
        profile_q=row["profile_q"], decoder_scale_f16le=scale,
        logn=row["logn"], sc_seed_u32=row["sc_seed_u32"],
        rht_seed_u64=row["rht_seed_u64"], state=state,
        selected_bits=bits, levels=levels,
        selected_sha256=row["selected_sha256"], levels_sha256=row["levels_sha256"],
        base_frequencies_u16le_sha256=row["base_frequencies_u16le_sha256"],
        decoded_triplet_sha256=row["decoded_selected_decision_triplet_sha256"],
        baseline_payload_bytes=row["baseline_payload_bytes"],
        baseline_logical_bits=row["baseline_logical_bits"],
        baseline_payload_sha256=row["baseline_payload_sha256"],
        candidate_payload_bytes=row["candidate_payload_bytes"],
        candidate_logical_bits=row["candidate_logical_bits"],
        candidate_payload_sha256=row["candidate_payload_sha256"],
    )


def decode_expert(packet: bytes) -> dict[str, Any]:
    require(isinstance(packet, bytes) and len(packet) >= PAGE_BYTES and len(packet) % PAGE_BYTES == 0, "expert page geometry")
    core = EXPERT_CORE.unpack_from(packet, 0)
    (magic, version, header_bytes, expert, weights, total, stream_count,
     chunk_count, metadata_offset, metadata_bytes, payload_offset,
     payload_bytes, receipt_sha, candidate_sha, reconstruction_sha,
     metadata_sha, body_sha, claimed_crc) = core
    require((magic, version, header_bytes) == (EXPERT_MAGIC, VERSION, EXPERT_HEADER_BYTES), "expert header")
    require(packet[EXPERT_CORE.size:EXPERT_HEADER_BYTES] == bytes(EXPERT_HEADER_BYTES - EXPERT_CORE.size), "expert header padding")
    require(total == len(packet) and weights > 0 and stream_count > 0 and chunk_count > 0, "expert constants")
    require(metadata_offset == EXPERT_HEADER_BYTES and metadata_bytes > 0, "metadata offset")
    require(payload_offset == align_up(metadata_offset + metadata_bytes, 64), "payload offset")
    require(payload_offset + payload_bytes <= total, "payload end")
    require(packet[metadata_offset + metadata_bytes:payload_offset] == bytes(payload_offset - metadata_offset - metadata_bytes), "metadata padding")
    require(packet[payload_offset + payload_bytes:] == bytes(total - payload_offset - payload_bytes), "expert terminal padding")
    body = packet[EXPERT_HEADER_BYTES:]
    require(bytes.fromhex(sha256(body)) == body_sha, "expert body SHA")
    zero_core = EXPERT_CORE.pack(
        magic, version, header_bytes, expert, weights, total, stream_count,
        chunk_count, metadata_offset, metadata_bytes, payload_offset,
        payload_bytes, receipt_sha, candidate_sha, reconstruction_sha,
        metadata_sha, body_sha, 0,
    )
    zero_header = zero_core + bytes(EXPERT_HEADER_BYTES - len(zero_core))
    require(zlib.crc32(zero_header + body) & 0xFFFFFFFF == claimed_crc, "expert CRC")
    metadata_payload = packet[metadata_offset:metadata_offset + metadata_bytes]
    require(bytes.fromhex(sha256(metadata_payload)) == metadata_sha, "metadata SHA")
    metadata = strict_json(metadata_payload, "expert metadata")
    require(canonical_json(metadata) == metadata_payload, "canonical expert metadata JSON")
    require(isinstance(metadata, dict) and set(metadata) == {
        "schema", "format_scope", "owner_descriptor_bits",
        "recurrence_scope", "expert_ordinal", "source_weights", "chunk_decisions",
        "six_level_major", "semantic_state_hex", "semantic_state_sha256",
        "streams",
    }, "expert metadata schema")
    require(
        metadata["schema"] == "strata-sc-gf2-expert-metadata-v1"
        and metadata["format_scope"] == FORMAT_SCOPE
        and metadata["owner_descriptor_bits"] == OWNER_DESCRIPTOR_BITS
        and metadata["recurrence_scope"] == RECURRENCE_SCOPE
        and metadata["expert_ordinal"] == expert
        and metadata["source_weights"] == weights
        and metadata["chunk_decisions"] == CHUNK_DECISIONS
        and metadata["six_level_major"] is True,
        "expert metadata identity",
    )
    try:
        semantic_state = bytes.fromhex(metadata["semantic_state_hex"])
    except (TypeError, ValueError) as error:
        raise CodecError("semantic state hex") from error
    require(semantic_state.hex() == metadata["semantic_state_hex"], "canonical semantic state hex")
    require(sha256(semantic_state) == metadata["semantic_state_sha256"], "semantic state digest")
    rows = metadata["streams"]
    require(isinstance(rows, list) and len(rows) == stream_count, "stream metadata count")
    require([row.get("ordinal") for row in rows] == sorted({row.get("ordinal") for row in rows}), "stream metadata order")

    cursor = payload_offset
    chunks = []
    while cursor < payload_offset + payload_bytes:
        require(cursor + CHUNK_HEADER.size <= payload_offset + payload_bytes, "chunk header coverage")
        header = CHUNK_HEADER.unpack_from(packet, cursor)
        packet_bytes = CHUNK_HEADER.size + int(header[11])
        end = cursor + packet_bytes
        require(end <= payload_offset + payload_bytes, "chunk coverage")
        chunks.append(decode_chunk(packet[cursor:end]))
        cursor = end
    require(cursor == payload_offset + payload_bytes and len(chunks) == chunk_count, "chunk payload closure")
    require([row["chunk_ordinal"] for row in chunks] == list(range(chunk_count)), "global chunk order")

    expected_geometry = []
    expected_chunk_ordinal = 0
    for metadata_row in rows:
        begin = 0
        for level, count in enumerate(metadata_row["level_counts"]):
            require(type(count) is int and count > 0, "metadata level count")
            for offset in range(0, count, CHUNK_DECISIONS):
                decisions = min(CHUNK_DECISIONS, count - offset)
                expected_geometry.append((
                    metadata_row["ordinal"], level, expected_chunk_ordinal,
                    begin + offset, decisions,
                ))
                expected_chunk_ordinal += 1
            begin += count
    observed_geometry = [(
        row["stream_ordinal"], row["level"], row["chunk_ordinal"],
        row["stream_begin"], row["decisions"],
    ) for row in chunks]
    require(observed_geometry == expected_geometry, "canonical expert chunk geometry/order")

    decoded_sources = []
    local_coverage = 0
    for metadata_row in rows:
        ordinal = metadata_row["ordinal"]
        selected = [row for row in chunks if row["stream_ordinal"] == ordinal]
        require(bool(selected), "stream chunks")
        bits = bytearray()
        levels = bytearray()
        expected_begin = 0
        previous_level = -1
        for chunk in selected:
            require(chunk["stream_begin"] == expected_begin and chunk["level"] >= previous_level, "chunk stream order")
            require(chunk["level"] <= previous_level + 1, "chunk level boundary jump")
            bits.extend(chunk["bits"])
            levels.extend(bytes((chunk["level"],)) * chunk["decisions"])
            expected_begin += chunk["decisions"]
            previous_level = chunk["level"]
        source = _metadata_stream_to_source(metadata_row, bytes(bits), bytes(levels))
        validated = validate_stream(source, expert)
        require(list(validated["level_counts"]) == metadata_row["level_counts"], "decoded level counts")
        local_coverage += source.local_weight_count
        decoded_sources.append(source)
    require(local_coverage == weights, "decoded expert weight coverage")
    _validate_expert_role_coverage(decoded_sources, expert, weights)
    return {
        "expert_ordinal": expert, "source_weights": weights,
        "physical_bytes": total, "physical_rate_bpw": float(Fraction(8 * total, weights)),
        "audit_receipt_sha256": receipt_sha.hex(),
        "candidate_sha256": candidate_sha.hex(),
        "reconstruction_sha256": reconstruction_sha.hex(),
        "semantic_state": semantic_state,
        "streams": tuple(decoded_sources), "chunks": tuple(chunks),
        "model_bytes": 0, "exception_bytes": 0,
    }


def packet_rate_bounds(
    decoded_packets: Sequence[Mapping[str, Any]], *, catalog_bytes: int = PAGE_BYTES,
) -> dict[str, Any]:
    """Derive unconditional floor, raw fallback, actual rate, and BM threshold.

    The selected-decision count is reported only as a literal-symbol count. It
    is never substituted for the audited arithmetic rate.
    """

    require(bool(decoded_packets), "packet set")
    require(type(catalog_bytes) is int and catalog_bytes >= 0, "catalog bytes")
    weights = sum(int(row["source_weights"]) for row in decoded_packets)
    actual = sum(int(row["physical_bytes"]) for row in decoded_packets)
    floor_total = 0
    raw_total = 0
    decisions = 0
    chunks = 0
    actual_complexity = 0
    for decoded in decoded_packets:
        metadata_record = {
            "schema": "strata-sc-gf2-expert-metadata-v1",
            "format_scope": FORMAT_SCOPE,
            "owner_descriptor_bits": OWNER_DESCRIPTOR_BITS,
            "recurrence_scope": RECURRENCE_SCOPE,
            "expert_ordinal": decoded["expert_ordinal"],
            "source_weights": decoded["source_weights"],
            "chunk_decisions": CHUNK_DECISIONS,
            "six_level_major": True,
            "semantic_state_hex": decoded["semantic_state"].hex(),
            "semantic_state_sha256": sha256(decoded["semantic_state"]),
            "streams": [
                _stream_metadata(source, validate_stream(source, decoded["expert_ordinal"]))
                for source in decoded["streams"]
            ],
        }
        metadata_bytes = len(canonical_json(metadata_record))
        chunk_count = len(decoded["chunks"])
        selected = sum(chunk["decisions"] for chunk in decoded["chunks"])
        floor_payload = chunk_count * CHUNK_HEADER.size
        raw_payload = floor_payload + sum((chunk["decisions"] + 7) // 8 for chunk in decoded["chunks"])
        prefix = align_up(EXPERT_HEADER_BYTES + metadata_bytes, 64)
        floor_total += align_up(prefix + floor_payload, PAGE_BYTES)
        raw_total += align_up(prefix + raw_payload, PAGE_BYTES)
        decisions += selected
        chunks += chunk_count
        actual_complexity += sum(chunk["complexity"] for chunk in decoded["chunks"])
    cap_bytes = (5 * weights) // 16
    cap_exact = (5 * weights) % 16 == 0
    floor_bits_unaligned = sum(
        EXPERT_HEADER_BYTES * 8
        + len(canonical_json({
            "schema": "strata-sc-gf2-expert-metadata-v1",
            "format_scope": FORMAT_SCOPE,
            "owner_descriptor_bits": OWNER_DESCRIPTOR_BITS,
            "recurrence_scope": RECURRENCE_SCOPE,
            "expert_ordinal": decoded["expert_ordinal"],
            "source_weights": decoded["source_weights"],
            "chunk_decisions": CHUNK_DECISIONS,
            "six_level_major": True,
            "semantic_state_hex": decoded["semantic_state"].hex(),
            "semantic_state_sha256": sha256(decoded["semantic_state"]),
            "streams": [_stream_metadata(source, validate_stream(source, decoded["expert_ordinal"])) for source in decoded["streams"]],
        })) * 8
        + len(decoded["chunks"]) * CHUNK_HEADER.size * 8
        for decoded in decoded_packets
    )
    maximum_sum_complexity_before_alignment = max(
        -1, (8 * (cap_bytes - catalog_bytes) - floor_bits_unaligned) // 2,
    )
    floor_with_catalog = catalog_bytes + floor_total
    raw_with_catalog = catalog_bytes + raw_total
    actual_with_catalog = catalog_bytes + actual
    return {
        "source_weights": weights,
        "selected_sc_decisions_with_expert_duplication": decisions,
        "selected_decisions_per_weight_is_not_a_rate": float(Fraction(decisions, weights)),
        "chunk_count": chunks,
        "catalog_bytes": catalog_bytes,
        "unconditional_zero_complexity_floor_bytes": floor_with_catalog,
        "unconditional_zero_complexity_floor_bpw": float(Fraction(8 * floor_with_catalog, weights)),
        "raw_fallback_bytes": raw_with_catalog,
        "raw_fallback_bpw": float(Fraction(8 * raw_with_catalog, weights)),
        "actual_packet_bytes": actual_with_catalog,
        "actual_packet_bpw": float(Fraction(8 * actual_with_catalog, weights)),
        "sum_exact_linear_complexity": actual_complexity,
        "optimistic_maximum_sum_linear_complexity_at_2p5_necessary_not_sufficient": maximum_sum_complexity_before_alignment,
        "exact_2p5_byte_cap_exists": cap_exact,
        "exact_2p5_byte_cap": cap_bytes if cap_exact else None,
        "zero_complexity_floor_can_fit_2p5": cap_exact and floor_with_catalog <= cap_bytes,
        "raw_fallback_can_fit_2p5": cap_exact and raw_with_catalog <= cap_bytes,
        "actual_can_fit_2p5": cap_exact and actual_with_catalog <= cap_bytes,
        "selected_decisions_are_arithmetic_coded_in_current_strata": True,
        "comparison_baseline_must_be_audited_literal_arithmetic_object": True,
        "shared_model_bytes": 0,
        "exception_bytes": 0,
        "exceptions_or_discrepancies_implemented": False,
        "format_scope": FORMAT_SCOPE,
        "owner_descriptor_bits": OWNER_DESCRIPTOR_BITS,
        "universal_successor_requires_variable_expert_cardinality_and_charged_descriptor": True,
        "recurrence_scope": RECURRENCE_SCOPE,
        "recurrences_longer_than_4096_can_be_missed": True,
        "negative_result_is_bounded_to_frozen_chunking": True,
    }
