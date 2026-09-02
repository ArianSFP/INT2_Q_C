#!/usr/bin/env python3
"""Canonical UWFA-SC v7 literal container and routed-read ledger.

The v7 ABI uses one fixed 32-byte little-endian owner set everywhere. It
supports 1..256 experts, serializes exact per-owner source intervals, validates
all size fields before dependent work, and emits one byte-partition ledger.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence


MAGIC = b"UWFCV7\x00\x00"
DIRECTORY_MAGIC = b"UWFDIR4\x00"
REGION_MAGIC = b"UWFREG4\x00"
FRAME_MAGIC = b"UWFFRM4\x00"
VERSION = 4
HEADER_BYTES = 4096
DIRECTORY_RECORD_BYTES = 256
REGION_HEADER_BYTES = 256
FRAME_HEADER_BYTES = 256
CONTRIBUTION_RECORD_BYTES = 24
OWNER_SET_BYTES = 32
PAGE_BYTES = 4096
MAX_EXPERTS = 256
MAX_STREAMS = 65_536
MAX_REGIONS = 65_536
MAX_WEIGHTS = 1 << 50
MAX_SYMBOLS = 1 << 54
MAX_LOGICAL_BITS = 1 << 56
MAX_DIMENSION = 1 << 24
MAX_CONTRIBUTIONS_PER_STREAM = 3 * MAX_EXPERTS
MAX_MODEL_BYTES = 1 << 26
MAX_SEMANTIC_BYTES = 1 << 26
MAX_IMMUTABLE_BYTES = 1 << 26
MAX_FRAME_BYTES = 1 << 26
MAX_CONTAINER_BYTES = 1 << 40
MAX_READER_CHUNK_BYTES = 1 << 20
RATE_MIN_NUMERATOR = 43
RATE_MIN_DENOMINATOR = 20

_HEADER_SEAL_BEGIN = 416
_HEADER_SEAL_END = 448
_HEADER_BINDINGS = (
    "baseline_plan_sha256",
    "baseline_score_sha256",
    "universal_decoder_sha256",
    "producer_manifest_sha256",
    "audit_bootstrap_sha256",
    "source_full_geometry_sha256",
    "source_structural_geometry_sha256",
    "extraction_program_sha256",
    "universal_adapter_sha256",
    "pipeline_sha256",
    "source_snapshot_root_sha256",
    "source_preflight_receipt_sha256",
)
_HEADER_BINDINGS_BEGIN = 448
_HEADER_CRC_OFFSET = _HEADER_BINDINGS_BEGIN + 32 * len(_HEADER_BINDINGS)


@dataclass(frozen=True)
class OwnerContribution:
    expert: int
    role: str
    source_offset: int
    weight_count: int


@dataclass(frozen=True)
class StreamSpec:
    ordinal: int
    symbols: int
    logical_bits: int
    payload: bytes
    source_digest: str
    profile_q: int
    decoder_scale: float
    role: str
    group_rows: int
    group_cols: int
    owner_contributions: tuple[OwnerContribution, ...]

    @property
    def source_weights(self) -> int:
        return sum(row.weight_count for row in self.owner_contributions)


@dataclass(frozen=True)
class RegionSpec:
    owner_set: bytes
    streams: tuple[StreamSpec, ...]


def _sha(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _hash_reader_range(reader: Any, begin: int, length: int) -> bytes:
    begin = _integer(begin, "hash range begin", 0, MAX_CONTAINER_BYTES)
    length = _integer(length, "hash range length", 0, MAX_CONTAINER_BYTES)
    _checked_add(begin, length, "hash range end", MAX_CONTAINER_BYTES)
    digest = hashlib.sha256()
    cursor = begin
    remaining = length
    while remaining:
        amount = min(MAX_READER_CHUNK_BYTES, remaining)
        digest.update(reader.read(cursor, amount))
        cursor += amount
        remaining -= amount
    return digest.digest()


def _range_is_zero(reader: Any, begin: int, length: int) -> bool:
    begin = _integer(begin, "zero range begin", 0, MAX_CONTAINER_BYTES)
    length = _integer(length, "zero range length", 0, MAX_CONTAINER_BYTES)
    _checked_add(begin, length, "zero range end", MAX_CONTAINER_BYTES)
    cursor = begin
    remaining = length
    while remaining:
        amount = min(MAX_READER_CHUNK_BYTES, remaining)
        if any(reader.read(cursor, amount)):
            return False
        cursor += amount
        remaining -= amount
    return True


def _hex32(value: str, label: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not hexadecimal") from exc
    if len(raw) != 32:
        raise ValueError(f"{label} digest geometry")
    return raw


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} outside exact integer bound")
    return value


def _checked_add(left: int, right: int, label: str, maximum: int = MAX_CONTAINER_BYTES) -> int:
    left = _integer(left, f"{label} left", 0, maximum)
    right = _integer(right, f"{label} right", 0, maximum)
    if left > maximum - right:
        raise ValueError(f"{label} addition overflow")
    return left + right


def _checked_mul(left: int, right: int, label: str, maximum: int = MAX_CONTAINER_BYTES) -> int:
    left = _integer(left, f"{label} left", 0, maximum)
    right = _integer(right, f"{label} right", 0, maximum)
    if right and left > maximum // right:
        raise ValueError(f"{label} multiplication overflow")
    return left * right


def _align(value: int, alignment: int = PAGE_BYTES) -> int:
    value = _integer(value, "alignment value", 0, MAX_CONTAINER_BYTES)
    alignment = _integer(alignment, "alignment", 1, PAGE_BYTES)
    extra = alignment - 1
    if value > MAX_CONTAINER_BYTES - extra:
        raise ValueError("alignment overflow")
    return ((value + extra) // alignment) * alignment


def measure_literal_container_layout(
    *,
    semantic_packet_bytes: int,
    immutable_state_bytes: int,
    model_packet_bytes: int,
    weights: int,
    experts: int,
    regions: Sequence[Mapping[str, Any]],
    minimum_rate_numerator: int = RATE_MIN_NUMERATOR,
    minimum_rate_denominator: int = RATE_MIN_DENOMINATOR,
) -> dict[str, Any]:
    """Measure the one frozen literal grammar without allocating payloads.

    This is the sole layout arithmetic used by both candidate selection and
    the real serializer.  Each frame row contains only ``logical_bits``,
    ``payload_bytes`` and ``contribution_count``; the payload length must be
    exactly ``ceil(logical_bits/8)``.  The serializer asserts that its emitted
    byte length and every region/frame measurement equal this result.
    """
    expert_count = _validate_experts(experts)
    weights = _integer(weights, "layout weights", 1, MAX_WEIGHTS)
    semantic_bytes = _integer(semantic_packet_bytes, "layout semantic bytes", 1, MAX_SEMANTIC_BYTES)
    immutable_bytes = _integer(immutable_state_bytes, "layout immutable bytes", 0, MAX_IMMUTABLE_BYTES)
    model_bytes = _integer(model_packet_bytes, "layout model bytes", 1, MAX_MODEL_BYTES)
    rate_num = _integer(minimum_rate_numerator, "layout rate numerator", 0, 10_000)
    rate_den = _integer(minimum_rate_denominator, "layout rate denominator", 1, 10_000)
    if (rate_num, rate_den) != (RATE_MIN_NUMERATOR, RATE_MIN_DENOMINATOR):
        raise ValueError("layout rate floor must be frozen 43/20")
    if not isinstance(regions, (tuple, list)) or not 1 <= len(regions) <= MAX_REGIONS:
        raise ValueError("layout region count")
    owner_sets: list[bytes] = []
    region_frame_bytes: list[list[int]] = []
    stream_count = 0
    for region in regions:
        if not isinstance(region, Mapping) or set(region) != {"owner_set", "frames"}:
            raise ValueError("layout region schema")
        owner_set = region["owner_set"]
        owner_ordinals(owner_set, expert_count)
        if owner_set in owner_sets:
            raise ValueError("layout duplicate owner region")
        owner_sets.append(owner_set)
        frames = region["frames"]
        if not isinstance(frames, (tuple, list)) or not frames:
            raise ValueError("layout empty region")
        measured_frames: list[int] = []
        for frame in frames:
            if not isinstance(frame, Mapping) or set(frame) != {
                "logical_bits", "payload_bytes", "contribution_count"
            }:
                raise ValueError("layout frame schema")
            logical = _integer(frame["logical_bits"], "layout logical bits", 1, MAX_LOGICAL_BITS)
            payload = _integer(frame["payload_bytes"], "layout payload bytes", 1, MAX_FRAME_BYTES)
            contributions = _integer(
                frame["contribution_count"], "layout contribution count", 1,
                MAX_CONTRIBUTIONS_PER_STREAM,
            )
            if payload != _ceil_div(logical, 8, "layout payload bytes"):
                raise ValueError("layout payload/logical mismatch")
            metadata = _align(
                _checked_add(
                    FRAME_HEADER_BYTES,
                    _checked_mul(CONTRIBUTION_RECORD_BYTES, contributions, "layout contribution bytes"),
                    "layout frame metadata",
                ),
                64,
            )
            frame_bytes = _align(_checked_add(metadata, payload, "layout frame end"), 64)
            if frame_bytes > MAX_FRAME_BYTES:
                raise ValueError("layout frame bound")
            measured_frames.append(frame_bytes)
        if stream_count > MAX_STREAMS - len(measured_frames):
            raise ValueError("layout stream count")
        stream_count += len(measured_frames)
        region_frame_bytes.append(measured_frames)
    expected_owner_order = sorted(owner_sets, key=lambda value: _owner_sort_key(value, expert_count))
    if owner_sets != expected_owner_order:
        raise ValueError("layout noncanonical owner-region order")
    directory_bytes = _checked_mul(stream_count, DIRECTORY_RECORD_BYTES, "layout directory bytes")
    semantic_offset = HEADER_BYTES
    immutable_offset = _align(_checked_add(semantic_offset, semantic_bytes, "layout semantic end"), 64)
    model_offset = _align(_checked_add(immutable_offset, immutable_bytes, "layout immutable end"), PAGE_BYTES)
    directory_offset = _align(_checked_add(model_offset, model_bytes, "layout model end"), PAGE_BYTES)
    shared_bytes = _align(_checked_add(directory_offset, directory_bytes, "layout directory end"), PAGE_BYTES)
    region_content_bytes = [
        _checked_add(REGION_HEADER_BYTES, sum(frames), "layout region content")
        for frames in region_frame_bytes
    ]
    region_lengths = [_align(value, PAGE_BYTES) for value in region_content_bytes]
    current_total = shared_bytes
    for length in region_lengths:
        current_total = _checked_add(current_total, length, "layout region sum")
    rate_bits_numerator = _checked_mul(weights, rate_num, "layout rate-floor weight product", MAX_WEIGHTS * 10_000)
    minimum_total = _ceil_div(rate_bits_numerator, 8 * rate_den, "layout rate-floor bytes")
    required_padding_bytes = max(0, minimum_total - current_total)
    padding_pages = _ceil_div(required_padding_bytes, PAGE_BYTES, "layout rate-floor padding pages")
    pages_each, leading_extra = divmod(padding_pages, len(region_lengths))
    for index in range(len(region_lengths)):
        pages = pages_each + (1 if index < leading_extra else 0)
        region_lengths[index] = _checked_add(
            region_lengths[index],
            _checked_mul(pages, PAGE_BYTES, "layout rate-floor region padding", MAX_CONTAINER_BYTES),
            "layout padded region",
        )
    total_bytes = shared_bytes
    for length in region_lengths:
        total_bytes = _checked_add(total_bytes, length, "layout total bytes")
    return {
        "semantic_offset": semantic_offset,
        "immutable_offset": immutable_offset,
        "model_offset": model_offset,
        "directory_offset": directory_offset,
        "directory_bytes": directory_bytes,
        "shared_bytes": shared_bytes,
        "stream_count": stream_count,
        "region_frame_bytes": region_frame_bytes,
        "region_content_bytes": region_content_bytes,
        "region_bytes": region_lengths,
        "rate_floor_padding_pages": padding_pages,
        "total_bytes": total_bytes,
    }


def _ceil_div(numerator: int, denominator: int, label: str) -> int:
    numerator = _integer(numerator, f"{label} numerator", 0, MAX_WEIGHTS * 100)
    denominator = _integer(denominator, f"{label} denominator", 1, MAX_WEIGHTS * 100)
    return numerator // denominator + (1 if numerator % denominator else 0)


def _validate_experts(experts: Any) -> int:
    return _integer(experts, "experts", 1, MAX_EXPERTS)


def owner_set_from_ordinals(experts: Any, owners: Sequence[Any]) -> bytes:
    expert_count = _validate_experts(experts)
    if not isinstance(owners, (tuple, list)) or not 1 <= len(owners) <= expert_count:
        raise ValueError("owner ordinal list geometry")
    values = [_integer(row, "owner ordinal", 0, expert_count - 1) for row in owners]
    if values != sorted(set(values)):
        raise ValueError("owner ordinals must be unique and sorted")
    result = bytearray(OWNER_SET_BYTES)
    for ordinal in values:
        result[ordinal >> 3] |= 1 << (ordinal & 7)
    return bytes(result)


def owner_ordinals(owner_set: Any, experts: Any) -> tuple[int, ...]:
    expert_count = _validate_experts(experts)
    if not isinstance(owner_set, bytes) or len(owner_set) != OWNER_SET_BYTES:
        raise ValueError("owner set must use fixed 32-byte ABI")
    used = (expert_count + 7) // 8
    if any(owner_set[used:]):
        raise ValueError("owner set has bytes above expert universe")
    if expert_count & 7 and owner_set[used - 1] & ~((1 << (expert_count & 7)) - 1):
        raise ValueError("owner set has noncanonical high bits")
    if not any(owner_set[:used]):
        raise ValueError("empty owner set")
    return tuple(index for index in range(expert_count) if owner_set[index >> 3] & (1 << (index & 7)))


def _owner_sort_key(owner_set: bytes, experts: int) -> tuple[Any, ...]:
    owners = owner_ordinals(owner_set, experts)
    return (len(owners) != 1, owners)


def _role_bytes(role: Any) -> bytes:
    if not isinstance(role, str) or role not in {"gate", "up", "down", "mixed"}:
        raise ValueError("universal SwiGLU stream role")
    raw = role.encode("ascii")
    return raw + bytes(32 - len(raw))


def _parse_role(raw: bytes) -> str:
    if len(raw) != 32:
        raise ValueError("role field geometry")
    end = raw.find(b"\x00")
    if end <= 0 or any(raw[end:]):
        raise ValueError("noncanonical role field")
    try:
        role = raw[:end].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("role field encoding") from exc
    if _role_bytes(role) != raw:
        raise ValueError("role field value")
    return role


def _validate_stream(spec: StreamSpec, owner_set: bytes, experts: int) -> tuple[int, ...]:
    _integer(spec.ordinal, "stream ordinal", 0, MAX_STREAMS - 1)
    _integer(spec.symbols, "stream symbols", 1, MAX_SYMBOLS)
    logical = _integer(spec.logical_bits, "stream logical bits", 1, MAX_LOGICAL_BITS)
    expected_payload = _ceil_div(logical, 8, "payload bytes")
    if not isinstance(spec.payload, bytes) or len(spec.payload) != expected_payload or len(spec.payload) > MAX_FRAME_BYTES - FRAME_HEADER_BYTES:
        raise ValueError("payload/logical-bit geometry")
    if logical & 7 and spec.payload[-1] & ((1 << (8 - (logical & 7))) - 1):
        raise ValueError("nonzero terminal arithmetic padding")
    _hex32(spec.source_digest, "source digest")
    _integer(spec.profile_q, "profile id", 0, 0xFFFF)
    if not math.isfinite(spec.decoder_scale) or spec.decoder_scale <= 0.0:
        raise ValueError("decoder scale")
    _role_bytes(spec.role)
    rows = _integer(spec.group_rows, "group rows", 1, MAX_DIMENSION)
    cols = _integer(spec.group_cols, "group cols", 1, MAX_DIMENSION)
    group_product = _checked_mul(rows, cols, "group shape", MAX_WEIGHTS)
    owners = owner_ordinals(owner_set, experts)
    if not isinstance(spec.owner_contributions, tuple) or not 1 <= len(spec.owner_contributions) <= MAX_CONTRIBUTIONS_PER_STREAM:
        raise ValueError("owner contribution cardinality")
    observed: list[int] = []
    previous_key: tuple[int, int, int] | None = None
    total = 0
    for contribution in spec.owner_contributions:
        if not isinstance(contribution, OwnerContribution):
            raise ValueError("owner contribution type")
        expert = _integer(contribution.expert, "contribution expert", 0, experts - 1)
        role_bytes = _role_bytes(contribution.role)
        if contribution.role == "mixed":
            raise ValueError("contribution role must be Gate, Up, or Down")
        begin = _integer(contribution.source_offset, "contribution offset", 0, MAX_WEIGHTS - 1)
        count = _integer(contribution.weight_count, "contribution weights", 1, MAX_WEIGHTS)
        key = (expert, ("gate", "up", "down").index(contribution.role), begin)
        if previous_key is not None and key <= previous_key:
            raise ValueError("noncanonical contribution order")
        previous_key = key
        _checked_add(begin, count, "contribution interval", MAX_WEIGHTS)
        total = _checked_add(total, count, "stream weight sum", MAX_WEIGHTS)
        observed.append(expert)
    if tuple(sorted(set(observed))) != owners:
        raise ValueError("contribution experts differ from owner set")
    if total != group_product:
        raise ValueError("group shape does not conserve source weights")
    return owners


def _contribution_bytes(spec: StreamSpec) -> bytes:
    packet = bytearray()
    for row in spec.owner_contributions:
        role_id = ("gate", "up", "down").index(row.role) + 1
        packet.extend(struct.pack("<IIQQ", row.expert, role_id, row.source_offset, row.weight_count))
    return bytes(packet)


def _frame_packet(spec: StreamSpec, owner_set: bytes, experts: int, region_ordinal: int) -> bytes:
    _validate_stream(spec, owner_set, experts)
    contributions = _contribution_bytes(spec)
    metadata_bytes = _align(FRAME_HEADER_BYTES + len(contributions), 64)
    payload_offset = metadata_bytes
    frame_bytes = _align(_checked_add(payload_offset, len(spec.payload), "frame payload end"), 64)
    if frame_bytes > MAX_FRAME_BYTES:
        raise ValueError("frame exceeds frozen byte bound")
    header = bytearray(FRAME_HEADER_BYTES)
    struct.pack_into(
        "<8sIIHHIQQQQQQdQQ",
        header,
        0,
        FRAME_MAGIC,
        spec.ordinal,
        region_ordinal,
        spec.profile_q,
        len(spec.owner_contributions),
        0,
        spec.symbols,
        spec.logical_bits,
        len(spec.payload),
        spec.source_weights,
        spec.group_rows,
        spec.group_cols,
        spec.decoder_scale,
        metadata_bytes,
        frame_bytes,
    )
    header[96:128] = owner_set
    header[128:160] = _hex32(spec.source_digest, "source digest")
    header[160:192] = _sha(spec.payload)
    header[192:224] = _role_bytes(spec.role)
    header[224:256] = _sha(bytes(header[:224]))
    result = bytes(header) + contributions
    result += bytes(metadata_bytes - len(result))
    result += spec.payload
    result += bytes(frame_bytes - len(result))
    if len(result) != frame_bytes:
        raise AssertionError("frame assembly")
    return result


def _directory_record(spec: StreamSpec, owner_set: bytes, region_ordinal: int, region_offset: int, region_bytes: int, frame_offset: int, frame_bytes: int) -> bytes:
    row = bytearray(DIRECTORY_RECORD_BYTES)
    metadata_bytes = _align(FRAME_HEADER_BYTES + CONTRIBUTION_RECORD_BYTES * len(spec.owner_contributions), 64)
    payload_offset = _checked_add(frame_offset, metadata_bytes, "directory payload offset")
    struct.pack_into(
        "<8sIIHHIQQQQQQQQQQQd",
        row,
        0,
        DIRECTORY_MAGIC,
        spec.ordinal,
        region_ordinal,
        spec.profile_q,
        len(spec.owner_contributions),
        0,
        spec.symbols,
        spec.logical_bits,
        payload_offset,
        len(spec.payload),
        region_offset,
        region_bytes,
        frame_offset,
        frame_bytes,
        spec.source_weights,
        spec.group_rows,
        spec.group_cols,
        spec.decoder_scale,
    )
    row[120:152] = owner_set
    row[152:184] = _hex32(spec.source_digest, "source digest")
    row[184:216] = _sha(spec.payload)
    row[216:248] = _role_bytes(spec.role)
    row[248:256] = _sha(bytes(row[:248]))[:8]
    return bytes(row)


def _region_header(region_ordinal: int, owner_set: bytes, stream_count: int, region_bytes: int, content_bytes: int, frame_area: bytes) -> bytes:
    header = bytearray(REGION_HEADER_BYTES)
    struct.pack_into(
        "<8sIIHHIQQQ",
        header,
        0,
        REGION_MAGIC,
        region_ordinal,
        stream_count,
        OWNER_SET_BYTES,
        0,
        0,
        region_bytes,
        content_bytes,
        len(frame_area),
    )
    header[48:80] = owner_set
    header[80:112] = _sha(frame_area)
    header[112:144] = _sha(bytes(header[:112]))
    return bytes(header)


def _header(*, weights: int, experts: int, streams: int, regions: int, baseline_object_bytes: int, audited_relative_mse: float, semantic_offset: int, semantic_bytes: int, immutable_offset: int, immutable_bytes: int, model_offset: int, model_bytes: int, directory_offset: int, directory_bytes: int, shared_bytes: int, total_bytes: int, rate_num: int, rate_den: int, baseline_artifact_sha256: str, reconstruction_sha256: str, audit_binding_sha256: str, semantic_sha: bytes, immutable_sha: bytes, model_sha: bytes, directory_sha: bytes, body_sha: bytes, binding_hashes: Mapping[str, str]) -> bytes:
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<8sHHIIQIIHHII", header, 0, MAGIC, VERSION, HEADER_BYTES, PAGE_BYTES, 0, weights, experts, streams, OWNER_SET_BYTES, DIRECTORY_RECORD_BYTES, regions, 0)
    struct.pack_into("<Qd", header, 48, baseline_object_bytes, audited_relative_mse)
    struct.pack_into("<QQQQQQQQQQ", header, 64, semantic_offset, semantic_bytes, immutable_offset, immutable_bytes, model_offset, model_bytes, directory_offset, directory_bytes, shared_bytes, total_bytes)
    struct.pack_into("<QQ", header, 144, rate_num, rate_den)
    header[160:192] = _hex32(baseline_artifact_sha256, "baseline artifact")
    header[192:224] = _hex32(reconstruction_sha256, "reconstruction")
    header[224:256] = _hex32(audit_binding_sha256, "audit binding")
    header[256:288] = semantic_sha
    header[288:320] = immutable_sha
    header[320:352] = model_sha
    header[352:384] = directory_sha
    header[384:416] = body_sha
    if set(binding_hashes) != set(_HEADER_BINDINGS):
        raise ValueError("exact v7 container binding fields required")
    for index, name in enumerate(_HEADER_BINDINGS):
        begin = _HEADER_BINDINGS_BEGIN + 32 * index
        header[begin:begin + 32] = _hex32(binding_hashes[name], name)
    header[_HEADER_SEAL_BEGIN:_HEADER_SEAL_END] = bytes(32)
    struct.pack_into("<I", header, _HEADER_CRC_OFFSET, 0)
    header[_HEADER_SEAL_BEGIN:_HEADER_SEAL_END] = _sha(bytes(header))
    struct.pack_into("<I", header, _HEADER_CRC_OFFSET, zlib.crc32(header) & 0xFFFFFFFF)
    return bytes(header)


def build_container(
    common: Any,
    semantic_codec: Any,
    *,
    model_packet: bytes,
    semantic_packet: bytes,
    immutable_state: bytes,
    regions: Sequence[RegionSpec],
    weights: int,
    experts: int,
    baseline_object_bytes: int,
    audited_relative_mse: float,
    baseline_artifact_sha256: str,
    reconstruction_sha256: str,
    audit_binding_sha256: str,
    binding_hashes: Mapping[str, str],
    minimum_rate_numerator: int = RATE_MIN_NUMERATOR,
    minimum_rate_denominator: int = RATE_MIN_DENOMINATOR,
) -> tuple[bytes, dict[str, Any]]:
    expert_count = _validate_experts(experts)
    weights = _integer(weights, "source weights", 1, MAX_WEIGHTS)
    baseline_object_bytes = _integer(baseline_object_bytes, "baseline bytes", 1, MAX_CONTAINER_BYTES)
    if not math.isfinite(audited_relative_mse) or audited_relative_mse <= 0.0:
        raise ValueError("audited relative MSE")
    rate_num = _integer(minimum_rate_numerator, "rate numerator", 0, 10_000)
    rate_den = _integer(minimum_rate_denominator, "rate denominator", 1, 10_000)
    if (rate_num, rate_den) != (RATE_MIN_NUMERATOR, RATE_MIN_DENOMINATOR):
        raise ValueError("builder rate floor must be frozen 43/20")
    if not isinstance(model_packet, bytes) or not 1 <= len(model_packet) <= MAX_MODEL_BYTES:
        raise ValueError("model packet bound")
    candidate, frequencies = common.deserialize_model(model_packet)
    if common.serialize_model(candidate, frequencies) != model_packet:
        raise ValueError("noncanonical serialized model")
    if not isinstance(semantic_packet, bytes) or not 1 <= len(semantic_packet) <= MAX_SEMANTIC_BYTES:
        raise ValueError("semantic packet bound")
    semantics = semantic_codec.parse_semantic_packet(semantic_packet)
    if int(semantics["experts"]) != expert_count or int(semantics["source_weights"]) != weights:
        raise ValueError("semantic/header expert or weight mismatch")
    if not isinstance(immutable_state, bytes) or len(immutable_state) > MAX_IMMUTABLE_BYTES:
        raise ValueError("immutable state bound")
    if not isinstance(regions, (tuple, list)) or not 1 <= len(regions) <= MAX_REGIONS:
        raise ValueError("region count bound")
    region_owner_sets: list[bytes] = []
    flat: list[StreamSpec] = []
    frame_areas: list[bytes] = []
    for region_ordinal, region in enumerate(regions):
        if not isinstance(region, RegionSpec) or not isinstance(region.streams, tuple) or not region.streams:
            raise ValueError("region specification")
        owner_ordinals(region.owner_set, expert_count)
        if region.owner_set in region_owner_sets:
            raise ValueError("duplicate owner-set region")
        region_owner_sets.append(region.owner_set)
        if len(flat) > MAX_STREAMS - len(region.streams):
            raise ValueError("stream count bound before frame loop")
        if tuple(sorted(region.streams, key=lambda row: row.ordinal)) != region.streams:
            raise ValueError("frames must be canonical ordinal order")
        packets = [_frame_packet(row, region.owner_set, expert_count, region_ordinal) for row in region.streams]
        frame_areas.append(b"".join(packets))
        flat.extend(region.streams)
    expected_owner_order = sorted(region_owner_sets, key=lambda value: _owner_sort_key(value, expert_count))
    if region_owner_sets != expected_owner_order:
        raise ValueError("noncanonical region owner-set order")
    if not 1 <= len(flat) <= MAX_STREAMS or sorted(row.ordinal for row in flat) != list(range(len(flat))):
        raise ValueError("stream ordinals must be one complete canonical range")
    semantic_codec.validate_stream_coverage(
        semantics,
        [
            {
                "ordinal": row.ordinal,
                "role": row.role,
                "owners": owner_ordinals(owner_set, expert_count),
                "owner_contributions": tuple({"expert": item.expert, "role": item.role, "source_offset": item.source_offset, "weight_count": item.weight_count} for item in row.owner_contributions),
                "source_weights": row.source_weights,
                "group_rows": row.group_rows,
                "group_cols": row.group_cols,
            }
            for region, owner_set in zip(regions, region_owner_sets, strict=True)
            for row in region.streams
        ],
    )
    measured = measure_literal_container_layout(
        semantic_packet_bytes=len(semantic_packet),
        immutable_state_bytes=len(immutable_state),
        model_packet_bytes=len(model_packet),
        weights=weights,
        experts=expert_count,
        regions=[
            {
                "owner_set": region.owner_set,
                "frames": [
                    {
                        "logical_bits": spec.logical_bits,
                        "payload_bytes": len(spec.payload),
                        "contribution_count": len(spec.owner_contributions),
                    }
                    for spec in region.streams
                ],
            }
            for region in regions
        ],
        minimum_rate_numerator=rate_num,
        minimum_rate_denominator=rate_den,
    )
    directory_bytes = int(measured["directory_bytes"])
    semantic_offset = int(measured["semantic_offset"])
    immutable_offset = int(measured["immutable_offset"])
    model_offset = int(measured["model_offset"])
    directory_offset = int(measured["directory_offset"])
    shared_bytes = int(measured["shared_bytes"])
    region_lengths = [int(value) for value in measured["region_bytes"]]
    for actual, expected_frames in zip(frame_areas, measured["region_frame_bytes"], strict=True):
        if len(actual) != sum(int(value) for value in expected_frames):
            raise AssertionError("serialized frame area differs from shared literal measurement")
    region_offsets: list[int] = []
    cursor = shared_bytes
    for length in region_lengths:
        region_offsets.append(cursor)
        cursor = _checked_add(cursor, length, "region placement")
    total_bytes = cursor
    if total_bytes != int(measured["total_bytes"]):
        raise AssertionError("container placement differs from shared literal measurement")
    directory_rows: dict[int, bytes] = {}
    region_packets: list[bytes] = []
    for region_ordinal, (region, area, region_offset, region_bytes) in enumerate(zip(regions, frame_areas, region_offsets, region_lengths, strict=True)):
        local = 0
        for spec in region.streams:
            frame = _frame_packet(spec, region.owner_set, expert_count, region_ordinal)
            frame_offset = _checked_add(region_offset + REGION_HEADER_BYTES, local, "frame offset")
            directory_rows[spec.ordinal] = _directory_record(spec, region.owner_set, region_ordinal, region_offset, region_bytes, frame_offset, len(frame))
            local = _checked_add(local, len(frame), "frame area")
        content_bytes = REGION_HEADER_BYTES + len(area)
        header = _region_header(region_ordinal, region.owner_set, len(region.streams), region_bytes, content_bytes, area)
        packet = header + area + bytes(region_bytes - content_bytes)
        if len(packet) != region_bytes:
            raise AssertionError("region assembly geometry")
        region_packets.append(packet)
    directory = b"".join(directory_rows[index] for index in range(len(flat)))
    shared_tail = bytearray(shared_bytes - HEADER_BYTES)
    shared_tail[semantic_offset - HEADER_BYTES:semantic_offset - HEADER_BYTES + len(semantic_packet)] = semantic_packet
    shared_tail[immutable_offset - HEADER_BYTES:immutable_offset - HEADER_BYTES + len(immutable_state)] = immutable_state
    shared_tail[model_offset - HEADER_BYTES:model_offset - HEADER_BYTES + len(model_packet)] = model_packet
    shared_tail[directory_offset - HEADER_BYTES:directory_offset - HEADER_BYTES + len(directory)] = directory
    body = bytes(shared_tail) + b"".join(region_packets)
    header = _header(
        weights=weights,
        experts=expert_count,
        streams=len(flat),
        regions=len(regions),
        baseline_object_bytes=baseline_object_bytes,
        audited_relative_mse=audited_relative_mse,
        semantic_offset=semantic_offset,
        semantic_bytes=len(semantic_packet),
        immutable_offset=immutable_offset,
        immutable_bytes=len(immutable_state),
        model_offset=model_offset,
        model_bytes=len(model_packet),
        directory_offset=directory_offset,
        directory_bytes=len(directory),
        shared_bytes=shared_bytes,
        total_bytes=total_bytes,
        rate_num=rate_num,
        rate_den=rate_den,
        baseline_artifact_sha256=baseline_artifact_sha256,
        reconstruction_sha256=reconstruction_sha256,
        audit_binding_sha256=audit_binding_sha256,
        semantic_sha=_sha(semantic_packet),
        immutable_sha=_sha(immutable_state),
        model_sha=_sha(model_packet),
        directory_sha=_sha(directory),
        body_sha=_sha(body),
        binding_hashes=binding_hashes,
    )
    container = header + body
    if len(container) != int(measured["total_bytes"]):
        raise AssertionError("serialized container differs from shared literal measurement")
    parsed = parse_container(common, semantic_codec, container)
    return container, physical_metrics(common, semantic_codec, parsed)


class MemoryReader:
    """Bounded seek/read interface used by both full and routed parsers."""

    def __init__(self, raw: bytes):
        if not isinstance(raw, bytes) or not HEADER_BYTES <= len(raw) <= MAX_CONTAINER_BYTES:
            raise ValueError("container byte envelope")
        self._raw = raw
        self.size = len(raw)

    def read(self, begin: int, length: int) -> bytes:
        begin = _integer(begin, "read begin", 0, self.size)
        length = _integer(length, "read length", 0, self.size)
        end = _checked_add(begin, length, "read end", self.size)
        if end > self.size:
            raise ValueError("read beyond bounded container")
        return self._raw[begin:end]


class InstrumentedReader(MemoryReader):
    """Fresh routed reader recording exact byte requests and page union."""

    def __init__(self, raw: bytes):
        super().__init__(raw)
        self.pages: set[int] = set()
        self.ranges: list[tuple[int, int]] = []

    def read(self, begin: int, length: int) -> bytes:
        result = super().read(begin, length)
        end = begin + length
        if length:
            self.pages.update(range(begin // PAGE_BYTES, (end - 1) // PAGE_BYTES + 1))
        self.ranges.append((begin, end))
        return result


class DescriptorReader:
    """Bounded reader over an already-authenticated regular-file descriptor.

    Paths are deliberately absent from this API. The external bootstrap owns
    no-follow path authentication; this class duplicates the held descriptor,
    fstats and bounds it before a read/allocation, and re-fstats on close.
    """

    def __init__(self, fd: int):
        if type(fd) is not int or fd < 0:
            raise ValueError("container descriptor")
        self._fd = os.dup(fd)
        try:
            info = os.fstat(self._fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("container descriptor is not regular")
            if not HEADER_BYTES <= info.st_size <= MAX_CONTAINER_BYTES:
                raise ValueError("descriptor container byte envelope")
            self._identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            self.size = int(info.st_size)
        except Exception:
            os.close(self._fd)
            self._fd = -1
            raise

    def read(self, begin: int, length: int) -> bytes:
        if self._fd < 0:
            raise ValueError("closed container descriptor")
        begin = _integer(begin, "descriptor read begin", 0, self.size)
        length = _integer(length, "descriptor read length", 0, self.size)
        end = _checked_add(begin, length, "descriptor read end", self.size)
        if end > self.size:
            raise ValueError("descriptor read beyond bounded container")
        if hasattr(os, "pread"):
            chunks = []
            cursor = begin
            remaining = length
            while remaining:
                chunk = os.pread(self._fd, remaining, cursor)
                if not chunk:
                    raise ValueError("short descriptor read")
                chunks.append(chunk)
                cursor += len(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)
        os.lseek(self._fd, begin, os.SEEK_SET)
        chunks = []
        remaining = length
        while remaining:
            chunk = os.read(self._fd, remaining)
            if not chunk:
                raise ValueError("short descriptor read")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def verify_stable(self) -> None:
        if self._fd < 0:
            raise ValueError("closed container descriptor")
        info = os.fstat(self._fd)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != self._identity:
            raise ValueError("container descriptor changed")

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return self._identity

    def close(self) -> None:
        if self._fd >= 0:
            self.verify_stable()
            os.close(self._fd)
            self._fd = -1


class InstrumentedDescriptorReader(DescriptorReader):
    """Fresh descriptor reader recording the exact routed page union."""

    def __init__(self, fd: int):
        super().__init__(fd)
        self.pages: set[int] = set()
        self.ranges: list[tuple[int, int]] = []

    def read(self, begin: int, length: int) -> bytes:
        result = super().read(begin, length)
        end = begin + length
        if length:
            self.pages.update(range(begin // PAGE_BYTES, (end - 1) // PAGE_BYTES + 1))
        self.ranges.append((begin, end))
        return result


class AuthenticatedDescriptorSource:
    """One full installation authentication scan plus fresh routed readers."""

    def __init__(self, fd: int, expected_sha256: str):
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("authenticated descriptor digest")
        try:
            bytes.fromhex(expected_sha256)
        except ValueError as exc:
            raise ValueError("authenticated descriptor digest encoding") from exc
        self._master = DescriptorReader(fd)
        digest = hashlib.sha256()
        cursor = 0
        authentication_ranges: list[tuple[int, int]] = []
        while cursor < self._master.size:
            amount = min(1 << 20, self._master.size - cursor)
            digest.update(self._master.read(cursor, amount))
            authentication_ranges.append((cursor, cursor + amount))
            cursor += amount
        self._master.verify_stable()
        if digest.hexdigest() != expected_sha256.lower():
            self._master.close()
            raise ValueError("authenticated descriptor content digest")
        self.container_sha256 = expected_sha256.lower()
        self.installation_authentication_scan_bytes = self._master.size
        self.installation_authentication_read_ranges = tuple(authentication_ranges)
        self.installation_authentication_touched_page_indices = tuple(
            range((self._master.size + PAGE_BYTES - 1) // PAGE_BYTES)
        )
        self.installation_authentication_touched_page_bytes = (
            len(self.installation_authentication_touched_page_indices) * PAGE_BYTES
        )
        self.size = self._master.size
        self.identity = self._master.identity

    def fresh_reader(self) -> InstrumentedDescriptorReader:
        reader = InstrumentedDescriptorReader(self._master._fd)
        if reader.identity != self.identity or reader.size != self.size:
            reader.close()
            raise ValueError("fresh routed descriptor identity")
        return reader

    def verify_stable(self) -> None:
        self._master.verify_stable()

    def close(self) -> None:
        self._master.close()


def _parse_header(header: bytes, file_size: int) -> dict[str, Any]:
    file_size = _integer(file_size, "bounded file size", HEADER_BYTES, MAX_CONTAINER_BYTES)
    if len(header) != HEADER_BYTES:
        raise ValueError("header read geometry")
    fields = struct.unpack_from("<8sHHIIQIIHHII", header, 0)
    magic, version, header_bytes, page_bytes, flags, weights, experts, streams, owner_bytes, directory_record_bytes, regions, reserved = fields
    if (magic, version, header_bytes, page_bytes, flags, owner_bytes, directory_record_bytes, reserved) != (MAGIC, VERSION, HEADER_BYTES, PAGE_BYTES, 0, OWNER_SET_BYTES, DIRECTORY_RECORD_BYTES, 0):
        raise ValueError("container header constants")
    weights = _integer(weights, "header weights", 1, MAX_WEIGHTS)
    experts = _validate_experts(experts)
    streams = _integer(streams, "header streams", 1, MAX_STREAMS)
    regions = _integer(regions, "header regions", 1, min(MAX_REGIONS, streams))
    expected_directory_bytes = _checked_mul(streams, DIRECTORY_RECORD_BYTES, "header directory bytes")
    baseline_object_bytes, audited_relative_mse = struct.unpack_from("<Qd", header, 48)
    baseline_object_bytes = _integer(baseline_object_bytes, "header baseline bytes", 1, MAX_CONTAINER_BYTES)
    if not math.isfinite(audited_relative_mse) or audited_relative_mse <= 0.0:
        raise ValueError("header audited MSE")
    section_values = struct.unpack_from("<QQQQQQQQQQ", header, 64)
    semantic_offset, semantic_bytes, immutable_offset, immutable_bytes, model_offset, model_bytes, directory_offset, directory_bytes, shared_bytes, total_bytes = section_values
    rate_num, rate_den = struct.unpack_from("<QQ", header, 144)
    rate_num = _integer(rate_num, "rate numerator", 0, 10_000)
    rate_den = _integer(rate_den, "rate denominator", 1, 10_000)
    if (rate_num, rate_den) != (RATE_MIN_NUMERATOR, RATE_MIN_DENOMINATOR):
        raise ValueError("header rate rational differs from frozen 43/20")
    semantic_bytes = _integer(semantic_bytes, "semantic bytes", 1, MAX_SEMANTIC_BYTES)
    immutable_bytes = _integer(immutable_bytes, "immutable bytes", 0, MAX_IMMUTABLE_BYTES)
    model_bytes = _integer(model_bytes, "model bytes", 1, MAX_MODEL_BYTES)
    if directory_bytes != expected_directory_bytes:
        raise ValueError("directory byte/count mismatch")
    if total_bytes != file_size:
        raise ValueError("header total/file-size mismatch")
    expected_semantic_offset = HEADER_BYTES
    expected_immutable_offset = _align(_checked_add(expected_semantic_offset, semantic_bytes, "semantic section end"), 64)
    expected_model_offset = _align(_checked_add(expected_immutable_offset, immutable_bytes, "immutable section end"), PAGE_BYTES)
    expected_directory_offset = _align(_checked_add(expected_model_offset, model_bytes, "model section end"), PAGE_BYTES)
    expected_shared_bytes = _align(_checked_add(expected_directory_offset, directory_bytes, "directory section end"), PAGE_BYTES)
    if (semantic_offset, immutable_offset, model_offset, directory_offset, shared_bytes) != (expected_semantic_offset, expected_immutable_offset, expected_model_offset, expected_directory_offset, expected_shared_bytes):
        raise ValueError("noncanonical shared section placement")
    if shared_bytes >= total_bytes or shared_bytes % PAGE_BYTES:
        raise ValueError("shared/region envelope")
    clean = bytearray(header)
    observed_seal = bytes(clean[_HEADER_SEAL_BEGIN:_HEADER_SEAL_END])
    observed_crc = struct.unpack_from("<I", clean, _HEADER_CRC_OFFSET)[0]
    struct.pack_into("<I", clean, _HEADER_CRC_OFFSET, 0)
    if observed_crc != zlib.crc32(clean) & 0xFFFFFFFF:
        raise ValueError("header CRC")
    clean[_HEADER_SEAL_BEGIN:_HEADER_SEAL_END] = bytes(32)
    if observed_seal != _sha(bytes(clean)):
        raise ValueError("header SHA seal")
    if any(header[_HEADER_CRC_OFFSET + 4:]):
        raise ValueError("header reserved bytes")
    return {
        "weights": weights,
        "experts": experts,
        "streams": streams,
        "region_count": regions,
        "baseline_object_bytes": baseline_object_bytes,
        "audited_relative_mse": audited_relative_mse,
        "semantic_offset": semantic_offset,
        "semantic_bytes": semantic_bytes,
        "immutable_offset": immutable_offset,
        "immutable_bytes": immutable_bytes,
        "model_offset": model_offset,
        "model_bytes": model_bytes,
        "directory_offset": directory_offset,
        "directory_bytes": directory_bytes,
        "shared_bytes": shared_bytes,
        "total_bytes": total_bytes,
        "minimum_rate_numerator": rate_num,
        "minimum_rate_denominator": rate_den,
        "baseline_artifact_sha256": header[160:192].hex(),
        "reconstruction_sha256": header[192:224].hex(),
        "audit_binding_sha256": header[224:256].hex(),
        "semantic_sha256": header[256:288],
        "immutable_sha256": header[288:320],
        "model_sha256": header[320:352],
        "directory_sha256": header[352:384],
        "body_sha256": header[384:416],
        "binding_hashes": {
            name: header[_HEADER_BINDINGS_BEGIN + 32 * index:_HEADER_BINDINGS_BEGIN + 32 * (index + 1)].hex()
            for index, name in enumerate(_HEADER_BINDINGS)
        },
    }


def _parse_directory_record(raw: bytes, header: Mapping[str, Any]) -> dict[str, Any]:
    if len(raw) != DIRECTORY_RECORD_BYTES:
        raise ValueError("directory record geometry")
    fields = struct.unpack_from("<8sIIHHIQQQQQQQQQQQd", raw, 0)
    magic, ordinal, region_ordinal, profile_q, contribution_count, reserved, symbols, logical_bits, payload_offset, payload_bytes, region_offset, region_bytes, frame_offset, frame_bytes, source_weights, group_rows, group_cols, decoder_scale = fields
    if magic != DIRECTORY_MAGIC or reserved != 0:
        raise ValueError("directory constants")
    ordinal = _integer(ordinal, "directory ordinal", 0, int(header["streams"]) - 1)
    region_ordinal = _integer(region_ordinal, "directory region", 0, int(header["region_count"]) - 1)
    profile_q = _integer(profile_q, "directory profile", 0, 0xFFFF)
    contribution_count = _integer(contribution_count, "directory contributions", 1, min(MAX_CONTRIBUTIONS_PER_STREAM, 3 * int(header["experts"])))
    symbols = _integer(symbols, "directory symbols", 1, MAX_SYMBOLS)
    logical_bits = _integer(logical_bits, "directory logical bits", 1, MAX_LOGICAL_BITS)
    expected_payload = _ceil_div(logical_bits, 8, "directory payload bytes")
    if payload_bytes != expected_payload:
        raise ValueError("directory payload/logical mismatch")
    source_weights = _integer(source_weights, "directory source weights", 1, MAX_WEIGHTS)
    group_rows = _integer(group_rows, "directory group rows", 1, MAX_DIMENSION)
    group_cols = _integer(group_cols, "directory group cols", 1, MAX_DIMENSION)
    if _checked_mul(group_rows, group_cols, "directory group shape", MAX_WEIGHTS) != source_weights:
        raise ValueError("directory source weight mismatch")
    if not math.isfinite(decoder_scale) or decoder_scale <= 0.0:
        raise ValueError("directory decoder scale")
    owner_set = bytes(raw[120:152])
    owners = owner_ordinals(owner_set, int(header["experts"]))
    if len(owners) > contribution_count:
        raise ValueError("directory has fewer contributions than owners")
    role = _parse_role(bytes(raw[216:248]))
    if raw[248:256] != _sha(raw[:248])[:8]:
        raise ValueError("directory row seal")
    total_bytes = int(header["total_bytes"])
    shared_bytes = int(header["shared_bytes"])
    region_offset = _integer(region_offset, "directory region offset", shared_bytes, total_bytes)
    region_bytes = _integer(region_bytes, "directory region bytes", PAGE_BYTES, total_bytes)
    frame_offset = _integer(frame_offset, "directory frame offset", shared_bytes, total_bytes)
    frame_bytes = _integer(frame_bytes, "directory frame bytes", FRAME_HEADER_BYTES, min(total_bytes, MAX_FRAME_BYTES))
    payload_offset = _integer(payload_offset, "directory payload offset", shared_bytes, total_bytes)
    if region_offset % PAGE_BYTES or region_bytes % PAGE_BYTES or frame_offset % 64 or frame_bytes % 64:
        raise ValueError("directory placement alignment")
    region_end = _checked_add(region_offset, region_bytes, "directory region end", total_bytes)
    frame_end = _checked_add(frame_offset, frame_bytes, "directory frame end", total_bytes)
    payload_end = _checked_add(payload_offset, payload_bytes, "directory payload end", total_bytes)
    if not region_offset + REGION_HEADER_BYTES <= frame_offset < frame_end <= region_end or not frame_offset + FRAME_HEADER_BYTES <= payload_offset < payload_end <= frame_end:
        raise ValueError("directory nested range geometry")
    return {
        "ordinal": ordinal,
        "region_ordinal": region_ordinal,
        "profile_q": profile_q,
        "contribution_count": contribution_count,
        "symbols": symbols,
        "logical_bits": logical_bits,
        "payload_offset": payload_offset,
        "payload_bytes": payload_bytes,
        "region_offset": region_offset,
        "region_bytes": region_bytes,
        "frame_offset": frame_offset,
        "frame_bytes": frame_bytes,
        "source_weights": source_weights,
        "group_rows": group_rows,
        "group_cols": group_cols,
        "decoder_scale": decoder_scale,
        "owner_set": owner_set,
        "owner_set_hex": owner_set.hex(),
        "owners": owners,
        "source_digest": raw[152:184].hex(),
        "payload_sha256": raw[184:216].hex(),
        "role": role,
    }


def _parse_frame(frame: bytes, row: Mapping[str, Any], experts: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(frame) != int(row["frame_bytes"]):
        raise ValueError("frame read geometry")
    header = frame[:FRAME_HEADER_BYTES]
    fields = struct.unpack_from("<8sIIHHIQQQQQQdQQ", header, 0)
    magic, ordinal, region_ordinal, profile_q, contribution_count, reserved, symbols, logical_bits, payload_bytes, source_weights, group_rows, group_cols, decoder_scale, metadata_bytes, frame_bytes = fields
    expected = (FRAME_MAGIC, row["ordinal"], row["region_ordinal"], row["profile_q"], row["contribution_count"], 0, row["symbols"], row["logical_bits"], row["payload_bytes"], row["source_weights"], row["group_rows"], row["group_cols"])
    observed = (magic, ordinal, region_ordinal, profile_q, contribution_count, reserved, symbols, logical_bits, payload_bytes, source_weights, group_rows, group_cols)
    if observed != expected or struct.pack("<d", decoder_scale) != struct.pack("<d", row["decoder_scale"]):
        raise ValueError("frame/directory scalar mismatch")
    expected_metadata = _align(FRAME_HEADER_BYTES + CONTRIBUTION_RECORD_BYTES * int(row["contribution_count"]), 64)
    if metadata_bytes != expected_metadata or frame_bytes != len(frame) or frame_bytes % 64:
        raise ValueError("frame metadata/length geometry")
    if header[96:128] != row["owner_set"] or header[128:160].hex() != row["source_digest"] or header[160:192].hex() != row["payload_sha256"] or _parse_role(header[192:224]) != row["role"] or header[224:256] != _sha(header[:224]):
        raise ValueError("frame/directory repeated field mismatch")
    contributions: list[dict[str, Any]] = []
    previous_key: tuple[int, int, int] | None = None
    total = 0
    cursor = FRAME_HEADER_BYTES
    for _index in range(int(row["contribution_count"])):
        expert, role_id, source_offset, weight_count = struct.unpack_from("<IIQQ", frame, cursor)
        expert = _integer(expert, "frame contribution expert", 0, experts - 1)
        role_id = _integer(role_id, "frame contribution role", 1, 3)
        role = ("gate", "up", "down")[role_id - 1]
        source_offset = _integer(source_offset, "frame source offset", 0, MAX_WEIGHTS - 1)
        weight_count = _integer(weight_count, "frame contribution weights", 1, MAX_WEIGHTS)
        key = (expert, role_id, source_offset)
        if previous_key is not None and key <= previous_key:
            raise ValueError("frame contribution canonical order")
        previous_key = key
        _checked_add(source_offset, weight_count, "frame contribution interval", MAX_WEIGHTS)
        total = _checked_add(total, weight_count, "frame contribution total", MAX_WEIGHTS)
        contributions.append({"expert": expert, "role": role, "source_offset": source_offset, "weight_count": weight_count})
        cursor += CONTRIBUTION_RECORD_BYTES
    if tuple(sorted(set(item["expert"] for item in contributions))) != tuple(row["owners"]) or total != int(row["source_weights"]):
        raise ValueError("frame contribution/owner conservation")
    if any(frame[cursor:metadata_bytes]):
        raise ValueError("nonzero frame metadata padding")
    payload_begin = metadata_bytes
    payload_end = _checked_add(payload_begin, int(row["payload_bytes"]), "frame local payload end", len(frame))
    payload = frame[payload_begin:payload_end]
    if _sha(payload).hex() != row["payload_sha256"]:
        raise ValueError("frame payload digest")
    if int(row["logical_bits"]) & 7 and payload[-1] & ((1 << (8 - (int(row["logical_bits"]) & 7))) - 1):
        raise ValueError("nonzero terminal payload padding")
    if any(frame[payload_end:]):
        raise ValueError("nonzero frame alignment padding")
    parsed = dict(row)
    parsed["owner_contributions"] = tuple(contributions)
    parsed["payload"] = payload
    parsed["metadata_bytes"] = metadata_bytes
    parsed["frame_local_payload_begin"] = payload_begin
    return parsed, contributions


def _interval(begin: int, end: int, kind: str, owner_set: bytes, *, padding: bool) -> dict[str, Any]:
    if not 0 <= begin < end <= MAX_CONTAINER_BYTES:
        raise ValueError("ledger interval geometry")
    return {
        "begin": begin,
        "end": end,
        "bytes": end - begin,
        "kind": kind,
        "owner_set": owner_set,
        "owner_set_hex": owner_set.hex(),
        "padding": bool(padding),
    }


def _global_sections(common: Any, semantic_codec: Any, reader: Any, header: Mapping[str, Any], *, verify_body: bool) -> dict[str, Any]:
    all_owners = owner_set_from_ordinals(int(header["experts"]), list(range(int(header["experts"]))))
    semantic_offset = int(header["semantic_offset"])
    semantic_bytes = int(header["semantic_bytes"])
    immutable_offset = int(header["immutable_offset"])
    immutable_bytes = int(header["immutable_bytes"])
    model_offset = int(header["model_offset"])
    model_bytes = int(header["model_bytes"])
    directory_offset = int(header["directory_offset"])
    directory_bytes = int(header["directory_bytes"])
    shared_bytes = int(header["shared_bytes"])
    semantic_packet = reader.read(semantic_offset, semantic_bytes)
    immutable_state = reader.read(immutable_offset, immutable_bytes)
    model_packet = reader.read(model_offset, model_bytes)
    directory_blob = reader.read(directory_offset, directory_bytes)
    if _sha(semantic_packet) != header["semantic_sha256"] or _sha(immutable_state) != header["immutable_sha256"] or _sha(model_packet) != header["model_sha256"] or _sha(directory_blob) != header["directory_sha256"]:
        raise ValueError("global section digest")
    if verify_body:
        if _hash_reader_range(reader, HEADER_BYTES, int(header["total_bytes"]) - HEADER_BYTES) != header["body_sha256"]:
            raise ValueError("container body digest")
    padding_ranges = (
        (semantic_offset + semantic_bytes, immutable_offset),
        (immutable_offset + immutable_bytes, model_offset),
        (model_offset + model_bytes, directory_offset),
        (directory_offset + directory_bytes, shared_bytes),
    )
    if verify_body:
        for begin, end in padding_ranges:
            if end > begin and not _range_is_zero(reader, begin, end - begin):
                raise ValueError("nonzero global alignment padding")
    semantics = semantic_codec.parse_semantic_packet(semantic_packet)
    if int(semantics["experts"]) != int(header["experts"]) or int(semantics["source_weights"]) != int(header["weights"]):
        raise ValueError("semantic/header geometry")
    candidate, frequencies = common.deserialize_model(model_packet)
    if common.serialize_model(candidate, frequencies) != model_packet:
        raise ValueError("noncanonical serialized model")
    directory = [
        _parse_directory_record(directory_blob[index * DIRECTORY_RECORD_BYTES:(index + 1) * DIRECTORY_RECORD_BYTES], header)
        for index in range(int(header["streams"]))
    ]
    if [row["ordinal"] for row in directory] != list(range(int(header["streams"]))):
        raise ValueError("directory canonical ordinal order")
    ledger = [_interval(0, HEADER_BYTES, "container_header", all_owners, padding=False)]
    ledger.append(_interval(semantic_offset, semantic_offset + semantic_bytes, "universal_semantics", all_owners, padding=False))
    if immutable_offset > semantic_offset + semantic_bytes:
        ledger.append(_interval(semantic_offset + semantic_bytes, immutable_offset, "semantic_alignment_padding", all_owners, padding=True))
    if immutable_bytes:
        ledger.append(_interval(immutable_offset, immutable_offset + immutable_bytes, "evaluation_plugin_immutable_state", all_owners, padding=False))
    if model_offset > immutable_offset + immutable_bytes:
        ledger.append(_interval(immutable_offset + immutable_bytes, model_offset, "immutable_alignment_padding", all_owners, padding=True))
    ledger.append(_interval(model_offset, model_offset + model_bytes, "serialized_unifilar_model", all_owners, padding=False))
    if directory_offset > model_offset + model_bytes:
        ledger.append(_interval(model_offset + model_bytes, directory_offset, "model_alignment_padding", all_owners, padding=True))
    ledger.append(_interval(directory_offset, directory_offset + directory_bytes, "stream_directory", all_owners, padding=False))
    if shared_bytes > directory_offset + directory_bytes:
        ledger.append(_interval(directory_offset + directory_bytes, shared_bytes, "directory_alignment_padding", all_owners, padding=True))
    return {
        "semantic_packet": semantic_packet,
        "semantics": semantics,
        "immutable_state": immutable_state,
        "model_packet": model_packet,
        "candidate": candidate,
        "frequencies": frequencies,
        "directory": directory,
        "directory_blob": directory_blob,
        "ledger": ledger,
    }


def parse_container_reader(common: Any, semantic_codec: Any, reader: Any, file_size: int) -> dict[str, Any]:
    """Full parser. The caller bounds file size before this first read."""
    file_size = _integer(file_size, "external file size", HEADER_BYTES, MAX_CONTAINER_BYTES)
    if getattr(reader, "size", file_size) != file_size:
        raise ValueError("reader/file-size mismatch")
    header_raw = reader.read(0, HEADER_BYTES)
    header = _parse_header(header_raw, file_size)
    global_state = _global_sections(common, semantic_codec, reader, header, verify_body=True)
    directory = global_state["directory"]
    regions: dict[int, dict[str, Any]] = {}
    seen_owner_sets: set[bytes] = set()
    union = bytearray(OWNER_SET_BYTES)
    for row in directory:
        owner_set = row["owner_set"]
        if owner_set not in seen_owner_sets:
            seen_owner_sets.add(owner_set)
        for index in range(OWNER_SET_BYTES):
            union[index] |= owner_set[index]
        key = int(row["region_ordinal"])
        identity = (int(row["region_offset"]), int(row["region_bytes"]), owner_set)
        if key in regions and regions[key]["identity"] != identity:
            raise ValueError("inconsistent region identity in directory")
        regions.setdefault(key, {"identity": identity, "rows": []})["rows"].append(row)
    if len(regions) != int(header["region_count"]) or sorted(regions) != list(range(int(header["region_count"]))):
        raise ValueError("region ordinal coverage")
    if len(seen_owner_sets) != len(regions):
        raise ValueError("duplicate owner-set region")
    all_owners = owner_set_from_ordinals(int(header["experts"]), list(range(int(header["experts"]))))
    if bytes(union) != all_owners:
        raise ValueError("declared expert universe contains an empty expert")
    owner_order = [regions[index]["identity"][2] for index in range(len(regions))]
    if owner_order != sorted(owner_order, key=lambda value: _owner_sort_key(value, int(header["experts"]))):
        raise ValueError("noncanonical region order")
    ledger = list(global_state["ledger"])
    cursor = int(header["shared_bytes"])
    parsed_rows: dict[int, dict[str, Any]] = {}
    parsed_regions: list[dict[str, Any]] = []
    for region_ordinal in range(len(regions)):
        region = regions[region_ordinal]
        region_offset, region_bytes, owner_set = region["identity"]
        if region_offset != cursor or region_offset % PAGE_BYTES or region_bytes % PAGE_BYTES:
            raise ValueError("region contiguous page placement")
        packet_header = reader.read(region_offset, REGION_HEADER_BYTES)
        fields = struct.unpack_from("<8sIIHHIQQQ", packet_header, 0)
        magic, ordinal, stream_count, owner_bytes, reserved_h, reserved_i, region_bytes_h, content_bytes, frame_area_bytes = fields
        if (magic, ordinal, stream_count, owner_bytes, reserved_h, reserved_i, region_bytes_h) != (REGION_MAGIC, region_ordinal, len(region["rows"]), OWNER_SET_BYTES, 0, 0, region_bytes):
            raise ValueError("region header constants")
        if packet_header[48:80] != owner_set or packet_header[112:144] != _sha(packet_header[:112]) or any(packet_header[144:]):
            raise ValueError("region owner/seal/reserved")
        frame_area_begin = region_offset + REGION_HEADER_BYTES
        content_bytes = _integer(content_bytes, "region content bytes", REGION_HEADER_BYTES, region_bytes)
        frame_area_bytes = _integer(frame_area_bytes, "region frame area bytes", FRAME_HEADER_BYTES, region_bytes)
        if content_bytes != REGION_HEADER_BYTES + frame_area_bytes:
            raise ValueError("region content/frame-area mismatch")
        rows = sorted(region["rows"], key=lambda row: int(row["ordinal"]))
        frame_cursor = frame_area_begin
        ledger.append(_interval(region_offset, frame_area_begin, "region_header", owner_set, padding=False))
        for row in rows:
            if row["owner_set"] != owner_set or int(row["frame_offset"]) != frame_cursor:
                raise ValueError("frame owner/contiguity mismatch")
            frame = reader.read(frame_cursor, int(row["frame_bytes"]))
            parsed_row, _contributions = _parse_frame(frame, row, int(header["experts"]))
            absolute_payload = frame_cursor + int(parsed_row["metadata_bytes"])
            if absolute_payload != int(row["payload_offset"]):
                raise ValueError("directory/frame payload offset")
            contribution_end = frame_cursor + FRAME_HEADER_BYTES + CONTRIBUTION_RECORD_BYTES * int(row["contribution_count"])
            metadata_end = frame_cursor + int(parsed_row["metadata_bytes"])
            payload_end = absolute_payload + int(row["payload_bytes"])
            frame_end = frame_cursor + int(row["frame_bytes"])
            ledger.append(_interval(frame_cursor, frame_cursor + FRAME_HEADER_BYTES, "frame_header", owner_set, padding=False))
            ledger.append(_interval(frame_cursor + FRAME_HEADER_BYTES, contribution_end, "owner_contribution_records", owner_set, padding=False))
            if metadata_end > contribution_end:
                ledger.append(_interval(contribution_end, metadata_end, "frame_metadata_padding", owner_set, padding=True))
            ledger.append(_interval(absolute_payload, payload_end, "arithmetic_payload", owner_set, padding=False))
            if frame_end > payload_end:
                ledger.append(_interval(payload_end, frame_end, "frame_alignment_padding", owner_set, padding=True))
            parsed_rows[int(row["ordinal"])] = parsed_row
            frame_cursor = frame_end
        if frame_cursor != frame_area_begin + frame_area_bytes or frame_cursor != region_offset + content_bytes:
            raise ValueError("region frame-area coverage")
        if _hash_reader_range(reader, frame_area_begin, frame_area_bytes) != packet_header[80:112]:
            raise ValueError("region frame-area digest")
        region_end = region_offset + region_bytes
        if region_end > frame_cursor:
            if not _range_is_zero(reader, frame_cursor, region_end - frame_cursor):
                raise ValueError("nonzero region rate padding")
            ledger.append(_interval(frame_cursor, region_end, "owner_region_rate_padding", owner_set, padding=True))
        parsed_regions.append({
            "ordinal": region_ordinal,
            "offset": region_offset,
            "bytes": region_bytes,
            "content_bytes": content_bytes,
            "owner_set": owner_set,
            "owner_set_hex": owner_set.hex(),
            "owners": owner_ordinals(owner_set, int(header["experts"])),
            "rows": rows,
        })
        cursor = region_end
    base_region_lengths = [_align(int(region["content_bytes"]), PAGE_BYTES) for region in parsed_regions]
    canonical_base_total = int(header["shared_bytes"])
    for length in base_region_lengths:
        canonical_base_total = _checked_add(canonical_base_total, length, "canonical base region total")
    rate_product = _checked_mul(int(header["weights"]), RATE_MIN_NUMERATOR, "canonical rate product", MAX_WEIGHTS * 10_000)
    canonical_minimum_total = _ceil_div(rate_product, 8 * RATE_MIN_DENOMINATOR, "canonical rate floor")
    if canonical_minimum_total > MAX_CONTAINER_BYTES:
        raise ValueError("canonical rate floor exceeds container bound")
    padding_pages = _ceil_div(max(0, canonical_minimum_total - canonical_base_total), PAGE_BYTES, "canonical padding pages")
    pages_each, leading_extra = divmod(padding_pages, len(base_region_lengths))
    canonical_region_lengths = []
    for index, length in enumerate(base_region_lengths):
        pages = pages_each + (1 if index < leading_extra else 0)
        canonical_region_lengths.append(_checked_add(length, _checked_mul(pages, PAGE_BYTES, "canonical region padding", MAX_CONTAINER_BYTES), "canonical region length"))
    if [int(region["bytes"]) for region in parsed_regions] != canonical_region_lengths:
        raise ValueError("noncanonical rate-padding distribution")
    if cursor != file_size or sorted(parsed_rows) != list(range(int(header["streams"]))):
        raise ValueError("container/stream coverage does not end canonically")
    ordered_rows = [parsed_rows[index] for index in range(int(header["streams"]))]
    coverage = semantic_codec.validate_stream_coverage(global_state["semantics"], ordered_rows)
    if int(coverage["source_weights"]) != int(header["weights"]):
        raise ValueError("source-weight conservation")
    expected_begin = 0
    for entry in ledger:
        if int(entry["begin"]) != expected_begin:
            raise ValueError("byte ledger overlap or hole")
        expected_begin = int(entry["end"])
    if expected_begin != file_size:
        raise ValueError("byte ledger does not cover canonical EOF")
    return {
        "raw": reader.read(0, file_size) if isinstance(reader, MemoryReader) else None,
        **header,
        **global_state,
        "directory": ordered_rows,
        "regions": parsed_regions,
        "byte_ledger": ledger,
        "coverage": coverage,
    }


def parse_container(common: Any, semantic_codec: Any, container: bytes) -> dict[str, Any]:
    reader = MemoryReader(container)
    return parse_container_reader(common, semantic_codec, reader, reader.size)


def parse_container_descriptor(common: Any, semantic_codec: Any, fd: int) -> dict[str, Any]:
    """Parse from a bounded held descriptor without accepting a path."""
    reader = DescriptorReader(fd)
    try:
        result = parse_container_reader(common, semantic_codec, reader, reader.size)
        reader.verify_stable()
        return result
    finally:
        reader.close()


def routed_read_expert(
    common: Any,
    semantic_codec: Any,
    reader: InstrumentedReader,
    *,
    file_size: int,
    expert: int,
    externally_authenticated_container_sha256: str,
    decode_routed_expert: Any | None = None,
) -> dict[str, Any]:
    """Fresh routed parse/decode input discovery through instrumented reads.

    The externally authenticated digest records the separate installation scan;
    this routed path deliberately does not scan unread regions merely to hash
    them. It builds no state from a prior full parse. In particular, the body
    digest and rate-padding zeros are full-audit facts, not routed-decode
    prepasses: a selected expert never reads, hashes, or parses an unowned
    frame or region.
    """
    file_size = _integer(file_size, "routed file size", HEADER_BYTES, MAX_CONTAINER_BYTES)
    if not isinstance(externally_authenticated_container_sha256, str) or len(externally_authenticated_container_sha256) != 64:
        raise ValueError("external container authentication digest")
    try:
        bytes.fromhex(externally_authenticated_container_sha256)
    except ValueError as exc:
        raise ValueError("external container authentication digest encoding") from exc
    header_raw = reader.read(0, HEADER_BYTES)
    header = _parse_header(header_raw, file_size)
    expert = _integer(expert, "routed expert", 0, int(header["experts"]) - 1)
    global_state = _global_sections(common, semantic_codec, reader, header, verify_body=False)
    selected = [row for row in global_state["directory"] if expert in row["owners"]]
    if not selected:
        raise ValueError("routed expert owns no stream")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(int(row["region_ordinal"]), []).append(row)
    parsed_rows: list[dict[str, Any]] = []
    for region_ordinal in sorted(grouped):
        region_rows = sorted(grouped[region_ordinal], key=lambda row: int(row["frame_offset"]))
        first = region_rows[0]
        packet = reader.read(int(first["region_offset"]), REGION_HEADER_BYTES)
        fields = struct.unpack_from("<8sIIHHIQQQ", packet, 0)
        magic, ordinal, stream_count, owner_bytes, reserved_h, reserved_i, region_bytes, content_bytes, frame_area_bytes = fields
        if magic != REGION_MAGIC or ordinal != region_ordinal or owner_bytes != OWNER_SET_BYTES or reserved_h != 0 or reserved_i != 0 or region_bytes != int(first["region_bytes"]):
            raise ValueError("routed region header")
        if packet[48:80] != first["owner_set"] or packet[112:144] != _sha(packet[:112]) or any(packet[144:]):
            raise ValueError("routed region repeated owner/seal")
        stream_count = _integer(stream_count, "routed region streams", 1, int(header["streams"]))
        content_bytes = _integer(content_bytes, "routed region content", REGION_HEADER_BYTES, int(first["region_bytes"]))
        frame_area_bytes = _integer(frame_area_bytes, "routed frame area", FRAME_HEADER_BYTES, int(first["region_bytes"]))
        if stream_count != len(region_rows) or content_bytes != REGION_HEADER_BYTES + frame_area_bytes:
            raise ValueError("routed region stream/content geometry")
        frame_cursor = int(first["region_offset"]) + REGION_HEADER_BYTES
        for row in region_rows:
            if row["owner_set"] != first["owner_set"] or int(row["region_offset"]) != int(first["region_offset"]) or int(row["region_bytes"]) != int(first["region_bytes"]) or int(row["frame_offset"]) != frame_cursor:
                raise ValueError("routed frame owner/region/contiguity")
            frame = reader.read(frame_cursor, int(row["frame_bytes"]))
            parsed_row, _contributions = _parse_frame(frame, row, int(header["experts"]))
            parsed_rows.append(parsed_row)
            frame_cursor += int(row["frame_bytes"])
        if frame_cursor != int(first["region_offset"]) + content_bytes:
            raise ValueError("routed region frame-area coverage")
    shape = global_state["semantics"]["shapes"][expert]
    target = int(shape.matrix_weights)
    for role in ("gate", "up", "down"):
        intervals = []
        for row in parsed_rows:
            for contribution in row["owner_contributions"]:
                if int(contribution["expert"]) == expert and contribution["role"] == role:
                    begin = int(contribution["source_offset"])
                    intervals.append((begin, begin + int(contribution["weight_count"])))
        cursor = 0
        for begin, end in sorted(intervals):
            if begin != cursor:
                raise ValueError("routed expert scalar coverage overlap or hole")
            cursor = end
        if cursor != target:
            raise ValueError("routed expert role coverage")
    routed = {
        "expert_ordinal": expert,
        "rows": tuple(sorted(parsed_rows, key=lambda row: int(row["ordinal"]))),
        "semantic_shape": shape,
        "semantics": global_state["semantics"],
        "candidate": global_state["candidate"],
        "frequencies": global_state["frequencies"],
        "immutable_state": global_state["immutable_state"],
        "external_authentication_sha256": externally_authenticated_container_sha256.lower(),
        "installation_authentication_scan_bytes": file_size,
        "routed_read_ranges": tuple(reader.ranges),
        "touched_page_indices": tuple(sorted(reader.pages)),
        "touched_page_bytes": len(reader.pages) * PAGE_BYTES,
    }
    if decode_routed_expert is None:
        routed["causal_decode_reencode_reconstruction"] = None
        return routed
    if not callable(decode_routed_expert):
        raise ValueError("routed expert decoder callback")
    decoded = decode_routed_expert(routed)
    if not isinstance(decoded, dict) or set(decoded) != {
        "expert_ordinal", "decoded_streams", "all_payloads_canonically_reencoded",
        "all_three_roles_reconstructed", "routed_expert_reconstruction_sha256",
    }:
        raise ValueError("routed decoder result schema")
    if decoded["expert_ordinal"] != expert or type(decoded["decoded_streams"]) is not int or decoded["decoded_streams"] != len(parsed_rows):
        raise ValueError("routed decoder expert/stream binding")
    if decoded["all_payloads_canonically_reencoded"] is not True or decoded["all_three_roles_reconstructed"] is not True:
        raise ValueError("routed decode/re-encode/reconstruction failure")
    _hex32(decoded["routed_expert_reconstruction_sha256"], "routed expert reconstruction")
    routed["causal_decode_reencode_reconstruction"] = dict(decoded)
    return routed


def instrument_expert_pages(common: Any, semantic_codec: Any, raw: bytes, expert: int) -> dict[str, Any]:
    reader = InstrumentedReader(raw)
    return routed_read_expert(
        common,
        semantic_codec,
        reader,
        file_size=reader.size,
        expert=expert,
        externally_authenticated_container_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": f"{value.numerator}/{value.denominator}",
        "float": float(value),
    }


def _read_request_summary(ranges: Sequence[Sequence[int]]) -> dict[str, int]:
    """Account for literal read calls both with and without overlap.

    The frozen cold gate deliberately uses the union of touched 4-KiB pages.
    This separate diagnostic preserves the other operationally useful view:
    bytes requested by every read call, including repeated/overlapping bytes.
    """
    normalized: list[tuple[int, int]] = []
    requested = 0
    for item in ranges:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("routed read range schema")
        begin, end = item
        if type(begin) is not int or type(end) is not int or begin < 0 or end < begin:
            raise ValueError("routed read range bounds")
        normalized.append((begin, end))
        requested += end - begin
    unique = 0
    if normalized:
        union_begin, union_end = sorted(normalized)[0]
        for begin, end in sorted(normalized)[1:]:
            if begin > union_end:
                unique += union_end - union_begin
                union_begin, union_end = begin, end
            else:
                union_end = max(union_end, end)
        unique += union_end - union_begin
    return {
        "read_request_count": len(normalized),
        "requested_bytes_with_repetition": requested,
        "unique_requested_bytes": unique,
        "overlap_bytes_requested_again": requested - unique,
    }


def posterior_diagnostic_handoff(common: Any, parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a future diagnostic to the authenticated literal decision source.

    This does not compute a posterior reconstruction.  It exposes only the
    immutable evidence needed by a separately frozen diagnostic to re-decode
    every selected SC decision from the literal object: payload/model bytes,
    public-context state, semantic routes, per-stream decoded selected-decision
    `(bits, levels, base-frequency-u16le)` triplet commitments,
    and the original source/reconstruction identities.
    """
    raw = parsed.get("raw")
    if not isinstance(raw, bytes):
        raise ValueError("diagnostic handoff requires literal container bytes")
    decision_rows = []
    for ordinal, row in enumerate(parsed["directory"]):
        if int(row["ordinal"]) != ordinal:
            raise ValueError("diagnostic handoff stream order")
        decision_rows.append({
            "ordinal": ordinal,
            "symbols": int(row["symbols"]),
            "logical_bits": int(row["logical_bits"]),
            "decoded_selected_decision_triplet_sha256": str(row["source_digest"]),
            "payload_sha256": hashlib.sha256(bytes(row["payload"])).hexdigest(),
            "profile_q": int(row["profile_q"]),
            "role": str(row["role"]),
            "owner_set_hex": bytes(row["owner_set"]).hex(),
            "owner_contributions": [dict(item) for item in row["owner_contributions"]],
        })
    commitment = hashlib.sha256(common.canonical_json(decision_rows)).hexdigest()
    return {
        "schema": "uwfa-sc-v7-posterior-diagnostic-handoff",
        "literal_container_sha256": hashlib.sha256(raw).hexdigest(),
        "source_artifact_sha256": str(parsed["baseline_artifact_sha256"]),
        "source_score_binding_sha256": str(parsed["audit_binding_sha256"]),
        "source_full_geometry_sha256": str(parsed["binding_hashes"]["source_full_geometry_sha256"]),
        "source_structural_geometry_sha256": str(parsed["binding_hashes"]["source_structural_geometry_sha256"]),
        "extraction_program_sha256": str(parsed["binding_hashes"]["extraction_program_sha256"]),
        "universal_decoder_sha256": str(parsed["binding_hashes"]["universal_decoder_sha256"]),
        "universal_adapter_sha256": str(parsed["binding_hashes"]["universal_adapter_sha256"]),
        "pipeline_sha256": str(parsed["binding_hashes"]["pipeline_sha256"]),
        "source_snapshot_root_sha256": str(parsed["binding_hashes"]["source_snapshot_root_sha256"]),
        "source_preflight_receipt_sha256": str(parsed["binding_hashes"]["source_preflight_receipt_sha256"]),
        "full_reconstruction_f64_sha256": str(parsed["reconstruction_sha256"]),
        "semantic_packet_sha256": hashlib.sha256(bytes(parsed["semantic_packet"])).hexdigest(),
        "immutable_context_state_sha256": hashlib.sha256(bytes(parsed["immutable_state"])).hexdigest(),
        "serialized_model_sha256": hashlib.sha256(bytes(parsed["model_packet"])).hexdigest(),
        "directory_sha256": hashlib.sha256(bytes(parsed["directory_blob"])).hexdigest(),
        "stream_decision_triplet_commitments": decision_rows,
        "decoded_sc_decision_triplet_commitment_sha256": commitment,
        "stream_count": len(decision_rows),
        "requires_literal_redecode": True,
        "contains_posterior_or_MMSE_result": False,
    }


def physical_metrics(
    common: Any,
    semantic_codec: Any,
    parsed: Mapping[str, Any],
    *,
    routed_descriptor_source: Any | None = None,
    externally_authenticated_container_sha256: str | None = None,
    routed_decoder: Any | None = None,
) -> dict[str, Any]:
    raw = bytes(parsed["raw"])
    total = len(raw)
    weights = int(parsed["weights"])
    experts = int(parsed["experts"])
    ledger = parsed["byte_ledger"]
    directory_modeled_symbols = sum(int(row["symbols"]) for row in parsed["directory"])
    directory_source_weights = sum(int(row["source_weights"]) for row in parsed["directory"])
    if directory_source_weights != weights:
        raise AssertionError("directory source-weight accounting does not equal header weights")
    attributed_total = [Fraction(0, 1) for _ in range(experts)]
    attributed_nonpadding = [Fraction(0, 1) for _ in range(experts)]
    allocation_sum = Fraction(0, 1)
    for entry in ledger:
        owners = owner_ordinals(bytes(entry["owner_set"]), experts)
        share = Fraction(int(entry["bytes"]), len(owners))
        for owner in owners:
            attributed_total[owner] += share
            if not bool(entry["padding"]):
                attributed_nonpadding[owner] += share
            allocation_sum += share
    if allocation_sum != total:
        raise AssertionError("exact owner allocation does not equal literal bytes")
    authoritative_routed_io = routed_descriptor_source is not None
    if authoritative_routed_io:
        if not isinstance(routed_descriptor_source, AuthenticatedDescriptorSource):
            raise ValueError("authenticated descriptor source")
        if externally_authenticated_container_sha256 != hashlib.sha256(raw).hexdigest():
            raise ValueError("routed reader/container authentication binding")
        if routed_descriptor_source.container_sha256 != externally_authenticated_container_sha256 or routed_descriptor_source.size != total:
            raise ValueError("authenticated descriptor source binding")
        if not callable(getattr(routed_decoder, "decode_expert", None)) or not callable(getattr(routed_decoder, "finalize", None)):
            raise ValueError("authoritative cold proof requires routed decoder session")
    rows = []
    maximum = Fraction(0, 1)
    routed_modeled_symbols_sum = 0
    routed_source_weights_sum = 0
    requested_bytes_sum = 0
    unique_requested_bytes_sum = 0
    overlap_bytes_sum = 0
    read_request_count_sum = 0
    unique_touched_page_bytes_sum = 0
    maximum_requested_bytes = 0
    for expert in range(experts):
        if authoritative_routed_io:
            reader = routed_descriptor_source.fresh_reader()
            try:
                if reader.size != total:
                    raise ValueError("authoritative routed reader size")
                trace = routed_read_expert(
                    common,
                    semantic_codec,
                    reader,
                    file_size=reader.size,
                    expert=expert,
                    externally_authenticated_container_sha256=str(externally_authenticated_container_sha256),
                    decode_routed_expert=routed_decoder.decode_expert,
                )
                reader.verify_stable()
                routed_descriptor_source.verify_stable()
            finally:
                reader.close()
        else:
            trace = instrument_expert_pages(common, semantic_codec, raw, expert)
        read_summary = _read_request_summary(trace["routed_read_ranges"])
        routed_modeled_symbols = sum(int(row["symbols"]) for row in trace["rows"])
        expert_source_weights = sum(
            int(contribution["weight_count"])
            for row in trace["rows"]
            for contribution in row["owner_contributions"]
            if int(contribution["expert"]) == expert
        )
        if expert_source_weights <= 0:
            raise AssertionError("routed expert source-weight accounting")
        routed_modeled_symbols_sum += routed_modeled_symbols
        routed_source_weights_sum += expert_source_weights
        requested_bytes_sum += read_summary["requested_bytes_with_repetition"]
        unique_requested_bytes_sum += read_summary["unique_requested_bytes"]
        overlap_bytes_sum += read_summary["overlap_bytes_requested_again"]
        read_request_count_sum += read_summary["read_request_count"]
        maximum_requested_bytes = max(maximum_requested_bytes, read_summary["requested_bytes_with_repetition"])
        cold = int(trace["touched_page_bytes"])
        unique_touched_page_bytes_sum += cold
        total_amplification = Fraction(cold, 1) / attributed_total[expert]
        nonpadding_amplification = Fraction(cold, 1) / attributed_nonpadding[expert]
        strict = max(total_amplification, nonpadding_amplification)
        maximum = max(maximum, strict)
        rows.append({
            "expert_ordinal": expert,
            "touched_page_indices": list(trace["touched_page_indices"]),
            "touched_page_bytes": cold,
            "instrumented_routed_read_ranges": [list(item) for item in trace["routed_read_ranges"]],
            "instrumented_routed_read_request_count": read_summary["read_request_count"],
            "instrumented_routed_requested_bytes_with_repetition": read_summary["requested_bytes_with_repetition"],
            "instrumented_routed_unique_requested_bytes": read_summary["unique_requested_bytes"],
            "instrumented_routed_overlap_bytes_requested_again": read_summary["overlap_bytes_requested_again"],
            "routed_modeled_symbols": routed_modeled_symbols,
            "expert_source_weights": expert_source_weights,
            "routed_modeled_symbols_per_source_weight": _fraction_record(Fraction(routed_modeled_symbols, expert_source_weights)),
            "installation_authentication_scan_bytes_reported_separately": int(trace["installation_authentication_scan_bytes"]),
            "causal_decode_reencode_reconstruction": trace.get("causal_decode_reencode_reconstruction"),
            "attributable_total_physical_bytes": _fraction_record(attributed_total[expert]),
            "attributable_nonpadding_decodable_bytes": _fraction_record(attributed_nonpadding[expert]),
            "cold_amplification_total_physical": _fraction_record(total_amplification),
            "cold_amplification_nonpadding": _fraction_record(nonpadding_amplification),
            "strict_cold_amplification": _fraction_record(strict),
        })
    routed_reconstruction = None
    if authoritative_routed_io:
        routed_reconstruction = routed_decoder.finalize(
            experts=experts,
            expected_full_reconstruction_sha256=str(parsed["reconstruction_sha256"]),
        )
        if not isinstance(routed_reconstruction, dict) or set(routed_reconstruction) != {
            "experts", "full_reconstruction_f64_sha256", "matches_container_reconstruction",
        }:
            raise ValueError("routed reconstruction finalization schema")
        if routed_reconstruction["experts"] != experts or routed_reconstruction["matches_container_reconstruction"] is not True or routed_reconstruction["full_reconstruction_f64_sha256"] != parsed["reconstruction_sha256"]:
            raise ValueError("routed reconstruction does not match container binding")
    if routed_source_weights_sum != weights:
        raise AssertionError("routed expert source-weight accounting does not equal header weights")
    rate = Fraction(8 * total, weights)
    rate_float = float(rate)
    mse = float(parsed["audited_relative_mse"])
    try:
        f_value = mse * math.pow(2.0, 2.0 * rate_float)
    except OverflowError:
        f_value = math.inf
    return {
        "actual_container_bytes": total,
        "source_weights": weights,
        "modeled_symbol_density": {
            "unique_directory_modeled_symbols": directory_modeled_symbols,
            "source_weights": weights,
            "unique_directory_modeled_symbols_per_source_weight": _fraction_record(Fraction(directory_modeled_symbols, weights)),
            "routed_modeled_symbols_sum_across_experts": routed_modeled_symbols_sum,
            "routed_source_weights_sum_across_experts": routed_source_weights_sum,
            "routed_modeled_symbols_per_source_weight_sum_across_experts": _fraction_record(Fraction(routed_modeled_symbols_sum, routed_source_weights_sum)),
            "shared_stream_symbol_reuse_across_expert_routes": routed_modeled_symbols_sum - directory_modeled_symbols,
        },
        "actual_physical_rate_bpw": rate_float,
        "actual_physical_rate_rational": _fraction_record(rate),
        "baseline_object_bytes": int(parsed["baseline_object_bytes"]),
        "net_physical_saving_bpw": 8.0 * (int(parsed["baseline_object_bytes"]) - total) / weights,
        "audited_identical_reconstruction_relative_mse": mse,
        "F_from_actual_bytes_and_identical_reconstruction": f_value,
        "passes_rate_interval": Fraction(43, 20) <= rate <= Fraction(5, 2),
        "passes_F_target": f_value <= 0.8,
        "cold_gate_definition": "maximum of routed touched-pages/attributable-total and routed touched-pages/attributable-nonpadding",
        "experts": rows,
        "routed_read_request_aggregates": {
            "read_request_count_sum_across_experts": read_request_count_sum,
            "requested_bytes_with_repetition_sum_across_experts": requested_bytes_sum,
            "unique_requested_bytes_sum_across_experts": unique_requested_bytes_sum,
            "overlap_bytes_requested_again_sum_across_experts": overlap_bytes_sum,
            "unique_touched_page_bytes_sum_across_experts": unique_touched_page_bytes_sum,
            "maximum_requested_bytes_with_repetition_per_expert": maximum_requested_bytes,
            "mean_requested_bytes_with_repetition_per_expert": _fraction_record(Fraction(requested_bytes_sum, experts)),
            "frozen_cold_gate_uses_unique_touched_page_bytes_only": True,
        },
        "maximum_strict_cold_read_amplification": _fraction_record(maximum),
        "routed_io_authoritative_descriptor_backed": authoritative_routed_io,
        "installation_authentication_reported_separately": (
            {
                "container_sha256": routed_descriptor_source.container_sha256,
                "read_ranges": [list(item) for item in routed_descriptor_source.installation_authentication_read_ranges],
                **_read_request_summary(routed_descriptor_source.installation_authentication_read_ranges),
                "scan_bytes": routed_descriptor_source.installation_authentication_scan_bytes,
                "touched_page_indices": list(routed_descriptor_source.installation_authentication_touched_page_indices),
                "touched_page_bytes": routed_descriptor_source.installation_authentication_touched_page_bytes,
                "excluded_from_per_expert_cold_numerator": True,
            }
            if authoritative_routed_io else None
        ),
        "routed_full_reconstruction": routed_reconstruction,
        "passes_cold_read_below_2x": authoritative_routed_io and maximum < 2,
        "diagnostic_memory_routed_ratio_below_2x": (not authoritative_routed_io) and maximum < 2,
        "ownership_allocated_bytes_sum": _fraction_record(allocation_sum),
        "complete_byte_partition_entries": len(ledger),
        "complete_byte_partition_exact": True,
    }


def canonical_rebuild(common: Any, semantic_codec: Any, parsed: Mapping[str, Any]) -> bytes:
    regions: list[RegionSpec] = []
    for region in parsed["regions"]:
        specs = []
        for row in region["rows"]:
            parsed_row = parsed["directory"][int(row["ordinal"])]
            contributions = tuple(
                OwnerContribution(int(item["expert"]), str(item["role"]), int(item["source_offset"]), int(item["weight_count"]))
                for item in parsed_row["owner_contributions"]
            )
            specs.append(StreamSpec(
                ordinal=int(parsed_row["ordinal"]),
                symbols=int(parsed_row["symbols"]),
                logical_bits=int(parsed_row["logical_bits"]),
                payload=bytes(parsed_row["payload"]),
                source_digest=str(parsed_row["source_digest"]),
                profile_q=int(parsed_row["profile_q"]),
                decoder_scale=float(parsed_row["decoder_scale"]),
                role=str(parsed_row["role"]),
                group_rows=int(parsed_row["group_rows"]),
                group_cols=int(parsed_row["group_cols"]),
                owner_contributions=contributions,
            ))
        regions.append(RegionSpec(bytes(region["owner_set"]), tuple(specs)))
    rebuilt, _metrics = build_container(
        common,
        semantic_codec,
        model_packet=bytes(parsed["model_packet"]),
        semantic_packet=bytes(parsed["semantic_packet"]),
        immutable_state=bytes(parsed["immutable_state"]),
        regions=regions,
        weights=int(parsed["weights"]),
        experts=int(parsed["experts"]),
        baseline_object_bytes=int(parsed["baseline_object_bytes"]),
        audited_relative_mse=float(parsed["audited_relative_mse"]),
        baseline_artifact_sha256=str(parsed["baseline_artifact_sha256"]),
        reconstruction_sha256=str(parsed["reconstruction_sha256"]),
        audit_binding_sha256=str(parsed["audit_binding_sha256"]),
        binding_hashes=dict(parsed["binding_hashes"]),
        minimum_rate_numerator=int(parsed["minimum_rate_numerator"]),
        minimum_rate_denominator=int(parsed["minimum_rate_denominator"]),
    )
    return rebuilt
