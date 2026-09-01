#!/usr/bin/env python3
"""Literal UWFA v2 container and exact physical/cold-read accounting.

The container is the only physical object scored by v2.  A shared prefix holds
the universal model, immutable baseline-decoder state, and a fixed directory.
Page-aligned owner regions then hold expert-private or explicitly multi-owner
arithmetic streams.  Every byte, including headers and padding, is emitted and
parsed; no ledger-only byte exists.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


MAGIC = b"UWFCV2\x00\x00"
REGION_MAGIC = b"UWFREG2\x00"
FRAME_MAGIC = b"UWFFRM2\x00"
VERSION = 2
HEADER_BYTES = 4096
DIRECTORY_RECORD_BYTES = 160
REGION_HEADER_BYTES = 128
FRAME_HEADER_BYTES = 128
PAGE_BYTES = 4096

_HEADER_SEAL_BEGIN = 340
_HEADER_SEAL_END = 372
_HEADER_BINDINGS = (
    "baseline_plan_sha256",
    "baseline_score_sha256",
    "universal_decoder_sha256",
    "producer_manifest_sha256",
    "audit_bootstrap_sha256",
    "source_panel_sha256",
    "extraction_program_sha256",
)
_HEADER_BINDINGS_BEGIN = 372
_HEADER_CRC_OFFSET = _HEADER_BINDINGS_BEGIN + 32 * len(_HEADER_BINDINGS)


@dataclass(frozen=True)
class StreamSpec:
    ordinal: int
    symbols: int
    logical_bits: int
    payload: bytes
    source_digest: str
    profile_q: int = 0
    decoder_scale: float = 1.0


@dataclass(frozen=True)
class RegionSpec:
    owner_mask: int
    streams: tuple[StreamSpec, ...]


def _sha(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


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


def _align(value: int, alignment: int = PAGE_BYTES) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("alignment geometry")
    return (value + alignment - 1) // alignment * alignment


def _owners(mask: int, experts: int) -> tuple[int, ...]:
    if mask <= 0 or mask >= (1 << experts):
        raise ValueError("owner mask outside expert geometry")
    result = tuple(index for index in range(experts) if mask & (1 << index))
    if not result:
        raise ValueError("empty owner set")
    return result


def _validate_stream(spec: StreamSpec) -> None:
    if spec.ordinal < 0 or spec.symbols <= 0 or spec.logical_bits <= 0:
        raise ValueError("stream geometry")
    if len(spec.payload) != (spec.logical_bits + 7) // 8:
        raise ValueError("payload/logical-bit geometry")
    if spec.logical_bits & 7:
        low = 8 - (spec.logical_bits & 7)
        if spec.payload[-1] & ((1 << low) - 1):
            raise ValueError("nonzero terminal arithmetic padding")
    _hex32(spec.source_digest, "source_digest")
    if not 0 <= spec.profile_q <= 0xFFFF:
        raise ValueError("profile id")
    if not math.isfinite(spec.decoder_scale) or spec.decoder_scale <= 0.0:
        raise ValueError("decoder scale")


def _frame_bytes(spec: StreamSpec, owner_mask: int) -> bytes:
    _validate_stream(spec)
    header = bytearray(FRAME_HEADER_BYTES)
    struct.pack_into(
        "<8sIIQQQH2xd",
        header,
        0,
        FRAME_MAGIC,
        spec.ordinal,
        owner_mask,
        spec.symbols,
        spec.logical_bits,
        len(spec.payload),
        spec.profile_q,
        spec.decoder_scale,
    )
    header[56:88] = _hex32(spec.source_digest, "source_digest")
    header[88:120] = _sha(spec.payload)
    header[120:128] = _sha(bytes(header[:120]))[:8]
    raw = bytes(header) + spec.payload
    return raw + b"\x00" * (_align(len(raw), 64) - len(raw))


def _directory_record(
    spec: StreamSpec,
    *,
    owner_mask: int,
    region_ordinal: int,
    region_offset: int,
    region_bytes: int,
    frame_offset: int,
) -> bytes:
    row = bytearray(DIRECTORY_RECORD_BYTES)
    payload_offset = frame_offset + FRAME_HEADER_BYTES
    struct.pack_into(
        "<IIIIQQQQQQH2xd",
        row,
        0,
        spec.ordinal,
        owner_mask,
        region_ordinal,
        0,
        spec.symbols,
        spec.logical_bits,
        payload_offset,
        len(spec.payload),
        region_offset,
        region_bytes,
        spec.profile_q,
        spec.decoder_scale,
    )
    row[80:112] = _hex32(spec.source_digest, "source_digest")
    row[112:144] = _sha(spec.payload)
    row[144:152] = _sha(bytes(row[:144]))[:8]
    # Last eight bytes are reserved and remain zero.
    return bytes(row)


def _region_header(
    *,
    region_ordinal: int,
    owner_mask: int,
    stream_count: int,
    region_bytes: int,
    content_bytes: int,
    frame_area: bytes,
) -> bytes:
    header = bytearray(REGION_HEADER_BYTES)
    struct.pack_into(
        "<8sIIIIQQ",
        header,
        0,
        REGION_MAGIC,
        region_ordinal,
        owner_mask,
        stream_count,
        0,
        region_bytes,
        content_bytes,
    )
    header[40:72] = _sha(frame_area)
    header[72:104] = _sha(bytes(header[:72]))
    return bytes(header)


def _header(
    *,
    weights: int,
    experts: int,
    streams: int,
    baseline_object_bytes: int,
    audited_relative_mse: float,
    model_offset: int,
    model_bytes: int,
    immutable_offset: int,
    immutable_bytes: int,
    directory_offset: int,
    directory_bytes: int,
    shared_bytes: int,
    total_bytes: int,
    baseline_artifact_sha256: str,
    reconstruction_sha256: str,
    audit_binding_sha256: str,
    model_sha: bytes,
    immutable_sha: bytes,
    directory_sha: bytes,
    body_sha: bytes,
    binding_hashes: Mapping[str, str],
) -> bytes:
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<8sHHIIQII", header, 0, MAGIC, VERSION, HEADER_BYTES, PAGE_BYTES, 0, weights, experts, streams)
    struct.pack_into("<QdQQQQQQQQ", header, 36, baseline_object_bytes, audited_relative_mse, model_offset, model_bytes, immutable_offset, immutable_bytes, directory_offset, directory_bytes, shared_bytes, total_bytes)
    header[116:148] = _hex32(baseline_artifact_sha256, "baseline artifact")
    header[148:180] = _hex32(reconstruction_sha256, "reconstruction")
    header[180:212] = _hex32(audit_binding_sha256, "audit binding")
    header[212:244] = model_sha
    header[244:276] = immutable_sha
    header[276:308] = directory_sha
    header[308:340] = body_sha
    if set(binding_hashes) != set(_HEADER_BINDINGS):
        raise ValueError("exact container binding fields required")
    for index, name in enumerate(_HEADER_BINDINGS):
        begin = _HEADER_BINDINGS_BEGIN + 32 * index
        header[begin:begin + 32] = _hex32(binding_hashes[name], name)
    # Both integrity fields are zero for the SHA seal.  The CRC then covers
    # the header with its SHA seal present and only the CRC field zero.
    header[_HEADER_SEAL_BEGIN:_HEADER_SEAL_END] = b"\x00" * 32
    struct.pack_into("<I", header, _HEADER_CRC_OFFSET, 0)
    header[_HEADER_SEAL_BEGIN:_HEADER_SEAL_END] = _sha(bytes(header))
    struct.pack_into("<I", header, _HEADER_CRC_OFFSET, zlib.crc32(header) & 0xFFFFFFFF)
    return bytes(header)


def build_container(
    common: Any,
    *,
    model_packet: bytes,
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
    minimum_rate_bpw: float = 2.15,
) -> tuple[bytes, dict[str, Any]]:
    """Emit one complete, page-placed physical object.

    Rate-floor padding is literal zero padding in owner regions and is added in
    complete pages round-robin.  This preserves exact page ownership and makes
    cold traffic independently recomputable from the bytes.
    """
    if weights <= 0 or experts <= 0 or baseline_object_bytes <= 0:
        raise ValueError("container panel geometry")
    if not math.isfinite(audited_relative_mse) or audited_relative_mse <= 0.0:
        raise ValueError("audited relative MSE")
    if not math.isfinite(minimum_rate_bpw) or minimum_rate_bpw < 0.0:
        raise ValueError("minimum rate")
    # The physical decoder must consume the serialized model.
    common.deserialize_model(model_packet)
    if not regions:
        raise ValueError("at least one owner region")
    flat: list[StreamSpec] = []
    raw_frame_areas: list[bytes] = []
    owner_masks: list[int] = []
    for region in regions:
        _owners(region.owner_mask, experts)
        if not region.streams:
            raise ValueError("empty owner region")
        area = b"".join(_frame_bytes(row, region.owner_mask) for row in region.streams)
        raw_frame_areas.append(area)
        owner_masks.append(region.owner_mask)
        flat.extend(region.streams)
    if sorted(row.ordinal for row in flat) != list(range(len(flat))):
        raise ValueError("stream ordinals must be a complete canonical range")
    directory_bytes = len(flat) * DIRECTORY_RECORD_BYTES
    immutable_offset = HEADER_BYTES
    model_offset = _align(immutable_offset + len(immutable_state))
    directory_offset = _align(model_offset + len(model_packet))
    shared_bytes = _align(directory_offset + directory_bytes)
    region_lengths = [_align(REGION_HEADER_BYTES + len(area)) for area in raw_frame_areas]
    minimum_total = math.ceil(weights * minimum_rate_bpw / 8.0)
    current_total = shared_bytes + sum(region_lengths)
    pad_cursor = 0
    while current_total < minimum_total:
        region_lengths[pad_cursor % len(region_lengths)] += PAGE_BYTES
        current_total += PAGE_BYTES
        pad_cursor += 1
    region_offsets: list[int] = []
    cursor = shared_bytes
    for length in region_lengths:
        region_offsets.append(cursor)
        cursor += length
    total_bytes = cursor

    directory_rows: dict[int, bytes] = {}
    region_packets: list[bytes] = []
    for region_ordinal, (region, area, region_offset, region_bytes) in enumerate(
        zip(regions, raw_frame_areas, region_offsets, region_lengths, strict=True)
    ):
        frame_cursor = region_offset + REGION_HEADER_BYTES
        local_cursor = 0
        for spec in region.streams:
            frame = _frame_bytes(spec, region.owner_mask)
            directory_rows[spec.ordinal] = _directory_record(
                spec,
                owner_mask=region.owner_mask,
                region_ordinal=region_ordinal,
                region_offset=region_offset,
                region_bytes=region_bytes,
                frame_offset=frame_cursor + local_cursor,
            )
            local_cursor += len(frame)
        content_bytes = REGION_HEADER_BYTES + len(area)
        header = _region_header(
            region_ordinal=region_ordinal,
            owner_mask=region.owner_mask,
            stream_count=len(region.streams),
            region_bytes=region_bytes,
            content_bytes=content_bytes,
            frame_area=area,
        )
        packet = header + area + b"\x00" * (region_bytes - content_bytes)
        if len(packet) != region_bytes:
            raise AssertionError("region assembly geometry")
        region_packets.append(packet)
    directory = b"".join(directory_rows[index] for index in range(len(flat)))
    shared_tail = bytearray(shared_bytes - HEADER_BYTES)
    immutable_local = immutable_offset - HEADER_BYTES
    model_local = model_offset - HEADER_BYTES
    directory_local = directory_offset - HEADER_BYTES
    shared_tail[immutable_local:immutable_local + len(immutable_state)] = immutable_state
    shared_tail[model_local:model_local + len(model_packet)] = model_packet
    shared_tail[directory_local:directory_local + len(directory)] = directory
    body = bytes(shared_tail) + b"".join(region_packets)
    if len(body) != total_bytes - HEADER_BYTES:
        raise AssertionError("container body geometry")
    header = _header(
        weights=weights,
        experts=experts,
        streams=len(flat),
        baseline_object_bytes=baseline_object_bytes,
        audited_relative_mse=audited_relative_mse,
        model_offset=model_offset,
        model_bytes=len(model_packet),
        immutable_offset=immutable_offset,
        immutable_bytes=len(immutable_state),
        directory_offset=directory_offset,
        directory_bytes=len(directory),
        shared_bytes=shared_bytes,
        total_bytes=total_bytes,
        baseline_artifact_sha256=baseline_artifact_sha256,
        reconstruction_sha256=reconstruction_sha256,
        audit_binding_sha256=audit_binding_sha256,
        model_sha=_sha(model_packet),
        immutable_sha=_sha(immutable_state),
        directory_sha=_sha(directory),
        body_sha=_sha(body),
        binding_hashes=binding_hashes,
    )
    container = header + body
    parsed = parse_container(common, container)
    return container, physical_metrics(parsed)


def _parse_directory_row(raw: bytes) -> dict[str, Any]:
    if len(raw) != DIRECTORY_RECORD_BYTES:
        raise ValueError("directory row geometry")
    fields = struct.unpack_from("<IIIIQQQQQQH2xd", raw, 0)
    ordinal, owner_mask, region_ordinal, reserved, symbols, logical_bits, payload_offset, payload_bytes, region_offset, region_bytes, profile_q, decoder_scale = fields
    if reserved != 0 or raw[152:160] != b"\x00" * 8:
        raise ValueError("directory reserved bytes")
    if raw[144:152] != _sha(raw[:144])[:8]:
        raise ValueError("directory row seal")
    return {
        "ordinal": ordinal,
        "owner_mask": owner_mask,
        "region_ordinal": region_ordinal,
        "symbols": symbols,
        "logical_bits": logical_bits,
        "payload_offset": payload_offset,
        "payload_bytes": payload_bytes,
        "region_offset": region_offset,
        "region_bytes": region_bytes,
        "profile_q": profile_q,
        "decoder_scale": decoder_scale,
        "source_digest": raw[80:112].hex(),
        "payload_sha256": raw[112:144].hex(),
    }


def parse_container(common: Any, container: bytes) -> dict[str, Any]:
    if len(container) < HEADER_BYTES or len(container) % PAGE_BYTES:
        raise ValueError("container byte/page geometry")
    header = container[:HEADER_BYTES]
    magic, version, header_bytes, page_bytes, flags, weights, experts, streams = struct.unpack_from("<8sHHIIQII", header, 0)
    if (magic, version, header_bytes, page_bytes, flags) != (MAGIC, VERSION, HEADER_BYTES, PAGE_BYTES, 0):
        raise ValueError("container header constants")
    values = struct.unpack_from("<QdQQQQQQQQ", header, 36)
    baseline_object_bytes, audited_relative_mse, model_offset, model_bytes, immutable_offset, immutable_bytes, directory_offset, directory_bytes, shared_bytes, total_bytes = values
    clean_header = bytearray(header)
    observed_seal = bytes(clean_header[_HEADER_SEAL_BEGIN:_HEADER_SEAL_END])
    observed_crc = struct.unpack_from("<I", clean_header, _HEADER_CRC_OFFSET)[0]
    struct.pack_into("<I", clean_header, _HEADER_CRC_OFFSET, 0)
    if observed_crc != zlib.crc32(clean_header) & 0xFFFFFFFF:
        raise ValueError("container header CRC")
    clean_header[_HEADER_SEAL_BEGIN:_HEADER_SEAL_END] = b"\x00" * 32
    if observed_seal != _sha(bytes(clean_header)):
        raise ValueError("container header seal")
    if header[_HEADER_CRC_OFFSET + 4:] != b"\x00" * (HEADER_BYTES - _HEADER_CRC_OFFSET - 4):
        raise ValueError("container header reserved bytes")
    if total_bytes != len(container) or weights <= 0 or experts <= 0 or streams <= 0:
        raise ValueError("container header geometry")
    if not math.isfinite(audited_relative_mse) or audited_relative_mse <= 0.0:
        raise ValueError("container audited MSE")
    if immutable_offset != HEADER_BYTES or model_offset != _align(immutable_offset + immutable_bytes) or directory_offset != _align(model_offset + model_bytes):
        raise ValueError("shared-section canonical placement")
    if directory_bytes != streams * DIRECTORY_RECORD_BYTES or shared_bytes != _align(directory_offset + directory_bytes):
        raise ValueError("shared directory geometry")
    if shared_bytes > len(container) or shared_bytes % PAGE_BYTES:
        raise ValueError("shared placement")
    immutable_state = container[immutable_offset:immutable_offset + immutable_bytes]
    model_packet = container[model_offset:model_offset + model_bytes]
    directory_blob = container[directory_offset:directory_offset + directory_bytes]
    if _sha(model_packet) != header[212:244] or _sha(immutable_state) != header[244:276] or _sha(directory_blob) != header[276:308]:
        raise ValueError("shared-section digest")
    if _sha(container[HEADER_BYTES:]) != header[308:340]:
        raise ValueError("container body digest")
    padding_ranges = (
        (immutable_offset + immutable_bytes, model_offset),
        (model_offset + model_bytes, directory_offset),
        (directory_offset + directory_bytes, shared_bytes),
    )
    if any(any(container[begin:end]) for begin, end in padding_ranges):
        raise ValueError("nonzero shared alignment padding")
    candidate, frequencies = common.deserialize_model(model_packet)
    directory = [
        _parse_directory_row(directory_blob[index * DIRECTORY_RECORD_BYTES:(index + 1) * DIRECTORY_RECORD_BYTES])
        for index in range(streams)
    ]
    if [row["ordinal"] for row in directory] != list(range(streams)):
        raise ValueError("directory ordinal order")
    regions: dict[int, dict[str, Any]] = {}
    for row in directory:
        _owners(int(row["owner_mask"]), experts)
        key = int(row["region_ordinal"])
        pair = (int(row["region_offset"]), int(row["region_bytes"]), int(row["owner_mask"]))
        if key in regions and regions[key]["identity"] != pair:
            raise ValueError("inconsistent region directory")
        regions.setdefault(key, {"identity": pair, "rows": []})["rows"].append(row)
    if sorted(regions) != list(range(len(regions))):
        raise ValueError("region ordinal coverage")
    cursor = shared_bytes
    for region_ordinal in range(len(regions)):
        region = regions[region_ordinal]
        region_offset, region_bytes, owner_mask = region["identity"]
        if region_offset != cursor or region_offset % PAGE_BYTES or region_bytes % PAGE_BYTES or region_bytes < REGION_HEADER_BYTES:
            raise ValueError("region placement")
        packet = container[region_offset:region_offset + region_bytes]
        if len(packet) != region_bytes:
            raise ValueError("truncated region")
        magic_r, ordinal_r, mask_r, stream_count, reserved, region_bytes_r, content_bytes = struct.unpack_from("<8sIIIIQQ", packet, 0)
        if (magic_r, ordinal_r, mask_r, reserved, region_bytes_r) != (REGION_MAGIC, region_ordinal, owner_mask, 0, region_bytes):
            raise ValueError("region header")
        if stream_count != len(region["rows"]) or not REGION_HEADER_BYTES <= content_bytes <= region_bytes:
            raise ValueError("region content geometry")
        if packet[72:104] != _sha(packet[:72]) or packet[104:128] != b"\x00" * 24:
            raise ValueError("region header seal/reserved")
        frame_cursor = region_offset + REGION_HEADER_BYTES
        for row in sorted(region["rows"], key=lambda item: int(item["payload_offset"])):
            frame_begin = int(row["payload_offset"]) - FRAME_HEADER_BYTES
            if frame_begin != frame_cursor:
                raise ValueError("frame contiguity")
            frame_header = container[frame_begin:int(row["payload_offset"])]
            values_f = struct.unpack_from("<8sIIQQQH2xd", frame_header, 0)
            magic_f, ordinal_f, mask_f, symbols_f, logical_f, payload_bytes_f, profile_f, scale_f = values_f
            if (magic_f, ordinal_f, mask_f, symbols_f, logical_f, payload_bytes_f, profile_f) != (
                FRAME_MAGIC, row["ordinal"], owner_mask, row["symbols"], row["logical_bits"], row["payload_bytes"], row["profile_q"]
            ) or struct.pack("<d", scale_f) != struct.pack("<d", row["decoder_scale"]):
                raise ValueError("frame/directory mismatch")
            if frame_header[56:88].hex() != row["source_digest"] or frame_header[88:120].hex() != row["payload_sha256"] or frame_header[120:128] != _sha(frame_header[:120])[:8]:
                raise ValueError("frame digest")
            payload_begin = int(row["payload_offset"])
            payload_end = payload_begin + int(row["payload_bytes"])
            payload = container[payload_begin:payload_end]
            if _sha(payload).hex() != row["payload_sha256"]:
                raise ValueError("payload digest")
            if row["payload_bytes"] != (row["logical_bits"] + 7) // 8:
                raise ValueError("payload logical geometry")
            if row["logical_bits"] & 7 and payload[-1] & ((1 << (8 - (row["logical_bits"] & 7))) - 1):
                raise ValueError("payload terminal padding")
            row["payload"] = payload
            frame_end = _align(payload_end, 64)
            if any(container[payload_end:frame_end]):
                raise ValueError("nonzero frame alignment padding")
            frame_cursor = frame_end
        if frame_cursor - region_offset != content_bytes or packet[40:72] != _sha(container[region_offset + REGION_HEADER_BYTES:frame_cursor]):
            raise ValueError("region frame-area digest")
        if any(container[frame_cursor:region_offset + region_bytes]):
            raise ValueError("nonzero region/rate padding")
        region["offset"] = region_offset
        region["bytes"] = region_bytes
        region["owner_mask"] = owner_mask
        cursor += region_bytes
    if cursor != len(container):
        raise ValueError("region coverage does not end at EOF")
    return {
        "raw": container,
        "weights": weights,
        "experts": experts,
        "streams": streams,
        "baseline_object_bytes": baseline_object_bytes,
        "audited_relative_mse": audited_relative_mse,
        "baseline_artifact_sha256": header[116:148].hex(),
        "reconstruction_sha256": header[148:180].hex(),
        "audit_binding_sha256": header[180:212].hex(),
        "binding_hashes": {
            name: header[_HEADER_BINDINGS_BEGIN + 32 * index:_HEADER_BINDINGS_BEGIN + 32 * (index + 1)].hex()
            for index, name in enumerate(_HEADER_BINDINGS)
        },
        "model_packet": model_packet,
        "candidate": candidate,
        "frequencies": frequencies,
        "immutable_state": immutable_state,
        "directory": directory,
        "shared_bytes": shared_bytes,
        "regions": [regions[index] for index in range(len(regions))],
    }


def _page_set(begin: int, end: int) -> set[int]:
    if not 0 <= begin <= end:
        raise ValueError("page range")
    if begin == end:
        return set()
    return set(range(begin // PAGE_BYTES, (end - 1) // PAGE_BYTES + 1))


class InstrumentedColdReader:
    """Literal byte reader that records the exact physical page union."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.pages: set[int] = set()
        self.ranges: list[tuple[int, int]] = []

    def read(self, begin: int, end: int) -> bytes:
        if not 0 <= begin <= end <= len(self.raw):
            raise ValueError("instrumented read outside container")
        self.pages.update(_page_set(begin, end))
        self.ranges.append((begin, end))
        return self.raw[begin:end]


def instrument_expert_pages(parsed: dict[str, Any], expert: int) -> dict[str, Any]:
    experts = int(parsed["experts"])
    if not 0 <= expert < experts:
        raise ValueError("expert ordinal")
    reader = InstrumentedColdReader(parsed["raw"])
    # The decoder reads the fixed global metadata/model/directory prefix, then
    # only owner regions required by this routed expert.
    reader.read(0, int(parsed["shared_bytes"]))
    for region in parsed["regions"]:
        if int(region["owner_mask"]) & (1 << expert):
            begin = int(region["offset"])
            reader.read(begin, begin + int(region["bytes"]))
    return {
        "expert_ordinal": expert,
        "read_ranges": reader.ranges,
        "touched_page_indices": sorted(reader.pages),
        "touched_page_bytes": len(reader.pages) * PAGE_BYTES,
    }


def physical_metrics(parsed: dict[str, Any]) -> dict[str, Any]:
    total = len(parsed["raw"])
    weights = int(parsed["weights"])
    experts = int(parsed["experts"])
    shared = int(parsed["shared_bytes"])
    rows = []
    for expert in range(experts):
        trace = instrument_expert_pages(parsed, expert)
        pages = set(trace["touched_page_indices"])
        allocated = shared / experts
        formula = [f"{shared}/{experts}"]
        touched_regions = []
        for region in parsed["regions"]:
            owners = _owners(int(region["owner_mask"]), experts)
            if expert in owners:
                begin = int(region["offset"])
                end = begin + int(region["bytes"])
                pages.update(_page_set(begin, end))
                allocated += int(region["bytes"]) / len(owners)
                formula.append(f"{int(region['bytes'])}/{len(owners)}")
                touched_regions.append(int(region["identity"][0]))
        cold = len(pages) * PAGE_BYTES
        rows.append(
            {
                "expert_ordinal": expert,
                "touched_page_indices": sorted(pages),
                "touched_page_bytes": cold,
                "instrumented_read_ranges": trace["read_ranges"],
                "allocated_physical_denominator_bytes": allocated,
                "allocation_formula_terms": formula,
                "cold_read_amplification": cold / allocated,
                "touched_region_offsets": touched_regions,
            }
        )
    rate = 8.0 * total / weights
    mse = float(parsed["audited_relative_mse"])
    f_value = mse * math.pow(2.0, 2.0 * rate)
    allocated_sum = sum(float(row["allocated_physical_denominator_bytes"]) for row in rows)
    if abs(allocated_sum - total) > max(1e-8, 1e-12 * total):
        raise AssertionError("ownership allocation does not sum to physical bytes")
    return {
        "actual_container_bytes": total,
        "actual_physical_rate_bpw": rate,
        "baseline_object_bytes": int(parsed["baseline_object_bytes"]),
        "net_physical_saving_bpw": 8.0 * (int(parsed["baseline_object_bytes"]) - total) / weights,
        "audited_identical_reconstruction_relative_mse": mse,
        "F_from_actual_bytes_and_identical_reconstruction": f_value,
        "passes_rate_interval": 2.15 <= rate <= 2.5,
        "passes_F_target": f_value <= 0.8,
        "read_denominator_definition": "sum of each touched physical section divided by its exact owner count; global term is Bshared/E, never total/E",
        "experts": rows,
        "maximum_cold_read_amplification": max(float(row["cold_read_amplification"]) for row in rows),
        "passes_cold_read_below_2x": max(float(row["cold_read_amplification"]) for row in rows) < 2.0,
        "ownership_allocated_bytes_sum": allocated_sum,
    }


def canonical_rebuild(common: Any, parsed: dict[str, Any]) -> bytes:
    """Re-emit a parsed container through the canonical builder.

    The original minimum rate is inferred from the literal byte count, causing
    the same number of whole-page rate pads to be redistributed deterministically.
    This is used only after payloads have been independently decoded and
    re-encoded.
    """
    region_specs = []
    for region in parsed["regions"]:
        specs = []
        for row in sorted(region["rows"], key=lambda item: int(item["ordinal"])):
            specs.append(StreamSpec(
                ordinal=int(row["ordinal"]),
                symbols=int(row["symbols"]),
                logical_bits=int(row["logical_bits"]),
                payload=bytes(row["payload"]),
                source_digest=str(row["source_digest"]),
                profile_q=int(row["profile_q"]),
                decoder_scale=float(row["decoder_scale"]),
            ))
        region_specs.append(RegionSpec(int(region["owner_mask"]), tuple(specs)))
    inferred_floor = 8.0 * len(parsed["raw"]) / int(parsed["weights"])
    rebuilt, _ = build_container(
        common,
        model_packet=bytes(parsed["model_packet"]),
        immutable_state=bytes(parsed["immutable_state"]),
        regions=region_specs,
        weights=int(parsed["weights"]),
        experts=int(parsed["experts"]),
        baseline_object_bytes=int(parsed["baseline_object_bytes"]),
        audited_relative_mse=float(parsed["audited_relative_mse"]),
        baseline_artifact_sha256=str(parsed["baseline_artifact_sha256"]),
        reconstruction_sha256=str(parsed["reconstruction_sha256"]),
        audit_binding_sha256=str(parsed["audit_binding_sha256"]),
        binding_hashes=dict(parsed["binding_hashes"]),
        minimum_rate_bpw=inferred_floor,
    )
    return rebuilt
