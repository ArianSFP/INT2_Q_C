#!/usr/bin/env python3
"""Literal page-local multi-owner packet used by the bound v1 driver."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Sequence


MAGIC = b"ETCQSBV1"
FRAME_MAGIC = b"ETCQFRV1"
VERSION = 1
HEADER_BYTES = 256
DIRECTORY_RECORD_BYTES = 64
FRAME_HEADER_BYTES = 128
PAGE_BYTES = 4096


class PacketError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _owner_mask(owners: Sequence[int], experts: int) -> bytes:
    values = tuple(int(value) for value in owners)
    require(values == tuple(sorted(set(values))) and values and
            0 <= values[0] and values[-1] < experts and experts <= 256,
            "frame owner set")
    mask = bytearray(32)
    for owner in values:
        mask[owner >> 3] |= 1 << (owner & 7)
    return bytes(mask)


def _mask_owners(mask: bytes, experts: int) -> tuple[int, ...]:
    require(type(mask) is bytes and len(mask) == 32 and 1 <= experts <= 256,
            "owner mask")
    require(not any(mask[(experts + 7) // 8:]) and
            (experts & 7 == 0 or not (mask[experts >> 3] &
                                      ~((1 << (experts & 7)) - 1))),
            "owner mask tail")
    owners = tuple(index for index in range(experts)
                   if mask[index >> 3] & (1 << (index & 7)))
    require(owners, "nonempty owner mask")
    return owners


def _align(value: int) -> int:
    return (value + PAGE_BYTES - 1) // PAGE_BYTES * PAGE_BYTES


@dataclass(frozen=True)
class FrameInput:
    owners: tuple[int, ...]
    weight_count: int
    labels_u8: bytes
    payload: bytes
    logical_bits: int

    def validate(self, experts: int) -> None:
        _owner_mask(self.owners, experts)
        require(type(self.weight_count) is int and self.weight_count > 0 and
                type(self.labels_u8) is bytes and
                len(self.labels_u8) == self.weight_count and
                all(value < 64 for value in self.labels_u8), "frame labels")
        require(type(self.payload) is bytes and type(self.logical_bits) is int and
                0 <= self.logical_bits <= 8 * len(self.payload), "frame payload")
        if self.logical_bits & 7:
            require(not (self.payload[-1] &
                         ((1 << (8 - (self.logical_bits & 7))) - 1)),
                    "frame zero terminal bits")


def _frame_header(ordinal: int, frame: FrameInput, experts: int) -> bytes:
    frame.validate(experts)
    header = bytearray(FRAME_HEADER_BYTES)
    struct.pack_into("<8sHHIQQQQ", header, 0, FRAME_MAGIC, VERSION,
                     FRAME_HEADER_BYTES, ordinal, frame.weight_count,
                     len(frame.payload), frame.logical_bits, len(frame.labels_u8))
    header[48:80] = hashlib.sha256(frame.labels_u8).digest()
    header[80:112] = hashlib.sha256(frame.payload).digest()
    header[112:128] = hashlib.sha256(_owner_mask(frame.owners, experts)).digest()[:16]
    return bytes(header)


def build_packet(*, topology: bytes, frequencies: bytes, centroids: bytes,
                 frames: Sequence[FrameInput], weights: int,
                 experts: int) -> bytes:
    require(type(topology) is bytes and topology and type(frequencies) is bytes and
            frequencies and type(centroids) is bytes and centroids and frames,
            "packet nonempty sections")
    require(type(weights) is int and weights > 0 and type(experts) is int and
            1 <= experts <= 256, "packet dimensions")
    for frame in frames:
        frame.validate(experts)
    require(sum(frame.weight_count for frame in frames) == weights,
            "frame weight conservation")
    directory_bytes = DIRECTORY_RECORD_BYTES * len(frames)
    global_end = (HEADER_BYTES + len(topology) + len(frequencies) +
                  len(centroids) + directory_bytes)
    global_padded_end = _align(global_end)
    rows = []
    frame_blobs = []
    cursor = global_padded_end
    for ordinal, frame in enumerate(frames):
        header = _frame_header(ordinal, frame, experts)
        logical = header + frame.payload
        region = logical + bytes(_align(len(logical)) - len(logical))
        rows.append((_owner_mask(frame.owners, experts), cursor, len(region),
                     len(frame.payload), frame.logical_bits))
        frame_blobs.append(region)
        cursor += len(region)
    total = cursor
    directory = bytearray(directory_bytes)
    for ordinal, row in enumerate(rows):
        mask, offset, region_bytes, payload_bytes, logical_bits = row
        begin = ordinal * DIRECTORY_RECORD_BYTES
        directory[begin:begin + 32] = mask
        struct.pack_into("<QQQQ", directory, begin + 32, offset, region_bytes,
                         payload_bytes, logical_bits)
    body = (topology + frequencies + centroids + bytes(directory) +
            bytes(global_padded_end - global_end) + b"".join(frame_blobs))
    require(len(body) == total - HEADER_BYTES, "packet body geometry")
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<8sHHIQQIIIIIIQQQ", header, 0, MAGIC, VERSION,
                     HEADER_BYTES, 1, total, weights, experts, len(frames),
                     len(topology), len(frequencies), len(centroids),
                     directory_bytes, global_end, global_padded_end, len(body))
    header[96:128] = hashlib.sha256(body).digest()
    header[128:160] = hashlib.sha256(topology).digest()
    header[160:192] = hashlib.sha256(frequencies).digest()
    header[192:224] = hashlib.sha256(centroids).digest()
    header[224:256] = hashlib.sha256(header[:224]).digest()
    packet = bytes(header) + body
    require(len(packet) == total and total % PAGE_BYTES == 0,
            "packet physical geometry")
    return packet


def parse_packet(packet: bytes) -> dict[str, Any]:
    require(type(packet) is bytes and len(packet) >= 2 * PAGE_BYTES and
            len(packet) % PAGE_BYTES == 0, "packet physical lower bound")
    header = packet[:HEADER_BYTES]
    values = struct.unpack_from("<8sHHIQQIIIIIIQQQ", header, 0)
    (magic, version, header_bytes, flags, total, weights, experts, frame_count,
     topology_bytes, frequency_bytes, centroid_bytes, directory_bytes,
     global_end, global_padded_end, body_bytes) = values
    require((magic, version, header_bytes, flags, total, body_bytes) ==
            (MAGIC, VERSION, HEADER_BYTES, 1, len(packet), len(packet) - HEADER_BYTES),
            "packet header identity")
    require(1 <= experts <= 256 and weights > 0 and frame_count > 0 and
            all(value > 0 for value in (topology_bytes, frequency_bytes,
                                        centroid_bytes)), "packet dimensions")
    require(directory_bytes == DIRECTORY_RECORD_BYTES * frame_count,
            "directory bytes")
    expected_global_end = (HEADER_BYTES + topology_bytes + frequency_bytes +
                           centroid_bytes + directory_bytes)
    require(global_end == expected_global_end and
            global_padded_end == _align(global_end) and
            global_padded_end < total, "global prefix geometry")
    require(not any(header[88:96]) and
            header[224:256] == hashlib.sha256(header[:224]).digest() and
            header[96:128] == hashlib.sha256(packet[HEADER_BYTES:]).digest(),
            "packet header hashes/reserved")
    cursor = HEADER_BYTES
    topology = packet[cursor:cursor + topology_bytes]
    cursor += topology_bytes
    frequencies = packet[cursor:cursor + frequency_bytes]
    cursor += frequency_bytes
    centroids = packet[cursor:cursor + centroid_bytes]
    cursor += centroid_bytes
    directory = packet[cursor:cursor + directory_bytes]
    cursor += directory_bytes
    require(header[128:160] == hashlib.sha256(topology).digest() and
            header[160:192] == hashlib.sha256(frequencies).digest() and
            header[192:224] == hashlib.sha256(centroids).digest() and
            cursor == global_end and not any(packet[cursor:global_padded_end]),
            "global sections/padding")
    frames_out = []
    expected_offset = global_padded_end
    weight_sum = 0
    for ordinal in range(frame_count):
        begin = ordinal * DIRECTORY_RECORD_BYTES
        mask = directory[begin:begin + 32]
        owners = _mask_owners(mask, experts)
        offset, region_bytes, payload_bytes, logical_bits = struct.unpack_from(
            "<QQQQ", directory, begin + 32)
        require(offset == expected_offset and region_bytes >= FRAME_HEADER_BYTES and
                region_bytes % PAGE_BYTES == 0 and offset + region_bytes <= total,
                "frame region geometry")
        raw_header = packet[offset:offset + FRAME_HEADER_BYTES]
        fields = struct.unpack_from("<8sHHIQQQQ", raw_header, 0)
        (frame_magic, frame_version, frame_header_bytes, frame_ordinal,
         weight_count, header_payload_bytes, header_logical_bits, labels_count) = fields
        require((frame_magic, frame_version, frame_header_bytes, frame_ordinal) ==
                (FRAME_MAGIC, VERSION, FRAME_HEADER_BYTES, ordinal) and
                header_payload_bytes == payload_bytes and
                header_logical_bits == logical_bits and
                labels_count == weight_count and weight_count > 0 and
                logical_bits <= 8 * payload_bytes, "frame header identity")
        require(raw_header[112:128] == hashlib.sha256(mask).digest()[:16],
                "frame owner binding")
        payload = packet[offset + FRAME_HEADER_BYTES:
                         offset + FRAME_HEADER_BYTES + payload_bytes]
        require(raw_header[80:112] == hashlib.sha256(payload).digest(),
                "frame payload hash")
        if logical_bits & 7:
            require(not (payload[-1] & ((1 << (8 - (logical_bits & 7))) - 1)),
                    "frame terminal bits")
        logical_end = offset + FRAME_HEADER_BYTES + payload_bytes
        require(not any(packet[logical_end:offset + region_bytes]),
                "frame zero physical padding")
        frames_out.append({
            "ordinal": ordinal, "owners": owners, "owner_mask": mask,
            "offset": offset, "region_bytes": region_bytes,
            "weight_count": int(weight_count), "labels_count": int(labels_count),
            "labels_sha256": raw_header[48:80].hex(), "payload": payload,
            "payload_bytes": int(payload_bytes), "logical_bits": int(logical_bits),
            "raw_header": raw_header,
        })
        weight_sum += int(weight_count)
        expected_offset = offset + region_bytes
    require(expected_offset == total and weight_sum == weights,
            "packet frame coverage")
    ledger = {
        "header_bytes": HEADER_BYTES,
        "model_bytes": int(topology_bytes + frequency_bytes),
        "topology_bytes": int(topology_bytes),
        "frequency_bytes": int(frequency_bytes),
        "centroid_bytes": int(centroid_bytes),
        "directory_bytes": int(directory_bytes),
        "frame_header_bytes": FRAME_HEADER_BYTES * int(frame_count),
        "payload_bytes": sum(row["payload_bytes"] for row in frames_out),
        "padding_bytes": (global_padded_end - global_end) + sum(
            row["region_bytes"] - FRAME_HEADER_BYTES - row["payload_bytes"]
            for row in frames_out),
        "total_bytes": len(packet),
    }
    physical = (ledger["header_bytes"] + ledger["model_bytes"] +
                ledger["centroid_bytes"] + ledger["directory_bytes"] +
                ledger["frame_header_bytes"] + ledger["payload_bytes"] +
                ledger["padding_bytes"])
    require(physical == ledger["total_bytes"] and
            ledger["model_bytes"] == ledger["topology_bytes"] +
            ledger["frequency_bytes"], "literal byte ledger conservation")
    return {
        "weights": int(weights), "experts": int(experts),
        "topology": topology, "frequencies": frequencies,
        "centroids": centroids, "frames": tuple(frames_out),
        "global_end": int(global_end), "global_padded_end": int(global_padded_end),
        "total_bytes": len(packet), "packet_sha256": sha256(packet),
        "byte_ledger": ledger,
    }


def canonical_reencode(parsed: dict[str, Any]) -> bytes:
    frames = []
    for row in parsed["frames"]:
        # Labels themselves are not part of the decoder input, only their
        # authenticated digest.  Rebuild the bytes directly and compare below.
        frames.append(row)
    # Independent parse already proves all canonical layout rules.  A bytewise
    # canonical rebuild from decoded fields is implemented in
    # independent_decoder.py and is the promotion authority.
    raise PacketError("USE_INDEPENDENT_DECODER_FOR_CANONICAL_REENCODE")


def owner_read_trace(packet: bytes, expert: int) -> dict[str, Any]:
    parsed = parse_packet(packet)
    require(type(expert) is int and 0 <= expert < parsed["experts"],
            "routed expert")
    ranges = [(0, parsed["global_padded_end"])]
    for frame in parsed["frames"]:
        if expert in frame["owners"]:
            ranges.append((frame["offset"], frame["offset"] + frame["region_bytes"]))
    require(len(ranges) > 1, "routed expert has owner-local frame")
    requested = sum(end - begin for begin, end in ranges)
    # Regions are canonical, page-aligned and non-overlapping.
    require(all(begin % PAGE_BYTES == end % PAGE_BYTES == 0 and begin < end
                for begin, end in ranges), "read range alignment")
    ordered = sorted(ranges)
    require(all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:])),
            "read ranges nonoverlap")
    attributable = parsed["global_padded_end"] / parsed["experts"]
    for frame in parsed["frames"]:
        if expert in frame["owners"]:
            attributable += frame["region_bytes"] / len(frame["owners"])
    amplification = requested / attributable
    range_rows = [{"begin": begin, "end": end,
                   "sha256": sha256(packet[begin:end])}
                  for begin, end in ranges]
    return {
        "schema": "epsilon-tcq-bound-owner-read-trace-v1",
        "packet_sha256": parsed["packet_sha256"],
        "expert": expert, "ranges": range_rows,
        "read_request_count": len(ranges),
        "requested_bytes": requested, "unique_requested_bytes": requested,
        "touched_page_bytes": requested,
        "attributable_physical_bytes": attributable,
        "cold_read_amplification": amplification,
        "compressed_expert_second_pass_count": 0,
    }
