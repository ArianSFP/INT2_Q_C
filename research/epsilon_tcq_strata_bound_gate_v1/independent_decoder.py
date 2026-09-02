#!/usr/bin/env python3
"""Independent ETCQSBV1 decoder and canonical byte re-encoder.

This module intentionally imports no packet builder or driver code.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any


MAGIC = b"ETCQSBV1"
FRAME_MAGIC = b"ETCQFRV1"
VERSION = 1
HEADER_BYTES = 256
DIRECTORY_RECORD_BYTES = 64
FRAME_HEADER_BYTES = 128
PAGE_BYTES = 4096


class DecodeError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise DecodeError(message)


def _align(value: int) -> int:
    return (value + PAGE_BYTES - 1) // PAGE_BYTES * PAGE_BYTES


def _owners(mask: bytes, experts: int) -> tuple[int, ...]:
    need(len(mask) == 32 and 1 <= experts <= 256, "owner mask")
    output = tuple(index for index in range(experts)
                   if mask[index >> 3] & (1 << (index & 7)))
    need(output, "owner set")
    for index in range(experts, 256):
        need(not (mask[index >> 3] & (1 << (index & 7))), "owner tail")
    return output


def decode_and_reencode(raw: bytes) -> dict[str, Any]:
    need(type(raw) is bytes and len(raw) >= 2 * PAGE_BYTES and
         len(raw) % PAGE_BYTES == 0, "physical packet")
    header = raw[:HEADER_BYTES]
    fields = struct.unpack_from("<8sHHIQQIIIIIIQQQ", header, 0)
    (magic, version, header_size, flags, total, weights, experts, count,
     topology_size, frequency_size, centroid_size, directory_size,
     global_end, global_page_end, body_size) = fields
    need((magic, version, header_size, flags) == (MAGIC, VERSION, HEADER_BYTES, 1),
         "header identity")
    need(total == len(raw) and body_size == total - HEADER_BYTES and
         directory_size == count * DIRECTORY_RECORD_BYTES and count > 0,
         "header geometry")
    need(global_end == HEADER_BYTES + topology_size + frequency_size +
         centroid_size + directory_size and global_page_end == _align(global_end),
         "global geometry")
    need(header[224:256] == hashlib.sha256(header[:224]).digest() and
         header[96:128] == hashlib.sha256(raw[HEADER_BYTES:]).digest() and
         not any(header[88:96]), "header authentication")
    cursor = HEADER_BYTES
    topology = raw[cursor:cursor + topology_size]
    cursor += topology_size
    frequencies = raw[cursor:cursor + frequency_size]
    cursor += frequency_size
    centroids = raw[cursor:cursor + centroid_size]
    cursor += centroid_size
    directory = raw[cursor:cursor + directory_size]
    cursor += directory_size
    need(cursor == global_end and not any(raw[cursor:global_page_end]),
         "global padding")
    need(header[128:160] == hashlib.sha256(topology).digest() and
         header[160:192] == hashlib.sha256(frequencies).digest() and
         header[192:224] == hashlib.sha256(centroids).digest(),
         "global section hashes")
    frames = []
    expected_offset = global_page_end
    decoded_weights = 0
    for ordinal in range(count):
        begin = ordinal * DIRECTORY_RECORD_BYTES
        mask = directory[begin:begin + 32]
        owners = _owners(mask, experts)
        offset, region_size, payload_size, logical_bits = struct.unpack_from(
            "<QQQQ", directory, begin + 32)
        need(offset == expected_offset and region_size % PAGE_BYTES == 0 and
             region_size >= FRAME_HEADER_BYTES + payload_size and
             offset + region_size <= total, "frame directory")
        frame_header = raw[offset:offset + FRAME_HEADER_BYTES]
        values = struct.unpack_from("<8sHHIQQQQ", frame_header, 0)
        (frame_magic, frame_version, frame_size, frame_ordinal, frame_weights,
         header_payload, header_logical, label_count) = values
        need((frame_magic, frame_version, frame_size, frame_ordinal) ==
             (FRAME_MAGIC, VERSION, FRAME_HEADER_BYTES, ordinal), "frame identity")
        need(frame_weights == label_count and frame_weights > 0 and
             header_payload == payload_size and header_logical == logical_bits and
             logical_bits <= 8 * payload_size, "frame geometry")
        payload = raw[offset + FRAME_HEADER_BYTES:
                      offset + FRAME_HEADER_BYTES + payload_size]
        need(frame_header[80:112] == hashlib.sha256(payload).digest() and
             frame_header[112:128] == hashlib.sha256(mask).digest()[:16],
             "frame hashes")
        if logical_bits & 7:
            need(not (payload[-1] & ((1 << (8 - (logical_bits & 7))) - 1)),
                 "terminal bits")
        need(not any(raw[offset + FRAME_HEADER_BYTES + payload_size:
                         offset + region_size]), "frame padding")
        frames.append((mask, owners, offset, region_size, payload, logical_bits,
                       frame_weights, frame_header[48:80]))
        decoded_weights += frame_weights
        expected_offset = offset + region_size
    need(expected_offset == total and decoded_weights == weights, "frame coverage")

    rebuilt_directory = bytearray(directory_size)
    rebuilt_regions = []
    rebuilt_cursor = global_page_end
    for ordinal, (mask, _owners_value, _offset, region_size, payload,
                  logical_bits, frame_weights, labels_digest) in enumerate(frames):
        frame_header = bytearray(FRAME_HEADER_BYTES)
        struct.pack_into("<8sHHIQQQQ", frame_header, 0, FRAME_MAGIC, VERSION,
                         FRAME_HEADER_BYTES, ordinal, frame_weights, len(payload),
                         logical_bits, frame_weights)
        frame_header[48:80] = labels_digest
        frame_header[80:112] = hashlib.sha256(payload).digest()
        frame_header[112:128] = hashlib.sha256(mask).digest()[:16]
        region = bytes(frame_header) + payload
        region += bytes(region_size - len(region))
        row = ordinal * DIRECTORY_RECORD_BYTES
        rebuilt_directory[row:row + 32] = mask
        struct.pack_into("<QQQQ", rebuilt_directory, row + 32, rebuilt_cursor,
                         region_size, len(payload), logical_bits)
        rebuilt_regions.append(region)
        rebuilt_cursor += region_size
    rebuilt_body = (topology + frequencies + centroids + bytes(rebuilt_directory) +
                    bytes(global_page_end - global_end) + b"".join(rebuilt_regions))
    rebuilt_header = bytearray(HEADER_BYTES)
    struct.pack_into("<8sHHIQQIIIIIIQQQ", rebuilt_header, 0, MAGIC, VERSION,
                     HEADER_BYTES, 1, len(raw), weights, experts, count,
                     topology_size, frequency_size, centroid_size, directory_size,
                     global_end, global_page_end, len(rebuilt_body))
    rebuilt_header[96:128] = hashlib.sha256(rebuilt_body).digest()
    rebuilt_header[128:160] = hashlib.sha256(topology).digest()
    rebuilt_header[160:192] = hashlib.sha256(frequencies).digest()
    rebuilt_header[192:224] = hashlib.sha256(centroids).digest()
    rebuilt_header[224:256] = hashlib.sha256(rebuilt_header[:224]).digest()
    rebuilt = bytes(rebuilt_header) + rebuilt_body
    need(rebuilt == raw, "canonical independent byte reencode")
    return {
        "schema": "epsilon-tcq-bound-independent-decode-v1",
        "status": "PASS_LITERAL_DECODE_AND_CANONICAL_REENCODE",
        "packet_sha256": hashlib.sha256(raw).hexdigest(),
        "packet_bytes": len(raw), "weights": int(weights),
        "experts": int(experts), "frames": int(count),
        "all_labels_are_current_strata_indices_by_contract": True,
        "canonical_reencode_matches": True,
    }
