#!/usr/bin/env python3
"""Literal finite packet grammar and one-pass projection for epsilon-TCQ v0."""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


PACKET_MAGIC = b"ETCQWF01"
MODEL_MAGIC = b"ETMODEL1"
CENTROID_MAGIC = b"ETCENT01"
FRAME_MAGIC = b"ETFRAME1"
PACKET_HEADER_BYTES = 256
MODEL_HEADER_BYTES = 96
CENTROID_HEADER_BYTES = 96
FRAME_HEADER_BYTES = 128
PAGE_BYTES = 4096
INTERFACE_IDS = {
    "strata_sc_6bit_legal_replay": 1,
    "direct_int2_4level_new_codec": 2,
}
INTERFACE_NAMES = {value: key for key, value in INTERFACE_IDS.items()}
MODE_IDS = {"nominal": 0, "local": 1, "state": 2, "state_permuted": 3}
MODE_NAMES = {value: key for key, value in MODE_IDS.items()}


class PacketError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def sha256(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zero_tail(payload: bytes, begin: int, end: int, label: str) -> None:
    require(0 <= begin <= end <= len(payload) and not any(payload[begin:end]),
            f"{label} zero reserved bytes")


def serialize_model(interface: str, model: Any) -> bytes:
    require(interface in INTERFACE_IDS, "model interface")
    model.validate()
    frequencies = struct.pack(
        "<" + "H" * len(model.frequencies_q16), *model.frequencies_q16)
    header = bytearray(MODEL_HEADER_BYTES)
    struct.pack_into(
        "<8sIIIIIIII", header, 0, MODEL_MAGIC, 1,
        INTERFACE_IDS[interface], model.candidate.selector,
        model.candidate.states, model.candidate.reset, 16,
        len(model.frequencies_q16), len(frequencies))
    header[40:72] = sha256(frequencies)
    struct.pack_into("<I", header, 72, zlib.crc32(header[:72]) & 0xFFFFFFFF)
    return bytes(header) + frequencies


def parse_model(payload: bytes, core: Any) -> tuple[str, Any]:
    require(type(payload) is bytes and len(payload) >= MODEL_HEADER_BYTES,
            "model packet lower bound")
    header = payload[:MODEL_HEADER_BYTES]
    fields = struct.unpack_from("<8sIIIIIIII", header, 0)
    magic, version, interface_id, selector, states, reset, contexts, count, data_bytes = fields
    require(magic == MODEL_MAGIC and version == 1 and
            interface_id in INTERFACE_NAMES and contexts == 16,
            "model header identity")
    require(0 <= selector < len(core.FROZEN_BANK) and
            core.FROZEN_BANK[selector] ==
            (core.FROZEN_BANK[selector][0], states, reset),
            "model frozen selector geometry")
    require(count == states * 16 and data_bytes == 2 * count and
            len(payload) == MODEL_HEADER_BYTES + data_bytes,
            "model table geometry")
    require(header[40:72] == sha256(payload[MODEL_HEADER_BYTES:]) and
            struct.unpack_from("<I", header, 72)[0] ==
            (zlib.crc32(header[:72]) & 0xFFFFFFFF), "model hashes")
    _zero_tail(header, 76, MODEL_HEADER_BYTES, "model")
    frequencies = struct.unpack_from("<" + "H" * count,
                                     payload, MODEL_HEADER_BYTES)
    candidate = core.ModelCandidate(*core.FROZEN_BANK[selector])
    model = core.FittedModel(candidate, tuple(int(value) for value in frequencies))
    model.validate()
    require(serialize_model(INTERFACE_NAMES[interface_id], model) == payload,
            "canonical model packet")
    return INTERFACE_NAMES[interface_id], model


def serialize_centroid(interface: str, head: Any) -> bytes:
    require(interface in INTERFACE_IDS, "centroid interface")
    head.validate()
    values = b"".join(struct.pack("<e", value) for value in head.values)
    header = bytearray(CENTROID_HEADER_BYTES)
    struct.pack_into(
        "<8sIIIIIIII", header, 0, CENTROID_MAGIC, 1,
        INTERFACE_IDS[interface], MODE_IDS[head.mode], head.states,
        head.labels, len(head.values), len(values), 2)
    header[40:72] = sha256(values)
    struct.pack_into("<I", header, 72, zlib.crc32(header[:72]) & 0xFFFFFFFF)
    return bytes(header) + values


def parse_centroid(payload: bytes, core: Any) -> tuple[str, Any]:
    require(type(payload) is bytes and len(payload) >= CENTROID_HEADER_BYTES,
            "centroid packet lower bound")
    header = payload[:CENTROID_HEADER_BYTES]
    fields = struct.unpack_from("<8sIIIIIIII", header, 0)
    magic, version, interface_id, mode_id, states, labels, count, data_bytes, width = fields
    require(magic == CENTROID_MAGIC and version == 1 and
            interface_id in INTERFACE_NAMES and mode_id in MODE_NAMES and width == 2,
            "centroid header identity")
    require(labels == (64 if interface_id == 1 else 4) and
            data_bytes == 2 * count and
            len(payload) == CENTROID_HEADER_BYTES + data_bytes,
            "centroid geometry")
    require(header[40:72] == sha256(payload[CENTROID_HEADER_BYTES:]) and
            struct.unpack_from("<I", header, 72)[0] ==
            (zlib.crc32(header[:72]) & 0xFFFFFFFF), "centroid hashes")
    _zero_tail(header, 76, CENTROID_HEADER_BYTES, "centroid")
    values = tuple(struct.unpack_from("<e", payload,
                                      CENTROID_HEADER_BYTES + 2 * index)[0]
                   for index in range(count))
    head = core.CentroidHead(MODE_NAMES[mode_id], states, labels, values)
    head.validate()
    require(serialize_centroid(INTERFACE_NAMES[interface_id], head) == payload,
            "canonical centroid packet")
    return INTERFACE_NAMES[interface_id], head


def _label_hash(labels: Sequence[int]) -> bytes:
    require(all(type(value) is int and 0 <= value <= 255 for value in labels),
            "label hash values")
    return sha256(bytes(labels))


def build_packet(interface: str, model: Any, head: Any,
                 labels: Sequence[int], payload: bytes,
                 logical_bits: int) -> bytes:
    require(interface in INTERFACE_IDS and labels and type(payload) is bytes and
            0 <= logical_bits <= 8 * len(payload), "packet inputs")
    require(head.states == model.candidate.states and
            head.labels == (64 if interface ==
                            "strata_sc_6bit_legal_replay" else 4),
            "model/centroid/interface geometry")
    if logical_bits & 7:
        require(not (payload[-1] & ((1 << (8 - (logical_bits & 7))) - 1)),
                "payload canonical tail bits")
    model_packet = serialize_model(interface, model)
    centroid_packet = serialize_centroid(interface, head)
    frame = bytearray(FRAME_HEADER_BYTES)
    struct.pack_into("<8sIIQQQ", frame, 0, FRAME_MAGIC, 1,
                     INTERFACE_IDS[interface], len(labels), logical_bits,
                     len(payload))
    frame[40:72] = _label_hash(labels)
    frame[72:104] = sha256(payload)
    struct.pack_into("<I", frame, 104, zlib.crc32(frame[:104]) & 0xFFFFFFFF)
    frame_packet = bytes(frame) + payload
    model_offset = PACKET_HEADER_BYTES
    centroid_offset = model_offset + len(model_packet)
    frame_offset = centroid_offset + len(centroid_packet)
    total = frame_offset + len(frame_packet)
    header = bytearray(PACKET_HEADER_BYTES)
    struct.pack_into(
        "<8sIIQQQQQQQQ", header, 0, PACKET_MAGIC, 1,
        INTERFACE_IDS[interface], total, len(labels), model_offset,
        len(model_packet), centroid_offset, len(centroid_packet),
        frame_offset, len(frame_packet))
    header[96:128] = sha256(model_packet)
    header[128:160] = sha256(centroid_packet)
    header[160:192] = sha256(frame_packet)
    body = model_packet + centroid_packet + frame_packet
    header[192:224] = sha256(body)
    header[224:256] = sha256(header[:224])
    packet = bytes(header) + body
    require(len(packet) == total, "packet exact total")
    return packet


def parse_packet(packet: bytes, core: Any) -> dict[str, Any]:
    require(type(packet) is bytes and len(packet) >= PACKET_HEADER_BYTES,
            "packet lower bound")
    header = packet[:PACKET_HEADER_BYTES]
    fields = struct.unpack_from("<8sIIQQQQQQQQ", header, 0)
    (magic, version, interface_id, total, count, model_offset, model_bytes,
     centroid_offset, centroid_bytes, frame_offset, frame_bytes) = fields
    require(magic == PACKET_MAGIC and version == 1 and
            interface_id in INTERFACE_NAMES and total == len(packet) and count > 0,
            "packet header identity")
    require(model_offset == PACKET_HEADER_BYTES and
            centroid_offset == model_offset + model_bytes and
            frame_offset == centroid_offset + centroid_bytes and
            total == frame_offset + frame_bytes,
            "packet contiguous section layout")
    model_packet = packet[model_offset:model_offset + model_bytes]
    centroid_packet = packet[centroid_offset:centroid_offset + centroid_bytes]
    frame_packet = packet[frame_offset:frame_offset + frame_bytes]
    require(header[96:128] == sha256(model_packet) and
            header[128:160] == sha256(centroid_packet) and
            header[160:192] == sha256(frame_packet) and
            header[192:224] == sha256(packet[PACKET_HEADER_BYTES:]) and
            header[224:256] == sha256(header[:224]), "packet hashes")
    _zero_tail(header, 80, 96, "packet")
    model_interface, model = parse_model(model_packet, core)
    centroid_interface, head = parse_centroid(centroid_packet, core)
    interface = INTERFACE_NAMES[interface_id]
    require(model_interface == centroid_interface == interface,
            "packet interface consistency")
    require(head.states == model.candidate.states,
            "packet model/centroid state geometry")
    require(len(frame_packet) >= FRAME_HEADER_BYTES, "frame lower bound")
    frame = frame_packet[:FRAME_HEADER_BYTES]
    frame_fields = struct.unpack_from("<8sIIQQQ", frame, 0)
    frame_magic, frame_version, frame_interface, frame_count, logical_bits, payload_bytes = frame_fields
    require(frame_magic == FRAME_MAGIC and frame_version == 1 and
            frame_interface == interface_id and frame_count == count and
            frame_bytes == FRAME_HEADER_BYTES + payload_bytes,
            "frame header identity")
    payload = frame_packet[FRAME_HEADER_BYTES:]
    require(len(payload) == payload_bytes and 0 <= logical_bits <= 8 * payload_bytes and
            frame[72:104] == sha256(payload) and
            struct.unpack_from("<I", frame, 104)[0] ==
            (zlib.crc32(frame[:104]) & 0xFFFFFFFF), "frame hashes")
    _zero_tail(frame, 108, FRAME_HEADER_BYTES, "frame")
    return {
        "interface": interface, "count": int(count),
        "model": model, "centroid": head,
        "model_packet": model_packet, "centroid_packet": centroid_packet,
        "payload": payload, "logical_bits": int(logical_bits),
        "label_sha256": frame[40:72], "packet_sha256": sha256_hex(packet),
        "total_bytes": len(packet),
    }


def decode_and_reencode(packet: bytes, adapter: Any, core: Any,
                        stream_ordinal: int = 0) -> dict[str, Any]:
    parsed = parse_packet(packet, core)
    require(adapter.interface == parsed["interface"], "adapter/packet interface")
    decoded = core.decode_payload(
        parsed["count"], parsed["payload"], parsed["logical_bits"], adapter,
        parsed["model"], parsed["centroid"], stream_ordinal=stream_ordinal)
    require(_label_hash(decoded["labels"]) == parsed["label_sha256"],
            "decoded label digest")
    canonical = build_packet(
        parsed["interface"], parsed["model"], parsed["centroid"],
        decoded["labels"], parsed["payload"], parsed["logical_bits"])
    require(canonical == packet, "literal complete packet reencode")
    return {
        **decoded, "packet_sha256": sha256_hex(packet),
        "packet_bytes": len(packet), "literal_packet_reencode_matches": True,
        "model_bytes": len(parsed["model_packet"]),
        "centroid_bytes": len(parsed["centroid_packet"]),
        "header_bytes": PACKET_HEADER_BYTES,
        "stream_header_bytes": FRAME_HEADER_BYTES,
    }


def fixed_packet_bytes(interface: str, model: Any, head: Any) -> int:
    return (PACKET_HEADER_BYTES + len(serialize_model(interface, model)) +
            len(serialize_centroid(interface, head)) + FRAME_HEADER_BYTES)


def owner_page_ledger(global_logical_bytes: int,
                      frames: Iterable[tuple[Sequence[int], int]],
                      expert_count: int) -> dict[str, Any]:
    require(type(global_logical_bytes) is int and global_logical_bytes > 0 and
            type(expert_count) is int and 1 <= expert_count <= 256,
            "owner ledger global inputs")
    rows = []
    offset = math.ceil(global_logical_bytes / PAGE_BYTES) * PAGE_BYTES
    for owners, frame_bytes in frames:
        owner_tuple = tuple(sorted(set(int(value) for value in owners)))
        require(owner_tuple and all(0 <= value < expert_count for value in owner_tuple)
                and frame_bytes > 0, "owner frame")
        region_bytes = math.ceil(frame_bytes / PAGE_BYTES) * PAGE_BYTES
        rows.append({"owners": owner_tuple, "logical_bytes": int(frame_bytes),
                     "offset": offset, "region_bytes": region_bytes})
        offset += region_bytes
    global_pages = math.ceil(global_logical_bytes / PAGE_BYTES) * PAGE_BYTES
    experts = []
    worst = 0.0
    for expert in range(expert_count):
        selected = [row for row in rows if expert in row["owners"]]
        fair = global_logical_bytes / expert_count + math.fsum(
            row["logical_bytes"] / len(row["owners"]) for row in selected)
        touched = global_pages + sum(row["region_bytes"] for row in selected)
        ratio = touched / fair if fair else math.inf
        worst = max(worst, ratio)
        experts.append({
            "expert": expert, "selected_regions": len(selected),
            "owner_local_fair_logical_bytes": fair,
            "unique_touched_page_bytes": touched,
            "cold_read_amplification": ratio,
            "external_storage_passes": 1,
            "compressed_page_refetch_bytes": 0,
        })
    return {
        "schema": "epsilon-tcq-owner-page-ledger-v0",
        "global_logical_bytes": global_logical_bytes,
        "global_page_bytes": global_pages,
        "regions": rows, "experts": experts,
        "worst_cold_read_amplification": worst,
        "strictly_below_2x": worst < 2.0,
        "external_storage_host_scratch_hbm_are_disjoint": True,
        "host_parse_and_beam_scratch_measured": False,
        "accelerator_hbm_kernel_traffic_measured": False,
        "inference_claim_authority": False,
    }
