#!/usr/bin/env python3
"""Literal aligned packet for the local RM-frozen-set STRATA candidate."""

from __future__ import annotations

import struct
import zlib
from typing import Any, Sequence

import numpy as np

from rm6_core import (ALIGNMENT_BYTES, LOCAL_LOG2, LOCAL_N, ORDER_BANK,
                      PHYSICAL_CAP_BYTES, PLANES, TARGET_MAX_BPW,
                      TARGET_MIN_BPW, align, bank_dimensions, dimension_ledger,
                      fp16_from_bits, require)
from strata_rm_sc import decode_six_payload, replay_six_prescribed


MAGIC = b"SRM6"
VERSION = 0
HEADER = struct.Struct("<4sBBBBIHHHBB6BIQH")
HEADER_BYTES = HEADER.size
CRC = struct.Struct("<I")
COSET_TO_ID = {"zero": 0, "current_random": 1}
ID_TO_COSET = {value: key for key, value in COSET_TO_ID.items()}


def packet_ledger(bank_id: int, logical_bits: int | None = None) -> dict[str, Any]:
    dimension = dimension_ledger(bank_id, header_bytes=HEADER_BYTES,
                                 crc_bytes=CRC.size)
    if logical_bits is None:
        return {**dimension, "actual_arithmetic_known": False,
                "emitted_arithmetic_bits": None, "actual_packet_bytes": None,
                "emitted_packet_bytes": None, "actual_physical_bpw": None,
                "actual_target_rate_eligible": None,
                "promotion_status": "HOLD_NO_LITERAL_ARITHMETIC_PACKET"}
    require(type(logical_bits) is int and logical_bits > 0, "logical bits")
    raw = HEADER_BYTES + (logical_bits + 7) // 8 + CRC.size
    packet_bytes = align(raw)
    physical_bpw = packet_bytes * 8.0 / LOCAL_N
    eligible = TARGET_MIN_BPW <= physical_bpw <= TARGET_MAX_BPW
    return {**dimension, "actual_arithmetic_known": True,
            "actual_logical_bits": logical_bits,
            "emitted_arithmetic_bits": logical_bits,
            "actual_payload_bytes": (logical_bits + 7) // 8,
            "actual_raw_bytes": raw, "actual_packet_bytes": packet_bytes,
            "emitted_packet_bytes": packet_bytes,
            "actual_physical_bpw": physical_bpw,
            "actual_passes_2_5_bpw": packet_bytes <= PHYSICAL_CAP_BYTES,
            "target_min_bpw": TARGET_MIN_BPW,
            "target_max_bpw": TARGET_MAX_BPW,
            "actual_target_rate_eligible": eligible,
            "requires_literal_padding_or_refinement_for_target":
                physical_bpw < TARGET_MIN_BPW,
            "promotion_status": ("TARGET_RATE_ELIGIBLE_LITERAL_PACKET" if eligible else
                                 "MECHANISM_FIXTURE_BELOW_2_15_BPW")}


def _header(bank_id: int, logical_bits: int, scale_fp16_bits: int,
            profile_q: int, coset_mode: str, sc_seed: int, rht_seed: int) -> bytes:
    require(bank_id in ORDER_BANK and coset_mode in COSET_TO_ID, "packet selectors")
    require(0 < logical_bits <= 0xFFFF and 0 <= profile_q <= 255 and
            0 <= sc_seed <= 0xFFFFFFFF and 0 <= rht_seed <= 0xFFFFFFFFFFFFFFFF,
            "packet scalar ranges")
    fp16_from_bits(scale_fp16_bits)
    orders = ORDER_BANK[bank_id]
    information_bits = sum(bank_dimensions(bank_id))
    return HEADER.pack(MAGIC, VERSION, LOCAL_LOG2, PLANES, bank_id, LOCAL_N,
                       logical_bits, information_bits, scale_fp16_bits,
                       profile_q, COSET_TO_ID[coset_mode], *orders,
                       sc_seed, rht_seed, 0)


def _build_packet(header: bytes, payload: bytes, logical_bits: int) -> bytes:
    require(len(header) == HEADER_BYTES and len(payload) == (logical_bits + 7) // 8,
            "packet build geometry")
    if logical_bits & 7:
        unused = 8 - (logical_bits & 7)
        require(payload[-1] & ((1 << unused) - 1) == 0, "arithmetic terminal padding")
    body = header + payload
    raw = body + CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)
    packet = raw + bytes(align(len(raw)) - len(raw))
    require(len(packet) <= PHYSICAL_CAP_BYTES, "actual arithmetic packet exceeds 2.5 bpw")
    return packet


def encode_packet(decisions: Sequence[Sequence[int]], *, bank_id: int,
                  scale_fp16_bits: int, profile_q: int, coset_mode: str,
                  sc_seed: int, rht_seed: int) -> tuple[bytes, dict[str, Any]]:
    replay = replay_six_prescribed(bank_id, profile_q, sc_seed, coset_mode, decisions)
    header = _header(bank_id, replay["logical_bits"], scale_fp16_bits, profile_q,
                     coset_mode, sc_seed, rht_seed)
    packet = _build_packet(header, replay["payload"], replay["logical_bits"])
    return packet, {"indices": replay["indices"], "selected": replay["selected"],
                    "frequencies": replay["frequencies"],
                    "ledger": packet_ledger(bank_id, replay["logical_bits"])}


def decode_packet(raw: bytes) -> dict[str, Any]:
    require(type(raw) is bytes and len(raw) >= HEADER_BYTES + CRC.size and
            len(raw) % ALIGNMENT_BYTES == 0 and len(raw) <= PHYSICAL_CAP_BYTES,
            "literal packet size")
    fields = HEADER.unpack(raw[:HEADER_BYTES])
    (magic, version, log2_n, planes, bank_id, block_values, logical_bits,
     information_bits, scale_bits, profile_q, coset_id, *tail) = fields
    orders, sc_seed, rht_seed, reserved = tuple(tail[:6]), tail[6], tail[7], tail[8]
    require(magic == MAGIC and version == VERSION and log2_n == LOCAL_LOG2 and
            planes == PLANES and block_values == LOCAL_N and bank_id in ORDER_BANK,
            "packet fixed header")
    require(tuple(orders) == ORDER_BANK[bank_id] and
            information_bits == sum(bank_dimensions(bank_id)) and
            coset_id in ID_TO_COSET and reserved == 0, "packet bank header")
    fp16_from_bits(scale_bits)
    payload_bytes = (logical_bits + 7) // 8
    crc_offset = HEADER_BYTES + payload_bytes
    expected_bytes = align(crc_offset + CRC.size)
    require(len(raw) == expected_bytes and expected_bytes <= PHYSICAL_CAP_BYTES,
            "packet derived length")
    payload = raw[HEADER_BYTES:crc_offset]
    if logical_bits & 7:
        require(payload[-1] & ((1 << (8 - (logical_bits & 7))) - 1) == 0,
                "packet terminal bits")
    stored_crc, = CRC.unpack(raw[crc_offset:crc_offset + CRC.size])
    require(stored_crc == (zlib.crc32(raw[:crc_offset]) & 0xFFFFFFFF), "packet CRC")
    require(not any(raw[crc_offset + CRC.size:]), "packet alignment padding")
    coset_mode = ID_TO_COSET[coset_id]
    replay = decode_six_payload(bank_id, profile_q, sc_seed, coset_mode,
                                payload, logical_bits)
    dimensions = bank_dimensions(bank_id)
    decisions, cursor = [], 0
    for dimension in dimensions:
        decisions.append(replay["selected"][cursor:cursor + dimension].copy())
        cursor += dimension
    require(cursor == information_bits, "decoded information coverage")
    canonical_header = _header(bank_id, logical_bits, scale_bits, profile_q,
                               coset_mode, sc_seed, rht_seed)
    canonical_packet = _build_packet(canonical_header, replay["payload"], logical_bits)
    require(canonical_packet == raw, "canonical packet re-encode")
    return {"bank_id": bank_id, "orders": list(orders),
            "dimensions": list(dimensions), "information_bits": information_bits,
            "logical_bits": logical_bits, "scale_fp16_bits": scale_bits,
            "profile_q": profile_q, "coset_mode": coset_mode,
            "sc_seed": sc_seed, "rht_seed": rht_seed,
            "decisions": decisions, "indices": replay["indices"],
            "frequencies": replay["frequencies"],
            "ledger": packet_ledger(bank_id, logical_bits),
            "canonical_reencode_match": True}
