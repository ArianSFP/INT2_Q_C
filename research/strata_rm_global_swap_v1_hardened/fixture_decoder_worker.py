#!/usr/bin/env python3
"""Independent source-only decoder for the authority test packet.

Packet format (fixture only): ``magic | canonical JSON header | FP64 arrays | CRC``.
The packet itself carries the synthetic reconstructions.  This proves literal
decode/re-encode enforcement; it is explicitly not a compression result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import zlib
from pathlib import Path


MAGIC = b"SRMGF1\0\0"
PREFIX = struct.Struct("<8sII")
TRAILER = struct.Struct("<I")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def parse_strict(payload: bytes):
    def hook(pairs):
        row = {}
        for key, value in pairs:
            if key in row:
                raise ValueError("duplicate key")
            row[key] = value
        return row
    return json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ValueError(token)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise ValueError("requires -I -B")
    if "PYTHONPATH" in os.environ:
        raise ValueError("PYTHONPATH inherited")
    request = parse_strict(args.request.read_bytes())
    packet = args.packet.read_bytes()
    if request["packet_sha256"] != hashlib.sha256(packet).hexdigest() or \
            request["packet_bytes"] != len(packet):
        raise ValueError("request packet pin")
    if len(packet) < PREFIX.size + TRAILER.size:
        raise ValueError("short fixture packet")
    magic, header_bytes, payload_bytes = PREFIX.unpack_from(packet)
    if magic != MAGIC:
        raise ValueError("fixture magic")
    body_end = PREFIX.size + header_bytes + payload_bytes
    if body_end + TRAILER.size != len(packet):
        raise ValueError("fixture exact packet consumption")
    header_raw = packet[PREFIX.size:PREFIX.size + header_bytes]
    payload = packet[PREFIX.size + header_bytes:body_end]
    claimed_crc, = TRAILER.unpack_from(packet, body_end)
    if zlib.crc32(packet[:body_end]) & 0xFFFFFFFF != claimed_crc:
        raise ValueError("fixture CRC")
    header = parse_strict(header_raw)
    if canonical(header) != header_raw or header.get("schema") != \
            "strata-rm-v1-authority-fixture-packet":
        raise ValueError("fixture canonical header")
    lengths = header.get("reconstruction_f64_bytes")
    if not isinstance(lengths, list) or sum(lengths) != len(payload) or \
            len(lengths) != len(request["sources"]):
        raise ValueError("fixture reconstruction lengths")
    output = args.output_dir
    cursor = 0
    names = []
    for ordinal, length in enumerate(lengths):
        if not isinstance(length, int) or length <= 0 or length % 8:
            raise ValueError("fixture reconstruction length")
        name = f"reconstruction-{ordinal:04d}.f64"
        (output / name).write_bytes(payload[cursor:cursor + length])
        names.append(name)
        cursor += length
    # Independently reconstruct canonical bytes from parsed fields.
    canonical_packet = PREFIX.pack(MAGIC, len(header_raw), len(payload)) + \
        canonical(header) + payload
    canonical_packet += TRAILER.pack(zlib.crc32(canonical_packet) & 0xFFFFFFFF)
    (output / "canonical_packet.bin").write_bytes(canonical_packet)
    trace = {"schema": "strata-rm-v1-read-trace", "packet_bytes": len(packet),
             "operations": [{"object": "packet", "offset": 0,
                              "length": len(packet)}]}
    (output / "read_trace.json").write_bytes(canonical(trace) + b"\n")
    receipt = {
        "schema": "strata-rm-v1-independent-decoder-receipt",
        "case_id": request["case_id"], "packet_sha256": hashlib.sha256(packet).hexdigest(),
        "packet_bytes": len(packet),
        "canonical_packet_sha256": hashlib.sha256(canonical_packet).hexdigest(),
        "canonical_packet_bytes": len(canonical_packet),
        "independent_decode_complete": True, "canonical_reencode_complete": True,
        "causal_probabilities_regenerated": False, "packet_consumed_exactly": True,
        "encoder_decisions_read": False, "encoder_probabilities_read": False,
        "source_payloads_opened": False, "reconstruction_files": names,
        "read_trace_file": "read_trace.json",
        "status": "PASS_SYNTHETIC_AUTHORITY_FIXTURE_ONLY",
    }
    (output / "receipt.json").write_bytes(canonical(receipt) + b"\n")


if __name__ == "__main__":
    main()
