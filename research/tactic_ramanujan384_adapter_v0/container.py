#!/usr/bin/env python3
"""Canonical expert-local container for coarse plus Ramanujan-384 fine streams."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import struct
import zlib
from pathlib import Path
from typing import Any, Sequence


MAGIC = b"TRM384C0"
VERSION = 1
HEADER_BYTES = 512
PAGE_BYTES = 4096
BLOCK_VALUES = 4096
ROLE_ORDER = ("gate", "up", "down_transposed")
PREFIX = struct.Struct("<8sIIIIIIQQ32s32s32s")
CRC = struct.Struct("<I")


class ContainerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContainerError(message)


def sha256(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def parse_digest(value: str) -> bytes:
    require(isinstance(value, str) and len(value) == 64, "binding SHA256")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise ContainerError("binding SHA256") from exc
    require(len(result) == 32, "binding SHA256")
    return result


def _packet_module() -> Any:
    path = Path(__file__).resolve().parent / "packet.py"
    spec = importlib.util.spec_from_file_location("tactic_ramanujan384_container_packet", path)
    require(spec is not None and spec.loader is not None, "packet import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_blocks(intermediate: int, hidden: int) -> tuple[int, int, int]:
    require(type(intermediate) is int and intermediate > 0
            and type(hidden) is int and hidden > 0, "universal SwiGLU shape")
    per_role = intermediate * hidden
    blocks = (per_role + BLOCK_VALUES - 1) // BLOCK_VALUES
    return (blocks, blocks, blocks)


def expected_coarse_bytes(intermediate: int, hidden: int) -> int:
    weights = 3 * intermediate * hidden
    numerator = 307 * weights
    require(numerator % 1024 == 0, "shape has no integral 307/128-bpw coarse length")
    return numerator // 1024


def encode_header(
    *,
    intermediate: int,
    hidden: int,
    block_counts: Sequence[int],
    coarse_payload: bytes,
    fine_payload: bytes,
    source_binding_sha256: str,
) -> bytes:
    require(tuple(block_counts) == expected_blocks(intermediate, hidden), "canonical role block counts")
    require(type(coarse_payload) is bytes and coarse_payload, "coarse payload")
    require(len(coarse_payload) == expected_coarse_bytes(intermediate, hidden),
            "literal 307/128-bpw coarse payload length")
    require(type(fine_payload) is bytes and fine_payload, "fine payload")
    binding = parse_digest(source_binding_sha256)
    prefix = PREFIX.pack(
        MAGIC,
        VERSION,
        intermediate,
        hidden,
        int(block_counts[0]),
        int(block_counts[1]),
        int(block_counts[2]),
        len(coarse_payload),
        len(fine_payload),
        sha256(coarse_payload),
        sha256(fine_payload),
        binding,
    )
    require(len(prefix) <= HEADER_BYTES - CRC.size, "header prefix")
    body = prefix + bytes(HEADER_BYTES - CRC.size - len(prefix))
    return body + CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def decode_header(payload: bytes) -> dict[str, Any]:
    require(type(payload) is bytes and len(payload) == HEADER_BYTES, "literal header size")
    body = payload[:-CRC.size]
    observed_crc = CRC.unpack(payload[-CRC.size:])[0]
    require((zlib.crc32(body) & 0xFFFFFFFF) == observed_crc, "header CRC32")
    fields = PREFIX.unpack_from(body, 0)
    (magic, version, intermediate, hidden, gate_blocks, up_blocks, down_blocks,
     coarse_bytes, fine_bytes, coarse_sha, fine_sha, binding_sha) = fields
    require(magic == MAGIC and version == VERSION, "header identity")
    counts = (gate_blocks, up_blocks, down_blocks)
    require(counts == expected_blocks(intermediate, hidden), "canonical role block counts")
    require(coarse_bytes > 0 and fine_bytes == 48 * sum(counts), "header payload lengths")
    require(coarse_bytes == expected_coarse_bytes(intermediate, hidden),
            "literal 307/128-bpw coarse payload length")
    require(body[PREFIX.size:] == bytes(len(body) - PREFIX.size), "canonical header zero padding")
    result = {
        "intermediate": intermediate,
        "hidden": hidden,
        "role_order": ROLE_ORDER,
        "block_counts": counts,
        "coarse_bytes": coarse_bytes,
        "fine_bytes": fine_bytes,
        "coarse_sha256": coarse_sha.hex(),
        "fine_sha256": fine_sha.hex(),
        "source_binding_sha256": binding_sha.hex(),
    }
    return result


def _validate_fine(fine_payload: bytes, counts: Sequence[int]) -> None:
    packet = _packet_module()
    rows = packet.split_packets(fine_payload)
    require(len(rows) == sum(counts), "fine packet count")
    cursor = 0
    for role, count in zip(ROLE_ORDER, counts, strict=True):
        for payload in rows[cursor:cursor + count]:
            require(packet.decode_packet(payload)["role"] == role, "canonical fine role order")
        cursor += count


def encode_composite(
    *,
    intermediate: int,
    hidden: int,
    coarse_payload: bytes,
    role_fine_streams: Sequence[bytes],
    source_binding_sha256: str,
) -> bytes:
    counts = expected_blocks(intermediate, hidden)
    require(len(role_fine_streams) == 3, "three fine role streams")
    for stream, count in zip(role_fine_streams, counts, strict=True):
        require(type(stream) is bytes and len(stream) == 48 * count, "fine role stream size")
    fine = b"".join(role_fine_streams)
    _validate_fine(fine, counts)
    header = encode_header(
        intermediate=intermediate,
        hidden=hidden,
        block_counts=counts,
        coarse_payload=coarse_payload,
        fine_payload=fine,
        source_binding_sha256=source_binding_sha256,
    )
    unpadded = header + coarse_payload + fine
    physical_bytes = ((len(unpadded) + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES
    composite = unpadded + bytes(physical_bytes - len(unpadded))
    require(len(composite) % PAGE_BYTES == 0, "page-aligned composite")
    return composite


def decode_composite(payload: bytes) -> dict[str, Any]:
    require(type(payload) is bytes and len(payload) >= HEADER_BYTES
            and len(payload) % PAGE_BYTES == 0, "page-aligned composite size")
    header_payload = payload[:HEADER_BYTES]
    header = decode_header(header_payload)
    coarse_begin = HEADER_BYTES
    fine_begin = coarse_begin + header["coarse_bytes"]
    end = fine_begin + header["fine_bytes"]
    require(end <= len(payload), "composite payload bounds")
    expected_physical = ((end + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES
    require(len(payload) == expected_physical, "minimal page padding")
    require(payload[end:] == bytes(len(payload) - end), "canonical composite zero padding")
    coarse = payload[coarse_begin:fine_begin]
    fine = payload[fine_begin:end]
    require(sha256(coarse).hex() == header["coarse_sha256"], "coarse payload SHA256")
    require(sha256(fine).hex() == header["fine_sha256"], "fine payload SHA256")
    _validate_fine(fine, header["block_counts"])
    rebuilt_header = encode_header(
        intermediate=header["intermediate"],
        hidden=header["hidden"],
        block_counts=header["block_counts"],
        coarse_payload=coarse,
        fine_payload=fine,
        source_binding_sha256=header["source_binding_sha256"],
    )
    require(rebuilt_header == header_payload, "canonical header reencode")
    return {
        "header": header,
        "coarse_payload": coarse,
        "fine_payload": fine,
        "physical_bytes": len(payload),
        "page_padding_bytes": len(payload) - end,
        "external_storage_passes": 1,
        "external_storage_refetches": 0,
        "external_read_amplification": 1.0,
        "accelerator_hbm_measured": False,
    }
