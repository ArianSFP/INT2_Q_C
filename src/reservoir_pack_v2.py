#!/usr/bin/env python3
"""Pack variable polar arithmetic messages into a global overflow reservoir.

POLARIS-SC-v2 removes the brittle per-block maximum while preserving the v1
whole-checkpoint bit allocation.  Each block contributes a six-byte directory
entry (u32 logical bit length and FP16 scale); only its logical arithmetic bits
are appended to one bit-contiguous payload.  Local rate excursions are thus
paid by savings in other blocks, with a hard global payload budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np


MAGIC = b"PLRSV2\0\0"
VERSION = 2
FLAGS = 0
HEADER = struct.Struct(">8sHHIQQ32s32s")
HEADER_BYTES = HEADER.size
DIRECTORY_BITS_PER_BLOCK = 48
DIRECTORY_BYTES_PER_BLOCK = 6
PAYLOAD_BUDGET_BITS_PER_BLOCK = 563_464
SCALE_BITS_PER_BLOCK = 16
LENGTH_BITS_PER_BLOCK = 32
TOTAL_RANK2_BUDGET_BITS_PER_BLOCK = 563_512


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_legacy(path: Path) -> tuple[int, float, bytes, dict[str, object]]:
    container = path.read_bytes()
    if len(container) < 8:
        raise ValueError(f"truncated legacy container: {path}")
    logical_bits, scale_fp32 = struct.unpack("<If", container[:8])
    payload = container[8:]
    expected_bytes = (logical_bits + 7) // 8
    if len(payload) != expected_bytes:
        raise ValueError(
            f"{path}: logical length requires {expected_bytes} bytes, found {len(payload)}"
        )
    if logical_bits <= 0:
        raise ValueError(f"{path}: empty arithmetic message")
    if not math.isfinite(scale_fp32) or scale_fp32 <= 0.0:
        raise ValueError(f"{path}: invalid scale {scale_fp32}")
    if logical_bits % 8:
        unused = 8 - logical_bits % 8
        if payload[-1] & ((1 << unused) - 1):
            raise ValueError(f"{path}: nonzero tail padding")
    scale_fp16 = np.asarray([scale_fp32], dtype="<f2")
    restored_scale = float(scale_fp16[0])
    if not math.isfinite(restored_scale) or restored_scale <= 0.0:
        raise ValueError(f"{path}: scale is invalid after FP16 conversion")
    return int(logical_bits), restored_scale, payload, {
        "input": str(path),
        "legacy_container_bytes": len(container),
        "legacy_container_sha256": sha256_bytes(container),
        "logical_arithmetic_bits": int(logical_bits),
        "stored_payload_bytes": len(payload),
        "stored_payload_sha256": sha256_bytes(payload),
        "discarded_per_block_tail_bits": len(payload) * 8 - int(logical_bits),
        "scale_fp32_input": float(scale_fp32),
        "scale_fp16_serialized": restored_scale,
    }


class StreamingBitSink:
    """MSB-first bit concatenation with at most seven pending bits."""

    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.pending = np.empty(0, dtype=np.uint8)
        self.logical_bits = 0
        self.physical_bytes = 0
        self.digest = hashlib.sha256()

    def append(self, payload: bytes, logical_bits: int) -> None:
        source = np.frombuffer(payload, dtype=np.uint8)
        bits = np.unpackbits(source, bitorder="big")[:logical_bits]
        if self.pending.size:
            bits = np.concatenate((self.pending, bits))
        complete = (bits.size // 8) * 8
        if complete:
            encoded = np.packbits(bits[:complete], bitorder="big").tobytes()
            self.handle.write(encoded)
            self.digest.update(encoded)
            self.physical_bytes += len(encoded)
        self.pending = bits[complete:].copy()
        self.logical_bits += logical_bits

    def finish(self, capacity_bits: int) -> tuple[int, int, str]:
        if capacity_bits % 8:
            raise ValueError("global reservoir capacity must be byte aligned")
        if self.logical_bits > capacity_bits:
            raise OverflowError(
                f"logical payload {self.logical_bits} exceeds capacity {capacity_bits}"
            )
        if self.pending.size:
            encoded = np.packbits(self.pending, bitorder="big").tobytes()
            self.handle.write(encoded)
            self.digest.update(encoded)
            self.physical_bytes += 1
        logical_physical_bytes = (self.logical_bits + 7) // 8
        if self.physical_bytes != logical_physical_bytes:
            raise AssertionError((self.physical_bytes, logical_physical_bytes))
        capacity_bytes = capacity_bits // 8
        remaining = capacity_bytes - self.physical_bytes
        zero_chunk = bytes(1 << 20)
        while remaining:
            take = min(remaining, len(zero_chunk))
            chunk = zero_chunk[:take]
            self.handle.write(chunk)
            self.digest.update(chunk)
            self.physical_bytes += take
            remaining -= take
        expected = capacity_bytes
        if self.physical_bytes != expected:
            raise AssertionError((self.physical_bytes, expected))
        return self.logical_bits, self.physical_bytes, self.digest.hexdigest()


def pack(inputs: list[Path], output: Path) -> dict[str, object]:
    if not inputs:
        raise ValueError("at least one block is required")
    if len(inputs) > 0xFFFFFFFF:
        raise OverflowError("block count does not fit u32")

    parsed = [parse_legacy(path) for path in inputs]
    directory = bytearray()
    for logical_bits, scale, _, _ in parsed:
        directory.extend(struct.pack(">I", logical_bits))
        directory.extend(np.asarray([scale], dtype="<f2").tobytes())
    expected_directory_bytes = len(inputs) * DIRECTORY_BYTES_PER_BLOCK
    if len(directory) != expected_directory_bytes:
        raise AssertionError(len(directory))
    directory_bytes = bytes(directory)
    directory_hash = hashlib.sha256(directory_bytes).digest()
    total_logical_bits = sum(row[0] for row in parsed)
    payload_budget = PAYLOAD_BUDGET_BITS_PER_BLOCK * len(inputs)
    if total_logical_bits > payload_budget:
        raise OverflowError(
            f"global payload overflow: {total_logical_bits} > {payload_budget} bits"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w+b") as handle:
        handle.write(bytes(HEADER_BYTES))
        handle.write(directory_bytes)
        sink = StreamingBitSink(handle)
        for logical_bits, _, payload, _ in parsed:
            sink.append(payload, logical_bits)
        payload_bits, payload_bytes, payload_hash_hex = sink.finish(payload_budget)
        if payload_bits != total_logical_bits:
            raise AssertionError((payload_bits, total_logical_bits))
        header = HEADER.pack(
            MAGIC,
            VERSION,
            FLAGS,
            len(inputs),
            len(directory_bytes) * 8,
            payload_bits,
            directory_hash,
            bytes.fromhex(payload_hash_hex),
        )
        handle.seek(0)
        handle.write(header)

    expected_output_bytes = HEADER_BYTES + len(directory_bytes) + payload_bytes
    if output.stat().st_size != expected_output_bytes:
        raise AssertionError((output.stat().st_size, expected_output_bytes))
    header_reservation_bits = 4096
    rank2_budget_bits = TOTAL_RANK2_BUDGET_BITS_PER_BLOCK * len(inputs)
    actual_rank2_bits_excluding_global_header = len(directory_bytes) * 8 + payload_bits
    return {
        "architecture": "POLARIS-SC-v2 checkpoint-global overflow reservoir",
        "format": {
            "magic_hex": MAGIC.hex(),
            "version": VERSION,
            "header_bytes": HEADER_BYTES,
            "block_count": len(inputs),
            "directory_bits_per_block": DIRECTORY_BITS_PER_BLOCK,
            "length_bits_per_block": LENGTH_BITS_PER_BLOCK,
            "scale_bits_per_block": SCALE_BITS_PER_BLOCK,
            "payload_concatenation": (
                "logical arithmetic bits, MSB-first, followed by zero fill to the "
                "fixed checkpoint-global reservoir capacity"
            ),
            "fixed_capacity_physical_reservoir": True,
        },
        "rate": {
            "total_logical_payload_bits": payload_bits,
            "mean_logical_payload_bits": payload_bits / len(inputs),
            "payload_budget_bits_per_block": PAYLOAD_BUDGET_BITS_PER_BLOCK,
            "global_payload_budget_bits": payload_budget,
            "global_payload_headroom_bits": payload_budget - payload_bits,
            "directory_plus_logical_payload_bits": actual_rank2_bits_excluding_global_header,
            "directory_plus_physical_payload_bits": (
                len(directory_bytes) * 8 + payload_bytes * 8
            ),
            "rank2_budget_bits_excluding_global_header": rank2_budget_bits,
            "rank2_budget_headroom_bits": rank2_budget_bits - actual_rank2_bits_excluding_global_header,
            "global_header_physical_bits": HEADER_BYTES * 8,
            "global_header_reserved_bits": header_reservation_bits,
            "global_header_fits_reservation": HEADER_BYTES * 8 <= header_reservation_bits,
            "global_zero_reserve_bits": payload_bytes * 8 - payload_bits,
            "physical_file_bits": output.stat().st_size * 8,
        },
        "hashes": {
            "directory_sha256": directory_hash.hex(),
            "fixed_capacity_payload_sha256": payload_hash_hex,
            "reservoir_sha256": sha256_file(output),
        },
        "blocks": [row[3] for row in parsed],
        "passed": (
            payload_bits <= payload_budget
            and HEADER_BYTES * 8 <= header_reservation_bits
            and actual_rank2_bits_excluding_global_header <= rank2_budget_bits
            and len(directory_bytes) * 8 + payload_bytes * 8 == rank2_budget_bits
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = pack(args.inputs, args.output)
    write_json(args.audit, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
