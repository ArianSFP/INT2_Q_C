#!/usr/bin/env python3
"""Strict, standalone unpacker for the POLARIS v2 shared-bit reservoir.

The parser deliberately does not import any encoder or packer implementation.
It validates the complete physical container before creating output, then emits
one exact variable-length record per directory entry so an independent decoder
can audit every reconstructed block.

Container layout (all header/directory integers are big-endian)::

    >8sHHIQQ32s32s header
    block_count * (>I logical_bits, raw little-endian binary16 scale)
    fixed 563464 * block_count-bit physical payload reservoir

The concatenated logical arithmetic payload is an MSB-first prefix of that
fixed reservoir.  Every bit after ``payload_logical_bits`` must be zero, and
the payload hash covers the full fixed-capacity byte region.

Each extracted audit record is::

    <I logical_bits, raw little-endian binary16 scale, block payload MSB first

The six-byte header is called ``variable-u32-fp16``.  The scale bytes are
copied verbatim from the directory, with no floating-point round trip.  The
last payload byte in an audit record is independently zero padded.  This
means the extracted files are byte-addressable even when adjacent reservoir
blocks were not byte aligned.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


MAGIC = b"PLRSV2\0\0"
VERSION = 2
FLAGS = 0
HEADER_STRUCT = struct.Struct(">8sHHIQQ32s32s")
HEADER_BYTES = HEADER_STRUCT.size
DIRECTORY_ENTRY_BYTES = 6
DIRECTORY_ENTRY_BITS = DIRECTORY_ENTRY_BYTES * 8
PAYLOAD_CAPACITY_BITS_PER_BLOCK = 563_464
PAYLOAD_CAPACITY_BYTES_PER_BLOCK = PAYLOAD_CAPACITY_BITS_PER_BLOCK // 8
VARIABLE_HEADER_U32_STRUCT = struct.Struct("<I")
VARIABLE_HEADER_BYTES = 6
HASH_CHUNK_BYTES = 8 * 1024 * 1024
EXTRACT_CHUNK_BITS = 8 * 1024 * 1024 * 8


class ReservoirFormatError(ValueError):
    """Raised when a reservoir violates the frozen v2 format."""


@dataclass(frozen=True)
class DirectoryEntry:
    index: int
    logical_bits: int
    scale_bytes_le: bytes
    scale_fp16: float
    physical_sha256: str


@dataclass(frozen=True)
class ValidatedReservoir:
    path: Path
    file_bytes: int
    block_count: int
    directory_bits: int
    directory_bytes: int
    payload_logical_bits: int
    payload_capacity_bits: int
    payload_physical_bytes: int
    payload_unused_zero_bits: int
    payload_offset: int
    header_sha256: str
    directory_sha256: str
    payload_sha256: str
    reservoir_sha256: str
    entries: tuple[DirectoryEntry, ...]


def _sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_exact(stream: BinaryIO, count: int, context: str) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise ReservoirFormatError(
            f"truncated {context}: expected {count} bytes, got {len(value)}"
        )
    return value


def _checked_half(scale_bytes_le: bytes, index: int) -> float:
    """Decode a raw LE binary16 and enforce the codec's positive-scale rule."""

    try:
        value = float(struct.unpack("<e", scale_bytes_le)[0])
    except (OverflowError, struct.error) as exc:
        raise ReservoirFormatError(
            f"directory entry {index} has an invalid binary16 scale"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ReservoirFormatError(
            f"directory entry {index} has nonpositive or nonfinite scale {value}"
        )
    return value


def validate_reservoir(path: Path) -> ValidatedReservoir:
    """Validate all framing, hashes, lengths, and global zero padding."""

    path = path.resolve(strict=True)
    before = path.stat()
    if not path.is_file():
        raise ReservoirFormatError(f"input is not a regular file: {path}")
    if before.st_size < HEADER_BYTES:
        raise ReservoirFormatError(
            f"container is shorter than its {HEADER_BYTES}-byte header"
        )

    with path.open("rb") as stream:
        header = _read_exact(stream, HEADER_BYTES, "header")
        (
            magic,
            version,
            flags,
            block_count,
            directory_bits,
            payload_logical_bits,
            expected_directory_hash,
            expected_payload_hash,
        ) = HEADER_STRUCT.unpack(header)

        if magic != MAGIC:
            raise ReservoirFormatError(
                f"bad magic {magic!r}; expected {MAGIC!r}"
            )
        if version != VERSION:
            raise ReservoirFormatError(
                f"unsupported version {version}; expected {VERSION}"
            )
        if flags != FLAGS:
            raise ReservoirFormatError(
                f"unsupported flags 0x{flags:04x}; expected 0"
            )
        if block_count == 0:
            raise ReservoirFormatError("block_count must be greater than zero")

        expected_directory_bits = DIRECTORY_ENTRY_BITS * block_count
        if directory_bits != expected_directory_bits:
            raise ReservoirFormatError(
                f"directory_bits={directory_bits}, but {block_count} entries "
                f"require exactly {expected_directory_bits} bits"
            )
        directory_bytes = directory_bits // 8
        payload_capacity_bits = PAYLOAD_CAPACITY_BITS_PER_BLOCK * block_count
        payload_physical_bytes = PAYLOAD_CAPACITY_BYTES_PER_BLOCK * block_count
        expected_file_bytes = HEADER_BYTES + directory_bytes + payload_physical_bytes
        # Reject an impossible block count/size before allocating or reading its
        # claimed directory.  This also makes truncation and trailing data the
        # same unambiguous framing error.
        if before.st_size != expected_file_bytes:
            raise ReservoirFormatError(
                f"physical size is {before.st_size} bytes, but block_count "
                f"requires exactly {expected_file_bytes} bytes"
            )
        directory = _read_exact(stream, directory_bytes, "directory")
        actual_directory_hash = hashlib.sha256(directory).digest()
        if not hmac.compare_digest(actual_directory_hash, expected_directory_hash):
            raise ReservoirFormatError(
                "directory SHA-256 mismatch: header="
                f"{expected_directory_hash.hex()} actual="
                f"{actual_directory_hash.hex()}"
            )

        entries: list[DirectoryEntry] = []
        logical_sum = 0
        for index in range(block_count):
            offset = index * DIRECTORY_ENTRY_BYTES
            raw_entry = directory[offset : offset + DIRECTORY_ENTRY_BYTES]
            logical_bits = struct.unpack(">I", raw_entry[:4])[0]
            if logical_bits == 0:
                raise ReservoirFormatError(
                    f"directory entry {index} has zero logical length"
                )
            scale_bytes_le = raw_entry[4:6]
            entries.append(
                DirectoryEntry(
                    index=index,
                    logical_bits=logical_bits,
                    scale_bytes_le=scale_bytes_le,
                    scale_fp16=_checked_half(scale_bytes_le, index),
                    physical_sha256=_sha256_hex(raw_entry),
                )
            )
            logical_sum += logical_bits

        if logical_sum != payload_logical_bits:
            raise ReservoirFormatError(
                f"directory lengths sum to {logical_sum} bits, but the header "
                f"declares {payload_logical_bits} payload bits"
            )

        if payload_logical_bits > payload_capacity_bits:
            raise ReservoirFormatError(
                f"logical payload uses {payload_logical_bits} bits, exceeding "
                f"the fixed {payload_capacity_bits}-bit reservoir capacity"
            )
        payload_hasher = hashlib.sha256()
        reservoir_hasher = hashlib.sha256()
        reservoir_hasher.update(header)
        reservoir_hasher.update(directory)
        remaining = payload_physical_bytes
        while remaining:
            chunk = _read_exact(
                stream,
                min(remaining, HASH_CHUNK_BYTES),
                "physical payload",
            )
            payload_hasher.update(chunk)
            reservoir_hasher.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ReservoirFormatError("container has trailing bytes")

        # The logical stream occupies a prefix of a fixed physical reservoir.
        # Validate every bit in the unused suffix, including the low bits of a
        # partially used boundary byte.  Hashing above intentionally covered
        # the suffix as physical serialized data.
        logical_full_bytes, logical_remainder_bits = divmod(
            payload_logical_bits, 8
        )
        stream.seek(HEADER_BYTES + directory_bytes + logical_full_bytes)
        suffix_remaining = payload_physical_bytes - logical_full_bytes
        if logical_remainder_bits:
            boundary = _read_exact(stream, 1, "logical payload boundary")
            suffix_remaining -= 1
            boundary_zero_mask = (1 << (8 - logical_remainder_bits)) - 1
            if boundary[0] & boundary_zero_mask:
                raise ReservoirFormatError(
                    "nonzero bit found after payload_logical_bits in the "
                    "boundary byte"
                )
        while suffix_remaining:
            zero_chunk = _read_exact(
                stream,
                min(suffix_remaining, HASH_CHUNK_BYTES),
                "unused reservoir suffix",
            )
            if zero_chunk.count(0) != len(zero_chunk):
                raise ReservoirFormatError(
                    "nonzero byte found in the unused fixed-reservoir suffix"
                )
            suffix_remaining -= len(zero_chunk)

    actual_payload_hash = payload_hasher.digest()
    if not hmac.compare_digest(actual_payload_hash, expected_payload_hash):
        raise ReservoirFormatError(
            "payload SHA-256 mismatch: header="
            f"{expected_payload_hash.hex()} actual={actual_payload_hash.hex()}"
        )

    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or getattr(before, "st_ino", None) != getattr(after, "st_ino", None)
    ):
        raise ReservoirFormatError("input changed while it was being validated")

    return ValidatedReservoir(
        path=path,
        file_bytes=before.st_size,
        block_count=block_count,
        directory_bits=directory_bits,
        directory_bytes=directory_bytes,
        payload_logical_bits=payload_logical_bits,
        payload_capacity_bits=payload_capacity_bits,
        payload_physical_bytes=payload_physical_bytes,
        payload_unused_zero_bits=payload_capacity_bits - payload_logical_bits,
        payload_offset=HEADER_BYTES + directory_bytes,
        header_sha256=_sha256_hex(header),
        directory_sha256=actual_directory_hash.hex(),
        payload_sha256=actual_payload_hash.hex(),
        reservoir_sha256=reservoir_hasher.hexdigest(),
        entries=tuple(entries),
    )


def _iter_repacked_bits(
    stream: BinaryIO,
    payload_offset: int,
    logical_start: int,
    logical_bits: int,
) -> Iterator[bytes]:
    """Yield one block as byte-aligned MSB-first chunks with final zero pad."""

    cursor = logical_start
    remaining = logical_bits
    while remaining:
        take = min(remaining, EXTRACT_CHUNK_BITS)
        # Non-final chunks must end on an output-byte boundary so yielded chunks
        # can be concatenated without adding internal padding.
        if take < remaining:
            take -= take % 8
        intra_byte_offset = cursor % 8
        physical_byte_offset = cursor // 8
        physical_count = (intra_byte_offset + take + 7) // 8
        stream.seek(payload_offset + physical_byte_offset)
        physical = _read_exact(stream, physical_count, "block payload slice")

        if intra_byte_offset == 0 and take % 8 == 0:
            yield physical
        else:
            physical_value = int.from_bytes(physical, "big")
            trailing_bits = len(physical) * 8 - intra_byte_offset - take
            logical_value = physical_value >> trailing_bits
            logical_value &= (1 << take) - 1
            output_padding_bits = (-take) % 8
            logical_value <<= output_padding_bits
            yield logical_value.to_bytes((take + 7) // 8, "big")

        cursor += take
        remaining -= take


def _json_float(value: float) -> float | str:
    """Represent non-finite IEEE values without emitting nonstandard JSON."""

    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "+Infinity"
    if value == -math.inf:
        return "-Infinity"
    return value


def extract_variable_records(
    reservoir: ValidatedReservoir,
    output_dir: Path,
) -> list[dict[str, object]]:
    """Atomically create independently padded variable records for every block."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)
        )
    )

    block_audits: list[dict[str, object]] = []
    logical_start = 0
    name_width = max(6, len(str(reservoir.block_count - 1)))
    try:
        with reservoir.path.open("rb") as source:
            for entry in reservoir.entries:
                filename = (
                    f"block_{entry.index:0{name_width}d}.variable-u32-fp16.bin"
                )
                destination = staging / filename
                legacy_header = (
                    VARIABLE_HEADER_U32_STRUCT.pack(entry.logical_bits)
                    + entry.scale_bytes_le
                )
                if len(legacy_header) != VARIABLE_HEADER_BYTES:
                    raise AssertionError(len(legacy_header))
                record_hasher = hashlib.sha256()
                payload_hasher = hashlib.sha256()
                record_hasher.update(legacy_header)
                payload_bytes = 0
                with destination.open("xb") as target:
                    target.write(legacy_header)
                    for chunk in _iter_repacked_bits(
                        source,
                        reservoir.payload_offset,
                        logical_start,
                        entry.logical_bits,
                    ):
                        target.write(chunk)
                        record_hasher.update(chunk)
                        payload_hasher.update(chunk)
                        payload_bytes += len(chunk)

                expected_payload_bytes = (entry.logical_bits + 7) // 8
                if payload_bytes != expected_payload_bytes:
                    raise AssertionError(
                        f"block {entry.index} emitted {payload_bytes} payload "
                        f"bytes; expected {expected_payload_bytes}"
                    )
                record_bytes = VARIABLE_HEADER_BYTES + payload_bytes
                block_audits.append(
                    {
                        "index": entry.index,
                        "logical_start_bit": logical_start,
                        "logical_arithmetic_bits": entry.logical_bits,
                        "variable_payload_physical_bytes": payload_bytes,
                        "variable_payload_padding_bits": (-entry.logical_bits) % 8,
                        "scale_fp16_raw_le_hex": entry.scale_bytes_le.hex(),
                        "scale_fp16_value": _json_float(entry.scale_fp16),
                        "scale_fp32_expanded_le_hex": struct.pack(
                            "<f", entry.scale_fp16
                        ).hex(),
                        "scale_fp32_expanded_value": _json_float(
                            struct.unpack("<f", struct.pack("<f", entry.scale_fp16))[0]
                        ),
                        "directory_entry_sha256": entry.physical_sha256,
                        "variable_payload_physical_sha256": payload_hasher.hexdigest(),
                        "variable_record_bytes": record_bytes,
                        "variable_record_sha256": record_hasher.hexdigest(),
                        "relative_path": filename,
                    }
                )
                logical_start += entry.logical_bits

        if logical_start != reservoir.payload_logical_bits:
            raise AssertionError(
                f"extracted {logical_start} bits; expected "
                f"{reservoir.payload_logical_bits}"
            )
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return block_audits


def build_audit(
    reservoir: ValidatedReservoir,
    output_dir: Path,
    blocks: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "format": "POLARIS shared-bit reservoir v2",
        "validation": "passed",
        "magic_hex": MAGIC.hex(),
        "version": VERSION,
        "flags": FLAGS,
        "input_path": str(reservoir.path),
        "input_physical_bytes": reservoir.file_bytes,
        "header_physical_bytes": HEADER_BYTES,
        "header_sha256": reservoir.header_sha256,
        "block_count": reservoir.block_count,
        "directory_bits": reservoir.directory_bits,
        "directory_physical_bytes": reservoir.directory_bytes,
        "directory_sha256": reservoir.directory_sha256,
        "payload_logical_bits": reservoir.payload_logical_bits,
        "payload_capacity_bits": reservoir.payload_capacity_bits,
        "payload_capacity_bits_per_block": PAYLOAD_CAPACITY_BITS_PER_BLOCK,
        "payload_physical_bytes": reservoir.payload_physical_bytes,
        "payload_unused_zero_bits": reservoir.payload_unused_zero_bits,
        "payload_sha256": reservoir.payload_sha256,
        "reservoir_sha256": reservoir.reservoir_sha256,
        "output_directory": str(output_dir.resolve()),
        "extracted_record_layout": "variable-u32-fp16",
        "extracted_record_header_bytes": VARIABLE_HEADER_BYTES,
        "extracted_record_header": (
            "little-endian uint32 logical bits + raw little-endian FP16 scale"
        ),
        "payload_bit_order": "MSB-first",
        "blocks": blocks,
    }


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and independently unpack a POLARIS v2 reservoir"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audit",
        type=Path,
        help="JSON audit path (default: <output-dir>.audit.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = args.audit
    if audit_path is None:
        audit_path = args.output_dir.with_name(args.output_dir.name + ".audit.json")

    reservoir = validate_reservoir(args.input)
    blocks = extract_variable_records(reservoir, args.output_dir)
    audit = build_audit(reservoir, args.output_dir, blocks)
    _write_json_atomic(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
