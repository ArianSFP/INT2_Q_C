#!/usr/bin/env python3
"""Independent unpacker/decoder/auditor for the STRATA-v2 mixed KLT format.

This implementation is deliberately self-contained.  It imports no encoder,
no encoder decisions, no encoder-generated probabilities, and no reliability
tables.  It regenerates the periodic-channel capacities, unsigned-Q31 BEC
construction, frozen cosets, causal arithmetic probabilities, signed RHTs,
coarse-stratum permutation, and inverse cross-tensor KLT from the literal
physical container alone.

Normative byte layout (format version 1)::

    128-byte header
    144-byte literal Qwen route table
    5,184-byte raw MSB-first 3-bit group-label stream
    14 * 7-byte directory records, struct ``<BeI``
    7,603,175-byte global arithmetic reservoir

Each arithmetic stream occupies ``ceil(logical_bits/8)`` bytes, its unused
low bits in the final byte are zero, and the next stream starts at the next
byte.  The geometry is 13 blocks of 2**21 values followed by one block of
2**20.
The optional source scorer is inert unless explicit, hash-bound source paths
are supplied after unblinding.  This file never discovers or opens a blind
protocol directory on its own.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import re
import struct
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MAGIC = b"PLRKLT2\0"
FORMAT_VERSION = 1
HEADER_BYTES = 128
ROUTE_BYTES = 144
LABEL_BYTES = 5_184
DIRECTORY_RECORD = struct.Struct("<BeI")
DIRECTORY_RECORDS = 14
DIRECTORY_BYTES = DIRECTORY_RECORDS * DIRECTORY_RECORD.size
RESERVOIR_BYTES = 7_603_175
CONTAINER_BYTES = HEADER_BYTES + ROUTE_BYTES + LABEL_BYTES + DIRECTORY_BYTES + RESERVOIR_BYTES

HEADER_PREFIX = struct.Struct("<8sHHIIHHBBBBf")
HEADER_COEFFICIENTS = struct.Struct("<12f")
HEADER_ANGLES = struct.Struct("<6h")
HEADER_FLAGS = 0x7F

WEIGHTS = 28_311_552
GROUP_LENGTH = 2_048
GROUPS = 13_824
MATRICES = 18
GROUPS_PER_MATRIX = 768
BLOCKS = 14
LEADING_N21_BLOCKS = 13
LEADING_LOG2 = 21
TAIL_LOG2 = 20
ALPHABET_SIZE = 64
SIGMA_SOURCE = 1.0
Q15_DENOMINATOR = 32_768
ETA = 0.25

PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "cupy": "cupy-cuda12x",
    "scipy": "scipy",
    "cuda.pathfinder": "cuda-pathfinder",
}

RUNTIME_ARTIFACT_FREEZE_KEYS = {
    "python_interpreter": "python_interpreter",
    "runner": "one_shot_runner",
    "polar_encoder": "polar_encoder",
    "base_encoder": "base_cupy_encoder",
    "procedural_bec_builder": "procedural_q31_bec",
    "common": "common",
    "emitter": "emitter",
    "format": "format",
    "independent_auditor": "independent_auditor",
}

ROUTE_RECORD = struct.Struct(">HHBBH")
SEED_DOMAIN = b"POLARIS-STRATA-V2-KLT-MIXED-SEED-v1\0"

HEADER_OFFSET_COEFFICIENTS = 32
HEADER_OFFSET_ANGLES = 80
HEADER_OFFSET_CONTROL_SHA256 = 92
HEADER_OFFSET_CRC32 = 124
ROUTE_OFFSET = HEADER_BYTES
LABEL_OFFSET = ROUTE_OFFSET + ROUTE_BYTES
DIRECTORY_OFFSET = LABEL_OFFSET + LABEL_BYTES
RESERVOIR_OFFSET = DIRECTORY_OFFSET + DIRECTORY_BYTES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_distribution_tree_receipt(
    module_name: str, distribution_name: str
) -> dict[str, Any]:
    """Independently hash a complete installed wheel and verify its RECORD.

    This deliberately duplicates the frozen-runtime algorithm instead of
    importing the emitter's ``common`` module.  Extension modules and every
    other installed wheel file are included, not just the import origin.
    """
    module = importlib.import_module(module_name)
    origin = Path(str(module.__file__)).resolve(strict=True)
    distribution = importlib.metadata.distribution(distribution_name)
    files = sorted(
        distribution.files or (), key=lambda row: str(row).replace("\\", "/")
    )
    require(bool(files), f"distribution has no RECORD file list: {distribution_name}")
    record_rows = [row for row in files if row.name == "RECORD"]
    require(
        len(record_rows) == 1,
        f"distribution RECORD is not unique: {distribution_name}",
    )
    aggregate = hashlib.sha256()
    total_bytes = 0
    origin_is_recorded = False
    for row in files:
        relative = str(row).replace("\\", "/")
        path = Path(distribution.locate_file(row)).resolve(strict=True)
        require(path.is_file(), f"distribution entry is not a file: {path}")
        origin_is_recorded = origin_is_recorded or path == origin
        size = path.stat().st_size
        digest_hex = sha256_file(path)
        declared_hash = row.hash
        if declared_hash is not None:
            require(
                declared_hash.mode == "sha256",
                f"unsupported RECORD hash mode {declared_hash.mode}: {relative}",
            )
            encoded = (
                base64.urlsafe_b64encode(bytes.fromhex(digest_hex))
                .rstrip(b"=")
                .decode()
            )
            require(encoded == declared_hash.value, f"wheel RECORD hash mismatch: {relative}")
        if row.size is not None:
            require(int(row.size) == size, f"wheel RECORD size mismatch: {relative}")
        relative_bytes = relative.encode("utf-8")
        aggregate.update(len(relative_bytes).to_bytes(4, "big"))
        aggregate.update(relative_bytes)
        aggregate.update(size.to_bytes(8, "big"))
        aggregate.update(bytes.fromhex(digest_hex))
        total_bytes += size
    require(
        origin_is_recorded,
        f"import origin is not recorded by {distribution_name}: {origin}",
    )
    record_path = Path(distribution.locate_file(record_rows[0])).resolve(strict=True)
    return {
        "module": module_name,
        "distribution": distribution_name,
        "version": str(distribution.version),
        "import_origin": str(origin),
        "import_origin_sha256": sha256_file(origin),
        "record_path": str(record_path),
        "record_sha256": sha256_file(record_path),
        "recorded_file_count": len(files),
        "recorded_total_bytes": total_bytes,
        "distribution_tree_sha256": aggregate.hexdigest(),
    }


def independent_runtime_environment_receipt() -> dict[str, Any]:
    """Recreate the scoring process's Python/package/CUDA freeze receipt."""
    import cupy as cp

    python_invocation = Path(sys.executable).absolute()
    require(python_invocation.is_file(), "Python invocation path is not a file")
    python_resolved = python_invocation.resolve(strict=True)
    packages = {
        name: independent_distribution_tree_receipt(name, distribution)
        for name, distribution in PACKAGE_DISTRIBUTIONS.items()
    }
    device = cp.cuda.runtime.getDeviceProperties(0)
    external_paths = {"python_interpreter": str(python_invocation)}
    for name, row in packages.items():
        external_paths[f"{name}_import_origin"] = row["import_origin"]
        external_paths[f"{name}_wheel_record"] = row["record_path"]
    return {
        "python_interpreter": {
            "invocation_path": str(python_invocation),
            "resolved_path": str(python_resolved),
            "sha256": sha256_file(python_invocation),
            "version": sys.version.split()[0],
        },
        "packages": packages,
        "cuda": {
            "cupy_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "device_name": device["name"].decode(),
            "compute_capability": [int(device["major"]), int(device["minor"])],
        },
        "external_frozen_artifact_paths": external_paths,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def load_json_object(path: Path, description: str) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    require(isinstance(value, dict), f"{description} is not a JSON object")
    return value, sha256_bytes(payload)


def verify_internal_lock(value: dict[str, Any], description: str) -> str:
    expected = value.get("lock_sha256")
    require(
        isinstance(expected, str)
        and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
        f"{description} has no canonical lowercase lock_sha256",
    )
    clean = dict(value)
    clean.pop("lock_sha256", None)
    require(
        sha256_bytes(canonical_json_bytes(clean)) == expected,
        f"{description} internal seal mismatch",
    )
    return expected


def require_sha256(value: Any, description: str) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{description} is not a lowercase SHA256",
    )
    return value


def float32_bits(value: float) -> bytes:
    return struct.pack("<f", float(value))


@dataclass(frozen=True)
class Header:
    raw: bytes
    flags: int
    eta: float
    coefficients: tuple[tuple[float, float], ...]
    angle_codes: tuple[int, ...]
    control_sha256: bytes
    crc32: int


@dataclass(frozen=True)
class DirectoryRow:
    block_ordinal: int
    profile_q: int
    decoder_scale: float
    decoder_scale_fp16_hex: str
    logical_bits: int
    block_log2: int
    block_values: int


@dataclass(frozen=True)
class ParsedContainer:
    path: Path
    raw_sha256: str
    header: Header
    route: bytes
    labels_packed: bytes
    labels: np.ndarray
    directory_raw: bytes
    directory: tuple[DirectoryRow, ...]
    reservoir: bytes
    profile_bytes: bytes
    sc_seeds: tuple[int, ...]
    rht_seeds: tuple[int, ...]


def parse_header(raw: bytes, route: bytes, labels: bytes) -> Header:
    require(len(raw) == HEADER_BYTES, "header byte count mismatch")
    require(HEADER_PREFIX.size == HEADER_OFFSET_COEFFICIENTS, "internal prefix offset drift")
    (
        magic,
        version,
        header_bytes,
        flags,
        weights,
        group_length,
        groups,
        blocks,
        leading_n21,
        leading_log2,
        tail_log2,
        eta,
    ) = HEADER_PREFIX.unpack_from(raw, 0)
    expected = (
        MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        HEADER_FLAGS,
        WEIGHTS,
        GROUP_LENGTH,
        GROUPS,
        BLOCKS,
        LEADING_N21_BLOCKS,
        LEADING_LOG2,
        TAIL_LOG2,
    )
    actual = (
        magic,
        version,
        header_bytes,
        flags,
        weights,
        group_length,
        groups,
        blocks,
        leading_n21,
        leading_log2,
        tail_log2,
    )
    require(actual == expected, f"header constants mismatch: {actual!r}")
    require(float32_bits(eta) == float32_bits(ETA), "header eta is not exact FP32 0.25")

    flat_coefficients = HEADER_COEFFICIENTS.unpack_from(raw, HEADER_OFFSET_COEFFICIENTS)
    coefficients = tuple(
        (float(flat_coefficients[2 * index]), float(flat_coefficients[2 * index + 1]))
        for index in range(6)
    )
    angle_codes = tuple(int(value) for value in HEADER_ANGLES.unpack_from(raw, HEADER_OFFSET_ANGLES))
    for triplet, ((cosine, sine), code) in enumerate(zip(coefficients, angle_codes)):
        require(math.isfinite(cosine) and math.isfinite(sine), f"nonfinite KLT coefficient {triplet}")
        require(-16_384 <= code <= 16_384, f"Q15 angle outside canonical range at triplet {triplet}")
        theta = code * math.pi / Q15_DENOMINATOR
        expected_cosine = float(np.float32(math.cos(theta)))
        expected_sine = float(np.float32(math.sin(theta)))
        require(
            float32_bits(cosine) == float32_bits(expected_cosine)
            and float32_bits(sine) == float32_bits(expected_sine),
            f"stored FP32 coefficients do not derive from Q15 angle {triplet}",
        )
        require(abs(cosine * cosine + sine * sine - 1.0) < 2e-7, f"KLT norm drift {triplet}")

    control_sha256 = raw[HEADER_OFFSET_CONTROL_SHA256:HEADER_OFFSET_CRC32]
    require(
        control_sha256 == hashlib.sha256(route + labels).digest(),
        "header SHA256(route||labels) mismatch",
    )
    crc32 = struct.unpack_from("<I", raw, HEADER_OFFSET_CRC32)[0]
    require(crc32 == (zlib.crc32(raw[:HEADER_OFFSET_CRC32]) & 0xFFFFFFFF), "header CRC32 mismatch")
    return Header(
        raw=raw,
        flags=flags,
        eta=float(eta),
        coefficients=coefficients,
        angle_codes=angle_codes,
        control_sha256=control_sha256,
        crc32=crc32,
    )


def unpack_labels(raw: bytes) -> np.ndarray:
    require(len(raw) == LABEL_BYTES, "label byte count mismatch")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    require(bits.size == GROUPS * 3, "label bit count mismatch")
    triples = bits.reshape(GROUPS, 3)
    labels = (
        triples[:, 0] * np.uint8(4)
        + triples[:, 1] * np.uint8(2)
        + triples[:, 2]
    ).astype(np.uint8)
    require(np.all(labels <= 7), "label outside three-bit alphabet")
    return labels


def validate_route(raw: bytes) -> list[dict[str, int]]:
    require(len(raw) == ROUTE_BYTES == MATRICES * ROUTE_RECORD.size, "route byte count mismatch")
    rows = []
    for matrix in range(MATRICES):
        layer, expert, role, axis, groups = ROUTE_RECORD.unpack_from(raw, matrix * ROUTE_RECORD.size)
        triplet = matrix // 3
        role_in_triplet = matrix % 3
        require(0 <= layer < 64 and 0 <= expert < 128, f"route range error at matrix {matrix}")
        require(role == role_in_triplet, f"route role order mismatch at matrix {matrix}")
        require(axis == (1 if role == 2 else 0), f"route axis mismatch at matrix {matrix}")
        require(groups == GROUPS_PER_MATRIX, f"route group count mismatch at matrix {matrix}")
        if role_in_triplet:
            previous = rows[3 * triplet]
            require(
                layer == previous["layer"] and expert == previous["expert"],
                f"route triplet identity mismatch at matrix {matrix}",
            )
        rows.append(
            {
                "matrix_ordinal": matrix,
                "triplet": triplet,
                "layer": layer,
                "expert": expert,
                "role": role,
                "axis": axis,
                "groups": groups,
            }
        )
    require(len({(row["layer"], row["expert"]) for row in rows[::3]}) == 6, "duplicate route triplet")
    return rows


def derive_seed_pair(
    header: bytes,
    route: bytes,
    labels: bytes,
    profile_bytes: bytes,
    block_ordinal: int,
) -> tuple[int, int]:
    require(len(header) == HEADER_BYTES, "seed header length mismatch")
    require(len(route) == ROUTE_BYTES, "seed route length mismatch")
    require(len(labels) == LABEL_BYTES, "seed label length mismatch")
    require(len(profile_bytes) == BLOCKS, "seed profile vector length mismatch")
    require(0 <= block_ordinal < BLOCKS, "seed block ordinal outside uint8 domain")
    digest = hashlib.sha256(
        SEED_DOMAIN
        + header
        + route
        + labels
        + profile_bytes
        + bytes((block_ordinal,))
    ).digest()
    sc_seed = int.from_bytes(digest[:4], "big") or 1
    rht_seed = int.from_bytes(digest[4:12], "big")
    return sc_seed, rht_seed


def bit_range_has_one(raw: bytes, begin: int, end: int) -> bool:
    require(0 <= begin <= end <= len(raw) * 8, "invalid bit range")
    if begin == end:
        return False
    first_byte = begin // 8
    last_byte = (end - 1) // 8
    if first_byte == last_byte:
        left_offset = begin & 7
        end_offset = end & 7
        left_mask = (1 << (8 - left_offset)) - 1
        right_mask = 0xFF if end_offset == 0 else (0xFF << (8 - end_offset)) & 0xFF
        mask = left_mask & right_mask
        return bool(raw[first_byte] & mask)
    if begin & 7:
        mask = (1 << (8 - (begin & 7))) - 1
        if raw[first_byte] & mask:
            return True
        first_byte += 1
    if end & 7:
        mask = (0xFF << (8 - (end & 7))) & 0xFF
        if raw[last_byte] & mask:
            return True
        last_byte -= 1
    return any(raw[first_byte : last_byte + 1]) if first_byte <= last_byte else False


def packed_bit_slice(raw: bytes, begin: int, length: int) -> bytes:
    require(0 <= begin and 0 <= length and begin + length <= len(raw) * 8, "invalid bit slice")
    if length == 0:
        return b""
    source = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    return np.packbits(source[begin : begin + length], bitorder="big").tobytes()


def parse_container(path: Path) -> ParsedContainer:
    path = path.resolve(strict=True)
    require(path.stat().st_size == CONTAINER_BYTES, "container physical byte count mismatch")
    raw = path.read_bytes()
    require(len(raw) == CONTAINER_BYTES, "short container read")
    header_raw = raw[:HEADER_BYTES]
    route = raw[ROUTE_OFFSET:LABEL_OFFSET]
    labels_packed = raw[LABEL_OFFSET:DIRECTORY_OFFSET]
    directory_raw = raw[DIRECTORY_OFFSET:RESERVOIR_OFFSET]
    reservoir = raw[RESERVOIR_OFFSET:]
    require(len(reservoir) == RESERVOIR_BYTES, "reservoir byte count mismatch")
    validate_route(route)
    labels = unpack_labels(labels_packed)
    require(
        np.array_equal(np.bincount(labels, minlength=8), np.full(8, GROUPS // 8)),
        "literal labels are not exactly eight equipopulous strata",
    )
    header = parse_header(header_raw, route, labels_packed)

    directory = []
    profile_values = []
    for block in range(BLOCKS):
        offset = block * DIRECTORY_RECORD.size
        q, scale, logical_bits = DIRECTORY_RECORD.unpack_from(directory_raw, offset)
        scale_raw = directory_raw[offset + 1 : offset + 3]
        require(math.isfinite(scale) and scale > 0.0, f"invalid decoder scale at block {block}")
        block_log2 = LEADING_LOG2 if block < LEADING_N21_BLOCKS else TAIL_LOG2
        block_values = 1 << block_log2
        require(logical_bits > 0, f"zero arithmetic length at nonzero block {block}")
        # Six binary levels can never select more than 6*N bits; this is a
        # corruption bound, not a prediction of compressed size.
        require(logical_bits <= 6 * block_values + 64, f"implausible logical length at block {block}")
        profile_values.append(int(q))
        directory.append(
            DirectoryRow(
                block_ordinal=block,
                profile_q=int(q),
                decoder_scale=float(scale),
                decoder_scale_fp16_hex=scale_raw.hex(),
                logical_bits=int(logical_bits),
                block_log2=block_log2,
                block_values=block_values,
            )
        )
    profile_bytes = bytes(profile_values)
    used_bytes = 0
    for row in directory:
        payload_bytes = (row.logical_bits + 7) // 8
        require(
            used_bytes + payload_bytes <= len(reservoir),
            "global arithmetic reservoir overflow",
        )
        payload = reservoir[used_bytes : used_bytes + payload_bytes]
        require(
            not bit_range_has_one(payload, row.logical_bits, payload_bytes * 8),
            f"nonzero canonical final-byte padding at block {row.block_ordinal}",
        )
        used_bytes += payload_bytes
    require(not any(reservoir[used_bytes:]), "nonzero global-reservoir terminal byte fill")
    seeds = [derive_seed_pair(header.raw, route, labels_packed, profile_bytes, block) for block in range(BLOCKS)]
    return ParsedContainer(
        path=path,
        raw_sha256=sha256_bytes(raw),
        header=header,
        route=route,
        labels_packed=labels_packed,
        labels=labels,
        directory_raw=directory_raw,
        directory=tuple(directory),
        reservoir=reservoir,
        profile_bytes=profile_bytes,
        sc_seeds=tuple(row[0] for row in seeds),
        rht_seeds=tuple(row[1] for row in seeds),
    )


class ArithmeticBinaryDecoder:
    """32-bit integer arithmetic decoder over a non-byte-aligned bit window."""

    def __init__(self, raw: bytes, bit_offset: int, logical_bits: int):
        require(0 <= bit_offset <= len(raw) * 8, "arithmetic offset outside reservoir")
        require(0 <= logical_bits <= len(raw) * 8 - bit_offset, "arithmetic window overflow")
        self.raw = raw
        self.bit_offset = bit_offset
        self.logical_bits = logical_bits
        self.cursor = 0
        self.full = 1 << 32
        self.half = 1 << 31
        self.quarter = 1 << 30
        self.three_quarters = 3 << 30
        self.low = 0
        self.high = self.full - 1
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) & (self.full - 1)) | self._read()

    def _read(self) -> int:
        if self.cursor >= self.logical_bits:
            return 0
        position = self.bit_offset + self.cursor
        self.cursor += 1
        return (self.raw[position >> 3] >> (7 - (position & 7))) & 1

    def decode(self, freq1: int) -> int:
        f1 = min(65_535, max(1, int(freq1)))
        f0 = 65_536 - f1
        width = self.high - self.low + 1
        scaled = ((self.code - self.low + 1) * 65_536 - 1) // width
        split = self.low + width * f0 // 65_536 - 1
        if scaled < f0:
            value = 0
            self.high = split
        else:
            value = 1
            self.low = split + 1
        while True:
            if self.high < self.half:
                pass
            elif self.low >= self.half:
                self.low -= self.half
                self.high -= self.half
                self.code -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarters:
                self.low -= self.quarter
                self.high -= self.quarter
                self.code -= self.quarter
            else:
                break
            self.low = (self.low << 1) & (self.full - 1)
            self.high = ((self.high << 1) & (self.full - 1)) | 1
            self.code = ((self.code << 1) & (self.full - 1)) | self._read()
        return value


def arithmetic_encode_binary(bits: np.ndarray, freq1: np.ndarray) -> tuple[bytes, int]:
    """Canonical 32-bit/16-frequency binary arithmetic re-encoder."""
    require(bits.shape == freq1.shape, "arithmetic bit/frequency shape mismatch")
    full = 1 << 32
    half = 1 << 31
    quarter = 1 << 30
    three_quarters = 3 << 30
    low = 0
    high = full - 1
    pending = 0
    output: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        output.append(bit)
        if pending:
            output.extend([1 - bit] * pending)
            pending = 0

    for bit_u8, f1_u16 in zip(bits, freq1, strict=True):
        bit = int(bit_u8)
        f1 = int(f1_u16)
        require(bit in (0, 1) and 1 <= f1 <= 65_535, "invalid arithmetic symbol")
        f0 = 65_536 - f1
        width = high - low + 1
        split = low + width * f0 // 65_536 - 1
        if bit == 0:
            high = split
        else:
            low = split + 1
        while True:
            if high < half:
                emit(0)
            elif low >= half:
                emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
    pending += 1
    emit(0 if low < quarter else 1)
    logical_bits = len(output)
    packed = np.packbits(np.asarray(output, dtype=np.uint8), bitorder="big").tobytes()
    return packed, logical_bits


def bit_reverse_indices(n: int) -> np.ndarray:
    depth = int(math.log2(n))
    require(1 << depth == n, "polar length is not a power of two")
    source = np.arange(n, dtype=np.uint32)
    result = np.zeros(n, dtype=np.uint32)
    for _ in range(depth):
        result = (result << np.uint32(1)) | (source & np.uint32(1))
        source >>= np.uint32(1)
    return result.astype(np.int64)


def sc_layers(n: int) -> np.ndarray:
    depth = int(math.log2(n))
    result = np.ones(n + 1, dtype=np.int32)
    result[0] = depth
    for one_based in range(2, n + 1):
        layer = 1
        cursor = one_based
        while cursor % 2 == 1:
            layer += 1
            cursor = (cursor + 1) // 2
        result[one_based - 1] = layer
    return result


def polar_transform(bits: np.ndarray) -> np.ndarray:
    result = np.asarray(bits, dtype=np.uint8).copy()
    stride = 1
    while stride < result.size:
        rows = result.reshape(-1, 2 * stride)
        rows[:, :stride] ^= rows[:, stride:]
        stride *= 2
    return result


def periodic_binary_capacity(sigma: float, grid: int = 1 << 17, neighbors: int = 16) -> float:
    require(sigma > 0.0 and math.isfinite(sigma), "invalid periodic-channel sigma")
    y = (np.arange(grid, dtype=np.float64) + 0.5) * (2.0 / grid)
    ks = np.arange(-neighbors, neighbors + 1, dtype=np.float64)
    norm = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    p0 = np.exp(-0.5 * ((y[:, None] + 2.0 * ks[None, :]) / sigma) ** 2).sum(1) * norm
    p1 = np.exp(-0.5 * ((y[:, None] - 1.0 + 2.0 * ks[None, :]) / sigma) ** 2).sum(1) * norm
    mixture = 0.5 * (p0 + p1)
    posterior = np.divide(
        p1, p0 + p1, out=np.full_like(p1, 0.5), where=(p0 + p1) > 0
    )
    entropy = -(
        posterior * np.log2(np.maximum(posterior, 1e-300))
        + (1.0 - posterior) * np.log2(np.maximum(1.0 - posterior, 1e-300))
    )
    return float(1.0 - np.sum(mixture * entropy) * (2.0 / grid))


def profile_parameters(profile_q: int, eta: float) -> dict[str, Any]:
    require(0 <= profile_q <= 255, "profile outside uint8")
    rate = 1.75 + profile_q / 256.0
    distortion = math.exp2(-2.0 * rate)
    sigma_reconstruction = math.sqrt(SIGMA_SOURCE**2 - distortion)
    tilde_sigma = sigma_reconstruction * math.sqrt(distortion) / SIGMA_SOURCE
    capacities = [
        periodic_binary_capacity(tilde_sigma / eta / (1 << level0))
        for level0 in range(6)
    ]
    return {
        "profile_q": profile_q,
        "rate_bpw": rate,
        "test_channel_distortion": distortion,
        "sigma_reconstruction": sigma_reconstruction,
        "tilde_sigma": tilde_sigma,
        "capacities": capacities,
    }


def allocate_profiles_independent(block_energy: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Recompute the fixed mixed-N multiple-choice DP from source energies."""
    energy = np.asarray(block_energy, dtype=np.float64)
    require(
        energy.shape == (BLOCKS,) and np.all(np.isfinite(energy)) and np.all(energy > 0.0),
        "allocation requires fourteen positive finite block energies",
    )
    block_logs = [LEADING_LOG2] * LEADING_N21_BLOCKS + [TAIL_LOG2]
    finite_factor = {20: 1.0124498003545317, 21: 1.0107341453912242}
    profile_base = 1.75
    nominal_budget_bits = RESERVOIR_BYTES * 8 - 65_536
    base_bits = int(WEIGHTS * profile_base)
    unit_bits = (1 << TAIL_LOG2) // 256
    budget_units = (nominal_budget_bits - base_bits) // unit_bits
    dp = np.full(budget_units + 1, np.inf, dtype=np.float64)
    dp[0] = 0.0
    choices = []
    for block, (block_value, logn) in enumerate(zip(energy, block_logs, strict=True)):
        weight = 1 << (logn - TAIL_LOG2)
        updated = np.full_like(dp, np.inf)
        picked = np.full(dp.size, -1, dtype=np.int16)
        for q in range(256):
            cost = weight * q
            if cost > budget_units:
                break
            distortion = (
                finite_factor[logn]
                * float(block_value)
                * math.pow(2.0, -2.0 * (profile_base + q / 256.0))
            )
            candidate = dp[: dp.size - cost] + distortion
            target = updated[cost:]
            improve = candidate < target
            target[improve] = candidate[improve]
            picked[cost:][improve] = q
        require(np.any(np.isfinite(updated)), f"independent allocation lost feasibility {block}")
        dp = updated
        choices.append(picked)
    terminal = int(np.argmin(dp))
    profiles = np.empty(BLOCKS, dtype=np.uint8)
    cursor = terminal
    for block in range(BLOCKS - 1, -1, -1):
        chosen = int(choices[block][cursor])
        require(chosen >= 0, "independent allocation backtrack failed")
        profiles[block] = chosen
        cursor -= (1 << (block_logs[block] - TAIL_LOG2)) * chosen
    require(cursor == 0, "independent allocation did not return to origin")
    nominal_bits = base_bits + terminal * unit_bits
    return profiles, {
        "profile_ids": profiles.astype(int).tolist(),
        "base_bits": base_bits,
        "unit_bits": unit_bits,
        "budget_units": budget_units,
        "terminal_units": terminal,
        "nominal_profile_bits": nominal_bits,
        "nominal_budget_bits": nominal_budget_bits,
        "nominal_unused_bits": nominal_budget_bits - nominal_bits,
        "tie_rule": "strict-less updates; ascending q; lowest terminal index",
    }


def bec_synthesized_z(capacity: float, n: int) -> np.ndarray:
    full = 1 << 31
    capacity_q31 = min(full, max(0, int(round(float(capacity) * full))))
    z = np.full(n, full - capacity_q31, dtype=np.uint64)
    width = 1
    while width < n:
        view = z.reshape(-1, 2 * width)
        left = view[:, :width].copy()
        right = view[:, width:].copy()
        product = (left * right + np.uint64(1 << 30)) >> np.uint64(31)
        view[:, :width] = left + right - product
        view[:, width:] = product
        width *= 2
    return z


def bec_freeze_flags(n: int, capacities: Iterable[float], reverse: np.ndarray) -> list[np.ndarray]:
    flags = []
    canonical_index = np.arange(n, dtype=np.int64)
    for capacity in capacities:
        keep = min(n, max(0, int(math.ceil(n * float(capacity)))))
        external = np.ones(n, dtype=np.uint8)
        if keep == n:
            external[:] = 0
        elif keep:
            scores = bec_synthesized_z(float(capacity), n)
            order = np.lexsort((canonical_index, scores))
            external[order[:keep]] = 0
        flags.append(external[reverse].copy())
    return flags


def leaf_prior_ratios(weights: np.ndarray, previous: np.ndarray, level: int) -> np.ndarray:
    lower_mod = 1 << (level - 1)
    bit_value = 1 << (level - 1)
    ratios = np.empty(lower_mod, dtype=np.float64)
    indices = np.arange(weights.size)
    for context in range(lower_mod):
        matching = indices % lower_mod == context
        mass0 = weights[matching & ((indices & bit_value) == 0)].sum()
        mass1 = weights[matching & ((indices & bit_value) != 0)].sum()
        ratios[context] = mass0 / max(float(mass1), 1e-300)
    return np.clip(ratios[previous], 1e-30, 1e30)


def decode_sc_level(
    leaf_lr: np.ndarray,
    freeze_flag: np.ndarray,
    frozen_external: np.ndarray,
    reverse: np.ndarray,
    layers: np.ndarray,
    arithmetic: ArithmeticBinaryDecoder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal SC entropy decode; probabilities use decoded lower levels only."""
    n = leaf_lr.size
    depth = int(math.log2(n))
    lr_in = np.clip(np.asarray(leaf_lr, dtype=np.float64), 1e-30, 1e30)
    lr_reg = np.ones((n // 2, depth), dtype=np.float64)
    mu_reg = np.zeros((n // 2, depth), dtype=np.uint8)
    internal = np.zeros(n, dtype=np.uint8)
    frequencies = np.empty(int(np.count_nonzero(freeze_flag == 0)), dtype=np.uint16)
    selected_values = np.empty(frequencies.size, dtype=np.uint8)
    frequency_cursor = 0

    def bounded(value: np.ndarray | float) -> np.ndarray | float:
        return np.clip(value, 1e-30, 1e30)

    for i0 in range(n):
        one_based = i0 + 1
        if one_based == 1:
            end = int(layers[i0])
            col = end - 1
            left = lr_in[0::2]
            right = lr_in[1::2]
            lr_reg[:, col] = bounded((left * right + 1.0) / (left + right))
            for layer in range(end - 1, 0, -1):
                count = 1 << layer
                left = lr_reg[0:count:2, layer]
                right = lr_reg[1:count:2, layer]
                lr_reg[: count // 2, layer - 1] = bounded((left * right + 1.0) / (left + right))
        elif one_based == n // 2 + 1:
            end = int(layers[i0])
            col = end - 1
            left = lr_in[0::2]
            right = lr_in[1::2]
            used = mu_reg[:, -1].astype(np.int8)
            lr_reg[:, col] = bounded(np.power(left, 1 - 2 * used) * right)
            for layer in range(end - 1, 0, -1):
                count = 1 << layer
                left = lr_reg[0:count:2, layer]
                right = lr_reg[1:count:2, layer]
                lr_reg[: count // 2, layer - 1] = bounded((left * right + 1.0) / (left + right))
        elif one_based % 2 == 0:
            end = int(layers[i0])
            left = float(lr_reg[0, end])
            right = float(lr_reg[1, end])
            used = int(mu_reg[0, 0])
            lr_reg[0, end - 1] = bounded(left ** (1 - 2 * used) * right)
        else:
            end = int(layers[i0])
            count = 1 << end
            left = lr_reg[0:count:2, end]
            right = lr_reg[1:count:2, end]
            used = mu_reg[: count // 2, end - 1].astype(np.int8)
            lr_reg[: count // 2, end - 1] = bounded(np.power(left, 1 - 2 * used) * right)
            for layer in range(end - 1, 0, -1):
                count2 = 1 << layer
                left = lr_reg[0:count2:2, layer]
                right = lr_reg[1:count2:2, layer]
                lr_reg[: count2 // 2, layer - 1] = bounded((left * right + 1.0) / (left + right))

        root_lr = float(np.clip(lr_reg[0, 0], 1e-30, 1e30))
        if freeze_flag[i0]:
            value = int(frozen_external[reverse[i0]])
        else:
            p1 = 1.0 / (1.0 + root_lr)
            freq1 = min(65_535, max(1, int(math.floor(p1 * 65_536.0 + 0.5))))
            frequencies[frequency_cursor] = freq1
            value = arithmetic.decode(freq1)
            selected_values[frequency_cursor] = value
            frequency_cursor += 1
        internal[i0] = value

        if one_based % 2 == 1:
            mu_reg[0, 0] = value
        else:
            end = int(layers[one_based])
            temp = np.zeros(1 << max(end - 1, 0), dtype=np.uint8)
            temp[0] = value
            for layer in range(1, end):
                length = 1 << (layer - 1)
                left = mu_reg[:length, layer - 1]
                right = temp[:length].copy()
                merged = np.empty(2 * length, dtype=np.uint8)
                merged[0::2] = left ^ right
                merged[1::2] = right
                temp[: 2 * length] = merged
            mu_reg[: 1 << max(end - 1, 0), end - 1] = temp

    require(frequency_cursor == frequencies.size, "SC selected-frequency count drift")
    return polar_transform(internal[reverse]), frequencies, selected_values


def splitmix64_rademacher(n: int, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = np.arange(n, dtype=np.uint64) + np.uint64(seed)
        z += np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    return np.where((z & np.uint64(1)) == 0, 1.0, -1.0).astype(np.float64)


def inverse_signed_rht(values: np.ndarray, seed: int, device: str) -> np.ndarray:
    n = int(values.size)
    require(n > 0 and not (n & (n - 1)), "RHT length is not power of two")
    signs = splitmix64_rademacher(n, seed)
    if device == "cupy":
        import cupy as cp

        out = cp.asarray(values, dtype=cp.float64)
        width = 1
        while width < n:
            view = out.reshape(-1, 2, width)
            left = view[:, 0, :].copy()
            right = view[:, 1, :].copy()
            view[:, 0, :] = left + right
            view[:, 1, :] = left - right
            width *= 2
        out *= 1.0 / math.sqrt(n)
        out *= cp.asarray(signs)
        result = cp.asnumpy(out)
        del out
        cp.get_default_memory_pool().free_all_blocks()
        return result
    if device != "numpy":
        raise ValueError(device)
    out = np.asarray(values, dtype=np.float64).copy()
    width = 1
    while width < n:
        view = out.reshape(-1, 2, width)
        left = view[:, 0, :].copy()
        right = view[:, 1, :].copy()
        view[:, 0, :] = left + right
        view[:, 1, :] = left - right
        width *= 2
    out *= 1.0 / math.sqrt(n)
    out *= signs
    return out


def forward_signed_rht_and_rms(values: np.ndarray, seed: int, device: str) -> tuple[np.ndarray, float]:
    """Apply H diag(sign)/sqrt(N) and measure the post-RHT FP64 RMS."""
    n = int(values.size)
    require(n > 0 and not (n & (n - 1)), "RHT length is not power of two")
    signs = splitmix64_rademacher(n, seed)
    if device == "cupy":
        import cupy as cp

        out = cp.asarray(values, dtype=cp.float64)
        out *= cp.asarray(signs)
        width = 1
        while width < n:
            view = out.reshape(-1, 2, width)
            left = view[:, 0, :].copy()
            right = view[:, 1, :].copy()
            view[:, 0, :] = left + right
            view[:, 1, :] = left - right
            width *= 2
        out *= 1.0 / math.sqrt(n)
        rms = float(cp.sqrt(cp.mean(out * out)).get())
        result = cp.asnumpy(out)
        del out
        cp.get_default_memory_pool().free_all_blocks()
        return result, rms
    if device != "numpy":
        raise ValueError(device)
    out = np.asarray(values, dtype=np.float64).copy()
    out *= signs
    width = 1
    while width < n:
        view = out.reshape(-1, 2, width)
        left = view[:, 0, :].copy()
        right = view[:, 1, :].copy()
        view[:, 0, :] = left + right
        view[:, 1, :] = left - right
        width *= 2
    out *= 1.0 / math.sqrt(n)
    rms = float(np.sqrt(np.mean(out * out, dtype=np.float64)))
    return out, rms


def decode_one_block(
    container_path: str,
    block_ordinal: int,
    output_path: str,
    inverse_device: str,
) -> dict[str, Any]:
    parsed = parse_container(Path(container_path))
    row = parsed.directory[block_ordinal]
    profile = profile_parameters(row.profile_q, parsed.header.eta)
    n = row.block_values
    reverse = bit_reverse_indices(n)
    layers = sc_layers(n)
    flags = bec_freeze_flags(n, profile["capacities"], reverse)
    byte_offset = sum((item.logical_bits + 7) // 8 for item in parsed.directory[:block_ordinal])
    bit_offset = 8 * byte_offset
    arithmetic = ArithmeticBinaryDecoder(parsed.reservoir, bit_offset, row.logical_bits)
    sigma_reconstruction = float(profile["sigma_reconstruction"])
    alphabet = parsed.header.eta * np.arange(
        -ALPHABET_SIZE // 2 + 1, ALPHABET_SIZE // 2 + 1, dtype=np.float64
    )
    weights = np.exp(-0.5 * (alphabet / sigma_reconstruction) ** 2)
    previous = np.zeros(n, dtype=np.int16)
    frequency_hash = hashlib.sha256()
    selected = 0
    selected_chunks = []
    frequency_chunks = []
    level_rows = []
    sc_seed = parsed.sc_seeds[block_ordinal]
    started = time.perf_counter()
    for level_index, flag in enumerate(flags):
        level = level_index + 1
        frozen_rng = np.random.default_rng(sc_seed + 1_000_003 * level)
        frozen_external = frozen_rng.integers(0, 2, size=n, dtype=np.uint8)
        prior = leaf_prior_ratios(weights, previous, level)
        x_bit, frequencies, selected_values = decode_sc_level(
            prior, flag, frozen_external, reverse, layers, arithmetic
        )
        previous += (1 << level_index) * x_bit.astype(np.int16)
        frequency_hash.update(frequencies.astype("<u2", copy=False).tobytes())
        selected += int(frequencies.size)
        selected_chunks.append(selected_values)
        frequency_chunks.append(frequencies)
        level_rows.append(
            {
                "level": level,
                "capacity": float(profile["capacities"][level_index]),
                "selected": int(frequencies.size),
                "selected_fraction": frequencies.size / n,
            }
        )
    transformed = alphabet[previous] * row.decoder_scale
    reconstructed = inverse_signed_rht(transformed, parsed.rht_seeds[block_ordinal], inverse_device)
    output = Path(output_path)
    np.asarray(reconstructed, dtype="<f8").tofile(output)
    payload_bytes = (row.logical_bits + 7) // 8
    payload_packed = parsed.reservoir[byte_offset : byte_offset + payload_bytes]
    all_selected = np.concatenate(selected_chunks)
    all_frequencies = np.concatenate(frequency_chunks)
    canonical_payload, canonical_logical_bits = arithmetic_encode_binary(
        all_selected, all_frequencies
    )
    require(canonical_logical_bits == row.logical_bits, "canonical arithmetic length mismatch")
    require(canonical_payload == payload_packed, "canonical arithmetic payload mismatch")
    return {
        "block_ordinal": block_ordinal,
        "block_log2": row.block_log2,
        "values": n,
        "profile_q": row.profile_q,
        "profile_rate_bpw": float(profile["rate_bpw"]),
        "test_channel_distortion": float(profile["test_channel_distortion"]),
        "capacity_schedule": [float(value) for value in profile["capacities"]],
        "decoder_scale": row.decoder_scale,
        "decoder_scale_fp16_hex": row.decoder_scale_fp16_hex,
        "logical_bits": row.logical_bits,
        "reservoir_byte_offset": byte_offset,
        "reservoir_bit_offset": bit_offset,
        "payload_packed_sha256": sha256_bytes(payload_packed),
        "payload_terminal_padding_bits": len(payload_packed) * 8 - row.logical_bits,
        "payload_terminal_padding_rule": "low bits of final byte are zero",
        "canonical_reencode_logical_length_match": True,
        "canonical_reencode_payload_bytes_match": True,
        "canonical_reencode_sha256": sha256_bytes(canonical_payload),
        "sc_seed_u32": sc_seed,
        "rht_seed_u64": parsed.rht_seeds[block_ordinal],
        "selected_polar_bits": selected,
        "arithmetic_bits_read_including_zero_extension": arithmetic.cursor,
        "causal_frequency_u16_sha256": frequency_hash.hexdigest(),
        "reconstruction_indices_i16_sha256": hashlib.sha256(
            previous.astype("<i2", copy=False).tobytes()
        ).hexdigest(),
        "sorted_post_inverse_rht_f64_path": str(output),
        "sorted_post_inverse_rht_f64_sha256": sha256_file(output),
        "levels": level_rows,
        "seconds": time.perf_counter() - started,
        "decoder_imported_encoder": False,
        "decoder_read_encoder_probabilities": False,
        "decoder_read_encoder_decisions": False,
    }


def assemble_reconstructions(
    parsed: ParsedContainer,
    block_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    canonical_order = np.arange(GROUPS, dtype=np.int64)
    sorted_ordinals = np.lexsort((canonical_order, parsed.labels))
    require(np.array_equal(np.sort(sorted_ordinals), canonical_order), "label sort is not permutation")
    post_path = output_dir / "post_klt_canonical_natural_groups.f64.bin"
    post = np.memmap(post_path, dtype="<f8", mode="w+", shape=(GROUPS, GROUP_LENGTH))
    sorted_cursor = 0
    for row in sorted(block_rows, key=lambda item: int(item["block_ordinal"])):
        values = np.memmap(
            row["sorted_post_inverse_rht_f64_path"], dtype="<f8", mode="r",
            shape=(int(row["values"]),),
        ).reshape(-1, GROUP_LENGTH)
        count = int(values.shape[0])
        ordinals = sorted_ordinals[sorted_cursor : sorted_cursor + count]
        require(len(ordinals) == count, "decoded block exceeds label permutation")
        post[ordinals] = values
        sorted_cursor += count
    require(sorted_cursor == GROUPS, "decoded blocks do not cover all groups")
    post.flush()
    post_hash = sha256_file(post_path)

    final_path = output_dir / "original_domain_canonical_natural_groups.f64.bin"
    final = np.memmap(final_path, dtype="<f8", mode="w+", shape=(GROUPS, GROUP_LENGTH))
    for triplet, (cosine, sine) in enumerate(parsed.header.coefficients):
        gate_begin = triplet * 3 * GROUPS_PER_MATRIX
        up_begin = gate_begin + GROUPS_PER_MATRIX
        down_begin = up_begin + GROUPS_PER_MATRIX
        final[gate_begin:up_begin] = post[gate_begin:up_begin]
        z0 = np.asarray(post[up_begin:down_begin])
        z1 = np.asarray(post[down_begin : down_begin + GROUPS_PER_MATRIX])
        # The forward source transform is [c s; -s c].  Stored FP32 c/s are
        # extremely close to unit norm but not algebraically exact, so the
        # normative inverse is A^T/(c^2+s^2), not merely A^T.
        norm_squared = cosine * cosine + sine * sine
        require(norm_squared > 0.0 and math.isfinite(norm_squared), "invalid KLT inverse norm")
        final[up_begin:down_begin] = (cosine * z0 - sine * z1) / norm_squared
        final[down_begin : down_begin + GROUPS_PER_MATRIX] = (
            sine * z0 + cosine * z1
        ) / norm_squared
    final.flush()
    final_hash = sha256_file(final_path)
    return {
        "label_sort_u32_le_sha256": hashlib.sha256(sorted_ordinals.astype("<u4").tobytes()).hexdigest(),
        "post_klt_canonical_path": str(post_path),
        "post_klt_canonical_sha256": post_hash,
        "original_domain_canonical_path": str(final_path),
        "original_domain_canonical_sha256": final_hash,
        "values": WEIGHTS,
        "dtype": "little-endian float64",
        "inverse_klt": {
            "formula": "A^T/(c^2+s^2) for A=[[c,s],[-s,c]]",
            "coefficient_norm_squared": [
                float(cosine * cosine + sine * sine)
                for cosine, sine in parsed.header.coefficients
            ],
            "division_by_norm_squared_enforced": True,
        },
    }


def safe_resolve(root: Path, relpath: str) -> Path:
    relative = Path(relpath)
    require(
        not relative.is_absolute() and bool(relative.parts) and all(part != ".." for part in relative.parts),
        f"unsafe reference relpath {relpath!r}",
    )
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / relative).resolve(strict=True)
    try:
        path.relative_to(resolved_root)
    except ValueError as exc:
        raise AssertionError(f"reference path escaped root: {relpath}") from exc
    return path


def tensor_identity(row: dict[str, Any]) -> str:
    for key in ("canonical_tensor_id", "tensor"):
        if key in row:
            return str(row[key])
    raise AssertionError("reference matrix has no tensor identity")


def normalized_role(value: Any) -> str:
    role = str(value).lower()
    return role[:-5] if role.endswith("_proj") else role


def validate_source_lineage(
    parsed: ParsedContainer,
    protocol_mode: str,
    selection_path: Path,
    source_lock_path: Path,
    codec_freeze_path: Path,
    format_freeze_path: Path,
    preencoding_manifest_path: Path,
    allocation_lock_path: Path,
    one_shot_intent_path: Path,
    one_shot_summary_path: Path,
    source_root_override: Path | None,
) -> dict[str, Any]:
    """Validate all pre-source and one-shot bindings without encoder imports."""
    require(protocol_mode in ("blind", "development"), "invalid lineage protocol mode")
    selection, selection_file_hash = load_json_object(selection_path, "selection lock")
    source_lock, source_file_hash = load_json_object(source_lock_path, "source lock")
    codec_freeze, codec_file_hash = load_json_object(codec_freeze_path, "codec freeze")
    manifest, manifest_file_hash = load_json_object(
        preencoding_manifest_path, "preencoding manifest"
    )
    allocation, allocation_file_hash = load_json_object(allocation_lock_path, "allocation lock")
    intent, intent_file_hash = load_json_object(one_shot_intent_path, "one-shot intent")
    summary, summary_file_hash = load_json_object(one_shot_summary_path, "one-shot summary")
    format_path = format_freeze_path.resolve(strict=True)
    format_file_hash = sha256_file(format_path)

    selection_internal = verify_internal_lock(selection, "selection lock")
    source_internal = verify_internal_lock(source_lock, "source lock")
    codec_internal = verify_internal_lock(codec_freeze, "codec freeze")
    allocation_internal = verify_internal_lock(allocation, "allocation lock")
    validation_path: Path | None = None
    validation: dict[str, Any] | None = None
    validation_file_hash: str | None = None
    validation_internal: str | None = None
    if protocol_mode == "blind":
        validation_path = codec_freeze_path.resolve(strict=True).with_name(
            "codec_freeze.validation.json"
        )
        validation, validation_file_hash = load_json_object(
            validation_path, "codec-freeze validation receipt"
        )
        validation_internal = verify_internal_lock(
            validation, "codec-freeze validation receipt"
        )

    expected_selection_contract = (
        (
            "int2-qwen-blind-selection-proposal-v2",
            "sealed_metadata_only_proposal_payload_unopened_not_codec_frozen",
        )
        if protocol_mode == "blind"
        else (
            "int2-qwen-blind-selection-v1",
            "selected_and_header_validated_tensor_payload_unopened",
        )
    )
    expected_source_contract = (
        "int2-qwen-blind-source-finalization-v2"
        if protocol_mode == "blind"
        else "int2-qwen-blind-source-finalization-v1"
    )
    require(
        (selection.get("schema"), selection.get("status")) == expected_selection_contract,
        f"{protocol_mode} selection schema/status mismatch",
    )
    require(
        (source_lock.get("schema"), source_lock.get("status"))
        == (
            expected_source_contract,
            "all_locked_sources_materialized_and_hash_finalized",
        ),
        f"{protocol_mode} source-finalization schema/status mismatch",
    )
    if protocol_mode == "blind":
        require(
            source_lock.get("dtype") == "BF16",
            "blind source-finalization top-level dtype must be BF16",
        )
    require(
        (codec_freeze.get("schema"), codec_freeze.get("status"))
        == (
            "strata_xklt_sc_v2_codec_freeze_v1"
            if protocol_mode == "blind"
            else "polaris_strata_blind_codec_freeze_v1",
            "frozen_before_blind_source_access",
        ),
        "codec freeze schema/status is not the pre-source contract",
    )
    require(codec_freeze.get("selection_lock_sha256") == selection_internal, "freeze selection seal mismatch")
    require(source_lock.get("selection_lock_sha256") == selection_internal, "source selection seal mismatch")
    source_freeze_binding = source_lock.get("codec_freeze")
    require(
        isinstance(source_freeze_binding, dict)
        and source_freeze_binding.get("file_sha256") == codec_file_hash
        and source_freeze_binding.get("internal_lock_sha256") == codec_internal,
        "source lock does not bind the exact codec-freeze file and seal",
    )
    require(
        float(codec_freeze.get("physical_rate_limit_bpw", float("nan"))) == 2.15,
        "codec freeze physical-rate limit mismatch",
    )
    require(
        float(codec_freeze.get("primary_mse_threshold", float("nan"))) == math.exp2(-4.3),
        "codec freeze primary MSE threshold mismatch",
    )
    require(codec_freeze.get("allocator_frozen") is True, "codec freeze allocator is not frozen")

    frozen_artifacts = codec_freeze.get("frozen_artifact_sha256s")
    require(isinstance(frozen_artifacts, dict), "codec freeze lacks frozen artifact map")
    for name, value in frozen_artifacts.items():
        require_sha256(value, f"codec frozen artifact {name}")


    route_hash = sha256_bytes(parsed.route)
    current_auditor_hash = sha256_file(Path(__file__).resolve())
    if protocol_mode == "blind":
        assert validation is not None
        assert validation_file_hash is not None
        assert validation_internal is not None
        expected_validation_keys = {
            "schema",
            "status",
            "passed",
            "freeze_path",
            "freeze_file_sha256",
            "freeze_internal_lock_sha256",
            "executing_validator_sha256",
            "frozen_artifact_count",
            "development_pooled_relative_mse",
            "gaussian_mse_reference",
            "physical_bits",
            "physical_bpw",
            "preaccess_state",
            "lock_sha256",
        }
        development_mse = float(
            validation.get("development_pooled_relative_mse", float("nan"))
        )
        require(set(validation) == expected_validation_keys, "validation receipt key set")
        require(
            (validation.get("schema"), validation.get("status"), validation.get("passed"))
            == (
                "strata_xklt_sc_v2_codec_freeze_validation_v1",
                "validated_before_blind_source_access",
                True,
            ),
            "validation receipt schema/status/pass",
        )
        require(
            validation.get("freeze_path") == "blind_protocol_v2/codec_freeze.lock.json"
            and validation.get("freeze_file_sha256") == codec_file_hash
            and validation.get("freeze_internal_lock_sha256") == codec_internal,
            "validation receipt freeze binding",
        )
        require(
            validation.get("executing_validator_sha256")
            == frozen_artifacts.get("freeze_validator")
            and int(validation.get("frozen_artifact_count", -1))
            == len(frozen_artifacts),
            "validation receipt validator/artifact binding",
        )
        require(
            math.isfinite(development_mse)
            and development_mse < math.exp2(-4.3)
            and validation.get("gaussian_mse_reference") == math.exp2(-4.3)
            and int(validation.get("physical_bits", -1)) == CONTAINER_BYTES * 8
            and validation.get("physical_bpw") == (CONTAINER_BYTES * 8) / WEIGHTS
            and validation.get("preaccess_state") == codec_freeze.get("preaccess_state"),
            "validation receipt result/physical contract",
        )
        source_validation_binding = source_lock.get("codec_freeze_validation")
        require(
            isinstance(source_validation_binding, dict)
            and set(source_validation_binding)
            == {"file_sha256", "internal_lock_sha256"}
            and source_validation_binding.get("file_sha256") == validation_file_hash
            and source_validation_binding.get("internal_lock_sha256")
            == validation_internal,
            "source lock does not bind exact validation receipt",
        )
        require(
            codec_freeze.get("selection_lock_file_sha256") == selection_file_hash,
            "codec freeze does not bind exact selection-lock bytes",
        )
        require(
            codec_freeze.get("route_file_sha256") == route_hash,
            "codec freeze does not bind exact literal route bytes",
        )
        require(
            frozen_artifacts.get("format") == format_file_hash,
            "codec freeze named FORMAT digest mismatch",
        )
        require(
            frozen_artifacts.get("independent_auditor") == current_auditor_hash,
            "codec freeze named executing-auditor digest mismatch",
        )

    require(selection.get("checkpoint") == source_lock.get("checkpoint"), "checkpoint binding mismatch")
    selection_matrices = selection.get("matrices")
    source_matrices = source_lock.get("matrices")
    require(isinstance(selection_matrices, list) and len(selection_matrices) == MATRICES, "selection matrix count mismatch")
    require(isinstance(source_matrices, list) and len(source_matrices) == MATRICES, "source matrix count mismatch")
    require(
        (
            int(source_lock.get("matrix_count", -1)),
            int(source_lock.get("block_count", -1)),
            int(source_lock.get("source_values", -1)),
            int(source_lock.get("source_bytes", -1)),
        )
        == (MATRICES, MATRICES * 6, WEIGHTS, 2 * WEIGHTS),
        "source-finalization aggregate geometry mismatch",
    )
    source_root = (
        source_root_override.resolve(strict=True)
        if source_root_override is not None
        else (source_lock_path.resolve(strict=True).parent / str(source_lock.get("source_root", "."))).resolve(strict=True)
    )
    route_rows = validate_route(parsed.route)
    source_by_ordinal = {int(row["matrix_ordinal"]): row for row in source_matrices}
    require(len(source_by_ordinal) == MATRICES, "duplicate source matrix ordinals")
    matrix_rows: list[dict[str, Any]] = []
    expected_source_bindings: list[dict[str, Any]] = []
    for ordinal, (selected, route) in enumerate(zip(selection_matrices, route_rows, strict=True)):
        require(int(selected.get("matrix_ordinal", -1)) == ordinal, f"selection ordinal mismatch {ordinal}")
        source = source_by_ordinal.get(ordinal)
        require(source is not None, f"source omits ordinal {ordinal}")
        assert source is not None
        selected_blocks = selected.get("blocks")
        source_blocks = source.get("blocks")
        if protocol_mode == "development" and selected_blocks is None:
            # Historical v1 metadata-only selections predate nested null-hash
            # rows; the finalized v1 source receipt still supplies all six
            # canonical hashes and geometries below.
            selected_blocks = [
                {"canonical_block_index": index, "source_bf16_sha256": None}
                for index in range(6)
            ]
        require(isinstance(selected_blocks, list) and len(selected_blocks) == 6, f"selection block count {ordinal}")
        require(isinstance(source_blocks, list) and len(source_blocks) == 6, f"source block count {ordinal}")
        require(selected.get("source_bf16_sha256") is None, f"selection leaked matrix hash {ordinal}")
        require(
            all(block.get("source_bf16_sha256") is None for block in selected_blocks),
            f"selection leaked nested block hash {ordinal}",
        )
        role = normalized_role(selected.get("role"))
        expected_role = ("gate", "up", "down")[int(route["role"])]
        expected_axis = "column" if expected_role == "down" else "row"
        expected_shape = [GROUP_LENGTH, GROUPS_PER_MATRIX] if role == "down" else [GROUPS_PER_MATRIX, GROUP_LENGTH]
        shape = [int(value) for value in selected.get("shape", [])]
        source_shape = [int(value) for value in source.get("shape", [])]
        identity = tensor_identity(selected)
        source_identity = tensor_identity(source)
        match = re.fullmatch(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight",
            identity,
        )
        require(match is not None, f"noncanonical selected tensor {identity}")
        assert match is not None
        if protocol_mode == "blind":
            future_relpath = selected.get("future_output_relpath")
            require("output_relpath" not in selected, f"blind proposal uses opened path key {ordinal}")
        else:
            future_relpath = selected.get("output_relpath")
        source_relpath = source.get("output_relpath")
        required_source_identity = {"role", "layer", "expert", "block_count"}
        if protocol_mode == "blind":
            require(
                required_source_identity.issubset(source),
                f"source omits finalized matrix identity fields {ordinal}",
            )
        # Development-v1 source locks predate these redundant fields. Blind-v2
        # requires them explicitly; development may derive the same identity
        # from the canonical tensor already sealed in the source receipt.
        source_role = normalized_role(source.get("role", role))
        source_layer = int(source.get("layer", selected.get("layer", -1)))
        source_expert = int(source.get("expert", selected.get("expert", -1)))
        selected_shard = selected.get("shard")
        selected_range = selected.get("absolute_http_byte_range_inclusive")
        source_range = source.get("http_range_inclusive")
        range_ok = (
            isinstance(selected_range, list)
            and len(selected_range) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in selected_range
            )
            and int(selected_range[0]) >= 0
            and int(selected_range[1]) >= int(selected_range[0])
            and source_range == selected_range
            and int(selected_range[1]) - int(selected_range[0]) + 1
            == 2 * GROUPS_PER_MATRIX * GROUP_LENGTH
        )
        http_response = source.get("http_response")
        blind_http_response_ok = protocol_mode != "blind"
        if protocol_mode == "blind" and range_ok and isinstance(http_response, dict):
            range_begin, range_end = int(selected_range[0]), int(selected_range[1])
            content_range_match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)",
                str(http_response.get("content_range", "")),
            )
            checkpoint = selection.get("checkpoint", {})
            expected_url = (
                f"https://huggingface.co/{checkpoint.get('repo')}/resolve/"
                f"{checkpoint.get('revision')}/{selected_shard}"
            )
            blind_http_response_ok = (
                set(http_response)
                == {
                    "status",
                    "request_url",
                    "requested_range",
                    "content_range",
                    "content_length",
                    "content_encoding",
                    "body_bytes",
                    "body_sha256",
                }
                and int(http_response.get("status", -1)) == 206
                and http_response.get("request_url") == expected_url
                and http_response.get("requested_range")
                == f"bytes={range_begin}-{range_end}"
                and int(http_response.get("content_length", -1))
                == range_end - range_begin + 1
                and http_response.get("content_encoding") == "identity"
                and int(http_response.get("body_bytes", -1))
                == range_end - range_begin + 1
                and http_response.get("body_sha256")
                == source.get("source_bf16_sha256")
                and content_range_match is not None
                and int(content_range_match.group(1)) == range_begin
                and int(content_range_match.group(2)) == range_end
                and int(content_range_match.group(3)) > range_end
            )
        critical_checks = {
            "source_ordinal": int(source.get("matrix_ordinal", -1)) == ordinal,
            "tensor": identity == source_identity,
            "role": role == source_role == expected_role == match.group(3),
            "layer": int(selected.get("layer", -1)) == source_layer == int(route["layer"]) == int(match.group(1)),
            "expert": int(selected.get("expert", -1)) == source_expert == int(route["expert"]) == int(match.group(2)),
            "shape": shape == source_shape == expected_shape,
            "dtype": str(selected.get("dtype", "BF16")).upper() == "BF16"
            and str(source.get("dtype", "")).upper() == "BF16",
            "nvalues": int(selected.get("nvalues", GROUPS_PER_MATRIX * GROUP_LENGTH))
            == int(source.get("nvalues", -1))
            == GROUPS_PER_MATRIX * GROUP_LENGTH,
            "nbytes": int(selected.get("nbytes", 2 * GROUPS_PER_MATRIX * GROUP_LENGTH))
            == int(source.get("nbytes", -1))
            == 2 * GROUPS_PER_MATRIX * GROUP_LENGTH,
            "block_count": int(source.get("block_count", len(source_blocks))) == 6,
            "shard": isinstance(selected_shard, str)
            and bool(selected_shard)
            and source.get("shard") == selected_shard,
            "http_range": range_ok,
            "blind_http_206_receipt": blind_http_response_ok,
            "path": isinstance(future_relpath, str) and future_relpath == source_relpath,
        }
        require(all(critical_checks.values()), f"selection/source/route matrix mismatch {ordinal}: {critical_checks}")
        path = safe_resolve(source_root, str(source_relpath))
        payload = path.read_bytes()
        require(len(payload) == 2 * GROUPS_PER_MATRIX * GROUP_LENGTH, f"source byte count {ordinal}")
        matrix_digest = require_sha256(source.get("source_bf16_sha256"), f"source matrix hash {ordinal}")
        require(sha256_bytes(payload) == matrix_digest, f"source matrix hash mismatch {ordinal}")
        source_block_digests = []
        for block_ordinal, (selected_block, source_block) in enumerate(
            zip(selected_blocks, source_blocks, strict=True)
        ):
            begin = block_ordinal * (1 << 18)
            end = begin + (1 << 18)
            selection_optional_geometry_ok = (
                ("flat_value_start" not in selected_block or int(selected_block["flat_value_start"]) == begin)
                and (
                    "flat_value_end_exclusive" not in selected_block
                    or int(selected_block["flat_value_end_exclusive"]) == end
                )
                and int(selected_block.get("nvalues", 1 << 18)) == 1 << 18
                and int(selected_block.get("nbytes", 1 << 19)) == 1 << 19
            )
            source_optional_geometry_ok = (
                ("flat_value_start" not in source_block or int(source_block["flat_value_start"]) == begin)
                and (
                    "flat_value_end_exclusive" not in source_block
                    or int(source_block["flat_value_end_exclusive"]) == end
                )
                and int(source_block.get("nvalues", -1)) == 1 << 18
                and int(source_block.get("nbytes", -1)) == 1 << 19
            )
            require(
                int(selected_block.get("canonical_block_index", -1)) == block_ordinal
                and int(source_block.get("canonical_block_index", -1)) == block_ordinal
                and selection_optional_geometry_ok
                and source_optional_geometry_ok,
                f"nested source block geometry mismatch matrix {ordinal} block {block_ordinal}",
            )
            block_digest = require_sha256(
                source_block.get("source_bf16_sha256"),
                f"source block hash matrix {ordinal} block {block_ordinal}",
            )
            require(
                sha256_bytes(payload[2 * begin : 2 * end]) == block_digest,
                f"nested source block hash mismatch matrix {ordinal} block {block_ordinal}",
            )
            source_block_digests.append(block_digest)
        row = {
            "matrix_ordinal": ordinal,
            "tensor": identity,
            "role": role,
            "layer": int(route["layer"]),
            "expert": int(route["expert"]),
            "axis": expected_axis,
            "groups": int(route["groups"]),
            "shape": source_shape,
            "source_path": path,
            "source_payload": payload,
            "source_relpath": str(source_relpath),
            "source_bf16_sha256": matrix_digest,
            "source_block_sha256s": source_block_digests,
            "shard": selected_shard,
            "http_range_inclusive": source_range,
            **(
                {"http_response": http_response}
                if protocol_mode == "blind"
                else {}
            ),
        }
        matrix_rows.append(row)
        expected_source_bindings.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"source_path", "source_payload"}
            }
        )

    require(
        (manifest.get("schema"), manifest.get("status"))
        == (
            "strata_xklt_sc_v2_preencoding_manifest_v1",
            "complete_and_allocation_sealed_before_encoding",
        ),
        "preencoding manifest schema/status mismatch",
    )
    require(manifest.get("protocol_mode") == protocol_mode, "preencoding manifest mode mismatch")
    bindings = manifest.get("bindings")
    require(
        isinstance(bindings, dict) and bindings.get("protocol_mode") == protocol_mode,
        "manifest binding mode mismatch",
    )
    expected_binding_fields = {
        "selection_lock": {
            "file_sha256": selection_file_hash,
            "internal_lock_sha256": selection_internal,
        },
        "source_lock": {
            "file_sha256": source_file_hash,
            "internal_lock_sha256": source_internal,
            "selection_lock_sha256": selection_internal,
        },
        "codec_freeze": {
            "file_sha256": codec_file_hash,
            "internal_lock_sha256": codec_internal,
        },
        "route": {"sha256": route_hash, "bytes": ROUTE_BYTES},
        "format_freeze": {"sha256": format_file_hash},
    }
    if protocol_mode == "blind":
        assert validation_file_hash is not None
        assert validation_internal is not None
        expected_binding_fields["codec_freeze_validation"] = {
            "file_sha256": validation_file_hash,
            "internal_lock_sha256": validation_internal,
        }
    for name, expected_fields in expected_binding_fields.items():
        actual = bindings.get(name)
        require(isinstance(actual, dict), f"manifest lacks {name} binding")
        require(
            all(actual.get(key) == value for key, value in expected_fields.items()),
            f"manifest {name} binding mismatch",
        )
    require(bindings.get("sources") == expected_source_bindings, "manifest source binding rows mismatch")
    for executable_name in ("emitter", "common"):
        executable = bindings.get(executable_name)
        require(isinstance(executable, dict), f"manifest lacks {executable_name} binding")
        executable_hash = require_sha256(executable.get("sha256"), f"manifest {executable_name} hash")
        if protocol_mode == "blind":
            require(
                executable_hash == frozen_artifacts.get(executable_name),
                f"manifest {executable_name} does not match its named codec-freeze entry",
            )

    assets = manifest.get("assets")
    require(isinstance(assets, dict), "manifest assets missing")
    literal_assets = {
        "header.bin": parsed.header.raw,
        "route.bin": parsed.route,
        "labels_3bit.bin": parsed.labels_packed,
        "profiles.bin": parsed.profile_bytes,
    }
    for filename, payload in literal_assets.items():
        row = assets.get(filename)
        require(
            isinstance(row, dict)
            and int(row.get("bytes", -1)) == len(payload)
            and row.get("sha256") == sha256_bytes(payload),
            f"manifest literal asset mismatch {filename}",
        )
    blocks = manifest.get("blocks")
    require(isinstance(blocks, list) and len(blocks) == BLOCKS, "manifest block count mismatch")
    for ordinal, (block, directory) in enumerate(zip(blocks, parsed.directory, strict=True)):
        require(
            int(block.get("block_ordinal", -1)) == ordinal
            and int(block.get("block_log2", -1)) == directory.block_log2
            and int(block.get("values", -1)) == directory.block_values
            and int(block.get("profile_id", -1)) == directory.profile_q
            and int(block.get("sc_seed_u32", -1)) == parsed.sc_seeds[ordinal]
            and int(block.get("rht_seed_u64", -1)) == parsed.rht_seeds[ordinal],
            f"manifest/container block decision mismatch {ordinal}",
        )
    require(
        list(manifest.get("allocation", {}).get("profile_ids", []))
        == [row.profile_q for row in parsed.directory],
        "manifest allocation profile vector mismatch",
    )
    require(
        (allocation.get("schema"), allocation.get("status"))
        == (
            "strata_xklt_sc_v2_allocation_lock_v1",
            "allocation_sealed_before_first_encoder_invocation",
        ),
        "allocation lock schema/status mismatch",
    )
    require(allocation.get("manifest_sha256") == manifest_file_hash, "allocation does not bind manifest")
    for key in ("bindings", "assets", "physical_format", "allocation", "blocks"):
        require(
            canonical_json_bytes(allocation.get(key)) == canonical_json_bytes(manifest.get(key)),
            f"allocation/manifest {key} mismatch",
        )

    require(
        (intent.get("schema"), intent.get("status"))
        == ("strata_xklt_sc_v2_one_shot_intent_v1", "sealed_before_first_encoder_invocation"),
        "one-shot intent schema/status mismatch",
    )
    require(
        intent.get("allocation_lock_file_sha256") == allocation_file_hash
        and intent.get("allocation_lock_internal_sha256") == allocation_internal
        and intent.get("manifest_sha256") == manifest_file_hash
        and intent.get("protocol_mode") == protocol_mode
        and int(intent.get("encoder_invocations_planned", -1)) == BLOCKS
        and intent.get("retry_resume_or_adaptive_rate_change_allowed") is False,
        "one-shot intent binding mismatch",
    )
    runtime_freeze = intent.get("runtime_freeze")
    require(isinstance(runtime_freeze, dict), "intent lacks runtime freeze receipt")
    runtime_codec = runtime_freeze.get("codec_freeze")
    require(
        isinstance(runtime_codec, dict)
        and runtime_codec.get("file_sha256") == codec_file_hash
        and runtime_codec.get("internal_lock_sha256") == codec_internal,
        "intent runtime codec-freeze binding mismatch",
    )
    runtime_artifacts = runtime_freeze.get("artifacts")
    require(isinstance(runtime_artifacts, dict), "intent lacks frozen runtime artifacts")
    required_runtime_keys = set(RUNTIME_ARTIFACT_FREEZE_KEYS)
    # The scoring process itself uses NumPy/CuPy.  Measure and report its full
    # Python/package/CUDA receipt in development as well as blind mode so the
    # final rehearsal can bind the exact environment that produced the score.
    actual_runtime_environment = independent_runtime_environment_receipt()
    if protocol_mode == "blind":
        require(
            set(runtime_artifacts) == required_runtime_keys,
            "blind intent runtime artifact key set is incomplete or noncanonical",
        )
    for name, row in runtime_artifacts.items():
        require(isinstance(row, dict), f"invalid intent runtime artifact {name}")
        runtime_hash = require_sha256(row.get("sha256"), f"intent runtime artifact {name}")
        if protocol_mode == "blind":
            freeze_key = RUNTIME_ARTIFACT_FREEZE_KEYS[name]
            require(
                runtime_hash == frozen_artifacts.get(freeze_key),
                f"intent runtime artifact {name} does not match named freeze entry {freeze_key}",
            )
    runner_hash = require_sha256(intent.get("runner_sha256"), "intent runner_sha256")
    encoder_hash = require_sha256(intent.get("encoder_sha256"), "intent encoder_sha256")
    if protocol_mode == "blind":
        require(
            runner_hash == frozen_artifacts.get("one_shot_runner")
            and encoder_hash == frozen_artifacts.get("polar_encoder"),
            "intent runner/encoder do not match their named codec-freeze entries",
        )
        require(
            runner_hash == runtime_artifacts["runner"]["sha256"]
            and encoder_hash == runtime_artifacts["polar_encoder"]["sha256"]
            and current_auditor_hash == runtime_artifacts["independent_auditor"]["sha256"]
            and format_file_hash == runtime_artifacts["format"]["sha256"]
            and bindings["common"]["sha256"] == runtime_artifacts["common"]["sha256"]
            and bindings["emitter"]["sha256"] == runtime_artifacts["emitter"]["sha256"],
            "blind intent runtime identity binding mismatch",
        )
        frozen_runtime_environment = codec_freeze.get("runtime_environment")
        require(
            isinstance(frozen_runtime_environment, dict),
            "blind codec freeze lacks runtime environment receipt",
        )
        frozen_python = frozen_runtime_environment.get("python_interpreter")
        require(
            isinstance(frozen_python, dict)
            and runtime_artifacts["python_interpreter"].get("path")
            == frozen_python.get("invocation_path")
            and runtime_artifacts["python_interpreter"].get("sha256")
            == frozen_python.get("sha256"),
            "intent Python artifact does not match frozen invocation path/hash",
        )
        require(
            runtime_freeze.get("packages") == frozen_runtime_environment.get("packages"),
            "intent package receipt does not match codec freeze",
        )
        require(
            runtime_freeze.get("cuda") == frozen_runtime_environment.get("cuda"),
            "intent CUDA receipt does not match codec freeze",
        )
        require(
            actual_runtime_environment == frozen_runtime_environment,
            "independently measured auditor Python/package/CUDA runtime differs from freeze",
        )
    require(
        (summary.get("schema"), summary.get("status"))
        == ("strata_xklt_sc_v2_one_shot_summary_v1", "one-shot physical artifact complete"),
        "one-shot summary schema/status mismatch",
    )
    require(
        summary.get("protocol_mode") == protocol_mode
        and summary.get("allocation_lock_file_sha256") == allocation_file_hash
        and summary.get("allocation_lock_internal_sha256") == allocation_internal
        and summary.get("intent_sha256") == intent_file_hash
        and int(summary.get("encoder_invocations", -1)) == BLOCKS
        and int(summary.get("retries", -1)) == 0
        and int(summary.get("resumes", -1)) == 0
        and int(summary.get("postencoding_profile_changes", -1)) == 0,
        "one-shot summary lineage mismatch",
    )
    encoded_blocks = summary.get("encoded_blocks")
    require(isinstance(encoded_blocks, list) and len(encoded_blocks) == BLOCKS, "summary block count mismatch")
    for ordinal, (encoded, directory) in enumerate(zip(encoded_blocks, parsed.directory, strict=True)):
        require(
            int(encoded.get("block_ordinal", -1)) == ordinal
            and int(encoded.get("encoder_invocations", -1)) == 1
            and int(encoded.get("logical_bits", -1)) == directory.logical_bits,
            f"summary/container encoded block mismatch {ordinal}",
        )
    physical = summary.get("physical")
    require(isinstance(physical, dict), "summary physical record missing")
    used_bytes = sum((row.logical_bits + 7) // 8 for row in parsed.directory)
    require(
        physical.get("artifact_sha256") == parsed.raw_sha256
        and int(physical.get("physical_bytes", -1)) == CONTAINER_BYTES
        and int(physical.get("physical_bits", -1)) == CONTAINER_BYTES * 8
        and physical.get("integer_2p15_gate_passed") is True
        and int(physical.get("directory_bytes", -1)) == DIRECTORY_BYTES
        and int(physical.get("logical_payload_bits", -1)) == sum(row.logical_bits for row in parsed.directory)
        and int(physical.get("payload_byte_count", -1)) == used_bytes
        and int(physical.get("zero_reservoir_tail_bytes", -1)) == RESERVOIR_BYTES - used_bytes
        and physical.get("reservoir_fit") is True,
        "summary/container physical record mismatch",
    )
    return {
        "selection": selection,
        "source_lock": source_lock,
        "codec_freeze": codec_freeze,
        "manifest": manifest,
        "allocation": allocation,
        "intent": intent,
        "summary": summary,
        "matrices": matrix_rows,
        "source_root": source_root,
        "report": {
            "all_checks_passed": True,
            "protocol_mode": protocol_mode,
            "blind_positive_claim_eligible": protocol_mode == "blind",
            "selection_lock": {
                "path": str(selection_path.resolve(strict=True)),
                "file_sha256": selection_file_hash,
                "internal_lock_sha256": selection_internal,
            },
            "source_lock": {
                "path": str(source_lock_path.resolve(strict=True)),
                "file_sha256": source_file_hash,
                "internal_lock_sha256": source_internal,
            },
            "codec_freeze": {
                "path": str(codec_freeze_path.resolve(strict=True)),
                "file_sha256": codec_file_hash,
                "internal_lock_sha256": codec_internal,
            },
            **(
                {
                    "codec_freeze_validation": {
                        "path": str(validation_path),
                        "file_sha256": validation_file_hash,
                        "internal_lock_sha256": validation_internal,
                    }
                }
                if protocol_mode == "blind"
                else {}
            ),
            "format_freeze": {"path": str(format_path), "sha256": format_file_hash},
            "preencoding_manifest": {
                "path": str(preencoding_manifest_path.resolve(strict=True)),
                "sha256": manifest_file_hash,
            },
            "allocation_lock": {
                "path": str(allocation_lock_path.resolve(strict=True)),
                "file_sha256": allocation_file_hash,
                "internal_lock_sha256": allocation_internal,
            },
            "one_shot_intent": {"path": str(one_shot_intent_path.resolve(strict=True)), "sha256": intent_file_hash},
            "one_shot_summary": {"path": str(one_shot_summary_path.resolve(strict=True)), "sha256": summary_file_hash},
            "executing_independent_auditor_sha256": current_auditor_hash,
            "independently_measured_runtime_environment": actual_runtime_environment,
            "matrix_bindings_and_nested_hashes": f"{MATRICES}/{MATRICES} matrices, {MATRICES * 6}/{MATRICES * 6} nested blocks",
        },
    }


def bf16_words_to_fp32(words: np.ndarray) -> np.ndarray:
    source = np.asarray(words, dtype=np.uint16)
    return (source.astype(np.uint32) << np.uint32(16)).view(np.float32)


def fp32_to_bf16_rne(values: np.ndarray) -> np.ndarray:
    words = np.asarray(values, dtype="<f4").view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2")


def exact_fp32_klt(
    up: np.ndarray, down: np.ndarray, cosine: float, sine: float
) -> tuple[np.ndarray, np.ndarray]:
    """Separate FP32 multiplies/adds; FMA contraction is structurally impossible."""
    c = np.float32(cosine)
    s = np.float32(sine)
    c_up = np.asarray(c * up, dtype=np.float32)
    s_down = np.asarray(s * down, dtype=np.float32)
    neg_s_up = np.asarray(-s * up, dtype=np.float32)
    c_down = np.asarray(c * down, dtype=np.float32)
    return (
        np.asarray(c_up + s_down, dtype=np.float32),
        np.asarray(neg_s_up + c_down, dtype=np.float32),
    )


def derive_q15_klt_from_source(up: np.ndarray, down: np.ndarray) -> tuple[int, float, float, dict[str, float]]:
    up64 = np.asarray(up, dtype=np.float64)
    down64 = np.asarray(down, dtype=np.float64)
    a = float(np.sum(up64 * up64, dtype=np.float64))
    b = float(np.sum(down64 * down64, dtype=np.float64))
    cross = float(np.sum(up64 * down64, dtype=np.float64))
    theta = 0.5 * math.atan2(2.0 * cross, a - b)
    code = int(np.clip(np.rint(theta / math.pi * Q15_DENOMINATOR), -16_384, 16_384))
    decoded = code * math.pi / Q15_DENOMINATOR
    cosine = float(np.float32(math.cos(decoded)))
    sine = float(np.float32(math.sin(decoded)))
    return code, cosine, sine, {"energy_up_fp64": a, "energy_down_fp64": b, "cross_fp64": cross}


def audit_source_staging_and_scales(
    parsed: ParsedContainer,
    lineage: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Recreate KLT BF16 staging, labels, RHTs and directory scales from sources."""
    matrices = lineage["matrices"]
    manifest = lineage["manifest"]
    summary = lineage["summary"]
    canonical_words = np.empty((GROUPS, GROUP_LENGTH), dtype="<u2")
    triplet_rows = []
    source_energy = 0.0
    for triplet in range(6):
        rows = matrices[3 * triplet : 3 * triplet + 3]
        natural_words = []
        natural_values = []
        for row in rows:
            shape = tuple(int(value) for value in row["shape"])
            payload = row["source_payload"]
            require(
                sha256_bytes(payload) == row["source_bf16_sha256"],
                f"retained source snapshot hash mismatch triplet {triplet}",
            )
            raw = np.frombuffer(payload, dtype="<u2").reshape(shape)
            natural = np.ascontiguousarray(raw.T if row["role"] == "down" else raw)
            require(natural.shape == (GROUPS_PER_MATRIX, GROUP_LENGTH), "natural source shape drift")
            values = bf16_words_to_fp32(natural)
            require(np.all(np.isfinite(values)), f"nonfinite source values in triplet {triplet}")
            natural_words.append(natural)
            natural_values.append(values)
            values64 = values.astype(np.float64)
            source_energy += float(np.sum(values64 * values64, dtype=np.float64))
        gate_words, _, _ = natural_words
        _, up, down = natural_values
        code, cosine, sine, reductions = derive_q15_klt_from_source(up, down)
        header_cosine, header_sine = parsed.header.coefficients[triplet]
        require(code == parsed.header.angle_codes[triplet], f"source-derived Q15 angle mismatch {triplet}")
        require(
            float32_bits(cosine) == float32_bits(header_cosine)
            and float32_bits(sine) == float32_bits(header_sine),
            f"source-derived FP32 KLT coefficient mismatch {triplet}",
        )
        component0, component1 = exact_fp32_klt(up, down, header_cosine, header_sine)
        component0_words = fp32_to_bf16_rne(component0)
        component1_words = fp32_to_bf16_rne(component1)
        begin = 3 * triplet * GROUPS_PER_MATRIX
        canonical_words[begin : begin + GROUPS_PER_MATRIX] = gate_words
        canonical_words[begin + GROUPS_PER_MATRIX : begin + 2 * GROUPS_PER_MATRIX] = component0_words
        canonical_words[begin + 2 * GROUPS_PER_MATRIX : begin + 3 * GROUPS_PER_MATRIX] = component1_words
        component_hashes = (
            sha256_bytes(component0_words.astype("<u2", copy=False).tobytes()),
            sha256_bytes(component1_words.astype("<u2", copy=False).tobytes()),
        )
        sealed_klt = manifest.get("klt", {}).get("rows", [])[triplet]
        require(
            int(sealed_klt.get("triplet", -1)) == triplet
            and int(sealed_klt.get("angle_code_q15_pi", 100_000)) == code
            and float32_bits(sealed_klt.get("cosine_fp32")) == float32_bits(cosine)
            and float32_bits(sealed_klt.get("sine_fp32")) == float32_bits(sine)
            and sealed_klt.get("component0_bf16_sha256") == component_hashes[0]
            and sealed_klt.get("component1_bf16_sha256") == component_hashes[1],
            f"sealed KLT staging mismatch triplet {triplet}",
        )
        triplet_rows.append(
            {
                "triplet": triplet,
                "source_derived_angle_code_q15_pi": code,
                "source_derived_coefficients_fp32": [cosine, sine],
                "component0_bf16_sha256": component_hashes[0],
                "component1_bf16_sha256": component_hashes[1],
                **reductions,
            }
        )

    if device == "cupy":
        import cupy as cp

        words_gpu = cp.asarray(canonical_words, dtype=cp.uint16)
        values_gpu = (words_gpu.astype(cp.uint32) << cp.uint32(16)).view(cp.float32).astype(cp.float64)
        group_energy = cp.asnumpy(cp.sum(values_gpu * values_gpu, axis=1, dtype=cp.float64))
        del words_gpu, values_gpu
        cp.get_default_memory_pool().free_all_blocks()
    elif device == "numpy":
        staged64 = bf16_words_to_fp32(canonical_words).astype(np.float64)
        group_energy = np.sum(staged64 * staged64, axis=1, dtype=np.float64)
    else:
        raise ValueError(device)
    ordinals = np.arange(GROUPS, dtype=np.int64)
    energy_order = np.lexsort((ordinals, group_energy))
    ranks = np.empty(GROUPS, dtype=np.int64)
    ranks[energy_order] = ordinals
    derived_labels = np.minimum(7, ranks * 8 // GROUPS).astype(np.uint8)
    require(np.array_equal(derived_labels, parsed.labels), "source-derived 3-bit label mismatch")
    require(
        np.array_equal(np.bincount(derived_labels, minlength=8), np.full(8, GROUPS // 8)),
        "source-derived labels are not equipopulous",
    )
    permutation = np.lexsort((ordinals, derived_labels))
    require(np.array_equal(np.sort(permutation), ordinals), "source-derived label permutation invalid")

    block_rows = []
    cursor = 0
    encoded_rows = summary["encoded_blocks"]
    for block, (directory, sealed_block, encoded) in enumerate(
        zip(parsed.directory, manifest["blocks"], encoded_rows, strict=True)
    ):
        groups = directory.block_values // GROUP_LENGTH
        selected_ordinals = permutation[cursor : cursor + groups]
        selected_words = np.ascontiguousarray(canonical_words[selected_ordinals], dtype="<u2")
        staging_hash = sha256_bytes(selected_words.tobytes())
        ordinal_hash = sha256_bytes(selected_ordinals.astype("<i8", copy=False).tobytes())
        require(
            int(sealed_block.get("sorted_group_begin", -1)) == cursor
            and int(sealed_block.get("sorted_group_end_exclusive", -1)) == cursor + groups
            and int(sealed_block.get("staging_bytes", -1)) == selected_words.nbytes
            and sealed_block.get("staging_sha256") == staging_hash
            and sealed_block.get("selected_group_ordinals_sha256") == ordinal_hash,
            f"source-derived staging binding mismatch block {block}",
        )
        selected_values = bf16_words_to_fp32(selected_words).reshape(-1).astype(np.float64)
        pre_rms = float(np.sqrt(np.mean(selected_values * selected_values, dtype=np.float64)))
        transformed, post_rms = forward_signed_rht_and_rms(
            selected_values, parsed.rht_seeds[block], device
        )
        transformed_hash = sha256_bytes(np.asarray(transformed, dtype="<f8").tobytes())
        expected_scale_bytes = np.asarray([post_rms], dtype="<f2").tobytes()
        actual_scale_bytes = bytes.fromhex(directory.decoder_scale_fp16_hex)
        require(
            actual_scale_bytes == expected_scale_bytes,
            f"directory FP16 scale is not source-derived post-RHT RMS at block {block}",
        )
        encoded_rms = float(encoded.get("block_rms_fp64", float("nan")))
        require(
            np.asarray([encoded_rms], dtype="<f2").tobytes() == expected_scale_bytes,
            f"summary block RMS rounds to a different FP16 scale at block {block}",
        )
        block_energy = float(np.sum(group_energy[selected_ordinals], dtype=np.float64))
        sealed_energy = float(sealed_block.get("source_energy_fp64", float("nan")))
        require(
            math.isclose(block_energy, sealed_energy, rel_tol=2e-15, abs_tol=0.0),
            f"source-derived block energy mismatch {block}",
        )
        block_rows.append(
            {
                "block_ordinal": block,
                "staging_bf16_sha256": staging_hash,
                "selected_group_ordinals_i64_sha256": ordinal_hash,
                "post_rht_f64_sha256": transformed_hash,
                "source_energy_fp64": block_energy,
                "pre_rht_rms_fp64": pre_rms,
                "post_rht_rms_fp64": post_rms,
                "rms_relative_energy_drift": (post_rms * post_rms - pre_rms * pre_rms) / (pre_rms * pre_rms),
                "expected_scale_fp16_hex": expected_scale_bytes.hex(),
                "actual_directory_scale_fp16_hex": directory.decoder_scale_fp16_hex,
                "fp16_ties_to_even_exact_match": True,
            }
        )
        cursor += groups
        del transformed, selected_values, selected_words
    require(cursor == GROUPS, "source scale audit did not consume all groups")
    staged_energy = float(np.sum(group_energy, dtype=np.float64))
    sealed_energy_audit = manifest.get("klt", {}).get("energy_audit")
    require(isinstance(sealed_energy_audit, dict), "manifest KLT energy audit missing")
    require(
        math.isclose(
            source_energy,
            float(sealed_energy_audit.get("original_source_energy_fp64", float("nan"))),
            rel_tol=2e-15,
            abs_tol=0.0,
        )
        and math.isclose(
            staged_energy,
            float(sealed_energy_audit.get("staged_klt_bf16_energy_fp64", float("nan"))),
            rel_tol=2e-15,
            abs_tol=0.0,
        ),
        "source-derived KLT energy audit mismatch",
    )
    independent_profiles, independent_allocation = allocate_profiles_independent(
        np.asarray([row["source_energy_fp64"] for row in block_rows], dtype=np.float64)
    )
    sealed_allocation = manifest.get("allocation", {})
    require(
        independent_profiles.tobytes() == parsed.profile_bytes
        and independent_allocation["profile_ids"] == sealed_allocation.get("profile_ids")
        and independent_allocation["nominal_profile_bits"]
        == int(sealed_allocation.get("nominal_profile_bits", -1))
        and independent_allocation["terminal_units"]
        == int(sealed_allocation.get("terminal_units", -1))
        and independent_allocation["nominal_budget_bits"]
        == int(sealed_allocation.get("nominal_budget_bits", -1)),
        "independently recomputed allocation DP differs from sealed/container profiles",
    )
    return {
        "all_checks_passed": True,
        "device": device,
        "source_angle_codes_match_header": True,
        "fp32_klt_and_bf16_staging_hashes_match": True,
        "source_derived_labels_match_literal_labels": True,
        "all_14_fp16_directory_scales_are_ties_to_even_post_rht_rms": True,
        "source_energy_allocation_dp_exact_match": True,
        "independent_allocation": independent_allocation,
        "original_source_energy_fp64": source_energy,
        "post_klt_bf16_staging_energy_fp64": staged_energy,
        "relative_staging_energy_drift": (staged_energy - source_energy) / source_energy,
        "label_histogram": np.bincount(derived_labels, minlength=8).astype(int).tolist(),
        "triplets": triplet_rows,
        "blocks": block_rows,
    }


def score_reference(
    parsed: ParsedContainer,
    reconstruction_path: Path,
    matrices: list[dict[str, Any]],
    source_root: Path,
) -> dict[str, Any]:
    require(len(matrices) == MATRICES, "reference matrix count mismatch")
    route_rows = validate_route(parsed.route)
    reconstruction = np.memmap(
        reconstruction_path, dtype="<f8", mode="r", shape=(GROUPS, GROUP_LENGTH)
    )
    sse = 0.0
    energy = 0.0
    rows = []
    for ordinal, (matrix, route) in enumerate(zip(matrices, route_rows)):
        identity = tensor_identity(matrix)
        match = re.fullmatch(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight",
            identity,
        )
        require(match is not None, f"unsupported reference tensor {identity}")
        assert match is not None
        role_name = match.group(3)
        role = {"gate": 0, "up": 1, "down": 2}[role_name]
        require(int(match.group(1)) == route["layer"] and int(match.group(2)) == route["expert"], "route/reference identity mismatch")
        require(role == route["role"], "route/reference role mismatch")
        shape = tuple(int(value) for value in matrix["shape"])
        expected_shape = (2048, 768) if role == 2 else (768, 2048)
        require(shape == expected_shape, "reference shape mismatch")
        path = Path(matrix["source_path"]).resolve(strict=True)
        expected_hash = matrix.get("source_bf16_sha256")
        payload = matrix["source_payload"]
        require(
            isinstance(expected_hash, str) and sha256_bytes(payload) == expected_hash,
            "retained reference source snapshot hash mismatch",
        )
        words = np.frombuffer(payload, dtype="<u2").reshape(shape)
        values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
        natural = values.T if role == 2 else values
        begin = ordinal * GROUPS_PER_MATRIX
        decoded = np.asarray(reconstruction[begin : begin + GROUPS_PER_MATRIX])
        delta = natural.astype(np.float64) - decoded
        matrix_sse = float(np.sum(delta * delta, dtype=np.float64))
        matrix_energy = float(np.sum(natural.astype(np.float64) ** 2, dtype=np.float64))
        sse += matrix_sse
        energy += matrix_energy
        rows.append(
            {
                "matrix_ordinal": ordinal,
                "tensor": identity,
                "source_path": str(path),
                "source_sha256": expected_hash,
                "sse_fp64": matrix_sse,
                "source_energy_fp64": matrix_energy,
                "relative_mse": matrix_sse / matrix_energy,
            }
        )
    return {
        "source_root": str(source_root.resolve()),
        "matrices": rows,
        "sse_sum_fp64": sse,
        "source_energy_sum_fp64": energy,
        "energy_weighted_relative_mse": sse / energy,
        "gaussian_limit_at_2p15": math.exp2(-4.3),
        "beats_gaussian_limit": sse / energy < math.exp2(-4.3),
    }


def inspection_report(parsed: ParsedContainer) -> dict[str, Any]:
    route_rows = validate_route(parsed.route)
    valid_payload_bits = sum(row.logical_bits for row in parsed.directory)
    used_payload_bytes = sum((row.logical_bits + 7) // 8 for row in parsed.directory)
    cap_floor = (43 * WEIGHTS) // 20
    block_offsets = []
    cursor = 0
    for row in parsed.directory:
        payload_bytes = (row.logical_bits + 7) // 8
        payload = parsed.reservoir[cursor : cursor + payload_bytes]
        block_offsets.append(
            {
                "block_ordinal": row.block_ordinal,
                "block_log2": row.block_log2,
                "block_values": row.block_values,
                "profile_q": row.profile_q,
                "profile_rate_bpw": 1.75 + row.profile_q / 256.0,
                "decoder_scale": row.decoder_scale,
                "decoder_scale_fp16_hex": row.decoder_scale_fp16_hex,
                "logical_bits": row.logical_bits,
                "reservoir_byte_begin": cursor,
                "reservoir_byte_end_exclusive": cursor + payload_bytes,
                "reservoir_bit_begin": 8 * cursor,
                "reservoir_logical_bit_end": 8 * cursor + row.logical_bits,
                "payload_bytes": payload_bytes,
                "payload_terminal_padding_bits": payload_bytes * 8 - row.logical_bits,
                "payload_terminal_padding_all_zero": True,
                "packed_payload_sha256": sha256_bytes(payload),
                "sc_seed_u32": parsed.sc_seeds[row.block_ordinal],
                "rht_seed_u64": parsed.rht_seeds[row.block_ordinal],
            }
        )
        cursor += payload_bytes
    return {
        "schema": "strata_v2_klt_mixed_independent_container_inspection_v1",
        "passed": True,
        "container": str(parsed.path),
        "container_sha256": parsed.raw_sha256,
        "format": {
            "magic": MAGIC.decode("ascii", errors="backslashreplace"),
            "version": FORMAT_VERSION,
            "layout_bytes": {
                "header": HEADER_BYTES,
                "route": ROUTE_BYTES,
                "labels": LABEL_BYTES,
                "directory": DIRECTORY_BYTES,
                "reservoir": RESERVOIR_BYTES,
                "total": CONTAINER_BYTES,
            },
            "directory_struct": "<BeI",
            "seed_domain_hex": SEED_DOMAIN.hex(),
            "seed_material": "domain||header||route||labels||14 profile bytes||u8 block ordinal",
        },
        "header": {
            "flags": parsed.header.flags,
            "eta_fp32": parsed.header.eta,
            "coefficient_pairs_fp32": [list(pair) for pair in parsed.header.coefficients],
            "coefficient_norm_squared": [
                float(cosine * cosine + sine * sine)
                for cosine, sine in parsed.header.coefficients
            ],
            "coefficients_bit_exact_from_q15_codes": True,
            "angle_codes_q15_pi": list(parsed.header.angle_codes),
            "control_sha256": parsed.header.control_sha256.hex(),
            "crc32": parsed.header.crc32,
            "sha256": sha256_bytes(parsed.header.raw),
        },
        "route": {"sha256": sha256_bytes(parsed.route), "rows": route_rows},
        "labels": {
            "sha256": sha256_bytes(parsed.labels_packed),
            "histogram": np.bincount(parsed.labels, minlength=8).astype(int).tolist(),
            "equipopulous": bool(np.array_equal(np.bincount(parsed.labels, minlength=8), np.full(8, GROUPS // 8))),
        },
        "directory": {
            "sha256": sha256_bytes(parsed.directory_raw),
            "profile_bytes_hex": parsed.profile_bytes.hex(),
            "rows": block_offsets,
        },
        "reservoir": {
            "sha256": sha256_bytes(parsed.reservoir),
            "capacity_bits": len(parsed.reservoir) * 8,
            "valid_payload_bits": valid_payload_bits,
            "used_payload_bytes": used_payload_bytes,
            "per_block_padding_bits": used_payload_bytes * 8 - valid_payload_bits,
            "zero_terminal_bytes": len(parsed.reservoir) - used_payload_bytes,
            "terminal_fill_all_zero": True,
            "stream_framing": "ceil(logical_bits/8) bytes per block; unused low bits zero",
        },
        "physical_rate": {
            "weights": WEIGHTS,
            "bits": CONTAINER_BYTES * 8,
            "bpw": CONTAINER_BYTES * 8 / WEIGHTS,
            "integer_cap_floor_bits": cap_floor,
            "headroom_bits_to_integer_floor": cap_floor - CONTAINER_BYTES * 8,
            "exact_gate": "bits*20 <= 43*weights",
            "passes_2p15": CONTAINER_BYTES * 8 * 20 <= 43 * WEIGHTS,
        },
        "independence": {
            "encoder_imported": False,
            "encoder_decisions_read": False,
            "encoder_probabilities_read": False,
            "external_reliability_tables_read": False,
            "capacities_and_BEC_flags_regenerated_procedurally": True,
        },
    }


def self_test() -> dict[str, Any]:
    # Header offsets, Q15 coefficient derivation, non-byte-aligned arithmetic
    # windowing, bit slicing, and terminal-fill checks are exercised without
    # touching any workspace artifact.
    route = b"".join(
        ROUTE_RECORD.pack(3 + triplet, 5 + triplet, role, 1 if role == 2 else 0, GROUPS_PER_MATRIX)
        for triplet in range(6)
        for role in range(3)
    )
    labels = bytes(LABEL_BYTES)
    codes = (-12_000, -8_000, -1, 0, 8_000, 12_000)
    coefficients = []
    for code in codes:
        theta = code * math.pi / Q15_DENOMINATOR
        coefficients.extend((float(np.float32(math.cos(theta))), float(np.float32(math.sin(theta)))))
    prefix = HEADER_PREFIX.pack(
        MAGIC, FORMAT_VERSION, HEADER_BYTES, HEADER_FLAGS, WEIGHTS,
        GROUP_LENGTH, GROUPS, BLOCKS, LEADING_N21_BLOCKS, LEADING_LOG2,
        TAIL_LOG2, ETA,
    )
    first124 = (
        prefix
        + HEADER_COEFFICIENTS.pack(*coefficients)
        + HEADER_ANGLES.pack(*codes)
        + hashlib.sha256(route + labels).digest()
    )
    header_raw = first124 + struct.pack("<I", zlib.crc32(first124) & 0xFFFFFFFF)
    header = parse_header(header_raw, route, labels)
    validate_route(route)
    raw = bytes((0b10110110, 0b01101001, 0b11100000))
    require(packed_bit_slice(raw, 3, 13) == bytes((0b10110011, 0b01001000)), "bit-slice self-test failed")
    require(bit_range_has_one(raw, 3, 16), "bit-range positive self-test failed")
    require(not bit_range_has_one(bytes(3), 3, 16), "bit-range zero self-test failed")
    unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    for begin in range(unpacked.size + 1):
        for end in range(begin, unpacked.size + 1):
            require(
                bit_range_has_one(raw, begin, end) == bool(np.any(unpacked[begin:end])),
                f"bit-range exhaustive self-test failed [{begin},{end})",
            )
    rng = np.random.default_rng(20260831)
    symbols = rng.integers(0, 2, size=4096, dtype=np.uint8)
    frequencies = rng.integers(1, 65_536, size=4096, dtype=np.uint16)
    encoded, logical = arithmetic_encode_binary(symbols, frequencies)
    arithmetic = ArithmeticBinaryDecoder(encoded, 0, logical)
    decoded = np.asarray([arithmetic.decode(int(freq)) for freq in frequencies], dtype=np.uint8)
    require(np.array_equal(decoded, symbols), "arithmetic self-test decode mismatch")
    encoded2, logical2 = arithmetic_encode_binary(decoded, frequencies)
    require(encoded2 == encoded and logical2 == logical, "arithmetic canonical self-test mismatch")
    require(
        not bit_range_has_one(encoded, logical, len(encoded) * 8),
        "arithmetic final-byte padding self-test failed",
    )
    sealed_probe: dict[str, Any] = {"schema": "probe", "status": "frozen"}
    sealed_probe["lock_sha256"] = sha256_bytes(canonical_json_bytes(sealed_probe))
    verify_internal_lock(sealed_probe, "self-test seal")
    tampered_probe = dict(sealed_probe)
    tampered_probe["status"] = "tampered"
    try:
        verify_internal_lock(tampered_probe, "self-test tampered seal")
    except AssertionError:
        pass
    else:
        raise AssertionError("internal-seal tamper self-test was accepted")
    sc, rht = derive_seed_pair(header_raw, route, labels, bytes(range(14)), 13)
    return {
        "passed": True,
        "header_bytes": len(header.raw),
        "header_crc32": header.crc32,
        "route_bytes": len(route),
        "label_bytes": len(labels),
        "seed_domain_hex": SEED_DOMAIN.hex(),
        "sample_sc_seed_u32": sc,
        "sample_rht_seed_u64": rht,
        "bit_range_cases": (unpacked.size + 1) * (unpacked.size + 2) // 2,
        "canonical_arithmetic_symbols": int(symbols.size),
        "internal_seal_tamper_rejected": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--inverse-device", choices=("cupy", "numpy"), default="cupy")
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--codec-freeze", type=Path)
    parser.add_argument("--format-freeze", type=Path)
    parser.add_argument("--preencoding-manifest", type=Path)
    parser.add_argument("--allocation-lock", type=Path)
    parser.add_argument("--one-shot-intent", type=Path)
    parser.add_argument("--one-shot-summary", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--protocol-mode", choices=("blind", "development"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if args.container is None or args.output_dir is None:
        parser.error("--container and --output-dir are required unless --self-test is used")
    lineage_names = (
        "selection_lock",
        "source_lock",
        "codec_freeze",
        "format_freeze",
        "preencoding_manifest",
        "allocation_lock",
        "one_shot_intent",
        "one_shot_summary",
    )
    supplied_lineage = [getattr(args, name) is not None for name in lineage_names]
    if any(supplied_lineage) and not all(supplied_lineage):
        parser.error(
            "a source-domain claim requires all of --selection-lock, --source-lock, "
            "--codec-freeze, --format-freeze, --preencoding-manifest, "
            "--allocation-lock, --one-shot-intent, and --one-shot-summary"
        )
    if all(supplied_lineage) and args.protocol_mode is None:
        parser.error("--protocol-mode blind|development is required with source lineage")
    if not any(supplied_lineage) and args.protocol_mode is not None:
        parser.error("--protocol-mode is valid only with the complete lineage argument set")
    if args.source_root is not None and not all(supplied_lineage):
        parser.error("--source-root is valid only with the complete lineage argument set")
    require(args.workers >= 1, "workers must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    started = time.perf_counter()
    parsed = parse_container(args.container)
    lineage = None
    scale_audit = None
    if all(supplied_lineage):
        lineage = validate_source_lineage(
            parsed,
            args.protocol_mode,
            args.selection_lock,
            args.source_lock,
            args.codec_freeze,
            args.format_freeze,
            args.preencoding_manifest,
            args.allocation_lock,
            args.one_shot_intent,
            args.one_shot_summary,
            args.source_root,
        )
        scale_audit = audit_source_staging_and_scales(parsed, lineage, args.inverse_device)
    inspection = inspection_report(parsed)
    output_dir.mkdir(parents=True)
    block_dir = output_dir / "decoded_sorted_blocks"
    block_dir.mkdir()
    inspection_path = output_dir / "inspection.json"
    inspection_path.write_text(json.dumps(inspection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.inspect_only:
        print(
            json.dumps(
                {
                    "passed": True,
                    "inspection": str(inspection_path),
                    "inspection_sha256": sha256_file(inspection_path),
                    "container_sha256": parsed.raw_sha256,
                    "physical_bpw": inspection["physical_rate"]["bpw"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    jobs = [
        (
            str(parsed.path),
            block,
            str(block_dir / f"block_{block:02d}_sorted_post_inverse_rht.f64.bin"),
            args.inverse_device,
        )
        for block in range(BLOCKS)
    ]
    decoded_rows = []
    if args.workers == 1:
        for job in jobs:
            row = decode_one_block(*job)
            decoded_rows.append(row)
            print(
                f"decoded {len(decoded_rows)}/{BLOCKS} block={row['block_ordinal']} "
                f"N=2^{row['block_log2']} q={row['profile_q']} bits={row['logical_bits']}",
                flush=True,
            )
    else:
        # Spawned processes independently reopen and validate the literal
        # container; no in-memory encoder state can leak into a worker.
        # Source-derived scale validation initializes CUDA in the parent.
        # Spawn avoids inheriting that live CUDA context into forked workers.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {pool.submit(decode_one_block, *job): job[1] for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                decoded_rows.append(row)
                print(
                    f"decoded {len(decoded_rows)}/{BLOCKS} block={row['block_ordinal']} "
                    f"N=2^{row['block_log2']} q={row['profile_q']} bits={row['logical_bits']}",
                    flush=True,
                )
    decoded_rows.sort(key=lambda row: int(row["block_ordinal"]))
    reconstruction = assemble_reconstructions(parsed, decoded_rows, output_dir)
    score = None
    if lineage is not None:
        score = score_reference(
            parsed,
            Path(reconstruction["original_domain_canonical_path"]),
            lineage["matrices"],
            lineage["source_root"],
        )
        require(
            math.isclose(
                score["source_energy_sum_fp64"],
                scale_audit["original_source_energy_fp64"],
                rel_tol=2e-15,
                abs_tol=0.0,
            ),
            "source score and source-staging audit energy disagree",
        )

    primary_conditions = {
        "physical_rate_at_most_2p15": bool(inspection["physical_rate"]["passes_2p15"]),
        "complete_source_lineage_present": lineage is not None,
        "blind_protocol_mode": lineage is not None
        and lineage["report"]["protocol_mode"] == "blind",
        "source_staging_label_scale_and_dp_audit_passed": scale_audit is not None
        and bool(scale_audit["all_checks_passed"]),
        "source_domain_mse_below_gaussian_limit": score is not None
        and bool(score["beats_gaussian_limit"]),
    }
    primary_gate_passed = all(primary_conditions.values())
    report = {
        "schema": "strata_v2_klt_mixed_independent_decode_audit_v1",
        "passed": primary_gate_passed,
        "audit_execution_passed": True,
        "primary_claim_gate": {
            "passed": primary_gate_passed,
            "rule": (
                "physical_bpw<=2.15 AND complete blind lineage/source-derived metadata audit "
                "AND pooled source-domain MSE<2^-4.3"
            ),
            "conditions": primary_conditions,
        },
        "container_inspection": inspection,
        "decode": {
            "blocks": decoded_rows,
            "all_blocks_decoded": len(decoded_rows) == BLOCKS,
            "reconstruction": reconstruction,
            "inverse_rht_device": args.inverse_device,
        },
        "source_lineage": None if lineage is None else lineage["report"],
        "source_staging_and_scale_audit": scale_audit,
        "source_score": score,
        "claim_boundary": (
            "Physical parsing and decode are independent and complete. Source-domain MSE "
            "is present only when every pre-source freeze, source-finalization, allocation, "
            "one-shot, source-derived staging, label, and FP16-scale binding passes."
        ),
        "independence": {
            "encoder_imported": False,
            "encoder_decisions_read": False,
            "encoder_probabilities_read": False,
            "external_reliability_tables_read": False,
            "source_files_auto_discovered": False,
            "procedural_q31_bec_regenerated": True,
            "causal_probabilities_regenerated_from_decoded_lower_levels": True,
            "canonical_arithmetic_payloads_reencoded_and_byte_compared": True,
            "source_derived_fp16_scales_checked": scale_audit is not None,
        },
        "runtime_seconds": time.perf_counter() - started,
        "workers": args.workers,
    }
    report_path = output_dir / "independent_decode_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": primary_gate_passed,
                "audit_execution_passed": True,
                "primary_claim_gate_passed": primary_gate_passed,
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "container_sha256": parsed.raw_sha256,
                "physical_bpw": inspection["physical_rate"]["bpw"],
                "relative_mse": None if score is None else score["energy_weighted_relative_mse"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
