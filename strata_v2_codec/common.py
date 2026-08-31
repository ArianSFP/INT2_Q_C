#!/usr/bin/env python3
"""Shared deterministic primitives for the STRATA-XKLT-SC v2 candidate."""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np


MAGIC = b"PLRKLT2\0"
FORMAT_VERSION = 1
FLAGS = 0x0000007F
WEIGHTS = 28_311_552
GROUP_VALUES = 2_048
GROUPS = 13_824
BLOCK_LOG2 = (21,) * 13 + (20,)
BLOCK_SIZES = tuple(1 << value for value in BLOCK_LOG2)
BLOCK_GROUPS = tuple(value // GROUP_VALUES for value in BLOCK_SIZES)
HEADER_BYTES = 128
ROUTE_BYTES = 144
LABEL_BYTES = 5_184
DIRECTORY_RECORD_BYTES = 7
DIRECTORY_BYTES = 98
PHYSICAL_BYTES = 7_608_729
RESERVOIR_BYTES = 7_603_175
PHYSICAL_BITS = 60_869_832
INTEGER_CAP_BITS = 60_869_836
NOMINAL_PROFILE_BUDGET_BITS = 60_759_864
GLOBAL_RESERVE_BITS = 65_536
PROFILE_BASE = 1.75
PROFILE_STEP = 1.0 / 256.0
FINITE_FACTOR = {20: 1.0124498003545317, 21: 1.0107341453912242}
SEED_DOMAIN = b"POLARIS-STRATA-V2-KLT-MIXED-SEED-v1\0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_tree_receipt(
    module_name: str, distribution_name: str
) -> dict[str, Any]:
    """Hash every installed wheel file and verify its RECORD declaration.

    Hashing only ``__init__.py`` or the textual RECORD would not detect a
    replaced extension module.  This receipt binds normalized RECORD paths,
    actual byte counts, and actual SHA-256 digests for the complete installed
    distribution tree.  Entries with a wheel-declared hash/size are checked
    before the aggregate digest is returned.
    """
    module = importlib.import_module(module_name)
    origin = Path(str(module.__file__)).resolve(strict=True)
    distribution = importlib.metadata.distribution(distribution_name)
    files = sorted(distribution.files or (), key=lambda row: str(row).replace("\\", "/"))
    if not files:
        raise RuntimeError(f"distribution has no RECORD file list: {distribution_name}")
    record_rows = [row for row in files if row.name == "RECORD"]
    if len(record_rows) != 1:
        raise RuntimeError(f"distribution RECORD is not unique: {distribution_name}")
    aggregate = hashlib.sha256()
    total_bytes = 0
    origin_is_recorded = False
    for row in files:
        relative = str(row).replace("\\", "/")
        path = Path(distribution.locate_file(row)).resolve(strict=True)
        if not path.is_file():
            raise RuntimeError(f"distribution entry is not a file: {path}")
        origin_is_recorded = origin_is_recorded or path == origin
        size = path.stat().st_size
        digest_hex = sha256_file(path)
        declared_hash = row.hash
        if declared_hash is not None:
            if declared_hash.mode != "sha256":
                raise RuntimeError(
                    f"unsupported RECORD hash mode {declared_hash.mode}: {relative}"
                )
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest_hex)).rstrip(b"=").decode()
            if encoded != declared_hash.value:
                raise RuntimeError(f"wheel RECORD hash mismatch: {relative}")
        if row.size is not None and int(row.size) != size:
            raise RuntimeError(f"wheel RECORD size mismatch: {relative}")
        relative_bytes = relative.encode("utf-8")
        aggregate.update(len(relative_bytes).to_bytes(4, "big"))
        aggregate.update(relative_bytes)
        aggregate.update(size.to_bytes(8, "big"))
        aggregate.update(bytes.fromhex(digest_hex))
        total_bytes += size
    if not origin_is_recorded:
        raise RuntimeError(
            f"import origin is not recorded by {distribution_name}: {origin}"
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


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sealed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    clean = dict(result)
    clean.pop("lock_sha256", None)
    result["lock_sha256"] = hashlib.sha256(canonical_bytes(clean)).hexdigest()
    return result


def verify_internal_seal(value: dict[str, Any]) -> bool:
    expected = value.get("lock_sha256")
    if not isinstance(expected, str):
        return False
    clean = dict(value)
    clean.pop("lock_sha256", None)
    return hashlib.sha256(canonical_bytes(clean)).hexdigest() == expected


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def bf16_to_fp32(words: np.ndarray) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint16)
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def fp32_to_bf16_rne(values: np.ndarray) -> np.ndarray:
    words = np.asarray(values, dtype="<f4").view(np.uint32)
    rounded = words + np.uint32(0x7FFF) + ((words >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2")


def pack_labels(labels: np.ndarray) -> bytes:
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.shape != (GROUPS,) or np.any(labels > 7):
        raise ValueError("labels must be 13,824 uint3 values")
    bits = ((labels[:, None] >> np.arange(2, -1, -1, dtype=np.uint8)) & 1).reshape(-1)
    payload = np.packbits(bits, bitorder="big").tobytes()
    if len(payload) != LABEL_BYTES:
        raise AssertionError("label packing size changed")
    return payload


def unpack_labels(payload: bytes) -> np.ndarray:
    if len(payload) != LABEL_BYTES:
        raise ValueError("wrong label payload size")
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    rows = bits.reshape(GROUPS, 3)
    return (4 * rows[:, 0] + 2 * rows[:, 1] + rows[:, 2]).astype(np.uint8)


def parse_route(payload: bytes) -> list[dict[str, int | str]]:
    if len(payload) != ROUTE_BYTES:
        raise ValueError(f"route must be exactly {ROUTE_BYTES} bytes")
    roles = {0: "gate", 1: "up", 2: "down"}
    axes = {0: "row", 1: "column"}
    rows = []
    for ordinal in range(18):
        layer, expert, role_id, axis_id, groups = struct.unpack_from(
            ">HHBBH", payload, 8 * ordinal
        )
        if role_id not in roles or axis_id not in axes:
            raise ValueError(f"invalid route enum in record {ordinal}")
        rows.append(
            {
                "matrix_ordinal": ordinal,
                "layer": layer,
                "expert": expert,
                "role": roles[role_id],
                "axis": axes[axis_id],
                "groups": groups,
            }
        )
    for triplet in range(6):
        gate, up, down = rows[3 * triplet : 3 * triplet + 3]
        if [gate["role"], up["role"], down["role"]] != ["gate", "up", "down"]:
            raise ValueError(f"triplet {triplet} role order is not gate/up/down")
        if not (gate["layer"] == up["layer"] == down["layer"]):
            raise ValueError(f"triplet {triplet} layer mismatch")
        if not (gate["expert"] == up["expert"] == down["expert"]):
            raise ValueError(f"triplet {triplet} expert mismatch")
        if [gate["axis"], up["axis"], down["axis"]] != ["row", "row", "column"]:
            raise ValueError(f"triplet {triplet} axis mismatch")
        if [gate["groups"], up["groups"], down["groups"]] != [768, 768, 768]:
            raise ValueError(f"triplet {triplet} group geometry mismatch")
    return rows


def derive_klt(a: float, b: float, cross: float) -> tuple[int, np.float32, np.float32]:
    theta = 0.5 * math.atan2(2.0 * cross, a - b)
    code = int(np.clip(np.rint(theta / math.pi * 32768.0), -16384, 16384))
    decoded = code * math.pi / 32768.0
    return code, np.float32(math.cos(decoded)), np.float32(math.sin(decoded))


def build_header(
    coefficients: list[tuple[np.float32, np.float32]],
    angle_codes: list[int],
    route: bytes,
    labels: bytes,
) -> bytes:
    if len(coefficients) != 6 or len(angle_codes) != 6:
        raise ValueError("header requires six KLT coefficient pairs")
    if len(route) != ROUTE_BYTES or len(labels) != LABEL_BYTES:
        raise ValueError("header asset geometry mismatch")
    header = bytearray(HEADER_BYTES)
    struct.pack_into(
        "<8sHHIIHHBBBBf",
        header,
        0,
        MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        FLAGS,
        WEIGHTS,
        GROUP_VALUES,
        GROUPS,
        14,
        13,
        21,
        20,
        0.25,
    )
    struct.pack_into(
        "<12f", header, 32, *(float(value) for pair in coefficients for value in pair)
    )
    struct.pack_into("<6h", header, 80, *angle_codes)
    header[92:124] = hashlib.sha256(route + labels).digest()
    struct.pack_into("<I", header, 124, zlib.crc32(header[:124]) & 0xFFFFFFFF)
    result = bytes(header)
    validate_header(result, route, labels)
    return result


def validate_header(header: bytes, route: bytes, labels: bytes) -> None:
    if len(header) != HEADER_BYTES:
        raise ValueError("header length mismatch")
    fields = struct.unpack_from("<8sHHIIHHBBBBf", header, 0)
    expected = (
        MAGIC, FORMAT_VERSION, HEADER_BYTES, FLAGS, WEIGHTS, GROUP_VALUES,
        GROUPS, 14, 13, 21, 20, 0.25,
    )
    if fields != expected:
        raise ValueError(f"header constants mismatch: {fields!r}")
    coefficients = struct.unpack_from("<12f", header, 32)
    angle_codes = struct.unpack_from("<6h", header, 80)
    for triplet, code in enumerate(angle_codes):
        if not -16384 <= code <= 16384:
            raise ValueError(f"KLT angle code {triplet} is outside Q15-over-pi range")
        decoded = code * math.pi / 32768.0
        expected_pair = (
            np.float32(math.cos(decoded)),
            np.float32(math.sin(decoded)),
        )
        for component, expected_value in enumerate(expected_pair):
            index = 2 * triplet + component
            actual_bytes = struct.pack("<f", coefficients[index])
            expected_bytes = struct.pack("<f", float(expected_value))
            if actual_bytes != expected_bytes:
                name = "cosine" if component == 0 else "sine"
                raise ValueError(
                    f"KLT {name} {triplet} is not the bit-exact value derived "
                    "from its angle code"
                )
    if header[92:124] != hashlib.sha256(route + labels).digest():
        raise ValueError("header route/label binding mismatch")
    crc, = struct.unpack_from("<I", header, 124)
    if crc != (zlib.crc32(header[:124]) & 0xFFFFFFFF):
        raise ValueError("header CRC mismatch")


def derive_seeds(
    header: bytes, route: bytes, labels: bytes, profiles: bytes, ordinal: int
) -> tuple[int, int, str]:
    if len(profiles) != 14 or not 0 <= ordinal < 14:
        raise ValueError("seed input geometry mismatch")
    digest = hashlib.sha256(
        SEED_DOMAIN + header + route + labels + profiles + bytes([ordinal])
    ).digest()
    sc = int.from_bytes(digest[:4], "big") or 1
    rht = int.from_bytes(digest[4:12], "big")
    return sc, rht, digest.hex()


def allocate_profiles(block_energy: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    energy = np.asarray(block_energy, dtype=np.float64)
    if energy.shape != (14,) or np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("allocation requires fourteen positive finite energies")
    base_bits = int(sum(BLOCK_SIZES) * PROFILE_BASE)
    unit_bits = (1 << 20) // 256
    budget_units = (NOMINAL_PROFILE_BUDGET_BITS - base_bits) // unit_bits
    dp = np.full(budget_units + 1, np.inf, dtype=np.float64)
    dp[0] = 0.0
    choices: list[np.ndarray] = []
    for index, (e, logn) in enumerate(zip(energy, BLOCK_LOG2)):
        weight = 1 << (logn - 20)
        factor = FINITE_FACTOR[logn]
        new = np.full_like(dp, np.inf)
        picked = np.full(dp.size, -1, dtype=np.int16)
        for q in range(256):
            cost = weight * q
            if cost > budget_units:
                break
            distortion = factor * e * math.pow(2.0, -2.0 * (PROFILE_BASE + q / 256.0))
            candidate = dp[: dp.size - cost] + distortion
            target = new[cost:]
            improve = candidate < target
            target[improve] = candidate[improve]
            picked[cost:][improve] = q
        if not np.any(np.isfinite(new)):
            raise AssertionError(f"allocation DP lost feasibility at block {index}")
        dp = new
        choices.append(picked)
    terminal = int(np.argmin(dp))
    q = np.empty(14, dtype=np.uint8)
    cursor = terminal
    for index in range(13, -1, -1):
        chosen = int(choices[index][cursor])
        if chosen < 0:
            raise AssertionError("allocation DP backtrack failed")
        q[index] = chosen
        cursor -= (1 << (BLOCK_LOG2[index] - 20)) * chosen
    if cursor != 0:
        raise AssertionError("allocation DP did not return to origin")
    nominal_bits = base_bits + terminal * unit_bits
    rates = PROFILE_BASE + q.astype(np.float64) / 256.0
    objective = float(
        sum(
            FINITE_FACTOR[logn] * e * math.pow(2.0, -2.0 * rate)
            for logn, e, rate in zip(BLOCK_LOG2, energy, rates)
        )
    )
    return q, {
        "profile_ids": q.astype(int).tolist(),
        "rates_bpw": rates.tolist(),
        "base_bits": base_bits,
        "unit_bits": unit_bits,
        "budget_units": budget_units,
        "terminal_units": terminal,
        "nominal_profile_bits": nominal_bits,
        "nominal_budget_bits": NOMINAL_PROFILE_BUDGET_BITS,
        "nominal_unused_bits": NOMINAL_PROFILE_BUDGET_BITS - nominal_bits,
        "finite_factor_objective_sse": objective,
        "projected_relative_mse": objective / float(energy.sum(dtype=np.float64)),
        "tie_rule": "strict-less updates; ascending q; lowest terminal index",
    }


def validate_global_constants() -> None:
    if sum(BLOCK_SIZES) != WEIGHTS or sum(BLOCK_GROUPS) != GROUPS:
        raise AssertionError("mixed geometry constants do not cover the panel")
    if HEADER_BYTES + ROUTE_BYTES + LABEL_BYTES + DIRECTORY_BYTES + RESERVOIR_BYTES != PHYSICAL_BYTES:
        raise AssertionError("physical layout constants do not sum")
    if PHYSICAL_BYTES * 8 != PHYSICAL_BITS or PHYSICAL_BITS > INTEGER_CAP_BITS:
        raise AssertionError("physical rate constants violate the integer gate")
    if RESERVOIR_BYTES * 8 - GLOBAL_RESERVE_BITS != NOMINAL_PROFILE_BUDGET_BITS:
        raise AssertionError("profile budget does not equal reservoir minus reserve")


validate_global_constants()
