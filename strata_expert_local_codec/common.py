#!/usr/bin/env python3
"""Deterministic constants and primitives for the expert-affine checkpoint.

This is a format fork.  It deliberately does not change the frozen v2 codec.
The payload primitive remains the v2 N=2^20/2^21 POLARIS encoder, while block
ownership is aligned with the MoE routing unit.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np


MAGIC = b"PLRLOC3\0"
FORMAT_VERSION = 1
FLAGS = 0x000001FF
WEIGHTS = 28_311_552
EXPERTS = 6
MATRICES = 18
GROUP_VALUES = 2_048
GROUPS_PER_MATRIX = 768
GROUPS_PER_EXPERT = 2_304
GROUPS = 13_824
BLOCK_LOG2 = (21, 21) * EXPERTS + (20,) * 3
BLOCK_SIZES = tuple(1 << value for value in BLOCK_LOG2)
BLOCK_GROUPS = tuple(value // GROUP_VALUES for value in BLOCK_SIZES)
BLOCKS = len(BLOCK_LOG2)
PRIVATE_BLOCKS = 12
HEADER_BYTES = 128
ROUTE_BYTES = 144
LABEL_BYTES = 5_184
DIRECTORY_RECORD_BYTES = 7
DIRECTORY_BYTES = BLOCKS * DIRECTORY_RECORD_BYTES
PHYSICAL_BYTES = 8_847_360
PHYSICAL_BITS = PHYSICAL_BYTES * 8
RESERVOIR_BYTES = PHYSICAL_BYTES - HEADER_BYTES - ROUTE_BYTES - LABEL_BYTES - DIRECTORY_BYTES
GLOBAL_RESERVE_BITS = 65_536
NOMINAL_PROFILE_BUDGET_BITS = RESERVOIR_BYTES * 8 - GLOBAL_RESERVE_BITS
PROFILE_BASE = 1.75
PROFILE_STEP = 1.0 / 256.0
FINITE_FACTOR = {20: 1.0124498003545317, 21: 1.0107341453912242}
SEED_DOMAIN = b"POLARIS-STRATA-EXPERT-AFFINE-N20N21-v1\0"
GAUSSIAN_GAIN_TARGET = 0.20
CURRENT_SOURCE_MSE = 0.04985939119332436


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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
    labels = (4 * rows[:, 0] + 2 * rows[:, 1] + rows[:, 2]).astype(np.uint8)
    if np.bincount(labels, minlength=8).tolist() != [1728] * 8:
        raise ValueError("global label histogram is not equipopulous")
    return labels


def parse_route(payload: bytes) -> list[dict[str, int | str]]:
    if len(payload) != ROUTE_BYTES:
        raise ValueError("route byte length mismatch")
    roles = {0: "gate", 1: "up", 2: "down"}
    axes = {0: "row", 1: "column"}
    rows: list[dict[str, int | str]] = []
    for ordinal in range(MATRICES):
        layer, expert, role_id, axis_id, groups = struct.unpack_from(
            ">HHBBH", payload, 8 * ordinal
        )
        if role_id not in roles or axis_id not in axes or groups != GROUPS_PER_MATRIX:
            raise ValueError(f"invalid route record {ordinal}")
        expected_axis_id = 1 if role_id == 2 else 0
        if axis_id != expected_axis_id:
            raise ValueError(f"role/axis mismatch route record {ordinal}")
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
    identities: set[tuple[int, int]] = set()
    for expert_ordinal in range(EXPERTS):
        gate, up, down = rows[3 * expert_ordinal : 3 * expert_ordinal + 3]
        if [gate["role"], up["role"], down["role"]] != ["gate", "up", "down"]:
            raise ValueError(f"role order mismatch expert {expert_ordinal}")
        if not (gate["layer"] == up["layer"] == down["layer"]):
            raise ValueError(f"layer mismatch expert {expert_ordinal}")
        if not (gate["expert"] == up["expert"] == down["expert"]):
            raise ValueError(f"expert id mismatch expert {expert_ordinal}")
        identity = (int(gate["layer"]), int(gate["expert"]))
        if identity in identities:
            raise ValueError(f"duplicate layer/expert identity {identity}")
        identities.add(identity)
    return rows


def expected_block_group_ordinals(labels: np.ndarray) -> list[np.ndarray]:
    """Return twelve private chunks followed by three paired tail chunks."""

    labels = np.asarray(labels, dtype=np.uint8)
    if labels.shape != (GROUPS,):
        raise ValueError("label geometry mismatch")
    private: list[np.ndarray] = []
    tails: list[np.ndarray] = []
    for expert_ordinal in range(EXPERTS):
        begin = expert_ordinal * GROUPS_PER_EXPERT
        end = begin + GROUPS_PER_EXPERT
        ordinals = np.arange(begin, end, dtype=np.int64)
        order = ordinals[np.lexsort((ordinals, labels[begin:end]))]
        private.extend((order[:1024], order[1024:2048]))
        tails.append(order[2048:])
    blocks = private + [
        np.concatenate((tails[0], tails[1])),
        np.concatenate((tails[2], tails[3])),
        np.concatenate((tails[4], tails[5])),
    ]
    if [len(row) for row in blocks] != list(BLOCK_GROUPS):
        raise AssertionError("expert-affine block geometry changed")
    coverage = np.concatenate(blocks)
    if not np.array_equal(np.sort(coverage), np.arange(GROUPS, dtype=np.int64)):
        raise AssertionError("expert-affine mapping is not exact coverage")
    return blocks


def block_owner_experts(ordinal: int) -> list[int]:
    if not 0 <= ordinal < BLOCKS:
        raise ValueError("block ordinal outside format")
    if ordinal < PRIVATE_BLOCKS:
        return [ordinal // 2]
    pair = ordinal - PRIVATE_BLOCKS
    return [2 * pair, 2 * pair + 1]


def expert_required_blocks(expert_ordinal: int) -> tuple[int, int, int]:
    if not 0 <= expert_ordinal < EXPERTS:
        raise ValueError("expert ordinal outside format")
    return 2 * expert_ordinal, 2 * expert_ordinal + 1, PRIVATE_BLOCKS + expert_ordinal // 2


def build_header(
    coefficients: list[tuple[np.float32, np.float32]],
    angle_codes: list[int],
    route: bytes,
    labels: bytes,
) -> bytes:
    if len(coefficients) != EXPERTS or len(angle_codes) != EXPERTS:
        raise ValueError("header KLT geometry mismatch")
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
        BLOCKS,
        PRIVATE_BLOCKS,
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
        MAGIC,
        FORMAT_VERSION,
        HEADER_BYTES,
        FLAGS,
        WEIGHTS,
        GROUP_VALUES,
        GROUPS,
        BLOCKS,
        PRIVATE_BLOCKS,
        21,
        20,
        0.25,
    )
    if fields != expected:
        raise ValueError(f"header constants mismatch: {fields!r}")
    coefficients = struct.unpack_from("<12f", header, 32)
    codes = struct.unpack_from("<6h", header, 80)
    for expert_ordinal, code in enumerate(codes):
        if not -16384 <= code <= 16384:
            raise ValueError("KLT code outside Q15-over-pi range")
        theta = code * math.pi / 32768.0
        expected_pair = (np.float32(math.cos(theta)), np.float32(math.sin(theta)))
        for component, expected_value in enumerate(expected_pair):
            actual = struct.pack("<f", coefficients[2 * expert_ordinal + component])
            expected_bytes = struct.pack("<f", float(expected_value))
            if actual != expected_bytes:
                raise ValueError("stored KLT coefficient is not regenerated by its code")
    if header[92:124] != hashlib.sha256(route + labels).digest():
        raise ValueError("route/label binding mismatch")
    crc, = struct.unpack_from("<I", header, 124)
    if crc != zlib.crc32(header[:124]) & 0xFFFFFFFF:
        raise ValueError("header CRC mismatch")


def derive_seeds(
    header: bytes, route: bytes, labels: bytes, profiles: bytes, ordinal: int
) -> tuple[int, int, str]:
    if len(profiles) != BLOCKS or not 0 <= ordinal < BLOCKS:
        raise ValueError("seed input geometry mismatch")
    digest = hashlib.sha256(
        SEED_DOMAIN + header + route + labels + profiles + bytes((ordinal,))
    ).digest()
    return int.from_bytes(digest[:4], "big") or 1, int.from_bytes(digest[4:12], "big"), digest.hex()


def allocate_profiles(block_energy: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    energy = np.asarray(block_energy, dtype=np.float64)
    if energy.shape != (BLOCKS,) or np.any(~np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError("allocation requires fifteen positive finite energies")
    base_bits = int(WEIGHTS * PROFILE_BASE)
    unit_bits = (1 << 20) // 256
    budget_units = (NOMINAL_PROFILE_BUDGET_BITS - base_bits) // unit_bits
    dp = np.full(budget_units + 1, np.inf, dtype=np.float64)
    dp[0] = 0.0
    choices: list[np.ndarray] = []
    for block_energy_value, logn in zip(energy, BLOCK_LOG2, strict=True):
        weight = 1 << (logn - 20)
        factor = FINITE_FACTOR[logn]
        new = np.full_like(dp, np.inf)
        picked = np.full(dp.size, -1, dtype=np.int16)
        for q in range(256):
            cost = weight * q
            if cost > budget_units:
                break
            distortion = factor * block_energy_value * math.pow(
                2.0, -2.0 * (PROFILE_BASE + q / 256.0)
            )
            candidate = dp[: dp.size - cost] + distortion
            target = new[cost:]
            improve = candidate < target
            target[improve] = candidate[improve]
            picked[cost:][improve] = q
        dp = new
        choices.append(picked)
    terminal = int(np.argmin(dp))
    profiles = np.empty(BLOCKS, dtype=np.uint8)
    cursor = terminal
    for ordinal in range(BLOCKS - 1, -1, -1):
        chosen = int(choices[ordinal][cursor])
        if chosen < 0:
            raise AssertionError("allocation DP backtrack failed")
        profiles[ordinal] = chosen
        cursor -= (1 << (BLOCK_LOG2[ordinal] - 20)) * chosen
    if cursor != 0:
        raise AssertionError("allocation DP did not return to origin")
    nominal_bits = base_bits + terminal * unit_bits
    rates = PROFILE_BASE + profiles.astype(np.float64) / 256.0
    objective = float(
        sum(
            FINITE_FACTOR[logn] * value * math.pow(2.0, -2.0 * rate)
            for logn, value, rate in zip(BLOCK_LOG2, energy, rates, strict=True)
        )
    )
    return profiles, {
        "profile_ids": profiles.astype(int).tolist(),
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
    }


def gaussian_limit(rate_bpw: float) -> float:
    return math.pow(2.0, -2.0 * rate_bpw)


def gaussian_gain(mse: float, rate_bpw: float) -> float:
    return 1.0 - mse / gaussian_limit(rate_bpw)


def validate_constants() -> None:
    if BLOCKS != 15 or BLOCK_LOG2 != (21, 21) * 6 + (20, 20, 20):
        raise AssertionError("block constants changed")
    if sum(BLOCK_SIZES) != WEIGHTS or sum(BLOCK_GROUPS) != GROUPS:
        raise AssertionError("block geometry does not cover the panel")
    if HEADER_BYTES + ROUTE_BYTES + LABEL_BYTES + DIRECTORY_BYTES + RESERVOIR_BYTES != PHYSICAL_BYTES:
        raise AssertionError("physical ledger does not sum")
    if PHYSICAL_BITS != 70_778_880 or PHYSICAL_BITS / WEIGHTS != 2.5:
        raise AssertionError("physical rate is not exactly 2.5 bpw")
    if RESERVOIR_BYTES * 8 - GLOBAL_RESERVE_BITS != NOMINAL_PROFILE_BUDGET_BITS:
        raise AssertionError("reserve ledger mismatch")


validate_constants()
