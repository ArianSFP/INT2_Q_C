#!/usr/bin/env python3
"""Standard-library constants and strict data primitives for N18 v3."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


DESIGN_SCHEMA = "tactic_actual_coarse_n18_source_design_v3"
INVENTORY_SCHEMA = "tactic_actual_coarse_n18_external_inventory_v3"
RUNTIME_LOCK_SCHEMA = "tactic_actual_coarse_n18_runtime_lock_v3"
DISPATCH_SCHEMA = "tactic_actual_coarse_n18_dispatch_assertion_v3"
DEPENDENCY_SCHEMA = "tactic_actual_coarse_n18_dependency_graph_v3"
TELEMETRY_SCHEMA = "tactic_actual_coarse_n18_telemetry_v3"

N18 = 1 << 18
MICRO = 4096
MICROS_PER_N18 = 64
COARSE_BYTES_PER_MICRO = 1228
FINE_BYTES_PER_MICRO = 48
METADATA_BYTES_PER_MICRO = 4
TOTAL_BYTES_PER_MICRO = 1280
N18_COARSE_RESERVOIR_BYTES = 78_592
PAGE_BYTES = 4096
MIN_EXPLICIT_FRAME_BYTES = 8191

MAX_EXPERTS = 256
MAX_DIMENSION = 1 << 24
MAX_MATRIX_VALUES = 1 << 34
MAX_INVENTORY_ROWS = 32
MAX_INVENTORY_NAME_BYTES = 128
MAX_SOURCE_MEMBER_BYTES = 1 << 20
MAX_SOURCE_AGGREGATE_BYTES = 8 << 20
MAX_EXTERNAL_INVENTORY_BYTES = 128 << 10
MAX_DEPENDENCIES = 16
MAX_DEPENDENCY_ID_BYTES = 64
MAX_RELATIVE_PATH_BYTES = 512
MAX_RUNTIME_LOCK_BYTES = 4 << 20
MAX_DISTRIBUTION_FILES = 250_000
MAX_DISTRIBUTION_NAME_BYTES = 1024
MAX_DISTRIBUTION_AGGREGATE_BYTES = 32 << 30
MAX_RUNTIME_FILE_BYTES = 8 << 30

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEPENDENCY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
UUID_PATTERN = re.compile(r"^GPU-[0-9a-fA-F-]{16,64}$")
PCI_PATTERN = re.compile(r"^(?:[0-9a-fA-F]{4,8}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
VERSION_PATTERN = re.compile(r"^[^\x00-\x20]{1,128}$")


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant: {value}")


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in rows:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _finite(value: Any) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite JSON number")
    elif isinstance(value, list):
        for child in value:
            _finite(child)
    elif isinstance(value, dict):
        for child in value.values():
            _finite(child)


def strict_json_loads(raw: bytes | str) -> Any:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError) as exc:
        raise ContractError(f"invalid strict JSON: {exc}") from exc
    _finite(value)
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def valid_sha256(value: Any, *, nonzero: bool = False) -> bool:
    return (
        isinstance(value, str)
        and SHA256_PATTERN.fullmatch(value) is not None
        and (not nonzero or value != "0" * 64)
    )


def exact_positive_int(value: Any, maximum: int, label: str) -> int:
    require(type(value) is int and 0 < value <= maximum, label)
    return value


def exact_nonnegative_int(value: Any, maximum: int, label: str) -> int:
    require(type(value) is int and 0 <= value <= maximum, label)
    return value


def exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == keys, f"{label} exact keys")
    return value


def checked_product(left: int, right: int, maximum: int, label: str) -> int:
    require(type(left) is int and type(right) is int and left > 0 and right > 0, label)
    require(left <= maximum // right, label)
    return left * right


def validate_constants() -> None:
    require(MICROS_PER_N18 * MICRO == N18, "N18/micro partition")
    require(MICROS_PER_N18 * COARSE_BYTES_PER_MICRO == N18_COARSE_RESERVOIR_BYTES, "307/128 N18 reservoir")
    require(8 * COARSE_BYTES_PER_MICRO * 128 == MICRO * 307, "307/128 full-micro coarse rate")
    require(
        COARSE_BYTES_PER_MICRO + FINE_BYTES_PER_MICRO + METADATA_BYTES_PER_MICRO
        == TOTAL_BYTES_PER_MICRO,
        "DH384 physical handoff",
    )
    require(16 * TOTAL_BYTES_PER_MICRO == 5 * MICRO, "2.5-bpw full-micro slot")
    require(MIN_EXPLICIT_FRAME_BYTES > 2 * PAGE_BYTES - 2, "strict worst-offset page proof threshold")


validate_constants()
