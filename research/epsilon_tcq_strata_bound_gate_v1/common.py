#!/usr/bin/env python3
"""Strict, standard-library utilities for the epsilon-TCQ v1 source gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(payload: bytes) -> str:
    require(type(payload) is bytes, "SHA-256 bytes")
    return hashlib.sha256(payload).hexdigest()


def require_digest(value: Any, label: str) -> str:
    require(type(value) is str and len(value) == 64, f"{label} digest syntax")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise GateError(f"{label} digest syntax") from error
    require(value == value.lower(), f"{label} lowercase digest")
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes) -> Any:
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, f"duplicate JSON key {key!r}")
            output[key] = value
        return output

    return json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            GateError(f"nonfinite JSON value {item}")))


def exact_keys(row: Mapping[str, Any], keys: set[str], label: str) -> None:
    require(isinstance(row, Mapping) and set(row) == keys, f"{label} exact schema")


def sealed_record(row: Mapping[str, Any], seal_key: str = "seal_sha256") -> dict[str, Any]:
    require(seal_key not in row, "seal key absent before sealing")
    output = dict(row)
    output[seal_key] = sha256(canonical_json(output))
    return output


def verify_seal(row: Mapping[str, Any], seal_key: str = "seal_sha256") -> str:
    require(isinstance(row, Mapping) and seal_key in row, "sealed record")
    seal = require_digest(row[seal_key], "record seal")
    body = dict(row)
    del body[seal_key]
    require(sha256(canonical_json(body)) == seal, "record seal mismatch")
    return seal


def read_regular(path: Path, *, expected_bytes: int | None = None,
                 expected_sha256: str | None = None,
                 maximum_bytes: int = 1 << 30) -> bytes:
    require(path.is_absolute(), "absolute regular-file path")
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                "regular single-link file")
        require(0 <= before.st_size <= maximum_bytes, "regular-file size cap")
        if expected_bytes is not None:
            require(before.st_size == expected_bytes, "regular-file byte pin")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(chunk, "regular-file short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_nlink), "regular-file identity drift")
        payload = b"".join(chunks)
        if expected_sha256 is not None:
            require(sha256(payload) == expected_sha256, "regular-file digest pin")
        return payload
    finally:
        os.close(descriptor)


def fp64_values(payload: bytes, *, count: int) -> tuple[float, ...]:
    import struct

    require(type(payload) is bytes and len(payload) == 8 * count,
            "binary64 artifact geometry")
    values = struct.unpack("<" + "d" * count, payload)
    require(all(math.isfinite(value) for value in values),
            "finite binary64 artifact")
    return tuple(float(value) for value in values)
