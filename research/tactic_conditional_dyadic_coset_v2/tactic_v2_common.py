#!/usr/bin/env python3
"""Standard-library primitives shared by TACTIC-DH384 v2.

This module has no NumPy, CuPy, CUDA, model-discovery, or network dependency.
The held-descriptor helpers are intentionally POSIX-only because the future
payload gate is frozen for the supplied Linux RunPod.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


DESIGN_SCHEMA = "tactic_dh384_source_only_design_v2"
COARSE_LOCK_SCHEMA = "tactic_dh384_actual_coarse_lock_v2"
SYNTHETIC_AUTHORIZATION = "SYNTHETIC_ONLY_TACTIC_DH384_V2"
PAYLOAD_AUTHORIZATION = "OPEN_AUTHENTICATED_ACTUAL_LOWER_RATE_TACTIC_DH384_V2"
SPLITMIX_DOMAIN = 0x5441435449434448
MASK64 = (1 << 64) - 1
UNIVERSAL_SELECTOR_ORDINAL = 17


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: bytes | str) -> Any:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError) as exc:
        raise ContractError(f"invalid strict JSON: {exc}") from exc
    _walk_finite(value)
    return value


def _walk_finite(value: Any) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite JSON number")
    elif isinstance(value, dict):
        for child in value.values():
            _walk_finite(child)
    elif isinstance(value, list):
        for child in value:
            _walk_finite(child)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def universal_selector_table() -> bytes:
    """Return the sole source-independent table admitted by the v2 format."""
    state = (
        SPLITMIX_DOMAIN ^ ((UNIVERSAL_SELECTOR_ORDINAL + 1) * 0xD1B54A32D192ED03)
    ) & MASK64
    output = bytearray(12 * 256)
    for index in range(len(output)):
        state, word = splitmix64(state)
        output[index] = word & 7
    return bytes(output)


def selector_packet(table: bytes) -> bytes:
    require(len(table) == 12 * 256, "selector table length")
    require(table == universal_selector_table(), "non-universal selector table rejected")
    header = canonical_json({
        "format": "TACTIC-DH384-UNIVERSAL-SELECTOR-v2",
        "generator": "SplitMix64",
        "generator_domain_u64_hex": f"{SPLITMIX_DOMAIN:016x}",
        "universal_ordinal": UNIVERSAL_SELECTOR_ORDINAL,
        "stages": 12,
        "states": 256,
        "ops": "swap/sign0/sign1",
    }) + b"\n"
    require(len(header) + len(table) <= 16_384, "selector packet overflow")
    return header + table + bytes(16_384 - len(header) - len(table))


def feature_state(role: int, u: int, v: int, mean_abs: int) -> int:
    require(0 <= role < 3, "role ordinal")
    au, av = abs(int(u)), abs(int(v))
    threshold = max(0, int(mean_abs))
    feature = (
        (role << 6)
        | ((int(u) < 0) << 5)
        | ((int(v) < 0) << 4)
        | ((au > av) << 3)
        | (((au + av) > 2 * threshold) << 2)
        | ((au > threshold) << 1)
        | (av > threshold)
    )
    require(0 <= feature < 256, "feature state overflow")
    return feature


def _pair_indices(length: int, stage: int) -> Iterable[tuple[int, int]]:
    stride = 1 << stage
    for base in range(0, length, 2 * stride):
        for offset in range(stride):
            yield base + offset, base + stride + offset


def cpu_schedule(symbols: list[int], role: int, table: bytes) -> tuple[list[list[int]], list[int]]:
    length = len(symbols)
    require(length > 0 and length & (length - 1) == 0, "block is not power-of-two")
    stages = length.bit_length() - 1
    require(stages % 2 == 0, "fixture requires an exact dyadic sqrt normalization")
    require(len(table) >= stages * 256, "selector table too short")
    mean_abs = sum(abs(int(value)) for value in symbols) // length
    shadow = [int(value) for value in symbols]
    schedule: list[list[int]] = []
    for stage in range(stages):
        ops: list[int] = []
        next_shadow = shadow.copy()
        for left, right in _pair_indices(length, stage):
            u, v = shadow[left], shadow[right]
            feature = feature_state(role, u, v, mean_abs)
            op = table[stage * 256 + feature] & 7
            swap = op & 1
            a, b = (v, u) if swap else (u, v)
            if op & 2:
                a = -a
            if op & 4:
                b = -b
            next_shadow[left] = a + b
            next_shadow[right] = a - b
            ops.append(op)
        shadow = next_shadow
        schedule.append(ops)
    return schedule, shadow


def cpu_transpose(error: list[float], schedule: list[list[int]]) -> list[float]:
    length = len(error)
    stages = len(schedule)
    require(length == 1 << stages and stages % 2 == 0, "schedule geometry")
    values = [float(value) for value in error]
    for stage in reversed(range(stages)):
        next_values = values.copy()
        ops = schedule[stage]
        pairs = list(_pair_indices(length, stage))
        require(len(ops) == len(pairs), "schedule pair count")
        for pair_ordinal, (left, right) in enumerate(pairs):
            op = ops[pair_ordinal]
            a, b = values[left], values[right]
            x0, x1 = a + b, a - b
            if op & 2:
                x0 = -x0
            if op & 4:
                x1 = -x1
            if op & 1:
                x0, x1 = x1, x0
            next_values[left], next_values[right] = x0, x1
        values = next_values
    normalization = 1 << (stages // 2)
    return [value / normalization for value in values]


def cpu_projection(symbols: list[int], error: list[float], role: int, table: bytes, rank: int) -> dict[str, float]:
    require(len(symbols) == len(error), "symbol/error geometry")
    require(0 < rank < len(error), "projection rank")
    schedule, _ = cpu_schedule(symbols, role, table)
    coefficients = cpu_transpose(error, schedule)
    energy = math.fsum(value * value for value in error)
    transformed_energy = math.fsum(value * value for value in coefficients)
    projected = math.fsum(value * value for value in coefficients[:rank])
    require(math.isclose(energy, transformed_energy, rel_tol=2e-12, abs_tol=2e-12),
            "dyadic transform norm identity failed")
    return {
        "energy": energy,
        "transformed_energy": transformed_energy,
        "projected_energy": projected,
        "residual_energy": energy - projected,
    }


class HeldAbsolute:
    """No-follow component walk retaining every descriptor until close."""

    def __init__(self, path: Path, want_directory: bool):
        require(os.name == "posix", "held-descriptor contract is POSIX-only")
        raw = os.fspath(path)
        require(os.path.isabs(raw), "held path must be absolute")
        require(".." not in Path(raw).parts, "dot-dot path rejected")
        self.path = Path(raw)
        self._fds: list[int] = []
        flags_dir = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        root_fd = os.open("/", flags_dir)
        self._fds.append(root_fd)
        current_fd = root_fd
        parts = [part for part in PurePosixPath(raw).parts if part != "/"]
        require(parts, "root is not an admissible target")
        for component in parts[:-1]:
            fd = os.open(component, flags_dir, dir_fd=current_fd)
            require(stat.S_ISDIR(os.fstat(fd).st_mode), "non-directory path component")
            self._fds.append(fd)
            current_fd = fd
        final_flags = flags_dir if want_directory else os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        final_fd = os.open(parts[-1], final_flags, dir_fd=current_fd)
        mode = os.fstat(final_fd).st_mode
        require(stat.S_ISDIR(mode) if want_directory else stat.S_ISREG(mode),
                "held final object type mismatch")
        self._fds.append(final_fd)
        self.fd = final_fd

    def close(self) -> None:
        for fd in reversed(self._fds):
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def __enter__(self) -> "HeldAbsolute":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class HeldRoot(HeldAbsolute):
    def __init__(self, path: Path):
        super().__init__(path, want_directory=True)
        self._children: list[int] = []

    def open_relative(self, relative: str) -> int:
        pure = PurePosixPath(relative)
        require(not pure.is_absolute() and pure.parts and ".." not in pure.parts,
                "invalid relative record path")
        current = self.fd
        dirs: list[int] = []
        flags_dir = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        for component in pure.parts[:-1]:
            fd = os.open(component, flags_dir, dir_fd=current)
            require(stat.S_ISDIR(os.fstat(fd).st_mode), "record parent is not directory")
            dirs.append(fd)
            current = fd
        fd = os.open(pure.parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current)
        require(stat.S_ISREG(os.fstat(fd).st_mode), "record is not regular")
        self._children.extend(dirs)
        self._children.append(fd)
        return fd

    def close(self) -> None:
        for fd in reversed(getattr(self, "_children", [])):
            try:
                os.close(fd)
            except OSError:
                pass
        self._children = []
        super().close()


def read_stable_fd(fd: int, expected_bytes: int, expected_sha256: str) -> bytes:
    before = os.fstat(fd)
    require(stat.S_ISREG(before.st_mode), "descriptor no longer regular")
    require(before.st_size == expected_bytes, "held file byte count mismatch")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = expected_bytes
    while remaining:
        chunk = os.read(fd, min(1 << 20, remaining))
        require(bool(chunk), "unexpected EOF on held descriptor")
        chunks.append(chunk)
        remaining -= len(chunk)
    require(os.read(fd, 1) == b"", "trailing bytes on held descriptor")
    payload = b"".join(chunks)
    require(sha256_bytes(payload) == expected_sha256, "held file hash mismatch")
    after = os.fstat(fd)
    require((before.st_dev, before.st_ino, before.st_mode, before.st_size) ==
            (after.st_dev, after.st_ino, after.st_mode, after.st_size),
            "held file identity changed during read")
    return payload


def read_bounded_stable_fd(fd: int, maximum_bytes: int) -> bytes:
    """Read one held regular descriptor exactly, without a caller-supplied hash."""
    before = os.fstat(fd)
    require(stat.S_ISREG(before.st_mode), "descriptor no longer regular")
    require(0 < before.st_size <= maximum_bytes, "held file outside bounded byte count")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = before.st_size
    while remaining:
        chunk = os.read(fd, min(1 << 20, remaining))
        require(bool(chunk), "unexpected EOF on held descriptor")
        chunks.append(chunk)
        remaining -= len(chunk)
    require(os.read(fd, 1) == b"", "trailing bytes on held descriptor")
    after = os.fstat(fd)
    require((before.st_dev, before.st_ino, before.st_mode, before.st_size) ==
            (after.st_dev, after.st_ino, after.st_mode, after.st_size),
            "held file identity changed during read")
    return b"".join(chunks)


class HeldOutput:
    """Create-new output directory retained by descriptor."""

    def __init__(self, path: Path):
        require(os.name == "posix", "held output is POSIX-only")
        raw = os.fspath(path)
        require(os.path.isabs(raw) and ".." not in Path(raw).parts, "invalid output path")
        require(not os.path.lexists(raw), "output must be absent")
        self.parent = HeldAbsolute(Path(raw).parent, want_directory=True)
        name = Path(raw).name
        require(name not in ("", ".", ".."), "invalid output basename")
        os.mkdir(name, 0o700, dir_fd=self.parent.fd)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(name, flags, dir_fd=self.parent.fd)
        require(stat.S_ISDIR(os.fstat(self.fd).st_mode), "created output is not directory")

    def write_new(self, name: str, payload: bytes) -> None:
        pure = PurePosixPath(name)
        require(len(pure.parts) == 1 and pure.name not in ("", ".", ".."), "invalid output member")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(pure.name, flags, 0o600, dir_fd=self.fd)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                require(written > 0, "short output write")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def close(self) -> None:
        try:
            os.fsync(self.fd)
        finally:
            os.close(self.fd)
            self.parent.close()

    def __enter__(self) -> "HeldOutput":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
