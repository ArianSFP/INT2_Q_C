#!/usr/bin/env python3
"""Exact source/runtime closure for the immutable N18 v6 successor.

Import is inert.  NumPy, CuPy and predecessor code are loaded only by
``load_runtime`` after both lock packets have been authenticated.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ClosureError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosureError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode), f"{label} symlink chain: {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def read_held_regular(
    path: Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    maximum_bytes: int = 4 * (1 << 20),
    label: str,
) -> bytes:
    require(path.is_absolute(), f"{label} absolute path")
    _reject_symlink_chain(path, label)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} regular file")
        require(0 < before.st_size <= maximum_bytes, f"{label} byte bound")
        if expected_bytes is not None:
            require(before.st_size == expected_bytes, f"{label} exact bytes")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label} short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", f"{label} trailing read")
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns),
            f"{label} identity drift",
        )
        payload = b"".join(chunks)
        if expected_sha256 is not None:
            require(sha256(payload) == expected_sha256, f"{label} SHA-256")
        return payload
    finally:
        os.close(descriptor)


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ClosureError(f"{label} nonfinite {item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ClosureError(f"{label} JSON: {error}") from error
    require(isinstance(value, dict), f"{label} JSON object")
    return value


def _load_module(name: str, source: bytes, digest: str) -> Any:
    require(name not in sys.modules, f"authenticated module name collision: {name}")
    require(sha256(source) == digest, f"authenticated module digest: {name}")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{digest}>"
    module.__package__ = ""
    module.__authenticated_source_sha256__ = digest
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _v4_source_root(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"TACTIC-N18-V4-SOURCE-ROOT-v1\0")
    for row in sorted(rows, key=lambda item: item["name"].encode("utf-8")):
        name = row["name"].encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(int(row["bytes"]).to_bytes(8, "big"))
        digest.update(bytes.fromhex(row["sha256"]))
    return digest.hexdigest()


def runtime_fingerprint(np: Any, cp: Any) -> dict[str, Any]:
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "device_name": str(name),
        "compute_capability": str(cp.cuda.Device(0).compute_capability),
    }


@dataclass(frozen=True)
class RuntimeClosure:
    packet: Any
    numeric_encoder: Any
    independent_decoder: Any
    encoder_runtime: Any
    decoder_runtime: Any
    numpy: Any
    cupy: Any
    inverse_symbols_i32: Any
    receipt: Mapping[str, Any]


def load_runtime(
    repo_root: Path,
    package_dir: Path,
    *,
    expected_predecessor_lock_sha256: str,
    expected_runtime_lock_sha256: str,
) -> RuntimeClosure:
    root = repo_root.resolve(strict=True)
    package = package_dir.resolve(strict=True)
    require(root.is_dir() and package.is_dir(), "runtime directories")
    require(package == root / "research" / "tactic_actual_coarse_n18_v6",
            "v6 canonical package location")
    predecessor_payload = read_held_regular(
        package / "PREDECESSOR_LOCK.json",
        expected_sha256=expected_predecessor_lock_sha256,
        label="predecessor lock",
    )
    runtime_payload = read_held_regular(
        package / "RUNTIME_LOCK.json",
        expected_sha256=expected_runtime_lock_sha256,
        label="runtime lock",
    )
    predecessor = _strict_json(predecessor_payload, "predecessor lock")
    runtime_lock = _strict_json(runtime_payload, "runtime lock")
    require(
        predecessor.get("schema") == "tactic-actual-coarse-n18-v6-predecessor-lock-v1",
        "predecessor lock schema",
    )
    require(
        runtime_lock.get("schema") == "tactic-actual-coarse-n18-v6-runtime-lock-v1",
        "runtime lock schema",
    )
    v4_dir = root / predecessor["relative_directory"]
    require(v4_dir.is_dir(), "v4 source directory")
    _reject_symlink_chain(v4_dir, "v4 directory")
    rows = predecessor["members"]
    require(isinstance(rows, list) and rows, "v4 member lock")
    sources: dict[str, bytes] = {}
    observed: list[dict[str, Any]] = []
    for row in rows:
        require(set(row) == {"name", "bytes", "sha256"}, "v4 member row")
        name = row["name"]
        require(isinstance(name, str) and name and "/" not in name and "\\" not in name, "v4 member name")
        payload = read_held_regular(
            v4_dir / name,
            expected_bytes=int(row["bytes"]),
            expected_sha256=str(row["sha256"]),
            label=f"v4 member {name}",
        )
        sources[name] = payload
        observed.append({"name": name, "bytes": len(payload), "sha256": sha256(payload)})
    require(
        _v4_source_root(observed) == predecessor["source_root_sha256"],
        "v4 source root",
    )
    actual_names = {
        entry.name for entry in os.scandir(v4_dir) if entry.is_file(follow_symlinks=False)
    }
    require(actual_names == {row["name"] for row in observed}, "v4 exact file closure")
    for dependency in predecessor["numerical_dependencies"]:
        read_held_regular(
            root / dependency["relative_path"],
            expected_bytes=int(dependency["bytes"]),
            expected_sha256=str(dependency["sha256"]),
            label=f"numerical dependency {dependency['relative_path']}",
        )

    # The v4 modules use `from packet_format import ...`; bind that exact
    # authenticated dependency before executing either numerical module.
    packet = _load_module("packet_format", sources["packet_format.py"], sha256(sources["packet_format.py"]))
    numeric = _load_module(
        "tacn18_v6_authenticated_v4_numeric",
        sources["numeric_encoder.py"],
        sha256(sources["numeric_encoder.py"]),
    )
    decoder = _load_module(
        "tacn18_v6_authenticated_v4_decoder",
        sources["independent_decoder.py"],
        sha256(sources["independent_decoder.py"]),
    )

    import numpy as np
    import cupy as cp

    observed_runtime = runtime_fingerprint(np, cp)
    for key, value in observed_runtime.items():
        require(runtime_lock.get(key) == value, f"runtime lock mismatch: {key}")
    encoder_runtime = numeric.load_encoder_runtime(root)
    decoder_runtime = decoder.load_decoder_runtime(root)

    # V6's only numerical semantic override: retain the v4 packet and SC
    # decode, but widen the exact inverse-Hadamard transient/output to I32.
    def integer_inverse_symbols_i32(np_module: Any, indices: Any, rht_seed: int) -> Any:
        values = indices.astype(np_module.int32) - np_module.int32(31)
        width = 1
        while width < packet.N:
            view = values.reshape(-1, 2, width)
            left = view[:, 0, :].copy()
            right = view[:, 1, :].copy()
            view[:, 0, :] = left + right
            view[:, 1, :] = left - right
            width *= 2
        with np_module.errstate(over="ignore"):
            z = np_module.arange(packet.N, dtype=np_module.uint64) + np_module.uint64(rht_seed)
            z += np_module.uint64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> np_module.uint64(30))) * np_module.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np_module.uint64(27))) * np_module.uint64(0x94D049BB133111EB)
            z ^= z >> np_module.uint64(31)
        signs = np_module.where((z & np_module.uint64(1)) == 0, 1, -1).astype(np_module.int32)
        values *= signs
        maximum = int(np_module.max(np_module.abs(values.astype(np_module.int64))))
        require(maximum <= 32 * packet.N, "I32 inverse mathematical bound")
        require(32 * packet.N < 2**31, "I32 inverse type bound")
        return values.astype("<i4", copy=False)

    decoder._integer_inverse_symbols = integer_inverse_symbols_i32
    receipt = {
        "schema": "tactic-actual-coarse-n18-v6-runtime-closure-v1",
        "predecessor_lock_sha256": sha256(predecessor_payload),
        "predecessor_source_root_sha256": predecessor["source_root_sha256"],
        "runtime_lock_sha256": sha256(runtime_payload),
        "runtime": observed_runtime,
        "executed_v4_sources": {
            name: sha256(sources[name])
            for name in ("packet_format.py", "numeric_encoder.py", "independent_decoder.py")
        },
        "inverse_transient_dtype": "<i4",
        "inverse_transient_abs_bound": 32 * packet.N,
        "inverse_override_installed_before_any_reservoir_decode": True,
        "v4_source_modified": False,
    }
    return RuntimeClosure(
        packet,
        numeric,
        decoder,
        encoder_runtime,
        decoder_runtime,
        np,
        cp,
        integer_inverse_symbols_i32,
        receipt,
    )
