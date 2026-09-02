#!/usr/bin/env python3
"""Authenticate and load the exact frozen N18-v6 decoder/runtime closure."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any


class BridgeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BridgeError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                BridgeError(f"{label}: nonfinite {item}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BridgeError(f"{label}: JSON: {error}") from error
    require(isinstance(value, dict), f"{label}: object")
    return value


def load_module(name: str, source: bytes) -> Any:
    require(name not in sys.modules, f"module collision: {name}")
    digest = sha256(source)
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{digest}>"
    module.__package__ = ""
    module.__authenticated_source_sha256__ = digest
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec", dont_inherit=True,
                     optimize=0), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode,
        metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _pread(descriptor: int, size: int, label: str) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1 << 20, size - offset), offset)
        require(bool(chunk), f"{label}: short read")
        chunks.append(chunk)
        offset += len(chunk)
    require(os.pread(descriptor, 1, size) == b"", f"{label}: trailing bytes")
    return b"".join(chunks)


class HeldV6Package:
    """Exact immutable v6 source tree retained through the finite run."""

    def __init__(self, repo_root: Path, package_dir: Path,
                 lock_payload: bytes) -> None:
        lock = strict_json(lock_payload, "v6 bridge lock")
        require(lock.get("schema") == "tactic-dh384-finite-v3-v6-lock-v1",
                "v6 bridge lock schema")
        expected_dir = repo_root / lock["relative_directory"]
        require(package_dir.resolve(strict=True) == expected_dir.resolve(strict=True),
                "canonical v6 package directory")
        self.path = package_dir.resolve(strict=True)
        self.directory_fd = os.open(
            os.fspath(self.path), os.O_RDONLY |
            getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        self.directory_identity = _identity(os.fstat(self.directory_fd))
        self.descriptors: dict[str, int] = {}
        self.identities: dict[str, tuple[int, int, int, int, int, int, int]] = {}
        self.sources: dict[str, bytes] = {}
        try:
            manifest_row = lock["source_manifest"]
            manifest_fd = self._open(
                "SOURCE_MANIFEST.json", manifest_row["bytes"],
                manifest_row["sha256"])
            manifest_payload = self.sources["SOURCE_MANIFEST.json"]
            manifest = strict_json(manifest_payload, "v6 source manifest")
            require(manifest.get("schema") ==
                    "tactic-actual-coarse-n18-v6-source-manifest-v1",
                    "v6 source manifest schema")
            require(manifest.get("source_root_sha256") ==
                    lock["source_root_sha256"], "v6 source root pin")
            rows = manifest.get("members")
            require(rows == lock["members"], "v6 exact member inventory lock")
            observed: list[dict[str, Any]] = []
            for row in rows:
                self._open(row["name"], row["bytes"], row["sha256"])
                observed.append({
                    "name": row["name"], "bytes": row["bytes"],
                    "sha256": row["sha256"],
                })
            require(sha256(canonical_json(observed)) ==
                    lock["source_root_sha256"],
                    "v6 canonical source-root reconstruction")
            entries = list(os.scandir(self.directory_fd))
            require({entry.name for entry in entries} == set(self.sources) and
                    all(entry.is_file(follow_symlinks=False) for entry in entries),
                    "v6 exact regular source closure")
            require(sha256(self.sources["PREDECESSOR_LOCK.json"]) ==
                    lock["predecessor_lock_sha256"],
                    "v6 predecessor pin")
            require(sha256(self.sources["RUNTIME_LOCK.json"]) ==
                    lock["runtime_lock_sha256"], "v6 runtime pin")
            self.lock = lock
            self.manifest_sha256 = manifest_row["sha256"]
            self.source_root_sha256 = lock["source_root_sha256"]
        except BaseException:
            self.close()
            raise

    def _open(self, name: str, expected_bytes: int, expected_sha256: str) -> int:
        require(isinstance(name, str) and name and name not in {".", ".."}
                and "/" not in name and "\\" not in name and
                name not in self.sources, "v6 safe unique source name")
        descriptor = os.open(
            name, os.O_RDONLY | getattr(os, "O_BINARY", 0) |
            getattr(os, "O_NOFOLLOW", 0), dir_fd=self.directory_fd)
        try:
            metadata = os.fstat(descriptor)
            require(stat.S_ISREG(metadata.st_mode) and
                    metadata.st_size == expected_bytes and
                    metadata.st_nlink == 1,
                    f"v6 source identity: {name}")
            payload = _pread(descriptor, expected_bytes, name)
            require(sha256(payload) == expected_sha256,
                    f"v6 source digest: {name}")
            inode = (metadata.st_dev, metadata.st_ino)
            require(all((row[0], row[1]) != inode
                        for row in self.identities.values()),
                    f"v6 source inode alias: {name}")
            self.descriptors[name] = descriptor
            self.identities[name] = _identity(metadata)
            self.sources[name] = payload
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def load_runtime(self, repo_root: Path) -> tuple[Any, Any]:
        runtime_module = load_module(
            "tactic_dh384_v3_exact_v6_runtime_closure",
            self.sources["runtime_closure.py"])
        codec_module = load_module(
            "tactic_dh384_v3_exact_v6_successor_codec",
            self.sources["successor_codec.py"])
        runtime = runtime_module.load_runtime(
            repo_root, self.path,
            expected_predecessor_lock_sha256=
                self.lock["predecessor_lock_sha256"],
            expected_runtime_lock_sha256=self.lock["runtime_lock_sha256"],
        )
        require(runtime.receipt["runtime_lock_sha256"] ==
                self.lock["runtime_lock_sha256"],
                "loaded v6 runtime receipt pin")
        return runtime, codec_module

    def receipt(self) -> dict[str, Any]:
        return {
            "v6_source_manifest_sha256": self.manifest_sha256,
            "v6_source_root_sha256": self.source_root_sha256,
            "v6_predecessor_lock_sha256":
                self.lock["predecessor_lock_sha256"],
            "v6_runtime_lock_sha256": self.lock["runtime_lock_sha256"],
            "retained_no_follow_source_descriptors": True,
            "exact_v6_successor_codec_sha256":
                sha256(self.sources["successor_codec.py"]),
            "exact_v6_runtime_closure_sha256":
                sha256(self.sources["runtime_closure.py"]),
        }

    def verify_final(self) -> None:
        require(_identity(os.fstat(self.directory_fd)) ==
                self.directory_identity, "v6 directory changed")
        require(_identity(os.stat(self.path, follow_symlinks=False)) ==
                self.directory_identity, "v6 package path rebound")
        for name, descriptor in self.descriptors.items():
            require(_identity(os.fstat(descriptor)) == self.identities[name],
                    f"v6 source changed: {name}")
            require(_identity(os.stat(name, dir_fd=self.directory_fd,
                                      follow_symlinks=False)) ==
                    self.identities[name], f"v6 source rebound: {name}")
            require(_pread(descriptor, len(self.sources[name]), name) ==
                    self.sources[name], f"v6 source bytes changed: {name}")

    def close(self) -> None:
        for descriptor in list(getattr(self, "descriptors", {}).values()):
            try:
                os.close(descriptor)
            except OSError:
                pass
        getattr(self, "descriptors", {}).clear()
        descriptor = getattr(self, "directory_fd", None)
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self.directory_fd = None

    def __enter__(self) -> "HeldV6Package":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
