#!/usr/bin/env python3
"""POSIX held-descriptor input and completion-last publication primitives."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from typing import Any

from n18_common import canonical_json, is_sha256, require


def _posix() -> None:
    require(os.name == "posix", "payload I/O requires POSIX openat/O_NOFOLLOW semantics")


def _components(path: str) -> list[str]:
    require(isinstance(path, str) and os.path.isabs(path), "absolute path required")
    normalized = os.path.normpath(path)
    require(normalized == path and normalized != "/", "canonical absolute leaf path")
    parts = [part for part in normalized.split("/") if part]
    require(parts and all(part not in (".", "..") for part in parts), "safe path components")
    return parts


def open_absolute_directory(path: str) -> int:
    _posix()
    require(isinstance(path, str) and os.path.isabs(path), "absolute directory path")
    normalized = os.path.normpath(path)
    require(normalized == path, "canonical absolute directory path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in [part for part in normalized.split("/") if part]:
            require(component not in (".", ".."), "safe directory components")
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        require(stat.S_ISDIR(os.fstat(descriptor).st_mode), "held directory type")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_all(descriptor: int, expected_bytes: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    pieces: list[bytes] = []
    remaining = expected_bytes
    while remaining:
        packet = os.read(descriptor, min(1 << 20, remaining))
        require(bool(packet), "held input early EOF")
        pieces.append(packet)
        remaining -= len(packet)
    require(os.read(descriptor, 1) == b"", "held input trailing bytes")
    return b"".join(pieces)


def _hash_all(descriptor: int, expected_bytes: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    hasher = hashlib.sha256()
    remaining = expected_bytes
    while remaining:
        packet = os.read(descriptor, min(1 << 20, remaining))
        require(bool(packet), "held input early EOF")
        hasher.update(packet)
        remaining -= len(packet)
    require(os.read(descriptor, 1) == b"", "held input trailing bytes")
    return hasher.hexdigest()


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


class HeldRegularFile:
    """A no-follow absolute input retained by descriptor for its whole use."""

    def __init__(
        self,
        absolute_path: str,
        *,
        maximum_bytes: int,
        expected_bytes: int | None = None,
        expected_sha256: str | None = None,
        allow_empty: bool = False,
    ) -> None:
        _posix()
        require(type(maximum_bytes) is int and maximum_bytes > 0, "input byte cap")
        parts = _components(absolute_path)
        parent_path = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
        parent = open_absolute_directory(parent_path)
        try:
            self.descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        finally:
            os.close(parent)
        self.path = absolute_path
        try:
            metadata = os.fstat(self.descriptor)
            require(stat.S_ISREG(metadata.st_mode), "held input regular file")
            minimum = 0 if allow_empty else 1
            require(minimum <= metadata.st_size <= maximum_bytes, "held input byte bounds")
            if expected_bytes is not None:
                require(type(expected_bytes) is int and metadata.st_size == expected_bytes, "held input exact bytes")
            if expected_sha256 is not None:
                require(is_sha256(expected_sha256), "held input expected SHA-256 syntax")
            self.bytes = metadata.st_size
            self.initial_identity = _identity(metadata)
            self.sha256 = _hash_all(self.descriptor, self.bytes)
            if expected_sha256 is not None:
                require(self.sha256 == expected_sha256, "held input SHA-256")
            require(_identity(os.fstat(self.descriptor)) == self.initial_identity, "held input changed while hashing")
        except Exception:
            os.close(self.descriptor)
            raise

    def read(self, *, maximum_materialize_bytes: int = 16 << 20) -> bytes:
        require(
            type(maximum_materialize_bytes) is int
            and 0 <= self.bytes <= maximum_materialize_bytes,
            "held input materialization cap",
        )
        packet = _read_all(self.descriptor, self.bytes)
        require(hashlib.sha256(packet).hexdigest() == self.sha256, "held input content changed")
        require(_identity(os.fstat(self.descriptor)) == self.initial_identity, "held input identity changed")
        return packet

    def proc_path(self) -> str:
        """Path alias to this exact held FD for legacy read-only libraries."""
        require(os.path.isdir("/proc/self/fd"), "/proc/self/fd required")
        return f"/proc/self/fd/{self.descriptor}"

    def verify_stable(self) -> None:
        require(_hash_all(self.descriptor, self.bytes) == self.sha256, "held input content changed")
        require(_identity(os.fstat(self.descriptor)) == self.initial_identity, "held input identity changed")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "HeldRegularFile":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class HeldFileSet:
    def __init__(self) -> None:
        self.files: list[HeldRegularFile] = []

    def add(self, value: HeldRegularFile) -> HeldRegularFile:
        self.files.append(value)
        return value

    def verify_stable(self) -> None:
        for value in self.files:
            value.verify_stable()

    def close(self) -> None:
        for value in reversed(self.files):
            value.close()
        self.files.clear()

    def __enter__(self) -> "HeldFileSet":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _write_all(descriptor: int, packet: bytes) -> None:
    view = memoryview(packet)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        require(written > 0, "publication short write")
        offset += written


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    require(function is not None, "renameat2(RENAME_NOREPLACE) required")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(parent_fd, source.encode(), parent_fd, destination.encode(), 1) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code))


@dataclass(frozen=True)
class PublicationReceipt:
    output_path: str
    artifact_index_sha256: str
    completion_sha256: str
    files: tuple[dict[str, Any], ...]


class CompletionLastPublisher:
    """Flat, durable staging tree atomically made visible without replacement."""

    RESERVED = frozenset({"ARTIFACTS.json", "COMPLETE.json"})

    def __init__(self, output_path: str, source_manifest_sha256: str) -> None:
        _posix()
        require(is_sha256(source_manifest_sha256), "source manifest digest")
        absolute = os.path.normpath(output_path)
        require(os.path.isabs(output_path) and absolute == output_path, "canonical absolute output")
        parent_path, final_name = os.path.split(absolute)
        require(final_name not in ("", ".", "..") and "/" not in final_name, "output leaf")
        self.parent_fd = open_absolute_directory(parent_path)
        self.final_name = final_name
        self.output_path = absolute
        self.source_manifest_sha256 = source_manifest_sha256
        self.staging_name = f".{final_name}.staging.{os.getpid()}.{secrets.token_hex(12)}"
        self.files: list[dict[str, Any]] = []
        self.finished = False
        try:
            # Fail before payload access if the final leaf already exists.
            try:
                os.stat(final_name, dir_fd=self.parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(final_name)
            os.mkdir(self.staging_name, 0o700, dir_fd=self.parent_fd)
            self.staging_fd = os.open(
                self.staging_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self.parent_fd,
            )
        except Exception:
            os.close(self.parent_fd)
            raise

    @staticmethod
    def _name(name: str) -> None:
        require(
            isinstance(name, str)
            and name not in ("", ".", "..")
            and "/" not in name
            and "\\" not in name,
            "flat output name",
        )

    def _write(self, name: str, packet: bytes) -> None:
        self._name(name)
        require(type(packet) is bytes, "artifact packet bytes")
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.staging_fd,
        )
        try:
            _write_all(descriptor, packet)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write(self, name: str, packet: bytes) -> dict[str, Any]:
        require(not self.finished and name not in self.RESERVED, "artifact write state/name")
        require(all(row["name"] != name for row in self.files), "duplicate artifact name")
        self._write(name, packet)
        row = {"name": name, "bytes": len(packet), "sha256": hashlib.sha256(packet).hexdigest()}
        self.files.append(row)
        return row

    def complete(self, result: dict[str, Any]) -> PublicationReceipt:
        require(not self.finished, "publication already complete")
        rows = sorted(self.files, key=lambda row: row["name"])
        index = {
            "schema": "tactic_actual_coarse_n18_artifact_index_v2",
            "source_manifest_sha256": self.source_manifest_sha256,
            "files": rows,
        }
        index_packet = canonical_json(index) + b"\n"
        self._write("ARTIFACTS.json", index_packet)
        completion = {
            "schema": "tactic_actual_coarse_n18_completion_v2",
            "complete": True,
            "source_manifest_sha256": self.source_manifest_sha256,
            "artifact_index_sha256": hashlib.sha256(index_packet).hexdigest(),
            "result": result,
        }
        completion_packet = canonical_json(completion) + b"\n"
        # COMPLETE.json is created last, after every artifact and the index fsync.
        self._write("COMPLETE.json", completion_packet)
        os.fsync(self.staging_fd)
        _rename_noreplace(self.parent_fd, self.staging_name, self.final_name)
        os.fsync(self.parent_fd)
        # Re-open the now-public exact directory and rehash every byte before
        # returning a completion receipt.
        final_fd = os.open(
            self.final_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=self.parent_fd,
        )
        try:
            expected = {
                **{str(row["name"]): (int(row["bytes"]), str(row["sha256"])) for row in rows},
                "ARTIFACTS.json": (len(index_packet), hashlib.sha256(index_packet).hexdigest()),
                "COMPLETE.json": (len(completion_packet), hashlib.sha256(completion_packet).hexdigest()),
            }
            require(set(os.listdir(final_fd)) == set(expected), "published file set")
            for name, (expected_bytes, expected_sha256) in expected.items():
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=final_fd,
                )
                try:
                    metadata = os.fstat(descriptor)
                    require(stat.S_ISREG(metadata.st_mode) and metadata.st_size == expected_bytes, "published file type/bytes")
                    require(_hash_all(descriptor, expected_bytes) == expected_sha256, "published file SHA-256")
                finally:
                    os.close(descriptor)
        finally:
            os.close(final_fd)
        self.finished = True
        os.close(self.staging_fd)
        os.close(self.parent_fd)
        return PublicationReceipt(
            self.output_path,
            hashlib.sha256(index_packet).hexdigest(),
            hashlib.sha256(completion_packet).hexdigest(),
            tuple(rows),
        )

    def abort(self) -> None:
        if self.finished:
            return
        try:
            for name in os.listdir(self.staging_fd):
                os.unlink(name, dir_fd=self.staging_fd)
            os.close(self.staging_fd)
            os.rmdir(self.staging_name, dir_fd=self.parent_fd)
            os.fsync(self.parent_fd)
        finally:
            os.close(self.parent_fd)
            self.finished = True

    def __enter__(self) -> "CompletionLastPublisher":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self.finished:
            self.abort()
