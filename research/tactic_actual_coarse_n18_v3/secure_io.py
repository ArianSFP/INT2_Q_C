#!/usr/bin/env python3
"""Held no-follow inputs and verify-before-publish atomic output for v3."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from v3_common import canonical_json, require, valid_sha256


def _posix() -> None:
    require(os.name == "posix", "canonical payload I/O requires POSIX")


def _canonical_absolute(path: str) -> tuple[str, list[str]]:
    require(isinstance(path, str) and os.path.isabs(path), "absolute path")
    normalized = os.path.normpath(path)
    require(normalized == path and normalized != "/", "canonical absolute leaf")
    parts = [part for part in normalized.split("/") if part]
    require(parts and all(part not in (".", "..") for part in parts), "safe absolute components")
    return normalized, parts


def open_absolute_directory(path: str) -> int:
    _posix()
    require(isinstance(path, str) and os.path.isabs(path), "absolute directory")
    normalized = os.path.normpath(path)
    require(normalized == path, "canonical absolute directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in [part for part in normalized.split("/") if part]:
            require(component not in (".", ".."), "safe directory component")
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        require(stat.S_ISDIR(os.fstat(descriptor).st_mode), "held directory type")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_descriptor(descriptor: int, size: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = size
    digest = hashlib.sha256()
    while remaining:
        packet = os.read(descriptor, min(1 << 20, remaining))
        require(bool(packet), "held file early EOF")
        digest.update(packet)
        remaining -= len(packet)
    require(os.read(descriptor, 1) == b"", "held file trailing bytes")
    return digest.hexdigest()


def _read_descriptor(descriptor: int, size: int, cap: int) -> bytes:
    require(0 <= size <= cap, "held materialization cap")
    os.lseek(descriptor, 0, os.SEEK_SET)
    result = bytearray()
    while len(result) < size:
        packet = os.read(descriptor, min(1 << 20, size - len(result)))
        require(bool(packet), "held file early EOF")
        result.extend(packet)
    require(os.read(descriptor, 1) == b"", "held file trailing bytes")
    return bytes(result)


class HeldRegularFile:
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
        normalized, parts = _canonical_absolute(absolute_path)
        require(type(maximum_bytes) is int and maximum_bytes > 0, "held maximum bytes")
        parent_path = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
        parent = open_absolute_directory(parent_path)
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        finally:
            os.close(parent)
        self.descriptor = descriptor
        self.path = normalized
        try:
            metadata = os.fstat(descriptor)
            lower = 0 if allow_empty else 1
            require(stat.S_ISREG(metadata.st_mode), "held regular file")
            require(lower <= metadata.st_size <= maximum_bytes, "held byte bounds")
            if expected_bytes is not None:
                require(type(expected_bytes) is int and expected_bytes >= lower, "held expected-byte type")
                require(metadata.st_size == expected_bytes, "held expected bytes")
            if expected_sha256 is not None:
                require(valid_sha256(expected_sha256, nonzero=True), "held expected digest")
            self.bytes = metadata.st_size
            self.initial_identity = _identity(metadata)
            self.sha256 = _hash_descriptor(descriptor, self.bytes)
            if expected_sha256 is not None:
                require(self.sha256 == expected_sha256, "held expected SHA-256")
            require(_identity(os.fstat(descriptor)) == self.initial_identity, "held changed during initial hash")
        except Exception:
            os.close(descriptor)
            self.descriptor = -1
            raise

    def read(self, cap: int) -> bytes:
        packet = _read_descriptor(self.descriptor, self.bytes, cap)
        require(hashlib.sha256(packet).hexdigest() == self.sha256, "held content drift")
        require(_identity(os.fstat(self.descriptor)) == self.initial_identity, "held identity drift")
        return packet

    def verify(self) -> None:
        require(_hash_descriptor(self.descriptor, self.bytes) == self.sha256, "held content drift")
        require(_identity(os.fstat(self.descriptor)) == self.initial_identity, "held identity drift")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> "HeldRegularFile":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class PublicationPhase(Enum):
    NEW = "new"
    PARENT_HELD = "parent_held"
    STAGING_CREATED = "staging_created"
    STAGING_HELD = "staging_held"
    WRITING = "writing"
    INDEX_WRITTEN = "index_written"
    COMPLETE_WRITTEN = "complete_written"
    STAGING_SYNCED = "staging_synced"
    STAGING_VERIFIED = "staging_verified"
    PUBLISHED = "published"
    PARENT_SYNCED = "parent_synced"
    CLOSED = "closed"
    ABORTED = "aborted"


FaultHook = Callable[[PublicationPhase, "VerifiedPublisher"], None]


def _rename_noreplace(parent: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    require(function is not None, "renameat2 required")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(parent, source.encode(), parent, destination.encode(), 1) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code))


def _write_all(descriptor: int, packet: bytes) -> None:
    offset = 0
    while offset < len(packet):
        written = os.write(descriptor, packet[offset:])
        require(written > 0, "publication short write")
        offset += written


@dataclass(frozen=True)
class PublicationReceipt:
    output_path: str
    index_sha256: str
    completion_sha256: str
    phase: str


class VerifiedPublisher:
    """Verify the complete held staging tree before its only publication rename."""

    RESERVED = frozenset({"ARTIFACTS.json", "COMPLETE.json"})

    def __init__(
        self,
        output_path: str,
        authenticated_source_root: str,
        *,
        fault_hook: FaultHook | None = None,
    ) -> None:
        _posix()
        require(valid_sha256(authenticated_source_root, nonzero=True), "authenticated source root")
        absolute, parts = _canonical_absolute(output_path)
        self.output_path = absolute
        self.final_name = parts[-1]
        self.parent_path = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
        self.source_root = authenticated_source_root
        self.fault_hook = fault_hook
        self.phase = PublicationPhase.NEW
        self.parent_fd = -1
        self.staging_fd = -1
        self.staging_name = f".{self.final_name}.staging.{os.getpid()}.{secrets.token_hex(12)}"
        self.staging_exists = False
        self.published = False
        self.files: list[dict[str, Any]] = []
        try:
            self.parent_fd = open_absolute_directory(self.parent_path)
            self._phase(PublicationPhase.PARENT_HELD)
            try:
                os.stat(self.final_name, dir_fd=self.parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(self.output_path)
            os.mkdir(self.staging_name, 0o700, dir_fd=self.parent_fd)
            self.staging_exists = True
            self._phase(PublicationPhase.STAGING_CREATED)
            self.staging_fd = os.open(
                self.staging_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self.parent_fd,
            )
            self._phase(PublicationPhase.STAGING_HELD)
        except Exception:
            self._constructor_cleanup()
            raise

    def _phase(self, value: PublicationPhase) -> None:
        self.phase = value
        if self.fault_hook is not None:
            self.fault_hook(value, self)

    @staticmethod
    def _name(name: str) -> None:
        require(
            isinstance(name, str)
            and 0 < len(name.encode("utf-8")) <= 128
            and name not in (".", "..")
            and "/" not in name
            and "\\" not in name,
            "flat bounded artifact name",
        )

    def _write(self, name: str, packet: bytes) -> dict[str, Any]:
        self._name(name)
        require(type(packet) is bytes, "artifact bytes")
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.staging_fd,
        )
        try:
            _write_all(descriptor, packet)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
        finally:
            os.close(descriptor)
        return {"name": name, "bytes": len(packet), "sha256": hashlib.sha256(packet).hexdigest()}

    def write(self, name: str, packet: bytes) -> dict[str, Any]:
        require(self.phase in (PublicationPhase.STAGING_HELD, PublicationPhase.WRITING), "publisher write phase")
        require(name not in self.RESERVED and all(row["name"] != name for row in self.files), "artifact name reservation")
        row = self._write(name, packet)
        self.files.append(row)
        self._phase(PublicationPhase.WRITING)
        return row

    def _verify_staging(self, expected: dict[str, tuple[int, str]]) -> None:
        require(self.staging_fd >= 0, "held staging descriptor")
        require(set(os.listdir(self.staging_fd)) == set(expected), "staging exact file set")
        for name, (size, digest) in expected.items():
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self.staging_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                require(stat.S_ISREG(metadata.st_mode) and metadata.st_size == size, "staging file type/bytes")
                require(_hash_descriptor(descriptor, size) == digest, "staging file SHA-256")
            finally:
                os.close(descriptor)

    def complete(self, result: dict[str, Any]) -> PublicationReceipt:
        require(self.phase in (PublicationPhase.STAGING_HELD, PublicationPhase.WRITING), "publisher completion phase")
        try:
            ordered = sorted(self.files, key=lambda row: row["name"].encode("utf-8"))
            index = {
                "schema": "tactic_actual_coarse_n18_artifact_index_v3",
                "authenticated_source_root": self.source_root,
                "files": ordered,
            }
            index_packet = canonical_json(index) + b"\n"
            index_row = self._write("ARTIFACTS.json", index_packet)
            self._phase(PublicationPhase.INDEX_WRITTEN)
            completion = {
                "schema": "tactic_actual_coarse_n18_completion_v3",
                "complete": True,
                "authenticated_source_root": self.source_root,
                "artifact_index_sha256": index_row["sha256"],
                "result": result,
            }
            completion_packet = canonical_json(completion) + b"\n"
            completion_row = self._write("COMPLETE.json", completion_packet)
            self._phase(PublicationPhase.COMPLETE_WRITTEN)
            os.fsync(self.staging_fd)
            self._phase(PublicationPhase.STAGING_SYNCED)
            expected = {
                **{str(row["name"]): (int(row["bytes"]), str(row["sha256"])) for row in ordered},
                "ARTIFACTS.json": (index_row["bytes"], index_row["sha256"]),
                "COMPLETE.json": (completion_row["bytes"], completion_row["sha256"]),
            }
            self._verify_staging(expected)
            os.fchmod(self.staging_fd, 0o500)
            os.fsync(self.staging_fd)
            self._phase(PublicationPhase.STAGING_VERIFIED)
            _rename_noreplace(self.parent_fd, self.staging_name, self.final_name)
            self.staging_exists = False
            self.published = True
            self._phase(PublicationPhase.PUBLISHED)
            os.fsync(self.parent_fd)
            self._phase(PublicationPhase.PARENT_SYNCED)
            self._close_descriptors()
            self.phase = PublicationPhase.CLOSED
            return PublicationReceipt(
                self.output_path,
                index_row["sha256"],
                completion_row["sha256"],
                self.phase.value,
            )
        except Exception:
            if not self.published:
                self.abort()
            else:
                # The only public tree was already fully enumerated, rehashed,
                # synced and chmod-restricted before rename. Keep that verified
                # tree; merely close descriptors and expose the committed phase.
                self._close_descriptors()
            raise

    def _close_descriptors(self) -> None:
        for attribute in ("staging_fd", "parent_fd"):
            descriptor = getattr(self, attribute, -1)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                finally:
                    setattr(self, attribute, -1)

    def _constructor_cleanup(self) -> None:
        if self.staging_fd >= 0:
            os.close(self.staging_fd)
            self.staging_fd = -1
        if self.staging_exists and self.parent_fd >= 0:
            try:
                os.rmdir(self.staging_name, dir_fd=self.parent_fd)
                self.staging_exists = False
            except FileNotFoundError:
                self.staging_exists = False
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1

    def abort(self) -> None:
        if self.published:
            self._close_descriptors()
            return
        if self.staging_fd >= 0:
            # STAGING_VERIFIED has already restricted the private directory to
            # 0500.  A fault hook at that exact pre-rename boundary must still
            # be able to remove it without exposing a public tree.
            os.fchmod(self.staging_fd, 0o700)
            for name in os.listdir(self.staging_fd):
                metadata = os.stat(name, dir_fd=self.staging_fd, follow_symlinks=False)
                require(stat.S_ISREG(metadata.st_mode), "abort flat regular staging")
                os.unlink(name, dir_fd=self.staging_fd)
            os.close(self.staging_fd)
            self.staging_fd = -1
        if self.staging_exists and self.parent_fd >= 0:
            os.rmdir(self.staging_name, dir_fd=self.parent_fd)
            self.staging_exists = False
            os.fsync(self.parent_fd)
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1
        self.phase = PublicationPhase.ABORTED

    def __enter__(self) -> "VerifiedPublisher":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.phase not in (PublicationPhase.CLOSED, PublicationPhase.ABORTED):
            self.abort()
