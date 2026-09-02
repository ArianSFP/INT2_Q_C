#!/usr/bin/env python3
"""Linux-only exclusive, no-follow, atomic directory publication for v1."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass


class PublicationError(RuntimeError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def _open_absolute_directory_no_symlinks(path: str) -> int:
    _check(os.name == "posix" and os.path.isabs(path), "absolute POSIX parent required")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in [value for value in path.split("/") if value]:
            _check(component not in (".", ".."), "canonical parent components")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        _check(stat.S_ISDIR(os.fstat(descriptor).st_mode), "parent directory")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, packet: bytes) -> None:
    view = memoryview(packet)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        _check(written > 0, "short publication write")
        offset += written


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _check(renameat2 is not None, "renameat2 is mandatory")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(parent_fd, source.encode(), parent_fd, destination.encode(), 1)  # RENAME_NOREPLACE
    if result != 0:
        error = ctypes.get_errno()
        raise PublicationError(f"renameat2(RENAME_NOREPLACE) failed: {os.strerror(error)}")


def _read_file_no_follow(directory_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        metadata = os.fstat(descriptor)
        _check(stat.S_ISREG(metadata.st_mode), "published artifact regular file")
        pieces: list[bytes] = []
        while True:
            value = os.read(descriptor, 1 << 20)
            if not value:
                break
            pieces.append(value)
        return b"".join(pieces)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class PublishedReceipt:
    output_path: str
    artifact_root_sha256: str
    completion_sha256: str
    artifact_rows: tuple[dict[str, object], ...]


class SafePublisher:
    """Build a hidden staging tree and atomically publish it without replacement."""

    def __init__(self, output_path: str, authenticated_source_root: str, fault_after: str | None = None) -> None:
        _check(os.name == "posix", "canonical publication requires Linux/POSIX")
        _check(len(authenticated_source_root) == 64, "authenticated source root")
        absolute = os.path.abspath(output_path)
        parent, final_name = os.path.split(absolute)
        _check(final_name not in ("", ".", "..") and "/" not in final_name, "final output name")
        self.parent_path = parent
        self.final_name = final_name
        self.parent_fd = _open_absolute_directory_no_symlinks(parent)
        self.staging_name = f".{final_name}.staging.{os.getpid()}.{secrets.token_hex(12)}"
        self.source_root = authenticated_source_root
        self.fault_after = fault_after
        self.artifacts: list[dict[str, object]] = []
        self.finished = False
        try:
            os.mkdir(self.staging_name, mode=0o700, dir_fd=self.parent_fd)
            self.staging_fd = os.open(
                self.staging_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self.parent_fd,
            )
            self._checkpoint("staging_created")
        except Exception:
            os.close(self.parent_fd)
            raise

    def _checkpoint(self, stage: str) -> None:
        if self.fault_after == stage:
            raise PublicationError(f"injected publication fault after {stage}")

    @staticmethod
    def _validate_name(name: str) -> None:
        _check(name not in ("", ".", "..") and "/" not in name and "\\" not in name, "flat artifact name")

    def write(self, name: str, packet: bytes) -> dict[str, object]:
        self._validate_name(name)
        _check(name not in ("ARTIFACTS.json", "COMPLETE"), "reserved artifact name")
        _check(not self.finished, "publisher already complete")
        _check(isinstance(packet, bytes), "artifact bytes")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(name, flags, 0o600, dir_fd=self.staging_fd)
        try:
            _write_all(descriptor, packet)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        row = {"name": name, "bytes": len(packet), "sha256": hashlib.sha256(packet).hexdigest()}
        self.artifacts.append(row)
        self._checkpoint(f"artifact:{name}")
        return row

    def _write_reserved(self, name: str, packet: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(name, flags, 0o600, dir_fd=self.staging_fd)
        try:
            _write_all(descriptor, packet)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def finish(self) -> PublishedReceipt:
        _check(not self.finished, "publisher already complete")
        ordered = sorted(self.artifacts, key=lambda row: str(row["name"]))
        manifest = {
            "schema": "silt-v1-unsealed-artifact-index",
            "authenticated_source_root": self.source_root,
            "artifacts": ordered,
            "result_frozen": False,
            "source_gain_claim": False,
        }
        manifest_packet = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self._write_reserved("ARTIFACTS.json", manifest_packet)
        self._checkpoint("artifact_index_fsynced")
        root_hasher = hashlib.sha256()
        root_hasher.update(b"SILT-V1-ARTIFACT-ROOT\0")
        for row in ordered:
            name = str(row["name"]).encode()
            root_hasher.update(len(name).to_bytes(4, "big"))
            root_hasher.update(name)
            root_hasher.update(bytes.fromhex(str(row["sha256"])))
            root_hasher.update(int(row["bytes"]).to_bytes(8, "big"))
        root_hasher.update(hashlib.sha256(manifest_packet).digest())
        artifact_root = root_hasher.hexdigest()
        completion = {
            "schema": "silt-v1-unsealed-completion",
            "artifact_root_sha256": artifact_root,
            "artifact_index_sha256": hashlib.sha256(manifest_packet).hexdigest(),
            "authenticated_source_root": self.source_root,
            "complete": True,
            "result_frozen": False,
        }
        completion_packet = (json.dumps(completion, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self._write_reserved(".COMPLETE.tmp", completion_packet)
        os.link(
            ".COMPLETE.tmp",
            "COMPLETE",
            src_dir_fd=self.staging_fd,
            dst_dir_fd=self.staging_fd,
            follow_symlinks=False,
        )
        os.unlink(".COMPLETE.tmp", dir_fd=self.staging_fd)
        os.fsync(self.staging_fd)
        self._checkpoint("completion_linked_and_directory_fsynced")
        _rename_noreplace(self.parent_fd, self.staging_name, self.final_name)
        os.fsync(self.parent_fd)
        self._checkpoint("published_and_parent_fsynced")
        final_fd = os.open(
            self.final_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=self.parent_fd,
        )
        try:
            for row in ordered:
                observed = _read_file_no_follow(final_fd, str(row["name"]))
                _check(len(observed) == row["bytes"] and hashlib.sha256(observed).hexdigest() == row["sha256"], "published artifact rehash")
            _check(_read_file_no_follow(final_fd, "ARTIFACTS.json") == manifest_packet, "published artifact index")
            observed_completion = _read_file_no_follow(final_fd, "COMPLETE")
            _check(observed_completion == completion_packet, "published completion")
        finally:
            os.close(final_fd)
        self.finished = True
        os.close(self.staging_fd)
        os.close(self.parent_fd)
        return PublishedReceipt(
            os.path.join(self.parent_path, self.final_name),
            artifact_root,
            hashlib.sha256(completion_packet).hexdigest(),
            tuple(ordered),
        )

    def abort(self) -> None:
        if self.finished:
            return
        try:
            for name in os.listdir(self.staging_fd):
                try:
                    os.unlink(name, dir_fd=self.staging_fd)
                except OSError:
                    pass
            os.close(self.staging_fd)
            os.rmdir(self.staging_name, dir_fd=self.parent_fd)
            os.fsync(self.parent_fd)
        finally:
            os.close(self.parent_fd)
            self.finished = True

    def __enter__(self) -> "SafePublisher":
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        if not self.finished:
            self.abort()

