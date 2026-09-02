#!/usr/bin/env python3
"""Retained, no-follow source closure for TACTIC N18 v6.

This module is standard-library only and inert at import. Entry points load
its externally manifest-pinned bytes explicitly, which works under CPython
isolated mode without adding the package directory to ``sys.path``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "tactic-actual-coarse-n18-v6-source-manifest-v1"
MANIFEST_STATUS = "SEALED_SOURCE_ONLY_AWAITING_SOURCE_FREE_SMOKE"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MAX_SOURCE_BYTES = 4 * (1 << 20)


class SourceAuthError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuthError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in rows:
            require(key not in output, f"{label}: duplicate JSON key {key!r}")
            output[key] = value
        return output

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                SourceAuthError(f"{label}: nonfinite {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceAuthError(f"{label}: JSON: {error}") from error
    require(isinstance(value, dict), f"{label}: JSON object")
    return value


def _reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode),
                f"{label}: symlink component {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode,
        metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _pread_exact(descriptor: int, size: int, label: str) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1 << 20, size - offset), offset)
        require(bool(chunk), f"{label}: premature EOF")
        chunks.append(chunk)
        offset += len(chunk)
    require(os.pread(descriptor, 1, size) == b"", f"{label}: trailing bytes")
    return b"".join(chunks)


@dataclass
class HeldMember:
    directory_fd: int
    name: str
    descriptor: int
    identity: tuple[int, int, int, int, int, int, int]
    data: bytes
    digest: str

    @classmethod
    def open(
        cls,
        directory_fd: int,
        name: str,
        *,
        expected_bytes: int,
        expected_sha256: str,
        label: str,
    ) -> "HeldMember":
        require(isinstance(name, str) and name and name not in {".", ".."}
                and "/" not in name and "\\" not in name,
                f"{label}: safe name")
        require(type(expected_bytes) is int and
                0 < expected_bytes <= MAX_SOURCE_BYTES,
                f"{label}: byte bound")
        require(isinstance(expected_sha256, str) and
                HEX64.fullmatch(expected_sha256) is not None,
                f"{label}: SHA-256 syntax")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_BINARY", 0) |
            getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            metadata = os.fstat(descriptor)
            require(stat.S_ISREG(metadata.st_mode) and
                    metadata.st_size == expected_bytes and
                    metadata.st_nlink == 1,
                    f"{label}: regular sole-link identity")
            data = _pread_exact(descriptor, expected_bytes, label)
            require(sha256(data) == expected_sha256,
                    f"{label}: content digest")
            return cls(
                directory_fd, name, descriptor, _identity(metadata), data,
                expected_sha256,
            )
        except BaseException:
            os.close(descriptor)
            raise

    def verify_final(self) -> None:
        require(_identity(os.fstat(self.descriptor)) == self.identity,
                f"source member changed: {self.name}")
        named = os.stat(self.name, dir_fd=self.directory_fd,
                        follow_symlinks=False)
        require(_identity(named) == self.identity,
                f"source member name rebound: {self.name}")
        require(_pread_exact(self.descriptor, len(self.data), self.name) ==
                self.data,
                f"source member bytes changed: {self.name}")

    def close(self) -> None:
        os.close(self.descriptor)


class HeldSourcePackage:
    def __init__(
        self,
        package_dir: Path,
        expected_manifest_sha256: str,
        *,
        executing_path: Path,
    ) -> None:
        require(package_dir.is_absolute() and executing_path.is_absolute(),
                "source package/executable absolute paths")
        require(HEX64.fullmatch(expected_manifest_sha256) is not None,
                "external source-manifest SHA-256")
        _reject_symlink_chain(package_dir, "source package")
        self.path = package_dir
        self.directory_fd = os.open(
            os.fspath(package_dir),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0),
        )
        self.directory_identity = _identity(os.fstat(self.directory_fd))
        self.members: dict[str, HeldMember] = {}
        try:
            manifest_metadata = os.stat(
                "SOURCE_MANIFEST.json", dir_fd=self.directory_fd,
                follow_symlinks=False,
            )
            self.manifest = HeldMember.open(
                self.directory_fd, "SOURCE_MANIFEST.json",
                expected_bytes=int(manifest_metadata.st_size),
                expected_sha256=expected_manifest_sha256,
                label="source manifest",
            )
            record = strict_json(self.manifest.data, "source manifest")
            require(set(record) == {
                "schema", "status", "source_root_sha256", "members",
                "claim_boundary", "access_attestation",
            }, "source manifest exact schema")
            require(record["schema"] == MANIFEST_SCHEMA and
                    record["status"] == MANIFEST_STATUS,
                    "source manifest schema/status")
            require(record["claim_boundary"] ==
                    "source-free mechanics only; Qwen pilot, universal-tail, MSE, fine-code and inference-HBM claims require separate authorization and audit",
                    "source manifest claim boundary")
            require(record["access_attestation"] == {
                "runpod_accessed": False,
                "qwen_or_model_payload_accessed": False,
                "cuda_or_cupy_initialized_during_source_build": False,
                "network_accessed": False,
            }, "source-build access attestation")
            rows = record["members"]
            require(isinstance(rows, list) and rows,
                    "source manifest member rows")
            names = []
            observed = []
            inode_domain = {
                (self.manifest.identity[0], self.manifest.identity[1])
            }
            for row in rows:
                require(isinstance(row, dict) and
                        set(row) == {"name", "bytes", "sha256"},
                        "source manifest member row")
                name = row["name"]
                require(name != "SOURCE_MANIFEST.json" and name not in names,
                        "source manifest unique nonself member")
                held = HeldMember.open(
                    self.directory_fd, name,
                    expected_bytes=row["bytes"],
                    expected_sha256=row["sha256"],
                    label=f"source member {name}",
                )
                inode = (held.identity[0], held.identity[1])
                require(inode not in inode_domain,
                        f"source member inode alias: {name}")
                inode_domain.add(inode)
                self.members[name] = held
                names.append(name)
                observed.append({
                    "name": name, "bytes": len(held.data),
                    "sha256": held.digest,
                })
            require(names == sorted(names, key=lambda item: item.encode("utf-8")),
                    "source manifest canonical member order")
            require(record["source_root_sha256"] ==
                    sha256(canonical_json(observed)),
                    "source manifest canonical member root")
            actual_entries = list(os.scandir(self.directory_fd))
            require({entry.name for entry in actual_entries} ==
                    set(names) | {"SOURCE_MANIFEST.json"} and
                    all(entry.is_file(follow_symlinks=False)
                        for entry in actual_entries),
                    "source package exact regular-file closure")
            executable = os.lstat(executing_path)
            executable_name = executing_path.name
            require(executable_name in self.members and
                    (executable.st_dev, executable.st_ino) ==
                    (self.members[executable_name].identity[0],
                     self.members[executable_name].identity[1]),
                    "executing entry/source-manifest inode binding")
            self.executing_entry_name = executable_name
            self.record = record
            self.manifest_sha256 = expected_manifest_sha256
            self.source_root_sha256 = record["source_root_sha256"]
        except BaseException:
            self.close()
            raise

    @property
    def sources(self) -> dict[str, bytes]:
        return {name: held.data for name, held in self.members.items()}

    def receipt(self) -> dict[str, Any]:
        return {
            "source_manifest_sha256": self.manifest_sha256,
            "source_root_sha256": self.source_root_sha256,
            "member_hashes": {
                name: self.members[name].digest for name in sorted(self.members)
            },
            "retained_no_follow_descriptors": True,
            "executing_entry_inode_bound": True,
            "executing_entry_name": self.executing_entry_name,
        }

    def verify_final(self) -> None:
        require(_identity(os.fstat(self.directory_fd)) ==
                self.directory_identity,
                "source package directory changed")
        require(_identity(os.stat(self.path, follow_symlinks=False)) ==
                self.directory_identity,
                "source package path rebound")
        self.manifest.verify_final()
        for held in self.members.values():
            held.verify_final()
        require({entry.name for entry in os.scandir(self.directory_fd)} ==
                set(self.members) | {"SOURCE_MANIFEST.json"},
                "source package member set changed")

    def close(self) -> None:
        members = getattr(self, "members", {})
        for held in list(members.values()):
            try:
                held.close()
            except OSError:
                pass
        members.clear()
        manifest = getattr(self, "manifest", None)
        if manifest is not None:
            try:
                manifest.close()
            except OSError:
                pass
            self.manifest = None
        descriptor = getattr(self, "directory_fd", None)
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self.directory_fd = None

    def __enter__(self) -> "HeldSourcePackage":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
