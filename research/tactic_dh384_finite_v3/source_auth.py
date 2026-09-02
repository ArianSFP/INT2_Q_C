#!/usr/bin/env python3
"""Retained no-follow source closure for finite TACTIC-DH384 v3."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "tactic-dh384-finite-v3-source-manifest-v1"
MANIFEST_STATUS = "SEALED_SOURCE_ONLY_AWAITING_INDEPENDENT_REVIEW"
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
            require(key not in output, f"{label}: duplicate key")
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


def reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode),
                f"{label}: symlink component {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode,
        metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def pread_exact(descriptor: int, size: int, label: str) -> bytes:
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
    held_identity: tuple[int, int, int, int, int, int, int]
    data: bytes
    digest: str

    @classmethod
    def open(cls, directory_fd: int, name: str, *, expected_bytes: int,
             expected_sha256: str, label: str) -> "HeldMember":
        require(isinstance(name, str) and name and name not in {".", ".."}
                and "/" not in name and "\\" not in name,
                f"{label}: safe name")
        require(type(expected_bytes) is int and
                0 < expected_bytes <= MAX_SOURCE_BYTES,
                f"{label}: byte bound")
        require(isinstance(expected_sha256, str) and
                HEX64.fullmatch(expected_sha256) is not None,
                f"{label}: digest syntax")
        descriptor = os.open(
            name, os.O_RDONLY | getattr(os, "O_BINARY", 0) |
            getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        try:
            metadata = os.fstat(descriptor)
            require(stat.S_ISREG(metadata.st_mode) and
                    metadata.st_size == expected_bytes and
                    metadata.st_nlink == 1,
                    f"{label}: regular sole-link identity")
            data = pread_exact(descriptor, expected_bytes, label)
            require(sha256(data) == expected_sha256,
                    f"{label}: digest")
            return cls(directory_fd, name, descriptor, identity(metadata),
                       data, expected_sha256)
        except BaseException:
            os.close(descriptor)
            raise

    def verify_final(self) -> None:
        require(identity(os.fstat(self.descriptor)) == self.held_identity,
                f"source member changed: {self.name}")
        named = os.stat(self.name, dir_fd=self.directory_fd,
                        follow_symlinks=False)
        require(identity(named) == self.held_identity,
                f"source member rebound: {self.name}")
        require(pread_exact(self.descriptor, len(self.data), self.name) ==
                self.data, f"source bytes changed: {self.name}")

    def close(self) -> None:
        os.close(self.descriptor)


class HeldSourcePackage:
    def __init__(self, package_dir: Path, expected_manifest_sha256: str,
                 *, executing_path: Path) -> None:
        require(package_dir.is_absolute() and executing_path.is_absolute(),
                "absolute package and executable")
        require(HEX64.fullmatch(expected_manifest_sha256) is not None,
                "external source manifest digest")
        reject_symlink_chain(package_dir, "source package")
        self.path = package_dir
        self.directory_fd = os.open(
            os.fspath(package_dir), os.O_RDONLY |
            getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        self.directory_identity = identity(os.fstat(self.directory_fd))
        self.members: dict[str, HeldMember] = {}
        self.manifest: HeldMember | None = None
        try:
            metadata = os.stat("SOURCE_MANIFEST.json", dir_fd=self.directory_fd,
                               follow_symlinks=False)
            self.manifest = HeldMember.open(
                self.directory_fd, "SOURCE_MANIFEST.json",
                expected_bytes=metadata.st_size,
                expected_sha256=expected_manifest_sha256,
                label="source manifest")
            record = strict_json(self.manifest.data, "source manifest")
            require(set(record) == {
                "schema", "status", "source_root_sha256", "members",
                "claim_boundary", "access_attestation",
            }, "source manifest exact schema")
            require(record["schema"] == MANIFEST_SCHEMA and
                    record["status"] == MANIFEST_STATUS,
                    "source manifest schema/status")
            require(record["access_attestation"] == {
                "cuda_or_cupy_initialized_during_source_build": False,
                "network_accessed": False,
                "qwen_or_model_payload_accessed": False,
                "runpod_accessed": False,
                "v6_live_result_accessed": False,
            }, "source-build access attestation")
            require(record["claim_boundary"] ==
                    "source-only finite-code implementation; launch requires an externally hashed review receipt binding this source and one audited v6 result",
                    "source claim boundary")
            rows = record["members"]
            require(isinstance(rows, list) and rows,
                    "source manifest rows")
            names: list[str] = []
            observed: list[dict[str, Any]] = []
            inodes = {(self.manifest.held_identity[0],
                       self.manifest.held_identity[1])}
            for row in rows:
                require(isinstance(row, dict) and
                        set(row) == {"name", "bytes", "sha256"},
                        "source member row")
                name = row["name"]
                require(name != "SOURCE_MANIFEST.json" and name not in names,
                        "unique nonself source member")
                held = HeldMember.open(
                    self.directory_fd, name,
                    expected_bytes=row["bytes"],
                    expected_sha256=row["sha256"],
                    label=f"source member {name}")
                inode = (held.held_identity[0], held.held_identity[1])
                require(inode not in inodes, f"source inode alias: {name}")
                inodes.add(inode)
                self.members[name] = held
                names.append(name)
                observed.append({
                    "name": name, "bytes": len(held.data),
                    "sha256": held.digest,
                })
            require(names == sorted(names, key=lambda item: item.encode("utf-8")),
                    "canonical member order")
            require(record["source_root_sha256"] ==
                    sha256(canonical_json(observed)),
                    "canonical member root")
            entries = list(os.scandir(self.directory_fd))
            require({entry.name for entry in entries} ==
                    set(names) | {"SOURCE_MANIFEST.json"} and
                    all(entry.is_file(follow_symlinks=False)
                        for entry in entries),
                    "exact regular source closure")
            executable = os.lstat(executing_path)
            name = executing_path.name
            require(name in self.members and
                    (executable.st_dev, executable.st_ino) ==
                    (self.members[name].held_identity[0],
                     self.members[name].held_identity[1]),
                    "executing entry inode binding")
            self.record = record
            self.manifest_sha256 = expected_manifest_sha256
            self.source_root_sha256 = record["source_root_sha256"]
            self.executing_entry_name = name
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
        require(identity(os.fstat(self.directory_fd)) == self.directory_identity,
                "source directory changed")
        require(identity(os.stat(self.path, follow_symlinks=False)) ==
                self.directory_identity, "source directory path rebound")
        require(self.manifest is not None, "source manifest retained")
        self.manifest.verify_final()
        for held in self.members.values():
            held.verify_final()
        require({entry.name for entry in os.scandir(self.directory_fd)} ==
                set(self.members) | {"SOURCE_MANIFEST.json"},
                "source member set changed")

    def close(self) -> None:
        for held in list(getattr(self, "members", {}).values()):
            try:
                held.close()
            except OSError:
                pass
        getattr(self, "members", {}).clear()
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
