#!/usr/bin/env python3
"""Complete-or-absent Linux publication for finite TACTIC results."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping


class PublishError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublishError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True,
        allow_nan=False) + "\n").encode("ascii")


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode,
        metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _write_member(directory_fd: int, name: str, payload: bytes) -> dict[str, Any]:
    require(isinstance(name, str) and name and name not in {".", ".."}
            and "/" not in name and "\\" not in name,
            "safe output member name")
    require(type(payload) is bytes, "output payload bytes")
    descriptor = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
        getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600, dir_fd=directory_fd)
    try:
        cursor = 0
        while cursor < len(payload):
            written = os.write(descriptor, payload[cursor:])
            require(written > 0, "output short write")
            cursor += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode) and
                metadata.st_size == len(payload) and metadata.st_nlink == 1,
                "output member identity")
    finally:
        os.close(descriptor)
    return {"name": name, "bytes": len(payload), "sha256": sha256(payload)}


def _rehash(directory_fd: int, row: dict[str, Any], label: str) -> None:
    descriptor = os.open(
        row["name"], os.O_RDONLY | getattr(os, "O_BINARY", 0) |
        getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                before.st_size == row["bytes"], f"{label}: identity")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label}: short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"" and
                sha256(b"".join(chunks)) == row["sha256"],
                f"{label}: bytes")
        after = os.fstat(descriptor)
        named = os.stat(row["name"], dir_fd=directory_fd,
                        follow_symlinks=False)
        require(_identity(after) == _identity(before) == _identity(named),
                f"{label}: name/inode binding")
    finally:
        os.close(descriptor)


def _rename_noreplace(directory_fd: int, old: str, new: str) -> None:
    require(os.name == "posix", "Linux atomic publication only")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        directory_fd, os.fsencode(old), directory_fd, os.fsencode(new), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise PublishError("output namespace already exists")
        raise PublishError(f"renameat2 RENAME_NOREPLACE: errno {error}")


def _cleanup_owned_staging(parent_fd: int, directory_fd: int,
                           staging_name: str,
                           directory_identity: tuple[int, int, int]) -> None:
    current = os.fstat(directory_fd)
    named = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
    require((current.st_dev, current.st_ino, stat.S_IFMT(current.st_mode)) ==
            directory_identity ==
            (named.st_dev, named.st_ino, stat.S_IFMT(named.st_mode)),
            "staging ownership before cleanup")
    for entry in list(os.scandir(directory_fd)):
        metadata = entry.stat(follow_symlinks=False)
        require(entry.is_file(follow_symlinks=False) and
                metadata.st_nlink == 1,
                "cleanup only owned sole-link regular files")
        os.unlink(entry.name, dir_fd=directory_fd)
    os.fsync(directory_fd)
    os.rmdir(staging_name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def publish_atomic(output_dir: Path, members: Mapping[str, bytes],
                   completion: Mapping[str, Any]) -> dict[str, Any]:
    require(os.name == "posix" and output_dir.is_absolute(),
            "absolute POSIX output")
    require("COMPLETE.json" not in members, "terminal completion reserved")
    parent = output_dir.parent
    require(parent.resolve(strict=True) == parent and
            stat.S_ISDIR(os.lstat(parent).st_mode),
            "canonical output parent")
    require(not os.path.lexists(output_dir), "output must be absent")
    parent_fd = os.open(
        os.fspath(parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
        getattr(os, "O_NOFOLLOW", 0))
    directory_fd = -1
    final_fd = -1
    staging_name = (
        f".{output_dir.name}.partial.{os.getpid()}.{secrets.token_hex(8)}")
    staging_created = False
    renamed = False
    try:
        os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        staging_created = True
        directory_fd = os.open(
            staging_name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        metadata = os.fstat(directory_fd)
        staging_identity = (
            metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode))
        require(len(members) == len(set(members)), "unique output members")
        rows = [
            _write_member(directory_fd, name, members[name])
            for name in sorted(members, key=lambda item: item.encode("utf-8"))
        ]
        complete = dict(completion)
        complete["members"] = rows
        complete["members_root_sha256"] = sha256(canonical_json(rows))
        complete["completion_claim_sha256"] = sha256(canonical_json(complete))
        pending_name = ".COMPLETE.pending"
        pending_row = _write_member(
            directory_fd, pending_name, pretty_json(complete))
        os.fsync(directory_fd)
        require({entry.name for entry in os.scandir(directory_fd)} ==
                {row["name"] for row in rows} | {pending_name},
                "exact staging member closure")
        for row in rows:
            _rehash(directory_fd, row, f"staging {row['name']}")
        _rehash(directory_fd, pending_row, "staging completion")
        _rename_noreplace(parent_fd, staging_name, output_dir.name)
        renamed = True
        os.fsync(parent_fd)
        final_fd = os.open(
            output_dir.name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        final_metadata = os.fstat(final_fd)
        require((final_metadata.st_dev, final_metadata.st_ino,
                 stat.S_IFMT(final_metadata.st_mode)) == staging_identity,
                "published directory inode rebind")
        for row in rows:
            _rehash(final_fd, row, f"published {row['name']}")
        _rehash(final_fd, pending_row, "published completion pending")
        _rename_noreplace(final_fd, pending_name, "COMPLETE.json")
        os.fsync(final_fd)
        terminal_row = {**pending_row, "name": "COMPLETE.json"}
        _rehash(final_fd, terminal_row, "published terminal completion")
        require({entry.name for entry in os.scandir(final_fd)} ==
                {row["name"] for row in rows} | {"COMPLETE.json"},
                "terminal exact member closure")
        os.fsync(parent_fd)
        return {
            "output_directory": os.fspath(output_dir),
            "members": rows,
            "complete": terminal_row,
            "atomic_directory_rename_noreplace": True,
            "terminal_completion_rename_noreplace": True,
            "staging_rehashed_before_publication": True,
            "published_members_rehashed_before_completion": True,
        }
    except BaseException as original:
        if staging_created and not renamed and directory_fd >= 0:
            try:
                current = os.fstat(directory_fd)
                _cleanup_owned_staging(
                    parent_fd, directory_fd, staging_name,
                    (current.st_dev, current.st_ino,
                     stat.S_IFMT(current.st_mode)))
                staging_created = False
            except BaseException as cleanup_error:
                raise PublishError(
                    f"publication failed and owned staging cleanup failed: "
                    f"{cleanup_error}") from original
        raise
    finally:
        if final_fd >= 0:
            os.close(final_fd)
        if directory_fd >= 0:
            os.close(directory_fd)
        os.close(parent_fd)
