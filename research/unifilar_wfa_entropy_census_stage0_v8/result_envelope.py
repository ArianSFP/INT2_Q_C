#!/usr/bin/env python3
"""Independent parent-marker completion verifier (standard library only).

``COMPLETE.json`` is a content record, never publication authority. A result
is consumable only when the retained parent contains the V8 commit marker and
that marker binds the final name, held directory inode, exact member table,
directory root, and expected source root.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any


MAX_COMMIT_MARKER_BYTES = 16 << 20
MAX_COMPLETE_RECORD_BYTES = 16 << 20
MAX_COMMITTED_MEMBER_BYTES = 1 << 40
MAX_BUFFERED_MEMBER_BYTES = 64 << 20
MAX_BUFFERED_AGGREGATE_BYTES = 256 << 20


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{label} digest geometry")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} digest encoding") from exc
    _require(len(raw) == 32, f"{label} digest width")
    return value


def _safe_name(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and value == Path(value).name
        and value not in {"", ".", ".."} and "/" not in value and "\\" not in value,
        f"{label} name",
    )
    return value


def _open_regular_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    capture: bool,
) -> tuple[int, bytes | None, os.stat_result, str]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), f"nonregular committed member: {name}")
        _require(0 <= int(before.st_size) <= maximum_bytes, f"committed member size bound: {name}")
        chunks = [] if capture else None
        digest = hashlib.sha256()
        observed = 0
        while observed < int(before.st_size):
            chunk = os.read(fd, min(1 << 20, int(before.st_size) - observed))
            _require(bool(chunk), f"short committed member read: {name}")
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
            observed += len(chunk)
        _require(os.read(fd, 1) == b"", f"committed member grew while reading: {name}")
        after = os.fstat(fd)
        identity = lambda info: (int(info.st_dev), int(info.st_ino), int(info.st_size), int(info.st_mtime_ns))
        _require(identity(before) == identity(after), f"committed member changed while reading: {name}")
        return fd, b"".join(chunks) if chunks is not None else None, before, digest.hexdigest()
    except BaseException:
        os.close(fd)
        raise


def _member_table(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and value, "commit member table")
    rows = []
    seen = set()
    for item in value:
        _require(isinstance(item, dict) and set(item) == {"name", "bytes", "sha256"}, "commit member row")
        name = _safe_name(item["name"], "commit member")
        _require(name not in seen, "duplicate commit member")
        _require(type(item["bytes"]) is int and item["bytes"] >= 0, "commit member byte count")
        digest = _digest(item["sha256"], "commit member")
        rows.append({"name": name, "bytes": item["bytes"], "sha256": digest})
        seen.add(name)
    expected = sorted(rows, key=lambda row: row["name"].encode("utf-8"))
    _require(rows == expected, "commit member table is not canonical UTF-8 order")
    _require("COMPLETE.json" in seen, "commit table omits COMPLETE.json")
    return rows


def _fd_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_mode), int(info.st_size),
        int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


class VerifiedOutputBundle:
    """Context-managed authenticated output whose descriptors outlive use.

    Verification never hands a caller a pathname to reopen. Every committed
    member fd, the final-directory fd, and the parent-marker fd remain held
    until ``__exit__``. Requested bytes are read only from those held fds and
    are capped, rehashed, and identity-checked before exposure.
    """

    def __init__(
        self,
        common: Any,
        parent: Any,
        final_name: str,
        expected_source_manifest_sha256: str,
        *,
        owns_parent: bool = False,
    ):
        _require(isinstance(parent, common.RetainedOutputParent), "retained output parent required")
        self.common = common
        self.parent = parent
        self.final_name = _safe_name(final_name, "final output")
        self.expected_source = _digest(expected_source_manifest_sha256, "expected source root")
        self.owns_parent = bool(owns_parent)
        # Snapshot the code-frozen caps before verification or buffering;
        # later ambient mutation cannot widen an active consumer's authority.
        self._per_member_buffer_cap = int(MAX_BUFFERED_MEMBER_BYTES)
        self._aggregate_buffer_cap = int(MAX_BUFFERED_AGGREGATE_BYTES)
        _require(0 < self._per_member_buffer_cap <= self._aggregate_buffer_cap, "verified buffer caps")
        self.marker_name = common.parent_commit_marker_name(self.final_name)
        self._marker_fd = -1
        self._directory_fd = -1
        self._member_fds: dict[str, int] = {}
        self._member_rows: dict[str, dict[str, Any]] = {}
        self._member_identities: dict[str, tuple[int, int, int, int, int, int]] = {}
        self._buffers: dict[str, bytes] = {}
        self._buffered_bytes = 0
        self._metadata: dict[str, Any] | None = None
        self._entered = False
        self._closed = False

    def __enter__(self) -> "VerifiedOutputBundle":
        _require(not self._entered and not self._closed, "verified bundle is single-use")
        try:
            self._verify_and_hold()
            self._entered = True
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_active(self) -> None:
        _require(self._entered and not self._closed, "verified bundle is not active")

    @property
    def metadata(self) -> dict[str, Any]:
        self._require_active()
        _require(self._metadata is not None, "verified bundle metadata absent")
        return {
            **self._metadata,
            "members": [dict(row) for row in self._metadata["members"]],
            "buffer_caps": {
                "per_member_bytes": self._per_member_buffer_cap,
                "aggregate_bytes": self._aggregate_buffer_cap,
            },
            "currently_buffered_bytes": self._buffered_bytes,
        }

    def _cache_verified_bytes(self, name: str, data: bytes) -> None:
        if name in self._buffers:
            _require(self._buffers[name] == data, "verified buffer identity")
            return
        _require(len(data) <= self._per_member_buffer_cap, f"verified member exceeds buffer cap: {name}")
        _require(
            self._buffered_bytes + len(data) <= self._aggregate_buffer_cap,
            "verified aggregate buffer cap",
        )
        self._buffers[name] = bytes(data)
        self._buffered_bytes += len(data)

    def read_member_bytes(self, name: str) -> bytes:
        """Return bytes read from the held authenticated fd, never a path."""
        self._require_active()
        name = _safe_name(name, "verified member")
        _require(name in self._member_fds, "unknown verified member")
        if name in self._buffers:
            return self._buffers[name]
        row = self._member_rows[name]
        size = int(row["bytes"])
        _require(size <= self._per_member_buffer_cap, f"verified member exceeds buffer cap: {name}")
        _require(self._buffered_bytes + size <= self._aggregate_buffer_cap, "verified aggregate buffer cap")
        fd = self._member_fds[name]
        before = os.fstat(fd)
        _require(_fd_identity(before) == self._member_identities[name], f"held member changed before consumption: {name}")
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        chunks = []
        observed = 0
        while observed < size:
            chunk = os.read(fd, min(1 << 20, size - observed))
            _require(bool(chunk), f"short held member consumption: {name}")
            digest.update(chunk)
            chunks.append(chunk)
            observed += len(chunk)
        _require(os.read(fd, 1) == b"", f"held member grew during consumption: {name}")
        after = os.fstat(fd)
        _require(_fd_identity(after) == self._member_identities[name], f"held member changed during consumption: {name}")
        _require(digest.hexdigest() == row["sha256"], f"held member digest changed during consumption: {name}")
        data = b"".join(chunks)
        _require(len(data) == size, f"held member consumption size: {name}")
        self._cache_verified_bytes(name, data)
        return self._buffers[name]

    def _verify_and_hold(self) -> None:
        self.parent.verify_stable()
        marker_fd, marker_bytes, marker_info, _marker_digest = _open_regular_at(
            self.parent.fd, self.marker_name, maximum_bytes=MAX_COMMIT_MARKER_BYTES, capture=True
        )
        self._marker_fd = marker_fd
        _require(marker_bytes is not None, "parent marker capture")
        marker = self.common.strict_json_loads(marker_bytes)
        required = {
            "schema", "status", "final_name", "transaction_id",
            "output_parent_authority_sha256", "parent_device", "parent_inode",
            "final_directory_device", "final_directory_inode",
            "commit_marker_device", "commit_marker_inode",
            "source_manifest_sha256", "members", "directory_root_sha256",
            "completion_sha256", "parent_commit_sha256",
        }
        _require(isinstance(marker, dict) and set(marker) == required, "parent commit marker fields")
        _require(marker["schema"] == "unifilar-wfa-parent-commit-v8", "parent commit marker schema")
        _require(marker["status"] == "PARENT_MARKER_COMMITTED", "parent commit marker status")
        self.common.verify_internal_seal(marker, "parent_commit_sha256")
        _require(marker["final_name"] == self.final_name, "parent marker final-name binding")
        transaction_id = marker["transaction_id"]
        _require(isinstance(transaction_id, str) and len(transaction_id) == 32, "parent marker transaction id")
        try:
            _require(len(bytes.fromhex(transaction_id)) == 16, "parent marker transaction width")
        except ValueError as exc:
            raise ValueError("parent marker transaction encoding") from exc
        _require(
            _digest(marker["output_parent_authority_sha256"], "output parent authority") == self.parent.authority_sha256,
            "parent marker authority mismatch",
        )
        _require(_digest(marker["source_manifest_sha256"], "marker source root") == self.expected_source, "marker source root mismatch")
        _digest(marker["directory_root_sha256"], "directory root")
        _digest(marker["completion_sha256"], "completion seal")
        for field in (
            "parent_device", "parent_inode", "final_directory_device", "final_directory_inode",
            "commit_marker_device", "commit_marker_inode",
        ):
            _require(type(marker[field]) is int and marker[field] >= 0, f"parent marker integer: {field}")
        _require(marker["parent_inode"] > 0 and marker["final_directory_inode"] > 0 and marker["commit_marker_inode"] > 0, "parent marker inode")
        parent_info = os.fstat(self.parent.fd)
        _require(
            (marker["parent_device"], marker["parent_inode"]) == (int(parent_info.st_dev), int(parent_info.st_ino)),
            "parent marker retained-parent inode mismatch",
        )
        _require(
            (marker["commit_marker_device"], marker["commit_marker_inode"])
            == (int(marker_info.st_dev), int(marker_info.st_ino)),
            "parent marker held-marker inode mismatch",
        )

        members = _member_table(marker["members"])
        expected_names = [row["name"] for row in members]
        self._directory_fd = os.open(
            self.final_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=self.parent.fd,
        )
        directory_info = os.fstat(self._directory_fd)
        _require(stat.S_ISDIR(directory_info.st_mode), "committed final is not a directory")
        _require(
            (marker["final_directory_device"], marker["final_directory_inode"])
            == (int(directory_info.st_dev), int(directory_info.st_ino)),
            "parent marker final-directory inode mismatch",
        )
        with os.scandir(self._directory_fd) as entries:
            actual_names = sorted(entry.name for entry in entries)
        _require(actual_names == sorted(expected_names), "committed directory exact membership")

        complete_bytes: bytes | None = None
        for row in members:
            capture = row["name"] == "COMPLETE.json"
            limit = MAX_COMPLETE_RECORD_BYTES if capture else MAX_COMMITTED_MEMBER_BYTES
            fd, data, info, observed_digest = _open_regular_at(
                self._directory_fd, row["name"], maximum_bytes=limit, capture=capture
            )
            self._member_fds[row["name"]] = fd
            self._member_rows[row["name"]] = dict(row)
            self._member_identities[row["name"]] = _fd_identity(info)
            _require(int(info.st_size) == row["bytes"], f"committed member size: {row['name']}")
            _require(observed_digest == row["sha256"], f"committed member digest: {row['name']}")
            if capture:
                _require(data is not None, "completion capture")
                complete_bytes = data
                self._cache_verified_bytes(row["name"], data)

        expected_root = self.common.committed_directory_root(self.expected_source, members)
        _require(marker["directory_root_sha256"] == expected_root, "committed directory root mismatch")
        _require(complete_bytes is not None, "completion content absent")
        complete = self.common.strict_json_loads(complete_bytes)
        _require(isinstance(complete, dict) and set(complete) == {
            "schema", "status", "source_manifest_sha256", "members", "completion_sha256"
        }, "completion content fields")
        _require(complete["schema"] == "unifilar-wfa-completion-v8" and complete["status"] == "COMPLETE_LAST", "completion content schema/status")
        self.common.verify_internal_seal(complete, "completion_sha256")
        _require(complete["source_manifest_sha256"] == self.expected_source, "completion source root mismatch")
        _require(isinstance(complete["members"], list), "completion member list")
        completion_members = sorted(complete["members"], key=lambda row: row["name"].encode("utf-8"))
        marker_without_complete = [row for row in members if row["name"] != "COMPLETE.json"]
        _require(completion_members == marker_without_complete, "completion/parent-marker member mismatch")
        _require(complete["completion_sha256"] == marker["completion_sha256"], "completion seal/marker mismatch")

        current_directory = os.stat(self.final_name, dir_fd=self.parent.fd, follow_symlinks=False)
        current_marker = os.stat(self.marker_name, dir_fd=self.parent.fd, follow_symlinks=False)
        _require(
            (int(current_directory.st_dev), int(current_directory.st_ino))
            == (int(directory_info.st_dev), int(directory_info.st_ino)),
            "final directory name substituted during verification",
        )
        _require(
            (int(current_marker.st_dev), int(current_marker.st_ino))
            == (int(marker_info.st_dev), int(marker_info.st_ino)),
            "parent marker name substituted during verification",
        )
        self.parent.verify_stable()
        self._metadata = {
            "status": "PASS_PARENT_MARKER_COMMITTED_BUNDLE_HELD",
            "final_name": self.final_name,
            "transaction_id": transaction_id,
            "source_manifest_sha256": self.expected_source,
            "members": members,
            "directory_root_sha256": expected_root,
            "completion_sha256": marker["completion_sha256"],
            "parent_commit_sha256": marker["parent_commit_sha256"],
            "parent_commit_marker_name": self.marker_name,
            "all_member_directory_and_marker_descriptors_retained": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        for fd in reversed(tuple(self._member_fds.values())):
            try:
                os.close(fd)
            except OSError:
                pass
        self._member_fds.clear()
        if self._directory_fd >= 0:
            try:
                os.close(self._directory_fd)
            except OSError:
                pass
            self._directory_fd = -1
        if self._marker_fd >= 0:
            try:
                os.close(self._marker_fd)
            except OSError:
                pass
            self._marker_fd = -1
        if self.owns_parent:
            try:
                self.parent.close()
            except OSError:
                pass
        self._closed = True


def verify_completed_under_parent(
    common: Any,
    parent: Any,
    final_name: str,
    *,
    expected_source_manifest_sha256: str,
) -> VerifiedOutputBundle:
    """Return a single-use context manager; verification occurs on entry."""
    return VerifiedOutputBundle(
        common, parent, final_name, expected_source_manifest_sha256, owns_parent=False
    )


def verify_completed_directory(
    common: Any,
    path: Path,
    *,
    output_parent_authority_sha256: str,
    expected_source_manifest_sha256: str,
) -> VerifiedOutputBundle:
    """Pin the parent immediately and return an owning verified-bundle context."""
    if not path.is_absolute():
        raise ValueError("result path must be absolute")
    final_name = _safe_name(path.name, "final output")
    parent = common.RetainedOutputParent.open_path_source_only(
        path.parent, _digest(output_parent_authority_sha256, "output parent authority")
    )
    try:
        return VerifiedOutputBundle(
            common,
            parent,
            final_name,
            expected_source_manifest_sha256,
            owns_parent=True,
        )
    except BaseException:
        parent.close()
        raise
