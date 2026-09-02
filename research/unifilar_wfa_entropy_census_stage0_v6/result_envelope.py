#!/usr/bin/env python3
"""Independent parent-marker completion verifier (standard library only).

``COMPLETE.json`` is a content record, never publication authority. A result
is consumable only when the retained parent contains the V6 commit marker and
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


def verify_completed_under_parent(
    common: Any,
    parent: Any,
    final_name: str,
    *,
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify and retain all authority through ``parent.fd`` until return."""
    _require(isinstance(parent, common.RetainedOutputParent), "retained output parent required")
    final_name = _safe_name(final_name, "final output")
    expected_source = _digest(expected_source_manifest_sha256, "expected source root")
    parent.verify_stable()
    marker_name = common.parent_commit_marker_name(final_name)
    marker_fd = -1
    directory_fd = -1
    member_fds: list[int] = []
    try:
        marker_fd, marker_bytes, marker_info, _marker_digest = _open_regular_at(
            parent.fd, marker_name, maximum_bytes=MAX_COMMIT_MARKER_BYTES, capture=True
        )
        _require(marker_bytes is not None, "parent marker capture")
        marker = common.strict_json_loads(marker_bytes)
        required = {
            "schema", "status", "final_name", "transaction_id",
            "output_parent_authority_sha256", "parent_device", "parent_inode",
            "final_directory_device", "final_directory_inode",
            "commit_marker_device", "commit_marker_inode",
            "source_manifest_sha256", "members", "directory_root_sha256",
            "completion_sha256", "parent_commit_sha256",
        }
        _require(isinstance(marker, dict) and set(marker) == required, "parent commit marker fields")
        _require(marker["schema"] == "unifilar-wfa-parent-commit-v6", "parent commit marker schema")
        _require(marker["status"] == "PARENT_MARKER_COMMITTED", "parent commit marker status")
        common.verify_internal_seal(marker, "parent_commit_sha256")
        _require(marker["final_name"] == final_name, "parent marker final-name binding")
        transaction_id = marker["transaction_id"]
        _require(isinstance(transaction_id, str) and len(transaction_id) == 32, "parent marker transaction id")
        try:
            _require(len(bytes.fromhex(transaction_id)) == 16, "parent marker transaction width")
        except ValueError as exc:
            raise ValueError("parent marker transaction encoding") from exc
        _require(
            _digest(marker["output_parent_authority_sha256"], "output parent authority") == parent.authority_sha256,
            "parent marker authority mismatch",
        )
        _require(_digest(marker["source_manifest_sha256"], "marker source root") == expected_source, "marker source root mismatch")
        _digest(marker["directory_root_sha256"], "directory root")
        _digest(marker["completion_sha256"], "completion seal")
        for field in (
            "parent_device", "parent_inode", "final_directory_device", "final_directory_inode",
            "commit_marker_device", "commit_marker_inode",
        ):
            _require(type(marker[field]) is int and marker[field] >= 0, f"parent marker integer: {field}")
        _require(marker["parent_inode"] > 0 and marker["final_directory_inode"] > 0 and marker["commit_marker_inode"] > 0, "parent marker inode")
        parent_info = os.fstat(parent.fd)
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
        directory_fd = os.open(
            final_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent.fd,
        )
        directory_info = os.fstat(directory_fd)
        _require(stat.S_ISDIR(directory_info.st_mode), "committed final is not a directory")
        _require(
            (marker["final_directory_device"], marker["final_directory_inode"])
            == (int(directory_info.st_dev), int(directory_info.st_ino)),
            "parent marker final-directory inode mismatch",
        )
        with os.scandir(directory_fd) as entries:
            actual_names = sorted(entry.name for entry in entries)
        _require(actual_names == sorted(expected_names), "committed directory exact membership")

        member_bytes: dict[str, bytes] = {}
        for row in members:
            capture = row["name"] == "COMPLETE.json"
            limit = MAX_COMPLETE_RECORD_BYTES if capture else MAX_COMMITTED_MEMBER_BYTES
            fd, data, _info, observed_digest = _open_regular_at(
                directory_fd, row["name"], maximum_bytes=limit, capture=capture
            )
            member_fds.append(fd)
            _require(int(_info.st_size) == row["bytes"], f"committed member size: {row['name']}")
            _require(observed_digest == row["sha256"], f"committed member digest: {row['name']}")
            if data is not None:
                member_bytes[row["name"]] = data

        expected_root = common.committed_directory_root(expected_source, members)
        _require(marker["directory_root_sha256"] == expected_root, "committed directory root mismatch")
        complete = common.strict_json_loads(member_bytes["COMPLETE.json"])
        _require(isinstance(complete, dict) and set(complete) == {
            "schema", "status", "source_manifest_sha256", "members", "completion_sha256"
        }, "completion content fields")
        _require(complete["schema"] == "unifilar-wfa-completion-v6" and complete["status"] == "COMPLETE_LAST", "completion content schema/status")
        common.verify_internal_seal(complete, "completion_sha256")
        _require(complete["source_manifest_sha256"] == expected_source, "completion source root mismatch")
        _require(isinstance(complete["members"], list), "completion member list")
        completion_members = sorted(complete["members"], key=lambda row: row["name"].encode("utf-8"))
        marker_without_complete = [row for row in members if row["name"] != "COMPLETE.json"]
        _require(completion_members == marker_without_complete, "completion/parent-marker member mismatch")
        _require(complete["completion_sha256"] == marker["completion_sha256"], "completion seal/marker mismatch")

        # Do not verify and then silently re-resolve substituted names.
        for fd in member_fds:
            _require(stat.S_ISREG(os.fstat(fd).st_mode), "held committed member ceased to be regular")
        current_directory = os.stat(final_name, dir_fd=parent.fd, follow_symlinks=False)
        current_marker = os.stat(marker_name, dir_fd=parent.fd, follow_symlinks=False)
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
        parent.verify_stable()
        return {
            "status": "PASS_PARENT_MARKER_COMMITTED_ENVELOPE",
            "final_name": final_name,
            "transaction_id": transaction_id,
            "source_manifest_sha256": expected_source,
            "members": members,
            "directory_root_sha256": expected_root,
            "completion_sha256": marker["completion_sha256"],
            "parent_commit_sha256": marker["parent_commit_sha256"],
            "parent_commit_marker_name": marker_name,
        }
    finally:
        for fd in reversed(member_fds):
            os.close(fd)
        if directory_fd >= 0:
            os.close(directory_fd)
        if marker_fd >= 0:
            os.close(marker_fd)


def verify_completed_directory(
    common: Any,
    path: Path,
    *,
    output_parent_authority_sha256: str,
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    """Path convenience wrapper that immediately pins the parent descriptor."""
    if not path.is_absolute():
        raise ValueError("result path must be absolute")
    final_name = _safe_name(path.name, "final output")
    parent = common.RetainedOutputParent.open_path_source_only(
        path.parent, _digest(output_parent_authority_sha256, "output parent authority")
    )
    try:
        return verify_completed_under_parent(
            common,
            parent,
            final_name,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
        )
    finally:
        parent.close()
