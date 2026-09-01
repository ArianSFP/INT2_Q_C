#!/usr/bin/env python3
"""Reference contract for the *external* independently pinned dispatcher.

This module cannot grant payload authority and is never a numeric entrypoint.
It exists so source-free hostile tests can exercise the exact snapshot closure
that a later independent audit package must duplicate while hard-coding the
producer-manifest and independent-review digests outside this package.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class DispatchContractError(RuntimeError):
    pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise DispatchContractError(message)


def _strict_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            _require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise DispatchContractError(f"nonfinite JSON constant: {value}")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject_constant)
    except Exception as exc:
        raise DispatchContractError(f"invalid JSON: {exc}") from exc
    _require(isinstance(value, dict), "JSON root must be object")
    return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_external_pin(data: bytes, externally_pinned_sha256: str, label: str) -> None:
    _require(isinstance(externally_pinned_sha256, str) and len(externally_pinned_sha256) == 64, f"{label} external pin geometry")
    _require(_sha(data) == externally_pinned_sha256, f"{label} external pin mismatch")


def require_exact_declared_members(actual: set[str], declared: set[str]) -> None:
    _require(actual == declared, f"undeclared/missing package members: actual={sorted(actual)} declared={sorted(declared)}")


def _absolute_parts(path: Path) -> tuple[str, tuple[str, ...]]:
    text = os.fspath(path)
    _require(os.path.isabs(text), "dispatcher paths must be absolute")
    drive, tail = os.path.splitdrive(text)
    parts = tuple(part for part in tail.replace("\\", "/").split("/") if part)
    _require(parts and all(part not in {".", ".."} for part in parts), "noncanonical dispatcher path")
    return drive + os.path.sep if drive else os.path.sep, parts


@dataclass
class SnapshotFile:
    path: Path
    fd: int
    ancestor_fds: list[int]
    identity: tuple[int, int, int, int]
    data: bytes
    sha256: str

    def verify_stable(self) -> None:
        info = os.fstat(self.fd)
        _require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) == self.identity, f"snapshot changed: {self.path}")

    def close(self) -> None:
        os.close(self.fd)
        for fd in reversed(self.ancestor_fds):
            os.close(fd)
        self.ancestor_fds.clear()


def open_snapshot(path: Path) -> SnapshotFile:
    anchor, parts = _absolute_parts(path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    binary = getattr(os, "O_BINARY", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    ancestors: list[int] = []
    if os.name != "nt" and os.open in os.supports_dir_fd:
        parent = os.open(anchor, os.O_RDONLY | directory | nofollow)
        ancestors.append(parent)
        try:
            for component in parts[:-1]:
                child = os.open(component, os.O_RDONLY | directory | nofollow, dir_fd=parent)
                _require(stat.S_ISDIR(os.fstat(child).st_mode), "snapshot ancestor is not directory")
                ancestors.append(child)
                parent = child
            fd = os.open(parts[-1], os.O_RDONLY | binary | nofollow, dir_fd=parent)
        except Exception:
            for item in reversed(ancestors):
                os.close(item)
            raise
    else:
        cursor = Path(anchor)
        for component in parts:
            cursor = cursor / component
            _require(os.path.lexists(cursor), f"snapshot component absent: {cursor}")
            info = os.lstat(cursor)
            _require(not stat.S_ISLNK(info.st_mode), f"snapshot symlink component: {cursor}")
        fd = os.open(str(path), os.O_RDONLY | binary | nofollow)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        for item in reversed(ancestors):
            os.close(item)
        raise DispatchContractError("snapshot leaf is not regular")
    chunks = []
    while chunk := os.read(fd, 1 << 20):
        chunks.append(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    data = b"".join(chunks)
    return SnapshotFile(path, fd, ancestors, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns), data, _sha(data))


@dataclass
class AuthenticatedSourceClosure:
    manifest_sha256: str
    review_sha256: str
    members: dict[str, SnapshotFile]
    interpreter_executable: str
    interpreter_version: str

    def snapshot_bytes(self) -> dict[str, bytes]:
        return {name: item.data for name, item in self.members.items()}

    def verify_stable(self) -> None:
        for item in self.members.values():
            item.verify_stable()

    def close(self) -> None:
        for item in self.members.values():
            item.close()
        self.members.clear()


def authenticate_reference_closure(
    *,
    package: Path,
    review_path: Path,
    externally_pinned_manifest_sha256: str,
    externally_pinned_review_sha256: str,
) -> AuthenticatedSourceClosure:
    """Authenticate once; caller must itself be an externally pinned script."""
    _require(sys.flags.isolated == 1 and sys.dont_write_bytecode, "dispatcher requires python -I -B")
    manifest_file = open_snapshot(package / "SOURCE_MANIFEST.json")
    review_file = open_snapshot(review_path)
    held: dict[str, SnapshotFile] = {}
    try:
        require_external_pin(manifest_file.data, externally_pinned_manifest_sha256, "manifest")
        require_external_pin(review_file.data, externally_pinned_review_sha256, "review")
        manifest = _strict_json(manifest_file.data)
        review = _strict_json(review_file.data)
        _require(manifest.get("schema") == "unifilar-wfa-source-manifest-v2", "manifest schema")
        _require(manifest.get("status") == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "manifest status")
        _require(review.get("schema") == "unifilar-wfa-entropy-census-independent-source-review-v2", "review schema")
        _require(review.get("status") == "PASS_INDEPENDENT_SOURCE_REVIEW", "review status")
        _require(review.get("reviewed_source_manifest_sha256") == manifest_file.sha256, "review/manifest binding")
        rows = manifest.get("members")
        _require(isinstance(rows, list) and rows, "manifest members")
        declared = {"SOURCE_MANIFEST.json"}
        for row in rows:
            _require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "manifest row schema")
            name = row["name"]
            _require(isinstance(name, str) and name == Path(name).name and name not in declared, "manifest member name")
            item = open_snapshot(package / name)
            _require(type(row["bytes"]) is int and row["bytes"] == len(item.data), f"manifest bytes {name}")
            _require(isinstance(row["sha256"], str) and row["sha256"] == item.sha256, f"manifest digest {name}")
            held[name] = item
            declared.add(name)
        # Reject undeclared regular/symlink leaves.  __pycache__ is forbidden;
        # isolated -B means it should not appear in a clean package.
        actual = {entry.name for entry in os.scandir(package)}
        require_exact_declared_members(actual, declared)
        # The entrypoint itself is among authenticated snapshots and is not
        # imported by path.  Keep manifest/review held as dynamic authorities.
        held["SOURCE_MANIFEST.json"] = manifest_file
        held["__EXTERNAL_REVIEW__.json"] = review_file
        return AuthenticatedSourceClosure(
            manifest_file.sha256,
            review_file.sha256,
            held,
            sys.executable,
            sys.version,
        )
    except Exception:
        manifest_file.close()
        review_file.close()
        for item in held.values():
            item.close()
        raise


def exec_snapshot_module(name: str, source: bytes, *, expected_sha256: str) -> types.ModuleType:
    _require(_sha(source) == expected_sha256, "module snapshot digest")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated-snapshot:{name}>"
    module.__package__ = ""
    code = compile(source, module.__file__, "exec", dont_inherit=True, optimize=0)
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


def reject_direct_payload_launch() -> None:
    raise DispatchContractError(
        "reference dispatcher is inside producer closure and cannot be a root of trust; "
        "an independent audit package must hard-code both external digests"
    )


if __name__ == "__main__":
    reject_direct_payload_launch()
