#!/usr/bin/env python3
"""Held executable plus installed-distribution RECORD/tree authentication."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import posixpath
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from secure_io import HeldRegularFile
from v3_common import (
    MAX_DISTRIBUTION_AGGREGATE_BYTES,
    MAX_DISTRIBUTION_FILES,
    MAX_DISTRIBUTION_NAME_BYTES,
    MAX_RUNTIME_FILE_BYTES,
    MAX_RUNTIME_LOCK_BYTES,
    RUNTIME_LOCK_SCHEMA,
    VERSION_PATTERN,
    exact_keys,
    require,
    strict_json_loads,
    valid_sha256,
)


REQUIRED_DISTRIBUTIONS = ("cupy-cuda12x", "numpy", "nvidia-ml-py", "scipy")
TREE_DOMAIN = b"TACTIC-N18-V3-DISTRIBUTION-TREE-v1\0"


def _nonempty(value: Any, label: str, maximum: int = 4096) -> str:
    require(
        isinstance(value, str)
        and 0 < len(value.encode("utf-8")) <= maximum
        and "\x00" not in value,
        label,
    )
    return value


def validate_runtime_lock_schema(raw: bytes) -> dict[str, Any]:
    value = exact_keys(
        strict_json_loads(raw),
        {
            "schema",
            "status",
            "lock_id",
            "interpreter",
            "distributions",
            "tree_algorithm",
            "claim_boundary",
        },
        "runtime lock",
    )
    require(value["schema"] == RUNTIME_LOCK_SCHEMA, "runtime lock schema")
    require(value["status"] == "FROZEN_EXTERNAL_RUNTIME_AUTHORITY", "runtime lock status")
    _nonempty(value["lock_id"], "runtime lock id", 128)
    require(
        value["tree_algorithm"]
        == "SHA256 domain + bytewise metadata-path + uint64 bytes + file SHA256; every importlib.metadata file",
        "runtime tree algorithm",
    )
    interpreter = exact_keys(
        value["interpreter"],
        {"absolute_path", "bytes", "sha256", "python_version"},
        "runtime interpreter",
    )
    path = _nonempty(interpreter["absolute_path"], "interpreter absolute path")
    require(os.path.isabs(path) and os.path.normpath(path) == path, "canonical interpreter path")
    require(type(interpreter["bytes"]) is int and 0 < interpreter["bytes"] <= MAX_RUNTIME_FILE_BYTES, "positive interpreter bytes")
    require(valid_sha256(interpreter["sha256"], nonzero=True), "nonzero interpreter digest")
    _nonempty(interpreter["python_version"], "nonempty Python version")

    rows = value["distributions"]
    require(isinstance(rows, list) and len(rows) == len(REQUIRED_DISTRIBUTIONS), "exact runtime distributions")
    names: list[str] = []
    for row in rows:
        row = exact_keys(
            row,
            {
                "name",
                "version",
                "installation_root",
                "record_path",
                "record_bytes",
                "record_sha256",
                "tree_files",
                "tree_bytes",
                "tree_sha256",
            },
            "runtime distribution",
        )
        name = _nonempty(row["name"], "distribution name", 128)
        require(name in REQUIRED_DISTRIBUTIONS, "required distribution name")
        require(
            isinstance(row["version"], str)
            and VERSION_PATTERN.fullmatch(row["version"]) is not None,
            "nonempty distribution version",
        )
        root = _nonempty(row["installation_root"], "distribution installation root")
        require(os.path.isabs(root) and os.path.normpath(root) == root, "canonical distribution installation root")
        record_path = _nonempty(row["record_path"], "distribution RECORD path")
        require(
            not record_path.startswith("/")
            and posixpath.normpath(record_path) == record_path
            and all(part not in ("", ".", "..") for part in PurePosixPath(record_path).parts),
            "canonical metadata RECORD path",
        )
        require(type(row["record_bytes"]) is int and 0 < row["record_bytes"] <= MAX_RUNTIME_FILE_BYTES, "positive RECORD bytes")
        require(valid_sha256(row["record_sha256"], nonzero=True), "nonzero RECORD digest")
        require(type(row["tree_files"]) is int and 0 < row["tree_files"] <= MAX_DISTRIBUTION_FILES, "positive tree file count")
        require(type(row["tree_bytes"]) is int and 0 < row["tree_bytes"] <= MAX_DISTRIBUTION_AGGREGATE_BYTES, "positive tree bytes")
        require(valid_sha256(row["tree_sha256"], nonzero=True), "nonzero tree digest")
        names.append(name)
    require(tuple(names) == REQUIRED_DISTRIBUTIONS, "bytewise sorted exact distribution order")
    return value


def _metadata_path(value: Any) -> str:
    text = str(value).replace("\\", "/")
    require(
        0 < len(text.encode("utf-8")) <= MAX_DISTRIBUTION_NAME_BYTES
        and not text.startswith("/")
        and "\x00" not in text
        and posixpath.normpath(text) == text,
        "canonical distribution metadata path",
    )
    return text


def _tree_distribution(row: dict[str, Any]) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(row["name"])
    require(distribution.version == row["version"], "installed distribution version")
    files = distribution.files
    require(files is not None, "installed distribution file inventory")
    pairs = sorted(
        ((_metadata_path(value), value) for value in files),
        key=lambda pair: pair[0].encode("utf-8"),
    )
    metadata_paths = [pair[0] for pair in pairs]
    require(
        len(metadata_paths) == len(set(metadata_paths)) == row["tree_files"],
        "installed distribution exact unique file count",
    )
    installation_root = row["installation_root"]
    tree = hashlib.sha256()
    tree.update(TREE_DOMAIN)
    total_bytes = 0
    record_observed: tuple[int, str] | None = None
    for metadata_path, package_path in pairs:
        located = os.path.normpath(str(distribution.locate_file(package_path)))
        require(os.path.isabs(located), "installed distribution absolute file")
        require(
            os.path.commonpath((installation_root, located)) == installation_root,
            "installed distribution file within locked installation root",
        )
        source = HeldRegularFile(
            located,
            maximum_bytes=MAX_RUNTIME_FILE_BYTES,
            allow_empty=True,
        )
        try:
            source.verify()
            size = source.bytes
            digest = source.sha256
        finally:
            source.close()
        total_bytes += size
        require(total_bytes <= MAX_DISTRIBUTION_AGGREGATE_BYTES, "distribution aggregate byte cap")
        encoded = metadata_path.encode("utf-8")
        tree.update(len(encoded).to_bytes(4, "big"))
        tree.update(encoded)
        tree.update(size.to_bytes(8, "big"))
        tree.update(bytes.fromhex(digest))
        if metadata_path == row["record_path"]:
            record_observed = (size, digest)
    require(record_observed is not None, "locked RECORD belongs to installed distribution")
    require(
        record_observed == (row["record_bytes"], row["record_sha256"]),
        "installed RECORD bytes/digest",
    )
    require(total_bytes == row["tree_bytes"], "installed distribution aggregate bytes")
    observed_tree = tree.hexdigest()
    require(observed_tree == row["tree_sha256"], "installed distribution tree digest")
    return {
        "name": row["name"],
        "version": row["version"],
        "record_bytes": record_observed[0],
        "record_sha256": record_observed[1],
        "tree_files": len(metadata_paths),
        "tree_bytes": total_bytes,
        "tree_sha256": observed_tree,
    }


@dataclass
class RuntimeAuthority:
    lock: HeldRegularFile
    interpreter: HeldRegularFile
    lock_sha256: str
    value: dict[str, Any]
    distribution_receipts: tuple[dict[str, Any], ...]

    def verify_held(self) -> None:
        self.lock.verify()
        self.interpreter.verify()

    def reverify_distribution_trees(self) -> None:
        for row in self.value["distributions"]:
            _tree_distribution(row)
        self.verify_held()

    def close(self) -> None:
        self.interpreter.close()
        self.lock.close()

    def __enter__(self) -> "RuntimeAuthority":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def authenticate_runtime(absolute_lock_path: str, expected_lock_sha256: str) -> RuntimeAuthority:
    require(os.name == "posix", "runtime authentication requires POSIX")
    require(valid_sha256(expected_lock_sha256, nonzero=True), "externally pinned runtime-lock digest")
    lock = HeldRegularFile(
        absolute_lock_path,
        maximum_bytes=MAX_RUNTIME_LOCK_BYTES,
        expected_sha256=expected_lock_sha256,
    )
    interpreter: HeldRegularFile | None = None
    try:
        raw = lock.read(MAX_RUNTIME_LOCK_BYTES)
        value = validate_runtime_lock_schema(raw)
        require(value["interpreter"]["absolute_path"] == sys.executable, "running interpreter path")
        require(value["interpreter"]["python_version"] == sys.version, "running Python version")
        interpreter = HeldRegularFile(
            value["interpreter"]["absolute_path"],
            maximum_bytes=MAX_RUNTIME_FILE_BYTES,
            expected_bytes=value["interpreter"]["bytes"],
            expected_sha256=value["interpreter"]["sha256"],
        )
        receipts = tuple(_tree_distribution(row) for row in value["distributions"])
        lock.verify()
        interpreter.verify()
        return RuntimeAuthority(lock, interpreter, lock.sha256, value, receipts)
    except Exception:
        if interpreter is not None:
            interpreter.close()
        lock.close()
        raise
