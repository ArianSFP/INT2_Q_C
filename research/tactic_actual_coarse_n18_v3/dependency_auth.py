#!/usr/bin/env python3
"""Safe dependency graph parser and immutable external-source byte bundle."""

from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any

from secure_io import HeldRegularFile
from v3_common import (
    DEPENDENCY_ID_PATTERN,
    DEPENDENCY_SCHEMA,
    MAX_DEPENDENCIES,
    MAX_DEPENDENCY_ID_BYTES,
    MAX_RELATIVE_PATH_BYTES,
    exact_keys,
    require,
    strict_json_loads,
    valid_sha256,
)


def _imports(packet: bytes) -> set[str]:
    tree = ast.parse(packet.decode("utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    roots.discard("")
    roots.discard("__future__")
    return roots


def _safe_dependency_id(value: Any) -> str:
    require(
        isinstance(value, str)
        and DEPENDENCY_ID_PATTERN.fullmatch(value) is not None,
        "safe dependency id",
    )
    require(len(value.encode("ascii")) <= MAX_DEPENDENCY_ID_BYTES, "bounded dependency id")
    return value


def _safe_relative_path(value: Any) -> str:
    require(
        isinstance(value, str)
        and 0 < len(value.encode("utf-8")) <= MAX_RELATIVE_PATH_BYTES,
        "bounded dependency relative path",
    )
    parsed = PurePosixPath(value)
    require(
        not parsed.is_absolute()
        and value == parsed.as_posix()
        and all(part not in ("", ".", "..") for part in parsed.parts),
        "safe dependency relative path",
    )
    return value


def validate_dependency_graph(raw: bytes) -> tuple[dict[str, Any], ...]:
    value = exact_keys(
        strict_json_loads(raw),
        {"schema", "status", "external_python_sources", "execution_rule", "claim_boundary"},
        "dependency graph",
    )
    require(value["schema"] == DEPENDENCY_SCHEMA, "dependency graph schema")
    require(
        value["status"] == "PINNED_PROTOTYPE_SOURCES_NO_IMPORT_NO_RUNTIME_AUTHORITY",
        "dependency graph status",
    )
    rows = value["external_python_sources"]
    require(isinstance(rows, list) and 1 <= len(rows) <= MAX_DEPENDENCIES, "bounded dependency rows")
    parsed: list[dict[str, Any]] = []
    ids: list[str] = []
    paths: list[str] = []
    for row in rows:
        row = exact_keys(
            row,
            {"id", "relative_path", "bytes", "sha256", "allowed_import_roots"},
            "dependency row",
        )
        identifier = _safe_dependency_id(row["id"])
        relative = _safe_relative_path(row["relative_path"])
        require(type(row["bytes"]) is int and 0 < row["bytes"] <= 1 << 20, "dependency byte cap")
        require(valid_sha256(row["sha256"], nonzero=True), "dependency digest")
        imports = row["allowed_import_roots"]
        require(
            isinstance(imports, list)
            and imports == sorted(set(imports))
            and all(isinstance(name, str) and DEPENDENCY_ID_PATTERN.fullmatch(name) for name in imports),
            "dependency import roots",
        )
        ids.append(identifier)
        paths.append(relative)
        parsed.append(dict(row))
    require(ids == sorted(set(ids)), "unique bytewise-sorted dependency ids")
    require(len(paths) == len(set(paths)), "unique dependency paths")
    return tuple(parsed)


@dataclass
class DependencyBundle:
    packets: MappingProxyType[str, bytes]
    rows: tuple[dict[str, Any], ...]
    held: tuple[HeldRegularFile, ...]

    def verify(self) -> None:
        for source in self.held:
            source.verify()

    def close(self) -> None:
        for source in reversed(self.held):
            source.close()

    def __enter__(self) -> "DependencyBundle":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def authenticate_dependency_sources(raw_graph: bytes, absolute_repo_root: str) -> DependencyBundle:
    require(os.name == "posix" and os.path.isabs(absolute_repo_root), "absolute POSIX repository root")
    require(os.path.normpath(absolute_repo_root) == absolute_repo_root, "canonical repository root")
    rows = validate_dependency_graph(raw_graph)
    held: list[HeldRegularFile] = []
    packets: dict[str, bytes] = {}
    receipts: list[dict[str, Any]] = []
    try:
        for row in rows:
            absolute = os.path.join(absolute_repo_root, *PurePosixPath(row["relative_path"]).parts)
            require(
                os.path.commonpath((absolute_repo_root, absolute)) == absolute_repo_root,
                "dependency repository containment",
            )
            source = HeldRegularFile(
                absolute,
                maximum_bytes=1 << 20,
                expected_bytes=row["bytes"],
                expected_sha256=row["sha256"],
            )
            held.append(source)
            packet = source.read(1 << 20)
            observed_imports = _imports(packet)
            require(observed_imports == set(row["allowed_import_roots"]), "dependency AST import drift")
            packets[row["id"]] = packet
            receipts.append(
                {
                    "id": row["id"],
                    "bytes": len(packet),
                    "sha256": hashlib.sha256(packet).hexdigest(),
                    "imports": sorted(observed_imports),
                }
            )
        for source in held:
            source.verify()
        return DependencyBundle(MappingProxyType(packets), tuple(receipts), tuple(held))
    except Exception:
        for source in reversed(held):
            source.close()
        raise


class DependencyBytesLoader(importlib.abc.Loader):
    def __init__(self, identifier: str, packet: bytes) -> None:
        self.identifier = identifier
        self.packet = packet
        self.digest = hashlib.sha256(packet).hexdigest()

    def create_module(self, spec: Any) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = f"<authenticated-dependency-bytes:{self.identifier}:{self.digest}>"
        module.__cached__ = None
        module.__authenticated_dependency_sha256__ = self.digest
        exec(compile(self.packet, module.__file__, "exec", dont_inherit=True), module.__dict__)


class DependencyBytesFinder(importlib.abc.MetaPathFinder):
    """Dedicated loader; caller installs it only after runtime authentication."""

    def __init__(self, bundle: DependencyBundle) -> None:
        self.packets = bundle.packets

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if "." in fullname or fullname not in self.packets:
            return None
        require(fullname not in sys.modules, "dependency preloaded from live path")
        loader = DependencyBytesLoader(fullname, self.packets[fullname])
        return importlib.util.spec_from_loader(fullname, loader, origin=f"<authenticated-dependency:{fullname}>")
