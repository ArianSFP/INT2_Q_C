#!/usr/bin/env python3
"""Stdlib-only authenticated-bytes bootstrap; never imports from the live tree."""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
import stat
import sys
from types import MappingProxyType, ModuleType
from typing import Any, Sequence


INVENTORY_SCHEMA = "tactic_actual_coarse_n18_external_inventory_v3"
ROOT_DOMAIN = b"TACTIC-N18-V3-AUTHENTICATED-SOURCE-ROOT-v1\0"
EXPECTED_SOURCE_FILES = (
    "POSTIMPLEMENTATION_REVIEW.md",
    "README.md",
    "dependency_auth.py",
    "dependency_graph.json",
    "design_lock.json",
    "dispatcher_contract.py",
    "immutable_bootstrap.py",
    "runtime_auth.py",
    "safe_telemetry.py",
    "secure_io.py",
    "test_source_only.py",
    "universal_layout.py",
    "v3_common.py",
    "verify_source.py",
)
MAX_INVENTORY_BYTES = 128 << 10
MAX_ROWS = 32
MAX_NAME_BYTES = 128
MAX_MEMBER_BYTES = 1 << 20
MAX_AGGREGATE_BYTES = 8 << 20
MAX_BOOTSTRAP_BYTES = 1 << 20


class BootstrapError(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapError(message)


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in rows:
        check(key not in value, f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _constant(value: str) -> None:
    raise BootstrapError(f"non-finite JSON constant: {value}")


def _valid_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower() or value == "0" * 64:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def _canonical_absolute(path: str, *, leaf: bool) -> tuple[str, list[str]]:
    check(os.name == "posix" and isinstance(path, str) and os.path.isabs(path), "absolute POSIX path")
    normalized = os.path.normpath(path)
    check(normalized == path and (not leaf or normalized != "/"), "canonical absolute path")
    parts = [part for part in normalized.split("/") if part]
    check((not leaf or parts) and all(part not in (".", "..") for part in parts), "safe path components")
    return normalized, parts


def _open_directory(path: str) -> int:
    normalized, parts = _canonical_absolute(path, leaf=False)
    del normalized
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for part in parts:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_file(path: str, cap: int) -> tuple[int, os.stat_result]:
    normalized, parts = _canonical_absolute(path, leaf=True)
    del normalized
    parent_path = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
    parent = _open_directory(parent_path)
    try:
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        metadata = os.fstat(descriptor)
        check(stat.S_ISREG(metadata.st_mode) and 0 < metadata.st_size <= cap, "held file type/size")
        return descriptor, metadata
    except Exception:
        os.close(descriptor)
        raise


def _read_exact(descriptor: int, size: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    result = bytearray()
    while len(result) < size:
        packet = os.read(descriptor, min(1 << 20, size - len(result)))
        check(bool(packet), "held file early EOF")
        result.extend(packet)
    check(os.read(descriptor, 1) == b"", "held file trailing bytes")
    return bytes(result)


def validate_inventory_bytes(raw: bytes) -> dict[str, Any]:
    check(type(raw) is bytes and 0 < len(raw) <= MAX_INVENTORY_BYTES, "bounded inventory bytes")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError) as exc:
        raise BootstrapError(f"strict external inventory JSON: {exc}") from exc
    check(
        isinstance(value, dict)
        and set(value) == {"schema", "status", "files", "authority_boundary"},
        "external inventory exact keys",
    )
    check(value["schema"] == INVENTORY_SCHEMA, "external inventory schema")
    check(value["status"] == "EXTERNAL_INVENTORY_NO_EXECUTION_AUTHORITY", "external inventory status")
    check(
        isinstance(value["authority_boundary"], str)
        and 0 < len(value["authority_boundary"].encode("utf-8")) <= 4096,
        "bounded inventory authority boundary",
    )
    rows = value["files"]
    check(isinstance(rows, list) and len(rows) == len(EXPECTED_SOURCE_FILES) <= MAX_ROWS, "exact bounded inventory rows")
    names: list[str] = []
    aggregate = 0
    for row in rows:
        check(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "inventory row exact keys")
        name = row["name"]
        size = row["bytes"]
        digest = row["sha256"]
        check(
            isinstance(name, str)
            and 0 < len(name.encode("utf-8")) <= MAX_NAME_BYTES
            and name not in (".", "..")
            and "/" not in name
            and "\\" not in name,
            "inventory flat bounded name",
        )
        check(type(size) is int and 0 < size <= MAX_MEMBER_BYTES, "inventory member byte cap")
        check(_valid_digest(digest), "inventory member digest")
        aggregate += size
        check(aggregate <= MAX_AGGREGATE_BYTES, "inventory aggregate byte cap")
        names.append(name)
    check(names == sorted(set(names), key=lambda name: name.encode("utf-8")), "inventory unique bytewise-sorted names")
    check(tuple(names) == tuple(sorted(EXPECTED_SOURCE_FILES, key=lambda name: name.encode("utf-8"))), "inventory exact source names")
    return value


def authenticate_bootstrap_launch(inherited_fd: int, expected_sha256: str) -> tuple[tuple[int, ...], str]:
    """Confirm the bootstrap was launched as /proc/self/fd/N from a held FD.

    The independent dispatcher, not this package, owns and authenticates this
    FD.  Requiring the procfd spelling excludes a live repository-path launch.
    """

    check(os.name == "posix" and type(inherited_fd) is int and inherited_fd >= 3, "held bootstrap FD")
    check(_valid_digest(expected_sha256), "externally pinned bootstrap SHA-256")
    check(os.path.abspath(__file__) == f"/proc/self/fd/{inherited_fd}", "bootstrap must execute through held procfd")
    before = os.fstat(inherited_fd)
    check(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_BOOTSTRAP_BYTES, "bootstrap FD type/size")
    packet = os.pread(inherited_fd, before.st_size + 1, 0)
    check(len(packet) == before.st_size, "bootstrap FD exact bytes")
    digest = hashlib.sha256(packet).hexdigest()
    check(digest == expected_sha256, "bootstrap FD digest")
    after = os.fstat(inherited_fd)
    identity = (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    check(identity == (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "bootstrap FD identity drift")
    return identity, digest


def _verify_bootstrap_launch(inherited_fd: int, identity: tuple[int, ...], digest: str) -> None:
    current = os.fstat(inherited_fd)
    check(
        identity == (current.st_dev, current.st_ino, current.st_mode, current.st_size, current.st_mtime_ns, current.st_ctime_ns),
        "bootstrap FD final identity drift",
    )
    packet = os.pread(inherited_fd, current.st_size + 1, 0)
    check(len(packet) == current.st_size and hashlib.sha256(packet).hexdigest() == digest, "bootstrap FD final content drift")


def _read_external_inventory(path: str, expected_sha256: str) -> tuple[bytes, dict[str, Any]]:
    check(_valid_digest(expected_sha256), "externally pinned inventory SHA-256")
    descriptor, before = _open_file(path, MAX_INVENTORY_BYTES)
    try:
        raw = _read_exact(descriptor, before.st_size)
        check(hashlib.sha256(raw).hexdigest() == expected_sha256.lower(), "external inventory digest")
        after = os.fstat(descriptor)
        check(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "external inventory identity drift",
        )
    finally:
        os.close(descriptor)
    return raw, validate_inventory_bytes(raw)


def authenticate_source_bytes(
    source_dir: str,
    inventory_path: str,
    expected_inventory_sha256: str,
    expected_source_root: str,
) -> tuple[MappingProxyType[str, bytes], str, str]:
    check(
        _valid_digest(expected_source_root),
        "externally pinned source root",
    )
    inventory_raw, inventory = _read_external_inventory(inventory_path, expected_inventory_sha256)
    directory = _open_directory(source_dir)
    try:
        observed_names = set(os.listdir(directory))
        expected_names = {row["name"] for row in inventory["files"]}
        check(observed_names == expected_names, "source directory exact closure; pycache/live extras forbidden")
        packets: dict[str, bytes] = {}
        root = hashlib.sha256()
        root.update(ROOT_DOMAIN)
        for row in inventory["files"]:
            name = row["name"]
            metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
            check(stat.S_ISREG(metadata.st_mode) and metadata.st_size == row["bytes"], f"source member type/bytes: {name}")
            descriptor = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory)
            try:
                before = os.fstat(descriptor)
                packet = _read_exact(descriptor, before.st_size)
                digest = hashlib.sha256(packet).hexdigest()
                check(digest == row["sha256"], f"source member digest: {name}")
                after = os.fstat(descriptor)
                check(
                    (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    f"source member identity drift: {name}",
                )
            finally:
                os.close(descriptor)
            packets[name] = packet
            encoded = name.encode("utf-8")
            root.update(len(encoded).to_bytes(4, "big"))
            root.update(encoded)
            root.update(len(packet).to_bytes(8, "big"))
            root.update(bytes.fromhex(digest))
        observed_root = root.hexdigest()
        check(observed_root == expected_source_root.lower(), "external source-root mismatch")
    finally:
        os.close(directory)
    return MappingProxyType(packets), observed_root, hashlib.sha256(inventory_raw).hexdigest()


class AuthenticatedBytesLoader(importlib.abc.Loader):
    def __init__(self, module_name: str, filename: str, packet: bytes, digest: str) -> None:
        self.module_name = module_name
        self.filename = filename
        self.packet = packet
        self.digest = digest

    def create_module(self, spec: Any) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__file__ = f"<authenticated-bytes:{self.filename}:{self.digest}>"
        module.__cached__ = None
        module.__authenticated_source_sha256__ = self.digest
        code = compile(self.packet, module.__file__, "exec", dont_inherit=True)
        exec(code, module.__dict__)


class AuthenticatedBytesFinder(importlib.abc.MetaPathFinder):
    def __init__(self, packets: MappingProxyType[str, bytes]) -> None:
        self.sources = {
            name[:-3]: (name, packet, hashlib.sha256(packet).hexdigest())
            for name, packet in packets.items()
            if name.endswith(".py")
        }
        self.loaders: dict[str, AuthenticatedBytesLoader] = {}

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if "." in fullname or fullname not in self.sources:
            return None
        filename, packet, digest = self.sources[fullname]
        loader = AuthenticatedBytesLoader(fullname, filename, packet, digest)
        self.loaders[fullname] = loader
        return importlib.util.spec_from_loader(fullname, loader, origin=f"<authenticated-bytes:{filename}>")


def execute_authenticated_entry(
    packets: MappingProxyType[str, bytes],
    source_root: str,
    inventory_sha256: str,
    entry: str,
    entry_arguments: Sequence[str],
) -> int:
    module_name = {"verify": "verify_source", "tests": "test_source_only"}[entry]
    sibling_names = {name[:-3] for name in packets if name.endswith(".py")}
    check(not (sibling_names & set(sys.modules)), "sibling module preloaded before authenticated loader")
    source_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [path for path in sys.path if path not in ("", ".", source_dir, os.getcwd())]
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    finder = AuthenticatedBytesFinder(packets)
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module(module_name)
        check(hasattr(module, "authenticated_main"), "authenticated entry ABI")
        context = MappingProxyType(
            {
                "packets": packets,
                "source_root": source_root,
                "inventory_sha256": inventory_sha256,
                "loader_kind": "immutable_authenticated_bytes_no_pycache_no_live_path",
            }
        )
        result = module.authenticated_main(context, list(entry_arguments))
        for loaded_name in sibling_names & set(sys.modules):
            loaded = sys.modules[loaded_name]
            check(
                isinstance(getattr(loaded, "__loader__", None), AuthenticatedBytesLoader)
                and getattr(loaded, "__cached__", None) is None,
                f"sibling live-path/pycache fallback: {loaded_name}",
            )
        return int(result)
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--bootstrap-fd", type=int, required=True)
    parser.add_argument("--expected-bootstrap-sha256", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--expected-source-root", required=True)
    parser.add_argument("--entry", choices=("verify", "tests"), required=True)
    parser.add_argument("entry_arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    identity, bootstrap_digest = authenticate_bootstrap_launch(
        arguments.bootstrap_fd,
        arguments.expected_bootstrap_sha256,
    )
    try:
        packets, source_root, inventory_sha256 = authenticate_source_bytes(
            arguments.source_dir,
            arguments.inventory,
            arguments.expected_inventory_sha256,
            arguments.expected_source_root,
        )
        forwarded = list(arguments.entry_arguments)
        if forwarded and forwarded[0] == "--":
            forwarded.pop(0)
        return execute_authenticated_entry(
            packets,
            source_root,
            inventory_sha256,
            arguments.entry,
            forwarded,
        )
    finally:
        _verify_bootstrap_launch(arguments.bootstrap_fd, identity, bootstrap_digest)


if __name__ == "__main__":
    raise SystemExit(main())
