#!/usr/bin/env python3
"""Final source-only Wasm authority repair for the global STRATA RM swap.

V4 preserves v3's independently audited scientific provenance and one-packet
per-routed-expert ledger.  It adds an authenticated Wasmtime distribution
closure, native-library and runtime-version pins, pre-instantiation Store
limits and fuel, an architecturally immutable host-owned packet capability,
and independent semantic decode/canonical-encode authority.
"""

from __future__ import annotations

import array
import hashlib
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


V3_SOURCE_ROOT_SHA256 = (
    "83d79990515fca16387723cdea544d41fac76413fe80f919c30517d14551d6ad")
V3_MANIFEST_SHA256 = (
    "9105dd69a2a82d1eaf14e176e4334189a4c31be840dafee467d243c231788e83")
V3_REVIEW_SOURCE_ROOT_SHA256 = (
    "3113631a5c64255d919f2bb5c545436452c8a721eb4130fcd32d7ffc4b2cdfe0")
V3_REVIEW_MANIFEST_SHA256 = (
    "ebe65fcf1abd73263be0176cdb70244ebca4f0a883eb6815c24c8956b0d0d89c")
V3_AUTHORITY_MEMBER_SHA256 = (
    "21d98cc772ac9f58880cce9ea7542bc16ae60cef6bc063cc9f31d47287952e16")
V4_MANIFEST_SCHEMA = "strata-rm-global-swap-v4-wasm-authority-source-manifest"
PAGE_BYTES = 4096
STORE_MEMORY_LIMIT_BYTES = 1 << 30
STORE_TABLE_ELEMENTS = 10_000
STORE_INSTANCE_LIMIT = 2
STORE_TABLE_LIMIT = 2
STORE_MEMORY_COUNT_LIMIT = 2
FUEL_BASE = 100_000_000
FUEL_PER_PACKET_BYTE = 50_000
MAX_FUEL = 1_000_000_000_000
RATE_MIN = 2.15
RATE_MAX = 2.5
TARGET_F = 0.8
MAX_COLD_READ_AMPLIFICATION = 2.0
MIN_SOURCE_SPECIFIC_BPW = 0.03
PRODUCTION_AUTHORIZATION = "AUDIT_ROUTED_EXPERT_GLOBAL_RM_SWAP_RESULT_V4"
HEX = frozenset("0123456789abcdef")


class AuthorityError(RuntimeError):
    """The v4 authority failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AuthorityError("noncanonical JSON value") from exc


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label}: duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AuthorityError(f"{label}: nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"{label}: strict JSON") from exc
    require(isinstance(value, dict), f"{label}: JSON object")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
    candidate = Path(path)
    try:
        before = candidate.lstat()
        require(stat.S_ISREG(before.st_mode) and not candidate.is_symlink(),
                f"{label}: regular non-link file")
        payload = candidate.read_bytes()
        after = candidate.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label}: read") from exc
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_mode)
    require(identity(before) == identity(after), f"{label}: changed during read")
    return payload


def real_directory(path: Path, label: str) -> Path:
    original = Path(path)
    try:
        before = original.lstat()
        require(stat.S_ISDIR(before.st_mode) and not original.is_symlink(),
                f"{label}: real non-link directory")
        resolved = original.resolve(strict=True)
        after = original.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label}: directory resolution") from exc
    require((before.st_dev, before.st_ino, before.st_mode) ==
            (after.st_dev, after.st_ino, after.st_mode),
            f"{label}: root changed during resolution")
    return resolved


def _safe_relative(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label}: relative path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts and
            "." not in pure.parts and "\\" not in value,
            f"{label}: safe POSIX relative path")
    return Path(*pure.parts)


def resolve_member(root: Path, relative: Any, label: str) -> Path:
    base = real_directory(root, f"{label} root")
    current = base
    try:
        for part in _safe_relative(relative, label).parts:
            current = current / part
            require(not stat.S_ISLNK(current.lstat().st_mode),
                    f"{label}: symlink component")
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise AuthorityError(f"{label}: resolution") from exc
    require(base in resolved.parents and resolved != base, f"{label}: containment")
    return resolved


def _row_root(rows: list[dict[str, Any]]) -> str:
    return sha256(canonical_json(rows))


def authenticate_flat_package(package: Path, *, manifest_name: str,
                              expected_manifest_sha256: str,
                              expected_source_root_sha256: str,
                              expected_schema: str) -> dict[str, Any]:
    require(is_sha256(expected_manifest_sha256) and
            is_sha256(expected_source_root_sha256), "package external pins")
    root = real_directory(package, "dependency package")
    manifest_payload = regular_bytes(root / manifest_name, "dependency manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "dependency manifest external pin")
    manifest = strict_json(manifest_payload, "dependency manifest")
    require(manifest.get("schema") == expected_schema and
            manifest.get("source_root_sha256") == expected_source_root_sha256,
            "dependency schema/root")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "dependency members")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "dependency member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name and
                name not in names and name != manifest_name,
                "dependency member name")
        payload = regular_bytes(root / name, f"dependency member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"dependency member pin {name}")
        observed.append(item)
        names.append(name)
    require(_row_root(observed) == expected_source_root_sha256,
            "dependency recomputed root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {manifest_name} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "dependency exact regular closure")
    return {"path": str(root), "manifest": manifest,
            "source_root_sha256": expected_source_root_sha256,
            "manifest_sha256": expected_manifest_sha256, "member_rows": observed}


def authenticate_v3_and_review(v3_package: Path,
                               review_package: Path) -> dict[str, Any]:
    v3 = authenticate_flat_package(
        v3_package, manifest_name="source_manifest.json",
        expected_manifest_sha256=V3_MANIFEST_SHA256,
        expected_source_root_sha256=V3_SOURCE_ROOT_SHA256,
        expected_schema="strata-rm-global-swap-v3-physical-authority-source-manifest")
    review = authenticate_flat_package(
        review_package, manifest_name="source_manifest.json",
        expected_manifest_sha256=V3_REVIEW_MANIFEST_SHA256,
        expected_source_root_sha256=V3_REVIEW_SOURCE_ROOT_SHA256,
        expected_schema=(
            "strata-rm-global-swap-v3-physical-authority-independent-source-review-manifest"))
    require(review["manifest"].get("producer_manifest_sha256") ==
            V3_MANIFEST_SHA256 and
            review["manifest"].get("producer_source_root_sha256") ==
            V3_SOURCE_ROOT_SHA256, "review-to-v3 binding")
    authority_row = next((row for row in v3["member_rows"]
                          if row["name"] == "authority_v3.py"), None)
    require(authority_row is not None and
            authority_row["sha256"] == V3_AUTHORITY_MEMBER_SHA256,
            "v3 authority member pin")
    return {"v3": v3, "review": review,
            "status": "PASS_PINNED_V3_AND_INDEPENDENT_REVIEW"}


def load_v3_authority_snapshot(v3_auth: Mapping[str, Any]):
    source = Path(v3_auth["v3"]["path"]) / "authority_v3.py"
    payload = regular_bytes(source, "v3 authority source")
    require(sha256(payload) == V3_AUTHORITY_MEMBER_SHA256,
            "v3 authority exact source")
    with tempfile.TemporaryDirectory(prefix="strata-rm-v4-v3-") as directory:
        path = Path(directory) / "authority_v3.py"
        path.write_bytes(payload)
        os.chmod(path, 0o444)
        spec = importlib.util.spec_from_file_location("pinned_authority_v3", path)
        require(spec is not None and spec.loader is not None, "v3 authority spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def authenticate_v4_package(package: Path,
                            expected_manifest_sha256: str) -> dict[str, Any]:
    require(is_sha256(expected_manifest_sha256), "v4 manifest external pin")
    root = real_directory(package, "v4 package")
    payload = regular_bytes(root / "source_manifest.json", "v4 manifest")
    require(sha256(payload) == expected_manifest_sha256, "v4 manifest pin")
    manifest = strict_json(payload, "v4 manifest")
    require(canonical_json(manifest) + b"\n" == payload and set(manifest) ==
            {"schema", "status", "v3_source_root_sha256",
             "v3_manifest_sha256", "v3_review_source_root_sha256",
             "v3_review_manifest_sha256", "source_root_sha256", "members"} and
            manifest["schema"] == V4_MANIFEST_SCHEMA and
            manifest["status"] ==
            "FROZEN_V4_WASM_AUTHORITY_SOURCE_ONLY__HOLD_RUNTIME_PAYLOAD_AND_RD" and
            manifest["v3_source_root_sha256"] == V3_SOURCE_ROOT_SHA256 and
            manifest["v3_manifest_sha256"] == V3_MANIFEST_SHA256 and
            manifest["v3_review_source_root_sha256"] ==
            V3_REVIEW_SOURCE_ROOT_SHA256 and
            manifest["v3_review_manifest_sha256"] ==
            V3_REVIEW_MANIFEST_SHA256, "v4 canonical manifest and lineage")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "v4 members")
    names = []
    observed = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "v4 member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name and
                name not in names and name != "source_manifest.json",
                "v4 member name")
        member = regular_bytes(root / name, f"v4 member {name}")
        item = {"name": name, "bytes": len(member), "sha256": sha256(member)}
        require(item == row, f"v4 member pin {name}")
        names.append(name)
        observed.append(item)
    require(_row_root(observed) == manifest.get("source_root_sha256"),
            "v4 source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {"source_manifest.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "v4 exact regular closure")
    return {"path": str(root), "source_root_sha256": manifest["source_root_sha256"],
            "manifest_sha256": expected_manifest_sha256, "member_rows": observed}


def _authenticate_flat_audit_members(root: Path, rows: Any,
                                     source_root: str, reserved: set[str],
                                     label: str) -> list[str]:
    require(isinstance(rows, list) and rows and _row_root(rows) == source_root,
            f"{label}: source root")
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                f"{label}: member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name and
                name not in names and name not in reserved,
                f"{label}: member name")
        payload = regular_bytes(root / name, f"{label}: member {name}")
        require({"name": name, "bytes": len(payload), "sha256": sha256(payload)} == row,
                f"{label}: member pin {name}")
        names.append(name)
    return names


def _all_recursive_entries(root: Path) -> tuple[set[str], set[str]]:
    files = set()
    directories = set()
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        for name in directory_names:
            path = current / name
            require(not path.is_symlink() and stat.S_ISDIR(path.lstat().st_mode),
                    "runtime closure real directory")
            directories.add(path.relative_to(root).as_posix())
        for name in file_names:
            path = current / name
            require(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode),
                    "runtime closure regular file")
            files.add(path.relative_to(root).as_posix())
    return files, directories


def _runtime_limits() -> dict[str, int | bool]:
    return {"consume_fuel": True,
            "store_memory_limit_bytes": STORE_MEMORY_LIMIT_BYTES,
            "store_table_elements": STORE_TABLE_ELEMENTS,
            "store_instance_limit": STORE_INSTANCE_LIMIT,
            "store_table_limit": STORE_TABLE_LIMIT,
            "store_memory_count_limit": STORE_MEMORY_COUNT_LIMIT,
            "fuel_base": FUEL_BASE,
            "fuel_per_packet_byte": FUEL_PER_PACKET_BYTE,
            "maximum_fuel": MAX_FUEL}


def _native_runtime_name(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return (name.endswith((".dll", ".dylib", ".pyd")) or ".so" in name)


def authenticate_runtime_audit_package(
        package: Path, *, expected_manifest_sha256: str,
        expected_source_root_sha256: str, expected_receipt_sha256: str,
        expected_capability_sha256: str,
        expected_runtime_tree_root_sha256: str) -> dict[str, Any]:
    """Authenticate the complete Wasmtime binding/native distribution tree."""
    require(all(is_sha256(value) for value in
                (expected_manifest_sha256, expected_source_root_sha256,
                 expected_receipt_sha256, expected_capability_sha256,
                 expected_runtime_tree_root_sha256)),
            "runtime audit out-of-band pins")
    root = real_directory(package, "runtime audit package")
    manifest_payload = regular_bytes(root / "source_manifest.json",
                                     "runtime audit manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "runtime audit manifest pin")
    manifest = strict_json(manifest_payload, "runtime audit manifest")
    required_manifest = {"schema", "source_root_sha256", "receipt_name",
                         "capability_name", "capability_sha256",
                         "runtime_tree_root_sha256", "members", "runtime_files"}
    require(canonical_json(manifest) + b"\n" == manifest_payload and
            set(manifest) == required_manifest and manifest["schema"] ==
            "strata-rm-global-swap-v4-wasmtime-runtime-audit-manifest" and
            manifest["source_root_sha256"] == expected_source_root_sha256 and
            manifest["receipt_name"] == "AUDIT_RECEIPT.json" and
            manifest["capability_name"] == "RUNTIME_CAPABILITY.json" and
            manifest["capability_sha256"] == expected_capability_sha256 and
            manifest["runtime_tree_root_sha256"] ==
            expected_runtime_tree_root_sha256,
            "runtime audit manifest binding")
    member_names = _authenticate_flat_audit_members(
        root, manifest["members"], expected_source_root_sha256,
        {"source_manifest.json", "AUDIT_RECEIPT.json", "RUNTIME_CAPABILITY.json",
         "runtime"}, "runtime audit")
    runtime_rows = manifest["runtime_files"]
    require(isinstance(runtime_rows, list) and runtime_rows and
            runtime_rows == sorted(runtime_rows, key=lambda row: row.get("path", "")) and
            _row_root(runtime_rows) == expected_runtime_tree_root_sha256,
            "runtime sorted recursive tree root")
    runtime_payloads = {}
    runtime_paths = []
    allowed_kinds = {"python_module", "native_library", "metadata", "resource"}
    for row in runtime_rows:
        require(isinstance(row, dict) and set(row) ==
                {"path", "bytes", "sha256", "kind"} and
                row["kind"] in allowed_kinds and
                isinstance(row["bytes"], int) and row["bytes"] > 0 and
                is_sha256(row["sha256"]), "runtime file row")
        relative = _safe_relative(row["path"], "runtime file")
        require(relative.parts[0] == "runtime" and row["path"] not in runtime_paths,
                "runtime file location/uniqueness")
        payload = regular_bytes(resolve_member(root, row["path"], "runtime file"),
                                f"runtime file {row['path']}")
        require(len(payload) == row["bytes"] and sha256(payload) == row["sha256"],
                f"runtime file pin {row['path']}")
        runtime_paths.append(row["path"])
        runtime_payloads[row["path"]] = payload
    all_files, all_directories = _all_recursive_entries(root)
    expected_top = set(member_names) | {"source_manifest.json", "AUDIT_RECEIPT.json",
                                        "RUNTIME_CAPABILITY.json"}
    expected_directories = {"runtime"}
    for runtime_path in runtime_paths:
        parts = PurePosixPath(runtime_path).parts
        expected_directories.update(
            PurePosixPath(*parts[:end]).as_posix()
            for end in range(1, len(parts)))
    require(all_files == expected_top | set(runtime_paths),
            "runtime package exact recursive file closure")
    require(all_directories == expected_directories,
            "runtime package exact recursive directory closure")
    capability_payload = regular_bytes(root / "RUNTIME_CAPABILITY.json",
                                       "runtime capability")
    require(sha256(capability_payload) == expected_capability_sha256,
            "runtime capability pin")
    capability = strict_json(capability_payload, "runtime capability")
    require(canonical_json(capability) + b"\n" == capability_payload,
            "runtime capability canonical bytes")
    required_capability = {
        "schema", "distribution_name", "python_distribution_version",
        "wasmtime_runtime_version", "python_abi", "platform_tag", "target",
        "module_entry_relative_path", "metadata_relative_path",
        "runtime_tree_root_sha256", "module_tree_root_sha256",
        "native_library_root_sha256", "python_module_files", "native_libraries",
        "engine_limits"}
    require(set(capability) == required_capability and capability["schema"] ==
            "strata-rm-global-swap-v4-wasmtime-runtime-capability" and
            capability["distribution_name"] == "wasmtime" and
            all(isinstance(capability[name], str) and capability[name] for name in
                ("python_distribution_version", "wasmtime_runtime_version",
                 "python_abi", "platform_tag", "target")) and
            capability["runtime_tree_root_sha256"] ==
            expected_runtime_tree_root_sha256 and
            capability["wasmtime_runtime_version"] ==
            capability["python_distribution_version"] and
            capability["engine_limits"] == _runtime_limits(),
            "runtime capability schema/limits")
    rows_by_path = {row["path"]: row for row in runtime_rows}
    module_paths = capability["python_module_files"]
    native_paths = capability["native_libraries"]
    require(isinstance(module_paths, list) and module_paths and
            module_paths == sorted(module_paths) and
            len(module_paths) == len(set(module_paths)) and
            all(path in rows_by_path and rows_by_path[path]["kind"] ==
                "python_module" for path in module_paths) and
            isinstance(native_paths, list) and native_paths and
            native_paths == sorted(native_paths) and
            len(native_paths) == len(set(native_paths)) and
            all(path in rows_by_path and rows_by_path[path]["kind"] ==
                "native_library" for path in native_paths),
            "runtime module/native inventories")
    module_rows = [rows_by_path[path] for path in module_paths]
    native_rows = [rows_by_path[path] for path in native_paths]
    require(set(module_paths) == {path for path in runtime_paths
                                  if PurePosixPath(path).suffix.lower() in
                                  {".py", ".pyi"}} and
            set(native_paths) == {path for path in runtime_paths
                                  if _native_runtime_name(path)} and
            _row_root(module_rows) == capability["module_tree_root_sha256"] and
            _row_root(native_rows) == capability["native_library_root_sha256"] and
            capability["module_entry_relative_path"] in module_paths and
            capability["metadata_relative_path"] in rows_by_path and
            rows_by_path[capability["metadata_relative_path"]]["kind"] == "metadata",
            "runtime module/native roots and entry points")
    metadata = runtime_payloads[capability["metadata_relative_path"]].decode(
        "utf-8", errors="strict")
    require(any(line.strip() ==
                f"Version: {capability['python_distribution_version']}"
                for line in metadata.splitlines()),
            "runtime distribution METADATA version")
    receipt_payload = regular_bytes(root / "AUDIT_RECEIPT.json",
                                    "runtime audit receipt")
    require(sha256(receipt_payload) == expected_receipt_sha256,
            "runtime audit receipt pin")
    receipt = strict_json(receipt_payload, "runtime audit receipt")
    required_receipt = {
        "schema", "executed", "status", "audit_source_root_sha256",
        "runtime_capability_sha256", "runtime_tree_root_sha256",
        "python_distribution_version_observed", "wasmtime_runtime_version_observed",
        "module_tree_rehashed", "native_libraries_loaded_and_rehashed",
        "module_origin_from_snapshot", "target_observed", "engine_compile_probe",
        "store_memory_limit_probe", "fuel_exhaustion_probe", "hostile_tests",
        "payloads_opened"}
    require(canonical_json(receipt) + b"\n" == receipt_payload and
            set(receipt) == required_receipt and receipt["schema"] ==
            "strata-rm-global-swap-v4-wasmtime-runtime-audit-receipt" and
            receipt["executed"] is True and receipt["status"] ==
            "PASS_PINNED_WASMTIME_RUNTIME_AUDIT_V4" and
            receipt["audit_source_root_sha256"] == expected_source_root_sha256 and
            receipt["runtime_capability_sha256"] == expected_capability_sha256 and
            receipt["runtime_tree_root_sha256"] ==
            expected_runtime_tree_root_sha256 and
            receipt["python_distribution_version_observed"] ==
            capability["python_distribution_version"] and
            receipt["wasmtime_runtime_version_observed"] ==
            capability["wasmtime_runtime_version"] and
            receipt["target_observed"] == capability["target"] and
            all(receipt[name] is True for name in
                ("module_tree_rehashed", "native_libraries_loaded_and_rehashed",
                 "module_origin_from_snapshot", "engine_compile_probe",
                 "store_memory_limit_probe", "fuel_exhaustion_probe")) and
            isinstance(receipt["hostile_tests"], int) and
            receipt["hostile_tests"] >= 12 and receipt["payloads_opened"] == 0,
            "successful pinned Wasmtime runtime receipt")
    return {"capability": capability, "capability_payload": capability_payload,
            "runtime_rows": runtime_rows, "runtime_payloads": runtime_payloads,
            "manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": expected_source_root_sha256,
            "receipt_sha256": expected_receipt_sha256,
            "capability_sha256": expected_capability_sha256,
            "runtime_tree_root_sha256": expected_runtime_tree_root_sha256,
            "status": "PASS_SEPARATELY_PINNED_WASMTIME_DISTRIBUTION_AND_RUNTIME"}


def authenticate_semantic_decoder_audit_package(
        package: Path, *, expected_manifest_sha256: str,
        expected_source_root_sha256: str, expected_receipt_sha256: str,
        expected_decoder_sha256: str, expected_canonical_encoder_sha256: str,
        expected_semantic_schema_sha256: str, expected_sandbox_sha256: str
        ) -> dict[str, Any]:
    """Authenticate independently audited semantic decoder/canonical encoder."""
    require(all(is_sha256(value) for value in
                (expected_manifest_sha256, expected_source_root_sha256,
                 expected_receipt_sha256, expected_decoder_sha256,
                 expected_canonical_encoder_sha256,
                 expected_semantic_schema_sha256, expected_sandbox_sha256)),
            "semantic decoder audit out-of-band pins")
    root = real_directory(package, "semantic decoder audit package")
    manifest_payload = regular_bytes(root / "source_manifest.json",
                                     "semantic decoder audit manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "semantic decoder audit manifest pin")
    manifest = strict_json(manifest_payload, "semantic decoder audit manifest")
    required_manifest = {"schema", "source_root_sha256", "receipt_name",
                         "decoder_name", "decoder_sha256",
                         "canonical_encoder_name", "canonical_encoder_sha256",
                         "semantic_schema_name", "semantic_schema_sha256",
                         "sandbox_sha256", "members"}
    require(canonical_json(manifest) + b"\n" == manifest_payload and
            set(manifest) == required_manifest and manifest["schema"] ==
            "strata-rm-global-swap-v4-semantic-decoder-audit-manifest" and
            manifest["source_root_sha256"] == expected_source_root_sha256 and
            manifest["receipt_name"] == "AUDIT_RECEIPT.json" and
            manifest["decoder_name"] == "DECODER.wasm" and
            manifest["decoder_sha256"] == expected_decoder_sha256 and
            manifest["canonical_encoder_name"] == "CANONICAL_ENCODER.wasm" and
            manifest["canonical_encoder_sha256"] ==
            expected_canonical_encoder_sha256 and
            manifest["semantic_schema_name"] == "SEMANTIC_SCHEMA.json" and
            manifest["semantic_schema_sha256"] ==
            expected_semantic_schema_sha256 and
            manifest["sandbox_sha256"] == expected_sandbox_sha256,
            "semantic decoder manifest binding")
    names = _authenticate_flat_audit_members(
        root, manifest["members"], expected_source_root_sha256,
        {"source_manifest.json", "AUDIT_RECEIPT.json", "DECODER.wasm",
         "CANONICAL_ENCODER.wasm", "SEMANTIC_SCHEMA.json"},
        "semantic decoder audit")
    expected_files = set(names) | {"source_manifest.json", "AUDIT_RECEIPT.json",
                                   "DECODER.wasm", "CANONICAL_ENCODER.wasm",
                                   "SEMANTIC_SCHEMA.json"}
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == expected_files and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "semantic decoder audit exact closure")
    decoder = regular_bytes(root / "DECODER.wasm", "semantic decoder")
    encoder = regular_bytes(root / "CANONICAL_ENCODER.wasm",
                            "independent canonical encoder")
    schema_payload = regular_bytes(root / "SEMANTIC_SCHEMA.json",
                                   "semantic state schema")
    require(sha256(decoder) == expected_decoder_sha256 and
            sha256(encoder) == expected_canonical_encoder_sha256 and
            expected_decoder_sha256 != expected_canonical_encoder_sha256 and
            sha256(schema_payload) == expected_semantic_schema_sha256,
            "semantic decoder/encoder/schema pins")
    schema = strict_json(schema_payload, "semantic state schema")
    require(canonical_json(schema) + b"\n" == schema_payload and
            set(schema) == {"schema", "version", "state_fields",
                            "raw_packet_bytes_permitted",
                            "complete_quantizer_decisions",
                            "canonical_field_order", "maximum_state_bytes_formula"} and
            schema["schema"] == "strata-rm-v4-decoded-semantic-state" and
            isinstance(schema["version"], int) and schema["version"] >= 1 and
            isinstance(schema["state_fields"], list) and schema["state_fields"] and
            all(isinstance(name, str) and name for name in schema["state_fields"]) and
            len(schema["state_fields"]) == len(set(schema["state_fields"])) and
            schema["raw_packet_bytes_permitted"] is False and
            schema["complete_quantizer_decisions"] is True and
            isinstance(schema["canonical_field_order"], list) and
            schema["canonical_field_order"] == schema["state_fields"] and
            schema["maximum_state_bytes_formula"] ==
            "min(8*packet_bytes+1048576,536870912)",
            "semantic state schema authority")
    receipt_payload = regular_bytes(root / "AUDIT_RECEIPT.json",
                                    "semantic decoder receipt")
    require(sha256(receipt_payload) == expected_receipt_sha256,
            "semantic decoder receipt pin")
    receipt = strict_json(receipt_payload, "semantic decoder receipt")
    required_receipt = {
        "schema", "executed", "status", "audit_source_root_sha256",
        "decoder_sha256", "canonical_encoder_sha256", "semantic_schema_sha256",
        "sandbox_sha256", "decoder_only_safe_packet_import",
        "canonical_encoder_zero_imports", "semantic_decode_complete",
        "semantic_state_excludes_raw_packet",
        "canonical_encoder_independent_from_decoder",
        "canonical_encoder_no_packet_capability", "causal_decisions_regenerated",
        "complete_packet_consumption_verified", "trailing_bytes_rejected",
        "noncanonical_alias_rejection_verified",
        "decode_then_independent_encode_verified", "canonical_uniqueness_verified",
        "fixed_universal_swiglu_moe_decoder", "qwen_specific_tables_absent",
        "hostile_tests", "payloads_opened"}
    require(canonical_json(receipt) + b"\n" == receipt_payload and
            set(receipt) == required_receipt and receipt["schema"] ==
            "strata-rm-global-swap-v4-semantic-decoder-audit-receipt" and
            receipt["executed"] is True and receipt["status"] ==
            "PASS_INDEPENDENT_SEMANTIC_CANONICALITY_AUDIT_V4" and
            receipt["audit_source_root_sha256"] == expected_source_root_sha256 and
            receipt["decoder_sha256"] == expected_decoder_sha256 and
            receipt["canonical_encoder_sha256"] ==
            expected_canonical_encoder_sha256 and
            receipt["semantic_schema_sha256"] == expected_semantic_schema_sha256 and
            receipt["sandbox_sha256"] == expected_sandbox_sha256 and
            all(receipt[name] is True for name in
                ("decoder_only_safe_packet_import", "canonical_encoder_zero_imports",
                 "semantic_decode_complete", "semantic_state_excludes_raw_packet",
                 "canonical_encoder_independent_from_decoder",
                 "canonical_encoder_no_packet_capability",
                 "causal_decisions_regenerated", "complete_packet_consumption_verified",
                 "trailing_bytes_rejected", "noncanonical_alias_rejection_verified",
                 "decode_then_independent_encode_verified",
                 "canonical_uniqueness_verified",
                 "fixed_universal_swiglu_moe_decoder",
                 "qwen_specific_tables_absent")) and
            isinstance(receipt["hostile_tests"], int) and
            receipt["hostile_tests"] >= 20 and receipt["payloads_opened"] == 0,
            "independently audited semantic canonicality receipt")
    return {"decoder_payload": decoder, "canonical_encoder_payload": encoder,
            "semantic_schema_payload": schema_payload, "semantic_schema": schema,
            "manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": expected_source_root_sha256,
            "receipt_sha256": expected_receipt_sha256,
            "decoder_sha256": expected_decoder_sha256,
            "canonical_encoder_sha256": expected_canonical_encoder_sha256,
            "semantic_schema_sha256": expected_semantic_schema_sha256,
            "status": "PASS_SEPARATELY_PINNED_SEMANTIC_DECODER_AND_ENCODER_AUDIT"}


def _strict_commitment(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(is_sha256(expected_sha256), "commitment external pin")
    payload = regular_bytes(path, "physical commitment")
    require(sha256(payload) == expected_sha256,
            "physical commitment out-of-band pin")
    record = strict_json(payload, "physical commitment")
    require(canonical_json(record) + b"\n" == payload,
            "physical commitment canonical bytes")
    required = {"schema", "mode", "v3_source_root_sha256",
                "v3_review_source_root_sha256", "runtime_capability_sha256",
                "decoder_sha256", "canonical_encoder_sha256",
                "semantic_schema_sha256", "sandbox_sha256", "route_packets"}
    require(set(record) == required and record["schema"] ==
            "strata-rm-global-swap-v4-routed-expert-physical-commitment" and
            record["mode"] == "production_routed_expert" and
            record["v3_source_root_sha256"] == V3_SOURCE_ROOT_SHA256 and
            record["v3_review_source_root_sha256"] ==
            V3_REVIEW_SOURCE_ROOT_SHA256 and
            all(is_sha256(record[name]) for name in
                ("runtime_capability_sha256", "decoder_sha256",
                 "canonical_encoder_sha256", "semantic_schema_sha256",
                 "sandbox_sha256")), "physical commitment schema")
    rows = record["route_packets"]
    require(isinstance(rows, list) and rows, "route packets")
    ids = set()
    paths = set()
    hashes = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) ==
                {"route_id", "relative_path", "bytes", "sha256"} and
                isinstance(row["route_id"], str) and row["route_id"] and
                row["route_id"] not in ids and
                isinstance(row["bytes"], int) and row["bytes"] > 0 and
                is_sha256(row["sha256"]), "route packet row")
        _safe_relative(row["relative_path"], "route packet")
        require(row["relative_path"] not in paths and row["sha256"] not in hashes,
                "distinct expert packet per route")
        ids.add(row["route_id"])
        paths.add(row["relative_path"])
        hashes.add(row["sha256"])
    return record


def _write_immutable(path: Path, payload: bytes, label: str) -> None:
    require(not path.exists(), f"{label}: destination fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    require(regular_bytes(path, label) == payload, f"{label}: snapshot parity")


def _sanitized_environment() -> dict[str, str]:
    allowed = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")
    result = {name: os.environ[name] for name in allowed if name in os.environ}
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONHASHSEED"] = "0"
    return result


def _read_pinned(root: Path, row: Mapping[str, Any], label: str) -> bytes:
    path = resolve_member(root, row["relative_path"], label)
    payload = regular_bytes(path, label)
    require(len(payload) == row["bytes"] and sha256(payload) == row["sha256"],
            f"{label}: literal pin")
    return payload


def _bf16(payload: bytes) -> array.array:
    require(payload and len(payload) % 2 == 0, "nonempty BF16")
    words = array.array("H")
    words.frombytes(payload)
    if sys.byteorder != "little":
        words.byteswap()
    wide = array.array("I", (int(value) << 16 for value in words))
    result = array.array("f")
    result.frombytes(wide.tobytes())
    return result


def _f64(payload: bytes) -> array.array:
    require(payload and len(payload) % 8 == 0, "nonempty FP64")
    result = array.array("d")
    result.frombytes(payload)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def _score(source_payload: bytes, reconstruction_payload: bytes) -> dict[str, Any]:
    source = _bf16(source_payload)
    reconstruction = _f64(reconstruction_payload)
    require(len(source) == len(reconstruction) and
            all(math.isfinite(value) for value in source) and
            all(math.isfinite(value) for value in reconstruction),
            "finite source/reconstruction")
    energy = math.fsum(float(value) ** 2 for value in source)
    sse = math.fsum((float(left) - float(right)) ** 2
                    for left, right in zip(source, reconstruction, strict=True))
    require(energy > 0.0 and math.isfinite(sse), "finite scoring domain")
    return {"weights": len(source), "sse_fp64_hex": sse.hex(),
            "energy_fp64_hex": energy.hex()}


def _validate_sandbox_receipt(record: Mapping[str, Any], *, route_id: str,
                              packet: bytes, runtime: Mapping[str, Any],
                              semantic: Mapping[str, Any], sandbox_sha: str
                              ) -> dict[str, Any]:
    required = {
        "schema", "route_id", "sandbox_sha256", "runtime_capability_sha256",
        "runtime_tree_root_sha256", "python_distribution_version",
        "wasmtime_runtime_version", "module_tree_root_sha256",
        "native_library_root_sha256", "native_libraries_loaded",
        "decoder_imports", "canonical_encoder_imports", "wasi_enabled",
        "store_limits_installed_before_instantiation", "store_memory_limit_bytes",
        "fuel_budget", "decoder_fuel_remaining", "encoder_fuel_remaining",
        "packet_host_buffer_immutable", "packet_capability",
        "packet_sha256", "packet_bytes", "packet_read_operations",
        "literal_bytes_supplied_total", "unique_literal_bytes_supplied",
        "pages_supplied", "physical_page_bytes_supplied", "semantic_state_bytes",
        "semantic_schema_sha256", "canonical_encoder_received_packet_capability",
        "canonical_packet_bytes", "canonical_packet_sha256", "decode_status",
        "status"}
    capability = runtime["capability"]
    expected_fuel = min(MAX_FUEL, FUEL_BASE + FUEL_PER_PACKET_BYTE * len(packet))
    expected_page_count = (len(packet) + PAGE_BYTES - 1) // PAGE_BYTES
    require(set(record) == required and record["schema"] ==
            "strata-rm-global-swap-v4-pinned-wasmtime-sandbox-receipt" and
            record["route_id"] == route_id and record["sandbox_sha256"] == sandbox_sha and
            record["runtime_capability_sha256"] == runtime["capability_sha256"] and
            record["runtime_tree_root_sha256"] == runtime["runtime_tree_root_sha256"] and
            record["python_distribution_version"] ==
            capability["python_distribution_version"] and
            record["wasmtime_runtime_version"] ==
            capability["wasmtime_runtime_version"] and
            record["module_tree_root_sha256"] ==
            capability["module_tree_root_sha256"] and
            record["native_library_root_sha256"] ==
            capability["native_library_root_sha256"] and
            record["native_libraries_loaded"] == capability["native_libraries"] and
            record["decoder_imports"] ==
            [{"module": "authority", "name": "read_packet", "kind": "func"}] and
            record["canonical_encoder_imports"] == [] and
            record["wasi_enabled"] is False and
            record["store_limits_installed_before_instantiation"] is True and
            record["store_memory_limit_bytes"] == STORE_MEMORY_LIMIT_BYTES and
            record["fuel_budget"] == expected_fuel and
            isinstance(record["decoder_fuel_remaining"], int) and
            0 <= record["decoder_fuel_remaining"] < expected_fuel and
            isinstance(record["encoder_fuel_remaining"], int) and
            0 <= record["encoder_fuel_remaining"] < expected_fuel and
            record["packet_host_buffer_immutable"] is True and
            record["packet_capability"] == "read-only bounded host callback" and
            record["packet_sha256"] == sha256(packet) and
            record["packet_bytes"] == len(packet) and
            isinstance(record["packet_read_operations"], list) and
            record["unique_literal_bytes_supplied"] == len(packet) and
            record["pages_supplied"] == list(range(expected_page_count)) and
            record["physical_page_bytes_supplied"] == expected_page_count * PAGE_BYTES and
            isinstance(record["semantic_state_bytes"], int) and
            0 < record["semantic_state_bytes"] <=
            min(8 * len(packet) + 1_048_576, 536_870_912) and
            record["semantic_schema_sha256"] == semantic["semantic_schema_sha256"] and
            record["canonical_encoder_received_packet_capability"] is False and
            record["canonical_packet_bytes"] == len(packet) and
            record["canonical_packet_sha256"] == sha256(packet) and
            record["decode_status"] == 0 and record["status"] ==
            "PASS_PINNED_BOUNDED_IMMUTABLE_SEMANTIC_WASM_DECODE",
            "pinned bounded semantic sandbox receipt")
    operations = record["packet_read_operations"]
    intervals = []
    total = 0
    for row in operations:
        require(isinstance(row, dict) and set(row) == {"offset", "length"} and
                isinstance(row["offset"], int) and row["offset"] >= 0 and
                isinstance(row["length"], int) and row["length"] > 0 and
                row["offset"] + row["length"] <= len(packet),
                "packet capability operation")
        total += row["length"]
        intervals.append((row["offset"], row["offset"] + row["length"]))
    intervals.sort()
    require(intervals, "packet capability used")
    covered = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            covered += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    covered += end - start
    require(total == record["literal_bytes_supplied_total"] == len(packet) and
            covered == record["unique_literal_bytes_supplied"] == len(packet),
            "complete exactly-once packet capability consumption")
    return {"literal_packet_bytes": len(packet), "page_bytes": PAGE_BYTES,
            "pages_supplied": expected_page_count,
            "physical_page_bytes": expected_page_count * PAGE_BYTES,
            "cold_read_amplification": expected_page_count * PAGE_BYTES / len(packet),
            "one_independently_routed_expert": True,
            "immutable_host_packet_capability": True}


def _run_route(*, route: Mapping[str, Any], packet_row: Mapping[str, Any],
               evidence_root: Path, runtime: Mapping[str, Any],
               semantic: Mapping[str, Any], sandbox_payload: bytes,
               timeout_seconds: int) -> dict[str, Any]:
    packet = _read_pinned(evidence_root, packet_row, f"packet {route['route_id']}")
    sources = [_read_pinned(evidence_root, source,
                            f"source {route['route_id']}:{source['role']}")
               for source in route["sources"]]
    request = {"schema": "strata-rm-global-swap-v4-semantic-route-request",
               "route_id": route["route_id"], "packet_sha256": sha256(packet),
               "packet_bytes": len(packet), "page_bytes": PAGE_BYTES,
               "semantic_schema_sha256": semantic["semantic_schema_sha256"],
               "sources": [{key: source[key] for key in
                            ("ordinal", "role", "layer", "expert", "shape")}
                           for source in route["sources"]]}
    with tempfile.TemporaryDirectory(prefix="strata-rm-v4-route-") as directory:
        root = Path(directory).resolve(strict=True)
        runtime_root = root / "runtime"
        for row in runtime["runtime_rows"]:
            relative = Path(*PurePosixPath(row["path"]).parts[1:])
            _write_immutable(runtime_root / relative,
                             runtime["runtime_payloads"][row["path"]],
                             f"runtime snapshot {row['path']}")
        runtime_capability = root / "RUNTIME_CAPABILITY.json"
        decoder = root / "DECODER.wasm"
        encoder = root / "CANONICAL_ENCODER.wasm"
        semantic_schema = root / "SEMANTIC_SCHEMA.json"
        sandbox = root / "wasm_runtime_sandbox.py"
        packet_path = root / "packet.bin"
        request_path = root / "request.json"
        output = root / "output"
        output.mkdir()
        _write_immutable(runtime_capability, runtime["capability_payload"],
                         "runtime capability snapshot")
        _write_immutable(decoder, semantic["decoder_payload"], "decoder snapshot")
        _write_immutable(encoder, semantic["canonical_encoder_payload"],
                         "canonical encoder snapshot")
        _write_immutable(semantic_schema, semantic["semantic_schema_payload"],
                         "semantic schema snapshot")
        _write_immutable(sandbox, sandbox_payload, "sandbox snapshot")
        _write_immutable(packet_path, packet, "packet snapshot")
        _write_immutable(request_path, canonical_json(request) + b"\n",
                         "request snapshot")
        command = [sys.executable, "-I", "-B", str(sandbox),
                   "--runtime-root", str(runtime_root),
                   "--runtime-capability", str(runtime_capability),
                   "--decoder", str(decoder), "--canonical-encoder", str(encoder),
                   "--semantic-schema", str(semantic_schema),
                   "--packet", str(packet_path), "--request", str(request_path),
                   "--output-dir", str(output)]
        completed = subprocess.run(
            command, cwd=root, env=_sanitized_environment(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout_seconds, check=False)
        require(completed.returncode == 0,
                "pinned Wasmtime sandbox failed: " +
                completed.stderr.decode("utf-8", errors="replace")[-4000:])
        receipt = strict_json(regular_bytes(output / "SANDBOX_RECEIPT.json",
                                            "sandbox receipt"), "sandbox receipt")
        read = _validate_sandbox_receipt(
            receipt, route_id=route["route_id"], packet=packet,
            runtime=runtime, semantic=semantic,
            sandbox_sha=sha256(sandbox_payload))
        canonical = regular_bytes(output / "canonical_packet.bin",
                                  "canonical packet")
        reconstructions = [regular_bytes(
            output / f"reconstruction-{source['ordinal']:04d}.f64",
            f"reconstruction {source['ordinal']}") for source in route["sources"]]
        require(sha256(regular_bytes(sandbox, "post-run sandbox")) ==
                sha256(sandbox_payload) and
                sha256(regular_bytes(decoder, "post-run decoder")) ==
                semantic["decoder_sha256"] and
                sha256(regular_bytes(encoder, "post-run canonical encoder")) ==
                semantic["canonical_encoder_sha256"],
                "executable snapshots unchanged")
    require(canonical == packet, "semantic decode plus independent canonical encode")
    scores = [_score(source, reconstruction)
              for source, reconstruction in zip(sources, reconstructions, strict=True)]
    weights = sum(row["weights"] for row in scores)
    sse = math.fsum(float.fromhex(row["sse_fp64_hex"]) for row in scores)
    energy = math.fsum(float.fromhex(row["energy_fp64_hex"]) for row in scores)
    rate = 8.0 * len(packet) / weights
    relative = sse / energy
    factor = relative * 2.0 ** (2.0 * rate)
    return {"route_id": route["route_id"], "kind": route["kind"],
            "architecture_family": route["architecture_family"],
            "control_family": route["control_family"],
            "paired_model_route_id": route["paired_model_route_id"],
            "layer": route["sources"][0]["layer"],
            "expert": route["sources"][0]["expert"],
            "weights": weights, "literal_packet_bytes": len(packet),
            "physical_rate_bpw": rate, "sse_fp64_hex": sse.hex(),
            "energy_fp64_hex": energy.hex(), "relative_mse": relative,
            "F": factor, "saving_bpw": -0.5 * math.log2(factor),
            "cold_read": read, "matrix_rows": scores,
            "semantic_canonicality_independently_audited": True}


def validate_physical_bundle(
        *, v4_package: Path, expected_v4_manifest_sha256: str,
        v3_package: Path, v3_review_package: Path,
        evidence_root: Path, commitment_path: Path,
        expected_commitment_sha256: str,
        scientific_audit_package: Path,
        expected_scientific_manifest_sha256: str,
        expected_scientific_source_root_sha256: str,
        expected_scientific_receipt_sha256: str,
        expected_scientific_capability_sha256: str,
        runtime_audit_package: Path, expected_runtime_manifest_sha256: str,
        expected_runtime_source_root_sha256: str,
        expected_runtime_receipt_sha256: str,
        expected_runtime_capability_sha256: str,
        expected_runtime_tree_root_sha256: str,
        semantic_decoder_audit_package: Path,
        expected_decoder_manifest_sha256: str,
        expected_decoder_source_root_sha256: str,
        expected_decoder_receipt_sha256: str,
        expected_decoder_sha256: str,
        expected_canonical_encoder_sha256: str,
        expected_semantic_schema_sha256: str,
        authorization: str, timeout_seconds: int = 3600) -> dict[str, Any]:
    require(authorization == PRODUCTION_AUTHORIZATION,
            "explicit v4 physical authorization")
    audit_roots = [real_directory(scientific_audit_package,
                                  "scientific audit package identity"),
                   real_directory(runtime_audit_package,
                                  "runtime audit package identity"),
                   real_directory(semantic_decoder_audit_package,
                                  "semantic decoder audit package identity")]
    require(len(set(audit_roots)) == 3 and len({
                expected_scientific_source_root_sha256,
                expected_runtime_source_root_sha256,
                expected_decoder_source_root_sha256}) == 3,
            "three physically and cryptographically independent audits")
    v4 = authenticate_v4_package(v4_package, expected_v4_manifest_sha256)
    lineage = authenticate_v3_and_review(v3_package, v3_review_package)
    v3 = load_v3_authority_snapshot(lineage)
    scientific = v3.authenticate_scientific_audit_package(
        scientific_audit_package,
        expected_manifest_sha256=expected_scientific_manifest_sha256,
        expected_source_root_sha256=expected_scientific_source_root_sha256,
        expected_receipt_sha256=expected_scientific_receipt_sha256,
        expected_capability_sha256=expected_scientific_capability_sha256)
    runtime = authenticate_runtime_audit_package(
        runtime_audit_package,
        expected_manifest_sha256=expected_runtime_manifest_sha256,
        expected_source_root_sha256=expected_runtime_source_root_sha256,
        expected_receipt_sha256=expected_runtime_receipt_sha256,
        expected_capability_sha256=expected_runtime_capability_sha256,
        expected_runtime_tree_root_sha256=expected_runtime_tree_root_sha256)
    sandbox_payload = regular_bytes(Path(v4["path"]) / "wasm_runtime_sandbox.py",
                                    "v4 runtime sandbox")
    semantic = authenticate_semantic_decoder_audit_package(
        semantic_decoder_audit_package,
        expected_manifest_sha256=expected_decoder_manifest_sha256,
        expected_source_root_sha256=expected_decoder_source_root_sha256,
        expected_receipt_sha256=expected_decoder_receipt_sha256,
        expected_decoder_sha256=expected_decoder_sha256,
        expected_canonical_encoder_sha256=expected_canonical_encoder_sha256,
        expected_semantic_schema_sha256=expected_semantic_schema_sha256,
        expected_sandbox_sha256=sha256(sandbox_payload))
    evidence = real_directory(evidence_root, "evidence root")
    try:
        relative = str(Path(commitment_path).resolve(strict=True).relative_to(evidence)
                       ).replace(os.sep, "/")
    except (OSError, ValueError) as exc:
        raise AuthorityError("commitment inside evidence root") from exc
    commitment = _strict_commitment(
        resolve_member(evidence, relative, "physical commitment"),
        expected_commitment_sha256)
    require(commitment["runtime_capability_sha256"] ==
            runtime["capability_sha256"] and
            commitment["decoder_sha256"] == semantic["decoder_sha256"] and
            commitment["canonical_encoder_sha256"] ==
            semantic["canonical_encoder_sha256"] and
            commitment["semantic_schema_sha256"] ==
            semantic["semantic_schema_sha256"] and
            commitment["sandbox_sha256"] == sha256(sandbox_payload),
            "commitment-to-runtime/semantic authority binding")
    packet_rows = {row["route_id"]: row for row in commitment["route_packets"]}
    require(set(packet_rows) == set(scientific["routes"]),
            "one packet for every audited route")
    results = [_run_route(
        route=route, packet_row=packet_rows[route_id], evidence_root=evidence,
        runtime=runtime, semantic=semantic, sandbox_payload=sandbox_payload,
        timeout_seconds=timeout_seconds)
        for route_id, route in scientific["routes"].items()]
    acceptance = v3.evaluate_acceptance(results, scientific["record"], enforce=True)
    return {"schema": "strata-rm-global-swap-v4-routed-physical-result",
            "commitment_sha256": expected_commitment_sha256,
            "runtime_authority": {key: runtime[key] for key in
                                  ("manifest_sha256", "source_root_sha256",
                                   "receipt_sha256", "capability_sha256",
                                   "runtime_tree_root_sha256", "status")},
            "semantic_decoder_authority": {key: semantic[key] for key in
                                           ("manifest_sha256", "source_root_sha256",
                                            "receipt_sha256", "decoder_sha256",
                                            "canonical_encoder_sha256",
                                            "semantic_schema_sha256", "status")},
            "routes": results, "acceptance": acceptance,
            "wasmtime_distribution_and_native_libraries_pinned": True,
            "store_limits_and_fuel_installed_before_execution": True,
            "packet_is_host_owned_and_guest_immutable": True,
            "semantic_canonicality_independently_audited": True,
            "caller_supplied_metrics_accepted": False,
            "status": "PASS_FINAL_PINNED_BOUNDED_SEMANTIC_WASM_AUTHORITY"}
