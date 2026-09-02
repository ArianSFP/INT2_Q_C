#!/usr/bin/env python3
"""Fail-closed independent audit of one real N18-v6 Qwen pilot result.

The unresolved package contains no payload path or result digest.  A run needs
an independently hashed external pin bundle.  All executable and data bytes
are retained and authenticated before NumPy or the separately pinned polar
decoder is loaded.  The producer's RESULT is comparison-only, never a
numerical input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import types
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


AUTHORIZATION = "AUDIT_EXACT_TACTIC_ACTUAL_COARSE_N18_V6_QWEN_RESULT_V0"
PINS_SCHEMA = "tactic-actual-coarse-n18-v6-qwen-result-external-pins-v0"
AUDIT_SCHEMA = "tactic-actual-coarse-n18-v6-qwen-independent-result-audit-v0"
OWN_MANIFEST_SCHEMA = "tactic-actual-coarse-n18-v6-result-auditor-source-manifest-v0"
OWN_MANIFEST_STATUS = "SEALED_SOURCE_ONLY_AWAITING_EXTERNAL_QWEN_RESULT_PINS"
OWN_ROOT_DOMAIN = b"TACTIC-ACTUAL-COARSE-N18-V6-RESULT-AUDITOR-SOURCE-ROOT-V0\0"

KNOWN_V6_MANIFEST_SHA256 = "31662539a4c55926f47b378d15a0d8e23c90aa0903328c44be2e237eca48b15d"
KNOWN_V6_SOURCE_ROOT_SHA256 = "161ab23169af3427648ec1bbcb9402568a0fb8aefc4a794daf3ebd1c56cc83f2"
KNOWN_PREDECESSOR_LOCK_SHA256 = "645310404673e944c0f61e08747b4d7d50e6681cd450eb829acd8614c41f4322"
KNOWN_RUNTIME_LOCK_SHA256 = "de1464d23de161d90f0784183743252631385ad69ba2620697dea7df763c3490"
KNOWN_V4_SOURCE_ROOT_SHA256 = "1f9f2c92df3796f5f23b7e3a6b0826d6d8a2ea53bc70014fb75e61e7bc8a9fbf"
KNOWN_FROZEN_DECODER_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"
KNOWN_FROZEN_DECODER_BYTES = 116_835
KNOWN_POLARIS_ENCODER_SHA256 = "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0"
KNOWN_POLARIS_ENCODER_BYTES = 29_633
KNOWN_SMOKE_RECEIPT_FILE_SHA256 = "480e6d0667380ac7b3bc6ab6be9f54db0348cd2b78e1ca6948962d7915b8b1cd"

PUBLICATION_DATA_MEMBERS = frozenset({
    "COARSE.bin",
    "ENCODER_RECEIPT.json",
    "DECODER_RECEIPT.json",
    "INPUT_BINDING.json",
    "RUNTIME_RECEIPT.json",
    "SMOKE_BINDING.json",
    "RESULT.json",
})
PUBLICATION_MEMBERS = PUBLICATION_DATA_MEMBERS | {"COMPLETE.json"}
MAX_SOURCE = 4 * (1 << 20)
MAX_JSON = 32 * (1 << 20)
MAX_FRAME = 64 * (1 << 20)
MAX_INPUT = 1 << 34


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(AuditError(f"{label}: nonfinite {item}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label}: JSON: {error}") from error
    require(isinstance(value, dict), f"{label}: object")
    _finite_tree(value, label, 0)
    return value


def _finite_tree(value: Any, label: str, depth: int) -> None:
    require(depth <= 64, f"{label}: JSON depth")
    if value is None or isinstance(value, (bool, str)) or type(value) is int:
        return
    if type(value) is float:
        require(math.isfinite(value), f"{label}: finite float")
        return
    if isinstance(value, list):
        for item in value:
            _finite_tree(item, label, depth + 1)
        return
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), f"{label}: keys")
        for item in value.values():
            _finite_tree(item, label, depth + 1)
        return
    raise AuditError(f"{label}: unsupported JSON value")


def digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"{label}: digest")
    return value


def exact_int(value: Any, label: str, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    require(type(value) is int and minimum <= value <= maximum, f"{label}: integer")
    return value


def safe_leaf(value: Any, label: str) -> str:
    require(isinstance(value, str) and 0 < len(value) <= 127, f"{label}: name")
    require(value not in {".", ".."} and "/" not in value and "\\" not in value, f"{label}: safe leaf")
    return value


def absolute_path(value: Any, label: str, *, directory: bool | None = None) -> Path:
    require(isinstance(value, str) and value.startswith("/"), f"{label}: absolute POSIX path")
    pure = PurePosixPath(value)
    require(".." not in pure.parts and "." not in pure.parts, f"{label}: normalized path")
    path = Path(value)
    reject_symlink_chain(path, label)
    metadata = os.lstat(path)
    if directory is True:
        require(stat.S_ISDIR(metadata.st_mode), f"{label}: directory")
    elif directory is False:
        require(stat.S_ISREG(metadata.st_mode), f"{label}: regular")
    return path


def reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode), f"{label}: symlink {cursor}")
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size,
        metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_nlink,
    )


@dataclass
class HeldRegular:
    path: Path
    descriptor: int
    identity: tuple[int, int, int, int, int, int, int]
    data: bytes
    sha256: str

    @classmethod
    def open(
        cls, path: Path, *, label: str, maximum: int,
        expected_bytes: int | None = None, expected_sha256: str | None = None,
    ) -> "HeldRegular":
        require(path.is_absolute(), f"{label}: absolute")
        reject_symlink_chain(path, label)
        descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        try:
            before = os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum, f"{label}: regular size")
            if expected_bytes is not None:
                require(before.st_size == expected_bytes, f"{label}: exact bytes")
            chunks = []
            offset = 0
            while offset < before.st_size:
                chunk = os.pread(descriptor, min(1 << 20, before.st_size - offset), offset)
                require(bool(chunk), f"{label}: short read")
                chunks.append(chunk)
                offset += len(chunk)
            require(os.pread(descriptor, 1, before.st_size) == b"", f"{label}: trailing read")
            data = b"".join(chunks)
            observed = sha256(data)
            if expected_sha256 is not None:
                require(observed == expected_sha256, f"{label}: SHA-256")
            held = cls(path, descriptor, _identity(before), data, observed)
            held.verify()
            return held
        except BaseException:
            os.close(descriptor)
            raise

    def verify(self) -> None:
        require(_identity(os.fstat(self.descriptor)) == self.identity, f"held file drift: {self.path}")
        named = os.stat(self.path, follow_symlinks=False)
        require(_identity(named) == self.identity, f"held file rebound: {self.path}")
        require(sha256(self.data) == self.sha256, f"held memory drift: {self.path}")

    def close(self) -> None:
        os.close(self.descriptor)


@dataclass
class HeldDirectory:
    path: Path
    descriptor: int
    identity: tuple[int, int, int, int, int, int, int]
    expected_names: frozenset[str]

    @classmethod
    def open(cls, path: Path, expected_names: set[str] | frozenset[str], label: str) -> "HeldDirectory":
        reject_symlink_chain(path, label)
        descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        before = os.fstat(descriptor)
        try:
            require(stat.S_ISDIR(before.st_mode), f"{label}: directory")
            held = cls(path, descriptor, _identity(before), frozenset(expected_names))
            held.verify()
            return held
        except BaseException:
            os.close(descriptor)
            raise

    def verify(self) -> None:
        require(_identity(os.fstat(self.descriptor)) == self.identity, f"directory drift: {self.path}")
        require(_identity(os.stat(self.path, follow_symlinks=False)) == self.identity, f"directory rebound: {self.path}")
        entries = list(os.scandir(self.descriptor))
        require({entry.name for entry in entries} == set(self.expected_names), f"directory member set: {self.path}")
        require(all(entry.is_file(follow_symlinks=False) for entry in entries), f"directory regular closure: {self.path}")

    def close(self) -> None:
        os.close(self.descriptor)


def source_root(rows: list[dict[str, Any]], *, domain: bytes | None = None) -> str:
    ordered = sorted(rows, key=lambda row: row["name"].encode("ascii"))
    return sha256((domain or b"") + canonical_json(ordered))


def v4_source_root(rows: list[dict[str, Any]]) -> str:
    output = hashlib.sha256(b"TACTIC-N18-V4-SOURCE-ROOT-v1\0")
    for row in sorted(rows, key=lambda item: item["name"].encode("utf-8")):
        name = row["name"].encode("utf-8")
        output.update(len(name).to_bytes(4, "big"))
        output.update(name)
        output.update(int(row["bytes"]).to_bytes(8, "big"))
        output.update(bytes.fromhex(row["sha256"]))
    return output.hexdigest()


def parse_member_pins(value: Any, expected_names: set[str] | frozenset[str], label: str, maximum: int) -> dict[str, dict[str, Any]]:
    require(isinstance(value, Mapping) and set(value) == set(expected_names), f"{label}: exact names")
    result = {}
    for name in expected_names:
        row = value[name]
        require(isinstance(row, Mapping) and set(row) == {"bytes", "sha256"}, f"{label}.{name}: row")
        result[name] = {
            "bytes": exact_int(row["bytes"], f"{label}.{name}.bytes", 1, maximum),
            "sha256": digest(row["sha256"], f"{label}.{name}.sha256"),
        }
    return result


def parse_pins(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "pins object")
    require(set(value) == {"schema", "status", "paths", "hashes", "input_roles", "publication_members"}, "pins exact fields")
    require(value["schema"] == PINS_SCHEMA and value["status"] == "EXTERNALLY_RECORDED_AFTER_TERMINAL_PUBLICATION", "pins resolved status")
    paths = value["paths"]
    require(isinstance(paths, Mapping) and set(paths) == {
        "v6_package", "v4_package", "frozen_decoder_core", "polaris_encoder_source",
        "smoke_receipt", "input_manifest", "publication_directory",
    }, "pins paths")
    for key, path in paths.items():
        require(isinstance(path, str) and path.startswith("/") and ".." not in PurePosixPath(path).parts, f"pins path {key}")
    hashes = value["hashes"]
    require(isinstance(hashes, Mapping) and set(hashes) == {
        "v6_source_manifest_sha256", "v6_source_root_sha256", "predecessor_lock_sha256",
        "runtime_lock_sha256", "v4_source_root_sha256", "frozen_decoder_core_sha256",
        "polaris_encoder_source_sha256", "smoke_receipt_sha256", "input_manifest_sha256",
    }, "pins hashes")
    expected = {
        "v6_source_manifest_sha256": KNOWN_V6_MANIFEST_SHA256,
        "v6_source_root_sha256": KNOWN_V6_SOURCE_ROOT_SHA256,
        "predecessor_lock_sha256": KNOWN_PREDECESSOR_LOCK_SHA256,
        "runtime_lock_sha256": KNOWN_RUNTIME_LOCK_SHA256,
        "v4_source_root_sha256": KNOWN_V4_SOURCE_ROOT_SHA256,
        "frozen_decoder_core_sha256": KNOWN_FROZEN_DECODER_SHA256,
        "polaris_encoder_source_sha256": KNOWN_POLARIS_ENCODER_SHA256,
        "smoke_receipt_sha256": KNOWN_SMOKE_RECEIPT_FILE_SHA256,
    }
    for key, expected_digest in expected.items():
        require(digest(hashes[key], f"pins {key}") == expected_digest, f"pins frozen hash {key}")
    digest(hashes["input_manifest_sha256"], "pins input manifest")
    roles = value["input_roles"]
    require(isinstance(roles, Mapping) and set(roles) == {"gate", "up", "down_transposed"}, "pins input roles")
    clean_roles = {}
    for role, row in roles.items():
        require(isinstance(row, Mapping) and set(row) == {"absolute_path", "bytes", "sha256"}, f"pins role {role}")
        require(isinstance(row["absolute_path"], str) and row["absolute_path"].startswith("/"), f"pins role path {role}")
        clean_roles[role] = {
            "absolute_path": row["absolute_path"],
            "bytes": exact_int(row["bytes"], f"pins role bytes {role}", 1, MAX_INPUT),
            "sha256": digest(row["sha256"], f"pins role hash {role}"),
        }
    publications = parse_member_pins(value["publication_members"], PUBLICATION_MEMBERS, "publication pins", MAX_FRAME)
    return {"paths": dict(paths), "hashes": dict(hashes), "input_roles": clean_roles, "publication_members": publications}


def authenticate_source_package(path: Path, expected_manifest_sha256: str, *, own: bool = False) -> dict[str, Any]:
    manifest = HeldRegular.open(path / "SOURCE_MANIFEST.json", label="source manifest", maximum=MAX_SOURCE, expected_sha256=expected_manifest_sha256)
    record = strict_json(manifest.data, "source manifest")
    if own:
        require(record.get("schema") == OWN_MANIFEST_SCHEMA and record.get("status") == OWN_MANIFEST_STATUS, "own source manifest schema/status")
    else:
        require(record.get("schema") == "tactic-actual-coarse-n18-v6-source-manifest-v1", "v6 manifest schema")
        require(record.get("source_root_sha256") == KNOWN_V6_SOURCE_ROOT_SHA256, "v6 manifest root pin")
    rows = record.get("members")
    require(isinstance(rows, list) and rows, "source manifest rows")
    clean_rows = []
    members: dict[str, HeldRegular] = {}
    try:
        names = []
        for row in rows:
            require(isinstance(row, Mapping) and set(row) == {"name", "bytes", "sha256"}, "source row")
            name = safe_leaf(row["name"], "source row name")
            require(name != "SOURCE_MANIFEST.json" and name not in names, "source unique member")
            held = HeldRegular.open(path / name, label=f"source {name}", maximum=MAX_SOURCE, expected_bytes=exact_int(row["bytes"], "source bytes", 1, MAX_SOURCE), expected_sha256=digest(row["sha256"], "source hash"))
            members[name] = held
            names.append(name)
            clean_rows.append({"name": name, "bytes": len(held.data), "sha256": held.sha256})
        require(names == sorted(names, key=lambda name: name.encode("ascii")), "source canonical order")
        expected_root = source_root(clean_rows, domain=OWN_ROOT_DOMAIN if own else None)
        root_key = "source_snapshot_root_sha256" if own else "source_root_sha256"
        require(record.get(root_key) == expected_root, "source manifest recomputed root")
        expected_names = set(names) | {"SOURCE_MANIFEST.json"}
        directory = HeldDirectory.open(path, expected_names, "source package")
        if own:
            executing = members.get(Path(__file__).name)
            require(executing is not None and _identity(os.lstat(__file__))[:2] == executing.identity[:2], "own executing inode binding")
        return {
            "manifest": manifest, "record": record, "members": members, "directory": directory,
            "manifest_sha256": manifest.sha256, "root_sha256": expected_root,
            "rows": clean_rows,
        }
    except BaseException:
        for held in members.values():
            held.close()
        manifest.close()
        raise


def close_source(package: Mapping[str, Any]) -> None:
    package["directory"].verify()
    package["manifest"].verify()
    for held in package["members"].values():
        held.verify()
    package["directory"].close()
    package["manifest"].close()
    for held in package["members"].values():
        held.close()


def authenticate_v4(v4_path: Path, predecessor: Mapping[str, Any]) -> dict[str, Any]:
    require(predecessor.get("schema") == "tactic-actual-coarse-n18-v6-predecessor-lock-v1", "predecessor schema")
    require(predecessor.get("source_root_sha256") == KNOWN_V4_SOURCE_ROOT_SHA256, "predecessor root")
    rows = predecessor.get("members")
    require(isinstance(rows, list) and rows, "predecessor rows")
    held = {}
    clean = []
    try:
        for row in rows:
            require(isinstance(row, Mapping) and set(row) == {"name", "bytes", "sha256"}, "v4 row")
            name = safe_leaf(row["name"], "v4 name")
            item = HeldRegular.open(v4_path / name, label=f"v4 {name}", maximum=MAX_SOURCE, expected_bytes=exact_int(row["bytes"], "v4 bytes", 1, MAX_SOURCE), expected_sha256=digest(row["sha256"], "v4 hash"))
            held[name] = item
            clean.append({"name": name, "bytes": len(item.data), "sha256": item.sha256})
        require(v4_source_root(clean) == KNOWN_V4_SOURCE_ROOT_SHA256, "v4 source root recomputation")
        directory = HeldDirectory.open(v4_path, set(held), "v4 package")
        return {"members": held, "rows": clean, "directory": directory}
    except BaseException:
        for item in held.values():
            item.close()
        raise


def validate_smoke(core: Any, record: Mapping[str, Any], v6: Mapping[str, Any], predecessor: Mapping[str, Any]) -> dict[str, Any]:
    require(set(record) == {
        "schema", "status", "source_closure", "runtime_closure", "numeric_tile",
        "i32_stress_lifetime", "aggregate_zero_frame", "traffic_ledgers",
        "payload_accessed", "model_or_qwen_path_discovered_or_enumerated",
        "claim_boundary", "receipt_sha256",
    }, "smoke fields")
    require(record["schema"] == "tactic-actual-coarse-n18-v6-source-free-cupy-smoke-v1" and record["status"] == "PASS_SOURCE_FREE_V6_REPAIRS_CUPY_SOURCE_BOUND", "smoke schema/status")
    clean = dict(record)
    claimed = digest(clean.pop("receipt_sha256"), "smoke internal receipt")
    require(core.sha256(core.canonical_json(clean)) == claimed, "smoke internal seal")
    source = record["source_closure"]
    expected_hashes = {name: row["sha256"] for name, row in ((row["name"], row) for row in v6["rows"])}
    require(source.get("source_manifest_sha256") == KNOWN_V6_MANIFEST_SHA256 and source.get("source_root_sha256") == KNOWN_V6_SOURCE_ROOT_SHA256, "smoke source closure")
    require(source.get("member_hashes") == expected_hashes and source.get("retained_no_follow_descriptors") is True, "smoke source members")
    require(source.get("executing_entry_inode_bound") is True and source.get("executing_entry_name") == "synthetic_cupy_smoke.py", "smoke executing source")
    runtime = record["runtime_closure"]
    require(runtime.get("predecessor_lock_sha256") == KNOWN_PREDECESSOR_LOCK_SHA256 and runtime.get("runtime_lock_sha256") == KNOWN_RUNTIME_LOCK_SHA256, "smoke runtime locks")
    require(runtime.get("predecessor_source_root_sha256") == predecessor["source_root_sha256"] and runtime.get("inverse_transient_dtype") == "<i4", "smoke runtime I32")
    stress = record["i32_stress_lifetime"]
    require(stress.get("input_index") == 63 and stress.get("expected_abs_max") == 8_388_608 and stress.get("observed_abs_max") == 8_388_608, "smoke I32 stress magnitude")
    require(stress.get("inverse_output_dtype_before_facade") == "<i4" and stress.get("facade_retained_dtype") == "<i4" and stress.get("no_copy_or_downcast") is True, "smoke I32 lifetime")
    tile = record["numeric_tile"]
    require(tile.get("packet_bytes") == 78_592 and tile.get("all_encoder_self_checks_required_and_passed") is True and tile.get("canonical_reencode_matches") is True, "smoke numeric tile")
    aggregate = record["aggregate_zero_frame"]
    require(aggregate.get("roles") == ["gate", "up", "down_transposed"] and aggregate.get("literal_aggregate_reencode_matches") is True, "smoke aggregate ABI")
    require(record["payload_accessed"] is False and record["model_or_qwen_path_discovered_or_enumerated"] is False, "smoke source-free")
    for key, passes in (("prebuffered_decode", 0), ("modeled_one_external_pass", 1), ("modeled_two_external_passes", 2)):
        ledger = record["traffic_ledgers"][key]
        external = ledger["external_compressed_read"]
        require(external["passes"] == passes and external["reread_bytes"] == max(0, passes - 1) * ledger["frame_bytes"], f"smoke traffic {key}")
        require(ledger["accelerator_hbm"]["below_2x_claim_authority"] is False, f"smoke HBM {key}")
    return {
        "receipt_sha256": claimed,
        "source_manifest_sha256": KNOWN_V6_MANIFEST_SHA256,
        "source_root_sha256": KNOWN_V6_SOURCE_ROOT_SHA256,
        "i32_stress_above_i16": True,
        "qwen_payload_accessed": False,
        "positive_claim_authority": False,
    }


def parse_input_manifest(core: Any, payload: bytes, path: Path, pins: Mapping[str, Any], held_files: list[HeldRegular]) -> dict[str, Any]:
    record = core.strict_json(payload, "input manifest")
    require(set(record) == {"schema", "geometry", "roles", "output_directory_name"}, "input manifest fields")
    require(record["schema"] == "tactic-actual-coarse-n18-v6-input-manifest-v1", "input schema")
    require(record["geometry"] == {"intermediate": 768, "hidden": 2048}, "Qwen input geometry")
    rows = record["roles"]
    require(isinstance(rows, list) and len(rows) == 3, "input roles")
    expected_bytes = 2 * 768 * 2048
    role_bytes = {}
    bindings = []
    for row in rows:
        require(isinstance(row, Mapping) and set(row) == {"role", "relative_path", "bytes", "sha256"}, "input role row")
        role = row["role"]
        require(role in {"gate", "up", "down_transposed"} and role not in role_bytes, "input role ABI")
        require(row["bytes"] == expected_bytes and row["sha256"] == pins[role]["sha256"], "input role size/hash")
        require(pins[role]["bytes"] == expected_bytes, "input/external role byte binding")
        relative = row["relative_path"]
        require(isinstance(relative, str) and not Path(relative).is_absolute() and ".." not in Path(relative).parts, "input relative path")
        expected_path = Path(os.path.normpath(os.fspath(path.parent / relative)))
        require(os.fspath(expected_path) == pins[role]["absolute_path"], "input/external absolute path binding")
        held = HeldRegular.open(expected_path, label=f"input BF16 {role}", maximum=MAX_INPUT, expected_bytes=expected_bytes, expected_sha256=row["sha256"])
        held_files.append(held)
        words = memoryview(held.data).cast("H")
        require(all((int(word) & 0x7F80) != 0x7F80 for word in words), f"input finite BF16 {role}")
        role_bytes[role] = held.data
        bindings.append({"role": role, "bytes": expected_bytes, "sha256": held.sha256})
    require(set(role_bytes) == {"gate", "up", "down_transposed"}, "complete input roles")
    return {
        "manifest_sha256": sha256(payload), "geometry": dict(record["geometry"]),
        "roles": bindings, "role_bytes": role_bytes,
        "output_directory_name": record["output_directory_name"],
    }


def load_authenticated_module(name: str, source: bytes, expected_sha256: str) -> Any:
    require(sha256(source) == expected_sha256 and name not in sys.modules, f"module source {name}")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{expected_sha256}>"
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def expected_runtime_receipt(predecessor: Mapping[str, Any], runtime_lock: Mapping[str, Any]) -> dict[str, Any]:
    runtime = {key: runtime_lock[key] for key in (
        "python", "implementation", "system", "machine", "numpy", "cupy", "cuda_runtime",
        "cuda_driver", "device_count", "device_name", "compute_capability",
    )}
    members = {row["name"]: row["sha256"] for row in predecessor["members"]}
    return {
        "schema": "tactic-actual-coarse-n18-v6-runtime-closure-v1",
        "predecessor_lock_sha256": KNOWN_PREDECESSOR_LOCK_SHA256,
        "predecessor_source_root_sha256": KNOWN_V4_SOURCE_ROOT_SHA256,
        "runtime_lock_sha256": KNOWN_RUNTIME_LOCK_SHA256,
        "runtime": runtime,
        "executed_v4_sources": {
            "packet_format.py": members["packet_format.py"],
            "numeric_encoder.py": members["numeric_encoder.py"],
            "independent_decoder.py": members["independent_decoder.py"],
        },
        "inverse_transient_dtype": "<i4",
        "inverse_transient_abs_bound": 8_388_608,
        "inverse_override_installed_before_any_reservoir_decode": True,
        "v4_source_modified": False,
    }


def authenticate_publication(core: Any, path: Path, member_pins: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    directory = HeldDirectory.open(path, PUBLICATION_MEMBERS, "publication")
    members = {}
    try:
        for name in PUBLICATION_MEMBERS:
            row = member_pins[name]
            maximum = MAX_FRAME if name == "COARSE.bin" else MAX_JSON
            members[name] = HeldRegular.open(path / name, label=f"publication {name}", maximum=maximum, expected_bytes=row["bytes"], expected_sha256=row["sha256"])
        json_names = PUBLICATION_MEMBERS - {"COARSE.bin"}
        parsed = {}
        for name in json_names:
            parsed[name] = core.strict_json(members[name].data, name)
            require(members[name].data == core.pretty_json(parsed[name]), f"{name}: canonical pretty encoding")
        # Static files cannot prove historical rename order, but COMPLETE may
        # not be observably older than an ordinary member.
        complete_mtime = members["COMPLETE.json"].identity[4]
        require(all(complete_mtime >= members[name].identity[4] for name in PUBLICATION_DATA_MEMBERS), "completion not older than data")
        return {"directory": directory, "members": members, "parsed": parsed}
    except BaseException:
        for member in members.values():
            member.close()
        directory.close()
        raise


def verify_publication_final(publication: Mapping[str, Any]) -> None:
    publication["directory"].verify()
    for member in publication["members"].values():
        member.verify()


def close_publication(publication: Mapping[str, Any]) -> None:
    verify_publication_final(publication)
    publication["directory"].close()
    for member in publication["members"].values():
        member.close()


def validate_input_binding(core: Any, value: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    require(set(value) == {"schema", "manifest_sha256", "geometry", "roles", "identity_fields_available_to_codec"}, "input binding fields")
    require(value["schema"] == "tactic-actual-coarse-n18-v6-input-binding-v1", "input binding schema")
    require(value["manifest_sha256"] == inputs["manifest_sha256"] and value["geometry"] == inputs["geometry"], "input binding manifest/geometry")
    require(value["roles"] == inputs["roles"] and value["identity_fields_available_to_codec"] is False, "input binding roles/nonidentity")


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    require(arguments.authorization == AUTHORIZATION, "explicit authorization")
    require(os.name == "posix" and sys.flags.isolated == 1 and sys.dont_write_bytecode, "run with Linux CPython -I -B")
    digest(arguments.expected_auditor_source_manifest_sha256, "auditor manifest")
    digest(arguments.expected_external_pins_sha256, "external pins")
    own = v6 = v4 = publication = None
    held_misc: list[HeldRegular] = []
    try:
        own_path = Path(__file__).resolve().parent
        own = authenticate_source_package(own_path, arguments.expected_auditor_source_manifest_sha256, own=True)
        core = load_authenticated_module("tacn18_v6_result_audit_core", own["members"]["audit_core.py"].data, own["members"]["audit_core.py"].sha256)

        pins_path = absolute_path(arguments.external_pins, "external pins", directory=False)
        pins_file = HeldRegular.open(pins_path, label="external pins", maximum=MAX_JSON, expected_sha256=arguments.expected_external_pins_sha256)
        held_misc.append(pins_file)
        pins_raw = core.strict_json(pins_file.data, "external pins")
        require(pins_file.data == core.pretty_json(pins_raw), "canonical external pins encoding")
        pins = parse_pins(pins_raw)

        v6_path = absolute_path(pins["paths"]["v6_package"], "v6 package", directory=True)
        v6 = authenticate_source_package(v6_path, KNOWN_V6_MANIFEST_SHA256, own=False)
        require(v6["root_sha256"] == KNOWN_V6_SOURCE_ROOT_SHA256, "v6 source root")
        predecessor = core.strict_json(v6["members"]["PREDECESSOR_LOCK.json"].data, "predecessor lock")
        runtime_lock = core.strict_json(v6["members"]["RUNTIME_LOCK.json"].data, "runtime lock")
        require(v6["members"]["PREDECESSOR_LOCK.json"].sha256 == KNOWN_PREDECESSOR_LOCK_SHA256, "predecessor lock pin")
        require(v6["members"]["RUNTIME_LOCK.json"].sha256 == KNOWN_RUNTIME_LOCK_SHA256, "runtime lock pin")
        require(runtime_lock.get("schema") == "tactic-actual-coarse-n18-v6-runtime-lock-v1", "runtime lock schema")

        v4_path = absolute_path(pins["paths"]["v4_package"], "v4 package", directory=True)
        expected_repo = v6_path.parents[1]
        require(v4_path == expected_repo / "research" / "tactic_actual_coarse_n18_v4", "v4 canonical repo binding")
        v4 = authenticate_v4(v4_path, predecessor)
        decoder_path = absolute_path(pins["paths"]["frozen_decoder_core"], "frozen decoder", directory=False)
        encoder_path = absolute_path(pins["paths"]["polaris_encoder_source"], "POLARIS encoder", directory=False)
        require(decoder_path == expected_repo / "strata_v2_klt_mixed_independent_auditor_v1.py", "decoder canonical repo binding")
        require(encoder_path == expected_repo / "src" / "polaris_sc_v2_rht_encoder.py", "encoder canonical repo binding")
        decoder_source = HeldRegular.open(decoder_path, label="frozen decoder core", maximum=MAX_SOURCE, expected_bytes=KNOWN_FROZEN_DECODER_BYTES, expected_sha256=KNOWN_FROZEN_DECODER_SHA256)
        encoder_source = HeldRegular.open(encoder_path, label="POLARIS encoder source", maximum=MAX_SOURCE, expected_bytes=KNOWN_POLARIS_ENCODER_BYTES, expected_sha256=KNOWN_POLARIS_ENCODER_SHA256)
        held_misc.extend([decoder_source, encoder_source])

        smoke_path = absolute_path(pins["paths"]["smoke_receipt"], "smoke receipt", directory=False)
        smoke_file = HeldRegular.open(smoke_path, label="smoke receipt", maximum=MAX_JSON, expected_sha256=KNOWN_SMOKE_RECEIPT_FILE_SHA256)
        held_misc.append(smoke_file)
        smoke_record = core.strict_json(smoke_file.data, "smoke receipt")
        require(smoke_file.data == core.pretty_json(smoke_record), "canonical smoke receipt encoding")
        smoke_binding = validate_smoke(core, smoke_record, v6, predecessor)
        smoke_binding_with_file = {**smoke_binding, "receipt_file_sha256": smoke_file.sha256}

        input_path = absolute_path(pins["paths"]["input_manifest"], "input manifest", directory=False)
        input_file = HeldRegular.open(input_path, label="input manifest", maximum=MAX_JSON, expected_sha256=pins["hashes"]["input_manifest_sha256"])
        held_misc.append(input_file)
        inputs = parse_input_manifest(core, input_file.data, input_path, pins["input_roles"], held_misc)

        publication_path = absolute_path(pins["paths"]["publication_directory"], "publication", directory=True)
        require(publication_path.name == inputs["output_directory_name"], "input/output namespace binding")
        publication = authenticate_publication(core, publication_path, pins["publication_members"])
        parsed = publication["parsed"]
        validate_input_binding(core, parsed["INPUT_BINDING.json"], inputs)
        runtime_expected = expected_runtime_receipt(predecessor, runtime_lock)
        core.require_deep_close(parsed["RUNTIME_RECEIPT.json"], runtime_expected, "runtime receipt")
        core.require_deep_close(parsed["SMOKE_BINDING.json"], smoke_binding_with_file, "smoke binding")

        frame = publication["members"]["COARSE.bin"].data
        # Only now, after complete closure authentication, may NumPy and the
        # separately pinned CPU numerical decoder be loaded.
        decoder = load_authenticated_module("tacn18_v6_result_audit_polar_core", decoder_source.data, decoder_source.sha256)
        import numpy as np

        require(getattr(decoder, "np", None) is np, "decoder NumPy identity")
        recomputed = core.decode_frame(frame, np, decoder, inputs["role_bytes"])
        expected_status = core.qwen_geometry_gate(recomputed)
        require(recomputed["frame_sha256"] == publication["members"]["COARSE.bin"].sha256, "frame external hash")
        expected_source_closure = {
            "source_manifest_sha256": KNOWN_V6_MANIFEST_SHA256,
            "source_root_sha256": KNOWN_V6_SOURCE_ROOT_SHA256,
            "member_hashes": {row["name"]: row["sha256"] for row in v6["rows"]},
            "retained_no_follow_descriptors": True,
            "executing_entry_inode_bound": True,
            "executing_entry_name": "dispatcher.py",
        }
        core.require_deep_close(parsed["RESULT.json"].get("source_closure"), expected_source_closure, "RESULT complete source closure")
        core.validate_producer_receipts(parsed["ENCODER_RECEIPT.json"], parsed["DECODER_RECEIPT.json"], recomputed)
        core.validate_result_material(
            parsed["RESULT.json"], recomputed, parsed["INPUT_BINDING.json"],
            expected_status=expected_status, source_root_sha256=KNOWN_V6_SOURCE_ROOT_SHA256,
            smoke_binding=smoke_binding_with_file, runtime_receipt=runtime_expected,
        )
        data_rows = {
            name: {"bytes": len(publication["members"][name].data), "sha256": publication["members"][name].sha256}
            for name in PUBLICATION_DATA_MEMBERS
        }
        core.verify_completion(
            parsed["COMPLETE.json"], data_rows, expected_status,
            source_root_sha256=KNOWN_V6_SOURCE_ROOT_SHA256,
            smoke_sha256=smoke_file.sha256, frame_sha256=recomputed["frame_sha256"],
        )
        require(parsed["COMPLETE.json"]["positive_claim_authority"] is False, "completion positive authority")
        verify_publication_final(publication)
        for held in held_misc:
            held.verify()
        v4["directory"].verify()
        for held in v4["members"].values():
            held.verify()
        v6["directory"].verify()
        own["directory"].verify()

        receipt = {
            "schema": AUDIT_SCHEMA,
            "status": "PASS_EXACT_NONPROMOTING_TACTIC_ACTUAL_COARSE_N18_V6_QWEN_RESULT_AUDIT",
            "positive_claim_authority": False,
            "auditor_source_manifest_sha256": own["manifest_sha256"],
            "auditor_source_snapshot_root_sha256": own["root_sha256"],
            "external_pins_sha256": pins_file.sha256,
            "v6_source_manifest_sha256": v6["manifest_sha256"],
            "v6_source_root_sha256": v6["root_sha256"],
            "v4_source_root_sha256": KNOWN_V4_SOURCE_ROOT_SHA256,
            "smoke_receipt_file_sha256": smoke_file.sha256,
            "input_manifest_sha256": inputs["manifest_sha256"],
            "input_roles": inputs["roles"],
            "publication_members": [
                {"name": name, **pins["publication_members"][name]}
                for name in sorted(PUBLICATION_MEMBERS, key=lambda item: item.encode("ascii"))
            ],
            "producer_terminal_status": expected_status,
            "producer_RESULT_used_as_numerical_input": False,
            "independent_packet_parser_and_causal_CPU_decoder_used": True,
            "literal_COARSE_canonical_reencode_matches": True,
            "all_18_inverse_states_verified_as_little_endian_I32": True,
            "exact_Qwen_shape_only": True,
            "recomputed": recomputed,
            "atomic_completion_evidence": {
                "external_member_pins": True,
                "exact_terminal_member_set": True,
                "internal_member_root_and_completion_seal": True,
                "completion_not_observably_older_than_data": True,
                "historical_renameat2_syscall_order_provable_from_static_files": False,
            },
            "claim_boundary": {
                "one_externally_pinned_768x2048_three_role_coarse pilot": True,
                "Qwen_checkpoint_provenance_inferred_from_shape": False,
                "universal_arbitrary_shape_below_2_5_bpw": False,
                "fine_TACTIC_codec_present": False,
                "final_2_5_bpw_codec_result": False,
                "accelerator_HBM_measured": False,
                "strict_below_2x_inference_HBM_authority": False,
                "universal_SwiGLU_MoE_performance_authority": False,
            },
        }
        receipt["audit_receipt_sha256"] = core.sha256(core.canonical_json(receipt))
        return receipt
    finally:
        if publication is not None:
            close_publication(publication)
        if v4 is not None:
            v4["directory"].verify()
            v4["directory"].close()
            for held in v4["members"].values():
                held.verify()
                held.close()
        for held in reversed(held_misc):
            held.verify()
            held.close()
        if v6 is not None:
            close_source(v6)
        if own is not None:
            close_source(own)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--expected-auditor-source-manifest-sha256", required=True)
    result.add_argument("--external-pins", required=True)
    result.add_argument("--expected-external-pins-sha256", required=True)
    return result


def main() -> int:
    receipt = run(parser().parse_args())
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL_TACTIC_ACTUAL_COARSE_N18_V6_RESULT_AUDIT: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
