#!/usr/bin/env python3
"""Pinned, bounded Wasmtime host for one immutable routed-expert packet.

The packet is a host-owned ``bytes`` object.  It is never copied wholesale
into guest memory and no guest receives a pointer, descriptor, path, native
read primitive, or WASI capability for it.  The decoder's sole import is a
bounded host callback that copies a requested, non-overlapping packet slice
into decoder-owned memory and records that exact slice.  A distinct zero-
import WebAssembly module canonicalizes only the decoded semantic state.

Decoder ABI:

* import ``authority.read_packet(i32 offset, i32 destination, i32 length)
  -> i32``;
* export memory ``memory``;
* export ``scratch_base() -> i32``;
* export ``decode_route(i32 packet_len, i32 reconstruction_ptr,
  i32 reconstruction_bytes, i32 semantic_ptr, i32 semantic_capacity,
  i32 gate_values, i32 up_values, i32 down_values) -> i32``.  It returns the
  positive semantic-state length.

Independent canonical encoder ABI:

* no imports;
* export memory ``memory``;
* export ``scratch_base() -> i32``;
* export ``encode_canonical(i32 semantic_ptr, i32 semantic_len,
  i32 output_ptr, i32 output_capacity) -> i32``.  It returns the positive
  canonical packet length.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


PAGE_BYTES = 4096
ROLE_ORDER = ("gate", "up", "down")
STORE_MEMORY_LIMIT_BYTES = 1 << 30
STORE_TABLE_ELEMENTS = 10_000
STORE_INSTANCE_LIMIT = 2
STORE_TABLE_LIMIT = 2
STORE_MEMORY_COUNT_LIMIT = 2
FUEL_BASE = 100_000_000
FUEL_PER_PACKET_BYTE = 50_000
MAX_FUEL = 1_000_000_000_000
HEX = frozenset("0123456789abcdef")


class SandboxError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SandboxError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SandboxError("noncanonical JSON value") from exc


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SandboxError(f"{label}: nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SandboxError(f"{label}: strict JSON") from exc
    require(isinstance(value, dict), f"{label}: object")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise SandboxError(f"{label}: lstat") from exc
    require(not path.is_symlink() and stat.S_ISREG(before.st_mode),
            f"{label}: regular non-link")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise SandboxError(f"{label}: stable read") from exc
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"{label}: changed during read")
    return payload


def real_directory(path: Path, label: str) -> Path:
    try:
        before = path.lstat()
    except OSError as exc:
        raise SandboxError(f"{label}: lstat") from exc
    require(not path.is_symlink() and stat.S_ISDIR(before.st_mode),
            f"{label}: real directory")
    try:
        resolved = path.resolve(strict=True)
        after = path.lstat()
    except OSError as exc:
        raise SandboxError(f"{label}: stable resolve") from exc
    require((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
            f"{label}: changed during resolve")
    return resolved


def row_root(rows: list[dict[str, Any]]) -> str:
    return sha256(canonical_json(rows))


def align64(value: int) -> int:
    return (value + 63) & ~63


def runtime_limits() -> dict[str, int | bool]:
    return {"consume_fuel": True,
            "store_memory_limit_bytes": STORE_MEMORY_LIMIT_BYTES,
            "store_table_elements": STORE_TABLE_ELEMENTS,
            "store_instance_limit": STORE_INSTANCE_LIMIT,
            "store_table_limit": STORE_TABLE_LIMIT,
            "store_memory_count_limit": STORE_MEMORY_COUNT_LIMIT,
            "fuel_base": FUEL_BASE,
            "fuel_per_packet_byte": FUEL_PER_PACKET_BYTE,
            "maximum_fuel": MAX_FUEL}


def native_runtime_name(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return (name.endswith((".dll", ".dylib", ".pyd")) or ".so" in name)


def classify_runtime_path(logical: str, capability: dict[str, Any]) -> str:
    if logical in capability["python_module_files"]:
        return "python_module"
    if logical in capability["native_libraries"]:
        return "native_library"
    if logical == capability["metadata_relative_path"]:
        return "metadata"
    return "resource"


def authenticate_runtime_tree(root: Path, capability: dict[str, Any]) -> tuple[
        list[dict[str, Any]], dict[str, bytes]]:
    files: list[Path] = []
    directories: set[str] = set()
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        for name in directory_names:
            item = current / name
            require(not item.is_symlink() and stat.S_ISDIR(item.lstat().st_mode),
                    "runtime tree directory")
            directories.add(item.relative_to(root).as_posix())
        for name in file_names:
            item = current / name
            require(not item.is_symlink() and stat.S_ISREG(item.lstat().st_mode),
                    "runtime tree file")
            files.append(item)
    rows = []
    payloads = {}
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        logical = "runtime/" + path.relative_to(root).as_posix()
        payload = regular_bytes(path, f"runtime file {logical}")
        rows.append({"path": logical, "bytes": len(payload),
                     "sha256": sha256(payload),
                     "kind": classify_runtime_path(logical, capability)})
        payloads[logical] = payload
    expected_files = set(capability["python_module_files"]) | set(
        capability["native_libraries"]) | {capability["metadata_relative_path"]}
    require(expected_files <= set(payloads), "runtime inventory files present")
    require(set(capability["python_module_files"]) ==
            {logical for logical in payloads
             if PurePosixPath(logical).suffix.lower() in {".py", ".pyi"}} and
            set(capability["native_libraries"]) ==
            {logical for logical in payloads if native_runtime_name(logical)},
            "complete runtime module/native inventories")
    expected_directories = set()
    for logical in payloads:
        parts = PurePosixPath(logical).parts[1:]
        expected_directories.update(
            PurePosixPath(*parts[:end]).as_posix()
            for end in range(1, len(parts)))
    require(directories == expected_directories,
            "runtime tree exact directory closure")
    require(row_root(rows) == capability["runtime_tree_root_sha256"],
            "runtime tree aggregate pin")
    by_path = {row["path"]: row for row in rows}
    modules = [by_path[path] for path in capability["python_module_files"]]
    natives = [by_path[path] for path in capability["native_libraries"]]
    require(row_root(modules) == capability["module_tree_root_sha256"] and
            row_root(natives) == capability["native_library_root_sha256"],
            "runtime module/native aggregate pins")
    return rows, payloads


def validate_runtime_capability(payload: bytes) -> dict[str, Any]:
    capability = strict_json(payload, "runtime capability")
    required = {"schema", "distribution_name", "python_distribution_version",
                "wasmtime_runtime_version", "python_abi", "platform_tag",
                "target", "module_entry_relative_path", "metadata_relative_path",
                "runtime_tree_root_sha256", "module_tree_root_sha256",
                "native_library_root_sha256", "python_module_files",
                "native_libraries", "engine_limits"}
    require(canonical_json(capability) + b"\n" == payload and
            set(capability) == required and capability["schema"] ==
            "strata-rm-global-swap-v4-wasmtime-runtime-capability" and
            capability["distribution_name"] == "wasmtime" and
            capability["wasmtime_runtime_version"] ==
            capability["python_distribution_version"] and
            capability["engine_limits"] == runtime_limits() and
            isinstance(capability["python_module_files"], list) and
            capability["python_module_files"] ==
            sorted(set(capability["python_module_files"])) and
            isinstance(capability["native_libraries"], list) and
            capability["native_libraries"] ==
            sorted(set(capability["native_libraries"])) and
            capability["native_libraries"], "runtime capability schema")
    return capability


def logical_to_runtime_path(root: Path, logical: str) -> Path:
    parts = PurePosixPath(logical).parts
    require(parts and parts[0] == "runtime" and all(
        part not in ("", ".", "..") for part in parts), "runtime logical path")
    path = root.joinpath(*parts[1:])
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise SandboxError("runtime logical path containment") from exc
    return path


def loaded_native_paths() -> set[Path]:
    require(sys.platform.startswith("linux"),
            "native-library observation requires Linux /proc")
    payload = regular_bytes(Path("/proc/self/maps"), "process memory map")
    result = set()
    for line in payload.decode("utf-8", errors="strict").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            text = fields[5]
            require(not text.endswith(" (deleted)"), "deleted loaded native image")
            result.add(Path(text).resolve(strict=True))
    return result


def verify_loaded_native_runtime(root: Path, capability: dict[str, Any]) -> None:
    observed = loaded_native_paths()
    expected = {logical_to_runtime_path(root, logical).resolve(strict=True)
                for logical in capability["native_libraries"]}
    require(expected <= observed, "every pinned Wasmtime native library loaded")
    loaded_from_snapshot = {path for path in observed
                            if path == root or root in path.parents}
    require(loaded_from_snapshot == expected,
            "no unpinned native image loaded from runtime snapshot")
    for logical in capability["native_libraries"]:
        path = logical_to_runtime_path(root, logical)
        require(sha256(regular_bytes(path, f"loaded native {logical}")) ==
                next(row["sha256"] for row in _RUNTIME_ROWS
                     if row["path"] == logical), "loaded native hash")


def import_pinned_wasmtime(root: Path, capability: dict[str, Any]):
    require("wasmtime" not in sys.modules and
            not any(name.startswith("wasmtime.") for name in sys.modules),
            "Wasmtime not pre-imported")
    sys.path.insert(0, str(root))
    metadata_path = logical_to_runtime_path(
        root, capability["metadata_relative_path"])
    distribution = importlib.metadata.Distribution.at(metadata_path.parent)
    require(distribution.metadata["Name"].lower() == "wasmtime" and
            distribution.version == capability["python_distribution_version"],
            "pinned Wasmtime distribution metadata")
    wasmtime = importlib.import_module("wasmtime")
    entry = logical_to_runtime_path(
        root, capability["module_entry_relative_path"]).resolve(strict=True)
    require(Path(wasmtime.__file__).resolve(strict=True) == entry,
            "Wasmtime entry module from pinned snapshot")
    allowed = set(capability["python_module_files"]) | set(
        capability["native_libraries"])
    for name, module in list(sys.modules.items()):
        if name == "wasmtime" or name.startswith("wasmtime."):
            module_file = getattr(module, "__file__", None)
            if module_file is not None:
                path = Path(module_file).resolve(strict=True)
                try:
                    relative = path.relative_to(root).as_posix()
                except ValueError as exc:
                    raise SandboxError(
                        f"Wasmtime module outside snapshot: {name}") from exc
                require("runtime/" + relative in allowed,
                        f"Wasmtime module is pinned: {name}")
    return wasmtime


def module_imports(wasmtime, module) -> list[dict[str, str]]:
    result = []
    for item in module.imports:
        if isinstance(item.type, wasmtime.FuncType):
            kind = "func"
        elif isinstance(item.type, wasmtime.MemoryType):
            kind = "memory"
        elif isinstance(item.type, wasmtime.TableType):
            kind = "table"
        elif isinstance(item.type, wasmtime.GlobalType):
            kind = "global"
        else:
            kind = "unknown"
        result.append({"module": item.module, "name": item.name, "kind": kind})
    return result


def configure_store(wasmtime, engine, fuel: int):
    store = wasmtime.Store(engine)
    # These limits and fuel are deliberately installed before instantiation.
    store.set_limits(memory_size=STORE_MEMORY_LIMIT_BYTES,
                     table_elements=STORE_TABLE_ELEMENTS,
                     instances=STORE_INSTANCE_LIMIT, tables=STORE_TABLE_LIMIT,
                     memories=STORE_MEMORY_COUNT_LIMIT)
    store.set_fuel(fuel)
    return store


def ensure_memory(wasmtime, memory, store, required: int) -> None:
    require(isinstance(memory, wasmtime.Memory) and
            0 <= required <= STORE_MEMORY_LIMIT_BYTES, "bounded guest memory need")
    current = int(memory.data_len(store))
    if current < required:
        memory.grow(store, math.ceil((required - current) / 65536))
    require(required <= int(memory.data_len(store)) <= STORE_MEMORY_LIMIT_BYTES,
            "Store memory limiter enforced")


def parse_request(payload: bytes, packet: bytes, semantic_schema_sha: str
                  ) -> tuple[dict[str, Any], list[int]]:
    request = strict_json(payload, "route request")
    required = {"schema", "route_id", "packet_sha256", "packet_bytes",
                "page_bytes", "semantic_schema_sha256", "sources"}
    require(canonical_json(request) + b"\n" == payload and
            set(request) == required and request["schema"] ==
            "strata-rm-global-swap-v4-semantic-route-request" and
            isinstance(request["route_id"], str) and request["route_id"] and
            request["packet_sha256"] == sha256(packet) and
            request["packet_bytes"] == len(packet) and
            request["page_bytes"] == PAGE_BYTES and
            request["semantic_schema_sha256"] == semantic_schema_sha,
            "route request binding")
    sources = request["sources"]
    require(isinstance(sources, list) and len(sources) == 3 and
            [row.get("role") for row in sources] == list(ROLE_ORDER),
            "one ordered SwiGLU route")
    counts = []
    for ordinal, source in enumerate(sources):
        require(isinstance(source, dict) and set(source) ==
                {"ordinal", "role", "layer", "expert", "shape"} and
                source["ordinal"] == ordinal and
                isinstance(source["layer"], int) and source["layer"] >= 0 and
                isinstance(source["expert"], int) and source["expert"] >= 0 and
                isinstance(source["shape"], list) and len(source["shape"]) == 2 and
                all(isinstance(value, int) and value > 0
                    for value in source["shape"]), "route source geometry")
        counts.append(source["shape"][0] * source["shape"][1])
    require(len({(row["layer"], row["expert"]) for row in sources}) == 1,
            "single routed expert")
    return request, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-capability", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--canonical-encoder", type=Path, required=True)
    parser.add_argument("--semantic-schema", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1,
            "sandbox host requires python -I -B")
    require("PYTHONPATH" not in os.environ, "PYTHONPATH inherited")

    runtime_root = real_directory(args.runtime_root, "runtime root")
    output = real_directory(args.output_dir, "output directory")
    require(not any(output.iterdir()), "fresh empty output directory")
    capability_payload = regular_bytes(args.runtime_capability,
                                       "runtime capability snapshot")
    capability = validate_runtime_capability(capability_payload)
    global _RUNTIME_ROWS
    _RUNTIME_ROWS, _ = authenticate_runtime_tree(runtime_root, capability)
    pre_runtime_root = row_root(_RUNTIME_ROWS)

    decoder_payload = regular_bytes(args.decoder, "semantic decoder snapshot")
    encoder_payload = regular_bytes(args.canonical_encoder,
                                    "canonical encoder snapshot")
    schema_payload = regular_bytes(args.semantic_schema,
                                   "semantic schema snapshot")
    schema = strict_json(schema_payload, "semantic schema")
    require(canonical_json(schema) + b"\n" == schema_payload and
            schema.get("schema") == "strata-rm-v4-decoded-semantic-state" and
            schema.get("raw_packet_bytes_permitted") is False and
            schema.get("complete_quantizer_decisions") is True and
            schema.get("maximum_state_bytes_formula") ==
            "min(8*packet_bytes+1048576,536870912)", "semantic schema")
    semantic_schema_sha = sha256(schema_payload)

    # This immutable Python bytes object remains host-owned for the lifetime of
    # both guests.  The file and path are never exposed to either guest.
    packet_payload = regular_bytes(args.packet, "immutable packet snapshot")
    packet_initial_sha = sha256(packet_payload)
    request_payload = regular_bytes(args.request, "route request snapshot")
    request, counts = parse_request(request_payload, packet_payload,
                                    semantic_schema_sha)
    fuel_budget = min(MAX_FUEL, FUEL_BASE + FUEL_PER_PACKET_BYTE *
                      len(packet_payload))

    wasmtime = import_pinned_wasmtime(runtime_root, capability)
    config = wasmtime.Config()
    config.consume_fuel = True
    engine = wasmtime.Engine(config)
    decoder_module = wasmtime.Module(engine, decoder_payload)
    encoder_module = wasmtime.Module(engine, encoder_payload)
    decoder_import_list = module_imports(wasmtime, decoder_module)
    encoder_import_list = module_imports(wasmtime, encoder_module)
    require(decoder_import_list ==
            [{"module": "authority", "name": "read_packet", "kind": "func"}],
            "decoder receives exactly one bounded packet callback")
    require(encoder_import_list == [],
            "canonical encoder receives no imports or packet capability")

    packet_reads: list[dict[str, int]] = []
    read_intervals: list[tuple[int, int]] = []

    def read_packet(caller, offset, destination, length):
        require(all(type(value) is int for value in
                    (offset, destination, length)) and offset >= 0 and
                destination >= 0 and length > 0 and
                offset + length <= len(packet_payload),
                "bounded packet callback request")
        end = offset + length
        require(all(end <= left or offset >= right
                    for left, right in read_intervals),
                "packet bytes may be supplied exactly once")
        memory = caller.get("memory")
        require(isinstance(memory, wasmtime.Memory),
                "decoder exports callback-visible memory")
        require(destination + length <= int(memory.data_len(caller)),
                "packet callback destination in guest memory")
        memory.write(caller, packet_payload[offset:end], destination)
        read_intervals.append((offset, end))
        packet_reads.append({"offset": offset, "length": length})
        return 0

    decoder_store = configure_store(wasmtime, engine, fuel_budget)
    linker = wasmtime.Linker(engine)
    read_type = wasmtime.FuncType(
        [wasmtime.ValType.i32(), wasmtime.ValType.i32(), wasmtime.ValType.i32()],
        [wasmtime.ValType.i32()])
    read_function = wasmtime.Func(decoder_store, read_type, read_packet,
                                  access_caller=True)
    linker.define(decoder_store, "authority", "read_packet", read_function)
    decoder_instance = linker.instantiate(decoder_store, decoder_module)
    decoder_exports = decoder_instance.exports(decoder_store)
    try:
        decoder_memory = decoder_exports["memory"]
        decoder_scratch = decoder_exports["scratch_base"]
        decode_route = decoder_exports["decode_route"]
    except KeyError as exc:
        raise SandboxError("semantic decoder ABI exports") from exc
    reconstruction_bytes = 8 * sum(counts)
    semantic_capacity = min(8 * len(packet_payload) + 1_048_576, 536_870_912)
    scratch = align64(int(decoder_scratch(decoder_store)))
    require(scratch >= 0, "decoder scratch base")
    reconstruction_offset = scratch
    semantic_offset = align64(reconstruction_offset + reconstruction_bytes)
    decoder_required = semantic_offset + semantic_capacity
    ensure_memory(wasmtime, decoder_memory, decoder_store, decoder_required)
    semantic_bytes = int(decode_route(
        decoder_store, len(packet_payload), reconstruction_offset,
        reconstruction_bytes, semantic_offset, semantic_capacity,
        counts[0], counts[1], counts[2]))
    require(0 < semantic_bytes <= semantic_capacity,
            "semantic decoder success and bounded state")
    require(sum(row["length"] for row in packet_reads) == len(packet_payload) and
            sorted(read_intervals) and sorted(read_intervals)[0][0] == 0 and
            sorted(read_intervals)[-1][1] == len(packet_payload) and
            all(left[1] == right[0] for left, right in
                zip(sorted(read_intervals), sorted(read_intervals)[1:])),
            "decoder consumed every literal packet byte exactly once")
    reconstruction = bytes(decoder_memory.read(
        decoder_store, reconstruction_offset,
        reconstruction_offset + reconstruction_bytes))
    semantic_state = bytes(decoder_memory.read(
        decoder_store, semantic_offset, semantic_offset + semantic_bytes))
    decoder_fuel_remaining = int(decoder_store.get_fuel())
    require(0 <= decoder_fuel_remaining < fuel_budget and
            int(decoder_memory.data_len(decoder_store)) <=
            STORE_MEMORY_LIMIT_BYTES, "decoder fuel/memory bounds")

    # A separate zero-import instance receives semantic state only.  Neither
    # packet_payload nor read_packet is supplied to this linker/store.
    encoder_store = configure_store(wasmtime, engine, fuel_budget)
    encoder_instance = wasmtime.Instance(encoder_store, encoder_module, [])
    encoder_exports = encoder_instance.exports(encoder_store)
    try:
        encoder_memory = encoder_exports["memory"]
        encoder_scratch = encoder_exports["scratch_base"]
        encode_canonical = encoder_exports["encode_canonical"]
    except KeyError as exc:
        raise SandboxError("canonical encoder ABI exports") from exc
    encoder_base = align64(int(encoder_scratch(encoder_store)))
    require(encoder_base >= 0, "encoder scratch base")
    encoder_semantic_offset = encoder_base
    canonical_offset = align64(encoder_semantic_offset + semantic_bytes)
    encoder_required = canonical_offset + len(packet_payload)
    ensure_memory(wasmtime, encoder_memory, encoder_store, encoder_required)
    encoder_memory.write(encoder_store, semantic_state, encoder_semantic_offset)
    canonical_bytes = int(encode_canonical(
        encoder_store, encoder_semantic_offset, semantic_bytes,
        canonical_offset, len(packet_payload)))
    require(canonical_bytes == len(packet_payload),
            "independent semantic canonical encoder length")
    canonical = bytes(encoder_memory.read(
        encoder_store, canonical_offset, canonical_offset + canonical_bytes))
    encoder_fuel_remaining = int(encoder_store.get_fuel())
    require(0 <= encoder_fuel_remaining < fuel_budget and
            int(encoder_memory.data_len(encoder_store)) <=
            STORE_MEMORY_LIMIT_BYTES, "encoder fuel/memory bounds")
    require(canonical == packet_payload,
            "decoded semantics have one independently regenerated canonical packet")

    verify_loaded_native_runtime(runtime_root, capability)
    post_rows, _ = authenticate_runtime_tree(runtime_root, capability)
    require(row_root(post_rows) == pre_runtime_root and
            sha256(packet_payload) == packet_initial_sha,
            "runtime and immutable host packet remain unchanged")

    cursor = 0
    for ordinal, count in enumerate(counts):
        length = count * 8
        (output / f"reconstruction-{ordinal:04d}.f64").write_bytes(
            reconstruction[cursor:cursor + length])
        cursor += length
    require(cursor == len(reconstruction), "reconstruction split")
    (output / "canonical_packet.bin").write_bytes(canonical)
    pages = sorted({index for row in packet_reads for index in range(
        row["offset"] // PAGE_BYTES,
        (row["offset"] + row["length"] - 1) // PAGE_BYTES + 1)})
    page_count = (len(packet_payload) + PAGE_BYTES - 1) // PAGE_BYTES
    require(pages == list(range(page_count)), "complete physical packet pages")
    sandbox_sha = sha256(regular_bytes(Path(__file__), "sandbox source"))
    receipt = {
        "schema": "strata-rm-global-swap-v4-pinned-wasmtime-sandbox-receipt",
        "route_id": request["route_id"], "sandbox_sha256": sandbox_sha,
        "runtime_capability_sha256": sha256(capability_payload),
        "runtime_tree_root_sha256": capability["runtime_tree_root_sha256"],
        "python_distribution_version": capability["python_distribution_version"],
        "wasmtime_runtime_version": capability["wasmtime_runtime_version"],
        "module_tree_root_sha256": capability["module_tree_root_sha256"],
        "native_library_root_sha256": capability["native_library_root_sha256"],
        "native_libraries_loaded": capability["native_libraries"],
        "decoder_imports": decoder_import_list,
        "canonical_encoder_imports": encoder_import_list,
        "wasi_enabled": False,
        "store_limits_installed_before_instantiation": True,
        "store_memory_limit_bytes": STORE_MEMORY_LIMIT_BYTES,
        "fuel_budget": fuel_budget,
        "decoder_fuel_remaining": decoder_fuel_remaining,
        "encoder_fuel_remaining": encoder_fuel_remaining,
        "packet_host_buffer_immutable": True,
        "packet_capability": "read-only bounded host callback",
        "packet_sha256": packet_initial_sha, "packet_bytes": len(packet_payload),
        "packet_read_operations": packet_reads,
        "literal_bytes_supplied_total": sum(row["length"] for row in packet_reads),
        "unique_literal_bytes_supplied": len(packet_payload),
        "pages_supplied": pages,
        "physical_page_bytes_supplied": page_count * PAGE_BYTES,
        "semantic_state_bytes": semantic_bytes,
        "semantic_schema_sha256": semantic_schema_sha,
        "canonical_encoder_received_packet_capability": False,
        "canonical_packet_bytes": len(canonical),
        "canonical_packet_sha256": sha256(canonical), "decode_status": 0,
        "status": "PASS_PINNED_BOUNDED_IMMUTABLE_SEMANTIC_WASM_DECODE"}
    (output / "SANDBOX_RECEIPT.json").write_bytes(canonical_json(receipt) + b"\n")


_RUNTIME_ROWS: list[dict[str, Any]] = []


if __name__ == "__main__":
    main()
