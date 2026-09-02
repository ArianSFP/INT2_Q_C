#!/usr/bin/env python3
"""Zero-import WebAssembly sandbox for one routed expert packet.

This trusted host reads the already authenticated snapshots.  The untrusted
decoder is a WebAssembly module instantiated with an empty import list and no
WASI linker.  It therefore receives no path, file descriptor, callback,
socket, environment, clock, random source, subprocess, ctypes, or native I/O
capability.  Its only input is a page-padded copy of one immutable expert
packet placed in its private linear memory before either exported function is
called.  Mutation of that input region is detected and rejected.

Required module ABI:

* export memory ``memory``;
* export function ``decode_route(i32 packet_ptr, i32 packet_len,
  i32 reconstruction_ptr, i32 reconstruction_bytes, i32 gate_values,
  i32 up_values, i32 down_values) -> i32``; zero means success;
* export function ``canonical_reencode(i32 packet_ptr, i32 packet_len,
  i32 output_ptr, i32 output_capacity) -> i32``; return exact output length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any


PAGE_BYTES = 4096
ROLE_ORDER = ("gate", "up", "down")
MAX_LINEAR_MEMORY_BYTES = 1 << 30


class SandboxError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SandboxError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           SandboxError(f"{label}: nonfinite {token}")))
    require(isinstance(value, dict), f"{label}: JSON object")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            f"{label}: regular non-link")
    payload = path.read_bytes()
    after = path.lstat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"{label}: stable read")
    return payload


def align64(value: int) -> int:
    return (value + 63) & ~63


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1,
            "sandbox host requires python -I -B")
    require("PYTHONPATH" not in os.environ, "PYTHONPATH inherited")
    module_payload = regular_bytes(args.module.resolve(strict=True),
                                   "decoder module snapshot")
    packet_payload = regular_bytes(args.packet.resolve(strict=True),
                                   "expert packet snapshot")
    request_payload = regular_bytes(args.request.resolve(strict=True),
                                    "route request snapshot")
    request = strict_json(request_payload, "route request")
    require(canonical_json(request) + b"\n" == request_payload and
            set(request) == {"schema", "route_id", "packet_sha256",
                             "packet_bytes", "page_bytes", "sources"} and
            request["schema"] == "strata-rm-global-swap-v3-wasm-route-request" and
            request["packet_sha256"] == hashlib.sha256(packet_payload).hexdigest() and
            request["packet_bytes"] == len(packet_payload) and
            request["page_bytes"] == PAGE_BYTES,
            "route request binding")
    sources = request["sources"]
    require(isinstance(sources, list) and len(sources) == 3 and
            [row.get("role") for row in sources] == list(ROLE_ORDER),
            "one ordered SwiGLU triplet")
    counts = []
    for ordinal, source in enumerate(sources):
        require(isinstance(source, dict) and set(source) ==
                {"ordinal", "role", "layer", "expert", "shape"} and
                source["ordinal"] == ordinal and
                isinstance(source["shape"], list) and len(source["shape"]) == 2 and
                all(isinstance(value, int) and value > 0
                    for value in source["shape"]), "route source geometry")
        counts.append(source["shape"][0] * source["shape"][1])
    require(len({(row["layer"], row["expert"]) for row in sources}) == 1,
            "single routed expert")

    # Imported only after all authority-owned path reads.  It is the host
    # runtime, not an import visible to the WebAssembly decoder.
    import wasmtime

    engine = wasmtime.Engine()
    module = wasmtime.Module(engine, module_payload)
    module_imports = list(module.imports)
    require(module_imports == [],
            "decoder module must have zero imports (including no WASI)")
    store = wasmtime.Store(engine)
    instance = wasmtime.Instance(store, module, [])
    exports = instance.exports(store)
    try:
        memory = exports["memory"]
        decode_route = exports["decode_route"]
        canonical_reencode = exports["canonical_reencode"]
    except KeyError as exc:
        raise SandboxError("decoder ABI exports") from exc

    page_count = (len(packet_payload) + PAGE_BYTES - 1) // PAGE_BYTES
    supplied_packet = packet_payload + bytes(page_count * PAGE_BYTES -
                                             len(packet_payload))
    reconstruction_bytes = 8 * sum(counts)
    packet_offset = 0
    reconstruction_offset = align64(len(supplied_packet))
    canonical_offset = align64(reconstruction_offset + reconstruction_bytes)
    required_memory = canonical_offset + len(packet_payload)
    require(required_memory <= MAX_LINEAR_MEMORY_BYTES,
            "bounded decoder linear memory")
    current_memory = int(memory.data_len(store))
    if current_memory < required_memory:
        memory.grow(store, math.ceil((required_memory - current_memory) / 65536))
    require(int(memory.data_len(store)) >= required_memory and
            int(memory.data_len(store)) <= MAX_LINEAR_MEMORY_BYTES,
            "decoder memory bounds")
    memory.write(store, supplied_packet, packet_offset)
    pre_decode_hash = hashlib.sha256(bytes(memory.read(
        store, packet_offset, packet_offset + len(supplied_packet)))).hexdigest()
    require(pre_decode_hash == hashlib.sha256(supplied_packet).hexdigest(),
            "packet buffer initial parity")
    decode_status = int(decode_route(
        store, packet_offset, len(packet_payload), reconstruction_offset,
        reconstruction_bytes, counts[0], counts[1], counts[2]))
    require(decode_status == 0, "decoder status")
    canonical_bytes = int(canonical_reencode(
        store, packet_offset, len(packet_payload), canonical_offset,
        len(packet_payload)))
    require(canonical_bytes == len(packet_payload), "canonical replay length")
    require(int(memory.data_len(store)) <= MAX_LINEAR_MEMORY_BYTES,
            "decoder may not grow memory beyond bound")
    post_decode_packet = bytes(memory.read(
        store, packet_offset, packet_offset + len(supplied_packet)))
    require(post_decode_packet == supplied_packet,
            "decoder may not mutate pre-opened packet buffer")
    reconstruction = bytes(memory.read(
        store, reconstruction_offset, reconstruction_offset + reconstruction_bytes))
    canonical = bytes(memory.read(
        store, canonical_offset, canonical_offset + canonical_bytes))
    require(canonical == packet_payload, "canonical packet replay")

    output = args.output_dir.resolve(strict=True)
    require(output.is_dir() and not output.is_symlink(), "real output directory")
    cursor = 0
    for ordinal, count in enumerate(counts):
        length = count * 8
        (output / f"reconstruction-{ordinal:04d}.f64").write_bytes(
            reconstruction[cursor:cursor + length])
        cursor += length
    require(cursor == len(reconstruction), "reconstruction split")
    (output / "canonical_packet.bin").write_bytes(canonical)
    pages = [{"page_index": index, "literal_offset": index * PAGE_BYTES,
              "literal_bytes": min(PAGE_BYTES,
                                   len(packet_payload) - index * PAGE_BYTES),
              "supplied_bytes": PAGE_BYTES}
             for index in range(page_count)]
    sandbox_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = {
        "schema": "strata-rm-global-swap-v3-zero-import-wasm-sandbox-receipt",
        "route_id": request["route_id"],
        "decoder_module_sha256": hashlib.sha256(module_payload).hexdigest(),
        "sandbox_sha256": sandbox_sha,
        "module_imports": [], "wasi_enabled": False,
        "filesystem_api_exposed": False, "descriptor_api_exposed": False,
        "native_io_imports_exposed": False,
        "packet_buffer_preopened": True, "packet_input_unchanged": True,
        "packet_sha256": hashlib.sha256(packet_payload).hexdigest(),
        "literal_packet_bytes_supplied": len(packet_payload),
        "page_bytes": PAGE_BYTES, "pages_supplied": pages,
        "zero_padding_bytes_supplied": len(supplied_packet) - len(packet_payload),
        "physical_page_bytes_supplied": len(supplied_packet),
        "decode_status": decode_status,
        "canonical_reencode_bytes": canonical_bytes,
        "status": "PASS_ZERO_IMPORT_WASM_EXPERT_PACKET_BUFFER_DECODE",
    }
    (output / "SANDBOX_RECEIPT.json").write_bytes(canonical_json(receipt) + b"\n")


if __name__ == "__main__":
    main()
