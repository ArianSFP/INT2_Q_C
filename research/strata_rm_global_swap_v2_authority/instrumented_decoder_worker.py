#!/usr/bin/env python3
"""Execute a snapshotted decoder while measuring its literal packet reads.

The launcher is deliberately small enough for an independent source audit.
It compiles already-read decoder bytes, supplies no source paths, rejects
explicit reads outside the snapshotted request and packet, denies ``os.open``
and process/network escape, and records packet byte intervals itself.  A
decoder-emitted read trace is neither requested nor trusted.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


class InstrumentationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InstrumentationError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


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


def _resolve(path: Any) -> Path:
    if isinstance(path, int):
        raise InstrumentationError("integer file descriptors forbidden")
    return Path(os.fspath(path)).resolve(strict=False)


class PacketReader:
    def __init__(self, handle, operations: list[dict[str, int]]):
        self._handle = handle
        self._operations = operations

    def _record(self, before: int, after: int) -> None:
        if after > before:
            self._operations.append({"offset": before, "length": after - before})

    def read(self, size: int = -1):
        before = int(self._handle.tell())
        value = self._handle.read(size)
        after = int(self._handle.tell())
        self._record(before, after)
        return value

    def read1(self, size: int = -1):
        before = int(self._handle.tell())
        value = self._handle.read1(size)
        after = int(self._handle.tell())
        self._record(before, after)
        return value

    def readinto(self, buffer):
        before = int(self._handle.tell())
        value = self._handle.readinto(buffer)
        after = int(self._handle.tell())
        self._record(before, after)
        return value

    def readline(self, size: int = -1):
        before = int(self._handle.tell())
        value = self._handle.readline(size)
        after = int(self._handle.tell())
        self._record(before, after)
        return value

    def readlines(self, hint: int = -1):
        values = []
        total = 0
        while hint < 0 or total < hint:
            row = self.readline()
            if not row:
                break
            values.append(row)
            total += len(row)
        return values

    def __iter__(self):
        return self

    def __next__(self):
        row = self.readline()
        if not row:
            raise StopIteration
        return row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._handle.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str):
        return getattr(self._handle, name)


def _covered_bytes(operations: list[dict[str, int]]) -> int:
    intervals = sorted((row["offset"], row["offset"] + row["length"])
                       for row in operations)
    if not intervals:
        return 0
    covered = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            covered += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return covered + end - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instrumentation-output", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1,
            "launcher requires python -I -B")
    decoder = args.decoder.resolve(strict=True)
    request = args.request.resolve(strict=True)
    packet = args.packet.resolve(strict=True)
    output_dir = args.output_dir.resolve(strict=True)
    instrumentation_output = args.instrumentation_output.resolve(strict=False)
    require(output_dir.is_dir() and not output_dir.is_symlink(),
            "real output directory")
    decoder_payload = regular_bytes(decoder, "decoder snapshot")
    packet_payload = regular_bytes(packet, "literal packet snapshot")
    regular_bytes(request, "request snapshot")
    code = compile(decoder_payload, str(decoder), "exec", dont_inherit=True)

    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    operations: list[dict[str, int]] = []
    packet_open_count = 0
    denied_read_paths = 0

    def intercepted_open(file, mode="r", *open_args, **open_kwargs):
        nonlocal packet_open_count, denied_read_paths
        path = _resolve(file)
        reading = "r" in mode or "+" in mode
        writing = any(flag in mode for flag in ("w", "a", "x", "+"))
        if path == packet:
            require(reading and not writing and "b" in mode,
                    "packet requires binary read-only mode")
            packet_open_count += 1
            handle = original_io_open(path, mode, *open_args, **open_kwargs)
            return PacketReader(handle, operations)
        if path == request:
            require(reading and not writing, "request read-only")
            return original_io_open(path, mode, *open_args, **open_kwargs)
        if writing and (path.parent == output_dir or output_dir in path.parents):
            require(path != output_dir, "output member path")
            return original_io_open(path, mode, *open_args, **open_kwargs)
        denied_read_paths += 1
        raise InstrumentationError(f"decoder explicit filesystem access denied: {path}")

    def denied_os_open(*_args, **_kwargs):
        raise InstrumentationError("decoder os.open access denied")

    def denied_escape(*_args, **_kwargs):
        raise InstrumentationError("decoder process/network escape denied")

    builtins.open = intercepted_open
    io.open = intercepted_open
    os.open = denied_os_open
    escape_originals = {
        "popen": subprocess.Popen,
        "run": subprocess.run,
        "call": subprocess.call,
        "check_call": subprocess.check_call,
        "check_output": subprocess.check_output,
        "system": os.system,
        "popen_os": os.popen,
        "socket": socket.socket,
    }
    subprocess.Popen = denied_escape
    subprocess.run = denied_escape
    subprocess.call = denied_escape
    subprocess.check_call = denied_escape
    subprocess.check_output = denied_escape
    os.system = denied_escape
    os.popen = denied_escape
    socket.socket = denied_escape
    previous_argv = list(sys.argv)
    previous_path = list(sys.path)
    try:
        sys.argv = [str(decoder), "--request", str(request),
                    "--packet", str(packet), "--output-dir", str(output_dir)]
        sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() !=
                    decoder.parent]
        globals_dict = {"__name__": "__main__", "__file__": str(decoder),
                        "__package__": None, "__cached__": None,
                        "__builtins__": builtins.__dict__}
        exec(code, globals_dict, globals_dict)
    finally:
        sys.argv = previous_argv
        sys.path = previous_path
        builtins.open = original_builtin_open
        io.open = original_io_open
        os.open = original_os_open
        subprocess.Popen = escape_originals["popen"]
        subprocess.run = escape_originals["run"]
        subprocess.call = escape_originals["call"]
        subprocess.check_call = escape_originals["check_call"]
        subprocess.check_output = escape_originals["check_output"]
        os.system = escape_originals["system"]
        os.popen = escape_originals["popen_os"]
        socket.socket = escape_originals["socket"]

    require(packet_open_count >= 1 and operations,
            "decoder must read literal packet")
    total = sum(row["length"] for row in operations)
    covered = _covered_bytes(operations)
    require(covered == len(packet_payload),
            "decoder reads cover every literal packet byte")
    receipt = {
        "schema": "strata-rm-v2-instrumented-decoder-io-receipt",
        "case_id": args.case_id,
        "packet_sha256": hashlib.sha256(packet_payload).hexdigest(),
        "packet_bytes": len(packet_payload),
        "packet_open_count": packet_open_count,
        "packet_read_operations": len(operations),
        "denied_read_paths": denied_read_paths,
        "denied_os_open": True,
        "denied_process_escape": True,
        "source_paths_supplied": 0,
        "total_packet_bytes_read": total,
        "unique_packet_bytes_read": covered,
        "operations": operations,
        "status": "PASS_INSTRUMENTED_LITERAL_PACKET_IO",
    }
    instrumentation_output.write_bytes(canonical_json(receipt) + b"\n")


if __name__ == "__main__":
    main()
