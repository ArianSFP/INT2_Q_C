#!/usr/bin/env python3
"""Instrumented regular-file read trace, kept distinct from layout arithmetic."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any


class ReadTraceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadTraceError(message)


def read_once(path: Path, expected_bytes: int) -> tuple[bytes, dict[str, Any]]:
    absolute = path.resolve(strict=True)
    require(absolute == path.absolute(), "trace path must be canonical and contain no symlink")
    descriptor = os.open(
        os.fspath(absolute),
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    calls = []
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "traced regular single-link file")
        require(before.st_size == expected_bytes and expected_bytes > 0, "traced file size")
        payload = os.read(descriptor, expected_bytes)
        calls.append({"offset": 0, "requested_bytes": expected_bytes, "returned_bytes": len(payload)})
        require(len(payload) == expected_bytes, "short traced read")
        require(os.read(descriptor, 1) == b"", "traced file grew")
        calls.append({"offset": expected_bytes, "requested_bytes": 1, "returned_bytes": 0})
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
            "traced file identity drift",
        )
    finally:
        os.close(descriptor)
    data_calls = [row for row in calls if row["returned_bytes"]]
    returned = sum(row["returned_bytes"] for row in data_calls)
    return payload, {
        "schema": "tactic-ramanujan384-instrumented-file-read-trace-v1",
        "data_read_calls": len(data_calls),
        "eof_probe_calls": len(calls) - len(data_calls),
        "returned_data_bytes": returned,
        "object_bytes": expected_bytes,
        "instrumented_file_read_amplification": returned / expected_bytes,
        "events": calls,
        "layout_read_amplification_upper_bound": 1.0,
        "layout_bound_is_not_a_measurement": True,
        "instrumented_trace_is_not_physical_storage_or_hbm_telemetry": True,
        "accelerator_hbm_measured": False,
    }
