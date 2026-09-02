#!/usr/bin/env python3
"""Source-only entry point for the corrected epsilon-TCQ STRATA v1 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


AUTHORIZATION = "REPORT_EPSILON_TCQ_STRATA_BOUND_V1_SOURCE_HOLD"
SCHEMA = "epsilon-tcq-strata-bound-gate-v1-source-manifest"
STATUS = "FROZEN_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY"


class RunError(RuntimeError):
    pass


def need(condition, message):
    if not condition:
        raise RunError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def read_one(path, expected=None):
    need(path.is_absolute(), "absolute source member")
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
             0 < before.st_size <= 4 * (1 << 20), "source member file")
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            need(chunk, "source member short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
              before.st_nlink) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
              after.st_nlink), "source member identity drift")
        payload = b"".join(chunks)
        if expected is not None:
            need(digest(payload) == expected, "source member SHA-256")
        return payload
    finally:
        os.close(descriptor)


def strict_json(payload):
    def pairs(rows):
        output = {}
        for key, value in rows:
            need(key not in output, "duplicate JSON key")
            output[key] = value
        return output
    return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda item: (_ for _ in ()).throw(
                          RunError(f"nonfinite JSON {item}")))


def authenticate(root, manifest_sha256):
    root = root.resolve(strict=True)
    manifest_raw = read_one(root / "SOURCE_MANIFEST.json", manifest_sha256)
    manifest = strict_json(manifest_raw)
    need(manifest.get("schema") == SCHEMA and manifest.get("status") == STATUS,
         "source manifest identity")
    rows = manifest.get("members")
    need(isinstance(rows, list) and rows, "source manifest rows")
    observed, names = [], []
    for row in rows:
        need(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
             "source member row")
        name = row["name"]
        need(type(name) is str and name not in names and "/" not in name and
             "\\" not in name, "source member name")
        payload = read_one(root / name, row["sha256"])
        need(len(payload) == row["bytes"], "source member bytes")
        names.append(name)
        observed.append({"name": name, "bytes": len(payload),
                         "sha256": digest(payload)})
    need(names == sorted(names, key=lambda value: value.encode("utf-8")) and
         manifest.get("source_root_sha256") == digest(canonical(observed)),
         "source root/order")
    need({entry.name for entry in os.scandir(root)} ==
         set(names) | {"SOURCE_MANIFEST.json"}, "exact source package closure")
    return digest(manifest_raw), manifest["source_root_sha256"]


def main(arguments):
    need(arguments.authorization == AUTHORIZATION and sys.flags.isolated == 1 and
         sys.dont_write_bytecode, "authorization and CPython -I -B")
    manifest_sha, root_sha = authenticate(
        Path(__file__).resolve().parent, arguments.package_manifest_sha256)
    return {
        "schema": "epsilon-tcq-strata-bound-v1-typed-hold",
        "status": "HOLD_BLOCK_LEVEL_POLAR_LIST_ENGINE_NOT_SCALABLE",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": root_sha,
        "source_package_authenticated": True,
        "qwen_payload_may_open": False,
        "current_codec_payload_may_open": False,
        "matched_controls_may_open": False,
        "legacy_six_events_per_coordinate_abi_valid": False,
        "exact_six_sc_level_read_only_replay_present": True,
        "legal_search_unit": "whole_six_level_polar_block",
        "replacement_codec_fallback_allowed": False,
        "production_block_lengths": [1 << 20, 1 << 21],
        "straightforward_beam32_2pow21_peak_lower_bound_bytes": 7_147_102_208,
        "frozen_memory_cap_bytes": 4 * (1 << 30),
        "required_next_artifact": (
            "a separately frozen device-resident resumable polar list-state "
            "engine whose literal prefix storage stays below the memory cap"
        ),
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--package-manifest-sha256", required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(main(parser().parse_args()), sort_keys=True,
                     separators=(",", ":")))
