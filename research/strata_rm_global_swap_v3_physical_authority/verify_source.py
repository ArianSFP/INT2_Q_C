#!/usr/bin/env python3
"""Standalone exact-closure verifier for v3 physical authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


SCHEMA = "strata-rm-global-swap-v3-physical-authority-source-manifest"
V2_ROOT = "e9ce4c24017831fab50696c2c5d81739d1f24d8121075c3aa56612b9a77013c9"
REVIEW_ROOT = "d642889efcf8c54173eb7659602181cb9e71e122ce11ff05da6b24e45c47a113"
HEX = frozenset("0123456789abcdef")


def fail(message: str) -> None:
    raise SystemExit("v3 source verification failed: " + message)


def strict_json(payload: bytes):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail("duplicate key")
            result[key] = value
        return result
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
                       parse_constant=lambda token: fail("nonfinite " + token))
    if not isinstance(value, dict):
        fail("manifest object")
    return value


def read_regular(path: Path, label: str) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        fail(label + " regular non-link")
    payload = path.read_bytes()
    after = path.lstat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        fail(label + " changed during read")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    if len(args.expected_manifest_sha256) != 64 or not (
            set(args.expected_manifest_sha256) <= HEX):
        fail("external manifest pin")
    original = args.package
    before = original.lstat()
    if not stat.S_ISDIR(before.st_mode) or original.is_symlink():
        fail("linked package root")
    package = original.resolve(strict=True)
    manifest_payload = read_regular(package / "source_manifest.json", "manifest")
    if hashlib.sha256(manifest_payload).hexdigest() != args.expected_manifest_sha256:
        fail("manifest hash")
    manifest = strict_json(manifest_payload)
    if (manifest.get("schema") != SCHEMA or
            manifest.get("v2_source_root_sha256") != V2_ROOT or
            manifest.get("v2_review_source_root_sha256") != REVIEW_ROOT):
        fail("schema/lineage")
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        fail("member rows")
    observed = []
    names = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "bytes", "sha256"}:
            fail("member schema")
        name = row["name"]
        if (not isinstance(name, str) or not name or Path(name).name != name or
                name in names or name == "source_manifest.json"):
            fail("member name")
        payload = read_regular(package / name, "member " + name)
        item = {"name": name, "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}
        if item != row:
            fail("member pin " + name)
        names.append(name)
        observed.append(item)
    root = hashlib.sha256(json.dumps(
        observed, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()
    if root != manifest.get("source_root_sha256"):
        fail("source root")
    entries = list(os.scandir(package))
    if ({entry.name for entry in entries} != set(names) | {"source_manifest.json"}
            or not all(entry.is_file(follow_symlinks=False) for entry in entries)):
        fail("exact regular closure")
    print(json.dumps({"schema": "strata-rm-global-swap-v3-source-verification",
                      "manifest_sha256": args.expected_manifest_sha256,
                      "source_root_sha256": root, "members": len(rows),
                      "status": "PASS_FROZEN_V3_SOURCE_CLOSURE"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
