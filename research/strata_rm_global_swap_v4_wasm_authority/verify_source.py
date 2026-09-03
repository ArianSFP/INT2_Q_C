#!/usr/bin/env python3
"""Standalone exact-closure verifier for the final v4 Wasm authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


SCHEMA = "strata-rm-global-swap-v4-wasm-authority-source-manifest"
V3_ROOT = "83d79990515fca16387723cdea544d41fac76413fe80f919c30517d14551d6ad"
V3_MANIFEST = "9105dd69a2a82d1eaf14e176e4334189a4c31be840dafee467d243c231788e83"
REVIEW_ROOT = "3113631a5c64255d919f2bb5c545436452c8a721eb4130fcd32d7ffc4b2cdfe0"
REVIEW_MANIFEST = "ebe65fcf1abd73263be0176cdb70244ebca4f0a883eb6815c24c8956b0d0d89c"
HEX = frozenset("0123456789abcdef")


def fail(message: str) -> None:
    raise SystemExit("v4 source verification failed: " + message)


def strict_json(payload: bytes):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail("duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda token: fail("nonfinite " + token))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail("strict JSON: " + str(exc))
    if not isinstance(value, dict):
        fail("manifest object")
    return value


def read_regular(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        fail(label + " lstat: " + str(exc))
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        fail(label + " regular non-link")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        fail(label + " read: " + str(exc))
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        fail(label + " changed during read")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    if (len(args.expected_manifest_sha256) != 64 or
            not set(args.expected_manifest_sha256) <= HEX):
        fail("external manifest pin")
    original = args.package
    try:
        before = original.lstat()
    except OSError as exc:
        fail("package lstat: " + str(exc))
    if original.is_symlink() or not stat.S_ISDIR(before.st_mode):
        fail("linked package root")
    package = original.resolve(strict=True)
    manifest_payload = read_regular(package / "source_manifest.json", "manifest")
    if hashlib.sha256(manifest_payload).hexdigest() != args.expected_manifest_sha256:
        fail("manifest hash")
    manifest = strict_json(manifest_payload)
    if (manifest.get("schema") != SCHEMA or
            manifest.get("v3_source_root_sha256") != V3_ROOT or
            manifest.get("v3_manifest_sha256") != V3_MANIFEST or
            manifest.get("v3_review_source_root_sha256") != REVIEW_ROOT or
            manifest.get("v3_review_manifest_sha256") != REVIEW_MANIFEST):
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
        observed.append(item)
        names.append(name)
    root = hashlib.sha256(json.dumps(
        observed, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()
    if root != manifest.get("source_root_sha256"):
        fail("source root")
    entries = list(os.scandir(package))
    if ({entry.name for entry in entries} != set(names) | {"source_manifest.json"}
            or not all(entry.is_file(follow_symlinks=False) for entry in entries)):
        fail("exact regular closure")
    print(json.dumps({
        "schema": "strata-rm-global-swap-v4-source-verification",
        "manifest_sha256": args.expected_manifest_sha256,
        "source_root_sha256": root, "members": len(rows),
        "status": "PASS_FROZEN_V4_SOURCE_CLOSURE"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
