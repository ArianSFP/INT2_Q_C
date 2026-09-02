#!/usr/bin/env python3
"""Standalone verifier for the exact v2 flat source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


SCHEMA = "strata-rm-global-swap-v2-authority-source-manifest"
V1_ROOT = "980a5f1d272ca5ffc7b4d35e7c234a86994d135fcacaf0d47a8b3e00fc3d4f14"
REVIEW_ROOT = "1dfa55969b87543adbee785d72933f9ccb6f754eaade9e4e340a022c96c1afa8"
HEX = frozenset("0123456789abcdef")


def fail(message: str) -> None:
    raise SystemExit("source verification failed: " + message)


def strict_json(payload: bytes):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail("duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
                           parse_constant=lambda token: fail("nonfinite " + token))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail("strict JSON: " + str(exc))
    if not isinstance(value, dict):
        fail("manifest object")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
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
        fail("external manifest SHA-256")
    # Gap 1 repair: lstat and reject the caller's root before resolve().
    try:
        original = args.package
        before = original.lstat()
        if not stat.S_ISDIR(before.st_mode) or original.is_symlink():
            fail("package root is linked or not a directory")
        package = original.resolve(strict=True)
    except OSError as exc:
        fail("package resolution: " + str(exc))
    manifest_payload = regular_bytes(package / "source_manifest.json", "manifest")
    if hashlib.sha256(manifest_payload).hexdigest() != args.expected_manifest_sha256:
        fail("external manifest pin")
    manifest = strict_json(manifest_payload)
    if (manifest.get("schema") != SCHEMA or
            manifest.get("v1_source_root_sha256") != V1_ROOT or
            manifest.get("v1_review_source_root_sha256") != REVIEW_ROOT):
        fail("schema or lineage")
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        fail("members")
    observed = []
    names = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "bytes", "sha256"}:
            fail("member schema")
        name = row["name"]
        if (not isinstance(name, str) or not name or Path(name).name != name or
                name in names or name == "source_manifest.json"):
            fail("member name")
        payload = regular_bytes(package / name, "member " + name)
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
        fail("exact regular-file closure")
    print(json.dumps({"schema": "strata-rm-global-swap-v2-source-verification",
                      "manifest_sha256": args.expected_manifest_sha256,
                      "source_root_sha256": root,
                      "members": len(rows),
                      "root_link_rejected_before_resolve": True,
                      "status": "PASS_FROZEN_V2_SOURCE_CLOSURE"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
