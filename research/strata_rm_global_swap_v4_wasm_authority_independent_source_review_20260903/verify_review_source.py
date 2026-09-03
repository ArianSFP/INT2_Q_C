#!/usr/bin/env python3
"""Verify the exact independent-review source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


SCHEMA = "strata-rm-global-swap-v4-independent-source-review-manifest"
PRODUCER_MANIFEST = "62bf04cd413317e2e8b98635713419c84394db7b7d2bd4567afddf56957a5e2f"
PRODUCER_ROOT = "f535699c4828a02e5769b916b1207309768f7381db5f92a0fb58e10915ae8a25"
HEX = frozenset("0123456789abcdef")


def fail(message: str) -> None:
    raise SystemExit("v4 independent review verification failed: " + message)


def strict_json(payload: bytes, label: str):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label}: duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
                           parse_constant=lambda token: fail(
                               f"{label}: nonfinite {token}"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"{label}: strict JSON {exc}")
    if not isinstance(value, dict):
        fail(f"{label}: object")
    return value


def regular(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            fail(f"{label}: regular non-link")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        fail(f"{label}: read {exc}")
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_mode)
    if identity(before) != identity(after):
        fail(f"{label}: changed")
    return payload


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    if (len(args.expected_manifest_sha256) != 64 or
            not set(args.expected_manifest_sha256) <= HEX):
        fail("external manifest pin")
    before = args.package.lstat()
    if args.package.is_symlink() or not stat.S_ISDIR(before.st_mode):
        fail("package real directory")
    root = args.package.resolve(strict=True)
    payload = regular(root / "source_manifest.json", "manifest")
    if hashlib.sha256(payload).hexdigest() != args.expected_manifest_sha256:
        fail("manifest hash")
    manifest = strict_json(payload, "manifest")
    required = {"schema", "status", "producer_manifest_sha256",
                "producer_source_root_sha256", "source_root_sha256",
                "members", "execution_attestation", "access_attestation",
                "disposition", "claim_boundary"}
    if (canonical(manifest) + b"\n" != payload or set(manifest) != required or
            manifest["schema"] != SCHEMA or
            manifest["producer_manifest_sha256"] != PRODUCER_MANIFEST or
            manifest["producer_source_root_sha256"] != PRODUCER_ROOT):
        fail("canonical manifest/schema/producer binding")
    rows = manifest["members"]
    if not isinstance(rows, list) or not rows:
        fail("members")
    observed = []
    names = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "bytes", "sha256"}:
            fail("member row")
        name = row["name"]
        if (not isinstance(name, str) or not name or Path(name).name != name or
                name == "source_manifest.json" or name in names):
            fail("member name")
        member = regular(root / name, "member " + name)
        item = {"name": name, "bytes": len(member),
                "sha256": hashlib.sha256(member).hexdigest()}
        if item != row:
            fail("member pin " + name)
        names.append(name)
        observed.append(item)
    if names != sorted(names, key=lambda value: value.encode("utf-8")):
        fail("UTF-8 member order")
    root_hash = hashlib.sha256(canonical(observed)).hexdigest()
    if root_hash != manifest["source_root_sha256"]:
        fail("review source root")
    entries = list(os.scandir(root))
    if ({entry.name for entry in entries} != set(names) | {"source_manifest.json"}
            or not all(entry.is_file(follow_symlinks=False) for entry in entries)):
        fail("exact flat closure")
    static = strict_json(regular(root / "STATIC_REVIEW_RECEIPT.json", "static"),
                         "static")
    if (static.get("producer_manifest_sha256") != PRODUCER_MANIFEST or
            static.get("producer_source_root_sha256") != PRODUCER_ROOT or
            static.get("python_tests_executed") is not False or
            static.get("wasmtime_executed") is not False):
        fail("static review claim boundary")
    readme = regular(root / "README.md", "README").decode("utf-8")
    for phrase in ("semantic-state purity is audit-enforced",
                   "executing-host provenance is not fully closed",
                   "read metric remains narrow", "runtime remains pending"):
        if phrase not in readme:
            fail("README finding " + phrase)
    print(json.dumps({
        "schema": "strata-rm-global-swap-v4-independent-review-verification",
        "manifest_sha256": args.expected_manifest_sha256,
        "source_root_sha256": root_hash,
        "members": len(rows),
        "status": "PASS_FROZEN_V4_INDEPENDENT_SOURCE_REVIEW_CLOSURE",
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

