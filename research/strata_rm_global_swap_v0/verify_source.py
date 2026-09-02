#!/usr/bin/env python3
"""Verify the immutable member list and source-root hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_hash(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: str(value["name"])):
        digest.update(str(row["name"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = json.loads((package / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "strata-rm-global-swap-v0-source-manifest":
        raise ValueError("manifest schema")
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest members")
    names = [row.get("name") for row in rows]
    if len(names) != len(set(names)) or "source_manifest.json" in names:
        raise ValueError("manifest member names")
    for row in rows:
        path = package / str(row["name"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or \
                sha256_file(path) != row["sha256"]:
            raise ValueError(f"manifest member mismatch: {row['name']}")
    if root_hash(rows) != manifest.get("source_root_sha256"):
        raise ValueError("source root hash")
    actual = sorted(path.name for path in package.iterdir()
                    if path.is_file() and path.name != "source_manifest.json")
    if sorted(names) != actual:
        raise ValueError(f"unmanifested package files: {sorted(set(actual) ^ set(names))}")
    print(json.dumps({
        "passed": True,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": len(rows),
        "status": manifest["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

