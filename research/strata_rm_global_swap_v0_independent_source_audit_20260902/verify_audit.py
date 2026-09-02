#!/usr/bin/env python3
"""Verify the frozen independent-audit package, including extra directories."""

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
    manifest_path = package / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "strata-rm-global-swap-v0-independent-audit-manifest":
        raise ValueError("audit manifest schema")
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("audit manifest members")
    names = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "bytes", "sha256"}:
            raise ValueError("audit manifest member fields")
        name = row["name"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("flat audit member name")
        names.append(name)
    if len(names) != len(set(names)) or "source_manifest.json" in names:
        raise ValueError("unique audit manifest names")
    expected_entries = set(names) | {"source_manifest.json"}
    actual_entries = {path.name for path in package.iterdir()}
    if actual_entries != expected_entries:
        raise ValueError("unexpected audit file or directory")
    for row in rows:
        path = package / str(row["name"])
        if (not path.is_file() or path.is_symlink() or
                path.stat().st_size != row["bytes"] or
                sha256_file(path) != row["sha256"]):
            raise ValueError(f"audit member mismatch: {row['name']}")
    actual_root = root_hash(rows)
    if actual_root != manifest.get("source_root_sha256"):
        raise ValueError("audit source root")
    print(json.dumps({
        "passed": True,
        "members": len(rows),
        "source_root_sha256": actual_root,
        "producer_source_root_sha256": manifest["producer_source_root_sha256"],
        "status": manifest["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
