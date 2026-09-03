#!/usr/bin/env python3
"""Verify this review's exact flat closure and canonical member-row root."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")

def main() -> int:
    directory = Path(__file__).resolve().parent
    manifest_path = directory / "source_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    rows, names = [], set()
    for expected in manifest["members"]:
        name = expected["name"]
        if Path(name).name != name or name in names or name == manifest_path.name:
            raise RuntimeError("unsafe, duplicate, or recursive manifest member")
        payload = (directory / name).read_bytes()
        observed = {"name": name, "bytes": len(payload), "sha256": sha(payload)}
        if observed != expected:
            raise RuntimeError(f"member mismatch: {name}")
        rows.append(observed)
        names.add(name)
    root = sha(canonical(rows))
    if root != manifest["source_root_sha256"]:
        raise RuntimeError("canonical source root mismatch")
    if {p.name for p in directory.iterdir()} != names | {manifest_path.name}:
        raise RuntimeError("review directory is not an exact flat closure")
    print(f"PASS manifest_sha256={sha(manifest_path.read_bytes())}")
    print(f"PASS source_root_sha256={root}")
    print(f"PASS members={len(rows)} exact_flat_closure=true")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
