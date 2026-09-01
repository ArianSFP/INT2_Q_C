#!/usr/bin/env python3
"""Verify the frozen source-only closure without opening any external input."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ALLOWED = {
    "README.md",
    "SOURCE_MANIFEST.json",
    "design_lock.json",
    "nonlocal_wfa.py",
    "run_synthetic.py",
    "test_source_only.py",
    "verify_result.py",
    "verify_source.py",
}
ALLOWED_IMPORT_ROOTS = {
    "argparse",
    "ast",
    "dataclasses",
    "fractions",
    "hashlib",
    "json",
    "math",
    "nonlocal_wfa",
    "numpy",
    "pathlib",
    "platform",
    "struct",
    "tempfile",
    "typing",
    "unittest",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    observed = {path.name for path in ROOT.iterdir() if path.is_file()}
    if observed != ALLOWED:
        raise RuntimeError(f"source file set mismatch: {sorted(observed ^ ALLOWED)}")
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="ascii"))
    if manifest["schema"] != "nonlocal-wfa-global-state-source-manifest-v0":
        raise RuntimeError("manifest schema")
    if manifest["status"] != "SEALED_SOURCE_FREE_SYNTHETIC_ONLY_NO_PAYLOAD_AUTHORITY":
        raise RuntimeError("manifest status")
    rows = manifest["files"]
    expected_names = ALLOWED - {"SOURCE_MANIFEST.json"}
    if {row["path"] for row in rows} != expected_names:
        raise RuntimeError("manifest file names")
    for row in rows:
        path = ROOT / row["path"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise RuntimeError(f"manifest mismatch: {row['path']}")
    for filename in ("nonlocal_wfa.py", "run_synthetic.py", "test_source_only.py", "verify_result.py", "verify_source.py"):
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"), filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".", 1)[0]]
            else:
                continue
            for root in roots:
                if root and root not in ALLOWED_IMPORT_ROOTS and root != "__future__":
                    raise RuntimeError(f"forbidden import {root} in {filename}")
    lock = json.loads((ROOT / "design_lock.json").read_text(encoding="ascii"))
    access = lock["access_attestation"]
    if any(bool(value) for value in access.values()):
        raise RuntimeError("source-free access attestation")
    print(json.dumps({
        "status": "PASS_SEALED_SOURCE_FREE_SYNTHETIC_ONLY",
        "files_authenticated": len(rows),
        "payload_authority": False,
        "external_inputs_opened": 0,
        "gpu_paths": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

