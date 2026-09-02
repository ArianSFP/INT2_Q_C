#!/usr/bin/env python3
"""Fail-closed verifier using exactly the freezer's manifest implementation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def load_algorithm(root: Path):
    path = root / "manifest.py"
    spec = importlib.util.spec_from_file_location("tactic_ramanujan384_authority_verify_manifest", path)
    if spec is None or spec.loader is None:
        raise SystemExit("manifest algorithm loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    arguments = parser.parse_args()
    root = arguments.package.resolve(strict=True)
    if root != arguments.package.absolute() or root != Path(__file__).resolve().parent:
        raise SystemExit("canonical exact package path required")
    manifest_path = root / "SOURCE_MANIFEST.json"
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != arguments.manifest_sha256:
        raise SystemExit("source manifest SHA256")
    document = json.loads(payload.decode("ascii"), parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"nonfinite {token}")
    ))
    if document.get("schema") != "tactic-ramanujan384-authority-source-manifest-v1":
        raise SystemExit("source manifest schema")
    algorithm = load_algorithm(root)
    recorded = algorithm.canonical_member_rows(document.get("members", []))
    actual = algorithm.collect(root)
    if recorded != actual:
        raise SystemExit("source member drift")
    actual_root = algorithm.source_root(actual)
    if actual_root != document.get("source_root_sha256"):
        raise SystemExit("canonical source root")
    lock = json.loads((root / "dependency_lock.json").read_text(encoding="utf-8"))
    parents = {
        "producer_v0": root.parent / "tactic_ramanujan384_adapter_v0" / "SOURCE_MANIFEST.json",
        "independent_audit_v0": (root.parent
            / "tactic_ramanujan384_adapter_v0_independent_source_audit_20260902"
            / "SOURCE_MANIFEST.json"),
    }
    for key, path in parents.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != lock[key]["source_manifest_sha256"]:
            raise SystemExit(f"pinned dependency drift: {key}")
    print(json.dumps({
        "schema": document["schema"],
        "status": "PASS_CANONICAL_SOURCE_ROOT_AND_PINNED_DEPENDENCIES",
        "source_root_sha256": actual_root,
        "members": len(actual),
        "qwen_payload_accessed": False,
        "coarse_model_payload_accessed": False,
        "network_accessed": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
