#!/usr/bin/env python3
"""Payload-free CLI for the Ramanujan-384 CuPy smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name: str, filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    arguments = parser.parse_args()
    manifest_payload = (ROOT / "SOURCE_MANIFEST.json").read_bytes()
    if hashlib.sha256(manifest_payload).hexdigest() != arguments.manifest_sha256:
        raise RuntimeError("source manifest SHA256")
    codec = load("tactic_ramanujan384_cupy_codec", "ramanujan_codec.py")
    backend = load("tactic_ramanujan384_cupy_backend", "cupy_backend.py")
    return backend.source_free_smoke(codec, authorization=arguments.authorization)


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True, separators=(",", ":")))
