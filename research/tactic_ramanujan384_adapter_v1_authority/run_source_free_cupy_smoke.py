#!/usr/bin/env python3
"""Explicit source-free CuPy smoke; has no payload path arguments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


AUTHORIZATION = "RUN_SOURCE_FREE_TACTIC_RAMANUJAN384_AUTHORITY_CUPY_V1"
ROOT = Path(__file__).resolve().parent


def load_fixture():
    path = ROOT / "run_source_free_fixture.py"
    spec = importlib.util.spec_from_file_location("tactic_ramanujan384_authority_cupy_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    arguments = parser.parse_args()
    if arguments.authorization != AUTHORIZATION:
        raise SystemExit("explicit source-free authorization required")
    manifest = ROOT / "SOURCE_MANIFEST.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != arguments.manifest_sha256:
        raise SystemExit("authority source manifest drift")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() <= 0:
        raise SystemExit("CUDA device required")
    result = load_fixture().run(cp)
    cp.cuda.Stream.null.synchronize()
    result.update({
        "schema": "tactic-ramanujan384-authority-source-free-cupy-smoke-v1",
        "cupy_version": cp.__version__,
        "device_id": int(cp.cuda.Device().id),
        "qwen_payload_accessed": False,
        "coarse_model_payload_accessed": False,
    })
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
