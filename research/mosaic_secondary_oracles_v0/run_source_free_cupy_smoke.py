#!/usr/bin/env python3
"""Manifest-authenticated source-free CuPy smoke; accepts no payload path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path


SCHEMA = "mosaic-secondary-oracles-source-manifest-v0"
STATUS = "SEALED_SOURCE_ONLY_NO_QWEN_OR_COARSE_PAYLOAD_AUTHORITY"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(name: str, payload: bytes):
    require(name not in sys.modules, "module collision")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{digest(payload)}>"
    module.__package__ = ""
    module.__authenticated_sha256__ = digest(payload)
    sys.modules[name] = module
    exec(compile(payload, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


def run(arguments: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parent
    manifest_payload = (root / "SOURCE_MANIFEST.json").read_bytes()
    require(digest(manifest_payload) == arguments.manifest_sha256, "manifest authorization")
    manifest = json.loads(manifest_payload.decode("utf-8"))
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == STATUS, "manifest identity")
    sources = {}
    for row in manifest["members"]:
        payload = (root / row["name"]).read_bytes()
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"], "source member closure")
        sources[row["name"]] = payload
    require(sources["run_source_free_cupy_smoke.py"] == Path(__file__).read_bytes(), "running source binding")
    contract = load("mosaic_secondary_gpu_contract", sources["gate_contract.py"])
    residual = load("mosaic_secondary_gpu_oracles", sources["residual_oracles.py"])
    backend = load("mosaic_secondary_gpu_backend", sources["cupy_backend.py"])
    return backend.source_free_smoke(
        residual,
        authorization=arguments.authorization,
        periods=contract.NON_DYADIC_PERIODS,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--manifest-sha256", required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True, separators=(",", ":"), allow_nan=False))
