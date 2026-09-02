#!/usr/bin/env python3
"""External source-free CuPy parity check for frozen finite v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path


EXPECTED_MANIFEST = "bf0659d1fd6742768d14790ea980aa17321818d15e19ddd7d0dfaa8a223009b8"
EXPECTED_ROOT = "725991e0c1e10c67db4ba36097f80e78ffed158ea36b1b746bbbd6cef50ffa98"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(name: str, payload: bytes):
    module = types.ModuleType(name)
    module.__file__ = f"<sourcefree-smoke:{name}:{sha256(payload)}>"
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(payload, module.__file__, "exec", dont_inherit=True,
                 optimize=0), module.__dict__)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    arguments = parser.parse_args()
    package = arguments.package.resolve(strict=True)
    manifest_payload = (package / "SOURCE_MANIFEST.json").read_bytes()
    assert sha256(manifest_payload) == EXPECTED_MANIFEST
    manifest = json.loads(manifest_payload)
    assert manifest["source_root_sha256"] == EXPECTED_ROOT
    members = {}
    for row in manifest["members"]:
        payload = (package / row["name"]).read_bytes()
        assert len(payload) == row["bytes"] and sha256(payload) == row["sha256"]
        members[row["name"]] = payload
    spec = load("tactic_v3_smoke_spec", members["format_spec.py"])
    encoder = load("tactic_v3_smoke_encoder", members["finite_encoder.py"])
    decoder = load("tactic_v3_smoke_decoder", members["independent_decoder.py"])
    import cupy as cp
    import numpy as np

    index = np.arange(spec.COARSE_TILE_VALUES, dtype=np.int64)
    symbols = (((index * 37 + 11) % 127) - 63).astype("<i4")
    residual = (0.013 * np.sin(index * 0.0017) +
                0.002 * np.cos(index * 0.0061)).astype(np.float64)
    reconstruction = (0.12 + 0.015 * np.sin(index * 0.00031)).astype(np.float32)
    continuous = encoder.continuous_tile(
        cp, symbols, residual, 1, spec)
    records, correction_encoder, encode_receipt = encoder.encode_tile(
        cp, np, symbols, residual, reconstruction, 1, spec)
    correction_decoder, decode_receipt = decoder.decode_fine_tile(
        cp, np, records, symbols, reconstruction, 1, spec)
    maximum_difference = float(cp.max(cp.abs(
        correction_encoder - correction_decoder)).item())
    assert maximum_difference == 0.0
    error = cp.asarray(residual, dtype=cp.float64)
    decoded_sse = float(cp.sum(
        (error - correction_decoder) ** 2, dtype=cp.float64).item())
    assert abs(decoded_sse - encode_receipt["finite_sse_fp64"]) <= 2e-10
    record = {
        "schema": "tactic-dh384-finite-v3-sourcefree-cupy-smoke-v1",
        "status": "PASS_SOURCEFREE_CUPY_ENCODER_INDEPENDENT_DECODER_PARITY",
        "source_manifest_sha256": EXPECTED_MANIFEST,
        "source_root_sha256": EXPECTED_ROOT,
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
        "records": len(records) // spec.FINE_RECORD_BYTES,
        "fine_bytes": len(records),
        "continuous": continuous,
        "encode": encode_receipt,
        "decode": decode_receipt,
        "maximum_encoder_decoder_correction_abs_difference_fp64":
            maximum_difference,
        "decoded_sse_fp64": decoded_sse,
        "qwen_or_model_payload_accessed": False,
        "v6_live_result_accessed": False,
        "positive_claim_authority": False,
    }
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
