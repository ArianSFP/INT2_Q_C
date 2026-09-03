#!/usr/bin/env python3
"""Run the target-rate fixture and all controls on CuPy; no payload aperture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
AUTHORIZATION = "RUN_SOURCE_FREE_TACTIC_RAMANUJAN384_V2_TARGET_RATE_CUPY"


def load(name: str, filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"loader {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    arguments = parser.parse_args()
    if arguments.authorization != AUTHORIZATION:
        raise SystemExit("source-free authorization")
    manifest = ROOT / "SOURCE_MANIFEST.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != arguments.manifest_sha256:
        raise SystemExit("source manifest SHA256")
    core = load("tactic_ramanujan384_v2_core", "scalable_core.py")
    io = load("tactic_ramanujan384_v2_io", "authenticated_io.py")
    capability = load("tactic_ramanujan384_v2_capability", "coarse_capability.py")
    adapter = load("tactic_ramanujan384_v2_adapter", "adapter.py")
    fixture = load("tactic_ramanujan384_v2_fixture", "source_free_fixture.py")
    decoder_module = load("tactic_ramanujan384_v2_fixture_decoder", "fixture_coarse_decoder.py")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() <= 0:
        raise SystemExit("CUDA device")
    canonical = core.canonical_gaussian_f64((2, core.BLOCK_VALUES), core.GAUSSIAN_SEEDS[0])
    copied = cp.asnumpy(cp.asarray(canonical, dtype=cp.float64))
    if canonical.tobytes(order="C") != np.ascontiguousarray(copied, dtype="<f8").tobytes(order="C"):
        raise SystemExit("CPU/CuPy canonical Gaussian bytes")
    reference = np.arange(2 * core.BLOCK_VALUES, dtype=np.float64).reshape(2, core.BLOCK_VALUES)
    cpu_matched, cpu_record = core.moment_matched_gaussian(
        np, reference, core.GAUSSIAN_SEEDS[1], (core.BLOCK_VALUES, core.BLOCK_VALUES)
    )
    gpu_matched, gpu_record = core.moment_matched_gaussian(
        cp, cp.asarray(reference), core.GAUSSIAN_SEEDS[1],
        (core.BLOCK_VALUES, core.BLOCK_VALUES),
    )
    if (np.ascontiguousarray(cpu_matched, dtype="<f8").tobytes(order="C")
            != np.ascontiguousarray(cp.asnumpy(gpu_matched), dtype="<f8").tobytes(order="C")
            or cpu_record["f64_sha256"] != gpu_record["f64_sha256"]):
        raise SystemExit("CPU/CuPy moment-matched Gaussian bytes")
    result = fixture.run(
        cp, core=core, io=io, capability_api=capability, adapter=adapter,
        decoder_class=decoder_module.SourceFreeZeroCoarseDecoder,
        decoder_source_path=ROOT / "fixture_coarse_decoder.py",
    )
    cp.cuda.Stream.null.synchronize()
    if result["physical_rate_bpw"] != 2.5 or not result["controls_rerun"]:
        raise SystemExit("target-rate fixture did not execute controls")
    if result["per_candidate_host_scalar_syncs"] != 0:
        raise SystemExit("per-candidate synchronization")
    result.update({
        "schema": "tactic-ramanujan384-v2-source-free-target-rate-cupy-receipt",
        "status": "PASS_TARGET_RATE_LITERAL_REPLAY_AND_ALL_CONTROLS",
        "canonical_gaussian_cpu_cupy_bytes_identical": True,
        "moment_matched_gaussian_cpu_cupy_bytes_identical": True,
        "canonical_gaussian_probe_sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
        "cupy_version": cp.__version__, "device_id": int(cp.cuda.Device().id),
        "qwen_payload_accessed": False, "coarse_model_payload_accessed": False,
        "network_accessed": False,
    })
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
