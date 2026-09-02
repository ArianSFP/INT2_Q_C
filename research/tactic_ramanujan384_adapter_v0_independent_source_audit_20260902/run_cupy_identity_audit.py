#!/usr/bin/env python3
"""Source-free CPU/CuPy identity audit for frozen Ramanujan-384 mechanics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


AUTHORIZATION = "AUDIT_SOURCE_FREE_TACTIC_RAMANUJAN384_CPU_CUPY_IDENTITY_V0"
EXPECTED_MANIFEST = "287b8ad4c377956c9bb264d9d8731893a83e45180f75472f9b42968e3f20acde"
ROOT = Path(__file__).resolve().parents[1] / "tactic_ramanujan384_adapter_v0"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--producer-manifest-sha256", required=True)
    arguments = parser.parse_args()
    if arguments.authorization != AUTHORIZATION:
        raise RuntimeError("explicit source-free audit authorization")
    if arguments.producer_manifest_sha256 != EXPECTED_MANIFEST:
        raise RuntimeError("expected producer manifest argument")
    observed = hashlib.sha256((ROOT / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()
    if observed != EXPECTED_MANIFEST:
        raise RuntimeError("producer manifest drift")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CUDA device")
    codec = load("ramanujan_identity_codec", ROOT / "ramanujan_codec.py")
    parent = codec.load_audited_parent()
    cpu_basis = codec.build_public_dictionary(np)
    gpu_basis = codec.build_public_dictionary(cp)
    dictionary_equal = np.array_equal(cpu_basis["dictionary"], cp.asnumpy(gpu_basis["dictionary"]))
    norms_equal = np.array_equal(cpu_basis["norms"], cp.asnumpy(gpu_basis["norms"]))
    coordinate = np.arange(codec.BLOCK_VALUES, dtype=np.int64)
    p7 = np.asarray([codec.ramanujan_sum(7, value) for value in range(7)], dtype=np.float64)
    p11 = np.asarray([codec.ramanujan_sum(11, value) for value in range(11)], dtype=np.float64)
    residual = np.stack([
        0.013 * p7[(coordinate - block) % 7] + 0.004 * p11[(coordinate - 2 * block) % 11]
        for block in range(2)
    ])
    cpu_encoded = codec.encode_residual_blocks(np, residual, cpu_basis, "gate")
    gpu_encoded = codec.encode_residual_blocks(cp, cp.asarray(residual), gpu_basis, "gate")
    packet_equal = b"".join(cpu_encoded["packets"]) == b"".join(gpu_encoded["packets"])
    remaining_equal = (
        np.asarray(cpu_encoded["correction"]).astype("<f8", copy=False).tobytes()
        == cp.asnumpy(gpu_encoded["correction"]).astype("<f8", copy=False).tobytes()
    )
    phase_cpu = parent.phase_destroyed_blocks(np, residual, codec.PHASE_SEED)
    phase_gpu = cp.asnumpy(parent.phase_destroyed_blocks(cp, cp.asarray(residual), codec.PHASE_SEED))
    phase_equal = np.array_equal(phase_cpu, phase_gpu)
    gaussian = []
    for seed in codec.GAUSSIAN_SEEDS:
        cpu = parent.moment_matched_gaussian_blocks(np, residual, seed)
        gpu = cp.asnumpy(parent.moment_matched_gaussian_blocks(cp, cp.asarray(residual), seed))
        gaussian.append({
            "seed": seed,
            "bitwise_equal": bool(np.array_equal(cpu, gpu)),
            "cpu_mean_energy_closed": bool(np.allclose(
                np.sum((cpu - np.mean(cpu, axis=1, keepdims=True)) ** 2, axis=1),
                np.sum((residual - np.mean(residual, axis=1, keepdims=True)) ** 2, axis=1),
                rtol=2e-12, atol=2e-12,
            )),
            "gpu_mean_energy_closed": bool(np.allclose(
                np.sum((gpu - np.mean(gpu, axis=1, keepdims=True)) ** 2, axis=1),
                np.sum((residual - np.mean(residual, axis=1, keepdims=True)) ** 2, axis=1),
                rtol=2e-12, atol=2e-12,
            )),
        })
    cp.cuda.Stream.null.synchronize()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    all_gaussian_equal = all(row["bitwise_equal"] for row in gaussian)
    full_identity = dictionary_equal and norms_equal and packet_equal and remaining_equal and phase_equal and all_gaussian_equal
    return {
        "schema": "tactic-ramanujan384-source-free-cpu-cupy-identity-audit-v0",
        "status": (
            "PASS_FULL_CPU_CUPY_IDENTITY"
            if full_identity
            else "HOLD_CPU_CUPY_IDENTITY__BACKEND_STABLE_CONTROLS_REQUIRED"
        ),
        "device_id": int(cp.cuda.Device().id),
        "device_name": str(name),
        "dictionary_bitwise_equal": dictionary_equal,
        "norms_bitwise_equal": norms_equal,
        "source_packet_stream_equal": packet_equal,
        "source_correction_bitwise_equal": remaining_equal,
        "phase_control_bitwise_equal": phase_equal,
        "gaussian_controls": gaussian,
        "all_gaussian_controls_bitwise_equal": all_gaussian_equal,
        "payload_authorized": False,
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "network_accessed": False,
    }


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True, separators=(",", ":")))

