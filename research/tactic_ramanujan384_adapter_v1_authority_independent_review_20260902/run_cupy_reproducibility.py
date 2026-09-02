#!/usr/bin/env python3
"""Source-free CPU/CuPy control and codec reproducibility review."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


AUTHORIZATION = "AUDIT_SOURCE_FREE_TACTIC_RAMANUJAN384_V1_CPU_CUPY_REPRODUCIBILITY"
EXPECTED_MANIFEST = "f4ba72b9371d77ad4347d5a4fe377677473844dd696032e662acc6cd3bde22b4"
PRODUCER = Path(__file__).resolve().parents[1] / "tactic_ramanujan384_adapter_v1_authority"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PRODUCER / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
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
    if arguments.authorization != AUTHORIZATION or arguments.producer_manifest_sha256 != EXPECTED_MANIFEST:
        raise RuntimeError("explicit source-free pinned authorization")
    if hashlib.sha256((PRODUCER / "SOURCE_MANIFEST.json").read_bytes()).hexdigest() != EXPECTED_MANIFEST:
        raise RuntimeError("producer manifest drift")
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CUDA device")
    contract = load("review_cupy_contract", "contract.py")
    controls = load("review_cupy_controls", "stable_controls.py")
    codec = load("review_cupy_codec", "codec_authority.py")
    shape = contract.define_shape(32, 160)
    reference = np.arange(shape.blocks_per_role * 4096, dtype=np.float64).reshape(
        shape.blocks_per_role, 4096
    )
    valid = tuple(shape.valid_values_for_block(block) for block in range(shape.blocks_per_role))
    control_rows = []
    for seed in codec.GAUSSIAN_SEEDS:
        cpu = controls.moment_matched_blocks(np, reference, seed, valid)
        gpu = controls.moment_matched_blocks(cp, cp.asarray(reference), seed, valid)
        cpu_bytes = controls.host_bytes(np, cpu)
        gpu_bytes = controls.host_bytes(cp, gpu)
        control_rows.append({
            "seed": seed,
            "host_f64_sha256": hashlib.sha256(cpu_bytes).hexdigest(),
            "cpu_cupy_control_bytes_equal": cpu_bytes == gpu_bytes,
        })
    coordinate = np.arange(shape.role_values, dtype=np.float64)
    source = 0.01 * ((coordinate % 7) - 3) + 0.002 * ((coordinate % 11) - 5)
    cpu_prepared = codec.prepare_basis(np)
    cp.get_default_memory_pool().free_all_blocks()
    gpu_prepared = codec.prepare_basis(cp)
    pool_after_prepared = int(cp.get_default_memory_pool().total_bytes())
    cpu_encoded = codec.encode_role(np, source, np.zeros_like(source), shape, "gate", cpu_prepared)
    gpu_encoded = codec.encode_role(
        cp, cp.asarray(source), cp.zeros(shape.role_values, dtype=cp.float64),
        shape, "gate", gpu_prepared,
    )
    cp.cuda.Stream.null.synchronize()
    pool_after_first = int(cp.get_default_memory_pool().total_bytes())
    gpu_second = codec.encode_role(
        cp, cp.asarray(source), cp.zeros(shape.role_values, dtype=cp.float64),
        shape, "gate", gpu_prepared,
    )
    cp.cuda.Stream.null.synchronize()
    pool_after_second = int(cp.get_default_memory_pool().total_bytes())
    device = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(device, bytes):
        device = device.decode("utf-8")
    packet_equal = cpu_encoded["stream"] == gpu_encoded["stream"]
    correction_equal = (
        np.ascontiguousarray(cpu_encoded["correction"], dtype="<f8").tobytes()
        == np.ascontiguousarray(cp.asnumpy(gpu_encoded["correction"]), dtype="<f8").tobytes()
    )
    repeat_equal = gpu_encoded["stream"] == gpu_second["stream"]
    controls_equal = all(row["cpu_cupy_control_bytes_equal"] for row in control_rows)
    return {
        "schema": "tactic-ramanujan384-authority-v1-source-free-cpu-cupy-review-v0",
        "status": (
            "PASS_CONTROL_BYTES__CODEC_REPRODUCIBILITY_RECORDED"
            if controls_equal and repeat_equal
            else "HOLD_CPU_CUPY_REPRODUCIBILITY"
        ),
        "device": str(device),
        "cupy_version": cp.__version__,
        "controls": control_rows,
        "all_control_bytes_equal": controls_equal,
        "cpu_cupy_packet_stream_equal": packet_equal,
        "cpu_cupy_correction_bytes_equal": correction_equal,
        "cupy_repeat_packet_stream_equal": repeat_equal,
        "shared_dictionary_pointer_stable": int(gpu_prepared["dictionary"].data.ptr) != 0,
        "shared_gram_pointer_stable": int(gpu_prepared["gram"].data.ptr) != 0,
        "pool_bytes_after_prepared": pool_after_prepared,
        "pool_bytes_after_first_encode": pool_after_first,
        "pool_bytes_after_second_encode": pool_after_second,
        "candidate_scratch_explicitly_preallocated": False,
        "payload_authorized": False,
        "qwen_payload_accessed": False,
        "coarse_model_payload_accessed": False,
        "network_accessed": False,
    }


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True, separators=(",", ":")))

