#!/usr/bin/env python3
"""Authenticate a real CuPy runtime, then execute the producer GPU smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from independent_auth import authenticate_source


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth = authenticate_source(args.source)

    import cupy as cp

    module_path = Path(cp.__file__).resolve(strict=True)
    require(module_path.is_file() and not module_path.is_symlink(),
            "CuPy module regular file")
    require(args.source.resolve() not in module_path.parents,
            "CuPy must not be supplied by producer source")
    distribution_version = importlib.metadata.version("cupy-cuda12x")
    require(distribution_version == cp.__version__, "CuPy distribution identity")
    require(cp.ndarray.__module__.startswith("cupy"), "real CuPy ndarray class")
    require(cp.cuda.runtime.getDeviceCount() > 0, "CUDA device")
    active_device = int(cp.cuda.Device().id)
    properties = cp.cuda.runtime.getDeviceProperties(active_device)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    command = [
        sys.executable, "-I", "-B", str(args.source / "run_cupy_smoke.py")
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=300
    )
    producer = json.loads(completed.stdout)
    require(producer["status"] == "PASS_SOURCE_FREE_CUPY_SMOKE",
            "producer CuPy smoke status")
    require(producer["cupy_version"] == cp.__version__, "CuPy version receipt")
    require(producer["device_name"] == str(device_name),
            "producer records device zero; audit requires active device zero")
    require(active_device == 0,
            "producer hard-codes device-zero receipt; rerun on active device zero")
    require(producer["distortion_max_abs_error"] <= 1e-15,
            "distortion parity")
    require(producer["six_plane_index_equal"] and producer["bmp_equal"],
            "generated primitive parity")
    require(not producer["model_or_qwen_payload_accessed"] and
            not producer["network_accessed"], "source-free boundary")
    cp.cuda.get_current_stream().synchronize()
    receipt = {
        "schema": "strata-bmp-qtt6-independent-real-cupy-audit-v0",
        "status": "PASS_REAL_CUPY_GENERATED_PRIMITIVE_SMOKE__HOLD_PAYLOAD",
        "source_auth": auth,
        "cupy_module_path": str(module_path),
        "cupy_version": cp.__version__,
        "distribution_version": distribution_version,
        "active_device": active_device,
        "device_name": str(device_name),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "producer_receipt": producer,
        "producer_stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
        "payloads_opened": 0,
        "claim_boundary": (
            "Generated primitive parity only; the production search remains NumPy "
            "and no Qwen/STRATA/control payload was accessed."
        ),
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
