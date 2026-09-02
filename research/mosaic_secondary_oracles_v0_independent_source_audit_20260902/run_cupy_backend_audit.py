#!/usr/bin/env python3
"""Payload-free NumPy/CuPy reproducibility audit for MOSAIC secondary oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import types
from pathlib import Path


UPSTREAM_MANIFEST_SHA256 = "4259e8e8dc87b4c25301ca89ade7dbd63c1e0c9e3415fdaa4d7881d7d10ccc06"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(name: str, payload: bytes):
    require(name not in sys.modules, "module collision")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{digest(payload)}>"
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(payload, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


def f64_sha(array, np) -> str:
    return digest(np.asarray(array).astype("<f8", copy=False).tobytes(order="C"))


def run(root: Path) -> dict:
    import cupy as cp
    import numpy as np

    resolved = root.resolve(strict=True)
    manifest_payload = (resolved / "SOURCE_MANIFEST.json").read_bytes()
    require(digest(manifest_payload) == UPSTREAM_MANIFEST_SHA256, "upstream manifest pin")
    manifest = json.loads(manifest_payload.decode("utf-8"))
    sources = {}
    for row in manifest["members"]:
        payload = (resolved / row["name"]).read_bytes()
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"], "member closure")
        sources[row["name"]] = payload
    residual = load("mosaic_secondary_cupy_audit_oracles", sources["residual_oracles.py"])
    contract = load("mosaic_secondary_cupy_audit_contract", sources["gate_contract.py"])

    require(cp.__name__ == "cupy", "actual CuPy name")
    require(getattr(cp, "__file__", None), "actual CuPy module file")
    require(cp.cuda.runtime.getDeviceCount() > 0, "CUDA device")
    device = int(cp.cuda.Device().id)
    props = cp.cuda.runtime.getDeviceProperties(device)
    name = props.get("name", b"")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")

    length = 256
    columns = 64
    cpu_basis = residual.build_ramanujan_basis(
        np, length=length, periods=contract.NON_DYADIC_PERIODS, maximum_columns=columns
    )
    gpu_basis = residual.build_ramanujan_basis(
        cp, length=length, periods=contract.NON_DYADIC_PERIODS, maximum_columns=columns
    )
    gpu_q = cp.asnumpy(gpu_basis["basis"])
    cpu_q = np.asarray(cpu_basis["basis"])
    basis_max_abs = float(np.max(np.abs(cpu_q - gpu_q)))

    coordinate_np = np.arange(length, dtype=np.float64)
    host_residual = np.stack([
        np.sin((2.0 * math.pi / 7.0) * coordinate_np + 0.13 * block)
        + 0.07 * np.cos((4.0 * math.pi / 11.0) * coordinate_np - 0.09 * block)
        for block in range(8)
    ]).astype(np.float64)
    gpu_residual = cp.asarray(host_residual)
    cpu_gaussian = residual.moment_matched_gaussian_blocks(np, host_residual, 10619863)
    gpu_gaussian = cp.asnumpy(residual.moment_matched_gaussian_blocks(cp, gpu_residual, 10619863))
    source_energy = float(np.sum(host_residual * host_residual, dtype=np.float64)) / 0.04
    cpu_ram = residual.ramanujan_panel_metrics(
        np, host_residual, cpu_basis, source_energy=source_energy
    )
    gpu_ram = residual.ramanujan_panel_metrics(
        cp, gpu_residual, gpu_basis, source_energy=source_energy
    )
    cpu_ar = residual.ar_hankel_panel_metrics(
        np, host_residual, source_energy=source_energy, orders=(1, 2, 4, 8, 12)
    )
    gpu_ar = residual.ar_hankel_panel_metrics(
        cp, gpu_residual, source_energy=source_energy, orders=(1, 2, 4, 8, 12)
    )
    cp.cuda.Stream.null.synchronize()

    cpu_basis_sha = f64_sha(cpu_q, np)
    gpu_basis_sha = f64_sha(gpu_q, np)
    cpu_gaussian_sha = f64_sha(cpu_gaussian, np)
    gpu_gaussian_sha = f64_sha(gpu_gaussian, np)
    require(cpu_basis_sha != gpu_basis_sha, "backend basis is unexpectedly bit-identical")
    require(cpu_gaussian_sha != gpu_gaussian_sha, "backend RNG is unexpectedly bit-identical")
    return {
        "schema": "mosaic-secondary-oracles-cupy-backend-audit-v1",
        "status": "PASS_MECHANICS__BACKEND_NOT_BIT_IDENTICAL",
        "upstream_manifest_sha256": UPSTREAM_MANIFEST_SHA256,
        "cupy_version": cp.__version__,
        "numpy_version": np.__version__,
        "cupy_module_file": str(cp.__file__),
        "device_id": device,
        "device_name": str(name),
        "cupy_array_type": f"{type(gpu_basis['basis']).__module__}.{type(gpu_basis['basis']).__name__}",
        "cpu_basis_f64_sha256": cpu_basis_sha,
        "gpu_basis_f64_sha256": gpu_basis_sha,
        "basis_bit_identical": False,
        "basis_max_abs_difference": basis_max_abs,
        "cpu_gaussian_f64_sha256": cpu_gaussian_sha,
        "gpu_gaussian_f64_sha256": gpu_gaussian_sha,
        "gaussian_control_bit_identical": False,
        "cpu_ramanujan_ideal_remaining_sse": cpu_ram["ideal_public_basis_waterfill_remaining_sse"],
        "gpu_ramanujan_ideal_remaining_sse": gpu_ram["ideal_public_basis_waterfill_remaining_sse"],
        "cpu_ar_winner_order": cpu_ar["winner"]["order"],
        "gpu_ar_winner_order": gpu_ar["winner"]["order"],
        "production_requirement": "freeze one actual CuPy version/device arithmetic and generate source plus every control through that identical authenticated backend; QR/RNG are not cross-backend bit-exact",
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "matched_control_payload_accessed": False,
        "network_accessed": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--upstream-source", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args().upstream_source), sort_keys=True, separators=(",", ":"), allow_nan=False))
