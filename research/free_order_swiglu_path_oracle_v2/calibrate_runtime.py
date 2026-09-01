#!/usr/bin/env python3
"""Source-free full-geometry CuPy calibration for FOSP-ARX-v2-DIRECT.

This program accepts no workspace, source, model, manifest-selection, or
validation path.  It generates one deterministic synthetic 768x3x2048 tensor
on GPU and exercises the exact production pair, assignment, factoradic, and
FP16 replay implementation.  Its create-new receipt is not an execution
authorization; an independent runtime audit must first prove the zero-access
ledger and bind the runtime tuple.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "free_order_swiglu_path_runtime_calibration_v2"
SEED = 26_091_003


def _load_oracle(package: Path) -> Any:
    path = package / "free_order_oracle_v2.py"
    spec = importlib.util.spec_from_file_location("fosp_v2_calibrated_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    started = time.time()
    args = _parse_args()
    package = Path(__file__).absolute().parent.resolve(strict=True)
    output_argument = args.output.absolute()
    if output_argument.exists() or output_argument.is_symlink():
        raise RuntimeError("calibration output already exists")
    output_parent = output_argument.parent.resolve(strict=True)
    if output_parent == package or package in output_parent.parents or output_parent in package.parents:
        raise RuntimeError("calibration output parent must be disjoint from frozen package")
    output = output_parent / output_argument.name

    oracle = _load_oracle(package)
    artifact_rows, artifact_raw = oracle._artifact_rows(package)

    import numpy as np  # type: ignore
    import cupy as cp  # type: ignore
    import scipy  # type: ignore
    from scipy.optimize import linear_sum_assignment  # type: ignore

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    cp.cuda.Device(0).use()
    free_before, total_memory = cp.cuda.runtime.memGetInfo()
    random = cp.random.RandomState(SEED)
    synthetic = random.standard_normal(
        (oracle.ROWS, oracle.ROLES, oracle.COLS), dtype=cp.float64
    )
    # Nonuniform deterministic scale and cross-role mixing prevent a trivial
    # isotropic code path while remaining wholly source-free.
    neuron_scale = cp.linspace(0.75, 1.25, oracle.ROWS, dtype=cp.float64)[:, None, None]
    mixing = cp.asarray(
        [[1.0, 0.13, -0.07], [0.09, 0.93, 0.11], [-0.05, 0.17, 1.08]],
        dtype=cp.float64,
    )
    synthetic = cp.einsum("ab,nbd->nad", mixing, synthetic) * neuron_scale
    cp.cuda.Stream.null.synchronize()
    kernel_started = time.time()
    panel = oracle._pair_panel([synthetic], np, cp, linear_sum_assignment)
    cp.cuda.Stream.null.synchronize()
    kernel_seconds = time.time() - kernel_started
    free_after, total_after = cp.cuda.runtime.memGetInfo()
    if total_after != total_memory:
        raise RuntimeError("device memory total changed during calibration")

    reverse = tuple(reversed(range(oracle.ROWS)))
    factoradic = oracle.serialize_permutation(reverse)
    if oracle.unrank_permutation(oracle.ROWS, int.from_bytes(factoradic, "big")) != reverse:
        raise RuntimeError("full factoradic calibration failed")
    for metric in ("relaxed_reuse_exact", "legal_path_exact", "legal_path_fp16"):
        if not math.isfinite(float(panel[metric]["s_bpw"])):
            raise RuntimeError(f"nonfinite calibrated metric: {metric}")

    device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_SOURCE_FREE_FULL_GEOMETRY_RUNTIME_CALIBRATION",
        "artifact_binding": {
            "artifact_manifest_sha256": hashlib.sha256(artifact_raw).hexdigest(),
            "runner_sha256": artifact_rows["free_order_oracle_v2.py"],
            "calibration_script_sha256": artifact_rows["calibrate_runtime.py"],
        },
        "backend": {
            "python_executable_resolved": os.fspath(Path(sys.executable).resolve(strict=True)),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "cupy_version": cp.__version__,
            "scipy_version": scipy.__version__,
            "device_name": device_name,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "calibration": {
            "seed": SEED,
            "shape": [oracle.ROWS, oracle.ROLES, oracle.COLS],
            "dtype": "float64",
            "full_direct_pair_panel_executions": 1,
            "kernel_seconds": kernel_seconds,
            "elapsed_seconds": time.time() - started,
            "device_total_bytes": int(total_memory),
            "device_free_before_bytes": int(free_before),
            "device_free_after_bytes": int(free_after),
            "factoradic_bytes": len(factoradic),
            "factoradic_sha256": hashlib.sha256(factoradic).hexdigest(),
            "metrics": {
                metric: {
                    "s_bpw": float(panel[metric]["s_bpw"]),
                    "residual_ratio": float(panel[metric]["residual_ratio"]),
                }
                for metric in ("relaxed_reuse_exact", "legal_path_exact", "legal_path_fp16")
            },
        },
        "zero_access_ledger": {
            "workspace_or_source_arguments_supported": 0,
            "source_bindings_loaded": 0,
            "qwen_or_model_payload_files_opened": 0,
            "qwen_or_model_payload_bytes_read": 0,
            "pinned_panel_files_opened": 0,
            "validation_files_opened": 0,
            "external_data_fetches": 0,
            "production_result_files_opened": 0,
            "production_gpu_jobs": 0,
            "synthetic_gpu_jobs": 1,
        },
        "claim_boundary": "Runtime calibration only; no source/model evidence and no execution authority.",
    }
    result["canonical_unsigned_sha256"] = oracle.canonical_sha256(result)
    oracle._write_create_new(output, result)
    print(json.dumps({"output": os.fspath(output), "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
