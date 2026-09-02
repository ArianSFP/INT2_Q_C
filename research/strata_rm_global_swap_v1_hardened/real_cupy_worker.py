#!/usr/bin/env python3
"""Fresh isolated real-CuPy parity worker for both production lengths."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from authority import (EXTERNAL_PINS, authenticate_current_external_root,
                       module_origin_outside_controlled_roots, require)
from rm_order import TARGET_N, rm_full_order_cupy, rm_full_order_numpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1,
            "worker requires python -I -B")
    require("PYTHONPATH" not in os.environ, "PYTHONPATH inherited")
    require("cupy" not in sys.modules and "numpy" not in sys.modules,
            "array backend was preloaded")
    external = authenticate_current_external_root(args.external_root)
    controlled = [PACKAGE, Path(external["external_root"]), Path.cwd()]
    cupy_origin = module_origin_outside_controlled_roots("cupy", controlled)
    numpy_origin = module_origin_outside_controlled_roots("numpy", controlled)
    import cupy as cp
    import numpy as np
    require(Path(cp.__file__).resolve(strict=True) == cupy_origin and
            Path(np.__file__).resolve(strict=True) == numpy_origin,
            "array module origin changed")
    count = int(cp.cuda.runtime.getDeviceCount())
    runtime = int(cp.cuda.runtime.runtimeGetVersion())
    driver = int(cp.cuda.runtime.driverGetVersion())
    require(count >= 1 and runtime > 0 and driver > 0, "live CUDA runtime")
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    probe = cp.arange(4096, dtype=cp.int64)
    probe = (probe * cp.int64(17) + cp.int64(3)) % cp.int64(65521)
    observed_probe = int(cp.sum(probe, dtype=cp.int64).get())
    expected_probe = sum((index * 17 + 3) % 65521 for index in range(4096))
    require(observed_probe == expected_probe, "synchronized GPU probe")
    rows = []
    for n in TARGET_N:
        expected = rm_full_order_numpy(n, np)
        started = time.perf_counter()
        gpu = rm_full_order_cupy(n, cp)
        cp.cuda.Stream.null.synchronize()
        actual = cp.asnumpy(gpu)
        require(np.array_equal(actual, expected), f"full RM order parity N={n}")
        rows.append({
            "n": n, "exact_full_order_match": True,
            "order_sha256": hashlib.sha256(actual.astype("<i8", copy=False).tobytes()).hexdigest(),
            "seconds": time.perf_counter() - started,
        })
        del expected, actual, gpu
        cp.get_default_memory_pool().free_all_blocks()
    record = {
        "schema": "strata-rm-global-swap-v1-real-cupy-receipt",
        "external_pins": EXTERNAL_PINS,
        "cupy": {"version": cp.__version__, "origin": str(cupy_origin),
                 "origin_sha256": hashlib.sha256(cupy_origin.read_bytes()).hexdigest(),
                 "device_count": count, "device_id": int(device.id),
                 "device_name": str(name), "runtime_version": runtime,
                 "driver_version": driver, "probe_sum": observed_probe},
        "numpy_origin": str(numpy_origin), "rows": rows,
        "fresh_interpreter": True, "python_isolated_flag": True,
        "pythonpath_inherited": False, "sys_modules_backend_preload_rejected": True,
        "payloads_opened": 0, "rd_claim": False,
        "status": "PASS_REAL_CUPY_FULL_ORDER_PARITY__NO_RD__HOLD_PAYLOAD",
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
