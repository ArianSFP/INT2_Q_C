#!/usr/bin/env python3
"""Fresh CuPy parity worker using independent CPU and GPU RM constructions."""

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
from authority_v2 import (module_origin_outside_controlled_roots, regular_bytes,
                          require, sha256)
from independent_rm_order import (TARGET_N, independent_cpu_order,
                                  independent_gpu_order, little_i64_sha256,
                                  validate_small_orders)


V1_RM_ORDER_SHA256 = "e5d85d844633d206125a775efcd35711d02bf9eec5060715c17e8e7d50df0f92"


def load_producer(path: Path):
    payload = regular_bytes(path, "snapshotted v1 rm_order.py")
    require(sha256(payload) == V1_RM_ORDER_SHA256, "v1 rm-order source pin")
    spec = importlib.util.spec_from_file_location("pinned_v1_rm_order", path)
    require(spec is not None and spec.loader is not None, "v1 module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-rm-order", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1,
            "worker requires python -I -B")
    require("PYTHONPATH" not in os.environ and "cupy" not in sys.modules and
            "numpy" not in sys.modules, "fresh array backend state")
    controlled = [PACKAGE, Path.cwd(), args.production_rm_order.parent]
    cupy_origin = module_origin_outside_controlled_roots("cupy", controlled)
    numpy_origin = module_origin_outside_controlled_roots("numpy", controlled)
    import cupy as cp
    import numpy as np
    require(Path(cp.__file__).resolve(strict=True) == cupy_origin and
            Path(np.__file__).resolve(strict=True) == numpy_origin,
            "array origins stable")
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
    probe_sum = int(cp.sum((probe * 19 + 7) % 65521, dtype=cp.int64).get())
    require(probe_sum == sum((index * 19 + 7) % 65521 for index in range(4096)),
            "synchronized GPU probe")
    producer = load_producer(args.production_rm_order.resolve(strict=True))
    validate_small_orders(np)
    rows = []
    for n in TARGET_N:
        started = time.perf_counter()
        cpu = independent_cpu_order(n, np)
        producer_cpu = producer.rm_full_order_numpy(n, np)
        gpu = independent_gpu_order(n, cp)
        cp.cuda.Stream.null.synchronize()
        gpu_host = cp.asnumpy(gpu)
        require(np.array_equal(cpu, producer_cpu),
                f"independent CPU vs producer N={n}")
        require(np.array_equal(cpu, gpu_host),
                f"independent CPU vs GPU N={n}")
        rows.append({"n": n, "independent_cpu_vs_producer": True,
                     "independent_cpu_vs_gpu": True,
                     "order_sha256": little_i64_sha256(cpu),
                     "seconds": time.perf_counter() - started})
        del cpu, producer_cpu, gpu, gpu_host
        cp.get_default_memory_pool().free_all_blocks()
    receipt = {
        "schema": "strata-rm-global-swap-v2-independent-parity-receipt",
        "producer_rm_order_sha256": V1_RM_ORDER_SHA256,
        "cpu_algorithm": "Gosper fixed-weight enumeration",
        "gpu_algorithm": "byte-LUT popcount plus GPU argsort",
        "rows": rows,
        "cupy": {"version": cp.__version__, "origin": str(cupy_origin),
                 "origin_sha256": hashlib.sha256(cupy_origin.read_bytes()).hexdigest(),
                 "device_count": count, "device_id": int(device.id),
                 "device_name": str(name), "runtime_version": runtime,
                 "driver_version": driver, "probe_sum": probe_sum},
        "numpy_origin": str(numpy_origin), "fresh_interpreter": True,
        "python_isolated_flag": True, "payloads_opened": 0,
        "rd_claim": False,
        "status": "PASS_INDEPENDENT_CPU_GPU_AND_V1_RM_ORDER_PARITY__NO_PAYLOAD",
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
