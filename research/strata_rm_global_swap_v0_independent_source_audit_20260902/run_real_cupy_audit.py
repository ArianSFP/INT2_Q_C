#!/usr/bin/env python3
"""Fresh-process, source-free CuPy parity check at N=2**20 and N=2**21."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_order(n: int) -> np.ndarray:
    phases = np.arange(n, dtype=np.uint32)
    octets = phases.view(np.uint8).reshape(n, phases.dtype.itemsize)
    popcount = np.unpackbits(octets, axis=1).sum(axis=1, dtype=np.uint16)
    return np.lexsort((phases.astype(np.int64), -popcount.astype(np.int16)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(audit_dir))
    from independent_auth import authenticate_external, authenticate_source
    source_auth = authenticate_source(args.source)
    external_auth = authenticate_external(args.external_root)
    sys.path.insert(0, str(args.source.resolve()))
    from rm_order import TARGET_N, rm_full_order_cupy

    spec = importlib.util.find_spec("cupy")
    if spec is None or spec.origin is None:
        raise ValueError("CuPy import origin")
    origin = Path(spec.origin).resolve()
    forbidden_roots = (audit_dir, args.source.resolve(), args.external_root.resolve())
    if any(origin == root or root in origin.parents for root in forbidden_roots):
        raise ValueError("CuPy resolves inside an experiment-controlled source root")
    import cupy as cp
    if Path(cp.__file__).resolve() != origin:
        raise ValueError("CuPy import origin changed")
    device_count = int(cp.cuda.runtime.getDeviceCount())
    if device_count < 1:
        raise ValueError("no CUDA device")
    runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
    driver_version = int(cp.cuda.runtime.driverGetVersion())
    if runtime_version <= 0 or driver_version <= 0:
        raise ValueError("CUDA runtime/driver provenance")
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")

    rows = []
    for n in TARGET_N:
        expected = independent_order(n)
        started = time.perf_counter()
        actual_gpu = rm_full_order_cupy(n)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - started
        actual = cp.asnumpy(actual_gpu)
        if not np.array_equal(actual, expected):
            raise AssertionError(f"CuPy/independent order mismatch at N={n}")
        rows.append({
            "n": n,
            "elapsed_seconds": elapsed,
            "order_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "exact_full_order_match": True,
            "unique_phases": int(np.unique(actual).size),
        })
        del actual_gpu, actual, expected
        cp.get_default_memory_pool().free_all_blocks()

    receipt = {
        "schema": "strata-rm-global-swap-v0-independent-real-cupy-audit",
        "source_auth": source_auth,
        "external_auth": external_auth,
        "cupy": {
            "version": cp.__version__,
            "origin": str(origin),
            "origin_sha256": sha256_file(origin),
            "runtime_version": runtime_version,
            "driver_version": driver_version,
            "device_count": device_count,
            "device_id": device.id,
            "device_name": device_name,
        },
        "rows": rows,
        "payloads_opened": 0,
        "rd_claim": False,
        "status": "PASS_REAL_CUPY_FULL_ORDER_PARITY__NO_RD__HOLD_PAYLOAD",
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
