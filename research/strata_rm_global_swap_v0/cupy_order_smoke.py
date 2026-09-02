#!/usr/bin/env python3
"""Source-free CuPy construction smoke at the intended global lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cupy as cp

from rm_order import TARGET_N, rm_full_order_cupy


CAPACITY_FIXTURE = [
    0.0008227374118798814,
    0.237747929331251,
    0.9153259168218427,
    0.9999815811734327,
    1.0,
    1.0,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    rows = []
    for n in TARGET_N:
        cp.get_default_memory_pool().free_all_blocks()
        start = time.perf_counter()
        order = rm_full_order_cupy(n)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        phases = cp.asnumpy(order)
        prefix = phases[: min(n, 65536)]
        popcount = [int(value).bit_count() for value in prefix]
        prefix_tie_ok = all(
            popcount[i - 1] > popcount[i] or
            (popcount[i - 1] == popcount[i] and int(prefix[i - 1]) < int(prefix[i]))
            for i in range(1, len(popcount))
        )
        ks = [min(n, max(0, int(math.ceil(n * capacity))))
              for capacity in CAPACITY_FIXTURE]
        flags = []
        for k in ks:
            flag = cp.ones(n, dtype=cp.uint8)
            flag[order[:k]] = cp.uint8(0)
            selected = int(cp.count_nonzero(flag == 0).get())
            if selected != k:
                raise AssertionError("CuPy selected-count equality")
            flags.append({
                "k": k,
                "selected": selected,
                "flag_sha256": hashlib.sha256(cp.asnumpy(flag).tobytes()).hexdigest(),
            })
        rows.append({
            "n": n,
            "elapsed_seconds": elapsed,
            "order_sha256": hashlib.sha256(phases.tobytes()).hexdigest(),
            "prefix_tie_order_exact": prefix_tie_ok,
            "levels": flags,
        })
        del order
        cp.get_default_memory_pool().free_all_blocks()

    result = {
        "schema": "strata-rm-global-swap-v0-cupy-source-free-smoke",
        "backend": {
            "cupy_version": cp.__version__,
            "device_id": device.id,
            "device_name": name,
        },
        "rows": rows,
        "payloads_opened": 0,
        "claim_boundary": "ordering construction and K equality only; no RD evidence",
        "status": "PASS_CUPY_GLOBAL_CONSTRUCTION__HOLD_PAYLOAD",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

