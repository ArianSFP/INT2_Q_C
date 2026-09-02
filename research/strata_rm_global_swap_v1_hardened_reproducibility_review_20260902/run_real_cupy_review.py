#!/usr/bin/env python3
"""Optional trusted-runner accelerator provenance review.

This opens no model or packet payload.  It launches the producer's exact
snapshot worker and independently checks the two reported full-order hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


REVIEW = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW))
from independent_auth import (PRODUCER_MANIFEST_SHA256,
                              authenticate_external_sources,
                              authenticate_producer)


def independent_order_hash(n: int) -> str:
    phases = np.arange(n, dtype=np.uint32)
    octets = phases.view(np.uint8).reshape(n, phases.dtype.itemsize)
    popcount = np.unpackbits(octets, axis=1).sum(axis=1, dtype=np.uint16)
    order = np.lexsort((phases.astype(np.int64),
                        -popcount.astype(np.int16))).astype("<i8", copy=False)
    return hashlib.sha256(order.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    producer_auth = authenticate_producer(args.producer)
    external_auth = authenticate_external_sources(args.external_root)
    sys.path.insert(0, str(args.producer.resolve()))
    from authority import run_isolated_worker
    worker = run_isolated_worker(
        args.producer, expected_manifest_sha256=PRODUCER_MANIFEST_SHA256,
        worker_name="real_cupy_worker.py", external_root=args.external_root)
    if worker.get("schema") != "strata-rm-global-swap-v1-real-cupy-receipt":
        raise ValueError("real-CuPy receipt schema")
    expected_lengths = (1 << 20, 1 << 21)
    rows = worker.get("rows")
    if not isinstance(rows, list) or [row.get("n") for row in rows] != \
            list(expected_lengths):
        raise ValueError("real-CuPy production lengths")
    independent = []
    for row, n in zip(rows, expected_lengths, strict=True):
        expected = independent_order_hash(n)
        if row.get("exact_full_order_match") is not True or \
                row.get("order_sha256") != expected:
            raise ValueError(f"independent full-order hash N={n}")
        independent.append({"n": n, "independent_order_sha256": expected,
                            "matches_worker": True})
    cupy = worker.get("cupy")
    if not isinstance(cupy, dict) or cupy.get("device_count", 0) < 1 or \
            cupy.get("runtime_version", 0) <= 0 or \
            cupy.get("driver_version", 0) <= 0:
        raise ValueError("live CUDA provenance fields")
    receipt = {
        "schema": "strata-rm-global-swap-v1-independent-real-cupy-review",
        "producer_auth": producer_auth, "external_auth": external_auth,
        "producer_worker_receipt": worker,
        "independent_full_order_hashes": independent,
        "payloads_opened": 0, "model_data_accessed": False,
        "network_accessed": False, "rd_claim": False,
        "status": "PASS_TRUSTED_RUNNER_REAL_CUPY_AND_INDEPENDENT_ORDER_HASHES__NO_RD",
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True,
                         ensure_ascii=True, allow_nan=False) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
