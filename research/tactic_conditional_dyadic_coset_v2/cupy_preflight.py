#!/usr/bin/env python3
"""Synthetic-only CuPy parity preflight for TACTIC-DH384 v2.

There is intentionally no model, coarse-lock, root, manifest, URL, or resume
argument.  Running this file still requires a post-review external token.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from tactic_v2_common import (
    SYNTHETIC_AUTHORIZATION,
    HeldOutput,
    canonical_json,
    cpu_projection,
    require,
    selector_packet,
    sha256_bytes,
    splitmix64,
    universal_selector_table,
)


AUTHORIZATION = SYNTHETIC_AUTHORIZATION


def fixture() -> tuple[list[int], list[float]]:
    state = 0xD484333854414354
    symbols: list[int] = []
    errors: list[float] = []
    for index in range(4096):
        state, word = splitmix64(state)
        symbols.append(int((word & 0xFFF) - 2048))
        state, word = splitmix64(state)
        signed = int((word & 0xFFFFFF) - (1 << 23))
        errors.append((signed + ((index * 17) & 255)) / float(1 << 23))
    return symbols, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.authorization != AUTHORIZATION:
        raise SystemExit("synthetic authorization mismatch; CUDA not imported")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be exactly 0; CUDA not imported")
    if not args.output.is_absolute() or os.path.lexists(args.output):
        raise SystemExit("output must be an absent absolute path; CUDA not imported")

    from verify_source import verify_package
    source_receipt = verify_package(Path(__file__).resolve().parent)

    started = time.monotonic()
    import cupy as cp
    import numpy as np
    from stage0_gate import gpu_projection

    symbols, errors = fixture()
    table = universal_selector_table()
    cpu = cpu_projection(symbols, errors, role=2, table=table, rank=384)
    gpu_energy, gpu_projected = gpu_projection(
        cp,
        np.asarray(symbols, dtype=np.int64),
        np.asarray(errors, dtype=np.float64),
        role=2,
        table=table,
    )
    require(math.isclose(gpu_energy, cpu["energy"], rel_tol=2e-13, abs_tol=2e-11),
            "CPU/CuPy error-energy parity")
    require(math.isclose(gpu_projected, cpu["projected_energy"], rel_tol=2e-12, abs_tol=2e-11),
            "CPU/CuPy projection parity")

    packet = selector_packet(table)
    require(len(packet) == 16_384, "selector packet bytes")
    total_bytes = 24_576 + 6 * (512 + 1_152 * 48 + 18 * 78_592)
    require(total_bytes == 8_847_360, "physical byte ledger")
    physical_bpw = 8.0 * total_bytes / 28_311_552
    cold = (6 + 359) / 360
    require(physical_bpw == 2.5 and cold == 73 / 72, "rate/read ledger")

    receipt = {
        "schema": "tactic_dh384_synthetic_cupy_preflight_v2",
        "status": "PASS_SYNTHETIC_ONLY_NO_PAYLOAD",
        "claim_boundary": "Synthetic dyadic parity and accounting only; no model/coarse payload was accepted or opened.",
        "source_manifest_sha256": source_receipt["manifest_sha256"],
        "fixture": {
            "values": 4096,
            "role": 2,
            "universal_selector_ordinal": 17,
            "selector_searches": 0,
            "cpu_energy": cpu["energy"],
            "cpu_projected_energy": cpu["projected_energy"],
            "gpu_energy": gpu_energy,
            "gpu_projected_energy": gpu_projected,
            "selector_packet_sha256": sha256_bytes(packet),
        },
        "ledger": {
            "container_bytes": total_bytes,
            "physical_bpw": physical_bpw,
            "cold_read_amplification": cold,
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": sys.version,
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "device": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
        },
        "access": {
            "payload_arguments": 0,
            "payload_files_opened": 0,
            "network_calls": 0,
        },
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))
    with HeldOutput(args.output) as output:
        output.write_new("selector_packet.bin", packet)
        output.write_new(
            "receipt.json",
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n",
        )
    print(json.dumps({"status": receipt["status"], "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
