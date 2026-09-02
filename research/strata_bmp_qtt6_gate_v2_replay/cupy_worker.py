#!/usr/bin/env python3
"""Fresh isolated CuPy worker; source-free and payload-ineligible."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from codec import Geometry
from cupy_backend import search_bmp_rank01_cupy
from search import CompleteRateCap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    geometry = Geometry(704, 2304, 1, 320, 16, 1024, 256)
    coordinate = np.arange(geometry.count, dtype=np.float64)
    source = (0.71 * np.sin(coordinate * 0.017) +
              0.23 * np.cos(coordinate * 0.071))
    levels = np.linspace(-1.5, 1.5, 64, dtype=np.float64)
    distortion = (source[:, None] - levels[None, :]) ** 2
    rate_cap = CompleteRateCap(total_weights=geometry.count, outer_bits=0)
    result = search_bmp_rank01_cupy(distortion, geometry, 0.01, rate_cap)
    winner = result["winner"]
    receipt = {
        "schema": "strata-bmp-qtt6-v2-fresh-cupy-worker-receipt",
        "nonce": args.nonce,
        "pid": os.getpid(),
        "isolated_flag": bool(sys.flags.isolated),
        "dont_write_bytecode_flag": bool(sys.dont_write_bytecode),
        "source_root": str(HERE),
        "backend_scope": result["backend_scope"],
        "runtime_identity": result["runtime_identity"],
        "workspace": result["workspace"],
        "winner": {
            "family": winner["family"],
            "requested_rank": winner["requested_rank"],
            "sse": winner["sse"],
            "physical_bits": winner["physical_bits"],
            "objective": winner["objective"],
            "packet_sha256": hashlib.sha256(winner["packet"]).hexdigest(),
        },
        "candidate_count": len(result["candidates"]),
        "held_families": result["held_families"],
        "model_or_qwen_payload_opened_statted_hashed_or_enumerated": False,
        "payload_authority": False,
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
