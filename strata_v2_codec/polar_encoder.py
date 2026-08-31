#!/usr/bin/env python3
"""One-block CuPy POLARIS encoder for the STRATA-XKLT-SC v2 candidate.

This entry point is intentionally stateless.  Profiles, sources, and seeds are
supplied by the already sealed allocation manifest, and one invocation emits
exactly one legacy staging container for the one-shot packer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_polaris_qwen_rht_encoder as base
from bg_codec_bec_encoder import bec_flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polar-repo", type=Path, default=Path("/root/PolarLatticeQuantization"))
    ap.add_argument("--block-length", type=int, required=True)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--sigma-source", type=float, default=1.0)
    ap.add_argument("--test-distortion", type=float, required=True)
    ap.add_argument("--eta", type=float, default=0.25)
    ap.add_argument("--alphabet-size", type=int, default=64)
    ap.add_argument("--decision", choices=("map",), default="map")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--input-bf16", type=Path, required=True)
    ap.add_argument("--input-block-start", type=int, default=0)
    ap.add_argument("--canonical-source-id", required=True)
    ap.add_argument("--canonical-block-index", type=int, default=0)
    ap.add_argument("--apply-rht", action="store_true", required=True)
    ap.add_argument("--rht-seed", type=int, required=True)
    ap.add_argument("--emit-container-hex", action="store_true", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    n = args.block_length
    if n not in (1 << 20, 1 << 21):
        raise ValueError("STRATA-v2 permits only N=2^20 or N=2^21")
    if args.trials != 1 or args.input_block_start != 0 or args.canonical_block_index != 0:
        raise ValueError("production entry point encodes one staging file exactly once")
    if args.input_bf16.stat().st_size != 2 * n:
        raise ValueError("staging source size does not equal one block")
    sigma_recon = math.sqrt(args.sigma_source**2 - args.test_distortion)
    tilde_sigma = sigma_recon * math.sqrt(args.test_distortion) / args.sigma_source
    levels = int(math.log2(args.alphabet_size))
    capacities = [
        base.periodic_binary_capacity(tilde_sigma / args.eta / (1 << level0))
        for level0 in range(levels)
    ]
    flags = bec_flags(args.polar_repo, n, capacities)
    started = time.perf_counter()
    row = base.run_trial(args, 0, capacities, flags)
    elapsed = time.perf_counter() - started
    container_hex = row.pop("_container_hex")
    if container_hex is None:
        raise AssertionError("encoder did not return the required literal container")
    result = {
        "schema": "strata_xklt_sc_v2_single_block_encoder_v1",
        "architecture": "procedural-Q31-BEC POLARIS; one sealed STRATA-v2 staging block",
        "parameters": {
            "block_length": n,
            "trials": 1,
            "sigma_source": args.sigma_source,
            "test_channel_distortion": args.test_distortion,
            "eta": args.eta,
            "alphabet_size": args.alphabet_size,
            "decision": args.decision,
            "tilde_sigma": tilde_sigma,
            "capacity_schedule": capacities,
            "seed": args.seed,
        },
        "trials": [row],
        "cupy_version": cp.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "seconds": elapsed,
        "polar_construction": {
            "name": "capacity-matched BEC surrogate",
            "arithmetic": "unsigned Q31",
            "external_tables": False,
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    args.output.with_suffix(".polar.bin").write_bytes(bytes.fromhex(container_hex))
    print(payload, end="")


if __name__ == "__main__":
    main()
