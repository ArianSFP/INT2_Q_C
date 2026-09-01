#!/usr/bin/env python3
"""Run a payload-free nonlocal-state/arithmetic/lifecycle fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from label_copula_common import (
    Candidate,
    STANDALONE_REQUIRED_SAVING_BPW,
    decode_stream,
    encode_panel,
    encode_stream,
    evaluate_nested,
    factorized_bank,
    fit_model,
    matched_control_gate,
    nested_partition,
    next_state,
    pretty_json,
    synthetic_parity_streams,
)


def _trace(candidate: Candidate, symbols: tuple[int, ...]) -> int:
    state = 0
    for position, symbol in enumerate(symbols):
        state = next_state(candidate, state, symbol, 0, 0, position)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--experts", type=int, default=5)
    parser.add_argument("--blocks-per-stream", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = synthetic_parity_streams(
        layers=args.layers,
        experts=args.experts,
        blocks_per_stream=args.blocks_per_stream,
        seed=1729,
        constrained=True,
    )
    # The small default intentionally uses only two representative cells.  It
    # is an implementation fixture, not the frozen scientific 240-cell run.
    probe_bank = (
        Candidate("suffix", 64, 32),
        Candidate("parity_sketch", 64, 32),
    )
    result = evaluate_nested(source, probe_bank)

    folds = nested_partition(source)
    parity_candidate = Candidate("parity_sketch", 64, 32)
    parity_model = fit_model(folds["train"] + folds["validation"], parity_candidate)
    parity_ledger, parity_encoded = encode_panel(parity_model, folds["test"])
    roundtrips = []
    for stream, payload, meaningful in parity_encoded:
        decoded = decode_stream(parity_model, stream.roles, stream.planes, payload, meaningful)
        roundtrips.append(decoded == stream.symbols)

    # Two prefixes have the same final six symbols and therefore collide under
    # suffix-6, but differ in a body decision and remain distinct under the
    # six-sketch parity state after the body.
    prefix_a = tuple([0] * 20 + [1, 0, 1, 1, 0, 1])
    prefix_b = tuple([1] + [0] * 19 + [1, 0, 1, 1, 0, 1])
    suffix = Candidate("suffix", 64, 32)
    parity = Candidate("parity_sketch", 64, 32)
    witness = {
        "same_last_six": prefix_a[-6:] == prefix_b[-6:],
        "suffix_state_a": _trace(suffix, prefix_a),
        "suffix_state_b": _trace(suffix, prefix_b),
        "parity_state_a": _trace(parity, prefix_a),
        "parity_state_b": _trace(parity, prefix_b),
    }

    output = {
        "schema": "label-copula-source-free-fixture-v1",
        "status": "PASS_SOURCE_FREE_NONLOCAL_INTEGER_AND_ARITHMETIC_FIXTURE",
        "scientific_result": False,
        "payloads_opened": 0,
        "cupy_imported": False,
        "cuda_jobs": 0,
        "fixture_geometry": {
            "layers": args.layers,
            "experts": args.experts,
            "blocks_per_stream": args.blocks_per_stream,
            "checks_per_32_symbols": 6,
        },
        "long_memory_witness": witness,
        "all_parity_frames_roundtrip": all(roundtrips),
        "parity_test_ledger": parity_ledger,
        "nested_probe": result,
        "source_absolute_gate_bpw": STANDALONE_REQUIRED_SAVING_BPW,
        "payload_controls_would_be_allowed": matched_control_gate(result),
        "claim_boundary": "Synthetic matched-marginal implementation probe; no evidence about model weights.",
    }
    if not (
        all(roundtrips)
        and witness["same_last_six"]
        and witness["suffix_state_a"] == witness["suffix_state_b"]
        and witness["parity_state_a"] != witness["parity_state_b"]
    ):
        raise RuntimeError("source-free fixture invariant failed")

    payload = pretty_json(output)
    if args.output is None:
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    else:
        with args.output.open("xb") as stream:
            stream.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
