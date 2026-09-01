#!/usr/bin/env python3
"""Source-free long-memory fixture for the unifilar census.

The synthetic bit law depends on cumulative prefix parity since a 4096-symbol
reset.  A two-state XOR unifilar model observes that state exactly.  A bounded
suffix model sees only the last one to six bits.  No model or codec payload is
read; all bytes are generated deterministically in memory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _common() -> Any:
    path = Path(__file__).absolute().with_name("uwfa_common.py")
    spec = importlib.util.spec_from_file_location("uwfa_fixture_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load same-directory CPU reference")
    module = importlib.util.module_from_spec(spec)
    sys.modules["uwfa_fixture_common"] = module
    spec.loader.exec_module(module)
    return module


def _xorshift32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= (value << 13) & 0xFFFFFFFF
    value ^= value >> 17
    value ^= (value << 5) & 0xFFFFFFFF
    return value & 0xFFFFFFFF


def generate_stream(seed: int, length: int = 4096) -> tuple[list[int], list[int], list[int]]:
    rng = seed & 0xFFFFFFFF or 1
    parity = 0
    bits: list[int] = []
    levels: list[int] = []
    base: list[int] = []
    for position in range(length):
        rng = _xorshift32(rng)
        # p(bit=1 | prefix parity) is 0.35 or 0.65.  The modest contrast
        # prevents a short suffix from trivially filtering the old parity.
        threshold = 22938 if parity == 0 else 42598
        bit = 1 if (rng & 0xFFFF) < threshold else 0
        bits.append(bit)
        levels.append(2)
        base.append(32768)
        parity ^= bit
    return bits, levels, base


def run_fixture() -> dict[str, Any]:
    common = _common()
    training = [generate_stream(0xC001D00D ^ (index * 0x9E3779B1)) for index in range(48)]
    testing = [generate_stream(0x51A7E000 ^ (index * 0x85EBCA6B)) for index in range(24)]

    parity_candidate = common.Candidate("xor_sketch", 2, 4096)
    parity_counts = common.zero_counts(parity_candidate)
    for bits, levels, base in training:
        common.count_stream_cpu(bits, levels, base, parity_candidate, parity_counts)
    parity_frequencies = common.q16_frequencies_from_counts(parity_counts)
    parity_bits = sum(
        common.exact_stream_length_cpu(bits, levels, base, parity_candidate, parity_frequencies)
        for bits, levels, base in testing
    )

    suffix_rows = []
    for states in common.STATE_SIZES:
        candidate = common.Candidate("suffix", states, 4096)
        counts = common.zero_counts(candidate)
        for bits, levels, base in training:
            common.count_stream_cpu(bits, levels, base, candidate, counts)
        frequencies = common.q16_frequencies_from_counts(counts)
        logical_bits = sum(
            common.exact_stream_length_cpu(bits, levels, base, candidate, frequencies)
            for bits, levels, base in testing
        )
        suffix_rows.append({"states": states, "suffix_depth": states.bit_length() - 1, "logical_bits": logical_bits})

    symbols = sum(len(bits) for bits, _, _ in testing)
    best_suffix = min(suffix_rows, key=lambda row: (row["logical_bits"], row["states"]))
    separation = (best_suffix["logical_bits"] - parity_bits) / symbols
    common.require(separation > 0.01, "fixture failed to separate cumulative parity from bounded suffix")
    return {
        "schema": "unifilar-wfa-source-free-long-memory-fixture-v2",
        "status": "PASS_LONG_MEMORY_SEPARATION",
        "law": "p(bit=1|cumulative prefix parity) in {0.35,0.65}; reset=4096",
        "training_streams": len(training),
        "testing_streams": len(testing),
        "symbols_scored": symbols,
        "xor_parity_cell": {**parity_candidate.as_dict(), "logical_bits": parity_bits},
        "suffix_cells": suffix_rows,
        "best_bounded_suffix": best_suffix,
        "exact_arithmetic_separation_bits_per_symbol": separation,
        "claim": "separates this frozen synthetic law only; it is not evidence about Qwen or arbitrary MPS",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_fixture(), indent=2, sort_keys=True, allow_nan=False))




