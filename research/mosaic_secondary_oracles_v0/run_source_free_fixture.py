#!/usr/bin/env python3
"""Run deterministic source-free fixtures; accepts no filesystem payload."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load(name: str, filename: str):
    payload = (ROOT / filename).read_bytes()
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError("fixture module loader")
    module = importlib.util.module_from_spec(spec)
    module.__authenticated_sha256__ = hashlib.sha256(payload).hexdigest()
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load("mosaic_secondary_fixture_contract", "gate_contract.py")
recurrence = load("mosaic_secondary_fixture_recurrence", "gf2_recurrence.py")
oracles = load("mosaic_secondary_fixture_oracles", "residual_oracles.py")


def run() -> dict:
    length = 256
    connection_a = 1 | (1 << 1) | (1 << 5)
    connection_b = 1 | (1 << 2) | (1 << 7)
    first = recurrence.generate_lfsr((1, 0, 0, 1, 1), connection_a, length)
    second = recurrence.generate_lfsr((1, 1, 0, 0, 1, 0, 1), connection_b, length)
    labels = recurrence.labels_from_gray(first, second)
    packets = []
    packet_rows = []
    for ordinal in range(256):
        shifted = labels[ordinal % length:] + labels[:ordinal % length]
        packet, row = recurrence.encode_block(shifted)
        packets.append(packet)
        packet_rows.append(row)
    blocks = tuple(recurrence.decode_block(packet) for packet in packets)
    scales = (b"\x00<",) * len(blocks)
    components = tuple(
        recurrence.encode_component(role, blocks, scales)
        for role in ("gate", "up", "down_transposed")
    )
    weights = 3 * len(packets) * length
    expert = recurrence.encode_expert(components, weights=weights)
    decoded_expert = recurrence.decode_expert(expert)
    ledger = contract.physical_expert_ledger(
        weights=weights,
        role_component_bytes=tuple(len(packet) for packet in components),
    )
    if len(expert) != ledger["physical_bytes"]:
        raise RuntimeError("literal expert packet disagrees with physical ledger")
    if decoded_expert["component_packets"] != components:
        raise RuntimeError("literal expert packet replay")
    if not ledger["passes_rate_interval"]:
        raise RuntimeError("source-free literal packet misses declared rate interval")

    basis = oracles.build_ramanujan_basis(
        np,
        length=length,
        periods=contract.NON_DYADIC_PERIODS,
        maximum_columns=64,
    )
    coordinate = np.arange(length, dtype=np.float64)
    residual = np.stack([
        np.sin((2.0 * math.pi / 7.0) * coordinate + 0.17 * block)
        + 0.08 * np.cos((4.0 * math.pi / 11.0) * coordinate - 0.09 * block)
        for block in range(16)
    ])
    source_energy = float(np.sum(residual * residual, dtype=np.float64)) / 0.04
    ramanujan = oracles.ramanujan_panel_metrics(
        np,
        residual,
        basis,
        source_energy=source_energy,
    )
    ar = oracles.ar_hankel_panel_metrics(
        np,
        residual,
        source_energy=source_energy,
        orders=(1, 2, 4),
    )
    return {
        "schema": "mosaic-secondary-oracles-source-free-fixture-v0",
        "status": "PASS_SOURCE_FREE_MECHANICS",
        "recurrence": {
            "linear_complexities": [
                recurrence.berlekamp_massey_gf2(first)[0],
                recurrence.berlekamp_massey_gf2(second)[0],
            ],
            "block_packet_bytes": len(packets[0]),
            "block_saving_bits": packet_rows[0]["saving_bits_before_outer_headers"],
            "component_bytes": [len(packet) for packet in components],
            "expert_packet_bytes": len(expert),
            "expert_packet_sha256": hashlib.sha256(expert).hexdigest(),
            "expert_canonical_reencode": recurrence.encode_expert(components, weights=weights) == expert,
            "physical_ledger": ledger,
        },
        "ramanujan": {
            "orthogonality_max_abs_error": basis["orthogonality_max_abs_error"],
            "free_prefix_remaining_fraction": ramanujan["fixed_free_prefix_remaining_sse"] / ramanujan["input_sse"],
            "ideal_remaining_fraction": ramanujan["ideal_public_basis_waterfill_remaining_sse"] / ramanujan["input_sse"],
        },
        "ar_hankel": {
            "winner_order": ar["winner"]["order"],
            "winner_relative_mse": ar["winner"]["relative_mse"],
            "pullback_charged": ar["winner"]["pullback_noise_amplification_charged"],
        },
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "cuda_initialized": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, separators=(",", ":"), allow_nan=False))
