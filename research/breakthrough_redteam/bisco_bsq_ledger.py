#!/usr/bin/env python3
"""Recompute the physical-rate and cold-read ledgers in the BiSCo assessment."""

from __future__ import annotations

import json
import math


WEIGHTS_PER_EXPERT = 4_718_592
TARGET_S = -0.5 * math.log2(0.8)
HEADER_BYTES = 256
LOCAL_SCALE_BYTES = 12  # FP16 mean and RMS for each of three matrices.


def decoder_parameters(d: int, hidden: int, total_bits: int) -> int:
    """Three roles times two one-hidden-layer stage decoders, biases included."""
    return 3 * hidden * (total_bits + 2 * d + 2) + 6 * d


def production_row(d: int, experts: int = 128) -> dict[str, float | int]:
    hidden = 4 * d
    total_bits = 9 * d // 4
    assert total_bits * 4 == 9 * d
    params = decoder_parameters(d, hidden, total_bits)
    decoder_bytes = 2 * params
    global_bytes = decoder_bytes + HEADER_BYTES
    code_rate = total_bits / d
    code_bytes = total_bits * (WEIGHTS_PER_EXPERT // d) // 8
    attributed_bytes = code_bytes + global_bytes / experts + LOCAL_SCALE_BYTES
    cold_bytes = code_bytes + global_bytes + LOCAL_SCALE_BYTES
    physical_rate = 8 * attributed_bytes / WEIGHTS_PER_EXPERT
    side_rate = physical_rate - code_rate
    return {
        "d": d,
        "hidden": hidden,
        "b1": total_bits // 2,
        "b2": total_bits - total_bits // 2,
        "decoder_parameters": params,
        "decoder_bytes": decoder_bytes,
        "header_bytes": HEADER_BYTES,
        "local_scale_bytes": LOCAL_SCALE_BYTES,
        "experts_amortized": experts,
        "expert_code_bytes": code_bytes,
        "attributed_physical_bytes_per_expert": attributed_bytes,
        "cold_bytes_per_expert": cold_bytes,
        "physical_bpw": physical_rate,
        "cold_read_amplification": cold_bytes / attributed_bytes,
        "side_bpw": side_rate,
        "minimum_matched_s_if_gaussian_code_is_ideal": TARGET_S + side_rate,
        "target_relative_mse": 0.8 * 2 ** (-2 * physical_rate),
    }


def panel_row(d: int, hidden: int, total_bits: int, experts: int = 6) -> dict[str, float | int]:
    params = decoder_parameters(d, hidden, total_bits)
    decoder_bytes = 2 * params
    panel_scale_bytes = LOCAL_SCALE_BYTES * experts
    global_bytes = decoder_bytes + HEADER_BYTES
    code_rate = total_bits / d
    code_bytes = total_bits * (WEIGHTS_PER_EXPERT // d) // 8
    attributed_bytes = code_bytes + global_bytes / experts + LOCAL_SCALE_BYTES
    cold_bytes = code_bytes + global_bytes + LOCAL_SCALE_BYTES
    physical_rate = 8 * attributed_bytes / WEIGHTS_PER_EXPERT
    side_rate = physical_rate - code_rate
    return {
        "d": d,
        "hidden": hidden,
        "b1": (total_bits + 1) // 2,
        "b2": total_bits // 2,
        "decoder_parameters": params,
        "decoder_bytes": decoder_bytes,
        "header_bytes": HEADER_BYTES,
        "panel_scale_bytes": panel_scale_bytes,
        "experts_amortized": experts,
        "expert_code_bytes": code_bytes,
        "attributed_physical_bytes_per_expert": attributed_bytes,
        "cold_bytes_per_expert": cold_bytes,
        "physical_bpw": physical_rate,
        "cold_read_amplification": cold_bytes / attributed_bytes,
        "side_bpw": side_rate,
        "minimum_matched_s_if_gaussian_code_is_ideal": TARGET_S + side_rate,
        "target_relative_mse": 0.8 * 2 ** (-2 * physical_rate),
    }


def main() -> None:
    production = [production_row(d) for d in (16, 32, 64)]
    panel = [
        panel_row(16, 64, 36),
        panel_row(32, 128, 71),
        panel_row(64, 256, 137),
    ]
    for row in production + panel:
        assert 2.15 <= row["physical_bpw"] <= 2.5
        assert row["cold_read_amplification"] < 2.0
    print(
        json.dumps(
            {
                "target_s": TARGET_S,
                "weights_per_expert": WEIGHTS_PER_EXPERT,
                "production_128_expert_amortization": production,
                "self_contained_six_expert_panel": panel,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
