#!/usr/bin/env python3
"""Exact source-free byte/read ledgers for finite SILWARP successors.

This module deliberately has no tensor-loading code.  It turns a system rate
cap into integer per-block capacities and reports how much denoising a finite
base channel would need at the resulting *total* physical rate.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path


WEIGHTS_PER_N19_BLOCK = 1 << 19
BLOCKS_PER_EXPERT = 9
WEIGHTS_PER_EXPERT = BLOCKS_PER_EXPERT * WEIGHTS_PER_N19_BLOCK
PRODUCTION_EXPERTS = 128

# Frozen SILWARP-v2 FP16 model, including its 4 KiB header.
SILWARP_FP16_BYTES = 475_654
# A deterministic symmetric per-output-channel INT8 successor: all six weight
# matrices in int8, 1,280 binary16 scales, all 1,027 biases/gains in binary16,
# and the same 4 KiB header.  It is a proposal, not a measured survivor.
SILWARP_INT8_BYTES = 234_752 + 2 * 1_280 + 2 * 1_027 + 4_096

# Nine N19 streams need nine (u32 arithmetic bits, u16 escape bytes) records,
# moments, magic/version/flags and a checksum.  128 bytes is deliberately
# conservative and leaves unused reserved bytes.
EXPERT_HEADER_BYTES = 128

REPRESENTED_IDEAL_MSE = 0.05076578709329227
IDEAL_GATE_REQUIRED_S = 0.1673974074587855
PUBLISHED_BASE_2P5_MSE = 0.030902167403153148
PUBLISHED_BASE_2P15_MSE = 0.04985939119332436

RATE_FRACTIONS = {
    "2.15": Fraction(43, 20),
    "2.30": Fraction(23, 10),
    "2.50": Fraction(5, 2),
}


def shaping_gap_bpw(normalized_second_moment: float) -> float:
    return 0.5 * math.log2(2.0 * math.pi * math.e * normalized_second_moment)


SHAPING_GAPS = {
    "ideal_sphere": 0.0,
    "polar_n19_0p2dB_bound": 0.5 * math.log2(10.0 ** (0.2 / 10.0)),
    "leech_G_0p065771": shaping_gap_bpw(0.065771),
    "e8_G_0p071682": shaping_gap_bpw(0.071682),
    "scalar_z_G_1over12": shaping_gap_bpw(1.0 / 12.0),
}


def exact_budget(
    rate: Fraction,
    *,
    experts: int,
    global_bytes: int,
    expert_header_bytes: int = EXPERT_HEADER_BYTES,
) -> dict[str, int | float]:
    """Allocate equal byte capacities to nine expert-private N19 streams."""

    cap_num = rate.numerator * WEIGHTS_PER_EXPERT * experts
    cap_den = rate.denominator * 8
    cap_bytes = cap_num // cap_den
    usable = cap_bytes - global_bytes - experts * expert_header_bytes
    if usable < 0:
        raise ValueError("global/header bytes exceed the total cap")
    block_bytes = usable // (experts * BLOCKS_PER_EXPERT)
    local_bytes = expert_header_bytes + BLOCKS_PER_EXPERT * block_bytes
    total_bytes = global_bytes + experts * local_bytes
    physical_bpw = 8.0 * total_bytes / (experts * WEIGHTS_PER_EXPERT)
    equal_share = total_bytes / experts
    cold_bytes = local_bytes + global_bytes
    cold_amp = cold_bytes / equal_share
    payload_bpw = 8.0 * block_bytes / WEIGHTS_PER_N19_BLOCK
    target = 0.8 * 2.0 ** (-2.0 * physical_bpw)
    return {
        "requested_cap_bpw": float(rate),
        "experts": experts,
        "global_bytes": global_bytes,
        "expert_header_bytes": expert_header_bytes,
        "n19_streams_per_expert": BLOCKS_PER_EXPERT,
        "block_capacity_bytes": block_bytes,
        "expert_local_bytes": local_bytes,
        "total_bytes": total_bytes,
        "cap_bytes": cap_bytes,
        "unused_cap_bytes": cap_bytes - total_bytes,
        "payload_bpw": payload_bpw,
        "physical_bpw": physical_bpw,
        "target_mse_same_rate": target,
        "cold_bytes_per_expert": cold_bytes,
        "cold_read_amplification": cold_amp,
    }


def correction_requirement(identity_mse: float, target_mse: float) -> dict[str, float]:
    ratio = target_mse / identity_mse
    return {
        "identity_mse": identity_mse,
        "required_after_over_identity": ratio,
        "required_fractional_mse_reduction": 1.0 - ratio,
        "required_s_bpw": -0.5 * math.log2(ratio),
    }


def direct_polar_projection(row: dict[str, int | float]) -> dict[str, float]:
    """Transparent slope-only projection, never represented as measurement."""

    cap = float(row["requested_cap_bpw"])
    payload = float(row["payload_bpw"])
    if abs(cap - 2.5) < 1e-9:
        reference_mse = PUBLISHED_BASE_2P5_MSE
        reference_rate = 2.5
    elif abs(cap - 2.15) < 1e-9:
        reference_mse = PUBLISHED_BASE_2P15_MSE
        reference_rate = 2.15
    else:
        # Interpolate log distortion only to make the 2.30 planning row useful.
        t = (cap - 2.15) / 0.35
        reference_mse = math.exp(
            (1.0 - t) * math.log(PUBLISHED_BASE_2P15_MSE)
            + t * math.log(PUBLISHED_BASE_2P5_MSE)
        )
        reference_rate = cap
    projected = reference_mse * 2.0 ** (2.0 * (reference_rate - payload))
    out = correction_requirement(projected, float(row["target_mse_same_rate"]))
    out.update(
        {
            "status": "SLOPE_ONLY_PROJECTION_NOT_MEASUREMENT",
            "reference_rate_bpw": reference_rate,
            "reference_mse": reference_mse,
            "base_payload_rate_bpw": payload,
            "rate_removed_for_global_and_headers_bpw": reference_rate - payload,
            "surplus_required_over_ideal_gate_s_bpw": out["required_s_bpw"]
            - IDEAL_GATE_REQUIRED_S,
        }
    )
    return out


def dither_channel_requirements(row: dict[str, int | float]) -> dict[str, dict[str, float]]:
    payload = float(row["payload_bpw"])
    target = float(row["target_mse_same_rate"])
    answer: dict[str, dict[str, float]] = {}
    for name, gap in SHAPING_GAPS.items():
        simulated_mutual_information = max(0.0, payload - gap)
        identity_mse = 2.0 ** (-2.0 * simulated_mutual_information)
        requirement = correction_requirement(identity_mse, target)
        requirement.update(
            {
                "shaping_or_finite_gap_bpw": gap,
                "max_simulated_gaussian_I_bpw": simulated_mutual_information,
                "surplus_required_over_ideal_gate_s_bpw": requirement["required_s_bpw"]
                - IDEAL_GATE_REQUIRED_S,
            }
        )
        answer[name] = requirement
    return answer


def build_report(entropy_phase_bins: int = 32) -> dict[str, object]:
    # A stored u16 binary-tree frequency table has 64 internal nodes for the
    # 65-symbol alphabet.  A 256-byte header binds its construction.
    entropy_table_bytes = entropy_phase_bins * 64 * 2 + 256
    variants = {
        "fp16_direct_polar": (SILWARP_FP16_BYTES, "direct_polar"),
        "fp16_rdq_p32": (SILWARP_FP16_BYTES + entropy_table_bytes, "rotated_dither"),
        "int8_direct_polar_proposal": (SILWARP_INT8_BYTES, "direct_polar"),
        "int8_rdq_p32_proposal": (
            SILWARP_INT8_BYTES + entropy_table_bytes,
            "rotated_dither",
        ),
    }
    systems: dict[str, object] = {}
    for model_name, (global_bytes, channel_kind) in variants.items():
        for experts in (6, PRODUCTION_EXPERTS):
            key = f"{model_name}_{experts}_experts"
            rate_rows: dict[str, object] = {}
            for rate_name, rate in RATE_FRACTIONS.items():
                row = exact_budget(rate, experts=experts, global_bytes=global_bytes)
                row["channel_kind"] = channel_kind
                if channel_kind == "direct_polar":
                    row["direct_polar_projection"] = direct_polar_projection(row)
                else:
                    row["dither_channel_requirements"] = dither_channel_requirements(row)
                rate_rows[rate_name] = row
            systems[key] = rate_rows
    return {
        "schema": "silwarp_finite_bridge_source_free_ledger_v1",
        "claim_boundary": (
            "Exact integer byte/read arithmetic plus explicitly labelled RD projections; "
            "contains no Qwen payload measurement and proves no finite survivor."
        ),
        "constants": {
            "weights_per_expert": WEIGHTS_PER_EXPERT,
            "n19_blocks_per_expert": BLOCKS_PER_EXPERT,
            "fp16_decoder_bytes": SILWARP_FP16_BYTES,
            "int8_decoder_proposal_bytes": SILWARP_INT8_BYTES,
            "entropy_phase_bins": entropy_phase_bins,
            "entropy_table_and_header_bytes": entropy_table_bytes,
            "expert_header_bytes": EXPERT_HEADER_BYTES,
            "ideal_gate_required_s_bpw": IDEAL_GATE_REQUIRED_S,
            "represented_ideal_mse": REPRESENTED_IDEAL_MSE,
            "shaping_gaps_bpw": SHAPING_GAPS,
        },
        "systems": systems,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy-phase-bins", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.entropy_phase_bins <= 0:
        raise SystemExit("entropy phase bins must be positive")
    report = build_report(args.entropy_phase_bins)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
