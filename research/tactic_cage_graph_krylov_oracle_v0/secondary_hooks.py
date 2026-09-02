#!/usr/bin/env python3
"""Small frozen routing/gate hooks for decoder-legal secondary screens."""

from __future__ import annotations

import math
from typing import Any, Mapping


SCREENS = (
    "ramanujan_non_dyadic_phase",
    "coarse_signature_collaborative_patches",
    "coarse_seriated_hankel_displacement_rank",
)
MIN_CONTROL_EXCESS_BPW = 0.03
TARGET_RELATIVE_MSE = 0.025
BLOCK_BITS = 384


class HookError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HookError(message)


def routing_record(graph_status: str) -> dict[str, Any]:
    require(isinstance(graph_status, str) and graph_status, "graph status")
    return {
        "schema": "tactic-cage-secondary-screen-routing-v0",
        "upstream_graph_status": graph_status,
        "execution_order": list(SCREENS),
        "branches_are_independent": True,
        "compressed_frame_refetch_allowed": False,
        "reuse_buffered_decoded_coarse_state": True,
        "reuse_buffered_fp64_residual": True,
        "controls_open_separately_after_each_source_survival": True,
        "gains_may_be_added_without_nested_reconstruction": False,
        "bispectral_volterra": {
            "status": "DEFERRED_CONDITIONAL_GATE",
            "requires_ramanujan_control_excess_bpw": 0.03,
            "requires_heldout_remaining_sse_prediction_fraction": 0.05,
            "maximum_public_lag_pairs": 8,
            "maximum_dyadic_coefficients": 32,
        },
    }

def containment_gate(
    *,
    input_sse: float,
    source_energy: float,
    source_remaining_sse: float,
    control_remaining_sse: Mapping[str, float] | None,
    descriptor_bits_per_block: int,
) -> dict[str, Any]:
    values = (input_sse, source_energy, source_remaining_sse)
    require(all(math.isfinite(value) and value >= 0.0 for value in values) and
            source_energy > 0.0 and input_sse > 0.0 and
            source_remaining_sse <= input_sse,
            "finite containment values")
    require(type(descriptor_bits_per_block) is int and
            0 <= descriptor_bits_per_block <= BLOCK_BITS,
            "descriptor bit budget")
    relative_mse = source_remaining_sse / source_energy
    source_survives = relative_mse <= TARGET_RELATIVE_MSE + 1e-12
    if not source_survives:
        return {
            "status": "HARD_KILL_SOURCE_CONTAINMENT_MISSES_TARGET",
            "controls_may_open": False,
            "final_relative_mse": relative_mse,
            "remaining_refinement_bits_per_block":
                BLOCK_BITS - descriptor_bits_per_block,
        }
    require(control_remaining_sse is not None and
            set(control_remaining_sse) == {"permutation", "gaussian"},
            "both controls after source survival")
    source_gain = -0.5 * math.log2(source_remaining_sse / input_sse)
    control_gains = {
        name: -0.5 * math.log2(float(remaining) / input_sse)
        for name, remaining in control_remaining_sse.items()
    }
    excess = source_gain - max(control_gains.values())
    status = (
        "ELIGIBLE_FOR_BOUNDED_FINITE_SCREEN_BUILD"
        if excess + 1e-15 >= MIN_CONTROL_EXCESS_BPW and
        descriptor_bits_per_block < BLOCK_BITS
        else "HARD_KILL_CONTROL_EXCESS_OR_DESCRIPTOR_BUDGET"
    )
    return {
        "status": status,
        "controls_may_open": True,
        "final_relative_mse": relative_mse,
        "source_rate_gain_bpw": source_gain,
        "control_rate_gain_bpw": control_gains,
        "qwen_minus_stronger_control_excess_bpw": excess,
        "descriptor_bits_per_block": descriptor_bits_per_block,
        "remaining_refinement_bits_per_block":
            BLOCK_BITS - descriptor_bits_per_block,
        "promotion_uses_only_control_subtracted_excess": True,
    }
