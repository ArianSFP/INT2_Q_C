#!/usr/bin/env python3
"""Source-independent contracts for the MOSAIC secondary-oracle ladder."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from typing import Any, Mapping, Sequence


SCHEMA = "mosaic-secondary-oracles-contract-v0"
TARGET_RELATIVE_MSE = 0.025
TARGET_F = 0.8
RATE_MIN = Fraction(43, 20)
RATE_MAX = Fraction(5, 2)
COARSE_RATE = Fraction(307, 128)
FINE_RATE = Fraction(12, 128)
METADATA_RATE = Fraction(1, 128)
COARSE_RELATIVE_MSE = 0.036975150060595235
REQUIRED_COARSE_CAPTURE = 1.0 - TARGET_RELATIVE_MSE / COARSE_RELATIVE_MSE
BLOCK_VALUES = 4096
FINE_BITS_PER_BLOCK = 384
PAGE_BYTES = 4096
ROLE_ALIGNMENT_BYTES = 64
EXPERT_HEADER_BYTES = 64
MIN_CONTROL_EXCESS_BPW = 0.03
ROLES = ("gate", "up", "down_transposed")
CONTROL_SEEDS = (
    10619863,
    10619881,
    10619909,
    10619927,
    10619953,
    10619971,
    10619999,
    10620017,
)
NON_DYADIC_PERIODS = (
    3, 5, 6, 7, 9, 10, 11, 12, 13, 15, 17, 19, 21, 23, 25, 27,
    29, 31, 33, 35, 37, 41, 43, 47, 53, 59, 61, 63, 65, 67,
    71, 73, 79, 83, 89, 97, 101, 107, 113, 127,
)


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "float": float(value),
    }


def align_up(value: int, alignment: int) -> int:
    require(type(value) is int and value >= 0, "alignment value")
    require(type(alignment) is int and alignment > 0, "alignment")
    return ((value + alignment - 1) // alignment) * alignment


def physical_expert_ledger(
    *,
    weights: int,
    role_component_bytes: Sequence[int],
    shared_model_bytes: int = 0,
) -> dict[str, Any]:
    """Charge one page-aligned expert-local packet and every model byte."""

    require(type(weights) is int and weights > 0, "expert weights")
    require(
        len(role_component_bytes) == len(ROLES)
        and all(type(value) is int and value > 0 for value in role_component_bytes),
        "role component bytes",
    )
    require(type(shared_model_bytes) is int and shared_model_bytes >= 0, "shared model bytes")
    cursor = EXPERT_HEADER_BYTES + shared_model_bytes
    placements = []
    for role, size in zip(ROLES, role_component_bytes, strict=True):
        cursor = align_up(cursor, ROLE_ALIGNMENT_BYTES)
        begin = cursor
        cursor += size
        placements.append({"role": role, "begin": begin, "bytes": size, "end": cursor})
    unpadded = cursor
    lower_bytes = math.ceil(float(RATE_MIN) * weights / 8.0)
    physical = align_up(max(unpadded, lower_bytes), PAGE_BYTES)
    rate = Fraction(8 * physical, weights)
    cold_bytes = physical
    amplification = cold_bytes / physical
    return {
        "weights": weights,
        "expert_header_bytes": EXPERT_HEADER_BYTES,
        "shared_model_bytes": shared_model_bytes,
        "role_alignment_bytes": ROLE_ALIGNMENT_BYTES,
        "placements": placements,
        "unpadded_bytes": unpadded,
        "canonical_zero_padding_bytes": physical - unpadded,
        "physical_bytes": physical,
        "physical_rate_bpw": fraction_record(rate),
        "passes_rate_interval": RATE_MIN <= rate <= RATE_MAX,
        "expert_packet_page_aligned": True,
        "external_storage_reads": 1,
        "external_storage_refetches": 0,
        "cold_storage_bytes": cold_bytes,
        "cold_read_amplification": amplification,
        "passes_strict_cold_read_below_2x": amplification < 2.0,
        "host_scratch_and_hbm_excluded_from_storage_ledger": True,
    }


def tactic_fine_ledger(descriptor_bits_per_block: int) -> dict[str, Any]:
    require(
        type(descriptor_bits_per_block) is int
        and 0 <= descriptor_bits_per_block <= FINE_BITS_PER_BLOCK,
        "fine descriptor bits",
    )
    total = COARSE_RATE + FINE_RATE + METADATA_RATE
    return {
        "coarse_rate_bpw": fraction_record(COARSE_RATE),
        "fine_rate_bpw": fraction_record(FINE_RATE),
        "metadata_rate_bpw": fraction_record(METADATA_RATE),
        "total_rate_bpw": fraction_record(total),
        "fine_bits_per_block": FINE_BITS_PER_BLOCK,
        "descriptor_bits_per_block": descriptor_bits_per_block,
        "innovation_bits_per_block": FINE_BITS_PER_BLOCK - descriptor_bits_per_block,
        "descriptor_displaces_innovation_bits": True,
        "rate_equals_2p5": total == RATE_MAX,
    }


def f_value(relative_mse: float, physical_rate_bpw: float) -> float:
    require(
        math.isfinite(relative_mse)
        and relative_mse > 0.0
        and math.isfinite(physical_rate_bpw),
        "finite score",
    )
    return relative_mse * 2.0 ** (2.0 * physical_rate_bpw)


def residual_source_gate(
    *,
    input_sse: float,
    source_energy: float,
    source_remaining_sse: float,
    descriptor_bits_per_block: int,
    controls: Mapping[str, float] | None,
) -> dict[str, Any]:
    """Source-first target gate, then Qwen-minus-stronger-control excess."""

    require(
        all(math.isfinite(value) and value > 0.0 for value in (input_sse, source_energy))
        and math.isfinite(source_remaining_sse)
        and 0.0 < source_remaining_sse <= input_sse,
        "residual gate values",
    )
    ledger = tactic_fine_ledger(descriptor_bits_per_block)
    relative_mse = source_remaining_sse / source_energy
    source_gain = -0.5 * math.log2(source_remaining_sse / input_sse)
    if relative_mse > TARGET_RELATIVE_MSE + 1e-12:
        require(controls is None, "controls forbidden after absolute source miss")
        return {
            "status": "HARD_KILL_ABSOLUTE_SOURCE_MISSES_D_0P025",
            "controls_may_open": False,
            "relative_mse": relative_mse,
            "source_gain_bpw": source_gain,
            "fine_ledger": ledger,
        }
    require(
        controls is not None and set(controls) == {"permutation", "gaussian"},
        "both controls required after source survival",
    )
    control_gains = {}
    for name, remaining in controls.items():
        require(math.isfinite(remaining) and 0.0 < remaining <= input_sse, "control SSE")
        control_gains[name] = -0.5 * math.log2(remaining / input_sse)
    excess = source_gain - max(control_gains.values())
    status = (
        "ELIGIBLE_FOR_ONE_NESTED_FINITE_BUILD"
        if excess + 1e-15 >= MIN_CONTROL_EXCESS_BPW
        else "HARD_KILL_SOURCE_NOT_SPECIFIC_0P03_BPW"
    )
    return {
        "status": status,
        "controls_may_open": True,
        "relative_mse": relative_mse,
        "source_gain_bpw": source_gain,
        "control_gains_bpw": control_gains,
        "source_minus_stronger_control_bpw": excess,
        "fine_ledger": ledger,
        "gains_may_be_added_to_separate_oracles": False,
    }


def recurrence_codec_gate(
    *,
    relative_mse: float,
    ledger: Mapping[str, Any],
    literal4_physical_rate_bpw: float,
    matched_control_saving_bpw: float | None,
) -> dict[str, Any]:
    rate = float(ledger["physical_rate_bpw"]["float"])
    score = f_value(relative_mse, rate)
    absolute = (
        ledger["passes_rate_interval"]
        and ledger["passes_strict_cold_read_below_2x"]
        and score <= TARGET_F
    )
    saving = literal4_physical_rate_bpw - rate
    if not absolute:
        require(matched_control_saving_bpw is None, "control forbidden after recurrence absolute miss")
        status = "HARD_KILL_RECURRENCE_PACKET_RATE_F_OR_READ"
        excess = None
    else:
        require(
            matched_control_saving_bpw is not None
            and math.isfinite(matched_control_saving_bpw),
            "matched control saving after recurrence survival",
        )
        excess = saving - matched_control_saving_bpw
        status = (
            "ELIGIBLE_FOR_PORTABILITY_AND_LITERAL_NESTING"
            if excess + 1e-15 >= MIN_CONTROL_EXCESS_BPW
            else "HARD_KILL_RECURRENCE_NOT_SOURCE_SPECIFIC_0P03_BPW"
        )
    return {
        "status": status,
        "physical_rate_bpw": rate,
        "relative_mse": relative_mse,
        "F": score,
        "literal4_saving_bpw": saving,
        "matched_control_saving_bpw": matched_control_saving_bpw,
        "source_minus_control_saving_bpw": excess,
        "controls_may_open": absolute,
        "packet_is_finite": True,
        "universal_claim_authority": False,
    }
