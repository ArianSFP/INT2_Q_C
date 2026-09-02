#!/usr/bin/env python3
"""Payload-free optimistic physical lower bound from sealed v9 source pins."""

from __future__ import annotations

import json
import math
from fractions import Fraction


SOURCE_WEIGHTS = 28_311_552
CURRENT_STRATA_BYTES = 8_847_360
UWFA_V9_BYTES = 8_892_416
PINNED_UNIQUE_SELECTED_DECISIONS = 126_627_266
PINNED_STREAMS = 15
PINNED_EVALUATION_EXPERTS = 6
LEVELS = 6
CHUNK_DECISIONS = 4096
CHUNK_HEADER_BYTES = 80
EXPERT_HEADER_BYTES = 256
CATALOG_BYTES = 4096


def fraction_record(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "float": float(value),
    }


def derive() -> dict[str, object]:
    """Return unconditional optimistic bounds; no filesystem access occurs."""

    # Every selected decision belongs to one of six nonempty level segments in
    # one stream. Sum ceil(n_i/K) >= ceil(sum n_i/K); 15*6 supplies the weaker
    # nonempty-segment bound. Expert-local duplication can only increase this.
    minimum_unique_chunks = max(
        PINNED_STREAMS * LEVELS,
        math.ceil(PINNED_UNIQUE_SELECTED_DECISIONS / CHUNK_DECISIONS),
    )
    bare_floor = (
        CATALOG_BYTES
        + PINNED_EVALUATION_EXPERTS * EXPERT_HEADER_BYTES
        + minimum_unique_chunks * CHUNK_HEADER_BYTES
    )
    raw_floor = bare_floor + math.ceil(PINNED_UNIQUE_SELECTED_DECISIONS / 8)
    cap = CURRENT_STRATA_BYTES
    payload_bit_budget = max(0, 8 * (cap - bare_floor))
    max_sum_complexity = payload_bit_budget // 2
    return {
        "schema": "strata-sc-gf2-prepayload-rate-bound-v1",
        "status": "RAW_EXACT_DECISION_PACKET_HARD_KILL_LFSR_ONLY_REMAINS_PLAUSIBLE",
        "source_weights": SOURCE_WEIGHTS,
        "pinned_unique_selected_sc_decisions": PINNED_UNIQUE_SELECTED_DECISIONS,
        "selected_decisions_per_weight_is_not_a_rate": fraction_record(
            Fraction(PINNED_UNIQUE_SELECTED_DECISIONS, SOURCE_WEIGHTS)
        ),
        "audited_current_arithmetic_bytes": CURRENT_STRATA_BYTES,
        "audited_current_arithmetic_rate": fraction_record(
            Fraction(8 * CURRENT_STRATA_BYTES, SOURCE_WEIGHTS)
        ),
        "audited_v9_arithmetic_bytes": UWFA_V9_BYTES,
        "audited_v9_arithmetic_rate": fraction_record(
            Fraction(8 * UWFA_V9_BYTES, SOURCE_WEIGHTS)
        ),
        "minimum_unique_chunks_before_expert_duplication": minimum_unique_chunks,
        "optimistic_zero_complexity_bare_floor_bytes": bare_floor,
        "optimistic_zero_complexity_bare_floor_bpw": float(
            Fraction(8 * bare_floor, SOURCE_WEIGHTS)
        ),
        "floor_omits_metadata_page_padding_and_shared_stream_duplication": True,
        "zero_complexity_grammar_floor_can_fit_2p5": bare_floor <= cap,
        "optimistic_raw_fallback_floor_bytes": raw_floor,
        "optimistic_raw_fallback_floor_bpw": float(
            Fraction(8 * raw_floor, SOURCE_WEIGHTS)
        ),
        "raw_fallback_can_fit_2p5": raw_floor <= cap,
        "raw_fallback_hard_kill_before_payload": raw_floor > cap,
        "raw_floor_excess_bytes_over_2p5": raw_floor - cap,
        "optimistic_maximum_sum_BM_complexity_at_2p5_necessary_not_sufficient": max_sum_complexity,
        "optimistic_maximum_mean_BM_complexity_per_minimum_chunk": max_sum_complexity / minimum_unique_chunks,
        "optimistic_maximum_two_L_payload_bits_per_unique_decision": payload_bit_budget / PINNED_UNIQUE_SELECTED_DECISIONS,
        "positive_recurrence_claim_permitted": False,
        "reason": "actual level counts, expert duplication, metadata, page padding, BM complexities, packet bytes, and independent Q0.16 replay are not available source-only",
    }


if __name__ == "__main__":
    print(json.dumps(derive(), sort_keys=True, separators=(",", ":"), allow_nan=False))

