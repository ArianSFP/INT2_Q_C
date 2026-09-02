#!/usr/bin/env python3
"""Emit the source-only split memory/compute gate for epsilon-TCQ v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from memory_plan import all_beam_table, memory_plan, work_plan


AUDITOR_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"
V1_SOURCE_MANIFEST_SHA256 = "e926575ac1a78a85d08e94e63d1cc85d70b1544e5b352b6abc45cb8653d83706"
BLOCK_VALUES = 1 << 21


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=True).encode("ascii")


def make_receipt() -> dict[str, object]:
    beams = all_beam_table(BLOCK_VALUES)
    maximum = memory_plan(BLOCK_VALUES, 32)
    maximum_work = work_plan(BLOCK_VALUES, 32)
    if not all(bool(row["passes_4gib_cap"]) for row in beams):
        raise RuntimeError("frozen compact representation breached 4 GiB")
    receipt: dict[str, object] = {
        "schema": "epsilon-tcq-polar-cow-source-gate-v2",
        "status": "SOURCE_ONLY_GO_MEMORY_CAPACITY_HOLD_COMPUTE_AND_PAYLOAD",
        "authenticated_current_decoder": {
            "sha256": AUDITOR_SHA256,
            "strata_indices": 64,
            "level_major_sc_passes": 6,
            "coordinate_local_six_event_abi": False,
        },
        "v1_source_manifest_sha256": V1_SOURCE_MANIFEST_SHA256,
        "geometry": {"block_values": BLOCK_VALUES, "depth": 21,
                     "beam_widths": [4, 8, 16, 32]},
        "verdicts": {
            "memory": "GO_MEMORY_CAPACITY",
            "compute": "HOLD_COMPUTE_AND_DEVICE_COW_IMPLEMENTATION",
            "payload": "HOLD_PAYLOAD",
        },
        "beam_table": beams,
        "maximum_beam_memory": maximum,
        "maximum_beam_work": maximum_work,
        "semantic_evidence": {
            "dense_reference_vs_ragged_exact_array_equality_required": True,
            "single_level_tested_block_lengths": [8, 16, 32, 64, 128],
            "six_level_tested_block_lengths": [8, 16, 32, 64],
            "tested_freeze_families": 3,
            "full_level_major_state_replay_preserved": True,
            "survivor_ancestry_and_selected_decision_backtrace": True,
            "canonical_arithmetic_replay_after_backtrace_required": True,
            "direct_int2_fallback": False,
            "scope": "source-free dense/ragged harness; no production device list search",
        },
        "architecture": {
            "likelihood_cells_per_path": BLOCK_VALUES - 1,
            "partial_sum_cells_per_path": BLOCK_VALUES - 1,
            "cow_physical_banks_upper_bound": 32,
            "two_b_candidate_bank_materialization": False,
            "candidate_children_pruned_before_state_mutation": True,
            "prefix_copying": False,
            "ancestry_symbols_packed": True,
            "checkpoint_recompute_selected": False,
            "packed_partial_sums_selected": False,
            "shared_immutable_likelihood_gain_assumed_in_bound": False,
            "device_cow_implementation_demonstrated": False,
        },
        "compute_hold_reason": {
            "causal_decision_rounds_worst_case": maximum_work[
                "causal_selected_decision_rounds"],
            "likelihood_node_updates_b32": maximum_work["likelihood_node_updates"],
            "partial_sum_state_writes_upper_b32": maximum_work[
                "partial_sum_state_writes_worst_active_upper_bound"],
            "partial_sum_xors_upper_b32": maximum_work[
                "partial_sum_xors_worst_active_upper_bound"],
            "polar_xors_b32": maximum_work["level_end_polar_xors"],
            "lower_index_adds_b32": maximum_work[
                "lower_index_adds_worst_active_upper_bound"],
            "winner_full_sc_replay_likelihood_updates": maximum_work[
                "winner_replay_likelihood_node_updates"],
            "required_next_evidence": (
                "persistent whole-six-level CuPy kernel with bounded launch count, "
                "measured throughput, exact compact-state replay, and independent audit"
            ),
            "allocation_or_primitive_smoke_is_not_throughput_evidence": True,
            "cupy_primitive_smoke_is_not_q0_16_frequency_equivalence": True,
        },
        "payload_authority": {"qwen": False, "current_codec": False,
                              "matched_gaussian": False},
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt)).hexdigest()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    encoded = canonical(make_receipt()).decode("ascii")
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="ascii", newline="\n")
    print(encoded)


if __name__ == "__main__":
    main()
