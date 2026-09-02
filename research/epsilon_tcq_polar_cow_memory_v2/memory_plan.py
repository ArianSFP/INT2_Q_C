#!/usr/bin/env python3
"""Exact logical-memory and worst-work ledger for compact STRATA polar SCL."""

from __future__ import annotations

import math
from typing import Any


LEVELS = 6
ALIGNMENT = 256
RATE_MAX_NUMERATOR = 5
RATE_MAX_DENOMINATOR = 2


class PlanError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def align(value: int, alignment: int = ALIGNMENT) -> int:
    require(type(value) is int and value >= 0 and type(alignment) is int and
            alignment > 0 and not (alignment & (alignment - 1)), "alignment")
    return (value + alignment - 1) // alignment * alignment


def symbol_bits(beam: int) -> int:
    require(type(beam) is int and beam in (4, 8, 16, 32), "frozen beam")
    return int(math.log2(beam)) + 1


def memory_plan(block_values: int, beam: int, *, selected_events: int | None = None,
                packed_partial_sums: bool = False) -> dict[str, Any]:
    require(type(block_values) is int and block_values >= 8 and
            not (block_values & (block_values - 1)), "polar block length")
    bits = symbol_bits(beam)
    depth = int(math.log2(block_values))
    maximum_events = LEVELS * block_values
    events = maximum_events if selected_events is None else int(selected_events)
    require(0 < events <= maximum_events, "selected-event bound")
    ragged = block_values - 1
    rows: list[dict[str, Any]] = []

    def add(name: str, logical: int, domain: str, note: str) -> None:
        rows.append({"name": name, "logical_bytes": int(logical),
                     "allocated_bytes": align(int(logical)),
                     "domain": domain, "note": note})

    add("leaf_prior_f64_banks", beam * block_values * 8, "device_level_persistent",
        "explicit per-path decoder-visible LUT gathers; no uncharged fused generation")
    add("likelihood_ragged_f64_banks", beam * ragged * 8, "device_persistent",
        "B paths times sum(1,2,...,N/2)=N-1 active LR cells")
    if packed_partial_sums:
        add("partial_sum_packed_banks", (beam * ragged + 7) // 8,
            "device_persistent", "one packed bit per active ragged partial sum")
        add("partial_sum_u8_working_layer", beam * (block_values // 2),
            "device_scratch", "largest unpacked mutable layer")
    else:
        add("partial_sum_u8_ragged_banks", beam * ragged, "device_persistent",
            "one uint8 per active ragged partial sum")
    add("lower_index_u8_banks", beam * block_values, "device_persistent",
        "one 0..63 lower-level index per coordinate and path")
    add("level_internal_plane_u8_scratch", beam * block_values, "device_scratch",
        "materialize each survivor from tape, then bit-reverse/in-place polar transform")
    tape_bits = events * beam * bits
    add("survivor_ancestry_packed", (tape_bits + 7) // 8, "host_or_device_persistent",
        f"{bits}-bit (parent,decision) symbol per survivor/event")
    add("path_metric_f64", beam * 8, "device_persistent", "one exact path objective")
    add("path_arithmetic_low_u32", beam * 4, "device_persistent", "canonical coder low")
    add("path_arithmetic_high_u32", beam * 4, "device_persistent", "canonical coder high")
    add("path_arithmetic_pending_u64", beam * 8, "device_persistent", "pending underflow bits")
    add("path_arithmetic_emitted_u64", beam * 8, "device_persistent", "logical bits so far")
    add("path_causal_state_u16", beam * 2, "device_persistent", "up to 64 frozen WFA states")
    add("path_active_u8", beam, "device_persistent", "active survivor mask")
    add("path_lower_bank_u8", beam, "device_persistent", "immutable lower-index bank handle")
    add("controller_level_u8", 1, "device_control", "global level in 0..5")
    add("controller_phase_u32", 4, "device_control", "global SC phase in 0..N-1")
    add("controller_event_u64", 8, "device_control", "global selected-event ordinal")
    add("controller_active_paths_u8", 1, "device_control", "one-path startup through B")
    add("level_event_offsets_u64", (LEVELS + 1) * 8, "device_control",
        "six level boundaries in packed ancestry")
    add("layer_cow_handles_u32", beam * depth * 4, "device_persistent",
        "one slot handle per active path/layer")
    add("layer_cow_refcounts_u16", beam * depth * 2, "device_persistent",
        "worst one physical slot per path/layer")
    add("transformed_target_f64", block_values * 8, "device_shared",
        "one decoder-visible transformed source block")
    add("six_freeze_flag_u8", LEVELS * block_values, "device_shared",
        "conservative materialized flags; procedural generation may remove")
    add("current_frozen_external_u8", block_values, "device_shared",
        "only one of six seeded frozen vectors resident")
    add("ratio_lut_f64", LEVELS * 64 * 8, "device_shared",
        "leaf_prior_ratios reduced to decoder-visible lookup")
    add("candidate_metric_f64", 2 * beam * 8, "device_scratch", "two branch metrics/path")
    add("candidate_parent_u8", 2 * beam, "device_scratch", "parent ids; B<=32")
    add("candidate_decision_u8", 2 * beam, "device_scratch", "legal branch decisions")
    add("candidate_tie_index_u8", 2 * beam, "device_scratch", "stable top-k secondary key")
    add("survivor_parent_u8", beam, "device_scratch", "selected parent ids")
    add("survivor_decision_u8", beam, "device_scratch", "selected decisions")
    add("wfa_transition_frequency_table", 64 * 2 * 4, "device_shared",
        "uint16 next state plus uint16 branch frequency for 64-state source law")
    add("backtrace_selected_bits", (events + 7) // 8, "host_scratch",
        "one winning selected-event sequence")
    # Any candidate exceeding 2.5 bpw cannot pass the frozen rate gate and may
    # stop emission. Add 64 termination bits conservatively.
    payload_bits = (RATE_MAX_NUMERATOR * block_values +
                    RATE_MAX_DENOMINATOR - 1) // RATE_MAX_DENOMINATOR + 64
    add("winner_payload_rate_capped", (payload_bits + 7) // 8, "host_output",
        "fail-closed at the 2.5-bpw promotion ceiling")
    logical_total = sum(row["logical_bytes"] for row in rows)
    allocated_total = sum(row["allocated_bytes"] for row in rows)
    return {
        "schema": "epsilon-tcq-polar-cow-memory-plan-v2",
        "block_values": block_values, "depth": depth, "sc_levels": LEVELS,
        "beam_width": beam, "selected_events": events,
        "selected_events_is_worst_case_6N": events == maximum_events,
        "survivor_symbol_bits": bits,
        "partial_sum_storage": "packed+capped-u8-layer" if packed_partial_sums else "ragged-u8",
        "buffers": rows, "logical_peak_bytes": logical_total,
        "aligned_peak_bytes": allocated_total,
        "memory_cap_bytes": 4 * (1 << 30),
        "passes_4gib_cap": allocated_total < 4 * (1 << 30),
        "cow_worst_case_assumes_no_sharing_gain": True,
        "candidate_children_mutate_only_after_topk": True,
        "maximum_physical_state_banks": beam,
        "metadata_abi": "explicit aligned struct-of-arrays; no implicit C/CUDA struct padding",
    }


def work_plan(block_values: int, beam: int, *, selected_events: int | None = None,
              checkpoint_spacing: int = 1) -> dict[str, Any]:
    require(type(block_values) is int and block_values >= 8 and
            not (block_values & (block_values - 1)), "work block length")
    symbol_bits(beam)
    require(type(checkpoint_spacing) is int and 1 <= checkpoint_spacing <=
            int(math.log2(block_values)), "checkpoint spacing")
    depth = int(math.log2(block_values))
    maximum_events = LEVELS * block_values
    events = maximum_events if selected_events is None else int(selected_events)
    require(0 < events <= maximum_events, "work selected events")
    # These are exact evaluated *worst-active upper bounds* for the frozen v2
    # schedule: B paths are charged at every phase even during startup.  They are
    # not measurements and do not imply that a practical kernel exists.
    base_lr = LEVELS * beam * block_values * depth
    lr = base_lr * checkpoint_spacing
    partial_writes = LEVELS * beam * (block_values * depth // 2 + 1)
    partial_xors = LEVELS * beam * (block_values * (depth - 2) // 2 + 1)
    polar_xor = LEVELS * beam * block_values * depth // 2
    startup = int(math.log2(beam))
    if events < startup:
        branch_candidates = sum(2 << event for event in range(events))
        tape_symbols = sum(2 << event for event in range(events))
    else:
        branch_candidates = 2 * (beam - 1) + 2 * beam * (events - startup)
        tape_symbols = 2 * beam - 2 + beam * (events - startup)
    bitonic_width = 2 * beam
    bitonic_log = int(math.log2(bitonic_width))
    bitonic_comparators_per_round = (
        bitonic_width * bitonic_log * (bitonic_log + 1) // 4)
    lower_index_adds = LEVELS * beam * block_values
    lower_index_clone_bytes_upper = LEVELS * (beam - 1) * block_values
    level_boundary_tape_reads_upper = beam * events
    # A deliberately loose but mechanically safe COW traffic cap.  It assumes
    # every split that retains both children eventually privatizes every ragged
    # LR and mu byte.  Production must measure a much tighter schedule.
    cow_full_tree_bytes = (block_values - 1) * 9
    cow_copy_bytes_absolute_upper = events * (beam - 1) * cow_full_tree_bytes
    replay_lr = LEVELS * block_values * depth
    replay_partial_writes = LEVELS * (block_values * depth // 2 + 1)
    replay_partial_xors = LEVELS * (block_values * (depth - 2) // 2 + 1)
    replay_polar = LEVELS * block_values * depth // 2
    replay_index = LEVELS * block_values
    return {
        "schema": "epsilon-tcq-polar-cow-work-plan-v2",
        "block_values": block_values, "depth": depth, "beam_width": beam,
        "selected_events": events, "checkpoint_spacing": checkpoint_spacing,
        "likelihood_node_updates": lr,
        "likelihood_node_updates_without_recompute": base_lr,
        "likelihood_node_updates_is_worst_active_upper_bound": True,
        "partial_sum_state_writes_worst_active_upper_bound": partial_writes,
        "partial_sum_xors_worst_active_upper_bound": partial_xors,
        "level_end_polar_xors": polar_xor,
        "branch_candidates_scored": branch_candidates,
        "survivor_tape_symbols_written": tape_symbols,
        "fixed_tape_symbol_capacity": beam * events,
        "stable_bitonic_comparators": events * bitonic_comparators_per_round,
        "stable_bitonic_comparators_per_round": bitonic_comparators_per_round,
        "leaf_prior_lut_gathers_worst_active_upper_bound": lower_index_adds,
        "lower_index_adds_worst_active_upper_bound": lower_index_adds,
        "lower_index_clone_bytes_upper_bound": lower_index_clone_bytes_upper,
        "level_boundary_tape_reads_upper_bound": level_boundary_tape_reads_upper,
        "cow_layer_copy_bytes_absolute_upper_bound": cow_copy_bytes_absolute_upper,
        "cow_copy_bound_is_intentionally_loose_compute_hold": True,
        "winner_backtrace_events": events,
        "winner_replay_likelihood_node_updates": replay_lr,
        "winner_replay_partial_sum_state_writes": replay_partial_writes,
        "winner_replay_partial_sum_xors": replay_partial_xors,
        "winner_replay_polar_xors": replay_polar,
        "winner_replay_lower_index_adds": replay_index,
        "winner_arithmetic_replay_events": events,
        "causal_selected_decision_rounds": events,
        "persistent_device_kernel_required": True,
        "per_decision_host_or_kernel_launch_forbidden": True,
    }


def all_beam_table(block_values: int = 1 << 21) -> list[dict[str, Any]]:
    output = []
    for beam in (4, 8, 16, 32):
        memory = memory_plan(block_values, beam)
        work = work_plan(block_values, beam)
        output.append({
            "beam_width": beam,
            "aligned_peak_bytes": memory["aligned_peak_bytes"],
            "logical_peak_bytes": memory["logical_peak_bytes"],
            "passes_4gib_cap": memory["passes_4gib_cap"],
            "likelihood_node_updates": work["likelihood_node_updates"],
            "partial_sum_state_writes": work["partial_sum_state_writes_worst_active_upper_bound"],
            "partial_sum_xors": work["partial_sum_xors_worst_active_upper_bound"],
            "level_end_polar_xors": work["level_end_polar_xors"],
            "lower_index_adds": work["lower_index_adds_worst_active_upper_bound"],
            "causal_selected_decision_rounds": work["causal_selected_decision_rounds"],
            "branch_candidates_scored": work["branch_candidates_scored"],
            "survivor_tape_symbols_written": work["survivor_tape_symbols_written"],
        })
    return output
