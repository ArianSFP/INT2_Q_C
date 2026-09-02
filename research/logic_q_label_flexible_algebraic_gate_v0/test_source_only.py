#!/usr/bin/env python3
"""Hostile source-only tests for LOGIC-Q v0."""

from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent


class TestFailure(RuntimeError):
    pass


def check(condition: bool, name: str) -> None:
    if not condition:
        raise TestFailure(name)


def raises(function, fragment: str) -> None:
    try:
        function()
    except Exception as error:
        check(fragment.lower() in str(error).lower(),
              f"exception fragment {fragment!r}: {error}")
        return
    raise TestFailure(f"expected exception containing {fragment!r}")


def load(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    check(specification is not None and specification.loader is not None,
          f"module spec {filename}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def synthetic_panel(protocol):
    rows = []
    for layer in range(10):
        for slot in range(4):
            for role in ("gate", "up", "down_transposed"):
                rows.append(protocol.PanelRow(
                    f"layer-{layer:02d}", f"slot-{slot:02d}", role, 8, 16))
    return rows


def main() -> None:
    before = set(sys.modules)
    import numpy as np

    core = load("logicq_core", "logicq_core.py")
    protocol = load("logicq_panel_protocol", "panel_protocol.py")
    tests = []

    budget = core.exact_budget_identities()
    check(abs(budget["reference_F"] - 0.9888693569009007) < 1e-15 and
          abs(budget["required_saving_bpw"] - 0.1528899669629145) < 2e-16 and
          abs(budget["required_mse_reduction_fraction"]
              - 0.19099525693951513) < 1e-15,
          "audited target identities")
    tests.append("audited_target_identities")

    writer = core.BitWriter()
    writer.write(5, 3)
    writer.write(0, 0)
    writer.write(19, 5)
    payload = writer.finish()
    check(payload == bytes([0b10110011]) and writer.bit_length == 8,
          "bit writer")
    reader = core.BitReader(payload, 8)
    check(reader.read(3) == 5 and reader.read(0) == 0 and reader.read(5) == 19,
          "bit reader")
    reader.finish()
    raises(lambda: core.BitReader(b"\x01", 1), "padding")
    tests.append("canonical_bitstream_and_padding")

    for n in range(1, 9):
        for count in range(n + 1):
            total = math.comb(n, count)
            for rank in range(total):
                positions = core.subset_unrank_lex(rank, n, count)
                check(core.subset_rank_lex(positions, n) == rank,
                      "combinatorial rank roundtrip")
    tests.append("complete_small_combinatorial_rank_roundtrip")

    base = (0, 1, 2, 3, 0)
    plan = core.ExceptionPlan((1, 4), (3, 2), 1.0,
                              core.exception_bits(5, 2)["total_bits"],
                              core.exception_bits(5, 2)["total_bits"], 1.0)
    writer = core.BitWriter()
    core.write_exceptions(writer, 5, base, plan)
    payload = writer.finish()
    reader = core.BitReader(payload, writer.bit_length)
    decoded = core.read_exceptions(reader, 5, base)
    reader.finish()
    check(decoded == (0, 3, 2, 3, 2), "exception finite decode")
    tests.append("exception_count_subset_trit_finite_decode")

    costs = np.full((4, 4), 10.0, dtype=np.float64)
    costs[:, 0] = [0.0, 0.0, 0.0, 0.0]
    costs[2, 3] = -0.0
    # One weighted/valuable alternative at position two; exact optimizer must
    # select by real distortion delta, not by a fixed-label Hamming proxy.
    costs[2, 0] = 8.0
    costs[2, 3] = 0.1
    exception = core.optimize_exceptions(
        np, costs, np.zeros(4, dtype=np.uint8), 0.01, maximum=1)
    check(exception.positions == (2,) and exception.labels == (3,),
          "weighted exception choice")
    tests.append("exception_optimizer_uses_weighted_four_level_distortion")

    for value in (2.0 ** -20, 0.125, 1.0, math.pi, 4096.0):
        bits = core.fp32_to_bf16_bits(value)
        decoded_scale = core.bf16_bits_to_float(bits)
        check(decoded_scale > 0.0 and math.isfinite(decoded_scale),
              "BF16 scale")
    tests.append("transmitted_bf16_scale_roundtrip")

    rm_costs = np.asarray([
        [0.1, 2.0, 3.0, 4.0],
        [2.0, 0.1, 4.0, 3.0],
        [4.0, 3.0, 0.1, 2.0],
        [3.0, 4.0, 2.0, 0.1],
    ], dtype=np.float64)
    rm = core.search_rm1_block(
        np, rm_costs, lambda_per_bit=0.03, exception_limit=2,
        exact_pair_max=4096, list_pairs=4)
    words = core.rm1_codewords(np, 4)
    brute = None
    for first in range(words.shape[0]):
        for second in range(words.shape[0]):
            base_labels = core.gray_labels_from_planes(
                np, words[first], words[second]).reshape(-1)
            overlay = core.optimize_exceptions(
                np, rm_costs, base_labels, 0.03, maximum=2,
                fixed_prefix_bits=6, byte_align_total=True)
            key = (overlay.objective, overlay.charged_total_bits,
                   overlay.distortion, first, second,
                   overlay.positions, overlay.labels)
            if brute is None or key < brute:
                brute = key
    check(rm.exact_pair_search and brute is not None and
          abs(rm.objective - brute[0]) < 1e-12 and
          rm.total_bits == brute[1], "RM exact weighted objective")
    writer = core.BitWriter()
    core.write_rm1_block(writer, np, 4, rm)
    payload = writer.finish()
    reader = core.BitReader(payload, writer.bit_length)
    check(core.read_rm1_block(reader, np, 4) == rm.labels,
          "RM finite block replay")
    reader.finish()
    tests.append("exact_small_rm1_joint_gray_search_and_decode")

    list_rm = core.search_rm1_block(
        np, np.tile(rm_costs, (4, 1)), lambda_per_bit=0.01,
        exception_limit=2, exact_pair_max=16, list_pairs=7)
    check(not list_rm.exact_pair_search and
          list_rm.pair_candidates_evaluated == 7,
          "RM bounded list authority")
    tests.append("rm_list_miss_has_no_global_negative_authority")

    matrix = np.asarray([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    check(core.gf2_rank(np, matrix) == 2, "GF2 rank")
    left = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    right = np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    check(np.array_equal(core.gf2_product(np, left, right), matrix),
          "GF2 product")
    tests.append("gf2_exact_arithmetic")

    rank680 = core.rank680_accounting()
    check(rank680["implemented_raw_factor_bits_per_role"] == 3_829_760 and
          rank680["implemented_raw_factor_bits_all_three_roles"] == 11_489_280 and
          abs(rank680["implemented_raw_factor_bpw_per_role_and_expert"]
              - 2.4348958333333335) < 1e-15 and
          abs(rank680["ideal_asymptotic_bpw_per_role_and_expert"]
              - 1.846923828125) < 1e-15 and
          rank680["ideal_counting_serializer_implemented"] is False,
          "rank680 charged distinction")
    tests.append("rank680_raw_packet_not_ideal_counting_bound")

    tiny_values = np.asarray([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)
    tiny_weights = np.asarray([1.0, 1.5, 0.75, 2.0], dtype=np.float64)
    gf_component = core.encode_gf2_component(
        np, tiny_values, tiny_weights, role="gate", rows=2, cols=2,
        block_size=4, lambda_per_bit=0.01, ranks=(0, 1), profile=1,
        exception_limit=2, exact_factor_pair_max=65536,
        heuristic_sweeps=1)
    gf_labels, gf_values, gf_header = core.decode_component(np, gf_component.packet)
    check(gf_labels == gf_component.labels and
          tuple(gf_values) == gf_component.reconstruction and
          gf_header.family == core.FAMILY_GF2 and gf_component.exact_search,
          "GF2 component replay")
    tests.append("exact_tiny_gf2_rank_plus_exceptions_finite_decode")

    random_costs = np.random.default_rng(7).random((16, 4), dtype=np.float64)
    heuristic = core.search_gf2(
        np, random_costs, 4, 4, 2, 0.01, exception_limit=2,
        exact_factor_pair_max=1, heuristic_sweeps=1)
    check(not heuristic.exact_search and heuristic.candidates_evaluated > 1,
          "GF2 heuristic authority")
    tests.append("gf2_alternating_miss_is_not_global_oracle")

    domain = core.qwen_coordinate_domain_record()
    check(domain["role_radices"] == [3] + [2] * 19 and
          domain["sites_per_role"] == 1_572_864 and
          domain["valid_expert_sites"] == 4_718_592 and
          domain["invalid_or_unused_naive_sites"] == 3_670_016 and
          domain["invalid_mask_transmitted_or_assumed_free"] is False,
          "exact mixed-radix domain")
    tests.append("no_free_23bit_coordinate_mask")

    romdd_component = core.encode_romdd_component(
        np, tiny_values, tiny_weights, role="up", rows=2, cols=2,
        block_size=4, lambda_per_bit=0.01, depths=(0, 1, 2), profile=1,
        exception_limit=2)
    romdd_labels, romdd_values, romdd_header = core.decode_component(
        np, romdd_component.packet)
    check(romdd_labels == romdd_component.labels and
          romdd_values == romdd_component.reconstruction and
          romdd_header.family == core.FAMILY_ROMDD and
          romdd_component.diagnostics["invalid_padded_coordinate_mask_used"] is False,
          "ROMDD replay")
    tests.append("bounded_valid_domain_romdd_finite_decode")

    literal_gate = core.encode_literal_component(
        np, tiny_values, tiny_weights, role="gate", rows=2, cols=2,
        block_size=4, profile=1)
    literal_up = core.encode_literal_component(
        np, tiny_values + 0.01, tiny_weights, role="up", rows=2, cols=2,
        block_size=4, profile=1)
    literal_down = core.encode_literal_component(
        np, tiny_values - 0.01, tiny_weights, role="down_transposed",
        rows=2, cols=2, block_size=4, profile=1)
    expert = core.pack_expert({
        "gate": literal_gate.packet,
        "up": literal_up.packet,
        "down_transposed": literal_down.packet,
    })
    decoded_expert = core.unpack_expert(np, expert)
    ledger = core.expert_ledger({
        "gate": literal_gate.packet,
        "up": literal_up.packet,
        "down_transposed": literal_down.packet,
    })
    check(len(expert) == 4096 and set(decoded_expert) == set(core.ROLE_IDS) and
          ledger["physical_expert_bytes"] == 4096 and
          ledger["all_bytes_charged"] is True,
          "expert envelope ledger")
    mutated = bytearray(expert)
    mutated[-1] = 1
    raises(lambda: core.unpack_expert(np, bytes(mutated)), "padding")
    tests.append("three_role_headers_alignment_page_and_finite_decode")

    literal_min = core.family_minimum_component_bits(
        core.FAMILY_LITERAL, 768, 2048, 2048, 0)
    bound680 = core.optimistic_family_bound(
        family=core.FAMILY_GF2, rows=768, cols=2048, block_size=2048,
        parameter=680, nearest_weighted_sse=1.0, source_energy=32.0,
        baseline_physical_bits=literal_min)
    check(bound680["mandatory_packet_exceeds_2p5_bpw"] is False and
          bound680["minimum_rate_bpw"] > 2.43 and
          bound680["unchanged_mse_saving_bound_misses_required"] is True and
          bound680["status"] ==
              "RATE_ONLY_HARD_KILL_UNCHANGED_MSE_SAVING_BOUND",
          "rank680 descriptor bound")
    tests.append("hard_kill_bounds_precede_rank_search")

    max_rank = core.maximum_raw_gf2_rank_under_rate(
        768, 2048, 2048, 2.0 - core.REQUIRED_SAVING_BPW)
    check(0 <= max_rank < 680, "raw-factor max rank")
    tests.append("raw_factor_rank_budget_excludes_680_before_exceptions")

    rm_controls = core.rm_descriptor_controls()
    rm3_2048 = next(row for row in rm_controls["blockwise_descriptor_rows"]
                    if row["block_size"] == 2048 and row["order"] == 3)
    check(rm_controls["global_rm3_23_dimension_per_bitplane"] == 2048 and
          rm_controls["global_rm3_23_descriptor_bits_both_planes"] == 4096 and
          rm3_2048["descriptor_bits_both_planes"] == 1_069_056 and
          abs(rm3_2048["descriptor_bpw_before_headers_exceptions"]
              - 0.2265625) < 1e-15 and
          rm_controls["random_plane_expected_to_be_near_half_hamming_distance"],
          "RM descriptor controls")
    tests.append("global_rm3_and_block_repetition_are_distinct_ledgers")

    panel = synthetic_panel(protocol)
    split = protocol.whole_component_split(
        panel, test_layer_count=5, validation_slot_count=1)
    check(len(split["test_layers"]) == 5 and
          len(split["validation_slots"]) == 1 and
          not ({row.layer for row in split["test"]}
               & {row.layer for row in split["train"] + split["validation"]}) and
          not ({row.slot for row in split["validation"]}
               & {row.slot for row in split["train"]}),
          "whole component split")
    bad_panel = panel[:-1]
    raises(lambda: protocol.validate_panel(bad_panel), "exact three")
    tests.append("whole_layer_outer_and_whole_slot_inner_holdout")

    control1, receipt1 = protocol.moment_matched_gaussian(
        np, tiny_values, block_size=4, seed=core.CONTROL_SEEDS[0],
        component_ordinal=9)
    control2, receipt2 = protocol.moment_matched_gaussian(
        np, tiny_values, block_size=4, seed=core.CONTROL_SEEDS[0],
        component_ordinal=9)
    check(np.array_equal(control1, control2) and receipt1 == receipt2 and
          receipt1["maximum_mean_absolute_error"] < 1e-12 and
          receipt1["maximum_centered_sse_relative_error"] < 1e-12 and
          receipt1["prebuilt_labels_accepted"] is False,
          "matched Gaussian replay")
    tests.append("matched_gaussian_continuous_pipeline_deterministic")

    bank = protocol.encode_family_bank(
        np, control1, tiny_weights, role="gate", rows=2, cols=2,
        block_size=4, lambda_per_bit=0.01, rm_exception_limit=2,
        gf2_ranks=(0, 1), gf2_exception_limit=2,
        romdd_depths=(0, 1, 2), romdd_exception_limit=2,
        rm_exact_pair_max=4096, rm_list_pairs=8,
        gf2_exact_pair_max=65536, gf2_heuristic_sweeps=1)
    check(set(bank) == {
        "literal4", "rm1_plus_exceptions", "gf2_rank_plus_exceptions",
        "romdd_plus_exceptions"} and
          all(core.decode_component(np, component.packet)[0] == component.labels
              for component in bank.values()), "control full bank")
    tests.append("control_reruns_profile_scale_search_and_all_finite_families")

    source_gate = protocol.absolute_source_gate(literal_gate)
    check(source_gate["controls_may_run"] is False and
          source_gate["control_subtraction_can_create_pass"] is False,
          "control cannot create pass")
    tests.append("absolute_source_gate_precedes_production_controls")

    bootstrap = protocol.paired_layer_bootstrap(
        {f"layer-{index}": 0.1 + 0.01 * index for index in range(5)},
        replicates=512)
    check(bootstrap["clusters"] == 5 and
          bootstrap["per_weight_iid_interval_used"] is False and
          bootstrap["lower_95"] <= bootstrap["point_mean"] <= bootstrap["upper_95"],
          "whole layer bootstrap")
    tests.append("paired_whole_layer_cluster_bootstrap")

    design = json.loads((PACKAGE / "design_lock.json").read_text("utf-8"))
    check(design["source_access"]["model_or_checkpoint_payload_opened_statted_hashed_enumerated"] is False and
          design["families"]["gf2_rank_plus_exceptions"]["global_negative_authority"] is False and
          design["coordinate_domain"]["invalid_domain_mask_is_free"] is False and
          design["claim_boundary"].startswith("Source-only"),
          "design claim boundary")
    tests.append("sealed_no_payload_no_claim_design_boundary")

    check(not any(name == "cupy" or name.startswith("cupy.")
                  for name in sys.modules), "CuPy not imported")
    tests.append("source_tests_do_not_initialize_cupy_or_cuda")

    print(json.dumps({
        "schema": "logic-q-label-flexible-algebraic-gate-v0-source-test-receipt",
        "status": "PASS_SOURCE_ONLY_TESTS",
        "test_count": len(tests),
        "tests": tests,
        "payload_accessed": False,
        "model_accessed": False,
        "network_accessed": False,
        "cupy_imported": False,
        "new_modules": sorted(set(sys.modules) - before),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
