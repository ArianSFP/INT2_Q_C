#!/usr/bin/env python3
"""Source-free tests for the frozen epsilon-TCQ/WFA early gate.

These tests use only deterministic synthetic values.  They never accept a
model/checkpoint/container path and deliberately do not import CuPy.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


legal = load("legal_interface")
core = load("tcq_core")
packet = load("packet_codec")
gate = load("gate_contract")


def expect_raises(action, exception=Exception):
    try:
        action()
    except exception:
        return
    raise AssertionError("expected exception")


def direct_fixture():
    adapter = legal.DirectFourLevelAdapter((-1.0, -0.25, 0.25, 1.0))
    nearest = (1, 2, 2, 1, 0, 3, 2, 1)
    targets = (-0.37, 0.42, 0.18, -0.11, -0.91, 0.88, 0.31, -0.22)
    trajectory = []
    legal_state = adapter.initial_state()
    for position, label in enumerate(nearest):
        rows = adapter.encode_choices(position, legal_state, label, 1)
        choice = [row for row in rows if row.label == label][0]
        trajectory.append(choice)
        legal_state = choice.next_legal_state
    candidate = core.ModelCandidate("suffix", 4, 32)
    model = core.fit_model(candidate, (trajectory, trajectory))
    samples = []
    state = 0
    for position, (choice, target) in enumerate(zip(trajectory, targets)):
        if position % candidate.reset == 0:
            state = 0
        pre_state = state
        for ordinal, (bit, context) in enumerate(zip(choice.event_bits,
                                                      choice.event_contexts)):
            state = core.transition(candidate, state, bit, context,
                                    position * 2 + ordinal)
        samples.append((0, pre_state, choice.label, target, choice.nominal))
    head = core.fit_centroid_head("nominal", 4, 4, samples)
    return adapter, targets, nearest, trajectory, model, head, samples


def test_interfaces_are_literal_and_non_aliasing():
    direct = legal.DirectFourLevelAdapter((-1.0, -0.25, 0.25, 1.0))
    strata = legal.SyntheticStrataLegalAdapter(0.125)
    assert legal.validate_adapter(direct)["events_per_label"] == 2
    assert legal.validate_adapter(strata)["events_per_label"] == 6
    d = direct.encode_choices(0, 0, 2, 1)[1]
    s = strata.encode_choices(0, 0, 2, 1)[1]
    legal.assert_no_interface_alias(s, d)
    assert d.nominal != s.nominal or d.event_bits != s.event_bits


def test_actual_strata_adapter_is_absent_and_hold_is_typed():
    hold = gate.missing_strata_adapter_hold()
    assert hold["status"].startswith("HOLD_NO_AUTHENTICATED")
    assert hold["qwen_payload_may_open"] is False
    assert hold["direct_four_level_fallback_allowed"] is False
    assert not hasattr(legal, "PolarisAdapter")


def test_frozen_bank_and_bounded_initializer():
    assert core.FROZEN_BANK == (
        ("suffix", 4, 32), ("suffix", 8, 128),
        ("xor_sketch", 8, 64), ("modular_ones", 8, 64),
        ("rolling_affine", 8, 128), ("signed_saturating", 8, 64),
    )
    result = core.bounded_hankel_cssr_diagnostic(
        tuple((index ^ (index >> 1)) & 1 for index in range(128)))
    assert result["maximum_rank"] == 16
    assert result["new_topology_created"] is False
    assert sorted(result["frozen_candidate_ranking"]) == list(range(6))


def test_exact_joint_search_dominates_fixed_nearest_objective():
    adapter, targets, nearest, _, model, head, _ = direct_fixture()
    fixed_bytes = packet.fixed_packet_bytes(adapter.interface, model, head)
    nearest_result = core.fixed_nearest_search(
        targets, nearest, adapter, model, head, rate_lambda=2.0 ** -8,
        fixed_packet_bytes=fixed_bytes)
    joint = core.search_labels(
        targets, nearest, adapter, model, head, epsilon=1,
        rate_lambda=2.0 ** -8, fixed_packet_bytes=fixed_bytes, exact=True)
    assert joint.exact is True
    assert joint.objective <= nearest_result.objective + 1e-15
    assert all(abs(a - b) <= 1 for a, b in zip(joint.labels, nearest))


def test_beam_is_decoder_replayable():
    adapter, targets, nearest, _, model, head, _ = direct_fixture()
    fixed_bytes = packet.fixed_packet_bytes(adapter.interface, model, head)
    result = core.search_labels(
        targets * 3, nearest * 3, adapter, model, head, epsilon=1,
        rate_lambda=2.0 ** -8, fixed_packet_bytes=fixed_bytes,
        exact=False, beam_width=32)
    decoded = core.decode_payload(
        len(result.labels), result.payload, result.logical_bits, adapter,
        model, head)
    assert decoded["labels"] == result.labels
    assert decoded["literal_payload_reencode_matches"] is True


def test_literal_packet_decode_reencode_and_complete_bytes():
    adapter, targets, nearest, _, model, head, _ = direct_fixture()
    fixed_bytes = packet.fixed_packet_bytes(adapter.interface, model, head)
    result = core.search_labels(
        targets, nearest, adapter, model, head, epsilon=1,
        rate_lambda=2.0 ** -8, fixed_packet_bytes=fixed_bytes, exact=True)
    blob = packet.build_packet(adapter.interface, model, head, result.labels,
                               result.payload, result.logical_bits)
    receipt = packet.decode_and_reencode(blob, adapter, core)
    assert receipt["literal_packet_reencode_matches"] is True
    assert receipt["labels"] == result.labels
    assert receipt["packet_bytes"] == result.physical_bytes
    assert receipt["model_bytes"] > 0 and receipt["centroid_bytes"] > 0
    damaged = bytearray(blob)
    damaged[-1] ^= 1
    expect_raises(lambda: packet.parse_packet(bytes(damaged), core))


def test_synthetic_six_event_packet_exercises_primary_abi_only():
    adapter = legal.SyntheticStrataLegalAdapter(0.125)
    nearest = (30, 31, 32, 33, 30, 34, 29, 35)
    targets = tuple(adapter.scale * (label - 31) + (0.01 if index & 1 else -0.01)
                    for index, label in enumerate(nearest))
    trajectory = []
    legal_state = adapter.initial_state()
    for position, label in enumerate(nearest):
        choice = [row for row in adapter.encode_choices(
            position, legal_state, label, 1) if row.label == label][0]
        trajectory.append(choice)
        legal_state = choice.next_legal_state
    model = core.fit_model(core.ModelCandidate("suffix", 4, 32), (trajectory,))
    head = core.CentroidHead("nominal", 4, 64, ())
    fixed = packet.fixed_packet_bytes(adapter.interface, model, head)
    result = core.search_labels(
        targets, nearest, adapter, model, head, epsilon=1,
        rate_lambda=2.0 ** -8, fixed_packet_bytes=fixed, exact=True)
    blob = packet.build_packet(adapter.interface, model, head, result.labels,
                               result.payload, result.logical_bits)
    receipt = packet.decode_and_reencode(blob, adapter, core)
    assert receipt["labels"] == result.labels
    assert receipt["literal_packet_reencode_matches"] is True
    parsed = packet.parse_packet(blob, core)
    assert parsed["interface"] == legal.STRATA_INTERFACE
    direct = legal.DirectFourLevelAdapter((-1.0, -0.25, 0.25, 1.0))
    expect_raises(lambda: packet.decode_and_reencode(blob, direct, core))


def test_exact_search_rejects_unbounded_epsilon_two_frontier():
    adapter, targets, nearest, _, model, head, _ = direct_fixture()
    fixed = packet.fixed_packet_bytes(adapter.interface, model, head)
    expect_raises(lambda: core.search_labels(
        targets, nearest, adapter, model, head, epsilon=2,
        rate_lambda=2.0 ** -8, fixed_packet_bytes=fixed, exact=True))


def test_centroid_heads_charge_all_cells_and_state_permutation():
    _, _, _, _, _, _, samples = direct_fixture()
    local = core.fit_centroid_head("local", 4, 4, samples)
    state = core.fit_centroid_head("state", 4, 4, samples)
    permuted = core.fit_centroid_head("state_permuted", 4, 4, samples)
    assert len(local.values) == 4
    assert len(state.values) == len(permuted.values) == 16
    assert packet.serialize_centroid(legal.DIRECT4_INTERFACE, state)
    assert state.correction(1, 2, 0) == state.values[6]
    assert math.isfinite(permuted.correction(1, 2, 7))


def test_owner_component_crossfit_has_no_stream_leakage():
    owners = ((0,), (1,), (2, 3), (3,), (4,))
    assert core.connected_owner_components(owners) == ((0,), (1,), (2, 3), (4,))
    folds = gate.owner_component_folds(owners, core)
    assert len(folds) == 4
    for fold in folds:
        held = set(fold["held_owner_component"])
        for stream in fold["development_stream_ordinals"]:
            assert held.isdisjoint(owners[stream])


def test_one_pass_page_ledger_and_separate_memory_domains():
    ledger = packet.owner_page_ledger(
        4096, (((0,), 65536), ((1,), 65536)), expert_count=2)
    assert ledger["strictly_below_2x"] is True
    assert ledger["external_storage_host_scratch_hbm_are_disjoint"] is True
    assert all(row["external_storage_passes"] == 1 and
               row["compressed_page_refetch_bytes"] == 0
               for row in ledger["experts"])


def complete_byte_ledger(total_bytes):
    return {
        "header_bytes": 32, "model_bytes": 24,
        "topology_bytes": 8, "frequency_bytes": 16,
        "centroid_bytes": 8, "directory_bytes": 8,
        "frame_header_bytes": 16, "payload_bytes": total_bytes - 88,
        "padding_bytes": 0, "total_bytes": total_bytes,
    }


def test_source_gate_requires_state_ablation_and_all_bytes():
    row = {
        "weights": 4096, "sse": 0.02, "source_energy": 1.0,
        "bytes": 1280, "nearest_bytes": 1280, "nearest_sse": 0.03,
        "state_gain_bpw": 0.08, "local_gain_bpw": 0.02,
        "permuted_gain_bpw": 0.01, "literal_reencode": True,
        "read_amplification": 1.25, "byte_ledger": complete_byte_ledger(1280),
    }
    result = gate.source_gate((row,))
    assert result["controls_may_open"] is True
    damaged = dict(row)
    damaged["state_gain_bpw"] = damaged["local_gain_bpw"]
    expect_raises(lambda: gate.source_gate((damaged,)))


def test_controls_are_closed_then_require_all_eight_full_pipelines():
    expect_raises(lambda: gate.final_control_gate(0.1, (), source_survived=False))
    controls = []
    for ordinal in range(8):
        controls.append({
            "ordinal": ordinal, "full_ptq_pipeline": True,
            "legal_trace_regenerated": True, "nested_selection_repeated": True,
            "literal_reencode": True, "gain_bpw": 0.04 + 0.001 * ordinal,
            "closure_sha256": hashlib.sha256(str(ordinal).encode()).hexdigest(),
        })
    passed = gate.final_control_gate(0.08, controls, source_survived=True)
    assert passed["status"].startswith("ELIGIBLE")
    killed = gate.final_control_gate(0.06, controls, source_survived=True)
    assert killed["status"].startswith("HARD_KILL")


def test_static_source_only_boundary():
    runner = (ROOT / "run_gate.py").read_text(encoding="utf-8")
    assert "--payload" not in runner and "--qwen" not in runner
    assert "import cupy" not in runner
    design = json.loads((ROOT / "design_lock.json").read_text(encoding="utf-8"))
    assert design["interfaces"]["primary_adapter_present_in_v0"] is False
    assert design["controls"]["open_only_after_source_survival"] is True


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(json.dumps({
        "schema": "epsilon-tcq-wfa-v0-source-test-receipt",
        "status": "PASS", "tests": len(tests),
        "qwen_payload_accessed": False, "current_codec_payload_accessed": False,
        "cupy_imported": "cupy" in sys.modules,
        "network_accessed": False,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
