#!/usr/bin/env python3
"""Hostile source-only tests; no Qwen/current packet/control payload is opened."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name):
    specification = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    if specification is None or specification.loader is None:
        raise RuntimeError(name)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


adapter_module = load("strata_replay_adapter")
packet_module = load("candidate_packet")
independent = load("independent_decoder")
oracle = load("polar_list_oracle")
driver = load("bound_driver")


def expect_raises(action, exception=Exception):
    try:
        action()
    except exception:
        return
    raise AssertionError("expected exception")


def pack_bits(bits):
    result = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        result[index >> 3] |= int(bit) << (7 - (index & 7))
    return bytes(result)


def polar_output(internal):
    n = len(internal)
    depth = int(math.log2(n))
    reverse = []
    for position in range(n):
        source, value = position, 0
        for _ in range(depth):
            value = (value << 1) | (source & 1)
            source >>= 1
        reverse.append(value)
    result = [internal[index] for index in reverse]
    stride = 1
    while stride < n:
        for base in range(0, n, 2 * stride):
            for offset in range(stride):
                result[base + offset] ^= result[base + stride + offset]
        stride *= 2
    return tuple(result)


def replay_fixture():
    n = 8
    decoder_source = b"source-free independent decoder fixture"
    levels = []
    receipt_levels = []
    planes = []
    event_material = bytearray()
    all_bits, all_freqs = [], []
    for level in range(6):
        mask = tuple(1 if position in ((level + 1) % n, (level + 4) % n) else 0
                     for position in range(n))
        internal = tuple((position * 3 + level + (position >> 1)) & 1
                         for position in range(n))
        bits = tuple(value for value, selected in zip(internal, mask, strict=True)
                     if selected)
        frequencies = tuple(20_000 + 1000 * level + 17 * index
                            for index in range(len(bits)))
        selected_packed = pack_bits(bits)
        frequency_bytes = struct.pack("<" + "H" * len(frequencies), *frequencies)
        mask_packed = pack_bits(mask)
        internal_packed = pack_bits(internal)
        plane = polar_output(internal)
        plane_packed = pack_bits(plane)
        planes.append(plane)
        level_artifacts = adapter_module.LevelArtifacts(
            selected_packed, len(bits), frequency_bytes, mask_packed,
            internal_packed, plane_packed)
        levels.append(level_artifacts)
        receipt_levels.append({
            "level": level, "selected_count": len(bits),
            "selected_bits_msb_sha256": hashlib.sha256(selected_packed).hexdigest(),
            "causal_frequencies_u16le_sha256": hashlib.sha256(frequency_bytes).hexdigest(),
            "selected_mask_msb_sha256": hashlib.sha256(mask_packed).hexdigest(),
            "internal_sc_bits_msb_sha256": hashlib.sha256(internal_packed).hexdigest(),
            "output_plane_msb_sha256": hashlib.sha256(plane_packed).hexdigest(),
        })
        all_bits.extend(bits)
        all_freqs.extend(frequencies)
        event_material.extend(struct.pack("<Q", len(bits)))
        event_material.extend(selected_packed)
        event_material.extend(frequency_bytes)
        event_material.extend(mask_packed)
        event_material.extend(internal_packed)
    indices = bytes(sum(planes[level][position] << level for level in range(6))
                    for position in range(n))
    payload, logical_bits = adapter_module.arithmetic_encode_binary(all_bits, all_freqs)
    reconstruction = struct.pack("<" + "d" * n,
                                 *(0.125 * (value - 31) for value in indices))
    current_packet = b"source-free literal current packet fixture"
    body = {
        "schema": adapter_module.REPLAY_SCHEMA,
        "status": "INDEPENDENT_CURRENT_CODEC_SIX_LEVEL_REPLAY",
        "interface": adapter_module.INTERFACE,
        "decoder_bytes": len(decoder_source),
        "decoder_sha256": hashlib.sha256(decoder_source).hexdigest(),
        "current_packet_bytes": len(current_packet),
        "current_packet_sha256": hashlib.sha256(current_packet).hexdigest(),
        "block_ordinal": 0, "block_values": n, "block_log2": 3,
        "logical_bits": logical_bits,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "indices_u8_sha256": hashlib.sha256(indices).hexdigest(),
        "reconstruction_f64le_sha256": hashlib.sha256(reconstruction).hexdigest(),
        "levels": receipt_levels,
        "event_stream_sha256": hashlib.sha256(bytes(event_material)).hexdigest(),
    }
    receipt = adapter_module.seal_replay_receipt(body)
    artifacts = adapter_module.ReplayArtifacts(
        decoder_source, current_packet, current_packet, payload, indices,
        reconstruction, reconstruction, tuple(levels))
    return receipt, artifacts


def test_exact_six_level_current_path_replay_and_scientific_interface_correction():
    receipt, artifacts = replay_fixture()
    result = adapter_module.ReadOnlyStrataReplayAdapter().validate(
        receipt, artifacts, expected_decoder_bytes=len(artifacts.decoder_source),
        expected_decoder_sha256=hashlib.sha256(artifacts.decoder_source).hexdigest(),
        allow_source_free_fixture_pin=True)
    assert result["status"].startswith("PASS_EXACT")
    assert result["sc_levels"] == 6 and result["coordinate_local_choice_api_available"] is False
    assert adapter_module.INTERFACE != adapter_module.LEGACY_INVALID_INTERFACE
    assert adapter_module.AUDITOR_SHA256 == "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"


def test_coordinate_local_choice_and_direct_int2_fallback_are_impossible():
    instance = adapter_module.ReadOnlyStrataReplayAdapter()
    expect_raises(lambda: instance.coordinate_choices(0, 0, 31, 1),
                  adapter_module.CoordinateLocalTransitionHold)
    assert instance.direct_int2_fallback is False
    source = (ROOT / "strata_replay_adapter.py").read_text(encoding="utf-8")
    assert "DirectFour" not in source and "direct_int2_4level_new_codec" not in source


def test_replay_rejects_polar_state_payload_and_cross_decoder_tampering():
    receipt, artifacts = replay_fixture()
    instance = adapter_module.ReadOnlyStrataReplayAdapter()
    kwargs = dict(expected_decoder_bytes=len(artifacts.decoder_source),
                  expected_decoder_sha256=hashlib.sha256(artifacts.decoder_source).hexdigest(),
                  allow_source_free_fixture_pin=True)
    broken_level = list(artifacts.levels)
    row = broken_level[0]
    damaged = bytearray(row.internal_sc_bits_msb)
    damaged[0] ^= 0x80
    broken_level[0] = adapter_module.LevelArtifacts(
        row.selected_bits_msb, row.selected_count, row.causal_frequencies_u16le,
        row.selected_mask_msb, bytes(damaged), row.output_plane_msb)
    expect_raises(lambda: instance.validate(
        receipt, adapter_module.ReplayArtifacts(
            artifacts.decoder_source, artifacts.current_packet,
            artifacts.independently_reencoded_current_packet, artifacts.payload,
            artifacts.indices_u8, artifacts.primary_reconstruction_f64le,
            artifacts.independent_reconstruction_f64le, tuple(broken_level)), **kwargs))
    expect_raises(lambda: instance.validate(
        receipt, adapter_module.ReplayArtifacts(
            artifacts.decoder_source, artifacts.current_packet,
            artifacts.current_packet + b"x", artifacts.payload, artifacts.indices_u8,
            artifacts.primary_reconstruction_f64le,
            artifacts.independent_reconstruction_f64le, artifacts.levels), **kwargs))
    expect_raises(lambda: instance.validate(
        receipt, adapter_module.ReplayArtifacts(
            artifacts.decoder_source, artifacts.current_packet,
            artifacts.independently_reencoded_current_packet, artifacts.payload + b"\0",
            artifacts.indices_u8, artifacts.primary_reconstruction_f64le,
            artifacts.independent_reconstruction_f64le, artifacts.levels), **kwargs))


WEIGHTS = 40_000
EXPERTS = 2


def packet_fixture(payload_byte=0x40):
    frames = (
        packet_module.FrameInput((0,), WEIGHTS // 2, bytes([31]) * (WEIGHTS // 2),
                                 bytes([payload_byte]) * 900, 7200),
        packet_module.FrameInput((1,), WEIGHTS // 2, bytes([32]) * (WEIGHTS // 2),
                                 bytes([payload_byte ^ 0x10]) * 900, 7200),
    )
    return packet_module.build_packet(
        topology=b"topology-v1", frequencies=b"frequency-table-v1",
        centroids=b"centroid-table-v1", frames=frames,
        weights=WEIGHTS, experts=EXPERTS)


def test_literal_packet_independent_decode_reencode_and_ledger_binding():
    raw = packet_fixture()
    parsed = packet_module.parse_packet(raw)
    decoded = independent.decode_and_reencode(raw)
    assert decoded["canonical_reencode_matches"] is True
    assert decoded["packet_sha256"] == parsed["packet_sha256"]
    assert parsed["total_bytes"] == parsed["byte_ledger"]["total_bytes"]
    ledger = parsed["byte_ledger"]
    assert ledger["model_bytes"] == ledger["topology_bytes"] + ledger["frequency_bytes"]
    damaged = bytearray(raw)
    damaged[-1] ^= 1
    expect_raises(lambda: independent.decode_and_reencode(bytes(damaged)))


def test_read_amplification_is_derived_from_literal_ranges():
    raw = packet_fixture()
    trace = packet_module.owner_read_trace(raw, 0)
    assert trace["compressed_expert_second_pass_count"] == 0
    assert trace["read_request_count"] == 2
    assert trace["cold_read_amplification"] < 2.0
    assert trace["requested_bytes"] == sum(
        row["end"] - row["begin"] for row in trace["ranges"])


def outer_plan_fixture():
    return driver.build_outer_plan(((0,), (1,)), EXPERTS)


def score_bytes(error):
    source = struct.pack("<" + "d" * WEIGHTS, *([1.0] * WEIGHTS))
    reconstruction = struct.pack("<" + "d" * WEIGHTS,
                                 *([1.0 - math.sqrt(error)] * WEIGHTS))
    return source, reconstruction


def fold_bundle(plan, fold, source_kind="QWEN", state_error=0.020):
    source, nearest = score_bytes(0.031)
    _, local = score_bytes(0.028)
    _, state = score_bytes(state_error)
    _, permuted = score_bytes(0.027)
    packets = {name: packet_fixture(0x40 + index)
               for index, name in enumerate(
                   ("nearest", "local", "state", "state_permuted"))}
    reconstructions = {"nearest": nearest, "local": local, "state": state,
                       "state_permuted": permuted}
    replay_sha = hashlib.sha256(f"replay-{source_kind}-{fold}".encode()).hexdigest()
    receipt = driver.artifact_receipt(
        source_kind=source_kind, outer_plan_sha256=plan["seal_sha256"],
        fold_ordinal=fold,
        held_stream_ordinals=plan["folds"][fold]["held_stream_ordinals"],
        adapter_replay_receipt_sha256=replay_sha,
        source_f64le=source, reconstructions=reconstructions, packets=packets)
    return {"receipt": receipt, "source_f64le": source,
            "reconstructions": reconstructions, "packets": packets,
            "routed_expert": fold}


def test_outer_fold_plan_is_exact_and_closes_every_fit_selection_input():
    plan = outer_plan_fixture()
    result = driver.validate_outer_plan(plan)
    assert result["folds"] == 2
    damaged = json.loads(json.dumps(plan))
    damaged["folds"][0]["topology_fit_stream_ordinals"] = [0]
    damaged = driver.seal({key: value for key, value in damaged.items()
                           if key != "seal_sha256"})
    expect_raises(lambda: driver.validate_outer_plan(damaged))


def test_fold_gains_scores_bytes_and_reads_are_all_derived():
    plan = outer_plan_fixture()
    bundle = fold_bundle(plan, 0)
    row = driver.derive_fold(bundle, plan, packet_module=packet_module,
                             independent_decoder=independent)
    assert row["checks"]["fold_pass"] is True
    assert row["gains_bpw"]["state"] > max(
        row["gains_bpw"]["local"], row["gains_bpw"]["state_permuted"])
    assert row["scores"]["state"]["bytes"] == row["scores"]["state"]["byte_ledger"]["total_bytes"]
    injected = dict(bundle)
    injected["read_amplification"] = 0.0
    expect_raises(lambda: driver.derive_fold(
        injected, plan, packet_module=packet_module,
        independent_decoder=independent))


def test_fold_rejects_artifact_and_packet_hash_tampering():
    plan = outer_plan_fixture()
    bundle = fold_bundle(plan, 0)
    damaged = dict(bundle)
    damaged["reconstructions"] = dict(bundle["reconstructions"])
    damaged["reconstructions"]["state"] += b"\0" * 8
    expect_raises(lambda: driver.derive_fold(
        damaged, plan, packet_module=packet_module,
        independent_decoder=independent))
    damaged = dict(bundle)
    damaged["packets"] = dict(bundle["packets"])
    raw = bytearray(damaged["packets"]["state"])
    raw[-1] ^= 1
    damaged["packets"]["state"] = bytes(raw)
    expect_raises(lambda: driver.derive_fold(
        damaged, plan, packet_module=packet_module,
        independent_decoder=independent))


def test_full_source_panel_requires_every_outer_fold():
    plan = outer_plan_fixture()
    bundles = [fold_bundle(plan, fold) for fold in range(2)]
    panel = driver.derive_panel(
        bundles, plan, packet_module=packet_module,
        independent_decoder=independent, require_target_pass=True)
    assert panel["controls_may_open"] is True and panel["all_outer_folds_pass"] is True
    expect_raises(lambda: driver.derive_panel(
        bundles[:1], plan, packet_module=packet_module,
        independent_decoder=independent, require_target_pass=True))


def test_control_receipts_are_externally_pinned_not_assertion_booleans():
    plan = outer_plan_fixture()
    source_bundles = [fold_bundle(plan, fold) for fold in range(2)]
    source_panel = driver.derive_panel(
        source_bundles, plan, packet_module=packet_module,
        independent_decoder=independent, require_target_pass=True)
    controls, pins = [], []
    for ordinal in range(8):
        bundles = [fold_bundle(plan, fold, "MATCHED_GAUSSIAN_FULL_PTQ", 0.024)
                   for fold in range(2)]
        receipt = driver.control_receipt(
            ordinal=ordinal, outer_plan_sha256=plan["seal_sha256"],
            pipeline_source_root_sha256=hashlib.sha256(
                f"pipeline-{ordinal}".encode()).hexdigest(),
            source_producer_receipt_sha256=hashlib.sha256(
                f"producer-{ordinal}".encode()).hexdigest(),
            legal_trace_panel_receipt_sha256=hashlib.sha256(
                f"trace-{ordinal}".encode()).hexdigest(),
            fold_artifact_receipt_sha256=[row["receipt"]["seal_sha256"]
                                          for row in bundles])
        controls.append({"receipt": receipt, "outer_plan": plan,
                         "fold_bundles": bundles})
        pins.append(hashlib.sha256(driver.canonical_json(receipt)).hexdigest())
    result = driver.final_control_gate(
        source_panel, controls, pins, packet_module=packet_module,
        independent_decoder=independent)
    assert len(result["control_receipt_sha256"]) == 8
    damaged = list(pins)
    damaged[3] = hashlib.sha256(b"wrong pin").hexdigest()
    expect_raises(lambda: driver.final_control_gate(
        source_panel, controls, damaged, packet_module=packet_module,
        independent_decoder=independent))
    assert "full_ptq_pipeline" not in driver.canonical_json(controls[0]["receipt"]).decode()


def test_tiny_oracle_searches_whole_polar_block_and_exposes_coupling():
    n = 8
    masks = [[0] * n for _ in range(6)]
    masks[0][1] = 1
    frozen = [[0] * n for _ in range(6)]
    result = oracle.tiny_six_level_oracle(
        [1] * n, masks, frozen, maximum_information_bits=1)
    assert result["search_unit"] == "whole_six_level_polar_block"
    assert result["coordinate_local_search"] is False
    # One internal information decision affects multiple output coordinates.
    assert sum(result["selected_indices"]) > 1


def test_production_2pow21_beam_is_hard_held_by_memory_gate():
    gate = oracle.production_gate(
        1 << 21, 32, memory_cap_bytes=4 * (1 << 30),
        cupy_topk_wired=True, device_resident_polar_state=True,
        bounded_prefix_storage=True)
    assert gate["status"] == "HOLD_PRODUCTION_POLAR_LIST_SCALABILITY"
    assert gate["estimate"]["total_peak_bytes_lower_bound"] > gate["memory_cap_bytes"]
    assert gate["qwen_payload_may_open"] is False


def test_cupy_topk_must_be_device_resident_wired_and_deterministic():
    class FakeCuPy:
        is_cupy = True
        device_resident = True

        def __init__(self):
            self.calls = 0

        def topk(self, costs, count):
            self.calls += 1
            return sorted(range(len(costs)), key=lambda index: (costs[index], index))[:count]

    backend = FakeCuPy()
    assert oracle.cupy_topk((3.0, 1.0, 2.0, 1.0), 2, backend) == (1, 3)
    assert backend.calls == 1
    backend.device_resident = False
    expect_raises(lambda: oracle.cupy_topk((1.0, 2.0), 1, backend))


def test_static_runner_has_no_payload_authority():
    runner = (ROOT / "run_gate.py").read_text(encoding="utf-8")
    assert "--payload" not in runner and "--qwen" not in runner
    assert "direct_int2" not in runner.lower()


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(json.dumps({
        "schema": "epsilon-tcq-strata-bound-v1-source-test-receipt",
        "status": "PASS", "tests": len(tests),
        "qwen_payload_accessed": False,
        "current_codec_payload_accessed": False,
        "matched_control_payload_accessed": False,
        "network_accessed": False,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
