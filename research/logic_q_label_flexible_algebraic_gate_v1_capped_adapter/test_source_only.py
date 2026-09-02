#!/usr/bin/env python3
"""Hostile source-only tests for the capped LOGIC-Q v1 adapter."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parent
PARENT = PACKAGE.parent / "logic_q_label_flexible_algebraic_gate_v0"


class FakeCupy:
    """NumPy-backed API shim used only to execute the batched code path."""
    __name__ = "cupy"

    @staticmethod
    def asnumpy(value):
        return np.asarray(value)

    def __getattr__(self, name):
        return getattr(np, name)


def load_adapter():
    spec = importlib.util.spec_from_file_location("logicq_v1_adapter_test",
                                                  PACKAGE / "capped_adapter.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("adapter import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check(condition: bool, name: str, tests: list[str]) -> None:
    if not condition:
        raise RuntimeError(name)
    tests.append(name)


def raises(function, fragment: str, name: str, tests: list[str]) -> None:
    try:
        function()
    except Exception as exc:  # hostile boundary intentionally broad
        if fragment not in str(exc):
            raise RuntimeError(f"{name}: wrong error {exc}") from exc
    else:
        raise RuntimeError(f"{name}: accepted")
    tests.append(name)


def synthetic_role(rows: int, cols: int, phase: float) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(rows * cols, dtype=np.float64)
    values = (0.73 * np.sin(index * 0.071 + phase) +
              0.21 * np.cos(index * 0.019 - phase))
    weights = 0.8 + (index % 11) / 20.0
    return values, weights


def panel_rows(adapter, rows: int = 4, cols: int = 256):
    result = []
    for layer_index in range(10):
        layer = f"layer-{layer_index:02d}"
        for slot_index in range(4):
            slot = f"expert-{slot_index:02d}"
            for role in adapter.ROLE_ORDER:
                digest = hashlib.sha256(
                    f"{layer}:{slot}:{role}:{rows}:{cols}".encode("ascii")).hexdigest()
                result.append(adapter.PanelRow(layer, slot, role, rows, cols, digest))
    return result


def metrics(adapter):
    result = {"train": {}, "validation": {}}
    for ordinal, config in enumerate(adapter.FROZEN_CONFIGS):
        for partition, offset in (("train", 0.0), ("validation", 0.001)):
            result[partition][config.config_id] = {
                "physical_bits": float(2_300_000 + ordinal * 10_000),
                "weights": 1_000_000.0,
                "weighted_sse": float(30_000 + ordinal * 100 + offset),
                "source_energy": 1_000_000.0,
                "expert_count": 8.0,
            }
    return result


def gauge_swap_packet(adapter, core, packet: bytes) -> bytes:
    record, scales, payload = core.parse_component_envelope(packet)
    if record.family != core.FAMILY_GF2 or record.parameter < 2:
        raise RuntimeError("GF2 rank-two fixture")
    rank = record.parameter
    reader = core.BitReader(payload, record.payload_bits)
    shapes = ((record.rows, rank), (rank, record.cols),
              (record.rows, rank), (rank, record.cols))
    matrices = []
    for shape in shapes:
        bits = [reader.read(1) for _ in range(shape[0] * shape[1])]
        matrices.append(np.asarray(bits, dtype=np.uint8).reshape(shape))
    matrices[0][:, [0, 1]] = matrices[0][:, [1, 0]]
    matrices[1][[0, 1], :] = matrices[1][[1, 0], :]
    writer = core.BitWriter()
    for matrix in matrices:
        for value in matrix.reshape(-1):
            writer.write(int(value), 1)
    while reader.position < reader.bit_length:
        writer.write(reader.read(1), 1)
    reader.finish()
    return core.component_packet(record, scales, writer.finish())


def main() -> None:
    adapter = load_adapter()
    core = adapter.load_parent_core(PARENT)
    tests: list[str] = []

    parent_receipt = adapter.verify_parent_package(PARENT)
    check(parent_receipt["manifest_sha256"] == adapter.PARENT_MANIFEST_SHA256,
          "pinned_parent_full_member_verification", tests)
    check(adapter.frozen_grid_record()["sha256"] ==
          adapter.frozen_grid_record()["sha256"],
          "frozen_grid_deterministic", tests)

    config = adapter.FROZEN_CONFIGS[0]
    adapter.validate_config(config, 4, 256)
    check(config.rm_word_shortlist <= adapter.MAX_RM_WORD_SHORTLIST and
          config.rm_pair_keep <= adapter.MAX_RM_PAIR_KEEP and
          config.rm_exception_cap <= adapter.MAX_RM_EXCEPTIONS and
          config.romdd_exception_cap <= adapter.MAX_ROMDD_EXCEPTIONS,
          "finite_qwen_pilot_caps", tests)
    bad = dataclasses.replace(config, rm_word_shortlist=16)
    raises(lambda: adapter.validate_config(bad, 4, 256),
           "frozen grid", "unfrozen_cap_rejected", tests)
    raises(lambda: adapter.require_live_cupy(np, True), "CuPy",
           "live_requires_injected_cupy", tests)

    called = []
    killed, receipt = adapter.execute_if_survives(
        {"hard_kill": True, "search_invoked": False},
        lambda: called.append(True))
    check(killed is None and not called and receipt["search_invoked"] is False,
          "executable_hard_kill_precedes_search", tests)

    costs = np.arange(256 * 4, dtype=np.float64).reshape(256, 4) / 1000.0
    base = np.zeros(256, dtype=np.uint8)
    plan = adapter._fast_exceptions(np, core, costs, base, 0.01, 16,
                                    fixed_prefix_bits=18)
    check(plan.count <= 16, "exception_search_strict_cap", tests)
    raises(lambda: adapter._fast_exceptions(
        np, core, costs, base, 0.01, 65, fixed_prefix_bits=18),
        "capped exception", "exception_search_unbounded_rejected", tests)

    role_values, role_weights = synthetic_role(4, 256, 0.1)
    shortlist = adapter.scale_shortlists(
        np, core, role_values, role_weights, block_size=256,
        profile=0, keep=2)
    check(len(shortlist) == 4 and all(len(row) == 2 for row in shortlist),
          "frozen_two_scale_shortlist", tests)

    original_pair = core._rm_base_cost_matrix
    core._rm_base_cost_matrix = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("full pair matrix forbidden"))
    try:
        rm = adapter.encode_rm1_capped(
            np, core, role_values, role_weights, role="gate", rows=4, cols=256,
            config=config)
    finally:
        core._rm_base_cost_matrix = original_pair
    check(rm.diagnostics["full_rm_pair_matrix_built"] is False and
          rm.diagnostics["scale_and_labels_jointly_scored"] is True,
          "rm_capped_no_full_pair_and_joint_scale_labels", tests)
    words = adapter._rm_words(np, 256)
    batch_plans, batch_costs, batch_reconstruction = adapter._rm_batch_for_scales(
        np, np, core, role_values.reshape(-1, 256)[:2],
        role_weights.reshape(-1, 256)[:2],
        [shortlist[0][0], shortlist[1][0]], profile=0, words=words,
        lambda_per_bit=0.01, config=config)
    check(len(batch_plans) == 2 and batch_costs.shape == (2, 256, 4) and
          batch_reconstruction.shape == (2, 256, 4),
          "batched_accelerator_search_shape_and_pair_gather", tests)
    fake_cupy_rm = adapter.encode_rm1_capped(
        FakeCupy(), core, role_values, role_weights, role="gate",
        rows=4, cols=256, config=config)
    check(fake_cupy_rm.diagnostics["cupy_backend"] is True and
          fake_cupy_rm.diagnostics["batched_accelerator_blocks"] == 64,
          "production_batched_backend_branch_executes", tests)
    check(max(plan.exceptions.count for plan in [
        adapter._rm_plan_for_costs(
            np, core, costs[:256], lambda_per_bit=0.01,
            word_shortlist=8, pair_keep=4, exception_cap=16,
            words=adapter._rm_words(np, 256))]) <= 16,
          "rm_exception_cap_operational", tests)

    # Canonical GF(2) gauge is enforced even though GF(2) search is not scheduled.
    tiny_values, tiny_weights = synthetic_role(3, 4, 0.3)
    gf2 = core.encode_gf2_component(
        np, tiny_values, tiny_weights, role="gate", rows=3, cols=4,
        block_size=4, lambda_per_bit=0.001, ranks=(2,), profile=0,
        exception_limit=2, exact_factor_pair_max=1, heuristic_sweeps=0)
    canonical_gf2 = adapter.canonicalize_component(np, core, gf2.packet)
    check(adapter.canonicalize_component(np, core, canonical_gf2) == canonical_gf2,
          "gf2_canonicalization_idempotent", tests)
    swapped = gauge_swap_packet(adapter, core, canonical_gf2)
    check(core.decode_component(np, swapped)[0] ==
          core.decode_component(np, canonical_gf2)[0],
          "gf2_gauge_tamper_preserves_labels", tests)
    raises(lambda: adapter.decode_canonical_component(np, core, swapped),
           "noncanonical", "gf2_gauge_tamper_rejected", tests)

    romdd = core.encode_romdd_component(
        np, role_values, role_weights, role="gate", rows=4, cols=256,
        block_size=256, lambda_per_bit=0.001, depths=(2,), profile=0,
        exception_limit=2)
    canonical_romdd = adapter.canonicalize_component(np, core, romdd.packet)
    record, scales, payload = core.parse_component_envelope(canonical_romdd)
    tampered_depth = core.component_packet(
        dataclasses.replace(record, parameter=(record.parameter + 1)), scales, payload)
    check(core.decode_component(np, tampered_depth)[0] ==
          core.decode_component(np, canonical_romdd)[0],
          "romdd_depth_tamper_preserves_labels", tests)
    raises(lambda: adapter.decode_canonical_component(np, core, tampered_depth),
           "noncanonical", "romdd_depth_header_tamper_rejected", tests)

    roles = {role: synthetic_role(4, 256, 0.2 + ordinal)
             for ordinal, role in enumerate(adapter.ROLE_ORDER)}
    encoded = adapter.encode_expert(np, core, roles, rows=4, cols=256,
                                    config=config, live=False)
    score = encoded["score"]
    check(score["component_or_per_role_gate_used"] is False and
          score["all_headers_alignment_and_final_page_charged"] is True,
          "pooled_three_role_page_physical_gate", tests)
    check(score["cold_read_amplification"] == 1.0 and
          score["cold_read_below_2x"] is True and score["read_passes"] == 1,
          "one_pass_read_ledger_below_2x", tests)
    check(score["gf2_search"].startswith("NOT_SCHEDULED") and
          score["romdd_scale_search"].startswith("NEAREST_SCALE"),
          "bounded_negative_claims_literal", tests)
    check(encoded["packet"] == adapter.pack_canonical_expert(
        np, core, adapter._expert_component_slices(core, encoded["packet"])),
        "canonical_expert_exact_reencode", tests)

    mismatch = {}
    for role in adapter.ROLE_ORDER:
        shape = (8, 128) if role == "down_transposed" else (4, 256)
        values, weights = synthetic_role(*shape, 0.7 + adapter.ROLE_ORDINAL[role])
        component = core.encode_literal_component(
            np, values, weights, role=role, rows=shape[0], cols=shape[1],
            block_size=256)
        mismatch[role] = component.packet
    raises(lambda: adapter.pack_canonical_expert(np, core, mismatch),
           "shape equality", "swiglu_role_shape_mismatch_rejected", tests)

    panel = adapter.panel_record(panel_rows(adapter))
    check([row["component_ordinal"] for row in panel["rows"]] ==
          list(range(len(panel["rows"]))),
          "canonical_panel_and_control_ordinals", tests)
    check(panel["partition_component_counts"] ==
          {"train": 45, "validation": 15, "test": 60},
          "whole_layer_and_slot_split", tests)
    receipt = adapter.selection_receipt(panel, metrics(adapter))
    selected = adapter.authorize_test(panel, receipt)
    check(selected.config_id == receipt["selected_config_id"] and
          receipt["test_metrics_opened_or_accepted"] is False,
          "executable_train_validation_selection_receipt", tests)
    with_test = metrics(adapter)
    with_test["test"] = with_test["validation"]
    raises(lambda: adapter.selection_receipt(panel, with_test),
           "no test metrics", "test_metrics_forbidden_during_selection", tests)
    tampered_receipt = dict(receipt)
    tampered_receipt["selected_config_id"] = adapter.FROZEN_CONFIGS[-1].config_id
    raises(lambda: adapter.authorize_test(panel, tampered_receipt),
           "receipt seal", "selection_receipt_tamper_rejected", tests)

    first_source = roles["gate"][0]
    ordinal = adapter.control_ordinal(panel, "layer-00", "expert-00", "gate")
    control_a, control_receipt_a = adapter.moment_matched_gaussian(
        np, first_source, block_size=256, seed=adapter.CONTROL_SEEDS[0],
        component_ordinal=ordinal)
    control_b, control_receipt_b = adapter.moment_matched_gaussian(
        np, first_source, block_size=256, seed=adapter.CONTROL_SEEDS[0],
        component_ordinal=ordinal)
    check(np.array_equal(control_a, control_b) and
          control_receipt_a == control_receipt_b,
          "matched_control_deterministic", tests)
    source_blocks = first_source.reshape(-1, 256)
    control_blocks = control_a.reshape(-1, 256)
    moment_ok = True
    for source_block, control_block in zip(source_blocks, control_blocks):
        moment_ok &= abs(float(source_block.mean()) - float(control_block.mean())) < 1e-12
        moment_ok &= abs(float(np.sum((source_block-source_block.mean())**2)) -
                         float(np.sum((control_block-control_block.mean())**2))) < 1e-9
    check(moment_ok, "matched_control_block_moments", tests)

    controls = adapter.rerun_matched_controls(
        np, np, core, roles, layer="layer-00", slot="expert-00",
        rows=4, cols=256, config=config, panel=panel, live=False)
    check(len(controls["results"]) == 8 and
          controls["full_capped_pipeline_rerun_including_presearch_decisions"] is True and
          all(row["score"]["all_headers_alignment_and_final_page_charged"]
              for row in controls["results"]),
          "all_eight_controls_full_capped_pipeline_rerun", tests)

    # A modified parent manifest is rejected before import.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for path in PARENT.iterdir():
            (root / path.name).write_bytes(path.read_bytes())
        manifest = bytearray((root / "SOURCE_MANIFEST.json").read_bytes())
        manifest[-2] ^= 1
        (root / "SOURCE_MANIFEST.json").write_bytes(manifest)
        raises(lambda: adapter.verify_parent_package(root), "external pin",
               "parent_manifest_tamper_rejected", tests)

    check("cupy" not in sys.modules, "source_tests_do_not_initialize_cupy", tests)
    result = {
        "schema": "logic-q-v1-capped-adapter-source-tests",
        "status": "PASS_SOURCE_ONLY_TESTS",
        "test_count": len(tests), "tests": tests,
        "parent_manifest_sha256": adapter.PARENT_MANIFEST_SHA256,
        "model_or_payload_accessed": False, "cupy_initialized": False,
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
