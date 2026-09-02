#!/usr/bin/env python3
"""Hostile source-only tests for the LOGIC-Q v2 production bindings."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parent
V1_PACKAGE = PACKAGE.parent / "logic_q_label_flexible_algebraic_gate_v1_capped_adapter"
V0_PACKAGE = PACKAGE.parent / "logic_q_label_flexible_algebraic_gate_v0"


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"import spec {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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
            raise RuntimeError(f"{name}: wrong error: {exc}") from exc
    else:
        raise RuntimeError(f"{name}: accepted")
    tests.append(name)


def source_values(layer: str, slot: str, role: str, count: int) -> np.ndarray:
    material = hashlib.sha256(f"{layer}:{slot}:{role}".encode("utf-8")).digest()
    phase = int.from_bytes(material[:4], "big") / 2.0**32
    index = np.arange(count, dtype=np.float64)
    return (0.71 * np.sin(index * 0.037 + phase) +
            0.19 * np.cos(index * 0.113 - phase) + 0.001 * phase)


def source_blob(layer: str, slot: str, role: str, count: int) -> bytes:
    return np.asarray(source_values(layer, slot, role, count),
                      dtype="<f8").tobytes(order="C")


def panel_rows(binder, rows: int, cols: int):
    result = []
    count = rows * cols
    for layer_index in range(10):
        layer = f"fixture-layer-{layer_index:02d}"
        for slot_index in range(4):
            slot = f"fixture-expert-{slot_index:02d}"
            for role in binder.ROLE_ORDER:
                blob = source_blob(layer, slot, role, count)
                result.append(binder.PanelRow(
                    layer, slot, role, rows, cols,
                    hashlib.sha256(blob).hexdigest(), "float64-le"))
    return result


def literal_expert(v1, core, rows: int, cols: int,
                   layer: str, slot: str):
    components = {}
    sources = {}
    count = rows * cols
    for role in v1.ROLE_ORDER:
        values = source_values(layer, slot, role, count)
        sources[role] = np.asarray(values, dtype="<f8").tobytes(order="C")
        encoded = core.encode_literal_component(
            np, values, np.ones(count, dtype=np.float64), role=role,
            rows=rows, cols=cols, block_size=256)
        components[role] = encoded.packet
    return v1.pack_canonical_expert(np, core, components), sources


def fabricated_selection_rows(binder, v1, panel, packet_receipt):
    rows_by_config = {}
    for config_index, config in enumerate(v1.FROZEN_CONFIGS):
        scored = []
        for panel_row in panel["rows"]:
            if panel_row["partition"] == "test":
                continue
            role = panel_row["role"]
            component = packet_receipt["components"][role]
            base = {
                "schema": "logic-q-v2-independent-scored-row-v1",
                "config_id": config.config_id,
                "layer": panel_row["layer"], "slot": panel_row["slot"],
                "role": role, "partition": panel_row["partition"],
                "component_ordinal": panel_row["component_ordinal"],
                "panel_source_sha256": panel_row["source_sha256"],
                "source_dtype": panel_row["source_dtype"],
                "source_blob_bytes": (panel_row["rows"] * panel_row["cols"] * 8),
                "expert_packet_sha256": packet_receipt["expert_packet_sha256"],
                "expert_packet_bytes": packet_receipt["expert_packet_bytes"],
                "packet_geometry_sha256":
                    packet_receipt["packet_geometry_sha256"],
                "component_packet_sha256": component["packet_sha256"],
                "component_packet_bytes": component["packet_bytes"],
                "decoded_source_count": component["source_count"],
                "raw_sse_f64_hex": (0.01 + 0.01 * config_index +
                                    1e-9 * panel_row["component_ordinal"]).hex(),
                "raw_energy_f64_hex": (1.0 +
                                       1e-8 * panel_row["component_ordinal"]).hex(),
                "scorer_schema": "logic-q-v2-independent-source-scorer-v1",
            }
            scored.append(binder.seal_scored_row(base))
        rows_by_config[config.config_id] = scored
    return rows_by_config


def reseal_receipt(binder, receipt):
    result = copy.deepcopy(receipt)
    unsigned = dict(result)
    unsigned.pop("receipt_sha256", None)
    result["receipt_sha256"] = binder.sha256(binder.canonical_json(unsigned))
    return result


def main() -> None:
    binder = load_file("logicq_v2_bound_test", PACKAGE / "bound_adapter.py")
    scorer = load_file("logicq_v2_scorer_test", PACKAGE / "independent_scorer.py")
    v1 = binder.load_v1(V1_PACKAGE)
    core = binder.load_v0(v1, V0_PACKAGE)
    tests: list[str] = []

    v1_receipt = binder.verify_source_dependency(
        V1_PACKAGE, expected_manifest_sha256=binder.V1_MANIFEST_SHA256,
        expected_source_root_sha256=binder.V1_SOURCE_ROOT_SHA256)
    check(v1_receipt["source_root_sha256"] == binder.V1_SOURCE_ROOT_SHA256,
          "pinned_v1_exact_dependency", tests)
    v0_receipt = binder.verify_source_dependency(
        V0_PACKAGE, expected_manifest_sha256=binder.V0_MANIFEST_SHA256,
        expected_source_root_sha256=binder.V0_SOURCE_ROOT_SHA256)
    check(v0_receipt["source_root_sha256"] == binder.V0_SOURCE_ROOT_SHA256,
          "pinned_v0_exact_dependency", tests)
    check(v1.frozen_grid_record()["sha256"] ==
          binder.canonical_copy(v1.frozen_grid_record())["sha256"],
          "frozen_capped_algorithm_grid_unchanged", tests)

    # This geometry makes the literal three-role page packet 13 pages and
    # 2.1666... bpw, so authorization exercises the real [2.15, 2.5] gate.
    rows, cols = 256, 256
    panel = binder.make_panel_record(panel_rows(binder, rows, cols))
    panel_sha = panel["panel_sha256"]
    binder.validate_panel_record(panel, expected_panel_sha256=panel_sha)
    check([row["component_ordinal"] for row in panel["rows"]] ==
          list(range(len(panel["rows"]))) and
          panel["partition_component_counts"] ==
          {"train": 45, "validation": 15, "test": 60},
          "canonical_control_ordinals_and_whole_partitions", tests)

    panel_tamper = copy.deepcopy(panel)
    panel_tamper["rows"][0]["source_sha256"] = "f" * 64
    panel_unsigned = dict(panel_tamper)
    panel_unsigned.pop("panel_sha256")
    panel_tamper["panel_sha256"] = binder.sha256(
        binder.canonical_json(panel_unsigned))
    raises(lambda: binder.validate_panel_record(
        panel_tamper, expected_panel_sha256=panel_sha),
        "external pin", "public_panel_reseal_cannot_override_external_pin", tests)
    bad_rows = panel_rows(binder, rows, cols)
    bad = bad_rows[-1]
    bad_rows[-1] = binder.PanelRow(
        bad.layer, bad.slot, bad.role, 512, 128, bad.source_sha256,
        bad.source_dtype)
    raises(lambda: binder.make_panel_record(bad_rows), "shape equality",
           "gate_up_downT_shape_mismatch_rejected", tests)

    target = next(row for row in panel["rows"]
                  if row["role"] == "gate" and row["partition"] == "train")
    packet, blobs = literal_expert(
        v1, core, rows, cols, target["layer"], target["slot"])
    geometry = binder.packet_geometry(np, v1, core, packet)
    binder.validate_packet_geometry_receipt(core, geometry)
    check(geometry["expert_weights_from_headers"] == 3 * rows * cols and
          geometry["physical_bits"] == len(packet) * 8 and
          geometry["cold_read_amplification"] == 1.0 and
          geometry["cold_read_below_2x"] is True,
          "packet_header_count_and_one_pass_ledger", tests)
    check(all(geometry["components"][role]["source_count"] == rows * cols
              for role in binder.ROLE_ORDER),
          "component_counts_derived_from_literal_headers", tests)

    geometry_attack = copy.deepcopy(geometry)
    geometry_attack["components"]["gate"]["source_count"] += 1
    geometry_unsigned = dict(geometry_attack)
    geometry_unsigned.pop("packet_geometry_sha256")
    geometry_attack["packet_geometry_sha256"] = binder.sha256(
        binder.canonical_json(geometry_unsigned))
    raises(lambda: binder.validate_packet_geometry_receipt(core, geometry_attack),
           "header-derived source_count",
           "public_packet_count_reseal_rejected_by_header", tests)

    score = scorer.score_expert_packet(
        binder, v1, core, packet=packet, source_blobs=blobs, panel=panel,
        expected_panel_sha256=panel_sha, layer=target["layer"],
        slot=target["slot"], config_id=v1.FROZEN_CONFIGS[0].config_id)
    check(score["packet_receipt"] == geometry and
          score["pooled"]["expert_weights_from_headers"] == 3 * rows * cols and
          score["encoder_metric_objects_used"] is False,
          "independent_source_score_uses_packet_and_raw_bytes", tests)
    decoded = v1.unpack_canonical_expert(np, core, packet)
    expected_sse = 0.0
    expected_energy = 0.0
    for role in binder.ROLE_ORDER:
        source = np.frombuffer(blobs[role], dtype="<f8")
        reconstruction = np.asarray(decoded[role][1], dtype=np.float64)
        expected_sse += float(np.sum((source - reconstruction) ** 2,
                                     dtype=np.float64))
        expected_energy += float(np.sum(source ** 2, dtype=np.float64))
    check(float.fromhex(score["pooled"]["raw_sse_f64_hex"]) == expected_sse and
          float.fromhex(score["pooled"]["raw_energy_f64_hex"]) == expected_energy,
          "independent_scorer_recomputes_exact_raw_fp64_metrics", tests)
    signature = inspect.signature(scorer.score_expert_packet)
    check("weighted_sse" not in signature.parameters and
          "components" not in signature.parameters and
          "np" not in signature.parameters,
          "encoder_metric_and_backend_injection_interfaces_absent", tests)
    tampered_blobs = dict(blobs)
    tampered_blobs["gate"] = bytes([blobs["gate"][0] ^ 1]) + blobs["gate"][1:]
    raises(lambda: scorer.score_expert_packet(
        binder, v1, core, packet=packet, source_blobs=tampered_blobs,
        panel=panel, expected_panel_sha256=panel_sha,
        layer=target["layer"], slot=target["slot"],
        config_id=v1.FROZEN_CONFIGS[0].config_id),
        "authenticated source blob", "source_byte_tamper_rejected", tests)
    bf16_words = np.asarray([0x3F80, 0xC000, 0x0000], dtype="<u2")
    bf16_values = scorer._decode_source(
        np, bf16_words.tobytes(), "bf16-le", 3)
    check(np.array_equal(bf16_values, np.asarray([1.0, -2.0, 0.0])),
          "literal_little_endian_bf16_source_parser", tests)

    packet_receipts = {geometry["expert_packet_sha256"]: geometry}
    selection_rows = fabricated_selection_rows(binder, v1, panel, geometry)
    receipt = binder.make_selection_receipt(
        v1, core, panel, expected_panel_sha256=panel_sha,
        rows_by_config=selection_rows,
        packet_receipts_by_sha256=packet_receipts)
    selected = binder.authorize_test(
        v1, core, panel, receipt, expected_panel_sha256=panel_sha,
        expected_receipt_sha256=receipt["receipt_sha256"])
    check(selected.config_id == v1.FROZEN_CONFIGS[0].config_id,
          "selector_recomputed_from_literal_rows", tests)
    portable = json.loads(binder.canonical_json(receipt).decode("ascii"))
    selected_portable = binder.authorize_test(
        v1, core, panel, portable, expected_panel_sha256=panel_sha,
        expected_receipt_sha256=portable["receipt_sha256"])
    check(selected_portable.config_id == selected.config_id,
          "selection_receipt_json_roundtrip_portable", tests)
    validation_experts = panel["partition_component_counts"]["validation"] // 3
    selected_metrics = receipt["derived_metrics"][selected.config_id]["validation"]
    check(selected_metrics["weights"] == validation_experts * 3 * rows * cols and
          selected_metrics["physical_bits"] ==
          validation_experts * geometry["physical_bits"],
          "pooled_packet_bits_once_and_header_weights_only", tests)

    selected_attack = copy.deepcopy(receipt)
    selected_attack["selected_config_id"] = v1.FROZEN_CONFIGS[-1].config_id
    selected_attack = reseal_receipt(binder, selected_attack)
    raises(lambda: binder.authorize_test(
        v1, core, panel, selected_attack, expected_panel_sha256=panel_sha,
        expected_receipt_sha256=selected_attack["receipt_sha256"]),
        "recomputed selected config",
        "public_selected_config_reseal_attack_rejected", tests)

    aggregate_attack = copy.deepcopy(receipt)
    aggregate_attack["derived_metrics"][selected.config_id]["validation"]["F"] = 0.0
    aggregate_attack = reseal_receipt(binder, aggregate_attack)
    raises(lambda: binder.authorize_test(
        v1, core, panel, aggregate_attack, expected_panel_sha256=panel_sha,
        expected_receipt_sha256=aggregate_attack["receipt_sha256"]),
        "recomputed aggregate metrics",
        "public_aggregate_reseal_attack_rejected", tests)

    row_attack = copy.deepcopy(receipt)
    first_config = sorted(row_attack["literal_scored_rows_by_config"])[0]
    attacked_row = row_attack["literal_scored_rows_by_config"][first_config][0]
    attacked_row["decoded_source_count"] += 1
    unsigned_row = dict(attacked_row)
    unsigned_row.pop("row_receipt_sha256")
    attacked_row["row_receipt_sha256"] = binder.sha256(
        binder.canonical_json(unsigned_row))
    row_attack["literal_scored_rows_root_sha256"] = binder.sha256(
        binder.canonical_json(row_attack["literal_scored_rows_by_config"]))
    row_attack = reseal_receipt(binder, row_attack)
    raises(lambda: binder.authorize_test(
        v1, core, panel, row_attack, expected_panel_sha256=panel_sha,
        expected_receipt_sha256=row_attack["receipt_sha256"]),
        "source size", "row_count_reseal_cannot_override_packet_header", tests)

    partition_attack = copy.deepcopy(receipt)
    attacked_row = partition_attack["literal_scored_rows_by_config"][first_config][0]
    attacked_row["partition"] = ("validation" if attacked_row["partition"] == "train"
                                  else "train")
    unsigned_row = dict(attacked_row)
    unsigned_row.pop("row_receipt_sha256")
    attacked_row["row_receipt_sha256"] = binder.sha256(
        binder.canonical_json(unsigned_row))
    partition_attack["literal_scored_rows_root_sha256"] = binder.sha256(
        binder.canonical_json(partition_attack["literal_scored_rows_by_config"]))
    partition_attack = reseal_receipt(binder, partition_attack)
    raises(lambda: binder.authorize_test(
        v1, core, panel, partition_attack, expected_panel_sha256=panel_sha,
        expected_receipt_sha256=partition_attack["receipt_sha256"]),
        "partition binding", "row_partition_reseal_rejected", tests)

    complete_reseal = copy.deepcopy(selection_rows)
    altered = complete_reseal[selected.config_id][0]
    altered_base = dict(altered)
    altered_base.pop("row_receipt_sha256")
    altered_base["raw_sse_f64_hex"] = (0.9).hex()
    complete_reseal[selected.config_id][0] = binder.seal_scored_row(altered_base)
    independently_changed = binder.make_selection_receipt(
        v1, core, panel, expected_panel_sha256=panel_sha,
        rows_by_config=complete_reseal,
        packet_receipts_by_sha256=packet_receipts)
    raises(lambda: binder.authorize_test(
        v1, core, panel, independently_changed,
        expected_panel_sha256=panel_sha,
        expected_receipt_sha256=receipt["receipt_sha256"]),
        "external pin", "complete_public_reseal_rejected_by_external_pin", tests)

    test_injection = copy.deepcopy(selection_rows)
    test_row = next(row for row in panel["rows"] if row["partition"] == "test")
    template = dict(test_injection[selected.config_id][0])
    template.pop("row_receipt_sha256")
    template.update({
        "layer": test_row["layer"], "slot": test_row["slot"],
        "role": test_row["role"], "partition": "test",
        "component_ordinal": test_row["component_ordinal"],
        "panel_source_sha256": test_row["source_sha256"],
    })
    test_injection[selected.config_id].append(binder.seal_scored_row(template))
    raises(lambda: binder.make_selection_receipt(
        v1, core, panel, expected_panel_sha256=panel_sha,
        rows_by_config=test_injection,
        packet_receipts_by_sha256=packet_receipts),
        "selection domain", "test_row_forbidden_during_selection", tests)

    context = binder.make_launch_context(
        panel_sha256=panel_sha,
        selection_receipt_sha256=receipt["receipt_sha256"],
        config_id=selected.config_id, layer=target["layer"], slot=target["slot"],
        rows=rows, cols=cols)
    check(context["launch_nonce"] == binder.sha256(
        binder.canonical_json({key: value for key, value in context.items()
                               if key != "launch_nonce"})),
        "launch_context_bound_to_panel_selection_config_and_shape", tests)

    class NameOnlyBackend:
        __name__ = "cupy"

    raises(lambda: binder.collect_cupy_launch_receipt(
        NameOnlyBackend(), launch_context=context),
        "actual CuPy module object", "name_only_fake_cupy_rejected", tests)
    fake_module = types.ModuleType("cupy")
    raises(lambda: binder.collect_cupy_launch_receipt(
        fake_module, launch_context=context),
        "CuPy", "module_shell_fake_cupy_rejected", tests)
    tampered_context = dict(context)
    tampered_context["slot"] = "other-slot"
    raises(lambda: binder.collect_cupy_launch_receipt(
        fake_module, launch_context=tampered_context),
        "launch context seal", "launch_context_tamper_rejected", tests)

    result = {
        "schema": "logic-q-v2-bound-adapter-hostile-tests-v1",
        "status": "PASS_SOURCE_ONLY_HOSTILE_TESTS",
        "tests": tests, "test_count": len(tests),
        "v1_source_root_sha256": binder.V1_SOURCE_ROOT_SHA256,
        "v0_source_root_sha256": binder.V0_SOURCE_ROOT_SHA256,
        "panel_sha256": panel_sha,
        "selection_receipt_sha256": receipt["receipt_sha256"],
        "packet_sha256": geometry["expert_packet_sha256"],
        "model_or_qwen_payload_accessed": False,
        "cupy_imported_or_initialized": False,
        "network_accessed": False,
        "strata_six_pass_semantics_tested": False,
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
