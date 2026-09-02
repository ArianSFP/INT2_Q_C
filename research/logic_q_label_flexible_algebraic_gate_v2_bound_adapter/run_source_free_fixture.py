#!/usr/bin/env python3
"""Deterministic source-free replay of the v2 binding and scoring path."""

from __future__ import annotations

import importlib.util
import json
import sys
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


def main() -> None:
    binder = load_file("logicq_v2_bound_fixture", PACKAGE / "bound_adapter.py")
    scorer = load_file("logicq_v2_scorer_fixture", PACKAGE / "independent_scorer.py")
    helpers = load_file("logicq_v2_helpers_fixture", PACKAGE / "test_source_only.py")
    v1 = binder.load_v1(V1_PACKAGE)
    core = binder.load_v0(v1, V0_PACKAGE)
    rows, cols = 256, 256
    panel = binder.make_panel_record(helpers.panel_rows(binder, rows, cols))
    panel_sha = panel["panel_sha256"]
    target = next(row for row in panel["rows"]
                  if row["role"] == "gate" and row["partition"] == "train")
    packet, blobs = helpers.literal_expert(
        v1, core, rows, cols, target["layer"], target["slot"])
    score = scorer.score_expert_packet(
        binder, v1, core, packet=packet, source_blobs=blobs, panel=panel,
        expected_panel_sha256=panel_sha, layer=target["layer"],
        slot=target["slot"], config_id=v1.FROZEN_CONFIGS[0].config_id)
    geometry = score["packet_receipt"]
    selection_rows = helpers.fabricated_selection_rows(
        binder, v1, panel, geometry)
    selection = binder.make_selection_receipt(
        v1, core, panel, expected_panel_sha256=panel_sha,
        rows_by_config=selection_rows,
        packet_receipts_by_sha256={geometry["expert_packet_sha256"]: geometry})
    selected = binder.authorize_test(
        v1, core, panel, selection, expected_panel_sha256=panel_sha,
        expected_receipt_sha256=selection["receipt_sha256"])
    launch_context = binder.make_launch_context(
        panel_sha256=panel_sha,
        selection_receipt_sha256=selection["receipt_sha256"],
        config_id=selected.config_id, layer=target["layer"],
        slot=target["slot"], rows=rows, cols=cols)
    result = {
        "schema": "logic-q-v2-bound-adapter-source-free-fixture-v1",
        "status": "PASS_SOURCE_FREE_BINDING_FIXTURE",
        "v1_manifest_sha256": binder.V1_MANIFEST_SHA256,
        "v1_source_root_sha256": binder.V1_SOURCE_ROOT_SHA256,
        "v0_manifest_sha256": binder.V0_MANIFEST_SHA256,
        "v0_source_root_sha256": binder.V0_SOURCE_ROOT_SHA256,
        "panel_sha256": panel_sha,
        "selection_receipt_sha256": selection["receipt_sha256"],
        "selected_config_id": selected.config_id,
        "packet_sha256": geometry["expert_packet_sha256"],
        "packet_bytes": geometry["expert_packet_bytes"],
        "packet_weights_from_headers": geometry["expert_weights_from_headers"],
        "physical_rate_bpw": geometry["physical_rate_bpw"],
        "cold_read_amplification": geometry["cold_read_amplification"],
        "source_score_bundle_sha256": score["bundle_sha256"],
        "launch_context_nonce": launch_context["launch_nonce"],
        "selector_recomputed_from_literal_source_rows": True,
        "encoder_metrics_authoritative": False,
        "actual_cupy_launch_deferred_to_independent_gpu_audit": True,
        "model_or_qwen_payload_accessed": False,
        "network_accessed": False,
        "cupy_imported_or_initialized": False,
        "strata_six_pass_semantics_bound": False,
        "claim_boundary": (
            "Synthetic four-level v1 mechanism plus v2 orchestration bindings; "
            "not a Qwen, STRATA-RM6, F<=0.8, or universal codec result."),
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
