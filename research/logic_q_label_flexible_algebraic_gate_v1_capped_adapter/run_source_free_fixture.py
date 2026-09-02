#!/usr/bin/env python3
"""Deterministic source-free capped-adapter mechanism fixture."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parent
PARENT = PACKAGE.parent / "logic_q_label_flexible_algebraic_gate_v0"


def load_adapter():
    spec = importlib.util.spec_from_file_location("logicq_v1_fixture_adapter",
                                                  PACKAGE / "capped_adapter.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("adapter import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def role_source(rows: int, cols: int, phase: float):
    index = np.arange(rows * cols, dtype=np.float64)
    values = (0.84 * np.sin(index * 0.061 + phase) +
              0.09 * np.cos(index * 0.127 - phase))
    weights = 0.9 + (index % 7) * 0.025
    return values, weights


def panel_rows(adapter, rows: int, cols: int):
    result = []
    for layer_index in range(10):
        for slot_index in range(4):
            layer = f"fixture-layer-{layer_index:02d}"
            slot = f"fixture-expert-{slot_index:02d}"
            for role in adapter.ROLE_ORDER:
                source_hash = hashlib.sha256(
                    f"{layer}:{slot}:{role}:{rows}:{cols}".encode("ascii")).hexdigest()
                result.append(adapter.PanelRow(layer, slot, role, rows, cols,
                                               source_hash))
    return result


def selection_metrics(adapter):
    result = {"train": {}, "validation": {}}
    for index, config in enumerate(adapter.FROZEN_CONFIGS):
        for partition in result:
            result[partition][config.config_id] = {
                "physical_bits": float(2_280_000 + 20_000 * index),
                "weights": 1_000_000.0,
                "weighted_sse": float(31_000 + 150 * index),
                "source_energy": 1_000_000.0,
                "expert_count": 8.0,
            }
    return result


def main() -> None:
    adapter = load_adapter()
    parent_receipt = adapter.verify_parent_package(PARENT)
    core = adapter.load_parent_core(PARENT)
    rows, cols = 4, 256
    roles = {role: role_source(rows, cols, 0.4 + ordinal)
             for ordinal, role in enumerate(adapter.ROLE_ORDER)}
    config = adapter.FROZEN_CONFIGS[0]
    encoded = adapter.encode_expert(np, core, roles, rows=rows, cols=cols,
                                    config=config, live=False)
    replay = adapter.unpack_canonical_expert(np, core, encoded["packet"])
    panel = adapter.panel_record(panel_rows(adapter, rows, cols))
    selection = adapter.selection_receipt(panel, selection_metrics(adapter))
    selected = adapter.authorize_test(panel, selection)
    result = {
        "schema": "logic-q-v1-capped-adapter-source-free-fixture",
        "status": "PASS_SOURCE_FREE_MECHANISM_FIXTURE",
        "parent": parent_receipt,
        "config_id": config.config_id,
        "selected_fixture_config_id": selected.config_id,
        "score": encoded["score"],
        "packet_sha256": adapter.sha256(encoded["packet"]),
        "packet_bytes": len(encoded["packet"]),
        "decoded_roles": sorted(replay),
        "panel_sha256": panel["panel_sha256"],
        "selection_receipt_sha256": selection["receipt_sha256"],
        "mechanism_only": True,
        "model_or_qwen_payload_accessed": False,
        "matched_control_artifact_accessed": False,
        "network_accessed": False,
        "cupy_initialized": False,
        "claim_boundary": "Synthetic finite mechanism; not a model result.",
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
