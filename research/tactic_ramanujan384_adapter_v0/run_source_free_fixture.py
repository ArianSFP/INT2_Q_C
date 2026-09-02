#!/usr/bin/env python3
"""Small CPU fixture for the complete literal packet and control path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load(name: str, filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run() -> dict[str, object]:
    codec = load("tactic_ramanujan384_fixture_codec", "ramanujan_codec.py")
    coordinate = np.arange(codec.BLOCK_VALUES, dtype=np.int64)
    period7 = np.asarray([codec.ramanujan_sum(7, index) for index in range(7)], dtype=np.float64)
    period11 = np.asarray([codec.ramanujan_sum(11, index) for index in range(11)], dtype=np.float64)
    residual = np.stack([
        0.013 * period7[(coordinate - block) % 7]
        + 0.004 * period11[(coordinate - 2 * block) % 11]
        for block in range(2)
    ])
    coarse = np.zeros_like(residual)
    source_energy = float(np.sum(residual * residual, dtype=np.float64)) / codec.COARSE_RELATIVE_MSE
    result = codec.run_finite_panel(np, residual, coarse, role="gate", source_energy=source_energy)
    if not result["controls_rerun"] or len(result.get("gaussian_controls", [])) != 8:
        raise RuntimeError("source-free fixture did not execute the complete control contract")
    return {
        "schema": "tactic-ramanujan384-source-free-fixture-v0",
        "status": "PASS_SOURCE_FREE_FINITE_PACKET_AND_ALL_CONTROLS",
        "finite_status": result["status"],
        "relative_mse": result["relative_mse"],
        "controls_rerun": result["controls_rerun"],
        "gaussian_controls": len(result.get("gaussian_controls", [])),
        "fine_stream_bytes": result["fine_stream_bytes"],
        "fine_stream_sha256": result["fine_stream_sha256"],
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "cuda_initialized": False,
        "network_accessed": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, separators=(",", ":")))
