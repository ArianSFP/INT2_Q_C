#!/usr/bin/env python3
"""Executable assertions for the independent source audit; no payload aperture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def load_audit():
    path = Path(__file__).resolve().parent / "audit_source.py"
    payload = path.read_bytes()
    spec = importlib.util.spec_from_file_location("mosaic_secondary_independent_audit_tests", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("audit loader")
    module = importlib.util.module_from_spec(spec)
    module.__authenticated_sha256__ = hashlib.sha256(payload).hexdigest()
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(upstream: Path) -> dict:
    audit = load_audit()
    result = audit.run(upstream)
    assert result["status"] == "MECHANISMS_VALID__HOLD_PRODUCTION_ADAPTER_SCORER_BACKEND_AND_IO_BINDING"
    assert result["recurrence"]["exhaustive_binary_sequences_n1_through_n10"] == 2046
    assert result["recurrence"]["bm_minimality_and_exact_replay"] is True
    assert result["recurrence"]["large_synthetic_expert_physical_bytes"] == 61440
    assert result["recurrence"]["large_synthetic_expert_physical_rate_bpw"] == 2.5
    assert result["gate_and_traffic"]["gate_recomputes_source_sse_from_reconstruction"] is False
    assert result["gate_and_traffic"]["cold_read_claim_is_observed_runtime_IO"] is False
    assert result["residual_oracles"]["inverse_noise_gain_trace_identity"] is True
    assert result["production_binding"]["direct_alias_is_valid"] is False
    assert result["qwen_payload_accessed"] is False
    assert result["coarse_payload_accessed"] is False
    assert result["matched_control_payload_accessed"] is False
    assert result["production_launch_authorized"] is False
    return {
        "schema": "mosaic-secondary-oracles-independent-source-audit-tests-v1",
        "status": "PASS_12_ASSERTION_GROUPS",
        "audit_source_sha256": audit.digest((Path(__file__).resolve().parent / "audit_source.py").read_bytes()),
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "matched_control_payload_accessed": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--upstream-source", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args().upstream_source), sort_keys=True, separators=(",", ":")))
