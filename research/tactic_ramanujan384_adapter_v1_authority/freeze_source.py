#!/usr/bin/env python3
"""Mechanically freeze the source-only package with the shared root algorithm."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_manifest_module():
    path = ROOT / "manifest.py"
    spec = importlib.util.spec_from_file_location("tactic_ramanujan384_authority_freeze_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    algorithm = load_manifest_module()
    rows = algorithm.collect(ROOT)
    payload = {
        "schema": "tactic-ramanujan384-authority-source-manifest-v1",
        "status": "FROZEN_SOURCE_ONLY_NO_QWEN_OR_COARSE_PAYLOAD_AUTHORITY",
        "source_root_algorithm": "sha256(canonical_json(sorted_member_rows)); sorted object keys",
        "source_root_sha256": algorithm.source_root(rows),
        "members": rows,
        "dependency_pins": json.loads((ROOT / "dependency_lock.json").read_text(encoding="utf-8")),
        "access_attestation": {
            "qwen_payload_accessed": False,
            "coarse_model_payload_accessed": False,
            "matched_model_control_payload_accessed": False,
            "network_accessed": False,
        },
        "execution_attestation": {
            "source_only_tests": "NOT_EXECUTED_LOCAL_PYTHON_ABSENT_AND_RUNPOD_ENDPOINT_REFUSED",
            "source_free_cupy_smoke": "NOT_EXECUTED_RUNPOD_ENDPOINT_REFUSED",
            "payload_execution_authorized": False,
        },
        "claim_boundary": "source authority repair only; no payload result or performance claim",
    }
    (ROOT / "SOURCE_MANIFEST.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii", newline="\n",
    )
    print(json.dumps({
        "source_root_sha256": payload["source_root_sha256"],
        "members": len(rows),
        "qwen_payload_accessed": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
