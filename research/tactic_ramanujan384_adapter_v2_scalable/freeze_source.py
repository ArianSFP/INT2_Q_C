#!/usr/bin/env python3
"""Freeze the exact v2 source closure; rejects undeclared package entries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_MEMBERS = (
    "AUDIT_REPAIR_MAP.json", "README.md", "adapter.py", "authenticated_io.py",
    "SOURCE_FREE_CPU_TARGET_RECEIPT.json",
    "coarse_capability.py", "dependency_lock.json", "design_lock.json",
    "fixture_coarse_decoder.py", "freeze_source.py",
    "run_source_free_cupy_target_fixture.py", "scalable_core.py",
    "source_free_fixture.py", "test_source_only.py",
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def main() -> int:
    expected_before = sorted(SOURCE_MEMBERS + (("SOURCE_MANIFEST.json",)
                                               if (ROOT / "SOURCE_MANIFEST.json").exists() else ()))
    actual = sorted(path.name for path in ROOT.iterdir())
    if actual != expected_before or any(not (ROOT / name).is_file() for name in actual):
        raise SystemExit("exact source closure: extra, missing, or nested entry")
    rows = []
    for name in sorted(SOURCE_MEMBERS):
        payload = (ROOT / name).read_bytes()
        rows.append({"name": name, "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()})
    dependency_pins = json.loads((ROOT / "dependency_lock.json").read_text(encoding="ascii"))
    document = {
        "schema": "tactic-ramanujan384-scalable-source-manifest-v2",
        "status": "FROZEN_SOURCE_ONLY_CPU_PASS_CUPY_PENDING_NO_PAYLOAD_AUTHORITY",
        "source_root_sha256": hashlib.sha256(canonical_json(rows)).hexdigest(),
        "members": rows,
        "dependency_pins": dependency_pins,
        "execution": {
            "source_tests": "PASS_9_OF_9_PYTHON_3_12_13_NUMPY_2_3_5",
            "source_free_cpu_target_rate_and_all_controls":
                "PASS_D_9_5367431640625E_7_F_3_0517578125E_5",
            "target_rate_cupy_fixture_and_controls": "PENDING_RUNPOD_ENDPOINT",
            "payload_authorized": False,
        },
        "access": {"qwen_payload": False, "coarse_model_payload": False,
                   "network": False},
    }
    (ROOT / "SOURCE_MANIFEST.json").write_text(
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii", newline="\n",
    )
    print(json.dumps({"source_root_sha256": document["source_root_sha256"],
                      "members": len(rows)}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
