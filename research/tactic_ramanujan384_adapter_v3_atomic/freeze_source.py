#!/usr/bin/env python3
"""Freeze exact v3 source closure; no runtime or payload access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MEMBERS = (
    "EXECUTION_STATUS.json", "README.md", "STATIC_SOURCE_RECEIPT.json",
    "adapter_atomic.py", "coarse_byte_worker.py", "dependency_lock.json",
    "design_lock.json", "freeze_source.py", "snapshot_runner.py",
    "source_free_fixture_atomic.py", "test_source_only.py",
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def main() -> int:
    expected = sorted(MEMBERS + (("SOURCE_MANIFEST.json",)
                                if (ROOT / "SOURCE_MANIFEST.json").exists() else ()))
    actual = sorted(path.name for path in ROOT.iterdir())
    if actual != expected or any(not (ROOT / name).is_file() for name in actual):
        raise SystemExit("exact flat v3 source closure")
    rows = []
    for name in sorted(MEMBERS):
        payload = (ROOT / name).read_bytes()
        rows.append({"name": name, "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()})
    dependencies = json.loads((ROOT / "dependency_lock.json").read_text(encoding="ascii"))
    document = {
        "schema": "tactic-ramanujan384-atomic-source-manifest-v3",
        "status": "FROZEN_SOURCE_ONLY_RUNTIME_AND_PAYLOAD_HELD",
        "source_root_sha256": hashlib.sha256(canonical_json(rows)).hexdigest(),
        "members": rows, "dependency_pins": dependencies,
        "execution": {
            "powershell_static_source_checks": "PASS",
            "python_source_tests": "NOT_EXECUTED_NO_LOCAL_PYTHON",
            "atomic_bootstrap_verify_only": "NOT_EXECUTED_NO_LOCAL_PYTHON",
            "source_free_cpu": "HELD", "source_free_cupy": "HELD",
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
