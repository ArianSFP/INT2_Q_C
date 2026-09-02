#!/usr/bin/env python3
"""Verify the frozen source closure and the split memory/compute verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "epsilon-tcq-polar-cow-memory-v2-source-manifest"
STATUS = "FROZEN_SOURCE_ONLY_GO_MEMORY_CAPACITY_HOLD_COMPUTE_AND_PAYLOAD"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def verify(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    manifest_raw = (root / "SOURCE_MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS:
        raise RuntimeError("manifest schema/status")
    observed = []
    for row in manifest["members"]:
        if set(row) != {"name", "bytes", "sha256"}:
            raise RuntimeError("manifest member fields")
        raw = (root / row["name"]).read_bytes()
        if len(raw) != row["bytes"] or sha(raw) != row["sha256"]:
            raise RuntimeError(f"manifest member mismatch: {row['name']}")
        observed.append({"name": row["name"], "bytes": len(raw), "sha256": sha(raw)})
    names = [row["name"] for row in observed]
    if names != sorted(names, key=lambda value: value.encode("utf-8")):
        raise RuntimeError("manifest ordering")
    if sha(canonical(observed)) != manifest["source_root_sha256"]:
        raise RuntimeError("source root")
    if {path.name for path in root.iterdir()} != set(names) | {"SOURCE_MANIFEST.json"}:
        raise RuntimeError("undeclared package member")

    compact = (root / "compact_sc.py").read_text(encoding="utf-8")
    memory = (root / "memory_plan.py").read_text(encoding="utf-8")
    cupy = (root / "cupy_state_smoke.py").read_text(encoding="utf-8")
    runner = (root / "run_gate.py").read_text(encoding="utf-8")
    lock = json.loads((root / "design_lock.json").read_text(encoding="utf-8"))
    if "lr_flat = np.ones(n - 1" not in compact or "mu_flat = np.zeros(n - 1" not in compact:
        raise RuntimeError("ragged state absent")
    if "maximum_events = LEVELS * block_values" not in memory:
        raise RuntimeError("six-level ancestry absent")
    if "PASS_GO_MEMORY_CAPACITY_ONLY_HOLD_COMPUTE_AND_DEVICE_COW" not in cupy:
        raise RuntimeError("CuPy smoke overclaims")
    if "HOLD_COMPUTE_AND_DEVICE_COW_IMPLEMENTATION" not in runner:
        raise RuntimeError("compute hold absent")
    if "--payload" in runner or "--qwen" in runner:
        raise RuntimeError("payload option forbidden")
    if lock["verdicts"]["memory"] != "GO_MEMORY_CAPACITY":
        raise RuntimeError("memory verdict")
    if not str(lock["verdicts"]["compute"]).startswith("HOLD_COMPUTE"):
        raise RuntimeError("compute verdict")
    return {
        "schema": "epsilon-tcq-polar-cow-memory-v2-source-verification",
        "status": "PASS",
        "manifest_sha256": sha(manifest_raw),
        "source_root_sha256": manifest["source_root_sha256"],
        "members": len(observed),
        "memory_verdict": lock["verdicts"]["memory"],
        "compute_verdict": lock["verdicts"]["compute"],
        "qwen_payload_accessed": False,
        "current_codec_payload_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    print(json.dumps(verify(Path(parser.parse_args().package)), sort_keys=True,
                     separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
