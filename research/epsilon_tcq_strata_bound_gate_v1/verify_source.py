#!/usr/bin/env python3
"""Verify exact v1 source closure and non-negotiable static boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "epsilon-tcq-strata-bound-gate-v1-source-manifest"
STATUS = "FROZEN_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def verify(root):
    root = root.resolve(strict=True)
    manifest_raw = (root / "SOURCE_MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest["schema"] == SCHEMA and manifest["status"] == STATUS
    observed = []
    for row in manifest["members"]:
        assert set(row) == {"name", "bytes", "sha256"}
        raw = (root / row["name"]).read_bytes()
        assert len(raw) == row["bytes"] and sha(raw) == row["sha256"]
        observed.append({"name": row["name"], "bytes": len(raw), "sha256": sha(raw)})
    assert [row["name"] for row in observed] == sorted(
        (row["name"] for row in observed), key=lambda value: value.encode("utf-8"))
    assert sha(canonical(observed)) == manifest["source_root_sha256"]
    assert {path.name for path in root.iterdir()} == {
        row["name"] for row in observed} | {"SOURCE_MANIFEST.json"}
    adapter = (root / "strata_replay_adapter.py").read_text(encoding="utf-8")
    driver = (root / "bound_driver.py").read_text(encoding="utf-8")
    runner = (root / "run_gate.py").read_text(encoding="utf-8")
    independent = (root / "independent_decoder.py").read_text(encoding="utf-8")
    oracle = (root / "polar_list_oracle.py").read_text(encoding="utf-8")
    assert "AUDITOR_SHA256 = \"85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e\"" in adapter
    assert "coordinate_local_arithmetic_events = False" in adapter
    assert "HOLD_COORDINATE_LOCAL_EPSILON_INVALID_FOR_LEVEL_MAJOR_POLAR_SC" in adapter
    assert "row bytes exactly equal literal byte ledger" in driver
    assert "control receipt external pin" in driver
    assert "outer fold fit/selection closure" in driver
    assert "canonical independent byte reencode" in independent
    assert "HOLD_PRODUCTION_POLAR_LIST_SCALABILITY" in oracle
    assert "--payload" not in runner and "--qwen" not in runner
    return {
        "schema": "epsilon-tcq-strata-bound-v1-source-verification",
        "status": "PASS", "manifest_sha256": sha(manifest_raw),
        "source_root_sha256": manifest["source_root_sha256"],
        "members": len(observed), "qwen_payload_accessed": False,
        "current_codec_payload_accessed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    print(json.dumps(verify(Path(parser.parse_args().package)), sort_keys=True,
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
