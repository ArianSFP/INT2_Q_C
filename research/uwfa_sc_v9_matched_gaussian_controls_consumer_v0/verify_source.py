#!/usr/bin/env python3
"""Standard-library verifier for the frozen source-only package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REQUIRED = {
    "API_AUDIT.md",
    "BLOCK.json",
    "README.md",
    "SOURCE_MANIFEST.json",
    "consumer_contract.py",
    "design_lock.json",
    "matched_controls_consumer.py",
    "test_source_only.py",
    "verify_source.py",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    observed = {path.name for path in package.iterdir() if path.is_file()}
    require(observed == REQUIRED, f"member set {sorted(observed)}")
    manifest = json.loads(
        (package / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    require(
        manifest.get("schema")
        == "uwfa-sc-v9-matched-controls-consumer-source-manifest-v0",
        "manifest schema",
    )
    require(manifest.get("status") == "SOURCE_ONLY_NONPROMOTING", "manifest status")
    rows = manifest.get("members")
    require(isinstance(rows, list) and len(rows) == len(REQUIRED) - 1, "manifest rows")
    require(
        [row["name"] for row in rows]
        == sorted(REQUIRED - {"SOURCE_MANIFEST.json"}),
        "manifest order",
    )
    for row in rows:
        path = package / row["name"]
        require(path.stat().st_size == row["bytes"], f"bytes {row['name']}")
        require(sha(path) == row["sha256"], f"hash {row['name']}")
    lock = json.loads((package / "design_lock.json").read_text(encoding="utf-8"))
    require(
        lock.get("status")
        == "SOURCE_ONLY_CONSUMER_AND_INPUT_AUDITOR_FROZEN_EXTERNAL_PINS_BLOCKED",
        "design status",
    )
    require(
        lock["scientific_contract"]["canonical_candidates_per_fold"] == 150,
        "all150 lock",
    )
    runtime = (package / "matched_controls_consumer.py").read_text(encoding="utf-8")
    consume = runtime[runtime.index("def consume_controls(") :]
    require(
        consume.index("authenticate_eight_control_root")
        < consume.index("backend_factory()"),
        "authentication before backend",
    )
    require(
        consume.index("_prepare_all_controls") < consume.index("backend_factory()"),
        "all preparation before backend",
    )
    require("all_150_inner_validation_cells" in runtime, "150-cell evidence")
    require("model.layers." not in runtime, "no model identity")
    block = json.loads((package / "BLOCK.json").read_text(encoding="utf-8"))
    require(block.get("payload_access_authority") is False, "no payload authority")
    print(
        json.dumps(
            {
                "status": "PASS_SOURCE_ONLY_MATCHED_CONTROLS_CONSUMER",
                "members": len(REQUIRED),
                "payload_access_authority": False,
                "cuda_imported": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCK_SOURCE_VERIFICATION: {exc}", file=sys.stderr)
        raise
