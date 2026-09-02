#!/usr/bin/env python3
"""Standard-library source verifier; never imports runtime numeric modules."""

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
    "design_lock.json",
    "full_ptq_producer.py",
    "producer_contract.py",
    "test_source_only.py",
    "verify_source.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    observed = {path.name for path in package.iterdir() if path.is_file()}
    require(observed == REQUIRED, f"package member set: {sorted(observed)}")
    manifest = json.loads((package / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == "uwfa-sc-v9-matched-gaussian-ptq-source-manifest-v0", "manifest schema")
    require(manifest.get("status") == "SOURCE_ONLY_NONPROMOTING", "manifest status")
    rows = manifest.get("members")
    require(isinstance(rows, list) and len(rows) == len(REQUIRED) - 1, "manifest row count")
    require([row["name"] for row in rows] == sorted(REQUIRED - {"SOURCE_MANIFEST.json"}), "manifest order")
    for row in rows:
        path = package / row["name"]
        require(path.stat().st_size == row["bytes"], f"member bytes {row['name']}")
        require(sha(path) == row["sha256"], f"member digest {row['name']}")
    lock = json.loads((package / "design_lock.json").read_text(encoding="utf-8"))
    require(lock.get("schema") == "uwfa-sc-v9-matched-gaussian-ptq-design-v0", "design schema")
    require(lock.get("status") == "SOURCE_ONLY_PRODUCER_BUILT_V9_CONSUMER_BLOCKED", "design status")
    require(lock["frozen_controls"]["seeds"] == [10619863,10619881,10619909,10619927,10619953,10619971,10619999,10620017], "control seeds")
    require(lock["scientific_contract"]["all_150_independently_selected_per_control"] is True, "all150 invariant")
    block = json.loads((package / "BLOCK.json").read_text(encoding="utf-8"))
    require(block.get("positive_claim_authority") is False, "no positive authority")
    runtime = (package / "full_ptq_producer.py").read_text(encoding="utf-8")
    for fragment in (
        "v2_emitter.build_staging",
        "run_and_pack.run_block",
        "adapter.extract_from_current",
        '"all_150_wfa_search_run": False',
    ):
        require(fragment in runtime, f"runtime exact boundary {fragment}")
    require("model.layers." not in runtime, "runtime must not bind Qwen tensor identity")
    print(json.dumps({
        "status": "PASS_SOURCE_ONLY_PRODUCER_PACKAGE",
        "package": str(package),
        "members": len(REQUIRED),
        "payload_authority": False,
        "cuda_imported": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCK_SOURCE_VERIFICATION: {exc}", file=sys.stderr)
        raise
