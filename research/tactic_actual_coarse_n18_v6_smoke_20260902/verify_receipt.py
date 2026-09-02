#!/usr/bin/env python3
"""Independent source-only validator for the frozen v6 CuPy smoke receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


EXPECTED_MANIFEST = "31662539a4c55926f47b378d15a0d8e23c90aa0903328c44be2e237eca48b15d"
EXPECTED_ROOT = "161ab23169af3427648ec1bbcb9402568a0fb8aefc4a794daf3ebd1c56cc83f2"
EXPECTED_PREDECESSOR = "645310404673e944c0f61e08747b4d7d50e6681cd450eb829acd8614c41f4322"
EXPECTED_RUNTIME = "de1464d23de161d90f0784183743252631385ad69ba2620697dea7df763c3490"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-file-sha256", required=True)
    args = parser.parse_args()

    manifest_bytes = (args.package / "SOURCE_MANIFEST.json").read_bytes()
    if sha256(manifest_bytes) != EXPECTED_MANIFEST:
        raise ValueError("source manifest hash")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("source_root_sha256") != EXPECTED_ROOT:
        raise ValueError("source root")
    rows = manifest.get("members")
    if not isinstance(rows, list) or len(rows) != 13:
        raise ValueError("source rows")
    member_hashes: dict[str, str] = {}
    for row in rows:
        name = row["name"]
        payload = (args.package / name).read_bytes()
        if len(payload) != row["bytes"] or sha256(payload) != row["sha256"]:
            raise ValueError(f"source member {name}")
        member_hashes[name] = row["sha256"]

    contract_path = args.package / "smoke_contract.py"
    spec = importlib.util.spec_from_file_location("tactic_v6_frozen_smoke_contract", contract_path)
    if spec is None or spec.loader is None:
        raise ValueError("smoke contract import spec")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)

    receipt_bytes = args.receipt.read_bytes()
    expected_receipt_file_sha256 = args.expected_receipt_file_sha256.lower()
    if sha256(receipt_bytes) != expected_receipt_file_sha256:
        raise ValueError("receipt file hash")
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    validated = contract.validate_smoke_receipt(
        receipt,
        source_manifest_sha256=EXPECTED_MANIFEST,
        source_root_sha256=EXPECTED_ROOT,
        predecessor_lock_sha256=EXPECTED_PREDECESSOR,
        runtime_lock_sha256=EXPECTED_RUNTIME,
        source_member_hashes=member_hashes,
    )
    output = {
        "schema": "tactic-actual-coarse-n18-v6-independent-smoke-audit-v1",
        "status": "PASS_SOURCE_FREE_CUPY_SMOKE_RECEIPT",
        "receipt_file_bytes": len(receipt_bytes),
        "receipt_file_sha256": expected_receipt_file_sha256,
        **validated,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
