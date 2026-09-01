#!/usr/bin/env python3
"""Verify a completed source-free synthetic output and exact packet closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nonlocal_wfa import (
    SparseUnifilarWFA,
    canonical_json,
    decode_blocks,
    generate_syndrome_blocks,
)


EXPECTED_FILES = {
    "COMPLETE.json",
    "control_model.bin",
    "control_payload.bin",
    "structured_model.bin",
    "structured_payload.bin",
    "synthetic_result.json",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_seal(record: dict[str, object], field: str) -> None:
    claimed = record.get(field)
    clean = dict(record)
    clean.pop(field, None)
    if claimed != sha256(canonical_json(clean)):
        raise RuntimeError(f"{field} mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    root = args.result.resolve()
    if {path.name for path in root.iterdir() if path.is_file()} != EXPECTED_FILES:
        raise RuntimeError("result file set")
    receipt_bytes = (root / "synthetic_result.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    complete = json.loads((root / "COMPLETE.json").read_bytes())
    verify_seal(receipt, "receipt_sha256")
    verify_seal(complete, "completion_sha256")
    if receipt["status"] != "PASS_SOURCE_FREE_SYNTHETIC_ONLY":
        raise RuntimeError("result status")
    if complete["status"] != receipt["status"]:
        raise RuntimeError("completion status")
    if complete["synthetic_result_sha256"] != sha256(receipt_bytes):
        raise RuntimeError("result hash")
    for prefix in ("structured", "control"):
        model_bytes = (root / f"{prefix}_model.bin").read_bytes()
        payload_bytes = (root / f"{prefix}_payload.bin").read_bytes()
        if complete[f"{prefix}_model_sha256"] != sha256(model_bytes):
            raise RuntimeError(f"{prefix} model hash")
        if complete[f"{prefix}_payload_sha256"] != sha256(payload_bytes):
            raise RuntimeError(f"{prefix} payload hash")
        model = SparseUnifilarWFA.deserialize(model_bytes)
        if model.serialize() != model_bytes:
            raise RuntimeError(f"{prefix} model byte replay")
        row = receipt["structured" if prefix == "structured" else "matched_control"]
        if row["packet_sha256"] != sha256(model_bytes):
            raise RuntimeError(f"{prefix} receipt model hash")
        if row["payload_sha256"] != sha256(payload_bytes):
            raise RuntimeError(f"{prefix} receipt payload hash")
        if not row["roundtrip_exact"]:
            raise RuntimeError(f"{prefix} roundtrip flag")
        constrained = prefix == "structured"
        seed_base = 7001 if constrained else 17001
        expected = generate_syndrome_blocks(
            int(row["validation_blocks"]), seed_base + 1, constrained
        )
        decoded = decode_blocks(model, payload_bytes, int(row["validation_blocks"]))
        if not np.array_equal(decoded, expected):
            raise RuntimeError(f"{prefix} independent arithmetic decode")
    if receipt["structured"]["selected_chi"] != 64:
        raise RuntimeError("structured chi")
    if receipt["detected_control_minus_structured_logical_bps"] <= 0.15:
        raise RuntimeError("nonlocal detection margin")
    if receipt["structured"]["best_suffix_logical_bps"] <= 0.99:
        raise RuntimeError("suffix control")
    if receipt["matched_control"]["logical_bps"] <= 0.999:
        raise RuntimeError("iid control")
    if not receipt["structured"]["stream_ledger"]["cold_below_two"]:
        raise RuntimeError("synthetic cold ledger")
    access = receipt["access_attestation"]
    if any(bool(value) for value in access.values()):
        raise RuntimeError("payload access attestation")
    print(json.dumps({
        "status": "PASS_VERIFIED_SOURCE_FREE_SYNTHETIC_RESULT",
        "selected_chi": receipt["structured"]["selected_chi"],
        "detected_bps": receipt["detected_control_minus_structured_logical_bps"],
        "model_charged_saving_bps": receipt["structured_aggregate_model_charged_saving_bps"],
        "cold_amplification": receipt["structured"]["stream_ledger"]["synthetic_cold_read_amplification_vs_raw_one_bit_frame"],
        "qwen_evidence": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
