#!/usr/bin/env python3
"""Independent fail-closed validator for the sealed v2 panel proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any


EXPECTED_FILES = {
    "selection.proposal.lock.json": "528250d8c6bac52dfdf64958d7f4929a115ff68d907a47880cab85d532aade14",
    "route_table.proposal.bin": "94feb3564fe0c3eddfc745703f1f6001b5ae316e7146209e6b45323cdf81697c",
    "route_table.proposal.audit.json": "a95b17ff26027b6a76ad42c04b2b1e655fb80307d168a795f7c7e6c5305de22c",
    "unopened_snapshot.audit.json": "0d0a7de5ecca5f6ca914841dcbad028275c2c36ca31dffcf8cd37c0fc975ebe3",
    "v1_failure_independent_audit.json": "5ebe6fd5efbc10162a49a84083e99ae0123daa1680fd54b47a12d79f99369ea3",
}
EXPECTED_PAIRS = [(5, 18), (12, 7), (18, 20), (28, 83), (36, 76), (45, 41)]
EXPECTED_SELECTION_INTERNAL = "cd8cb70ca7509d2ddd4899df8a7047b7b8f47d381b637e2eb497db9ecd4eb9f8"
ROUTE_RECORD = struct.Struct(">HHBBH")
ROLE_ENUM = {"gate": 0, "up": 1, "down": 2}
AXIS_ENUM = {"row": 0, "column": 1}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def verify_seal(value: dict[str, Any]) -> str:
    clean = dict(value)
    declared = clean.pop("lock_sha256", None)
    actual = hashlib.sha256(canonical_bytes(clean)).hexdigest()
    if actual != declared:
        raise AssertionError(f"internal seal mismatch: {actual} != {declared}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    proposal = args.proposal_dir.resolve(strict=True)

    actual_files = {name: sha256_file(proposal / name) for name in EXPECTED_FILES}
    if actual_files != EXPECTED_FILES:
        raise AssertionError(f"proposal artifact hash drift: {actual_files}")
    selection = load_object(proposal / "selection.proposal.lock.json")
    route_audit = load_object(proposal / "route_table.proposal.audit.json")
    unopened = load_object(proposal / "unopened_snapshot.audit.json")
    v1 = load_object(proposal / "v1_failure_independent_audit.json")
    if verify_seal(selection) != EXPECTED_SELECTION_INTERNAL:
        raise AssertionError("selection internal hash drift")
    verify_seal(route_audit)
    verify_seal(unopened)
    verify_seal(v1)

    if selection.get("status") != "sealed_metadata_only_proposal_payload_unopened_not_codec_frozen":
        raise AssertionError("selection is not the unopened proposal")
    if selection["proposal_semantics"] != {
        "authorizes_encode": False,
        "authorizes_payload_access": False,
        "is_codec_freeze": False,
        "purpose": "precommit a rigorously disjoint second panel while codec development remains separate",
    }:
        raise AssertionError("proposal semantics drift")
    if selection["payload_interlock"]["authorized"]:
        raise AssertionError("payload access is unexpectedly authorized")
    pairs = [
        (int(row["selected_layer"]), int(row["selected_expert"]))
        for row in selection["selected_pairs"]
    ]
    if pairs != EXPECTED_PAIRS:
        raise AssertionError(f"selected pairs drift: {pairs}")
    excluded_layers = set(selection["contamination_ledger"]["excluded_layer_indices"])
    excluded_experts = set(selection["contamination_ledger"]["excluded_expert_indices"])
    if any(layer in excluded_layers or expert in excluded_experts for layer, expert in pairs):
        raise AssertionError("selected panel is not coordinate-disjoint")

    matrices = selection["matrices"]
    if len(matrices) != 18 or sum(int(row["block_count"]) for row in matrices) != 108:
        raise AssertionError("panel cardinality drift")
    if any(row["source_bf16_sha256"] is not None for row in matrices):
        raise AssertionError("selection proposal unexpectedly contains source hashes")
    if any(block["source_bf16_sha256"] is not None for row in matrices for block in row["blocks"]):
        raise AssertionError("selection proposal unexpectedly contains block hashes")

    derived = bytearray()
    for ordinal, row in enumerate(matrices):
        if int(row["matrix_ordinal"]) != ordinal:
            raise AssertionError("matrix ordinal drift")
        role = str(row["role"])
        axis = "column" if role == "down" else "row"
        groups = int(row["shape"][1] if axis == "column" else row["shape"][0])
        derived.extend(
            ROUTE_RECORD.pack(
                int(row["layer"]), int(row["expert"]), ROLE_ENUM[role], AXIS_ENUM[axis], groups
            )
        )
    route = (proposal / "route_table.proposal.bin").read_bytes()
    if route != bytes(derived) or len(route) != 144:
        raise AssertionError("literal route does not derive from the selected matrices")
    if hashlib.sha256(route).hexdigest() != route_audit["route_table_sha256"]:
        raise AssertionError("route audit hash mismatch")

    if v1["primary_claim"]["claim_passed"] or v1["distortion"]["mse_gate_passed"]:
        raise AssertionError("v1 failure was rewritten as a pass")
    if not v1["physical_rate"]["integer_gate_passed"]:
        raise AssertionError("v1 failure mechanism drift")
    if not unopened.get("passed") or unopened.get("selector_tensor_payload_bytes_read") != 0:
        raise AssertionError("unopened snapshot is invalid")

    forbidden = [
        proposal / "codec_freeze.lock.json",
        proposal / "unblinded",
        proposal / "materialize_full_tensors.py",
    ]
    if any(path.exists() for path in forbidden):
        raise AssertionError(f"premature v2 execution artifact exists: {forbidden}")
    candidate_tensors = {str(row["tensor"]) for row in matrices}
    candidate_payload_paths = []
    full_shards = []
    for directory, _, filenames in os.walk(workspace):
        for filename in filenames:
            path = Path(directory) / filename
            if filename.startswith("model-") and filename.endswith("-of-00016.safetensors"):
                full_shards.append(str(path))
            if any(tensor in filename for tensor in candidate_tensors) and (
                filename.endswith(".bf16") or filename.endswith(".bf16.bin") or filename.endswith(".safetensors")
            ):
                candidate_payload_paths.append(str(path))
    if candidate_payload_paths or full_shards:
        raise AssertionError(
            f"candidate payload/full-shard evidence appeared: candidates={candidate_payload_paths}, shards={full_shards}"
        )

    print(
        json.dumps(
            {
                "passed": True,
                "selection_file_sha256": actual_files["selection.proposal.lock.json"],
                "selection_internal_sha256": selection["lock_sha256"],
                "route_table_sha256": actual_files["route_table.proposal.bin"],
                "pairs": [list(pair) for pair in pairs],
                "payload_bytes_read": 0,
                "codec_freeze_present": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
