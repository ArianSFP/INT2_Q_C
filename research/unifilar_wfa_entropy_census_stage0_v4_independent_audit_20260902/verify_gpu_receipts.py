#!/usr/bin/env python3
"""Independent verifier for two source-free UWFA-v4 RTX development receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def load(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    def pairs(rows):
        out = {}
        for key, value in rows:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ValueError("JSON root")
    return raw, value


def verify_identity(row: dict) -> None:
    claimed = row["identity_receipt_sha256"]
    clean = dict(row)
    clean.pop("identity_receipt_sha256")
    if sha(canonical(clean)) != claimed:
        raise ValueError("GPU identity seal")


def verify_complete(path: Path, receipt_raw: bytes) -> str:
    _raw, row = load(path)
    clean = dict(row)
    claimed = clean.pop("completion_sha256")
    if sha(canonical(clean)) != claimed:
        raise ValueError("completion seal")
    members = row["members"]
    if len(members) != 2 or members[1]["name"] != "GPU_DEV_RECEIPT.json":
        raise ValueError("completion member order")
    member = members[1]
    if member["bytes"] != len(receipt_raw) or member["sha256"] != sha(receipt_raw):
        raise ValueError("completion receipt binding")
    return claimed


def verify_one(receipt: dict, expected_inventory: list[dict]) -> None:
    if receipt["schema"] != "uwfa-sc-v4-direct-copy-development-gpu-receipt":
        raise ValueError("receipt schema")
    if receipt["status"] != "PASS_SOURCE_FREE_DEVELOPMENT_REPLAY_NO_CLAIM_AUTHORITY":
        raise ValueError("receipt status")
    if receipt["public_commit_evidence"] is not False or receipt["payload_authority_granted"] is not False:
        raise ValueError("claim boundary")
    if receipt["development_source_inventory"] != expected_inventory:
        raise ValueError("source inventory")
    root = sha(canonical(expected_inventory))
    if receipt["development_source_root_sha256"] != root:
        raise ValueError("development source root")
    identity = receipt["independent_gpu_identity"]
    verify_identity(identity)
    all150 = receipt["all150"]
    cells = all150["cells"]
    if (all150["status"] != "PASS_ALL_150_CPU_CUPY_EXACT_REPEATED"
            or all150["cell_count"] != 150 or len(cells) != 150
            or sorted(row["selector_ordinal"] for row in cells) != list(range(150))
            or not all(row["repeated_gpu_run_exact"] is True for row in cells)):
        raise ValueError("all150 coverage")
    representative = receipt["representative"]
    if representative["status"] != "PASS_REPRESENTATIVE_SOURCE_FREE_OUTER_FOLD":
        raise ValueError("representative status")
    if representative["outer_fold"]["all_150_candidates_fit_and_scored"] is not True:
        raise ValueError("representative candidate coverage")
    if representative["outer_fold"]["literal_container_parse_decode_reencode_rebuild"] is not True:
        raise ValueError("representative codec closure")
    if representative["runtime_projection"]["passes"] is not True:
        raise ValueError("runtime projection")
    for env in (all150["environment"], representative["telemetry"]):
        if (env["device_uuid"], env["pci_bus_id"], env["device_name"]) != (
                identity["device_uuid"], identity["pci_bus_id"], identity["device_name"]):
            raise ValueError("CUDA/NVML identity mismatch")
        if env["fatal_telemetry_sampling"] is not True:
            raise ValueError("fatal telemetry")
    stats = representative["measured_phase_statistics_delta"]
    for field in ("h2d_payload_bytes", "h2d_model_table_bytes", "h2d_launch_descriptor_bytes",
                  "h2d_kernel_scalar_bytes", "d2h_bytes"):
        if int(stats[field]) <= 0:
            raise ValueError(f"missing measured category: {field}")
    full_stats = representative["telemetry"]["statistics"]
    for field in ("peak_process_tree_rss_bytes", "peak_vram_incremental_bytes", "telemetry_samples"):
        if int(full_stats[field]) <= 0:
            raise ValueError(f"missing fatal telemetry category: {field}")
    if full_stats["last_pack_resource_plan"]["checked_before_blob_concatenation_or_cupy_allocation"] is not True:
        raise ValueError("dynamic memory admission")
    bound = {
        "schema": "uwfa-sc-v4-bound-source-preflight",
        "source_snapshot_root_sha256": root,
        "all150": all150,
        "representative": representative,
        "independent_gpu_identity": identity,
    }
    if sha(canonical(bound)) != receipt["bound_source_preflight_receipt_sha256"]:
        raise ValueError("bound preflight seal")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--receipt-a", required=True)
    parser.add_argument("--receipt-b", required=True)
    parser.add_argument("--complete-a", required=True)
    parser.add_argument("--complete-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inventory_raw, inventory = load(Path(args.inventory))
    expected = sorted(inventory["members"], key=lambda row: row["name"])
    raws, receipts, completions = [], [], []
    for receipt_name, complete_name in ((args.receipt_a, args.complete_a), (args.receipt_b, args.complete_b)):
        raw, receipt = load(Path(receipt_name))
        verify_one(receipt, expected)
        raws.append(raw)
        receipts.append(receipt)
        completions.append(verify_complete(Path(complete_name), raw))
    deterministic = {
        "all150_cells": receipts[0]["all150"]["cells"] == receipts[1]["all150"]["cells"],
        "representative_fixture": receipts[0]["representative"]["fixture"] == receipts[1]["representative"]["fixture"],
        "winner": receipts[0]["representative"]["outer_fold"]["winner"] == receipts[1]["representative"]["outer_fold"]["winner"],
        "container_sha256": receipts[0]["representative"]["outer_fold"]["container_sha256"] == receipts[1]["representative"]["outer_fold"]["container_sha256"],
        "full_panel_lengths": receipts[0]["representative"]["outer_fold"]["final_full_panel_logical_lengths"] == receipts[1]["representative"]["outer_fold"]["final_full_panel_logical_lengths"],
        "measured_update_count": receipts[0]["representative"]["runtime_projection"]["measured_cell_symbol_updates"] == receipts[1]["representative"]["runtime_projection"]["measured_cell_symbol_updates"],
    }
    if not all(deterministic.values()):
        raise ValueError("replay deterministic fields disagree")
    output = {
        "schema": "uwfa-v4-independent-gpu-receipt-audit-v1",
        "status": "PASS_TWO_SOURCE_FREE_RTX5090_REPLAYS_NO_CLAIM_AUTHORITY",
        "receipt_sha256": [sha(raw) for raw in raws],
        "completion_sha256": completions,
        "development_source_root_sha256": receipts[0]["development_source_root_sha256"],
        "audit_inventory_sha256": sha(inventory_raw),
        "cell_count_each": 150,
        "device_uuid": receipts[0]["independent_gpu_identity"]["device_uuid"],
        "pci_bus_id": receipts[0]["independent_gpu_identity"]["pci_bus_id"],
        "winner": receipts[0]["representative"]["outer_fold"]["winner"],
        "representative_container_sha256": receipts[0]["representative"]["outer_fold"]["container_sha256"],
        "deterministic_replay_equal": deterministic,
        "payload_authority_granted": False,
    }
    output["audit_receipt_sha256"] = sha(canonical(output))
    path = Path(args.output)
    if path.exists():
        raise RuntimeError("refusing overwrite")
    path.write_bytes(json.dumps(output, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
