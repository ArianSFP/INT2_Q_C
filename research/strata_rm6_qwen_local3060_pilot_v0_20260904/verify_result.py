#!/usr/bin/env python3
"""Stdlib verification of the sealed bounded STRATA-RM6 Qwen pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_member_manifest(root: Path, manifest: dict[str, object]) -> None:
    members = manifest["members"]
    need(isinstance(members, list), "manifest members")
    expected = []
    for row in members:
        path = root / row["name"]
        raw = path.read_bytes()
        need(len(raw) == int(row["bytes"]) and sha(raw) == row["sha256"],
             f"member mismatch: {row['name']}")
        expected.append({"name": row["name"], "bytes": len(raw), "sha256": sha(raw)})
    need(sha(canonical(expected)) == manifest["source_root_sha256"], "source root")
    need({path.name for path in root.iterdir()} ==
         {row["name"] for row in members} | {"SOURCE_MANIFEST.json"},
         "package closure")


def verify(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.package).resolve(strict=True)
    manifest_raw = (root / "SOURCE_MANIFEST.json").read_bytes()
    need(sha(manifest_raw) == args.manifest_sha256, "published manifest hash")
    manifest = json.loads(manifest_raw)
    need(manifest["schema"] == "strata-rm6-qwen-local3060-pilot-v0-manifest",
         "manifest schema")
    verify_member_manifest(root, manifest)
    design_raw = (root / "DESIGN_LOCK.json").read_bytes()
    design = json.loads(design_raw)
    result_raw = (root / "RESULT.json").read_bytes()
    result = json.loads(result_raw)
    need(result["design_lock_sha256"] == sha(design_raw), "design binding")
    result_without_self = dict(result)
    declared_self = result_without_self.pop("result_sha256_excluding_self")
    need(declared_self == sha(canonical(result_without_self)), "result self hash")
    need(result["status"] ==
         "HARD_KILL_BANK0_QWEN_PHYSICAL_OVERFLOW_AT_ALL_CHECKPOINTS",
         "terminal verdict")
    source = design["immutable_source"]
    source_root = root.parents[1] / source["package"]
    source_manifest = (source_root / "SOURCE_MANIFEST.json").read_bytes()
    need(sha(source_manifest) == source["source_manifest_sha256"], "source manifest")
    source_data = json.loads(source_manifest)
    need(source_data["source_root_sha256"] == source["source_root_sha256"],
         "source root declaration")
    for row in source_data["members"]:
        raw = (source_root / row["name"]).read_bytes()
        need(len(raw) == int(row["bytes"]) and sha(raw) == row["sha256"],
             f"immutable source member: {row['name']}")
    audit_root = root.parents[1] / source["static_independent_audit_package"]
    audit_raw = (audit_root / "AUDIT_SOURCE_MANIFEST.json").read_bytes()
    need(sha(audit_raw) == source["static_independent_audit_manifest_sha256"],
         "static audit manifest")
    audit = json.loads(audit_raw)
    for row in audit["members"]:
        raw = (audit_root / row["name"]).read_bytes()
        need(len(raw) == int(row["bytes"]) and sha(raw) == row["sha256"],
             f"static audit member: {row['name']}")
    panel_path = root.parents[1] / design["qwen"]["panel_lock"]
    panel_raw = panel_path.read_bytes()
    need(sha(panel_raw) == design["qwen"]["panel_lock_sha256"], "panel lock")
    payload_path = root.parents[2] / design["qwen"]["payload_relative_path"]
    payload_raw = payload_path.read_bytes()
    need(len(payload_raw) == int(design["qwen"]["payload_bytes"]) and
         sha(payload_raw) == design["qwen"]["payload_sha256"], "Qwen payload")
    rows = result["results"]["qwen"]
    controls = result["results"]["matched_gaussian"]
    best = max(rows, key=lambda row: row["selected_mse_reduction_fraction_vs_rm_sc"])
    matched = next(row for row in controls if row["coset_mode"] == best["coset_mode"])
    metrics = result["decision_metrics"]
    need(math.isclose(metrics["qwen_mse_reduction_fraction"],
                      best["selected_mse_reduction_fraction_vs_rm_sc"],
                      rel_tol=0.0, abs_tol=1e-15), "Qwen gain")
    need(math.isclose(metrics["matched_control_mse_reduction_fraction"],
                      matched["selected_mse_reduction_fraction_vs_rm_sc"],
                      rel_tol=0.0, abs_tol=1e-15), "control gain")
    need(math.isclose(metrics["qwen_minus_control_percentage_points"],
                      metrics["qwen_mse_reduction_fraction"] -
                      metrics["matched_control_mse_reduction_fraction"],
                      rel_tol=0.0, abs_tol=1e-15), "control correction")
    all_qwen = [checkpoint for row in rows for checkpoint in row["checkpoints"]]
    need(all(not row["literal_packet_emitted"] and not row["passes_2_5_bpw"]
                 and row["physical_failure"] ==
                 "actual arithmetic packet exceeds 2.5 bpw"
             for row in all_qwen), "all Qwen checkpoint physical failures")
    need(result["runtime"]["uuid"] ==
         "GPU-458a424a-76e3-65e5-0470-803e0ed131ca" and
         result["runtime"]["nvidia_smi_name"] == "NVIDIA GeForce RTX 3060",
         "local GPU identity")
    need(result["network_accessed"] is False and result["runpod_accessed"] is False,
         "local-only declaration")
    return {"schema": "strata-rm6-qwen-local3060-pilot-v0-verification",
            "status": "PASS_AUTHENTICATED_BOUNDED_HARD_KILL",
            "manifest_sha256": sha(manifest_raw),
            "source_root_sha256": manifest["source_root_sha256"],
            "result_sha256": sha(result_raw),
            "qwen_gain": metrics["qwen_mse_reduction_fraction"],
            "control_gain": metrics["matched_control_mse_reduction_fraction"],
            "qwen_minus_control": metrics["qwen_minus_control_percentage_points"],
            "qwen_checkpoints_rejected_over_2_5": len(all_qwen),
            "whole_expert_claim": False, "target_F_claim": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    print(json.dumps(verify(parser.parse_args()), sort_keys=True,
                     separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
