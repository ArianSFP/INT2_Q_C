#!/usr/bin/env python3
"""Development-only RTX/CuPy replay; never grants payload authority.

This runner hashes only the exact producer source directory, executes the two
frozen synthetic GPU gates, joins CUDA telemetry to an independently queried
``nvidia-smi`` UUID/PCI identity, validates the typed bundle, and publishes a
completion-last receipt.  A direct-copy receipt is explicitly not final claim
evidence; the external bootstrap must repeat it from the frozen public commit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).absolute().parent
EXPECTED_SOURCE_MEMBERS = {
    "README.md", "INDEPENDENT_BOOTSTRAP_ABI.md", "design_lock.json",
    "uwfa_common.py", "protocol.py", "universal_adapter.py",
    "container_codec.py", "strata_sc_adapter.py", "stage0_census.py",
    "cupy_backend.py", "dispatcher_contract.py", "result_envelope.py",
    "fixture_long_memory.py", "fixture_portability.py", "test_source_only.py",
    "verify_source.py", "run_source_free_gpu_dev.py",
}


def load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load("uwfa_v7_gpu_dev_common", "uwfa_common.py")
protocol = load("uwfa_v7_gpu_dev_protocol", "protocol.py")
semantic = load("uwfa_v7_gpu_dev_semantic", "universal_adapter.py")
codec = load("uwfa_v7_gpu_dev_codec", "container_codec.py")
stage = load("uwfa_v7_gpu_dev_stage", "stage0_census.py")
cupy_backend = load("uwfa_v7_gpu_dev_backend", "cupy_backend.py")
envelope = load("uwfa_v7_gpu_dev_envelope", "result_envelope.py")


def development_source_root() -> tuple[str, list[dict[str, Any]]]:
    actual = {entry.name for entry in os.scandir(PACKAGE)}
    if actual != EXPECTED_SOURCE_MEMBERS:
        raise RuntimeError(f"unexpected development source inventory: {sorted(actual ^ EXPECTED_SOURCE_MEMBERS)}")
    rows = []
    for name in sorted(EXPECTED_SOURCE_MEMBERS):
        with common.HeldRegularFile(PACKAGE / name) as held:
            data = held.read_all()
            held.verify_stable()
        rows.append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return hashlib.sha256(common.canonical_json(rows)).hexdigest(), rows


def independent_gpu_identity(backend: Any) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=name,uuid,pci.bus_id",
            "--format=csv,noheader,nounits", "--id=0",
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("independent GPU identity row count")
    columns = [column.strip() for column in rows[0].split(",")]
    if len(columns) != 3:
        raise RuntimeError("independent GPU identity columns")
    name, uuid, pci = columns
    record = {
        "schema": "uwfa-sc-v7-independent-gpu-identity",
        "status": "PASS_INDEPENDENT_GPU_IDENTITY",
        "device_uuid": backend._canonical_device_uuid(uuid),
        "pci_bus_id": backend._canonical_pci_bus_id(pci),
        "device_name": name,
        "provider": "nvidia-smi",
    }
    record["identity_receipt_sha256"] = hashlib.sha256(common.canonical_json(record)).hexdigest()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--final-name", required=True)
    parser.add_argument("--transaction-id", required=True)
    parser.add_argument("--identity-only", action="store_true")
    arguments = parser.parse_args()
    import cupy as cp

    source_root, inventory = development_source_root()
    backend = cupy_backend.build_backend(cp)
    identity = independent_gpu_identity(backend)
    if arguments.identity_only:
        # The strict V7 environment boundary requires a real, measured resource
        # admission/pack record rather than accepting initialization-only
        # telemetry.  This one-symbol source-free probe populates that record
        # without fitting a candidate or touching any payload.
        backend.pack_streams([
            (bytes((0,)), bytes((0,)), bytes((0, 128))),
        ])
        environment = backend.environment_receipt()
        stage._validate_environment_identity(protocol, environment, identity)
        print(json.dumps({"environment": environment, "independent_gpu_identity": identity}, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    all150 = stage.gpu_preflight_all_150(common, backend, source_root)
    representative = stage.representative_outer_fold_benchmark(
        common, protocol, codec, semantic, backend, source_root
    )
    bound = {
        "schema": "uwfa-sc-v7-bound-source-preflight",
        "source_snapshot_root_sha256": source_root,
        "all150": all150,
        "representative": representative,
        "independent_gpu_identity": identity,
    }
    receipt_sha = hashlib.sha256(common.canonical_json(bound)).hexdigest()
    evidence = stage.BoundEvidence(
        baseline_plan_sha256="10" * 32,
        baseline_score_sha256="11" * 32,
        universal_decoder_sha256="12" * 32,
        producer_manifest_sha256="13" * 32,
        audit_bootstrap_sha256="14" * 32,
        source_full_geometry_sha256="15" * 32,
        source_structural_geometry_sha256="16" * 32,
        extraction_program_sha256="17" * 32,
        universal_adapter_sha256="18" * 32,
        pipeline_sha256="19" * 32,
        source_snapshot_root_sha256=source_root,
        source_preflight_receipt_sha256=receipt_sha,
    )
    typed = stage.SourcePreflightEvidence(all150, representative, identity, receipt_sha)
    stage.validate_source_preflight(common, protocol, typed, evidence)
    receipt = {
        "schema": "uwfa-sc-v7-direct-copy-development-gpu-receipt",
        "status": "PASS_SOURCE_FREE_DEVELOPMENT_REPLAY_NO_CLAIM_AUTHORITY",
        "development_source_root_sha256": source_root,
        "development_source_inventory": inventory,
        "bound_source_preflight_receipt_sha256": receipt_sha,
        "all150": all150,
        "representative": representative,
        "independent_gpu_identity": identity,
        "public_commit_evidence": False,
        "payload_authority_granted": False,
    }
    parent = common.RetainedOutputParent.open_path_source_only(
        Path(arguments.output_parent).absolute(), hashlib.sha256(common.canonical_json(inventory)).hexdigest()
    )
    try:
        with common.CompletionLastOutput(parent, arguments.final_name, arguments.transaction_id) as transaction:
            transaction.write_new("GPU_DEV_RECEIPT.json", common.pretty_json(receipt))
            transaction.complete(list(transaction.members), source_root)
        with envelope.verify_completed_under_parent(
            common,
            parent,
            arguments.final_name,
            expected_source_manifest_sha256=source_root,
        ) as verified_bundle:
            committed_receipt_bytes = verified_bundle.read_member_bytes("GPU_DEV_RECEIPT.json")
            if committed_receipt_bytes != common.pretty_json(receipt):
                raise RuntimeError("held committed GPU receipt differs from emitted receipt")
            verified_output = verified_bundle.metadata
    finally:
        parent.close()
    print(json.dumps({
        "status": receipt["status"],
        "development_source_root_sha256": source_root,
        "bound_source_preflight_receipt_sha256": receipt_sha,
        "parent_commit_sha256": verified_output["parent_commit_sha256"],
        "directory_root_sha256": verified_output["directory_root_sha256"],
        "output": str(Path(arguments.output_parent).absolute() / arguments.final_name),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
