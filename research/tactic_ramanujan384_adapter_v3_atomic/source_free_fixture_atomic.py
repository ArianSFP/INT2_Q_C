#!/usr/bin/env python3
"""Construct the target-rate periodic source and audited byte-worker fixture."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _write(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _bf16(values: np.ndarray) -> bytes:
    f32 = np.ascontiguousarray(values, dtype="<f4")
    return (f32.view("<u4") >> 16).astype("<u2").tobytes(order="C")


def _closure(directory: Path, schema: str, members: dict[str, bytes]) -> tuple[bytes, str]:
    directory.mkdir()
    rows = []
    for name in sorted(members):
        payload = members[name]
        _write(directory / name, payload)
        rows.append({"name": name, "bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()})
    root = hashlib.sha256(canonical_json(rows)).hexdigest()
    return canonical_json({"schema": schema, "source_root_sha256": root,
                           "members": rows}), root


def build(directory: Path, *, core: Any, io: Any,
          intermediate: int = 128, hidden: int = 2048):
    shape = core.define_shape(intermediate, hidden)
    coordinate = np.arange(shape.role_values, dtype=np.int64)
    local_coordinate = coordinate % core.BLOCK_VALUES
    specifications = {
        "gate": (7, 0, 2.0 ** -7),
        "up": (11, 1, 2.0 ** -8),
        "down_transposed": (13, 2, 2.0 ** -8),
    }
    sources = {}
    for role, (period, shift, scale) in specifications.items():
        lookup = np.asarray(
            [core.ramanujan_sum(period, index) for index in range(period)],
            dtype=np.float32,
        )
        sources[role] = np.ascontiguousarray(
            scale * lookup[(local_coordinate - shift) % period], dtype="<f4"
        )
    coarse = bytes(shape.coarse_bytes)
    coarse_path = directory / "COARSE.synthetic"
    _write(coarse_path, coarse)
    coarse_sha = hashlib.sha256(coarse).hexdigest()
    zero_f32 = bytes(4 * shape.role_values)
    zero_sha = hashlib.sha256(zero_f32).hexdigest()
    source_rows = []
    source_paths = {}
    reconstruction_paths = {}
    for role in core.ROLE_ORDER:
        payload = _bf16(sources[role])
        source_path = directory / f"{role}.bf16"
        reconstruction_path = directory / f"{role}.coarse.f32"
        _write(source_path, payload)
        _write(reconstruction_path, zero_f32)
        source_paths[role] = source_path
        reconstruction_paths[role] = reconstruction_path
        source_rows.append({"role": role, "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()})
    input_manifest_payload = canonical_json({
        "schema": "tactic-ramanujan384-target-rate-source-free-input-v3",
        "shape": [intermediate, hidden], "roles": source_rows,
        "coarse": {"bytes": len(coarse), "sha256": coarse_sha},
        "model_payload_accessed": False,
    })
    input_manifest_path = directory / "INPUT_MANIFEST.json"
    _write(input_manifest_path, input_manifest_payload)
    source_auditor_payload = canonical_json({
        "schema": "tactic-ramanujan384-source-free-source-auditor-v3",
        "synthetic_fixture": True, "model_payload_accessed": False,
    })
    source_auditor_path = directory / "SOURCE_AUDITOR_MANIFEST.json"
    _write(source_auditor_path, source_auditor_payload)
    source_audit_payload = canonical_json({
        "status": "INDEPENDENT_SOURCE_AND_COARSE_RECONSTRUCTION_AUDIT_PASS",
        "input_manifest_sha256": hashlib.sha256(input_manifest_payload).hexdigest(),
        "auditor_source_manifest_sha256": hashlib.sha256(source_auditor_payload).hexdigest(),
        "publication_members": [{"name": "COARSE.bin", "bytes": len(coarse),
                                 "sha256": coarse_sha}],
        "input_roles": source_rows,
        "reconstruction_f32_sha256": {role: zero_sha for role in core.ROLE_ORDER},
        "source_free_fixture": True,
    })
    source_audit_path = directory / "SOURCE_AUDIT.json"
    _write(source_audit_path, source_audit_payload)
    source_audit_sha = hashlib.sha256(source_audit_payload).hexdigest()
    role_inputs = []
    for row in source_rows:
        role = row["role"]
        binding_payload = canonical_json({
            "schema": io.SCHEMA, "status": io.STATUS, "role": role,
            "shape": [intermediate, hidden],
            "normalized_layout": "intermediate_by_hidden_row_major",
            "source": {"bytes": row["bytes"], "sha256": row["sha256"]},
            "coarse_artifact": {"bytes": len(coarse), "sha256": coarse_sha},
            "coarse_reconstruction": {"bytes": len(zero_f32), "sha256": zero_sha},
            "source_audit": source_audit_sha,
            "input_manifest_sha256": hashlib.sha256(input_manifest_payload).hexdigest(),
            "auditor_source_manifest_sha256": hashlib.sha256(source_auditor_payload).hexdigest(),
        })
        binding_path = directory / f"{role}.binding.json"
        _write(binding_path, binding_payload)
        role_inputs.append({
            "binding_path": binding_path,
            "expected_binding_sha256": hashlib.sha256(binding_payload).hexdigest(),
            "source_audit_receipt_path": source_audit_path,
            "expected_source_audit_receipt_sha256": source_audit_sha,
            "input_manifest_path": input_manifest_path,
            "auditor_source_manifest_path": source_auditor_path,
            "coarse_artifact_path": coarse_path,
            "source_bf16_path": source_paths[role],
            "coarse_reconstruction_f32_path": reconstruction_paths[role],
        })

    program_payload = canonical_json({
        "schema": "tactic-coarse-byte-worker-program-v3", "version": 1,
        "imports": [], "opcode": "ZERO_F32_LE", "coarse_sha256": coarse_sha,
        "shape": [intermediate, hidden], "roles": list(core.ROLE_ORDER),
    })
    worker_spec_payload = canonical_json({
        "schema": "tactic-coarse-byte-worker-spec-v3", "imports": [],
        "opcodes": ["ZERO_F32_LE"], "path_operands": 0, "callback_operands": 0,
        "synthetic_fixture": True,
    })
    worker_directory = directory / "WORKER_SOURCE"
    worker_manifest_payload, worker_root = _closure(
        worker_directory, "tactic-coarse-worker-source-manifest-v3",
        {"worker_program.json": program_payload, "worker_spec.json": worker_spec_payload},
    )
    worker_manifest_path = directory / "WORKER_SOURCE_MANIFEST.json"
    _write(worker_manifest_path, worker_manifest_payload)
    worker_manifest_sha = hashlib.sha256(worker_manifest_payload).hexdigest()
    auditor_method_payload = canonical_json({
        "schema": "tactic-coarse-worker-fixture-audit-method-v3",
        "tests": ["literal-payload-only", "zero-import", "no-path", "determinism",
                  "mutation", "wrong-hash", "wrong-shape", "wrong-role", "extra-key",
                  "noncanonical-json", "closure-extra", "closure-link"],
        "synthetic_fixture": True,
    })
    auditor_directory = directory / "AUDITOR_SOURCE"
    auditor_manifest_payload, auditor_root = _closure(
        auditor_directory, "tactic-coarse-worker-auditor-source-manifest-v3",
        {"audit_method.json": auditor_method_payload},
    )
    auditor_manifest_path = directory / "AUDITOR_SOURCE_MANIFEST.json"
    _write(auditor_manifest_path, auditor_manifest_payload)
    auditor_manifest_sha = hashlib.sha256(auditor_manifest_payload).hexdigest()
    capability_id = "TACTIC_RAMANUJAN384_SOURCE_FREE_ZERO_BYTE_WORKER_V3"
    program_sha = hashlib.sha256(program_payload).hexdigest()
    receipt_payload = canonical_json({
        "schema": "tactic-independent-coarse-worker-audit-receipt-v3",
        "status": "INDEPENDENT_COARSE_WORKER_AUDIT_PASS",
        "capability_id": capability_id, "program_sha256": program_sha,
        "worker_source_manifest_sha256": worker_manifest_sha,
        "worker_source_root_sha256": worker_root,
        "auditor_source_manifest_sha256": auditor_manifest_sha,
        "auditor_source_root_sha256": auditor_root,
        "coarse_sha256": coarse_sha, "shape": [intermediate, hidden],
        "role_order": list(core.ROLE_ORDER),
        "output_f32_sha256_by_role": {role: zero_sha for role in core.ROLE_ORDER},
        "literal_payload_only_pass": True, "zero_import_no_path_pass": True,
        "deterministic_output_hashes_recorded": True, "hostile_tests_passed": 12,
    })
    receipt_path = directory / "WORKER_AUDIT_RECEIPT.json"
    _write(receipt_path, receipt_payload)
    receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
    capability_payload = canonical_json({
        "schema": "tactic-coarse-byte-worker-capability-v3",
        "status": "INDEPENDENT_ZERO_IMPORT_WORKER_AUDIT_REQUIRED",
        "capability_id": capability_id,
        "program_name": "worker_program.json", "program_sha256": program_sha,
        "worker_source_manifest_sha256": worker_manifest_sha,
        "worker_source_root_sha256": worker_root,
        "auditor_source_manifest_sha256": auditor_manifest_sha,
        "auditor_source_root_sha256": auditor_root,
        "independent_audit_receipt_sha256": receipt_sha,
    })
    capability_path = directory / "WORKER_CAPABILITY.json"
    _write(capability_path, capability_payload)
    worker_arguments = {
        "capability_path": capability_path,
        "expected_capability_sha256": hashlib.sha256(capability_payload).hexdigest(),
        "worker_source_directory": worker_directory,
        "worker_source_manifest_path": worker_manifest_path,
        "expected_worker_source_manifest_sha256": worker_manifest_sha,
        "expected_worker_source_root_sha256": worker_root,
        "auditor_source_directory": auditor_directory,
        "auditor_source_manifest_path": auditor_manifest_path,
        "expected_auditor_source_manifest_sha256": auditor_manifest_sha,
        "expected_auditor_source_root_sha256": auditor_root,
        "independent_audit_receipt_path": receipt_path,
        "expected_independent_audit_receipt_sha256": receipt_sha,
    }
    return role_inputs, worker_arguments


def run(xp: Any, *, core: Any, io: Any, adapter: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        role_inputs, worker_arguments = build(directory, core=core, io=io)
        result = adapter.run_authenticated_expert(
            xp, core=core, io=io,
            role_inputs=role_inputs, coarse_worker_arguments=worker_arguments,
            composite_output_path=directory / "COMPOSITE.trm384a3",
        )
        result["synthetic_fixture_mechanism_only"] = True
        return result
