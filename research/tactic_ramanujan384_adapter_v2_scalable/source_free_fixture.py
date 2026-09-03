#!/usr/bin/env python3
"""Build and execute an authenticated source-free target-rate fixture."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _write(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _bf16(values: np.ndarray) -> bytes:
    f32 = np.ascontiguousarray(values, dtype="<f4")
    return (f32.view("<u4") >> 16).astype("<u2").tobytes(order="C")


def build(
    directory: Path,
    *,
    core: Any,
    io: Any,
    decoder_class: Any,
    decoder_source_path: Path,
    intermediate: int = 128,
    hidden: int = 2048,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shape = core.define_shape(intermediate, hidden)
    coordinate = np.arange(shape.role_values, dtype=np.int64)
    specifications = {
        "gate": (7, 0, 2.0 ** -7),
        "up": (11, 1, 2.0 ** -8),
        "down_transposed": (13, 2, 2.0 ** -8),
    }
    sources = {}
    local_coordinate = coordinate % core.BLOCK_VALUES
    for role, (period, shift, scale) in specifications.items():
        lookup = np.asarray(
            [core.ramanujan_sum(period, index) for index in range(period)], dtype=np.float32
        )
        sources[role] = np.ascontiguousarray(
            scale * lookup[(local_coordinate - shift) % period], dtype="<f4"
        )
    coarse = bytes(shape.coarse_bytes)
    coarse_path = directory / "COARSE.synthetic"
    _write(coarse_path, coarse)
    coarse_sha = hashlib.sha256(coarse).hexdigest()
    zero_f32 = np.zeros(shape.role_values, dtype="<f4").tobytes(order="C")
    zero_sha = hashlib.sha256(zero_f32).hexdigest()

    source_rows = []
    source_paths = {}
    coarse_reconstruction_paths = {}
    for role in core.ROLE_ORDER:
        payload = _bf16(sources[role])
        path = directory / f"{role}.bf16"
        reconstruction_path = directory / f"{role}.coarse.f32"
        _write(path, payload)
        _write(reconstruction_path, zero_f32)
        source_paths[role] = path
        coarse_reconstruction_paths[role] = reconstruction_path
        source_rows.append({
            "role": role, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

    input_manifest_payload = canonical_json({
        "schema": "tactic-ramanujan384-target-rate-source-free-input-v2",
        "shape": [intermediate, hidden], "roles": source_rows,
        "coarse": {"bytes": len(coarse), "sha256": coarse_sha},
        "model_payload_accessed": False,
    })
    input_manifest_path = directory / "INPUT_MANIFEST.json"
    _write(input_manifest_path, input_manifest_payload)
    source_auditor_payload = canonical_json({
        "schema": "tactic-ramanujan384-source-free-source-auditor-v2",
        "independent_fixture": True, "model_payload_accessed": False,
    })
    source_auditor_path = directory / "SOURCE_AUDITOR_MANIFEST.json"
    _write(source_auditor_path, source_auditor_payload)
    source_audit = {
        "status": "INDEPENDENT_SOURCE_AND_COARSE_RECONSTRUCTION_AUDIT_PASS",
        "input_manifest_sha256": hashlib.sha256(input_manifest_payload).hexdigest(),
        "auditor_source_manifest_sha256": hashlib.sha256(source_auditor_payload).hexdigest(),
        "publication_members": [{"name": "COARSE.bin", "bytes": len(coarse),
                                 "sha256": coarse_sha}],
        "input_roles": source_rows,
        "reconstruction_f32_sha256": {role: zero_sha for role in core.ROLE_ORDER},
        "source_free_fixture": True,
    }
    source_audit_payload = canonical_json(source_audit)
    source_audit_path = directory / "SOURCE_AUDIT.json"
    _write(source_audit_path, source_audit_payload)
    source_audit_sha = hashlib.sha256(source_audit_payload).hexdigest()

    role_inputs = []
    for row in source_rows:
        role = row["role"]
        binding = {
            "schema": io.SCHEMA, "status": io.STATUS, "role": role,
            "shape": [intermediate, hidden],
            "normalized_layout": "intermediate_by_hidden_row_major",
            "source": {"bytes": row["bytes"], "sha256": row["sha256"]},
            "coarse_artifact": {"bytes": len(coarse), "sha256": coarse_sha},
            "coarse_reconstruction": {"bytes": len(zero_f32), "sha256": zero_sha},
            "source_audit": source_audit_sha,
            "input_manifest_sha256": hashlib.sha256(input_manifest_payload).hexdigest(),
            "auditor_source_manifest_sha256": hashlib.sha256(source_auditor_payload).hexdigest(),
        }
        binding_payload = canonical_json(binding)
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
            "coarse_reconstruction_f32_path": coarse_reconstruction_paths[role],
        })

    decoder_source_payload = decoder_source_path.read_bytes()
    decoder_source_sha = hashlib.sha256(decoder_source_payload).hexdigest()
    decoder_manifest_payload = canonical_json({
        "schema": "source-free-zero-coarse-decoder-manifest-v2",
        "members": [{"name": decoder_source_path.name,
                     "bytes": len(decoder_source_payload), "sha256": decoder_source_sha}],
    })
    decoder_manifest_path = directory / "COARSE_DECODER_SOURCE_MANIFEST.json"
    _write(decoder_manifest_path, decoder_manifest_payload)
    coarse_auditor_payload = canonical_json({
        "schema": "source-free-coarse-decoder-auditor-manifest-v2",
        "independent_fixture": True,
    })
    coarse_auditor_path = directory / "COARSE_AUDITOR_SOURCE_MANIFEST.json"
    _write(coarse_auditor_path, coarse_auditor_payload)
    capability_id = decoder_class.capability_id
    coarse_audit = {
        "status": "INDEPENDENT_COARSE_DECODER_AUDIT_PASS",
        "capability_id": capability_id,
        "decoder_source_sha256": decoder_source_sha,
        "decoder_source_manifest_sha256": hashlib.sha256(decoder_manifest_payload).hexdigest(),
        "auditor_source_manifest_sha256": hashlib.sha256(coarse_auditor_payload).hexdigest(),
        "literal_decode_from_payload_only_pass": True,
        "source_weights_inaccessible_to_decoder": True,
        "independent_output_hashes_recorded": True,
        "source_free_fixture": True,
    }
    coarse_audit_payload = canonical_json(coarse_audit)
    coarse_audit_path = directory / "COARSE_DECODER_AUDIT.json"
    _write(coarse_audit_path, coarse_audit_payload)
    capability = {
        "schema": "tactic-independent-coarse-decoder-capability-v2",
        "status": "INDEPENDENT_COARSE_DECODER_AUDIT_REQUIRED",
        "capability_id": capability_id,
        "decoder_source_sha256": decoder_source_sha,
        "decoder_source_manifest_sha256": hashlib.sha256(decoder_manifest_payload).hexdigest(),
        "auditor_source_manifest_sha256": hashlib.sha256(coarse_auditor_payload).hexdigest(),
        "independent_audit_receipt_sha256": hashlib.sha256(coarse_audit_payload).hexdigest(),
    }
    capability_payload = canonical_json(capability)
    capability_path = directory / "COARSE_DECODER_CAPABILITY.json"
    _write(capability_path, capability_payload)
    decoder = decoder_class(coarse_sha)
    capability_arguments = {
        "decoder": decoder,
        "capability_path": capability_path,
        "expected_capability_sha256": hashlib.sha256(capability_payload).hexdigest(),
        "decoder_source_path": decoder_source_path,
        "decoder_source_manifest_path": decoder_manifest_path,
        "auditor_source_manifest_path": coarse_auditor_path,
        "independent_audit_receipt_path": coarse_audit_path,
    }
    return role_inputs, capability_arguments


def run(xp: Any, *, core: Any, io: Any, capability_api: Any, adapter: Any,
        decoder_class: Any, decoder_source_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        role_inputs, capability = build(
            directory, core=core, io=io, decoder_class=decoder_class,
            decoder_source_path=decoder_source_path,
        )
        result = adapter.run_authenticated_expert(
            xp, core=core, io=io, coarse_capability_api=capability_api,
            role_inputs=role_inputs, coarse_capability_arguments=capability,
            composite_output_path=directory / "COMPOSITE.trm384s2",
        )
        return result
