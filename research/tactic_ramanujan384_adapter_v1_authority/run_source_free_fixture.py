#!/usr/bin/env python3
"""Synthetic, source-free literal composite replay; no model/coarse payload."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str) -> Any:
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _bf16_payload(values: np.ndarray) -> bytes:
    f32 = np.ascontiguousarray(values, dtype="<f4")
    return (f32.view("<u4") >> 16).astype("<u2").tobytes(order="C")


def _write(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def build_authenticated_fixture(directory: Path) -> tuple[list[dict[str, Any]], Any]:
    auth = _load("tactic_ramanujan384_fixture_auth", "authenticated_io.py")
    contract = _load("tactic_ramanujan384_fixture_contract", "contract.py")
    shape = contract.define_shape(64, 64)
    coordinate = np.arange(shape.role_values, dtype=np.float64)
    sources = {
        "gate": 0.015 * ((coordinate % 7.0) - 3.0) + 0.001 * ((coordinate % 11.0) - 5.0),
        "up": 0.011 * ((coordinate % 5.0) - 2.0) - 0.002 * ((coordinate % 13.0) - 6.0),
        "down_transposed": 0.013 * ((coordinate % 9.0) - 4.0) + 0.0015 * ((coordinate % 17.0) - 8.0),
    }
    coarse_payload = bytes(shape.coarse_bytes)
    coarse_path = directory / "COARSE.synthetic"
    _write(coarse_path, coarse_payload)
    source_records = []
    reconstruction_hashes = {}
    source_paths = {}
    reconstruction_paths = {}
    for role, values in sources.items():
        source_payload = _bf16_payload(values)
        reconstruction_payload = np.zeros(shape.role_values, dtype="<f4").tobytes(order="C")
        source_path = directory / f"{role}.bf16"
        reconstruction_path = directory / f"{role}.coarse.f32"
        _write(source_path, source_payload)
        _write(reconstruction_path, reconstruction_payload)
        source_paths[role] = source_path
        reconstruction_paths[role] = reconstruction_path
        reconstruction_hashes[role] = hashlib.sha256(reconstruction_payload).hexdigest()
        source_records.append({
            "role": role,
            "bytes": len(source_payload),
            "sha256": hashlib.sha256(source_payload).hexdigest(),
        })

    input_manifest = {
        "schema": "tactic-ramanujan384-source-free-input-manifest-v1",
        "shape": [shape.intermediate, shape.hidden],
        "roles": source_records,
        "coarse": {
            "bytes": len(coarse_payload),
            "sha256": hashlib.sha256(coarse_payload).hexdigest(),
        },
        "qwen_payload_accessed": False,
    }
    input_payload = canonical_json(input_manifest)
    input_path = directory / "INPUT_MANIFEST.json"
    _write(input_path, input_payload)
    auditor_manifest = {
        "schema": "tactic-ramanujan384-source-free-auditor-manifest-v1",
        "members": [{"name": "fixture-independent-parser.py", "sha256": "3" * 64}],
        "qwen_payload_accessed": False,
    }
    auditor_payload = canonical_json(auditor_manifest)
    auditor_path = directory / "AUDITOR_SOURCE_MANIFEST.json"
    _write(auditor_path, auditor_payload)
    audit = {
        "auditor_source_manifest_sha256": hashlib.sha256(auditor_payload).hexdigest(),
        "input_manifest_sha256": hashlib.sha256(input_payload).hexdigest(),
        "independent_packet_parser_and_causal_CPU_decoder_used": True,
        "literal_COARSE_canonical_reencode_matches": True,
        "publication_members": [{
            "name": "COARSE.bin",
            "bytes": len(coarse_payload),
            "sha256": hashlib.sha256(coarse_payload).hexdigest(),
        }],
        "input_roles": source_records,
        "recomputed": {"reconstruction_f32_sha256": reconstruction_hashes},
        "source_free_fixture": True,
    }
    audit_payload = canonical_json(audit)
    audit_path = directory / "AUDIT_RECEIPT.json"
    _write(audit_path, audit_payload)

    role_inputs = []
    for row in source_records:
        role = row["role"]
        binding = {
            "schema": "tactic-ramanujan384-input-binding-v0",
            "status": "INDEPENDENT_COARSE_AND_SOURCE_HASHES_REQUIRED",
            "role": role,
            "shape": [shape.intermediate, shape.hidden],
            "normalized_layout": "intermediate_by_hidden_row_major",
            "source": {"bytes": row["bytes"], "sha256": row["sha256"]},
            "coarse_artifact": {
                "bytes": len(coarse_payload),
                "sha256": hashlib.sha256(coarse_payload).hexdigest(),
            },
            "coarse_reconstruction": {
                "bytes": 4 * shape.role_values,
                "sha256": reconstruction_hashes[role],
            },
            "independent_audit": {
                "receipt_sha256": hashlib.sha256(audit_payload).hexdigest(),
                "auditor_source_manifest_sha256": hashlib.sha256(auditor_payload).hexdigest(),
            },
            "input_manifest_sha256": hashlib.sha256(input_payload).hexdigest(),
        }
        binding_payload = canonical_json(binding)
        binding_path = directory / f"{role}.binding.json"
        _write(binding_path, binding_payload)
        role_inputs.append({
            "binding_path": binding_path,
            "expected_binding_sha256": hashlib.sha256(binding_payload).hexdigest(),
            "audit_receipt_path": audit_path,
            "coarse_artifact_path": coarse_path,
            "source_bf16_path": source_paths[role],
            "coarse_reconstruction_f32_path": reconstruction_paths[role],
            "input_manifest_path": input_path,
            "auditor_source_manifest_path": auditor_path,
        })

    def zero_coarse_decoder(payload: bytes, intermediate: int, hidden: int,
                            role_order: tuple[str, ...]) -> dict[str, np.ndarray]:
        assert payload == coarse_payload
        assert (intermediate, hidden) == (shape.intermediate, shape.hidden)
        return {
            role: np.zeros((intermediate, hidden), dtype="<f4") for role in role_order
        }

    return role_inputs, zero_coarse_decoder


def run(xp: Any = np) -> dict[str, Any]:
    adapter = _load("tactic_ramanujan384_fixture_adapter", "adapter.py")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        role_inputs, decoder = build_authenticated_fixture(directory)
        result = adapter.run_authenticated_expert(
            xp,
            role_inputs=role_inputs,
            coarse_decoder=decoder,
            composite_output_path=directory / "COMPOSITE.trm384",
        )
        return {
            "schema": "tactic-ramanujan384-authority-source-free-fixture-v1",
            "status": "PASS_SOURCE_FREE_LITERAL_COARSE_FINE_WEIGHT_REPLAY",
            "adapter_status": result["status"],
            "literal_composite_reconstructed_to_weights": result[
                "literal_composite_reconstructed_to_weights"
            ],
            "independent_source_domain_fp64_rescore": result[
                "independent_source_domain_fp64_rescore"
            ],
            "every_defined_candidate_packet_replayed_before_selection": result[
                "every_defined_candidate_packet_replayed_before_selection"
            ],
            "actual_input_manifests_opened": result["actual_input_manifests_opened"],
            "actual_auditor_manifests_opened": result["actual_auditor_manifests_opened"],
            "instrumented_file_read_amplification": result["read_trace"][
                "instrumented_file_read_amplification"
            ],
            "layout_bound_is_not_a_measurement": result["layout"][
                "layout_bound_is_not_a_measurement"
            ],
            "relative_mse": result["relative_mse"],
            "F": result["F"],
            "physical_rate_bpw": result["physical_rate_bpw"],
            "qwen_payload_accessed": False,
            "coarse_model_payload_accessed": False,
            "network_accessed": False,
        }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, separators=(",", ":"), allow_nan=False))
