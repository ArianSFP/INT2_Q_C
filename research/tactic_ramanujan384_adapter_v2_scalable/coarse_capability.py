#!/usr/bin/env python3
"""Authenticate an independently audited literal coarse-decoder capability."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any


class CoarseCapabilityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CoarseCapabilityError(message)


def authenticate(
    *,
    decoder: Any,
    capability_path: Path,
    expected_capability_sha256: str,
    decoder_source_path: Path,
    decoder_source_manifest_path: Path,
    auditor_source_manifest_path: Path,
    independent_audit_receipt_path: Path,
    io: Any,
) -> dict[str, Any]:
    capability_payload = io.read_regular(capability_path, 2 << 20)
    require(hashlib.sha256(capability_payload).hexdigest() == expected_capability_sha256,
            "coarse capability SHA256")
    capability = io.strict_json(capability_payload, "coarse decoder capability")
    require(set(capability) == {
        "schema", "status", "capability_id", "decoder_source_sha256",
        "decoder_source_manifest_sha256", "auditor_source_manifest_sha256",
        "independent_audit_receipt_sha256",
    }, "coarse capability exact schema")
    require(capability["schema"] == "tactic-independent-coarse-decoder-capability-v2"
            and capability["status"] == "INDEPENDENT_COARSE_DECODER_AUDIT_REQUIRED",
            "coarse capability identity")
    source_payload = io.read_regular(decoder_source_path, 16 << 20)
    decoder_manifest = io.read_regular(decoder_source_manifest_path, 16 << 20)
    auditor_manifest = io.read_regular(auditor_source_manifest_path, 16 << 20)
    receipt_payload = io.read_regular(independent_audit_receipt_path, 16 << 20)
    require(hashlib.sha256(source_payload).hexdigest() == capability["decoder_source_sha256"],
            "actual coarse decoder source SHA256")
    require(hashlib.sha256(decoder_manifest).hexdigest()
            == capability["decoder_source_manifest_sha256"],
            "actual coarse decoder source manifest SHA256")
    require(hashlib.sha256(auditor_manifest).hexdigest()
            == capability["auditor_source_manifest_sha256"],
            "actual coarse auditor source manifest SHA256")
    require(hashlib.sha256(receipt_payload).hexdigest()
            == capability["independent_audit_receipt_sha256"],
            "actual coarse independent audit receipt SHA256")
    decoder_document = io.strict_json(decoder_manifest, "coarse decoder source manifest")
    io.strict_json(auditor_manifest, "coarse auditor source manifest")
    members = decoder_document.get("members", [])
    require(isinstance(members, list) and all(isinstance(row, dict) for row in members),
            "coarse decoder manifest members")
    names = [row.get("name") for row in members]
    require(len(names) == len(set(names)), "unique coarse decoder manifest members")
    by_name = {row.get("name"): row for row in members}
    require(by_name.get(decoder_source_path.name) == {
        "name": decoder_source_path.name,
        "bytes": len(source_payload),
        "sha256": capability["decoder_source_sha256"],
    }, "coarse decoder source is a manifest member")
    receipt = io.strict_json(receipt_payload, "coarse decoder independent audit")
    require(receipt.get("status") == "INDEPENDENT_COARSE_DECODER_AUDIT_PASS",
            "independent coarse decoder PASS")
    require(receipt.get("capability_id") == capability["capability_id"],
            "coarse capability identity pin")
    for key in (
        "decoder_source_sha256", "decoder_source_manifest_sha256",
        "auditor_source_manifest_sha256",
    ):
        require(receipt.get(key) == capability[key], f"coarse audit {key}")
    require(receipt.get("literal_decode_from_payload_only_pass") is True
            and receipt.get("source_weights_inaccessible_to_decoder") is True
            and receipt.get("independent_output_hashes_recorded") is True,
            "coarse decoder independent causal evidence")
    source_file = inspect.getsourcefile(type(decoder))
    require(source_file is not None and Path(source_file).resolve(strict=True)
            == decoder_source_path.resolve(strict=True), "runtime coarse decoder source identity")
    require(getattr(decoder, "capability_id", None) == capability["capability_id"]
            and callable(getattr(decoder, "decode_literal", None)),
            "runtime coarse decoder interface")
    return {
        "decoder": decoder,
        "capability_id": capability["capability_id"],
        "capability_sha256": expected_capability_sha256,
        "decoder_source_sha256": capability["decoder_source_sha256"],
        "independent_audit_receipt_sha256": capability["independent_audit_receipt_sha256"],
        "independently_audited": True,
    }
