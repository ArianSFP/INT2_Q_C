#!/usr/bin/env python3
"""Fail-closed authority gate for the completed UWFA-SC v9 publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "uwfa-sc-v9-primary-independent-result-audit-v0"
AUDIT_STATUS = "PASS_FAIL_CLOSED_NONPROMOTING_PRIMARY_RESULT_AUDIT"
AUDIT_SOURCE_MANIFEST_SHA256 = "885f41e27c439c808e2118de52184feaec58efe9f14bbc0e02a377e3b189f5ee"
SOURCE_WEIGHTS = 28_311_552
BASELINE_ARTIFACT_BYTES = 8_847_360
BASELINE_ARTIFACT_SHA256 = "4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b"
CANDIDATE_BYTES = 8_892_416
CANDIDATE_SHA256 = "4475b782b06776c84ff2b4f795d3a10f5857f8477cd9c54e9bd750296b41fb79"
RECONSTRUCTION_SHA256 = "84309366c3bbc6459d461f1b3e23c48944623ae7dfb8bab7e0c4698f3e661d67"
EXPECTED_SOURCE_CLOSURE = {
    "v9_manifest_sha256": "d1e3eaff6762df2e273f6e3f4216ff9110abe74a7534a0098544a4ceef632c5e",
    "v9_source_root_sha256": "4f99644a8d36eb15d6ff966db25f01e3e10f6d0f481af5fe0fd507c647eadca5",
    "v9_runner_sha256": "d1ff04ce3c2cc36208e464eaed943d6c94eb91a47e9d3c460b2d562b7162cc4d",
    "support_sha256": "399cb25260d34ec299cc91a17f129da9be5ba5b799c961e43f0c1b0637ee0174",
    "v8_manifest_sha256": "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6",
    "v8_source_root_sha256": "be06cf4d6c474a01517c4062f448b0c41c7f59d31724d6d5af380b8c064de4fa",
    "strata_common_sha256": "3f085c9531b714d0d7877388f54ae50495dc3ea631491563abceb4db55608fd1",
    "frozen_auditor_sha256": "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e",
    "artifact_sha256": BASELINE_ARTIFACT_SHA256,
    "artifact_bytes": BASELINE_ARTIFACT_BYTES,
}
EXPECTED_PUBLICATION = {
    "BOUND_BASELINE_SCORE.json": (823, "308af3bedcb17e523495efc726b0e3b35bc621abb45efaec6be522adcb55aca8"),
    "COMPLETE.json": (1374, "dc3e64e5af61440ce630f71449dbabb8c1537cd8b38f2eb19e34c622902b68bf"),
    "DECODER_BUNDLE.json": (1383, "20d1d8918056297f06d79b8c4d5b0934ed41f6fc0da8bf40e63835f507f3e0ec"),
    "IDENTITY_FRAMING.bin": (8_888_320, "96a2dca6b57a39a2cb3b73ec17812ed2819c4926b9dcfabb052f355bf848d2b0"),
    "RESULT.json": (1_999_689, "771021d8862ac2c802cd7751b2a8679db3004ad3c5b8ba6102e3c5062d96ec0f"),
    "SOURCE_PREFLIGHT.json": (1_666_660, "655af2ec59b671e0cc4b5ae40ada648208596e5c47be159ac03bbc5bc8ffdf04"),
    "UWFCV8.bin": (CANDIDATE_BYTES, CANDIDATE_SHA256),
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), label)
    return value


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, f"{label} duplicate key")
            output[key] = value
        return output

    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(GateError(f"{label} nonfinite {token}")),
    )
    require(isinstance(value, dict), f"{label} object")
    return value


def _fraction(row: Any, label: str) -> Fraction:
    require(isinstance(row, dict) and set(row) == {"numerator", "denominator", "float"}, label)
    numerator, denominator = row["numerator"], row["denominator"]
    require(type(numerator) is int and type(denominator) is int and denominator > 0, label)
    value = Fraction(numerator, denominator)
    require(math.isfinite(row["float"]) and float(value) == row["float"], label)
    return value


@dataclass(frozen=True)
class AuditAuthority:
    receipt_sha256: str
    publication_members: dict[str, dict[str, Any]]
    decision_commitment_sha256: str
    reconstruction_sha256: str
    candidate_rate: Fraction
    inherited_relative_mse: float
    primary_result_status: str


def authorize_v9_audit_receipt(payload: bytes, *, expected_receipt_sha256: str) -> AuditAuthority:
    require(isinstance(payload, bytes) and payload, "audit receipt bytes")
    expected = _digest(expected_receipt_sha256, "expected audit receipt digest")
    observed = sha256(payload)
    require(observed == expected, "audit receipt external digest")
    row = strict_json(payload, "v9 audit receipt")
    required = {
        "schema", "status", "positive_claim_authority", "controls_run_by_this_audit",
        "shuffles_run_by_this_audit", "coordinate_diagnostic_run_by_this_audit",
        "primary_result_status", "publication_members", "source_closure",
        "scientific_replay", "runtime_replay", "literal_container_audit",
        "evidence_limitations",
    }
    require(set(row) == required, "audit receipt exact schema")
    require(row["schema"] == AUDIT_SCHEMA and row["status"] == AUDIT_STATUS, "audit receipt status")
    require(row["positive_claim_authority"] is False, "nonpromoting audit authority")
    require(row["controls_run_by_this_audit"] is False and row["shuffles_run_by_this_audit"] is False and row["coordinate_diagnostic_run_by_this_audit"] is False, "audit scope")
    require(isinstance(row["primary_result_status"], str) and row["primary_result_status"], "primary result status")
    require(row["source_closure"] == EXPECTED_SOURCE_CLOSURE, "audit source closure")

    members = row["publication_members"]
    require(isinstance(members, dict) and set(members) == set(EXPECTED_PUBLICATION), "publication member set")
    for name, (size, member_sha) in EXPECTED_PUBLICATION.items():
        require(members[name] == {"bytes": size, "sha256": member_sha}, f"publication member {name}")

    replay = row["scientific_replay"]
    require(isinstance(replay, dict), "scientific replay")
    selected = replay.get("selected_q016_cpu_replay")
    require(isinstance(selected, dict) and selected.get("status") == "PASS_EXACT_SELECTED_CELL_Q016_CPU_REPLAY", "Q0.16 replay")

    physical = row["literal_container_audit"]
    expected_physical_fields = {
        "candidate_sha256", "identity_sha256", "model_packet_sha256",
        "directory_sha256", "identity_directory_sha256",
        "decision_commitment_sha256", "reconstruction_sha256", "rate", "F",
        "physical_pass", "cold_pass", "bandwidth", "independent_container_parser",
        "independent_byte_ledger_entries", "independent_selected_stream_causal_replay",
        "identity_rate", "identity_semantic_decode",
    }
    require(isinstance(physical, dict) and set(physical) == expected_physical_fields, "literal audit schema")
    require(physical["candidate_sha256"] == CANDIDATE_SHA256, "candidate binding")
    require(physical["reconstruction_sha256"] == RECONSTRUCTION_SHA256, "reconstruction binding")
    require(physical["independent_container_parser"] is True and physical["independent_selected_stream_causal_replay"] is True, "independent causal replay")
    candidate_rate = _fraction(physical["rate"], "candidate rate")
    require(candidate_rate == Fraction(8 * CANDIDATE_BYTES, SOURCE_WEIGHTS), "candidate rate from bytes")
    f_value = physical["F"]
    require(type(f_value) in (int, float) and math.isfinite(f_value) and f_value > 0.0, "audited F")
    inherited_mse = float(f_value) / math.pow(2.0, 2.0 * float(candidate_rate))
    decision = _digest(physical["decision_commitment_sha256"], "decision commitment")
    return AuditAuthority(
        receipt_sha256=observed,
        publication_members={name: dict(value) for name, value in members.items()},
        decision_commitment_sha256=decision,
        reconstruction_sha256=RECONSTRUCTION_SHA256,
        candidate_rate=candidate_rate,
        inherited_relative_mse=inherited_mse,
        primary_result_status=row["primary_result_status"],
    )


def read_publication_member(authority: AuditAuthority, publication: Path, name: str) -> bytes:
    """First publication opener; impossible to call without validated authority."""

    require(isinstance(authority, AuditAuthority), "audit authority capability")
    require(name in authority.publication_members, "authorized publication member")
    root = publication.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "real publication directory")
    require(set(entry.name for entry in os.scandir(root)) == set(EXPECTED_PUBLICATION), "publication exact set")
    path = root / name
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        expected = authority.publication_members[name]
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size == expected["bytes"], "publication regular identity")
        output = bytearray()
        while len(output) < before.st_size:
            chunk = os.read(descriptor, min(1 << 20, before.st_size - len(output)))
            require(bool(chunk), "publication short read")
            output.extend(chunk)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink), "publication identity drift")
        payload = bytes(output)
        require(sha256(payload) == expected["sha256"], "publication member digest")
        return payload
    finally:
        os.close(descriptor)
