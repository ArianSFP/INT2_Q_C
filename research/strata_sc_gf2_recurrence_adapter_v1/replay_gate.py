#!/usr/bin/env python3
"""Capability gate for a future independent STRATA Q0.16 replay.

This module does not implement that replay and opens no publication member.
It deliberately keeps the source-only packet codec non-publishable until a
separately frozen decoder has regenerated every level/base-frequency context,
canonically re-encoded every audited UWFCV8 stream, and reproduced the full
FP64 reconstruction digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


SCHEMA = "strata-sc-gf2-independent-q016-replay-v1"
STATUS = "PASS_EXACT_INDEPENDENT_Q016_RECURRENCE_REPLAY"


class ReplayGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayGateError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        label,
    )
    return value


def _strict_json(payload: bytes) -> dict[str, Any]:
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "replay receipt duplicate key")
            output[key] = value
        return output

    value = json.loads(
        payload.decode("ascii"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ReplayGateError(f"replay receipt nonfinite {token}")
        ),
    )
    require(isinstance(value, dict), "replay receipt object")
    return value


@dataclass(frozen=True)
class ReplayAuthority:
    receipt_sha256: str
    v9_audit_receipt_sha256: str
    catalog_sha256: str
    packet_set_sha256: str
    decision_commitment_sha256: str
    reconstruction_sha256: str
    unique_selected_decisions: int


def authorize_replay_receipt(
    payload: bytes, *, expected_receipt_sha256: str,
) -> ReplayAuthority:
    """Authorize only an externally pinned, exact independent replay receipt."""

    require(isinstance(payload, bytes) and payload, "replay receipt bytes")
    observed = sha256(payload)
    require(observed == _digest(expected_receipt_sha256, "external replay receipt digest"), "external replay receipt binding")
    row = _strict_json(payload)
    require(set(row) == {
        "schema", "status", "v9_audit_receipt_sha256", "catalog_sha256",
        "packet_set_sha256", "decision_commitment_sha256",
        "reconstruction_sha256", "unique_selected_decisions",
        "six_level_major_replayed", "all_level_boundaries_exact",
        "all_selected_bits_packet_derived", "all_base_frequencies_regenerated",
        "all_triplet_digests_recomputed", "all_candidate_payloads_q016_reencoded",
        "full_reconstruction_recomputed", "caller_metrics_used",
        "publication_payload_opened_only_after_v9_authority",
    }, "replay receipt exact schema")
    require(row["schema"] == SCHEMA and row["status"] == STATUS, "replay receipt status")
    for name in (
        "six_level_major_replayed", "all_level_boundaries_exact",
        "all_selected_bits_packet_derived", "all_base_frequencies_regenerated",
        "all_triplet_digests_recomputed", "all_candidate_payloads_q016_reencoded",
        "full_reconstruction_recomputed",
        "publication_payload_opened_only_after_v9_authority",
    ):
        require(row[name] is True, f"replay proof {name}")
    require(row["caller_metrics_used"] is False, "replay caller metrics forbidden")
    unique = row["unique_selected_decisions"]
    require(type(unique) is int and unique > 0, "replay selected count")
    return ReplayAuthority(
        receipt_sha256=observed,
        v9_audit_receipt_sha256=_digest(row["v9_audit_receipt_sha256"], "v9 receipt digest"),
        catalog_sha256=_digest(row["catalog_sha256"], "catalog digest"),
        packet_set_sha256=_digest(row["packet_set_sha256"], "packet set digest"),
        decision_commitment_sha256=_digest(row["decision_commitment_sha256"], "decision commitment"),
        reconstruction_sha256=_digest(row["reconstruction_sha256"], "reconstruction digest"),
        unique_selected_decisions=unique,
    )

