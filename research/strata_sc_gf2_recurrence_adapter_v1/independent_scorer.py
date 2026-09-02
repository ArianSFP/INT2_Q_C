#!/usr/bin/env python3
"""Packet-derived rate/F and one-pass ledgers for STRATA recurrence packets."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from fractions import Fraction
from typing import Any, Mapping, Sequence


CATALOG_MAGIC = b"SGFCAT1\0"
CATALOG_BYTES = 4096
RATE_MIN = Fraction(43, 20)
RATE_MAX = Fraction(5, 2)
TARGET_F = 0.8


class ScoreError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def encode_catalog(codec: Any, packets: Sequence[bytes]) -> bytes:
    require(bool(packets), "expert packets")
    decoded = [codec.decode_expert(packet) for packet in packets]
    require([row["expert_ordinal"] for row in decoded] == list(range(len(decoded))), "catalog expert order")
    receipt = {row["audit_receipt_sha256"] for row in decoded}
    candidate = {row["candidate_sha256"] for row in decoded}
    reconstruction = {row["reconstruction_sha256"] for row in decoded}
    require(len(receipt) == len(candidate) == len(reconstruction) == 1, "catalog shared binding")
    record = {
        "schema": "strata-sc-gf2-packet-catalog-v1",
        "audit_receipt_sha256": next(iter(receipt)),
        "candidate_sha256": next(iter(candidate)),
        "reconstruction_sha256": next(iter(reconstruction)),
        "experts": [
            {
                "expert_ordinal": row["expert_ordinal"],
                "source_weights": row["source_weights"],
                "physical_bytes": len(packet),
                "packet_sha256": sha256(packet),
            }
            for packet, row in zip(packets, decoded, strict=True)
        ],
    }
    payload = canonical_json(record)
    require(len(payload) + 16 <= CATALOG_BYTES, "catalog page capacity")
    prefix = CATALOG_MAGIC + struct.pack("<II", len(payload), zlib.crc32(payload) & 0xFFFFFFFF)
    return prefix + payload + bytes(CATALOG_BYTES - len(prefix) - len(payload))


def decode_catalog(payload: bytes) -> dict[str, Any]:
    require(isinstance(payload, bytes) and len(payload) == CATALOG_BYTES, "catalog page")
    require(payload[:8] == CATALOG_MAGIC, "catalog magic")
    length, crc = struct.unpack_from("<II", payload, 8)
    require(0 < length <= CATALOG_BYTES - 16, "catalog length")
    body = payload[16:16 + length]
    require(zlib.crc32(body) & 0xFFFFFFFF == crc and payload[16 + length:] == bytes(CATALOG_BYTES - 16 - length), "catalog CRC/padding")
    record = json.loads(body.decode("ascii"))
    require(canonical_json(record) == body, "catalog canonical JSON")
    require(isinstance(record, dict) and set(record) == {"schema", "audit_receipt_sha256", "candidate_sha256", "reconstruction_sha256", "experts"}, "catalog schema")
    require(record["schema"] == "strata-sc-gf2-packet-catalog-v1", "catalog identity")
    return record


def _decision_commitment_rows(decoded: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: dict[int, tuple[dict[str, Any], str, str]] = {}
    duplicated_decisions = 0
    for expert in decoded:
        for source in expert["streams"]:
            duplicated_decisions += len(source.selected_bits)
            row = {
                "ordinal": source.ordinal,
                "symbols": len(source.selected_bits),
                "logical_bits": source.candidate_logical_bits,
                "decoded_selected_decision_triplet_sha256": source.decoded_triplet_sha256,
                "payload_sha256": source.candidate_payload_sha256,
                "profile_q": source.profile_q,
                "role": source.role,
                "owner_set_hex": source.owner_set_hex,
                "owner_contributions": [dict(value) for value in source.owner_contributions],
            }
            identity = (row, source.selected_sha256, source.levels_sha256)
            if source.ordinal in unique:
                require(unique[source.ordinal] == identity, "duplicated stream identity")
            else:
                unique[source.ordinal] = identity
    ordinals = sorted(unique)
    require(ordinals == list(range(len(ordinals))), "unique stream ordinal coverage")
    return [unique[index][0] for index in ordinals], duplicated_decisions


def score_authenticated_packets(
    codec: Any, gate: Any, replay_gate: Any, *, authority: Any,
    replay_authority: Any,
    catalog: bytes, packets: Sequence[bytes],
) -> dict[str, Any]:
    require(isinstance(authority, gate.AuditAuthority), "v9 audit authority")
    require(isinstance(replay_authority, replay_gate.ReplayAuthority), "independent Q0.16 replay authority")
    parsed_catalog = decode_catalog(catalog)
    decoded = [codec.decode_expert(packet) for packet in packets]
    require(len(decoded) == len(parsed_catalog["experts"]), "catalog packet count")
    require([row["expert_ordinal"] for row in decoded] == list(range(len(decoded))), "expert packet order")
    for packet, expert, catalog_row in zip(packets, decoded, parsed_catalog["experts"], strict=True):
        require(catalog_row == {
            "expert_ordinal": expert["expert_ordinal"],
            "source_weights": expert["source_weights"],
            "physical_bytes": len(packet),
            "packet_sha256": sha256(packet),
        }, "catalog packet binding")
        require(expert["audit_receipt_sha256"] == authority.receipt_sha256, "packet receipt binding")
        require(expert["candidate_sha256"] == gate.CANDIDATE_SHA256, "packet candidate binding")
        require(expert["reconstruction_sha256"] == authority.reconstruction_sha256, "packet reconstruction binding")
    require(parsed_catalog["audit_receipt_sha256"] == authority.receipt_sha256, "catalog receipt binding")
    require(parsed_catalog["candidate_sha256"] == gate.CANDIDATE_SHA256, "catalog candidate binding")
    require(parsed_catalog["reconstruction_sha256"] == authority.reconstruction_sha256, "catalog reconstruction binding")

    weights = sum(row["source_weights"] for row in decoded)
    require(weights == gate.SOURCE_WEIGHTS, "authenticated source weight coverage")
    commitment_rows, duplicated_decisions = _decision_commitment_rows(decoded)
    commitment = sha256(canonical_json(commitment_rows))
    require(commitment == authority.decision_commitment_sha256, "audited selected-decision commitment")
    packet_set_record = [
        {
            "expert_ordinal": row["expert_ordinal"],
            "packet_bytes": len(packet),
            "packet_sha256": sha256(packet),
        }
        for packet, row in zip(packets, decoded, strict=True)
    ]
    packet_set_sha = sha256(canonical_json(packet_set_record))
    require(replay_authority.v9_audit_receipt_sha256 == authority.receipt_sha256, "replay/v9 receipt binding")
    require(replay_authority.catalog_sha256 == sha256(catalog), "replay catalog binding")
    require(replay_authority.packet_set_sha256 == packet_set_sha, "replay packet-set binding")
    require(replay_authority.decision_commitment_sha256 == commitment, "replay decision commitment")
    require(replay_authority.reconstruction_sha256 == authority.reconstruction_sha256, "replay reconstruction binding")
    require(replay_authority.unique_selected_decisions == sum(row["symbols"] for row in commitment_rows), "replay selected-count binding")
    bounds = codec.packet_rate_bounds(decoded, catalog_bytes=len(catalog))
    require(bounds["source_weights"] == weights, "rate-bound weight closure")
    physical_bytes = len(catalog) + sum(len(packet) for packet in packets)
    rate = Fraction(8 * physical_bytes, weights)
    relative_mse = authority.inherited_relative_mse
    f_value = relative_mse * math.pow(2.0, 2.0 * float(rate))
    baseline_rate = Fraction(8 * gate.BASELINE_ARTIFACT_BYTES, weights)
    require(baseline_rate == Fraction(5, 2), "audited current arithmetic rate")
    candidate_rate = Fraction(8 * gate.CANDIDATE_BYTES, weights)
    require(candidate_rate == authority.candidate_rate, "audited candidate rate")

    expert_ledgers = []
    for packet, row in zip(packets, decoded, strict=True):
        current_budget = Fraction(5 * row["source_weights"], 16)
        cold_vs_current = Fraction(len(packet), 1) / current_budget
        attributable = Fraction(len(packet), 1) + Fraction(len(catalog) * row["source_weights"], weights)
        cold_vs_new = Fraction(len(packet), 1) / attributable
        expert_ledgers.append({
            "expert_ordinal": row["expert_ordinal"],
            "source_weights": row["source_weights"],
            "contiguous_read_begin": 0,
            "contiguous_read_bytes": len(packet),
            "read_requests": 1,
            "refetches": 0,
            "touched_page_bytes": len(packet),
            "cold_amplification_vs_current_2p5_weight_budget": float(cold_vs_current),
            "cold_amplification_vs_new_attributable_physical_bytes": float(cold_vs_new),
            "passes_below_2x_vs_current": cold_vs_current < 2,
            "runtime_io_measured": False,
        })
    maximum_cold = max(row["cold_amplification_vs_current_2p5_weight_budget"] for row in expert_ledgers)
    status = (
        "PASS_EXACT_RECURRENCE_PACKET_TARGET"
        if RATE_MIN <= rate <= RATE_MAX and f_value <= TARGET_F and maximum_cold < 2
        else "HARD_KILL_EXACT_RECURRENCE_RATE_F_OR_COLD"
    )
    return {
        "status": status,
        "source_weights": weights,
        "catalog_bytes": len(catalog),
        "expert_packet_bytes": [len(packet) for packet in packets],
        "physical_bytes": physical_bytes,
        "physical_rate_rational": {"numerator": rate.numerator, "denominator": rate.denominator},
        "physical_rate_bpw": float(rate),
        "inherited_identical_reconstruction_relative_mse": relative_mse,
        "F_from_packet_bytes": f_value,
        "audited_current_arithmetic_object_bytes": gate.BASELINE_ARTIFACT_BYTES,
        "audited_current_arithmetic_rate_bpw": float(baseline_rate),
        "audited_v9_candidate_bytes": gate.CANDIDATE_BYTES,
        "audited_v9_candidate_rate_bpw": float(candidate_rate),
        "selected_sc_decisions_unique": sum(row["symbols"] for row in commitment_rows),
        "selected_sc_decisions_with_expert_duplication": duplicated_decisions,
        "selected_decision_count_used_as_rate": False,
        "q016_candidate_replay_bound_by_audit_receipt": True,
        "q016_recurrence_packet_replay_independently_authorized": True,
        "independent_replay_receipt_sha256": replay_authority.receipt_sha256,
        "packet_set_sha256": packet_set_sha,
        "decision_commitment_sha256": commitment,
        "reconstruction_sha256": authority.reconstruction_sha256,
        "packet_rate_bounds": bounds,
        "model_bytes": 0,
        "exception_bytes": 0,
        "exceptions_or_discrepancies_implemented": False,
        "one_pass_expert_ledgers": expert_ledgers,
        "maximum_cold_amplification_vs_current_2p5_weight_budget": maximum_cold,
        "all_values_scorer_derived": True,
        "caller_supplied_mse_rate_ledger_or_controls": False,
        "universal_swiglu_moe_claim_authority": False,
    }
