#!/usr/bin/env python3
"""Source-only contracts for UWFA-SC v9 deferred controls.

This module is deliberately standard-library-only.  It contains the frozen
control ordering, exact work ledger, deterministic reference semantics for the
two new selected-symbol shuffles, and the fail-closed runtime disposition.  It
does not import NumPy/CuPy, inspect a result, or open a model/control payload.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA = "uwfa-sc-v9-deferred-controls-source-v0"
PRIMARY_SURVIVOR_STATUS = (
    "PRIMARY_SOURCE_SURVIVOR_NONPROMOTING_DEFERRED_STAGES_REQUIRED"
)
PRIMARY_UPDATES = 38_621_316_130
PRIMARY_SYMBOLS = 126_627_266
PRIMARY_STREAMS = 15
CANDIDATE_CELLS = 150
CONTROL_SEEDS = (
    10_619_863,
    10_619_881,
    10_619_909,
    10_619_927,
    10_619_953,
    10_619_971,
    10_619_999,
    10_620_017,
)

# The retained within-context and three chunk names call byte-sealed v8
# implementations.  The two role/profile diagnostics have the pure reference
# semantics implemented below.  None may select a smaller topology bank or
# reuse the source winner.
SHUFFLE_SEQUENCE = (
    "v8_within_public_context_phase_preserving",
    "v9_within_role_profile_public_context_phase_preserving",
    "v9_within_role_profile_level_prior_phase_destroying",
    "v8_multiscale_chunk_32_phase_preserving",
    "v8_multiscale_chunk_128_phase_preserving",
    "v8_multiscale_chunk_512_phase_preserving",
)

ROW_COLUMN_DISPOSITION = (
    "INAPPLICABLE_AT_SELECTED_SC_SYMBOL_LAYER_REQUIRES_REENCODE_FROM_SOURCE"
)


class ControlContractError(RuntimeError):
    pass


class DeferredRuntimeBlock(ControlContractError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlContractError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prior_bin(base_frequency: int) -> int:
    require(type(base_frequency) is int, "base frequency type")
    require(1 <= base_frequency <= 65_535, "base frequency range")
    return min(15, base_frequency * 16 // 65_536)


def public_context(level: int, base_frequency: int, position: int) -> int:
    require(type(level) is int and 0 <= level < 6, "level")
    require(type(position) is int and position >= 0, "position")
    return ((level * 16 + prior_bin(base_frequency)) * 4) + (position & 3)


def _strict_streams(panel: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    require(isinstance(panel, Mapping), "panel mapping")
    streams = panel.get("streams")
    require(isinstance(streams, list) and streams, "nonempty stream list")
    ordinals: list[int] = []
    for row in streams:
        require(isinstance(row, Mapping), "stream mapping")
        required = {
            "stream_ordinal",
            "role",
            "profile_q",
            "symbols",
            "bits",
            "levels",
            "base",
            "bits_bytes",
            "levels_bytes",
            "base_bytes",
        }
        require(required.issubset(row), "stream fields")
        ordinal = row["stream_ordinal"]
        require(type(ordinal) is int and ordinal >= 0, "stream ordinal")
        ordinals.append(ordinal)
        role = row["role"]
        require(
            isinstance(role, str) and role in {"gate", "up", "down", "mixed"},
            "stream role",
        )
        profile = row["profile_q"]
        require(type(profile) is int and 0 <= profile <= 255, "profile q")
        symbols = row["symbols"]
        require(type(symbols) is int and symbols > 0, "symbols")
        bits = row["bits"]
        levels = row["levels"]
        base = row["base"]
        require(
            isinstance(bits, list)
            and isinstance(levels, list)
            and isinstance(base, list),
            "selected-symbol lists",
        )
        require(len(bits) == len(levels) == len(base) == symbols, "stream geometry")
        require(
            bytes(bits) == bytes(row["bits_bytes"])
            and bytes(levels) == bytes(row["levels_bytes"]),
            "stream byte/list agreement",
        )
        require(len(bytes(row["base_bytes"])) == 2 * symbols, "base byte geometry")
        for bit in bits:
            require(type(bit) is int and bit in (0, 1), "selected bit")
        for level in levels:
            require(type(level) is int and 0 <= level < 6, "selected level")
        for value in base:
            prior_bin(value)
    require(ordinals == sorted(ordinals) and len(set(ordinals)) == len(ordinals), "canonical stream order")
    return streams


def _affine_parameters(domain: bytes, key: tuple[Any, ...], size: int) -> tuple[int, int]:
    require(size > 0, "bucket size")
    if size == 1:
        return 0, 0
    digest = hashlib.sha256(domain + canonical_json(list(key))).digest()
    raw_a = int.from_bytes(digest[:8], "little")
    b = int.from_bytes(digest[8:16], "little") % size
    a = 1 + raw_a % (size - 1)
    while math.gcd(a, size) != 1:
        a += 1
        if a >= size:
            a = 1
    return a, b


def _reference_bucket_permutation(
    panel: Mapping[str, Any],
    *,
    preserve_phase: bool,
) -> dict[str, Any]:
    """Reference semantics; production must implement the same gather in CuPy.

    Buckets span streams only when role and STRATA profile agree.  They also
    preserve polar level and prior-frequency bin.  The phase-preserving form
    includes position modulo four; the phase-destroying form deliberately does
    not.  The permutation is a bijective affine gather over each canonical
    bucket, so bit counts are conserved exactly and no RNG state is implicit.
    """

    streams = _strict_streams(panel)
    buckets: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
    for stream_index, row in enumerate(streams):
        role = str(row["role"])
        profile = int(row["profile_q"])
        for position, (level, base) in enumerate(
            zip(row["levels"], row["base"], strict=True)
        ):
            key: tuple[Any, ...] = (role, profile, int(level), prior_bin(int(base)))
            if preserve_phase:
                key += (position & 3,)
            buckets.setdefault(key, []).append((stream_index, position))

    output_streams = [dict(row) for row in streams]
    output_bits = [list(row["bits"]) for row in streams]
    domain = (
        b"UWFA-SC-V9-ROLE-PROFILE-PHASE-PRESERVING\x00"
        if preserve_phase
        else b"UWFA-SC-V9-ROLE-PROFILE-PHASE-DESTROYING\x00"
    )
    for key in sorted(buckets, key=canonical_json):
        positions = buckets[key]
        values = [int(streams[s]["bits"][p]) for s, p in positions]
        a, b = _affine_parameters(domain, key, len(positions))
        for destination, (stream_index, position) in enumerate(positions):
            source = 0 if len(positions) == 1 else (a * destination + b) % len(positions)
            output_bits[stream_index][position] = values[source]

    for stream_index, row in enumerate(output_streams):
        row["bits"] = output_bits[stream_index]
        row["bits_bytes"] = bytes(output_bits[stream_index])
        row["diagnostic_transform"] = (
            "within_role_profile_public_context_phase_preserving"
            if preserve_phase
            else "within_role_profile_level_prior_phase_destroying"
        )
    result = dict(panel)
    result["streams"] = output_streams
    return result


def role_profile_phase_preserving_reference(panel: Mapping[str, Any]) -> dict[str, Any]:
    return _reference_bucket_permutation(panel, preserve_phase=True)


def role_profile_phase_destroying_reference(panel: Mapping[str, Any]) -> dict[str, Any]:
    return _reference_bucket_permutation(panel, preserve_phase=False)


def bucket_bit_histogram(
    panel: Mapping[str, Any], *, preserve_phase: bool
) -> dict[tuple[Any, ...], tuple[int, int]]:
    """Return `(zeroes, ones)` by the exact diagnostic buckets."""

    streams = _strict_streams(panel)
    counts: dict[tuple[Any, ...], list[int]] = {}
    for row in streams:
        for position, (bit, level, base) in enumerate(
            zip(row["bits"], row["levels"], row["base"], strict=True)
        ):
            key: tuple[Any, ...] = (
                str(row["role"]),
                int(row["profile_q"]),
                int(level),
                prior_bin(int(base)),
            )
            if preserve_phase:
                key += (position & 3,)
            counts.setdefault(key, [0, 0])[int(bit)] += 1
    return {key: (value[0], value[1]) for key, value in counts.items()}


@dataclass(frozen=True)
class WorkLedger:
    primary_updates_per_pipeline: int = PRIMARY_UPDATES
    matched_controls_maximum: int = len(CONTROL_SEEDS)
    shuffles_maximum: int = len(SHUFFLE_SEQUENCE)

    def as_dict(self) -> dict[str, Any]:
        matched = self.primary_updates_per_pipeline * self.matched_controls_maximum
        shuffles = self.primary_updates_per_pipeline * self.shuffles_maximum
        return {
            "schema": "uwfa-sc-v9-deferred-work-ledger-v0",
            "candidate_cells_per_pipeline": CANDIDATE_CELLS,
            "selected_symbols_per_pipeline": PRIMARY_SYMBOLS,
            "streams_per_pipeline": PRIMARY_STREAMS,
            "updates_per_complete_pipeline": self.primary_updates_per_pipeline,
            "matched_control_pipelines_for_positive_specificity": self.matched_controls_maximum,
            "matched_control_updates_for_positive_specificity": matched,
            "structure_shuffle_pipelines": self.shuffles_maximum,
            "structure_shuffle_updates": shuffles,
            "maximum_deferred_updates": matched + shuffles,
            "minimum_decisive_null_kill_updates": self.primary_updates_per_pipeline,
            "early_stop": (
                "authenticate all eight bundles before fit; fit controls in seed order; "
                "stop after the first null saving greater than or equal to source; "
                "all eight are mandatory for a specificity survivor"
            ),
        }


def runtime_block_record() -> dict[str, Any]:
    """The exact v0 disposition, computed without inspecting any path."""

    record = {
        "schema": "uwfa-sc-v9-deferred-controls-block-v0",
        "status": "BLOCK_MISSING_DECODER_CLOSED_MATCHED_CONTROL_PRODUCER_AND_AUDIT_PINS",
        "positive_claim_authority": False,
        "payload_access_authority": False,
        "primary_result_opened": False,
        "qwen_artifact_opened": False,
        "original_bf16_source_opened": False,
        "gaussian_control_opened": False,
        "cuda_launched": False,
        "missing_immutable_pins": [
            "independent_v9_primary_result_auditor_manifest_sha256",
            "independent_v9_primary_result_audit_receipt_sha256",
            "decoder_closed_matched_gaussian_producer_manifest_sha256",
            "eight_control_bundle_root_sha256",
        ],
        "bounded_reason": (
            "sealed v8 exposes decoder/extractor and a consumer for already-authenticated "
            "Gaussian artifacts, but no encoder that maps moment-matched BF16 SwiGLU weights "
            "through the identical STRATA transform/quantizer/container pipeline"
        ),
        "forbidden_shortcuts": [
            "shuffle already-decoded selected SC bits and call them Gaussian",
            "sample labels from fitted marginal frequencies",
            "reuse source baseline payload sizes without re-encoding the Gaussian source",
            "reuse the Qwen-selected WFA cell instead of all-150 selection per control",
            "accept an unpinned self-authored audit receipt",
        ],
        "work_ledger": WorkLedger().as_dict(),
        "row_column_control": ROW_COLUMN_DISPOSITION,
    }
    record["block_sha256"] = sha256(canonical_json(record))
    return record


def payload_entrypoint(*_args: Any, **_kwargs: Any) -> None:
    """Fail before argument/path inspection; v1 must freeze external pins first."""

    raise DeferredRuntimeBlock(runtime_block_record()["status"])


def validate_primary_summary_without_opening(value: Mapping[str, Any]) -> None:
    """Pure schema helper for a future pinned-successor runner."""

    require(isinstance(value, Mapping), "primary summary mapping")
    require(value.get("status") == PRIMARY_SURVIVOR_STATUS, "primary survivor status")
    require(value.get("positive_claim_authority") is False, "primary remains nonpromoting")
    require(value.get("controls_run") is False, "controls must be deferred")
    require(value.get("shuffles_run") is False, "shuffles must be deferred")
    physical = value.get("physical")
    require(isinstance(physical, Mapping), "primary physical row")
    require(physical.get("passes_rate_interval") is True, "primary rate gate")
    require(physical.get("passes_F_target") is True, "primary F gate")
    require(physical.get("passes_cold_read_below_2x") is True, "primary cold gate")
