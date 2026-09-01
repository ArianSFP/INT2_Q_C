#!/usr/bin/env python3
"""Strict, bounded, source-independent records for UWFA-SC v2."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from typing import Any, Iterable, Mapping, Sequence


MAX_EXPERTS = 4096
MAX_BLOCKS = 65536
MAX_WEIGHTS = 1 << 50
MAX_SYMBOLS = 1 << 54
MAX_FILE_BYTES = 1 << 50
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:@+-]{1,128}$")


def exact_int(value: Any, label: str, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if type(value) is not int:  # bool is intentionally rejected
        raise ValueError(f"{label} must be an exact JSON integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} outside [{minimum},{maximum}]")
    return value


def finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} invalid finite value")
    return result


def sha256_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be SHA-256 hex")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} not hexadecimal") from exc
    if len(raw) != 32:
        raise ValueError(f"{label} digest geometry")
    return value.lower()


def identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must use the frozen ASCII identifier alphabet")
    return value


def strict_fields(record: Any, *, required: Iterable[str], optional: Iterable[str] = (), label: str) -> Mapping[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an object")
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set - set(record)
    unknown = set(record) - allowed
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} unknown fields: {sorted(unknown)}")
    return record


def length_prefixed_digest(parts: Sequence[str | bytes | int], *, domain: bytes) -> str:
    digest = hashlib.sha256(domain)
    for value in parts:
        if isinstance(value, str):
            payload = value.encode("utf-8", "strict")
            tag = 1
        elif isinstance(value, bytes):
            payload = value
            tag = 2
        elif type(value) is int:
            payload = str(value).encode("ascii")
            tag = 3
        else:
            raise TypeError("unsupported digest tuple member")
        digest.update(bytes((tag,)))
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def validate_score_receipt(
    record: Any,
    *,
    artifact_sha256: str,
    artifact_bytes: int,
    weights: int,
    reconstruction_sha256: str,
) -> dict[str, Any]:
    row = strict_fields(
        record,
        required=(
            "schema",
            "status",
            "artifact_sha256",
            "artifact_bytes",
            "weights",
            "relative_mse",
            "sse_fp64",
            "source_energy_fp64",
            "normalization",
            "reconstruction_f64_sha256",
            "original_source_panel_sha256",
            "independent_decoder_source_sha256",
            "score_receipt_sha256",
        ),
        label="baseline score receipt",
    )
    if row["schema"] != "uwfa-bound-baseline-score-v2" or row["status"] != "PASS_INDEPENDENT_BASELINE_SCORE":
        raise ValueError("baseline score receipt schema/status")
    if sha256_hex(row["artifact_sha256"], "score artifact") != artifact_sha256:
        raise ValueError("score/artifact digest mismatch")
    if exact_int(row["artifact_bytes"], "score artifact bytes", 1, MAX_FILE_BYTES) != artifact_bytes:
        raise ValueError("score/artifact byte mismatch")
    if exact_int(row["weights"], "score weights", 1, MAX_WEIGHTS) != weights:
        raise ValueError("score/shape weight mismatch")
    if sha256_hex(row["reconstruction_f64_sha256"], "score reconstruction") != reconstruction_sha256:
        raise ValueError("score/reconstruction digest mismatch")
    sse = finite_float(row["sse_fp64"], "score SSE", positive=True)
    energy = finite_float(row["source_energy_fp64"], "score energy", positive=True)
    mse = finite_float(row["relative_mse"], "score relative MSE", positive=True)
    if row["normalization"] != "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM":
        raise ValueError("score normalization")
    # Bind the printed MSE to the two authoritative sums.  Four ulps permits
    # JSON round-trip while preventing an independently chosen D.
    expected = sse / energy
    if abs(expected - mse) > 4.0 * math.ulp(expected):
        raise ValueError("score relative MSE does not equal SSE/energy")
    sha256_hex(row["original_source_panel_sha256"], "source panel")
    sha256_hex(row["independent_decoder_source_sha256"], "decoder source")
    claimed = sha256_hex(row["score_receipt_sha256"], "score receipt seal")
    clean = dict(row)
    clean.pop("score_receipt_sha256")
    import json
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    if hashlib.sha256(encoded).hexdigest() != claimed:
        raise ValueError("score receipt internal integrity")
    return dict(row)


def panel_geometry(panel: Mapping[str, Any]) -> dict[str, Any]:
    streams = panel["streams"]
    return {
        "weights": exact_int(panel["weights"], "panel weights", 1, MAX_WEIGHTS),
        "experts": exact_int(panel["experts"], "panel experts", 1, MAX_EXPERTS),
        "artifact_bytes": exact_int(panel["artifact"]["raw_bytes"], "artifact bytes", 1, MAX_FILE_BYTES),
        "immutable_state_bytes": len(panel["immutable_state"]),
        "streams": [
            {
                "ordinal": exact_int(row["stream_ordinal"], "stream ordinal", 0, MAX_BLOCKS - 1),
                "owner_mask": exact_int(row["owner_mask"], "owner mask", 1, (1 << min(panel["experts"], 62)) - 1),
                "weight_charge": exact_int(row["weight_charge"], "weight charge", 1, MAX_WEIGHTS),
                "symbols": exact_int(row["symbols"], "symbols", 1, MAX_SYMBOLS),
                "logn": exact_int(row["logn"], "logn", 1, 62),
                "profile_q": exact_int(row["profile_q"], "profile q", 0, 255),
                "baseline_payload_bytes": exact_int(row["baseline_payload_bytes"], "baseline payload bytes", 1, MAX_FILE_BYTES),
                "baseline_logical_bits": exact_int(row["baseline_logical_bits"], "baseline logical bits", 1, MAX_SYMBOLS),
            }
            for row in streams
        ],
    }


def geometry_sha256(common: Any, panel: Mapping[str, Any]) -> str:
    return hashlib.sha256(common.canonical_json(panel_geometry(panel))).hexdigest()


def validate_control_binding(
    common: Any,
    record: Any,
    *,
    seed: int,
    source_artifact_sha256: str,
    source_geometry_sha256: str,
    source_pipeline_sha256: str,
    control_artifact_sha256: str,
    control_geometry_sha256: str,
) -> dict[str, Any]:
    row = strict_fields(
        record,
        required=(
            "schema", "seed", "source_artifact_sha256", "source_geometry_sha256",
            "pipeline_sha256", "generator_source_sha256", "moment_match_receipt_sha256",
            "control_artifact_sha256", "control_geometry_sha256", "binding_sha256",
        ),
        label="Gaussian control binding",
    )
    if row["schema"] != "uwfa-matched-gaussian-control-binding-v2":
        raise ValueError("control binding schema")
    if exact_int(row["seed"], "control seed", 0, (1 << 63) - 1) != seed:
        raise ValueError("control seed/order")
    expected = {
        "source_artifact_sha256": source_artifact_sha256,
        "source_geometry_sha256": source_geometry_sha256,
        "pipeline_sha256": source_pipeline_sha256,
        "control_artifact_sha256": control_artifact_sha256,
        "control_geometry_sha256": control_geometry_sha256,
    }
    for key, value in expected.items():
        if sha256_hex(row[key], key) != value:
            raise ValueError(f"control binding mismatch: {key}")
    sha256_hex(row["generator_source_sha256"], "control generator")
    sha256_hex(row["moment_match_receipt_sha256"], "moment match")
    claimed = sha256_hex(row["binding_sha256"], "control seal")
    clean = dict(row)
    clean.pop("binding_sha256")
    if hashlib.sha256(common.canonical_json(clean)).hexdigest() != claimed:
        raise ValueError("control binding integrity")
    return dict(row)
