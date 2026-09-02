#!/usr/bin/env python3
"""Strict bounded records for the source-only UWFA-SC v3 producer.

Every size capable of driving a shift, loop, allocation, product, or device
copy is bounded here before use. Ownership has exactly one ABI everywhere:
a 32-byte little-endian bit set. There are no integer-mask aliases.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from typing import Any, Iterable, Mapping, Sequence


MIN_EXPERTS = 1
MAX_EXPERTS = 256
OWNER_SET_BYTES = 32
MAX_STREAMS = 65_536
MAX_REGIONS = 65_536
MAX_WEIGHTS = 1 << 50
MAX_SYMBOLS = 1 << 54
MAX_FILE_BYTES = 1 << 40
MAX_DIMENSION = 1 << 24
MAX_LOGICAL_BITS = 1 << 56
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:@+-]{1,128}$")


def exact_int(value: Any, label: str, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact JSON integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} outside [{minimum},{maximum}]")
    return value


def bounded_count(value: Any, label: str, maximum: int) -> int:
    return exact_int(value, label, 1, maximum)


def checked_product(left: Any, right: Any, label: str, maximum: int = MAX_WEIGHTS) -> int:
    a = exact_int(left, f"{label} left", 1, MAX_DIMENSION)
    b = exact_int(right, f"{label} right", 1, MAX_DIMENSION)
    if a > maximum // b:
        raise ValueError(f"{label} product exceeds bound")
    return a * b


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


def validate_experts(value: Any) -> int:
    return exact_int(value, "experts", MIN_EXPERTS, MAX_EXPERTS)


def owner_set_from_ordinals(experts: Any, owners: Sequence[Any]) -> bytes:
    """Return the only legal owner representation: fixed 32-byte LE bits."""
    expert_count = validate_experts(experts)
    if not isinstance(owners, (tuple, list)) or not 1 <= len(owners) <= expert_count:
        raise ValueError("owner ordinal list geometry")
    canonical: list[int] = []
    for raw in owners:
        canonical.append(exact_int(raw, "owner ordinal", 0, expert_count - 1))
    if canonical != sorted(set(canonical)):
        raise ValueError("owners must be unique and sorted")
    result = bytearray(OWNER_SET_BYTES)
    for ordinal in canonical:
        result[ordinal >> 3] |= 1 << (ordinal & 7)
    return bytes(result)


def owner_ordinals(owner_set: Any, experts: Any) -> tuple[int, ...]:
    expert_count = validate_experts(experts)
    if not isinstance(owner_set, bytes) or len(owner_set) != OWNER_SET_BYTES:
        raise ValueError("owner set must be exactly 32 bytes")
    used_bytes = (expert_count + 7) // 8
    if any(owner_set[used_bytes:]):
        raise ValueError("noncanonical owner bits above expert universe")
    if expert_count & 7:
        legal = (1 << (expert_count & 7)) - 1
        if owner_set[used_bytes - 1] & ~legal:
            raise ValueError("noncanonical terminal owner bits")
    if not any(owner_set[:used_bytes]):
        raise ValueError("empty owner set")
    return tuple(index for index in range(expert_count) if owner_set[index >> 3] & (1 << (index & 7)))


def owner_set_hex(value: Any, experts: Any) -> str:
    validate_experts(experts)
    if not isinstance(value, str) or len(value) != 2 * OWNER_SET_BYTES:
        raise ValueError("owner_set_hex must encode exactly 32 bytes")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("owner_set_hex is not hexadecimal") from exc
    owner_ordinals(raw, experts)
    return raw.hex()


def length_prefixed_digest(parts: Sequence[str | bytes | int], *, domain: bytes) -> str:
    digest = hashlib.sha256(domain)
    for value in parts:
        if isinstance(value, str):
            payload, tag = value.encode("utf-8", "strict"), 1
        elif isinstance(value, bytes):
            payload, tag = value, 2
        elif type(value) is int:
            payload, tag = str(value).encode("ascii"), 3
        else:
            raise TypeError("unsupported digest tuple member")
        digest.update(bytes((tag,)))
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def validate_score_receipt(record: Any, *, artifact_sha256: str, artifact_bytes: int, weights: int, reconstruction_sha256: str) -> dict[str, Any]:
    row = strict_fields(
        record,
        required=("schema", "status", "artifact_sha256", "artifact_bytes", "weights", "relative_mse", "sse_fp64", "source_energy_fp64", "normalization", "reconstruction_f64_sha256", "original_source_panel_sha256", "independent_decoder_source_sha256", "score_receipt_sha256"),
        label="baseline score receipt",
    )
    if row["schema"] != "uwfa-bound-baseline-score-v3" or row["status"] != "PASS_INDEPENDENT_BASELINE_SCORE":
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
    expected = sse / energy
    if abs(expected - mse) > 4.0 * math.ulp(expected):
        raise ValueError("score relative MSE does not equal SSE/energy")
    sha256_hex(row["original_source_panel_sha256"], "source panel")
    sha256_hex(row["independent_decoder_source_sha256"], "decoder source")
    claimed = sha256_hex(row["score_receipt_sha256"], "score receipt seal")
    clean = dict(row)
    clean.pop("score_receipt_sha256")
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    if hashlib.sha256(encoded).hexdigest() != claimed:
        raise ValueError("score receipt internal integrity")
    return dict(row)


def panel_geometry(panel: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(panel, dict):
        raise ValueError("panel must be an object")
    expert_count = validate_experts(panel.get("experts"))
    total_weights = exact_int(panel.get("weights"), "panel weights", 1, MAX_WEIGHTS)
    streams = panel.get("streams")
    if not isinstance(streams, list) or not 1 <= len(streams) <= MAX_STREAMS:
        raise ValueError("panel stream count outside bound")
    artifact = panel.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("panel artifact")
    raw_identities = panel.get("semantic_identities")
    if not isinstance(raw_identities, (tuple, list)) or len(raw_identities) != expert_count:
        raise ValueError("panel semantic identity geometry")
    semantic_identities = []
    for raw_identity in raw_identities:
        if not isinstance(raw_identity, (tuple, list)) or len(raw_identity) != 2:
            raise ValueError("panel semantic identity row")
        semantic_identities.append([
            exact_int(raw_identity[0], "semantic layer identity", 0, (1 << 31) - 1),
            exact_int(raw_identity[1], "semantic expert identity", 0, (1 << 31) - 1),
        ])
    if len({tuple(row) for row in semantic_identities}) != expert_count:
        raise ValueError("panel semantic identities must be unique")
    raw_shapes = panel.get("expert_shapes")
    if not isinstance(raw_shapes, (tuple, list)) or len(raw_shapes) != expert_count:
        raise ValueError("panel expert shape geometry")
    expert_shapes = []
    shape_weights = 0
    for ordinal, raw_shape in enumerate(raw_shapes):
        if not isinstance(raw_shape, dict) or set(raw_shape) != {"expert", "hidden", "intermediate"}:
            raise ValueError("panel expert shape row")
        if exact_int(raw_shape["expert"], "shape expert", 0, expert_count - 1) != ordinal:
            raise ValueError("panel expert shapes must use canonical order")
        hidden = exact_int(raw_shape["hidden"], "shape hidden", 1, MAX_DIMENSION)
        intermediate = exact_int(raw_shape["intermediate"], "shape intermediate", 1, MAX_DIMENSION)
        matrix_weights = checked_product(hidden, intermediate, "shape matrix weights")
        if matrix_weights > MAX_WEIGHTS // 3:
            raise ValueError("shape expert weights overflow")
        contribution = 3 * matrix_weights
        if shape_weights > MAX_WEIGHTS - contribution:
            raise ValueError("shape-derived weight sum overflow")
        shape_weights += contribution
        expert_shapes.append({"expert": ordinal, "hidden": hidden, "intermediate": intermediate})
    if shape_weights != total_weights:
        raise ValueError("shape-derived panel weights mismatch")
    rows: list[dict[str, Any]] = []
    seen_owners = bytearray(OWNER_SET_BYTES)
    weight_sum = 0
    for index, raw_row in enumerate(streams):
        if not isinstance(raw_row, dict):
            raise ValueError("panel stream must be an object")
        required = {"stream_ordinal", "owner_set_hex", "weight_charge", "shape_rows", "shape_cols", "role", "symbols", "logn", "profile_q", "baseline_payload_bytes", "baseline_logical_bits"}
        if not required.issubset(raw_row):
            raise ValueError(f"panel stream missing geometry fields: {sorted(required - set(raw_row))}")
        row = raw_row
        ordinal = exact_int(row["stream_ordinal"], "stream ordinal", 0, MAX_STREAMS - 1)
        if ordinal != index:
            raise ValueError("stream ordinals must be one canonical range")
        owner_hex = owner_set_hex(row["owner_set_hex"], expert_count)
        owner_raw = bytes.fromhex(owner_hex)
        for offset in range(OWNER_SET_BYTES):
            seen_owners[offset] |= owner_raw[offset]
        shape_rows = exact_int(row["shape_rows"], "shape rows", 1, MAX_DIMENSION)
        shape_cols = exact_int(row["shape_cols"], "shape cols", 1, MAX_DIMENSION)
        product = checked_product(shape_rows, shape_cols, "stream shape")
        charge = exact_int(row["weight_charge"], "weight charge", 1, MAX_WEIGHTS)
        if charge != product:
            raise ValueError("stream weight charge must equal literal shape product")
        if weight_sum > MAX_WEIGHTS - charge:
            raise ValueError("panel weight sum overflow")
        weight_sum += charge
        raw_contributions = row.get("owner_contributions")
        if not isinstance(raw_contributions, (tuple, list)) or not 1 <= len(raw_contributions) <= 3 * expert_count:
            raise ValueError("panel owner contribution count")
        contributions = []
        previous_key: tuple[int, int, int] | None = None
        contribution_sum = 0
        for contribution in raw_contributions:
            if not isinstance(contribution, dict) or set(contribution) != {"expert", "role", "source_offset", "weight_count"}:
                raise ValueError("panel owner contribution schema")
            owner = exact_int(contribution["expert"], "contribution expert", 0, expert_count - 1)
            contribution_role = identifier(contribution["role"], "contribution role")
            if contribution_role not in {"gate", "up", "down"}:
                raise ValueError("contribution role outside SwiGLU")
            offset = exact_int(contribution["source_offset"], "contribution offset", 0, MAX_WEIGHTS - 1)
            count = exact_int(contribution["weight_count"], "contribution count", 1, MAX_WEIGHTS)
            if offset > MAX_WEIGHTS - count:
                raise ValueError("contribution interval overflow")
            key = (owner, ("gate", "up", "down").index(contribution_role), offset)
            if previous_key is not None and key <= previous_key:
                raise ValueError("noncanonical contribution order")
            previous_key = key
            contribution_sum += count
            if contribution_sum > MAX_WEIGHTS:
                raise ValueError("contribution sum overflow")
            contributions.append({"expert": owner, "role": contribution_role, "source_offset": offset, "weight_count": count})
        if contribution_sum != charge or tuple(sorted({item["expert"] for item in contributions})) != owner_ordinals(owner_raw, expert_count):
            raise ValueError("contribution/owner/weight conservation")
        rows.append({
            "ordinal": ordinal,
            "owner_set_hex": owner_hex,
            "weight_charge": charge,
            "shape_rows": shape_rows,
            "shape_cols": shape_cols,
            "role": identifier(row["role"], "stream role"),
            "owner_contributions": contributions,
            "symbols": exact_int(row["symbols"], "symbols", 1, MAX_SYMBOLS),
            "logn": exact_int(row["logn"], "logn", 1, 62),
            "profile_q": exact_int(row["profile_q"], "profile q", 0, 65535),
            "baseline_payload_bytes": exact_int(row["baseline_payload_bytes"], "baseline payload bytes", 1, MAX_FILE_BYTES),
            "baseline_logical_bits": exact_int(row["baseline_logical_bits"], "baseline logical bits", 1, MAX_LOGICAL_BITS),
        })
    if weight_sum != total_weights:
        raise ValueError("source weight conservation failure")
    expected_universe = owner_set_from_ordinals(expert_count, list(range(expert_count)))
    if bytes(seen_owners) != expected_universe:
        raise ValueError("declared expert universe has empty/missing experts")
    return {
        "weights": total_weights,
        "experts": expert_count,
        "artifact_bytes": exact_int(artifact.get("raw_bytes"), "artifact bytes", 1, MAX_FILE_BYTES),
        "immutable_state_bytes": len(panel.get("immutable_state", b"")),
        "semantic_identities": semantic_identities,
        "expert_shapes": expert_shapes,
        "streams": rows,
    }


def geometry_sha256(common: Any, panel: Mapping[str, Any]) -> str:
    return hashlib.sha256(common.canonical_json(panel_geometry(panel))).hexdigest()


def validate_control_binding(common: Any, record: Any, *, seed: int, source_artifact_sha256: str, source_geometry_sha256: str, source_pipeline_sha256: str, control_artifact_sha256: str, control_geometry_sha256: str) -> dict[str, Any]:
    row = strict_fields(record, required=("schema", "seed", "source_artifact_sha256", "source_geometry_sha256", "pipeline_sha256", "generator_source_sha256", "moment_match_receipt_sha256", "control_artifact_sha256", "control_geometry_sha256", "binding_sha256"), label="Gaussian control binding")
    if row["schema"] != "uwfa-matched-gaussian-control-binding-v3":
        raise ValueError("control binding schema")
    if exact_int(row["seed"], "control seed", 0, (1 << 63) - 1) != seed:
        raise ValueError("control seed/order")
    expected = {"source_artifact_sha256": source_artifact_sha256, "source_geometry_sha256": source_geometry_sha256, "pipeline_sha256": source_pipeline_sha256, "control_artifact_sha256": control_artifact_sha256, "control_geometry_sha256": control_geometry_sha256}
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
