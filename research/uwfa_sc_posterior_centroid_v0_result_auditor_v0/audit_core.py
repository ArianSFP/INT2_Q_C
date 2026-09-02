#!/usr/bin/env python3
"""Independent numerical and byte grammar for posterior-centroid v0 results.

This module is import inert and deliberately imports no numerical backend.
The result auditor injects NumPy only after every executable/input closure has
been authenticated.  None of the posterior producer's fitting, scoring,
head/wrapper, ledger, or decision helpers are imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence


SCHEMA = "uwfa-sc-posterior-centroid-v0-result-audit-core-v0"
LEVELS = 6
PAGE_BYTES = 4096
MAX_STATES = 64
RIDGE_EXPONENTS = (-28, -24, -20, -16, -12, -8, -4, 0)
LAW_LOCAL = 0
LAW_STATE = 1
LAW_PERMUTED = 2
LAW_NAMES = {
    LAW_LOCAL: "local-only",
    LAW_STATE: "state-aware",
    LAW_PERMUTED: "state-permuted",
}
LAW_MEMBER_NAMES = {
    LAW_LOCAL: "LOCAL_ONLY",
    LAW_STATE: "STATE_AWARE",
    LAW_PERMUTED: "STATE_PERMUTED",
}

PERMUTATION_DOMAIN = b"UWFA-SC-POSTERIOR-STATE-PERM-V0\x00"
PERMUTATION_SEED = bytes.fromhex(
    "c7a995f5ba0b0a6a3097890ad936f6ef5a9233faf5976abc02ae3eacb4400f7d"
)

HEAD_MAGIC = b"CAGEPC0\x00"
HEAD_VERSION = 1
HEAD_HEADER = struct.Struct("<8sHHHHIIhH32s32sI")
HEAD_HEADER_BYTES = 96
HEAD_FLAGS = (1 << 0) | (1 << 1)

WRAPPER_MAGIC = b"CAGEPST1"
WRAPPER_VERSION = 1
WRAPPER_FOOTER = struct.Struct("<8sHHIQQIIQQIi32s32s32sI28s")
WRAPPER_FOOTER_BYTES = 192
WRAPPER_FLAGS = (1 << 0) | (1 << 1) | (1 << 2)

HANDOFF_KEYS = (
    "literal_container_sha256",
    "source_artifact_sha256",
    "source_score_binding_sha256",
    "source_full_geometry_sha256",
    "source_structural_geometry_sha256",
    "extraction_program_sha256",
    "universal_decoder_sha256",
    "universal_adapter_sha256",
    "pipeline_sha256",
    "source_snapshot_root_sha256",
    "source_preflight_receipt_sha256",
    "full_reconstruction_f64_sha256",
    "semantic_packet_sha256",
    "immutable_context_state_sha256",
    "serialized_model_sha256",
    "directory_sha256",
    "decoded_sc_decision_triplet_commitment_sha256",
)


class AuditError(RuntimeError):
    """Fail-closed audit error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} SHA-256",
    )
    return value


def exact_int(value: Any, label: str, minimum: int | None = None) -> int:
    require(type(value) is int, f"{label} integer")
    if minimum is not None:
        require(value >= minimum, f"{label} lower bound")
    return value


def finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float), f"{label} number")
    result = float(value)
    require(math.isfinite(result), f"{label} finite")
    if positive:
        require(result > 0.0, f"{label} positive")
    return result


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "float": float(value),
    }


def fraction_from_record(value: Any, label: str) -> Fraction:
    require(isinstance(value, Mapping), f"{label} fraction record")
    require(set(value) == {"numerator", "denominator", "float"}, f"{label} fraction fields")
    numerator = exact_int(value["numerator"], f"{label} numerator")
    denominator = exact_int(value["denominator"], f"{label} denominator", 1)
    result = Fraction(numerator, denominator)
    observed = finite_float(value["float"], f"{label} float")
    require(_float_close(observed, float(result), rel=0.0, abs_=2.0 ** -48), f"{label} float binding")
    return result


def _float_close(left: float, right: float, *, rel: float = 2.0 ** -42, abs_: float = 1e-13) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rel, abs_tol=abs_)


def require_float_close(left: Any, right: Any, label: str, *, rel: float = 2.0 ** -42, abs_: float = 1e-13) -> None:
    a = finite_float(left, f"{label} observed")
    b = finite_float(right, f"{label} expected")
    require(_float_close(a, b, rel=rel, abs_=abs_), label)


def require_deep_close(observed: Any, expected: Any, label: str) -> None:
    """Deep comparison with exact structure and narrow FP64 tolerance."""

    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(observed) is type(expected) and observed == expected, label)
        return
    if type(expected) is int:
        require(type(observed) is int and observed == expected, label)
        return
    if type(expected) is float:
        require(type(observed) in (int, float), label)
        require_float_close(observed, expected, label)
        return
    if isinstance(expected, Mapping):
        require(isinstance(observed, Mapping) and set(observed) == set(expected), f"{label} fields")
        for key in sorted(expected):
            require_deep_close(observed[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        require(isinstance(observed, (list, tuple)) and len(observed) == len(expected), f"{label} sequence")
        for ordinal, (left, right) in enumerate(zip(observed, expected, strict=True)):
            require_deep_close(left, right, f"{label}[{ordinal}]")
        return
    require(observed == expected, label)


def posterior_handoff_root(handoff: Mapping[str, Any]) -> str:
    require(handoff.get("schema") == "uwfa-sc-v8-posterior-diagnostic-handoff", "handoff schema")
    require(handoff.get("requires_literal_redecode") is True, "handoff literal redecode")
    require(handoff.get("contains_posterior_or_MMSE_result") is False, "clean predecessor handoff")
    bindings = {key: digest(handoff.get(key), f"handoff {key}") for key in HANDOFF_KEYS}
    rows = handoff.get("stream_decision_triplet_commitments")
    require(isinstance(rows, list) and rows, "handoff stream commitments")
    require(exact_int(handoff.get("stream_count"), "handoff stream count", 1) == len(rows), "handoff stream count binding")
    clean = []
    for ordinal, row in enumerate(rows):
        require(isinstance(row, Mapping), f"handoff row {ordinal}")
        require(exact_int(row.get("ordinal"), f"handoff row {ordinal} ordinal") == ordinal, "handoff row order")
        clean.append({
            "ordinal": ordinal,
            "symbols": exact_int(row["symbols"], "handoff symbols", 1),
            "logical_bits": exact_int(row["logical_bits"], "handoff logical bits", 1),
            "decoded_selected_decision_triplet_sha256": digest(row["decoded_selected_decision_triplet_sha256"], "handoff triplet"),
            "payload_sha256": digest(row["payload_sha256"], "handoff payload"),
            "profile_q": exact_int(row["profile_q"], "handoff profile"),
            "role": str(row["role"]),
            "owner_set_hex": str(row["owner_set_hex"]),
            "owner_contributions": row["owner_contributions"],
        })
    require(sha256(canonical_json(clean)) == bindings["decoded_sc_decision_triplet_commitment_sha256"], "handoff aggregate commitment")
    return sha256(canonical_json({
        "schema": "uwfa-sc-posterior-handoff-root-v0",
        "bindings": bindings,
        "stream_decision_triplet_commitments": clean,
    }))


def trace_predecision_states(common: Any, candidate: Any, bits: Sequence[int], levels: Sequence[int], bases: Sequence[int]) -> list[int]:
    require(len(bits) == len(levels) == len(bases) and len(bits) > 0, "state trace geometry")
    states = int(candidate.states)
    reset = int(candidate.reset_length)
    require(1 <= states <= MAX_STATES and reset > 0, "candidate state/reset")
    state = 0
    result = []
    for position, (raw_bit, raw_level, raw_base) in enumerate(zip(bits, levels, bases, strict=True)):
        within = position % reset
        if within == 0:
            state = 0
        bit, level, base = int(raw_bit), int(raw_level), int(raw_base)
        require(bit in (0, 1) and 0 <= level < LEVELS, "state trace symbol")
        context = common.public_context(level, base, within)
        result.append(state)
        state = int(common.transition(candidate, state, bit, context, within))
        require(0 <= state < states, "state trace bound")
    return result


def occupancy_features(np: Any, levels: Any, pre_states: Any, states: int) -> Any:
    require(type(states) is int and 1 <= states <= MAX_STATES, "occupancy states")
    levels_array = np.asarray(levels)
    states_array = np.asarray(pre_states)
    require(levels_array.ndim == states_array.ndim == 1, "occupancy rank")
    require(levels_array.size == states_array.size and levels_array.size > 0, "occupancy size")
    require(bool(np.all((levels_array >= 0) & (levels_array < LEVELS))), "occupancy levels")
    require(bool(np.all((states_array >= 0) & (states_array < states))), "occupancy state values")
    output = np.zeros((LEVELS, states), dtype=np.float64)
    for level in range(LEVELS):
        selected = states_array[levels_array == level]
        if int(selected.size) > 0:
            counts = np.bincount(selected.astype(np.int64), minlength=states).astype(np.float64)
            output[level] = counts / float(selected.size) - 1.0 / float(states)
            require(abs(float(np.sum(output[level], dtype=np.float64))) <= 64.0 * math.ulp(1.0), "occupancy centering")
    return output


@dataclass(frozen=True)
class Observation:
    ordinal: int
    owners: tuple[int, ...]
    indices: Any
    target_normalized: Any
    occupancy: Any
    coordinate_mapping_sha256: str


def state_permutation(block_ordinal: int, level: int, states: int) -> tuple[int, ...]:
    require(type(block_ordinal) is int and block_ordinal >= 0, "permutation block")
    require(type(level) is int and 0 <= level < LEVELS, "permutation level")
    require(type(states) is int and 1 <= states <= MAX_STATES, "permutation states")
    rows = []
    for state in range(states):
        material = PERMUTATION_DOMAIN + PERMUTATION_SEED + struct.pack("<IHH", block_ordinal, level, state)
        rows.append((hashlib.sha256(material).digest(), state))
    rows.sort(key=lambda row: (row[0], row[1]))
    return tuple(row[1] for row in rows)


def permuted_occupancy(np: Any, occupancy: Any, block_ordinal: int) -> Any:
    array = np.asarray(occupancy, dtype=np.float64)
    require(array.ndim == 2 and array.shape[0] == LEVELS, "permuted occupancy geometry")
    output = np.empty_like(array)
    for level in range(LEVELS):
        order = np.asarray(state_permutation(block_ordinal, level, int(array.shape[1])), dtype=np.int64)
        output[level] = array[level, order]
        require(bool(np.array_equal(np.sort(output[level]), np.sort(array[level]))), "permutation multiset")
    return output


def parameter_count(law: int, states: int) -> int:
    require(law in LAW_NAMES and type(states) is int and 1 <= states <= MAX_STATES, "parameter geometry")
    return 2 if law == LAW_LOCAL else 2 + 2 * LEVELS * states


def validate_observation(np: Any, block: Observation, states: int, *, require_target: bool) -> None:
    require(type(block.ordinal) is int and block.ordinal >= 0, "observation ordinal")
    require(block.owners and tuple(sorted(set(block.owners))) == block.owners, "observation owners")
    indices = np.asarray(block.indices)
    require(indices.ndim == 1 and indices.size > 0, "observation indices")
    require(bool(np.all((indices >= 0) & (indices < 64))), "observation index range")
    occupancy = np.asarray(block.occupancy)
    require(occupancy.shape == (LEVELS, states) and bool(np.all(np.isfinite(occupancy))), "observation occupancy")
    if require_target:
        target = np.asarray(block.target_normalized)
        require(target.ndim == 1 and target.size == indices.size and bool(np.all(np.isfinite(target))), "observation target")
    digest(block.coordinate_mapping_sha256, "coordinate mapping")


def _feature_rows(np: Any, block: Observation, law: int, states: int) -> tuple[Any, Any, Any]:
    validate_observation(np, block, states, require_target=True)
    indices = np.asarray(block.indices, dtype=np.int64)
    target = np.asarray(block.target_normalized, dtype=np.float64)
    occupancy = np.asarray(block.occupancy, dtype=np.float64)
    if law == LAW_PERMUTED:
        occupancy = permuted_occupancy(np, occupancy, block.ordinal)
    flattened = occupancy.reshape(-1)
    rows, counts, residual_sums = [], [], []
    for index in range(64):
        selected = indices == index
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        q = 0.25 * float(index - 31)
        feature = np.asarray((1.0, q), dtype=np.float64)
        if law != LAW_LOCAL:
            feature = np.concatenate((feature, flattened, q * flattened))
        rows.append(feature)
        counts.append(count)
        residual_sums.append(float(np.sum(target[selected] - q, dtype=np.float64)))
    require(rows, "nonempty feature rows")
    return np.stack(rows), np.asarray(counts, dtype=np.float64), np.asarray(residual_sums, dtype=np.float64)


def fit_head(np: Any, blocks: Sequence[Observation], *, law: int, states: int, ridge_exponent: int) -> Any:
    require(blocks and ridge_exponent in RIDGE_EXPONENTS, "fit contract")
    width = parameter_count(law, states)
    gram = np.zeros((width, width), dtype=np.float64)
    cross = np.zeros(width, dtype=np.float64)
    observations = 0
    for block in sorted(blocks, key=lambda item: item.ordinal):
        features, counts, residuals = _feature_rows(np, block, law, states)
        gram += features.T @ (counts[:, None] * features)
        cross += features.T @ residuals
        observations += int(np.sum(counts, dtype=np.float64))
    require(observations > 0, "fit observations")
    gram /= float(observations)
    cross /= float(observations)
    diagonal = np.diag(gram)
    require(bool(np.all(diagonal >= 0.0)), "fit Gram diagonal")
    scales = np.maximum(np.sqrt(diagonal), float(2.0 ** -20))
    normalized_gram = gram / scales[:, None] / scales[None, :]
    normalized_cross = cross / scales
    regularized = normalized_gram + float(2.0 ** ridge_exponent) * np.eye(width, dtype=np.float64)
    try:
        solution = np.linalg.solve(regularized, normalized_cross) / scales
    except Exception as error:
        raise AuditError(f"independent ridge solve failed: {error}") from error
    require(solution.shape == (width,) and bool(np.all(np.isfinite(solution))), "fit output")
    return solution.astype(np.float64, copy=False)


def predict_normalized(np: Any, block: Observation, parameters: Any, *, law: int, states: int) -> Any:
    validate_observation(np, block, states, require_target=False)
    values = np.asarray(parameters, dtype=np.float64)
    require(values.shape == (parameter_count(law, states),), "prediction parameters")
    occupancy = np.asarray(block.occupancy, dtype=np.float64)
    if law == LAW_PERMUTED:
        occupancy = permuted_occupancy(np, occupancy, block.ordinal)
    flat = occupancy.reshape(-1)
    indices = np.asarray(block.indices, dtype=np.int64)
    output = np.empty(indices.size, dtype=np.float64)
    for index in range(64):
        selected = indices == index
        if not bool(np.any(selected)):
            continue
        q = 0.25 * float(index - 31)
        correction = float(values[0]) + float(values[1]) * q
        if law != LAW_LOCAL:
            split = LEVELS * states
            correction += float(np.dot(values[2 : 2 + split], flat))
            correction += q * float(np.dot(values[2 + split :], flat))
        output[selected] = q + correction
    require(bool(np.all(np.isfinite(output))), "prediction finite")
    return output


def owner_components(experts: int, owner_sets: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    require(type(experts) is int and experts > 0 and owner_sets, "owner component input")
    parent = list(range(experts))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for raw in owner_sets:
        owners = tuple(int(value) for value in raw)
        require(owners and tuple(sorted(set(owners))) == owners, "owner hyperedge")
        require(0 <= owners[0] and owners[-1] < experts, "owner bound")
        for owner in owners[1:]:
            union(owners[0], owner)
    groups: dict[int, list[int]] = {}
    for expert in range(experts):
        groups.setdefault(find(expert), []).append(expert)
    return tuple(tuple(values) for values in sorted(groups.values(), key=lambda row: row[0]))


def component_blocks(blocks: Sequence[Observation], components: Sequence[Sequence[int]], ordinals: Sequence[int]) -> tuple[Observation, ...]:
    selected = {int(expert) for ordinal in ordinals for expert in components[int(ordinal)]}
    output = []
    for block in blocks:
        owners = set(block.owners)
        require(owners <= selected or owners.isdisjoint(selected), "component cut crosses stream")
        if owners <= selected:
            output.append(block)
    require(output, "component block selection")
    return tuple(sorted(output, key=lambda item: item.ordinal))


def serialize_head(np: Any, parameters: Any, *, law: int, states: int, ridge_exponent: int, handoff_root_sha256: str) -> bytes:
    width = parameter_count(law, states)
    require(ridge_exponent in RIDGE_EXPONENTS, "head ridge")
    values = np.asarray(parameters, dtype=np.float64)
    require(values.shape == (width,) and bool(np.all(np.isfinite(values))), "head parameters")
    half = values.astype("<f2")
    require(bool(np.all(np.isfinite(half))), "head binary16 range")
    payload = half.tobytes(order="C")
    binding = bytes.fromhex(digest(handoff_root_sha256, "head handoff"))
    zero = HEAD_HEADER.pack(HEAD_MAGIC, HEAD_VERSION, law, states, LEVELS, width, len(payload), ridge_exponent, HEAD_FLAGS, binding, hashlib.sha256(payload).digest(), 0)
    checksum = zlib.crc32(zero + payload) & 0xFFFFFFFF
    return HEAD_HEADER.pack(HEAD_MAGIC, HEAD_VERSION, law, states, LEVELS, width, len(payload), ridge_exponent, HEAD_FLAGS, binding, hashlib.sha256(payload).digest(), checksum) + payload


def parse_head(np: Any, packet: bytes, *, expected_handoff_root_sha256: str) -> dict[str, Any]:
    require(isinstance(packet, bytes) and len(packet) >= HEAD_HEADER_BYTES, "head bytes")
    fields = HEAD_HEADER.unpack(packet[:HEAD_HEADER_BYTES])
    magic, version, law, states, levels, width, payload_bytes, exponent, flags, binding, payload_hash, checksum = fields
    require(magic == HEAD_MAGIC and version == HEAD_VERSION, "head magic/version")
    require(law in LAW_NAMES and levels == LEVELS and 1 <= states <= MAX_STATES, "head law geometry")
    require(width == parameter_count(law, states) and payload_bytes == 2 * width, "head payload geometry")
    require(len(packet) == HEAD_HEADER_BYTES + payload_bytes and exponent in RIDGE_EXPONENTS, "head length/ridge")
    require(flags == HEAD_FLAGS and binding.hex() == digest(expected_handoff_root_sha256, "head binding"), "head flags/binding")
    payload = packet[HEAD_HEADER_BYTES:]
    require(hashlib.sha256(payload).digest() == payload_hash, "head payload hash")
    zero = HEAD_HEADER.pack(*fields[:-1], 0)
    require((zlib.crc32(zero + payload) & 0xFFFFFFFF) == checksum, "head CRC")
    parameters = np.frombuffer(payload, dtype="<f2").astype(np.float64)
    require(bool(np.all(np.isfinite(parameters))), "head decoded finite")
    require(serialize_head(np, parameters, law=law, states=states, ridge_exponent=exponent, handoff_root_sha256=expected_handoff_root_sha256) == packet, "head canonical reencode")
    return {
        "law": law,
        "law_name": LAW_NAMES[law],
        "states": states,
        "levels": levels,
        "parameters": parameters,
        "parameter_count": width,
        "ridge_exponent": exponent,
        "packet_bytes": len(packet),
        "packet_sha256": sha256(packet),
        "canonical_reencode_matches": True,
    }


def _footer(*, flags: int, inner_bytes: int, head_bytes: int, logical_end: int, weights: int, experts: int, fold_ordinal: int, inner_hash: bytes, head_hash: bytes, handoff_hash: bytes, checksum: int) -> bytes:
    return WRAPPER_FOOTER.pack(WRAPPER_MAGIC, WRAPPER_VERSION, WRAPPER_FOOTER_BYTES, flags, inner_bytes, inner_bytes, head_bytes, PAGE_BYTES, logical_end, weights, experts, fold_ordinal, inner_hash, head_hash, handoff_hash, checksum, bytes(28))


def build_wrapper(inner: bytes, head: bytes, *, weights: int, experts: int, fold_ordinal: int, handoff_root_sha256: str) -> bytes:
    require(isinstance(inner, bytes) and inner and len(inner) % PAGE_BYTES == 0, "wrapper inner")
    require(isinstance(head, bytes) and 0 < len(head) <= PAGE_BYTES - WRAPPER_FOOTER_BYTES, "wrapper head")
    require(type(weights) is int and weights > 0 and type(experts) is int and experts > 0, "wrapper dimensions")
    require(type(fold_ordinal) is int and -1 <= fold_ordinal < experts, "wrapper fold")
    extension = bytearray(PAGE_BYTES)
    extension[:len(head)] = head
    logical_end = len(inner) + PAGE_BYTES
    parameters = dict(flags=WRAPPER_FLAGS, inner_bytes=len(inner), head_bytes=len(head), logical_end=logical_end, weights=weights, experts=experts, fold_ordinal=fold_ordinal, inner_hash=hashlib.sha256(inner).digest(), head_hash=hashlib.sha256(head).digest(), handoff_hash=bytes.fromhex(digest(handoff_root_sha256, "wrapper handoff")))
    extension[-WRAPPER_FOOTER_BYTES:] = _footer(**parameters, checksum=0)
    checksum = zlib.crc32(extension) & 0xFFFFFFFF
    extension[-WRAPPER_FOOTER_BYTES:] = _footer(**parameters, checksum=checksum)
    return inner + bytes(extension)


def parse_wrapper(np: Any, raw: bytes, *, expected_handoff_root_sha256: str) -> dict[str, Any]:
    require(isinstance(raw, bytes) and len(raw) >= 2 * PAGE_BYTES and len(raw) % PAGE_BYTES == 0, "wrapper object")
    fields = WRAPPER_FOOTER.unpack(raw[-WRAPPER_FOOTER_BYTES:])
    magic, version, footer_bytes, flags, inner_bytes, head_offset, head_bytes, extension_bytes, logical_end, weights, experts, fold_ordinal, inner_hash, head_hash, handoff_hash, checksum, reserved = fields
    require(magic == WRAPPER_MAGIC and version == WRAPPER_VERSION, "wrapper magic/version")
    require(footer_bytes == WRAPPER_FOOTER_BYTES and extension_bytes == PAGE_BYTES and flags == WRAPPER_FLAGS, "wrapper footer geometry")
    require(inner_bytes + PAGE_BYTES == len(raw) == logical_end and head_offset == inner_bytes and inner_bytes % PAGE_BYTES == 0, "wrapper offsets")
    require(0 < head_bytes <= PAGE_BYTES - WRAPPER_FOOTER_BYTES and weights > 0 and experts > 0 and -1 <= fold_ordinal < experts, "wrapper fields")
    require(reserved == bytes(28) and handoff_hash.hex() == digest(expected_handoff_root_sha256, "wrapper handoff"), "wrapper reserved/handoff")
    inner = raw[:inner_bytes]
    extension = bytearray(raw[inner_bytes:])
    head = bytes(extension[:head_bytes])
    require(hashlib.sha256(inner).digest() == inner_hash and hashlib.sha256(head).digest() == head_hash, "wrapper embedded hashes")
    require(bytes(extension[head_bytes:-WRAPPER_FOOTER_BYTES]) == bytes(PAGE_BYTES - WRAPPER_FOOTER_BYTES - head_bytes), "wrapper zero padding")
    parameters = dict(flags=flags, inner_bytes=inner_bytes, head_bytes=head_bytes, logical_end=logical_end, weights=weights, experts=experts, fold_ordinal=fold_ordinal, inner_hash=inner_hash, head_hash=head_hash, handoff_hash=handoff_hash)
    extension[-WRAPPER_FOOTER_BYTES:] = _footer(**parameters, checksum=0)
    require((zlib.crc32(extension) & 0xFFFFFFFF) == checksum, "wrapper CRC")
    parsed_head = parse_head(np, head, expected_handoff_root_sha256=expected_handoff_root_sha256)
    require(build_wrapper(inner, head, weights=weights, experts=experts, fold_ordinal=fold_ordinal, handoff_root_sha256=expected_handoff_root_sha256) == raw, "wrapper canonical reencode")
    return {
        "inner": inner,
        "head": head,
        "parsed_head": parsed_head,
        "weights": weights,
        "experts": experts,
        "fold_ordinal": fold_ordinal,
        "inner_bytes": inner_bytes,
        "head_bytes": head_bytes,
        "extension_bytes": extension_bytes,
        "total_bytes": len(raw),
        "wrapper_sha256": sha256(raw),
        "canonical_reencode_matches": True,
    }


def read_summary(ranges: Sequence[Sequence[int]], total_bytes: int) -> dict[str, Any]:
    normalized = []
    pages = set()
    requested = 0
    for item in ranges:
        require(isinstance(item, (list, tuple)) and len(item) == 2, "read range")
        begin, end = int(item[0]), int(item[1])
        require(0 <= begin <= end <= total_bytes, "read bounds")
        normalized.append((begin, end))
        requested += end - begin
        if end > begin:
            pages.update(range(begin // PAGE_BYTES, (end - 1) // PAGE_BYTES + 1))
    intervals = sorted((begin, end) for begin, end in normalized if end > begin)
    unique = 0
    if intervals:
        start, stop = intervals[0]
        for begin, end in intervals[1:]:
            if begin > stop:
                unique += stop - start
                start, stop = begin, end
            else:
                stop = max(stop, end)
        unique += stop - start
    return {
        "ranges": normalized,
        "touched_page_indices": sorted(pages),
        "touched_page_bytes": len(pages) * PAGE_BYTES,
        "read_request_count": len(normalized),
        "requested_bytes_with_repetition": requested,
        "unique_requested_bytes": unique,
        "overlap_bytes_requested_again": requested - unique,
    }


def wrapper_ledger(*, inner_metrics: Mapping[str, Any], wrapper: Mapping[str, Any], weights_by_expert: Sequence[int]) -> dict[str, Any]:
    """Recompute the v0 nonpromoting read projection from literal geometry."""

    experts = len(weights_by_expert)
    rows = inner_metrics.get("experts")
    require(isinstance(rows, list) and len(rows) == experts, "inner metric experts")
    require(int(inner_metrics.get("actual_container_bytes", -1)) == int(wrapper["inner_bytes"]), "inner byte ledger")
    total_weights = sum(int(value) for value in weights_by_expert)
    require(total_weights == int(wrapper["weights"]) and all(type(value) is int and value > 0 for value in weights_by_expert), "ledger weights")
    head_bytes = int(wrapper["head_bytes"])
    nonpadding_extension = head_bytes + WRAPPER_FOOTER_BYTES
    output = []
    maximum_page = Fraction(0, 1)
    maximum_repeated = Fraction(0, 1)
    maximum_unique = Fraction(0, 1)
    for expert, inner in enumerate(rows):
        require(int(inner["expert_ordinal"]) == expert, "inner expert order")
        causal = inner.get("causal_decode_reencode_reconstruction")
        require(
            isinstance(causal, Mapping)
            and causal.get("all_payloads_canonically_reencoded") is True
            and causal.get("all_three_roles_reconstructed") is True,
            "inner causal reconstruction proof",
        )
        inner_ranges = inner["instrumented_routed_read_ranges"]
        require(isinstance(inner_ranges, list) and inner_ranges, "inner routed ranges")
        combined = [(int(wrapper["inner_bytes"]), int(wrapper["inner_bytes"]) + PAGE_BYTES)] + [tuple(item) for item in inner_ranges]
        summary = read_summary(combined, int(wrapper["total_bytes"]))
        require(summary["read_request_count"] == 1 + int(inner["instrumented_routed_read_request_count"]), "projected request count")
        alpha = Fraction(int(weights_by_expert[expert]), total_weights)
        attributed_total = fraction_from_record(inner["attributable_total_physical_bytes"], "inner attributed total") + alpha * PAGE_BYTES
        attributed_nonpadding = fraction_from_record(inner["attributable_nonpadding_decodable_bytes"], "inner attributed nonpadding") + alpha * nonpadding_extension
        touched, requested, unique = summary["touched_page_bytes"], summary["requested_bytes_with_repetition"], summary["unique_requested_bytes"]
        page_total, page_nonpadding = Fraction(touched, 1) / attributed_total, Fraction(touched, 1) / attributed_nonpadding
        repeated_total, repeated_nonpadding = Fraction(requested, 1) / attributed_total, Fraction(requested, 1) / attributed_nonpadding
        unique_total, unique_nonpadding = Fraction(unique, 1) / attributed_total, Fraction(unique, 1) / attributed_nonpadding
        page_strict = max(page_total, page_nonpadding)
        repeated_strict = max(repeated_total, repeated_nonpadding)
        unique_strict = max(unique_total, unique_nonpadding)
        strict = max(page_strict, repeated_strict, unique_strict)
        maximum_page = max(maximum_page, page_strict)
        maximum_repeated = max(maximum_repeated, repeated_strict)
        maximum_unique = max(maximum_unique, unique_strict)
        output.append({
            "expert_ordinal": expert,
            "source_weights": int(weights_by_expert[expert]),
            "extension_owner_fraction": fraction_record(alpha),
            "attributable_total_physical_bytes": fraction_record(attributed_total),
            "attributable_nonpadding_decodable_bytes": fraction_record(attributed_nonpadding),
            "touched_page_bytes": touched,
            "requested_bytes_with_repetition": requested,
            "unique_requested_bytes": unique,
            "read_request_count": summary["read_request_count"],
            "overlap_bytes_requested_again": summary["overlap_bytes_requested_again"],
            "descriptor_page_amplification_total_physical": fraction_record(page_total),
            "descriptor_page_amplification_nonpadding": fraction_record(page_nonpadding),
            "requested_with_repetition_amplification_total_physical": fraction_record(repeated_total),
            "requested_with_repetition_amplification_nonpadding": fraction_record(repeated_nonpadding),
            "unique_requested_amplification_total_physical": fraction_record(unique_total),
            "unique_requested_amplification_nonpadding": fraction_record(unique_nonpadding),
            "cold_amplification_total_physical": fraction_record(max(page_total, repeated_total, unique_total)),
            "cold_amplification_nonpadding": fraction_record(max(page_nonpadding, repeated_nonpadding, unique_nonpadding)),
            "strict_cold_amplification": fraction_record(strict),
            "passes_descriptor_pages_below_2x": page_strict < 2,
            "passes_requested_with_repetition_below_2x": repeated_strict < 2,
            "passes_unique_requested_below_2x": unique_strict < 2,
            "compressed_expert_second_pass": False,
            "compressed_expert_second_pass_absent_derived_from_instrumented_invocations": True,
            "inner_decode_invocations": 1,
            "actual_inner_routed_decode_proof_sha256": None,
            "independent_projected_outer_read_ranges": [list(item) for item in combined],
        })
    rate = Fraction(8 * int(wrapper["total_bytes"]), total_weights)
    maximum = max(maximum_page, maximum_repeated, maximum_unique)
    return {
        "inner_bytes": int(wrapper["inner_bytes"]),
        "head_bytes": head_bytes,
        "footer_bytes": WRAPPER_FOOTER_BYTES,
        "extension_nonpadding_bytes": nonpadding_extension,
        "extension_zero_padding_bytes": PAGE_BYTES - nonpadding_extension,
        "extension_physical_bytes": PAGE_BYTES,
        "outer_bytes": int(wrapper["total_bytes"]),
        "physical_rate_bpw": fraction_record(rate),
        "experts": output,
        "maximum_descriptor_page_amplification_strict": fraction_record(maximum_page),
        "maximum_requested_with_repetition_amplification_strict": fraction_record(maximum_repeated),
        "maximum_unique_requested_amplification_strict": fraction_record(maximum_unique),
        "maximum_strict_cold_read_amplification": fraction_record(maximum),
        "passes_descriptor_pages_below_2x": maximum_page < 2,
        "passes_requested_with_repetition_below_2x": maximum_repeated < 2,
        "passes_unique_requested_below_2x": maximum_unique < 2,
        "passes_strict_cold_read_below_2x": maximum < 2,
        "additional_storage_pages_per_routed_expert": 1,
        "compressed_expert_second_pass_forbidden_and_absent": True,
        "actual_inner_routed_decode_executed": True,
        "actual_posterior_wrapper_routed_decode_executed": False,
        "posterior_head_applied_to_routed_reconstruction": False,
        "read_claim_is_nonpromoting_projection_from_instrumented_inner_decode_plus_literal_suffix": True,
        "scratch_hbm_is_not_counted_as_storage_read": True,
    }


def allocated_component_rate(ledger: Mapping[str, Any], weights_by_expert: Sequence[int], component: Sequence[int]) -> float:
    rows = ledger["experts"]
    attributed = sum((fraction_from_record(rows[int(expert)]["attributable_total_physical_bytes"], "component attributed") for expert in component), Fraction(0, 1))
    weights = sum(int(weights_by_expert[int(expert)]) for expert in component)
    require(weights > 0, "component weights")
    return float(Fraction(8, weights) * attributed)


def delta_s(*, baseline_rate: float, candidate_rate: float, baseline_distortion: float, candidate_distortion: float) -> float:
    values = (baseline_rate, candidate_rate, baseline_distortion, candidate_distortion)
    require(all(math.isfinite(float(value)) for value in values) and baseline_distortion > 0.0 and candidate_distortion > 0.0, "Delta-s inputs")
    return (baseline_rate - candidate_rate) - 0.5 * math.log2(candidate_distortion / baseline_distortion)


def fold_gate(*, delta_s_value: float, g_state_value: float, candidate_rate_bpw: float, candidate_f: float, cold_read_below_2x: bool) -> dict[str, bool]:
    finite = all(math.isfinite(float(value)) for value in (delta_s_value, g_state_value, candidate_rate_bpw, candidate_f))
    checks = {
        "passes_positive_Delta_s": finite and delta_s_value > 0.0,
        "passes_positive_G_state": finite and g_state_value > 0.0,
        "passes_rate_interval": finite and 2.15 <= candidate_rate_bpw <= 2.5,
        "passes_F_target": finite and candidate_f <= 0.8,
        "passes_cold_read_below_2x": cold_read_below_2x is True,
    }
    checks["passes_all_fold_gates"] = all(checks.values())
    return checks


require(HEAD_HEADER.size == HEAD_HEADER_BYTES, "head struct size")
require(WRAPPER_FOOTER.size == WRAPPER_FOOTER_BYTES, "footer struct size")
