#!/usr/bin/env python3
"""Source-independent core for the UWFA-SC posterior-centroid v0 gate.

The module deliberately imports no numerical library and performs no path,
payload, network, subprocess, or CUDA access at import time.  Callers inject a
NumPy-compatible module after authenticating their inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA = "uwfa-sc-posterior-centroid-core-v0"
LEVELS = 6
PAGE_BYTES = 4096
MAX_STATES = 64
RIDGE_EXPONENTS = (-28, -24, -20, -16, -12, -8, -4, 0)
PERMUTATION_DOMAIN = b"UWFA-SC-POSTERIOR-STATE-PERM-V0\x00"
PERMUTATION_SEED = bytes.fromhex(
    "c7a995f5ba0b0a6a3097890ad936f6ef5a9233faf5976abc02ae3eacb4400f7d"
)

LAW_LOCAL = 0
LAW_STATE = 1
LAW_STATE_PERMUTED = 2
LAW_NAMES = {
    LAW_LOCAL: "local-only",
    LAW_STATE: "state-aware",
    LAW_STATE_PERMUTED: "state-permuted",
}

HEAD_MAGIC = b"CAGEPC0\x00"
HEAD_VERSION = 1
HEAD_HEADER = struct.Struct("<8sHHHHIIhH32s32sI")
HEAD_HEADER_BYTES = 96
HEAD_FLAG_CENTERED_NONEMPTY = 1 << 0
HEAD_FLAG_ALL_PARAMETERS_RIDGED = 1 << 1

WRAPPER_MAGIC = b"CAGEPST1"
WRAPPER_VERSION = 1
WRAPPER_FOOTER = struct.Struct("<8sHHIQQIIQQIi32s32s32sI28s")
WRAPPER_FOOTER_BYTES = 192
WRAPPER_EXTENSION_BYTES = PAGE_BYTES
WRAPPER_FLAG_SUFFIX_LAYOUT = 1 << 0
WRAPPER_FLAG_INNER_UNCHANGED = 1 << 1
WRAPPER_FLAG_ZERO_PADDING = 1 << 2

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


class PosteriorContractError(RuntimeError):
    """Fail-closed contract error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PosteriorContractError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} SHA-256",
    )
    return value


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "float": float(value),
    }


def posterior_handoff_root(handoff: Mapping[str, Any]) -> str:
    require(
        handoff.get("schema") == "uwfa-sc-v8-posterior-diagnostic-handoff",
        "posterior handoff schema",
    )
    require(handoff.get("requires_literal_redecode") is True, "literal redecode requirement")
    require(handoff.get("contains_posterior_or_MMSE_result") is False, "clean predecessor handoff")
    bound = {key: require_digest(handoff.get(key), key) for key in HANDOFF_KEYS}
    rows = handoff.get("stream_decision_triplet_commitments")
    require(isinstance(rows, list) and rows, "stream decision commitments")
    require(int(handoff.get("stream_count", -1)) == len(rows), "handoff stream count")
    previous = -1
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, dict), "stream commitment row")
        ordinal = row.get("ordinal")
        require(type(ordinal) is int and ordinal == previous + 1, "stream commitment order")
        previous = ordinal
        clean_rows.append({
            "ordinal": ordinal,
            "symbols": int(row["symbols"]),
            "logical_bits": int(row["logical_bits"]),
            "decoded_selected_decision_triplet_sha256": require_digest(
                row["decoded_selected_decision_triplet_sha256"], "stream decision triplet"
            ),
            "payload_sha256": require_digest(row["payload_sha256"], "stream payload"),
            "profile_q": int(row["profile_q"]),
            "role": str(row["role"]),
            "owner_set_hex": str(row["owner_set_hex"]),
            "owner_contributions": row["owner_contributions"],
        })
    require(
        sha256(canonical_json(clean_rows))
        == bound["decoded_sc_decision_triplet_commitment_sha256"],
        "aggregate decision commitment",
    )
    record = {
        "schema": "uwfa-sc-posterior-handoff-root-v0",
        "bindings": bound,
        "stream_decision_triplet_commitments": clean_rows,
    }
    return sha256(canonical_json(record))


def trace_predecision_states(
    common: Any,
    candidate: Any,
    bits: Sequence[int],
    levels: Sequence[int],
    base_frequencies: Sequence[int],
) -> list[int]:
    """Replay the exact state before every causal selected SC decision."""

    require(
        len(bits) == len(levels) == len(base_frequencies) and len(bits) > 0,
        "state-trace stream geometry",
    )
    states = int(candidate.states)
    require(1 <= states <= MAX_STATES, "candidate state bound")
    reset = int(candidate.reset_length)
    require(reset > 0, "candidate reset length")
    state = 0
    output: list[int] = []
    for position, (raw_bit, raw_level, raw_base) in enumerate(
        zip(bits, levels, base_frequencies, strict=True)
    ):
        within = position % reset
        if within == 0:
            state = 0
        bit = int(raw_bit)
        level = int(raw_level)
        base = int(raw_base)
        require(bit in (0, 1), "nonbinary selected decision")
        require(0 <= level < LEVELS, "selected decision level")
        context = common.public_context(level, base, within)
        output.append(state)
        state = int(common.transition(candidate, state, bit, context, within))
        require(0 <= state < states, "state transition bound")
    return output


def occupancy_features(
    np: Any,
    levels: Any,
    pre_states: Any,
    states: int,
) -> Any:
    """Return centered per-level pre-decision-state occupancy."""

    require(type(states) is int and 1 <= states <= MAX_STATES, "occupancy state bound")
    levels_array = np.asarray(levels)
    states_array = np.asarray(pre_states)
    require(levels_array.ndim == states_array.ndim == 1, "occupancy rank")
    require(levels_array.size == states_array.size and levels_array.size > 0, "occupancy length")
    require(bool(np.all((levels_array >= 0) & (levels_array < LEVELS))), "occupancy levels")
    require(bool(np.all((states_array >= 0) & (states_array < states))), "occupancy states")
    result = np.zeros((LEVELS, states), dtype=np.float64)
    for level in range(LEVELS):
        selected = states_array[levels_array == level]
        if int(selected.size) == 0:
            continue
        counts = np.bincount(selected.astype(np.int64), minlength=states).astype(np.float64)
        result[level] = counts / float(selected.size) - 1.0 / float(states)
        require(abs(float(np.sum(result[level], dtype=np.float64))) <= 64.0 * math.ulp(1.0), "centered occupancy sum")
    return result


def state_permutation(block_ordinal: int, level: int, states: int) -> tuple[int, ...]:
    require(type(block_ordinal) is int and block_ordinal >= 0, "permutation block ordinal")
    require(type(level) is int and 0 <= level < LEVELS, "permutation level")
    require(type(states) is int and 1 <= states <= MAX_STATES, "permutation states")
    rows = []
    for state in range(states):
        material = (
            PERMUTATION_DOMAIN
            + PERMUTATION_SEED
            + struct.pack("<IHH", block_ordinal, level, state)
        )
        rows.append((hashlib.sha256(material).digest(), state))
    rows.sort(key=lambda item: (item[0], item[1]))
    return tuple(state for _digest, state in rows)


def permute_occupancy(np: Any, occupancy: Any, block_ordinal: int) -> Any:
    array = np.asarray(occupancy, dtype=np.float64)
    require(array.ndim == 2 and array.shape[0] == LEVELS, "permuted occupancy geometry")
    states = int(array.shape[1])
    output = np.empty_like(array)
    for level in range(LEVELS):
        permutation = state_permutation(block_ordinal, level, states)
        output[level] = array[level, np.asarray(permutation, dtype=np.int64)]
        require(
            bool(np.array_equal(np.sort(output[level]), np.sort(array[level]))),
            "permutation multiset preservation",
        )
    return output


@dataclass(frozen=True)
class BlockObservation:
    """Coordinate-aligned decoder observation and continuous fit target."""

    ordinal: int
    owners: tuple[int, ...]
    indices: Any
    target_normalized: Any
    occupancy: Any
    coordinate_mapping_sha256: str


def validate_block(
    np: Any,
    block: BlockObservation,
    states: int,
    *,
    require_continuous_target: bool = True,
) -> None:
    require(type(block.ordinal) is int and block.ordinal >= 0, "block ordinal")
    require(
        block.owners
        and tuple(sorted(set(block.owners))) == block.owners
        and all(type(value) is int and value >= 0 for value in block.owners),
        "block owners",
    )
    indices = np.asarray(block.indices)
    target = None if block.target_normalized is None else np.asarray(block.target_normalized)
    occupancy = np.asarray(block.occupancy)
    require(indices.ndim == 1 and indices.size > 0, "decoded coordinate array")
    require(bool(np.all((indices >= 0) & (indices < 64))), "decoded coordinate lattice indices")
    if require_continuous_target:
        require(
            target is not None
            and target.ndim == 1
            and indices.size == target.size,
            "coordinate-aligned block arrays",
        )
        require(bool(np.all(np.isfinite(target))), "continuous target finite")
    require(occupancy.shape == (LEVELS, states), "state occupancy geometry")
    require(bool(np.all(np.isfinite(occupancy))), "state occupancy finite")
    require_digest(block.coordinate_mapping_sha256, "coordinate mapping")


def parameter_count(law: int, states: int) -> int:
    require(law in LAW_NAMES, "posterior law")
    require(type(states) is int and 1 <= states <= MAX_STATES, "parameter state bound")
    return 2 if law == LAW_LOCAL else 2 + 2 * LEVELS * states


def block_feature_rows(np: Any, block: BlockObservation, law: int, states: int) -> tuple[Any, Any, Any]:
    """Aggregate exact sufficient rows by decoded coordinate index.

    The selected SC decisions never appear as bin labels.  `indices` is the
    final six-level coordinate lattice index and is required to be aligned
    one-to-one with the continuous target.
    """

    validate_block(np, block, states, require_continuous_target=True)
    indices = np.asarray(block.indices, dtype=np.int64)
    target = np.asarray(block.target_normalized, dtype=np.float64)
    occupancy = np.asarray(block.occupancy, dtype=np.float64)
    if law == LAW_STATE_PERMUTED:
        occupancy = permute_occupancy(np, occupancy, block.ordinal)
    flattened = occupancy.reshape(-1)
    rows = []
    counts = []
    residual_sums = []
    for index in range(64):
        mask = indices == index
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        q_value = 0.25 * float(index - 31)
        if law == LAW_LOCAL:
            feature = np.asarray((1.0, q_value), dtype=np.float64)
        else:
            feature = np.concatenate((
                np.asarray((1.0, q_value), dtype=np.float64),
                flattened,
                q_value * flattened,
            ))
        residual_sum = float(np.sum(target[mask] - q_value, dtype=np.float64))
        rows.append(feature)
        counts.append(count)
        residual_sums.append(residual_sum)
    require(rows, "nonempty block feature rows")
    return (
        np.stack(rows).astype(np.float64, copy=False),
        np.asarray(counts, dtype=np.float64),
        np.asarray(residual_sums, dtype=np.float64),
    )


def fit_head(
    np: Any,
    blocks: Sequence[BlockObservation],
    *,
    law: int,
    states: int,
    ridge_exponent: int,
) -> Any:
    require(blocks, "nonempty head fit")
    require(ridge_exponent in RIDGE_EXPONENTS, "frozen ridge exponent")
    expected = parameter_count(law, states)
    gram = np.zeros((expected, expected), dtype=np.float64)
    cross = np.zeros(expected, dtype=np.float64)
    observations = 0
    for block in sorted(blocks, key=lambda item: item.ordinal):
        x_rows, counts, residual_sums = block_feature_rows(np, block, law, states)
        require(x_rows.shape[1] == expected, "feature width")
        gram += x_rows.T @ (counts[:, None] * x_rows)
        cross += x_rows.T @ residual_sums
        observations += int(np.sum(counts, dtype=np.float64))
    require(observations > 0, "fit observation count")
    gram /= float(observations)
    cross /= float(observations)
    diagonal = np.diag(gram)
    require(bool(np.all(diagonal >= 0.0)), "Gram diagonal")
    scales = np.maximum(np.sqrt(diagonal), float(2.0 ** -20))
    standardized_gram = gram / scales[:, None] / scales[None, :]
    standardized_cross = cross / scales
    regularized = standardized_gram + float(2.0 ** ridge_exponent) * np.eye(expected, dtype=np.float64)
    try:
        standardized = np.linalg.solve(regularized, standardized_cross)
    except Exception as error:  # pragma: no cover - backend-specific exception type
        raise PosteriorContractError(f"deterministic ridge solve failed: {error}") from error
    parameters = standardized / scales
    require(parameters.shape == (expected,) and bool(np.all(np.isfinite(parameters))), "fitted parameters")
    return parameters.astype(np.float64, copy=False)


def predict_normalized(
    np: Any,
    block: BlockObservation,
    parameters: Any,
    *,
    law: int,
    states: int,
) -> Any:
    validate_block(np, block, states, require_continuous_target=False)
    values = np.asarray(parameters, dtype=np.float64)
    require(values.shape == (parameter_count(law, states),), "prediction parameters")
    occupancy = np.asarray(block.occupancy, dtype=np.float64)
    if law == LAW_STATE_PERMUTED:
        occupancy = permute_occupancy(np, occupancy, block.ordinal)
    flattened = occupancy.reshape(-1)
    output = np.empty(np.asarray(block.indices).size, dtype=np.float64)
    indices = np.asarray(block.indices, dtype=np.int64)
    for index in range(64):
        mask = indices == index
        if not bool(np.any(mask)):
            continue
        q_value = 0.25 * float(index - 31)
        correction = float(values[0]) + float(values[1]) * q_value
        if law != LAW_LOCAL:
            split = LEVELS * states
            correction += float(np.dot(values[2 : 2 + split], flattened))
            correction += q_value * float(np.dot(values[2 + split :], flattened))
        output[mask] = q_value + correction
    require(bool(np.all(np.isfinite(output))), "posterior prediction finite")
    return output


def serialize_head(
    np: Any,
    parameters: Any,
    *,
    law: int,
    states: int,
    ridge_exponent: int,
    handoff_root_sha256: str,
) -> bytes:
    expected = parameter_count(law, states)
    require(ridge_exponent in RIDGE_EXPONENTS, "serialized ridge exponent")
    binding = bytes.fromhex(require_digest(handoff_root_sha256, "posterior handoff root"))
    values = np.asarray(parameters, dtype=np.float64)
    require(values.shape == (expected,) and bool(np.all(np.isfinite(values))), "serialized head parameters")
    half = values.astype("<f2")
    require(bool(np.all(np.isfinite(half))), "binary16 head overflow")
    payload = half.tobytes(order="C")
    payload_digest = hashlib.sha256(payload).digest()
    flags = HEAD_FLAG_CENTERED_NONEMPTY | HEAD_FLAG_ALL_PARAMETERS_RIDGED
    header_zero = HEAD_HEADER.pack(
        HEAD_MAGIC,
        HEAD_VERSION,
        law,
        states,
        LEVELS,
        expected,
        len(payload),
        ridge_exponent,
        flags,
        binding,
        payload_digest,
        0,
    )
    require(len(header_zero) == HEAD_HEADER_BYTES, "head header geometry")
    checksum = zlib.crc32(header_zero + payload) & 0xFFFFFFFF
    header = HEAD_HEADER.pack(
        HEAD_MAGIC,
        HEAD_VERSION,
        law,
        states,
        LEVELS,
        expected,
        len(payload),
        ridge_exponent,
        flags,
        binding,
        payload_digest,
        checksum,
    )
    return header + payload


def parse_head(np: Any, packet: bytes, *, expected_handoff_root_sha256: str) -> dict[str, Any]:
    require(isinstance(packet, bytes) and len(packet) >= HEAD_HEADER_BYTES, "posterior head packet")
    fields = HEAD_HEADER.unpack(packet[:HEAD_HEADER_BYTES])
    (
        magic,
        version,
        law,
        states,
        levels,
        parameters,
        payload_bytes,
        ridge_exponent,
        flags,
        binding,
        payload_digest,
        checksum,
    ) = fields
    require(magic == HEAD_MAGIC and version == HEAD_VERSION, "head magic/version")
    require(levels == LEVELS and law in LAW_NAMES, "head law geometry")
    require(1 <= states <= MAX_STATES, "head state bound")
    require(parameters == parameter_count(law, states), "head parameter count")
    require(payload_bytes == 2 * parameters, "head payload geometry")
    require(len(packet) == HEAD_HEADER_BYTES + payload_bytes, "head logical end")
    require(ridge_exponent in RIDGE_EXPONENTS, "head ridge exponent")
    require(flags == HEAD_FLAG_CENTERED_NONEMPTY | HEAD_FLAG_ALL_PARAMETERS_RIDGED, "head flags")
    require(binding.hex() == require_digest(expected_handoff_root_sha256, "expected handoff root"), "head handoff binding")
    payload = packet[HEAD_HEADER_BYTES:]
    require(hashlib.sha256(payload).digest() == payload_digest, "head payload SHA-256")
    header_zero = HEAD_HEADER.pack(*fields[:-1], 0)
    require((zlib.crc32(header_zero + payload) & 0xFFFFFFFF) == checksum, "head CRC")
    values = np.frombuffer(payload, dtype="<f2").astype(np.float64)
    require(values.shape == (parameters,) and bool(np.all(np.isfinite(values))), "decoded head parameters")
    canonical = serialize_head(
        np,
        values,
        law=law,
        states=states,
        ridge_exponent=ridge_exponent,
        handoff_root_sha256=expected_handoff_root_sha256,
    )
    require(canonical == packet, "head canonical re-encode")
    return {
        "law": law,
        "law_name": LAW_NAMES[law],
        "states": states,
        "levels": levels,
        "parameters": values,
        "parameter_count": parameters,
        "ridge_exponent": ridge_exponent,
        "packet_bytes": len(packet),
        "packet_sha256": sha256(packet),
        "canonical_reencode_matches": True,
    }


def _pack_footer(
    *,
    flags: int,
    inner_bytes: int,
    head_bytes: int,
    logical_end: int,
    weights: int,
    experts: int,
    fold_ordinal: int,
    inner_digest: bytes,
    head_digest: bytes,
    handoff_digest: bytes,
    checksum: int,
) -> bytes:
    return WRAPPER_FOOTER.pack(
        WRAPPER_MAGIC,
        WRAPPER_VERSION,
        WRAPPER_FOOTER_BYTES,
        flags,
        inner_bytes,
        inner_bytes,
        head_bytes,
        WRAPPER_EXTENSION_BYTES,
        logical_end,
        weights,
        experts,
        fold_ordinal,
        inner_digest,
        head_digest,
        handoff_digest,
        checksum,
        bytes(28),
    )


def build_wrapper(
    inner: bytes,
    head: bytes,
    *,
    weights: int,
    experts: int,
    fold_ordinal: int,
    handoff_root_sha256: str,
) -> bytes:
    require(isinstance(inner, bytes) and inner and len(inner) % PAGE_BYTES == 0, "page-aligned unchanged inner")
    require(isinstance(head, bytes) and 0 < len(head) <= PAGE_BYTES - WRAPPER_FOOTER_BYTES, "one-page head")
    require(type(weights) is int and weights > 0, "wrapper weights")
    require(type(experts) is int and experts > 0, "wrapper experts")
    require(type(fold_ordinal) is int and -1 <= fold_ordinal < experts, "wrapper fold ordinal")
    flags = WRAPPER_FLAG_SUFFIX_LAYOUT | WRAPPER_FLAG_INNER_UNCHANGED | WRAPPER_FLAG_ZERO_PADDING
    extension = bytearray(PAGE_BYTES)
    extension[: len(head)] = head
    logical_end = len(inner) + PAGE_BYTES
    footer_zero = _pack_footer(
        flags=flags,
        inner_bytes=len(inner),
        head_bytes=len(head),
        logical_end=logical_end,
        weights=weights,
        experts=experts,
        fold_ordinal=fold_ordinal,
        inner_digest=hashlib.sha256(inner).digest(),
        head_digest=hashlib.sha256(head).digest(),
        handoff_digest=bytes.fromhex(require_digest(handoff_root_sha256, "wrapper handoff root")),
        checksum=0,
    )
    extension[-WRAPPER_FOOTER_BYTES:] = footer_zero
    checksum = zlib.crc32(extension) & 0xFFFFFFFF
    extension[-WRAPPER_FOOTER_BYTES:] = _pack_footer(
        flags=flags,
        inner_bytes=len(inner),
        head_bytes=len(head),
        logical_end=logical_end,
        weights=weights,
        experts=experts,
        fold_ordinal=fold_ordinal,
        inner_digest=hashlib.sha256(inner).digest(),
        head_digest=hashlib.sha256(head).digest(),
        handoff_digest=bytes.fromhex(handoff_root_sha256),
        checksum=checksum,
    )
    return inner + bytes(extension)


def parse_wrapper(np: Any, raw: bytes, *, expected_handoff_root_sha256: str) -> dict[str, Any]:
    require(isinstance(raw, bytes) and len(raw) >= 2 * PAGE_BYTES and len(raw) % PAGE_BYTES == 0, "wrapper object")
    footer_bytes = raw[-WRAPPER_FOOTER_BYTES:]
    fields = WRAPPER_FOOTER.unpack(footer_bytes)
    (
        magic,
        version,
        footer_size,
        flags,
        inner_bytes,
        head_offset,
        head_bytes,
        extension_bytes,
        logical_end,
        weights,
        experts,
        fold_ordinal,
        inner_digest,
        head_digest,
        handoff_digest,
        checksum,
        reserved,
    ) = fields
    require(magic == WRAPPER_MAGIC and version == WRAPPER_VERSION, "wrapper magic/version")
    require(footer_size == WRAPPER_FOOTER_BYTES and extension_bytes == PAGE_BYTES, "wrapper geometry")
    require(flags == WRAPPER_FLAG_SUFFIX_LAYOUT | WRAPPER_FLAG_INNER_UNCHANGED | WRAPPER_FLAG_ZERO_PADDING, "wrapper flags")
    require(inner_bytes + extension_bytes == len(raw) == logical_end, "wrapper logical end")
    require(head_offset == inner_bytes and inner_bytes % PAGE_BYTES == 0, "wrapper inner/head offset")
    require(0 < head_bytes <= PAGE_BYTES - WRAPPER_FOOTER_BYTES, "wrapper head bound")
    require(weights > 0 and experts > 0 and -1 <= fold_ordinal < experts, "wrapper dimensions")
    require(reserved == bytes(28), "wrapper reserved bytes")
    require(handoff_digest.hex() == require_digest(expected_handoff_root_sha256, "expected wrapper handoff"), "wrapper handoff binding")
    inner = raw[:inner_bytes]
    extension = bytearray(raw[inner_bytes:])
    head = bytes(extension[:head_bytes])
    require(hashlib.sha256(inner).digest() == inner_digest, "wrapper inner SHA-256")
    require(hashlib.sha256(head).digest() == head_digest, "wrapper head SHA-256")
    require(bytes(extension[head_bytes:-WRAPPER_FOOTER_BYTES]) == bytes(PAGE_BYTES - WRAPPER_FOOTER_BYTES - head_bytes), "wrapper zero padding")
    extension[-WRAPPER_FOOTER_BYTES:] = _pack_footer(
        flags=flags,
        inner_bytes=inner_bytes,
        head_bytes=head_bytes,
        logical_end=logical_end,
        weights=weights,
        experts=experts,
        fold_ordinal=fold_ordinal,
        inner_digest=inner_digest,
        head_digest=head_digest,
        handoff_digest=handoff_digest,
        checksum=0,
    )
    require((zlib.crc32(extension) & 0xFFFFFFFF) == checksum, "wrapper CRC")
    parsed_head = parse_head(np, head, expected_handoff_root_sha256=expected_handoff_root_sha256)
    canonical = build_wrapper(
        inner,
        head,
        weights=weights,
        experts=experts,
        fold_ordinal=fold_ordinal,
        handoff_root_sha256=expected_handoff_root_sha256,
    )
    require(canonical == raw, "wrapper canonical re-encode")
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


def owner_components(experts: int, owner_sets: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    require(type(experts) is int and experts > 0, "component expert count")
    parent = list(range(experts))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a == b:
            return
        if a < b:
            parent[b] = a
        else:
            parent[a] = b

    require(owner_sets, "owner hyperedges")
    for raw in owner_sets:
        owners = tuple(int(value) for value in raw)
        require(owners and tuple(sorted(set(owners))) == owners, "owner hyperedge")
        require(0 <= owners[0] and owners[-1] < experts, "owner hyperedge bound")
        for value in owners[1:]:
            union(owners[0], value)
    groups: dict[int, list[int]] = {}
    for expert in range(experts):
        groups.setdefault(find(expert), []).append(expert)
    return tuple(tuple(values) for _root, values in sorted(groups.items(), key=lambda item: item[1][0]))


def blocks_for_components(
    blocks: Sequence[BlockObservation],
    components: Sequence[Sequence[int]],
    selected_component_ordinals: Sequence[int],
) -> tuple[BlockObservation, ...]:
    selected = set()
    for ordinal in selected_component_ordinals:
        require(type(ordinal) is int and 0 <= ordinal < len(components), "selected component ordinal")
        selected.update(int(value) for value in components[ordinal])
    output = []
    for block in blocks:
        owners = set(block.owners)
        require(owners <= selected or owners.isdisjoint(selected), "block crosses connected-component boundary")
        if owners <= selected:
            output.append(block)
    require(output, "selected component blocks")
    return tuple(sorted(output, key=lambda item: item.ordinal))


def select_ridge_for_outer(
    np: Any,
    blocks: Sequence[BlockObservation],
    components: Sequence[Sequence[int]],
    *,
    outer_component: int,
    law: int,
    states: int,
    score_sse: Callable[[Any, int, int], float],
) -> dict[str, Any]:
    """Two-direction inner selection for exactly three ownership components."""

    require(len(components) == 3, "v0 requires three whole ownership components")
    require(type(outer_component) is int and 0 <= outer_component < 3, "outer component")
    development = tuple(value for value in range(3) if value != outer_component)
    rows = []
    for exponent in RIDGE_EXPONENTS:
        summed = 0.0
        directions = []
        for train_component, validation_component in (development, development[::-1]):
            train_blocks = blocks_for_components(blocks, components, (train_component,))
            parameters = fit_head(
                np,
                train_blocks,
                law=law,
                states=states,
                ridge_exponent=exponent,
            )
            sse = float(score_sse(parameters, law, validation_component))
            require(math.isfinite(sse) and sse >= 0.0, "inner validation SSE")
            summed += sse
            directions.append({
                "train_component": train_component,
                "validation_component": validation_component,
                "validation_sse_fp64": sse,
            })
        rows.append({
            "ridge_exponent": exponent,
            "summed_bidirectional_validation_sse_fp64": summed,
            "directions": directions,
        })
    winner = min(rows, key=lambda row: (row["summed_bidirectional_validation_sse_fp64"], row["ridge_exponent"]))
    final_blocks = blocks_for_components(blocks, components, development)
    parameters = fit_head(
        np,
        final_blocks,
        law=law,
        states=states,
        ridge_exponent=int(winner["ridge_exponent"]),
    )
    return {
        "outer_component": outer_component,
        "development_components": list(development),
        "law": law,
        "law_name": LAW_NAMES[law],
        "ridge_grid": rows,
        "selected_ridge_exponent": int(winner["ridge_exponent"]),
        "selected_by_bidirectional_inner_validation_only": True,
        "refit_parameters": parameters,
    }


def wrapper_read_ledger(
    *,
    routed_wrapper_trace: Mapping[str, Any],
    weights_by_expert: Sequence[int],
    inner_attributed_total: Sequence[Fraction],
    inner_attributed_nonpadding: Sequence[Fraction],
    head_bytes: int,
) -> dict[str, Any]:
    experts = len(weights_by_expert)
    require(experts > 0, "ledger experts")
    require(
        all(len(values) == experts for values in (
            inner_attributed_total,
            inner_attributed_nonpadding,
        )),
        "ledger vector geometry",
    )
    require(
        isinstance(routed_wrapper_trace, Mapping)
        and routed_wrapper_trace.get("schema") == "uwfa-sc-posterior-wrapper-routed-read-proof-v0",
        "instrumented wrapper trace",
    )
    trace_rows = routed_wrapper_trace.get("experts")
    require(isinstance(trace_rows, list) and len(trace_rows) == experts, "instrumented wrapper experts")
    require(
        routed_wrapper_trace.get("proof_uses_actual_authenticated_v8_routed_decoder") is True,
        "actual authenticated inner routed proof",
    )
    require(
        routed_wrapper_trace.get("compressed_expert_second_pass_forbidden_and_absent") is True,
        "wrapper compressed second-pass proof",
    )
    inner_bytes = int(routed_wrapper_trace.get("inner_bytes", -1))
    require(type(inner_bytes) is int and inner_bytes > 0 and inner_bytes % PAGE_BYTES == 0, "ledger inner bytes")
    require(0 < head_bytes <= PAGE_BYTES - WRAPPER_FOOTER_BYTES, "ledger head bytes")
    require(int(routed_wrapper_trace.get("head_bytes", -1)) == head_bytes, "ledger wrapper head bytes")
    total_weights = sum(int(value) for value in weights_by_expert)
    require(total_weights > 0 and all(type(value) is int and value > 0 for value in weights_by_expert), "ledger weights")
    nonpadding_extension = head_bytes + WRAPPER_FOOTER_BYTES
    rows = []
    maximum_page = Fraction(0, 1)
    maximum_repeated = Fraction(0, 1)
    maximum_unique = Fraction(0, 1)
    all_second_pass_absent = True
    for expert in range(experts):
        trace = trace_rows[expert]
        require(isinstance(trace, Mapping) and int(trace.get("expert_ordinal", -1)) == expert, "wrapper trace expert")
        require(int(trace.get("extension_page_read_requests", -1)) == 1, "one suffix-page request")
        require(int(trace.get("inner_decode_invocations", -1)) == 1, "one inner routed decode invocation")
        require(trace.get("compressed_expert_second_pass_absent_derived") is True, "derived second-pass absence")
        require(trace.get("overlap_is_charged_not_interpreted_as_second_pass") is True, "overlap accounting boundary")
        causal = trace.get("causal_decode_reencode_reconstruction")
        require(
            isinstance(causal, Mapping)
            and causal.get("all_payloads_canonically_reencoded") is True
            and causal.get("all_three_roles_reconstructed") is True,
            "causal inner reconstruction proof",
        )
        alpha = Fraction(int(weights_by_expert[expert]), total_weights)
        attributed_total = Fraction(inner_attributed_total[expert]) + alpha * PAGE_BYTES
        attributed_nonpadding = Fraction(inner_attributed_nonpadding[expert]) + alpha * nonpadding_extension
        touched = int(trace["touched_page_bytes"])
        requested = int(trace["requested_bytes_with_repetition"])
        unique = int(trace["unique_requested_bytes"])
        request_count = int(trace["read_request_count"])
        require(touched >= unique >= 0 and requested >= unique and request_count > 0, "wrapper read accounting")
        page_total_amp = Fraction(touched, 1) / attributed_total
        page_nonpadding_amp = Fraction(touched, 1) / attributed_nonpadding
        repeated_total_amp = Fraction(requested, 1) / attributed_total
        repeated_nonpadding_amp = Fraction(requested, 1) / attributed_nonpadding
        unique_total_amp = Fraction(unique, 1) / attributed_total
        unique_nonpadding_amp = Fraction(unique, 1) / attributed_nonpadding
        page_strict = max(page_total_amp, page_nonpadding_amp)
        repeated_strict = max(repeated_total_amp, repeated_nonpadding_amp)
        unique_strict = max(unique_total_amp, unique_nonpadding_amp)
        strict = max(page_strict, repeated_strict, unique_strict)
        maximum_page = max(maximum_page, page_strict)
        maximum_repeated = max(maximum_repeated, repeated_strict)
        maximum_unique = max(maximum_unique, unique_strict)
        second_pass = bool(trace["compressed_expert_second_pass"])
        all_second_pass_absent = all_second_pass_absent and not second_pass
        rows.append({
            "expert_ordinal": expert,
            "source_weights": int(weights_by_expert[expert]),
            "extension_owner_fraction": fraction_record(alpha),
            "attributable_total_physical_bytes": fraction_record(attributed_total),
            "attributable_nonpadding_decodable_bytes": fraction_record(attributed_nonpadding),
            "touched_page_bytes": touched,
            "requested_bytes_with_repetition": requested,
            "unique_requested_bytes": unique,
            "read_request_count": request_count,
            "overlap_bytes_requested_again": int(trace["overlap_bytes_requested_again"]),
            "descriptor_page_amplification_total_physical": fraction_record(page_total_amp),
            "descriptor_page_amplification_nonpadding": fraction_record(page_nonpadding_amp),
            "requested_with_repetition_amplification_total_physical": fraction_record(repeated_total_amp),
            "requested_with_repetition_amplification_nonpadding": fraction_record(repeated_nonpadding_amp),
            "unique_requested_amplification_total_physical": fraction_record(unique_total_amp),
            "unique_requested_amplification_nonpadding": fraction_record(unique_nonpadding_amp),
            "cold_amplification_total_physical": fraction_record(max(page_total_amp, repeated_total_amp, unique_total_amp)),
            "cold_amplification_nonpadding": fraction_record(max(page_nonpadding_amp, repeated_nonpadding_amp, unique_nonpadding_amp)),
            "strict_cold_amplification": fraction_record(strict),
            "passes_descriptor_pages_below_2x": page_strict < 2,
            "passes_requested_with_repetition_below_2x": repeated_strict < 2,
            "passes_unique_requested_below_2x": unique_strict < 2,
            "compressed_expert_second_pass": second_pass,
            "compressed_expert_second_pass_absent_derived_from_instrumented_invocations": not second_pass,
            "inner_decode_invocations": int(trace["inner_decode_invocations"]),
            "actual_inner_routed_decode_proof_sha256": trace.get("actual_decode_proof_sha256", routed_wrapper_trace.get("proof_sha256")),
        })
    outer_bytes = inner_bytes + PAGE_BYTES
    rate = Fraction(8 * outer_bytes, total_weights)
    maximum = max(maximum_page, maximum_repeated, maximum_unique)
    passes_all = maximum < 2 and all_second_pass_absent
    return {
        "inner_bytes": inner_bytes,
        "head_bytes": head_bytes,
        "footer_bytes": WRAPPER_FOOTER_BYTES,
        "extension_nonpadding_bytes": nonpadding_extension,
        "extension_zero_padding_bytes": PAGE_BYTES - nonpadding_extension,
        "extension_physical_bytes": PAGE_BYTES,
        "outer_bytes": outer_bytes,
        "physical_rate_bpw": fraction_record(rate),
        "experts": rows,
        "maximum_descriptor_page_amplification_strict": fraction_record(maximum_page),
        "maximum_requested_with_repetition_amplification_strict": fraction_record(maximum_repeated),
        "maximum_unique_requested_amplification_strict": fraction_record(maximum_unique),
        "maximum_strict_cold_read_amplification": fraction_record(maximum),
        "passes_descriptor_pages_below_2x": maximum_page < 2,
        "passes_requested_with_repetition_below_2x": maximum_repeated < 2,
        "passes_unique_requested_below_2x": maximum_unique < 2,
        "passes_strict_cold_read_below_2x": passes_all,
        "additional_storage_pages_per_routed_expert": 1,
        "compressed_expert_second_pass_forbidden_and_absent": all_second_pass_absent,
        "actual_inner_routed_decode_executed": True,
        "actual_posterior_wrapper_routed_decode_executed": False,
        "posterior_head_applied_to_routed_reconstruction": False,
        "read_claim_is_nonpromoting_projection_from_instrumented_inner_decode_plus_literal_suffix": True,
        "scratch_hbm_is_not_counted_as_storage_read": True,
    }


def delta_s(*, baseline_rate: float, candidate_rate: float, baseline_distortion: float, candidate_distortion: float) -> float:
    require(
        all(math.isfinite(value) for value in (baseline_rate, candidate_rate, baseline_distortion, candidate_distortion))
        and baseline_distortion > 0.0
        and candidate_distortion > 0.0,
        "Delta-s inputs",
    )
    return (baseline_rate - candidate_rate) - 0.5 * math.log2(candidate_distortion / baseline_distortion)


def state_fold_gate(
    *,
    delta_s_value: float,
    g_state_value: float,
    candidate_rate_bpw: float,
    candidate_f: float,
    cold_read_below_2x: bool,
) -> dict[str, bool]:
    """One fail-closed fold gate shared by the runner and hostile tests."""

    finite = all(
        math.isfinite(value)
        for value in (delta_s_value, g_state_value, candidate_rate_bpw, candidate_f)
    )
    checks = {
        "passes_positive_Delta_s": finite and delta_s_value > 0.0,
        "passes_positive_G_state": finite and g_state_value > 0.0,
        "passes_rate_interval": finite and 2.15 <= candidate_rate_bpw <= 2.5,
        "passes_F_target": finite and candidate_f <= 0.8,
        "passes_cold_read_below_2x": cold_read_below_2x is True,
    }
    checks["passes_all_fold_gates"] = all(checks.values())
    return checks


require(HEAD_HEADER.size == HEAD_HEADER_BYTES, "internal head struct size")
require(WRAPPER_FOOTER.size == WRAPPER_FOOTER_BYTES, "internal wrapper struct size")
