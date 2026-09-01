"""Synthetic mini-STRATA integration fixture for the independent bridge.

No checkpoint, existing container, selected-bit dump, or Gaussian-control
artifact is read.  Original SC frequencies are regenerated causally from the
literal metadata seed and earlier decoded decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Sequence

from uwfa_bridge import (
    ARITH_TOTAL,
    BLOCK_COUNT,
    BinaryArithmeticDecoder,
    BinaryArithmeticEncoder,
    BlockSpec,
    HeaderBindings,
    ParsedContainer,
    PhysicalScore,
    ReadLedger,
    UWFADecoderAdapter,
    UWFAEncoderAdapter,
    UWFAModel,
    build_container,
    build_frame,
    decision_hash,
    parse_container,
    physical_score,
    routed_read_ledger,
    sha256,
)


MINI_METADATA_MAGIC = b"MSTRAT2\x00"
SEMANTIC_HEADER_BYTES = 128
ROUTE_TABLE_BYTES = 144
THREE_BIT_LABEL_BYTES = 5184
PROFILE_BYTES = BLOCK_COUNT
SCALE_BYTES = 2 * BLOCK_COUNT
MINI_METADATA_BYTES = (
    SEMANTIC_HEADER_BYTES
    + ROUTE_TABLE_BYTES
    + THREE_BIT_LABEL_BYTES
    + PROFILE_BYTES
    + SCALE_BYTES
)


@dataclass(frozen=True)
class MiniMetadata:
    source_weights: int
    expert_count: int
    role_count: int
    level_count: int
    seeds: tuple[int, ...]
    log2ns: tuple[int, ...]
    owner_masks: tuple[int, ...]
    roles: tuple[int, ...]
    profiles: tuple[int, ...]
    scale_bits: tuple[int, ...]


def _hash_stream(label: bytes, length: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < length:
        output.extend(hashlib.sha256(label + struct.pack("<I", counter)).digest())
        counter += 1
    return bytes(output[:length])


def build_mini_metadata(
    *,
    source_weights: int,
    expert_count: int = 6,
    role_count: int = 3,
    level_count: int = 4,
    log2n: int = 16,
) -> tuple[bytes, MiniMetadata]:
    if expert_count != 6:
        raise ValueError("the synthetic 15-block route fixture uses six experts")
    seeds = tuple(
        struct.unpack("<I", hashlib.sha256(b"mini-seed" + bytes([index])).digest()[:4])[0]
        for index in range(BLOCK_COUNT)
    )
    log2ns = (log2n,) * BLOCK_COUNT
    owner_masks = tuple(
        (1 << (index // 2)) if index < 12 else (3 << (2 * (index - 12)))
        for index in range(BLOCK_COUNT)
    )
    roles = tuple(index % role_count for index in range(BLOCK_COUNT))
    profiles = tuple((index * 7 + 3) & 0xFF for index in range(BLOCK_COUNT))
    scale_bits = tuple(
        struct.unpack("<H", struct.pack("<e", 0.125 + index / 64.0))[0]
        for index in range(BLOCK_COUNT)
    )

    semantic = bytearray(SEMANTIC_HEADER_BYTES)
    semantic[0:8] = MINI_METADATA_MAGIC
    struct.pack_into("<HHI", semantic, 8, 1, 0, SEMANTIC_HEADER_BYTES)
    struct.pack_into("<Q", semantic, 16, source_weights)
    struct.pack_into("<HHHH", semantic, 24, expert_count, BLOCK_COUNT, role_count, level_count)
    struct.pack_into("<" + "I" * BLOCK_COUNT, semantic, 32, *seeds)
    semantic[92:107] = bytes(log2ns)

    routes = bytearray(ROUTE_TABLE_BYTES)
    struct.pack_into("<" + "Q" * BLOCK_COUNT, routes, 0, *owner_masks)
    routes[120:135] = bytes(roles)
    labels = _hash_stream(b"literal-three-bit-label-packet", THREE_BIT_LABEL_BYTES)
    metadata = bytes(semantic + routes + labels + bytes(profiles)) + struct.pack(
        "<" + "H" * BLOCK_COUNT, *scale_bits
    )
    parsed = MiniMetadata(
        source_weights,
        expert_count,
        role_count,
        level_count,
        seeds,
        log2ns,
        owner_masks,
        roles,
        profiles,
        scale_bits,
    )
    return metadata, parsed


def parse_mini_metadata(data: bytes) -> MiniMetadata:
    if len(data) != MINI_METADATA_BYTES or data[0:8] != MINI_METADATA_MAGIC:
        raise ValueError("bad synthetic inherited metadata packet")
    major, minor, header_bytes = struct.unpack_from("<HHI", data, 8)
    if (major, minor, header_bytes) != (1, 0, SEMANTIC_HEADER_BYTES):
        raise ValueError("bad synthetic metadata version")
    source_weights = struct.unpack_from("<Q", data, 16)[0]
    expert_count, block_count, role_count, level_count = struct.unpack_from("<HHHH", data, 24)
    if block_count != BLOCK_COUNT:
        raise ValueError("synthetic metadata block count mismatch")
    seeds = struct.unpack_from("<" + "I" * BLOCK_COUNT, data, 32)
    log2ns = tuple(data[92:107])
    if any(data[107:SEMANTIC_HEADER_BYTES]):
        raise ValueError("nonzero synthetic semantic-header reserved bytes")
    route_start = SEMANTIC_HEADER_BYTES
    owner_masks = struct.unpack_from("<" + "Q" * BLOCK_COUNT, data, route_start)
    roles = tuple(data[route_start + 120 : route_start + 135])
    if any(data[route_start + 135 : route_start + ROUTE_TABLE_BYTES]):
        raise ValueError("nonzero synthetic route-table reserved bytes")
    profile_start = SEMANTIC_HEADER_BYTES + ROUTE_TABLE_BYTES + THREE_BIT_LABEL_BYTES
    profiles = tuple(data[profile_start : profile_start + PROFILE_BYTES])
    scale_bits = struct.unpack_from("<" + "H" * BLOCK_COUNT, data, profile_start + PROFILE_BYTES)
    return MiniMetadata(
        source_weights,
        expert_count,
        role_count,
        level_count,
        tuple(seeds),
        log2ns,
        tuple(owner_masks),
        roles,
        profiles,
        tuple(scale_bits),
    )


def make_synthetic_model(
    state_count: int = 4,
    reset_length: int = 8,
    level_count: int = 4,
    prior_bin_count: int = 8,
) -> UWFAModel:
    frequencies: list[int] = []
    for level in range(level_count):
        for prior_bin in range(prior_bin_count):
            for position in range(reset_length):
                for state in range(state_count):
                    mixed = (
                        level * 1009
                        + prior_bin * 9176
                        + position * 6113
                        + state * 3571
                        + 0xA51D
                    ) & 0xFFFFFFFF
                    mixed ^= mixed >> 16
                    frequencies.append(22768 + mixed % 20001)
    return UWFAModel(
        total=ARITH_TOTAL,
        state_count=state_count,
        reset_length=reset_length,
        level_count=level_count,
        prior_bin_count=prior_bin_count,
        topology_id=1,
        frequencies=tuple(frequencies),
    )


def _mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF


def regenerated_original_frequency(
    seed: int,
    ordinal: int,
    level: int,
    local_index: int,
    history: int,
) -> int:
    mixed = _mix32(
        seed
        ^ (ordinal * 0x9E3779B9)
        ^ (level * 0x85EBCA6B)
        ^ (local_index * 0xC2B2AE35)
        ^ history
    )
    return 1 + mixed % (ARITH_TOTAL - 1)


def _history_update(history: int, bit: int, frequency: int, index: int) -> int:
    return _mix32(history ^ (bit * 0xA5A5A5A5) ^ frequency ^ (index * 0x27D4EB2D))


def synthetic_decisions(seed: int, ordinal: int, count: int) -> tuple[int, ...]:
    return tuple((_mix32(seed ^ (ordinal << 24) ^ index) >> 31) & 1 for index in range(count))


def encode_sc_loop(
    model: UWFAModel,
    seed: int,
    ordinal: int,
    decisions: Sequence[int],
) -> tuple[bytes, int, bytes]:
    arithmetic = BinaryArithmeticEncoder(model.total)
    adapter = UWFAEncoderAdapter(model, arithmetic)
    context_hasher = hashlib.sha256()
    history = _mix32(seed ^ ordinal)
    cursor = 0
    for level in range(model.level_count):
        adapter.set_level(level)
        count = len(decisions) // model.level_count + (level < len(decisions) % model.level_count)
        for local_index in range(count):
            frequency = regenerated_original_frequency(seed, ordinal, level, local_index, history)
            context_hasher.update(struct.pack("<H", frequency))
            bit = decisions[cursor]
            adapter.encode(bit, frequency)
            history = _history_update(history, bit, frequency, cursor)
            cursor += 1
    if cursor != len(decisions):
        raise AssertionError("SC encoder did not consume each decision exactly once")
    payload, logical_bits = arithmetic.finish()
    return payload, logical_bits, context_hasher.digest()


def decode_sc_loop(
    model: UWFAModel,
    seed: int,
    ordinal: int,
    decision_count: int,
    payload: bytes,
    logical_bits: int,
) -> tuple[tuple[int, ...], bytes]:
    arithmetic = BinaryArithmeticDecoder(payload, logical_bits, model.total)
    adapter = UWFADecoderAdapter(model, arithmetic)
    decisions: list[int] = []
    context_hasher = hashlib.sha256()
    history = _mix32(seed ^ ordinal)
    for level in range(model.level_count):
        adapter.set_level(level)
        count = decision_count // model.level_count + (level < decision_count % model.level_count)
        for local_index in range(count):
            frequency = regenerated_original_frequency(seed, ordinal, level, local_index, history)
            context_hasher.update(struct.pack("<H", frequency))
            bit = adapter.decode(frequency)
            decisions.append(bit)
            history = _history_update(history, bit, frequency, len(decisions) - 1)
    if len(decisions) != decision_count:
        raise AssertionError("SC decoder did not restore each decision exactly once")
    return tuple(decisions), context_hasher.digest()


def reconstruction_hash(block_hashes: Sequence[bytes]) -> bytes:
    return sha256(b"MINI-STRATA-RECONSTRUCTION-v1" + b"".join(block_hashes))


def synthetic_bindings(decoded_hash: bytes) -> HeaderBindings:
    return HeaderBindings(
        baseline_container_bytes=123456789,
        baseline_relative_mse=0.03,
        energy_convention=1,
        baseline_container_hash=sha256(b"synthetic baseline container"),
        baseline_plan_lock_hash=sha256(b"synthetic baseline plan lock"),
        baseline_audit_hash=sha256(b"synthetic independent audit"),
        universal_decoder_hash=sha256(b"synthetic universal decoder source"),
        source_manifest_hash=sha256(b"synthetic immutable source manifest"),
        audit_bootstrap_hash=sha256(b"synthetic external audit bootstrap"),
        decoded_reconstruction_hash=decoded_hash,
    )


def build_synthetic_container(log2n: int = 16) -> bytes:
    source_weights = BLOCK_COUNT * (1 << log2n)
    metadata_bytes, metadata = build_mini_metadata(source_weights=source_weights, log2n=log2n)
    model = make_synthetic_model(level_count=metadata.level_count)
    blocks: list[BlockSpec] = []
    block_hashes: list[bytes] = []
    for ordinal in range(BLOCK_COUNT):
        decisions = synthetic_decisions(metadata.seeds[ordinal], ordinal, 1 << metadata.log2ns[ordinal])
        encoded, logical_bits, _ = encode_sc_loop(
            model, metadata.seeds[ordinal], ordinal, decisions
        )
        decisions_digest = decision_hash(decisions)
        block_hashes.append(decisions_digest)
        blocks.append(
            BlockSpec(
                log2n=metadata.log2ns[ordinal],
                role=metadata.roles[ordinal],
                owner_mask=metadata.owner_masks[ordinal],
                profile_id=metadata.profiles[ordinal],
                scale_bits=metadata.scale_bits[ordinal],
                frame=build_frame(ordinal, len(decisions), encoded, logical_bits),
                decisions_hash=decisions_digest,
            )
        )
    return build_container(
        metadata=metadata_bytes,
        model=model,
        blocks=blocks,
        source_weights=source_weights,
        expert_count=metadata.expert_count,
        role_count=metadata.role_count,
        bindings=synthetic_bindings(reconstruction_hash(block_hashes)),
    )


@dataclass(frozen=True)
class SyntheticVerification:
    parsed: ParsedContainer
    score: PhysicalScore
    ledger: ReadLedger
    context_hashes: tuple[bytes, ...]
    decoded_hash: bytes


def verify_synthetic_container(raw: bytes) -> SyntheticVerification:
    parsed = parse_container(raw)
    metadata = parse_mini_metadata(parsed.metadata)
    header = parsed.header
    if (
        metadata.source_weights != header.source_weights
        or metadata.expert_count != header.expert_count
        or metadata.role_count != header.role_count
        or metadata.level_count != header.level_count
    ):
        raise ValueError("literal metadata/global header mismatch")
    block_hashes: list[bytes] = []
    context_hashes: list[bytes] = []
    for ordinal, (record, frame) in enumerate(zip(parsed.records, parsed.frames)):
        if (
            record.log2n != metadata.log2ns[ordinal]
            or record.owner_mask != metadata.owner_masks[ordinal]
            or record.role != metadata.roles[ordinal]
            or record.profile_id != metadata.profiles[ordinal]
            or record.scale_bits != metadata.scale_bits[ordinal]
            or record.decision_count != 1 << metadata.log2ns[ordinal]
        ):
            raise ValueError("literal metadata/directory mismatch")
        decisions, decoded_context_hash = decode_sc_loop(
            parsed.model,
            metadata.seeds[ordinal],
            ordinal,
            record.decision_count,
            frame.encoded_bytes,
            frame.logical_bits,
        )
        digest = decision_hash(decisions)
        if digest != record.decisions_hash:
            raise ValueError("decoded decision hash mismatch")
        reencoded, reencoded_bits, reencoded_context_hash = encode_sc_loop(
            parsed.model, metadata.seeds[ordinal], ordinal, decisions
        )
        if reencoded != frame.encoded_bytes or reencoded_bits != frame.logical_bits:
            raise ValueError("canonical arithmetic re-encode mismatch")
        if reencoded_context_hash != decoded_context_hash:
            raise ValueError("SC context regeneration mismatch")
        block_hashes.append(digest)
        context_hashes.append(decoded_context_hash)
    decoded_hash = reconstruction_hash(block_hashes)
    if decoded_hash != header.bindings.decoded_reconstruction_hash:
        raise ValueError("synthetic reconstruction hash mismatch")
    score = physical_score(len(raw), header.source_weights, header.bindings.baseline_relative_mse)
    ledger = routed_read_ledger(parsed)
    return SyntheticVerification(parsed, score, ledger, tuple(context_hashes), decoded_hash)

