#!/usr/bin/env python3
"""Deterministic source-free universal SwiGLU portability fixtures."""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from typing import Any, Sequence


def public_context_rows(symbols: int) -> tuple[list[int], list[int]]:
    if type(symbols) is not int or symbols <= 0:
        raise ValueError("fixture symbol count")
    levels = [position % 3 for position in range(symbols)]
    base = [1 if position % 97 == 0 else 65535 if position % 97 == 1 else 32768 for position in range(symbols)]
    return levels, base


def _bit_rows(ordinal: int, symbols: int) -> list[int]:
    return [((position * 29) ^ (position >> 2) ^ (ordinal * 17) ^ (position // 11)) & 1 for position in range(symbols)]


def _scalar_bytes(bit: int, ordinal: int, scalar_index: int) -> bytes:
    magnitude = 1.0 + ((ordinal * 13 + scalar_index * 7) % 31)
    value = magnitude / 31.0 if bit else -magnitude / 31.0
    return struct.pack("<d", value)


def _fixture_reconstruction(
    *,
    experts: int,
    matrix_weights: int,
    stream_specs: Sequence[tuple[bytes, Any]],
    references: Sequence[dict[str, Any]],
) -> str:
    matrices = {
        (expert, role): [None] * matrix_weights
        for expert in range(experts)
        for role in ("gate", "up", "down")
    }
    for (_owner_set, spec), reference in zip(stream_specs, references, strict=True):
        bits = reference["bits"]
        for contribution in spec.owner_contributions:
            target = matrices[(contribution.expert, contribution.role)]
            for local in range(contribution.weight_count):
                position = contribution.source_offset + local
                if target[position] is not None:
                    raise ValueError("fixture reconstruction overlap")
                target[position] = _scalar_bytes(bits[local % len(bits)], spec.ordinal, position)
    digest = hashlib.sha256()
    for expert in range(experts):
        for role in ("gate", "up", "down"):
            values = matrices[(expert, role)]
            if any(value is None for value in values):
                raise ValueError("fixture reconstruction hole")
            digest.update(b"".join(values))
    return digest.hexdigest()


def make_fixture(
    common: Any,
    codec: Any,
    semantic_codec: Any,
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    shared_groups: Sequence[tuple[str, tuple[tuple[int, int], ...]]] = (),
) -> dict[str, Any]:
    """Build exact role intervals, then encode source-free causal bit streams.

    Each shared group is (role, ((expert, tail_count), ...)). Tail counts may
    differ, exercising exact contribution weights rather than equal division.
    """
    if type(experts) is not int or not 1 <= experts <= 256:
        raise ValueError("fixture experts")
    shapes = tuple(semantic_codec.ExpertShape(index, hidden, intermediate) for index in range(experts))
    semantic_packet = semantic_codec.build_semantic_packet(shapes, b"source-free-universal-fixture-v3")
    matrix_weights = shapes[0].matrix_weights
    tail_by_key: dict[tuple[int, str], tuple[int, int]] = {}
    for group_index, (role, members) in enumerate(shared_groups):
        if role not in semantic_codec.ROLES or not isinstance(members, tuple) or len(members) < 2:
            raise ValueError("shared fixture group")
        previous = -1
        for expert, count in members:
            if type(expert) is not int or not 0 <= expert < experts or expert <= previous:
                raise ValueError("shared fixture owner ordering")
            if type(count) is not int or not 1 <= count < matrix_weights:
                raise ValueError("shared fixture tail count")
            if (expert, role) in tail_by_key:
                raise ValueError("duplicate shared role contribution")
            tail_by_key[(expert, role)] = (group_index, count)
            previous = expert
    raw_specs: list[dict[str, Any]] = []
    for expert in range(experts):
        for role in semantic_codec.ROLES:
            tail = tail_by_key.get((expert, role))
            prefix = matrix_weights if tail is None else matrix_weights - tail[1]
            raw_specs.append({
                "role": role,
                "contributions": ((expert, 0, prefix),),
                "group_weights": prefix,
            })
    for group_index, (role, members) in enumerate(shared_groups):
        contributions = tuple((expert, matrix_weights - count, count) for expert, count in members)
        raw_specs.append({
            "role": role,
            "contributions": contributions,
            "group_weights": sum(count for _expert, count in members),
        })
    candidate = common.Candidate("suffix", 2, 32)
    frequencies = [32768] * common.model_frequency_count(candidate)
    model_packet = common.serialize_model(candidate, frequencies)
    stream_specs = []
    streams_for_decode = []
    for ordinal, raw in enumerate(raw_specs):
        symbols = 33 + (ordinal % 17)
        levels, base = public_context_rows(symbols)
        bits = _bit_rows(ordinal, symbols)
        payload, logical_bits = common.encode_unifilar_stream(bits, levels, base, candidate, frequencies)
        contributions = tuple(codec.OwnerContribution(expert, raw["role"], offset, count) for expert, offset, count in raw["contributions"])
        owner_set = codec.owner_set_from_ordinals(experts, [row.expert for row in contributions])
        spec = codec.StreamSpec(
            ordinal=ordinal,
            symbols=symbols,
            logical_bits=logical_bits,
            payload=payload,
            source_digest=hashlib.sha256(bytes(bits)).hexdigest(),
            profile_q=0,
            decoder_scale=1.0,
            role=raw["role"],
            group_rows=1,
            group_cols=raw["group_weights"],
            owner_contributions=contributions,
        )
        stream_specs.append((owner_set, spec))
        streams_for_decode.append({"bits": bits, "levels": levels, "base": base})
    grouped: dict[bytes, list[Any]] = defaultdict(list)
    for owner_set, spec in stream_specs:
        grouped[owner_set].append(spec)
    ordered_owner_sets = sorted(grouped, key=lambda value: (len(codec.owner_ordinals(value, experts)) != 1, codec.owner_ordinals(value, experts)))
    regions = tuple(codec.RegionSpec(owner_set, tuple(sorted(grouped[owner_set], key=lambda row: row.ordinal))) for owner_set in ordered_owner_sets)
    reconstruction_sha256 = _fixture_reconstruction(
        experts=experts,
        matrix_weights=matrix_weights,
        stream_specs=stream_specs,
        references=streams_for_decode,
    )
    return {
        "experts": experts,
        "weights": experts * 3 * matrix_weights,
        "semantic_packet": semantic_packet,
        "model_packet": model_packet,
        "candidate": candidate,
        "frequencies": frequencies,
        "regions": regions,
        "reference_streams": tuple(streams_for_decode),
        "reconstruction_sha256": reconstruction_sha256,
    }


class FixtureRoutedDecoder:
    """Causal routed decoder plus deterministic Gate/Up/Down reconstruction."""

    def __init__(self, common: Any):
        self.common = common
        self._next_expert = 0
        self._full_digest = hashlib.sha256()

    def decode_expert(self, route: dict[str, Any]) -> dict[str, Any]:
        expert = int(route["expert_ordinal"])
        if expert != self._next_expert:
            raise ValueError("fixture routed expert order")
        matrix_weights = int(route["semantic_shape"].matrix_weights)
        matrices = {role: [None] * matrix_weights for role in ("gate", "up", "down")}
        for row in route["rows"]:
            levels, base = public_context_rows(int(row["symbols"]))
            bits = self.common.decode_unifilar_stream(
                bytes(row["payload"]), int(row["logical_bits"]), levels, base,
                route["candidate"], route["frequencies"],
            )
            if hashlib.sha256(bytes(bits)).hexdigest() != row["source_digest"]:
                raise ValueError("fixture routed decoded source digest")
            replay, logical = self.common.encode_unifilar_stream(
                bits, levels, base, route["candidate"], route["frequencies"]
            )
            if replay != row["payload"] or logical != row["logical_bits"]:
                raise ValueError("fixture routed noncanonical re-encode")
            for contribution in row["owner_contributions"]:
                if int(contribution["expert"]) != expert:
                    continue
                target = matrices[str(contribution["role"])]
                begin = int(contribution["source_offset"])
                count = int(contribution["weight_count"])
                for local in range(count):
                    position = begin + local
                    if target[position] is not None:
                        raise ValueError("fixture routed reconstruction overlap")
                    target[position] = _scalar_bytes(bits[local % len(bits)], int(row["ordinal"]), position)
        expert_digest = hashlib.sha256()
        for role in ("gate", "up", "down"):
            values = matrices[role]
            if any(value is None for value in values):
                raise ValueError("fixture routed reconstruction hole")
            packet = b"".join(values)
            expert_digest.update(packet)
            self._full_digest.update(packet)
        self._next_expert += 1
        return {
            "expert_ordinal": expert,
            "decoded_streams": len(route["rows"]),
            "all_payloads_canonically_reencoded": True,
            "all_three_roles_reconstructed": True,
            "routed_expert_reconstruction_sha256": expert_digest.hexdigest(),
        }

    def finalize(self, *, experts: int, expected_full_reconstruction_sha256: str) -> dict[str, Any]:
        observed = self._full_digest.hexdigest()
        return {
            "experts": experts,
            "full_reconstruction_f64_sha256": observed,
            "matches_container_reconstruction": self._next_expert == experts and observed == expected_full_reconstruction_sha256,
        }


def decode_and_reencode(common: Any, parsed: dict[str, Any]) -> dict[str, Any]:
    candidate, frequencies = common.deserialize_model(bytes(parsed["model_packet"]))
    digests = []
    for row in parsed["directory"]:
        levels, base = public_context_rows(int(row["symbols"]))
        bits = common.decode_unifilar_stream(bytes(row["payload"]), int(row["logical_bits"]), levels, base, candidate, frequencies)
        if hashlib.sha256(bytes(bits)).hexdigest() != row["source_digest"]:
            raise ValueError("fixture decoded source digest")
        payload, logical = common.encode_unifilar_stream(bits, levels, base, candidate, frequencies)
        if payload != row["payload"] or logical != row["logical_bits"]:
            raise ValueError("fixture noncanonical arithmetic re-encode")
        digests.append(row["source_digest"])
    return {
        "decoded_streams": len(digests),
        "all_payloads_canonically_reencoded": True,
        "stream_source_digest_sha256": hashlib.sha256("".join(digests).encode("ascii")).hexdigest(),
    }
