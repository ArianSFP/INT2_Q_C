#!/usr/bin/env python3
"""Read-only authenticated six-level STRATA replay adapter.

The real STRATA codec does *not* emit six independent arithmetic events per
weight.  It decodes six full polar levels.  Selected internal SC decisions are
level-major and each polar transform couples many output coordinates.  The six
output bitplanes are then assembled into one index in ``0..63`` per weight.

This adapter authenticates that real relationship and the current arithmetic
state replay.  It deliberately exposes no coordinate-local candidate API.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


INTERFACE = "strata_sc_six_level_block_replay"
LEGACY_INVALID_INTERFACE = "strata_sc_6bit_legal_replay"
LEVELS = 6
ALPHABET = 64
AUDITOR_BYTES = 116_835
AUDITOR_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"
REPLAY_SCHEMA = "epsilon-tcq-strata-six-level-replay-receipt-v1"


class AdapterError(RuntimeError):
    pass


class CoordinateLocalTransitionHold(AdapterError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def _sha(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(value: Any, label: str) -> str:
    require(type(value) is str and len(value) == 64, f"{label} digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise AdapterError(f"{label} digest") from error
    require(value == value.lower(), f"{label} lowercase")
    return value


def _unpack_msb(payload: bytes, logical_bits: int) -> tuple[int, ...]:
    require(type(payload) is bytes and type(logical_bits) is int and
            0 <= logical_bits <= 8 * len(payload), "packed-bit geometry")
    if logical_bits & 7:
        require(not (payload[-1] & ((1 << (8 - (logical_bits & 7))) - 1)),
                "canonical zero terminal bits")
    return tuple((payload[index >> 3] >> (7 - (index & 7))) & 1
                 for index in range(logical_bits))


def arithmetic_encode_binary(bits: Sequence[int], frequencies: Sequence[int]) -> tuple[bytes, int]:
    """Canonical current-codec 32-bit/Q0.16 binary arithmetic replay."""

    require(len(bits) == len(frequencies), "arithmetic event geometry")
    full, half, quarter = 1 << 32, 1 << 31, 1 << 30
    three_quarters = 3 << 30
    low, high, pending = 0, full - 1, 0
    output: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        output.append(bit)
        if pending:
            output.extend([1 - bit] * pending)
            pending = 0

    for raw_bit, raw_frequency in zip(bits, frequencies, strict=True):
        bit, frequency = int(raw_bit), int(raw_frequency)
        require(bit in (0, 1) and 1 <= frequency <= 65_535,
                "arithmetic event value")
        width = high - low + 1
        split = low + width * (65_536 - frequency) // 65_536 - 1
        require(low <= split < high, "arithmetic nonempty split")
        if bit == 0:
            high = split
        else:
            low = split + 1
        while True:
            if high < half:
                emit(0)
            elif low >= half:
                emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low = (low << 1) & (full - 1)
            high = ((high << 1) & (full - 1)) | 1
    pending += 1
    emit(0 if low < quarter else 1)
    packed = bytearray((len(output) + 7) // 8)
    for index, bit in enumerate(output):
        packed[index >> 3] |= bit << (7 - (index & 7))
    return bytes(packed), len(output)


@dataclass(frozen=True)
class LevelArtifacts:
    selected_bits_msb: bytes
    selected_count: int
    causal_frequencies_u16le: bytes
    selected_mask_msb: bytes
    internal_sc_bits_msb: bytes
    output_plane_msb: bytes

    def events(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        bits = _unpack_msb(self.selected_bits_msb, self.selected_count)
        require(len(self.causal_frequencies_u16le) == 2 * self.selected_count,
                "causal frequency geometry")
        frequencies = struct.unpack(
            "<" + "H" * self.selected_count,
            self.causal_frequencies_u16le) if self.selected_count else ()
        require(all(1 <= int(value) <= 65_535 for value in frequencies),
                "causal frequency range")
        return bits, tuple(int(value) for value in frequencies)


@dataclass(frozen=True)
class ReplayArtifacts:
    decoder_source: bytes
    current_packet: bytes
    independently_reencoded_current_packet: bytes
    payload: bytes
    indices_u8: bytes
    primary_reconstruction_f64le: bytes
    independent_reconstruction_f64le: bytes
    levels: tuple[LevelArtifacts, ...]


class ReadOnlyStrataReplayAdapter:
    interface = INTERFACE
    labels = ALPHABET
    sc_levels = LEVELS
    coordinate_local_arithmetic_events = False
    direct_int2_fallback = False
    read_only = True

    def coordinate_choices(self, *_args: Any, **_kwargs: Any) -> tuple[()]:
        raise CoordinateLocalTransitionHold(
            "HOLD_COORDINATE_LOCAL_EPSILON_INVALID_FOR_LEVEL_MAJOR_POLAR_SC")

    def validate(
        self,
        receipt: Mapping[str, Any],
        artifacts: ReplayArtifacts,
        *,
        expected_decoder_bytes: int = AUDITOR_BYTES,
        expected_decoder_sha256: str = AUDITOR_SHA256,
        allow_source_free_fixture_pin: bool = False,
    ) -> dict[str, Any]:
        keys = {
            "schema", "status", "interface", "decoder_bytes", "decoder_sha256",
            "current_packet_bytes", "current_packet_sha256", "block_ordinal",
            "block_values", "block_log2", "logical_bits", "payload_sha256",
            "indices_u8_sha256", "reconstruction_f64le_sha256", "levels",
            "event_stream_sha256", "seal_sha256",
        }
        require(isinstance(receipt, Mapping) and set(receipt) == keys,
                "replay receipt exact schema")
        body = dict(receipt)
        seal = _digest(body.pop("seal_sha256"), "replay seal")
        require(_sha(_canonical_json(body)) == seal, "replay receipt seal")
        require(receipt["schema"] == REPLAY_SCHEMA and
                receipt["status"] == "INDEPENDENT_CURRENT_CODEC_SIX_LEVEL_REPLAY" and
                receipt["interface"] == INTERFACE, "replay identity")
        require(LEGACY_INVALID_INTERFACE not in _canonical_json(receipt).decode("ascii"),
                "legacy coordinate-local ABI forbidden")
        if not allow_source_free_fixture_pin:
            require(expected_decoder_bytes == AUDITOR_BYTES and
                    expected_decoder_sha256 == AUDITOR_SHA256,
                    "production decoder pin is immutable")
        require(type(expected_decoder_bytes) is int and expected_decoder_bytes > 0 and
                type(expected_decoder_sha256) is str and
                len(expected_decoder_sha256) == 64, "decoder pin")
        require(len(artifacts.decoder_source) == expected_decoder_bytes and
                _sha(artifacts.decoder_source) == expected_decoder_sha256 and
                receipt["decoder_bytes"] == expected_decoder_bytes and
                receipt["decoder_sha256"] == expected_decoder_sha256,
                "authenticated decoder source")
        require(type(receipt["block_ordinal"]) is int and
                receipt["block_ordinal"] >= 0, "block ordinal")
        n = receipt["block_values"]
        require(type(n) is int and n >= 2 and n & (n - 1) == 0 and
                receipt["block_log2"] == int(math.log2(n)), "polar block geometry")
        require(len(artifacts.current_packet) == receipt["current_packet_bytes"] and
                _sha(artifacts.current_packet) == receipt["current_packet_sha256"] and
                artifacts.independently_reencoded_current_packet == artifacts.current_packet,
                "literal current packet independent reencode")
        require(len(artifacts.levels) == LEVELS and
                isinstance(receipt["levels"], list) and
                len(receipt["levels"]) == LEVELS, "exact six-level replay")
        require(len(artifacts.indices_u8) == n and
                all(value < ALPHABET for value in artifacts.indices_u8),
                "64-index vector")
        plane_bytes = (n + 7) // 8
        reconstructed_indices = bytearray(n)
        all_bits: list[int] = []
        all_frequencies: list[int] = []
        event_digest_material = bytearray()
        for level, (row, level_artifacts) in enumerate(
                zip(receipt["levels"], artifacts.levels, strict=True)):
            row_keys = {
                "level", "selected_count", "selected_bits_msb_sha256",
                "causal_frequencies_u16le_sha256", "selected_mask_msb_sha256",
                "internal_sc_bits_msb_sha256", "output_plane_msb_sha256",
            }
            require(isinstance(row, Mapping) and set(row) == row_keys and
                    row["level"] == level, "level receipt schema/order")
            require(len(level_artifacts.output_plane_msb) == plane_bytes and
                    len(level_artifacts.selected_mask_msb) == plane_bytes and
                    len(level_artifacts.internal_sc_bits_msb) == plane_bytes,
                    "polar level packed bytes")
            output_bits = _unpack_msb(level_artifacts.output_plane_msb, n)
            selected_mask = _unpack_msb(level_artifacts.selected_mask_msb, n)
            internal_bits = _unpack_msb(level_artifacts.internal_sc_bits_msb, n)
            bits, frequencies = level_artifacts.events()
            require(row["selected_count"] == level_artifacts.selected_count and
                    row["selected_count"] == len(bits), "selected event count")
            require(row["selected_bits_msb_sha256"] ==
                    _sha(level_artifacts.selected_bits_msb) and
                    row["causal_frequencies_u16le_sha256"] ==
                    _sha(level_artifacts.causal_frequencies_u16le) and
                    row["selected_mask_msb_sha256"] ==
                    _sha(level_artifacts.selected_mask_msb) and
                    row["internal_sc_bits_msb_sha256"] ==
                    _sha(level_artifacts.internal_sc_bits_msb) and
                    row["output_plane_msb_sha256"] ==
                    _sha(level_artifacts.output_plane_msb), "level artifact hashes")
            require(tuple(value for value, selected in zip(
                        internal_bits, selected_mask, strict=True) if selected) == bits,
                    "selected events equal selected internal SC decisions")
            depth = int(math.log2(n))
            reverse = []
            for position in range(n):
                source, value = position, 0
                for _ in range(depth):
                    value = (value << 1) | (source & 1)
                    source >>= 1
                reverse.append(value)
            polar = [internal_bits[index] for index in reverse]
            stride = 1
            while stride < n:
                for base in range(0, n, 2 * stride):
                    for offset in range(stride):
                        polar[base + offset] ^= polar[base + stride + offset]
                stride *= 2
            require(tuple(polar) == output_bits,
                    "internal SC state reproduces polar output bitplane")
            for position, bit in enumerate(output_bits):
                reconstructed_indices[position] |= bit << level
            all_bits.extend(bits)
            all_frequencies.extend(frequencies)
            event_digest_material.extend(struct.pack("<Q", len(bits)))
            event_digest_material.extend(level_artifacts.selected_bits_msb)
            event_digest_material.extend(level_artifacts.causal_frequencies_u16le)
            event_digest_material.extend(level_artifacts.selected_mask_msb)
            event_digest_material.extend(level_artifacts.internal_sc_bits_msb)
        require(bytes(reconstructed_indices) == artifacts.indices_u8 and
                receipt["indices_u8_sha256"] == _sha(artifacts.indices_u8),
                "six output planes assemble exact 64-index vector")
        canonical_payload, logical_bits = arithmetic_encode_binary(
            all_bits, all_frequencies)
        require(logical_bits == receipt["logical_bits"] and
                canonical_payload == artifacts.payload and
                receipt["payload_sha256"] == _sha(artifacts.payload),
                "literal arithmetic state replay")
        require(receipt["event_stream_sha256"] == _sha(bytes(event_digest_material)),
                "event stream aggregate")
        require(len(artifacts.primary_reconstruction_f64le) == 8 * n and
                artifacts.primary_reconstruction_f64le ==
                artifacts.independent_reconstruction_f64le and
                receipt["reconstruction_f64le_sha256"] ==
                _sha(artifacts.primary_reconstruction_f64le),
                "exact current-codec reconstruction cross-replay")
        for (value,) in struct.iter_unpack("<d", artifacts.primary_reconstruction_f64le):
            require(math.isfinite(value), "finite replay reconstruction")
        return {
            "schema": "epsilon-tcq-strata-read-only-adapter-validation-v1",
            "status": "PASS_EXACT_CURRENT_PATH_READ_ONLY_REPLAY",
            "interface": INTERFACE,
            "block_values": n,
            "sc_levels": LEVELS,
            "selected_sc_events": len(all_bits),
            "current_packet_sha256": receipt["current_packet_sha256"],
            "indices_u8_sha256": receipt["indices_u8_sha256"],
            "reconstruction_f64le_sha256": receipt["reconstruction_f64le_sha256"],
            "literal_current_packet_reencode_matches": True,
            "coordinate_local_choice_api_available": False,
            "direct_int2_fallback_available": False,
        }


def seal_replay_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    require("seal_sha256" not in body, "unsealed replay body")
    output = dict(body)
    output["seal_sha256"] = _sha(_canonical_json(output))
    return output
