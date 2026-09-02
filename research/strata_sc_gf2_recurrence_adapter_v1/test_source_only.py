#!/usr/bin/env python3
"""Source-only tests. No Qwen, STRATA artifact, control, or publication payload."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import struct
import sys
import unittest
import zlib
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import independent_scorer as scorer
import prepayload_rate_gate
import publication_gate as gate
import replay_gate
import strata_recurrence_codec as codec


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def owner_set(*experts: int) -> str:
    value = bytearray(16)
    for expert in experts:
        value[expert >> 3] |= 1 << (expert & 7)
    return value.hex()


def deterministic_bits(seed: int, count: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.getrandbits(1) for _ in range(count))


def six_level_bits(per_level: int, *, kind: str, seed: int) -> tuple[bytes, bytes]:
    parts = []
    for level in range(codec.LEVELS):
        if kind == "lfsr":
            initial = bytes(((seed + level + index) >> (index & 3)) & 1 for index in range(7))
            part = codec.generate_lfsr(initial, 1 | (1 << 1) | (1 << 7), per_level)
        elif kind == "zero":
            part = bytes(per_level)
        elif kind == "random":
            part = deterministic_bits(seed + 1009 * level, per_level)
        else:
            raise AssertionError(kind)
        parts.append(part)
    return b"".join(parts), b"".join(bytes((level,)) * per_level for level in range(codec.LEVELS))


def make_stream(
    ordinal: int, expert: int, role: str, matrix_weights: int,
    bits: bytes, levels: bytes, *, candidate_sha: str | None = None,
) -> codec.StreamSource:
    base = b"".join(struct.pack("<H", 8192 + ((index * 7919) % 49151)) for index in range(len(bits)))
    baseline_payload = codec.pack_bits(bits)
    candidate_payload = baseline_payload[::-1] if candidate_sha is None else b"candidate-physical-placeholder"
    candidate_digest = digest(candidate_payload) if candidate_sha is None else candidate_sha
    contribution = ({
        "expert": expert, "role": role, "source_offset": 0,
        "weight_count": matrix_weights,
    },)
    return codec.StreamSource(
        ordinal=ordinal, role=role, owner_set_hex=owner_set(expert),
        owner_contributions=contribution, local_source_offset=0,
        local_weight_count=matrix_weights, global_source_weights=matrix_weights,
        profile_q=1234 + ordinal, decoder_scale_f16le=struct.pack("<e", 1.25),
        logn=12, sc_seed_u32=17 + ordinal, rht_seed_u64=91 + ordinal,
        state=b"state\x00" + ordinal.to_bytes(4, "little"),
        selected_bits=bits, levels=levels,
        selected_sha256=digest(bits), levels_sha256=digest(levels),
        base_frequencies_u16le_sha256=digest(base),
        decoded_triplet_sha256=digest(bits + levels + base),
        baseline_payload_bytes=len(baseline_payload),
        baseline_logical_bits=len(bits), baseline_payload_sha256=digest(baseline_payload),
        candidate_payload_bytes=len(candidate_payload),
        candidate_logical_bits=min(len(bits), 8 * len(candidate_payload)),
        candidate_payload_sha256=candidate_digest,
    )


def streams_for_expert(
    expert: int, matrix_weights: int, bits: bytes, levels: bytes,
    *, ordinal_begin: int = 0,
) -> tuple[codec.StreamSource, ...]:
    return tuple(
        make_stream(ordinal_begin + index, expert, role, matrix_weights, bits, levels)
        for index, role in enumerate(("gate", "up", "down_transposed"))
    )


def brute_complexity(sequence: bytes) -> int:
    for order in range(len(sequence) + 1):
        for coefficients in range(1 << order):
            okay = True
            for index in range(order, len(sequence)):
                value = 0
                for lag in range(1, order + 1):
                    value ^= ((coefficients >> (lag - 1)) & 1) & sequence[index - lag]
                if value != sequence[index]:
                    okay = False
                    break
            if okay:
                return order
    raise AssertionError("no recurrence")


def reseal_expert(packet: bytes, body: bytes, *, metadata_changed: bool = False) -> bytes:
    fields = list(codec.EXPERT_CORE.unpack_from(packet, 0))
    if metadata_changed:
        metadata_offset, metadata_bytes = fields[8], fields[9]
        fields[15] = bytes.fromhex(digest(body[
            metadata_offset - codec.EXPERT_HEADER_BYTES:
            metadata_offset - codec.EXPERT_HEADER_BYTES + metadata_bytes
        ]))
    fields[16] = bytes.fromhex(digest(body))
    fields[17] = 0
    zero_core = codec.EXPERT_CORE.pack(*fields)
    zero_header = zero_core + bytes(codec.EXPERT_HEADER_BYTES - len(zero_core))
    fields[17] = zlib.crc32(zero_header + body) & 0xFFFFFFFF
    core = codec.EXPERT_CORE.pack(*fields)
    return core + bytes(codec.EXPERT_HEADER_BYTES - len(core)) + body


def fake_v9_receipt(decision_commitment: str, *, f_value: float = 0.99) -> bytes:
    physical = {
        "candidate_sha256": gate.CANDIDATE_SHA256,
        "identity_sha256": "1" * 64,
        "model_packet_sha256": "2" * 64,
        "directory_sha256": "3" * 64,
        "identity_directory_sha256": "4" * 64,
        "decision_commitment_sha256": decision_commitment,
        "reconstruction_sha256": gate.RECONSTRUCTION_SHA256,
        "rate": {
            "numerator": (8 * gate.CANDIDATE_BYTES) // math.gcd(8 * gate.CANDIDATE_BYTES, gate.SOURCE_WEIGHTS),
            "denominator": gate.SOURCE_WEIGHTS // math.gcd(8 * gate.CANDIDATE_BYTES, gate.SOURCE_WEIGHTS),
            "float": float(Fraction(8 * gate.CANDIDATE_BYTES, gate.SOURCE_WEIGHTS)),
        },
        "F": f_value, "physical_pass": False, "cold_pass": True,
        "bandwidth": {}, "independent_container_parser": True,
        "independent_byte_ledger_entries": [],
        "independent_selected_stream_causal_replay": True,
        "identity_rate": {}, "identity_semantic_decode": {},
    }
    return canonical({
        "schema": gate.AUDIT_SCHEMA, "status": gate.AUDIT_STATUS,
        "positive_claim_authority": False,
        "controls_run_by_this_audit": False,
        "shuffles_run_by_this_audit": False,
        "coordinate_diagnostic_run_by_this_audit": False,
        "primary_result_status": "SYNTHETIC_SOURCE_ONLY_NONPROMOTING",
        "publication_members": {
            name: {"bytes": size, "sha256": member_sha}
            for name, (size, member_sha) in gate.EXPECTED_PUBLICATION.items()
        },
        "source_closure": gate.EXPECTED_SOURCE_CLOSURE,
        "scientific_replay": {
            "selected_q016_cpu_replay": {
                "status": "PASS_EXACT_SELECTED_CELL_Q016_CPU_REPLAY"
            }
        },
        "runtime_replay": {}, "literal_container_audit": physical,
        "evidence_limitations": [],
    })


def fake_replay_receipt(
    authority, catalog: bytes, packets: tuple[bytes, ...], commitment: str,
    unique_selected: int,
) -> bytes:
    decoded = [codec.decode_expert(packet) for packet in packets]
    packet_set_record = [{
        "expert_ordinal": row["expert_ordinal"],
        "packet_bytes": len(packet), "packet_sha256": digest(packet),
    } for packet, row in zip(packets, decoded, strict=True)]
    return canonical({
        "schema": replay_gate.SCHEMA, "status": replay_gate.STATUS,
        "v9_audit_receipt_sha256": authority.receipt_sha256,
        "catalog_sha256": digest(catalog),
        "packet_set_sha256": digest(canonical(packet_set_record)),
        "decision_commitment_sha256": commitment,
        "reconstruction_sha256": authority.reconstruction_sha256,
        "unique_selected_decisions": unique_selected,
        "six_level_major_replayed": True,
        "all_level_boundaries_exact": True,
        "all_selected_bits_packet_derived": True,
        "all_base_frequencies_regenerated": True,
        "all_triplet_digests_recomputed": True,
        "all_candidate_payloads_q016_reencoded": True,
        "full_reconstruction_recomputed": True,
        "caller_metrics_used": False,
        "publication_payload_opened_only_after_v9_authority": True,
    })


class BMTests(unittest.TestCase):
    def test_exhaustive_minimality_through_nine(self) -> None:
        for length in range(1, 10):
            for word in range(1 << length):
                sequence = bytes((word >> (length - 1 - index)) & 1 for index in range(length))
                complexity, connection = codec.berlekamp_massey(sequence)
                self.assertEqual(complexity, brute_complexity(sequence))
                self.assertEqual(
                    codec.generate_lfsr(sequence[:complexity], connection, length),
                    sequence,
                )

    def test_chunk_lfsr_random_fallback_crc_and_canonicality(self) -> None:
        lfsr = codec.generate_lfsr(bytes((1, 0, 1, 0, 0, 1, 1)), 1 | (1 << 1) | (1 << 7), 1001)
        packet, row = codec.encode_chunk(
            stream_ordinal=0, level=0, chunk_ordinal=0,
            stream_begin=0, bits=lfsr,
        )
        self.assertEqual(row["mode"], "lfsr")
        self.assertEqual(codec.decode_chunk(packet)["bits"], lfsr)
        corrupt = bytearray(packet)
        corrupt[-1] ^= 0x80
        with self.assertRaises(codec.CodecError):
            codec.decode_chunk(bytes(corrupt))

        random_bits = deterministic_bits(991, 1001)
        raw, raw_row = codec.encode_chunk(
            stream_ordinal=1, level=2, chunk_ordinal=3,
            stream_begin=7, bits=random_bits,
        )
        self.assertEqual(raw_row["mode"], "raw")
        self.assertEqual(codec.decode_chunk(raw)["bits"], random_bits)

        # Nonzero terminal pad bits remain invalid after recomputing SHA/CRC.
        fields = list(codec.CHUNK_HEADER.unpack_from(raw, 0))
        padded_payload = bytearray(raw[codec.CHUNK_HEADER.size:])
        padded_payload[-1] |= 1
        fields[12] = bytes.fromhex(digest(bytes(padded_payload)))
        fields[13] = 0
        zero_header = codec.CHUNK_HEADER.pack(*fields)
        fields[13] = zlib.crc32(zero_header + padded_payload) & 0xFFFFFFFF
        with self.assertRaisesRegex(codec.CodecError, "padding"):
            codec.decode_chunk(codec.CHUNK_HEADER.pack(*fields) + padded_payload)

        # A syntactically valid raw alias of a shorter LFSR representation is
        # rejected by canonical re-encoding even with fresh SHA and CRC.
        payload = codec.pack_bits(lfsr)
        zero = codec.CHUNK_HEADER.pack(
            codec.CHUNK_MAGIC, codec.VERSION, 0, codec.MODE_RAW, 0,
            0, 0, 0, len(lfsr), 0, len(lfsr), len(payload),
            bytes.fromhex(digest(payload)), 0, bytes(4),
        )
        crc = zlib.crc32(zero + payload) & 0xFFFFFFFF
        header = codec.CHUNK_HEADER.pack(
            codec.CHUNK_MAGIC, codec.VERSION, 0, codec.MODE_RAW, 0,
            0, 0, 0, len(lfsr), 0, len(lfsr), len(payload),
            bytes.fromhex(digest(payload)), crc, bytes(4),
        )
        with self.assertRaisesRegex(codec.CodecError, "canonical"):
            codec.decode_chunk(header + payload)


class ExpertPacketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bits, self.levels = six_level_bits(5000, kind="lfsr", seed=31)
        self.streams = streams_for_expert(0, 4096, self.bits, self.levels)
        self.packet, self.metrics = codec.encode_expert(
            expert_ordinal=0, source_weights=3 * 4096,
            semantic_state=b"semantic\x00state\xff", streams=self.streams,
            audit_receipt_sha256="a" * 64, candidate_sha256="b" * 64,
            reconstruction_sha256="c" * 64,
        )

    def test_six_levels_scale_state_and_exact_decisions_retained(self) -> None:
        decoded = codec.decode_expert(self.packet)
        self.assertEqual(decoded["semantic_state"], b"semantic\x00state\xff")
        self.assertEqual(len(decoded["streams"]), 3)
        for before, after in zip(self.streams, decoded["streams"], strict=True):
            self.assertEqual(after.selected_bits, before.selected_bits)
            self.assertEqual(after.levels, before.levels)
            self.assertEqual(after.decoder_scale_f16le, before.decoder_scale_f16le)
            self.assertEqual(after.state, before.state)
            self.assertEqual(tuple(after.levels.count(level) for level in range(6)), (5000,) * 6)
        self.assertGreater(self.metrics["lfsr_chunks"], 0)
        self.assertEqual(len(self.packet) % codec.PAGE_BYTES, 0)

    def test_hostile_packet_mutations_reject(self) -> None:
        corrupt = bytearray(self.packet)
        corrupt[-1] = 1
        with self.assertRaises(codec.CodecError):
            codec.decode_expert(bytes(corrupt))

        # Header padding is authenticated and canonical.
        corrupt = bytearray(self.packet)
        corrupt[codec.EXPERT_CORE.size] ^= 1
        with self.assertRaises(codec.CodecError):
            codec.decode_expert(bytes(corrupt))

        # Swap two literal chunks. Even before resealing, this must fail; the
        # source additionally checks canonical global stream/level geometry.
        fields = codec.EXPERT_CORE.unpack_from(self.packet, 0)
        payload_offset, payload_bytes = fields[10], fields[11]
        body = bytearray(self.packet[codec.EXPERT_HEADER_BYTES:])
        relative = payload_offset - codec.EXPERT_HEADER_BYTES
        first_header = codec.CHUNK_HEADER.unpack_from(body, relative)
        first_bytes = codec.CHUNK_HEADER.size + first_header[11]
        second_header = codec.CHUNK_HEADER.unpack_from(body, relative + first_bytes)
        second_bytes = codec.CHUNK_HEADER.size + second_header[11]
        first = bytes(body[relative:relative + first_bytes])
        second = bytes(body[relative + first_bytes:relative + first_bytes + second_bytes])
        body[relative:relative + first_bytes + second_bytes] = second + first
        self.assertEqual(len(body), len(self.packet) - codec.EXPERT_HEADER_BYTES)
        with self.assertRaises(codec.CodecError):
            codec.decode_expert(reseal_expert(self.packet, bytes(body)))

        # Canonical JSON alone does not canonicalize hex spelling. Uppercase
        # scale hex is therefore independently rejected after a full reseal.
        body = bytearray(self.packet[codec.EXPERT_HEADER_BYTES:])
        old = b'"decoder_scale_f16le_hex":"003d"'
        new = b'"decoder_scale_f16le_hex":"003D"'
        self.assertIn(old, body)
        body[:] = body.replace(old, new, 1)
        with self.assertRaisesRegex(codec.CodecError, "canonical stream hex"):
            codec.decode_expert(reseal_expert(self.packet, bytes(body), metadata_changed=True))

    def test_level_order_missing_level_scale_owner_and_role_coverage_reject(self) -> None:
        bad = copy.copy(self.streams[0])
        object.__setattr__(bad, "levels", bytes(reversed(bad.levels)))
        object.__setattr__(bad, "levels_sha256", digest(bad.levels))
        with self.assertRaises(codec.CodecError):
            codec.validate_stream(bad, 0)

        bad = copy.copy(self.streams[0])
        bad_levels = bytes(0 if value == 5 else value for value in bad.levels)
        object.__setattr__(bad, "levels", bad_levels)
        object.__setattr__(bad, "levels_sha256", digest(bad_levels))
        with self.assertRaises(codec.CodecError):
            codec.validate_stream(bad, 0)

        bad = copy.copy(self.streams[0])
        object.__setattr__(bad, "decoder_scale_f16le", struct.pack("<H", 0x7E00))
        with self.assertRaises(codec.CodecError):
            codec.validate_stream(bad, 0)

        bad = copy.copy(self.streams[0])
        object.__setattr__(bad, "owner_set_hex", owner_set(0, 1))
        with self.assertRaises(codec.CodecError):
            codec.validate_stream(bad, 0)

        all_gate = tuple(copy.copy(row) for row in self.streams)
        for row in all_gate:
            object.__setattr__(row, "role", "gate")
            contribution = dict(row.owner_contributions[0])
            contribution["role"] = "gate"
            object.__setattr__(row, "owner_contributions", (contribution,))
        with self.assertRaises(codec.CodecError):
            codec.encode_expert(
                expert_ordinal=0, source_weights=3 * 4096,
                semantic_state=b"state", streams=all_gate,
                audit_receipt_sha256="a" * 64, candidate_sha256="b" * 64,
                reconstruction_sha256="c" * 64,
            )

    def test_catalog_canonicality_crc_and_packet_binding(self) -> None:
        catalog = scorer.encode_catalog(codec, (self.packet,))
        self.assertEqual(len(catalog), scorer.CATALOG_BYTES)
        parsed = scorer.decode_catalog(catalog)
        self.assertEqual(parsed["experts"][0]["packet_sha256"], digest(self.packet))
        bad = bytearray(catalog)
        bad[20] ^= 1
        with self.assertRaises(scorer.ScoreError):
            scorer.decode_catalog(bytes(bad))


class PhysicalAndAuthorityTests(unittest.TestCase):
    def test_prepayload_bound_hard_kills_raw_but_not_zero_complexity_grammar(self) -> None:
        result = prepayload_rate_gate.derive()
        self.assertEqual(result["pinned_unique_selected_sc_decisions"], 126_627_266)
        self.assertEqual(result["audited_current_arithmetic_rate"]["float"], 2.5)
        self.assertGreater(result["selected_decisions_per_weight_is_not_a_rate"]["float"], 4.0)
        self.assertTrue(result["raw_fallback_hard_kill_before_payload"])
        self.assertFalse(result["raw_fallback_can_fit_2p5"])
        self.assertTrue(result["zero_complexity_grammar_floor_can_fit_2p5"])
        self.assertFalse(result["positive_recurrence_claim_permitted"])

    def test_raw_selected_decisions_are_not_the_arithmetic_rate(self) -> None:
        bits, levels = six_level_bits(4096, kind="random", seed=8301)
        streams = streams_for_expert(0, 4096, bits, levels)
        packet, _ = codec.encode_expert(
            expert_ordinal=0, source_weights=3 * 4096,
            semantic_state=b"s", streams=streams,
            audit_receipt_sha256="a" * 64, candidate_sha256="b" * 64,
            reconstruction_sha256="c" * 64,
        )
        decoded = codec.decode_expert(packet)
        bounds = codec.packet_rate_bounds((decoded,))
        self.assertEqual(bounds["selected_decisions_per_weight_is_not_a_rate"], 6.0)
        self.assertTrue(bounds["selected_decisions_are_arithmetic_coded_in_current_strata"])
        self.assertGreaterEqual(bounds["actual_packet_bytes"], bounds["unconditional_zero_complexity_floor_bytes"])
        self.assertGreater(bounds["raw_fallback_bpw"], 2.5)
        self.assertFalse(bounds["raw_fallback_can_fit_2p5"])
        self.assertFalse(bounds["actual_can_fit_2p5"])
        self.assertEqual(bounds["model_bytes" if "model_bytes" in bounds else "shared_model_bytes"], 0)
        self.assertFalse(bounds["exceptions_or_discrepancies_implemented"])

    def test_v9_and_replay_authorities_are_external_digest_gated(self) -> None:
        receipt = fake_v9_receipt("d" * 64)
        authority = gate.authorize_v9_audit_receipt(receipt, expected_receipt_sha256=digest(receipt))
        self.assertEqual(authority.decision_commitment_sha256, "d" * 64)
        mutated = bytearray(receipt)
        mutated[-2] ^= 1
        with self.assertRaises(gate.GateError):
            gate.authorize_v9_audit_receipt(bytes(mutated), expected_receipt_sha256=digest(receipt))

        replay = canonical({
            "schema": replay_gate.SCHEMA, "status": replay_gate.STATUS,
            "v9_audit_receipt_sha256": authority.receipt_sha256,
            "catalog_sha256": "1" * 64, "packet_set_sha256": "2" * 64,
            "decision_commitment_sha256": "d" * 64,
            "reconstruction_sha256": gate.RECONSTRUCTION_SHA256,
            "unique_selected_decisions": 1,
            "six_level_major_replayed": True, "all_level_boundaries_exact": True,
            "all_selected_bits_packet_derived": True,
            "all_base_frequencies_regenerated": True,
            "all_triplet_digests_recomputed": True,
            "all_candidate_payloads_q016_reencoded": True,
            "full_reconstruction_recomputed": True, "caller_metrics_used": False,
            "publication_payload_opened_only_after_v9_authority": True,
        })
        replay_authority = replay_gate.authorize_replay_receipt(
            replay, expected_receipt_sha256=digest(replay),
        )
        self.assertEqual(replay_authority.v9_audit_receipt_sha256, authority.receipt_sha256)

    def test_full_scorer_derives_rate_and_f_and_requires_replay(self) -> None:
        # Synthetic mechanics only: six experts sum to the exact pinned source
        # weight count, but streams contain six decisions each and cannot be
        # model evidence. This exercises all trust/ledger paths without payload.
        expert_weights = gate.SOURCE_WEIGHTS // 6
        matrix_weights = expert_weights // 3
        all_streams = []
        for expert in range(6):
            bits, levels = six_level_bits(1, kind="zero", seed=expert)
            all_streams.append(streams_for_expert(
                expert, matrix_weights, bits, levels,
                ordinal_begin=3 * expert,
            ))
        rows = []
        for streams in all_streams:
            for source in streams:
                rows.append({
                    "ordinal": source.ordinal, "symbols": len(source.selected_bits),
                    "logical_bits": source.candidate_logical_bits,
                    "decoded_selected_decision_triplet_sha256": source.decoded_triplet_sha256,
                    "payload_sha256": source.candidate_payload_sha256,
                    "profile_q": source.profile_q, "role": source.role,
                    "owner_set_hex": source.owner_set_hex,
                    "owner_contributions": [dict(value) for value in source.owner_contributions],
                })
        rows.sort(key=lambda row: row["ordinal"])
        commitment = digest(canonical(rows))
        receipt = fake_v9_receipt(commitment)
        authority = gate.authorize_v9_audit_receipt(receipt, expected_receipt_sha256=digest(receipt))
        packets = tuple(
            codec.encode_expert(
                expert_ordinal=expert, source_weights=expert_weights,
                semantic_state=b"synthetic-semantic-state", streams=streams,
                audit_receipt_sha256=authority.receipt_sha256,
                candidate_sha256=gate.CANDIDATE_SHA256,
                reconstruction_sha256=authority.reconstruction_sha256,
            )[0]
            for expert, streams in enumerate(all_streams)
        )
        catalog = scorer.encode_catalog(codec, packets)
        replay = fake_replay_receipt(authority, catalog, packets, commitment, len(rows) * 6)
        replay_authority = replay_gate.authorize_replay_receipt(
            replay, expected_receipt_sha256=digest(replay),
        )
        result = scorer.score_authenticated_packets(
            codec, gate, replay_gate, authority=authority,
            replay_authority=replay_authority, catalog=catalog, packets=packets,
        )
        expected_bytes = len(catalog) + sum(map(len, packets))
        self.assertEqual(result["physical_bytes"], expected_bytes)
        self.assertEqual(
            result["physical_rate_rational"],
            {"numerator": Fraction(8 * expected_bytes, gate.SOURCE_WEIGHTS).numerator,
             "denominator": Fraction(8 * expected_bytes, gate.SOURCE_WEIGHTS).denominator},
        )
        self.assertTrue(result["all_values_scorer_derived"])
        self.assertFalse(result["caller_supplied_mse_rate_ledger_or_controls"])
        self.assertTrue(result["q016_recurrence_packet_replay_independently_authorized"])
        self.assertEqual(result["status"], "HARD_KILL_EXACT_RECURRENCE_RATE_F_OR_COLD")
        with self.assertRaises(scorer.ScoreError):
            scorer.score_authenticated_packets(
                codec, gate, replay_gate, authority=authority,
                replay_authority=None, catalog=catalog, packets=packets,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
