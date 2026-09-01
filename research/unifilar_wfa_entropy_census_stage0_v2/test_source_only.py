#!/usr/bin/env python3
"""Hostile producer-side tests; all dynamic data is synthetic and temporary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).absolute().parent


def load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load("uwfa_v2_test_common", "uwfa_common.py")
codec = load("uwfa_v2_test_container", "container_codec.py")
protocol = load("uwfa_v2_test_protocol", "protocol.py")
adapter_module = load("uwfa_v2_test_adapter", "strata_sc_adapter.py")
stage = load("uwfa_v2_test_stage", "stage0_census.py")
dispatcher = load("uwfa_v2_test_dispatcher", "dispatcher_contract.py")
envelope = load("uwfa_v2_test_result_envelope", "result_envelope.py")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bindings() -> dict[str, str]:
    return {
        "baseline_plan_sha256": sha(b"plan"),
        "baseline_score_sha256": sha(b"score"),
        "universal_decoder_sha256": sha(b"decoder"),
        "producer_manifest_sha256": sha(b"manifest"),
        "audit_bootstrap_sha256": sha(b"bootstrap"),
        "source_panel_sha256": sha(b"panel"),
        "extraction_program_sha256": sha(b"extractor"),
    }


def synthetic_stream(seed: int, ordinal: int, length: int) -> tuple[list[int], list[int], list[int]]:
    bits = [((seed >> (index & 15)) ^ (ordinal * 19) ^ (index * 13) ^ (index >> 3)) & 1 for index in range(length)]
    levels = [(index + ordinal) % common.LEVELS for index in range(length)]
    base = [1 + ((seed + ordinal * 811 + index * 7919) % 65535) for index in range(length)]
    return bits, levels, base


def u16le(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}H", *values)


class CpuBackend:
    def __init__(self) -> None:
        self.pack_calls = 0

    def pack_streams(self, rows: list[tuple[bytes, bytes, bytes]]) -> dict[str, Any]:
        self.pack_calls += 1
        return {"rows": rows}

    def subset(self, packed: dict[str, Any], indices: list[int]) -> dict[str, Any]:
        return {"rows": [packed["rows"][index] for index in indices]}

    def fit_counts(self, packed: dict[str, Any], topology_id: int, states: int, reset: int) -> list[int]:
        candidate = common.Candidate(common.TOPOLOGIES[topology_id], states, reset)
        rows = []
        for bits_b, levels_b, base_b in packed["rows"]:
            base = list(struct.unpack(f"<{len(base_b)//2}H", base_b))
            rows.append(common.count_stream_cpu(list(bits_b), list(levels_b), base, candidate))
        return common.merge_counts(rows)

    def exact_lengths(self, packed: dict[str, Any], topology_id: int, states: int, reset: int, frequencies: list[int]) -> list[int]:
        candidate = common.Candidate(common.TOPOLOGIES[topology_id], states, reset)
        output = []
        for bits_b, levels_b, base_b in packed["rows"]:
            base = list(struct.unpack(f"<{len(base_b)//2}H", base_b))
            output.append(common.exact_stream_length_cpu(list(bits_b), list(levels_b), base, candidate, frequencies))
        return output


def build_literal_fixture(*, unequal: bool = False) -> tuple[bytes, dict[str, Any], dict[int, tuple[list[int], list[int], list[int]]]]:
    candidate = common.Candidate("xor_sketch", 4, 128)
    generated = {}
    counts = []
    for ordinal in range(15):
        length = 97 + (ordinal * 17 if unequal else ordinal % 3)
        bits, levels, base = synthetic_stream(0xA5C31, ordinal, length)
        generated[ordinal] = (bits, levels, base)
        counts.append(common.count_stream_cpu(bits, levels, base, candidate))
    frequencies = common.q16_frequencies_from_counts(common.merge_counts(counts))
    model = common.serialize_model(candidate, frequencies)
    region_rows: dict[int, list[Any]] = {}
    for ordinal in range(15):
        bits, levels, base = generated[ordinal]
        payload, logical = common.encode_unifilar_stream(bits, levels, base, candidate, frequencies)
        if ordinal < 12:
            mask = 1 << (ordinal // 2)
        else:
            pair = ordinal - 12
            mask = (1 << (2 * pair)) | (1 << (2 * pair + 1))
        digest = hashlib.sha256(bytes(bits) + bytes(levels) + u16le(base)).hexdigest()
        spec = codec.StreamSpec(ordinal, len(bits), logical, payload, digest, ordinal, 0.5 + ordinal / 32.0)
        region_rows.setdefault(mask, []).append(spec)
    masks = sorted(region_rows, key=lambda mask: (mask.bit_count() != 1, mask))
    regions = [codec.RegionSpec(mask, tuple(region_rows[mask])) for mask in masks]
    immutable = b"synthetic-universal-context-seed:" + (0xA5C31).to_bytes(8, "little")
    container, metrics = codec.build_container(
        common,
        model_packet=model,
        immutable_state=immutable,
        regions=regions,
        weights=300_000,
        experts=6,
        baseline_object_bytes=100_000,
        audited_relative_mse=0.025,
        baseline_artifact_sha256=sha(b"baseline-artifact"),
        reconstruction_sha256=sha(b"synthetic-reconstruction"),
        audit_binding_sha256=bindings()["baseline_score_sha256"],
        binding_hashes=bindings(),
        minimum_rate_bpw=2.15,
    )
    return container, metrics, generated


class FrozenMathAndArithmetic(unittest.TestCase):
    def test_exact_threshold_and_bank(self) -> None:
        self.assertAlmostEqual(common.STANDALONE_REQUIRED_SAVING_BPW, 0.15288996696291447, places=15)
        bank = common.candidate_bank()
        self.assertEqual(len(bank), 150)
        self.assertEqual([row.selector_ordinal for row in bank], list(range(150)))

    def test_all_transition_extremes_and_reset_law(self) -> None:
        for candidate in common.candidate_bank():
            for state in {0, candidate.states - 1}:
                for bit in (0, 1):
                    for context in (0, common.CONTEXTS - 1):
                        for position in (0, candidate.reset_length - 1):
                            observed = common.transition(candidate, state, bit, context, position)
                            self.assertTrue(0 <= observed < candidate.states)
            bits = [1] * candidate.reset_length + [0]
            levels = [0] * len(bits)
            base = [1] * len(bits)
            frequencies = [1 + index % 65535 for index in range(common.model_frequency_count(candidate))]
            used = common.stream_frequencies_cpu(bits, levels, base, candidate, frequencies)
            context0 = common.public_context(0, 1, 0)
            self.assertEqual(used[0], frequencies[context0])
            self.assertEqual(used[candidate.reset_length], frequencies[context0])

    def test_exhaustive_small_arithmetic_and_corruption(self) -> None:
        for length in range(1, 10):
            for word in range(1 << length):
                bits = [(word >> (length - 1 - index)) & 1 for index in range(length)]
                frequencies = [1, 65535, 32768, 8192, 57344] * 2
                payload, logical = common.arithmetic_encode_binary(bits, frequencies[:length])
                decoded = common.arithmetic_decode_binary(payload, logical, length, lambda index: frequencies[index])
                self.assertEqual(decoded, bits)
        bits = [index & 1 for index in range(257)]
        frequencies = [32768] * len(bits)
        payload, logical = common.arithmetic_encode_binary(bits, frequencies)
        # Bare arithmetic coding has no independent message checksum: a bit
        # mutation can be another canonical message.  The literal container's
        # payload SHA/body root/source digest provide mandatory corruption
        # detection and are tested below.

    def test_model_is_canonical(self) -> None:
        candidate = common.Candidate("suffix", 8, 32)
        frequencies = [1 + index % 65535 for index in range(common.model_frequency_count(candidate))]
        packet = common.serialize_model(candidate, frequencies)
        self.assertEqual(common.deserialize_model(packet), (candidate, frequencies))
        for offset in (0, 9, 63, len(packet) - 1):
            tampered = bytearray(packet)
            tampered[offset] ^= 1
            with self.assertRaises(Exception):
                common.deserialize_model(bytes(tampered))


class LiteralContainerTests(unittest.TestCase):
    def test_complete_parse_decode_reencode_and_rebuild(self) -> None:
        container, metrics, generated = build_literal_fixture()
        self.assertEqual(len(container) % codec.PAGE_BYTES, 0)
        parsed = codec.parse_container(common, container)
        self.assertEqual(codec.canonical_rebuild(common, parsed), container)
        self.assertEqual(parsed["binding_hashes"], bindings())
        self.assertEqual(struct.unpack_from("<H", container, 10)[0], codec.HEADER_BYTES)
        for row in parsed["directory"]:
            bits, levels, base = generated[int(row["ordinal"])]
            decoded = common.decode_unifilar_stream(
                row["payload"], int(row["logical_bits"]), levels, base, parsed["candidate"], parsed["frequencies"]
            )
            self.assertEqual(decoded, bits)
            replay, logical = common.encode_unifilar_stream(bits, levels, base, parsed["candidate"], parsed["frequencies"])
            self.assertEqual((replay, logical), (row["payload"], row["logical_bits"]))
        self.assertAlmostEqual(metrics["actual_physical_rate_bpw"], 8 * len(container) / 300_000)
        self.assertAlmostEqual(metrics["F_from_actual_bytes_and_identical_reconstruction"], 0.025 * 2 ** (2 * 8 * len(container) / 300_000))

    def test_cold_owner_attribution_is_not_total_over_e(self) -> None:
        container, _metrics, _ = build_literal_fixture(unequal=True)
        parsed = codec.parse_container(common, container)
        metrics = codec.physical_metrics(parsed)
        self.assertAlmostEqual(metrics["ownership_allocated_bytes_sum"], len(container))
        total_over_e = len(container) / 6
        self.assertTrue(any(abs(row["allocated_physical_denominator_bytes"] - total_over_e) > 1 for row in metrics["experts"]))
        for row in metrics["experts"]:
            self.assertEqual(row["touched_page_bytes"], len(row["touched_page_indices"]) * 4096)
            self.assertEqual(len(row["touched_page_indices"]), len(set(row["touched_page_indices"])))
            independent_trace = codec.instrument_expert_pages(parsed, int(row["expert_ordinal"]))
            self.assertEqual(independent_trace["touched_page_indices"], row["touched_page_indices"])
            self.assertEqual(independent_trace["read_ranges"], row["instrumented_read_ranges"])

    def test_every_section_tamper_truncation_trailing_and_version_reject(self) -> None:
        container, _metrics, _ = build_literal_fixture()
        parsed = codec.parse_container(common, container)
        offsets = [0, 8, 116, codec.HEADER_BYTES, parsed["directory"][0]["payload_offset"], len(container) - 1]
        for offset in offsets:
            tampered = bytearray(container)
            tampered[int(offset)] ^= 1
            with self.subTest(offset=offset), self.assertRaises(Exception):
                codec.parse_container(common, bytes(tampered))
        with self.assertRaises(Exception):
            codec.parse_container(common, container[:-1])
        with self.assertRaises(Exception):
            codec.parse_container(common, container + b"\x00" * 4096)
        tampered = bytearray(container)
        struct.pack_into("<H", tampered, 8, 99)
        with self.assertRaises(Exception):
            codec.parse_container(common, bytes(tampered))

    def test_corrupt_serialized_model_rejected_before_acceptance(self) -> None:
        container, _metrics, _ = build_literal_fixture()
        parsed = codec.parse_container(common, container)
        bad = bytearray(parsed["model_packet"])
        bad[0] ^= 0xFF
        with self.assertRaises(Exception):
            codec.build_container(
                common,
                model_packet=bytes(bad),
                immutable_state=b"x",
                regions=[codec.RegionSpec(1, (codec.StreamSpec(0, 1, 1, b"\x00", sha(b"x")),))],
                weights=100000,
                experts=1,
                baseline_object_bytes=50000,
                audited_relative_mse=0.1,
                baseline_artifact_sha256=sha(b"a"),
                reconstruction_sha256=sha(b"r"),
                audit_binding_sha256=bindings()["baseline_score_sha256"],
                binding_hashes=bindings(),
            )


class IntegratedAdapterContracts(unittest.TestCase):
    def test_original_frequency_is_regenerated_context_not_side_array(self) -> None:
        class Base:
            def __init__(self) -> None:
                self.frequencies = []

            def decode(self, frequency: int) -> int:
                self.frequencies.append(frequency)
                return len(self.frequencies) & 1

        candidate = common.Candidate("suffix", 4, 32)
        table = [1 + index % 65535 for index in range(common.model_frequency_count(candidate))]
        base_decoder = Base()
        adapter = adapter_module.UWFAArithmeticDecoder(common, base_decoder, candidate, table)
        adapter.set_level(3)
        original = [1, 4096, 32768, 65535, 17]
        observed = [adapter.decode(value) for value in original]
        self.assertEqual(adapter.original_frequencies, original)
        self.assertEqual(base_decoder.frequencies, adapter.uwfa_frequencies)
        self.assertEqual(observed, adapter.decoded_bits)
        first_context = common.public_context(3, original[0], 0)
        self.assertEqual(adapter.uwfa_frequencies[0], table[first_context])

    def test_inherited_metadata_is_literal_and_tamper_closed(self) -> None:
        class FakeStrata:
            WEIGHTS = 96
            EXPERTS = 2
            MATRICES = 6
            BLOCKS = 3
            HEADER_BYTES = 8
            ROUTE_BYTES = 6
            LABEL_BYTES = 3
            BLOCK_LOG2 = (5, 5, 5)

            @staticmethod
            def block_owner_experts(ordinal: int) -> list[int]:
                return [0] if ordinal == 0 else ([1] if ordinal == 1 else [0, 1])

            @staticmethod
            def validate_header(header: bytes, route: bytes, labels: bytes) -> None:
                if (header, route, labels) != (b"H" * 8, b"R" * 6, b"L" * 3):
                    raise ValueError("semantic bytes")

            @staticmethod
            def unpack_labels(payload: bytes) -> bytes:
                return payload

            @staticmethod
            def parse_route(_payload: bytes) -> list[dict[str, Any]]:
                return [{"role": "gate"}] * 6

            @staticmethod
            def derive_seeds(_h: bytes, _r: bytes, _l: bytes, _p: bytes, ordinal: int) -> tuple[int, int, str]:
                return ordinal + 1, ordinal + 7, sha(bytes((ordinal,)))

        blocks = []
        for ordinal in range(3):
            blocks.append(adapter_module.BaselineBlock(
                ordinal, 5, adapter_module.owner_mask_for_block(FakeStrata, ordinal), ordinal,
                float(struct.unpack("<e", struct.pack("<e", 0.5 + ordinal))[0]), struct.pack("<e", 0.5 + ordinal),
                10, b"\x00\x00", ordinal + 1, ordinal + 7,
            ))
        parsed = {"header": b"H" * 8, "route": b"R" * 6, "labels_packed": b"L" * 3, "blocks": blocks}
        packet = adapter_module.pack_immutable_metadata(parsed, FakeStrata)
        recovered = adapter_module.unpack_immutable_metadata(packet, FakeStrata)
        self.assertEqual(recovered["profiles"], b"\x00\x01\x02")
        for offset in (0, 40, len(packet) - 1):
            bad = bytearray(packet)
            bad[offset] ^= 1
            with self.assertRaises(Exception):
                adapter_module.unpack_immutable_metadata(bytes(bad), FakeStrata)


class SecureLifecycleTests(unittest.TestCase):
    def test_symlink_leaf_and_ancestor_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).absolute()
            real = root / "real"
            real.mkdir()
            target = real / "file.bin"
            target.write_bytes(b"x")
            leaf = root / "leaf.bin"
            ancestor = root / "ancestor"
            try:
                leaf.symlink_to(target)
                ancestor.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with self.assertRaises(Exception):
                common.HeldRegularFile(leaf).open()
            with self.assertRaises(Exception):
                common.HeldRegularFile(ancestor / "file.bin").open()
            with self.assertRaises(Exception):
                dispatcher.open_snapshot(ancestor / "file.bin")

    @unittest.skipIf(os.name == "nt", "descriptor-relative rename attack requires Unix")
    def test_output_path_replacement_cannot_redirect_writes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).absolute()
            output = root / "output"
            moved = root / "moved"
            evil = root / "evil"
            evil.mkdir()
            with common.CompletionLastOutput(output) as transaction:
                os.rename(output, moved)
                output.symlink_to(evil, target_is_directory=True)
                transaction.write_new("payload.bin", b"safe")
                self.assertEqual((moved / "payload.bin").read_bytes(), b"safe")
                self.assertFalse((evil / "payload.bin").exists())

    def test_completion_last_and_incomplete_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).absolute()
            incomplete = root / "incomplete"
            with common.CompletionLastOutput(incomplete) as transaction:
                transaction.write_new("payload.bin", b"x")
            with self.assertRaises(Exception):
                envelope.verify_completed_directory(common, incomplete)
            complete = root / "complete"
            with common.CompletionLastOutput(complete) as transaction:
                transaction.write_new("payload.bin", b"x")
                transaction.complete(list(transaction.members), sha(b"manifest"))
                with self.assertRaises(Exception):
                    transaction.write_new("late.bin", b"bad")
            receipt = envelope.verify_completed_directory(common, complete)
            self.assertEqual(receipt["status"], "PASS_COMPLETE_LAST_ENVELOPE")

    def test_direct_producer_and_reference_dispatcher_never_grant_authority(self) -> None:
        run = subprocess.run([sys.executable, "-I", "-B", str(PACKAGE / "stage0_census.py"), "--fabricated", "PASS"], capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 2)
        self.assertIn(stage.DIRECT_LAUNCH_STATUS, run.stderr)
        with self.assertRaises(dispatcher.DispatchContractError):
            dispatcher.reject_direct_payload_launch()

    def test_authenticated_snapshot_exec_ignores_replaced_path(self) -> None:
        source = b"VALUE = 'authenticated'\n"
        module = dispatcher.exec_snapshot_module("uwfa_snapshot_test", source, expected_sha256=sha(source))
        self.assertEqual(module.VALUE, "authenticated")
        with self.assertRaises(Exception):
            dispatcher.exec_snapshot_module("uwfa_snapshot_bad", b"VALUE=2\n", expected_sha256=sha(source))

    def test_fabricated_self_consistent_review_and_undeclared_member_reject(self) -> None:
        fabricated = common.seal_record(
            {
                "schema": "unifilar-wfa-entropy-census-independent-source-review-v2",
                "status": "PASS_INDEPENDENT_SOURCE_REVIEW",
                "reviewed_source_manifest_sha256": sha(b"manifest"),
            },
            "review_sha256",
        )
        payload = common.canonical_json(fabricated)
        # Internal integrity is deliberately irrelevant to the external pin.
        with self.assertRaises(dispatcher.DispatchContractError):
            dispatcher.require_external_pin(payload, sha(b"independently-pinned-different-review"), "review")
        dispatcher.require_external_pin(payload, sha(payload), "review")
        with self.assertRaises(dispatcher.DispatchContractError):
            dispatcher.require_exact_declared_members({"a.py", "extra.py"}, {"a.py"})


class StrictProtocolTests(unittest.TestCase):
    def test_integer_identifier_and_unknown_field_rejection(self) -> None:
        for value in (True, 1.0, "1", -1):
            with self.assertRaises(Exception):
                protocol.exact_int(value, "x", 0, 10)
        for value in ("a\x00b", "unicode-\u2603", "", "a" * 129):
            with self.assertRaises(Exception):
                protocol.identifier(value, "id")
        with self.assertRaises(Exception):
            protocol.strict_fields({"x": 1, "y": 2}, required=("x",), label="row")
        digest1 = protocol.length_prefixed_digest(["a", "bc"], domain=b"d")
        digest2 = protocol.length_prefixed_digest(["ab", "c"], domain=b"d")
        self.assertNotEqual(digest1, digest2)

    def test_score_receipt_binds_D_to_artifact_weights_reconstruction_and_sums(self) -> None:
        record = {
            "schema": "uwfa-bound-baseline-score-v2",
            "status": "PASS_INDEPENDENT_BASELINE_SCORE",
            "artifact_sha256": sha(b"artifact"),
            "artifact_bytes": 8,
            "weights": 4,
            "relative_mse": 0.25,
            "sse_fp64": 1.0,
            "source_energy_fp64": 4.0,
            "normalization": "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM",
            "reconstruction_f64_sha256": sha(b"recon"),
            "original_source_panel_sha256": sha(b"source"),
            "independent_decoder_source_sha256": sha(b"decoder"),
        }
        record["score_receipt_sha256"] = sha(common.canonical_json(record))
        validated = protocol.validate_score_receipt(record, artifact_sha256=sha(b"artifact"), artifact_bytes=8, weights=4, reconstruction_sha256=sha(b"recon"))
        self.assertEqual(validated["relative_mse"], 0.25)
        for key, value in (("artifact_bytes", 9), ("weights", 5), ("relative_mse", 0.2)):
            bad = dict(record)
            bad[key] = value
            clean = dict(bad)
            clean.pop("score_receipt_sha256")
            bad["score_receipt_sha256"] = sha(common.canonical_json(clean))
            with self.assertRaises(Exception):
                protocol.validate_score_receipt(bad, artifact_sha256=sha(b"artifact"), artifact_bytes=8, weights=4, reconstruction_sha256=sha(b"recon"))


def scientific_panel() -> dict[str, Any]:
    streams = []
    weights = 0
    for ordinal in range(15):
        bits, levels, base = synthetic_stream(0x5555, ordinal, 32 + ordinal % 4)
        payload, logical = common.arithmetic_encode_binary(bits, base)
        mask = 1 << (ordinal // 2) if ordinal < 12 else (3 << (2 * (ordinal - 12)))
        owners = [expert for expert in range(6) if mask & (1 << expert)]
        weight = 1000
        weights += weight
        streams.append({
            "stream_ordinal": ordinal,
            "owner_mask": mask,
            "owner_identity_indices": owners,
            "weight_charge": weight,
            "symbols": len(bits),
            "bits": bits,
            "levels": levels,
            "base": base,
            "bits_bytes": bytes(bits),
            "levels_bytes": bytes(levels),
            "base_bytes": u16le(base),
            "baseline_payload_bytes": len(payload),
        })
    return {"streams": streams, "weights": weights, "semantic_identities": [(10 + index, 100 + index) for index in range(6)]}


class NestedAndControlTests(unittest.TestCase):
    def test_predeclared_shuffles_are_deterministic_and_preserve_declared_statistics(self) -> None:
        panel = scientific_panel()
        first = stage.within_context_permutation(common, protocol, panel)
        second = stage.within_context_permutation(common, protocol, panel)
        self.assertEqual([row["bits_bytes"] for row in first["streams"]], [row["bits_bytes"] for row in second["streams"]])
        for original, shuffled in zip(panel["streams"], first["streams"], strict=True):
            original_counts = {}
            shuffled_counts = {}
            for position, (bit_a, bit_b, level, base) in enumerate(zip(original["bits"], shuffled["bits"], original["levels"], original["base"], strict=True)):
                context = common.public_context(level, base, position & 3)
                original_counts[(context, bit_a)] = original_counts.get((context, bit_a), 0) + 1
                shuffled_counts[(context, bit_b)] = shuffled_counts.get((context, bit_b), 0) + 1
            self.assertEqual(original_counts, shuffled_counts)
        for chunk in (32, 128, 512):
            shuffled = stage.multiscale_chunk_shuffle(protocol, panel, chunk)
            for original, changed in zip(panel["streams"], shuffled["streams"], strict=True):
                triples_a = sorted(zip(original["bits"], original["levels"], original["base"], strict=True))
                triples_b = sorted(zip(changed["bits"], changed["levels"], changed["base"], strict=True))
                self.assertEqual(triples_a, triples_b)

    def test_nested_semantic_exclusion_full_model_charge_and_determinism(self) -> None:
        panel = scientific_panel()
        backend = CpuBackend()
        cache = backend.pack_streams(stage.packed_rows(panel["streams"]))
        first = stage.nested_holdout(common, protocol, backend, cache, panel)
        second = stage.nested_holdout(common, protocol, backend, cache, panel)
        self.assertEqual(first, second)
        self.assertEqual(len(first["folds"]), 6)
        self.assertAlmostEqual(sum(row["allocated_test_weights"] for row in first["folds"]), panel["weights"])
        for fold in first["folds"]:
            selected = common.candidate_bank()[fold["selected_by_inner_validation_only"]["selector_ordinal"]]
            self.assertEqual(fold["charged_full_fold_model_bits"], 8 * common.model_ledger(selected)["physical_model_bytes"])
            outer = (fold["outer_layer_from_artifact"], fold["outer_expert_from_artifact"])
            for ordinal in fold["development_stream_ordinals"]:
                row = panel["streams"][ordinal]
                for owner in row["owner_identity_indices"]:
                    identity = panel["semantic_identities"][owner]
                    self.assertNotEqual(identity[0], outer[0])
                    self.assertNotEqual(identity[1], outer[1])

    def test_promotion_is_impossible_if_any_single_gate_false(self) -> None:
        names = ["physical", "cold", "heldout", "specificity", "standalone_decode", "integrity", "independent_result_audit"]
        self.assertTrue(stage.promotion_conjunction(**{name: True for name in names}))
        for missing in names:
            row = {name: True for name in names}
            row[missing] = False
            self.assertFalse(stage.promotion_conjunction(**row), missing)

    def test_controls_repeat_full_selection_after_all_eight_replays(self) -> None:
        events = []

        class FakeAdapter:
            def extract_from_current(self, raw: bytes) -> dict[str, Any]:
                events.append(("extract", raw))
                streams = []
                for ordinal in range(6):
                    streams.append({"owner_mask": 1 << ordinal, "stream_ordinal": ordinal})
                return {
                    "artifact": {"route_rows": [{"layer": index // 3, "expert": index // 3} for index in range(18)], "raw_bytes": len(raw), "raw_sha256": sha(raw)},
                    "streams": streams,
                    "weights": 6,
                    "experts": 6,
                    "immutable_state": b"x",
                    "reconstruction": {"full_reconstruction_f64_sha256": sha(raw + b"r")},
                }

        class FakeProtocol:
            @staticmethod
            def geometry_sha256(_common: Any, _panel: Any) -> str:
                return "g"

            @staticmethod
            def validate_control_binding(*_args: Any, **_kwargs: Any) -> None:
                return None

            @staticmethod
            def validate_score_receipt(record: Any, **_kwargs: Any) -> Any:
                return record

        originals = (stage.projected_updates, stage.prepare_backend_cache, stage.nested_holdout, stage.final_container)
        try:
            stage.projected_updates = lambda *_args: {"passes_pre_fit_runtime_budget": True}
            stage.prepare_backend_cache = lambda *_args: events.append(("cache", None)) or object()
            stage.nested_holdout = lambda *_args: events.append(("nested150", None)) or {"final_topology_selected_from_nested_fold_votes": {"selector_ordinal": 0}}
            stage.final_container = lambda *_args: {
                "absolute_saving_vs_bound_current_artifact_bpw": 0.1,
                "container": b"c",
                "identity_framing_container": b"i",
            }
            controls = []
            for seed in common.CONTROL_SEEDS:
                raw = f"control-{seed}".encode()
                controls.append({
                    "artifact_bytes": raw,
                    "score_receipt_bytes": common.canonical_json({"relative_mse": 1.0}),
                    "binding_record": {},
                    "bindings": object(),
                })
            result = stage.controls_phase(
                common=common,
                protocol=FakeProtocol,
                container_codec=codec,
                adapter_factory=FakeAdapter,
                backend_factory=lambda: events.append(("backend", None)) or object(),
                source_result={
                    "controls_may_be_opened": True,
                    "source_geometry_sha256": "g",
                    "source_pipeline_sha256": "p",
                    "source_final": {"absolute_saving_vs_bound_current_artifact_bpw": 0.2},
                },
                source_artifact_sha256=sha(b"source"),
                controls=controls,
            )
            self.assertTrue(result["specificity_pass"])
            self.assertEqual(sum(1 for kind, _ in events if kind == "extract"), 8)
            self.assertEqual(sum(1 for kind, _ in events if kind == "nested150"), 8)
            first_fit = min(index for index, event in enumerate(events) if event[0] == "backend")
            self.assertEqual(sum(1 for event in events[:first_fit] if event[0] == "extract"), 8)
        finally:
            stage.projected_updates, stage.prepare_backend_cache, stage.nested_holdout, stage.final_container = originals


class PackageSealTests(unittest.TestCase):
    def test_manifest_if_present(self) -> None:
        manifest = PACKAGE / "SOURCE_MANIFEST.json"
        if not manifest.exists():
            self.skipTest("source manifest is intentionally created only after all tests pass")
        run = subprocess.run([sys.executable, "-I", "-B", str(PACKAGE / "verify_source.py"), "--package", str(PACKAGE), "--compact"], capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
