#!/usr/bin/env python3
"""Hostile source-only tests for the SILT finite mechanism.

The test module creates all arrays internally and never opens an input payload.
Set ``SILT_REQUIRE_CUPY_TEST=1`` to make the GPU search test mandatory; the
canonical RunPod verification does so and also sets ``SILT_REQUIRE_RTX5090=1``.
"""

from __future__ import annotations

import math
import os
import unittest

import numpy as np

import independent_decoder
import silt_mechanism as sm


class LiftTests(unittest.TestCase):
    def test_all_selectors_arbitrary_positive_lanes(self) -> None:
        rng = np.random.default_rng(88013)
        for alphabet in (2, 4):
            for lanes in (1, 2, 3, 5, 17, 97, 257):
                leaves = rng.integers(0, alphabet, size=(7, lanes), dtype=np.uint8)
                permutations = [list(range(lanes)), list(reversed(range(lanes)))]
                if lanes > 1:
                    permutations.append(sm.deterministic_permutation(lanes, 7000 + lanes))
                for permutation in permutations:
                    for code in range(8):
                        selectors = [code] * (lanes - 1)
                        lifted = sm.lift_forward(leaves, alphabet, permutation, selectors)
                        self.assertEqual(sum(value.shape[1] for value in lifted.detail_levels), lanes - 1)
                        flat = sm.flatten_details(lifted)
                        rebuilt_lifted = sm.LiftedTensor(
                            lifted.roots.copy(), sm.unflatten_details(flat, leaves.shape[0], lanes)
                        )
                        decoded = sm.lift_inverse(
                            rebuilt_lifted, lanes, alphabet, permutation, selectors
                        )
                        self.assertTrue(np.array_equal(decoded, leaves))

    def test_reject_bad_geometry_and_alphabet(self) -> None:
        leaves = np.zeros((2, 3), dtype=np.uint8)
        with self.assertRaises(sm.ContractError):
            sm.lift_forward(leaves, 3, [0, 1, 2], [0, 0])
        with self.assertRaises(sm.ContractError):
            sm.lift_forward(leaves.astype(np.int16), 2, [0, 1, 2], [0, 0])
        with self.assertRaises(sm.ContractError):
            sm.level_sizes(0)
        with self.assertRaises(sm.ContractError):
            sm.lift_forward(leaves, 2, [0, 1, 1], [0, 0])
        with self.assertRaises(sm.ContractError):
            sm.lift_forward(leaves, 2, [0, 1, 2], [0])


class MetadataTests(unittest.TestCase):
    def test_factoradic_canonical_roundtrip(self) -> None:
        for lanes in (1, 2, 3, 5, 17, 97):
            for seed in (0, 1, 9173):
                permutation = sm.deterministic_permutation(lanes, seed)
                packet = sm.serialize_permutation(permutation)
                self.assertEqual(len(packet), sm.permutation_byte_count(lanes))
                self.assertEqual(sm.deserialize_permutation(lanes, packet), permutation)
                self.assertEqual(sm.rank_permutation(permutation), int.from_bytes(packet, "big") if packet else 0)

    def test_factoradic_rejects_noncanonical_rank_and_duplicates(self) -> None:
        with self.assertRaises(sm.ContractError):
            sm.rank_permutation([0, 0])
        lanes = 3
        self.assertEqual(sm.permutation_byte_count(lanes), 1)
        with self.assertRaises(sm.ContractError):
            sm.deserialize_permutation(lanes, b"\xff")
        with self.assertRaises(sm.ContractError):
            sm.deserialize_permutation(1, b"\x00")

    def test_selector_canonical_bits_and_tail(self) -> None:
        for count in (0, 1, 2, 3, 7, 16, 96):
            selectors = [index & 7 for index in range(count)]
            packet = sm.pack_selectors(selectors)
            self.assertEqual(sm.unpack_selectors(packet, count), selectors)
        packet = bytearray(sm.pack_selectors([7]))
        packet[-1] |= 1
        with self.assertRaises(sm.ContractError):
            sm.unpack_selectors(bytes(packet), 1)


class ModelAndCoderTests(unittest.TestCase):
    def test_q16_exact_rows_and_roundtrip(self) -> None:
        for alphabet in (2, 4):
            roots, details = sm.generate_transformed_source(alphabet, 521, 17, 500 + alphabet, True)
            model = sm.fit_model(alphabet, roots, details)
            self.assertTrue(np.all(model.frequencies.sum(axis=2, dtype=np.uint64) == sm.Q16_TOTAL))
            self.assertTrue(np.all(model.frequencies >= 1))
            serialized = model.serialize()
            restored = sm.Q16TreeModel.deserialize(serialized)
            self.assertTrue(np.array_equal(restored.frequencies, model.frequencies))
            packet, meaningful = sm.encode_coefficients(model, roots, details)
            decoded_roots, decoded_details = sm.decode_coefficients(
                model, packet, roots.size, details.size
            )
            self.assertGreater(meaningful, 0)
            self.assertLessEqual(meaningful, 8 * len(packet))
            self.assertTrue(np.array_equal(decoded_roots, roots))
            self.assertTrue(np.array_equal(decoded_details, details))

    def test_model_corruption_rejected(self) -> None:
        roots, details = sm.generate_transformed_source(2, 32, 5, 19, True)
        model = sm.fit_model(2, roots, details)
        packet = bytearray(model.serialize())
        packet[sm.MODEL_HEADER_BYTES] ^= 1
        with self.assertRaises(sm.ContractError):
            sm.Q16TreeModel.deserialize(bytes(packet))

    def test_structured_control_finite_early_gate(self) -> None:
        for alphabet in (2, 4):
            train_roots, train_details = sm.generate_transformed_source(
                alphabet, 8192, 17, 117 * alphabet, True
            )
            model = sm.fit_model(alphabet, train_roots, train_details)
            structured_roots, structured_details = sm.generate_transformed_source(
                alphabet, 8192, 17, 991 * alphabet, True
            )
            control_roots, control_details = sm.generate_transformed_source(
                alphabet, 8192, 17, 991 * alphabet, False
            )
            _, structured_bits = sm.encode_coefficients(
                model, structured_roots, structured_details
            )
            # Give the iid control its own matched auxiliary model.  The gate is
            # therefore not manufactured by using the structured model on it.
            control_train_roots, control_train_details = sm.generate_transformed_source(
                alphabet, 8192, 17, 117 * alphabet, False
            )
            control_model = sm.fit_model(alphabet, control_train_roots, control_train_details)
            _, control_bits = sm.encode_coefficients(control_model, control_roots, control_details)
            symbols = structured_roots.size + structured_details.size
            gap = (control_bits - structured_bits) / symbols
            self.assertGreater(gap, 0.15)
            self.assertGreaterEqual(control_bits / symbols, 0.98 * math.log2(alphabet))


class ContainerTests(unittest.TestCase):
    @staticmethod
    def fixture(alphabet: int = 2, lanes: int = 17, vectors: int = 521, experts: int = 3):
        permutation = sm.deterministic_permutation(lanes, 447)
        selectors = sm.deterministic_selectors(lanes, 448)
        train = sm.synthesize_leaves(
            alphabet, 2048, lanes, 1001, True, permutation, selectors
        )
        train_lifted = sm.lift_forward(train, alphabet, permutation, selectors)
        model = sm.fit_model(alphabet, train_lifted.roots, sm.flatten_details(train_lifted))
        leaves = [
            sm.synthesize_leaves(
                alphabet, vectors, lanes, 5000 + index, True, permutation, selectors
            )
            for index in range(experts)
        ]
        packet = sm.build_container(
            model, leaves, [permutation] * experts, [selectors] * experts
        )
        return packet, leaves

    def test_independent_decode_and_byte_reencode(self) -> None:
        for alphabet, lanes in ((2, 1), (2, 17), (4, 18), (4, 97)):
            packet, leaves = self.fixture(alphabet, lanes, vectors=257, experts=3)
            self.assertEqual(sm.reencode_container(packet), packet)
            expected = [sm.leaf_digest(value) for value in leaves]
            receipt, independent, rebuilt = independent_decoder.verify_decode_reencode(
                packet, expected
            )
            self.assertEqual(rebuilt, packet)
            self.assertEqual(receipt["status"], "PASS_INDEPENDENT_DECODE_REENCODE")
            self.assertFalse(receipt["source_gain_claim"])
            self.assertTrue(
                all(np.array_equal(a, b) for a, b in zip(independent, leaves, strict=True))
            )

    def test_physical_and_cold_ledger(self) -> None:
        packet, _ = self.fixture(2, 97, vectors=1024, experts=8)
        ledger = sm.physical_ledger(packet)
        self.assertEqual(ledger["container_bytes"], len(packet))
        self.assertTrue(ledger["cold_below_two"])
        self.assertLess(ledger["max_cold_amplification"], 2.0)
        self.assertIn("not a model-weight", ledger["scope"])

    def test_truncation_and_corruption_fail_closed(self) -> None:
        packet, _ = self.fixture(2, 18, vectors=257, experts=3)
        corruptions: list[bytes] = []
        corruptions.append(packet[:-1])
        global_tail = bytearray(packet)
        global_tail[sm.GLOBAL_STRUCT.size + 3 * sm.DIRECTORY_ENTRY.size + 5] = 1
        corruptions.append(bytes(global_tail))
        parsed = sm.parse_container(packet)
        model_tail = bytearray(packet)
        model_tail[sm.GLOBAL_HEADER_BYTES + parsed.model_packet_bytes] = 1
        corruptions.append(bytes(model_tail))
        frame_tail = bytearray(packet)
        frame_offset, frame_length = parsed.directory[0]
        row = sm.parse_frame_header(packet[frame_offset : frame_offset + frame_length])
        frame_tail[frame_offset + row["logical_bytes"]] = 1
        corruptions.append(bytes(frame_tail))
        frame_body = bytearray(packet)
        frame_body[frame_offset + sm.FRAME_HEADER_BYTES] ^= 0x80
        corruptions.append(bytes(frame_body))
        for corrupted in corruptions:
            with self.assertRaises((sm.ContractError, ValueError, OverflowError)):
                sm.parse_container(corrupted)
            with self.assertRaises((independent_decoder.IndependentError, ValueError, OverflowError)):
                independent_decoder.parse_container(corrupted)

    def test_selector_tail_and_factoradic_rank_fail_after_valid_crc(self) -> None:
        packet, _ = self.fixture(2, 18, vectors=257, experts=2)
        parsed = sm.parse_container(packet)
        for mode in ("selector_tail", "factoradic_rank"):
            mutated = bytearray(packet)
            frame_offset, frame_length = parsed.directory[0]
            frame = bytearray(mutated[frame_offset : frame_offset + frame_length])
            row = sm.parse_frame_header(bytes(frame))
            body = bytearray(frame[sm.FRAME_HEADER_BYTES : row["logical_bytes"]])
            if mode == "selector_tail":
                # lanes=18 -> 51 selector bits, leaving five canonical zero bits.
                selector_last = row["permutation_bytes"] + row["selector_bytes"] - 1
                body[selector_last] |= 1
            else:
                body[: row["permutation_bytes"]] = b"\xff" * row["permutation_bytes"]
            body_crc = __import__("zlib").crc32(body) & 0xFFFFFFFF
            zero = sm._frame_header(
                0,
                row["alphabet"],
                row["lanes"],
                row["vectors"],
                row["permutation_bytes"],
                row["selector_bytes"],
                row["payload_bytes"],
                row["meaningful_bits"],
                row["logical_bytes"],
                row["padded_bytes"],
                row["leaf_symbols"],
                body_crc,
                0,
            )
            header_crc = __import__("zlib").crc32(zero) & 0xFFFFFFFF
            header = sm._frame_header(
                0,
                row["alphabet"],
                row["lanes"],
                row["vectors"],
                row["permutation_bytes"],
                row["selector_bytes"],
                row["payload_bytes"],
                row["meaningful_bits"],
                row["logical_bytes"],
                row["padded_bytes"],
                row["leaf_symbols"],
                body_crc,
                header_crc,
            )
            rebuilt_frame = header + bytes(body) + bytes(row["padded_bytes"] - row["logical_bytes"])
            mutated[frame_offset : frame_offset + frame_length] = rebuilt_frame
            with self.assertRaises(sm.ContractError):
                sm.decode_container(bytes(mutated))
            with self.assertRaises(independent_decoder.IndependentError):
                independent_decoder.verify_decode_reencode(bytes(mutated))


class SyntheticMarginalTests(unittest.TestCase):
    def test_long_range_source_and_control_have_matched_uniform_marginals(self) -> None:
        for alphabet in (2, 4):
            lanes = 17
            vectors = 32768
            permutation = sm.deterministic_permutation(lanes, 812)
            selectors = sm.deterministic_selectors(lanes, 813)
            structured = sm.synthesize_leaves(
                alphabet, vectors, lanes, 771, True, permutation, selectors
            )
            control = sm.synthesize_leaves(
                alphabet, vectors, lanes, 771, False, permutation, selectors
            )
            for values in (structured, control):
                frequencies = np.bincount(values.reshape(-1), minlength=alphabet) / values.size
                self.assertLess(float(np.max(np.abs(frequencies - 1.0 / alphabet))), 0.01)


class CuPyTests(unittest.TestCase):
    def test_mandatory_gpu_search_and_exact_equality(self) -> None:
        required = os.environ.get("SILT_REQUIRE_CUPY_TEST") == "1"
        require_5090 = os.environ.get("SILT_REQUIRE_RTX5090") == "1"
        try:
            from cupy_backend import CuPyRequiredError, search_metadata_cupy

            hidden = 0x2221
            lanes = 17
            permutation = sm.deterministic_permutation(lanes, hidden)
            selectors = sm.deterministic_selectors(lanes, hidden ^ 0x5A17)
            train = sm.synthesize_leaves(2, 512, lanes, 1801, True, permutation, selectors)
            validation = sm.synthesize_leaves(2, 257, lanes, 1802, True, permutation, selectors)
            result = search_metadata_cupy(
                train,
                validation,
                2,
                [0x1111, hidden, 0x3331],
                require_rtx_5090=require_5090,
            )
        except (ImportError, CuPyRequiredError) as exc:
            if required:
                raise
            self.skipTest(str(exc))
        self.assertTrue(result.telemetry["cpu_cupy_selected_coefficients_equal"])
        self.assertTrue(result.telemetry["cupy_inverse_roundtrip_equal"])
        self.assertTrue(result.telemetry["telemetry_values_measured_not_inferred"])
        for key in ("h2d_ms", "kernel_ms", "d2h_ms", "wall_ms"):
            self.assertGreaterEqual(float(result.telemetry[key]), 0.0)
        self.assertGreater(int(result.telemetry["sampled_peak_device_used_bytes"]), 0)
        self.assertEqual(len(result.candidate_rows), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

