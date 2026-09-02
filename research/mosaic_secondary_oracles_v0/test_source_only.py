#!/usr/bin/env python3
"""Hostile source-only tests for MOSAIC secondary oracles v0."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import unittest
import zlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load(name: str, filename: str):
    payload = (ROOT / filename).read_bytes()
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__authenticated_sha256__ = hashlib.sha256(payload).hexdigest()
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


contract = load("mosaic_secondary_test_contract", "gate_contract.py")
recurrence = load("mosaic_secondary_test_recurrence", "gf2_recurrence.py")
oracles = load("mosaic_secondary_test_oracles", "residual_oracles.py")
fixture = load("mosaic_secondary_test_fixture", "run_source_free_fixture.py")


def reference_bm(values):
    sequence = [int(value) for value in values]
    n = len(sequence)
    connection = [0] * (n + 1)
    previous = [0] * (n + 1)
    connection[0] = previous[0] = 1
    complexity = 0
    shift = 1
    for index in range(n):
        discrepancy = sequence[index]
        for lag in range(1, complexity + 1):
            discrepancy ^= connection[lag] & sequence[index - lag]
        if discrepancy:
            held = connection.copy()
            for position in range(n + 1 - shift):
                connection[position + shift] ^= previous[position]
            if 2 * complexity <= index:
                complexity = index + 1 - complexity
                previous = held
                shift = 1
            else:
                shift += 1
        else:
            shift += 1
    packed = sum((connection[lag] & 1) << lag for lag in range(complexity + 1))
    return complexity, packed


class RecurrenceTests(unittest.TestCase):
    def test_bigint_berlekamp_massey_matches_reference(self):
        rng = np.random.default_rng(17191)
        for length in (1, 2, 3, 17, 64, 257):
            for _ in range(12):
                bits = rng.integers(0, 2, size=length, dtype=np.uint8).tolist()
                self.assertEqual(recurrence.berlekamp_massey_gf2(bits), reference_bm(bits))

    def test_known_lfsr_exact_replay_and_complexity(self):
        initial = (1, 0, 0, 1, 1)
        connection = 1 | (1 << 1) | (1 << 5)
        sequence = recurrence.generate_lfsr(initial, connection, 4096)
        complexity, observed = recurrence.berlekamp_massey_gf2(sequence)
        self.assertEqual(complexity, 5)
        self.assertEqual(recurrence.generate_lfsr(sequence[:complexity], observed, len(sequence)), sequence)

    def test_packet_roundtrip_canonicality_and_tamper(self):
        first = recurrence.generate_lfsr((1, 0, 1), 1 | (1 << 1) | (1 << 3), 256)
        second = recurrence.generate_lfsr((1, 1, 0, 1), 1 | (1 << 1) | (1 << 4), 256)
        labels = recurrence.labels_from_gray(first, second)
        packet, row = recurrence.encode_block(labels)
        self.assertEqual(recurrence.decode_block(packet), labels)
        self.assertEqual(recurrence.encode_block_no_check(labels), packet)
        self.assertTrue(all(item["mode"] == "lfsr" for item in row["planes"]))
        hostile = bytearray(packet)
        hostile[-1] ^= 1
        with self.assertRaisesRegex(recurrence.RecurrenceError, "CRC"):
            recurrence.decode_block(bytes(hostile))

    def test_random_planes_canonically_fall_back_to_raw(self):
        rng = np.random.default_rng(881)
        labels = rng.integers(0, 4, size=4096, dtype=np.uint8)
        packet, row = recurrence.encode_block(labels)
        self.assertEqual(recurrence.decode_block(packet), tuple(int(value) for value in labels))
        self.assertTrue(all(item["mode"] in {"raw", "lfsr"} for item in row["planes"]))
        self.assertLessEqual(sum(item["linear_complexity"] for item in row["planes"]), 4096)

    def test_physical_ledger_charges_headers_scales_offsets_pages_and_tail(self):
        labels = [0] * 256
        packet, _ = recurrence.encode_block(labels)
        component = recurrence.component_packet_bytes([packet] * 256)
        ledger = contract.physical_expert_ledger(
            weights=3 * 256 * 256,
            role_component_bytes=(component, component, component),
        )
        self.assertEqual(ledger["external_storage_reads"], 1)
        self.assertEqual(ledger["external_storage_refetches"], 0)
        self.assertEqual(ledger["physical_bytes"] % 4096, 0)
        self.assertTrue(ledger["passes_strict_cold_read_below_2x"])
        self.assertGreaterEqual(ledger["physical_rate_bpw"]["float"], 2.15)

    def test_literal_component_and_expert_roundtrip_match_ledger(self):
        length = 256
        first = recurrence.generate_lfsr(
            (1, 0, 0, 1, 1), 1 | (1 << 1) | (1 << 5), length
        )
        second = recurrence.generate_lfsr(
            (1, 1, 0, 0, 1, 0, 1), 1 | (1 << 2) | (1 << 7), length
        )
        labels = recurrence.labels_from_gray(first, second)
        blocks = tuple(
            labels[ordinal % length:] + labels[:ordinal % length]
            for ordinal in range(256)
        )
        scales = (b"\x00<",) * len(blocks)
        components = tuple(
            recurrence.encode_component(role, blocks, scales)
            for role in ("gate", "up", "down_transposed")
        )
        for role, packet in zip(("gate", "up", "down_transposed"), components, strict=True):
            decoded = recurrence.decode_component(packet)
            self.assertEqual(decoded["role"], role)
            self.assertEqual(decoded["label_blocks"], blocks)
            self.assertEqual(decoded["scale_f16le"], scales)
            self.assertEqual(
                recurrence.encode_component(role, decoded["label_blocks"], decoded["scale_f16le"]),
                packet,
            )
        weights = 3 * len(blocks) * length
        expert = recurrence.encode_expert(components, weights=weights)
        decoded_expert = recurrence.decode_expert(expert)
        ledger = contract.physical_expert_ledger(
            weights=weights,
            role_component_bytes=tuple(len(packet) for packet in components),
        )
        self.assertEqual(decoded_expert["component_packets"], components)
        self.assertEqual(recurrence.encode_expert(components, weights=weights), expert)
        self.assertEqual(len(expert), ledger["physical_bytes"])
        self.assertEqual(decoded_expert["physical_rate_bpw"], ledger["physical_rate_bpw"]["float"])
        self.assertTrue(ledger["passes_rate_interval"])
        hostile = bytearray(expert)
        hostile[-1] ^= 1
        with self.assertRaisesRegex(recurrence.RecurrenceError, "CRC|padding"):
            recurrence.decode_expert(bytes(hostile))
        fields = list(recurrence.EXPERT_HEADER.unpack_from(expert, 0))
        fields[4] += recurrence.PAGE_BYTES
        fields[9] = 0
        body = expert[recurrence.EXPERT_HEADER.size:] + bytes(recurrence.PAGE_BYTES)
        zero_header = recurrence.EXPERT_HEADER.pack(*fields)
        fields[9] = zlib.crc32(zero_header + body) & 0xFFFFFFFF
        overpadded = recurrence.EXPERT_HEADER.pack(*fields) + body
        with self.assertRaisesRegex(recurrence.RecurrenceError, "canonical padding"):
            recurrence.decode_expert(overpadded)

    def test_component_rejects_noncanonical_padding_and_role_reorder(self):
        labels = tuple(index & 3 for index in range(64))
        scales = (b"\x00<",) * 4
        blocks = (labels,) * 4
        components = tuple(
            recurrence.encode_component(role, blocks, scales)
            for role in ("gate", "up", "down_transposed")
        )
        hostile = bytearray(components[0])
        hostile[-1] = 1
        with self.assertRaisesRegex(recurrence.RecurrenceError, "CRC|padding"):
            recurrence.decode_component(bytes(hostile))
        fields = list(recurrence.COMPONENT_HEADER.unpack_from(components[0], 0))
        fields[8] += 64
        fields[9] = 0
        body = components[0][recurrence.COMPONENT_HEADER.size:] + bytes(64)
        zero_header = recurrence.COMPONENT_HEADER.pack(*fields)
        fields[9] = zlib.crc32(zero_header + body) & 0xFFFFFFFF
        overpadded = recurrence.COMPONENT_HEADER.pack(*fields) + body
        with self.assertRaisesRegex(recurrence.RecurrenceError, "canonical padding"):
            recurrence.decode_component(overpadded)
        with self.assertRaisesRegex(recurrence.RecurrenceError, "role order"):
            recurrence.encode_expert(
                (components[1], components[0], components[2]),
                weights=3 * len(blocks) * len(labels),
            )

    def test_block_rejects_valid_crc_noncanonical_raw_alias(self):
        labels = (0,) * 64
        packet, row = recurrence.encode_block(labels)
        self.assertTrue(all(item["mode"] == "lfsr" for item in row["planes"]))
        first, second = recurrence.gray_planes(labels)
        directory = (
            recurrence.PLANE.pack(recurrence.MODE_RAW, 0, 0)
            + recurrence.PLANE.pack(recurrence.MODE_RAW, 0, 0)
        )
        writer = recurrence.BitWriter()
        writer.write(first)
        writer.write(second)
        body = recurrence.HEADER.pack(
            recurrence.MAGIC, len(labels), 2, recurrence.VERSION
        ) + directory + writer.payload()
        alias = body + recurrence.CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)
        with self.assertRaisesRegex(recurrence.RecurrenceError, "canonical encoding"):
            recurrence.decode_block(alias)


class ResidualOracleTests(unittest.TestCase):
    def test_period_bank_is_strictly_non_dyadic_and_expected_dimension(self):
        self.assertTrue(all(period & (period - 1) for period in contract.NON_DYADIC_PERIODS))
        for period in contract.NON_DYADIC_PERIODS[:12]:
            rows = oracles.primitive_real_frequencies(period)
            phi = sum(math.gcd(value, period) == 1 for value in range(1, period))
            self.assertEqual(len(rows), phi)

    def test_ramanujan_basis_is_public_orthonormal_and_captures_period7(self):
        basis = oracles.build_ramanujan_basis(
            np,
            length=256,
            periods=contract.NON_DYADIC_PERIODS,
            maximum_columns=64,
        )
        self.assertLessEqual(basis["orthogonality_max_abs_error"], 5e-9)
        coordinate = np.arange(256, dtype=np.float64)
        residual = np.stack([
            np.sin(2.0 * math.pi * coordinate / 7.0 + 0.1 * block)
            for block in range(8)
        ])
        energy = float(np.sum(residual * residual)) / 0.04
        metrics = oracles.ramanujan_panel_metrics(np, residual, basis, source_energy=energy)
        self.assertLess(metrics["fixed_free_prefix_remaining_sse"], 1e-12 * metrics["input_sse"])
        self.assertFalse(metrics["ideal_waterfill_has_finite_backend"])
        self.assertLessEqual(metrics["source_selected_literal_bits_per_block"], 384)

    def test_ramanujan_random_source_does_not_gain_free_energy(self):
        rng = np.random.default_rng(107)
        basis = oracles.build_ramanujan_basis(
            np,
            length=256,
            periods=contract.NON_DYADIC_PERIODS,
            maximum_columns=64,
        )
        residual = rng.normal(size=(32, 256))
        energy = float(np.sum(residual * residual)) / 0.04
        metrics = oracles.ramanujan_panel_metrics(np, residual, basis, source_energy=energy)
        capture = 1.0 - metrics["fixed_free_prefix_remaining_sse"] / metrics["input_sse"]
        self.assertAlmostEqual(capture, 64 / 256, delta=0.035)

    def test_ar_pullback_charges_inverse_noise_amplification(self):
        self.assertAlmostEqual(oracles.inverse_noise_gain((0.0,), 256), 1.0)
        gain = oracles.inverse_noise_gain((-0.95,), 256)
        self.assertGreater(gain, 1.0)
        coordinate = np.arange(256, dtype=np.float64)
        residual = np.stack([
            np.sin(2.0 * math.pi * coordinate / 17.0 + 0.2 * block)
            for block in range(8)
        ])
        energy = float(np.sum(residual * residual)) / 0.04
        metrics = oracles.ar_hankel_panel_metrics(
            np,
            residual,
            source_energy=energy,
            orders=(1, 2, 4),
        )
        self.assertTrue(metrics["source_metric_pullback_is_not_ignored"])
        self.assertTrue(all(row["pullback_noise_amplification_charged"] for row in metrics["orders"]))
        self.assertTrue(all(row["descriptor_bits_per_block"] + row["innovation_bits_per_block"] == 384 for row in metrics["orders"]))

    def test_odd_affine_permutation_is_bijective(self):
        observed = oracles.odd_affine_permutation(4096, 10619863)
        self.assertEqual(len(set(observed)), 4096)
        self.assertNotEqual(observed, tuple(range(4096)))

    def test_permutation_and_gaussian_controls_preserve_declared_moments(self):
        rng = np.random.default_rng(331)
        residual = rng.normal(size=(8, 256)) * np.linspace(0.3, 1.7, 8)[:, None]
        residual += np.linspace(-0.2, 0.4, 8)[:, None]
        permuted = oracles.phase_destroyed_blocks(np, residual, 10619863)
        gaussian = oracles.moment_matched_gaussian_blocks(np, residual, 10619863)
        for observed in (permuted, gaussian):
            self.assertTrue(np.allclose(observed.mean(axis=1), residual.mean(axis=1), rtol=0.0, atol=2e-12))
            source_energy = np.sum((residual - residual.mean(axis=1, keepdims=True)) ** 2, axis=1)
            observed_energy = np.sum((observed - observed.mean(axis=1, keepdims=True)) ** 2, axis=1)
            self.assertTrue(np.allclose(observed_energy, source_energy, rtol=2e-12, atol=2e-12))
        self.assertFalse(np.array_equal(gaussian, residual))


class ContractTests(unittest.TestCase):
    def test_correct_coarse_capture_and_exact_rate_identity(self):
        self.assertAlmostEqual(contract.REQUIRED_COARSE_CAPTURE, 0.32387022205373717)
        ledger = contract.tactic_fine_ledger(160)
        self.assertEqual(ledger["total_rate_bpw"]["numerator"], 5)
        self.assertEqual(ledger["total_rate_bpw"]["denominator"], 2)
        self.assertEqual(ledger["innovation_bits_per_block"], 224)

    def test_residual_controls_forbidden_before_source_survival(self):
        killed = contract.residual_source_gate(
            input_sse=100.0,
            source_energy=2500.0,
            source_remaining_sse=80.0,
            descriptor_bits_per_block=64,
            controls=None,
        )
        self.assertEqual(killed["status"], "HARD_KILL_ABSOLUTE_SOURCE_MISSES_D_0P025")
        with self.assertRaisesRegex(contract.ContractError, "controls forbidden"):
            contract.residual_source_gate(
                input_sse=100.0,
                source_energy=2500.0,
                source_remaining_sse=80.0,
                descriptor_bits_per_block=64,
                controls={"permutation": 85.0, "gaussian": 84.0},
            )

    def test_control_excess_not_raw_oracle_gain_promotes(self):
        result = contract.residual_source_gate(
            input_sse=100.0,
            source_energy=3000.0,
            source_remaining_sse=60.0,
            descriptor_bits_per_block=64,
            controls={"permutation": 61.0, "gaussian": 60.5},
        )
        self.assertEqual(result["status"], "HARD_KILL_SOURCE_NOT_SPECIFIC_0P03_BPW")
        self.assertFalse(result["gains_may_be_added_to_separate_oracles"])

    def test_cupy_import_is_lazy(self):
        source = (ROOT / "cupy_backend.py").read_text(encoding="utf-8")
        prefix = source.split("def load_cupy", 1)[0]
        self.assertNotIn("import cupy", prefix)
        self.assertIn("import cupy as cp", source)

    def test_source_free_fixture_runs_without_payload_or_cuda(self):
        receipt = fixture.run()
        self.assertEqual(receipt["status"], "PASS_SOURCE_FREE_MECHANICS")
        self.assertFalse(receipt["qwen_payload_accessed"])
        self.assertFalse(receipt["coarse_payload_accessed"])
        self.assertFalse(receipt["cuda_initialized"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "schema": "mosaic-secondary-oracles-source-test-v0",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.wasSuccessful(),
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "cuda_initialized": False,
    }, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
