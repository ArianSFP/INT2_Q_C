#!/usr/bin/env python3
"""Hostile source-only tests. No payload, network, NumPy, CuPy, or CUDA."""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).absolute().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load("uwfa_test_common", "uwfa_common.py")


class FrozenMathTests(unittest.TestCase):
    def test_standalone_threshold(self) -> None:
        expected = -0.5 * math.log2(0.8) - 0.008074080480766676
        self.assertAlmostEqual(common.STANDALONE_REQUIRED_SAVING_BPW, expected, places=15)
        self.assertAlmostEqual(expected, 0.15288996696291447, places=15)

    def test_complete_finite_bank_and_selector(self) -> None:
        bank = common.candidate_bank()
        self.assertEqual(len(bank), 150)
        self.assertEqual({row.states for row in bank}, {2, 4, 8, 16, 32, 64})
        self.assertEqual({row.reset_length for row in bank}, {32, 128, 512, 2048, 4096})
        self.assertEqual({row.topology for row in bank}, set(common.TOPOLOGIES))
        self.assertEqual([row.selector_ordinal for row in bank], list(range(150)))

    def test_transition_range_and_t0_convention(self) -> None:
        for candidate in common.candidate_bank():
            for state in range(candidate.states):
                for bit in (0, 1):
                    for position in (0, 1, candidate.reset_length - 1):
                        context = common.public_context(2, 32768, position)
                        observed = common.transition(candidate, state, bit, context, position)
                        self.assertGreaterEqual(observed, 0)
                        self.assertLess(observed, candidate.states)
        candidate = common.Candidate("suffix", 8, 32)
        bits = [1, 1, 1] + [0] * 29 + [1]
        levels = [0] * len(bits)
        base = [32768] * len(bits)
        frequencies = list(range(1, common.model_frequency_count(candidate) + 1))
        frequencies = [1 + (value % 65535) for value in frequencies]
        used = common.stream_frequencies_cpu(bits, levels, base, candidate, frequencies)
        context_zero = common.public_context(0, 32768, 0)
        self.assertEqual(used[0], frequencies[context_zero])
        self.assertEqual(used[32], frequencies[context_zero])

    def test_exact_integer_jeffreys_q16(self) -> None:
        self.assertEqual(common.q16_frequencies_from_counts([0, 0]), [32768])
        values = common.q16_frequencies_from_counts([100, 0, 0, 100, 7, 11])
        self.assertTrue(all(1 <= value <= 65535 for value in values))
        self.assertLess(values[0], 32768)
        self.assertGreater(values[1], 32768)


class ArithmeticAndModelTests(unittest.TestCase):
    def test_canonical_arithmetic_roundtrip_and_tamper(self) -> None:
        bits = [((index * 13) ^ (index >> 2)) & 1 for index in range(3000)]
        frequencies = [1 + ((index * 7919 + 31) % 65535) for index in range(len(bits))]
        payload, logical_bits = common.arithmetic_encode_binary(bits, frequencies)
        decoded = common.arithmetic_decode_binary(payload, logical_bits, len(bits), lambda index: frequencies[index])
        self.assertEqual(decoded, bits)
        corrupted = bytearray(payload)
        corrupted[0] ^= 0x80
        with self.assertRaises(common.ContractError):
            common.arithmetic_decode_binary(bytes(corrupted), logical_bits, len(bits), lambda index: frequencies[index])

    def test_unifilar_model_packet_and_decode(self) -> None:
        bits = [((index * index + 3 * index + 1) >> 1) & 1 for index in range(5000)]
        levels = [(index * 5) % 6 for index in range(len(bits))]
        base = [1 + ((index * 997) % 65535) for index in range(len(bits))]
        for topology in common.TOPOLOGIES:
            candidate = common.Candidate(topology, 8, 128)
            counts = common.count_stream_cpu(bits, levels, base, candidate)
            frequencies = common.q16_frequencies_from_counts(counts)
            packet = common.serialize_model(candidate, frequencies)
            recovered_candidate, recovered_frequencies = common.deserialize_model(packet)
            self.assertEqual(recovered_candidate, candidate)
            self.assertEqual(recovered_frequencies, frequencies)
            payload, logical_bits = common.encode_unifilar_stream(bits, levels, base, candidate, frequencies)
            decoded = common.decode_unifilar_stream(payload, logical_bits, levels, base, candidate, frequencies)
            self.assertEqual(decoded, bits)

    def test_model_overhead_and_packet_ledger_are_literal(self) -> None:
        candidate = common.Candidate("xor_sketch", 64, 4096)
        ledger = common.model_ledger(candidate)
        self.assertEqual(ledger["frequency_u16_values"], 64 * 384)
        self.assertEqual(ledger["physical_model_bytes"], 64 + 2 + 2 * 64 * 384)
        packet = common.packet_ledger(
            weights=28_311_552,
            current_object_bytes=8_847_360,
            immutable_global_bytes=1024,
            immutable_local_bytes=[1024] * 6,
            model_packet_bytes=ledger["physical_model_bytes"],
            stream_payload_bytes=[[1_000_000, 180_000] for _ in range(6)],
        )
        self.assertLess(packet["maximum_cold_read_amplification"], 2.0)
        self.assertTrue(packet["passes_rate_interval"])


class LongMemoryFixtureTests(unittest.TestCase):
    def test_cumulative_parity_beats_every_bounded_suffix(self) -> None:
        fixture = _load("uwfa_test_fixture", "fixture_long_memory.py")
        receipt = fixture.run_fixture()
        self.assertEqual(receipt["status"], "PASS_LONG_MEMORY_SEPARATION")
        self.assertGreater(receipt["exact_arithmetic_separation_bits_per_symbol"], 0.01)
        self.assertEqual(max(row["suffix_depth"] for row in receipt["suffix_cells"]), 6)


class SecurityAndLifecycleTests(unittest.TestCase):
    def test_strict_json(self) -> None:
        with self.assertRaisesRegex(common.ContractError, "duplicate JSON key"):
            common.strict_json_loads('{"x":1,"x":2}')
        for value in ("NaN", "Infinity", "1e9999"):
            with self.subTest(value=value), self.assertRaises(common.ContractError):
                common.strict_json_loads(value)

    def test_symlink_leaf_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.bin"
            leaf = root / "leaf.bin"
            target.write_bytes(b"secret")
            try:
                leaf.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            self.assertTrue(leaf.absolute().is_absolute())
            with self.assertRaisesRegex(common.ContractError, "symlink leaf forbidden"):
                common.HeldRegularFile(leaf.absolute()).open()

    def test_wrong_token_touches_neither_output_nor_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).absolute()
            output = root / "must-not-exist"
            absent = root / "absent.json"
            command = [
                sys.executable, "-B", str(PACKAGE / "stage0_census.py"),
                "--authorization", "WRONG",
                "--review-receipt", str(absent),
                "--stream-lock", str(absent),
                "--gaussian-control-lock", str(absent),
                "--output", str(output),
            ]
            run = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(run.returncode, 2, run.stderr + run.stdout)
            self.assertIn("project code and CuPy not imported", run.stderr + run.stdout)
            self.assertFalse(output.exists())

    def test_valid_token_reserves_output_before_missing_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).absolute()
            output = root / "reserved"
            absent = root / "absent.json"
            command = [
                sys.executable, "-B", str(PACKAGE / "stage0_census.py"),
                "--authorization", common.AUTHORIZATION,
                "--review-receipt", str(absent),
                "--stream-lock", str(absent),
                "--gaussian-control-lock", str(absent),
                "--output", str(output),
            ]
            run = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            self.assertNotEqual(run.returncode, 0)
            self.assertEqual({path.name for path in output.iterdir()}, {"RUN_STATE.json"})

    def test_cupy_import_is_textually_after_bootstrap_and_baseline(self) -> None:
        source = (PACKAGE / "stage0_census.py").read_text(encoding="utf-8")
        import_at = source.index("import cupy as cp")
        self.assertLess(source.index("bootstrap_source(Path(args.review_receipt))"), import_at)
        self.assertLess(source.index("source_panel = load_panel"), import_at)


class SealedPackageTests(unittest.TestCase):
    def test_native_verifier_and_tamper(self) -> None:
        command = [sys.executable, "-B", str(PACKAGE / "verify_source.py"), "--package", str(PACKAGE), "--compact"]
        run = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
        receipt = common.strict_json_loads(run.stdout)
        self.assertEqual(receipt["status"], "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY")
        verifier = _load("uwfa_test_verifier", "verify_source.py")
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "package"
            shutil.copytree(PACKAGE, clone)
            target = clone / "README.md"
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaises(Exception):
                verifier.verify_package(clone)


if __name__ == "__main__":
    unittest.main(verbosity=2)
