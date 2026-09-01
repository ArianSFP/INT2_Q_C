#!/usr/bin/env python3
"""Hostile source-only tests; no NumPy, CuPy, CUDA, payload or network."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

_COMMON_PATH = Path(__file__).resolve().with_name("mps_common.py")
_COMMON_SPEC = importlib.util.spec_from_file_location("mps_common", _COMMON_PATH)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError("cannot load sealed same-directory mps_common.py")
_COMMON_MODULE = importlib.util.module_from_spec(_COMMON_SPEC)
sys.modules["mps_common"] = _COMMON_MODULE
_COMMON_SPEC.loader.exec_module(_COMMON_MODULE)

from mps_common import (
    AUTHORIZATION,
    HIDDEN_DIMENSIONS,
    PERIODS,
    STANDALONE_REQUIRED_SAVING_BPW,
    CompletionLastOutput,
    ContractError,
    HeldRegularFile,
    arithmetic_encode_binary,
    context_count,
    hmm_model_ledger,
    packet_ledger,
    public_context,
    quantize_probability,
    quantize_simplex,
    sha256_bytes,
    strict_json_loads,
    suffix_model_ledger,
)


PACKAGE = Path(__file__).resolve().parent


class MathAndModelTests(unittest.TestCase):
    def test_threshold_is_standalone_not_composite(self) -> None:
        expected = -0.5 * math.log2(0.8) - 0.008074080480766676
        self.assertAlmostEqual(STANDALONE_REQUIRED_SAVING_BPW, expected, places=15)
        self.assertGreater(STANDALONE_REQUIRED_SAVING_BPW, 0.11356063457)

    def test_public_context_has_only_small_period(self) -> None:
        for period in PERIODS:
            seen = {
                public_context(level, frequency, position, period)
                for level in range(6)
                for frequency in (1, 8192, 32768, 65535)
                for position in range(8)
            }
            self.assertTrue(seen)
            self.assertLess(max(seen), context_count(period))
        with self.assertRaises(ContractError):
            public_context(0, 32768, 0, 16)

    def test_true_hmm_tensor_charge(self) -> None:
        for period in PERIODS:
            for chi in HIDDEN_DIMENSIONS:
                row = hmm_model_ledger(chi, period)
                self.assertEqual(row["tensor_u16_values"], chi + chi * chi + 6 * 16 * period * chi)
                self.assertEqual(row["physical_model_bytes"], 256 + 2 * row["tensor_u16_values"])
                self.assertGreater(row["transition_u16_values"], 0)

    def test_suffix_is_separate_local_subclass(self) -> None:
        suffix = suffix_model_ledger(8, 4)
        hidden = hmm_model_ledger(64, 4)
        self.assertEqual(suffix["states"], 256)
        self.assertIn("transition_u16_values", hidden)
        self.assertNotIn("transition_u16_values", suffix)

    def test_quantized_simplex_exact(self) -> None:
        for values in ([1.0, 1.0, 1.0], [0.0, 0.2, 0.8], [1e-20, 1.0]):
            row = quantize_simplex(values)
            self.assertEqual(sum(row), 65535)
            self.assertTrue(all(value >= 1 for value in row))
        self.assertEqual(quantize_probability(0.0), 1)
        self.assertEqual(quantize_probability(1.0), 65534)


class ArithmeticAndLedgerTests(unittest.TestCase):
    def test_exact_arithmetic_determinism_and_termination(self) -> None:
        bits = [0, 1, 1, 0, 1, 0, 0, 1] * 19
        frequencies = [32768, 50000, 1000, 65535, 1, 12000, 45000, 32768] * 19
        first = arithmetic_encode_binary(bits, frequencies)
        second = arithmetic_encode_binary(bits, frequencies)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), (first[1] + 7) // 8)
        self.assertGreater(first[1], 0)
        with self.assertRaises(ContractError):
            arithmetic_encode_binary([0, 1], [32768])

    def test_packet_is_under_two_x_and_charges_padding(self) -> None:
        row = packet_ledger(
            weights=28_311_552,
            current_object_bytes=8_847_360,
            immutable_global_bytes=1024,
            immutable_local_bytes=[1024] * 6,
            model_bytes=hmm_model_ledger(16, 2)["physical_model_bytes"],
            stream_payload_bytes=[[1_000_000, 180_000] for _ in range(6)],
        )
        self.assertGreaterEqual(row["physical_rate_bpw"], 2.15)
        self.assertLessEqual(row["physical_rate_bpw"], 2.5)
        self.assertLess(row["maximum_cold_read_amplification"], 2.0)
        self.assertGreater(row["net_physical_saving_bpw"], 0.0)
        self.assertEqual(len(row["cold_rows"]), 6)


class StrictJsonAndLifecycleTests(unittest.TestCase):
    def test_duplicate_and_nonfinite_json_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
            strict_json_loads('{"x":1,"x":2}')
        for payload in ("NaN", "Infinity", "1e9999"):
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                strict_json_loads(payload)

    def test_held_regular_file_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = (Path(directory) / "trace.bin").resolve()
            path.write_bytes(b"trace")
            with HeldRegularFile(path, 5, sha256_bytes(b"trace")) as held:
                self.assertEqual(held.read_all(), b"trace")
                held.verify_stable()
            with self.assertRaises(ContractError):
                HeldRegularFile(path, 5, "0" * 64).open()

    def test_symlink_input_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = (root / "target").resolve()
            link = (root / "link").resolve()
            target.write_bytes(b"x")
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with self.assertRaises(ContractError):
                HeldRegularFile(link).open()

    def test_completion_is_exclusive_and_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = (Path(directory) / "out").resolve()
            with CompletionLastOutput(output) as writer:
                member = writer.write_new("result.json", b"{}\n")
                writer.complete([member], "a" * 64)
            self.assertEqual({path.name for path in output.iterdir()}, {"RUN_STATE.json", "result.json", "COMPLETE.json"})
            with self.assertRaises(ContractError):
                CompletionLastOutput(output).__enter__()

    def test_wrong_token_rejects_before_output_or_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
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
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            run = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=20, check=False)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("CuPy not imported", run.stderr + run.stdout)
            self.assertFalse(output.exists())

    def test_valid_token_reserves_output_before_missing_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "reserved"
            absent = root / "absent.json"
            command = [
                sys.executable, "-B", str(PACKAGE / "stage0_census.py"),
                "--authorization", AUTHORIZATION,
                "--review-receipt", str(absent),
                "--stream-lock", str(absent),
                "--gaussian-control-lock", str(absent),
                "--output", str(output),
            ]
            run = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
            self.assertNotEqual(run.returncode, 0)
            self.assertTrue(output.is_dir())
            self.assertEqual({path.name for path in output.iterdir()}, {"RUN_STATE.json"})


class FrozenPackageTests(unittest.TestCase):
    def test_native_verifier(self) -> None:
        command = [sys.executable, "-B", str(PACKAGE / "verify_source.py"), "--package", str(PACKAGE), "--compact"]
        run = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
        receipt = strict_json_loads(run.stdout)
        self.assertEqual(receipt["status"], "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY")

    def test_current_package_and_tamper(self) -> None:
        from verify_source import verify_package

        self.assertEqual(verify_package(PACKAGE)["status"], "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY")
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "package"
            shutil.copytree(PACKAGE, clone)
            target = clone / "README.md"
            target.write_bytes(target.read_bytes() + b"x")
            with self.assertRaisesRegex(ContractError, "member (bytes|hash)"):
                verify_package(clone)


if __name__ == "__main__":
    unittest.main(verbosity=2)
