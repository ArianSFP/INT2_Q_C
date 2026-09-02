#!/usr/bin/env python3
"""Independent semantic tests and attacks for the frozen global-RM gate."""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from independent_auth import (EXPECTED_EXTERNAL_PINS, EXPECTED_SOURCE_ROOT,
                              authenticate_external, authenticate_source)


AUDIT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = AUDIT_DIR.parents[1]
SOURCE = Path(os.environ.get(
    "STRATA_RM_GLOBAL_SOURCE",
    str(DEFAULT_REPO / "research/strata_rm_global_swap_v0"))).resolve()
EXTERNAL_ROOT = Path(os.environ.get(
    "STRATA_RM_GLOBAL_EXTERNAL_ROOT", str(DEFAULT_REPO.parent))).resolve()
SOURCE_AUTH = authenticate_source(SOURCE)
EXTERNAL_AUTH = authenticate_external(EXTERNAL_ROOT)
sys.path.insert(0, str(SOURCE))

from coset_contract import CURRENT_RANDOM, ZERO, HeldCosetFork, frozen_external  # noqa: E402
from result_contract import REQUIRED_PACKET_FIELDS, validate_independent_result  # noqa: E402
from rm_order import (TARGET_N, bit_reverse_indices, generated_row,  # noqa: E402
                      rm_full_order_numpy)
from swap_adapter import install, make_replacement  # noqa: E402


def fabricated_receipt() -> dict:
    n = 1 << 20
    packet_bytes = 327_680
    levels = []
    for k in (1, 12_345, n // 2, n - 7, n, n):
        # Only the name convention is material to the schema validator.
        name = "RM-ordered truncated polar"
        if k == 1:
            name = "RM(0,20)"
        elif k == n:
            name = "RM(20,20)"
        levels.append({"reference_bec_k": k, "rm_ordered_k": k, "set_name": name})
    return {
        "schema": "strata-rm-global-swap-v0-independent-physical-result",
        "external_pins": dict(EXPECTED_EXTERNAL_PINS),
        "candidate": "RM-ordered truncated polar",
        "coset": "current_random",
        "rate_basis": "literal_full_packet_bytes_plus_charged_shared_bytes",
        "independent_decoder_source_sha256": "x",
        "independent_decode_complete": True,
        "causal_probabilities_regenerated": True,
        "packet_consumed_exactly": True,
        "canonical_reencode_byte_identical": True,
        "source_domain_score_from_decoded_packet": True,
        "overlap_receipt_used_for_rd": False,
        "charged_packet_fields": sorted(REQUIRED_PACKET_FIELDS),
        "blocks": [{
            "n": n,
            "levels": levels,
            "literal_packet_bytes": packet_bytes,
            "literal_packet_sha256": "z" * 64,
            "canonical_reencode_sha256": "z" * 64,
        }],
        "charged_shared_bytes": 0,
        "total_original_weights": n,
        "total_physical_bytes": packet_bytes,
        "actual_physical_bpw": 2.5,
        "selected_count_used_as_rate": False,
    }


def independent_bit_reverse(value: int, width: int) -> int:
    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


class AuthenticationTests(unittest.TestCase):
    def test_exact_source_and_external_roots(self) -> None:
        self.assertEqual(SOURCE_AUTH["source_root_sha256"], EXPECTED_SOURCE_ROOT)
        self.assertTrue(EXTERNAL_AUTH["status"].startswith("PASS_"))

    def test_member_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            clone = Path(td) / "source"
            shutil.copytree(SOURCE, clone)
            with (clone / "rm_order.py").open("ab") as stream:
                stream.write(b"\n# hostile mutation\n")
            with self.assertRaises(ValueError):
                authenticate_source(clone)

    def test_pinned_external_orientation_and_current_coset_semantics(self) -> None:
        def function_text(path: Path, name: str) -> str:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            node = next(item for item in tree.body
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name == name)
            return ast.get_source_segment(text, node) or ""

        base = EXTERNAL_ROOT / "agent_polaris_qwen_rht_encoder.py"
        bec = EXTERNAL_ROOT / "bg_codec_bec_encoder.py"
        decoder = EXTERNAL_ROOT / "strata_v2_klt_mixed_independent_auditor_v1.py"
        sc = function_text(base, "sc_encode_ratio").replace(" ", "")
        trial = function_text(base, "run_trial")
        flags = function_text(bec, "bec_flags").replace(" ", "")
        decoded = function_text(decoder, "decode_one_block")
        self.assertIn("external_u=u[reverse].copy()", sc)
        self.assertIn("x_bit = polar_transform(chosen.external_u)", trial)
        self.assertIn("external[order[:keep]]=0", flags)
        self.assertIn("external[reverse].copy()", flags)
        self.assertIn("sc_seed + 1_000_003 * level", decoded)
        self.assertIn("previous += (1 << level_index) * x_bit.astype(np.int16)", decoded)

    def test_unmanifested_import_directory_exposes_producer_verifier_gap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            clone = Path(td) / "source"
            shutil.copytree(SOURCE, clone)
            spoof = clone / "cupy"
            spoof.mkdir()
            (spoof / "__init__.py").write_text("__version__ = 'spoof'\n", encoding="utf-8")
            # The producer verifier checks only top-level files, so it accepts
            # this importable unmanifested directory.  The independent wrapper
            # must reject it.
            completed = subprocess.run(
                [sys.executable, "-B", str(clone / "verify_source.py"),
                 "--package", str(clone)],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with self.assertRaises(ValueError):
                authenticate_source(clone)


class OrientationTests(unittest.TestCase):
    def test_internal_phase_orientation_against_kronecker_generator(self) -> None:
        kernel = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)
        for m in range(1, 8):
            n = 1 << m
            generator = np.asarray([[1]], dtype=np.uint8)
            for _ in range(m):
                generator = np.kron(generator, kernel) & np.uint8(1)
            reverse = np.asarray(
                [independent_bit_reverse(i, m) for i in range(n)], dtype=np.int64)
            self.assertTrue(np.array_equal(bit_reverse_indices(n), reverse))
            for phase in range(n):
                expected = generator[reverse[phase]]
                actual = generated_row(phase, n)
                self.assertTrue(np.array_equal(actual, expected))
                self.assertEqual(int(np.count_nonzero(actual)), 1 << phase.bit_count())

    def test_normative_order_is_exact_and_ties_are_explicit(self) -> None:
        for m in range(1, 13):
            n = 1 << m
            expected = np.asarray(sorted(range(n), key=lambda i: (-i.bit_count(), i)))
            self.assertTrue(np.array_equal(rm_full_order_numpy(n), expected))


class GlobalCountAndIntegrationTests(unittest.TestCase):
    def test_actual_k_is_preserved_at_both_production_lengths(self) -> None:
        for n in TARGET_N:
            ks = (0, 1, n // 17, n // 2, n - 1, n)
            calls = []

            def reference(_repo, called_n, capacities):
                calls.append((called_n, tuple(capacities)))
                rows = []
                for k in ks:
                    row = np.ones(n, dtype=np.uint8)
                    row[:k] = 0
                    rows.append(row)
                return rows

            replacement = make_replacement(reference)
            result = replacement(None, n, iter([0.0] * 6))
            self.assertEqual(calls, [(n, (0.0,) * 6)])
            self.assertEqual([int(np.count_nonzero(row == 0)) for row in result], list(ks))
            order = rm_full_order_numpy(n)
            for k, row in zip(ks, result, strict=True):
                self.assertTrue(np.all(row[order[:k]] == 0))
                self.assertTrue(np.all(row[order[k:]] == 1))

    def test_install_enforces_n20_n21_before_calling_reference(self) -> None:
        calls = []

        def reference(_repo, n, _capacities):
            calls.append(n)
            return [np.ones(n, dtype=np.uint8) for _ in range(6)]

        base = types.SimpleNamespace(reliability_freeze_flags=object())
        install(base, reference)
        for n in (1 << 18, 1 << 19, 1 << 22):
            with self.assertRaises(ValueError):
                base.reliability_freeze_flags(None, n, [0.0] * 6)
        self.assertEqual(calls, [])

    def test_install_accepts_unauthenticated_objects_and_is_not_launch_authority(self) -> None:
        fake_base = types.SimpleNamespace(reliability_freeze_flags="not-pinned")
        fake_reference = lambda *_args: []
        row = install(fake_base, fake_reference)
        self.assertTrue(row["installed"])
        self.assertEqual(fake_base.reliability_freeze_flags.__name__,
                         "rm_ordered_truncated_polar_flags")
        # This acceptance is an expected boundary exposure: an authenticated
        # integration launcher must bind object identity and final hook state.


class CosetTests(unittest.TestCase):
    def test_current_random_matches_independent_rng_and_zero_is_held(self) -> None:
        for n, seed, level in ((1 << 10, 0, 1), (1 << 12, 123, 3),
                               (1 << 13, 0xFFFFFFFF, 6)):
            expected = np.random.default_rng(seed + 1_000_003 * level).integers(
                0, 2, size=n, dtype=np.uint8)
            self.assertTrue(np.array_equal(
                frozen_external(n, seed, level, CURRENT_RANDOM), expected))
        with self.assertRaises(HeldCosetFork):
            frozen_external(1 << 12, 123, 1, ZERO)


class PhysicalContractBoundaryTests(unittest.TestCase):
    def test_fabricated_no_packet_receipt_is_accepted_as_schema_only(self) -> None:
        receipt = fabricated_receipt()
        result = validate_independent_result(receipt)
        self.assertTrue(result["passed"])
        # No packet bytes, source identity, reconstruction, distortion, F,
        # controls, routed reads, model shapes, or universal-SwiGLU evidence
        # are supplied.  The validator therefore cannot authenticate a result.
        for absent in ("packet_path", "source_sha256", "relative_mse", "F",
                       "gaussian_control", "cold_read_amplification",
                       "universal_swiglu_moe"):
            self.assertNotIn(absent, receipt)

    def test_declared_external_pins_are_not_live_external_authority(self) -> None:
        receipt = fabricated_receipt()
        self.assertNotIn("external_root", receipt)
        self.assertTrue(validate_independent_result(receipt)["passed"])

    def test_canonical_reencode_is_only_a_declared_hash_equality(self) -> None:
        receipt = fabricated_receipt()
        self.assertEqual(receipt["blocks"][0]["literal_packet_sha256"], "z" * 64)
        self.assertTrue(validate_independent_result(receipt)["passed"])
        bad = copy.deepcopy(receipt)
        bad["blocks"][0]["canonical_reencode_sha256"] = "y" * 64
        with self.assertRaises(ValueError):
            validate_independent_result(bad)

    def test_literal_byte_arithmetic_and_target_interval_are_enforced(self) -> None:
        receipt = fabricated_receipt()
        bad = copy.deepcopy(receipt)
        bad["actual_physical_bpw"] = 2.49
        with self.assertRaises(ValueError):
            validate_independent_result(bad)
        bad = copy.deepcopy(receipt)
        bad["blocks"][0]["literal_packet_bytes"] -= 1
        bad["total_physical_bytes"] -= 1
        bad["actual_physical_bpw"] = 8 * bad["total_physical_bytes"] / (1 << 20)
        self.assertTrue(validate_independent_result(bad)["passed"])


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _FakePool:
    def free_all_blocks(self):
        return None


class _FakeDevice:
    id = 0


class CuPyAuthorityTests(unittest.TestCase):
    def test_cupy_smoke_accepts_a_numpy_facade(self) -> None:
        fake = types.ModuleType("cupy")
        fake.__version__ = "fabricated-cupy"
        fake.uint8, fake.uint32, fake.uint64, fake.int64 = (
            np.uint8, np.uint32, np.uint64, np.int64)
        fake.arange = np.arange
        fake.argsort = np.argsort
        fake.ones = np.ones
        fake.asnumpy = np.asarray
        fake.count_nonzero = lambda value: _FakeScalar(np.count_nonzero(value))
        fake.get_default_memory_pool = lambda: _FakePool()
        fake.cuda = types.SimpleNamespace(
            Device=lambda: _FakeDevice(),
            runtime=types.SimpleNamespace(
                getDeviceProperties=lambda _device: {"name": b"fabricated CPU facade"}),
            Stream=types.SimpleNamespace(null=types.SimpleNamespace(synchronize=lambda: None)),
        )
        module_name = "audited_cupy_order_smoke_spoof_probe"
        spec = importlib.util.spec_from_file_location(module_name, SOURCE / "cupy_order_smoke.py")
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(sys.modules, {"cupy": fake}):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            module.TARGET_N = (1 << 10,)
            output = Path(td) / "spoof.json"
            with mock.patch.object(sys, "argv", [module_name, "--output", str(output)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                module.main()
            row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(row["backend"]["cupy_version"], "fabricated-cupy")
        self.assertEqual(row["status"], "PASS_CUPY_GLOBAL_CONSTRUCTION__HOLD_PAYLOAD")
        # A fresh isolated real-CUDA audit is required; the producer smoke is
        # a functional smoke, not backend provenance.


if __name__ == "__main__":
    unittest.main(verbosity=2)
