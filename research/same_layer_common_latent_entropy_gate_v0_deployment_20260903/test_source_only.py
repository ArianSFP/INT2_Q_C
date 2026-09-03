from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import common_latent_core as core
import cupy_worker as worker_core
import run_gate
import verify_source
from source_free_fixture import (
    binary_objective_counterexample,
    make_common_label_fixture,
    make_quantizer_fixture,
)


class CoreTests(unittest.TestCase):
    def test_down_transpose_canonicalization(self):
        raw = np.arange(15, dtype=np.float32).reshape(5, 3)
        got = core.canonicalize_role_cpu(raw, "down", 3, 5)
        self.assertEqual(got.shape, (3, 5))
        np.testing.assert_array_equal(got, raw.T)
        self.assertTrue(got.flags.c_contiguous)

    def test_scale_bits_replay_and_boundary_labels(self):
        values = make_quantizer_fixture(32)
        first = core.quantize_canonical_cpu(values, 32)
        second = core.quantize_canonical_cpu(values.copy(), 32)
        np.testing.assert_array_equal(first.scale_u16, second.scale_u16)
        np.testing.assert_array_equal(first.labels, second.labels)
        replay = first.scale_u16.view(np.float16).astype(np.float64)
        for block, scale in enumerate(replay):
            lo, hi = block * 32, (block + 1) * 32
            q = first.labels.reshape(-1)[lo:hi]
            x = values.reshape(-1)[lo:hi].astype(np.float64)
            t = core.THRESHOLD_RMS * scale
            expected = np.where(x < -t, 0, np.where(x < 0, 1, np.where(x <= t, 2, 3)))
            np.testing.assert_array_equal(q, expected)

    def test_modal_ties_choose_low_symbol(self):
        q = np.asarray([[0, 3], [1, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(
            core.modal_common_latent_cpu(q, 4), np.asarray([0, 2], dtype=np.uint8)
        )

    def test_binary_favorable_and_charged_objectives_are_distinct(self):
        labels = binary_objective_counterexample()
        favorable = core.score_labels_cpu(labels, 2, selection_objective="favorable")
        charged = core.score_labels_cpu(labels, 2, selection_objective="charged")
        self.assertEqual(favorable["planes"], [0, 0])
        self.assertEqual(charged["planes"], [0, 1])
        self.assertLess(favorable["conditional_data_bits"], charged["conditional_data_bits"])
        self.assertLess(charged["common_two_part_bits"], favorable["common_two_part_bits"])
        self.assertEqual(len(favorable["binary_plane_candidate_scores"]), 4)

    def test_hostile_count_summary_rejected(self):
        labels = make_common_label_fixture(3, 97)
        summary = core.summarize_counts_cpu(labels, 2, (0, 1))
        bad = dict(summary)
        conditional = summary["conditional_counts"].copy()
        source = np.argwhere(conditional[0, 0, 0] > 0)[0, 0]
        conditional[0, 0, 0, source] -= 1
        conditional[0, 0, 1, source] += 1
        bad["conditional_counts"] = conditional
        with self.assertRaisesRegex(core.GateError, "conditional state total"):
            core.score_count_summary(bad)

    def test_scrambles_are_bijective_deterministic_and_marginal_preserving(self):
        labels = make_common_label_fixture(5, 257)
        one = core.coordinate_scramble_cpu(labels, core.CONTROL_SEEDS[0])
        two = core.coordinate_scramble_cpu(labels, core.CONTROL_SEEDS[0])
        np.testing.assert_array_equal(one, two)
        self.assertFalse(np.array_equal(one, labels))
        for expert in range(5):
            for role in range(2):
                np.testing.assert_array_equal(
                    np.bincount(one[expert, role], minlength=4),
                    np.bincount(labels[expert, role], minlength=4),
                )
                a, b = core.affine_permutation_parameters(257, core.CONTROL_SEEDS[0], expert, role)
                indices = (a * np.arange(257) + b) % 257
                self.assertEqual(np.unique(indices).size, 257)

    def test_exact_qwen_page_anchors_and_page_boundary(self):
        e, n = 16, 768 * 2048
        private = [780_000] * e
        binary = core.physical_page_envelope(
            expert_count=e,
            coordinates_per_role=n,
            latent_bits_per_coordinate=1,
            requested_rate=Fraction(43, 20),
            common_model_bits=45,
            private_required_bytes=private,
        )
        self.assertEqual(binary["total_pages"], 3303)
        self.assertEqual(binary["actual_rate_fraction"], "1101/512")
        self.assertEqual(binary["common_pages"], 98)
        self.assertTrue(binary["strictly_below_2x"])
        at_cap = core.physical_page_envelope(
            expert_count=e,
            coordinates_per_role=n,
            latent_bits_per_coordinate=2,
            requested_rate=Fraction(5, 2),
            common_model_bits=127,
            private_required_bytes=private,
        )
        self.assertEqual(at_cap["total_pages"], 3840)
        self.assertEqual(at_cap["actual_rate_fraction"], "5/2")
        self.assertEqual(at_cap["common_pages"], 194)
        no_model = core.physical_page_envelope(
            expert_count=e,
            coordinates_per_role=n,
            latent_bits_per_coordinate=1,
            requested_rate=Fraction(43, 20),
            common_model_bits=0,
            private_required_bytes=private,
        )
        one_bit = core.physical_page_envelope(
            expert_count=e,
            coordinates_per_role=n,
            latent_bits_per_coordinate=1,
            requested_rate=Fraction(43, 20),
            common_model_bits=1,
            private_required_bytes=private,
        )
        self.assertEqual(no_model["common_pages"], 97)
        self.assertEqual(one_bit["common_pages"], 98)

    def test_exact_two_x_and_padding_attack_fail(self):
        equality = core.physical_page_envelope(
            expert_count=2,
            coordinates_per_role=16384,
            latent_bits_per_coordinate=1,
            requested_rate=Fraction(5, 2),
            common_model_bits=0,
            private_required_bytes=[4096, 2048],
        )
        self.assertFalse(equality["strictly_below_2x"])
        self.assertEqual(equality["max_amplification"], 2.0)
        self.assertEqual(equality["status"], "FAIL_CAPACITY_OR_STRICT_READ_AMPLIFICATION")
        padded = core.physical_page_envelope(
            expert_count=2,
            coordinates_per_role=16384,
            latent_bits_per_coordinate=1,
            requested_rate=Fraction(5, 2),
            common_model_bits=0,
            private_required_bytes=[1, 1],
        )
        self.assertLess(max(Fraction(x) for x in padded["amplification_physical_fraction"]), 2)
        self.assertGreater(padded["max_amplification"], 2)
        self.assertFalse(padded["strictly_below_2x"])

    def test_valid_small_geometry_returns_scientific_failure(self):
        result = core.physical_page_envelope(
            expert_count=2,
            coordinates_per_role=8,
            latent_bits_per_coordinate=1,
            requested_rate=Fraction(5, 2),
            common_model_bits=0,
            private_required_bytes=[0, 0],
        )
        self.assertEqual(result["status"], "FAIL_PAGE_ROUNDING_EXCEEDS_RATE_CAP")

    def test_failed_read_envelopes_cannot_promote(self):
        failed = {
            rate: {
                "status": "FAIL_CAPACITY_OR_STRICT_READ_AMPLIFICATION",
                "capacity_ok": True,
                "strictly_below_2x": False,
            }
            for rate in ("2.15", "2.5")
        }
        self.assertEqual(worker_core._feasible_rate_endpoints(failed), [])
        status, eligible = worker_core._final_disposition(
            favorable_below_target=False,
            read_eligible_charged_gain_bpw=None,
            control_corrected_gain_bpw=1.0,
        )
        self.assertEqual(
            status, "HOLD_NO_CAPACITY_AND_STRICT_READ_FEASIBLE_RATE_ENDPOINT"
        )
        self.assertFalse(eligible)

        mixed = dict(failed)
        mixed["2.15"] = {
            "status": "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC",
            "capacity_ok": True,
            "strictly_below_2x": True,
        }
        self.assertEqual(worker_core._feasible_rate_endpoints(mixed), ["2.15"])
        status, eligible = worker_core._final_disposition(
            favorable_below_target=False,
            read_eligible_charged_gain_bpw=core.TARGET_GAIN_BPW,
            control_corrected_gain_bpw=core.TARGET_GAIN_BPW,
        )
        self.assertEqual(status, "SURVIVE_IDEAL_APERTURE_REQUIRES_FINITE_CODER")
        self.assertTrue(eligible)


class PanelAndHoldTests(unittest.TestCase):
    def test_panel_is_exact_and_explicit(self):
        panel = json.loads((PACKAGE / "panel_lock.json").read_text(encoding="utf-8"))
        experts = list(range(0, 128, 8))
        self.assertEqual(panel["experts"], experts)
        self.assertEqual(panel["layer"], 15)
        self.assertEqual(len(panel["files"]), 32)
        self.assertEqual(
            [(row["expert"], row["role"]) for row in panel["files"]],
            [(expert, role) for expert in experts for role in ("up", "down")],
        )
        for row in panel["files"]:
            self.assertEqual(row["bytes"], 3145728)
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")
            self.assertNotIn("*", row["relative_path"])

    def test_hold_precedes_any_path_or_import_or_output_access(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "must_not_exist.json"
            with mock.patch.object(run_gate, "Path", side_effect=AssertionError("Path touched")):
                code = run_gate.main([
                    "--authorization", run_gate.AUTHORIZATION_PHRASE,
                    "--payload-root", str(Path(td) / "payload"),
                    "--output", str(output),
                ])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            self.assertNotIn("same_layer_common_latent_cupy_worker_v0", sys.modules)


class OptionalCupyParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            cp.zeros(1).sum().item()
            cls.cp = cp
            import cupy_worker
            cls.worker = cupy_worker
        except Exception as exc:  # environment-dependent preflight
            raise unittest.SkipTest(f"CuPy device unavailable: {exc}")

    def test_counts_and_dual_objectives_match_cpu(self):
        labels = make_common_label_fixture(5, 521)
        gpu = self.cp.asarray(labels)
        for cardinality, objective in ((2, "favorable"), (2, "charged"), (4, "charged")):
            cpu_score = core.score_labels_cpu(labels, cardinality, 0, objective)
            gpu_score = self.worker.score_labels_gpu(gpu, cardinality, 0, objective)
            self.assertEqual(cpu_score["planes"], gpu_score["planes"])
            self.assertEqual(cpu_score["count_evidence"], gpu_score["count_evidence"])
            self.assertEqual(cpu_score["common_two_part_bits"], gpu_score["common_two_part_bits"])

    def test_quantizer_scale_and_label_bytes_match_cpu(self):
        values = make_quantizer_fixture(32)
        cpu_q = core.quantize_canonical_cpu(values, 32)
        gpu_q, gpu_scale = self.worker.quantize_canonical_gpu(values, 32)
        np.testing.assert_array_equal(cpu_q.scale_u16, gpu_scale)
        np.testing.assert_array_equal(cpu_q.labels.reshape(-1), self.cp.asnumpy(gpu_q))


class ManifestClosureTests(unittest.TestCase):
    def _digest(self, root: Path) -> str:
        return hashlib.sha256((root / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()

    def test_live_source_closure(self):
        receipt = verify_source.verify(PACKAGE, self._digest(PACKAGE))
        self.assertEqual(receipt["status"], "PASS_SOURCE_CLOSED_HOLD")

    def test_tamper_and_extra_entries_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            copied = Path(td) / "package"
            shutil.copytree(PACKAGE, copied)
            digest = self._digest(copied)
            with (copied / "README.md").open("a", encoding="utf-8") as handle:
                handle.write("tamper")
            with self.assertRaisesRegex(RuntimeError, "(byte|hash) mismatch"):
                verify_source.verify(copied, digest)
        with tempfile.TemporaryDirectory() as td:
            copied = Path(td) / "package"
            shutil.copytree(PACKAGE, copied)
            digest = self._digest(copied)
            (copied / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "extra"):
                verify_source.verify(copied, digest)
        with tempfile.TemporaryDirectory() as td:
            copied = Path(td) / "package"
            shutil.copytree(PACKAGE, copied)
            digest = self._digest(copied)
            (copied / "extra_directory").mkdir()
            with self.assertRaisesRegex(RuntimeError, "extra"):
                verify_source.verify(copied, digest)

    def test_symlink_member_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as td:
            copied = Path(td) / "package"
            shutil.copytree(PACKAGE, copied)
            digest = self._digest(copied)
            target = copied / "README.md"
            backing = Path(td) / "backing.md"
            shutil.copyfile(target, backing)
            target.unlink()
            try:
                os.symlink(backing, target)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            with self.assertRaisesRegex(RuntimeError, "non-regular"):
                verify_source.verify(copied, digest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
