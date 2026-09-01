from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import common
import initialization_anchor_gate as gate
import verify_result


class FrozenProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = common.default_workspace_root()

    def test_lock_seals_and_exact_tier_a_candidates(self) -> None:
        lock = common.load_candidate_lock()
        candidates = common.enumerate_candidates(lock)
        self.assertEqual(56, len(candidates))
        self.assertEqual(
            "0000|hf451_tensor_reset|seed=0|dtype=fp32_then_bfloat16",
            candidates[0].id,
        )
        self.assertEqual(
            "0055|hf451_constructor_then_global_post_init|seed=3407|dtype=bfloat16_direct",
            candidates[-1].id,
        )
        self.assertEqual(len(candidates), len({candidate.id for candidate in candidates}))
        self.assertEqual(
            common.CANDIDATE_LOCK_FILE_SHA256,
            common.sha256_file(common.CANDIDATE_LOCK_PATH),
        )

    def test_frozen_source_exclusion_and_splits(self) -> None:
        rows = common.load_frozen_source_rows(self.workspace_root)
        self.assertEqual(32, len(rows))
        excluded = [row.tensor_name for row in rows if row.excluded]
        self.assertEqual(list(common.EXPECTED_EXCLUDED_TENSORS), excluded)
        self.assertEqual(23, sum(not row.excluded and row.split == "candidate_selection" for row in rows))
        self.assertEqual(8, sum(not row.excluded and row.split == "validation" for row in rows))

    def test_coordinate_plan_is_exact_disjoint_and_deterministic(self) -> None:
        rows = common.load_frozen_source_rows(self.workspace_root)
        left = common.make_coordinate_plan(rows)
        right = common.make_coordinate_plan(rows)
        self.assertEqual(common.coordinate_plan_sha256(left), common.coordinate_plan_sha256(right))
        self.assertEqual(common.FIT_COORDINATES, sum(len(row.fit) for row in left))
        self.assertEqual(common.SCORE_COORDINATES, sum(len(row.score) for row in left))
        for row in left:
            self.assertFalse(set(row.fit) & set(row.score))
            self.assertEqual(len(row.fit), len(set(row.fit)))
            self.assertEqual(len(row.score), len(set(row.score)))
            self.assertGreaterEqual(min(row.fit + row.score), 0)
            self.assertLess(max(row.fit + row.score), common.WEIGHTS_PER_MATRIX)

    def test_canonical_native_mapping(self) -> None:
        canonical = np.asarray([0, 1, 2047, 2048, 2049, common.WEIGHTS_PER_MATRIX - 1])
        np.testing.assert_array_equal(canonical, common.canonical_to_native_flat("up", canonical))
        expected_down = np.asarray([0, 768, 2047 * 768, 1, 769, common.WEIGHTS_PER_MATRIX - 1])
        np.testing.assert_array_equal(expected_down, common.canonical_to_native_flat("down", canonical))

    def test_bfloat16_decode(self) -> None:
        words = np.asarray([0x3F80, 0xC000, 0x3F00, 0x0000], dtype="<u2")
        decoded = common.decode_bfloat16_words(words)
        np.testing.assert_array_equal(decoded, np.asarray([1.0, -2.0, 0.5, 0.0], dtype=np.float32))

    def test_fit_only_parameters_do_not_depend_on_score(self) -> None:
        w_fit = np.asarray([1.0, 2.0, 4.0, 7.0])
        g_fit = np.asarray([-2.0, -1.0, 1.0, 3.0])
        fit_a = common.fit_affine_moments(w_fit, g_fit)
        common.score_affine_moments([10.0, 20.0], [8.0, -4.0], fit_a["alpha"], fit_a["mu"], fit_a["fit_mean_w"])
        fit_b = common.fit_affine_moments(w_fit, g_fit)
        self.assertEqual(fit_a, fit_b)

    def test_controls_are_deterministic_and_split_separated(self) -> None:
        coordinates = [1, 17, 999, 1_000_000]
        left = common.stateless_standard_normals("tensor", "fit", coordinates)
        right = common.stateless_standard_normals("tensor", "fit", coordinates)
        other = common.stateless_standard_normals("tensor", "score", coordinates)
        np.testing.assert_array_equal(left, right)
        self.assertFalse(np.array_equal(left, other))
        first = common.deterministic_permutation("tensor", "fit", 31)
        second = common.deterministic_permutation("tensor", "fit", 31)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.arange(31), np.sort(first))

    def test_philox_increment_formula(self) -> None:
        sm = 128
        threads = 1536
        self.assertEqual(4, gate.philox_offset_increment(1, sm, threads))
        self.assertEqual(4, gate.philox_offset_increment(257, sm, threads))
        for numel in (65_537, common.WEIGHTS_PER_MATRIX, 2 * common.WEIGHTS_PER_MATRIX):
            grid = min((numel + 255) // 256, sm * (threads // 256))
            expected = ((numel - 1) // (256 * grid * 4) + 1) * 4
            self.assertEqual(expected, gate.philox_offset_increment(numel, sm, threads))

    def test_decision_boundaries_use_frozen_inequalities(self) -> None:
        ledger = common.physical_ledger()
        composite = ledger["metadata_adjusted_composite_required_capture"]
        current = ledger["metadata_adjusted_current_required_capture"]

        def folds(capture: float):
            return {
                "pooled": {"capture": capture},
                "whole_expert_capture_standard_error": 0.0,
                "whole_experts": [{"capture": max(capture, 1e-12)}],
            }

        self.assertEqual(
            "HARD_KILL_BOUNDED_INITIALIZER_SET",
            common.make_decision(folds(math.nextafter(composite, -math.inf)), 0.0, 0.0)["state"],
        )
        self.assertEqual(
            "COMPOSITE_ONLY_OR_INCONCLUSIVE",
            common.make_decision(folds(composite), 0.0, 0.0)["state"],
        )
        self.assertEqual(
            "COMPOSITE_ONLY_OR_INCONCLUSIVE",
            common.make_decision(folds(math.nextafter(current, -math.inf)), 0.0, 0.0)["state"],
        )
        self.assertEqual(
            "AUXILIARY_STANDALONE_SURVIVOR_REQUIRES_GATE_ROLE_PROTOCOL",
            common.make_decision(folds(current), 0.0, 0.0)["state"],
        )

    def test_physical_ledger_and_read_gate(self) -> None:
        ledger = common.physical_ledger()
        self.assertEqual(414, ledger["metadata_bytes_total"])
        self.assertAlmostEqual(0.00011698404947916667, ledger["model_specific_side_bpw"], places=16)
        self.assertEqual(0, ledger["generator_external_read_bytes"])
        self.assertLess(ledger["conservative_metadata_appended_worst_cold_read_amplification"], 2.0)
        self.assertTrue(ledger["passes_read_gate"])

    def test_detail_algebra_verifier(self) -> None:
        source = common.SourceRow(
            0, 8, "up", common.tensor_name(8, "up"), common.tensor_basename(8, "up"),
            "0" * 64, 8, (2, 2), (2, 2), False, "candidate_selection",
        )
        w_fit = np.asarray([1.0, 2.0, 4.0, 8.0])
        g_fit = np.asarray([-1.0, 0.5, 2.0, 3.0])
        w_score = np.asarray([3.0, -1.0, 6.0])
        g_score = np.asarray([0.1, 1.5, -2.0])
        fit = common.fit_affine_moments(w_fit, g_fit)
        score = common.score_affine_moments(w_score, g_score, fit["alpha"], fit["mu"], fit["fit_mean_w"])
        row = {"tensor_name": source.tensor_name, "expert": 8, "role": "up", "fit": fit, "score": score}
        verify_result._verify_detail_row(row, source, "synthetic")
        corrupted = json.loads(json.dumps(row))
        corrupted["score"]["sse"] *= 1.1
        with self.assertRaises(common.ProtocolError):
            verify_result._verify_detail_row(corrupted, source, "synthetic")

    def test_aux_directory_firewall_checks_exact_set_without_hashing(self) -> None:
        rows = common.load_frozen_source_rows(self.workspace_root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for row in rows:
                with (root / row.basename).open("wb") as handle:
                    handle.truncate(row.bytes)
            paths = common.validate_aux_directory(root, rows)
            self.assertEqual(32, len(paths))
            (root / "extra.bf16.bin").touch()
            with self.assertRaises(common.ProtocolError):
                common.validate_aux_directory(root, rows)

    def test_cpu_preflight_imports_no_cuda_library(self) -> None:
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("cupy", sys.modules)
        report = gate.cpu_preflight(self.workspace_root)
        self.assertEqual("PASS_CUDA_NOT_IMPORTED_OR_TOUCHED", report["status"])
        self.assertEqual(56, report["candidate_count"])
        self.assertFalse(report["cuda_modules_imported"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("cupy", sys.modules)


if __name__ == "__main__":
    unittest.main()
