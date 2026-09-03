from __future__ import annotations

from fractions import Fraction
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pairpath_r2_core as core
from source_free_fixtures import aligned_fixture, boundary_fixture, iid_fixture


class OracleTests(unittest.TestCase):
    def test_iid_is_killed_and_aligned_positive_control_survives(self):
        grid = (core.LAMBDA_GRID[0], core.LAMBDA_GRID[-1])
        iid = core.optimistic_single_letter_joint_gate(iid_fixture(16384), grid)
        aligned = core.optimistic_single_letter_joint_gate(aligned_fixture(), grid)
        self.assertEqual(iid["status"], "HARD_KILL_OPTIMISTIC_JOINT_GATE_BELOW_0P045")
        self.assertLess(iid["best_G_eq_UD_bpw"], core.ORACLE_EARLY_KILL_BPW)
        self.assertEqual(aligned["status"], "SURVIVE_OPTIMISTIC_GATE_WITH_PHYSICAL_MARGIN")
        self.assertGreater(aligned["best_G_eq_UD_bpw"], core.ORACLE_ENGINEERING_MARGIN_BPW)
        self.assertGreater(aligned["fixed_assignment_mi"]
                           ["mutual_information_bits_per_coordinate_pair"],
                           core.FIXED_ASSIGNMENT_MI_REQUIRED_BITS_PER_PAIR)

    def test_value_controls_recompute_scales_and_labels(self):
        source = aligned_fixture()
        permuted = core.affine_value_control(source, core.CONTROL_SEEDS[0])
        gaussian = core.gaussian_value_control(source)
        for controlled in (permuted, gaussian):
            self.assertEqual(controlled.dtype, np.float64)
            scales = core.estimate_scale_bits(controlled)
            for e, role in itertools.product(range(2), range(3)):
                labels = core.nearest_labels(controlled[e, role], scales[e, role])
                self.assertEqual(labels.shape, (source.shape[2],))
        # Exact marginal-preserving permutation, performed on values, not labels.
        for e, role, block in itertools.product(range(2), range(3), range(8)):
            lo = block * core.BLOCK_VALUES
            a = np.sort(source[e, role, lo:lo + core.BLOCK_VALUES])
            b = np.sort(permuted[e, role, lo:lo + core.BLOCK_VALUES])
            np.testing.assert_array_equal(a, b)


class LiteralCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = iid_fixture(32768)
        cls.result = core.run_micro_oracle(cls.source, (core.LAMBDA_GRID[0],))

    def test_literal_packet_independent_decode_and_binding(self):
        packet = self.result["selected_packet"]
        decoded = core.decode_packet(packet)
        self.assertEqual(decoded["packet_sha256"], self.result["selected_packet_sha256"])
        binding = core.make_binding(self.source, packet)
        receipt = core.validate_binding(binding, self.source, packet)
        self.assertEqual(receipt["status"], "PASS_REAL_BYTE_BINDING")
        changed = self.source.copy()
        changed[0, 0, 0] += 1e-9
        with self.assertRaisesRegex(core.CodecError, "binding hash"):
            core.validate_binding(binding, changed, packet)

    def test_packet_is_source_closed_rate_legal_and_below_2x(self):
        score = self.result["selected_score"]
        self.assertGreaterEqual(Fraction(score["rate_fraction"]), core.RATE_MIN)
        self.assertLessEqual(Fraction(score["rate_fraction"]), core.RATE_MAX)
        self.assertTrue(score["read_ledger"]["strictly_below_2x"])
        self.assertLess(score["read_ledger"]["max_amplification"], 2.0)
        self.assertEqual(core.decode_packet(self.result["selected_packet"])["header"]
                         ["source_sha256"], core.source_sha256(self.source))

    def test_invalid_state_and_negative_energy_fail_closed(self):
        values = self.source[:, 1]
        scales = core.estimate_scale_bits(self.source)[:, 1]
        result = core.choose_pair_labels(values, scales, core.LAMBDA_GRID[0], True)
        bad = result["states"].astype(np.int64)
        bad[0] = 999
        models = [core.prefix_model(row["state_counts"]) for row in result["models"]]
        with self.assertRaisesRegex(core.CodecError, "symbol range"):
            core.encode_contextual(bad.astype(np.int64), core.fold_ids(bad.size), models)
        with self.assertRaisesRegex(core.CodecError, "nonnegative finite SSE"):
            core.score_from_energies(sse_by_role=(1.0, -1.0, 1.0), source_energy=2.0,
                                     physical_bytes=17600, total_weights=65536)

    def test_complete_control_replays_full_pipeline(self):
        receipt = core.run_complete_controls(
            self.source, control_seeds=(core.CONTROL_SEEDS[0],), include_gaussian=True,
            lambda_grid=(core.LAMBDA_GRID[0],))
        self.assertEqual(receipt["control_count"], 2)
        self.assertEqual({r["kind"] for r in receipt["controls"]}, {"affine", "gaussian"})


class FlexibleLabelAndTreeTests(unittest.TestCase):
    def test_flexible_labels_beat_fixed_pair_on_boundary_kat_but_not_independent(self):
        result = core.run_micro_oracle(boundary_fixture(), (core.LAMBDA_GRID[-1],))
        rows = {row["candidate"]: row for row in result["candidate_rows"]}
        self.assertTrue(rows["pair_k2_fixed"]["eligible"])
        self.assertTrue(rows["pair_k2_flexible"]["eligible"])
        self.assertLess(rows["pair_k2_flexible"]["score"]["F"],
                        rows["pair_k2_fixed"]["score"]["F"])
        # The strict independent baseline prevents a false breakthrough claim.
        self.assertEqual(result["selected_candidate"], "independent_fixed")
        self.assertEqual(result["status"], "HARD_KILL_BELOW_REQUIRED_GAIN")

    def test_tree_descriptor_materializes_and_small_codes_are_exhaustive(self):
        expected_valid = {2: 1, 4: 3, 6: 45}
        for expert_count, expected in expected_valid.items():
            width = core.tree_descriptor_bits(expert_count)
            valid = 0
            for packed in range(1 << width if width else 1):
                try:
                    decoded = core.decode_tree_descriptor(packed, expert_count)
                except core.CodecError:
                    continue
                valid += 1
                self.assertEqual(core.flatten_tree(decoded["tree"]), tuple(range(expert_count)))
                replay, replay_width = core.encode_tree_descriptor(
                    expert_count, decoded["pairs"], decoded["merge_ranks"])
                self.assertEqual((replay, replay_width), (packed, width))
            self.assertEqual(valid, expected)

    def test_bootstrap_and_both_jackknives_are_executable(self):
        source = iid_fixture(32768)
        bootstrap = core.bootstrap_full_refit(
            source, 1, lambda_grid=(core.LAMBDA_GRID[0],))
        self.assertTrue(bootstrap["full_pipeline_refit_each_replicate"])
        expert = core.leave_one_expert_out_refit(source)
        self.assertTrue(expert["full_refit"])
        layer = core.leave_one_layer_out_refit(
            (source, affine_copy(source)), lambda_grid=(core.LAMBDA_GRID[0],))
        self.assertTrue(layer["full_refit"])


def affine_copy(values: np.ndarray) -> np.ndarray:
    return core.affine_value_control(values, core.CONTROL_SEEDS[1])


if __name__ == "__main__":
    unittest.main()
