from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import clustered_ib_core as core
import run_gate
import verify_source
from source_free_fixture import make_clustered_nonmodal_fixture


class MathematicalContractTests(unittest.TestCase):
    def test_frozen_group_bank_and_partition_bits(self):
        self.assertEqual(core.compatible_group_sizes(16), (2, 4, 8, 16))
        self.assertEqual(core.partition_count(4, 2), 3)
        self.assertEqual(core.partition_descriptor_bits(4, 2), 2)
        self.assertEqual(core.selector_bits_for_group_bank(16), 2)
        self.assertEqual(core.compatible_group_sizes(3), ())

    def test_crossfit_fold_is_superblock_deterministic(self):
        ids = core.fold_ids(1024, fold_count=4, superblock_values=128)
        for block in range(8):
            np.testing.assert_array_equal(ids[block * 128:(block + 1) * 128], block % 4)

    def test_partition_for_fold_never_reads_that_fold(self):
        labels = make_clustered_nonmodal_fixture(coordinates=512)
        ids = core.fold_ids(512, fold_count=2, superblock_values=128)
        train = ids != 0
        before = core._fold_partition(labels, train, 4)
        changed = labels.copy()
        rng = np.random.default_rng(17)
        changed[:, :, ~train] = rng.integers(
            0, 4, size=changed[:, :, ~train].shape, dtype=np.uint8
        )
        after = core._fold_partition(changed, train, 4)
        self.assertEqual(before, after)

    def test_arbitrary_binary_latent_finds_nonmodal_fixture(self):
        labels = make_clustered_nonmodal_fixture(coordinates=512)
        score = core.crossfit_group_size(
            labels, 4, fold_count=2, superblock_values=128
        )
        self.assertGreater(score["favorable_gross_gain_bpw"], 0.5)
        self.assertEqual(score["partition_bits"], 2 * core.partition_descriptor_bits(16, 4))
        self.assertEqual(len(score["segment_members"]), 8)
        self.assertEqual(score["selector_bits"], 2)
        self.assertGreater(score["structured_framing_bits"], score["baseline_framing_bits"])

    def test_binary_descriptor_derives_final_counts(self):
        self.assertEqual(core.marginal_model_descriptor_bits(100), 21)
        expected = 7 + 4 * (3 * 6 + 3 * 6)
        self.assertEqual(core.binary_model_descriptor_bits((50, 50), 4), expected)

    def test_marginal_scramble_is_exact_and_independent(self):
        labels = make_clustered_nonmodal_fixture(coordinates=257)
        controlled = core.marginal_preserving_control(labels, core.CONTROL_SEEDS[0])
        self.assertFalse(np.array_equal(labels, controlled))
        for expert in range(16):
            for role in range(2):
                np.testing.assert_array_equal(
                    np.bincount(labels[expert, role], minlength=4),
                    np.bincount(controlled[expert, role], minlength=4),
                )


class PhysicalLedgerTests(unittest.TestCase):
    def test_exact_flat_path_formulas(self):
        segments = [
            {"members": [0, 1], "required_bytes": 4096},
            {"members": [2, 3], "required_bytes": 4096},
        ]
        result = core.physical_read_envelope(
            expert_count=4,
            weights_per_expert=49152,
            requested_rate=Fraction(5, 2),
            global_required_bytes=4096,
            common_segments=segments,
            private_required_bytes=[8192] * 4,
        )
        self.assertTrue(result["capacity_ok"])
        self.assertTrue(result["strictly_below_2x"])
        self.assertEqual(result["touched_bytes"], [20480] * 4)
        self.assertEqual(result["owned_physical_bytes"], ["15360"] * 4)
        self.assertEqual(result["owned_nonpadding_bytes"], ["11264"] * 4)

    def test_nonpadding_denominator_blocks_padding_attack(self):
        result = core.physical_read_envelope(
            expert_count=4,
            weights_per_expert=32768,
            requested_rate=Fraction(5, 2),
            global_required_bytes=1,
            common_segments=[
                {"members": [0, 1], "required_bytes": 1},
                {"members": [2, 3], "required_bytes": 1},
            ],
            private_required_bytes=[1] * 4,
        )
        self.assertTrue(result["capacity_ok"])
        self.assertFalse(result["strictly_below_2x"])
        self.assertEqual(result["status"], "FAIL_STRICT_READ_AMPLIFICATION")
        self.assertLess(max(Fraction(x) for x in result["amplification_physical_fraction"]), 2)
        self.assertGreater(max(Fraction(x) for x in result["amplification_nonpadding_fraction"]), 2)

    def test_exact_two_x_fails_strictly(self):
        # Touched=3 pages; nonpadding ownership=1.5 pages exactly.
        result = core.physical_read_envelope(
            expert_count=2,
            weights_per_expert=32768,
            requested_rate=Fraction(5, 2),
            global_required_bytes=4096,
            common_segments=[{"members": [0, 1], "required_bytes": 4096}],
            private_required_bytes=[2048, 2048],
        )
        self.assertFalse(result["strictly_below_2x"])
        self.assertGreaterEqual(result["max_amplification"], 2.0)


class DecisionOrderTests(unittest.TestCase):
    def test_controls_are_skipped_before_favorable_survival(self):
        labels = np.zeros((4, 2, 128), dtype=np.uint8)
        with mock.patch.object(
            core, "marginal_preserving_control",
            side_effect=AssertionError("control executed before source survival"),
        ):
            result = core.score_source_gate(
                labels, scale_bytes_per_expert=0,
                fold_count=2, superblock_values=64,
            )
        self.assertEqual(result["status"], "HARD_KILL_FAVORABLE_BELOW_TARGET")
        self.assertFalse(result["controls_executed"])
        self.assertEqual(result["controls"], [])

    def test_threshold_is_not_triage_or_rounded(self):
        self.assertEqual(core.TARGET_GAIN_BPW, 0.22933495044437175)
        self.assertFalse(core.TARGET_GAIN_BPW - 1e-15 >= core.TARGET_GAIN_BPW)


class HoldAndClosureTests(unittest.TestCase):
    def _digest(self, root: Path) -> str:
        return hashlib.sha256((root / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()

    def test_source_entrypoint_has_no_enabled_branch(self):
        self.assertFalse(run_gate.PAYLOAD_EXECUTION_ENABLED)
        self.assertEqual(run_gate.main([]), 2)

    def test_live_source_closure(self):
        receipt = verify_source.verify(PACKAGE, self._digest(PACKAGE))
        self.assertEqual(receipt["status"], "PASS_SOURCE_CLOSED_HOLD_NO_DEPLOYMENT")

    def test_tamper_and_extra_member_fail(self):
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
            (copied / "extra").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "extra"):
                verify_source.verify(copied, digest)


if __name__ == "__main__":
    unittest.main()
