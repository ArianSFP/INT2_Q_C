from __future__ import annotations

from fractions import Fraction
import copy
import itertools
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pairpath_r3_core as core
from source_free_fixtures import (adversarial_solver_fixture, aligned_fixture,
                                  iid_fixture, unequal_role_energy_fixture)


class GlobalMultiplierTests(unittest.TestCase):
    def test_finite_pair_plan_uses_one_exact_global_updown_multiplier(self):
        source = unequal_role_energy_fixture()
        lagrange = core.LAMBDA_GRID[4]
        expected = core.global_updown_bit_weight(source, lagrange)
        observed = []
        original = core._fit_pair_fold

        def record(*args, **kwargs):
            observed.append(float(args[-1]))
            return original(*args, **kwargs)

        with mock.patch.object(core, "_fit_pair_fold", side_effect=record):
            plan = core._make_plan(source, "pair_k2_flexible", lagrange)
        self.assertEqual(len(observed), core.FOLD_COUNT * len(core.OPTIMIZED_ROLES))
        self.assertTrue(all(value.hex() == expected.hex() for value in observed))
        cert = plan["r3_encoder_certificate"]
        self.assertEqual(cert["global_bit_weight_hex"], expected.hex())
        self.assertEqual(set(cert["optimized_role_bit_weight_hex"].values()),
                         {expected.hex()})

        # The hostile-audit construction has radically unequal role-local values.
        local = []
        for role in core.OPTIMIZED_ROLES:
            values = source[:, role]
            energy = float(np.sum(values * values, dtype=np.float64))
            local.append(float(lagrange) * energy / values.size)
        self.assertNotEqual(local[0].hex(), local[1].hex())
        self.assertNotEqual(local[0].hex(), expected.hex())
        self.assertNotEqual(local[1].hex(), expected.hex())

    def test_finite_packet_serializes_and_decoder_checks_multiplier_certificate(self):
        source = iid_fixture(32768)
        plan = core._make_plan(source, "pair_k2_fixed", core.LAMBDA_GRID[0])
        packet = core._encode_plan(source, plan)
        decoded = core.decode_packet(packet)
        cert = decoded["header"]["r3_encoder_certificate"]
        self.assertEqual(set(cert["optimized_role_bit_weight_hex"].values()),
                         {cert["global_bit_weight_hex"]})

        header, common, privates = core._packet_parts(packet)
        bad = copy.deepcopy(header)
        bad["r3_encoder_certificate"]["optimized_role_bit_weight_hex"]["2"] = "0x1p+0"
        with self.assertRaisesRegex(core.CodecError, "multiplier divergence"):
            core.decode_packet(core._pack_parts(bad, common, privates))


class DominanceCertificateTests(unittest.TestCase):
    def _assert_gate(self, source: np.ndarray):
        gate = core.optimistic_single_letter_joint_gate(
            source, (core.LAMBDA_GRID[0], core.LAMBDA_GRID[-1]))
        self.assertFalse(gate["hard_kill_authority"])
        self.assertFalse(gate["global_optimality_proven"])
        self.assertNotIn("HARD_KILL", gate["status"])
        certificates = gate["independent_candidate_dominance_certificates"]
        self.assertEqual(len(certificates), 3 * len(core.OPTIMIZED_ROLES))
        for row in certificates:
            self.assertTrue(row["dominates_independent_candidate"])
            tolerance = 1e-12 * max(
                1.0, abs(row["independent_labels_under_joint_objective"]))
            self.assertLessEqual(row["joint_objective"],
                                 row["independent_labels_under_joint_objective"] + tolerance)
        return gate

    def test_iid_below_gate_is_explicitly_non_authoritative(self):
        gate = self._assert_gate(iid_fixture())
        self.assertEqual(gate["status"], "HEURISTIC_BELOW_GATE_NONAUTHORITATIVE")

    def test_aligned_and_adversarial_sources_have_dominance_certificates(self):
        aligned = self._assert_gate(aligned_fixture())
        self.assertEqual(aligned["status"], "SURVIVE_HEURISTIC_WITH_PHYSICAL_MARGIN")
        self._assert_gate(adversarial_solver_fixture())

    def test_required_candidate_is_scored_under_same_joint_objective(self):
        rng = np.random.default_rng(0x43455254)
        for n in (7, 19, 61):
            values = rng.normal(size=(2, n))
            levels = np.broadcast_to(
                np.asarray(core.LEVELS_RMS)[None, None, :], (2, n, core.ALPHABET)).copy()
            independent = core._ideal_flexible_role_certified(
                values, levels, 0.1, False)
            joint = core._ideal_flexible_role_certified(
                values, levels, 0.1, True, required_candidate=independent["labels"])
            direct = core._empirical_role_score(
                values, levels, independent["labels"], 0.1, True)[0]
            self.assertEqual(joint["required_candidate_objective"], direct)
            self.assertLessEqual(joint["objective"], direct + 1e-12 * max(1.0, abs(direct)))


class TreeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = iid_fixture(32768)
        cls.packet = core._encode_plan(
            cls.source,
            core._make_plan(cls.source, "independent_fixed", core.LAMBDA_GRID[0]))

    def test_valid_descriptor_is_causally_replayed(self):
        decoded = core.decode_packet(self.packet)
        tree = core._validate_and_replay_tree_descriptor(decoded["header"])
        self.assertEqual(tree["pairs"], ((0, 1),))
        for experts, expected in ((2, 1), (4, 3), (6, 45)):
            width = core.tree_descriptor_bits(experts)
            valid = 0
            for packed in range(1 << width if width else 1):
                try:
                    row = core.decode_tree_descriptor(packed, experts)
                except core.CodecError:
                    continue
                replay = core.encode_tree_descriptor(
                    experts, row["pairs"], row["merge_ranks"])
                self.assertEqual(replay, (packed, width))
                valid += 1
            self.assertEqual(valid, expected)

    def test_every_redundant_tree_field_is_validated(self):
        header, common, privates = core._packet_parts(self.packet)
        mutations = []
        for key, value in (
                ("materialized", [1, 0]),
                ("pairs", [[1, 0]]),
                ("bits", 1),
                ("packed", True),
                ("merge_ranks", [0])):
            bad = copy.deepcopy(header)
            bad["tree_descriptor"][key] = value
            mutations.append(bad)
        bad = copy.deepcopy(header)
        del bad["tree_descriptor"]["materialized"]
        mutations.append(bad)
        for mutated in mutations:
            with self.subTest(descriptor=mutated["tree_descriptor"]):
                packet = core._pack_parts(mutated, common, privates)
                with self.assertRaisesRegex(core.CodecError, "tree"):
                    core.decode_packet(packet)


class BoundaryTests(unittest.TestCase):
    def test_source_only_and_dependency_pin(self):
        self.assertEqual(core.hashlib.sha256(core.R2_CORE_PATH.read_bytes()).hexdigest(),
                         core.R2_CORE_SHA256)
        import run_gate
        self.assertFalse(run_gate.PAYLOAD_EXECUTION_ENABLED)
        self.assertFalse(run_gate.LOCAL_GPU_EXECUTION_ENABLED)
        self.assertFalse(run_gate.QWEN_APERTURE_AUTHORIZED)

    def test_literal_roundtrip_rate_and_read(self):
        source = iid_fixture(32768)
        packet = core._encode_plan(
            source, core._make_plan(source, "pair_k2_fixed", core.LAMBDA_GRID[0]))
        decoded = core.decode_packet(packet)
        self.assertEqual(decoded["header"]["source_sha256"], core.source_sha256(source))
        score = core.evaluate_packet(source, packet)
        self.assertGreaterEqual(Fraction(score["rate_fraction"]), core.RATE_MIN)
        self.assertLessEqual(Fraction(score["rate_fraction"]), core.RATE_MAX)
        self.assertTrue(score["read_ledger"]["strictly_below_2x"])


if __name__ == "__main__":
    unittest.main()
