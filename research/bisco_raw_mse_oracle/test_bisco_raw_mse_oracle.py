from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import bisco_raw_mse_oracle as oracle
import verify_bisco_raw_mse as verifier


class BiSCoRawMSEOracleTests(unittest.TestCase):
    def test_frozen_protocol_hashes_and_split(self) -> None:
        binding = oracle.validate_protocol_bindings()
        self.assertEqual(binding["hashes"]["launch"], oracle.LAUNCH_PROTOCOL_SHA256)
        self.assertEqual(tuple(binding["launch"]["data"]["validation_experts"]), oracle.VALIDATION_EXPERTS)
        self.assertEqual(set(oracle.TRAIN_EXPERTS).intersection(oracle.VALIDATION_EXPERTS), set())

    def test_exact_rate_and_read_ledgers(self) -> None:
        ledgers = oracle.validate_ledgers()
        production = ledgers["production_128"]
        panel = ledgers["self_contained_panel_6"]
        self.assertEqual(production["decoder_parameters"], 13_536)
        self.assertEqual(production["decoder_bytes"], 27_072)
        self.assertEqual(production["code_bytes_per_expert"], 1_327_104)
        self.assertAlmostEqual(production["physical_bpw"], 2.250382317437066, places=14)
        self.assertAlmostEqual(production["cold_read_amplification"], 1.020427859096027, places=14)
        self.assertAlmostEqual(panel["physical_bpw"], 2.2577424225983798, places=14)
        self.assertAlmostEqual(panel["cold_read_amplification"], 1.0171013253527148, places=14)
        self.assertLess(production["cold_read_amplification"], 2.0)
        self.assertLess(panel["cold_read_amplification"], 2.0)
        for experts, name in ((128, "production_128"), (6, "self_contained_panel_6")):
            independent = verifier.ledger(experts)
            for field in independent:
                self.assertAlmostEqual(float(ledgers[name][field]), float(independent[field]), places=14)

    def test_auxiliary_firewall_requires_exact_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for expert in oracle.EXPERTS:
                for role in oracle.ROLES:
                    (root / f"l15e{expert}_{role}.bf16.bin").touch()
            files = oracle.discover_auxiliary(root)
            self.assertEqual(tuple(sorted(files["up"])), oracle.EXPERTS)
            (root / "model.layers.5.mlp.experts.18.up_proj.weight.bf16.bin").touch()
            with self.assertRaises(RuntimeError):
                oracle.discover_auxiliary(root)

    def test_stored_normalization_is_fp16_charged_and_invertible(self) -> None:
        values = np.linspace(-0.7, 1.3, 4 * oracle.D, dtype=np.float32)
        chunks, metadata = oracle.stored_normalize(values)
        reconstructed = (
            chunks.reshape(-1) * np.float32(metadata["stored_fp16_centered_rms"])
            + np.float32(metadata["stored_fp16_mean"])
        )
        np.testing.assert_allclose(reconstructed, values, rtol=1e-6, atol=1e-6)
        self.assertEqual(chunks.shape, (4, oracle.D))
        self.assertGreater(metadata["source_energy"], 0.0)

    def test_paired_initialization_is_bit_identical(self) -> None:
        initial = oracle.initialize_codec(123, np)
        models = {
            role: {"qwen": oracle.copy_codec(initial), "gaussian": oracle.copy_codec(initial)}
            for role in oracle.ROLES
        }
        hashes = oracle.model_initialization_hashes(models, np)
        for role in oracle.ROLES:
            self.assertEqual(hashes[role]["qwen"], hashes[role]["gaussian"])
        values = sum(int(initial[name].size) for name in oracle.PARAMETER_NAMES)
        self.assertEqual(values, 9_028)

    def test_decoder_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(size=(7, oracle.D)).astype(np.float32)
        params = oracle.initialize_codec(9, np)
        loss, grads = oracle.codec_loss_and_grads(params, x, 0.8, 0.0, np)
        self.assertTrue(np.isfinite(float(loss)))
        epsilon = 2e-3
        original = float(params["s2_db2"][0])
        params["s2_db2"][0] = original + epsilon
        plus = float(oracle.codec_loss_and_grads(params, x, 0.8, 0.0, np)[0])
        params["s2_db2"][0] = original - epsilon
        minus = float(oracle.codec_loss_and_grads(params, x, 0.8, 0.0, np)[0])
        params["s2_db2"][0] = original
        numeric = (plus - minus) / (2.0 * epsilon)
        self.assertAlmostEqual(float(grads["s2_db2"][0]), numeric, delta=2e-4)

    def test_greedy_bitflip_never_increases_chunk_error(self) -> None:
        rng = np.random.default_rng(88)
        x = rng.normal(size=(23, oracle.D)).astype(np.float32)
        params = oracle.quantized_decoder_codec(oracle.initialize_codec(5, np), np)
        before = oracle.reconstruct_batch(params, x, 0, np)
        after = oracle.reconstruct_batch(params, x, 1, np)
        before_error = np.sum((before - x) ** 2, axis=1)
        after_error = np.sum((after - x) ** 2, axis=1)
        self.assertTrue(np.all(after_error <= before_error + 1e-6))

    def test_whole_expert_aggregation_and_target_identity(self) -> None:
        rows = [
            {"expert": expert, "qwen_sse": 0.8, "qwen_energy": 1.0, "gaussian_sse": 1.0, "gaussian_energy": 1.0}
            for expert in oracle.VALIDATION_EXPERTS
        ]
        aggregate = oracle.aggregate_evaluation(rows)
        aggregate["per_matrix"] = [
            {
                "expert": expert,
                "role": role,
                "qwen_sse": 0.4,
                "qwen_energy": 0.5,
                "gaussian_sse": 0.5,
                "gaussian_energy": 0.5,
            }
            for role in oracle.ROLES
            for expert in oracle.VALIDATION_EXPERTS
        ]
        self.assertAlmostEqual(aggregate["D_Qwen"], 0.8)
        self.assertAlmostEqual(aggregate["D_Gaussian"], 1.0)
        self.assertAlmostEqual(aggregate["s_match"], oracle.TARGET_S)
        self.assertAlmostEqual(aggregate["whole_expert_standard_error"], 0.0)
        independently_recomputed = verifier.recompute_evaluation(aggregate)
        self.assertAlmostEqual(independently_recomputed["s_match"], oracle.TARGET_S)
        aggregate["per_matrix"][0]["qwen_sse"] += 0.01
        with self.assertRaises(AssertionError):
            verifier.recompute_evaluation(aggregate)

    def test_preregistered_early_kill_boundary(self) -> None:
        killed = oracle.early_kill_decision(
            {"upper_s_match_2se": 0.071}, {"upper_s_match_2se": 0.079}
        )
        self.assertTrue(killed["kill"])
        self.assertAlmostEqual(killed["constant_recent_slope_projection_to_full_budget"], 0.127)
        self.assertLess(killed["constant_recent_slope_projection_to_full_budget"], 0.14)
        high_level = oracle.early_kill_decision(
            {"upper_s_match_2se": 0.071}, {"upper_s_match_2se": 0.081}
        )
        self.assertFalse(high_level["kill"])
        steep = oracle.early_kill_decision(
            {"upper_s_match_2se": 0.060}, {"upper_s_match_2se": 0.0711}
        )
        self.assertFalse(steep["kill"])

    def test_matched_gaussian_is_deterministic(self) -> None:
        first = oracle.matched_gaussian(1024, 0.2, 1.7, 991)
        second = oracle.matched_gaussian(1024, 0.2, 1.7, 991)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(np.std(first)), 1.5)


if __name__ == "__main__":
    unittest.main()
