#!/usr/bin/env python3
"""CPU-only scientific/decision tests for lossy-tail v7."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lossy_tail_v7_test_core", HERE / "lossy_tail_core.py")
CORE = importlib.util.module_from_spec(SPEC)
CORE.__dict__["__V7_CORE_CONTEXT__"] = {
    "schema": "lossy-tail-v7-core-context-v1",
    "mode": "source_cpu_test",
}
if SPEC.loader is None:
    raise RuntimeError("oracle loader missing")
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


def row(mode: str, absolute: float, calibrated: float, *, below: bool = True, rate: float = 2.15) -> dict:
    return {
        "mode": mode, "qwen_s_bpw": absolute, "qwen_excess_s_bpw": calibrated,
        "below_2x": below, "rate": rate, "geometry": "raw_adaptive",
    }


class LossyTailOracleV7Tests(unittest.TestCase):
    def test_constants_and_self_test(self):
        self.assertAlmostEqual(CORE.TARGET_S, 0.16096404744368115, places=16)
        self.assertAlmostEqual(CORE.KILL_THRESHOLD_S, 0.14096404744368115, places=16)
        self.assertEqual(CORE.NUMERIC_BOUNDARY_GUARD_S, 1.0e-4)
        self.assertEqual(CORE.MODES, ("free_lloyd", "finite_fp16", "zero_tail_error"))
        CORE.self_test()

    def test_post_fp32_moment_tolerance_and_normalized_reporting(self):
        mean_tol, variance_tol = CORE.control_moment_tolerances(0.01, 0.0004)
        self.assertEqual(mean_tol, 64 * 2.0 ** -23 * 0.02)
        self.assertEqual(variance_tol, 256 * 2.0 ** -23 * 0.0004)
        accepted = CORE.validate_control_moments(
            target_mean=0.01, target_variance=0.0004,
            observed_mean=0.01 + 0.99 * mean_tol, observed_variance=0.0004 - 0.99 * variance_tol,
            scale=1.0, offset=0.0,
        )
        self.assertLessEqual(accepted["mean_normalized_mismatch"], 1.0 + 1e-9)
        self.assertLessEqual(accepted["variance_normalized_mismatch"], 1.0 + 1e-9)
        with self.assertRaisesRegex(RuntimeError, "post-FP32"):
            CORE.validate_control_moments(
                target_mean=0.01, target_variance=0.0004,
                observed_mean=0.01 + 1.01 * mean_tol, observed_variance=0.0004,
                scale=1.0, offset=0.0,
            )
        for variance in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(RuntimeError):
                CORE.control_moment_tolerances(0.0, variance)

    def test_decision_states(self):
        early = CORE.decision_from_calibrated([
            row("free_lloyd", 0.12, 0.11), row("zero_tail_error", 0.13, 0.12),
            row("finite_fp16", 0.10, 0.09),
        ])
        self.assertEqual(early["status"], "EARLY_KILL_FAR_SHORT")
        self.assertTrue(early["early_kill"])

        hold = CORE.decision_from_calibrated([
            row("free_lloyd", 0.15, 0.15), row("zero_tail_error", 0.149, 0.149),
            row("finite_fp16", 0.13, 0.13),
        ])
        self.assertEqual(hold["status"], "HOLD_OPTIMISTIC_NEAR_BOUNDARY")

        survivor = CORE.decision_from_calibrated([
            row("free_lloyd", 0.18, 0.17), row("zero_tail_error", 0.17, 0.17),
            row("finite_fp16", 0.15, 0.15),
        ])
        self.assertEqual(survivor["status"], "OPTIMISTIC_SURVIVOR")
        self.assertFalse(survivor["early_kill"])

        promoted = CORE.decision_from_calibrated([
            row("free_lloyd", 0.20, 0.19), row("zero_tail_error", 0.19, 0.19),
            row("finite_fp16", 0.18, 0.17),
        ])
        self.assertEqual(promoted["status"], "FINITE_CODEC_WARRANTED")
        self.assertTrue(promoted["finite_residual_codec_warranted"])

    def test_numeric_boundary_never_kills_or_promotes(self):
        near_kill = CORE.decision_from_calibrated([
            row("free_lloyd", CORE.KILL_THRESHOLD_S + 0.5e-4, CORE.KILL_THRESHOLD_S + 0.5e-4),
            row("zero_tail_error", 0.12, 0.12), row("finite_fp16", 0.10, 0.10),
        ])
        self.assertEqual(near_kill["status"], "HOLD_NUMERIC_BOUNDARY")
        self.assertFalse(near_kill["early_kill"])
        near_promote = CORE.decision_from_calibrated([
            row("free_lloyd", 0.19, 0.19), row("zero_tail_error", 0.18, 0.18),
            row("finite_fp16", CORE.TARGET_S + 0.5e-4, 0.18),
        ])
        self.assertEqual(near_promote["status"], "HOLD_NUMERIC_BOUNDARY")
        self.assertFalse(near_promote["finite_residual_codec_warranted"])

    def test_finite_cannot_exceed_optimistic_envelope(self):
        with self.assertRaisesRegex(RuntimeError, "finite joint score"):
            CORE.decision_from_calibrated([
                row("free_lloyd", 0.10, 0.10), row("zero_tail_error", 0.11, 0.11),
                row("finite_fp16", 0.12, 0.12),
            ])

    def test_rank_one_both_axis_still_charges_angle(self):
        blank = {
            **CORE.PROFILES[0], "selected_units": 0, "selected_scalars": 0,
            "support_bits": 0, "support_stream_bits": 0, "symbol_bits": 0,
            "symbol_stream_bits": 0, "codebook_bits": 0, "tail_energy": 0.0,
            "bulk_energy": 100.0, "bulk_dimension": CORE.N,
            "free_lloyd_sse": 0.0, "fp16_sse": 0.0,
            "centroids": [], "fp16_centroids": [],
        }
        panel = {
            "label": "rank_one", "total_energy": 100.0,
            "matrices": [[dict(blank) for _ in CORE.PROFILES] for _ in range(CORE.MATRICES)],
            "uniform_components": {
                "coordinate:0.00000000": [CORE.Component("e0.both_axis_1", 0, CORE.N, 100.0)]
            },
        }
        scored = CORE.score(panel, [0] * CORE.MATRICES, 2.15, "finite_fp16", "support_xklt_uniform")
        self.assertTrue(scored["valid"])
        self.assertEqual(scored["side_ledger"]["xklt_angle_bits"], CORE.ANGLE_BITS)
        self.assertEqual(scored["side_ledger"]["bit_closure"], scored["physical_bits"])

    def test_strict_json_rejects_duplicate_keys(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
            CORE.strict_json_bytes(b'{"x":1,"x":2}', "duplicate fixture")

    def test_read_invalid_global_winner_cannot_erase_read_valid_candidate(self):
        saved = (CORE.RATES, CORE.MODES, CORE.PROFILES, CORE.MATRICES)
        calls: list[tuple[str, tuple[int, ...]]] = []

        def fake_score(_panel, choices, rate, mode, geometry, include_allocations=False):
            choice_tuple = tuple(choices)
            calls.append((geometry, choice_tuple))
            below = all(choice == 1 for choice in choice_tuple)
            amplification = 1.25 if below else 2.25
            return {
                "valid": True,
                "requested_rate_bpw": rate,
                "physical_rate_bpw": rate,
                "ideal_relative_mse": 0.1 if not below else 0.2,
                "F": 0.1 if not below else 0.2,
                "s_bpw": 1.0,
                "source_energy": 1.0,
                "tail_distortion_sse": 0.0,
                "bulk_ideal_distortion_sse": 0.1 if not below else 0.2,
                "total_distortion_sse": 0.1 if not below else 0.2,
                "mode": mode,
                "geometry": geometry,
                "choices": list(choice_tuple),
                "side_ledger": {"tail_and_codebook_bits": sum(choice_tuple)},
                "read_ledger": {
                    "below_2x": below,
                    "maximum_cold_logical_amplification": amplification,
                    "maximum_cold_page_amplification": amplification,
                    "experts": [
                        {
                            "cold_logical_amplification": amplification,
                            "cold_page_amplification": amplification,
                        }
                        for _ in range(CORE.EXPERTS)
                    ],
                },
                "allocations": [] if include_allocations else None,
            }

        try:
            CORE.RATES = (2.15,)
            CORE.MODES = ("finite_fp16",)
            CORE.PROFILES = ({"levels": 2}, {"levels": 2})
            CORE.MATRICES = 2
            original_score = CORE.score
            CORE.score = fake_score
            search = CORE.search_panel({}, 4)
        finally:
            CORE.score = original_score
            CORE.RATES, CORE.MODES, CORE.PROFILES, CORE.MATRICES = saved

        rows = search["2.15"]["finite_fp16"]
        self.assertEqual(rows["raw_uniform_global_diagnostic"]["choices"], [0, 0])
        self.assertFalse(rows["raw_uniform_global_diagnostic"]["read_ledger"]["below_2x"])
        for key in ("raw_uniform_best", "raw_adaptive", "support_xklt_uniform"):
            self.assertEqual(rows[key]["choices"], [1, 1])
            self.assertTrue(rows[key]["read_ledger"]["below_2x"])
        ledger = rows["read_valid_selection_ledger"]
        self.assertTrue(ledger["every_uniform_profile_evaluated"])
        self.assertTrue(ledger["every_support_xklt_profile_evaluated"])
        self.assertTrue(ledger["every_coordinate_trial_profile_evaluated"])
        self.assertIn(("raw", (0, 0)), calls)
        self.assertIn(("raw", (1, 1)), calls)

    def test_every_nonfinite_individual_or_aggregate_decision_score_rejects(self):
        def base_rows():
            return [
                {**row("free_lloyd", 0.18, 0.17), "control_s_bpw": [0.01] * 4, "control_mean_s_bpw": 0.01},
                {**row("zero_tail_error", 0.17, 0.16), "control_s_bpw": [0.01] * 4, "control_mean_s_bpw": 0.01},
                {**row("finite_fp16", 0.15, 0.14), "control_s_bpw": [0.01] * 4, "control_mean_s_bpw": 0.01},
            ]

        for bad in (math.nan, math.inf, -math.inf):
            for field in ("qwen_s_bpw", "qwen_excess_s_bpw", "rate", "control_mean_s_bpw"):
                rows = base_rows()
                rows[1][field] = bad
                with self.subTest(value=bad, field=field), self.assertRaisesRegex(RuntimeError, "non-finite"):
                    CORE.decision_from_calibrated(rows)
            rows = base_rows()
            rows[0]["control_s_bpw"][2] = bad
            with self.subTest(value=bad, field="control_s_bpw[2]"), self.assertRaisesRegex(RuntimeError, "non-finite"):
                CORE.decision_from_calibrated(rows)

    def test_nonfinite_scored_candidates_reject_before_ranking(self):
        expert = {"cold_logical_amplification": 1.0, "cold_page_amplification": 1.0}
        candidate = {
            "valid": True, "requested_rate_bpw": 2.15, "physical_rate_bpw": 2.15,
            "ideal_relative_mse": 0.8, "F": math.inf, "s_bpw": 0.1,
            "source_energy": 1.0, "tail_distortion_sse": 0.0,
            "bulk_ideal_distortion_sse": 0.8, "total_distortion_sse": 0.8,
            "choices": [0] * CORE.MATRICES,
            "side_ledger": {"tail_and_codebook_bits": 0},
            "read_ledger": {
                "below_2x": True,
                "maximum_cold_logical_amplification": 1.0,
                "maximum_cold_page_amplification": 1.0,
                "experts": [dict(expert) for _ in range(CORE.EXPERTS)],
            },
        }
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            CORE.best_scored([candidate], "positive infinity adversary", require_read_valid=True)


if __name__ == "__main__":
    unittest.main()
