#!/usr/bin/env python3
"""Unit and tamper tests for the independent BiSCo state replay."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from . import independent_replay as replay
except ImportError:  # direct invocation from this directory
    import independent_replay as replay  # type: ignore


HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("BISCO_RUN_DIR", str(HERE / "run_1"))).resolve()
RESULT = RUN / "bisco_raw_mse_result.json"


class IndependentReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_frozen_schema_counts_and_offsets(self) -> None:
        self.assertEqual(replay.STATE_VALUES, 18_056)
        self.assertEqual(replay.STATE_VALUES_PER_ROLE, 9_028)
        self.assertEqual(replay.EXPECTED_STATE_BYTES, 72_224)
        self.assertEqual(replay.DECODER_VALUES, 9_024)
        self.assertEqual(replay.EXPECTED_DECODER_BYTES, 18_048)
        self.assertEqual(replay.EXPECTED_STATE_SCHEMA[0]["offset_values"], 0)
        self.assertEqual(
            replay.EXPECTED_STATE_SCHEMA[-1]["offset_values"] + replay.EXPECTED_STATE_SCHEMA[-1]["values"],
            replay.STATE_VALUES,
        )
        self.assertEqual(
            replay.EXPECTED_DECODER_SCHEMA[-1]["offset_values"] + replay.EXPECTED_DECODER_SCHEMA[-1]["values"],
            replay.DECODER_VALUES,
        )

    def test_actual_states_parse_from_independent_schema(self) -> None:
        for domain in replay.DOMAINS:
            state = replay.parse_state_file(
                RUN / f"{domain}_training_state.fp32.bin", replay.EXPECTED_STATE_SCHEMA
            )
            self.assertEqual(tuple(state), replay.ROLES)
            for role in replay.ROLES:
                self.assertEqual(tuple(state[role]), tuple(name for name, _ in replay.STATE_PARAMETER_SHAPES))
                for name, shape in replay.STATE_PARAMETER_SHAPES:
                    self.assertEqual(state[role][name].shape, shape)
                    self.assertEqual(state[role][name].dtype, np.float32)

    def test_fp16_decoder_is_literal_state_projection(self) -> None:
        for domain in replay.DOMAINS:
            state = replay.parse_state_file(
                RUN / f"{domain}_training_state.fp32.bin", replay.EXPECTED_STATE_SCHEMA
            )
            expected = replay.decoder_bytes_from_state(state)
            actual = (RUN / f"{domain}_aux_up_down_decoder.fp16.bin").read_bytes()
            self.assertEqual(actual, expected)
            self.assertEqual(len(expected), replay.EXPECTED_DECODER_BYTES)

    def test_one_bit_decoder_tamper_no_longer_equals_state(self) -> None:
        state = replay.parse_state_file(RUN / "qwen_training_state.fp32.bin", replay.EXPECTED_STATE_SCHEMA)
        expected = replay.decoder_bytes_from_state(state)
        tampered = bytearray(expected)
        tampered[len(tampered) // 2] ^= 1
        self.assertNotEqual(bytes(tampered), expected)

    def test_state_schema_offset_tamper_is_rejected(self) -> None:
        schema = copy.deepcopy(replay.EXPECTED_STATE_SCHEMA)
        schema[4]["offset_values"] += 1
        with self.assertRaises(replay.AuditFailure):
            replay.parse_state_file(RUN / "qwen_training_state.fp32.bin", schema)

    def test_history_and_decision_are_exact(self) -> None:
        evidence = replay.enforce_history_and_decision(self.result)
        self.assertEqual(evidence["history_updates_exact"], [256, 512])
        self.assertEqual(evidence["decision"], "HARD_KILL_D16_SHALLOW_BEFORE_PINNED")
        self.assertTrue(evidence["early_kill_exact"]["kill"])

    def test_history_extra_field_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.result)
        tampered["training"]["history"][1]["unbound_note"] = "would previously be ignored"
        with self.assertRaises(replay.AuditFailure):
            replay.enforce_history_and_decision(tampered)

    def test_history_base_statistic_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.result)
        tampered["training"]["history"][0]["evaluation"]["per_matrix"][0]["qwen_sse"] += 1e-6
        with self.assertRaises(replay.AuditFailure):
            replay.enforce_history_and_decision(tampered)

    def test_decision_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.result)
        tampered["decision"] = "NO_PROMOTION_FROM_AUXILIARY_D16"
        with self.assertRaises(replay.AuditFailure):
            replay.enforce_history_and_decision(tampered)

    def test_final_history_unlink_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.result)
        tampered["final_evaluation"]["D_Qwen"] = math.nextafter(
            tampered["final_evaluation"]["D_Qwen"], math.inf
        )
        with self.assertRaises(replay.AuditFailure):
            replay.enforce_history_and_decision(tampered)

    def test_seal_round_trip_and_tamper_detection(self) -> None:
        sealed = replay.seal_receipt({"protocol": "unit", "verified": True, "nested": {"x": 1.25}})
        digest = replay.verify_receipt_seal(sealed)
        self.assertEqual(digest, sealed["receipt_seal"]["sha256"])
        tampered = copy.deepcopy(sealed)
        tampered["nested"]["x"] = 1.5
        with self.assertRaises(replay.AuditFailure):
            replay.verify_receipt_seal(tampered)

    def test_gaussian_seed_is_filename_bound_and_stable(self) -> None:
        self.assertEqual(replay.gaussian_seed("l15e24_up.bf16.bin"), 8_425_621_512_093_229_098)
        self.assertEqual(replay.gaussian_seed("l15e24_down.bf16.bin"), 11_057_747_418_777_782_567)
        self.assertNotEqual(
            replay.gaussian_seed("l15e24_up.bf16.bin"), replay.gaussian_seed("l15e56_up.bf16.bin")
        )

    def test_down_role_canonical_transpose(self) -> None:
        raw = np.arange(replay.VALUES_PER_MATRIX, dtype=np.uint32).astype(np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "l15e24_down.bf16.bin"
            raw.astype("<u2").tofile(path)
            actual = replay.load_bf16_canonical(path, "down")
        expected_bits = (raw.astype(np.uint32) << np.uint32(16)).reshape(replay.COLS, replay.ROWS).T
        np.testing.assert_array_equal(actual.view(np.uint32), expected_bits)

    def test_greedy_replay_does_not_increase_its_initial_sse(self) -> None:
        generator = np.random.default_rng(9)
        params = {}
        for name, shape in replay.STATE_PARAMETER_SHAPES:
            params[name] = generator.normal(0.0, 0.08, size=shape).astype(np.float32)
        source = generator.normal(size=(11, replay.CHUNK_DIMENSION)).astype(np.float32)
        q1, y1 = replay.initial_stage(params, "s1", source, np)
        q2, y2 = replay.initial_stage(params, "s2", source - y1, np)
        baseline = float(np.sum((y1 + y2 - source).astype(np.float64) ** 2))
        reconstruction, final_q1, final_q2 = replay.independent_reconstruct(params, source, np)
        final = float(np.sum((reconstruction - source).astype(np.float64) ** 2))
        self.assertLessEqual(final, baseline)
        for code in (q1, q2, final_q1, final_q2):
            np.testing.assert_allclose(np.abs(code), np.float32(1.0 / math.sqrt(replay.BITS_PER_STAGE)))

    def test_actual_result_and_artifacts_are_frozen(self) -> None:
        self.assertEqual(replay.sha256_file(RESULT), replay.EXPECTED_RESULT_SHA256)
        models, evidence = replay.parse_and_bind_models(self.result, RUN)
        self.assertEqual(set(models), set(replay.DOMAINS))
        for domain in replay.DOMAINS:
            self.assertTrue(evidence[domain]["decoder_equals_state_rounded_fp16"])
            self.assertEqual(
                evidence[domain]["decoder_sha256"], evidence[domain]["decoder_expected_bytes_sha256"]
            )

    def test_sealed_run_1_receipt_rebinds_local_inputs_and_math(self) -> None:
        receipt_path = RUN / "independent_replay_receipt.json"
        verified = replay.verify_receipt_file(receipt_path)
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["result_sha256"], replay.EXPECTED_RESULT_SHA256)
        self.assertEqual(verified["script_sha256"], replay.sha256_file(Path(replay.__file__).resolve()))
        self.assertEqual(verified["decision"], "HARD_KILL_D16_SHALLOW_BEFORE_PINNED")
        self.assertAlmostEqual(verified["D_Qwen_fp64"], 0.11020813758494276, places=15)
        self.assertAlmostEqual(verified["D_Gaussian_fp64"], 0.1096131632504529, places=15)


if __name__ == "__main__":
    unittest.main()
