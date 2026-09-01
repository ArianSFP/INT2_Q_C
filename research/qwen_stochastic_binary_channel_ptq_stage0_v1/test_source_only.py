#!/usr/bin/env python3
"""Pure standard-library regression tests; never imports CuPy or opens panel payloads."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


runner = load("qsb_stage0_runner", HERE / "stage0_screen.py")
verifier = load("qsb_stage0_verifier", HERE / "verify_design.py")


class SourceOnlyTests(unittest.TestCase):
    def test_strict_json_accepts_finite(self):
        self.assertEqual(runner.strict_json(b'{"a":1.25,"b":[2]}'), {"a": 1.25, "b": [2]})

    def test_strict_json_rejects_duplicate(self):
        with self.assertRaises(ValueError):
            runner.strict_json(b'{"a":1,"a":2}')

    def test_strict_json_rejects_nonfinite(self):
        for raw in (b'{"a":NaN}', b'{"a":Infinity}', b'{"a":-Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                runner.strict_json(raw)

    def test_rate_constants(self):
        self.assertEqual(runner.FIT_TARGET_FILL, 0.965)
        self.assertEqual(runner.EXECUTION_LIMIT_FILL, 0.97)
        self.assertAlmostEqual(runner.EXECUTION_LIMIT_FILL - runner.FIT_TARGET_FILL, 0.005)
        expected = (("QSB215", 7_618_560, 155 / 72),
                    ("QSB230", 8_159_232, 83 / 36),
                    ("QSB250", 8_847_360, 5 / 2))
        for row, (name, byte_count, rate) in zip(runner.RATE_CELLS, expected):
            self.assertEqual(row["cell"], name)
            self.assertEqual(row["container_bytes"], byte_count)
            self.assertEqual(row["container_bytes"] * 8 / runner.PANEL_VALUES, rate)
            self.assertLess(row["cold_read_amplification"], 2.0)

    def test_policy_limit_is_not_physical_reservoir(self):
        payload = 10_000
        ideal_kl = 9_675
        self.assertGreater(ideal_kl, runner.FIT_TARGET_FILL * payload)
        self.assertLess(ideal_kl, runner.EXECUTION_LIMIT_FILL * payload)
        self.assertGreater(payload - ideal_kl, 0)
        source = (HERE / "stage0_screen.py").read_text(encoding="utf-8")
        self.assertIn("EXCEEDS_PREREGISTERED_0P97_EXECUTION_LIMIT", source)
        self.assertIn("EXCEEDS_PHYSICAL_RESERVOIR", source)
        self.assertNotIn("EXCEEDS_FROZEN_RESERVOIR", source)

    def test_jackknife_equal_experts(self):
        row = runner.jackknife([1.0] * 6, [10.0] * 6)
        self.assertAlmostEqual(row["estimate"], 0.9)
        self.assertAlmostEqual(row["jackknife_se"], 0.0)

    @staticmethod
    def oracle(capture, lower=None, upper=None):
        return {"uncertainty": {"lower_three_se": capture if lower is None else lower,
                                "upper_three_se": capture if upper is None else upper},
                "experts": [{"capture": capture} for _ in range(6)],
                "roles": [{"capture": capture} for _ in range(3)]}

    def test_control_gate_promotes_only_strict_advantage(self):
        qwen = self.oracle(0.98, lower=0.975, upper=0.985)
        controls = [{"oracle": self.oracle(0.96, lower=0.95, upper=0.97)} for _ in range(8)]
        row = runner.apply_control_gate(qwen, controls)
        self.assertEqual(row["status"], "POLICY_HOLD_FOR_OPERATIONAL_IMPLEMENTATION")
        self.assertGreater(row["aggregate_margin"], 0.0)

    def test_control_gate_rejects_fold_failure(self):
        qwen = self.oracle(0.98, lower=0.975, upper=0.985)
        controls = [{"oracle": self.oracle(0.96, lower=0.95, upper=0.97)} for _ in range(8)]
        controls[0]["oracle"]["roles"][1]["capture"] = 0.981
        row = runner.apply_control_gate(qwen, controls)
        self.assertEqual(row["status"], "POLICY_REJECT_SOURCE_NOT_ABOVE_MATCHED_CONTROLS")

    def test_control_gate_rejects_incomparable_rate(self):
        qwen = self.oracle(0.98)
        controls = [{"oracle": self.oracle(0.96)} for _ in range(7)] + [
            {"oracle": {"status": "HARD_KILL_IDEAL_KL_EXCEEDS_FROZEN_RESERVOIR"}}]
        self.assertEqual(runner.apply_control_gate(qwen, controls)["status"],
                         "POLICY_REJECT_CONTROL_RATE_INCOMPARABLE")

    def test_verifier_strict_json_rejects_duplicate_and_nonfinite(self):
        with self.assertRaises(ValueError):
            verifier.strict_json(b'{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            verifier.strict_json(b'{"x":NaN}')

    def test_held_read_rejects_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                verifier.held_regular_read(Path(temporary))

    def test_runner_held_read_and_directory_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = root / "x.bin"
            regular.write_bytes(b"held bytes")
            self.assertEqual(runner.held_regular_read(regular), b"held bytes")
            with self.assertRaises(ValueError):
                runner.held_regular_read(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
