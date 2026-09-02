#!/usr/bin/env python3
"""Standard-library tests for the inert fixed-pentad source package."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import unittest

import contract
import cupy_anchor
import inert_entrypoint
import plan_snapshot
import stage1_core


def _feature_row(descriptor: tuple[int, str, int, int]) -> list[float]:
    label = f"{descriptor[0]}|{descriptor[1]}|{descriptor[2]}|{descriptor[3]}"
    values = []
    for column in range(5):
        digest = hashlib.sha256(f"pentad-test|{column}|{label}".encode("ascii")).digest()
        word = int.from_bytes(digest[:4], "little")
        values.append((float(word) / float(2**32 - 1)) * 2.0 - 1.0)
    return values


def _synthetic_payload():
    fit_keys, score_keys = plan_snapshot.stage1_keys()
    descriptors = {
        "fit": [plan_snapshot.parse_key(key) for key in fit_keys],
        "score": [plan_snapshot.parse_key(key) for key in score_keys],
    }
    coefficients = (0.5, -0.25, 0.125, 0.0625, -0.03125)
    targets = {}
    anchors = {}
    for split in ("fit", "score"):
        anchors[split] = [_feature_row(row) for row in descriptors[split]]
        targets[split] = [
            0.25 + math.fsum(value * coefficient for value, coefficient in zip(row, coefficients))
            for row in anchors[split]
        ]
    return targets, descriptors, anchors


class _FakeArray(list):
    def tolist(self):
        return list(self)


class _FakeGenerator:
    def __init__(self, seed: int):
        self.seed = seed

    def permutation(self, count: int):
        values = list(range(count))
        if count:
            shift = self.seed % count
            values = values[shift:] + values[:shift]
            if self.seed & 1:
                values.reverse()
        self.seed += 1
        return _FakeArray(values)


class _FakeRandom:
    @staticmethod
    def PCG64(seed: int):
        return int(seed)

    @staticmethod
    def Generator(seed: int):
        return _FakeGenerator(int(seed))


class _FakeNumpy:
    random = _FakeRandom()
    uint64 = "uint64"
    uint8 = "uint8"

    @staticmethod
    def asarray(values, dtype=None):
        return {"values": tuple(values), "dtype": dtype}


class ContractTests(unittest.TestCase):
    def test_constants_and_physical_ledger(self):
        self.assertEqual(len(contract.SEEDS_U32), 5)
        self.assertEqual(contract.identity_set(), tuple(
            (expert, role)
            for expert in contract.SELECTION_EXPERTS
            for role in contract.ROLES
            if not (expert == 0 and role == "up")
        ))
        self.assertEqual(len(contract.identity_set()), 23)
        self.assertAlmostEqual(contract.REQUIRED_CAPTURE, 0.1910966610577134, places=15)
        ledger = contract.physical_ledger()
        self.assertAlmostEqual(ledger["six_expert_side_bpw"], 0.0000904224537037037, places=19)
        self.assertAlmostEqual(ledger["conservative_page_read_amplification"], 1.175, places=15)
        self.assertTrue(ledger["strictly_below_2x"])

    def test_strict_json(self):
        self.assertEqual(contract.strict_json_loads(b'{"a":1}'), {"a": 1})
        with self.assertRaises(contract.ContractError):
            contract.strict_json_loads(b'{"a":1,"a":2}')
        with self.assertRaises(contract.ContractError):
            contract.strict_json_loads(b'{"a":NaN}')
        with self.assertRaises(contract.ContractError):
            contract.canonical_json({"a": float("inf")})

    def test_fp16_words(self):
        for value in (0.0, 0.5, -0.25, 1.0, 65504.0):
            word = contract.fp16_word(value)
            self.assertTrue(math.isfinite(contract.decode_fp16_word(word)))
        with self.assertRaises(contract.ContractError):
            contract.fp16_word(-0.0)
        with self.assertRaises(contract.ContractError):
            contract.fp16_word(float("nan"))
        with self.assertRaises(contract.ContractError):
            contract.decode_fp16_word(0x7C00)

    def test_joint_fit_and_rank_rejection(self):
        anchors = []
        targets = []
        coefficients = (0.5, -0.25, 0.125, 0.0625, -0.03125)
        for index in range(80):
            row = _feature_row((8, "up", index % 768, (index * 37) % 2048))
            anchors.append(row)
            targets.append(0.25 + math.fsum(a * b for a, b in zip(row, coefficients)))
        fit = contract.fit_decoded_fp16(anchors[:40], targets[:40])
        score = contract.score_decoded_fit(fit, anchors[40:], targets[40:])
        self.assertEqual(len(fit["fp16_words"]), 6)
        self.assertLess(fit["condition"], 20.0)
        self.assertGreater(score["capture"], 0.999)
        bad = [[float(index)] * 5 for index in range(32)]
        with self.assertRaises(contract.ContractError):
            contract.fit_decoded_fp16(bad, [float(index) for index in range(32)])

    def test_expert_not_matrix_jackknife(self):
        rows = []
        for expert, role in contract.identity_set():
            rows.append({"expert": expert, "role": role, "sse": 0.75, "source_energy": 1.0, "capture": 0.25})
        uncertainty = contract.expert_jackknife(rows)
        self.assertEqual(len(uncertainty["delete_expert_values"]), 12)
        self.assertLess(uncertainty["standard_error"], 1e-14)
        result = contract.decision(rows)
        self.assertTrue(result["survives"])
        rows[0] = {**rows[0], "sse": 1.01, "capture": -0.01}
        self.assertFalse(contract.decision(rows)["survives"])


class PlanTests(unittest.TestCase):
    def test_historical_plan_hashes_and_disjointness(self):
        fit_keys, score_keys = plan_snapshot.stage1_keys()
        self.assertEqual(len(fit_keys), 2048)
        self.assertEqual(len(score_keys), 2048)
        self.assertFalse(set(fit_keys) & set(score_keys))
        self.assertEqual(
            hashlib.sha256(("\n".join(fit_keys) + "\n").encode("ascii")).hexdigest(),
            plan_snapshot.FIT_KEY_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(("\n".join(score_keys) + "\n").encode("ascii")).hexdigest(),
            plan_snapshot.SCORE_KEY_SHA256,
        )


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets, cls.descriptors, cls.anchors = _synthetic_payload()
        cls.primary = stage1_core.evaluate(cls.targets, cls.descriptors, cls.anchors)

    def test_full_geometry_and_joint_fit(self):
        self.assertEqual(len(self.primary["records"]), 23)
        self.assertGreater(self.primary["aggregate"]["capture"], 0.999)
        self.assertEqual(len(self.primary["delete_one_expert"]["delete_expert_values"]), 12)
        self.assertTrue(all(len(row["fp16_words_hex"]) == 6 for row in self.primary["records"]))

    def test_control_permutation_closure(self):
        indices = stage1_core._validate_inputs(self.targets, self.descriptors, self.anchors)
        permutations = stage1_core.frozen_scramble_permutations(_FakeNumpy, indices, 26090100)
        self.assertEqual(len(permutations), 46)
        controlled = stage1_core.evaluate(self.targets, self.descriptors, self.anchors, permutations)
        self.assertEqual(len(controlled["records"]), 23)
        with self.assertRaises(contract.ContractError):
            stage1_core.frozen_scramble_permutations(_FakeNumpy, indices, 1)

    def test_result_gate_and_control_ordering(self):
        controls = {
            "rows": [
                {"replicate": index, "seed": seed, "capture": -0.01, "centered_capture": -0.01}
                for index, seed in enumerate(contract.CONTROL_SEEDS)
            ],
            "mean_capture": -0.01,
            "maximum_capture": -0.01,
            "minimum_capture": -0.01,
            "mc_standard_error": 0.0,
        }
        result = stage1_core.build_result(self.primary, controls)
        self.assertTrue(result["gate"]["survives"])
        controls["rows"][-1]["capture"] = 1.0
        result = stage1_core.build_result(self.primary, controls)
        self.assertFalse(result["gate"]["survives"])

    def test_cupy_mapping_is_inert_and_five_wide(self):
        arrays = cupy_anchor.coordinate_arrays(_FakeNumpy, [(8, "up", 1, 2), (8, "down", 1, 2)])
        self.assertEqual(len(arrays), 6)
        self.assertTrue(all(len(row["values"]) == 2 for row in arrays))
        self.assertEqual(
            cupy_anchor.CUDA_SOURCE_SHA256,
            "580ea565670dbc41319abc3277d733d9160e7043ffec25e07df914ae8bb64701",
        )
        self.assertEqual(
            cupy_anchor.CUDA_SOURCE_SHA256,
            hashlib.sha256(cupy_anchor.CUDA_SOURCE.encode("utf-8")).hexdigest(),
        )
        self.assertIn("curand_Philox4x32_10", cupy_anchor.CUDA_SOURCE)


class RefusalTests(unittest.TestCase):
    def test_entrypoint_refuses(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            status = inert_entrypoint.main()
        self.assertEqual(status, 3)
        receipt = json.loads(stream.getvalue())
        self.assertEqual(receipt["status"], "REFUSED_NO_PAYLOAD_OR_RUN_AUTHORITY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
