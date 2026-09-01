#!/usr/bin/env python3
"""Pure-stdlib tests for the frozen FOSP-ARX source package."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORACLE_PATH = ROOT / "free_order_oracle.py"
LOCK_PATH = ROOT / "protocol_lock.json"
BINDINGS_PATH = ROOT / "source_bindings.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_oracle():
    spec = importlib.util.spec_from_file_location("fosp_source_only", ORACLE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceOnlyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.oracle = load_oracle()
        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        cls.bindings = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))

    def test_zero_bit_variant_is_explicitly_ineligible(self) -> None:
        metric = self.lock["metric_compatibility"]
        self.assertEqual(metric["zero_bit_deployed_gauge"], "INELIGIBLE_IMMEDIATE_KILL")
        self.assertEqual(metric["information_lower_bound_bits"], 6260)
        self.assertEqual(metric["physical_factoradic_bytes"], 783)
        self.assertTrue(self.lock["promotion"]["zero_bit_variant_can_never_be_promoted"])

    def test_factoradic_bound_and_roundtrip(self) -> None:
        self.assertEqual(self.oracle.ceil_log2_factorial(768), 6260)
        for permutation in (
            (0,),
            (0, 1, 2, 3),
            (3, 2, 1, 0),
            (2, 0, 3, 1),
        ):
            rank = self.oracle.rank_permutation(permutation)
            self.assertEqual(self.oracle.unrank_permutation(len(permutation), rank), permutation)
        self.assertLess(math.factorial(768) - 1, 1 << (783 * 8))

    def test_exact_joint_permutation_function_identity_fixture(self) -> None:
        # Tiny direct SwiGLU fixture.  Gate/Up rows and Down columns move
        # together.  This proves function equivalence but deliberately does
        # not alter the original-coordinate MSE contract above.
        gate = [[0.2, -0.7], [0.4, 0.3], [-0.8, 0.1]]
        up = [[0.5, 0.6], [-0.2, 0.9], [0.7, -0.4]]
        down = [[0.3, -0.1, 0.8], [-0.5, 0.6, 0.2]]
        vector = [0.25, -0.75]
        permutation = [2, 0, 1]

        def matvec(matrix, values):
            return [math.fsum(a * b for a, b in zip(row, values)) for row in matrix]

        def silu(value):
            return value / (1.0 + math.exp(-value))

        def evaluate(g, u, d):
            gv = matvec(g, vector)
            uv = matvec(u, vector)
            hidden = [silu(a) * b for a, b in zip(gv, uv)]
            return matvec(d, hidden)

        pg = [gate[index] for index in permutation]
        pu = [up[index] for index in permutation]
        pd = [[row[index] for index in permutation] for row in down]
        before = evaluate(gate, up, down)
        after = evaluate(pg, pu, pd)
        for left, right in zip(before, after):
            self.assertAlmostEqual(left, right, places=15)

    def test_original_coordinate_scatter_needs_permutation(self) -> None:
        original = [10, 20, 30, 40]
        permutation = (2, 0, 3, 1)
        encoded = [original[index] for index in permutation]
        restored = [None] * len(original)
        for encoded_index, original_index in enumerate(permutation):
            restored[original_index] = encoded[encoded_index]
        self.assertEqual(restored, original)
        # Treating encoded positions as original coordinates is wrong.
        self.assertNotEqual(encoded, original)

    def test_all_three_roles_are_bound_for_every_expert(self) -> None:
        self.assertEqual(len(self.bindings["experts"]), 2)
        for expert in self.bindings["experts"]:
            self.assertEqual([row["role"] for row in expert["roles"]], ["gate", "up", "down"])
            self.assertEqual(len({row["sha256"] for row in expert["roles"]}), 3)
            for row in expert["roles"]:
                self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")

    def test_bindings_are_hash_frozen(self) -> None:
        expected = self.lock["execution_firewall"]["source_bindings_sha256"]
        self.assertEqual(sha256(BINDINGS_PATH), expected)
        self.assertEqual(self.oracle.BINDINGS_SHA256, expected)

    def test_runtime_cli_cannot_select_a_manifest_or_matrix(self) -> None:
        tree = ast.parse(ORACLE_PATH.read_text(encoding="utf-8"), filename=ORACLE_PATH.name)
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("--workspace-root", literals)
        self.assertIn("--output", literals)
        self.assertNotIn("--plan", literals)
        self.assertNotIn("--manifest", literals)
        self.assertNotIn("--source", literals)

    def test_heavy_imports_are_deferred_until_main(self) -> None:
        tree = ast.parse(ORACLE_PATH.read_text(encoding="utf-8"), filename=ORACLE_PATH.name)
        top_level_roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_roots.add(node.module.split(".")[0])
        self.assertFalse({"cupy", "numpy", "scipy", "torch"} & top_level_roots)

    def test_exact_rate_and_read_ledgers(self) -> None:
        rows = {
            float(row["requested_rate_bpw"]): row
            for row in self.lock["rate_and_read"]["rows"]
        }
        for rate in self.oracle.RATES:
            observed = self.oracle.frame_ledger(rate, 0)
            expected = rows[rate]
            self.assertEqual(observed["frame_bytes"], expected["frame_bytes"])
            self.assertEqual(observed["actual_rate_bpw"], expected["actual_rate_bpw"])
            self.assertEqual(
                observed["cold_page_amplification"], expected["cold_page_amplification"]
            )
            self.assertTrue(observed["strictly_below_2x"])
            self.assertEqual(observed["logical_byte_read_amplification"], 1.0)

    def test_coefficient_ledgers_charge_every_selected_edge(self) -> None:
        modes = {row["name"]: row for row in self.lock["eligible_codec"]["coefficient_modes"]}
        self.assertEqual(modes["diag3_fp16_oracle_bridge"]["coefficient_bits"], 767 * 3 * 16)
        self.assertEqual(modes["full3x3_fp16_oracle_bridge"]["coefficient_bits"], 767 * 9 * 16)
        self.assertEqual(
            modes["diag3_fixed_nibble"]["coefficient_bits"], math.ceil(767 * 3 * 4 / 8) * 8
        )
        self.assertEqual(
            modes["full3x3_fixed_nibble"]["coefficient_bits"],
            math.ceil(767 * 9 * 4 / 8) * 8,
        )

    def test_create_new_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            self.oracle._write_create_new(output, {"ok": True})
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                self.oracle._write_create_new(output, {"ok": False})


if __name__ == "__main__":
    unittest.main(verbosity=2)
