from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import run_gate
import run_source_free_cupy as preflight
import source_free_fixture as fixture


class ClosureAndLineageTests(unittest.TestCase):
    def test_vendored_scientific_inputs_are_byte_exact(self):
        expected = {
            "clustered_ib_core.py": "25e84b9d5e598a72984e48cb5593c41725d096e36082b20b3d47a78f2100e340",
            "cupy_worker.py": "a34ca17dd8f76afa0331bb56d5b5dec26dcde693d05755ea2ca342a76a6badfc",
            "panel_lock.json": "1da2d993aee033b6dc9d165dc8d5482eecfb276d30e5e398edc388a83b8f5af5",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest(), digest)

    def test_panel_binding(self):
        panel = json.loads((PACKAGE / "panel_lock.json").read_text())
        self.assertEqual(panel["model"], "Qwen/Qwen3-30B-A3B")
        self.assertEqual(panel["layer"], 15)
        self.assertEqual(panel["experts"], list(range(0, 128, 8)))
        self.assertEqual(sum(row["bytes"] for row in panel["files"]), 100663296)

    def test_live_manifest(self):
        import verify_source
        digest = hashlib.sha256((PACKAGE / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()
        receipt = verify_source.verify(PACKAGE, digest)
        self.assertEqual(
            receipt["status"],
            "PASS_SOURCE_CLOSED_R3_REQUIRES_NEW_INDEPENDENT_REVIEW",
        )


class RuntimeAndOneUseTests(unittest.TestCase):
    def test_wrong_authorization_holds_without_path_or_runtime(self):
        with mock.patch.object(run_gate, "Path", side_effect=AssertionError("Path touched")), \
             mock.patch.object(run_gate, "_validate_runtime",
                               side_effect=AssertionError("runtime touched")):
            self.assertEqual(run_gate.main(["--authorization", "wrong"]), 2)

    def test_numpy_closure_is_before_claim(self):
        source = inspect.getsource(run_gate.main)
        self.assertLess(source.index("_validate_runtime()"), source.index("_claim_once("))
        closure = inspect.getsource(run_gate._verify_numpy_record_closure)
        for token in ("NUMPY_VERSION", "NUMPY_FILE_SHA256", "NUMPY_RECORD_SHA256",
                      "numpy.libs/", "native_checked", "hashlib.sha256(data).digest()"):
            self.assertIn(token, closure)

    def test_new_fixed_capability(self):
        self.assertEqual(
            run_gate.AUTHORIZATION_PHRASE,
            "EXECUTE_AUTHENTICATED_QWEN_L15_CBIB1_V0_R3_ONCE",
        )
        self.assertEqual(
            run_gate.OUTPUT_PARENT,
            "/tmp/codex_cbib1_qwen_l15_oneuse_20260903_r3",
        )
        self.assertEqual(run_gate.NUMPY_VERSION, "2.5.2")

    def test_claim_is_atomic_and_persistent(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "claim.json"
            run_gate._claim_once(path, "a" * 64)
            first = path.read_bytes()
            with self.assertRaises(FileExistsError):
                run_gate._claim_once(path, "a" * 64)
            self.assertEqual(path.read_bytes(), first)


class PreflightContractTests(unittest.TestCase):
    def test_r3_fixture_exact_scale_geometry_and_targeted_survivor(self):
        self.assertEqual(fixture.BLOCKS_PER_ROLE, 131072 // 2048)
        self.assertEqual(
            fixture.SCALE_BYTES_PER_EXPERT,
            fixture.ROLES * fixture.BLOCKS_PER_ROLE * fixture.SCALE_BYTES_PER_VALUE,
        )
        self.assertEqual(fixture.SCALE_BYTES_PER_EXPERT, 256)
        self.assertEqual(fixture.LATENT_PROBABILITY, 0.5)
        self.assertEqual(fixture.SIGN_FLIP_PROBABILITY, 0.105)
        receipt = json.loads((PACKAGE / "TARGETED_REGRESSION_RECEIPT.json").read_text())
        self.assertEqual(
            receipt["status"],
            "PASS_TARGETED_GROUP2_SOURCE_AND_STRICT_READ_SURVIVOR",
        )
        self.assertEqual(receipt["fixture_labels_sha256"], fixture.fixture_sha256())
        self.assertEqual(receipt["scale_bytes_per_expert"], 256)
        source = (PACKAGE / "run_fixture_survivor_regression.py").read_text()
        for token in ("crossfit_group_size(", "packet_requirements(",
                      "physical_read_envelope(", "strictly_below_2x"):
            self.assertIn(token, source)

    def test_evaluator_schema_regression_reconstructs_counts(self):
        labels = [[0, 1, 2, 3], [3, 2, 1, 0]]
        cpu_eval = {"assignments": [0, 0, 1, 1]}
        gpu_eval = {
            "assignments": [0, 0, 1, 1],
            "test_latent_counts": [2, 2],
            "test_conditional_counts": [
                [[1, 1, 0, 0], [0, 0, 1, 1]],
                [[0, 0, 1, 1], [1, 1, 0, 0]],
            ],
        }
        self.assertEqual(
            preflight._compare_assignment_count_evidence(
                labels, cpu_eval, gpu_eval, lambda value: value, "regression"
            ),
            4,
        )
        self.assertNotIn("test_latent_counts", cpu_eval)
        self.assertNotIn("test_conditional_counts", cpu_eval)

    def test_complete_production_geometry_contract(self):
        contract = json.loads((PACKAGE / "EXPECTED_PREFLIGHT.json").read_text())
        fixture = contract["fixture"]
        self.assertEqual(fixture["experts"], 16)
        self.assertEqual(fixture["fold_count"], 8)
        self.assertEqual(fixture["superblock_values"], 2048)
        self.assertEqual(fixture["group_sizes"], [2, 4, 8, 16])
        self.assertEqual(contract["acceptance"]["controls_executed"], 8)
        source = (PACKAGE / "run_source_free_cupy.py").read_text()
        for token in ("for group_size in (2, 4, 8, 16)",
                      "for fold in range(FOLD_COUNT)",
                      "training_assignments_checked",
                      "heldout_assignments_checked",
                      "_independent_counts(",
                      "all eight controls required",
                      "_compare(cpu_gate, gpu_gate"):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
