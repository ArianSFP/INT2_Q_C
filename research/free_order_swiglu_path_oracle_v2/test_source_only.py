#!/usr/bin/env python3
"""Pure-standard-library tests for the source-only FOSP-v2 package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import platform
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ORACLE_PATH = ROOT / "free_order_oracle_v2.py"
CALIBRATION_PATH = ROOT / "calibrate_runtime.py"
BUILDER_PATH = ROOT / "create_authorization.py"
LOCK_PATH = ROOT / "protocol_lock.json"
BINDINGS_PATH = ROOT / "source_bindings.json"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_oracle() -> Any:
    spec = importlib.util.spec_from_file_location("fosp_v2_source_only_tests", ORACLE_PATH)
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

    def test_v1_noncontainment_is_repaired_by_direct_routing(self) -> None:
        stage = self.lock["scientific_gate"]
        self.assertEqual(stage["marginal_klt_stage"], "REMOVED_NONCONTAINING")
        self.assertEqual(stage["joint_dense_klt_stage"], "NOT_USED_AND_RECEIVES_NO_CREDIT")
        self.assertEqual(stage["stage_order"][0], "direct_qwen_full3x3_pair_panel")
        source = ORACLE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("_dense_stage", source)
        self.assertNotIn("eigvalsh", source)
        self.assertNotIn("reverse_waterfill", source)

    def test_v1_counterexample_reaches_and_beats_direct_threshold(self) -> None:
        # Audit construction: 768 neurons each carry (e_i,e_i+1,e_i+2).
        # A legal sequential path captures two units on all 767 edges.
        energy = 3 * 768
        capture = 2 * 767
        residual_ratio = (energy - capture) / energy
        s_value = -0.5 * math.log2(residual_ratio)
        self.assertEqual(residual_ratio, 770 / 2304)
        self.assertGreater(s_value, self.oracle.REQUIRED_GROSS_S)
        self.assertAlmostEqual(s_value, 0.7906051829300244, places=15)

    def test_factoradic_width_and_roundtrip(self) -> None:
        self.assertEqual(self.oracle.ceil_log2_factorial(768), 6260)
        self.assertEqual(self.oracle.FACTORADIC_BITS, 6264)
        for permutation in (
            (0,),
            (0, 1, 2, 3),
            (3, 2, 1, 0),
            (2, 0, 3, 1),
        ):
            rank = self.oracle.rank_permutation(permutation)
            self.assertEqual(self.oracle.unrank_permutation(len(permutation), rank), permutation)
        reverse = tuple(reversed(range(768)))
        encoded = self.oracle.serialize_permutation(reverse)
        self.assertEqual(len(encoded), 783)
        self.assertEqual(encoded[0] >> 4, 0)

    def test_physical_fp16_and_rate_ledgers(self) -> None:
        self.assertEqual(self.oracle.FP16_COEFFICIENT_BITS, 767 * 9 * 16)
        self.assertEqual(self.oracle.TOTAL_SIDE_BITS, 64 * 8 + 783 * 8 + 767 * 9 * 16)
        self.assertAlmostEqual(self.oracle.SIDE_BPW, 0.024843004014756944, places=18)
        self.assertAlmostEqual(self.oracle.REQUIRED_GROSS_S, 0.1858070514584381, places=15)
        lock_rows = {
            float(row["requested_rate_bpw"]): row for row in self.lock["rate_and_read"]["rows"]
        }
        for rate in self.oracle.RATES:
            observed = self.oracle.frame_ledger(rate)
            expected = lock_rows[rate]
            self.assertEqual(observed["frame_bytes"], expected["frame_bytes"])
            self.assertEqual(observed["actual_rate_bpw"], expected["actual_rate_bpw"])
            self.assertEqual(observed["residual_payload_bpw"], expected["residual_payload_bpw"])
            self.assertEqual(observed["cold_page_amplification"], expected["cold_page_amplification"])
            self.assertTrue(observed["strictly_below_2x"])
            self.assertEqual(observed["logical_byte_read_amplification"], 1.0)

    def test_cycle_cover_to_legal_path_fixture(self) -> None:
        # Pure-Python stand-ins for NumPy and scipy assignment isolate path logic.
        class TinyNP:
            float64 = float

            @staticmethod
            def asarray(value, dtype=None):
                return value

            @staticmethod
            def arange(n):
                return list(range(n))

            @staticmethod
            def array_equal(left, right):
                return list(left) == list(right)

        class Matrix:
            def __init__(self, rows):
                self.rows = rows
                self.shape = (len(rows), len(rows))

            def __getitem__(self, key):
                row, column = key
                return self.rows[row][column]

            def __neg__(self):
                return self

        scores = Matrix(
            [
                [-math.inf, 9.0, 1.0, 0.0],
                [8.0, -math.inf, 0.0, 1.0],
                [1.0, 0.0, -math.inf, 7.0],
                [0.0, 1.0, 6.0, -math.inf],
            ]
        )

        def assignment(_cost):
            return [0, 1, 2, 3], [1, 0, 3, 2]

        result = self.oracle._legal_path_from_cycle_cover(scores, TinyNP, assignment)
        self.assertEqual(result["cycle_count"], 2)
        path = result["path"]
        self.assertEqual(sorted(path), [0, 1, 2, 3])
        self.assertEqual(len(path) - 1, 3)

    def test_relaxed_targetwise_reuse_contains_every_legal_path_fixture(self) -> None:
        scores = [
            [-math.inf, 4.0, 2.0, 1.0],
            [3.0, -math.inf, 5.0, 2.0],
            [1.0, 6.0, -math.inf, 2.0],
            [7.0, 2.0, 3.0, -math.inf],
        ]
        relaxed = math.fsum(max(row) for row in scores)
        import itertools

        for path in itertools.permutations(range(4)):
            legal = math.fsum(scores[target][pred] for pred, target in zip(path[:-1], path[1:]))
            self.assertLessEqual(legal, relaxed)

    def test_bindings_are_exactly_frozen(self) -> None:
        self.assertEqual(sha256(BINDINGS_PATH), self.oracle.BINDINGS_SHA256)
        self.assertEqual(self.oracle.BINDINGS_SHA256, self.lock["execution_firewalls"]["source"]["source_bindings_sha256"])
        self.assertEqual([(row["layer"], row["expert"]) for row in self.bindings["experts"]], [(3, 57), (3, 121)])
        for expert in self.bindings["experts"]:
            self.assertEqual([role["role"] for role in expert["roles"]], ["gate", "up", "down"])

    def test_heavy_imports_are_deferred_and_runtime_cli_is_closed(self) -> None:
        tree = ast.parse(ORACLE_PATH.read_text(encoding="utf-8"), filename=ORACLE_PATH.name)
        top_imports: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.add(node.module.split(".")[0])
        self.assertFalse({"cupy", "numpy", "scipy", "torch"} & top_imports)
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for required in ("--workspace-root", "--output", "--authorization", "--authorization-sha256"):
            self.assertIn(required, literals)
        for forbidden in ("--source", "--manifest", "--panel", "--validation", "--target"):
            self.assertNotIn(forbidden, literals)

    def test_calibration_has_no_source_selector(self) -> None:
        tree = ast.parse(CALIBRATION_PATH.read_text(encoding="utf-8"), filename=CALIBRATION_PATH.name)
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("--output", literals)
        for forbidden in ("--workspace-root", "--source", "--manifest", "--panel", "--validation"):
            self.assertNotIn(forbidden, literals)

    def test_noncanonical_path_spellings_fail_closed(self) -> None:
        with self.assertRaises(self.oracle.ProtocolError):
            self.oracle._path_from_frozen_spelling("relative/path", "fixture")
        base = Path(tempfile.gettempdir()).resolve()
        noncanonical = os.fspath(base / "a" / ".." / "b")
        self.assertNotEqual(noncanonical, os.path.normpath(noncanonical))
        with self.assertRaises(self.oracle.ProtocolError):
            self.oracle._path_from_frozen_spelling(noncanonical, "fixture")

    def test_canonical_seal_rejects_forged_status(self) -> None:
        receipt = {"schema": "fixture", "status": "PASS", "value": 7}
        receipt["canonical_unsigned_sha256"] = self.oracle.canonical_sha256(receipt)
        self.assertEqual(
            self.oracle._verify_canonical_seal(receipt, "canonical_unsigned_sha256", "fixture"),
            receipt["canonical_unsigned_sha256"],
        )
        forged = dict(receipt)
        forged["status"] = "FORGED_PASS"
        with self.assertRaises(self.oracle.ProtocolError):
            self.oracle._verify_canonical_seal(forged, "canonical_unsigned_sha256", "fixture")

    def test_audit_manifest_must_bind_receipt_and_verifier(self) -> None:
        receipt_sha = "1" * 64
        valid = f"{receipt_sha}  audit_receipt.json\n{'2' * 64}  verify_audit.py\n".encode("ascii")
        self.oracle._audit_manifest_binds(valid, "audit_receipt.json", receipt_sha, "fixture")
        with self.assertRaises(self.oracle.ProtocolError):
            self.oracle._audit_manifest_binds(valid, "audit_receipt.json", "3" * 64, "fixture")

    def _make_authorization_fixture(self, base: Path) -> tuple[argparse.Namespace, dict[str, Any]]:
        oracle = self.oracle
        package = base / "package"
        source = base / "source"
        output_parent = base / "output"
        auth_parent = base / "authorization"
        source_audit_parent = base / "source_audit"
        runtime_parent = base / "runtime"
        runtime_audit_parent = base / "runtime_audit"
        for directory in (
            package,
            source,
            output_parent,
            auth_parent,
            source_audit_parent,
            runtime_parent,
            runtime_audit_parent,
        ):
            directory.mkdir()
        artifact_raw = b"synthetic-frozen-manifest\n"
        artifact_rows = {
            "source_only_receipt.json": "1" * 64,
            "free_order_oracle_v2.py": "2" * 64,
            "calibrate_runtime.py": "3" * 64,
        }
        expected_package = {
            "artifact_manifest_sha256": sha256_bytes(artifact_raw),
            "source_only_receipt_sha256": artifact_rows["source_only_receipt.json"],
            "runner_sha256": artifact_rows["free_order_oracle_v2.py"],
            "runtime_calibration_script_sha256": artifact_rows["calibrate_runtime.py"],
            "source_bindings_sha256": oracle.BINDINGS_SHA256,
        }
        source_audit: dict[str, Any] = {
            "schema": "free-order-swiglu-path-v2-independent-source-audit-receipt-v1",
            "status": "PASS_V2_INDEPENDENT_SOURCE_AUDIT",
            "artifact_set_status": "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET",
            "audited_package": expected_package,
            "v1_counterexample_replay": {"status": "PASS_COUNTEREXAMPLE_REACHES_DIRECT_STAGE"},
            "zero_access_ledger": {
                "qwen_or_model_payload_files_opened": 0,
                "qwen_or_model_payload_bytes_read": 0,
                "pinned_panel_files_opened": 0,
                "validation_files_opened": 0,
                "cupy_imports": 0,
                "cuda_api_calls": 0,
                "gpu_device_calls": 0,
                "external_data_fetches": 0,
            },
        }
        source_audit["canonical_unsigned_sha256"] = oracle.canonical_sha256(source_audit)
        runtime_backend = {
            "python_executable_resolved": os.fspath(Path(os.sys.executable).resolve()),
            "python_version": platform.python_version(),
            "numpy_version": "fixture-numpy",
            "cupy_version": "fixture-cupy",
            "scipy_version": "fixture-scipy",
            "device_name": "fixture-device",
            "cuda_runtime": 12090,
            "cuda_visible_devices": "0",
        }
        runtime: dict[str, Any] = {
            "schema": "free_order_swiglu_path_runtime_calibration_v2",
            "status": "PASS_SOURCE_FREE_FULL_GEOMETRY_RUNTIME_CALIBRATION",
            "artifact_binding": {
                "artifact_manifest_sha256": sha256_bytes(artifact_raw),
                "runner_sha256": artifact_rows["free_order_oracle_v2.py"],
                "calibration_script_sha256": artifact_rows["calibrate_runtime.py"],
            },
            "backend": runtime_backend,
            "zero_access_ledger": {
                "workspace_or_source_arguments_supported": 0,
                "source_bindings_loaded": 0,
                "qwen_or_model_payload_files_opened": 0,
                "qwen_or_model_payload_bytes_read": 0,
                "pinned_panel_files_opened": 0,
                "validation_files_opened": 0,
                "external_data_fetches": 0,
                "production_result_files_opened": 0,
                "production_gpu_jobs": 0,
                "synthetic_gpu_jobs": 1,
            },
        }
        runtime["canonical_unsigned_sha256"] = oracle.canonical_sha256(runtime)
        runtime_path = runtime_parent / "runtime_receipt.json"
        runtime_path.write_text(json.dumps(runtime, sort_keys=True), encoding="utf-8")
        runtime_sha = sha256(runtime_path)
        runtime_internal = runtime["canonical_unsigned_sha256"]
        runtime_audit: dict[str, Any] = {
            "schema": "free-order-swiglu-path-v2-independent-runtime-audit-receipt-v1",
            "status": "PASS_V2_INDEPENDENT_RUNTIME_AUDIT",
            "artifact_set_status": "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET",
            "audited_package": expected_package,
            "audited_runtime_receipt": {
                "file_sha256": runtime_sha,
                "internal_sha256": runtime_internal,
            },
            "zero_access_ledger": {
                "qwen_or_model_payload_files_opened": 0,
                "qwen_or_model_payload_bytes_read": 0,
                "pinned_panel_files_opened": 0,
                "validation_files_opened": 0,
                "production_result_files_opened": 0,
                "production_gpu_jobs": 0,
                "cupy_imports": 0,
                "cuda_api_calls": 0,
                "gpu_device_calls": 0,
            },
        }
        runtime_audit["canonical_unsigned_sha256"] = oracle.canonical_sha256(runtime_audit)
        source_receipt_path = source_audit_parent / "audit_receipt.json"
        runtime_audit_path = runtime_audit_parent / "audit_receipt.json"
        source_receipt_path.write_text(json.dumps(source_audit, sort_keys=True), encoding="utf-8")
        runtime_audit_path.write_text(json.dumps(runtime_audit, sort_keys=True), encoding="utf-8")
        source_manifest_path = source_audit_parent / "AUDIT_SHA256SUMS.txt"
        runtime_manifest_path = runtime_audit_parent / "AUDIT_SHA256SUMS.txt"
        source_manifest_path.write_text(
            f"{sha256(source_receipt_path)}  audit_receipt.json\n{'4' * 64}  verify_audit.py\n",
            encoding="ascii",
        )
        runtime_manifest_path.write_text(
            f"{sha256(runtime_audit_path)}  audit_receipt.json\n{'5' * 64}  verify_audit.py\n",
            encoding="ascii",
        )
        audit_binding = {
            "source_audit_status": "PASS_V2_INDEPENDENT_SOURCE_AUDIT",
            "source_audit_manifest_sha256": sha256(source_manifest_path),
            "source_audit_receipt_sha256": sha256(source_receipt_path),
            "source_audit_receipt_internal_sha256": source_audit["canonical_unsigned_sha256"],
            "runtime_receipt_sha256": runtime_sha,
            "runtime_receipt_internal_sha256": runtime_internal,
            "runtime_audit_status": "PASS_V2_INDEPENDENT_RUNTIME_AUDIT",
            "runtime_audit_manifest_sha256": sha256(runtime_manifest_path),
            "runtime_audit_receipt_sha256": sha256(runtime_audit_path),
            "runtime_audit_receipt_internal_sha256": runtime_audit["canonical_unsigned_sha256"],
        }
        artifact_binding = {
            "artifact_manifest_sha256": sha256_bytes(artifact_raw),
            "source_only_receipt_sha256": artifact_rows["source_only_receipt.json"],
            "runner_sha256": artifact_rows["free_order_oracle_v2.py"],
            "source_bindings_sha256": oracle.BINDINGS_SHA256,
            "runtime_calibration_script_sha256": artifact_rows["calibrate_runtime.py"],
        }
        output = output_parent / "result.json"
        run_material = {
            "artifact_binding": artifact_binding,
            "path_binding": {
                "workspace_root": os.fspath(source.resolve()),
                "output": os.fspath(output.resolve()),
                "authorization_parent": os.fspath(auth_parent.resolve()),
            },
            "audit_paths": {
                "source_audit_manifest": os.fspath(source_manifest_path.resolve()),
                "source_audit_receipt": os.fspath(source_receipt_path.resolve()),
                "runtime_receipt": os.fspath(runtime_path.resolve()),
                "runtime_audit_manifest": os.fspath(runtime_manifest_path.resolve()),
                "runtime_audit_receipt": os.fspath(runtime_audit_path.resolve()),
            },
            "audit_binding": audit_binding,
            "runtime_binding": runtime_backend,
        }
        authorization: dict[str, Any] = {
            "schema": oracle.AUTHORIZATION_SCHEMA,
            "status": "AUTHORIZED_ONE_SHOT_AUXILIARY_SOURCE_RUN",
            "one_shot": True,
            "pinned_panel_authorized": False,
            "scope_literal": "FOSP_V2_AUXILIARY_DISCOVERY_ONLY_NO_PINNED_PANEL",
            "run_id": oracle.canonical_sha256(run_material),
            **run_material,
        }
        authorization["canonical_unsigned_sha256"] = oracle.canonical_sha256(authorization)
        authorization_path = auth_parent / "authorization.json"
        authorization_path.write_text(json.dumps(authorization, sort_keys=True), encoding="utf-8")
        args = argparse.Namespace(
            authorization=os.fspath(authorization_path.resolve()),
            authorization_sha256=sha256(authorization_path),
            workspace_root=os.fspath(source.resolve()),
            output=os.fspath(output.resolve()),
        )
        return args, {"package": package.resolve(), "artifact_rows": artifact_rows, "artifact_raw": artifact_raw}

    def test_valid_external_evidence_is_opened_and_bound(self) -> None:
        if os.name == "nt":
            self.skipTest("production held-directory/openat contract is Linux-only")
        with tempfile.TemporaryDirectory() as directory:
            args, fixture = self._make_authorization_fixture(Path(directory))
            result = self.oracle._load_authorization(
                args, fixture["package"], fixture["artifact_rows"], fixture["artifact_raw"]
            )
            self.assertEqual(result[0]["status"], "AUTHORIZED_ONE_SHOT_AUXILIARY_SOURCE_RUN")
            os.close(result[3])
            os.close(result[5])

    def test_forged_runtime_audit_status_fails_even_with_resealed_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args, fixture = self._make_authorization_fixture(Path(directory))
            authorization_path = Path(args.authorization)
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
            runtime_audit_path = Path(authorization["audit_paths"]["runtime_audit_receipt"])
            runtime_audit = json.loads(runtime_audit_path.read_text(encoding="utf-8"))
            runtime_audit["status"] = "FORGED_PASS"
            runtime_audit.pop("canonical_unsigned_sha256")
            runtime_audit["canonical_unsigned_sha256"] = self.oracle.canonical_sha256(runtime_audit)
            runtime_audit_path.write_text(json.dumps(runtime_audit, sort_keys=True), encoding="utf-8")
            runtime_manifest = Path(authorization["audit_paths"]["runtime_audit_manifest"])
            runtime_manifest.write_text(
                f"{sha256(runtime_audit_path)}  audit_receipt.json\n{'5' * 64}  verify_audit.py\n",
                encoding="ascii",
            )
            authorization["audit_binding"]["runtime_audit_manifest_sha256"] = sha256(runtime_manifest)
            authorization["audit_binding"]["runtime_audit_receipt_sha256"] = sha256(runtime_audit_path)
            authorization["audit_binding"]["runtime_audit_receipt_internal_sha256"] = runtime_audit["canonical_unsigned_sha256"]
            authorization.pop("canonical_unsigned_sha256")
            authorization["canonical_unsigned_sha256"] = self.oracle.canonical_sha256(authorization)
            authorization_path.write_text(json.dumps(authorization, sort_keys=True), encoding="utf-8")
            args.authorization_sha256 = sha256(authorization_path)
            with self.assertRaisesRegex(self.oracle.ProtocolError, "runtime audit is not PASS"):
                self.oracle._load_authorization(
                    args, fixture["package"], fixture["artifact_rows"], fixture["artifact_raw"]
                )

    def test_symlinked_evidence_component_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args, fixture = self._make_authorization_fixture(Path(directory))
            authorization_path = Path(args.authorization)
            authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
            real_parent = Path(authorization["audit_paths"]["runtime_receipt"]).parent
            link_parent = Path(directory) / "runtime_link"
            try:
                link_parent.symlink_to(real_parent, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlink creation unavailable")
            authorization["audit_paths"]["runtime_receipt"] = os.fspath(link_parent / "runtime_receipt.json")
            authorization.pop("canonical_unsigned_sha256")
            authorization["canonical_unsigned_sha256"] = self.oracle.canonical_sha256(authorization)
            authorization_path.write_text(json.dumps(authorization, sort_keys=True), encoding="utf-8")
            args.authorization_sha256 = sha256(authorization_path)
            with self.assertRaisesRegex(self.oracle.ProtocolError, "symlinked component"):
                self.oracle._load_authorization(
                    args, fixture["package"], fixture["artifact_rows"], fixture["artifact_raw"]
                )

    def test_held_output_parent_detects_replacement(self) -> None:
        if os.name == "nt":
            self.skipTest("Windows does not expose the required dir_fd openat semantics")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            parent = base / "output"
            parent.mkdir()
            descriptor = self.oracle._directory_descriptor(parent.resolve(), "fixture")
            moved = base / "moved"
            parent.rename(moved)
            parent.mkdir()
            try:
                with self.assertRaisesRegex(self.oracle.ProtocolError, "identity changed"):
                    self.oracle._write_create_new_at(descriptor, parent, "result.json", {"ok": True})
            finally:
                os.close(descriptor)
            self.assertFalse((parent / "result.json").exists())
            self.assertFalse((moved / "result.json").exists())

    def test_create_new_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            self.oracle._write_create_new(output, {"ok": True})
            with self.assertRaises(FileExistsError):
                self.oracle._write_create_new(output, {"ok": False})


if __name__ == "__main__":
    unittest.main(verbosity=2)
