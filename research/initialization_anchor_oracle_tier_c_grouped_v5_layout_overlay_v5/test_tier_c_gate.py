"""CPU-only tests for Tier-C grouped-v5 layout-overlay orchestration."""

from __future__ import annotations

import json
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import common
import kernels
import tier_c_gate


class TierCGroupedV5LayoutOverlayTests(unittest.TestCase):
    def _valid_stage0_shard_state(self, shard=0):
        seed_start = shard * common.SEED_SHARD_SIZE
        candidates = common.representative_ordinals(
            seed_start, seed_start + common.SEED_SHARD_SIZE
        )
        ordinals = np.tile(candidates[: common.STAGE0_TOP_K], (len(common.DOMAIN_IDS), 1))
        metrics = np.tile(
            np.arange(common.STAGE0_TOP_K, dtype=np.float64),
            (len(common.DOMAIN_IDS), 1),
        )
        return {
            "seed_start": np.asarray([seed_start], dtype=np.int32),
            "seed_stop": np.asarray(
                [seed_start + common.SEED_SHARD_SIZE], dtype=np.int32
            ),
            "top_ordinals": ordinals,
            "top_q": metrics,
        }

    def _authorization_fixture(self, root: Path):
        lock = json.loads(json.dumps(common.load_candidate_lock()))
        authorization_path = (root / "authorization_v5.json").resolve()
        lock["authorization_protocol"]["receipt_path"] = str(authorization_path)
        calibration_path = (root / "calibration.json").resolve()
        source_trace_path = (root / "source_trace.json").resolve()
        calibration_path.write_text("calibration fixture\n", encoding="utf-8")
        source_trace_path.write_text("source trace fixture\n", encoding="utf-8")
        calibration = {"receipt_sha256": "1" * 64}
        package_manifest_sha = common.sha256_file(common.PACKAGE_DIR / "ARTIFACT_SHA256SUMS.txt")
        target_binding = tier_c_gate._audit_target_binding(package_manifest_sha)

        def write_audit_package(directory: Path, value):
            directory.mkdir()
            receipt = directory / "audit_receipt.json"
            evidence = directory / "evidence.txt"
            manifest = directory / "ARTIFACT_SHA256SUMS.txt"
            normalized = dict(value)
            value["audit_receipt_sha256"] = common.sha256_bytes(
                common.canonical_json_bytes(normalized)
            )
            receipt.write_text(json.dumps(value), encoding="utf-8")
            evidence.write_text("source-only audit fixture\n", encoding="ascii")
            rows = {
                receipt.name: common.sha256_file(receipt),
                evidence.name: common.sha256_file(evidence),
            }
            manifest.write_text(
                "".join(f"{rows[name]}  {name}\n" for name in sorted(rows)),
                encoding="ascii",
            )
            return manifest.resolve(), receipt.resolve(), {
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": common.sha256_file(manifest),
                "receipt_path": str(receipt.resolve()),
                "receipt_sha256": common.sha256_file(receipt),
                "receipt_internal_sha256": value["audit_receipt_sha256"],
            }

        source_value = {
            "schema": lock["authorization_protocol"]["source_audit_receipt_schema"],
            "status": lock["authorization_protocol"]["source_audit_required_status"],
            "audited_target": target_binding,
            "verification": dict(tier_c_gate._AUDIT_VERIFICATION),
            "access_attestation": dict(tier_c_gate._AUDIT_ACCESS),
            "authorization": dict(tier_c_gate._SOURCE_AUDIT_AUTHORIZATION),
        }
        source_manifest, source_receipt, source_binding = write_audit_package(
            root / "source_audit", source_value
        )

        calibration_value = {
            "schema": lock["authorization_protocol"]["calibration_audit_receipt_schema"],
            "status": lock["authorization_protocol"]["calibration_audit_required_status"],
            "audited_target": target_binding,
            "bindings": {
                "calibration_receipt_sha256": common.sha256_file(calibration_path),
                "calibration_receipt_internal_sha256": calibration["receipt_sha256"],
                "source_audit_manifest_sha256": source_binding["manifest_sha256"],
                "source_audit_receipt_sha256": source_binding["receipt_sha256"],
            },
            "verification": dict(tier_c_gate._AUDIT_VERIFICATION),
            "access_attestation": dict(tier_c_gate._AUDIT_ACCESS),
            "authorization": dict(tier_c_gate._CALIBRATION_AUDIT_AUTHORIZATION),
        }
        calibration_manifest, calibration_receipt, calibration_binding = write_audit_package(
            root / "calibration_audit", calibration_value
        )

        workspace = (root / "workspace").resolve()
        aux = (root / "aux").resolve()
        output = (root / "output").resolve()
        v4_run = (root / "v4_run").resolve()
        v4_audit = (root / "v4_audit.json").resolve()
        workspace.mkdir()
        aux.mkdir()
        v4_run.mkdir()
        v4_audit.write_text("grouped-v4 audit fixture\n", encoding="utf-8")
        # The authorization receipt is a runtime input, but its stable boundary
        # authorization deliberately omits its own descriptor to avoid a
        # self-referential byte/size hash.  The full runtime guard still binds it.
        authorization_path.write_text("{}\n", encoding="utf-8")
        value = {
            "schema": lock["authorization_protocol"]["receipt_schema"],
            "status": lock["authorization_protocol"]["receipt_status"],
            "authorization_nonce": "4" * 64,
            "created_unix_ns": 1,
            "action": "QWEN_AUX_33_DOMAIN_SINGLE_RUN",
            "one_shot": True,
            "package": {
                "artifact_manifest_sha256": package_manifest_sha,
                "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
                "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
                "runner_sha256": common.sha256_file(Path(tier_c_gate.__file__).resolve()),
                "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
                "kernels_sha256": common.sha256_file(common.PACKAGE_DIR / "kernels.py"),
                "overlay_sha256": common.sha256_file(common.PACKAGE_DIR / "overlay.py"),
                "parity_sha256": common.sha256_file(common.PACKAGE_DIR / "parity.py"),
            },
            "source_audit": source_binding,
            "calibration": {
                "receipt_path": str(calibration_path),
                "receipt_sha256": common.sha256_file(calibration_path),
                "receipt_internal_sha256": calibration["receipt_sha256"],
                "source_trace_path": str(source_trace_path),
                "source_trace_sha256": common.sha256_file(source_trace_path),
            },
            "calibration_audit": calibration_binding,
            "run": {
                "workspace_root": str(workspace),
                "aux_dir": str(aux),
                "output_dir": str(output),
                "v4_run_root": str(v4_run),
                "v4_result_audit": str(v4_audit),
                "v4_topk_authentication_receipt_sha256": lock["v4_reuse"][
                    "expected_authentication_receipt_sha256"
                ],
                "resume_same_output_permitted": True,
            },
            "path_boundary": common.BoundaryGuard(
                "QWEN_AUX_33_DOMAIN_SINGLE_RUN",
                outputs=(("production output/run root", output, "directory", False),),
                inputs=(
                    ("workspace input root", workspace, "directory"),
                    ("auxiliary input root", aux, "directory"),
                    ("calibration receipt input", calibration_path, "file"),
                    ("source-trace receipt input", source_trace_path, "file"),
                    ("grouped-v4 run input root", v4_run, "directory"),
                    ("grouped-v4 result audit input", v4_audit, "file"),
                    ("production authorization input", authorization_path, "file"),
                    ("source-audit input root", source_manifest.parent, "directory"),
                    ("source-audit manifest input", source_manifest, "file"),
                    ("source-audit receipt input", source_receipt, "file"),
                    ("calibration-audit input root", calibration_manifest.parent, "directory"),
                    ("calibration-audit manifest input", calibration_manifest, "file"),
                    ("calibration-audit receipt input", calibration_receipt, "file"),
                ),
            ).authorization_receipt(),
            "gpu_runtime_authorized": True,
            "qwen_manifest_or_payload_access_authorized": True,
        }
        value["authorization_receipt_sha256"] = common.sha256_bytes(
            common.canonical_json_bytes(value)
        )
        authorization_path.write_text(json.dumps(value), encoding="utf-8")
        return {
            "lock": lock,
            "path": authorization_path,
            "value": value,
            "workspace": workspace,
            "aux": aux,
            "output": output,
            "calibration_path": calibration_path,
            "source_trace_path": source_trace_path,
            "v4_run": v4_run,
            "v4_audit": v4_audit,
            "calibration": calibration,
            "source_receipt": source_receipt,
            "source_manifest": source_manifest,
            "source_binding": source_binding,
            "source_value": source_value,
            "calibration_audit_receipt": calibration_receipt,
            "calibration_audit_manifest": calibration_manifest,
            "calibration_binding": calibration_binding,
            "calibration_value": calibration_value,
        }

    def _load_authorization_fixture(self, fixture):
        return tier_c_gate._load_production_authorization(
            fixture["path"], fixture["lock"],
            workspace_root=fixture["workspace"], aux_dir=fixture["aux"],
            output_dir=fixture["output"], calibration_path=fixture["calibration_path"],
            source_trace_path=fixture["source_trace_path"], v4_run_root=fixture["v4_run"],
            v4_result_audit_path=fixture["v4_audit"], calibration=fixture["calibration"],
            resume=False,
        )

    def _refresh_authorization_fixture(
        self, fixture, *, source=False, calibration=False, recompute_internal=True
    ):
        def refresh(value, receipt, manifest, binding, do_internal):
            if do_internal:
                normalized = dict(value)
                normalized.pop("audit_receipt_sha256", None)
                value["audit_receipt_sha256"] = common.sha256_bytes(
                    common.canonical_json_bytes(normalized)
                )
            receipt.write_text(json.dumps(value), encoding="utf-8")
            members = sorted(
                path for path in receipt.parent.iterdir()
                if path.name != "ARTIFACT_SHA256SUMS.txt"
            )
            manifest.write_text(
                "".join(
                    f"{common.sha256_file(path)}  {path.name}\n" for path in members
                ),
                encoding="ascii",
            )
            binding.update({
                "manifest_sha256": common.sha256_file(manifest),
                "receipt_sha256": common.sha256_file(receipt),
                "receipt_internal_sha256": value["audit_receipt_sha256"],
            })

        if source:
            refresh(
                fixture["source_value"], fixture["source_receipt"],
                fixture["source_manifest"], fixture["source_binding"],
                recompute_internal,
            )
            fixture["calibration_value"]["bindings"].update({
                "source_audit_manifest_sha256": fixture["source_binding"]["manifest_sha256"],
                "source_audit_receipt_sha256": fixture["source_binding"]["receipt_sha256"],
            })
            calibration = True
        if calibration:
            refresh(
                fixture["calibration_value"], fixture["calibration_audit_receipt"],
                fixture["calibration_audit_manifest"], fixture["calibration_binding"],
                recompute_internal if not source else True,
            )
        outer = fixture["value"]
        outer["path_boundary"] = common.BoundaryGuard(
            "QWEN_AUX_33_DOMAIN_SINGLE_RUN",
            outputs=(("production output/run root", fixture["output"], "directory", False),),
            inputs=(
                ("workspace input root", fixture["workspace"], "directory"),
                ("auxiliary input root", fixture["aux"], "directory"),
                ("calibration receipt input", fixture["calibration_path"], "file"),
                ("source-trace receipt input", fixture["source_trace_path"], "file"),
                ("grouped-v4 run input root", fixture["v4_run"], "directory"),
                ("grouped-v4 result audit input", fixture["v4_audit"], "file"),
                ("production authorization input", fixture["path"], "file"),
                ("source-audit input root", fixture["source_manifest"].parent, "directory"),
                ("source-audit manifest input", fixture["source_manifest"], "file"),
                ("source-audit receipt input", fixture["source_receipt"], "file"),
                ("calibration-audit input root", fixture["calibration_audit_manifest"].parent, "directory"),
                ("calibration-audit manifest input", fixture["calibration_audit_manifest"], "file"),
                ("calibration-audit receipt input", fixture["calibration_audit_receipt"], "file"),
            ),
        ).authorization_receipt()
        normalized = dict(outer)
        normalized.pop("authorization_receipt_sha256", None)
        outer["authorization_receipt_sha256"] = common.sha256_bytes(
            common.canonical_json_bytes(normalized)
        )
        fixture["path"].write_text(json.dumps(outer), encoding="utf-8")

    def test_import_is_source_only_and_tier_b_is_unmodified_dependency(self):
        self.assertIs(tier_c_gate.BASE.common, common)
        self.assertIs(tier_c_gate.BASE.kernels, kernels)
        self.assertIs(tier_c_gate.BASE._affine_sse_from_moments,
                      tier_c_gate._affine_sse_from_moments_f16)
        self.assertEqual(common.sha256_file(tier_c_gate.TIER_B_GATE_PATH),
                         tier_c_gate.EXPECTED_TIER_B_GATE_SHA256)
        self.assertFalse(common.environment_has_cuda_imports())
        for forbidden in ("torch", "cupy", "transformer_engine", "megatron"):
            self.assertNotIn(forbidden, sys.modules)

    def test_calibration_fixture_exactly_matches_production_roles(self):
        first = tier_c_gate._synthetic_calibration_coordinates()
        second = tier_c_gate._synthetic_calibration_coordinates()
        for left, right in zip(first[:3], second[:3]):
            self.assertTrue(np.array_equal(left, right))
        self.assertEqual(first[3], second[3])
        experts, roles, coordinates, layout = first
        self.assertEqual((experts.shape, roles.shape, coordinates.shape), ((512,), (512,), (512,)))
        self.assertEqual(len(layout["matrix_rows"]), 23)
        self.assertEqual(layout["role_counts"], {
            "up": {"fit": 122, "score": 122},
            "down": {"fit": 134, "score": 134},
        })
        self.assertEqual(sum(roles == 0), 244)
        self.assertEqual(sum(roles == 1), 268)
        self.assertEqual([(row["expert"], row["role"]) for row in layout["matrix_rows"]],
                         list(common.CALIBRATION_SELECTION_IDENTITIES))
        self.assertTrue(np.all(coordinates < common.WEIGHTS_PER_MATRIX))
        self.assertEqual(tier_c_gate._synthetic_coordinate_sha256(*first),
                         tier_c_gate._synthetic_coordinate_sha256(*second))

    def test_search_scorer_uses_fp16_decoded_coefficients(self):
        g_fit = np.asarray([[0.1, 0.4, -0.2], [0.2, 0.5, -0.1]], dtype=np.float64)
        g_score = np.asarray([[0.3, -0.7], [0.1, 0.8]], dtype=np.float64)
        w_fit = np.asarray([[0.9, -0.2, 0.4]], dtype=np.float64)
        w_score = np.asarray([[0.3, -0.5]], dtype=np.float64)
        observed, baseline = tier_c_gate._affine_sse_from_moments_f16(
            np, g_fit, g_score, w_fit, w_score
        )
        expected = []
        for anchor_fit, anchor_score in zip(g_fit, g_score):
            fit = common.fit_affine_moments(w_fit[0], anchor_fit)
            residual = w_score[0] - (fit["mu"] + fit["alpha"] * anchor_score)
            expected.append(float(np.dot(residual, residual)))
        self.assertTrue(np.allclose(observed[:, 0], expected, rtol=1e-12, atol=1e-12))
        self.assertTrue(np.all(np.asarray(baseline) > 0.0))

    def test_synthetic_domains_fixed_finite_and_33_wide(self):
        fit, score = tier_c_gate._synthetic_domains()
        self.assertEqual(fit.shape, (33, 256))
        self.assertEqual(score.shape, (33, 256))
        self.assertEqual((fit.dtype, score.dtype), (np.float32, np.float32))
        self.assertTrue(np.all(np.isfinite(fit)))
        self.assertFalse(np.array_equal(fit, score))

    def test_exact_cascade_accounting(self):
        shard0 = len(common.representative_ordinals(0, 256))
        shard1 = len(common.representative_ordinals(256, 512))
        stage0 = common.EFFECTIVE_CANDIDATES * 512
        union_max = len(common.DOMAIN_IDS) * common.STAGE0_TOP_K
        stage1 = union_max * 48_624
        reporting = len(common.DOMAIN_IDS) * 65_536
        self.assertEqual((shard0, shard1), (164_864, 164_864))
        self.assertEqual(stage0, 21_609_054_208)
        union_max *= 2
        stage1 *= 2
        self.assertEqual(union_max, 135_168)
        self.assertEqual(stage1, 6_572_408_832)
        self.assertEqual(stage0 + stage1, 28_181_463_040)
        self.assertEqual(reporting, 2_162_688)
        self.assertEqual(stage0 + stage1 + reporting, 28_183_625_728)
        self.assertEqual(len(common.NULL_DOMAIN_IDS), 32)

    def test_cpu_preflight_never_reaches_manifest_or_cuda(self):
        with mock.patch.object(common, "load_candidate_lock", return_value={}), mock.patch.object(
            common, "load_source_rows", side_effect=AssertionError("source manifest reached")
        ), mock.patch.object(
            kernels, "PhiloxRandomAccess", side_effect=AssertionError("CUDA reached")
        ):
            result = tier_c_gate.cpu_preflight()
        self.assertEqual(result["status"], "PASS_NO_QWEN_ACCESS_CUDA_NOT_IMPORTED_OR_TOUCHED")
        self.assertEqual(result["stage0_max_generated_values"], 21_609_054_208)
        self.assertEqual(result["stage1_max_generated_values"], 6_572_408_832)
        self.assertEqual(result["post_selection_reporting_generated_values"], 2_162_688)
        self.assertEqual(result["end_to_end_max_generated_values"], 28_183_625_728)
        self.assertEqual(result["calibration_role_counts"]["up"], {"fit": 122, "score": 122})
        self.assertFalse(result["cuda_modules_imported"])

    def test_production_orders_content_auth_and_parity_before_manifest(self):
        calls: list[str] = []
        seen_log = []

        class FakeAccess:
            pass

        class FakeV4:
            receipt = {"receipt_sha256": "a" * 64}

        class FakeBoundary:
            def revalidate(self, _phase):
                calls.append("boundary")

            def receipt(self):
                return {"schema": "fixture-boundary"}

        lock = {"execution": {
            "production_authorization_receipt": "authorization.json",
            "state_directory": "state",
            "output_json": "result.json",
        }}
        authorization = {
            "file_sha256": "b" * 64,
            "source_audit": {"manifest_sha256": "c" * 64},
            "calibration_audit": {"manifest_sha256": "d" * 64},
            "run": {"v4_topk_authentication_receipt_sha256": "a" * 64},
            "boundary_guard": FakeBoundary(),
        }
        def stop_at_manifest(_root, log):
            calls.append("manifest")
            seen_log.extend(log)
            raise RuntimeError("stop")

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {tier_c_gate.parity.SINGLE_PARAM_ENV: "1"}
        ), mock.patch.object(
            common, "load_candidate_lock", side_effect=lambda: calls.append("lock") or lock
        ), mock.patch.object(
            tier_c_gate, "_load_calibration", side_effect=lambda *_: calls.append("calibration") or {}
        ), mock.patch.object(
            tier_c_gate, "_load_production_authorization",
            side_effect=lambda *_args, **_kwargs: calls.append("authorization") or authorization
        ), mock.patch.object(
            kernels, "PhiloxRandomAccess", side_effect=lambda *_: calls.append("cuda") or FakeAccess()
        ), mock.patch.object(
            tier_c_gate.parity, "run_parity",
            side_effect=lambda *_: calls.append("parity") or {"all_required_checks_passed": True}
        ), mock.patch.object(
            tier_c_gate.overlay, "authenticate_v4_topk",
            side_effect=lambda *_: calls.append("v4") or FakeV4()
        ), mock.patch.object(common, "load_source_rows", side_effect=stop_at_manifest):
            calibration_path = Path(directory) / "calibration.json"
            source_trace_path = Path(directory) / "trace.json"
            v4_run_root = Path(directory) / "v4"
            v4_audit_path = Path(directory) / "audit.json"
            workspace_root = Path(directory) / "workspace"
            aux_root = Path(directory) / "aux"
            calibration_path.write_text("fixture", encoding="utf-8")
            source_trace_path.write_text("fixture", encoding="utf-8")
            v4_run_root.mkdir()
            workspace_root.mkdir()
            aux_root.mkdir()
            v4_audit_path.write_text("fixture", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "stop"):
                tier_c_gate.run_gate(workspace_root, aux_root, Path(directory) / "output",
                                     calibration_path, source_trace_path, v4_run_root,
                                     v4_audit_path, resume=False)
        self.assertEqual(calls, [
            "lock", "calibration", "authorization", "boundary", "cuda", "parity", "v4", "manifest"
        ])
        self.assertEqual(
            seen_log[0]["action"],
            "content_bound_v5_production_authorization_authenticated_before_cuda_or_qwen_access",
        )
        self.assertFalse(seen_log[0]["manifest_lstat_or_stat_performed"])
        self.assertEqual(
            seen_log[2]["action"],
            "runtime_parity_passed_before_any_manifest_directory_or_payload_operation",
        )

    def test_complete_workspace_aux_closure_precedes_any_output_or_journal_create(self):
        calls: list[str] = []

        class FakeBoundary:
            def revalidate(self, phase):
                calls.append(f"boundary:{phase}")

            def receipt(self):
                return {"schema": "fixture-boundary"}

        class FakeV4:
            receipt = {"receipt_sha256": "a" * 64}

        lock = {
            "execution": {
                "production_authorization_receipt": "/tmp/fixture-authorization-v5.json",
                "state_directory": "state_v5",
                "output_json": "result_v5.json",
            },
            "coordinate_protocol": {
                "stage0_coordinate_plan_sha256": "stage0",
                "full_coordinate_plan_sha256": "full",
            },
        }
        authorization = {
            "file_sha256": "b" * 64,
            "source_audit": {"manifest_sha256": "c" * 64},
            "calibration_audit": {"manifest_sha256": "d" * 64},
            "run": {"v4_topk_authentication_receipt_sha256": "a" * 64},
            "boundary_guard": FakeBoundary(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            workspace, aux, v4_run = root / "workspace", root / "aux", root / "v4"
            for path in (workspace, aux, v4_run):
                path.mkdir()
            payload = aux / "eligible.bf16.bin"
            payload.write_bytes(b"fixture")
            calibration, trace, v4_audit = (
                root / "calibration.json", root / "trace.json", root / "v4-audit.json"
            )
            for path in (calibration, trace, v4_audit):
                path.write_text("fixture\n", encoding="utf-8")
            output = root / "must-remain-absent"

            def plan_sha(value):
                return value

            with mock.patch.dict(
                os.environ, {tier_c_gate.parity.SINGLE_PARAM_ENV: "1"}
            ), mock.patch.object(common, "load_candidate_lock", return_value=lock), mock.patch.object(
                tier_c_gate, "_load_calibration", return_value={}
            ), mock.patch.object(
                tier_c_gate, "_load_production_authorization", return_value=authorization
            ), mock.patch.object(
                kernels, "PhiloxRandomAccess", return_value=object()
            ), mock.patch.object(
                tier_c_gate.parity, "run_parity", return_value={"all_required_checks_passed": True}
            ), mock.patch.object(
                tier_c_gate.overlay, "authenticate_v4_topk", return_value=FakeV4()
            ), mock.patch.object(
                common, "load_source_rows", return_value=(object(),)
            ), mock.patch.object(
                common, "validate_aux_directory",
                return_value=({"fixture": payload}, {"frozen": True}),
            ), mock.patch.object(
                common, "make_plan", side_effect=lambda _rows, *, stage0: "stage0" if stage0 else "full"
            ), mock.patch.object(
                common, "plan_sha256", side_effect=plan_sha
            ), mock.patch.object(
                common, "revalidate_workspace_aux_closure",
                side_effect=common.ProtocolError("closure rejected before output"),
            ), mock.patch.object(
                common, "ensure_output_directory",
                side_effect=AssertionError("output/journal creation reached"),
            ):
                with self.assertRaisesRegex(common.ProtocolError, "closure rejected"):
                    tier_c_gate.run_gate(
                        workspace, aux, output, calibration, trace, v4_run, v4_audit,
                        resume=False,
                    )
            self.assertFalse(output.exists())

    def test_content_bound_v4_authorization_accepts_exact_external_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            observed = self._load_authorization_fixture(fixture)
        self.assertEqual(
            observed["internal_sha256"],
            fixture["value"]["authorization_receipt_sha256"],
        )
        self.assertEqual(
            observed["run"]["v4_topk_authentication_receipt_sha256"],
            fixture["lock"]["v4_reuse"]["expected_authentication_receipt_sha256"],
        )
        self.assertTrue(observed["source_audit"]["manifest_closure_verified"])
        self.assertTrue(
            observed["source_audit"][
                "manifest_closure_reauthenticated_after_receipt_semantics"
            ]
        )
        self.assertTrue(observed["calibration_audit"]["receipt_internal_sha256_recomputed"])
        self.assertEqual(observed["source_audit"]["manifest_member_count"], 2)

    def test_audit_receipt_internal_seal_is_recomputed_not_trusted_literal(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            fixture["calibration_value"]["audit_receipt_sha256"] = "3" * 64
            self._refresh_authorization_fixture(
                fixture, calibration=True, recompute_internal=False
            )
            with self.assertRaisesRegex(common.ProtocolError, "internal hash mismatch"):
                self._load_authorization_fixture(fixture)

    def test_audit_receipts_reject_extra_keys_and_broader_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            fixture["source_value"]["unreviewed_extension"] = True
            self._refresh_authorization_fixture(fixture, source=True)
            with self.assertRaisesRegex(common.ProtocolError, "keys mismatch"):
                self._load_authorization_fixture(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            fixture["source_value"]["authorization"][
                "qwen_payload_or_manifest_launch_authorized"
            ] = True
            self._refresh_authorization_fixture(fixture, source=True)
            with self.assertRaisesRegex(common.ProtocolError, "authorization semantics"):
                self._load_authorization_fixture(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            fixture["calibration_value"]["authorization"]["retry_authorized"] = True
            self._refresh_authorization_fixture(fixture, calibration=True)
            with self.assertRaisesRegex(common.ProtocolError, "keys mismatch"):
                self._load_authorization_fixture(fixture)

    def test_audit_target_and_cross_audit_bindings_are_semantically_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            fixture["source_value"]["audited_target"]["runner_sha256"] = "0" * 64
            self._refresh_authorization_fixture(fixture, source=True)
            with self.assertRaisesRegex(common.ProtocolError, "target binding"):
                self._load_authorization_fixture(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            fixture["calibration_value"]["bindings"][
                "calibration_receipt_internal_sha256"
            ] = "0" * 64
            self._refresh_authorization_fixture(fixture, calibration=True)
            with self.assertRaisesRegex(common.ProtocolError, "bound Qwen run"):
                self._load_authorization_fixture(fixture)

    def test_audit_manifest_requires_exact_flat_regular_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            (fixture["source_receipt"].parent / "unexpected.txt").write_text(
                "not manifested\n", encoding="ascii"
            )
            with self.assertRaisesRegex(common.ProtocolError, "does not close"):
                self._load_authorization_fixture(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            fixture["source_manifest"].write_text(
                f"{'0'*64}  ../audit_receipt.json\n", encoding="ascii"
            )
            fixture["source_binding"]["manifest_sha256"] = common.sha256_file(
                fixture["source_manifest"]
            )
            self._refresh_authorization_fixture(fixture)
            with self.assertRaisesRegex(common.ProtocolError, "row is malformed"):
                self._load_authorization_fixture(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            link = fixture["source_receipt"].parent / "linked.txt"
            try:
                link.symlink_to(fixture["source_receipt"])
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            self._refresh_authorization_fixture(fixture, source=True)
            with self.assertRaisesRegex(common.ProtocolError, "non-regular"):
                self._load_authorization_fixture(fixture)

    def test_audit_manifest_order_and_verification_semantics_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            lines = fixture["source_manifest"].read_text(encoding="ascii").splitlines(True)
            fixture["source_manifest"].write_text(
                "".join(reversed(lines)), encoding="ascii"
            )
            fixture["source_binding"]["manifest_sha256"] = common.sha256_file(
                fixture["source_manifest"]
            )
            self._refresh_authorization_fixture(fixture)
            with self.assertRaisesRegex(common.ProtocolError, "not strictly sorted"):
                self._load_authorization_fixture(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            fixture["source_value"]["verification"]["manifest_closure_verified"] = False
            self._refresh_authorization_fixture(fixture, source=True)
            with self.assertRaisesRegex(common.ProtocolError, "verification semantics"):
                self._load_authorization_fixture(fixture)

    def test_authorization_json_rejects_duplicate_keys_and_nonfinite_constants(self):
        with self.assertRaisesRegex(common.ProtocolError, "duplicate JSON key"):
            tier_c_gate._json_object(b'{"action":"a","action":"b"}', "fixture")
        with self.assertRaisesRegex(common.ProtocolError, "non-finite JSON constant"):
            tier_c_gate._json_object(b'{"created_unix_ns":NaN}', "fixture")

    def test_content_authorization_rejects_action_manifest_audit_and_path_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            value = fixture["value"]
            value["action"] = "QWEN_AUX_33_DOMAIN_RETRY"
            clean = dict(value); clean.pop("authorization_receipt_sha256")
            value["authorization_receipt_sha256"] = common.sha256_bytes(
                common.canonical_json_bytes(clean)
            )
            fixture["path"].write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(common.ProtocolError, "action/one-shot"):
                self._load_authorization_fixture(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            value = fixture["value"]
            value["package"]["artifact_manifest_sha256"] = "0" * 64
            clean = dict(value); clean.pop("authorization_receipt_sha256")
            value["authorization_receipt_sha256"] = common.sha256_bytes(
                common.canonical_json_bytes(clean)
            )
            fixture["path"].write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(common.ProtocolError, "package binding"):
                self._load_authorization_fixture(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            fixture["source_receipt"].write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(common.ProtocolError, "member hash mismatch"):
                self._load_authorization_fixture(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            fixture["aux"] = (Path(directory) / "different_aux").resolve()
            with self.assertRaisesRegex(common.ProtocolError, "run-path binding"):
                self._load_authorization_fixture(fixture)

    def test_content_authorization_rejects_legacy_sentinel_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            original_lstat = common.lstat_or_none
            legacy = common.absolute_unresolved(
                Path(fixture["lock"]["authorization_protocol"]["forbidden_legacy_sentinel_paths"][0])
            )

            def inject_legacy(path):
                if common.absolute_unresolved(Path(path)) == legacy:
                    return mock.Mock()
                return original_lstat(path)

            with mock.patch.object(common, "lstat_or_none", side_effect=inject_legacy):
                with self.assertRaisesRegex(common.ProtocolError, "legacy existence-only"):
                    self._load_authorization_fixture(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = self._authorization_fixture(Path(directory))
            real = fixture["path"]
            link = Path(directory) / "authorization_link.json"
            try:
                link.symlink_to(real)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            fixture["path"] = common.absolute_unresolved(link)
            fixture["lock"]["authorization_protocol"]["receipt_path"] = str(fixture["path"])
            with self.assertRaisesRegex(common.ProtocolError, "symlink"):
                self._load_authorization_fixture(fixture)

    def test_authorization_rejects_original_link_dotdot_run_path_before_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._authorization_fixture(root)
            real = root / "real-qwen-parent"
            real.mkdir()
            link = root / "LINK"
            try:
                link.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            fixture["aux"] = link / ".." / "aux"
            self.assertIn("..", fixture["aux"].parts)
            with self.assertRaisesRegex(common.ProtocolError, "parent traversal"):
                self._load_authorization_fixture(fixture)

    def test_link_dotdot_run_path_fails_before_cuda_or_qwen_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real-qwen-parent"
            real.mkdir()
            link = root / "LINK"
            try:
                link.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            workspace = link / ".." / "workspace"
            output = root / "output"
            calibration_path = root / "calibration"
            source_trace_path = root / "trace"
            calibration_path.write_text("fixture", encoding="utf-8")
            source_trace_path.write_text("fixture", encoding="utf-8")
            lock = {"execution": {
                "production_authorization_receipt": str(root / "authorization.json"),
                "state_directory": "state_v4",
                "output_json": "result_v4.json",
            }}
            with mock.patch.object(common, "load_candidate_lock", return_value=lock), mock.patch.object(
                tier_c_gate, "_load_calibration", return_value={}
            ), mock.patch.object(
                kernels, "PhiloxRandomAccess", side_effect=AssertionError("CUDA reached")
            ), mock.patch.object(
                common, "load_source_rows", side_effect=AssertionError("Qwen consumer reached")
            ):
                with self.assertRaisesRegex(common.ProtocolError, "parent traversal"):
                    tier_c_gate.run_gate(
                        workspace, root / "aux", output,
                        calibration_path, source_trace_path, root / "v4", root / "audit",
                        resume=False,
                    )
            self.assertFalse(output.exists())

    def test_calibration_checks_te_environment_before_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            trace = root / "trace.json"
            trace.write_text("fixture\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {tier_c_gate.parity.SINGLE_PARAM_ENV: "0"}), mock.patch.object(
                kernels, "PhiloxRandomAccess", side_effect=AssertionError("CUDA reached")
            ):
                with self.assertRaisesRegex(common.ProtocolError, "before calibration"):
                    tier_c_gate.run_calibration(root / "calibration.json", trace)

    def test_failed_content_authorization_precedes_output_creation_and_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "must-remain-absent"
            calibration_path = root / "calibration"
            source_trace_path = root / "trace"
            calibration_path.write_text("fixture", encoding="utf-8")
            source_trace_path.write_text("fixture", encoding="utf-8")
            lock = {"execution": {
                "production_authorization_receipt": "missing-authorization.json",
                "state_directory": "state_v4",
                "output_json": "result_v4.json",
            }}
            with mock.patch.object(common, "load_candidate_lock", return_value=lock), mock.patch.object(
                tier_c_gate, "_load_calibration", return_value={}
            ), mock.patch.object(
                tier_c_gate, "_load_production_authorization",
                side_effect=common.ProtocolError("authorization rejected"),
            ), mock.patch.object(
                kernels, "PhiloxRandomAccess", side_effect=AssertionError("CUDA reached")
            ):
                with self.assertRaisesRegex(common.ProtocolError, "authorization rejected"):
                    tier_c_gate.run_gate(
                        root / "workspace", root / "aux", output,
                        calibration_path, source_trace_path, root / "v4", root / "audit",
                        resume=False,
                    )
            self.assertFalse(output.exists())

    def test_calibration_dangling_output_rejected_before_cuda_or_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "must-not-be-created.json"
            output = root / "dangling-calibration.json"
            try:
                output.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with mock.patch.object(
                kernels, "PhiloxRandomAccess", side_effect=AssertionError("CUDA reached")
            ), mock.patch.object(
                tier_c_gate.parity, "run_parity", side_effect=AssertionError("parity reached")
            ):
                with self.assertRaises(common.ProtocolError):
                    tier_c_gate.run_calibration(output, Path("trace.json"))
            self.assertFalse(target.exists())

    def test_production_output_symlink_and_missing_resume_rejected_before_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            link = root / "dangling-output-dir"
            try:
                link.symlink_to(root / "missing", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with mock.patch.object(
                common, "load_candidate_lock", side_effect=AssertionError("lock reached")
            ):
                with self.assertRaises(common.ProtocolError):
                    tier_c_gate.run_gate(
                        root / "workspace", root / "aux", link,
                        root / "calibration", root / "trace", root / "v4", root / "audit",
                        resume=False,
                    )
                with self.assertRaises(common.ProtocolError):
                    tier_c_gate.run_gate(
                        root / "workspace", root / "aux", root / "missing-resume",
                        root / "calibration", root / "trace", root / "v4", root / "audit",
                        resume=True,
                    )

    def test_strict_stage0_shard_accepts_only_exact_new_family(self):
        state = self._valid_stage0_shard_state()
        observed_ordinals, observed_metrics = tier_c_gate._validate_stage0_shard_state(
            state, 0
        )
        self.assertIs(observed_ordinals, state["top_ordinals"])
        self.assertIs(observed_metrics, state["top_q"])

    def test_stage0_resume_rejects_arbitrary_but_structurally_valid_metrics(self):
        replayed = self._valid_stage0_shard_state()
        observed = dict(replayed)
        observed["top_q"] = replayed["top_q"] * np.float64(17.0) + np.float64(12345.0)
        tier_c_gate._validate_stage0_shard_state(observed, 0)
        with self.assertRaisesRegex(common.ProtocolError, "payload-derived replay"):
            tier_c_gate._compare_stage0_shard_replay(observed, replayed, 0)

    def test_stage0_replay_scores_complete_164864_candidate_shard(self):
        candidates = np.arange(164_864, dtype=np.uint64)
        coordinates = np.arange(512, dtype=np.uint64)
        calls = {}

        class FakeAccess:
            cp = np

            def generate(self, ordinals, experts, roles, seen_coordinates):
                calls["generated_ordinals"] = ordinals
                calls["generated_coordinates"] = seen_coordinates
                return object()

        q_marker = np.zeros(
            (len(candidates), len(common.DOMAIN_IDS)), dtype=np.float64
        )
        top_ordinals = np.zeros((len(common.DOMAIN_IDS), common.STAGE0_TOP_K), dtype=np.uint64)
        top_q = np.zeros((len(common.DOMAIN_IDS), common.STAGE0_TOP_K), dtype=np.float64)

        def exact_topk(_cp, q, ordinals, width):
            self.assertIs(q, q_marker)
            self.assertIs(ordinals, candidates)
            self.assertEqual(width, common.STAGE0_TOP_K)
            return top_ordinals, top_q

        with mock.patch.object(
            common, "representative_ordinals", return_value=candidates
        ), mock.patch.object(
            tier_c_gate.BASE, "_stage0_q", return_value=q_marker
        ), mock.patch.object(
            tier_c_gate.BASE, "_exact_top_k", side_effect=exact_topk
        ):
            state = tier_c_gate._compute_stage0_shard_state(
                FakeAccess(), (), np.zeros(1, dtype=np.int32),
                np.zeros(1, dtype=np.int32), coordinates, (), 0,
            )
        self.assertEqual(len(calls["generated_ordinals"]), 164_864)
        self.assertEqual(len(calls["generated_coordinates"]), 512)
        self.assertIs(state["top_ordinals"], top_ordinals)
        self.assertIs(state["top_q"], top_q)

    def test_strict_stage0_rejects_stale_seed_metadata_and_seed_range(self):
        stale_metadata = self._valid_stage0_shard_state()
        stale_metadata["seed_start"] = np.asarray([1], dtype=np.int32)
        with self.assertRaisesRegex(common.ProtocolError, "stale or malformed"):
            tier_c_gate._validate_stage0_shard_state(stale_metadata, 0)

        stale_ordinal = self._valid_stage0_shard_state()
        stale_ordinal["top_ordinals"] = stale_ordinal["top_ordinals"].copy()
        stale_ordinal["top_ordinals"][0, 0] = common.representative_ordinals(256, 257)[0]
        with self.assertRaisesRegex(common.ProtocolError, "stale, retained-v4"):
            tier_c_gate._validate_stage0_shard_state(stale_ordinal, 0)

    def test_strict_stage0_rejects_old_noncanonical_and_out_of_family_ordinals(self):
        adversaries = {
            "retained_old_family": common.logical_ordinal(0, 0, 1, 0, 0, 0, 0),
            "noncanonical_abi": common.logical_ordinal(0, 4, 1, 0, 0, 0, 1),
            "collapsed_pp_descriptor": common.logical_ordinal(0, 1, 1, 3, 0, 0, 0),
        }
        for name, ordinal in adversaries.items():
            with self.subTest(name=name):
                state = self._valid_stage0_shard_state()
                state["top_ordinals"] = state["top_ordinals"].copy()
                state["top_ordinals"][0, 0] = ordinal
                with self.assertRaisesRegex(common.ProtocolError, "stale, retained-v4"):
                    tier_c_gate._validate_stage0_shard_state(state, 0)

    def test_strict_stage0_rejects_npz_schema_dtype_finite_and_order_tamper(self):
        cases = []
        extra = self._valid_stage0_shard_state(); extra["extra"] = np.asarray([1])
        cases.append(extra)
        wrong_dtype = self._valid_stage0_shard_state()
        wrong_dtype["top_q"] = wrong_dtype["top_q"].astype(np.float32)
        cases.append(wrong_dtype)
        nonfinite = self._valid_stage0_shard_state(); nonfinite["top_q"] = nonfinite["top_q"].copy()
        nonfinite["top_q"][0, 0] = np.nan; cases.append(nonfinite)
        unstable = self._valid_stage0_shard_state(); unstable["top_q"] = unstable["top_q"].copy()
        unstable["top_q"][0, :2] = (1.0, 0.0); cases.append(unstable)
        for index, state in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(common.ProtocolError):
                tier_c_gate._validate_stage0_shard_state(state, 0)

    def test_complete_stage0_q_rejects_dtype_shape_and_any_nonfinite_before_topk(self):
        class Access:
            cp = np

        tier_c_gate._validate_full_stage0_q(
            Access(), np.zeros((4, len(common.DOMAIN_IDS)), dtype=np.float64), 4
        )
        cases = [
            np.zeros((4, len(common.DOMAIN_IDS)), dtype=np.float32),
            np.zeros((3, len(common.DOMAIN_IDS)), dtype=np.float64),
            np.zeros((4, len(common.DOMAIN_IDS) - 1), dtype=np.float64),
        ]
        nan = np.zeros((4, len(common.DOMAIN_IDS)), dtype=np.float64)
        nan[-1, -1] = np.nan
        cases.append(nan)
        inf = np.zeros((4, len(common.DOMAIN_IDS)), dtype=np.float64)
        inf[2, 7] = np.inf
        cases.append(inf)
        for index, q in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(common.ProtocolError):
                tier_c_gate._validate_full_stage0_q(Access(), q, 4)

    def test_stage0_full_q_guard_precedes_exact_topk(self):
        class Access:
            cp = np

            @staticmethod
            def generate(ordinals, experts, roles, coordinates):
                return np.zeros((len(ordinals), len(coordinates)), dtype=np.float32)

        q = np.zeros((3, len(common.DOMAIN_IDS)), dtype=np.float64)
        q[-1, -1] = np.nan
        with mock.patch.object(
            common, "representative_ordinals", return_value=np.arange(3, dtype=np.uint64)
        ), mock.patch.object(
            tier_c_gate.BASE, "_stage0_q", return_value=q
        ), mock.patch.object(
            tier_c_gate.BASE, "_exact_top_k",
            side_effect=AssertionError("TopK must not be called"),
        ):
            with self.assertRaisesRegex(common.ProtocolError, "before TopK"):
                tier_c_gate._compute_stage0_shard_state(
                    Access(), (), np.zeros(1), np.zeros(1), np.zeros(1), (), 0
                )

    def test_global_stage0_must_equal_exact_recomputation_and_union(self):
        shard = self._valid_stage0_shard_state()
        ordinals = shard["top_ordinals"]
        metrics = shard["top_q"]
        union = np.unique(ordinals.reshape(-1))
        state = {
            "domain_top_ordinals": ordinals.copy(),
            "domain_top_q": metrics.copy(),
            "union_ordinals": union.copy(),
        }
        tier_c_gate._validate_stage0_global_state(
            state, ordinals, metrics, union
        )
        tampered = dict(state)
        tampered["domain_top_q"] = metrics.copy()
        tampered["domain_top_q"][0, 0] = -1.0
        with self.assertRaisesRegex(common.ProtocolError, "exact shard recomputation"):
            tier_c_gate._validate_stage0_global_state(
                tampered, ordinals, metrics, union
            )
        bad_union = dict(state)
        bad_union["union_ordinals"] = union[:-1]
        with self.assertRaisesRegex(common.ProtocolError, "union is not exact"):
            tier_c_gate._validate_stage0_global_state(
                bad_union, ordinals, metrics, union
            )

    def test_stage1_batch_schema_dtype_finiteness_and_union_slice_are_strict(self):
        union = common.full_representative_ordinals(0, 1)[:3]
        valid = {
            "ordinals": union.copy(),
            "q": np.zeros((3, len(common.DOMAIN_IDS)), dtype=np.float64),
        }
        tier_c_gate._validate_stage1_batch_state(valid, union, label="fixture")
        cases = []
        extra = dict(valid); extra["extra"] = np.asarray([1]); cases.append(extra)
        ordinal_dtype = dict(valid); ordinal_dtype["ordinals"] = union.astype(np.int64); cases.append(ordinal_dtype)
        q_dtype = dict(valid); q_dtype["q"] = valid["q"].astype(np.float32); cases.append(q_dtype)
        nonfinite = dict(valid); nonfinite["q"] = valid["q"].copy(); nonfinite["q"][0, 0] = np.nan; cases.append(nonfinite)
        stale_slice = dict(valid); stale_slice["ordinals"] = union[::-1].copy(); cases.append(stale_slice)
        for index, state in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(common.ProtocolError):
                tier_c_gate._validate_stage1_batch_state(state, union, label="fixture")

    def test_stage1_resume_replays_scores_and_rejects_stale_batch(self):
        union = common.full_representative_ordinals(0, 1)[:3]
        coordinates = np.zeros(48_624, dtype=np.uint64)
        replay = np.tile(np.arange(3, dtype=np.float64)[:, None], (1, len(common.DOMAIN_IDS)))

        class FakeCP:
            @staticmethod
            def asnumpy(value):
                return np.asarray(value)

        class FakeAccess:
            cp = FakeCP()

            def generate(self, ordinals, experts, roles, seen_coordinates):
                self.last_ordinals = ordinals.copy()
                self.last_coordinate_count = len(seen_coordinates)
                return np.zeros((len(ordinals), 1), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            tier_c_gate.BASE, "_flatten_coordinate_metadata",
            return_value=(np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32), coordinates, ()),
        ), mock.patch.object(
            tier_c_gate.BASE, "_stage1_q", side_effect=lambda *_: replay.copy()
        ), mock.patch.object(
            tier_c_gate.StateJournal, "_assert_next_event", return_value=None
        ):
            journal = tier_c_gate.StateJournal(Path(directory) / "state")
            access = FakeAccess()
            observed_o, observed_q = tier_c_gate._run_stage1_strict(access, journal, (), union)
            self.assertTrue(np.all(observed_o == union[0]))
            self.assertTrue(np.all(observed_q == 0.0))
            self.assertEqual(access.last_coordinate_count, 48_624)
            replay[0, :] = 99.0
            with self.assertRaisesRegex(common.ProtocolError, "payload-derived replay"):
                tier_c_gate._run_stage1_strict(access, journal, (), union)

    def test_stage1_resume_rejects_noncanonical_out_of_union_winners(self):
        union = common.full_representative_ordinals(0, 1)[:3]
        coordinates = np.zeros(48_624, dtype=np.uint64)
        q = np.tile(np.arange(3, dtype=np.float64)[:, None], (1, len(common.DOMAIN_IDS)))

        class FakeCP:
            @staticmethod
            def asnumpy(value):
                return np.asarray(value)

        class FakeAccess:
            cp = FakeCP()

            def generate(self, ordinals, experts, roles, seen_coordinates):
                return np.zeros((len(ordinals), 1), dtype=np.float32)

        noncanonical = common.logical_ordinal(0, 1, 1, 0, 0, 0, 1)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            tier_c_gate.BASE, "_flatten_coordinate_metadata",
            return_value=(np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32), coordinates, ()),
        ), mock.patch.object(
            tier_c_gate.BASE, "_stage1_q", return_value=q.copy()
        ), mock.patch.object(
            tier_c_gate.StateJournal, "_assert_next_event", return_value=None
        ):
            journal = tier_c_gate.StateJournal(Path(directory) / "state")
            journal.write_npz("stage1", "0000", ordinals=union.copy(), q=q.copy())
            journal.write_npz(
                "stage1_winners", "global",
                winner_ordinals=np.full(len(common.DOMAIN_IDS), noncanonical, dtype=np.uint64),
                winner_q=np.zeros(len(common.DOMAIN_IDS), dtype=np.float64),
            )
            with self.assertRaisesRegex(common.ProtocolError, "noncanonical/out-of-family"):
                tier_c_gate._run_stage1_strict(FakeAccess(), journal, (), union)

        outside = common.full_representative_ordinals(1, 2)[0]
        expected_o = np.full(len(common.DOMAIN_IDS), union[0], dtype=np.uint64)
        expected_q = np.zeros(len(common.DOMAIN_IDS), dtype=np.float64)
        state = {
            "winner_ordinals": np.full(len(common.DOMAIN_IDS), outside, dtype=np.uint64),
            "winner_q": expected_q.copy(),
        }
        with self.assertRaisesRegex(common.ProtocolError, "absent from the exact overlay union"):
            tier_c_gate._validate_stage1_winner_state(
                state, union, expected_o, expected_q
            )

    def test_stage1_rejects_surplus_recorded_batch_beyond_replayed_union(self):
        union = common.full_representative_ordinals(0, 1)[:3]

        class Journal:
            events_list = [
                {"kind": "stage1", "key": "0000"},
                {"kind": "stage1", "key": "0001"},
            ]

        with mock.patch.object(
            tier_c_gate.BASE, "_flatten_coordinate_metadata",
            return_value=(
                np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32),
                np.zeros(48_624, dtype=np.uint64), (),
            ),
        ):
            with self.assertRaisesRegex(common.ProtocolError, "exact prefix"):
                tier_c_gate._run_stage1_strict(object(), Journal(), (), union)

    def test_overlay_rejects_any_old_family_ordinal_before_state_access(self):
        state = self._valid_stage0_shard_state()
        state["top_ordinals"] = state["top_ordinals"].copy()
        state["top_ordinals"][0, 0] = common.logical_ordinal(0, 0, 1, 0, 0, 0, 0)
        with self.assertRaisesRegex(common.ProtocolError, "retained-v4"):
            tier_c_gate._run_overlay_merge(
                None, None, state["top_ordinals"], state["top_q"]
            )

    def test_completed_result_is_final_grammar_and_cannot_shortcut_replay(self):
        source = inspect.getsource(tier_c_gate.run_gate)
        prepare = source.index("_prepare_completed_result_replay")
        stage0 = source.index("_run_stage0_strict")
        stage1 = source.index("_run_stage1_strict")
        validation = source.index("_load_plan_payloads")
        finish = source.index("_commit_or_verify_result")
        self.assertLess(prepare, stage0)
        self.assertLess(stage0, stage1)
        self.assertLess(stage1, validation)
        self.assertLess(validation, finish)
        self.assertNotIn("return result_path", source[prepare:finish])

    def test_completed_result_exact_full_replay_comparison_rejects_any_drift(self):
        expected = {
            "schema": common.SCHEMA,
            "bindings": {"candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256},
            "search": {"stage1_winners": {"source": {"selection_q": 1.0}}},
            "validation": {"folds": {"source": {"pooled": {"capture": 0.2}}}},
            "decision": {"state": "fixture"},
        }
        completed = {
            "value": json.loads(json.dumps(expected)),
            "raw": tier_c_gate._canonical_result_file_bytes(expected),
        }
        result_path = Path("result.json")
        self.assertEqual(
            tier_c_gate._commit_or_verify_result(result_path, None, expected, completed),
            result_path,
        )
        mutations = []
        extra = json.loads(json.dumps(expected)); extra["extra"] = True; mutations.append(extra)
        winner = json.loads(json.dumps(expected)); winner["search"]["stage1_winners"]["source"]["selection_q"] = 9.0; mutations.append(winner)
        validation = json.loads(json.dumps(expected)); validation["validation"]["folds"]["source"]["pooled"]["capture"] = -1.0; mutations.append(validation)
        decision = json.loads(json.dumps(expected)); decision["decision"]["state"] = "fabricated"; mutations.append(decision)
        for index, replayed in enumerate(mutations):
            with self.subTest(index=index), self.assertRaisesRegex(
                common.ProtocolError, "full payload-derived scientific replay"
            ):
                tier_c_gate._commit_or_verify_result(
                    result_path, None, replayed, completed
                )

        noncanonical = dict(completed)
        noncanonical["raw"] = json.dumps(expected, sort_keys=True).encode("utf-8")
        with self.assertRaisesRegex(
            common.ProtocolError, "full payload-derived scientific replay"
        ):
            tier_c_gate._commit_or_verify_result(
                result_path, None, expected, noncanonical
            )

    def test_resumed_header_duplicate_keys_fail_before_semantic_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header.json"
            path.write_text('{"schema":"x","schema":"x"}', encoding="utf-8")

            class Journal:
                @staticmethod
                def lookup(kind, key):
                    self.assertEqual((kind, key), ("run_header", "immutable"))
                    return path

            with self.assertRaisesRegex(common.ProtocolError, "duplicate JSON key"):
                tier_c_gate._verify_or_create_header_strict(
                    Journal(), {"schema": "x"}
                )

    def test_self_consistent_completed_result_receipt_still_requires_recomputed_result(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            common, "SEED_SHARD_COUNT", 1
        ):
            root = Path(directory)
            journal = tier_c_gate.StateJournal(root / "state")
            journal.write_json("run_header", "immutable", {"fixture": True})
            journal.write_npz("stage0", "000", x=np.asarray([1]))
            journal.write_npz("stage0_merged", "global", x=np.asarray([1]))
            journal.write_npz("layout_overlay_merged", "global", x=np.asarray([1]))
            journal.write_json("layout_overlay_receipt", "global", {"fixture": True})
            journal.write_npz("stage1", "0000", x=np.asarray([1]))
            journal.write_npz("stage1_winners", "global", x=np.asarray([1]))
            journal.write_json("validation_firewall", "winners_frozen", {"fixture": True})
            result_path = root / "result.json"
            fabricated = {"schema": common.SCHEMA, "decision": {"state": "fabricated"}}
            common.write_json_create_new(result_path, fabricated, "fixture result")
            journal.write_json("result", "final", {
                "output_basename": result_path.name,
                "sha256": common.sha256_file(result_path),
                "bytes": result_path.stat().st_size,
            })
            completed, prior_events = tier_c_gate._prepare_completed_result_replay(
                journal, result_path, resume=True
            )
            self.assertEqual(len(prior_events), 8)
            with self.assertRaisesRegex(
                common.ProtocolError, "full payload-derived scientific replay"
            ):
                tier_c_gate._commit_or_verify_result(
                    result_path, journal,
                    {"schema": common.SCHEMA, "decision": {"state": "recomputed"}},
                    completed,
                )
            with self.assertRaisesRegex(common.ProtocolError, "immutable"):
                journal.write_json("result", "after_final", {"forbidden": True})

    def test_journal_rejects_result_before_header_or_firewall(self):
        with self.assertRaisesRegex(common.ProtocolError, "must begin"):
            tier_c_gate.StateJournal._validate_event_grammar([
                {"kind": "result", "key": "final"}
            ])
        with self.assertRaisesRegex(common.ProtocolError, "stage0 grammar"):
            tier_c_gate.StateJournal._validate_event_grammar([
                {"kind": "run_header", "key": "immutable"},
                {"kind": "result", "key": "final"},
            ])

    def test_state_journal_rejects_future_target_and_event_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = tier_c_gate.StateJournal(Path(directory) / "state")
            missing_target = Path(directory) / "never-created-target"
            state_link = journal.files / "run_header_immutable.json"
            event_link = journal.events / "000000.json"
            try:
                state_link.symlink_to(missing_target)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with self.assertRaises(common.ProtocolError):
                journal.write_json("run_header", "immutable", {"ok": True})
            state_link.unlink()
            event_link.symlink_to(Path(directory) / "missing-event")
            with self.assertRaises(common.ProtocolError):
                journal.write_json("run_header", "immutable", {"ok": True})
            self.assertFalse((journal.files / "run_header_immutable.json").exists())

    def test_state_journal_root_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real-state"
            real.mkdir()
            linked = root / "linked-state"
            dangling = root / "dangling-state"
            try:
                linked.symlink_to(real, target_is_directory=True)
                dangling.symlink_to(root / "missing-state", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            with self.assertRaises(common.ProtocolError):
                tier_c_gate.StateJournal(linked)
            with self.assertRaises(common.ProtocolError):
                tier_c_gate.StateJournal(dangling)

    def test_cli_has_no_scientific_search_knobs(self):
        self.assertEqual(vars(tier_c_gate.parse_args(["preflight"])), {"command": "preflight"})
        run = tier_c_gate.parse_args(["run", "--workspace-root", "workspace", "--aux-dir", "aux",
                                      "--output-dir", "output", "--calibration", "receipt.json",
                                      "--source-trace", "trace.json", "--v4-run-root", "v4",
                                      "--v4-result-audit", "audit.json"])
        self.assertEqual(set(vars(run)), {"command", "workspace_root", "aux_dir", "output_dir",
                                          "calibration", "source_trace", "v4_run_root",
                                          "v4_result_audit", "resume"})


if __name__ == "__main__":
    unittest.main()
