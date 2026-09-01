from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import common
import parity


GOOD_HASH = hashlib.sha256(b"nonempty-runtime-parity-content").hexdigest()


class _FakeGenerator:
    def __init__(self):
        self.offset = 0

    def get_offset(self):
        return self.offset


class _FakeTorch:
    def __init__(self, *, tamper_no_grad: bool = False):
        self.grad_enabled = True
        self.tamper_no_grad = tamper_no_grad
        self.no_grad_entries = 0

    def no_grad(self):
        owner = self

        class Guard:
            def __enter__(self):
                owner.no_grad_entries += 1
                self.previous = owner.grad_enabled
                if not owner.tamper_no_grad:
                    owner.grad_enabled = False

            def __exit__(self, *_):
                owner.grad_enabled = self.previous

        return Guard()


class _RequiresGradLeaf:
    requires_grad = True
    shape = (2, 3)
    dtype = "torch.bfloat16"
    device = "cuda:0"

    def __init__(self, torch, generator):
        self.torch = torch
        self.generator = generator
        self.normal_calls = 0

    def normal_(self, mean, std):
        if self.requires_grad and self.torch.grad_enabled:
            raise RuntimeError(
                "a leaf Variable that requires grad is being used in an in-place operation."
            )
        self.normal_calls += 1
        self.generator.offset += 4
        self.last_parameters = (mean, std)
        return self


class TEInitCallbackNoGradRegressionTests(unittest.TestCase):
    def test_callback_initializes_requires_grad_leaf_under_no_grad(self):
        torch = _FakeTorch()
        generator = _FakeGenerator()
        tensor = _RequiresGradLeaf(torch, generator)
        events = []
        with mock.patch.object(
            parity, "_rng_sha", side_effect=lambda *_: f"rng-{generator.offset}"
        ), mock.patch.object(parity, "_raw_sha", return_value=GOOD_HASH):
            parity._te_init_callback(torch, generator, "cuda:0", events, "fc1")(tensor)
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(torch.grad_enabled)
        self.assertEqual(torch.no_grad_entries, 1)
        self.assertEqual(tensor.normal_calls, 1)
        self.assertEqual(tensor.last_parameters, (0.0, 0.02))
        self.assertEqual((events[0]["offset_before"], events[0]["offset_after"]), (0, 4))
        self.assertEqual(events[0]["weight_sha256"], GOOD_HASH)

    def test_tampered_no_grad_guard_reproduces_leaf_inplace_failure(self):
        torch = _FakeTorch(tamper_no_grad=True)
        generator = _FakeGenerator()
        tensor = _RequiresGradLeaf(torch, generator)
        events = []
        with mock.patch.object(parity, "_rng_sha", return_value="rng"):
            callback = parity._te_init_callback(
                torch, generator, "cuda:0", events, "fc1"
            )
            with self.assertRaisesRegex(RuntimeError, "leaf Variable"):
                callback(tensor)
        self.assertEqual(events, [])
        self.assertEqual(generator.offset, 0)


class RuntimeReceiptFixture:
    def __init__(self, root: Path):
        self.root = root
        self.source_paths = {}
        self.hashes = {}
        for index, label in enumerate(parity.EXPECTED_RUNTIME_SOURCE_HASHES):
            path = root / f"{index:02d}_{label}.source"
            path.write_bytes(f"pinned-source-fixture:{label}\n".encode())
            self.source_paths[label] = path
            self.hashes[label] = common.sha256_file(path)
        self.trace_path = root / "source_trace.json"
        trace = {
            "schema": "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_source_trace_v2",
            "status": "PASS_EXACT_SEVEN_FILE_SOURCE_TRACE_RUNTIME_PARITY_STILL_REQUIRED",
            "mcore_revision": common.MCORE_REVISION,
            "transformer_engine_revision": common.TE_REVISION,
            "transformer_engine_source_version": common.TE_SOURCE_VERSION,
            "transformer_engine_pypi_version_policy": {
                "version": common.TE_PYPI_VERSION,
                "accepted_only_if_all_runtime_source_files_rehash_to_this_receipt": True,
            },
            "files": {label: {"relative_path": path.name, "sha256": self.hashes[label]}
                      for label, path in self.source_paths.items()},
            "semantic_edges": {"fixture": True},
            "procedural_geometry_trace": {"fixture": True},
            "claim_boundary": {
                "source_proves_ordinary_bf16_fc1_all_then_fc2_all_constructor_callback_order": True,
                "source_proves_numbered_then_copy_pack_order": True,
                "copy_pack_bitwise_and_terminal_rng_parity_source_only": False,
                "pytorch_cupy_philox_parity_source_only": False,
                "direct_bf16_vs_cast_parity_source_only": False,
                "numeric_full_pre_layer_15_expert_rng_lifecycle_source_only": False,
                "pp_cross_seed_equivalence_used": False,
            },
            "access_attestation": {
                "qwen_payload_or_manifest_opened_statted_or_hashed": False,
                "cuda_or_forbidden_runtime_imported": False,
                "only_the_seven_explicit_source_files_opened": True,
            },
            "execution_boundary": {
                "schema": "qwen3_tier_c_grouped_v5_path_boundary_v1",
                "action": "SOURCE_TRACE_CREATE_ONCE",
                "pairwise_lexical_inode_and_mount_disjoint": True,
                "revalidation_required_before_every_create_new": True,
            },
        }
        trace["receipt_sha256"] = common.sha256_bytes(common.canonical_json_bytes(trace))
        self.trace_path.write_text(json.dumps(trace, sort_keys=True), encoding="utf-8")

    def receipt(self):
        descriptors = []
        for row in parity._expected_descriptor_rows(170, 1536):
            descriptors.append({**row, "float32_sha256": GOOD_HASH,
                                "bf16_widened_sha256": GOOD_HASH})
        candidates = [{"candidate": candidate, "expert": expert, "role": role,
                       "coordinate_count": 9, "scaled_sha256": GOOD_HASH}
                      for candidate, expert, role in parity._expected_candidate_rows()]
        storage = [{"case": dict(case), "callback_count": 2 * case["local_experts"],
                    "events_sha256": GOOD_HASH, "member_hashes_sha256": GOOD_HASH,
                    "final_rng_offset": 4, "final_rng_state_sha256": GOOD_HASH,
                    "numbered_equals_copy_packed": True}
                   for case in parity.RUNTIME_TRACE_CASES]
        return {
            "schema": "qwen3_initialization_anchor_tier_c_grouped_v5_layout_overlay_runtime_parity_v1",
            "all_required_checks_passed": True,
            "source_trace_sha256": common.sha256_file(self.trace_path),
            "source_trace_internal_sha256": json.loads(self.trace_path.read_text())["receipt_sha256"],
            "transformer_engine_version": common.TE_PYPI_VERSION,
            "version_acceptance": {
                "observed": common.TE_PYPI_VERSION,
                "pypi_2_18_0_requires_exact_seven_file_source_rehash": True,
                "all_seven_source_hashes_matched": True,
            },
            "transformer_engine_revision": common.TE_REVISION,
            "mcore_revision": common.MCORE_REVISION,
            "runtime_source_files": {label: {"path": str(self.source_paths[label]),
                                               "sha256": self.hashes[label]}
                                     for label in self.source_paths},
            "single_param_environment": {"name": parity.SINGLE_PARAM_ENV, "value": "1"},
            "te_storage_parity": storage,
            "torch_version": "fixture", "cupy_version": "fixture", "device_name": "fixture",
            "device_index": 0, "multi_processor_count": 170,
            "max_threads_per_multi_processor": 1536,
            "descriptor_checks": descriptors, "candidate_coordinate_checks": candidates,
            "dlpack_sha256_f32le": GOOD_HASH,
            "qwen_manifest_directory_or_payload_accessed": False,
        }


class StrictRuntimeParityReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = RuntimeReceiptFixture(Path(self.temp.name))
        self.pin = mock.patch.object(parity, "EXPECTED_RUNTIME_SOURCE_HASHES", self.fixture.hashes)
        self.pin.start()

    def tearDown(self):
        self.pin.stop()
        self.temp.cleanup()

    def validate(self, value):
        return parity.validate_runtime_parity_receipt(value, self.fixture.trace_path)

    def test_full_nonempty_receipt_passes(self):
        observed = self.validate(self.fixture.receipt())
        self.assertEqual(len(observed["descriptor_checks"]), 40)
        self.assertEqual(len(observed["candidate_coordinate_checks"]), 1920)

    def test_empty_descriptor_rows_do_not_pass_by_length(self):
        value = self.fixture.receipt()
        value["descriptor_checks"] = [{} for _ in range(40)]
        with self.assertRaises(common.ProtocolError):
            self.validate(value)

    def test_empty_candidate_rows_do_not_pass_by_length(self):
        value = self.fixture.receipt()
        value["candidate_coordinate_checks"] = [{} for _ in range(1920)]
        with self.assertRaises(common.ProtocolError):
            self.validate(value)

    def test_descriptor_shape_or_increment_tamper_rejected(self):
        value = self.fixture.receipt()
        value["descriptor_checks"][0]["increment"] += 4
        with self.assertRaises(common.ProtocolError):
            self.validate(value)

    def test_candidate_identity_tamper_rejected(self):
        value = self.fixture.receipt()
        value["candidate_coordinate_checks"][0]["expert"] = 1
        with self.assertRaises(common.ProtocolError):
            self.validate(value)

    def test_empty_content_hash_rejected(self):
        value = self.fixture.receipt()
        value["te_storage_parity"][0]["events_sha256"] = hashlib.sha256(b"").hexdigest()
        with self.assertRaises(common.ProtocolError):
            self.validate(value)

    def test_runtime_source_is_rehashed_not_just_receipt_bound(self):
        value = self.fixture.receipt()
        self.fixture.source_paths["te_base"].write_bytes(b"tampered-after-receipt")
        with self.assertRaises(common.ProtocolError):
            self.validate(value)

    def test_runtime_source_symlink_rejected(self):
        value = self.fixture.receipt()
        link = Path(self.temp.name) / "source-link"
        try:
            link.symlink_to(self.fixture.source_paths["te_base"])
        except (OSError, NotImplementedError):
            self.skipTest("symlink unavailable")
        value["runtime_source_files"]["te_base"]["path"] = str(link)
        with self.assertRaises(common.ProtocolError):
            self.validate(value)

    def test_runtime_source_symlink_is_rejected_before_resolve(self):
        link = Path(self.temp.name) / "source-link-order"
        try:
            link.symlink_to(self.fixture.source_paths["te_base"])
        except (OSError, NotImplementedError):
            self.skipTest("symlink unavailable")
        row = {"path": str(link), "sha256": self.fixture.hashes["te_base"]}
        with mock.patch.object(Path, "resolve", side_effect=AssertionError("resolve reached")):
            with self.assertRaises(common.ProtocolError):
                parity._rehash_runtime_source(row, "te_base", self.fixture.hashes["te_base"])

    def test_runtime_source_dangling_symlink_rejected_without_target_access(self):
        target = Path(self.temp.name) / "missing-source"
        link = Path(self.temp.name) / "dangling-source-link"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("symlink unavailable")
        row = {"path": str(link), "sha256": self.fixture.hashes["te_base"]}
        with self.assertRaises(common.ProtocolError):
            parity._rehash_runtime_source(row, "te_base", self.fixture.hashes["te_base"])
        self.assertFalse(target.exists())

    def test_source_trace_internal_hash_tamper_rejected(self):
        trace = json.loads(self.fixture.trace_path.read_text())
        trace["status"] = "tampered"
        self.fixture.trace_path.write_text(json.dumps(trace), encoding="utf-8")
        with self.assertRaises(common.ProtocolError):
            self.validate(self.fixture.receipt())

    def test_pypi_2_18_0_is_allowed_only_with_exact_hash_map(self):
        fake = SimpleNamespace(__version__=common.TE_PYPI_VERSION)
        with mock.patch("importlib.metadata.version", side_effect=parity.importlib.metadata.PackageNotFoundError):
            self.assertEqual(parity._te_version(fake, {label: {"sha256": digest}
                                                       for label, digest in self.fixture.hashes.items()}),
                             common.TE_PYPI_VERSION)
            broken = {label: {"sha256": digest} for label, digest in self.fixture.hashes.items()}
            broken.pop(next(iter(broken)))
            with self.assertRaises(common.ProtocolError):
                parity._te_version(fake, broken)


if __name__ == "__main__":
    unittest.main()
