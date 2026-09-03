#!/usr/bin/env python3
"""Source-only v3 unit tests; no network or model payload aperture."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType


ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = load("tactic_ramanujan384_v3_atomic_worker", ROOT / "coarse_byte_worker.py")
fixture = load("tactic_r384_v3_test_fixture", ROOT / "source_free_fixture_atomic.py")
adapter = load("tactic_r384_v3_test_adapter", ROOT / "adapter_atomic.py")
runner = load("tactic_r384_v3_test_runner", ROOT / "snapshot_runner.py")
core = load("tactic_r384_v3_test_core",
            RESEARCH / "tactic_ramanujan384_adapter_v2_scalable" / "scalable_core.py")
io = load("tactic_r384_v3_test_io",
          RESEARCH / "tactic_ramanujan384_adapter_v2_scalable" / "authenticated_io.py")


class ByteWorkerTests(unittest.TestCase):
    def test_byte_only_zero_import_program_is_deterministic(self):
        coarse = bytes(3072)
        document = {
            "schema": "tactic-coarse-byte-worker-program-v3", "version": 1,
            "imports": [], "opcode": "ZERO_F32_LE",
            "coarse_sha256": hashlib.sha256(coarse).hexdigest(),
            "shape": [32, 32], "roles": list(worker.ROLE_ORDER),
        }
        payload = worker.canonical_json(document)
        first = worker._execute_program(payload, coarse, 32, 32, worker.ROLE_ORDER)
        second = worker._execute_program(payload, coarse, 32, 32, worker.ROLE_ORDER)
        self.assertIsInstance(first, MappingProxyType)
        self.assertEqual(dict(first), dict(second))
        self.assertTrue(all(value == bytes(4096) for value in first.values()))

    def test_worker_rejects_import_and_path_fields(self):
        coarse = bytes(16)
        base = {
            "schema": "tactic-coarse-byte-worker-program-v3", "version": 1,
            "imports": [], "opcode": "ZERO_F32_LE",
            "coarse_sha256": hashlib.sha256(coarse).hexdigest(),
            "shape": [1, 1], "roles": list(worker.ROLE_ORDER),
        }
        hostile = dict(base)
        hostile["imports"] = ["filesystem"]
        with self.assertRaisesRegex(worker.CoarseWorkerError, "zero-import"):
            worker._execute_program(worker.canonical_json(hostile), coarse, 1, 1,
                                    worker.ROLE_ORDER)
        hostile = dict(base)
        hostile["path"] = "source.bf16"
        with self.assertRaisesRegex(worker.CoarseWorkerError, "exact schema"):
            worker._execute_program(worker.canonical_json(hostile), coarse, 1, 1,
                                    worker.ROLE_ORDER)

    def test_full_capability_has_exact_separate_closures_and_no_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            role_inputs, arguments = fixture.build(
                Path(temporary), core=core, io=io, intermediate=32, hidden=32
            )
            coarse = role_inputs[0]["coarse_artifact_path"].read_bytes()
            result = worker.authenticate_and_decode(
                **arguments, coarse_payload=coarse, intermediate=32, hidden=32,
                role_order=worker.ROLE_ORDER,
            )
            self.assertFalse(result["mutable_decoder_object_used"])
            self.assertTrue(result["zero_import_no_path_byte_worker"])
            (arguments["worker_source_directory"] / "EXTRA").write_bytes(b"x")
            with self.assertRaisesRegex(worker.CoarseWorkerError, "exact closure"):
                worker.authenticate_and_decode(
                    **arguments, coarse_payload=coarse, intermediate=32, hidden=32,
                    role_order=worker.ROLE_ORDER,
                )


class SnapshotTests(unittest.TestCase):
    def test_external_bootstrap_is_pinned_and_preimport(self):
        lock = json.loads((ROOT / "dependency_lock.json").read_text(encoding="ascii"))
        path = RESEARCH / "tactic_ramanujan384_adapter_v3_atomic_bootstrap.py"
        source = path.read_text(encoding="utf-8")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         lock["external_bootstrap"]["sha256"])
        self.assertNotIn("importlib", source)
        self.assertIn("MappingProxyType(output)", source)
        self.assertIn("exact stable closure before and after descriptor reads", source)
        self.assertLess(source.index("immutable = verify_snapshot"),
                        source.index("return execute_runner"))

    def test_runner_has_no_filesystem_module_loader(self):
        source = inspect.getsource(runner)
        self.assertNotIn("spec_from_file_location", source)
        self.assertNotIn("read_bytes", source)
        self.assertIn("compile(snapshot[key]", source)
        self.assertIn("isinstance(snapshot_bytes, MappingProxyType)", source)


class PreservedV2Tests(unittest.TestCase):
    def test_dependency_roots_and_core_hash_are_exact(self):
        lock = json.loads((ROOT / "dependency_lock.json").read_text(encoding="ascii"))
        v2_manifest_path = RESEARCH / lock["v2"]["path"].split("research/", 1)[1] / "SOURCE_MANIFEST.json"
        review_manifest_path = RESEARCH / lock["v2_review"]["path"].split("research/", 1)[1] / "source_manifest.json"
        self.assertEqual(hashlib.sha256(v2_manifest_path.read_bytes()).hexdigest(),
                         lock["v2"]["source_manifest_sha256"])
        self.assertEqual(hashlib.sha256(review_manifest_path.read_bytes()).hexdigest(),
                         lock["v2_review"]["source_manifest_sha256"])
        v2_manifest = json.loads(v2_manifest_path.read_text(encoding="ascii"))
        self.assertEqual(v2_manifest["source_root_sha256"], lock["v2"]["source_root_sha256"])
        by_name = {row["name"]: row for row in v2_manifest["members"]}
        self.assertEqual(hashlib.sha256(
            (v2_manifest_path.parent / "scalable_core.py").read_bytes()).hexdigest(),
            by_name["scalable_core.py"]["sha256"])

    def test_exact_target_fixture_rate_and_controls(self):
        shape = core.define_shape(128, 2048)
        ledger = core.physical_ledger(shape)
        self.assertEqual(shape.total_values, 786432)
        self.assertEqual(ledger["physical_bytes"], 245760)
        self.assertEqual(ledger["physical_rate_bpw"], 2.5)
        source = inspect.getsource(adapter)
        self.assertIn("for seed in core.GAUSSIAN_SEEDS", source)
        self.assertIn("controls = _run_controls", source)
        self.assertNotIn("decoder: Any", source)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    )
    print(json.dumps({
        "schema": "tactic-ramanujan384-v3-atomic-source-only-tests",
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "passed": result.wasSuccessful(),
        "qwen_payload_accessed": False, "coarse_model_payload_accessed": False,
        "network_accessed": False,
    }, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
