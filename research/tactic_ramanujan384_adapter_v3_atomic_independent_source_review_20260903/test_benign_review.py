#!/usr/bin/env python3
"""Independent source-only tests for the atomic-v3 adapter."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType


HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
PRODUCER = RESEARCH / "tactic_ramanujan384_adapter_v3_atomic"
BOOTSTRAP = RESEARCH / "tactic_ramanujan384_adapter_v3_atomic_bootstrap.py"
V2 = RESEARCH / "tactic_ramanujan384_adapter_v2_scalable"
V2_REVIEW = RESEARCH / "tactic_ramanujan384_adapter_v2_scalable_independent_source_review_20260903"
MANIFEST_SHA = "97fb4cba64ff884615810fc8fc835c12ce98bf3e9db37b8a77be93d0d5372be1"
SOURCE_ROOT = "5f86d9a1b48f7769867c828322132be303617d0444d50b5439f7b9d0074ab674"
BOOTSTRAP_SHA = "f7e8cd469b0ff9dd9ef09b400c63ec9f91e067f849d6b009588ea94ad6494375"
V2_MANIFEST_SHA = "1f579f33216edeebbebb6c1714a4e56739da30ae0f12ae9bd44baf15a6163209"
V2_ROOT = "bff5a0c541cb2117a8cc1db3e539493bacc590b4e007ab7f193ca615e03a7495"
REVIEW_MANIFEST_SHA = "4ed8c0fe24db072e22aef84791a01ccf637cb337376a389d47119248fd257281"
REVIEW_ROOT = "16ea8dfde5cf7a48552dc7b5a74b209488934b8764e890bf51bb5cd02985cd39"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load("independent_review_coarse_worker", PRODUCER / "coarse_byte_worker.py")


def closure_root(directory: Path, members: dict[str, bytes]) -> tuple[list[dict], str]:
    rows = []
    for name, payload in sorted(members.items()):
        (directory / name).write_bytes(payload)
        rows.append({"name": name, "bytes": len(payload), "sha256": sha(payload)})
    return rows, sha(canonical(rows))


class AtomicV3IndependentReview(unittest.TestCase):
    def test_01_producer_closure_and_root(self):
        payload = (PRODUCER / "SOURCE_MANIFEST.json").read_bytes()
        self.assertEqual(sha(payload), MANIFEST_SHA)
        manifest = json.loads(payload)
        observed = []
        for row in manifest["members"]:
            member = (PRODUCER / row["name"]).read_bytes()
            item = {"name": row["name"], "bytes": len(member), "sha256": sha(member)}
            self.assertEqual(item, row)
            observed.append(item)
        self.assertEqual(sha(canonical(observed)), SOURCE_ROOT)
        self.assertEqual(len(observed), 11)
        self.assertEqual({path.name for path in PRODUCER.iterdir()},
                         {row["name"] for row in observed} | {"SOURCE_MANIFEST.json"})

    def test_02_bootstrap_and_dependencies_match_publication_pins(self):
        self.assertEqual(sha(BOOTSTRAP.read_bytes()), BOOTSTRAP_SHA)
        self.assertEqual(sha((V2 / "SOURCE_MANIFEST.json").read_bytes()),
                         V2_MANIFEST_SHA)
        self.assertEqual(sha((V2_REVIEW / "source_manifest.json").read_bytes()),
                         REVIEW_MANIFEST_SHA)
        self.assertEqual(json.loads((V2 / "SOURCE_MANIFEST.json").read_bytes())
                         ["source_root_sha256"], V2_ROOT)
        self.assertEqual(json.loads((V2_REVIEW / "source_manifest.json").read_bytes())
                         ["source_root_sha256"], REVIEW_ROOT)

    def test_03_bootstrap_authenticates_before_project_compile(self):
        source = BOOTSTRAP.read_text("utf-8")
        self.assertIn("exact stable closure before and after descriptor reads", source)
        self.assertIn("write_snapshot(snapshot_root, combined)", source)
        self.assertIn("immutable = verify_snapshot(snapshot_root, combined)", source)
        self.assertIn("return MappingProxyType(output)", source)
        self.assertLess(source.index("immutable = verify_snapshot"),
                        source.index("return execute_runner"))

    def test_04_snapshot_mapping_is_shallow_read_only_with_bytes_values(self):
        source = BOOTSTRAP.read_text("utf-8")
        self.assertIn("output[relative] = payload", source)
        self.assertIn("return MappingProxyType(output)", source)
        proxy = MappingProxyType({"member.py": b"pass\n"})
        self.assertIs(type(proxy["member.py"]), bytes)
        with self.assertRaises(TypeError):
            proxy["member.py"] = b"raise SystemExit\n"  # type: ignore[index]

    def test_05_runner_compiles_project_modules_only_from_snapshot(self):
        source = (PRODUCER / "snapshot_runner.py").read_text("utf-8")
        self.assertIn("compile(snapshot[key]", source)
        self.assertIn("isinstance(snapshot_bytes, MappingProxyType)", source)
        self.assertNotIn("spec_from_file_location", source)
        self.assertNotIn("read_bytes", source)

    def test_06_bootstrap_self_hash_is_after_interpreter_execution_begins(self):
        source = BOOTSTRAP.read_text("utf-8")
        self.assertLess(source.index("import argparse"),
                        source.index("own_payload = safe_read(Path(__file__)"))
        self.assertLess(source.index("def main()"),
                        source.index("own_payload = safe_read(Path(__file__)"))

    def test_07_bootstrap_does_not_enforce_isolated_startup(self):
        source = BOOTSTRAP.read_text("utf-8")
        self.assertNotIn("sys.flags.isolated", source)
        self.assertNotIn("PYTHONPATH", source)

    def test_08_adapter_accepts_no_live_decoder_capability(self):
        source = (PRODUCER / "adapter_atomic.py").read_text("utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                        node.name == "run_authenticated_expert")
        names = [argument.arg for argument in function.args.args +
                 function.args.kwonlyargs]
        self.assertNotIn("decoder", names)
        self.assertNotIn("decode_literal", names)
        self.assertIn("no live decoder capability", source)

    def test_09_program_schema_is_byte_only_zero_import_and_pathless(self):
        coarse = bytes(64)
        program = {"schema": "tactic-coarse-byte-worker-program-v3",
                   "version": 1, "imports": [], "opcode": "ZERO_F32_LE",
                   "coarse_sha256": sha(coarse), "shape": [2, 2],
                   "roles": list(worker.ROLE_ORDER)}
        result = worker._execute_program(canonical(program), coarse, 2, 2,
                                         worker.ROLE_ORDER)
        self.assertIsInstance(result, MappingProxyType)
        hostile = dict(program)
        hostile["path"] = "model.bin"
        with self.assertRaises(worker.CoarseWorkerError):
            worker._execute_program(canonical(hostile), coarse, 2, 2,
                                    worker.ROLE_ORDER)

    def test_10_worker_and_auditor_roots_can_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = root / "COMMON"
            common.mkdir()
            coarse = bytes(3072)
            program = canonical({
                "schema": "tactic-coarse-byte-worker-program-v3", "version": 1,
                "imports": [], "opcode": "ZERO_F32_LE", "coarse_sha256": sha(coarse),
                "shape": [32, 32], "roles": list(worker.ROLE_ORDER)})
            spec = canonical({"schema": "shared-worker-auditor-fixture"})
            rows, shared_root = closure_root(
                common, {"worker_program.json": program, "shared.json": spec})
            worker_manifest = canonical({
                "schema": "tactic-coarse-worker-source-manifest-v3",
                "source_root_sha256": shared_root, "members": rows})
            auditor_manifest = canonical({
                "schema": "tactic-coarse-worker-auditor-source-manifest-v3",
                "source_root_sha256": shared_root, "members": rows})
            worker_manifest_path = root / "WORKER_MANIFEST.json"
            auditor_manifest_path = root / "AUDITOR_MANIFEST.json"
            worker_manifest_path.write_bytes(worker_manifest)
            auditor_manifest_path.write_bytes(auditor_manifest)
            program_sha = sha(program)
            zero_sha = sha(bytes(4096))
            receipt = canonical({
                "schema": "tactic-independent-coarse-worker-audit-receipt-v3",
                "status": "INDEPENDENT_COARSE_WORKER_AUDIT_PASS",
                "capability_id": "ALIASED_FIXTURE", "program_sha256": program_sha,
                "worker_source_manifest_sha256": sha(worker_manifest),
                "worker_source_root_sha256": shared_root,
                "auditor_source_manifest_sha256": sha(auditor_manifest),
                "auditor_source_root_sha256": shared_root,
                "coarse_sha256": sha(coarse), "shape": [32, 32],
                "role_order": list(worker.ROLE_ORDER),
                "output_f32_sha256_by_role": {role: zero_sha for role in worker.ROLE_ORDER},
                "literal_payload_only_pass": True, "zero_import_no_path_pass": True,
                "deterministic_output_hashes_recorded": True,
                "hostile_tests_passed": 12})
            receipt_path = root / "RECEIPT.json"
            receipt_path.write_bytes(receipt)
            capability = canonical({
                "schema": "tactic-coarse-byte-worker-capability-v3",
                "status": "INDEPENDENT_ZERO_IMPORT_WORKER_AUDIT_REQUIRED",
                "capability_id": "ALIASED_FIXTURE", "program_name": "worker_program.json",
                "program_sha256": program_sha,
                "worker_source_manifest_sha256": sha(worker_manifest),
                "worker_source_root_sha256": shared_root,
                "auditor_source_manifest_sha256": sha(auditor_manifest),
                "auditor_source_root_sha256": shared_root,
                "independent_audit_receipt_sha256": sha(receipt)})
            capability_path = root / "CAPABILITY.json"
            capability_path.write_bytes(capability)
            result = worker.authenticate_and_decode(
                capability_path=capability_path,
                expected_capability_sha256=sha(capability),
                worker_source_directory=common,
                worker_source_manifest_path=worker_manifest_path,
                expected_worker_source_manifest_sha256=sha(worker_manifest),
                expected_worker_source_root_sha256=shared_root,
                auditor_source_directory=common,
                auditor_source_manifest_path=auditor_manifest_path,
                expected_auditor_source_manifest_sha256=sha(auditor_manifest),
                expected_auditor_source_root_sha256=shared_root,
                independent_audit_receipt_path=receipt_path,
                expected_independent_audit_receipt_sha256=sha(receipt),
                coarse_payload=coarse, intermediate=32, hidden=32,
                role_order=worker.ROLE_ORDER)
            self.assertEqual(result["worker_source_root_sha256"],
                             result["auditor_source_root_sha256"])

    def test_11_six_pins_are_not_checked_for_uniqueness(self):
        source = (PRODUCER / "coarse_byte_worker.py").read_text("utf-8")
        self.assertIn("all(isinstance(value, str) and len(value) == 64 for value in pins)",
                      source)
        self.assertNotIn("len(set(pins))", source)

    def test_12_vm_contains_only_fixture_zero_opcode(self):
        source = (PRODUCER / "coarse_byte_worker.py").read_text("utf-8")
        self.assertIn('program["opcode"] == "ZERO_F32_LE"', source)
        self.assertNotIn('elif program["opcode"]', source)

    def test_13_pinned_v2_batched_search_and_controls_are_reused(self):
        core = (V2 / "scalable_core.py").read_text("utf-8")
        adapter = (PRODUCER / "adapter_atomic.py").read_text("utf-8")
        self.assertIn("solved = xp.linalg.solve", core)
        self.assertIn('candidate_corrections = xp.einsum("bnk,brk->brn"', core)
        self.assertIn('"per_candidate_host_scalar_syncs": 0', core)
        self.assertIn("for seed in core.GAUSSIAN_SEEDS", adapter)
        self.assertIn("controls = _run_controls", adapter)
        self.assertIn("core.MIN_CONTROL_EXCESS_BPW", adapter)

    def test_14_target_fixture_ledger_is_exactly_2p5_bpw(self):
        weights = 3 * 128 * 2048
        coarse_bytes = 307 * weights // 1024
        fine_bytes = 3 * 64 * 48
        unpadded = 512 + coarse_bytes + fine_bytes
        physical = ((unpadded + 4095) // 4096) * 4096
        self.assertEqual((weights, coarse_bytes, fine_bytes, physical),
                         (786432, 235776, 9216, 245760))
        self.assertEqual(8 * physical / weights, 2.5)

    def test_15_runtime_payload_and_network_are_held(self):
        status = json.loads((PRODUCER / "EXECUTION_STATUS.json").read_bytes())
        self.assertEqual(status["python_source_tests"], "NOT_EXECUTED_NO_LOCAL_PYTHON")
        self.assertEqual(status["atomic_bootstrap_verify_only"],
                         "NOT_EXECUTED_NO_LOCAL_PYTHON")
        self.assertFalse(status["qwen_payload_accessed"])
        self.assertFalse(status["coarse_model_payload_accessed"])
        self.assertFalse(status["network_accessed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

