#!/usr/bin/env python3
"""Source-only adversarial release tests; never import CuPy or touch payload."""

from __future__ import annotations

import hashlib
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGE_MEMBERS = {
    "authorization_contract.json", "audit_lock_entrypoint.py",
    "launch_manifest.json", "lossy_tail_core.py", "lossy_tail_oracle.py", "preflight_launch.py",
    "protocol_lock.json", "repair_lock.json", "runtime_calibrate.py",
    "runtime_contract.json", "source_bindings.json",
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("loader missing")
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREFLIGHT = load("lossy_tail_v8_preflight_test", HERE / "preflight_launch.py")


class ReleaseSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.stage = self.root / "stage"
        self.stage.mkdir()
        for name in STAGE_MEMBERS:
            shutil.copyfile(HERE / name, self.stage / name)
        self.manifest = self.stage / "launch_manifest.json"
        self.manifest_sha = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def audit(self, launcher: Path | None = None, *, optimize: bool = False, manifest: Path | None = None, digest: str | None = None, cwd: Path | None = None):
        command = [sys.executable, "-B", "-I"]
        if optimize:
            command.append("-O")
        command.extend([
            os.fspath(launcher or self.stage / "audit_lock_entrypoint.py"),
            "--manifest", os.fspath(manifest or self.manifest),
            "--manifest-sha256", digest or self.manifest_sha,
        ])
        return subprocess.run(command, text=True, capture_output=True, check=False, cwd=cwd or self.root)

    def test_exact_isolated_stage_audit_passes(self):
        result = self.audit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(receipt["event"], "V8_SOURCE_ONLY_STAGE_AND_LOCK_PASS")
        self.assertEqual(receipt["payload_files_opened"], 0)

    def test_optimized_audit_is_rejected(self):
        result = self.audit(optimize=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without optimization", result.stdout + result.stderr)

    def test_added_directory_is_rejected(self):
        (self.stage / "undeclared_directory").mkdir()
        result = self.audit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden stage entry", result.stdout + result.stderr)

    def test_duplicate_manifest_row_is_rejected(self):
        value = json.loads(self.manifest.read_text())
        value["members"].append(dict(value["members"][0]))
        self.manifest.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))
        digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        result = self.audit(digest=digest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cardinality", result.stdout + result.stderr)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink unavailable")
    def test_raw_link_parent_launcher_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        alias = outside / "LINK"
        try:
            os.symlink(self.stage, alias, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink unavailable: {exc}")
        spelling = Path(os.fspath(alias) + os.sep + ".." + os.sep + "stage" + os.sep + "audit_lock_entrypoint.py")
        result = self.audit(launcher=spelling)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("raw argv0", result.stdout + result.stderr)

    def test_relative_launcher_is_rejected(self):
        result = self.audit(launcher=Path("audit_lock_entrypoint.py"), cwd=self.stage)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("raw argv0", result.stdout + result.stderr)

    def test_runtime_calibrator_unknown_or_bypass_flags_reject_before_output(self):
        output = self.root / "runtime_run" / "runtime_receipt.json"
        environment = dict(os.environ)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        command = [
            sys.executable, "-B", "-I", os.fspath(self.stage / "runtime_calibrate.py"),
            "--manifest", os.fspath(self.manifest), "--manifest-sha256", self.manifest_sha,
            "--output", os.fspath(output), "--self-test", "1",
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact grammar", result.stdout + result.stderr)
        self.assertFalse(output.exists())
        self.assertFalse(output.parent.exists())

    def test_runtime_calibrator_requires_explicit_cuda_visibility_before_import(self):
        output = self.root / "runtime_run" / "runtime_receipt.json"
        environment = dict(os.environ)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        command = [
            sys.executable, "-B", "-I", os.fspath(self.stage / "runtime_calibrate.py"),
            "--manifest", os.fspath(self.manifest), "--manifest-sha256", self.manifest_sha,
            "--output", os.fspath(output),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", result.stdout + result.stderr)
        self.assertFalse(output.exists())

    def test_preflight_requires_explicit_cuda_visibility_before_authorization_open(self):
        environment = dict(os.environ)
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        missing = self.root / "absent_authorization.json"
        command = [
            sys.executable, "-B", "-I", os.fspath(self.stage / "preflight_launch.py"),
            "--manifest", os.fspath(self.manifest), "--manifest-sha256", self.manifest_sha,
            "--authorization", os.fspath(missing), "--authorization-sha256", "0" * 64,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", result.stdout + result.stderr)

    def test_disjoint_predicate_rejects_both_ancestry_directions(self):
        stage = self.root / "protected"
        nested = stage / "nested"
        with self.assertRaises(SystemExit):
            PREFLIGHT.disjoint((("stage", stage), ("run", nested)))
        with self.assertRaises(SystemExit):
            PREFLIGHT.disjoint((("run", nested), ("stage", stage)))

    def test_scientific_core_direct_entry_rejects_before_numpy_import(self):
        command = [
            sys.executable, "-X", "importtime", "-B", "-I",
            os.fspath(self.stage / "lossy_tail_core.py"),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("V8_CORE_FIREWALL_REJECT", combined)
        self.assertNotIn("import time: numpy", combined)

    def test_forged_production_context_rejects_before_numpy_import(self):
        authorization = self.root / "forged_authorization.json"
        authorization.write_text("{}\n", encoding="utf-8")
        wrapper = self.root / "forged_context_wrapper.py"
        values = [
            "--bindings", os.fspath(self.stage / "source_bindings.json"),
            "--protocol", os.fspath(self.stage / "protocol_lock.json"),
            "--repair-lock", os.fspath(self.stage / "repair_lock.json"),
            "--runtime-contract", os.fspath(self.stage / "runtime_contract.json"),
            "--authorization-contract", os.fspath(self.stage / "authorization_contract.json"),
            "--launch-manifest", os.fspath(self.manifest),
            "--launch-manifest-sha256", self.manifest_sha,
            "--authorization", os.fspath(authorization),
            "--authorization-sha256", hashlib.sha256(authorization.read_bytes()).hexdigest(),
            "--control-replicates", "4", "--maximum-coordinate-passes", "4",
        ]
        wrapper.write_text(
            "import os, sys, types\n"
            f"core_path = {os.fspath(self.stage / 'lossy_tail_core.py')!r}\n"
            f"bootstrap_path = {os.fspath(self.stage / 'lossy_tail_oracle.py')!r}\n"
            f"arguments = {values!r}\n"
            "module = types.ModuleType('forged_v8_core')\n"
            "module.__file__ = core_path\n"
            "module.__dict__['__V8_CORE_CONTEXT__'] = {\n"
            " 'schema':'lossy-tail-v8-core-context-v1', 'mode':'production_child',\n"
            " 'parent_pid':os.getppid(), 'child_pid':os.getpid(),\n"
            " 'capability_sha256':'0'*64,\n"
            " 'preflight_memfd_fd':31337, 'preflight_memfd_st_dev':0,\n"
            " 'preflight_memfd_st_ino':0, 'preflight_memfd_bytes':0,\n"
            " 'preflight_memfd_seals':0, 'preflight_memfd_sha256':'0'*64,\n"
            " 'output_parent_fd':31338, 'output_parent_st_dev':0, 'output_parent_st_ino':0,\n"
            " 'launch_manifest_sha256':arguments[13], 'authorization_file_sha256':arguments[17],\n"
            " 'authorization_internal_sha256':'0'*64}\n"
            "sys.argv = [bootstrap_path, *arguments]\n"
            "exec(compile(open(core_path, 'rb').read(), core_path, 'exec'), module.__dict__)\n",
            encoding="utf-8",
        )
        command = [sys.executable, "-X", "importtime", "-B", "-I", os.fspath(wrapper)]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("V8_CORE_FIREWALL_REJECT", combined)
        self.assertNotIn("import time: numpy", combined)

    def test_bootstrap_is_stdlib_only_and_core_firewall_precedes_numpy(self):
        bootstrap = ast.parse((HERE / "lossy_tail_oracle.py").read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(bootstrap):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertTrue(imported_roots.isdisjoint({"numpy", "cupy", "torch"}))

        core = ast.parse((HERE / "lossy_tail_core.py").read_text(encoding="utf-8"))
        numpy_index = next(
            index for index, node in enumerate(core.body)
            if isinstance(node, ast.Import) and any(alias.name == "numpy" for alias in node.names)
        )
        firewall_index = next(
            index for index, node in enumerate(core.body)
            if isinstance(node, ast.If) and "V8_CORE_FIREWALL_REJECT" in ast.unparse(node)
        )
        self.assertLess(firewall_index, numpy_index)

    def test_direct_bootstrap_without_capability_cannot_reach_core(self):
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        command = [sys.executable, "-B", "-I", os.fspath(self.stage / "lossy_tail_oracle.py")]
        result = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid frozen v8 child grammar", result.stdout + result.stderr)

    @unittest.skipUnless(os.name == "posix", "inherited-descriptor adversary requires POSIX")
    def test_direct_bootstrap_rejects_non_socket_capability_descriptor(self):
        authorization = self.root / "not_authority.json"
        authorization.write_text("{}\n", encoding="utf-8")
        descriptor = os.open(os.fspath(authorization), os.O_RDONLY)
        preflight_descriptor = os.open(os.fspath(self.stage / "preflight_launch.py"), os.O_RDONLY)
        output_descriptor = os.open(os.fspath(self.root), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = "0"
            command = [
                sys.executable, "-B", "-I", os.fspath(self.stage / "lossy_tail_oracle.py"),
                "--bindings", os.fspath(self.stage / "source_bindings.json"),
                "--protocol", os.fspath(self.stage / "protocol_lock.json"),
                "--repair-lock", os.fspath(self.stage / "repair_lock.json"),
                "--runtime-contract", os.fspath(self.stage / "runtime_contract.json"),
                "--authorization-contract", os.fspath(self.stage / "authorization_contract.json"),
                "--launch-manifest", os.fspath(self.manifest),
                "--launch-manifest-sha256", self.manifest_sha,
                "--authorization", os.fspath(authorization),
                "--authorization-sha256", hashlib.sha256(authorization.read_bytes()).hexdigest(),
                "--control-replicates", "4", "--maximum-coordinate-passes", "4",
                "--preflight-memfd-fd", str(preflight_descriptor),
                "--output-parent-fd", str(output_descriptor),
                "--capability-fd", str(descriptor),
            ]
            result = subprocess.run(
                command, text=True, capture_output=True, check=False, env=environment,
                pass_fds=(descriptor, preflight_descriptor, output_descriptor),
            )
        finally:
            os.close(descriptor)
            os.close(preflight_descriptor)
            os.close(output_descriptor)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not an inherited socket", result.stdout + result.stderr)

    def test_runtime_probe_release_barrier_covers_live_affine_consumers(self):
        source = (HERE / "lossy_tail_core.py").read_text(encoding="utf-8")

        def release_barrier_accepts(candidate: str) -> bool:
            tree = ast.parse(candidate)
            runtime = next(
                node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "runtime_probe"
            )
            def deleted_names(node: ast.Delete) -> set[str]:
                names: set[str] = set()
                for target in ast.walk(node):
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                return names

            deletion = next(
                node for node in ast.walk(runtime)
                if isinstance(node, ast.Delete)
                and deleted_names(node) >= {
                    "affine_table", "gathered", "rounding", "table", "table_words", "words", "zbf"
                }
            )
            syncs = [
                node for node in ast.walk(runtime)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "synchronize"
            ]
            frees = [
                node for node in ast.walk(runtime)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "free_all_blocks"
            ]
            return bool(syncs and frees and deletion.lineno < min(node.lineno for node in frees))

        self.assertTrue(release_barrier_accepts(source))
        for removed in ("affine_table", "gathered", "rounding"):
            mutated = source.replace(f"                affine_table, gathered, rng,", f"                {', '.join(name for name in ('affine_table', 'gathered') if name != removed)}, rng,") if removed in {"affine_table", "gathered"} else source.replace("raw, bits, rounding, words", "raw, bits, words")
            with self.subTest(removed=removed), self.assertRaises(StopIteration):
                release_barrier_accepts(mutated)

    def test_immutable_preflight_descriptor_provenance_mutation_panel(self):
        preflight = (HERE / "preflight_launch.py").read_text(encoding="utf-8")
        bootstrap = (HERE / "lossy_tail_oracle.py").read_text(encoding="utf-8")
        core = (HERE / "lossy_tail_core.py").read_text(encoding="utf-8")

        def accepts(preflight_source: str, bootstrap_source: str, core_source: str) -> bool:
            preflight_tokens = (
                "os.memfd_create", "fcntl.F_ADD_SEALS", "fcntl.F_GET_SEALS",
                "os.execve", "descriptor_entry", "validate_sealed_preflight_descriptor",
                '"preflight_memfd_sha256"', '"preflight_memfd_seals"',
            )
            bootstrap_tokens = (
                "verify_inherited_preflight", "payload != expected_payload",
                "F_GET_SEALS", 'f"/proc/{peer_pid}/fd/{descriptor}"',
            )
            core_tokens = (
                "preflight_payload != stage_preflight_payload", "F_GET_SEALS",
                'f"/proc/{context[\'parent_pid\']}/fd/{preflight_fd}"',
            )
            combined = preflight_source + bootstrap_source + core_source
            return (
                all(token in preflight_source for token in preflight_tokens)
                and all(token in bootstrap_source for token in bootstrap_tokens)
                and all(token in core_source for token in core_tokens)
                and "/proc/self/cmdline" not in combined
                and "/proc/{peer_pid}/cmdline" not in combined
            )

        self.assertTrue(accepts(preflight, bootstrap, core))
        mutations = (
            (preflight.replace("fcntl.F_ADD_SEALS", "fcntl.F_GET_SEALS", 1), bootstrap, core),
            (preflight.replace("os.memfd_create", "os.open", 1), bootstrap, core),
            (preflight, bootstrap.replace("payload != expected_payload", "False", 1), core),
            (preflight, bootstrap, core.replace("preflight_payload != stage_preflight_payload", "False", 1)),
            (preflight + "\n# /proc/self/cmdline\n", bootstrap, core),
        )
        for ordinal, mutation in enumerate(mutations):
            with self.subTest(mutation=ordinal):
                self.assertFalse(accepts(*mutation))

    def test_independent_pass_statuses_are_frozen_not_authorization_chosen(self):
        preflight = (HERE / "preflight_launch.py").read_text(encoding="utf-8")
        core = (HERE / "lossy_tail_core.py").read_text(encoding="utf-8")
        contract = json.loads((HERE / "authorization_contract.json").read_text(encoding="utf-8"))
        frozen = {
            "SOURCE_AUDIT_MANIFEST_SCHEMA": "lossy-tail-v8-independent-source-audit-manifest-v1",
            "SOURCE_AUDIT_MANIFEST_STATUS": "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET",
            "SOURCE_AUDIT_RECEIPT_SCHEMA": "lossy-tail-v8-independent-source-audit-receipt-v1",
            "SOURCE_AUDIT_PASS_STATUS": "PASS_V8_INDEPENDENT_SOURCE_AUDIT",
            "RUNTIME_RECEIPT_STATUS": "UNTRUSTED_UNTIL_INDEPENDENT_RUNTIME_AUDIT",
            "RUNTIME_AUDIT_MANIFEST_SCHEMA": "lossy-tail-v8-independent-runtime-audit-manifest-v1",
            "RUNTIME_AUDIT_MANIFEST_STATUS": "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET",
            "RUNTIME_AUDIT_RECEIPT_SCHEMA": "lossy-tail-v8-independent-runtime-audit-receipt-v1",
            "RUNTIME_AUDIT_PASS_STATUS": "PASS_V8_INDEPENDENT_RUNTIME_AUDIT",
        }

        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        def accepts(preflight_source: str, core_source: str, contract_value: dict) -> bool:
            required = contract_value.get("required_values", {})
            return (
                all(f'{name} = "{value}"' in preflight_source for name, value in frozen.items())
                and all(f'{name} = "{value}"' in core_source for name, value in frozen.items())
                and all(value in required.values() for value in frozen.values())
                and "required_status" not in set(keys(contract_value))
                and '"required_status"' not in json.dumps(contract_value, sort_keys=True)
                and '"required_status"' not in preflight_source
                and '"required_status"' not in core_source
            )

        self.assertTrue(accepts(preflight, core, contract))
        mutated_preflight = preflight.replace(
            'SOURCE_AUDIT_PASS_STATUS = "PASS_V8_INDEPENDENT_SOURCE_AUDIT"',
            'SOURCE_AUDIT_PASS_STATUS = "AUTHORIZATION_CHOSEN"', 1,
        )
        self.assertFalse(accepts(mutated_preflight, core, contract))
        mutated_contract = json.loads(json.dumps(contract))
        mutated_contract["identity_fields"]["source_audit"].append("required_status")
        self.assertFalse(accepts(preflight, core, mutated_contract))

    def test_output_commit_is_descriptor_relative_mutation_panel(self):
        source = (HERE / "lossy_tail_core.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        writer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "write_sealed_json_descriptor")
        writer_source = ast.get_source_segment(source, writer) or ""

        def accepts(candidate: str) -> bool:
            return (
                candidate.count("dir_fd=output_parent_descriptor") >= 3
                and "dir_fd=run_descriptor" in candidate
                and "src_dir_fd=run_descriptor" in candidate
                and "dst_dir_fd=run_descriptor" in candidate
                and "os.O_EXCL" in candidate and candidate.count("O_NOFOLLOW") >= 2
                and "os.fsync(run_descriptor)" in candidate
                and "os.fsync(output_parent_descriptor)" in candidate
                and "os.fspath(" not in candidate
                and ".exists(" not in candidate
            )

        self.assertTrue(accepts(writer_source))
        for old, new in (
            ("dir_fd=output_parent_descriptor", "",),
            ("src_dir_fd=run_descriptor", "src_dir_fd=None"),
            ("os.O_EXCL", "0"),
            ("O_NOFOLLOW", "NO_FOLLOW_REMOVED"),
            ("os.fsync(output_parent_descriptor)", "pass"),
        ):
            with self.subTest(removed=old):
                self.assertFalse(accepts(writer_source.replace(old, new, 1)))
        authorization_source = ast.get_source_segment(
            source,
            next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_production_authorization"),
        ) or ""
        self.assertNotIn("run_root.exists", authorization_source)
        self.assertNotIn("result.exists", authorization_source)
        self.assertIn("output_parent_metadata = os.fstat(output_parent_descriptor)", authorization_source)

    def test_all_runtime_probe_rows_emit_six_memory_fields(self):
        source = (HERE / "lossy_tail_core.py").read_text(encoding="utf-8")
        required = {
            "stream_synchronized", "used_bytes_before_free", "total_bytes_before_free",
            "used_bytes_after_free", "total_bytes_after_free",
            "all_per_cell_gpu_arrays_deleted_before_free",
        }

        def ledger_count(candidate: str) -> int:
            tree = ast.parse(candidate)
            runtime = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "runtime_probe")
            count = 0
            for node in ast.walk(runtime):
                if not isinstance(node, ast.Dict):
                    continue
                keys = {key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
                if required <= keys:
                    count += 1
            return count

        self.assertEqual(ledger_count(source), 2)
        for field in sorted(required):
            mutated = source.replace(f'                "{field}":', f'                "removed_{field}":', 1)
            with self.subTest(field=field):
                self.assertEqual(ledger_count(mutated), 1)

    def test_build_panel_drops_every_gpu_alias_before_pool_free(self):
        source = (HERE / "lossy_tail_core.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        panel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_panel")
        panel_source = textwrap.dedent(ast.get_source_segment(source, panel) or "")

        def accepts(candidate: str) -> bool:
            candidate_tree = ast.parse(candidate)
            function = candidate_tree.body[0]
            deleted = {
                node.id
                for delete in ast.walk(function) if isinstance(delete, ast.Delete)
                for node in ast.walk(delete) if isinstance(node, ast.Name)
            }
            required_deleted = {"masks", "x", "words", "pair_masks", "pair_x", "pair_words"}
            return (
                required_deleted <= deleted
                and "cp.cuda.get_current_stream().synchronize()" in candidate
                and "if used_before_free != 0:" in candidate
                and "pool.free_all_blocks()" in candidate
                and "if used_after_free != 0 or total_after_free != 0:" in candidate
                and candidate.index("del masks") < candidate.index("pool.free_all_blocks()")
            )

        self.assertTrue(accepts(panel_source))
        mutations = (
            panel_source.replace("del masks, pair_masks", "del pair_masks", 1),
            panel_source.replace("del x, words", "del words", 1),
            panel_source.replace("del x, words", "del x", 1),
            panel_source.replace("cp.cuda.get_current_stream().synchronize()", "pass", 1),
            panel_source.replace("if used_before_free != 0:", "if False:", 1),
            panel_source.replace("if used_after_free != 0 or total_after_free != 0:", "if False:", 1),
        )
        for ordinal, mutation in enumerate(mutations):
            with self.subTest(mutation=ordinal):
                self.assertFalse(accepts(mutation))

    def test_release_entrypoints_contain_no_assert_statements(self):
        for name in (
            "audit_lock_entrypoint.py", "preflight_launch.py", "runtime_calibrate.py",
            "lossy_tail_oracle.py", "lossy_tail_core.py",
        ):
            source = (HERE / name).read_text(encoding="utf-8")
            tree = ast.parse(source)
            with self.subTest(name=name):
                self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
                self.assertNotIn("AssertionError", source)

    @unittest.skipUnless(Path("/proc/self/mountinfo").is_file(), "Linux mountinfo required")
    def test_full_authorized_preflight_reaches_descriptor_stub_without_payload(self):
        fixture = self.root / "full_fixture"
        stage = fixture / "stage"
        source = fixture / "source"
        evidence = fixture / "evidence"
        auth_dir = evidence / "authorization"
        output_parent = fixture / "output_parent"
        for path in (stage, source, evidence, auth_dir, output_parent):
            path.mkdir(parents=True, exist_ok=True)

        stub = (
            b'import sys\n'
            b'def main():\n'
            b'    context = globals().get("__V8_CORE_CONTEXT__")\n'
            b'    if not isinstance(context, dict) or context.get("mode") != "production_child":\n'
            b'        raise RuntimeError("missing authenticated production-child context")\n'
            b'    if "numpy" in sys.modules or "cupy" in sys.modules:\n'
            b'        raise RuntimeError("third-party module imported before authenticated core boundary")\n'
            b'    print("V8_CAPABILITY_STUB_BOUNDARY payload_files_opened=0 numpy_preimported=0 capability=" + context["capability_sha256"])\n'
        )
        for name in STAGE_MEMBERS:
            if name == "lossy_tail_core.py":
                (stage / name).write_bytes(stub)
            elif name == "source_bindings.json":
                (stage / name).write_text(json.dumps({
                    "source_directory_at_execution": os.fspath(source), "files": []
                }) + "\n")
            elif name not in ("repair_lock.json", "launch_manifest.json"):
                shutil.copyfile(HERE / name, stage / name)

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        identity_names = {
            "scientific_protocol_sha256": "protocol_lock.json",
            "source_bindings_sha256": "source_bindings.json",
            "runtime_contract_sha256": "runtime_contract.json",
            "authorization_contract_sha256": "authorization_contract.json",
            "oracle_bootstrap_sha256": "lossy_tail_oracle.py",
            "scientific_core_sha256": "lossy_tail_core.py",
            "preflight_sha256": "preflight_launch.py",
            "audit_entrypoint_sha256": "audit_lock_entrypoint.py",
            "runtime_calibrate_sha256": "runtime_calibrate.py",
        }
        repair = {
            "schema": "lossy-tail-release-repair-lock-v8",
            "status": "FROZEN_V8_SOURCE_PACKAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION",
            "authenticated_identities": {key: digest(stage / name) for key, name in identity_names.items()},
        }
        repair["repair_lock_sha256"] = hashlib.sha256(PREFLIGHT.canonical_bytes(repair)).hexdigest()
        (stage / "repair_lock.json").write_text(json.dumps(repair, indent=2, sort_keys=True) + "\n")
        members = []
        for name in sorted(STAGE_MEMBERS - {"launch_manifest.json"}):
            path = stage / name
            members.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
        manifest_value = {
            "schema": "lossy-tail-v8-launch-manifest-v1",
            "status": "FROZEN_V8_SOURCE_STAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION",
            "allowed_members": sorted(STAGE_MEMBERS),
            "members": members,
            "source_audit_invocation": "fixture",
            "runtime_calibration_invocation_after_independent_source_pass_only": "fixture",
            "production_invocation_after_independent_runtime_receipt_audit_and_separate_authorization_only": "fixture",
            "production_child_grammar": "fixture",
            "authorization": "NONE_EXISTS; fixture",
        }
        manifest = stage / "launch_manifest.json"
        manifest.write_text(json.dumps(manifest_value, indent=2, sort_keys=True) + "\n")
        manifest_sha = digest(manifest)

        def write_sealed(path: Path, value: dict, field: str) -> tuple[str, str]:
            internal = hashlib.sha256(PREFLIGHT.canonical_bytes(value)).hexdigest()
            value[field] = internal
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            return digest(path), internal

        source_manifest = evidence / "source_audit_manifest.json"
        source_manifest.write_text(json.dumps({
            "schema": PREFLIGHT.SOURCE_AUDIT_MANIFEST_SCHEMA,
            "status": PREFLIGHT.SOURCE_AUDIT_MANIFEST_STATUS,
        }, sort_keys=True) + "\n")
        source_receipt = evidence / "source_audit_receipt.json"
        source_file, source_internal = write_sealed(source_receipt, {
            "schema": PREFLIGHT.SOURCE_AUDIT_RECEIPT_SCHEMA,
            "status": PREFLIGHT.SOURCE_AUDIT_PASS_STATUS,
            "audited_target": {"launch_manifest_sha256": manifest_sha, "repair_lock_internal_sha256": repair["repair_lock_sha256"]},
            "access_ledger": {"model_payload_files_opened": 0, "cupy_imports": 0, "cuda_initializations": 0, "gpu_jobs": 0},
        }, "audit_receipt_sha256")
        runtime_contract_sha = digest(stage / "runtime_contract.json")
        runtime_receipt = evidence / "runtime_receipt.json"
        runtime_tuple = {"fixture": "source-free"}
        runtime_file, runtime_internal = write_sealed(runtime_receipt, {
            "schema": "lossy-tail-v8-source-free-runtime-receipt-v1",
            "status": PREFLIGHT.RUNTIME_RECEIPT_STATUS,
            "runtime_contract": {"sha256": runtime_contract_sha},
            "runtime_probe": {"runtime_tuple": runtime_tuple},
            "access_ledger": {
                "model_or_qwen_paths_supplied": 0, "model_or_qwen_paths_opened": 0,
                "payload_files_opened": 0, "production_results_opened": 0,
            },
        }, "runtime_receipt_sha256")
        runtime_audit_manifest = evidence / "runtime_audit_manifest.json"
        runtime_audit_manifest.write_text(json.dumps({
            "schema": PREFLIGHT.RUNTIME_AUDIT_MANIFEST_SCHEMA,
            "status": PREFLIGHT.RUNTIME_AUDIT_MANIFEST_STATUS,
        }, sort_keys=True) + "\n")
        runtime_audit_receipt = evidence / "runtime_audit_receipt.json"
        runtime_audit_file, runtime_audit_internal = write_sealed(runtime_audit_receipt, {
            "schema": PREFLIGHT.RUNTIME_AUDIT_RECEIPT_SCHEMA,
            "status": PREFLIGHT.RUNTIME_AUDIT_PASS_STATUS,
            "audited_runtime_receipt": {"file_sha256": runtime_file, "internal_sha256": runtime_internal},
            "access_ledger": {"model_payload_files_opened": 0, "production_result_files_opened": 0, "gpu_jobs": 0},
        }, "audit_receipt_sha256")
        authorization_path = auth_dir / "authorization.json"
        mount_payload, mount_rows = PREFLIGHT.mount_snapshot()
        live_paths = {
            "stage": stage, "source": source, "output_existing_parent": output_parent,
            "authorization_parent": auth_dir, "source_audit_manifest": source_manifest,
            "source_audit_receipt": source_receipt, "runtime_receipt": runtime_receipt,
            "runtime_audit_manifest": runtime_audit_manifest, "runtime_audit_receipt": runtime_audit_receipt,
        }
        identity_rows = []
        for label, path in live_paths.items():
            metadata = os.stat(path, follow_symlinks=False)
            identity_rows.append({
                "label": label, "path": os.fspath(path), "st_dev": metadata.st_dev,
                "st_ino": metadata.st_ino, "mount_id": PREFLIGHT.mount_for(path, mount_rows)["mount_id"],
            })
        output_root = output_parent / "one_shot"
        authorization = {
            "schema": "lossy-tail-v8-one-shot-production-authorization-v1",
            "status": "AUTHORIZED_ONCE_AFTER_INDEPENDENT_SOURCE_AND_RUNTIME_AUDITS",
            "authorization_path": os.fspath(authorization_path),
            "authorization_nonce": "SOURCE_FREE_TEST_STUB_ONLY",
            "action": "CREATE_NEW_RUN_ROOT_AND_RESULT_JSON",
            "stage": {"path": os.fspath(stage), "launch_manifest_file_sha256": manifest_sha, "launch_manifest_internal_stage_member_count": len(STAGE_MEMBERS)},
            "source": {"path": os.fspath(source), "bindings_file_sha256": digest(stage / "source_bindings.json")},
            "output": {"run_root": os.fspath(output_root), "result_path": os.fspath(output_root / "result.json")},
            "source_audit": {
                "manifest_path": os.fspath(source_manifest), "manifest_file_sha256": digest(source_manifest),
                "receipt_path": os.fspath(source_receipt), "receipt_file_sha256": source_file,
                "receipt_internal_field": "audit_receipt_sha256", "receipt_internal_sha256": source_internal,
            },
            "runtime_receipt": {
                "path": os.fspath(runtime_receipt), "file_sha256": runtime_file,
                "internal_sha256": runtime_internal,
                "runtime_contract_file_sha256": runtime_contract_sha,
            },
            "runtime_audit": {
                "manifest_path": os.fspath(runtime_audit_manifest), "manifest_file_sha256": digest(runtime_audit_manifest),
                "receipt_path": os.fspath(runtime_audit_receipt), "receipt_file_sha256": runtime_audit_file,
                "receipt_internal_field": "audit_receipt_sha256", "receipt_internal_sha256": runtime_audit_internal,
            },
            "execution": {
                "python_executable": sys.executable, "raw_launcher_path": os.fspath(stage / "preflight_launch.py"),
                "cuda_visible_devices": "0", "runtime_tuple": runtime_tuple,
            },
            "filesystem": {
                "mountinfo_path": "/proc/self/mountinfo",
                "mountinfo_file_sha256": hashlib.sha256(mount_payload).hexdigest(),
                "identities": identity_rows,
            },
            "fixed_scientific_arguments": {"control_replicates": 4, "maximum_coordinate_passes": 4},
        }
        auth_internal = hashlib.sha256(PREFLIGHT.canonical_bytes(authorization)).hexdigest()
        authorization["authorization_sha256"] = auth_internal
        authorization_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")
        authorization_file = digest(authorization_path)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        command = [
            sys.executable, "-B", "-I", os.fspath(stage / "preflight_launch.py"),
            "--manifest", os.fspath(manifest), "--manifest-sha256", manifest_sha,
            "--authorization", os.fspath(authorization_path), "--authorization-sha256", authorization_file,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS_AUTHORIZATION_BOUND_BEFORE_CHILD_CAPABILITY", result.stdout)
        self.assertIn("V8_CAPABILITY_STUB_BOUNDARY payload_files_opened=0 numpy_preimported=0", result.stdout)
        self.assertIn("EXITED_SUCCESS_AFTER_ONE_USE_CAPABILITY", result.stdout)
        self.assertEqual(result.stdout.count("V8_CAPABILITY_STUB_BOUNDARY"), 1)
        self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
