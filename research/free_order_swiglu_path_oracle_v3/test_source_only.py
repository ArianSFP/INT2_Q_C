#!/usr/bin/env python3
"""Pure-standard-library regression suite for source-only FOSP-v3."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORACLE_PATH = ROOT / "free_order_oracle_v3.py"
BOOTSTRAP_PATH = ROOT / "bootstrap_v3.py"
MANIFEST_PATH = ROOT / "ARTIFACT_SHA256SUMS.txt"
LOCK_PATH = ROOT / "protocol_lock.json"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_oracle():
    spec = importlib.util.spec_from_file_location("fosp_v3_source_only_tests", ORACLE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load v3 oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def matmul(left, right):
    columns = transpose(right)
    return [[dot(row, column) for column in columns] for row in left]


def inverse(matrix):
    n = len(matrix)
    work = [list(row) + [Fraction(int(i == j)) for j in range(n)]
            for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = next(row for row in range(column, n) if work[row][column] != 0)
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(n):
            if row == column:
                continue
            scale = work[row][column]
            work[row] = [a - scale * b for a, b in zip(work[row], work[column])]
    return [row[n:] for row in work]


class SourceOnlyV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = load_oracle()
        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        cls.manifest_sha = sha256(MANIFEST_PATH.read_bytes())

    def test_full_cross_role_fraction_regression_identity(self):
        f = Fraction
        x = [[f(1), f(0), f(1), f(2), f(0)],
             [f(0), f(2), f(1), f(1), f(1)],
             [f(1), f(1), f(0), f(1), f(2)]]
        y = [[f(2), f(1), f(0), f(1), f(0)],
             [f(0), f(1), f(3), f(0), f(2)],
             [f(1), f(0), f(1), f(1), f(3)]]
        gram = matmul(x, transpose(x))
        cross = matmul(y, transpose(x))
        coefficients = matmul(cross, inverse(gram))
        predicted = matmul(coefficients, x)
        residual = [[a - b for a, b in zip(yr, pr)] for yr, pr in zip(y, predicted)]
        energy = sum((v * v for row in y for v in row), f(0))
        residual_energy = sum((v * v for row in residual for v in row), f(0))
        capture = sum((cross[i][j] * coefficients[i][j] for i in range(3) for j in range(3)), f(0))
        trace_form = matmul(matmul(cross, inverse(gram)), transpose(cross))
        self.assertEqual(capture, sum(trace_form[i][i] for i in range(3)))
        self.assertEqual(residual_energy, energy - capture)

    def test_science_source_retains_all_3x3_roles(self):
        source = ORACLE_PATH.read_text(encoding="utf-8")
        for fragment in (
            'gram = cp.einsum("nrd,nsd->nrs", expert, expert)',
            "cross[:, :, target_role, predecessor_role]",
            'full = cp.einsum("ijab,jbc,ijac->ij", cross, inverse, cross)',
            'exact_coefficients = cp.einsum("eab,ebc->eac", selected_cross, inverse[predecessors])',
            'predicted = cp.einsum("eab,ebd->ead", replay_coefficients, expert[predecessors])',
        ):
            self.assertIn(fragment, source)
        self.assertEqual(self.oracle.ROLES, 3)
        self.assertEqual(self.oracle.FP16_COEFFICIENTS_PER_EDGE, 9)

    def test_gross_relaxed_contains_all_small_legal_paths(self):
        for n in range(2, 8):
            scores = [[Fraction(((target + 1) * 11 + (pred + 1) * 7) % 23 + 1, 23)
                       for pred in range(n)] for target in range(n)]
            relaxed = sum((max(scores[target][pred] for pred in range(n) if pred != target)
                           for target in range(n)), Fraction(0))
            for path in itertools.permutations(range(n)):
                legal = sum((scores[target][pred] for pred, target in zip(path[:-1], path[1:])),
                            Fraction(0))
                self.assertLessEqual(legal, relaxed)

    def test_cycle_cover_direction_and_bridge(self):
        class TinyMatrix:
            def __init__(self, rows):
                self.rows = rows
                self.shape = (len(rows), len(rows))

            def __getitem__(self, key):
                return self.rows[key[0]][key[1]]

            def __neg__(self):
                return TinyMatrix([[-v for v in row] for row in self.rows])

        class TinyNP:
            float64 = float

            @staticmethod
            def asarray(value, dtype=None):
                del dtype
                return value

            @staticmethod
            def arange(n):
                return list(range(n))

            @staticmethod
            def array_equal(left, right):
                return list(left) == list(right)

        predecessor = [2, 0, 1, 4, 3]
        scores = [[0.0] * 5 for _ in range(5)]
        scores[0][2], scores[1][0], scores[2][1] = 1.0, 9.0, 8.0
        scores[3][4], scores[4][3], scores[4][2] = 7.0, 2.0, 6.0

        def assignment(_):
            return list(range(5)), predecessor

        result = self.oracle._legal_path_from_cycle_cover(TinyMatrix(scores), TinyNP, assignment)
        self.assertEqual(result["path"], [0, 1, 2, 4, 3])
        self.assertEqual(result["legal_path_capture"], 30.0)
        self.assertEqual([(r["predecessor"], r["target"]) for r in result["dropped_edges"]],
                         [(2, 0), (3, 4)])

    def test_exact_n8_corrected_relaxed_noncontainment(self):
        row = self.oracle.adversarial_n8_statistics()
        self.assertEqual(struct.unpack("<e", struct.pack("<e", row["r"]))[0], 0.875)
        self.assertEqual(struct.unpack("<e", struct.pack("<e", row["rho"]))[0], 0.765625)
        self.assertEqual(row["corrected_relaxed_s_bpw"], 0.0)
        self.assertAlmostEqual(row["qwen_legal_fp16_s_bpw"], 0.7995602818589078, places=15)
        self.assertAlmostEqual(row["control_legal_fp16_s_bpw"], 0.5885652320580218, places=15)
        self.assertAlmostEqual(row["corrected_legal_fp16_s_bpw"], 0.21099504980088601, places=15)
        self.assertGreater(row["corrected_legal_fp16_s_bpw"], self.oracle.REQUIRED_GROSS_S)

        n, r, rho = 8, Fraction(7, 8), Fraction(49, 64)
        total = Fraction(3 * n)
        self.assertEqual(Fraction(3 * n) * rho, Fraction(3 * n) * rho)
        self.assertEqual(total - Fraction(3 * (n - 1)) * rho,
                         Fraction(3) + Fraction(3 * (n - 1)) * (1 - rho))
        self.assertEqual(Fraction(r), Fraction(7, 8))

    def test_corrected_relaxed_is_never_decision_eligible(self):
        source = ORACLE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("HARD_KILL_CONTROL_CORRECTED_RELAXED", source)
        self.assertIn('statistics["relaxed_reuse_exact"]["decision_eligible"] = False', source)
        self.assertIn('"legal_path_fp16": _controlled_statistic', source)
        gate = self.lock["scientific_gate"]
        self.assertFalse(gate["corrected_relaxed_reuse"]["containing"])
        self.assertFalse(gate["corrected_relaxed_reuse"]["decision_eligible"])

    def test_control_decision_requires_legal_fp16(self):
        with self.assertRaisesRegex(self.oracle.ProtocolError, "legal FP16"):
            self.oracle._decision_after_legal_statistics({"legal_path_fp16": {"s_bpw": 1.0}}, {})
        qwen = {"legal_path_fp16": {"s_bpw": 1.0}}
        statistics = {
            "legal_path_fp16": {"upper_confidence_survives_target": True},
            "relaxed_reuse_exact": {"upper_confidence_survives_target": False},
        }
        self.assertEqual(
            self.oracle._decision_after_legal_statistics(qwen, statistics),
            "SURVIVE_SOURCE_ORACLE_FP16_PATH_RESIDUAL_CODEC_REQUIRED",
        )

    def test_factoradic_and_rate_ledgers_unchanged(self):
        self.assertEqual(self.oracle.ceil_log2_factorial(768), 6260)
        reverse = tuple(reversed(range(768)))
        encoded = self.oracle.serialize_permutation(reverse)
        self.assertEqual(len(encoded), 783)
        self.assertEqual(encoded[0] >> 4, 0)
        self.assertEqual(self.oracle.TOTAL_SIDE_BITS, 117224)
        self.assertAlmostEqual(self.oracle.REQUIRED_GROSS_S, 0.1858070514584381, places=15)
        self.assertLess(max(self.oracle.frame_ledger(rate)["cold_page_amplification"]
                            for rate in self.oracle.RATES), 2.0)

    def test_source_default_has_no_access_or_authorization(self):
        status = self.oracle.source_only_status()
        self.assertEqual(status["status"], "SOURCE_ONLY_DEPLOYMENT_BLOCKED")
        self.assertFalse(status["source_access_authorized"])
        self.assertFalse(status["calibration_authorized"])
        self.assertFalse(self.lock["execution"]["source_access_authorized"])
        self.assertFalse(self.lock["execution"]["calibration_authorized"])

    def test_no_heavy_top_level_imports(self):
        tree = ast.parse(ORACLE_PATH.read_text(encoding="utf-8"), filename=ORACLE_PATH.name)
        roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        self.assertFalse({"cupy", "numpy", "scipy", "torch", "transformers", "cuda"} & roots)

    def _bootstrap(self, package, *extra):
        command = [
            sys.executable,
            "-I",
            "-S",
            os.fspath(package / "bootstrap_v3.py"),
            "--package-manifest-sha256",
            sha256((package / "ARTIFACT_SHA256SUMS.txt").read_bytes()),
            *extra,
        ]
        environment = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                       "WINDIR": os.environ.get("WINDIR", "")}
        return subprocess.run(command, cwd=package, env=environment, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)

    def _runtime_fixture(self, scratch):
        root = scratch / "runtime"
        library = root / "lib"
        library.mkdir(parents=True)
        dependency = library / "fosp_runtime_probe_dependency.py"
        dependency.write_bytes(b'VALUE = "SEALED_RUNTIME_SOURCE_PASS"\n')
        raw = dependency.read_bytes()
        manifest = scratch / "runtime_manifest.txt"
        manifest.write_text(
            "FOSP_RUNTIME_CLOSURE_V1\n"
            "D  lib\n"
            "I  lib\n"
            f"F  {sha256(raw)}  {len(raw)}  lib/fosp_runtime_probe_dependency.py\n",
            encoding="ascii",
            newline="\n",
        )
        return root, manifest

    def test_bootstrap_accepts_exact_flat_snapshot(self):
        completed = self._bootstrap(ROOT, "--verify-package")
        self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
        self.assertIn(b"FOSP_V3_PACKAGE_SNAPSHOT_PASS", completed.stdout)

    def test_sealed_runtime_source_imports_from_authenticated_bytes(self):
        with tempfile.TemporaryDirectory(prefix="fosp_v3_runtime_source_") as text:
            scratch = Path(text)
            root, manifest = self._runtime_fixture(scratch)
            completed = self._bootstrap(
                ROOT,
                "--runtime-root", os.fspath(root.resolve()),
                "--runtime-manifest", os.fspath(manifest.resolve()),
                "--runtime-manifest-sha256", sha256(manifest.read_bytes()),
                "--python-sha256", sha256(Path(sys.executable).read_bytes()),
                "--entrypoint", "sealed_runtime_probe.py",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode(errors="replace"))
            self.assertEqual(completed.stdout.strip(), b"SEALED_RUNTIME_SOURCE_PASS")

    def test_runtime_directory_injection_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="fosp_v3_runtime_injection_") as text:
            scratch = Path(text)
            root, manifest = self._runtime_fixture(scratch)
            injected = root / "lib" / "json"
            injected.mkdir()
            (injected / "__init__.py").write_text("raise RuntimeError('injected')\n", encoding="utf-8")
            completed = self._bootstrap(
                ROOT,
                "--runtime-root", os.fspath(root.resolve()),
                "--runtime-manifest", os.fspath(manifest.resolve()),
                "--runtime-manifest-sha256", sha256(manifest.read_bytes()),
                "--python-sha256", sha256(Path(sys.executable).read_bytes()),
                "--entrypoint", "sealed_runtime_probe.py",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"runtime exact object closure mismatch", completed.stderr)

    def test_directory_import_injection_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory(prefix="fosp_v3_directory_injection_") as text:
            scratch = Path(text)
            package = scratch / "package"
            shutil.copytree(ROOT, package)
            injected = package / "json"
            injected.mkdir()
            sentinel = scratch / "injected-code-executed"
            (injected / "__init__.py").write_text(
                "open(" + repr(os.fspath(sentinel)) + ", 'wb').write(b'EXECUTED')\n",
                encoding="utf-8",
            )
            completed = self._bootstrap(package, "--verify-package")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"nonregular package member forbidden: json", completed.stderr)
            self.assertFalse(sentinel.exists())

    def test_symlink_member_is_rejected_when_supported(self):
        with tempfile.TemporaryDirectory(prefix="fosp_v3_symlink_") as text:
            scratch = Path(text)
            package = scratch / "package"
            shutil.copytree(ROOT, package)
            link = package / "unsealed-link"
            try:
                link.symlink_to(package / "README.md")
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            completed = self._bootstrap(package, "--verify-package")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"nonregular package member forbidden", completed.stderr)

    def test_bootstrap_rejects_unisolated_launch(self):
        completed = subprocess.run(
            [sys.executable, "-S", os.fspath(BOOTSTRAP_PATH),
             "--package-manifest-sha256", self.manifest_sha, "--verify-package"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(b"require Python -I -S", completed.stderr)

    def test_manifest_digest_is_external_trust_input(self):
        command = [sys.executable, "-I", "-S", os.fspath(BOOTSTRAP_PATH),
                   "--package-manifest-sha256", "0" * 64, "--verify-package"]
        completed = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=30, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(b"externally pinned package-manifest hash mismatch", completed.stderr)

    def test_bootstrap_has_no_filesystem_import_before_closure(self):
        tree = ast.parse(BOOTSTRAP_PATH.read_text(encoding="utf-8"), filename=BOOTSTRAP_PATH.name)
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(len(imports), 1)
        self.assertIsInstance(imports[0], ast.Import)
        self.assertEqual(imports[0].names[0].name, "sys")
        source = BOOTSTRAP_PATH.read_text(encoding="utf-8")
        self.assertLess(source.index("_sys.path[:] = []"), source.index("def _snapshot_package"))
        self.assertLess(source.index("snapshot = _snapshot_package"),
                        source.index("import_roots, runtime_snapshot = _verify_runtime("))
        self.assertIn('_sys.meta_path[:] = [frozen.BuiltinImporter, frozen.FrozenImporter, sealed_finder]',
                      source)
        self.assertIn('source = self.snapshot[relative].decode("utf-8")', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
