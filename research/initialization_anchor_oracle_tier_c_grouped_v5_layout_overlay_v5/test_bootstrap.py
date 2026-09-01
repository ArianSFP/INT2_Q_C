"""Adversarial regressions for the clean v5 package bootstrap."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import verify_prelaunch


PACKAGE = Path(__file__).resolve().parent


class CleanBootstrapTests(unittest.TestCase):
    def _copy_closed_package(self, destination: Path) -> Path:
        destination.mkdir()
        for name in sorted(verify_prelaunch.EXPECTED_PACKAGE_MEMBERS):
            shutil.copy2(PACKAGE / name, destination / name)
        return destination

    def _run(self, package: Path, *flags: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *flags, str(package / "verify_prelaunch.py"), "--auth-only"],
            cwd=package,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_exact_closed_package_authenticates_without_scientific_imports(self):
        before = set(sys.modules)
        receipt = verify_prelaunch.authenticate_package_before_imports(
            require_runtime_firewall=False
        )
        self.assertEqual(receipt["artifact_count"], 16)
        self.assertEqual(set(sys.modules), before)

    def test_unexpected_regular_member_and_pycache_directory_fail_closed(self):
        for unexpected, directory in (("surprise.txt", False), ("__pycache__", True)):
            with self.subTest(unexpected=unexpected), tempfile.TemporaryDirectory() as temp:
                fixture = self._copy_closed_package(Path(temp) / "fixture")
                target = fixture / unexpected
                if directory:
                    target.mkdir()
                else:
                    target.write_text("unexpected\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    verify_prelaunch.BootstrapError, "directory closure mismatch"
                ):
                    verify_prelaunch.authenticate_package_before_imports(
                        fixture, require_runtime_firewall=False
                    )

    def test_missing_manifested_member_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._copy_closed_package(Path(temp) / "fixture")
            (fixture / "kernels.py").unlink()
            with self.assertRaisesRegex(
                verify_prelaunch.BootstrapError, "directory closure mismatch"
            ):
                verify_prelaunch.authenticate_package_before_imports(
                    fixture, require_runtime_firewall=False
                )

    def test_manifest_rows_must_be_strictly_sorted(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._copy_closed_package(Path(temp) / "fixture")
            manifest = fixture / verify_prelaunch.MANIFEST_BASENAME
            lines = manifest.read_text(encoding="ascii").splitlines(True)
            manifest.write_text("".join(reversed(lines)), encoding="ascii")
            with self.assertRaisesRegex(
                verify_prelaunch.BootstrapError, "strictly sorted"
            ):
                verify_prelaunch.authenticate_package_before_imports(
                    fixture, require_runtime_firewall=False
                )

    def test_source_hash_tamper_never_executes_import_tripwire(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self._copy_closed_package(root / "fixture")
            tripwire = root / "IMPORT_TRIPWIRE"
            (fixture / "common.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(tripwire)!r}).write_text('IMPORTED', encoding='utf-8')\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                verify_prelaunch.BootstrapError,
                "artifact hash mismatch: common.py",
            ):
                verify_prelaunch.authenticate_package_before_imports(
                    fixture, require_runtime_firewall=False
                )
            self.assertFalse(tripwire.exists())

    def test_raw_relative_dotdot_and_symlink_entrypoint_spellings_fail_closed(self):
        cases = [
            [sys.executable, "-B", "-I", "verify_prelaunch.py", "--auth-only"],
            [
                sys.executable, "-B", "-I",
                str(PACKAGE / ".." / PACKAGE.name / "verify_prelaunch.py"),
                "--auth-only",
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "copied-verifier.py"
            shutil.copy2(PACKAGE / "verify_prelaunch.py", copied)
            cases.append(
                [sys.executable, "-B", "-I", str(copied), "--auth-only"]
            )
            hardlink = Path(directory) / "hardlink-verifier.py"
            try:
                hardlink.hardlink_to(PACKAGE / "verify_prelaunch.py")
            except (OSError, NotImplementedError):
                hardlink = None
            if hardlink is not None:
                cases.append(
                    [sys.executable, "-B", "-I", str(hardlink), "--auth-only"]
                )
            alias = Path(directory) / "verifier-alias.py"
            try:
                alias.symlink_to(PACKAGE / "verify_prelaunch.py")
            except (OSError, NotImplementedError):
                alias = None
            if alias is not None:
                cases.append(
                    [sys.executable, "-B", "-I", str(alias), "--auth-only"]
                )
            for command in cases:
                with self.subTest(command=command):
                    result = subprocess.run(
                        command,
                        cwd=PACKAGE,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertRegex(
                        result.stderr,
                        "raw argv0 (must be absolute|must use one canonical|differs from the frozen)",
                    )

    def test_isolation_and_bytecode_flags_are_independently_mandatory(self):
        without_isolation = self._run(PACKAGE, "-B")
        self.assertNotEqual(without_isolation.returncode, 0)
        self.assertIn("requires isolated Python", without_isolation.stderr)
        without_bytecode_firewall = self._run(PACKAGE, "-I")
        self.assertNotEqual(without_bytecode_firewall.returncode, 0)
        self.assertIn("requires bytecode disabled", without_bytecode_firewall.stderr)

    def test_isolated_bytecode_free_cli_auth_succeeds_without_cache_creation(self):
        result = self._run(PACKAGE, "-B", "-I")
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(
            receipt["status"],
            "PASS_EXACT_CLOSED_PACKAGE_AUTHENTICATED_BEFORE_SCIENTIFIC_IMPORTS",
        )
        raw = receipt["raw_entrypoint_identity"]
        self.assertEqual(raw["raw_argv0"], str(PACKAGE / "verify_prelaunch.py"))
        self.assertEqual(raw["mount"]["mount_point"], "/workspace")
        self.assertGreater(raw["component_count"], 1)
        self.assertFalse((PACKAGE / "__pycache__").exists())

    def test_scientific_runner_refuses_direct_execution_before_package_imports(self):
        result = subprocess.run(
            [sys.executable, "-B", "-I", str(PACKAGE / "tier_c_gate.py"), "preflight"],
            cwd=PACKAGE,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("direct execution is forbidden", result.stderr)
        self.assertFalse((PACKAGE / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main()
