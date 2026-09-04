"""Fail-closed verifier for the TETRAPATH-4 independent hostile audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "tetrapath4_source_oracle_v0_20260904"
EXPECTED_CORE = "b303d9d87659d0ae36687fed9ab82b00e1eea8a6bd94ea4769453e42b5fb611a"
EXPECTED_VERDICT = (
    "FAIL_AS_HARD_KILL_OR_PROMOTION_ORACLE__PASS_SOURCE_ONLY_MECHANISM_FIXTURE")
EXPECTED_FILES = {
    "AUDIT_EVIDENCE.json", "AUDIT_MANIFEST.json", "AUDIT_REPORT.md",
    "AUDIT_VERDICT.json", "README.md", "hostile_audit.py", "verify_audit.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    names = {path.name for path in HERE.iterdir() if path.is_file()}
    if names != EXPECTED_FILES:
        raise SystemExit(f"FAIL audit closure: {sorted(names)}")
    manifest = json.loads((HERE / "AUDIT_MANIFEST.json").read_text(encoding="utf-8"))
    if set(manifest.get("files_sha256", {})) != EXPECTED_FILES - {"AUDIT_MANIFEST.json"}:
        raise SystemExit("FAIL manifest membership")
    for name, expected in manifest["files_sha256"].items():
        if sha256(HERE / name) != expected:
            raise SystemExit(f"FAIL audit hash: {name}")
    if sha256(TARGET / "tetrapath4_oracle.py") != EXPECTED_CORE:
        raise SystemExit("FAIL target core changed")
    verdict = json.loads((HERE / "AUDIT_VERDICT.json").read_text(encoding="utf-8"))
    if verdict.get("verdict") != EXPECTED_VERDICT:
        raise SystemExit("FAIL verdict changed")
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(HERE / "hostile_audit.py")],
        cwd=HERE, check=False)
    if completed.returncode:
        raise SystemExit("FAIL hostile audit evidence")
    evidence = json.loads((HERE / "AUDIT_EVIDENCE.json").read_text(encoding="utf-8"))
    if evidence["local_search_globality_counterexample"] is None:
        raise SystemExit("FAIL missing local-search counterexample")
    if evidence["smoothing_noncontainment_counterexample"][
            "full_minus_independent_bpw"] <= 0:
        raise SystemExit("FAIL missing smoothing counterexample")
    print(EXPECTED_VERDICT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
