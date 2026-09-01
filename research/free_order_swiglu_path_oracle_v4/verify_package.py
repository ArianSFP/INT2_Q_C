"""Verify the sealed, source-only FOSP-v4 native-launch contract package."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import stat
import sys
import unittest
from pathlib import Path
from typing import Any


V3_ORACLE_SHA256 = "9ca6f4bdd4150c8c0c68c0a298c00eb45c088a4af287895ebfdf9bf1e661a070"
V3_PROTOCOL_SHA256 = "f4660cb8876a749eb1635dbf010a8df6199e845b0517dd8b15039ac9cf1fd097"
EXPECTED_TESTS = 18


class VerificationFailure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, label: str) -> None:
        if not condition:
            raise VerificationFailure(label)
        self.count += 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8", errors="strict"))


def direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".", 1)[0])
    return result


def verify(package_root: Path) -> dict[str, Any]:
    checks = Checks()
    root = package_root.resolve(strict=True)
    manifest = load_json(root / "PACKAGE_MANIFEST.json")
    checks.require(manifest["schema"] == "free_order_swiglu_path_v4_package_manifest_v1", "manifest schema")
    checks.require(manifest["status"] == "SEALED_INERT_SOURCE_ONLY_NO_AUTHORITY", "manifest status")
    checks.require(manifest["authorization"] == "NONE", "manifest authorization")
    rows = manifest["artifacts"]
    checks.require(manifest["artifact_count"] == len(rows), "manifest count")
    paths = [row["path"] for row in rows]
    checks.require(paths == sorted(paths) and len(paths) == len(set(paths)), "manifest sorted unique paths")
    expected_names = sorted(paths + ["PACKAGE_MANIFEST.json"])
    observed_names = sorted(entry.name for entry in os.scandir(root))
    checks.require(observed_names == expected_names, "exact directory closure")
    for row in rows:
        checks.require(set(row) == {"path", "bytes", "sha256"}, "manifest row closure")
        path = root / row["path"]
        info = path.lstat()
        checks.require(path.name == row["path"], "flat package member")
        checks.require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "regular nonlink package member")
        checks.require(info.st_nlink == 1, "single-link package member")
        checks.require(info.st_size == row["bytes"], "package member size")
        checks.require(sha256_file(path) == row["sha256"], "package member hash")

    checks.require(sha256_file(root / "scientific_oracle_v3.py") == V3_ORACLE_SHA256, "exact v3 oracle bytes")
    checks.require(sha256_file(root / "scientific_protocol_v3.json") == V3_PROTOCOL_SHA256, "exact v3 protocol bytes")
    v3_protocol = load_json(root / "scientific_protocol_v3.json")
    contract_raw = (root / "native_launcher_contract.json").read_bytes()
    contract_json = load_json(root / "native_launcher_contract.json")
    checks.require(contract_json["lineage"]["v3_independent_audit_trust_input"] == "NONE_PROVIDED_OR_CLAIMED", "no unfinished v3 audit trust claim")
    checks.require(contract_json["frozen_science"]["stage_order"] == v3_protocol["scientific_gate"]["stage_order"], "exact v3 stage order")
    checks.require(contract_json["frozen_science"]["total_side_bits"] == v3_protocol["eligible_physical_bridge"]["total_side_bits"], "exact v3 side bits")
    checks.require(contract_json["frozen_science"]["total_side_bpw"] == v3_protocol["eligible_physical_bridge"]["total_side_bpw"], "exact v3 side bpw")
    checks.require(contract_json["frozen_science"]["required_net_s_bpw"] == v3_protocol["objective"]["required_net_s_bpw"], "exact v3 net rate")
    checks.require(contract_json["frozen_science"]["required_gross_s_bpw"] == v3_protocol["objective"]["required_gross_s_after_side_bpw"], "exact v3 gross rate")
    checks.require(contract_json["frozen_science"]["n8_regression"] == v3_protocol["scientific_gate"]["n8_adversarial_regression"], "exact v3 n=8 regression")
    checks.require(contract_json["frozen_science"]["scientific_semantics_changed_from_v3"] is False, "science unchanged flag")
    checks.require(all(value == 0 for value in contract_json["zero_access"].values()), "zero-access ledger")
    checks.require(all(value is False for value in contract_json["authorization"].values()), "contract authorization firewall")

    allowed = {"__future__", "hashlib", "json", "re", "pathlib", "typing"}
    checks.require(direct_imports(root / "launch_contract.py") <= allowed, "launch validator standard-library import closure")
    launch_source = (root / "launch_contract.py").read_text(encoding="utf-8")
    for forbidden in (
        "import numpy",
        "import cupy",
        "import torch",
        "import socket",
        "import subprocess",
        "import ctypes",
        "import requests",
        "import urllib",
        "execve",
        "createprocess",
    ):
        checks.require(forbidden not in launch_source.lower(), "forbidden validator operation absent: " + forbidden)
    checks.require("AUTHORITY_GRANTED = False" in launch_source, "validator source-default authority false")
    checks.require("ordinary mutable venv forbidden" in launch_source, "mutable venv rejection")
    checks.require("bootstrap substitution before execution" in launch_source, "bootstrap substitution rejection")
    checks.require("before Python startup" in launch_source, "pre-start runtime gate")
    checks.require("return 78" in launch_source, "validator direct invocation refusal")

    readme = (root / "README.md").read_text(encoding="utf-8")
    checks.require("/srv/fosp-v4-sealed/bin/fosp4-native-launcher" in readme, "regular native launcher invocation")
    checks.require(re.search(r"(^|\s)python\s+-", readme, flags=re.MULTILINE) is None, "no generic python invocation")
    checks.require("ordinary mutable virtual environment does **not** satisfy" in readme, "mutable venv documentation")
    checks.require("before creating a Python process" in readme, "pre-start authentication documentation")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import launch_contract
    import test_source_only

    contract = launch_contract.validate_contract(
        contract_raw,
        (root / "bootstrap_v4.py").read_bytes(),
        (root / "scientific_oracle_v3.py").read_bytes(),
        (root / "scientific_protocol_v3.json").read_bytes(),
    )
    checks.require(all(value is False for value in contract["authorization"].values()), "runtime contract validation authority false")
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_source_only)
    checks.require(suite.countTestCases() == EXPECTED_TESTS, "exact hostile test count")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    checks.require(result.testsRun == EXPECTED_TESTS, "executed hostile test count")
    checks.require(result.wasSuccessful(), "hostile tests\n" + stream.getvalue())
    checks.require(sorted(entry.name for entry in os.scandir(root)) == expected_names, "post-test exact closure")

    receipt = load_json(root / "PRODUCER_RECEIPT.json")
    checks.require(receipt["status"] == "SEALED_SOURCE_ONLY_NATIVE_PRESTART_REPAIR_AUDIT_REQUIRED", "producer status")
    checks.require(all(value is False for value in receipt["authorization"].values()), "producer authorization firewall")
    return {
        "schema": "free_order_swiglu_path_v4_package_verification_v1",
        "status": "PASS_SOURCE_ONLY_NATIVE_PRESTART_CONTRACT_AUTHORIZES_NOTHING",
        "checks": checks.count,
        "hostile_tests": result.testsRun,
        "v3_oracle_sha256": V3_ORACLE_SHA256,
        "v3_protocol_sha256": V3_PROTOCOL_SHA256,
        "model_or_qwen_access": 0,
        "gpu_operations": 0,
        "network_operations": 0,
        "authorizations_issued": 0,
        "authorization": "NONE",
    }


if __name__ == "__main__":
    try:
        outcome = verify(Path(__file__).resolve().parent)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True, separators=(",", ":")))
        raise SystemExit(1)
    print(json.dumps(outcome, sort_keys=True, separators=(",", ":")))
