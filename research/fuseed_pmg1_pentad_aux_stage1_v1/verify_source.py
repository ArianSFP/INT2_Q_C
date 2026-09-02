#!/usr/bin/env python3
"""Fail-closed verifier for the fixed-pentad source-only package."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent
EXPECTED_MEMBERS = {
    "README.md",
    "SOURCE_MANIFEST.json",
    "__init__.py",
    "contract.py",
    "cupy_anchor.py",
    "design_lock.json",
    "inert_entrypoint.py",
    "plan_snapshot.py",
    "reference_bindings.json",
    "stage1_core.py",
    "test_source.py",
    "verify_source.py",
}
RUNTIME_FILES = {
    "__init__.py",
    "contract.py",
    "cupy_anchor.py",
    "inert_entrypoint.py",
    "plan_snapshot.py",
    "stage1_core.py",
}
FORBIDDEN_RUNTIME_IMPORTS = {
    "argparse",
    "http",
    "os",
    "paramiko",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "torch",
    "urllib",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def strict_json(path: Path):
    def reject_constant(value: str):
        raise RuntimeError(f"nonfinite JSON constant in {path.name}: {value}")

    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, f"duplicate key in {path.name}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    return value


def verify_closure() -> dict:
    observed = {path.name for path in ROOT.iterdir()}
    require(observed == EXPECTED_MEMBERS, f"source closure mismatch: {sorted(observed ^ EXPECTED_MEMBERS)}")
    for path in ROOT.iterdir():
        require(path.is_file() and not path.is_symlink(), f"non-regular member: {path.name}")
    manifest = strict_json(ROOT / "SOURCE_MANIFEST.json")
    require(manifest["schema"] == "fuseed-pmg1-fixed-pentad-aux-stage1-source-manifest-v1", "manifest schema")
    require(manifest["self_unlisted"] == "SOURCE_MANIFEST.json", "manifest self rule")
    rows = manifest["files"]
    require({row["path"] for row in rows} == EXPECTED_MEMBERS - {"SOURCE_MANIFEST.json"}, "manifest member closure")
    for row in rows:
        path = ROOT / row["path"]
        require(path.stat().st_size == int(row["bytes"]), f"byte mismatch: {row['path']}")
        require(sha256_file(path) == row["sha256"], f"hash mismatch: {row['path']}")
    return manifest


def verify_ast() -> int:
    checks = 0
    for name in sorted(RUNTIME_FILES):
        source = (ROOT / name).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=name)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                require(node.func.id not in {"open", "exec", "eval", "compile", "__import__"}, f"forbidden runtime call {node.func.id}: {name}")
            checks += 1
        require(not (imports & FORBIDDEN_RUNTIME_IMPORTS), f"forbidden runtime import in {name}: {sorted(imports & FORBIDDEN_RUNTIME_IMPORTS)}")
        if name not in {"inert_entrypoint.py", "plan_snapshot.py"}:
            require("if __name__" not in source, f"unexpected runtime entry point: {name}")
        require(".bf16.bin" not in source and "/workspace/" not in source, f"payload path in runtime: {name}")
    return checks


def verify_design() -> int:
    design = strict_json(ROOT / "design_lock.json")
    bindings = strict_json(ROOT / "reference_bindings.json")
    require(design["status"] == "FROZEN_SOURCE_ONLY_AWAITING_INDEPENDENT_AUDIT_NO_PAYLOAD_AUTHORITY", "design status")
    require(design["fixed_hypothesis"]["seeds_u32_in_decoder_order"] == [3306464084, 235286348, 2174751347, 256779041, 118211936], "seed tuple")
    require(design["frozen_coordinate_plan"]["fit_coordinates"] == 2048, "fit count")
    require(design["frozen_coordinate_plan"]["score_coordinates"] == 2048, "score count")
    require(math.isclose(design["decision"]["required_capture"], 0.1910966610577134, rel_tol=0.0, abs_tol=1e-15), "required capture")
    require(design["controls_and_uncertainty"]["jackknife"].startswith("Delete one whole expert"), "expert jackknife")
    require(len(design["launch_blockers"]) == 6, "launch blocker count")
    require(all(value is False for value in design["claim_boundary"].values()), "claim boundary")
    require("no tuple retry" in design["decision"]["failure_status"].lower().replace("_", " "), "no retry status")
    require(bindings["authority"].startswith("No referenced artifact"), "binding non-authority")
    require(len(bindings["references"]) == 6, "reference count")
    return 13


def verify_plan_and_contract() -> int:
    sys.path.insert(0, str(ROOT))
    import contract
    import cupy_anchor
    import plan_snapshot

    fit_keys, score_keys = plan_snapshot.stage1_keys()
    require(len(fit_keys) == len(score_keys) == 2048, "plan sizes")
    require(not (set(fit_keys) & set(score_keys)), "plan overlap")
    require(contract.SEEDS_U32 == (3306464084, 235286348, 2174751347, 256779041, 118211936), "contract seeds")
    require(abs(contract.REQUIRED_CAPTURE - 0.1910966610577134) <= 1e-15, "contract capture")
    require(abs(contract.CONSERVATIVE_READ_AMPLIFICATION - 1.175) <= 1e-15, "contract read")
    require(cupy_anchor.CUDA_SOURCE_SHA256 == hashlib.sha256(cupy_anchor.CUDA_SOURCE.encode("utf-8")).hexdigest(), "CUDA source hash")
    require(cupy_anchor.CUDA_SOURCE_SHA256 == "580ea565670dbc41319abc3277d733d9160e7043ffec25e07df914ae8bb64701", "frozen CUDA source hash")
    return 8


def run_tests() -> int:
    sys.path.insert(0, str(ROOT))
    import test_source

    suite = unittest.defaultTestLoader.loadTestsFromModule(test_source)
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    require(result.wasSuccessful(), "source tests failed")
    return result.testsRun


def main() -> int:
    manifest = verify_closure()
    ast_checks = verify_ast()
    design_checks = verify_design()
    semantic_checks = verify_plan_and_contract()
    tests = run_tests()
    receipt = {
        "schema": "fuseed-pmg1-fixed-pentad-aux-stage1-source-verification-v1",
        "status": "PASS_SOURCE_ONLY_NO_PAYLOAD_OR_RUN_AUTHORITY",
        "manifest_sha256": sha256_file(ROOT / "SOURCE_MANIFEST.json"),
        "manifest_members": len(manifest["files"]),
        "ast_nodes_checked": ast_checks,
        "design_checks": design_checks,
        "semantic_checks": semantic_checks,
        "unit_tests": tests,
        "payload_files_opened": 0,
        "network_operations": 0,
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
