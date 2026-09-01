#!/usr/bin/env python3
"""Pure-stdlib verifier for the source-only PSNO-v1 proposal."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "polar_normal_predictor/result.json"
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
EXPECTED_PARENT = "e4fecac5f676d84739972bbf0e04467027aeae1356e62e1dc3cd2b84bff67026"
HEAVY = {"numpy", "cupy", "torch", "scipy", "transformers", "safetensors", "cuda"}


class VerificationFailure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, label: str) -> None:
        if not condition:
            raise VerificationFailure(label)
        self.count += 1

    def equal(self, observed: Any, expected: Any, label: str) -> None:
        self.require(observed == expected, f"{label}: {observed!r} != {expected!r}")

    def close(self, observed: float, expected: float, label: str, tolerance: float) -> None:
        self.require(math.isfinite(observed) and abs(observed - expected) <= tolerance,
                     f"{label}: {observed!r} != {expected!r}")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def regular_bytes(path: Path, checks: Checks, label: str) -> bytes:
    info = path.lstat()
    checks.require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), label + " regular nonlink")
    checks.equal(info.st_nlink, 1, label + " single link")
    raw = path.read_bytes()
    checks.equal(len(raw), info.st_size, label + " stable size")
    return raw


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationFailure("duplicate JSON key " + key)
        result[key] = value
    return result


def load_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerificationFailure(f"nonfinite {label} {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationFailure(f"invalid {label}: {exc}") from exc


def imports(source: str) -> set[str]:
    tree = ast.parse(source)
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".", 1)[0])
    return result


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerificationFailure("module spec " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify() -> dict[str, Any]:
    checks = Checks()
    manifest_raw = regular_bytes(MANIFEST, checks, "manifest")
    manifest = load_json(manifest_raw, "manifest")
    checks.equal(manifest["schema"], "polar_sparse_normal_source_package_manifest_v1", "manifest schema")
    checks.equal(manifest["status"], "SEALED_SOURCE_ONLY_NO_AUTHORITY", "manifest status")
    checks.equal(manifest["authorization"], "NONE", "manifest authority")
    rows = manifest["artifacts"]
    checks.equal(manifest["artifact_count"], len(rows), "manifest count")
    names = [row["path"] for row in rows]
    checks.equal(names, sorted(names), "manifest sorted")
    checks.equal(len(names), len(set(names)), "manifest unique")
    checks.equal(sorted(entry.name for entry in os.scandir(ROOT)),
                 sorted(names + ["PACKAGE_MANIFEST.json"]), "package exact closure")
    sources: dict[str, str] = {}
    for row in rows:
        checks.equal(set(row), {"path", "bytes", "sha256"}, "manifest row closure")
        raw = regular_bytes(ROOT / row["path"], checks, "artifact " + row["path"])
        checks.equal(len(raw), row["bytes"], "artifact size " + row["path"])
        checks.equal(sha(raw), row["sha256"], "artifact hash " + row["path"])
        if row["path"].endswith(".py"):
            sources[row["path"]] = raw.decode("utf-8", errors="strict")
            ast.parse(sources[row["path"]], filename=row["path"])
            checks.require(not (imports(sources[row["path"]]) & HEAVY),
                           "no heavy import " + row["path"])

    parent_raw = regular_bytes(PARENT, checks, "parent result")
    checks.equal(sha(parent_raw), EXPECTED_PARENT, "parent hash")
    parent = load_json(parent_raw, "parent result")
    checks.equal(parent["decision"]["status"], "KILL_POLAR_NORMAL_PREDICTOR_BRANCH",
                 "parent remains killed")
    protocol = load_json((ROOT / "protocol.json").read_bytes(), "protocol")
    checks.equal(protocol["status"], "SOURCE_ONLY_NO_TENSOR_OR_GPU_AUTHORITY", "protocol status")
    checks.require(protocol["lineage"]["overrides_parent_kill"] is False, "does not override parent")
    checks.equal(protocol["geometry"]["unique_symmetric_coordinates_per_normal"], 295296,
                 "symmetric coordinate count")
    checks.equal(protocol["exact_equal_k_reference"]["free_value_last_viable_k"], 48018,
                 "protocol exact boundary")
    checks.require(all(value is False for value in protocol["authorization"].values()),
                   "protocol grants no authority")

    cupy_source = sources["cupy_gate_proposal.py"]
    checks.equal(imports(cupy_source), {"__future__", "math", "typing"},
                 "CuPy proposal inert import closure")
    checks.require("import cupy" not in cupy_source.lower(), "CuPy not imported on module import")
    for forbidden in ("Path(", "open(", "fromfile", "socket", "subprocess", "requests", "urllib"):
        checks.require(forbidden.lower() not in cupy_source.lower(), "CuPy proposal no I/O " + forbidden)
    checks.require("AUTHORITY_GRANTED = False" in cupy_source, "CuPy proposal no authority")
    checks.require("a_ij = sqrt(2) N_ij" in (ROOT / "README.md").read_text(encoding="utf-8"),
                   "orthonormal symmetric documentation")

    threshold = load_module(ROOT / "threshold_math.py", "psno_threshold_verify")
    report = threshold.build_report(PARENT)
    checks.equal(report["tensor_files_opened"], 0, "threshold opens no tensors")
    checks.equal(report["gpu_operations"], 0, "threshold runs no GPU")
    checks.close(float(report["baseline_F"]), 0.9520339564260487, "threshold baseline", 3e-15)
    boundary = report["boundaries_by_exact_value_bits"]["0"]
    checks.equal(boundary["last_viable_k"], 48018, "exact last free-value k")
    checks.equal(boundary["last_support_bits_per_matrix"], 189133, "exact last support bits")
    checks.close(float(boundary["last_full_capture_F"]), 0.7999988173257864,
                 "exact last full-capture F", 3e-15)
    checks.equal(boundary["next_k"], 48019, "exact next k")
    checks.equal(boundary["next_support_bits_per_matrix"], 189136, "exact next support bits")
    checks.require(float(boundary["next_full_capture_F"]) > 0.8, "next row impossible")

    dual = load_module(ROOT / "dual_bound.py", "psno_dual_verify")
    records = parent["source_normal_records"]
    curves = [
        [{
            "name": "identity_k0",
            "k": 0,
            "residual_energy": float(row["normal_energy"]),
            "removed_dof": 0,
            "support_bits": 0,
            "value_symbols": 0,
        }]
        for row in records
    ]
    multipliers = [10.0 ** (-6.0 + 7.0 * index / 7999.0) for index in range(8000)]
    scan = dual.scan_dual(
        records,
        curves,
        multipliers,
        base_side_bpw=float(parent["scope"]["base_polar_explicit_side_bpw"]),
        value_bits_per_symbol=0.0,
    )
    primal_distortion = float(report["baseline_F"]) / 32.0
    dual_distortion = float(scan["best"]["raw_dual_distortion"])
    checks.require(dual_distortion <= primal_distortion + 2e-12, "dual stays below primal")
    checks.close(dual_distortion, primal_distortion, "dual reproduces fixed waterfill", 2e-8)
    checks.require(dual.source_only_status()["authorization"] is False, "dual no authority")
    checks.require(not (set(sys.modules) & HEAVY), "verifier imported no heavy modules")

    return {
        "schema": "polar_sparse_normal_source_verification_v1",
        "status": "PASS_SOURCE_ONLY_GATE_PROPOSAL",
        "checks": checks.count,
        "package_manifest_sha256": sha(manifest_raw),
        "parent_result_sha256": EXPECTED_PARENT,
        "free_value_equal_k_boundary": 48018,
        "fixed_curve_dual_F": dual_distortion * 32.0,
        "tensor_files_opened": 0,
        "cupy_imports": 0,
        "gpu_operations": 0,
        "authorizations_issued": 0,
        "authorization": "NONE",
    }


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2, sort_keys=True, allow_nan=False))
    except Exception as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        raise
