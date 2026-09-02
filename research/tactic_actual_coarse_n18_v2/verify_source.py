#!/usr/bin/env python3
"""Standard-library verifier for the closed UNIPOLAR-N18-307 v2 source tree."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


MANIFEST_NAME = "SOURCE_MANIFEST.json"
EXPECTED_FILES = {
    "POSTIMPLEMENTATION_REVIEW.md",
    "README.md",
    "dependency_graph.json",
    "design_lock.json",
    "n18_common.py",
    "preflight_gate.py",
    "runtime_contract.py",
    "runtime_environment_lock.json",
    "secure_io.py",
    "source_adapter.py",
    "test_source_only.py",
    "verify_source.py",
}


class VerificationError(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        check(key not in value, f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _reject(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant: {value}")


def strict_json(raw: bytes) -> Any:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject)
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError) as exc:
        raise VerificationError(f"strict JSON: {exc}") from exc
    return value


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for packet in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(packet)
    return hasher.hexdigest()


def verify_manifest(package: Path) -> dict[str, Any]:
    manifest_path = package / MANIFEST_NAME
    check(manifest_path.is_file() and not manifest_path.is_symlink(), "regular source manifest")
    raw = manifest_path.read_bytes()
    check(0 < len(raw) <= 1 << 20, "source manifest byte cap")
    value = strict_json(raw)
    check(isinstance(value, dict), "source manifest object")
    check(value.get("schema") == "tactic_actual_coarse_n18_source_manifest_v2", "source manifest schema")
    check(value.get("status") == "POSTIMPLEMENTATION_REVIEW_SOURCE_ONLY_RUNTIME_BLOCKED", "source manifest status")
    rows = value.get("files")
    check(isinstance(rows, list) and len(rows) == len(EXPECTED_FILES), "source manifest rows")
    check([row.get("name") for row in rows] == sorted(EXPECTED_FILES), "source manifest sorted names")
    check({path.name for path in package.iterdir()} == EXPECTED_FILES | {MANIFEST_NAME}, "exact source package closure")
    hashes: dict[str, str] = {}
    for row in rows:
        check(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "source manifest row schema")
        name = row["name"]
        check(name in EXPECTED_FILES, "source manifest member")
        path = package / name
        check(path.is_file() and not path.is_symlink(), f"regular source member: {name}")
        check(path.stat().st_size == row["bytes"], f"source member bytes: {name}")
        observed = digest(path)
        check(observed == row["sha256"], f"source member SHA-256: {name}")
        hashes[name] = observed
    return {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "member_sha256": hashes,
    }


def _imports(raw: str) -> set[str]:
    tree = ast.parse(raw)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    names.discard("")
    names.discard("__future__")
    return names


def verify_dependency_graph(package: Path, repo_root: Path) -> dict[str, Any]:
    graph = strict_json((package / "dependency_graph.json").read_bytes())
    check(graph.get("schema") == "tactic_actual_coarse_n18_dependency_graph_v2", "dependency graph schema")
    check(graph.get("status") == "SOURCE_PINNED_RUNTIME_ENVIRONMENT_SEPARATELY_LOCKED", "dependency graph status")
    rows = graph.get("external_python_sources")
    check(isinstance(rows, list) and len(rows) == 2, "dependency source rows")
    result = []
    for row in rows:
        check(
            set(row) == {"id", "relative_path", "bytes", "sha256", "allowed_import_roots"},
            "dependency row exact keys",
        )
        relative = row["relative_path"]
        check(isinstance(relative, str) and not relative.startswith("/") and ".." not in relative.split("/"), "dependency relative path")
        path = repo_root.joinpath(*relative.split("/"))
        check(path.is_file() and not path.is_symlink(), f"dependency regular file: {relative}")
        check(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], f"dependency bytes/hash: {relative}")
        imports = _imports(path.read_text(encoding="utf-8"))
        check(row["allowed_import_roots"] == sorted(set(row["allowed_import_roots"])), "dependency sorted imports")
        check(imports == set(row["allowed_import_roots"]), f"dependency AST import graph: {relative}")
        result.append({"id": row["id"], "sha256": row["sha256"], "imports": sorted(imports)})
    check(
        graph.get("numeric_runtime_distributions")
        == ["cupy-cuda12x", "numpy", "nvidia-ml-py", "scipy"],
        "numeric runtime distribution graph",
    )
    return {"external_sources": result, "runtime_distributions": graph["numeric_runtime_distributions"]}


def verify_static(package: Path) -> dict[str, Any]:
    forbidden_numeric = {"cupy", "numpy", "scipy", "pynvml", "torch", "tensorflow"}
    imports: dict[str, list[str]] = {}
    for name in sorted(path for path in EXPECTED_FILES if path.endswith(".py")):
        text = (package / name).read_text(encoding="utf-8")
        roots = _imports(text)
        check(not (roots & forbidden_numeric), f"source-only module imports numeric runtime: {name}")
        imports[name] = sorted(roots)
    preflight = (package / "preflight_gate.py").read_text(encoding="utf-8")
    check(preflight.index("_bootstrap_source(arguments.expected_source_manifest_sha256)") < preflight.index("from n18_common import"), "auth-before-sibling-import")
    check(preflight.index("authorization mismatch") < preflight.index("_bootstrap_source("), "authorization-before-source-open")
    check(preflight.index("CompletionLastPublisher(arguments.output") < preflight.index("HeldRegularFile(arguments.review_receipt"), "output-reservation-before-review")
    check(preflight.index("validate_environment_lock(environment.read())") < preflight.index("parse_source_plan(source_plan.read())"), "environment-before-plan/payload")
    check("import cupy" not in preflight and "import numpy" not in preflight, "preflight never imports numeric runtime")
    secure = (package / "secure_io.py").read_text(encoding="utf-8")
    for phrase in ("O_NOFOLLOW", "O_EXCL", "os.fsync", "renameat2", "COMPLETE.json"):
        check(phrase in secure, f"secure I/O primitive: {phrase}")
    check(secure.index('self._write("ARTIFACTS.json"') < secure.index('self._write("COMPLETE.json"'), "completion-last source order")
    runtime = (package / "runtime_contract.py").read_text(encoding="utf-8")
    check("validate_telemetry_receipt" in runtime and "LogicalTransferLedger" in runtime, "telemetry contract")
    return {"imports": imports}


def verify_design(package: Path) -> dict[str, Any]:
    design = strict_json((package / "design_lock.json").read_bytes())
    check(design.get("schema") == "tactic_actual_coarse_n18_source_design_v2", "design schema")
    check(design.get("status") == "POSTIMPLEMENTATION_REVIEW_SOURCE_ONLY_NO_PAYLOAD_NO_CUDA_AUTHORITY", "design status")
    packet = design["packet"]
    check(packet["version"] == 2 and packet["tile_values"] == 262144, "packet version/geometry")
    check(packet["header_bytes"] == 128 and packet["reservoir_bytes"] == 78592, "packet byte ledger")
    check(packet["payload_capacity_bits"] == 627712 and packet["logical_reserve_bits"] == 1024, "packet logical ledger")
    check(math.isclose(packet["physical_bpw_for_full_tiles"], 307 / 128, rel_tol=0, abs_tol=0), "packet physical rate")
    check(design["evaluation_order"]["cage_nonconverse"].startswith("Failure of the rank-384"), "CAGE non-converse boundary")
    authorization = design["authorization"]
    check(authorization["source_package_authorizes_payload"] is False, "no payload authority")
    check(authorization["source_package_authorizes_cuda"] is False, "no CUDA authority")
    check(authorization["token_alone_sufficient"] is False, "token not sufficient")
    environment = strict_json((package / "runtime_environment_lock.json").read_bytes())
    check(environment.get("schema") == "tactic_actual_coarse_n18_runtime_environment_v2", "environment schema")
    check(environment.get("status") == "UNFROZEN_BLOCK_RUNTIME", "environment intentionally blocks runtime")
    check(environment.get("interpreter") is None and environment.get("distributions") == [], "environment has no fake pins")
    return {
        "packet": packet,
        "environment_status": environment["status"],
        "authorization": authorization,
    }


def verify_readme(package: Path) -> None:
    text = (package / "README.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "arbitrary positive swiglu-moe",
        "hard logical eof",
        "completion-last",
        "held descriptor",
        "source before controls",
        "cannot kill cage",
        "runtime environment is deliberately unfrozen",
        "no payload",
        "no cuda",
        "307/128",
    ):
        check(phrase in text, f"README boundary phrase: {phrase}")


def verify_package(package: Path, repo_root: Path | None = None) -> dict[str, Any]:
    package = package.absolute()
    check(package.is_dir() and not package.is_symlink(), "regular package directory")
    resolved_package = package.resolve(strict=True)
    check(resolved_package == package, "canonical package path without symlink components")
    package = resolved_package
    repo = repo_root.resolve(strict=True) if repo_root is not None else package.parents[1]
    manifest = verify_manifest(package)
    # The package is now authenticated; sibling imports are permitted.
    import sys

    sys.path.insert(0, str(package))
    from n18_common import validate_fixed_ledger

    ledger = validate_fixed_ledger()
    design = verify_design(package)
    dependencies = verify_dependency_graph(package, repo)
    static = verify_static(package)
    verify_readme(package)
    return {
        "schema": "tactic_actual_coarse_n18_source_verification_v2",
        "status": "PASS_SOURCE_CLOSURE_RUNTIME_INTENTIONALLY_BLOCKED",
        **manifest,
        "fixed_ledger": ledger,
        "design": design,
        "dependencies": dependencies,
        "static": static,
        "payloads_opened": 0,
        "numeric_imports": 0,
        "cuda_contexts": 0,
        "claim_boundary": "Source closure and design only; no payload, CUDA, lower-rate artifact, DH384 result, or CAGE result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args()
    receipt = verify_package(arguments.package, arguments.repo_root)
    print(json.dumps(receipt, indent=None if arguments.compact else 2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
