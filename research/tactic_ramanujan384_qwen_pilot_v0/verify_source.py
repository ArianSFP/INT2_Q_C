#!/usr/bin/env python3
"""Stdlib-only source verifier for the held Ramanujan-384 Qwen pilot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


SCHEMA = "tactic-ramanujan384-qwen-pilot-v0-source-manifest"
STATUS = "FROZEN_SOURCE_ONLY__COMPILE_TIME_CAPABILITY_PIN_NONE__NO_PAYLOAD_OPENED"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MEMBERS = {
    "README.md", "SOURCE_ONLY_TEST_RESULT.json", "aperture.py", "capability.py",
    "design_lock.json", "pilot_runner.py", "test_source_only.py", "verify_source.py",
}
DEPENDENCIES = {
    "tactic_ramanujan384_adapter_v3_atomic": {
        "manifest_name": "SOURCE_MANIFEST.json",
        "manifest_sha256": "97fb4cba64ff884615810fc8fc835c12ce98bf3e9db37b8a77be93d0d5372be1",
        "source_root_sha256": "5f86d9a1b48f7769867c828322132be303617d0444d50b5439f7b9d0074ab674",
        "root_field": "source_root_sha256",
        "schema": "tactic-ramanujan384-atomic-source-manifest-v3",
        "root_domain_hex": "",
        "root_row_order": "sorted_keys",
    },
    "tactic_ramanujan384_adapter_v3_atomic_independent_source_review_20260903": {
        "manifest_name": "source_manifest.json",
        "manifest_sha256": "60feb6ae08b3d57df6056e0912759b1e4eb9eb7888c90467cbfd37e72ba97173",
        "source_root_sha256": "27f422950b7bdd686541677341665fb075295cdfbdd2e1acac3a5c42ce089cd2",
        "root_field": "source_root_sha256",
        "schema": "tactic-ramanujan384-atomic-v3-independent-source-review-manifest-v1",
        "root_domain_hex": "",
        "root_row_order": "name_bytes_sha256",
    },
    "tactic_ramanujan384_adapter_v2_scalable": {
        "manifest_name": "SOURCE_MANIFEST.json",
        "manifest_sha256": "1f579f33216edeebbebb6c1714a4e56739da30ae0f12ae9bd44baf15a6163209",
        "source_root_sha256": "bff5a0c541cb2117a8cc1db3e539493bacc590b4e007ab7f193ca615e03a7495",
        "root_field": "source_root_sha256",
        "schema": "tactic-ramanujan384-scalable-source-manifest-v2",
        "root_domain_hex": "",
        "root_row_order": "sorted_keys",
    },
    "tactic_ramanujan384_adapter_v2_scalable_independent_source_review_20260903": {
        "manifest_name": "SOURCE_MANIFEST.json",
        "manifest_sha256": "4ed8c0fe24db072e22aef84791a01ccf637cb337376a389d47119248fd257281",
        "source_root_sha256": "16ea8dfde5cf7a48552dc7b5a74b209488934b8764e890bf51bb5cd02985cd39",
        "root_field": "source_root_sha256",
        "schema": "tactic-ramanujan384-v2-scalable-independent-source-review-manifest",
        "root_domain_hex": "",
        "root_row_order": "sorted_keys",
    },
    "tactic_actual_coarse_n18_v6": {
        "manifest_name": "SOURCE_MANIFEST.json",
        "manifest_sha256": "31662539a4c55926f47b378d15a0d8e23c90aa0903328c44be2e237eca48b15d",
        "source_root_sha256": "161ab23169af3427648ec1bbcb9402568a0fb8aefc4a794daf3ebd1c56cc83f2",
        "root_field": "source_root_sha256",
        "schema": "tactic-actual-coarse-n18-v6-source-manifest-v1",
        "root_domain_hex": "",
        "root_row_order": "sorted_keys",
    },
    "tactic_actual_coarse_n18_v6_result_auditor_v1": {
        "manifest_name": "SOURCE_MANIFEST.json",
        "manifest_sha256": "5386571db2a8e828c09368f603b3ccf0ccf3936204e7e06231d5c5798eb9f97f",
        "source_root_sha256": "59387c67a18bb776cca820e658be998d75f9c3c1a9b7ef5c809e692f78a50742",
        "root_field": "source_snapshot_root_sha256",
        "schema": "tactic-actual-coarse-n18-v6-result-auditor-source-manifest-v1",
        "root_domain_hex": (
            "5441435449432d41435455414c2d434f415253452d4e31382d56362d524553554c"
            "542d41554449544f522d534f555243452d524f4f542d563100"),
        "root_row_order": "sorted_keys",
    },
}
COARSE_AUDIT_PATH = "tactic_actual_coarse_n18_v6_qwen_result_audit_20260902_538657/AUDIT_RECEIPT.json"
COARSE_AUDIT_SHA256 = "e03af88a5d33eaca30f935fffc8fcade477219c1be1afebb952428982e4d48e7"
FORBIDDEN_IMPORTS = {"cupy", "torch", "safetensors", "transformers",
                     "huggingface_hub", "requests", "socket", "urllib", "http",
                     "ftplib", "paramiko"}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def closure_root_rows(rows: list[dict[str, Any]], row_order: str) -> bytes:
    if row_order == "sorted_keys":
        return canonical_json(rows)
    require(row_order == "name_bytes_sha256", "dependency root row order")
    ordered = [{"name": row["name"], "bytes": row["bytes"],
                "sha256": row["sha256"]} for row in rows]
    return json.dumps(ordered, sort_keys=False, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str, *, canonical: bool = False) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerifyError(f"{label}: nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifyError(f"{label}: JSON") from exc
    require(isinstance(value, dict), f"{label}: object")
    if canonical:
        require(payload == canonical_json(value) + b"\n", f"{label}: canonical JSON")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                f"{label}: regular non-link")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise VerifyError(f"{label}: read") from exc
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_mode) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
             after.st_mode), f"{label}: identity drift")
    return payload


def import_roots(source: str, filename: str) -> set[str]:
    tree = ast.parse(source, filename=filename)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def verify_flat_dependency(root: Path, expected: dict[str, str]) -> dict[str, Any]:
    manifest_payload = regular_bytes(root / expected["manifest_name"],
                                     "dependency manifest")
    require(sha256(manifest_payload) == expected["manifest_sha256"],
            "dependency manifest hash")
    manifest = strict_json(manifest_payload, "dependency manifest")
    require(manifest.get("schema") == expected["schema"] and
            manifest.get(expected["root_field"]) == expected["source_root_sha256"],
            "dependency schema/root")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "dependency members")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "dependency member row")
        payload = regular_bytes(root / row["name"], "dependency member")
        item = {"name": row["name"], "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row and row["name"] not in names, "dependency member pin")
        names.append(row["name"])
        observed.append(item)
    domain = bytes.fromhex(expected["root_domain_hex"])
    require(rows == sorted(rows, key=lambda row: row["name"].encode("utf-8")) and
            sha256(domain + closure_root_rows(
                observed, expected["root_row_order"])) == expected["source_root_sha256"],
            "dependency canonical source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {expected["manifest_name"]} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "dependency exact closure")
    return {"manifest_sha256": expected["manifest_sha256"],
            "source_root_sha256": expected["source_root_sha256"],
            "members": len(names)}


def verify(package: Path, expected_manifest_sha256: str,
           repository_root: Path) -> dict[str, Any]:
    root = package.resolve(strict=True)
    repo = repository_root.resolve(strict=True)
    require(root.is_dir() and repo.is_dir(), "source/repository roots")
    manifest_payload = regular_bytes(root / "SOURCE_MANIFEST.json", "source manifest")
    require(HEX64.fullmatch(expected_manifest_sha256) is not None and
            sha256(manifest_payload) == expected_manifest_sha256,
            "external source manifest pin")
    manifest = strict_json(manifest_payload, "source manifest", canonical=True)
    require(set(manifest) == {"schema", "status", "source_root_sha256", "members",
                              "dependencies", "execution_attestation", "claim_boundary"} and
            manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "source manifest schema/status")
    require(manifest["dependencies"] == {
        key: {field: value for field, value in expected.items()
              if field in {"manifest_sha256", "source_root_sha256"}}
        for key, expected in DEPENDENCIES.items()
    }, "source manifest dependency pins")
    require(manifest["execution_attestation"] == {
        "final_frozen_source_tests_executed": False,
        "pre_hardening_source_only_tests": 17,
        "pre_hardening_failures": 0, "pre_hardening_errors": 0,
        "cupy_initialized": False,
        "qwen_or_coarse_payload_opened": False, "runpod_executed": False,
        "production_authorized": False,
    }, "honest execution attestation")
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(MEMBERS), "source members")
    observed = []
    names = []
    imports = {}
    sources = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "source member row")
        name = row["name"]
        require(name in MEMBERS and name not in names, "source member name")
        payload = regular_bytes(root / name, f"source member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"source member pin {name}")
        names.append(name)
        observed.append(item)
        if name.endswith(".py"):
            source = payload.decode("utf-8")
            roots = import_roots(source, name)
            require(not roots & FORBIDDEN_IMPORTS, f"{name}: forbidden static import")
            if name not in {"verify_source.py", "test_source_only.py"}:
                require("root@" not in source and ".safetensors" not in source,
                        f"{name}: payload locator")
            imports[name] = sorted(roots)
            sources[name] = source
    require(set(names) == MEMBERS and
            names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical source member order")
    require(sha256(canonical_json(observed)) == manifest["source_root_sha256"],
            "source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == MEMBERS | {"SOURCE_MANIFEST.json"} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "exact source closure")

    capability_source = sources["capability.py"]
    aperture_source = sources["aperture.py"]
    runner_source = sources["pilot_runner.py"]
    require("TRUSTED_CAPABILITY_SHA256: str | None = None" in capability_source and
            "HOLD: compile-time external capability SHA-256 is None" in capability_source and
            "REQUIRED_CAPTURE = 0.32387022205373717" in capability_source and
            "COARSE_FRAME_SHA256" in capability_source,
            "compiled fail-closed capability")
    require("score_literal_rank_packets" in aperture_source and
            "core.encode_packet" in aperture_source and "core.decode_packet" in aperture_source and
            "per_candidate_solve_calls" in aperture_source and
            "owner_lcb_capture" in aperture_source,
            "literal-rank owner-LCB aperture")
    require(runner_source.index("capability.authorize_production") <
            runner_source.index('importlib.import_module("cupy")') and
            "one_pass_page_trace" in runner_source and
            "HARD_KILL_SOURCE_FIRST_APERTURE" in runner_source and
            "HARD_KILL_FULL_EXPERT_D_GT_0_025" in runner_source and
            "projected_transfer_used\": False" in runner_source,
            "source-first runner gates")
    design = strict_json(regular_bytes(root / "design_lock.json", "design"), "design")
    require(design["status"] == STATUS and
            design["authority"]["compile_time_capability_sha256"] is None and
            design["aperture"]["required_coarse_residual_capture"] ==
            0.32387022205373717 and
            design["survivor"]["physical_rate_exact"] == "359/144" and
            design["payload_execution"]["qwen_opened"] is False,
            "design lock")
    test_result = strict_json(
        regular_bytes(root / "SOURCE_ONLY_TEST_RESULT.json", "test result"), "test result")
    pre = test_result["pre_hardening_execution"]
    require(test_result["status"] ==
            "HOLD_FINAL_FROZEN_SOURCE_NOT_EXECUTED__PRE_HARDENING_17_PASS" and
            pre["tests_run"] == 17 and pre["failures"] == 0 and
            pre["errors"] == 0 and
            test_result["final_frozen_source_tests_executed"] is False and
            test_result["mechanism_only"] is True and
            test_result["cupy_imported_or_initialized"] is False and
            test_result["production_authorized"] is False,
            "source-only test receipt")

    dependencies = {}
    research = repo / "research"
    for directory, expected in DEPENDENCIES.items():
        dependencies[directory] = verify_flat_dependency(research / directory, expected)
    coarse_audit_payload = regular_bytes(research / COARSE_AUDIT_PATH,
                                         "coarse result audit receipt")
    require(sha256(coarse_audit_payload) == COARSE_AUDIT_SHA256,
            "coarse result audit receipt external pin")
    coarse_audit = strict_json(coarse_audit_payload, "coarse result audit")
    require(coarse_audit.get("status") ==
            "PASS_EXACT_NONPROMOTING_TACTIC_ACTUAL_COARSE_N18_V6_QWEN_RESULT_AUDIT" and
            coarse_audit.get("literal_COARSE_canonical_reencode_matches") is True and
            coarse_audit.get("recomputed", {}).get("frame_sha256") ==
            "6c13780bf1494567f91bc73bf6afd8846c6e3326cac329e4d8e3faf48a9051d7",
            "coarse result audit semantic pin")
    require("cupy" not in __import__("sys").modules, "source verifier initialized CuPy")
    return {
        "schema": "tactic-ramanujan384-qwen-pilot-v0-source-verification",
        "status": "PASS_SOURCE_AND_DEPENDENCY_CLOSURES__RUNTIME_PAYLOAD_HELD",
        "source_manifest_sha256": expected_manifest_sha256,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": observed, "python_import_roots": imports,
        "dependencies": dependencies,
        "coarse_result_audit_file_sha256": COARSE_AUDIT_SHA256,
        "cupy_initialized": False, "qwen_or_coarse_payload_opened": False,
        "production_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.expected_manifest_sha256,
                            args.repository_root), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
