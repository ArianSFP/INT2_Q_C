#!/usr/bin/env python3
"""Independent standard-library verifier for LOGIC-Q v1 capped adapter."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


SCHEMA = "logic-q-label-flexible-algebraic-gate-v1-capped-adapter-source-manifest"
STATUS = "SEALED_SOURCE_ONLY_AWAITING_INDEPENDENT_AUDIT"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_NAMES = {
    "PREFLIGHT_HISTORY.json", "README.md", "capped_adapter.py", "design_lock.json",
    "run_source_free_fixture.py", "test_source_only.py", "verify_source.py",
}
FORBIDDEN_IMPORT_ROOTS = {
    "cupy", "torch", "safetensors", "transformers", "huggingface_hub",
    "requests", "socket", "urllib", "http", "ftplib", "paramiko",
    "subprocess",
}
PARENT_MANIFEST_SHA256 = "31edbc3325dfdae2b3f43cce4afb360062d5c70583b57dd1e6530835a178cced"
PARENT_SOURCE_ROOT_SHA256 = "2177f2aec39a65afddbbded9b6b3cd2c2a33118c060a41e070102f9fb6c95d4a"


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


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerifyError(f"{label} nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifyError(f"{label} strict JSON") from exc
    require(isinstance(value, dict), f"{label} object")
    return value


def read_regular(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise VerifyError(f"{label} unavailable") from exc
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            f"{label} regular non-link")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise VerifyError(f"{label} read") from exc
    require((before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino),
            f"{label} changed during read")
    return payload


def imported_roots(source: str, filename: str) -> set[str]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise VerifyError(f"{filename} syntax") from exc
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module.split(".", 1)[0])
    return result


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir(), "package directory")
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json", "manifest")
    manifest_sha = sha256(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(HEX64.fullmatch(expected_manifest_sha256) is not None and
                manifest_sha == expected_manifest_sha256,
                "external source manifest SHA-256")
    manifest = strict_json(manifest_payload, "manifest")
    require(set(manifest) == {
        "schema", "status", "source_root_sha256", "members",
        "parent_dependency", "access_attestation", "test_attestation",
        "claim_boundary",
    }, "manifest exact schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "manifest schema/status")
    require(manifest["parent_dependency"] == {
        "source_manifest_sha256": PARENT_MANIFEST_SHA256,
        "source_root_sha256": PARENT_SOURCE_ROOT_SHA256,
        "parent_modified": False,
    }, "pinned parent dependency")
    require(manifest["access_attestation"] == {
        "model_checkpoint_or_qwen_payload_opened_statted_hashed_or_enumerated": False,
        "current_codec_or_coarse_payload_opened_statted_hashed_or_enumerated": False,
        "matched_control_artifact_opened_statted_hashed_or_enumerated": False,
        "network_used_by_source_fixture_or_tests": False,
        "cupy_imported_or_initialized_by_source_fixture_or_tests": False,
        "live_qwen_authority": False,
    }, "source access attestation")
    require(manifest["test_attestation"] == {
        "local_source_only_tests_passed": True,
        "local_source_only_test_count": 33,
        "local_source_free_fixture_passed": True,
        "independent_source_audit_passed": False,
    }, "test attestation")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "member rows")
    observed = []
    names = []
    imports = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "member row schema")
        name = row["name"]
        require(isinstance(name, str) and name in REQUIRED_NAMES and
                name not in names and "/" not in name and "\\" not in name,
                "safe unique member")
        payload = read_regular(root / name, f"member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"member pin {name}")
        observed.append(item)
        names.append(name)
        if name.endswith(".py"):
            roots = imported_roots(payload.decode("utf-8"), name)
            require(not roots & FORBIDDEN_IMPORT_ROOTS,
                    f"{name} forbidden import")
            imports[name] = sorted(roots)
    require(set(names) == REQUIRED_NAMES and
            names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical complete members")
    require(manifest["source_root_sha256"] == sha256(canonical_json(observed)),
            "source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == REQUIRED_NAMES | {"SOURCE_MANIFEST.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "exact regular package closure")
    design = strict_json(read_regular(root / "design_lock.json", "design"), "design")
    require(design["status"] == "SEALED_SOURCE_ONLY_NO_QWEN_AUTHORITY" and
            design["parent_dependency"]["source_manifest_sha256"] ==
            PARENT_MANIFEST_SHA256 and
            design["source_access"]["live_qwen_authority"] is False and
            design["claim_boundary"].startswith("Source-only"),
            "design source boundary")
    history = strict_json(read_regular(root / "PREFLIGHT_HISTORY.json", "preflight"),
                          "preflight")
    require(history["status"] ==
            "INITIAL_FAILURE_PRESERVED__FINAL_EXACT_STAGE_REQUIRED" and
            history["attempts"][0]["result"] == "FAIL_CLOSED" and
            history["attempts"][0]["failure_ignored"] is False and
            history["attempts"][0]["error"] ==
            "VerifyError: exact regular package closure",
            "preflight failure history")
    require("cupy" not in sys.modules, "verifier initialized CuPy")
    return {
        "schema": "logic-q-v1-capped-adapter-source-verification",
        "status": "PASS_INDEPENDENT_SOURCE_VERIFICATION",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": observed, "python_import_roots": imports,
        "parent_dependency": manifest["parent_dependency"],
        "payload_accessed": False, "network_accessed": False,
        "cupy_imported": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    value.add_argument("--expected-manifest-sha256")
    return value


def main() -> None:
    arguments = parser().parse_args()
    print(json.dumps(verify(arguments.package, arguments.expected_manifest_sha256),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
