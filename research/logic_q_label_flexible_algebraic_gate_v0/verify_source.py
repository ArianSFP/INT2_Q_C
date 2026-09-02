#!/usr/bin/env python3
"""Independent standard-library verifier for the frozen LOGIC-Q v0 source."""

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


SCHEMA = "logic-q-label-flexible-algebraic-gate-v0-source-manifest"
STATUS = "SEALED_SOURCE_ONLY_AWAITING_INDEPENDENT_AUDIT"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_NAMES = {
    "PRIOR_ART.md",
    "README.md",
    "design_lock.json",
    "logicq_core.py",
    "panel_protocol.py",
    "run_source_free_fixture.py",
    "test_source_only.py",
    "verify_source.py",
}
FORBIDDEN_IMPORT_ROOTS = {
    "cupy", "torch", "safetensors", "transformers", "huggingface_hub",
    "requests", "socket", "urllib", "http", "ftplib", "paramiko",
    "subprocess",
}


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
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"{label}: duplicate key {key!r}")
            result[key] = value
        return result
    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            VerifyError(f"{label}: nonfinite {item}")))
    require(isinstance(value, dict), f"{label}: JSON object")
    return value


def read_regular(path: Path, label: str) -> bytes:
    metadata = os.lstat(path)
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1,
            f"{label}: sole-link regular file")
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        remaining = before.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(chunk, f"{label}: short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", f"{label}: trailing byte")
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size,
                 before.st_mtime_ns, before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_size,
                 after.st_mtime_ns, after.st_nlink),
                f"{label}: identity drift")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def imported_roots(source: str, filename: str) -> set[str]:
    tree = ast.parse(source, filename=filename)
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
        "access_attestation", "test_attestation", "claim_boundary",
    }, "manifest exact schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "manifest schema/status")
    require(manifest["access_attestation"] == {
        "model_checkpoint_or_weight_payload_opened_statted_hashed_or_enumerated": False,
        "current_codec_or_coarse_payload_opened_statted_hashed_or_enumerated": False,
        "matched_control_artifact_opened_statted_hashed_or_enumerated": False,
        "cuda_or_cupy_initialized_during_source_build_fixture_or_tests": False,
        "source_fixture_or_tests_accessed_network": False,
        "payload_adapter_included": False,
    }, "source access attestation")
    require(manifest["test_attestation"] == {
        "runpod_source_only_tests_passed": True,
        "runpod_source_only_test_count": 25,
        "runpod_source_free_fixture_passed": True,
        "independent_source_verification_passed_after_freeze": True,
    }, "test attestation")
    require(isinstance(manifest["claim_boundary"], str) and
            "no model result" in manifest["claim_boundary"].lower(),
            "claim boundary")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "member rows")
    names = []
    observed = []
    import_receipt = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "member row schema")
        name = row["name"]
        require(isinstance(name, str) and name and name not in names and
                name != "SOURCE_MANIFEST.json" and "/" not in name and
                "\\" not in name, "safe unique member name")
        payload = read_regular(root / name, f"member {name}")
        observed.append({"name": name, "bytes": len(payload),
                         "sha256": sha256(payload)})
        names.append(name)
        if name.endswith(".py"):
            source = payload.decode("utf-8")
            roots = imported_roots(source, name)
            require(not roots & FORBIDDEN_IMPORT_ROOTS,
                    f"{name}: forbidden payload/network/accelerator import")
            import_receipt[name] = sorted(roots)
    require(set(names) == REQUIRED_NAMES, "complete frozen source set")
    require(names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical member order")
    require(rows == observed, "literal member rows")
    require(manifest["source_root_sha256"] == sha256(canonical_json(observed)),
            "source snapshot root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == REQUIRED_NAMES | {"SOURCE_MANIFEST.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "exact regular-file package closure")
    design = strict_json(read_regular(root / "design_lock.json", "design lock"),
                         "design lock")
    require(design["status"] == "SEALED_SOURCE_ONLY_NO_MODEL_PAYLOAD_AUTHORITY" and
            design["source_access"]["payload_adapter_included"] is False and
            design["claim_boundary"].startswith("Source-only"),
            "design source-only boundary")
    require(not any(name == "cupy" or name.startswith("cupy.")
                    for name in sys.modules), "verifier initialized CuPy")
    return {
        "schema": "logic-q-label-flexible-algebraic-gate-v0-source-verification",
        "status": "PASS_INDEPENDENT_SOURCE_VERIFICATION",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": manifest["source_root_sha256"],
        "members": observed,
        "python_import_roots": import_receipt,
        "payload_accessed": False,
        "network_accessed": False,
        "cupy_imported": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--package", type=Path,
                       default=Path(__file__).resolve().parent)
    value.add_argument("--expected-manifest-sha256")
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(verify(arguments.package,
                            arguments.expected_manifest_sha256),
                     sort_keys=True, separators=(",", ":"), allow_nan=False))
