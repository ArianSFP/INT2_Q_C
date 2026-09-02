#!/usr/bin/env python3
"""Verify the exact independent replay-v2 audit source inventory."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat


SCHEMA = "strata-bmp-qtt6-v2-independent-audit-source-manifest-v1"
STATUS = "FROZEN_INDEPENDENT_V2_SOURCE_AUDIT__RUNTIME_PENDING__HOLD_PAYLOAD"
PRODUCER_MANIFEST = "84df0d32a55682f6565ac9d144f7de850acf77cde27bffdefa77a151211906f8"
PRODUCER_ROOT = "b518b203c43fd401c94e1bfcf67e029a85a95f1f7ce244fcd864a96d0780da47"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED = {
    "README.md", "STATIC_REVIEW.json", "THREAT_MODEL.md", "run_audit.py",
    "run_real_cupy_audit.py", "test_benign_audit.py", "verify_audit_source.py",
}
FORBIDDEN_IMPORTS = {
    "requests", "socket", "urllib", "http", "ftplib", "paramiko",
    "safetensors", "transformers", "huggingface_hub", "torch",
}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def read_regular(path: Path, label: str) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            f"{label} regular non-link")
    payload = path.read_bytes()
    after = path.lstat()
    require((before.st_size, before.st_mtime_ns, before.st_mode,
             getattr(before, "st_ino", 0)) ==
            (after.st_size, after.st_mtime_ns, after.st_mode,
             getattr(after, "st_ino", 0)), f"{label} changed during read")
    return payload


def strict_json(payload: bytes, label: str) -> dict:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           VerifyError(f"{label} nonfinite {token}")))
    require(isinstance(value, dict), f"{label} object")
    return value


def import_roots(source: str, filename: str) -> set[str]:
    tree = ast.parse(source, filename=filename)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def verify(package: Path, expected_manifest_sha256: str | None) -> dict:
    root = package.resolve(strict=True)
    require(root.is_dir(), "audit package directory")
    manifest_payload = read_regular(root / "AUDIT_SOURCE_MANIFEST.json",
                                    "audit manifest")
    manifest_sha = sha256(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(HEX64.fullmatch(expected_manifest_sha256) is not None and
                expected_manifest_sha256 == manifest_sha,
                "external audit manifest pin")
    manifest = strict_json(manifest_payload, "audit manifest")
    require(set(manifest) == {
        "schema", "status", "producer_pins", "audit_source_root_sha256",
        "members", "static_disposition", "execution_attestation",
        "access_attestation", "claim_boundary",
    }, "audit manifest schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "audit schema/status")
    require(manifest["producer_pins"] == {
        "source_manifest_sha256": PRODUCER_MANIFEST,
        "source_root_sha256": PRODUCER_ROOT,
    }, "producer pins")
    require(manifest["execution_attestation"] == {
        "independent_cpu_tests_declared": 19,
        "independent_cpu_tests_executed": False,
        "producer_manifest_self_replay_declared": True,
        "fresh_cupy_validator_present": True,
        "fresh_cupy_validator_executed": False,
        "runtime_pass_claimed": False,
    }, "execution attestation")
    require(manifest["access_attestation"] == {
        "qwen_or_other_model_payload_opened_statted_hashed_or_enumerated": False,
        "strata_or_coarse_payload_opened_statted_hashed_or_enumerated": False,
        "matched_control_payload_opened_statted_hashed_or_enumerated": False,
        "network_used": False,
        "producer_modified": False,
        "payload_authority": False,
    }, "access attestation")
    require(manifest["static_disposition"].startswith(
        "PASS_V1_REPLAY_AND_WORKSPACE_REPAIRS__"), "static disposition")

    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "audit members")
    observed, names, imports = [], [], {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "member schema")
        name = row["name"]
        require(isinstance(name, str) and name in REQUIRED and name not in names and
                "/" not in name and "\\" not in name, "member name")
        payload = read_regular(root / name, f"member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"member pin {name}")
        observed.append(item)
        names.append(name)
        if name.endswith(".py"):
            roots = import_roots(payload.decode("utf-8"), name)
            require(not roots & FORBIDDEN_IMPORTS, f"forbidden import {name}")
            if "subprocess" in roots:
                require(name in {"run_real_cupy_audit.py", "test_benign_audit.py"},
                        "subprocess boundary")
            imports[name] = sorted(roots)
    require(set(names) == REQUIRED and
            names == sorted(names, key=lambda value: value.encode("utf-8")),
            "canonical complete audit members")
    require(sha256(canonical_json(observed)) ==
            manifest["audit_source_root_sha256"], "audit source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == REQUIRED |
            {"AUDIT_SOURCE_MANIFEST.json"} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "exact audit filesystem closure")

    review = strict_json(read_regular(root / "STATIC_REVIEW.json", "review"),
                         "review")
    require(review["producer_manifest_sha256"] == PRODUCER_MANIFEST and
            review["producer_source_root_sha256"] == PRODUCER_ROOT and
            len(review["findings"]) == 3 and
            review["access_attestation"]["payload_authority"] is False,
            "review closure")
    readme = read_regular(root / "README.md", "README").decode("utf-8")
    for phrase in ("v1 replay and workspace findings are repaired",
                   "thirteen non-manifest members",
                   "not a trusted production capability",
                   "runtime remains pending", "HOLD_TRUSTED_PRODUCTION"):
        require(phrase in readme, f"README boundary {phrase}")
    return {
        "schema": "strata-bmp-qtt6-v2-independent-audit-source-verification-v1",
        "status": "PASS_EXACT_AUDIT_SOURCE__RUNTIME_PENDING__HOLD_PAYLOAD",
        "audit_manifest_sha256": manifest_sha,
        "audit_source_root_sha256": manifest["audit_source_root_sha256"],
        "producer_manifest_sha256": PRODUCER_MANIFEST,
        "producer_source_root_sha256": PRODUCER_ROOT,
        "members": observed,
        "python_import_roots": imports,
        "runtime_pass_claimed": False,
        "payload_accessed": False,
        "network_accessed": False,
        "producer_modified": False,
        "payload_authority": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.expected_manifest_sha256),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

