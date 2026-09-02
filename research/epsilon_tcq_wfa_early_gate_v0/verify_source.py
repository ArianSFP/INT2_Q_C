#!/usr/bin/env python3
"""Independent standard-library verifier for epsilon-TCQ source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


SCHEMA = "epsilon-tcq-wfa-early-gate-v0-source-manifest"
STATUS = "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY"
EXPECTED_MEMBERS = {
    "README.md", "cupy_backend.py", "design_lock.json", "gate_contract.py",
    "legal_interface.py", "packet_codec.py", "run_gate.py",
    "tcq_core.py", "test_source_only.py", "verify_source.py",
}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            VerifyError(f"nonfinite token {token}")))
    require(isinstance(value, dict), "top-level JSON object")
    return value


def read_regular(path: Path, maximum: int = 4 * (1 << 20)) -> bytes:
    require(path.is_absolute(), "absolute path")
    descriptor = os.open(os.fspath(path), os.O_RDONLY |
                         getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 < before.st_size <= maximum, "regular single-link file")
        pieces = []
        remaining = before.st_size
        while remaining:
            piece = os.read(descriptor, min(1 << 20, remaining))
            require(piece, "short source read")
            pieces.append(piece)
            remaining -= len(piece)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size,
                 before.st_mtime_ns, before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_size,
                 after.st_mtime_ns, after.st_nlink), "identity drift")
        return b"".join(pieces)
    finally:
        os.close(descriptor)


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "real package directory")
    entries = list(os.scandir(root))
    require(all(not entry.is_symlink() and entry.is_file(follow_symlinks=False)
                for entry in entries), "files only, no links")
    actual_names = {entry.name for entry in entries}
    require(actual_names == EXPECTED_MEMBERS | {"SOURCE_MANIFEST.json"},
            "exact frozen member set")

    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json")
    manifest_sha = digest(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(manifest_sha == expected_manifest_sha256,
                "expected manifest SHA-256")
    manifest = strict_json(manifest_payload)
    require(set(manifest) == {
        "schema", "status", "date", "members", "source_root_sha256",
        "access_attestation", "claim_boundary",
    }, "manifest exact schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS and
            manifest["date"] == "2026-09-02", "manifest identity")
    require(manifest["claim_boundary"] ==
            "source mechanics only; no Qwen/POLARIS payload authority",
            "claim boundary")
    access = manifest["access_attestation"]
    require(set(access) == {
        "qwen_payload_accessed", "current_codec_payload_accessed",
        "legal_candidate_trace_accessed", "matched_control_accessed",
        "cuda_accessed_by_isolated_source_tests", "network_accessed_by_tests",
        "tests_used_isolated_cpython",
    } and all(access[key] is False for key in access
              if key != "tests_used_isolated_cpython") and
            access["tests_used_isolated_cpython"] is True,
            "source-only access attestation")

    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_MEMBERS),
            "member row count")
    observed = []
    names = []
    payloads = {}
    for row in rows:
        require(set(row) == {"name", "bytes", "sha256"}, "member row schema")
        name = row["name"]
        require(name in EXPECTED_MEMBERS and name not in names and
                type(row["bytes"]) is int and row["bytes"] > 0 and
                isinstance(row["sha256"], str) and len(row["sha256"]) == 64,
                "member row values")
        payload = read_regular(root / name)
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"],
                f"member closure {name}")
        names.append(name)
        payloads[name] = payload
        observed.append({"name": name, "bytes": len(payload),
                         "sha256": digest(payload)})
    require(names == sorted(names, key=lambda value: value.encode("utf-8")),
            "bytewise member ordering")
    root_sha = digest(canonical_json(observed))
    require(root_sha == manifest["source_root_sha256"], "source root SHA-256")

    design = strict_json(payloads["design_lock.json"])
    require(design["status"] == "SOURCE_ONLY_NO_PAYLOAD_AUTHORITY" and
            design["interfaces"]["primary"] ==
            "strata_sc_6bit_legal_replay" and
            design["interfaces"]["primary_adapter_present_in_v0"] is False and
            design["interfaces"]["secondary"] ==
            "direct_int2_4level_new_codec" and
            design["physical"]["compressed_expert_refetch_allowed"] is False and
            design["controls"]["count"] == 8,
            "frozen design boundary")
    runner = payloads["run_gate.py"].decode("utf-8")
    require("--payload" not in runner and "--qwen" not in runner and
            "import cupy" not in runner,
            "runner accepts no payload and imports no accelerator")
    legal_source = payloads["legal_interface.py"].decode("utf-8")
    require("class SyntheticStrataLegalAdapter" in legal_source and
            "class DirectFourLevelAdapter" in legal_source and
            "class PolarisAdapter" not in legal_source,
            "production adapter absent")
    require("import cupy as cp" in
            payloads["cupy_backend.py"].decode("utf-8"),
            "CuPy heavy-path backend is explicit and lazy")
    return {
        "schema": "epsilon-tcq-wfa-v0-source-verifier-receipt",
        "status": "PASS",
        "source_manifest_sha256": manifest_sha,
        "source_root_sha256": root_sha,
        "members": len(rows),
        "qwen_payload_accessed": False,
        "current_codec_payload_accessed": False,
        "claim_authority": "SOURCE_MECHANICS_ONLY",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--package", type=Path, required=True)
    result.add_argument("--manifest-sha256")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(verify(arguments.package, arguments.manifest_sha256),
                     sort_keys=True, separators=(",", ":")))
