#!/usr/bin/env python3
"""Independent standard-library verifier for posterior-centroid v1 source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


SCHEMA = "uwfa-sc-posterior-centroid-source-manifest-v1"
STATUS = "SEALED_SOURCE_ONLY_NONPROMOTING_NO_PAYLOAD_AUTHORITY"
EXPECTED_MEMBERS = {
    "README.md",
    "design_lock.json",
    "diagnostic.py",
    "posterior_core.py",
    "result_bridge.py",
    "test_source_only.py",
    "verify_source.py",
}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                VerifyError(f"{label} nonfinite {token}")
            ),
        )
    except VerifyError:
        raise
    except Exception as error:
        raise VerifyError(f"{label} JSON: {error}") from error
    require(isinstance(value, dict), f"{label} object")
    return value


def read_regular(path: Path, *, maximum: int = 4 * (1 << 20)) -> bytes:
    require(path.is_absolute(), "source path absolute")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and 0 < before.st_size <= maximum,
            "regular single-link source",
        )
        output = bytearray()
        while len(output) < before.st_size:
            piece = os.read(descriptor, min(1 << 20, before.st_size - len(output)))
            require(bool(piece), "short source read")
            output.extend(piece)
        after = os.fstat(descriptor)
        require(
            (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_nlink,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_nlink,
            ),
            "source identity drift",
        )
        return bytes(output)
    finally:
        os.close(descriptor)


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "real package directory")
    entries = list(os.scandir(root))
    require(
        all(not entry.is_symlink() and entry.is_file(follow_symlinks=False) for entry in entries),
        "package contains files only",
    )
    require(
        {entry.name for entry in entries} == EXPECTED_MEMBERS | {"SOURCE_MANIFEST.json"},
        "exact package member set",
    )

    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json")
    manifest_sha = digest(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(manifest_sha == expected_manifest_sha256, "expected manifest SHA-256")
    manifest = strict_json(manifest_payload, "source manifest")
    require(
        set(manifest)
        == {
            "schema",
            "status",
            "source_snapshot_root_sha256",
            "members",
            "access_attestation",
            "claim_boundary",
        },
        "source manifest exact schema",
    )
    require(manifest["schema"] == SCHEMA, "source manifest schema")
    require(manifest["status"] == STATUS, "source manifest status")
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_MEMBERS), "manifest rows")
    require([row.get("name") for row in rows] == sorted(EXPECTED_MEMBERS), "manifest bytewise names")
    observed = []
    payloads: dict[str, bytes] = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "member row")
        name = row["name"]
        require(
            name in EXPECTED_MEMBERS
            and type(row["bytes"]) is int
            and row["bytes"] > 0
            and isinstance(row["sha256"], str)
            and len(row["sha256"]) == 64,
            f"member metadata {name}",
        )
        payload = read_regular(root / name)
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"], f"member closure {name}")
        observed.append({"name": name, "bytes": len(payload), "sha256": digest(payload)})
        payloads[name] = payload
    root_sha = digest(canonical_json(observed))
    require(root_sha == manifest["source_snapshot_root_sha256"], "source snapshot root")

    access = manifest["access_attestation"]
    require(
        isinstance(access, dict)
        and set(access)
        == {
            "model_or_qwen_payload_opened_statted_hashed_or_enumerated",
            "completed_v9_result_opened_statted_hashed_or_enumerated",
            "failed_v0_result_opened_statted_hashed_or_enumerated",
            "qwen_bf16_source_opened_statted_hashed_or_enumerated",
            "gaussian_control_opened_statted_hashed_or_enumerated",
            "cuda_initialized_during_source_build_or_tests",
            "isolated_source_only_tests_run",
        },
        "access attestation schema",
    )
    require(
        all(value is False for key, value in access.items() if key != "isolated_source_only_tests_run")
        and access["isolated_source_only_tests_run"] is True,
        "source-only access attestation",
    )
    require(
        manifest["claim_boundary"]
        == "source mechanics and synthetic regression only; no Qwen posterior result or universal SwiGLU-MoE claim",
        "claim boundary",
    )

    design = strict_json(payloads["design_lock.json"], "design lock")
    require(design["schema"] == "uwfa-sc-posterior-centroid-design-v1", "design schema")
    require(design["predecessor"]["predecessor_is_modified"] is False, "predecessor immutable")
    require(design["predecessor"]["failed_v0_run_is_reused"] is False, "v0 result not reused")
    require(design["ordinal_bridge"]["exact_v8_unpack_regression_required"] is True, "unpack regression")
    require(design["controls"]["matched_gaussian_source_pipelines_required"] == 8, "matched controls")
    require(
        design["launch_interlock"]["independent_predecessor_result_audit_must_pass_before_qwen_publication_access"] is True,
        "audit-before-Qwen interlock",
    )
    bridge_source = payloads["result_bridge.py"].decode("utf-8")
    for fragment in (
        "class PythonIntOrdinalStrataBridge",
        "operator.index(value)",
        "sorted(covered) == list(range(groups))",
        "authenticated_strata = _load_module",
    ):
        require(fragment in bridge_source, f"ordinal bridge boundary {fragment}")
    diagnostic_source = payloads["diagnostic.py"].decode("utf-8")
    require(
        diagnostic_source.index("authenticate_launch_preconditions(arguments, package_closure)")
        < diagnostic_source.index("bridge.authenticate_result_directory(Path(arguments.v9_result_dir))"),
        "audit/review precede Qwen publication access",
    )
    require("bind_publication_to_audit(publication, launch)" in diagnostic_source, "audited publication binding")
    test_source = payloads["test_source_only.py"].decode("utf-8")
    require(
        "test_numpy_int64_regression_exercises_exact_v8_unpack_route" in test_source
        and "with self.assertRaisesRegex(ValueError, \"STRATA group ordinal\")" in test_source,
        "exact v8 unpack regression present",
    )
    return {
        "schema": "uwfa-sc-posterior-centroid-v1-source-verifier-receipt",
        "status": "PASS_SOURCE_ONLY",
        "source_manifest_sha256": manifest_sha,
        "source_snapshot_root_sha256": root_sha,
        "members": len(rows),
        "qwen_payload_accessed": False,
        "failed_v0_result_reused": False,
        "cuda_initialized": False,
        "claim_authority": "SOURCE_MECHANICS_ONLY",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--package", type=Path, required=True)
    result.add_argument("--manifest-sha256")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(
        json.dumps(
            verify(arguments.package, arguments.manifest_sha256),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
