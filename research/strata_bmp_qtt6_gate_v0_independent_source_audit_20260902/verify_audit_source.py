#!/usr/bin/env python3
"""Verify the frozen independent-audit source inventory without payload access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_regular(path: Path, label: str) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            f"{label} regular non-link")
    payload = path.read_bytes()
    after = path.lstat()
    require(
        (before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino) ==
        (after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino),
        f"{label} changed during read",
    )
    return payload


def strict_json(payload: bytes) -> dict:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate manifest key")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"), object_pairs_hook=hook,
        parse_constant=lambda token: (_ for _ in ()).throw(
            VerifyError(f"nonfinite {token}")),
    )
    require(isinstance(value, dict), "manifest object")
    return value


def root_hash(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["name"]):
        digest.update(row["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify(package: Path, expected_manifest_sha256: str | None) -> dict:
    root = package.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "audit real directory")
    manifest_payload = read_regular(root / "AUDIT_SOURCE_MANIFEST.json", "manifest")
    manifest_sha = sha256(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(HEX64.fullmatch(expected_manifest_sha256) is not None and
                manifest_sha == expected_manifest_sha256,
                "external audit manifest pin")
    manifest = strict_json(manifest_payload)
    require(set(manifest) == {
        "schema", "status", "producer_source_manifest_sha256",
        "producer_source_root_sha256", "audit_source_root_sha256", "members",
        "execution_attestation", "claim_boundary",
    }, "manifest exact schema")
    require(manifest["schema"] == "strata-bmp-qtt6-independent-audit-source-v0",
            "manifest schema")
    require(manifest["execution_attestation"] == {
        "hostile_tests_executed": False,
        "n4096_fixture_executed": False,
        "real_cupy_audit_executed": False,
        "payloads_accessed": False,
        "runtime_pass_claimed": False,
    }, "unexecuted attestation")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "member rows")
    observed = []
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and
                set(row) == {"name", "bytes", "sha256"}, "member schema")
        name = row["name"]
        require(isinstance(name, str) and name == Path(name).name and
                name not in seen and name != "AUDIT_SOURCE_MANIFEST.json",
                "flat unique member")
        payload = read_regular(root / name, f"member {name}")
        actual = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(actual == row, f"member pin {name}")
        observed.append(actual)
        seen.add(name)
    require(manifest["audit_source_root_sha256"] == root_hash(observed),
            "audit source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} ==
            seen | {"AUDIT_SOURCE_MANIFEST.json"}, "exact audit closure")
    require(all(entry.is_file(follow_symlinks=False) for entry in entries),
            "regular audit closure")
    return {
        "schema": "strata-bmp-qtt6-independent-audit-source-verification-v0",
        "status": "PASS_EXACT_UNEXECUTED_AUDIT_SOURCE__HOLD_RUNTIME_AND_PAYLOAD",
        "audit_source_manifest_sha256": manifest_sha,
        "audit_source_root_sha256": manifest["audit_source_root_sha256"],
        "members": observed,
        "payloads_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    print(json.dumps(
        verify(args.package, args.expected_manifest_sha256),
        indent=2, sort_keys=True,
    ))


if __name__ == "__main__":
    main()
