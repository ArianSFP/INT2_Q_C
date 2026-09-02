#!/usr/bin/env python3
"""Standard-library closure verifier for the independent audit package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


EXPECTED = {
    "README.md", "audit_contract.json", "run_cupy_identity_audit.py",
    "test_independent_source.py", "verify_audit.py",
}
SCHEMA = "tactic-ramanujan384-independent-source-audit-manifest-v0"
STATUS = "HOLD_PAYLOAD_AUTHORITY_PENDING_LITERAL_WEIGHT_REPLAY_AND_BACKEND_STABLE_CONTROLS"
PRODUCER_MANIFEST = "287b8ad4c377956c9bb264d9d8731893a83e45180f75472f9b42968e3f20acde"
PRODUCER_ROOT = "2a66a5d745fc0a31e311cf6ab5f44836726ae341db977bca8eac314df61124ad"
PRODUCER_OWN_VERIFIER_ROOT = "64669f3eeb9dd4f34a9fa36c9c6db592dcf5e37bdeb5ce149b3dbd51e2e24733"


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_rows_digest(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("ascii")
    return digest(payload)


def strict_json(payload: bytes, label: str) -> dict[str, object]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    result = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                        parse_constant=lambda token: (_ for _ in ()).throw(VerifyError(token)))
    require(isinstance(result, dict), f"{label} object")
    return result


def read_regular(path: Path, maximum: int = 4 << 20) -> bytes:
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_BINARY", 0)
                         | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and 0 < before.st_size <= maximum, "regular single-link audit source")
        payload = bytearray()
        while len(payload) < before.st_size:
            row = os.read(descriptor, min(1 << 20, before.st_size - len(payload)))
            require(bool(row), "short audit source read")
            payload.extend(row)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink),
                "audit source identity drift")
        return bytes(payload)
    finally:
        os.close(descriptor)


def verify(package: Path, expected_manifest: str | None) -> dict[str, object]:
    root = package.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "real audit directory")
    entries = list(os.scandir(root))
    require(all(entry.is_file(follow_symlinks=False) and not entry.is_symlink() for entry in entries),
            "audit files only")
    require({entry.name for entry in entries} == EXPECTED | {"SOURCE_MANIFEST.json"},
            "exact audit member set")
    manifest_payload = read_regular(root / "SOURCE_MANIFEST.json")
    manifest_sha = digest(manifest_payload)
    if expected_manifest is not None:
        require(manifest_sha == expected_manifest, "expected audit manifest digest")
    manifest = strict_json(manifest_payload, "audit manifest")
    require(set(manifest) == {"schema", "status", "source_root_sha256", "producer",
                             "members", "access_attestation", "execution"},
            "audit manifest schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "audit manifest identity")
    require(manifest["producer"] == {
        "source_manifest_sha256": PRODUCER_MANIFEST,
        "source_root_sha256": PRODUCER_ROOT,
    }, "producer pins")
    rows = manifest["members"]
    require(isinstance(rows, list) and [row.get("name") for row in rows] == sorted(EXPECTED),
            "canonical audit rows")
    observed = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "audit member row")
        payload = read_regular(root / row["name"])
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"],
                f"audit member closure {row['name']}")
        observed.append({"name": row["name"], "bytes": len(payload), "sha256": digest(payload)})
    require(canonical_rows_digest(observed) == manifest["source_root_sha256"], "audit source root")
    producer = root.parent / "tactic_ramanujan384_adapter_v0"
    producer_manifest_payload = read_regular(producer / "SOURCE_MANIFEST.json")
    require(digest(producer_manifest_payload) == PRODUCER_MANIFEST, "producer manifest closure")
    producer_manifest = strict_json(producer_manifest_payload, "producer manifest")
    require(producer_manifest.get("source_root_sha256") == PRODUCER_ROOT, "producer root pin")
    producer_rows = []
    for row in producer_manifest.get("members", []):
        payload = read_regular(producer / row["name"])
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"],
                f"producer member closure {row['name']}")
        producer_rows.append({"name": row["name"], "bytes": len(payload), "sha256": digest(payload)})
    own_verifier_root = canonical_rows_digest(producer_rows)
    require(own_verifier_root == PRODUCER_OWN_VERIFIER_ROOT, "pinned producer verifier root")
    require(own_verifier_root != PRODUCER_ROOT, "producer root/verifier mismatch remains visible")
    contract = strict_json(read_regular(root / "audit_contract.json"), "audit contract")
    require(contract.get("status") == STATUS, "contract disposition")
    gaps = contract.get("authority_gaps")
    require(isinstance(gaps, dict) and gaps
            and all(value is False for value in gaps.values()), "fail-closed authority gaps")
    require(manifest["access_attestation"] == {
        "qwen_payload_accessed": False,
        "coarse_payload_accessed": False,
        "matched_control_payload_accessed": False,
        "network_accessed": False,
    }, "source-only access")
    require(manifest["execution"] == {
        "source_runtime_tests": "PENDING_PYTHON_NUMPY_ENVIRONMENT",
        "source_free_cpu_cupy_identity": "PENDING_CUPY_GPU_ENVIRONMENT",
        "payload_execution_authorized": False,
    }, "pending runtime boundary")
    return {
        "schema": "tactic-ramanujan384-independent-source-audit-verifier-receipt-v0",
        "status": STATUS,
        "audit_manifest_sha256": manifest_sha,
        "audit_source_root_sha256": manifest["source_root_sha256"],
        "producer_manifest_sha256": PRODUCER_MANIFEST,
        "producer_source_root_sha256": PRODUCER_ROOT,
        "producer_own_verifier_source_root_sha256": own_verifier_root,
        "members": len(rows),
        "payload_authorized": False,
        "qwen_payload_accessed": False,
        "network_accessed": False,
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
