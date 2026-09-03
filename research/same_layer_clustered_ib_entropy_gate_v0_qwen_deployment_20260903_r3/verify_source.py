"""Stdlib-only closure verifier for the repaired CBIB-1 r3 deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat


SCHEMA = "same-layer-clustered-ib-qwen-deployment-manifest-v0-r3"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(package: Path, expected_manifest_sha256: str) -> dict:
    package = Path(package).resolve(strict=True)
    manifest_path = package / "SOURCE_MANIFEST.json"
    if _sha(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("external manifest SHA-256 mismatch")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    canonical = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if raw != canonical or manifest.get("schema") != SCHEMA:
        raise RuntimeError("noncanonical or wrong manifest schema")
    rows = manifest.get("files")
    names = [row.get("name") for row in rows]
    if names != sorted(names) or len(names) != len(set(names)):
        raise RuntimeError("manifest member ordering")
    actual = sorted(path.name for path in package.iterdir())
    if actual != sorted(names + ["SOURCE_MANIFEST.json"]):
        raise RuntimeError("package closure mismatch")
    normalized = []
    for row in rows:
        path = package / row["name"]
        if not stat.S_ISREG(path.lstat().st_mode) or path.is_symlink():
            raise RuntimeError(f"nonregular closure member: {row['name']}")
        size = int(row["bytes"])
        if path.stat().st_size != size or _sha(path) != row["sha256"]:
            raise RuntimeError(f"closure member mismatch: {row['name']}")
        normalized.append(
            {"bytes": size, "name": row["name"], "sha256": row["sha256"]}
        )
    root = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if root != manifest.get("source_root_sha256"):
        raise RuntimeError("source root mismatch")
    if manifest.get("parent_source_manifest_sha256") != (
        "1d07f1c0a057db3ba74f91062c06d39c23e39f5dd3da74a373c790345a9e7a9a"
    ):
        raise RuntimeError("parent source pin")
    if manifest.get("parent_audit_manifest_sha256") != (
        "5c07e720928f2642867524b201d0abef5a17ea57b4cae68f5c0df59010e3f051"
    ):
        raise RuntimeError("parent audit pin")
    return {
        "schema": "same-layer-clustered-ib-qwen-deployment-verification-v0-r3",
        "status": "PASS_SOURCE_CLOSED_R3_REQUIRES_NEW_INDEPENDENT_REVIEW",
        "manifest_sha256": expected_manifest_sha256,
        "source_root_sha256": root,
        "member_count": len(rows),
        "payload_or_qwen_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(
        verify(args.package, args.manifest_sha256),
        sort_keys=True, separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
