"""Stdlib-only verifier for the sealed independent CBIB-1 audit package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    root = Path(args.package).resolve()
    need(root.is_dir() and not root.is_symlink(), "real package directory required")
    manifest_path = root / "SOURCE_MANIFEST.json"
    need(sha(manifest_path) == args.manifest_sha256, "external manifest mismatch")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    need(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "noncanonical manifest")
    need(manifest["schema"] == "same-layer-clustered-ib-independent-source-audit-manifest-v0",
         "schema")
    rows = manifest["files"]
    need([r["name"] for r in rows] == sorted(r["name"] for r in rows), "member order")
    need(sorted(p.name for p in root.iterdir()) ==
         sorted([r["name"] for r in rows] + ["SOURCE_MANIFEST.json"]), "closure")
    normalized = []
    for row in rows:
        member = root / row["name"]
        need(stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink(), "member type")
        need(member.stat().st_size == row["bytes"] and sha(member) == row["sha256"],
             f"member mismatch: {row['name']}")
        normalized.append({"bytes": row["bytes"], "name": row["name"],
                           "sha256": row["sha256"]})
    source_root = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    need(source_root == manifest["source_root_sha256"], "source root")
    evidence = json.loads((root / "AUDIT_EVIDENCE.json").read_text())
    gpu = json.loads((root / "INDEPENDENT_LOCAL_RTX3060_RECEIPT.json").read_text())
    need(evidence["status"] == "PASS_INDEPENDENT_PAYLOAD_BLIND_SOURCE_AUDIT", "audit status")
    need(evidence["payload_accessed"] is False, "payload access")
    need(gpu["status"] == "PASS_SOURCE_FREE_CPU_CUPY_PARITY", "GPU status")
    need(gpu["model_or_qwen_payload_accessed"] is False, "GPU payload access")
    need(evidence["source_manifest_sha256"] == manifest["audited_source_manifest_sha256"],
         "audited manifest pin")
    need(evidence["source_root_sha256"] == manifest["audited_source_root_sha256"],
         "audited source-root pin")
    print(json.dumps({
        "schema": "same-layer-clustered-ib-independent-source-audit-verification-v0",
        "status": "PASS_SEALED_INDEPENDENT_SOURCE_AUDIT",
        "manifest_sha256": sha(manifest_path), "source_root_sha256": source_root,
        "member_count_excluding_manifest": len(rows),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
