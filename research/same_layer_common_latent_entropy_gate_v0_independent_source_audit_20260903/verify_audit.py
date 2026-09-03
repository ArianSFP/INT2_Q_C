"""Stdlib-only verifier for this sealed independent review package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat

EXPECTED_PRODUCER_MANIFEST = "238ce67f3670566c277d7baac019a69227ccff05ae944f1561ff8a0d32b1bce9"
EXPECTED_PRODUCER_ROOT = "16cceafcfe06e1c2683c0e89048700edd47fda395a2a6d06a70cef19d8eb858b"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path, expected_manifest_sha256: str) -> dict:
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "review root")
    require(len(expected_manifest_sha256) == 64 and all(c in "0123456789abcdef" for c in expected_manifest_sha256), "external manifest digest")
    manifest_path = root / "SOURCE_MANIFEST.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "review manifest")
    require(sha(manifest_path) == expected_manifest_sha256, "review manifest digest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    require(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(), "review manifest canonical encoding")
    require(manifest.get("schema") == "same_layer_common_latent_independent_audit_manifest_v0", "review manifest schema")
    rows = manifest["files"]
    names = [row["name"] for row in rows]
    require(names == sorted(names) and len(names) == len(set(names)), "review member order")
    require(sorted(path.name for path in root.iterdir()) == sorted(names + ["SOURCE_MANIFEST.json"]), "review exact closure")
    canonical_rows = []
    for row in rows:
        require(set(row) == {"bytes", "name", "sha256"}, "review row fields")
        path = root / row["name"]
        require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), "review regular member")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], f"review member mismatch: {path.name}")
        canonical_rows.append({"bytes": row["bytes"], "name": row["name"], "sha256": row["sha256"]})
    audit_root = hashlib.sha256(json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(audit_root == manifest["audit_root_sha256"], "review root")
    receipt = json.loads((root / "AUDIT_RECEIPT.json").read_text(encoding="utf-8"))
    require(receipt["schema"] == "same_layer_common_latent_independent_source_audit_receipt_v0", "receipt schema")
    require(receipt["status"] == "BLOCKED_MATERIAL_READ_GATE_DEFECT", "receipt verdict")
    require(receipt["source_manifest_sha256"] == EXPECTED_PRODUCER_MANIFEST, "producer manifest pin")
    require(receipt["source_root_sha256"] == EXPECTED_PRODUCER_ROOT, "producer root pin")
    require(receipt["payload_accessed"] is False and receipt["source_files_modified"] is False, "review boundary")
    require(receipt["material_defect"]["eligible_for_finite_coder_research_returned"] is True and receipt["material_defect"]["all_four_read_envelopes_failed"] is True, "defect regression evidence")
    require(receipt["authorization"] == "DENY_PAYLOAD_DEPLOYMENT_FROM_THIS_SOURCE_SNAPSHOT", "authorization verdict")
    cupy = json.loads((root / "CUPY_PARITY_RECEIPT.json").read_text(encoding="utf-8"))
    require(cupy["status"] == "PASS_SOURCE_FREE_CPU_CUPY_PARITY" and cupy["manifest_sha256"] == EXPECTED_PRODUCER_MANIFEST and cupy["payload_accessed"] is False, "CuPy receipt")
    return {
        "schema": "same_layer_common_latent_independent_audit_verification_v0",
        "status": "PASS_SEALED_NEGATIVE_SOURCE_REVIEW",
        "manifest_sha256": expected_manifest_sha256,
        "audit_root_sha256": audit_root,
        "producer_manifest_sha256": EXPECTED_PRODUCER_MANIFEST,
        "verdict": receipt["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(Path(args.package).resolve(), args.manifest_sha256), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
