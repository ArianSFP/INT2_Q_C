"""Stdlib closure and arithmetic verifier for the independent result audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat


SCHEMA = "same_layer_common_latent_qwen_result_independent_audit_manifest_v0"


def req(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    package = Path(args.package).resolve()
    req(package.is_absolute() and package.is_dir() and not package.is_symlink(), "package root")
    manifest_path = package / "SOURCE_MANIFEST.json"
    req(sha256(manifest_path) == args.manifest_sha256, "manifest SHA-256")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    req(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(), "canonical manifest")
    req(manifest.get("schema") == SCHEMA, "manifest schema")
    rows = manifest.get("files")
    req(isinstance(rows, list) and rows, "manifest rows")
    names = [row.get("name") for row in rows]
    req(names == sorted(names) and len(names) == len(set(names)), "member ordering")
    req(sorted(item.name for item in package.iterdir()) == sorted(names + ["SOURCE_MANIFEST.json"]), "closure")
    canonical_rows = []
    for row in rows:
        path = package / row["name"]
        req(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), f"unsafe member: {path}")
        req(path.stat().st_size == int(row["bytes"]), f"member bytes: {path}")
        req(sha256(path) == row["sha256"], f"member hash: {path}")
        canonical_rows.append({"bytes": int(row["bytes"]), "name": row["name"], "sha256": row["sha256"]})
    root_hash = hashlib.sha256(
        json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    req(root_hash == manifest.get("audit_root_sha256"), "audit root")

    module_path = package / "audit_result.py"
    spec = importlib.util.spec_from_file_location("sealed_same_layer_qwen_auditor", module_path)
    req(spec is not None and spec.loader is not None, "auditor import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recomputed = module.audit(package.parent)
    receipt_path = package / "AUDIT_RECEIPT.json"
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    canonical_receipt = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    req(receipt_raw == canonical_receipt, "canonical receipt")
    req(recomputed == receipt, "receipt does not match independent recomputation")
    req(receipt["status"] == "PASS_INTERNAL_MATH_CONFIRMS_HARD_KILL", "verdict")
    req(receipt["payload_accessed"] is False and receipt["payload_files_opened"] == 0, "payload boundary")
    print(json.dumps({
        "schema": "same_layer_common_latent_qwen_result_independent_audit_verification_v0",
        "status": "PASS_SEALED_INTERNAL_MATH_HARD_KILL_AUDIT",
        "manifest_sha256": args.manifest_sha256,
        "audit_root_sha256": root_hash,
        "result_sha256": receipt["dependencies"]["result_sha256"],
        "payload_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
