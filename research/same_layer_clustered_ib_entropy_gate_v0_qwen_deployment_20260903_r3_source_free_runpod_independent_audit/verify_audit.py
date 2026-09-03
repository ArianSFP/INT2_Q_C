"""Stdlib-only verifier for the sealed independent RunPod evidence audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat


SCHEMA = "same-layer-clustered-ib-r3-source-free-runpod-independent-audit-manifest-v0"
STATUS = "PASS_AUTHORIZED_SINGLE_PREFLIGHT_RECEIPT_INDEPENDENTLY_AUDITED"


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-package", required=True, type=Path)
    parser.add_argument("--audit-manifest-sha256", required=True)
    args = parser.parse_args()
    root = args.audit_package.resolve(strict=True)
    need(root.is_dir() and not root.is_symlink(), "real audit root")
    manifest_path = root / "SOURCE_MANIFEST.json"
    need(sha(manifest_path) == args.audit_manifest_sha256, "external manifest digest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    need(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "canonical manifest")
    need(manifest.get("schema") == SCHEMA and manifest.get("status") == STATUS,
         "manifest identity")
    rows = manifest.get("files")
    need(isinstance(rows, list) and rows, "manifest rows")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)), "manifest ordering")
    need(sorted(path.name for path in root.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "audit closure")
    normalized = []
    for row in rows:
        need(set(row) == {"bytes", "name", "sha256"}, "row fields")
        member = root / row["name"]
        need(stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink(),
             "regular nonsymlink member")
        need(member.stat().st_size == int(row["bytes"]) and sha(member) == row["sha256"],
             f"member mismatch: {row['name']}")
        normalized.append({"bytes": int(row["bytes"]), "name": row["name"],
                           "sha256": row["sha256"]})
    observed_root = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    need(observed_root == manifest.get("source_root_sha256"), "audit source root")
    receipt = json.loads((root / "AUDIT_RECEIPT.json").read_bytes())
    lock = json.loads((root / "EVIDENCE_LOCK.json").read_bytes())
    need(receipt.get("status") == STATUS and
         receipt["subject"]["receipt_sha256"] ==
         "38a4eb497983aa8b5a559fa96fcbcbb11dc77cdd78dabfff9d2d4d06c5bf1913" and
         receipt["one_use_chain"]["preflight_attempts_remaining"] == 0 and
         receipt["access"]["payload_or_qwen_accessed"] is False and
         receipt["access"]["run_gate_invoked"] is False,
         "audit receipt semantics")
    need(lock["remote_execution_claim"]["attempts_remaining"] == 0 and
         len(lock["members"]) == 4, "evidence lock semantics")
    print(json.dumps({
        "audit_manifest_sha256": sha(manifest_path),
        "audit_source_root_sha256": observed_root,
        "payload_or_qwen_accessed": False,
        "preflight_attempts_remaining": 0,
        "receipt_sha256": receipt["subject"]["receipt_sha256"],
        "schema": "same-layer-clustered-ib-r3-source-free-runpod-independent-audit-verification-v0",
        "status": STATUS,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
