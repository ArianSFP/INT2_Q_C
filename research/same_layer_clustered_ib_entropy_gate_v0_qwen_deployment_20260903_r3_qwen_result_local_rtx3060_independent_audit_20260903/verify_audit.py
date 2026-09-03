"""Fail-closed verifier for the sealed independent CBIB-r3 result audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat


SCHEMA = "cbib-r3-qwen-result-local-rtx3060-independent-audit-manifest-v0"
STATUS = (
    "PASS_COMPLETED_CHILD_RESULT_WITH_HARMLESS_STDERR_WARNING__"
    "HARD_KILL_CBIB_FIXED_LABEL"
)


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
         receipt.get("result", {}).get("result_sha256") ==
             "e24d8795c655704732a42b2fb6e39ca323c2cc09d0d0e5cf34a070de9ef5b916" and
         receipt.get("scientific_conclusion", {}).get("eligible_for_finite_codec") is False,
         "audit receipt semantics")
    need(lock.get("deployment", {}).get("source_root_sha256") ==
             "ae1ae10c0a39b8167739498db13b69aaaf738a17b247ca7861c5d76a57d6c1ee" and
         lock.get("capability", {}).get("source_root_sha256") ==
             "8f36ce5e3476e7058375622c2175b0624b27fb04904f4f1d70ac3baec3565fbf" and
         len(lock.get("preserved_execution_files", [])) == 6,
         "evidence lock semantics")
    print(json.dumps({
        "audit_manifest_sha256": sha(manifest_path),
        "audit_source_root_sha256": observed_root,
        "result_sha256": receipt["result"]["result_sha256"],
        "schema": "cbib-r3-qwen-result-local-rtx3060-independent-audit-verification-v0",
        "status": STATUS,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
