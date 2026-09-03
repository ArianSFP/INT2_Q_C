"""Stdlib-only closure verifier for the sealed CBIB-1 r3 PASS review."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat


R3_MANIFEST = "5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f"
R3_ROOT = "ae1ae10c0a39b8167739498db13b69aaaf738a17b247ca7861c5d76a57d6c1ee"
STATUS = "PASS_R3_AUTHORIZE_ONE_SOURCE_FREE_RTX5090_PREFLIGHT_ONLY"
AUTHORITY_SHA = "ff38022186f39d610a9108e0926be0263b4fad065146725e9311a637219371a2"
WRAPPER_SHA = "91d55282c1d84b3e108e54c9772267a234cba7aeb8784504183badc90b25d71e"
EVIDENCE_SHA = "5b0f0fe567db43fe14ecf53c1f883945a95d71b886aee81d1f0510e1b134ae84"


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-package", required=True)
    parser.add_argument("--review-manifest-sha256", required=True)
    args = parser.parse_args()
    root = Path(args.review_package).resolve(strict=True)
    need(root.is_dir() and not root.is_symlink(), "real review root")
    need(len(args.review_manifest_sha256) == 64 and
         all(c in "0123456789abcdef" for c in args.review_manifest_sha256),
         "lowercase external manifest")
    manifest_path = root / "SOURCE_MANIFEST.json"
    need(sha(manifest_path) == args.review_manifest_sha256, "external review manifest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    need(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         "canonical review manifest")
    need(manifest.get("schema") ==
         "same-layer-clustered-ib-qwen-deployment-r3-independent-review-manifest-v0" and
         manifest.get("audited_deployment_manifest_sha256") == R3_MANIFEST and
         manifest.get("audited_deployment_root_sha256") == R3_ROOT and
         manifest.get("status") == STATUS, "review pins")
    rows = manifest.get("files")
    need(isinstance(rows, list) and rows, "manifest rows")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)), "row order")
    need(sorted(path.name for path in root.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "review closure")
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
    review_root = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    need(review_root == manifest.get("source_root_sha256"), "review source root")
    need(sha(root / "AUTHORIZED_PREFLIGHT.json") == AUTHORITY_SHA and
         sha(root / "run_authorized_preflight_once.py") == WRAPPER_SHA and
         sha(root / "CPU_TARGETED_EVIDENCE.json") == EVIDENCE_SHA,
         "authority/wrapper/evidence digests")
    receipt = json.loads((root / "AUDIT_RECEIPT.json").read_bytes())
    authority = json.loads((root / "AUTHORIZED_PREFLIGHT.json").read_bytes())
    evidence = json.loads((root / "CPU_TARGETED_EVIDENCE.json").read_bytes())
    need(receipt["status"] == STATUS and
         receipt["audited_deployment"]["manifest_sha256"] == R3_MANIFEST and
         receipt["audited_deployment"]["source_root_sha256"] == R3_ROOT and
         receipt["authorization"]["authorized_source_free_rtx5090_preflight_attempts"] == 1 and
         receipt["authorization"]["authorized_qwen_invocations"] == 0 and
         receipt["authorization"]["authorized_payload_file_reads"] == 0 and
         receipt["authorization"]["authorized_capability_or_production_launcher_invocations"] == 0 and
         receipt["review_process"]["qwen_accessed"] is False and
         receipt["review_process"]["payload_files_opened"] == 0,
         "PASS receipt semantics")
    need(authority["status"] == "AUTHORIZED_NOT_STAGED_NOT_EXECUTED" and
         authority["permitted_attempts"] == 1 and authority["command"]["shell"] is False and
         authority["command"]["cwd"] ==
         "/tmp/codex_cbib1_r3_source_free_rtx5090_preflight_20260903_5bac3594" and
         authority["forbidden"]["authorized_qwen_invocations"] == 0 and
         authority["forbidden"]["authorized_payload_file_reads"] == 0 and
         authority["forbidden"]["authorized_capability_or_production_launcher_invocations"] == 0,
         "narrow one-attempt authority")
    need(evidence["fixture_probe"]["scale_bytes_per_expert"] == 256 and
         evidence["fixture_probe"]["fixture_labels_sha256"] ==
         "33f7ba9d4ae0589d06abcfab06bac46d06ef75188d714350ba993df0ca9bbab5" and
         evidence["sealed_targeted_regression_reproduction"]["status"] ==
         "PASS_TARGETED_GROUP2_SOURCE_AND_STRICT_READ_SURVIVOR" and
         evidence["sealed_targeted_regression_reproduction"]["read_envelopes"]["5/2"]
         ["max_amplification"] == 1.9651249492746525, "targeted evidence")
    print(json.dumps({
        "audited_deployment_manifest_sha256": R3_MANIFEST,
        "audited_deployment_root_sha256": R3_ROOT,
        "authorized_capability_or_production_launcher_invocations": 0,
        "authorized_qwen_invocations": 0,
        "authorized_source_free_rtx5090_preflight_attempts": 1,
        "payload_accessed": False,
        "review_manifest_sha256": sha(manifest_path),
        "review_source_root_sha256": review_root,
        "schema": "same-layer-clustered-ib-qwen-deployment-r3-independent-review-verification-v0",
        "status": STATUS,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
