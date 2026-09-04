"""Fail-closed source package verifier; never reads payloads or invokes GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    package = Path(args.package).resolve()
    manifest_path = package / "SOURCE_MANIFEST.json"
    if not package.is_dir() or not manifest_path.is_file():
        raise SystemExit("missing source package or manifest")
    if args.manifest_sha256 and sha256(manifest_path) != args.manifest_sha256.lower():
        raise SystemExit("manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cochain_q.source_manifest.v0":
        raise SystemExit("wrong manifest schema")
    for relative, expected in manifest["files_sha256"].items():
        path = package / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"source hash mismatch: {relative}")
    canonical_map = json.dumps(manifest["files_sha256"], sort_keys=True,
                               separators=(",", ":")).encode()
    root = hashlib.sha256(canonical_map).hexdigest()
    if root != manifest.get("source_root_sha256"):
        raise SystemExit("source root mismatch")
    receipt_path = package / "SOURCE_FREE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    emitted = subprocess.run(
        [sys.executable, "-I", "-B", str(package / "source_receipt.py")],
        check=True, capture_output=True, text=True,
    ).stdout
    if json.loads(emitted) != receipt:
        raise SystemExit("receipt regeneration mismatch")
    test_summary = None
    if args.self_test:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(package / "test_source.py")],
            check=True, capture_output=True, text=True,
        )
        test_summary = {
            "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            "stderr_tail": completed.stderr.strip().splitlines()[-4:],
        }
    print(json.dumps({
        "status": "PASS_SOURCE_ONLY_READY_FOR_INDEPENDENT_AUDIT",
        "manifest_sha256": sha256(manifest_path),
        "source_root_sha256": manifest["source_root_sha256"],
        "receipt_sha256": sha256(receipt_path),
        "self_test": test_summary,
        "authority": "NONE_SOURCE_ONLY",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
