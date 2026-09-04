"""Reproduce the hostile audit and compare its parsed result exactly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    expected = json.loads((ROOT / "HOSTILE_AUDIT_RESULT.json").read_text(encoding="utf-8"))
    process = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / "hostile_audit.py")],
        check=True, capture_output=True, text=True,
    )
    observed = json.loads(process.stdout)
    if observed != expected:
        raise SystemExit("hostile audit regeneration mismatch")
    print(json.dumps({
        "status": "PASS_REPRODUCED_BLOCKING_HOSTILE_AUDIT",
        "verdict": observed["verdict"],
        "target_manifest_sha256": observed["target_closure"]["manifest_sha256"],
        "target_source_root_sha256": observed["target_closure"]["source_root_sha256"],
        "result_sha256": sha256(ROOT / "HOSTILE_AUDIT_RESULT.json"),
        "qwen_payload_opened": False,
        "gpu_accessed": False,
        "network_accessed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
