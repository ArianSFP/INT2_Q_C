"""Fail-closed source-only verifier; this is not a payload authority."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


EXPECTED = {
    "tetrapath4_oracle.py", "test_source.py", "verify_source.py", "README.md",
    "CHECKPOINT_STATUS.md", "DESIGN_LOCK.json", "RUN_DISABLED.txt",
}
BANNED_CODE = (
    "import socket", "import requests", "import torch", "import cupy",
    "subprocess.run([\"ssh\"", "huggingface_hub", "snapshot_download",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    names = {p.name for p in package.iterdir() if p.is_file()}
    if names != EXPECTED:
        raise SystemExit(f"FAIL source closure: {sorted(names)}")
    for name in ("tetrapath4_oracle.py", "test_source.py"):
        text = (package / name).read_text(encoding="utf-8")
        for banned in BANNED_CODE:
            if banned in text:
                raise SystemExit(f"FAIL banned code token {banned!r} in {name}")
    if "payload_execution_gate" not in (package / "tetrapath4_oracle.py").read_text(encoding="utf-8"):
        raise SystemExit("FAIL missing disabled payload gate")
    if args.self_test:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(package / "test_source.py")],
            cwd=package, check=False)
        if completed.returncode:
            raise SystemExit("FAIL source tests")
    print("PASS_UNSEALED_SOURCE_ONLY_TETRAPATH4_NO_PAYLOAD_AUTHORITY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
