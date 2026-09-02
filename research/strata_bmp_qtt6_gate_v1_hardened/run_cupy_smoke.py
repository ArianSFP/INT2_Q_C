#!/usr/bin/env python3
"""Launch and authenticate the source-free CuPy search in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    worker = Path(__file__).resolve().with_name("cupy_worker.py")
    nonce = secrets.token_hex(32)
    command = [sys.executable, "-I", "-B", str(worker), "--nonce", nonce]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("worker stdout must contain exactly one JSON record")
    record = json.loads(lines[0])
    if record.get("nonce") != nonce or record.get("pid") == os.getpid():
        raise RuntimeError("fresh worker nonce/PID authentication")
    if not record.get("isolated_flag") or not record.get("dont_write_bytecode_flag"):
        raise RuntimeError("fresh worker must use -I -B")
    envelope = {
        "schema": "strata-bmp-qtt6-v1-fresh-cupy-launch-receipt",
        "launcher_pid": os.getpid(),
        "command_shape": ["PYTHON", "-I", "-B", "WORKER", "--nonce", "NONCE"],
        "worker_stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")).hexdigest(),
        "worker_stderr_sha256": hashlib.sha256(
            completed.stderr.encode("utf-8")).hexdigest(),
        "worker": record,
    }
    payload = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")


if __name__ == "__main__":
    main()
