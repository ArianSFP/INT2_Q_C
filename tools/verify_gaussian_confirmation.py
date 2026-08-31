#!/usr/bin/env python3
"""Safely run the frozen Gaussian confirmation from a pristine copy.

The original harness is intentionally bound to historical basenames and
creates a one-time opened lock.  This wrapper never executes it in the
archival tree: it first copies the complete frozen workspace to a destination
that must not exist, then forwards the canonical arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_DECODER_MAP_SHA256 = "a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--polar-repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--decoder-map",
        type=Path,
        default=root / "codec_data/polaris_sc_v1_decoder_map.npz",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    source = (root / "frozen/gaussian_confirmation").resolve(strict=True)
    workspace = args.workspace.resolve()
    if workspace.exists():
        raise FileExistsError(
            f"destination already exists: {workspace}; use a new path so the "
            "opened-lock protocol cannot be resumed or relabelled"
        )
    if args.workers < 1:
        raise ValueError("workers must be positive")
    polar_repo = args.polar_repo.resolve(strict=True)
    decoder_map = args.decoder_map.resolve(strict=True)
    observed_map_hash = sha256(decoder_map)
    if observed_map_hash != EXPECTED_DECODER_MAP_SHA256:
        raise AssertionError(f"decoder map SHA-256 mismatch: {observed_map_hash}")
    run_dir = args.run_dir.resolve()
    summary = args.summary.resolve()
    if run_dir.exists() or summary.exists():
        raise FileExistsError("run-dir and summary must both be new paths")

    shutil.copytree(source, workspace, copy_function=shutil.copy2)
    shutil.copy2(
        decoder_map,
        workspace / "agent_polaris_sc_v1_decoder_map.npz",
    )
    harness = workspace / "agent_polaris_confirmation_verify_v2.py"
    command = [
        str(args.python),
        str(harness),
        "--workspace",
        str(workspace),
        "--polar-repo",
        str(polar_repo),
        "--run-dir",
        str(run_dir),
        "--summary",
        str(summary),
        "--workers",
        str(args.workers),
    ]
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == "__main__":
    main()
