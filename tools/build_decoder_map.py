#!/usr/bin/env python3
"""Generate the frozen decoder set map from a licensed upstream checkout.

The upstream reliability tables are not redistributed here because the pinned
repository exposes no explicit license.  This builder reads them from a
separately obtained checkout, emits the compact runtime map, and refuses any
result that is not byte-identical to the map used by the published run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
N = 1 << 18
LEVELS = 6
SIGMA = 1.0
TEST_DISTORTION = 0.05110
ETA = 0.25
ALPHABET_SIZE = 64
UPSTREAM_COMMIT = "458187b9b03db1768a4b72d617e591f7862f6fca"
EXPECTED_ENCODER_SHA256 = "95cfd32e5d026f07ceffe90daa7f88ca5e62f9f90546dfe74fc37cf06854d9b8"
EXPECTED_MAP_SHA256 = "a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoder() -> Any:
    path = ROOT / "src/polaris_sc_v2_encoder.py"
    observed = sha256(path)
    if observed != EXPECTED_ENCODER_SHA256:
        raise AssertionError(f"encoder SHA-256 mismatch: {observed}")
    spec = importlib.util.spec_from_file_location("decoder_map_encoder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen encoder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_upstream_commit(repo: Path) -> None:
    process = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"cannot inspect upstream checkout: {process.stderr.strip()}")
    observed = process.stdout.strip()
    if observed != UPSTREAM_COMMIT:
        raise AssertionError(f"upstream commit {observed} != {UPSTREAM_COMMIT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polar-repo", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "codec_data/polaris_sc_v1_decoder_map.npz",
    )
    args = parser.parse_args()
    repo = args.polar_repo.resolve(strict=True)
    output = args.output.resolve()
    require_upstream_commit(repo)
    if output.exists():
        observed = sha256(output)
        if observed == EXPECTED_MAP_SHA256:
            print(json.dumps({"status": "already valid", "path": str(output), "sha256": observed}, indent=2))
            return
        raise FileExistsError(f"refusing to overwrite wrong-hash output: {output}")

    polar = load_encoder()
    sigma_recon = math.sqrt(SIGMA * SIGMA - TEST_DISTORTION)
    tilde_sigma = sigma_recon * math.sqrt(TEST_DISTORTION) / SIGMA
    capacities = [
        polar.periodic_binary_capacity(tilde_sigma / ETA / (1 << level))
        for level in range(LEVELS)
    ]
    flags = polar.reliability_freeze_flags(repo, N, capacities)
    packed = np.stack(
        [np.packbits(flag, bitorder="little") for flag in flags]
    ).astype(np.uint8)
    counts = [int(np.count_nonzero(flag == 0)) for flag in flags]

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".npz", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(
            temporary,
            packed_freeze_flags=packed,
            block_length=np.asarray(N, dtype=np.int32),
            levels=np.asarray(LEVELS, dtype=np.int16),
            bitorder=np.asarray("little"),
            sigma_source=np.asarray(SIGMA, dtype=np.float64),
            test_distortion=np.asarray(TEST_DISTORTION, dtype=np.float64),
            eta=np.asarray(ETA, dtype=np.float64),
            alphabet_size=np.asarray(ALPHABET_SIZE, dtype=np.int16),
            capacity_schedule=np.asarray(capacities, dtype=np.float64),
            selected_counts=np.asarray(counts, dtype=np.int32),
            frozen_seed_rule=np.asarray(
                "seed + 104729*block_index + 1000003*level"
            ),
            polar_repository_commit=np.asarray(UPSTREAM_COMMIT),
        )
        observed = sha256(temporary)
        if observed != EXPECTED_MAP_SHA256:
            raise AssertionError(
                f"generated map SHA-256 {observed} != {EXPECTED_MAP_SHA256}; "
                "check Python/NumPy/SciPy versions and upstream bytes"
            )
        unpacked = [
            np.unpackbits(packed[level], bitorder="little")[:N].astype(np.uint8)
            for level in range(LEVELS)
        ]
        if not all(np.array_equal(left, right) for left, right in zip(flags, unpacked)):
            raise AssertionError("packed map round trip failed")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        json.dumps(
            {
                "status": "generated and hash-verified",
                "path": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256(output),
                "upstream_commit": UPSTREAM_COMMIT,
                "selected_counts": counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
