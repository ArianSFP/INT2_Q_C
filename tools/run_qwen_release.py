#!/usr/bin/env python3
"""Repository-layout wrapper around the hash-pinned Qwen panel runner."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--variant", choices=("exact", "rht"), required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--polar-repo", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--manifest", type=Path, default=root / "results/qwen/manifest.json"
    )
    args = parser.parse_args()
    workdir = args.workdir if args.workdir.is_absolute() else root / args.workdir
    decoder_map = root / "codec_data/polaris_sc_v1_decoder_map.npz"
    if not decoder_map.is_file():
        raise FileNotFoundError(
            f"missing locally generated decoder map: {decoder_map}; run "
            "tools/build_decoder_map.py as documented"
        )
    observed_map_hash = sha256(decoder_map)
    if observed_map_hash != EXPECTED_DECODER_MAP_SHA256:
        raise AssertionError(f"decoder map SHA-256 mismatch: {observed_map_hash}")
    command = [
        str(args.python),
        str(root / "tools/run_qwen_panel.py"),
        "--workspace", str(root),
        "--manifest", str(args.manifest.resolve()),
        "--workdir", str(workdir.resolve()),
        "--variant", args.variant,
        "--workers", str(args.workers),
        "--python", str(args.python),
        "--polar-repo", str(args.polar_repo.resolve()),
        "--exact-encoder", str(root / "src/polaris_sc_v2_encoder.py"),
        "--rht-encoder", str(root / "src/polaris_sc_v2_rht_encoder.py"),
        "--packer", str(root / "src/reservoir_pack_v2.py"),
        "--unpacker", str(root / "src/reservoir_unpack_v2.py"),
        "--decoder", str(root / "src/qwen_reservoir_decode.py"),
        "--decoder-map", str(decoder_map),
    ]
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == "__main__":
    main()
