#!/usr/bin/env python3
"""Materialize and hash-check the frozen Qwen panel without full shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_MANIFEST_SHA256 = "3b882c74870c1e27bcddf7427e4c6ffea816d4f9847447eb218729ca69426a55"
EXPECTED_CHECKPOINT = "Qwen/Qwen3-30B-A3B"
EXPECTED_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
EXPECTED_BLOCKS = 32
EXPECTED_BLOCK_LENGTH = 1 << 18


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(
    workspace: Path,
    fetcher: Path,
    python: Path,
    block: dict[str, Any],
) -> dict[str, Any]:
    relative = Path(block["source_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe source path in manifest: {relative}")
    destination = (workspace / relative).resolve()
    if not destination.is_relative_to(workspace):
        raise ValueError(f"source path escapes workspace: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = str(block["source_bf16_sha256"])
    if destination.is_file() and sha256(destination) == expected:
        return {"id": block["id"], "status": "already_valid", "sha256": expected}
    if destination.exists():
        raise AssertionError(f"existing source has the wrong hash: {destination}")
    command = [
        str(python),
        str(fetcher),
        str(block["tensor"]),
        "--block-index",
        str(int(block["canonical_block_index"])),
        "--output-dir",
        str(destination.parent),
    ]
    process = subprocess.run(
        command,
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"fetch failed for {block['id']}:\n{process.stdout[-4000:]}")
    if not destination.is_file():
        raise FileNotFoundError(destination)
    observed = sha256(destination)
    if observed != expected:
        raise AssertionError(f"download hash mismatch for {block['id']}: {observed}")
    return {"id": block["id"], "status": "downloaded", "sha256": observed}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, default=root / "results/qwen/manifest.json")
    parser.add_argument("--fetcher", type=Path, default=root / "tools/fetch_qwen_block.py")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--audit",
        type=Path,
        default=root / "qwen_polaris_heldout32/source_materialization_audit.json",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise AssertionError(
            f"manifest hash {manifest_sha256} != {EXPECTED_MANIFEST_SHA256}"
        )
    manifest = json.loads(manifest_bytes)
    blocks = list(manifest["blocks"])
    if not (
        manifest.get("checkpoint") == EXPECTED_CHECKPOINT
        and manifest.get("revision") == EXPECTED_REVISION
        and int(manifest.get("block_length", -1)) == EXPECTED_BLOCK_LENGTH
        and len(blocks) == EXPECTED_BLOCKS
    ):
        raise AssertionError("manifest identity/shape does not match the frozen panel")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                materialize,
                workspace,
                args.fetcher.resolve(),
                args.python.resolve(),
                block,
            ): block
            for block in blocks
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{len(rows)}/{len(blocks)} {row['id']}: {row['status']}", flush=True)
    order = {block["id"]: index for index, block in enumerate(blocks)}
    rows.sort(key=lambda row: order[row["id"]])
    result = {
        "status": "all source blocks hash-valid",
        "manifest_sha256": manifest_sha256,
        "blocks": rows,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "blocks": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
