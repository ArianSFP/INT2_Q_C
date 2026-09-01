#!/usr/bin/env python3
"""Standard-library source-only verifier; never imports or runs CuPy code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path


PACKAGE_FILES = {
    "README.md",
    "SOURCE_MANIFEST.json",
    "design_lock.json",
    "meta_codebook_stage0.py",
    "verify_source.py",
}
SOURCE_LOCK_RELPATH = Path("blind_protocol_v2/unblinded/source_hashes.lock.json")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = Path(__file__).resolve().parent
    observed = {row.name for row in package.iterdir()}
    require(observed == PACKAGE_FILES, f"package closure mismatch: {sorted(observed)}")
    for name in PACKAGE_FILES:
        require(not (package / name).is_symlink(), f"symlink forbidden: {name}")
        require((package / name).is_file(), f"non-file member: {name}")

    manifest = json.loads((package / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["schema"] == "meta-codebook-stage0-source-manifest-v0", "manifest schema")
    require(set(manifest["closure"]) == PACKAGE_FILES, "manifest closure")
    for row in manifest["files"]:
        path = package / row["path"]
        require(path.stat().st_size == row["bytes"], f"byte mismatch: {row['path']}")
        require(sha256_file(path) == row["sha256"], f"hash mismatch: {row['path']}")

    lock = json.loads((package / "design_lock.json").read_text(encoding="utf-8"))
    require(lock["schema"] == "meta-codebook-whole-expert-stage0-design-lock-v0", "design schema")
    require(lock["panel"]["fit_slots"] == [0, 2, 3, 5], "fit split")
    require(lock["panel"]["holdout_slots"] == [1, 4], "holdout split")
    require(set(lock["panel"]["fit_slots"]).isdisjoint(lock["panel"]["holdout_slots"]), "split overlap")
    require(lock["panel"]["expert_count"] == 6 and lock["panel"]["matrix_count"] == 18, "panel cardinality")
    require(lock["panel"]["source_lock_relpath"] == str(SOURCE_LOCK_RELPATH).replace("\\", "/"), "source-lock relative path")
    require(lock["architecture"]["code_count"] == 32768, "code count")
    require(lock["architecture"]["index_bits_per_vector"] == 15, "index bits")
    require(lock["rate"]["global_side_bytes_including_zero_padding"] == 278528, "global side bytes")
    require(lock["rate"]["row_moments_fp16_bytes_per_expert"] == 9216, "row moments")
    require(lock["rate"]["index_bytes_per_expert"] == 1105920, "index bytes")

    values = 6 * 3 * 768 * 2048
    side = 278528
    header = 64
    moments = 9216
    index = 1105920
    prefix = (side * 8 + 6 * (header + moments + index) * 8) / values
    threshold = 0.8 / (2.0 ** (2.0 * prefix))
    require(math.isclose(prefix, 1.9694372106481481, rel_tol=0.0, abs_tol=1e-15), "prefix arithmetic")
    require(math.isclose(threshold, 0.05216397006684782, rel_tol=0.0, abs_tol=1e-15), "oracle threshold")
    expected_rate = {
        2.15: (7608730, 1221701, 639002, 1503232),
        2.30: (8139572, 1310174, 1169844, 1589248),
        2.50: (8847360, 1428139, 1877632, 1708032),
    }
    for rate, expected in expected_rate.items():
        total = math.ceil(rate * values / 8.0)
        local = total - side
        local_max = math.ceil(local / 6.0)
        residual = total - side - 6 * (index + moments + header)
        cold = side + 4096 * math.ceil(local_max / 4096.0)
        require((total, local_max, residual, cold) == expected, f"rate arithmetic {rate}")

    layer_values = 128 * 3 * 768 * 2048
    layer_prefix = (side * 8 + 128 * (header + moments + index) * 8) / layer_values
    layer_threshold = 0.8 / (2.0 ** (2.0 * layer_prefix))
    require(math.isclose(layer_prefix, 1.8944227430555556, rel_tol=0.0, abs_tol=1e-15), "layer prefix")
    require(math.isclose(layer_threshold, 0.05788070959170235, rel_tol=0.0, abs_tol=1e-15), "layer threshold")
    expected_layer_rate = {
        2.15: (162319565, 1265946, 19295437, 1548288),
        2.30: (173644186, 1354420, 30620058, 1634304),
        2.50: (188743680, 1472384, 45719552, 1753088),
    }
    for rate, expected in expected_layer_rate.items():
        total = math.ceil(rate * layer_values / 8.0)
        local = total - side
        local_max = math.ceil(local / 128.0)
        residual = total - side - 128 * (index + moments + header)
        cold = side + 4096 * math.ceil(local_max / 4096.0)
        require((total, local_max, residual, cold) == expected, f"layer rate arithmetic {rate}")

    source = (package / "meta_codebook_stage0.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    require(not imported.intersection({"requests", "urllib", "socket", "subprocess", "torch"}), "forbidden import")
    require("import cupy as cp" in source, "CuPy import missing")
    require("AUTHORIZATION" in source and "--authorization" in source, "execution interlock missing")
    require('SOURCE_LOCK_RELPATH = Path("blind_protocol_v2/unblinded/source_hashes.lock.json")' in source, "runner source-lock root")
    require("source_root = lock_path.parent.resolve()" in source, "payload root must be lock parent")
    require('path = (source_root / str(row["output_relpath"])).resolve()' in source, "payload resolution semantics")

    root = args.root.resolve()
    source_lock = (root / SOURCE_LOCK_RELPATH).resolve()
    require(source_lock.stat().st_size == lock["panel"]["source_lock_bytes"], "bound source-lock bytes")
    require(sha256_file(source_lock) == lock["panel"]["source_lock_file_sha256"], "bound source-lock hash")
    plan = json.loads(source_lock.read_text(encoding="utf-8"))
    require(plan["lock_sha256"] == lock["panel"]["source_lock_internal_sha256"], "bound internal lock")
    require(plan["matrix_count"] == 18 and plan["source_values"] == values, "bound plan dimensions")

    print("PASS: source-only closure, split, rate/oracle arithmetic, execution interlock, and authenticated plan binding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
