#!/usr/bin/env python3
"""Standard-library source-only verifier for the direct-output gate."""

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
    "direct_output_codebook_stage0.py",
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


def ledger(experts: int) -> tuple[float, float, list[tuple[int, int, int, int]]]:
    values = experts * 3 * 768 * 2048
    global_side = 528384
    local_fixed = 1105920 + 9216 + 64
    prefix = (global_side * 8 + experts * local_fixed * 8) / values
    threshold = 0.8 / (2.0 ** (2.0 * prefix))
    rows = []
    for rate in (2.15, 2.30, 2.50):
        physical = math.ceil(rate * values / 8.0)
        local_max = math.ceil((physical - global_side) / experts)
        residual = physical - global_side - experts * local_fixed
        cold = global_side + 4096 * math.ceil(local_max / 4096.0)
        rows.append((physical, local_max, residual, cold))
    return prefix, threshold, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = Path(__file__).resolve().parent
    observed = {entry.name for entry in package.iterdir()}
    require(observed == PACKAGE_FILES, f"package closure mismatch: {sorted(observed)}")
    for name in PACKAGE_FILES:
        require((package / name).is_file() and not (package / name).is_symlink(), f"invalid member: {name}")

    manifest = json.loads((package / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["schema"] == "direct-output-codebook-stage0-source-manifest-v0", "manifest schema")
    require(set(manifest["closure"]) == PACKAGE_FILES, "manifest closure")
    for row in manifest["files"]:
        path = package / row["path"]
        require(path.stat().st_size == row["bytes"], f"byte mismatch: {row['path']}")
        require(sha256_file(path) == row["sha256"], f"hash mismatch: {row['path']}")

    design = json.loads((package / "design_lock.json").read_text(encoding="utf-8"))
    require(design["schema"] == "direct-output-codebook-whole-expert-stage0-design-lock-v0", "design schema")
    require(design["panel"]["fit_slots"] == [0, 2, 3, 5], "fit split")
    require(design["panel"]["holdout_slots"] == [1, 4], "holdout split")
    require(set(design["panel"]["fit_slots"]).isdisjoint(design["panel"]["holdout_slots"]), "split overlap")
    require(design["panel"]["source_lock_relpath"] == str(SOURCE_LOCK_RELPATH).replace("\\", "/"), "source-lock path")
    require(design["architecture"]["code_count"] == 32768, "code count")
    require(design["architecture"]["vector_dimension"] == 8, "vector dimension")
    require(design["architecture"]["index_bits_per_vector"] == 15, "index bits")
    require(design["rate"]["codebook_fp16_bytes"] == 524288, "table bytes")
    require(design["rate"]["global_side_bytes"] == 528384, "global bytes")
    require(design["rate"]["row_moments_fp16_bytes_per_expert"] == 9216, "row moments")
    require(design["execution"]["steps"] == 1024, "update count")
    require(design["execution"]["full_lloyd_passes"] == 1, "Lloyd pass count")
    require([row["step"] for row in design["fixed_collapse_checkpoints"]] == [128, 256, 512, 1024, "full_lloyd_1"], "checkpoint schedule")

    panel_prefix, panel_q, panel_rows = ledger(6)
    require(math.isclose(panel_prefix, 2.0400390625, rel_tol=0.0, abs_tol=1e-15), "panel prefix")
    require(math.isclose(panel_q, 0.047300320854109984, rel_tol=0.0, abs_tol=1e-15), "panel q")
    require(panel_rows == [
        (7608730, 1180058, 389146, 1712128),
        (8139572, 1268532, 919988, 1798144),
        (8847360, 1386496, 1627776, 1916928),
    ], "panel rate ledger")
    layer_prefix, layer_q, layer_rows = ledger(128)
    require(math.isclose(layer_prefix, 1.8977322048611112, rel_tol=0.0, abs_tol=1e-15), "layer prefix")
    require(math.isclose(layer_q, 0.05761576759174624, rel_tol=0.0, abs_tol=1e-15), "layer q")
    require(layer_rows == [
        (162319565, 1263994, 19045581, 1794048),
        (173644186, 1352468, 30370202, 1884160),
        (188743680, 1470432, 45469696, 1998848),
    ], "layer rate ledger")
    for experts, rows in ((6, panel_rows), (128, layer_rows)):
        for physical, _local_max, _residual, cold in rows:
            require(cold / (physical / experts) < 2.0, "read amplification")

    runner = (package / "direct_output_codebook_stage0.py").read_text(encoding="utf-8")
    tree = ast.parse(runner)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    require(not imported.intersection({"requests", "urllib", "socket", "subprocess", "torch"}), "forbidden import")
    require("import cupy as cp" in runner, "CuPy import missing")
    require("AUTHORIZATION" in runner and "--authorization" in runner, "execution interlock")
    require('SOURCE_LOCK_RELPATH = Path("blind_protocol_v2/unblinded/source_hashes.lock.json")' in runner, "runner source-lock path")
    require("source_root = lock_path.parent.resolve()" in runner, "payload root")
    require('path = (source_root / str(row["output_relpath"])).resolve()' in runner, "payload resolution")
    require("CODE_COUNT = 32768" in runner and "VECTOR_DIM = 8" in runner, "fixed table shape")
    require("full_lloyd_pass" in runner and "repair_clusters" in runner, "training safeguards")
    require("parse_global" in runner and "exact_evaluation" in runner, "finite exact evaluation")

    source_lock = (args.root.resolve() / SOURCE_LOCK_RELPATH).resolve()
    require(source_lock.stat().st_size == design["panel"]["source_lock_bytes"], "bound source-lock bytes")
    require(sha256_file(source_lock) == design["panel"]["source_lock_file_sha256"], "bound source-lock hash")
    plan = json.loads(source_lock.read_text(encoding="utf-8"))
    require(plan["lock_sha256"] == design["panel"]["source_lock_internal_sha256"], "bound internal lock")
    require(plan["matrix_count"] == 18 and plan["source_values"] == 28311552, "bound plan dimensions")

    print("PASS: direct-output source closure, split, training lock, exact table, rate/read arithmetic, and authenticated plan binding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
