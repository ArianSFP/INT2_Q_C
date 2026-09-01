#!/usr/bin/env python3
"""Standard-library verifier for the inert same-layer alignment package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


EXPECTED = {"README.md", "design_lock.json", "same_layer_alignment_oracle.py",
            "verify_source.py", "PACKAGE_MANIFEST.json"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    files = {path.name for path in package.iterdir() if path.is_file()}
    require(files == EXPECTED, "exact five-file closure")
    manifest = json.loads((package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    require(manifest["schema"] == "same-layer-expert-alignment-source-manifest-v0", "manifest schema")
    for name, expected in manifest["files"].items():
        require(name != "PACKAGE_MANIFEST.json" and sha(package / name) == expected, f"hash {name}")
    design = json.loads((package / "design_lock.json").read_text(encoding="utf-8"))
    require(design["schema"] == "same-layer-expert-alignment-superoracle-design-v0", "design schema")
    require(design["oracle"]["reference_rows_per_target_role"] == 11520, "reference rows")
    require(abs(design["decision"]["required_up_down_capture_if_sole_missing_module"] -
                (1.0 - 2.0 ** (-2.0 * design["decision"]["existing_composite_missing_s_bpw"]))) < 2e-12,
            "capture identity")
    source_manifest = args.workspace.resolve(strict=True) / design["source"]["manifest_relpath"]
    require(sha(source_manifest) == design["source"]["manifest_sha256"], "source manifest pin")
    runner = (package / "same_layer_alignment_oracle.py").read_text(encoding="utf-8")
    ast.parse(runner)
    require("import cupy as cp" in runner, "CuPy import")
    require("references[selected]" in runner, "exact selected replay")
    require("target32 @ refs32.T" in runner, "batched CuPy search")
    print(json.dumps({"verdict": "PASS", "files": len(EXPECTED),
                      "manifest_sha256": sha(package / "PACKAGE_MANIFEST.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
