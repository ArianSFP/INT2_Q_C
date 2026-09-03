"""Stdlib-only verifier for the sealed CBIB-1 source package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import stat


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal_assignments(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def verify(package: Path, expected_manifest_sha256: str) -> dict:
    root = Path(package)
    _require(root.is_absolute(), "package path must be absolute")
    _require(root.is_dir() and not root.is_symlink(), "package must be a real directory")
    _require(len(expected_manifest_sha256) == 64 and
             all(c in "0123456789abcdef" for c in expected_manifest_sha256),
             "external manifest digest must be lowercase SHA-256")
    manifest_path = root / "SOURCE_MANIFEST.json"
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "manifest file")
    observed_manifest = _sha(manifest_path)
    _require(observed_manifest == expected_manifest_sha256,
             "external manifest digest mismatch")
    raw = manifest_path.read_bytes()
    _require(b"\r" not in raw and raw.endswith(b"\n"), "canonical LF manifest")
    manifest = json.loads(raw.decode("utf-8"))
    canonical = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _require(raw == canonical, "manifest JSON is not canonical")
    _require(manifest.get("schema") == "same_layer_clustered_ib_source_manifest_v0",
             "manifest schema")
    rows = manifest.get("files")
    _require(isinstance(rows, list) and rows, "manifest rows")
    names = [row.get("name") for row in rows]
    _require(names == sorted(names) and len(names) == len(set(names)),
             "manifest row order/uniqueness")
    _require("SOURCE_MANIFEST.json" not in names, "manifest excludes itself")
    actual = sorted(path.name for path in root.iterdir())
    _require(actual == sorted(names + ["SOURCE_MANIFEST.json"]),
             "package contains missing, extra, or directory entries")
    canonical_rows = []
    for row in rows:
        _require(set(row) == {"bytes", "name", "sha256"}, "manifest row fields")
        name = row["name"]
        _require(isinstance(name, str) and name and Path(name).name == name, "flat member")
        path = root / name
        mode = path.lstat().st_mode
        _require(stat.S_ISREG(mode) and not path.is_symlink(), f"non-regular member: {name}")
        _require(path.stat().st_size == int(row["bytes"]), f"byte mismatch: {name}")
        _require(_sha(path) == row["sha256"], f"hash mismatch: {name}")
        canonical_rows.append({"bytes": int(row["bytes"]), "name": name,
                               "sha256": row["sha256"]})
    source_root = hashlib.sha256(
        json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _require(source_root == manifest.get("source_root_sha256"), "source-root mismatch")

    run_assignments = _literal_assignments(root / "run_gate.py")
    _require(run_assignments.get("PAYLOAD_EXECUTION_ENABLED") is False,
             "payload switch must be literal false")
    run_text = (root / "run_gate.py").read_text(encoding="utf-8").lower()
    _require("payload-root" not in run_text and "payload_root" not in run_text,
             "source gate contains payload locator")
    _require("authorization" not in run_text and "deployment implementation is absent" in run_text,
             "source gate must not create deployment authority")
    fixture_text = (root / "run_source_free_cupy.py").read_text(encoding="utf-8").lower()
    _require("payload-root" not in fixture_text and "payload_root" not in fixture_text,
             "source-free fixture contains payload locator")
    _require(fixture_text.index("verify_source.verify") < fixture_text.index("import cupy"),
             "closure must precede CuPy import")

    design = json.loads((root / "design_lock.json").read_text(encoding="utf-8"))
    _require(design.get("schema") == "same_layer_clustered_ib_design_lock_v0",
             "design schema")
    _require(design["payload_execution_enabled"] is False, "design HOLD")
    _require(design["grouping"]["candidate_group_sizes"] == [2, 4, 8, 16],
             "group bank")
    _require(design["cross_fit"]["fold_count"] == 8 and
             design["cross_fit"]["superblock_values"] == 2048,
             "cross-fit lock")
    _require(design["decision"]["promotion_target_gain_bpw_on_up_down"] ==
             0.22933495044437175, "exact threshold")
    _require(design["source_pins"]["clustered_ib_core_sha256"] ==
             _sha(root / "clustered_ib_core.py"), "core pin")
    _require(design["source_pins"]["cupy_backend_sha256"] ==
             _sha(root / "cupy_backend.py"), "CuPy pin")

    core_assignments = _literal_assignments(root / "clustered_ib_core.py")
    _require(core_assignments.get("GROUP_SIZES") == (2, 4, 8, 16), "core group bank")
    _require(core_assignments.get("FOLD_COUNT") == 8 and
             core_assignments.get("SUPERBLOCK_VALUES") == 2048,
             "core cross-fit constants")
    _require(core_assignments.get("TARGET_GAIN_BPW") == 0.22933495044437175,
             "core threshold")
    core_text = (root / "clustered_ib_core.py").read_text(encoding="utf-8").lower()
    forbidden = ("safetensors", "huggingface", "qwen/", "payload_root", "ssh ")
    _require(not any(token in core_text for token in forbidden),
             "mathematical core contains a source locator")
    return {
        "schema": "same_layer_clustered_ib_source_verification_v0",
        "status": "PASS_SOURCE_CLOSED_HOLD_NO_DEPLOYMENT",
        "manifest_sha256": observed_manifest,
        "source_root_sha256": source_root,
        "member_count_excluding_manifest": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    receipt = verify(Path(args.package).resolve(), args.manifest_sha256)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
