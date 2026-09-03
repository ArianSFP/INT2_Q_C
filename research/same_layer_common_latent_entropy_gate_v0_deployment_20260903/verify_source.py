"""Stdlib-only verifier for the frozen source package."""

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
    _require(observed_manifest == expected_manifest_sha256, "external manifest digest mismatch")
    raw = manifest_path.read_bytes()
    _require(b"\r" not in raw and raw.endswith(b"\n"), "manifest must use canonical LF")
    manifest = json.loads(raw.decode("utf-8"))
    canonical = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    _require(raw == canonical, "manifest JSON is not canonical")
    _require(manifest.get("schema") == "same_layer_common_latent_source_manifest_v0",
             "manifest schema")
    rows = manifest.get("files")
    _require(isinstance(rows, list) and rows, "manifest rows")
    names = [row.get("name") for row in rows]
    _require(names == sorted(names) and len(names) == len(set(names)), "manifest row order/uniqueness")
    _require("SOURCE_MANIFEST.json" not in names, "manifest must exclude itself")
    actual_entries = list(root.iterdir())
    actual_names = sorted(path.name for path in actual_entries)
    _require(actual_names == sorted(names + ["SOURCE_MANIFEST.json"]),
             "package contains missing, extra, or directory entries")
    canonical_rows = []
    for row in rows:
        _require(set(row) == {"bytes", "name", "sha256"}, "manifest row fields")
        name = row["name"]
        _require(isinstance(name, str) and name and Path(name).name == name, "flat member name")
        path = root / name
        mode = path.lstat().st_mode
        _require(stat.S_ISREG(mode) and not path.is_symlink(), f"non-regular member: {name}")
        _require(path.stat().st_size == int(row["bytes"]), f"byte mismatch: {name}")
        _require(_sha(path) == row["sha256"], f"hash mismatch: {name}")
        canonical_rows.append({"bytes": int(row["bytes"]), "name": name, "sha256": row["sha256"]})
    source_root = hashlib.sha256(
        json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _require(source_root == manifest.get("source_root_sha256"), "source-root mismatch")

    assignments = _literal_assignments(root / "run_gate.py")
    _require(assignments.get("PAYLOAD_EXECUTION_ENABLED") is False,
             "source release payload switch is not false")
    _require(assignments.get("PANEL_LOCK_SHA256") == _sha(root / "panel_lock.json"),
             "run_gate panel pin mismatch")
    _require(assignments.get("CORE_SHA256") == _sha(root / "common_latent_core.py"),
             "run_gate core pin mismatch")
    _require(assignments.get("WORKER_SHA256") == _sha(root / "cupy_worker.py"),
             "run_gate worker pin mismatch")
    gate_text = (root / "run_gate.py").read_text(encoding="utf-8")
    _require("if not PAYLOAD_EXECUTION_ENABLED:" in gate_text and
             gate_text.index("if not PAYLOAD_EXECUTION_ENABLED:") < gate_text.index("Path(__file__)"),
             "HOLD must precede first runtime Path access")
    fixture_text = (root / "run_source_free_cupy.py").read_text(encoding="utf-8")
    _require("payload-root" not in fixture_text and "payload_root" not in fixture_text,
             "source-free runner contains payload locator")
    _require("__TO_FREEZE__" not in gate_text, "unfrozen source pin")

    panel = json.loads((root / "panel_lock.json").read_text(encoding="utf-8"))
    _require(panel.get("layer") == 15 and len(panel.get("experts", [])) == 16,
             "panel identity")
    _require(len(panel.get("files", [])) == 32, "panel file count")
    design = json.loads((root / "design_lock.json").read_text(encoding="utf-8"))
    _require(design["decision"]["promotion_target_gain_bpw_on_up_down"] ==
             0.22933495044437175, "target threshold")
    return {
        "schema": "same_layer_common_latent_source_verification_v0",
        "status": "PASS_SOURCE_CLOSED_HOLD",
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
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
