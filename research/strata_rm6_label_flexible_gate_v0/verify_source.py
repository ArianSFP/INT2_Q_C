#!/usr/bin/env python3
"""Verify the frozen no-payload STRATA-RM6 source package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "strata-rm6-label-flexible-gate-v0-source-manifest"
STATUS = "FROZEN_GO_LOCAL_MECHANISM_HOLD_PRODUCTION_AND_PAYLOAD"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def verify(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    manifest_raw = (root / "SOURCE_MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS:
        raise RuntimeError("manifest schema/status")
    observed = []
    for row in manifest["members"]:
        raw = (root / row["name"]).read_bytes()
        if len(raw) != row["bytes"] or sha(raw) != row["sha256"]:
            raise RuntimeError(f"source member mismatch: {row['name']}")
        observed.append({"name": row["name"], "bytes": len(raw), "sha256": sha(raw)})
    names = [row["name"] for row in observed]
    if names != sorted(names, key=lambda value: value.encode("utf-8")):
        raise RuntimeError("manifest order")
    if sha(canonical(observed)) != manifest["source_root_sha256"]:
        raise RuntimeError("source root")
    if {path.name for path in root.iterdir()} != set(names) | {"SOURCE_MANIFEST.json"}:
        raise RuntimeError("source closure")
    core = (root / "rm6_core.py").read_text(encoding="utf-8")
    sc = (root / "strata_rm_sc.py").read_text(encoding="utf-8")
    packet = (root / "packet_codec.py").read_text(encoding="utf-8")
    gate = (root / "run_gate.py").read_text(encoding="utf-8")
    red = (root / "RED_TEAM.md").read_text(encoding="utf-8")
    if "index.bit_count() >= threshold" not in core:
        raise RuntimeError("RM orientation lock")
    if "RM-ordered truncated polar set" not in sc:
        raise RuntimeError("global exact/truncated distinction")
    if "actual arithmetic packet exceeds 2.5 bpw" not in packet:
        raise RuntimeError("physical fail-closed cap")
    if "HOLD_NO_CURRENT_K_ARITHMETIC_PAYLOAD_MEASUREMENT" not in gate:
        raise RuntimeError("global hold")
    if "Random frozen values hide polynomial appearance" not in red:
        raise RuntimeError("coset red-team")
    if "--qwen" in gate or "--coarse" in gate or "--control" in gate:
        raise RuntimeError("payload CLI")
    return {"schema": "strata-rm6-source-verification-v0", "status": "PASS",
            "manifest_sha256": sha(manifest_raw),
            "source_root_sha256": manifest["source_root_sha256"],
            "members": len(observed), "qwen_payload_accessed": False,
            "coarse_payload_accessed": False, "control_payload_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    print(json.dumps(verify(Path(parser.parse_args().package)), sort_keys=True,
                     separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
