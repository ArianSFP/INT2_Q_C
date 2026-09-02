#!/usr/bin/env python3
"""Independent exact-closure authentication for the frozen v1 producer.

This review is source-only.  It authenticates code and public repository
dependencies and deliberately has no model, packet, checkpoint, or payload
discovery logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


PRODUCER_SOURCE_ROOT_SHA256 = (
    "980a5f1d272ca5ffc7b4d35e7c234a86994d135fcacaf0d47a8b3e00fc3d4f14")
PRODUCER_MANIFEST_SHA256 = (
    "4c2c5b371b1b9661d371de607e6a650f8c43fe0128726476854c2eb2ca560c85")
PRODUCER_SCHEMA = "strata-rm-global-swap-v1-hardened-source-manifest"
PRODUCER_MEMBERS = {
    "AUDIT_REPAIR_MAP.md": (3920, "74d2be202624b3ce22d8e7dd76af1392da4bf036c3f3ae5f4a795e87739ad400"),
    "authority.py": (17472, "31fba902286735b26ee4b1428640525d8c0811bb2322ffb0c16008ab3b5de7d0"),
    "current_integration_worker.py": (4030, "ee4c2bfe0ab4d2192db1e604701052f646debb09abe76d8ff61d54b89774b9c9"),
    "DESIGN_LOCK.json": (2086, "97d88affb3d5a582f2fcc99ba66b34cce84b2d940555f939c605b2349813c8b7"),
    "EXECUTION_STATUS.json": (1005, "d8741105e3e9b34a965a6fe9687d79a80132c4bd628909c5ac237599264cacb0"),
    "fixture_decoder_worker.py": (4932, "6dc3c91773a23ff4b58aaa97bfea8f80c238f9c935392a5d4c0b3bfc77da04ac"),
    "hostile_tests.py": (2859, "554f51d2a0116d495fb11412438ae3fe8915b8f71bfbc2833e2c693b57c23479"),
    "physical_authority.py": (27137, "89343cfa417fc5f4ab39fc1aef75aa85f9a1a6f66c041790f7426e56965d5678"),
    "README.md": (5573, "b11e98bd19e45d7be9035748d846a5a2062b221ced124a9b87cf36cbd5764a58"),
    "real_cupy_worker.py": (4003, "70c6e5c10cfafde8945ec30bd68663b9eff834ff0d66d9b9fcfdc1b86d92e379"),
    "rm_order.py": (3116, "e5d85d844633d206125a775efcd35711d02bf9eec5060715c17e8e7d50df0f92"),
    "run_source_gate.py": (2916, "5c6cdad5d512aa55fcd430bca9ed7587e40c52be77cebbfd28220ce13f30a450"),
    "test_source_only.py": (11982, "634067ea42b1ac81e844fb1b7befd8a08d939117cd5394895fd340b1ae689920"),
    "verify_source.py": (3248, "5e46ebce8a55fb59ee346dc09a3b12be684efe8633bd77b697e6df6bad36f666"),
}
EXTERNAL_PINS = {
    "agent_polaris_qwen_rht_encoder.py":
        "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0",
    "bg_codec_bec_encoder.py":
        "456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267",
    "strata_v2_klt_mixed_independent_auditor_v1.py":
        "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e",
}


class ReviewError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result

    try:
        row = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReviewError(f"{label}: nonfinite {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"{label}: strict JSON") from exc
    require(isinstance(row, dict), f"{label}: JSON object")
    return row


def regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                f"{label}: regular non-link file")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ReviewError(f"{label}: read") from exc
    require((before.st_dev, before.st_ino, before.st_mode, before.st_size,
             before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_mode, after.st_size,
             after.st_mtime_ns), f"{label}: changed while read")
    return payload


def authenticate_producer(package: Path) -> dict[str, Any]:
    original = Path(package)
    try:
        root_stat = original.lstat()
        require(stat.S_ISDIR(root_stat.st_mode) and not original.is_symlink(),
                "producer root must be a real directory, not a link")
        root = original.resolve(strict=True)
    except OSError as exc:
        raise ReviewError("producer root resolution") from exc
    manifest_payload = regular_bytes(root / "source_manifest.json",
                                     "producer manifest")
    require(sha256(manifest_payload) == PRODUCER_MANIFEST_SHA256,
            "producer manifest external pin")
    manifest = strict_json(manifest_payload, "producer manifest")
    require(manifest.get("schema") == PRODUCER_SCHEMA, "producer schema")
    require(manifest.get("source_root_sha256") == PRODUCER_SOURCE_ROOT_SHA256,
            "producer declared root")
    rows = manifest.get("members")
    require(isinstance(rows, list) and len(rows) == len(PRODUCER_MEMBERS),
            "producer member list")
    observed = []
    declared = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "producer member schema")
        name = row["name"]
        require(isinstance(name, str) and name and Path(name).name == name and
                name not in declared and name != "source_manifest.json",
                "producer flat unique member")
        declared[name] = (row["bytes"], row["sha256"])
        payload = regular_bytes(root / name, f"producer member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"producer member pin {name}")
        observed.append(item)
    require(declared == PRODUCER_MEMBERS, "producer independent member map")
    require(sha256(canonical_json(observed)) == PRODUCER_SOURCE_ROOT_SHA256,
            "producer independently recomputed root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} ==
            set(PRODUCER_MEMBERS) | {"source_manifest.json"},
            "producer exact top-level closure")
    require(all(entry.is_file(follow_symlinks=False) and
                not entry.is_dir(follow_symlinks=False) for entry in entries),
            "producer closure has regular files only")
    return {
        "path": str(root), "source_root_sha256": PRODUCER_SOURCE_ROOT_SHA256,
        "manifest_sha256": PRODUCER_MANIFEST_SHA256,
        "members": len(observed), "exact_regular_closure": True,
        "payloads_opened": 0, "status": "PASS_INDEPENDENT_PRODUCER_AUTH",
    }


def authenticate_external_sources(external_root: Path) -> dict[str, Any]:
    original = Path(external_root)
    try:
        root_stat = original.lstat()
        require(stat.S_ISDIR(root_stat.st_mode) and not original.is_symlink(),
                "external root real directory")
        root = original.resolve(strict=True)
    except OSError as exc:
        raise ReviewError("external root resolution") from exc
    for name, expected in EXTERNAL_PINS.items():
        payload = regular_bytes(root / name, f"external source {name}")
        require(sha256(payload) == expected, f"external source pin {name}")
    return {"external_root": str(root), "external_pins": dict(EXTERNAL_PINS),
            "payloads_opened": 0, "status": "PASS_EXTERNAL_SOURCE_PINS"}
