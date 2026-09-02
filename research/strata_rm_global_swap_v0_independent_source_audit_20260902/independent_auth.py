#!/usr/bin/env python3
"""Independent, strict authentication for the frozen global-RM source gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_ROOT = "4f856e268d37ee1d6f32b4a2d1b8cd6879c235639ad75809ffd75fc7c4372d6c"
EXPECTED_SOURCE_MANIFEST_SHA256 = "939b57518a4afe11c56c59a20f109e423c8eab5815c947cb3e5a91d559704b3c"
EXPECTED_SOURCE_MEMBERS = {
    "coset_contract.py": (1269, "3a6e9df5b604bedd5a656cc3f312cbf0cf9620794ce7951fd146aef8665696d8"),
    "cupy_order_smoke.py": (2980, "3d48c7c9ee64a7ab64b5cf1cbcfd49af293d9fa995d7742eff9b93c22e262a06"),
    "DESIGN_LOCK.json": (2531, "c08b4cec442d7dbfe5ce17f535e2bce0fbb4d455fbe37503d0111404c5ccbeb9"),
    "EXECUTION_STATUS.json": (897, "5a90d0f9247a7dd5b54f93f36f26d106687cbc0c1084b2b21f2266e857ab2aa2"),
    "hostile_tests.py": (2519, "c2485273f54fbe01131fedc6f6fc45d69e2a1b723beca24b659c98eccd6932f3"),
    "LAUNCH_CONTRACT.md": (1666, "86484293ea134a36f692f6bff6bc9f4fd735503b7e36f21bb1fdcf035fe9e874"),
    "pin_semantics.py": (5177, "9067b0a25d066a007be35def6a863cfa4b15b6b8cdd891b2e64fa7b38474d9b2"),
    "README.md": (4305, "a24a34a276173a0913a17f59041e0bdcf69c4730c8751c070695e078d703a986"),
    "RED_TEAM.md": (2948, "416b7117a9e11b6b02ceb5df5f58a9da3b2a98db1f8fc76dec7f4ca8abf9b0f6"),
    "result_contract.py": (4663, "24a23cb70604ffe995c84a01bcec6468c3773dc3c0da665635f853ff3e284c96"),
    "rm_order.py": (5947, "00b74403d877d7fa1e4d4facff71b59a35cc734ef4a52e092dc095c5abd6ff97"),
    "run_source_gate.py": (3119, "e0bf2db4c23f73ca38a2ae5f9a7a8f887d703783102bc6d7d96379e1ba62fd61"),
    "swap_adapter.py": (2555, "9155c2ff917a0f6d388218987dc670973df508da8c44b0cdfe5de0a841a2fc19"),
    "test_source_only.py": (7033, "115fd796f2bd040ab4e252fed1ebf6cfbc3e76c4e434641bc535b43274208df7"),
    "verify_source.py": (2441, "aac0a74376439649d522f4902bb009c61c0c8c0b5b4ba077f10eed5a5bcc785a"),
}
EXPECTED_EXTERNAL_PINS = {
    "agent_polaris_qwen_rht_encoder.py":
        "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0",
    "bg_codec_bec_encoder.py":
        "456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267",
    "strata_v2_klt_mixed_independent_auditor_v1.py":
        "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e",
}
OVERLAP_RELATIVE = Path("INT2_Q_C/research/rm_bec_overlap_probe_v0_all14_aggregate.json")
OVERLAP_SHA256 = "dc4eb2f4896a466226974ac98f0cab2d4f5e9640b49d7c1d7c63ae957f6b7db2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: str(value["name"])):
        digest.update(str(row["name"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def authenticate_source(source: Path) -> dict[str, Any]:
    source = source.resolve()
    require(source.is_dir() and not source.is_symlink(), "source package must be a real directory")
    manifest_path = source / "source_manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "source manifest file")
    require(sha256_file(manifest_path) == EXPECTED_SOURCE_MANIFEST_SHA256,
            "source manifest byte hash")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "strata-rm-global-swap-v0-source-manifest",
            "source manifest schema")
    require(manifest.get("source_root_sha256") == EXPECTED_SOURCE_ROOT,
            "declared source root")
    rows = manifest.get("members")
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_SOURCE_MEMBERS),
            "manifest member count")
    declared: dict[str, tuple[int, str]] = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "manifest member fields")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name not in declared,
                "flat unique manifest member")
        declared[name] = (row["bytes"], row["sha256"])
    require(declared == EXPECTED_SOURCE_MEMBERS, "manifest member authority")
    require(root_hash(rows) == EXPECTED_SOURCE_ROOT, "independent source root")

    expected_entries = set(EXPECTED_SOURCE_MEMBERS) | {"source_manifest.json"}
    actual_entries = {entry.name for entry in source.iterdir()}
    require(actual_entries == expected_entries, "unexpected file or directory in source package")
    for name, (size, expected_hash) in EXPECTED_SOURCE_MEMBERS.items():
        path = source / name
        require(path.is_file() and not path.is_symlink(), f"regular source member: {name}")
        require(path.stat().st_size == size, f"source member size: {name}")
        require(sha256_file(path) == expected_hash, f"source member hash: {name}")
    return {
        "source": str(source),
        "source_root_sha256": EXPECTED_SOURCE_ROOT,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "members": len(EXPECTED_SOURCE_MEMBERS),
        "strict_top_level_entries": True,
        "symlinks_rejected": True,
        "status": "PASS_PINNED_SOURCE_AUTH",
    }


def authenticate_external(external_root: Path) -> dict[str, Any]:
    root = external_root.resolve()
    require(root.is_dir(), "external root")
    for name, expected_hash in EXPECTED_EXTERNAL_PINS.items():
        path = root / name
        require(path.is_file() and not path.is_symlink(), f"regular external pin: {name}")
        require(sha256_file(path) == expected_hash, f"external pin mismatch: {name}")
    overlap = root / OVERLAP_RELATIVE
    require(overlap.is_file() and not overlap.is_symlink(), "overlap receipt file")
    require(sha256_file(overlap) == OVERLAP_SHA256, "overlap receipt hash")
    row = json.loads(overlap.read_text(encoding="utf-8"))
    require(row.get("claim_boundary") == "row-set overlap only; no Qwen payload or RD claim",
            "overlap receipt claim boundary")
    return {
        "external_root": str(root),
        "external_pins": EXPECTED_EXTERNAL_PINS,
        "overlap_sha256": OVERLAP_SHA256,
        "status": "PASS_EXTERNAL_SOURCE_PINS__NON_RD_OVERLAP_ONLY",
    }
