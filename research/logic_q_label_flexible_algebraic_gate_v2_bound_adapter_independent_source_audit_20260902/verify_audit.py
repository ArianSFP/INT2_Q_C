#!/usr/bin/env python3
"""Standard-library closure verifier for the independent LOGIC-Q v2 audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any


SCHEMA = "logic-q-v2-independent-source-audit-manifest-v1"
STATUS = "MECHANISM_VALID__HOLD_PRODUCTION_PROVENANCE_BACKEND_AND_STRATA"
FROZEN_MANIFEST_SHA256 = (
    "e97041b2debdd1a85ce32305f43aae1f76cf4ca937b52e275bdd246ae1b1b980")
FROZEN_ROOT_SHA256 = (
    "080de7a63e596ae34f9da90941d7fd9d07b70dfb2afad97103aa5ab5943d3776")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MEMBERS = {
    "AUDIT_DISPOSITION.json",
    "README.md",
    "RUNPOD_CUPY_PROBE.json",
    "hostile_audit.py",
    "verify_audit.py",
}


class AuditVerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditVerifyError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AuditVerifyError(f"{label} nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditVerifyError(f"{label} strict JSON") from exc
    require(isinstance(value, dict), f"{label} object")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
            f"{label} regular non-link")
    payload = path.read_bytes()
    after = path.lstat()
    require((before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino),
            f"{label} changed during read")
    return payload


def verify_frozen(package: Path) -> dict[str, Any]:
    root = package.resolve(strict=True)
    manifest_payload = regular_bytes(root / "SOURCE_MANIFEST.json",
                                     "frozen manifest")
    require(sha256(manifest_payload) == FROZEN_MANIFEST_SHA256,
            "frozen external manifest pin")
    manifest = strict_json(manifest_payload, "frozen manifest")
    require(manifest.get("source_root_sha256") == FROZEN_ROOT_SHA256,
            "frozen external source-root pin")
    observed = []
    names = []
    for row in manifest.get("members", []):
        require(isinstance(row, dict) and
                set(row) == {"name", "bytes", "sha256"},
                "frozen member row")
        name = row["name"]
        require(isinstance(name, str) and name not in names and "/" not in name
                and "\\" not in name, "frozen member name")
        payload = regular_bytes(root / name, f"frozen member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"frozen member pin {name}")
        observed.append(item)
        names.append(name)
    require(sha256(canonical_json(observed)) == FROZEN_ROOT_SHA256,
            "frozen observed source root")
    require({entry.name for entry in os.scandir(root)} ==
            set(names) | {"SOURCE_MANIFEST.json"}, "frozen exact closure")
    return {"manifest_sha256": FROZEN_MANIFEST_SHA256,
            "source_root_sha256": FROZEN_ROOT_SHA256}


def verify(package: Path, expected_manifest_sha256: str | None) -> dict[str, Any]:
    root = package.resolve(strict=True)
    manifest_payload = regular_bytes(root / "SOURCE_MANIFEST.json", "manifest")
    manifest_sha = sha256(manifest_payload)
    if expected_manifest_sha256 is not None:
        require(HEX64.fullmatch(expected_manifest_sha256) is not None and
                manifest_sha == expected_manifest_sha256,
                "audit external manifest pin")
    manifest = strict_json(manifest_payload, "manifest")
    require(set(manifest) == {"schema", "status", "source_root_sha256",
                              "frozen_source", "members", "execution",
                              "claim_boundary"}, "manifest schema")
    require(manifest["schema"] == SCHEMA and manifest["status"] == STATUS,
            "manifest schema/status")
    require(manifest["frozen_source"] == {
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "source_root_sha256": FROZEN_ROOT_SHA256,
        "modified": False,
    }, "frozen source pins")
    observed = []
    names = []
    for row in manifest["members"]:
        require(isinstance(row, dict) and
                set(row) == {"name", "bytes", "sha256"}, "member row")
        name = row["name"]
        require(name in MEMBERS and name not in names, "member name")
        payload = regular_bytes(root / name, f"member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"member pin {name}")
        observed.append(item)
        names.append(name)
    require(set(names) == MEMBERS and
            names == sorted(names, key=lambda value: value.encode("utf-8")),
            "complete canonical members")
    require(manifest["source_root_sha256"] == sha256(canonical_json(observed)),
            "audit source root")
    require({entry.name for entry in os.scandir(root)} ==
            MEMBERS | {"SOURCE_MANIFEST.json"} and
            all(entry.is_file(follow_symlinks=False) for entry in os.scandir(root)),
            "audit exact regular closure")

    disposition = strict_json(
        regular_bytes(root / "AUDIT_DISPOSITION.json", "disposition"),
        "disposition")
    require(disposition["status"] == STATUS and
            disposition["audited_source"]["manifest_sha256"] ==
            FROZEN_MANIFEST_SHA256 and
            disposition["hostile_test_program"]["runpod_executed"] is False,
            "honest audit disposition")
    probe = strict_json(regular_bytes(root / "RUNPOD_CUPY_PROBE.json", "probe"),
                        "probe")
    require(probe["status"] ==
            "PASS_GENERIC_REAL_DEVICE_PROBE__NOT_A_V2_LAUNCH_RECEIPT" and
            probe["runpod"]["probe_observed"] ==
            probe["runpod"]["probe_expected"] and
            probe["scope"]["frozen_v2_collect_receipt_called"] is False,
            "generic RunPod scope")
    hostile_source = regular_bytes(root / "hostile_audit.py", "hostile audit")
    ast.parse(hostile_source.decode("utf-8"), filename="hostile_audit.py")
    required_attacks = (
        "fully_forged_rows_packet_and_source_hashes_authorize_target_config",
        "cpu_facade_spoofs_canonical_cupy_device_and_probe",
        "same_packet_scores_under_distinct_config_id",
        "launch_context_accepts_nonselected_config_for_selection_hash",
        "nan_scale_rejected_from_actual_packet",
        "nonzero_page_padding_rejected",
    )
    text = hostile_source.decode("utf-8")
    require(all(token in text for token in required_attacks),
            "hostile attack coverage")
    frozen = verify_frozen(root.parent /
                           "logic_q_label_flexible_algebraic_gate_v2_bound_adapter")
    return {
        "schema": "logic-q-v2-independent-source-audit-verification-v1",
        "status": "PASS_EXACT_AUDIT_SOURCE_CLOSURE",
        "audit_manifest_sha256": manifest_sha,
        "audit_source_root_sha256": manifest["source_root_sha256"],
        "frozen_source": frozen,
        "execution": manifest["execution"],
        "payload_accessed": False,
        "cupy_imported": False,
        "network_accessed": False,
        "disposition": STATUS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.expected_manifest_sha256),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
