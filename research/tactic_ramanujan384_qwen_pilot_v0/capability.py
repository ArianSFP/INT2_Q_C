#!/usr/bin/env python3
"""Fail-closed external capability boundary for the Qwen pilot.

No payload path is resolved until the capability file matches the one digest
compiled into this module.  This source freeze deliberately compiles no digest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


TRUSTED_CAPABILITY_SHA256: str | None = None
PACKAGE_AUTHORITY_ID = "tactic-ramanujan384-qwen-pilot-v0-source-author"
CAPABILITY_SCHEMA = "tactic-ramanujan384-qwen-pilot-v0-external-capability-v1"
CAPABILITY_STATUS = "SEALED_BEFORE_ANY_MODEL_OR_COARSE_PAYLOAD_ACCESS"
ROLE_ORDER = ("gate", "up", "down_transposed")
GEOMETRY = {"intermediate": 768, "hidden": 2048, "blocks_per_role": 384}
SAMPLE_BLOCKS = {
    "gate": tuple((17 + 23 * index) % 384 for index in range(16)),
    "up": tuple((151 + 29 * index) % 384 for index in range(16)),
    "down_transposed": tuple((271 + 31 * index) % 384 for index in range(16)),
}
REQUIRED_CAPTURE = 0.32387022205373717
COARSE_RELATIVE_MSE = 0.036975150060595235
TARGET_D = 0.025
EXPECTED_PHYSICAL_BYTES = 1_470_464
EXPECTED_WEIGHTS = 4_718_592
EXPECTED_RATE_NUMERATOR = 359
EXPECTED_RATE_DENOMINATOR = 144
PAGE_BYTES = 4096
HEX64 = re.compile(r"[0-9a-f]{64}\Z")

KNOWN_CLOSURES = {
    "v3_atomic": {
        "manifest_name": "SOURCE_MANIFEST.json",
        "manifest_sha256": "97fb4cba64ff884615810fc8fc835c12ce98bf3e9db37b8a77be93d0d5372be1",
        "root_field": "source_root_sha256",
        "source_root_sha256": "5f86d9a1b48f7769867c828322132be303617d0444d50b5439f7b9d0074ab674",
        "schema": "tactic-ramanujan384-atomic-source-manifest-v3",
        "root_domain_hex": "",
        "root_row_order": "sorted_keys",
    },
    "v3_independent_review": {
        "manifest_name": "source_manifest.json",
        "manifest_sha256": "60feb6ae08b3d57df6056e0912759b1e4eb9eb7888c90467cbfd37e72ba97173",
        "root_field": "source_root_sha256",
        "source_root_sha256": "27f422950b7bdd686541677341665fb075295cdfbdd2e1acac3a5c42ce089cd2",
        "schema": "tactic-ramanujan384-atomic-v3-independent-source-review-manifest-v1",
        "root_domain_hex": "",
        "root_row_order": "name_bytes_sha256",
    },
    "v2_scalable": {
        "manifest_name": "SOURCE_MANIFEST.json",
        "manifest_sha256": "1f579f33216edeebbebb6c1714a4e56739da30ae0f12ae9bd44baf15a6163209",
        "root_field": "source_root_sha256",
        "source_root_sha256": "bff5a0c541cb2117a8cc1db3e539493bacc590b4e007ab7f193ca615e03a7495",
        "schema": "tactic-ramanujan384-scalable-source-manifest-v2",
        "root_domain_hex": "",
        "root_row_order": "sorted_keys",
    },
    "v2_independent_review": {
        "manifest_name": "SOURCE_MANIFEST.json",
        "manifest_sha256": "4ed8c0fe24db072e22aef84791a01ccf637cb337376a389d47119248fd257281",
        "root_field": "source_root_sha256",
        "source_root_sha256": "16ea8dfde5cf7a48552dc7b5a74b209488934b8764e890bf51bb5cd02985cd39",
        "schema": "tactic-ramanujan384-v2-scalable-independent-source-review-manifest",
        "root_domain_hex": "",
        "root_row_order": "sorted_keys",
    },
    "coarse_v6": {
        "manifest_name": "SOURCE_MANIFEST.json",
        "manifest_sha256": "31662539a4c55926f47b378d15a0d8e23c90aa0903328c44be2e237eca48b15d",
        "root_field": "source_root_sha256",
        "source_root_sha256": "161ab23169af3427648ec1bbcb9402568a0fb8aefc4a794daf3ebd1c56cc83f2",
        "schema": "tactic-actual-coarse-n18-v6-source-manifest-v1",
        "root_domain_hex": "",
        "root_row_order": "sorted_keys",
    },
    "coarse_result_auditor_v1": {
        "manifest_name": "SOURCE_MANIFEST.json",
        "manifest_sha256": "5386571db2a8e828c09368f603b3ccf0ccf3936204e7e06231d5c5798eb9f97f",
        "root_field": "source_snapshot_root_sha256",
        "source_root_sha256": "59387c67a18bb776cca820e658be998d75f9c3c1a9b7ef5c809e692f78a50742",
        "schema": "tactic-actual-coarse-n18-v6-result-auditor-source-manifest-v1",
        "root_domain_hex": (
            "5441435449432d41435455414c2d434f415253452d4e31382d56362d524553554c"
            "542d41554449544f522d534f555243452d524f4f542d563100"),
        "root_row_order": "sorted_keys",
    },
}
COARSE_RESULT_AUDIT_FILE_SHA256 = (
    "e03af88a5d33eaca30f935fffc8fcade477219c1be1afebb952428982e4d48e7")
COARSE_FRAME_SHA256 = (
    "6c13780bf1494567f91bc73bf6afd8846c6e3326cac329e4d8e3faf48a9051d7")
COARSE_FRAME_BYTES = 1_414_656
COARSE_AUDIT_STATUS = (
    "PASS_EXACT_NONPROMOTING_TACTIC_ACTUAL_COARSE_N18_V6_QWEN_RESULT_AUDIT")


class CapabilityError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapabilityError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def closure_root_rows(rows: list[dict[str, Any]], row_order: str) -> bytes:
    if row_order == "sorted_keys":
        return canonical_json(rows)
    require(row_order == "name_bytes_sha256", "closure root row order")
    ordered = [{"name": row["name"], "bytes": row["bytes"],
                "sha256": row["sha256"]} for row in rows]
    return json.dumps(ordered, sort_keys=False, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def strict_json(payload: bytes, label: str, *, canonical: bool = True) -> dict[str, Any]:
    def hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"{label}: duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CapabilityError(f"{label}: nonfinite JSON {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CapabilityError(f"{label}: strict JSON") from exc
    require(isinstance(value, dict), f"{label}: object")
    if canonical:
        require(payload == canonical_json(value) + b"\n", f"{label}: canonical JSON")
    return value


def regular_bytes(path: Path, label: str, maximum: int) -> tuple[bytes, tuple[int, int]]:
    try:
        absolute = path.resolve(strict=True)
        require(absolute == path.absolute() and not path.is_symlink(),
                f"{label}: canonical non-link path")
        descriptor = os.open(
            os.fspath(absolute), os.O_RDONLY | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise CapabilityError(f"{label}: open") from exc
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 < before.st_size <= maximum, f"{label}: bounded single-link file")
        chunks = bytearray()
        while len(chunks) < before.st_size:
            row = os.read(descriptor, min(8 << 20, before.st_size - len(chunks)))
            require(bool(row), f"{label}: short read")
            chunks.extend(row)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_nlink), f"{label}: identity drift")
        return bytes(chunks), (before.st_dev, before.st_ino)
    finally:
        os.close(descriptor)


def _flat_closure(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {"path", "manifest_name", "manifest_sha256", "source_root_sha256",
                "schema", "root_field", "root_domain_hex", "root_row_order"}
    require(isinstance(record, Mapping) and set(record) == required,
            "source closure record")
    root = Path(record["path"]).resolve(strict=True)
    require(root == Path(record["path"]).absolute() and root.is_dir() and
            not Path(record["path"]).is_symlink(), "source closure canonical directory")
    manifest_payload, _ = regular_bytes(root / record["manifest_name"],
                                        "source closure manifest", 32 << 20)
    require(sha256(manifest_payload) == record["manifest_sha256"],
            "source closure manifest pin")
    manifest = strict_json(manifest_payload, "source closure manifest", canonical=False)
    require(manifest.get("schema") == record["schema"] and
            manifest.get(record["root_field"]) == record["source_root_sha256"],
            "source closure schema/root")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "source closure members")
    observed = []
    names = []
    member_bytes = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "source member row")
        name = row["name"]
        require(isinstance(name, str) and name == Path(name).name and
                name not in names, "source member name")
        payload, _ = regular_bytes(root / name, f"source member {name}", 64 << 20)
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"source member pin {name}")
        names.append(name)
        observed.append(item)
        member_bytes[name] = payload
    domain = bytes.fromhex(record["root_domain_hex"])
    require(rows == sorted(rows, key=lambda row: row["name"].encode("utf-8")) and
            sha256(domain + closure_root_rows(
                observed, record["root_row_order"])) == record["source_root_sha256"],
            "source closure canonical root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {record["manifest_name"]} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "source package exact closure")
    return {"root": root, "manifest": manifest, "members": member_bytes}


def _validate_source_test_document(document: Mapping[str, Any]) -> None:
    """Schema-only validator used by tests; it never grants access."""
    required = {"schema", "status", "evidence_class", "issuer_authority_id",
                "issued_before_payload_access", "pilot_source_pins", "closures",
                "runtime_audits", "cupy_runtime", "coarse_result", "roles", "pilot",
                "output_parent"}
    require(isinstance(document, Mapping) and set(document) == required and
            document["schema"] == CAPABILITY_SCHEMA and
            document["status"] == CAPABILITY_STATUS and
            document["issued_before_payload_access"] is True,
            "capability exact schema/status")
    require(document["evidence_class"] in {"SOURCE_TEST_FIXTURE", "PRODUCTION"} and
            isinstance(document["issuer_authority_id"], str) and
            document["issuer_authority_id"] and
            document["issuer_authority_id"] != PACKAGE_AUTHORITY_ID,
            "independent capability issuer")
    pins = document["pilot_source_pins"]
    require(isinstance(pins, Mapping) and set(pins) == {
        "manifest_sha256", "source_root_sha256"
    } and all(is_sha256(value) for value in pins.values()), "pilot source pins")
    closures = document["closures"]
    require(isinstance(closures, Mapping) and set(closures) == set(KNOWN_CLOSURES),
            "exact dependency closure set")
    for key, expected in KNOWN_CLOSURES.items():
        row = closures[key]
        require(isinstance(row, Mapping) and set(row) == {
            "path", "manifest_name", "manifest_sha256", "source_root_sha256",
            "schema", "root_field", "root_domain_hex", "root_row_order"
        } and all(row[field] == expected[field] for field in expected),
                f"frozen dependency pins {key}")
    audits = document["runtime_audits"]
    require(isinstance(audits, list) and len(audits) == 2 and
            {row.get("kind") for row in audits} ==
            {"v3_atomic_cupy_runtime", "pilot_runner_independent_source"},
            "required independent runtime/source audits")
    source_fixture = document["evidence_class"] == "SOURCE_TEST_FIXTURE"
    for row in audits:
        require(isinstance(row, Mapping) and set(row) == {
            "kind", "package_path", "manifest_sha256", "source_root_sha256",
            "root_field", "root_domain_hex", "receipt_name", "receipt_sha256",
            "auditor_authority_id", "executed", "status", "dummy", "self_authored"
        } and all(is_sha256(row[field]) for field in
                  ("manifest_sha256", "source_root_sha256", "receipt_sha256")) and
                row["root_field"] in {"source_root_sha256", "audit_source_root_sha256",
                                      "source_snapshot_root_sha256"} and
                isinstance(row["root_domain_hex"], str) and
                len(row["root_domain_hex"]) % 2 == 0 and
                all(character in "0123456789abcdef"
                    for character in row["root_domain_hex"]) and
                row["executed"] is True and row["status"] == "PASS_INDEPENDENT_AUDIT" and
                row["dummy"] is source_fixture and row["self_authored"] is source_fixture and
                isinstance(row["auditor_authority_id"], str) and
                row["auditor_authority_id"] not in
                {document["issuer_authority_id"], PACKAGE_AUTHORITY_ID},
                "independent executed audit capability")
    require(len({row["auditor_authority_id"] for row in audits}) == 2 and
            len({row["package_path"] for row in audits}) == 2 and
            len({(row["manifest_sha256"], row["source_root_sha256"],
                  row["receipt_sha256"]) for row in audits}) == 2,
            "non-aliased independent audit declarations")
    runtime = document["cupy_runtime"]
    require(isinstance(runtime, Mapping) and set(runtime) == {
        "version", "module_file_sha256", "device_ordinal", "device_name",
        "compute_capability", "runtime_version", "driver_version"
    } and isinstance(runtime["version"], str) and runtime["version"] and
            is_sha256(runtime["module_file_sha256"]) and
            runtime["device_ordinal"] == 0 and isinstance(runtime["device_name"], str) and
            runtime["device_name"] and isinstance(runtime["compute_capability"], list) and
            len(runtime["compute_capability"]) == 2 and
            all(isinstance(value, int) for value in runtime["compute_capability"]) and
            isinstance(runtime["runtime_version"], int) and
            isinstance(runtime["driver_version"], int), "frozen CuPy/CUDA runtime")
    coarse = document["coarse_result"]
    require(isinstance(coarse, Mapping) and set(coarse) == {
        "publication_directory", "audit_receipt_path", "audit_receipt_sha256",
        "coarse_member", "completion_member", "input_manifest_sha256"
    } and coarse["audit_receipt_sha256"] == COARSE_RESULT_AUDIT_FILE_SHA256 and
            coarse["coarse_member"] == {"name": "COARSE.bin",
                                         "bytes": COARSE_FRAME_BYTES,
                                         "sha256": COARSE_FRAME_SHA256} and
            isinstance(coarse["completion_member"], Mapping) and
            coarse["completion_member"].get("name") == "COMPLETE.json" and
            is_sha256(coarse["completion_member"].get("sha256")) and
            is_sha256(coarse["input_manifest_sha256"]), "coarse result pins")
    roles = document["roles"]
    require(isinstance(roles, list) and len(roles) == 3 and
            tuple(row.get("role") for row in roles) == ROLE_ORDER,
            "canonical role sources")
    for row in roles:
        require(isinstance(row, Mapping) and set(row) == {
            "role", "source_bf16_path", "source_bytes", "source_sha256",
            "coarse_reconstruction_f32_path", "coarse_reconstruction_bytes",
            "coarse_reconstruction_sha256"
        } and row["source_bytes"] == 3_145_728 and
                row["coarse_reconstruction_bytes"] == 6_291_456 and
                is_sha256(row["source_sha256"]) and
                is_sha256(row["coarse_reconstruction_sha256"]),
                "literal source/coarse reconstruction role")
    pilot = document["pilot"]
    require(isinstance(pilot, Mapping) and set(pilot) == {
        "geometry", "sample_blocks", "sample_fixed_before_access",
        "bootstrap_replicates", "bootstrap_alpha", "required_capture",
        "coarse_relative_mse", "target_d", "physical_bytes", "weights",
        "rate_numerator", "rate_denominator", "controls"
    } and pilot["geometry"] == GEOMETRY and
            pilot["sample_blocks"] == {key: list(SAMPLE_BLOCKS[key]) for key in ROLE_ORDER} and
            pilot["sample_fixed_before_access"] is True and
            pilot["bootstrap_replicates"] == 4096 and pilot["bootstrap_alpha"] == 0.05 and
            pilot["required_capture"] == REQUIRED_CAPTURE and
            pilot["coarse_relative_mse"] == COARSE_RELATIVE_MSE and
            pilot["target_d"] == TARGET_D and
            pilot["physical_bytes"] == EXPECTED_PHYSICAL_BYTES and
            pilot["weights"] == EXPECTED_WEIGHTS and
            pilot["rate_numerator"] == EXPECTED_RATE_NUMERATOR and
            pilot["rate_denominator"] == EXPECTED_RATE_DENOMINATOR and
            pilot["controls"] == {"phase": 1, "gaussian": 8},
            "frozen source-first aperture and survivor plan")
    require(isinstance(document["output_parent"], str) and document["output_parent"],
            "output parent")


def _authenticate_audit(row: Mapping[str, Any],
                        audited_manifest_sha256: str,
                        audited_source_root_sha256: str) -> dict[str, Any]:
    root = Path(row["package_path"]).resolve(strict=True)
    require(root == Path(row["package_path"]).absolute() and root.is_dir() and
            not Path(row["package_path"]).is_symlink(),
            "runtime audit canonical non-link directory")
    manifest_path = root / "SOURCE_MANIFEST.json"
    manifest_payload, _ = regular_bytes(manifest_path, "runtime audit manifest", 16 << 20)
    require(sha256(manifest_payload) == row["manifest_sha256"],
            "runtime audit manifest pin")
    manifest = strict_json(manifest_payload, "runtime audit manifest", canonical=False)
    rows = manifest.get("members")
    root_value = manifest.get(row["root_field"])
    require(isinstance(rows, list) and root_value == row["source_root_sha256"],
            "runtime audit source root")
    observed = []
    names = []
    for member in rows:
        require(isinstance(member, Mapping) and
                set(member) == {"name", "bytes", "sha256"} and
                isinstance(member["name"], str) and
                member["name"] == Path(member["name"]).name and
                member["name"] not in names,
                "runtime audit canonical member row")
        payload, _ = regular_bytes(root / member["name"], "runtime audit member", 64 << 20)
        item = {"name": member["name"], "bytes": len(payload), "sha256": sha256(payload)}
        require(item == member, "runtime audit member pin")
        names.append(member["name"])
        observed.append(item)
    domain = bytes.fromhex(row["root_domain_hex"])
    require(rows == sorted(rows, key=lambda member: member["name"].encode("utf-8")) and
            sha256(domain + canonical_json(observed)) == row["source_root_sha256"] and
            set(entry.name for entry in os.scandir(root)) == set(names) | {"SOURCE_MANIFEST.json"},
            "runtime audit exact closure")
    require(row["receipt_name"] == Path(row["receipt_name"]).name and
            row["receipt_name"] in names, "runtime receipt is a pinned package member")
    receipt_payload, _ = regular_bytes(root / row["receipt_name"],
                                       "runtime independent PASS receipt", 16 << 20)
    require(sha256(receipt_payload) == row["receipt_sha256"],
            "runtime receipt separate pin")
    receipt = strict_json(receipt_payload, "runtime audit receipt", canonical=False)
    require(receipt.get("executed") is True and
            receipt.get("status") == "PASS_INDEPENDENT_AUDIT" and
            receipt.get("dummy") is False and receipt.get("self_authored") is False and
            receipt.get("auditor_authority_id") == row["auditor_authority_id"] and
            receipt.get("audit_kind") == row["kind"] and
            receipt.get("audited_manifest_sha256") == audited_manifest_sha256 and
            receipt.get("audited_source_root_sha256") == audited_source_root_sha256,
            "runtime independent executed PASS")
    metadata = os.stat(root, follow_symlinks=False)
    return {"root": root, "root_identity": (metadata.st_dev, metadata.st_ino),
            "receipt": receipt}


def authorize_production(capability_path: Path) -> dict[str, Any]:
    """Authenticate every source/audit/payload object after a compiled precommit."""
    require(TRUSTED_CAPABILITY_SHA256 is not None and
            is_sha256(TRUSTED_CAPABILITY_SHA256),
            "HOLD: compile-time external capability SHA-256 is None")
    payload, _ = regular_bytes(capability_path, "external capability", 16 << 20)
    require(sha256(payload) == TRUSTED_CAPABILITY_SHA256,
            "external capability compile-time pin")
    document = strict_json(payload, "external capability")
    _validate_source_test_document(document)
    require(document["evidence_class"] == "PRODUCTION", "production capability class")
    closures = {key: _flat_closure(document["closures"][key]) for key in KNOWN_CLOSURES}
    audit_targets = {
        "v3_atomic_cupy_runtime": (
            KNOWN_CLOSURES["v3_atomic"]["manifest_sha256"],
            KNOWN_CLOSURES["v3_atomic"]["source_root_sha256"]),
        "pilot_runner_independent_source": (
            document["pilot_source_pins"]["manifest_sha256"],
            document["pilot_source_pins"]["source_root_sha256"]),
    }
    audits = {row["kind"]: _authenticate_audit(
        row, *audit_targets[row["kind"]]) for row in document["runtime_audits"]}
    require(len({entry["root_identity"] for entry in audits.values()}) == 2,
            "runtime audit package identity alias")

    coarse = document["coarse_result"]
    audit_payload, _ = regular_bytes(Path(coarse["audit_receipt_path"]),
                                     "coarse result audit", 32 << 20)
    require(sha256(audit_payload) == COARSE_RESULT_AUDIT_FILE_SHA256,
            "coarse result audit file pin")
    audit = strict_json(audit_payload, "coarse result audit", canonical=False)
    require(audit.get("status") == COARSE_AUDIT_STATUS and
            audit.get("literal_COARSE_canonical_reencode_matches") is True and
            audit.get("input_manifest_sha256") == coarse["input_manifest_sha256"],
            "coarse result independent audit")
    publication = Path(coarse["publication_directory"]).resolve(strict=True)
    expected_members = {row["name"]: row for row in audit["publication_members"]}
    require(set(entry.name for entry in os.scandir(publication)) == set(expected_members),
            "coarse publication exact closure")
    publication_payloads = {}
    for name, row in expected_members.items():
        member, _ = regular_bytes(publication / name, f"coarse publication {name}", 64 << 20)
        require(len(member) == row["bytes"] and sha256(member) == row["sha256"],
                f"coarse publication member pin {name}")
        publication_payloads[name] = member
    require(sha256(publication_payloads["COARSE.bin"]) == COARSE_FRAME_SHA256 and
            len(publication_payloads["COARSE.bin"]) == COARSE_FRAME_BYTES,
            "literal audited coarse frame")

    role_payloads = {}
    identities = set()
    audit_roles = {row["role"]: row for row in audit["input_roles"]}
    audit_reconstruction = audit["recomputed"]["reconstruction_f32_sha256"]
    for row in document["roles"]:
        role = row["role"]
        source, source_identity = regular_bytes(
            Path(row["source_bf16_path"]), f"{role} BF16 source", row["source_bytes"])
        reconstruction, recon_identity = regular_bytes(
            Path(row["coarse_reconstruction_f32_path"]),
            f"{role} independently decoded coarse reconstruction",
            row["coarse_reconstruction_bytes"])
        require(source_identity not in identities and recon_identity not in identities,
                "role payload inode alias")
        identities.update((source_identity, recon_identity))
        require(len(source) == row["source_bytes"] and
                sha256(source) == row["source_sha256"] and
                audit_roles[role]["sha256"] == row["source_sha256"] and
                len(reconstruction) == row["coarse_reconstruction_bytes"] and
                sha256(reconstruction) == row["coarse_reconstruction_sha256"] ==
                audit_reconstruction[role], "audited literal source/reconstruction")
        role_payloads[role] = {"source_bf16": source,
                               "coarse_reconstruction_f32": reconstruction,
                               "source_sha256": row["source_sha256"],
                               "coarse_reconstruction_sha256":
                                   row["coarse_reconstruction_sha256"]}
    return {"document": document, "closures": closures, "audits": audits,
            "coarse_bytes": publication_payloads["COARSE.bin"],
            "role_payloads": role_payloads,
            "capability_sha256": TRUSTED_CAPABILITY_SHA256,
            "model_payload_access_authorized": True}
