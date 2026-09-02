#!/usr/bin/env python3
"""Standard-library verifier; never imports numeric or accelerator modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED = {
    "AUTHORIZATION_TEMPLATE.json",
    "BLOCK.json",
    "README.md",
    "SOURCE_MANIFEST.json",
    "design_lock.json",
    "moment_contract.py",
    "runtime_pins.json",
    "source_moment_publisher.py",
    "test_hostile.py",
    "test_source_only.py",
    "verify_source.py",
}
MANIFEST_SCHEMA = "uwfa-sc-v9-bf16-source-moment-source-manifest-v0"
MANIFEST_DOMAIN = b"UWFA-SC-V9-BF16-SOURCE-MOMENT-PACKAGE-v0\x00"
ROLES = ("gate", "up", "down")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def canonical_pretty(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def members_root(rows: Sequence[Mapping[str, Any]]) -> str:
    value = hashlib.sha256(MANIFEST_DOMAIN)
    for row in rows:
        name = str(row["name"]).encode("ascii")
        value.update(struct.pack("<Q", len(name)))
        value.update(name)
        value.update(struct.pack("<Q", int(row["bytes"])))
        value.update(bytes.fromhex(str(row["sha256"])))
    return value.hexdigest()


def load_canonical_json(path: Path) -> Any:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"canonical JSON {path.name}") from exc
    require(payload == canonical_pretty(value), f"canonical pretty JSON {path.name}")
    return value


def verify_manifest(package: Path) -> tuple[dict[str, Any], str]:
    observed = {path.name for path in package.iterdir() if path.is_file()}
    require(observed == REQUIRED, f"package member set: {sorted(observed)}")
    manifest_path = package / "SOURCE_MANIFEST.json"
    manifest = load_canonical_json(manifest_path)
    require(isinstance(manifest, dict), "source manifest object")
    require(
        set(manifest) == {
            "claim_boundary",
            "members",
            "members_root_sha256",
            "schema",
            "source_only_attestation",
            "status",
        },
        "source manifest fields",
    )
    require(manifest["schema"] == MANIFEST_SCHEMA, "source manifest schema")
    require(manifest["status"] == "SOURCE_ONLY_FROZEN_NONPROMOTING", "source manifest status")
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(REQUIRED) - 1, "manifest member count")
    names = [row.get("name") for row in rows]
    require(names == sorted(REQUIRED - {"SOURCE_MANIFEST.json"}), "manifest member order")
    for row in rows:
        require(set(row) == {"bytes", "name", "sha256"}, f"manifest row fields {row.get('name')}")
        require(type(row["bytes"]) is int and row["bytes"] >= 0, f"manifest bytes {row['name']}")
        require(is_sha256(row["sha256"]), f"manifest digest syntax {row['name']}")
        path = package / row["name"]
        require(path.stat().st_size == row["bytes"], f"manifest bytes {row['name']}")
        require(sha256_file(path) == row["sha256"], f"manifest digest {row['name']}")
    require(is_sha256(manifest["members_root_sha256"]), "manifest members root syntax")
    require(manifest["members_root_sha256"] == members_root(rows), "manifest members root")
    attestation = manifest["source_only_attestation"]
    require(isinstance(attestation, dict) and attestation, "manifest attestation")
    require(all(value is False for value in attestation.values()), "source-only attestation must be all false")
    return manifest, sha256_file(manifest_path)


def verify_template(package: Path) -> None:
    template = load_canonical_json(package / "AUTHORIZATION_TEMPLATE.json")
    require(template.get("schema") == "uwfa-sc-v9-bf16-source-authorization-template-v0", "template schema")
    require(template.get("status") == "TEMPLATE_ONLY_NOT_AUTHORIZATION", "template status")
    require("authorization_sha256" not in template, "template must not be self-authorized")
    panel = template.get("panel")
    require(
        panel
        == {
            "experts": 6,
            "hidden": 2048,
            "identity_semantics": "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY",
            "intermediate": 768,
            "roles": ["gate", "up", "down"],
            "weights": 28_311_552,
        },
        "template panel",
    )
    rows = template.get("matrices")
    require(isinstance(rows, list) and len(rows) == 18, "template matrix count")
    for ordinal, row in enumerate(rows):
        slot = ordinal // 3
        role = ROLES[ordinal % 3]
        shape = [2048, 768] if role == "down" else [768, 2048]
        expected = {
            "bytes": 3_145_728,
            "matrix_ordinal": ordinal,
            "role": role,
            "shape": shape,
            "slot": slot,
            "source_matrix_bf16_sha256": "__EXTERNAL_LOWERCASE_SHA256__",
            "source_relpath": f"matrix_{ordinal:02d}_slot_{slot:02d}_{role}.bf16",
            "values": 1_572_864,
        }
        require(row == expected, f"template row {ordinal}")
    closure = template.get("source_closure")
    require(isinstance(closure, dict) and len(closure) == 6, "template source closure")
    require(all(value == "__EXTERNAL_LOWERCASE_SHA256__" for value in closure.values()), "template closure placeholders")


def verify_locks(package: Path) -> dict[str, Any]:
    runtime = load_canonical_json(package / "runtime_pins.json")
    require(runtime.get("schema") == "uwfa-sc-v9-bf16-source-moment-runtime-pins-v0", "runtime schema")
    require(runtime.get("status") == "FROZEN_BEFORE_ANY_SOURCE_ACCESS", "runtime status")
    moment = runtime["moment_runtime"]
    require(moment["python_implementation"] == "cpython", "runtime Python implementation")
    require(moment["python_version"] == "3.12.3", "runtime Python version")
    require(moment["numpy"]["version"] == "2.5.2", "runtime NumPy version")
    require(moment["byteorder"] == "little", "runtime byte order")
    for field in ("python_executable_sha256",):
        require(is_sha256(moment[field]), f"runtime pin {field}")
    for field in ("origin_sha256", "record_sha256"):
        require(is_sha256(moment["numpy"][field]), f"runtime NumPy pin {field}")
    gate = runtime["publication_gate"]
    require(gate["direct_entrypoint_authorized"] is False, "direct entrypoint block")
    require(gate["source_open_before_package_authorization_runtime_checks"] is False, "pre-source gate")

    design = load_canonical_json(package / "design_lock.json")
    require(design.get("schema") == "uwfa-sc-v9-bf16-source-moment-design-v0", "design schema")
    require(design.get("status") == "SOURCE_ONLY_CONTRACT_BUILT_EXTERNAL_AUTHORIZATION_BLOCKED", "design status")
    require(design["authorization_contract"]["matrix_count"] == 18, "design matrix count")
    require(design["authorization_contract"]["total_source_bytes"] == 56_623_104, "design source bytes")
    require(design["publication_contract"]["source_bytes_all_authenticated_before_first_moment"] is True, "design auth-before-moment")

    block = load_canonical_json(package / "BLOCK.json")
    require(block.get("schema") == "uwfa-sc-v9-bf16-source-moment-block-v0", "block schema")
    require(block["payload_access_authority"] is False, "payload block")
    require(block["positive_claim_authority"] is False, "claim block")
    require(all(value is False for value in block["source_only_attestation"].values()), "block attestation")
    return runtime


def verify_runtime_source(package: Path) -> None:
    contract = (package / "moment_contract.py").read_text(encoding="utf-8")
    publisher = (package / "source_moment_publisher.py").read_text(encoding="utf-8")
    runtime = (contract + "\n" + publisher).lower()
    for forbidden in (
        "import cupy",
        "import torch",
        "import requests",
        "qwen",
        "model.layers.",
        "runpod",
    ):
        require(forbidden not in runtime, f"forbidden runtime dependency/identity: {forbidden}")
    for required in (
        "parse_external_authorization",
        "load_authenticated_payloads",
        "build_moment_contract",
        "regenerate_gaussian_bf16",
        "_atomic_publish_records",
        "validate_exact_runtime",
        "direct_main",
    ):
        require(re.search(rf"\b{re.escape(required)}\b", contract + publisher) is not None, f"runtime boundary {required}")
    require("return 3" in contract and "return 3" in publisher, "inert direct exit")
    require("os.rename(stage, output)" in contract, "atomic final publication")
    require("payloads = load_authenticated_payloads" in contract, "all-byte authentication")
    require(contract.index("payloads = load_authenticated_payloads") < contract.index("moment_contract = build_moment_contract"), "authenticate before moments")


def verify_consumer_pins(runtime: Mapping[str, Any], repository_root: Path) -> int:
    root = repository_root.resolve(strict=True)
    pins = runtime["consumer_source_pins"]
    package = root / pins["package_relpath"]
    require(package.resolve(strict=True).is_dir(), "pinned consumer package")
    manifest = package / "SOURCE_MANIFEST.json"
    manifest_pin = pins["source_manifest"]
    require(manifest.stat().st_size == manifest_pin["bytes"], "consumer manifest bytes")
    require(sha256_file(manifest) == manifest_pin["sha256"], "consumer manifest digest")
    checks = 1
    for row in pins["sources"]:
        path = package / row["relpath"]
        require(path.stat().st_size == row["bytes"], f"consumer bytes {row['relpath']}")
        require(sha256_file(path) == row["sha256"], f"consumer digest {row['relpath']}")
        checks += 1
    receipt_pin = pins["cpu_receipt"]
    receipt = root / receipt_pin["relpath"]
    require(receipt.stat().st_size == receipt_pin["bytes"], "consumer receipt bytes")
    require(sha256_file(receipt) == receipt_pin["sha256"], "consumer receipt digest")
    receipt_record = json.loads(receipt.read_text(encoding="utf-8"))
    require(receipt_record["source_manifest_sha256"] == receipt_pin["source_manifest_sha256"], "consumer receipt manifest pin")
    return checks + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    require(package.is_dir(), "package directory")
    manifest, manifest_hash = verify_manifest(package)
    verify_template(package)
    runtime = verify_locks(package)
    verify_runtime_source(package)
    consumer_checks = 0
    if args.repository_root is not None:
        consumer_checks = verify_consumer_pins(runtime, args.repository_root)
    report = {
        "status": "PASS_SOURCE_ONLY_UNIVERSAL_MOMENT_CONTRACT",
        "package": str(package),
        "members": len(REQUIRED),
        "members_root_sha256": manifest["members_root_sha256"],
        "source_manifest_sha256": manifest_hash,
        "consumer_pin_checks": consumer_checks,
        "bf16_payloads_opened": 0,
        "numeric_modules_imported": 0,
        "accelerator_modules_imported": 0,
        "payload_authority": False,
        "positive_claim_authority": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCK_SOURCE_VERIFICATION: {exc}", file=sys.stderr)
        raise
