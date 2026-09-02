#!/usr/bin/env python3
"""Physical-result authority for expert-local global STRATA RM swap v3.

The v3 boundary is source-only.  It authenticates the frozen v2 authority and
its independent review, then requires two separately frozen and successfully
executed audit packages: one for scientific provenance and one for the exact
WebAssembly decoder.  Every physical case is one routed SwiGLU expert and one
dedicated packet.  The decoder runs as zero-import WebAssembly and receives
only a pre-opened packet byte buffer in linear memory; it has no WASI, path,
file-descriptor, socket, subprocess, ctypes, or native-read capability.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


V2_SOURCE_ROOT_SHA256 = (
    "e9ce4c24017831fab50696c2c5d81739d1f24d8121075c3aa56612b9a77013c9")
V2_MANIFEST_SHA256 = (
    "1f1caf2884a8b0b8713f213a16a0a32194238b64969e9d9cf3aaa339ddb776be")
V2_REVIEW_SOURCE_ROOT_SHA256 = (
    "d642889efcf8c54173eb7659602181cb9e71e122ce11ff05da6b24e45c47a113")
V2_REVIEW_MANIFEST_SHA256 = (
    "c89e89e35bdbc36f4095e9939bb381b77d049963e07c846dbffd543541298b7b")
V3_MANIFEST_SCHEMA = "strata-rm-global-swap-v3-physical-authority-source-manifest"
RATE_MIN = 2.15
RATE_MAX = 2.5
TARGET_F = 0.8
MAX_COLD_READ_AMPLIFICATION = 2.0
MIN_SOURCE_SPECIFIC_BPW = 0.03
PAGE_BYTES = 4096
ROLE_ORDER = ("gate", "up", "down")
PRODUCTION_AUTHORIZATION = "AUDIT_ROUTED_EXPERT_GLOBAL_RM_SWAP_RESULT_V3"
HEX = frozenset("0123456789abcdef")


class AuthorityError(RuntimeError):
    """A v3 authority requirement failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AuthorityError("noncanonical JSON value") from exc


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
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
                AuthorityError(f"{label}: nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"{label}: strict JSON") from exc
    require(isinstance(value, dict), f"{label}: JSON object")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
    candidate = Path(path)
    try:
        before = candidate.lstat()
        require(stat.S_ISREG(before.st_mode) and not candidate.is_symlink(),
                f"{label}: regular non-link file")
        payload = candidate.read_bytes()
        after = candidate.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label}: read") from exc
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                            row.st_mtime_ns, row.st_mode)
    require(identity(before) == identity(after), f"{label}: changed during read")
    return payload


def real_directory(path: Path, label: str) -> Path:
    original = Path(path)
    try:
        before = original.lstat()
        require(stat.S_ISDIR(before.st_mode) and not original.is_symlink(),
                f"{label}: real non-link directory")
        resolved = original.resolve(strict=True)
        after = original.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label}: directory resolution") from exc
    require((before.st_dev, before.st_ino, before.st_mode) ==
            (after.st_dev, after.st_ino, after.st_mode),
            f"{label}: root changed during resolution")
    return resolved


def _safe_relative(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label}: relative path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts and
            "." not in pure.parts and "\\" not in value,
            f"{label}: safe POSIX relative path")
    return Path(*pure.parts)


def resolve_member(root: Path, relative: Any, label: str) -> Path:
    base = real_directory(root, f"{label} root")
    current = base
    try:
        for part in _safe_relative(relative, label).parts:
            current = current / part
            require(not stat.S_ISLNK(current.lstat().st_mode),
                    f"{label}: symlink component")
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise AuthorityError(f"{label}: resolution") from exc
    require(base in resolved.parents and resolved != base, f"{label}: containment")
    return resolved


def _member_root(rows: list[dict[str, Any]]) -> str:
    return sha256(canonical_json(rows))


def authenticate_flat_package(package: Path, *, manifest_name: str,
                              expected_manifest_sha256: str,
                              expected_source_root_sha256: str,
                              expected_schema: str) -> dict[str, Any]:
    require(is_sha256(expected_manifest_sha256) and
            is_sha256(expected_source_root_sha256), "package external pins")
    root = real_directory(package, "dependency package")
    manifest_payload = regular_bytes(root / manifest_name, "dependency manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "dependency manifest pin")
    manifest = strict_json(manifest_payload, "dependency manifest")
    require(manifest.get("schema") == expected_schema and
            manifest.get("source_root_sha256") == expected_source_root_sha256,
            "dependency schema/root")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "dependency members")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "dependency member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name and
                name not in names and name != manifest_name,
                "dependency member name")
        payload = regular_bytes(root / name, f"dependency member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"dependency member pin {name}")
        names.append(name)
        observed.append(item)
    require(_member_root(observed) == expected_source_root_sha256,
            "dependency independently recomputed root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {manifest_name} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "dependency exact regular closure")
    return {"path": str(root), "source_root_sha256": expected_source_root_sha256,
            "manifest_sha256": expected_manifest_sha256, "members": len(rows)}


def authenticate_v2_and_review(v2_package: Path,
                               review_package: Path) -> dict[str, Any]:
    v2 = authenticate_flat_package(
        v2_package, manifest_name="source_manifest.json",
        expected_manifest_sha256=V2_MANIFEST_SHA256,
        expected_source_root_sha256=V2_SOURCE_ROOT_SHA256,
        expected_schema="strata-rm-global-swap-v2-authority-source-manifest")
    review = authenticate_flat_package(
        review_package, manifest_name="source_manifest.json",
        expected_manifest_sha256=V2_REVIEW_MANIFEST_SHA256,
        expected_source_root_sha256=V2_REVIEW_SOURCE_ROOT_SHA256,
        expected_schema=(
            "strata-rm-global-swap-v2-authority-independent-source-review-manifest"))
    review_manifest = strict_json(
        regular_bytes(Path(review["path"]) / "source_manifest.json",
                      "v2 review manifest"), "v2 review manifest")
    require(review_manifest.get("producer_manifest_sha256") == V2_MANIFEST_SHA256 and
            review_manifest.get("producer_source_root_sha256") ==
            V2_SOURCE_ROOT_SHA256, "review-to-v2 binding")
    return {"v2": v2, "review": review,
            "status": "PASS_PINNED_V2_AND_INDEPENDENT_REVIEW"}


def authenticate_v3_package(package: Path,
                            expected_manifest_sha256: str) -> dict[str, Any]:
    require(is_sha256(expected_manifest_sha256), "v3 manifest external pin")
    root = real_directory(package, "v3 package")
    payload = regular_bytes(root / "source_manifest.json", "v3 manifest")
    require(sha256(payload) == expected_manifest_sha256, "v3 manifest pin")
    manifest = strict_json(payload, "v3 manifest")
    require(manifest.get("schema") == V3_MANIFEST_SCHEMA and
            manifest.get("v2_source_root_sha256") == V2_SOURCE_ROOT_SHA256 and
            manifest.get("v2_review_source_root_sha256") ==
            V2_REVIEW_SOURCE_ROOT_SHA256, "v3 lineage")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "v3 members")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "v3 member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name and
                name not in names and name != "source_manifest.json",
                "v3 member name")
        member = regular_bytes(root / name, f"v3 member {name}")
        item = {"name": name, "bytes": len(member), "sha256": sha256(member)}
        require(item == row, f"v3 member pin {name}")
        names.append(name)
        observed.append(item)
    require(_member_root(observed) == manifest.get("source_root_sha256"),
            "v3 source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {"source_manifest.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "v3 exact regular closure")
    return {"path": str(root), "source_root_sha256": manifest["source_root_sha256"],
            "manifest_sha256": expected_manifest_sha256, "member_rows": observed}


def _authenticate_audit_source_closure(
        root: Path, rows: Any, expected_source_root_sha256: str,
        reserved: set[str], label: str) -> list[str]:
    require(isinstance(rows, list) and rows and
            _member_root(rows) == expected_source_root_sha256,
            f"{label}: source root")
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                f"{label}: member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and name and
                name not in names and name not in reserved,
                f"{label}: member name")
        payload = regular_bytes(root / name, f"{label}: member {name}")
        require({"name": name, "bytes": len(payload), "sha256": sha256(payload)} == row,
                f"{label}: member pin {name}")
        names.append(name)
    return names


def _validate_one_expert_sources(rows: Any, route_id: str) -> tuple[int, int]:
    require(isinstance(rows, list) and len(rows) == 3,
            f"{route_id}: exactly one Gate/Up/Down triplet")
    roles = {}
    coordinates = set()
    for ordinal, row in enumerate(rows):
        required = {"ordinal", "role", "layer", "expert", "shape",
                    "relative_path", "bytes", "sha256"}
        require(isinstance(row, dict) and set(row) == required and
                row["ordinal"] == ordinal and row["role"] in ROLE_ORDER,
                f"{route_id}: source schema/order")
        shape = row["shape"]
        require(isinstance(row["layer"], int) and row["layer"] >= 0 and
                isinstance(row["expert"], int) and row["expert"] >= 0 and
                isinstance(shape, list) and len(shape) == 2 and
                all(isinstance(value, int) and value > 0 for value in shape) and
                row["bytes"] == 2 * shape[0] * shape[1] and
                is_sha256(row["sha256"]), f"{route_id}: source metadata")
        _safe_relative(row["relative_path"], f"{route_id}: source path")
        require(row["role"] not in roles, f"{route_id}: unique role")
        roles[row["role"]] = shape
        coordinates.add((row["layer"], row["expert"]))
    require(len(coordinates) == 1 and set(roles) == set(ROLE_ORDER) and
            [row["role"] for row in rows] == list(ROLE_ORDER) and
            roles["gate"] == roles["up"] and
            roles["down"] == [roles["gate"][1], roles["gate"][0]],
            f"{route_id}: one compatible routed expert")
    return next(iter(coordinates))


def _validate_scientific_capability(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "selection", "architecture_families", "routes"}
    require(set(record) == required and record["schema"] ==
            "strata-rm-global-swap-v3-scientific-capability",
            "scientific capability schema")
    selection = record["selection"]
    require(isinstance(selection, dict) and set(selection) ==
            {"pipeline_sha256", "frozen_before_test", "test_bytes_opened",
             "search_replayed_on_every_control"} and
            is_sha256(selection["pipeline_sha256"]) and
            selection["frozen_before_test"] is True and
            selection["test_bytes_opened"] == 0 and
            selection["search_replayed_on_every_control"] is True,
            "scientific selection boundary")
    families = record["architecture_families"]
    require(isinstance(families, list) and len(families) >= 2 and
            len(families) == len(set(families)) and
            all(isinstance(value, str) and value for value in families),
            "scientific architecture families")
    routes = record["routes"]
    require(isinstance(routes, list) and routes, "scientific routes")
    by_id = {}
    routed_coordinates = set()
    for row in routes:
        route_required = {
            "route_id", "kind", "architecture_family", "pipeline_sha256",
            "checkpoint_manifest_sha256", "tensor_manifest_sha256",
            "checkpoint_identity_sha256", "architecture_schema_sha256",
            "control_family", "paired_model_route_id", "generator_sha256",
            "seed_sha256", "moments_sha256", "required_control_route_ids",
            "sources"}
        require(isinstance(row, dict) and set(row) == route_required,
                "scientific route schema")
        route_id = row["route_id"]
        require(isinstance(route_id, str) and route_id and route_id not in by_id and
                row["architecture_family"] in families and
                row["pipeline_sha256"] == selection["pipeline_sha256"] and
                row["kind"] in {"qwen_bf16", "swiglu_moe_bf16",
                                "matched_gaussian_bf16"},
                "scientific route identity")
        coordinate = _validate_one_expert_sources(row["sources"], route_id)
        route_class = ("model" if row["kind"] != "matched_gaussian_bf16"
                       else f"control:{row['control_family']}")
        route_coordinate = (row["architecture_family"], route_class, *coordinate)
        require(route_coordinate not in routed_coordinates,
                "duplicate routed expert evidence")
        routed_coordinates.add(route_coordinate)
        if row["kind"] != "matched_gaussian_bf16":
            require(all(is_sha256(row[name]) for name in
                        ("checkpoint_manifest_sha256", "tensor_manifest_sha256",
                         "checkpoint_identity_sha256", "architecture_schema_sha256")) and
                    row["control_family"] is None and
                    row["paired_model_route_id"] is None and
                    row["generator_sha256"] is None and row["seed_sha256"] is None and
                    row["moments_sha256"] is None and
                    isinstance(row["required_control_route_ids"], list) and
                    row["required_control_route_ids"],
                    "model route provenance")
        else:
            require(all(row[name] is None for name in
                        ("checkpoint_manifest_sha256", "tensor_manifest_sha256",
                         "checkpoint_identity_sha256", "architecture_schema_sha256")) and
                    isinstance(row["control_family"], str) and
                    row["control_family"] and
                    isinstance(row["paired_model_route_id"], str) and
                    all(is_sha256(row[name]) for name in
                        ("generator_sha256", "seed_sha256", "moments_sha256")) and
                    row["required_control_route_ids"] == [],
                    "control route provenance")
        by_id[route_id] = row
    models = [row for row in routes if row["kind"] != "matched_gaussian_bf16"]
    controls = [row for row in routes if row["kind"] == "matched_gaussian_bf16"]
    require({row["architecture_family"] for row in models} == set(families) and
            any(row["kind"] == "qwen_bf16" for row in models),
            "every family plus Qwen model evidence")
    referenced = []
    for model in models:
        required_controls = model["required_control_route_ids"]
        require(len(required_controls) == len(set(required_controls)),
                "unique control routes")
        for control_id in required_controls:
            require(control_id in by_id and
                    by_id[control_id]["kind"] == "matched_gaussian_bf16" and
                    by_id[control_id]["paired_model_route_id"] == model["route_id"] and
                    by_id[control_id]["architecture_family"] ==
                    model["architecture_family"] and
                    [(source["layer"], source["expert"], source["shape"])
                     for source in by_id[control_id]["sources"]] ==
                    [(source["layer"], source["expert"], source["shape"])
                     for source in model["sources"]],
                    "exact matched-control routed expert")
            referenced.append(control_id)
    require(len(referenced) == len(set(referenced)) and
            set(referenced) == {row["route_id"] for row in controls},
            "exact matched-control closure")

    # A family name is not evidence if it aliases another family's model bytes.
    for field in ("checkpoint_manifest_sha256", "tensor_manifest_sha256",
                  "checkpoint_identity_sha256", "architecture_schema_sha256"):
        owners = {}
        for row in models:
            value = row[field]
            previous = owners.setdefault(value, row["architecture_family"])
            require(previous == row["architecture_family"],
                    f"cross-family {field} alias")
    source_hash_owners = {}
    source_paths = set()
    for row in routes:
        family = row["architecture_family"]
        for source in row["sources"]:
            previous_hash = source_hash_owners.setdefault(source["sha256"], family)
            require(previous_hash == family, "cross-family source-byte alias")
            require(source["relative_path"] not in source_paths,
                    "source-path alias across routed cases")
            source_paths.add(source["relative_path"])
    for model in models:
        model_hashes = {source["sha256"] for source in model["sources"]}
        for control_id in model["required_control_route_ids"]:
            require(model_hashes.isdisjoint(
                source["sha256"] for source in by_id[control_id]["sources"]),
                "model/control source-byte alias")
    return {"record": record, "routes": by_id, "models": models,
            "controls": controls, "cross_family_aliases_rejected": True}


def authenticate_scientific_audit_package(
        package: Path, *, expected_manifest_sha256: str,
        expected_source_root_sha256: str, expected_receipt_sha256: str,
        expected_capability_sha256: str) -> dict[str, Any]:
    """Authenticate scientific provenance as strongly as the decoder audit."""
    require(all(is_sha256(value) for value in
                (expected_manifest_sha256, expected_source_root_sha256,
                 expected_receipt_sha256, expected_capability_sha256)),
            "scientific audit out-of-band pins")
    root = real_directory(package, "scientific audit package")
    manifest_payload = regular_bytes(root / "source_manifest.json",
                                     "scientific audit manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "scientific audit manifest pin")
    manifest = strict_json(manifest_payload, "scientific audit manifest")
    require(canonical_json(manifest) + b"\n" == manifest_payload and
            set(manifest) == {"schema", "source_root_sha256", "receipt_name",
                              "capability_name", "capability_sha256", "members"} and
            manifest["schema"] ==
            "strata-rm-global-swap-v3-scientific-independent-audit-manifest" and
            manifest["source_root_sha256"] == expected_source_root_sha256 and
            manifest["receipt_name"] == "AUDIT_RECEIPT.json" and
            manifest["capability_name"] == "SCIENTIFIC_CAPABILITY.json" and
            manifest["capability_sha256"] == expected_capability_sha256,
            "scientific audit manifest binding")
    names = _authenticate_audit_source_closure(
        root, manifest["members"], expected_source_root_sha256,
        {"source_manifest.json", "AUDIT_RECEIPT.json", "SCIENTIFIC_CAPABILITY.json"},
        "scientific audit")
    require({entry.name for entry in os.scandir(root)} ==
            set(names) | {"source_manifest.json", "AUDIT_RECEIPT.json",
                          "SCIENTIFIC_CAPABILITY.json"} and
            all(entry.is_file(follow_symlinks=False) for entry in os.scandir(root)),
            "scientific audit exact closure")
    capability_payload = regular_bytes(root / "SCIENTIFIC_CAPABILITY.json",
                                       "scientific capability")
    require(sha256(capability_payload) == expected_capability_sha256,
            "scientific capability out-of-band pin")
    capability_record = strict_json(capability_payload, "scientific capability")
    require(canonical_json(capability_record) + b"\n" == capability_payload,
            "scientific capability canonical bytes")
    scientific = _validate_scientific_capability(capability_record)
    receipt_payload = regular_bytes(root / "AUDIT_RECEIPT.json",
                                    "scientific audit receipt")
    require(sha256(receipt_payload) == expected_receipt_sha256,
            "scientific audit receipt pin")
    receipt = strict_json(receipt_payload, "scientific audit receipt")
    require(canonical_json(receipt) + b"\n" == receipt_payload,
            "scientific audit receipt canonical bytes")
    required = {"schema", "executed", "status", "audit_source_root_sha256",
                "scientific_capability_sha256", "checkpoint_manifests_opened",
                "tensor_manifests_opened", "source_hashes_recomputed",
                "control_generator_replayed", "control_moments_recomputed",
                "family_identity_verified", "cross_family_aliases_rejected",
                "selection_replay_verified", "hostile_tests"}
    require(set(receipt) == required and receipt["schema"] ==
            "strata-rm-global-swap-v3-scientific-independent-audit-receipt" and
            receipt["executed"] is True and receipt["status"] ==
            "PASS_INDEPENDENT_SCIENTIFIC_PROVENANCE_AUDIT_V3" and
            receipt["audit_source_root_sha256"] == expected_source_root_sha256 and
            receipt["scientific_capability_sha256"] ==
            expected_capability_sha256 and
            all(receipt[name] is True for name in
                ("checkpoint_manifests_opened", "tensor_manifests_opened",
                 "source_hashes_recomputed", "control_generator_replayed",
                 "control_moments_recomputed", "family_identity_verified",
                 "cross_family_aliases_rejected", "selection_replay_verified")) and
            isinstance(receipt["hostile_tests"], int) and
            receipt["hostile_tests"] >= 12,
            "successful scientific provenance audit receipt")
    return {**scientific, "manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": expected_source_root_sha256,
            "receipt_sha256": expected_receipt_sha256,
            "capability_sha256": expected_capability_sha256,
            "status": "PASS_SEPARATELY_PINNED_SCIENTIFIC_AUDIT_AND_CAPABILITY"}


def authenticate_decoder_audit_package(
        package: Path, *, expected_manifest_sha256: str,
        expected_source_root_sha256: str, expected_receipt_sha256: str,
        expected_decoder_module_sha256: str, expected_sandbox_sha256: str
        ) -> dict[str, Any]:
    require(all(is_sha256(value) for value in
                (expected_manifest_sha256, expected_source_root_sha256,
                 expected_receipt_sha256, expected_decoder_module_sha256,
                 expected_sandbox_sha256)), "decoder audit out-of-band pins")
    root = real_directory(package, "decoder audit package")
    manifest_payload = regular_bytes(root / "source_manifest.json",
                                     "decoder audit manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "decoder audit manifest pin")
    manifest = strict_json(manifest_payload, "decoder audit manifest")
    require(canonical_json(manifest) + b"\n" == manifest_payload and
            set(manifest) == {"schema", "source_root_sha256", "receipt_name",
                              "decoder_module_name", "decoder_module_sha256",
                              "sandbox_sha256", "members"} and
            manifest["schema"] ==
            "strata-rm-global-swap-v3-wasm-decoder-independent-audit-manifest" and
            manifest["source_root_sha256"] == expected_source_root_sha256 and
            manifest["receipt_name"] == "AUDIT_RECEIPT.json" and
            manifest["decoder_module_name"] == "DECODER.wasm" and
            manifest["decoder_module_sha256"] == expected_decoder_module_sha256 and
            manifest["sandbox_sha256"] == expected_sandbox_sha256,
            "decoder audit manifest binding")
    names = _authenticate_audit_source_closure(
        root, manifest["members"], expected_source_root_sha256,
        {"source_manifest.json", "AUDIT_RECEIPT.json", "DECODER.wasm"},
        "decoder audit")
    require({entry.name for entry in os.scandir(root)} ==
            set(names) | {"source_manifest.json", "AUDIT_RECEIPT.json",
                          "DECODER.wasm"} and
            all(entry.is_file(follow_symlinks=False) for entry in os.scandir(root)),
            "decoder audit exact closure")
    module_payload = regular_bytes(root / "DECODER.wasm", "audited decoder module")
    require(sha256(module_payload) == expected_decoder_module_sha256,
            "audited decoder module pin")
    receipt_payload = regular_bytes(root / "AUDIT_RECEIPT.json",
                                    "decoder audit receipt")
    require(sha256(receipt_payload) == expected_receipt_sha256,
            "decoder audit receipt pin")
    receipt = strict_json(receipt_payload, "decoder audit receipt")
    require(canonical_json(receipt) + b"\n" == receipt_payload,
            "decoder audit receipt canonical bytes")
    required = {"schema", "executed", "status", "audit_source_root_sha256",
                "decoder_module_sha256", "sandbox_sha256", "zero_imports_verified",
                "no_wasi_verified", "abi_verified", "input_immutability_verified",
                "canonical_replay_verified", "fixed_universal_decoder",
                "qwen_specific_tables_absent", "hostile_tests", "payloads_opened"}
    require(set(receipt) == required and receipt["schema"] ==
            "strata-rm-global-swap-v3-wasm-decoder-independent-audit-receipt" and
            receipt["executed"] is True and receipt["status"] ==
            "PASS_INDEPENDENT_ZERO_IMPORT_WASM_DECODER_AUDIT_V3" and
            receipt["audit_source_root_sha256"] == expected_source_root_sha256 and
            receipt["decoder_module_sha256"] == expected_decoder_module_sha256 and
            receipt["sandbox_sha256"] == expected_sandbox_sha256 and
            all(receipt[name] is True for name in
                ("zero_imports_verified", "no_wasi_verified", "abi_verified",
                 "input_immutability_verified", "canonical_replay_verified",
                 "fixed_universal_decoder", "qwen_specific_tables_absent")) and
            isinstance(receipt["hostile_tests"], int) and
            receipt["hostile_tests"] >= 12 and receipt["payloads_opened"] == 0,
            "successful zero-import WebAssembly decoder audit")
    return {"module_payload": module_payload,
            "manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": expected_source_root_sha256,
            "receipt_sha256": expected_receipt_sha256,
            "decoder_module_sha256": expected_decoder_module_sha256,
            "status": "PASS_SEPARATELY_PINNED_ZERO_IMPORT_WASM_DECODER_AUDIT"}


def _strict_commitment(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(is_sha256(expected_sha256), "commitment external pin")
    payload = regular_bytes(path, "physical commitment")
    require(sha256(payload) == expected_sha256,
            "physical commitment out-of-band pin")
    record = strict_json(payload, "physical commitment")
    require(canonical_json(record) + b"\n" == payload,
            "physical commitment canonical bytes")
    required = {"schema", "mode", "v2_source_root_sha256",
                "v2_review_source_root_sha256", "decoder_module_sha256",
                "sandbox_sha256", "route_packets"}
    require(set(record) == required and record["schema"] ==
            "strata-rm-global-swap-v3-routed-expert-physical-commitment" and
            record["mode"] == "production_routed_expert" and
            record["v2_source_root_sha256"] == V2_SOURCE_ROOT_SHA256 and
            record["v2_review_source_root_sha256"] ==
            V2_REVIEW_SOURCE_ROOT_SHA256 and
            is_sha256(record["decoder_module_sha256"]) and
            is_sha256(record["sandbox_sha256"]), "physical commitment schema")
    rows = record["route_packets"]
    require(isinstance(rows, list) and rows, "route packet rows")
    route_ids = set()
    paths = set()
    hashes = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) ==
                {"route_id", "relative_path", "bytes", "sha256"} and
                isinstance(row["route_id"], str) and row["route_id"] and
                row["route_id"] not in route_ids and
                isinstance(row["bytes"], int) and row["bytes"] > 0 and
                is_sha256(row["sha256"]), "route packet schema")
        _safe_relative(row["relative_path"], "route packet path")
        require(row["relative_path"] not in paths and row["sha256"] not in hashes,
                "one distinct expert-local packet per route")
        route_ids.add(row["route_id"])
        paths.add(row["relative_path"])
        hashes.add(row["sha256"])
    return record


def _write_immutable(path: Path, payload: bytes, label: str) -> None:
    require(not path.exists(), f"{label}: destination fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    require(regular_bytes(path, label) == payload, f"{label}: snapshot parity")


def _sanitized_environment() -> dict[str, str]:
    allowed = ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT", "WINDIR", "COMSPEC",
               "PATHEXT")
    result = {name: os.environ[name] for name in allowed if name in os.environ}
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONHASHSEED"] = "0"
    return result


def _bf16_values(payload: bytes) -> array.array:
    require(payload and len(payload) % 2 == 0, "nonempty BF16 source")
    words = array.array("H")
    words.frombytes(payload)
    if sys.byteorder != "little":
        words.byteswap()
    wide = array.array("I", (int(value) << 16 for value in words))
    values = array.array("f")
    values.frombytes(wide.tobytes())
    return values


def _f64_values(payload: bytes) -> array.array:
    require(payload and len(payload) % 8 == 0, "nonempty FP64 reconstruction")
    values = array.array("d")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _score(source_payload: bytes, reconstruction_payload: bytes) -> dict[str, Any]:
    source = _bf16_values(source_payload)
    reconstruction = _f64_values(reconstruction_payload)
    require(len(source) == len(reconstruction), "source/reconstruction count")
    require(all(math.isfinite(value) for value in source) and
            all(math.isfinite(value) for value in reconstruction),
            "finite source/reconstruction")
    energy = math.fsum(float(value) ** 2 for value in source)
    sse = math.fsum((float(left) - float(right)) ** 2
                    for left, right in zip(source, reconstruction, strict=True))
    require(energy > 0.0 and math.isfinite(sse), "finite scoring domain")
    return {"weights": len(source), "sse_fp64_hex": sse.hex(),
            "energy_fp64_hex": energy.hex()}


def _read_pinned(root: Path, row: Mapping[str, Any], label: str) -> bytes:
    path = resolve_member(root, row["relative_path"], label)
    payload = regular_bytes(path, label)
    require(len(payload) == row["bytes"] and sha256(payload) == row["sha256"],
            f"{label}: literal pin")
    return payload


def _validate_sandbox_receipt(record: Mapping[str, Any], *, route_id: str,
                              packet: bytes, module_sha256: str,
                              sandbox_sha256: str) -> dict[str, Any]:
    required = {"schema", "route_id", "decoder_module_sha256", "sandbox_sha256",
                "module_imports", "wasi_enabled", "filesystem_api_exposed",
                "descriptor_api_exposed", "native_io_imports_exposed",
                "packet_buffer_preopened", "packet_input_unchanged",
                "packet_sha256", "literal_packet_bytes_supplied", "page_bytes",
                "pages_supplied", "zero_padding_bytes_supplied",
                "physical_page_bytes_supplied", "decode_status",
                "canonical_reencode_bytes", "status"}
    page_count = (len(packet) + PAGE_BYTES - 1) // PAGE_BYTES
    expected_pages = [
        {"page_index": index, "literal_offset": index * PAGE_BYTES,
         "literal_bytes": min(PAGE_BYTES, len(packet) - index * PAGE_BYTES),
         "supplied_bytes": PAGE_BYTES}
        for index in range(page_count)]
    require(set(record) == required and record["schema"] ==
            "strata-rm-global-swap-v3-zero-import-wasm-sandbox-receipt" and
            record["route_id"] == route_id and
            record["decoder_module_sha256"] == module_sha256 and
            record["sandbox_sha256"] == sandbox_sha256 and
            record["module_imports"] == [] and record["wasi_enabled"] is False and
            record["filesystem_api_exposed"] is False and
            record["descriptor_api_exposed"] is False and
            record["native_io_imports_exposed"] is False and
            record["packet_buffer_preopened"] is True and
            record["packet_input_unchanged"] is True and
            record["packet_sha256"] == sha256(packet) and
            record["literal_packet_bytes_supplied"] == len(packet) and
            record["page_bytes"] == PAGE_BYTES and
            record["pages_supplied"] == expected_pages and
            record["zero_padding_bytes_supplied"] ==
            page_count * PAGE_BYTES - len(packet) and
            record["physical_page_bytes_supplied"] == page_count * PAGE_BYTES and
            record["decode_status"] == 0 and
            record["canonical_reencode_bytes"] == len(packet) and
            record["status"] ==
            "PASS_ZERO_IMPORT_WASM_EXPERT_PACKET_BUFFER_DECODE",
            "zero-import WebAssembly sandbox receipt")
    return {"literal_packet_bytes": len(packet), "page_bytes": PAGE_BYTES,
            "pages_supplied": page_count,
            "physical_page_bytes": page_count * PAGE_BYTES,
            "cold_read_amplification": page_count * PAGE_BYTES / len(packet),
            "one_independently_routed_expert": True,
            "no_shared_or_common_stream": True}


def _run_route(*, route: Mapping[str, Any], packet_row: Mapping[str, Any],
               evidence_root: Path, module_payload: bytes, sandbox_payload: bytes,
               timeout_seconds: int) -> dict[str, Any]:
    packet = _read_pinned(evidence_root, packet_row,
                          f"packet {route['route_id']}")
    sources = [_read_pinned(evidence_root, source,
                            f"source {route['route_id']}:{source['role']}")
               for source in route["sources"]]
    request = {"schema": "strata-rm-global-swap-v3-wasm-route-request",
               "route_id": route["route_id"], "packet_sha256": sha256(packet),
               "packet_bytes": len(packet), "page_bytes": PAGE_BYTES,
               "sources": [{key: source[key] for key in
                            ("ordinal", "role", "layer", "expert", "shape")}
                           for source in route["sources"]]}
    with tempfile.TemporaryDirectory(prefix="strata-rm-v3-route-") as directory:
        root = Path(directory).resolve(strict=True)
        module = root / "decoder.wasm"
        sandbox = root / "wasm_decoder_sandbox.py"
        packet_path = root / "packet.bin"
        request_path = root / "request.json"
        output = root / "output"
        output.mkdir()
        _write_immutable(module, module_payload, "decoder module snapshot")
        _write_immutable(sandbox, sandbox_payload, "sandbox snapshot")
        _write_immutable(packet_path, packet, "expert packet snapshot")
        _write_immutable(request_path, canonical_json(request) + b"\n",
                         "route request snapshot")
        command = [sys.executable, "-I", "-B", str(sandbox),
                   "--module", str(module), "--packet", str(packet_path),
                   "--request", str(request_path), "--output-dir", str(output)]
        completed = subprocess.run(
            command, cwd=root, env=_sanitized_environment(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout_seconds, check=False)
        require(completed.returncode == 0,
                "WebAssembly decoder sandbox failed: " +
                completed.stderr.decode("utf-8", errors="replace")[-3000:])
        sandbox_receipt = strict_json(
            regular_bytes(output / "SANDBOX_RECEIPT.json", "sandbox receipt"),
            "sandbox receipt")
        read = _validate_sandbox_receipt(
            sandbox_receipt, route_id=route["route_id"], packet=packet,
            module_sha256=sha256(module_payload),
            sandbox_sha256=sha256(sandbox_payload))
        canonical = regular_bytes(output / "canonical_packet.bin",
                                  "canonical packet")
        reconstructions = [regular_bytes(
            output / f"reconstruction-{source['ordinal']:04d}.f64",
            f"reconstruction {source['ordinal']}") for source in route["sources"]]
        require(sha256(regular_bytes(module, "post-run decoder module")) ==
                sha256(module_payload) and
                sha256(regular_bytes(sandbox, "post-run sandbox")) ==
                sha256(sandbox_payload), "executable snapshots unchanged")
    require(canonical == packet, "canonical packet byte replay")
    scores = [_score(source, reconstruction)
              for source, reconstruction in zip(sources, reconstructions, strict=True)]
    weights = sum(row["weights"] for row in scores)
    sse = math.fsum(float.fromhex(row["sse_fp64_hex"]) for row in scores)
    energy = math.fsum(float.fromhex(row["energy_fp64_hex"]) for row in scores)
    rate = 8.0 * len(packet) / weights
    relative = sse / energy
    factor = relative * 2.0 ** (2.0 * rate)
    return {"route_id": route["route_id"], "kind": route["kind"],
            "architecture_family": route["architecture_family"],
            "control_family": route["control_family"],
            "paired_model_route_id": route["paired_model_route_id"],
            "layer": route["sources"][0]["layer"],
            "expert": route["sources"][0]["expert"],
            "weights": weights, "literal_packet_bytes": len(packet),
            "physical_rate_bpw": rate, "sse_fp64_hex": sse.hex(),
            "energy_fp64_hex": energy.hex(), "relative_mse": relative,
            "F": factor, "saving_bpw": -0.5 * math.log2(factor),
            "cold_read": read, "matrix_rows": scores,
            "canonical_reencode_byte_identical": True}


def _pool(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    require(rows, "nonempty result pool")
    weights = sum(row["weights"] for row in rows)
    bits = 8 * sum(row["literal_packet_bytes"] for row in rows)
    sse = math.fsum(float.fromhex(row["sse_fp64_hex"]) for row in rows)
    energy = math.fsum(float.fromhex(row["energy_fp64_hex"]) for row in rows)
    rate = bits / weights
    relative = sse / energy
    factor = relative * 2.0 ** (2.0 * rate)
    return {"weights": weights, "physical_bits": bits,
            "physical_rate_bpw": rate, "sse_fp64_hex": sse.hex(),
            "energy_fp64_hex": energy.hex(), "relative_mse": relative,
            "F": factor, "saving_bpw": -0.5 * math.log2(factor),
            "maximum_routed_expert_cold_read_amplification": max(
                row["cold_read"]["cold_read_amplification"] for row in rows)}


def evaluate_acceptance(results: list[Mapping[str, Any]],
                        scientific_record: Mapping[str, Any],
                        *, enforce: bool = True) -> dict[str, Any]:
    routes = {row["route_id"]: row for row in scientific_record["routes"]}
    by_id = {row["route_id"]: row for row in results}
    require(set(routes) == set(by_id), "exact routed result closure")
    family_results = []
    for family in scientific_record["architecture_families"]:
        models = [row for row in results
                  if row["architecture_family"] == family and
                  row["kind"] != "matched_gaussian_bf16"]
        require(models, f"family {family}: routed models")
        model_pool = _pool(models)
        controls_by_family: dict[str, list[Mapping[str, Any]]] = {}
        for model in models:
            for control_id in routes[model["route_id"]]["required_control_route_ids"]:
                control = by_id[control_id]
                controls_by_family.setdefault(control["control_family"], []).append(control)
        require(controls_by_family and
                all(len(rows) == len(models) for rows in controls_by_family.values()),
                f"family {family}: complete control panel")
        control_pools = {name: _pool(rows)
                         for name, rows in sorted(controls_by_family.items())}
        strongest_name, strongest = max(
            control_pools.items(), key=lambda item: item[1]["saving_bpw"])
        advantage = model_pool["saving_bpw"] - strongest["saving_bpw"]
        passed = (RATE_MIN <= model_pool["physical_rate_bpw"] <= RATE_MAX and
                  model_pool["F"] <= TARGET_F and
                  model_pool["maximum_routed_expert_cold_read_amplification"] <
                  MAX_COLD_READ_AMPLIFICATION and
                  all(RATE_MIN <= pool["physical_rate_bpw"] <= RATE_MAX and
                      pool["maximum_routed_expert_cold_read_amplification"] <
                      MAX_COLD_READ_AMPLIFICATION
                      for pool in control_pools.values()) and
                  advantage >= MIN_SOURCE_SPECIFIC_BPW)
        if enforce:
            require(passed, f"family {family}: routed physical acceptance")
        family_results.append({"architecture_family": family,
                               "model": model_pool, "controls": control_pools,
                               "strongest_control": strongest_name,
                               "source_specific_advantage_bpw": advantage,
                               "passed": passed})
    qwen = _pool([row for row in results if row["kind"] == "qwen_bf16"])
    if enforce:
        require(qwen["F"] <= TARGET_F, "absolute Qwen F <= 0.8")
    return {"families": family_results, "qwen": qwen,
            "all_families_passed": all(row["passed"] for row in family_results),
            "cold_read_unit": "one independently decoded expert-local packet",
            "page_bytes": PAGE_BYTES}


def validate_physical_bundle(
        *, v3_package: Path, expected_v3_manifest_sha256: str,
        evidence_root: Path, commitment_path: Path,
        expected_commitment_sha256: str,
        scientific_audit_package: Path,
        expected_scientific_manifest_sha256: str,
        expected_scientific_source_root_sha256: str,
        expected_scientific_receipt_sha256: str,
        expected_scientific_capability_sha256: str,
        decoder_audit_package: Path,
        expected_decoder_manifest_sha256: str,
        expected_decoder_source_root_sha256: str,
        expected_decoder_receipt_sha256: str,
        expected_decoder_module_sha256: str,
        authorization: str, timeout_seconds: int = 3600) -> dict[str, Any]:
    require(authorization == PRODUCTION_AUTHORIZATION,
            "explicit routed-expert physical authorization")
    v3 = authenticate_v3_package(v3_package, expected_v3_manifest_sha256)
    sandbox_payload = regular_bytes(
        Path(v3["path"]) / "wasm_decoder_sandbox.py", "v3 sandbox")
    scientific = authenticate_scientific_audit_package(
        scientific_audit_package,
        expected_manifest_sha256=expected_scientific_manifest_sha256,
        expected_source_root_sha256=expected_scientific_source_root_sha256,
        expected_receipt_sha256=expected_scientific_receipt_sha256,
        expected_capability_sha256=expected_scientific_capability_sha256)
    decoder = authenticate_decoder_audit_package(
        decoder_audit_package,
        expected_manifest_sha256=expected_decoder_manifest_sha256,
        expected_source_root_sha256=expected_decoder_source_root_sha256,
        expected_receipt_sha256=expected_decoder_receipt_sha256,
        expected_decoder_module_sha256=expected_decoder_module_sha256,
        expected_sandbox_sha256=sha256(sandbox_payload))
    evidence = real_directory(evidence_root, "evidence root")
    try:
        relative = str(Path(commitment_path).resolve(strict=True).relative_to(evidence)
                       ).replace(os.sep, "/")
    except (OSError, ValueError) as exc:
        raise AuthorityError("commitment inside evidence root") from exc
    commitment = _strict_commitment(
        resolve_member(evidence, relative, "physical commitment"),
        expected_commitment_sha256)
    require(commitment["decoder_module_sha256"] ==
            decoder["decoder_module_sha256"] and
            commitment["sandbox_sha256"] == sha256(sandbox_payload),
            "commitment-to-audited decoder/sandbox binding")
    packet_rows = {row["route_id"]: row for row in commitment["route_packets"]}
    require(set(packet_rows) == set(scientific["routes"]),
            "one packet for every exact audited route")
    results = [_run_route(
        route=route, packet_row=packet_rows[route_id], evidence_root=evidence,
        module_payload=decoder["module_payload"], sandbox_payload=sandbox_payload,
        timeout_seconds=timeout_seconds)
        for route_id, route in scientific["routes"].items()]
    acceptance = evaluate_acceptance(results, scientific["record"], enforce=True)
    return {"schema": "strata-rm-global-swap-v3-routed-physical-result",
            "commitment_sha256": expected_commitment_sha256,
            "scientific_audit": {key: scientific[key] for key in
                                 ("manifest_sha256", "source_root_sha256",
                                  "receipt_sha256", "capability_sha256", "status")},
            "decoder_audit": {key: decoder[key] for key in
                              ("manifest_sha256", "source_root_sha256",
                               "receipt_sha256", "decoder_module_sha256", "status")},
            "routes": results, "acceptance": acceptance,
            "one_decode_per_routed_expert": True,
            "expert_local_packet_only": True,
            "shared_streams_permitted": False,
            "decoder_has_path_fd_or_native_read_capability": False,
            "caller_supplied_metrics_accepted": False,
            "status":
            "PASS_AUDITED_UNIVERSAL_SWI_GLU_TARGET_WITH_ROUTED_PAGE_READS"}
