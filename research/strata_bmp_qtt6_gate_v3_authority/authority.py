#!/usr/bin/env python3
"""Fail-closed evidence authority for STRATA BMP/OBDD/QTT6 v3.

The module validates exact external capability closures and their executed,
independently audited receipts.  The shipped source freeze intentionally has no
trusted launch-manifest digest, so :func:`authorize_production` always refuses
payload access.  A later, independently reviewed deployment sibling must pin a
single launch manifest in source before that entry point can run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


V2_MANIFEST_SHA256 = (
    "84df0d32a55682f6565ac9d144f7de850acf77cde27bffdefa77a151211906f8")
V2_SOURCE_ROOT_SHA256 = (
    "b518b203c43fd401c94e1bfcf67e029a85a95f1f7ce244fcd864a96d0780da47")
V2_AUDIT_MANIFEST_SHA256 = (
    "324e9a6d7d16be7b57b4ae33599cce2e4b324848e279b59268826b5dcaaebd12")
V2_AUDIT_SOURCE_ROOT_SHA256 = (
    "c817b1f1c3c270cb1f0e332262dc46df4fe9eb39c4b4fafe70a23536203572d3")

PACKAGE_AUTHORITY_ID = "strata-bmp-qtt6-v3-source-author"
TRUSTED_LAUNCH_MANIFEST_SHA256: str | None = None
PAGE_BYTES = 4096
RATE_MIN = 2.15
RATE_MAX = 2.5
TARGET_F = 0.8
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
CAPABILITY_KINDS = (
    "predecessor_source_audit",
    "gaussian_control_generator",
    "current_strata_adapter",
    "independent_bf16_scorer",
    "routed_page_reader",
    "independent_launch_audit",
)
EXECUTION_STATUS = {
    kind: "PASS_EXECUTED_" + kind.upper() for kind in CAPABILITY_KINDS
}
CAPABILITY_SCHEMA = "strata-bmp-qtt6-v3-executed-capability-manifest-v1"
EXECUTION_SCHEMA = "strata-bmp-qtt6-v3-capability-execution-receipt-v1"
AUDIT_SCHEMA = "strata-bmp-qtt6-v3-independent-capability-audit-receipt-v1"
LAUNCH_SCHEMA = "strata-bmp-qtt6-v3-precommitted-launch-manifest-v1"


class AuthorityError(RuntimeError):
    """Evidence, provenance, metric, or read-trace authorization failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
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
                AuthorityError(f"{label}: nonfinite JSON {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"{label}: strict JSON") from exc
    require(isinstance(value, dict), f"{label}: JSON object")
    if canonical:
        require(payload == canonical_json(value) + b"\n",
                f"{label}: canonical JSON plus LF")
    return value


def real_directory(path: Path, label: str) -> Path:
    require(isinstance(path, Path), f"{label}: pathlib.Path")
    try:
        before = path.lstat()
        require(stat.S_ISDIR(before.st_mode) and not path.is_symlink(),
                f"{label}: real directory")
        root = path.resolve(strict=True)
        after = path.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label}: directory access") from exc
    require((before.st_dev, before.st_ino, before.st_mode) ==
            (after.st_dev, after.st_ino, after.st_mode),
            f"{label}: changed during resolution")
    return root


def regular_bytes(path: Path, label: str) -> tuple[bytes, tuple[int, int]]:
    require(isinstance(path, Path), f"{label}: pathlib.Path")
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                f"{label}: regular non-link")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label}: stable read") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size,
                       before.st_mtime_ns, before.st_mode)
    identity_after = (after.st_dev, after.st_ino, after.st_size,
                      after.st_mtime_ns, after.st_mode)
    require(identity_before == identity_after,
            f"{label}: changed during read")
    return payload, (before.st_dev, before.st_ino)


def resolve_member(root: Path, relative: Any, label: str, *, directory: bool = False) -> Path:
    require(isinstance(relative, str) and relative and "\\" not in relative,
            f"{label}: portable relative path")
    item = Path(relative)
    require(not item.is_absolute() and all(part not in ("", ".", "..")
                                           for part in item.parts),
            f"{label}: confined relative path")
    try:
        cursor = root
        for part in item.parts:
            cursor = cursor / part
            require(not cursor.is_symlink(), f"{label}: symlink component")
        resolved = (root / item).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise AuthorityError(f"{label}: confined member") from exc
    if directory:
        require(resolved.is_dir() and not (root / item).is_symlink(),
                f"{label}: real directory member")
    return resolved


def _member_root(rows: list[dict[str, Any]]) -> str:
    return sha256(canonical_json(rows))


def authenticate_flat_package(package: Path, *, manifest_name: str,
                              expected_manifest_sha256: str,
                              expected_source_root_sha256: str,
                              expected_schema: str,
                              source_root_field: str = "source_root_sha256") -> dict[str, Any]:
    """Open and independently re-hash every member of a flat source package."""
    require(is_sha256(expected_manifest_sha256) and
            is_sha256(expected_source_root_sha256), "dependency frozen pins")
    root = real_directory(package, "dependency package")
    manifest_payload, _ = regular_bytes(root / manifest_name, "dependency manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "dependency manifest external pin")
    manifest = strict_json(manifest_payload, "dependency manifest", canonical=False)
    require(manifest.get("schema") == expected_schema and
            manifest.get(source_root_field) == expected_source_root_sha256,
            "dependency schema/source-root pins")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "dependency member rows")
    observed: list[dict[str, Any]] = []
    names: list[str] = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "dependency member schema")
        name = row["name"]
        require(isinstance(name, str) and name not in names and
                "/" not in name and "\\" not in name and name != manifest_name,
                "dependency member name")
        payload, _ = regular_bytes(root / name, f"dependency member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"dependency member pin {name}")
        names.append(name)
        observed.append(item)
    require(names == sorted(names, key=lambda value: value.encode("utf-8")),
            "dependency canonical member order")
    require(_member_root(observed) == expected_source_root_sha256,
            "dependency observed source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {manifest_name} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "dependency exact regular closure")
    return {"manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": expected_source_root_sha256,
            "members": len(names), "path": str(root)}


def authenticate_pinned_predecessors(v2_package: Path,
                                     v2_audit_package: Path) -> dict[str, Any]:
    """Authenticate the actual v2 producer and auditor manifests and closures."""
    producer = authenticate_flat_package(
        v2_package, manifest_name="SOURCE_MANIFEST.json",
        expected_manifest_sha256=V2_MANIFEST_SHA256,
        expected_source_root_sha256=V2_SOURCE_ROOT_SHA256,
        expected_schema="strata-bmp-obdd-qtt6-replay-source-manifest-v2")
    audit = authenticate_flat_package(
        v2_audit_package, manifest_name="AUDIT_SOURCE_MANIFEST.json",
        expected_manifest_sha256=V2_AUDIT_MANIFEST_SHA256,
        expected_source_root_sha256=V2_AUDIT_SOURCE_ROOT_SHA256,
        expected_schema="strata-bmp-qtt6-v2-independent-audit-source-manifest-v1",
        source_root_field="audit_source_root_sha256")
    payload, _ = regular_bytes(
        Path(audit["path"]) / "AUDIT_SOURCE_MANIFEST.json", "v2 audit manifest")
    record = strict_json(payload, "v2 audit manifest", canonical=False)
    require(record.get("producer_pins") == {
        "source_manifest_sha256": V2_MANIFEST_SHA256,
        "source_root_sha256": V2_SOURCE_ROOT_SHA256,
    }, "v2 auditor binds producer pins")
    return {"producer": producer, "independent_source_audit": audit}


def _validate_identity(value: Any, label: str) -> str:
    require(isinstance(value, str) and SAFE_NAME.fullmatch(value) is not None,
            f"{label}: stable authority ID")
    return value


def _validate_common_execution(record: Mapping[str, Any], manifest: Mapping[str, Any],
                               implementation_sha256: str,
                               *, allow_source_test_fixture: bool) -> None:
    required = {
        "schema", "status", "executed", "evidence_class", "kind",
        "capability_id", "producer_authority_id", "executor_authority_id",
        "implementation_sha256", "invocation_sha256", "input_manifest_sha256",
        "output_sha256", "started_utc", "finished_utc", "test_fixture", "dummy",
        "self_authored", "details",
    }
    require(set(record) == required and record["schema"] == EXECUTION_SCHEMA and
            record["kind"] == manifest["kind"] and
            record["capability_id"] == manifest["capability_id"] and
            record["producer_authority_id"] == manifest["producer_authority_id"] and
            record["executor_authority_id"] == manifest["executor_authority_id"] and
            record["implementation_sha256"] == implementation_sha256 and
            record["status"] == EXECUTION_STATUS[manifest["kind"]] and
            record["executed"] is True and
            all(is_sha256(record[name]) for name in
                ("invocation_sha256", "input_manifest_sha256", "output_sha256")) and
            isinstance(record["started_utc"], str) and record["started_utc"] and
            isinstance(record["finished_utc"], str) and record["finished_utc"] and
            isinstance(record["details"], dict), "executed capability receipt")
    if allow_source_test_fixture:
        require(record["evidence_class"] == "SOURCE_TEST_FIXTURE" and
                record["test_fixture"] is True and record["dummy"] is True and
                record["self_authored"] is True,
                "honest source-test capability labelling")
    else:
        require(record["evidence_class"] == "PRODUCTION_EXECUTION" and
                record["test_fixture"] is False and record["dummy"] is False and
                record["self_authored"] is False,
                "production capability cannot be fixture, dummy, or self-authored")


def _validate_common_audit(record: Mapping[str, Any], manifest: Mapping[str, Any],
                           implementation_sha256: str, execution_sha256: str,
                           *, allow_source_test_fixture: bool) -> None:
    required = {
        "schema", "status", "executed", "evidence_class", "kind",
        "capability_id", "producer_authority_id", "auditor_authority_id",
        "implementation_sha256", "execution_receipt_sha256",
        "exact_closure_verified", "semantic_replay_verified", "hostile_tests",
        "test_fixture", "dummy", "self_authored", "findings",
    }
    require(set(record) == required and record["schema"] == AUDIT_SCHEMA and
            record["status"] == "PASS_INDEPENDENT_EXECUTED_CAPABILITY_AUDIT" and
            record["executed"] is True and record["kind"] == manifest["kind"] and
            record["capability_id"] == manifest["capability_id"] and
            record["producer_authority_id"] == manifest["producer_authority_id"] and
            record["auditor_authority_id"] == manifest["auditor_authority_id"] and
            record["implementation_sha256"] == implementation_sha256 and
            record["execution_receipt_sha256"] == execution_sha256 and
            record["exact_closure_verified"] is True and
            record["semantic_replay_verified"] is True and
            isinstance(record["hostile_tests"], int) and
            not isinstance(record["hostile_tests"], bool) and
            record["hostile_tests"] >= (1 if allow_source_test_fixture else 12) and
            record["findings"] == [], "independent capability audit receipt")
    if allow_source_test_fixture:
        require(record["evidence_class"] == "SOURCE_TEST_FIXTURE" and
                record["test_fixture"] is True and record["dummy"] is True and
                record["self_authored"] is True,
                "honest source-test audit labelling")
    else:
        require(record["evidence_class"] == "PRODUCTION_EXECUTION" and
                record["test_fixture"] is False and record["dummy"] is False and
                record["self_authored"] is False,
                "production audit cannot be fixture, dummy, or self-authored")


def _finite_nonnegative(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value)) and float(value) >= 0.0)


def _validate_predecessor_details(details: Mapping[str, Any], *, fixture: bool) -> None:
    required = {
        "v2_manifest_sha256", "v2_source_root_sha256", "v2_exact_closure_verified",
        "v2_audit_manifest_sha256", "v2_audit_source_root_sha256",
        "v2_audit_exact_closure_verified", "producer_source_tests_executed",
        "independent_cpu_audit_executed", "independent_cupy_audit_executed",
    }
    require(set(details) == required and
            details["v2_manifest_sha256"] == V2_MANIFEST_SHA256 and
            details["v2_source_root_sha256"] == V2_SOURCE_ROOT_SHA256 and
            details["v2_audit_manifest_sha256"] == V2_AUDIT_MANIFEST_SHA256 and
            details["v2_audit_source_root_sha256"] == V2_AUDIT_SOURCE_ROOT_SHA256 and
            all(details[name] is True for name in required if name.endswith("_verified")) and
            details["producer_source_tests_executed"] is True and
            details["independent_cpu_audit_executed"] is True and
            details["independent_cupy_audit_executed"] is True,
            "executed predecessor source/audit capability")


def _validate_control_details(details: Mapping[str, Any], *, fixture: bool) -> None:
    required = {
        "generator_sha256", "integer_prng_spec_sha256", "backend_byte_identical",
        "complete_selection_replayed", "control_routes", "model_route_ids",
    }
    require(set(details) == required and is_sha256(details["generator_sha256"]) and
            is_sha256(details["integer_prng_spec_sha256"]) and
            details["backend_byte_identical"] is True and
            details["complete_selection_replayed"] is True and
            isinstance(details["model_route_ids"], list),
            "Gaussian-control execution details")
    rows = details["control_routes"]
    require(isinstance(rows, list) and len(rows) >= (1 if fixture else 8),
            "complete Gaussian-control routes")
    ids = []
    hashes = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {
            "route_id", "source_role_sha256", "source_identity_sha256",
            "selected_packet_sha256", "selected_reconstruction_sha256",
        } and isinstance(row["route_id"], str) and row["route_id"] and
                isinstance(row["source_role_sha256"], dict) and
                set(row["source_role_sha256"]) == {"gate", "up", "down_transposed"} and
                all(is_sha256(value) for value in row["source_role_sha256"].values()) and
                all(is_sha256(row[name]) for name in
                    ("source_identity_sha256", "selected_packet_sha256",
                     "selected_reconstruction_sha256")),
                "Gaussian-control route row")
        ids.append(row["route_id"])
        hashes.extend(row["source_role_sha256"].values())
    require(len(ids) == len(set(ids)) and len(hashes) == len(set(hashes)),
            "distinct Gaussian-control routes and source bytes")


def _validate_adapter_details(details: Mapping[str, Any], *, fixture: bool) -> None:
    required = {"adapter_abi", "literal_current_strata", "completed_planes",
                "decoded_index_min", "decoded_index_max", "routes"}
    require(set(details) == required and
            details["adapter_abi"] == "CURRENT_STRATA_SIX_PLANE_INDEX64_V1" and
            details["literal_current_strata"] is True and
            details["completed_planes"] == 6 and details["decoded_index_min"] == 0 and
            details["decoded_index_max"] == 63,
            "literal current-STRATA adapter semantics")
    rows = details["routes"]
    require(isinstance(rows, list) and rows, "STRATA adapter route rows")
    seen = set()
    for row in rows:
        required_row = {
            "route_id", "packet_sha256", "packet_bytes", "scale_payload_sha256",
            "scale_bytes", "scale_payload_inside_packet", "forward_transform_id",
            "forward_transform_sha256", "inverse_transform_sha256",
            "framing_header_bytes", "framing_payload_bytes", "framing_trailer_bytes",
            "framing_padding_bytes", "canonical_reencode_equal",
            "decoded_reconstruction_sha256", "decoded_weight_count",
        }
        require(isinstance(row, dict) and set(row) == required_row and
                isinstance(row["route_id"], str) and row["route_id"] not in seen and
                all(is_sha256(row[name]) for name in
                    ("packet_sha256", "scale_payload_sha256",
                     "forward_transform_sha256", "inverse_transform_sha256",
                     "decoded_reconstruction_sha256")) and
                isinstance(row["forward_transform_id"], str) and
                row["forward_transform_id"] and
                row["scale_payload_inside_packet"] is True and
                row["canonical_reencode_equal"] is True and
                all(isinstance(row[name], int) and not isinstance(row[name], bool) and
                    row[name] >= 0 for name in
                    ("packet_bytes", "scale_bytes", "framing_header_bytes",
                     "framing_payload_bytes", "framing_trailer_bytes",
                     "framing_padding_bytes", "decoded_weight_count")) and
                row["packet_bytes"] == row["framing_header_bytes"] +
                row["framing_payload_bytes"] + row["framing_trailer_bytes"] +
                row["framing_padding_bytes"] and row["scale_bytes"] > 0 and
                row["decoded_weight_count"] > 0,
                "literal scale/transform/framing adapter route")
        seen.add(row["route_id"])


def _validate_scorer_details(details: Mapping[str, Any], *, fixture: bool) -> None:
    required = {"source_dtype", "accumulation", "independent_from_adapter",
                "routes", "pooled"}
    require(set(details) == required and details["source_dtype"] == "BF16_LE" and
            details["accumulation"] == "FP64" and
            details["independent_from_adapter"] is True,
            "independent BF16 scorer semantics")
    rows = details["routes"]
    require(isinstance(rows, list) and rows, "BF16 scorer route rows")
    total_sse = total_energy = 0.0
    total_count = total_bits = 0
    seen = set()
    for row in rows:
        required_row = {
            "route_id", "source_role_sha256", "decoded_reconstruction_sha256",
            "weight_count", "physical_bits", "sse_fp64", "source_energy_fp64",
            "relative_mse", "physical_rate_bpw", "f_value",
        }
        require(isinstance(row, dict) and set(row) == required_row and
                isinstance(row["route_id"], str) and row["route_id"] not in seen and
                isinstance(row["source_role_sha256"], dict) and
                set(row["source_role_sha256"]) == {"gate", "up", "down_transposed"} and
                all(is_sha256(value) for value in row["source_role_sha256"].values()) and
                is_sha256(row["decoded_reconstruction_sha256"]) and
                isinstance(row["weight_count"], int) and row["weight_count"] > 0 and
                isinstance(row["physical_bits"], int) and row["physical_bits"] > 0 and
                _finite_nonnegative(row["sse_fp64"]) and
                _finite_nonnegative(row["source_energy_fp64"]) and
                float(row["source_energy_fp64"]) > 0.0,
                "BF16 FP64 scorer route")
        n = row["weight_count"]
        bits = row["physical_bits"]
        relative = float(row["sse_fp64"]) / float(row["source_energy_fp64"])
        rate = bits / n
        f_value = relative * 2.0 ** (2.0 * rate)
        require(math.isclose(float(row["relative_mse"]), relative,
                             rel_tol=2e-15, abs_tol=0.0) and
                math.isclose(float(row["physical_rate_bpw"]), rate,
                             rel_tol=2e-15, abs_tol=0.0) and
                math.isclose(float(row["f_value"]), f_value,
                             rel_tol=2e-15, abs_tol=0.0),
                "BF16 scorer route recomputation")
        seen.add(row["route_id"])
        total_sse += float(row["sse_fp64"])
        total_energy += float(row["source_energy_fp64"])
        total_count += n
        total_bits += bits
    pooled = details["pooled"]
    require(isinstance(pooled, dict) and set(pooled) == {
        "sse_fp64", "source_energy_fp64", "weight_count", "physical_bits",
        "relative_mse", "physical_rate_bpw", "f_value"},
        "pooled BF16 scorer schema")
    relative = total_sse / total_energy
    rate = total_bits / total_count
    f_value = relative * 2.0 ** (2.0 * rate)
    require(math.isclose(float(pooled["sse_fp64"]), total_sse, rel_tol=2e-15) and
            math.isclose(float(pooled["source_energy_fp64"]), total_energy,
                         rel_tol=2e-15) and pooled["weight_count"] == total_count and
            pooled["physical_bits"] == total_bits and
            math.isclose(float(pooled["relative_mse"]), relative, rel_tol=2e-15) and
            math.isclose(float(pooled["physical_rate_bpw"]), rate, rel_tol=2e-15) and
            math.isclose(float(pooled["f_value"]), f_value, rel_tol=2e-15),
            "pooled FP64 score recomputation")


def _validate_read_details(details: Mapping[str, Any], *, fixture: bool) -> None:
    required = {"page_bytes", "instrumented_reads", "layout_only", "routes"}
    require(set(details) == required and details["page_bytes"] == PAGE_BYTES and
            details["instrumented_reads"] is True and
            details["layout_only"] is False,
            "instrumented read trace, not layout assertion")
    rows = details["routes"]
    require(isinstance(rows, list) and rows, "per-routed-expert read rows")
    seen = set()
    for row in rows:
        required_row = {
            "route_id", "layer", "expert", "packet_sha256", "literal_packet_bytes",
            "events", "unique_page_indices", "physical_page_bytes_read",
            "cold_read_amplification", "one_routed_expert_only",
        }
        require(isinstance(row, dict) and set(row) == required_row and
                isinstance(row["route_id"], str) and row["route_id"] not in seen and
                isinstance(row["layer"], str) and row["layer"] and
                isinstance(row["expert"], str) and row["expert"] and
                is_sha256(row["packet_sha256"]) and
                isinstance(row["literal_packet_bytes"], int) and
                row["literal_packet_bytes"] > 0 and
                row["one_routed_expert_only"] is True,
                "one per-routed-expert trace")
        events = row["events"]
        require(isinstance(events, list) and events, "nonempty page-read events")
        pages = []
        physical = 0
        for event in events:
            require(isinstance(event, dict) and set(event) == {
                "sequence", "page_index", "file_offset", "bytes_read", "page_sha256"
            } and isinstance(event["sequence"], int) and
                    event["sequence"] == len(pages) and
                    isinstance(event["page_index"], int) and event["page_index"] >= 0 and
                    event["file_offset"] == event["page_index"] * PAGE_BYTES and
                    event["bytes_read"] == PAGE_BYTES and
                    is_sha256(event["page_sha256"]), "instrumented page-read event")
            pages.append(event["page_index"])
            physical += event["bytes_read"]
        unique = sorted(set(pages))
        amplification = physical / row["literal_packet_bytes"]
        require(row["unique_page_indices"] == unique and
                row["physical_page_bytes_read"] == physical and
                math.isclose(float(row["cold_read_amplification"]), amplification,
                             rel_tol=2e-15) and amplification < 2.0,
                "per-routed-expert physical page-read amplification")
        seen.add(row["route_id"])


def _validate_launch_audit_details(details: Mapping[str, Any], *, fixture: bool) -> None:
    required = {"other_capability_pin_set_sha256", "launch_schema_verified",
                "route_closure_verified", "model_control_alias_checks_replayed",
                "strata_adapter_replayed", "bf16_scorer_replayed",
                "per_expert_read_trace_replayed"}
    require(set(details) == required and
            is_sha256(details["other_capability_pin_set_sha256"]) and
            all(details[name] is True for name in required
                if name != "other_capability_pin_set_sha256"),
            "independent launch audit replay")


DETAIL_VALIDATORS = {
    "predecessor_source_audit": _validate_predecessor_details,
    "gaussian_control_generator": _validate_control_details,
    "current_strata_adapter": _validate_adapter_details,
    "independent_bf16_scorer": _validate_scorer_details,
    "routed_page_reader": _validate_read_details,
    "independent_launch_audit": _validate_launch_audit_details,
}


def authenticate_capability(evidence_root: Path, pin: Mapping[str, Any], *,
                            allow_source_test_fixture: bool = False) -> dict[str, Any]:
    """Authenticate one independently pinned, executed capability closure."""
    required_pin = {"kind", "relative_path", "manifest_sha256", "source_root_sha256",
                    "execution_receipt_sha256", "audit_receipt_sha256"}
    require(isinstance(pin, Mapping) and set(pin) == required_pin and
            pin["kind"] in CAPABILITY_KINDS and
            all(is_sha256(pin[name]) for name in required_pin if name.endswith("sha256")),
            "capability external pin")
    root = resolve_member(real_directory(evidence_root, "evidence root"),
                          pin["relative_path"], "capability", directory=True)
    manifest_payload, _ = regular_bytes(root / "CAPABILITY_MANIFEST.json",
                                        "capability manifest")
    require(sha256(manifest_payload) == pin["manifest_sha256"],
            "capability independently pinned manifest")
    manifest = strict_json(manifest_payload, "capability manifest")
    required_manifest = {
        "schema", "status", "kind", "capability_id", "evidence_class",
        "producer_authority_id", "executor_authority_id", "auditor_authority_id",
        "source_root_sha256", "members", "implementation_name",
        "execution_receipt_name", "audit_receipt_name",
    }
    require(set(manifest) == required_manifest and manifest["schema"] == CAPABILITY_SCHEMA and
            manifest["status"] == "SEALED_EXECUTED_CAPABILITY" and
            manifest["kind"] == pin["kind"] and
            manifest["source_root_sha256"] == pin["source_root_sha256"],
            "capability manifest schema and pins")
    producer = _validate_identity(manifest["producer_authority_id"], "producer")
    executor = _validate_identity(manifest["executor_authority_id"], "executor")
    auditor = _validate_identity(manifest["auditor_authority_id"], "auditor")
    require(len({producer, executor, auditor}) == 3,
            "producer, executor, and auditor must be independent")
    if allow_source_test_fixture:
        require(manifest["evidence_class"] == "SOURCE_TEST_FIXTURE",
                "source-test capability class")
    else:
        require(manifest["evidence_class"] == "PRODUCTION_EXECUTION" and
                PACKAGE_AUTHORITY_ID not in {producer, executor, auditor},
                "capability cannot be authored, executed, or audited by this package")
    rows = manifest["members"]
    require(isinstance(rows, list) and rows, "capability members")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "capability member row")
        name = row["name"]
        require(isinstance(name, str) and SAFE_NAME.fullmatch(name) is not None and
                name != "CAPABILITY_MANIFEST.json" and name not in names,
                "capability flat member name")
        payload, _ = regular_bytes(root / name, f"capability member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"capability member pin {name}")
        names.append(name)
        observed.append(item)
    require(names == sorted(names, key=lambda value: value.encode("utf-8")) and
            _member_root(observed) == pin["source_root_sha256"],
            "capability canonical source root")
    require(set(names) >= {manifest["implementation_name"],
                           manifest["execution_receipt_name"],
                           manifest["audit_receipt_name"]},
            "capability required members")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {"CAPABILITY_MANIFEST.json"} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "capability exact regular closure")
    implementation, _ = regular_bytes(root / manifest["implementation_name"],
                                      "capability implementation")
    execution_payload, _ = regular_bytes(root / manifest["execution_receipt_name"],
                                         "capability execution receipt")
    audit_payload, _ = regular_bytes(root / manifest["audit_receipt_name"],
                                     "capability independent audit receipt")
    implementation_sha = sha256(implementation)
    execution_sha = sha256(execution_payload)
    audit_sha = sha256(audit_payload)
    require(execution_sha == pin["execution_receipt_sha256"] and
            audit_sha == pin["audit_receipt_sha256"],
            "capability separately pinned receipt hashes")
    execution = strict_json(execution_payload, "capability execution receipt")
    audit = strict_json(audit_payload, "capability audit receipt")
    _validate_common_execution(execution, manifest, implementation_sha,
                               allow_source_test_fixture=allow_source_test_fixture)
    _validate_common_audit(audit, manifest, implementation_sha, execution_sha,
                           allow_source_test_fixture=allow_source_test_fixture)
    DETAIL_VALIDATORS[manifest["kind"]](
        execution["details"], fixture=allow_source_test_fixture)
    return {"kind": manifest["kind"], "capability_id": manifest["capability_id"],
            "manifest_sha256": pin["manifest_sha256"],
            "source_root_sha256": pin["source_root_sha256"],
            "execution_receipt_sha256": execution_sha,
            "audit_receipt_sha256": audit_sha,
            "producer_authority_id": producer,
            "executor_authority_id": executor,
            "auditor_authority_id": auditor,
            "details": execution["details"]}


def _source_records(evidence_root: Path, routes: list[Mapping[str, Any]]) -> tuple[dict, dict]:
    by_id = {}
    identities: dict[tuple[int, int], str] = {}
    digests: dict[str, str] = {}
    paths: dict[str, str] = {}
    for route in routes:
        required = {"route_id", "kind", "architecture_family", "layer", "expert",
                    "sources", "packet", "required_control_route_ids"}
        require(isinstance(route, Mapping) and set(route) == required and
                isinstance(route["route_id"], str) and route["route_id"] and
                route["route_id"] not in by_id and route["kind"] in
                {"model_bf16", "matched_gaussian_bf16"} and
                isinstance(route["architecture_family"], str) and
                route["architecture_family"] and isinstance(route["layer"], str) and
                route["layer"] and isinstance(route["expert"], str) and
                route["expert"], "launch route schema")
        sources = route["sources"]
        require(isinstance(sources, list) and len(sources) == 3,
                "one Gate/Up/Down source triplet")
        role_hashes = {}
        for source in sources:
            require(isinstance(source, Mapping) and set(source) == {
                "role", "relative_path", "bytes", "sha256"
            } and source["role"] in {"gate", "up", "down_transposed"} and
                    source["role"] not in role_hashes and
                    isinstance(source["bytes"], int) and source["bytes"] > 0 and
                    is_sha256(source["sha256"]), "route BF16 source row")
            path = resolve_member(evidence_root, source["relative_path"], "BF16 source")
            payload, identity = regular_bytes(path, "BF16 source")
            require(len(payload) == source["bytes"] and sha256(payload) == source["sha256"] and
                    len(payload) % 2 == 0, "literal BF16 source bytes")
            canonical_path = str(path)
            owner = route["route_id"] + ":" + source["role"]
            require(identity not in identities, "source inode alias across routes/roles")
            require(source["sha256"] not in digests,
                    "source-byte alias across model/control routes")
            require(canonical_path not in paths, "source-path alias across routes/roles")
            identities[identity] = owner
            digests[source["sha256"]] = owner
            paths[canonical_path] = owner
            role_hashes[source["role"]] = source["sha256"]
        require(set(role_hashes) == {"gate", "up", "down_transposed"},
                "complete role sources")
        packet = route["packet"]
        require(isinstance(packet, Mapping) and set(packet) == {
            "relative_path", "bytes", "sha256"
        } and isinstance(packet["bytes"], int) and packet["bytes"] > 0 and
                is_sha256(packet["sha256"]), "route packet row")
        packet_path = resolve_member(evidence_root, packet["relative_path"], "STRATA packet")
        packet_payload, _ = regular_bytes(packet_path, "STRATA packet")
        require(len(packet_payload) == packet["bytes"] and
                sha256(packet_payload) == packet["sha256"], "literal STRATA packet bytes")
        by_id[route["route_id"]] = {"record": route, "source_role_sha256": role_hashes,
                                    "packet_sha256": packet["sha256"],
                                    "packet_bytes": packet["bytes"],
                                    "packet_payload": packet_payload}
    return by_id, {"source_identities": identities, "source_digests": digests,
                   "source_paths": paths}


def _bind_capabilities_to_routes(capabilities: dict[str, dict[str, Any]],
                                 routes: dict[str, dict[str, Any]]) -> None:
    route_ids = set(routes)
    adapter_rows = {row["route_id"]: row for row in
                    capabilities["current_strata_adapter"]["details"]["routes"]}
    scorer_rows = {row["route_id"]: row for row in
                   capabilities["independent_bf16_scorer"]["details"]["routes"]}
    read_rows = {row["route_id"]: row for row in
                 capabilities["routed_page_reader"]["details"]["routes"]}
    require(set(adapter_rows) == route_ids and set(scorer_rows) == route_ids and
            set(read_rows) == route_ids, "adapter/scorer/read exact route closure")
    require(capabilities["current_strata_adapter"]["executor_authority_id"] !=
            capabilities["independent_bf16_scorer"]["executor_authority_id"] and
            capabilities["current_strata_adapter"]["producer_authority_id"] !=
            capabilities["independent_bf16_scorer"]["producer_authority_id"],
            "BF16 scorer independent of STRATA adapter")
    for route_id, route in routes.items():
        adapter = adapter_rows[route_id]
        scorer = scorer_rows[route_id]
        read = read_rows[route_id]
        require(adapter["packet_sha256"] == route["packet_sha256"] and
                adapter["packet_bytes"] == route["packet_bytes"] and
                scorer["source_role_sha256"] == route["source_role_sha256"] and
                scorer["decoded_reconstruction_sha256"] ==
                adapter["decoded_reconstruction_sha256"] and
                scorer["physical_bits"] == route["packet_bytes"] * 8 and
                read["packet_sha256"] == route["packet_sha256"] and
                read["literal_packet_bytes"] == route["packet_bytes"] and
                read["layer"] == route["record"]["layer"] and
                read["expert"] == route["record"]["expert"],
                f"route evidence binding {route_id}")
        packet = route["packet_payload"]
        page_count = (len(packet) + PAGE_BYTES - 1) // PAGE_BYTES
        for event in read["events"]:
            page_index = event["page_index"]
            require(page_index < page_count, f"read page in packet {route_id}")
            page = packet[page_index * PAGE_BYTES:(page_index + 1) * PAGE_BYTES]
            page += bytes(PAGE_BYTES - len(page))
            require(event["page_sha256"] == sha256(page),
                    f"read page binds literal packet {route_id}")


def verify_precommitted_evidence(evidence_root: Path, launch_manifest_path: Path,
                                 expected_launch_manifest_sha256: str, *,
                                 allow_source_test_fixture: bool = False) -> dict[str, Any]:
    """Verify a previously pinned evidence set.

    This lower-level routine exists for an independent controller and for
    source-only hostile tests.  Passing a just-created digest here is not a
    production authorization; only :func:`authorize_production` can issue one.
    """
    require(is_sha256(expected_launch_manifest_sha256),
            "precommitted launch-manifest SHA-256")
    root = real_directory(evidence_root, "evidence root")
    try:
        launch = launch_manifest_path.resolve(strict=True)
        launch.relative_to(root)
    except (OSError, ValueError) as exc:
        raise AuthorityError("launch manifest confined to evidence root") from exc
    payload, _ = regular_bytes(launch, "launch manifest")
    require(sha256(payload) == expected_launch_manifest_sha256,
            "launch manifest precommit")
    record = strict_json(payload, "launch manifest")
    required = {"schema", "status", "evidence_class", "issuer_authority_id",
                "v3_source_manifest_sha256", "v3_source_root_sha256",
                "predecessor_pins", "capability_pins", "routes"}
    require(set(record) == required and record["schema"] == LAUNCH_SCHEMA and
            record["status"] == "SEALED_BEFORE_PAYLOAD_LAUNCH" and
            is_sha256(record["v3_source_manifest_sha256"]) and
            is_sha256(record["v3_source_root_sha256"]) and
            record["predecessor_pins"] == {
                "v2_manifest_sha256": V2_MANIFEST_SHA256,
                "v2_source_root_sha256": V2_SOURCE_ROOT_SHA256,
                "v2_audit_manifest_sha256": V2_AUDIT_MANIFEST_SHA256,
                "v2_audit_source_root_sha256": V2_AUDIT_SOURCE_ROOT_SHA256,
            }, "launch manifest schema and predecessor pins")
    issuer = _validate_identity(record["issuer_authority_id"], "launch issuer")
    if allow_source_test_fixture:
        require(record["evidence_class"] == "SOURCE_TEST_FIXTURE",
                "honest launch fixture class")
    else:
        require(record["evidence_class"] == "PRODUCTION_EXECUTION" and
                issuer != PACKAGE_AUTHORITY_ID,
                "launch issuer independent of package")
    pins = record["capability_pins"]
    require(isinstance(pins, list) and len(pins) == len(CAPABILITY_KINDS),
            "complete capability pin set")
    capabilities = {}
    relative_paths = set()
    for pin in pins:
        require(pin["kind"] not in capabilities and
                pin["relative_path"] not in relative_paths,
                "distinct capability kind/path")
        capabilities[pin["kind"]] = authenticate_capability(
            root, pin, allow_source_test_fixture=allow_source_test_fixture)
        relative_paths.add(pin["relative_path"])
    require(set(capabilities) == set(CAPABILITY_KINDS),
            "exact capability-kind closure")
    routes_raw = record["routes"]
    require(isinstance(routes_raw, list) and routes_raw, "nonempty routed cases")
    routes, alias_receipt = _source_records(root, routes_raw)
    model_ids = {route_id for route_id, value in routes.items()
                 if value["record"]["kind"] == "model_bf16"}
    control_ids = set(routes) - model_ids
    require(model_ids and len(control_ids) >= (1 if allow_source_test_fixture else 8),
            "model plus matched Gaussian routes")
    for route_id in model_ids:
        linked = routes[route_id]["record"]["required_control_route_ids"]
        require(isinstance(linked, list) and len(linked) >=
                (1 if allow_source_test_fixture else 8) and
                len(linked) == len(set(linked)) and set(linked) <= control_ids,
                "model exact matched-control links")
    for route_id in control_ids:
        require(routes[route_id]["record"]["required_control_route_ids"] == [],
                "control route has no child controls")
    controls = capabilities["gaussian_control_generator"]["details"]
    control_rows = {row["route_id"]: row for row in controls["control_routes"]}
    require(set(control_rows) == control_ids and
            set(controls["model_route_ids"]) == model_ids,
            "control capability exact route closure")
    for route_id in control_ids:
        require(control_rows[route_id]["source_role_sha256"] ==
                routes[route_id]["source_role_sha256"] and
                control_rows[route_id]["selected_packet_sha256"] ==
                routes[route_id]["packet_sha256"], "control literal source/packet binding")
    _bind_capabilities_to_routes(capabilities, routes)
    score_rows = {row["route_id"]: row for row in
                  capabilities["independent_bf16_scorer"]["details"]["routes"]}
    model_sse = sum(float(score_rows[route_id]["sse_fp64"])
                    for route_id in model_ids)
    model_energy = sum(float(score_rows[route_id]["source_energy_fp64"])
                       for route_id in model_ids)
    model_bits = sum(score_rows[route_id]["physical_bits"] for route_id in model_ids)
    model_weights = sum(score_rows[route_id]["weight_count"] for route_id in model_ids)
    model_rate = model_bits / model_weights
    model_relative_mse = model_sse / model_energy
    model_f = model_relative_mse * 2.0 ** (2.0 * model_rate)
    require(RATE_MIN <= model_rate <= RATE_MAX and model_f <= TARGET_F,
            "pooled model physical rate/F target")
    other_pins = [pin for pin in pins if pin["kind"] != "independent_launch_audit"]
    other_pins.sort(key=lambda item: item["kind"].encode("utf-8"))
    require(capabilities["independent_launch_audit"]["details"]
            ["other_capability_pin_set_sha256"] == sha256(canonical_json(other_pins)),
            "independent launch audit binds capability set")
    return {
        "verified": True, "production_authorized": not allow_source_test_fixture,
        "launch_manifest_sha256": expected_launch_manifest_sha256,
        "capability_manifest_sha256": {
            kind: value["manifest_sha256"] for kind, value in capabilities.items()},
        "route_ids": sorted(routes), "model_route_ids": sorted(model_ids),
        "control_route_ids": sorted(control_ids),
        "model_control_path_inode_and_byte_aliases_rejected": True,
        "instrumented_read_trace_bound": True,
        "literal_current_strata_adapter_bound": True,
        "independent_bf16_fp64_scorer_bound": True,
        "pooled_model_physical_rate_bpw": model_rate,
        "pooled_model_relative_mse": model_relative_mse,
        "pooled_model_f_value": model_f,
        "alias_receipt_counts": {key: len(value) for key, value in alias_receipt.items()},
    }


def authorize_production(evidence_root: Path, launch_manifest_path: Path) -> dict[str, Any]:
    """Production entry with a compiled trust root, deliberately held in v3."""
    require(TRUSTED_LAUNCH_MANIFEST_SHA256 is not None and
            is_sha256(TRUSTED_LAUNCH_MANIFEST_SHA256),
            "HOLD: no independently frozen production launch-manifest pin")
    result = verify_precommitted_evidence(
        evidence_root, launch_manifest_path, TRUSTED_LAUNCH_MANIFEST_SHA256,
        allow_source_test_fixture=False)
    result["production_authorized"] = True
    return result
