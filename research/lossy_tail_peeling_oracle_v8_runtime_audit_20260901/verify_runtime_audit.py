#!/usr/bin/env python3
"""Independent, source-free verifier for the sealed lossy-tail-v8 runtime audit.

This program intentionally imports only the Python standard library.  It does
not import CuPy, initialize CUDA, inspect a GPU, or open any model/Qwen/source
payload or production result.  ``--evidence-only`` verifies the frozen source
and source-audit bindings plus the copied runtime receipt.  The default mode
also verifies this audit package's manifest and PASS receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "lossy_tail_peeling_oracle_v8"
SOURCE_AUDIT = HERE.parent / "lossy_tail_peeling_oracle_v8_independent_source_audit_20260901"
RUNTIME_RECEIPT = HERE / "runtime_receipt.json"

RUNTIME_FILE_SHA256 = "45862549f34530964c4f8f7a4134228ccf036a8de3534f23e63e07acde7985b3"
RUNTIME_FILE_BYTES = 82235
RUNTIME_INTERNAL_SHA256 = "de096442f734689ed981c0b3870a335255c9914d9d818b56aca1ac956add4d50"
PROBE_AGGREGATE_SHA256 = "85f049ad8372e13df9c8643eb7b4727f167ec2fb4e4ef8aa2188e77b44ad30d1"

LAUNCH_SHA256 = "6c5f5cd05973dbc0bf16cd9ea39951e690b15e15e13e969d2a33823117c2aa94"
RUNTIME_CONTRACT_SHA256 = "c83f065122a05f05e5b4788ffc717b517e8c0310cb771dfc1324fc1115745f0a"
CALIBRATOR_SHA256 = "f454f69ee802a52b2797576fd426558a456d4398052aeb2cbeb8e7d941140a47"
CORE_SHA256 = "be51a7f895af2cf1f2863b491affa4e9dbdcc2eb59798e239560bd1f85172c66"
AUTHORIZATION_CONTRACT_SHA256 = "f97c11a5254deec2c140bb894c32d5f455af26ae8344b995569deb87c516f6eb"
SOURCE_AUDIT_MANIFEST_FILE_SHA256 = "045eac28701decf60837be335cfdf316b3ab1650125bc2f3744f097c0e75bb87"
SOURCE_AUDIT_RECEIPT_FILE_SHA256 = "6e2bc929904c7c05274a3c2bfc634707336d8646f85db2776ef024032988dde0"
SOURCE_AUDIT_MANIFEST_INTERNAL_SHA256 = "afe6c6f43a0939861973e4362a5ffa8389f0ac77941dfe8b8e9a5b98a6e1d483"
SOURCE_AUDIT_RECEIPT_INTERNAL_SHA256 = "b06d584ec0e5c8fd69d8f6b609f72631b69dc1feee0af467018e76607384f38b"
REPAIR_INTERNAL_SHA256 = "270746d99395b9c713500dad8b9c41d8c93aa58ee0d0e04adb815da44870fd32"

REMOTE_RECEIPT_PATH = "/var/tmp/int2_lossy_tail_v8_runtime_evidence_6c5f5cd0/calibration_v1/runtime_receipt.json"
STAGE_ROOT = "/workspace/INT2__compression/source_only_audits/lossy_v8_20260901T143028Z_n1g/isolated_stage"
EXPECTED_OUTPUT_PATH = REMOTE_RECEIPT_PATH
EXPECTED_CWD = "/workspace/INT2__compression"

HEX64 = re.compile(r"^[0-9a-f]{64}$")
FLOAT_HEX = re.compile(r"^-?0x[01]\.[0-9a-f]{13}p[+-][0-9]+$")


class AuditFailure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.evidence = 0
        self.package = 0
        self._mode = "evidence"

    def package_mode(self) -> None:
        self._mode = "package"

    def require(self, condition: bool, message: str) -> None:
        if self._mode == "evidence":
            self.evidence += 1
        else:
            self.package += 1
        if not condition:
            raise AuditFailure(message)


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise AuditFailure(f"duplicate JSON key: {key!r}")
        value[key] = child
    return value


def reject_constant(value: str) -> None:
    raise AuditFailure(f"non-finite JSON constant: {value}")


def read_regular(path: Path) -> bytes:
    if path.is_symlink():
        raise AuditFailure(f"symlink rejected: {path}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise AuditFailure(f"cannot stat {path}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise AuditFailure(f"not a regular file: {path}")
    payload = path.read_bytes()
    if len(payload) != metadata.st_size:
        raise AuditFailure(f"file changed while reading: {path}")
    return payload


def strict_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = read_regular(path)
    try:
        text = payload.decode("utf-8")
        value = json.loads(text, object_pairs_hook=strict_pairs, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"invalid UTF-8 JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"top level is not an object: {path}")
    return payload, value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def exact_keys(checks: Checks, value: Any, expected: Iterable[str], label: str) -> None:
    checks.require(isinstance(value, dict), f"{label} is not an object")
    checks.require(set(value) == set(expected), f"{label} key drift: {set(value) ^ set(expected)}")


def valid_sha(checks: Checks, value: Any, label: str) -> None:
    checks.require(isinstance(value, str) and HEX64.fullmatch(value) is not None, f"invalid SHA-256: {label}")


def sealed(checks: Checks, value: dict[str, Any], field: str, expected: str, label: str) -> None:
    valid_sha(checks, value.get(field), f"{label}.{field}")
    checks.require(value[field] == expected, f"{label} internal seal drift")
    clean = dict(value)
    clean.pop(field)
    checks.require(sha256(canonical_bytes(clean)) == expected, f"{label} internal seal recomputation failed")


def parse_finite_float_hex(checks: Checks, value: Any, label: str) -> float:
    checks.require(isinstance(value, str) and FLOAT_HEX.fullmatch(value) is not None, f"invalid float.hex format: {label}")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise AuditFailure(f"unparseable float.hex at {label}") from exc
    checks.require(math.isfinite(parsed), f"nonfinite float at {label}")
    checks.require(parsed.hex() == value, f"noncanonical float.hex at {label}")
    return parsed


def walk_hash_fields(checks: Checks, value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("sha256"):
                valid_sha(checks, child, f"{label}.{key}")
            walk_hash_fields(checks, child, f"{label}.{key}")
    elif isinstance(value, list):
        for ordinal, child in enumerate(value):
            walk_hash_fields(checks, child, f"{label}[{ordinal}]")


def stable_seed(replica: int, ordinal: int) -> int:
    blob = f"lossy-tail-peeling-gaussian-v1\0{replica}\0{ordinal}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


def stable_fixture_rows() -> list[dict[str, Any]]:
    fixtures: list[tuple[str, list[int | float], str]] = [
        ("float32", [3, 3, 2, 3, 2, 3], "f"),
        ("float32", [0.0, -0.0, 0.0, -0.0, 1.0, 1.0], "f"),
        ("int64", [7] * 257, "q"),
        ("float64", [2, 1, 2, 1] * 257, "d"),
        ("float32", [5] * 513 + [4] * 257 + [3] * 129 + [2] * 65 + [1] * 33, "f"),
    ]
    rows: list[dict[str, Any]] = []
    for ordinal, (dtype, values, code) in enumerate(fixtures):
        input_bytes = b"".join(struct.pack("<" + code, value) for value in values)
        order = sorted(range(len(values)), key=lambda index: (-values[index], index))
        order_bytes = b"".join(struct.pack("<q", index) for index in order)
        rows.append({
            "ordinal": ordinal,
            "dtype": dtype,
            "count": len(values),
            "input_sha256": sha256(input_bytes),
            "order_sha256": sha256(order_bytes),
        })
    return rows


def verify_memory(checks: Checks, value: Any, label: str) -> None:
    exact_keys(checks, value, {
        "stream_synchronized", "used_bytes_before_free", "total_bytes_before_free",
        "used_bytes_after_free", "total_bytes_after_free",
        "all_per_cell_gpu_arrays_deleted_before_free",
    }, label)
    checks.require(value["stream_synchronized"] is True, f"{label} was not synchronized")
    checks.require(type(value["used_bytes_before_free"]) is int and value["used_bytes_before_free"] == 0, f"{label} live bytes before free")
    checks.require(type(value["total_bytes_before_free"]) is int and value["total_bytes_before_free"] > 0, f"{label} lacks cached-byte evidence")
    checks.require(type(value["used_bytes_after_free"]) is int and value["used_bytes_after_free"] == 0, f"{label} live bytes after free")
    checks.require(type(value["total_bytes_after_free"]) is int and value["total_bytes_after_free"] == 0, f"{label} cached bytes after free")
    checks.require(value["all_per_cell_gpu_arrays_deleted_before_free"] is True, f"{label} deletion flag false")


def verify_source_bindings(checks: Checks) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    launch_payload, launch = strict_json(TARGET / "launch_manifest.json")
    contract_payload, contract = strict_json(TARGET / "runtime_contract.json")
    auth_payload, authorization = strict_json(TARGET / "authorization_contract.json")
    calibrator_payload = read_regular(TARGET / "runtime_calibrate.py")
    core_payload = read_regular(TARGET / "lossy_tail_core.py")
    checks.require(sha256(launch_payload) == LAUNCH_SHA256, "frozen launch manifest changed")
    checks.require(sha256(contract_payload) == RUNTIME_CONTRACT_SHA256, "frozen runtime contract changed")
    checks.require(sha256(auth_payload) == AUTHORIZATION_CONTRACT_SHA256, "frozen authorization contract changed")
    checks.require(sha256(calibrator_payload) == CALIBRATOR_SHA256, "frozen calibrator changed")
    checks.require(sha256(core_payload) == CORE_SHA256, "frozen scientific core changed")
    checks.require(launch.get("schema") == "lossy-tail-v8-launch-manifest-v1", "launch schema drift")
    checks.require(launch.get("status") == "FROZEN_V8_SOURCE_STAGE_NO_RUNTIME_OR_PRODUCTION_AUTHORIZATION", "launch status drift")
    checks.require(contract.get("schema") == "lossy-tail-v8-runtime-calibration-contract-v1", "runtime-contract schema drift")
    checks.require(contract.get("status") == "FROZEN_SOURCE_FREE_BEFORE_RUNTIME_CALIBRATION", "runtime-contract status drift")
    checks.require(authorization.get("schema") == "lossy-tail-v8-production-authorization-contract-v1", "authorization-contract schema drift")
    checks.require(authorization.get("status") == "FROZEN_TEMPLATE_ONLY_NO_AUTHORIZATION_EXISTS", "authorization-contract status drift")
    required = authorization["required_values"]
    checks.require(required.get("runtime_receipt_schema") == "lossy-tail-v8-source-free-runtime-receipt-v1", "required runtime receipt schema drift")
    checks.require(required.get("runtime_receipt_status") == "UNTRUSTED_UNTIL_INDEPENDENT_RUNTIME_AUDIT", "required runtime receipt status drift")
    checks.require(required.get("runtime_audit_manifest_schema") == "lossy-tail-v8-independent-runtime-audit-manifest-v1", "required runtime audit manifest schema drift")
    checks.require(required.get("runtime_audit_manifest_status") == "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET", "required runtime audit manifest status drift")
    checks.require(required.get("runtime_audit_receipt_schema") == "lossy-tail-v8-independent-runtime-audit-receipt-v1", "required runtime audit receipt schema drift")
    checks.require(required.get("runtime_audit_receipt_status") == "PASS_V8_INDEPENDENT_RUNTIME_AUDIT", "required runtime audit receipt status drift")

    source_manifest_payload, source_manifest = strict_json(SOURCE_AUDIT / "audit_manifest.json")
    source_receipt_payload, source_receipt = strict_json(SOURCE_AUDIT / "audit_receipt.json")
    checks.require(sha256(source_manifest_payload) == SOURCE_AUDIT_MANIFEST_FILE_SHA256, "source-audit manifest file hash drift")
    checks.require(sha256(source_receipt_payload) == SOURCE_AUDIT_RECEIPT_FILE_SHA256, "source-audit receipt file hash drift")
    checks.require(source_manifest.get("schema") == required["source_audit_manifest_schema"], "source-audit manifest schema drift")
    checks.require(source_manifest.get("status") == required["source_audit_manifest_status"], "source-audit manifest status drift")
    checks.require(source_receipt.get("schema") == required["source_audit_receipt_schema"], "source-audit receipt schema drift")
    checks.require(source_receipt.get("status") == required["source_audit_receipt_status"], "source-audit receipt status drift")
    sealed(checks, source_manifest, "audit_manifest_sha256", SOURCE_AUDIT_MANIFEST_INTERNAL_SHA256, "source-audit manifest")
    sealed(checks, source_receipt, "audit_receipt_sha256", SOURCE_AUDIT_RECEIPT_INTERNAL_SHA256, "source-audit receipt")
    checks.require(source_receipt["audited_target"]["launch_manifest_sha256"] == LAUNCH_SHA256, "source audit targets another launch manifest")
    checks.require(source_receipt["audited_target"]["repair_lock_internal_sha256"] == REPAIR_INTERNAL_SHA256, "source audit targets another repair lock")
    checks.require(source_receipt["verdict"]["source_free_runtime_calibration_warranted"] is True, "source audit did not warrant calibration")
    checks.require(source_receipt["verdict"]["payload_access_authorized"] is False, "source audit claims payload authority")
    checks.require(source_receipt["verdict"]["production_authorized"] is False, "source audit claims production authority")
    source_access = source_receipt["access_ledger"]
    for key in (
        "model_or_qwen_paths_traversed", "model_payload_files_opened",
        "validation_data_files_opened", "production_result_files_opened",
        "cupy_imports", "cuda_initializations", "gpu_jobs", "production_runs",
    ):
        checks.require(source_access.get(key) == 0, f"source-audit zero-access drift: {key}")
    return contract, authorization, source_receipt


def verify_runtime_evidence(checks: Checks) -> dict[str, Any]:
    contract, authorization, source_receipt = verify_source_bindings(checks)
    payload, receipt = strict_json(RUNTIME_RECEIPT)
    checks.require(len(payload) == RUNTIME_FILE_BYTES, "runtime receipt byte size drift")
    checks.require(sha256(payload) == RUNTIME_FILE_SHA256, "runtime receipt file SHA-256 drift")
    exact_keys(checks, receipt, {
        "schema", "status", "launch_manifest", "runtime_contract", "calibrator",
        "scientific_core", "invocation", "runtime_probe", "access_ledger",
        "authorization", "runtime_receipt_sha256",
    }, "runtime receipt")
    checks.require(receipt["schema"] == "lossy-tail-v8-source-free-runtime-receipt-v1", "runtime receipt schema drift")
    checks.require(receipt["status"] == "UNTRUSTED_UNTIL_INDEPENDENT_RUNTIME_AUDIT", "runtime receipt status drift")
    checks.require(receipt["authorization"] == "NOT_PRODUCTION_AUTHORITY_UNTIL_INDEPENDENT_AUDIT_AND_LATER_ONE_SHOT_AUTHORIZATION", "runtime receipt authority boundary drift")
    sealed(checks, receipt, "runtime_receipt_sha256", RUNTIME_INTERNAL_SHA256, "runtime receipt")
    walk_hash_fields(checks, receipt, "runtime_receipt")

    for label, section, path, digest in (
        ("launch_manifest", receipt["launch_manifest"], f"{STAGE_ROOT}/launch_manifest.json", LAUNCH_SHA256),
        ("runtime_contract", receipt["runtime_contract"], f"{STAGE_ROOT}/runtime_contract.json", RUNTIME_CONTRACT_SHA256),
        ("calibrator", receipt["calibrator"], f"{STAGE_ROOT}/runtime_calibrate.py", CALIBRATOR_SHA256),
        ("scientific_core", receipt["scientific_core"], f"{STAGE_ROOT}/lossy_tail_core.py", CORE_SHA256),
    ):
        exact_keys(checks, section, {"path", "sha256"}, label)
        checks.require(section["path"] == path, f"{label} path drift")
        checks.require(section["sha256"] == digest, f"{label} hash drift")

    invocation = receipt["invocation"]
    exact_keys(checks, invocation, {"argv", "cwd", "python_flags"}, "invocation")
    expected_argv = [
        f"{STAGE_ROOT}/runtime_calibrate.py", "--manifest",
        f"{STAGE_ROOT}/launch_manifest.json", "--manifest-sha256", LAUNCH_SHA256,
        "--output", EXPECTED_OUTPUT_PATH,
    ]
    checks.require(invocation["argv"] == expected_argv, "runtime invocation argv drift")
    checks.require(invocation["cwd"] == EXPECTED_CWD, "runtime invocation cwd drift")
    checks.require(invocation["python_flags"] == ["-B", "-I"], "runtime invocation Python flags drift")

    access = receipt["access_ledger"]
    exact_keys(checks, access, {
        "model_or_qwen_paths_supplied", "model_or_qwen_paths_opened", "payload_files_opened",
        "production_results_opened", "production_outputs_created", "runtime_receipts_created",
    }, "runtime access ledger")
    for key in (
        "model_or_qwen_paths_supplied", "model_or_qwen_paths_opened", "payload_files_opened",
        "production_results_opened", "production_outputs_created",
    ):
        checks.require(access[key] == 0, f"runtime source-free ledger drift: {key}")
    checks.require(access["runtime_receipts_created"] == 1, "runtime receipt creation count drift")

    probe = receipt["runtime_probe"]
    exact_keys(checks, probe, {"runtime_tuple", "cells", "stable_order", "probe_aggregate_sha256"}, "runtime probe")
    runtime_tuple = probe["runtime_tuple"]
    expected_tuple = {
        "python_implementation": "CPython",
        "python_version": "3.12.3",
        "python_executable": "/workspace/int2-cupy-venv/bin/python",
        "numpy_version": "2.5.2",
        "cupy_version": "14.2.0",
        "cuda_runtime_integer": 12090,
        "cuda_driver_integer": 13000,
        "device_count": 1,
        "device_ordinal": 0,
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_total_global_mem": 33668857856,
        "device_compute_capability": [12, 0],
        "device_multiprocessor_count": 170,
        "cuda_visible_devices": "0",
    }
    exact_keys(checks, runtime_tuple, expected_tuple, "runtime tuple")
    checks.require(runtime_tuple == expected_tuple, "exact runtime tuple mismatch")
    for key, expected in contract["expected_runtime_tuple"].items():
        checks.require(runtime_tuple.get(key) == expected, f"runtime tuple violates contract: {key}")

    cells = probe["cells"]
    checks.require(isinstance(cells, list) and len(cells) == 48, "runtime cell count is not 48")
    seen_seeds: set[int] = set()
    seen_raw: set[str] = set()
    seen_bf16: set[str] = set()
    seen_affine: set[str] = set()
    expected_sentinels = [
        ("0x1.4000000000000p-2", "-0x1.8000000000000p-7"),
        ("0x1.8000000000000p+0", "0x1.0000000000000p-8"),
    ]
    for index, cell in enumerate(cells):
        label = f"cell[{index}]"
        exact_keys(checks, cell, {
            "replica", "ordinal", "seed_u64", "raw_float32_u32_little_endian_sha256",
            "rne_bf16_u16_little_endian_sha256", "float64_mean_of_bf16_values_hex",
            "float64_population_variance_hex", "affine_sentinels", "memory_release",
        }, label)
        replica, ordinal = divmod(index, 12)
        checks.require(type(cell["replica"]) is int and cell["replica"] == replica, f"{label} replica/order drift")
        checks.require(type(cell["ordinal"]) is int and cell["ordinal"] == ordinal, f"{label} ordinal/order drift")
        expected_seed = stable_seed(replica, ordinal)
        checks.require(type(cell["seed_u64"]) is int and 0 <= cell["seed_u64"] < 2**64, f"{label} seed type/range drift")
        checks.require(cell["seed_u64"] == expected_seed, f"{label} seed mismatch")
        checks.require(cell["seed_u64"] not in seen_seeds, f"duplicate seed at {label}")
        seen_seeds.add(cell["seed_u64"])
        raw_hash = cell["raw_float32_u32_little_endian_sha256"]
        bf16_hash = cell["rne_bf16_u16_little_endian_sha256"]
        valid_sha(checks, raw_hash, f"{label}.raw")
        valid_sha(checks, bf16_hash, f"{label}.bf16")
        checks.require(raw_hash not in seen_raw, f"duplicate raw-vector hash at {label}")
        checks.require(bf16_hash not in seen_bf16, f"duplicate BF16-vector hash at {label}")
        seen_raw.add(raw_hash)
        seen_bf16.add(bf16_hash)
        mean = parse_finite_float_hex(checks, cell["float64_mean_of_bf16_values_hex"], f"{label}.mean")
        variance = parse_finite_float_hex(checks, cell["float64_population_variance_hex"], f"{label}.variance")
        checks.require(abs(mean) < 0.01, f"implausible Gaussian mean at {label}")
        checks.require(0.95 < variance < 1.05, f"implausible Gaussian variance at {label}")
        affine_rows = cell["affine_sentinels"]
        checks.require(isinstance(affine_rows, list) and len(affine_rows) == 2, f"{label} affine cardinality drift")
        for affine_index, affine in enumerate(affine_rows):
            affine_label = f"{label}.affine[{affine_index}]"
            exact_keys(checks, affine, {
                "scale_float32_hex", "offset_float32_hex",
                "gathered_float32_u32_little_endian_sha256", "float64_mean_hex",
                "float64_population_variance_hex",
            }, affine_label)
            scale_hex, offset_hex = expected_sentinels[affine_index]
            checks.require(affine["scale_float32_hex"] == scale_hex, f"{affine_label} scale/order drift")
            checks.require(affine["offset_float32_hex"] == offset_hex, f"{affine_label} offset/order drift")
            gathered_hash = affine["gathered_float32_u32_little_endian_sha256"]
            valid_sha(checks, gathered_hash, f"{affine_label}.gathered")
            checks.require(gathered_hash not in seen_affine, f"duplicate gathered hash at {affine_label}")
            seen_affine.add(gathered_hash)
            affine_mean = parse_finite_float_hex(checks, affine["float64_mean_hex"], f"{affine_label}.mean")
            affine_variance = parse_finite_float_hex(checks, affine["float64_population_variance_hex"], f"{affine_label}.variance")
            scale = float.fromhex(scale_hex)
            offset = float.fromhex(offset_hex)
            checks.require(abs(affine_mean - (scale * mean + offset)) < 2e-7, f"{affine_label} mean inconsistent with BF16 moments")
            checks.require(abs(affine_variance - scale * scale * variance) < 2e-6, f"{affine_label} variance inconsistent with BF16 moments")
            checks.require(affine_variance > 0.0, f"nonpositive affine variance at {affine_label}")
        verify_memory(checks, cell["memory_release"], f"{label}.memory_release")
        checks.require(cell["memory_release"]["total_bytes_before_free"] == 85917696, f"{label} cached-byte evidence drift")
    checks.require(len(seen_seeds) == 48, "seed cardinality drift")
    checks.require(len(seen_raw) == 48, "raw-hash cardinality drift")
    checks.require(len(seen_bf16) == 48, "BF16-hash cardinality drift")
    checks.require(len(seen_affine) == 96, "affine-hash cardinality drift")

    stable_rows = probe["stable_order"]
    checks.require(isinstance(stable_rows, list) and len(stable_rows) == 5, "stable-order row count is not five")
    expected_stable = stable_fixture_rows()
    stable_inputs: set[str] = set()
    stable_orders: set[str] = set()
    for index, row in enumerate(stable_rows):
        label = f"stable_order[{index}]"
        exact_keys(checks, row, {"ordinal", "dtype", "count", "input_sha256", "order_sha256", "memory_release"}, label)
        for key, expected in expected_stable[index].items():
            checks.require(row[key] == expected, f"{label}.{key} independent fixture mismatch")
        checks.require(row["input_sha256"] not in stable_inputs, f"duplicate stable input hash at {label}")
        checks.require(row["order_sha256"] not in stable_orders, f"duplicate stable order hash at {label}")
        stable_inputs.add(row["input_sha256"])
        stable_orders.add(row["order_sha256"])
        verify_memory(checks, row["memory_release"], f"{label}.memory_release")
        checks.require(row["memory_release"]["total_bytes_before_free"] > 0, f"{label} lacks memory allocation evidence")
    checks.require(len(stable_inputs) == 5 and len(stable_orders) == 5, "stable-order uniqueness closure failed")

    aggregate_core = {
        "runtime_tuple": runtime_tuple,
        "cells": cells,
        "stable_order": stable_rows,
    }
    valid_sha(checks, probe["probe_aggregate_sha256"], "probe aggregate")
    checks.require(probe["probe_aggregate_sha256"] == PROBE_AGGREGATE_SHA256, "probe aggregate frozen value drift")
    checks.require(sha256(canonical_bytes(aggregate_core)) == PROBE_AGGREGATE_SHA256, "probe aggregate recomputation failed")

    required_zero = contract["calibration_output"]["required_zero_access"]
    checks.require(required_zero == {
        "model_or_qwen_paths_supplied": 0,
        "model_or_qwen_paths_opened": 0,
        "payload_files_opened": 0,
        "production_results_opened": 0,
    }, "runtime-contract zero-access drift")
    checks.require(contract["probe"]["replicas"] == [0, 1, 2, 3], "runtime-contract replica drift")
    checks.require(contract["probe"]["ordinals"] == list(range(12)), "runtime-contract ordinal drift")
    checks.require(contract["probe"]["values_per_vector"] == 1572864, "runtime-contract vector size drift")
    checks.require(contract["probe"]["affine_sentinels"] == [
        {"scale_float32_hex": expected_sentinels[0][0], "offset_float32_hex": expected_sentinels[0][1]},
        {"scale_float32_hex": expected_sentinels[1][0], "offset_float32_hex": expected_sentinels[1][1]},
    ], "runtime-contract affine sentinel drift")
    checks.require(source_receipt["source_free_calibration_boundary"]["receipt_status"] == receipt["status"], "source-audit/runtime receipt status disagreement")
    checks.require(authorization["required_values"]["runtime_receipt_schema"] == receipt["schema"], "authorization/runtime receipt schema disagreement")
    checks.require(authorization["required_values"]["runtime_receipt_status"] == receipt["status"], "authorization/runtime receipt status disagreement")
    return receipt


def verify_audit_package(checks: Checks, evidence_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    checks.package_mode()
    manifest_payload, manifest = strict_json(HERE / "audit_manifest.json")
    receipt_payload, receipt = strict_json(HERE / "audit_receipt.json")
    exact_keys(checks, manifest, {
        "schema", "status", "created_utc", "audited_target", "source_audit_binding",
        "audit_artifacts", "audit_scope", "access_ledger", "authorization",
        "audit_manifest_sha256",
    }, "audit manifest")
    exact_keys(checks, receipt, {
        "schema", "status", "created_utc", "audited_target", "audited_runtime_receipt",
        "source_audit_binding", "runtime_identity", "runtime_tuple", "probe_closure", "check_summary",
        "access_ledger", "audit_transport", "authorization", "audit_receipt_sha256",
    }, "audit receipt")
    checks.require(manifest["schema"] == "lossy-tail-v8-independent-runtime-audit-manifest-v1", "audit manifest schema drift")
    checks.require(manifest["status"] == "IMMUTABLE_PASS_AUDIT_ARTIFACT_SET", "audit manifest status drift")
    checks.require(receipt["schema"] == "lossy-tail-v8-independent-runtime-audit-receipt-v1", "audit receipt schema drift")
    checks.require(receipt["status"] == "PASS_V8_INDEPENDENT_RUNTIME_AUDIT", "audit receipt status drift")
    sealed(checks, manifest, "audit_manifest_sha256", manifest["audit_manifest_sha256"], "audit manifest")
    sealed(checks, receipt, "audit_receipt_sha256", receipt["audit_receipt_sha256"], "audit receipt")
    checks.require(receipt["check_summary"]["independent_evidence_check_count"] == evidence_count, "recorded evidence-check count drift")
    checks.require(receipt["check_summary"]["blocker_count"] == 0, "PASS receipt records blockers")
    checks.require(receipt["probe_closure"]["ordered_rng_cells"] == 48, "audit receipt cell count drift")
    checks.require(receipt["probe_closure"]["affine_rows_per_cell"] == 2, "audit receipt affine cardinality drift")
    checks.require(receipt["probe_closure"]["stable_order_rows"] == 5, "audit receipt stable count drift")
    checks.require(receipt["probe_closure"]["memory_release_rows"] == 53, "audit receipt memory count drift")
    checks.require(receipt["runtime_identity"]["file_sha256"] == RUNTIME_FILE_SHA256, "audit receipt runtime file hash drift")
    checks.require(receipt["runtime_identity"]["bytes"] == RUNTIME_FILE_BYTES, "audit receipt runtime size drift")
    checks.require(receipt["runtime_identity"]["internal_sha256"] == RUNTIME_INTERNAL_SHA256, "audit receipt runtime seal drift")
    checks.require(receipt["runtime_identity"]["probe_aggregate_sha256"] == PROBE_AGGREGATE_SHA256, "audit receipt aggregate drift")
    exact_keys(checks, receipt["audited_runtime_receipt"], {"file_sha256", "internal_sha256"}, "audited runtime receipt binding")
    checks.require(receipt["audited_runtime_receipt"]["file_sha256"] == RUNTIME_FILE_SHA256, "preflight-visible runtime file binding drift")
    checks.require(receipt["audited_runtime_receipt"]["internal_sha256"] == RUNTIME_INTERNAL_SHA256, "preflight-visible runtime internal binding drift")
    for ledger_label, ledger in (("manifest", manifest["access_ledger"]), ("receipt", receipt["access_ledger"])):
        for key in ("model_payload_files_opened", "production_result_files_opened", "gpu_jobs"):
            checks.require(ledger.get(key) == 0, f"audit {ledger_label} required zero-access drift: {key}")
    rows = manifest["audit_artifacts"]
    checks.require(isinstance(rows, list) and len(rows) == 4, "audit artifact cardinality drift")
    expected_names = {"README.md", "audit_receipt.json", "runtime_receipt.json", "verify_runtime_audit.py"}
    checks.require({row.get("path") for row in rows if isinstance(row, dict)} == expected_names, "audit artifact closure drift")
    for row in rows:
        exact_keys(checks, row, {"path", "bytes", "sha256"}, f"audit artifact {row.get('path')}")
        artifact_payload = read_regular(HERE / row["path"])
        checks.require(len(artifact_payload) == row["bytes"], f"audit artifact size drift: {row['path']}")
        checks.require(sha256(artifact_payload) == row["sha256"], f"audit artifact hash drift: {row['path']}")
    checks.require(sha256(receipt_payload) == next(row["sha256"] for row in rows if row["path"] == "audit_receipt.json"), "manifest/receipt file hash mismatch")
    walk_hash_fields(checks, manifest, "audit_manifest")
    walk_hash_fields(checks, receipt, "audit_receipt")
    return manifest, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--print-seal", choices=("receipt", "manifest"))
    args = parser.parse_args()
    if args.print_seal:
        name, field = (
            ("audit_receipt.json", "audit_receipt_sha256")
            if args.print_seal == "receipt"
            else ("audit_manifest.json", "audit_manifest_sha256")
        )
        _, value = strict_json(HERE / name)
        value.pop(field, None)
        print(sha256(canonical_bytes(value)))
        return
    checks = Checks()
    try:
        verify_runtime_evidence(checks)
        if not args.evidence_only:
            verify_audit_package(checks, checks.evidence)
    except AuditFailure as exc:
        print(json.dumps({
            "status": "BLOCK_V8_INDEPENDENT_RUNTIME_AUDIT",
            "error": str(exc),
            "independent_evidence_checks_before_failure": checks.evidence,
            "package_checks_before_failure": checks.package,
            "model_payload_files_opened": 0,
            "production_result_files_opened": 0,
            "gpu_jobs": 0,
        }, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({
        "status": "PASS_V8_INDEPENDENT_RUNTIME_AUDIT_EVIDENCE_ONLY" if args.evidence_only else "PASS_V8_INDEPENDENT_RUNTIME_AUDIT",
        "independent_evidence_check_count": checks.evidence,
        "package_check_count": checks.package,
        "runtime_receipt_file_sha256": RUNTIME_FILE_SHA256,
        "runtime_receipt_bytes": RUNTIME_FILE_BYTES,
        "runtime_receipt_internal_sha256": RUNTIME_INTERNAL_SHA256,
        "probe_aggregate_sha256": PROBE_AGGREGATE_SHA256,
        "model_payload_files_opened": 0,
        "production_result_files_opened": 0,
        "gpu_jobs": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
