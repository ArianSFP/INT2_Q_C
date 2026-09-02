#!/usr/bin/env python3
"""Fail-closed independent auditor for a real posterior-centroid v0 result.

The verifier authenticates an externally pinned v9 publication, its completed
independent v9 result-audit receipt, the posterior publication, all decoder
sources and the BF16 score panel before it accepts any numerical claim.  It
then refits and scores the three posterior laws without importing the
producer's diagnostic or posterior core.

Direct execution is intentionally inert unless the explicit authorization,
isolated CPython flags, source-manifest hash and external pin hash are given.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


AUTHORIZATION = "AUDIT_EXACT_UWFA_SC_POSTERIOR_CENTROID_V0_RESULT"
PINS_SCHEMA = "uwfa-sc-posterior-centroid-v0-result-audit-external-pins-v0"
AUDIT_SCHEMA = "uwfa-sc-posterior-centroid-v0-independent-result-audit-v0"
AUDIT_COMPLETION_SCHEMA = "uwfa-sc-posterior-centroid-v0-independent-result-audit-completion-v0"
POSTERIOR_RESULT_SCHEMA = "uwfa-sc-posterior-centroid-result-v0"
POSTERIOR_COMPLETION_SCHEMA = "uwfa-sc-posterior-centroid-completion-v0"
V9_RESULT_SCHEMA = "uwfa-sc-v9-qwen-primary-gate-v0"
V9_AUDIT_SCHEMA = "uwfa-sc-v9-primary-independent-result-audit-v0"
SOURCE_PANEL_SCHEMA = "swiglu-bf16-score-panel-v0"

KNOWN_POSTERIOR_MANIFEST_SHA256 = "0ef30253d4d31504fbd8f88b8203cf35bce6c14952e570aace44b7bc089cb713"
KNOWN_POSTERIOR_SOURCE_ROOT_SHA256 = "ea3ad9cf9b723cdf7501eeff004bd7f2821af4d37ff186b72f2972482a05e11c"
KNOWN_RESULT_BRIDGE_SHA256 = "112efcad5fd3fe9bccfea11af03bd9124a3789b6107c4486dd961656398e4d79"
KNOWN_V9_RESULT_AUDIT_MANIFEST_SHA256 = "885f41e27c439c808e2118de52184feaec58efe9f14bbc0e02a377e3b189f5ee"
KNOWN_V8_MANIFEST_SHA256 = "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6"
KNOWN_STRATA_SHA256 = "3f085c9531b714d0d7877388f54ae50495dc3ea631491563abceb4db55608fd1"
KNOWN_FROZEN_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"
AUDITOR_SOURCE_ROOT_DOMAIN = b"UWFA-SC-POSTERIOR-CENTROID-V0-RESULT-AUDITOR-SOURCE-ROOT-V0\x00"

V9_MEMBERS = frozenset({
    "BOUND_BASELINE_SCORE.json",
    "COMPLETE.json",
    "DECODER_BUNDLE.json",
    "IDENTITY_FRAMING.bin",
    "RESULT.json",
    "SOURCE_PREFLIGHT.json",
    "UWFCV8.bin",
})
POSTERIOR_BASE_MEMBERS = frozenset({
    "COMPLETE.json",
    "RESULT.json",
    *(f"FOLD{outer}_{law}.cagepst1" for outer in range(3) for law in ("LOCAL_ONLY", "STATE_AWARE", "STATE_PERMUTED")),
})
POSTERIOR_FINAL_MEMBER = "FINAL_STATE_AWARE.cagepst1"
MAX_JSON = 1 << 25
MAX_SOURCE_MANIFEST = 1 << 22
MAX_MATRIX = 1 << 31
MAX_CONTAINER = 1 << 31
OWN_SOURCE_MEMBERS = frozenset({
    "API_AUDIT.md", "audit_core.py", "BLOCK.json", "design_lock.json", "README.md",
    "result_auditor.py", "test_source_only.py",
    "UNRESOLVED_EXTERNAL_PINS.json", "verify_source.py",
})
POSTERIOR_SOURCE_MEMBERS = frozenset({
    "README.md", "design_lock.json", "diagnostic.py", "posterior_core.py",
    "result_bridge.py", "test_source_only.py",
})


class ResultAuditError(RuntimeError):
    """Fail-closed result-audit error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultAuditError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} SHA-256",
    )
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def auditor_source_root(rows: Sequence[Mapping[str, Any]]) -> str:
    """Domain-separated root over exact ASCII-byte-sorted canonical rows."""

    clean = []
    for row in rows:
        require(isinstance(row, Mapping) and set(row) == {"name", "bytes", "sha256"}, "auditor source-root row")
        name = safe_name(row["name"], "auditor source-root member")
        require(name.isascii(), "auditor source-root ASCII name")
        size = row["bytes"]
        require(type(size) is int and size > 0, "auditor source-root bytes")
        clean.append({"name": name, "bytes": size, "sha256": digest(row["sha256"], "auditor source-root member")})
    clean.sort(key=lambda row: row["name"].encode("ascii"))
    require(len({row["name"] for row in clean}) == len(clean), "auditor source-root unique members")
    return sha256(AUDITOR_SOURCE_ROOT_DOMAIN + canonical_json(clean))


def _object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResultAuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(payload: bytes, label: str, maximum: int = MAX_JSON) -> dict[str, Any]:
    require(isinstance(payload, bytes) and 0 < len(payload) <= maximum, f"{label} byte bound")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ResultAuditError(f"{label} nonfinite: {token}")),
        )
    except ResultAuditError:
        raise
    except Exception as error:
        raise ResultAuditError(f"{label} JSON: {error}") from error
    require(isinstance(value, dict), f"{label} object")
    return value


def exact_fields(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping) and set(value) == fields, f"{label} exact fields")
    return value


def safe_name(value: Any, label: str) -> str:
    require(isinstance(value, str) and value and value not in (".", ".."), f"{label} name")
    require(Path(value).name == value and "/" not in value and "\\" not in value, f"{label} basename")
    return value


def absolute_path(value: Any, label: str, *, directory: bool | None = None) -> Path:
    require(isinstance(value, str) and value, f"{label} path")
    path = Path(value)
    require(path.is_absolute(), f"{label} absolute path")
    cursor = path
    while True:
        require(not stat.S_ISLNK(os.lstat(cursor).st_mode), f"{label} symlink chain")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    metadata = os.lstat(path)
    require(not stat.S_ISLNK(metadata.st_mode), f"{label} symlink")
    if directory is True:
        require(stat.S_ISDIR(metadata.st_mode), f"{label} directory")
    elif directory is False:
        require(stat.S_ISREG(metadata.st_mode), f"{label} regular file")
    return path


def regular_bytes(path: Path, *, maximum: int, label: str, expected_sha256: str | None = None, expected_bytes: int | None = None) -> bytes:
    before = os.lstat(path)
    require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"{label} regular nofollow")
    require(0 < before.st_size <= maximum, f"{label} size bound")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        require((opened.st_dev, opened.st_ino, opened.st_size) == (before.st_dev, before.st_ino, before.st_size), f"{label} inode binding")
        chunks = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label} short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns), f"{label} changed during read")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if expected_bytes is not None:
        require(len(payload) == expected_bytes, f"{label} external byte pin")
    if expected_sha256 is not None:
        require(sha256(payload) == digest(expected_sha256, f"{label} external"), f"{label} external hash pin")
    return payload


def load_module(name: str, payload: bytes, expected: str) -> ModuleType:
    require(sha256(payload) == digest(expected, f"module {name}"), f"module {name} retained bytes")
    require(name not in sys.modules, f"fresh module {name}")
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    module.__file__ = f"<authenticated:{name}:{expected}>"
    module.__authenticated_sha256__ = expected
    sys.modules[name] = module
    try:
        exec(compile(payload, module.__file__, "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def authenticate_source_package(package: Path, expected_manifest_sha256: str, *, own: bool) -> dict[str, Any]:
    manifest_payload = regular_bytes(package / "SOURCE_MANIFEST.json", maximum=MAX_JSON, label="source package manifest", expected_sha256=expected_manifest_sha256)
    manifest = strict_json(manifest_payload, "source package manifest")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "source package rows")
    observed = []
    sources = {}
    for row in rows:
        exact_fields(row, {"name", "bytes", "sha256"}, "source package member row")
        name = safe_name(row["name"], "source package member")
        require(name != "SOURCE_MANIFEST.json", "manifest recursion")
        size = row["bytes"]
        require(type(size) is int and 0 < size <= MAX_JSON, "source member bytes")
        member_hash = digest(row["sha256"], "source member")
        payload = regular_bytes(package / name, maximum=MAX_JSON, label=f"source member {name}", expected_sha256=member_hash, expected_bytes=size)
        sources[name] = payload
        observed.append({"name": name, "bytes": size, "sha256": member_hash})
    require(len(sources) == len(rows), "source package duplicate members")
    expected_names = OWN_SOURCE_MEMBERS if own else POSTERIOR_SOURCE_MEMBERS
    require(set(sources) == expected_names, "source package exact member names")
    require(set(os.listdir(package)) == expected_names | {"SOURCE_MANIFEST.json"}, "source package exact directory closure")
    root = (
        auditor_source_root(observed)
        if own
        else sha256(canonical_json(sorted(observed, key=lambda item: item["name"])))
    )
    if own:
        require(manifest.get("schema") == "uwfa-sc-posterior-centroid-v0-result-audit-source-manifest-v0", "auditor source schema")
        require(manifest.get("status") == "SEALED_SOURCE_ONLY_AWAITING_EXTERNAL_RESULT_PINS", "auditor source status")
        require(manifest.get("source_snapshot_root_sha256") == root, "auditor source root")
        require("result_auditor.py" in sources and sources["result_auditor.py"] == regular_bytes(Path(__file__).resolve(), maximum=MAX_JSON, label="running auditor"), "executing auditor closure")
    else:
        require(manifest.get("schema") == "uwfa-sc-posterior-centroid-source-manifest-v0", "posterior producer manifest schema")
        require(manifest.get("source_snapshot_root_sha256") == KNOWN_POSTERIOR_SOURCE_ROOT_SHA256 == root, "posterior producer source root")
        require(sha256(sources.get("result_bridge.py", b"")) == KNOWN_RESULT_BRIDGE_SHA256, "posterior bridge pin")
    return {"manifest": manifest, "manifest_sha256": sha256(manifest_payload), "source_snapshot_root_sha256": root, "sources": sources}


def member_rows(value: Any, label: str) -> dict[str, dict[str, Any]]:
    require(isinstance(value, Mapping) and value, f"{label} member pins")
    output = {}
    for raw_name, raw_row in value.items():
        name = safe_name(raw_name, f"{label} member")
        exact_fields(raw_row, {"bytes", "sha256"}, f"{label} {name}")
        size = raw_row["bytes"]
        require(type(size) is int and size > 0, f"{label} {name} bytes")
        output[name] = {"bytes": size, "sha256": digest(raw_row["sha256"], f"{label} {name}")}
    return output


def parse_pins(record: Mapping[str, Any]) -> dict[str, Any]:
    exact_fields(record, {"schema", "status", "paths", "hashes", "v9_publication_members", "posterior_publication_members", "access_authorization"}, "external pins")
    require(record["schema"] == PINS_SCHEMA and record["status"] == "RESOLVED_IMMUTABLE_EXTERNAL_AUTHORITY", "external pin schema/status")
    paths = exact_fields(record["paths"], {
        "posterior_producer_package", "v9_publication", "v9_result_audit_receipt",
        "v8_package", "strata_common", "frozen_auditor", "source_manifest",
        "posterior_publication", "audit_output_parent", "audit_output_name",
    }, "external paths")
    hashes = exact_fields(record["hashes"], {
        "posterior_producer_manifest_sha256", "v9_result_audit_source_manifest_sha256",
        "v9_result_audit_receipt_sha256", "v8_manifest_sha256", "strata_common_sha256",
        "frozen_auditor_sha256", "source_manifest_sha256",
    }, "external hashes")
    frozen = {
        "posterior_producer_manifest_sha256": KNOWN_POSTERIOR_MANIFEST_SHA256,
        "v9_result_audit_source_manifest_sha256": KNOWN_V9_RESULT_AUDIT_MANIFEST_SHA256,
        "v8_manifest_sha256": KNOWN_V8_MANIFEST_SHA256,
        "strata_common_sha256": KNOWN_STRATA_SHA256,
        "frozen_auditor_sha256": KNOWN_FROZEN_SHA256,
    }
    for name, expected in frozen.items():
        require(digest(hashes[name], name) == expected, f"{name} frozen pin")
    digest(hashes["v9_result_audit_receipt_sha256"], "v9 audit receipt")
    digest(hashes["source_manifest_sha256"], "source manifest")
    require(record["access_authorization"] == {
        "may_open_completed_v9_publication": True,
        "may_open_completed_posterior_publication": True,
        "may_open_bf16_score_panel": True,
        "may_initialize_cupy_for_exact_rht_replay": True,
        "positive_claim_authority": False,
    }, "external access authorization")
    v9_rows = member_rows(record["v9_publication_members"], "v9 publication")
    posterior_rows = member_rows(record["posterior_publication_members"], "posterior publication")
    require(set(v9_rows) == V9_MEMBERS, "v9 exact external member set")
    require(set(posterior_rows) in (set(POSTERIOR_BASE_MEMBERS), set(POSTERIOR_BASE_MEMBERS) | {POSTERIOR_FINAL_MEMBER}), "posterior exact external member set")
    output_name = safe_name(paths["audit_output_name"], "audit output name")
    require(output_name not in (Path(paths["v9_publication"]).name, Path(paths["posterior_publication"]).name), "audit output distinct")
    return {"paths": dict(paths), "hashes": dict(hashes), "v9_rows": v9_rows, "posterior_rows": posterior_rows}


def authenticate_publication(path: Path, rows: Mapping[str, Mapping[str, Any]], label: str) -> dict[str, Any]:
    metadata = os.lstat(path)
    require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"{label} directory")
    names = set(os.listdir(path))
    require(names == set(rows), f"{label} exact directory members")
    payloads = {}
    for name in sorted(rows):
        maximum = MAX_CONTAINER if name.endswith((".bin", ".cagepst1")) else MAX_JSON
        payloads[name] = regular_bytes(path / name, maximum=maximum, label=f"{label} {name}", expected_sha256=rows[name]["sha256"], expected_bytes=rows[name]["bytes"])
    complete = strict_json(payloads["COMPLETE.json"], f"{label} completion")
    completion_rows = complete.get("members")
    require(isinstance(completion_rows, list), f"{label} completion rows")
    expected_data = sorted(set(rows) - {"COMPLETE.json"})
    require([row.get("name") for row in completion_rows] == expected_data, f"{label} canonical completion member order")
    for row, name in zip(completion_rows, expected_data, strict=True):
        exact_fields(row, {"name", "bytes", "sha256"}, f"{label} completion row")
        require(row["bytes"] == len(payloads[name]) and row["sha256"] == sha256(payloads[name]), f"{label} completion member binding")
    if "completion_sha256" in complete:
        clean = dict(complete)
        seal = digest(clean.pop("completion_sha256"), f"{label} completion seal")
        require(sha256(canonical_json(clean)) == seal, f"{label} completion internal seal")
    return {
        "path": path,
        "payloads": payloads,
        "rows": {name: {"bytes": len(payload), "sha256": sha256(payload)} for name, payload in sorted(payloads.items())},
        "complete": complete,
        "publication_sha256": sha256(canonical_json({name: {"bytes": len(payload), "sha256": sha256(payload)} for name, payload in sorted(payloads.items())})),
    }


def authenticate_v9_audit_receipt(payload: bytes, v9: Mapping[str, Any], expected_manifest_sha256: str) -> dict[str, Any]:
    receipt = strict_json(payload, "v9 independent audit receipt", MAX_JSON)
    require(receipt.get("schema") == V9_AUDIT_SCHEMA, "v9 audit receipt schema")
    require(receipt.get("status") == "PASS_FAIL_CLOSED_NONPROMOTING_PRIMARY_RESULT_AUDIT", "v9 audit receipt status")
    require(receipt.get("positive_claim_authority") is False, "v9 audit nonpromotion")
    require(receipt.get("publication_members") == v9["rows"], "v9 audit/publication literal binding")
    source = receipt.get("source_closure")
    require(isinstance(source, Mapping), "v9 audit source closure")
    require(source.get("v8_manifest_sha256") == KNOWN_V8_MANIFEST_SHA256, "v9 audit v8 closure")
    require(source.get("strata_common_sha256") == KNOWN_STRATA_SHA256, "v9 audit STRATA closure")
    require(source.get("frozen_auditor_sha256") == KNOWN_FROZEN_SHA256, "v9 audit frozen closure")
    audit_source = receipt.get("audit_source_closure")
    require(isinstance(audit_source, Mapping) and audit_source.get("manifest_sha256") == expected_manifest_sha256, "v9 result-auditor source closure")
    literal = receipt.get("literal_container_audit")
    require(isinstance(literal, Mapping) and literal.get("candidate_sha256") == v9["rows"]["UWFCV8.bin"]["sha256"], "v9 independently audited literal")
    return receipt


def authenticate_source_manifest(core: Any, payload: bytes, path: Path, *, artifact_sha256: str, experts: int, intermediate: int, hidden: int) -> dict[str, Any]:
    manifest = strict_json(payload, "BF16 source manifest", MAX_SOURCE_MANIFEST)
    require(manifest.get("schema") == SOURCE_PANEL_SCHEMA, "source panel schema")
    require(manifest.get("bound_artifact_sha256") == digest(artifact_sha256, "source artifact"), "source/artifact binding")
    require(type(manifest.get("experts")) is int and manifest["experts"] == experts, "source expert count")
    forbidden = {"checkpoint", "checkpoint_name", "model", "model_name", "model_family", "layer", "layer_name", "tensor_name"}
    require(not (set(manifest) & forbidden), "source manifest identity fields")
    rows = manifest.get("matrices")
    require(isinstance(rows, list) and len(rows) == 3 * experts, "source matrix count")
    expected_keys = {(expert, role) for expert in range(experts) for role in ("gate", "up", "down")}
    records = {}
    clean = []
    root = path.parent.resolve()
    for ordinal, row in enumerate(rows):
        exact_fields(row, {"expert_ordinal", "role", "shape", "relative_path", "bytes", "sha256"}, f"source matrix row {ordinal}")
        expert, role = row["expert_ordinal"], row["role"]
        require(type(expert) is int and 0 <= expert < experts and role in ("gate", "up", "down"), "source matrix identity-free coordinate")
        key = (expert, role)
        require(key not in records, "source duplicate matrix")
        shape = [hidden, intermediate] if role == "down" else [intermediate, hidden]
        require(row["shape"] == shape, "source matrix shape")
        size = 2 * intermediate * hidden
        require(type(row["bytes"]) is int and row["bytes"] == size, "source matrix bytes")
        relative = row["relative_path"]
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts, "source relative path")
        candidate_path = root / relative
        cursor = candidate_path
        while True:
            require(not stat.S_ISLNK(os.lstat(cursor).st_mode), "source matrix symlink chain")
            if cursor == root:
                break
            require(root in cursor.parents, "source path containment")
            cursor = cursor.parent
        matrix_path = candidate_path.resolve(strict=True)
        require(matrix_path.parent == root or root in matrix_path.parents, "source path containment")
        record = {
            "expert_ordinal": expert,
            "role": role,
            "shape": shape,
            "bytes": size,
            "sha256": digest(row["sha256"], "source matrix"),
        }
        clean.append(record)
        records[key] = {**record, "path": matrix_path}
    require(set(records) == expected_keys, "source complete role grid")
    clean.sort(key=lambda row: (row["expert_ordinal"], ("gate", "up", "down").index(row["role"])))
    record_root = sha256(core.canonical_json(clean))
    require(manifest.get("source_record_set_sha256") == record_root, "source record-set root")
    return {
        "manifest": manifest,
        "manifest_sha256": sha256(payload),
        "source_record_set_sha256": record_root,
        "records": records,
        "experts": experts,
        "intermediate": intermediate,
        "hidden": hidden,
    }


def load_source_subset(np: Any, source: Mapping[str, Any], selected_experts: Sequence[int]) -> dict[str, Any]:
    selected = tuple(sorted(int(value) for value in selected_experts))
    require(selected and tuple(sorted(set(selected))) == selected, "selected source experts")
    matrices = {}
    energy = 0.0
    receipts = []
    for expert in selected:
        for role in ("gate", "up", "down"):
            row = source["records"][(expert, role)]
            raw = regular_bytes(row["path"], maximum=MAX_MATRIX, label=f"source expert {expert} {role}", expected_sha256=row["sha256"], expected_bytes=row["bytes"])
            words = np.frombuffer(raw, dtype="<u2")
            values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
            shape = tuple(row["shape"])
            require(values.size == shape[0] * shape[1] and bool(np.all(np.isfinite(values))), "BF16 source values")
            matrix = values.reshape(shape)
            matrices[(expert, role)] = matrix
            energy += float(np.sum(matrix * matrix, dtype=np.float64))
            receipts.append({"expert_ordinal": expert, "role": role, "bytes": len(raw), "sha256": sha256(raw)})
    return {"matrices": matrices, "materialized_experts": list(selected), "source_energy_fp64": energy, "member_receipts": receipts}


def source_post_coordinates(np: Any, source_subset: Mapping[str, Any], metadata: Mapping[str, Any], *, experts: int, intermediate: int, hidden: int) -> Any:
    coefficients = struct.unpack_from(f"<{2 * experts}f", bytes(metadata["header"]), 32)
    post = np.empty((experts * 3 * intermediate, hidden), dtype=np.float64)
    for expert in source_subset["materialized_experts"]:
        base = expert * 3 * intermediate
        gate = np.asarray(source_subset["matrices"][(expert, "gate")], dtype=np.float64)
        up = np.asarray(source_subset["matrices"][(expert, "up")], dtype=np.float64)
        down = np.asarray(source_subset["matrices"][(expert, "down")], dtype=np.float64).T
        cosine, sine = float(coefficients[2 * expert]), float(coefficients[2 * expert + 1])
        post[base:base + intermediate] = gate
        post[base + intermediate:base + 2 * intermediate] = cosine * up + sine * down
        post[base + 2 * intermediate:base + 3 * intermediate] = -sine * up + cosine * down
    return post


def build_observations(np: Any, core: Any, coordinate: Mapping[str, Any], source_post: Any, frozen: Any, *, rht_device: str, selected_experts: Sequence[int], with_target: bool = True) -> tuple[Any, ...]:
    selected = set(int(value) for value in selected_experts)
    output = []
    for block in coordinate["blocks"]:
        owners = set(block.owners)
        require(owners <= selected or owners.isdisjoint(selected), "source component cuts block")
        if owners.isdisjoint(selected):
            continue
        target = None
        if with_target:
            values = np.asarray(source_post[list(block.group_ordinals)], dtype=np.float64).reshape(-1)
            require(values.size == np.asarray(block.indices).size, "source/coordinate alignment")
            transformed, _rms = frozen.forward_signed_rht_and_rms(values, int(block.rht_seed_u64), rht_device)
            target = np.asarray(transformed, dtype=np.float64) / float(block.decoder_scale)
        output.append(core.Observation(
            ordinal=int(block.ordinal),
            owners=tuple(int(value) for value in block.owners),
            indices=np.asarray(block.indices, dtype=np.int16),
            target_normalized=target,
            occupancy=np.asarray(block.occupancy, dtype=np.float64),
            coordinate_mapping_sha256=block.coordinate_mapping_sha256,
        ))
    require(output, "observations nonempty")
    return tuple(output)


def decoder_observations(np: Any, core: Any, coordinate: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(core.Observation(
        ordinal=int(block.ordinal),
        owners=tuple(int(value) for value in block.owners),
        indices=np.asarray(block.indices, dtype=np.int16),
        target_normalized=None,
        occupancy=np.asarray(block.occupancy, dtype=np.float64),
        coordinate_mapping_sha256=block.coordinate_mapping_sha256,
    ) for block in coordinate["blocks"])


def inverse_role_subset(np: Any, post: Any, metadata: Mapping[str, Any], selected_experts: Sequence[int], *, experts: int, intermediate: int) -> dict[tuple[int, str], Any]:
    coefficients = struct.unpack_from(f"<{2 * experts}f", bytes(metadata["header"]), 32)
    output = {}
    for expert in sorted(int(value) for value in selected_experts):
        base = expert * 3 * intermediate
        gate = np.asarray(post[base:base + intermediate], dtype=np.float64)
        z0 = np.asarray(post[base + intermediate:base + 2 * intermediate], dtype=np.float64)
        z1 = np.asarray(post[base + 2 * intermediate:base + 3 * intermediate], dtype=np.float64)
        cosine, sine = float(coefficients[2 * expert]), float(coefficients[2 * expert + 1])
        norm2 = cosine * cosine + sine * sine
        require(math.isfinite(norm2) and norm2 > 0.0, "inverse role transform")
        output[(expert, "gate")] = gate
        output[(expert, "up")] = (cosine * z0 - sine * z1) / norm2
        output[(expert, "down")] = ((sine * z0 + cosine * z1) / norm2).T
    return output


def reconstruct_subset(np: Any, core: Any, coordinate: Mapping[str, Any], observations: Sequence[Any], parameters: Any | None, *, law: int | None, frozen: Any, strata: Any, metadata: Mapping[str, Any], rht_device: str, selected_experts: Sequence[int], identity: bool = False) -> dict[tuple[int, str], Any]:
    selected = tuple(sorted(int(value) for value in selected_experts))
    selected_set = set(selected)
    observation_by_ordinal = {block.ordinal: block for block in observations}
    hidden, intermediate, experts = int(strata.GROUP_VALUES), int(strata.GROUPS_PER_MATRIX), int(coordinate["experts"])
    post = np.empty((experts * 3 * intermediate, hidden), dtype=np.float64)
    covered = set()
    for decoded in coordinate["blocks"]:
        owners = set(decoded.owners)
        require(owners <= selected_set or owners.isdisjoint(selected_set), "reconstruction cuts stream")
        if owners.isdisjoint(selected_set):
            continue
        if identity:
            reconstructed = np.asarray(decoded.reconstructed, dtype=np.float64)
        else:
            normalized = core.predict_normalized(np, observation_by_ordinal[int(decoded.ordinal)], parameters, law=int(law), states=int(coordinate["states"]))
            transformed = normalized * float(decoded.decoder_scale)
            reconstructed = np.asarray(frozen.inverse_signed_rht(transformed, int(decoded.rht_seed_u64), rht_device), dtype=np.float64)
        rows = reconstructed.reshape(len(decoded.group_ordinals), hidden)
        for local, group in enumerate(decoded.group_ordinals):
            require(group not in covered, "reconstruction duplicate group")
            post[group] = rows[local]
            covered.add(group)
    expected = {group for expert in selected for group in range(expert * 3 * intermediate, (expert + 1) * 3 * intermediate)}
    require(covered == expected, "reconstruction group coverage")
    return inverse_role_subset(np, post, metadata, selected, experts=experts, intermediate=intermediate)


def score_subset(np: Any, source_subset: Mapping[str, Any], candidate: Mapping[tuple[int, str], Any], selected_experts: Sequence[int]) -> dict[str, Any]:
    sse = 0.0
    energy = 0.0
    rows = []
    selected = tuple(sorted(int(value) for value in selected_experts))
    for expert in selected:
        for role in ("gate", "up", "down"):
            target = np.asarray(source_subset["matrices"][(expert, role)], dtype=np.float64)
            reconstruction = np.asarray(candidate[(expert, role)], dtype=np.float64)
            require(target.shape == reconstruction.shape, "score shape")
            residual = target - reconstruction
            matrix_sse = float(np.sum(residual * residual, dtype=np.float64))
            matrix_energy = float(np.sum(target * target, dtype=np.float64))
            require(math.isfinite(matrix_sse) and math.isfinite(matrix_energy) and matrix_energy > 0.0, "score finite")
            sse += matrix_sse
            energy += matrix_energy
            rows.append({"expert_ordinal": expert, "role": role, "sse_fp64": matrix_sse, "source_energy_fp64": matrix_energy, "relative_mse": matrix_sse / matrix_energy})
    require(energy > 0.0, "score energy")
    return {"experts": list(selected), "matrices": rows, "sse_fp64": sse, "source_energy_fp64": energy, "relative_mse": sse / energy}


def inner_physical_metrics(np: Any, modules: Mapping[str, Any], inner_path: Path, inner: bytes, inner_sha256: str, *, rht_device: str) -> dict[str, Any]:
    codec, common, semantic = modules["codec"], modules["common"], modules["semantic"]
    parsed = codec.parse_container(common, semantic, inner)
    require(codec.canonical_rebuild(common, semantic, parsed) == inner, "inner canonical rebuild")
    adapter = modules["adapter_source"].StrataSCAdapter(
        common=common,
        semantic_codec=semantic,
        np=np,
        frozen_auditor=modules["frozen"],
        strata_common=modules["strata"],
        device=rht_device,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(inner_path, flags)
    source = None
    try:
        source = codec.AuthenticatedDescriptorSource(descriptor, inner_sha256)
        metrics = codec.physical_metrics(
            common,
            semantic,
            parsed,
            routed_descriptor_source=source,
            externally_authenticated_container_sha256=inner_sha256,
            routed_decoder=adapter.new_routed_decoder(),
        )
        source.verify_stable()
    finally:
        if source is not None:
            source.close()
        os.close(descriptor)
    require(int(metrics["actual_container_bytes"]) == len(inner), "inner metrics literal bytes")
    require(metrics.get("routed_io_authoritative_descriptor_backed") is True, "descriptor-backed inner routed proof")
    return metrics


def _baseline_component_rate(core: Any, metrics: Mapping[str, Any], weights_by_expert: Sequence[int], component: Sequence[int]) -> float:
    rows = metrics["experts"]
    attributed = sum((core.fraction_from_record(rows[int(expert)]["attributable_total_physical_bytes"], "baseline attributed") for expert in component), Fraction(0, 1))
    weights = sum(int(weights_by_expert[int(expert)]) for expert in component)
    require(weights > 0, "baseline component weights")
    return float(Fraction(8, weights) * attributed)


def _score_parameters(np: Any, core: Any, coordinate: Mapping[str, Any], decoder_rows: Sequence[Any], parameters: Any, law: int, source_subset: Mapping[str, Any], component: Sequence[int], modules: Mapping[str, Any], metadata: Mapping[str, Any], rht_device: str) -> dict[str, Any]:
    rounded = np.asarray(parameters, dtype=np.float64).astype("<f2").astype(np.float64)
    require(bool(np.all(np.isfinite(rounded))), "binary16 score parameters")
    reconstruction = reconstruct_subset(
        np,
        core,
        coordinate,
        decoder_rows,
        rounded,
        law=law,
        frozen=modules["frozen"],
        strata=modules["strata"],
        metadata=metadata,
        rht_device=rht_device,
        selected_experts=component,
    )
    return score_subset(np, source_subset, reconstruction, component)


def select_ridge(np: Any, core: Any, observations: Sequence[Any], components: Sequence[Sequence[int]], *, outer: int, law: int, states: int, source_subset: Mapping[str, Any], coordinate: Mapping[str, Any], decoder_rows: Sequence[Any], modules: Mapping[str, Any], metadata: Mapping[str, Any], rht_device: str) -> dict[str, Any]:
    require(len(components) == 3 and 0 <= outer < 3, "nested component geometry")
    development = tuple(value for value in range(3) if value != outer)
    grid = []
    for exponent in core.RIDGE_EXPONENTS:
        directions = []
        summed = 0.0
        for train_component, validation_component in (development, development[::-1]):
            parameters = core.fit_head(np, core.component_blocks(observations, components, (train_component,)), law=law, states=states, ridge_exponent=exponent)
            score = _score_parameters(np, core, coordinate, decoder_rows, parameters, law, source_subset, components[validation_component], modules, metadata, rht_device)
            sse = float(score["sse_fp64"])
            require(math.isfinite(sse) and sse >= 0.0, "inner validation SSE")
            summed += sse
            directions.append({"train_component": train_component, "validation_component": validation_component, "validation_sse_fp64": sse})
        grid.append({"ridge_exponent": exponent, "summed_bidirectional_validation_sse_fp64": summed, "directions": directions})
    winner = min(grid, key=lambda row: (row["summed_bidirectional_validation_sse_fp64"], row["ridge_exponent"]))
    refit = core.fit_head(np, core.component_blocks(observations, components, development), law=law, states=states, ridge_exponent=int(winner["ridge_exponent"]))
    return {
        "outer_component": outer,
        "development_components": list(development),
        "law": law,
        "law_name": core.LAW_NAMES[law],
        "ridge_grid": grid,
        "selected_ridge_exponent": int(winner["ridge_exponent"]),
        "selected_by_bidirectional_inner_validation_only": True,
        "refit_parameters": refit,
    }


def _ledger_without_unverifiable_proof(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    clean_rows = []
    for row in result["experts"]:
        clean = dict(row)
        clean.pop("actual_inner_routed_decode_proof_sha256", None)
        clean.pop("independent_projected_outer_read_ranges", None)
        clean_rows.append(clean)
    result["experts"] = clean_rows
    return result


def validate_producer_result(core: Any, producer: Mapping[str, Any], recomputed: Mapping[str, Any], *, bridge_publication_sha256: str, predecessor_inner_sha256: str, package_closure: Mapping[str, Any], decoder_hashes: Mapping[str, Any]) -> None:
    required_top = {
        "schema", "status", "positive_claim_authority", "selected_sc_decisions_treated_as_scalar_bins",
        "coordinate_aligned_lattice_indices_redecoded_from_literal", "predecessor_publication_sha256",
        "predecessor_inner_sha256", "posterior_handoff_root_sha256", "source_manifest_sha256",
        "source_record_set_sha256", "source_identity_fields_available_to_decoder", "decoder_source_hashes",
        "routed_read_proof", "posterior_package_closure", "rht_device", "components", "baseline_full",
        "folds", "pooled_crossfit", "pooled_crossfit_is_one_literal_packet",
        "crossfit_passes_state_specific_gate_on_every_component", "final_literal_passes_rate_F_and_cold_read",
        "overall_nonpromoting_source_survivor", "final_all_component_candidate", "controls_run",
        "matched_gaussian_controls_run", "structure_destroying_controls_run", "portability_family_run",
        "claim_boundary", "elapsed_seconds",
    }
    exact_fields(producer, required_top, "producer RESULT")
    require(producer["schema"] == POSTERIOR_RESULT_SCHEMA, "producer RESULT schema")
    require(producer["positive_claim_authority"] is False, "producer nonpromotion")
    for field in ("controls_run", "matched_gaussian_controls_run", "structure_destroying_controls_run", "portability_family_run"):
        require(producer[field] is False, f"producer {field}")
    require(producer["selected_sc_decisions_treated_as_scalar_bins"] is False, "producer SC semantic boundary")
    require(producer["coordinate_aligned_lattice_indices_redecoded_from_literal"] is True, "producer coordinate redecode")
    require(producer["source_identity_fields_available_to_decoder"] is False, "producer identity-free decoder")
    require(producer["predecessor_publication_sha256"] == bridge_publication_sha256, "producer predecessor publication binding")
    require(producer["predecessor_inner_sha256"] == predecessor_inner_sha256, "producer predecessor inner binding")
    require(producer["posterior_package_closure"]["manifest_sha256"] == package_closure["manifest_sha256"], "producer package closure binding")
    core.require_deep_close(producer["decoder_source_hashes"], decoder_hashes, "producer decoder hashes")
    for key in (
        "status", "posterior_handoff_root_sha256", "source_manifest_sha256", "source_record_set_sha256",
        "components", "baseline_full", "pooled_crossfit", "crossfit_passes_state_specific_gate_on_every_component",
        "final_literal_passes_rate_F_and_cold_read", "overall_nonpromoting_source_survivor",
    ):
        core.require_deep_close(producer[key], recomputed[key], f"producer recomputed {key}")
    require(producer["pooled_crossfit_is_one_literal_packet"] is False, "pooled crossfit packet boundary")
    observed_folds = producer["folds"]
    expected_folds = recomputed["folds"]
    require(isinstance(observed_folds, list) and len(observed_folds) == len(expected_folds) == 3, "producer fold count")
    for ordinal, (observed, expected) in enumerate(zip(observed_folds, expected_folds, strict=True)):
        for key in ("outer_component", "heldout_experts", "baseline", "G_state_bpw", "passes_positive_Delta_s", "passes_positive_G_state", "passes_rate_interval", "passes_F_target", "passes_cold_read_below_2x", "passes_all_fold_gates", "passes_positive_state_specific_gate"):
            core.require_deep_close(observed[key], expected[key], f"fold[{ordinal}].{key}")
        require(set(observed["laws"]) == set(expected["laws"]), f"fold[{ordinal}] laws")
        for law_name in expected["laws"]:
            left, right = observed["laws"][law_name], expected["laws"][law_name]
            for key in ("nested_selection", "head", "wrapper_bytes", "wrapper_sha256", "owner_allocated_rate_bpw", "score", "F_from_owner_allocated_rate", "Delta_s_from_owner_ledger", "owner_ledger_is_literal_heldout_packet"):
                core.require_deep_close(left[key], right[key], f"fold[{ordinal}].{law_name}.{key}")
            core.require_deep_close(_ledger_without_unverifiable_proof(left["physical_ledger"]), _ledger_without_unverifiable_proof(right["physical_ledger"]), f"fold[{ordinal}].{law_name}.ledger")
    observed_final, expected_final = producer["final_all_component_candidate"], recomputed["final_all_component_candidate"]
    if expected_final is None:
        require(observed_final is None, "producer absent final candidate")
    else:
        require(isinstance(observed_final, Mapping), "producer final candidate")
        for key in ("selected_ridge_exponent_by_outer_median", "head", "literal_wrapper_bytes", "literal_wrapper_sha256", "physical_rate_bpw", "score", "F", "passes_rate_interval", "passes_F_target", "passes_cold_read_below_2x", "training_panel_result_not_portability_evidence"):
            core.require_deep_close(observed_final[key], expected_final[key], f"final.{key}")
        core.require_deep_close(_ledger_without_unverifiable_proof(observed_final["physical_ledger"]), _ledger_without_unverifiable_proof(expected_final["physical_ledger"]), "final.ledger")
    routed = producer["routed_read_proof"]
    require(isinstance(routed, Mapping), "producer routed proof")
    require(routed.get("actual_inner_routed_decode_executed") is True, "producer inner routed decode")
    require(routed.get("actual_posterior_wrapper_routed_decode_executed") is False, "producer posterior routed boundary")
    require(routed.get("posterior_head_applied_to_routed_reconstruction") is False, "producer posterior application boundary")
    require(routed.get("compressed_expert_second_pass_forbidden_and_absent") is True, "producer second-pass boundary")
    require(routed.get("nonpromoting_inference_read_projection_only") is True, "producer read projection boundary")
    require(producer["rht_device"] == "cupy", "producer exact CuPy RHT result")
    require(producer["claim_boundary"] == "Qwen-panel discovery diagnostic only; no universal SwiGLU-MoE performance claim", "producer claim boundary")


def recompute_result(np: Any, core: Any, bridge: Any, modules: Mapping[str, Any], coordinate: Mapping[str, Any], source_descriptor: Mapping[str, Any], posterior: Mapping[str, Any], inner_metrics: Mapping[str, Any], *, rht_device: str) -> dict[str, Any]:
    strata = modules["strata"]
    hidden, intermediate, experts = int(strata.GROUP_VALUES), int(strata.GROUPS_PER_MATRIX), int(coordinate["experts"])
    metadata = coordinate["metadata"]
    states = int(coordinate["states"])
    components = core.owner_components(experts, [block.owners for block in coordinate["blocks"]])
    require(len(components) == 3, "posterior requires three owner components")
    weights_by_expert = tuple(3 * intermediate * hidden for _ in range(experts))
    require(sum(weights_by_expert) == int(coordinate["weights"]), "source weight geometry")
    feature_rows = decoder_observations(np, core, coordinate)

    wrappers = {}
    for outer in range(3):
        for law in (core.LAW_LOCAL, core.LAW_STATE, core.LAW_PERMUTED):
            name = f"FOLD{outer}_{core.LAW_MEMBER_NAMES[law]}.cagepst1"
            raw = posterior["payloads"][name]
            parsed = core.parse_wrapper(np, raw, expected_handoff_root_sha256=coordinate["handoff_root_sha256"])
            require(parsed["inner"] == posterior["predecessor_inner"], f"{name} unchanged inner")
            require(parsed["weights"] == coordinate["weights"] and parsed["experts"] == experts and parsed["fold_ordinal"] == outer, f"{name} dimensions")
            require(parsed["parsed_head"]["law"] == law and parsed["parsed_head"]["states"] == states, f"{name} law/states")
            wrappers[(outer, law)] = parsed
    final_wrapper = None
    if POSTERIOR_FINAL_MEMBER in posterior["payloads"]:
        final_wrapper = core.parse_wrapper(np, posterior["payloads"][POSTERIOR_FINAL_MEMBER], expected_handoff_root_sha256=coordinate["handoff_root_sha256"])
        require(final_wrapper["inner"] == posterior["predecessor_inner"] and final_wrapper["fold_ordinal"] == -1, "final wrapper inner/fold")
        require(final_wrapper["parsed_head"]["law"] == core.LAW_STATE and final_wrapper["parsed_head"]["states"] == states, "final wrapper law/states")

    baseline_rates = {ordinal: _baseline_component_rate(core, inner_metrics, weights_by_expert, component) for ordinal, component in enumerate(components)}
    folds = []
    pooled = {law: {"sse": 0.0, "energy": 0.0} for law in (core.LAW_LOCAL, core.LAW_STATE, core.LAW_PERMUTED)}
    baseline_parts = {}
    source_member_receipts = {}
    for outer in range(3):
        development_ordinals = tuple(value for value in range(3) if value != outer)
        development_experts = tuple(sorted(expert for component_ordinal in development_ordinals for expert in components[component_ordinal]))
        development_source = load_source_subset(np, source_descriptor, development_experts)
        for row in development_source["member_receipts"]:
            source_member_receipts[(row["expert_ordinal"], row["role"])] = row
        post = source_post_coordinates(np, development_source, metadata, experts=experts, intermediate=intermediate, hidden=hidden)
        observations = build_observations(np, core, coordinate, post, modules["frozen"], rht_device=rht_device, selected_experts=development_experts)
        pending = {}
        for law in (core.LAW_LOCAL, core.LAW_STATE, core.LAW_PERMUTED):
            selected = select_ridge(np, core, observations, components, outer=outer, law=law, states=states, source_subset=development_source, coordinate=coordinate, decoder_rows=feature_rows, modules=modules, metadata=metadata, rht_device=rht_device)
            parsed = wrappers[(outer, law)]
            expected_head = core.serialize_head(np, selected["refit_parameters"], law=law, states=states, ridge_exponent=selected["selected_ridge_exponent"], handoff_root_sha256=coordinate["handoff_root_sha256"])
            require(expected_head == parsed["head"], f"fold[{outer}] {core.LAW_NAMES[law]} independently refitted binary16 head")
            ledger = core.wrapper_ledger(inner_metrics=inner_metrics, wrapper=parsed, weights_by_expert=weights_by_expert)
            pending[law] = {"selected": selected, "parsed": parsed, "ledger": ledger}
        del observations, post, development_source

        heldout = load_source_subset(np, source_descriptor, components[outer])
        for row in heldout["member_receipts"]:
            source_member_receipts[(row["expert_ordinal"], row["role"])] = row
        identity = reconstruct_subset(np, core, coordinate, feature_rows, None, law=None, frozen=modules["frozen"], strata=strata, metadata=metadata, rht_device=rht_device, selected_experts=components[outer], identity=True)
        baseline = score_subset(np, heldout, identity, components[outer])
        baseline_parts[outer] = baseline
        laws = {}
        for law in (core.LAW_LOCAL, core.LAW_STATE, core.LAW_PERMUTED):
            item = pending[law]
            parsed, selected, ledger = item["parsed"], item["selected"], item["ledger"]
            score = _score_parameters(np, core, coordinate, feature_rows, parsed["parsed_head"]["parameters"], law, heldout, components[outer], modules, metadata, rht_device)
            rate = core.allocated_component_rate(ledger, weights_by_expert, components[outer])
            ds = core.delta_s(baseline_rate=baseline_rates[outer], candidate_rate=rate, baseline_distortion=baseline["relative_mse"], candidate_distortion=score["relative_mse"])
            f_value = score["relative_mse"] * math.pow(2.0, 2.0 * rate)
            public_selection = {key: value for key, value in selected.items() if key != "refit_parameters"}
            public_head = {key: value for key, value in parsed["parsed_head"].items() if key != "parameters"}
            laws[core.LAW_NAMES[law]] = {
                "nested_selection": public_selection,
                "head": public_head,
                "wrapper_bytes": parsed["total_bytes"],
                "wrapper_sha256": parsed["wrapper_sha256"],
                "owner_allocated_rate_bpw": rate,
                "score": score,
                "F_from_owner_allocated_rate": f_value,
                "Delta_s_from_owner_ledger": ds,
                "physical_ledger": ledger,
                "owner_ledger_is_literal_heldout_packet": False,
            }
            pooled[law]["sse"] += score["sse_fp64"]
            pooled[law]["energy"] += score["source_energy_fp64"]
        state_ds = laws["state-aware"]["Delta_s_from_owner_ledger"]
        g_state = state_ds - max(laws["local-only"]["Delta_s_from_owner_ledger"], laws["state-permuted"]["Delta_s_from_owner_ledger"])
        state = laws["state-aware"]
        gates = core.fold_gate(delta_s_value=state_ds, g_state_value=g_state, candidate_rate_bpw=state["owner_allocated_rate_bpw"], candidate_f=state["F_from_owner_allocated_rate"], cold_read_below_2x=state["physical_ledger"]["passes_strict_cold_read_below_2x"])
        folds.append({
            "outer_component": outer,
            "heldout_experts": list(components[outer]),
            "baseline": {"owner_allocated_rate_bpw": baseline_rates[outer], "score": baseline},
            "laws": laws,
            "G_state_bpw": g_state,
            **gates,
            "passes_positive_state_specific_gate": gates["passes_all_fold_gates"],
        })
        del heldout, identity

    require(len(source_member_receipts) == 3 * experts, "all source members authenticated during replay")
    crossfit_pass = all(row["passes_positive_state_specific_gate"] for row in folds)
    pooled_rows = {core.LAW_NAMES[law]: {"heldout_sse_sum_fp64": row["sse"], "heldout_energy_sum_fp64": row["energy"], "pooled_relative_mse": row["sse"] / row["energy"]} for law, row in pooled.items()}
    baseline_sse = sum(row["sse_fp64"] for row in baseline_parts.values())
    baseline_energy = sum(row["source_energy_fp64"] for row in baseline_parts.values())
    baseline_full = {
        "experts": list(range(experts)),
        "matrices": [matrix for ordinal in range(3) for matrix in baseline_parts[ordinal]["matrices"]],
        "sse_fp64": baseline_sse,
        "source_energy_fp64": baseline_energy,
        "relative_mse": baseline_sse / baseline_energy,
        "assembled_from_three_disjoint_heldout_apertures": True,
    }

    require((final_wrapper is not None) is crossfit_pass, "final wrapper presence follows crossfit decision")
    final_record = None
    if crossfit_pass:
        exponents = sorted(int(row["laws"]["state-aware"]["nested_selection"]["selected_ridge_exponent"]) for row in folds)
        exponent = exponents[1]
        final_source = load_source_subset(np, source_descriptor, range(experts))
        post = source_post_coordinates(np, final_source, metadata, experts=experts, intermediate=intermediate, hidden=hidden)
        observations = build_observations(np, core, coordinate, post, modules["frozen"], rht_device=rht_device, selected_experts=range(experts))
        parameters = core.fit_head(np, observations, law=core.LAW_STATE, states=states, ridge_exponent=exponent)
        expected_head = core.serialize_head(np, parameters, law=core.LAW_STATE, states=states, ridge_exponent=exponent, handoff_root_sha256=coordinate["handoff_root_sha256"])
        require(expected_head == final_wrapper["head"], "final independently refitted binary16 head")
        reconstruction = reconstruct_subset(np, core, coordinate, feature_rows, final_wrapper["parsed_head"]["parameters"], law=core.LAW_STATE, frozen=modules["frozen"], strata=strata, metadata=metadata, rht_device=rht_device, selected_experts=range(experts))
        score = score_subset(np, final_source, reconstruction, range(experts))
        ledger = core.wrapper_ledger(inner_metrics=inner_metrics, wrapper=final_wrapper, weights_by_expert=weights_by_expert)
        rate = float(ledger["physical_rate_bpw"]["float"])
        f_value = score["relative_mse"] * math.pow(2.0, 2.0 * rate)
        final_record = {
            "selected_ridge_exponent_by_outer_median": exponent,
            "head": {key: value for key, value in final_wrapper["parsed_head"].items() if key != "parameters"},
            "literal_wrapper_bytes": final_wrapper["total_bytes"],
            "literal_wrapper_sha256": final_wrapper["wrapper_sha256"],
            "physical_rate_bpw": rate,
            "score": score,
            "F": f_value,
            "physical_ledger": ledger,
            "passes_rate_interval": 2.15 <= rate <= 2.5,
            "passes_F_target": f_value <= 0.8,
            "passes_cold_read_below_2x": ledger["passes_strict_cold_read_below_2x"],
            "training_panel_result_not_portability_evidence": True,
        }
        del final_source, post, observations, reconstruction
    final_pass = bool(final_record is not None and final_record["passes_rate_interval"] and final_record["passes_F_target"] and final_record["passes_cold_read_below_2x"])
    overall = crossfit_pass and final_pass
    status = "CROSS_FIT_SOURCE_SURVIVOR_NONPROMOTING_CONTROLS_AND_PORTABILITY_REQUIRED" if overall else ("HARD_KILL_FINAL_LITERAL_RATE_F_OR_COLD_READ" if crossfit_pass else "HARD_KILL_POSTERIOR_STATE_SPECIFIC_CROSS_FIT")
    return {
        "status": status,
        "posterior_handoff_root_sha256": coordinate["handoff_root_sha256"],
        "source_manifest_sha256": source_descriptor["manifest_sha256"],
        "source_record_set_sha256": source_descriptor["source_record_set_sha256"],
        "components": [list(component) for component in components],
        "baseline_full": baseline_full,
        "folds": folds,
        "pooled_crossfit": pooled_rows,
        "crossfit_passes_state_specific_gate_on_every_component": crossfit_pass,
        "final_literal_passes_rate_F_and_cold_read": final_pass,
        "overall_nonpromoting_source_survivor": overall,
        "final_all_component_candidate": final_record,
        "source_member_receipts": [source_member_receipts[key] for key in sorted(source_member_receipts, key=lambda item: (item[0], ("gate", "up", "down").index(item[1])))],
    }


def write_publication(path: Path, members: Mapping[str, bytes], *, status: str) -> dict[str, Any]:
    require(os.name == "posix", "audit publication requires POSIX durability primitives")
    require(path.is_absolute() and not path.exists(), "fresh absolute audit output")
    parent = path.parent
    metadata = os.lstat(parent)
    require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), "audit output parent")
    os.mkdir(path, 0o700)
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        rows = []
        for name in sorted(members):
            safe_name(name, "audit output member")
            payload = members[name]
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=directory_fd)
            try:
                cursor = 0
                while cursor < len(payload):
                    cursor += os.write(descriptor, payload[cursor:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            rows.append({"name": name, "bytes": len(payload), "sha256": sha256(payload)})
        os.fsync(directory_fd)
        completion = {
            "schema": AUDIT_COMPLETION_SCHEMA,
            "status": status,
            "positive_claim_authority": False,
            "members": rows,
        }
        completion["completion_sha256"] = sha256(canonical_json(completion))
        payload = pretty_json(completion)
        descriptor = os.open("COMPLETE.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=directory_fd)
        try:
            cursor = 0
            while cursor < len(payload):
                cursor += os.write(descriptor, payload[cursor:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {"output_dir": os.fspath(path), "completion_sha256": sha256(payload), "members": rows}


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    require(arguments.authorization == AUTHORIZATION, "explicit result-audit authorization")
    require(os.name == "posix", "result audit requires POSIX")
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode, "invoke with CPython -I -B")
    require(arguments.rht_device == "cupy", "exact result replay requires CuPy RHT")
    digest(arguments.expected_auditor_source_manifest_sha256, "auditor source manifest")
    digest(arguments.expected_external_pins_sha256, "external pins")
    started = time.perf_counter()

    own_package = Path(__file__).resolve().parent
    own = authenticate_source_package(own_package, arguments.expected_auditor_source_manifest_sha256, own=True)
    pins_path = absolute_path(arguments.external_pins, "external pins", directory=False)
    pins_payload = regular_bytes(pins_path, maximum=MAX_JSON, label="external pins", expected_sha256=arguments.expected_external_pins_sha256)
    require(pins_payload == pretty_json(strict_json(pins_payload, "external pins")), "canonical external pins encoding")
    pins = parse_pins(strict_json(pins_payload, "external pins"))
    paths, hashes = pins["paths"], pins["hashes"]

    producer_path = absolute_path(paths["posterior_producer_package"], "posterior producer package", directory=True)
    producer = authenticate_source_package(producer_path, hashes["posterior_producer_manifest_sha256"], own=False)
    v9_path = absolute_path(paths["v9_publication"], "v9 publication", directory=True)
    posterior_path = absolute_path(paths["posterior_publication"], "posterior publication", directory=True)
    v9 = authenticate_publication(v9_path, pins["v9_rows"], "v9 publication")
    posterior = authenticate_publication(posterior_path, pins["posterior_rows"], "posterior publication")
    require(v9["complete"].get("schema") == "uwfa-sc-v9-qwen-primary-completion-v0", "v9 completion schema")
    require(v9["complete"].get("positive_claim_authority") is False, "v9 completion nonpromotion")
    require(posterior["complete"].get("schema") == POSTERIOR_COMPLETION_SCHEMA, "posterior completion schema")
    require(posterior["complete"].get("positive_claim_authority") is False, "posterior completion nonpromotion")

    v9_audit_path = absolute_path(paths["v9_result_audit_receipt"], "v9 audit receipt", directory=False)
    v9_audit_payload = regular_bytes(v9_audit_path, maximum=MAX_JSON, label="v9 audit receipt", expected_sha256=hashes["v9_result_audit_receipt_sha256"])
    v9_audit = authenticate_v9_audit_receipt(v9_audit_payload, v9, hashes["v9_result_audit_source_manifest_sha256"])
    source_manifest_path = absolute_path(paths["source_manifest"], "source manifest", directory=False)
    source_manifest_payload = regular_bytes(source_manifest_path, maximum=MAX_SOURCE_MANIFEST, label="source manifest", expected_sha256=hashes["source_manifest_sha256"])

    # All publications, executable closures and the source-manifest authority
    # are authenticated before the numerical backend or CuPy path can load.
    core = load_module("uwfa_pc_v0_result_audit_core", own["sources"]["audit_core.py"], sha256(own["sources"]["audit_core.py"]))
    bridge = load_module("uwfa_pc_v0_result_audit_bridge", producer["sources"]["result_bridge.py"], KNOWN_RESULT_BRIDGE_SHA256)
    publication = bridge.authenticate_result_directory(v9_path)
    require(publication["inner"] == v9["payloads"]["UWFCV8.bin"], "bridge/v9 retained inner binding")
    v9_result = publication["result"]
    require(v9_result.get("schema") == V9_RESULT_SCHEMA, "v9 result schema")
    require(v9_result.get("positive_claim_authority") is False, "v9 result nonpromotion")
    require(v9_result.get("artifact_identity", {}).get("sha256") == v9_audit["source_closure"]["artifact_sha256"], "v9 artifact/audit receipt binding")

    v8_path = absolute_path(paths["v8_package"], "v8 package", directory=True)
    strata_path = absolute_path(paths["strata_common"], "STRATA common", directory=False)
    frozen_path = absolute_path(paths["frozen_auditor"], "frozen auditor", directory=False)
    require(sha256(regular_bytes(strata_path, maximum=MAX_JSON, label="STRATA common")) == KNOWN_STRATA_SHA256, "STRATA exact pin")
    require(sha256(regular_bytes(frozen_path, maximum=MAX_JSON, label="frozen auditor")) == KNOWN_FROZEN_SHA256, "frozen auditor exact pin")
    v8 = bridge.authenticate_v8_package(v8_path, expected_manifest_sha256=KNOWN_V8_MANIFEST_SHA256)

    import numpy as np

    modules = bridge.load_authenticated_decoders(v9_result, v8, strata_common_path=strata_path, frozen_auditor_path=frozen_path)
    coordinate = bridge.decode_coordinate_panel(np, modules, publication["inner"], posterior_core=core, rht_device=arguments.rht_device)
    require(coordinate["coordinate_aligned_observations_redecoded_from_literal"] is True, "coordinate literal redecode")
    require(coordinate["selected_sc_decisions_treated_as_scalar_bins"] is False, "SC decision semantic boundary")
    source_descriptor = authenticate_source_manifest(
        core,
        source_manifest_payload,
        source_manifest_path,
        artifact_sha256=v9_result["artifact_identity"]["sha256"],
        experts=int(coordinate["experts"]),
        intermediate=int(modules["strata"].GROUPS_PER_MATRIX),
        hidden=int(modules["strata"].GROUP_VALUES),
    )
    metrics = inner_physical_metrics(np, modules, v9_path / "UWFCV8.bin", publication["inner"], sha256(publication["inner"]), rht_device=arguments.rht_device)
    posterior["predecessor_inner"] = publication["inner"]
    recomputed = recompute_result(np, core, bridge, modules, coordinate, source_descriptor, posterior, metrics, rht_device=arguments.rht_device)

    # Bind the independently rescored baseline to the predecessor result whose
    # entire publication was already accepted by the separately frozen v9
    # result auditor.  No posterior RESULT value is used as an input.
    core.require_float_close(recomputed["baseline_full"]["source_energy_fp64"], v9_result["baseline_score"]["source_energy_fp64"], "baseline source energy/v9 audited result", rel=2.0 ** -45, abs_=1e-10)
    core.require_float_close(recomputed["baseline_full"]["relative_mse"], v9_result["physical"]["relative_mse"], "baseline MSE/v9 audited result", rel=2.0 ** -43, abs_=1e-13)

    producer_result = strict_json(posterior["payloads"]["RESULT.json"], "posterior RESULT")
    validate_producer_result(
        core,
        producer_result,
        recomputed,
        bridge_publication_sha256=publication["publication_sha256"],
        predecessor_inner_sha256=sha256(publication["inner"]),
        package_closure=producer,
        decoder_hashes=modules["source_hashes"],
    )
    require(posterior["complete"].get("status") == recomputed["status"] == producer_result["status"], "posterior completion terminal status")

    input_manifest = {
        "schema": "uwfa-sc-posterior-centroid-v0-result-audit-input-manifest-v0",
        "external_pins_sha256": sha256(pins_payload),
        "auditor_source_manifest_sha256": own["manifest_sha256"],
        "auditor_source_snapshot_root_sha256": own["source_snapshot_root_sha256"],
        "posterior_producer_manifest_sha256": producer["manifest_sha256"],
        "posterior_producer_source_snapshot_root_sha256": producer["source_snapshot_root_sha256"],
        "v9_result_audit_receipt_sha256": sha256(v9_audit_payload),
        "v9_result_audit_source_manifest_sha256": hashes["v9_result_audit_source_manifest_sha256"],
        "v9_publication_members": v9["rows"],
        "posterior_publication_members": posterior["rows"],
        "source_manifest_sha256": source_descriptor["manifest_sha256"],
        "source_record_set_sha256": source_descriptor["source_record_set_sha256"],
        "authenticated_source_members": recomputed["source_member_receipts"],
        "decoder_source_hashes": modules["source_hashes"],
        "predecessor_inner_sha256": sha256(publication["inner"]),
        "posterior_handoff_root_sha256": coordinate["handoff_root_sha256"],
    }
    result_record = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS_EXACT_NONPROMOTING_POSTERIOR_RESULT_AUDIT",
        "positive_claim_authority": False,
        "producer_terminal_status": recomputed["status"],
        "producer_result_json_used_as_numerical_input": False,
        "crossfit_refit_recomputed": True,
        "local_state_permuted_heldout_scores_recomputed": True,
        "every_emitted_head_reproduced_binary16_exactly": True,
        "literal_wrapper_grammar_hash_rate_F_recomputed": True,
        "read_result_is_nonpromoting_projection": True,
        "actual_posterior_wrapper_routed_decode_executed": False,
        "posterior_head_applied_inside_routed_session": False,
        "inference_ready_routed_posterior_application_exists_in_v0": False,
        "matched_gaussian_controls_run_by_this_audit": False,
        "portability_family_run_by_this_audit": False,
        "universal_swiglu_moe_performance_authority": False,
        "recomputed": {key: value for key, value in recomputed.items() if key != "source_member_receipts"},
        "input_manifest_sha256": sha256(canonical_json(input_manifest)),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": "A real posterior diagnostic result has been independently replayed; v0 still lacks inference-ready routed posterior application and supplies no universal or control evidence.",
    }
    limits = {
        "schema": "uwfa-sc-posterior-centroid-v0-result-audit-limits-v0",
        "inference_ready_routed_posterior_decoder": False,
        "posterior_head_applied_during_one_pass_routed_expert_decode": False,
        "cold_read_numbers": "nonpromoting projection from an authenticated inner routed decode plus one literal suffix-page request",
        "crossfit_fold_packets_are_deployable_single_codec": False,
        "final_all_component_packet_is_training_panel_portability_evidence": False,
        "matched_gaussian_controls_run": False,
        "structure_destroying_controls_run": False,
        "disjoint_swiglu_moe_family_run": False,
        "scope": "exact integrity and numerical replay of one externally pinned posterior-centroid v0 diagnostic only",
    }
    output_parent = absolute_path(paths["audit_output_parent"], "audit output parent", directory=True)
    output_dir = output_parent / safe_name(paths["audit_output_name"], "audit output name")
    publication_record = write_publication(output_dir, {
        "AUDIT_RESULT.json": pretty_json(result_record),
        "INPUT_MANIFEST.json": pretty_json(input_manifest),
        "LIMITS.json": pretty_json(limits),
    }, status=result_record["status"])
    return {
        "schema": "uwfa-sc-posterior-centroid-v0-result-audit-launch-summary-v0",
        "status": result_record["status"],
        "positive_claim_authority": False,
        "producer_terminal_status": recomputed["status"],
        "audit_output_dir": publication_record["output_dir"],
        "audit_completion_sha256": publication_record["completion_sha256"],
        "inference_ready_routed_posterior_application_exists_in_v0": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--expected-auditor-source-manifest-sha256", required=True)
    result.add_argument("--external-pins", required=True)
    result.add_argument("--expected-external-pins-sha256", required=True)
    result.add_argument("--rht-device", default="cupy", choices=("cupy",))
    return result


def main() -> int:
    summary = run(parser().parse_args())
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL_UWFA_SC_POSTERIOR_CENTROID_V0_RESULT_AUDIT: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
