#!/usr/bin/env python3
"""Auditor-owned authority boundary for the abstract four-level LOGIC-Q codec.

The production entry points in this module never accept scored rows, aggregate
metrics, packet-geometry receipts, or a caller-selected config.  They receive
literal raw source bytes, invoke the pinned encoder in a fresh isolated child,
parse and reconstruct complete packets, and derive all evidence internally.

This remains an abstract four-level research codec.  It is not a STRATA-RM6
adapter and grants no model/Qwen authority by itself.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


V2_MANIFEST_SHA256 = (
    "e97041b2debdd1a85ce32305f43aae1f76cf4ca937b52e275bdd246ae1b1b980")
V2_SOURCE_ROOT_SHA256 = (
    "080de7a63e596ae34f9da90941d7fd9d07b70dfb2afad97103aa5ab5943d3776")
V1_SOURCE_ROOT_SHA256 = (
    "5d145d89a20d2ae256ea60f569fab97cd6372cde66f7df75f3e86b08b3a88560")
ROLE_ORDER = ("gate", "up", "down_transposed")
ROLE_ORDINAL = {role: ordinal for ordinal, role in enumerate(ROLE_ORDER)}
SOURCE_DTYPES = {"bf16-le": 2, "float32-le": 4, "float64-le": 8}
RATE_MIN = 2.15
RATE_MAX = 2.5
TARGET_F = 0.8
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
AUTH_MAGIC = b"LQ3AUTH\0"
AUTH_VERSION = 0
AUTH_PREFIX = struct.Struct("<8sHII")
AUTH_TRAILER = struct.Struct("<I")
AUTH_PAGE = 4096
PROBE_ELEMENTS = 4096
PROBE_EXPECTED = sum((index * 17 + 3) % 65521
                     for index in range(PROBE_ELEMENTS))


class AuthorityError(RuntimeError):
    """Fail-closed source, packet, selection, or backend authority error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def canonical_copy(value: Any) -> Any:
    return json.loads(canonical_json(value).decode("ascii"))


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
                AuthorityError(f"{label} nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"{label} strict JSON") from exc
    require(isinstance(value, dict), f"{label} object")
    return value


def regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                f"{label} regular non-link")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise AuthorityError(f"{label} read") from exc
    require((before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino),
            f"{label} changed during read")
    return payload


def verify_source_dependency(package: Path, *, expected_manifest_sha256: str,
                             expected_source_root_sha256: str) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir(), "dependency directory")
    manifest_payload = regular_bytes(root / "SOURCE_MANIFEST.json",
                                     "dependency manifest")
    require(sha256(manifest_payload) == expected_manifest_sha256,
            "dependency external manifest pin")
    manifest = strict_json(manifest_payload, "dependency manifest")
    require(manifest.get("source_root_sha256") == expected_source_root_sha256,
            "dependency external source-root pin")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows, "dependency members")
    observed = []
    names = []
    for row in rows:
        require(isinstance(row, dict) and
                set(row) == {"name", "bytes", "sha256"},
                "dependency member schema")
        name = row["name"]
        require(isinstance(name, str) and name and name not in names and
                name != "SOURCE_MANIFEST.json" and "/" not in name and
                "\\" not in name, "dependency member name")
        payload = regular_bytes(root / name, f"dependency member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"dependency member pin {name}")
        observed.append(item)
        names.append(name)
    root_payload = json.dumps(observed, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode("ascii")
    require(sha256(root_payload) == expected_source_root_sha256,
            "dependency observed source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} ==
            set(names) | {"SOURCE_MANIFEST.json"} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "dependency exact regular closure")
    return {"manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": expected_source_root_sha256,
            "members": tuple(names)}


def load_v2(package: Path) -> Any:
    verify_source_dependency(
        package, expected_manifest_sha256=V2_MANIFEST_SHA256,
        expected_source_root_sha256=V2_SOURCE_ROOT_SHA256)
    name = "logicq_v2_authority_" + V2_SOURCE_ROOT_SHA256[:16]
    if name in sys.modules:
        return sys.modules[name]
    path = package.resolve(strict=True) / "bound_adapter.py"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "v2 import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dependencies(v2_package: Path, v1_package: Path,
                      v0_package: Path) -> tuple[Any, Any, Any]:
    binder = load_v2(v2_package)
    v1 = binder.load_v1(v1_package)
    core = binder.load_v0(v1, v0_package)
    require(binder.V1_SOURCE_ROOT_SHA256 == V1_SOURCE_ROOT_SHA256,
            "v1 source root")
    return binder, v1, core


def source_key(layer: str, slot: str, role: str) -> tuple[str, str, str]:
    require(isinstance(layer, str) and layer and isinstance(slot, str) and slot
            and role in ROLE_ORDINAL, "source key")
    return layer, slot, role


def _panel_rows_by_ordinal(panel: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["component_ordinal"]): dict(row) for row in panel["rows"]}


def make_alias_map(panel: Mapping[str, Any],
                   groups: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    rows = _panel_rows_by_ordinal(panel)
    canonical_groups = []
    seen: set[int] = set()
    for group in groups:
        require(set(group) == {"component_ordinals", "reason"},
                "alias group schema")
        ordinals = sorted(int(value) for value in group["component_ordinals"])
        reason = group["reason"]
        require(len(ordinals) >= 2 and len(ordinals) == len(set(ordinals)) and
                all(value in rows and value not in seen for value in ordinals) and
                isinstance(reason, str) and reason.strip(), "alias group fields")
        hashes = {rows[value]["source_sha256"] for value in ordinals}
        require(len(hashes) == 1, "alias group one source hash")
        seen.update(ordinals)
        canonical_groups.append({"component_ordinals": ordinals,
                                 "source_sha256": next(iter(hashes)),
                                 "reason": reason.strip()})
    canonical_groups.sort(key=lambda row: row["component_ordinals"])
    record = {
        "schema": "logic-q-v3-source-alias-map-v1",
        "groups": canonical_groups,
        "default_duplicate_policy": "REJECT",
    }
    record["alias_map_sha256"] = sha256(canonical_json(record))
    return validate_alias_map(panel, record)


def validate_alias_map(panel: Mapping[str, Any],
                       alias_map: Mapping[str, Any]) -> dict[str, Any]:
    require(set(alias_map) == {"schema", "groups", "default_duplicate_policy",
                               "alias_map_sha256"} and
            alias_map["schema"] == "logic-q-v3-source-alias-map-v1" and
            alias_map["default_duplicate_policy"] == "REJECT",
            "alias map schema")
    base = dict(alias_map)
    claimed = base.pop("alias_map_sha256")
    require(HEX64.fullmatch(str(claimed)) is not None and
            claimed == sha256(canonical_json(base)), "alias map seal")
    rows = _panel_rows_by_ordinal(panel)
    by_hash: dict[str, list[int]] = {}
    for ordinal, row in rows.items():
        by_hash.setdefault(row["source_sha256"], []).append(ordinal)
    expected = {digest: sorted(ordinals) for digest, ordinals in by_hash.items()
                if len(ordinals) > 1}
    observed = {}
    seen = set()
    for group in alias_map["groups"]:
        require(isinstance(group, dict) and
                set(group) == {"component_ordinals", "source_sha256", "reason"},
                "alias group canonical schema")
        ordinals = group["component_ordinals"]
        require(isinstance(ordinals, list) and len(ordinals) >= 2 and
                ordinals == sorted(ordinals) and len(ordinals) == len(set(ordinals))
                and all(isinstance(value, int) and value in rows and value not in seen
                        for value in ordinals), "alias group ordinals")
        digest = group["source_sha256"]
        require(HEX64.fullmatch(str(digest)) is not None and
                all(rows[value]["source_sha256"] == digest for value in ordinals) and
                isinstance(group["reason"], str) and group["reason"].strip(),
                "alias group binding")
        seen.update(ordinals)
        require(digest not in observed, "one group per duplicate hash")
        observed[digest] = list(ordinals)
    require(observed == expected, "complete explicit duplicate alias map")
    return canonical_copy(alias_map)


BACKEND_POLICY_FIELDS = {
    "schema", "cupy_version", "module_file_sha256", "device_name",
    "compute_capability", "runtime_version", "driver_version",
    "probe_elements", "probe_expected",
}


def validate_backend_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    require(set(policy) == BACKEND_POLICY_FIELDS and
            policy["schema"] == "logic-q-v3-fresh-cupy-policy-v1" and
            isinstance(policy["cupy_version"], str) and policy["cupy_version"] and
            HEX64.fullmatch(str(policy["module_file_sha256"])) is not None and
            isinstance(policy["device_name"], str) and policy["device_name"] and
            isinstance(policy["compute_capability"], str) and
            policy["compute_capability"] and
            isinstance(policy["runtime_version"], int) and
            policy["runtime_version"] > 0 and
            isinstance(policy["driver_version"], int) and
            policy["driver_version"] > 0 and
            policy["probe_elements"] == PROBE_ELEMENTS and
            policy["probe_expected"] == PROBE_EXPECTED,
            "fresh CuPy policy")
    return canonical_copy(policy)


def make_precommit(binder: Any, v1: Any, panel: Mapping[str, Any], *,
                   expected_panel_sha256: str,
                   alias_map: Mapping[str, Any],
                   backend_policy: Mapping[str, Any],
                   worker_sha256: str) -> dict[str, Any]:
    validated_panel = binder.validate_panel_record(
        panel, expected_panel_sha256=expected_panel_sha256)
    aliases = validate_alias_map(validated_panel, alias_map)
    policy = validate_backend_policy(backend_policy)
    require(HEX64.fullmatch(worker_sha256) is not None,
            "worker external source hash")
    record = {
        "schema": "logic-q-v3-authority-precommit-v1",
        "panel_sha256": expected_panel_sha256,
        "panel": validated_panel,
        "alias_map": aliases,
        "backend_policy": policy,
        "worker_sha256": worker_sha256,
        "v2_source_root_sha256": V2_SOURCE_ROOT_SHA256,
        "v1_source_root_sha256": V1_SOURCE_ROOT_SHA256,
        "frozen_grid": canonical_copy(v1.frozen_grid_record()),
        "selection_partitions": ["train", "validation"],
        "test_opened": False,
        "scored_rows_or_encoder_metrics_accepted": False,
        "literal_source_and_packet_bytes_required": True,
        "strata_semantics": "NOT_BOUND__ABSTRACT_FOUR_LEVEL_ONLY",
    }
    record["precommit_sha256"] = sha256(canonical_json(record))
    return record


def validate_precommit(binder: Any, v1: Any, precommit: Mapping[str, Any], *,
                       expected_precommit_sha256: str) -> dict[str, Any]:
    require(HEX64.fullmatch(expected_precommit_sha256) is not None,
            "external precommit SHA-256")
    required = {
        "schema", "panel_sha256", "panel", "alias_map", "backend_policy",
        "worker_sha256", "v2_source_root_sha256", "v1_source_root_sha256",
        "frozen_grid", "selection_partitions", "test_opened",
        "scored_rows_or_encoder_metrics_accepted",
        "literal_source_and_packet_bytes_required", "strata_semantics",
        "precommit_sha256",
    }
    require(set(precommit) == required and precommit["schema"] ==
            "logic-q-v3-authority-precommit-v1", "precommit schema")
    base = dict(precommit)
    claimed = base.pop("precommit_sha256")
    require(claimed == sha256(canonical_json(base)) and
            claimed == expected_precommit_sha256, "precommit seal/external pin")
    panel = binder.validate_panel_record(
        precommit["panel"], expected_panel_sha256=precommit["panel_sha256"])
    validate_alias_map(panel, precommit["alias_map"])
    validate_backend_policy(precommit["backend_policy"])
    require(HEX64.fullmatch(str(precommit["worker_sha256"])) is not None and
            precommit["v2_source_root_sha256"] == V2_SOURCE_ROOT_SHA256 and
            precommit["v1_source_root_sha256"] == V1_SOURCE_ROOT_SHA256 and
            precommit["frozen_grid"] == canonical_copy(v1.frozen_grid_record()) and
            precommit["selection_partitions"] == ["train", "validation"] and
            precommit["test_opened"] is False and
            precommit["scored_rows_or_encoder_metrics_accepted"] is False and
            precommit["literal_source_and_packet_bytes_required"] is True and
            precommit["strata_semantics"] ==
            "NOT_BOUND__ABSTRACT_FOUR_LEVEL_ONLY", "precommit immutable fields")
    return canonical_copy(precommit)


AUTH_HEADER_FIELDS = {
    "schema", "mode", "precommit_sha256", "panel_sha256", "config_id",
    "layer", "slot", "rows", "cols", "source_sha256_by_role",
    "inner_packet_bytes", "inner_packet_sha256", "inner_packet_crc32",
    "inner_geometry_sha256", "backend_receipt_sha256",
    "v1_source_root_sha256", "alphabet_size", "strata_compatible",
}


def _pack_authority_bytes(header: Mapping[str, Any], inner_packet: bytes) -> bytes:
    header_bytes = canonical_json(header)
    require(len(header_bytes) <= 0xFFFFFFFF and len(inner_packet) <= 0xFFFFFFFF,
            "authority packet length")
    prefix = AUTH_PREFIX.pack(AUTH_MAGIC, AUTH_VERSION, len(header_bytes),
                              len(inner_packet))
    body = prefix + header_bytes + inner_packet
    trailer = AUTH_TRAILER.pack(zlib.crc32(body) & 0xFFFFFFFF)
    unpadded = body + trailer
    return unpadded + b"\0" * ((-len(unpadded)) % AUTH_PAGE)


def pack_authority_packet(np: Any, binder: Any, v1: Any, core: Any, *,
                          inner_packet: bytes, precommit: Mapping[str, Any],
                          expected_precommit_sha256: str, config_id: str,
                          layer: str, slot: str,
                          source_sha256_by_role: Mapping[str, str],
                          backend_receipt_sha256: str,
                          mode: str = "gpu_fresh") -> bytes:
    validated = validate_precommit(
        binder, v1, precommit,
        expected_precommit_sha256=expected_precommit_sha256)
    require(config_id in {config.config_id for config in v1.FROZEN_CONFIGS},
            "authority packet frozen config")
    require(mode in {"gpu_fresh", "source_free_fixture"},
            "authority packet mode")
    if mode == "gpu_fresh":
        require(HEX64.fullmatch(backend_receipt_sha256) is not None,
                "production backend receipt hash")
    else:
        require(backend_receipt_sha256 == "SOURCE_FREE_FIXTURE",
                "fixture backend marker")
    require(set(source_sha256_by_role) == set(ROLE_ORDER) and
            all(HEX64.fullmatch(str(source_sha256_by_role[role])) is not None
                for role in ROLE_ORDER), "authority source hashes")
    geometry = binder.packet_geometry(np, v1, core, bytes(inner_packet))
    rows, cols = geometry["expert_shape"]
    matches = [row for row in validated["panel"]["rows"]
               if row["layer"] == layer and row["slot"] == slot]
    matches.sort(key=lambda row: ROLE_ORDINAL[row["role"]])
    require([row["role"] for row in matches] == list(ROLE_ORDER) and
            all(row["rows"] == rows and row["cols"] == cols and
                row["source_sha256"] == source_sha256_by_role[row["role"]]
                for row in matches), "authority packet panel/source binding")
    header = {
        "schema": "logic-q-v3-authority-packet-header-v1",
        "mode": mode,
        "precommit_sha256": expected_precommit_sha256,
        "panel_sha256": validated["panel_sha256"],
        "config_id": config_id, "layer": layer, "slot": slot,
        "rows": rows, "cols": cols,
        "source_sha256_by_role": {role: source_sha256_by_role[role]
                                   for role in ROLE_ORDER},
        "inner_packet_bytes": len(inner_packet),
        "inner_packet_sha256": sha256(bytes(inner_packet)),
        "inner_packet_crc32": zlib.crc32(bytes(inner_packet)) & 0xFFFFFFFF,
        "inner_geometry_sha256": geometry["packet_geometry_sha256"],
        "backend_receipt_sha256": backend_receipt_sha256,
        "v1_source_root_sha256": V1_SOURCE_ROOT_SHA256,
        "alphabet_size": 4,
        "strata_compatible": False,
    }
    return _pack_authority_bytes(header, bytes(inner_packet))


def unpack_authority_packet(np: Any, binder: Any, v1: Any, core: Any,
                            packet: bytes, *,
                            precommit: Mapping[str, Any],
                            expected_precommit_sha256: str,
                            expected_mode: str) -> dict[str, Any]:
    validated = validate_precommit(
        binder, v1, precommit,
        expected_precommit_sha256=expected_precommit_sha256)
    payload = bytes(packet)
    require(len(payload) >= AUTH_PAGE and len(payload) % AUTH_PAGE == 0,
            "authority packet page closure")
    require(len(payload) >= AUTH_PREFIX.size + AUTH_TRAILER.size,
            "authority packet minimum")
    magic, version, header_length, inner_length = AUTH_PREFIX.unpack(
        payload[:AUTH_PREFIX.size])
    require(magic == AUTH_MAGIC and version == AUTH_VERSION and
            header_length > 0 and inner_length > 0, "authority packet prefix")
    body_end = AUTH_PREFIX.size + header_length + inner_length
    trailer_end = body_end + AUTH_TRAILER.size
    require(trailer_end <= len(payload) and
            payload[trailer_end:] == b"\0" * (len(payload) - trailer_end),
            "authority packet zero page padding")
    body = payload[:body_end]
    observed_crc = AUTH_TRAILER.unpack(payload[body_end:trailer_end])[0]
    require(observed_crc == (zlib.crc32(body) & 0xFFFFFFFF),
            "authority packet CRC32")
    header = strict_json(payload[AUTH_PREFIX.size:AUTH_PREFIX.size + header_length],
                         "authority header")
    require(set(header) == AUTH_HEADER_FIELDS and header["schema"] ==
            "logic-q-v3-authority-packet-header-v1" and
            header["mode"] == expected_mode and
            header["precommit_sha256"] == expected_precommit_sha256 and
            header["panel_sha256"] == validated["panel_sha256"] and
            header["v1_source_root_sha256"] == V1_SOURCE_ROOT_SHA256 and
            header["alphabet_size"] == 4 and
            header["strata_compatible"] is False,
            "authority header immutable fields")
    require(header["config_id"] in
            {config.config_id for config in v1.FROZEN_CONFIGS},
            "authority header frozen config")
    require(isinstance(header["layer"], str) and header["layer"] and
            isinstance(header["slot"], str) and header["slot"] and
            isinstance(header["rows"], int) and header["rows"] > 0 and
            isinstance(header["cols"], int) and header["cols"] > 0 and
            set(header["source_sha256_by_role"]) == set(ROLE_ORDER) and
            all(HEX64.fullmatch(str(header["source_sha256_by_role"][role]))
                is not None for role in ROLE_ORDER) and
            HEX64.fullmatch(str(header["inner_packet_sha256"])) is not None and
            isinstance(header["inner_packet_crc32"], int) and
            0 <= header["inner_packet_crc32"] <= 0xFFFFFFFF,
            "authority header literal domains")
    if expected_mode == "gpu_fresh":
        require(HEX64.fullmatch(str(header["backend_receipt_sha256"])) is not None,
                "authority production backend receipt")
    else:
        require(header["backend_receipt_sha256"] == "SOURCE_FREE_FIXTURE",
                "authority fixture backend marker")
    inner = payload[AUTH_PREFIX.size + header_length:body_end]
    require(header["inner_packet_bytes"] == len(inner) == inner_length and
            header["inner_packet_sha256"] == sha256(inner) and
            header["inner_packet_crc32"] == (zlib.crc32(inner) & 0xFFFFFFFF),
            "authority inner payload hash/CRC")
    geometry = binder.packet_geometry(np, v1, core, inner)
    require(header["inner_geometry_sha256"] ==
            geometry["packet_geometry_sha256"] and
            [header["rows"], header["cols"]] == geometry["expert_shape"],
            "authority inner geometry")
    matches = [row for row in validated["panel"]["rows"]
               if row["layer"] == header["layer"] and
               row["slot"] == header["slot"]]
    matches.sort(key=lambda row: ROLE_ORDINAL[row["role"]])
    require([row["role"] for row in matches] == list(ROLE_ORDER) and
            all(row["source_sha256"] ==
                header["source_sha256_by_role"][row["role"]]
                for row in matches), "authority header panel sources")
    require(_pack_authority_bytes(header, inner) == payload,
            "authority packet canonical re-encode")
    return {
        "header": header, "inner_packet": inner,
        "inner_geometry": geometry,
        "authority_packet_bytes": len(payload),
        "authority_packet_sha256": sha256(payload),
        "authority_packet_crc32": zlib.crc32(payload[:trailer_end]) & 0xFFFFFFFF,
        "expert_weights_from_inner_headers":
            geometry["expert_weights_from_headers"],
        "physical_bits": len(payload) * 8,
        "physical_rate_bpw": (len(payload) * 8 /
                              geometry["expert_weights_from_headers"]),
        "layout_contiguous_read_bytes": len(payload),
        "layout_addressable_read_amplification": 1.0,
        "runtime_read_amplification_measured": False,
    }


def _actual_numpy() -> Any:
    try:
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise AuthorityError("canonical NumPy import") from exc
    require(getattr(np, "__name__", "") == "numpy" and
            getattr(np, "__spec__", None) is not None and
            np.__spec__.name == "numpy", "canonical NumPy module")
    return np


def decode_source(np: Any, payload: bytes, dtype: str, count: int) -> Any:
    require(isinstance(payload, bytes) and dtype in SOURCE_DTYPES and
            len(payload) == count * SOURCE_DTYPES[dtype], "literal source bytes")
    if dtype == "bf16-le":
        words = np.frombuffer(payload, dtype="<u2", count=count)
        values = (words.astype(np.uint32) << np.uint32(16)).view(
            np.float32).astype(np.float64)
    elif dtype == "float32-le":
        values = np.frombuffer(payload, dtype="<f4", count=count).astype(np.float64)
    else:
        values = np.frombuffer(payload, dtype="<f8", count=count).astype(np.float64)
    require(values.size == count and np.all(np.isfinite(values)),
            "finite decoded source")
    return values


def score_authority_packet(binder: Any, v1: Any, core: Any, *,
                           packet: bytes,
                           source_blobs: Mapping[str, bytes],
                           precommit: Mapping[str, Any],
                           expected_precommit_sha256: str,
                           expected_mode: str = "gpu_fresh") -> dict[str, Any]:
    """Parse complete packet/source bytes and derive every metric internally."""
    np = _actual_numpy()
    parsed = unpack_authority_packet(
        np, binder, v1, core, packet, precommit=precommit,
        expected_precommit_sha256=expected_precommit_sha256,
        expected_mode=expected_mode)
    header = parsed["header"]
    require(set(source_blobs) == set(ROLE_ORDER), "scorer exact source roles")
    decoded = v1.unpack_canonical_expert(np, core, parsed["inner_packet"])
    panel_rows = [row for row in precommit["panel"]["rows"]
                  if row["layer"] == header["layer"] and
                  row["slot"] == header["slot"]]
    panel_rows.sort(key=lambda row: ROLE_ORDINAL[row["role"]])
    rows = []
    pooled_sse = 0.0
    pooled_energy = 0.0
    pooled_count = 0
    for panel_row in panel_rows:
        role = panel_row["role"]
        blob = source_blobs[role]
        require(sha256(blob) == panel_row["source_sha256"] ==
                header["source_sha256_by_role"][role],
                "auditor-owned source hash")
        count = int(panel_row["rows"]) * int(panel_row["cols"])
        source = decode_source(np, blob, panel_row["source_dtype"], count)
        reconstruction = np.asarray(decoded[role][1], dtype=np.float64)
        require(reconstruction.shape == source.shape and
                np.all(np.isfinite(reconstruction)), "finite reconstruction")
        residual = source - reconstruction
        sse = float(np.sum(residual * residual, dtype=np.float64))
        energy = float(np.sum(source * source, dtype=np.float64))
        require(math.isfinite(sse) and sse >= 0.0 and
                math.isfinite(energy) and energy > 0.0,
                "raw score domain")
        pooled_sse += sse
        pooled_energy += energy
        pooled_count += count
        rows.append({
            "component_ordinal": panel_row["component_ordinal"],
            "partition": panel_row["partition"],
            "layer": header["layer"], "slot": header["slot"], "role": role,
            "source_sha256": sha256(blob),
            "source_bytes": len(blob), "source_count": count,
            "raw_sse_f64_hex": sse.hex(),
            "raw_energy_f64_hex": energy.hex(),
        })
    require(pooled_count == parsed["expert_weights_from_inner_headers"],
            "packet-derived pooled source count")
    rate = parsed["physical_bits"] / pooled_count
    relative_mse = pooled_sse / pooled_energy
    result = {
        "schema": "logic-q-v3-auditor-owned-score-v1",
        "precommit_sha256": expected_precommit_sha256,
        "config_id": header["config_id"],
        "layer": header["layer"], "slot": header["slot"],
        "authority_packet_sha256": parsed["authority_packet_sha256"],
        "authority_packet_bytes": parsed["authority_packet_bytes"],
        "inner_packet_sha256": header["inner_packet_sha256"],
        "rows": rows,
        "pooled": {
            "physical_bits": parsed["physical_bits"], "weights": pooled_count,
            "physical_rate_bpw": rate,
            "raw_sse_f64_hex": pooled_sse.hex(),
            "raw_energy_f64_hex": pooled_energy.hex(),
            "relative_mse": relative_mse,
            "F": relative_mse * 2.0 ** (2.0 * rate),
        },
        "metrics_accepted_from_encoder": False,
        "packet_and_source_bytes_parsed": True,
        "layout_addressable_read_amplification": 1.0,
        "runtime_read_amplification_measured": False,
    }
    result["score_sha256"] = sha256(canonical_json(result))
    return result


def backend_receipt_from_cupy(cp: Any, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Collect a fresh-process CuPy/device record and enforce the precommit."""
    pinned = validate_backend_policy(policy)
    module_path = Path(cp.__file__).resolve(strict=True)
    module_hash = sha256(regular_bytes(module_path, "CuPy module file"))
    runtime = cp.cuda.runtime
    device_id = int(runtime.getDevice())
    require(0 <= device_id < int(runtime.getDeviceCount()), "CUDA device id")
    properties = runtime.getDeviceProperties(device_id)
    name_value = properties.get("name", properties.get(b"name"))
    device_name = (name_value.split(b"\0", 1)[0].decode("utf-8")
                   if isinstance(name_value, bytes) else str(name_value))
    device = cp.cuda.Device(device_id)
    probe = cp.arange(PROBE_ELEMENTS, dtype=cp.uint64)
    observed = int(cp.asnumpy(cp.sum(
        (probe * cp.uint64(17) + cp.uint64(3)) % cp.uint64(65521),
        dtype=cp.uint64)))
    cp.cuda.get_current_stream().synchronize()
    observed_policy = {
        "schema": "logic-q-v3-fresh-cupy-policy-v1",
        "cupy_version": str(cp.__version__),
        "module_file_sha256": module_hash,
        "device_name": device_name,
        "compute_capability": str(device.compute_capability),
        "runtime_version": int(runtime.runtimeGetVersion()),
        "driver_version": int(runtime.driverGetVersion()),
        "probe_elements": PROBE_ELEMENTS,
        "probe_expected": PROBE_EXPECTED,
    }
    require(observed_policy == pinned and observed == PROBE_EXPECTED,
            "fresh CuPy backend policy/probe")
    receipt = {
        "schema": "logic-q-v3-fresh-cupy-receipt-v1",
        "policy": pinned,
        "module_file": str(module_path),
        "device_id": device_id,
        "device_count": int(runtime.getDeviceCount()),
        "probe_observed": observed,
        "stream_synchronized": True,
        "fresh_isolated_process_required": True,
    }
    receipt["receipt_sha256"] = sha256(canonical_json(receipt))
    return receipt


WORKER_RECEIPT_FIELDS = {
    "schema", "request_sha256", "precommit_sha256", "config_id", "layer",
    "slot", "rows", "cols", "source_sha256_by_role", "inner_packet_bytes",
    "inner_packet_sha256", "backend_receipt", "worker_sha256",
    "v2_source_root_sha256", "receipt_sha256",
}


def validate_worker_receipt(receipt: Mapping[str, Any], *, request: Mapping[str, Any],
                            packet: bytes, precommit: Mapping[str, Any]) -> dict[str, Any]:
    require(set(receipt) == WORKER_RECEIPT_FIELDS and receipt["schema"] ==
            "logic-q-v3-fresh-worker-receipt-v1", "worker receipt schema")
    base = dict(receipt)
    claimed = base.pop("receipt_sha256")
    require(HEX64.fullmatch(str(claimed)) is not None and
            claimed == sha256(canonical_json(base)), "worker receipt seal")
    require(receipt["request_sha256"] == sha256(canonical_json(request)) and
            receipt["precommit_sha256"] == precommit["precommit_sha256"] and
            receipt["config_id"] == request["config_id"] and
            receipt["layer"] == request["layer"] and
            receipt["slot"] == request["slot"] and
            receipt["rows"] == request["rows"] and
            receipt["cols"] == request["cols"] and
            receipt["source_sha256_by_role"] == request["source_sha256_by_role"] and
            receipt["inner_packet_bytes"] == len(packet) and
            receipt["inner_packet_sha256"] == sha256(packet) and
            receipt["worker_sha256"] == precommit["worker_sha256"] and
            receipt["v2_source_root_sha256"] == V2_SOURCE_ROOT_SHA256,
            "worker request/packet/source bindings")
    backend = receipt["backend_receipt"]
    require(isinstance(backend, dict) and backend.get("policy") ==
            precommit["backend_policy"] and backend.get("probe_observed") ==
            PROBE_EXPECTED and backend.get("stream_synchronized") is True and
            backend.get("fresh_isolated_process_required") is True,
            "worker fresh backend binding")
    backend_base = dict(backend)
    backend_claimed = backend_base.pop("receipt_sha256", None)
    require(backend_claimed == sha256(canonical_json(backend_base)),
            "worker backend receipt seal")
    return canonical_copy(receipt)


def _safe_worker_environment() -> dict[str, str]:
    allowed = ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME",
               "CUDA_PATH")
    result = {name: os.environ[name] for name in allowed if name in os.environ}
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONHASHSEED"] = "0"
    return result


def fresh_worker_command(worker: Path, request_path: Path, packet_path: Path,
                         receipt_path: Path) -> list[str]:
    return [sys.executable, "-I", "-B", str(worker),
            "--request", str(request_path),
            "--packet-output", str(packet_path),
            "--receipt-output", str(receipt_path)]


def run_fresh_gpu_worker(binder: Any, v1: Any, core: Any, *,
                         precommit: Mapping[str, Any],
                         expected_precommit_sha256: str,
                         config_id: str, layer: str, slot: str,
                         source_blobs: Mapping[str, bytes],
                         worker_path: Path,
                         timeout_seconds: int = 1800) -> tuple[bytes, dict[str, Any]]:
    """Run the encoder in a fresh `python -I -B` child; no backend is injected."""
    validated = validate_precommit(
        binder, v1, precommit,
        expected_precommit_sha256=expected_precommit_sha256)
    worker = worker_path.resolve(strict=True)
    require(sha256(regular_bytes(worker, "fresh GPU worker")) ==
            validated["worker_sha256"], "fresh worker external source pin")
    require(config_id in {config.config_id for config in v1.FROZEN_CONFIGS},
            "fresh worker frozen config")
    panel_rows = [row for row in validated["panel"]["rows"]
                  if row["layer"] == layer and row["slot"] == slot]
    panel_rows.sort(key=lambda row: ROLE_ORDINAL[row["role"]])
    require([row["role"] for row in panel_rows] == list(ROLE_ORDER) and
            set(source_blobs) == set(ROLE_ORDER), "fresh worker expert triplet")
    source_hashes = {}
    for row in panel_rows:
        blob = source_blobs[row["role"]]
        require(isinstance(blob, bytes) and sha256(blob) == row["source_sha256"],
                "fresh worker auditor-owned source")
        source_hashes[row["role"]] = sha256(blob)
    rows = int(panel_rows[0]["rows"])
    cols = int(panel_rows[0]["cols"])
    request = {
        "schema": "logic-q-v3-fresh-worker-request-v1",
        "precommit_sha256": expected_precommit_sha256,
        "backend_policy": validated["backend_policy"],
        "config_id": config_id, "layer": layer, "slot": slot,
        "rows": rows, "cols": cols,
        "source_dtype": panel_rows[0]["source_dtype"],
        "source_sha256_by_role": source_hashes,
        "source_files_by_role": {role: f"source-{ordinal}.bin"
                                  for ordinal, role in enumerate(ROLE_ORDER)},
        "worker_sha256": validated["worker_sha256"],
        "v2_source_root_sha256": V2_SOURCE_ROOT_SHA256,
    }
    with tempfile.TemporaryDirectory(prefix="logicq-v3-fresh-") as directory:
        root = Path(directory).resolve(strict=True)
        request_path = root / "request.json"
        packet_path = root / "packet.bin"
        receipt_path = root / "receipt.json"
        request_path.write_bytes(canonical_json(request))
        for role, filename in request["source_files_by_role"].items():
            (root / filename).write_bytes(source_blobs[role])
        command = fresh_worker_command(worker, request_path, packet_path,
                                       receipt_path)
        completed = subprocess.run(
            command, cwd=root, env=_safe_worker_environment(),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout_seconds, check=False)
        require(completed.returncode == 0,
                "fresh GPU worker failed: " +
                completed.stderr.decode("utf-8", errors="replace")[-2000:])
        packet = regular_bytes(packet_path, "fresh worker packet")
        receipt = strict_json(regular_bytes(receipt_path, "fresh worker receipt"),
                              "fresh worker receipt")
    validate_worker_receipt(receipt, request=request, packet=packet,
                            precommit=validated)
    binder.packet_geometry(_actual_numpy(), v1, core, packet)
    return packet, receipt


def _source_triplet_for(precommit: Mapping[str, Any],
                        source_blobs: Mapping[tuple[str, str, str], bytes],
                        layer: str, slot: str) -> dict[str, bytes]:
    result = {}
    for role in ROLE_ORDER:
        key = source_key(layer, slot, role)
        require(key in source_blobs and isinstance(source_blobs[key], bytes),
                "auditor source triplet present")
        result[role] = source_blobs[key]
    return result


def _expected_selection_experts(precommit: Mapping[str, Any]) -> list[tuple[str, str]]:
    return sorted({(row["layer"], row["slot"])
                   for row in precommit["panel"]["rows"]
                   if row["partition"] in {"train", "validation"}})


def _assert_selection_source_domain(
        precommit: Mapping[str, Any],
        source_blobs: Mapping[tuple[str, str, str], bytes]) -> None:
    expected = {source_key(row["layer"], row["slot"], row["role"])
                for row in precommit["panel"]["rows"]
                if row["partition"] in {"train", "validation"}}
    require(set(source_blobs) == expected, "exact train/validation source domain")
    for row in precommit["panel"]["rows"]:
        key = source_key(row["layer"], row["slot"], row["role"])
        if key in expected:
            blob = source_blobs[key]
            require(isinstance(blob, bytes) and sha256(blob) ==
                    row["source_sha256"], "precommitted source bytes")


def _derive_metrics(scores: Sequence[Mapping[str, Any]],
                    config_ids: Sequence[str]) -> dict[str, Any]:
    output = {}
    for config_id in sorted(config_ids):
        config_scores = [score for score in scores
                         if score["config_id"] == config_id]
        partitions = {}
        for partition in ("train", "validation"):
            selected = [score for score in config_scores
                        if score["rows"][0]["partition"] == partition]
            require(selected, "nonempty selection partition")
            bits = sum(int(score["pooled"]["physical_bits"]) for score in selected)
            weights = sum(int(score["pooled"]["weights"]) for score in selected)
            sse = sum(float.fromhex(score["pooled"]["raw_sse_f64_hex"])
                      for score in selected)
            energy = sum(float.fromhex(score["pooled"]["raw_energy_f64_hex"])
                         for score in selected)
            require(weights > 0 and energy > 0.0, "selection aggregate domain")
            rate = bits / weights
            relative = sse / energy
            partitions[partition] = {
                "physical_bits": bits, "weights": weights,
                "expert_count": len(selected),
                "raw_sse_f64_hex": sse.hex(),
                "raw_energy_f64_hex": energy.hex(),
                "physical_rate_bpw": rate, "relative_mse": relative,
                "F": relative * 2.0 ** (2.0 * rate),
            }
        output[config_id] = partitions
    return output


def _selected_config(derived: Mapping[str, Any]) -> tuple[str, list[Any]]:
    def key(config_id: str) -> tuple[Any, ...]:
        row = derived[config_id]["validation"]
        rate = float(row["physical_rate_bpw"])
        return (not (RATE_MIN <= rate <= RATE_MAX), float(row["F"]),
                rate, config_id)
    selected = min(derived, key=key)
    return selected, list(key(selected))


def run_selection_authority(binder: Any, v1: Any, core: Any, *,
                            precommit: Mapping[str, Any],
                            expected_precommit_sha256: str,
                            source_blobs: Mapping[tuple[str, str, str], bytes],
                            worker_path: Path) -> dict[str, Any]:
    """Create selection from raw bytes; no row/metric/packet input exists."""
    validated = validate_precommit(
        binder, v1, precommit,
        expected_precommit_sha256=expected_precommit_sha256)
    _assert_selection_source_domain(validated, source_blobs)
    entries = []
    scores = []
    for config in sorted(v1.FROZEN_CONFIGS, key=lambda value: value.config_id):
        for layer, slot in _expected_selection_experts(validated):
            sources = _source_triplet_for(validated, source_blobs, layer, slot)
            inner, worker_receipt = run_fresh_gpu_worker(
                binder, v1, core, precommit=validated,
                expected_precommit_sha256=expected_precommit_sha256,
                config_id=config.config_id, layer=layer, slot=slot,
                source_blobs=sources, worker_path=worker_path)
            source_hashes = {role: sha256(sources[role]) for role in ROLE_ORDER}
            outer = pack_authority_packet(
                _actual_numpy(), binder, v1, core, inner_packet=inner,
                precommit=validated,
                expected_precommit_sha256=expected_precommit_sha256,
                config_id=config.config_id, layer=layer, slot=slot,
                source_sha256_by_role=source_hashes,
                backend_receipt_sha256=worker_receipt["backend_receipt"][
                    "receipt_sha256"], mode="gpu_fresh")
            score = score_authority_packet(
                binder, v1, core, packet=outer, source_blobs=sources,
                precommit=validated,
                expected_precommit_sha256=expected_precommit_sha256,
                expected_mode="gpu_fresh")
            entries.append({
                "config_id": config.config_id, "layer": layer, "slot": slot,
                "authority_packet_base64": base64.b64encode(outer).decode("ascii"),
                "authority_packet_sha256": sha256(outer),
                "worker_receipt": worker_receipt,
            })
            scores.append(score)
    config_ids = [config.config_id for config in v1.FROZEN_CONFIGS]
    derived = _derive_metrics(scores, config_ids)
    selected, key = _selected_config(derived)
    artifact = {
        "schema": "logic-q-v3-content-selection-artifact-v1",
        "precommit_sha256": expected_precommit_sha256,
        "entries": entries,
        "derived_metrics": derived,
        "selected_config_id": selected,
        "selection_key": key,
        "test_source_bytes_opened": False,
        "rows_metrics_or_packet_receipts_accepted": False,
        "all_packet_bytes_embedded": True,
    }
    artifact["selection_artifact_sha256"] = sha256(canonical_json(artifact))
    return artifact


def _decode_artifact_packet(value: Any) -> bytes:
    require(isinstance(value, str) and value, "artifact packet base64")
    try:
        payload = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise AuthorityError("artifact packet canonical base64") from exc
    require(base64.b64encode(payload).decode("ascii") == value,
            "artifact packet canonical base64")
    return payload


def authorize_selection(binder: Any, v1: Any, core: Any, *,
                        precommit: Mapping[str, Any],
                        expected_precommit_sha256: str,
                        artifact: Mapping[str, Any],
                        expected_selection_artifact_sha256: str,
                        source_blobs: Mapping[tuple[str, str, str], bytes],
                        worker_path: Path,
                        replay_fresh_workers: bool = True) -> Any:
    """Re-score bytes and, by default, replay every fresh GPU encode exactly."""
    validated = validate_precommit(
        binder, v1, precommit,
        expected_precommit_sha256=expected_precommit_sha256)
    require(HEX64.fullmatch(expected_selection_artifact_sha256) is not None,
            "external selection artifact SHA-256")
    required = {"schema", "precommit_sha256", "entries", "derived_metrics",
                "selected_config_id", "selection_key", "test_source_bytes_opened",
                "rows_metrics_or_packet_receipts_accepted",
                "all_packet_bytes_embedded", "selection_artifact_sha256"}
    require(set(artifact) == required and artifact["schema"] ==
            "logic-q-v3-content-selection-artifact-v1", "selection artifact schema")
    base = dict(artifact)
    claimed = base.pop("selection_artifact_sha256")
    require(claimed == sha256(canonical_json(base)) and
            claimed == expected_selection_artifact_sha256 and
            artifact["precommit_sha256"] == expected_precommit_sha256 and
            artifact["test_source_bytes_opened"] is False and
            artifact["rows_metrics_or_packet_receipts_accepted"] is False and
            artifact["all_packet_bytes_embedded"] is True,
            "selection artifact seal/immutable fields")
    _assert_selection_source_domain(validated, source_blobs)
    expected_keys = {(config.config_id, layer, slot)
                     for config in v1.FROZEN_CONFIGS
                     for layer, slot in _expected_selection_experts(validated)}
    observed_keys = set()
    scores = []
    for entry in artifact["entries"]:
        require(isinstance(entry, dict) and set(entry) == {
            "config_id", "layer", "slot", "authority_packet_base64",
            "authority_packet_sha256", "worker_receipt"}, "artifact entry schema")
        key = (entry["config_id"], entry["layer"], entry["slot"])
        require(key in expected_keys and key not in observed_keys,
                "artifact exact unique entry")
        observed_keys.add(key)
        outer = _decode_artifact_packet(entry["authority_packet_base64"])
        require(entry["authority_packet_sha256"] == sha256(outer),
                "artifact literal packet hash")
        sources = _source_triplet_for(validated, source_blobs,
                                      entry["layer"], entry["slot"])
        parsed = unpack_authority_packet(
            _actual_numpy(), binder, v1, core, outer, precommit=validated,
            expected_precommit_sha256=expected_precommit_sha256,
            expected_mode="gpu_fresh")
        require(parsed["header"]["config_id"] == entry["config_id"] and
                parsed["header"]["layer"] == entry["layer"] and
                parsed["header"]["slot"] == entry["slot"] and
                parsed["header"]["backend_receipt_sha256"] ==
                entry["worker_receipt"]["backend_receipt"]["receipt_sha256"],
                "artifact packet identity")
        if replay_fresh_workers:
            replay_packet, replay_receipt = run_fresh_gpu_worker(
                binder, v1, core, precommit=validated,
                expected_precommit_sha256=expected_precommit_sha256,
                config_id=entry["config_id"], layer=entry["layer"],
                slot=entry["slot"], source_blobs=sources,
                worker_path=worker_path)
            require(replay_packet == parsed["inner_packet"] and
                    replay_receipt == entry["worker_receipt"],
                    "fresh deterministic worker replay")
        else:
            raise AuthorityError("production authorization requires fresh worker replay")
        scores.append(score_authority_packet(
            binder, v1, core, packet=outer, source_blobs=sources,
            precommit=validated,
            expected_precommit_sha256=expected_precommit_sha256,
            expected_mode="gpu_fresh"))
    require(observed_keys == expected_keys, "complete artifact entry set")
    config_ids = [config.config_id for config in v1.FROZEN_CONFIGS]
    derived = _derive_metrics(scores, config_ids)
    require(derived == artifact["derived_metrics"],
            "selection metrics independently recomputed")
    selected, key = _selected_config(derived)
    require(selected == artifact["selected_config_id"] and
            key == artifact["selection_key"], "selection winner recomputed")
    rate = float(derived[selected]["validation"]["physical_rate_bpw"])
    require(RATE_MIN <= rate <= RATE_MAX, "selected validation rate interval")
    matches = [config for config in v1.FROZEN_CONFIGS
               if config.config_id == selected]
    require(len(matches) == 1, "one selected frozen config")
    return matches[0]


def run_selected_expert(binder: Any, v1: Any, core: Any, *,
                        precommit: Mapping[str, Any],
                        expected_precommit_sha256: str,
                        artifact: Mapping[str, Any],
                        expected_selection_artifact_sha256: str,
                        selection_source_blobs:
                            Mapping[tuple[str, str, str], bytes],
                        test_layer: str, test_slot: str,
                        test_source_blobs: Mapping[str, bytes],
                        worker_path: Path) -> dict[str, Any]:
    """Launch only the config re-derived from the pinned selection artifact."""
    selected = authorize_selection(
        binder, v1, core, precommit=precommit,
        expected_precommit_sha256=expected_precommit_sha256,
        artifact=artifact,
        expected_selection_artifact_sha256=expected_selection_artifact_sha256,
        source_blobs=selection_source_blobs, worker_path=worker_path,
        replay_fresh_workers=True)
    panel_rows = [row for row in precommit["panel"]["rows"]
                  if row["layer"] == test_layer and row["slot"] == test_slot]
    require(panel_rows and all(row["partition"] == "test" for row in panel_rows),
            "selected launch whole test expert")
    inner, worker_receipt = run_fresh_gpu_worker(
        binder, v1, core, precommit=precommit,
        expected_precommit_sha256=expected_precommit_sha256,
        config_id=selected.config_id, layer=test_layer, slot=test_slot,
        source_blobs=test_source_blobs, worker_path=worker_path)
    source_hashes = {role: sha256(test_source_blobs[role]) for role in ROLE_ORDER}
    outer = pack_authority_packet(
        _actual_numpy(), binder, v1, core, inner_packet=inner,
        precommit=precommit,
        expected_precommit_sha256=expected_precommit_sha256,
        config_id=selected.config_id, layer=test_layer, slot=test_slot,
        source_sha256_by_role=source_hashes,
        backend_receipt_sha256=worker_receipt["backend_receipt"]["receipt_sha256"],
        mode="gpu_fresh")
    score = score_authority_packet(
        binder, v1, core, packet=outer, source_blobs=test_source_blobs,
        precommit=precommit,
        expected_precommit_sha256=expected_precommit_sha256,
        expected_mode="gpu_fresh")
    return {"selected_config_id": selected.config_id,
            "authority_packet": outer, "score": score,
            "selection_artifact_sha256": expected_selection_artifact_sha256,
            "caller_selected_config_accepted": False}
