#!/usr/bin/env python3
"""Production-binding successor for the source-only LOGIC-Q v1 mechanism.

This module does not open model payloads and never imports CuPy at import time.
It authenticates the frozen v1 mechanics and closes selector, packet-accounting,
and live-backend bindings without changing the capped search algorithms.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import stat
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


V1_MANIFEST_SHA256 = "9bfd3d1225fb45a0518d2d4d6a4035262e87dc62563222e42e69665358b9aac5"
V1_SOURCE_ROOT_SHA256 = "5d145d89a20d2ae256ea60f569fab97cd6372cde66f7df75f3e86b08b3a88560"
V0_MANIFEST_SHA256 = "31edbc3325dfdae2b3f43cce4afb360062d5c70583b57dd1e6530835a178cced"
V0_SOURCE_ROOT_SHA256 = "2177f2aec39a65afddbbded9b6b3cd2c2a33118c060a41e070102f9fb6c95d4a"
V1_AUDIT_MANIFEST_SHA256 = "6a0e97d987a3288126632db29756681c2ee7c16e809d2a8466db16b22a78dfe1"
V1_AUDIT_ROOT_SHA256 = "d56f36015413694c45dd81b571b05974ef5541cf66e6099c4d2b518c75f1c63b"
ROLE_ORDER = ("gate", "up", "down_transposed")
ROLE_ORDINAL = {role: index for index, role in enumerate(ROLE_ORDER)}
SOURCE_DTYPES = {"bf16-le": 2, "float32-le": 4, "float64-le": 8}
TARGET_F = 0.8
RATE_MIN = 2.15
RATE_MAX = 2.5
PANEL_DOMAIN = b"logic-q-v2-bound-panel-split\0"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class BindingError(RuntimeError):
    """Fail-closed dependency, selector, scorer, packet, or launch error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def canonical_copy(value: Any) -> Any:
    """Return the literal JSON representation used by every portable receipt."""
    return json.loads(canonical_json(value).decode("ascii"))


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{label} duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=hook,
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               BindingError(f"{label} nonfinite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BindingError(f"{label} strict JSON") from exc
    require(isinstance(value, dict), f"{label} object")
    return value


def _regular_bytes(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),
                f"{label} regular non-link")
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise BindingError(f"{label} read") from exc
    require((before.st_size, before.st_mtime_ns, before.st_mode, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_mode, after.st_ino),
            f"{label} changed during read")
    return payload


def verify_source_dependency(package: Path, *, expected_manifest_sha256: str,
                             expected_source_root_sha256: str) -> dict[str, Any]:
    root = package.resolve(strict=True)
    require(root.is_dir(), "dependency directory")
    manifest_payload = _regular_bytes(root / "SOURCE_MANIFEST.json",
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
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "dependency member schema")
        name = row["name"]
        require(isinstance(name, str) and name and name not in names and
                name != "SOURCE_MANIFEST.json" and "/" not in name and "\\" not in name,
                "dependency safe member")
        payload = _regular_bytes(root / name, f"dependency member {name}")
        item = {"name": name, "bytes": len(payload), "sha256": sha256(payload)}
        require(item == row, f"dependency member pin {name}")
        observed.append(item)
        names.append(name)
    root_payload = json.dumps(observed, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False).encode("ascii")
    require(sha256(root_payload) == expected_source_root_sha256,
            "dependency observed source root")
    entries = list(os.scandir(root))
    require({entry.name for entry in entries} == set(names) | {"SOURCE_MANIFEST.json"}
            and all(entry.is_file(follow_symlinks=False) for entry in entries),
            "dependency exact regular closure")
    return {"manifest_sha256": expected_manifest_sha256,
            "source_root_sha256": expected_source_root_sha256,
            "members": tuple(names)}


def load_v1(package: Path) -> Any:
    verify_source_dependency(
        package, expected_manifest_sha256=V1_MANIFEST_SHA256,
        expected_source_root_sha256=V1_SOURCE_ROOT_SHA256)
    name = "logicq_v1_bound_" + V1_SOURCE_ROOT_SHA256[:16]
    if name in sys.modules:
        return sys.modules[name]
    path = package.resolve(strict=True) / "capped_adapter.py"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "v1 import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_v0(v1: Any, package: Path) -> Any:
    receipt = verify_source_dependency(
        package, expected_manifest_sha256=V0_MANIFEST_SHA256,
        expected_source_root_sha256=V0_SOURCE_ROOT_SHA256)
    core = v1.load_parent_core(package)
    require(receipt["source_root_sha256"] == V0_SOURCE_ROOT_SHA256,
            "v0 load receipt")
    return core


@dataclass(frozen=True)
class PanelRow:
    layer: str
    slot: str
    role: str
    rows: int
    cols: int
    source_sha256: str
    source_dtype: str


def _public_hash(kind: str, value: str) -> bytes:
    return hashlib.sha256(PANEL_DOMAIN + kind.encode("ascii") + b"\0" +
                          value.encode("utf-8")).digest()


def make_panel_record(rows: Iterable[PanelRow], *, test_layer_count: int = 5,
                      validation_slot_count: int | None = None) -> dict[str, Any]:
    panel = tuple(rows)
    require(panel, "nonempty panel")
    require(all(row.layer and row.slot and row.role in ROLE_ORDINAL and
                row.rows > 0 and row.cols > 0 and
                HEX64.fullmatch(row.source_sha256) is not None and
                row.source_dtype in SOURCE_DTYPES for row in panel),
            "panel fields")
    ordered = tuple(sorted(panel, key=lambda row: (
        row.layer.encode("utf-8"), row.slot.encode("utf-8"),
        ROLE_ORDINAL[row.role])))
    keys = [(row.layer, row.slot, row.role) for row in ordered]
    require(len(keys) == len(set(keys)), "panel unique components")
    layers = sorted({row.layer for row in ordered})
    slots = sorted({row.slot for row in ordered})
    require(len(layers) >= 10 and 5 <= test_layer_count < len(layers),
            "panel whole test layers")
    if validation_slot_count is None:
        validation_slot_count = max(1, (len(slots) + 3) // 4)
    require(1 <= validation_slot_count < len(slots), "panel validation slots")
    for layer in layers:
        subset = [row for row in ordered if row.layer == layer]
        require(sorted({row.slot for row in subset}) == slots,
                "identical slot universe")
        for slot in slots:
            triplet = [row for row in subset if row.slot == slot]
            require([row.role for row in triplet] == list(ROLE_ORDER),
                    "canonical role triplet")
            require(len({(row.rows, row.cols) for row in triplet}) == 1,
                    "SwiGLU role shape equality")
    shapes = {(row.rows, row.cols) for row in ordered}
    dtypes = {row.source_dtype for row in ordered}
    require(len(shapes) == 1 and len(dtypes) == 1, "one panel shape/dtype cohort")
    ranked_layers = sorted(layers,
                           key=lambda value: (_public_hash("layer", value), value))
    ranked_slots = sorted(slots,
                          key=lambda value: (_public_hash("slot", value), value))
    test_layers = set(ranked_layers[:test_layer_count])
    validation_slots = set(ranked_slots[:validation_slot_count])
    row_records = []
    counts = {"train": 0, "validation": 0, "test": 0}
    for ordinal, row in enumerate(ordered):
        partition = ("test" if row.layer in test_layers else
                     "validation" if row.slot in validation_slots else "train")
        counts[partition] += 1
        item = asdict(row)
        item.update({"component_ordinal": ordinal, "partition": partition})
        row_records.append(item)
    require(all(counts.values()), "nonempty panel partitions")
    base = {
        "schema": "logic-q-v2-bound-panel-v1",
        "rows": row_records,
        "test_layer_count": test_layer_count,
        "validation_slot_count": validation_slot_count,
        "test_layers": sorted(test_layers),
        "validation_slots": sorted(validation_slots),
        "partition_component_counts": counts,
        "split_uses_source_hash": False,
        "canonical_source_dtype": next(iter(dtypes)),
    }
    base["panel_sha256"] = sha256(canonical_json(base))
    return base


def validate_panel_record(panel: Mapping[str, Any], *,
                          expected_panel_sha256: str) -> dict[str, Any]:
    require(HEX64.fullmatch(expected_panel_sha256) is not None,
            "external panel SHA-256")
    required = {"schema", "rows", "test_layer_count", "validation_slot_count",
                "test_layers", "validation_slots", "partition_component_counts",
                "split_uses_source_hash", "canonical_source_dtype", "panel_sha256"}
    require(set(panel) == required and panel["schema"] ==
            "logic-q-v2-bound-panel-v1", "panel exact schema")
    base = dict(panel)
    claimed = base.pop("panel_sha256")
    require(claimed == sha256(canonical_json(base)) and
            claimed == expected_panel_sha256, "panel seal and external pin")
    rows = []
    for ordinal, row in enumerate(panel["rows"]):
        require(isinstance(row, dict) and set(row) == {
            "layer", "slot", "role", "rows", "cols", "source_sha256",
            "source_dtype", "component_ordinal", "partition"},
            "panel literal row schema")
        require(row["component_ordinal"] == ordinal, "panel canonical ordinal")
        rows.append(PanelRow(row["layer"], row["slot"], row["role"],
                             row["rows"], row["cols"], row["source_sha256"],
                             row["source_dtype"]))
    rebuilt = make_panel_record(
        rows, test_layer_count=int(panel["test_layer_count"]),
        validation_slot_count=int(panel["validation_slot_count"]))
    require(rebuilt == dict(panel), "panel canonical reconstruction")
    return rebuilt


SCORED_ROW_FIELDS = {
    "schema", "config_id", "layer", "slot", "role", "partition",
    "component_ordinal", "panel_source_sha256", "source_dtype",
    "source_blob_bytes", "expert_packet_sha256", "expert_packet_bytes",
    "packet_geometry_sha256",
    "component_packet_sha256", "component_packet_bytes",
    "decoded_source_count", "raw_sse_f64_hex", "raw_energy_f64_hex",
    "scorer_schema", "row_receipt_sha256",
}


def seal_scored_row(row: Mapping[str, Any]) -> dict[str, Any]:
    require("row_receipt_sha256" not in row, "unsealed scored row input")
    result = dict(row)
    result["row_receipt_sha256"] = sha256(canonical_json(result))
    return result


def _float_hex(value: Any, label: str, *, allow_zero: bool) -> float:
    require(isinstance(value, str), f"{label} hex string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise BindingError(f"{label} float hex") from exc
    require(math.isfinite(result) and (result >= 0.0 if allow_zero else result > 0.0),
            f"{label} finite domain")
    require(result.hex() == value, f"{label} canonical float hex")
    return result


def validate_scored_row(panel: Mapping[str, Any], row: Mapping[str, Any], *,
                        config_ids: set[str],
                        packet_receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    require(set(row) == SCORED_ROW_FIELDS and row["schema"] ==
            "logic-q-v2-independent-scored-row-v1", "scored row schema")
    base = dict(row)
    claimed = base.pop("row_receipt_sha256")
    require(HEX64.fullmatch(claimed) is not None and
            claimed == sha256(canonical_json(base)), "scored row seal")
    require(row["config_id"] in config_ids and row["partition"] in
            {"train", "validation"}, "scored row selection domain")
    ordinal = row["component_ordinal"]
    require(isinstance(ordinal, int) and 0 <= ordinal < len(panel["rows"]),
            "scored row ordinal")
    expected = panel["rows"][ordinal]
    for field, panel_field in (("layer", "layer"), ("slot", "slot"),
                               ("role", "role"), ("partition", "partition"),
                               ("panel_source_sha256", "source_sha256"),
                               ("source_dtype", "source_dtype")):
        require(row[field] == expected[panel_field], f"scored row {field} binding")
    count = int(expected["rows"]) * int(expected["cols"])
    require(row["decoded_source_count"] == count and
            row["source_blob_bytes"] == count * SOURCE_DTYPES[row["source_dtype"]],
            "scored row source size")
    require(HEX64.fullmatch(row["expert_packet_sha256"]) is not None and
            HEX64.fullmatch(row["component_packet_sha256"]) is not None and
            isinstance(row["expert_packet_bytes"], int) and
            row["expert_packet_bytes"] > 0 and
            isinstance(row["component_packet_bytes"], int) and
            row["component_packet_bytes"] > 0,
            "scored row packet fields")
    require(row["expert_packet_sha256"] in packet_receipts,
            "scored row packet receipt exists")
    packet_receipt = packet_receipts[row["expert_packet_sha256"]]
    require(row["packet_geometry_sha256"] ==
            packet_receipt["packet_geometry_sha256"] and
            row["expert_packet_bytes"] == packet_receipt["expert_packet_bytes"],
            "scored row expert packet receipt binding")
    component = packet_receipt["components"][row["role"]]
    require(row["component_packet_sha256"] == component["packet_sha256"] and
            row["component_packet_bytes"] == component["packet_bytes"] and
            row["decoded_source_count"] == component["source_count"],
            "scored row component packet receipt binding")
    require(row["scorer_schema"] == "logic-q-v2-independent-source-scorer-v1",
            "scored row independent scorer")
    _float_hex(row["raw_sse_f64_hex"], "raw SSE", allow_zero=True)
    _float_hex(row["raw_energy_f64_hex"], "raw energy", allow_zero=False)
    return dict(row)


def _config_ids(v1: Any) -> set[str]:
    return {config.config_id for config in v1.FROZEN_CONFIGS}


def _derive_metrics(v1: Any, core: Any, panel: Mapping[str, Any],
                    rows_by_config: Mapping[str, Sequence[Mapping[str, Any]]],
                    packet_receipts: Mapping[str, Mapping[str, Any]]
                    ) -> dict[str, Any]:
    config_ids = _config_ids(v1)
    require(set(rows_by_config) == config_ids, "complete frozen config rows")
    validated_packets = {
        packet_hash: validate_packet_geometry_receipt(core, packet_receipt)
        for packet_hash, packet_receipt in packet_receipts.items()
    }
    require(validated_packets and set(validated_packets) == set(packet_receipts) and
            all(packet_hash == packet_receipt["expert_packet_sha256"]
                for packet_hash, packet_receipt in validated_packets.items()),
            "validated packet receipt set")
    expected_ordinals = [row["component_ordinal"] for row in panel["rows"]
                         if row["partition"] in {"train", "validation"}]
    output: dict[str, Any] = {}
    for config_id in sorted(config_ids):
        rows = [validate_scored_row(panel, row, config_ids=config_ids,
                                    packet_receipts=validated_packets)
                for row in rows_by_config[config_id]]
        require(all(row["config_id"] == config_id for row in rows),
                "scored row outer config binding")
        rows.sort(key=lambda row: row["component_ordinal"])
        require([row["component_ordinal"] for row in rows] == expected_ordinals,
                "one scored row per train/validation panel component")
        partition_metrics = {}
        for partition in ("train", "validation"):
            selected = [row for row in rows if row["partition"] == partition]
            groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for row in selected:
                groups.setdefault((row["layer"], row["slot"]), []).append(row)
            physical_bits = 0
            raw_sse = 0.0
            raw_energy = 0.0
            weights = 0
            for key in sorted(groups):
                triplet = sorted(groups[key], key=lambda row: ROLE_ORDINAL[row["role"]])
                require([row["role"] for row in triplet] == list(ROLE_ORDER),
                        "scored expert role triplet")
                require(len({(row["expert_packet_sha256"],
                              row["expert_packet_bytes"]) for row in triplet}) == 1,
                        "scored expert shared packet binding")
                packet_receipt = validated_packets[
                    triplet[0]["expert_packet_sha256"]]
                require(len({row["packet_geometry_sha256"] for row in triplet}) == 1
                        and triplet[0]["packet_geometry_sha256"] ==
                        packet_receipt["packet_geometry_sha256"],
                        "scored expert geometry receipt triplet")
                physical_bits += int(packet_receipt["physical_bits"])
                packet_weights = int(packet_receipt["expert_weights_from_headers"])
                observed_weights = sum(int(row["decoded_source_count"])
                                       for row in triplet)
                require(observed_weights == packet_weights,
                        "aggregate packet-header weight count")
                weights += packet_weights
                for row in triplet:
                    raw_sse += _float_hex(row["raw_sse_f64_hex"], "aggregate SSE",
                                          allow_zero=True)
                    raw_energy += _float_hex(row["raw_energy_f64_hex"],
                                             "aggregate energy", allow_zero=False)
            require(groups and weights > 0 and raw_energy > 0.0,
                    "nonempty derived aggregate")
            rate = physical_bits / weights
            relative_mse = raw_sse / raw_energy
            partition_metrics[partition] = {
                "physical_bits": physical_bits, "weights": weights,
                "raw_sse_f64_hex": raw_sse.hex(),
                "raw_energy_f64_hex": raw_energy.hex(),
                "expert_count": len(groups), "physical_rate_bpw": rate,
                "relative_mse": relative_mse,
                "F": relative_mse * 2.0 ** (2.0 * rate),
            }
        output[config_id] = partition_metrics
    return output


def _selected_config(derived: Mapping[str, Any]) -> tuple[str, list[Any]]:
    def key(config_id: str) -> tuple[Any, ...]:
        row = derived[config_id]["validation"]
        rate = float(row["physical_rate_bpw"])
        return (not (RATE_MIN <= rate <= RATE_MAX), float(row["F"]),
                rate, config_id)
    selected = min(derived, key=key)
    return selected, list(key(selected))


def make_selection_receipt(v1: Any, core: Any, panel: Mapping[str, Any], *,
                           expected_panel_sha256: str,
                           rows_by_config: Mapping[str, Sequence[Mapping[str, Any]]],
                           packet_receipts_by_sha256: Mapping[str, Mapping[str, Any]]
                           ) -> dict[str, Any]:
    validated_panel = validate_panel_record(
        panel, expected_panel_sha256=expected_panel_sha256)
    canonical_packets = {
        packet_hash: canonical_copy(packet_receipts_by_sha256[packet_hash])
        for packet_hash in sorted(packet_receipts_by_sha256)
    }
    require(canonical_packets, "nonempty packet receipt set")
    for packet_hash, packet_receipt in canonical_packets.items():
        require(packet_hash == packet_receipt.get("expert_packet_sha256"),
                "packet receipt map key")
    canonical_rows = {config_id: sorted(
        [dict(row) for row in rows_by_config[config_id]],
        key=lambda row: row["component_ordinal"])
        for config_id in sorted(rows_by_config)}
    derived = _derive_metrics(v1, core, validated_panel, canonical_rows,
                              canonical_packets)
    selected, key = _selected_config(derived)
    receipt = {
        "schema": "logic-q-v2-bound-selection-receipt-v1",
        "panel_sha256": expected_panel_sha256,
        "v1_source_root_sha256": V1_SOURCE_ROOT_SHA256,
        "frozen_grid": canonical_copy(v1.frozen_grid_record()),
        "literal_scored_rows_by_config": canonical_rows,
        "literal_scored_rows_root_sha256": sha256(canonical_json(canonical_rows)),
        "literal_packet_receipts_by_sha256": canonical_packets,
        "literal_packet_receipts_root_sha256": sha256(
            canonical_json(canonical_packets)),
        "derived_metrics": derived,
        "selected_config_id": selected,
        "selection_key": key,
        "selection_algorithm": "recompute_validation_rate_interval_then_F_rate_id",
        "test_rows_or_metrics_accepted": False,
    }
    receipt["receipt_sha256"] = sha256(canonical_json(receipt))
    return receipt


def authorize_test(v1: Any, core: Any, panel: Mapping[str, Any],
                   receipt: Mapping[str, Any], *,
                   expected_panel_sha256: str,
                   expected_receipt_sha256: str) -> Any:
    validated_panel = validate_panel_record(
        panel, expected_panel_sha256=expected_panel_sha256)
    require(HEX64.fullmatch(expected_receipt_sha256) is not None,
            "external selection receipt SHA-256")
    required = {"schema", "panel_sha256", "v1_source_root_sha256", "frozen_grid",
                "literal_scored_rows_by_config", "literal_scored_rows_root_sha256",
                "literal_packet_receipts_by_sha256",
                "literal_packet_receipts_root_sha256",
                "derived_metrics", "selected_config_id", "selection_key",
                "selection_algorithm", "test_rows_or_metrics_accepted",
                "receipt_sha256"}
    require(set(receipt) == required and receipt["schema"] ==
            "logic-q-v2-bound-selection-receipt-v1", "selection receipt schema")
    base = dict(receipt)
    claimed = base.pop("receipt_sha256")
    require(claimed == sha256(canonical_json(base)) and
            claimed == expected_receipt_sha256,
            "selection receipt seal and external pin")
    require(receipt["panel_sha256"] == expected_panel_sha256 and
            receipt["v1_source_root_sha256"] == V1_SOURCE_ROOT_SHA256 and
            receipt["frozen_grid"] == canonical_copy(v1.frozen_grid_record()) and
            receipt["selection_algorithm"] ==
            "recompute_validation_rate_interval_then_F_rate_id" and
            receipt["test_rows_or_metrics_accepted"] is False,
            "selection immutable bindings")
    literal_rows = receipt["literal_scored_rows_by_config"]
    require(receipt["literal_scored_rows_root_sha256"] ==
            sha256(canonical_json(literal_rows)), "literal scored-row root")
    packet_receipts = receipt["literal_packet_receipts_by_sha256"]
    require(isinstance(packet_receipts, Mapping) and packet_receipts and
            receipt["literal_packet_receipts_root_sha256"] ==
            sha256(canonical_json(packet_receipts)),
            "literal packet-receipt root")
    derived = _derive_metrics(v1, core, validated_panel, literal_rows,
                              packet_receipts)
    require(derived == receipt["derived_metrics"], "recomputed aggregate metrics")
    selected, key = _selected_config(derived)
    require(selected == receipt["selected_config_id"] and
            key == receipt["selection_key"], "recomputed selected config")
    selected_rate = float(derived[selected]["validation"]["physical_rate_bpw"])
    require(RATE_MIN <= selected_rate <= RATE_MAX,
            "selected validation physical rate interval")
    matches = [config for config in v1.FROZEN_CONFIGS
               if config.config_id == selected]
    require(len(matches) == 1, "authorized frozen config")
    return matches[0]


def _hex_bytes(value: Any, length: int, label: str) -> bytes:
    require(isinstance(value, str) and len(value) == 2 * length and
            value == value.lower(), f"{label} canonical hex")
    try:
        payload = bytes.fromhex(value)
    except ValueError as exc:
        raise BindingError(f"{label} hex") from exc
    require(payload.hex() == value, f"{label} canonical hex")
    return payload


def validate_packet_geometry_receipt(core: Any,
                                     receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "expert_packet_bytes", "expert_packet_sha256",
        "expert_header_hex", "components", "expert_shape",
        "expert_weights_from_headers", "physical_bits", "physical_rate_bpw",
        "routed_storage_read_bytes", "read_passes", "cold_read_amplification",
        "cold_read_below_2x", "encoder_metric_objects_used",
        "packet_geometry_sha256",
    }
    require(set(receipt) == required and receipt["schema"] ==
            "logic-q-v2-packet-geometry-v2", "packet geometry exact schema")
    base = dict(receipt)
    claimed = base.pop("packet_geometry_sha256")
    require(HEX64.fullmatch(str(claimed)) is not None and
            claimed == sha256(canonical_json(base)), "packet geometry seal")
    packet_hash = receipt["expert_packet_sha256"]
    packet_bytes = receipt["expert_packet_bytes"]
    require(HEX64.fullmatch(str(packet_hash)) is not None and
            isinstance(packet_bytes, int) and packet_bytes >= core.EXPERT_PAGE and
            packet_bytes % core.EXPERT_PAGE == 0,
            "packet geometry physical expert object")
    header = _hex_bytes(receipt["expert_header_hex"], core.EXPERT_HEADER_BYTES,
                        "expert header")
    fields = core.EXPERT_HEADER.unpack(header)
    magic, version, count, reserved16 = fields[:4]
    lengths = tuple(int(value) for value in fields[4:7])
    reserved = fields[7]
    require(magic == core.EXPERT_MAGIC and version == 0 and count == 3 and
            reserved16 == 0 and reserved == b"\0" * 28,
            "packet receipt expert header")
    components = receipt["components"]
    require(isinstance(components, Mapping) and
            set(components) == set(ROLE_ORDER), "packet receipt exact roles")
    expected_component_fields = {
        "packet_sha256", "packet_bytes", "header_hex", "role", "family",
        "profile", "rows", "cols", "block_size", "parameter", "blocks",
        "scale_bytes", "payload_bits", "source_count",
    }
    observed_shapes = []
    observed_counts = []
    cursor = core.EXPERT_HEADER_BYTES
    for role, length in zip(ROLE_ORDER, lengths):
        component = components[role]
        require(isinstance(component, Mapping) and
                set(component) == expected_component_fields,
                "packet component receipt schema")
        require(component["role"] == role and
                component["packet_bytes"] == length and
                HEX64.fullmatch(str(component["packet_sha256"])) is not None,
                "packet component receipt identity")
        component_header = _hex_bytes(
            component["header_hex"], core.COMPONENT_HEADER_BYTES,
            f"{role} component header")
        record = core.decode_component_header(component_header)
        observed = {
            "role": record.role, "family": record.family,
            "profile": record.profile, "rows": record.rows,
            "cols": record.cols, "block_size": record.block_size,
            "parameter": record.parameter, "blocks": record.blocks,
            "scale_bytes": record.scale_bytes,
            "payload_bits": record.payload_bits,
            "source_count": record.source_count,
        }
        for key, value in observed.items():
            require(component[key] == value,
                    f"packet component header-derived {key}")
        expected_length = (core.COMPONENT_HEADER_BYTES + record.scale_bytes +
                           (record.payload_bits + 7) // 8)
        require(length == expected_length and cursor + length <= packet_bytes,
                "packet component header-derived length")
        cursor = core.align_up(cursor + length, core.COMPONENT_ALIGNMENT)
        observed_shapes.append((record.rows, record.cols))
        observed_counts.append(record.source_count)
    require(len(set(observed_shapes)) == 1 and cursor <= packet_bytes,
            "packet Gate/Up/DownT canonical shape closure")
    shape = list(observed_shapes[0])
    weights = sum(observed_counts)
    require(receipt["expert_shape"] == shape and
            receipt["expert_weights_from_headers"] == weights and
            receipt["physical_bits"] == packet_bytes * 8 and
            receipt["physical_rate_bpw"] == packet_bytes * 8 / weights,
            "packet header-derived aggregate")
    require(receipt["routed_storage_read_bytes"] == packet_bytes and
            receipt["read_passes"] == 1 and
            receipt["cold_read_amplification"] == 1.0 and
            receipt["cold_read_below_2x"] is True and
            receipt["encoder_metric_objects_used"] is False,
            "packet one-pass physical ledger")
    return dict(receipt)


def packet_geometry(np: Any, v1: Any, core: Any, packet: bytes) -> dict[str, Any]:
    parts = v1._expert_component_slices(core, bytes(packet))
    decoded = v1.unpack_canonical_expert(np, core, bytes(packet))
    records = {role: core.parse_component_envelope(parts[role])[0]
               for role in ROLE_ORDER}
    require(set(decoded) == set(ROLE_ORDER) and
            len({(record.rows, record.cols) for record in records.values()}) == 1,
            "packet geometry SwiGLU closure")
    role_counts = {role: int(records[role].source_count) for role in ROLE_ORDER}
    weights = sum(role_counts.values())
    require(weights == sum(len(decoded[role][0]) for role in ROLE_ORDER),
            "packet-derived count replay")
    components = {}
    for role in ROLE_ORDER:
        record = records[role]
        components[role] = {
            "packet_sha256": sha256(parts[role]),
            "packet_bytes": len(parts[role]),
            "header_hex": parts[role][:core.COMPONENT_HEADER_BYTES].hex(),
            "role": record.role, "family": record.family,
            "profile": record.profile, "rows": record.rows,
            "cols": record.cols, "block_size": record.block_size,
            "parameter": record.parameter, "blocks": record.blocks,
            "scale_bytes": record.scale_bytes,
            "payload_bits": record.payload_bits,
            "source_count": record.source_count,
        }
    result = {
        "schema": "logic-q-v2-packet-geometry-v2",
        "expert_packet_bytes": len(packet),
        "expert_packet_sha256": sha256(bytes(packet)),
        "expert_header_hex": bytes(packet[:core.EXPERT_HEADER_BYTES]).hex(),
        "components": components,
        "expert_shape": [records[ROLE_ORDER[0]].rows,
                         records[ROLE_ORDER[0]].cols],
        "expert_weights_from_headers": weights,
        "physical_bits": len(packet) * 8,
        "physical_rate_bpw": len(packet) * 8 / weights,
        "routed_storage_read_bytes": len(packet),
        "read_passes": 1,
        "cold_read_amplification": 1.0,
        "cold_read_below_2x": True,
        "encoder_metric_objects_used": False,
    }
    result["packet_geometry_sha256"] = sha256(canonical_json(result))
    return validate_packet_geometry_receipt(core, result)


def _device_name(properties: Mapping[Any, Any]) -> str:
    value = properties.get("name", properties.get(b"name"))
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("utf-8")
    require(isinstance(value, str) and value, "CUDA device name")
    return value


def make_launch_context(*, panel_sha256: str, selection_receipt_sha256: str,
                        config_id: str, layer: str, slot: str,
                        rows: int, cols: int) -> dict[str, Any]:
    require(HEX64.fullmatch(panel_sha256) is not None and
            HEX64.fullmatch(selection_receipt_sha256) is not None,
            "launch context external pins")
    require(config_id and layer and slot and rows > 0 and cols > 0,
            "launch context fields")
    context = {
        "schema": "logic-q-v2-cupy-launch-context-v1",
        "panel_sha256": panel_sha256,
        "selection_receipt_sha256": selection_receipt_sha256,
        "config_id": config_id, "layer": layer, "slot": slot,
        "rows": rows, "cols": cols,
        "v1_source_root_sha256": V1_SOURCE_ROOT_SHA256,
    }
    context["launch_nonce"] = sha256(canonical_json(context))
    return context


def collect_cupy_launch_receipt(xp: Any, *,
                                launch_context: Mapping[str, Any]) -> dict[str, Any]:
    required_context = {"schema", "panel_sha256", "selection_receipt_sha256",
                        "config_id", "layer", "slot", "rows", "cols",
                        "v1_source_root_sha256", "launch_nonce"}
    require(set(launch_context) == required_context and
            launch_context["schema"] == "logic-q-v2-cupy-launch-context-v1",
            "launch context schema")
    context_base = dict(launch_context)
    launch_nonce = context_base.pop("launch_nonce")
    require(HEX64.fullmatch(str(launch_nonce)) is not None and
            launch_nonce == sha256(canonical_json(context_base)) and
            launch_context["v1_source_root_sha256"] == V1_SOURCE_ROOT_SHA256,
            "launch context seal")
    require(isinstance(xp, types.ModuleType) and xp.__name__ == "cupy",
            "actual CuPy module object")
    try:
        canonical = importlib.import_module("cupy")
    except ImportError as exc:
        raise BindingError("canonical CuPy import") from exc
    require(xp is canonical and isinstance(xp.__spec__, importlib.machinery.ModuleSpec)
            and xp.__spec__.name == "cupy" and xp.__spec__.loader is not None and
            getattr(xp, "__package__", None) == "cupy" and
            getattr(xp.ndarray, "__module__", "").split(".", 1)[0] == "cupy",
            "canonical CuPy module identity")
    module_file = Path(xp.__file__).resolve(strict=True)
    module_payload = _regular_bytes(module_file, "CuPy module file")
    runtime = xp.cuda.runtime
    device_id = int(runtime.getDevice())
    require(0 <= device_id < int(runtime.getDeviceCount()), "CUDA device id")
    properties = runtime.getDeviceProperties(device_id)
    device = xp.cuda.Device(device_id)
    probe = xp.arange(4096, dtype=xp.uint64)
    observed = int(xp.asnumpy(xp.sum((probe * xp.uint64(17) + xp.uint64(3))
                                    % xp.uint64(65521), dtype=xp.uint64)))
    expected = sum((index * 17 + 3) % 65521 for index in range(4096))
    require(observed == expected, "CuPy launch arithmetic probe")
    xp.cuda.get_current_stream().synchronize()
    receipt = {
        "schema": "logic-q-v2-cupy-launch-receipt-v1",
        "launch_nonce": launch_nonce,
        "launch_context": canonical_copy(launch_context),
        "module_name": xp.__name__, "module_version": str(xp.__version__),
        "module_file": str(module_file),
        "module_file_sha256": sha256(module_payload),
        "device_id": device_id, "device_count": int(runtime.getDeviceCount()),
        "device_name": _device_name(properties),
        "compute_capability": str(device.compute_capability),
        "device_pci_bus_id": int(properties.get(
            "pciBusID", properties.get(b"pciBusID", -1))),
        "device_pci_device_id": int(properties.get(
            "pciDeviceID", properties.get(b"pciDeviceID", -1))),
        "device_multiprocessors": int(properties.get(
            "multiProcessorCount", properties.get(b"multiProcessorCount", -1))),
        "runtime_version": int(runtime.runtimeGetVersion()),
        "driver_version": int(runtime.driverGetVersion()),
        "probe_elements": 4096, "probe_observed": observed,
        "probe_expected": expected, "stream_synchronized": True,
        "v1_source_root_sha256": V1_SOURCE_ROOT_SHA256,
    }
    receipt["receipt_sha256"] = sha256(canonical_json(receipt))
    return receipt


def validate_cupy_launch_receipt(xp: Any, receipt: Mapping[str, Any], *,
                                 expected_receipt_sha256: str,
                                 expected_launch_context: Mapping[str, Any]
                                 ) -> dict[str, Any]:
    require(HEX64.fullmatch(expected_receipt_sha256) is not None,
            "external launch receipt SHA-256")
    require(isinstance(receipt, Mapping) and
            receipt.get("launch_context") == canonical_copy(expected_launch_context) and
            HEX64.fullmatch(str(receipt.get("launch_nonce", ""))) is not None,
            "launch receipt input")
    fresh = collect_cupy_launch_receipt(
        xp, launch_context=expected_launch_context)
    require(dict(receipt) == fresh and fresh["receipt_sha256"] ==
            expected_receipt_sha256, "current CuPy launch binding")
    return fresh


def encode_expert_bound(xp: Any, v1: Any, core: Any,
                        roles: Mapping[str, tuple[Any, Any]], *,
                        rows: int, cols: int, config: Any,
                        launch_receipt: Mapping[str, Any],
                        expected_launch_receipt_sha256: str,
                        launch_context: Mapping[str, Any]) -> dict[str, Any]:
    require(config.config_id == launch_context.get("config_id") and
            rows == launch_context.get("rows") and
            cols == launch_context.get("cols"),
            "encode/launch context binding")
    validated_launch = validate_cupy_launch_receipt(
        xp, launch_receipt,
        expected_receipt_sha256=expected_launch_receipt_sha256,
        expected_launch_context=launch_context)
    encoded = v1.encode_expert(xp, core, roles, rows=rows, cols=cols,
                               config=config, live=True)
    geometry = packet_geometry(importlib.import_module("numpy"), v1, core,
                               encoded["packet"])
    return {
        "packet": encoded["packet"], "packet_geometry": geometry,
        "launch_receipt": validated_launch,
        "encoder_diagnostic": encoded["score"],
        "encoder_diagnostic_authoritative_for_final_score": False,
        "independent_source_score_required": True,
    }
