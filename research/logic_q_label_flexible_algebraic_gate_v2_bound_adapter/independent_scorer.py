#!/usr/bin/env python3
"""Independent raw-source scorer for canonical LOGIC-Q expert packets.

This module has no encoder entry point. It derives counts and reconstruction
from the packet, parses authenticated source bytes, and emits sealed per-role
rows consumed by the v2 selector.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import types
from typing import Any, Mapping


class ScorerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScorerError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _decode_source(np: Any, payload: bytes, dtype: str, count: int) -> Any:
    require(isinstance(payload, bytes), "literal source bytes")
    if dtype == "bf16-le":
        require(len(payload) == count * 2, "BF16 source byte count")
        words = np.frombuffer(payload, dtype="<u2", count=count)
        bits = (words.astype(np.uint32) << np.uint32(16))
        values = bits.view(np.float32).astype(np.float64)
    elif dtype == "float32-le":
        require(len(payload) == count * 4, "FP32 source byte count")
        values = np.frombuffer(payload, dtype="<f4", count=count).astype(np.float64)
    elif dtype == "float64-le":
        require(len(payload) == count * 8, "FP64 source byte count")
        values = np.frombuffer(payload, dtype="<f8", count=count).astype(np.float64)
    else:
        raise ScorerError("unsupported source dtype")
    require(values.size == count and np.all(np.isfinite(values)),
            "finite decoded source")
    return values


def _actual_numpy() -> Any:
    try:
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise ScorerError("canonical NumPy import") from exc
    require(isinstance(np, types.ModuleType) and np.__name__ == "numpy" and
            getattr(np, "__spec__", None) is not None and
            np.__spec__.name == "numpy", "actual NumPy module")
    return np


def score_expert_packet(binder: Any, v1: Any, core: Any, *,
                        packet: bytes, source_blobs: Mapping[str, bytes],
                        panel: Mapping[str, Any], expected_panel_sha256: str,
                        layer: str, slot: str, config_id: str) -> dict[str, Any]:
    """Score one packet from independent source bytes; trust no encoder metrics."""
    np = _actual_numpy()
    validated_panel = binder.validate_panel_record(
        panel, expected_panel_sha256=expected_panel_sha256)
    require(config_id in {config.config_id for config in v1.FROZEN_CONFIGS},
            "scorer frozen config")
    require(set(source_blobs) == set(binder.ROLE_ORDER), "scorer exact roles")
    panel_rows = [row for row in validated_panel["rows"]
                  if row["layer"] == layer and row["slot"] == slot]
    panel_rows.sort(key=lambda row: binder.ROLE_ORDINAL[row["role"]])
    require([row["role"] for row in panel_rows] == list(binder.ROLE_ORDER),
            "scorer panel expert triplet")
    geometry = binder.packet_geometry(np, v1, core, bytes(packet))
    decoded = v1.unpack_canonical_expert(np, core, bytes(packet))
    components = v1._expert_component_slices(core, bytes(packet))
    row_receipts = []
    pooled_sse = 0.0
    pooled_energy = 0.0
    pooled_count = 0
    for panel_row in panel_rows:
        role = panel_row["role"]
        record = core.parse_component_envelope(components[role])[0]
        require(record.role == role and record.rows == panel_row["rows"] and
                record.cols == panel_row["cols"] and
                record.source_count == panel_row["rows"] * panel_row["cols"],
                "scorer packet/panel geometry")
        blob = source_blobs[role]
        require(sha256(blob) == panel_row["source_sha256"],
                "scorer authenticated source blob")
        source = _decode_source(np, blob, panel_row["source_dtype"],
                                int(record.source_count))
        reconstruction = np.asarray(decoded[role][1], dtype=np.float64)
        require(reconstruction.shape == source.shape and
                np.all(np.isfinite(reconstruction)),
                "scorer finite reconstruction")
        residual = source - reconstruction
        sse = float(np.sum(residual * residual, dtype=np.float64))
        energy = float(np.sum(source * source, dtype=np.float64))
        require(math.isfinite(sse) and sse >= 0.0 and
                math.isfinite(energy) and energy > 0.0,
                "scorer raw metric domain")
        pooled_sse += sse
        pooled_energy += energy
        pooled_count += int(record.source_count)
        base = {
            "schema": "logic-q-v2-independent-scored-row-v1",
            "config_id": config_id,
            "layer": layer, "slot": slot, "role": role,
            "partition": panel_row["partition"],
            "component_ordinal": panel_row["component_ordinal"],
            "panel_source_sha256": panel_row["source_sha256"],
            "source_dtype": panel_row["source_dtype"],
            "source_blob_bytes": len(blob),
            "expert_packet_sha256": geometry["expert_packet_sha256"],
            "expert_packet_bytes": geometry["expert_packet_bytes"],
            "packet_geometry_sha256": geometry["packet_geometry_sha256"],
            "component_packet_sha256": sha256(components[role]),
            "component_packet_bytes": len(components[role]),
            "decoded_source_count": int(record.source_count),
            "raw_sse_f64_hex": sse.hex(),
            "raw_energy_f64_hex": energy.hex(),
            "scorer_schema": "logic-q-v2-independent-source-scorer-v1",
        }
        base["row_receipt_sha256"] = sha256(canonical_json(base))
        row_receipts.append(base)
    require(pooled_count == geometry["expert_weights_from_headers"],
            "scorer packet-derived pooled count")
    rate = geometry["physical_bits"] / pooled_count
    relative_mse = pooled_sse / pooled_energy
    result = {
        "schema": "logic-q-v2-independent-source-score-bundle-v1",
        "panel_sha256": expected_panel_sha256,
        "config_id": config_id, "layer": layer, "slot": slot,
        "expert_packet_sha256": geometry["expert_packet_sha256"],
        "packet_receipt": geometry,
        "row_receipts": row_receipts,
        "pooled": {
            "expert_weights_from_headers": pooled_count,
            "physical_bits": geometry["physical_bits"],
            "physical_rate_bpw": rate,
            "raw_sse_f64_hex": pooled_sse.hex(),
            "raw_energy_f64_hex": pooled_energy.hex(),
            "relative_mse": relative_mse,
            "F": relative_mse * 2.0 ** (2.0 * rate),
        },
        "encoder_metric_objects_used": False,
        "source_bytes_authenticated_before_scoring": True,
    }
    result["bundle_sha256"] = sha256(canonical_json(result))
    return result
