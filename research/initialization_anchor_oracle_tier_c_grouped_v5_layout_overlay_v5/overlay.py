"""Source-only v4-shortlist authentication and exact layout-overlay merge.

This module imports no GPU runtime.  Production calls it after runtime parity
and before opening any Qwen payload.  The only reusable source-derived object
is the hash-bound grouped-v4 global per-domain stage-0 TopK state.
"""

from __future__ import annotations

import io
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import common


@dataclass(frozen=True)
class AuthenticatedV4TopK:
    translated_ordinals: np.ndarray
    metrics: np.ndarray
    receipt: dict[str, Any]


@dataclass(frozen=True)
class OverlayMerge:
    domain_ordinals: np.ndarray
    domain_metrics: np.ndarray
    union_ordinals: np.ndarray
    receipt: dict[str, Any]


def _array_sha(values: np.ndarray, dtype: str) -> str:
    return common.sha256_bytes(np.asarray(values, dtype=dtype).tobytes(order="C"))


def _read_bound_bytes(
    path: Path, *, expected_sha256: str, expected_bytes: int, label: str
) -> bytes:
    """Read one regular file once, then authenticate its complete byte string."""
    unresolved = common.require_regular_file_before_resolve(path, label)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(unresolved, flags)
    except OSError as error:
        raise common.ProtocolError(f"cannot single-open {label}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != int(expected_bytes):
            raise common.ProtocolError(f"{label} type/byte-count mismatch")
        chunks: list[bytes] = []
        remaining = int(expected_bytes)
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise common.ProtocolError(f"{label} ended before its bound byte count")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise common.ProtocolError(f"{label} grew beyond its bound byte count")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if common.sha256_bytes(raw) != expected_sha256:
        raise common.ProtocolError(f"{label} SHA-256 mismatch")
    return raw


def _json_from_bound_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise common.ProtocolError(f"invalid UTF-8 JSON in {label}") from error
    if not isinstance(value, dict):
        raise common.ProtocolError(f"{label} must contain one JSON object")
    return value


def _validate_v4_result(value: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    expected = lock["v4_reuse"]
    if value.get("schema") != "qwen3_initialization_anchor_tier_c_grouped_v4_result_v1":
        raise common.ProtocolError("reused v4 result schema mismatch")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping):
        raise common.ProtocolError("reused v4 result has no binding object")
    for key, wanted in expected["result_required_bindings"].items():
        if bindings.get(key) != wanted:
            raise common.ProtocolError(f"reused v4 result binding mismatch: {key}")
    candidate_space = value.get("candidate_space", {})
    if candidate_space.get("logical_candidate_count") != 50_331_648:
        raise common.ProtocolError("reused v4 logical count mismatch")
    if candidate_space.get("effective_candidate_count") != common.V4_EFFECTIVE_CANDIDATES:
        raise common.ProtocolError("reused v4 effective count mismatch")
    if tuple(candidate_space.get("domain_ids", ())) != common.DOMAIN_IDS:
        raise common.ProtocolError("reused v4 domain order mismatch")
    if candidate_space.get("equivalence_map_sha256") != "1699afb9596faf197971c704f16aefd2a20e39e267f2f81ea41cc94a1a46e1e5":
        raise common.ProtocolError("reused v4 equivalence-map hash mismatch")
    coordinates = value.get("coordinates", {})
    if coordinates.get("stage0_plan_sha256") != expected["stage0_plan_sha256"]:
        raise common.ProtocolError("reused v4 stage-0 coordinate plan mismatch")
    if coordinates.get("full_plan_sha256") != expected["full_plan_sha256"]:
        raise common.ProtocolError("reused v4 full-coordinate plan mismatch")
    search = value.get("search", {})
    if search.get("stage0_top_k_per_domain") != common.STAGE0_TOP_K:
        raise common.ProtocolError("reused v4 TopK width mismatch")
    if search.get("stage0_shard_count") != common.SEED_SHARD_COUNT:
        raise common.ProtocolError("reused v4 seed-shard count mismatch")
    expected_union_count = int(expected["merged_state_arrays"]["union_ordinals"]["shape"][0])
    if search.get("union_shortlist_count") != expected_union_count:
        raise common.ProtocolError("reused v4 union-shortlist count mismatch")
    ledger = value.get("physical_ledger", {})
    if ledger.get("scientific_scores_use_decoded_fp16_affines") is not True:
        raise common.ProtocolError("reused v4 did not bind decoded-FP16 scoring")
    calibration = value.get("backend", {}).get("source_free_calibration", {})
    if calibration.get("schema") != "qwen3_initialization_anchor_tier_c_grouped_source_free_calibration_v4":
        raise common.ProtocolError("reused v4 calibration schema mismatch")
    if calibration.get("receipt_sha256") != expected["calibration_internal_sha256"]:
        raise common.ProtocolError("reused v4 calibration internal receipt mismatch")
    if calibration.get("coordinate_count") != 512 or calibration.get("candidate_count") != 64_512:
        raise common.ProtocolError("reused v4 calibration geometry mismatch")
    resume = value.get("resume_state", {})
    if resume.get("event_count_before_result") != 392:
        raise common.ProtocolError("reused v4 pre-result journal length mismatch")


def _validate_v4_audit(value: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    expected = lock["v4_reuse"]["result_audit"]
    if value.get("schema") != expected["schema"] or value.get("status") != expected["status"]:
        raise common.ProtocolError("v4 result-audit schema/status mismatch")
    normalized = dict(value)
    observed_internal = normalized.pop("audit_receipt_sha256", None)
    if observed_internal != expected["internal_sha256"]:
        raise common.ProtocolError("v4 result-audit internal literal mismatch")
    if common.sha256_bytes(common.canonical_json_bytes(normalized)) != observed_internal:
        raise common.ProtocolError("v4 result-audit internal hash mismatch")
    # The final audit package freezes these field paths.  Requiring each one
    # prevents a matching audit file hash from becoming an opaque trust token.
    for dotted, wanted in expected["required_fields"].items():
        cursor: Any = value
        for component in dotted.split("."):
            if not isinstance(cursor, Mapping) or component not in cursor:
                raise common.ProtocolError(f"v4 audit missing field: {dotted}")
            cursor = cursor[component]
        if cursor != wanted:
            raise common.ProtocolError(f"v4 audit field mismatch: {dotted}")


def _parse_v4_topk(raw: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != {"domain_top_ordinals", "domain_top_q", "union_ordinals"}:
                raise common.ProtocolError("reused v4 merged NPZ member set mismatch")
            ordinals = archive["domain_top_ordinals"]
            metrics = archive["domain_top_q"]
            union = archive["union_ordinals"]
    except common.ProtocolError:
        raise
    except Exception as error:
        raise common.ProtocolError("invalid reused v4 merged NPZ") from error
    if ordinals.dtype != np.dtype("uint64") or ordinals.shape != (33, 2048):
        raise common.ProtocolError("reused v4 ordinal array shape/dtype mismatch")
    if metrics.dtype != np.dtype("float64") or metrics.shape != (33, 2048):
        raise common.ProtocolError("reused v4 metric array shape/dtype mismatch")
    if union.dtype != np.dtype("uint64") or union.ndim != 1:
        raise common.ProtocolError("reused v4 union array shape/dtype mismatch")
    if not np.all(np.isfinite(metrics)):
        raise common.ProtocolError("reused v4 metrics contain non-finite values")
    for index in range(33):
        row_ordinals = ordinals[index]
        row_metrics = metrics[index]
        if len(np.unique(row_ordinals)) != common.STAGE0_TOP_K:
            raise common.ProtocolError("reused v4 TopK row contains duplicate ordinals")
        order = np.lexsort((row_ordinals, row_metrics))
        if not np.array_equal(order, np.arange(common.STAGE0_TOP_K)):
            raise common.ProtocolError("reused v4 TopK row violates metric/ordinal order")
    expected_union = np.unique(ordinals.reshape(-1))
    if not np.array_equal(union, expected_union):
        raise common.ProtocolError("reused v4 union does not equal per-domain TopK union")
    return ordinals, metrics, union


def authenticate_v4_topk(
    v4_run_root: Path, v4_result_audit_path: Path, lock: Mapping[str, Any]
) -> AuthenticatedV4TopK:
    """Authenticate the immutable v4 result/event/TopK and translate ordinals."""
    binding = lock["v4_reuse"]
    root = common.preflight_output_directory(
        v4_run_root, allow_existing=True, label="reused grouped-v4 run root"
    )
    result_path = root / binding["result_basename"]
    result_raw = _read_bound_bytes(
        result_path,
        expected_sha256=binding["result_sha256"],
        expected_bytes=binding["result_bytes"],
        label="reused grouped-v4 result",
    )
    result = _json_from_bound_bytes(result_raw, "reused grouped-v4 result")
    _validate_v4_result(result, lock)

    audit_binding = binding["result_audit"]
    audit_raw = _read_bound_bytes(
        v4_result_audit_path,
        expected_sha256=audit_binding["file_sha256"],
        expected_bytes=audit_binding["file_bytes"],
        label="independent grouped-v4 result audit",
    )
    _validate_v4_audit(
        _json_from_bound_bytes(audit_raw, "independent grouped-v4 result audit"), lock
    )

    event_binding = binding["merged_event"]
    event_path = root / event_binding["relative_path"]
    event_raw = _read_bound_bytes(
        event_path,
        expected_sha256=event_binding["sha256"],
        expected_bytes=event_binding["bytes"],
        label="reused grouped-v4 merged-state event",
    )
    event = _json_from_bound_bytes(event_raw, "reused grouped-v4 merged-state event")
    common.strict_keys(
        event,
        {"sequence", "previous_event_sha256", "kind", "key", "relative_path",
         "file_sha256", "file_bytes", "created_unix_ns"},
        "reused grouped-v4 merged-state event",
    )
    for key, wanted in event_binding["required_fields"].items():
        if event.get(key) != wanted:
            raise common.ProtocolError(f"reused grouped-v4 event mismatch: {key}")
    result_events = result.get("resume_state", {}).get("events")
    if not isinstance(result_events, list) or len(result_events) != 392 or result_events[257] != event:
        raise common.ProtocolError("reused grouped-v4 result does not embed the exact merged event")
    relative = Path(str(event["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise common.ProtocolError("reused grouped-v4 state target escapes journal")
    state_path = root / "state" / relative
    state_raw = _read_bound_bytes(
        state_path,
        expected_sha256=event["file_sha256"],
        expected_bytes=int(event["file_bytes"]),
        label="reused grouped-v4 global per-domain TopK state",
    )
    old_ordinals, old_metrics, old_union = _parse_v4_topk(state_raw)
    expected_union_count = int(binding["merged_state_arrays"]["union_ordinals"]["shape"][0])
    if old_union.shape != (expected_union_count,):
        raise common.ProtocolError("reused grouped-v4 union count mismatch")
    translated_flat = np.fromiter(
        (common.translate_v4_ordinal(int(value)) for value in old_ordinals.reshape(-1)),
        dtype=np.uint64,
        count=old_ordinals.size,
    )
    translated = translated_flat.reshape(old_ordinals.shape)
    old_union_translated = np.fromiter(
        (common.translate_v4_ordinal(int(value)) for value in old_union),
        dtype=np.uint64,
        count=len(old_union),
    )
    if len(old_union_translated) > 1 and not np.all(old_union_translated[1:] > old_union_translated[:-1]):
        raise common.ProtocolError("v4-to-expanded ordinal translation is not strictly increasing")
    for index in range(33):
        order = np.lexsort((translated[index], old_metrics[index]))
        if not np.array_equal(order, np.arange(common.STAGE0_TOP_K)):
            raise common.ProtocolError("ordinal translation changed the v4 TopK total order")
    receipt = {
        "schema": "qwen3_tier_c_grouped_v5_authenticated_v4_topk_v1",
        "result_sha256": binding["result_sha256"],
        "result_audit_sha256": audit_binding["file_sha256"],
        "merged_event_sha256": event_binding["sha256"],
        "merged_state_sha256": event["file_sha256"],
        "old_domain_top_ordinals_sha256_u64le": _array_sha(old_ordinals, "<u8"),
        "translated_domain_top_ordinals_sha256_u64le": _array_sha(translated, "<u8"),
        "domain_top_metrics_sha256_f64le": _array_sha(old_metrics, "<f8"),
        "old_union_count": len(old_union),
        "translation_strictly_increasing_on_old_union": True,
        "old_topk_total_order_preserved": True,
        "qwen_payload_opened_by_authentication": False,
    }
    receipt["receipt_sha256"] = common.sha256_bytes(common.canonical_json_bytes(receipt))
    for key, wanted in binding["authenticated_array_hashes"].items():
        if receipt.get(key) != wanted:
            raise common.ProtocolError(f"reused grouped-v4 authenticated array hash mismatch: {key}")
    if receipt["receipt_sha256"] != binding["expected_authentication_receipt_sha256"]:
        raise common.ProtocolError("reused grouped-v4 authentication receipt mismatch")
    return AuthenticatedV4TopK(translated, old_metrics.copy(), receipt)


def merge_topk(
    old_ordinals: np.ndarray,
    old_metrics: np.ndarray,
    new_ordinals: np.ndarray,
    new_metrics: np.ndarray,
) -> OverlayMerge:
    """Merge complete old/new per-domain TopK lists without truncating to K."""
    arrays = (
        np.asarray(old_ordinals), np.asarray(old_metrics),
        np.asarray(new_ordinals), np.asarray(new_metrics),
    )
    if arrays[0].shape != (33, 2048) or arrays[2].shape != (33, 2048):
        raise common.ProtocolError("overlay ordinal TopK shape mismatch")
    if arrays[1].shape != (33, 2048) or arrays[3].shape != (33, 2048):
        raise common.ProtocolError("overlay metric TopK shape mismatch")
    if arrays[0].dtype != np.uint64 or arrays[2].dtype != np.uint64:
        raise common.ProtocolError("overlay ordinals must be uint64")
    if arrays[1].dtype != np.float64 or arrays[3].dtype != np.float64:
        raise common.ProtocolError("overlay metrics must be float64")
    if not np.all(np.isfinite(arrays[1])) or not np.all(np.isfinite(arrays[3])):
        raise common.ProtocolError("overlay metrics contain non-finite values")
    merged_ordinals = np.empty((33, 4096), dtype=np.uint64)
    merged_metrics = np.empty((33, 4096), dtype=np.float64)
    for domain_index in range(33):
        old_o, old_q = arrays[0][domain_index], arrays[1][domain_index]
        new_o, new_q = arrays[2][domain_index], arrays[3][domain_index]
        if len(np.unique(old_o)) != 2048 or len(np.unique(new_o)) != 2048:
            raise common.ProtocolError("overlay input TopK row contains duplicates")
        if np.intersect1d(old_o, new_o, assume_unique=True).size:
            raise common.ProtocolError("old and new overlay families overlap")
        ordinals = np.concatenate((old_o, new_o))
        metrics = np.concatenate((old_q, new_q))
        order = np.lexsort((ordinals, metrics))
        merged_ordinals[domain_index] = ordinals[order]
        merged_metrics[domain_index] = metrics[order]
    union = np.unique(merged_ordinals.reshape(-1))
    if len(union) > 33 * 4096:
        raise common.ProtocolError("overlay stage-1 union exceeds 135,168 candidates")
    receipt = {
        "schema": "qwen3_tier_c_grouped_v5_layout_overlay_merge_v1",
        "domain_count": 33,
        "old_top_k_per_domain": 2048,
        "new_top_k_per_domain": 2048,
        "merged_candidates_per_domain": 4096,
        "stage1_union_count": len(union),
        "stage1_union_max": 135_168,
        "stable_order": "minimum_float64_metric_then_smallest_expanded_uint64_ordinal",
        "no_post_merge_topk_truncation": True,
        "domain_ordinals_sha256_u64le": _array_sha(merged_ordinals, "<u8"),
        "domain_metrics_sha256_f64le": _array_sha(merged_metrics, "<f8"),
        "union_ordinals_sha256_u64le": _array_sha(union, "<u8"),
    }
    receipt["receipt_sha256"] = common.sha256_bytes(common.canonical_json_bytes(receipt))
    return OverlayMerge(merged_ordinals, merged_metrics, union, receipt)
