#!/usr/bin/env python3
"""Bound outer-fold, artifact, metric, control, and read driver for v1."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence


RATE_MIN = 2.15
RATE_MAX = 2.5
F_MAX = 0.8
READ_MAX = 2.0
CONTROL_COUNT = 8
CONTROL_EXCESS_MIN = 0.03
OUTER_SCHEMA = "epsilon-tcq-bound-outer-plan-v1"
ARTIFACT_SCHEMA = "epsilon-tcq-bound-fold-artifacts-v1"
CONTROL_SCHEMA = "epsilon-tcq-bound-matched-control-panel-v1"


class DriverError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DriverError(message)


def canonical_json(value: Any) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_digest(value: Any, label: str) -> str:
    require(type(value) is str and len(value) == 64, f"{label} digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise DriverError(f"{label} digest") from error
    require(value == value.lower(), f"{label} lowercase")
    return value


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    require("seal_sha256" not in body, "unsealed body")
    output = dict(body)
    output["seal_sha256"] = sha256(canonical_json(output))
    return output


def verify_seal(row: Mapping[str, Any]) -> str:
    require(isinstance(row, Mapping) and "seal_sha256" in row, "sealed mapping")
    body = dict(row)
    value = require_digest(body.pop("seal_sha256"), "record seal")
    require(sha256(canonical_json(body)) == value, "record seal mismatch")
    return value


def connected_owner_components(stream_owner_sets: Sequence[Sequence[int]],
                               experts: int) -> tuple[tuple[int, ...], ...]:
    require(type(experts) is int and experts >= 2 and stream_owner_sets,
            "owner graph geometry")
    parent = list(range(experts))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a == b:
            return
        if a < b:
            parent[b] = a
        else:
            parent[a] = b

    normalized = []
    for raw in stream_owner_sets:
        owners = tuple(int(value) for value in raw)
        require(owners == tuple(sorted(set(owners))) and owners and
                0 <= owners[0] and owners[-1] < experts, "stream owner set")
        normalized.append(owners)
        for value in owners[1:]:
            union(owners[0], value)
    groups: dict[int, list[int]] = {}
    for expert in range(experts):
        groups.setdefault(find(expert), []).append(expert)
    return tuple(tuple(values) for _key, values in
                 sorted(groups.items(), key=lambda item: item[1][0]))


def build_outer_plan(stream_owner_sets: Sequence[Sequence[int]], experts: int) -> dict[str, Any]:
    owners = [list(map(int, row)) for row in stream_owner_sets]
    components = connected_owner_components(stream_owner_sets, experts)
    require(len(components) >= 2, "at least two outer owner components")
    folds = []
    for ordinal, component in enumerate(components):
        held_owners = set(component)
        held = [index for index, row in enumerate(owners)
                if held_owners.intersection(row)]
        development = [index for index in range(len(owners)) if index not in held]
        require(held and development and all(
            held_owners.isdisjoint(owners[index]) for index in development),
            "whole-owner outer fold")
        folds.append({
            "fold_ordinal": ordinal,
            "held_owner_component": list(component),
            "held_stream_ordinals": held,
            "development_stream_ordinals": development,
            "topology_fit_stream_ordinals": development,
            "frequency_fit_stream_ordinals": development,
            "centroid_fit_stream_ordinals": development,
            "inner_selection_stream_ordinals": development,
        })
    body = {
        "schema": OUTER_SCHEMA,
        "status": "COMPLETE_WHOLE_OWNER_OUTER_CLOSURE",
        "experts": experts,
        "stream_owner_sets": owners,
        "components": [list(row) for row in components],
        "folds": folds,
    }
    return seal(body)


def validate_outer_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    keys = {"schema", "status", "experts", "stream_owner_sets", "components",
            "folds", "seal_sha256"}
    require(isinstance(plan, Mapping) and set(plan) == keys,
            "outer plan exact schema")
    plan_seal = verify_seal(plan)
    require(plan["schema"] == OUTER_SCHEMA and
            plan["status"] == "COMPLETE_WHOLE_OWNER_OUTER_CLOSURE",
            "outer plan identity")
    expected = build_outer_plan(plan["stream_owner_sets"], int(plan["experts"]))
    require(expected == dict(plan), "outer plan is exactly derived")
    # Explicitly recheck every fitting/selection list: no partial helper may
    # substitute a held stream or omit an inconvenient development stream.
    for row in plan["folds"]:
        development = row["development_stream_ordinals"]
        for name in ("topology_fit_stream_ordinals",
                     "frequency_fit_stream_ordinals",
                     "centroid_fit_stream_ordinals",
                     "inner_selection_stream_ordinals"):
            require(row[name] == development, "outer fold fit/selection closure")
    return {"outer_plan_sha256": plan_seal,
            "components": len(plan["components"]),
            "folds": len(plan["folds"]),
            "all_fit_and_selection_inputs_are_development_only": True}


def _float64_values(payload: bytes) -> tuple[float, ...]:
    import struct

    require(type(payload) is bytes and len(payload) > 0 and len(payload) % 8 == 0,
            "binary64 score artifact")
    values = tuple(row[0] for row in struct.iter_unpack("<d", payload))
    require(all(math.isfinite(value) for value in values), "finite score artifact")
    return values


def _score(source: bytes, reconstruction: bytes) -> tuple[int, float, float]:
    left, right = _float64_values(source), _float64_values(reconstruction)
    require(len(left) == len(right), "score artifact geometry")
    sse = math.fsum((a - b) * (a - b) for a, b in zip(left, right, strict=True))
    energy = math.fsum(value * value for value in left)
    require(energy > 0.0 and math.isfinite(sse), "score energy/SSE")
    return len(left), sse, energy


def effective_gain_bpw(*, baseline_rate: float, baseline_distortion: float,
                       candidate_rate: float, candidate_distortion: float) -> float:
    require(all(math.isfinite(value) and value > 0.0 for value in
                (baseline_rate, baseline_distortion, candidate_rate,
                 candidate_distortion)), "gain inputs")
    return ((baseline_rate - candidate_rate) -
            0.5 * math.log2(candidate_distortion / baseline_distortion))


def artifact_receipt(*, source_kind: str, outer_plan_sha256: str,
                     fold_ordinal: int, held_stream_ordinals: Sequence[int],
                     adapter_replay_receipt_sha256: str,
                     source_f64le: bytes,
                     reconstructions: Mapping[str, bytes],
                     packets: Mapping[str, bytes]) -> dict[str, Any]:
    require(source_kind in {"QWEN", "MATCHED_GAUSSIAN_FULL_PTQ"},
            "source kind")
    require(set(reconstructions) == set(packets) ==
            {"nearest", "local", "state", "state_permuted"},
            "artifact modes")
    body = {
        "schema": ARTIFACT_SCHEMA,
        "status": "AUTHENTICATED_DECODED_FOLD_ARTIFACTS",
        "source_kind": source_kind,
        "outer_plan_sha256": require_digest(outer_plan_sha256, "outer plan"),
        "fold_ordinal": int(fold_ordinal),
        "held_stream_ordinals": list(map(int, held_stream_ordinals)),
        "adapter_replay_receipt_sha256": require_digest(
            adapter_replay_receipt_sha256, "adapter replay"),
        "source_f64le_sha256": sha256(source_f64le),
        "reconstruction_f64le_sha256": {
            name: sha256(reconstructions[name]) for name in sorted(reconstructions)},
        "packet_sha256": {name: sha256(packets[name]) for name in sorted(packets)},
    }
    return seal(body)


def _validate_artifact_receipt(bundle: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    receipt = bundle["receipt"]
    keys = {"schema", "status", "source_kind", "outer_plan_sha256",
            "fold_ordinal", "held_stream_ordinals",
            "adapter_replay_receipt_sha256", "source_f64le_sha256",
            "reconstruction_f64le_sha256", "packet_sha256", "seal_sha256"}
    require(isinstance(receipt, Mapping) and set(receipt) == keys,
            "artifact receipt exact schema")
    receipt_seal = verify_seal(receipt)
    require(receipt["schema"] == ARTIFACT_SCHEMA and
            receipt["status"] == "AUTHENTICATED_DECODED_FOLD_ARTIFACTS" and
            receipt["outer_plan_sha256"] == plan["seal_sha256"],
            "artifact receipt identity")
    fold = plan["folds"][receipt["fold_ordinal"]]
    require(receipt["held_stream_ordinals"] == fold["held_stream_ordinals"],
            "artifact whole held fold binding")
    require(receipt["source_f64le_sha256"] == sha256(bundle["source_f64le"]),
            "source artifact binding")
    modes = {"nearest", "local", "state", "state_permuted"}
    require(set(bundle["reconstructions"]) == set(bundle["packets"]) == modes and
            set(receipt["reconstruction_f64le_sha256"]) ==
            set(receipt["packet_sha256"]) == modes, "artifact mode closure")
    for name in modes:
        require(receipt["reconstruction_f64le_sha256"][name] ==
                sha256(bundle["reconstructions"][name]) and
                receipt["packet_sha256"][name] == sha256(bundle["packets"][name]),
                "decoded artifact hash binding")
    require_digest(receipt["adapter_replay_receipt_sha256"], "adapter replay")
    return receipt_seal


def derive_fold(bundle: Mapping[str, Any], plan: Mapping[str, Any],
                *, packet_module: Any, independent_decoder: Any) -> dict[str, Any]:
    bundle_keys = {"receipt", "source_f64le", "reconstructions", "packets",
                   "routed_expert"}
    require(isinstance(bundle, Mapping) and set(bundle) == bundle_keys,
            "fold bundle exact schema")
    validate_outer_plan(plan)
    artifact_seal = _validate_artifact_receipt(bundle, plan)
    modes = ("nearest", "local", "state", "state_permuted")
    parsed = {}
    scores = {}
    for name in modes:
        packet = bundle["packets"][name]
        parsed[name] = packet_module.parse_packet(packet)
        independent = independent_decoder.decode_and_reencode(packet)
        require(independent["packet_sha256"] == parsed[name]["packet_sha256"] and
                independent["packet_bytes"] == parsed[name]["total_bytes"],
                "independent packet decode/reencode binding")
        count, sse, energy = _score(
            bundle["source_f64le"], bundle["reconstructions"][name])
        require(parsed[name]["weights"] == count, "packet/source weight binding")
        rate = 8.0 * parsed[name]["total_bytes"] / count
        distortion = sse / energy
        scores[name] = {
            "weights": count, "sse": sse, "source_energy": energy,
            "bytes": parsed[name]["total_bytes"], "rate_bpw": rate,
            "relative_mse": distortion,
            "F": distortion * 2.0 ** (2.0 * rate),
            "packet_sha256": parsed[name]["packet_sha256"],
            "byte_ledger": parsed[name]["byte_ledger"],
            "literal_independent_reencode": independent["canonical_reencode_matches"],
        }
        require(scores[name]["bytes"] == scores[name]["byte_ledger"]["total_bytes"],
                "row bytes exactly equal literal byte ledger")
    baseline = scores["nearest"]
    gains = {}
    for name in ("local", "state", "state_permuted"):
        gains[name] = effective_gain_bpw(
            baseline_rate=baseline["rate_bpw"],
            baseline_distortion=baseline["relative_mse"],
            candidate_rate=scores[name]["rate_bpw"],
            candidate_distortion=scores[name]["relative_mse"])
    expert = bundle["routed_expert"]
    trace = packet_module.owner_read_trace(bundle["packets"]["state"], expert)
    require(trace["packet_sha256"] == scores["state"]["packet_sha256"] and
            trace["compressed_expert_second_pass_count"] == 0,
            "derived one-pass routed read")
    state = scores["state"]
    checks = {
        "state_beats_local": gains["state"] > gains["local"],
        "state_beats_permuted": gains["state"] > gains["state_permuted"],
        "state_positive_gain": gains["state"] > 0.0,
        "rate_in_interval": RATE_MIN <= state["rate_bpw"] <= RATE_MAX,
        "F_at_most_0p8": state["F"] <= F_MAX,
        "cold_read_below_2x": trace["cold_read_amplification"] < READ_MAX,
        "literal_reencode": all(scores[name]["literal_independent_reencode"]
                                for name in modes),
    }
    checks["fold_pass"] = all(checks.values())
    return {
        "schema": "epsilon-tcq-bound-derived-fold-v1",
        "fold_ordinal": bundle["receipt"]["fold_ordinal"],
        "artifact_receipt_sha256": artifact_seal,
        "source_kind": bundle["receipt"]["source_kind"],
        "scores": scores, "gains_bpw": gains,
        "read_trace": trace, "checks": checks,
    }


def derive_panel(bundles: Sequence[Mapping[str, Any]], plan: Mapping[str, Any],
                 *, packet_module: Any, independent_decoder: Any,
                 require_target_pass: bool) -> dict[str, Any]:
    validate_outer_plan(plan)
    require(len(bundles) == len(plan["folds"]), "one artifact bundle per outer fold")
    rows = [derive_fold(bundle, plan, packet_module=packet_module,
                        independent_decoder=independent_decoder)
            for bundle in bundles]
    require([row["fold_ordinal"] for row in rows] == list(range(len(rows))),
            "outer result fold closure/order")
    source_kinds = {row["source_kind"] for row in rows}
    require(len(source_kinds) == 1, "panel source kind")
    pooled = {}
    for mode in ("nearest", "state"):
        weights = sum(row["scores"][mode]["weights"] for row in rows)
        sse = math.fsum(row["scores"][mode]["sse"] for row in rows)
        energy = math.fsum(row["scores"][mode]["source_energy"] for row in rows)
        physical_bytes = sum(row["scores"][mode]["bytes"] for row in rows)
        rate = 8.0 * physical_bytes / weights
        distortion = sse / energy
        pooled[mode] = {
            "weights": weights, "bytes": physical_bytes, "sse": sse,
            "source_energy": energy, "rate_bpw": rate,
            "relative_mse": distortion,
            "F": distortion * 2.0 ** (2.0 * rate),
        }
    gain = effective_gain_bpw(
        baseline_rate=pooled["nearest"]["rate_bpw"],
        baseline_distortion=pooled["nearest"]["relative_mse"],
        candidate_rate=pooled["state"]["rate_bpw"],
        candidate_distortion=pooled["state"]["relative_mse"])
    all_folds = all(row["checks"]["fold_pass"] for row in rows)
    if require_target_pass:
        require(all_folds, "every source outer fold must pass")
    return {
        "schema": "epsilon-tcq-bound-derived-panel-v1",
        "source_kind": next(iter(source_kinds)),
        "outer_plan_sha256": plan["seal_sha256"],
        "folds": rows, "pooled": pooled,
        "pooled_state_gain_bpw": gain,
        "all_outer_folds_pass": all_folds,
        "controls_may_open": require_target_pass and all_folds,
    }


def control_receipt(*, ordinal: int, outer_plan_sha256: str,
                    pipeline_source_root_sha256: str,
                    source_producer_receipt_sha256: str,
                    legal_trace_panel_receipt_sha256: str,
                    fold_artifact_receipt_sha256: Sequence[str]) -> dict[str, Any]:
    body = {
        "schema": CONTROL_SCHEMA,
        "status": "FULL_MATCHED_GAUSSIAN_PTQ_PANEL_CLOSED",
        "ordinal": int(ordinal),
        "outer_plan_sha256": require_digest(outer_plan_sha256, "control outer plan"),
        "pipeline_source_root_sha256": require_digest(
            pipeline_source_root_sha256, "control pipeline source"),
        "source_producer_receipt_sha256": require_digest(
            source_producer_receipt_sha256, "control producer"),
        "legal_trace_panel_receipt_sha256": require_digest(
            legal_trace_panel_receipt_sha256, "control legal trace"),
        "fold_artifact_receipt_sha256": [require_digest(value, "control fold artifact")
                                           for value in fold_artifact_receipt_sha256],
    }
    return seal(body)


def final_control_gate(source_panel: Mapping[str, Any],
                       controls: Sequence[Mapping[str, Any]],
                       pinned_receipt_sha256: Sequence[str],
                       *, packet_module: Any, independent_decoder: Any) -> dict[str, Any]:
    require(source_panel.get("controls_may_open") is True and
            source_panel.get("source_kind") == "QWEN",
            "controls stay closed before real source survival")
    require(len(controls) == len(pinned_receipt_sha256) == CONTROL_COUNT,
            "exact eight pinned controls")
    gains = []
    receipt_hashes = []
    for ordinal, (control, pin) in enumerate(zip(
            controls, pinned_receipt_sha256, strict=True)):
        require(isinstance(control, Mapping) and set(control) ==
                {"receipt", "outer_plan", "fold_bundles"},
                "control bundle exact schema")
        receipt = control["receipt"]
        keys = {"schema", "status", "ordinal", "outer_plan_sha256",
                "pipeline_source_root_sha256", "source_producer_receipt_sha256",
                "legal_trace_panel_receipt_sha256",
                "fold_artifact_receipt_sha256", "seal_sha256"}
        require(isinstance(receipt, Mapping) and set(receipt) == keys,
                "control receipt exact schema")
        verify_seal(receipt)
        receipt_hash = sha256(canonical_json(receipt))
        require(receipt_hash == require_digest(pin, "externally pinned control receipt"),
                "control receipt external pin")
        require(receipt["schema"] == CONTROL_SCHEMA and
                receipt["status"] == "FULL_MATCHED_GAUSSIAN_PTQ_PANEL_CLOSED" and
                receipt["ordinal"] == ordinal, "control identity/order")
        validate_outer_plan(control["outer_plan"])
        require(receipt["outer_plan_sha256"] ==
                control["outer_plan"]["seal_sha256"], "control outer plan binding")
        actual_fold_receipts = [bundle["receipt"]["seal_sha256"]
                                for bundle in control["fold_bundles"]]
        require(receipt["fold_artifact_receipt_sha256"] == actual_fold_receipts,
                "control fold artifact closure")
        panel = derive_panel(
            control["fold_bundles"], control["outer_plan"],
            packet_module=packet_module, independent_decoder=independent_decoder,
            require_target_pass=False)
        require(panel["source_kind"] == "MATCHED_GAUSSIAN_FULL_PTQ",
                "control source kind")
        gains.append(panel["pooled_state_gain_bpw"])
        receipt_hashes.append(receipt_hash)
    excess = source_panel["pooled_state_gain_bpw"] - max(gains)
    return {
        "status": ("ELIGIBLE_FOR_SEALED_EPSILON_TCQ_COMPOSITE" if
                   excess + 1e-15 >= CONTROL_EXCESS_MIN else
                   "HARD_KILL_QWEN_MINUS_CONTROL_BELOW_0P03_BPW"),
        "control_receipt_sha256": receipt_hashes,
        "control_gains_bpw": gains,
        "strongest_control_gain_bpw": max(gains),
        "qwen_minus_strongest_control_gain_bpw": excess,
        "minimum_excess_bpw": CONTROL_EXCESS_MIN,
    }
