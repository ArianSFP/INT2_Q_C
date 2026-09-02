#!/usr/bin/env python3
"""Cross-fit, control, rate and typed-HOLD contract for epsilon-TCQ v0."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


RATE_MIN = 2.15
RATE_MAX = 2.5
F_MAX = 0.8
READ_MAX = 2.0
CONTROL_GAIN_MIN = 0.03
CONTROL_COUNT = 8


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def owner_component_folds(stream_owner_sets: Sequence[Sequence[int]],
                          core: Any) -> tuple[dict[str, Any], ...]:
    components = core.connected_owner_components(stream_owner_sets)
    require(len(components) >= 2, "at least two owner components")
    folds = []
    for ordinal, component in enumerate(components):
        held = set(component)
        held_streams = [index for index, owners in enumerate(stream_owner_sets)
                        if held.intersection(int(value) for value in owners)]
        development = [index for index in range(len(stream_owner_sets))
                       if index not in held_streams]
        require(held_streams and development, "nonempty outer fold partitions")
        require(not any(held.intersection(stream_owner_sets[index])
                        for index in development), "whole-owner holdout")
        folds.append({
            "fold_ordinal": ordinal,
            "held_owner_component": list(component),
            "held_stream_ordinals": held_streams,
            "development_stream_ordinals": development,
        })
    return tuple(folds)


def effective_gain_bpw(rate: float, distortion: float,
                       baseline_rate: float, baseline_distortion: float) -> float:
    require(all(math.isfinite(value) and value > 0.0 for value in
                (rate, distortion, baseline_rate, baseline_distortion)),
            "effective gain inputs")
    return (baseline_rate - rate) - 0.5 * math.log2(distortion / baseline_distortion)


def validate_byte_ledger(ledger: Mapping[str, Any]) -> None:
    required = {
        "header_bytes", "model_bytes", "topology_bytes", "frequency_bytes",
        "centroid_bytes", "directory_bytes", "frame_header_bytes",
        "payload_bytes", "padding_bytes", "total_bytes",
    }
    require(set(ledger) == required and
            all(type(ledger[name]) is int and ledger[name] >= 0
                for name in required), "complete physical byte ledger")
    # topology/frequency are a required audit decomposition of model_bytes,
    # not additional bytes.  Count the physical model exactly once.
    physical_fields = (
        "header_bytes", "model_bytes", "centroid_bytes", "directory_bytes",
        "frame_header_bytes", "payload_bytes", "padding_bytes",
    )
    require(ledger["total_bytes"] == sum(
        ledger[name] for name in physical_fields),
        "byte ledger conservation")
    require(ledger["model_bytes"] ==
            ledger["topology_bytes"] + ledger["frequency_bytes"],
            "model topology/frequency conservation")


def source_gate(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(folds, "source folds")
    pooled_weights = 0
    pooled_sse = 0.0
    pooled_bytes = 0
    fold_rows = []
    for row in folds:
        required = {
            "weights", "sse", "source_energy", "bytes", "nearest_bytes",
            "nearest_sse", "state_gain_bpw", "local_gain_bpw",
            "permuted_gain_bpw", "literal_reencode", "read_amplification",
            "byte_ledger",
        }
        require(set(row) == required, "source fold exact schema")
        validate_byte_ledger(row["byte_ledger"])
        require(row["literal_reencode"] is True and row["weights"] > 0 and
                row["source_energy"] > 0.0 and row["sse"] >= 0.0,
                "source fold validity")
        require(row["state_gain_bpw"] > max(row["local_gain_bpw"],
                                             row["permuted_gain_bpw"]),
                "state-aware beats local/permuted")
        rate = 8.0 * row["bytes"] / row["weights"]
        distortion = row["sse"] / row["source_energy"]
        F = distortion * 2.0 ** (2.0 * rate)
        fold_pass = (RATE_MIN <= rate <= RATE_MAX and F <= F_MAX and
                     row["read_amplification"] < READ_MAX and
                     row["state_gain_bpw"] > 0.0)
        require(fold_pass, "every source outer fold passes")
        pooled_weights += row["weights"]
        pooled_sse += row["sse"]
        pooled_bytes += row["bytes"]
        fold_rows.append({"rate_bpw": rate, "relative_mse": distortion,
                          "F": F, "pass": fold_pass})
    return {
        "status": "SOURCE_SURVIVOR_OPEN_EIGHT_FULL_PIPELINE_CONTROLS",
        "controls_may_open": True,
        "folds": fold_rows,
        "pooled_weights": pooled_weights,
        "pooled_bytes": pooled_bytes,
        "pooled_sse": pooled_sse,
    }


def final_control_gate(source_gain_bpw: float,
                       controls: Sequence[Mapping[str, Any]],
                       *, source_survived: bool) -> dict[str, Any]:
    require(source_survived, "controls cannot open before source survival")
    require(len(controls) == CONTROL_COUNT, "all eight matched controls")
    gains = []
    for ordinal, row in enumerate(controls):
        require(set(row) == {
            "ordinal", "full_ptq_pipeline", "legal_trace_regenerated",
            "nested_selection_repeated", "literal_reencode", "gain_bpw",
            "closure_sha256",
        }, "control exact schema")
        require(row["ordinal"] == ordinal and
                all(row[name] is True for name in (
                    "full_ptq_pipeline", "legal_trace_regenerated",
                    "nested_selection_repeated", "literal_reencode")) and
                isinstance(row["closure_sha256"], str) and
                len(row["closure_sha256"]) == 64 and
                math.isfinite(row["gain_bpw"]), "matched control closure")
        gains.append(float(row["gain_bpw"]))
    excess = float(source_gain_bpw) - max(gains)
    status = ("ELIGIBLE_FOR_BOUND_QWEN_EPSILON_TCQ_PILOT"
              if excess + 1e-15 >= CONTROL_GAIN_MIN
              else "HARD_KILL_QWEN_MINUS_CONTROL_BELOW_0P03_BPW")
    return {
        "status": status, "control_gains_bpw": gains,
        "strongest_control_gain_bpw": max(gains),
        "qwen_minus_strongest_control_gain_bpw": excess,
        "minimum_excess_bpw": CONTROL_GAIN_MIN,
    }


def missing_strata_adapter_hold() -> dict[str, Any]:
    return {
        "schema": "epsilon-tcq-wfa-v0-typed-hold",
        "status": "HOLD_NO_AUTHENTICATED_CURRENT_CODEC_LEGAL_TRANSITION_ADAPTER",
        "qwen_payload_may_open": False,
        "direct_four_level_fallback_allowed": False,
        "required_next_artifact": (
            "a separately frozen adapter that enumerates actual nearby legal "
            "POLARIS/STRATA indices and causally replays their six SC events, "
            "contexts, state transitions and literal reconstruction"
        ),
    }
