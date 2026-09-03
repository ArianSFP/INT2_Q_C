"""CuPy payload worker for the frozen panel.

Imported only after the source entrypoint's compile-time HOLD has been lifted.
Every source file is read once into memory; authentication and BF16 decoding use
that same byte buffer.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from common_latent_core import (
    BLOCK_VALUES,
    CONTROL_SEEDS,
    GRAY_PLANES,
    RATE_MAX,
    RATE_MIN,
    RECONSTRUCTION_RMS,
    TARGET_GAIN_BPW,
    THRESHOLD_RMS,
    affine_permutation_parameters,
    physical_page_envelope,
    private_byte_requirements,
    score_count_summary,
    scale_u16_cpu,
    validate_geometry,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _cupy():
    import cupy as cp
    return cp


def _safe_explicit_path(payload_root: Path, relative_path: str) -> Path:
    root = Path(payload_root)
    _require(root.is_dir() and not root.is_symlink(), "payload root must be a real directory")
    rel = Path(relative_path)
    _require(not rel.is_absolute() and ".." not in rel.parts, "unsafe panel relative path")
    root_resolved = root.resolve(strict=True)
    cursor = root
    for component in rel.parts:
        cursor = cursor / component
        _require(not cursor.is_symlink(), "symlink component in panel path")
    candidate = cursor
    _require(candidate.is_file(), "missing panel file")
    resolved = candidate.resolve(strict=True)
    _require(resolved == root_resolved or root_resolved in resolved.parents,
             "panel path escapes payload root")
    return candidate


def _authenticated_bytes(payload_root: Path, item: Mapping) -> bytes:
    path = _safe_explicit_path(payload_root, str(item["relative_path"]))
    data = path.read_bytes()
    _require(len(data) == int(item["bytes"]), "panel byte count mismatch")
    _require(hashlib.sha256(data).hexdigest() == str(item["sha256"]),
             "panel SHA-256 mismatch")
    return data


def _bf16_bytes_to_numpy(data: bytes, raw_shape: Sequence[int]) -> np.ndarray:
    raw = np.frombuffer(data, dtype="<u2")
    expected = math.prod(int(x) for x in raw_shape)
    _require(raw.size == expected, "BF16 element count mismatch")
    bits = raw.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32).reshape(tuple(int(x) for x in raw_shape))


def quantize_canonical_gpu(canonical, block_values: int = BLOCK_VALUES):
    """Return labels and CPU-authoritative binary16 scale bits."""
    cp = _cupy()
    host = np.ascontiguousarray(canonical, dtype=np.float32).reshape(-1)
    _require(int(host.size) % int(block_values) == 0,
             "panel worker requires complete quantizer blocks")
    # The shared CPU reference fixes reduction order and binary16 bits.
    scale_u16 = scale_u16_cpu(host, int(block_values))
    rms_cpu = scale_u16.view(np.float16)
    _require(bool(np.all(np.isfinite(rms_cpu))) and bool(np.all(rms_cpu > 0)),
             "invalid decoded binary16 scale")
    decoded = cp.asarray(rms_cpu.astype(np.float64))
    flat = cp.asarray(host, dtype=cp.float64)
    scale_per_value = cp.repeat(decoded, int(block_values))
    threshold = scale_per_value * THRESHOLD_RMS
    labels = cp.where(
        flat < -threshold,
        0,
        cp.where(flat < 0, 1, cp.where(flat <= threshold, 2, 3)),
    ).astype(cp.uint8)
    return labels, scale_u16


def load_quantized_panel_gpu(panel: Mapping, payload_root: Path):
    cp = _cupy()
    experts = [int(x) for x in panel["experts"]]
    d_ff, d_model = int(panel["d_ff"]), int(panel["d_model"])
    validate_geometry(len(experts), d_ff, d_model)
    _require(len(panel["files"]) == 2 * len(experts), "panel file count")
    locked = {(int(item["expert"]), str(item["role"])): item for item in panel["files"]}
    _require(len(locked) == 2 * len(experts), "duplicate panel bindings")
    role_stacks = []
    scale_bits = 0
    source_bytes = 0
    source_files = 0
    authenticated_inputs = []
    for expert in experts:
        roles = []
        for role in ("up", "down"):
            _require((expert, role) in locked, "missing explicit expert-role binding")
            item = locked[(expert, role)]
            data = _authenticated_bytes(payload_root, item)
            source_bytes += len(data)
            source_files += 1
            authenticated_inputs.append({
                "expert": expert,
                "role": role,
                "relative_path": item["relative_path"],
                "bytes": len(data),
                "sha256": item["sha256"],
            })
            raw = _bf16_bytes_to_numpy(data, item["raw_shape"])
            canonical = raw if role == "up" else np.ascontiguousarray(raw.T)
            _require(tuple(map(int, canonical.shape)) == (d_ff, d_model),
                     "canonical panel shape")
            labels, scales = quantize_canonical_gpu(canonical)
            scale_bits += int(scales.size) * 16
            roles.append(labels)
            del raw, canonical
        role_stacks.append(cp.stack(roles, axis=0))
    labels = cp.stack(role_stacks, axis=0)
    return labels, {
        "source_files_read_once": source_files,
        "source_bytes_read_once": source_bytes,
        "source_logical_host_scan_amplification": 1.0,
        "source_read_metric_scope": "one application-level host byte scan; not a filesystem-page or HBM traffic measurement",
        "scale_bits": scale_bits,
        "scale_bytes_per_expert": scale_bits // 8 // len(experts),
        "authenticated_inputs": authenticated_inputs,
    }


def summarize_counts_gpu(labels, cardinality: int, planes: Sequence[int] | None = None) -> dict:
    cp = _cupy()
    _require(labels.dtype == cp.uint8 and labels.ndim == 3, "GPU label tensor")
    e, roles, n = map(int, labels.shape)
    _require(roles == 2 and 2 <= e <= 256 and n > 0, "GPU label geometry")
    _require(cardinality in (2, 4), "latent cardinality")
    if cardinality == 4:
        _require(planes is None, "quaternary planes")
        chosen = (None, None)
    else:
        _require(planes is not None and len(planes) == 2 and all(p in (0, 1) for p in planes),
                 "binary planes")
        chosen = tuple(int(p) for p in planes)

    marginal = cp.zeros((e, roles, 4), dtype=cp.int64)
    latent = cp.zeros((roles, cardinality), dtype=cp.int64)
    conditional = cp.zeros((e, roles, cardinality, 4), dtype=cp.int64)
    gray = cp.asarray(GRAY_PLANES)
    for expert in range(e):
        for role in range(roles):
            for symbol in range(4):
                marginal[expert, role, symbol] = cp.count_nonzero(labels[expert, role] == symbol)
    for role in range(roles):
        if cardinality == 4:
            source = labels[:, role]
        else:
            source = gray[labels[:, role], chosen[role]]
        state_counts = cp.stack(
            [cp.sum(source == state, axis=0) for state in range(cardinality)], axis=0
        )
        u = cp.argmax(state_counts, axis=0).astype(cp.uint8)
        for state in range(cardinality):
            latent[role, state] = cp.count_nonzero(u == state)
            mask = u == state
            for expert in range(e):
                for symbol in range(4):
                    conditional[expert, role, state, symbol] = cp.count_nonzero(
                        (labels[expert, role] == symbol) & mask
                    )
    return {
        "expert_count": e,
        "role_count": roles,
        "coordinates_per_role": n,
        "cardinality": cardinality,
        "planes": list(chosen),
        "marginal_counts": cp.asnumpy(marginal),
        "latent_counts": cp.asnumpy(latent),
        "conditional_counts": cp.asnumpy(conditional),
    }


def score_labels_gpu(
    labels,
    cardinality: int,
    scale_bits: int = 0,
    selection_objective: str = "charged",
) -> dict:
    if cardinality == 4:
        _require(selection_objective in ("charged", "favorable"), "selection objective")
        return score_count_summary(summarize_counts_gpu(labels, 4), scale_bits)
    _require(cardinality == 2, "latent cardinality")
    candidates = [
        score_count_summary(summarize_counts_gpu(labels, 2, (u, d)), scale_bits)
        for u in (0, 1) for d in (0, 1)
    ]
    _require(selection_objective in ("charged", "favorable"), "selection objective")
    objective = (
        "common_two_part_bits" if selection_objective == "charged"
        else "conditional_data_bits"
    )
    selected = min(candidates, key=lambda x: (x[objective], x["planes"]))
    selected = dict(selected)
    selected["binary_plane_selection_objective"] = selection_objective
    selected["binary_plane_candidate_scores"] = [
        {
            "planes": item["planes"],
            "conditional_data_bits": item["conditional_data_bits"],
            "latent_data_bits": item["latent_data_bits"],
            "common_two_part_bits": item["common_two_part_bits"],
            "favorable_gross_gain_bpw": item["favorable_gross_gain_bpw"],
            "two_part_gain_bpw": item["two_part_gain_bpw"],
            "count_evidence": item["count_evidence"],
        }
        for item in candidates
    ]
    return selected


def coordinate_scramble_gpu(labels, seed: int):
    cp = _cupy()
    e, roles, n = map(int, labels.shape)
    base = cp.arange(n, dtype=cp.int64)
    out = cp.empty_like(labels)
    for expert in range(e):
        for role in range(roles):
            a, b = affine_permutation_parameters(n, int(seed), expert, role)
            out[expert, role] = labels[expert, role, (a * base + b) % n]
    return out


def _jsonable_score(score: Mapping) -> dict:
    result = {}
    for key, value in score.items():
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def _feasible_rate_endpoints(family_envelopes: Mapping) -> list[str]:
    """Return only frozen endpoints satisfying capacity and both read ledgers."""
    _require(set(family_envelopes) == {"2.15", "2.5"}, "rate endpoint set")
    eligible = []
    for rate in ("2.15", "2.5"):
        envelope = family_envelopes[rate]
        if (
            envelope.get("status") == "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC"
            and envelope.get("capacity_ok") is True
            and envelope.get("strictly_below_2x") is True
        ):
            eligible.append(rate)
    return eligible


def _final_disposition(
    *,
    favorable_below_target: bool,
    read_eligible_charged_gain_bpw: float | None,
    control_corrected_gain_bpw: float | None,
) -> tuple[str, bool]:
    """Fail closed in scientific, physical, then control order."""
    if favorable_below_target:
        return "HARD_KILL_FAVORABLE_IDEAL_BELOW_TARGET", False
    if read_eligible_charged_gain_bpw is None:
        return "HOLD_NO_CAPACITY_AND_STRICT_READ_FEASIBLE_RATE_ENDPOINT", False
    if read_eligible_charged_gain_bpw < TARGET_GAIN_BPW:
        return "HOLD_READ_FEASIBLE_CHARGED_MDL_BELOW_TARGET", False
    if control_corrected_gain_bpw is None:
        return "HOLD_CONTROLS_REQUIRED_BUT_NOT_RUN", False
    if control_corrected_gain_bpw < TARGET_GAIN_BPW:
        return "HOLD_CONTROL_CORRECTED_FAVORABLE_BELOW_TARGET", False
    return "SURVIVE_IDEAL_APERTURE_REQUIRES_FINITE_CODER", True


def run_authorized_panel(panel_path: Path, payload_root: Path) -> dict:
    cp = _cupy()
    panel_bytes = Path(panel_path).read_bytes()
    panel = json.loads(panel_bytes.decode("utf-8"))
    _require(panel.get("schema") == "same_layer_common_latent_panel_lock_v0", "panel schema")
    labels, io = load_quantized_panel_gpu(panel, Path(payload_root))
    variants = {
        "binary_favorable_oracle": score_labels_gpu(
            labels, 2, int(io["scale_bits"]), "favorable"
        ),
        "binary_charged_mdl": score_labels_gpu(
            labels, 2, int(io["scale_bits"]), "charged"
        ),
        "quaternary": score_labels_gpu(labels, 4, int(io["scale_bits"])),
    }
    best_name, best = max(
        (("binary_favorable_oracle", variants["binary_favorable_oracle"]),
         ("quaternary", variants["quaternary"])),
        key=lambda item: item[1]["favorable_gross_gain_bpw"]
    )
    hard_kill = best["favorable_gross_gain_bpw"] < TARGET_GAIN_BPW
    charged_candidates = {
        "binary_charged_mdl": variants["binary_charged_mdl"],
        "quaternary": variants["quaternary"],
    }
    best_charged_name, best_charged = max(
        charged_candidates.items(), key=lambda item: item[1]["two_part_gain_bpw"]
    )

    envelopes = {}
    coordinates = int(best["source_weights"]) // (len(panel["experts"]) * 2)
    for name, score in (
        ("binary_charged_mdl", variants["binary_charged_mdl"]),
        ("quaternary", variants["quaternary"]),
    ):
        latent_width = 1 if name == "binary_charged_mdl" else 2
        common_model_bits = int(score["latent_model_bits"]) + int(score["selector_bits"])
        private = private_byte_requirements(score, int(io["scale_bytes_per_expert"]))
        envelopes[name] = {
            "2.15": physical_page_envelope(
                expert_count=len(panel["experts"]),
                coordinates_per_role=coordinates,
                latent_bits_per_coordinate=latent_width,
                requested_rate=RATE_MIN,
                common_model_bits=common_model_bits,
                private_required_bytes=private,
            ),
            "2.5": physical_page_envelope(
                expert_count=len(panel["experts"]),
                coordinates_per_role=coordinates,
                latent_bits_per_coordinate=latent_width,
                requested_rate=RATE_MAX,
                common_model_bits=common_model_bits,
                private_required_bytes=private,
            ),
        }

    eligible_rate_endpoints = {
        name: _feasible_rate_endpoints(family_envelopes)
        for name, family_envelopes in envelopes.items()
    }
    read_eligible_candidates = {
        name: charged_candidates[name]
        for name, endpoints in eligible_rate_endpoints.items()
        if endpoints
    }
    if read_eligible_candidates:
        best_read_name, best_read_charged = max(
            read_eligible_candidates.items(),
            key=lambda item: item[1]["two_part_gain_bpw"],
        )
        best_read_favorable = (
            variants["binary_favorable_oracle"]
            if best_read_name == "binary_charged_mdl"
            else variants["quaternary"]
        )
    else:
        best_read_name = None
        best_read_charged = None
        best_read_favorable = None

    # Controls are expensive and cannot rescue a source candidate that already
    # fails the favorable, charged-MDL, capacity, or strict-read gates.
    controls = []
    run_controls = (
        not hard_kill
        and best_read_charged is not None
        and best_read_charged["two_part_gain_bpw"] >= TARGET_GAIN_BPW
    )
    if run_controls:
        for seed in CONTROL_SEEDS:
            shuffled = coordinate_scramble_gpu(labels, seed)
            entry = {"seed": seed}
            entry["binary_favorable_oracle"] = score_labels_gpu(
                shuffled, 2, int(io["scale_bits"]), "favorable"
            )
            entry["binary_charged_mdl"] = score_labels_gpu(
                shuffled, 2, int(io["scale_bits"]), "charged"
            )
            entry["quaternary"] = score_labels_gpu(
                shuffled, 4, int(io["scale_bits"])
            )
            controls.append(entry)
            del shuffled

    control_best = []
    for entry in controls:
        feasible_control_rows = []
        for name in read_eligible_candidates:
            row_name = (
                "binary_favorable_oracle"
                if name == "binary_charged_mdl" else "quaternary"
            )
            feasible_control_rows.append(
                float(entry[row_name]["favorable_gross_gain_bpw"])
            )
        control_best.append(max(feasible_control_rows))
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="strict")
    control_summary = ({
        "minimum": min(control_best),
        "median": float(np.median(control_best)),
        "maximum": max(control_best),
        "source_minus_control_median": (
            best_read_favorable["favorable_gross_gain_bpw"]
            - float(np.median(control_best))
        ),
    } if control_best else None)
    status, finite_eligible = _final_disposition(
        favorable_below_target=hard_kill,
        read_eligible_charged_gain_bpw=(
            None if best_read_charged is None
            else float(best_read_charged["two_part_gain_bpw"])
        ),
        control_corrected_gain_bpw=(
            None if control_summary is None
            else float(control_summary["source_minus_control_median"])
        ),
    )
    return {
        "schema": "same_layer_common_latent_entropy_gate_result_v0",
        "status": status,
        "claim_boundary": "IDEAL_LABEL_MDL_APERTURE_ONLY_NOT_A_FINITE_CODEC",
        "panel_lock_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "target_gain_bpw_up_down": TARGET_GAIN_BPW,
        "triage_gain_bpw_nonpromoting": 0.045,
        "best_favorable_variant": best_name,
        "best_favorable_gross_gain_bpw": best["favorable_gross_gain_bpw"],
        "triage_threshold_reached_nonpromoting": best["favorable_gross_gain_bpw"] >= 0.045,
        "best_charged_variant": best_charged_name,
        "best_charged_two_part_gain_bpw": best_charged["two_part_gain_bpw"],
        "read_eligible_rate_endpoints": eligible_rate_endpoints,
        "best_read_eligible_charged_variant": best_read_name,
        "best_read_eligible_charged_two_part_gain_bpw": (
            None if best_read_charged is None
            else best_read_charged["two_part_gain_bpw"]
        ),
        "eligible_for_finite_coder_research": finite_eligible,
        "hard_kill_before_controls_and_finite_coder": hard_kill,
        "variants": {name: _jsonable_score(score) for name, score in variants.items()},
        "controls_run": len(controls),
        "controls": controls,
        "control_best_gain_summary": control_summary,
        "physical_page_envelopes": envelopes,
        "input_read_ledger": io,
        "cuda": {
            "cupy_version": cp.__version__,
            "device_id": int(cp.cuda.runtime.getDevice()),
            "device_name": str(device_name),
        },
    }
