"""Frozen CuPy runner for the SILWARP auxiliary ideal-channel gate.

The default ``preflight`` command is source-free and imports no CuPy.  The
``run`` command is deliberately auxiliary-only: its path firewall rejects the
pinned panel, and confirmation bytes are not opened unless the preregistered
calibration promotion rule passes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import silwarp_common as sw


@dataclass
class MatrixRecord:
    key: tuple[int, int, str]
    path: Path
    source_sha256: str
    source_tiles: np.ndarray
    controls: dict[str, np.ndarray]
    features: np.ndarray | None
    role_indices: np.ndarray
    serialized_mean: float
    serialized_rms: float
    exact_centered_rms: float
    normalized_second_moment: float
    source_energy: float
    control_energy: dict[str, float]
    start: int = 0
    stop: int = 0


@dataclass
class Dataset:
    records: list[MatrixRecord]
    sources: dict[str, np.ndarray]
    features: np.ndarray
    roles: np.ndarray
    loss_weights: np.ndarray


@dataclass
class AuthenticatedRecordPayload:
    key: tuple[int, int, str]
    path: Path
    source_sha256: str
    payload: bytes


def decode_bf16_matrix(payload: bytes, role: str, source_name: str) -> np.ndarray:
    words = np.frombuffer(payload, dtype="<u2")
    if words.size != sw.WEIGHTS_PER_ROLE:
        raise ValueError(f"wrong BF16 element count: {source_name}")
    values = (words.astype(np.uint32) << np.uint32(16)).view("<f4")
    sw.require_all_finite(f"BF16 source {source_name}", values)
    if role == "down":
        return values.reshape(sw.MATRIX_COLS, sw.MATRIX_ROWS).T.copy()
    return values.reshape(sw.MATRIX_ROWS, sw.MATRIX_COLS).copy()


def matrix_tiles(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (sw.MATRIX_ROWS, sw.MATRIX_COLS):
        raise ValueError("canonical matrix shape mismatch")
    return (
        matrix.reshape(sw.TILE_ROWS, sw.TILE_SIDE, sw.TILE_COLS, sw.TILE_SIDE)
        .transpose(0, 2, 1, 3)
        .reshape(sw.TILE_ROWS * sw.TILE_COLS, sw.TILE_VALUES)
        .astype(np.float32, copy=False)
    )


def _exact_unit_gaussian(shape: tuple[int, ...], seed: int) -> np.ndarray:
    values = sw.counter_standard_normal(
        (math.prod(shape),), "gaussian-control", seed, float64=True
    )
    values -= np.mean(values, dtype=np.float64)
    # A second subtraction removes the residual of the first finite sum.
    values -= np.mean(values, dtype=np.float64)
    rms = math.sqrt(float(np.mean(values * values, dtype=np.float64)))
    if not rms > 0.0:
        raise AssertionError("degenerate Gaussian control")
    values /= rms
    return values.reshape(shape)


def require_identical_fp16_bits(name: str, control: Any, source: Any) -> None:
    control16 = np.float16(control)
    source16 = np.float16(source)
    if control16.tobytes() != source16.tobytes():
        raise AssertionError(f"control serialized {name} mismatch")


def moment_matched_control(
    key: tuple[int, int, str],
    control_name: str,
    moments: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    gaussian = _exact_unit_gaussian(
        (sw.MATRIX_ROWS, sw.MATRIX_COLS),
        sw.derive_seed("control-source", control_name, *key),
    )
    mean16 = float(moments["serialized_mean_fp16"])
    rms16 = float(moments["serialized_rms_fp16"])
    previous_rms16 = float(
        np.nextafter(
            np.float16(rms16), np.float16(0.0), dtype=np.float16
        )
    )
    # Pick the interior of the same serialized FP16 scale cell.  This leaves
    # ample post-FP32 normalization headroom while preserving exact charged
    # mean/RMS metadata even when the Qwen source needed an extra safety ULP.
    control_exact_rms = 0.5 * (previous_rms16 + rms16)
    if not 0.0 < control_exact_rms <= rms16:
        raise AssertionError("invalid control RMS cell interior")
    raw = mean16 + control_exact_rms * gaussian
    sw.require_all_finite(f"{control_name} raw control", raw)
    control_moments = sw.upward_fp16_moments(raw)
    require_identical_fp16_bits(
        "mean",
        control_moments["serialized_mean_fp16"],
        moments["serialized_mean_fp16"],
    )
    require_identical_fp16_bits(
        "RMS",
        control_moments["serialized_rms_fp16"],
        moments["serialized_rms_fp16"],
    )
    normalized = sw.normalize_with_serialized_moments(raw, control_moments)
    sw.require_all_finite(f"{control_name} normalized control", normalized)
    second_moment = float(np.mean(normalized.astype(np.float64) ** 2))
    if second_moment > 1.0:
        raise AssertionError("control normalized second moment exceeds one")
    energy = sw.require_finite_scalar(
        f"{control_name} source energy", np.sum(raw * raw, dtype=np.float64)
    )
    return matrix_tiles(normalized), energy


def authenticate_record_payloads(
    paths: Mapping[tuple[int, int, str], Path],
    split_name: str,
    locked_rows: Mapping[str, Mapping[str, Any]],
) -> list[AuthenticatedRecordPayload]:
    authenticated = []
    for key, path in sorted(paths.items()):
        locked = locked_rows[path.name]
        if locked["split"] != split_name:
            raise ValueError("source lock split mismatch before decode")
        payload, source_sha = sw.read_authenticated_locked_file(path, locked)
        authenticated.append(
            AuthenticatedRecordPayload(key, path, source_sha, payload)
        )
    return authenticated


def decode_authenticated_records(
    authenticated: Iterable[AuthenticatedRecordPayload], split_name: str
) -> list[MatrixRecord]:
    records = []
    for authenticated_record in authenticated:
        key = authenticated_record.key
        path = authenticated_record.path
        source_sha = authenticated_record.source_sha256
        layer, expert, role = key
        source = decode_bf16_matrix(
            authenticated_record.payload, role, path.name
        )
        source_energy = sw.require_finite_scalar(
            f"source energy {path.name}",
            np.sum(source.astype(np.float64) ** 2, dtype=np.float64),
        )
        moments = sw.upward_fp16_moments(source)
        normalized = sw.normalize_with_serialized_moments(source, moments)
        controls: dict[str, np.ndarray] = {}
        control_energy: dict[str, float] = {}
        for control_name in sw.CONTROL_NAMES:
            control, energy = moment_matched_control(key, control_name, moments)
            controls[control_name] = control
            control_energy[control_name] = energy
        tile_count = sw.TILE_ROWS * sw.TILE_COLS
        records.append(
            MatrixRecord(
                key=key,
                path=path,
                source_sha256=source_sha,
                source_tiles=matrix_tiles(normalized),
                controls=controls,
                features=None,
                role_indices=np.full(tile_count, sw.ROLE_INDEX[role], dtype=np.int64),
                serialized_mean=float(moments["serialized_mean_fp16"]),
                serialized_rms=float(moments["serialized_rms_fp16"]),
                exact_centered_rms=float(moments["centered_rms_fp64"]),
                normalized_second_moment=float(
                    moments["normalized_second_moment_fp64"]
                ),
                source_energy=source_energy,
                control_energy=control_energy,
            )
        )
        print(
            json.dumps(
                {
                    "event": "source_loaded",
                    "split": split_name,
                    "key": key,
                    "sha256": source_sha,
                    "normalized_second_moment": records[-1].normalized_second_moment,
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
    return records


def load_records(
    paths: Mapping[tuple[int, int, str], Path],
    split_name: str,
    locked_rows: Mapping[str, Mapping[str, Any]],
) -> list[MatrixRecord]:
    """Authenticate a complete split first, then decode those exact buffers."""

    authenticated = authenticate_record_payloads(paths, split_name, locked_rows)
    return decode_authenticated_records(authenticated, split_name)


def fit_log_rms_normalizer(records: Iterable[MatrixRecord]) -> tuple[float, float]:
    values = np.asarray(
        [math.log(record.serialized_rms) for record in records], dtype=np.float64
    )
    center = np.float16(float(np.mean(values, dtype=np.float64)))
    scale = np.float16(float(np.std(values, dtype=np.float64)))
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid FP16 log-RMS normalizer")
    return float(center), float(scale)


def attach_features(
    records: Iterable[MatrixRecord], log_rms_center: float, log_rms_scale: float
) -> None:
    tile_rows = np.repeat(np.arange(sw.TILE_ROWS), sw.TILE_COLS)
    tile_cols = np.tile(np.arange(sw.TILE_COLS), sw.TILE_ROWS)
    for record in records:
        layer, expert, role = record.key
        record.features = sw.coordinate_features(
            layer,
            expert,
            role,
            tile_rows,
            tile_cols,
            record.serialized_rms,
            log_rms_center,
            log_rms_scale,
        )


def combine_records(records: list[MatrixRecord]) -> Dataset:
    offset = 0
    for record in records:
        record.start = offset
        offset += len(record.source_tiles)
        record.stop = offset
        if record.features is None:
            raise AssertionError("record features were not attached")
    sources = {
        "qwen": np.concatenate([record.source_tiles for record in records]),
        **{
            control: np.concatenate([record.controls[control] for record in records])
            for control in sw.CONTROL_NAMES
        },
    }
    return Dataset(
        records=records,
        sources=sources,
        features=np.concatenate([record.features for record in records]),
        roles=np.concatenate([record.role_indices for record in records]),
        loss_weights=np.concatenate(
            [
                np.full(
                    len(record.source_tiles),
                    record.serialized_rms * record.serialized_rms,
                    dtype=np.float32,
                )
                for record in records
            ]
        ),
    )


def model_to_cpu(params: Mapping[str, Any], cp: Any) -> dict[str, np.ndarray]:
    return {name: cp.asnumpy(params[name]) for name in sw.PARAMETER_ORDER}


def rounded_model_on_gpu(
    params: Mapping[str, Any],
    training_seed: int,
    log_rms_center: float,
    log_rms_scale: float,
    cp: Any,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    blob = sw.serialize_model_bytes(
        model_to_cpu(params, cp), training_seed, log_rms_center, log_rms_scale
    )
    rounded, header = sw.deserialize_model_bytes(blob)
    return {name: cp.asarray(value) for name, value in rounded.items()}, blob, header


def evaluation_noise(
    record: MatrixRecord, corpus: str, split_name: str
) -> np.ndarray:
    return sw.counter_standard_normal(
        record.source_tiles.shape,
        "evaluation-channel",
        corpus,
        split_name,
        *record.key,
        float64=True,
    )


def training_batch_indices(
    item_count: int, training_seed: int, update: int, xp: Any = np
) -> Any:
    """One paired batch shared exactly by Qwen and both controls."""

    return sw.counter_indices(
        item_count, 512, "training-batch", training_seed, update, xp=xp
    )


def training_channel_noise(
    shape: tuple[int, ...],
    training_seed: int,
    corpus: str,
    update: int,
    xp: Any = np,
) -> Any:
    if corpus not in ("qwen", *sw.CONTROL_NAMES):
        raise ValueError("unknown training corpus")
    return sw.counter_standard_normal(
        shape,
        "training-channel",
        training_seed,
        corpus,
        update,
        xp=xp,
    )


def evaluate_corpus(
    dataset: Dataset,
    corpus: str,
    params: Mapping[str, Any],
    split_name: str,
    cp: Any,
    batch_tiles: int = 512,
) -> dict[str, Any]:
    source_all = dataset.sources[corpus]
    rows = []
    selected_total = 0.0
    identity_total = 0.0
    source_energy_total = 0.0
    for record in dataset.records:
        source = source_all[record.start : record.stop]
        noise = evaluation_noise(record, corpus, split_name)
        y = sw.ideal_awgn_mc_channel(source, noise)
        learned_sse_norm = 0.0
        identity_sse_norm = 0.0
        for start in range(0, len(source), batch_tiles):
            stop = min(start + batch_tiles, len(source))
            x_gpu = cp.asarray(source[start:stop])
            y_gpu = cp.asarray(y[start:stop])
            feature_gpu = cp.asarray(record.features[start:stop])
            roles_gpu = cp.asarray(record.role_indices[start:stop])
            decoded = sw.forward(
                params, y_gpu, feature_gpu, roles_gpu, xp=cp, return_cache=False
            )
            identity_error = y_gpu.astype(cp.float64) - x_gpu.astype(cp.float64)
            learned_error = decoded.astype(cp.float64) - x_gpu.astype(cp.float64)
            identity_sse_norm += sw.require_finite_scalar(
                "identity batch SSE", cp.sum(identity_error * identity_error).item()
            )
            learned_sse_norm += sw.require_finite_scalar(
                "learned batch SSE", cp.sum(learned_error * learned_error).item()
            )
        scale2 = record.serialized_rms * record.serialized_rms
        identity_sse = sw.require_finite_scalar(
            "identity matrix SSE", identity_sse_norm * scale2
        )
        learned_sse = sw.require_finite_scalar(
            "learned matrix SSE", learned_sse_norm * scale2
        )
        if identity_sse <= 0.0 or learned_sse <= 0.0:
            raise ValueError("nonpositive evaluation SSE")
        bypass = learned_sse >= identity_sse
        selected_sse = identity_sse if bypass else learned_sse
        if corpus == "qwen":
            source_energy = record.source_energy
        else:
            source_energy = record.control_energy[corpus]
        selected_total += selected_sse
        identity_total += identity_sse
        source_energy_total += source_energy
        rows.append(
            {
                "layer": record.key[0],
                "expert": record.key[1],
                "role": record.key[2],
                "identity_sse": identity_sse,
                "learned_sse": learned_sse,
                "selected_sse": selected_sse,
                "source_energy": source_energy,
                "bypass_identity": bypass,
            }
        )
    ledger = sw.production_ledger(128)
    aggregate = sw.relative_metrics(
        source_energy_total,
        identity_total,
        selected_total,
        ledger["production_physical_bpw"],
    )
    return {
        "corpus": corpus,
        "split": split_name,
        "aggregate": aggregate,
        "identity_sse": identity_total,
        "selected_sse": selected_total,
        "source_energy": source_energy_total,
        "matrices": rows,
    }


def _pooled_s(rows: list[Mapping[str, Any]]) -> float:
    selected = sum(float(row["selected_sse"]) for row in rows)
    identity = sum(float(row["identity_sse"]) for row in rows)
    selected = sw.require_finite_scalar("pooled selected SSE", selected)
    identity = sw.require_finite_scalar("pooled identity SSE", identity)
    if selected <= 0.0 or identity <= 0.0:
        raise ValueError("nonpositive pooled SSE")
    return sw.require_finite_scalar("pooled s", -0.5 * math.log2(selected / identity))


def _filter_rows(
    rows: list[Mapping[str, Any]], dimension: str, value: Any, exclude: bool
) -> list[Mapping[str, Any]]:
    if dimension == "pair":
        predicate = lambda row: (int(row["layer"]), int(row["expert"])) == value
    else:
        predicate = lambda row: int(row[dimension]) == int(value)
    return [row for row in rows if bool(predicate(row)) != bool(exclude)]


def matched_summary(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    s_values = {
        corpus: sw.require_finite_scalar(
            f"{corpus} absolute s", result["aggregate"]["s_absolute_from_identity"]
        )
        for corpus, result in results.items()
    }
    worst_control = max(sw.CONTROL_NAMES, key=lambda name: s_values[name])
    matched = sw.require_finite_scalar(
        "worst-null matched s", s_values["qwen"] - s_values[worst_control]
    )
    se_by_dimension: dict[str, float] = {}
    folds: dict[str, list[dict[str, Any]]] = {}
    q_rows = results["qwen"]["matrices"]
    for dimension in ("pair", "layer", "expert"):
        if dimension == "pair":
            labels = sorted({(int(row["layer"]), int(row["expert"])) for row in q_rows})
        else:
            labels = sorted({int(row[dimension]) for row in q_rows})
        estimates = []
        fold_rows = []
        for label in labels:
            s_fold = {}
            for corpus in ("qwen", *sw.CONTROL_NAMES):
                remaining = _filter_rows(
                    results[corpus]["matrices"], dimension, label, exclude=True
                )
                s_fold[corpus] = _pooled_s(remaining)
            fold_worst = max(sw.CONTROL_NAMES, key=lambda name: s_fold[name])
            estimate = sw.require_finite_scalar(
                f"{dimension} matched jackknife estimate",
                s_fold["qwen"] - s_fold[fold_worst],
            )
            estimates.append(estimate)
            fold_rows.append(
                {
                    "left_out": list(label) if isinstance(label, tuple) else label,
                    "s_by_corpus": s_fold,
                    "worst_control": fold_worst,
                    "s_match_worst": estimate,
                }
            )
        mean = sw.require_finite_scalar(
            f"{dimension} jackknife mean", sum(estimates) / len(estimates)
        )
        variance = (len(estimates) - 1.0) / len(estimates) * sum(
            (value - mean) ** 2 for value in estimates
        )
        se_by_dimension[dimension] = sw.require_finite_scalar(
            f"{dimension} jackknife SE", math.sqrt(max(variance, 0.0))
        )
        folds[dimension] = fold_rows
    qwen_group_s = {"pair": {}, "layer": {}, "expert": {}}
    for row in q_rows:
        pass
    for dimension in qwen_group_s:
        if dimension == "pair":
            labels = sorted({(int(row["layer"]), int(row["expert"])) for row in q_rows})
        else:
            labels = sorted({int(row[dimension]) for row in q_rows})
        for label in labels:
            selected = _filter_rows(q_rows, dimension, label, exclude=False)
            key = f"{label[0]},{label[1]}" if isinstance(label, tuple) else str(label)
            qwen_group_s[dimension][key] = _pooled_s(selected)
    return {
        "s_by_corpus": s_values,
        "worst_control": worst_control,
        "s_match_worst": matched,
        "cluster_se_by_dimension": se_by_dimension,
        "cluster_se": max(se_by_dimension.values()),
        "jackknife_folds": folds,
        "qwen_group_s": qwen_group_s,
    }


def evaluate_seed(
    dataset: Dataset,
    states: Mapping[str, Mapping[str, Any]],
    training_seed: int,
    split_name: str,
    log_rms_center: float,
    log_rms_scale: float,
    cp: Any,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    corpus_results = {}
    model_blobs = {}
    model_headers = {}
    model_hashes = {}
    for corpus in ("qwen", *sw.CONTROL_NAMES):
        rounded, blob, header = rounded_model_on_gpu(
            states[corpus], training_seed, log_rms_center, log_rms_scale, cp
        )
        corpus_results[corpus] = evaluate_corpus(
            dataset, corpus, rounded, split_name, cp
        )
        model_blobs[corpus] = blob
        model_headers[corpus] = header
        model_hashes[corpus] = sw.sha256_bytes(blob)
    summary = matched_summary(corpus_results)
    return (
        {
            "training_seed": training_seed,
            "split": split_name,
            "models": {
                corpus: {
                    "sha256": model_hashes[corpus],
                    "bytes": len(model_blobs[corpus]),
                    "parameter_payload_sha256": model_headers[corpus][
                        "parameter_payload_sha256"
                    ],
                }
                for corpus in model_blobs
            },
            "corpora": corpus_results,
            "matched": summary,
        },
        model_blobs,
    )


def calibration_promotes(evaluations: Mapping[int, Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    required = sw.production_ledger(128)["required_absolute_s"]
    primary = evaluations[sw.TRAINING_SEEDS[0]]
    sensitivity = evaluations[sw.TRAINING_SEEDS[1]]
    primary_f = sw.require_finite_scalar(
        "primary calibration F",
        primary["corpora"]["qwen"]["aggregate"]["F_at_physical_rate"]
    )
    sensitivity_f = sw.require_finite_scalar(
        "sensitivity calibration F",
        sensitivity["corpora"]["qwen"]["aggregate"]["F_at_physical_rate"]
    )
    primary_lcb = sw.require_finite_scalar(
        "primary matched lower bound",
        float(primary["matched"]["s_match_worst"])
        - 2.0 * float(primary["matched"]["cluster_se"])
    )
    layer_s = [
        sw.require_finite_scalar("primary layer s", value)
        for value in primary["matched"]["qwen_group_s"]["layer"].values()
    ]
    pair_s = [
        sw.require_finite_scalar("primary pair s", value)
        for value in primary["matched"]["qwen_group_s"]["pair"].values()
    ]
    if not layer_s or not pair_s:
        raise ValueError("promotion groups may not be empty")
    positive_layers = all(value > 0.0 for value in layer_s)
    positive_pairs = all(value > 0.0 for value in pair_s)
    sensitivity_matched = sw.require_finite_scalar(
        "sensitivity matched s", sensitivity["matched"]["s_match_worst"]
    )
    checks = {
        "primary_F_le_0_8": primary_f <= 0.8,
        "primary_worst_null_lcb_ge_required": primary_lcb >= required,
        "every_primary_layer_positive": positive_layers,
        "every_primary_pair_positive": positive_pairs,
        "sensitivity_F_le_0_8": sensitivity_f <= 0.8,
        "sensitivity_matched_positive": sensitivity_matched > 0.0,
    }
    return all(checks.values()), {
        "required_absolute_s": required,
        "primary_F": primary_f,
        "primary_worst_null_lcb": primary_lcb,
        "sensitivity_F": sensitivity_f,
        "checks": checks,
    }


def source_free_preflight() -> dict[str, Any]:
    protocol = sw.load_protocol()
    sw.validate_frozen_constants(protocol)
    sw.validate_split(protocol)
    source_lock = sw.load_source_lock(protocol=protocol)
    sw.validate_source_lock(protocol, source_lock)
    moment_probe = sw.upward_fp16_moments(np.asarray([-1.0001, 1.0001]))
    params = sw.initialize_parameters(sw.TRAINING_SEEDS[0])
    blob = sw.serialize_model_bytes(params, sw.TRAINING_SEEDS[0], -3.5, 0.75)
    rounded, header = sw.deserialize_model_bytes(blob)
    y = sw.counter_standard_normal((2, 256), "source-free-preflight", 7)
    decoded = sw.forward(
        rounded,
        y,
        np.zeros((2, 21), dtype=np.float32),
        np.asarray([0, 2], dtype=np.int64),
    )
    if not np.array_equal(y, decoded):
        raise AssertionError("identity preflight failed")
    return {
        "schema": "silwarp_source_free_preflight_v2",
        "protocol_sha256": sw.protocol_sha256(),
        "source_lock_sha256": sw.source_lock_sha256(),
        "source_lock_internal_seal_sha256": source_lock[
            "internal_seal_sha256"
        ],
        "source_revision": source_lock["checkpoint"]["revision"],
        "runner_sha256": sw.sha256_file(Path(__file__).resolve()),
        "common_sha256": sw.sha256_file(Path(sw.__file__).resolve()),
        "payload_opened": False,
        "cuda_imported": False,
        "analytic_design_information_bpw": sw.information_upper_bound_bpw(sw.CHANNEL_D),
        "implemented_information_bound_bpw": sw.channel_second_moments()[
            "information_upper_bound_bpw"
        ],
        "physical_role_payload_bpw": sw.ROLE_PAYLOAD_BPW,
        "upward_rms_probe": {
            "exact": moment_probe["centered_rms_fp64"],
            "serialized": float(moment_probe["serialized_rms_fp16"]),
            "normalized_second_moment": moment_probe[
                "normalized_second_moment_fp64"
            ],
        },
        "model_bytes": len(blob),
        "model_sha256": sw.sha256_bytes(blob),
        "parameter_payload_sha256": header["parameter_payload_sha256"],
        "identity_exact": True,
        "ledger_128": sw.production_ledger(128),
        "ledger_6": sw.production_ledger(6),
        "fixed_training_seeds": list(sw.TRAINING_SEEDS),
        "independent_controls": list(sw.CONTROL_NAMES),
        "status": "PASS_SOURCE_FREE_PREFLIGHT",
    }


def gpu_source_free_preflight() -> dict[str, Any]:
    """Exercise the CuPy path without inventorying or opening any tensor payload."""

    protocol = sw.load_protocol()
    sw.validate_frozen_constants(protocol)
    sw.validate_split(protocol)
    source_lock = sw.load_source_lock(protocol=protocol)
    sw.validate_source_lock(protocol, source_lock)
    import cupy as cp  # type: ignore

    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("frozen SILWARP cell requires exactly one visible CUDA device")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    runtime = runtime_identity(cp, properties, 0)
    spec = sw.SmallSpec(values=8, features=5, hidden=12, bottleneck=6, steps=2)
    params = sw.initialize_parameters(sw.TRAINING_SEEDS[0], spec=spec, xp=cp)
    source = sw.counter_standard_normal((16, spec.values), "gpu-preflight-source", xp=cp)
    noise = sw.counter_standard_normal((16, spec.values), "gpu-preflight-noise", xp=cp)
    features = sw.counter_standard_normal(
        (16, spec.features), "gpu-preflight-features", xp=cp
    )
    roles = (cp.arange(16, dtype=cp.int64) % spec.roles).astype(cp.int64)
    weights = cp.linspace(cp.float32(0.25), cp.float32(2.0), 16, dtype=cp.float32)
    y = sw.gaussian_rdf_channel(source, noise, xp=cp)
    decoded = sw.forward(params, y, features, roles, spec=spec, xp=cp)
    if not bool(cp.array_equal(decoded, y).item()):
        raise AssertionError("GPU exact-identity path failed")
    loss, gradients = sw.mse_loss_and_gradients(
        params,
        y,
        source,
        features,
        roles,
        sample_weights=weights,
        spec=spec,
        xp=cp,
    )
    optimizer = sw.Adam(params, xp=cp, learning_rate=5e-4)
    optimizer.update(params, gradients)
    indices_a = sw.counter_indices(
        997, 512, "gpu-preflight-indices", sw.TRAINING_SEEDS[0], 1, xp=cp
    )
    indices_b = sw.counter_indices(
        997, 512, "gpu-preflight-indices", sw.TRAINING_SEEDS[0], 1, xp=cp
    )
    if not bool(cp.array_equal(indices_a, indices_b).item()):
        raise AssertionError("GPU counter replay failed")
    cpu_indices = sw.counter_indices(
        997, 512, "gpu-preflight-indices", sw.TRAINING_SEEDS[0], 1
    )
    if not np.array_equal(cp.asnumpy(indices_a), cpu_indices):
        raise AssertionError("integer counter differs across CPU/GPU")

    # Production-shape feasibility cell: one warmup plus three fully weighted
    # forward/backward/Adam steps at the frozen batch size.
    production_cells = []
    for qualification_seed in sw.TRAINING_SEEDS:
        for qualification_corpus in ("qwen", *sw.CONTROL_NAMES):
            cell_params = sw.initialize_parameters(qualification_seed, xp=cp)
            production_cells.append(
                (
                    qualification_seed,
                    qualification_corpus,
                    cell_params,
                    sw.Adam(cell_params, xp=cp, learning_rate=5e-4),
                )
            )
    production_params = production_cells[0][2]
    production_optimizer = production_cells[0][3]
    production_source = sw.counter_standard_normal(
        (512, sw.PRODUCTION_SPEC.values), "gpu-production-source", xp=cp
    )
    production_features = sw.counter_standard_normal(
        (512, sw.PRODUCTION_SPEC.features), "gpu-production-features", xp=cp
    )
    production_roles = (
        cp.arange(512, dtype=cp.int64) % sw.PRODUCTION_SPEC.roles
    ).astype(cp.int64)
    production_weights = cp.linspace(
        cp.float32(0.25), cp.float32(2.0), 512, dtype=cp.float32
    )
    memory_pool = cp.get_default_memory_pool()

    def production_step(step_index: int) -> float:
        production_noise = sw.counter_standard_normal(
            production_source.shape,
            "gpu-production-noise",
            step_index,
            xp=cp,
        )
        production_y = sw.gaussian_rdf_channel(
            production_source, production_noise, xp=cp
        )
        production_loss, production_gradients = sw.mse_loss_and_gradients(
            production_params,
            production_y,
            production_source,
            production_features,
            production_roles,
            sample_weights=production_weights,
            xp=cp,
        )
        production_optimizer.update(production_params, production_gradients)
        return sw.require_finite_scalar(
            "GPU production qualification loss", production_loss
        )

    production_step(0)
    cp.cuda.Stream.null.synchronize()
    durations = []
    pool_total_peak = int(memory_pool.total_bytes())
    pool_used_peak = int(memory_pool.used_bytes())
    final_production_loss = None
    for step_index in range(1, 4):
        start = time.perf_counter()
        final_production_loss = production_step(step_index)
        cp.cuda.Stream.null.synchronize()
        durations.append(time.perf_counter() - start)
        pool_total_peak = max(pool_total_peak, int(memory_pool.total_bytes()))
        pool_used_peak = max(pool_used_peak, int(memory_pool.used_bytes()))
    cp.cuda.Stream.null.synchronize()
    receipt = {
        "schema": "silwarp_gpu_source_free_preflight_v2",
        "protocol_sha256": sw.protocol_sha256(),
        "source_lock_sha256": sw.source_lock_sha256(),
        "runner_sha256": sw.sha256_file(Path(__file__).resolve()),
        "common_sha256": sw.sha256_file(Path(sw.__file__).resolve()),
        "payload_opened": False,
        "cuda_imported": True,
        "runtime_identity": runtime,
        "weighted_loss": sw.require_finite_scalar("GPU preflight loss", loss),
        "production_feasibility": {
            "spec": {
                "values": sw.PRODUCTION_SPEC.values,
                "features": sw.PRODUCTION_SPEC.features,
                "hidden": sw.PRODUCTION_SPEC.hidden,
                "bottleneck": sw.PRODUCTION_SPEC.bottleneck,
                "steps": sw.PRODUCTION_SPEC.steps,
            },
            "batch_tiles": 512,
            "resident_model_adam_cells": len(production_cells),
            "warmup_steps": 1,
            "timed_steps": 3,
            "step_seconds": durations,
            "median_step_seconds": float(np.median(durations)),
            "total_timed_seconds": float(sum(durations)),
            "projected_1536_updates_six_models_seconds": float(
                np.median(durations) * 1536 * 6
            ),
            "memory_pool_total_bytes_peak_observed": pool_total_peak,
            "memory_pool_used_bytes_peak_observed": pool_used_peak,
            "final_loss": final_production_loss,
        },
        "identity_exact": True,
        "counter_replay_exact": True,
        "status": "PASS_GPU_SOURCE_FREE_PREFLIGHT",
    }
    sw.require_json_finite("GPU source-free receipt", receipt)
    return receipt


def run_bindings(sentinel_path: Path) -> dict[str, Any]:
    return {
        "protocol_sha256": sw.protocol_sha256(),
        "source_lock_sha256": sw.source_lock_sha256(),
        "runner_sha256": sw.sha256_file(Path(__file__).resolve()),
        "common_sha256": sw.sha256_file(Path(sw.__file__).resolve()),
        "launch_sentinel_sha256": sw.sha256_file(sentinel_path),
    }


def runtime_identity(cp: Any, properties: Mapping[str, Any], device_id: int) -> dict[str, Any]:
    name = properties.get("name", "")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="strict")
    uuid = properties.get("uuid")
    if isinstance(uuid, bytes):
        uuid_value = uuid.hex()
    elif uuid is None:
        uuid_value = None
    else:
        try:
            uuid_value = bytes(uuid).hex()
        except (TypeError, ValueError):
            uuid_value = str(uuid)
    identity = {
        "python": sys.version,
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        "device_id": int(device_id),
        "device_name": str(name),
        "compute_capability": str(cp.cuda.Device(device_id).compute_capability),
        "property_major": int(properties["major"]),
        "property_minor": int(properties["minor"]),
        "device_uuid": uuid_value,
    }
    sw.require_json_finite("runtime identity", identity)
    return identity


def _expected_checkpoint_updates(update: int) -> tuple[int, ...]:
    if update not in sw.CHECKPOINTS:
        raise ValueError("checkpoint update is not frozen")
    return tuple(point for point in sw.CHECKPOINTS if point <= update)


def _validate_checkpoint_history(
    history: Mapping[int, Mapping[int, Mapping[str, Any]]], update: int
) -> None:
    if set(history) != set(sw.TRAINING_SEEDS):
        raise ValueError("checkpoint history seed schema mismatch")
    expected_points = set(_expected_checkpoint_updates(update))
    for seed in sw.TRAINING_SEEDS:
        points = history[seed]
        if set(points) != expected_points:
            raise ValueError("checkpoint history update schema mismatch")
        for point, evaluation in points.items():
            sw.require_json_finite(f"checkpoint history {seed}/{point}", evaluation)
            sw.require_finite_scalar(
                "checkpoint matched s", evaluation["matched"]["s_match_worst"]
            )
            sw.require_finite_scalar(
                "checkpoint cluster SE", evaluation["matched"]["cluster_se"]
            )
            sw.require_finite_scalar(
                "checkpoint Qwen F",
                evaluation["corpora"]["qwen"]["aggregate"]["F_at_physical_rate"],
            )


def _checkpoint_predecessor(
    output_dir: Path, update: int
) -> dict[str, Any] | None:
    expected = _expected_checkpoint_updates(update)
    if len(expected) == 1:
        return None
    predecessor_update = expected[-2]
    predecessor = output_dir / f"checkpoint_{predecessor_update:06d}"
    metadata_path = predecessor / "checkpoint.json"
    state_path = predecessor / "state.npz"
    if predecessor.is_symlink() or not predecessor.is_dir():
        raise ValueError("checkpoint predecessor directory is missing or nonregular")
    for member in (metadata_path, state_path):
        if member.is_symlink() or not member.is_file():
            raise ValueError("checkpoint predecessor member is missing or nonregular")
    return {
        "update": predecessor_update,
        "metadata_sha256": sw.sha256_file(metadata_path),
        "state_sha256": sw.sha256_file(state_path),
    }


def save_training_checkpoint(
    output_dir: Path,
    update: int,
    states: Mapping[int, Mapping[str, Mapping[str, Any]]],
    optimizers: Mapping[int, Mapping[str, sw.Adam]],
    history: Mapping[int, Mapping[int, Mapping[str, Any]]],
    bindings: Mapping[str, Any],
    log_rms_center: float,
    log_rms_scale: float,
    cp: Any,
) -> Path:
    expected_updates = _expected_checkpoint_updates(update)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("checkpoint output must be a regular directory")
    pending_paths = list(output_dir.glob("checkpoint_*.pending"))
    if pending_paths:
        raise ValueError("incomplete checkpoint blocks append-only save")
    checkpoint_entries = list(
        output_dir.glob("checkpoint_[0-9][0-9][0-9][0-9][0-9][0-9]")
    )
    if any(path.is_symlink() or not path.is_dir() for path in checkpoint_entries):
        raise ValueError("checkpoint entry must be a regular directory")
    existing_updates = {
        int(path.name.rsplit("_", 1)[1]) for path in checkpoint_entries
    }
    if existing_updates != set(expected_updates[:-1]):
        raise ValueError("append-only checkpoint predecessor set mismatch")
    _validate_checkpoint_history(history, update)
    log_rms_center = sw.require_finite_scalar(
        "checkpoint log-RMS center", log_rms_center
    )
    log_rms_scale = sw.require_finite_scalar(
        "checkpoint log-RMS scale", log_rms_scale
    )
    if log_rms_scale <= 0.0:
        raise ValueError("checkpoint log-RMS scale must be positive")
    predecessor = _checkpoint_predecessor(output_dir, update)
    final = output_dir / f"checkpoint_{update:06d}"
    pending = output_dir / f"checkpoint_{update:06d}.pending"
    if final.exists() or pending.exists():
        raise FileExistsError(f"append-only checkpoint already exists: {update}")
    pending.mkdir()
    arrays: dict[str, np.ndarray] = {}
    for seed in sw.TRAINING_SEEDS:
        for corpus in ("qwen", *sw.CONTROL_NAMES):
            optimizer = optimizers[seed][corpus]
            if optimizer.step != update:
                raise AssertionError("Adam/checkpoint update mismatch")
            for name in sw.PARAMETER_ORDER:
                for prefix, value in (
                    ("p", states[seed][corpus][name]),
                    ("m", optimizer.m[name]),
                    ("v", optimizer.v[name]),
                ):
                    array = cp.asnumpy(value).astype(np.float32, copy=False)
                    sw.require_all_finite(
                        f"checkpoint {prefix}/{seed}/{corpus}/{name}", array
                    )
                    arrays[f"{prefix}__{seed}__{corpus}__{name}"] = array
    expected_array_count = (
        3
        * len(sw.TRAINING_SEEDS)
        * (1 + len(sw.CONTROL_NAMES))
        * len(sw.PARAMETER_ORDER)
    )
    if len(arrays) != expected_array_count:
        raise AssertionError("checkpoint state-array count mismatch")
    state_path = pending / "state.npz"
    np.savez(state_path, **arrays)
    state_hash = sw.sha256_file(state_path)
    metadata: dict[str, Any] = {
        "schema": "silwarp_training_checkpoint_v2",
        "update": update,
        "bindings": dict(bindings),
        "state_file": state_path.name,
        "state_sha256": state_hash,
        "state_arrays": len(arrays),
        "adam_step": update,
        "log_rms_center_fp16": log_rms_center,
        "log_rms_scale_fp16": log_rms_scale,
        "predecessor": predecessor,
        "history": {
            str(seed): {str(point): value for point, value in points.items()}
            for seed, points in history.items()
        },
        "counter_randomness": True,
    }
    sw.require_json_finite("checkpoint metadata", metadata)
    metadata["internal_seal_sha256"] = sw.sha256_bytes(
        sw.canonical_json_bytes(metadata)
    )
    metadata_path = pending / "checkpoint.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(final)
    return final


def restore_latest_checkpoint(
    output_dir: Path,
    states: Mapping[int, Mapping[str, Mapping[str, Any]]],
    optimizers: Mapping[int, Mapping[str, sw.Adam]],
    bindings: Mapping[str, Any],
    log_rms_center: float,
    log_rms_scale: float,
    cp: Any,
) -> tuple[int, dict[int, dict[int, dict[str, Any]]]]:
    pending = list(output_dir.glob("checkpoint_*.pending"))
    if pending:
        raise ValueError(f"incomplete checkpoint(s) present: {[p.name for p in pending]}")
    candidates = []
    for path in output_dir.glob("checkpoint_[0-9][0-9][0-9][0-9][0-9][0-9]"):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("checkpoint must be a regular directory")
        update = int(path.name.rsplit("_", 1)[1])
        if update not in sw.CHECKPOINTS:
            raise ValueError("unexpected checkpoint update")
        candidates.append((update, path))
    if not candidates:
        raise FileNotFoundError("resume requested but no complete checkpoint exists")
    update, checkpoint = max(candidates)
    expected_updates = _expected_checkpoint_updates(update)
    if {point for point, _ in candidates} != set(expected_updates):
        raise ValueError("checkpoint chain is incomplete")
    expected_array_count = (
        3
        * len(sw.TRAINING_SEEDS)
        * (1 + len(sw.CONTROL_NAMES))
        * len(sw.PARAMETER_ORDER)
    )
    previous_descriptor = None
    metadata = None
    state_path = None
    for chain_update in expected_updates:
        chain_dir = output_dir / f"checkpoint_{chain_update:06d}"
        metadata_path = chain_dir / "checkpoint.json"
        chain_state_path = chain_dir / "state.npz"
        for path in (metadata_path, chain_state_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError("checkpoint member must be a regular non-symlink file")
        chain_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        seal = chain_metadata.get("internal_seal_sha256")
        unsigned = dict(chain_metadata)
        unsigned.pop("internal_seal_sha256", None)
        if seal != sw.sha256_bytes(sw.canonical_json_bytes(unsigned)):
            raise ValueError("checkpoint internal seal mismatch")
        exact_keys = {
            "schema", "update", "bindings", "state_file", "state_sha256",
            "state_arrays", "adam_step", "log_rms_center_fp16",
            "log_rms_scale_fp16", "predecessor", "history",
            "counter_randomness", "internal_seal_sha256",
        }
        if set(chain_metadata) != exact_keys:
            raise ValueError("checkpoint metadata schema mismatch")
        if chain_metadata.get("schema") != "silwarp_training_checkpoint_v2":
            raise ValueError("checkpoint schema mismatch")
        if int(chain_metadata.get("update", -1)) != chain_update:
            raise ValueError("checkpoint update mismatch")
        if chain_metadata.get("bindings") != dict(bindings):
            raise ValueError("checkpoint binding mismatch")
        if chain_metadata.get("state_file") != "state.npz":
            raise ValueError("checkpoint state filename mismatch")
        if chain_metadata.get("state_sha256") != sw.sha256_file(chain_state_path):
            raise ValueError("checkpoint state hash mismatch")
        if int(chain_metadata.get("state_arrays", -1)) != expected_array_count:
            raise ValueError("checkpoint state-array count mismatch")
        if int(chain_metadata.get("adam_step", -1)) != chain_update:
            raise ValueError("checkpoint Adam step mismatch")
        if chain_metadata.get("counter_randomness") is not True:
            raise ValueError("checkpoint counter-randomness marker mismatch")
        center = sw.require_finite_scalar(
            "restored log-RMS center", chain_metadata.get("log_rms_center_fp16")
        )
        scale = sw.require_finite_scalar(
            "restored log-RMS scale", chain_metadata.get("log_rms_scale_fp16")
        )
        if center != log_rms_center or scale != log_rms_scale or scale <= 0.0:
            raise ValueError("checkpoint log-RMS normalizer mismatch")
        if chain_metadata.get("predecessor") != previous_descriptor:
            raise ValueError("checkpoint predecessor binding mismatch")
        raw_history = chain_metadata.get("history", {})
        if set(raw_history) != {str(seed) for seed in sw.TRAINING_SEEDS}:
            raise ValueError("checkpoint history seed schema mismatch")
        chain_history = {
            seed: {
                int(point): value
                for point, value in raw_history[str(seed)].items()
            }
            for seed in sw.TRAINING_SEEDS
        }
        _validate_checkpoint_history(chain_history, chain_update)
        previous_descriptor = {
            "update": chain_update,
            "metadata_sha256": sw.sha256_file(metadata_path),
            "state_sha256": sw.sha256_file(chain_state_path),
        }
        metadata = chain_metadata
        state_path = chain_state_path
    if metadata is None or state_path is None:
        raise AssertionError("checkpoint chain traversal failed")
    expected_names = {
        f"{prefix}__{seed}__{corpus}__{name}"
        for prefix in ("p", "m", "v")
        for seed in sw.TRAINING_SEEDS
        for corpus in ("qwen", *sw.CONTROL_NAMES)
        for name in sw.PARAMETER_ORDER
    }
    with np.load(state_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise ValueError("checkpoint state-array schema mismatch")
        for seed in sw.TRAINING_SEEDS:
            for corpus in ("qwen", *sw.CONTROL_NAMES):
                optimizer = optimizers[seed][corpus]
                for name in sw.PARAMETER_ORDER:
                    for prefix, target in (
                        ("p", states[seed][corpus]),
                        ("m", optimizer.m),
                        ("v", optimizer.v),
                    ):
                        key = f"{prefix}__{seed}__{corpus}__{name}"
                        array = archive[key].astype(np.float32, copy=False)
                        if array.shape != tuple(target[name].shape):
                            raise ValueError("checkpoint state shape mismatch")
                        sw.require_all_finite(f"restored state {key}", array)
                        target[name][...] = cp.asarray(array)
                optimizer.step = update
    raw_history = metadata["history"]
    history = {
        seed: {int(point): value for point, value in raw_history[str(seed)].items()}
        for seed in sw.TRAINING_SEEDS
    }
    return update, history


def run_gate(
    aux_dir: Path,
    output_dir: Path,
    launch_sentinel: Path,
    resume: bool,
) -> dict[str, Any]:
    protocol = sw.load_protocol()
    sw.validate_frozen_constants(protocol)
    sw.validate_split(protocol)
    source_lock = sw.load_source_lock(protocol=protocol)
    sw.validate_source_lock(protocol, source_lock)
    sentinel = sw.validate_launch_sentinel(
        launch_sentinel, Path(__file__).resolve(), Path(sw.__file__).resolve()
    )
    inventory = sw.inventory_auxiliary(aux_dir, protocol)
    locked_rows = sw.source_lock_rows(protocol)
    if resume:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("resume output must be an existing regular directory")
        if (output_dir / "result.json").exists():
            raise FileExistsError("run already has a final result")
    else:
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
        output_dir.mkdir(parents=True)
    # CuPy is imported only after protocol/path/inventory checks pass.
    import cupy as cp  # type: ignore

    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("frozen SILWARP cell requires exactly one visible CUDA device")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    runtime = runtime_identity(cp, properties, 0)
    bindings = run_bindings(launch_sentinel)
    bindings["runtime_identity"] = runtime

    # Authenticate the entire accessible fit+calibration set before the first
    # numeric BF16 decode, while retaining the exact descriptor-read buffers.
    fit_payloads = authenticate_record_payloads(
        inventory["fit"], "fit", locked_rows
    )
    calibration_payloads = authenticate_record_payloads(
        inventory["calibration"], "calibration", locked_rows
    )
    fit_records = decode_authenticated_records(fit_payloads, "fit")
    calibration_records = decode_authenticated_records(
        calibration_payloads, "calibration"
    )
    del fit_payloads, calibration_payloads
    log_rms_center, log_rms_scale = fit_log_rms_normalizer(fit_records)
    attach_features(fit_records, log_rms_center, log_rms_scale)
    attach_features(calibration_records, log_rms_center, log_rms_scale)
    fit = combine_records(fit_records)
    calibration = combine_records(calibration_records)

    fit_gpu = {corpus: cp.asarray(values) for corpus, values in fit.sources.items()}
    feature_gpu = cp.asarray(fit.features)
    roles_gpu = cp.asarray(fit.roles)
    loss_weight_gpu = cp.asarray(fit.loss_weights)
    states: dict[int, dict[str, dict[str, Any]]] = {}
    optimizers: dict[int, dict[str, sw.Adam]] = {}
    for training_seed in sw.TRAINING_SEEDS:
        states[training_seed] = {}
        optimizers[training_seed] = {}
        for corpus in ("qwen", *sw.CONTROL_NAMES):
            states[training_seed][corpus] = sw.initialize_parameters(
                training_seed, xp=cp
            )
            optimizers[training_seed][corpus] = sw.Adam(
                states[training_seed][corpus], xp=cp, learning_rate=5e-4
            )

    history: dict[int, dict[int, dict[str, Any]]] = {
        seed: {} for seed in sw.TRAINING_SEEDS
    }
    start_update = 0
    if resume:
        start_update, history = restore_latest_checkpoint(
            output_dir,
            states,
            optimizers,
            bindings,
            log_rms_center,
            log_rms_scale,
            cp,
        )
    final_blobs: dict[int, dict[str, bytes]] = {}
    stopped = False
    stop_update = None
    if start_update >= 512:
        stop_input = {
            seed: {
                point: {
                    "s_match_worst": history[seed][point]["matched"][
                        "s_match_worst"
                    ],
                    "cluster_se": history[seed][point]["matched"]["cluster_se"],
                }
                for point in (256, 512)
            }
            for seed in sw.TRAINING_SEEDS
        }
        if sw.hard_kill_at_512(stop_input):
            stopped = True
            stop_update = 512
    for update in range(start_update + 1, 1537):
        if stopped:
            break
        losses = {}
        for training_seed in sw.TRAINING_SEEDS:
            # Paired Qwen/null optimization uses exactly the same tile indices.
            indices = training_batch_indices(
                len(fit.features), training_seed, update, xp=cp
            )
            batch_features = feature_gpu[indices]
            batch_roles = roles_gpu[indices]
            batch_loss_weights = loss_weight_gpu[indices]
            for corpus in ("qwen", *sw.CONTROL_NAMES):
                source = fit_gpu[corpus][indices]
                noise = training_channel_noise(
                    tuple(source.shape),
                    training_seed,
                    corpus,
                    update,
                    xp=cp,
                )
                y = sw.gaussian_rdf_channel(source, noise, xp=cp)
                loss, gradients = sw.mse_loss_and_gradients(
                    states[training_seed][corpus],
                    y,
                    source,
                    batch_features,
                    batch_roles,
                    sample_weights=batch_loss_weights,
                    xp=cp,
                )
                optimizers[training_seed][corpus].update(
                    states[training_seed][corpus], gradients
                )
                losses[f"{training_seed}:{corpus}"] = sw.require_finite_scalar(
                    "reported training loss", loss
                )

        if update in sw.CHECKPOINTS:
            checkpoint_evaluations = {}
            for training_seed in sw.TRAINING_SEEDS:
                evaluation, blobs = evaluate_seed(
                    calibration,
                    states[training_seed],
                    training_seed,
                    "calibration",
                    log_rms_center,
                    log_rms_scale,
                    cp,
                )
                evaluation["update"] = update
                checkpoint_evaluations[training_seed] = evaluation
                history[training_seed][update] = evaluation
                final_blobs[training_seed] = blobs
            print(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "update": update,
                        "losses": losses,
                        "summary": {
                            str(seed): {
                                "F": checkpoint_evaluations[seed]["corpora"]["qwen"][
                                    "aggregate"
                                ]["F_at_physical_rate"],
                                "s_match_worst": checkpoint_evaluations[seed]["matched"][
                                    "s_match_worst"
                                ],
                                "cluster_se": checkpoint_evaluations[seed]["matched"][
                                    "cluster_se"
                                ],
                            }
                            for seed in sw.TRAINING_SEEDS
                        },
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
            save_training_checkpoint(
                output_dir,
                update,
                states,
                optimizers,
                history,
                bindings,
                log_rms_center,
                log_rms_scale,
                cp,
            )
            if update == 512:
                stop_input = {
                    seed: {
                        point: {
                            "s_match_worst": history[seed][point]["matched"][
                                "s_match_worst"
                            ],
                            "cluster_se": history[seed][point]["matched"][
                                "cluster_se"
                            ],
                        }
                        for point in (256, 512)
                    }
                    for seed in sw.TRAINING_SEEDS
                }
                if sw.hard_kill_at_512(stop_input):
                    stopped = True
                    stop_update = 512
                    break

    final_update = 512 if stopped else 1536
    if final_update not in history[sw.TRAINING_SEEDS[0]]:
        raise AssertionError("final checkpoint history is unavailable")
    final_calibration = {
        seed: history[seed][final_update] for seed in sw.TRAINING_SEEDS
    }
    promotes = False
    promotion_evidence: dict[str, Any]
    confirmation_result = None
    confirmation_opened = False
    confirmation_records: list[MatrixRecord] = []
    if stopped:
        promotion_evidence = {
            "checks": {"update_512_hard_kill": True},
            "claim_boundary": "kills only the frozen SILWARP cell",
        }
    else:
        promotes, promotion_evidence = calibration_promotes(final_calibration)
        if promotes:
            # This is the first byte access to the untouched confirmation set.
            confirmation_records = load_records(
                inventory["confirmation"], "confirmation", locked_rows
            )
            confirmation_opened = True
            attach_features(confirmation_records, log_rms_center, log_rms_scale)
            confirmation = combine_records(confirmation_records)
            confirmation_result = {}
            for training_seed in sw.TRAINING_SEEDS:
                evaluation, _ = evaluate_seed(
                    confirmation,
                    states[training_seed],
                    training_seed,
                    "confirmation",
                    log_rms_center,
                    log_rms_scale,
                    cp,
                )
                confirmation_result[training_seed] = evaluation

    # Regenerate exact final FP16 objects from state, including on resume.
    final_blobs = {}
    for training_seed in sw.TRAINING_SEEDS:
        final_blobs[training_seed] = {}
        for corpus in ("qwen", *sw.CONTROL_NAMES):
            _, blob, _ = rounded_model_on_gpu(
                states[training_seed][corpus],
                training_seed,
                log_rms_center,
                log_rms_scale,
                cp,
            )
            final_blobs[training_seed][corpus] = blob
    for training_seed, blobs in final_blobs.items():
        for corpus, blob in blobs.items():
            model_path = output_dir / f"model_seed_{training_seed}_{corpus}.fp16.bin"
            if model_path.exists():
                raise FileExistsError("refusing to overwrite final model artifact")
            model_path.write_bytes(blob)

    if stopped:
        decision = "HARD_KILL_FROZEN_SILWARP_CELL_AT_UPDATE_512"
    elif not promotes:
        decision = "HARD_KILL_FROZEN_SILWARP_CELL_AFTER_FULL_CALIBRATION"
    else:
        # A confirmation pass is described but cannot authorize pinned access.
        primary_confirmation = confirmation_result[sw.TRAINING_SEEDS[0]]
        confirmation_f = sw.require_finite_scalar(
            "confirmation F",
            primary_confirmation["corpora"]["qwen"]["aggregate"][
                "F_at_physical_rate"
            ],
        )
        confirmation_lcb = sw.require_finite_scalar(
            "confirmation matched lower bound",
            primary_confirmation["matched"]["s_match_worst"]
            - 2.0 * primary_confirmation["matched"]["cluster_se"],
        )
        confirm_pass = (
            confirmation_f
            <= 0.8
            and confirmation_lcb
            >= sw.production_ledger(128)["required_absolute_s"]
        )
        decision = (
            "AUXILIARY_IDEAL_CHANNEL_SURVIVOR_REQUIRES_FINITE_CODE"
            if confirm_pass
            else "HARD_KILL_FROZEN_SILWARP_CELL_ON_UNTOUCHED_CONFIRMATION"
        )

    result = {
        "schema": "silwarp_auxiliary_result_v2",
        "decision": decision,
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": sw.protocol_sha256(),
        "source_lock_sha256": sw.source_lock_sha256(),
        "runner_sha256": sw.sha256_file(Path(__file__).resolve()),
        "common_sha256": sw.sha256_file(Path(sw.__file__).resolve()),
        "launch_sentinel": {
            "sha256": sw.sha256_file(launch_sentinel),
            "internal_seal_sha256": sentinel["internal_seal_sha256"],
        },
        "backend": {
            "name": "cupy",
            "version": cp.__version__,
            "device": device_name,
            "runtime_identity": runtime,
        },
        "pinned_panel": {"opened": False, "permitted": False},
        "confirmation_opened": confirmation_opened,
        "stopped_early": stopped,
        "stop_update": stop_update,
        "final_update": final_update,
        "information_channel": protocol["information_channel"],
        "architecture": protocol["architecture"],
        "ledger_128": sw.production_ledger(128),
        "ledger_6": sw.production_ledger(6),
        "log_rms_normalizer": {
            "center_fp16": log_rms_center,
            "scale_fp16": log_rms_scale,
        },
        "source_hashes": {
            "fit": {str(record.key): record.source_sha256 for record in fit_records},
            "calibration": {
                str(record.key): record.source_sha256
                for record in calibration_records
            },
            "confirmation": {
                str(record.key): record.source_sha256
                for record in confirmation_records
            },
        },
        "history": {
            str(seed): {str(update): value for update, value in points.items()}
            for seed, points in history.items()
        },
        "promotion_evidence": promotion_evidence,
        "confirmation": confirmation_result,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sw.sha256_file(path)}
            for path in sorted(output_dir.glob("*.fp16.bin"))
        },
    }
    sw.require_json_finite("final result", result)
    unsigned = dict(result)
    result["canonical_unsigned_sha256"] = sw.sha256_bytes(sw.canonical_json_bytes(unsigned))
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"decision": decision, "result": str(result_path)},
            sort_keys=True,
            allow_nan=False,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--output", type=Path)
    gpu_preflight_parser = subparsers.add_parser("gpu-preflight")
    gpu_preflight_parser.add_argument("--output", type=Path)
    sentinel_parser = subparsers.add_parser("make-sentinel")
    sentinel_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--aux-dir", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--launch-sentinel", type=Path, required=True)
    run_parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "preflight":
        receipt = source_free_preflight()
        text = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.write_text(text, encoding="utf-8")
        print(text, end="")
    elif args.command == "gpu-preflight":
        receipt = gpu_source_free_preflight()
        text = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.write_text(text, encoding="utf-8")
        print(text, end="")
    elif args.command == "make-sentinel":
        if args.output.exists():
            raise FileExistsError(args.output)
        sentinel = sw.build_launch_sentinel(
            Path(__file__).resolve(), Path(sw.__file__).resolve()
        )
        args.output.write_text(
            json.dumps(sentinel, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        sw.validate_launch_sentinel(
            args.output, Path(__file__).resolve(), Path(sw.__file__).resolve()
        )
        print(json.dumps(sentinel, sort_keys=True, allow_nan=False))
    elif args.command == "run":
        run_gate(
            args.aux_dir,
            args.output_dir,
            args.launch_sentinel,
            args.resume,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
