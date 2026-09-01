#!/usr/bin/env python3
"""Authorized tied nonnegative MPS/HMM entropy census.

Only standard-library modules are imported before authorization, output
reservation, independent review and source-closure verification.  The real
entrypoint consumes independently extracted decoder traces, not model weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_COMMON_PATH = Path(__file__).resolve().with_name("mps_common.py")
_COMMON_SPEC = importlib.util.spec_from_file_location("mps_common", _COMMON_PATH)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise RuntimeError("cannot load sealed same-directory mps_common.py")
_COMMON_MODULE = importlib.util.module_from_spec(_COMMON_SPEC)
sys.modules["mps_common"] = _COMMON_MODULE
_COMMON_SPEC.loader.exec_module(_COMMON_MODULE)

from mps_common import (
    AUTHORIZATION,
    CONTROL_SEEDS,
    EM_ITERATIONS,
    EXTRACTION_SCHEMA,
    FIT_SEEDS,
    HIDDEN_DIMENSIONS,
    PERIODS,
    RESET_SYMBOLS,
    RESULT_SCHEMA,
    REVIEW_SCHEMA,
    STANDALONE_REQUIRED_SAVING_BPW,
    STREAM_LOCK_SCHEMA,
    SUFFIX_DEPTHS,
    CompletionLastOutput,
    ContractError,
    HeldFileSet,
    HeldRegularFile,
    arithmetic_encode_binary,
    canonical_json,
    context_count,
    hmm_model_ledger,
    packet_ledger,
    pretty_json,
    prior_bin,
    quantize_probability,
    quantize_simplex,
    require,
    sha256_bytes,
    strict_json_loads,
    suffix_model_ledger,
)


PACKAGE = Path(__file__).resolve().parent
MANIFEST = PACKAGE / "SOURCE_MANIFEST.json"
CONTROL_LOCK_SCHEMA = "tied-mps-gaussian-control-lock-v0"


@dataclass
class Trace:
    ordinal: int
    expert_ordinal: int
    layer_group: str
    expert_group: str
    bits: Any
    base_freq1: Any
    levels: Any
    original_logical_bits: int
    original_payload_bytes: int
    original_payload: bytes


@dataclass
class HMM:
    chi: int
    period: int
    seed: int
    pi: Any
    transition: Any
    emission1: Any
    log2_likelihood: float


def _dynamic_verify_source() -> dict[str, Any]:
    path = PACKAGE / "verify_source.py"
    spec = importlib.util.spec_from_file_location("tied_mps_verify_source_runtime", path)
    require(spec is not None and spec.loader is not None, "source verifier import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_package(PACKAGE)


def _review(held: HeldRegularFile, manifest_sha256: str) -> dict[str, Any]:
    value = strict_json_loads(held.read_all())
    require(isinstance(value, dict), "review object")
    require(value.get("schema") == REVIEW_SCHEMA, "review schema")
    require(value.get("status") == "PASS_INDEPENDENT_SOURCE_REVIEW", "review status")
    require(value.get("source_manifest_sha256") == manifest_sha256, "review source hash")
    require(value.get("authorization") == AUTHORIZATION, "review authorization")
    return value


def _regular_descriptor(row: Any, name: str) -> tuple[Path, int, str]:
    require(isinstance(row, dict), f"{name} descriptor")
    require(set(row) == {"path", "bytes", "sha256"}, f"{name} descriptor fields")
    path = Path(row["path"])
    require(path.is_absolute(), f"{name} absolute path")
    size = int(row["bytes"])
    digest = str(row["sha256"])
    require(size >= 0 and len(digest) == 64, f"{name} descriptor values")
    return path, size, digest


def _validate_lock(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict) and value.get("schema") == STREAM_LOCK_SCHEMA, "stream lock schema")
    require(
        set(value) == {
            "schema", "status", "weights", "experts", "current_object_bytes",
            "current_artifact", "extraction_receipt", "immutable_local_bytes",
            "immutable_global_bytes", "streams",
        },
        "stream lock exact fields",
    )
    require(value.get("status") == "SEALED_INDEPENDENT_DECODER_TRACES", "stream lock status")
    weights = int(value.get("weights", 0))
    experts = int(value.get("experts", 0))
    require(weights > 0 and experts > 0 and weights % experts == 0, "stream lock geometry")
    require(int(value.get("current_object_bytes", -1)) == int(value["current_artifact"]["bytes"]), "current bytes")
    require(abs(8.0 * int(value["current_object_bytes"]) / weights - 2.5) <= 1e-12, "current exact 2.5 bpw")
    _regular_descriptor(value.get("current_artifact"), "current artifact")
    _regular_descriptor(value.get("extraction_receipt"), "extraction receipt")
    local = value.get("immutable_local_bytes")
    require(isinstance(local, list) and len(local) == experts and all(int(item) >= 0 for item in local), "immutable local")
    require(int(value.get("immutable_global_bytes", -1)) >= 0, "immutable global")
    streams = value.get("streams")
    require(isinstance(streams, list) and len(streams) >= experts, "stream rows")
    ordinals = []
    for row in streams:
        require(isinstance(row, dict), "stream row")
        require(
            set(row) == {
                "ordinal", "expert_ordinal", "layer_group", "expert_group",
                "selected_count", "original_logical_bits", "original_payload_bytes",
                "selected_bits", "base_freq1", "polar_levels", "original_payload",
            },
            "stream row exact fields",
        )
        ordinal = int(row.get("ordinal", -1))
        expert = int(row.get("expert_ordinal", -1))
        count = int(row.get("selected_count", 0))
        require(0 <= expert < experts and count > 0, "stream expert/count")
        require(isinstance(row.get("layer_group"), str) and row["layer_group"], "layer fold group")
        require(isinstance(row.get("expert_group"), str) and row["expert_group"], "expert fold group")
        require(int(row.get("original_logical_bits", 0)) > 0, "original logical bits")
        require(int(row.get("original_payload_bytes", -1)) == (int(row["original_logical_bits"]) + 7) // 8, "original payload bytes")
        for field, size in (("selected_bits", count), ("base_freq1", 2 * count), ("polar_levels", count)):
            _, observed, _ = _regular_descriptor(row.get(field), f"stream {ordinal} {field}")
            require(observed == size, f"stream {ordinal} {field} bytes")
        _, observed, _ = _regular_descriptor(row.get("original_payload"), f"stream {ordinal} original payload")
        require(observed == int(row["original_payload_bytes"]), f"stream {ordinal} original payload descriptor bytes")
        ordinals.append(ordinal)
    require(ordinals == list(range(len(streams))), "canonical stream ordinals")
    return value


def _trace_inventory_sha256(lock: dict[str, Any]) -> str:
    """Bind every source descriptor without creating a receipt/lock hash cycle."""
    inventory = {
        "schema": "tied-mps-trace-inventory-v0",
        "weights": lock["weights"],
        "experts": lock["experts"],
        "current_object_bytes": lock["current_object_bytes"],
        "current_artifact": lock["current_artifact"],
        "immutable_global_bytes": lock["immutable_global_bytes"],
        "immutable_local_bytes": lock["immutable_local_bytes"],
        "streams": lock["streams"],
    }
    return sha256_bytes(canonical_json(inventory))


def _open_lock(path: Path, held_files: HeldFileSet) -> tuple[HeldRegularFile, dict[str, Any]]:
    held = held_files.add(HeldRegularFile(path))
    return held, _validate_lock(strict_json_loads(held.read_all()))


def _open_panel(lock: dict[str, Any], held_files: HeldFileSet, cp: Any) -> list[Trace]:
    artifact_path, artifact_bytes, artifact_hash = _regular_descriptor(lock["current_artifact"], "artifact")
    held_files.add(HeldRegularFile(artifact_path, artifact_bytes, artifact_hash))
    receipt_path, receipt_bytes, receipt_hash = _regular_descriptor(lock["extraction_receipt"], "extraction")
    extraction_file = held_files.add(HeldRegularFile(receipt_path, receipt_bytes, receipt_hash))
    extraction = strict_json_loads(extraction_file.read_all())
    require(isinstance(extraction, dict) and extraction.get("schema") == EXTRACTION_SCHEMA, "extraction schema")
    require(extraction.get("status") == "PASS_INDEPENDENT_CANONICAL_EXTRACTION", "extraction status")
    require(extraction.get("current_artifact_sha256") == artifact_hash, "extraction artifact binding")
    require(extraction.get("trace_inventory_sha256") == _trace_inventory_sha256(lock), "extraction inventory binding")
    traces = []
    for row in lock["streams"]:
        buffers = {}
        for field in ("selected_bits", "base_freq1", "polar_levels"):
            path, size, digest = _regular_descriptor(row[field], f"stream {row['ordinal']} {field}")
            buffers[field] = held_files.add(HeldRegularFile(path, size, digest)).read_all()
        payload_path, payload_size, payload_digest = _regular_descriptor(
            row["original_payload"], f"stream {row['ordinal']} original payload"
        )
        original_payload = held_files.add(HeldRegularFile(payload_path, payload_size, payload_digest)).read_all()
        bits = cp.frombuffer(buffers["selected_bits"], dtype=cp.uint8).copy()
        frequencies = cp.frombuffer(buffers["base_freq1"], dtype=cp.dtype("<u2")).copy()
        levels = cp.frombuffer(buffers["polar_levels"], dtype=cp.uint8).copy()
        require(int(bits.size) == int(row["selected_count"]), "bits count")
        require(bool(cp.all((bits == 0) | (bits == 1)).item()), "binary selected bits")
        require(bool(cp.all((frequencies >= 1) & (frequencies <= 65535)).item()), "base frequencies")
        require(bool(cp.all(levels < 6).item()), "polar levels")
        traces.append(
            Trace(
                ordinal=int(row["ordinal"]),
                expert_ordinal=int(row["expert_ordinal"]),
                layer_group=row["layer_group"],
                expert_group=row["expert_group"],
                bits=bits,
                base_freq1=frequencies,
                levels=levels,
                original_logical_bits=int(row["original_logical_bits"]),
                original_payload_bytes=int(row["original_payload_bytes"]),
                original_payload=original_payload,
            )
        )
    return traces


def _contexts(trace: Trace, period: int, cp: Any) -> Any:
    positions = cp.arange(trace.bits.size, dtype=cp.int64) % period
    bins = cp.minimum(15, trace.base_freq1.astype(cp.int64) * 16 // 65536)
    return ((trace.levels.astype(cp.int64) * 16 + bins) * period + positions).astype(cp.int32)


def _nll_bits(bits: Any, freq1: Any, cp: Any) -> float:
    probability = cp.where(bits != 0, freq1.astype(cp.float64) / 65536.0, 1.0 - freq1.astype(cp.float64) / 65536.0)
    return float((-cp.log2(cp.maximum(probability, 2.0**-1074))).sum(dtype=cp.float64).item())


def _replay_original(traces: Sequence[Trace], cp: Any) -> dict[str, Any]:
    rows = []
    for trace in traces:
        bits = cp.asnumpy(trace.bits)
        frequencies = cp.asnumpy(trace.base_freq1)
        payload, logical = arithmetic_encode_binary(bits, frequencies)
        require(logical == trace.original_logical_bits, f"original logical replay {trace.ordinal}")
        require(len(payload) == trace.original_payload_bytes, f"original byte replay {trace.ordinal}")
        require(payload == trace.original_payload, f"original payload replay {trace.ordinal}")
        rows.append(
            {
                "ordinal": trace.ordinal,
                "selected": int(trace.bits.size),
                "logical_bits": logical,
                "payload_bytes": len(payload),
                "payload_sha256": sha256_bytes(payload),
            }
        )
    return {
        "status": "PASS_EXACT_CURRENT_ARITHMETIC_REPLAY",
        "streams": rows,
        "logical_bits": sum(row["logical_bits"] for row in rows),
        "payload_bytes": sum(row["payload_bytes"] for row in rows),
    }


def _suffix_ids(trace: Trace, depth: int, period: int, cp: Any) -> tuple[Any, Any]:
    contexts = _contexts(trace, period, cp)
    states = cp.zeros(trace.bits.size, dtype=cp.int32)
    for lag in range(1, depth + 1):
        states[lag:] |= trace.bits[:-lag].astype(cp.int32) << (lag - 1)
    states[::RESET_SYMBOLS] = 0
    return contexts, contexts * (1 << depth) + states


def _fit_suffix(traces: Sequence[Trace], depth: int, period: int, cp: Any) -> tuple[Any, list[float]]:
    cells = context_count(period) * (1 << depth)
    total = cp.zeros(cells, dtype=cp.int64)
    ones = cp.zeros(cells, dtype=cp.int64)
    identifiers = []
    for trace in traces:
        _, ids = _suffix_ids(trace, depth, period, cp)
        identifiers.append(ids)
        total += cp.bincount(ids, minlength=cells)
        ones += cp.bincount(ids, weights=trace.bits, minlength=cells).astype(cp.int64)
    probabilities = (ones.astype(cp.float64) + 0.5) / (total.astype(cp.float64) + 1.0)
    quantized = cp.clip(cp.floor(probabilities * 65536.0 + 0.5), 1, 65534).astype(cp.uint16)
    per_stream = [_nll_bits(trace.bits, quantized[ids], cp) for trace, ids in zip(traces, identifiers, strict=True)]
    return quantized, per_stream


def _favorable_payloads(nll_rows: Sequence[float]) -> list[int]:
    # A 32-bit arithmetic interval code is never credited with more than a
    # two-bit improvement over its ideal product codelength.  This is a lower
    # payload bound and therefore favorable for an absolute early kill.
    return [max(1, math.ceil(max(0.0, float(value) - 2.0) / 8.0)) for value in nll_rows]


def _payloads_by_expert(traces: Sequence[Trace], payloads: Sequence[int], experts: int) -> list[list[int]]:
    rows = [[] for _ in range(experts)]
    for trace, size in zip(traces, payloads, strict=True):
        rows[trace.expert_ordinal].append(int(size))
    require(all(rows), "every expert owns at least one stream")
    return rows


def _opportunity_ledger(lock: dict[str, Any], traces: Sequence[Trace], payloads: Sequence[int], model_bytes: int) -> dict[str, Any] | None:
    try:
        return packet_ledger(
            weights=int(lock["weights"]),
            current_object_bytes=int(lock["current_object_bytes"]),
            immutable_global_bytes=int(lock["immutable_global_bytes"]),
            immutable_local_bytes=[int(value) for value in lock["immutable_local_bytes"]],
            model_bytes=int(model_bytes),
            stream_payload_bytes=_payloads_by_expert(traces, payloads, int(lock["experts"])),
        )
    except ContractError as exc:
        return {
            "invalid": str(exc),
            "net_physical_saving_bpw": -1_000_000_000.0,
            "passes_F_le_0p8": False,
        }


def run_a0(lock: dict[str, Any], traces: Sequence[Trace], cp: Any) -> dict[str, Any]:
    candidates = []
    for period in PERIODS:
        for depth in SUFFIX_DEPTHS:
            _, nll = _fit_suffix(traces, depth, period, cp)
            model = suffix_model_ledger(depth, period)
            ledger = _opportunity_ledger(lock, traces, _favorable_payloads(nll), model["physical_model_bytes"])
            candidates.append(
                {
                    "depth": depth,
                    "period": period,
                    "ideal_quantized_nll_bits": math.fsum(nll),
                    "model": model,
                    "favorable_packet": ledger,
                }
            )
    best = max(candidates, key=lambda row: (row["favorable_packet"]["net_physical_saving_bpw"], -row["depth"], -row["period"]))
    return {
        "schema": "tied-mps-a0-suffix-screen-v0",
        "status": "LOCAL_SUBCLASS_ONLY",
        "candidates": candidates,
        "best": best,
        "claim_boundary": "A miss closes only deterministic suffix depth<=8. It cannot close a latent-state HMM or edge-emitting WFA.",
    }


def _initial_hmm(chi: int, period: int, seed: int, cp: Any) -> HMM:
    rng = cp.random.RandomState(seed)
    pi = cp.full(chi, 1.0 / chi, dtype=cp.float64)
    diagonal = cp.eye(chi, dtype=cp.float64) * 8.0
    transition = diagonal + 1.0 + 0.05 * rng.random_sample((chi, chi))
    transition /= transition.sum(axis=1, keepdims=True)
    contexts = context_count(period)
    bins = (cp.arange(contexts, dtype=cp.int64) // period) % 16
    base = (bins.astype(cp.float64) + 0.5) / 16.0
    offsets = cp.linspace(-0.08, 0.08, chi, dtype=cp.float64)
    emission = cp.clip(base[:, None] + offsets[None, :] + 0.005 * (rng.random_sample((contexts, chi)) - 0.5), 1e-5, 1.0 - 1e-5)
    return HMM(chi, period, seed, pi, transition, emission, -math.inf)


def _chunks(traces: Sequence[Trace], period: int, cp: Any) -> list[tuple[Any, Any]]:
    rows = []
    for trace in traces:
        contexts = _contexts(trace, period, cp)
        for start in range(0, int(trace.bits.size), RESET_SYMBOLS):
            end = min(int(trace.bits.size), start + RESET_SYMBOLS)
            rows.append((trace.bits[start:end], contexts[start:end]))
    return rows


def _chunk_batches(chunks: Sequence[tuple[Any, Any]], cp: Any, batch: int = 48) -> Iterable[tuple[Any, Any]]:
    by_length: dict[int, list[tuple[Any, Any]]] = {}
    for bits, contexts in chunks:
        by_length.setdefault(int(bits.size), []).append((bits, contexts))
    for length in sorted(by_length, reverse=True):
        rows = by_length[length]
        for start in range(0, len(rows), batch):
            subset = rows[start : start + batch]
            yield cp.stack([row[0] for row in subset]), cp.stack([row[1] for row in subset])


def _expectation_batch(bits: Any, contexts: Any, model: HMM, cp: Any) -> tuple[Any, Any, Any, float]:
    batch, length = map(int, bits.shape)
    chi = model.chi
    alpha = cp.empty((batch, length, chi), dtype=cp.float64)
    scales = cp.empty((batch, length), dtype=cp.float64)
    state = cp.broadcast_to(model.pi[None, :], (batch, chi)).copy()
    for time in range(length):
        if time:
            state = state @ model.transition
        p1 = model.emission1[contexts[:, time]]
        emission = cp.where(bits[:, time, None] != 0, p1, 1.0 - p1)
        state *= emission
        scale = cp.maximum(state.sum(axis=1), 1e-300)
        state /= scale[:, None]
        alpha[:, time] = state
        scales[:, time] = scale
    initial_counts = cp.zeros(chi, dtype=cp.float64)
    transition_counts = cp.zeros((chi, chi), dtype=cp.float64)
    emission_total = cp.zeros_like(model.emission1)
    emission_ones = cp.zeros_like(model.emission1)
    beta = cp.ones((batch, chi), dtype=cp.float64)
    states = cp.arange(chi, dtype=cp.int64)[None, :]
    for time in range(length - 1, -1, -1):
        gamma = alpha[:, time] * beta
        gamma /= cp.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        flat = contexts[:, time, None].astype(cp.int64) * chi + states
        cp.add.at(emission_total.ravel(), flat.ravel(), gamma.ravel())
        cp.add.at(emission_ones.ravel(), flat.ravel(), (gamma * bits[:, time, None]).ravel())
        if time == 0:
            initial_counts += gamma.sum(axis=0)
        else:
            p1 = model.emission1[contexts[:, time]]
            emission = cp.where(bits[:, time, None] != 0, p1, 1.0 - p1)
            weighted_next = emission * beta
            xi = (
                alpha[:, time - 1, :, None]
                * model.transition[None, :, :]
                * weighted_next[:, None, :]
            )
            xi /= cp.maximum(xi.sum(axis=(1, 2), keepdims=True), 1e-300)
            transition_counts += xi.sum(axis=0)
            beta = weighted_next @ model.transition.T
            beta /= cp.maximum(beta.sum(axis=1, keepdims=True), 1e-300)
    log2_likelihood = float((cp.log(scales).sum(dtype=cp.float64) / math.log(2.0)).item())
    return initial_counts, transition_counts, (emission_total, emission_ones), log2_likelihood


def fit_hmm(traces: Sequence[Trace], chi: int, period: int, seed: int, cp: Any) -> HMM:
    model = _initial_hmm(chi, period, seed, cp)
    chunks = _chunks(traces, period, cp)
    for _ in range(EM_ITERATIONS):
        initial = cp.zeros(chi, dtype=cp.float64)
        transitions = cp.zeros((chi, chi), dtype=cp.float64)
        total = cp.zeros_like(model.emission1)
        ones = cp.zeros_like(model.emission1)
        likelihood = 0.0
        for bits, contexts in _chunk_batches(chunks, cp):
            pi_count, transition_count, emissions, score = _expectation_batch(bits, contexts, model, cp)
            initial += pi_count
            transitions += transition_count
            total += emissions[0]
            ones += emissions[1]
            likelihood += score
        initial += 0.5
        transitions += 0.5
        model.pi = initial / initial.sum()
        model.transition = transitions / transitions.sum(axis=1, keepdims=True)
        model.emission1 = cp.clip((ones + 0.5) / (total + 1.0), 1e-8, 1.0 - 1e-8)
        model.log2_likelihood = likelihood
    return model


def score_hmm(traces: Sequence[Trace], model: HMM, cp: Any) -> list[float]:
    """Causal likelihood, batched across reset chunks to avoid scalar kernels."""
    scores = []
    for trace in traces:
        score = 0.0
        for bits, contexts in _chunk_batches(_chunks([trace], model.period, cp), cp):
            batch, length = map(int, bits.shape)
            state = cp.broadcast_to(model.pi[None, :], (batch, model.chi)).copy()
            row_scores = cp.zeros(batch, dtype=cp.float64)
            for time in range(length):
                if time:
                    state = state @ model.transition
                p1_state = model.emission1[contexts[:, time]]
                p1 = cp.sum(state * p1_state, axis=1) / cp.maximum(cp.sum(state, axis=1), 1e-300)
                bit = bits[:, time]
                probability = cp.where(bit != 0, p1, 1.0 - p1)
                row_scores -= cp.log2(cp.maximum(probability, 2.0**-1074))
                state *= cp.where(bit[:, None] != 0, p1_state, 1.0 - p1_state)
                state /= cp.maximum(state.sum(axis=1, keepdims=True), 1e-300)
            score += float(row_scores.sum(dtype=cp.float64).item())
        scores.append(score)
    return scores


def quantize_hmm(model: HMM, cp: Any) -> tuple[HMM, bytes]:
    pi_rows = quantize_simplex(cp.asnumpy(model.pi).tolist())
    transition_rows = [quantize_simplex(row) for row in cp.asnumpy(model.transition).tolist()]
    emission_rows = [quantize_probability(value) for value in cp.asnumpy(model.emission1).ravel().tolist()]
    pi = cp.asarray(pi_rows, dtype=cp.float64) / 65535.0
    transition = cp.asarray(transition_rows, dtype=cp.float64) / 65535.0
    emission = cp.asarray(emission_rows, dtype=cp.float64).reshape(model.emission1.shape) / 65536.0
    quantized = HMM(model.chi, model.period, model.seed, pi, transition, emission, model.log2_likelihood)
    ledger = hmm_model_ledger(model.chi, model.period)
    header = bytearray(256)
    struct.pack_into("<8sHHHHIII", header, 0, b"TMPSHMM\0", 1, model.chi, model.period, 6, RESET_SYMBOLS, model.seed, ledger["tensor_u16_values"])
    tensor = bytearray()
    for value in pi_rows:
        tensor += struct.pack("<H", value)
    for row in transition_rows:
        for value in row:
            tensor += struct.pack("<H", value)
    for value in emission_rows:
        tensor += struct.pack("<H", value)
    packet = bytes(header) + bytes(tensor)
    require(len(packet) == ledger["physical_model_bytes"], "quantized model packet bytes")
    return quantized, packet


def exact_hmm_payloads(traces: Sequence[Trace], model: HMM, cp: Any) -> tuple[list[bytes], list[int]]:
    """Generate frozen Q0.16 causal probabilities in reset-chunk batches."""
    payloads = []
    logical_rows = []
    for trace in traces:
        frequency_chunks = []
        for bits, contexts in _chunk_batches(_chunks([trace], model.period, cp), cp):
            batch, length = map(int, bits.shape)
            state = cp.broadcast_to(model.pi[None, :], (batch, model.chi)).copy()
            frequencies = cp.empty((batch, length), dtype=cp.uint16)
            for time in range(length):
                if time:
                    state = state @ model.transition
                emission = model.emission1[contexts[:, time]]
                p1 = cp.sum(state * emission, axis=1) / cp.maximum(cp.sum(state, axis=1), 1e-300)
                frequencies[:, time] = cp.clip(cp.floor(p1 * 65536.0 + 0.5), 1, 65534).astype(cp.uint16)
                bit = bits[:, time]
                state *= cp.where(bit[:, None] != 0, emission, 1.0 - emission)
                state /= cp.maximum(state.sum(axis=1, keepdims=True), 1e-300)
            frequency_chunks.extend(cp.asnumpy(frequencies))
        flat_frequencies = [int(value) for chunk in frequency_chunks for value in chunk]
        require(len(flat_frequencies) == int(trace.bits.size), f"frequency geometry {trace.ordinal}")
        payload, logical = arithmetic_encode_binary(cp.asnumpy(trace.bits), flat_frequencies)
        payloads.append(payload)
        logical_rows.append(logical)
    return payloads, logical_rows


def run_a1(lock: dict[str, Any], traces: Sequence[Trace], cp: Any) -> dict[str, Any]:
    candidates = []
    models: list[HMM] = []
    model_packets: list[bytes] = []
    for period in PERIODS:
        for chi in HIDDEN_DIMENSIONS:
            for seed in FIT_SEEDS:
                fitted = fit_hmm(traces, chi, period, seed, cp)
                model, model_packet = quantize_hmm(fitted, cp)
                nll = score_hmm(traces, model, cp)
                ledger = hmm_model_ledger(chi, period)
                require(len(model_packet) == ledger["physical_model_bytes"], "candidate model packet ledger")
                opportunity = _opportunity_ledger(lock, traces, _favorable_payloads(nll), len(model_packet))
                candidates.append(
                    {
                        "chi": chi,
                        "period": period,
                        "seed": seed,
                        "quantized_model_nll_bits": math.fsum(nll),
                        "model_packet_sha256": sha256_bytes(model_packet),
                        "model": ledger,
                        "favorable_packet": opportunity,
                    }
                )
                models.append(model)
                model_packets.append(model_packet)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -candidates[index]["favorable_packet"]["net_physical_saving_bpw"],
            candidates[index]["chi"],
            candidates[index]["period"],
            candidates[index]["seed"],
        ),
    )
    best_oracle = candidates[order[0]]
    if best_oracle["favorable_packet"]["net_physical_saving_bpw"] < STANDALONE_REQUIRED_SAVING_BPW:
        return {
            "schema": "tied-mps-a1-hidden-hmm-screen-v0",
            "status": "KILL_BOUNDED_HMM_SEARCH_CONTROLS_UNOPENED",
            "candidates": candidates,
            "best_favorable": best_oracle,
            "complete_family_grid": True,
            "global_MLE_proven": False,
            "claim_boundary": "Kills the frozen 12-iteration, three-seed chi<=64/P<=4/reset-4096 persistent-regime HMM search cell, not edge-emitting WFA, terminal-parity, every HMM, or tensor networks.",
        }
    exact_rows = []
    for index in order:
        if candidates[index]["favorable_packet"]["net_physical_saving_bpw"] + 0.01 < STANDALONE_REQUIRED_SAVING_BPW:
            continue
        quantized, packet = models[index], model_packets[index]
        payloads, logical = exact_hmm_payloads(traces, quantized, cp)
        ledger = _opportunity_ledger(lock, traces, [len(payload) for payload in payloads], len(packet))
        exact_rows.append(
            {
                "chi": quantized.chi,
                "period": quantized.period,
                "seed": quantized.seed,
                "model_packet_sha256": sha256_bytes(packet),
                "model_packet_bytes": len(packet),
                "arithmetic_logical_bits": logical,
                "arithmetic_payload_bytes": [len(payload) for payload in payloads],
                "packet": ledger,
                "_model": quantized,
                "_model_packet": packet,
            }
        )
    require(exact_rows, "favorable survivor must receive exact finite evaluation")
    best_exact = max(exact_rows, key=lambda row: row["packet"]["net_physical_saving_bpw"])
    serializable = []
    for row in exact_rows:
        serializable.append({key: value for key, value in row.items() if not key.startswith("_")})
    return {
        "schema": "tied-mps-a1-hidden-hmm-screen-v0",
        "status": "SURVIVE_EXACT_SOURCE_REQUIRES_HOLDOUT_AND_CONTROLS" if best_exact["packet"]["passes_F_le_0p8"] else "KILL_EXACT_QUANTIZED_HMM_CONTROLS_UNOPENED",
        "candidates": candidates,
        "best_favorable": best_oracle,
        "exact_rows": serializable,
        "best_exact_index": exact_rows.index(best_exact),
        "_best_model": best_exact["_model"],
        "_best_model_packet": best_exact["_model_packet"],
    }


def _crossfit_selected(lock: dict[str, Any], traces: Sequence[Trace], selected: HMM, cp: Any) -> dict[str, Any]:
    """Leave the union of each layer/expert group out; charge one model per fold.

    Streams with the same (layer_group, expert_group) pair share a fold, so its
    model is charged once rather than pessimistically once per stream.  Seed
    selection uses training likelihood only, never held-out payload length.
    """
    groups: dict[tuple[str, str], list[Trace]] = {}
    for trace in traces:
        groups.setdefault((trace.layer_group, trace.expert_group), []).append(trace)
    folds = []
    payload_bytes_by_ordinal: dict[int, int] = {}
    total_model_bytes = 0
    for (layer_group, expert_group), heldout_rows in sorted(groups.items()):
        training = [
            row
            for row in traces
            if row.layer_group != layer_group and row.expert_group != expert_group
        ]
        require(training, f"nonempty disjoint training fold {layer_group}/{expert_group}")
        fitted_rows = []
        for seed in FIT_SEEDS:
            fitted = fit_hmm(training, selected.chi, selected.period, seed, cp)
            quantized, packet = quantize_hmm(fitted, cp)
            fitted_rows.append((math.fsum(score_hmm(training, quantized, cp)), seed, quantized, packet))
        training_nll, seed, quantized, packet = min(fitted_rows, key=lambda row: (row[0], row[1]))
        payloads, logical = exact_hmm_payloads(heldout_rows, quantized, cp)
        for trace, payload in zip(heldout_rows, payloads, strict=True):
            require(trace.ordinal not in payload_bytes_by_ordinal, "holdout stream covered once")
            payload_bytes_by_ordinal[trace.ordinal] = len(payload)
        original_bytes = sum(row.original_payload_bytes for row in heldout_rows)
        candidate_bytes = sum(len(payload) for payload in payloads)
        total_model_bytes += len(packet)
        folds.append(
            {
                "heldout_ordinals": [row.ordinal for row in heldout_rows],
                "excluded_layer_group": layer_group,
                "excluded_expert_group": expert_group,
                "seed_selected_on_training_only": seed,
                "training_quantized_nll_bits": training_nll,
                "model_bytes_fully_charged_once_for_fold": len(packet),
                "heldout_original_payload_bytes": original_bytes,
                "heldout_candidate_payload_bytes": candidate_bytes,
                "heldout_candidate_logical_bits": logical,
                "fold_payload_saving_before_model_bytes": original_bytes - candidate_bytes,
                "fold_net_bytes_after_full_model_charge": original_bytes - candidate_bytes - len(packet),
            }
        )
    require(set(payload_bytes_by_ordinal) == {trace.ordinal for trace in traces}, "holdout partition closure")
    pooled = _opportunity_ledger(
        lock,
        traces,
        [payload_bytes_by_ordinal[trace.ordinal] for trace in traces],
        total_model_bytes,
    )
    return {
        "status": "COMPLETE_DISJOINT_LAYER_OR_EXPERT_HOLDOUT",
        "folds": folds,
        "fold_model_bytes_fully_charged": total_model_bytes,
        "pooled_physical_packet": pooled,
        "minimum_fold_net_bytes": min(row["fold_net_bytes_after_full_model_charge"] for row in folds),
        "identity_not_used_as_probability_context": True,
        "seed_selection_used_heldout_bits": False,
    }


def _controls_after_survival(control_lock_path: Path, selected: HMM, cp: Any) -> dict[str, Any]:
    # This function is deliberately first called only after the exact absolute
    # source pass.  Each row is a separately sealed canonical stream lock made
    # by independently encoding a Gaussian tensor control.
    with HeldFileSet() as control_files:
        lock_file = control_files.add(HeldRegularFile(control_lock_path))
        wrapper = strict_json_loads(lock_file.read_all())
        require(isinstance(wrapper, dict) and wrapper.get("schema") == CONTROL_LOCK_SCHEMA, "control lock schema")
        require(wrapper.get("status") == "SEALED_EIGHT_INDEPENDENT_GAUSSIAN_ENCODINGS", "control lock status")
        rows = wrapper.get("rows")
        require(isinstance(rows, list) and len(rows) == len(CONTROL_SEEDS), "control rows")
        results = []
        for ordinal, (row, expected_seed) in enumerate(zip(rows, CONTROL_SEEDS, strict=True)):
            require(int(row.get("seed", -1)) == expected_seed, "control seed/order")
            path, size, digest = _regular_descriptor(row.get("stream_lock"), f"control {ordinal} stream lock")
            stream_file = control_files.add(HeldRegularFile(path, size, digest))
            lock = _validate_lock(strict_json_loads(stream_file.read_all()))
            traces = _open_panel(lock, control_files, cp)
            best = None
            for fit_seed in FIT_SEEDS:
                fitted = fit_hmm(traces, selected.chi, selected.period, fit_seed, cp)
                quantized, packet = quantize_hmm(fitted, cp)
                payloads, _ = exact_hmm_payloads(traces, quantized, cp)
                ledger = _opportunity_ledger(lock, traces, [len(payload) for payload in payloads], len(packet))
                candidate = {"fit_seed": fit_seed, "packet": ledger, "model_bytes": len(packet)}
                if best is None or ledger["net_physical_saving_bpw"] > best["packet"]["net_physical_saving_bpw"]:
                    best = candidate
            results.append({"control_ordinal": ordinal, "generator_seed": expected_seed, **best})
        control_files.verify_stable()
    savings = [row["packet"]["net_physical_saving_bpw"] for row in results]
    return {
        "status": "COMPLETE_EIGHT_INDEPENDENTLY_ENCODED_REFIT_CONTROLS",
        "rows": results,
        "mean_control_saving_bpw": math.fsum(savings) / len(savings),
        "minimum_control_saving_bpw": min(savings),
        "maximum_control_saving_bpw": max(savings),
        "control_subtraction_creates_pass": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--stream-lock", type=Path, required=True)
    parser.add_argument("--gaussian-control-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.authorization != AUTHORIZATION:
        raise ContractError("authorization mismatch; output absent, review/stream/control unopened, CuPy not imported")
    output_path = args.output.resolve()
    with CompletionLastOutput(output_path) as output:
        with HeldFileSet() as held_files:
            review_path = args.review_receipt.resolve(strict=True)
            review_file = held_files.add(HeldRegularFile(review_path))
            source = _dynamic_verify_source()
            _review(review_file, source["manifest_sha256"])
            stream_lock_path = args.stream_lock.resolve(strict=True)
            stream_lock_file, stream_lock = _open_lock(stream_lock_path, held_files)

            import cupy as cp

            traces = _open_panel(stream_lock, held_files, cp)
            replay = _replay_original(traces, cp)
            a0 = run_a0(stream_lock, traces, cp)
            a1 = run_a1(stream_lock, traces, cp)
            controls = {
                "status": "UNOPENED_ABSOLUTE_SOURCE_DID_NOT_SURVIVE",
                "control_path_resolved_or_statted": False,
            }
            holdout = {"status": "NOT_RUN_ABSOLUTE_SOURCE_DID_NOT_SURVIVE"}
            model_member = None
            if a1["status"] == "SURVIVE_EXACT_SOURCE_REQUIRES_HOLDOUT_AND_CONTROLS":
                selected = a1.pop("_best_model")
                model_packet = a1.pop("_best_model_packet")
                holdout = _crossfit_selected(stream_lock, traces, selected, cp)
                controls = _controls_after_survival(args.gaussian_control_lock.resolve(strict=True), selected, cp)
                model_member = output.write_new("winner_model.bin", model_packet)
            else:
                a1.pop("_best_model", None)
                a1.pop("_best_model_packet", None)
            held_files.verify_stable()
            best_source_saving = (
                max((row["packet"]["net_physical_saving_bpw"] for row in a1.get("exact_rows", [])), default=a1["best_favorable"]["favorable_packet"]["net_physical_saving_bpw"])
            )
            control_mean = controls.get("mean_control_saving_bpw")
            result = {
                "schema": RESULT_SCHEMA,
                "status": a1["status"],
                "source_manifest_sha256": source["manifest_sha256"],
                "stream_lock_sha256": sha256_bytes(stream_lock_file.read_all()),
                "current_replay": replay,
                "A0": a0,
                "A1": a1,
                "whole_layer_expert_holdout": holdout,
                "gaussian_controls": controls,
                "best_source_direct_saving_bpw": best_source_saving,
                "qwen_specific_excess_over_independently_refit_gaussian_mean_bpw": None if control_mean is None else best_source_saving - control_mean,
                "standalone_required_saving_bpw": STANDALONE_REQUIRED_SAVING_BPW,
                "speculative_composite_gap_used_as_threshold": False,
                "model_identity_layer_identity_expert_identity_used_as_probability_key": False,
                "claim_boundary": "Scoped tied chi<=64/P<=4/reset-4096 persistent-regime HMM census. A0 is local-only; bounded EM is not a global HMM-MLE proof; edge-emitting/terminal-parity WFA remains open; no result is additive with another oracle.",
            }
            result_bytes = pretty_json(result)
            result_member = output.write_new("result.json", result_bytes)
            members = [result_member] + ([] if model_member is None else [model_member])
            output.complete(members, source["manifest_sha256"])
    print(json.dumps({"status": result["status"], "output": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"BLOCK: {exc}", file=sys.stderr)
        raise SystemExit(2)
