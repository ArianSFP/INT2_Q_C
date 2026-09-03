"""Authenticated full-panel CuPy worker for the sealed CBIB-1 aperture.

CuPy performs label storage, pairwise sufficient-statistic extraction, hard-EM
assignments, and control permutations.  The frozen NumPy core remains the
authority for combinatorial charges, KT log probabilities, packet capacity,
read denominators, and final decision predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import clustered_ib_core as core


BLOCK_VALUES = 2048
THRESHOLD_RMS = 0.981598821873


def _require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def _cupy():
    import cupy as cp
    return cp


def _safe_explicit_path(payload_root: Path, relative_path: str) -> Path:
    root = Path(payload_root)
    _require(root.is_dir() and not root.is_symlink(), "payload root must be real")
    rel = Path(relative_path)
    _require(not rel.is_absolute() and ".." not in rel.parts, "unsafe relative path")
    root_resolved = root.resolve(strict=True)
    cursor = root
    for component in rel.parts:
        cursor = cursor / component
        _require(not cursor.is_symlink(), "symlink component in panel path")
    _require(cursor.is_file() and not cursor.is_symlink(), "missing regular panel file")
    resolved = cursor.resolve(strict=True)
    _require(resolved == root_resolved or root_resolved in resolved.parents,
             "panel path escapes root")
    return cursor


def _authenticated_bytes(payload_root: Path, item: Mapping) -> bytes:
    data = _safe_explicit_path(payload_root, str(item["relative_path"])).read_bytes()
    _require(len(data) == int(item["bytes"]), "panel byte-count mismatch")
    _require(hashlib.sha256(data).hexdigest() == str(item["sha256"]),
             "panel SHA-256 mismatch")
    return data


def _bf16_bytes_to_numpy(data: bytes, raw_shape: Sequence[int]) -> np.ndarray:
    raw = np.frombuffer(data, dtype="<u2")
    expected = math.prod(int(x) for x in raw_shape)
    _require(raw.size == expected, "BF16 element-count mismatch")
    bits = raw.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32).reshape(tuple(int(x) for x in raw_shape))


def _scale_u16_cpu(values: np.ndarray, block_values: int = BLOCK_VALUES) -> np.ndarray:
    """Canonical FP64 dot-order RMS followed by decoder-visible binary16."""
    flat = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    _require(flat.size and flat.size % block_values == 0, "complete quantizer blocks")
    scales = np.empty(flat.size // block_values, dtype=np.uint16)
    for block in range(scales.size):
        segment = flat[block * block_values:(block + 1) * block_values]
        rms = math.sqrt(float(np.dot(segment, segment)) / block_values)
        decoded = np.float16(rms)
        _require(np.isfinite(decoded) and float(decoded) > 0.0, "invalid binary16 scale")
        scales[block] = decoded.view(np.uint16)
    return scales


def quantize_canonical_gpu(canonical: np.ndarray):
    cp = _cupy()
    host = np.ascontiguousarray(canonical, dtype=np.float32).reshape(-1)
    scales = _scale_u16_cpu(host)
    decoded = cp.asarray(scales.view(np.float16).astype(np.float64))
    flat = cp.asarray(host, dtype=cp.float64)
    threshold = cp.repeat(decoded, BLOCK_VALUES) * THRESHOLD_RMS
    labels = cp.where(flat < -threshold, 0,
                      cp.where(flat < 0.0, 1, cp.where(flat <= threshold, 2, 3)))
    return labels.astype(cp.uint8), scales


def load_quantized_panel_gpu(panel: Mapping, payload_root: Path):
    cp = _cupy()
    experts = [int(x) for x in panel["experts"]]
    _require(2 <= len(experts) <= 256 and len(set(experts)) == len(experts),
             "expert panel")
    d_ff, d_model = int(panel["d_ff"]), int(panel["d_model"])
    _require(d_ff > 0 and d_model > 0, "panel geometry")
    _require(len(panel["files"]) == 2 * len(experts), "panel file count")
    locked = {(int(row["expert"]), str(row["role"])): row for row in panel["files"]}
    _require(len(locked) == 2 * len(experts), "duplicate panel binding")
    scale_bits = 0
    source_bytes = 0
    authenticated = []
    expert_rows = []
    for expert in experts:
        roles = []
        for role in ("up", "down"):
            _require((expert, role) in locked, "missing expert/role")
            item = locked[(expert, role)]
            data = _authenticated_bytes(payload_root, item)
            source_bytes += len(data)
            raw = _bf16_bytes_to_numpy(data, item["raw_shape"])
            canonical = raw if role == "up" else np.ascontiguousarray(raw.T)
            _require(tuple(map(int, canonical.shape)) == (d_ff, d_model),
                     "canonical matrix shape")
            labels, scales = quantize_canonical_gpu(canonical)
            scale_bits += int(scales.size) * 16
            roles.append(labels)
            authenticated.append({
                "expert": expert, "role": role,
                "relative_path": item["relative_path"], "bytes": len(data),
                "sha256": item["sha256"],
            })
            del raw, canonical, data
        expert_rows.append(cp.stack(roles, axis=0))
    labels = cp.stack(expert_rows, axis=0)
    return labels, {
        "source_files_read_once": len(authenticated),
        "source_bytes_read_once": source_bytes,
        "source_logical_host_scan_amplification": 1.0,
        "source_read_metric_scope": (
            "one application-level authenticated host scan; not filesystem-page or HBM traffic"
        ),
        "scale_bits": scale_bits,
        "scale_bytes_per_expert": scale_bits // 8 // len(experts),
        "authenticated_inputs": authenticated,
    }


def model_counts_gpu(group_labels, assignments):
    cp = _cupy()
    q = cp.asarray(group_labels, dtype=cp.uint8)
    u = cp.asarray(assignments, dtype=cp.uint8)
    _require(q.ndim == 2 and u.shape == (q.shape[1],), "model-count shapes")
    _require(not bool(cp.any(q >= 4).item()) and not bool(cp.any(u >= 2).item()),
             "model-count values")
    latent = cp.bincount(u, minlength=2).astype(cp.int64)
    conditional = cp.empty((q.shape[0], 2, 4), dtype=cp.int64)
    for expert in range(q.shape[0]):
        for state in range(2):
            conditional[expert, state] = cp.bincount(
                q[expert, u == state], minlength=4
            )
    return latent, conditional


def assignment_costs_gpu(group_labels, latent, conditional):
    cp = _cupy()
    q = cp.asarray(group_labels, dtype=cp.uint8)
    latent = cp.asarray(latent, dtype=cp.int64)
    conditional = cp.asarray(conditional, dtype=cp.int64)
    k, n = map(int, q.shape)
    _require(latent.shape == (2,) and conditional.shape == (k, 2, 4),
             "assignment-model shapes")
    latent_logp = cp.log2((latent.astype(cp.float64) + 0.5) /
                          (latent.sum(dtype=cp.int64) + 1.0))
    costs = cp.empty((2, n), dtype=cp.float64)
    for state in range(2):
        cost = cp.full(n, -latent_logp[state], dtype=cp.float64)
        for expert in range(k):
            logp = cp.log2((conditional[expert, state].astype(cp.float64) + 0.5) /
                           (latent[state] + 2.0))
            cost -= logp[q[expert]]
        costs[state] = cost
    return costs


def _initial_assignments_gpu(group_labels) -> Iterable:
    cp = _cupy()
    q = cp.asarray(group_labels, dtype=cp.uint8)
    k, n = map(int, q.shape)
    gray = cp.asarray(core.GRAY_PLANES)

    def adjusted(candidate):
        candidate = candidate.astype(cp.uint8, copy=False)
        if bool(cp.all(candidate == candidate[0]).item()):
            candidate = candidate.copy()
            candidate[::2] ^= cp.uint8(1)
        return candidate

    for expert in range(k):
        for plane in range(2):
            yield adjusted(gray[q[expert], plane])
    for plane in range(2):
        bits = gray[q, plane]
        yield adjusted((cp.sum(bits, axis=0) * 2 >= k).astype(cp.uint8))
        # CuPy 14.2 does not implement bitwise_xor.reduce; the frozen row order
        # makes this explicit loop byte-for-byte deterministic.
        parity = bits[0].copy()
        for expert in range(1, k):
            parity ^= bits[expert]
        yield adjusted(parity.astype(cp.uint8))
    index = cp.arange(n, dtype=cp.uint64)
    hashed = ((index * cp.uint64(0x9E3779B97F4A7C15)) >> cp.uint64(63)).astype(cp.uint8)
    yield adjusted(hashed)


def _nll_from_counts(latent_train: np.ndarray, conditional_train: np.ndarray,
                     latent_test: np.ndarray, conditional_test: np.ndarray):
    latent_logp = core._kt_log_prob(latent_train, int(latent_train.sum()), 2)
    latent_bits = float(-np.dot(latent_test.astype(np.float64), latent_logp))
    private = []
    for expert in range(conditional_train.shape[0]):
        bits = 0.0
        for state in range(2):
            logp = core._kt_log_prob(
                conditional_train[expert, state], int(latent_train[state]), 4
            )
            bits -= float(np.dot(conditional_test[expert, state].astype(np.float64), logp))
        private.append(bits)
    return latent_bits, private


@dataclass(frozen=True)
class GPUModel:
    latent_counts: np.ndarray
    conditional_counts: np.ndarray
    train_nll_bits: float


def evaluate_binary_model_gpu(group_labels, latent: np.ndarray,
                              conditional: np.ndarray) -> dict:
    cp = _cupy()
    q = cp.asarray(group_labels, dtype=cp.uint8)
    costs = assignment_costs_gpu(q, cp.asarray(latent), cp.asarray(conditional))
    assignments = cp.argmin(costs, axis=0).astype(cp.uint8)
    test_latent, test_conditional = model_counts_gpu(q, assignments)
    latent_test = cp.asnumpy(test_latent)
    conditional_test = cp.asnumpy(test_conditional)
    latent_bits, private = _nll_from_counts(
        np.asarray(latent), np.asarray(conditional), latent_test, conditional_test
    )
    return {
        "assignments": assignments,
        "latent_bits": latent_bits,
        "private_bits": private,
        "total_bits": latent_bits + sum(private),
        "test_latent_counts": latent_test,
        "test_conditional_counts": conditional_test,
    }


def fit_binary_product_model_gpu(group_labels,
                                 max_iterations: int = core.MAX_EM_ITERATIONS) -> GPUModel:
    cp = _cupy()
    q = cp.asarray(group_labels, dtype=cp.uint8)
    _require(q.ndim == 2 and 2 <= q.shape[0] <= 16 and q.shape[1] > 0,
             "hard-EM geometry")
    best = None
    # Duplicate starts are harmless and cannot change the deterministic minimum.
    for initial in _initial_assignments_gpu(q):
        u = initial.copy()
        for _ in range(max_iterations):
            latent_gpu, conditional_gpu = model_counts_gpu(q, u)
            proposed = cp.argmin(
                assignment_costs_gpu(q, latent_gpu, conditional_gpu), axis=0
            ).astype(cp.uint8)
            if bool(cp.array_equal(proposed, u).item()):
                break
            u = proposed
        latent_gpu, conditional_gpu = model_counts_gpu(q, u)
        latent = cp.asnumpy(latent_gpu)
        conditional = cp.asnumpy(conditional_gpu)
        signature0 = tuple([int(latent[0])] + conditional[:, 0].reshape(-1).astype(int).tolist())
        signature1 = tuple([int(latent[1])] + conditional[:, 1].reshape(-1).astype(int).tolist())
        if signature1 < signature0:
            latent = latent[::-1].copy()
            conditional = conditional[:, ::-1].copy()
        evaluated = evaluate_binary_model_gpu(q, latent, conditional)
        tie = (
            float(evaluated["total_bits"]),
            core.binary_model_descriptor_bits(latent, int(q.shape[0])),
            tuple(latent.astype(int).tolist()),
            tuple(conditional.reshape(-1).astype(int).tolist()),
        )
        if best is None or tie < best[0]:
            best = (tie, GPUModel(latent, conditional, float(evaluated["total_bits"])))
    _require(best is not None, "no hard-EM initializer")
    return best[1]


def pairwise_scores_by_fold_gpu(labels, folds: np.ndarray,
                                fold_count: int) -> np.ndarray:
    """One GPU pass per expert pair/role; CPU-authoritative entropy arithmetic."""
    cp = _cupy()
    q = cp.asarray(labels, dtype=cp.uint8)
    fold_gpu = cp.asarray(folds, dtype=cp.int64)
    e, roles, n = map(int, q.shape)
    _require(roles == 2 and folds.shape == (n,), "pairwise fold geometry")
    scores = np.zeros((fold_count, e, e), dtype=np.float64)
    for left in range(e):
        for right in range(left + 1, e):
            value = np.zeros(fold_count, dtype=np.float64)
            for role in range(roles):
                code = (q[left, role].astype(cp.int64) * 4 +
                        q[right, role].astype(cp.int64))
                held = cp.bincount(fold_gpu * 16 + code,
                                   minlength=fold_count * 16).reshape(fold_count, 4, 4)
                held_cpu = cp.asnumpy(held)
                total = held_cpu.sum(axis=0)
                for fold in range(fold_count):
                    joint = total - held_cpu[fold]
                    count = int(joint.sum())
                    value[fold] += (
                        core.entropy_bits_from_counts(joint.sum(axis=1))
                        + core.entropy_bits_from_counts(joint.sum(axis=0))
                        - core.entropy_bits_from_counts(joint.reshape(-1))
                    ) / max(count, 1)
            scores[:, left, right] = value
            scores[:, right, left] = value
    return scores


def _marginal_nll_from_gpu(train_labels, test_labels):
    cp = _cupy()
    train_counts = cp.asnumpy(cp.bincount(train_labels, minlength=4)).astype(np.int64)
    test_counts = cp.asnumpy(cp.bincount(test_labels, minlength=4)).astype(np.int64)
    logp = core._kt_log_prob(train_counts, int(train_counts.sum()), 4)
    return float(-np.dot(test_counts.astype(np.float64), logp))


def crossfit_group_size_gpu(labels, group_size: int,
                            fold_count: int = core.FOLD_COUNT,
                            superblock_values: int = core.SUPERBLOCK_VALUES,
                            pairwise_scores: np.ndarray | None = None) -> dict:
    cp = _cupy()
    q = cp.asarray(labels, dtype=cp.uint8)
    _require(q.ndim == 3 and q.shape[1] == 2 and q.dtype == cp.uint8,
             "labels [expert,role,coordinate]")
    e, roles, n = map(int, q.shape)
    _require(group_size in core.compatible_group_sizes(e), "group size")
    folds = core.fold_ids(n, fold_count, superblock_values)
    if pairwise_scores is None:
        pairwise_scores = pairwise_scores_by_fold_gpu(q, folds, fold_count)
    _require(pairwise_scores.shape == (fold_count, e, e), "pairwise score cache")

    baseline_data = 0.0
    latent_data = 0.0
    private_data = np.zeros(e, dtype=np.float64)
    baseline_model_bits = 0
    conditional_model_bits = np.zeros(e, dtype=np.int64)
    latent_model_bits_by_segment = []
    common_data_bits_by_segment = []
    segment_members = []
    fold_evidence = []

    for fold in range(fold_count):
        train_index = cp.asarray(np.flatnonzero(folds != fold), dtype=cp.int64)
        test_index = cp.asarray(np.flatnonzero(folds == fold), dtype=cp.int64)
        train_n, test_n = int(train_index.size), int(test_index.size)
        partition = core.greedy_equal_partition(pairwise_scores[fold], group_size)
        fold_baseline = 0.0
        fold_latent = 0.0
        for expert in range(e):
            for role in range(roles):
                bits = _marginal_nll_from_gpu(
                    q[expert, role, train_index], q[expert, role, test_index]
                )
                baseline_data += bits
                fold_baseline += bits
                baseline_model_bits += core.marginal_model_descriptor_bits(train_n)
        for group in partition:
            members = cp.asarray(group, dtype=cp.int64)
            segment_latent_bits = 0.0
            for role in range(roles):
                train = q[members, role][:, train_index]
                test = q[members, role][:, test_index]
                model = fit_binary_product_model_gpu(train)
                scored = evaluate_binary_model_gpu(
                    test, model.latent_counts, model.conditional_counts
                )
                segment_latent_bits += float(scored["latent_bits"])
                for local, expert in enumerate(group):
                    private_data[expert] += float(scored["private_bits"][local])
                    conditional_model_bits[expert] += sum(
                        3 * core.ceil_log2_states(int(state_n) + 1)
                        for state_n in model.latent_counts
                    )
            latent_data += segment_latent_bits
            fold_latent += segment_latent_bits
            latent_model_bits_by_segment.append(
                sum(core.ceil_log2_states(train_n + 1) for _ in range(roles))
            )
            common_data_bits_by_segment.append(segment_latent_bits)
            segment_members.append(tuple(int(x) for x in group))
        fold_evidence.append({
            "fold": fold, "train_coordinates": train_n,
            "test_coordinates": test_n,
            "partition": [list(group) for group in partition],
            "baseline_data_bits": fold_baseline,
            "latent_data_bits": fold_latent,
        })

    private_total = float(private_data.sum())
    weights = e * roles * n
    partition_bits = fold_count * core.partition_descriptor_bits(e, group_size)
    selector_bits = core.selector_bits_for_group_bank(e)
    common_model_bits = int(sum(latent_model_bits_by_segment))
    structured_model_bits = common_model_bits + int(conditional_model_bits.sum())
    baseline_framing = e * core.PRIVATE_HEADER_BYTES * 8
    structured_framing = (
        core.GLOBAL_HEADER_BYTES * 8
        + len(segment_members) * core.GROUP_HEADER_BYTES * 8
        + e * core.PRIVATE_HEADER_BYTES * 8
    )
    baseline_charged = baseline_data + baseline_model_bits + baseline_framing
    structured_charged = (
        private_total + latent_data + structured_model_bits + partition_bits
        + selector_bits + structured_framing
    )
    return {
        "group_size": group_size, "expert_count": e, "roles": roles,
        "coordinates_per_role": n, "source_weights": weights,
        "fold_count": fold_count, "superblock_values": superblock_values,
        "baseline_data_bits": baseline_data,
        "private_conditional_data_bits": private_total,
        "latent_data_bits": latent_data,
        "favorable_gross_gain_bpw": (baseline_data - private_total) / weights,
        "net_ideal_gain_bpw": (baseline_data - private_total - latent_data) / weights,
        "baseline_model_bits": baseline_model_bits,
        "latent_model_bits": common_model_bits,
        "conditional_model_bits": int(conditional_model_bits.sum()),
        "partition_bits": partition_bits, "selector_bits": selector_bits,
        "baseline_framing_bits": baseline_framing,
        "structured_framing_bits": structured_framing,
        "baseline_charged_bits": baseline_charged,
        "structured_charged_bits": structured_charged,
        "charged_gain_bpw": (baseline_charged - structured_charged) / weights,
        "private_data_bits_by_expert": private_data.tolist(),
        "private_model_bits_by_expert": conditional_model_bits.astype(int).tolist(),
        "common_data_bits_by_segment": common_data_bits_by_segment,
        "common_model_bits_by_segment": latent_model_bits_by_segment,
        "segment_members": [list(group) for group in segment_members],
        "fold_evidence": fold_evidence,
    }


def marginal_preserving_control_gpu(labels, seed: int):
    cp = _cupy()
    q = cp.asarray(labels, dtype=cp.uint8)
    e, roles, n = map(int, q.shape)
    base = cp.arange(n, dtype=cp.int64)
    out = cp.empty_like(q)
    for expert in range(e):
        for role in range(roles):
            a, b = core.affine_permutation_parameters(n, int(seed), expert, role)
            out[expert, role] = q[expert, role, (a * base + b) % n]
    return out


def score_source_gate_gpu(labels, scale_bytes_per_expert: int,
                          fold_count: int = core.FOLD_COUNT,
                          superblock_values: int = core.SUPERBLOCK_VALUES,
                          run_controls: bool = True) -> dict:
    cp = _cupy()
    q = cp.asarray(labels, dtype=cp.uint8)
    e, roles, n = map(int, q.shape)
    _require(roles == 2 and 2 <= e <= 256, "gate geometry")
    sizes = core.compatible_group_sizes(e)
    folds = core.fold_ids(n, fold_count, superblock_values)
    pairwise = pairwise_scores_by_fold_gpu(q, folds, fold_count)
    source = []
    for group_size in sizes:
        score = crossfit_group_size_gpu(
            q, group_size, fold_count, superblock_values, pairwise
        )
        requirements = core.packet_requirements(score, scale_bytes_per_expert)
        envelopes = {
            str(rate): core.physical_read_envelope(
                expert_count=e, weights_per_expert=roles * n,
                requested_rate=rate, **requirements
            ) for rate in core.RATE_ENDPOINTS
        }
        feasible = [
            rate for rate, envelope in envelopes.items()
            if envelope.get("status") == "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC"
            and envelope.get("capacity_ok") is True
            and envelope.get("strictly_below_2x") is True
        ]
        row = dict(score)
        row["read_envelopes"] = envelopes
        row["feasible_rate_endpoints"] = feasible
        source.append(row)

    favorable = [r for r in source if r["favorable_gross_gain_bpw"] >= core.TARGET_GAIN_BPW]
    survivors = [r for r in favorable if r["feasible_rate_endpoints"]]
    result = {
        "schema": "same_layer_clustered_ib_entropy_gate_result_v0",
        "target_gain_bpw_on_up_down": core.TARGET_GAIN_BPW,
        "source_scores": source, "controls_executed": False,
        "controls": [], "eligible_for_finite_codec": False,
    }
    if not favorable:
        result["status"] = "HARD_KILL_FAVORABLE_BELOW_TARGET"
        return result
    if not survivors:
        result["status"] = "HOLD_NO_STRICT_READ_FEASIBLE_RATE"
        return result
    if not run_controls:
        result["status"] = "HOLD_CONTROLS_REQUIRED_AFTER_SOURCE_SURVIVAL"
        return result

    candidate_sizes = {int(row["group_size"]) for row in survivors}
    controls = []
    for seed in core.CONTROL_SEEDS:
        controlled = marginal_preserving_control_gpu(q, seed)
        control_folds = core.fold_ids(n, fold_count, superblock_values)
        control_pairwise = pairwise_scores_by_fold_gpu(controlled, control_folds, fold_count)
        rows = [
            crossfit_group_size_gpu(
                controlled, size, fold_count, superblock_values, control_pairwise
            ) for size in sorted(candidate_sizes)
        ]
        controls.append({"seed": seed, "scores": rows})
        del controlled
    result["controls_executed"] = True
    result["controls"] = controls
    promoted = []
    for row in survivors:
        size = int(row["group_size"])
        maximum_control = max(
            score["charged_gain_bpw"] for control in controls
            for score in control["scores"] if int(score["group_size"]) == size
        )
        corrected = float(row["charged_gain_bpw"]) - max(0.0, float(maximum_control))
        row["maximum_control_charged_gain_bpw"] = maximum_control
        row["control_corrected_charged_gain_bpw"] = corrected
        if corrected >= core.TARGET_GAIN_BPW:
            promoted.append(row)
    if promoted:
        result["status"] = "SURVIVE_SOURCE_ONLY_REQUIRES_FINITE_CODEC"
        result["eligible_for_finite_codec"] = True
        result["promoted_group_sizes"] = [row["group_size"] for row in promoted]
    else:
        result["status"] = "HARD_KILL_CHARGED_OR_CONTROLS_BELOW_TARGET"
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_authorized_panel(panel_path: Path, payload_root: Path) -> dict:
    cp = _cupy()
    panel_bytes = Path(panel_path).read_bytes()
    panel = json.loads(panel_bytes.decode("utf-8"))
    _require(panel.get("schema") == "same_layer_common_latent_panel_lock_v0",
             "panel schema")
    labels, io = load_quantized_panel_gpu(panel, payload_root)
    result = score_source_gate_gpu(
        labels, int(io["scale_bytes_per_expert"]),
        fold_count=core.FOLD_COUNT, superblock_values=core.SUPERBLOCK_VALUES,
        run_controls=True,
    )
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="strict")
    result.update({
        "claim_boundary": "IDEAL_LABEL_ENTROPY_CENSUS_ONLY_NOT_A_FINITE_CODEC_OR_MSE_RESULT",
        "panel_lock_sha256": hashlib.sha256(panel_bytes).hexdigest(),
        "input_read_ledger": io,
        "cuda": {
            "cupy_version": cp.__version__,
            "device_id": int(cp.cuda.runtime.getDevice()),
            "device_name": str(name),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        },
    })
    return _jsonable(result)
