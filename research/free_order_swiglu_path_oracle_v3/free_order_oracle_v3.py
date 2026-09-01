#!/usr/bin/env python3
"""Source-only FOSP-ARX-v3 science core.

This module contains the frozen 3x3 predictor/path oracle but performs no
source discovery, model access, heavy import, GPU call, calibration, or
authorization.  A future execution must enter through ``bootstrap_v3.py``
with independently pinned package, interpreter, and exact runtime closures.
"""

from __future__ import annotations

import math


SCHEMA = "free_order_swiglu_path_auxiliary_result_v3"
ROWS = 768
COLS = 2048
ROLES = 3
WEIGHTS_PER_EXPERT = ROWS * COLS * ROLES
REQUIRED_S = -0.5 * math.log2(0.8)
RATES = (2.15, 2.30, 2.50)
HEADER_BYTES = 64
FACTORADIC_BYTES = 783
FACTORADIC_BITS = FACTORADIC_BYTES * 8
FP16_COEFFICIENTS_PER_EDGE = 9
PATH_EDGES = ROWS - 1
FP16_COEFFICIENT_BITS = PATH_EDGES * FP16_COEFFICIENTS_PER_EDGE * 16
TOTAL_SIDE_BITS = HEADER_BYTES * 8 + FACTORADIC_BITS + FP16_COEFFICIENT_BITS
SIDE_BPW = TOTAL_SIDE_BITS / WEIGHTS_PER_EXPERT
REQUIRED_GROSS_S = REQUIRED_S + SIDE_BPW
CONTROL_SEEDS = (
    26_090_101,
    26_090_119,
    26_090_143,
    26_090_171,
    26_090_207,
    26_090_231,
    26_090_263,
    26_090_299,
)


class ProtocolError(RuntimeError):
    """Fail-closed protocol violation."""


def ceil_log2_factorial(n):
    if n < 0:
        raise ValueError("negative factorial")
    return (math.factorial(n) - 1).bit_length()


def rank_permutation(permutation):
    n = len(permutation)
    if sorted(permutation) != list(range(n)):
        raise ValueError("not a permutation")
    available = list(range(n))
    rank = 0
    for index, value in enumerate(permutation):
        position = available.index(value)
        rank += position * math.factorial(n - index - 1)
        del available[position]
    return rank


def unrank_permutation(n, rank):
    if n < 0 or rank < 0 or rank >= math.factorial(n):
        raise ValueError("factoradic rank outside domain")
    available = list(range(n))
    output = []
    for remaining in range(n, 0, -1):
        factorial = math.factorial(remaining - 1)
        position, rank = divmod(rank, factorial)
        output.append(available.pop(position))
    return tuple(output)


def serialize_permutation(permutation):
    rank = rank_permutation(permutation)
    encoded = rank.to_bytes(FACTORADIC_BYTES, "big", signed=False)
    if len(encoded) != FACTORADIC_BYTES:
        raise AssertionError("factoradic physical width drift")
    if unrank_permutation(len(permutation), int.from_bytes(encoded, "big")) != tuple(permutation):
        raise AssertionError("factoradic roundtrip failed")
    return encoded


def frame_ledger(rate):
    if rate not in RATES:
        raise ValueError("rate is not frozen")
    frame_bytes = math.floor(WEIGHTS_PER_EXPERT * rate / 8.0)
    frame_bits = frame_bytes * 8
    payload_bits = frame_bits - TOTAL_SIDE_BITS
    if payload_bits <= 0:
        raise AssertionError("side ledger exhausts frame")
    cold_page_bytes = (math.ceil(frame_bytes / 4096) + 1) * 4096
    return {
        "requested_rate_bpw": rate,
        "frame_bytes": frame_bytes,
        "actual_rate_bpw": frame_bits / WEIGHTS_PER_EXPERT,
        "header_bits": HEADER_BYTES * 8,
        "factoradic_bits": FACTORADIC_BITS,
        "fp16_coefficient_bits": FP16_COEFFICIENT_BITS,
        "total_side_bits": TOTAL_SIDE_BITS,
        "side_bpw": SIDE_BPW,
        "residual_payload_bits": payload_bits,
        "residual_payload_bpw": payload_bits / WEIGHTS_PER_EXPERT,
        "required_gross_s_bpw": REQUIRED_GROSS_S,
        "logical_byte_read_amplification": 1.0,
        "cold_page_bytes_including_one_shared_page": cold_page_bytes,
        "cold_page_amplification": cold_page_bytes / frame_bytes,
        "strictly_below_2x": cold_page_bytes / frame_bytes < 2.0,
    }


def _matched_control(source, seed, cp):
    """Preserve every neuron role mean and centered 3x3 role Gram exactly."""
    random = cp.random.RandomState(seed)
    raw = random.standard_normal(source.shape, dtype=cp.float64)
    source_mean = cp.mean(source, axis=2)
    source_centered = source - source_mean[:, :, None]
    source_gram = cp.einsum("nrd,nsd->nrs", source_centered, source_centered)

    raw -= cp.mean(raw, axis=2)[:, :, None]
    raw_gram = cp.einsum("nrd,nsd->nrs", raw, raw)
    source_eval, source_evec = cp.linalg.eigh(source_gram)
    raw_eval, raw_evec = cp.linalg.eigh(raw_gram)
    if bool(cp.any(source_eval <= 0.0)) or bool(cp.any(raw_eval <= 0.0)):
        raise ProtocolError("degenerate matched-control Gram")
    source_root = cp.einsum("nri,ni,nsi->nrs", source_evec, cp.sqrt(source_eval), source_evec)
    raw_invroot = cp.einsum("nri,ni,nsi->nrs", raw_evec, 1.0 / cp.sqrt(raw_eval), raw_evec)
    transform = cp.einsum("nri,nis->nrs", source_root, raw_invroot)
    control = cp.einsum("nrs,nsd->nrd", transform, raw) + source_mean[:, :, None]

    control_mean = cp.mean(control, axis=2)
    control_centered = control - control_mean[:, :, None]
    control_gram = cp.einsum("nrd,nsd->nrs", control_centered, control_centered)
    mean_error = float(cp.max(cp.abs(control_mean - source_mean)).item())
    gram_relative = float(
        (cp.max(cp.abs(control_gram - source_gram)) / cp.max(cp.abs(source_gram))).item()
    )
    if mean_error > 2e-13 or gram_relative > 2e-12:
        raise ProtocolError("matched-control moment closure failed")
    return control, {
        "maximum_absolute_role_mean_error": mean_error,
        "maximum_relative_centered_gram_error": gram_relative,
    }


def _pair_scores(expert, cp):
    """All ordered, nonself 3x3 predecessor-to-target regression captures."""
    if expert.ndim != 3 or int(expert.shape[1]) != ROLES:
        raise ProtocolError("pair-score tensor must be neuron x 3 roles x coordinates")
    neurons = int(expert.shape[0])
    gram = cp.einsum("nrd,nsd->nrs", expert, expert)
    inverse = cp.linalg.inv(gram)
    cross = cp.empty((neurons, neurons, ROLES, ROLES), dtype=cp.float64)
    for target_role in range(ROLES):
        for predecessor_role in range(ROLES):
            cross[:, :, target_role, predecessor_role] = (
                expert[:, target_role, :] @ expert[:, predecessor_role, :].T
            )
    full = cp.einsum("ijab,jbc,ijac->ij", cross, inverse, cross)
    indices = cp.arange(neurons)
    full[indices, indices] = -cp.inf
    return full, cross, inverse


def _cycles_from_assignment(predecessor):
    neurons = len(predecessor)
    successor = [-1] * neurons
    for target, pred in enumerate(predecessor):
        if target == pred or pred < 0 or pred >= neurons or successor[pred] != -1:
            raise ProtocolError("assignment is not a non-self cycle cover")
        successor[pred] = target
    cycles = []
    unseen = set(range(neurons))
    while unseen:
        start = min(unseen)
        cycle = []
        node = start
        while node not in cycle:
            if node not in unseen:
                raise ProtocolError("cycle-cover traversal collision")
            cycle.append(node)
            unseen.remove(node)
            node = successor[node]
        if node != start:
            raise ProtocolError("cycle does not close at its start")
        cycles.append(cycle)
    return cycles


def _legal_path_from_cycle_cover(scores, np, linear_sum_assignment):
    matrix = np.asarray(scores, dtype=np.float64)
    neurons = int(matrix.shape[0])
    if matrix.shape != (neurons, neurons):
        raise ProtocolError("score matrix is not square")
    rows, columns = linear_sum_assignment(-matrix)
    if not np.array_equal(rows, np.arange(neurons)):
        raise ProtocolError("unexpected assignment row order")
    predecessor = [int(value) for value in columns]
    cycles = _cycles_from_assignment(predecessor)
    successor = {pred: target for target, pred in enumerate(predecessor)}
    segments = []
    dropped = []
    for cycle in cycles:
        weakest_target = min(cycle, key=lambda target: (matrix[target, predecessor[target]], target))
        segment = [weakest_target]
        while len(segment) < len(cycle):
            segment.append(successor[segment[-1]])
        segments.append(segment)
        dropped.append({
            "predecessor": predecessor[weakest_target],
            "target": weakest_target,
            "capture": float(matrix[weakest_target, predecessor[weakest_target]]),
        })
    segments.sort(key=lambda row: row[0])
    path = [node for segment in segments for node in segment]
    if sorted(path) != list(range(neurons)):
        raise ProtocolError("cycle-cover construction did not produce a permutation")
    captures = [float(matrix[target, pred]) for pred, target in zip(path[:-1], path[1:])]
    return {
        "cycle_count": len(cycles),
        "cycle_cover_capture": float(math.fsum(
            float(matrix[target, predecessor[target]]) for target in range(neurons)
        )),
        "dropped_edges": dropped,
        "path": path,
        "legal_path_capture": math.fsum(captures),
    }


def _metric_from_residual(residual, energy):
    if not (math.isfinite(residual) and math.isfinite(energy) and energy > 0.0 and residual > 0.0):
        raise ProtocolError("invalid residual metric inputs")
    ratio = residual / energy
    s_value = -0.5 * math.log2(ratio)
    return {
        "residual_energy": residual,
        "residual_ratio": ratio,
        "energy_reduction": 1.0 - ratio,
        "s_bpw": s_value,
        "net_s_after_side_bpw": s_value - SIDE_BPW,
        "projected_F_after_side": 2.0 ** (-2.0 * (s_value - SIDE_BPW)),
    }


def _metric_from_capture(capture, energy):
    residual = energy - capture
    if residual <= 0.0:
        raise ProtocolError("optimistic capture exhausted source energy")
    value = _metric_from_residual(residual, energy)
    value["capture"] = capture
    return value


def _fp16_path_replay(expert, path, cross, inverse, np, cp):
    neurons = int(expert.shape[0])
    if len(path) != neurons or sorted(path) != list(range(neurons)):
        raise ProtocolError("FP16 replay path is not a permutation")
    predecessors = cp.asarray(path[:-1], dtype=cp.int64)
    targets = cp.asarray(path[1:], dtype=cp.int64)
    selected_cross = cross[targets, predecessors]
    exact_coefficients = cp.einsum("eab,ebc->eac", selected_cross, inverse[predecessors])
    fp16_coefficients = exact_coefficients.astype(cp.float16)
    replay_coefficients = fp16_coefficients.astype(cp.float64)
    predicted = cp.einsum("eab,ebd->ead", replay_coefficients, expert[predecessors])
    residuals = expert[targets] - predicted
    anchor_energy = float(cp.sum(expert[int(path[0])] ** 2, dtype=cp.float64).item())
    edge_residual_energy = float(cp.sum(residuals * residuals, dtype=cp.float64).item())
    coefficient_host = cp.asnumpy(fp16_coefficients)
    coefficient_bytes = coefficient_host.astype("<f2", copy=False).tobytes()
    if len(coefficient_bytes) != (neurons - 1) * ROLES * ROLES * 2:
        raise ProtocolError("FP16 coefficient byte ledger mismatch")
    # hashlib is supplied by the independently sealed runtime and imported by
    # the caller only after bootstrap; avoid any top-level dependency here.
    digest = __import__("hashlib").sha256(coefficient_bytes).hexdigest()
    return {
        "anchor_energy": anchor_energy,
        "edge_residual_energy": edge_residual_energy,
        "residual_energy": anchor_energy + edge_residual_energy,
        "coefficient_count": (neurons - 1) * ROLES * ROLES,
        "coefficient_bytes": len(coefficient_bytes),
        "coefficient_sha256_f16le": digest,
        "maximum_absolute_exact_coefficient": float(cp.max(cp.abs(exact_coefficients)).item()),
        "maximum_absolute_rounding_error": float(
            cp.max(cp.abs(replay_coefficients - exact_coefficients)).item()
        ),
    }


def _pair_panel(experts, np, cp, linear_sum_assignment):
    """Identical relaxed, legal-exact, and legal-FP16 processing for a panel."""
    rows = []
    total_energy = 0.0
    total_relaxed_capture = 0.0
    total_legal_capture = 0.0
    total_fp16_residual = 0.0
    for ordinal, expert in enumerate(experts):
        if tuple(expert.shape) != (ROWS, ROLES, COLS):
            raise ProtocolError("production expert geometry drift")
        energy = float(cp.sum(expert * expert, dtype=cp.float64).item())
        scores, cross, inverse = _pair_scores(expert, cp)
        score_host = cp.asnumpy(scores)
        relaxed_capture = float(np.sum(np.max(score_host, axis=1), dtype=np.float64))
        legal = _legal_path_from_cycle_cover(score_host, np, linear_sum_assignment)
        path = [int(value) for value in legal.pop("path")]
        permutation_bytes = serialize_permutation(path)
        fp16 = _fp16_path_replay(expert, path, cross, inverse, np, cp)
        hashlib = __import__("hashlib")
        path_u16 = np.asarray(path, dtype="<u2").tobytes()
        rows.append({
            "expert_ordinal": ordinal,
            "source_energy": energy,
            "relaxed_reuse_exact_capture": relaxed_capture,
            "legal_exact": legal,
            "legal_fp16": fp16,
            "path_sha256_u16le": hashlib.sha256(path_u16).hexdigest(),
            "factoradic_rank_sha256_783be": hashlib.sha256(permutation_bytes).hexdigest(),
            "factoradic_physical_bytes": len(permutation_bytes),
        })
        total_energy += energy
        total_relaxed_capture += relaxed_capture
        total_legal_capture += float(legal["legal_path_capture"])
        total_fp16_residual += float(fp16["residual_energy"])
        del scores, cross, inverse
        cp.get_default_memory_pool().free_all_blocks()
    return {
        "experts": rows,
        "total_source_energy": total_energy,
        "relaxed_reuse_exact": _metric_from_capture(total_relaxed_capture, total_energy),
        "legal_path_exact": _metric_from_capture(total_legal_capture, total_energy),
        "legal_path_fp16": _metric_from_residual(total_fp16_residual, total_energy),
    }


def _metric_for_subset(panel, metric, kept):
    rows = [panel["experts"][index] for index in kept]
    energy = math.fsum(float(row["source_energy"]) for row in rows)
    if metric == "relaxed_reuse_exact":
        capture = math.fsum(float(row["relaxed_reuse_exact_capture"]) for row in rows)
        return float(_metric_from_capture(capture, energy)["s_bpw"])
    if metric == "legal_path_exact":
        capture = math.fsum(float(row["legal_exact"]["legal_path_capture"]) for row in rows)
        return float(_metric_from_capture(capture, energy)["s_bpw"])
    if metric == "legal_path_fp16":
        residual = math.fsum(float(row["legal_fp16"]["residual_energy"]) for row in rows)
        return float(_metric_from_residual(residual, energy)["s_bpw"])
    raise ValueError("unknown metric")


def _jackknife_se(values):
    count = len(values)
    if count < 2:
        return 0.0
    mean = math.fsum(values) / count
    return math.sqrt((count - 1.0) / count * math.fsum((value - mean) ** 2 for value in values))


def _controlled_statistic(qwen, controls, metric):
    qwen_s = float(qwen[metric]["s_bpw"])
    control_s = [float(row[metric]["s_bpw"]) for row in controls]
    control_mean = math.fsum(control_s) / len(control_s)
    control_mc_se = math.sqrt(
        math.fsum((value - control_mean) ** 2 for value in control_s)
        / (len(control_s) * (len(control_s) - 1))
    )
    delete_estimates = []
    expert_count = len(qwen["experts"])
    for omitted in range(expert_count):
        kept = [index for index in range(expert_count) if index != omitted]
        qwen_delete = _metric_for_subset(qwen, metric, kept)
        control_delete = math.fsum(
            _metric_for_subset(row, metric, kept) for row in controls
        ) / len(controls)
        delete_estimates.append(qwen_delete - control_delete)
    jackknife_se = _jackknife_se(delete_estimates)
    combined_se = math.hypot(control_mc_se, jackknife_se)
    excess = qwen_s - control_mean
    optimistic = excess + 3.0 * combined_se
    net_optimistic = optimistic - SIDE_BPW
    return {
        "metric": metric,
        "qwen_gross_s_bpw": qwen_s,
        "control_s_bpw": control_s,
        "control_mean_s_bpw": control_mean,
        "control_mc_se_bpw": control_mc_se,
        "delete_one_expert_estimates_s_bpw": delete_estimates,
        "delete_one_expert_jackknife_se_bpw": jackknife_se,
        "combined_se_bpw": combined_se,
        "qwen_specific_excess_s_bpw": excess,
        "optimistic_excess_plus_3se_s_bpw": optimistic,
        "net_optimistic_s_after_side_bpw": net_optimistic,
        "projected_optimistic_F_after_side": 2.0 ** (-2.0 * net_optimistic),
        "required_gross_s_bpw": REQUIRED_GROSS_S,
        "upper_confidence_survives_target": optimistic >= REQUIRED_GROSS_S,
    }


def _decision_after_legal_statistics(qwen, statistics):
    """Make no control-corrected decision until legal FP16 is available.

    The corrected relaxed statistic is reported for diagnostics only.  It is
    mathematically noncontaining and therefore cannot kill or promote.
    """
    if "legal_path_fp16" not in statistics:
        raise ProtocolError("corrected legal FP16 statistic required before decision")
    legal_fp16 = statistics["legal_path_fp16"]
    gross_legal = float(qwen["legal_path_fp16"]["s_bpw"])
    if gross_legal >= REQUIRED_GROSS_S and legal_fp16["upper_confidence_survives_target"]:
        return "SURVIVE_SOURCE_ORACLE_FP16_PATH_RESIDUAL_CODEC_REQUIRED"
    return "AMBIGUOUS_LEGAL_FP16_PATH_MISS_STRONGER_PATH_SOLVER_REQUIRED"


def _direct_stage(experts, np, cp, linear_sum_assignment):
    qwen = _pair_panel(experts, np, cp, linear_sum_assignment)
    qwen_relaxed_s = float(qwen["relaxed_reuse_exact"]["s_bpw"])
    if qwen_relaxed_s < REQUIRED_GROSS_S:
        return {
            "qwen": qwen,
            "controls": [],
            "control_moment_closure": [],
            "statistics": {},
            "decision": "HARD_KILL_GROSS_QWEN_RELAXED_NECESSARY_BOUND",
            "early_stop": True,
            "reason": "Gross Qwen target-wise reuse contains every legal exact path.",
        }

    controls = []
    closures = []
    for replicate, base_seed in enumerate(CONTROL_SEEDS):
        arrays = []
        replicate_closure = []
        for ordinal, expert in enumerate(experts):
            control, closure = _matched_control(expert, base_seed + 1009 * ordinal, cp)
            arrays.append(control)
            replicate_closure.append(closure)
        # Each control panel computes legal exact and legal FP16 before any
        # corrected decision can be reached.
        controls.append(_pair_panel(arrays, np, cp, linear_sum_assignment))
        closures.append({"replicate": replicate, "seed": base_seed, "experts": replicate_closure})
        del arrays
        cp.get_default_memory_pool().free_all_blocks()

    # Compute the corrected legal FP16 statistic first and require it in the
    # decision helper.  Relaxed correction follows as diagnostic-only output.
    statistics = {
        "legal_path_fp16": _controlled_statistic(qwen, controls, "legal_path_fp16"),
        "legal_path_exact": _controlled_statistic(qwen, controls, "legal_path_exact"),
        "relaxed_reuse_exact": _controlled_statistic(qwen, controls, "relaxed_reuse_exact"),
    }
    statistics["relaxed_reuse_exact"]["containing_claim"] = False
    statistics["relaxed_reuse_exact"]["decision_eligible"] = False
    decision = _decision_after_legal_statistics(qwen, statistics)
    return {
        "qwen": qwen,
        "controls": controls,
        "control_moment_closure": closures,
        "statistics": statistics,
        "decision": decision,
        "early_stop": False,
        "claim_boundary": (
            "Gross Qwen relaxed reuse is a deterministic necessary bound only. "
            "Control-corrected relaxed reuse is diagnostic and never containing. "
            "The cycle-cover path remains achievable but not claimed optimal."
        ),
    }


def adversarial_n8_statistics():
    """Exact audit construction, expressed with integer ratios.

    Q uses per-role AR(1) correlation r=7/8.  Every matched control uses the
    star geometry with hub/leaf r and leaf/leaf rho=r^2.  Roles occupy three
    orthogonal subspaces.  r and rho are exactly representable in binary16.
    """
    n = 8
    r = 7 / 8
    rho = 49 / 64
    total_energy = 3 * n
    q_relaxed_capture = 3 * n * rho
    control_relaxed_capture = 3 * n * rho
    q_legal_capture = 3 * (n - 1) * rho
    control_legal_capture = 3 * (2 * rho + (n - 3) * rho * rho)
    q_relaxed = _metric_from_capture(q_relaxed_capture, total_energy)["s_bpw"]
    control_relaxed = _metric_from_capture(control_relaxed_capture, total_energy)["s_bpw"]
    q_legal = _metric_from_capture(q_legal_capture, total_energy)["s_bpw"]
    control_legal = _metric_from_capture(control_legal_capture, total_energy)["s_bpw"]
    return {
        "n": n,
        "r": r,
        "rho": rho,
        "binary16_coefficients": [r, rho],
        "qwen_relaxed_s_bpw": q_relaxed,
        "control_relaxed_s_bpw": control_relaxed,
        "corrected_relaxed_s_bpw": q_relaxed - control_relaxed,
        "qwen_legal_fp16_s_bpw": q_legal,
        "control_legal_fp16_s_bpw": control_legal,
        "corrected_legal_fp16_s_bpw": q_legal - control_legal,
        "required_gross_s_bpw": REQUIRED_GROSS_S,
    }


def source_only_status():
    return {
        "schema": SCHEMA,
        "status": "SOURCE_ONLY_DEPLOYMENT_BLOCKED",
        "source_access_authorized": False,
        "calibration_authorized": False,
        "gross_relaxed_necessary_bound_only": True,
        "control_corrected_relaxed_containing": False,
        "corrected_legal_fp16_required_before_control_decision": True,
    }


if __name__ == "__main__":
    # This package intentionally contains no production authorization path.
    raise SystemExit("FOSP_V3_SOURCE_ONLY_DEPLOYMENT_BLOCKED")
