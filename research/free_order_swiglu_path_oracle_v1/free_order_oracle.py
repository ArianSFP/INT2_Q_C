#!/usr/bin/env python3
"""CuPy opportunity gate for charged, within-expert SwiGLU path ordering.

This file is source-only until the package receives a separate independent
audit and execution authorization.  It has no argument for a model panel or
an alternate source manifest.  The fixed auxiliary bindings live beside this
file.  CuPy, NumPy, and SciPy are imported only after CLI/firewall checks.

The zero-bit permutation proposal is intentionally *not* implemented.  The
eligible codec contract serializes the factoradic permutation rank in the
same expert frame, then scatters the decoded triplet back to the original
coordinates before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "free_order_swiglu_path_auxiliary_result_v1"
ROWS = 768
COLS = 2048
ROLES = 3
WEIGHTS_PER_EXPERT = ROWS * COLS * ROLES
REQUIRED_S = -0.5 * math.log2(0.8)
RATES = (2.15, 2.3, 2.5)
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
BINDINGS_NAME = "source_bindings.json"
BINDINGS_SHA256 = "3454b718a65efc02c32463f955c10ff393f4218fac04f358107960ff3735990d"


class ProtocolError(RuntimeError):
    """Fail-closed protocol violation."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def ceil_log2_factorial(n: int) -> int:
    if n < 0:
        raise ValueError("negative factorial")
    return (math.factorial(n) - 1).bit_length()


def rank_permutation(permutation: Sequence[int]) -> int:
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


def unrank_permutation(n: int, rank: int) -> tuple[int, ...]:
    if n < 0 or rank < 0 or rank >= math.factorial(n):
        raise ValueError("factoradic rank outside domain")
    available = list(range(n))
    output: list[int] = []
    for remaining in range(n, 0, -1):
        factorial = math.factorial(remaining - 1)
        position, rank = divmod(rank, factorial)
        output.append(available.pop(position))
    return tuple(output)


def frame_ledger(rate: float, coefficient_bits: int) -> dict[str, Any]:
    if rate not in RATES or coefficient_bits < 0:
        raise ValueError("invalid ledger arguments")
    frame_bytes = math.floor(WEIGHTS_PER_EXPERT * rate / 8.0)
    actual_rate = 8.0 * frame_bytes / WEIGHTS_PER_EXPERT
    header_bits = 64 * 8
    permutation_bits = 783 * 8
    side_bits = header_bits + permutation_bits + coefficient_bits
    payload_bits = 8 * frame_bytes - side_bits
    if payload_bits <= 0:
        raise AssertionError("negative payload reservoir")
    cold_pages = math.ceil(frame_bytes / 4096) + 1
    page_bytes = cold_pages * 4096
    return {
        "requested_rate_bpw": rate,
        "frame_bytes": frame_bytes,
        "actual_rate_bpw": actual_rate,
        "header_bits": header_bits,
        "factoradic_bits_physical": permutation_bits,
        "coefficient_bits": coefficient_bits,
        "side_bpw": side_bits / WEIGHTS_PER_EXPERT,
        "payload_bits": payload_bits,
        "payload_bpw": payload_bits / WEIGHTS_PER_EXPERT,
        "logical_byte_read_amplification": 1.0,
        "cold_page_bytes_including_one_shared_page": page_bytes,
        "cold_page_amplification": page_bytes / frame_bytes,
        "strictly_below_2x": page_bytes / frame_bytes < 2.0,
        "required_gross_s_after_side_bpw": REQUIRED_S + side_bits / WEIGHTS_PER_EXPERT,
    }


def _regular_bytes_no_follow(path: Path, expected_bytes: int | None = None) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(os.fspath(path), flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ProtocolError(f"not a regular file: {path}")
        if expected_bytes is not None and info.st_size != expected_bytes:
            raise ProtocolError(
                f"wrong byte count for {path}: {info.st_size} != {expected_bytes}"
            )
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise ProtocolError(f"short read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProtocolError(f"file grew during read: {path}")
        after = os.fstat(descriptor)
        identity_before = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise ProtocolError(f"file changed during read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_bindings(package: Path) -> tuple[dict[str, Any], bytes]:
    path = package / BINDINGS_NAME
    raw = _regular_bytes_no_follow(path)
    if sha256_bytes(raw) != BINDINGS_SHA256:
        raise ProtocolError("fixed source-bindings hash mismatch")
    value = json.loads(raw.decode("utf-8"))
    if value.get("schema") != "free_order_swiglu_path_auxiliary_bindings_v1":
        raise ProtocolError("wrong source-bindings schema")
    experts = value.get("experts")
    if not isinstance(experts, list) or len(experts) != 2:
        raise ProtocolError("expected exactly two fixed auxiliary experts")
    if [int(row["ordinal"]) for row in experts] != [0, 1]:
        raise ProtocolError("expert ordinals are not frozen")
    for expert in experts:
        roles = expert.get("roles")
        if [row.get("role") for row in roles] != ["gate", "up", "down"]:
            raise ProtocolError("all three roles are required in frozen order")
    return value, raw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--authorization-sentinel",
        required=True,
        help="must equal the frozen literal after a separate independent source audit",
    )
    return parser.parse_args()


def _preflight_paths(args: argparse.Namespace, package: Path) -> tuple[Path, Path]:
    if args.authorization_sentinel != "INDEPENDENT_SOURCE_AUDIT_PASS_REQUIRED":
        raise ProtocolError("this source-only package is not execution-authorized")
    root = args.workspace_root.absolute()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ProtocolError("output already exists")
    if output.parent.resolve(strict=True) == package.resolve(strict=True):
        raise ProtocolError("runtime output may not mutate the frozen package")
    root = root.resolve(strict=True)
    output_parent = output.parent.resolve(strict=True)
    return root, output_parent / output.name


def _decode_bf16(raw: bytes, shape: tuple[int, int], np: Any, cp: Any) -> Any:
    words = np.frombuffer(raw, dtype="<u2")
    if words.size != math.prod(shape):
        raise ProtocolError("BF16 shape mismatch")
    gpu_words = cp.asarray(words, dtype=cp.uint16)
    values = (gpu_words.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)
    return values.reshape(shape).astype(cp.float64)


def _load_sources(root: Path, bindings: dict[str, Any], np: Any, cp: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    experts: list[Any] = []
    receipts: list[dict[str, Any]] = []
    for expert in bindings["experts"]:
        role_arrays: list[Any] = []
        for role in expert["roles"]:
            relative = Path(str(role["relative_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ProtocolError("unsafe fixed relative source path")
            path = root / relative
            raw = _regular_bytes_no_follow(path, int(bindings["expected_geometry"]["bf16_bytes_per_matrix"]))
            digest = sha256_bytes(raw)
            if digest != str(role["sha256"]):
                raise ProtocolError(f"source hash mismatch: {relative.as_posix()}")
            shape = tuple(int(x) for x in role["shape"])
            array = _decode_bf16(raw, shape, np, cp)
            if role["role"] == "down":
                array = cp.ascontiguousarray(array.T)
            if array.shape != (ROWS, COLS):
                raise ProtocolError("canonical role shape mismatch")
            role_arrays.append(array)
            receipts.append(
                {
                    "ordinal": int(expert["ordinal"]),
                    "layer": int(expert["layer"]),
                    "expert": int(expert["expert"]),
                    "role": str(role["role"]),
                    "relative_path": relative.as_posix(),
                    "bytes": len(raw),
                    "sha256": digest,
                }
            )
        experts.append(cp.stack(role_arrays, axis=1))  # neuron, role, model coordinate
    return experts, receipts


def _reverse_waterfill(variances: Any, rate: float, cp: Any) -> dict[str, float]:
    if rate <= 0.0:
        raise ValueError("nonpositive rate")
    variances = cp.asarray(variances, dtype=cp.float64)
    if bool(cp.any(variances <= 0.0)) or not bool(cp.all(cp.isfinite(variances))):
        raise ProtocolError("invalid covariance eigenvalue")
    log_low = float(cp.log(cp.min(variances)).item()) - 2.0 * rate * math.log(2.0) - 8.0
    log_high = float(cp.log(cp.max(variances)).item())
    for _ in range(100):
        log_mid = 0.5 * (log_low + log_high)
        theta = math.exp(log_mid)
        observed = float(
            (0.5 * cp.mean(cp.maximum(cp.log2(variances / theta), 0.0))).item()
        )
        if observed > rate:
            log_low = log_mid
        else:
            log_high = log_mid
    theta = math.exp(0.5 * (log_low + log_high))
    normalized_distortion = float(
        (cp.mean(cp.minimum(variances, theta)) / cp.mean(variances)).item()
    )
    f_value = normalized_distortion * 2.0 ** (2.0 * rate)
    return {
        "rate_bpw": rate,
        "waterlevel": theta,
        "active_components": int(cp.count_nonzero(variances > theta).item()),
        "components": int(variances.size),
        "normalized_distortion": normalized_distortion,
        "F": f_value,
        "s_bpw": -0.5 * math.log2(f_value),
    }


def _dense_curve(experts: Sequence[Any], cp: Any) -> dict[str, Any]:
    eigenvalues: list[Any] = []
    energies: list[float] = []
    for expert in experts:
        if expert.shape != (ROWS, ROLES, COLS):
            raise ProtocolError("expert geometry drift")
        energies.append(float(cp.sum(expert * expert, dtype=cp.float64).item()))
        for role in range(ROLES):
            matrix = expert[:, role, :]
            covariance = (matrix @ matrix.T) / float(COLS)
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues.append(cp.linalg.eigvalsh(covariance))
    spectrum = cp.concatenate(eigenvalues)
    curve = [_reverse_waterfill(spectrum, rate, cp) for rate in RATES]
    return {
        "experts": len(experts),
        "source_energy": math.fsum(energies),
        "minimum_eigenvalue": float(cp.min(spectrum).item()),
        "maximum_eigenvalue": float(cp.max(spectrum).item()),
        "spectrum_sha256_f64le": sha256_bytes(cp.asnumpy(spectrum).astype("<f8", copy=False).tobytes()),
        "rates": curve,
    }


def _s_at(curve: dict[str, Any], rate: float) -> float:
    rows = [row for row in curve["rates"] if float(row["rate_bpw"]) == rate]
    if len(rows) != 1:
        raise AssertionError("missing rate row")
    return float(rows[0]["s_bpw"])


def _jackknife_se(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = math.fsum(values) / n
    return math.sqrt((n - 1.0) / n * math.fsum((value - mean) ** 2 for value in values))


def _matched_control(source: Any, seed: int, cp: Any) -> tuple[Any, dict[str, float]]:
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
    source_root = cp.einsum(
        "nri,ni,nsi->nrs", source_evec, cp.sqrt(source_eval), source_evec
    )
    raw_invroot = cp.einsum(
        "nri,ni,nsi->nrs", raw_evec, 1.0 / cp.sqrt(raw_eval), raw_evec
    )
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
        raise ProtocolError(
            f"matched-control moment closure failed: mean={mean_error}, gram={gram_relative}"
        )
    return control, {
        "maximum_absolute_role_mean_error": mean_error,
        "maximum_relative_centered_gram_error": gram_relative,
    }


def _dense_stage(experts: Sequence[Any], cp: Any) -> tuple[dict[str, Any], list[list[Any]]]:
    qwen = _dense_curve(experts, cp)
    controls: list[dict[str, Any]] = []
    control_arrays: list[list[Any]] = []
    for replicate, base_seed in enumerate(CONTROL_SEEDS):
        arrays: list[Any] = []
        closures: list[dict[str, float]] = []
        for ordinal, expert in enumerate(experts):
            control, closure = _matched_control(expert, base_seed + 1009 * ordinal, cp)
            arrays.append(control)
            closures.append(closure)
        control_arrays.append(arrays)
        controls.append(
            {
                "replicate": replicate,
                "seed": base_seed,
                "moment_closure": closures,
                "curve": _dense_curve(arrays, cp),
            }
        )

    qwen_delete = [_dense_curve([x for j, x in enumerate(experts) if j != i], cp) for i in range(len(experts))]
    control_delete = [
        [_dense_curve([x for j, x in enumerate(arrays) if j != i], cp) for i in range(len(experts))]
        for arrays in control_arrays
    ]
    gates: list[dict[str, float]] = []
    for rate in RATES:
        qwen_s = _s_at(qwen, rate)
        control_s = [_s_at(row["curve"], rate) for row in controls]
        control_mean = math.fsum(control_s) / len(control_s)
        control_mc_se = math.sqrt(
            math.fsum((x - control_mean) ** 2 for x in control_s)
            / (len(control_s) * (len(control_s) - 1))
        )
        deletes: list[float] = []
        for omitted in range(len(experts)):
            q = _s_at(qwen_delete[omitted], rate)
            c = math.fsum(_s_at(row[omitted], rate) for row in control_delete) / len(control_delete)
            deletes.append(q - c)
        jackknife = _jackknife_se(deletes)
        combined = math.hypot(control_mc_se, jackknife)
        excess = qwen_s - control_mean
        optimistic = excess + 3.0 * combined
        gates.append(
            {
                "rate_bpw": rate,
                "qwen_s_bpw": qwen_s,
                "control_mean_s_bpw": control_mean,
                "control_mc_se_bpw": control_mc_se,
                "delete_expert_jackknife_se_bpw": jackknife,
                "combined_se_bpw": combined,
                "qwen_specific_excess_s_bpw": excess,
                "optimistic_excess_plus_3se_bpw": optimistic,
                "fraction_of_required_s": optimistic / REQUIRED_S,
            }
        )
    best = max(gates, key=lambda row: row["optimistic_excess_plus_3se_bpw"])
    decision = (
        "SURVIVE_DENSE_SECOND_ORDER_GATE_RUN_PAIR_STAGE"
        if best["optimistic_excess_plus_3se_bpw"] >= REQUIRED_S
        else "EARLY_KILL_BEFORE_PAIR_STAGE"
    )
    return {
        "qwen": qwen,
        "controls": controls,
        "delete_expert_qwen": qwen_delete,
        "rate_gates": gates,
        "best_gate": best,
        "decision": decision,
        "claim_boundary": (
            "The free three-basis dense KLT is a favorable second-order envelope. "
            "Failure is an opportunity kill for linear adjacency/covariance prediction, "
            "not a universal theorem for nonlinear higher-order codes."
        ),
    }, control_arrays


def _pair_scores(expert: Any, cp: Any) -> tuple[Any, Any]:
    gram = cp.einsum("nrd,nsd->nrs", expert, expert)
    inverse = cp.linalg.inv(gram)
    cross = cp.empty((ROWS, ROWS, ROLES, ROLES), dtype=cp.float64)
    for target_role in range(ROLES):
        for predecessor_role in range(ROLES):
            cross[:, :, target_role, predecessor_role] = (
                expert[:, target_role, :] @ expert[:, predecessor_role, :].T
            )
    full = cp.einsum("ijab,jbc,ijac->ij", cross, inverse, cross)
    diagonal = cp.zeros((ROWS, ROWS), dtype=cp.float64)
    for role in range(ROLES):
        denominator = gram[:, role, role][None, :]
        diagonal += cross[:, :, role, role] ** 2 / denominator
    indices = cp.arange(ROWS)
    full[indices, indices] = -cp.inf
    diagonal[indices, indices] = -cp.inf
    return diagonal, full


def _cycles_from_assignment(predecessor: Sequence[int]) -> list[list[int]]:
    n = len(predecessor)
    successor = [-1] * n
    for target, pred in enumerate(predecessor):
        if target == pred or pred < 0 or pred >= n or successor[pred] != -1:
            raise ProtocolError("assignment is not a non-self cycle cover")
        successor[pred] = target
    cycles: list[list[int]] = []
    unseen = set(range(n))
    while unseen:
        start = min(unseen)
        cycle: list[int] = []
        node = start
        while node not in cycle:
            if node not in unseen:
                raise ProtocolError("cycle cover traversal collision")
            cycle.append(node)
            unseen.remove(node)
            node = successor[node]
        if node != start:
            raise ProtocolError("cycle does not close at its start")
        cycles.append(cycle)
    return cycles


def _legal_path_from_cycle_cover(scores: Any, np: Any, linear_sum_assignment: Any) -> dict[str, Any]:
    matrix = np.asarray(scores, dtype=np.float64)
    rows, columns = linear_sum_assignment(-matrix)
    if not np.array_equal(rows, np.arange(ROWS)):
        raise ProtocolError("unexpected assignment row order")
    predecessor = [int(x) for x in columns]
    cycles = _cycles_from_assignment(predecessor)
    segments: list[list[int]] = []
    dropped: list[dict[str, Any]] = []
    for cycle in cycles:
        weakest_target = min(cycle, key=lambda target: (matrix[target, predecessor[target]], target))
        start = weakest_target
        successor = {pred: target for target, pred in enumerate(predecessor)}
        segment = [start]
        while len(segment) < len(cycle):
            segment.append(successor[segment[-1]])
        segments.append(segment)
        dropped.append(
            {
                "predecessor": predecessor[weakest_target],
                "target": weakest_target,
                "capture": float(matrix[weakest_target, predecessor[weakest_target]]),
            }
        )
    segments.sort(key=lambda row: row[0])
    path = [node for segment in segments for node in segment]
    if sorted(path) != list(range(ROWS)):
        raise ProtocolError("constructed path is not a permutation")
    captures = [float(matrix[target, pred]) for pred, target in zip(path[:-1], path[1:])]
    return {
        "cycle_count": len(cycles),
        "cycle_cover_capture": float(sum(matrix[target, predecessor[target]] for target in range(ROWS))),
        "dropped_edges": dropped,
        "path": path,
        "path_sha256_u16le": sha256_bytes(np.asarray(path, dtype="<u2").tobytes()),
        "legal_path_capture": math.fsum(captures),
    }


def _pair_panel(experts: Sequence[Any], np: Any, cp: Any, linear_sum_assignment: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_energy = 0.0
    total_reuse_diag = 0.0
    total_reuse_full = 0.0
    total_path_diag = 0.0
    total_path_full = 0.0
    for ordinal, expert in enumerate(experts):
        energy = float(cp.sum(expert * expert, dtype=cp.float64).item())
        diagonal, full = _pair_scores(expert, cp)
        diagonal_host = cp.asnumpy(diagonal)
        full_host = cp.asnumpy(full)
        reuse_diag = float(np.sum(np.max(diagonal_host, axis=1), dtype=np.float64))
        reuse_full = float(np.sum(np.max(full_host, axis=1), dtype=np.float64))
        path_diag = _legal_path_from_cycle_cover(diagonal_host, np, linear_sum_assignment)
        path_full = _legal_path_from_cycle_cover(full_host, np, linear_sum_assignment)
        rows.append(
            {
                "expert_ordinal": ordinal,
                "source_energy": energy,
                "relaxed_reuse_diag3_capture": reuse_diag,
                "relaxed_reuse_full3x3_capture": reuse_full,
                "diag3": path_diag,
                "full3x3": path_full,
            }
        )
        total_energy += energy
        total_reuse_diag += reuse_diag
        total_reuse_full += reuse_full
        total_path_diag += float(path_diag["legal_path_capture"])
        total_path_full += float(path_full["legal_path_capture"])

    def metric(capture: float) -> dict[str, float]:
        residual = 1.0 - capture / total_energy
        if not 0.0 < residual <= 1.0:
            raise ProtocolError("invalid pair residual")
        return {
            "capture": capture,
            "energy_reduction": 1.0 - residual,
            "residual_ratio": residual,
            "s_bpw": -0.5 * math.log2(residual),
            "fraction_of_required_s": (-0.5 * math.log2(residual)) / REQUIRED_S,
        }

    return {
        "experts": rows,
        "total_source_energy": total_energy,
        "relaxed_reuse_diag3": metric(total_reuse_diag),
        "relaxed_reuse_full3x3": metric(total_reuse_full),
        "legal_path_diag3": metric(total_path_diag),
        "legal_path_full3x3": metric(total_path_full),
    }


def _pair_stage(experts: Sequence[Any], controls: Sequence[Sequence[Any]], np: Any, cp: Any, linear_sum_assignment: Any) -> dict[str, Any]:
    qwen = _pair_panel(experts, np, cp, linear_sum_assignment)
    control_rows = [_pair_panel(row, np, cp, linear_sum_assignment) for row in controls]
    key = "relaxed_reuse_full3x3"
    control_s = [float(row[key]["s_bpw"]) for row in control_rows]
    control_mean = math.fsum(control_s) / len(control_s)
    control_se = math.sqrt(
        math.fsum((x - control_mean) ** 2 for x in control_s)
        / (len(control_s) * (len(control_s) - 1))
    )
    qwen_s = float(qwen[key]["s_bpw"])
    specific_upper = qwen_s - control_mean + 3.0 * control_se
    gross_kill = qwen_s < REQUIRED_S
    return {
        "qwen": qwen,
        "controls": control_rows,
        "gate": {
            "qwen_relaxed_gross_s_bpw": qwen_s,
            "control_mean_s_bpw": control_mean,
            "control_mc_se_bpw": control_se,
            "qwen_specific_plus_3se_s_bpw": specific_upper,
            "gross_relaxation_below_required_s": gross_kill,
            "decision": (
                "EARLY_KILL_CHARGED_PATH_FAMILY"
                if gross_kill or specific_upper < REQUIRED_S
                else "SURVIVE_SOURCE_ORACLE_ONLY_FINITE_BRIDGE_REQUIRED"
            ),
        },
    }


def _write_create_new(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise ProtocolError("short result write")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    started = time.time()
    args = _parse_args()
    package = Path(__file__).absolute().parent.resolve(strict=True)
    bindings, bindings_raw = _load_bindings(package)
    root, output = _preflight_paths(args, package)

    import numpy as np  # type: ignore
    import cupy as cp  # type: ignore
    import scipy  # type: ignore
    from scipy.optimize import linear_sum_assignment  # type: ignore

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ProtocolError("CUDA_VISIBLE_DEVICES must be exactly 0")
    cp.cuda.Device(0).use()
    experts, receipts = _load_sources(root, bindings, np, cp)
    dense, controls = _dense_stage(experts, cp)
    pair: dict[str, Any] | None = None
    if dense["decision"] == "SURVIVE_DENSE_SECOND_ORDER_GATE_RUN_PAIR_STAGE":
        pair = _pair_stage(experts, controls, np, cp, linear_sum_assignment)

    coefficient_bits = {
        "diag3_fp16": 767 * 3 * 16,
        "full3x3_fp16": 767 * 9 * 16,
        "diag3_fixed_nibble": math.ceil(767 * 3 * 4 / 8) * 8,
        "full3x3_fixed_nibble": math.ceil(767 * 9 * 4 / 8) * 8,
    }
    ledgers = {
        name: [frame_ledger(rate, bits) for rate in RATES]
        for name, bits in coefficient_bits.items()
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "COMPLETE_AUXILIARY_OPPORTUNITY_GATE",
        "metric": "original-coordinate pooled BF16 squared error",
        "zero_bit_permutation_variant": {
            "decision": "INELIGIBLE_IMMEDIATE_KILL",
            "reason": "decoder cannot recover original arbitrary labels from a quotient payload",
        },
        "eligible_variant": {
            "factoradic_information_bits": ceil_log2_factorial(ROWS),
            "factoradic_physical_bytes": 783,
            "factoradic_physical_bpw": 783 * 8 / WEIGHTS_PER_EXPERT,
            "scatter_to_original_coordinates_before_score": True,
            "one_local_frame_read": True,
        },
        "objective": {
            "required_s_bpw": REQUIRED_S,
            "rates_bpw": list(RATES),
            "read_limit": 2.0,
        },
        "dense_stage": dense,
        "pair_stage": pair,
        "rate_read_ledgers": ledgers,
        "source_receipts": receipts,
        "bindings_sha256": sha256_bytes(bindings_raw),
        "backend": {
            "python": sys.version,
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "scipy": scipy.__version__,
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        },
        "execution": {
            "script_sha256": sha256_bytes(_regular_bytes_no_follow(Path(__file__).resolve())),
            "elapsed_seconds": time.time() - started,
            "pinned_panel_opened": False,
            "package_mutated": False,
        },
        "claim_boundary": (
            "This is an auxiliary ideal opportunity gate. It emits no compressed stream and "
            "cannot establish a Qwen or pinned-panel compression result."
        ),
    }
    result["canonical_unsigned_sha256"] = canonical_sha256(result)
    _write_create_new(output, result)
    print(json.dumps({"output": str(output), "decision": dense["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
