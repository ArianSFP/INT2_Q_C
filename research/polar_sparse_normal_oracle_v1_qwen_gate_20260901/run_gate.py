#!/usr/bin/env python3
"""Authenticated CuPy gate for source-leaky sparse polar-normal oracles.

Only the frozen 18-matrix source panel is read.  The prior derivation module is
authenticated before import and its binary64 normal hashes must reproduce
exactly.  Selected values remain continuous/exact despite their one-bit ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any


ROWS = 768
COLS = 2048
MATRICES = 18
EXPERTS = 6
VALUES = ROWS * COLS
PANEL_VALUES = MATRICES * VALUES
UNIQUE = ROWS * (ROWS + 1) // 2
RATE = 2.5
TARGET_F = 0.8
BASE_SIDE_BPW = 0.0011701230649594908
NUMERIC_ALLOWANCE_DISTORTION = 1e-9
CONTROL_SEEDS = (
    27_090_101,
    27_090_119,
    27_090_143,
    27_090_171,
    27_090_207,
    27_090_231,
    27_090_263,
    27_090_299,
)

EXPECTED = {
    "parent_script": "04d033319b4bbab037b48355e5f296274ae77b1c787ddcd2508e9b58948d265e",
    "parent_result": "e4fecac5f676d84739972bbf0e04467027aeae1356e62e1dc3cd2b84bff67026",
    "composite_result": "565e1eb2122f2e476c5bd81e4205eeb3e4cede6e6a51149e95944355199eb41c",
    "source_manifest": "23a7566cf9ead5191c778a9dda30e880646a32d81357ff940182dc74e11bfe99",
    "source_protocol": "d519f217263823085e39b3b291863f123603fcb540ba912ffe25c0d13bfacf43",
    "cupy_primitives": "80b9608aaadaddc34c98fa70485b3e1de25074bdd423fdb211fa4d9d8d22bbee",
    "source_lock": "bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def verify_json_seal(report: dict[str, Any], field: str = "result_lock_sha256") -> None:
    claimed = report.get(field)
    clean = dict(report)
    clean.pop(field, None)
    observed = hashlib.sha256(canonical(clean)).hexdigest()
    if claimed != observed:
        raise AssertionError((field, claimed, observed))


def write_sealed(path: Path, report: dict[str, Any]) -> None:
    clean = dict(report)
    clean.pop("result_lock_sha256", None)
    clean["result_lock_sha256"] = hashlib.sha256(canonical(clean)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AssertionError((label, "regular nonsymlink file required", path))
    observed = sha256_file(path)
    if observed != expected:
        raise AssertionError((label, expected, observed))
    return {
        "label": label,
        "path": str(path.resolve()),
        "sha256": observed,
        "nbytes": path.stat().st_size,
    }


def authenticated_import(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError((module_name, "import spec"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def exact_choose_support(n: int) -> Any:
    """Return exact ceil(log2(C(n,k))) for every k, using integer recurrence."""

    import numpy as np

    if n < 0:
        raise ValueError(n)
    result = np.empty(n + 1, dtype=np.int64)
    coefficient = 1
    result[0] = 0
    for k in range(1, n + 1):
        coefficient = coefficient * (n - k + 1) // k
        bit_length = coefficient.bit_length()
        result[k] = bit_length - 1 if coefficient == 1 << (bit_length - 1) else bit_length
    if int(result[0]) != 0 or int(result[-1]) != 0:
        raise AssertionError((n, "support endpoints"))
    if not np.array_equal(result, result[::-1]):
        raise AssertionError((n, "support symmetry"))
    return result


def model_phi(dimension: float, energy: float, multiplier: float) -> tuple[float, float]:
    if energy <= 0.0:
        return 0.0, 0.0
    critical = 2.0 * math.log(2.0) * energy / dimension
    if multiplier >= critical:
        return energy, 0.0
    rate = dimension / (2.0 * math.log(2.0)) * math.log(critical / multiplier)
    objective = multiplier * dimension / (2.0 * math.log(2.0)) + multiplier * rate
    return objective, rate


def curve_length(curve: dict[str, Any]) -> int:
    return int(curve["capture"].shape[1])


def evaluate_dual(
    cp: Any,
    records: list[Any],
    total_source_energy: float,
    curve: dict[str, Any],
    multiplier: float,
    value_bits: int,
) -> dict[str, Any]:
    if not multiplier > 0.0:
        raise ValueError(multiplier)
    value = -multiplier * (RATE - BASE_SIDE_BPW)
    payload_rate = 0.0
    side_rate = 0.0
    for record in records:
        objective, rate = model_phi(
            record.model_dof / PANEL_VALUES,
            record.model_energy / total_source_energy,
            multiplier,
        )
        value += objective
        payload_rate += rate

    support = curve["support"]
    selected: list[dict[str, Any]] = []
    two_ln2 = 2.0 * math.log(2.0)
    for ordinal, record in enumerate(records):
        capture = curve["capture"][ordinal]
        symbols = curve["symbols"] if curve["symbols"].ndim == 1 else curve["symbols"][ordinal]
        removed = curve["removed"] if curve["removed"].ndim == 1 else curve["removed"][ordinal]
        residual = cp.maximum(float(record.normal_energy) - capture, 0.0) / total_source_energy
        remaining = cp.maximum(
            1,
            int(record.normal_dof) - cp.minimum(removed, int(record.normal_dof) - 1),
        ).astype(cp.float64) / PANEL_VALUES
        critical = two_ln2 * residual / remaining
        active = multiplier < critical
        ratio = cp.maximum(critical / multiplier, 1.0)
        phi = cp.where(
            residual <= 0.0,
            0.0,
            cp.where(
                active,
                multiplier * remaining / two_ln2 * (1.0 + cp.log(ratio)),
                residual,
            ),
        )
        option_side = (support + value_bits * symbols) / PANEL_VALUES
        objective = phi + multiplier * option_side
        index = int(cp.argmin(objective).item())
        best_objective = float(objective[index].item())
        best_residual = float(residual[index].item())
        best_dimension = float(remaining[index].item())
        best_side = float(option_side[index].item())
        best_critical = two_ln2 * best_residual / best_dimension if best_residual > 0.0 else 0.0
        best_payload = (
            best_dimension / two_ln2 * math.log(best_critical / multiplier)
            if multiplier < best_critical
            else 0.0
        )
        value += best_objective
        payload_rate += best_payload
        side_rate += best_side
        selected.append(
            {
                "matrix_ordinal": ordinal,
                "option_index": index,
                curve["option_label"]: index,
                "support_bits_exact": int(support[index].item()),
                "value_symbols": int(symbols[index].item()),
                "removed_dof": int(min(int(removed[index].item()), record.normal_dof - 1)),
                "captured_normal_energy_fraction": float(
                    min(float(capture[index].item()), record.normal_energy) / record.normal_energy
                ),
                "side_rate_bpw": best_side,
                "payload_rate_bpw": best_payload,
            }
        )

    conservative = max(0.0, value - NUMERIC_ALLOWANCE_DISTORTION)
    f_lower = conservative * 2.0 ** (2.0 * RATE)
    return {
        "multiplier": multiplier,
        "raw_dual_distortion": value,
        "numeric_allowance_distortion": NUMERIC_ALLOWANCE_DISTORTION,
        "dual_distortion_lower_bound": conservative,
        "dual_F_lower_bound": f_lower,
        "hard_kill_at_multiplier": bool(f_lower > TARGET_F),
        "implied_total_rate_bpw": BASE_SIDE_BPW + side_rate + payload_rate,
        "support_value_rate_bpw": side_rate,
        "payload_rate_bpw": payload_rate,
        "selected_options": selected,
    }


def scan_dual(
    cp: Any,
    records: list[Any],
    total_source_energy: float,
    curve: dict[str, Any],
    value_bits: int,
) -> dict[str, Any]:
    import numpy as np

    cache: dict[float, dict[str, Any]] = {}

    def evaluate(multiplier: float) -> dict[str, Any]:
        key = float(multiplier)
        if key not in cache:
            cache[key] = evaluate_dual(
                cp, records, total_source_energy, curve, key, value_bits
            )
        return cache[key]

    grid = np.logspace(-7.0, 0.0, 57, dtype=np.float64)
    rows = [evaluate(float(x)) for x in grid]
    best_index = max(range(len(rows)), key=lambda i: rows[i]["dual_distortion_lower_bound"])
    lower = float(grid[max(0, best_index - 1)])
    upper = float(grid[min(len(grid) - 1, best_index + 1)])
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - golden * (upper - lower)
    right = lower + golden * (upper - lower)
    left_row = evaluate(left)
    right_row = evaluate(right)
    for _ in range(28):
        if left_row["dual_distortion_lower_bound"] < right_row["dual_distortion_lower_bound"]:
            lower = left
            left = right
            left_row = right_row
            right = lower + golden * (upper - lower)
            right_row = evaluate(right)
        else:
            upper = right
            right = left
            right_row = left_row
            left = upper - golden * (upper - lower)
            left_row = evaluate(left)
    best = max(cache.values(), key=lambda row: row["dual_distortion_lower_bound"])
    return {
        "family": curve["name"],
        "curve_kind": curve["kind"],
        "option_count_per_matrix": curve_length(curve),
        "value_bits_per_selected_coordinate": value_bits,
        "continuous_selected_values_exact": True,
        "support_bits_exact": True,
        "headers_and_mode_labels_free": True,
        "evaluated_multipliers": len(cache),
        "best_evaluated_dual": best,
        "hard_kill": bool(best["dual_F_lower_bound"] > TARGET_F),
        "lower_bound_claim": (
            "Each evaluated Lagrangian value is a lower bound on the best curve option "
            "allocation. Missing the dual maximum can only weaken this result."
        ),
    }


def coordinate_curve(cp: Any, prefix: Any, support_host: Any) -> dict[str, Any]:
    coordinate_count = cp.arange(UNIQUE + 1, dtype=cp.int64)
    return {
        "name": "arbitrary_symmetric_coordinate_topk",
        "kind": "exact_best_k_all_coordinates",
        "capture": prefix,
        "symbols": coordinate_count,
        "removed": coordinate_count,
        "support": cp.asarray(support_host, dtype=cp.float64),
        "option_label": "k",
    }


def fixed_prefix_curve(cp: Any, group: dict[str, Any], support_host: Any) -> dict[str, Any]:
    energies = group["group_energies"]
    sizes = group["group_sizes"]
    order = cp.argsort(energies, axis=1)[:, ::-1]
    sorted_energy = cp.take_along_axis(energies, order, axis=1)
    sorted_sizes = sizes[order]
    zero_energy = cp.zeros((MATRICES, 1), dtype=cp.float64)
    zero_count = cp.zeros((MATRICES, 1), dtype=cp.int64)
    capture = cp.concatenate(
        (zero_energy, cp.cumsum(sorted_energy, axis=1, dtype=cp.float64)), axis=1
    )
    symbols = cp.concatenate(
        (zero_count, cp.cumsum(sorted_sizes, axis=1, dtype=cp.int64)), axis=1
    )
    family = str(group["family"])
    parameter = int(group.get("block_size", group.get("segment_length")))
    return {
        "name": f"{family}_{parameter}_descending_energy_prefix",
        "kind": "actual_fixed_group_energy_prefix",
        "capture": capture,
        "symbols": symbols,
        "removed": symbols,
        "support": cp.asarray(support_host, dtype=cp.float64),
        "option_label": "selected_groups",
    }


def gross_group_curve(
    cp: Any,
    coordinate_prefix: Any,
    group: dict[str, Any],
    support_host: Any,
) -> dict[str, Any]:
    import numpy as np

    sizes = cp.asnumpy(group["group_sizes"]).astype(np.int64, copy=False)
    ascending = np.sort(sizes)
    minimum = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(ascending)))
    maximum = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(ascending[::-1])))
    if int(minimum[-1]) != UNIQUE or int(maximum[-1]) != UNIQUE:
        raise AssertionError((group["family"], "gross count closure"))
    maximum_device = cp.asarray(maximum, dtype=cp.int64)
    capture = coordinate_prefix[:, maximum_device]
    family = str(group["family"])
    parameter = int(group.get("block_size", group.get("segment_length")))
    return {
        "name": f"{family}_{parameter}_gross_arbitrary_coordinate_relaxation",
        "kind": "broad_group_family_containing_relaxation",
        "capture": capture,
        "symbols": cp.asarray(minimum, dtype=cp.int64),
        "removed": maximum_device,
        "support": cp.asarray(support_host, dtype=cp.float64),
        "option_label": "selected_groups",
    }


def aggregate_group_knots(cp: Any, group: dict[str, Any], normal_energy: Any) -> dict[str, Any]:
    import numpy as np

    fractions = np.asarray((0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0))
    group_count = int(group["group_count"])
    counts = np.minimum(group_count, np.rint(fractions * group_count).astype(np.int64))
    counts[0] = 0
    counts[-1] = group_count
    descending = cp.sort(group["group_energies"], axis=1)[:, ::-1]
    prefix = cp.concatenate(
        (
            cp.zeros((MATRICES, 1), dtype=cp.float64),
            cp.cumsum(descending, axis=1, dtype=cp.float64),
        ),
        axis=1,
    )
    chosen = cp.asnumpy(prefix[:, cp.asarray(counts)])
    energy_host = cp.asnumpy(normal_energy)
    total = float(np.sum(energy_host, dtype=np.float64))
    aggregate = np.sum(chosen, axis=0, dtype=np.float64) / total
    matrix_fractions = chosen / energy_host[:, None]
    leave_expert = []
    for expert in range(EXPERTS):
        keep = np.ones(MATRICES, dtype=bool)
        keep[3 * expert : 3 * expert + 3] = False
        leave_expert.append(
            np.sum(chosen[keep], axis=0, dtype=np.float64)
            / np.sum(energy_host[keep], dtype=np.float64)
        )
    leave_expert = np.asarray(leave_expert)
    leave_mean = np.mean(leave_expert, axis=0)
    jackknife_se = np.sqrt(
        (EXPERTS - 1) / EXPERTS
        * np.sum(np.square(leave_expert - leave_mean[None, :]), axis=0)
    )
    return {
        "fractions": fractions.tolist(),
        "selected_group_counts": counts.tolist(),
        "aggregate_capture": aggregate.tolist(),
        "per_matrix_capture": matrix_fractions.tolist(),
        "delete_one_expert_jackknife_se": jackknife_se.tolist(),
    }


def control_summary(
    cp: Any,
    primitives: Any,
    coefficients: Any,
    row: Any,
    col: Any,
    source_groups: list[dict[str, Any]],
    normal_energy: Any,
) -> list[dict[str, Any]]:
    import numpy as np

    source_knots = [aggregate_group_knots(cp, group, normal_energy) for group in source_groups]
    control_values: list[list[list[float]]] = [[] for _ in source_groups]
    for control_index, seed in enumerate(CONTROL_SEEDS):
        control = primitives.gaussian_rank_heavy_tail_control(coefficients, seed, cp)
        original_prefix = primitives.coordinate_energy_prefix(coefficients, cp)
        control_prefix = primitives.coordinate_energy_prefix(control, cp)
        if not bool(cp.array_equal(original_prefix, control_prefix).item()):
            raise AssertionError((seed, "coordinate prefix identity"))
        control_groups: list[dict[str, Any]] = []
        for block_size in (8, 16, 32, 64):
            control_groups.append(
                primitives.triangular_tile_groups(control, row, col, block_size, cp)
            )
        for segment_length in (8, 16, 32, 64):
            control_groups.append(
                primitives.offset_segment_groups(control, row, col, segment_length, cp)
            )
        for family_index, group in enumerate(control_groups):
            knots = aggregate_group_knots(cp, group, normal_energy)
            control_values[family_index].append(knots["aggregate_capture"])
        del control, control_prefix, original_prefix, control_groups
        cp.get_default_memory_pool().free_all_blocks()
        print(f"[control {control_index + 1:02d}/{len(CONTROL_SEEDS)}] seed={seed}", flush=True)

    summaries: list[dict[str, Any]] = []
    for group, source, values in zip(source_groups, source_knots, control_values, strict=True):
        controls = np.asarray(values, dtype=np.float64)
        mean = np.mean(controls, axis=0)
        sample_sd = np.std(controls, axis=0, ddof=1)
        mc_se = sample_sd / math.sqrt(len(CONTROL_SEEDS))
        qwen = np.asarray(source["aggregate_capture"], dtype=np.float64)
        jack = np.asarray(source["delete_one_expert_jackknife_se"], dtype=np.float64)
        corrected_upper = qwen - mean + 3.0 * np.hypot(mc_se, jack)
        parameter = int(group.get("block_size", group.get("segment_length")))
        summaries.append(
            {
                "family": str(group["family"]),
                "parameter": parameter,
                "group_count": int(group["group_count"]),
                "selected_group_fractions": source["fractions"],
                "selected_group_counts": source["selected_group_counts"],
                "qwen_aggregate_capture": source["aggregate_capture"],
                "control_aggregate_capture_replicates": controls.tolist(),
                "control_mean_capture": mean.tolist(),
                "control_sample_sd": sample_sd.tolist(),
                "control_mc_se": mc_se.tolist(),
                "delete_one_expert_jackknife_se": jack.tolist(),
                "qwen_minus_control_mean_plus_3_combined_se": corrected_upper.tolist(),
                "decision_eligible": False,
            }
        )
    return summaries


def strip_record(record: Any, normal_sha256: str) -> dict[str, Any]:
    return {
        "matrix_ordinal": int(record.ordinal),
        "expert_ordinal": int(record.expert_ordinal),
        "role_ordinal": int(record.role_ordinal),
        "layer": int(record.layer),
        "expert": int(record.expert),
        "role": str(record.role),
        "source_energy": float(record.source_energy),
        "model_energy": float(record.model_energy),
        "normal_energy": float(record.normal_energy),
        "model_dof": int(record.model_dof),
        "normal_dof": int(record.normal_dof),
        "rank": int(record.rank),
        "window_start": int(record.window_start),
        "window_stop": int(record.window_stop),
        "common_scale": float(record.common_scale),
        "normal_sha256_f64": normal_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    repo_root = args.repo_root.resolve()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    package_dir = Path(__file__).resolve().parent
    source_package = package_dir.parent / "polar_sparse_normal_oracle_v1"
    parent_dir = package_dir.parent / "polar_normal_predictor"

    immutable_receipts = [
        require_hash(parent_dir / "polar_normal_predictor.py", EXPECTED["parent_script"], "parent_script"),
        require_hash(parent_dir / "result.json", EXPECTED["parent_result"], "parent_result"),
        require_hash(
            package_dir.parent / "composite_superoracle/result.json",
            EXPECTED["composite_result"],
            "composite_result",
        ),
        require_hash(
            source_package / "PACKAGE_MANIFEST.json",
            EXPECTED["source_manifest"],
            "source_only_manifest",
        ),
        require_hash(
            source_package / "protocol.json",
            EXPECTED["source_protocol"],
            "source_only_protocol",
        ),
        require_hash(
            source_package / "cupy_gate_proposal.py",
            EXPECTED["cupy_primitives"],
            "cupy_primitives",
        ),
    ]
    parent_result = json.loads((parent_dir / "result.json").read_text(encoding="utf-8"))
    verify_json_seal(parent_result)
    prior_records = parent_result["source_normal_records"]
    if len(prior_records) != MATRICES:
        raise AssertionError("prior record count")

    # No third-party import or inherited producer execution occurs before all
    # immutable Python source bytes above have been authenticated.
    import cupy as cp
    import numpy as np

    parent = authenticated_import(parent_dir / "polar_normal_predictor.py", "_psno_parent")
    primitives = authenticated_import(source_package / "cupy_gate_proposal.py", "_psno_primitives")
    if primitives.AUTHORITY_GRANTED is not False:
        raise AssertionError("source-only primitive authority changed")

    source_lock = source_root / "blind_protocol_v2/unblinded/source_hashes.lock.json"
    immutable_receipts.append(require_hash(source_lock, EXPECTED["source_lock"], "source_lock"))
    lock_path, lock, source_receipts, matrices = parent.load_sources(source_root)
    if lock_path.resolve() != source_lock.resolve() or len(matrices) != MATRICES:
        raise AssertionError("source scope")
    if lock.get("checkpoint") != parent_result["scope"]["checkpoint"]:
        raise AssertionError("checkpoint lineage")
    if source_receipts != parent_result["audit"]["source_receipts"]:
        raise AssertionError("exact 18-source receipt preservation")
    composite_path = package_dir.parent / "composite_superoracle/result.json"
    _, selections, base_side_bits = parent.load_base_selections(composite_path)
    if not math.isclose(base_side_bits / PANEL_VALUES, BASE_SIDE_BPW, rel_tol=0.0, abs_tol=1e-18):
        raise AssertionError("base side ledger")

    records: list[Any] = []
    normals: list[Any] = []
    stripped_records: list[dict[str, Any]] = []
    for ordinal, (matrix, metadata, selection, prior) in enumerate(
        zip(matrices, lock["matrices"], selections, prior_records, strict=True)
    ):
        record = parent.normal_record(matrix, metadata, int(selection["rank"]))
        normal_bytes = np.ascontiguousarray(record.normal, dtype="<f8").tobytes()
        normal_sha = hashlib.sha256(normal_bytes).hexdigest()
        exact_fields = (
            "matrix_ordinal",
            "layer",
            "expert",
            "role",
            "rank",
            "window_start",
            "window_stop",
            "model_dof",
            "normal_dof",
        )
        current = strip_record(record, normal_sha)
        for field in exact_fields:
            if current[field] != prior[field]:
                raise AssertionError((ordinal, field, current[field], prior[field]))
        for field in ("source_energy", "model_energy", "normal_energy", "common_scale"):
            if current[field] != prior[field]:
                raise AssertionError((ordinal, field, "binary64 derivation changed"))
        if normal_sha != prior["normal_sha256_f64"]:
            raise AssertionError((ordinal, "normal bytes changed", normal_sha, prior["normal_sha256_f64"]))
        normals.append(record.normal)
        record.error = np.empty((0, 0), dtype=np.float64)
        records.append(record)
        stripped_records.append(current)
        print(
            f"[normal {ordinal + 1:02d}/{MATRICES}] rank={record.rank} "
            f"energy={record.normal_energy:.9f} sha={normal_sha[:12]}",
            flush=True,
        )
    del matrices

    normal_batch_host = np.stack(normals, axis=0).astype(np.float64, copy=False)
    del normals
    normal_batch = cp.asarray(normal_batch_host, dtype=cp.float64)
    measurements = primitives.build_gate_measurements(normal_batch, cp)
    coefficients = measurements["coefficients"]
    row = measurements["row"]
    col = measurements["col"]
    coordinate_prefix = measurements["coordinate_prefix"]
    normal_energy = cp.sum(cp.square(normal_batch), axis=(1, 2), dtype=cp.float64)
    if float(
        cp.max(cp.abs(coordinate_prefix[:, -1] - normal_energy) / normal_energy).item()
    ) > 4e-13:
        raise AssertionError("coordinate prefix closure")
    total_source_energy = math.fsum(float(record.source_energy) for record in records)

    support_cache: dict[int, Any] = {}

    def support(n: int) -> Any:
        if n not in support_cache:
            support_cache[n] = exact_choose_support(n)
            print(f"[support] n={n} exact enumerative curve", flush=True)
        return support_cache[n]

    coordinate = coordinate_curve(cp, coordinate_prefix, support(UNIQUE))
    duals: list[dict[str, Any]] = []
    for bits in (0, 1, 16):
        result = scan_dual(cp, records, total_source_energy, coordinate, bits)
        duals.append(result)
        print(
            f"[dual coordinate b={bits}] F_lb={result['best_evaluated_dual']['dual_F_lower_bound']:.9f} "
            f"kill={result['hard_kill']}",
            flush=True,
        )

    source_groups = measurements["tiles"] + measurements["offset_segments"]
    actual_group_duals: list[dict[str, Any]] = []
    gross_group_duals: list[dict[str, Any]] = []
    for group in source_groups:
        group_support = support(int(group["group_count"]))
        actual = fixed_prefix_curve(cp, group, group_support)
        gross = gross_group_curve(cp, coordinate_prefix, group, group_support)
        actual_result = scan_dual(cp, records, total_source_energy, actual, 1)
        gross_result = scan_dual(cp, records, total_source_energy, gross, 1)
        actual_group_duals.append(actual_result)
        gross_group_duals.append(gross_result)
        print(
            f"[dual {actual['name']}] actual_F_lb="
            f"{actual_result['best_evaluated_dual']['dual_F_lower_bound']:.9f} "
            f"gross_F_lb={gross_result['best_evaluated_dual']['dual_F_lower_bound']:.9f}",
            flush=True,
        )
        del actual, gross

    controls = control_summary(
        cp, primitives, coefficients, row, col, source_groups, normal_energy
    )

    coordinate_onebit = next(row for row in duals if row["value_bits_per_selected_coordinate"] == 1)
    fixed_prefixes_killed = all(row["hard_kill"] for row in actual_group_duals)
    broad_group_relaxations_killed = all(row["hard_kill"] for row in gross_group_duals)
    if coordinate_onebit["hard_kill"] and broad_group_relaxations_killed:
        status = "KILL_ARBITRARY_COORDINATE_AND_ALL_TESTED_BROAD_GROUP_FAMILIES"
    elif coordinate_onebit["hard_kill"] and fixed_prefixes_killed:
        status = "KILL_ARBITRARY_COORDINATE_AND_FIXED_PREFIXES_GROSS_GROUP_RELAXATION_SURVIVES"
    else:
        status = "SPARSE_NORMAL_ORACLE_SURVIVES_EARLY_GATE"

    runtime = cp.cuda.runtime.runtimeGetVersion()
    driver = cp.cuda.runtime.driverGetVersion()
    device = cp.cuda.runtime.getDeviceProperties(0)
    device_name = device.get("name", b"unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", "replace")

    report = {
        "schema": "qwen_polar_sparse_normal_oracle_gate_v1",
        "scope": {
            "checkpoint": lock["checkpoint"],
            "matrices": MATRICES,
            "experts": EXPERTS,
            "matrix_shape": [ROWS, COLS],
            "normal_shape": [ROWS, ROWS],
            "unique_symmetric_coordinates": UNIQUE,
            "panel_values": PANEL_VALUES,
            "physical_rate_bpw": RATE,
            "target_F": TARGET_F,
            "base_side_bpw": BASE_SIDE_BPW,
            "fresh_validation_access": False,
            "router_access": False,
        },
        "decision": {
            "status": status,
            "coordinate_onebit_hard_kill": bool(coordinate_onebit["hard_kill"]),
            "all_fixed_group_energy_prefixes_hard_killed": fixed_prefixes_killed,
            "all_broad_group_containing_relaxations_hard_killed": broad_group_relaxations_killed,
            "finite_codec_followup_warranted": status == "SPARSE_NORMAL_ORACLE_SURVIVES_EARLY_GATE",
            "rule": (
                "A hard kill requires a valid dual lower bound above F=0.8. Broad arbitrary-subset "
                "group families may be killed only by their gross containing relaxation."
            ),
        },
        "ledger": {
            "hard_value_bits_per_selected_coordinate": 1,
            "selected_values_reconstructed_exactly_without_quantization_error": True,
            "support_bits": "exact ceil(log2 binomial(n,k)) via integer recurrence",
            "headers_bits": 0,
            "mode_labels_bits": 0,
            "read_amplification_logical": 1.0,
            "shared_cold_bytes": 0,
            "dense_normal_materialization_required": False,
        },
        "normal_records": stripped_records,
        "duals": {
            "coordinate_value_ledgers": duals,
            "fixed_group_energy_prefix_onebit": actual_group_duals,
            "broad_group_containing_relaxation_onebit": gross_group_duals,
        },
        "heavy_tail_matched_gaussian_controls": {
            "seeds": list(CONTROL_SEEDS),
            "absolute_coefficient_multiset_preserved_per_matrix": True,
            "coordinate_topk_curve_identity_verified": True,
            "decision_eligible": False,
            "group_knots": controls,
        },
        "audit": {
            "immutable_receipts": immutable_receipts,
            "source_lock_internal_sha256": lock.get("lock_sha256"),
            "source_receipts": source_receipts,
            "prior_normal_result_lock_sha256": parent_result["result_lock_sha256"],
            "run_script_path": str(Path(__file__).resolve()),
            "run_script_sha256": sha256_file(Path(__file__).resolve()),
            "output_path": str(output),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "cuda_runtime_version": int(runtime),
            "cuda_driver_version": int(driver),
            "device_name": str(device_name),
            "device_compute_capability": str(cp.cuda.Device(0).compute_capability),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "elapsed_seconds": time.time() - started,
        },
        "claim_boundary": (
            "This is a source-leaky ideal-Gaussian oracle and lower-bound kill gate. It is not an "
            "achieved finite-codec MSE, production kernel, or fresh-data validation result."
        ),
    }
    write_sealed(output, report)
    print(f"[done] status={status}", flush=True)
    print(f"[done] wrote={output}", flush=True)


if __name__ == "__main__":
    main()
