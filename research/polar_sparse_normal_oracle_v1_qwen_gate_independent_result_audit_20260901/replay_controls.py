#!/usr/bin/env python3
"""Independent exact fixed-family replay on Qwen and eight spatial nulls.

The executed producer package is immutable.  This audit authenticates it and
the downloaded result before importing any producer code.  The 4/4/4 BLAS
thread environment is mandatory because it is part of the empirically recovered
binary64 derivation contract.
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
VALUES = ROWS * COLS
PANEL_VALUES = MATRICES * VALUES
RATE = 2.5
TARGET_F = 0.8
BASE_SIDE_BPW = 0.0011701230649594908
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}
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
    "executed_runner": "e5540d0e9beabb984af15ab569aceac8a29cc0be91286e595d51bfafa3704f08",
    "executed_result": "586d093aa6c556000a8b591b2b437a1e8e02cb0c379893bd64ec8ed406a14ff5",
    "executed_receipt": "cf652ccae60536bc281ea4a44f1c0fb86e0316b1bf347c0667fa2d3c42a0b176",
    "parent_script": "04d033319b4bbab037b48355e5f296274ae77b1c787ddcd2508e9b58948d265e",
    "parent_result": "e4fecac5f676d84739972bbf0e04467027aeae1356e62e1dc3cd2b84bff67026",
    "composite_result": "565e1eb2122f2e476c5bd81e4205eeb3e4cede6e6a51149e95944355199eb41c",
    "primitives": "80b9608aaadaddc34c98fa70485b3e1de25074bdd423fdb211fa4d9d8d22bbee",
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


def verify_seal(report: dict[str, Any], field: str) -> str:
    clean = dict(report)
    claimed = clean.pop(field, None)
    observed = hashlib.sha256(canonical(clean)).hexdigest()
    if claimed != observed:
        raise AssertionError((field, claimed, observed))
    return observed


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
        raise AssertionError((label, "regular nonsymlink required"))
    observed = sha256_file(path)
    if observed != expected:
        raise AssertionError((label, expected, observed))
    return {"label": label, "path": str(path.resolve()), "sha256": observed, "bytes": path.stat().st_size}


def authenticated_import(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError((name, "import spec"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def waterfill(components: list[tuple[float, float]], rate: float) -> dict[str, float]:
    if not rate > 0.0 or not components:
        raise AssertionError("invalid waterfill")
    logv = [math.log2(energy / dimension) for dimension, energy in components]
    lower = max(logv) - 200.0
    upper = max(logv)
    for _ in range(300):
        level = 0.5 * (lower + upper)
        used = math.fsum(
            dimension * 0.5 * max(0.0, value - level)
            for (dimension, _), value in zip(components, logv, strict=True)
        )
        if used > rate:
            lower = level
        else:
            upper = level
    log_level = 0.5 * (lower + upper)
    level = 2.0**log_level
    used = math.fsum(
        dimension * 0.5 * max(0.0, value - log_level)
        for (dimension, _), value in zip(components, logv, strict=True)
    )
    distortion = math.fsum(
        dimension * level if value > log_level else energy
        for (dimension, energy), value in zip(components, logv, strict=True)
    )
    if not math.isclose(used, rate, rel_tol=3e-13, abs_tol=3e-13):
        raise AssertionError((used, rate))
    return {"distortion": distortion, "used_rate_bpw": used}


def feasible_from_scan(
    scan: dict[str, Any],
    records: list[Any],
    kept_indices: list[int] | None = None,
) -> dict[str, Any]:
    indices = list(range(len(records))) if kept_indices is None else list(kept_indices)
    panel_values = len(indices) * VALUES
    total_energy = math.fsum(float(records[index].source_energy) for index in indices)
    options = scan["best_evaluated_dual"]["selected_options"]
    components: list[tuple[float, float]] = []
    support_bits = 0
    value_symbols = 0
    selected_groups = 0
    per_matrix: list[dict[str, int]] = []
    for index in indices:
        record = records[index]
        option = options[index]
        components.append((record.model_dof / panel_values, record.model_energy / total_energy))
        captured_fraction = float(option["captured_normal_energy_fraction"])
        residual = max(0.0, record.normal_energy * (1.0 - captured_fraction))
        removed = int(option["removed_dof"])
        if residual > 0.0:
            components.append(
                (
                    max(1, record.normal_dof - removed) / panel_values,
                    residual / total_energy,
                )
            )
        support_bits += int(option["support_bits_exact"])
        value_symbols += int(option["value_symbols"])
        selected_groups += int(option["selected_groups"])
        per_matrix.append(
            {
                "matrix_ordinal": index,
                "selected_groups": int(option["selected_groups"]),
                "value_symbols": int(option["value_symbols"]),
                "support_bits": int(option["support_bits_exact"]),
            }
        )
    support_bpw = support_bits / panel_values
    value_bpw = value_symbols / panel_values
    total_side = BASE_SIDE_BPW + support_bpw + value_bpw
    payload = RATE - total_side
    score = waterfill(components, payload)
    f_value = score["distortion"] * 2.0 ** (2.0 * RATE)
    return {
        "F": f_value,
        "s_bpw": -0.5 * math.log2(f_value),
        "distortion": score["distortion"],
        "physical_rate_bpw": RATE,
        "payload_rate_bpw": payload,
        "used_payload_rate_bpw": score["used_rate_bpw"],
        "base_side_bpw": BASE_SIDE_BPW,
        "support_bpw": support_bpw,
        "value_bpw": value_bpw,
        "total_side_bpw": total_side,
        "support_bits_total": support_bits,
        "value_symbols_total": value_symbols,
        "selected_groups_total": selected_groups,
        "matrix_count": len(indices),
        "per_matrix": per_matrix,
    }


def sample_statistics(values: list[float]) -> tuple[float, float, float]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_deviation = math.sqrt(variance)
    return mean, standard_deviation, standard_deviation / math.sqrt(len(values))


def jackknife_se(values: list[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.sqrt((len(values) - 1) / len(values) * math.fsum((value - mean) ** 2 for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    observed_env = {name: os.environ.get(name) for name in THREAD_ENV}
    if observed_env != THREAD_ENV:
        raise AssertionError(("BLAS launch contract", THREAD_ENV, observed_env))
    if "numpy" in sys.modules or "cupy" in sys.modules:
        raise AssertionError("third-party numerical runtime imported before launch-contract check")

    repo = args.repo_root.resolve()
    source_root = args.source_root.resolve()
    producer = repo / "research/polar_sparse_normal_oracle_v1_qwen_gate_20260901"
    parent_dir = repo / "research/polar_normal_predictor"
    primitives_dir = repo / "research/polar_sparse_normal_oracle_v1"
    paths = {
        "executed_runner": producer / "run_gate.py",
        "executed_result": producer / "result.json",
        "executed_receipt": producer / "verification_receipt.json",
        "parent_script": parent_dir / "polar_normal_predictor.py",
        "parent_result": parent_dir / "result.json",
        "composite_result": repo / "research/composite_superoracle/result.json",
        "primitives": primitives_dir / "cupy_gate_proposal.py",
        "source_lock": source_root / "blind_protocol_v2/unblinded/source_hashes.lock.json",
    }
    receipts = [require_hash(paths[label], expected, label) for label, expected in EXPECTED.items()]
    executed_result = json.loads(paths["executed_result"].read_text(encoding="utf-8"))
    executed_receipt = json.loads(paths["executed_receipt"].read_text(encoding="utf-8"))
    parent_result = json.loads(paths["parent_result"].read_text(encoding="utf-8"))
    verify_seal(executed_result, "result_lock_sha256")
    verify_seal(executed_receipt, "receipt_lock_sha256")
    verify_seal(parent_result, "result_lock_sha256")
    if executed_receipt["verdict"] != "PASS" or executed_receipt["result_file_sha256"] != EXPECTED["executed_result"]:
        raise AssertionError("producer receipt binding")

    import cupy as cp
    import numpy as np

    runner = authenticated_import(paths["executed_runner"], "_psno_executed_runner")
    parent = authenticated_import(paths["parent_script"], "_psno_parent_for_control_audit")
    primitives = authenticated_import(paths["primitives"], "_psno_primitives_for_control_audit")
    lock_path, lock, source_receipts, matrices = parent.load_sources(source_root)
    _, selections, base_side_bits = parent.load_base_selections(paths["composite_result"])
    if lock_path.resolve() != paths["source_lock"].resolve() or base_side_bits / PANEL_VALUES != BASE_SIDE_BPW:
        raise AssertionError("source/base binding")
    if source_receipts != executed_result["audit"]["source_receipts"]:
        raise AssertionError("source receipt preservation")

    records: list[Any] = []
    normals: list[Any] = []
    for ordinal, (matrix, metadata, selection, prior) in enumerate(
        zip(matrices, lock["matrices"], selections, parent_result["source_normal_records"], strict=True)
    ):
        record = parent.normal_record(matrix, metadata, int(selection["rank"]))
        normal_sha = hashlib.sha256(
            np.ascontiguousarray(record.normal, dtype="<f8").tobytes()
        ).hexdigest()
        if normal_sha != prior["normal_sha256_f64"]:
            raise AssertionError((ordinal, "normal SHA", normal_sha, prior["normal_sha256_f64"]))
        records.append(record)
        normals.append(record.normal)
        record.error = np.empty((0, 0), dtype=np.float64)
        print(f"[normal {ordinal + 1:02d}/{MATRICES}] {normal_sha[:12]}", flush=True)
    del matrices

    normal_batch = cp.asarray(np.stack(normals, axis=0), dtype=cp.float64)
    del normals
    measurement = primitives.build_gate_measurements(normal_batch, cp)
    coefficients = measurement["coefficients"]
    row = measurement["row"]
    col = measurement["col"]
    qwen_groups = measurement["tiles"] + measurement["offset_segments"]
    total_source_energy = math.fsum(float(record.source_energy) for record in records)
    support_cache: dict[int, Any] = {}

    def support(group_count: int) -> Any:
        if group_count not in support_cache:
            support_cache[group_count] = runner.exact_choose_support(group_count)
        return support_cache[group_count]

    qwen_rows: list[dict[str, Any]] = []
    qwen_scans: dict[str, dict[str, Any]] = {}
    qwen_delete: dict[str, list[float]] = {}
    for group in qwen_groups:
        curve = runner.fixed_prefix_curve(cp, group, support(int(group["group_count"])))
        scan = runner.scan_dual(cp, records, total_source_energy, curve, 1)
        feasible = feasible_from_scan(scan, records)
        qwen_scans[scan["family"]] = scan
        qwen_rows.append({"family": scan["family"], "dual": scan, "feasible_primal": feasible})
        deletes = []
        for omitted in range(6):
            kept = [index for index in range(MATRICES) if index // 3 != omitted]
            deletes.append(feasible_from_scan(scan, records, kept)["s_bpw"])
        qwen_delete[scan["family"]] = deletes
        print(f"[qwen {scan['family']}] F={feasible['F']:.9f}", flush=True)

    control_rows: list[dict[str, Any]] = []
    control_deletes: dict[str, list[list[float]]] = {row["family"]: [] for row in qwen_rows}
    for control_ordinal, seed in enumerate(CONTROL_SEEDS):
        control_coefficients = primitives.gaussian_rank_heavy_tail_control(coefficients, seed, cp)
        groups: list[dict[str, Any]] = []
        for block_size in (8, 16, 32, 64):
            groups.append(primitives.triangular_tile_groups(control_coefficients, row, col, block_size, cp))
        for segment_length in (8, 16, 32, 64):
            groups.append(primitives.offset_segment_groups(control_coefficients, row, col, segment_length, cp))
        family_rows = []
        for group in groups:
            curve = runner.fixed_prefix_curve(cp, group, support(int(group["group_count"])))
            scan = runner.scan_dual(cp, records, total_source_energy, curve, 1)
            feasible = feasible_from_scan(scan, records)
            deletes = []
            for omitted in range(6):
                kept = [index for index in range(MATRICES) if index // 3 != omitted]
                deletes.append(feasible_from_scan(scan, records, kept)["s_bpw"])
            control_deletes[scan["family"]].append(deletes)
            family_rows.append({"family": scan["family"], "dual": scan, "feasible_primal": feasible})
        control_rows.append({"control_ordinal": control_ordinal, "seed": seed, "families": family_rows})
        del control_coefficients, groups
        cp.get_default_memory_pool().free_all_blocks()
        print(f"[control {control_ordinal + 1:02d}/{len(CONTROL_SEEDS)}] seed={seed}", flush=True)

    summaries = []
    for qwen_row in qwen_rows:
        family = qwen_row["family"]
        qwen_feasible = qwen_row["feasible_primal"]
        control_family = [
            next(row for row in control["families"] if row["family"] == family)
            for control in control_rows
        ]
        control_f = [float(row["feasible_primal"]["F"]) for row in control_family]
        control_s = [float(row["feasible_primal"]["s_bpw"]) for row in control_family]
        mean_f, sd_f, se_f = sample_statistics(control_f)
        mean_s, sd_s, se_s = sample_statistics(control_s)
        delete_estimates = []
        for omitted in range(6):
            control_delete_mean = math.fsum(
                values[omitted] for values in control_deletes[family]
            ) / len(CONTROL_SEEDS)
            delete_estimates.append(qwen_delete[family][omitted] - control_delete_mean)
        delete_se = jackknife_se(delete_estimates)
        combined_se = math.hypot(se_s, delete_se)
        delta_s = float(qwen_feasible["s_bpw"]) - mean_s
        summaries.append(
            {
                "family": family,
                "qwen_F": float(qwen_feasible["F"]),
                "qwen_s_bpw": float(qwen_feasible["s_bpw"]),
                "control_F": control_f,
                "control_mean_F": mean_f,
                "control_sample_sd_F": sd_f,
                "control_mc_se_F": se_f,
                "control_s_bpw": control_s,
                "control_mean_s_bpw": mean_s,
                "control_sample_sd_s_bpw": sd_s,
                "control_mc_se_s_bpw": se_s,
                "qwen_minus_control_mean_s_bpw": delta_s,
                "delete_one_expert_delta_s_estimates_bpw": delete_estimates,
                "delete_one_expert_jackknife_se_bpw": delete_se,
                "combined_se_bpw": combined_se,
                "conservative_delta_s_minus_3se_bpw": delta_s - 3.0 * combined_se,
                "optimistic_delta_s_plus_3se_bpw": delta_s + 3.0 * combined_se,
                "null_fully_reproduces_qwen_at_3se": bool(delta_s - 3.0 * combined_se <= 0.0),
                "decision_eligible": False,
            }
        )

    best_qwen = min(qwen_rows, key=lambda row: row["feasible_primal"]["F"])
    report = {
        "schema": "qwen_polar_sparse_normal_fixed_group_control_replay_v1",
        "scope": {
            "checkpoint": lock["checkpoint"],
            "matrices": MATRICES,
            "physical_rate_bpw": RATE,
            "target_F": TARGET_F,
            "fresh_validation_access": False,
            "finite_codec": False,
        },
        "qwen": qwen_rows,
        "controls": control_rows,
        "calibration": summaries,
        "summary": {
            "best_qwen_family": best_qwen["family"],
            "best_qwen_F": best_qwen["feasible_primal"]["F"],
            "best_qwen_s_bpw": best_qwen["feasible_primal"]["s_bpw"],
            "families_with_qwen_F_le_0p8": [
                row["family"] for row in qwen_rows if row["feasible_primal"]["F"] <= TARGET_F
            ],
            "control_is_diagnostic_not_kill_substitute": True,
        },
        "audit": {
            "immutable_receipts": receipts,
            "source_receipts": source_receipts,
            "thread_contract": observed_env,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "device": str(cp.cuda.runtime.getDeviceProperties(0).get("name", "unknown")),
            "hostname": socket.gethostname(),
            "elapsed_seconds": time.time() - started,
        },
        "claim_boundary": (
            "Selected values are still exact continuous values at a one-bit ledger and remaining "
            "payloads use ideal Gaussian reverse waterfilling. Controls diagnose generic spatial "
            "grouping gain and do not replace the absolute F gate."
        ),
    }
    write_sealed(args.output.resolve(), report)
    print(f"[done] wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
