#!/usr/bin/env python3
"""CuPy stage-1 oracle for the model-axis coupled polar geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np


ROWS = 768
COLS = 2048
ROLES = ("gate", "up", "down")
STACK_ROWS = ROWS * len(ROLES)
VALUES_PER_EXPERT = STACK_ROWS * COLS
EXPERTS = 6
PANEL_VALUES = EXPERTS * VALUES_PER_EXPERT
SYMMETRIC_DOF = COLS * (COLS + 1) // 2
STIEFEL_DOF = VALUES_PER_EXPERT - SYMMETRIC_DOF
RATES = (2.15, 2.30, 2.50)
TARGET_F = 0.8
GLOBAL_HEADER_BITS = 4096
EXPERT_HEADER_BITS = 128
RANK_BITS = 12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bf16(path: Path, shape: tuple[int, int]) -> cp.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != shape[0] * shape[1]:
        raise RuntimeError(f"source size mismatch: {path}")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite source: {path}")
    return cp.asarray(values, dtype=cp.float32)


def load_panel(lock_path: Path) -> tuple[list[dict], list[dict]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    matrices = lock.get("matrices")
    if not isinstance(matrices, list) or len(matrices) != 18:
        raise RuntimeError("pinned source lock must contain exactly 18 matrices")
    grouped: dict[tuple[int, int], dict[str, dict]] = {}
    receipts = []
    for row in matrices:
        role = str(row["role"])
        if role not in ROLES:
            raise RuntimeError(f"unexpected role {role}")
        path = lock_path.parent / row["output_relpath"]
        expected = str(row["source_bf16_sha256"])
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"source hash mismatch: {path}")
        tensor = str(row["tensor"])
        import re
        match = re.search(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", tensor)
        if not match:
            raise RuntimeError(f"cannot parse identity: {tensor}")
        key = (int(match.group(1)), int(match.group(2)))
        grouped.setdefault(key, {})[role] = {**row, "path": path}
        receipts.append({"tensor": tensor, "role": role, "sha256": observed, "bytes": path.stat().st_size})
    if len(grouped) != EXPERTS or any(set(x) != set(ROLES) for x in grouped.values()):
        raise RuntimeError("panel is not six complete expert triplets")
    experts = []
    for key in sorted(grouped):
        experts.append({"layer": key[0], "expert": key[1], "roles": grouped[key]})
    return experts, receipts


def rank_curve(singular_desc: np.ndarray) -> list[dict]:
    singular = np.sort(np.asarray(singular_desc, dtype=np.float64))
    if singular.shape != (COLS,):
        raise RuntimeError(f"wrong singular spectrum {singular.shape}")
    prefix = np.concatenate(([0.0], np.cumsum(singular)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(singular * singular)))
    curve = []
    for rank in range(COLS - 1):
        unmodelled = COLS - rank
        sums = prefix[unmodelled:] - prefix[:-unmodelled]
        sums_sq = prefix_sq[unmodelled:] - prefix_sq[:-unmodelled]
        errors = np.maximum(0.0, sums_sq - sums * sums / unmodelled)
        start = int(np.argmin(errors))
        symmetric_rank_dof = COLS * rank - rank * (rank - 1) // 2
        model_dof = STIEFEL_DOF + 1 + symmetric_rank_dof
        if model_dof >= VALUES_PER_EXPERT:
            break
        curve.append({
            "rank": rank,
            "model_dof": int(model_dof),
            "normal_dof": int(VALUES_PER_EXPERT - model_dof),
            "window_start": start,
            "window_stop": start + unmodelled,
            "common_scale": float(sums[start] / unmodelled),
            "residual_energy": float(errors[start]),
        })
    return curve


def reverse_waterfill(records: list[dict], ranks: list[int], physical_rate: float) -> dict:
    side_bits = GLOBAL_HEADER_BITS + EXPERTS * (EXPERT_HEADER_BITS + RANK_BITS)
    payload_bits = physical_rate * PANEL_VALUES - side_bits
    total_energy = sum(float(x["energy"]) for x in records)
    energies = []
    dimensions = []
    for record, rank in zip(records, ranks):
        cell = record["curve"][rank]
        residual = min(float(cell["residual_energy"]), float(record["energy"]) * (1.0 - 1e-15))
        energies.extend([float(record["energy"]) - residual, residual])
        dimensions.extend([int(cell["model_dof"]), int(cell["normal_dof"])])
    if sum(dimensions) != PANEL_VALUES:
        raise RuntimeError("dimension closure failed")
    variances = [e / d for e, d in zip(energies, dimensions)]
    lo = min(variances) * 2.0 ** -80
    hi = max(variances)
    for _ in range(150):
        theta = math.sqrt(lo * hi)
        used = 0.5 * sum(d * max(math.log2(v / theta), 0.0) for v, d in zip(variances, dimensions))
        if used > payload_bits:
            lo = theta
        else:
            hi = theta
    theta = hi
    distortion = sum(d * min(v, theta) for v, d in zip(variances, dimensions)) / total_energy
    f_value = distortion * 2.0 ** (2.0 * physical_rate)
    payload_bits_per_expert = payload_bits / EXPERTS
    cold_bits = payload_bits_per_expert + EXPERT_HEADER_BITS + RANK_BITS + GLOBAL_HEADER_BITS
    equal_share_bits = physical_rate * VALUES_PER_EXPERT
    return {
        "physical_rate_bpw": physical_rate,
        "side_bits": side_bits,
        "payload_rate_bpw": payload_bits / PANEL_VALUES,
        "relative_mse": distortion,
        "F": f_value,
        "s_bpw": -0.5 * math.log2(f_value),
        "water_level": theta,
        "ranks": list(ranks),
        "cold_read_amplification": cold_bits / equal_share_bits,
        "passes_target": f_value <= TARGET_F and cold_bits / equal_share_bits < 2.0,
    }


def optimize(records: list[dict], rate: float) -> dict:
    common = [reverse_waterfill(records, [rank] * EXPERTS, rate) for rank in range(len(records[0]["curve"]))]
    best_common = min(common, key=lambda x: (x["F"], x["ranks"]))
    ranks = list(best_common["ranks"])
    best = reverse_waterfill(records, ranks, rate)
    for _ in range(8):
        changed = False
        for expert in range(EXPERTS):
            options = []
            for rank in range(len(records[expert]["curve"])):
                trial = list(ranks)
                trial[expert] = rank
                options.append(reverse_waterfill(records, trial, rate))
            winner = min(options, key=lambda x: (x["F"], x["ranks"]))
            if winner["ranks"][expert] != ranks[expert]:
                ranks = list(winner["ranks"])
                best = winner
                changed = True
        if not changed:
            break
    best["common_rank_result"] = best_common
    best["selected"] = [
        {
            "layer": rec["layer"], "expert": rec["expert"],
            **rec["curve"][rank], "source_energy": rec["energy"],
            "residual_energy_ratio": rec["curve"][rank]["residual_energy"] / rec["energy"],
        }
        for rec, rank in zip(records, ranks)
    ]
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    lock_path = args.source_lock.resolve()
    experts, receipts = load_panel(lock_path)
    records = []
    for expert in experts:
        x = cp.empty((STACK_ROWS, COLS), dtype=cp.float32)
        for ordinal, role in enumerate(ROLES):
            row = expert["roles"][role]
            shape = tuple(int(v) for v in row["shape"])
            w = load_bf16(row["path"], shape)
            if role == "down":
                w = w.T
            if w.shape != (ROWS, COLS):
                raise RuntimeError(f"canonical shape mismatch {w.shape}")
            x[ordinal * ROWS : (ordinal + 1) * ROWS] = w
        energy = float(cp.sum(x.astype(cp.float64) ** 2).get())
        singular = cp.linalg.svd(x, compute_uv=False)
        singular_np = cp.asnumpy(singular).astype(np.float64)
        spectral_energy = float(np.sum(singular_np * singular_np))
        relative_closure = abs(spectral_energy - energy) / energy
        if relative_closure > 2e-5:
            raise RuntimeError(f"SVD energy closure failed: {relative_closure}")
        curve = rank_curve(singular_np)
        records.append({
            "layer": expert["layer"], "expert": expert["expert"],
            "energy": energy, "spectral_energy": spectral_energy,
            "relative_svd_energy_error": relative_closure,
            "singular_min": float(np.min(singular_np)),
            "singular_max": float(np.max(singular_np)),
            "singular_mean": float(np.mean(singular_np)),
            "curve": curve,
        })
        del x, singular
        cp.get_default_memory_pool().free_all_blocks()
        print(f"spectrum layer={expert['layer']} expert={expert['expert']} closure={relative_closure:.3e}", flush=True)

    scores = [optimize(records, rate) for rate in RATES]
    best = min(scores, key=lambda x: (x["F"], x["physical_rate_bpw"]))
    decision = "SURVIVE_TO_MATCHED_CONTROL" if best["passes_target"] else "HARD_KILL_DUAL_POLAR_BEFORE_FINITE_CODEC"
    result = {
        "schema": "qwen_pinned_dual_polar_stage1_v1",
        "status": "COMPLETE",
        "decision": decision,
        "claim_boundary": "continuous-coordinate source-specific ideal-RD oracle; not a finite codec",
        "geometry": {
            "stack_shape": [STACK_ROWS, COLS],
            "values_per_expert": VALUES_PER_EXPERT,
            "stiefel_dof": STIEFEL_DOF,
            "symmetric_dof": SYMMETRIC_DOF,
            "model": "X=QH; H_hat=cI+A_k; exhaustive contiguous unmodelled spectrum window",
        },
        "source_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "source_receipts": receipts,
        "records": records,
        "scores": scores,
        "best": best,
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(), "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        },
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["result_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "best": best, "seconds": result["runtime"]["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
