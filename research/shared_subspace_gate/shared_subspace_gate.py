#!/usr/bin/env python3
"""CuPy auxiliary gate for decoder-visible cross-expert shared subspaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np


MANIFEST_SHA256 = "4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782"
FIT_EXPERTS = (0, 8, 16, 32, 40, 48, 64, 72, 80, 96, 104, 112)
VALIDATION_EXPERTS = (24, 56, 88, 120)
ROLES = ("up", "down")
ROWS, COLS = 768, 2048
N_PER_MATRIX = ROWS * COLS
RIGHT_RANKS = (8, 16, 32, 64, 96, 128, 192, 256)
LEFT_RANKS = (4, 8, 16, 32, 48, 64, 96, 128)
RATES = (2.15, 2.30, 2.50)
FULL_LAYER_EXPERTS = 128
FRAME_BITS_PER_EXPERT = 512


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bf16(path: Path, role: str) -> cp.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != N_PER_MATRIX:
        raise RuntimeError(f"wrong BF16 length: {path}: {raw.size}")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    shape = (ROWS, COLS) if role == "up" else (COLS, ROWS)
    values = values.reshape(shape)
    if role == "down":
        values = values.T
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite source: {path}")
    return cp.asarray(values, dtype=cp.float32)


def load_manifest(root: Path, manifest_path: Path) -> tuple[dict, dict[tuple[int, str], dict]]:
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("source manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lookup: dict[tuple[int, str], dict] = {}
    for row in manifest["tensors"]:
        if int(row.get("layer", 15)) != 15:
            raise RuntimeError("unexpected layer")
        key = (int(row["expert"]), str(row["role"]))
        if key in lookup:
            raise RuntimeError(f"duplicate tensor {key}")
        path = root / row["local_path"]
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or non-regular source {path}")
        if path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"source size mismatch {path}")
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"source hash mismatch {path}")
        lookup[key] = row
    expected = {(e, r) for e in FIT_EXPERTS + VALIDATION_EXPERTS for r in ROLES}
    if set(lookup) != expected:
        raise RuntimeError(f"manifest identity set mismatch: got={len(lookup)} expected={len(expected)}")
    return manifest, lookup


def covariance_bases(root: Path, lookup: dict, separate_roles: bool) -> dict[str, dict[str, cp.ndarray]]:
    keys = ROLES if separate_roles else ("joint",)
    cov = {
        key: {
            "right": cp.zeros((COLS, COLS), dtype=cp.float32),
            "left": cp.zeros((ROWS, ROWS), dtype=cp.float32),
        }
        for key in keys
    }
    counts = {key: 0 for key in keys}
    for expert in FIT_EXPERTS:
        for role in ROLES:
            key = role if separate_roles else "joint"
            row = lookup[(expert, role)]
            w = load_bf16(root / row["local_path"], role)
            cov[key]["right"] += w.T @ w
            cov[key]["left"] += w @ w.T
            counts[key] += int(w.shape[0])
            del w
    out: dict[str, dict[str, cp.ndarray]] = {}
    for key in keys:
        right_vals, right_vecs = cp.linalg.eigh(cov[key]["right"])
        left_vals, left_vecs = cp.linalg.eigh(cov[key]["left"])
        out[key] = {
            "right": cp.ascontiguousarray(right_vecs[:, ::-1]),
            "left": cp.ascontiguousarray(left_vecs[:, ::-1]),
            "right_eigenvalues": right_vals[::-1],
            "left_eigenvalues": left_vals[::-1],
        }
        del cov[key]
    cp.get_default_memory_pool().free_all_blocks()
    return out


def reverse_waterfill(energies: list[float], dimensions: list[int], bits: float) -> tuple[float, float, list[float]]:
    if bits < 0:
        return math.inf, math.nan, []
    variances = [e / d for e, d in zip(energies, dimensions) if d > 0]
    dims = [d for d in dimensions if d > 0]
    ens = [e for e, d in zip(energies, dimensions) if d > 0]
    lo = min(variances) * 2.0 ** -80
    hi = max(variances)
    for _ in range(160):
        theta = math.sqrt(lo * hi) if lo > 0 else hi * 0.5
        used = 0.5 * sum(d * max(math.log2(v / theta), 0.0) for v, d in zip(variances, dims))
        if used > bits:
            lo = theta
        else:
            hi = theta
    theta = hi
    rates = [0.5 * max(math.log2(v / theta), 0.0) for v in variances]
    distortion = sum(d * min(v, theta) for v, d in zip(variances, dims)) / sum(ens)
    return distortion, theta, rates


def candidate_ledger(
    family: str,
    separate_roles: bool,
    right_rank: int,
    left_rank: int,
    energies_by_role: dict[str, dict[str, float]],
) -> list[dict]:
    basis_copies = len(ROLES) if separate_roles else 1
    basis_values = basis_copies * (COLS * right_rank + ROWS * left_rank)
    basis_bits = basis_values * 16
    full_layer_weights = FULL_LAYER_EXPERTS * len(ROLES) * N_PER_MATRIX
    side_bits_full_layer = basis_bits + FULL_LAYER_EXPERTS * FRAME_BITS_PER_EXPERT
    side_bpw = side_bits_full_layer / full_layer_weights

    energies: list[float] = []
    dimensions: list[int] = []
    n_val_per_role = len(VALIDATION_EXPERTS) * N_PER_MATRIX
    for role in ROLES:
        e = energies_by_role[role]
        if family == "right":
            energies.extend([e["right"], e["total"] - e["right"]])
            dimensions.extend([len(VALIDATION_EXPERTS) * ROWS * right_rank,
                               len(VALIDATION_EXPERTS) * ROWS * (COLS - right_rank)])
        elif family == "left":
            energies.extend([e["left"], e["total"] - e["left"]])
            dimensions.extend([len(VALIDATION_EXPERTS) * left_rank * COLS,
                               len(VALIDATION_EXPERTS) * (ROWS - left_rank) * COLS])
        else:
            uv = e["core"]
            u_only = e["left"] - uv
            v_only = e["right"] - uv
            rest = e["total"] - e["left"] - e["right"] + uv
            tiny = 1e-7 * e["total"]
            if min(uv, u_only, v_only, rest) < -tiny:
                raise RuntimeError(f"non-orthogonal energy partition {role}: {uv,u_only,v_only,rest}")
            energies.extend([max(uv, 0.0), max(u_only, 0.0), max(v_only, 0.0), max(rest, 0.0)])
            dimensions.extend([
                len(VALIDATION_EXPERTS) * left_rank * right_rank,
                len(VALIDATION_EXPERTS) * left_rank * (COLS - right_rank),
                len(VALIDATION_EXPERTS) * (ROWS - left_rank) * right_rank,
                len(VALIDATION_EXPERTS) * (ROWS - left_rank) * (COLS - right_rank),
            ])

    total_weights = len(VALIDATION_EXPERTS) * len(ROLES) * N_PER_MATRIX
    if sum(dimensions) != total_weights:
        raise RuntimeError("dimension closure failed")
    total_energy = sum(energies)
    rows = []
    for rate in RATES:
        payload_bits = (rate - side_bpw) * total_weights
        distortion, theta, component_rates = reverse_waterfill(energies, dimensions, payload_bits)
        f_value = distortion * 2.0 ** (2.0 * rate)
        payload_bits_per_expert = (rate - side_bpw) * len(ROLES) * N_PER_MATRIX
        cold_bits = payload_bits_per_expert + basis_bits + FRAME_BITS_PER_EXPERT
        equal_share_bits = rate * len(ROLES) * N_PER_MATRIX
        read_amp = cold_bits / equal_share_bits
        rows.append({
            "physical_rate_bpw": rate,
            "basis_physical_side_bpw": side_bpw,
            "basis_bits_cold": basis_bits,
            "payload_bits_validation": payload_bits,
            "relative_mse": distortion,
            "F": f_value,
            "s_bpw": -0.5 * math.log2(f_value),
            "water_level": theta,
            "component_rates_bpw": component_rates,
            "cold_read_amplification": read_amp,
            "passes_read_lt_2": read_amp < 2.0,
            "passes_target_F_0p8": f_value <= 0.8 and read_amp < 2.0,
        })
    return rows


def measure(root: Path, lookup: dict, bases: dict, separate_roles: bool) -> list[dict]:
    cache: dict[tuple[str, int, int], dict[str, dict[str, float]]] = {}
    families = []
    for rr in RIGHT_RANKS:
        families.append(("right", rr, 0))
    for lr in LEFT_RANKS:
        families.append(("left", 0, lr))
    for rr in RIGHT_RANKS:
        for lr in LEFT_RANKS:
            families.append(("two_sided", rr, lr))

    # Load each validation matrix once and accumulate every nested-rank energy.
    accum: dict[tuple[str, int, int], dict[str, dict[str, float]]] = {
        key: {role: {"total": 0.0, "right": 0.0, "left": 0.0, "core": 0.0} for role in ROLES}
        for key in families
    }
    source_rows = []
    for expert in VALIDATION_EXPERTS:
        for role in ROLES:
            row = lookup[(expert, role)]
            w = load_bf16(root / row["local_path"], role)
            total = float(cp.sum(w.astype(cp.float64) ** 2).get())
            bkey = role if separate_roles else "joint"
            right_full = w @ bases[bkey]["right"][:, : max(RIGHT_RANKS)]
            left_full = bases[bkey]["left"][:, : max(LEFT_RANKS)].T @ w
            right_prefix = cp.cumsum(cp.sum(right_full.astype(cp.float64) ** 2, axis=0))
            left_prefix = cp.cumsum(cp.sum(left_full.astype(cp.float64) ** 2, axis=1))
            for family, rr, lr in families:
                cell = accum[(family, rr, lr)][role]
                cell["total"] += total
                if rr:
                    cell["right"] += float(right_prefix[rr - 1].get())
                if lr:
                    cell["left"] += float(left_prefix[lr - 1].get())
                if rr and lr:
                    core = left_full[:lr, :] @ bases[bkey]["right"][:, :rr]
                    cell["core"] += float(cp.sum(core.astype(cp.float64) ** 2).get())
            source_rows.append({
                "expert": expert,
                "role": role,
                "sha256": row["sha256"],
                "energy": total,
            })
            del w, right_full, left_full, right_prefix, left_prefix
            cp.get_default_memory_pool().free_all_blocks()

    results = []
    mode = "role_specific" if separate_roles else "joint_role"
    for family, rr, lr in families:
        ledgers = candidate_ledger(family, separate_roles, rr, lr, accum[(family, rr, lr)])
        results.append({
            "mode": mode,
            "family": family,
            "right_rank": rr,
            "left_rank": lr,
            "validation_component_energies": accum[(family, rr, lr)],
            "rates": ledgers,
        })
    return results, source_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    _, lookup = load_manifest(root, manifest_path)
    all_results = []
    all_source_rows = None
    spectrum = {}
    for separate in (False, True):
        mode = "role_specific" if separate else "joint_role"
        bases = covariance_bases(root, lookup, separate)
        spectrum[mode] = {
            key: {
                axis: cp.asnumpy(value[:256]).astype(float).tolist()
                for axis, value in (("right_eigenvalues", cell["right_eigenvalues"]),
                                    ("left_eigenvalues", cell["left_eigenvalues"]))
            }
            for key, cell in bases.items()
        }
        results, source_rows = measure(root, lookup, bases, separate)
        all_results.extend(results)
        all_source_rows = source_rows
        del bases
        cp.get_default_memory_pool().free_all_blocks()

    eligible = [
        (row["F"], row["cold_read_amplification"], cand, row)
        for cand in all_results for row in cand["rates"] if row["passes_read_lt_2"]
    ]
    eligible.sort(key=lambda x: (x[0], x[1]))
    best_f, _, best_cand, best_rate = eligible[0]
    decision = (
        "PROMOTE_TO_PINNED_FINITE_CODEC" if best_f <= 0.8
        else "RETAIN_AS_COMPOSITE_LEAD" if best_f <= 0.90
        else "HARD_KILL_SHARED_LINEAR_SUBSPACE"
    )
    payload = {
        "schema": "qwen_aux_shared_subspace_raw_mse_gate_v1",
        "status": "COMPLETE",
        "decision": decision,
        "claim_boundary": "auxiliary exact-basis ideal-RD gate; not a finite codec or pinned-panel result",
        "target": {"F_max": 0.8, "rates_bpw": list(RATES), "cold_read_amplification_max_exclusive": 2.0},
        "source_manifest": {"path": str(manifest_path), "sha256": MANIFEST_SHA256},
        "split": {"fit_experts": list(FIT_EXPERTS), "validation_experts": list(VALIDATION_EXPERTS), "roles": list(ROLES)},
        "basis_ledger": "exact basis arithmetic granted after FP16 literal-size charge; physical basis amortized over 128 experts, full basis counted cold",
        "best_eligible": {
            "mode": best_cand["mode"], "family": best_cand["family"],
            "right_rank": best_cand["right_rank"], "left_rank": best_cand["left_rank"],
            **best_rate,
        },
        "validation_sources": all_source_rows,
        "fit_spectra_prefix": spectrum,
        "candidates": all_results,
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(),
            "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["result_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "best": payload["best_eligible"], "seconds": payload["runtime"]["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
