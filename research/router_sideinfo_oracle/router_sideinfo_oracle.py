#!/usr/bin/env python3
"""Favourable MoE-router side-information oracle for the pinned Qwen panel.

The router is evaluated before an expert is fetched, so its 128 x 2048 weight
matrix is unusually attractive decoder-visible side information.  This probe
asks whether expert rows concentrate in a small basis derived *only* from the
same layer's router.  It grants the basis, its rate model, and mode selection
for free, then applies ideal Gaussian reverse waterfilling.  Failure is thus a
strong early kill, not an operational codec result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


ROUTER_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.gate\.weight$")
EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
RATES = (2.15, 2.30, 2.50)
TARGET_F = 0.8
KS = (1, 2, 4, 8, 16, 32, 64, 128)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16(path: Path) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def verify_plan(plan: dict[str, Any]) -> None:
    expected = plan.get("lock_sha256")
    clean = dict(plan)
    clean.pop("lock_sha256", None)
    observed = hashlib.sha256(canonical_bytes(clean)).hexdigest()
    if observed != expected:
        raise ValueError(f"plan seal mismatch: {observed} != {expected}")


def orthonormal(columns: np.ndarray, k: int) -> np.ndarray:
    q, r = np.linalg.qr(np.asarray(columns, dtype=np.float64), mode="reduced")
    keep = np.abs(np.diag(r)) > 1e-10
    q = q[:, keep]
    if q.shape[1] < k:
        raise ValueError(f"basis lost rank: {q.shape[1]} < {k}")
    return np.asarray(q[:, :k], dtype=np.float64)


def rank_dct_basis(target_router: np.ndarray, k: int) -> np.ndarray:
    n = target_router.size
    order = np.argsort(target_router, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    x = (rank.astype(np.float64) + 0.5) / n
    cols = np.cos(np.pi * x[:, None] * np.arange(k, dtype=np.float64)[None, :])
    cols[:, 0] = 1.0
    return orthonormal(cols, k)


def router_bases(
    router: np.ndarray,
    router_pca: np.ndarray,
    expert: int,
    k: int,
) -> dict[str, np.ndarray]:
    # Router-only PCA is legal shared side information: no expert weight is
    # consulted in constructing or ordering these directions.
    pca = np.asarray(router_pca[:, :k], dtype=np.float64)

    target = router[expert].astype(np.float64)
    dct = rank_dct_basis(target, k)

    # A nonlinear router-derived family.  Modulation can expose interactions
    # between the routed expert vector and the shared router row space.
    take = max(1, (k + 1) // 2)
    seed = pca[:, :take]
    scale = target / max(float(np.sqrt(np.mean(target * target))), 1e-30)
    modulated = np.concatenate((seed, seed * scale[:, None]), axis=1)
    nonlinear = orthonormal(modulated, k)
    return {
        "router_pca": pca,
        "target_rank_dct": dct,
        "target_modulated_router_pca": nonlinear,
    }


def reverse_waterfill(component_dims: np.ndarray, component_energy: np.ndarray, rate: float) -> dict[str, float]:
    dims = np.asarray(component_dims, dtype=np.float64)
    energy = np.asarray(component_energy, dtype=np.float64)
    dims /= dims.sum()
    energy /= energy.sum()
    variance = energy / dims

    lo = math.log2(float(np.min(variance))) - 2.0 * rate / float(np.min(dims)) - 8.0
    hi = math.log2(float(np.max(variance)))
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        water = 2.0**mid
        used = float(np.sum(dims * np.maximum(0.0, 0.5 * np.log2(variance / water))))
        if used > rate:
            lo = mid
        else:
            hi = mid
    water = 2.0 ** (0.5 * (lo + hi))
    distortion = float(np.sum(dims * np.minimum(variance, water)))
    return {
        "rate_bpw": rate,
        "waterlevel": water,
        "relative_mse_oracle": distortion,
        "F": distortion * (2.0 ** (2.0 * rate)),
        "s_bpw": -0.5 * math.log2(distortion * (2.0 ** (2.0 * rate))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--router-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace evidence: {output}")
    plan_path = args.plan_dir.resolve(strict=True) / "plan.lock.json"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    verify_plan(plan)
    source_root = Path(plan["source_root"]).resolve(strict=True)

    routers: dict[int, np.ndarray] = {}
    router_bindings: dict[str, Any] = {}
    for manifest_path in sorted(args.router_dir.resolve(strict=True).glob("*.manifest.json")):
        manifest = json.loads(manifest_path.read_bytes())
        match = ROUTER_RE.fullmatch(str(manifest["tensor"]))
        if match is None:
            raise ValueError(f"unexpected router tensor: {manifest['tensor']}")
        layer = int(match.group(1))
        data_path = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".bin"))
        if sha256_file(data_path) != manifest["sha256"]:
            raise ValueError(f"router hash mismatch: {data_path}")
        values = bf16(data_path)
        if values.size != 128 * 2_048 or manifest["shape"] != [128, 2_048]:
            raise ValueError(f"router geometry mismatch: {data_path}")
        routers[layer] = values.reshape(128, 2_048)
        router_bindings[str(layer)] = {
            "tensor": manifest["tensor"],
            "sha256": manifest["sha256"],
            "manifest_sha256": sha256_file(manifest_path),
            "absolute_byte_range_in_shard": manifest["absolute_byte_range_in_shard"],
            "shard": manifest["shard"],
        }

    expected_layers = sorted({int(EXPERT_RE.search(row["tensor"]).group(1)) for row in plan["sources"]})
    if sorted(routers) != expected_layers:
        raise ValueError(f"router layer coverage mismatch: {sorted(routers)} != {expected_layers}")

    source_rows: list[dict[str, Any]] = []
    matrices: list[tuple[int, int, str, np.ndarray, float]] = []
    for row in plan["sources"]:
        match = EXPERT_RE.search(str(row["tensor"]))
        if match is None:
            raise ValueError(row["tensor"])
        layer, expert = map(int, match.groups())
        path = source_root / row["source_relpath"]
        observed_hash = sha256_file(path)
        if observed_hash != row["source_bf16_sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        values = bf16(path)
        shape = tuple(map(int, row["shape"]))
        matrix = values.reshape(shape)
        if row["axis"] == "column":
            matrix = matrix.T
        elif row["axis"] != "row":
            raise ValueError(f"unknown plan axis: {row['axis']}")
        if matrix.shape != (768, 2_048):
            raise ValueError((row["tensor"], matrix.shape))
        matrix64 = matrix.astype(np.float64)
        energy = float(np.sum(matrix64 * matrix64, dtype=np.float64))
        matrices.append((layer, expert, str(row["role"]), matrix64, energy))
        source_rows.append({
            "tensor": row["tensor"],
            "sha256": observed_hash,
            "canonical_shape": [768, 2_048],
            "energy_fp64": energy,
        })

    total_energy = sum(row[4] for row in matrices)
    # The router PCA depends only on the layer, so compute it once.  Repeating
    # this SVD for every expert role and every k would add 144 identical CPU
    # decompositions without changing the oracle.
    router_pca = {
        layer: np.asarray(
            np.linalg.svd(router.astype(np.float64), full_matrices=False)[2].T,
            dtype=np.float64,
        )
        for layer, router in routers.items()
    }
    curves: list[dict[str, Any]] = []
    matrix_projection_rows: list[dict[str, Any]] = []
    for k in KS:
        mode_energy: dict[str, float] = {
            "router_pca": 0.0,
            "target_rank_dct": 0.0,
            "target_modulated_router_pca": 0.0,
        }
        adaptive_energy = 0.0
        for layer, expert, role, matrix, energy in matrices:
            bases = router_bases(routers[layer], router_pca[layer], expert, k)
            local: dict[str, float] = {}
            for mode, basis in bases.items():
                coefficients = matrix @ basis
                captured = float(np.sum(coefficients * coefficients, dtype=np.float64))
                mode_energy[mode] += captured
                local[mode] = captured / energy
            best_mode = max(local, key=local.get)
            adaptive_energy += local[best_mode] * energy
            matrix_projection_rows.append({
                "k": k,
                "layer": layer,
                "expert": expert,
                "role": role,
                "captured_energy_fraction": local,
                "free_adaptive_mode": best_mode,
            })

        all_modes = dict(mode_energy)
        all_modes["free_per_matrix_adaptive"] = adaptive_energy
        for mode, captured in all_modes.items():
            captured_fraction = captured / total_energy
            dim_fraction = k / 2_048.0
            rates = [
                reverse_waterfill(
                    np.asarray([dim_fraction, 1.0 - dim_fraction]),
                    np.asarray([captured_fraction, 1.0 - captured_fraction]),
                    rate,
                )
                for rate in RATES
            ]
            curves.append({
                "mode": mode,
                "k": k,
                "dimension_fraction": dim_fraction,
                "captured_energy_fraction": captured_fraction,
                "isotropic_expected_energy_fraction": dim_fraction,
                "energy_enrichment": captured_fraction / dim_fraction,
                "ideal_rates": rates,
                "best_F": min(item["F"] for item in rates),
                "best_s_bpw": max(item["s_bpw"] for item in rates),
            })

    best = min(curves, key=lambda row: row["best_F"])
    report = {
        "schema": "qwen_router_sideinfo_oracle_v1",
        "claim_boundary": (
            "Favourable router-derived subspace plus ideal Gaussian coding; "
            "not an operational codec, and free adaptive mode selection is source-leaky."
        ),
        "decision": "promote" if best["best_F"] <= TARGET_F else "hard_kill",
        "target": {"F_max": TARGET_F, "rates_bpw": list(RATES)},
        "plan": {
            "path": str(plan_path),
            "sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "lock_sha256": plan["lock_sha256"],
        },
        "router_bindings": router_bindings,
        "sources": source_rows,
        "free_oracle_assumptions": [
            "same-layer router is already decoder-visible and its bytes/read are free",
            "orthonormal basis arithmetic is exact",
            "per-matrix choice among three router-only basis families is free",
            "all subspace and residual coordinates use infinite-block Gaussian reverse waterfilling",
            "all water levels, component scales, framing, and finite-code loss are free",
        ],
        "best": best,
        "curves": curves,
        "matrix_projections": matrix_projection_rows,
        "runtime": {"numpy": np.__version__},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps({
        "output": str(output),
        "decision": report["decision"],
        "best": best,
    }, indent=2))


if __name__ == "__main__":
    main()
