#!/usr/bin/env python3
"""Cross-fitted same-layer shared-expert basis oracle.

This is deliberately an optimistic source-energy screen.  It measures how
much of a held-out expert matrix lies in (a) the mean of the other sampled
experts, (b) the best rank-k expert-space PCA basis learned without that
expert, and (c) a fixed seed-generable two-dimensional cosine basis.  The
learned basis is treated as exact when computing residual energy, then its
physical storage/read cost is charged separately.  Thus a negative result is
stronger than one obtained with a quantized basis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


ROWS = 768
COLS = 2048
N = ROWS * COLS
TOTAL_EXPERTS = 128
TARGET_RATE = 2.5
REQUIRED_S = -0.5 * math.log2(0.8)
RANKS = (1, 2, 4, 8, 15)
FILE_RE = re.compile(r"l15e(?P<expert>\d+)_(?P<role>up|down)\.bf16\.bin$")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def load_bf16(path: Path, role: str) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != N:
        raise ValueError(f"{path}: {raw.size} values, expected {N}")
    values = (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if role == "down":
        values = values.reshape(COLS, ROWS).T.reshape(-1)
    return np.asarray(values, dtype=np.float64)


def metric(q: float, basis_count: int, coefficient_count: int) -> dict:
    s_oracle = -0.5 * math.log2(q)
    coefficient_bpw = (16.0 * coefficient_count + 512.0) / N
    # Optimistic: an exact learned basis is charged only as though it could be
    # represented by the target 2.5-bpw codec, while its quantization error is
    # ignored.  The lossless BF16 charge is also shown.
    basis_bpw_amortized = basis_count * TARGET_RATE / TOTAL_EXPERTS
    side_bpw_optimistic = basis_bpw_amortized + coefficient_bpw
    side_bpw_bf16 = basis_count * 16.0 / TOTAL_EXPERTS + coefficient_bpw
    s_charged_optimistic = s_oracle - side_bpw_optimistic
    s_charged_bf16 = s_oracle - side_bpw_bf16
    attributed_bpw = TARGET_RATE + basis_bpw_amortized + coefficient_bpw
    cold_read_bpw = TARGET_RATE + basis_count * TARGET_RATE + coefficient_bpw
    return {
        "residual_energy_ratio_q": q,
        "source_energy_removed_fraction": 1.0 - q,
        "s_oracle_bpw": s_oracle,
        "F_oracle": q,
        "coefficient_and_header_bpw": coefficient_bpw,
        "learned_basis_amortized_bpw_optimistic_2p5": basis_bpw_amortized,
        "side_bpw_optimistic": side_bpw_optimistic,
        "s_charged_optimistic_bpw": s_charged_optimistic,
        "F_charged_optimistic": 2.0 ** (-2.0 * s_charged_optimistic),
        "side_bpw_exact_bf16": side_bpw_bf16,
        "s_charged_exact_bf16_bpw": s_charged_bf16,
        "F_charged_exact_bf16": 2.0 ** (-2.0 * s_charged_bf16),
        "cold_read_amplification_2p5_basis": cold_read_bpw / attributed_bpw,
        "hot_cached_read_amplification": (TARGET_RATE + coefficient_bpw) / attributed_bpw,
        "passes_required_s_oracle": s_oracle >= REQUIRED_S,
        "passes_required_s_charged_optimistic": s_charged_optimistic >= REQUIRED_S,
    }


def loeo_role(paths: list[tuple[int, Path]], role: str) -> dict:
    experts = [expert for expert, _ in paths]
    x = np.stack([load_bf16(path, role) for _, path in paths])
    gram = x @ x.T
    source = np.diag(gram).copy()
    per_expert: list[dict] = []
    aggregate_residual = {"template": 0.0, **{f"pca_rank_{k}": 0.0 for k in RANKS}}
    total_source = float(source.sum())

    for heldout in range(len(experts)):
        train = np.asarray([i for i in range(len(experts)) if i != heldout], dtype=np.int64)
        gtt = gram[np.ix_(train, train)]
        gth = gram[train, heldout]
        eh = float(source[heldout])

        # Fixed coefficient-one mean template, strictly excluding the target.
        count = len(train)
        mean_norm = float(gtt.sum()) / (count * count)
        cross_mean = float(gth.sum()) / count
        template_sse = eh + mean_norm - 2.0 * cross_mean
        aggregate_residual["template"] += template_sse

        evals, evecs = np.linalg.eigh(gtt)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        evecs = evecs[:, order]
        projections = np.zeros_like(evals)
        valid = evals > max(evals[0] * 1e-13, 1e-30)
        projections[valid] = (evecs[:, valid].T @ gth) ** 2 / evals[valid]
        cumulative = np.cumsum(projections)

        row = {
            "expert": experts[heldout],
            "source_energy": eh,
            "template_residual_ratio": template_sse / eh,
            "pca_residual_ratio": {},
        }
        for k in RANKS:
            kk = min(k, len(train))
            sse = max(0.0, eh - float(cumulative[kk - 1]))
            aggregate_residual[f"pca_rank_{k}"] += sse
            row["pca_residual_ratio"][str(k)] = sse / eh
        per_expert.append(row)

    aggregate = {"template": metric(aggregate_residual["template"] / total_source, 1, 0)}
    for k in RANKS:
        aggregate[f"pca_rank_{k}"] = metric(
            aggregate_residual[f"pca_rank_{k}"] / total_source,
            k,
            k,
        )
    return {
        "experts": experts,
        "source_energy": total_source,
        "aggregate": aggregate,
        "per_expert": per_expert,
    }


def procedural_modes(max_modes: int = 64) -> np.ndarray:
    """Seed-free orthonormal separable cosine modes, ordered low-frequency first."""
    rr = np.arange(ROWS, dtype=np.float64) + 0.5
    cc = np.arange(COLS, dtype=np.float64) + 0.5
    modes: list[np.ndarray] = []
    # Frequency pairs in increasing Manhattan shell; DCT-II vectors are
    # analytic and need no payload reads.
    for shell in range(32):
        for fr in range(shell + 1):
            fc = shell - fr
            vr = np.cos(math.pi * fr * rr / ROWS)
            vc = np.cos(math.pi * fc * cc / COLS)
            vr /= np.linalg.norm(vr)
            vc /= np.linalg.norm(vc)
            modes.append(np.outer(vr, vc).reshape(-1))
            if len(modes) == max_modes:
                return np.stack(modes)
    raise AssertionError("insufficient modes")


def procedural_role(paths: list[tuple[int, Path]], role: str, modes: np.ndarray) -> dict:
    total_source = 0.0
    removed = np.zeros(modes.shape[0], dtype=np.float64)
    per_expert = []
    for expert, path in paths:
        x = load_bf16(path, role)
        energy = float(x @ x)
        coeff = modes @ x
        captured = np.cumsum(coeff * coeff)
        total_source += energy
        removed += coeff * coeff
        per_expert.append(
            {
                "expert": expert,
                "source_energy": energy,
                "captured_fraction_rank64": float(captured[-1] / energy),
            }
        )
    aggregate = {}
    for k in (1, 4, 16, 64):
        q = 1.0 - float(removed[:k].sum()) / total_source
        # Analytic modes need no basis stream; only k FP16 coefficients plus a
        # 64-byte local frame are charged.
        m = metric(q, 0, k)
        m["basis_generation"] = "analytic DCT-II; zero basis payload bytes"
        aggregate[f"dct_rank_{k}"] = m
    return {"source_energy": total_source, "aggregate": aggregate, "per_expert": per_expert}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    by_role: dict[str, list[tuple[int, Path]]] = {"up": [], "down": []}
    for path in sorted(args.source_dir.glob("*.bf16.bin")):
        match = FILE_RE.fullmatch(path.name)
        if match:
            by_role[match.group("role")].append((int(match.group("expert")), path))
    if len(by_role["up"]) != 16 or len(by_role["down"]) != 16:
        raise RuntimeError({role: len(paths) for role, paths in by_role.items()})
    if [e for e, _ in by_role["up"]] != [e for e, _ in by_role["down"]]:
        raise RuntimeError("role expert sets differ")

    hashes = {path.name: sha256_file(path) for paths in by_role.values() for _, path in paths}
    modes = procedural_modes()
    role_results = {}
    for role in ("up", "down"):
        role_results[role] = {
            "learned_shared_basis": loeo_role(by_role[role], role),
            "analytic_procedural_basis": procedural_role(by_role[role], role, modes),
        }

    # Pool exact source and residual energies across both roles.
    pooled = {}
    for family, keys in (
        ("learned_shared_basis", ["template", *(f"pca_rank_{k}" for k in RANKS)]),
        ("analytic_procedural_basis", [f"dct_rank_{k}" for k in (1, 4, 16, 64)]),
    ):
        for key in keys:
            entries = [role_results[r][family]["aggregate"][key] for r in ("up", "down")]
            energies = np.asarray(
                [role_results[r][family]["source_energy"] for r in ("up", "down")],
                dtype=np.float64,
            )
            q = float(np.dot(energies, [e["residual_energy_ratio_q"] for e in entries]) / energies.sum())
            if key == "template":
                basis_count, coeff_count = 1, 0
            elif key.startswith("pca_rank_"):
                basis_count = coeff_count = int(key.rsplit("_", 1)[1])
            else:
                basis_count = 0
                coeff_count = int(key.rsplit("_", 1)[1])
            pooled[key] = metric(q, basis_count, coeff_count)
            if basis_count == 0:
                pooled[key]["basis_generation"] = "analytic DCT-II; zero basis payload bytes"

    result = {
        "decision": "PROMOTE" if any(v["passes_required_s_charged_optimistic"] for v in pooled.values()) else "HARD_KILL_SHARED_EXPERT_BASIS",
        "hypothesis": "same-layer experts share a template or low-dimensional matrix subspace that reduces source-relative residual energy after amortized side cost",
        "protocol": {
            "layer": 15,
            "sampled_experts": [e for e, _ in by_role["up"]],
            "roles": ["up", "down_transposed"],
            "weights_per_matrix": N,
            "heldout_rule": "each learned template/PCA basis excludes the evaluated expert exactly",
            "pca_oracle": "exact FP64 Gram projection; basis quantization error ignored",
            "amortization_experts": TOTAL_EXPERTS,
            "target_rate_bpw": TARGET_RATE,
            "required_s_bpw": REQUIRED_S,
            "identities": {
                "F": "D * 2^(2R)",
                "s_from_residual_energy": "s = -0.5*log2(q)",
                "oracle_F_multiplier": "2^(-2s) = q",
                "charged_F_multiplier": "q * 2^(2*c) = 2^(-2*(s-c))",
            },
        },
        "source_sha256": hashes,
        "roles": role_results,
        "pooled": pooled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result["script_sha256"] = sha256_file(Path(__file__))
    # Rewrite with the executing script identity included.
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "pooled": pooled}, indent=2))


if __name__ == "__main__":
    main()
