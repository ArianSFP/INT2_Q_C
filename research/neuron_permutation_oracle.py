#!/usr/bin/env python3
"""CPU-only optimistic oracle for SwiGLU-neuron canonicalization.

The experiment is intentionally more favorable than a deployable codec:

* the exact source weights select every permutation;
* the reference expert is available without quantization error;
* predictors are fitted and scored in continuous precision; and
* a reuse oracle may map many target neurons to the same reference neuron.

Failure of these source-bound oracles is an early-kill certificate for the
corresponding stored-permutation predictive-codec family.  The script never
imports CuPy or Torch and performs no encoder or GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


CHANNELS = 768
WIDTH = 2_048
ROLES = ("gate", "up", "down")
WEIGHTS_PER_EXPERT = 3 * CHANNELS * WIDTH
TARGET_GAIN_BPW = -0.5 * math.log2(0.8)
RATE_POINTS = (2.15, 2.5)
TENSOR_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != math.prod(shape):
        raise ValueError(f"BF16 geometry mismatch for {path}: {raw.size} != {math.prod(shape)}")
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def semantic_roles(rows: list[dict[str, Any]], source_root: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if [str(row["role"]) for row in rows] != list(ROLES):
        raise ValueError("each expert must be ordered gate, up, down")
    arrays: list[np.ndarray] = []
    bindings: list[dict[str, Any]] = []
    for row in rows:
        path = source_root / str(row["source_relpath"])
        digest = sha256_file(path)
        if digest != str(row["source_bf16_sha256"]):
            raise ValueError(f"source hash mismatch: {path}")
        shape = tuple(int(x) for x in row["shape"])
        matrix = load_bf16(path, shape)
        role = str(row["role"])
        if role in ("gate", "up"):
            if matrix.shape != (CHANNELS, WIDTH):
                raise ValueError(f"unexpected {role} shape {matrix.shape}")
            semantic = matrix
        else:
            if matrix.shape != (WIDTH, CHANNELS):
                raise ValueError(f"unexpected down shape {matrix.shape}")
            semantic = matrix.T
        arrays.append(np.ascontiguousarray(semantic, dtype=np.float32))
        bindings.append(
            {
                "matrix_ordinal": int(row["matrix_ordinal"]),
                "role": role,
                "shape": list(shape),
                "source_relpath": str(row["source_relpath"]),
                "source_bf16_sha256": digest,
                "source_bytes": int(path.stat().st_size),
            }
        )
    return np.stack(arrays), bindings


def signature_features(roles: np.ndarray) -> np.ndarray:
    """Permutation-equivariant neuron signatures; no other expert is used."""
    features: list[np.ndarray] = []
    eps = np.finfo(np.float64).tiny
    x64 = roles.astype(np.float64)
    rms = np.sqrt(np.mean(x64 * x64, axis=2))
    for role in range(3):
        x = x64[role]
        r = np.maximum(rms[role], eps)
        features.extend(
            [
                np.log(r),
                np.mean(x, axis=1) / r,
                np.mean(np.abs(x), axis=1) / r,
                np.mean((x / r[:, None]) ** 4, axis=1),
                np.max(np.abs(x), axis=1) / r,
            ]
        )
    for left, right in ((0, 1), (0, 2), (1, 2)):
        denom = np.maximum(rms[left] * rms[right] * WIDTH, eps)
        features.append(np.sum(x64[left] * x64[right], axis=1) / denom)
    return np.stack(features, axis=1)


def rank_mapping(source_key: np.ndarray, target_key: np.ndarray) -> np.ndarray:
    source_order = np.argsort(source_key, kind="stable")
    target_order = np.argsort(target_key, kind="stable")
    mapping = np.empty(CHANNELS, dtype=np.int32)
    mapping[target_order] = source_order
    return mapping


def lex_mapping(source_sig: np.ndarray, target_sig: np.ndarray) -> np.ndarray:
    # np.lexsort uses the final key as primary.  Reverse columns so feature 0
    # (combined scale) remains the primary deterministic key.
    source_order = np.lexsort(tuple(source_sig[:, k] for k in range(source_sig.shape[1] - 1, -1, -1)))
    target_order = np.lexsort(tuple(target_sig[:, k] for k in range(target_sig.shape[1] - 1, -1, -1)))
    mapping = np.empty(CHANNELS, dtype=np.int32)
    mapping[target_order] = source_order
    return mapping


def mapping_sha(mapping: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mapping, dtype="<u2").tobytes()).hexdigest()


def rate_gain(residual_ratio: float) -> float:
    if not (0.0 < residual_ratio <= 1.0):
        raise ValueError(f"invalid residual ratio {residual_ratio}")
    return -0.5 * math.log2(residual_ratio)


def score_mapping(
    mapping: np.ndarray,
    dots: np.ndarray,
    dots_by_role: np.ndarray,
    source_energy: np.ndarray,
    source_energy_by_role: np.ndarray,
    target_total_energy: float,
) -> dict[str, Any]:
    index = np.arange(CHANNELS)
    chosen_joint = dots[index, mapping].astype(np.float64)
    chosen_source = source_energy[mapping]
    joint_capture = float(np.sum(chosen_joint * chosen_joint / chosen_source, dtype=np.float64))
    chosen_roles = dots_by_role[:, index, mapping].astype(np.float64)
    role_capture = float(
        np.sum(chosen_roles * chosen_roles / source_energy_by_role[:, mapping], dtype=np.float64)
    )
    total_dot = float(np.sum(chosen_joint, dtype=np.float64))
    total_source = float(np.sum(chosen_source, dtype=np.float64))
    global_capture = total_dot * total_dot / total_source

    def entry(capture: float) -> dict[str, float]:
        reduction = capture / target_total_energy
        residual = 1.0 - reduction
        return {
            "captured_energy": capture,
            "energy_reduction": reduction,
            "residual_ratio": residual,
            "rate_equivalent_gain_bpw": rate_gain(residual),
        }

    return {
        "mapping_sha256_u16le": mapping_sha(mapping),
        "global_scalar": entry(global_capture),
        "per_channel_joint_scalar": entry(joint_capture),
        "per_channel_per_role_3scalar": entry(role_capture),
    }


def pair_probe(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    source_roles = source["roles"]
    target_roles = target["roles"]
    source64 = source_roles.astype(np.float64)
    target64 = target_roles.astype(np.float64)
    source_energy_by_role = np.sum(source64 * source64, axis=2, dtype=np.float64)
    source_energy = np.sum(source_energy_by_role, axis=0, dtype=np.float64)
    target_energy = float(np.sum(target64 * target64, dtype=np.float64))

    # Three CPU BLAS calls.  Summing them gives the exact concatenated-vector
    # dot product; retaining them permits the stronger three-scalar oracle.
    dots_by_role = np.stack(
        [
            np.asarray(target_roles[r] @ source_roles[r].T, dtype=np.float32)
            for r in range(3)
        ]
    )
    dots = np.sum(dots_by_role.astype(np.float64), axis=0, dtype=np.float64)
    joint_score = dots * dots / source_energy[None, :]
    role_score = np.sum(
        dots_by_role.astype(np.float64) ** 2 / source_energy_by_role[:, None, :],
        axis=0,
        dtype=np.float64,
    )

    source_sig = source["signatures"]
    target_sig = target["signatures"]
    sig_all = np.concatenate([source_sig, target_sig], axis=0)
    sig_mean = np.mean(sig_all, axis=0)
    sig_scale = np.std(sig_all, axis=0)
    sig_scale[sig_scale < 1e-12] = 1.0
    ss = (source_sig - sig_mean) / sig_scale
    ts = (target_sig - sig_mean) / sig_scale
    sig_distance = (
        np.sum(ts * ts, axis=1)[:, None]
        + np.sum(ss * ss, axis=1)[None, :]
        - 2.0 * (ts @ ss.T)
    )

    combined_source_norm = np.sqrt(source_energy)
    combined_target_norm = np.sqrt(np.sum(target64 * target64, axis=(0, 2), dtype=np.float64))
    mappings: dict[str, np.ndarray] = {
        "identity": np.arange(CHANNELS, dtype=np.int32),
        "combined_norm_rank": rank_mapping(combined_source_norm, combined_target_norm),
        "gate_norm_rank": rank_mapping(
            np.sqrt(source_energy_by_role[0]),
            np.sqrt(np.sum(target64[0] * target64[0], axis=1, dtype=np.float64)),
        ),
        "signature_lex_rank": lex_mapping(ss, ts),
    }
    rows, cols = linear_sum_assignment(sig_distance)
    signature_mapping = np.empty(CHANNELS, dtype=np.int32)
    signature_mapping[rows] = cols
    mappings["signature_hungarian"] = signature_mapping
    rows, cols = linear_sum_assignment(-joint_score)
    joint_mapping = np.empty(CHANNELS, dtype=np.int32)
    joint_mapping[rows] = cols
    mappings["exact_joint_scalar_hungarian"] = joint_mapping
    rows, cols = linear_sum_assignment(-role_score)
    role_mapping = np.empty(CHANNELS, dtype=np.int32)
    role_mapping[rows] = cols
    mappings["exact_rolewise_3scalar_hungarian"] = role_mapping

    scored = {
        name: score_mapping(
            mapping,
            dots,
            dots_by_role,
            source_energy,
            source_energy_by_role,
            target_energy,
        )
        for name, mapping in mappings.items()
    }
    joint_reuse = float(np.sum(np.max(joint_score, axis=1), dtype=np.float64))
    role_reuse = float(np.sum(np.max(role_score, axis=1), dtype=np.float64))

    def reuse_entry(capture: float) -> dict[str, float]:
        reduction = capture / target_energy
        residual = 1.0 - reduction
        return {
            "captured_energy": capture,
            "energy_reduction": reduction,
            "residual_ratio": residual,
            "rate_equivalent_gain_bpw": rate_gain(residual),
        }

    return {
        "source_expert_ordinal": int(source["ordinal"]),
        "target_expert_ordinal": int(target["ordinal"]),
        "source_layer_expert": [int(source["layer"]), int(source["expert"])],
        "target_layer_expert": [int(target["layer"]), int(target["expert"])],
        "target_energy": target_energy,
        "methods": scored,
        "impossible_source_reuse_upper_bound": {
            "note": "many target neurons may select the same source neuron; stronger than any permutation",
            "joint_scalar": reuse_entry(joint_reuse),
            "per_role_3scalar": reuse_entry(role_reuse),
        },
        # The full mapping is retained only for the strongest legal oracle.
        "strongest_legal_mapping_target_to_source": role_mapping.tolist(),
    }


def follows_root(parents: dict[int, int], root: int, nodes: range) -> bool:
    for node in nodes:
        if node == root:
            continue
        seen: set[int] = set()
        cursor = node
        while cursor != root:
            if cursor in seen or cursor not in parents:
                return False
            seen.add(cursor)
            cursor = parents[cursor]
    return True


def best_tree(pair_capture: dict[tuple[int, int], float], count: int) -> dict[str, Any]:
    best: tuple[float, int, dict[int, int]] | None = None
    nodes = range(count)
    for root in nodes:
        targets = [node for node in nodes if node != root]
        choices = [[parent for parent in nodes if parent != target] for target in targets]
        for selected in itertools.product(*choices):
            parents = dict(zip(targets, selected))
            if not follows_root(parents, root, nodes):
                continue
            captured = sum(pair_capture[(parents[target], target)] for target in targets)
            if best is None or captured > best[0]:
                best = (captured, root, parents)
    if best is None:
        raise AssertionError("no rooted prediction tree")
    captured, root, parents = best
    depth: dict[int, int] = {root: 0}
    for node in nodes:
        cursor = node
        d = 0
        while cursor != root:
            d += 1
            cursor = parents[cursor]
        depth[node] = d
    return {
        "captured_energy": captured,
        "root": root,
        "parents": {str(k): v for k, v in sorted(parents.items())},
        "depths": {str(k): v for k, v in sorted(depth.items())},
        "maximum_dependency_depth": max(depth.values()),
    }


def side_ledger(coefficient_count: int) -> dict[str, float | int]:
    enumerative_map_bits = math.ceil(math.lgamma(CHANNELS + 1) / math.log(2.0))
    raw_map_bits = CHANNELS * math.ceil(math.log2(CHANNELS))
    coefficient_bits = coefficient_count * CHANNELS * 16
    return {
        "coefficient_count_per_channel": coefficient_count,
        "fp16_coefficient_bits": coefficient_bits,
        "permutation_enumerative_lower_bound_bits_ceil_log2_factorial": enumerative_map_bits,
        "permutation_raw_u10_bits": raw_map_bits,
        "enumerative_plus_fp16_side_bits": enumerative_map_bits + coefficient_bits,
        "raw_u10_plus_fp16_side_bits": raw_map_bits + coefficient_bits,
        "enumerative_plus_fp16_side_bpw_per_target_triplet": (enumerative_map_bits + coefficient_bits)
        / WEIGHTS_PER_EXPERT,
        "raw_u10_plus_fp16_side_bpw_per_target_triplet": (raw_map_bits + coefficient_bits)
        / WEIGHTS_PER_EXPERT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()

    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    sources = plan.get("sources")
    if not isinstance(sources, list) or len(sources) != 18:
        raise ValueError("expected the pinned 18-matrix plan")
    source_root = Path(str(plan["source_root"]))

    experts: list[dict[str, Any]] = []
    all_bindings: list[dict[str, Any]] = []
    for ordinal in range(6):
        rows = sources[3 * ordinal : 3 * ordinal + 3]
        roles, bindings = semantic_roles(rows, source_root)
        match = TENSOR_RE.search(str(rows[0]["tensor"]))
        if match is None:
            raise ValueError(f"cannot parse tensor identity: {rows[0]['tensor']}")
        layer, expert = (int(match.group(1)), int(match.group(2)))
        if any(TENSOR_RE.search(str(row["tensor"])).groups() != match.groups() for row in rows):
            raise ValueError("triplet identities disagree")
        energy = float(np.sum(roles.astype(np.float64) ** 2, dtype=np.float64))
        experts.append(
            {
                "ordinal": ordinal,
                "layer": layer,
                "expert": expert,
                "roles": roles,
                "signatures": signature_features(roles),
                "energy": energy,
            }
        )
        all_bindings.extend(bindings)

    pairs: list[dict[str, Any]] = []
    for target in experts:
        for source in experts:
            if source["ordinal"] == target["ordinal"]:
                continue
            print(
                f"source {source['ordinal']} -> target {target['ordinal']}",
                file=sys.stderr,
                flush=True,
            )
            pairs.append(pair_probe(source, target))

    pair_by_key = {
        (int(row["source_expert_ordinal"]), int(row["target_expert_ordinal"])): row
        for row in pairs
    }
    strongest_path = ("exact_rolewise_3scalar_hungarian", "per_channel_per_role_3scalar")
    pair_capture = {
        key: float(row["methods"][strongest_path[0]][strongest_path[1]]["captured_energy"])
        for key, row in pair_by_key.items()
    }
    total_energy = float(sum(float(expert["energy"]) for expert in experts))
    tree = best_tree(pair_capture, len(experts))
    tree_reduction = float(tree["captured_energy"]) / total_energy
    tree_residual = 1.0 - tree_reduction
    tree.update(
        {
            "energy_reduction": tree_reduction,
            "panel_residual_ratio": tree_residual,
            "panel_rate_equivalent_gain_bpw": rate_gain(tree_residual),
            "note": "best causal tree; a depth above one requires more than two cold streams",
        }
    )

    best_star: dict[str, Any] | None = None
    for root in range(len(experts)):
        captured = sum(pair_capture[(root, target)] for target in range(len(experts)) if target != root)
        if best_star is None or captured > float(best_star["captured_energy"]):
            best_star = {
                "root": root,
                "targets": [target for target in range(len(experts)) if target != root],
                "captured_energy": captured,
            }
    assert best_star is not None
    star_reduction = float(best_star["captured_energy"]) / total_energy
    best_star.update(
        {
            "energy_reduction": star_reduction,
            "panel_residual_ratio": 1.0 - star_reduction,
            "panel_rate_equivalent_gain_bpw": rate_gain(1.0 - star_reduction),
            "note": "one independently coded root and five direct residuals; at most two cold streams",
        }
    )

    # A still stronger impossible panel oracle: every non-root target channel
    # can reuse its best source neuron.  Choose the root that loses least gain.
    reuse_stars: list[dict[str, Any]] = []
    for root in range(len(experts)):
        captured = sum(
            float(
                pair_by_key[(root, target)]["impossible_source_reuse_upper_bound"]["per_role_3scalar"][
                    "captured_energy"
                ]
            )
            for target in range(len(experts))
            if target != root
        )
        reuse_stars.append({"root": root, "captured_energy": captured})
    reuse_star = max(reuse_stars, key=lambda row: float(row["captured_energy"]))
    reuse_reduction = float(reuse_star["captured_energy"]) / total_energy
    reuse_star.update(
        {
            "energy_reduction": reuse_reduction,
            "panel_residual_ratio": 1.0 - reuse_reduction,
            "panel_rate_equivalent_gain_bpw": rate_gain(1.0 - reuse_reduction),
            "note": "impossible star: source-neuron reuse allowed; no bijection/map constraint",
        }
    )

    ledger_1 = side_ledger(1)
    ledger_3 = side_ledger(3)
    pair_read_rows: list[dict[str, Any]] = []
    for (source, target), row in sorted(pair_by_key.items()):
        gain = float(
            row["methods"][strongest_path[0]][strongest_path[1]]["rate_equivalent_gain_bpw"]
        )
        side = float(ledger_3["enumerative_plus_fp16_side_bpw_per_target_triplet"])
        pair_read_rows.append(
            {
                "source": source,
                "target": target,
                "oracle_gain_bpw": gain,
                "charged_side_bpw": side,
                "net_gain_bpw": gain - side,
                "cold_read_amplification": {
                    str(rate): 2.0 + (side - gain) / rate for rate in RATE_POINTS
                },
                "cached_reference_read_amplification": {
                    str(rate): 1.0 + (side - gain) / rate for rate in RATE_POINTS
                },
            }
        )

    max_reuse_pair = max(
        pairs,
        key=lambda row: float(
            row["impossible_source_reuse_upper_bound"]["per_role_3scalar"]["rate_equivalent_gain_bpw"]
        ),
    )
    max_reuse_gain = float(
        max_reuse_pair["impossible_source_reuse_upper_bound"]["per_role_3scalar"][
            "rate_equivalent_gain_bpw"
        ]
    )
    verdict = "KILL" if max_reuse_gain < TARGET_GAIN_BPW else "SURVIVES_OPTIMISTIC_GATE"
    script_path = Path(__file__).resolve()
    result = {
        "schema": "qwen_neuron_permutation_oracle_v1",
        "decision": {
            "verdict": verdict,
            "required_gain_bpw_for_20pct_below_gaussian": TARGET_GAIN_BPW,
            "strongest_pair_impossible_reuse_gain_bpw": max_reuse_gain,
            "strongest_pair_fraction_of_required_gain": max_reuse_gain / TARGET_GAIN_BPW,
            "reason": (
                "hard-kill if even source-reuse, exact-source, continuous-coefficient oracle is below the gate"
            ),
        },
        "protocol": {
            "panel": "pinned STRATA-v2 Qwen3-30B-A3B six expert triplets / 18 matrices",
            "backend": "NumPy/SciPy CPU only; CuPy and Torch are not imported",
            "source_bound_optimism": [
                "exact source weights choose mappings",
                "reference weights are unquantized",
                "continuous fitted coefficients have no quantization error",
                "reuse upper bound violates one-to-one permutation legality",
            ],
            "legal_neuron_geometry": (
                "the same 768-channel permutation is applied to gate rows, up rows, and down columns"
            ),
            "gain_definition": "s = -0.5*log2(residual_energy/source_energy)",
            "early_kill_gate_bpw": TARGET_GAIN_BPW,
        },
        "input": {
            "plan_path": str(args.plan.resolve()),
            "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "plan_internal_seal": plan.get("seal_sha256"),
            "source_root": str(source_root),
            "bindings": all_bindings,
            "expert_identities": [
                {
                    "ordinal": int(expert["ordinal"]),
                    "layer": int(expert["layer"]),
                    "expert": int(expert["expert"]),
                    "energy": float(expert["energy"]),
                }
                for expert in experts
            ],
            "total_source_energy": total_energy,
        },
        "side_information": {
            "one_scalar_per_channel": ledger_1,
            "three_role_scalars_per_channel": ledger_3,
            "side_cost_note": (
                "log2(768!) is a favorable information-theoretic lower bound; u10 is the simple physical map"
            ),
        },
        "directed_pair_results": pairs,
        "panel_graph_oracles": {
            "best_legal_rolewise_tree": tree,
            "best_legal_rolewise_star": best_star,
            "best_impossible_reuse_star": reuse_star,
        },
        "read_bandwidth": {
            "model": (
                "cold target read = one full-rate reference plus a shortened residual plus charged map/scalars"
            ),
            "formula": "amplification = 2 + (side_bpw - oracle_gain_bpw) / baseline_bpw",
            "strict_below_2_condition": "oracle_gain_bpw > side_bpw",
            "pair_rows_for_strongest_legal_three_scalar_mapping": pair_read_rows,
            "dependency_warning": (
                "a non-star tree can require depth+1 cold streams; only a direct-reference star is capped at two"
            ),
        },
        "execution": {
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "elapsed_seconds": time.time() - started,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
