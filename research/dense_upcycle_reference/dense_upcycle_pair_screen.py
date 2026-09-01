#!/usr/bin/env python3
"""Favorable CuPy screen for a public dense-to-MoE upcycling reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cupy as cp
import numpy as np
from scipy.optimize import linear_sum_assignment


SCHEMA = "qwen_dense_upcycle_pair_screen_v1"
MODEL = "Qwen/Qwen3-1.7B-Base"
REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
TARGET_LAYER = 15
TARGET_EXPERT = 8
REFERENCE_LAYERS = (9, 15)
ROWS = 768
REFERENCE_ROWS = 6144
COLS = 2048
PINNED_LAYERS = {5, 12, 18, 28, 36, 45}
PINNED_EXPERTS = {7, 18, 20, 41, 76, 83}
EXISTING_COMPOSITE_F = 0.936397621
REQUIRED_INCREMENTAL_CAPTURE = 1.0 - 0.8 / EXISTING_COMPOSITE_F
CONTROL_SEEDS = (260901503, 260901519)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source must be a regular non-symlink file: {path}")
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != math.prod(shape):
        raise ValueError(f"shape mismatch for {path}: {raw.size}")
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)


def load_target(root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    up_path = root / f"l{TARGET_LAYER}e{TARGET_EXPERT}_up.bf16.bin"
    down_path = root / f"l{TARGET_LAYER}e{TARGET_EXPERT}_down.bf16.bin"
    up = read_bf16(up_path, (ROWS, COLS))
    down = read_bf16(down_path, (COLS, ROWS)).T.copy()
    return up, down, {up_path.name: sha256(up_path), down_path.name: sha256(down_path)}


def load_reference(root: Path, layer: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    up_path = root / f"layer_{layer:02d}_up_proj.bf16.bin"
    down_path = root / f"layer_{layer:02d}_down_proj.bf16.bin"
    up = read_bf16(up_path, (REFERENCE_ROWS, COLS))
    down = read_bf16(down_path, (COLS, REFERENCE_ROWS)).T.copy()
    return up, down, {up_path.name: sha256(up_path), down_path.name: sha256(down_path)}


def scrambled_target(
    up: np.ndarray, down: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    result = []
    for role in (up, down):
        permutation = rng.permutation(COLS)
        signs = rng.integers(0, 2, size=COLS, dtype=np.int8) * 2 - 1
        result.append((role[:, permutation] * signs[None, :]).astype(np.float32))
    return result[0], result[1]


def centered_rows(value: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    mean = cp.mean(value.astype(cp.float64), axis=1)
    centered = value.astype(cp.float32) - mean.astype(cp.float32)[:, None]
    return centered, mean


def solve_reference(
    reference_up_np: np.ndarray,
    reference_down_np: np.ndarray,
    target_up_np: np.ndarray,
    target_down_np: np.ndarray,
) -> dict[str, object]:
    started = time.perf_counter()
    reference_roles = []
    target_roles = []
    score = None
    for reference_np, target_np in (
        (reference_up_np, target_up_np),
        (reference_down_np, target_down_np),
    ):
        reference, reference_mean = centered_rows(cp.asarray(reference_np))
        target, target_mean = centered_rows(cp.asarray(target_np))
        reference_energy = cp.sum(reference.astype(cp.float64) ** 2, axis=1)
        dots = reference @ target.T
        explained = dots.astype(cp.float64) ** 2 / reference_energy[:, None]
        score = explained if score is None else score + explained
        reference_roles.append((reference, reference_mean, reference_energy, dots))
        target_roles.append((target, target_mean))
    assert score is not None
    target_index, reference_index = linear_sum_assignment(-cp.asnumpy(score.T))
    order = np.argsort(target_index)
    reference_for_target = reference_index[order]
    if not np.array_equal(target_index[order], np.arange(ROWS)):
        raise AssertionError("assignment failed to cover every target neuron")

    total_source = 0.0
    total_residual = 0.0
    role_reports = {}
    coefficient_hash = hashlib.sha256()
    for role_name, (reference, reference_mean, reference_energy, dots), (
        target,
        target_mean,
    ) in zip(("up", "down"), reference_roles, target_roles):
        ref_index_gpu = cp.asarray(reference_for_target)
        alpha = dots[ref_index_gpu, cp.arange(ROWS)].astype(cp.float64) / reference_energy[
            ref_index_gpu
        ]
        beta = target_mean - alpha * reference_mean[ref_index_gpu]
        reconstruction = (
            alpha[:, None] * reference[ref_index_gpu].astype(cp.float64)
            + target_mean[:, None]
        )
        source = target.astype(cp.float64) + target_mean[:, None]
        residual = source - reconstruction
        source_energy = float(cp.asnumpy(cp.sum(source * source)))
        residual_energy = float(cp.asnumpy(cp.sum(residual * residual)))
        coefficient_hash.update(cp.asnumpy(alpha).astype("<f8", copy=False).tobytes())
        coefficient_hash.update(cp.asnumpy(beta).astype("<f8", copy=False).tobytes())
        role_reports[role_name] = {
            "source_energy": source_energy,
            "residual_energy": residual_energy,
            "capture": 1.0 - residual_energy / source_energy,
        }
        total_source += source_energy
        total_residual += residual_energy
    selected_score = cp.asnumpy(
        score[cp.asarray(reference_for_target), cp.arange(ROWS)]
    )
    return {
        "capture": 1.0 - total_residual / total_source,
        "residual_fraction": total_residual / total_source,
        "source_energy": total_source,
        "residual_energy": total_residual,
        "selected_explained_sse_mean": float(np.mean(selected_score)),
        "selected_explained_sse_max": float(np.max(selected_score)),
        "unique_reference_neurons": int(len(np.unique(reference_for_target))),
        "reference_index_u16_sha256": hashlib.sha256(
            np.asarray(reference_for_target, dtype="<u2").tobytes()
        ).hexdigest(),
        "coefficients_fp64_sha256": coefficient_hash.hexdigest(),
        "roles": role_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert TARGET_LAYER not in PINNED_LAYERS
    assert TARGET_EXPERT not in PINNED_EXPERTS

    manifest_path = args.reference_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned_manifest = dict(manifest)
    claimed_manifest_seal = unsigned_manifest.pop("canonical_unsigned_sha256")
    actual_manifest_seal = hashlib.sha256(
        json.dumps(
            unsigned_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()
    if claimed_manifest_seal != actual_manifest_seal:
        raise ValueError("reference manifest seal mismatch")
    if manifest["model"] != MODEL or manifest["revision"] != REVISION:
        raise ValueError("reference model identity mismatch")

    target_up, target_down, target_hashes = load_target(args.target_dir.resolve())
    source_rows = []
    control_rows = []
    reference_hashes = {}
    for layer in REFERENCE_LAYERS:
        reference_up, reference_down, hashes = load_reference(
            args.reference_dir.resolve(), layer
        )
        reference_hashes.update(hashes)
        source_rows.append(
            {
                "reference_layer": layer,
                **solve_reference(
                    reference_up, reference_down, target_up, target_down
                ),
            }
        )
        for seed in CONTROL_SEEDS:
            control_up, control_down = scrambled_target(target_up, target_down, seed)
            control_rows.append(
                {
                    "reference_layer": layer,
                    "control_seed": seed,
                    **solve_reference(
                        reference_up, reference_down, control_up, control_down
                    ),
                }
            )
    best_source = max(source_rows, key=lambda row: float(row["capture"]))
    max_control = max(float(row["capture"]) for row in control_rows)
    corrected = float(best_source["capture"]) - max_control
    result = {
        "schema": SCHEMA,
        "claim_boundary": "One fixed non-pinned Qwen3-30B-A3B auxiliary expert against two locked Qwen3-1.7B-Base dense layers. Exact dense tensors, optimal rectangular neuron assignment, layer selection, and four FP64 affine coefficients per neuron are free. This is an upper opportunity screen, not a codec or a claim of training lineage.",
        "hypothesis": "The public dense model may retain a neuron-level ancestor relation to the MoE expert, motivated by matching hidden size 2048 and dense width 6144 = 8 active experts * 768.",
        "backend": {
            "cupy": cp.__version__,
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        },
        "reference": {
            "model": MODEL,
            "revision": REVISION,
            "layers": list(REFERENCE_LAYERS),
            "manifest_sha256": sha256(manifest_path),
            "tensor_hashes": reference_hashes,
        },
        "target": {
            "layer": TARGET_LAYER,
            "expert": TARGET_EXPERT,
            "hashes": target_hashes,
            "pinned_panel_opened": False,
        },
        "favorable_grants": [
            "exact uncompressed dense reference tensors at zero rate",
            "best of two dense layers at zero selection cost",
            "optimal one-to-one assignment of 768 target neurons into 6144 reference neurons",
            "separate exact FP64 scale and offset for Up and Down per target neuron",
            "zero index, coefficient, reference-storage, or read charge",
        ],
        "source_candidates": source_rows,
        "scramble_controls": control_rows,
        "best_source": best_source,
        "max_control_capture": max_control,
        "control_corrected_capture": corrected,
        "existing_composite_F": EXISTING_COMPOSITE_F,
        "required_incremental_capture": REQUIRED_INCREMENTAL_CAPTURE,
        "decision": (
            "SURVIVE_FOR_DENSE_REFERENCE_CODEC"
            if corrected >= REQUIRED_INCREMENTAL_CAPTURE
            else "EARLY_KILL_DENSE_UPCYCLE_REFERENCE"
        ),
    }
    unsigned = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    result["canonical_unsigned_sha256"] = hashlib.sha256(unsigned).hexdigest()
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
