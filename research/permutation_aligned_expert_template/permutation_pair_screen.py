#!/usr/bin/env python3
"""CuPy early-kill screen for permutation-aligned MoE expert templates.

The screen uses two fixed, non-pinned layer-15 experts.  It gives the candidate
an exact reference expert, an optimal 768-neuron permutation, and one free
least-squares coefficient per matched neuron.  This is strictly more favorable
than a finite shared-template codec.  A moment-matched iid Gaussian pair goes
through the identical assignment and regression as a chance control.
"""

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


SCHEMA = "qwen_permutation_aligned_pair_screen_v1"
LAYER = 15
REFERENCE_EXPERT = 0
TARGET_EXPERT = 8
ROWS = 768
COLS = 2048
PINNED_LAYERS = {5, 12, 18, 28, 36, 45}
PINNED_EXPERTS = {7, 18, 20, 41, 76, 83}
REQUIRED_INCREMENTAL_CAPTURE = 1.0 - 0.8 / 0.936397621
SEEDS = (260901401, 260901409)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bf16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != math.prod(shape):
        raise ValueError(f"shape mismatch for {path}: {raw.size}")
    return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)


def load_expert(root: Path, expert: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    up_path = root / f"l{LAYER}e{expert}_up.bf16.bin"
    down_path = root / f"l{LAYER}e{expert}_down.bf16.bin"
    if up_path.is_symlink() or down_path.is_symlink():
        raise ValueError("symlinked source is forbidden")
    up = read_bf16(up_path, (ROWS, COLS))
    down_t = read_bf16(down_path, (COLS, ROWS)).T.copy()
    return up, down_t, {up_path.name: sha256(up_path), down_path.name: sha256(down_path)}


def exact_moment_gaussian(source: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(source.shape, dtype=np.float64)
    value -= np.mean(value, dtype=np.float64)
    value *= math.sqrt(
        float(np.sum(source.astype(np.float64) ** 2))
        / float(np.sum(value * value))
    )
    return value.astype(np.float32)


def solve_pair(
    reference_up: np.ndarray,
    reference_down: np.ndarray,
    target_up: np.ndarray,
    target_down: np.ndarray,
) -> dict[str, object]:
    started = time.perf_counter()
    reference = cp.concatenate(
        (cp.asarray(reference_up), cp.asarray(reference_down)), axis=1
    ).astype(cp.float32, copy=False)
    target = cp.concatenate((cp.asarray(target_up), cp.asarray(target_down)), axis=1).astype(
        cp.float32, copy=False
    )
    ref_norm = cp.sqrt(cp.sum(reference.astype(cp.float64) ** 2, axis=1))
    target_norm = cp.sqrt(cp.sum(target.astype(cp.float64) ** 2, axis=1))
    dots = reference @ target.T
    normalized = dots.astype(cp.float64) / (ref_norm[:, None] * target_norm[None, :])
    score = cp.asnumpy(normalized * normalized)
    ref_index, target_index = linear_sum_assignment(-score)
    order = np.argsort(target_index)
    ref_for_target = ref_index[order]
    if not np.array_equal(target_index[order], np.arange(ROWS)):
        raise AssertionError("assignment is not a permutation")

    ref_selected = reference[cp.asarray(ref_for_target)]
    dot_selected = cp.sum(ref_selected.astype(cp.float64) * target.astype(cp.float64), axis=1)
    ref_energy = cp.sum(ref_selected.astype(cp.float64) ** 2, axis=1)
    alpha = dot_selected / ref_energy
    reconstruction = alpha[:, None] * ref_selected
    residual = target.astype(cp.float64) - reconstruction
    source_energy = float(cp.asnumpy(cp.sum(target.astype(cp.float64) ** 2)))
    residual_energy = float(cp.asnumpy(cp.sum(residual * residual)))

    role_reports: dict[str, dict[str, float]] = {}
    for name, start, stop in (("up", 0, COLS), ("down", COLS, 2 * COLS)):
        role_source = target[:, start:stop].astype(cp.float64)
        role_reconstruction = reconstruction[:, start:stop]
        role_sse = float(cp.asnumpy(cp.sum((role_source - role_reconstruction) ** 2)))
        role_energy = float(cp.asnumpy(cp.sum(role_source**2)))
        role_reports[name] = {
            "source_energy": role_energy,
            "residual_energy": role_sse,
            "capture": 1.0 - role_sse / role_energy,
        }
    selected_scores = score[ref_for_target, np.arange(ROWS)]
    return {
        "capture": 1.0 - residual_energy / source_energy,
        "residual_fraction": residual_energy / source_energy,
        "source_energy": source_energy,
        "residual_energy": residual_energy,
        "mean_selected_cosine_squared": float(np.mean(selected_scores)),
        "minimum_selected_cosine_squared": float(np.min(selected_scores)),
        "maximum_selected_cosine_squared": float(np.max(selected_scores)),
        "permutation_sha256": hashlib.sha256(
            np.asarray(ref_for_target, dtype="<u2").tobytes()
        ).hexdigest(),
        "alpha_fp64_sha256": hashlib.sha256(
            cp.asnumpy(alpha).astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "roles": role_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert LAYER not in PINNED_LAYERS
    assert REFERENCE_EXPERT not in PINNED_EXPERTS
    assert TARGET_EXPERT not in PINNED_EXPERTS

    reference_up, reference_down, reference_hashes = load_expert(
        args.source_dir.resolve(), REFERENCE_EXPERT
    )
    target_up, target_down, target_hashes = load_expert(
        args.source_dir.resolve(), TARGET_EXPERT
    )
    source = solve_pair(reference_up, reference_down, target_up, target_down)
    controls = []
    for seed in SEEDS:
        controls.append(
            solve_pair(
                exact_moment_gaussian(reference_up, seed),
                exact_moment_gaussian(reference_down, seed + 1),
                exact_moment_gaussian(target_up, seed + 2),
                exact_moment_gaussian(target_down, seed + 3),
            )
        )
    max_control = max(float(row["capture"]) for row in controls)
    corrected = float(source["capture"]) - max_control
    result = {
        "schema": SCHEMA,
        "claim_boundary": "Two fixed auxiliary Up/Down experts; exact free template, optimal neuron assignment, and free per-neuron coefficients. This is an upper opportunity screen, not a codec or a universal converse.",
        "backend": {
            "cupy": cp.__version__,
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        },
        "source_identity": {
            "layer": LAYER,
            "reference_expert": REFERENCE_EXPERT,
            "target_expert": TARGET_EXPERT,
            "hashes": {**reference_hashes, **target_hashes},
            "pinned_panel_opened": False,
        },
        "favorable_grants": [
            "exact uncharged reference expert",
            "optimal 768-neuron permutation",
            "one exact uncharged FP64 regression coefficient per target neuron",
            "no permutation, coefficient, template, or framing rate",
        ],
        "source": source,
        "controls": controls,
        "required_incremental_capture_over_existing_composite": REQUIRED_INCREMENTAL_CAPTURE,
        "max_control_capture": max_control,
        "control_corrected_capture": corrected,
        "decision": (
            "SURVIVE_FOR_CROSSFIT_TEMPLATE_TEST"
            if corrected >= REQUIRED_INCREMENTAL_CAPTURE
            else "EARLY_KILL_PERMUTATION_ALIGNED_SINGLE_TEMPLATE"
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
