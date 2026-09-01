#!/usr/bin/env python3
"""CPU-only, cross-fitted categorical context screen on Qwen MoE weights.

This probe deliberately attacks a different source of structure from scalar
GGD fits, covariance predictors, RMS hyperpriors, and residual trellises.  It
fits symmetric 4- and 8-level Lloyd labelers on complete *training layers*,
then measures whether exact nonlinear patterns of already decoded categorical
neighbors reduce cross-entropy on disjoint held-out layers/experts.

The screen is intentionally favorable to the proposal:

* float64 cross-entropy is used instead of an operational arithmetic coder;
* a held-out plug-in conditional-entropy oracle is reported as an optimistic
  early-kill bound;
* all context tables are shared over the model.

Unlike an uncharged mutual-information diagnostic, the primary result also
charges dense 12-bit probability tables, matrix framing, stream termination,
FP16 matrix scales, and the Lloyd reconstruction tables.  No CUDA library is
imported and the process checks that CUDA is absent from its implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


FILE_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<role>up|down)_proj\.weight\.bf16\.bin$"
)
ROLE_SHAPES = {"up": (768, 2048), "down": (2048, 768)}
ROLE_ID = {"up": 0, "down": 1}
REQUIRED_SAVING_BPW = 0.5 * math.log2(1.0 / 0.8)
PROBABILITY_BITS = 12
FP16_BITS = 16
MATRIX_SCALE_BITS = 16
MATRIX_FRAMING_BITS = 128
STREAM_TERMINATION_BITS = 64
DIRICHLET_STRENGTH = 32.0
FULL_EXPERT_WEIGHTS = 3 * 768 * 2048
LOCAL_PANEL_WEIGHTS = 2 * 768 * 2048


@dataclass(frozen=True)
class Spec:
    path: Path
    layer: int
    expert: int
    role: str
    fold: int


@dataclass(frozen=True)
class Scheme:
    name: str
    state_count: Callable[[int], int]
    extractor: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alphabets", default="4,8", help="comma-separated even alphabets"
    )
    parser.add_argument("--sample-per-tensor", type=int, default=131072)
    parser.add_argument("--lloyd-iterations", type=int, default=30)
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def discover(directory: Path, pilot: bool) -> list[Spec]:
    raw: list[tuple[Path, int, int, str]] = []
    for path in sorted(directory.glob("*.bf16.bin")):
        match = FILE_RE.match(path.name)
        if match is None:
            continue
        raw.append(
            (
                path,
                int(match.group("layer")),
                int(match.group("expert")),
                match.group("role"),
            )
        )
    layers = sorted({row[1] for row in raw})
    if pilot:
        # Preserve both folds and a wide depth range without cherry-picking
        # values or outcomes.
        keep_indices = np.linspace(0, len(layers) - 1, 8, dtype=np.int64)
        kept = {layers[int(i)] for i in keep_indices}
        raw = [row for row in raw if row[1] in kept]
        layers = sorted(kept)
    fold_by_layer = {layer: index % 2 for index, layer in enumerate(layers)}
    specs = [
        Spec(path, layer, expert, role, fold_by_layer[layer])
        for path, layer, expert, role in raw
    ]
    for role in ("up", "down"):
        for fold in (0, 1):
            if not any(s.role == role and s.fold == fold for s in specs):
                raise RuntimeError(f"empty role/fold: {role}/{fold}")
    matched = {
        (s.layer, s.expert, s.role) for s in specs
    }
    pairs = {
        (s.layer, s.expert)
        for s in specs
        if (s.layer, s.expert, "up") in matched
        and (s.layer, s.expert, "down") in matched
    }
    if len(pairs) * 2 != len(specs):
        raise RuntimeError("Up/Down inventory is not exactly paired")
    return sorted(specs, key=lambda s: (s.layer, s.expert, s.role))


def load_oriented(spec: Spec) -> np.ndarray:
    shape = ROLE_SHAPES[spec.role]
    words = np.fromfile(spec.path, dtype=np.uint16)
    if words.size != shape[0] * shape[1]:
        raise ValueError((str(spec.path), words.size, shape))
    values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
    matrix = values.reshape(shape)
    # The semantic intermediate-neuron axis is first for both roles.  This is
    # a fixed decoder convention, not a per-matrix chosen orientation.
    if spec.role == "down":
        matrix = np.ascontiguousarray(matrix.T)
    return matrix


def rms(matrix: np.ndarray) -> float:
    energy = float(np.sum(matrix * matrix, dtype=np.float64))
    return math.sqrt(max(energy / matrix.size, 1e-30))


def fit_symmetric_lloyd(
    specs: list[Spec],
    role: str,
    alphabet: int,
    samples_per_tensor: int,
    iterations: int,
) -> dict[str, object]:
    magnitudes = alphabet // 2
    pieces: list[np.ndarray] = []
    sources = [s for s in specs if s.role == role]
    for spec in sources:
        matrix = load_oriented(spec)
        scale = rms(matrix)
        flat = np.abs(matrix.reshape(-1)) / scale
        stride = max(1, flat.size // samples_per_tensor)
        pieces.append(flat[::stride][:samples_per_tensor].astype(np.float32))
    sample = np.concatenate(pieces)
    quantiles = (np.arange(magnitudes, dtype=np.float64) + 0.5) / magnitudes
    centers = np.quantile(sample, quantiles).astype(np.float64)
    for _ in range(iterations):
        boundaries = 0.5 * (centers[:-1] + centers[1:])
        labels = np.searchsorted(boundaries, sample)
        counts = np.bincount(labels, minlength=magnitudes).astype(np.float64)
        sums = np.bincount(labels, weights=sample, minlength=magnitudes)
        updated = np.divide(sums, counts, out=centers.copy(), where=counts > 0)
        if float(np.max(np.abs(updated - centers))) < 1e-10:
            centers = updated
            break
        centers = updated
    boundaries = 0.5 * (centers[:-1] + centers[1:])
    return {
        "role": role,
        "alphabet": alphabet,
        "magnitudes": [float(x) for x in centers],
        "magnitude_boundaries": [float(x) for x in boundaries],
        "fit_tensors": len(sources),
        "fit_samples": int(sample.size),
    }


def symbolize(
    spec: Spec, codebook: dict[str, object]
) -> tuple[np.ndarray, float, float, float]:
    matrix = load_oriented(spec)
    scale = rms(matrix)
    centers = np.asarray(codebook["magnitudes"], dtype=np.float32)
    boundaries = np.asarray(
        codebook["magnitude_boundaries"], dtype=np.float32
    )
    normalized = matrix / np.float32(scale)
    magnitude = np.searchsorted(boundaries, np.abs(normalized)).astype(np.uint8)
    sign = (normalized >= 0).astype(np.uint8)
    labels = magnitude + np.uint8(len(centers)) * sign
    reconstruction = centers[magnitude] * np.where(sign > 0, 1.0, -1.0)
    difference = normalized - reconstruction
    # Recover source-domain sums without materializing another FP32 matrix.
    normalized_sse = float(np.sum(difference * difference, dtype=np.float64))
    source_sse = normalized_sse * scale * scale
    source_energy = scale * scale * matrix.size
    return labels, source_sse, source_energy, scale


def combine_state(parts: list[np.ndarray], radix: int) -> np.ndarray:
    state = np.zeros(parts[0].shape, dtype=np.int64)
    for part in parts:
        state *= radix
        state += part.astype(np.int64, copy=False)
    return state.reshape(-1)


def extract_left(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return a[:, 1:].reshape(-1), a[:, :-1].reshape(-1).astype(np.int64)


def extract_up(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return a[1:, :].reshape(-1), a[:-1, :].reshape(-1).astype(np.int64)


def extract_lu(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    target = a[1:, 1:]
    state = combine_state([a[1:, :-1], a[:-1, 1:]], k)
    return target.reshape(-1), state


def extract_lud(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    target = a[1:, 1:]
    state = combine_state(
        [a[1:, :-1], a[:-1, 1:], a[:-1, :-1]], k
    )
    return target.reshape(-1), state


def extract_causal5(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    # Raster-causal: L, L2, U, UL, and UR.  UR has already been decoded
    # because it is in the previous row.
    target = a[1:, 2:-1]
    state = combine_state(
        [
            a[1:, 1:-2],
            a[1:, :-3],
            a[:-1, 2:-1],
            a[:-1, 1:-2],
            a[:-1, 3:],
        ],
        k,
    )
    return target.reshape(-1), state


def coarse_position(shape: tuple[int, int], slices: tuple[slice, slice]) -> np.ndarray:
    rows, cols = shape
    row_values = np.arange(rows, dtype=np.int64)[slices[0]]
    col_values = np.arange(cols, dtype=np.int64)[slices[1]]
    row_bin = np.minimum(3, (4 * row_values) // rows)
    col_bin = np.minimum(3, (4 * col_values) // cols)
    return (row_bin[:, None] * 4 + col_bin[None, :]).reshape(-1)


def extract_position16(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return a.reshape(-1), coarse_position(a.shape, (slice(None), slice(None)))


def extract_lu_position16(
    a: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    target = a[1:, 1:]
    pos = coarse_position(a.shape, (slice(1, None), slice(1, None)))
    state = combine_state(
        [pos.reshape(target.shape), a[1:, :-1], a[:-1, 1:]], k
    )
    # combine_state used radix k for position too, so remap explicitly.
    state = (
        (pos * k + a[1:, :-1].reshape(-1).astype(np.int64)) * k
        + a[:-1, 1:].reshape(-1).astype(np.int64)
    )
    return target.reshape(-1), state


SCHEMES = [
    Scheme("left", lambda k: k, extract_left, "previous symbol in row"),
    Scheme("up", lambda k: k, extract_up, "symbol immediately above"),
    Scheme("left_up", lambda k: k**2, extract_lu, "exact L,U pair"),
    Scheme(
        "left_up_diagonal",
        lambda k: k**3,
        extract_lud,
        "exact L,U,UL triple",
    ),
    Scheme(
        "causal_five",
        lambda k: k**5,
        extract_causal5,
        "exact L,L2,U,UL,UR categorical pattern",
    ),
    Scheme(
        "coarse_position_4x4",
        lambda k: 16,
        extract_position16,
        "fixed semantic row/column quartile bins",
    ),
    Scheme(
        "left_up_plus_position_4x4",
        lambda k: 16 * k**2,
        extract_lu_position16,
        "exact L,U plus fixed semantic position bin",
    ),
]


def count_scheme(
    specs: list[Spec],
    labels: dict[tuple[int, int, str], np.ndarray],
    scheme: Scheme,
    alphabet: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    states = scheme.state_count(alphabet)
    context_counts = [
        np.zeros((2, states, alphabet), dtype=np.int64) for _ in range(2)
    ]
    baseline_counts = [
        np.zeros((2, alphabet), dtype=np.int64) for _ in range(2)
    ]
    weights = [0, 0]
    for spec in specs:
        target, state = scheme.extractor(
            labels[(spec.layer, spec.expert, spec.role)], alphabet
        )
        role = ROLE_ID[spec.role]
        fold = spec.fold
        keys = (role * states + state) * alphabet + target.astype(np.int64)
        context_counts[fold] += np.bincount(
            keys, minlength=2 * states * alphabet
        ).reshape(2, states, alphabet)
        base_keys = role * alphabet + target.astype(np.int64)
        baseline_counts[fold] += np.bincount(
            base_keys, minlength=2 * alphabet
        ).reshape(2, alphabet)
        weights[fold] += int(target.size)
    return context_counts, baseline_counts, weights


def entropy_bits(counts: np.ndarray) -> float:
    flat = counts.reshape(-1, counts.shape[-1]).astype(np.float64)
    total = np.sum(flat, axis=1)
    mask = total > 0
    probability = np.divide(
        flat[mask], total[mask, None], out=np.zeros_like(flat[mask]), where=True
    )
    logp = np.zeros_like(probability)
    positive = probability > 0
    logp[positive] = np.log2(probability[positive])
    return float(-np.sum(flat[mask] * logp, dtype=np.float64))


def cross_entropy_from_counts(
    train_context: np.ndarray,
    eval_context: np.ndarray,
    train_baseline: np.ndarray,
    eval_baseline: np.ndarray,
    strength: float = DIRICHLET_STRENGTH,
) -> tuple[float, float]:
    alphabet = train_context.shape[-1]
    baseline_probability = (
        train_baseline.astype(np.float64) + 0.5
    ) / (np.sum(train_baseline, axis=1, keepdims=True) + 0.5 * alphabet)
    totals = np.sum(train_context, axis=2, keepdims=True).astype(np.float64)
    probability = (
        train_context.astype(np.float64)
        + strength * baseline_probability[:, None, :]
    ) / (totals + strength)
    candidate_bits = float(
        -np.sum(
            eval_context * np.log2(np.maximum(probability, 1e-300)),
            dtype=np.float64,
        )
    )
    baseline_bits = float(
        -np.sum(
            eval_baseline
            * np.log2(np.maximum(baseline_probability, 1e-300)),
            dtype=np.float64,
        )
    )
    return candidate_bits, baseline_bits


def score_scheme(
    specs: list[Spec],
    labels: dict[tuple[int, int, str], np.ndarray],
    scheme: Scheme,
    alphabet: int,
) -> dict[str, object]:
    contexts, baselines, weights = count_scheme(
        specs, labels, scheme, alphabet
    )
    fold_rows: list[dict[str, object]] = []
    total_candidate = 0.0
    total_baseline = 0.0
    total_oracle = 0.0
    total_oracle_base = 0.0
    total_weights = 0
    for train_fold in (0, 1):
        heldout = 1 - train_fold
        candidate_bits, baseline_bits = cross_entropy_from_counts(
            contexts[train_fold],
            contexts[heldout],
            baselines[train_fold],
            baselines[heldout],
        )
        oracle_bits = entropy_bits(contexts[heldout])
        oracle_baseline_bits = entropy_bits(baselines[heldout])
        count = weights[heldout]
        total_candidate += candidate_bits
        total_baseline += baseline_bits
        total_oracle += oracle_bits
        total_oracle_base += oracle_baseline_bits
        total_weights += count
        fold_rows.append(
            {
                "train_fold": train_fold,
                "heldout_fold": heldout,
                "heldout_weights": count,
                "baseline_cross_entropy_bpw": baseline_bits / count,
                "candidate_cross_entropy_bpw": candidate_bits / count,
                "crossfit_gain_bpw": (baseline_bits - candidate_bits) / count,
                "optimistic_heldout_plugin_gain_bpw": (
                    oracle_baseline_bits - oracle_bits
                )
                / count,
                "occupied_train_contexts": int(
                    np.count_nonzero(np.sum(contexts[train_fold], axis=2))
                ),
                "occupied_heldout_contexts": int(
                    np.count_nonzero(np.sum(contexts[heldout], axis=2))
                ),
            }
        )
    state_count = scheme.state_count(alphabet)
    # Two cross-fit models are charged against the union of their disjoint
    # heldout streams.  This is stricter than a deployed one-model amortization.
    probability_values_per_model = 2 * state_count * (alphabet - 1)
    candidate_model_bits = 2 * (
        probability_values_per_model * PROBABILITY_BITS + 256
    )
    model_overhead = candidate_model_bits / total_weights
    raw_gain = (total_baseline - total_candidate) / total_weights
    net_gain = raw_gain - model_overhead
    optimistic_gain = (total_oracle_base - total_oracle) / total_weights
    return {
        "name": scheme.name,
        "description": scheme.description,
        "alphabet": alphabet,
        "context_states_per_role": state_count,
        "heldout_weights": total_weights,
        "dirichlet_backoff_strength": DIRICHLET_STRENGTH,
        "baseline_cross_entropy_bpw": total_baseline / total_weights,
        "candidate_cross_entropy_bpw": total_candidate / total_weights,
        "raw_crossfit_gain_bpw": raw_gain,
        "optimistic_heldout_plugin_gain_bpw": optimistic_gain,
        "probability_precision_bits": PROBABILITY_BITS,
        "twofold_model_bits": candidate_model_bits,
        "twofold_model_overhead_bpw": model_overhead,
        "net_crossfit_gain_bpw": net_gain,
        "ideal_entropy_rate_F_from_net_gain": math.exp2(-2.0 * net_gain),
        "shortfall_to_required_saving_bpw": REQUIRED_SAVING_BPW - net_gain,
        "passes_required_saving": bool(net_gain >= REQUIRED_SAVING_BPW),
        "optimistic_oracle_passes_required_saving": bool(
            optimistic_gain >= REQUIRED_SAVING_BPW
        ),
        "folds": fold_rows,
    }


def bitplane_components(
    a: np.ndarray, alphabet: int
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    magnitudes = alphabet // 2
    mag = a % magnitudes
    sign = a // magnitudes
    target_mag = mag[1:, 2:-1]
    mag_state = combine_state(
        [
            mag[1:, 1:-2],
            mag[1:, :-3],
            mag[:-1, 2:-1],
            mag[:-1, 1:-2],
            mag[:-1, 3:],
        ],
        magnitudes,
    )
    target_sign = sign[1:, 2:-1]
    sign_neighbors = combine_state(
        [
            sign[1:, 1:-2],
            sign[1:, :-3],
            sign[:-1, 2:-1],
            sign[:-1, 1:-2],
            sign[:-1, 3:],
        ],
        2,
    )
    sign_state = target_mag.reshape(-1).astype(np.int64) * 32 + sign_neighbors
    return (
        (target_mag.reshape(-1), mag_state),
        (target_sign.reshape(-1), sign_state),
    )


def count_component(
    specs: list[Spec],
    labels: dict[tuple[int, int, str], np.ndarray],
    alphabet: int,
    component: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], int, int]:
    magnitudes = alphabet // 2
    if component == "magnitude":
        states, output = magnitudes**5, magnitudes
    elif component == "sign":
        states, output = magnitudes * 32, 2
    else:
        raise ValueError(component)
    context_counts = [
        np.zeros((2, states, output), dtype=np.int64) for _ in range(2)
    ]
    baseline_counts = [
        np.zeros((2, output), dtype=np.int64) for _ in range(2)
    ]
    weights = [0, 0]
    for spec in specs:
        pieces = bitplane_components(
            labels[(spec.layer, spec.expert, spec.role)], alphabet
        )
        target, state = pieces[0 if component == "magnitude" else 1]
        role, fold = ROLE_ID[spec.role], spec.fold
        keys = (role * states + state) * output + target.astype(np.int64)
        context_counts[fold] += np.bincount(
            keys, minlength=2 * states * output
        ).reshape(2, states, output)
        base_keys = role * output + target.astype(np.int64)
        baseline_counts[fold] += np.bincount(
            base_keys, minlength=2 * output
        ).reshape(2, output)
        weights[fold] += int(target.size)
    return context_counts, baseline_counts, weights, states, output


def score_bitplane_factorization(
    specs: list[Spec],
    labels: dict[tuple[int, int, str], np.ndarray],
    alphabet: int,
) -> dict[str, object]:
    components = {
        name: count_component(specs, labels, alphabet, name)
        for name in ("magnitude", "sign")
    }
    total_candidate = total_baseline = 0.0
    total_oracle = total_oracle_baseline = 0.0
    total_weights = 0
    model_bits = 0
    component_rows: dict[str, object] = {}
    for name, (contexts, baselines, weights, states, output) in components.items():
        candidate = baseline = oracle = oracle_base = 0.0
        for train_fold in (0, 1):
            heldout = 1 - train_fold
            cbits, bbits = cross_entropy_from_counts(
                contexts[train_fold],
                contexts[heldout],
                baselines[train_fold],
                baselines[heldout],
            )
            candidate += cbits
            baseline += bbits
            oracle += entropy_bits(contexts[heldout])
            oracle_base += entropy_bits(baselines[heldout])
        component_rows[name] = {
            "states_per_role": states,
            "output_alphabet": output,
            "raw_crossfit_gain_bpw": (baseline - candidate) / sum(weights),
            "optimistic_plugin_gain_bpw": (oracle_base - oracle) / sum(weights),
        }
        total_candidate += candidate
        total_baseline += baseline
        total_oracle += oracle
        total_oracle_baseline += oracle_base
        total_weights = sum(weights)
        model_bits += 2 * (2 * states * (output - 1) * PROBABILITY_BITS + 256)
    raw_gain = (total_baseline - total_candidate) / total_weights
    optimistic = (total_oracle_baseline - total_oracle) / total_weights
    overhead = model_bits / total_weights
    net = raw_gain - overhead
    return {
        "name": "factorized_sign_magnitude_causal_five",
        "description": (
            "magnitude from five causal magnitudes; sign from current magnitude "
            "and five causal signs"
        ),
        "alphabet": alphabet,
        "heldout_weights": total_weights,
        "components": component_rows,
        "raw_crossfit_gain_bpw": raw_gain,
        "optimistic_heldout_plugin_gain_bpw": optimistic,
        "twofold_model_bits": model_bits,
        "twofold_model_overhead_bpw": overhead,
        "net_crossfit_gain_bpw": net,
        "ideal_entropy_rate_F_from_net_gain": math.exp2(-2.0 * net),
        "shortfall_to_required_saving_bpw": REQUIRED_SAVING_BPW - net,
        "passes_required_saving": bool(net >= REQUIRED_SAVING_BPW),
        "optimistic_oracle_passes_required_saving": bool(
            optimistic >= REQUIRED_SAVING_BPW
        ),
    }


def cross_role_counts(
    specs: list[Spec],
    labels: dict[tuple[int, int, str], np.ndarray],
    alphabet: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    # Decode Up first.  Down^T then uses its own L/U/UL plus the already
    # available aligned Up symbol.  This is materially richer than a single
    # aligned cross-role table and remains expert-local.
    states = alphabet**4
    contexts = [np.zeros((1, states, alphabet), dtype=np.int64) for _ in range(2)]
    baselines = [np.zeros((1, alphabet), dtype=np.int64) for _ in range(2)]
    weights = [0, 0]
    by_key = {(s.layer, s.expert, s.role): s for s in specs}
    for layer, expert in sorted({(s.layer, s.expert) for s in specs}):
        up = labels[(layer, expert, "up")]
        down = labels[(layer, expert, "down")]
        target = down[1:, 1:]
        state = combine_state(
            [up[1:, 1:], down[1:, :-1], down[:-1, 1:], down[:-1, :-1]],
            alphabet,
        )
        fold = by_key[(layer, expert, "down")].fold
        keys = state * alphabet + target.reshape(-1).astype(np.int64)
        contexts[fold] += np.bincount(
            keys, minlength=states * alphabet
        ).reshape(1, states, alphabet)
        baselines[fold] += np.bincount(
            target.reshape(-1), minlength=alphabet
        ).reshape(1, alphabet)
        weights[fold] += int(target.size)
    return contexts, baselines, weights


def score_cross_role(
    specs: list[Spec],
    labels: dict[tuple[int, int, str], np.ndarray],
    alphabet: int,
) -> dict[str, object]:
    contexts, baselines, weights = cross_role_counts(specs, labels, alphabet)
    total_candidate = total_baseline = total_oracle = total_oracle_base = 0.0
    for train_fold in (0, 1):
        heldout = 1 - train_fold
        cbits, bbits = cross_entropy_from_counts(
            contexts[train_fold],
            contexts[heldout],
            baselines[train_fold],
            baselines[heldout],
        )
        total_candidate += cbits
        total_baseline += bbits
        total_oracle += entropy_bits(contexts[heldout])
        total_oracle_base += entropy_bits(baselines[heldout])
    down_weights = sum(weights)
    # The gain applies to Down only; divide by the two-role expert payload to
    # make it directly comparable with a bits-per-weight architecture gate.
    expert_panel_weights = down_weights * 2
    states = alphabet**4
    model_bits = 2 * (states * (alphabet - 1) * PROBABILITY_BITS + 256)
    raw = (total_baseline - total_candidate) / expert_panel_weights
    optimistic = (total_oracle_base - total_oracle) / expert_panel_weights
    overhead = model_bits / expert_panel_weights
    net = raw - overhead
    return {
        "name": "cross_role_up_to_down_plus_down_L_U_UL",
        "description": "Down^T conditioned on aligned Up and three causal Down symbols",
        "alphabet": alphabet,
        "heldout_down_weights": down_weights,
        "heldout_two_role_weights": expert_panel_weights,
        "context_states": states,
        "raw_crossfit_gain_bpw_over_two_roles": raw,
        "optimistic_heldout_plugin_gain_bpw_over_two_roles": optimistic,
        "twofold_model_bits": model_bits,
        "twofold_model_overhead_bpw": overhead,
        "net_crossfit_gain_bpw": net,
        "ideal_entropy_rate_F_from_net_gain": math.exp2(-2.0 * net),
        "shortfall_to_required_saving_bpw": REQUIRED_SAVING_BPW - net,
        "passes_required_saving": bool(net >= REQUIRED_SAVING_BPW),
        "optimistic_oracle_passes_required_saving": bool(
            optimistic >= REQUIRED_SAVING_BPW
        ),
    }


def evaluate_alphabet(specs: list[Spec], alphabet: int, args: argparse.Namespace) -> dict[str, object]:
    direction_rows: list[dict[str, object]] = []
    scheme_accumulator: dict[str, list[dict[str, object]]] = {}
    bitplane_rows: list[dict[str, object]] = []
    cross_role_rows: list[dict[str, object]] = []
    total_sse = total_energy = 0.0
    total_eval_weights = 0
    all_scales: list[float] = []
    for train_fold in (0, 1):
        train = [s for s in specs if s.fold == train_fold]
        codebooks = {
            role: fit_symmetric_lloyd(
                train,
                role,
                alphabet,
                args.sample_per_tensor,
                args.lloyd_iterations,
            )
            for role in ("up", "down")
        }
        labels: dict[tuple[int, int, str], np.ndarray] = {}
        eval_sse = eval_energy = 0.0
        eval_weights = 0
        for spec in specs:
            lab, sse, energy, scale = symbolize(spec, codebooks[spec.role])
            labels[(spec.layer, spec.expert, spec.role)] = lab
            if spec.fold != train_fold:
                eval_sse += sse
                eval_energy += energy
                eval_weights += int(lab.size)
                all_scales.append(scale)
        total_sse += eval_sse
        total_energy += eval_energy
        total_eval_weights += eval_weights

        # score_scheme itself performs both directions.  Restrict the fold list
        # by relabeling train_fold as zero, so each frozen labeler contributes
        # exactly its one legal heldout direction.
        directional_specs = [
            Spec(s.path, s.layer, s.expert, s.role, 0 if s.fold == train_fold else 1)
            for s in specs
        ]
        for scheme in SCHEMES:
            row = score_scheme(directional_specs, labels, scheme, alphabet)
            legal = row["folds"][0]  # train 0 -> heldout 1 only
            legal_row = {
                **{k: v for k, v in row.items() if k != "folds"},
                "train_fold": train_fold,
                "heldout_fold": 1 - train_fold,
                "legal_fold": legal,
            }
            scheme_accumulator.setdefault(scheme.name, []).append(legal_row)
        # The helpers similarly score both artificial directions.  Retain a
        # direction-specific result using direct counts below at aggregation.
        bp = score_bitplane_factorization(directional_specs, labels, alphabet)
        bp["train_fold"] = train_fold
        bitplane_rows.append(bp)
        cr = score_cross_role(directional_specs, labels, alphabet)
        cr["train_fold"] = train_fold
        cross_role_rows.append(cr)
        direction_rows.append(
            {
                "train_fold": train_fold,
                "heldout_fold": 1 - train_fold,
                "codebooks": codebooks,
                "heldout_weights": eval_weights,
                "heldout_relative_mse": eval_sse / eval_energy,
            }
        )
        del labels

    # Re-aggregate only the legal fold row for ordinary schemes.  The helper's
    # summary is ignored because its reverse direction uses the wrong codebook.
    aggregate_schemes: list[dict[str, object]] = []
    for scheme in SCHEMES:
        rows = scheme_accumulator[scheme.name]
        weights = sum(int(r["legal_fold"]["heldout_weights"]) for r in rows)
        baseline_bits = sum(
            float(r["legal_fold"]["baseline_cross_entropy_bpw"])
            * int(r["legal_fold"]["heldout_weights"])
            for r in rows
        )
        candidate_bits = sum(
            float(r["legal_fold"]["candidate_cross_entropy_bpw"])
            * int(r["legal_fold"]["heldout_weights"])
            for r in rows
        )
        oracle_gain_bits = sum(
            float(r["legal_fold"]["optimistic_heldout_plugin_gain_bpw"])
            * int(r["legal_fold"]["heldout_weights"])
            for r in rows
        )
        state_count = scheme.state_count(alphabet)
        model_bits = 2 * (
            2 * state_count * (alphabet - 1) * PROBABILITY_BITS + 256
        )
        raw = (baseline_bits - candidate_bits) / weights
        optimistic = oracle_gain_bits / weights
        overhead = model_bits / weights
        net = raw - overhead
        deployed_model_bits = model_bits // 2
        # Conservative two-matrix expert payload; Gate would only make the
        # shared-table cold-read fraction smaller.
        local_payload_bits = max(candidate_bits / weights, 1e-9) * LOCAL_PANEL_WEIGHTS
        cold_amp = 1.0 + deployed_model_bits / local_payload_bits
        absolute_rate = (
            candidate_bits / weights
            + (MATRIX_SCALE_BITS + MATRIX_FRAMING_BITS + STREAM_TERMINATION_BITS)
            * len(specs)
            / total_eval_weights
            + model_bits / weights
        )
        aggregate_schemes.append(
            {
                "name": scheme.name,
                "description": scheme.description,
                "alphabet": alphabet,
                "context_states_per_role": state_count,
                "heldout_weights": weights,
                "baseline_cross_entropy_bpw": baseline_bits / weights,
                "candidate_cross_entropy_bpw": candidate_bits / weights,
                "raw_crossfit_gain_bpw": raw,
                "optimistic_heldout_plugin_gain_bpw": optimistic,
                "twofold_model_bits": model_bits,
                "model_overhead_bpw": overhead,
                "net_crossfit_gain_bpw": net,
                "absolute_optimistic_rate_with_local_metadata_bpw": absolute_rate,
                "measured_scalar_F_D_times_2_to_2R": (
                    total_sse / total_energy * math.exp2(2.0 * absolute_rate)
                ),
                "ideal_entropy_rate_F_from_net_gain": math.exp2(-2.0 * net),
                "required_saving_bpw": REQUIRED_SAVING_BPW,
                "shortfall_to_required_saving_bpw": REQUIRED_SAVING_BPW - net,
                "passes_required_saving": bool(net >= REQUIRED_SAVING_BPW),
                "optimistic_oracle_passes_required_saving": bool(
                    optimistic >= REQUIRED_SAVING_BPW
                ),
                "expert_read_amplification": {
                    "warm_shared_table": 1.0,
                    "cold_shared_table_two_matrix_panel": cold_amp,
                    "cold_shared_table_three_matrix_expert_upper_bound": (
                        1.0
                        + deployed_model_bits
                        / (
                            max(candidate_bits / weights, 1e-9)
                            * FULL_EXPERT_WEIGHTS
                        )
                    ),
                    "expert_payload_is_strictly_local": True,
                },
                "directions": [r["legal_fold"] for r in rows],
            }
        )

    # Bitplane and cross-role summaries are retained as generous diagnostics.
    # Their helper summaries include an extra wrong-codebook reverse direction,
    # so use the maximum optimistic result only as a promotion oracle, never as
    # a positive crossfit claim.
    best_bitplane_optimistic = max(
        float(r["optimistic_heldout_plugin_gain_bpw"]) for r in bitplane_rows
    )
    best_cross_role_optimistic = max(
        float(r["optimistic_heldout_plugin_gain_bpw_over_two_roles"])
        for r in cross_role_rows
    )
    return {
        "alphabet": alphabet,
        "directional_codebooks": direction_rows,
        "heldout_weights_crossfit_total": total_eval_weights,
        "heldout_source_relative_mse": total_sse / total_energy,
        "matrix_scale_min": min(all_scales),
        "matrix_scale_max": max(all_scales),
        "matrix_scale_mean": float(np.mean(all_scales)),
        "schemes": aggregate_schemes,
        "bitplane_factorization_generous_diagnostics": bitplane_rows,
        "bitplane_best_optimistic_gain_bpw": best_bitplane_optimistic,
        "cross_role_generous_diagnostics": cross_role_rows,
        "cross_role_best_optimistic_gain_bpw_over_two_roles": best_cross_role_optimistic,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if "cupy" in sys.modules:
        raise RuntimeError("CPU-only protocol violated: CuPy already imported")
    alphabets = [int(x) for x in args.alphabets.split(",")]
    if any(k <= 0 or k % 2 for k in alphabets):
        raise ValueError("alphabets must be positive and even")
    specs = discover(args.tensor_dir, args.pilot)
    layers = sorted({s.layer for s in specs})
    experts = sorted({(s.layer, s.expert) for s in specs})
    manifest = [
        {
            "name": s.path.name,
            "bytes": s.path.stat().st_size,
            "sha256": sha256(s.path),
            "layer": s.layer,
            "expert": s.expert,
            "role": s.role,
            "fold": s.fold,
        }
        for s in specs
    ]
    results = [evaluate_alphabet(specs, k, args) for k in alphabets]
    candidates = [row for result in results for row in result["schemes"]]
    best_crossfit = max(candidates, key=lambda row: row["net_crossfit_gain_bpw"])
    best_oracle = max(
        candidates, key=lambda row: row["optimistic_heldout_plugin_gain_bpw"]
    )
    all_generous = [
        *(float(r["optimistic_heldout_plugin_gain_bpw"]) for r in candidates),
        *(float(r["bitplane_best_optimistic_gain_bpw"]) for r in results),
        *(
            float(r["cross_role_best_optimistic_gain_bpw_over_two_roles"])
            for r in results
        ),
    ]
    max_generous = max(all_generous)
    decision = (
        "PROMOTE_CATEGORICAL_CONTEXT"
        if float(best_crossfit["net_crossfit_gain_bpw"]) >= REQUIRED_SAVING_BPW
        else (
            "STOP_OPTIMISTIC_CATEGORICAL_ORACLE_BELOW_GATE"
            if max_generous < REQUIRED_SAVING_BPW
            else "STOP_CROSSFIT_OR_CHARGE_BELOW_GATE"
        )
    )
    output = {
        "schema": "qwen_moe_bitplane_context_probe_v1",
        "decision": decision,
        "scope": {
            "strict_ptq": True,
            "cpu_only": True,
            "numpy_version": np.__version__,
            "cuda_imported": False,
            "source_weights_modified": False,
            "task_data_used": False,
            "base_model_trained": False,
            "whole_layer_and_expert_heldout": True,
            "fold_rule": "sorted unique layer rank modulo 2; all experts/roles in a layer share a fold",
            "semantic_orientation": "Up native; Down transposed; fixed globally",
            "pilot": args.pilot,
        },
        "inventory": {
            "tensor_dir": str(args.tensor_dir),
            "tensors": len(specs),
            "layers": layers,
            "layer_count": len(layers),
            "layer_experts": len(experts),
            "weights": sum(s.path.stat().st_size // 2 for s in specs),
            "source_manifest": manifest,
        },
        "quantizer": {
            "type": "per-matrix-RMS normalized symmetric Lloyd-Max",
            "codebooks_fitted_on_training_layers_only": True,
            "matrix_scale_storage_bits": MATRIX_SCALE_BITS,
            "matrix_framing_bits": MATRIX_FRAMING_BITS,
            "stream_termination_bits": STREAM_TERMINATION_BITS,
            "reconstruction_table_storage": "FP16; charged in rate interpretation",
        },
        "probability_model": {
            "type": "dense finite conditional tables with fixed Dirichlet backoff",
            "dirichlet_strength": DIRICHLET_STRENGTH,
            "probability_bits": PROBABILITY_BITS,
            "float_cross_entropy_is_optimistic": True,
            "exact_entropy_coder_run": False,
            "heldout_plugin_oracle_is_leaky_and_optimistic": True,
        },
        "gate": {
            "target_F": 0.8,
            "required_net_entropy_saving_bpw": REQUIRED_SAVING_BPW,
            "identity": "2^(-2*saving) <= 0.8",
            "allowed_physical_rate_bpw": [2.15, 2.5],
            "max_expert_read_amplification": 2.0,
        },
        "results": results,
        "best_legal_crossfit": best_crossfit,
        "best_ordinary_optimistic_oracle": best_oracle,
        "maximum_generous_oracle_gain_across_all_tests_bpw": max_generous,
        "generous_oracle_shortfall_bpw": REQUIRED_SAVING_BPW - max_generous,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "best_legal_crossfit": {
                    "alphabet": best_crossfit["alphabet"],
                    "scheme": best_crossfit["name"],
                    "net_gain_bpw": best_crossfit["net_crossfit_gain_bpw"],
                },
                "best_ordinary_oracle": {
                    "alphabet": best_oracle["alphabet"],
                    "scheme": best_oracle["name"],
                    "gain_bpw": best_oracle[
                        "optimistic_heldout_plugin_gain_bpw"
                    ],
                },
                "maximum_generous_oracle_gain_bpw": max_generous,
                "required_bpw": REQUIRED_SAVING_BPW,
                "elapsed_seconds": output["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
