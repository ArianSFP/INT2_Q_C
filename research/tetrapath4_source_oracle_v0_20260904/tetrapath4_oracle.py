"""Source-only TETRAPATH-4 dominant rate-distortion oracle.

This module accepts caller-supplied distortion fields only.  It has no model,
network, accelerator, filesystem-payload, or deployment entry point.  Empirical
probability tables and time sharing are deliberately free: this is a kill-only
upper envelope, never a finite-codec promotion result.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import math
from typing import Iterable, Sequence

import numpy as np


VARIABLES = 4
ALPHABET = 4
TUPLES = ALPHABET ** VARIABLES
SMOOTHING = Fraction(1, 2)
MAX_ALTERNATIONS = 12
EARLY_KILL_BPW = 0.045
STANDALONE_UPDOWN_BPW = 0.22933495044437174
ENGINEERING_MARGIN_BPW = 0.27
LAMBDA_GRID = tuple(Fraction(n, 4096) for n in
                    (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
                     2048, 4096, 8192, 16384))
PAIRINGS = {
    "pair_01_23": ((0, 1), (2, 3)),
    "pair_02_13": ((0, 2), (1, 3)),
    "pair_03_12": ((0, 3), (1, 2)),
}
FIBER_FAMILIES = ("fiber_gray_low", "fiber_gray_high", "fiber_gf4_xor")
FAMILIES = ("independent", *PAIRINGS, "tree", "pairwise_maxent", "parity",
            *FIBER_FAMILIES, "full")


class OracleError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def tuple_table() -> np.ndarray:
    """Canonical lexicographic [256,4] quaternary tuple table."""
    return np.asarray(list(itertools.product(range(ALPHABET), repeat=VARIABLES)),
                      dtype=np.uint8)


QTABLE = tuple_table()
QPOWERS = np.asarray((64, 16, 4, 1), dtype=np.int64)


def tuple_ids(labels: np.ndarray) -> np.ndarray:
    q = np.asarray(labels)
    require(q.ndim == 2 and q.shape[1] == VARIABLES and
            np.issubdtype(q.dtype, np.integer) and
            bool(np.all((q >= 0) & (q < ALPHABET))), "labels must be [N,4] in 0..3")
    return (q.astype(np.int64) @ QPOWERS).astype(np.int64)


def labels_from_ids(ids: np.ndarray) -> np.ndarray:
    x = np.asarray(ids)
    require(x.ndim == 1 and np.issubdtype(x.dtype, np.integer) and
            bool(np.all((x >= 0) & (x < TUPLES))), "tuple ids")
    return QTABLE[x.astype(np.int64)].copy()


def validate_costs(costs: np.ndarray, source_energy: float) -> np.ndarray:
    c = np.asarray(costs)
    require(c.dtype == np.float64 and c.ndim == 3 and
            c.shape[1:] == (VARIABLES, ALPHABET) and c.shape[0] > 0 and
            bool(np.all(np.isfinite(c))) and bool(np.all(c >= 0)),
            "costs must be finite nonnegative FP64 [N,4,4]")
    require(math.isfinite(float(source_energy)) and float(source_energy) > 0,
            "positive finite source energy")
    return np.ascontiguousarray(c)


def fourway_distortion_costs(values: np.ndarray, reconstructions: np.ndarray) -> tuple[np.ndarray, float]:
    """Build exact scalar distortion fields for [N,4] values and [N,4,4] levels."""
    x, y = np.asarray(values), np.asarray(reconstructions)
    require(x.dtype == np.float64 and x.ndim == 2 and x.shape[1] == VARIABLES and
            y.dtype == np.float64 and y.shape == (x.shape[0], VARIABLES, ALPHABET) and
            bool(np.all(np.isfinite(x))) and bool(np.all(np.isfinite(y))),
            "canonical FP64 values/levels")
    energy = float(np.square(x).sum(dtype=np.float64))
    require(energy > 0 and math.isfinite(energy), "source energy")
    return np.square(x[:, :, None] - y, dtype=np.float64), energy


def aligned_up_down_values(up_e: np.ndarray, down_e_transposed: np.ndarray,
                           up_f: np.ndarray, down_f_transposed: np.ndarray) -> np.ndarray:
    """Canonical aligned geometry: Down must already be transposed to Up layout."""
    arrays = [np.asarray(v) for v in
              (up_e, down_e_transposed, up_f, down_f_transposed)]
    require(all(v.dtype == np.float64 and v.ndim == 2 and v.shape == arrays[0].shape and
                bool(np.all(np.isfinite(v))) for v in arrays),
            "aligned FP64 Up and explicitly transposed Down matrices")
    return np.column_stack(tuple(v.reshape(-1, order="C") for v in arrays))


def expanded_tuple_costs(costs: np.ndarray) -> np.ndarray:
    c = np.asarray(costs)
    require(c.dtype == np.float64 and c.ndim == 3 and c.shape[1:] ==
            (VARIABLES, ALPHABET), "distortion field geometry")
    out = np.zeros((c.shape[0], TUPLES), dtype=np.float64)
    rows = np.arange(c.shape[0], dtype=np.int64)[:, None]
    for variable in range(VARIABLES):
        out += c[rows, variable, QTABLE[None, :, variable]]
    return out


def _entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log2(p[positive]), dtype=np.float64))


def _empirical(ids: np.ndarray, cardinality: int) -> np.ndarray:
    counts = np.bincount(np.asarray(ids, dtype=np.int64), minlength=cardinality)[:cardinality]
    require(int(counts.sum()) > 0, "empty empirical distribution")
    return counts.astype(np.float64) / float(counts.sum())


def _pair_code(labels: np.ndarray, a: int, b: int) -> np.ndarray:
    return labels[:, a].astype(np.int64) * ALPHABET + labels[:, b].astype(np.int64)


def _mutual_information(labels: np.ndarray, a: int, b: int) -> float:
    pa = _empirical(labels[:, a], ALPHABET)
    pb = _empirical(labels[:, b], ALPHABET)
    pab = _empirical(_pair_code(labels, a, b), ALPHABET ** 2).reshape(ALPHABET, ALPHABET)
    product = pa[:, None] * pb[None, :]
    positive = pab > 0
    return float(np.sum(pab[positive] * np.log2(pab[positive] / product[positive])))


def all_labeled_trees() -> tuple[tuple[tuple[int, int], ...], ...]:
    """All 4^(4-2)=16 labelled trees, in deterministic edge order."""
    trees = set()
    for seq in itertools.product(range(VARIABLES), repeat=VARIABLES - 2):
        degree = [1] * VARIABLES
        for v in seq:
            degree[v] += 1
        edges = []
        for v in seq:
            leaf = min(i for i in range(VARIABLES) if degree[i] == 1)
            edges.append(tuple(sorted((leaf, v))))
            degree[leaf] -= 1
            degree[v] -= 1
        remaining = [i for i in range(VARIABLES) if degree[i] == 1]
        edges.append(tuple(sorted(remaining)))
        trees.add(tuple(sorted(edges)))
    result = tuple(sorted(trees))
    require(len(result) == 16, "Cayley tree enumeration")
    return result


TREES = all_labeled_trees()


def chow_liu_tree(labels: np.ndarray) -> tuple[tuple[int, int], ...]:
    q = labels_from_ids(tuple_ids(labels))
    mi = {(a, b): _mutual_information(q, a, b)
          for a in range(VARIABLES) for b in range(a + 1, VARIABLES)}
    return min(TREES, key=lambda tree: (-sum(mi[e] for e in tree), tree))


def _smoothed_counts(ids: np.ndarray, cardinality: int) -> np.ndarray:
    counts = np.bincount(np.asarray(ids, dtype=np.int64), minlength=cardinality)[:cardinality]
    alpha = float(SMOOTHING)
    return (counts.astype(np.float64) + alpha) / (float(counts.sum()) + alpha * cardinality)


def _independent_probability(labels: np.ndarray) -> np.ndarray:
    p = np.ones(TUPLES, dtype=np.float64)
    for v in range(VARIABLES):
        marginal = _smoothed_counts(labels[:, v], ALPHABET)
        p *= marginal[QTABLE[:, v]]
    return p / p.sum()


def _paired_probability(labels: np.ndarray, pairing: Sequence[tuple[int, int]]) -> np.ndarray:
    p = np.ones(TUPLES, dtype=np.float64)
    for a, b in pairing:
        joint = _smoothed_counts(_pair_code(labels, a, b), ALPHABET ** 2)
        p *= joint[QTABLE[:, a].astype(np.int64) * ALPHABET + QTABLE[:, b]]
    return p / p.sum()


def _tree_probability(labels: np.ndarray, tree: Sequence[tuple[int, int]]) -> np.ndarray:
    adjacency = {v: [] for v in range(VARIABLES)}
    for a, b in tree:
        adjacency[a].append(b)
        adjacency[b].append(a)
    root = 0
    parent = {root: -1}
    order = [root]
    for node in order:
        for child in sorted(adjacency[node]):
            if child not in parent:
                parent[child] = node
                order.append(child)
    root_p = _smoothed_counts(labels[:, root], ALPHABET)
    p = root_p[QTABLE[:, root]].copy()
    alpha = float(SMOOTHING)
    for child in order[1:]:
        par = parent[child]
        counts = np.full((ALPHABET, ALPHABET), alpha, dtype=np.float64)
        np.add.at(counts, (labels[:, par], labels[:, child]), 1.0)
        conditional = counts / counts.sum(axis=1, keepdims=True)
        p *= conditional[QTABLE[:, par], QTABLE[:, child]]
    return p / p.sum()


def parity_signature(labels: np.ndarray) -> np.ndarray:
    """Two Gray-bit four-way parities, yielding a four-state signature."""
    q = np.asarray(labels, dtype=np.uint8)
    require(q.ndim == 2 and q.shape[1] == VARIABLES and
            bool(np.all(q < ALPHABET)), "parity labels")
    gray = q ^ (q >> 1)
    low = np.bitwise_xor.reduce(gray & 1, axis=1)
    high = np.bitwise_xor.reduce((gray >> 1) & 1, axis=1)
    return (low + 2 * high).astype(np.uint8)


QPARITY = parity_signature(QTABLE)


def _parity_probability(labels: np.ndarray) -> np.ndarray:
    """Unary plus two-bit four-way parity log-linear family via exact IPF."""
    target_unary = [_smoothed_counts(labels[:, v], ALPHABET) for v in range(VARIABLES)]
    target_parity = _smoothed_counts(parity_signature(labels), 4)
    p = np.full(TUPLES, 1.0 / TUPLES, dtype=np.float64)
    for _ in range(128):
        for v in range(VARIABLES):
            current = np.bincount(QTABLE[:, v], weights=p, minlength=ALPHABET)
            p *= (target_unary[v] / current)[QTABLE[:, v]]
            p /= p.sum()
        current = np.bincount(QPARITY, weights=p, minlength=4)
        p *= (target_parity / current)[QPARITY]
        p /= p.sum()
    require(bool(np.all(np.isfinite(p))) and bool(np.all(p > 0)), "parity IPF")
    return p


def _pairwise_maxent_probability(labels: np.ndarray, smoothed: bool = True) -> np.ndarray:
    """Maximum-entropy surrogate matching all six two-way marginals by IPF."""
    alpha = float(SMOOTHING) if smoothed else 0.0
    targets = {}
    for a in range(VARIABLES):
        for b in range(a + 1, VARIABLES):
            counts = np.bincount(_pair_code(labels, a, b), minlength=16).astype(np.float64)
            counts += alpha
            targets[(a, b)] = counts.reshape(4, 4) / counts.sum()
    p = np.full(TUPLES, 1.0 / TUPLES, dtype=np.float64)
    for _ in range(1024 if not smoothed else 256):
        old = p.copy()
        for (a, b), target in targets.items():
            current = np.zeros((4, 4), dtype=np.float64)
            np.add.at(current, (QTABLE[:, a], QTABLE[:, b]), p)
            ratio = np.divide(target, current, out=np.zeros_like(target), where=current > 0)
            p *= ratio[QTABLE[:, a], QTABLE[:, b]]
            total = p.sum()
            require(total > 0 and math.isfinite(float(total)), "pairwise maxent IPF")
            p /= total
        if float(np.max(np.abs(p - old))) < 1e-14:
            break
    return p


def _fiber_pair_map(family: str) -> np.ndarray:
    require(family in FIBER_FAMILIES, "fiber family")
    q = np.arange(4, dtype=np.uint8)
    gray = q ^ (q >> 1)
    a, b = np.meshgrid(gray, gray, indexing="ij")
    if family == "fiber_gray_low":
        return ((a & 1) ^ (b & 1)).astype(np.uint8)
    if family == "fiber_gray_high":
        return (((a >> 1) & 1) ^ ((b >> 1) & 1)).astype(np.uint8)
    return (a ^ b).astype(np.uint8)


def _fiber_support(family: str) -> tuple[np.ndarray, np.ndarray, int]:
    fmap = _fiber_pair_map(family)
    z0 = fmap[QTABLE[:, 0], QTABLE[:, 1]]
    z1 = fmap[QTABLE[:, 2], QTABLE[:, 3]]
    return z0 == z1, z0, int(fmap.max()) + 1


def _fiber_probability(labels: np.ndarray, family: str) -> np.ndarray:
    """Gray-Wyner-local p(z)p(q_e|z)p(q_f|z), on exact equal fibers."""
    fmap = _fiber_pair_map(family)
    support, tuple_z, states = _fiber_support(family)
    z = fmap[labels[:, 0], labels[:, 1]]
    require(bool(np.all(z == fmap[labels[:, 2], labels[:, 3]])),
            "assignments outside equal-fiber support")
    alpha = float(SMOOTHING)
    z_counts = np.bincount(z, minlength=states).astype(np.float64) + alpha
    pz = z_counts / z_counts.sum()
    conditionals = []
    for left, right in ((0, 1), (2, 3)):
        counts = np.zeros((states, 16), dtype=np.float64)
        pair = _pair_code(labels, left, right)
        for state in range(states):
            allowed = fmap.reshape(-1) == state
            counts[state, allowed] = alpha
        np.add.at(counts, (z, pair), 1.0)
        counts /= counts.sum(axis=1, keepdims=True)
        conditionals.append(counts)
    pair0 = QTABLE[:, 0].astype(np.int64) * 4 + QTABLE[:, 1]
    pair1 = QTABLE[:, 2].astype(np.int64) * 4 + QTABLE[:, 3]
    p = np.zeros(TUPLES, dtype=np.float64)
    p[support] = (pz[tuple_z[support]] *
                  conditionals[0][tuple_z[support], pair0[support]] *
                  conditionals[1][tuple_z[support], pair1[support]])
    require(abs(float(p.sum()) - 1.0) < 1e-11, "fiber normalization")
    return p


def fit_probability(labels: np.ndarray, family: str) -> tuple[np.ndarray, dict]:
    q = labels_from_ids(tuple_ids(labels))
    require(family in FAMILIES, "known family")
    meta: dict = {}
    if family == "independent":
        p = _independent_probability(q)
    elif family in PAIRINGS:
        p = _paired_probability(q, PAIRINGS[family])
        meta["pairing"] = [list(x) for x in PAIRINGS[family]]
    elif family == "tree":
        tree = chow_liu_tree(q)
        p = _tree_probability(q, tree)
        meta["tree"] = [list(x) for x in tree]
    elif family == "pairwise_maxent":
        p = _pairwise_maxent_probability(q, smoothed=True)
        meta["constraints"] = "all six pairwise marginals; no explicit 3/4-way term"
    elif family == "parity":
        p = _parity_probability(q)
        meta["parity"] = "xor of each Gray bitplane across all four variables"
    elif family in FIBER_FAMILIES:
        p = _fiber_probability(q, family)
        meta["local_decode"] = "common fiber z plus one expert-local pair conditional"
    else:
        p = _smoothed_counts(tuple_ids(q), TUPLES)
    require(p.shape == (TUPLES,) and bool(np.all(np.isfinite(p))) and
            bool(np.all(p >= 0)) and bool(np.any(p > 0)) and
            abs(float(p.sum()) - 1.0) < 1e-11,
            "valid fitted probability")
    return p, meta


def fixed_assignment_census(labels: np.ndarray) -> dict:
    """Unsmoothed plug-in entropy census, exact for balanced fixtures."""
    q = labels_from_ids(tuple_ids(labels))
    result = {"coordinates": int(q.shape[0])}
    individual = sum(_entropy(_empirical(q[:, v], ALPHABET)) for v in range(VARIABLES)) / VARIABLES
    result["independent_bpw"] = individual
    pair_rates = {}
    for name, pairing in PAIRINGS.items():
        pair_rates[name] = sum(_entropy(_empirical(_pair_code(q, *edge), 16))
                               for edge in pairing) / VARIABLES
    result["pair_bpw"] = pair_rates
    result["best_pair_bpw"] = min(pair_rates.values())
    tree = chow_liu_tree(q)
    tree_rate = individual - sum(_mutual_information(q, *edge) for edge in tree) / VARIABLES
    result["tree"] = [list(x) for x in tree]
    result["tree_bpw"] = tree_rate
    result["full_bpw"] = _entropy(_empirical(tuple_ids(q), TUPLES)) / VARIABLES
    pairwise_maxent = _pairwise_maxent_probability(q, smoothed=False)
    observed = tuple_ids(q)
    require(bool(np.all(pairwise_maxent[observed] > 0)), "maxent covers observations")
    result["pairwise_maxent_bpw"] = float(-np.mean(np.log2(pairwise_maxent[observed])) /
                                               VARIABLES)
    # The deployable sparse model is evaluated by its exact fitted distribution.
    parity_p, _ = fit_probability(q, "parity")
    result["parity_bpw"] = float(-np.mean(np.log2(parity_p[tuple_ids(q)])) / VARIABLES)
    result["fourway_gain_over_best_factorized_bpw"] = (
        min(individual, result["best_pair_bpw"], tree_rate) - result["full_bpw"])
    result["total_correlation_bpw"] = individual - result["full_bpw"]
    result["best_2plus2_saving_bpw"] = individual - result["best_pair_bpw"]
    result["best_chow_liu_saving_bpw"] = individual - tree_rate
    result["residual_connected_information_bpw"] = (
        result["pairwise_maxent_bpw"] - result["full_bpw"])
    result["pairwise_mi"] = {f"{a}{b}": _mutual_information(q, a, b)
                             for a in range(VARIABLES) for b in range(a + 1, VARIABLES)}
    # Bind the exact assignment without retaining source data in a report.
    result["label_sha256"] = hashlib.sha256(q.tobytes(order="C")).hexdigest()
    return result


def fiber_fixed_ledger(labels: np.ndarray, family: str) -> dict:
    """Ideal logical common/private ledger; returns encodable false off-fiber."""
    q = labels_from_ids(tuple_ids(labels))
    fmap = _fiber_pair_map(family)
    z0 = fmap[q[:, 0], q[:, 1]]
    z1 = fmap[q[:, 2], q[:, 3]]
    if not bool(np.all(z0 == z1)):
        return {"family": family, "encodable": False}
    z = z0
    hz = _entropy(_empirical(z, int(fmap.max()) + 1))
    private = []
    for a, b in ((0, 1), (2, 3)):
        pair = _pair_code(q, a, b)
        h_pair_z = _entropy(_empirical(z.astype(np.int64) * 16 + pair,
                                      (int(fmap.max()) + 1) * 16)) - hz
        private.append(h_pair_z)
    total = hz + sum(private)
    ownership = total / 2.0
    amp = [((hz + private[e]) / ownership if ownership > 0 else 1.0) for e in range(2)]
    return {
        "family": family, "encodable": True,
        "common_bits_per_tetrad": hz,
        "private_bits_per_tetrad": private,
        "total_bits_per_tetrad": total,
        "total_bpw": total / 4.0,
        "logical_read_amplification_by_expert": amp,
        "max_logical_read_amplification": max(amp),
        "warning": "logical entropy ledger only; page-rounded finite bytes are not constructed",
    }


def symmetric_multistarts(nearest: np.ndarray) -> tuple[np.ndarray, ...]:
    q = labels_from_ids(tuple_ids(nearest))
    offsets = {(0, 0, 0, 0)}
    for s in (1, 2, 3):
        offsets.add((s, s, s, s))
        for v in range(VARIABLES):
            x = [0] * VARIABLES
            x[v] = s
            offsets.add(tuple(x))
    for s in (1, 2, 3):
        for a in range(VARIABLES):
            for b in range(a + 1, VARIABLES):
                x = [0] * VARIABLES
                x[a] = x[b] = s
                offsets.add(tuple(x))
    ordered = sorted(offsets, key=lambda x: (sum(v != 0 for v in x), x))
    return tuple(((q.astype(np.int16) + np.asarray(offset, dtype=np.int16)) % ALPHABET)
                 .astype(np.uint8) for offset in ordered)


def assign_given_probability(tuple_costs: np.ndarray, probability: np.ndarray,
                             bit_weight: float) -> np.ndarray:
    d, p = np.asarray(tuple_costs), np.asarray(probability)
    require(d.dtype == np.float64 and d.ndim == 2 and d.shape[1] == TUPLES and
            p.dtype == np.float64 and p.shape == (TUPLES,) and
            bool(np.all(p >= 0)) and bool(np.any(p > 0)) and
            math.isfinite(float(bit_weight)) and bit_weight >= 0,
            "assignment inputs")
    bits = np.full(TUPLES, math.inf, dtype=np.float64)
    positive = p > 0
    bits[positive] = -np.log2(p[positive])
    # Even at lambda zero, a structurally forbidden tuple remains forbidden.
    penalty = np.full(TUPLES, math.inf, dtype=np.float64)
    penalty[positive] = float(bit_weight) * bits[positive]
    objective = d + penalty[None, :]
    return np.argmin(objective, axis=1).astype(np.uint16)


@dataclass(frozen=True)
class RDPoint:
    family: str
    lambda_numerator: int
    lambda_denominator: int
    rate_bpw: float
    relative_mse: float
    objective: float
    assignment_sha256: str
    fit_meta: dict

    def as_dict(self) -> dict:
        return {
            "family": self.family,
            "lambda": [self.lambda_numerator, self.lambda_denominator],
            "rate_bpw": self.rate_bpw,
            "relative_mse": self.relative_mse,
            "objective_relative_mse_plus_lambda_bpw": self.objective,
            "assignment_sha256": self.assignment_sha256,
            "fit_meta": self.fit_meta,
        }


def optimize_family(costs: np.ndarray, source_energy: float, family: str,
                    multiplier: Fraction, starts: Sequence[np.ndarray] | None = None) -> RDPoint:
    c = validate_costs(costs, source_energy)
    require(family in FAMILIES and isinstance(multiplier, Fraction) and multiplier >= 0,
            "family/multiplier")
    tc = expanded_tuple_costs(c)
    nearest = labels_from_ids(np.argmin(tc, axis=1))
    start_bank = symmetric_multistarts(nearest) if starts is None else tuple(starts)
    require(len(start_bank) > 0, "multistart bank")
    bit_weight = float(multiplier) * float(source_energy) / (VARIABLES * c.shape[0])
    best = None
    for start_index, start in enumerate(start_bank):
        labels = labels_from_ids(tuple_ids(start))
        if family in FIBER_FAMILIES:
            support, _, _ = _fiber_support(family)
            # Project each symmetric start into the public equal-fiber codebook
            # by label Hamming distance. This changes initialization only; every
            # reported point is subsequently scored by the exact RD objective.
            hamming = np.count_nonzero(QTABLE[None, :, :] != labels[:, None, :], axis=2)
            projection = np.where(support[None, :], hamming, VARIABLES + 1)
            labels = labels_from_ids(np.argmin(projection, axis=1))
        seen = set()
        candidates = []
        for _ in range(MAX_ALTERNATIONS):
            p, meta = fit_probability(labels, family)
            ids = assign_given_probability(tc, p, bit_weight)
            key = hashlib.sha256(ids.astype("<u2", copy=False).tobytes()).hexdigest()
            candidates.append((ids, p, meta))
            if key in seen or np.array_equal(ids, tuple_ids(labels)):
                break
            seen.add(key)
            labels = labels_from_ids(ids)
        for ids, _, _ in candidates:
            final_labels = labels_from_ids(ids)
            final_p, final_meta = fit_probability(final_labels, family)
            sse = float(tc[np.arange(c.shape[0]), ids].sum(dtype=np.float64))
            require(bool(np.all(final_p[ids] > 0)), "assigned tuple probability")
            rate = float(-np.log2(final_p[ids]).sum(dtype=np.float64) /
                         (VARIABLES * c.shape[0]))
            mse = sse / float(source_energy)
            objective = mse + float(multiplier) * rate
            identity = hashlib.sha256(ids.astype("<u2", copy=False).tobytes()).hexdigest()
            candidate = (objective, rate, mse, identity, start_index, final_meta)
            if best is None or candidate[:5] < best[:5]:
                best = candidate
    require(best is not None, "optimizer result")
    return RDPoint(family, multiplier.numerator, multiplier.denominator,
                   best[1], best[2], best[0], best[3], best[5])


def lower_convex_frontier(points: Iterable[RDPoint | tuple[float, float]]) -> list[tuple[float, float]]:
    raw = []
    for point in points:
        raw.append((float(point.rate_bpw), float(point.relative_mse)) if isinstance(point, RDPoint)
                   else (float(point[0]), float(point[1])))
    require(raw and all(math.isfinite(r) and math.isfinite(d) and r >= 0 and d > 0
                        for r, d in raw), "frontier points")
    by_rate = {}
    for rate, distortion in raw:
        by_rate[rate] = min(distortion, by_rate.get(rate, math.inf))
    nondominated = []
    best_d = math.inf
    for rate, distortion in sorted(by_rate.items()):
        if distortion < best_d - 1e-15:
            nondominated.append((rate, distortion))
            best_d = distortion
    hull: list[tuple[float, float]] = []
    for point in nondominated:
        while len(hull) >= 2:
            s1 = (hull[-1][1] - hull[-2][1]) / (hull[-1][0] - hull[-2][0])
            s2 = (point[1] - hull[-1][1]) / (point[0] - hull[-1][0])
            if s1 < s2 - 1e-15:
                break
            hull.pop()
        hull.append(point)
    return hull


def _d_at_rate(hull: Sequence[tuple[float, float]], rate: float) -> float:
    require(hull and hull[0][0] - 1e-12 <= rate <= hull[-1][0] + 1e-12, "rate overlap")
    if len(hull) == 1:
        return hull[0][1]
    for (r0, d0), (r1, d1) in zip(hull, hull[1:]):
        if rate <= r1 + 1e-12:
            t = (rate - r0) / (r1 - r0)
            return d0 + t * (d1 - d0)
    return hull[-1][1]


def _r_at_distortion(hull: Sequence[tuple[float, float]], distortion: float) -> float:
    require(hull and hull[-1][1] - 1e-12 <= distortion <= hull[0][1] + 1e-12,
            "distortion overlap")
    if len(hull) == 1:
        return hull[0][0]
    for (r0, d0), (r1, d1) in zip(hull, hull[1:]):
        if d1 - 1e-12 <= distortion <= d0 + 1e-12:
            t = (distortion - d0) / (d1 - d0)
            return r0 + t * (r1 - r0)
    return hull[-1][0]


def compare_frontiers(baseline: Sequence[tuple[float, float]], challenger: Sequence[tuple[float, float]]) -> dict:
    """Optimistic time-shared advantages at equal rate and equal distortion."""
    a, b = lower_convex_frontier(baseline), lower_convex_frontier(challenger)
    rate_lo, rate_hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    rate_gain = -math.inf
    rate_witness = None
    if rate_lo <= rate_hi + 1e-12:
        rates = sorted({rate_lo, rate_hi, *(r for r, _ in a if rate_lo <= r <= rate_hi),
                        *(r for r, _ in b if rate_lo <= r <= rate_hi)})
        for rate in rates:
            da, db = _d_at_rate(a, rate), _d_at_rate(b, rate)
            gain = 0.5 * math.log2(da / db)
            if gain > rate_gain:
                rate_gain, rate_witness = gain, [rate, da, db]
    d_lo, d_hi = max(a[-1][1], b[-1][1]), min(a[0][1], b[0][1])
    distortion_gain = -math.inf
    distortion_witness = None
    if d_lo <= d_hi + 1e-12:
        distortions = sorted({d_lo, d_hi, *(d for _, d in a if d_lo <= d <= d_hi),
                              *(d for _, d in b if d_lo <= d <= d_hi)}, reverse=True)
        for distortion in distortions:
            ra, rb = _r_at_distortion(a, distortion), _r_at_distortion(b, distortion)
            gain = ra - rb
            if gain > distortion_gain:
                distortion_gain, distortion_witness = gain, [distortion, ra, rb]
    candidates = [x for x in (rate_gain, distortion_gain) if math.isfinite(x)]
    return {
        "equal_rate_best_equivalent_bpw": None if not math.isfinite(rate_gain) else rate_gain,
        "equal_rate_witness_rate_Dbase_Dchallenger": rate_witness,
        "equal_mse_best_saved_bpw": None if not math.isfinite(distortion_gain) else distortion_gain,
        "equal_mse_witness_D_Rbase_Rchallenger": distortion_witness,
        "optimistic_best_equivalent_bpw": max(candidates) if candidates else None,
    }


def run_dominant_oracle(costs: np.ndarray, source_energy: float,
                        multipliers: Sequence[Fraction] = LAMBDA_GRID) -> dict:
    """Run all equally flexible families and return the free-table kill envelope."""
    c = validate_costs(costs, source_energy)
    multipliers = tuple(multipliers)
    require(multipliers and all(isinstance(x, Fraction) and x >= 0 for x in multipliers) and
            list(multipliers) == sorted(set(multipliers)), "canonical multipliers")
    nearest = labels_from_ids(np.argmin(expanded_tuple_costs(c), axis=1))
    starts = symmetric_multistarts(nearest)
    results = {family: [optimize_family(c, source_energy, family, lam, starts)
                        for lam in multipliers] for family in FAMILIES}
    hulls = {family: lower_convex_frontier(points) for family, points in results.items()}
    best_pair_points = [point for family in PAIRINGS for point in results[family]]
    best_factorized_points = results["independent"] + best_pair_points + results["tree"]
    best_factorized_hull = lower_convex_frontier(best_factorized_points)
    full_vs_independent = compare_frontiers(hulls["independent"], hulls["full"])
    full_vs_pair = compare_frontiers(lower_convex_frontier(best_pair_points), hulls["full"])
    full_vs_tree = compare_frontiers(hulls["tree"], hulls["full"])
    full_vs_pairwise_maxent = compare_frontiers(hulls["pairwise_maxent"], hulls["full"])
    full_comparison = compare_frontiers(best_factorized_hull, hulls["full"])
    sparse_comparison = compare_frontiers(best_factorized_hull, hulls["parity"])
    full_gain = full_comparison["optimistic_best_equivalent_bpw"]
    sparse_gain = sparse_comparison["optimistic_best_equivalent_bpw"]
    require(full_gain is not None, "full-joint comparison overlap")
    best_fourway = max(x for x in (full_gain, sparse_gain) if x is not None)
    status = ("HARD_KILL_MEMORYLESS_FOURWAY_BELOW_0P045_BPW" if
              full_gain < EARLY_KILL_BPW else
              "SURVIVES_KILL_ONLY_ORACLE_REQUIRES_HELDOUT_CONTROLS_AND_FINITE_PACKET")
    return {
        "schema": "tetrapath4.dominant_oracle.v0",
        "status": status,
        "kill_only": True,
        "tables_and_time_sharing_charged": False,
        "coordinates": int(c.shape[0]),
        "weights": int(VARIABLES * c.shape[0]),
        "source_energy": float(source_energy),
        "multistarts": len(starts),
        "global_multipliers": [[x.numerator, x.denominator] for x in multipliers],
        "thresholds_bpw": {
            "early_kill": EARLY_KILL_BPW,
            "standalone_updown": STANDALONE_UPDOWN_BPW,
            "engineering_margin": ENGINEERING_MARGIN_BPW,
        },
        "frontiers": {family: [[r, d] for r, d in hull] for family, hull in hulls.items()},
        "points": {family: [point.as_dict() for point in points]
                   for family, points in results.items()},
        "best_factorized_frontier": [[r, d] for r, d in best_factorized_hull],
        "full_over_independent_codec_relevance": full_vs_independent,
        "full_over_best_2plus2": full_vs_pair,
        "full_over_chow_liu_tree": full_vs_tree,
        "full_over_pairwise_maxent_residual_synergy": full_vs_pairwise_maxent,
        "full_over_best_factorized": full_comparison,
        "sparse_parity_over_best_factorized": sparse_comparison,
        "full_joint_best_G4_bpw": full_gain,
        "best_memoryless_fourway_equivalent_gain_bpw": best_fourway,
        "nearest_assignment_census_diagnostic_only": fixed_assignment_census(nearest),
        "nearest_assignment_fiber_ledgers": {
            family: fiber_fixed_ledger(nearest, family) for family in FIBER_FAMILIES
        },
        "scientific_boundary": (
            "A survival is not Qwen evidence or a finite codec: empirical tables, model "
            "selection, and convexified time sharing are free. A sub-threshold result is "
            "a valid hard kill for these memoryless four-variable families."),
    }


def payload_execution_gate(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("TETRAPATH-4 v0 is source-only and grants no payload execution")
