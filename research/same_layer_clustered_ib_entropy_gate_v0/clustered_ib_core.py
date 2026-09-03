"""Source-only mathematics for the CBIB-1 same-layer entropy aperture.

The module accepts already-quantized labels only.  It contains no filesystem,
model, network, or payload access.  A binary latent is fitted independently in
each flat expert group and is evaluated strictly out of fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


ALPHABET = 4
ROLES = 2
GROUP_SIZES = (2, 4, 8, 16)
FOLD_COUNT = 8
SUPERBLOCK_VALUES = 2048
PAGE_BYTES = 4096
GLOBAL_HEADER_BYTES = 4096
GROUP_HEADER_BYTES = 256
PRIVATE_HEADER_BYTES = 256
TARGET_GAIN_BPW = 0.22933495044437175
RATE_ENDPOINTS = (Fraction(43, 20), Fraction(5, 2))
RATE_MIN = RATE_ENDPOINTS[0]
RATE_MAX = RATE_ENDPOINTS[1]
KT_ALPHA = 0.5
MAX_EM_ITERATIONS = 32
CONTROL_SEEDS = (
    0xCB1B0001,
    0xCB1B0003,
    0xCB1B0007,
    0xCB1B000B,
    0xCB1B0011,
    0xCB1B0013,
    0xCB1B0017,
    0xCB1B001D,
)
GRAY_PLANES = np.asarray(((0, 0), (0, 1), (1, 1), (1, 0)), dtype=np.uint8)


class GateError(ValueError):
    """A fail-closed validation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def ceil_div(a: int, b: int) -> int:
    _require(isinstance(a, int) and isinstance(b, int) and a >= 0 and b > 0,
             "ceil_div domain")
    return (a + b - 1) // b


def ceil_log2_states(states: int) -> int:
    _require(isinstance(states, int) and states >= 1, "state count")
    return 0 if states == 1 else (states - 1).bit_length()


def _labels(labels: np.ndarray) -> tuple[np.ndarray, int, int]:
    q = np.asarray(labels)
    _require(q.ndim == 3, "labels must be [expert,role,coordinate]")
    e, roles, n = map(int, q.shape)
    _require(2 <= e <= 256, "expert count must be in [2,256]")
    _require(roles == ROLES and n > 0, "exactly two nonempty Up/Down roles required")
    _require(q.dtype == np.uint8, "labels must be uint8")
    _require(bool(np.all(q < ALPHABET)), "label outside four-level alphabet")
    return np.ascontiguousarray(q), e, n


def compatible_group_sizes(expert_count: int) -> tuple[int, ...]:
    _require(isinstance(expert_count, int) and 2 <= expert_count <= 256,
             "expert count")
    return tuple(k for k in GROUP_SIZES if k <= expert_count and expert_count % k == 0)


def fold_ids(coordinates: int, fold_count: int = FOLD_COUNT,
             superblock_values: int = SUPERBLOCK_VALUES) -> np.ndarray:
    _require(isinstance(coordinates, int) and coordinates > 0, "coordinates")
    _require(isinstance(fold_count, int) and fold_count >= 2, "fold count")
    _require(isinstance(superblock_values, int) and superblock_values > 0,
             "superblock size")
    blocks = ceil_div(coordinates, superblock_values)
    _require(blocks >= fold_count, "at least one superblock per fold is required")
    ids = (np.arange(coordinates, dtype=np.int64) // superblock_values) % fold_count
    _require(all(bool(np.any(ids == fold)) for fold in range(fold_count)),
             "empty fold")
    return ids.astype(np.uint8)


def entropy_bits_from_counts(counts: Iterable[int]) -> float:
    c = np.asarray(tuple(int(x) for x in counts), dtype=np.int64)
    _require(c.ndim == 1 and bool(np.all(c >= 0)), "entropy counts")
    total = int(c.sum())
    if total == 0:
        return 0.0
    z = c[c > 0].astype(np.float64)
    return float(total * math.log2(total) - np.dot(z, np.log2(z)))


def marginal_model_descriptor_bits(train_coordinates: int) -> int:
    """Three counts are sent; the fourth is derived from the row total."""
    _require(isinstance(train_coordinates, int) and train_coordinates >= 0,
             "train coordinates")
    return 3 * ceil_log2_states(train_coordinates + 1)


def binary_model_descriptor_bits(latent_counts: Sequence[int], expert_count: int) -> int:
    """Exact fixed-width count description for one binary product model.

    One latent count is sent.  For every expert/state, three categorical counts
    are sent and the fourth is derived from that transmitted state total.
    """
    counts = tuple(int(x) for x in latent_counts)
    _require(len(counts) == 2 and all(x >= 0 for x in counts), "latent counts")
    _require(isinstance(expert_count, int) and expert_count >= 2, "expert count")
    train_n = sum(counts)
    return (
        ceil_log2_states(train_n + 1)
        + expert_count * sum(3 * ceil_log2_states(value + 1) for value in counts)
    )


def partition_count(expert_count: int, group_size: int) -> int:
    _require(group_size in GROUP_SIZES and expert_count % group_size == 0,
             "equal flat partition geometry")
    groups = expert_count // group_size
    return math.factorial(expert_count) // (
        math.factorial(group_size) ** groups * math.factorial(groups)
    )


def partition_descriptor_bits(expert_count: int, group_size: int) -> int:
    return ceil_log2_states(partition_count(expert_count, group_size))


def selector_bits_for_group_bank(expert_count: int) -> int:
    choices = compatible_group_sizes(expert_count)
    _require(bool(choices), "no compatible frozen group size")
    return ceil_log2_states(len(choices))


def _kt_log_prob(count: np.ndarray | float, total: np.ndarray | float,
                 alphabet: int) -> np.ndarray:
    return np.log2((np.asarray(count, dtype=np.float64) + KT_ALPHA) /
                   (np.asarray(total, dtype=np.float64) + KT_ALPHA * alphabet))


def marginal_train_counts(group_labels: np.ndarray) -> np.ndarray:
    q = np.asarray(group_labels, dtype=np.uint8)
    _require(q.ndim == 2 and q.shape[0] >= 1, "group labels")
    _require(bool(np.all(q < ALPHABET)), "group label range")
    return np.stack([np.bincount(row, minlength=ALPHABET) for row in q], axis=0).astype(np.int64)


def marginal_nll_bits(test_labels: np.ndarray, train_counts: np.ndarray) -> tuple[float, list[float]]:
    q = np.asarray(test_labels, dtype=np.uint8)
    counts = np.asarray(train_counts, dtype=np.int64)
    _require(q.ndim == 2 and counts.shape == (q.shape[0], ALPHABET),
             "marginal score shapes")
    _require(bool(np.all(q < ALPHABET)) and bool(np.all(counts >= 0)),
             "marginal score values")
    totals = counts.sum(axis=1)
    per_expert = []
    for expert in range(q.shape[0]):
        logp = _kt_log_prob(counts[expert], totals[expert], ALPHABET)
        per_expert.append(float(-np.sum(logp[q[expert]], dtype=np.float64)))
    return float(sum(per_expert)), per_expert


@dataclass(frozen=True)
class BinaryProductModel:
    latent_counts: np.ndarray
    conditional_counts: np.ndarray
    assignments: np.ndarray
    train_nll_bits: float


def _model_counts(group_labels: np.ndarray, assignments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(group_labels, dtype=np.uint8)
    u = np.asarray(assignments, dtype=np.uint8)
    _require(q.ndim == 2 and u.shape == (q.shape[1],), "model count shapes")
    _require(bool(np.all(q < ALPHABET)) and bool(np.all(u < 2)), "model count values")
    latent = np.bincount(u, minlength=2).astype(np.int64)
    conditional = np.zeros((q.shape[0], 2, ALPHABET), dtype=np.int64)
    for expert in range(q.shape[0]):
        for state in range(2):
            conditional[expert, state] = np.bincount(q[expert, u == state], minlength=ALPHABET)
    return latent, conditional


def _assignment_costs(group_labels: np.ndarray, latent: np.ndarray,
                      conditional: np.ndarray) -> np.ndarray:
    q = np.asarray(group_labels, dtype=np.uint8)
    k, n = q.shape
    _require(latent.shape == (2,) and conditional.shape == (k, 2, ALPHABET),
             "assignment model shapes")
    latent_logp = _kt_log_prob(latent, int(latent.sum()), 2)
    costs = np.empty((2, n), dtype=np.float64)
    for state in range(2):
        cost = np.full(n, -latent_logp[state], dtype=np.float64)
        for expert in range(k):
            symbol_logp = _kt_log_prob(
                conditional[expert, state], latent[state], ALPHABET
            )
            cost -= symbol_logp[q[expert]]
        costs[state] = cost
    return costs


def evaluate_binary_model(group_labels: np.ndarray, latent: np.ndarray,
                          conditional: np.ndarray) -> dict:
    q = np.asarray(group_labels, dtype=np.uint8)
    costs = _assignment_costs(q, np.asarray(latent), np.asarray(conditional))
    # argmin is the frozen lower-state tie rule.
    u = np.argmin(costs, axis=0).astype(np.uint8)
    latent_logp = _kt_log_prob(latent, int(np.asarray(latent).sum()), 2)
    latent_bits = float(-np.sum(latent_logp[u], dtype=np.float64))
    private = []
    for expert in range(q.shape[0]):
        bits = 0.0
        for state in range(2):
            mask = u == state
            if bool(np.any(mask)):
                logp = _kt_log_prob(conditional[expert, state], latent[state], ALPHABET)
                bits -= float(np.sum(logp[q[expert, mask]], dtype=np.float64))
        private.append(bits)
    return {
        "assignments": u,
        "latent_bits": latent_bits,
        "private_bits": private,
        "total_bits": latent_bits + sum(private),
    }


def _canonicalize_states(assignments: np.ndarray, latent: np.ndarray,
                         conditional: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signature0 = tuple([int(latent[0])] + conditional[:, 0].reshape(-1).astype(int).tolist())
    signature1 = tuple([int(latent[1])] + conditional[:, 1].reshape(-1).astype(int).tolist())
    if signature1 < signature0:
        return (1 - assignments).astype(np.uint8), latent[::-1].copy(), conditional[:, ::-1].copy()
    return assignments, latent, conditional


def _initial_assignments(group_labels: np.ndarray) -> list[np.ndarray]:
    q = np.asarray(group_labels, dtype=np.uint8)
    k, n = q.shape
    candidates: list[np.ndarray] = []
    for expert in range(k):
        for plane in range(2):
            candidates.append(GRAY_PLANES[q[expert], plane].copy())
    for plane in range(2):
        bits = GRAY_PLANES[q, plane]
        candidates.append((bits.sum(axis=0) * 2 >= k).astype(np.uint8))
        candidates.append(np.bitwise_xor.reduce(bits, axis=0).astype(np.uint8))
    candidates.append((np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15) >> np.uint64(63)).astype(np.uint8))
    unique = {}
    for candidate in candidates:
        if bool(np.all(candidate == candidate[0])):
            candidate = candidate.copy()
            candidate[::2] ^= np.uint8(1)
        key = candidate.tobytes(order="C")
        unique.setdefault(key, candidate)
    return list(unique.values())


def fit_binary_product_model(group_labels: np.ndarray,
                             max_iterations: int = MAX_EM_ITERATIONS) -> BinaryProductModel:
    """Hard-EM binary latent fit with arbitrary per-coordinate assignments.

    The latent is not tied to a modal label or bitplane.  Frozen initializers
    include expert planes, modal planes, parity planes, and a public hash split.
    """
    q = np.asarray(group_labels, dtype=np.uint8)
    _require(q.ndim == 2 and 2 <= q.shape[0] <= 16 and q.shape[1] > 0,
             "binary product model geometry")
    _require(bool(np.all(q < ALPHABET)), "binary product label range")
    _require(isinstance(max_iterations, int) and max_iterations >= 1, "EM iterations")
    best = None
    for initial in _initial_assignments(q):
        u = initial.copy()
        for _ in range(max_iterations):
            latent, conditional = _model_counts(q, u)
            proposed = np.argmin(_assignment_costs(q, latent, conditional), axis=0).astype(np.uint8)
            if np.array_equal(proposed, u):
                break
            u = proposed
        latent, conditional = _model_counts(q, u)
        u, latent, conditional = _canonicalize_states(u, latent, conditional)
        evaluated = evaluate_binary_model(q, latent, conditional)
        # Frozen model complexity and counts break equal-NLL ties without source hashes.
        tie = (
            float(evaluated["total_bits"]),
            binary_model_descriptor_bits(latent, q.shape[0]),
            tuple(latent.astype(int).tolist()),
            tuple(conditional.reshape(-1).astype(int).tolist()),
        )
        if best is None or tie < best[0]:
            best = (tie, BinaryProductModel(latent, conditional, u,
                                            float(evaluated["total_bits"])))
    _require(best is not None, "no binary model initializer")
    return best[1]


def pairwise_information_scores(labels: np.ndarray) -> np.ndarray:
    """Plugin mutual information used only to build a charged partition."""
    q = np.asarray(labels, dtype=np.uint8)
    _require(q.ndim == 3 and q.shape[1] == ROLES, "pairwise labels")
    e = q.shape[0]
    scores = np.zeros((e, e), dtype=np.float64)
    for left in range(e):
        for right in range(left + 1, e):
            value = 0.0
            for role in range(ROLES):
                joint = np.bincount(
                    (q[left, role].astype(np.int64) * ALPHABET + q[right, role]).astype(np.int64),
                    minlength=ALPHABET * ALPHABET,
                ).reshape(ALPHABET, ALPHABET)
                total = int(joint.sum())
                value += (
                    entropy_bits_from_counts(joint.sum(axis=1))
                    + entropy_bits_from_counts(joint.sum(axis=0))
                    - entropy_bits_from_counts(joint.reshape(-1))
                ) / max(total, 1)
            scores[left, right] = scores[right, left] = value
    return scores


def greedy_equal_partition(scores: np.ndarray, group_size: int) -> tuple[tuple[int, ...], ...]:
    matrix = np.asarray(scores, dtype=np.float64)
    _require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "score matrix")
    e = matrix.shape[0]
    _require(group_size in GROUP_SIZES and e % group_size == 0, "partition group size")
    _require(bool(np.all(np.isfinite(matrix))) and bool(np.allclose(matrix, matrix.T)),
             "finite symmetric scores")
    remaining = set(range(e))
    groups = []
    while remaining:
        group = [min(remaining)]
        remaining.remove(group[0])
        while len(group) < group_size:
            ranked = sorted(
                remaining,
                key=lambda candidate: (
                    -float(np.mean([matrix[candidate, member] for member in group])),
                    candidate,
                ),
            )
            chosen = ranked[0]
            group.append(chosen)
            remaining.remove(chosen)
        groups.append(tuple(sorted(group)))
    result = tuple(sorted(groups))
    _require(sorted(itertools.chain.from_iterable(result)) == list(range(e)),
             "partition coverage")
    return result


def _fold_partition(labels: np.ndarray, train_mask: np.ndarray,
                    group_size: int) -> tuple[tuple[int, ...], ...]:
    return greedy_equal_partition(pairwise_information_scores(labels[:, :, train_mask]), group_size)


def crossfit_group_size(labels: np.ndarray, group_size: int,
                        fold_count: int = FOLD_COUNT,
                        superblock_values: int = SUPERBLOCK_VALUES) -> dict:
    q, e, n = _labels(labels)
    _require(group_size in compatible_group_sizes(e), "incompatible frozen group size")
    folds = fold_ids(n, fold_count, superblock_values)
    baseline_data = 0.0
    latent_data = 0.0
    private_data = np.zeros(e, dtype=np.float64)
    baseline_model_bits = 0
    conditional_model_bits = np.zeros(e, dtype=np.int64)
    latent_model_bits_by_segment: list[int] = []
    common_data_bits_by_segment: list[float] = []
    segment_members: list[tuple[int, ...]] = []
    partitions = []
    fold_evidence = []

    for fold in range(fold_count):
        test_mask = folds == fold
        train_mask = ~test_mask
        train_n = int(train_mask.sum())
        test_n = int(test_mask.sum())
        _require(train_n > 0 and test_n > 0, "cross-fit split")
        partition = _fold_partition(q, train_mask, group_size)
        partitions.append(partition)
        fold_baseline = 0.0
        fold_latent = 0.0
        for expert in range(e):
            for role in range(ROLES):
                counts = marginal_train_counts(q[expert:expert + 1, role, train_mask])
                bits, _ = marginal_nll_bits(q[expert:expert + 1, role, test_mask], counts)
                baseline_data += bits
                fold_baseline += bits
                baseline_model_bits += marginal_model_descriptor_bits(train_n)
        for group in partition:
            segment_latent_bits = 0.0
            for role in range(ROLES):
                train = q[np.asarray(group), role][:, train_mask]
                test = q[np.asarray(group), role][:, test_mask]
                model = fit_binary_product_model(train)
                scored = evaluate_binary_model(
                    test, model.latent_counts, model.conditional_counts
                )
                segment_latent_bits += float(scored["latent_bits"])
                for local, expert in enumerate(group):
                    private_data[expert] += float(scored["private_bits"][local])
                    # Conditional count tables live with the routed private expert.
                    conditional_model_bits[expert] += sum(
                        3 * ceil_log2_states(int(state_n) + 1)
                        for state_n in model.latent_counts
                    )
            latent_data += segment_latent_bits
            fold_latent += segment_latent_bits
            # Only latent-count descriptors are common; expert conditional rows
            # are separated into the private streams above.
            latent_model_bits_by_segment.append(sum(
                ceil_log2_states(train_n + 1) for _ in range(ROLES)
            ))
            common_data_bits_by_segment.append(segment_latent_bits)
            segment_members.append(group)
        fold_evidence.append({
            "fold": fold,
            "train_coordinates": train_n,
            "test_coordinates": test_n,
            "partition": [list(group) for group in partition],
            "baseline_data_bits": fold_baseline,
            "latent_data_bits": fold_latent,
        })

    private_total = float(private_data.sum())
    weights = e * ROLES * n
    partition_bits = fold_count * partition_descriptor_bits(e, group_size)
    selector_bits = selector_bits_for_group_bank(e)
    common_model_bits = int(sum(latent_model_bits_by_segment))
    structured_model_bits = common_model_bits + int(conditional_model_bits.sum())
    baseline_framing = e * PRIVATE_HEADER_BYTES * 8
    structured_framing = (
        GLOBAL_HEADER_BYTES * 8
        + len(segment_members) * GROUP_HEADER_BYTES * 8
        + e * PRIVATE_HEADER_BYTES * 8
    )
    baseline_charged = baseline_data + baseline_model_bits + baseline_framing
    structured_charged = (
        private_total + latent_data + structured_model_bits + partition_bits
        + selector_bits + structured_framing
    )
    return {
        "group_size": group_size,
        "expert_count": e,
        "roles": ROLES,
        "coordinates_per_role": n,
        "source_weights": weights,
        "fold_count": fold_count,
        "superblock_values": superblock_values,
        "baseline_data_bits": baseline_data,
        "private_conditional_data_bits": private_total,
        "latent_data_bits": latent_data,
        "favorable_gross_gain_bpw": (baseline_data - private_total) / weights,
        "net_ideal_gain_bpw": (baseline_data - private_total - latent_data) / weights,
        "baseline_model_bits": baseline_model_bits,
        "latent_model_bits": common_model_bits,
        "conditional_model_bits": int(conditional_model_bits.sum()),
        "partition_bits": partition_bits,
        "selector_bits": selector_bits,
        "baseline_framing_bits": baseline_framing,
        "structured_framing_bits": structured_framing,
        "baseline_charged_bits": baseline_charged,
        "structured_charged_bits": structured_charged,
        "charged_gain_bpw": (baseline_charged - structured_charged) / weights,
        "private_data_bits_by_expert": private_data.tolist(),
        "private_model_bits_by_expert": conditional_model_bits.astype(int).tolist(),
        "common_data_bits_by_segment": common_data_bits_by_segment,
        "common_model_bits_by_segment": latent_model_bits_by_segment,
        "segment_members": [list(group) for group in segment_members],
        "fold_evidence": fold_evidence,
    }


def packet_requirements(score: Mapping, scale_bytes_per_expert: int) -> dict:
    e = int(score["expert_count"])
    _require(isinstance(scale_bytes_per_expert, int) and scale_bytes_per_expert >= 0,
             "scale bytes")
    global_bits = int(score["partition_bits"]) + int(score["selector_bits"])
    global_bytes = GLOBAL_HEADER_BYTES + ceil_div(global_bits, 8)
    common = []
    for members, data, model in zip(
        score["segment_members"],
        score["common_data_bits_by_segment"],
        score["common_model_bits_by_segment"],
    ):
        common.append({
            "members": [int(x) for x in members],
            "required_bytes": GROUP_HEADER_BYTES + ceil_div(int(math.ceil(data)) + int(model), 8),
        })
    private = [
        PRIVATE_HEADER_BYTES + scale_bytes_per_expert
        + ceil_div(int(math.ceil(data)) + int(model), 8)
        for data, model in zip(
            score["private_data_bits_by_expert"], score["private_model_bits_by_expert"]
        )
    ]
    _require(len(private) == e, "private requirement rows")
    return {"global_required_bytes": global_bytes,
            "common_segments": common,
            "private_required_bytes": private}


def physical_read_envelope(*, expert_count: int, weights_per_expert: int,
                           requested_rate: Fraction, global_required_bytes: int,
                           common_segments: Sequence[Mapping],
                           private_required_bytes: Sequence[int]) -> dict:
    """Exact page/read ledger for global + flat group path + private expert."""
    _require(2 <= expert_count <= 256 and weights_per_expert > 0, "read geometry")
    _require(isinstance(requested_rate, Fraction) and RATE_MIN <= requested_rate <= RATE_MAX,
             "requested rate")
    _require(isinstance(global_required_bytes, int) and global_required_bytes >= 0,
             "global bytes")
    _require(len(private_required_bytes) == expert_count and
             all(isinstance(x, int) and x >= 0 for x in private_required_bytes),
             "private bytes")
    normalized = []
    membership = [[] for _ in range(expert_count)]
    for index, row in enumerate(common_segments):
        members = tuple(int(x) for x in row["members"])
        required = int(row["required_bytes"])
        _require(len(members) in GROUP_SIZES and len(set(members)) == len(members),
                 "flat common group")
        _require(all(0 <= x < expert_count for x in members) and required >= 0,
                 "common segment values")
        pages = ceil_div(required, PAGE_BYTES)
        normalized.append((members, required, pages))
        for expert in members:
            membership[expert].append(index)

    total_weights = expert_count * weights_per_expert
    total_pages = ceil_div(
        total_weights * requested_rate.numerator,
        requested_rate.denominator * 8 * PAGE_BYTES,
    )
    actual_rate = Fraction(total_pages * PAGE_BYTES * 8, total_weights)
    global_pages = ceil_div(global_required_bytes, PAGE_BYTES)
    common_pages_total = sum(row[2] for row in normalized)
    minimum_private_pages = [ceil_div(int(value), PAGE_BYTES) for value in private_required_bytes]
    minimum_pages = global_pages + common_pages_total + sum(minimum_private_pages)
    base = {
        "requested_rate": str(requested_rate),
        "actual_rate_fraction": str(actual_rate),
        "actual_rate_bpw": float(actual_rate),
        "total_pages": total_pages,
        "minimum_required_pages": minimum_pages,
        "global_required_bytes": global_required_bytes,
        "global_pages": global_pages,
        "common_segment_pages": [row[2] for row in normalized],
    }
    if actual_rate > RATE_MAX:
        return {**base, "status": "FAIL_PAGE_ROUNDING_EXCEEDS_RATE_CAP",
                "capacity_ok": False, "strictly_below_2x": False}
    if minimum_pages > total_pages:
        return {**base, "status": "FAIL_PACKET_EXCEEDS_RATE_CAP",
                "capacity_ok": False, "strictly_below_2x": False}

    private_pages = minimum_private_pages[:]
    remaining = total_pages - minimum_pages
    quotient, remainder = divmod(remaining, expert_count)
    private_pages = [pages + quotient + (1 if i < remainder else 0)
                     for i, pages in enumerate(private_pages)]
    global_padded = global_pages * PAGE_BYTES
    amp_physical = []
    amp_nonpadding = []
    touched_bytes = []
    owned_physical = []
    owned_nonpadding = []
    strict = True
    for expert in range(expert_count):
        selected = [normalized[index] for index in membership[expert]]
        touched = global_padded + private_pages[expert] * PAGE_BYTES + sum(
            row[2] * PAGE_BYTES for row in selected
        )
        physical = (
            Fraction(global_padded, expert_count)
            + private_pages[expert] * PAGE_BYTES
            + sum(Fraction(row[2] * PAGE_BYTES, len(row[0])) for row in selected)
        )
        nonpadding = (
            Fraction(global_required_bytes, expert_count)
            + int(private_required_bytes[expert])
            + sum(Fraction(row[1], len(row[0])) for row in selected)
        )
        _require(physical > 0 and nonpadding > 0, "read denominator")
        ap = Fraction(touched, 1) / physical
        an = Fraction(touched, 1) / nonpadding
        touched_bytes.append(touched)
        owned_physical.append(str(physical))
        owned_nonpadding.append(str(nonpadding))
        amp_physical.append(ap)
        amp_nonpadding.append(an)
        strict = strict and ap < 2 and an < 2
    worst = max(amp_physical + amp_nonpadding)
    return {
        **base,
        "status": ("IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC" if strict
                   else "FAIL_STRICT_READ_AMPLIFICATION"),
        "capacity_ok": True,
        "strictly_below_2x": strict,
        "private_pages": private_pages,
        "touched_bytes": touched_bytes,
        "owned_physical_bytes": owned_physical,
        "owned_nonpadding_bytes": owned_nonpadding,
        "amplification_physical_fraction": [str(x) for x in amp_physical],
        "amplification_nonpadding_fraction": [str(x) for x in amp_nonpadding],
        "max_amplification": float(worst),
    }


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    z = (int(value) + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def affine_permutation_parameters(n: int, seed: int, expert: int,
                                  role: int) -> tuple[int, int]:
    _require(isinstance(n, int) and n > 1, "permutation length")
    material = int(seed) ^ (int(expert) << 24) ^ (int(role) << 56)
    a = int(_splitmix64(material) % n) or 1
    while math.gcd(a, n) != 1:
        a = (a + 1) % n or 1
    b = int(_splitmix64(material ^ 0xD1B54A32D192ED03) % n)
    return a, b


def marginal_preserving_control(labels: np.ndarray, seed: int) -> np.ndarray:
    q, e, n = _labels(labels)
    base = np.arange(n, dtype=np.int64)
    out = np.empty_like(q)
    for expert in range(e):
        for role in range(ROLES):
            a, b = affine_permutation_parameters(n, seed, expert, role)
            out[expert, role] = q[expert, role, (a * base + b) % n]
    return out


def score_source_gate(labels: np.ndarray, scale_bytes_per_expert: int,
                      fold_count: int = FOLD_COUNT,
                      superblock_values: int = SUPERBLOCK_VALUES,
                      run_controls: bool = True) -> dict:
    """Evaluate source first; controls are unreachable before source survival."""
    q, e, n = _labels(labels)
    sizes = compatible_group_sizes(e)
    _require(bool(sizes), "no compatible group size")
    source = []
    for group_size in sizes:
        score = crossfit_group_size(q, group_size, fold_count, superblock_values)
        requirements = packet_requirements(score, scale_bytes_per_expert)
        envelopes = {
            str(rate): physical_read_envelope(
                expert_count=e,
                weights_per_expert=ROLES * n,
                requested_rate=rate,
                **requirements,
            )
            for rate in RATE_ENDPOINTS
        }
        feasible = [rate for rate, envelope in envelopes.items()
                    if envelope.get("capacity_ok") is True
                    and envelope.get("strictly_below_2x") is True
                    and envelope.get("status") == "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC"]
        row = dict(score)
        row["read_envelopes"] = envelopes
        row["feasible_rate_endpoints"] = feasible
        source.append(row)

    favorable_survivors = [row for row in source
                           if row["favorable_gross_gain_bpw"] >= TARGET_GAIN_BPW]
    source_survivors = [row for row in favorable_survivors
                        if row["feasible_rate_endpoints"]]
    result = {
        "schema": "same_layer_clustered_ib_entropy_gate_result_v0",
        "target_gain_bpw_on_up_down": TARGET_GAIN_BPW,
        "source_scores": source,
        "controls_executed": False,
        "controls": [],
        "eligible_for_finite_codec": False,
    }
    if not favorable_survivors:
        result["status"] = "HARD_KILL_FAVORABLE_BELOW_TARGET"
        return result
    if not source_survivors:
        result["status"] = "HOLD_NO_STRICT_READ_FEASIBLE_RATE"
        return result
    if not run_controls:
        result["status"] = "HOLD_CONTROLS_REQUIRED_AFTER_SOURCE_SURVIVAL"
        return result

    candidate_sizes = {int(row["group_size"]) for row in source_survivors}
    controls = []
    for seed in CONTROL_SEEDS:
        controlled = marginal_preserving_control(q, seed)
        rows = [crossfit_group_size(controlled, size, fold_count, superblock_values)
                for size in sorted(candidate_sizes)]
        controls.append({"seed": seed, "scores": rows})
    result["controls_executed"] = True
    result["controls"] = controls
    promoted = []
    for row in source_survivors:
        size = int(row["group_size"])
        maximum_control = max(
            score["charged_gain_bpw"]
            for control in controls for score in control["scores"]
            if int(score["group_size"]) == size
        )
        corrected = float(row["charged_gain_bpw"]) - max(0.0, float(maximum_control))
        row["maximum_control_charged_gain_bpw"] = maximum_control
        row["control_corrected_charged_gain_bpw"] = corrected
        if corrected >= TARGET_GAIN_BPW:
            promoted.append(row)
    if promoted:
        result["status"] = "SURVIVE_SOURCE_ONLY_REQUIRES_FINITE_CODEC"
        result["eligible_for_finite_codec"] = True
        result["promoted_group_sizes"] = [row["group_size"] for row in promoted]
    else:
        result["status"] = "HARD_KILL_CHARGED_OR_CONTROLS_BELOW_TARGET"
    return result
