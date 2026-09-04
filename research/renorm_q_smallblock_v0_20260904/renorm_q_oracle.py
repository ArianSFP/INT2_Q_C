"""Bounded source-only RENORM-Q collective-variable oracle.

This module deliberately has no model, filesystem-payload, network, GPU, or
deployment entry point.  It accepts caller-supplied discrete labels or FP64
distortion fields.  Source-derived probability laws are allowed only as a
favourable kill-only oracle; they do not constitute a finite codec.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Sequence

import numpy as np


HARD_KILL_CONTROL_CORRECTED_BPW = 0.03
MEMORYLESS_FOLLOWUP_BPW = 0.045
TARGET_MSE_REDUCTION = 0.1909952569401769
MAX_SITES = 6
MAX_ALPHABET = 4


class OracleError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def tuple_table(sites: int, alphabet: int) -> np.ndarray:
    require(1 <= int(sites) <= MAX_SITES, "sites must be in 1..6")
    require(2 <= int(alphabet) <= MAX_ALPHABET, "alphabet must be in 2..4")
    return np.asarray(list(itertools.product(range(alphabet), repeat=sites)),
                      dtype=np.uint8)


def tuple_ids(labels: np.ndarray, alphabet: int) -> np.ndarray:
    q = np.asarray(labels)
    require(q.ndim == 2 and 1 <= q.shape[1] <= MAX_SITES and
            np.issubdtype(q.dtype, np.integer) and
            bool(np.all((q >= 0) & (q < alphabet))),
            "labels must be integer [N,sites] in the declared alphabet")
    powers = np.power(int(alphabet), np.arange(q.shape[1] - 1, -1, -1),
                      dtype=np.int64)
    return (q.astype(np.int64) @ powers).astype(np.int64)


@dataclass(frozen=True)
class MapSpec:
    """A public, source-independent map from one tiny label cell to z."""

    name: str
    outputs: np.ndarray
    cardinality: int
    descriptor_bits: int
    semantic: str

    def validate(self, tuple_count: int) -> None:
        out = np.asarray(self.outputs)
        require(out.dtype == np.uint8 and out.shape == (tuple_count,),
                f"{self.name}: canonical uint8 outputs")
        require(1 < int(self.cardinality) <= 16 and
                bool(np.all(out < self.cardinality)),
                f"{self.name}: cardinality")
        require(int(self.descriptor_bits) >= 0, f"{self.name}: descriptor")


def public_map_bank(sites: int = 4, alphabet: int = 4) -> tuple[MapSpec, ...]:
    """Return a frozen small map bank requiring no learned neural decoder.

    The first four sites have the conventional interpretation
    ``[role0/expert0, role1/expert0, role0/expert1, role1/expert1]``.
    Maps remain well-defined for other tiny cells.
    """
    table = tuple_table(sites, alphabet)
    gray = table ^ (table >> 1)
    low = gray & 1
    high = (gray >> 1) & 1
    descriptor = max(1, math.ceil(math.log2(9)))
    specs = [
        MapSpec("gray_low_parity", np.bitwise_xor.reduce(low, axis=1), 2,
                descriptor, "XOR of the low Gray bit over the cell"),
        MapSpec("gray_high_parity", np.bitwise_xor.reduce(high, axis=1), 2,
                descriptor, "XOR of the high Gray bit over the cell"),
        MapSpec("gray_two_parity",
                (np.bitwise_xor.reduce(low, axis=1) +
                 2 * np.bitwise_xor.reduce(high, axis=1)).astype(np.uint8),
                4, descriptor, "two Gray-bit cell parities"),
        MapSpec("sum_mod_alphabet",
                (np.sum(table, axis=1) % alphabet).astype(np.uint8), alphabet,
                descriptor, "quaternary/binary modular count"),
        MapSpec("nonzero_count_mod_alphabet",
                (np.count_nonzero(table, axis=1) % alphabet).astype(np.uint8),
                alphabet, descriptor, "modular support count"),
    ]
    if sites >= 4:
        specs.extend([
            MapSpec("within_expert_low_syndrome",
                    ((low[:, 0] ^ low[:, 1]) +
                     2 * (low[:, 2] ^ low[:, 3])).astype(np.uint8),
                    4, descriptor,
                    "separate low-bit role syndrome for two experts"),
            MapSpec("cross_expert_low_syndrome",
                    ((low[:, 0] ^ low[:, 2]) +
                     2 * (low[:, 1] ^ low[:, 3])).astype(np.uint8),
                    4, descriptor,
                    "same-role cross-expert low-bit syndrome"),
            MapSpec("role_expert_cube",
                    ((low[:, 0] ^ low[:, 1] ^ low[:, 2] ^ low[:, 3]) +
                     2 * (high[:, 0] ^ high[:, 1] ^ high[:, 2] ^ high[:, 3])
                     ).astype(np.uint8),
                    4, descriptor, "two-bit role/expert cube parity"),
        ])
    result = tuple(specs)
    for spec in result:
        spec.validate(table.shape[0])
    require(len({s.name for s in result}) == len(result), "unique map names")
    return result


def _entropy_from_ids(ids: np.ndarray, cardinality: int) -> float:
    counts = np.bincount(np.asarray(ids, dtype=np.int64), minlength=cardinality)
    total = int(counts.sum())
    require(total > 0, "nonempty symbols")
    p = counts[counts > 0].astype(np.float64) / float(total)
    return float(-np.sum(p * np.log2(p), dtype=np.float64))


def mutual_information(x: np.ndarray, x_cardinality: int,
                       y: np.ndarray, y_cardinality: int) -> float:
    a, b = np.asarray(x), np.asarray(y)
    require(a.ndim == b.ndim == 1 and a.shape == b.shape and a.size > 0 and
            np.issubdtype(a.dtype, np.integer) and
            np.issubdtype(b.dtype, np.integer) and
            bool(np.all((a >= 0) & (a < x_cardinality))) and
            bool(np.all((b >= 0) & (b < y_cardinality))), "MI symbols")
    joint = a.astype(np.int64) * int(y_cardinality) + b.astype(np.int64)
    return (_entropy_from_ids(a, x_cardinality) +
            _entropy_from_ids(b, y_cardinality) -
            _entropy_from_ids(joint, x_cardinality * y_cardinality))


def collective_variable_census(block_labels: np.ndarray,
                               environment_labels: np.ndarray,
                               alphabet: int,
                               beta: float = 0.25,
                               map_bank: Sequence[MapSpec] | None = None,
                               charge_descriptor: bool = True) -> list[dict]:
    """RSMI-style census ``I(z;E)-beta H(z)-B(f)/N``.

    Environment tuples are treated as categorical symbols.  The census is a
    discovery diagnostic only: held-out probability coding and a literal
    packet are still required before any compression claim.
    """
    q, env = np.asarray(block_labels), np.asarray(environment_labels)
    require(q.ndim == 2 and env.ndim == 2 and q.shape[0] == env.shape[0] and
            q.shape[0] > 0 and np.issubdtype(q.dtype, np.integer) and
            np.issubdtype(env.dtype, np.integer) and
            bool(np.all((q >= 0) & (q < alphabet))) and
            bool(np.all((env >= 0) & (env < alphabet))), "census geometry")
    require(math.isfinite(float(beta)) and beta >= 0, "nonnegative beta")
    bank = tuple(map_bank) if map_bank is not None else public_map_bank(q.shape[1], alphabet)
    table_count = alphabet ** q.shape[1]
    ids = tuple_ids(q, alphabet)
    env_ids = tuple_ids(env, alphabet)
    env_cardinality = alphabet ** env.shape[1]
    rows = []
    for spec in bank:
        spec.validate(table_count)
        z = spec.outputs[ids]
        hz = _entropy_from_ids(z, spec.cardinality)
        mi = mutual_information(z, spec.cardinality, env_ids, env_cardinality)
        mi_bpw = mi / float(q.shape[1])
        z_entropy_bpw = hz / float(q.shape[1])
        descriptor_bpw = ((spec.descriptor_bits / float(q.size))
                          if charge_descriptor else 0.0)
        score_bpw = mi_bpw - float(beta) * z_entropy_bpw - descriptor_bpw
        rows.append({
            "map": spec.name,
            "mutual_information_bits_per_cell": mi,
            "mutual_information_bpw": mi_bpw,
            "z_entropy_bits_per_cell": hz,
            "z_entropy_bpw": z_entropy_bpw,
            "descriptor_bpw": descriptor_bpw,
            "rsmi_score_bpw": score_bpw,
            "cardinality": spec.cardinality,
        })
    return sorted(rows, key=lambda r: (-r["rsmi_score_bpw"], r["map"]))


def uniform_fiber_leaf_nll(spec: MapSpec, tuple_count: int) -> np.ndarray:
    """Exact conditional NLL for a uniform code inside every map fibre."""
    spec.validate(tuple_count)
    nll = np.full((spec.cardinality, tuple_count), np.inf, dtype=np.float64)
    for z in range(spec.cardinality):
        members = np.flatnonzero(spec.outputs == z)
        require(members.size > 0, f"{spec.name}: nonempty fibre {z}")
        nll[z, members] = math.log2(float(members.size))
    return nll


@dataclass
class HierarchyResult:
    objective: float
    distortion: float
    modeled_bits: float
    tuple_ids: np.ndarray
    leaf_states: np.ndarray
    state_levels: tuple[np.ndarray, ...]


def validate_distortion_fields(costs: np.ndarray, sites: int,
                               alphabet: int) -> np.ndarray:
    c = np.asarray(costs)
    require(c.dtype == np.float64 and c.ndim == 3 and
            c.shape[1:] == (sites, alphabet) and c.shape[0] > 0 and
            (c.shape[0] & (c.shape[0] - 1)) == 0 and
            bool(np.all(np.isfinite(c))) and bool(np.all(c >= 0)),
            "costs must be finite nonnegative FP64 [power_of_two_leaves,sites,alphabet]")
    return np.ascontiguousarray(c)


def expanded_cell_costs(costs: np.ndarray, table: np.ndarray) -> np.ndarray:
    leaves, sites, _ = costs.shape
    out = np.zeros((leaves, table.shape[0]), dtype=np.float64)
    rows = np.arange(leaves, dtype=np.int64)[:, None]
    for site in range(sites):
        out += costs[rows, site, table[None, :, site]]
    return out


def exact_tree_min_sum(costs: np.ndarray, alphabet: int, spec: MapSpec,
                       root_nll: np.ndarray, transition_nll: np.ndarray,
                       leaf_nll: np.ndarray, lambda_rate: float) -> HierarchyResult:
    """Exact tree contraction for flexible labels and collective states.

    ``transition_nll[level,parent,child]`` uses level zero immediately above
    leaves.  The same transition is used for both children, keeping the model
    intentionally tiny and decoder-shareable.
    """
    sites = int(np.asarray(costs).shape[1])
    c = validate_distortion_fields(costs, sites, alphabet)
    table = tuple_table(sites, alphabet)
    tuples = table.shape[0]
    spec.validate(tuples)
    leaves = c.shape[0]
    depth = int(math.log2(leaves))
    r = np.asarray(root_nll)
    tr = np.asarray(transition_nll)
    ln = np.asarray(leaf_nll)
    require(r.dtype == tr.dtype == ln.dtype == np.float64 and
            r.shape == (spec.cardinality,) and
            tr.shape == (depth, spec.cardinality, spec.cardinality) and
            ln.shape == (spec.cardinality, tuples) and
            bool(np.all(np.isfinite(r))) and bool(np.all(r >= 0)) and
            bool(np.all((np.isfinite(tr) | np.isposinf(tr)))) and
            bool(np.all(tr >= 0)) and
            bool(np.all((np.isfinite(ln) | np.isposinf(ln)))) and
            bool(np.all(ln >= 0)), "canonical nonnegative NLL arrays")
    require(math.isfinite(float(lambda_rate)) and lambda_rate >= 0,
            "nonnegative lambda")
    expanded = expanded_cell_costs(c, table)
    messages = np.full((leaves, spec.cardinality), np.inf, dtype=np.float64)
    leaf_choice = np.full((leaves, spec.cardinality), -1, dtype=np.int64)
    for leaf in range(leaves):
        objective = expanded[leaf][None, :] + float(lambda_rate) * ln
        for z in range(spec.cardinality):
            members = np.flatnonzero(spec.outputs == z)
            local = objective[z, members]
            best_local = int(np.argmin(local))
            messages[leaf, z] = local[best_local]
            leaf_choice[leaf, z] = int(members[best_local])
    backs: list[list[np.ndarray]] = []
    current = messages
    for level in range(depth):
        parents = current.shape[0] // 2
        nxt = np.empty((parents, spec.cardinality), dtype=np.float64)
        level_backs: list[np.ndarray] = []
        for node in range(parents):
            back = np.empty((spec.cardinality, 2), dtype=np.int64)
            for parent_state in range(spec.cardinality):
                left_scores = current[2 * node] + float(lambda_rate) * tr[level, parent_state]
                right_scores = current[2 * node + 1] + float(lambda_rate) * tr[level, parent_state]
                zl, zr = int(np.argmin(left_scores)), int(np.argmin(right_scores))
                back[parent_state] = (zl, zr)
                nxt[node, parent_state] = left_scores[zl] + right_scores[zr]
            level_backs.append(back)
        backs.append(level_backs)
        current = nxt
    root_scores = current[0] + float(lambda_rate) * r
    root_state = int(np.argmin(root_scores))
    state_levels: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(depth + 1)]
    state_levels[depth] = np.asarray([root_state], dtype=np.int64)
    for level in range(depth - 1, -1, -1):
        child_states = np.empty(2 * state_levels[level + 1].size, dtype=np.int64)
        for node, parent_state in enumerate(state_levels[level + 1]):
            child_states[2 * node:2 * node + 2] = backs[level][node][int(parent_state)]
        state_levels[level] = child_states
    leaf_states = state_levels[0]
    chosen_ids = np.asarray([leaf_choice[i, leaf_states[i]] for i in range(leaves)],
                            dtype=np.int64)
    distortion = float(expanded[np.arange(leaves), chosen_ids].sum(dtype=np.float64))
    modeled_bits = float(r[root_state])
    for level in range(depth):
        child = state_levels[level]
        parent = state_levels[level + 1]
        for node, parent_state in enumerate(parent):
            modeled_bits += float(tr[level, parent_state, child[2 * node]])
            modeled_bits += float(tr[level, parent_state, child[2 * node + 1]])
    modeled_bits += float(ln[leaf_states, chosen_ids].sum(dtype=np.float64))
    objective = distortion + float(lambda_rate) * modeled_bits
    require(abs(objective - float(root_scores[root_state])) <=
            1e-10 * max(1.0, abs(objective)), "DP accounting closure")
    return HierarchyResult(objective, distortion, modeled_bits, chosen_ids,
                           leaf_states.copy(), tuple(x.copy() for x in state_levels))


def brute_force_tree(costs: np.ndarray, alphabet: int, spec: MapSpec,
                     root_nll: np.ndarray, transition_nll: np.ndarray,
                     leaf_nll: np.ndarray, lambda_rate: float,
                     max_assignments: int = 2_000_000) -> HierarchyResult:
    """Independent exhaustive reference for tiny KATs only."""
    sites = int(np.asarray(costs).shape[1])
    c = validate_distortion_fields(costs, sites, alphabet)
    table = tuple_table(sites, alphabet)
    tuples, leaves = table.shape[0], c.shape[0]
    require(tuples ** leaves <= max_assignments, "brute-force cap")
    spec.validate(tuples)
    expanded = expanded_cell_costs(c, table)
    best: tuple[float, float, float, tuple[int, ...], list[np.ndarray]] | None = None
    for assignment in itertools.product(range(tuples), repeat=leaves):
        ids = np.asarray(assignment, dtype=np.int64)
        states = [spec.outputs[ids].astype(np.int64)]
        # Parent state is also optimized; enumerate each internal level.
        internal_counts = [leaves // (2 ** (level + 1)) for level in range(int(math.log2(leaves)))]
        possibilities = int(spec.cardinality) ** sum(internal_counts)
        require(possibilities * tuples ** leaves <= max_assignments * 64,
                "internal brute-force cap")
        for flat_internal in itertools.product(range(spec.cardinality),
                                               repeat=sum(internal_counts)):
            levels = [states[0]]
            offset = 0
            for count in internal_counts:
                levels.append(np.asarray(flat_internal[offset:offset + count], dtype=np.int64))
                offset += count
            distortion = float(expanded[np.arange(leaves), ids].sum(dtype=np.float64))
            bits = float(root_nll[levels[-1][0]]) + float(leaf_nll[levels[0], ids].sum())
            for level in range(len(internal_counts)):
                for node, parent in enumerate(levels[level + 1]):
                    bits += float(transition_nll[level, parent, levels[level][2 * node]])
                    bits += float(transition_nll[level, parent, levels[level][2 * node + 1]])
            objective = distortion + float(lambda_rate) * bits
            key = (objective, tuple(assignment), tuple(flat_internal))
            if best is None or key < (best[0], best[3], tuple(np.concatenate(best[4][1:]))):
                best = (objective, distortion, bits, tuple(assignment), levels)
    require(best is not None, "brute-force result")
    return HierarchyResult(best[0], best[1], best[2],
                           np.asarray(best[3], dtype=np.int64), best[4][0].copy(),
                           tuple(x.copy() for x in best[4]))


def logical_common_private_read_amplification(experts: int, common_bits: float,
                                              private_bits_per_expert: float) -> float:
    """Ideal logical route-read / equal ownership; excludes page rounding."""
    require(int(experts) >= 2 and math.isfinite(common_bits) and common_bits >= 0 and
            math.isfinite(private_bits_per_expert) and private_bits_per_expert > 0,
            "read ledger inputs")
    fair_share = private_bits_per_expert + common_bits / int(experts)
    return float((private_bits_per_expert + common_bits) / fair_share)


def kill_decision(qwen_gain_bpw: float, matched_control_gain_bpw: float,
                  lower_confidence_bound_bpw: float) -> str:
    """Frozen decision rule for a future held-out payload capability."""
    values = (qwen_gain_bpw, matched_control_gain_bpw, lower_confidence_bound_bpw)
    require(all(math.isfinite(float(v)) for v in values), "finite gains")
    corrected = float(qwen_gain_bpw) - float(matched_control_gain_bpw)
    if corrected < HARD_KILL_CONTROL_CORRECTED_BPW or lower_confidence_bound_bpw < 0:
        return "HARD_KILL"
    if corrected < MEMORYLESS_FOLLOWUP_BPW:
        return "SCIENTIFIC_SIGNAL_ONLY"
    return "PROMOTE_TO_SEPARATE_FINITE_PROJECTION"
