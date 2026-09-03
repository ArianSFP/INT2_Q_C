"""Source-only mathematics for the same-layer common-label aperture.

This module contains no model paths and performs no filesystem access.  Small
NumPy fixtures are the authority for all integer counts and MDL arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


ALPHABET = 4
BLOCK_VALUES = 2048
PAGE_BYTES = 4096
GLOBAL_HEADER_BYTES = 4096
PRIVATE_HEADER_BYTES = 256
TARGET_GAIN_BPW = 0.22933495044437175
TRIAGE_GAIN_BPW = 0.045
RATE_MIN = Fraction(43, 20)  # 2.15 bpw
RATE_MAX = Fraction(5, 2)    # 2.50 bpw
THRESHOLD_RMS = 0.981598821873
RECONSTRUCTION_RMS = np.asarray(
    (-1.510417608, -0.452780039, 0.452780039, 1.510417608),
    dtype=np.float64,
)
GRAY_PLANES = np.asarray(((0, 0), (0, 1), (1, 1), (1, 0)), dtype=np.uint8)
CONTROL_SEEDS = (
    10619863,
    10619881,
    10619909,
    10619927,
    10619953,
    10619971,
    10619999,
    10620017,
)


class GateError(ValueError):
    """A fail-closed validation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def ceil_div(a: int, b: int) -> int:
    _require(isinstance(a, int) and isinstance(b, int) and a >= 0 and b > 0,
             "ceil_div domain")
    return (a + b - 1) // b


def ceil_log2_positive_states(states: int) -> int:
    _require(isinstance(states, int) and states >= 1, "state count")
    return 0 if states == 1 else (states - 1).bit_length()


def validate_geometry(expert_count: int, d_ff: int, d_model: int) -> None:
    _require(isinstance(expert_count, int) and 2 <= expert_count <= 256,
             "expert_count must be in [2, 256]")
    _require(isinstance(d_ff, int) and d_ff > 0, "d_ff must be positive")
    _require(isinstance(d_model, int) and d_model > 0, "d_model must be positive")
    _require(d_ff * d_model <= (1 << 40), "unreasonable SwiGLU matrix shape")


def canonicalize_role_cpu(raw: np.ndarray, role: str, d_ff: int, d_model: int) -> np.ndarray:
    """Return a C-contiguous [d_ff,d_model] Up or Down.T matrix."""
    a = np.asarray(raw)
    if role == "up":
        _require(a.shape == (d_ff, d_model), "Up shape")
        return np.ascontiguousarray(a)
    _require(role == "down", "role must be up or down")
    _require(a.shape == (d_model, d_ff), "Down shape")
    return np.ascontiguousarray(a.T)


@dataclass(frozen=True)
class QuantizedRole:
    labels: np.ndarray
    scale_u16: np.ndarray
    reconstruction: np.ndarray


def scale_u16_cpu(values: np.ndarray, block_values: int = BLOCK_VALUES) -> np.ndarray:
    """Canonical FP64 RMS -> binary16 scale bits (shared CPU authority)."""
    x = np.asarray(values, dtype=np.float64)
    _require(x.size > 0, "scale source must be nonempty")
    _require(isinstance(block_values, int) and block_values > 0, "block_values")
    flat = np.ascontiguousarray(x).reshape(-1)
    n_blocks = ceil_div(int(flat.size), block_values)
    result = np.empty(n_blocks, dtype=np.uint16)
    for block in range(n_blocks):
        lo = block * block_values
        hi = min(flat.size, lo + block_values)
        segment = flat[lo:hi]
        rms = math.sqrt(float(np.dot(segment, segment)) / int(segment.size))
        scale16 = np.float16(rms)
        _require(np.isfinite(scale16) and float(scale16) > 0.0,
                 "nonfinite or zero decoded scale")
        result[block] = scale16.view(np.uint16)
    return result


def quantize_canonical_cpu(values: np.ndarray, block_values: int = BLOCK_VALUES) -> QuantizedRole:
    """Quantize using a decoded FP16 RMS scale and a fixed four-level rule.

    The binary16 scale is the decoder-visible scale.  Both thresholds and
    reconstructions use that rounded scale, so replay has no hidden FP64 field.
    """
    x = np.asarray(values, dtype=np.float64)
    _require(x.ndim == 2 and x.size > 0, "canonical role must be a nonempty matrix")
    _require(isinstance(block_values, int) and block_values > 0, "block_values")
    flat = np.ascontiguousarray(x).reshape(-1)
    scale_bits = scale_u16_cpu(flat, block_values)
    n_blocks = int(scale_bits.size)
    labels = np.empty(flat.size, dtype=np.uint8)
    recon = np.empty(flat.size, dtype=np.float64)
    for block in range(n_blocks):
        lo = block * block_values
        hi = min(flat.size, lo + block_values)
        segment = flat[lo:hi]
        scale16 = scale_bits[block].view(np.float16)
        scale = float(scale16)
        threshold = THRESHOLD_RMS * scale
        q = np.where(
            segment < -threshold,
            0,
            np.where(segment < 0.0, 1, np.where(segment <= threshold, 2, 3)),
        ).astype(np.uint8)
        labels[lo:hi] = q
        recon[lo:hi] = RECONSTRUCTION_RMS[q] * scale
    return QuantizedRole(
        labels=labels.reshape(x.shape),
        scale_u16=scale_bits,
        reconstruction=recon.reshape(x.shape),
    )


def _validate_labels(labels: np.ndarray) -> tuple[int, int, int]:
    q = np.asarray(labels)
    _require(q.ndim == 3, "labels must have shape [expert, role, coordinate]")
    e, r, n = map(int, q.shape)
    validate_geometry(e, 1, n)
    _require(r == 2 and n > 0, "exactly Up and Down.T roles are required")
    _require(q.dtype == np.uint8, "labels must be uint8")
    _require(bool(np.all(q < ALPHABET)), "labels outside four-level alphabet")
    return e, r, n


def modal_common_latent_cpu(labels: np.ndarray, cardinality: int, plane: int | None = None) -> np.ndarray:
    q = np.asarray(labels, dtype=np.uint8)
    _require(q.ndim == 2, "role labels must have shape [expert, coordinate]")
    _require(cardinality in (2, 4), "latent cardinality")
    if cardinality == 2:
        _require(plane in (0, 1), "binary Gray plane")
        source = GRAY_PLANES[q, int(plane)]
    else:
        _require(plane is None, "quaternary latent has no plane")
        source = q
    counts = np.stack([(source == a).sum(axis=0) for a in range(cardinality)], axis=0)
    # np.argmax breaks ties toward the lowest symbol, which is part of the wire contract.
    return np.argmax(counts, axis=0).astype(np.uint8)


def summarize_counts_cpu(labels: np.ndarray, cardinality: int, planes: Sequence[int] | None = None) -> dict:
    q = np.asarray(labels, dtype=np.uint8)
    e, roles, n = _validate_labels(q)
    _require(cardinality in (2, 4), "latent cardinality")
    if cardinality == 4:
        _require(planes is None, "quaternary planes")
        chosen_planes = (None, None)
    else:
        _require(planes is not None and len(planes) == roles and all(p in (0, 1) for p in planes),
                 "two binary plane selectors required")
        chosen_planes = tuple(int(p) for p in planes)
    marginal = np.zeros((e, roles, ALPHABET), dtype=np.int64)
    latent = np.zeros((roles, cardinality), dtype=np.int64)
    conditional = np.zeros((e, roles, cardinality, ALPHABET), dtype=np.int64)
    for expert in range(e):
        for role in range(roles):
            marginal[expert, role] = np.bincount(q[expert, role], minlength=ALPHABET)
    for role in range(roles):
        u = modal_common_latent_cpu(q[:, role], cardinality, chosen_planes[role])
        latent[role] = np.bincount(u, minlength=cardinality)
        for expert in range(e):
            for state in range(cardinality):
                conditional[expert, role, state] = np.bincount(
                    q[expert, role, u == state], minlength=ALPHABET
                )
    return {
        "expert_count": e,
        "role_count": roles,
        "coordinates_per_role": n,
        "cardinality": cardinality,
        "planes": list(chosen_planes),
        "marginal_counts": marginal,
        "latent_counts": latent,
        "conditional_counts": conditional,
    }


def entropy_bits_from_counts(counts: Iterable[int]) -> float:
    c = np.asarray(tuple(int(x) for x in counts), dtype=np.int64)
    _require(c.ndim == 1 and bool(np.all(c >= 0)), "entropy counts")
    total = int(c.sum())
    if total == 0:
        return 0.0
    nonzero = c[c > 0].astype(np.float64)
    return float(total * math.log2(total) - np.dot(nonzero, np.log2(nonzero)))


def multinomial_count_descriptor_bits(counts: Iterable[int]) -> int:
    """Fixed-width two-part descriptor; final alphabet count is derived."""
    c = tuple(int(x) for x in counts)
    _require(len(c) >= 2 and all(x >= 0 for x in c), "descriptor counts")
    return (len(c) - 1) * ceil_log2_positive_states(sum(c) + 1)


def score_count_summary(summary: Mapping, scale_bits: int = 0) -> dict:
    marginal = np.asarray(summary["marginal_counts"], dtype=np.int64)
    latent = np.asarray(summary["latent_counts"], dtype=np.int64)
    conditional = np.asarray(summary["conditional_counts"], dtype=np.int64)
    e = int(summary["expert_count"])
    roles = int(summary["role_count"])
    n = int(summary["coordinates_per_role"])
    k = int(summary["cardinality"])
    _require(marginal.shape == (e, roles, ALPHABET), "marginal count shape")
    _require(latent.shape == (roles, k), "latent count shape")
    _require(conditional.shape == (e, roles, k, ALPHABET), "conditional count shape")
    _require(bool(np.all(marginal >= 0)) and bool(np.all(latent >= 0)) and
             bool(np.all(conditional >= 0)), "negative count")
    _require(int(marginal.sum()) == e * roles * n, "marginal total")
    _require(int(latent.sum()) == roles * n, "latent total")
    _require(int(conditional.sum()) == e * roles * n, "conditional total")
    _require(bool(np.all(marginal.sum(axis=2) == n)), "per-expert marginal total")
    _require(bool(np.all(latent.sum(axis=1) == n)), "per-role latent total")
    for expert in range(e):
        for role in range(roles):
            _require(bool(np.all(conditional[expert, role].sum(axis=1) == latent[role])),
                     "conditional state total disagrees with latent")
            _require(bool(np.all(conditional[expert, role].sum(axis=0) == marginal[expert, role])),
                     "conditional symbols disagree with marginal")
    _require(isinstance(scale_bits, int) and scale_bits >= 0, "scale_bits")

    marginal_data = sum(entropy_bits_from_counts(row) for row in marginal.reshape(-1, ALPHABET))
    latent_data = sum(entropy_bits_from_counts(row) for row in latent)
    conditional_data = sum(
        entropy_bits_from_counts(row) for row in conditional.reshape(-1, ALPHABET)
    )
    marginal_model = sum(
        multinomial_count_descriptor_bits(row) for row in marginal.reshape(-1, ALPHABET)
    )
    latent_model = sum(multinomial_count_descriptor_bits(row) for row in latent)
    conditional_model = sum(
        multinomial_count_descriptor_bits(row) for row in conditional.reshape(-1, ALPHABET)
    )
    plane_selector_bits = roles if k == 2 else 0
    # Both latent families are evaluated on the source; an eventual packet must
    # identify the selected family.  This bit is charged in either branch.
    family_selector_bits = 1
    selector_bits = plane_selector_bits + family_selector_bits
    weights = e * roles * n
    marginal_two_part = marginal_data + marginal_model + scale_bits
    common_two_part = (
        latent_data + conditional_data + latent_model + conditional_model
        + selector_bits + scale_bits
    )

    per_expert_conditional_bits: list[float] = []
    per_expert_model_bits: list[int] = []
    for expert in range(e):
        per_expert_conditional_bits.append(sum(
            entropy_bits_from_counts(conditional[expert, role, state])
            for role in range(roles) for state in range(k)
        ))
        per_expert_model_bits.append(sum(
            multinomial_count_descriptor_bits(conditional[expert, role, state])
            for role in range(roles) for state in range(k)
        ))

    return {
        "cardinality": k,
        "planes": list(summary.get("planes", (None, None))),
        "source_weights": weights,
        "marginal_data_bits": marginal_data,
        "conditional_data_bits": conditional_data,
        "latent_data_bits": latent_data,
        "favorable_gross_gain_bpw": (marginal_data - conditional_data) / weights,
        "net_ideal_gain_bpw": (marginal_data - conditional_data - latent_data) / weights,
        "marginal_model_bits": marginal_model,
        "latent_model_bits": latent_model,
        "conditional_model_bits": conditional_model,
        "plane_selector_bits": plane_selector_bits,
        "family_selector_bits": family_selector_bits,
        "selector_bits": selector_bits,
        "scale_bits_identical_each_scheme": scale_bits,
        "marginal_two_part_bits": marginal_two_part,
        "common_two_part_bits": common_two_part,
        "two_part_gain_bpw": (marginal_two_part - common_two_part) / weights,
        "per_expert_conditional_data_bits": per_expert_conditional_bits,
        "per_expert_conditional_model_bits": per_expert_model_bits,
        "count_evidence": {
            "marginal_counts": marginal.tolist(),
            "latent_counts": latent.tolist(),
            "conditional_counts": conditional.tolist(),
        },
    }


def score_labels_cpu(
    labels: np.ndarray,
    cardinality: int,
    scale_bits: int = 0,
    selection_objective: str = "charged",
) -> dict:
    if cardinality == 4:
        _require(selection_objective in ("charged", "favorable"), "selection objective")
        return score_count_summary(summarize_counts_cpu(labels, 4), scale_bits)
    _require(cardinality == 2, "latent cardinality")
    # Plane selection is independent by role.  Exhaust four choices; a two-bit
    # selector is charged by score_count_summary.
    candidates = []
    for up_plane in (0, 1):
        for down_plane in (0, 1):
            score = score_count_summary(
                summarize_counts_cpu(labels, 2, (up_plane, down_plane)), scale_bits
            )
            candidates.append(score)
    _require(selection_objective in ("charged", "favorable"), "selection objective")
    objective = (
        "common_two_part_bits" if selection_objective == "charged"
        else "conditional_data_bits"
    )
    selected = min(candidates, key=lambda item: (item[objective], item["planes"]))
    selected = dict(selected)
    selected["binary_plane_selection_objective"] = selection_objective
    selected["binary_plane_candidate_scores"] = [
        {
            "planes": item["planes"],
            "conditional_data_bits": item["conditional_data_bits"],
            "latent_data_bits": item["latent_data_bits"],
            "common_two_part_bits": item["common_two_part_bits"],
            "favorable_gross_gain_bpw": item["favorable_gross_gain_bpw"],
            "two_part_gain_bpw": item["two_part_gain_bpw"],
            "count_evidence": item["count_evidence"],
        }
        for item in candidates
    ]
    return selected


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    z = (int(value) + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def affine_permutation_parameters(n: int, seed: int, expert: int, role: int) -> tuple[int, int]:
    _require(isinstance(n, int) and n > 1, "permutation length")
    material = int(seed) ^ (int(expert) << 24) ^ (int(role) << 56)
    a = int(_splitmix64(material) % n)
    if a == 0:
        a = 1
    while math.gcd(a, n) != 1:
        a = (a + 1) % n
        if a == 0:
            a = 1
    b = int(_splitmix64(material ^ 0xD1B54A32D192ED03) % n)
    return a, b


def coordinate_scramble_cpu(labels: np.ndarray, seed: int) -> np.ndarray:
    q = np.asarray(labels, dtype=np.uint8)
    e, roles, n = _validate_labels(q)
    out = np.empty_like(q)
    base = np.arange(n, dtype=np.int64)
    for expert in range(e):
        for role in range(roles):
            a, b = affine_permutation_parameters(n, seed, expert, role)
            out[expert, role] = q[expert, role, (a * base + b) % n]
    return out


def physical_page_envelope(
    *,
    expert_count: int,
    coordinates_per_role: int,
    latent_bits_per_coordinate: int,
    requested_rate: Fraction,
    common_model_bits: int,
    private_required_bytes: Sequence[int],
) -> dict:
    """Exact page envelope for [global+U][private_i] routed storage.

    This is capacity/read arithmetic, not evidence of an emitted coder.
    """
    validate_geometry(expert_count, 1, coordinates_per_role)
    _require(latent_bits_per_coordinate in (1, 2), "latent width")
    _require(isinstance(requested_rate, Fraction) and RATE_MIN <= requested_rate <= RATE_MAX,
             "requested rate outside [2.15, 2.5]")
    _require(isinstance(common_model_bits, int) and common_model_bits >= 0,
             "common_model_bits")
    _require(len(private_required_bytes) == expert_count and
             all(isinstance(x, int) and x >= 0 for x in private_required_bytes),
             "private byte requirements")

    # Up + Down.T are the scored weights for every expert.
    weights_total = expert_count * 2 * coordinates_per_role
    minimum_bits_num = weights_total * requested_rate.numerator
    minimum_bits_den = requested_rate.denominator
    total_pages = ceil_div(minimum_bits_num, minimum_bits_den * 8 * PAGE_BYTES)
    actual_rate = Fraction(total_pages * PAGE_BYTES * 8, weights_total)

    u_symbols = 2 * coordinates_per_role
    common_bytes_unpadded = (
        GLOBAL_HEADER_BYTES
        + ceil_div(u_symbols * latent_bits_per_coordinate, 8)
        + ceil_div(common_model_bits, 8)
    )
    common_pages = ceil_div(common_bytes_unpadded, PAGE_BYTES)
    common_failure = {
        "requested_rate": str(requested_rate),
        "actual_rate_fraction": str(actual_rate),
        "actual_rate_bpw": float(actual_rate),
        "total_pages": total_pages,
        "common_bytes_unpadded": common_bytes_unpadded,
        "common_pages": common_pages,
        "strictly_below_2x": False,
    }
    if actual_rate > RATE_MAX:
        return {
            **common_failure,
            "status": "FAIL_PAGE_ROUNDING_EXCEEDS_RATE_CAP",
            "failure_reason": "page rounding breaches 2.5 bpw",
        }
    if common_pages + expert_count > total_pages:
        return {
            **common_failure,
            "status": "FAIL_NO_PRIVATE_PAGE_PER_EXPERT",
            "failure_reason": "no page remains for every private expert stream",
        }
    private_pages_total = total_pages - common_pages
    base, extra = divmod(private_pages_total, expert_count)
    private_pages = [base + (1 if i < extra else 0) for i in range(expert_count)]
    capacity_ok = all(
        required <= pages * PAGE_BYTES
        for required, pages in zip(private_required_bytes, private_pages)
    )

    common_bytes = common_pages * PAGE_BYTES
    route_bytes = []
    amortized_denominator_bytes = []
    amplification_physical_fraction = []
    amplification_nonpadding_fraction = []
    strict = capacity_ok
    for required, pages in zip(private_required_bytes, private_pages):
        private_bytes = pages * PAGE_BYTES
        routed = common_bytes + private_bytes
        # Physical owner share: A = (C+P)/(C/E+P) = E(C+P)/(C+EP).
        numerator = expert_count * routed
        denominator_physical = common_bytes + expert_count * private_bytes
        # Do not let page-fill padding make the cold-read metric look better.
        # The second denominator uses only decodable bytes attributable to the
        # owner, while the numerator remains the exact union of touched pages.
        denominator_nonpadding = common_bytes_unpadded + expert_count * required
        route_bytes.append(routed)
        amortized_denominator_bytes.append(Fraction(denominator_physical, expert_count))
        amplification_physical_fraction.append(Fraction(numerator, denominator_physical))
        amplification_nonpadding_fraction.append(Fraction(numerator, denominator_nonpadding))
        strict = (
            strict
            and numerator < 2 * denominator_physical
            and numerator < 2 * denominator_nonpadding
        )
    worst = max(amplification_physical_fraction + amplification_nonpadding_fraction)
    return {
        "status": (
            "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC"
            if strict else "FAIL_CAPACITY_OR_STRICT_READ_AMPLIFICATION"
        ),
        "requested_rate": str(requested_rate),
        "actual_rate_fraction": str(actual_rate),
        "actual_rate_bpw": float(actual_rate),
        "total_pages": total_pages,
        "common_bytes_unpadded": common_bytes_unpadded,
        "common_pages": common_pages,
        "private_pages": private_pages,
        "route_bytes": route_bytes,
        "amortized_denominator_bytes": [str(x) for x in amortized_denominator_bytes],
        "amplification_physical_fraction": [str(x) for x in amplification_physical_fraction],
        "amplification_nonpadding_fraction": [str(x) for x in amplification_nonpadding_fraction],
        "max_amplification": float(worst),
        "capacity_ok": capacity_ok,
        "strictly_below_2x": strict,
    }


def private_byte_requirements(score: Mapping, scale_bytes_per_expert: int) -> list[int]:
    _require(isinstance(scale_bytes_per_expert, int) and scale_bytes_per_expert >= 0,
             "scale_bytes_per_expert")
    data = score["per_expert_conditional_data_bits"]
    model = score["per_expert_conditional_model_bits"]
    _require(len(data) == len(model), "per-expert score rows")
    return [
        PRIVATE_HEADER_BYTES + scale_bytes_per_expert + ceil_div(int(math.ceil(d)) + int(m), 8)
        for d, m in zip(data, model)
    ]
