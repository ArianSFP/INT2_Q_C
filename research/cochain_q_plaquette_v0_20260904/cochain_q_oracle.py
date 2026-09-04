"""Source-only exact COCHAIN-Q plaquette/cube rate-distortion oracle.

The module has no model, filesystem-payload, network, GPU, or deployment entry
point.  Caller-supplied binary-label distortion fields are the only source input.

For one 2^d cell, the highest mixed GF(2) difference is the XOR of all
vertices.  ``(first 2^d-1 labels, syndrome)`` is a bijection, so merely
changing to cochain coordinates cannot save rate.  A real structured quantizer
instead fixes the public syndrome and searches the resulting half-sized fiber.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from typing import Iterable

import numpy as np


SCHEMA = "cochain_q.plaquette_cube_oracle.v0"
SUPPORTED_DIMENSIONS = (2, 3)
HARD_KILL_BPW = 0.045
SCIENTIFICALLY_REAL_BPW = 0.10
STANDALONE_TARGET_BPW = 0.15288996696
ENGINEERING_MARGIN_BPW = 0.18


class OracleError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def vertex_count(dimension: int) -> int:
    require(dimension in SUPPORTED_DIMENSIONS, "dimension must be 2 or 3")
    return 1 << dimension


def all_patterns(dimension: int) -> np.ndarray:
    """Canonical lexicographic binary cell patterns."""
    vertices = vertex_count(dimension)
    return np.asarray(list(itertools.product((0, 1), repeat=vertices)), dtype=np.uint8)


def mixed_syndrome(labels: np.ndarray) -> np.ndarray:
    """Highest mixed difference on each complete 2^d cell."""
    q = np.asarray(labels)
    require(q.ndim == 2 and q.shape[1] in (4, 8) and
            np.issubdtype(q.dtype, np.integer) and
            bool(np.all((q == 0) | (q == 1))), "binary [cells,4 or 8] labels")
    return np.bitwise_xor.reduce(q.astype(np.uint8), axis=1)


def cochain_coordinates(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bijective boundary/fiber coordinates: first V-1 bits and syndrome."""
    q = np.asarray(labels, dtype=np.uint8)
    mixed_syndrome(q)
    return q[:, :-1].copy(), mixed_syndrome(q)


def inverse_cochain_coordinates(boundary: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
    b, s = np.asarray(boundary), np.asarray(syndrome)
    require(b.ndim == 2 and b.shape[1] in (3, 7) and
            np.issubdtype(b.dtype, np.integer) and bool(np.all((b == 0) | (b == 1))) and
            s.shape == (b.shape[0],) and np.issubdtype(s.dtype, np.integer) and
            bool(np.all((s == 0) | (s == 1))), "cochain coordinates")
    last = np.bitwise_xor.reduce(b.astype(np.uint8), axis=1) ^ s.astype(np.uint8)
    return np.column_stack((b.astype(np.uint8), last)).astype(np.uint8)


def fixed_reparameterization_audit(labels: np.ndarray) -> dict:
    """Prove that lossless fixed differencing is a rate-neutral bijection."""
    q = np.asarray(labels, dtype=np.uint8)
    boundary, syndrome = cochain_coordinates(q)
    decoded = inverse_cochain_coordinates(boundary, syndrome)
    require(np.array_equal(decoded, q), "cochain round trip")
    cells, vertices = q.shape
    return {
        "cells": int(cells),
        "vertices_per_cell": int(vertices),
        "raw_bits": int(cells * vertices),
        "cochain_boundary_bits": int(cells * (vertices - 1)),
        "cochain_syndrome_bits": int(cells),
        "cochain_total_bits": int(cells * vertices),
        "saved_bits": 0,
        "bijective": True,
        "labels_sha256": hashlib.sha256(q.tobytes(order="C")).hexdigest(),
        "decoded_sha256": hashlib.sha256(decoded.tobytes(order="C")).hexdigest(),
        "conclusion": "invertible differencing alone cannot reduce joint entropy or rate",
    }


def validate_costs(costs: np.ndarray, source_energy: float, dimension: int) -> np.ndarray:
    c = np.asarray(costs)
    vertices = vertex_count(dimension)
    require(c.dtype == np.float64 and c.ndim == 3 and c.shape[1:] == (vertices, 2) and
            c.shape[0] > 0 and bool(np.all(np.isfinite(c))) and bool(np.all(c >= 0)),
            "finite nonnegative FP64 [cells,vertices,2] costs")
    require(math.isfinite(float(source_energy)) and float(source_energy) > 0,
            "positive finite source energy")
    return np.ascontiguousarray(c)


def pattern_costs(costs: np.ndarray, dimension: int) -> np.ndarray:
    c = np.asarray(costs)
    vertices = vertex_count(dimension)
    require(c.dtype == np.float64 and c.ndim == 3 and c.shape[1:] == (vertices, 2),
            "cost geometry")
    patterns = all_patterns(dimension)
    out = np.zeros((c.shape[0], patterns.shape[0]), dtype=np.float64)
    rows = np.arange(c.shape[0], dtype=np.int64)[:, None]
    for v in range(vertices):
        out += c[rows, v, patterns[None, :, v]]
    return out


def best_labels(costs: np.ndarray, dimension: int, syndrome: int | None = None) -> tuple[np.ndarray, float]:
    """Exact per-cell exhaustive optimum, optionally in one public fiber."""
    c = np.asarray(costs)
    pc = pattern_costs(c, dimension)
    patterns = all_patterns(dimension)
    if syndrome is None:
        legal = np.ones(patterns.shape[0], dtype=bool)
    else:
        require(syndrome in (0, 1), "binary public syndrome")
        legal = mixed_syndrome(patterns) == syndrome
    objective = np.where(legal[None, :], pc, math.inf)
    ids = np.argmin(objective, axis=1)
    labels = patterns[ids]
    sse = float(pc[np.arange(c.shape[0]), ids].sum(dtype=np.float64))
    return labels.astype(np.uint8), sse


def brute_force_global(costs: np.ndarray, dimension: int, syndrome: int | None = None) -> tuple[np.ndarray, float]:
    """Literal global exhaustive search for tiny fixtures (at most 20 sites)."""
    c = np.asarray(costs)
    vertices = vertex_count(dimension)
    sites = int(c.shape[0] * vertices)
    require(sites <= 20, "global brute force is intentionally limited to 20 sites")
    best: tuple[float, tuple[int, ...]] | None = None
    for flat in itertools.product((0, 1), repeat=sites):
        q = np.asarray(flat, dtype=np.uint8).reshape(c.shape[0], vertices)
        if syndrome is not None and not bool(np.all(mixed_syndrome(q) == syndrome)):
            continue
        total = float(sum(c[cell, v, q[cell, v]]
                          for cell in range(c.shape[0]) for v in range(vertices)))
        candidate = (total, flat)
        if best is None or candidate < best:
            best = candidate
    require(best is not None, "nonempty global codebook")
    return np.asarray(best[1], dtype=np.uint8).reshape(c.shape[0], vertices), best[0]


def _pack_bits(bits: np.ndarray) -> bytes:
    x = np.asarray(bits, dtype=np.uint8).reshape(-1)
    require(bool(np.all((x == 0) | (x == 1))), "pack binary")
    return np.packbits(x, bitorder="little").tobytes()


def _unpack_bits(packet: bytes, count: int) -> np.ndarray:
    require(isinstance(packet, bytes) and count >= 0 and len(packet) == (count + 7) // 8,
            "canonical packet length")
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8), bitorder="little")
    require(not bool(np.any(bits[count:])), "noncanonical nonzero padding")
    return bits[:count].astype(np.uint8)


def encode_public_fiber(labels: np.ndarray, public_syndrome: int = 0) -> bytes:
    """Literal fixed-rate expert-local packet: omit the dependent last vertex."""
    q = np.asarray(labels, dtype=np.uint8)
    require(public_syndrome in (0, 1), "public syndrome")
    require(bool(np.all(mixed_syndrome(q) == public_syndrome)), "labels outside public fiber")
    return _pack_bits(q[:, :-1])


def decode_public_fiber(packet: bytes, cells: int, dimension: int,
                        public_syndrome: int = 0) -> np.ndarray:
    vertices = vertex_count(dimension)
    require(isinstance(cells, int) and cells > 0 and public_syndrome in (0, 1),
            "decode geometry")
    boundary = _unpack_bits(packet, cells * (vertices - 1)).reshape(cells, vertices - 1)
    syndrome = np.full(cells, public_syndrome, dtype=np.uint8)
    return inverse_cochain_coordinates(boundary, syndrome)


def physical_ledger(cells: int, dimension: int, page_bytes: int = 4096) -> dict:
    vertices = vertex_count(dimension)
    require(isinstance(cells, int) and cells > 0 and isinstance(page_bytes, int) and page_bytes > 0,
            "ledger geometry")
    raw_bits = cells * vertices
    payload_bits = cells * (vertices - 1)
    payload_bytes = (payload_bits + 7) // 8
    pages = (payload_bytes + page_bytes - 1) // page_bytes
    return {
        "cells": cells,
        "raw_label_bits": raw_bits,
        "public_fiber_payload_bits": payload_bits,
        "public_fiber_packed_bytes": payload_bytes,
        "ideal_rate_bpw": (vertices - 1) / vertices,
        "fixed_map_or_selector_bits": 0,
        "expert_local": True,
        "cross_expert_bytes": 0,
        "logical_routed_read_amplification": 1.0,
        "page_bytes": page_bytes,
        "page_rounded_bytes": pages * page_bytes,
        "page_rounding_over_payload": pages * page_bytes / payload_bytes,
        "warning": "page rounding is packaging overhead, not cross-expert read amplification",
    }


def _equivalent_gain(rate_base: float, distortion_base: float,
                     rate_candidate: float, distortion_candidate: float) -> float:
    require(distortion_base > 0 and distortion_candidate > 0, "positive distortions")
    return ((rate_base - rate_candidate) +
            0.5 * math.log2(distortion_base / distortion_candidate))


@dataclass(frozen=True)
class FiberPoint:
    public_syndrome: int
    ideal_rate_bpw: float
    physical_rate_bpw: float
    relative_mse: float
    ideal_equivalent_gain_bpw: float
    physical_equivalent_gain_bpw: float
    labels_sha256: str
    packet_sha256: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def run_oracle(costs: np.ndarray, source_energy: float, dimension: int,
               public_syndrome: int = 0) -> dict:
    """Run exact public-fiber flexible-label gate and a charged selector diagnostic."""
    c = validate_costs(costs, source_energy, dimension)
    require(public_syndrome in (0, 1), "public syndrome")
    cells, vertices = c.shape[:2]
    baseline, baseline_sse = best_labels(c, dimension)
    d0 = baseline_sse / float(source_energy)
    require(d0 > 0, "positive nearest-label distortion for RD comparison")
    base_rate = 1.0
    points = []
    for syndrome in (0, 1):
        q, sse = best_labels(c, dimension, syndrome)
        packet = encode_public_fiber(q, syndrome)
        decoded = decode_public_fiber(packet, cells, dimension, syndrome)
        require(np.array_equal(q, decoded), "literal public-fiber decode")
        ideal_rate = (vertices - 1) / vertices
        physical_rate = (8 * len(packet)) / (cells * vertices)
        d = sse / float(source_energy)
        points.append(FiberPoint(
            syndrome, ideal_rate, physical_rate, d,
            _equivalent_gain(base_rate, d0, ideal_rate, d),
            _equivalent_gain(base_rate, d0, physical_rate, d),
            hashlib.sha256(q.tobytes(order="C")).hexdigest(),
            hashlib.sha256(packet).hexdigest(),
        ))
    chosen = points[public_syndrome]
    # Optimistic source-adaptive selection is not public: charge its one literal bit.
    selector_candidates = []
    for point in points:
        selected_rate = point.physical_rate_bpw + 1.0 / (cells * vertices)
        selector_candidates.append({
            "syndrome": point.public_syndrome,
            "charged_rate_bpw": selected_rate,
            "charged_equivalent_gain_bpw": _equivalent_gain(
                base_rate, d0, selected_rate, point.relative_mse),
        })
    selector_best = max(selector_candidates,
                        key=lambda x: (x["charged_equivalent_gain_bpw"], -x["syndrome"]))
    gain = chosen.physical_equivalent_gain_bpw
    status = ("HARD_KILL_PUBLIC_COCHAIN_BELOW_0P045_BPW" if gain < HARD_KILL_BPW else
              "SURVIVES_SOURCE_ONLY_REQUIRES_QWEN_CONTROLS_AND_SIX_PLANE_PACKET")
    return {
        "schema": SCHEMA,
        "status": status,
        "source_only": True,
        "dimension": dimension,
        "cells": int(cells),
        "weights_or_bitplane_sites": int(cells * vertices),
        "public_syndrome": public_syndrome,
        "thresholds_bpw": {
            "hard_kill": HARD_KILL_BPW,
            "scientifically_real": SCIENTIFICALLY_REAL_BPW,
            "standalone_target": STANDALONE_TARGET_BPW,
            "engineering_margin": ENGINEERING_MARGIN_BPW,
        },
        "baseline": {
            "rate_bpw": base_rate,
            "relative_mse": d0,
            "labels_sha256": hashlib.sha256(baseline.tobytes(order="C")).hexdigest(),
        },
        "fixed_label_reparameterization": fixed_reparameterization_audit(baseline),
        "public_fiber": chosen.as_dict(),
        "both_public_fibers_diagnostic": [x.as_dict() for x in points],
        "charged_one_bit_source_adaptive_selector_diagnostic": selector_best,
        "physical_ledger": physical_ledger(cells, dimension),
        "scientific_boundary": (
            "Only the fixed public syndrome is promotion-eligible. Choosing a syndrome from "
            "source data requires the charged selector. This binary-cell oracle does not "
            "establish a six-plane STRATA codec or Qwen gain."),
    }


def preference_costs(preferred: np.ndarray, preferred_cost: float = 1.0,
                     alternate_cost: float = 101.0) -> tuple[np.ndarray, float]:
    q = np.asarray(preferred)
    require(q.ndim == 2 and q.shape[1] in (4, 8) and
            np.issubdtype(q.dtype, np.integer) and bool(np.all((q == 0) | (q == 1))) and
            0 < preferred_cost < alternate_cost and math.isfinite(alternate_cost),
            "preference fixture")
    costs = np.full((q.shape[0], q.shape[1], 2), alternate_cost, dtype=np.float64)
    rows = np.arange(q.shape[0])[:, None]
    cols = np.arange(q.shape[1])[None, :]
    costs[rows, cols, q.astype(np.int64)] = preferred_cost
    return costs, float(q.size)


def pairwise_mutual_information(labels: np.ndarray) -> dict[str, float]:
    q = np.asarray(labels, dtype=np.uint8)
    mixed_syndrome(q)
    out = {}
    for a in range(q.shape[1]):
        for b in range(a + 1, q.shape[1]):
            joint = np.bincount((2 * q[:, a] + q[:, b]).astype(np.int64), minlength=4)
            p = joint / joint.sum()
            pa = np.bincount(q[:, a], minlength=2) / q.shape[0]
            pb = np.bincount(q[:, b], minlength=2) / q.shape[0]
            value = 0.0
            for x in range(2):
                for y in range(2):
                    if p[2 * x + y] > 0:
                        value += p[2 * x + y] * math.log2(p[2 * x + y] / (pa[x] * pb[y]))
            out[f"{a}{b}"] = value
    return out


def low_degree_even_ensemble(dimension: int) -> np.ndarray:
    """All cell truth tables with the highest GF(2) monomial absent."""
    patterns = all_patterns(dimension)
    return patterns[mixed_syndrome(patterns) == 0]


def payload_execution_gate(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("COCHAIN-Q v0 is source-only and grants no Qwen/GPU/payload execution")
