#!/usr/bin/env python3
"""Source-locked sparse-tail peeling and robust-bulk ideal-RD oracle.

This is an intentionally favourable architecture gate.  It peels stable
top-|w| BF16 values losslessly, charges a combinatorial support code and a
canonical entropy code for the values, and then gives the remaining bulk an
ideal RHT/polar-lattice test channel.  A support-pattern-conditioned role KLT
is evaluated without pretending that a dense 3x3 KLT preserves the known-zero
support.  All residual components share one panel-wide reverse waterfill.

The ideal Gaussian residual channel omits finite-block and lattice shaping
losses.  Consequently F > 0.8 is a valid early stop for this tested family;
F <= 0.8 is only a promotion to a real serialized codec experiment.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import heapq
import itertools
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROWS = 768
COLS = 2048
ROLES = 3
EXPERTS = 6
MATRICES = 18
VALUES_PER_MATRIX = ROWS * COLS
VALUES_PER_EXPERT = ROLES * VALUES_PER_MATRIX
PANEL_VALUES = MATRICES * VALUES_PER_MATRIX
SOURCE_BYTES = VALUES_PER_MATRIX * 2

RATES = (2.15, 2.30, 2.50)
TARGET_F = 0.8
TARGET_S_BPW = -0.5 * math.log2(TARGET_F)

# Nested stable top-|w| counts.  The tiny counts make a single pathological
# weight visible; the upper end deliberately gives the hypothesis ample room.
TAIL_COUNTS = (
    0,
    1,
    2,
    4,
    8,
    16,
    24,
    48,
    96,
    192,
    384,
    768,
    1_536,
    3_072,
    6_144,
    12_288,
    24_576,
    49_152,
    98_304,
    196_608,
)

# Physical layout.  Every non-common field is stored inside its owning expert
# frame.  The 64-bit residual directory includes profile, scale and length.
GLOBAL_HEADER_BITS = 4096 * 8
ROUTE_TABLE_BITS = 144 * 8
EXPERT_HEADER_BITS = 256
RESIDUAL_DIRECTORY_BITS = 64
MATRIX_DESCRIPTOR_BITS = 128
SUPPORT_PATTERN_MODE_BITS = 7
ANGLE_BITS = 16
COMMON_PREFIX_BITS = GLOBAL_HEADER_BITS + ROUTE_TABLE_BITS

VALUE_MODE_BITS = 2  # stored inside the 128-bit matrix descriptor
VALUE_CODE_LENGTH_BITS = 16


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def write_sealed_json(path: Path, report: dict[str, Any]) -> None:
    clean = dict(report)
    clean.pop("result_lock_sha256", None)
    clean["result_lock_sha256"] = hashlib.sha256(canonical_json_bytes(clean)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def get_backend(name: str) -> tuple[Any, str]:
    if name == "numpy":
        return np, np.__version__
    if name != "cupy":
        raise ValueError(f"unknown backend: {name}")
    import cupy as cp  # type: ignore

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CuPy requested but no CUDA device is visible")
    return cp, cp.__version__


def as_numpy(value: Any, xp: Any) -> np.ndarray:
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def bf16_words_to_float32(words: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(words, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def float32_to_bf16_rne_words(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype("<u2")


@functools.lru_cache(maxsize=None)
def ceil_log2_binomial(n: int, k: int) -> int:
    """Exact fixed-length enumerative support-code size."""
    if not 0 <= k <= n:
        raise ValueError((n, k))
    count = math.comb(n, min(k, n - k))
    return 0 if count <= 1 else (count - 1).bit_length()


def huffman_payload_bits(counts: Sequence[int]) -> int:
    """Exact weighted path length of an optimal binary Huffman tree."""
    heap = [int(value) for value in counts if int(value) > 0]
    if len(heap) <= 1:
        return 0
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        merged = heapq.heappop(heap) + heapq.heappop(heap)
        total += merged
        heapq.heappush(heap, merged)
    return total


def canonical_huffman_stream_bits(
    counts: Sequence[int], *, symbol_bits: int, symbol_count_bits: int
) -> dict[str, int]:
    nonzero = [int(value) for value in counts if int(value) > 0]
    distinct = len(nonzero)
    model_bits = symbol_count_bits + distinct * (symbol_bits + VALUE_CODE_LENGTH_BITS)
    payload_bits = huffman_payload_bits(nonzero)
    return {
        "distinct_symbols": distinct,
        "model_bits": model_bits,
        "payload_bits": payload_bits,
        "total_bits": model_bits + payload_bits,
    }


def value_code_candidates(word_counts: np.ndarray) -> dict[str, dict[str, int]]:
    word_counts = np.asarray(word_counts, dtype=np.int64)
    if word_counts.shape != (1 << 16,) or np.any(word_counts < 0):
        raise ValueError("word histogram must contain 65536 non-negative counts")
    count = int(np.sum(word_counts, dtype=np.int64))
    literal = {
        "distinct_symbols": int(np.count_nonzero(word_counts)),
        "model_bits": 0,
        "payload_bits": 16 * count,
        "total_bits": 16 * count,
    }
    word = canonical_huffman_stream_bits(
        word_counts, symbol_bits=16, symbol_count_bits=17
    )

    symbols = np.flatnonzero(word_counts).astype(np.uint16)
    frequencies = word_counts[symbols.astype(np.int64)]
    magnitude_counts = np.bincount(
        (symbols & np.uint16(0x7FFF)).astype(np.int64),
        weights=frequencies,
        minlength=1 << 15,
    ).astype(np.int64)
    sign_counts = np.bincount(
        (symbols >> np.uint16(15)).astype(np.int64),
        weights=frequencies,
        minlength=2,
    ).astype(np.int64)
    magnitude = canonical_huffman_stream_bits(
        magnitude_counts, symbol_bits=15, symbol_count_bits=16
    )
    sign = canonical_huffman_stream_bits(sign_counts, symbol_bits=1, symbol_count_bits=2)
    magnitude_sign = {
        "distinct_symbols": magnitude["distinct_symbols"] + sign["distinct_symbols"],
        "model_bits": magnitude["model_bits"] + sign["model_bits"],
        "payload_bits": magnitude["payload_bits"] + sign["payload_bits"],
        "total_bits": magnitude["total_bits"] + sign["total_bits"],
    }

    exponent_counts = np.bincount(
        ((symbols >> np.uint16(7)) & np.uint16(0xFF)).astype(np.int64),
        weights=frequencies,
        minlength=1 << 8,
    ).astype(np.int64)
    mantissa_counts = np.bincount(
        (symbols & np.uint16(0x7F)).astype(np.int64),
        weights=frequencies,
        minlength=1 << 7,
    ).astype(np.int64)
    exponent = canonical_huffman_stream_bits(
        exponent_counts, symbol_bits=8, symbol_count_bits=9
    )
    mantissa = canonical_huffman_stream_bits(
        mantissa_counts, symbol_bits=7, symbol_count_bits=8
    )
    lanes = {
        "distinct_symbols": (
            sign["distinct_symbols"]
            + exponent["distinct_symbols"]
            + mantissa["distinct_symbols"]
        ),
        "model_bits": sign["model_bits"] + exponent["model_bits"] + mantissa["model_bits"],
        "payload_bits": (
            sign["payload_bits"] + exponent["payload_bits"] + mantissa["payload_bits"]
        ),
        "total_bits": sign["total_bits"] + exponent["total_bits"] + mantissa["total_bits"],
    }
    return {
        "literal16": literal,
        "word_huffman": word,
        "magnitude_sign_huffman": magnitude_sign,
        "sign_exponent_mantissa_huffman": lanes,
    }


def best_value_code(word_counts: np.ndarray) -> dict[str, Any]:
    candidates = value_code_candidates(word_counts)
    mode, row = min(candidates.items(), key=lambda item: (item[1]["total_bits"], item[0]))
    return {
        "mode": mode,
        **row,
        "mode_bits_in_matrix_descriptor": VALUE_MODE_BITS,
        "all_mode_total_bits": {name: value["total_bits"] for name, value in candidates.items()},
    }


@dataclass(frozen=True)
class TailCandidate:
    candidate_index: int
    k: int
    tail_energy: float
    residual_energy: float
    mask_bits: int
    value_bits: int
    value_mode: str
    value_detail: dict[str, Any]

    @property
    def variable_side_bits(self) -> int:
        return self.mask_bits + self.value_bits

    def public(self, total_energy: float) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "k": self.k,
            "tail_fraction": self.k / VALUES_PER_MATRIX,
            "tail_energy": self.tail_energy,
            "tail_energy_fraction": self.tail_energy / total_energy,
            "residual_energy": self.residual_energy,
            "mask_code": {
                "kind": "fixed-length combinatorial number system",
                "bits": self.mask_bits,
                "cardinality": f"C({VALUES_PER_MATRIX},{self.k})",
            },
            "value_code": self.value_detail,
            "variable_side_bits": self.variable_side_bits,
        }


@dataclass
class MatrixWork:
    values: Any
    stages: Any
    candidates: list[TailCandidate]
    energy: float
    receipt: dict[str, Any]


@dataclass(frozen=True)
class Component:
    name: str
    owner_expert: int
    dimension: int
    energy: float


@dataclass
class SupportTable:
    expert_ordinal: int
    category_count: int
    counts: np.ndarray
    grams: np.ndarray
    cache: dict[tuple[int, int, int], tuple[list[Component], int]]
    count_prefix: np.ndarray | None = None
    gram_prefix: np.ndarray | None = None

    def ensure_prefixes(self) -> None:
        if self.count_prefix is not None and self.gram_prefix is not None:
            return
        categories = self.category_count
        count_grid = self.counts.reshape(categories, categories, categories)
        gram_grid = self.grams.reshape(categories, categories, categories, ROLES, ROLES)
        self.count_prefix = np.zeros(
            (categories + 1, categories + 1, categories + 1), dtype=np.int64
        )
        self.gram_prefix = np.zeros(
            (categories + 1, categories + 1, categories + 1, ROLES, ROLES),
            dtype=np.float64,
        )
        self.count_prefix[1:, 1:, 1:] = np.cumsum(
            np.cumsum(np.cumsum(count_grid, axis=0), axis=1), axis=2
        )
        self.gram_prefix[1:, 1:, 1:] = np.cumsum(
            np.cumsum(np.cumsum(gram_grid, axis=0), axis=1), axis=2
        )

    @staticmethod
    def rectangle_sum(prefix: np.ndarray, lower: Sequence[int], upper: Sequence[int]) -> Any:
        """Inclusive 3-D rectangle sum from an origin-padded prefix array."""
        if any(int(lower[i]) > int(upper[i]) for i in range(ROLES)):
            return 0
        result: Any = 0
        for corner in range(1 << ROLES):
            index = []
            lower_count = 0
            for axis in range(ROLES):
                if corner & (1 << axis):
                    index.append(int(lower[axis]))
                    lower_count += 1
                else:
                    index.append(int(upper[axis]) + 1)
            sign = -1 if lower_count % 2 else 1
            result = result + sign * prefix[tuple(index)]
        return result

    def components(self, choices: Sequence[int]) -> tuple[list[Component], int]:
        key = tuple(int(x) for x in choices)
        if len(key) != ROLES:
            raise ValueError(key)
        if key in self.cache:
            return self.cache[key]
        grouped_counts = np.zeros(1 << ROLES, dtype=np.int64)
        grouped_grams = np.zeros((1 << ROLES, ROLES, ROLES), dtype=np.float64)
        categories = self.category_count
        self.ensure_prefixes()
        assert self.count_prefix is not None and self.gram_prefix is not None
        for mask in range(1, 1 << ROLES):
            lower = []
            upper = []
            for role in range(ROLES):
                if mask & (1 << role):
                    lower.append(key[role] + 1)
                    upper.append(categories - 1)
                else:
                    lower.append(0)
                    upper.append(key[role])
            grouped_counts[mask] = int(
                self.rectangle_sum(self.count_prefix, lower, upper)
            )
            grouped_grams[mask] = self.rectangle_sum(self.gram_prefix, lower, upper)

        components: list[Component] = []
        angle_count = 0
        for mask in range(1, 1 << ROLES):
            count = int(grouped_counts[mask])
            if count == 0:
                continue
            active = [role for role in range(ROLES) if mask & (1 << role)]
            covariance = grouped_grams[mask][np.ix_(active, active)]
            eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
            tolerance = max(float(np.max(eigenvalues)), 1.0) * 2e-13
            if float(np.min(eigenvalues)) < -tolerance:
                raise AssertionError((key, mask, eigenvalues.tolist()))
            eigenvalues = np.maximum(eigenvalues, 0.0)
            angle_count += len(active) * (len(active) - 1) // 2
            for channel, energy in enumerate(eigenvalues):
                if float(energy) <= tolerance:
                    continue
                components.append(
                    Component(
                        name=(
                            f"expert_{self.expert_ordinal:02d}.support_{mask:03b}."
                            f"xklt_{channel}"
                        ),
                        owner_expert=self.expert_ordinal,
                        dimension=count,
                        energy=float(energy),
                    )
                )
        result = (components, angle_count)
        self.cache[key] = result
        return result


@dataclass
class PanelStats:
    label: str
    matrix_candidates: list[list[TailCandidate]]
    matrix_energies: list[float]
    matrix_receipts: list[dict[str, Any]]
    support_tables: list[SupportTable]
    total_energy: float
    control_replica: int | None = None


@dataclass(frozen=True)
class DualOptionBank:
    expert_ordinal: int
    geometry: str
    choices: np.ndarray
    side_bits: np.ndarray
    dimensions: np.ndarray
    energies: np.ndarray


def stable_tail_order(values: Any, xp: Any) -> np.ndarray:
    flat = values.reshape(-1)
    index = xp.arange(flat.size, dtype=xp.int64)
    order = xp.lexsort((index, -xp.abs(flat)))
    return np.ascontiguousarray(as_numpy(order, xp), dtype=np.int64)


def build_matrix_work(
    values: Any,
    words: np.ndarray,
    receipt: dict[str, Any],
    xp: Any,
) -> MatrixWork:
    flat = values.reshape(-1)
    host_words = np.ascontiguousarray(words.reshape(-1), dtype="<u2")
    if flat.size != VALUES_PER_MATRIX or host_words.size != VALUES_PER_MATRIX:
        raise AssertionError((flat.size, host_words.size))
    energy = float(as_numpy(xp.sum(flat.astype(xp.float64) ** 2, dtype=xp.float64), xp))
    if not energy > 0.0:
        raise ValueError("non-positive matrix energy")
    order = stable_tail_order(flat, xp)
    maximum_k = TAIL_COUNTS[-1]
    selected_values = as_numpy(flat[xp.asarray(order[:maximum_k])], xp).astype(np.float64)
    energy_prefix = np.concatenate(
        ([0.0], np.cumsum(np.square(selected_values), dtype=np.float64))
    )
    selected_words = host_words[order[:maximum_k]]
    word_counts = np.zeros(1 << 16, dtype=np.int64)
    stages = np.full(VALUES_PER_MATRIX, len(TAIL_COUNTS), dtype=np.uint8)
    candidates: list[TailCandidate] = []
    previous = 0
    for candidate_index, k in enumerate(TAIL_COUNTS):
        if k > previous:
            np.add.at(word_counts, selected_words[previous:k].astype(np.int64), 1)
            stages[order[previous:k]] = candidate_index
        value = best_value_code(word_counts)
        tail_energy = float(energy_prefix[k])
        candidate = TailCandidate(
            candidate_index=candidate_index,
            k=k,
            tail_energy=tail_energy,
            residual_energy=max(0.0, energy - tail_energy),
            mask_bits=ceil_log2_binomial(VALUES_PER_MATRIX, k),
            value_bits=int(value["total_bits"]),
            value_mode=str(value["mode"]),
            value_detail=value,
        )
        candidates.append(candidate)
        previous = k
    if any(
        candidates[i].tail_energy > candidates[i + 1].tail_energy + 1e-12
        for i in range(len(candidates) - 1)
    ):
        raise AssertionError("tail energy is not nested")
    return MatrixWork(
        values=flat,
        stages=xp.asarray(stages),
        candidates=candidates,
        energy=energy,
        receipt=receipt,
    )


def build_support_table(
    expert_ordinal: int, triplet: Sequence[MatrixWork], xp: Any
) -> SupportTable:
    categories = len(TAIL_COUNTS) + 1
    code = (
        (triplet[0].stages.astype(xp.int64) * categories + triplet[1].stages) * categories
        + triplet[2].stages
    ).astype(xp.int64)
    bins = categories**ROLES
    counts = as_numpy(xp.bincount(code, minlength=bins), xp).astype(np.int64)
    grams = np.zeros((bins, ROLES, ROLES), dtype=np.float64)
    for left in range(ROLES):
        for right in range(left, ROLES):
            weights = triplet[left].values.astype(xp.float64) * triplet[right].values.astype(
                xp.float64
            )
            values = as_numpy(xp.bincount(code, weights=weights, minlength=bins), xp).astype(
                np.float64
            )
            grams[:, left, right] = values
            grams[:, right, left] = values
    if int(np.sum(counts, dtype=np.int64)) != VALUES_PER_MATRIX:
        raise AssertionError("support histogram does not close")
    return SupportTable(expert_ordinal, categories, counts, grams, {})


def validate_lock(lock: dict[str, Any]) -> None:
    rows = lock.get("matrices", [])
    if len(rows) != MATRICES:
        raise ValueError("source lock is not the pinned 18-matrix panel")
    for ordinal, row in enumerate(rows):
        if int(row.get("matrix_ordinal", -1)) != ordinal:
            raise ValueError("non-canonical source ordinal")
        if str(row.get("role")) != ("gate", "up", "down")[ordinal % 3]:
            raise ValueError("non-canonical role order")
        shape = tuple(int(x) for x in row["shape"])
        expected = (COLS, ROWS) if row["role"] == "down" else (ROWS, COLS)
        if shape != expected:
            raise ValueError((ordinal, shape, expected))


def load_source_matrix(
    source_root: Path, row: dict[str, Any], xp: Any
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    path = (source_root / str(row["output_relpath"])).resolve(strict=True)
    if source_root != path and source_root not in path.parents:
        raise ValueError("source escaped source root")
    if path.stat().st_size != SOURCE_BYTES:
        raise ValueError(f"wrong BF16 source length: {path}")
    observed = sha256_file(path)
    declared = str(row["source_bf16_sha256"])
    if observed != declared:
        raise ValueError(f"source hash mismatch: {path}")
    shape = tuple(int(x) for x in row["shape"])
    words = np.fromfile(path, dtype="<u2").reshape(shape)
    if str(row["role"]) == "down":
        words = np.ascontiguousarray(words.T)
    values = bf16_words_to_float32(words)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite BF16 source: {path}")
    receipt = {
        "matrix_ordinal": int(row["matrix_ordinal"]),
        "layer": int(row["layer"]),
        "expert": int(row["expert"]),
        "role": str(row["role"]),
        "tensor": str(row["tensor"]),
        "relative_path": str(path.relative_to(source_root)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "declared_sha256": declared,
        "observed_sha256": observed,
    }
    return xp.asarray(values), words, receipt


def build_source_panel(source_lock: Path, source_root: Path, xp: Any) -> tuple[PanelStats, dict[str, Any]]:
    lock = json.loads(source_lock.read_text(encoding="utf-8"))
    validate_lock(lock)
    matrix_candidates: list[list[TailCandidate]] = []
    matrix_energies: list[float] = []
    receipts: list[dict[str, Any]] = []
    support_tables: list[SupportTable] = []
    for expert in range(EXPERTS):
        triplet: list[MatrixWork] = []
        for role in range(ROLES):
            row = lock["matrices"][ROLES * expert + role]
            values, words, receipt = load_source_matrix(source_root, row, xp)
            work = build_matrix_work(values, words, receipt, xp)
            triplet.append(work)
            matrix_candidates.append(work.candidates)
            matrix_energies.append(work.energy)
            receipts.append(receipt)
        support_tables.append(build_support_table(expert, triplet, xp))
        print(f"[qwen] prepared expert {expert + 1}/{EXPERTS}", flush=True)
    total = float(sum(matrix_energies))
    return (
        PanelStats(
            label="qwen",
            matrix_candidates=matrix_candidates,
            matrix_energies=matrix_energies,
            matrix_receipts=receipts,
            support_tables=support_tables,
            total_energy=total,
        ),
        lock,
    )


def make_gaussian_words(count: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    raw = rng.standard_normal(count, dtype=np.float32)
    words = float32_to_bf16_rne_words(raw)
    if not np.all(np.isfinite(bf16_words_to_float32(words))):
        raise AssertionError("non-finite Gaussian control")
    return words


def build_control_panel(
    reference: PanelStats, replica: int, checkpoint: str, xp: Any
) -> PanelStats:
    matrix_candidates: list[list[TailCandidate]] = []
    matrix_energies: list[float] = []
    receipts: list[dict[str, Any]] = []
    support_tables: list[SupportTable] = []
    for expert in range(EXPERTS):
        triplet: list[MatrixWork] = []
        for role in range(ROLES):
            ordinal = ROLES * expert + role
            seed = stable_seed("tail-peeling-gaussian-v1", checkpoint, replica, ordinal)
            words = make_gaussian_words(VALUES_PER_MATRIX, seed).reshape(ROWS, COLS)
            base = bf16_words_to_float32(words)
            base_energy = float(np.sum(np.square(base), dtype=np.float64))
            scale = math.sqrt(reference.matrix_energies[ordinal] / base_energy)
            # Keep the conceptual normalization in FP64.  The entropy stream
            # still codes the underlying BF16 symbols exactly; scale affects
            # only the matched-control metric and Gram matrix.
            values = xp.asarray(base, dtype=xp.float64) * scale
            receipt = {
                "matrix_ordinal": ordinal,
                "role": ("gate", "up", "down")[role],
                "replica": replica,
                "pcg64dxsm_seed": seed,
                "conceptual_energy_match_scale": scale,
                "base_bf16_words_sha256": hashlib.sha256(words.tobytes()).hexdigest(),
            }
            work = build_matrix_work(values, words, receipt, xp)
            if not math.isclose(
                work.energy, reference.matrix_energies[ordinal], rel_tol=2e-12, abs_tol=2e-8
            ):
                raise AssertionError((work.energy, reference.matrix_energies[ordinal]))
            triplet.append(work)
            matrix_candidates.append(work.candidates)
            matrix_energies.append(work.energy)
            receipts.append(receipt)
        support_tables.append(build_support_table(expert, triplet, xp))
        print(f"[gaussian {replica}] prepared expert {expert + 1}/{EXPERTS}", flush=True)
    return PanelStats(
        label=f"gaussian_{replica}",
        matrix_candidates=matrix_candidates,
        matrix_energies=matrix_energies,
        matrix_receipts=receipts,
        support_tables=support_tables,
        total_energy=float(sum(matrix_energies)),
        control_replica=replica,
    )


def continuous_waterfill(dimensions: np.ndarray, energies: np.ndarray, budget_bits: int) -> np.ndarray:
    dimensions = np.asarray(dimensions, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    if budget_bits < 0 or np.any(dimensions <= 0.0) or np.any(energies <= 0.0):
        raise ValueError("invalid waterfill geometry")
    logv = np.log2(energies / dimensions)
    order = np.argsort(logv)[::-1]
    sorted_logv = logv[order]
    sorted_d = dimensions[order]
    cumulative_d = np.cumsum(sorted_d)
    cumulative_dlogv = np.cumsum(sorted_d * sorted_logv)
    levels = (cumulative_dlogv - 2.0 * budget_bits) / cumulative_d
    active_count = len(dimensions)
    for count in range(1, len(dimensions) + 1):
        level = levels[count - 1]
        active_ok = level <= sorted_logv[count - 1] + 2e-13
        inactive_ok = count == len(dimensions) or level >= sorted_logv[count] - 2e-13
        if active_ok and inactive_ok:
            active_count = count
            break
    log_level = float(levels[active_count - 1])
    allocation = 0.5 * dimensions * np.maximum(0.0, logv - log_level)
    if not math.isclose(float(np.sum(allocation)), budget_bits, rel_tol=2e-11, abs_tol=1e-4):
        raise AssertionError((float(np.sum(allocation)), budget_bits))
    return allocation


def integer_waterfill(components: Sequence[Component], budget_bits: int) -> dict[str, Any]:
    if not components:
        raise ValueError("empty component bank")
    dimensions = np.asarray([item.dimension for item in components], dtype=np.float64)
    energies = np.asarray([item.energy for item in components], dtype=np.float64)
    real = continuous_waterfill(dimensions, energies, budget_bits)
    bits = np.floor(real + 1e-9).astype(np.int64)
    missing = int(budget_bits - int(np.sum(bits, dtype=np.int64)))
    if not 0 <= missing <= len(components) + 2:
        raise AssertionError((missing, len(components), float(np.sum(real))))
    for _ in range(missing):
        # Exact one-bit marginal reduction, evaluated in log space.
        log_marginal = (
            np.log(energies)
            - 2.0 * bits / dimensions * math.log(2.0)
            + np.log(-np.expm1(-2.0 * math.log(2.0) / dimensions))
        )
        winner = int(np.argmax(log_marginal))
        bits[winner] += 1
    if int(np.sum(bits, dtype=np.int64)) != budget_bits:
        raise AssertionError("integer payload does not close")
    distortion = energies * np.exp(-2.0 * bits / dimensions * math.log(2.0))
    return {
        "payload_bits": budget_bits,
        "distortion_sse": float(np.sum(distortion, dtype=np.float64)),
        "active_components": int(np.count_nonzero(bits)),
        "component_count": len(components),
        "dimension_sum": int(sum(item.dimension for item in components)),
        "energy_sum": float(sum(item.energy for item in components)),
        "allocations": [
            {
                "name": item.name,
                "owner_expert": item.owner_expert,
                "dimension": item.dimension,
                "energy": item.energy,
                "payload_bits": int(bit_count),
                "distortion_sse": float(error),
            }
            for item, bit_count, error in zip(components, bits, distortion, strict=True)
        ],
    }


def fixed_side_ledger(geometry: str, residual_component_count: int) -> dict[str, int]:
    row = {
        "global_header": GLOBAL_HEADER_BITS,
        "route_table": ROUTE_TABLE_BITS,
        "expert_headers": EXPERTS * EXPERT_HEADER_BITS,
        # A support-XKLT frame can contain more than one independently
        # allocated channel per source matrix.  Every live waterfill component
        # therefore receives its own literal scale/profile/length directory.
        "residual_directories": residual_component_count * RESIDUAL_DIRECTORY_BITS,
        "matrix_tail_descriptors": MATRICES * MATRIX_DESCRIPTOR_BITS,
        "support_pattern_modes": (
            EXPERTS * SUPPORT_PATTERN_MODE_BITS if geometry == "support_xklt" else 0
        ),
    }
    row["total"] = sum(row.values())
    return row


def build_components(
    panel: PanelStats, choices: Sequence[int], geometry: str
) -> tuple[list[Component], list[int]]:
    if len(choices) != MATRICES:
        raise ValueError("choice vector must cover all matrices")
    if geometry == "raw":
        components = []
        for ordinal, choice in enumerate(choices):
            candidate = panel.matrix_candidates[ordinal][choice]
            if candidate.residual_energy > 0.0 and candidate.k < VALUES_PER_MATRIX:
                components.append(
                    Component(
                        name=f"matrix_{ordinal:02d}.robust_bulk",
                        owner_expert=ordinal // ROLES,
                        dimension=VALUES_PER_MATRIX - candidate.k,
                        energy=candidate.residual_energy,
                    )
                )
        return components, [0] * EXPERTS
    if geometry != "support_xklt":
        raise ValueError(geometry)
    components: list[Component] = []
    angles: list[int] = []
    for expert, table in enumerate(panel.support_tables):
        local, angle_count = table.components(choices[ROLES * expert : ROLES * expert + ROLES])
        components.extend(local)
        angles.append(angle_count)
    return components, angles


def candidate_variable_bits(candidate: TailCandidate, side_mode: str) -> int:
    if side_mode == "charged":
        return candidate.mask_bits + candidate.value_bits
    if side_mode == "free_values":
        return candidate.mask_bits
    if side_mode == "free_mask_values":
        return 0
    raise ValueError(side_mode)


def read_ledger(
    capacity_bytes: int,
    geometry: str,
    choices: Sequence[int],
    panel: PanelStats,
    allocations: Sequence[dict[str, Any]],
    angle_counts: Sequence[int],
    side_mode: str,
    charge_basis: bool,
) -> dict[str, Any]:
    payload_by_expert = [0] * EXPERTS
    components_by_expert = [0] * EXPERTS
    for row in allocations:
        owner = int(row["owner_expert"])
        payload_by_expert[owner] += int(row["payload_bits"])
        components_by_expert[owner] += 1
    frame_bits: list[int] = []
    for expert in range(EXPERTS):
        fixed = (
            EXPERT_HEADER_BITS
            + components_by_expert[expert] * RESIDUAL_DIRECTORY_BITS
            + ROLES * MATRIX_DESCRIPTOR_BITS
        )
        if geometry == "support_xklt":
            fixed += SUPPORT_PATTERN_MODE_BITS
        variable = sum(
            candidate_variable_bits(
                panel.matrix_candidates[ROLES * expert + role][choices[ROLES * expert + role]],
                side_mode,
            )
            for role in range(ROLES)
        )
        basis = angle_counts[expert] * ANGLE_BITS if charge_basis else 0
        frame_bits.append(fixed + variable + basis + payload_by_expert[expert])
    physical_bits = capacity_bytes * 8
    closure = COMMON_PREFIX_BITS + sum(frame_bits)
    if closure != physical_bits:
        raise AssertionError((closure, physical_bits))
    reference_bytes = capacity_bytes / EXPERTS
    experts = []
    for expert, bits in enumerate(frame_bits):
        cold_bytes = math.ceil(COMMON_PREFIX_BITS / 8) + math.ceil(bits / 8)
        page_bytes = (
            math.ceil(COMMON_PREFIX_BITS / (4096 * 8))
            + math.ceil(bits / (4096 * 8))
        ) * 4096
        experts.append(
            {
                "expert_ordinal": expert,
                "frame_bits": bits,
                "residual_payload_bits": payload_by_expert[expert],
                "cold_bytes": cold_bytes,
                "cold_amplification": cold_bytes / reference_bytes,
                "cold_4k_bytes": page_bytes,
                "cold_4k_amplification": page_bytes / reference_bytes,
            }
        )
    return {
        "reference_one_sixth_container_bytes": reference_bytes,
        "common_prefix_bits_read_per_cold_expert": COMMON_PREFIX_BITS,
        "experts": experts,
        "maximum_cold_amplification": max(row["cold_amplification"] for row in experts),
        "maximum_cold_4k_amplification": max(
            row["cold_4k_amplification"] for row in experts
        ),
        "below_2x": all(row["cold_amplification"] < 2.0 for row in experts),
        "bit_closure": closure,
        "claim_boundary": (
            "Compressed-object traffic only; RHT/XKLT/polar reconstruction must be fused "
            "with expert GEMMs to avoid a dense-weight HBM materialization."
        ),
    }


def score_configuration(
    panel: PanelStats,
    choices: Sequence[int],
    *,
    geometry: str,
    requested_rate: float,
    side_mode: str = "charged",
    charge_basis: bool = True,
    include_allocations: bool = False,
) -> dict[str, Any]:
    requested_fraction = Fraction(str(requested_rate))
    capacity_bytes = (requested_fraction.numerator * PANEL_VALUES) // (
        requested_fraction.denominator * 8
    )
    physical_bits = capacity_bytes * 8
    physical_rate = physical_bits / PANEL_VALUES
    components, angle_counts = build_components(panel, choices, geometry)
    fixed = fixed_side_ledger(geometry, len(components))
    variable_by_matrix = [
        candidate_variable_bits(panel.matrix_candidates[i][choice], side_mode)
        for i, choice in enumerate(choices)
    ]
    basis_bits = sum(angle_counts) * ANGLE_BITS if charge_basis else 0
    side_bits = fixed["total"] + sum(variable_by_matrix) + basis_bits
    payload_bits = physical_bits - side_bits
    if payload_bits <= 0 or not components:
        return {
            "valid": False,
            "requested_rate_bpw": requested_rate,
            "physical_rate_bpw": physical_rate,
            "capacity_bytes": capacity_bytes,
            "physical_bits": physical_bits,
            "side_bits": side_bits,
            "payload_bits": payload_bits,
            "choices": list(choices),
        }
    waterfill = integer_waterfill(components, payload_bits)
    mse = waterfill["distortion_sse"] / panel.total_energy
    f_value = mse * 2.0 ** (2.0 * physical_rate)
    ledger = read_ledger(
        capacity_bytes,
        geometry,
        choices,
        panel,
        waterfill["allocations"],
        angle_counts,
        side_mode,
        charge_basis,
    )
    result: dict[str, Any] = {
        "valid": True,
        "panel": panel.label,
        "geometry": geometry,
        "side_mode": side_mode,
        "basis_charged": charge_basis,
        "requested_rate_bpw": requested_rate,
        "physical_rate_bpw": physical_rate,
        "capacity_bytes": capacity_bytes,
        "physical_bits": physical_bits,
        "choices": list(int(x) for x in choices),
        "tail_counts": [
            panel.matrix_candidates[i][choice].k for i, choice in enumerate(choices)
        ],
        "peeled_weights": sum(
            panel.matrix_candidates[i][choice].k for i, choice in enumerate(choices)
        ),
        "peeled_fraction": sum(
            panel.matrix_candidates[i][choice].k for i, choice in enumerate(choices)
        )
        / PANEL_VALUES,
        "peeled_energy_fraction": sum(
            panel.matrix_candidates[i][choice].tail_energy
            for i, choice in enumerate(choices)
        )
        / panel.total_energy,
        "fixed_side_bits": fixed,
        "tail_variable_bits_by_matrix": variable_by_matrix,
        "tail_variable_bits": sum(variable_by_matrix),
        "support_xklt_angle_counts_by_expert": angle_counts,
        "support_xklt_angle_bits": basis_bits,
        "side_bits": side_bits,
        "side_bpw": side_bits / PANEL_VALUES,
        "payload_bits": payload_bits,
        "payload_bpw": payload_bits / PANEL_VALUES,
        "residual_dimension_sum": waterfill["dimension_sum"],
        "residual_energy_fraction": waterfill["energy_sum"] / panel.total_energy,
        "active_components": waterfill["active_components"],
        "component_count": waterfill["component_count"],
        "distortion_sse": waterfill["distortion_sse"],
        "ideal_relative_mse": mse,
        "gaussian_reference_mse": 2.0 ** (-2.0 * physical_rate),
        "target_mse": TARGET_F * 2.0 ** (-2.0 * physical_rate),
        "F": f_value,
        "s_bpw": -0.5 * math.log2(f_value),
        "passes_F_le_0p8": bool(f_value <= TARGET_F),
        "read_ledger": ledger,
    }
    if include_allocations:
        result["allocations"] = waterfill["allocations"]
    return result


def score_order(row: dict[str, Any]) -> tuple[Any, ...]:
    if not row.get("valid", False):
        return (math.inf, math.inf, math.inf, ())
    return (
        float(row["F"]),
        int(row["side_bits"]),
        int(row["peeled_weights"]),
        tuple(row["choices"]),
    )


def coordinate_search(
    panel: PanelStats,
    *,
    geometry: str,
    requested_rate: float,
    side_mode: str,
    charge_basis: bool,
    maximum_passes: int,
    extra_seeds: Iterable[Sequence[int]] = (),
) -> dict[str, Any]:
    evaluations = 0
    uniform: list[dict[str, Any]] = []
    for candidate in range(len(TAIL_COUNTS)):
        choices = [candidate] * MATRICES
        row = score_configuration(
            panel,
            choices,
            geometry=geometry,
            requested_rate=requested_rate,
            side_mode=side_mode,
            charge_basis=charge_basis,
        )
        uniform.append(row)
        evaluations += 1
    seeds: list[list[int]] = [list(row["choices"]) for row in sorted(uniform, key=score_order)[:4]]
    seeds.extend([list(int(x) for x in seed) for seed in extra_seeds])
    unique: dict[tuple[int, ...], list[int]] = {}
    for seed in seeds:
        if len(seed) == MATRICES:
            unique[tuple(seed)] = seed
    best_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for seed in unique.values():
        choices = list(seed)
        current = score_configuration(
            panel,
            choices,
            geometry=geometry,
            requested_rate=requested_rate,
            side_mode=side_mode,
            charge_basis=charge_basis,
        )
        evaluations += 1
        passes = 0
        changes = 0
        while passes < maximum_passes:
            changed = False
            for ordinal in range(MATRICES):
                original = choices[ordinal]
                local_best = current
                local_choice = original
                for candidate in range(len(TAIL_COUNTS)):
                    if candidate == original:
                        continue
                    choices[ordinal] = candidate
                    row = score_configuration(
                        panel,
                        choices,
                        geometry=geometry,
                        requested_rate=requested_rate,
                        side_mode=side_mode,
                        charge_basis=charge_basis,
                    )
                    evaluations += 1
                    if score_order(row) < score_order(local_best):
                        local_best = row
                        local_choice = candidate
                choices[ordinal] = local_choice
                if local_choice != original:
                    changes += 1
                    changed = True
                current = local_best
            passes += 1
            if not changed:
                break
        final = score_configuration(
            panel,
            choices,
            geometry=geometry,
            requested_rate=requested_rate,
            side_mode=side_mode,
            charge_basis=charge_basis,
            include_allocations=True,
        )
        evaluations += 1
        best_rows.append(final)
        traces.append(
            {
                "initial_choices": seed,
                "final_choices": list(choices),
                "passes": passes,
                "coordinate_changes": changes,
                "final_F": final.get("F"),
            }
        )
    best = min(best_rows, key=score_order)
    best["search"] = {
        "kind": "uniform-grid multistart plus deterministic coordinate descent",
        "uniform_candidates": len(uniform),
        "multistarts": len(unique),
        "maximum_passes": maximum_passes,
        "evaluations": evaluations,
        "traces": traces,
    }
    return best


def build_dual_option_banks(panel: PanelStats, geometry: str) -> list[DualOptionBank]:
    """Enumerate every 20^3 expert-local tail choice for a dual certificate."""
    if geometry not in ("raw", "support_xklt"):
        raise ValueError(geometry)
    choice_grid = np.asarray(
        list(itertools.product(range(len(TAIL_COUNTS)), repeat=ROLES)), dtype=np.uint8
    )
    maximum_components = ROLES if geometry == "raw" else ROLES * (1 << (ROLES - 1))
    banks: list[DualOptionBank] = []
    for expert in range(EXPERTS):
        dimensions = np.zeros((len(choice_grid), maximum_components), dtype=np.float64)
        energies = np.zeros_like(dimensions)
        sides = np.zeros(len(choice_grid), dtype=np.float64)
        for option_index, local_choices_raw in enumerate(choice_grid):
            local_choices = tuple(int(value) for value in local_choices_raw)
            if geometry == "raw":
                components = []
                angle_count = 0
                for role, choice in enumerate(local_choices):
                    candidate = panel.matrix_candidates[ROLES * expert + role][choice]
                    if candidate.residual_energy > 0.0:
                        components.append(
                            Component(
                                name=f"expert_{expert:02d}.raw_{role}",
                                owner_expert=expert,
                                dimension=VALUES_PER_MATRIX - candidate.k,
                                energy=candidate.residual_energy,
                            )
                        )
            else:
                components, angle_count = panel.support_tables[expert].components(local_choices)
            if len(components) > maximum_components:
                raise AssertionError((geometry, len(components), maximum_components))
            for component_index, component in enumerate(components):
                dimensions[option_index, component_index] = component.dimension
                energies[option_index, component_index] = component.energy
            side = (
                EXPERT_HEADER_BITS
                + ROLES * MATRIX_DESCRIPTOR_BITS
                + len(components) * RESIDUAL_DIRECTORY_BITS
                + sum(
                    panel.matrix_candidates[ROLES * expert + role][choice].variable_side_bits
                    for role, choice in enumerate(local_choices)
                )
            )
            if geometry == "support_xklt":
                side += SUPPORT_PATTERN_MODE_BITS + angle_count * ANGLE_BITS
            sides[option_index] = side
        banks.append(
            DualOptionBank(
                expert_ordinal=expert,
                geometry=geometry,
                choices=choice_grid.copy(),
                side_bits=sides,
                dimensions=dimensions,
                energies=energies,
            )
        )
        print(
            f"[{panel.label}] enumerated dual bank {geometry} expert "
            f"{expert + 1}/{EXPERTS}",
            flush=True,
        )
    return banks


def dual_point(
    banks: Sequence[DualOptionBank], theta: float, physical_bits: int
) -> dict[str, Any]:
    if not theta > 0.0:
        raise ValueError(theta)
    multiplier = 2.0 * math.log(2.0) * theta
    selected_options = []
    selected_choices: list[int] = []
    selected_bits = float(COMMON_PREFIX_BITS)
    selected_distortion = 0.0
    minimum_lagrangian = multiplier * (COMMON_PREFIX_BITS - physical_bits)
    for bank in banks:
        positive = bank.dimensions > 0.0
        variance = np.ones_like(bank.energies)
        np.divide(bank.energies, bank.dimensions, out=variance, where=positive)
        log_ratio = np.zeros_like(variance)
        np.log2(variance / theta, out=log_ratio, where=positive)
        payload = 0.5 * np.sum(
            bank.dimensions * np.maximum(log_ratio, 0.0), axis=1, dtype=np.float64
        )
        distortion = np.sum(
            np.where(positive & (variance > theta), bank.dimensions * theta, bank.energies),
            axis=1,
            dtype=np.float64,
        )
        lagrangian = distortion + multiplier * (bank.side_bits + payload)
        winner = int(np.argmin(lagrangian))
        minimum_lagrangian += float(lagrangian[winner])
        selected_bits += float(bank.side_bits[winner] + payload[winner])
        selected_distortion += float(distortion[winner])
        local = [int(value) for value in bank.choices[winner]]
        selected_choices.extend(local)
        selected_options.append(
            {
                "expert_ordinal": bank.expert_ordinal,
                "option_index": winner,
                "choices": local,
                "side_bits": int(bank.side_bits[winner]),
                "continuous_payload_bits": float(payload[winner]),
                "continuous_distortion_sse": float(distortion[winner]),
            }
        )
    return {
        "theta": theta,
        "lagrange_multiplier_distortion_per_bit": multiplier,
        "dual_lower_bound_sse": minimum_lagrangian,
        "selected_continuous_bits": selected_bits,
        "selected_continuous_distortion_sse": selected_distortion,
        "selected_choices": selected_choices,
        "selected_options": selected_options,
    }


def lagrange_dual_certificate(
    panel: PanelStats,
    banks: Sequence[DualOptionBank],
    *,
    geometry: str,
    requested_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    requested_fraction = Fraction(str(requested_rate))
    capacity_bytes = (requested_fraction.numerator * PANEL_VALUES) // (
        requested_fraction.denominator * 8
    )
    physical_bits = capacity_bytes * 8
    physical_rate = physical_bits / PANEL_VALUES
    positive_variances = np.concatenate(
        [
            bank.energies[bank.dimensions > 0.0] / bank.dimensions[bank.dimensions > 0.0]
            for bank in banks
        ]
    )
    lower_log = math.log(max(float(np.min(positive_variances)) * 2.0**-8, 1e-300))
    upper_log = math.log(float(np.max(positive_variances)) * 4.0)
    all_points: dict[float, dict[str, Any]] = {}

    def evaluate(log_theta: float) -> dict[str, Any]:
        if log_theta not in all_points:
            all_points[log_theta] = dual_point(banks, math.exp(log_theta), physical_bits)
        return all_points[log_theta]

    grid = np.linspace(lower_log, upper_log, 129)
    for value in grid:
        evaluate(float(value))
    for _ in range(5):
        ordered = sorted(all_points)
        best_index = max(
            range(len(ordered)),
            key=lambda index: all_points[ordered[index]]["dual_lower_bound_sse"],
        )
        left = ordered[max(0, best_index - 1)]
        right = ordered[min(len(ordered) - 1, best_index + 1)]
        if not right > left:
            break
        for value in np.linspace(left, right, 33):
            evaluate(float(value))
    best_point = max(all_points.values(), key=lambda row: row["dual_lower_bound_sse"])
    lower_sse = float(best_point["dual_lower_bound_sse"])
    lower_mse = lower_sse / panel.total_energy
    lower_f = lower_mse * 2.0 ** (2.0 * physical_rate)

    # Every dual-selected configuration is a useful primal candidate.  Score
    # all unique choices exactly with the integer-bit waterfiller and retain
    # the best exhibited construction; this does not affect the certificate.
    unique_choices = {
        tuple(point["selected_choices"]) for point in all_points.values()
    }
    primal_rows = [
        score_configuration(
            panel,
            choices,
            geometry=geometry,
            requested_rate=requested_rate,
            side_mode="charged",
            charge_basis=True,
            include_allocations=True,
        )
        for choices in sorted(unique_choices)
    ]
    best_primal = min(primal_rows, key=score_order)
    best_primal["search"] = {
        "kind": "best exact integer-bit row among Lagrange-dual selected configurations",
        "dual_unique_choice_vectors": len(unique_choices),
        "dual_theta_evaluations": len(all_points),
    }
    certificate = {
        "geometry": geometry,
        "requested_rate_bpw": requested_rate,
        "physical_rate_bpw": physical_rate,
        "physical_bits": physical_bits,
        "expert_option_count_each": len(banks[0].choices),
        "expert_count": len(banks),
        "complete_expert_local_grid_enumerated": all(
            len(bank.choices) == len(TAIL_COUNTS) ** ROLES for bank in banks
        ),
        "theta_evaluations": len(all_points),
        "best_theta": best_point["theta"],
        "lagrange_multiplier_distortion_per_bit": best_point[
            "lagrange_multiplier_distortion_per_bit"
        ],
        "selected_continuous_bits_at_dual_point": best_point["selected_continuous_bits"],
        "physical_bit_slack_at_dual_point": physical_bits
        - best_point["selected_continuous_bits"],
        "certified_lower_bound_sse": lower_sse,
        "certified_lower_bound_relative_mse": lower_mse,
        "certified_lower_bound_F": lower_f,
        "certifies_F_gt_0p8_for_complete_grid": bool(lower_f > TARGET_F),
        "selected_options_at_dual_point": best_point["selected_options"],
        "unique_dual_selected_choice_vectors": len(unique_choices),
        "best_dual_selected_primal_F": best_primal["F"],
        "derivation": (
            "For mu=2 ln(2) theta, sum_e min_(tail option,payload) "
            "[D_e + mu(side_e+payload_e)] + mu(common-B) is a weak-duality "
            "lower bound on every feasible charged configuration."
        ),
    }
    return certificate, best_primal


def attach_dual_certificates(panel: PanelStats, qwen: dict[str, Any]) -> None:
    certificates: dict[str, Any] = {}
    for geometry, variant in (
        ("raw", "charged_raw_bulk"),
        ("support_xklt", "charged_support_xklt_bulk"),
    ):
        banks = build_dual_option_banks(panel, geometry)
        geometry_rows: dict[str, Any] = {}
        for rate in RATES:
            rate_text = f"{rate:.2f}"
            certificate, primal = lagrange_dual_certificate(
                panel, banks, geometry=geometry, requested_rate=rate
            )
            coordinate = qwen["rates"][rate_text][variant]
            certificate["coordinate_descent_exhibited_F"] = coordinate["F"]
            if score_order(primal) < score_order(coordinate):
                qwen["rates"][rate_text][variant] = primal
                certificate["dual_selected_primal_improved_exhibited_row"] = True
            else:
                certificate["dual_selected_primal_improved_exhibited_row"] = False
            certificate["retained_exhibited_F"] = qwen["rates"][rate_text][variant]["F"]
            geometry_rows[rate_text] = certificate
            print(
                f"[{panel.label} {geometry} R={rate_text}] dual lower F="
                f"{certificate['certified_lower_bound_F']:.9f}; exhibited F="
                f"{certificate['retained_exhibited_F']:.9f}",
                flush=True,
            )
        certificates[geometry] = geometry_rows
        for table in panel.support_tables:
            table.cache.clear()
    qwen["dual_certificates"] = certificates


def matrix_candidate_report(panel: PanelStats) -> list[dict[str, Any]]:
    result = []
    for ordinal, (candidates, energy, receipt) in enumerate(
        zip(panel.matrix_candidates, panel.matrix_energies, panel.matrix_receipts, strict=True)
    ):
        result.append(
            {
                "matrix_ordinal": ordinal,
                "energy": energy,
                "receipt": receipt,
                "candidates": [row.public(energy) for row in candidates],
            }
        )
    return result


def run_panel_searches(panel: PanelStats, maximum_passes: int) -> dict[str, Any]:
    rates: dict[str, Any] = {}
    raw_seeds: dict[str, list[int]] = {}
    for rate in RATES:
        raw = coordinate_search(
            panel,
            geometry="raw",
            requested_rate=rate,
            side_mode="charged",
            charge_basis=True,
            maximum_passes=maximum_passes,
        )
        raw_seeds[f"{rate:.2f}"] = raw["choices"]
        support = coordinate_search(
            panel,
            geometry="support_xklt",
            requested_rate=rate,
            side_mode="charged",
            charge_basis=True,
            maximum_passes=maximum_passes,
            extra_seeds=(raw["choices"],),
        )
        free_basis = coordinate_search(
            panel,
            geometry="support_xklt",
            requested_rate=rate,
            side_mode="charged",
            charge_basis=False,
            maximum_passes=maximum_passes,
            extra_seeds=(support["choices"], raw["choices"]),
        )
        free_values = coordinate_search(
            panel,
            geometry="support_xklt",
            requested_rate=rate,
            side_mode="free_values",
            charge_basis=False,
            maximum_passes=maximum_passes,
            extra_seeds=(support["choices"],),
        )
        free_all = coordinate_search(
            panel,
            geometry="support_xklt",
            requested_rate=rate,
            side_mode="free_mask_values",
            charge_basis=False,
            maximum_passes=maximum_passes,
            extra_seeds=(free_values["choices"],),
        )
        rates[f"{rate:.2f}"] = {
            "charged_raw_bulk": raw,
            "charged_support_xklt_bulk": support,
            "charged_tail_free_exact_xklt_basis": free_basis,
            "mask_charged_tail_values_and_basis_free": free_values,
            "mask_values_and_basis_free_source_leaky_envelope": free_all,
        }
        print(
            f"[{panel.label} R={rate:.2f}] charged support F={support['F']:.9f}; "
            f"free-value F={free_values['F']:.9f}",
            flush=True,
        )
    return {"rates": rates, "raw_seeds": raw_seeds}


def run_control_searches(panel: PanelStats, maximum_passes: int) -> dict[str, Any]:
    rates: dict[str, Any] = {}
    for rate in RATES:
        support = coordinate_search(
            panel,
            geometry="support_xklt",
            requested_rate=rate,
            side_mode="charged",
            charge_basis=True,
            maximum_passes=maximum_passes,
        )
        rates[f"{rate:.2f}"] = support
        print(
            f"[{panel.label} R={rate:.2f}] charged support F={support['F']:.9f}",
            flush=True,
        )
    return {"replica": panel.control_replica, "rates": rates}


def summarize_decision(qwen: dict[str, Any], controls: list[dict[str, Any]]) -> dict[str, Any]:
    charged = []
    free_values = []
    for rate, bundle in qwen["rates"].items():
        for name in ("charged_raw_bulk", "charged_support_xklt_bulk"):
            charged.append({"rate": rate, "variant": name, **bundle[name]})
        free_values.append(
            {
                "rate": rate,
                "variant": "mask_charged_tail_values_and_basis_free",
                **bundle["mask_charged_tail_values_and_basis_free"],
            }
        )
    best = min(charged, key=score_order)
    best_free_values = min(free_values, key=score_order)
    control_at_rate = [
        control["rates"][str(best["rate"])] for control in controls
    ] if controls else []
    control_s = [float(row["s_bpw"]) for row in control_at_rate]
    matched_advantage = float(best["s_bpw"] - np.mean(control_s)) if control_s else None
    dual_rows = [
        row
        for geometry in qwen.get("dual_certificates", {}).values()
        for row in geometry.values()
    ]
    complete_dual_kill = bool(dual_rows) and all(
        bool(row["certifies_F_gt_0p8_for_complete_grid"]) for row in dual_rows
    )
    weakest_dual_f = (
        min(float(row["certified_lower_bound_F"]) for row in dual_rows)
        if dual_rows
        else None
    )
    return {
        "charged_target_reached": bool(best["F"] <= TARGET_F),
        "best_charged": {
            key: best[key]
            for key in (
                "rate",
                "variant",
                "physical_rate_bpw",
                "F",
                "s_bpw",
                "ideal_relative_mse",
                "target_mse",
                "side_bpw",
                "peeled_fraction",
                "peeled_energy_fraction",
                "choices",
                "read_ledger",
            )
        },
        "best_mask_charged_but_values_and_basis_free": {
            key: best_free_values[key]
            for key in (
                "rate",
                "physical_rate_bpw",
                "F",
                "s_bpw",
                "ideal_relative_mse",
                "side_bpw",
                "peeled_fraction",
                "peeled_energy_fraction",
                "choices",
            )
        },
        "matched_gaussian_control_s_bpw": control_s,
        "matched_structural_advantage_bpw": matched_advantage,
        "required_s_bpw": TARGET_S_BPW,
        "remaining_s_gap_bpw": TARGET_S_BPW - float(best["s_bpw"]),
        "complete_grid_dual_certificate_passed": complete_dual_kill,
        "weakest_certified_lower_bound_F_across_rates_and_geometries": weakest_dual_f,
        "hard_kill_tested_charged_family": bool(best["F"] > TARGET_F and complete_dual_kill),
        "hard_kill_even_with_free_tail_values": bool(best_free_values["F"] > TARGET_F),
        "gpu_finite_codec_followup_warranted": bool(
            best["F"] <= TARGET_F or not complete_dual_kill
        ),
        "rule": (
            "A hard kill requires both the best exhibited charged row to miss F<=0.8 "
            "and an F>0.8 Lagrange-dual lower bound for the complete frozen grid at "
            "every rate and residual geometry."
        ),
    }


def self_test() -> None:
    assert ceil_log2_binomial(5, 0) == 0
    assert ceil_log2_binomial(5, 1) == 3
    assert ceil_log2_binomial(5, 2) == 4
    assert huffman_payload_bits([5]) == 0
    assert huffman_payload_bits([1, 1]) == 2
    assert huffman_payload_bits([5, 2, 1]) == 11
    words = np.asarray([0x3F80, 0x3F80, 0xBF80, 0x4000], dtype=np.uint16)
    histogram = np.bincount(words.astype(np.int64), minlength=1 << 16)
    value = best_value_code(histogram)
    assert value["total_bits"] <= 16 * len(words)
    components = [
        Component("a", 0, 100, 100.0),
        Component("b", 1, 100, 25.0),
    ]
    row = integer_waterfill(components, 200)
    assert row["payload_bits"] == 200
    assert sum(item["payload_bits"] for item in row["allocations"]) == 200
    values = np.asarray([3.0, -3.0, 2.0, 2.0, 1.0], dtype=np.float32)
    order = stable_tail_order(values, np)
    assert order.tolist() == [0, 1, 2, 3, 4]
    print("tail_peeling_composite self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="cupy")
    parser.add_argument("--control-replicates", type=int, default=4)
    parser.add_argument("--maximum-coordinate-passes", type=int, default=5)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol_lock.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.source_lock is None or args.output is None:
        parser.error("--source-lock and --output are required outside --self-test")
    if not 0 <= args.control_replicates <= 8:
        raise ValueError("control replicate count must be in [0,8]")
    if not 1 <= args.maximum_coordinate_passes <= 12:
        raise ValueError("coordinate passes must be in [1,12]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")

    started = time.time()
    xp, backend_version = get_backend(args.backend)
    source_lock = args.source_lock.resolve(strict=True)
    source_root = (
        args.source_root.resolve(strict=True)
        if args.source_root is not None
        else source_lock.parent.resolve(strict=True)
    )
    protocol = args.protocol.resolve(strict=True)
    protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
    if tuple(protocol_data["tail_counts"]) != TAIL_COUNTS:
        raise ValueError("protocol tail-count grid differs from implementation")
    if tuple(float(x) for x in protocol_data["physical_rates_bpw"]) != RATES:
        raise ValueError("protocol rate grid differs from implementation")
    if str(protocol_data["pinned_source_lock_sha256"]) != sha256_file(source_lock):
        raise ValueError("protocol is not bound to the supplied source lock")

    qwen_panel, source_lock_data = build_source_panel(source_lock, source_root, xp)
    qwen_result = run_panel_searches(qwen_panel, args.maximum_coordinate_passes)
    attach_dual_certificates(qwen_panel, qwen_result)
    control_results = []
    control_candidate_reports = []
    checkpoint = str(source_lock_data.get("checkpoint", "unknown"))
    for replica in range(args.control_replicates):
        panel = build_control_panel(qwen_panel, replica, checkpoint, xp)
        control_results.append(run_control_searches(panel, args.maximum_coordinate_passes))
        control_candidate_reports.append(
            {
                "replica": replica,
                "matrix_candidates": matrix_candidate_report(panel),
            }
        )
    decision = summarize_decision(qwen_result, control_results)

    script_path = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema": "qwen_sparse_tail_peeling_composite_oracle_v1",
        "scope": {
            "checkpoint": checkpoint,
            "matrix_count": MATRICES,
            "expert_count": EXPERTS,
            "canonical_matrix_shape": [ROWS, COLS],
            "panel_values": PANEL_VALUES,
            "physical_rates_bpw": list(RATES),
            "target_F": TARGET_F,
            "target_s_bpw": TARGET_S_BPW,
            "target_formula": "MSE <= 0.8 * 2^(-2R_actual)",
            "backend": args.backend,
            "backend_version": backend_version,
        },
        "method": {
            "tail_selection": (
                "per-matrix stable top absolute BF16 value; ties use canonical flat ordinal"
            ),
            "mask_code": (
                "exact ceil(log2 C(1572864,k))-bit combinatorial-number-system index"
            ),
            "value_code": (
                "lossless BF16 words; best of literal16, canonical word Huffman, "
                "magnitude+sign Huffman, and sign/exponent/mantissa Huffman; 2-bit mode "
                "is inside each charged descriptor"
            ),
            "robust_bulk": (
                "known support is partitioned by the 7 nonempty role masks; an exact "
                "source-fitted KLT is applied inside each support pattern, followed by "
                "a procedural orthogonal RHT and an ideal Gaussian polar-lattice test channel"
            ),
            "one_joint_waterfill": (
                "all residual components across all experts share one water level; real "
                "allocations are closed to exactly the integer physical payload bits by "
                "largest exact one-bit marginal reduction"
            ),
            "optimism": [
                "ideal asymptotic Gaussian residual RD with no finite polar/lattice shaping loss",
                "source-fitted KLT basis is treated as exact after a Q15-sized angle ledger",
                "no RHT padding or block-boundary loss",
                "tail values are exact, so their source-domain error is zero",
            ],
        },
        "physical_layout": {
            "global_header_bits": GLOBAL_HEADER_BITS,
            "route_table_bits": ROUTE_TABLE_BITS,
            "expert_header_bits_each": EXPERT_HEADER_BITS,
            "residual_directory_bits_each": RESIDUAL_DIRECTORY_BITS,
            "matrix_descriptor_bits_each": MATRIX_DESCRIPTOR_BITS,
            "support_pattern_mode_bits_each_expert": SUPPORT_PATTERN_MODE_BITS,
            "xklt_angle_bits_each": ANGLE_BITS,
            "expert_local_frames": True,
        },
        "protocol": {
            "path": str(protocol),
            "sha256": sha256_file(protocol),
            "contents": protocol_data,
        },
        "source_audit": {
            "source_lock_path": str(source_lock),
            "source_lock_file_sha256": sha256_file(source_lock),
            "source_lock_internal_sha256": source_lock_data.get("lock_sha256"),
            "all_source_hashes_matched": all(
                row["declared_sha256"] == row["observed_sha256"]
                for row in qwen_panel.matrix_receipts
            ),
            "receipts": qwen_panel.matrix_receipts,
            "panel_source_energy": qwen_panel.total_energy,
        },
        "qwen_matrix_candidates": matrix_candidate_report(qwen_panel),
        "qwen": qwen_result,
        "matched_gaussian_controls": {
            "replicates": args.control_replicates,
            "construction": (
                "PCG64DXSM normal -> BF16 RNE symbols, then one conceptual exact energy "
                "normalization per matrix; every control repeats tail search, side coding, "
                "support XKLT and joint integer-bit waterfill"
            ),
            "results": control_results,
            "candidate_ledgers": control_candidate_reports,
        },
        "decision": decision,
        "audit": {
            "script_path": str(script_path),
            "script_sha256": sha256_file(script_path),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "elapsed_seconds": time.time() - started,
        },
        "claim_boundary": (
            "This is a charged source-locked ideal-RD architecture oracle, not an emitted "
            "codec and not achieved reconstructed-weight MSE. A charged failure is a valid "
            "early stop for this exact-tail/support-XKLT/RHT-polar family because every "
            "omitted finite-code effect favours it. Free-side rows are diagnostic only."
        ),
    }
    write_sealed_json(args.output, report)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
