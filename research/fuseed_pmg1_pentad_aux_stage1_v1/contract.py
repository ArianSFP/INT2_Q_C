#!/usr/bin/env python3
"""Source-only contracts for the fixed PMG1 pentad auxiliary stage-1 gate.

This module is deliberately standard-library-only.  It contains the fixed
hypothesis, arithmetic ledger, strict JSON parser, FP16 codec, and the small
deterministic linear algebra used by the future independently dispatched
producer.  Importing it performs no filesystem, payload, accelerator, or
network action.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


DESIGN_SCHEMA = "fuseed-pmg1-fixed-pentad-aux-stage1-design-v1"
RESULT_SCHEMA = "fuseed-pmg1-fixed-pentad-aux-stage1-result-v1"
MANIFEST_SCHEMA = "fuseed-pmg1-fixed-pentad-aux-stage1-source-manifest-v1"
STATUS = "FROZEN_SOURCE_ONLY_AWAITING_INDEPENDENT_AUDIT_NO_PAYLOAD_AUTHORITY"

# These are public constants of one fixed hypothesis.  They are not claimed
# to be authenticated winners of an exhaustive u32 search.
SEEDS_U32 = (
    3306464084,
    235286348,
    2174751347,
    256779041,
    118211936,
)
SELECTION_EXPERTS = (0, 8, 16, 32, 40, 48, 64, 72, 80, 96, 104, 112)
ROLES = ("up", "down")
IDENTITY_COUNT = 23
FIT_COORDINATES = 2048
SCORE_COORDINATES = 2048
CONTROL_SEEDS = tuple(range(26090100, 26090116))

BASE_FACTOR = 0.9888693569009007
TARGET_FACTOR = 0.8
SIDE_BYTES_SIX_EXPERTS = 320
WEIGHTS_SIX_EXPERTS = 28_311_552
SIDE_BPW = 8.0 * SIDE_BYTES_SIX_EXPERTS / WEIGHTS_SIX_EXPERTS
REQUIRED_CAPTURE = 1.0 - TARGET_FACTOR / (BASE_FACTOR * 2.0 ** (2.0 * SIDE_BPW))
BASE_PAGE_READ_AMPLIFICATION = 1.1694444444444445
CONSERVATIVE_NEW_PAGES = 2
BYTES_PER_EXPERT_AT_2P5 = 1_474_560
CONSERVATIVE_READ_AMPLIFICATION = (
    BASE_PAGE_READ_AMPLIFICATION
    + CONSERVATIVE_NEW_PAGES * 4096.0 / BYTES_PER_EXPERT_AT_2P5
)

RIDGE_EXPONENT = -20
CONDITION_LIMIT = 2**20
JACOBI_SWEEPS = 48


class ContractError(RuntimeError):
    """A frozen source, arithmetic, protocol, or numerical invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"non-canonical JSON value: {exc}") from exc


def _reject_constant(text: str) -> None:
    raise ContractError(f"non-finite JSON constant: {text}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ContractError(f"non-finite JSON number: {text}")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: bytes | str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"invalid JSON: {exc}") from exc


def identity_set() -> tuple[tuple[int, str], ...]:
    rows = tuple(
        (expert, role)
        for expert in SELECTION_EXPERTS
        for role in ROLES
        if not (expert == 0 and role == "up")
    )
    require(len(rows) == IDENTITY_COUNT, "fixed identity cardinality")
    return rows


def fp16_word(value: float) -> int:
    require(math.isfinite(value), "non-finite coefficient before FP16")
    try:
        raw = struct.pack("<e", float(value))
    except (OverflowError, struct.error) as exc:
        raise ContractError(f"coefficient is not finite FP16: {value}") from exc
    word = int.from_bytes(raw, "little")
    require(word != 0x8000, "negative-zero FP16 coefficient")
    decoded = struct.unpack("<e", raw)[0]
    require(math.isfinite(decoded), "non-finite decoded FP16 coefficient")
    return word


def decode_fp16_word(word: int) -> float:
    require(0 <= int(word) <= 0xFFFF, "FP16 word range")
    require(int(word) != 0x8000, "negative-zero FP16 coefficient")
    value = struct.unpack("<e", int(word).to_bytes(2, "little"))[0]
    require(math.isfinite(value), "non-finite decoded FP16 coefficient")
    return float(value)


def _finite_matrix(rows: Sequence[Sequence[float]], columns: int) -> list[list[float]]:
    result = [[float(value) for value in row] for row in rows]
    require(len(result) > columns, "insufficient fit rows")
    require(all(len(row) == columns for row in result), "matrix width")
    require(all(math.isfinite(value) for row in result for value in row), "finite matrix")
    return result


def _finite_vector(values: Sequence[float], expected: int | None = None) -> list[float]:
    result = [float(value) for value in values]
    if expected is not None:
        require(len(result) == expected, "vector length")
    require(result and all(math.isfinite(value) for value in result), "finite vector")
    return result


def _means(matrix: Sequence[Sequence[float]], target: Sequence[float]) -> tuple[list[float], float]:
    n = len(matrix)
    p = len(matrix[0])
    xsum = [0.0] * p
    ysum = 0.0
    for row, value in zip(matrix, target, strict=True):
        for column in range(p):
            xsum[column] = float(xsum[column] + float(row[column]))
        ysum = float(ysum + float(value))
    return [float(value / n) for value in xsum], float(ysum / n)


def _gram_rhs(
    matrix: Sequence[Sequence[float]],
    target: Sequence[float],
    xmean: Sequence[float],
    ymean: float,
) -> tuple[list[list[float]], list[float]]:
    p = len(xmean)
    gram = [[0.0] * p for _ in range(p)]
    rhs = [0.0] * p
    for row, value in zip(matrix, target, strict=True):
        centered = [float(float(row[column]) - float(xmean[column])) for column in range(p)]
        yc = float(float(value) - ymean)
        for left in range(p):
            rhs[left] = float(rhs[left] + float(centered[left] * yc))
            for right in range(left + 1):
                gram[left][right] = float(
                    gram[left][right] + float(centered[left] * centered[right])
                )
    for left in range(p):
        for right in range(left):
            gram[right][left] = gram[left][right]
    return gram, rhs


def _jacobi_eigenvalues(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Fixed-order binary64 Jacobi eigenvalue approximation for a 5x5 SPD matrix."""
    p = len(matrix)
    values = [[float(value) for value in row] for row in matrix]
    require(p == len(SEEDS_U32) and all(len(row) == p for row in values), "Jacobi geometry")
    for _sweep in range(JACOBI_SWEEPS):
        for left in range(p - 1):
            for right in range(left + 1, p):
                apq = values[left][right]
                if apq == 0.0:
                    continue
                app = values[left][left]
                aqq = values[right][right]
                tau = float((aqq - app) / (2.0 * apq))
                sign = 1.0 if tau >= 0.0 else -1.0
                tangent = float(sign / (abs(tau) + math.sqrt(1.0 + tau * tau)))
                cosine = float(1.0 / math.sqrt(1.0 + tangent * tangent))
                sine = float(tangent * cosine)
                for index in range(p):
                    if index in (left, right):
                        continue
                    ail = values[index][left]
                    air = values[index][right]
                    new_left = float(cosine * ail - sine * air)
                    new_right = float(sine * ail + cosine * air)
                    values[index][left] = values[left][index] = new_left
                    values[index][right] = values[right][index] = new_right
                values[left][left] = float(
                    cosine * cosine * app
                    - 2.0 * sine * cosine * apq
                    + sine * sine * aqq
                )
                values[right][right] = float(
                    sine * sine * app
                    + 2.0 * sine * cosine * apq
                    + cosine * cosine * aqq
                )
                values[left][right] = values[right][left] = 0.0
    eigenvalues = sorted(float(values[index][index]) for index in range(p))
    require(all(math.isfinite(value) for value in eigenvalues), "finite Jacobi eigenvalues")
    return eigenvalues


def _cholesky_solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    p = len(matrix)
    lower = [[0.0] * p for _ in range(p)]
    for row in range(p):
        for column in range(row + 1):
            value = float(matrix[row][column])
            for index in range(column):
                value = float(value - float(lower[row][index] * lower[column][index]))
            if row == column:
                require(value > 0.0 and math.isfinite(value), "nonpositive Cholesky pivot")
                lower[row][column] = math.sqrt(value)
            else:
                lower[row][column] = float(value / lower[column][column])
    forward = [0.0] * p
    for row in range(p):
        value = float(rhs[row])
        for index in range(row):
            value = float(value - float(lower[row][index] * forward[index]))
        forward[row] = float(value / lower[row][row])
    solution = [0.0] * p
    for row in range(p - 1, -1, -1):
        value = forward[row]
        for index in range(row + 1, p):
            value = float(value - float(lower[index][row] * solution[index]))
        solution[row] = float(value / lower[row][row])
    require(all(math.isfinite(value) for value in solution), "finite Cholesky solution")
    return solution


def fit_decoded_fp16(
    anchors: Sequence[Sequence[float]], target: Sequence[float]
) -> dict[str, Any]:
    """Fit five anchors plus intercept and return only decoded FP16 parameters."""
    matrix = _finite_matrix(anchors, len(SEEDS_U32))
    values = _finite_vector(target, len(matrix))
    xmean, ymean = _means(matrix, values)
    gram, rhs = _gram_rhs(matrix, values, xmean, ymean)
    trace = math.fsum(gram[index][index] for index in range(len(SEEDS_U32)))
    require(trace > 0.0 and math.isfinite(trace), "positive Gram trace")
    ridge = math.ldexp(trace, RIDGE_EXPONENT) / len(SEEDS_U32)
    system = [
        [
            float(gram[row][column] + (ridge if row == column else 0.0))
            for column in range(len(SEEDS_U32))
        ]
        for row in range(len(SEEDS_U32))
    ]
    eigenvalues = _jacobi_eigenvalues(system)
    require(eigenvalues[0] > 0.0, "positive numerical rank")
    condition = float(eigenvalues[-1] / eigenvalues[0])
    require(condition <= CONDITION_LIMIT, "pentad fit condition exceeds limit")
    beta = _cholesky_solve(system, rhs)
    mu = ymean
    for coefficient, mean in zip(beta, xmean, strict=True):
        mu = float(mu - float(coefficient * mean))
    words = tuple(fp16_word(value) for value in (*beta, mu))
    decoded = tuple(decode_fp16_word(word) for word in words)
    return {
        "beta": decoded[: len(SEEDS_U32)],
        "mu": decoded[len(SEEDS_U32)],
        "fp16_words": words,
        "fit_target_mean": ymean,
        "condition": condition,
        "ridge": ridge,
        "eigenvalues": tuple(eigenvalues),
    }


def score_decoded_fit(
    fit: Mapping[str, Any],
    anchors: Sequence[Sequence[float]],
    target: Sequence[float],
) -> dict[str, float]:
    matrix = _finite_matrix(anchors, len(SEEDS_U32))
    values = _finite_vector(target, len(matrix))
    beta = tuple(float(value) for value in fit["beta"])
    mu = float(fit["mu"])
    require(len(beta) == len(SEEDS_U32), "decoded beta count")
    sse = 0.0
    energy = 0.0
    centered_baseline_sse = 0.0
    fit_target_mean = float(fit["fit_target_mean"])
    require(math.isfinite(fit_target_mean), "finite fit target mean")
    for row, value in zip(matrix, values, strict=True):
        reconstruction = mu
        for coefficient, anchor in zip(beta, row, strict=True):
            reconstruction = float(reconstruction + float(coefficient * float(anchor)))
        error = float(value - reconstruction)
        sse = float(sse + float(error * error))
        energy = float(energy + float(value * value))
        centered_error = float(value - fit_target_mean)
        centered_baseline_sse = float(
            centered_baseline_sse + float(centered_error * centered_error)
        )
    require(sse >= 0.0 and energy > 0.0, "valid score energy")
    require(centered_baseline_sse > 0.0, "positive centered baseline")
    require(
        math.isfinite(sse)
        and math.isfinite(energy)
        and math.isfinite(centered_baseline_sse),
        "finite score energy",
    )
    return {
        "sse": sse,
        "source_energy": energy,
        "centered_baseline_sse": centered_baseline_sse,
        "capture": 1.0 - sse / energy,
        "centered_capture": 1.0 - sse / centered_baseline_sse,
    }


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    require(records, "nonempty record set")
    sse = math.fsum(float(row["sse"]) for row in records)
    energy = math.fsum(float(row["source_energy"]) for row in records)
    require(sse >= 0.0 and energy > 0.0, "aggregate energy")
    return {"sse": sse, "source_energy": energy, "capture": 1.0 - sse / energy}


def role_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for role in ROLES:
        rows = [row for row in records if row["role"] == role]
        result[role] = aggregate_records(rows)
    return result


def expert_jackknife(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Delete one complete expert (both roles when present), never one matrix."""
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[int(row["expert"])].append(row)
    require(tuple(sorted(grouped)) == SELECTION_EXPERTS, "jackknife expert identities")
    total = aggregate_records(records)
    values = []
    for expert in SELECTION_EXPERTS:
        omitted = aggregate_records(grouped[expert])
        denominator = total["source_energy"] - omitted["source_energy"]
        numerator = total["sse"] - omitted["sse"]
        require(denominator > 0.0 and numerator >= 0.0, "delete-expert energy")
        values.append(1.0 - numerator / denominator)
    mean = math.fsum(values) / len(values)
    variance_sum = math.fsum((value - mean) ** 2 for value in values)
    standard_error = math.sqrt((len(values) - 1.0) / len(values) * variance_sum)
    point = total["capture"]
    return {
        "delete_expert_values": values,
        "mean": mean,
        "standard_error": standard_error,
        "lower_3se": point - 3.0 * standard_error,
        "upper_3se": point + 3.0 * standard_error,
    }


def decision(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_records(records)
    roles = role_aggregates(records)
    uncertainty = expert_jackknife(records)
    all_identity_positive = all(float(row["capture"]) > 0.0 for row in records)
    all_roles_positive = all(row["capture"] > 0.0 for row in roles.values())
    survives = (
        uncertainty["lower_3se"] >= REQUIRED_CAPTURE
        and all_identity_positive
        and all_roles_positive
    )
    return {
        "status": (
            "SURVIVES_FIXED_PENTAD_AUXILIARY_STAGE1_ONLY"
            if survives
            else "HARD_KILL_FIXED_PENTAD_AUXILIARY_STAGE1_NO_TUPLE_RETRY"
        ),
        "aggregate": aggregate,
        "roles": roles,
        "uncertainty": uncertainty,
        "required_capture": REQUIRED_CAPTURE,
        "all_identity_positive": all_identity_positive,
        "all_roles_positive": all_roles_positive,
        "survives": survives,
    }


def physical_ledger() -> dict[str, Any]:
    return {
        "six_expert_side_bytes": SIDE_BYTES_SIX_EXPERTS,
        "six_expert_side_bpw": SIDE_BPW,
        "total_rate_bpw_if_residual_is_debited": 2.5,
        "metadata_adjusted_required_capture": REQUIRED_CAPTURE,
        "base_page_read_amplification": BASE_PAGE_READ_AMPLIFICATION,
        "conservative_new_pages_per_route": CONSERVATIVE_NEW_PAGES,
        "conservative_page_read_amplification": CONSERVATIVE_READ_AMPLIFICATION,
        "strictly_below_2x": CONSERVATIVE_READ_AMPLIFICATION < 2.0,
        "claim": "planning ledger only; no pentad container exists",
    }


def ensure_all_finite(values: Iterable[float]) -> None:
    require(all(math.isfinite(float(value)) for value in values), "non-finite numeric value")
