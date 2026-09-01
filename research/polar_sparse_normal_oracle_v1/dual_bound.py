"""Pure-stdlib Lagrangian lower bound for PSNO-v1 support curves."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


AUTHORITY_GRANTED = False
RATE = 2.5
TARGET_F = 0.8
PANEL_VALUES = 18 * 768 * 2048


def gaussian_component_phi(
    dimension: float, energy: float, multiplier: float
) -> tuple[float, float]:
    """Return min_r D(r)+lambda*r and its minimizing nonnegative rate."""

    if not (dimension > 0.0 and energy >= 0.0 and multiplier > 0.0):
        raise ValueError("invalid Gaussian component")
    if energy == 0.0:
        return 0.0, 0.0
    critical = 2.0 * math.log(2.0) * energy / dimension
    if multiplier >= critical:
        return energy, 0.0
    rate = dimension / (2.0 * math.log(2.0)) * math.log(critical / multiplier)
    distortion = multiplier * dimension / (2.0 * math.log(2.0))
    return distortion + multiplier * rate, rate


def dual_at_multiplier(
    records: Sequence[dict[str, Any]],
    normal_curves: Sequence[Sequence[dict[str, Any]]],
    *,
    base_side_bpw: float,
    value_bits_per_symbol: float,
    multiplier: float,
) -> dict[str, Any]:
    """Evaluate one valid dual lower bound over every discrete support option.

    Each curve row contains absolute `residual_energy`, integer `removed_dof`,
    integer `support_bits`, integer `value_symbols`, and an identifying `name`.
    The minimum over rows is exact at the supplied multiplier.  Consequently
    every returned `dual_distortion` is a valid lower bound even when the
    multiplier grid misses the dual maximum.
    """

    if len(records) != 18 or len(normal_curves) != 18:
        raise ValueError("18 matrix records and curves required")
    if not (0.0 <= base_side_bpw < RATE and value_bits_per_symbol >= 0.0):
        raise ValueError("invalid side/value ledger")
    total_source_energy = math.fsum(float(row["source_energy"]) for row in records)
    value = -multiplier * (RATE - base_side_bpw)
    payload_rate = 0.0
    support_value_rate = 0.0
    selected: list[dict[str, Any]] = []

    for record in records:
        objective, component_rate = gaussian_component_phi(
            float(record["model_dof"]) / PANEL_VALUES,
            float(record["model_energy"]) / total_source_energy,
            multiplier,
        )
        value += objective
        payload_rate += component_rate

    for ordinal, (record, curve) in enumerate(zip(records, normal_curves, strict=True)):
        if not curve:
            raise ValueError(f"empty normal curve {ordinal}")
        best: tuple[float, float, float, dict[str, Any]] | None = None
        for option in curve:
            residual = float(option["residual_energy"])
            removed = int(option["removed_dof"])
            support_bits = int(option["support_bits"])
            symbols = int(option["value_symbols"])
            if residual < 0.0 or not 0 <= removed < int(record["normal_dof"]):
                raise ValueError(f"invalid normal option {ordinal}")
            if support_bits < 0 or symbols < 0:
                raise ValueError(f"negative rate option {ordinal}")
            side_rate = (support_bits + value_bits_per_symbol * symbols) / PANEL_VALUES
            if residual == 0.0:
                component_objective, component_rate = 0.0, 0.0
            else:
                component_objective, component_rate = gaussian_component_phi(
                    max(1, int(record["normal_dof"]) - removed) / PANEL_VALUES,
                    residual / total_source_energy,
                    multiplier,
                )
            objective = component_objective + multiplier * side_rate
            candidate = (objective, component_rate, side_rate, option)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        value += best[0]
        payload_rate += best[1]
        support_value_rate += best[2]
        selected.append(
            {
                "matrix_ordinal": ordinal,
                "name": best[3].get("name"),
                "k": best[3].get("k"),
                "payload_rate_bpw": best[1],
                "support_value_rate_bpw": best[2],
            }
        )

    # Round down rather than up before the hard decision.  This allowance is
    # far larger than accumulated binary64 error for 36 components.
    conservative = max(0.0, value - 1e-12)
    return {
        "multiplier": multiplier,
        "raw_dual_distortion": value,
        "dual_distortion": conservative,
        "dual_F_lower_bound": conservative * 2.0 ** (2.0 * RATE),
        "hard_kill_at_this_multiplier": conservative * 2.0 ** (2.0 * RATE) > TARGET_F,
        "implied_total_rate_bpw": base_side_bpw + support_value_rate + payload_rate,
        "support_value_rate_bpw": support_value_rate,
        "payload_rate_bpw": payload_rate,
        "selected_options": selected,
        "authorization": False,
    }


def scan_dual(
    records: Sequence[dict[str, Any]],
    normal_curves: Sequence[Sequence[dict[str, Any]]],
    multipliers: Iterable[float],
    *,
    base_side_bpw: float,
    value_bits_per_symbol: float,
) -> dict[str, Any]:
    rows = [
        dual_at_multiplier(
            records,
            normal_curves,
            base_side_bpw=base_side_bpw,
            value_bits_per_symbol=value_bits_per_symbol,
            multiplier=float(multiplier),
        )
        for multiplier in multipliers
    ]
    if not rows:
        raise ValueError("empty multiplier grid")
    best = max(rows, key=lambda row: float(row["dual_distortion"]))
    return {
        "schema": "polar_sparse_normal_dual_scan_v1",
        "evaluated_multipliers": len(rows),
        "best": best,
        "hard_kill": bool(best["hard_kill_at_this_multiplier"]),
        "claim": (
            "Any evaluated dual value is a lower bound; a missed maximum can only weaken, "
            "never create, a hard kill."
        ),
        "authorization": False,
    }


def source_only_status() -> dict[str, Any]:
    return {
        "status": "SOURCE_ONLY_DUAL_MATH_NO_PAYLOAD",
        "tensor_access": False,
        "third_party_imports": False,
        "gpu_execution": False,
        "authorization": False,
    }


if __name__ == "__main__":
    raise SystemExit("PSNO_V1_SOURCE_ONLY_DUAL_MATH")
