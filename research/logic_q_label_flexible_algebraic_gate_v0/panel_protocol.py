#!/usr/bin/env python3
"""Whole-component holdout and matched-Gaussian source-only protocol."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    from logicq_core import (
        CONTROL_SEEDS,
        ROLE_IDS,
        TARGET_F,
        EncodedComponent,
        LogicQError,
        encode_gf2_component,
        encode_literal_component,
        encode_rm1_component,
        encode_romdd_component,
        pack_expert,
        require,
    )
except ImportError:  # source-only importlib loading in hostile tests
    import importlib.util
    import sys
    from pathlib import Path

    _path = Path(__file__).resolve().parent / "logicq_core.py"
    _spec = importlib.util.spec_from_file_location("logicq_core", _path)
    if _spec is None or _spec.loader is None:
        raise
    _module = importlib.util.module_from_spec(_spec)
    sys.modules.setdefault("logicq_core", _module)
    _spec.loader.exec_module(_module)
    CONTROL_SEEDS = _module.CONTROL_SEEDS
    ROLE_IDS = _module.ROLE_IDS
    TARGET_F = _module.TARGET_F
    EncodedComponent = _module.EncodedComponent
    LogicQError = _module.LogicQError
    encode_gf2_component = _module.encode_gf2_component
    encode_literal_component = _module.encode_literal_component
    encode_rm1_component = _module.encode_rm1_component
    encode_romdd_component = _module.encode_romdd_component
    pack_expert = _module.pack_expert
    require = _module.require


SPLIT_DOMAIN = b"logic-q-label-flexible-algebraic-gate-v0-split\0"


@dataclass(frozen=True)
class PanelRow:
    layer: str
    slot: str
    role: str
    rows: int
    cols: int


def _hash_key(kind: str, value: str) -> bytes:
    return hashlib.sha256(SPLIT_DOMAIN + kind.encode("ascii") + b"\0"
                          + value.encode("utf-8")).digest()


def validate_panel(rows: Iterable[PanelRow]) -> tuple[PanelRow, ...]:
    panel = tuple(rows)
    require(panel, "nonempty panel")
    keys = [(row.layer, row.slot, row.role) for row in panel]
    require(len(keys) == len(set(keys)), "unique layer/slot/role rows")
    layers = sorted({row.layer for row in panel})
    require(len(layers) >= 10, "minimum ten whole layers")
    expected_slots = None
    expected_shapes: dict[str, tuple[int, int]] | None = None
    for layer in layers:
        layer_rows = [row for row in panel if row.layer == layer]
        slots = sorted({row.slot for row in layer_rows})
        require(slots, "layer has expert slots")
        if expected_slots is None:
            expected_slots = slots
        require(slots == expected_slots, "identical expert-slot universe")
        shapes: dict[str, tuple[int, int]] = {}
        for slot in slots:
            role_rows = [row for row in layer_rows if row.slot == slot]
            require(sorted(row.role for row in role_rows) == sorted(ROLE_IDS),
                    "exact three semantic roles")
            for row in role_rows:
                require(row.rows > 0 and row.cols > 0, "positive panel shape")
                previous = shapes.setdefault(row.role, (row.rows, row.cols))
                require(previous == (row.rows, row.cols),
                        "one role shape within layer")
        if expected_shapes is None:
            expected_shapes = shapes
        require(shapes == expected_shapes, "same canonical role shapes across layers")
    return panel


def whole_component_split(rows: Iterable[PanelRow], *,
                          test_layer_count: int = 5,
                          validation_slot_count: int | None = None) -> dict[str, Any]:
    panel = validate_panel(rows)
    layers = sorted({row.layer for row in panel},
                    key=lambda value: (_hash_key("layer", value), value))
    slots = sorted({row.slot for row in panel},
                   key=lambda value: (_hash_key("slot", value), value))
    require(5 <= test_layer_count < len(layers), "whole-test-layer count")
    if validation_slot_count is None:
        validation_slot_count = max(1, ceil_fraction(len(slots), 4))
    require(1 <= validation_slot_count < len(slots),
            "whole-validation-slot count")
    test_layers = frozenset(layers[:test_layer_count])
    validation_slots = frozenset(slots[:validation_slot_count])
    partitions = {"train": [], "validation": [], "test": []}
    for row in panel:
        if row.layer in test_layers:
            partitions["test"].append(row)
        elif row.slot in validation_slots:
            partitions["validation"].append(row)
        else:
            partitions["train"].append(row)
    require(all(partitions.values()), "nonempty nested partitions")
    train_keys = {(row.layer, row.slot, row.role) for row in partitions["train"]}
    validation_keys = {(row.layer, row.slot, row.role)
                       for row in partitions["validation"]}
    test_keys = {(row.layer, row.slot, row.role) for row in partitions["test"]}
    require(not train_keys & validation_keys and not train_keys & test_keys and
            not validation_keys & test_keys, "disjoint nested partitions")
    require(not ({row.layer for row in partitions["test"]}
                 & {row.layer for row in partitions["train"] +
                    partitions["validation"]}), "whole test layers untouched")
    require(not ({row.slot for row in partitions["validation"]}
                 & {row.slot for row in partitions["train"]}),
            "whole validation slots absent from training")
    return {
        "train": tuple(partitions["train"]),
        "validation": tuple(partitions["validation"]),
        "test": tuple(partitions["test"]),
        "test_layers": tuple(sorted(test_layers)),
        "validation_slots": tuple(sorted(validation_slots)),
        "minimum_whole_test_layer_clusters": 5,
        "selection_uses_test": False,
    }


def ceil_fraction(value: int, denominator: int) -> int:
    require(value >= 0 and denominator > 0, "ceil fraction")
    return (value + denominator - 1) // denominator


def moment_matched_gaussian(np: Any, source: Any, *, block_size: int,
                            seed: int, component_ordinal: int) -> tuple[Any, dict[str, Any]]:
    values = np.asarray(source, dtype=np.float64).reshape(-1)
    require(values.size > 0 and values.size % block_size == 0 and
            np.all(np.isfinite(values)), "control source geometry")
    require(seed in CONTROL_SEEDS and component_ordinal >= 0,
            "frozen control seed/ordinal")
    generated = np.empty_like(values)
    maximum_mean_error = 0.0
    maximum_centered_relative_error = 0.0
    block_rows = []
    for block_index, start in enumerate(range(0, values.size, block_size)):
        block = values[start:start + block_size]
        mean = float(np.mean(block, dtype=np.float64))
        centered = block - mean
        centered_sse = float(np.sum(centered * centered, dtype=np.float64))
        # The source/control distinction is not a probability-model key. This
        # counter exists only in the offline matched-control generator.
        counter_material = (f"{seed}:{component_ordinal}:{block_index}"
                            .encode("ascii"))
        counter = int.from_bytes(hashlib.sha256(counter_material).digest()[:8],
                                 "big")
        rng = np.random.Generator(np.random.PCG64(counter))
        z = rng.standard_normal(block_size, dtype=np.float64)
        z -= np.mean(z, dtype=np.float64)
        z_sse = float(np.sum(z * z, dtype=np.float64))
        require(z_sse > 0.0, "nondegenerate Gaussian block")
        if centered_sse == 0.0:
            control = np.full(block_size, mean, dtype=np.float64)
        else:
            control = mean + z * math.sqrt(centered_sse / z_sse)
            # One deterministic recenter/rescale pass keeps binary64 moment
            # drift far below the gate tolerance without touching labels.
            control -= float(np.mean(control, dtype=np.float64)) - mean
            control_centered = control - mean
            observed = float(np.sum(control_centered * control_centered,
                                    dtype=np.float64))
            control = mean + control_centered * math.sqrt(centered_sse / observed)
        generated[start:start + block_size] = control
        observed_mean = float(np.mean(control, dtype=np.float64))
        observed_centered = float(np.sum((control - observed_mean) ** 2,
                                         dtype=np.float64))
        mean_error = abs(observed_mean - mean)
        relative_error = (abs(observed_centered - centered_sse)
                          / max(centered_sse, 2.0 ** -1022))
        maximum_mean_error = max(maximum_mean_error, mean_error)
        maximum_centered_relative_error = max(maximum_centered_relative_error,
                                               relative_error)
        block_rows.append({
            "block": block_index,
            "source_mean_f64_hex": float(mean).hex(),
            "source_centered_sse_f64_hex": float(centered_sse).hex(),
            "control_mean_f64_hex": float(observed_mean).hex(),
            "control_centered_sse_f64_hex": float(observed_centered).hex(),
        })
    receipt = {
        "seed": seed,
        "component_ordinal": component_ordinal,
        "block_size": block_size,
        "blocks": values.size // block_size,
        "maximum_mean_absolute_error": maximum_mean_error,
        "maximum_centered_sse_relative_error": maximum_centered_relative_error,
        "source_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "control_sha256": hashlib.sha256(generated.tobytes(order="C")).hexdigest(),
        "block_moments": block_rows,
        "prebuilt_labels_accepted": False,
    }
    return generated, receipt


def encode_family_bank(np: Any, values: Any, weights: Any, *, role: str,
                       rows: int, cols: int, block_size: int,
                       lambda_per_bit: float,
                       rm_exception_limit: int | None,
                       gf2_ranks: Sequence[int],
                       gf2_exception_limit: int | None,
                       romdd_depths: Sequence[int],
                       romdd_exception_limit: int | None,
                       rm_exact_pair_max: int = 4096,
                       rm_list_pairs: int = 256,
                       gf2_exact_pair_max: int = 65536,
                       gf2_heuristic_sweeps: int = 2) -> dict[str, EncodedComponent]:
    """Run the same paid-selector family bank for source or control."""
    literal = encode_literal_component(
        np, values, weights, role=role, rows=rows, cols=cols,
        block_size=block_size)
    rm = encode_rm1_component(
        np, values, weights, role=role, rows=rows, cols=cols,
        block_size=block_size, lambda_per_bit=lambda_per_bit,
        exception_limit=rm_exception_limit,
        exact_pair_max=rm_exact_pair_max, list_pairs=rm_list_pairs)
    gf2 = encode_gf2_component(
        np, values, weights, role=role, rows=rows, cols=cols,
        block_size=block_size, lambda_per_bit=lambda_per_bit, ranks=gf2_ranks,
        exception_limit=gf2_exception_limit,
        exact_factor_pair_max=gf2_exact_pair_max,
        heuristic_sweeps=gf2_heuristic_sweeps)
    romdd = encode_romdd_component(
        np, values, weights, role=role, rows=rows, cols=cols,
        block_size=block_size, lambda_per_bit=lambda_per_bit,
        depths=romdd_depths, exception_limit=romdd_exception_limit)
    return {component.family: component for component in (literal, rm, gf2, romdd)}


def choose_paid_mode(bank: dict[str, EncodedComponent],
                     lambda_per_bit: float) -> EncodedComponent:
    require(set(bank) == {
        "literal4", "rm1_plus_exceptions", "gf2_rank_plus_exceptions",
        "romdd_plus_exceptions"}, "complete family bank")
    return min(bank.values(), key=lambda component: (
        component.weighted_sse + lambda_per_bit * component.physical_bits,
        component.physical_bits, component.weighted_sse, component.family))


def absolute_source_gate(component: EncodedComponent) -> dict[str, Any]:
    in_rate = 2.15 <= component.rate_bpw <= 2.5
    target = component.F <= TARGET_F
    return {
        "family": component.family,
        "physical_rate_bpw": component.rate_bpw,
        "relative_mse": component.relative_mse,
        "F": component.F,
        "rate_interval_pass": in_rate,
        "target_F_pass": target,
        "controls_may_run": in_rate and target,
        "status": ("SOURCE_ABSOLUTE_SURVIVOR_CONTROLS_MAY_RUN"
                   if in_rate and target else
                   "SOURCE_ABSOLUTE_MISS_CONTROLS_FORBIDDEN"),
        "control_subtraction_can_create_pass": False,
    }


def paired_layer_bootstrap(deltas_by_layer: dict[str, float], *,
                           replicates: int = 4096,
                           seed: int = 0x4C4F47494351) -> dict[str, Any]:
    """Deterministic whole-layer cluster bootstrap for paired rate deltas."""
    require(len(deltas_by_layer) >= 5 and replicates >= 100,
            "bootstrap whole-layer clusters")
    layers = sorted(deltas_by_layer)
    values = [float(deltas_by_layer[layer]) for layer in layers]
    require(all(math.isfinite(value) for value in values), "bootstrap deltas")
    # Standard-library counter PRF avoids importing a global RNG.
    samples = []
    for replicate in range(replicates):
        total = 0.0
        for draw in range(len(layers)):
            digest = hashlib.sha256(
                f"{seed}:{replicate}:{draw}".encode("ascii")).digest()
            index = int.from_bytes(digest[:8], "big") % len(layers)
            total += values[index]
        samples.append(total / len(layers))
    samples.sort()
    lower = samples[math.floor(0.025 * replicates)]
    upper = samples[min(replicates - 1, math.ceil(0.975 * replicates) - 1)]
    return {
        "clusters": len(layers),
        "replicates": replicates,
        "point_mean": sum(values) / len(values),
        "lower_95": lower,
        "upper_95": upper,
        "resampling_unit": "whole test layer paired delta",
        "per_weight_iid_interval_used": False,
    }

