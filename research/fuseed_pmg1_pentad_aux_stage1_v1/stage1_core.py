#!/usr/bin/env python3
"""Pure orchestration for the fixed PMG1 pentad auxiliary stage-1 screen.

There is intentionally no command-line entry point, source loader, output
writer, network client, or payload path here.  A future independently audited
dispatcher may supply already-authenticated in-memory targets, descriptors,
and regenerated anchors.  This module then performs the one frozen joint fit,
the 16 frozen scramble controls, and the delete-one-expert gate.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

try:
    from . import contract
except ImportError:  # Direct source-tree import for the standard-library tests.
    import contract  # type: ignore[no-redef]


Descriptor = tuple[int, str, int, int]


def _validate_inputs(
    targets: Mapping[str, Sequence[float]],
    descriptors: Mapping[str, Sequence[Descriptor]],
    anchors: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, dict[tuple[int, str], tuple[int, ...]]]:
    """Validate exact geometry and return canonical per-identity indices."""
    contract.require(set(targets) == {"fit", "score"}, "target split closure")
    contract.require(set(descriptors) == {"fit", "score"}, "descriptor split closure")
    contract.require(set(anchors) == {"fit", "score"}, "anchor split closure")
    expected_counts = {"fit": contract.FIT_COORDINATES, "score": contract.SCORE_COORDINATES}
    expected_identities = set(contract.identity_set())
    indices: dict[str, dict[tuple[int, str], tuple[int, ...]]] = {}
    for split in ("fit", "score"):
        rows = descriptors[split]
        contract.require(len(rows) == expected_counts[split], f"{split} descriptor count")
        contract.require(len(targets[split]) == len(rows), f"{split} target count")
        contract.require(len(anchors[split]) == len(rows), f"{split} anchor count")
        contract.ensure_all_finite(targets[split])
        normalized: dict[tuple[int, str], list[int]] = {
            identity: [] for identity in contract.identity_set()
        }
        seen_coordinates: set[Descriptor] = set()
        for index, descriptor in enumerate(rows):
            contract.require(len(descriptor) == 4, f"{split} descriptor width")
            expert, role, row, column = descriptor
            normalized_descriptor = (int(expert), str(role), int(row), int(column))
            identity = normalized_descriptor[:2]
            contract.require(identity in expected_identities, f"{split} identity")
            contract.require(0 <= normalized_descriptor[2] < 768, f"{split} row")
            contract.require(0 <= normalized_descriptor[3] < 2048, f"{split} column")
            contract.require(
                normalized_descriptor not in seen_coordinates,
                f"duplicate {split} coordinate",
            )
            seen_coordinates.add(normalized_descriptor)
            normalized[identity].append(index)
            anchor_row = anchors[split][index]
            contract.require(len(anchor_row) == len(contract.SEEDS_U32), f"{split} anchor width")
            contract.ensure_all_finite(anchor_row)
        contract.require(set(normalized) == expected_identities, f"{split} identity closure")
        contract.require(
            all(len(values) > len(contract.SEEDS_U32) for values in normalized.values()),
            f"{split} per-identity sample minimum",
        )
        indices[split] = {identity: tuple(values) for identity, values in normalized.items()}
    contract.require(
        not (set(descriptors["fit"]) & set(descriptors["score"])),
        "fit/score coordinate overlap",
    )
    return indices


def _gather(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def _validate_permutations(
    permutations: Mapping[tuple[tuple[int, str], str], Sequence[int]] | None,
    indices: Mapping[str, Mapping[tuple[int, str], tuple[int, ...]]],
) -> None:
    if permutations is None:
        return
    expected = {
        (identity, split)
        for identity in contract.identity_set()
        for split in ("fit", "score")
    }
    contract.require(set(permutations) == expected, "control permutation closure")
    for identity, split in sorted(expected):
        count = len(indices[split][identity])
        permutation = tuple(int(value) for value in permutations[(identity, split)])
        contract.require(
            tuple(sorted(permutation)) == tuple(range(count)),
            "control permutation bijection",
        )


def evaluate(
    targets: Mapping[str, Sequence[float]],
    descriptors: Mapping[str, Sequence[Descriptor]],
    anchors: Mapping[str, Sequence[Sequence[float]]],
    permutations: Mapping[tuple[tuple[int, str], str], Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Fit and score the complete fixed pentad once on all 23 identities."""
    indices = _validate_inputs(targets, descriptors, anchors)
    _validate_permutations(permutations, indices)
    records = []
    for identity in contract.identity_set():
        fit_index = indices["fit"][identity]
        score_index = indices["score"][identity]
        x_fit = _gather(anchors["fit"], fit_index)
        x_score = _gather(anchors["score"], score_index)
        if permutations is not None:
            x_fit = _gather(x_fit, permutations[(identity, "fit")])
            x_score = _gather(x_score, permutations[(identity, "score")])
        y_fit = _gather(targets["fit"], fit_index)
        y_score = _gather(targets["score"], score_index)
        fit = contract.fit_decoded_fp16(x_fit, y_fit)
        score = contract.score_decoded_fit(fit, x_score, y_score)
        records.append(
            {
                "expert": identity[0],
                "role": identity[1],
                "fit_coordinates": len(fit_index),
                "score_coordinates": len(score_index),
                "fp16_words_hex": [format(int(word), "04x") for word in fit["fp16_words"]],
                "condition": float(fit["condition"]),
                "ridge": float(fit["ridge"]),
                "gram_eigenvalues": [float(value) for value in fit["eigenvalues"]],
                "fit_target_mean": float(fit["fit_target_mean"]),
                "sse": score["sse"],
                "source_energy": score["source_energy"],
                "centered_baseline_sse": score["centered_baseline_sse"],
                "capture": score["capture"],
                "centered_capture": score["centered_capture"],
            }
        )
    aggregate = contract.aggregate_records(records)
    centered_sse = math.fsum(float(row["centered_baseline_sse"]) for row in records)
    contract.require(centered_sse > 0.0 and math.isfinite(centered_sse), "centered total")
    return {
        "records": records,
        "aggregate": aggregate,
        "centered_baseline_sse": centered_sse,
        "centered_capture": 1.0 - aggregate["sse"] / centered_sse,
        "roles": contract.role_aggregates(records),
        "delete_one_expert": contract.expert_jackknife(records),
    }


def frozen_scramble_permutations(
    numpy_module: Any,
    indices: Mapping[str, Mapping[tuple[int, str], tuple[int, ...]]],
    seed: int,
) -> dict[tuple[tuple[int, str], str], tuple[int, ...]]:
    """Exact historical PCG64 within-identity fit/score permutation control."""
    contract.require(seed in contract.CONTROL_SEEDS, "frozen control seed")
    random = numpy_module.random.Generator(numpy_module.random.PCG64(int(seed)))
    result = {}
    for identity in contract.identity_set():
        for split in ("fit", "score"):
            count = len(indices[split][identity])
            result[(identity, split)] = tuple(
                int(value) for value in random.permutation(count).tolist()
            )
    return result


def scramble_controls(
    targets: Mapping[str, Sequence[float]],
    descriptors: Mapping[str, Sequence[Descriptor]],
    anchors: Mapping[str, Sequence[Sequence[float]]],
    numpy_module: Any,
) -> dict[str, Any]:
    """Evaluate all 16 predeclared source-preserving anchor-row controls."""
    indices = _validate_inputs(targets, descriptors, anchors)
    rows = []
    for replicate, seed in enumerate(contract.CONTROL_SEEDS):
        permutations = frozen_scramble_permutations(numpy_module, indices, seed)
        result = evaluate(targets, descriptors, anchors, permutations)
        rows.append(
            {
                "replicate": replicate,
                "seed": seed,
                "capture": result["aggregate"]["capture"],
                "centered_capture": result["centered_capture"],
            }
        )
    values = [float(row["capture"]) for row in rows]
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {
        "construction": (
            "NumPy PCG64; independent within-expert-role anchor-row permutations "
            "for fit and score; seeds 26090100..26090115"
        ),
        "rows": rows,
        "mean_capture": mean,
        "maximum_capture": max(values),
        "minimum_capture": min(values),
        "mc_standard_error": math.sqrt(variance / len(values)),
    }


def build_result(primary: Mapping[str, Any], controls: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the complete conservative gate and build a source-result body."""
    records = primary["records"]
    contract.require(len(records) == contract.IDENTITY_COUNT, "primary record count")
    control_rows = controls["rows"]
    contract.require(len(control_rows) == len(contract.CONTROL_SEEDS), "control count")
    contract.require(
        tuple(int(row["seed"]) for row in control_rows) == contract.CONTROL_SEEDS,
        "control seed order",
    )
    base = contract.decision(records)
    primary_capture = float(base["aggregate"]["capture"])
    maximum_control = max(float(row["capture"]) for row in control_rows)
    source_exceeds_every_control = primary_capture > maximum_control
    survives = bool(base["survives"] and source_exceeds_every_control)
    status = (
        "SURVIVES_FIXED_PENTAD_AUXILIARY_STAGE1_ONLY"
        if survives
        else "HARD_KILL_FIXED_PENTAD_AUXILIARY_STAGE1_NO_TUPLE_RETRY"
    )
    return {
        "schema": contract.RESULT_SCHEMA,
        "status": status,
        "claim_boundary": (
            "Fixed five-seed, disjoint-coordinate Qwen Up/Down development-panel "
            "falsification only; never Gate evidence, family evidence, validation, "
            "a rebuilt residual-codec score, a finite bitstream, or target achievement."
        ),
        "fixed_hypothesis": {
            "seeds_u32": list(contract.SEEDS_U32),
            "fit_coordinates": contract.FIT_COORDINATES,
            "score_coordinates": contract.SCORE_COORDINATES,
            "identities": contract.IDENTITY_COUNT,
            "decoded_fp16_joint_5x5_fit": True,
            "no_tuple_retry": True,
        },
        "primary": primary,
        "scramble_controls": controls,
        "gate": {
            **base,
            "status": status,
            "survives": survives,
            "maximum_control_capture": maximum_control,
            "source_exceeds_every_control": source_exceeds_every_control,
            "raw_minus_mean_control_capture": (
                primary_capture - float(controls["mean_capture"])
            ),
        },
        "physical_planning_ledger": contract.physical_ledger(),
        "access_claim": {
            "selection_up_down_files_expected_from_independent_dispatcher": 23,
            "gate_files": 0,
            "validation_files": 0,
            "payload_or_network_access_performed_by_this_module": 0,
        },
    }


__all__ = (
    "build_result",
    "evaluate",
    "frozen_scramble_permutations",
    "scramble_controls",
)
