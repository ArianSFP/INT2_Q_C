#!/usr/bin/env python3
"""Independent, payload-free audit for epsilon-TCQ/WFA early gate v0."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


EXPECTED_MANIFEST = (
    "0d146f3510d0d42e90d5fd58fe283a8b2dcbf2bd278fa4201fd1601dd301383b"
)
EXPECTED_ROOT = (
    "17581794ba7c6c35faf76f5c3926b59a71fa0b57b3df41c73082ceee202a13e0"
)
MASK32 = (1 << 32) - 1
HALF = 1 << 31
QUARTER = 1 << 30
THREE_QUARTER = 3 << 30
TOTAL = 1 << 16


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(root: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(
        f"epsilon_tcq_v0_audit_{name}", root / f"{name}.py")
    require(specification is not None and specification.loader is not None,
            f"load {name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def arithmetic_encode(bits_and_frequencies: Sequence[tuple[int, int]]) -> tuple[bytes, int]:
    """Independent implementation of the frozen binary arithmetic grammar."""
    low = 0
    high = MASK32
    pending = 0
    emitted: list[int] = []

    def emit(bit: int) -> None:
        nonlocal pending
        emitted.append(bit)
        emitted.extend([1 - bit] * pending)
        pending = 0

    for bit, frequency_one in bits_and_frequencies:
        require(bit in (0, 1) and 1 <= frequency_one <= 65535,
                "independent arithmetic input")
        width = high - low + 1
        split = low + width * (TOTAL - frequency_one) // TOTAL - 1
        require(low <= split < high, "independent arithmetic split")
        if bit == 0:
            high = split
        else:
            low = split + 1
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTER:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low = (low << 1) & MASK32
            high = ((high << 1) & MASK32) | 1
    pending += 1
    emit(0 if low < QUARTER else 1)
    payload = bytearray((len(emitted) + 7) // 8)
    for index, bit in enumerate(emitted):
        payload[index >> 3] |= bit << (7 - (index & 7))
    return bytes(payload), len(emitted)


def suffix_path(model: Any, labels: Sequence[int], targets: Sequence[float],
                reproduction: Sequence[float], fixed_bytes: int,
                rate_lambda: float) -> tuple[Any, ...]:
    """Independent direct-four evaluator for the suffix topology fixture."""
    require(model.candidate.topology == "suffix", "suffix audit fixture")
    state = 0
    mask = model.candidate.states - 1
    events: list[tuple[int, int]] = []
    distortion = 0.0
    for position, (label, target) in enumerate(zip(labels, targets)):
        if position % model.candidate.reset == 0:
            state = 0
        bits = ((label >> 1) & 1, label & 1)
        for level, bit in enumerate(bits):
            context = (2 * (position & 3) + level) % 16
            frequency = model.frequencies_q16[state * 16 + context]
            events.append((bit, frequency))
            state = ((state << 1) | bit) & mask
        error = float(target) - float(reproduction[label])
        distortion += error * error
    payload, logical = arithmetic_encode(events)
    objective = distortion + rate_lambda * 8 * (fixed_bytes + len(payload))
    return objective, distortion, len(payload), logical, tuple(labels), payload


def make_fixture(core: Any, legal: Any) -> tuple[Any, ...]:
    adapter = legal.DirectFourLevelAdapter((-1.0, -0.25, 0.25, 1.0))
    nearest = (1, 2, 2, 1, 0, 3, 2, 1)
    targets = (-0.37, 0.42, 0.18, -0.11, -0.91, 0.88, 0.31, -0.22)
    trajectory = []
    legal_state = adapter.initial_state()
    for position, label in enumerate(nearest):
        choice = [row for row in adapter.encode_choices(
            position, legal_state, label, 1) if row.label == label][0]
        trajectory.append(choice)
        legal_state = choice.next_legal_state
    model = core.fit_model(core.ModelCandidate("suffix", 4, 32),
                           (trajectory, trajectory))
    head = core.CentroidHead("nominal", 4, 4, ())
    return adapter, targets, nearest, model, head


def complete_ledger(total: int) -> dict[str, int]:
    require(total >= 88, "audit ledger total")
    return {
        "header_bytes": 32,
        "model_bytes": 24,
        "topology_bytes": 8,
        "frequency_bytes": 16,
        "centroid_bytes": 8,
        "directory_bytes": 8,
        "frame_header_bytes": 16,
        "payload_bytes": total - 88,
        "padding_bytes": 0,
        "total_bytes": total,
    }


def run(package: Path, run_cupy: bool) -> dict[str, Any]:
    package = package.resolve(strict=True)
    manifest_payload = (package / "SOURCE_MANIFEST.json").read_bytes()
    require(digest(manifest_payload) == EXPECTED_MANIFEST, "manifest pin")
    manifest = json.loads(manifest_payload)
    require(manifest["source_root_sha256"] == EXPECTED_ROOT, "root pin")

    legal = load(package, "legal_interface")
    core = load(package, "tcq_core")
    packet = load(package, "packet_codec")
    gate = load(package, "gate_contract")

    adapter, targets, nearest, model, head = make_fixture(core, legal)
    rate_lambda = 2.0 ** -8
    fixed = packet.fixed_packet_bytes(adapter.interface, model, head)
    exact = core.search_labels(
        targets, nearest, adapter, model, head, epsilon=1,
        rate_lambda=rate_lambda, fixed_packet_bytes=fixed, exact=True)
    candidate_sets = [range(max(0, label - 1), min(3, label + 1) + 1)
                      for label in nearest]
    brute = min(
        suffix_path(model, labels, targets, adapter.reproduction,
                    fixed, rate_lambda)
        for labels in itertools.product(*candidate_sets))
    require(exact.labels == brute[4] and exact.payload == brute[5] and
            exact.logical_bits == brute[3] and
            math.isclose(exact.distortion, brute[1], rel_tol=0.0, abs_tol=1e-15) and
            math.isclose(exact.objective, brute[0], rel_tol=0.0, abs_tol=1e-15),
            "exact search equals independent exhaustive oracle")

    beam_targets = targets * 3
    beam_nearest = nearest * 3
    beam = core.search_labels(
        beam_targets, beam_nearest, adapter, model, head, epsilon=1,
        rate_lambda=rate_lambda, fixed_packet_bytes=fixed,
        exact=False, beam_width=32)
    beam_eval = suffix_path(model, beam.labels, beam_targets,
                            adapter.reproduction, fixed, rate_lambda)
    require(beam.payload == beam_eval[5] and beam.logical_bits == beam_eval[3] and
            math.isclose(beam.distortion, beam_eval[1], rel_tol=0.0, abs_tol=1e-15) and
            math.isclose(beam.objective, beam_eval[0], rel_tol=0.0, abs_tol=1e-15),
            "beam result independently rescored")
    blob = packet.build_packet(adapter.interface, model, head, beam.labels,
                               beam.payload, beam.logical_bits)
    decoded = packet.decode_and_reencode(blob, adapter, core)
    require(decoded["labels"] == beam.labels and
            decoded["packet_bytes"] == beam.physical_bytes,
            "beam literal packet replay")

    # The primitive accepts any nonnegative caller-provided fixed cost. This is
    # safe only if the missing production driver binds it to the built packet.
    undercharged = core.search_labels(
        targets, nearest, adapter, model, head, epsilon=1,
        rate_lambda=rate_lambda, fixed_packet_bytes=0, exact=True)
    undercharged_blob = packet.build_packet(
        adapter.interface, model, head, undercharged.labels,
        undercharged.payload, undercharged.logical_bits)
    require(undercharged.physical_bytes != len(undercharged_blob),
            "caller fixed-byte undercharge is observable")

    # The current source gate validates the ledger internally but does not bind
    # it to row['bytes'], does not derive gain fields, and trusts read_amplification.
    forged_row = {
        "weights": 4096,
        "sse": 0.2,
        "source_energy": 10.0,
        "bytes": 1280,
        "nearest_bytes": 1,
        "nearest_sse": 0.001,
        "state_gain_bpw": 0.08,
        "local_gain_bpw": 0.02,
        "permuted_gain_bpw": 0.01,
        "literal_reencode": True,
        "read_amplification": 0.01,
        "byte_ledger": complete_ledger(8192),
    }
    forged_gate = gate.source_gate((forged_row,))
    require(forged_gate["controls_may_open"] is True and
            forged_row["bytes"] != forged_row["byte_ledger"]["total_bytes"] and
            forged_row["sse"] > forged_row["nearest_sse"],
            "unbound source gate witness")

    controls = [{
        "ordinal": ordinal,
        "full_ptq_pipeline": True,
        "legal_trace_regenerated": True,
        "nested_selection_repeated": True,
        "literal_reencode": True,
        "gain_bpw": 0.01,
        "closure_sha256": "z" * 64,
    } for ordinal in range(8)]
    unbound_control_gate = gate.final_control_gate(
        0.05, controls, source_survived=True)
    require(unbound_control_gate["status"].startswith("ELIGIBLE"),
            "control receipt strings are schema-only")

    # The deterministic affine map is a true permutation for every stream and
    # therefore provides a mechanically capacity-matched state null.
    permuted_head = core.CentroidHead(
        "state_permuted", 8, 4, (0.0,) * 32)
    for stream in range(32):
        mapped = [permuted_head._mapped_state(state, stream)
                  for state in range(8)]
        require(sorted(mapped) == list(range(8)),
                "state-permuted map is bijective")

    require(not hasattr(legal, "PolarisAdapter"), "production adapter absent")
    direct = legal.DirectFourLevelAdapter((-1.0, -0.25, 0.25, 1.0))
    strata = legal.SyntheticStrataLegalAdapter(0.125)
    require(legal.validate_adapter(direct)["events_per_label"] == 2 and
            legal.validate_adapter(strata)["events_per_label"] == 6,
            "non-aliasing event widths")

    owners = ((0,), (1,), (2, 3), (3,), (4,))
    folds = gate.owner_component_folds(owners, core)
    for fold in folds:
        held = set(fold["held_owner_component"])
        require(all(held.isdisjoint(owners[index])
                    for index in fold["development_stream_ordinals"]),
                "owner component separation")

    cupy_receipt = None
    if run_cupy:
        cupy = load(package, "cupy_backend")
        cupy_receipt = cupy.source_free_smoke()
        require(cupy_receipt["status"] == "PASS_SOURCE_FREE_CUPY_BEAM_SMOKE" and
                cupy_receipt["payload_accessed"] is False,
                "payload-free CuPy smoke")

    return {
        "schema": "epsilon-tcq-wfa-early-gate-v0-independent-audit-receipt",
        "status": "HOLD_BINDINGS_REQUIRED_BEFORE_PAYLOAD",
        "source_manifest_sha256": EXPECTED_MANIFEST,
        "source_root_sha256": EXPECTED_ROOT,
        "checks": {
            "exact_search_matches_independent_exhaustive_oracle": True,
            "beam_objective_independently_rescored": True,
            "literal_packet_decode_reencode": True,
            "model_and_centroid_bytes_present_when_helper_used": True,
            "direct4_and_synthetic6_interfaces_non_aliasing": True,
            "state_permutation_bijective_and_source_independent_formula": True,
            "owner_component_partition_mechanics": True,
            "production_polaris_strata_adapter_absent": True,
        },
        "blocking_findings": [
            "search fixed_packet_bytes is caller supplied and not bound to the built packet",
            "source_gate does not bind row bytes to byte-ledger total bytes",
            "source_gate trusts un-derived gain and read-amplification fields",
            "final_control_gate accepts caller booleans and unvalidated 64-character receipt strings",
            "outer-fold fitting, matched-control execution, multi-owner container, and measured inference read path are planned, not executable",
            "production-length beam scalability is unvalidated; paths copy complete label, state, and arithmetic-prefix tuples and CuPy top-k is not wired into search",
        ],
        "cupy": cupy_receipt,
        "qwen_payload_accessed": False,
        "current_codec_payload_accessed": False,
        "matched_control_payload_accessed": False,
        "safe_next_step": "separately freeze and audit a read-only legal-transition adapter, but do not open Qwen through this gate until a bound driver/auditor closes every blocking finding",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--package", type=Path, required=True)
    value.add_argument("--cupy", action="store_true")
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(run(arguments.package, arguments.cupy), sort_keys=True,
                     separators=(",", ":"), allow_nan=False))
