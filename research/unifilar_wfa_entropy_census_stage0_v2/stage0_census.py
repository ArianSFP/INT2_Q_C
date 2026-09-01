#!/usr/bin/env python3
"""Authenticated-snapshot producer core for integrated UWFA-SC v2.

This file is intentionally *not* a directly executable payload launcher.  A
separate independently reviewed dispatcher must pin the producer manifest and
review digest, read every producer member once through retained descriptors,
and compile/exec this exact entrypoint snapshot.  A public token or self-sealed
JSON cannot call the producer.

The functions below contain the complete source/control scientific pipeline.
They accept already-authenticated byte snapshots and injected modules; no
repository-relative import or dynamic input pathname is used.
"""

from __future__ import annotations

import hashlib
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PRODUCER_ABI = "uwfa-sc-v2-authenticated-snapshot-producer-abi-1"
DIRECT_LAUNCH_STATUS = "BLOCK_DIRECT_EXECUTION_REQUIRES_EXTERNALLY_PINNED_DISPATCHER"
MAX_PROJECTED_WALL_SECONDS = 21_600.0
MAX_HOST_BYTES = 96 * (1 << 30)
MAX_VRAM_BYTES = 28 * (1 << 30)
MIN_CALIBRATED_CELL_SYMBOLS_PER_SECOND = 1_000_000.0
HELDOUT_T_CRITICAL_95_TWO_SIDED_DF5 = 2.570581835636305


@dataclass(frozen=True)
class BoundEvidence:
    baseline_plan_sha256: str
    baseline_score_sha256: str
    universal_decoder_sha256: str
    producer_manifest_sha256: str
    audit_bootstrap_sha256: str
    source_panel_sha256: str
    extraction_program_sha256: str
    pipeline_sha256: str

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be SHA-256 hex")
            try:
                raw = bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{name} not hexadecimal") from exc
            if len(raw) != 32:
                raise ValueError(f"{name} digest geometry")

    def container_hashes(self) -> dict[str, str]:
        return {
            "baseline_plan_sha256": self.baseline_plan_sha256,
            "baseline_score_sha256": self.baseline_score_sha256,
            "universal_decoder_sha256": self.universal_decoder_sha256,
            "producer_manifest_sha256": self.producer_manifest_sha256,
            "audit_bootstrap_sha256": self.audit_bootstrap_sha256,
            "source_panel_sha256": self.source_panel_sha256,
            "extraction_program_sha256": self.extraction_program_sha256,
        }


def _host_list(value: Any) -> list[int]:
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def packed_rows(streams: Sequence[Mapping[str, Any]]) -> list[tuple[bytes, bytes, bytes]]:
    return [(bytes(row["bits_bytes"]), bytes(row["levels_bytes"]), bytes(row["base_bytes"])) for row in streams]


def prepare_backend_cache(backend: Any, panel: Mapping[str, Any]) -> Any:
    return backend.pack_streams(packed_rows(panel["streams"]))


def _subset(backend: Any, cache: Any, indices: Sequence[int]) -> Any:
    if not indices:
        raise ValueError("empty stream subset")
    if hasattr(backend, "subset"):
        return backend.subset(cache, list(indices))
    # Source-free CPU reference backends may expose rows directly.
    if isinstance(cache, dict) and "rows" in cache:
        return {"rows": [cache["rows"][index] for index in indices]}
    raise TypeError("backend must provide descriptor-only subset selection")


def fit_candidate(common: Any, backend: Any, cache: Any, indices: Sequence[int], candidate: Any) -> list[int]:
    subset = _subset(backend, cache, indices)
    counts = _host_list(backend.fit_counts(subset, candidate.topology_id, candidate.states, candidate.reset_length))
    return common.q16_frequencies_from_counts(counts)


def exact_lengths(
    common: Any,
    backend: Any,
    cache: Any,
    indices: Sequence[int],
    candidate: Any,
    frequencies: Sequence[int],
) -> list[int]:
    subset = _subset(backend, cache, indices)
    return _host_list(backend.exact_lengths(subset, candidate.topology_id, candidate.states, candidate.reset_length, frequencies))


def validation_score(common: Any, lengths: Sequence[int], candidate: Any) -> int:
    return 8 * sum((int(value) + 7) // 8 for value in lengths) + 8 * common.model_ledger(candidate)["physical_model_bytes"]


def gpu_preflight_all_150(common: Any, backend: Any) -> dict[str, Any]:
    """Exact all-cell CPU/CuPy source-free replay on the actual runtime."""
    from array import array

    rows = []
    for stream_ordinal, length in enumerate((4097, 2053, 1031, 521)):
        bits = bytes((((index * (17 + stream_ordinal)) ^ (index >> (1 + stream_ordinal)) ^ (index // 31)) & 1) for index in range(length))
        levels = bytes((index + 3 * stream_ordinal) % common.LEVELS for index in range(length))
        # Visit every prior bin and both arithmetic extremes.
        base_values = []
        for index in range(length):
            if index % 257 == 0:
                base_values.append(1)
            elif index % 257 == 1:
                base_values.append(65535)
            else:
                bucket = (index // 4 + stream_ordinal) % common.PRIOR_BINS
                base_values.append(min(65535, max(1, bucket * 4096 + 2048)))
        packed_u16 = array("H", base_values)
        if sys.byteorder != "little":
            packed_u16.byteswap()
        rows.append((bits, levels, packed_u16.tobytes(), base_values))
    cache = backend.pack_streams([(bits, levels, base_bytes) for bits, levels, base_bytes, _base in rows])
    checked = []
    started = time.perf_counter()
    for candidate in common.candidate_bank():
        cpu_counts = common.merge_counts(
            common.count_stream_cpu(list(bits), list(levels), base_values, candidate)
            for bits, levels, _base_bytes, base_values in rows
        )
        first_counts = _host_list(backend.fit_counts(cache, candidate.topology_id, candidate.states, candidate.reset_length))
        second_counts = _host_list(backend.fit_counts(cache, candidate.topology_id, candidate.states, candidate.reset_length))
        if first_counts != cpu_counts or second_counts != cpu_counts:
            raise ValueError(f"all-150 CPU/CuPy count mismatch: {candidate}")
        fitted_cpu = common.q16_frequencies_from_counts(cpu_counts)
        fitted_gpu = common.q16_frequencies_from_counts(first_counts)
        if fitted_gpu != fitted_cpu:
            raise ValueError(f"all-150 fitted Q0.16 mismatch: {candidate}")
        cpu_lengths = [
            common.exact_stream_length_cpu(list(bits), list(levels), base_values, candidate, fitted_cpu)
            for bits, levels, _base_bytes, base_values in rows
        ]
        first_lengths = _host_list(backend.exact_lengths(cache, candidate.topology_id, candidate.states, candidate.reset_length, fitted_cpu))
        second_lengths = _host_list(backend.exact_lengths(cache, candidate.topology_id, candidate.states, candidate.reset_length, fitted_cpu))
        if first_lengths != cpu_lengths or second_lengths != cpu_lengths:
            raise ValueError(f"all-150 CPU/CuPy arithmetic mismatch: {candidate}")
        extreme = [1 if index & 1 else 65535 for index in range(common.model_frequency_count(candidate))]
        extreme_cpu = [
            common.exact_stream_length_cpu(list(bits), list(levels), base_values, candidate, extreme)
            for bits, levels, _base_bytes, base_values in rows
        ]
        extreme_gpu = _host_list(backend.exact_lengths(cache, candidate.topology_id, candidate.states, candidate.reset_length, extreme))
        if extreme_gpu != extreme_cpu:
            raise ValueError(f"frequency-extreme CPU/CuPy mismatch: {candidate}")
        checked.append({
            **candidate.as_dict(),
            "count_tensor_sha256": hashlib.sha256(b"".join(int(value).to_bytes(8, "little") for value in cpu_counts)).hexdigest(),
            "fitted_q016_sha256": hashlib.sha256(b"".join(int(value).to_bytes(2, "little") for value in fitted_cpu)).hexdigest(),
            "logical_lengths": cpu_lengths,
            "repeated_gpu_run_exact": True,
        })
    elapsed = time.perf_counter() - started
    return {
        "status": "PASS_ALL_150_CPU_CUPY_EXACT_REPEATED",
        "cells": checked,
        "cell_count": len(checked),
        "streams": len(rows),
        "symbols_per_complete_bank": sum(len(row[0]) for row in rows),
        "elapsed_seconds": elapsed,
        "environment": backend.environment_receipt() if hasattr(backend, "environment_receipt") else {},
        "frequency_extremes_tested": [1, 65535],
        "reset_boundaries_through_4096_tested": True,
    }


def attach_semantic_owners(panel: dict[str, Any]) -> None:
    route = panel["artifact"]["route_rows"]
    experts = int(panel["experts"])
    identities = []
    for expert in range(experts):
        gate = route[3 * expert]
        identities.append((int(gate["layer"]), int(gate["expert"])))
    if len(set(identities)) != experts:
        raise ValueError("artifact route identities are not unique")
    owned_counts = [0] * experts
    for row in panel["streams"]:
        owners = [expert for expert in range(experts) if int(row["owner_mask"]) & (1 << expert)]
        if not owners:
            raise ValueError("stream has no semantic owner")
        for expert in owners:
            owned_counts[expert] += 1
        row["owner_expert_ordinals"] = owners
        row["owner_identity_indices"] = owners
    if any(value == 0 for value in owned_counts):
        raise ValueError("empty expert may not amortize shared bytes")
    panel["semantic_identities"] = identities


def _split_digest(protocol: Any, identity: tuple[int, int], row: Mapping[str, Any]) -> bytes:
    digest = protocol.length_prefixed_digest(
        [identity[0], identity[1], int(row["stream_ordinal"]), int(row["owner_mask"])],
        domain=b"UWFA-SC-V2-NESTED-SPLIT-2026-09-01\x00",
    )
    return bytes.fromhex(digest)


def projected_updates(common: Any, protocol: Any, panel: Mapping[str, Any]) -> dict[str, Any]:
    total = 0
    folds = []
    streams = panel["streams"]
    identities = panel["semantic_identities"]
    for identity_index, identity in enumerate(identities):
        layer, expert = identity
        development_indices = [
            index for index, row in enumerate(streams)
            if all(
                identities[owner][0] != layer and identities[owner][1] != expert
                for owner in row["owner_identity_indices"]
            )
        ]
        test_indices = [index for index, row in enumerate(streams) if identity_index in row["owner_identity_indices"]]
        if len(development_indices) < 2 or not test_indices:
            raise ValueError("nested fold geometry")
        ranked = sorted(
            development_indices,
            key=lambda index: (_split_digest(protocol, identity, streams[index]), int(streams[index]["stream_ordinal"])),
        )
        validation_count = min(max(1, len(ranked) // common.INNER_VALIDATION_MODULUS), len(ranked) - 1)
        validation = ranked[:validation_count]
        train = ranked[validation_count:]
        train_symbols = sum(int(streams[index]["symbols"]) for index in train)
        validation_symbols = sum(int(streams[index]["symbols"]) for index in validation)
        development_symbols = sum(int(streams[index]["symbols"]) for index in development_indices)
        test_symbols = sum(int(streams[index]["symbols"]) for index in test_indices)
        updates = len(common.candidate_bank()) * (train_symbols + validation_symbols) + development_symbols + test_symbols
        total += updates
        folds.append({"identity_index": identity_index, "cell_symbol_updates": updates})
    final_symbols = sum(int(row["symbols"]) for row in streams)
    total += 2 * final_symbols  # final fit plus exact final scoring
    maximum_survivor_updates = 5 * total  # source plus four predeclared shuffles
    projected_seconds = maximum_survivor_updates / MIN_CALIBRATED_CELL_SYMBOLS_PER_SECOND
    return {
        "exact_cell_symbol_updates": total,
        "maximum_source_survivor_updates_including_four_shuffles": maximum_survivor_updates,
        "folds": folds,
        "conservative_minimum_throughput_updates_per_second": MIN_CALIBRATED_CELL_SYMBOLS_PER_SECOND,
        "projected_wall_seconds": projected_seconds,
        "frozen_maximum_wall_seconds": MAX_PROJECTED_WALL_SECONDS,
        "passes_pre_fit_runtime_budget": projected_seconds <= MAX_PROJECTED_WALL_SECONDS,
        "host_byte_budget": MAX_HOST_BYTES,
        "vram_byte_budget": MAX_VRAM_BYTES,
    }


def _clone_panel_with_streams(panel: Mapping[str, Any], streams: list[dict[str, Any]]) -> dict[str, Any]:
    clone = dict(panel)
    clone["streams"] = streams
    return clone


def within_context_permutation(common: Any, protocol: Any, panel: Mapping[str, Any], seed: int = 0x5C11A2) -> dict[str, Any]:
    """Preserve exact public-context bit counts while destroying sequence order."""
    output = []
    for row in panel["streams"]:
        clone = dict(row)
        bits = list(row["bits_bytes"])
        levels = list(row["levels_bytes"])
        base_values = list(row["base"])
        groups: dict[int, list[int]] = {}
        for position, (level, base) in enumerate(zip(levels, base_values, strict=True)):
            context = common.public_context(level, base, position & 3)
            groups.setdefault(context, []).append(position)
        shuffled = list(bits)
        for context, positions in groups.items():
            ranked_sources = sorted(
                positions,
                key=lambda position: protocol.length_prefixed_digest(
                    [seed, int(row["stream_ordinal"]), context, position],
                    domain=b"UWFA-SC-V2-WITHIN-CONTEXT-PERMUTATION\x00",
                ),
            )
            values = [bits[position] for position in ranked_sources]
            for destination, value in zip(positions, values, strict=True):
                shuffled[destination] = value
        clone["bits"] = shuffled
        clone["bits_bytes"] = bytes(shuffled)
        output.append(clone)
    return _clone_panel_with_streams(panel, output)


def multiscale_chunk_shuffle(protocol: Any, panel: Mapping[str, Any], chunk: int, seed: int = 0x71A55E) -> dict[str, Any]:
    if chunk not in {32, 128, 512}:
        raise ValueError("frozen multiscale chunk")
    output = []
    for row in panel["streams"]:
        length = int(row["symbols"])
        spans = [(begin, min(length, begin + chunk)) for begin in range(0, length, chunk)]
        ranked = sorted(
            enumerate(spans),
            key=lambda item: protocol.length_prefixed_digest(
                [seed, chunk, int(row["stream_ordinal"]), item[0]],
                domain=b"UWFA-SC-V2-MULTISCALE-CHUNK-SHUFFLE\x00",
            ),
        )
        order = [spans[index] for index, _span in ranked]
        bits = b"".join(bytes(row["bits_bytes"])[begin:end] for begin, end in order)
        levels = b"".join(bytes(row["levels_bytes"])[begin:end] for begin, end in order)
        base_parts = []
        base_raw = bytes(row["base_bytes"])
        for begin, end in order:
            base_parts.append(base_raw[2 * begin:2 * end])
        base_bytes = b"".join(base_parts)
        base_values = list(int.from_bytes(base_bytes[index:index + 2], "little") for index in range(0, len(base_bytes), 2))
        clone = dict(row)
        clone.update({
            "bits": list(bits), "bits_bytes": bits,
            "levels": list(levels), "levels_bytes": levels,
            "base": base_values, "base_bytes": base_bytes,
        })
        output.append(clone)
    return _clone_panel_with_streams(panel, output)


def survivor_shuffle_diagnostics(common: Any, protocol: Any, backend: Any, panel: Mapping[str, Any]) -> dict[str, Any]:
    variants = [("within_public_context", within_context_permutation(common, protocol, panel))]
    variants.extend((f"chunk_{chunk}", multiscale_chunk_shuffle(protocol, panel, chunk)) for chunk in (32, 128, 512))
    rows = []
    for name, variant in variants:
        cache = prepare_backend_cache(backend, variant)
        scientific = nested_holdout(common, protocol, backend, cache, variant)
        rows.append({
            "variant": name,
            "pooled_exact_heldout_saving_bpw": scientific["pooled_exact_heldout_saving_bpw"],
            "selected": scientific["final_topology_selected_from_nested_fold_votes"],
            "complete_150_cell_nested_search_repeated": True,
        })
    return {
        "status": "PASS_PREDECLARED_SHUFFLE_PIPELINES_EXECUTED",
        "variants": rows,
        "claim_use": "source minus shuffle gaps diagnose nonlocal order; diagnostics never rescue a physical failure",
    }


def nested_holdout(common: Any, protocol: Any, backend: Any, cache: Any, panel: Mapping[str, Any]) -> dict[str, Any]:
    streams = panel["streams"]
    identities = panel["semantic_identities"]
    fold_rows = []
    for identity_index, identity in enumerate(identities):
        layer, expert = identity
        test_indices = [index for index, row in enumerate(streams) if identity_index in row["owner_identity_indices"]]
        development_indices = [
            index for index, row in enumerate(streams)
            if all(
                identities[owner][0] != layer and identities[owner][1] != expert
                for owner in row["owner_identity_indices"]
            )
        ]
        if not test_indices or len(development_indices) < 2:
            raise ValueError(f"nonempty nested fold {identity_index}")
        ranked = sorted(
            development_indices,
            key=lambda index: (_split_digest(protocol, identity, streams[index]), int(streams[index]["stream_ordinal"])),
        )
        validation_count = min(max(1, len(ranked) // common.INNER_VALIDATION_MODULUS), len(ranked) - 1)
        validation_indices = ranked[:validation_count]
        train_indices = ranked[validation_count:]
        selections = []
        for candidate in common.candidate_bank():
            frequencies = fit_candidate(common, backend, cache, train_indices, candidate)
            lengths = exact_lengths(common, backend, cache, validation_indices, candidate, frequencies)
            selections.append((validation_score(common, lengths, candidate), candidate.selector_ordinal, candidate))
        validation_bits, _ordinal, selected = min(selections)
        frequencies = fit_candidate(common, backend, cache, development_indices, selected)
        test_lengths = exact_lengths(common, backend, cache, test_indices, selected, frequencies)
        baseline_allocated_bits = 0.0
        candidate_allocated_bits = 0.0
        allocated_weights = 0.0
        for index, logical in zip(test_indices, test_lengths, strict=True):
            row = streams[index]
            owner_count = len(row["owner_identity_indices"])
            baseline_allocated_bits += 8.0 * int(row["baseline_payload_bytes"]) / owner_count
            candidate_allocated_bits += 8.0 * ((int(logical) + 7) // 8) / owner_count
            allocated_weights += int(row["weight_charge"]) / owner_count
        # Each scientific fold must carry a full independent model.
        model_bits = 8 * common.model_ledger(selected)["physical_model_bytes"]
        saving = (baseline_allocated_bits - candidate_allocated_bits - model_bits) / allocated_weights
        fold_rows.append({
            "outer_identity_index": identity_index,
            "outer_layer_from_artifact": layer,
            "outer_expert_from_artifact": expert,
            "test_stream_ordinals": [int(streams[index]["stream_ordinal"]) for index in test_indices],
            "development_stream_ordinals": [int(streams[index]["stream_ordinal"]) for index in development_indices],
            "inner_train_stream_ordinals": [int(streams[index]["stream_ordinal"]) for index in train_indices],
            "inner_validation_stream_ordinals": [int(streams[index]["stream_ordinal"]) for index in validation_indices],
            "selected_by_inner_validation_only": selected.as_dict(),
            "inner_validation_exact_charged_bits": validation_bits,
            "charged_full_fold_model_bits": model_bits,
            "allocated_test_weights": allocated_weights,
            "allocated_baseline_bits": baseline_allocated_bits,
            "allocated_candidate_bits": candidate_allocated_bits,
            "exact_test_saving_bpw": saving,
        })
    allocated_total = sum(float(row["allocated_test_weights"]) for row in fold_rows)
    if abs(allocated_total - int(panel["weights"])) > 1e-6:
        raise ValueError("owner-attributed outer folds do not partition weights")
    pooled_saved_bits = sum(
        float(row["allocated_baseline_bits"]) - float(row["allocated_candidate_bits"]) - float(row["charged_full_fold_model_bits"])
        for row in fold_rows
    )
    pooled = pooled_saved_bits / allocated_total
    values = [float(row["exact_test_saving_bpw"]) for row in fold_rows]
    mean = statistics.fmean(values)
    if len(values) > 1:
        margin = HELDOUT_T_CRITICAL_95_TWO_SIDED_DF5 * statistics.stdev(values) / math.sqrt(len(values))
    else:
        margin = math.inf
    candidate_votes: dict[int, int] = {}
    for row in fold_rows:
        ordinal = int(row["selected_by_inner_validation_only"]["selector_ordinal"])
        candidate_votes[ordinal] = candidate_votes.get(ordinal, 0) + 1
    selected_ordinal = min(candidate_votes, key=lambda ordinal: (-candidate_votes[ordinal], ordinal))
    selected = common.candidate_bank()[selected_ordinal]
    return {
        "kind": "whole-artifact-semantic-expert-folds_with_owner_attribution",
        "folds": fold_rows,
        "pooled_exact_heldout_saving_bpw": pooled,
        "minimum_fold_exact_saving_bpw": min(values),
        "whole_expert_mean_saving_bpw": mean,
        "whole_expert_t95_lower_bpw": mean - margin,
        "whole_expert_t95_upper_bpw": mean + margin,
        "confidence_rule": "predeclared two-sided Student-t interval across six semantic experts; df=5 critical 2.570581835636305",
        "candidate_vote_counts": candidate_votes,
        "final_topology_selected_from_nested_fold_votes": selected.as_dict(),
        "passes_pooled_standalone_threshold": pooled >= common.STANDALONE_REQUIRED_SAVING_BPW,
        "passes_positive_whole_expert_lower_confidence": mean - margin > 0.0,
        "passes_heldout_gate": pooled >= common.STANDALONE_REQUIRED_SAVING_BPW and mean - margin > 0.0,
    }


def _regions_from_specs(container_codec: Any, stream_specs: Sequence[Any]) -> list[Any]:
    grouped: dict[int, list[Any]] = {}
    for spec, owner_mask in stream_specs:
        grouped.setdefault(int(owner_mask), []).append(spec)
    # Private expert regions first in expert order, then explicitly shared
    # owner regions.  Within each region stream ordinals remain canonical.
    masks = sorted(grouped, key=lambda mask: (mask.bit_count() != 1, (mask & -mask).bit_length() - 1, mask))
    return [
        container_codec.RegionSpec(mask, tuple(sorted(grouped[mask], key=lambda row: row.ordinal)))
        for mask in masks
    ]


def final_container(
    common: Any,
    container_codec: Any,
    adapter: Any,
    backend: Any,
    cache: Any,
    panel: Mapping[str, Any],
    candidate: Any,
    score: Mapping[str, Any],
    bindings: BoundEvidence,
) -> dict[str, Any]:
    indices = list(range(len(panel["streams"])))
    fitted = fit_candidate(common, backend, cache, indices, candidate)
    transmitted_model = common.serialize_model(candidate, fitted)
    # Everything below consumes only the serialized model copy.
    decoded_candidate, decoded_frequencies = common.deserialize_model(transmitted_model)
    stream_specs = []
    identity_specs = []
    payload_rows = []
    for row in panel["streams"]:
        payload, logical = common.encode_unifilar_stream(
            row["bits"], row["levels"], row["base"], decoded_candidate, decoded_frequencies
        )
        decoded = common.decode_unifilar_stream(
            payload, logical, row["levels"], row["base"], decoded_candidate, decoded_frequencies
        )
        if decoded != row["bits"]:
            raise ValueError("serialized-model causal decode mismatch")
        spec = container_codec.StreamSpec(
            ordinal=int(row["stream_ordinal"]),
            symbols=int(row["symbols"]),
            logical_bits=int(logical),
            payload=payload,
            source_digest=str(row["source_digest"]),
            profile_q=int(row["profile_q"]),
            decoder_scale=float(row["decoder_scale"]),
        )
        identity = container_codec.StreamSpec(
            ordinal=int(row["stream_ordinal"]),
            symbols=int(row["symbols"]),
            logical_bits=int(row["baseline_logical_bits"]),
            payload=bytes(row["baseline_payload"]),
            source_digest=str(row["source_digest"]),
            profile_q=int(row["profile_q"]),
            decoder_scale=float(row["decoder_scale"]),
        )
        stream_specs.append((spec, int(row["owner_mask"])))
        identity_specs.append((identity, int(row["owner_mask"])))
        payload_rows.append({
            "ordinal": int(row["stream_ordinal"]),
            "baseline_payload_bytes": int(row["baseline_payload_bytes"]),
            "new_payload_bytes": len(payload),
            "baseline_logical_bits": int(row["baseline_logical_bits"]),
            "new_logical_bits": int(logical),
        })
    regions = _regions_from_specs(container_codec, stream_specs)
    identity_regions = _regions_from_specs(container_codec, identity_specs)
    artifact_sha = str(panel["artifact"]["raw_sha256"])
    reconstruction_sha = str(panel["reconstruction"]["full_reconstruction_f64_sha256"])
    build_args = dict(
        common=common,
        model_packet=transmitted_model,
        immutable_state=bytes(panel["immutable_state"]),
        weights=int(panel["weights"]),
        experts=int(panel["experts"]),
        baseline_object_bytes=int(panel["artifact"]["raw_bytes"]),
        audited_relative_mse=float(score["relative_mse"]),
        baseline_artifact_sha256=artifact_sha,
        reconstruction_sha256=reconstruction_sha,
        audit_binding_sha256=bindings.baseline_score_sha256,
        binding_hashes=bindings.container_hashes(),
        minimum_rate_bpw=common.RATE_MIN,
    )
    container, _predicted = container_codec.build_container(regions=regions, **build_args)
    parsed = container_codec.parse_container(common, container)
    metrics = container_codec.physical_metrics(parsed)
    standalone = adapter.decode_new_container(parsed)
    if standalone["reconstruction"]["full_reconstruction_f64_sha256"] != reconstruction_sha:
        raise ValueError("candidate/current full reconstruction differs")
    rebuilt = container_codec.canonical_rebuild(common, parsed)
    if rebuilt != container:
        raise ValueError("literal container does not canonically rebuild")
    identity_container, _ = container_codec.build_container(regions=identity_regions, **build_args)
    identity_parsed = container_codec.parse_container(common, identity_container)
    identity_metrics = container_codec.physical_metrics(identity_parsed)
    payload_saved_bits = sum(
        8 * (int(row["baseline_payload_bytes"]) - int(row["new_payload_bytes"])) for row in payload_rows
    ) - 8 * len(transmitted_model)
    return {
        "container": container,
        "identity_framing_container": identity_container,
        "parsed_metrics": metrics,
        "identity_framing_metrics": identity_metrics,
        "absolute_saving_vs_bound_current_artifact_bpw": 8.0 * (int(panel["artifact"]["raw_bytes"]) - len(container)) / int(panel["weights"]),
        "incremental_same_framing_WFA_saving_bpw": 8.0 * (len(identity_container) - len(container)) / int(panel["weights"]),
        "raw_payload_minus_full_model_saving_bpw": payload_saved_bits / int(panel["weights"]),
        "payload_rows": payload_rows,
        "standalone_decode": standalone,
        "model_packet_sha256": hashlib.sha256(transmitted_model).hexdigest(),
        "container_sha256": hashlib.sha256(container).hexdigest(),
        "identity_framing_container_sha256": hashlib.sha256(identity_container).hexdigest(),
        "candidate": decoded_candidate.as_dict(),
        "all_adapted_values_deserialized_from_transmitted_model": True,
        "identical_reconstruction_proved_by_full_f64_digest": True,
    }


def prepare_panel(adapter: Any, artifact_bytes: bytes) -> dict[str, Any]:
    panel = adapter.extract_from_current(artifact_bytes)
    attach_semantic_owners(panel)
    return panel


def _result_without_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"container", "identity_framing_container"}}


def source_phase(
    *,
    common: Any,
    protocol: Any,
    container_codec: Any,
    adapter: Any,
    backend: Any,
    artifact_bytes: bytes,
    score_receipt_bytes: bytes,
    bindings: BoundEvidence,
    gpu_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    panel = prepare_panel(adapter, artifact_bytes)
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_sha != panel["artifact"]["raw_sha256"] or len(artifact_bytes) != panel["artifact"]["raw_bytes"]:
        raise ValueError("held artifact descriptor binding")
    score_record = common.strict_json_loads(score_receipt_bytes)
    score_sha = hashlib.sha256(score_receipt_bytes).hexdigest()
    if score_sha != bindings.baseline_score_sha256:
        raise ValueError("externally authorized score receipt digest")
    score = protocol.validate_score_receipt(
        score_record,
        artifact_sha256=artifact_sha,
        artifact_bytes=len(artifact_bytes),
        weights=int(panel["weights"]),
        reconstruction_sha256=str(panel["reconstruction"]["full_reconstruction_f64_sha256"]),
    )
    runtime_projection = projected_updates(common, protocol, panel)
    if not runtime_projection["passes_pre_fit_runtime_budget"]:
        return {
            "status": "ABORT_RUNTIME_BUDGET_BEFORE_FIT",
            "runtime_projection": runtime_projection,
            "controls_may_be_opened": False,
            "artifact_was_parsed_and_baseline_replayed": True,
        }
    cache = prepare_backend_cache(backend, panel)
    scientific = nested_holdout(common, protocol, backend, cache, panel)
    selected = common.candidate_bank()[int(scientific["final_topology_selected_from_nested_fold_votes"]["selector_ordinal"])]
    physical = final_container(common, container_codec, adapter, backend, cache, panel, selected, score, bindings)
    metrics = physical["parsed_metrics"]
    physical_pass = bool(metrics["passes_rate_interval"] and metrics["passes_F_target"])
    cold_pass = bool(metrics["passes_cold_read_below_2x"])
    heldout_pass = bool(scientific["passes_heldout_gate"])
    integrity_pass = bool(
        physical["standalone_decode"]["all_payloads_canonically_reencoded"]
        and physical["identical_reconstruction_proved_by_full_f64_digest"]
        and physical["all_adapted_values_deserialized_from_transmitted_model"]
    )
    if not integrity_pass:
        status = "FAIL_EVIDENCE_INTEGRITY_SOURCE_STANDALONE_DECODE"
    elif not physical_pass:
        status = "HARD_KILL_PHYSICAL_RATE_OR_F"
    elif not cold_pass:
        status = "FAIL_STRICT_COLD_READ"
    elif not heldout_pass:
        status = "NO_PROMOTION_NESTED_HELDOUT"
    else:
        status = "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED"
    shuffles = None
    if status == "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED":
        shuffles = survivor_shuffle_diagnostics(common, protocol, backend, panel)
    source_geometry = protocol.geometry_sha256(common, panel)
    elapsed = time.perf_counter() - started
    result = {
        "schema": "uwfa-sc-v2-source-phase-result",
        "status": status,
        "source_geometry_sha256": source_geometry,
        "source_pipeline_sha256": bindings.pipeline_sha256,
        "score_receipt_sha256": score_sha,
        "gpu_preflight": dict(gpu_preflight),
        "runtime_projection": runtime_projection,
        "scientific_nested_holdout": scientific,
        "predeclared_shuffle_diagnostics": shuffles,
        "source_final": _result_without_payload(physical),
        "source_phase_elapsed_seconds": elapsed,
        "controls_may_be_opened": status == "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED",
        "physical_Qwen_failure_is_final_regardless_of_controls": True,
        "claim_boundary": "frozen selected-SC-decision recoder only; Qwen is an evaluation panel, not a universal performance proof",
        "requires_external_fresh_process_independent_result_audit": True,
    }
    result["_container"] = physical["container"]
    result["_identity_framing_container"] = physical["identity_framing_container"]
    result["_panel"] = panel
    return result


def controls_phase(
    *,
    common: Any,
    protocol: Any,
    container_codec: Any,
    adapter_factory: Any,
    backend_factory: Any,
    source_result: Mapping[str, Any],
    source_artifact_sha256: str,
    controls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if source_result.get("controls_may_be_opened") is not True:
        raise ValueError("control payload access forbidden before all source gates")
    if len(controls) != len(common.CONTROL_SEEDS):
        raise ValueError("exact eight controls required")
    source_geometry = str(source_result["source_geometry_sha256"])
    source_pipeline = str(source_result["source_pipeline_sha256"])
    # First parse, authenticate, replay, and geometry-check all controls.  No
    # candidate fit occurs until this complete loop succeeds.
    prepared = []
    for expected_seed, item in zip(common.CONTROL_SEEDS, controls, strict=True):
        if set(item) != {"artifact_bytes", "score_receipt_bytes", "binding_record", "bindings"}:
            raise ValueError("unknown/missing control bundle field")
        adapter = adapter_factory()
        panel = prepare_panel(adapter, bytes(item["artifact_bytes"]))
        geometry = protocol.geometry_sha256(common, panel)
        if geometry != source_geometry:
            raise ValueError("Gaussian control full geometry differs from source")
        artifact_sha = hashlib.sha256(item["artifact_bytes"]).hexdigest()
        protocol.validate_control_binding(
            common,
            item["binding_record"],
            seed=int(expected_seed),
            source_artifact_sha256=source_artifact_sha256,
            source_geometry_sha256=source_geometry,
            source_pipeline_sha256=source_pipeline,
            control_artifact_sha256=artifact_sha,
            control_geometry_sha256=geometry,
        )
        score = common.strict_json_loads(bytes(item["score_receipt_bytes"]))
        score = protocol.validate_score_receipt(
            score,
            artifact_sha256=artifact_sha,
            artifact_bytes=len(item["artifact_bytes"]),
            weights=int(panel["weights"]),
            reconstruction_sha256=str(panel["reconstruction"]["full_reconstruction_f64_sha256"]),
        )
        prepared.append((expected_seed, adapter, panel, score, item["bindings"]))
    rows = []
    for seed, adapter, panel, score, bindings in prepared:
        backend = backend_factory()
        projection = projected_updates(common, protocol, panel)
        if not projection["passes_pre_fit_runtime_budget"]:
            return {
                "status": "ABORT_CONTROL_RUNTIME_BUDGET_BEFORE_FIT",
                "seed": seed,
                "positive_promotion": False,
            }
        cache = prepare_backend_cache(backend, panel)
        scientific = nested_holdout(common, protocol, backend, cache, panel)
        selected = common.candidate_bank()[int(scientific["final_topology_selected_from_nested_fold_votes"]["selector_ordinal"])]
        physical = final_container(common, container_codec, adapter, backend, cache, panel, selected, score, bindings)
        rows.append({
            "seed": seed,
            "scientific_nested_holdout": scientific,
            "final": _result_without_payload(physical),
            "repeated_complete_150_cell_selection_fit_pack_decode": True,
        })
    source_gain = float(source_result["source_final"]["absolute_saving_vs_bound_current_artifact_bpw"])
    strongest = max(float(row["final"]["absolute_saving_vs_bound_current_artifact_bpw"]) for row in rows)
    specificity = source_gain > strongest
    return {
        "schema": "uwfa-sc-v2-control-phase-result",
        "status": "PASS_MATCHED_NULL_SPECIFICITY_AWAITING_EXTERNAL_RESULT_AUDIT" if specificity else "NO_PROMOTION_GAUSSIAN_SPECIFICITY",
        "controls": rows,
        "specificity_statistic": "absolute physical saving bpw from each independently selected and packed full pipeline",
        "G_operational_source_bpw": source_gain,
        "strongest_matched_null_bpw": strongest,
        "source_minus_strongest_null_bpw": source_gain - strongest,
        "specificity_pass": specificity,
        "positive_promotion": False,
        "promotion_withheld_until_external_fresh_process_result_audit": True,
    }


def promotion_conjunction(
    *,
    physical: bool,
    cold: bool,
    heldout: bool,
    specificity: bool,
    standalone_decode: bool,
    integrity: bool,
    independent_result_audit: bool,
) -> bool:
    return all((physical, cold, heldout, specificity, standalone_decode, integrity, independent_result_audit))


def direct_main() -> int:
    # Do not parse arguments: a direct invocation must not touch any dynamic
    # input or output and imports no sibling module, NumPy, CuPy, or CUDA.
    print(DIRECT_LAUNCH_STATUS, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(direct_main())
