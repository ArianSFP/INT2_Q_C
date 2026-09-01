#!/usr/bin/env python3
"""Authenticated producer for the sparse unifilar WFA entropy census v1.

Lifecycle order is security-relevant.  A wrong token exits before output or
input access.  With the right token, an absent output directory is reserved;
then an external independent receipt and this package's source manifest are
verified using only the standard library.  Project code is imported only after
that bootstrap.  CuPy/CUDA is imported only after the source panel and its
current arithmetic baseline have also been held, authenticated, and replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from array import array
from pathlib import Path
from typing import Any, Iterable


AUTHORIZATION_BOOTSTRAP = "OPEN_AUTHENTICATED_UNIFILAR_WFA_CENSUS_AFTER_INDEPENDENT_SOURCE_REVIEW_V1"
REVIEW_SCHEMA_BOOTSTRAP = "unifilar-wfa-entropy-census-independent-source-review-v1"


class BootstrapError(RuntimeError):
    pass


def bootstrap_require(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapError(message)


def bootstrap_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_regular_leaf(path: Path) -> bytes:
    bootstrap_require(path.is_absolute(), f"bootstrap path not absolute: {path}")
    bootstrap_require(os.path.lexists(path), f"bootstrap path absent: {path}")
    info = os.lstat(path)
    bootstrap_require(not stat.S_ISLNK(info.st_mode), f"bootstrap symlink leaf forbidden: {path}")
    bootstrap_require(stat.S_ISREG(info.st_mode), f"bootstrap object not regular: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags)
    try:
        observed = os.fstat(fd)
        bootstrap_require(stat.S_ISREG(observed.st_mode), f"bootstrap descriptor not regular: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def bootstrap_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            bootstrap_require(key not in result, f"duplicate bootstrap JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise BootstrapError(f"nonfinite bootstrap JSON: {value}")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject_constant)
    except Exception as exc:
        raise BootstrapError(f"bootstrap JSON: {exc}") from exc
    bootstrap_require(isinstance(value, dict), "bootstrap JSON object")
    return value


def reserve_output(path: Path) -> None:
    bootstrap_require(path.is_absolute(), "output path must be absolute")
    bootstrap_require(not os.path.lexists(path), "output path already exists")
    os.mkdir(path, 0o700)
    payload = b'{"complete":false,"schema":"unifilar-wfa-run-state-v1"}\n'
    fd = os.open(
        str(path / "RUN_STATE.json"),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def bootstrap_source(review_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    package = Path(os.path.abspath(Path(__file__).parent))
    review_bytes = read_regular_leaf(review_path)
    review = bootstrap_json(review_bytes)
    bootstrap_require(review.get("schema") == REVIEW_SCHEMA_BOOTSTRAP, "review schema")
    bootstrap_require(review.get("status") == "PASS_INDEPENDENT_SOURCE_REVIEW", "independent review status")
    bootstrap_require(review.get("payload_authority_granted") is True, "payload authority absent")
    bootstrap_require(review.get("authorization_token") == AUTHORIZATION_BOOTSTRAP, "review token")
    claimed_review_seal = review.get("review_sha256")
    bootstrap_require(isinstance(claimed_review_seal, str) and len(claimed_review_seal) == 64, "review seal")
    clean_review = dict(review)
    clean_review.pop("review_sha256", None)
    canonical_review = json.dumps(clean_review, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    bootstrap_require(bootstrap_sha256(canonical_review) == claimed_review_seal, "review seal mismatch")
    manifest_path = package / "SOURCE_MANIFEST.json"
    manifest_bytes = read_regular_leaf(manifest_path)
    manifest_sha = bootstrap_sha256(manifest_bytes)
    bootstrap_require(review.get("reviewed_source_manifest_sha256") == manifest_sha, "reviewed manifest hash")
    manifest = bootstrap_json(manifest_bytes)
    bootstrap_require(manifest.get("schema") == "unifilar-wfa-source-manifest-v1", "source manifest schema")
    bootstrap_require(manifest.get("status") == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "source manifest status")
    members = manifest.get("members")
    bootstrap_require(isinstance(members, list) and members, "source manifest members")
    for row in members:
        bootstrap_require(isinstance(row, dict), "manifest member row")
        name = row.get("name")
        bootstrap_require(isinstance(name, str) and name == Path(name).name and name != "SOURCE_MANIFEST.json", "manifest member name")
        data = read_regular_leaf(package / name)
        bootstrap_require(len(data) == row.get("bytes"), f"manifest member bytes: {name}")
        bootstrap_require(bootstrap_sha256(data) == row.get("sha256"), f"manifest member hash: {name}")
    return package, review, manifest, manifest_sha


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    bootstrap_require(spec is not None and spec.loader is not None, f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--review-receipt", required=True)
    parser.add_argument("--stream-lock", required=True)
    parser.add_argument("--gaussian-control-lock", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def path_from_ref(common: Any, row: dict[str, Any], label: str) -> Path:
    common.require(isinstance(row, dict), f"{label} reference")
    path = Path(row.get("path", ""))
    common.require(path.is_absolute(), f"{label} path absolute")
    common.require(isinstance(row.get("bytes"), int) and row["bytes"] >= 0, f"{label} bytes")
    common.require(isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64, f"{label} sha256")
    return path


def u16le_values(common: Any, data: bytes) -> list[int]:
    common.require(len(data) % 2 == 0, "u16 byte geometry")
    values = array("H")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return [int(value) for value in values]


def load_panel(common: Any, lock_path: Path) -> dict[str, Any]:
    held = common.HeldFileSet()
    try:
        lock_file = held.add(common.HeldRegularFile(lock_path))
        lock = common.strict_json_loads(lock_file.read_all())
        common.require(isinstance(lock, dict) and lock.get("schema") == common.STREAM_LOCK_SCHEMA, "stream lock schema")
        common.verify_internal_seal(lock, "lock_sha256")
        weights = int(lock.get("weights", 0))
        current_object_bytes = int(lock.get("current_object_bytes", 0))
        common.require(weights > 0 and current_object_bytes > 0, "panel physical geometry")
        for label in ("current_artifact", "extraction_receipt"):
            ref = lock.get(label)
            path = path_from_ref(common, ref, label)
            held.add(common.HeldRegularFile(path, ref["bytes"], ref["sha256"]))
        expert_rows = lock.get("experts")
        common.require(isinstance(expert_rows, list) and expert_rows, "expert rows")
        experts: dict[int, int] = {}
        for row in expert_rows:
            ordinal = int(row.get("expert_ordinal", -1))
            immutable = int(row.get("immutable_local_bytes", -1))
            common.require(ordinal == len(experts) and immutable >= 0, "canonical expert rows")
            experts[ordinal] = immutable
        stream_rows = lock.get("streams")
        common.require(isinstance(stream_rows, list) and stream_rows, "stream rows")
        streams: list[dict[str, Any]] = []
        stream_keys: set[str] = set()
        total_weight_charge = 0
        for ordinal, row in enumerate(stream_rows):
            common.require(isinstance(row, dict), "stream row")
            key = row.get("stream_key")
            layer = row.get("layer_group")
            expert = row.get("expert_group")
            expert_ordinal = int(row.get("expert_ordinal", -1))
            weight_charge = int(row.get("weight_charge", 0))
            symbols = int(row.get("symbols", 0))
            logical_bits = int(row.get("original_logical_bits", 0))
            common.require(isinstance(key, str) and key and key not in stream_keys, "unique stream key")
            common.require(isinstance(layer, str) and layer and isinstance(expert, str) and expert, "partition labels")
            common.require(expert_ordinal in experts and weight_charge > 0 and symbols > 0 and logical_bits > 0, "stream geometry")
            stream_keys.add(key)
            total_weight_charge += weight_charge
            blobs: dict[str, bytes] = {}
            for label in ("selected_bits_u8", "polar_level_u8", "regenerated_base_freq1_u16le", "original_arithmetic_payload"):
                ref = row.get(label)
                path = path_from_ref(common, ref, f"stream {ordinal} {label}")
                item = held.add(common.HeldRegularFile(path, ref["bytes"], ref["sha256"]))
                blobs[label] = item.read_all()
            common.require(len(blobs["selected_bits_u8"]) == symbols, "selected-bit length")
            common.require(len(blobs["polar_level_u8"]) == symbols, "level length")
            common.require(len(blobs["regenerated_base_freq1_u16le"]) == 2 * symbols, "frequency length")
            common.require(len(blobs["original_arithmetic_payload"]) == (logical_bits + 7) // 8, "payload length")
            bits = list(blobs["selected_bits_u8"])
            levels = list(blobs["polar_level_u8"])
            base = u16le_values(common, blobs["regenerated_base_freq1_u16le"])
            common.require(all(bit in (0, 1) for bit in bits), "binary source stream")
            common.require(all(0 <= level < common.LEVELS for level in levels), "polar levels")
            common.require(all(1 <= value <= 65535 for value in base), "base frequencies")
            replay, replay_bits = common.arithmetic_encode_binary(bits, base)
            common.require(replay_bits == logical_bits, f"baseline logical replay: {key}")
            common.require(replay == blobs["original_arithmetic_payload"], f"baseline byte replay: {key}")
            streams.append(
                {
                    "stream_key": key,
                    "layer_group": layer,
                    "expert_group": expert,
                    "expert_ordinal": expert_ordinal,
                    "weight_charge": weight_charge,
                    "symbols": symbols,
                    "bits": bits,
                    "levels": levels,
                    "base": base,
                    "bits_bytes": blobs["selected_bits_u8"],
                    "levels_bytes": blobs["polar_level_u8"],
                    "base_bytes": blobs["regenerated_base_freq1_u16le"],
                    "baseline_payload_bytes": len(blobs["original_arithmetic_payload"]),
                    "baseline_logical_bits": logical_bits,
                }
            )
        common.require(total_weight_charge == weights, "stream weight charges must partition panel weights")
        held.verify_stable()
        return {
            "lock": lock,
            "lock_sha256": lock_file.sha256,
            "lock_bytes": lock_file.size,
            "weights": weights,
            "current_object_bytes": current_object_bytes,
            "immutable_global_bytes": int(lock.get("immutable_global_bytes", 0)),
            "experts": experts,
            "streams": streams,
            "held": held,
            "baseline_replayed_before_candidate": True,
        }
    except Exception:
        held.close()
        raise


def packed_rows(streams: Iterable[dict[str, Any]]) -> list[tuple[bytes, bytes, bytes]]:
    return [(row["bits_bytes"], row["levels_bytes"], row["base_bytes"]) for row in streams]


def fit_candidate(common: Any, backend: Any, streams: list[dict[str, Any]], candidate: Any) -> list[int]:
    common.require(bool(streams), "nonempty fit set")
    packed = backend.pack_streams(packed_rows(streams))
    counts_gpu = backend.fit_counts(packed, candidate.topology_id, candidate.states, candidate.reset_length)
    counts = [int(value) for value in counts_gpu.get().tolist()]
    return common.q16_frequencies_from_counts(counts)


def exact_lengths(common: Any, backend: Any, streams: list[dict[str, Any]], candidate: Any, frequencies: list[int]) -> list[int]:
    common.require(bool(streams), "nonempty score set")
    packed = backend.pack_streams(packed_rows(streams))
    rows = backend.exact_lengths(packed, candidate.topology_id, candidate.states, candidate.reset_length, frequencies)
    return [int(value) for value in rows.get().tolist()]


def gpu_preflight(common: Any, backend: Any) -> dict[str, Any]:
    n = 8192
    bits = bytes((((index * 17) ^ (index >> 3) ^ ((index // 257) & 1)) & 1) for index in range(n))
    levels = bytes((index * 5 + 1) % common.LEVELS for index in range(n))
    base_values = [1 + ((index * 7919 + 17) % 65535) for index in range(n)]
    base_array = array("H", base_values)
    if sys.byteorder != "little":
        base_array.byteswap()
    packed = backend.pack_streams([(bits, levels, base_array.tobytes())])
    checked = []
    for candidate in (
        common.Candidate("suffix", 4, 32),
        common.Candidate("xor_sketch", 16, 512),
        common.Candidate("rolling_affine", 64, 4096),
        common.Candidate("signed_saturating", 8, 128),
    ):
        cpu_counts = common.count_stream_cpu(list(bits), list(levels), base_values, candidate)
        gpu_counts = [int(value) for value in backend.fit_counts(packed, candidate.topology_id, candidate.states, candidate.reset_length).get().tolist()]
        common.require(cpu_counts == gpu_counts, f"GPU count mismatch: {candidate}")
        frequencies = common.q16_frequencies_from_counts(cpu_counts)
        cpu_length = common.exact_stream_length_cpu(list(bits), list(levels), base_values, candidate, frequencies)
        gpu_length = int(backend.exact_lengths(packed, candidate.topology_id, candidate.states, candidate.reset_length, frequencies).get().tolist()[0])
        common.require(cpu_length == gpu_length, f"GPU arithmetic length mismatch: {candidate}")
        checked.append({**candidate.as_dict(), "symbols": n, "exact_logical_bits": cpu_length})
    return {"status": "PASS_CPU_CUPY_EXACT", "cells": checked}


def validation_score(common: Any, lengths: list[int], candidate: Any) -> int:
    return 8 * sum((value + 7) // 8 for value in lengths) + 8 * common.model_ledger(candidate)["physical_model_bytes"]


def nested_holdout(common: Any, backend: Any, panel: dict[str, Any]) -> dict[str, Any]:
    streams = panel["streams"]
    pairs = sorted({(row["layer_group"], row["expert_group"]) for row in streams})
    fold_rows = []
    for layer, expert in pairs:
        test = [row for row in streams if row["layer_group"] == layer and row["expert_group"] == expert]
        development = [row for row in streams if row["layer_group"] != layer and row["expert_group"] != expert]
        common.require(test and len(development) >= 2, f"nonempty nested fold: {layer}/{expert}")
        ranked_development = sorted(
            development,
            key=lambda row: (
                common.nested_split_digest(row["layer_group"], row["expert_group"], row["stream_key"]),
                row["stream_key"],
            ),
        )
        validation_count = max(1, len(ranked_development) // common.INNER_VALIDATION_MODULUS)
        validation_count = min(validation_count, len(ranked_development) - 1)
        inner_validation = ranked_development[:validation_count]
        inner_train = ranked_development[validation_count:]
        selections = []
        for candidate in common.candidate_bank():
            frequencies = fit_candidate(common, backend, inner_train, candidate)
            lengths = exact_lengths(common, backend, inner_validation, candidate, frequencies)
            selections.append((validation_score(common, lengths, candidate), candidate.selector_ordinal, candidate))
        _, _, selected = min(selections)
        final_frequencies = fit_candidate(common, backend, development, selected)
        test_lengths = exact_lengths(common, backend, test, selected, final_frequencies)
        baseline_bytes = sum(row["baseline_payload_bytes"] for row in test)
        candidate_bytes = sum((value + 7) // 8 for value in test_lengths)
        model_bytes = common.model_ledger(selected)["physical_model_bytes"]
        weight_charge = sum(row["weight_charge"] for row in test)
        saving_bpw = 8.0 * (baseline_bytes - candidate_bytes - model_bytes) / weight_charge
        fold_rows.append(
            {
                "outer_layer_group": layer,
                "outer_expert_group": expert,
                "test_stream_keys": [row["stream_key"] for row in test],
                "development_stream_count": len(development),
                "inner_train_stream_count": len(inner_train),
                "inner_validation_stream_count": len(inner_validation),
                "selected_by_inner_validation_only": selected.as_dict(),
                "test_baseline_payload_bytes": baseline_bytes,
                "test_candidate_payload_bytes": candidate_bytes,
                "charged_fold_model_bytes": model_bytes,
                "test_weight_charge": weight_charge,
                "exact_test_saving_bpw": saving_bpw,
            }
        )
    total_weights = sum(row["test_weight_charge"] for row in fold_rows)
    common.require(total_weights == panel["weights"], "outer test folds partition weight charges")
    pooled_saved_bits = sum(
        8 * (row["test_baseline_payload_bytes"] - row["test_candidate_payload_bytes"] - row["charged_fold_model_bytes"])
        for row in fold_rows
    )
    pooled_bpw = pooled_saved_bits / total_weights
    candidate_votes: dict[int, int] = {}
    for row in fold_rows:
        ordinal = int(row["selected_by_inner_validation_only"]["selector_ordinal"])
        candidate_votes[ordinal] = candidate_votes.get(ordinal, 0) + 1
    selected_ordinal = min(candidate_votes, key=lambda ordinal: (-candidate_votes[ordinal], ordinal))
    selected = common.candidate_bank()[selected_ordinal]
    return {
        "kind": "scientific_fold_replicas_not_one_physical_packet",
        "selection_never_uses_its_outer_test_fold": True,
        "folds": fold_rows,
        "pooled_exact_heldout_saving_bpw": pooled_bpw,
        "minimum_fold_exact_saving_bpw": min(row["exact_test_saving_bpw"] for row in fold_rows),
        "candidate_vote_counts": candidate_votes,
        "final_topology_selected_from_nested_fold_votes": selected.as_dict(),
        "passes_heldout_standalone_threshold": pooled_bpw >= common.STANDALONE_REQUIRED_SAVING_BPW,
    }


def final_packet(common: Any, backend: Any, panel: dict[str, Any], candidate: Any, emit_payload: bool) -> dict[str, Any]:
    frequencies = fit_candidate(common, backend, panel["streams"], candidate)
    model_packet = common.serialize_model(candidate, frequencies)
    per_expert: list[list[int]] = [[] for _ in panel["experts"]]
    payload_rows = []
    concatenated = bytearray()
    for row in panel["streams"]:
        payload, logical_bits = common.encode_unifilar_stream(row["bits"], row["levels"], row["base"], candidate, frequencies)
        decoded = common.decode_unifilar_stream(payload, logical_bits, row["levels"], row["base"], candidate, frequencies)
        common.require(decoded == row["bits"], f"independent final decode: {row['stream_key']}")
        offset = len(concatenated)
        concatenated.extend(payload)
        per_expert[row["expert_ordinal"]].append(len(payload))
        payload_rows.append(
            {
                "stream_key": row["stream_key"],
                "offset_bytes": offset,
                "payload_bytes": len(payload),
                "logical_bits": logical_bits,
                "sha256": common.sha256_bytes(payload),
            }
        )
    ledger = common.packet_ledger(
        weights=panel["weights"],
        current_object_bytes=panel["current_object_bytes"],
        immutable_global_bytes=panel["immutable_global_bytes"],
        immutable_local_bytes=[panel["experts"][ordinal] for ordinal in range(len(panel["experts"]))],
        model_packet_bytes=len(model_packet),
        stream_payload_bytes=per_expert,
    )
    result = {
        "kind": "one_final_whole_panel_two_part_physical_packet",
        "candidate_fixed_by_nested_holdout_before_full_panel_parameter_fit": candidate.as_dict(),
        "model_packet_bytes": len(model_packet),
        "model_packet_sha256": common.sha256_bytes(model_packet),
        "concatenated_payload_bytes": len(concatenated),
        "concatenated_payload_sha256": common.sha256_bytes(bytes(concatenated)),
        "streams": payload_rows,
        "ledger": ledger,
    }
    if emit_payload:
        result["_model_packet"] = model_packet
        result["_payload_packet"] = bytes(concatenated)
    return result


def load_control_locks(common: Any, path: Path) -> tuple[Any, list[tuple[int, Path, dict[str, Any]]]]:
    held = common.HeldRegularFile(path).open()
    try:
        record = common.strict_json_loads(held.read_all())
        common.require(record.get("schema") == common.CONTROL_LOCK_SCHEMA, "control lock schema")
        common.verify_internal_seal(record, "lock_sha256")
        rows = record.get("controls")
        common.require(isinstance(rows, list) and len(rows) == len(common.CONTROL_SEEDS), "control count")
        output = []
        for expected_seed, row in zip(common.CONTROL_SEEDS, rows, strict=True):
            common.require(int(row.get("seed", -1)) == expected_seed, "control seed/order")
            ref = row.get("stream_lock")
            lock_path = path_from_ref(common, ref, "control stream lock")
            output.append((expected_seed, lock_path, ref))
        held.verify_stable()
        return held, output
    except Exception:
        held.close()
        raise


def write_new(common: Any, output: Path, name: str, data: bytes) -> dict[str, Any]:
    common.require(name == Path(name).name and name not in {"", ".", ".."}, "output member")
    target = output / name
    fd = os.open(
        str(target),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            common.require(count > 0, f"short output write: {name}")
            written += count
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"name": name, "bytes": len(data), "sha256": common.sha256_bytes(data)}


def main() -> int:
    args = parse_arguments()
    if args.authorization != AUTHORIZATION_BOOTSTRAP:
        print("REJECT_WRONG_TOKEN: output and inputs untouched; project code and CuPy not imported", file=sys.stderr)
        return 2
    output = Path(args.output)
    reserve_output(output)
    package, review, manifest, manifest_sha = bootstrap_source(Path(args.review_receipt))
    common = load_module("uwfa_common", package / "uwfa_common.py")
    common.require(args.authorization == common.AUTHORIZATION, "authorization drift")
    source_panel = load_panel(common, Path(args.stream_lock))
    try:
        # This is the first permitted CuPy import.  Source closure, review,
        # stream authentication, and every current arithmetic replay passed.
        import cupy as cp  # type: ignore

        backend_module = load_module("uwfa_cupy_backend", package / "cupy_backend.py")
        backend = backend_module.build_backend(cp)
        preflight = gpu_preflight(common, backend)
        scientific = nested_holdout(common, backend, source_panel)
        selected = common.candidate_bank()[scientific["final_topology_selected_from_nested_fold_votes"]["selector_ordinal"]]
        source_final = final_packet(common, backend, source_panel, selected, emit_payload=True)
        physical_pass = bool(source_final["ledger"]["passes_absolute_physical_target"] and source_final["ledger"]["passes_cold_read"])
        heldout_pass = bool(scientific["passes_heldout_standalone_threshold"])
        controls: list[dict[str, Any]] = []
        control_lock_held = None
        control_panels: list[dict[str, Any]] = []
        if physical_pass and heldout_pass:
            control_lock_held, control_rows = load_control_locks(common, Path(args.gaussian_control_lock))
            # First load *all* controls. load_panel independently authenticates
            # and exactly replays every Gaussian current baseline. No candidate
            # fitting begins until this loop has completely succeeded.
            for seed, lock_path, ref in control_rows:
                panel = load_panel(common, lock_path)
                common.require(panel["lock_sha256"] == ref["sha256"], "control stream-lock digest")
                common.require(panel["lock_bytes"] == ref["bytes"], "control stream-lock bytes")
                control_panels.append(panel)
            for seed, panel in zip(common.CONTROL_SEEDS, control_panels, strict=True):
                row = final_packet(common, backend, panel, selected, emit_payload=False)
                controls.append({"seed": seed, **row})
        source_saving = float(source_final["ledger"]["net_physical_saving_bpw"])
        specificity_pass = bool(controls) and all(
            source_saving > float(row["ledger"]["net_physical_saving_bpw"]) for row in controls
        )
        if not physical_pass:
            status = "HARD_KILL_EXACT_PHYSICAL_SOURCE_PACKET"
        elif not heldout_pass:
            status = "NO_PROMOTION_NESTED_HELDOUT_THRESHOLD_FAIL"
        elif not specificity_pass:
            status = "NO_PROMOTION_GAUSSIAN_SPECIFICITY_FAIL"
        else:
            status = "PROMOTE_FINITE_UNIVERSAL_CELL"
        model_packet = source_final.pop("_model_packet")
        payload_packet = source_final.pop("_payload_packet")
        result = common.seal_record(
            {
                "schema": common.RESULT_SCHEMA,
                "status": status,
                "source_manifest_sha256": manifest_sha,
                "independent_review_sha256": common.sha256_file(Path(args.review_receipt)),
                "access_order": {
                    "external_manifest_before_project_import": True,
                    "source_baseline_replayed_before_cupy": True,
                    "every_gaussian_baseline_replayed_before_any_control_candidate": bool(controls),
                },
                "gpu_preflight": preflight,
                "scientific_nested_holdout": scientific,
                "source_final_physical_packet": source_final,
                "gaussian_controls": controls,
                "controls_can_never_make_physical_fail_pass": True,
                "positive_status_requires_physical_and_heldout_and_control_specificity": True,
                "claim_boundary": "selected SC arithmetic bits only; not source-coordinate label copula and not arbitrary MPS/MERA/TTN",
            },
            "result_sha256",
        )
        members = [
            write_new(common, output, "final_model.bin", model_packet),
            write_new(common, output, "final_arithmetic_payloads.bin", payload_packet),
            write_new(common, output, "result.json", common.pretty_json(result)),
        ]
        source_panel["held"].verify_stable()
        for panel in control_panels:
            panel["held"].verify_stable()
        complete = common.seal_record(
            {
                "schema": "unifilar-wfa-completion-v1",
                "status": "COMPLETE_LAST",
                "source_manifest_sha256": manifest_sha,
                "members": members,
            },
            "completion_sha256",
        )
        write_new(common, output, "COMPLETE.json", common.pretty_json(complete))
        return 0
    finally:
        source_panel["held"].close()
        for panel in locals().get("control_panels", []):
            panel["held"].close()
        if locals().get("control_lock_held") is not None:
            control_lock_held.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BootstrapError, Exception) as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
