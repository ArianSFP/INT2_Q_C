#!/usr/bin/env python3
"""Literal-packet, independent-decode physical result authority.

The validator accepts no decoded arrays, rates, MSE values, selected config,
packet hashes, read totals, or universality booleans from its caller.  Those
objects are read from an externally hash-pinned commitment and recomputed from
literal files after a fresh ``python -I -B`` decoder invocation.

The bundled fixture decoder is mechanism-only.  A production commitment must
pin a separately audited production decoder under the authenticated external
root and use ``mode=production_global_rm_swap``.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from authority import (AuthorityError, EXTERNAL_PINS,
                       authenticate_current_external_root, canonical_json,
                       is_sha256, regular_bytes, require,
                       sanitized_worker_environment, sha256, strict_json)


RATE_MIN = 2.15
RATE_MAX = 2.5
TARGET_F = 0.8
ROLE_ORDER = ("gate", "up", "down")
PRODUCTION_AUTHORIZATION = "AUDIT_LITERAL_GLOBAL_RM_SWAP_RESULT_V1"
FIXTURE_AUTHORIZATION = "SOURCE_ONLY_SYNTHETIC_AUTHORITY_FIXTURE_V1"
REQUIRED_DECODER_FIELDS = {
    "schema", "case_id", "packet_sha256", "packet_bytes",
    "canonical_packet_sha256", "canonical_packet_bytes",
    "independent_decode_complete", "canonical_reencode_complete",
    "causal_probabilities_regenerated", "packet_consumed_exactly",
    "encoder_decisions_read", "encoder_probabilities_read",
    "source_payloads_opened", "reconstruction_files", "read_trace_file",
    "status",
}


def _safe_relative(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label}: relative path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts and
            "." not in pure.parts and "\\" not in value,
            f"{label}: safe POSIX relative path")
    return Path(*pure.parts)


def _resolve_member(root: Path, relative: Any, label: str) -> Path:
    rel = _safe_relative(relative, label)
    try:
        resolved_root = root.resolve(strict=True)
        current = resolved_root
        for part in rel.parts:
            current = current / part
            before = current.lstat()
            require(not stat.S_ISLNK(before.st_mode),
                    f"{label}: symlink path component")
        candidate = current.resolve(strict=True)
    except OSError as exc:
        raise AuthorityError(f"{label}: resolution") from exc
    require(resolved_root in candidate.parents and candidate != resolved_root,
            f"{label}: containment")
    return candidate


def _read_pinned(root: Path, relative: Any, expected_bytes: Any,
                 expected_sha256: Any, label: str) -> tuple[Path, bytes]:
    require(isinstance(expected_bytes, int) and expected_bytes > 0 and
            is_sha256(expected_sha256), f"{label}: pin metadata")
    path = _resolve_member(root, relative, label)
    payload = regular_bytes(path, label)
    require(len(payload) == expected_bytes and sha256(payload) == expected_sha256,
            f"{label}: literal byte pin")
    return path, payload


def _bf16_values(payload: bytes):
    require(payload and len(payload) % 2 == 0, "nonempty BF16 source bytes")
    words = array.array("H")
    words.frombytes(payload)
    if sys.byteorder != "little":
        words.byteswap()
    wide = array.array("I", (int(value) << 16 for value in words))
    values = array.array("f")
    values.frombytes(wide.tobytes())
    return values


def _f64_values(payload: bytes):
    require(payload and len(payload) % 8 == 0, "nonempty FP64 reconstruction")
    values = array.array("d")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def exact_bf16_f64_score(source_bf16: bytes,
                         reconstruction_f64: bytes) -> dict[str, Any]:
    source = _bf16_values(source_bf16)
    reconstruction = _f64_values(reconstruction_f64)
    require(len(source) == len(reconstruction), "source/reconstruction count")
    require(all(math.isfinite(value) for value in source) and
            all(math.isfinite(value) for value in reconstruction),
            "finite source/reconstruction")
    energy = math.fsum(float(value) * float(value) for value in source)
    sse = math.fsum((float(left) - float(right)) ** 2
                    for left, right in zip(source, reconstruction, strict=True))
    require(energy > 0.0 and math.isfinite(energy) and math.isfinite(sse),
            "finite positive scoring domain")
    return {"weights": len(source), "sse_fp64_hex": sse.hex(),
            "energy_fp64_hex": energy.hex(), "relative_mse": sse / energy}


def _strict_commitment(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(is_sha256(expected_sha256), "external commitment SHA-256")
    payload = regular_bytes(path, "experiment commitment")
    require(sha256(payload) == expected_sha256,
            "experiment commitment external hash")
    record = strict_json(payload, "experiment commitment")
    require(canonical_json(record) + b"\n" == payload,
            "experiment commitment canonical bytes")
    return record


def _validate_commitment(record: Mapping[str, Any], *, mode: str) -> None:
    required = {"schema", "mode", "v0_source_root_sha256",
                "v0_audit_source_root_sha256", "external_pins", "decoder_worker",
                "cases", "universal_contract", "shared_model_bytes",
                "selection_frozen_before_test", "test_bytes_opened_during_selection"}
    require(set(record) == required and record.get("schema") ==
            "strata-rm-global-swap-v1-physical-commitment",
            "experiment commitment schema")
    require(record.get("mode") == mode, "experiment commitment mode")
    require(record.get("v0_source_root_sha256") ==
            "4f856e268d37ee1d6f32b4a2d1b8cd6879c235639ad75809ffd75fc7c4372d6c" and
            record.get("v0_audit_source_root_sha256") ==
            "7eabe4580908d4a79eceb2f7fdaf838d535028c06263c2f4841032664db11ad0" and
            record.get("external_pins") == EXTERNAL_PINS,
            "experiment dependency roots")
    require(record.get("selection_frozen_before_test") is True and
            record.get("test_bytes_opened_during_selection") is False,
            "sealed selection boundary")
    require(isinstance(record.get("shared_model_bytes"), int) and
            record["shared_model_bytes"] >= 0, "shared model bytes")
    worker = record.get("decoder_worker")
    require(isinstance(worker, dict) and set(worker) ==
            {"relative_path", "sha256", "protocol", "independent_from_encoder",
             "independent_audit"} and
            is_sha256(worker.get("sha256")) and
            worker.get("protocol") == "strata-rm-v1-decoder-worker-protocol" and
            worker.get("independent_from_encoder") is True,
            "decoder worker commitment")
    if mode == "production_global_rm_swap":
        audit = worker.get("independent_audit")
        require(isinstance(audit, dict) and set(audit) ==
                {"manifest_relative_path", "manifest_sha256",
                 "source_root_sha256"} and
                is_sha256(audit.get("manifest_sha256")) and
                is_sha256(audit.get("source_root_sha256")),
                "production independent decoder audit commitment")
    else:
        require(worker.get("independent_audit") is None,
                "fixture has no production decoder-audit authority")
    cases = record.get("cases")
    require(isinstance(cases, list) and cases, "committed cases")
    ids = [row.get("case_id") for row in cases if isinstance(row, dict)]
    require(len(ids) == len(cases) == len(set(ids)) and
            all(isinstance(value, str) and value for value in ids),
            "unique committed case IDs")


def _decoder_command(worker: Path, request: Path, packet: Path,
                     output_dir: Path) -> list[str]:
    return [sys.executable, "-I", "-B", str(worker), "--request", str(request),
            "--packet", str(packet), "--output-dir", str(output_dir)]


def _case_schema(case: Mapping[str, Any]) -> None:
    required = {"case_id", "kind", "architecture_family", "pipeline_id",
                "matched_case_id", "packet", "sources", "charged_shared_bytes"}
    require(set(case) == required and case["kind"] in
            {"qwen_bf16", "swiglu_moe_bf16", "matched_gaussian_bf16",
             "synthetic_fixture"},
            "case schema")
    require(isinstance(case["architecture_family"], str) and
            case["architecture_family"] and isinstance(case["pipeline_id"], str) and
            case["pipeline_id"], "case family/pipeline")
    packet = case["packet"]
    require(isinstance(packet, dict) and set(packet) ==
            {"relative_path", "bytes", "sha256"}, "case packet schema")
    require(isinstance(case["charged_shared_bytes"], int) and
            case["charged_shared_bytes"] >= 0, "case shared bytes")
    sources = case["sources"]
    require(isinstance(sources, list) and sources, "case source rows")
    ordinals = []
    for source in sources:
        required_source = {"ordinal", "role", "layer", "expert", "shape",
                           "source_relative_path", "source_bytes", "source_sha256"}
        require(isinstance(source, dict) and set(source) == required_source,
                "source row schema")
        ordinal = source["ordinal"]
        shape = source["shape"]
        require(isinstance(ordinal, int) and ordinal >= 0 and
                isinstance(shape, list) and len(shape) == 2 and
                all(isinstance(value, int) and value > 0 for value in shape) and
                source["role"] in ROLE_ORDER and
                source["source_bytes"] == 2 * shape[0] * shape[1],
                "source row geometry")
        ordinals.append(ordinal)
    require(ordinals == list(range(len(sources))), "contiguous source ordinals")
    groups: dict[tuple[int, int], dict[str, list[int]]] = {}
    for source in sources:
        key = (source["layer"], source["expert"])
        require(all(isinstance(value, int) and value >= 0 for value in key),
                "nonnegative layer/expert")
        require(source["role"] not in groups.setdefault(key, {}),
                "one source per role")
        groups[key][source["role"]] = source["shape"]
    for roles in groups.values():
        require(set(roles) == set(ROLE_ORDER), "complete SwiGLU role triplet")
        require(roles["gate"] == roles["up"] and
                roles["down"] == [roles["gate"][1], roles["gate"][0]],
                "SwiGLU Gate/Up/Down shape compatibility")


def _audit_root(rows: list[dict[str, Any]]) -> str:
    return sha256(canonical_json(rows))


def _authenticate_decoder_audit(external_root: Path, worker_sha256: str,
                                audit_pin: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _resolve_member(
        external_root, audit_pin["manifest_relative_path"],
        "decoder independent-audit manifest")
    payload = regular_bytes(manifest_path, "decoder independent-audit manifest")
    require(sha256(payload) == audit_pin["manifest_sha256"],
            "decoder independent-audit external manifest pin")
    manifest = strict_json(payload, "decoder independent-audit manifest")
    require(manifest.get("schema") ==
            "strata-rm-v1-production-decoder-independent-audit-manifest" and
            manifest.get("producer_worker_sha256") == worker_sha256 and
            manifest.get("source_root_sha256") == audit_pin["source_root_sha256"],
            "decoder independent-audit producer/root binding")
    rows = manifest.get("members")
    require(isinstance(rows, list) and rows and
            _audit_root(rows) == audit_pin["source_root_sha256"],
            "decoder independent-audit member root")
    directory = manifest_path.parent
    names = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"},
                "decoder independent-audit member schema")
        name = row["name"]
        require(isinstance(name, str) and Path(name).name == name and
                name not in names and name != manifest_path.name,
                "decoder independent-audit member name")
        member = regular_bytes(directory / name,
                               f"decoder independent-audit member {name}")
        require(len(member) == row["bytes"] and sha256(member) == row["sha256"],
                f"decoder independent-audit member pin {name}")
        names.append(name)
    entries = list(os.scandir(directory))
    require({entry.name for entry in entries} == set(names) | {manifest_path.name} and
            all(entry.is_file(follow_symlinks=False) for entry in entries),
            "decoder independent-audit exact closure")
    return {"manifest_sha256": audit_pin["manifest_sha256"],
            "source_root_sha256": audit_pin["source_root_sha256"],
            "producer_worker_sha256": worker_sha256}


def _validate_trace(trace: Mapping[str, Any], packet_bytes: int) -> dict[str, Any]:
    require(set(trace) == {"schema", "packet_bytes", "operations"} and
            trace["schema"] == "strata-rm-v1-read-trace" and
            trace["packet_bytes"] == packet_bytes and
            isinstance(trace["operations"], list) and trace["operations"],
            "read trace schema")
    total = 0
    intervals = []
    for row in trace["operations"]:
        require(isinstance(row, dict) and set(row) ==
                {"object", "offset", "length"} and row["object"] == "packet" and
                isinstance(row["offset"], int) and row["offset"] >= 0 and
                isinstance(row["length"], int) and row["length"] > 0 and
                row["offset"] + row["length"] <= packet_bytes,
                "packet-local read operation")
        total += row["length"]
        intervals.append((row["offset"], row["offset"] + row["length"]))
    intervals.sort()
    covered = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            covered += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    covered += end - start
    require(covered == packet_bytes, "decoder trace covers literal packet")
    amplification = total / packet_bytes
    return {"literal_read_bytes": total, "unique_packet_bytes": covered,
            "read_amplification": amplification,
            "one_expert_local_object": True}


def _run_case(case: Mapping[str, Any], *, evidence_root: Path,
              decoder_worker: Path, mode: str,
              timeout_seconds: int) -> dict[str, Any]:
    _case_schema(case)
    packet_path, packet = _read_pinned(
        evidence_root, case["packet"]["relative_path"],
        case["packet"]["bytes"], case["packet"]["sha256"],
        f"packet {case['case_id']}")
    source_payloads = []
    request_sources = []
    for source in case["sources"]:
        _, payload = _read_pinned(
            evidence_root, source["source_relative_path"], source["source_bytes"],
            source["source_sha256"], f"source {case['case_id']}:{source['ordinal']}")
        source_payloads.append(payload)
        request_sources.append({key: source[key] for key in
                                ("ordinal", "role", "layer", "expert", "shape")})
    request = {"schema": "strata-rm-v1-decoder-request",
               "case_id": case["case_id"], "packet_sha256": sha256(packet),
               "packet_bytes": len(packet), "sources": request_sources}
    with tempfile.TemporaryDirectory(prefix="strata-rm-v1-decode-") as directory:
        root = Path(directory).resolve(strict=True)
        request_path = root / "request.json"
        output_dir = root / "output"
        output_dir.mkdir()
        request_path.write_bytes(canonical_json(request) + b"\n")
        completed = subprocess.run(
            _decoder_command(decoder_worker, request_path, packet_path, output_dir),
            cwd=root, env=sanitized_worker_environment(), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout_seconds, check=False)
        require(completed.returncode == 0,
                "independent decoder failed: " +
                completed.stderr.decode("utf-8", errors="replace")[-2000:])
        receipt = strict_json(regular_bytes(output_dir / "receipt.json",
                                            "decoder receipt"), "decoder receipt")
        require(set(receipt) == REQUIRED_DECODER_FIELDS,
                "decoder receipt exact fields")
        canonical = regular_bytes(output_dir / "canonical_packet.bin",
                                  "canonical replay packet")
        trace = strict_json(regular_bytes(output_dir / "read_trace.json",
                                          "decoder read trace"), "decoder read trace")
        reconstructions = []
        for source in case["sources"]:
            expected_name = f"reconstruction-{source['ordinal']:04d}.f64"
            path = output_dir / expected_name
            reconstructions.append(regular_bytes(path, expected_name))
    require(receipt["schema"] == "strata-rm-v1-independent-decoder-receipt" and
            receipt["case_id"] == case["case_id"] and
            receipt["packet_sha256"] == sha256(packet) and
            receipt["packet_bytes"] == len(packet) and
            receipt["canonical_packet_sha256"] == sha256(canonical) and
            receipt["canonical_packet_bytes"] == len(canonical) and
            receipt["reconstruction_files"] ==
            [f"reconstruction-{row['ordinal']:04d}.f64" for row in case["sources"]] and
            receipt["read_trace_file"] == "read_trace.json",
            "decoder receipt byte/identity binding")
    require(receipt["independent_decode_complete"] is True and
            receipt["canonical_reencode_complete"] is True and
            receipt["packet_consumed_exactly"] is True and
            receipt["encoder_decisions_read"] is False and
            receipt["encoder_probabilities_read"] is False and
            receipt["source_payloads_opened"] is False,
            "independent decoder authority semantics")
    if mode == "production_global_rm_swap":
        require(receipt["causal_probabilities_regenerated"] is True,
                "production causal probability regeneration")
    else:
        require(receipt["causal_probabilities_regenerated"] is False,
                "fixture cannot claim polar causal regeneration")
    require(canonical == packet, "literal canonical replay byte identity")
    rows = [exact_bf16_f64_score(source, reconstruction)
            for source, reconstruction in zip(source_payloads, reconstructions,
                                               strict=True)]
    weights = sum(row["weights"] for row in rows)
    sse = math.fsum(float.fromhex(row["sse_fp64_hex"]) for row in rows)
    energy = math.fsum(float.fromhex(row["energy_fp64_hex"]) for row in rows)
    physical_bytes = len(packet) + case["charged_shared_bytes"]
    rate = 8.0 * physical_bytes / weights
    relative = sse / energy
    return {
        "case_id": case["case_id"], "kind": case["kind"],
        "architecture_family": case["architecture_family"],
        "pipeline_id": case["pipeline_id"], "matched_case_id": case["matched_case_id"],
        "source_geometry": [{key: source[key] for key in
                             ("ordinal", "role", "layer", "expert", "shape")}
                            for source in case["sources"]],
        "literal_packet_sha256": sha256(packet), "literal_packet_bytes": len(packet),
        "canonical_reencode_byte_identical": True, "weights": weights,
        "sse_fp64_hex": sse.hex(), "energy_fp64_hex": energy.hex(),
        "physical_bytes": physical_bytes, "physical_rate_bpw": rate,
        "relative_mse": relative, "F": relative * 2.0 ** (2.0 * rate),
        "read": _validate_trace(trace, len(packet)), "matrix_rows": rows,
    }


def validate_physical_bundle(*, evidence_root: Path, external_root: Path,
                             commitment_path: Path,
                             expected_commitment_sha256: str,
                             authorization: str,
                             timeout_seconds: int = 3600) -> dict[str, Any]:
    """Validate one externally committed bundle from literal bytes.

    Production mode requires the explicit production authorization token,
    matched controls, target rate/F, `<2x` reads, and universal portability.
    The fixture mode can validate only this verifier's mechanics.
    """
    mode = ("production_global_rm_swap" if authorization == PRODUCTION_AUTHORIZATION
            else "synthetic_authority_fixture" if authorization == FIXTURE_AUTHORIZATION
            else None)
    require(mode is not None, "explicit physical authority token")
    try:
        evidence_stat = Path(evidence_root).lstat()
        require(stat.S_ISDIR(evidence_stat.st_mode) and
                not Path(evidence_root).is_symlink(),
                "evidence root real directory")
        evidence = Path(evidence_root).resolve(strict=True)
    except OSError as exc:
        raise AuthorityError("evidence root resolution") from exc
    authenticate_current_external_root(external_root)
    external = Path(external_root).resolve(strict=True)
    try:
        commitment_resolved = Path(commitment_path).resolve(strict=True)
    except OSError as exc:
        raise AuthorityError("commitment path resolution") from exc
    require(evidence in commitment_resolved.parents,
            "commitment must be inside evidence root")
    commitment = _strict_commitment(commitment_path, expected_commitment_sha256)
    _validate_commitment(commitment, mode=mode)
    worker_row = commitment["decoder_worker"]
    worker = _resolve_member(external, worker_row["relative_path"], "decoder worker")
    worker_payload = regular_bytes(worker, "decoder worker")
    require(sha256(worker_payload) == worker_row["sha256"],
            "decoder worker source pin")
    decoder_audit = None
    if mode == "production_global_rm_swap":
        decoder_audit = _authenticate_decoder_audit(
            external, worker_row["sha256"], worker_row["independent_audit"])
    results = [_run_case(case, evidence_root=evidence, decoder_worker=worker,
                         mode=mode,
                         timeout_seconds=timeout_seconds)
               for case in commitment["cases"]]
    by_id = {row["case_id"]: row for row in results}
    universal = commitment["universal_contract"]
    require(isinstance(universal, dict) and set(universal) ==
            {"roles", "shape_parameterized", "qwen_specific_tables",
             "model_family_agnostic", "architecture_families"} and
            universal["roles"] == list(ROLE_ORDER) and
            universal["shape_parameterized"] is True and
            universal["qwen_specific_tables"] is False and
            universal["model_family_agnostic"] is True and
            isinstance(universal["architecture_families"], list),
            "universal SwiGLU contract")
    qwen = [row for row in results if row["kind"] == "qwen_bf16"]
    model_cases = [row for row in results if row["kind"] in
                   {"qwen_bf16", "swiglu_moe_bf16"}]
    controls = [row for row in results if row["kind"] == "matched_gaussian_bf16"]
    if mode == "production_global_rm_swap":
        require(qwen and controls and model_cases, "production model and matched controls")
        for row in model_cases:
            require(row["matched_case_id"] in by_id and
                    by_id[row["matched_case_id"]]["kind"] == "matched_gaussian_bf16" and
                    by_id[row["matched_case_id"]]["weights"] == row["weights"] and
                    by_id[row["matched_case_id"]]["pipeline_id"] == row["pipeline_id"] and
                    by_id[row["matched_case_id"]]["source_geometry"] ==
                    row["source_geometry"],
                    "exact matched-control pairing")
        families = set(universal["architecture_families"])
        require(len(families) == len(universal["architecture_families"]) >= 2 and
                all(isinstance(value, str) and value for value in families) and
                families == {row["architecture_family"] for row in model_cases},
                "cross-family universal evidence")
        require(commitment["shared_model_bytes"] == 0,
                "expert-local v1 result permits no untraced shared reads")
        total_weights = sum(row["weights"] for row in qwen)
        total_bits = (sum(row["physical_bytes"] for row in qwen) +
                      commitment["shared_model_bytes"]) * 8
        sse = math.fsum(float.fromhex(row["sse_fp64_hex"]) for row in qwen)
        energy = math.fsum(float.fromhex(row["energy_fp64_hex"]) for row in qwen)
        rate = total_bits / total_weights
        relative = sse / energy
        factor = relative * 2.0 ** (2.0 * rate)
        read_max = max(row["read"]["read_amplification"] for row in qwen)
        require(RATE_MIN <= rate <= RATE_MAX, "production target rate interval")
        require(factor <= TARGET_F, "production target F")
        require(read_max < 2.0, "production routed read amplification")
        status = "PASS_LITERAL_QWEN_TARGET_WITH_MATCHED_CONTROLS_AND_UNIVERSALITY"
    else:
        require(not qwen and not controls and
                all(row["kind"] == "synthetic_fixture" for row in results),
                "fixture cannot claim Qwen/control")
        total_weights = sum(row["weights"] for row in results)
        total_bits = sum(row["physical_bytes"] for row in results) * 8
        sse = math.fsum(float.fromhex(row["sse_fp64_hex"]) for row in results)
        energy = math.fsum(float.fromhex(row["energy_fp64_hex"]) for row in results)
        rate = total_bits / total_weights
        relative = sse / energy
        factor = relative * 2.0 ** (2.0 * rate)
        read_max = max(row["read"]["read_amplification"] for row in results)
        status = "PASS_SOURCE_ONLY_PHYSICAL_AUTHORITY_FIXTURE__NO_QWEN_OR_RD_AUTHORITY"
    return {
        "schema": "strata-rm-global-swap-v1-derived-physical-result",
        "commitment_sha256": expected_commitment_sha256, "mode": mode,
        "cases": results, "pooled": {"weights": total_weights,
            "physical_bits": total_bits, "physical_rate_bpw": rate,
            "sse_fp64_hex": sse.hex(), "energy_fp64_hex": energy.hex(),
            "relative_mse": relative, "F": factor,
            "maximum_read_amplification": read_max},
        "literal_packets_opened": len(results),
        "source_metrics_recomputed_from_exact_bf16": True,
        "canonical_packets_compared_as_bytes": True,
        "caller_supplied_metrics_accepted": False,
        "decoder_independent_audit": decoder_audit,
        "status": status,
    }
