#!/usr/bin/env python3
"""Hostile source-only tests for the LOGIC-Q v3 authority boundary."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np


PACKAGE = Path(__file__).resolve().parent
RESEARCH = PACKAGE.parent
V2 = RESEARCH / "logic_q_label_flexible_algebraic_gate_v2_bound_adapter"
V1 = RESEARCH / "logic_q_label_flexible_algebraic_gate_v1_capped_adapter"
V0 = RESEARCH / "logic_q_label_flexible_algebraic_gate_v0"


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(condition: bool, name: str, tests: list[str]) -> None:
    if not condition:
        raise RuntimeError(name)
    tests.append(name)


def rejected(call, fragment: str, name: str, tests: list[str]) -> None:
    try:
        call()
    except Exception as exc:  # hostile boundary intentionally broad
        if fragment not in str(exc):
            raise RuntimeError(f"{name}: wrong rejection: {exc}") from exc
    else:
        raise RuntimeError(f"{name}: accepted")
    tests.append(name)


def source_values(layer: str, slot: str, role: str, count: int) -> np.ndarray:
    digest = hashlib.sha256(f"{layer}:{slot}:{role}".encode()).digest()
    phase = int.from_bytes(digest[:4], "big") / 2.0**32
    index = np.arange(count, dtype=np.float64)
    return (0.67 * np.sin(index * 0.031 + phase) +
            0.23 * np.cos(index * 0.097 - phase) + 0.002 * phase)


def placeholder_hash(layer: str, slot: str, role: str) -> str:
    return hashlib.sha256(f"placeholder:{layer}:{slot}:{role}".encode()).hexdigest()


def panel_rows(binder, rows: int, cols: int, replacements=None):
    replacements = replacements or {}
    result = []
    for layer_index in range(10):
        layer = f"v3-layer-{layer_index:02d}"
        for slot_index in range(4):
            slot = f"v3-expert-{slot_index:02d}"
            for role in binder.ROLE_ORDER:
                digest = replacements.get(
                    (layer, slot, role), placeholder_hash(layer, slot, role))
                result.append(binder.PanelRow(
                    layer, slot, role, rows, cols, digest, "float64-le"))
    return result


def backend_policy(authority):
    return {
        "schema": "logic-q-v3-fresh-cupy-policy-v1",
        "cupy_version": "14.2.0",
        "module_file_sha256":
            "8c4724758587dea5f1c1d7c217c74a9fa0e4ed7f9d76a2b86fa001117cf3c718",
        "device_name": "NVIDIA GeForce RTX 5090",
        "compute_capability": "120",
        "runtime_version": 12090,
        "driver_version": 13000,
        "probe_elements": authority.PROBE_ELEMENTS,
        "probe_expected": authority.PROBE_EXPECTED,
    }


def make_literal_expert(v1, core, layer: str, slot: str,
                        rows: int, cols: int):
    count = rows * cols
    components = {}
    blobs = {}
    for role in v1.ROLE_ORDER:
        values = source_values(layer, slot, role, count)
        blobs[role] = np.asarray(values, dtype="<f8").tobytes(order="C")
        encoded = core.encode_literal_component(
            np, values, np.ones(count, dtype=np.float64), role=role,
            rows=rows, cols=cols, block_size=256)
        components[role] = encoded.packet
    return v1.pack_canonical_expert(np, core, components), blobs


def main() -> None:
    authority = load_file("logicq_v3_authority_test", PACKAGE / "authority.py")
    binder, v1, core = authority.load_dependencies(V2, V1, V0)
    tests: list[str] = []
    rows, cols = 256, 256

    preliminary = binder.make_panel_record(panel_rows(binder, rows, cols))
    target = next(row for row in preliminary["rows"]
                  if row["partition"] == "train" and row["role"] == "gate")
    inner, blobs = make_literal_expert(
        v1, core, target["layer"], target["slot"], rows, cols)
    replacements = {(target["layer"], target["slot"], role):
                    hashlib.sha256(blobs[role]).hexdigest()
                    for role in authority.ROLE_ORDER}
    panel = binder.make_panel_record(
        panel_rows(binder, rows, cols, replacements))
    panel_sha = panel["panel_sha256"]
    target = next(row for row in panel["rows"]
                  if row["partition"] == "train" and row["role"] == "gate")
    aliases = authority.make_alias_map(panel)
    worker_sha = hashlib.sha256(
        (PACKAGE / "gpu_worker.py").read_bytes()).hexdigest()
    precommit = authority.make_precommit(
        binder, v1, panel, expected_panel_sha256=panel_sha,
        alias_map=aliases, backend_policy=backend_policy(authority),
        worker_sha256=worker_sha)
    precommit_sha = precommit["precommit_sha256"]
    authority.validate_precommit(
        binder, v1, precommit, expected_precommit_sha256=precommit_sha)
    check(precommit["panel_sha256"] == panel_sha and
          precommit["scored_rows_or_encoder_metrics_accepted"] is False and
          precommit["test_opened"] is False,
          "external_precommit_binds_panel_grid_aliases_backend_before_rows", tests)

    source_hashes = {role: hashlib.sha256(blobs[role]).hexdigest()
                     for role in authority.ROLE_ORDER}
    outer = authority.pack_authority_packet(
        np, binder, v1, core, inner_packet=inner, precommit=precommit,
        expected_precommit_sha256=precommit_sha,
        config_id=v1.FROZEN_CONFIGS[0].config_id,
        layer=target["layer"], slot=target["slot"],
        source_sha256_by_role=source_hashes,
        backend_receipt_sha256="SOURCE_FREE_FIXTURE",
        mode="source_free_fixture")
    parsed = authority.unpack_authority_packet(
        np, binder, v1, core, outer, precommit=precommit,
        expected_precommit_sha256=precommit_sha,
        expected_mode="source_free_fixture")
    check(parsed["inner_packet"] == inner and
          parsed["physical_bits"] == len(outer) * 8 and
          parsed["physical_rate_bpw"] ==
          len(outer) * 8 / (3 * rows * cols) and
          parsed["layout_addressable_read_amplification"] == 1.0 and
          parsed["runtime_read_amplification_measured"] is False,
          "full_outer_bytes_crc_hash_inner_geometry_and_rate", tests)

    score = authority.score_authority_packet(
        binder, v1, core, packet=outer, source_blobs=blobs,
        precommit=precommit, expected_precommit_sha256=precommit_sha,
        expected_mode="source_free_fixture")
    decoded = v1.unpack_canonical_expert(np, core, inner)
    expected_sse = 0.0
    expected_energy = 0.0
    for role in authority.ROLE_ORDER:
        source = np.frombuffer(blobs[role], dtype="<f8")
        reconstruction = np.asarray(decoded[role][1], dtype=np.float64)
        expected_sse += float(np.sum((source - reconstruction) ** 2,
                                     dtype=np.float64))
        expected_energy += float(np.sum(source * source, dtype=np.float64))
    check(float.fromhex(score["pooled"]["raw_sse_f64_hex"]) == expected_sse and
          float.fromhex(score["pooled"]["raw_energy_f64_hex"]) == expected_energy
          and score["metrics_accepted_from_encoder"] is False,
          "auditor_owned_raw_bytes_reconstruction_and_fp64_score", tests)

    # Outer corruption is rejected before any metric can be consumed.
    crc_attack = bytearray(outer)
    _, _, header_length, _ = authority.AUTH_PREFIX.unpack(
        outer[:authority.AUTH_PREFIX.size])
    inner_start = authority.AUTH_PREFIX.size + header_length
    crc_attack[inner_start + 7] ^= 1
    rejected(lambda: authority.unpack_authority_packet(
        np, binder, v1, core, bytes(crc_attack), precommit=precommit,
        expected_precommit_sha256=precommit_sha,
        expected_mode="source_free_fixture"),
        "CRC32", "outer_payload_mutation_rejected_by_crc", tests)

    # Recomputing the outer CRC is insufficient because the header binds the
    # complete inner SHA-256 and CRC as well.
    hash_attack = bytearray(outer)
    hash_attack[inner_start + 11] ^= 1
    body_end = inner_start + len(inner)
    hash_attack[body_end:body_end + authority.AUTH_TRAILER.size] = (
        authority.AUTH_TRAILER.pack(zlib.crc32(hash_attack[:body_end]) & 0xFFFFFFFF))
    rejected(lambda: authority.unpack_authority_packet(
        np, binder, v1, core, bytes(hash_attack), precommit=precommit,
        expected_precommit_sha256=precommit_sha,
        expected_mode="source_free_fixture"),
        "inner payload hash", "crc_reseal_cannot_override_inner_hash", tests)

    page_attack = bytearray(outer)
    page_attack[-1] = 1
    rejected(lambda: authority.unpack_authority_packet(
        np, binder, v1, core, bytes(page_attack), precommit=precommit,
        expected_precommit_sha256=precommit_sha,
        expected_mode="source_free_fixture"),
        "zero page padding", "nonzero_outer_page_padding_rejected", tests)

    # Illegal scales are rejected by parsing literal inner bytes, before wrap.
    nan_inner = bytearray(inner)
    scale_offset = core.EXPERT_HEADER_BYTES + core.COMPONENT_HEADER_BYTES
    nan_inner[scale_offset:scale_offset + 2] = b"\x7f\xc1"
    rejected(lambda: authority.pack_authority_packet(
        np, binder, v1, core, inner_packet=bytes(nan_inner),
        precommit=precommit, expected_precommit_sha256=precommit_sha,
        config_id=v1.FROZEN_CONFIGS[0].config_id,
        layer=target["layer"], slot=target["slot"],
        source_sha256_by_role=source_hashes,
        backend_receipt_sha256="SOURCE_FREE_FIXTURE",
        mode="source_free_fixture"),
        "positive scale", "nan_inner_scale_rejected_from_literal_payload", tests)

    tampered_sources = dict(blobs)
    tampered_sources["gate"] = bytes([blobs["gate"][0] ^ 1]) + blobs["gate"][1:]
    rejected(lambda: authority.score_authority_packet(
        binder, v1, core, packet=outer, source_blobs=tampered_sources,
        precommit=precommit, expected_precommit_sha256=precommit_sha,
        expected_mode="source_free_fixture"),
        "auditor-owned source hash", "raw_source_tamper_rejected", tests)

    # Full public precommit reseal cannot replace the separately supplied pin.
    precommit_attack = copy.deepcopy(precommit)
    precommit_attack["backend_policy"]["driver_version"] += 1
    unsigned = dict(precommit_attack)
    unsigned.pop("precommit_sha256")
    precommit_attack["precommit_sha256"] = authority.sha256(
        authority.canonical_json(unsigned))
    rejected(lambda: authority.validate_precommit(
        binder, v1, precommit_attack,
        expected_precommit_sha256=precommit_sha),
        "external pin", "public_precommit_reseal_rejected", tests)

    # Duplicate content is rejected unless one exact externally committed alias
    # group accounts for every occurrence of that digest.
    duplicate_rows = panel_rows(binder, rows, cols)
    first = duplicate_rows[0]
    second = duplicate_rows[1]
    duplicate_rows[1] = binder.PanelRow(
        second.layer, second.slot, second.role, second.rows, second.cols,
        first.source_sha256, second.source_dtype)
    duplicate_panel = binder.make_panel_record(duplicate_rows)
    rejected(lambda: authority.make_alias_map(duplicate_panel),
             "complete explicit duplicate alias map",
             "duplicate_source_alias_rejected_by_default", tests)
    duplicate_ordinals = [row["component_ordinal"]
                          for row in duplicate_panel["rows"]
                          if row["source_sha256"] == first.source_sha256]
    explicit_alias = authority.make_alias_map(duplicate_panel, [{
        "component_ordinals": duplicate_ordinals,
        "reason": "source-free deliberate alias fixture",
    }])
    check(explicit_alias["groups"][0]["component_ordinals"] ==
          sorted(duplicate_ordinals),
          "explicit_complete_audited_alias_map_accepted", tests)

    # Production APIs expose no row, metric, packet receipt, backend object, or
    # caller-selected test config injection point.
    selection_parameters = inspect.signature(
        authority.run_selection_authority).parameters
    launch_parameters = inspect.signature(authority.run_selected_expert).parameters
    check(not ({"rows_by_config", "metrics", "packet_receipts", "xp", "backend"}
               & set(selection_parameters)) and
          "config" not in launch_parameters and
          "config_id" not in launch_parameters,
          "production_api_has_no_metrics_packet_backend_or_config_injection", tests)
    authorize_source = inspect.getsource(authority.authorize_selection)
    check("production authorization requires fresh worker replay" in
          authorize_source and "replay_packet == parsed[\"inner_packet\"]" in
          authorize_source,
          "authorization_requires_exact_fresh_worker_replay", tests)
    command = authority.fresh_worker_command(
        PACKAGE / "gpu_worker.py", Path("request"), Path("packet"),
        Path("receipt"))
    environment = authority._safe_worker_environment()
    check(command[1:3] == ["-I", "-B"] and
          "PYTHONPATH" not in environment and "PYTHONHOME" not in environment and
          environment["PYTHONNOUSERSITE"] == "1",
          "fresh_worker_isolated_command_and_sanitized_environment", tests)

    worker_source = (PACKAGE / "gpu_worker.py").read_text(encoding="utf-8")
    check('require("cupy" not in sys.modules' in worker_source and
          'cp = importlib.import_module("cupy")' in worker_source and
          worker_source.index('require("cupy" not in sys.modules') <
          worker_source.index('cp = importlib.import_module("cupy")') <
          worker_source.index("authority = load_authority"),
          "worker_imports_cupy_fresh_before_repository_modules", tests)

    result = {
        "schema": "logic-q-v3-authority-source-only-hostile-tests-v1",
        "status": "PASS_SOURCE_ONLY_AUTHORITY_MECHANICS__NO_PAYLOAD_AUTHORITY",
        "tests": tests, "test_count": len(tests),
        "precommit_sha256": precommit_sha,
        "inner_packet_sha256": hashlib.sha256(inner).hexdigest(),
        "authority_packet_sha256": hashlib.sha256(outer).hexdigest(),
        "fixture_weights": 3 * rows * cols,
        "fixture_physical_bytes": len(outer),
        "fixture_physical_rate_bpw": len(outer) * 8 / (3 * rows * cols),
        "model_qwen_strata_control_payload_accessed": False,
        "cupy_imported_or_initialized": False,
        "network_accessed": False,
        "strata_semantics_bound": False,
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
