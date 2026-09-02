#!/usr/bin/env python3
"""Independent hostile audit of the frozen LOGIC-Q v2 bound adapter.

This program is deliberately source-free.  It opens only the audited source
packages named below, generates synthetic arrays in memory, and never accepts
or discovers a model, STRATA, coarse-code, or matched-control payload.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np


AUDIT = Path(__file__).resolve().parent
RESEARCH = AUDIT.parent
V2 = RESEARCH / "logic_q_label_flexible_algebraic_gate_v2_bound_adapter"
V1 = RESEARCH / "logic_q_label_flexible_algebraic_gate_v1_capped_adapter"
V0 = RESEARCH / "logic_q_label_flexible_algebraic_gate_v0"
V2_MANIFEST_SHA256 = (
    "e97041b2debdd1a85ce32305f43aae1f76cf4ca937b52e275bdd246ae1b1b980")
V2_ROOT_SHA256 = (
    "080de7a63e596ae34f9da90941d7fd9d07b70dfb2afad97103aa5ab5943d3776")


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(condition: bool, name: str, passed: list[str]) -> None:
    if not condition:
        raise RuntimeError(name)
    passed.append(name)


def rejected(call, fragment: str, name: str, passed: list[str]) -> None:
    try:
        call()
    except Exception as exc:  # hostile boundary intentionally broad
        if fragment not in str(exc):
            raise RuntimeError(f"{name}: unexpected rejection: {exc}") from exc
    else:
        raise RuntimeError(f"{name}: accepted")
    passed.append(name)


def source_values(layer: str, slot: str, role: str, count: int) -> np.ndarray:
    digest = hashlib.sha256(f"{layer}\0{slot}\0{role}".encode()).digest()
    phase = int.from_bytes(digest[:8], "big") / 2.0**64
    index = np.arange(count, dtype=np.float64)
    return (0.613 * np.sin(index * 0.019 + phase) +
            0.271 * np.cos(index * 0.071 - phase) + 0.003 * phase)


def source_blob(layer: str, slot: str, role: str, count: int) -> bytes:
    return np.asarray(source_values(layer, slot, role, count),
                      dtype="<f8").tobytes(order="C")


def make_panel(binder, rows: int, cols: int, *, real_hashes: bool):
    records = []
    count = rows * cols
    for layer_index in range(10):
        layer = f"audit-layer-{layer_index:02d}"
        for slot_index in range(4):
            slot = f"audit-expert-{slot_index:02d}"
            for role in binder.ROLE_ORDER:
                digest = (hashlib.sha256(source_blob(layer, slot, role, count)).hexdigest()
                          if real_hashes else "0" * 64)
                records.append(binder.PanelRow(
                    layer, slot, role, rows, cols, digest, "float64-le"))
    return binder.make_panel_record(records)


def make_literal_expert(v1, core, rows: int, cols: int,
                        layer: str, slot: str):
    components = {}
    blobs = {}
    count = rows * cols
    for role in v1.ROLE_ORDER:
        values = source_values(layer, slot, role, count)
        blobs[role] = np.asarray(values, dtype="<f8").tobytes(order="C")
        encoded = core.encode_literal_component(
            np, values, np.ones(count, dtype=np.float64), role=role,
            rows=rows, cols=cols, block_size=256)
        components[role] = encoded.packet
    return v1.pack_canonical_expert(np, core, components), blobs


def fake_geometry(binder, geometry: dict) -> dict:
    result = copy.deepcopy(geometry)
    result["expert_packet_sha256"] = "1" * 64
    for ordinal, role in enumerate(binder.ROLE_ORDER, start=2):
        result["components"][role]["packet_sha256"] = str(ordinal) * 64
    unsigned = dict(result)
    unsigned.pop("packet_geometry_sha256")
    result["packet_geometry_sha256"] = binder.sha256(
        binder.canonical_json(unsigned))
    return result


def forged_rows(binder, v1, panel: dict, geometry: dict,
                winner_config_id: str) -> dict:
    rows = {}
    for config in v1.FROZEN_CONFIGS:
        scored = []
        for panel_row in panel["rows"]:
            if panel_row["partition"] == "test":
                continue
            component = geometry["components"][panel_row["role"]]
            sse = 0.0 if config.config_id == winner_config_id else 1.0
            base = {
                "schema": "logic-q-v2-independent-scored-row-v1",
                "config_id": config.config_id,
                "layer": panel_row["layer"], "slot": panel_row["slot"],
                "role": panel_row["role"],
                "partition": panel_row["partition"],
                "component_ordinal": panel_row["component_ordinal"],
                "panel_source_sha256": panel_row["source_sha256"],
                "source_dtype": panel_row["source_dtype"],
                "source_blob_bytes": panel_row["rows"] * panel_row["cols"] * 8,
                "expert_packet_sha256": geometry["expert_packet_sha256"],
                "expert_packet_bytes": geometry["expert_packet_bytes"],
                "packet_geometry_sha256": geometry["packet_geometry_sha256"],
                "component_packet_sha256": component["packet_sha256"],
                "component_packet_bytes": component["packet_bytes"],
                "decoded_source_count": component["source_count"],
                "raw_sse_f64_hex": sse.hex(),
                "raw_energy_f64_hex": (1.0).hex(),
                "scorer_schema": "logic-q-v2-independent-source-scorer-v1",
            }
            scored.append(binder.seal_scored_row(base))
        rows[config.config_id] = scored
    return rows


def fake_cupy_module() -> types.ModuleType:
    """Construct a CPU NumPy facade that passes v2's CuPy identity/probe gate."""
    module = types.ModuleType("cupy")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = "cupy"
    module.__spec__ = importlib.machinery.ModuleSpec(
        "cupy", importlib.machinery.SourceFileLoader("cupy", module.__file__))
    module.__version__ = "forged-cpu-facade"

    class FakeNdarray:
        pass

    FakeNdarray.__module__ = "cupy"
    module.ndarray = FakeNdarray
    module.uint64 = np.uint64
    module.arange = np.arange
    module.sum = np.sum
    module.asnumpy = np.asarray

    class Runtime:
        @staticmethod
        def getDevice():
            return 0

        @staticmethod
        def getDeviceCount():
            return 1

        @staticmethod
        def getDeviceProperties(_device):
            return {"name": b"forged CPU facade", "pciBusID": 0,
                    "pciDeviceID": 0, "multiProcessorCount": 999}

        @staticmethod
        def runtimeGetVersion():
            return 999999

        @staticmethod
        def driverGetVersion():
            return 999999

    class Device:
        def __init__(self, _device):
            self.compute_capability = "99"

    class Stream:
        @staticmethod
        def synchronize():
            return None

    module.cuda = types.SimpleNamespace(
        runtime=Runtime(), Device=Device,
        get_current_stream=lambda: Stream())
    return module


def main() -> None:
    binder = load_file("logicq_v2_independent_audit_binder",
                       V2 / "bound_adapter.py")
    scorer = load_file("logicq_v2_independent_audit_scorer",
                       V2 / "independent_scorer.py")
    verifier = load_file("logicq_v2_independent_audit_verifier",
                         V2 / "verify_source.py")
    source_receipt = verifier.verify(V2, V2_MANIFEST_SHA256)
    v1 = binder.load_v1(V1)
    core = binder.load_v0(v1, V0)
    passed: list[str] = []
    findings: list[dict] = []

    check(source_receipt["source_root_sha256"] == V2_ROOT_SHA256,
          "v2_exact_manifest_and_source_root", passed)
    check(source_receipt["strata_six_level_semantics_bound"] is False,
          "v2_explicitly_not_strata_bound", passed)

    rows, cols = 256, 256
    panel = make_panel(binder, rows, cols, real_hashes=True)
    panel_sha = panel["panel_sha256"]
    target = next(row for row in panel["rows"]
                  if row["partition"] == "train" and row["role"] == "gate")
    packet, blobs = make_literal_expert(
        v1, core, rows, cols, target["layer"], target["slot"])
    geometry = binder.packet_geometry(np, v1, core, packet)
    check(geometry["physical_bits"] == len(packet) * 8 and
          geometry["expert_weights_from_headers"] == 3 * rows * cols and
          geometry["physical_rate_bpw"] == len(packet) * 8 / (3 * rows * cols),
          "literal_packet_counts_and_rate_from_actual_bytes", passed)
    check(len(packet) % core.EXPERT_PAGE == 0 and
          geometry["routed_storage_read_bytes"] == len(packet),
          "literal_packet_is_one_page_contiguous_object", passed)

    score = scorer.score_expert_packet(
        binder, v1, core, packet=packet, source_blobs=blobs, panel=panel,
        expected_panel_sha256=panel_sha, layer=target["layer"],
        slot=target["slot"], config_id=v1.FROZEN_CONFIGS[0].config_id)
    decoded = v1.unpack_canonical_expert(np, core, packet)
    manual_sse = 0.0
    manual_energy = 0.0
    for role in binder.ROLE_ORDER:
        source = np.frombuffer(blobs[role], dtype="<f8")
        reconstruction = np.asarray(decoded[role][1], dtype=np.float64)
        manual_sse += float(np.sum((source - reconstruction) ** 2,
                                   dtype=np.float64))
        manual_energy += float(np.sum(source * source, dtype=np.float64))
    check(float.fromhex(score["pooled"]["raw_sse_f64_hex"]) == manual_sse and
          float.fromhex(score["pooled"]["raw_energy_f64_hex"]) == manual_energy,
          "independent_scorer_matches_manual_raw_fp64", passed)

    # The byte parser has explicit dtype and nonfinite checks.
    bf16 = np.asarray([0x3F80, 0xC000, 0x0000], dtype="<u2").tobytes()
    check(np.array_equal(scorer._decode_source(np, bf16, "bf16-le", 3),
                         np.asarray([1.0, -2.0, 0.0])),
          "bf16_little_endian_parser", passed)
    rejected(lambda: scorer._decode_source(
        np, np.asarray([math.nan], dtype="<f8").tobytes(), "float64-le", 1),
        "finite decoded source", "nonfinite_source_rejected", passed)

    # Canonical parser rejects illegal scale words and nonzero physical padding.
    nan_scale = bytearray(packet)
    scale_offset = core.EXPERT_HEADER_BYTES + core.COMPONENT_HEADER_BYTES
    nan_scale[scale_offset:scale_offset + 2] = b"\x7f\xc1"
    rejected(lambda: binder.packet_geometry(np, v1, core, bytes(nan_scale)),
             "positive scale", "nan_scale_rejected_from_actual_packet", passed)
    nonzero_page = bytearray(packet)
    nonzero_page[-1] = 1
    rejected(lambda: binder.packet_geometry(np, v1, core, bytes(nonzero_page)),
             "zero", "nonzero_page_padding_rejected", passed)

    # There is no embedded CRC.  A label-bit mutation remains a legal canonical
    # packet and is distinguished only by an externally authenticated hash.
    first_record = core.parse_component_envelope(
        v1._expert_component_slices(core, packet)["gate"])[0]
    payload_offset = (core.EXPERT_HEADER_BYTES + core.COMPONENT_HEADER_BYTES +
                      first_record.scale_bytes)
    changed_packet = bytearray(packet)
    changed_packet[payload_offset] ^= 1
    changed_geometry = binder.packet_geometry(np, v1, core, bytes(changed_packet))
    check(changed_geometry["expert_packet_sha256"] !=
          geometry["expert_packet_sha256"],
          "valid_label_mutation_has_only_changed_external_hash", passed)
    findings.append({
        "id": "NO_EMBEDDED_PACKET_CRC",
        "severity": "HOLD_PRODUCTION_INTEGRITY",
        "evidence": "A literal label-bit mutation is still canonical and decodes; only the external SHA-256 changes.",
    })

    # Receipt validation never sees the packet or scale/payload bytes.  Entirely
    # fictitious packet/component hashes validate against copied literal headers.
    invented_geometry = fake_geometry(binder, geometry)
    binder.validate_packet_geometry_receipt(core, invented_geometry)
    check(invented_geometry["expert_packet_sha256"] == "1" * 64,
          "header_only_geometry_accepts_invented_packet_hashes", passed)
    findings.append({
        "id": "PACKET_RECEIPT_OMITS_LITERAL_BYTES",
        "severity": "BLOCKS_SELECTION_PROVENANCE",
        "evidence": "Header-only receipt accepted arbitrary expert/component hashes and contains neither payload nor scales.",
    })

    # Build a panel whose every component hash is the same nonexistent digest.
    # Self-sealed rows assert zero SSE and reuse one invented packet under every
    # config.  No source bytes, scorer bundle, packet bytes, or control bytes are
    # supplied.  Selection and authorization nevertheless accept the target.
    forged_panel = make_panel(binder, rows, cols, real_hashes=False)
    forged_panel_sha = forged_panel["panel_sha256"]
    target_config = v1.FROZEN_CONFIGS[-1].config_id
    rows_by_config = forged_rows(
        binder, v1, forged_panel, invented_geometry, target_config)
    receipt = binder.make_selection_receipt(
        v1, core, forged_panel, expected_panel_sha256=forged_panel_sha,
        rows_by_config=rows_by_config,
        packet_receipts_by_sha256={"1" * 64: invented_geometry})
    selected = binder.authorize_test(
        v1, core, forged_panel, receipt,
        expected_panel_sha256=forged_panel_sha,
        expected_receipt_sha256=receipt["receipt_sha256"])
    check(selected.config_id == target_config and
          receipt["derived_metrics"][target_config]["validation"]["F"] == 0.0,
          "fully_forged_rows_packet_and_source_hashes_authorize_target_config",
          passed)
    findings.append({
        "id": "SELF_SEALED_METRICS_ARE_FORGEABLE",
        "severity": "BLOCKS_QWEN_AND_CONTROL_AUTHORITY",
        "evidence": "authorize_test accepted arbitrary self-sealed FP64 metrics, duplicate nonexistent source hashes, an invented packet receipt, and one packet aliased to all configs when the attacker supplied the matching external pin.",
    })

    # The real scorer also does not prove that a packet was produced by the
    # claimed config: identical bytes score successfully under another config.
    alias_score = scorer.score_expert_packet(
        binder, v1, core, packet=packet, source_blobs=blobs, panel=panel,
        expected_panel_sha256=panel_sha, layer=target["layer"],
        slot=target["slot"], config_id=v1.FROZEN_CONFIGS[-1].config_id)
    check(alias_score["expert_packet_sha256"] == score["expert_packet_sha256"],
          "same_packet_scores_under_distinct_config_id", passed)
    findings.append({
        "id": "CONFIG_NOT_BOUND_TO_PACKET",
        "severity": "BLOCKS_MODEL_SELECTION_AUTHORITY",
        "evidence": "The same literal packet is accepted and scored under distinct frozen config IDs; config_id is not serialized or derivable from the packet.",
    })

    # Whole-layer/whole-slot partition mechanics are correctly reconstructed,
    # but duplicate source hashes across owners are accepted.
    for layer in sorted({row["layer"] for row in forged_panel["rows"]}):
        partitions = {row["partition"] for row in forged_panel["rows"]
                      if row["layer"] == layer}
        check(partitions == ({"test"} if layer in forged_panel["test_layers"]
                             else {"train", "validation"}),
              f"whole_layer_test_partition_{layer}", passed)
    check(len({row["source_sha256"] for row in forged_panel["rows"]}) == 1,
          "duplicate_cross_owner_source_hashes_are_accepted", passed)
    findings.append({
        "id": "OWNER_CONTENT_ALIAS_NOT_CLOSED",
        "severity": "HOLD_PORTABILITY_SPLIT",
        "evidence": "Identifier partitions are whole-layer/slot, but identical source hashes may appear in train, validation, and test without rejection.",
    })

    # A complete CPU facade inserted as sys.modules['cupy'] defeats the claimed
    # canonical-import identity and synchronized arithmetic probe.
    fake = fake_cupy_module()
    context = binder.make_launch_context(
        panel_sha256=panel_sha,
        selection_receipt_sha256=receipt["receipt_sha256"],
        config_id=v1.FROZEN_CONFIGS[0].config_id,
        layer=target["layer"], slot=target["slot"], rows=rows, cols=cols)
    prior = sys.modules.get("cupy")
    sys.modules["cupy"] = fake
    try:
        fake_launch = binder.collect_cupy_launch_receipt(
            fake, launch_context=context)
        binder.validate_cupy_launch_receipt(
            fake, fake_launch,
            expected_receipt_sha256=fake_launch["receipt_sha256"],
            expected_launch_context=context)
    finally:
        if prior is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = prior
    check(fake_launch["module_version"] == "forged-cpu-facade" and
          fake_launch["device_name"] == "forged CPU facade" and
          fake_launch["probe_observed"] == fake_launch["probe_expected"],
          "cpu_facade_spoofs_canonical_cupy_device_and_probe", passed)
    findings.append({
        "id": "CUPY_GATE_SPOOFABLE_IN_PROCESS",
        "severity": "BLOCKS_ADVERSARIAL_BACKEND_ATTESTATION",
        "evidence": "A ModuleType injected into sys.modules with NumPy arithmetic passed collect_cupy_launch_receipt and validate_cupy_launch_receipt as a forged CUDA device.",
    })
    check(context["selection_receipt_sha256"] == receipt["receipt_sha256"] and
          context["config_id"] != receipt["selected_config_id"],
          "launch_context_accepts_nonselected_config_for_selection_hash", passed)
    findings.append({
        "id": "LAUNCH_NOT_SEMANTICALLY_BOUND_TO_SELECTION",
        "severity": "BLOCKS_SELECTED_CODEC_EXECUTION_AUTHORITY",
        "evidence": "make_launch_context accepted a config different from the config selected by the referenced receipt; encode_expert_bound receives neither the receipt nor authorize_test output provenance.",
    })

    check(not any("control" in name and callable(getattr(binder, name))
                  for name in dir(binder)),
          "v2_has_no_executable_control_result_gate", passed)
    findings.append({
        "id": "NO_BOUND_CONTROL_OR_TEST_RESULT_GATE",
        "severity": "BLOCKS_FINAL_CLAIM",
        "evidence": "V2 binds train/validation selection only; it has no authenticated matched-control/test aggregation path or final absolute F gate.",
    })

    output = {
        "schema": "logic-q-v2-independent-source-hostile-audit-v1",
        "status": "MECHANISM_VALID__HOLD_PRODUCTION_PROVENANCE_BACKEND_AND_STRATA",
        "source_manifest_sha256": V2_MANIFEST_SHA256,
        "source_root_sha256": V2_ROOT_SHA256,
        "passed_checks": passed,
        "passed_check_count": len(passed),
        "findings": findings,
        "literal_fixture": {
            "weights": geometry["expert_weights_from_headers"],
            "bytes": geometry["expert_packet_bytes"],
            "bpw": geometry["physical_rate_bpw"],
            "layout_addressable_read_amplification": 1.0,
            "runtime_read_amplification_measured": False,
        },
        "forgery_receipt": {
            "panel_sha256": forged_panel_sha,
            "selection_receipt_sha256": receipt["receipt_sha256"],
            "selected_config_id": selected.config_id,
            "accepted_validation_F": receipt["derived_metrics"][
                selected.config_id]["validation"]["F"],
            "raw_source_bytes_supplied": False,
            "packet_bytes_supplied_to_authorizer": False,
            "control_bytes_supplied": False,
        },
        "fake_cupy_receipt": fake_launch,
        "model_qwen_strata_control_payload_accessed": False,
        "network_accessed_by_script": False,
        "real_cupy_imported_by_script": prior is not None,
        "claim_boundary": (
            "The literal four-level codec/scorer mechanics are valid on source-free "
            "bytes. This audit grants no Qwen, current STRATA, F<=0.8, control, "
            "runtime-bandwidth, or universal SwiGLU-MoE authority."),
    }
    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
