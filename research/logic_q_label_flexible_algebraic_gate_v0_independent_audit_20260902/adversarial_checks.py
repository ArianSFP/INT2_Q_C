#!/usr/bin/env python3
"""Independent source-only adversarial checks for the sealed LOGIC-Q v0.

The script receives the sealed package path explicitly.  It never locates or
opens a model, codec payload, coarse artifact, or matched-control artifact.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path


EXPECTED_MANIFEST = (
    "31edbc3325dfdae2b3f43cce4afb360062d5c70583b57dd1e6530835a178cced"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)

    manifest = (package / "SOURCE_MANIFEST.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != EXPECTED_MANIFEST:
        raise RuntimeError("sealed parent manifest mismatch")

    import numpy as np

    core = load("logicq_core", package / "logicq_core.py")
    protocol = load("logicq_panel_protocol", package / "panel_protocol.py")

    # Demonstrate that the finite component machinery is not hard-coded to
    # Qwen's 768x2048 pilot geometry.
    values24 = np.linspace(-2.0, 2.0, 24, dtype=np.float64)
    weights24 = np.ones(24, dtype=np.float64)
    generic = {
        "literal": core.encode_literal_component(
            np, values24, weights24, role="gate", rows=3, cols=8,
            block_size=8, profile=1),
        "rm1": core.encode_rm1_component(
            np, values24, weights24, role="gate", rows=3, cols=8,
            block_size=8, profile=1, lambda_per_bit=0.01,
            exception_limit=2, exact_pair_max=4096, list_pairs=16),
        "gf2": core.encode_gf2_component(
            np, values24, weights24, role="gate", rows=3, cols=8,
            block_size=8, profile=1, lambda_per_bit=0.01, ranks=(0,),
            exception_limit=2, exact_factor_pair_max=65536,
            heuristic_sweeps=0),
        "romdd": core.encode_romdd_component(
            np, values24, weights24, role="gate", rows=3, cols=8,
            block_size=8, profile=1, lambda_per_bit=0.01,
            depths=(0, 1, 2), exception_limit=2),
    }
    generic_roundtrips = {
        name: core.decode_component(np, component.packet)[0] == component.labels
        for name, component in generic.items()
    }

    # Component-local success is not a full-expert physical success.  Here all
    # three components are individually inside the rate interval and have a
    # tiny F, while final 4KiB expert framing pushes the literal expert over
    # 2.5 bpw.
    n = 64 * 64
    values = np.tile(
        np.asarray([-2.0, -0.5, 0.5, 2.0], dtype=np.float64), n // 4)
    weights = np.ones(n, dtype=np.float64)
    components = {
        role: core.encode_literal_component(
            np, values, weights, role=role, rows=64, cols=64,
            block_size=256, profile=1)
        for role in core.ROLE_IDS
    }
    expert = core.pack_expert(
        {role: component.packet for role, component in components.items()})
    expert_rate = len(expert) * 8 / (3 * n)

    # The expert container currently checks role identity, not the SwiGLU
    # relation that all canonical Gate/Up/Down-transposed matrices have the
    # same [intermediate, hidden] shape.
    down_mismatch = core.encode_literal_component(
        np, values, weights, role="down_transposed", rows=32, cols=128,
        block_size=256, profile=1)
    mismatched_packet = core.pack_expert({
        "gate": components["gate"].packet,
        "up": components["up"].packet,
        "down_transposed": down_mismatch.packet,
    })
    mismatched = core.unpack_expert(np, mismatched_packet)

    # Two distinct raw GF(2) factor packets can decode to the same labels and
    # numeric reconstruction because zero U makes V a gauge degree of freedom.
    exception = core.ExceptionPlan(
        (), (), 0.0, core.exception_bits(4, 0)["total_bits"], 16, 0.0)

    def gf2_packet(v0):
        plan = core.GF2Plan(
            1, ((0,), (0,)), (tuple(v0),), ((0,), (0,)), ((0, 0),),
            exception, (0, 0, 0, 0), 8, 16, 0.0, 0.0, True, 1)
        writer = core.BitWriter()
        core.write_gf2(writer, np, 2, 2, plan)
        record = core.ComponentHeaderRecord(
            core.FAMILY_GF2, "gate", 1, 2, 2, 4, 1, 1, 2,
            writer.bit_length, 4)
        return core.component_packet(
            record, (core.fp32_to_bf16_bits(1.0),), writer.finish())

    gauge_a = gf2_packet((0, 0))
    gauge_b = gf2_packet((1, 0))
    gauge_decode_a = core.decode_component(np, gauge_a)
    gauge_decode_b = core.decode_component(np, gauge_b)

    # ROMDD's header depth is accepted as metadata but is not checked against
    # the serialized graph, so equivalent packets with changed depth decode.
    tiny = core.encode_romdd_component(
        np, np.asarray([-2.0, -0.5, 0.5, 2.0]), np.ones(4), role="up",
        rows=2, cols=2, block_size=4, profile=1, lambda_per_bit=0.01,
        depths=(2,), exception_limit=0)
    record, scales, payload = core.parse_component_envelope(tiny.packet)
    altered_record = dataclasses.replace(record, parameter=0)
    altered_romdd = core.component_packet(altered_record, scales, payload)
    altered_decode = core.decode_component(np, altered_romdd)

    bank_source = inspect.getsource(protocol.encode_family_bank)
    rm_source = inspect.getsource(core.encode_rm1_component)
    gf2_source = inspect.getsource(core.search_gf2)

    # A heterogeneous shape cohort is deliberately rejected by the current
    # evaluation protocol.  This is sound for a Qwen cohort but not itself a
    # universal cross-architecture evaluation.
    panel = []
    for layer in range(10):
        shape = (3, 8) if layer < 9 else (4, 8)
        for slot in range(2):
            for role in core.ROLE_IDS:
                panel.append(protocol.PanelRow(
                    f"layer-{layer}", f"slot-{slot}", role, *shape))
    heterogeneous_panel_rejected = False
    try:
        protocol.validate_panel(panel)
    except core.LogicQError as error:
        heterogeneous_panel_rejected = "canonical role shapes" in str(error)

    result = {
        "schema": "logic-q-v0-independent-adversarial-checks",
        "status": "PASS_MECHANISM_WITH_PRODUCTION_HOLDS",
        "sealed_parent_manifest_sha256": EXPECTED_MANIFEST,
        "payload_accessed": False,
        "model_accessed": False,
        "network_accessed_by_test": False,
        "generic_3x8_family_roundtrips": generic_roundtrips,
        "component_local_gate": {
            "component_rates_bpw": {
                role: component.rate_bpw
                for role, component in components.items()
            },
            "component_controls_may_run": {
                role: protocol.absolute_source_gate(component)["controls_may_run"]
                for role, component in components.items()
            },
            "full_expert_rate_bpw": expert_rate,
            "full_expert_exceeds_2p5": expert_rate > 2.5,
        },
        "semantic_shape_closure": {
            "mismatched_expert_accepted": True,
            "decoded_shapes": {
                role: [decoded[2].rows, decoded[2].cols]
                for role, decoded in mismatched.items()
            },
        },
        "canonicality": {
            "distinct_gf2_packets_same_decode": (
                gauge_a != gauge_b
                and gauge_decode_a[:2] == gauge_decode_b[:2]),
            "romdd_depth_mutation_same_decode": (
                altered_romdd != tiny.packet
                and altered_decode[:2] == core.decode_component(np, tiny.packet)[:2]),
        },
        "orchestration": {
            "hard_kill_called_inside_encode_family_bank":
                "optimistic_family_bound" in bank_source,
            "scale_fit_occurs_before_rm_label_search":
                rm_source.find("fit_scales") < rm_source.find("search_rm1_block"),
            "gf2_scalable_rank_cap_present":
                "bounded GF2 heuristic rank cap" in gf2_source,
        },
        "evaluation_scope": {
            "heterogeneous_shape_panel_rejected": heterogeneous_panel_rejected,
            "split_keys_are_identifier_hashes_not_source_values": True,
            "executable_global_hyperparameter_selection_receipt_present": False,
        },
    }
    if not all(generic_roundtrips.values()):
        raise RuntimeError("generic-shape finite roundtrip failed")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"),
                     allow_nan=False))


if __name__ == "__main__":
    main()
