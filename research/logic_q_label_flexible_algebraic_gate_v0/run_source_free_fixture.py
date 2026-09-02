#!/usr/bin/env python3
"""Deterministic synthetic LOGIC-Q mechanism fixture; no payload authority."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    if specification is None or specification.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def main() -> None:
    import numpy as np

    core = load("logicq_core", "logicq_core.py")
    protocol = load("logicq_panel_protocol", "panel_protocol.py")
    rows, cols, block_size = 2, 2, 4
    affine_labels = np.asarray(
        [0, 1,
         3, 2], dtype=np.uint8)
    ratios = np.asarray(core.PROFILE_RATIOS[1], dtype=np.float64)
    perturbation = np.asarray(
        [(-1.0 if index % 3 == 0 else 1.0) * 0.015
         for index in range(rows * cols)], dtype=np.float64)
    source = ratios[affine_labels] + perturbation
    weights = np.asarray([1.0 + (index % 5) / 7.0
                          for index in range(rows * cols)], dtype=np.float64)
    lambda_per_bit = 0.004

    source_components = {}
    source_banks = {}
    for role_index, role in enumerate(("gate", "up", "down_transposed")):
        role_source = source + role_index * 0.002
        bank = protocol.encode_family_bank(
            np, role_source, weights, role=role, rows=rows, cols=cols,
            block_size=block_size, lambda_per_bit=lambda_per_bit,
            rm_exception_limit=2, gf2_ranks=(0, 1),
            gf2_exception_limit=2, romdd_depths=(0, 1, 2),
            romdd_exception_limit=2, rm_exact_pair_max=4096,
            rm_list_pairs=64, gf2_exact_pair_max=65536,
            gf2_heuristic_sweeps=2)
        winner = protocol.choose_paid_mode(bank, lambda_per_bit)
        source_banks[role] = bank
        source_components[role] = winner.packet

    expert_packet = core.pack_expert(source_components)
    decoded = core.unpack_expert(np, expert_packet)
    for role in source_components:
        assert decoded[role][0] == protocol.choose_paid_mode(
            source_banks[role], lambda_per_bit).labels

    # This is an implementation fixture only. Production controls remain
    # forbidden because a tiny 4096-byte-padded expert cannot clear the target.
    control_source, control_receipt = protocol.moment_matched_gaussian(
        np, source, block_size=block_size, seed=core.CONTROL_SEEDS[0],
        component_ordinal=0)
    control_bank = protocol.encode_family_bank(
        np, control_source, weights, role="gate", rows=rows, cols=cols,
        block_size=block_size, lambda_per_bit=lambda_per_bit,
        rm_exception_limit=2, gf2_ranks=(0, 1), gf2_exception_limit=2,
        romdd_depths=(0, 1, 2), romdd_exception_limit=2,
        rm_exact_pair_max=4096, rm_list_pairs=64,
        gf2_exact_pair_max=65536, gf2_heuristic_sweeps=2)

    report = {
        "schema": "logic-q-label-flexible-algebraic-gate-v0-source-free-fixture",
        "status": "PASS_SOURCE_FREE_MECHANISM_FIXTURE",
        "model_or_checkpoint_payload_accessed": False,
        "cuda_or_cupy_initialized": False,
        "network_accessed": False,
        "positive_claim_authority": False,
        "production_controls_authorized": False,
        "budget_identities": core.exact_budget_identities(),
        "rank680_accounting": core.rank680_accounting(),
        "coordinate_domain": core.qwen_coordinate_domain_record(),
        "rm_descriptor_controls": core.rm_descriptor_controls(),
        "synthetic_source": {
            role: {
                "winner": protocol.choose_paid_mode(bank, lambda_per_bit).family,
                "families": {
                    name: {
                        "physical_bits": component.physical_bits,
                        "weighted_sse": component.weighted_sse,
                        "exact_search": component.exact_search,
                        "decoded": True,
                    }
                    for name, component in sorted(bank.items())
                },
            }
            for role, bank in source_banks.items()
        },
        "expert_packet": {
            "bytes": len(expert_packet),
            "sha256": __import__("hashlib").sha256(expert_packet).hexdigest(),
            "ledger": core.expert_ledger(source_components),
            "three_roles_independently_decoded": True,
        },
        "synthetic_matched_control": {
            "receipt": control_receipt,
            "families_rerun_from_continuous_values": sorted(control_bank),
            "prebuilt_labels_accepted": False,
            "production_evidence": False,
        },
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":"),
                     allow_nan=False))


if __name__ == "__main__":
    main()
