#!/usr/bin/env python3
"""Run the no-payload STRATA-RM6 mechanism and ledger gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from exact_oracle import exact_joint_oracle, unconstrained_64way_bound
from packet_codec import decode_packet, encode_packet
from rm6_core import (LOCAL_N, ORDER_BANK, dimension_ledger,
                      exact_distortion_costs, reconstruction_levels)
from strata_rm_sc import (classify_selected_dimension, replay_six_greedy)
from strata_semantics import authenticate_auditor, verify_rm_orientation


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _local_packet_control(coset_mode: str) -> dict[str, object]:
    bank_id, profile_q, sc_seed = 0, 96, 0x13579BDF
    greedy = replay_six_greedy(bank_id, profile_q, sc_seed, coset_mode)
    packet, encoded = encode_packet(
        greedy["decisions"], bank_id=bank_id, scale_fp16_bits=0x3C00,
        profile_q=profile_q, coset_mode=coset_mode, sc_seed=sc_seed,
        rht_seed=0x0123456789ABCDEF)
    decoded = decode_packet(packet)
    if not np.array_equal(encoded["indices"], decoded["indices"]):
        raise RuntimeError("local packet canonical index mismatch")
    return {"coset_mode": coset_mode, "logical_bits": decoded["logical_bits"],
            "packet_bytes": len(packet), "physical_bpw": len(packet) * 8.0 / LOCAL_N,
            "selected_information_bits": decoded["information_bits"],
            "emitted_arithmetic_bits": decoded["ledger"]["emitted_arithmetic_bits"],
            "target_rate_eligible": decoded["ledger"]["actual_target_rate_eligible"],
            "promotion_status": decoded["ledger"]["promotion_status"],
            "packet_sha256": hashlib.sha256(packet).hexdigest(),
            "canonical_reencode_match": decoded["canonical_reencode_match"]}


def _tiny_oracle() -> dict[str, object]:
    variables, orders, scale_bits = 3, (1, 1, 0, 0, 0, 0), 0x3C00
    n = 1 << variables
    coordinate = np.arange(n)
    levels = reconstruction_levels(scale_bits)
    target_indices = ((11 * coordinate + 7 * (coordinate >> 1)) & 63).astype(np.uint8)
    target = levels[target_indices] + 0.013 * np.cos(coordinate)
    costs = exact_distortion_costs(target, scale_bits)
    exact = exact_joint_oracle(costs, variables, orders, sc_seed=17,
                               coset_mode="zero")
    favorable = unconstrained_64way_bound(costs)
    return {"variables": variables, "orders": list(orders),
            "information_bits": exact["information_bits"],
            "candidate_messages": exact["candidate_messages"],
            "exact_legal_distortion": exact["distortion"],
            "unconstrained_64way_distortion": favorable["distortion"],
            "legal_ge_unconstrained": exact["distortion"] + 1e-12 >=
            favorable["distortion"], "scope": "bounded mechanism oracle only"}


def make_receipt(auditor: Path) -> dict[str, object]:
    semantics = authenticate_auditor(auditor)
    orientation = verify_rm_orientation()
    bank_rows = [dimension_ledger(bank_id) for bank_id in sorted(ORDER_BANK)]
    zero = _local_packet_control("zero")
    random_coset = _local_packet_control("current_random")
    receipt: dict[str, object] = {
        "schema": "strata-rm6-label-flexible-source-gate-v0",
        "status": "GO_LOCAL_MECHANISM_HOLD_PRODUCTION_AND_PAYLOAD",
        "authenticated_strata": semantics,
        "rm_orientation": orientation,
        "local_direct_rm4096": {
            "status": "GO_SOURCE_FREE_MECHANISM_ONLY",
            "block_values": 4096, "exact_rm": True,
            "bank_ledger": bank_rows,
            "uniform_rm5_information_bits": 9516,
            "uniform_rm5_raw_dimension_bpw_before_metadata": 9516 / 4096.0,
            "all_dimension_ledgers_at_most_2_5_bpw": all(
                row["passes_2_5_bpw_dimension_ledger"] for row in bank_rows),
            "dimension_screen_target_eligible_bank_ids": [
                row["bank_id"] for row in bank_rows
                if row["dimension_screen_target_rate_eligible"]],
            "target_rate_contract": {
                "minimum_bpw": 2.15, "maximum_bpw": 2.5,
                "actual_literal_packet_controls_promotion": True,
                "subminimum_packets_are_mechanism_fixtures_only": True,
                "literal_padding_or_refinement_implemented": False,
            },
            "zero_coset_packet_control": zero,
            "current_random_coset_packet_control": random_coset,
            "random_coset_warning": (
                "current random frozen values form a public affine coset; they do not "
                "provide low-degree coordinate-function label evidence"
            ),
            "arithmetic_length_is_data_dependent_and_fail_closed": True,
            "not_drop_in_current_global_block": True,
        },
        "global_strata_rm_ordered_swap": {
            "status": "HOLD_NO_PAYLOAD_ARITHMETIC_LENGTH",
            "current_block_log2": [20, 21],
            "rule": "keep each current K; rank phases by descending popcount then index",
            "classification_example": classify_selected_dimension(21, 700000),
            "exact_rm_only_when_k_matches_full_rm_dimension": True,
            "otherwise_name": "RM-ordered truncated polar set",
            "raw_4096_dimension_ledger_applies": False,
            "required_physical_measurement": "canonical current arithmetic length",
        },
        "bounded_joint_oracle": _tiny_oracle(),
        "verdicts": {
            "mechanism": "GO_LOCAL_RM_FROZEN_SET_AND_LITERAL_PACKET",
            "dominant_oracle": "GO_SMALL_N_EXACT_ONLY",
            "production_local_search": "HOLD_NO_FULL_JOINT_RM5_12_ENCODER",
            "global_swap": "HOLD_NO_CURRENT_K_ARITHMETIC_PAYLOAD_MEASUREMENT",
            "local_target_packet":
                "HOLD_CONTROL_PACKETS_BELOW_2_15_AND_NO_LITERAL_REFINEMENT",
            "payload": "HOLD_NO_QWEN_COARSE_OR_CONTROL_PAYLOAD",
        },
        "no_raw_information_fallback": True,
        "qwen_payload_accessed": False, "coarse_payload_accessed": False,
        "control_payload_accessed": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt)).hexdigest()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auditor", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    encoded = canonical(make_receipt(Path(args.auditor))).decode("ascii")
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="ascii", newline="\n")
    print(encoded)


if __name__ == "__main__":
    main()
