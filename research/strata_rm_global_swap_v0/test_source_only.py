#!/usr/bin/env python3
"""Source-only mechanism and exact-orientation tests."""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

import numpy as np

from coset_contract import (CURRENT_RANDOM, ZERO, HeldCosetFork,
                            current_random_frozen_external, frozen_external)
from pin_semantics import authenticate
from result_contract import (EXPECTED_PINS, REQUIRED_PACKET_FIELDS,
                             validate_independent_result)
from rm_order import (bit_reverse_indices, classify_selected_count, generated_row,
                      rm_dimension, rm_full_order_numpy, rm_information_positions,
                      swap_reference_flags)
from swap_adapter import make_replacement


def external_root() -> Path:
    configured = os.environ.get("STRATA_EXTERNAL_ROOT")
    return Path(configured).resolve() if configured else Path(__file__).resolve().parents[3]


def valid_result_fixture() -> dict:
    n = 1 << 20
    ks = [1, 12345, n // 2, n - 7, n, n]
    levels = []
    for k in ks:
        levels.append({
            "reference_bec_k": k,
            "rm_ordered_k": k,
            "set_name": classify_selected_count(n, k)["name"],
        })
    packet_bytes = 327680  # exactly 2.5 bpw for 2**20 weights
    return {
        "schema": "strata-rm-global-swap-v0-independent-physical-result",
        "external_pins": EXPECTED_PINS,
        "candidate": "RM-ordered truncated polar",
        "coset": "current_random",
        "rate_basis": "literal_full_packet_bytes_plus_charged_shared_bytes",
        "independent_decoder_source_sha256": "1" * 64,
        "independent_decode_complete": True,
        "causal_probabilities_regenerated": True,
        "packet_consumed_exactly": True,
        "canonical_reencode_byte_identical": True,
        "source_domain_score_from_decoded_packet": True,
        "overlap_receipt_used_for_rd": False,
        "charged_packet_fields": sorted(REQUIRED_PACKET_FIELDS),
        "blocks": [{
            "n": n,
            "levels": levels,
            "literal_packet_bytes": packet_bytes,
            "literal_packet_sha256": "a" * 64,
            "canonical_reencode_sha256": "a" * 64,
        }],
        "charged_shared_bytes": 0,
        "total_original_weights": n,
        "total_physical_bytes": packet_bytes,
        "actual_physical_bpw": 2.5,
        "selected_count_used_as_rate": False,
    }


class OrientationTests(unittest.TestCase):
    def test_internal_phase_row_weight(self) -> None:
        for m in range(1, 8):
            n = 1 << m
            for phase in range(n):
                self.assertEqual(int(np.count_nonzero(generated_row(phase, n))),
                                 1 << phase.bit_count())

    def test_exact_external_base_orientation(self) -> None:
        root = external_root()
        authenticate(root)
        sys.path.insert(0, str(root))
        try:
            base = importlib.import_module("agent_polaris_qwen_rht_encoder")
            for m in range(1, 7):
                n = 1 << m
                self.assertTrue(np.array_equal(base.bit_reverse_indices(n),
                                               bit_reverse_indices(n)))
                reverse = base.bit_reverse_indices(n)
                for phase in range(n):
                    internal = np.zeros(n, dtype=np.uint8)
                    internal[phase] = 1
                    external_row = base.polar_transform(internal[reverse])
                    self.assertTrue(np.array_equal(external_row,
                                                   generated_row(phase, n)))
        finally:
            sys.path.pop(0)

    def test_normative_tie_order(self) -> None:
        order = rm_full_order_numpy(1 << 10)
        for left, right in zip(order[:-1], order[1:], strict=True):
            wl, wr = int(left).bit_count(), int(right).bit_count()
            self.assertTrue(wl > wr or (wl == wr and int(left) < int(right)))
        self.assertTrue(np.array_equal(order, rm_full_order_numpy(1 << 10)))

    def test_exact_rm_classification(self) -> None:
        for m in range(1, 12):
            n = 1 << m
            for r in range(m + 1):
                k = rm_dimension(r, m)
                row = classify_selected_count(n, k)
                self.assertEqual(row["name"], f"RM({r},{m})")
                self.assertTrue(row["exact_rm"])
            truncated = classify_selected_count(n, 0)
            self.assertEqual(truncated["name"], "RM-ordered truncated polar")


class CountAndCosetTests(unittest.TestCase):
    def test_six_level_count_equality(self) -> None:
        rng = np.random.default_rng(929)
        n = 1 << 11
        flags = []
        for k in (0, 1, 73, n // 2, n - 1, n):
            flag = np.ones(n, dtype=np.uint8)
            flag[rng.permutation(n)[:k]] = 0
            flags.append(flag)
        swapped = swap_reference_flags(flags)
        self.assertEqual([int(np.count_nonzero(x == 0)) for x in flags],
                         [int(np.count_nonzero(x == 0)) for x in swapped])

    def test_actual_bec_hook_count_equality_small_n(self) -> None:
        root = external_root()
        authenticate(root)
        sys.path.insert(0, str(root))
        try:
            base = importlib.import_module("agent_polaris_qwen_rht_encoder")
            bg = importlib.import_module("bg_codec_bec_encoder")
            capacities = [0.0, 0.001, 0.237747929331251,
                          0.9153259168218427, 0.9999815811734327, 1.0]
            reference = bg.bec_flags(None, 1 << 12, capacities)
            replacement = make_replacement(bg.bec_flags,
                                           production_lengths_only=False)
            candidate = replacement(None, 1 << 12, capacities)
            self.assertEqual([int(np.count_nonzero(x == 0)) for x in reference],
                             [int(np.count_nonzero(x == 0)) for x in candidate])
            self.assertIs(base.reliability_freeze_flags, bg.bec_flags)
        finally:
            sys.path.pop(0)

    def test_coset_modes_are_not_aliases(self) -> None:
        first = current_random_frozen_external(1 << 12, 123, 1)
        second = frozen_external(1 << 12, 123, 1, CURRENT_RANDOM)
        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(int(np.count_nonzero(first)), 0)
        with self.assertRaises(HeldCosetFork):
            frozen_external(1 << 12, 123, 1, ZERO)


class ContractTests(unittest.TestCase):
    def test_valid_future_contract_fixture(self) -> None:
        self.assertTrue(validate_independent_result(valid_result_fixture())["passed"])

    def test_global_lengths_only_in_adapter(self) -> None:
        def reference(_repo, n, _capacities):
            return [np.zeros(n, dtype=np.uint8) for _ in range(6)]
        replacement = make_replacement(reference)
        with self.assertRaises(ValueError):
            replacement(None, 1 << 18, [1.0] * 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

