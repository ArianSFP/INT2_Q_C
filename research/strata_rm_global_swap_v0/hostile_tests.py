#!/usr/bin/env python3
"""Hostile receipt and semantic-boundary tests."""

from __future__ import annotations

import copy
import unittest

from result_contract import validate_independent_result
from test_source_only import valid_result_fixture


class HostileResultTests(unittest.TestCase):
    def reject(self, mutate) -> None:
        receipt = copy.deepcopy(valid_result_fixture())
        mutate(receipt)
        with self.assertRaises((ValueError, TypeError)):
            validate_independent_result(receipt)

    def test_wrong_pin(self) -> None:
        self.reject(lambda x: x["external_pins"].__setitem__(
            "bg_codec_bec_encoder.py", "0" * 64))

    def test_zero_coset(self) -> None:
        self.reject(lambda x: x.__setitem__("coset", "zero"))

    def test_selected_count_changed(self) -> None:
        self.reject(lambda x: x["blocks"][0]["levels"][1].__setitem__(
            "rm_ordered_k", x["blocks"][0]["levels"][1]["reference_bec_k"] + 1))

    def test_false_exact_rm_name(self) -> None:
        self.reject(lambda x: x["blocks"][0]["levels"][1].__setitem__(
            "set_name", "RM(5,20)"))

    def test_n18_proxy(self) -> None:
        self.reject(lambda x: x["blocks"][0].__setitem__("n", 1 << 18))

    def test_missing_padding_charge(self) -> None:
        self.reject(lambda x: x["charged_packet_fields"].remove("padding"))

    def test_selected_dimension_as_rate(self) -> None:
        self.reject(lambda x: x.__setitem__("selected_count_used_as_rate", True))

    def test_ideal_nll_rate_basis(self) -> None:
        self.reject(lambda x: x.__setitem__("rate_basis", "ideal_nll"))

    def test_noncanonical_packet(self) -> None:
        self.reject(lambda x: x["blocks"][0].__setitem__(
            "canonical_reencode_sha256", "b" * 64))

    def test_claimed_rate_not_bytes(self) -> None:
        self.reject(lambda x: x.__setitem__("actual_physical_bpw", 2.49))

    def test_unaccounted_shared_bytes(self) -> None:
        self.reject(lambda x: x.__setitem__("charged_shared_bytes", 7))

    def test_overlap_promoted_to_rd(self) -> None:
        self.reject(lambda x: x.__setitem__("overlap_receipt_used_for_rd", True))

    def test_causal_probabilities_not_rebuilt(self) -> None:
        self.reject(lambda x: x.__setitem__("causal_probabilities_regenerated", False))

    def test_trailing_bytes_unchecked(self) -> None:
        self.reject(lambda x: x.__setitem__("packet_consumed_exactly", False))


if __name__ == "__main__":
    unittest.main(verbosity=2)

