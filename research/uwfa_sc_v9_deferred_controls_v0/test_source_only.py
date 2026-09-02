#!/usr/bin/env python3
"""Standard-library hostile tests for the source-only deferred-control lock."""

from __future__ import annotations

import contextlib
import io
import json
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PACKAGE = Path(__file__).resolve().parent


def load_core() -> dict[str, object]:
    source = (PACKAGE / "control_core.py").read_bytes()
    name = "uwfa_v9_deferred_control_core_test_snapshot"
    module = types.ModuleType(name)
    module.__file__ = "<authenticated-control-core-snapshot>"
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module.__dict__


def synthetic_panel() -> dict[str, object]:
    streams = []
    for ordinal, (role, profile, bits) in enumerate(
        (
            ("gate", 11, [0, 0, 1, 1, 0, 1, 0, 1]),
            ("gate", 11, [1, 1, 0, 0, 1, 0, 1, 0]),
            ("mixed", 13, [0, 1, 1, 0, 0, 0, 1, 1]),
        )
    ):
        levels = [position % 2 for position in range(len(bits))]
        base = [4096 + 8192 * (position % 3) for position in range(len(bits))]
        base_bytes = b"".join(value.to_bytes(2, "little") for value in base)
        streams.append(
            {
                "stream_ordinal": ordinal,
                "role": role,
                "profile_q": profile,
                "symbols": len(bits),
                "bits": list(bits),
                "levels": levels,
                "base": base,
                "bits_bytes": bytes(bits),
                "levels_bytes": bytes(levels),
                "base_bytes": base_bytes,
            }
        )
    return {"streams": streams, "weights": 24, "experts": 2}


class SourceOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.core = load_core()

    def test_exact_seeds_aperture_and_work(self) -> None:
        self.assertEqual(
            self.core["CONTROL_SEEDS"],
            (10619863, 10619881, 10619909, 10619927, 10619953, 10619971, 10619999, 10620017),
        )
        ledger = self.core["WorkLedger"]().as_dict()
        self.assertEqual(ledger["updates_per_complete_pipeline"], 38_621_316_130)
        self.assertEqual(ledger["matched_control_updates_for_positive_specificity"], 308_970_529_040)
        self.assertEqual(ledger["structure_shuffle_updates"], 231_727_896_780)
        self.assertEqual(ledger["maximum_deferred_updates"], 540_698_425_820)

    def test_phase_preserving_reference_is_deterministic_and_conservative(self) -> None:
        panel = synthetic_panel()
        histogram = self.core["bucket_bit_histogram"](panel, preserve_phase=True)
        first = self.core["role_profile_phase_preserving_reference"](panel)
        second = self.core["role_profile_phase_preserving_reference"](panel)
        self.assertEqual(first, second)
        self.assertEqual(
            histogram,
            self.core["bucket_bit_histogram"](first, preserve_phase=True),
        )
        for before, after in zip(panel["streams"], first["streams"], strict=True):
            self.assertEqual(before["levels"], after["levels"])
            self.assertEqual(before["base"], after["base"])
            self.assertEqual(before["role"], after["role"])
            self.assertEqual(before["profile_q"], after["profile_q"])

    def test_phase_destroying_reference_preserves_coarser_buckets(self) -> None:
        panel = synthetic_panel()
        coarse = self.core["bucket_bit_histogram"](panel, preserve_phase=False)
        transformed = self.core["role_profile_phase_destroying_reference"](panel)
        self.assertEqual(
            coarse,
            self.core["bucket_bit_histogram"](transformed, preserve_phase=False),
        )
        self.assertTrue(
            all(row["diagnostic_transform"].endswith("phase_destroying") for row in transformed["streams"])
        )

    def test_mixed_role_never_relabelled(self) -> None:
        panel = synthetic_panel()
        transformed = self.core["role_profile_phase_preserving_reference"](panel)
        self.assertEqual(transformed["streams"][2]["role"], "mixed")

    def test_runtime_is_static_block_and_payload_entry_raises(self) -> None:
        record = self.core["runtime_block_record"]()
        self.assertEqual(
            record["status"],
            "BLOCK_MISSING_DECODER_CLOSED_MATCHED_CONTROL_PRODUCER_AND_AUDIT_PINS",
        )
        self.assertFalse(record["payload_access_authority"])
        with self.assertRaises(self.core["DeferredRuntimeBlock"]):
            self.core["payload_entrypoint"]("ignored", artifact="ignored")

    def test_import_core_is_path_and_cuda_inert(self) -> None:
        source = (PACKAGE / "control_core.py").read_bytes()
        module = types.ModuleType("inert_snapshot")
        module.__file__ = "<snapshot>"
        with mock.patch("builtins.open", side_effect=AssertionError("open")), mock.patch(
            "pathlib.Path.open", side_effect=AssertionError("path open")
        ):
            previous = sys.modules.get(module.__name__)
            sys.modules[module.__name__] = module
            try:
                exec(compile(source, "<snapshot>", "exec"), module.__dict__)
            finally:
                if previous is None:
                    sys.modules.pop(module.__name__, None)
                else:
                    sys.modules[module.__name__] = previous
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("cupy", sys.modules)

    def test_primary_summary_helper_rejects_untrusted_non_survivor(self) -> None:
        with self.assertRaises(self.core["ControlContractError"]):
            self.core["validate_primary_summary_without_opening"](
                {"status": "HARD_KILL_PRIMARY_PHYSICAL_RATE_OR_F"}
            )
        value = {
            "status": self.core["PRIMARY_SURVIVOR_STATUS"],
            "positive_claim_authority": False,
            "controls_run": False,
            "shuffles_run": False,
            "physical": {
                "passes_rate_interval": True,
                "passes_F_target": True,
                "passes_cold_read_below_2x": True,
            },
        }
        self.core["validate_primary_summary_without_opening"](value)

    def test_block_file_matches_bounded_finding(self) -> None:
        row = json.loads((PACKAGE / "BLOCK.json").read_text(encoding="utf-8"))
        self.assertFalse(row["payload_access_authority"])
        self.assertIn("encode arbitrary matrices", " ".join(row["bounded_api_finding"]["v8_cannot"]))
        self.assertIn("REQUIRES_REENCODE", self.core["ROW_COLUMN_DISPOSITION"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
