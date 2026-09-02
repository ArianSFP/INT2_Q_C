#!/usr/bin/env python3
"""Named regression ledger for every independent-v0 audit exploit."""

from __future__ import annotations

import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
import authority
import physical_authority


class PriorAuditExploitRegressions(unittest.TestCase):
    def test_exact_closure_enumerates_all_entry_kinds(self):
        text = inspect.getsource(authority.authenticate_flat_package)
        self.assertIn("os.scandir", text)
        self.assertIn("follow_symlinks=False", text)

    def test_hook_api_has_no_injected_module_or_reference_callable(self):
        parameters = inspect.signature(authority.run_isolated_worker).parameters
        self.assertNotIn("base_module", parameters)
        self.assertNotIn("reference_hook", parameters)
        self.assertNotIn("backend", parameters)

    def test_result_api_has_no_rate_mse_f_packet_or_decoded_array_parameters(self):
        parameters = inspect.signature(
            physical_authority.validate_physical_bundle).parameters
        for forbidden in ("rate", "mse", "F", "packet", "reconstruction",
                          "decoder", "read_amplification", "universal"):
            self.assertNotIn(forbidden, parameters)

    def test_fake_backend_inside_controlled_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            (root / "audit_fake_cupy.py").write_text("fabricated = True\n",
                                                      encoding="utf-8")
            sys.path.insert(0, str(root))
            try:
                with self.assertRaises(Exception):
                    authority.module_origin_outside_controlled_roots(
                        "audit_fake_cupy", [root])
            finally:
                sys.path.pop(0)
                sys.modules.pop("audit_fake_cupy", None)

    def test_production_and_fixture_authorizations_are_not_aliases(self):
        self.assertNotEqual(physical_authority.PRODUCTION_AUTHORIZATION,
                            physical_authority.FIXTURE_AUTHORIZATION)

    def test_read_trace_requires_literal_packet_object_only(self):
        with self.assertRaises(Exception):
            physical_authority._validate_trace({
                "schema": "strata-rm-v1-read-trace", "packet_bytes": 8,
                "operations": [{"object": "common-model", "offset": 0,
                                "length": 8}]}, 8)

    def test_no_declared_boolean_can_replace_canonical_packet_bytes(self):
        source = inspect.getsource(physical_authority._run_case)
        self.assertIn("canonical == packet", source)
        self.assertIn("exact_bf16_f64_score", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)

