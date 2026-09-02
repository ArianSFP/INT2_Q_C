#!/usr/bin/env python3
"""Hostile source-only tests for UWFA-SC posterior-centroid v0."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__authenticated_sha256__ = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


core = load("uwfa_pc_v0_test_core", "posterior_core.py")
bridge = load("uwfa_pc_v0_test_bridge", "result_bridge.py")
diagnostic = load("uwfa_pc_v0_test_diagnostic", "diagnostic.py")


class FakeCommon:
    LEVELS = 6
    CONTEXTS = 16

    @staticmethod
    def public_context(level, base, within):
        return (int(level) + (int(base) >> 12) + (int(within) & 3)) & 15

    @staticmethod
    def transition(candidate, state, bit, context, within):
        return (int(state) * 3 + int(bit) + int(context) + int(within)) % int(candidate.states)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def synthetic_blocks(states: int = 4):
    blocks = []
    ordinal = 0
    for component, owners in enumerate(((0, 1), (2, 3), (4, 5))):
        for local in range(7):
            indices = np.tile(np.arange(64, dtype=np.int16), 4)
            occupancy = np.zeros((6, states), dtype=np.float64)
            for level in range(6):
                raw = np.asarray([
                    math.sin((component + 1) * (local + 2) * (level + 1) * (state + 1))
                    for state in range(states)
                ], dtype=np.float64)
                raw -= np.mean(raw)
                occupancy[level] = 0.045 * raw
            q = 0.25 * (indices.astype(np.float64) - 31.0)
            correction = (
                0.08 * occupancy[0, 0]
                - 0.11 * occupancy[2, 1]
                + q * (0.06 * occupancy[1, 2] - 0.05 * occupancy[5, 3])
            )
            target = q + correction
            blocks.append(core.BlockObservation(
                ordinal=ordinal,
                owners=owners,
                indices=indices,
                target_normalized=target,
                occupancy=occupancy,
                coordinate_mapping_sha256=digest(f"map-{ordinal}".encode()),
            ))
            ordinal += 1
    return tuple(blocks)


class CoreTests(unittest.TestCase):
    def test_struct_geometry_and_bound(self):
        self.assertEqual(core.HEAD_HEADER.size, 96)
        self.assertEqual(core.WRAPPER_FOOTER.size, 192)
        self.assertEqual(core.parameter_count(core.LAW_STATE, 64), 770)
        self.assertEqual(96 + 2 * 770, 1636)
        self.assertLessEqual(1636, 4096 - 192)

    def test_predecision_trace_is_before_update_and_resets(self):
        common = FakeCommon()
        candidate = SimpleNamespace(states=4, reset_length=5)
        bits = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
        levels = [index % 6 for index in range(len(bits))]
        base = [20000 + 1000 * (index % 4) for index in range(len(bits))]
        observed = core.trace_predecision_states(common, candidate, bits, levels, base)
        self.assertEqual(observed[0], 0)
        self.assertEqual(observed[5], 0)
        self.assertEqual(observed[10], 0)
        state = 0
        expected = []
        for position, bit in enumerate(bits):
            within = position % 5
            if within == 0:
                state = 0
            context = common.public_context(levels[position], base[position], within)
            expected.append(state)
            state = common.transition(candidate, state, bit, context, within)
        self.assertEqual(observed, expected)

    def test_occupancy_centering_empty_level_and_permutation(self):
        levels = np.asarray([0, 0, 0, 1, 1, 3], dtype=np.uint8)
        states = np.asarray([0, 1, 1, 2, 2, 3], dtype=np.uint16)
        occupancy = core.occupancy_features(np, levels, states, 4)
        np.testing.assert_allclose(np.sum(occupancy, axis=1), 0.0, atol=1e-15)
        np.testing.assert_array_equal(occupancy[2], np.zeros(4))
        permuted = core.permute_occupancy(np, occupancy, 7)
        for level in range(6):
            np.testing.assert_array_equal(np.sort(permuted[level]), np.sort(occupancy[level]))
        self.assertFalse(np.array_equal(permuted, occupancy))
        self.assertNotEqual(core.state_permutation(7, 0, 4), core.state_permutation(8, 0, 4))

    def test_hard_fail_without_coordinate_alignment(self):
        block = core.BlockObservation(
            ordinal=0,
            owners=(0,),
            indices=np.asarray([0, 1, 2], dtype=np.int16),
            target_normalized=np.asarray([0.0, 1.0], dtype=np.float64),
            occupancy=np.zeros((6, 4), dtype=np.float64),
            coordinate_mapping_sha256=digest(b"mapping"),
        )
        with self.assertRaises(core.PosteriorContractError):
            core.validate_block(np, block, 4)

    def test_source_free_decoder_features_predict_but_cannot_fit(self):
        block = synthetic_blocks()[0]
        decoder_only = core.BlockObservation(
            ordinal=block.ordinal,
            owners=block.owners,
            indices=block.indices,
            target_normalized=None,
            occupancy=block.occupancy,
            coordinate_mapping_sha256=block.coordinate_mapping_sha256,
        )
        parameters = np.zeros(core.parameter_count(core.LAW_STATE, 4))
        predicted = core.predict_normalized(
            np, decoder_only, parameters, law=core.LAW_STATE, states=4
        )
        np.testing.assert_allclose(
            predicted, 0.25 * (block.indices.astype(np.float64) - 31.0)
        )
        with self.assertRaises(core.PosteriorContractError):
            core.fit_head(
                np,
                (decoder_only,),
                law=core.LAW_STATE,
                states=4,
                ridge_exponent=-20,
            )

    def test_fit_recovers_continuous_coordinate_law(self):
        blocks = synthetic_blocks()
        state = core.fit_head(np, blocks, law=core.LAW_STATE, states=4, ridge_exponent=-28)
        local = core.fit_head(np, blocks, law=core.LAW_LOCAL, states=4, ridge_exponent=-28)
        state_sse = 0.0
        local_sse = 0.0
        for block in blocks:
            state_error = block.target_normalized - core.predict_normalized(
                np, block, state, law=core.LAW_STATE, states=4
            )
            local_error = block.target_normalized - core.predict_normalized(
                np, block, local, law=core.LAW_LOCAL, states=4
            )
            state_sse += float(np.sum(state_error * state_error))
            local_sse += float(np.sum(local_error * local_error))
        self.assertLess(state_sse, 1e-6 * local_sse)

    def test_whole_component_nested_selection(self):
        blocks = synthetic_blocks()
        components = core.owner_components(6, [block.owners for block in blocks])
        self.assertEqual(components, ((0, 1), (2, 3), (4, 5)))

        def score(parameters, law, component):
            selected = core.blocks_for_components(blocks, components, (component,))
            total = 0.0
            rounded = np.asarray(parameters).astype("<f2").astype(np.float64)
            for block in selected:
                error = block.target_normalized - core.predict_normalized(
                    np, block, rounded, law=law, states=4
                )
                total += float(np.sum(error * error))
            return total

        selected = core.select_ridge_for_outer(
            np,
            blocks,
            components,
            outer_component=0,
            law=core.LAW_STATE,
            states=4,
            score_sse=score,
        )
        self.assertEqual(selected["development_components"], [1, 2])
        self.assertEqual(len(selected["ridge_grid"]), 8)
        for row in selected["ridge_grid"]:
            self.assertEqual(
                {(item["train_component"], item["validation_component"]) for item in row["directions"]},
                {(1, 2), (2, 1)},
            )

    def test_head_roundtrip_and_tamper(self):
        parameters = np.linspace(-0.1, 0.1, core.parameter_count(core.LAW_STATE, 4))
        binding = digest(b"handoff")
        packet = core.serialize_head(
            np,
            parameters,
            law=core.LAW_STATE,
            states=4,
            ridge_exponent=-20,
            handoff_root_sha256=binding,
        )
        parsed = core.parse_head(np, packet, expected_handoff_root_sha256=binding)
        self.assertTrue(parsed["canonical_reencode_matches"])
        self.assertEqual(parsed["packet_bytes"], 96 + 2 * 50)
        damaged = bytearray(packet)
        damaged[-1] ^= 1
        with self.assertRaises(core.PosteriorContractError):
            core.parse_head(np, bytes(damaged), expected_handoff_root_sha256=binding)

    def test_suffix_wrapper_roundtrip_and_tamper(self):
        binding = digest(b"handoff")
        head = core.serialize_head(
            np,
            np.zeros(core.parameter_count(core.LAW_STATE, 4)),
            law=core.LAW_STATE,
            states=4,
            ridge_exponent=-20,
            handoff_root_sha256=binding,
        )
        inner = bytes((index * 17) & 255 for index in range(8192))
        wrapper = core.build_wrapper(
            inner,
            head,
            weights=12288,
            experts=6,
            fold_ordinal=1,
            handoff_root_sha256=binding,
        )
        self.assertEqual(wrapper[: len(inner)], inner)
        self.assertEqual(len(wrapper), len(inner) + 4096)
        parsed = core.parse_wrapper(np, wrapper, expected_handoff_root_sha256=binding)
        self.assertEqual(parsed["inner"], inner)
        self.assertTrue(parsed["canonical_reencode_matches"])
        damaged = bytearray(wrapper)
        damaged[len(inner) + len(head) + 3] = 1
        with self.assertRaises(core.PosteriorContractError):
            core.parse_wrapper(np, bytes(damaged), expected_handoff_root_sha256=binding)

    def test_instrumented_inner_decode_through_literal_wrapper_charges_overlap(self):
        binding = digest(b"instrumented-handoff")
        head = core.serialize_head(
            np,
            np.zeros(core.parameter_count(core.LAW_STATE, 4)),
            law=core.LAW_STATE,
            states=4,
            ridge_exponent=-20,
            handoff_root_sha256=binding,
        )
        inner = bytes((index * 29) & 255 for index in range(8192))
        wrapper = core.build_wrapper(
            inner,
            head,
            weights=12288,
            experts=2,
            fold_ordinal=-1,
            handoff_root_sha256=binding,
        )

        class FakeSession:
            def __init__(self):
                self.next = 0

            def decode_expert(self, route):
                expert = route["expert_ordinal"]
                self.next += 1
                return {
                    "expert_ordinal": expert,
                    "decoded_streams": 1,
                    "all_payloads_canonically_reencoded": True,
                    "all_three_roles_reconstructed": True,
                    "routed_expert_reconstruction_sha256": digest(f"expert-{expert}".encode()),
                }

            def finalize(self, *, experts, expected_full_reconstruction_sha256):
                return {
                    "experts": experts,
                    "full_reconstruction_f64_sha256": expected_full_reconstruction_sha256,
                    "matches_container_reconstruction": self.next == experts,
                }

        class FakeAdapter:
            def __init__(self, **_kwargs):
                pass

            def new_routed_decoder(self):
                return FakeSession()

        class FakeCodec:
            @staticmethod
            def parse_container(_common, _semantic, raw):
                self.assertEqual(raw, inner)
                return {"reconstruction_sha256": digest(b"full")}

            @staticmethod
            def routed_read_expert(_common, _semantic, reader, *, file_size, expert, externally_authenticated_container_sha256, decode_routed_expert):
                self.assertEqual(file_size, len(inner))
                self.assertEqual(externally_authenticated_container_sha256, digest(inner))
                reader.read(0, 4096)
                reader.read(32 + expert, 96)
                route = {"expert_ordinal": expert}
                causal = decode_routed_expert(route)
                return {
                    "routed_read_ranges": tuple(reader.ranges),
                    "causal_decode_reencode_reconstruction": causal,
                }

        modules = {
            "common": object(),
            "codec": FakeCodec,
            "semantic": object(),
            "adapter_source": SimpleNamespace(StrataSCAdapter=FakeAdapter),
            "frozen": object(),
            "strata": object(),
        }
        proof = bridge.instrument_inner_routed_decode_through_wrapper(
            np,
            core,
            modules,
            wrapper,
            expected_handoff_root_sha256=binding,
            rht_device="numpy",
        )
        self.assertTrue(proof["actual_inner_routed_decode_executed"])
        self.assertFalse(proof["actual_posterior_wrapper_routed_decode_executed"])
        self.assertTrue(proof["compressed_expert_second_pass_forbidden_and_absent"])
        self.assertGreater(proof["experts"][0]["overlap_bytes_requested_again"], 0)
        rebound = bridge.bind_wrapper_to_routed_proof(
            np,
            core,
            wrapper,
            proof,
            expected_handoff_root_sha256=binding,
        )
        self.assertEqual(
            rebound["experts"][0]["requested_bytes_with_repetition"],
            proof["experts"][0]["requested_bytes_with_repetition"],
        )

    def test_read_ledger_adds_one_page_without_second_pass(self):
        trace = {
            "schema": "uwfa-sc-posterior-wrapper-routed-read-proof-v0",
            "proof_uses_actual_authenticated_v8_routed_decoder": True,
            "compressed_expert_second_pass_forbidden_and_absent": True,
            "inner_bytes": 8192,
            "head_bytes": 196,
            "proof_sha256": digest(b"actual-proof"),
            "experts": [],
        }
        for expert, requested in enumerate((7100, 7200)):
            trace["experts"].append({
                "expert_ordinal": expert,
                "extension_page_read_requests": 1,
                "inner_decode_invocations": 1,
                "compressed_expert_second_pass": False,
                "compressed_expert_second_pass_absent_derived": True,
                "overlap_is_charged_not_interpreted_as_second_pass": True,
                "touched_page_bytes": 8192,
                "requested_bytes_with_repetition": requested,
                "unique_requested_bytes": requested - 100,
                "read_request_count": 5,
                "overlap_bytes_requested_again": 100,
                "causal_decode_reencode_reconstruction": {
                    "all_payloads_canonically_reencoded": True,
                    "all_three_roles_reconstructed": True,
                },
            })
        ledger = core.wrapper_read_ledger(
            routed_wrapper_trace=trace,
            weights_by_expert=(2048, 2048),
            inner_attributed_total=(Fraction(4096), Fraction(4096)),
            inner_attributed_nonpadding=(Fraction(3900), Fraction(3900)),
            head_bytes=196,
        )
        self.assertEqual(ledger["outer_bytes"], 12288)
        self.assertEqual(ledger["additional_storage_pages_per_routed_expert"], 1)
        self.assertTrue(ledger["compressed_expert_second_pass_forbidden_and_absent"])
        self.assertEqual(ledger["experts"][0]["touched_page_bytes"], 8192)
        self.assertEqual(ledger["experts"][0]["read_request_count"], 5)
        self.assertTrue(ledger["passes_requested_with_repetition_below_2x"])
        self.assertFalse(ledger["actual_posterior_wrapper_routed_decode_executed"])

    def test_read_ledger_rejects_second_pass_and_repeated_request_over_2x(self):
        def make_trace(*, invocations=1, requested=8100):
            return {
                "schema": "uwfa-sc-posterior-wrapper-routed-read-proof-v0",
                "proof_uses_actual_authenticated_v8_routed_decoder": True,
                "compressed_expert_second_pass_forbidden_and_absent": invocations == 1,
                "inner_bytes": 8192,
                "head_bytes": 196,
                "proof_sha256": digest(b"hostile-proof"),
                "experts": [{
                    "expert_ordinal": 0,
                    "extension_page_read_requests": 1,
                    "inner_decode_invocations": invocations,
                    "compressed_expert_second_pass": invocations > 1,
                    "compressed_expert_second_pass_absent_derived": invocations == 1,
                    "overlap_is_charged_not_interpreted_as_second_pass": True,
                    "touched_page_bytes": 8192,
                    "requested_bytes_with_repetition": requested,
                    "unique_requested_bytes": 7900,
                    "read_request_count": 5,
                    "overlap_bytes_requested_again": requested - 7900,
                    "causal_decode_reencode_reconstruction": {
                        "all_payloads_canonically_reencoded": True,
                        "all_three_roles_reconstructed": True,
                    },
                }],
            }
        with self.assertRaises(core.PosteriorContractError):
            core.wrapper_read_ledger(
                routed_wrapper_trace=make_trace(invocations=2, requested=16000),
                weights_by_expert=(4096,),
                inner_attributed_total=(Fraction(8192),),
                inner_attributed_nonpadding=(Fraction(8000),),
                head_bytes=196,
            )
        ledger = core.wrapper_read_ledger(
            routed_wrapper_trace=make_trace(requested=26000),
            weights_by_expert=(4096,),
            inner_attributed_total=(Fraction(8192),),
            inner_attributed_nonpadding=(Fraction(8000),),
            head_bytes=196,
        )
        self.assertFalse(ledger["passes_requested_with_repetition_below_2x"])
        self.assertFalse(ledger["passes_strict_cold_read_below_2x"])

    def test_fold_gate_rejects_each_physical_failure(self):
        base = dict(
            delta_s_value=0.01,
            g_state_value=0.001,
            candidate_rate_bpw=2.4,
            candidate_f=0.79,
            cold_read_below_2x=True,
        )
        self.assertTrue(core.state_fold_gate(**base)["passes_all_fold_gates"])
        cases = (
            ("passes_rate_interval", {"candidate_rate_bpw": 2.5000001}),
            ("passes_F_target", {"candidate_f": 0.8000001}),
            ("passes_cold_read_below_2x", {"cold_read_below_2x": False}),
            ("passes_positive_Delta_s", {"delta_s_value": 0.0}),
            ("passes_positive_G_state", {"g_state_value": 0.0}),
        )
        for failed_key, replacement in cases:
            values = dict(base)
            values.update(replacement)
            observed = core.state_fold_gate(**values)
            self.assertFalse(observed[failed_key])
            self.assertFalse(observed["passes_all_fold_gates"])


class PublicationAndSourceTests(unittest.TestCase):
    def make_result_publication(self, root: Path):
        inner = bytes((index * 13) & 255 for index in range(8192))
        members = {
            "BOUND_BASELINE_SCORE.json": b"{}\n",
            "DECODER_BUNDLE.json": b"{}\n",
            "IDENTITY_FRAMING.bin": b"identity",
            "SOURCE_PREFLIGHT.json": b"{}\n",
            "UWFCV8.bin": inner,
        }
        result = {
            "schema": bridge.RESULT_SCHEMA,
            "positive_claim_authority": False,
            "controls_run": False,
            "physical": {"container_sha256": digest(inner), "container_bytes": len(inner)},
            "source_final": {"container_sha256": digest(inner)},
        }
        members["RESULT.json"] = (
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        ).encode()
        for name, payload in members.items():
            (root / name).write_bytes(payload)
        complete = {
            "schema": bridge.COMPLETE_SCHEMA,
            "status": "NONPROMOTING_TEST",
            "positive_claim_authority": False,
            "members": [
                {"name": name, "bytes": len(payload), "sha256": digest(payload)}
                for name, payload in sorted(members.items())
            ],
        }
        complete["completion_sha256"] = digest(bridge.canonical_json(complete))
        (root / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
        return inner

    def test_completed_result_authentication_and_member_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inner = self.make_result_publication(root)
            authenticated = bridge.authenticate_result_directory(root)
            self.assertEqual(authenticated["inner"], inner)
            with (root / "UWFCV8.bin").open("ab") as stream:
                stream.write(b"x")
            with self.assertRaises(bridge.BridgeError):
                bridge.authenticate_result_directory(root)

    def test_new_publication_writes_completion_last_and_rejects_existing(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            output = parent / "result"
            record = diagnostic._write_exclusive(
                output,
                {"RESULT.json": b"result\n", "Z.bin": b"z"},
                completion_payload=b"complete\n",
            )
            self.assertTrue(record["completion_written_last"])
            self.assertEqual(record["write_order"][-1], "COMPLETE.json")
            self.assertEqual((output / "COMPLETE.json").read_bytes(), b"complete\n")
            with self.assertRaises(diagnostic.DiagnosticError):
                diagnostic._write_exclusive(
                    output,
                    {"RESULT.json": b"again"},
                    completion_payload=b"again-complete",
                )

    def test_new_publication_rejects_symlinked_parent(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symlink unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            try:
                os.symlink(real, linked, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(diagnostic.DiagnosticError):
                diagnostic._write_exclusive(
                    linked / "result",
                    {"RESULT.json": b"x"},
                    completion_payload=b"done",
                )

    def test_generic_source_manifest_rejects_identity_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifact = digest(b"artifact")
            rows = []
            clean = []
            experts, intermediate, hidden = 2, 3, 4
            for expert in range(experts):
                for role in ("gate", "up", "down"):
                    shape = [hidden, intermediate] if role == "down" else [intermediate, hidden]
                    values = np.linspace(-1.0, 1.0, intermediate * hidden, dtype=np.float32)
                    words = values.view(np.uint32)
                    raw = (words >> np.uint32(16)).astype("<u2").tobytes()
                    name = f"e{expert}_{role}.bf16"
                    (root / name).write_bytes(raw)
                    row = {
                        "expert_ordinal": expert,
                        "role": role,
                        "shape": shape,
                        "relative_path": name,
                        "bytes": len(raw),
                        "sha256": digest(raw),
                    }
                    rows.append(row)
                    clean.append({key: row[key] for key in ("expert_ordinal", "role", "shape", "bytes", "sha256")})
            clean.sort(key=lambda row: (row["expert_ordinal"], ("gate", "up", "down").index(row["role"])))
            manifest = {
                "schema": diagnostic.SOURCE_MANIFEST_SCHEMA,
                "bound_artifact_sha256": artifact,
                "experts": experts,
                "source_record_set_sha256": digest(diagnostic.canonical_json(clean)),
                "matrices": rows,
            }
            path = root / "manifest.json"
            payload = json.dumps(manifest, sort_keys=True).encode()
            path.write_bytes(payload)
            loaded = diagnostic.authenticate_source_panel(
                np,
                path,
                expected_manifest_sha256=digest(payload),
                expected_artifact_sha256=artifact,
                experts=experts,
                intermediate=intermediate,
                hidden=hidden,
            )
            self.assertFalse(loaded["identity_fields_available_to_decoder"])
            manifest["matrices"][0]["tensor_name"] = "forbidden"
            bad = json.dumps(manifest, sort_keys=True).encode()
            path.write_bytes(bad)
            with self.assertRaises(diagnostic.DiagnosticError):
                diagnostic.authenticate_source_panel(
                    np,
                    path,
                    expected_manifest_sha256=digest(bad),
                    expected_artifact_sha256=artifact,
                    experts=experts,
                    intermediate=intermediate,
                    hidden=hidden,
                )

    def test_source_aperture_does_not_open_unselected_leaves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifact = digest(b"artifact-aperture")
            rows = []
            clean = []
            experts, intermediate, hidden = 2, 2, 3
            for expert in range(experts):
                for role in ("gate", "up", "down"):
                    shape = [hidden, intermediate] if role == "down" else [intermediate, hidden]
                    values = np.arange(intermediate * hidden, dtype=np.float32)
                    raw = (values.view(np.uint32) >> np.uint32(16)).astype("<u2").tobytes()
                    name = f"expert{expert}_{role}.bf16"
                    if expert == 0:
                        (root / name).write_bytes(raw)
                    row = {
                        "expert_ordinal": expert,
                        "role": role,
                        "shape": shape,
                        "relative_path": name,
                        "bytes": len(raw),
                        "sha256": digest(raw),
                    }
                    rows.append(row)
                    clean.append({key: row[key] for key in ("expert_ordinal", "role", "shape", "bytes", "sha256")})
            clean.sort(key=lambda row: (row["expert_ordinal"], ("gate", "up", "down").index(row["role"])))
            manifest = {
                "schema": diagnostic.SOURCE_MANIFEST_SCHEMA,
                "bound_artifact_sha256": artifact,
                "experts": experts,
                "source_record_set_sha256": digest(diagnostic.canonical_json(clean)),
                "matrices": rows,
            }
            payload = json.dumps(manifest, sort_keys=True).encode()
            path = root / "manifest.json"
            path.write_bytes(payload)
            selected = diagnostic.authenticate_source_panel(
                np,
                path,
                expected_manifest_sha256=digest(payload),
                expected_artifact_sha256=artifact,
                experts=experts,
                intermediate=intermediate,
                hidden=hidden,
                selected_experts=(0,),
            )
            self.assertEqual(selected["materialized_experts"], [0])
            self.assertFalse(
                selected["unselected_BF16_leaves_opened_statted_hashed_or_enumerated"]
            )
            with self.assertRaises(OSError):
                diagnostic.authenticate_source_panel(
                    np,
                    path,
                    expected_manifest_sha256=digest(payload),
                    expected_artifact_sha256=artifact,
                    experts=experts,
                    intermediate=intermediate,
                    hidden=hidden,
                    selected_experts=(1,),
                )


class SourceBoundaryTests(unittest.TestCase):
    def test_exact_v8_authenticated_dataclass_sources_load(self):
        v8 = (ROOT.parent / "unifilar_wfa_entropy_census_stage0_v8").resolve()
        manifest_payload = (v8 / "SOURCE_MANIFEST.json").read_bytes()
        closure = bridge.authenticate_v8_package(
            v8,
            expected_manifest_sha256=digest(manifest_payload),
        )
        strata = (ROOT.parent.parent / "strata_expert_local_codec" / "common.py").resolve()
        frozen = (ROOT.parent.parent / "strata_v2_klt_mixed_independent_auditor_v1.py").resolve()
        result_record = {
            "source_hashes": {
                "sealed_v8_manifest_sha256": digest(manifest_payload),
                "strata_expert_local_codec_common_sha256": digest(strata.read_bytes()),
                "strata_v2_klt_mixed_independent_auditor_sha256": digest(frozen.read_bytes()),
            },
        }
        loaded = bridge.load_authenticated_decoders(
            result_record,
            closure,
            strata_common_path=strata,
            frozen_auditor_path=frozen,
        )
        self.assertTrue(hasattr(loaded["common"], "Candidate"))
        self.assertTrue(hasattr(loaded["codec"], "OwnerContribution"))
        self.assertIs(sys.modules[loaded["common"].__name__], loaded["common"])
        self.assertEqual(
            loaded["common"].__authenticated_sha256__,
            closure["member_hashes"]["uwfa_common.py"],
        )

    def test_owned_modules_compile_from_retained_authenticated_bytes(self):
        payload = (ROOT / "SOURCE_MANIFEST.json").read_bytes()
        closure = diagnostic.authenticate_own_package(
            ROOT,
            expected_manifest_sha256=digest(payload),
        )
        loaded = diagnostic._load_authenticated_owned_module(
            closure,
            member_name="posterior_core.py",
            private_name="uwfa_pc_v0_test_retained_core",
        )
        self.assertEqual(
            loaded.__authenticated_sha256__,
            digest(closure["sources"]["posterior_core.py"]),
        )
        hostile = dict(closure)
        hostile["sources"] = dict(closure["sources"])
        hostile["sources"]["posterior_core.py"] += b"\n# mutable sibling injection\n"
        with self.assertRaises(diagnostic.DiagnosticError):
            diagnostic._load_authenticated_owned_module(
                hostile,
                member_name="posterior_core.py",
                private_name="uwfa_pc_v0_test_hostile_core",
            )

    def test_frozen_source_manifest_authenticates_exact_closure(self):
        payload = (ROOT / "SOURCE_MANIFEST.json").read_bytes()
        closure = diagnostic.authenticate_own_package(
            ROOT,
            expected_manifest_sha256=digest(payload),
        )
        self.assertEqual(len(closure["members"]), 6)
        self.assertEqual(
            closure["source_snapshot_root_sha256"],
            json.loads(payload)["source_snapshot_root_sha256"],
        )

    def test_no_sealed_source_edits_or_payload_paths(self):
        design = json.loads((ROOT / "design_lock.json").read_text(encoding="utf-8"))
        self.assertFalse(design["predecessor"]["predecessor_is_modified"])
        self.assertTrue(design["semantic_correction"]["hard_fail_without_coordinate_aligned_observations"])
        self.assertFalse(design["semantic_correction"]["selected_sc_decisions_are_scalar_bins"])
        source = (ROOT / "diagnostic.py").read_text(encoding="utf-8")
        for forbidden in (
            "model.layers.5",
            "Qwen/Qwen3",
            "ad44e777",
            "fe4fd2b8438d",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "schema": "uwfa-sc-posterior-centroid-source-test-v0",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.wasSuccessful(),
        "payload_accessed": False,
        "cuda_initialized": False,
    }, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if result.wasSuccessful() else 1)
