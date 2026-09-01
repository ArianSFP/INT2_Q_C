"""No-payload, no-CUDA tests for the frozen SILWARP gate."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import silwarp_common as sw
import silwarp_gate as gate


class FakeCP:
    @staticmethod
    def asnumpy(value):
        return np.asarray(value)

    @staticmethod
    def asarray(value):
        return np.asarray(value)


class ProtocolTests(unittest.TestCase):
    def test_frozen_constants_and_split(self) -> None:
        protocol = sw.load_protocol()
        sw.validate_frozen_constants(protocol)
        sw.validate_split(protocol)
        sets = sw.split_sets(protocol)
        self.assertEqual({name: len(value) for name, value in sets.items()}, {
            "fit": 41,
            "calibration": 8,
            "confirmation": 8,
        })

    def test_no_layer_or_expert_overlap(self) -> None:
        sets = sw.split_sets()
        names = list(sets)
        for index, left_name in enumerate(names):
            left = sets[left_name]
            for right_name in names[index + 1 :]:
                right = sets[right_name]
                self.assertFalse({row[0] for row in left} & {row[0] for row in right})
                self.assertFalse({row[1] for row in left} & {row[1] for row in right})

    def test_path_firewall_precedes_existence(self) -> None:
        with self.assertRaises(ValueError):
            sw.reject_forbidden_path(Path("x") / "blind_protocol_v2" / "unblinded")

    def test_coordinate_builder_has_no_source_argument(self) -> None:
        names = set(inspect.signature(sw.coordinate_features).parameters)
        self.assertFalse(names & {"x", "source", "tile", "original"})
        features = sw.coordinate_features(
            17, 53, "gate", np.array([0, 47]), np.array([0, 127]),
            0.025, -3.7, 0.8,
        )
        self.assertEqual(features.shape, (2, 21))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_runner_launch_and_access_order_are_fail_closed(self) -> None:
        self.assertNotIn("cupy", sys.modules)
        run_parameters = inspect.signature(gate.run_gate).parameters
        self.assertIn("launch_sentinel", run_parameters)
        self.assertIs(run_parameters["launch_sentinel"].default, inspect.Parameter.empty)
        decoder_parameters = inspect.signature(gate.decode_bf16_matrix).parameters
        self.assertIn("payload", decoder_parameters)
        self.assertNotIn("path", decoder_parameters)
        source = inspect.getsource(gate.run_gate)
        fit_auth = source.index("fit_payloads = authenticate_record_payloads")
        calibration_auth = source.index(
            "calibration_payloads = authenticate_record_payloads"
        )
        first_decode = source.index("fit_records = decode_authenticated_records")
        self.assertLess(fit_auth, calibration_auth)
        self.assertLess(calibration_auth, first_decode)
        gpu_source = inspect.getsource(gate.gpu_source_free_preflight)
        self.assertNotIn("inventory_auxiliary", gpu_source)
        self.assertNotIn("load_records", gpu_source)

    def test_closed_source_lock_and_exact_split_membership(self) -> None:
        protocol = sw.load_protocol()
        lock = sw.load_source_lock(protocol=protocol)
        sw.validate_source_lock(protocol, lock)
        self.assertEqual(sw.sha256_file(sw.SOURCE_LOCK_PATH), protocol[
            "source_authentication"
        ]["source_lock_sha256"])
        self.assertEqual(len(lock["files"]), 116)
        self.assertEqual(lock["split_counts"], {
            "fit": 82, "calibration": 16, "confirmation": 18,
        })
        tampered = copy.deepcopy(lock)
        tampered["files"][0]["split"] = "confirmation"
        with self.assertRaises(ValueError):
            sw.validate_source_lock(protocol, tampered)

    def test_inventory_uses_closed_regular_exact_identities(self) -> None:
        key = (1, 2, "up")
        filename = "model.layers.1.mlp.experts.2.up_proj.weight.bf16.bin"
        payload = b"abcd"
        row = {
            "filename": filename,
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "split": "fit",
        }
        expected = {"fit": {key}, "calibration": set(), "confirmation": set()}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / filename
            path.write_bytes(payload)
            with (
                mock.patch.object(sw, "source_lock_rows", return_value={filename: row}),
                mock.patch.object(sw, "expected_split_keys", return_value=expected),
                mock.patch.object(sw, "reject_forbidden_path", return_value=directory),
            ):
                inventory = sw.inventory_auxiliary(directory, sw.load_protocol())
                self.assertEqual(inventory["fit"], {key: path})
                path.write_bytes(payload + b"x")
                with self.assertRaises(ValueError):
                    sw.inventory_auxiliary(directory, sw.load_protocol())

    def test_single_descriptor_authentication_and_symlink_rejection(self) -> None:
        payload = b"authenticated bytes"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "source.bin"
            path.write_bytes(payload)
            row = {
                "filename": path.name,
                "nbytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            decoded, digest = sw.read_authenticated_locked_file(path, row)
            self.assertEqual(decoded, payload)
            self.assertEqual(digest, row["sha256"])
            path.write_bytes(b"X" * len(payload))
            with self.assertRaises(ValueError):
                sw.read_authenticated_locked_file(path, row)
            target = directory / "target.bin"
            target.write_bytes(payload)
            link = directory / "link.bin"
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError):
                return
            link_row = dict(row, filename=link.name)
            with self.assertRaises(ValueError):
                sw.read_authenticated_locked_file(link, link_row)

    def test_launch_sentinel_binding_and_tamper(self) -> None:
        runner = Path(gate.__file__).resolve()
        common = Path(sw.__file__).resolve()
        sentinel = sw.build_launch_sentinel(runner, common)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "launch.json"
            path.write_text(
                json.dumps(sentinel, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            self.assertEqual(sw.validate_launch_sentinel(path, runner, common), sentinel)
            tampered = dict(sentinel)
            tampered["authorization_phrase"] = "UNAUTHORIZED"
            tampered.pop("internal_seal_sha256")
            tampered["internal_seal_sha256"] = sw.sha256_bytes(
                sw.canonical_json_bytes(tampered)
            )
            path.write_text(
                json.dumps(tampered, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                sw.validate_launch_sentinel(path, runner, common)


class InformationAndMomentTests(unittest.TestCase):
    def test_analytic_information_bound(self) -> None:
        moments = sw.channel_second_moments()
        self.assertEqual(float(sw.CHANNEL_A_FP32), 0.9492341876029968)
        self.assertEqual(float(sw.CHANNEL_SIGMA_FP32), 0.21951906383037567)
        self.assertAlmostEqual(
            moments["information_upper_bound_bpw"], 2.149999824926515, places=14
        )
        self.assertAlmostEqual(
            moments["error_variance"], 0.05076578709329227, places=15
        )
        self.assertLessEqual(moments["information_upper_bound_bpw"], 2.15)
        self.assertLessEqual(
            moments["information_upper_bound_bpw"], sw.ROLE_PAYLOAD_BPW
        )

    def test_postcast_rms_counterexample_advances_an_ulp(self) -> None:
        source = np.asarray([2.46875, 1.515625, 3.453125], dtype=np.float32)
        moments = sw.upward_fp16_moments(source)
        self.assertEqual(moments["centered_rms_fp64"], 0.791015625)
        self.assertEqual(float(moments["serialized_rms_fp16"]), 0.79150390625)
        self.assertEqual(moments["precast_normalized_second_moment_fp64"], 1.0)
        normalized = sw.normalize_with_serialized_moments(source, moments)
        actual = float(np.mean(normalized.astype(np.float64) ** 2))
        self.assertEqual(actual, moments["normalized_second_moment_fp64"])
        self.assertLessEqual(actual, 1.0)

    def test_both_controls_preserve_safe_serialized_rms_at_boundary(self) -> None:
        source = np.asarray([2.46875, 1.515625, 3.453125], dtype=np.float32)
        moments = sw.upward_fp16_moments(source)
        for control_name in sw.CONTROL_NAMES:
            tiles, energy = gate.moment_matched_control(
                (1, 2, "up"), control_name, moments
            )
            self.assertTrue(math.isfinite(energy))
            actual = float(np.mean(tiles.astype(np.float64) ** 2))
            self.assertLessEqual(actual, 1.0)

    def test_proof_aligned_evaluation_casts_once(self) -> None:
        source = np.asarray([[0.25, -0.75]], dtype=np.float32)
        noise = np.asarray([[0.125, -1.25]], dtype=np.float64)
        observed = sw.ideal_awgn_mc_channel(source, noise)
        a, sigma = sw.implemented_channel_constants()
        expected = (a * source.astype(np.float64) + sigma * noise).astype(np.float32)
        self.assertTrue(np.array_equal(observed, expected))

    def test_rms_rounds_up_not_nearest(self) -> None:
        # Nearest FP16 maps 1.0001 down to 1.0; the frozen algorithm must not.
        source = np.array([-1.0001, 1.0001], dtype=np.float64)
        moments = sw.upward_fp16_moments(source)
        self.assertEqual(float(np.float16(1.0001)), 1.0)
        self.assertGreaterEqual(
            float(moments["serialized_rms_fp16"]), moments["centered_rms_fp64"]
        )
        self.assertGreater(float(moments["serialized_rms_fp16"]), 1.0)
        normalized = sw.normalize_with_serialized_moments(source, moments)
        self.assertLessEqual(float(np.mean(normalized.astype(np.float64) ** 2)), 1.0)

    def test_mean_is_serialized_before_rms(self) -> None:
        source = np.array([0.10001, 0.10003, 0.10007], dtype=np.float64)
        moments = sw.upward_fp16_moments(source)
        mean16 = float(moments["serialized_mean_fp16"])
        expected = math.sqrt(float(np.mean((source - mean16) ** 2)))
        self.assertEqual(moments["centered_rms_fp64"], expected)
        self.assertLessEqual(moments["normalized_second_moment_fp64"], 1.0)

    def test_zero_rms_uses_minimum_positive_subnormal(self) -> None:
        moments = sw.upward_fp16_moments(np.full(8, 3.0, dtype=np.float32))
        expected = np.nextafter(np.float16(0), np.float16(np.inf), dtype=np.float16)
        self.assertEqual(moments["serialized_rms_fp16"], expected)
        normalized = sw.normalize_with_serialized_moments(
            np.full(8, 3.0, dtype=np.float32), moments
        )
        self.assertTrue(np.array_equal(normalized, np.zeros_like(normalized)))

    def test_nonfinite_and_overflow_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            sw.upward_fp16_moments(np.array([0.0, np.nan]))
        with self.assertRaises(OverflowError):
            sw.upward_fp16_moments(np.array([-1e9, 1e9]))


class LedgerAndSerializationTests(unittest.TestCase):
    def test_exact_ledgers(self) -> None:
        production = sw.production_ledger(128)
        self.assertEqual(sw.PARAMETER_COUNT, 235779)
        self.assertEqual(sw.MODEL_PARAMETER_BYTES, 471558)
        self.assertEqual(sw.MODEL_TOTAL_BYTES, 475654)
        self.assertEqual(sw.ROLE_PAYLOAD_BYTES, 422708)
        self.assertEqual(production["production_cold_expert_bytes"], 1743854)
        self.assertAlmostEqual(production["production_physical_bpw"], 2.15643318494161)
        self.assertAlmostEqual(production["production_cold_read_amplification"], 1.37104489269124)
        self.assertLess(production["production_cold_read_amplification"], 2.0)
        self.assertGreaterEqual(production["production_physical_bpw"], 2.15)
        self.assertLessEqual(production["production_physical_bpw"], 2.5)
        six = sw.production_ledger(6)
        self.assertAlmostEqual(six["production_physical_bpw"], 2.28453855161314)

    def test_fp16_model_roundtrip_and_tamper(self) -> None:
        params = sw.initialize_parameters(sw.TRAINING_SEEDS[0])
        params["bo"][3] = np.float32(0.1234567)
        blob = sw.serialize_model_bytes(params, sw.TRAINING_SEEDS[0], -3.5, 0.75)
        self.assertEqual(len(blob), sw.MODEL_TOTAL_BYTES)
        rounded, header = sw.deserialize_model_bytes(blob)
        self.assertEqual(header["parameter_payload_sha256"], sw.sha256_bytes(blob[4096:]))
        self.assertEqual(header["source_lock_sha256"], sw.source_lock_sha256())
        self.assertEqual(rounded["bo"][3], np.float32(np.float16(0.1234567)))
        corrupt = bytearray(blob)
        corrupt[-1] ^= 1
        with self.assertRaises(ValueError):
            sw.deserialize_model_bytes(bytes(corrupt))

    def test_all_serialized_numeric_payloads_fail_nonfinite(self) -> None:
        params = sw.initialize_parameters(sw.TRAINING_SEEDS[0])
        params["bo"][0] = np.nan
        with self.assertRaises(FloatingPointError):
            sw.serialize_model_bytes(params, sw.TRAINING_SEEDS[0], -3.5, 0.75)
        params = sw.initialize_parameters(sw.TRAINING_SEEDS[0])
        params["bo"][0] = np.float32(1e10)
        with self.assertRaises(FloatingPointError):
            sw.serialize_model_bytes(params, sw.TRAINING_SEEDS[0], -3.5, 0.75)
        with self.assertRaises(FloatingPointError):
            sw.pack_moments(
                {role: math.inf for role in sw.ROLE_ORDER},
                {role: 1.0 for role in sw.ROLE_ORDER},
            )
        with self.assertRaises(ValueError):
            sw.canonical_json_bytes({"bad": math.nan})

    def test_expert_header_moments_and_identity_flags(self) -> None:
        params = sw.initialize_parameters(sw.TRAINING_SEEDS[0])
        model = sw.serialize_model_bytes(params, sw.TRAINING_SEEDS[0], -3.5, 0.75)
        model_sha = sw.sha256_bytes(model)
        header = sw.pack_expert_header(17, 53, ["gate", "down"], model_sha)
        parsed = sw.parse_expert_header(header, model_sha)
        self.assertEqual(parsed["bypass_roles"], ["gate", "down"])
        moments = sw.pack_moments(
            {role: 0.001 * (index + 1) for index, role in enumerate(sw.ROLE_ORDER)},
            {role: 0.02 * (index + 1) for index, role in enumerate(sw.ROLE_ORDER)},
        )
        self.assertEqual(len(header) + len(moments), sw.EXPERT_LOCAL_BYTES)
        decoded = sw.unpack_moments(moments)
        self.assertEqual(set(decoded), {"mean", "rms"})


class ModelTests(unittest.TestCase):
    def test_exact_identity_after_serialized_normalization(self) -> None:
        rng = np.random.default_rng(9)
        source = rng.standard_normal((3, 256)).astype(np.float32)
        moments = sw.upward_fp16_moments(source)
        normalized = sw.normalize_with_serialized_moments(source, moments)
        noise = rng.standard_normal(normalized.shape).astype(np.float32)
        y = sw.gaussian_rdf_channel(normalized, noise)
        params = sw.initialize_parameters(sw.TRAINING_SEEDS[0])
        model = sw.serialize_model_bytes(params, sw.TRAINING_SEEDS[0], -3.5, 0.75)
        rounded, _ = sw.deserialize_model_bytes(model)
        features = np.zeros((3, 21), dtype=np.float32)
        roles = np.array([0, 1, 2], dtype=np.int64)
        decoded = sw.forward(rounded, y, features, roles)
        self.assertTrue(np.array_equal(decoded, y))
        identity_sse = float(np.sum((y.astype(np.float64) - normalized) ** 2))
        learned_sse = float(np.sum((decoded.astype(np.float64) - normalized) ** 2))
        self.assertEqual(identity_sse, learned_sse)

    def test_gradient_against_finite_difference(self) -> None:
        spec = sw.SmallSpec(values=4, features=3, hidden=5, bottleneck=3, steps=2)
        rng = np.random.default_rng(12)
        params = sw.initialize_parameters(17, spec=spec)
        params["Wo"][:] = rng.standard_normal(params["Wo"].shape).astype(np.float32) * 0.03
        y = rng.standard_normal((3, 4)).astype(np.float32)
        target = rng.standard_normal((3, 4)).astype(np.float32)
        features = rng.standard_normal((3, 3)).astype(np.float32)
        roles = np.array([0, 1, 2], dtype=np.int64)
        _, gradients = sw.mse_loss_and_gradients(
            params, y, target, features, roles, spec=spec
        )
        probes = [("Wo", (1, 2)), ("A", (2, 1)), ("C", (3, 2)), ("Wy", (1, 4)), ("role_gain", (1,))]
        epsilon = 2e-3
        for name, index in probes:
            original = float(params[name][index])
            params[name][index] = original + epsilon
            plus = sw.mse_loss_and_gradients(
                params, y, target, features, roles, spec=spec
            )[0]
            params[name][index] = original - epsilon
            minus = sw.mse_loss_and_gradients(
                params, y, target, features, roles, spec=spec
            )[0]
            params[name][index] = original
            numerical = (plus - minus) / (2.0 * epsilon)
            analytic = float(gradients[name][index])
            self.assertAlmostEqual(analytic, numerical, delta=3e-3, msg=f"{name}{index}")

    def test_raw_sse_weighted_loss_value_and_gradient(self) -> None:
        spec = sw.SmallSpec(values=4, features=3, hidden=5, bottleneck=3, steps=2)
        params = sw.initialize_parameters(99, spec=spec)
        params["Wo"][:] = sw.counter_standard_normal(
            params["Wo"].shape, "weighted-gradient", 1
        ) * np.float32(0.02)
        y = sw.counter_standard_normal((3, 4), "weighted-y", 1)
        target = sw.counter_standard_normal((3, 4), "weighted-target", 1)
        features = sw.counter_standard_normal((3, 3), "weighted-features", 1)
        roles = np.asarray([0, 1, 2], dtype=np.int64)
        weights = np.asarray([0.25, 2.0, 7.0], dtype=np.float32)
        decoded = sw.forward(params, y, features, roles, spec=spec)
        error = decoded - target
        expected = float(
            np.sum(
                error.astype(np.float64) ** 2
                * weights.astype(np.float64)[:, None]
            )
            / (float(np.sum(weights.astype(np.float64))) * spec.values)
        )
        loss, gradients = sw.mse_loss_and_gradients(
            params, y, target, features, roles,
            sample_weights=weights, spec=spec,
        )
        self.assertAlmostEqual(loss, expected, places=12)
        name, index = "Wo", (1, 2)
        epsilon = 2e-3
        original = float(params[name][index])
        params[name][index] = original + epsilon
        plus = sw.mse_loss_and_gradients(
            params, y, target, features, roles,
            sample_weights=weights, spec=spec,
        )[0]
        params[name][index] = original - epsilon
        minus = sw.mse_loss_and_gradients(
            params, y, target, features, roles,
            sample_weights=weights, spec=spec,
        )[0]
        params[name][index] = original
        numerical = (plus - minus) / (2.0 * epsilon)
        self.assertAlmostEqual(float(gradients[name][index]), numerical, delta=3e-3)

    def test_synthetic_structured_sensitivity_and_gaussian_null(self) -> None:
        spec = sw.SmallSpec(values=16, features=4, hidden=24, bottleneck=12, steps=3)
        rng = np.random.default_rng(314)
        count = 768
        train_count = 512
        latent = rng.standard_normal((count, 3)).astype(np.float32)
        basis = rng.standard_normal((3, 16)).astype(np.float32)
        structured = np.tanh(latent @ basis).astype(np.float32)
        structured /= np.float32(math.sqrt(float(np.mean(structured.astype(np.float64) ** 2))))
        gaussian = rng.standard_normal((count, 16)).astype(np.float32)
        gaussian /= np.float32(math.sqrt(float(np.mean(gaussian.astype(np.float64) ** 2))))
        noise = rng.standard_normal((count, 16)).astype(np.float32)
        y_structured = sw.gaussian_rdf_channel(structured, noise)
        y_gaussian = sw.gaussian_rdf_channel(gaussian, noise)
        features = np.zeros((count, 4), dtype=np.float32)
        features[:, 0] = np.linspace(-1.0, 1.0, count, dtype=np.float32)
        roles = np.arange(count, dtype=np.int64) % 3
        results = []
        for source, y in ((structured, y_structured), (gaussian, y_gaussian)):
            params = sw.initialize_parameters(2718, spec=spec)
            optimizer = sw.Adam(params, learning_rate=3e-3)
            for _ in range(320):
                _, grads = sw.mse_loss_and_gradients(
                    params,
                    y[:train_count],
                    source[:train_count],
                    features[:train_count],
                    roles[:train_count],
                    spec=spec,
                )
                optimizer.update(params, grads)
            decoded = sw.forward(
                params,
                y[train_count:],
                features[train_count:],
                roles[train_count:],
                spec=spec,
            )
            identity = float(
                np.sum((y[train_count:].astype(np.float64) - source[train_count:]) ** 2)
            )
            learned = float(
                np.sum((decoded.astype(np.float64) - source[train_count:]) ** 2)
            )
            results.append(-0.5 * math.log2(learned / identity))
        self.assertGreater(results[0], 0.18)
        self.assertLess(results[1], 0.08)
        self.assertGreater(results[0] - results[1], 0.16)

    def test_two_independent_nulls_and_fixed_training_seeds(self) -> None:
        self.assertEqual(sw.TRAINING_SEEDS, (26090131, 26090179))
        seeds = {
            sw.derive_seed(control, "matrix", 17, 53, "up")
            for control in sw.CONTROL_NAMES
        }
        self.assertEqual(len(seeds), 2)

    def test_counter_randomness_replay_pairing_and_domain_separation(self) -> None:
        first = gate.training_batch_indices(997, sw.TRAINING_SEEDS[0], 513)
        replay = gate.training_batch_indices(997, sw.TRAINING_SEEDS[0], 513)
        self.assertTrue(np.array_equal(first, replay))
        # All three corpora consume this one shared batch object; corpus is not
        # an argument to the frozen batch-index function.
        self.assertNotIn("corpus", inspect.signature(gate.training_batch_indices).parameters)
        null_a = gate.training_channel_noise(
            (8, 16), sw.TRAINING_SEEDS[0], "null_a", 513
        )
        null_a_replay = gate.training_channel_noise(
            (8, 16), sw.TRAINING_SEEDS[0], "null_a", 513
        )
        null_b = gate.training_channel_noise(
            (8, 16), sw.TRAINING_SEEDS[0], "null_b", 513
        )
        self.assertTrue(np.array_equal(null_a, null_a_replay))
        self.assertFalse(np.array_equal(null_a, null_b))
        eval64 = sw.counter_standard_normal(
            (8, 16), "evaluation-test", 513, float64=True
        )
        self.assertEqual(eval64.dtype, np.float64)
        self.assertNotIn("default_rng", inspect.getsource(gate))
        self.assertNotIn("RandomState", inspect.getsource(gate))

    def test_forward_loss_gradient_and_decisions_abort_on_nan(self) -> None:
        spec = sw.SmallSpec(values=4, features=3, hidden=5, bottleneck=3, steps=2)
        params = sw.initialize_parameters(1, spec=spec)
        y = np.zeros((2, 4), dtype=np.float32)
        features = np.zeros((2, 3), dtype=np.float32)
        roles = np.zeros(2, dtype=np.int64)
        y[0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            sw.forward(params, y, features, roles, spec=spec)
        y.fill(0.0)
        target = np.zeros_like(y)
        target[0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            sw.mse_loss_and_gradients(
                params, y, target, features, roles, spec=spec
            )
        with self.assertRaises(FloatingPointError):
            sw.relative_metrics(1.0, 1.0, math.nan, 2.2)

        def evaluation(f_value, matched):
            return {
                "corpora": {"qwen": {"aggregate": {"F_at_physical_rate": f_value}}},
                "matched": {
                    "s_match_worst": matched,
                    "cluster_se": 0.01,
                    "qwen_group_s": {"layer": {"1": 0.1}, "pair": {"1,2": 0.1}},
                },
            }

        evaluations = {
            sw.TRAINING_SEEDS[0]: evaluation(0.7, math.nan),
            sw.TRAINING_SEEDS[1]: evaluation(0.7, 0.2),
        }
        with self.assertRaises(FloatingPointError):
            gate.calibration_promotes(evaluations)

    def test_exact_early_stop_boundary(self) -> None:
        qualifying = {
            seed: {
                256: {"s_match_worst": 0.070, "cluster_se": 0.005},
                512: {"s_match_worst": 0.079, "cluster_se": 0.005},
            }
            for seed in sw.TRAINING_SEEDS
        }
        self.assertTrue(sw.hard_kill_at_512(qualifying))
        qualifying[sw.TRAINING_SEEDS[1]][512]["s_match_worst"] = 0.091
        self.assertFalse(sw.hard_kill_at_512(qualifying))


class CheckpointTests(unittest.TestCase):
    spec = sw.SmallSpec(values=4, features=3, hidden=5, bottleneck=3, steps=2)
    center = -3.5
    scale = 0.75
    bindings = {
        "protocol_sha256": "a" * 64,
        "source_lock_sha256": "b" * 64,
        "runner_sha256": "c" * 64,
        "common_sha256": "d" * 64,
        "launch_sentinel_sha256": "e" * 64,
        "runtime_identity": {
            "numpy": np.__version__, "cupy": "fake", "device_id": 0,
        },
    }

    def make_states(self, step: int = 256):
        states = {}
        optimizers = {}
        for seed in sw.TRAINING_SEEDS:
            states[seed] = {}
            optimizers[seed] = {}
            for corpus in ("qwen", *sw.CONTROL_NAMES):
                params = sw.initialize_parameters(seed, spec=self.spec)
                states[seed][corpus] = params
                optimizer = sw.Adam(params, learning_rate=5e-4)
                optimizer.step = step
                optimizers[seed][corpus] = optimizer
        return states, optimizers

    @staticmethod
    def history_through(update: int):
        points = [point for point in sw.CHECKPOINTS if point <= update]
        return {
            seed: {
                point: {
                    "matched": {"s_match_worst": 0.1, "cluster_se": 0.01},
                    "corpora": {
                        "qwen": {"aggregate": {"F_at_physical_rate": 0.9}}
                    },
                }
                for point in points
            }
            for seed in sw.TRAINING_SEEDS
        }

    def test_append_only_checkpoint_roundtrip_and_state_tamper(self) -> None:
        states, optimizers = self.make_states()
        history = self.history_through(256)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint = gate.save_training_checkpoint(
                output, 256, states, optimizers, history, self.bindings,
                self.center, self.scale, FakeCP,
            )
            self.assertTrue((checkpoint / "checkpoint.json").is_file())
            restored_states, restored_optimizers = self.make_states()
            restored_states[sw.TRAINING_SEEDS[0]]["qwen"]["Wy"].fill(123.0)
            update, restored_history = gate.restore_latest_checkpoint(
                output, restored_states, restored_optimizers, self.bindings,
                self.center, self.scale, FakeCP,
            )
            self.assertEqual(update, 256)
            self.assertEqual(set(restored_history), set(sw.TRAINING_SEEDS))
            for seed in sw.TRAINING_SEEDS:
                for corpus in ("qwen", *sw.CONTROL_NAMES):
                    for name in sw.PARAMETER_ORDER:
                        self.assertTrue(np.array_equal(
                            restored_states[seed][corpus][name],
                            states[seed][corpus][name],
                        ))
                        self.assertTrue(np.array_equal(
                            restored_optimizers[seed][corpus].m[name],
                            optimizers[seed][corpus].m[name],
                        ))
                        self.assertTrue(np.array_equal(
                            restored_optimizers[seed][corpus].v[name],
                            optimizers[seed][corpus].v[name],
                        ))
            wrong_runtime = copy.deepcopy(self.bindings)
            wrong_runtime["runtime_identity"]["cupy"] = "different"
            with self.assertRaises(ValueError):
                gate.restore_latest_checkpoint(
                    output, restored_states, restored_optimizers, wrong_runtime,
                    self.center, self.scale, FakeCP,
                )
            with self.assertRaises((ValueError, FileExistsError)):
                gate.save_training_checkpoint(
                    output, 256, states, optimizers, history, self.bindings,
                    self.center, self.scale, FakeCP,
                )
            with (checkpoint / "state.npz").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(ValueError):
                gate.restore_latest_checkpoint(
                    output, restored_states, restored_optimizers, self.bindings,
                    self.center, self.scale, FakeCP,
                )

    def test_checkpoint_nonfinite_incomplete_and_chain_fail_closed(self) -> None:
        states, optimizers = self.make_states()
        states[sw.TRAINING_SEEDS[0]]["qwen"]["bo"][0] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaises(FloatingPointError):
                gate.save_training_checkpoint(
                    output, 256, states, optimizers, self.history_through(256),
                    self.bindings, self.center, self.scale, FakeCP,
                )
            clean_states, clean_optimizers = self.make_states()
            with self.assertRaises(ValueError):
                gate.restore_latest_checkpoint(
                    output, clean_states, clean_optimizers, self.bindings,
                    self.center, self.scale, FakeCP,
                )

        states, optimizers = self.make_states()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            gate.save_training_checkpoint(
                output, 256, states, optimizers, self.history_through(256),
                self.bindings, self.center, self.scale, FakeCP,
            )
            for seed in sw.TRAINING_SEEDS:
                for corpus in ("qwen", *sw.CONTROL_NAMES):
                    optimizers[seed][corpus].step = 512
            gate.save_training_checkpoint(
                output, 512, states, optimizers, self.history_through(512),
                self.bindings, self.center, self.scale, FakeCP,
            )
            (output / "checkpoint_000256").rename(output / "removed_000256")
            with self.assertRaises(ValueError):
                gate.restore_latest_checkpoint(
                    output, states, optimizers, self.bindings,
                    self.center, self.scale, FakeCP,
                )

    def test_counter_replay_gives_exact_post_resume_update(self) -> None:
        continuous_states, continuous_optimizers = self.make_states()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            gate.save_training_checkpoint(
                output, 256, continuous_states, continuous_optimizers,
                self.history_through(256), self.bindings,
                self.center, self.scale, FakeCP,
            )
            resumed_states, resumed_optimizers = self.make_states()
            gate.restore_latest_checkpoint(
                output, resumed_states, resumed_optimizers, self.bindings,
                self.center, self.scale, FakeCP,
            )
            source_all = sw.counter_standard_normal(
                (37, self.spec.values), "resume-source", 1
            )
            feature_all = sw.counter_standard_normal(
                (37, self.spec.features), "resume-features", 1
            )
            role_all = np.arange(37, dtype=np.int64) % self.spec.roles
            indices = gate.training_batch_indices(
                len(source_all), sw.TRAINING_SEEDS[0], 257
            )
            source = source_all[indices]
            features = feature_all[indices]
            roles = role_all[indices]
            noise = gate.training_channel_noise(
                source.shape, sw.TRAINING_SEEDS[0], "qwen", 257
            )
            y = sw.gaussian_rdf_channel(source, noise)
            for states, optimizers in (
                (continuous_states, continuous_optimizers),
                (resumed_states, resumed_optimizers),
            ):
                params = states[sw.TRAINING_SEEDS[0]]["qwen"]
                optimizer = optimizers[sw.TRAINING_SEEDS[0]]["qwen"]
                _, gradients = sw.mse_loss_and_gradients(
                    params, y, source, features, roles, spec=self.spec
                )
                optimizer.update(params, gradients)
            left = continuous_states[sw.TRAINING_SEEDS[0]]["qwen"]
            right = resumed_states[sw.TRAINING_SEEDS[0]]["qwen"]
            for name in sw.PARAMETER_ORDER:
                self.assertTrue(np.array_equal(left[name], right[name]), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
