#!/usr/bin/env python3
"""Hostile source-only tests for the graph/Krylov oracle v0."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path


class TestFailure(RuntimeError):
    pass


def check(condition: bool, name: str) -> None:
    if not condition:
        raise TestFailure(name)


def load_sibling(name: str, filename: str):
    path = Path(__file__).resolve().parent / filename
    specification = importlib.util.spec_from_file_location(name, path)
    check(specification is not None and specification.loader is not None,
          f"load spec {filename}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def metric_rows(core, *, top_relative: float, water_relative: float,
                graph_remaining: float, public_remaining: float,
                input_sse: float = 100.0):
    result = {}
    for family in core.ALL_FAMILIES:
        remaining = public_remaining if family == core.PUBLIC_FAMILY else graph_remaining
        result[family] = {
            "free_support_top384_final_relative_mse":
                top_relative if family != core.PUBLIC_FAMILY else top_relative + 0.005,
            "ideal_384bit_waterfill_final_relative_mse":
                water_relative if family != core.PUBLIC_FAMILY else water_relative + 0.005,
            "ideal_384bit_gaussian_waterfill_remaining_sse_fp64": remaining,
            "input_sse_fp64": input_sse,
            "rank_curve": {
                "fixed_capture_fraction": [0.001 * (index + 1) for index in range(384)],
                "free_support_topk_capture_fraction":
                    [0.0012 * (index + 1) for index in range(384)],
            },
        }
    return result


def main() -> None:
    before = set(sys.modules)
    core = load_sibling("tcgk_test_core", "oracle_core.py")
    runner = load_sibling("tcgk_test_runner", "run_oracle.py")
    secondary = load_sibling("tcgk_test_secondary", "secondary_hooks.py")
    tests = []

    check(not any(name == "cupy" or name.startswith("cupy.") for name in sys.modules),
          "source import initializes CuPy")
    tests.append("stdlib_import_is_inert")

    budget = core.exact_budget_record()
    check(budget == {
        "coarse_bpw_exact": "307/128", "cap_bpw_exact": "5/2",
        "remaining_bpw_exact": "13/128", "remaining_bits_per_4096": 416,
        "oracle_fine_bits_per_4096": 384,
        "reserved_noncoefficient_bits_per_4096": 32,
    }, "exact DH384 budget")
    tests.append("exact_rational_budget")

    expected_capture = 1.0 - 0.025 / 0.030902167403153148
    check(abs(core.required_capture(0.030902167403153148) - expected_capture) < 1e-15,
          "required capture identity")
    check(core.required_capture(0.02) == 0.0, "already-under-target capture")
    tests.append("required_capture_identity")

    reference = core.waterfill_reference([4.0, 1.0], 1.0)
    check(abs(reference["distortion"] - 2.0) < 1e-12 and
          abs(reference["bits"] - 1.0) < 1e-12,
          "two-mode waterfill")
    zero = core.waterfill_reference([0.0, 0.0], 384.0)
    check(zero == {"distortion": 0.0, "bits": 0.0, "theta": 0.0},
          "zero waterfill")
    tests.append("reverse_waterfill_reference")

    killed = core.source_gate(metric_rows(
        core, top_relative=0.026, water_relative=0.030,
        graph_remaining=60.0, public_remaining=70.0))
    check(killed["status"] ==
          "HARD_KILL_CONTINUOUS_GRAPH_ENVELOPE_MISSES_TARGET" and
          killed["controls_may_open"] is False,
          "continuous hard kill")
    tests.append("continuous_hard_kill_closes_controls")

    at_risk = core.source_gate(metric_rows(
        core, top_relative=0.024, water_relative=0.026,
        graph_remaining=60.0, public_remaining=70.0))
    check(at_risk["controls_may_open"] is True and
          at_risk["ideal_waterfill_target_pass"] is False,
          "continuous survivor waterfill risk")
    tests.append("continuous_survivor_opens_controls")

    survivor_metrics = metric_rows(
        core, top_relative=0.020, water_relative=0.024,
        graph_remaining=40.0, public_remaining=55.0)
    control_metrics = {
        "permutation": metric_rows(
            core, top_relative=0.021, water_relative=0.026,
            graph_remaining=55.0, public_remaining=60.0),
        "gaussian": metric_rows(
            core, top_relative=0.021, water_relative=0.026,
            graph_remaining=58.0, public_remaining=60.0),
    }
    promoted = core.controls_gate(
        survivor_metrics, control_metrics, "coarse_signed_path_dct")
    check(promoted["status"] == "ELIGIBLE_FOR_FINITE_GRAPH_LIFTING_BUILD" and
          promoted["promotion_uses_only_qwen_minus_matched_control_excess_gain"] is True,
          "control-subtracted promotion")
    check(len(promoted["rank_1_to_384_qwen_minus_control_excess_capture"]
              ["gaussian"]["fixed_qwen_minus_control_capture_fraction"]) == 384,
          "all-rank matched-control curve")
    tests.append("promotion_is_control_subtracted")

    adverse_controls = {
        "permutation": metric_rows(
            core, top_relative=0.021, water_relative=0.024,
            graph_remaining=40.5, public_remaining=55.0),
        "gaussian": metric_rows(
            core, top_relative=0.021, water_relative=0.024,
            graph_remaining=40.2, public_remaining=55.0),
    }
    rejected = core.controls_gate(
        survivor_metrics, adverse_controls, "coarse_signed_path_dct")
    check(rejected["status"] ==
          "HARD_KILL_COARSE_GRAPH_NOT_SOURCE_SPECIFIC_0P03_BPW",
          "matched controls reject generic gain")
    tests.append("matched_control_excess_threshold")

    check(abs(core.graph_advantage_bpw(0.5, 1.0) - 0.5) < 1e-15,
          "graph advantage definition")
    check(abs(core.rate_gain_bpw(0.25, 1.0) - 1.0) < 1e-15,
          "rate gain definition")
    tests.append("rate_equivalent_identities")

    duplicate_rejected = False
    try:
        runner.strict_json(b'{"a":1,"a":2}', "duplicate")
    except runner.RunError:
        duplicate_rejected = True
    check(duplicate_rejected, "duplicate JSON rejection")
    tests.append("strict_json_duplicate_rejection")

    source = (Path(__file__).resolve().parent / "run_oracle.py").read_text("utf-8")
    check(source.index("authenticate_v6_result(") < source.index("authenticate_inputs("),
          "result before BF16 source")
    check("compressed_frame_refetch\": False" in source and
          "compressed_expert_frame_file_read_count\": 1" in source,
          "one-pass source declaration")
    check("from cupyx.scipy.fft import dct" in source and
          source.index("authenticate_inputs(") < source.rindex("from cupyx.scipy.fft import dct"),
          "CuPy numerical import after source/result auth")
    tests.append("static_access_order_and_one_pass_boundary")

    design = json.loads((Path(__file__).resolve().parent / "design_lock.json").read_text("utf-8"))
    check(design["oracle_contract"]["finite_codec_executed"] is False and
          design["traffic"]["compressed_frame_file_reads"] == 1 and
          design["traffic"]["compressed_frame_refetches"] == 0 and
          design["oracle_contract"]["promotion_uses_raw_continuous_capture"] is False,
          "design claim boundary")
    tests.append("frozen_claim_and_traffic_boundary")

    routing = secondary.routing_record("HARD_KILL_SYNTHETIC")
    check(routing["compressed_frame_refetch_allowed"] is False and
          routing["execution_order"] == list(secondary.SCREENS) and
          routing["bispectral_volterra"][
              "requires_ramanujan_control_excess_bpw"] == 0.03,
          "secondary screen routing")
    killed_hook = secondary.containment_gate(
        input_sse=100.0, source_energy=1000.0,
        source_remaining_sse=30.0, control_remaining_sse=None,
        descriptor_bits_per_block=0)
    check(killed_hook["controls_may_open"] is False,
          "secondary source kill closes controls")
    promoted_hook = secondary.containment_gate(
        input_sse=100.0, source_energy=1000.0,
        source_remaining_sse=20.0,
        control_remaining_sse={"permutation": 30.0, "gaussian": 35.0},
        descriptor_bits_per_block=16)
    check(promoted_hook["status"] ==
          "ELIGIBLE_FOR_BOUNDED_FINITE_SCREEN_BUILD" and
          promoted_hook["remaining_refinement_bits_per_block"] == 368,
          "secondary control-subtracted hook")
    tests.append("bounded_secondary_screen_hooks")

    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        residual = np.arange(2 * 4096, dtype=np.float64).reshape(2, 4096) / 8192.0
        permuted, receipt = core.affine_permutation_control(residual, 11, np)
        check(receipt["odd_multiplier_proves_bijection_mod_4096"] is True and
              np.allclose(np.sum(permuted * permuted, axis=1),
                          np.sum(residual * residual, axis=1), rtol=0, atol=1e-12),
              "affine permutation")
        gaussian, gaussian_receipt = core.gaussian_moment_control(residual, 11, np)
        check(gaussian_receipt["maximum_fp64_mean_error"] < 1e-12 and
              np.allclose(np.mean(gaussian, axis=1), np.mean(residual, axis=1),
                          rtol=0, atol=1e-12),
              "Gaussian block moments")
        symbols = np.tile(np.arange(4096, dtype=np.int32), (2, 1))
        for family in core.ALL_FAMILIES:
            permutation = core._stable_permutation(symbols, family, np)
            check(permutation.shape == symbols.shape and
                  all(len(set(row.tolist())) == 4096 for row in permutation),
                  f"permutation family {family}")
        tests.append("numpy_synthetic_controls_and_graphs")
    else:
        tests.append("numpy_unavailable_control_test_skipped")

    if sys.platform.startswith("linux"):
        with tempfile.TemporaryDirectory(prefix="tcgk_source_test_") as temporary:
            root = Path(temporary)
            output = root / "result"
            publication = runner.publish_atomic(output, {"RESULT.json": b"{}\n"}, {
                "schema": runner.COMPLETE_SCHEMA,
                "status": "SYNTHETIC",
                "positive_claim_authority": False,
            })
            check(output.is_dir() and (output / "COMPLETE.json").is_file() and
                  publication["rename_noreplace"] is True,
                  "atomic publication")
            collision = False
            try:
                runner.publish_atomic(output, {"RESULT.json": b"{}\n"}, {
                    "schema": runner.COMPLETE_SCHEMA,
                    "status": "SYNTHETIC",
                    "positive_claim_authority": False,
                })
            except (OSError, runner.RunError):
                collision = True
            check(collision, "no-replace collision")
        tests.append("atomic_no_replace_publication")

    check(not any(name == "cupy" or name.startswith("cupy.") for name in sys.modules),
          "source tests initialize CuPy")
    tests.append("source_tests_no_cuda")
    print(json.dumps({
        "schema": "tactic-cage-graph-krylov-oracle-v0-source-test-receipt",
        "status": "PASS_SOURCE_ONLY_TESTS",
        "tests": tests,
        "test_count": len(tests),
        "cupy_imported": False,
        "payload_accessed": False,
        "result_accessed": False,
        "network_accessed": False,
        "new_modules": sorted(set(sys.modules) - before),
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
