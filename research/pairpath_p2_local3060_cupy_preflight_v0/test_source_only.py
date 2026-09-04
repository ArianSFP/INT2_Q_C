"""CPU-only and CuPy parity tests for the source-free PAIRPATH backend."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pairpath_cupy_backend as backend


REFERENCE_CORE_SHA256 = "2c99a31aef669cabbb67137061233640b013e8c50a5132ddbcc9ffec2c239034"


def load_frozen_reference():
    import hashlib

    reference = Path(__file__).resolve().parents[1] / \
        "pairpath_fl_same_layer_microcodec_v0_20260903_r2" / "pairpath_r2_core.py"
    assert reference.is_file()
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == REFERENCE_CORE_SHA256
    spec = importlib.util.spec_from_file_location("pairpath_r2_frozen_reference", reference)
    assert spec is not None and spec.loader is not None
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture(coordinates: int = 192) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0x504149525032)
    common = rng.normal(0.0, 0.35, coordinates)
    values = np.empty((2, 2, coordinates), dtype=np.float64)
    # Roles deliberately have very different energies.  The same global
    # multiplier must nevertheless be supplied to both solvers.
    values[0, 0] = 0.28 * common + rng.normal(0.0, 0.08, coordinates)
    values[1, 0] = 0.28 * common + rng.normal(0.0, 0.08, coordinates)
    values[0, 1] = 3.5 * common + rng.normal(0.0, 0.7, coordinates)
    values[1, 1] = -3.5 * common + rng.normal(0.0, 0.7, coordinates)
    scale = np.sqrt(np.mean(values * values, axis=2, dtype=np.float64))
    levels = scale[:, :, None, None] * backend.LEVELS_RMS[None, None, None, :]
    levels = np.broadcast_to(levels, values.shape + (backend.ALPHABET,)).copy()
    return values, levels


def test_symmetric_multistarts() -> None:
    values, levels = fixture(37)
    starts = backend.symmetric_initializations(values[:, 0], levels[:, 0])
    assert len(starts) <= 18 and len(starts) >= 16
    assert len({row.tobytes() for row in starts}) == len(starts)
    expected_nearest = backend.nearest_labels(values[:, 0], levels[:, 0])
    assert np.array_equal(starts[0], expected_nearest)
    for row in starts:
        assert row.dtype == np.uint8 and row.shape == (2, 37)
        assert np.all(row < backend.ALPHABET)


def test_role_conditioned_mi() -> None:
    # Up is exactly aligned; Down cycles independently.  Role pooling is not
    # allowed to fabricate dependence from their different marginals.
    coordinates = 64
    labels = np.empty((2, 2, coordinates), dtype=np.uint8)
    labels[0, 0] = np.arange(coordinates, dtype=np.uint8) % 4
    labels[1, 0] = labels[0, 0]
    labels[0, 1] = np.arange(coordinates, dtype=np.uint8) % 2
    labels[1, 1] = (np.arange(coordinates, dtype=np.uint8) // 2) % 2 + 2
    levels = np.broadcast_to(np.arange(4, dtype=np.float64),
                             (2, 2, coordinates, 4)).copy()
    values = np.take_along_axis(levels, labels[:, :, :, None], axis=3)[:, :, :, 0]
    result = backend.fixed_assignment_mi(values.astype(np.float64), levels)
    assert result["conditioning"] == "decoder-visible role"
    assert abs(result["role_rows"][0]["mutual_information_bits_per_coordinate_pair"] - 2.0) < 1e-12
    assert abs(result["role_rows"][1]["mutual_information_bits_per_coordinate_pair"]) < 1e-12
    assert abs(result["mutual_information_bits_per_coordinate_pair"] - 1.0) < 1e-12


def test_global_updown_bit_weight() -> None:
    values, levels = fixture(53)
    lagrange = Fraction(1, 64)
    result = backend.oracle_rd_points(values, levels, (lagrange,))
    expected = float(lagrange) * float(np.sum(values * values)) / values.size
    row = result["rows"][0]
    assert row["bit_weight"] == expected
    assert row["roles"][0]["bit_weight"] == expected
    assert row["roles"][1]["bit_weight"] == expected
    local0 = float(lagrange) * float(np.sum(values[:, 0] ** 2)) / values[:, 0].size
    local1 = float(lagrange) * float(np.sum(values[:, 1] ** 2)) / values[:, 1].size
    assert local0 != expected and local1 != expected


def test_cpu_tie_order() -> None:
    coordinates = 41
    values = np.zeros((2, coordinates), dtype=np.float64)
    levels = np.zeros((2, coordinates, 4), dtype=np.float64)
    labels = np.tile(np.arange(coordinates, dtype=np.uint8) % 4, (2, 1))
    independent, _ = backend.update_labels_cpu(values, levels, labels, 0.0, False)
    joined, _ = backend.update_labels_cpu(values, levels, labels, 0.0, True)
    assert np.array_equal(independent, np.zeros_like(independent))
    assert np.array_equal(joined, np.zeros_like(joined))


def test_frozen_reference_solver_semantics() -> dict:
    """Compare directly with the source-closed PAIRPATH-P2 r2 CPU solver."""
    module = load_frozen_reference()
    values, levels = fixture(113)
    rows = []
    for bit_weight in (0.0, 0.0007, 0.07):
        for joint in (False, True):
            labels_ref, sse_ref, rate_ref = module._ideal_flexible_role(
                values[:, 0], levels[:, 0], bit_weight, joint)
            result = backend.solve_role_cpu(values[:, 0], levels[:, 0], bit_weight, joint)
            assert np.array_equal(labels_ref, result["labels"])
            assert sse_ref == result["sse"] and rate_ref == result["rate_bpw"]
            rows.append({"joint": joint, "bit_weight": bit_weight,
                         "label_sha256": result["label_sha256"]})
    return {"reference_core_sha256": REFERENCE_CORE_SHA256, "rows": rows}


def test_frozen_reference_oracle_semantics() -> dict:
    """Compare the complete block-scale/RD/hull orchestration with r2."""
    module = load_frozen_reference()
    coordinates = module.FOLD_COUNT * module.BLOCK_VALUES
    rng = np.random.default_rng(0x504149524F524143)
    common = rng.normal(0.0, 0.4, coordinates)
    full = np.empty((2, 3, coordinates), dtype=np.float64)
    full[:, 0] = rng.normal(0.0, 1.0, (2, coordinates))
    full[0, 1] = 0.2 * common + rng.normal(0.0, 0.07, coordinates)
    full[1, 1] = 0.2 * common + rng.normal(0.0, 0.07, coordinates)
    full[0, 2] = 2.7 * common + rng.normal(0.0, 0.5, coordinates)
    full[1, 2] = -2.7 * common + rng.normal(0.0, 0.5, coordinates)
    lambdas = (Fraction(0, 1), Fraction(1, 64))
    reference = module.optimistic_single_letter_joint_gate(
        full, lambda_grid=(Fraction(1, 64),))
    values = np.ascontiguousarray(full[:, 1:3])
    _, levels = backend.prepare_levels(values)
    rd = backend.oracle_rd_points(values, levels, lambdas)
    candidate = backend.convexified_gate(rd)
    assert reference["independent_hull"] == candidate["independent_hull"]
    assert reference["pair_hull"] == candidate["pair_hull"]
    assert reference["equal_rate"] == candidate["equal_rate"]
    assert reference["equal_mse"] == candidate["equal_mse"]
    assert reference["best_G_eq_UD_bpw"] == candidate["best_G_eq_UD_bpw"]
    assert (reference["fixed_assignment_mi"]["mutual_information_bits_per_coordinate_pair"] ==
            rd["fixed_assignment_mi"]["mutual_information_bits_per_coordinate_pair"])
    return {"coordinates": coordinates,
            "independent_hull": candidate["independent_hull"],
            "pair_hull": candidate["pair_hull"],
            "best_G_eq_UD_bpw": candidate["best_G_eq_UD_bpw"]}


def run_cupy_tests(cp) -> dict:
    values, levels = fixture(192)
    parity_rows = []
    for bit_weight in (0.0, 0.0007, 0.07):
        for joint in (False, True):
            cpu = backend.solve_role_cpu(values[:, 0], levels[:, 0], bit_weight, joint)
            gpu = backend.solve_role_cupy(cp, values[:, 0], levels[:, 0], bit_weight,
                                          joint, chunk_coordinates=31)
            assert np.array_equal(cpu["labels"], gpu["labels"])
            assert np.array_equal(cpu["counts"], gpu["counts"])
            assert cpu["label_sha256"] == gpu["label_sha256"]
            assert cpu["sse"] == gpu["sse"]
            assert cpu["rate_bpw"] == gpu["rate_bpw"]
            parity_rows.append({"joint": joint, "bit_weight": bit_weight,
                                "label_sha256": cpu["label_sha256"],
                                "start_count": cpu["start_count"]})

    # Exact ties, including a chunk tail.  Lowest flattened joint index 0 is
    # authoritative and decodes to label pair (0,0).
    tie_n = 67
    tie_values = np.zeros((2, tie_n), dtype=np.float64)
    tie_levels = np.zeros((2, tie_n, 4), dtype=np.float64)
    tie_labels = np.tile(np.arange(tie_n, dtype=np.uint8) % 4, (2, 1))
    distortion = cp.zeros((2, tie_n, 4), dtype=cp.float64)
    for joint in (False, True):
        updated_gpu, counts_gpu = backend.update_labels_cupy_from_distortion(
            cp, distortion, cp.asarray(tie_labels), 0.0, joint, chunk_coordinates=13)
        updated_cpu, counts_cpu = backend.update_labels_cpu(
            tie_values, tie_levels, tie_labels, 0.0, joint)
        assert np.array_equal(cp.asnumpy(updated_gpu), updated_cpu)
        assert np.array_equal(counts_gpu, counts_cpu)
        assert np.array_equal(updated_cpu, np.zeros_like(updated_cpu))

    # Full RD point parity with strongly unequal role energies proves the
    # shared global multiplier survives orchestration, not only a unit call.
    lambdas = (Fraction(0, 1), Fraction(1, 64))
    cpu_rd = backend.oracle_rd_points(values, levels, lambdas)
    gpu_rd = backend.oracle_rd_points(values, levels, lambdas, cp=cp,
                                      chunk_coordinates=47)
    for cpu_row, gpu_row in zip(cpu_rd["rows"], gpu_rd["rows"]):
        assert cpu_row["bit_weight"] == gpu_row["bit_weight"]
        assert cpu_row["independent"] == gpu_row["independent"]
        assert cpu_row["joint"] == gpu_row["joint"]
        for cpu_role, gpu_role in zip(cpu_row["roles"], gpu_row["roles"]):
            assert cpu_role["bit_weight"] == gpu_role["bit_weight"]
            assert cpu_role["independent"] == gpu_role["independent"]
            assert cpu_role["joint"] == gpu_role["joint"]
    cpu_gate = backend.convexified_gate(cpu_rd)
    gpu_gate = backend.convexified_gate(gpu_rd)
    assert cpu_gate == gpu_gate
    return {"solver_parity_rows": parity_rows,
            "rd_rows": gpu_rd["rows"], "gate": gpu_gate,
            "tie_coordinates": tie_n}


def run_cpu_tests() -> dict:
    test_symmetric_multistarts()
    test_role_conditioned_mi()
    test_global_updown_bit_weight()
    test_cpu_tie_order()
    result = test_frozen_reference_solver_semantics()
    result["full_oracle"] = test_frozen_reference_oracle_semantics()
    return result


if __name__ == "__main__":
    run_cpu_tests()
    if os.environ.get("PAIRPATH_REQUIRE_CUPY") == "1":
        import cupy as cp
        result = run_cupy_tests(cp)
        assert result["solver_parity_rows"]
    print("PASS_SOURCE_FREE_PAIRPATH_CPU" +
          ("_CUPY_PARITY" if os.environ.get("PAIRPATH_REQUIRE_CUPY") == "1" else ""))
