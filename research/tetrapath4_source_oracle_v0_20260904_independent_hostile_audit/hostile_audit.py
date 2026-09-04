"""Independent hostile audit of the unsealed TETRAPATH-4 source package."""

from __future__ import annotations

from fractions import Fraction
import ast
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "tetrapath4_source_oracle_v0_20260904"
EXPECTED = {
    "CHECKPOINT_STATUS.md", "DESIGN_LOCK.json", "README.md", "RUN_DISABLED.txt",
    "test_source.py", "tetrapath4_oracle.py", "verify_source.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_target():
    path = TARGET / "tetrapath4_oracle.py"
    spec = importlib.util.spec_from_file_location("tetrapath4_hostile_target", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cross_entropy_bpw(module, labels: np.ndarray, family: str) -> float:
    probability, _ = module.fit_probability(labels, family)
    ids = module.tuple_ids(labels)
    return float(-np.log2(probability[ids]).sum() / (4 * len(ids)))


def exact_full_n2(module, tuple_costs: np.ndarray, source_energy: float,
                  multiplier: Fraction) -> tuple[float, tuple[int, int]]:
    """Exhaust all 256^2 label sequences under the target's fitted full law."""
    best = (math.inf, -1, -1)
    for a in range(256):
        for b in range(256):
            labels = module.labels_from_ids(np.asarray((a, b), dtype=np.int64))
            rate = cross_entropy_bpw(module, labels, "full")
            mse = float(tuple_costs[0, a] + tuple_costs[1, b]) / source_energy
            candidate = (mse + float(multiplier) * rate, a, b)
            if candidate < best:
                best = candidate
    return best[0], (best[1], best[2])


def find_local_search_counterexample(module) -> dict | None:
    multipliers = (Fraction(1, 8), Fraction(1, 2), Fraction(1), Fraction(2))
    for seed in range(64):
        rng = np.random.default_rng(seed)
        costs = rng.lognormal(mean=-1.0, sigma=1.1, size=(2, 4, 4)).astype(np.float64)
        tuple_costs = module.expanded_tuple_costs(costs)
        source_energy = float(1.0 + costs.sum())
        for multiplier in multipliers:
            observed = module.optimize_family(costs, source_energy, "full", multiplier)
            exact_objective, exact_ids = exact_full_n2(
                module, tuple_costs, source_energy, multiplier)
            gap = observed.objective - exact_objective
            if gap > 1e-12:
                return {
                    "seed": seed,
                    "multiplier": [multiplier.numerator, multiplier.denominator],
                    "reported_objective": observed.objective,
                    "exact_objective": exact_objective,
                    "gap": gap,
                    "reported_assignment_sha256": observed.assignment_sha256,
                    "exact_tuple_ids": list(exact_ids),
                    "costs": costs.tolist(),
                    "source_energy": source_energy,
                }
    return None


def main() -> int:
    names = {path.name for path in TARGET.iterdir() if path.is_file()}
    assert names == EXPECTED, (names, EXPECTED)
    module = load_target()
    files = {name: sha256(TARGET / name) for name in sorted(EXPECTED)}

    test_run = subprocess.run(
        [sys.executable, "-I", "-B", str(TARGET / "test_source.py")],
        cwd=TARGET, capture_output=True, text=True, check=False)
    assert test_run.returncode == 0, test_run.stdout + test_run.stderr
    verify_run = subprocess.run(
        [sys.executable, "-I", "-B", str(TARGET / "verify_source.py"),
         "--package", str(TARGET), "--self-test"],
        cwd=TARGET, capture_output=True, text=True, check=False)
    assert verify_run.returncode == 0, verify_run.stdout + verify_run.stderr

    imports = set()
    for source_name in ("tetrapath4_oracle.py", "test_source.py", "verify_source.py"):
        tree = ast.parse((TARGET / source_name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    assert not ({"socket", "requests", "torch", "cupy", "huggingface_hub"} & imports)

    rng = np.random.default_rng(20260904)
    labels = rng.integers(0, 4, size=(8192, 4), dtype=np.uint8)
    normalization = {}
    for family in module.FAMILIES:
        if family in module.FIBER_FAMILIES:
            support, _, _ = module._fiber_support(family)
            ids = rng.choice(np.flatnonzero(support), size=len(labels), replace=True)
            family_labels = module.labels_from_ids(ids)
        else:
            family_labels = labels
        probability, _ = module.fit_probability(family_labels, family)
        normalization[family] = {
            "sum": float(probability.sum()),
            "min": float(probability.min()),
            "finite": bool(np.all(np.isfinite(probability))),
        }
        assert abs(float(probability.sum()) - 1.0) < 1e-11

    pairwise = module._pairwise_maxent_probability(labels, smoothed=False)
    max_pair_margin_error = 0.0
    for a in range(4):
        for b in range(a + 1, 4):
            fitted = np.zeros((4, 4), dtype=np.float64)
            np.add.at(fitted, (module.QTABLE[:, a], module.QTABLE[:, b]), pairwise)
            target = np.bincount(module._pair_code(labels, a, b), minlength=16)
            target = target.reshape(4, 4) / len(labels)
            max_pair_margin_error = max(max_pair_margin_error,
                                        float(np.max(np.abs(fitted - target))))

    xor_rows = [(a, b, c, a ^ b ^ c)
                for a, b, c in itertools.product((0, 1), repeat=3)]
    xor_labels = np.repeat(np.asarray(xor_rows, dtype=np.uint8), 64, axis=0)
    xor_census = module.fixed_assignment_census(xor_labels)
    xor_fiber = module.fiber_fixed_ledger(xor_labels, "fiber_gray_low")
    assert abs(xor_census["fourway_gain_over_best_factorized_bpw"] - 0.25) < 1e-13
    assert abs(xor_fiber["max_logical_read_amplification"] - 4 / 3) < 1e-13

    iid_labels = np.repeat(module.QTABLE, 16, axis=0)
    iid_census = module.fixed_assignment_census(iid_labels)
    assert abs(iid_census["fourway_gain_over_best_factorized_bpw"]) < 1e-13

    singleton = np.zeros((1, 4), dtype=np.uint8)
    smoothing_counterexample = {
        "independent_fitted_rate_bpw": cross_entropy_bpw(module, singleton, "independent"),
        "full_fitted_rate_bpw": cross_entropy_bpw(module, singleton, "full"),
    }
    smoothing_counterexample["full_minus_independent_bpw"] = (
        smoothing_counterexample["full_fitted_rate_bpw"] -
        smoothing_counterexample["independent_fitted_rate_bpw"])
    assert smoothing_counterexample["full_minus_independent_bpw"] > 0

    local_counterexample = find_local_search_counterexample(module)

    report = {
        "schema": "tetrapath4.independent_hostile_audit.v0",
        "target": "research/tetrapath4_source_oracle_v0_20260904",
        "target_files_sha256": files,
        "target_tests": {
            "returncode": test_run.returncode,
            "ran_12_tests": "Ran 12 tests" in test_run.stderr,
            "status_ok": "OK" in test_run.stderr,
        },
        "target_verifier": {
            "returncode": verify_run.returncode,
            "pass_marker_present": (
                "PASS_UNSEALED_SOURCE_ONLY_TETRAPATH4_NO_PAYLOAD_AUTHORITY"
                in verify_run.stdout),
        },
        "source_only_imports": sorted(imports),
        "normalization": normalization,
        "pairwise_maxent_max_abs_margin_error": max_pair_margin_error,
        "xor_fixture": {
            "gain_bpw": xor_census["fourway_gain_over_best_factorized_bpw"],
            "pairwise_maxent_residual_bpw": xor_census["residual_connected_information_bpw"],
            "fiber_total_bpw": xor_fiber["total_bpw"],
            "fiber_max_logical_read_amplification": xor_fiber[
                "max_logical_read_amplification"],
        },
        "iid_fixture_gain_bpw": iid_census["fourway_gain_over_best_factorized_bpw"],
        "smoothing_noncontainment_counterexample": smoothing_counterexample,
        "local_search_globality_counterexample": local_counterexample,
    }
    (HERE / "AUDIT_EVIDENCE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
