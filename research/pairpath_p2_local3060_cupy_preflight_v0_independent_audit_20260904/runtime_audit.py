#!/usr/bin/env python3
"""Independent local-RTX3060 runtime audit of PAIRPATH-P2 CuPy preflight v0.

No model payload, Qwen locator, network client, RunPod path, or production
authority is present.  The frozen target package is only read; this program
does not call its receipt-writing main entry point.
"""

from __future__ import annotations

from fractions import Fraction
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys

import numpy as np


AUDIT = Path(__file__).resolve().parent
REPOSITORY = AUDIT.parent.parent
WORKSPACE = REPOSITORY.parent
TARGET = REPOSITORY / "research" / "pairpath_p2_local3060_cupy_preflight_v0"
REFERENCE = REPOSITORY / "research" / "pairpath_fl_same_layer_microcodec_v0_20260903_r2"
RECEIPT_SHA256 = "a6c1fd514ddafa5a3225a4c70b030cf80df75a41f127f829f8ccd4b92cbe53ab"
REFERENCE_CORE_SHA256 = "2c99a31aef669cabbb67137061233640b013e8c50a5132ddbcc9ffec2c239034"
REFERENCE_MANIFEST_SHA256 = "21983efff5ac5c0593a655cae4136d35ca24400fd807f9fe4be458a34b18e622"
EXPECTED_UUID_HEX = "458a424a76e365e50470803e0ed131ca"
EXPECTED_DEVICE = "NVIDIA GeForce RTX 3060"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_frozen_target() -> tuple[dict, dict[str, str]]:
    names = sorted(path.name for path in TARGET.iterdir())
    expected = sorted(("README.md", "RUN_GATE.json", "SOURCE_FREE_PREFLIGHT_RECEIPT.json",
                       "pairpath_cupy_backend.py", "run_source_free_preflight.py",
                       "test_source_only.py"))
    if names != expected:
        raise RuntimeError("target closure has missing/extra members")
    for path in TARGET.iterdir():
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError(f"nonregular target member: {path.name}")
    receipt_path = TARGET / "SOURCE_FREE_PREFLIGHT_RECEIPT.json"
    if sha256(receipt_path) != RECEIPT_SHA256:
        raise RuntimeError("receipt SHA mismatch")
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    expected_raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                               indent=2) + "\n").encode()
    if receipt_raw != expected_raw:
        raise RuntimeError("noncanonical frozen receipt")
    hashes = {name: sha256(TARGET / name) for name in expected}
    if {name: hashes[name] for name in receipt["source_files"]} != receipt["source_files"]:
        raise RuntimeError("receipt/source hash mismatch")
    if hashes["SOURCE_FREE_PREFLIGHT_RECEIPT.json"] != RECEIPT_SHA256:
        raise RuntimeError("receipt closure mismatch")
    if sha256(REFERENCE / "pairpath_r2_core.py") != REFERENCE_CORE_SHA256 or \
            sha256(REFERENCE / "SOURCE_MANIFEST.json") != REFERENCE_MANIFEST_SHA256:
        raise RuntimeError("frozen r2 reference drift")
    gate = json.loads((TARGET / "RUN_GATE.json").read_text(encoding="utf-8"))
    if gate != receipt["run_gate"] or gate["status"] != "HOLD_PRODUCTION_AND_PAYLOAD" or \
            gate["payload_authority_present"] or gate["payload_execution_enabled"] or \
            gate["qwen_or_model_payload_allowed"] or gate["remote_or_runpod_allowed"]:
        raise RuntimeError("source-free run gate")
    return receipt, hashes


def configure_cupy():
    pinned = WORKSPACE / ".venv-cupy" / "Scripts" / "python.exe"
    if Path(sys.executable).resolve() != pinned.resolve():
        raise RuntimeError("wrong Python executable")
    site = WORKSPACE / ".venv-cupy" / "Lib" / "site-packages"
    compatibility = WORKSPACE / ".tools" / "cuda_dlls_3060"
    directories = [compatibility]
    directories.extend(sorted(path for path in (site / "nvidia").glob("*\\bin")
                              if path.is_dir()))
    if not directories or not all(path.is_dir() for path in directories):
        raise RuntimeError("runtime DLL closure")
    cuda_path = site / "nvidia" / "cuda_runtime"
    cache = WORKSPACE / "tmp" / "pairpath_p2_sourcefree_cupy_cache_v0_audit"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_PATH"] = str(cuda_path)
    os.environ["CUPY_CACHE_DIR"] = str(cache)
    handles = [os.add_dll_directory(str(path)) for path in directories]
    import cupy as cp
    return cp, handles, cuda_path, cache


def device_evidence(cp, receipt: dict, cuda_path: Path, cache: Path) -> dict:
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    raw_uuid = props["uuid"]
    if not isinstance(raw_uuid, bytes) or len(raw_uuid) < 16:
        raise RuntimeError("unexpected CuPy UUID ABI")
    uuid_hex, tail = raw_uuid[:16].hex(), raw_uuid[16:].hex()
    canonical = "GPU-" + "-".join((uuid_hex[:8], uuid_hex[8:12], uuid_hex[12:16],
                                      uuid_hex[16:20], uuid_hex[20:]))
    if name != EXPECTED_DEVICE or uuid_hex != EXPECTED_UUID_HEX:
        raise RuntimeError("wrong local GPU")
    if canonical != receipt["runtime"]["device_uuid"]:
        raise RuntimeError("receipt GPU identity mismatch")
    runtime = cp.cuda.runtime.runtimeGetVersion()
    driver = cp.cuda.runtime.driverGetVersion()
    if runtime != receipt["runtime"]["cuda_runtime"] or \
            driver != receipt["runtime"]["cuda_driver_api"]:
        raise RuntimeError("CUDA version drift")
    if sha256(Path(sys.executable)) != receipt["runtime"]["python_executable_sha256"]:
        raise RuntimeError("Python executable drift")
    return {"hostname": platform.node(), "device_name": name, "device_uuid": canonical,
            "raw_uuid_length": len(raw_uuid), "uuid_trailing_hex_fresh": tail,
            "uuid_trailing_hex_receipt": receipt["runtime"]["cupy_uuid_trailing_hex"],
            "identity_uses_first_16_uuid_bytes_only": True,
            "cuda_runtime": runtime, "cuda_driver_api": driver,
            "cupy": cp.__version__, "numpy": np.__version__,
            "python_executable_sha256": sha256(Path(sys.executable)),
            "cuda_path": str(cuda_path), "audit_cache": str(cache)}


def parity_evidence(cp, receipt: dict, backend, tests) -> dict:
    cpu = tests.run_cpu_tests()
    if json_sha256(cpu) != json_sha256(receipt["frozen_reference_cpu_parity"]):
        raise RuntimeError("fresh CPU/reference parity differs from receipt")
    gpu = tests.run_cupy_tests(cp)
    if json_sha256(gpu) != json_sha256(receipt["parity"]):
        raise RuntimeError("fresh CPU/CuPy parity differs from receipt")
    # Explicitly restate the complete-oracle equality, not only label kernels.
    values, levels = tests.fixture(192)
    lambdas = (Fraction(0, 1), Fraction(1, 64))
    cpu_rd = backend.oracle_rd_points(values, levels, lambdas)
    gpu_rd = backend.oracle_rd_points(values, levels, lambdas, cp=cp,
                                       chunk_coordinates=47)
    cpu_gate = backend.convexified_gate(cpu_rd)
    gpu_gate = backend.convexified_gate(gpu_rd)
    if cpu_rd["rows"] != gpu_rd["rows"] or cpu_gate != gpu_gate:
        raise RuntimeError("complete CPU/CuPy oracle mismatch")
    expected_weight = float(Fraction(1, 64)) * float(np.sum(values * values)) / values.size
    if cpu_rd["rows"][1]["bit_weight"] != expected_weight or any(
            row["bit_weight"] != expected_weight for row in cpu_rd["rows"][1]["roles"]):
        raise RuntimeError("preflight global Up/Down multiplier")
    return {"fresh_cpu_matches_frozen_receipt": True,
            "fresh_cupy_matches_frozen_receipt": True,
            "complete_oracle_cpu_cupy_exact": True,
            "cpu_reference_json_sha256": json_sha256(cpu),
            "cupy_parity_json_sha256": json_sha256(gpu),
            "complete_gate_json_sha256": json_sha256(cpu_gate),
            "complete_gate_best_gain_bpw": cpu_gate["best_G_eq_UD_bpw"],
            "global_bit_weight": expected_weight,
            "solver_row_count": len(gpu["solver_parity_rows"]),
            "rd_row_count": len(gpu["rd_rows"])}


def joint_objective(backend, values: np.ndarray, levels: np.ndarray,
                    labels: np.ndarray, bit_weight: float) -> float:
    return float(backend.score_labels(values, levels, labels, bit_weight, True)["objective"])


def inherited_solver_blocker(cp, backend) -> dict:
    # Same deterministic legal-level counterexample as the independent r2
    # source audit.  Exact CPU/CuPy parity here proves the GPU faithfully
    # accelerates the dominance failure rather than repairing it.
    rng = np.random.default_rng(16010)
    values = rng.normal(size=(2, 16)).astype(np.float64)
    scales = np.asarray((0.7, 1.3), dtype=np.float64)
    levels = np.stack([np.tile(scales[e] * backend.LEVELS_RMS, (16, 1))
                       for e in range(2)])
    bit_weight = 0.1
    independent = backend.solve_role_cpu(values, levels, bit_weight, False)
    joint_cpu = backend.solve_role_cpu(values, levels, bit_weight, True)
    joint_gpu = backend.solve_role_cupy(cp, values, levels, bit_weight, True,
                                        chunk_coordinates=7)
    if not np.array_equal(joint_cpu["labels"], joint_gpu["labels"]):
        raise RuntimeError("counterexample CPU/CuPy mismatch")
    valid = joint_objective(backend, values, levels, independent["labels"], bit_weight)
    returned = joint_objective(backend, values, levels, joint_cpu["labels"], bit_weight)
    gap = returned - valid
    if gap <= 1e-12:
        raise RuntimeError("dominance counterexample disappeared")
    return {"finding": "BLOCK_GPU_PARITY_FAITHFULLY_REPRODUCES_UNCERTIFIED_JOINT_SOLVER",
            "valid_independent_labels_under_joint_objective": valid,
            "returned_joint_cpu_objective": returned,
            "returned_joint_gpu_objective": joint_objective(
                backend, values, levels, joint_gpu["labels"], bit_weight),
            "suboptimality_gap": gap, "cpu_gpu_labels_exact": True}


def performance_and_memory(cp, receipt: dict, runner, backend) -> dict:
    fresh = runner.benchmark_updates(cp, backend)
    receipt_perf = receipt["performance_and_memory_estimate"]
    for aperture in ("1/64", "1/8"):
        if fresh["apertures"][aperture]["memory"] != \
                receipt_perf["apertures"][aperture]["memory"]:
            raise RuntimeError("analytic memory ledger drift")
    # Exercise the 1/8 allocation geometry once and use the allocator pool's
    # retained total as an empirical lower bound on peak allocator demand.
    pool = cp.get_default_memory_pool()
    gc.collect()
    pool.free_all_blocks()
    coordinates = 768 * 2048 // 8
    rng = np.random.default_rng(0x4D454D31)
    values = rng.normal(size=(2, coordinates)).astype(np.float64)
    scale = np.sqrt(np.mean(values * values, axis=1, dtype=np.float64))
    levels = np.broadcast_to(scale[:, None, None] * backend.LEVELS_RMS[None, None, :],
                             (2, coordinates, backend.ALPHABET)).copy()
    distortion = cp.asarray((values[:, :, None] - levels) ** 2)
    current = cp.asarray(backend.nearest_labels(values, levels))
    updated, counts = backend.update_labels_cupy_from_distortion(
        cp, distortion, current, 0.01, True, 32768)
    cp.cuda.runtime.deviceSynchronize()
    pool_total = int(pool.total_bytes())
    pool_used = int(pool.used_bytes())
    if counts.shape != (16,) or tuple(updated.shape) != (2, coordinates):
        raise RuntimeError("1/8 memory exercise geometry")
    analytic = fresh["apertures"]["1/8"]["memory"]
    result = {
        "fresh_benchmark_coordinates": fresh["benchmark_coordinates"],
        "fresh_repeat_count": fresh["repeat_count"],
        "fresh_timings": fresh["timings"],
        "receipt_timings": receipt_perf["timings"],
        "fresh_aperture_linear_estimates_seconds": {
            key: value["worst_case_gpu_update_seconds_linear_estimate"]
            for key, value in fresh["apertures"].items()},
        "receipt_aperture_linear_estimates_seconds": {
            key: value["worst_case_gpu_update_seconds_linear_estimate"]
            for key, value in receipt_perf["apertures"].items()},
        "analytic_memory_ledgers_exactly_recomputed": True,
        "one_eighth_allocator_pool_used_bytes_after_joint_update": pool_used,
        "one_eighth_allocator_pool_total_bytes_after_joint_update": pool_total,
        "one_eighth_analytic_explicit_bytes": analytic["conservative_joint_peak_explicit_bytes"],
        "qualification": (
            "timing is a 24,576-coordinate update microbenchmark and linear projection; "
            "memory is explicit-array accounting plus a one-update allocator observation, "
            "not measured end-to-end full-aperture peak or runtime"
        ),
    }
    del updated, current, distortion
    gc.collect()
    pool.free_all_blocks()
    return result


def main() -> None:
    receipt, before_hashes = verify_frozen_target()
    runner = load_module("pairpath_preflight_runner_audit", TARGET / "run_source_free_preflight.py")
    backend = load_module("pairpath_cupy_backend", TARGET / "pairpath_cupy_backend.py")
    tests = load_module("pairpath_preflight_tests_audit", TARGET / "test_source_only.py")
    cp, handles, cuda_path, cache = configure_cupy()
    runtime = device_evidence(cp, receipt, cuda_path, cache)
    parity = parity_evidence(cp, receipt, backend, tests)
    inherited = inherited_solver_blocker(cp, backend)
    performance = performance_and_memory(cp, receipt, runner, backend)
    _, after_hashes = verify_frozen_target()
    if before_hashes != after_hashes:
        raise RuntimeError("target changed during read-only audit")
    report = {
        "schema": "pairpath_p2_local3060_cupy_preflight_v0_independent_audit_v1",
        "target": "research/pairpath_p2_local3060_cupy_preflight_v0",
        "target_receipt_sha256": RECEIPT_SHA256,
        "target_source_hashes": before_hashes,
        "source_and_receipt_closure_passed": True,
        "runtime": runtime,
        "parity": parity,
        "performance_and_memory": performance,
        "inherited_joint_solver_blocker": inherited,
        "inherited_r2_findings": {
            "oracle_global_multiplier": "PASS_IN_PREFLIGHT_BACKEND",
            "finite_r2_role_local_multiplier": "NOT_EXERCISED_AND_REMAINS_BLOCKED",
            "joint_oracle_dominance_certificate": "BLOCK_REPRODUCED_ON_CPU_AND_CUPY",
            "finite_r2_tree_descriptor_validation": "NOT_EXERCISED_AND_REMAINS_BLOCKED",
        },
        "verdict": "PASS_RUNTIME_PARITY__BLOCK_PAYLOAD_AND_HARD_KILL_AUTHORITY",
        "qwen_payload_opened": False,
        "network_accessed": False,
        "runpod_accessed": False,
        "target_modified": False,
    }
    if not handles:
        raise RuntimeError("DLL handles unexpectedly absent")
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
