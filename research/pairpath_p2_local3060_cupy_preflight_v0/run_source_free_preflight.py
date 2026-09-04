"""Pinned local-RTX3060 source-free PAIRPATH CuPy preflight.

This is deliberately not a production or payload runner.  It imports no model
library, accepts no input path, and fails unless RUN_GATE.json remains HOLD.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
VENV = WORKSPACE / ".venv-cupy"
SITE = VENV / "Lib" / "site-packages"
CUDA_PATH = SITE / "nvidia" / "cuda_runtime"
CACHE = WORKSPACE / "tmp" / "pairpath_p2_sourcefree_cupy_cache_v0"
EXPECTED_UUID_HEX = "458a424a76e365e50470803e0ed131ca"
EXPECTED_DEVICE = "NVIDIA GeForce RTX 3060"
REFERENCE_CORE_SHA256 = "2c99a31aef669cabbb67137061233640b013e8c50a5132ddbcc9ffec2c239034"
REFERENCE_MANIFEST_SHA256 = "21983efff5ac5c0593a655cae4136d35ca24400fd807f9fe4be458a34b18e622"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_process_local_runtime() -> list[object]:
    need(Path(sys.executable).resolve() == (VENV / "Scripts" / "python.exe").resolve(),
         "must use pinned workspace CuPy Python")
    compatibility = WORKSPACE / ".tools" / "cuda_dlls_3060"
    directories = [compatibility]
    directories.extend(sorted(path for path in (SITE / "nvidia").glob("*\\bin")
                              if path.is_dir()))
    for path in directories:
        need(path.is_dir(), f"missing process-local DLL directory: {path}")
    CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_PATH"] = str(CUDA_PATH)
    os.environ["CUPY_CACHE_DIR"] = str(CACHE)
    handles = [os.add_dll_directory(str(path)) for path in directories]
    return handles


def canonical_uuid(raw: object) -> tuple[str, str]:
    need(isinstance(raw, bytes) and len(raw) >= 16, "unexpected CuPy UUID ABI")
    uuid_hex = raw[:16].hex()
    return uuid_hex, raw[16:].hex()


def benchmark_updates(cp, backend) -> dict:
    import numpy as np

    rng = np.random.default_rng(0x306050414952)
    benchmark_n = 24576
    values = rng.normal(size=(2, benchmark_n)).astype(np.float64)
    scales = np.sqrt(np.mean(values * values, axis=1, dtype=np.float64))
    levels = np.broadcast_to(scales[:, None, None] *
                             backend.LEVELS_RMS[None, None, :],
                             (2, benchmark_n, backend.ALPHABET)).copy()
    distortion = cp.asarray((values[:, :, None] - levels) ** 2)
    labels = cp.asarray(backend.nearest_labels(values, levels))

    # Warm both paths, then measure complete update + exact count calls.
    for joint in (False, True):
        backend.update_labels_cupy_from_distortion(cp, distortion, labels, 0.01,
                                                   joint, 32768)
    cp.cuda.runtime.deviceSynchronize()
    timings = {}
    repeats = 8
    for joint, name in ((False, "independent"), (True, "joint")):
        start = time.perf_counter()
        for _ in range(repeats):
            backend.update_labels_cupy_from_distortion(cp, distortion, labels, 0.01,
                                                       joint, 32768)
        cp.cuda.runtime.deviceSynchronize()
        timings[name + "_update_seconds_at_24576"] = (
            time.perf_counter() - start) / repeats

    full_coordinates = 768 * 2048
    apertures = {}
    starts = 18
    updates = backend.MAX_ALTERNATIONS
    lambdas = len(backend.LAMBDA_GRID)
    roles = len(backend.ROLES)
    base_pair = (timings["independent_update_seconds_at_24576"] +
                 timings["joint_update_seconds_at_24576"])
    for denominator in (64, 8):
        coordinates = full_coordinates // denominator
        scale = coordinates / benchmark_n
        kernel_worst = base_pair * scale * starts * updates * lambdas * roles
        apertures[f"1/{denominator}"] = {
            "coordinates_per_role_per_expert": coordinates,
            "four_source_weights": 4 * coordinates,
            "source_fp64_bytes": 4 * coordinates * 8,
            "memory": backend.theoretical_memory_bytes(coordinates),
            "worst_case_gpu_update_seconds_linear_estimate": kernel_worst,
            "estimate_scope": (
                "18 starts * 8 updates * 12 lambdas * 2 roles * "
                "(independent+joint); excludes CPU canonical score/copy, allocator "
                "workspace, setup, controls; actual convergence may stop early"
            ),
        }
    return {"benchmark_coordinates": benchmark_n, "repeat_count": repeats,
            "timings": timings, "apertures": apertures}


def main() -> None:
    gate = json.loads((HERE / "RUN_GATE.json").read_text(encoding="utf-8"))
    need(gate == {
        "gpu_execution_enabled": True,
        "payload_authority_present": False,
        "payload_execution_enabled": False,
        "qwen_or_model_payload_allowed": False,
        "remote_or_runpod_allowed": False,
        "scope": "SOURCE_FREE_SYNTHETIC_PREFLIGHT_ONLY",
        "status": "HOLD_PRODUCTION_AND_PAYLOAD",
    }, "RUN_GATE.json is not the frozen source-free HOLD")
    reference_package = HERE.parent / "pairpath_fl_same_layer_microcodec_v0_20260903_r2"
    need(sha256_file(reference_package / "pairpath_r2_core.py") == REFERENCE_CORE_SHA256,
         "frozen PAIRPATH reference core hash mismatch")
    need(sha256_file(reference_package / "SOURCE_MANIFEST.json") == REFERENCE_MANIFEST_SHA256,
         "frozen PAIRPATH source manifest hash mismatch")
    handles = configure_process_local_runtime()
    sys.path.insert(0, str(HERE))
    import numpy as np
    import cupy as cp
    import pairpath_cupy_backend as backend
    import test_source_only

    props = cp.cuda.runtime.getDeviceProperties(0)
    device_name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    uuid_hex, uuid_tail = canonical_uuid(props["uuid"])
    need(device_name == EXPECTED_DEVICE, "wrong GPU name")
    need(uuid_hex == EXPECTED_UUID_HEX, "wrong GPU UUID")
    reference = test_source_only.run_cpu_tests()
    need(reference["reference_core_sha256"] == REFERENCE_CORE_SHA256,
         "reference semantic core mismatch")
    parity = test_source_only.run_cupy_tests(cp)
    performance = benchmark_updates(cp, backend)
    receipt = {
        "schema": "pairpath_p2_local3060_sourcefree_cupy_preflight_v0",
        "status": "PASS_SOURCE_FREE_CPU_CUPY_PARITY_HOLD_PAYLOAD",
        "claim_boundary": "synthetic mechanism only; no model/Qwen payload, network, RunPod, authority, or production execution",
        "run_gate": gate,
        "runtime": {
            "hostname": platform.node(), "python": list(sys.version_info[:3]),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": sha256_file(Path(sys.executable)),
            "numpy": np.__version__, "cupy": cp.__version__,
            "device_name": device_name,
            "device_uuid": "GPU-" + "-".join((uuid_hex[:8], uuid_hex[8:12],
                                                uuid_hex[12:16], uuid_hex[16:20],
                                                uuid_hex[20:])),
            "cupy_uuid_trailing_hex": uuid_tail,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver_api": cp.cuda.runtime.driverGetVersion(),
            "cuda_path": str(CUDA_PATH), "cupy_cache_dir": str(CACHE),
        },
        "semantic_contract": {
            "alphabet": backend.ALPHABET, "roles": list(backend.ROLES),
            "maximum_alternations": backend.MAX_ALTERNATIONS,
            "lambda_grid": [str(v) for v in backend.LAMBDA_GRID],
            "role_conditioned_mi": True, "global_updown_bit_weight": True,
            "symmetric_multistarts": True, "lowest_index_tie_break": True,
            "cpu_canonical_candidate_scoring": True,
            "chunk_coordinates": 32768,
            "frozen_pairpath_source_core_sha256": REFERENCE_CORE_SHA256,
            "frozen_pairpath_source_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        },
        "frozen_reference_cpu_parity": reference,
        "parity": parity,
        "performance_and_memory_estimate": performance,
        "source_files": {
            name: sha256_file(HERE / name) for name in
            ("pairpath_cupy_backend.py", "test_source_only.py", "run_source_free_preflight.py",
             "RUN_GATE.json", "README.md")
        },
    }
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), indent=2) + "\n"
    (HERE / "SOURCE_FREE_PREFLIGHT_RECEIPT.json").write_bytes(payload.encode("utf-8"))
    print(json.dumps({"status": receipt["status"],
                      "device_uuid": receipt["runtime"]["device_uuid"],
                      "receipt_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                      "apertures": performance["apertures"]},
                     sort_keys=True, separators=(",", ":")))
    # Keep add_dll_directory handles live until every CUDA object has finished.
    need(bool(handles), "DLL registration handles")


if __name__ == "__main__":
    main()
