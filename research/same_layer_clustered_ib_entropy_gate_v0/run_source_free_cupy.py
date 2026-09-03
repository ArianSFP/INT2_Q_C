"""Mandatory source-free CPU/CuPy parity preflight; no payload arguments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


FIXTURE_TOKEN = "RUN_SOURCE_FREE_CBIB1_FIXTURE_V0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-token", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.fixture_token != FIXTURE_TOKEN:
        raise SystemExit("fixture token mismatch")
    package = Path(__file__).resolve().parent
    manifest_path = package / "SOURCE_MANIFEST.json"
    observed = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if observed != args.source_manifest_sha256:
        raise SystemExit("external source-manifest digest mismatch")
    sys.path.insert(0, str(package))
    import verify_source
    verify_source.verify(package, observed)

    import cupy as cp
    import numpy as np
    from clustered_ib_core import _model_counts, evaluate_binary_model
    from cupy_backend import evaluate_binary_model_gpu, model_counts_gpu
    from source_free_fixture import make_gpu_model_fixture

    labels, assignments = make_gpu_model_fixture()
    cpu_latent, cpu_conditional = _model_counts(labels, assignments)
    gpu_latent, gpu_conditional = model_counts_gpu(cp.asarray(labels), cp.asarray(assignments))
    counts_equal = (
        np.array_equal(cpu_latent, cp.asnumpy(gpu_latent))
        and np.array_equal(cpu_conditional, cp.asnumpy(gpu_conditional))
    )
    if not counts_equal:
        raise RuntimeError("CPU/CuPy count parity failed")
    cpu = evaluate_binary_model(labels, cpu_latent, cpu_conditional)
    gpu = evaluate_binary_model_gpu(
        cp.asarray(labels), gpu_latent, gpu_conditional
    )
    assignments_equal = np.array_equal(cpu["assignments"], gpu["assignments"])
    numeric_error = max(
        abs(cpu["latent_bits"] - gpu["latent_bits"]),
        abs(cpu["total_bits"] - gpu["total_bits"]),
        *(abs(a - b) for a, b in zip(cpu["private_bits"], gpu["private_bits"])),
    )
    if not assignments_equal or numeric_error > 1e-9:
        raise RuntimeError("CPU/CuPy objective parity failed")
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    device = props["name"]
    if isinstance(device, bytes):
        device = device.decode("utf-8", errors="strict")
    receipt = {
        "schema": "same_layer_clustered_ib_source_free_cupy_receipt_v0",
        "status": "PASS_SOURCE_FREE_CPU_CUPY_PARITY",
        "manifest_sha256": observed,
        "fixture_labels_sha256": hashlib.sha256(labels.tobytes(order="C")).hexdigest(),
        "fixture_assignments_sha256": hashlib.sha256(assignments.tobytes(order="C")).hexdigest(),
        "counts_equal": counts_equal,
        "assignments_equal": assignments_equal,
        "maximum_absolute_bit_error": numeric_error,
        "cupy_version": cp.__version__,
        "device_name": str(device),
    }
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

