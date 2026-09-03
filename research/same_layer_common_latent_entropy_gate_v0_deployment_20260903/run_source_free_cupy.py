"""Mandatory source-free CPU/CuPy parity preflight; no payload arguments exist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


FIXTURE_TOKEN = "RUN_SOURCE_FREE_COMMON_LATENT_FIXTURE_V0"


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
    verify_source.verify(package, args.source_manifest_sha256)

    import cupy as cp
    from common_latent_core import quantize_canonical_cpu, score_labels_cpu
    from cupy_worker import quantize_canonical_gpu, score_labels_gpu
    from source_free_fixture import make_common_label_fixture, make_quantizer_fixture

    labels = make_common_label_fixture()
    gpu_labels = cp.asarray(labels)
    comparisons = {}
    for cardinality in (2, 4):
        objectives = ("favorable", "charged") if cardinality == 2 else ("charged",)
        for objective in objectives:
            cpu = score_labels_cpu(labels, cardinality, 0, objective)
            gpu = score_labels_gpu(gpu_labels, cardinality, 0, objective)
            key = f"k{cardinality}_{objective}"
            comparisons[key] = {
                "planes_equal": cpu["planes"] == gpu["planes"],
                "counts_equal": cpu["count_evidence"] == gpu["count_evidence"],
                "two_part_bits_equal": cpu["common_two_part_bits"] == gpu["common_two_part_bits"],
            }
            if not all(comparisons[key].values()):
                raise RuntimeError(f"CPU/CuPy count parity failed: {key}")

    values = make_quantizer_fixture()
    cpu_q = quantize_canonical_cpu(values, values.shape[1])
    gpu_q, gpu_scale = quantize_canonical_gpu(values, values.shape[1])
    quantizer_equal = (
        np_array_equal(cpu_q.scale_u16, gpu_scale)
        and np_array_equal(cpu_q.labels.reshape(-1), cp.asnumpy(gpu_q))
    )
    if not quantizer_equal:
        raise RuntimeError("CPU/CuPy quantizer replay parity failed")

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="strict")
    receipt = {
        "schema": "same_layer_common_latent_source_free_cupy_receipt_v0",
        "status": "PASS_SOURCE_FREE_CPU_CUPY_PARITY",
        "manifest_sha256": observed,
        "fixture_labels_sha256": hashlib.sha256(labels.tobytes(order="C")).hexdigest(),
        "quantizer_fixture_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "comparisons": comparisons,
        "quantizer_equal": quantizer_equal,
        "cupy_version": cp.__version__,
        "device_name": str(name),
    }
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


def np_array_equal(left, right) -> bool:
    # Import occurs only after source closure has been verified.
    import numpy as np
    return bool(np.array_equal(left, right))


if __name__ == "__main__":
    raise SystemExit(main())
