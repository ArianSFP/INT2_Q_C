"""Production-geometry, payload-free CPU/CuPy parity for CBIB-1 r3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


AUTHORIZATION = "RUN_SOURCE_FREE_CBIB1_QWEN_DEPLOYMENT_PARITY_V0_R3"
FLOAT_TOLERANCE = 1e-8


def _compare(left, right, path: str, stats: dict) -> None:
    """Require structural equality, exact discrete fields and bounded floats."""
    if isinstance(left, dict):
        if not isinstance(right, dict) or list(left) != list(right):
            raise RuntimeError(f"mapping mismatch at {path}")
        for key in left:
            _compare(left[key], right[key], f"{path}.{key}", stats)
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise RuntimeError(f"sequence mismatch at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            _compare(a, b, f"{path}[{index}]", stats)
        return
    if isinstance(left, float) or isinstance(right, float):
        a, b = float(left), float(right)
        if not math.isfinite(a) or not math.isfinite(b):
            raise RuntimeError(f"nonfinite float at {path}")
        delta = abs(a - b)
        stats["float_field_count"] += 1
        if delta > stats["max_float_absolute_delta"]:
            stats["max_float_absolute_delta"] = delta
            stats["max_float_path"] = path
        if delta > FLOAT_TOLERANCE:
            raise RuntimeError(f"float mismatch at {path}: {delta}")
        return
    if type(left) is not type(right) or left != right:
        raise RuntimeError(f"exact field mismatch at {path}")
    stats["exact_field_count"] += 1


def _as_int_list(value) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def _independent_counts(
    group_labels, assignments, np_module=None,
) -> tuple[list[int], list]:
    """Reconstruct latent/conditional counts without evaluator-only fields."""
    if np_module is not None:
        q = np_module.asarray(group_labels, dtype=np_module.uint8)
        states = np_module.asarray(assignments, dtype=np_module.uint8)
        if q.ndim != 2 or states.shape != (q.shape[1],):
            raise RuntimeError("independent count geometry")
        if bool(np_module.any(q >= 4)) or bool(np_module.any(states >= 2)):
            raise RuntimeError("independent count values")
        latent = np_module.bincount(states, minlength=2).astype(
            np_module.int64
        )
        conditional = np_module.empty(
            (q.shape[0], 2, 4), dtype=np_module.int64
        )
        for expert in range(q.shape[0]):
            code = states.astype(np_module.int64) * 4 + q[expert].astype(
                np_module.int64
            )
            conditional[expert] = np_module.bincount(
                code, minlength=8
            ).reshape(2, 4)
        return latent.astype(int).tolist(), conditional.astype(int).tolist()

    # Tiny stdlib-only regression fallback; production always passes NumPy.
    rows = [
        _as_int_list(row)
        for row in (group_labels.tolist()
                    if hasattr(group_labels, "tolist") else group_labels)
    ]
    states = _as_int_list(assignments)
    if not rows or any(len(row) != len(states) for row in rows):
        raise RuntimeError("independent count geometry")
    latent = [0, 0]
    conditional = [[[0, 0, 0, 0] for _ in range(2)] for _ in rows]
    for coordinate, state in enumerate(states):
        if state not in (0, 1):
            raise RuntimeError("independent count state")
        latent[state] += 1
        for expert, row in enumerate(rows):
            symbol = row[coordinate]
            if symbol not in (0, 1, 2, 3):
                raise RuntimeError("independent count symbol")
            conditional[expert][state][symbol] += 1
    return latent, conditional


def _compare_assignment_count_evidence(
    group_labels, cpu_eval: dict, gpu_eval: dict, gpu_assignment_to_host,
    evidence_label: str, np_module=None,
) -> int:
    """Compare assignments and counts while honoring the CPU evaluator schema.

    The frozen CPU evaluator returns no count fields.  Counts are deliberately
    reconstructed from its returned assignments and the literal labels.
    """
    cpu_assignments = _as_int_list(cpu_eval["assignments"])
    gpu_assignments = _as_int_list(
        gpu_assignment_to_host(gpu_eval["assignments"])
    )
    if cpu_assignments != gpu_assignments:
        raise RuntimeError(f"{evidence_label} assignment parity")
    cpu_latent, cpu_conditional = _independent_counts(
        group_labels, cpu_assignments, np_module
    )
    gpu_latent = _as_int_list(gpu_eval["test_latent_counts"])
    gpu_conditional = (
        gpu_eval["test_conditional_counts"].tolist()
        if hasattr(gpu_eval["test_conditional_counts"], "tolist")
        else gpu_eval["test_conditional_counts"]
    )
    gpu_conditional = [
        [[int(value) for value in symbols] for symbols in states]
        for states in gpu_conditional
    ]
    if cpu_latent != gpu_latent:
        raise RuntimeError(f"{evidence_label} latent-count parity")
    if cpu_conditional != gpu_conditional:
        raise RuntimeError(f"{evidence_label} conditional-count parity")
    return len(cpu_assignments)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--deployment-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.authorization != AUTHORIZATION:
        raise SystemExit("source-free authorization mismatch")
    package = Path(__file__).resolve().parent
    observed = hashlib.sha256((package / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()
    if observed != args.deployment_manifest_sha256:
        raise SystemExit("external deployment-manifest mismatch")
    sys.path.insert(0, str(package))
    import verify_source
    verify_source.verify(package, observed)

    # Accelerator/runtime imports are deliberately after full source closure.
    import cupy as cp
    import numpy as np
    import clustered_ib_core as core
    import cupy_worker as worker
    from source_free_fixture import (
        COORDINATES,
        EXPERT_COUNT,
        FOLD_COUNT,
        SCALE_BYTES_PER_EXPERT,
        SUPERBLOCK_VALUES,
        make_production_geometry_survivor_fixture,
        make_quantizer_fixture,
    )

    values = make_quantizer_fixture()
    scales_ref = worker._scale_u16_cpu(values)
    flat = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    labels_ref = np.empty(flat.size, dtype=np.uint8)
    for block, scale_bits in enumerate(scales_ref):
        lo, hi = block * worker.BLOCK_VALUES, (block + 1) * worker.BLOCK_VALUES
        scale = float(scale_bits.view(np.float16))
        threshold = worker.THRESHOLD_RMS * scale
        segment = flat[lo:hi]
        labels_ref[lo:hi] = np.where(
            segment < -threshold, 0,
            np.where(segment < 0.0, 1, np.where(segment <= threshold, 2, 3)),
        ).astype(np.uint8)
    labels_gpu, scales_gpu = worker.quantize_canonical_gpu(values)
    if not np.array_equal(scales_ref, scales_gpu):
        raise RuntimeError("quantizer scale parity")
    if not np.array_equal(labels_ref, cp.asnumpy(labels_gpu)):
        raise RuntimeError("quantizer label parity")

    labels = make_production_geometry_survivor_fixture()
    if labels.shape != (EXPERT_COUNT, 2, COORDINATES):
        raise RuntimeError("fixture geometry")
    folds = core.fold_ids(COORDINATES, FOLD_COUNT, SUPERBLOCK_VALUES)
    fold_counts = np.bincount(folds, minlength=FOLD_COUNT)
    if fold_counts.shape != (8,) or np.any(fold_counts == 0):
        raise RuntimeError("all production folds must be populated")

    q_gpu = cp.asarray(labels)
    pair_gpu = worker.pairwise_scores_by_fold_gpu(q_gpu, folds, FOLD_COUNT)
    pair_cpu = np.stack([
        core.pairwise_information_scores(labels[:, :, folds != fold])
        for fold in range(FOLD_COUNT)
    ])
    pair_delta = float(np.max(np.abs(pair_cpu - pair_gpu)))
    if not math.isfinite(pair_delta) or pair_delta > 1e-12:
        raise RuntimeError("all-fold pairwise-MI parity")

    detailed = {
        "groups_checked": 0,
        "models_checked": 0,
        "training_assignments_checked": 0,
        "heldout_assignments_checked": 0,
        "latent_count_arrays_checked": 0,
        "conditional_count_arrays_checked": 0,
        "max_model_float_absolute_delta": 0.0,
    }
    for group_size in (2, 4, 8, 16):
        for fold in range(FOLD_COUNT):
            train_mask = folds != fold
            test_mask = folds == fold
            partition_cpu = core.greedy_equal_partition(pair_cpu[fold], group_size)
            partition_gpu = core.greedy_equal_partition(pair_gpu[fold], group_size)
            if partition_cpu != partition_gpu:
                raise RuntimeError("partition parity")
            for group in partition_cpu:
                detailed["groups_checked"] += 1
                for role in range(2):
                    train = labels[np.asarray(group), role][:, train_mask]
                    test = labels[np.asarray(group), role][:, test_mask]
                    cpu_model = core.fit_binary_product_model(train)
                    gpu_model = worker.fit_binary_product_model_gpu(cp.asarray(train))
                    if not np.array_equal(cpu_model.latent_counts, gpu_model.latent_counts):
                        raise RuntimeError("hard-EM latent-count parity")
                    if not np.array_equal(
                        cpu_model.conditional_counts, gpu_model.conditional_counts
                    ):
                        raise RuntimeError("hard-EM conditional-count parity")
                    cpu_train_eval = core.evaluate_binary_model(
                        train, cpu_model.latent_counts, cpu_model.conditional_counts
                    )
                    gpu_train_eval = worker.evaluate_binary_model_gpu(
                        cp.asarray(train), gpu_model.latent_counts,
                        gpu_model.conditional_counts,
                    )
                    detailed["training_assignments_checked"] += (
                        _compare_assignment_count_evidence(
                            train, cpu_train_eval, gpu_train_eval, cp.asnumpy,
                            "training", np,
                        )
                    )
                    cpu_eval = core.evaluate_binary_model(
                        test, cpu_model.latent_counts, cpu_model.conditional_counts
                    )
                    gpu_eval = worker.evaluate_binary_model_gpu(
                        cp.asarray(test), gpu_model.latent_counts,
                        gpu_model.conditional_counts,
                    )
                    detailed["heldout_assignments_checked"] += (
                        _compare_assignment_count_evidence(
                            test, cpu_eval, gpu_eval, cp.asnumpy, "held-out", np
                        )
                    )
                    numeric = [
                        abs(float(cpu_model.train_nll_bits) -
                            float(gpu_model.train_nll_bits)),
                        abs(float(cpu_eval["latent_bits"]) -
                            float(gpu_eval["latent_bits"])),
                        abs(float(cpu_eval["total_bits"]) -
                            float(gpu_eval["total_bits"])),
                    ]
                    numeric.extend(
                        abs(float(a) - float(b))
                        for a, b in zip(cpu_eval["private_bits"], gpu_eval["private_bits"])
                    )
                    delta = max(numeric)
                    if not math.isfinite(delta) or delta > FLOAT_TOLERANCE:
                        raise RuntimeError("model float parity")
                    detailed["max_model_float_absolute_delta"] = max(
                        detailed["max_model_float_absolute_delta"], delta
                    )
                    detailed["models_checked"] += 1
                    detailed["latent_count_arrays_checked"] += 2
                    detailed["conditional_count_arrays_checked"] += 2

    cpu_gate = core.score_source_gate(
        labels, SCALE_BYTES_PER_EXPERT, fold_count=FOLD_COUNT,
        superblock_values=SUPERBLOCK_VALUES, run_controls=True,
    )
    gpu_gate = worker.score_source_gate_gpu(
        q_gpu, SCALE_BYTES_PER_EXPERT, fold_count=FOLD_COUNT,
        superblock_values=SUPERBLOCK_VALUES, run_controls=True,
    )
    stats = {
        "float_field_count": 0,
        "exact_field_count": 0,
        "max_float_absolute_delta": 0.0,
        "max_float_path": "",
    }
    _compare(cpu_gate, gpu_gate, "gate", stats)
    if [int(row["group_size"]) for row in cpu_gate["source_scores"]] != [2, 4, 8, 16]:
        raise RuntimeError("complete source group-size bank")
    if any(len(row["fold_evidence"]) != FOLD_COUNT
           for row in cpu_gate["source_scores"]):
        raise RuntimeError("complete source folds")
    source_survivors = [
        row for row in cpu_gate["source_scores"]
        if float(row["favorable_gross_gain_bpw"]) >= core.TARGET_GAIN_BPW
        and bool(row["feasible_rate_endpoints"])
    ]
    if not source_survivors:
        raise RuntimeError("fixture did not produce actual source/read survivor")
    if cpu_gate["controls_executed"] is not True:
        raise RuntimeError("control branch not executed")
    if len(cpu_gate["controls"]) != len(core.CONTROL_SEEDS) or len(core.CONTROL_SEEDS) != 8:
        raise RuntimeError("all eight controls required")
    if [int(row["seed"]) for row in cpu_gate["controls"]] != list(core.CONTROL_SEEDS):
        raise RuntimeError("control seed/order")

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    device = props["name"]
    if isinstance(device, bytes):
        device = device.decode("utf-8", errors="strict")
    receipt = {
        "schema": "same-layer-clustered-ib-qwen-deployment-source-free-cupy-v0-r3",
        "status": "PASS_PRODUCTION_GEOMETRY_FULL_CPU_CUPY_PARITY",
        "deployment_manifest_sha256": observed,
        "payload_or_qwen_accessed": False,
        "production_geometry": {
            "experts": EXPERT_COUNT,
            "roles": 2,
            "coordinates_per_role": COORDINATES,
            "fold_count": FOLD_COUNT,
            "superblock_values": SUPERBLOCK_VALUES,
            "scale_bytes_per_expert": SCALE_BYTES_PER_EXPERT,
            "group_sizes": [2, 4, 8, 16],
            "fold_coordinate_counts": fold_counts.astype(int).tolist(),
        },
        "quantizer_scales_labels_exact": True,
        "pairwise_mi_max_absolute_delta": pair_delta,
        "detailed_assignments_counts_partitions": detailed,
        "full_gate_exact_and_float_parity": stats,
        "source_read_survivor_group_sizes": [
            int(row["group_size"]) for row in source_survivors
        ],
        "source_read_survivor_endpoints": {
            str(row["group_size"]): list(row["feasible_rate_endpoints"])
            for row in source_survivors
        },
        "full_gate_status": cpu_gate["status"],
        "all_controls_executed": True,
        "control_count": len(cpu_gate["controls"]),
        "fixture_labels_sha256": hashlib.sha256(
            labels.tobytes(order="C")
        ).hexdigest(),
        "numpy_version": np.__version__,
        "numpy_file": str(Path(np.__file__).resolve()),
        "cupy_version": cp.__version__,
        "device_name": str(device),
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
