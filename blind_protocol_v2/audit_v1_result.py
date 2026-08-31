#!/usr/bin/env python3
"""Independent CuPy rescore and claim-boundary audit of blind panel v1.

This script only reads the already-opened v1 sources and the reconstruction
produced by the frozen independent decoder.  It has no Qwen network client and
cannot access any proposed v2 tensor payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np


GAUSSIAN_LIMIT = 0.050765774772264724
EXPECTED = {
    "auditor_summary": "6aa4d8538389179cdbdb2edaf332d707eef39aa4ef7cd395f8e82d755ca1bb37",
    "encoder_summary": "28b12f1da4bde80596b739e3d321a307d9029605b8ceafb384ea26ddad30ec47",
    "selection": "78a00016a0f62eca3f3c9b451b226d6821b38b8c596eb7b0f691bbe465174095",
    "source_lock": "b520461fa71783ad597e6f71211cd777c0af93baef14177edc9181f54cf918d5",
    "codec_freeze": "66619fddfe2b3ff3a7f5b086e92009df2c17697be68305c79a6d44ecf8abf4d9",
    "route_table": "1a2380a8750363a1fb0f6e2c30904c20d68ba78e59cb6ca15678395496adcf41",
    "container": "ef2cbabc7ad24ff1e5ee9cc7230494deb50d0e2e688455ecf2afcd9c891a8d47",
    "reconstruction": "7d1e00cb3f0328f4b814db5ac2e0814b7547b4552a5afd7ea7b682fb237f3908",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["lock_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def verify_content_seal(value: dict[str, Any]) -> str:
    declared = value.get("lock_sha256")
    clean = dict(value)
    clean.pop("lock_sha256", None)
    actual = hashlib.sha256(canonical_bytes(clean)).hexdigest()
    if declared != actual:
        raise AssertionError(f"bad internal seal: {declared} != {actual}")
    return actual


def assert_close(actual: float, expected: float, label: str) -> None:
    tolerance = max(2e-12, abs(expected) * 2e-12)
    if not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=tolerance):
        raise AssertionError(f"{label}: {actual} != {expected}")


def bf16_to_float64(path: Path, values: int) -> np.ndarray:
    words = np.memmap(path, dtype="<u2", mode="r")
    if words.size != values:
        raise AssertionError(f"BF16 value count mismatch: {path}")
    floats = (np.asarray(words).astype(np.uint32) << np.uint32(16)).view(np.float32)
    return floats.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)

    paths = {
        "auditor_summary": workspace / "frozen_strata_blind_independent_audit_v1/summary.json",
        "encoder_summary": workspace / "frozen_strata_blind_one_shot_v1/summary.json",
        "selection": workspace / "blind_protocol/selection.lock.json",
        "source_lock": workspace / "blind_protocol/unblinded/source_hashes.lock.json",
        "codec_freeze": workspace / "blind_protocol/codec_freeze.lock.json",
        "route_table": workspace / "blind_protocol/route_table.lock.bin",
        "container": workspace / "frozen_strata_blind_one_shot_v1/physical/frozen_strata.plrwf4",
        "reconstruction": workspace / "frozen_strata_blind_independent_audit_v1/reconstructed_canonical_groups.f64.bin",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED:
        raise AssertionError(f"v1 immutable artifact hash drift: {hashes}")

    audit = load_object(paths["auditor_summary"])
    encoder = load_object(paths["encoder_summary"])
    selection = load_object(paths["selection"])
    source_lock = load_object(paths["source_lock"])
    verify_content_seal(selection)
    verify_content_seal(source_lock)

    if audit.get("schema") != "frozen_strata_independent_weight_audit_v1":
        raise AssertionError("unexpected auditor schema")
    if audit.get("status") != "complete_primary_claim_failed":
        raise AssertionError("v1 audit no longer records the failed primary claim")
    if encoder.get("status") != "complete_one_shot_frozen_encode" or not encoder.get("passed"):
        raise AssertionError("v1 frozen encoder did not complete cleanly")
    if not audit.get("integrity_passed") or not audit.get("rate_passed"):
        raise AssertionError("v1 integrity/rate prerequisites unexpectedly failed")
    if audit.get("mse_passed") or audit.get("claim_passed") or audit.get("passed"):
        raise AssertionError("v1 primary claim is incorrectly marked passed")

    matrices = source_lock.get("matrices")
    if not isinstance(matrices, list) or len(matrices) != 18:
        raise AssertionError("source lock must contain exactly 18 matrices")
    reconstruction = np.memmap(paths["reconstruction"], dtype="<f8", mode="r")
    expected_values = sum(int(row["nvalues"]) for row in matrices)
    if reconstruction.size != expected_values:
        raise AssertionError("canonical reconstruction size mismatch")

    device = cp.cuda.Device()
    device.use()
    matrix_rows: list[dict[str, Any]] = []
    cursor = 0
    all_group_sse: list[float] = []
    all_group_energy: list[float] = []
    source_root = paths["source_lock"].parent
    audit_by_tensor = {row["tensor"]: row for row in audit["matrices"]}
    role_sums: dict[str, list[float]] = {}
    pair_sums: dict[str, list[float]] = {}

    for ordinal, matrix in enumerate(matrices):
        if int(matrix["matrix_ordinal"]) != ordinal:
            raise AssertionError("noncontiguous source-lock matrix ordinals")
        tensor = str(matrix["tensor"])
        shape = tuple(int(value) for value in matrix["shape"])
        values = math.prod(shape)
        source_path = source_root / str(matrix["output_relpath"])
        if sha256_file(source_path) != matrix["source_bf16_sha256"]:
            raise AssertionError(f"v1 source hash mismatch: {tensor}")
        source = bf16_to_float64(source_path, values).reshape(shape)
        role = tensor.split(".")[-2]
        layer = int(tensor.split(".")[2])
        expert = int(tensor.split(".")[5])
        natural = source if role in {"gate_proj", "up_proj"} else source.T
        groups = natural.shape[0]
        decoded = np.asarray(reconstruction[cursor * 2048 : (cursor + groups) * 2048]).reshape(groups, 2048)
        cursor += groups

        source_gpu = cp.asarray(natural, dtype=cp.float64)
        decoded_gpu = cp.asarray(decoded, dtype=cp.float64)
        difference = source_gpu - decoded_gpu
        group_sse = cp.asnumpy(cp.sum(difference * difference, axis=1, dtype=cp.float64))
        group_energy = cp.asnumpy(cp.sum(source_gpu * source_gpu, axis=1, dtype=cp.float64))
        max_error = float(cp.max(cp.abs(difference)).get())
        del source_gpu, decoded_gpu, difference
        cp.get_default_memory_pool().free_all_blocks()

        local_sse = math.fsum(float(value) for value in group_sse)
        local_energy = math.fsum(float(value) for value in group_energy)
        all_group_sse.extend(float(value) for value in group_sse)
        all_group_energy.extend(float(value) for value in group_energy)
        expected_row = audit_by_tensor[tensor]
        assert_close(local_sse, float(expected_row["sse_fp64"]), f"{tensor} SSE")
        assert_close(local_energy, float(expected_row["source_energy_fp64"]), f"{tensor} energy")
        assert_close(max_error, float(expected_row["max_absolute_error"]), f"{tensor} max error")
        role_sums.setdefault(role, [0.0, 0.0])
        role_sums[role][0] += local_sse
        role_sums[role][1] += local_energy
        pair_key = f"L{layer}:E{expert}"
        pair_sums.setdefault(pair_key, [0.0, 0.0])
        pair_sums[pair_key][0] += local_sse
        pair_sums[pair_key][1] += local_energy
        matrix_rows.append(
            {
                "matrix_ordinal": ordinal,
                "tensor": tensor,
                "source_bf16_sha256": matrix["source_bf16_sha256"],
                "sse_fp64_cupy": local_sse,
                "source_energy_fp64_cupy": local_energy,
                "relative_mse_cupy": local_sse / local_energy,
                "max_absolute_error_cupy": max_error,
            }
        )

    if cursor * 2048 != expected_values:
        raise AssertionError("canonical reconstruction coverage mismatch")
    total_sse = math.fsum(all_group_sse)
    total_energy = math.fsum(all_group_energy)
    relative_mse = total_sse / total_energy
    assert_close(total_sse, float(audit["distortion"]["sse_sum_fp64"]), "pooled SSE")
    assert_close(total_energy, float(audit["distortion"]["source_energy_sum_fp64"]), "pooled energy")
    assert_close(relative_mse, float(audit["distortion"]["energy_weighted_relative_mse"]), "pooled MSE")

    weights = int(source_lock["source_values"])
    v4_bits = paths["container"].stat().st_size * 8
    route_bits = paths["route_table"].stat().st_size * 8
    bundle_bits = v4_bits + route_bits
    if bundle_bits != int(audit["physical_rate_bundle"]["bundle_bits"]):
        raise AssertionError("physical bundle bit count mismatch")
    rate_passed = bundle_bits * 20 <= 43 * weights
    distortion_passed = relative_mse < GAUSSIAN_LIMIT
    if not rate_passed or distortion_passed:
        raise AssertionError("independent v1 gate result disagrees with expected fail-only-on-MSE")

    threshold_sse = total_energy * GAUSSIAN_LIMIT
    result = seal(
        {
            "schema": "int2-qwen-blind-v1-independent-cupy-reaudit-v1",
            "status": "complete_primary_claim_reconfirmed_failed",
            "passed": True,
            "meaning_of_passed": "this re-audit reproduced the v1 outcome; it does not mean the codec beat the benchmark",
            "runtime": {
                "cupy_version": cp.__version__,
                "device_id": int(device.id),
                "device_name": cp.cuda.runtime.getDeviceProperties(device.id)["name"].decode(),
                "score_accumulator": "CuPy FP64 group reductions then Python math.fsum",
            },
            "immutable_artifact_sha256s": hashes,
            "coverage": {
                "matrices": len(matrices),
                "groups": expected_values // 2048,
                "weights": weights,
                "source_hashes_reverified": len(matrices),
                "canonical_reconstruction_hash_reverified": True,
            },
            "physical_rate": {
                "v4_bits": v4_bits,
                "route_bits": route_bits,
                "bundle_bits": bundle_bits,
                "bundle_bpw": bundle_bits / weights,
                "integer_gate": f"{bundle_bits} * 20 <= 43 * {weights}",
                "integer_gate_passed": rate_passed,
            },
            "distortion": {
                "sse_sum_fp64_cupy": total_sse,
                "source_energy_sum_fp64_cupy": total_energy,
                "energy_weighted_relative_mse_cupy": relative_mse,
                "gaussian_limit_at_2p15": GAUSSIAN_LIMIT,
                "ratio_to_gaussian_limit": relative_mse / GAUSSIAN_LIMIT,
                "margin_below_gaussian_percent": 100.0 * (1.0 - relative_mse / GAUSSIAN_LIMIT),
                "threshold_sse": threshold_sse,
                "sse_excess_over_threshold": total_sse - threshold_sse,
                "relative_sse_reduction_needed_for_strict_pass": 1.0 - threshold_sse / total_sse,
                "mse_gate_passed": distortion_passed,
            },
            "role_breakdown": {
                role: {
                    "sse": sums[0],
                    "energy": sums[1],
                    "relative_mse": sums[0] / sums[1],
                }
                for role, sums in sorted(role_sums.items())
            },
            "expert_triplet_breakdown": {
                key: {
                    "sse": sums[0],
                    "energy": sums[1],
                    "relative_mse": sums[0] / sums[1],
                }
                for key, sums in sorted(pair_sums.items())
            },
            "matrices": matrix_rows,
            "primary_claim": {
                "gate": "artifact integrity && exact physical bundle <= 2.15 bpw && pooled original-BF16 FP64 SSE/energy < 2^-4.3",
                "integrity_passed": True,
                "rate_passed": rate_passed,
                "mse_passed": distortion_passed,
                "claim_passed": rate_passed and distortion_passed,
            },
            "claim_boundary": {
                "defensible": "POLARIS/STRATA v1 completed one frozen one-shot encode and independent decode on its precommitted 18-matrix Qwen panel at 2.1499091254 bpw, but did not beat the Gaussian 2.15-bpw reference.",
                "not_defensible": [
                    "that v1 beat the Gaussian reference",
                    "that a later disjoint panel can retroactively rescue the v1 claim",
                    "a universal Qwen checkpoint, perplexity, or end-to-end model-quality claim",
                    "treating the encoder-only pass flag as a distortion pass",
                ],
                "future_panel_rule": "v2 is a new confirmatory experiment and must be reported separately, regardless of outcome",
            },
            "candidate_v2_tensor_payload_bytes_read": 0,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = args.output.read_bytes()
        rendered = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        if existing != rendered:
            raise FileExistsError("refusing to overwrite a different v1 re-audit receipt")
    else:
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    print(json.dumps({"passed": True, "output": str(args.output), "lock_sha256": result["lock_sha256"], "v1_claim_passed": False, "relative_mse": relative_mse}, sort_keys=True))


if __name__ == "__main__":
    main()
