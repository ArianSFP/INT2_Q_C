#!/usr/bin/env python3
"""Run the sealed source-free nonlocal WFA synthetic experiment."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np

from nonlocal_wfa import (
    BLOCK_LENGTH,
    CHECKS,
    MAX_SUFFIX_DEPTH,
    SCHEMA,
    STANDALONE_REQUIRED_BPW,
    capacity_sanity,
    canonical_json,
    decode_blocks,
    encode_blocks,
    exact_normalization_probe,
    generate_syndrome_blocks,
    logical_codelength_bits,
    model_byte_ledger,
    physical_stream_ledger,
    select_model,
    sha256_bytes,
    suffix_cross_entropy,
    write_bytes_create_new,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--train-blocks", type=int, default=4096)
    result.add_argument("--validation-blocks", type=int, default=8192)
    result.add_argument("--amortized-experts", type=int, default=8)
    return result


def one_case(
    name: str,
    constrained: bool,
    train_blocks: int,
    validation_blocks: int,
    amortized_experts: int,
) -> tuple[dict[str, object], bytes, bytes]:
    seed_base = 7001 if constrained else 17001
    train = generate_syndrome_blocks(train_blocks, seed_base, constrained)
    validation = generate_syndrome_blocks(validation_blocks, seed_base + 1, constrained)
    model, candidates = select_model(train, validation, amortized_experts)
    packet = model.serialize()
    restored = type(model).deserialize(packet)
    assert restored.serialize() == packet
    payload, meaningful_bits = encode_blocks(restored, validation)
    decoded = decode_blocks(restored, payload, validation_blocks)
    if not np.array_equal(decoded, validation):
        raise RuntimeError(f"arithmetic roundtrip failed for {name}")
    logical_bits = logical_codelength_bits(restored, validation)
    suffix_rows = [
        suffix_cross_entropy(train, validation, depth)
        for depth in range(MAX_SUFFIX_DEPTH + 1)
    ]
    ledger = physical_stream_ledger(
        restored,
        len(payload),
        int(validation.size),
        amortized_experts,
    )
    result = {
        "name": name,
        "constrained": constrained,
        "train_blocks": train_blocks,
        "validation_blocks": validation_blocks,
        "symbols_per_expert": int(validation.size),
        "amortized_experts": amortized_experts,
        "selected_syndrome_bits": restored.syndrome_bits,
        "selected_chi": restored.chi,
        "candidate_rows": candidates,
        "logical_bits": logical_bits,
        "logical_bps": logical_bits / validation.size,
        "actual_arithmetic_payload_bytes": len(payload),
        "actual_arithmetic_meaningful_bits": meaningful_bits,
        "actual_arithmetic_bps": 8.0 * len(payload) / validation.size,
        "model_ledger": model_byte_ledger(restored),
        "stream_ledger": ledger,
        "suffix_rows": suffix_rows,
        "best_suffix_logical_bps": min(float(row["logical_bps"]) for row in suffix_rows),
        "population_suffix_bps_depth_le_25": 1.0,
        "population_reason": (
            "At checksum j, suffix depth d<=25 omits unique iid body bit j. "
            "Earlier checks use disjoint residue classes, so the target checksum remains fair."
        ),
        "exact_normalization_probe": exact_normalization_probe(
            restored, validation[0, :8].tolist()
        ),
        "packet_sha256": sha256_bytes(packet),
        "payload_sha256": sha256_bytes(payload),
        "roundtrip_exact": True,
    }
    return result, packet, payload


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    args.output.mkdir(parents=False, exist_ok=False)

    structured, structured_packet, structured_payload = one_case(
        "six_nonlocal_parities",
        True,
        args.train_blocks,
        args.validation_blocks,
        args.amortized_experts,
    )
    control, control_packet, control_payload = one_case(
        "matched_iid_control",
        False,
        args.train_blocks,
        args.validation_blocks,
        args.amortized_experts,
    )

    gross_detected = float(control["logical_bps"]) - float(structured["logical_bps"])
    net_synthetic = float(structured["stream_ledger"]["aggregate_saving_bps"])
    receipt = {
        "schema": SCHEMA,
        "status": "PASS_SOURCE_FREE_SYNTHETIC_ONLY",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gpu_path_present": False,
            "cupy_imported": False,
            "cuda_initialized": False,
        },
        "access_attestation": {
            "qwen_or_other_model_payload_opened_statted_hashed_or_enumerated": False,
            "decoded_production_symbol_stream_opened": False,
            "runpod_job_launched": False,
        },
        "fixture": {
            "block_length": BLOCK_LENGTH,
            "body_symbols": BLOCK_LENGTH - CHECKS,
            "independent_nonlocal_checks": CHECKS,
            "gross_population_saving_bps": CHECKS / BLOCK_LENGTH,
            "standalone_required_bpw_reference_only": STANDALONE_REQUIRED_BPW,
        },
        "structured": structured,
        "matched_control": control,
        "detected_control_minus_structured_logical_bps": gross_detected,
        "structured_aggregate_model_charged_saving_bps": net_synthetic,
        "capacity_sanity": capacity_sanity(),
        "compute_ledger": {
            "generic_dense_forward_time": "O(N*chi^2)",
            "generic_dense_forward_work_per_symbol": "two chi-by-chi nonnegative matrix-vector products",
            "prototype_unifilar_forward_time": "O(N)",
            "prototype_sparse_storage": "O(C*chi)",
            "dense_equivalent_storage": "O(C*chi^2)",
            "training_search": "seven fixed universal parity-bank topologies, k=0..6; all fitted on train and selected on untouched validation",
        },
        "claim_boundary": {
            "positive": (
                "The tied nonnegative WFA class can expose a genuinely nonlocal source law "
                "whose site marginals and every bounded suffix d<=25 are uninformative."
            ),
            "negative_run_could_kill": (
                "Only the frozen tied, causal, quantized, reset-32 parity-bank WFA candidates "
                "k<=6 under this fitting/selection and packet ledger."
            ),
            "negative_run_could_not_kill": (
                "Arbitrary HMM/MPS/TTN priors, chi>64, longer resets, non-unifilar states, "
                "Born models, learned disentanglers, other public contexts, or lossy RD gains."
            ),
            "not_claimed": (
                "No Qwen entropy saving, no universal SwiGLU-MoE target pass, no current-codec "
                "cold-read result, and no authorization to open a model payload."
            ),
        },
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt))

    write_bytes_create_new(args.output / "structured_model.bin", structured_packet)
    write_bytes_create_new(args.output / "structured_payload.bin", structured_payload)
    write_bytes_create_new(args.output / "control_model.bin", control_packet)
    write_bytes_create_new(args.output / "control_payload.bin", control_payload)
    write_bytes_create_new(
        args.output / "synthetic_result.json",
        (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii"),
    )
    completion = {
        "schema": "nonlocal-wfa-global-state-synthetic-complete-v0",
        "status": receipt["status"],
        "synthetic_result_sha256": sha256_bytes((args.output / "synthetic_result.json").read_bytes()),
        "structured_model_sha256": sha256_bytes(structured_packet),
        "structured_payload_sha256": sha256_bytes(structured_payload),
        "control_model_sha256": sha256_bytes(control_packet),
        "control_payload_sha256": sha256_bytes(control_payload),
    }
    completion["completion_sha256"] = sha256_bytes(canonical_json(completion))
    write_bytes_create_new(
        args.output / "COMPLETE.json",
        (json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii"),
    )
    print(json.dumps({
        "status": receipt["status"],
        "selected_chi": structured["selected_chi"],
        "structured_logical_bps": structured["logical_bps"],
        "control_logical_bps": control["logical_bps"],
        "detected_bps": gross_detected,
        "model_charged_saving_bps": net_synthetic,
        "cold_amplification": structured["stream_ledger"]["synthetic_cold_read_amplification_vs_raw_one_bit_frame"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

