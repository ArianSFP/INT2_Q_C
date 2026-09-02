#!/usr/bin/env python3
"""Canonical *unsealed* source-free SILT mechanism replay on the RTX 5090.

This executable has no input-payload option.  It constructs long-range
synthetic structured sources and iid uniform matched controls internally.  A
PASS demonstrates format/search/decoder mechanics only; it is not evidence of
structure or gain in any model weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from cupy_backend import search_metadata_cupy
from independent_decoder import verify_decode_reencode
from silt_mechanism import (
    build_container,
    deterministic_permutation,
    deterministic_selectors,
    fit_model,
    flatten_details,
    leaf_digest,
    lift_forward,
    physical_ledger,
    population_suffix_limit,
    require,
    sha256_bytes,
    synthesize_leaves,
)


SCHEMA = "silt-source-free-canonical-synthetic-run-v0"
HIDDEN_METADATA_SEED = 0x6C31
CANDIDATE_SEEDS = (0x0B51, 0x193D, 0x2E71, HIDDEN_METADATA_SEED, 0x79A3, 0x8849, 0xA117, 0xD20B)
SEARCH_TRAIN_VECTORS = 2048
SEARCH_VALIDATION_VECTORS = 1024
MODEL_TRAIN_VECTORS = 8192
EVALUATION_VECTORS = 8192
LANES = 97
EXPERTS = 8


def _json_write_new(path: Path, value: object) -> None:
    require(not path.exists(), f"refuse to overwrite {path.name}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bytes_write_new(path: Path, value: bytes) -> None:
    require(not path.exists(), f"refuse to overwrite {path.name}")
    path.write_bytes(value)


def _seed(alphabet: int, structured: bool, stage: int, index: int = 0) -> int:
    return 10_000_000 * alphabet + 1_000_000 * int(structured) + 10_000 * stage + index


def _make_source(
    alphabet: int,
    vectors: int,
    seed: int,
    structured: bool,
    hidden_permutation: list[int],
    hidden_selectors: list[int],
) -> np.ndarray:
    return synthesize_leaves(
        alphabet,
        vectors,
        LANES,
        seed,
        structured,
        hidden_permutation,
        hidden_selectors,
    )


def _run_class(alphabet: int, structured: bool) -> tuple[dict[str, object], bytes]:
    hidden_permutation = deterministic_permutation(LANES, HIDDEN_METADATA_SEED)
    hidden_selectors = deterministic_selectors(LANES, HIDDEN_METADATA_SEED ^ 0x5A17)
    search_train = _make_source(
        alphabet,
        SEARCH_TRAIN_VECTORS,
        _seed(alphabet, structured, 1),
        structured,
        hidden_permutation,
        hidden_selectors,
    )
    search_validation = _make_source(
        alphabet,
        SEARCH_VALIDATION_VECTORS,
        _seed(alphabet, structured, 2),
        structured,
        hidden_permutation,
        hidden_selectors,
    )
    search = search_metadata_cupy(
        search_train,
        search_validation,
        alphabet,
        CANDIDATE_SEEDS,
        require_rtx_5090=True,
    )
    permutation = list(search.selected_permutation)
    selectors = list(search.selected_selectors)

    model_leaves = _make_source(
        alphabet,
        MODEL_TRAIN_VECTORS,
        _seed(alphabet, structured, 3),
        structured,
        hidden_permutation,
        hidden_selectors,
    )
    model_coefficients = lift_forward(model_leaves, alphabet, permutation, selectors)
    model = fit_model(
        alphabet,
        model_coefficients.roots,
        flatten_details(model_coefficients),
    )

    leaves_by_expert = [
        _make_source(
            alphabet,
            EVALUATION_VECTORS,
            _seed(alphabet, structured, 4, expert),
            structured,
            hidden_permutation,
            hidden_selectors,
        )
        for expert in range(EXPERTS)
    ]
    packet = build_container(
        model,
        leaves_by_expert,
        [permutation] * EXPERTS,
        [selectors] * EXPERTS,
    )
    expected_digests = [leaf_digest(leaves) for leaves in leaves_by_expert]
    independent, decoded, rebuilt = verify_decode_reencode(packet, expected_digests)
    require(rebuilt == packet, "independent byte re-encode")
    require(
        all(np.array_equal(left, right) for left, right in zip(decoded, leaves_by_expert, strict=True)),
        "independent decoded leaves",
    )
    ledger = physical_ledger(packet)
    require(bool(ledger["cold_below_two"]), "cold amplification early kill")
    return (
        {
            "alphabet": alphabet,
            "structured": structured,
            "hidden_metadata_seed": HIDDEN_METADATA_SEED,
            "selected_metadata_seed": search.selected_seed,
            "hidden_candidate_selected": search.selected_seed == HIDDEN_METADATA_SEED,
            "search_candidates": list(search.candidate_rows),
            "gpu_telemetry": search.telemetry,
            "physical_ledger": ledger,
            "independent_receipt": independent,
            "population_suffix_oracle": population_suffix_limit(alphabet),
            "leaf_marginal_contract": "exactly uniform in population by uniform root and invertible modular lifting",
            "result_scope": "synthetic mechanism only; no model-source structure or gain claim",
        },
        packet,
    )


def run(output_dir: Path) -> dict[str, object]:
    require(not output_dir.exists(), "output directory must not exist")
    output_dir.mkdir(parents=False, exist_ok=False)
    rows: dict[str, dict[str, object]] = {}
    packets: dict[str, bytes] = {}
    for alphabet in (2, 4):
        for structured in (True, False):
            name = f"a{alphabet}_{'structured' if structured else 'control'}"
            row, packet = _run_class(alphabet, structured)
            rows[name] = row
            packets[name] = packet
            _bytes_write_new(output_dir / f"{name}.silt", packet)

    comparisons: dict[str, object] = {}
    pass_all = True
    for alphabet in (2, 4):
        structured = rows[f"a{alphabet}_structured"]
        control = rows[f"a{alphabet}_control"]
        structured_rate = float(structured["physical_ledger"]["physical_bits_per_leaf_symbol"])
        control_rate = float(control["physical_ledger"]["physical_bits_per_leaf_symbol"])
        gap = control_rate - structured_rate
        control_floor = 0.98 * math.log2(alphabet)
        conditions = {
            "finite_structured_control_gap_gt_0_15": gap > 0.15,
            "control_rate_ge_0_98_log2_alphabet": control_rate >= control_floor,
            "structured_cold_below_two": bool(structured["physical_ledger"]["cold_below_two"]),
            "control_cold_below_two": bool(control["physical_ledger"]["cold_below_two"]),
            "structured_cpu_cupy_equal": bool(
                structured["gpu_telemetry"]["cpu_cupy_selected_coefficients_equal"]
            ),
            "control_cpu_cupy_equal": bool(
                control["gpu_telemetry"]["cpu_cupy_selected_coefficients_equal"]
            ),
            "structured_independent_reencode": structured["independent_receipt"]["status"]
            == "PASS_INDEPENDENT_DECODE_REENCODE",
            "control_independent_reencode": control["independent_receipt"]["status"]
            == "PASS_INDEPENDENT_DECODE_REENCODE",
        }
        alphabet_pass = all(conditions.values())
        pass_all = pass_all and alphabet_pass
        comparisons[f"a{alphabet}"] = {
            "structured_physical_bits_per_leaf_symbol": structured_rate,
            "control_physical_bits_per_leaf_symbol": control_rate,
            "finite_structured_control_gap_bits_per_leaf_symbol": gap,
            "control_floor_bits_per_leaf_symbol": control_floor,
            "conditions": conditions,
            "status": "PASS_SYNTHETIC_MECHANISM_GATE" if alphabet_pass else "EARLY_KILL",
        }

    result = {
        "schema": SCHEMA,
        "status": "UNSEALED_SYNTHETIC_MECHANISM_PASS" if pass_all else "UNSEALED_EARLY_KILL",
        "source_gain_claim": False,
        "payload_authority": False,
        "canonical_fixture": {
            "lanes": LANES,
            "experts": EXPERTS,
            "evaluation_vectors": EVALUATION_VECTORS,
            "model_train_vectors": MODEL_TRAIN_VECTORS,
            "search_train_vectors": SEARCH_TRAIN_VECTORS,
            "search_validation_vectors": SEARCH_VALIDATION_VECTORS,
            "candidate_seeds": list(CANDIDATE_SEEDS),
            "detail_block": 32,
            "alphabets": [2, 4],
            "canonical_device": "RTX 5090 required and checked at runtime",
        },
        "rows": rows,
        "comparisons": comparisons,
        "artifacts": {
            name: {"bytes": len(packet), "sha256": sha256_bytes(packet)}
            for name, packet in packets.items()
        },
        "interpretation": (
            "A pass proves finite-stream, GPU-search, roundtrip, charged-byte, and cold-read "
            "mechanics on constructed synthetic sources only. It makes no claim about model weights."
        ),
    }
    _json_write_new(output_dir / "UNSEALED_RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new directory for synthetic-only unsealed artifacts; no input payload is accepted",
    )
    arguments = parser.parse_args()
    result = run(arguments.output_dir.resolve())
    print(json.dumps({"status": result["status"], "output_dir": str(arguments.output_dir.resolve())}))
    return 0 if result["status"] == "UNSEALED_SYNTHETIC_MECHANISM_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

