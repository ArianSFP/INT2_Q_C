#!/usr/bin/env python3
"""Verify that LiftQuant's Mq code is contained in binary additive VQ.

For binary additive books C[j,0], C[j,1], define

    a = 1/2 sum_j (C[j,0] + C[j,1])
    M[:,j] = 1/2 (C[j,1] - C[j,0])
    q[j] = 2 b[j] - 1.

Then sum_j C[j,b[j]] = a + M q.  Conversely, Mq is recovered by
C[j,0]=-M[:,j], C[j,1]=M[:,j].  Thus plain LiftQuant is a subset, and
affine LiftQuant is exactly the binary additive family already screened.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "72b3875c770e4579639931fed89dc95e4067edac"
EXPECTED_SOURCE_HASHES = {
    "README.md": "0e113b089e293b4e82e07962daf5e5f026d76b65f400d6b9cd2aab05c87dce6b",
    "lattice_generator2.py": "0914967462ec5e76ea27a4b38c7412082b9bbb61f0bef9ef1780178cfbcaaafd",
    "quantize/tmplinear.py": "ca4949d453b147501ed363efda1c56dc2cf3428628414ab719c5ecc5d8b27da9",
    "ICML2026_LiftQuant.pdf": "7065ebbde21fc8e7454aa249ec778ce06b2444f9d9f3f16bb42ad6c526107e01",
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_integer_equivalence_test() -> int:
    # Five binary books of three-dimensional integer vectors.  Integer halves
    # are avoided by comparing 2*reconstruction, making this an exact test.
    books = [
        [[2, -1, 4], [-3, 5, 0]],
        [[7, 2, -2], [1, -6, 3]],
        [[-4, 3, 5], [8, 0, -1]],
        [[6, -7, 2], [-2, 4, 9]],
        [[1, 8, -5], [5, -3, 7]],
    ]
    dimensions = len(books[0][0])
    offset_twice = [
        sum(book[0][axis] + book[1][axis] for book in books)
        for axis in range(dimensions)
    ]
    columns_twice = [
        [book[1][axis] - book[0][axis] for axis in range(dimensions)]
        for book in books
    ]
    checked = 0
    for bits in itertools.product((0, 1), repeat=len(books)):
        signs = [2 * bit - 1 for bit in bits]
        additive_twice = [
            2 * sum(books[stage][bits[stage]][axis] for stage in range(len(books)))
            for axis in range(dimensions)
        ]
        affine_twice = [
            offset_twice[axis]
            + sum(
                columns_twice[stage][axis] * signs[stage]
                for stage in range(len(books))
            )
            for axis in range(dimensions)
        ]
        if additive_twice != affine_twice:
            raise AssertionError("binary additive/LiftQuant equivalence failed")
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additive-result", required=True, type=Path)
    parser.add_argument("--additive-receipt", required=True, type=Path)
    parser.add_argument("--liftquant-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result_path = args.additive_result.resolve(strict=True)
    verification_path = args.additive_receipt.resolve(strict=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    result_hash = sha256_file(result_path)
    if verification.get("verified") is not True:
        raise ValueError("additive-VQ receipt is not verified")
    if verification.get("result_sha256") != result_hash:
        raise ValueError("additive-VQ result/receipt hash mismatch")
    if verification.get("decision") != "kill":
        raise ValueError("unexpected additive-VQ decision")

    official_source = None
    if args.liftquant_root is not None:
        root = args.liftquant_root.resolve(strict=True)
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != EXPECTED_COMMIT:
            raise ValueError(f"LiftQuant commit mismatch: {commit}")
        observed_hashes = {
            name: sha256_file((root / name).resolve(strict=True))
            for name in EXPECTED_SOURCE_HASHES
        }
        if observed_hashes != EXPECTED_SOURCE_HASHES:
            raise ValueError("LiftQuant source hash mismatch")
        official_source = {
            "repository": "https://github.com/Heliulu/LiftQuant",
            "commit": commit,
            "file_sha256": observed_hashes,
        }

    exact_cases = exact_integer_equivalence_test()
    binary_rows: list[dict[str, Any]] = []
    for row in result["results"]:
        if int(row["alphabet"]) != 2:
            continue
        matched_f = float(row["matched_F_source_over_control"])
        matched_s = float(row["matched_s_bpw_before_side_charge"])
        if not math.isclose(matched_f, 2.0 ** (-2.0 * matched_s), rel_tol=2e-13):
            raise ValueError("additive matched F/s identity failed")
        optimistic_s = float(row["optimistic_2se_s_bpw"])
        optimistic_f = float(row["optimistic_2se_F_identity"])
        if not math.isclose(optimistic_f, 2.0 ** (-2.0 * optimistic_s), rel_tol=2e-13):
            raise ValueError("additive optimistic F/s identity failed")
        binary_rows.append(
            {
                "architecture": row["architecture"],
                "dimension": int(row["dimension"]),
                "lifted_dimension_or_binary_stages": int(row["stages"]),
                "payload_rate_bpw": float(row["payload_rate_bpw"]),
                "physical_rate_bpw": float(row["physical_rate_bpw"]),
                "source_relative_mse": float(row["source_distortion"]),
                "matched_gaussian_relative_mse": float(
                    row["matched_gaussian_distortion"]
                ),
                "charged_matched_s_bpw": float(row["charged_matched_s_bpw"]),
                "optimistic_2se_s_bpw": optimistic_s,
                "optimistic_2se_F": optimistic_f,
                "source_codec_F": float(
                    row["exact_F_source_over_shannon_at_physical_rate"]
                ),
                "cold_expert_read_amplification": float(
                    row["cold_expert_read_amplification"]
                ),
            }
        )
    if [row["dimension"] for row in binary_rows] != [8, 16, 32]:
        raise ValueError("expected binary additive d=8/16/32 rows")

    receipt: dict[str, Any] = {
        "schema": "liftquant-additive-vq-subsumption-v1",
        "status": "NO_DISTINCT_EXPERIMENT_REQUIRED",
        "reference": {
            "paper": "https://arxiv.org/abs/2606.04050",
            "official_source": official_source,
        },
        "algebra": {
            "liftquant": "w_hat = M q, q_j in {-1,+1}",
            "binary_additive": "w_hat = sum_j C[j,b_j], b_j in {0,1}",
            "map_to_affine_liftquant": (
                "a=0.5 sum_j(C[j,0]+C[j,1]); "
                "M[:,j]=0.5(C[j,1]-C[j,0]); q_j=2b_j-1"
            ),
            "map_liftquant_to_binary_additive": (
                "C[j,0]=-M[:,j]; C[j,1]=M[:,j]; a=0"
            ),
            "relation": (
                "plain LiftQuant is a strict subset; affine LiftQuant equals the "
                "binary additive code family"
            ),
            "exact_integer_assignments_verified": exact_cases,
        },
        "existing_experiment": {
            "result_file_sha256": result_hash,
            "verification_receipt_sha256": sha256_file(verification_path),
            "plan_lock_sha256": verification["plan_lock_sha256"],
            "live_sources_verified": int(verification["verified_live_source_files"]),
            "representation": result["parameters"]["representation"],
            "binary_rows": binary_rows,
        },
        "decision": {
            "run_new_LiftQuant_beam_or_exact_screen": False,
            "reason": (
                "The requested code family is already contained in the verified "
                "cross-expert binary additive-VQ screen. Its binary rows have "
                "negative matched structural advantage even before the requested "
                "s=0.160964 threshold; another run would change search quality, "
                "not test a genuinely distinct architecture."
            ),
            "claim_boundary": (
                "The prior encoder used residual initialization plus coordinate "
                "sweeps, not globally exact nearest-hypercube search. This receipt "
                "establishes family subsumption and a strong negative screen, not "
                "an optimizer-independent converse for every M and every beam."
            ),
        },
    }
    receipt["receipt_lock_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    serialized = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
