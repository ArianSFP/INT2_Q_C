#!/usr/bin/env python3
"""Independent arithmetic/provenance verifier for the procedural oracle JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


WEIGHTS = 28_311_552
EXPERTS = 6
ROLES = 3
ROWS = 768
COLS = 2048
VALUES = ROWS * COLS
WEIGHTS_PER_EXPERT = ROLES * VALUES
SOURCE_BYTES = VALUES * 2
HEADER_BITS_PER_EXPERT = 512
SCALE_BITS_PER_BLOCK = 8
REQUIRED_S = -0.5 * math.log2(0.8)
SCHEMA = "qwen-procedural-subspace-oracle-v1"


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


def close(actual: float, expected: float, label: str, rel: float = 2e-12, abs_: float = 2e-14) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs_):
        raise ValueError(f"{label}: actual={actual!r}, expected={expected!r}")


def verify_seal(result: dict[str, Any]) -> str:
    clean = dict(result)
    claimed = clean.pop("result_lock_sha256", None)
    actual = hashlib.sha256(canonical_json(clean)).hexdigest()
    if claimed != actual:
        raise ValueError(f"result seal mismatch: claimed={claimed}, actual={actual}")
    return actual


def check_sources(result: dict[str, Any], plan_path: Path | None) -> dict[str, Any]:
    provenance = result["provenance"]
    sources = provenance["sources"]
    if len(sources) != EXPERTS * ROLES:
        raise ValueError("result does not bind exactly 18 sources")
    if [row["matrix_ordinal"] for row in sources] != list(range(EXPERTS * ROLES)):
        raise ValueError("source ordinals are not exactly 0..17")
    hashes = [row["sha256"] for row in sources]
    if len(set(hashes)) != len(hashes) or any(len(value) != 64 for value in hashes):
        raise ValueError("source hashes are missing, malformed, or duplicated")
    if sum(int(row["bytes"]) for row in sources) != WEIGHTS * 2:
        raise ValueError("source byte coverage mismatch")
    receipt: dict[str, Any] = {
        "source_count": len(sources),
        "source_bytes": sum(int(row["bytes"]) for row in sources),
        "all_source_hashes_unique": True,
        "plan_lock_sha256": provenance["plan_lock_sha256"],
        "plan_reopened": plan_path is not None,
    }
    if plan_path is not None:
        plan_path = plan_path.resolve(strict=True)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        clean = dict(plan)
        claimed = clean.pop("lock_sha256", None)
        actual = hashlib.sha256(canonical_json(clean)).hexdigest()
        if claimed != actual or actual != provenance["plan_lock_sha256"]:
            raise ValueError("plan seal/result binding mismatch")
        if sha256_file(plan_path) != provenance["plan_file_sha256"]:
            raise ValueError("plan file hash mismatch")
        if plan.get("coverage", {}).get("weights") != WEIGHTS:
            raise ValueError("plan weight coverage mismatch")
        source_root = Path(plan["source_root"]).resolve(strict=True)
        plan_rows = plan["sources"]
        if len(plan_rows) != len(sources):
            raise ValueError("plan/result source count mismatch")
        for expected, observed in zip(plan_rows, sources, strict=True):
            path = (source_root / expected["source_relpath"]).resolve(strict=True)
            if source_root not in path.parents:
                raise ValueError("source escaped plan root")
            actual_hash = sha256_file(path)
            if actual_hash != expected["source_bf16_sha256"] or actual_hash != observed["sha256"]:
                raise ValueError(f"source hash mismatch: {expected['tensor']}")
            if path.stat().st_size != SOURCE_BYTES or observed["bytes"] != SOURCE_BYTES:
                raise ValueError("source size mismatch")
            if expected["tensor"] != observed["tensor"]:
                raise ValueError("source tensor mismatch")
        receipt["all_18_sources_rehashed"] = True
        receipt["plan_path"] = str(plan_path)
        receipt["plan_file_sha256"] = sha256_file(plan_path)
    return receipt


def verify_candidate(candidate: dict[str, Any], protocol: dict[str, Any]) -> None:
    dimension = int(candidate["dimension"])
    k = int(candidate["coefficient_count"])
    seeds = int(candidate["library_seeds"])
    if dimension not in protocol["dimensions"] or candidate["representation"] not in protocol["representations"]:
        raise ValueError("candidate outside frozen grid")
    if candidate["family"] not in protocol["families"]:
        raise ValueError("candidate family outside frozen grid")
    if k not in protocol["coefficient_counts_by_dimension"][str(dimension)]:
        raise ValueError("candidate coefficient count outside frozen grid")
    if seeds not in protocol["seed_checkpoints"] or seeds <= 0 or seeds & (seeds - 1):
        raise ValueError("invalid candidate seed count")
    expected_id = (
        f"{candidate['representation']}:d{dimension}:k{k}:"
        f"{candidate['family']}:K{seeds}"
    )
    if candidate["candidate_id"] != expected_id:
        raise ValueError("candidate ID mismatch")
    if candidate["seed_bits_per_block"] != int(math.log2(seeds)):
        raise ValueError("seed bit count mismatch")
    expected_mode = "retained_basis" if k <= dimension // 2 else "excluded_complement"
    if candidate["evaluated_mode"] != expected_mode:
        raise ValueError("candidate projection mode mismatch")
    if candidate["evaluated_rank"] != min(k, dimension - k):
        raise ValueError("candidate evaluated rank mismatch")
    source = [float(value) for value in candidate["source_residual_sse_by_expert"]]
    gaussian = [float(value) for value in candidate["gaussian_residual_sse_by_expert"]]
    energy = [float(value) for value in candidate["sample_energy_by_expert"]]
    if not (len(source) == len(gaussian) == len(energy) == EXPERTS):
        raise ValueError("candidate expert vector length mismatch")
    if any(not (0.0 <= source[i] <= energy[i] and 0.0 <= gaussian[i] <= energy[i]) for i in range(EXPERTS)):
        raise ValueError("candidate SSE outside [0, energy]")
    total_energy = sum(energy)
    source_d = sum(source) / total_energy
    gaussian_d = sum(gaussian) / total_energy
    ratio = source_d / gaussian_d
    s_bpw = -0.5 * math.log2(ratio)
    close(candidate["source_relative_mse"], source_d, "candidate source D")
    close(candidate["gaussian_relative_mse"], gaussian_d, "candidate Gaussian D")
    close(candidate["source_over_matched_gaussian"], ratio, "candidate ratio")
    close(candidate["matched_percent_below_gaussian"], 100.0 * (1.0 - ratio), "candidate percent")
    close(candidate["matched_structural_advantage_bpw"], s_bpw, "candidate s")
    close(candidate["F_ratio_identity"], 2.0 ** (-2.0 * s_bpw), "candidate F identity")


def independent_rate(candidate: dict[str, Any], requested_rate: float) -> dict[str, Any]:
    dimension = int(candidate["dimension"])
    k = int(candidate["coefficient_count"])
    blocks = WEIGHTS_PER_EXPERT // dimension
    physical_bytes = math.ceil(WEIGHTS_PER_EXPERT * requested_rate / 8.0)
    physical_bits = 8 * physical_bytes
    physical_rate = physical_bits / WEIGHTS_PER_EXPERT
    seed_bits = int(candidate["seed_bits_per_block"])
    coefficient_payload = physical_bits - HEADER_BITS_PER_EXPERT - blocks * (
        seed_bits + SCALE_BITS_PER_BLOCK
    )
    q = coefficient_payload / (blocks * k)
    factor = 2.0 ** (-2.0 * q) if q >= 0.0 else 1.0
    source_residual = [float(x) for x in candidate["source_residual_sse_by_expert"]]
    gaussian_residual = [float(x) for x in candidate["gaussian_residual_sse_by_expert"]]
    energy = [float(x) for x in candidate["sample_energy_by_expert"]]
    source_sse = [source_residual[i] + (energy[i] - source_residual[i]) * factor for i in range(EXPERTS)]
    gaussian_sse = [gaussian_residual[i] + (energy[i] - gaussian_residual[i]) * factor for i in range(EXPERTS)]
    source_d = sum(source_sse) / sum(energy)
    gaussian_d = sum(gaussian_sse) / sum(energy)
    ratio = source_d / gaussian_d
    s_bpw = -0.5 * math.log2(ratio)
    gaussian_limit = 2.0 ** (-2.0 * physical_rate)
    return {
        "physical_bytes": physical_bytes,
        "physical_bits": physical_bits,
        "physical_rate": physical_rate,
        "blocks": blocks,
        "seed_bits": seed_bits,
        "coefficient_payload": coefficient_payload,
        "q": q,
        "factor": factor,
        "source_sse": source_sse,
        "gaussian_sse": gaussian_sse,
        "source_d": source_d,
        "gaussian_d": gaussian_d,
        "ratio": ratio,
        "s": s_bpw,
        "gaussian_limit": gaussian_limit,
        "F": source_d / gaussian_limit,
    }


def verify_rate_record(candidate: dict[str, Any], record: dict[str, Any], requested_rate: float) -> None:
    expected = independent_rate(candidate, requested_rate)
    exact = {
        "physical_bytes_per_expert": expected["physical_bytes"],
        "physical_bits_per_expert": expected["physical_bits"],
        "weights_per_expert": WEIGHTS_PER_EXPERT,
        "blocks_per_expert": expected["blocks"],
        "private_header_bits_per_expert": HEADER_BITS_PER_EXPERT,
        "framing_table_bits_per_expert": HEADER_BITS_PER_EXPERT,
        "shared_table_bits": 0,
        "seed_bits_per_block": expected["seed_bits"],
        "scale_bits_per_block": SCALE_BITS_PER_BLOCK,
        "coefficient_payload_bits_per_expert": expected["coefficient_payload"],
        "cold_expert_payload_bytes": expected["physical_bytes"],
        "cold_expert_bytes_read": expected["physical_bytes"],
    }
    for key, value in exact.items():
        if record[key] != value:
            raise ValueError(f"rate exact field mismatch: {key}")
    for key, value in {
        "requested_rate_bpw": requested_rate,
        "physical_rate_bpw": expected["physical_rate"],
        "ideal_coefficient_bits_each": expected["q"],
        "ideal_coefficient_distortion_factor": expected["factor"],
        "optimistic_source_relative_mse": expected["source_d"],
        "optimistic_matched_gaussian_relative_mse": expected["gaussian_d"],
        "source_over_matched_gaussian": expected["ratio"],
        "matched_structural_advantage_bpw": expected["s"],
        "F_ratio_identity": 2.0 ** (-2.0 * expected["s"]),
        "gaussian_assumed_limit": expected["gaussian_limit"],
        "source_F_equals_D_times_2pow2R": expected["F"],
        "cold_expert_read_amplification": 1.0,
    }.items():
        close(record[key], value, f"rate field {key}")
    for observed, expected_vector, label in (
        (record["optimistic_source_sse_by_expert"], expected["source_sse"], "source rate SSE"),
        (record["optimistic_gaussian_sse_by_expert"], expected["gaussian_sse"], "Gaussian rate SSE"),
    ):
        for i, (actual, target) in enumerate(zip(observed, expected_vector, strict=True)):
            close(actual, target, f"{label} expert {i}")
    if record["passes_twenty_percent_below_gaussian_limit"] != (expected["F"] <= 0.8):
        raise ValueError("rate pass/fail boolean mismatch")
    if record["cold_read_strictly_below_2x"] is not True:
        raise ValueError("cold-read gate mismatch")


def verify_free_crossfit(candidates: list[dict[str, Any]], record: dict[str, Any]) -> None:
    by_id = {row["candidate_id"]: row for row in candidates}
    total_source = total_gaussian = total_energy = 0.0
    for heldout, selection in enumerate(record["selections"]):
        if selection["heldout_expert"] != heldout:
            raise ValueError("free crossfit heldout ordering mismatch")
        def key(candidate: dict[str, Any]) -> tuple[float, str]:
            source = sum(value for i, value in enumerate(candidate["source_residual_sse_by_expert"]) if i != heldout)
            gaussian = sum(value for i, value in enumerate(candidate["gaussian_residual_sse_by_expert"]) if i != heldout)
            return source / gaussian, candidate["candidate_id"]
        winner = min(candidates, key=key)
        if selection["selected_candidate_id"] != winner["candidate_id"]:
            raise ValueError("free crossfit selection mismatch")
        if winner["candidate_id"] not in by_id:
            raise ValueError("unknown free crossfit selection")
        close(selection["training_source_over_gaussian"], key(winner)[0], "free training ratio")
        for field, vector in (
            ("heldout_source_sse", winner["source_residual_sse_by_expert"]),
            ("heldout_gaussian_sse", winner["gaussian_residual_sse_by_expert"]),
            ("heldout_energy", winner["sample_energy_by_expert"]),
        ):
            close(selection[field], vector[heldout], f"free {field}")
        total_source += float(selection["heldout_source_sse"])
        total_gaussian += float(selection["heldout_gaussian_sse"])
        total_energy += float(selection["heldout_energy"])
    source_d = total_source / total_energy
    gaussian_d = total_gaussian / total_energy
    ratio = source_d / gaussian_d
    s_bpw = -0.5 * math.log2(ratio)
    close(record["pooled_heldout_source_relative_mse"], source_d, "free LOO source D")
    close(record["pooled_heldout_gaussian_relative_mse"], gaussian_d, "free LOO Gaussian D")
    close(record["pooled_heldout_source_over_gaussian"], ratio, "free LOO ratio")
    close(record["pooled_heldout_structural_advantage_bpw"], s_bpw, "free LOO s")
    close(record["F_ratio_identity"], 2.0 ** (-2.0 * s_bpw), "free LOO F identity")


def verify_rate_crossfit(
    candidates: list[dict[str, Any]], requested_rate: float, record: dict[str, Any]
) -> None:
    accounted = [(candidate, independent_rate(candidate, requested_rate)) for candidate in candidates]
    total_source = total_gaussian = total_energy = 0.0
    physical_rate: float | None = None
    for heldout, selection in enumerate(record["selections"]):
        if selection["heldout_expert"] != heldout:
            raise ValueError("rate crossfit heldout ordering mismatch")
        def key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, str]:
            candidate, rate = item
            source = sum(value for i, value in enumerate(rate["source_sse"]) if i != heldout)
            energy = sum(value for i, value in enumerate(candidate["sample_energy_by_expert"]) if i != heldout)
            return source / energy, candidate["candidate_id"]
        candidate, rate = min(accounted, key=key)
        if selection["selected_candidate_id"] != candidate["candidate_id"]:
            raise ValueError("rate crossfit selection mismatch")
        close(selection["training_source_relative_mse"], key((candidate, rate))[0], "rate training D")
        close(selection["heldout_source_sse"], rate["source_sse"][heldout], "rate heldout source SSE")
        close(selection["heldout_gaussian_sse"], rate["gaussian_sse"][heldout], "rate heldout Gaussian SSE")
        close(selection["heldout_energy"], candidate["sample_energy_by_expert"][heldout], "rate heldout energy")
        close(selection["physical_rate_bpw"], rate["physical_rate"], "rate heldout R")
        close(selection["cold_expert_read_amplification"], 1.0, "rate heldout read amp")
        total_source += rate["source_sse"][heldout]
        total_gaussian += rate["gaussian_sse"][heldout]
        total_energy += candidate["sample_energy_by_expert"][heldout]
        physical_rate = rate["physical_rate"] if physical_rate is None else physical_rate
        close(rate["physical_rate"], physical_rate, "fold physical R")
    assert physical_rate is not None
    source_d = total_source / total_energy
    gaussian_d = total_gaussian / total_energy
    ratio = source_d / gaussian_d
    s_bpw = -0.5 * math.log2(ratio)
    gaussian_limit = 2.0 ** (-2.0 * physical_rate)
    f_source = source_d / gaussian_limit
    for key, value in {
        "requested_rate_bpw": requested_rate,
        "physical_rate_bpw": physical_rate,
        "pooled_heldout_source_relative_mse": source_d,
        "pooled_heldout_matched_gaussian_relative_mse": gaussian_d,
        "pooled_heldout_source_over_matched_gaussian": ratio,
        "pooled_heldout_structural_advantage_bpw": s_bpw,
        "F_ratio_identity": 2.0 ** (-2.0 * s_bpw),
        "gaussian_assumed_limit": gaussian_limit,
        "source_F_equals_D_times_2pow2R": f_source,
        "cold_expert_read_amplification": 1.0,
    }.items():
        close(record[key], value, f"rate crossfit {key}")
    if record["passes_twenty_percent_below_gaussian_limit"] != (f_source <= 0.8):
        raise ValueError("rate crossfit pass boolean mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--algorithm", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result_path = args.result.resolve(strict=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != SCHEMA:
        raise ValueError("wrong result schema")
    result_lock = verify_seal(result)
    if result["gpu_policy"] != {
        "cpu_only": True,
        "imports_cupy": False,
        "imports_torch": False,
        "invokes_cuda": False,
    }:
        raise ValueError("CPU-only policy receipt mismatch")
    target = result["target"]
    close(target["required_F"], 0.8, "required F")
    close(target["required_structural_advantage_bpw"], REQUIRED_S, "required s")
    close(target["identity_s_equals_minus_half_log2_F"], REQUIRED_S, "target identity")
    source_receipt = check_sources(result, args.plan)
    if args.algorithm is not None:
        algorithm = args.algorithm.resolve(strict=True)
        if sha256_file(algorithm) != result["provenance"]["algorithm_sha256"]:
            raise ValueError("algorithm source hash mismatch")

    candidates = result["candidates"]
    if result["candidate_count"] != len(candidates) or len(candidates) == 0:
        raise ValueError("candidate count mismatch")
    ids = [row["candidate_id"] for row in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate IDs")
    for candidate in candidates:
        verify_candidate(candidate, result["protocol"])

    expected_best = max(
        candidates,
        key=lambda row: (row["matched_structural_advantage_bpw"], row["candidate_id"]),
    )
    observed_best = result["favorable_free_oracle"]["panel_leaky_best"]
    if observed_best["candidate_id"] != expected_best["candidate_id"]:
        raise ValueError("panel-leaky free best mismatch")
    verify_free_crossfit(candidates, result["favorable_free_oracle"]["leave_one_expert_out"])

    for row in result["optimistic_rate_accounted"]:
        requested_rate = float(row["requested_rate_bpw"])
        accounted = [(candidate, independent_rate(candidate, requested_rate)) for candidate in candidates]
        candidate, rate = min(
            accounted,
            key=lambda item: (item[1]["F"], item[0]["candidate_id"]),
        )
        if row["panel_leaky_best_candidate_id"] != candidate["candidate_id"]:
            raise ValueError("panel-leaky rate best mismatch")
        verify_rate_record(candidate, row["panel_leaky_optimistic_accounting"], requested_rate)
        verify_rate_crossfit(candidates, requested_rate, row["leave_one_expert_out"])

    max_s = float(expected_best["matched_structural_advantage_bpw"])
    loo_s = float(
        result["favorable_free_oracle"]["leave_one_expert_out"]
        ["pooled_heldout_structural_advantage_bpw"]
    )
    expected_kill = max_s < REQUIRED_S / 4.0 and loo_s < REQUIRED_S / 4.0
    if result["decision"]["early_kill"] != expected_kill:
        raise ValueError("early-kill boolean mismatch")
    if result["status"] != ("EARLY_KILL" if expected_kill else "SURVIVES_FAVORABLE_SCREEN"):
        raise ValueError("status/decision mismatch")

    receipt: dict[str, Any] = {
        "schema": "qwen-procedural-subspace-verification-receipt-v1",
        "verified": True,
        "result_path": str(result_path),
        "result_file_sha256": sha256_file(result_path),
        "result_lock_sha256": result_lock,
        "candidate_count": len(candidates),
        "source_provenance": source_receipt,
        "panel_leaky_free_s_bpw": max_s,
        "loo_free_s_bpw": loo_s,
        "required_s_bpw": REQUIRED_S,
        "early_kill": expected_kill,
        "all_F_identities_recomputed": True,
        "all_rate_ledgers_recomputed": True,
        "all_crossfit_selections_recomputed": True,
    }
    receipt["receipt_lock_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
