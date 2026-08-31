#!/usr/bin/env python3
"""Independent paired audit of exact and RHT POLARIS Qwen panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


N = 1 << 18
CAPACITY = 563_464
DIRECTORY_BITS = 48
HEADER_BITS = 768
GAUSSIAN = 2.0 ** (-2.0 * 2.15)
CEILING = 1.05 * GAUSSIAN
ROLE_POPULATION = {
    "expert_gate": 36_864,
    "expert_up": 36_864,
    "expert_down": 36_864,
    "attention_q": 1_536,
    "attention_k": 192,
    "attention_v": 192,
    "attention_o": 1_536,
    "embedding": 1_187,
    "lm_head": 1_187,
    "router": 48,
}
EXPECTED_HASHES = {
    "manifest": "3b882c74870c1e27bcddf7427e4c6ffea816d4f9847447eb218729ca69426a55",
    "exact_encoder": "95cfd32e5d026f07ceffe90daa7f88ca5e62f9f90546dfe74fc37cf06854d9b8",
    "rht_encoder": "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0",
    "packer": "c5fda34242153365dac07b5990bcd1fa19f0ac98d2512d47c3c8e1ec2a81dde8",
    "unpacker": "cf7113c3fbc6340f0870dadcf7608739aa651f5706befa163b5d13516dac7e07",
    "decoder": "2e1e484bf8ba98d493cfda55d4b23e275267e097e08907f5a9c606ae7350c797",
    "decoder_map": "a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-14) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def audit_variant(
    variant: str,
    directory: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    blocks = manifest["blocks"]
    role_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"blocks": 0.0, "energy": 0.0, "sse": 0.0, "logical": 0.0}
    )
    detail: list[dict[str, Any]] = []
    total_energy = 0.0
    total_sse = 0.0
    total_logical = 0
    for panel_index, source in enumerate(blocks):
        stem = f"block_{panel_index:03d}"
        metadata = load(directory / "encoded" / f"{stem}.encoder.json")
        decoded = load(directory / "decoded" / f"{stem}.decode.json")
        trial = metadata["trials"][0]
        aggregate = decoded["aggregation"]
        if decoded["passed"] is not True:
            raise AssertionError(f"{variant} decoder failure at {panel_index}")
        if decoded["audits"]["normalized_decoder_match"] is not True:
            raise AssertionError(f"{variant} normalized mismatch at {panel_index}")
        if decoded["source"]["bf16_sha256"] != source["source_bf16_sha256"]:
            raise AssertionError(f"{variant} source mismatch at {panel_index}")
        logical = int(trial["arithmetic_logical_bits"])
        if logical != int(decoded["codec"]["logical_arithmetic_bits"]):
            raise AssertionError(f"{variant} logical mismatch at {panel_index}")
        if trial["arithmetic_payload_sha256"] != decoded["codec"]["payload_sha256"]:
            raise AssertionError(f"{variant} payload mismatch at {panel_index}")
        preconditioner = decoded.get("preconditioner")
        if variant == "exact":
            if preconditioner is not None:
                raise AssertionError(f"exact block {panel_index} has preconditioner")
        else:
            if not isinstance(preconditioner, dict) or not (
                preconditioner.get("mode") == "hadamard_rademacher_splitmix64"
                and preconditioner.get("normalization") == "orthonormal"
                and int(preconditioner.get("seed_u64", -1))
                == int(source["rht_seed_u64"])
                and int(preconditioner.get("side_bits", -1)) == 0
            ):
                raise AssertionError(f"RHT binding mismatch at {panel_index}")
            source_metadata = trial["source"]
            if not (
                source_metadata["canonical_source_id"] == source["tensor"]
                and int(source_metadata["canonical_block_index"])
                == int(source["canonical_block_index"])
            ):
                raise AssertionError(f"RHT canonical mismatch at {panel_index}")
        energy = float(aggregate["source_energy_sum_fp64"])
        sse = float(aggregate["fp16_sse_sum_fp64"])
        role = source["role"]
        role_totals[role]["blocks"] += 1
        role_totals[role]["energy"] += energy
        role_totals[role]["sse"] += sse
        role_totals[role]["logical"] += logical
        total_energy += energy
        total_sse += sse
        total_logical += logical
        detail.append(
            {
                "panel_index": panel_index,
                "id": source["id"],
                "role": role,
                "energy": energy,
                "sse": sse,
                "relative_mse": sse / energy,
                "logical_bits": logical,
                "source_sha256": source["source_bf16_sha256"],
                "reconstruction_sha256": aggregate[
                    "final_reconstruction_fp64_sha256"
                ],
            }
        )

    pack = load(directory / "pack.audit.json")
    unpack = load(directory / "unpack.audit.json")
    reservoir = directory / "panel.plrsv2.bin"
    if pack.get("passed") is not True or unpack.get("validation") != "passed":
        raise AssertionError(f"{variant} serialization audit failed")
    if int(pack["rate"]["global_payload_headroom_bits"]) != len(blocks) * CAPACITY - total_logical:
        raise AssertionError(f"{variant} pack headroom mismatch")
    expected_bits = HEADER_BITS + len(blocks) * (CAPACITY + DIRECTORY_BITS)
    physical_bits = reservoir.stat().st_size * 8
    if physical_bits != expected_bits:
        raise AssertionError(f"{variant} physical size mismatch")
    if physical_bits * 20 > 43 * N * len(blocks):
        raise AssertionError(f"{variant} exceeds 2.15 physical bpw")

    summary = load(directory / "summary.json")
    recomputed_mse = total_sse / total_energy
    if not close(recomputed_mse, float(summary["aggregate"]["energy_weighted_relative_mse"])):
        raise AssertionError(f"{variant} summary MSE mismatch")
    if total_logical != int(summary["aggregate"]["logical_bits_sum"]):
        raise AssertionError(f"{variant} summary logical mismatch")
    expected_impl = {
        "encoder": EXPECTED_HASHES[f"{variant}_encoder"],
        "packer": EXPECTED_HASHES["packer"],
        "unpacker": EXPECTED_HASHES["unpacker"],
        "decoder": EXPECTED_HASHES["decoder"],
        "decoder_map": EXPECTED_HASHES["decoder_map"],
    }
    for key, value in expected_impl.items():
        if summary["hashes"][key] != value:
            raise AssertionError(f"{variant} implementation hash mismatch: {key}")
    if summary["hashes"]["reservoir"] != sha(reservoir):
        raise AssertionError(f"{variant} reservoir hash mismatch")

    by_role = {
        role: {
            "blocks": int(value["blocks"]),
            "energy": value["energy"],
            "sse": value["sse"],
            "energy_weighted_relative_mse": value["sse"] / value["energy"],
            "mean_logical_bits": value["logical"] / value["blocks"],
        }
        for role, value in sorted(role_totals.items())
    }
    projected_energy = sum(
        row["energy"] / row["blocks"] * ROLE_POPULATION[role]
        for role, row in by_role.items()
    )
    projected_sse = sum(
        row["sse"] / row["blocks"] * ROLE_POPULATION[role]
        for role, row in by_role.items()
    )
    projected_logical = sum(
        row["mean_logical_bits"] * ROLE_POPULATION[role]
        for role, row in by_role.items()
    ) / sum(ROLE_POPULATION.values())
    return {
        "blocks": len(blocks),
        "logical_bits_sum": total_logical,
        "logical_bits_mean": total_logical / len(blocks),
        "payload_headroom_bits": len(blocks) * CAPACITY - total_logical,
        "source_energy": total_energy,
        "fp16_sse": total_sse,
        "energy_weighted_relative_mse": recomputed_mse,
        "relative_excess_over_gaussian": recomputed_mse / GAUSSIAN - 1.0,
        "passes_5pct_gate": recomputed_mse <= CEILING,
        "blocks_individually_passing_5pct_gate": sum(
            row["relative_mse"] <= CEILING for row in detail
        ),
        "max_block_relative_mse": max(row["relative_mse"] for row in detail),
        "physical_bits": physical_bits,
        "physical_bpw": physical_bits / (len(blocks) * N),
        "reservoir_sha256": sha(reservoir),
        "by_role": by_role,
        "population_projection_descriptive": {
            "method": "expand each sampled role mean by the exact checkpoint role block population; deterministic coverage panel, no design-based confidence claim",
            "rank2_blocks": sum(ROLE_POPULATION.values()),
            "energy_weighted_relative_mse": projected_sse / projected_energy,
            "mean_logical_bits": projected_logical,
            "payload_fits": projected_logical <= CAPACITY,
        },
        "detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exact-dir", type=Path, required=True)
    parser.add_argument("--rht-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.manifest) != EXPECTED_HASHES["manifest"]:
        raise AssertionError("manifest hash mismatch")
    manifest = load(args.manifest)
    exact = audit_variant("exact", args.exact_dir, manifest)
    rht = audit_variant("rht", args.rht_dir, manifest)
    if not close(exact["source_energy"], rht["source_energy"], 0.0):
        raise AssertionError("paired source energies differ")
    pairs = []
    for left, right in zip(exact["detail"], rht["detail"], strict=True):
        if left["id"] != right["id"] or left["source_sha256"] != right["source_sha256"]:
            raise AssertionError("paired block identity differs")
        pairs.append(
            {
                "id": left["id"],
                "role": left["role"],
                "exact_relative_mse": left["relative_mse"],
                "rht_relative_mse": right["relative_mse"],
                "relative_mse_reduction": 1.0 - right["relative_mse"] / left["relative_mse"],
                "logical_bits_delta": right["logical_bits"] - left["logical_bits"],
            }
        )
    result = {
        "status": "independent paired audit passed",
        "manifest_sha256": EXPECTED_HASHES["manifest"],
        "gaussian_limit_at_2p15": GAUSSIAN,
        "gaussian_5pct_ceiling": CEILING,
        "exact": exact,
        "rht": rht,
        "paired": {
            "energy_weighted_mse_relative_reduction": (
                1.0 - rht["energy_weighted_relative_mse"] / exact["energy_weighted_relative_mse"]
            ),
            "logical_bits_mean_delta": rht["logical_bits_mean"] - exact["logical_bits_mean"],
            "rht_headroom_after_delta": rht["payload_headroom_bits"],
            "pairs": pairs,
        },
        "claim_boundary": (
            "The deterministic 32-block coverage panel spans every rank-2 role but is not "
            "a probability sample or full 116,470-block checkpoint census. The population "
            "projection is descriptive; no confidence interval or perplexity claim is made."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "exact_mse": exact["energy_weighted_relative_mse"],
        "rht_mse": rht["energy_weighted_relative_mse"],
        "rht_excess_over_gaussian": rht["relative_excess_over_gaussian"],
        "relative_mse_reduction": result["paired"]["energy_weighted_mse_relative_reduction"],
        "rht_physical_bpw": rht["physical_bpw"],
        "rht_payload_headroom_bits": rht["payload_headroom_bits"],
        "rht_individual_blocks_passing": rht["blocks_individually_passing_5pct_gate"],
        "rht_population_projection_mse": rht["population_projection_descriptive"]["energy_weighted_relative_mse"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
