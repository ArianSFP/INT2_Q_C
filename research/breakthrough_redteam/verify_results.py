#!/usr/bin/env python3
"""Independent arithmetic/hash verifier for the breakthrough-redteam screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


ROWS = 768
COLS = 2048
WEIGHTS = ROWS * COLS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 2e-12) -> None:
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"mismatch: {left!r} != {right!r}")


def verify_cut(result: dict, source_dir: Path | None) -> dict:
    if result["schema"] != "qwen-nanoquant-cut-factor-opportunity-v1":
        raise ValueError("bad cut-factor schema")
    accumulated: dict[int, dict[str, float]] = {}
    source_hashes = []
    for record in result["records"]:
        if source_dir is not None:
            source_path = source_dir / record["tensor"]
            observed = sha256_file(source_path)
            if observed != record["source_sha256"]:
                raise ValueError(f"source hash mismatch: {record['tensor']}")
            source_hashes.append({"tensor": record["tensor"], "sha256": observed})
        for domain in ("qwen", "gaussian"):
            energy = float(record[domain]["initial_sse"])
            for point in record[domain]["curve"]:
                rank = int(point["rank"])
                expected_bits = rank * (ROWS + COLS) + 16 * (ROWS + COLS)
                if int(point["physical_bits_nanoquant_formula"]) != expected_bits:
                    raise ValueError("factor ledger mismatch")
                close(float(point["physical_bpw_nanoquant_formula"]), expected_bits / WEIGHTS)
                if int(point["free_coefficient_bits_omitted"]) != 32 * rank:
                    raise ValueError("free coefficient ledger mismatch")
                row = accumulated.setdefault(
                    rank,
                    {"qwen_energy": 0.0, "qwen_error": 0.0, "gaussian_energy": 0.0, "gaussian_error": 0.0},
                )
                row[f"{domain}_energy"] += energy
                row[f"{domain}_error"] += energy * float(point["relative_mse"])

    aggregate_by_rank = {int(row["rank"]): row for row in result["aggregate_curve"]}
    for rank, totals in accumulated.items():
        aggregate = aggregate_by_rank[rank]
        qwen_d = totals["qwen_error"] / totals["qwen_energy"]
        gaussian_d = totals["gaussian_error"] / totals["gaussian_energy"]
        rate = (rank + 16) * (ROWS + COLS) / WEIGHTS
        close(float(aggregate["qwen_pooled_relative_mse"]), qwen_d)
        close(float(aggregate["gaussian_pooled_relative_mse"]), gaussian_d)
        close(float(aggregate["qwen_vs_gaussian_advantage_s_bpw"]), -0.5 * math.log2(qwen_d / gaussian_d))
        close(float(aggregate["qwen_absolute_F_vs_gaussian_limit"]), qwen_d * 2.0 ** (2.0 * rate))
        close(float(aggregate["gaussian_operational_F"]), gaussian_d * 2.0 ** (2.0 * rate))

    final = aggregate_by_rank[1380]
    return {
        "sources_rehashed": source_hashes,
        "final_rank": 1380,
        "final_physical_bpw": final["physical_bpw_nanoquant_formula"],
        "final_source_advantage_s_bpw": final["qwen_vs_gaussian_advantage_s_bpw"],
        "final_absolute_F": final["qwen_absolute_F_vs_gaussian_limit"],
        "decision": "kill",
    }


def verify_semantic(result: dict, dictionary_paths: dict[str, Path] | None) -> dict:
    if result["schema"] != "qwen-existing-semantic-dictionary-cosine-oracle-v1":
        raise ValueError("bad semantic schema")
    dictionaries = []
    for row in result["dictionaries"]:
        count = int(row["actual"]["dictionary_rows"])
        expected_index_bits = math.ceil(math.log2(count))
        if int(row["index_bits_fixed"]) != expected_index_bits:
            raise ValueError("index width mismatch")
        side = (expected_index_bits + 16) / COLS
        close(float(row["side_bpw_with_fp16_coefficient"]), side)
        required_energy = 1.0 - 2.0 ** (-2.0 * (0.15287192093 + side))
        close(float(row["required_explained_energy_to_close_increment"]), required_energy)
        close(float(row["required_absolute_cosine_if_uniform"]), math.sqrt(required_energy))
        predicted = 2.0 * math.log(2.0 * count) / COLS
        close(float(row["iid_random_extreme_prediction_energy"]), predicted)
        actual_energy = float(row["actual"]["pooled_explained_energy"])
        actual_s = -0.5 * math.log2(1.0 - actual_energy)
        close(float(row["actual"]["free_predictor_s_bpw"]), actual_s)
        rehash = None
        if dictionary_paths is not None:
            path = dictionary_paths[row["name"]]
            rehash = sha256_file(path)
            if rehash != row["sha256"]:
                raise ValueError(f"dictionary hash mismatch: {row['name']}")
        dictionaries.append(
            {
                "name": row["name"],
                "sha256_rehashed": rehash,
                "actual_explained_energy": actual_energy,
                "required_explained_energy": required_energy,
                "net_s_after_index_and_fp16_coefficient": actual_s - side,
                "decision": "kill",
            }
        )
    return {"dictionaries": dictionaries, "decision": "kill"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--embedding", type=Path)
    parser.add_argument("--attention", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cut_path = args.directory / "cut_factor_result.json"
    semantic_path = args.directory / "semantic_dictionary_result.json"
    protocol_path = args.directory / "protocol_freeze.json"
    cut = json.loads(cut_path.read_text(encoding="utf-8"))
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    json.loads(protocol_path.read_text(encoding="utf-8"))
    dictionary_paths = None
    if args.embedding is not None or args.attention is not None:
        if args.embedding is None or args.attention is None:
            raise ValueError("both dictionary paths are required together")
        dictionary_paths = {"embedding": args.embedding, "attention_q": args.attention}

    receipt = {
        "schema": "breakthrough-redteam-independent-verification-v1",
        "python": platform.python_version(),
        "artifacts": {
            path.name: sha256_file(path)
            for path in (protocol_path, cut_path, semantic_path)
        },
        "cut_factor": verify_cut(cut, args.source_dir),
        "semantic_dictionary": verify_semantic(semantic, dictionary_paths),
        "overall_decision": "no evidence-backed survivor",
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
