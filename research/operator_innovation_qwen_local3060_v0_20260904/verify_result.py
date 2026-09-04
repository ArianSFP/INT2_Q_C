#!/usr/bin/env python3
"""Dependency-free consistency verifier for the operator-innovation aperture."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULT_SHA256 = "ae63d7a0dd931515eb64cefee3f30873c63100b179cb33084c78a374f4167a7b"
BLOCKWISE_SHA256 = "165b9e213867b36ee16d34f37b84a689c172d29a79d55985162420c40f49d3fd"
POST_SHA256 = "af801b41a37774d3f0ea65a00d929ff0004122caf4a5632457dbbe232e3f84d0"
CONTAINER_SHA256 = "4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b"
BASELINE_SSE = 500.39553685426534
BASELINE_ENERGY = 16192.89450885593
BASELINE_F = 0.9888693569009007
WEIGHTS = 28_311_552


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def close(a: float, b: float, tolerance: float = 3e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def main() -> int:
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    need(manifest["schema"] == "operator-innovation-qwen-local3060-v0-manifest",
         "manifest schema")
    expected_members = []
    for row in manifest["members"]:
        member_raw = (HERE / row["name"]).read_bytes()
        need(len(member_raw) == int(row["bytes"]) and sha(member_raw) == row["sha256"],
             f"manifest member {row['name']}")
        expected_members.append({"name": row["name"], "bytes": len(member_raw),
                                 "sha256": sha(member_raw)})
    need(sha(canonical(expected_members)) == manifest["source_root_sha256"],
         "manifest root")
    need({path.name for path in HERE.iterdir()} ==
         {row["name"] for row in manifest["members"]} | {"SOURCE_MANIFEST.json"},
         "package closure")
    path = HERE / "RESULT.json"
    raw = path.read_bytes()
    need(sha(raw) == RESULT_SHA256, "published result hash")
    result = json.loads(raw)
    need(result["schema"] == "operator-innovation-qwen-local3060-v0", "schema")
    unsigned = dict(result)
    declared = unsigned.pop("result_sha256_excluding_self")
    need(sha(canonical(unsigned)) == declared, "self hash")
    baseline = result["baseline"]
    # The probe intentionally materialises the audited FP64 reconstruction as
    # FP32 before CuPy feature GEMMs; its replay differs by <8e-7 absolute SSE.
    need(abs(float(baseline["sse"]) - BASELINE_SSE) < 2e-6, "baseline SSE")
    need(close(baseline["source_energy"], BASELINE_ENERGY), "source energy")
    need(close(baseline["relative_mse"], float(baseline["sse"]) / BASELINE_ENERGY),
         "relative MSE")
    need(close(baseline["F"], BASELINE_F), "baseline F")
    need(result["bindings"]["container_sha256"] == CONTAINER_SHA256, "container binding")
    need(result["bindings"]["decoded_post_sha256"] == POST_SHA256, "decoded binding")
    release = REPO / "results/qwen/strata_expert_affine_checkpoint"
    need(sha((release / "strata_expert_affine_n20n21.bin").read_bytes()) ==
         CONTAINER_SHA256, "local container")
    aggregates = result["aggregate"]
    replay_sse = float(baseline["sse"])
    need(set(aggregates) == {"scalar", "unary_cubic", "symmetry_aware_mixed",
                             "all_27_cubic", "all_27_cubic_plus_safe5"}, "banks")
    for name, row in aggregates.items():
        sse = float(row["source_fitted_sse"])
        capture = 1.0 - sse / replay_sse
        need(close(row["source_fitted_capture_fraction"], capture), f"capture {name}")
        count = int(row["source_fitted_coefficient_count"])
        side = 16.0 * count / WEIGHTS
        need(close(row["nominal_private_fp16_coefficient_bpw"], side), f"side {name}")
        expected_f = BASELINE_F * (sse / replay_sse) * 2.0 ** (2.0 * side)
        need(close(row["favourable_transfer_F"], expected_f), f"F {name}")
        loeo = 1.0 - float(row["leave_one_expert_out_sse"]) / replay_sse
        need(close(row["leave_one_expert_out_capture_fraction"], loeo), f"LOEO {name}")
    strongest = min(aggregates, key=lambda name: aggregates[name]["source_fitted_sse"])
    need(strongest == result["gate"]["strongest_bank"] ==
         "all_27_cubic_plus_safe5", "strongest bank")
    best = aggregates[strongest]
    need(float(best["source_fitted_capture_fraction"]) < 0.10, "kill threshold")
    need(float(best["favourable_transfer_F"]) > 0.8, "target miss")
    need(result["status"] ==
         "HARD_KILL_COMPACT_MIXED_OPERATOR_SPAN_BELOW_10_PERCENT_CAPTURE", "status")
    matrices = result["matrices"]
    need(len(matrices) == 18, "matrix coverage")
    need({int(row["expert_ordinal"]) for row in matrices} == set(range(6)),
         "expert coverage")
    need({row["role"] for row in matrices} == {"gate", "up", "down"}, "role coverage")
    full = [row["source_fitted"]["all_27_cubic_plus_safe5"] for row in matrices]
    need(all(int(row["effective_rank"]) == int(row["feature_count"]) == 30
             for row in full), "full-rank solves")
    block_raw = (HERE / "RESULT_BLOCKWISE.json").read_bytes()
    need(sha(block_raw) == BLOCKWISE_SHA256, "published blockwise result hash")
    blockwise = json.loads(block_raw)
    need(blockwise["schema"] ==
         "operator-innovation-blockwise-qwen-local3060-v0", "blockwise schema")
    block_unsigned = dict(blockwise)
    block_declared = block_unsigned.pop("result_sha256_excluding_self")
    need(sha(canonical(block_unsigned)) == block_declared, "blockwise self hash")
    block_sse = float(blockwise["baseline"]["sse"])
    need(abs(block_sse - float(baseline["sse"])) < 2e-9, "blockwise baseline")
    block_aggregates = blockwise["aggregate"]
    need(len(block_aggregates) == 20, "blockwise cell count")
    for name, row in block_aggregates.items():
        sse = float(row["sse"])
        capture = 1.0 - sse / block_sse
        need(close(row["capture_fraction"], capture), f"block capture {name}")
        side = 16.0 * int(row["coefficient_count"]) / WEIGHTS
        need(close(row["nominal_private_fp16_coefficient_bpw"], side),
             f"block side {name}")
        expected_f = BASELINE_F * sse / block_sse * 2.0 ** (2.0 * side)
        need(close(row["favourable_transfer_F"], expected_f), f"block F {name}")
    strongest_block = min(block_aggregates,
                          key=lambda name: block_aggregates[name]["sse"])
    need(strongest_block == blockwise["gate"]["strongest_cell"] ==
         "all_27_cubic_plus_safe5_rows32", "blockwise strongest")
    need(float(block_aggregates[strongest_block]["capture_fraction"]) < 0.10,
         "blockwise kill threshold")
    need(blockwise["status"] ==
         "HARD_KILL_BLOCKWISE_OPERATOR_SPAN_BELOW_10_PERCENT_CAPTURE",
         "blockwise status")
    print("PASS_OPERATOR_INNOVATION_QWEN_HARD_KILL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
