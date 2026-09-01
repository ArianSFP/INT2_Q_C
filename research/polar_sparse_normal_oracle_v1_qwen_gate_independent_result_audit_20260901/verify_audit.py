#!/usr/bin/env python3
"""Independent standard-library verifier for the PSNO-v1 result audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROWS = 768
COLS = 2048
VALUES = ROWS * COLS
MATRICES = 18
RATE = 2.5
TARGET_F = 0.8
BASE_SIDE = 0.0011701230649594908
REQUIRED_S = 0.16096404744368115
COMPOSITE_REQUIRED_S = 0.11356063456788208
EXPECTED = {
    "runner": "e5540d0e9beabb984af15ab569aceac8a29cc0be91286e595d51bfafa3704f08",
    "producer_result": "586d093aa6c556000a8b591b2b437a1e8e02cb0c379893bd64ec8ed406a14ff5",
    "producer_receipt": "cf652ccae60536bc281ea4a44f1c0fb86e0316b1bf347c0667fa2d3c42a0b176",
    "control_script": "fd86af76ef83a5db59293ab6d2139ea2881a25461fb05fdc0728a119424b70f9",
    "control_result": "76bcf7e52bb06f06e4b01652b886a45f071bfa86201dc7ead7e55a2221ed9f79",
    "parent_result": "e4fecac5f676d84739972bbf0e04467027aeae1356e62e1dc3cd2b84bff67026",
    "parent_script": "04d033319b4bbab037b48355e5f296274ae77b1c787ddcd2508e9b58948d265e",
    "source_lock": "bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23",
}


class Checks:
    def __init__(self) -> None:
        self.rows: list[str] = []

    def true(self, condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        self.rows.append(label)

    def equal(self, observed: Any, expected: Any, label: str) -> None:
        if observed != expected:
            raise AssertionError((label, observed, expected))
        self.rows.append(label)

    def close(
        self,
        observed: float,
        expected: float,
        label: str,
        *,
        absolute: float = 3e-13,
    ) -> None:
        if not math.isclose(float(observed), float(expected), rel_tol=3e-13, abs_tol=absolute):
            raise AssertionError((label, observed, expected))
        self.rows.append(label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def verify_seal(report: dict[str, Any], field: str, checks: Checks, label: str) -> str:
    clean = dict(report)
    claimed = clean.pop(field, None)
    observed = hashlib.sha256(canonical(clean)).hexdigest()
    checks.equal(claimed, observed, label)
    return observed


def group_count(family: str) -> int:
    fields = family.split("_")
    parameter = int(fields[3])
    if family.startswith("upper_triangular_tiles_"):
        tiles = (ROWS + parameter - 1) // parameter
        return tiles * (tiles + 1) // 2
    if family.startswith("diagonal_offset_segments_"):
        return sum((ROWS - offset + parameter - 1) // parameter for offset in range(ROWS))
    raise AssertionError(("family", family))


def exact_support(n: int, k: int) -> int:
    if not 0 <= k <= n:
        raise AssertionError((n, k))
    return (math.comb(n, k) - 1).bit_length()


def waterfill(components: list[tuple[float, float]], rate: float) -> tuple[float, float]:
    logv = [math.log2(energy / dimension) for dimension, energy in components]
    lower = max(logv) - 200.0
    upper = max(logv)
    for _ in range(300):
        level = 0.5 * (lower + upper)
        used = math.fsum(
            dimension * 0.5 * max(0.0, value - level)
            for (dimension, _), value in zip(components, logv, strict=True)
        )
        if used > rate:
            lower = level
        else:
            upper = level
    log_level = 0.5 * (lower + upper)
    level = 2.0**log_level
    used = math.fsum(
        dimension * 0.5 * max(0.0, value - log_level)
        for (dimension, _), value in zip(components, logv, strict=True)
    )
    distortion = math.fsum(
        dimension * level if value > log_level else energy
        for (dimension, energy), value in zip(components, logv, strict=True)
    )
    return distortion, used


def feasible(
    scan: dict[str, Any], records: list[dict[str, Any]], kept: list[int] | None = None
) -> dict[str, float | int]:
    indices = list(range(MATRICES)) if kept is None else list(kept)
    panel = len(indices) * VALUES
    total_energy = math.fsum(float(records[index]["source_energy"]) for index in indices)
    options = scan["best_evaluated_dual"]["selected_options"]
    components: list[tuple[float, float]] = []
    support_bits = 0
    symbols = 0
    groups = 0
    n = group_count(scan["family"])
    for index in indices:
        record = records[index]
        option = options[index]
        support = exact_support(n, int(option["selected_groups"]))
        if support != int(option["support_bits_exact"]):
            raise AssertionError((scan["family"], index, "support"))
        components.append(
            (
                int(record["model_dof"]) / panel,
                float(record["model_energy"]) / total_energy,
            )
        )
        residual = max(
            0.0,
            float(record["normal_energy"])
            * (1.0 - float(option["captured_normal_energy_fraction"])),
        )
        removed = int(option["removed_dof"])
        if residual > 0.0:
            components.append(
                (
                    max(1, int(record["normal_dof"]) - removed) / panel,
                    residual / total_energy,
                )
            )
        support_bits += support
        symbols += int(option["value_symbols"])
        groups += int(option["selected_groups"])
    support_bpw = support_bits / panel
    value_bpw = symbols / panel
    side = BASE_SIDE + support_bpw + value_bpw
    payload = RATE - side
    distortion, used = waterfill(components, payload)
    f_value = distortion * 32.0
    return {
        "F": f_value,
        "s_bpw": -0.5 * math.log2(f_value),
        "support_bits_total": support_bits,
        "value_symbols_total": symbols,
        "selected_groups_total": groups,
        "support_bpw": support_bpw,
        "value_bpw": value_bpw,
        "total_side_bpw": side,
        "payload_rate_bpw": payload,
        "used_payload_rate_bpw": used,
    }


def sample(values: list[float]) -> tuple[float, float, float]:
    mean = math.fsum(values) / len(values)
    sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return mean, sd, sd / math.sqrt(len(values))


def jackknife(values: list[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.sqrt((len(values) - 1) / len(values) * math.fsum((value - mean) ** 2 for value in values))


def verify_feasible(
    stored: dict[str, Any], replayed: dict[str, float | int], checks: Checks, label: str
) -> None:
    for field in ("support_bits_total", "value_symbols_total", "selected_groups_total"):
        checks.equal(int(stored[field]), int(replayed[field]), f"{label}:{field}")
    for field in (
        "F",
        "s_bpw",
        "support_bpw",
        "value_bpw",
        "total_side_bpw",
        "payload_rate_bpw",
        "used_payload_rate_bpw",
    ):
        checks.close(float(stored[field]), float(replayed[field]), f"{label}:{field}")
    checks.close(
        float(stored["payload_rate_bpw"]),
        float(stored["used_payload_rate_bpw"]),
        f"{label}:exact_rate",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    audit_dir = Path(__file__).resolve().parent
    producer_dir = repo / "research/polar_sparse_normal_oracle_v1_qwen_gate_20260901"
    parent_dir = repo / "research/polar_normal_predictor"
    checks = Checks()
    source_lock = repo / "blind_protocol_v2/unblinded/source_hashes.lock.json"
    if not source_lock.is_file():
        source_lock = repo.parent / "blind_protocol_v2/unblinded/source_hashes.lock.json"
    paths = {
        "runner": producer_dir / "run_gate.py",
        "producer_result": producer_dir / "result.json",
        "producer_receipt": producer_dir / "verification_receipt.json",
        "control_script": audit_dir / "replay_controls.py",
        "control_result": audit_dir / "control_replay.json",
        "parent_result": parent_dir / "result.json",
        "parent_script": parent_dir / "polar_normal_predictor.py",
        "source_lock": source_lock,
    }
    for label, expected in EXPECTED.items():
        path = paths[label]
        checks.true(path.is_file() and not path.is_symlink(), f"regular:{label}")
        checks.equal(sha256_file(path), expected, f"sha256:{label}")

    producer = json.loads(paths["producer_result"].read_text(encoding="utf-8"))
    producer_receipt = json.loads(paths["producer_receipt"].read_text(encoding="utf-8"))
    control = json.loads(paths["control_result"].read_text(encoding="utf-8"))
    parent = json.loads(paths["parent_result"].read_text(encoding="utf-8"))
    audit = json.loads((audit_dir / "audit_receipt.json").read_text(encoding="utf-8"))
    verify_seal(producer, "result_lock_sha256", checks, "producer_result_lock")
    verify_seal(producer_receipt, "receipt_lock_sha256", checks, "producer_receipt_lock")
    verify_seal(control, "result_lock_sha256", checks, "control_result_lock")
    verify_seal(parent, "result_lock_sha256", checks, "parent_result_lock")
    checks.equal(producer_receipt["verdict"], "PASS", "producer_receipt_PASS")
    checks.equal(producer_receipt["check_count"], 88, "producer_receipt_88")
    checks.equal(producer_receipt["result_file_sha256"], EXPECTED["producer_result"], "receipt_result_binding")
    checks.equal(producer["normal_records"], parent["source_normal_records"], "exact_prior_normal_records")

    runner_text = paths["runner"].read_text(encoding="utf-8")
    producer_readme = (producer_dir / "README.md").read_text(encoding="utf-8")
    parent_readme = (parent_dir / "README.md").read_text(encoding="utf-8")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        checks.true(variable not in runner_text, f"limitation_runner_omits:{variable}")
        checks.true(variable not in producer_readme, f"limitation_readme_omits:{variable}")
        checks.true(f"{variable}=4" in parent_readme, f"parent_contract:{variable}")
    control_source = paths["control_script"].read_text(encoding="utf-8")
    tree = ast.parse(control_source, filename="replay_controls.py")
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    checks.true(len(imports) > 0, "control_source_AST")
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        checks.true(f'"{variable}": "4"' in control_source, f"control_preflight:{variable}")
    checks.equal(
        control["audit"]["thread_contract"],
        {"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
        "control_thread_receipt",
    )

    records = producer["normal_records"]
    qwen = {row["family"]: row for row in control["qwen"]}
    controls = control["controls"]
    calibration = {row["family"]: row for row in control["calibration"]}
    receipt_rows = {row["family"]: row for row in audit["families"]}
    checks.equal(len(qwen), 8, "qwen_family_count")
    checks.equal(len(controls), 8, "control_replica_count")
    checks.equal(len(calibration), 8, "calibration_family_count")
    checks.equal(set(qwen), set(receipt_rows), "receipt_family_identity")
    qwen_delete: dict[str, list[float]] = {}
    control_delete: dict[str, list[list[float]]] = {family: [] for family in qwen}

    for family, row in qwen.items():
        replayed = feasible(row["dual"], records)
        verify_feasible(row["feasible_primal"], replayed, checks, f"qwen:{family}")
        checks.true(
            float(row["dual"]["best_evaluated_dual"]["dual_F_lower_bound"])
            <= float(replayed["F"]),
            f"weak_duality:{family}",
        )
        qwen_delete[family] = []
        for omitted in range(6):
            kept = [index for index in range(MATRICES) if index // 3 != omitted]
            qwen_delete[family].append(float(feasible(row["dual"], records, kept)["s_bpw"]))

    for replica, control_row in enumerate(controls):
        families = {row["family"]: row for row in control_row["families"]}
        checks.equal(set(families), set(qwen), f"control_family_identity:{replica}")
        for family, row in families.items():
            replayed = feasible(row["dual"], records)
            verify_feasible(
                row["feasible_primal"], replayed, checks, f"control:{replica}:{family}"
            )
            deletes = []
            for omitted in range(6):
                kept = [index for index in range(MATRICES) if index // 3 != omitted]
                deletes.append(float(feasible(row["dual"], records, kept)["s_bpw"]))
            control_delete[family].append(deletes)

    maximum_upper = -math.inf
    best_family = min(qwen, key=lambda family: float(qwen[family]["feasible_primal"]["F"]))
    passes = []
    for family, row in qwen.items():
        stored = row["feasible_primal"]
        cal = calibration[family]
        receipt = receipt_rows[family]
        control_family = [
            next(item for item in replica["families"] if item["family"] == family)
            for replica in controls
        ]
        control_f = [float(item["feasible_primal"]["F"]) for item in control_family]
        control_s = [float(item["feasible_primal"]["s_bpw"]) for item in control_family]
        mean_f, sd_f, se_f = sample(control_f)
        mean_s, sd_s, se_s = sample(control_s)
        checks.close(cal["control_mean_F"], mean_f, f"cal:{family}:meanF")
        checks.close(cal["control_sample_sd_F"], sd_f, f"cal:{family}:sdF")
        checks.close(cal["control_mc_se_F"], se_f, f"cal:{family}:seF")
        checks.close(cal["control_mean_s_bpw"], mean_s, f"cal:{family}:meanS")
        checks.close(cal["control_sample_sd_s_bpw"], sd_s, f"cal:{family}:sdS")
        checks.close(cal["control_mc_se_s_bpw"], se_s, f"cal:{family}:seS")
        delete_estimates = []
        for omitted in range(6):
            control_mean_delete = math.fsum(
                replica[omitted] for replica in control_delete[family]
            ) / len(controls)
            delete_estimates.append(qwen_delete[family][omitted] - control_mean_delete)
        delete_se = jackknife(delete_estimates)
        combined = math.hypot(se_s, delete_se)
        delta = float(stored["s_bpw"]) - mean_s
        lower = delta - 3.0 * combined
        upper = delta + 3.0 * combined
        checks.close(cal["delete_one_expert_jackknife_se_bpw"], delete_se, f"cal:{family}:deleteSE")
        checks.close(cal["combined_se_bpw"], combined, f"cal:{family}:combinedSE")
        checks.close(cal["qwen_minus_control_mean_s_bpw"], delta, f"cal:{family}:delta")
        checks.close(cal["conservative_delta_s_minus_3se_bpw"], lower, f"cal:{family}:lower")
        checks.close(cal["optimistic_delta_s_plus_3se_bpw"], upper, f"cal:{family}:upper")
        checks.equal(cal["null_fully_reproduces_qwen_at_3se"], lower <= 0.0, f"cal:{family}:null")
        for key, value in (
            ("qwen_F", stored["F"]),
            ("qwen_s_bpw", stored["s_bpw"]),
            ("selected_groups", stored["selected_groups_total"]),
            ("value_symbols", stored["value_symbols_total"]),
            ("support_bits", stored["support_bits_total"]),
            ("support_bpw", stored["support_bpw"]),
            ("value_bpw", stored["value_bpw"]),
            ("total_side_bpw", stored["total_side_bpw"]),
            ("payload_bpw", stored["payload_rate_bpw"]),
            ("control_mean_F", mean_f),
            ("control_mean_s_bpw", mean_s),
            ("delta_s_bpw", delta),
            ("combined_se_bpw", combined),
            ("delta_s_minus_3se_bpw", lower),
            ("delta_s_plus_3se_bpw", upper),
        ):
            if isinstance(value, int):
                checks.equal(receipt[key], value, f"receipt:{family}:{key}")
            else:
                checks.close(receipt[key], value, f"receipt:{family}:{key}")
        checks.equal(receipt["null_reproduced"], lower <= 0.0, f"receipt:{family}:null")
        maximum_upper = max(maximum_upper, upper)
        if float(stored["F"]) <= TARGET_F:
            passes.append(family)

    summary = audit["summary"]
    checks.equal(summary["best_apparent_family"], best_family, "summary:best_family")
    checks.close(summary["best_qwen_F"], qwen[best_family]["feasible_primal"]["F"], "summary:best_F")
    checks.close(summary["maximum_optimistic_source_specific_s_plus_3se_bpw"], maximum_upper, "summary:max_upper")
    checks.close(summary["standalone_shortfall_bpw"], REQUIRED_S - maximum_upper, "summary:standalone_gap")
    checks.close(summary["composite_incremental_shortfall_bpw"], COMPOSITE_REQUIRED_S - maximum_upper, "summary:composite_gap")
    checks.equal(summary["all_families_null_reproduced"], True, "summary:null_all")
    checks.equal(summary["finite_value_implementation_warranted"], False, "summary:no_finite")
    checks.equal(len(passes), 7, "seven_impossible_channel_passes")
    checks.equal(audit["verdict"], "KILL_QWEN_SPECIFIC_GROUPED_SPATIAL_SIGNAL_ROUTE", "verdict")

    expert_frame = 3 * VALUES * RATE / 8.0
    three_codebooks = 3 * 64 * 32 * 32 * 2
    checks.close(expert_frame, 1_474_560.0, "read:expert_frame_bytes", absolute=0.0)
    checks.close(three_codebooks, 393_216.0, "read:three_codebook_bytes", absolute=0.0)
    checks.close((expert_frame + three_codebooks) / expert_frame, 1.2666666666666666, "read:under_2x")

    print("PASS_POLAR_SPARSE_NORMAL_INDEPENDENT_RESULT_AUDIT")
    print(f"checks {len(checks.rows)}")
    print(f"best_family {best_family}")
    print(f"best_qwen_F {qwen[best_family]['feasible_primal']['F']}")
    print(f"maximum_optimistic_source_specific_s {maximum_upper}")
    print("verdict KILL_QWEN_SPECIFIC_GROUPED_SPATIAL_SIGNAL_ROUTE")


if __name__ == "__main__":
    main()
