#!/usr/bin/env python3
"""Standard-library verifier for the sealed KBVQ-IDRE source package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST = "PACKAGE_MANIFEST.json"
EXPECTED_FILES = {
    "README.md",
    "design_lock.json",
    "prior_evidence.json",
    "source_only_receipt.json",
    "stage0_gate.py",
    "test_source_only.py",
    "verify_design.py",
    MANIFEST,
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPERTS, ROLES, ROWS, COLS = 128, 2, 768, 2048
N_MATRIX = ROWS * COLS
N_LAYER = EXPERTS * ROLES * N_MATRIX
PAGE, SHARED_HEADER, EXPERT_HEADER = 4096, 4096, 512
RATES = (Fraction(43, 20), Fraction(23, 10), Fraction(5, 2))


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: Any, label: str) -> None:
        self.count += 1
        if not condition:
            raise AssertionError(f"check {self.count} failed: {label}")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out

    def finite(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value

    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=pairs,
        parse_float=finite,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError("nonfinite token " + token)),
    )


def held_regular_read(path: Path) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError("not a regular non-link file: " + str(path))
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev, opened.st_ino, opened.st_size
        ):
            raise ValueError("file changed before held read")
        parts = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            parts.append(block)
        after = os.fstat(descriptor)
        if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("file changed during held read")
        raw = b"".join(parts)
        if len(raw) != opened.st_size:
            raise ValueError("short held read")
        return raw
    finally:
        os.close(descriptor)


def close(actual: float, expected: float, tolerance: float = 3e-14) -> bool:
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def shared_bytes(mode: str, rank: int) -> int:
    copies = 1 if mode == "joint_role" else ROLES
    return SHARED_HEADER + 2 * copies * COLS * rank


def private_bytes(rank: int) -> int:
    return 2 * ROLES * ROWS * rank


def union_pages(shared: int, start: int, frame: int) -> int:
    shared_last = (shared - 1) // PAGE
    first = start // PAGE
    last = (start + frame - 1) // PAGE
    overlap = max(0, min(shared_last, last) - max(0, first) + 1)
    return shared_last + 1 + last - first + 1 - overlap


def layout(mode: str, rank: int, cap: Fraction) -> dict[str, Any] | None:
    shared = shared_bytes(mode, rank)
    cap_bytes = (N_LAYER * cap.numerator) // (8 * cap.denominator)
    frame = (cap_bytes - shared) // EXPERTS
    private = private_bytes(rank)
    residual = frame - EXPERT_HEADER - private
    if frame <= 0 or residual < 0:
        return None
    emitted = shared + EXPERTS * frame
    worst = max(union_pages(shared, shared + expert * frame, frame) for expert in range(EXPERTS))
    amp = worst * PAGE / (emitted / EXPERTS)
    side = 8.0 * (shared + EXPERTS * (EXPERT_HEADER + private)) / N_LAYER
    payload = 8.0 * EXPERTS * residual / N_LAYER
    actual = 8.0 * emitted / N_LAYER
    return {
        "legal": amp < 2.0,
        "shared": shared,
        "frame": frame,
        "residual": residual,
        "emitted": emitted,
        "worst_pages": worst,
        "amp": amp,
        "side": side,
        "payload": payload,
        "actual": actual,
    }


def waterfill(energies: list[float], dimensions: list[int], bits: float) -> dict[str, Any]:
    variances = [energy / dimension for energy, dimension in zip(energies, dimensions)]
    lo = min(variances) * 2.0 ** -120
    hi = max(variances)
    for _ in range(192):
        theta = math.sqrt(lo * hi)
        used = 0.5 * sum(
            dimension * max(math.log2(variance / theta), 0.0)
            for variance, dimension in zip(variances, dimensions)
        )
        if used > bits:
            lo = theta
        else:
            hi = theta
    rates = [0.5 * max(math.log2(variance / hi), 0.0) for variance in variances]
    allocated = sum(dimension * rate for dimension, rate in zip(dimensions, rates))
    distortion = sum(
        dimension * min(variance, hi)
        for variance, dimension in zip(variances, dimensions)
    )
    return {"variances": variances, "theta": hi, "rates": rates,
            "allocated": allocated, "distortion": distortion}


def verify(package: Path) -> dict[str, Any]:
    c = Checks()
    package = package.resolve(strict=True)
    c.require(package.is_dir() and not package.is_symlink(), "package directory")
    names = {entry.name for entry in package.iterdir()}
    c.require(names == EXPECTED_FILES, "exact eight-file closure")
    c.require(all(entry.is_file() and not entry.is_symlink() for entry in package.iterdir()),
              "regular non-link closure")
    held = {name: held_regular_read(package / name) for name in sorted(EXPECTED_FILES)}

    manifest = strict_json(held[MANIFEST])
    c.require(manifest.get("schema") == "kbvq_idre_raw_mse_gate_manifest_v0", "manifest schema")
    c.require(manifest.get("closed_world") is True, "closed-world manifest")
    entries = manifest.get("entries")
    c.require(isinstance(entries, list) and len(entries) == 7, "seven governed entries")
    c.require([row.get("path") for row in entries] == sorted(EXPECTED_FILES - {MANIFEST}),
              "manifest exact sorted paths")
    for row in entries:
        name = row["path"]
        c.require(PurePosixPath(name).name == name and name not in {".", ".."},
                  "safe manifest path " + name)
        c.require(row.get("bytes") == len(held[name]), "manifest bytes " + name)
        c.require(bool(HEX64.fullmatch(str(row.get("sha256")))) and
                  row["sha256"] == sha256(held[name]), "manifest digest " + name)

    design = strict_json(held["design_lock.json"])
    evidence = strict_json(held["prior_evidence.json"])
    receipt = strict_json(held["source_only_receipt.json"])
    c.require(design.get("schema") == "kbvq_idre_raw_mse_gate_design_v0", "design schema")
    c.require(design.get("status") ==
              "FROZEN_SOURCE_ONLY_BOUNDED_TWO_ROLE_WATERFILL_KILL_NO_GPU_OR_PAYLOAD_REPLAY",
              "source-only design status")
    c.require(evidence.get("schema") == "kbvq_idre_raw_mse_prior_sufficient_statistic_v0",
              "evidence schema")
    c.require(receipt.get("schema") == "kbvq_idre_raw_mse_gate_source_receipt_v0", "receipt schema")
    c.require(receipt.get("verdict") == "SEALED_SOURCE_ONLY_NO_GPU_OR_QWEN_PAYLOAD_REPLAY",
              "receipt verdict")
    c.require(receipt.get("package_file_count") == 8, "receipt closure count")
    c.require(receipt["access"]["numeric_qwen_payload_files_opened"] == 0 and
              receipt["access"]["gpu_jobs_submitted"] == 0, "receipt no payload/GPU")

    objective = design["objective"]
    required_s = -0.5 * math.log2(0.8)
    c.require(close(objective["required_s_min_bpw"], required_s), "required s arithmetic")
    c.require(objective["physical_rate_interval_bpw"] == [2.15, 2.5], "rate interval")
    c.require(objective["cold_page_read_amplification_max_exclusive"] == 2.0,
              "read threshold")
    c.require(objective["all_shared_private_header_padding_bytes_charged"] is True,
              "all bytes charged")
    sources = design["primary_sources"]
    c.require(len(sources) == 2 and sources[0]["url"].startswith("https://proceedings.iclr.cc/")
              and sources[1]["url"] == "https://github.com/xuzukang/kbvq_moe",
              "primary-only literature sources")
    specialization = design["raw_mse_specialization"]
    c.require("identity" in specialization["specialization"].lower() and
              specialization["BCOS_in_scope"] is False, "raw-MSE specialization/BCOS boundary")

    panel = design["auxiliary_panel"]
    c.require(panel["authorized_manifest_sha256"] ==
              "4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782",
              "auxiliary manifest pin")
    fit, validation = set(panel["fit_experts"]), set(panel["untouched_validation_experts"])
    c.require(len(fit) == 12 and len(validation) == 4 and not fit & validation,
              "whole-expert split")
    c.require(fit | validation == set(range(0, 128, 8)), "split closure")
    c.require(panel["pinned_target_panel_authorized"] is False, "pinned panel forbidden")
    c.require(panel["canonical_shapes"] == {"up": [ROWS, COLS], "down": [ROWS, COLS]},
              "canonical geometry")

    container = design["container"]
    c.require(container["weights_per_expert"] == ROLES * N_MATRIX and
              container["weights_per_layer"] == N_LAYER, "container geometry")
    expected_max = {
        "joint_role": {"2.15": 205, "2.30": 219, "2.50": 239},
        "role_specific": {"2.15": 102, "2.30": 109, "2.50": 119},
    }
    observed_max: dict[str, dict[str, int]] = {}
    for mode in ("joint_role", "role_specific"):
        observed_max[mode] = {}
        for cap in RATES:
            legal = [rank for rank in range(1, 257)
                     if (row := layout(mode, rank, cap)) is not None and row["legal"]]
            key = f"{float(cap):.2f}"
            observed_max[mode][key] = max(legal)
    c.require(observed_max == expected_max == container["precomputed_max_legal_rank"],
              "exact max legal ranks")
    c.require(container["global_max_legal_rank"] == 239 < 256, "rank envelope")

    prior = design["prior_sufficient_statistic"]
    c.require(prior["artifact_sha256"] == evidence["source"]["sha256"] and
              prior["independent_verification_sha256"] == evidence["independent_verification"]["sha256"],
              "design/evidence dependency pins")
    dependency_result = (package / prior["artifact"]).resolve(strict=True)
    dependency_receipt = (package / prior["independent_verification"]).resolve(strict=True)
    c.require(sha256(held_regular_read(dependency_result)) == prior["artifact_sha256"],
              "external aggregate result digest")
    c.require(sha256(held_regular_read(dependency_receipt)) == prior["independent_verification_sha256"],
              "external verification digest")
    dependency_verdict = strict_json(held_regular_read(dependency_receipt))
    c.require(dependency_verdict.get("status") == "PASS" and
              dependency_verdict.get("sources_verified") == 32, "external independent PASS")

    envelopes = evidence["rank256_exact_free_envelopes"]
    c.require([row["mode"] for row in envelopes] == ["joint_role", "role_specific"],
              "free envelope mode order")
    for row in envelopes:
        total = row["up_source_energy"] + row["down_source_energy"]
        captured = row["up_projected_energy"] + row["down_projected_energy"]
        capture = captured / total
        q = 1.0 - capture
        s = -0.5 * math.log2(q)
        c.require(close(row["pooled_capture"], capture) and
                  close(row["residual_energy_ratio_q"], q) and close(row["free_s_bpw"], s),
                  "rank256 gross capture arithmetic " + row["mode"])
        c.require(row["passes_required_s"] is False and s < required_s,
                  "gross free capture below target " + row["mode"])
    c.require("not physical" in evidence["free_capture_interpretation"]["warning"].lower(),
              "free capture warning")

    best = evidence["physical_two_role_waterfill_best"]
    ledger = layout(best["mode"], best["rank"], Fraction(str(best["requested_cap_bpw"])))
    assert ledger is not None
    c.require(ledger["legal"] and close(ledger["actual"], best["actual_bpw"]) and
              close(ledger["side"], best["component_header_side_bpw"]) and
              close(ledger["payload"], best["residual_payload_bpw"]) and
              close(ledger["amp"], best["cold_page_amplification"]), "best byte/read ledger")
    energies = [best["up_source_energy"] - best["up_captured_energy"],
                best["down_source_energy"] - best["down_captured_energy"]]
    dimensions = [best["residual_dimensions_per_role"]] * 2
    validation_values = len(validation) * ROLES * N_MATRIX
    bits = ledger["payload"] * validation_values
    wf = waterfill(energies, dimensions, bits)
    source_energy = best["up_source_energy"] + best["down_source_energy"]
    relative = wf["distortion"] / source_energy
    f_value = relative * 2.0 ** (2.0 * ledger["actual"])
    s_value = -0.5 * math.log2(f_value)
    c.require(all(close(a, b) for a, b in zip(energies, best["residual_energies_by_role"])),
              "best residual energies")
    c.require(all(close(a, b) for a, b in zip(wf["variances"], best["residual_variances_by_role"])),
              "best residual variances")
    c.require(close(bits, best["payload_bits_validation"]) and
              close(wf["allocated"], best["allocated_bits_validation"], 2e-12),
              "best payload allocation closure")
    c.require(close(wf["theta"], best["water_level"]) and
              all(close(a, b) for a, b in zip(wf["rates"], best["residual_role_rates_bpw"])),
              "best exact waterfill solution")
    c.require(close(relative, best["relative_mse"]) and close(f_value, best["F"]) and
              close(s_value, best["s_bpw"]), "best F/s arithmetic")
    c.require(f_value > 0.8 and best["favorable_grant"] == "FP16 factor rounding error is zero",
              "bounded physical hard kill and favorable grant")
    c.require("not per-mode" in evidence["decision_boundary"].lower(), "two-role claim boundary")

    composite = design["composite_nesting"]
    c.require(composite["verdict"] == "NO_ADDITIVE_OR_CONTAINMENT_CLAIM", "composite verdict")
    c.require(close(composite["incremental_s_needed_bpw"], required_s - composite["existing_composite_s_bpw"]),
              "composite missing s")
    hypothetical_s = composite["existing_composite_s_bpw"] + envelopes[1]["free_s_bpw"]
    c.require(close(composite["hypothetical_illegal_free_role_specific_sum_s_bpw"], hypothetical_s)
              and close(composite["hypothetical_illegal_free_role_specific_F"], 2.0 ** (-2.0 * hypothetical_s)),
              "hypothetical illegal composite arithmetic")
    reason = composite["reason"].lower()
    c.require(all(term in reason for term in ("same source energy", "composite residual", "containment")),
              "composite non-overlap warning")

    stage_text = held["stage0_gate.py"].decode("utf-8")
    tree = ast.parse(stage_text, filename="stage0_gate.py")
    top_imports = {
        alias.name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    c.require("cupy" not in top_imports and stage_text.count("import cupy as cp") == 1,
              "CuPy only behind replay interlock")
    c.require("--authorize-independent-payload-replay" in stage_text and
              "ROOT_COORDINATED_REPLAY_V0" not in stage_text,
              "flag present and token sourced from design lock")
    c.require("blind_protocol" not in stage_text and "unblinded" not in stage_text,
              "no pinned-panel path knowledge")
    banned = {"subprocess", "socket", "requests", "urllib", "torch"}
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    c.require(not banned & imported, "no network/subprocess/framework imports")
    c.require("per_mode_physical_rows" in stage_text and "reverse_waterfill" in stage_text,
              "per-mode and physical allocation implementations present")

    return {
        "schema": "kbvq_idre_raw_mse_gate_source_verification_v0",
        "status": "PASS",
        "checks": c.count,
        "package_manifest_sha256": sha256(held[MANIFEST]),
        "decision": evidence["decision"],
        "best_two_role_F": f_value,
        "best_two_role_cold_page_amplification": ledger["amp"],
        "gpu_or_qwen_payload_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.package), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
