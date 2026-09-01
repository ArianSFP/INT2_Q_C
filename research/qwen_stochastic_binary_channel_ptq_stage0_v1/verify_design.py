#!/usr/bin/env python3
"""Standard-library, source-only verifier for the frozen QSB-PTQ-v1 package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
import sys
from fractions import Fraction
from pathlib import Path, PurePosixPath


MANIFEST_NAME = "PACKAGE_MANIFEST.json"
EXPECTED_FILES = {
    "README.md", "design_lock.json", "panel_bindings.json", "stage0_screen.py",
    "verify_design.py", "test_source_only.py", "source_only_receipt.json", MANIFEST_NAME,
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PANEL_VALUES = 28_311_552
MATRIX_VALUES = 1_572_864
PAGE = 4096


class Checks:
    def __init__(self):
        self.count = 0

    def require(self, condition, label):
        self.count += 1
        if not condition:
            raise AssertionError(f"check {self.count} failed: {label}")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def strict_json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value

    def constant(text):
        raise ValueError("nonfinite JSON token: " + text)

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=constant)


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
        if (before.st_dev, before.st_ino, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise ValueError("file changed before held read: " + str(path))
        chunks = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("file changed during held read: " + str(path))
        raw = b"".join(chunks)
        if len(raw) != opened.st_size:
            raise ValueError("short held read: " + str(path))
        return raw
    finally:
        os.close(descriptor)


def relclose(actual, expected, tolerance=2e-15):
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def verify(package: Path):
    c = Checks()
    package = package.resolve(strict=True)
    c.require(package.is_dir() and not package.is_symlink(), "package is a non-link directory")
    names = {entry.name for entry in package.iterdir()}
    c.require(names == EXPECTED_FILES, "exact eight-file closure")
    c.require(all(entry.is_file() and not entry.is_symlink() for entry in package.iterdir()),
              "closure contains only regular non-link files")

    held = {name: held_regular_read(package / name) for name in sorted(EXPECTED_FILES)}
    manifest = strict_json(held[MANIFEST_NAME])
    c.require(manifest.get("schema") == "qwen_stochastic_binary_channel_ptq_stage0_manifest_v1",
              "manifest schema")
    c.require(manifest.get("closed_world") is True, "manifest closed-world flag")
    entries = manifest.get("entries")
    c.require(isinstance(entries, list) and len(entries) == 7, "seven governed manifest entries")
    c.require([row.get("path") for row in entries] == sorted(EXPECTED_FILES - {MANIFEST_NAME}),
              "manifest exact sorted paths")
    for row in entries:
        name = row["path"]
        c.require(PurePosixPath(name).name == name and name not in {".", ".."},
                  "manifest path safe: " + name)
        c.require(row.get("bytes") == len(held[name]), "manifest byte count: " + name)
        c.require(row.get("sha256") == sha256(held[name]) and bool(HEX64.fullmatch(row["sha256"])),
                  "manifest digest: " + name)

    design = strict_json(held["design_lock.json"])
    bindings = strict_json(held["panel_bindings.json"])
    receipt = strict_json(held["source_only_receipt.json"])
    c.require(design.get("schema") == "qwen_stochastic_binary_channel_ptq_stage0_design_v1",
              "design schema")
    c.require(design.get("status") == "FROZEN_SOURCE_ONLY_NO_PAYLOAD_OR_EXECUTION_AUTHORITY",
              "design source-only status")
    c.require(design.get("name") == "QSB-PTQ-v1", "design versioned name")
    c.require(bindings.get("schema") == "qwen_stochastic_binary_channel_ptq_panel_bindings_v1",
              "panel binding schema")
    c.require(receipt.get("schema") == "qwen_stochastic_binary_channel_ptq_stage0_source_receipt_v1",
              "receipt schema")
    claimed_receipt = receipt.get("receipt_sha256")
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256", None)
    c.require(bool(HEX64.fullmatch(str(claimed_receipt))) and sha256(canonical(receipt_body)) == claimed_receipt,
              "canonical receipt seal")
    c.require(receipt.get("verdict") == "READY_FOR_COORDINATED_STAGE0_NO_GPU_EXECUTED",
              "receipt verdict")
    c.require(receipt.get("package_file_count") == 8, "receipt closure count")
    lineage = design["lineage_and_adaptivity"]
    c.require(lineage["source_package"] == "QSB-PTQ-v0" and
              lineage["source_package_manifest_sha256"] ==
              "e4dc43066165bc9e5f9344d477e63f35ff46ceb58bbb209559d717fe0740b04c",
              "v0 source lineage")
    c.require("after inspecting" in lineage["adaptivity_disclosure"] and
              "not independent held-out" in lineage["evidence_consequence"] and
              "before any QSB-PTQ-v1" in lineage["preregistration_boundary"],
              "adaptivity and preregistration boundary")

    objective = design["objective"]
    c.require(objective["panel_values"] == PANEL_VALUES, "objective panel count")
    c.require(objective["physical_rate_interval_bpw"] == [2.15, 2.5], "physical rate interval")
    c.require(objective["cold_page_read_max_exclusive"] == 2.0, "cold-read threshold")
    required_s = -0.5 * math.log2(0.8)
    c.require(relclose(objective["required_s_min_bpw"], required_s), "objective s threshold")
    c.require(objective["padding_and_zero_bytes_charged"] is True, "padding charged")

    panel = design["panel"]
    c.require(panel["experts"] == 6 and panel["matrices"] == 18, "panel cardinality")
    c.require(panel["values_per_expert"] == 3 * MATRIX_VALUES, "values per expert")
    split = panel["whole_expert_split"]
    split_sets = [set(split[key]) for key in ("fit_expert_ordinals",
                                              "calibration_expert_ordinals",
                                              "closed_score_expert_ordinals")]
    c.require(split_sets == [{0, 2, 4}, {1}, {3, 5}], "frozen split identities")
    c.require(set.union(*split_sets) == set(range(6)) and sum(map(len, split_sets)) == 6,
              "split disjoint complete")
    c.require("source-leaking" in panel["stage0_dominance_oracle_scope"], "oracle leakage disclosed")

    plan = bindings["plan"]
    c.require(plan["sha256"] == panel["plan_sha256"] and bool(HEX64.fullmatch(plan["sha256"])),
              "plan external digest binding")
    c.require(plan["internal_lock_sha256"] == panel["plan_internal_lock_sha256"] and
              bool(HEX64.fullmatch(plan["internal_lock_sha256"])), "plan internal seal binding")
    c.require(plan["bytes"] == 24790 and plan["expected_schema"] ==
              "strata_expert_affine_n20n21_plan_v1", "plan byte/schema binding")
    geometry = bindings["geometry"]
    c.require(geometry == {"experts": 6, "matrices": 18, "panel_values": PANEL_VALUES,
                           "values_per_matrix": MATRIX_VALUES}, "binding geometry")
    sources = bindings["sources"]
    c.require(isinstance(sources, list) and len(sources) == 18, "eighteen source bindings")
    c.require([row["matrix_ordinal"] for row in sources] == list(range(18)), "canonical source order")
    c.require([row["role"] for row in sources] == [role for _ in range(6)
                                                    for role in ("gate", "up", "down")],
              "canonical source roles")
    c.require(len({row["source_relpath"] for row in sources}) == 18 and
              len({row["sha256"] for row in sources}) == 18, "unique paths and source hashes")
    for row in sources:
        role = row["role"]
        expected_shape = [2048, 768] if role == "down" else [768, 2048]
        path = PurePosixPath(row["source_relpath"])
        c.require(row["bytes"] == 3_145_728 and row["shape"] == expected_shape,
                  "source geometry: " + str(row["matrix_ordinal"]))
        c.require(bool(HEX64.fullmatch(row["sha256"])), "source hash syntax: " + str(row["matrix_ordinal"]))
        c.require(not path.is_absolute() and ".." not in path.parts and
                  path.parts[0] == "sources" and "validation" not in str(path).lower(),
                  "source path boundary: " + str(row["matrix_ordinal"]))
    access = bindings["access_boundary"]
    c.require(access["numeric_payload_files_opened"] == 0 and
              access["fresh_validation_files_opened"] == 0, "no payload access in binding preparation")
    c.require(access["gpu_jobs_submitted"] == 0 and access["network_operations_for_panel_data"] == 0,
              "no GPU/network panel operation")

    architecture = design["architecture"]
    c.require(architecture["source_block_values"] == 64 and
              architecture["blocks_per_matrix"] == MATRIX_VALUES // 64 and
              architecture["blocks_per_expert"] == 3 * MATRIX_VALUES // 64,
              "block topology arithmetic")
    c.require(architecture["bernoulli_latents_per_block"] == 160, "latent topology")
    c.require(architecture["encoder"]["projection_seed_u64"] == 14592251004518932763 and
              "96.5%" in architecture["encoder"]["alpha_selection"],
              "preserved projection seed and v1 fit target")
    c.require(architecture["shared_randomness"]["candidate_seeds_u64"] ==
              [16443857425729824865, 6983438078262162903, 11299122902407625677],
              "preserved common-randomness seeds")
    decoder = architecture["decoder"]
    c.require(decoder["raw_bipolar_features"] == 160 and
              decoder["pairwise_product_features"] == 96 and decoder["feature_count"] == 256,
              "nonlinear feature arithmetic")
    c.require(decoder["pair_schedule_seed_u64"] == 10058181636442808937,
              "preserved pair schedule seed")
    c.require(decoder["serialized_weight_shape"] == [64, 256] and 64 * 256 == 16384,
              "Q7 decoder byte arithmetic")
    c.require("no output cast" in decoder["bias_scale_and_output"] and
              "member" in decoder["dominance"], "exact operational-family dominance contract")
    c.require(architecture["channel_inputs_per_expert"] == (3 * MATRIX_VALUES // 64) * 160,
              "channel input count")
    c.require(architecture["polar_segment_channels"] == 2 ** 18 and
              architecture["polar_segments_per_expert"] == 45 and
              45 * 2 ** 18 == architecture["channel_inputs_per_expert"], "exact polar segmentation")
    distinctions = " ".join(architecture["distinct_from_additive_vq"]).lower()
    c.require(all(term in distinctions for term in ("no stored vector dictionary", "no nearest-codeword",
                                                     "multiplicative", "no reconstruction is a sum")),
              "non-additive-VQ distinctions")

    packet = design["global_packet_layout_bytes"]
    c.require(sum(value for key, value in packet.items() if key != "total") == packet["total"] == 24576,
              "global packet layout")
    c.require(packet["decoder_q7_weights"] == 64 * 256, "global decoder allocation")
    c.require(design["expert_header_bytes"] == PAGE, "expert header page")
    budget = design["ideal_kl_budget_policy"]
    c.require(budget["fit_target_fraction_of_physical_payload"] == 0.965 and
              budget["execution_limit_fraction_of_physical_payload"] == 0.97 and
              budget["physical_payload_fraction"] == 1.0, "frozen KL budget fractions")
    c.require(relclose(budget["execution_limit_fraction_of_physical_payload"] -
                       budget["fit_target_fraction_of_physical_payload"],
                       budget["calibration_generalization_cushion_fraction"]),
              "exact 0.5 percentage-point cushion")
    c.require("must never be described as physical overflow" in budget["status_semantics"],
              "policy-limit/physical-overflow semantic separation")
    expected_cells = [
        ("QSB215", 309, Fraction(155, 72), Fraction(77, 36), Fraction(63, 62)),
        ("QSB230", 331, Fraction(83, 36), Fraction(55, 24), Fraction(337, 332)),
        ("QSB250", 359, Fraction(5, 2), Fraction(179, 72), Fraction(73, 72)),
    ]
    cells = design["rate_cells"]
    c.require(len(cells) == 3, "three frozen rate cells")
    for cell, (name, pages, rate, payload_rate, read_amp) in zip(cells, expected_cells):
        frame = pages * PAGE
        payload = frame - PAGE
        container = packet["total"] + 6 * frame
        metadata_rate = Fraction((packet["total"] + 6 * PAGE) * 8, PANEL_VALUES)
        c.require(cell["cell"] == name and cell["expert_frame_pages"] == pages and
                  cell["expert_frame_bytes"] == frame, "frame ledger: " + name)
        c.require(cell["expert_payload_bytes"] == payload and
                  cell["expert_payload_bits"] == payload * 8, "payload ledger: " + name)
        c.require(cell["container_bytes"] == container and cell["container_bits"] == container * 8,
                  "container ledger: " + name)
        c.require(Fraction(cell["exact_physical_bpw_fraction"]) == rate ==
                  Fraction(container * 8, PANEL_VALUES), "physical bpw fraction: " + name)
        c.require(Fraction(cell["exact_channel_payload_bpw_fraction"]) == payload_rate ==
                  Fraction(6 * payload * 8, PANEL_VALUES), "payload bpw fraction: " + name)
        c.require(Fraction(cell["exact_metadata_bpw_fraction"]) == metadata_rate == Fraction(1, 72),
                  "metadata bpw fraction: " + name)
        c.require(Fraction(cell["cold_read_amplification_fraction"]) == read_amp ==
                  Fraction(pages + packet["total"] // PAGE, pages + packet["total"] // (6 * PAGE)),
                  "cold-read fraction: " + name)
        c.require(cell["cold_pages"] == pages + 6 and cell["equal_share_pages"] == pages + 1,
                  "cold-read pages: " + name)
        c.require(relclose(cell["exact_physical_bpw"], float(rate)) and
                  2.15 <= cell["exact_physical_bpw"] <= 2.5, "physical bpw float: " + name)
        c.require(relclose(cell["exact_channel_payload_bpw"], float(payload_rate)) and
                  relclose(cell["cold_read_amplification"], float(read_amp)) and
                  float(read_amp) < 2.0, "payload/read float: " + name)
        target = 0.8 * 2.0 ** (-2.0 * float(rate))
        c.require(relclose(cell["target_relative_mse"], target) and
                  relclose(cell["required_capture"], 1.0 - target), "objective threshold: " + name)

    protocol = design["stage0_protocol"]
    c.require("0.97 execution limit" in protocol["rate_gate"] and
              "physical-overflow status" in protocol["rate_gate"] and
              "before" in protocol["early_control_gate"] and
              "no raw-SSE shortcut" in protocol["early_control_gate"], "strict early rate/control ordering")
    c.require("plus three delete-expert jackknife SE" in protocol["aggregate_gate"] and
              "before controls" in protocol["aggregate_gate"], "favorable aggregate kill")
    controls = design["matched_gaussian_controls"]
    c.require(controls["replicates"] == 8 and controls["seeds_u64"] ==
              list(range(5850734194750267521, 5850734194750267529)), "control count/seeds")
    c.require("mean" in controls["construction"] and "centered energy" in controls["construction"] and
              "FP64" in controls["construction"], "matched-moment control contract")
    c.require("controls run only" in controls["execution_order"] and
              "lower-three-SE" in controls["promotion_gate"] and
              "strongest control upper-three-SE" in controls["promotion_gate"], "control ordering/gate")
    comparisons = design["comparison_semantics"]
    c.require("not operational" in comparisons["matched_gaussian_stage0"], "Gaussian control boundary")
    c.require("actual bytes" in comparisons["future_operational_tcq"] and
              "paper plots" in comparisons["future_operational_tcq"], "operational TCQ evidence contract")
    c.require("asymptotic information-theoretic lower bound" in comparisons["shannon_gaussian_mse_limit"] and
              "neither TCQ nor an empirical control" in comparisons["shannon_gaussian_mse_limit"],
              "Shannon/TCQ/control separation")
    forbidden = " ".join(comparisons["forbidden_claims"])
    c.require("Do not call TCQ the Shannon limit." in forbidden and
              "Do not transfer the paper's Gaussian result" in forbidden, "forbidden misleading claims")
    channel = design["channel_simulation"]
    c.require("optimistic stage-0 rate oracle" in channel["no_rate_claim"] and
              "byte-emitted" in channel["no_rate_claim"], "ideal KL claim boundary")
    c.require("overflow rejects" in channel["reservoir_rule"] and "zero padded and charged" in
              channel["reservoir_rule"], "fixed reservoir rule")
    authority = design["authority"]
    c.require(all(authority[key] is False for key in ("may_open_numeric_payloads", "may_submit_gpu_job",
                                                       "may_open_fresh_validation", "may_emit_compressed_weights")) and
              authority["network_calls"] == 0, "zero execution/payload authority")

    runner_text = held["stage0_screen.py"].decode("utf-8")
    tree = ast.parse(runner_text, filename="stage0_screen.py")
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.append((node.module or "").split(".")[0])
    c.require(set(top_imports) <= {"__future__", "argparse", "hashlib", "json", "math", "os", "re",
                                  "stat", "sys", "time", "pathlib"}, "top-level imports are standard library")
    c.require("import cupy as cp" in runner_text and "import numpy as np" in runner_text and
              runner_text.index("def main") < runner_text.index("import cupy as cp"),
              "CuPy/NumPy are deferred inside main")
    c.require(all(bad not in runner_text for bad in ("requests", "urllib", "socket", "subprocess", "paramiko")),
              "runner has no network/child-process mechanism")
    c.require("if output.exists(): raise FileExistsError" in runner_text and
              "CUDA_VISIBLE_DEVICES must be 0" in runner_text, "absent output and device gates")
    c.require("def held_regular_read" in runner_text and "O_NOFOLLOW" in runner_text and
              "O_EXCL" in runner_text, "held non-link inputs and exclusive result creation")
    c.require("FIT_TARGET_FILL = 0.965" in runner_text and
              "EXECUTION_LIMIT_FILL = 0.97" in runner_text and "KL_FILL" not in runner_text,
              "runner split fit-target/execution-limit constants")
    c.require(runner_text.count("FIT_TARGET_FILL * cell") == 2 and
              "execution_limit_bits = EXECUTION_LIMIT_FILL * payload_bits" in runner_text,
              "runner uses 0.965 for Qwen/control fits and 0.97 for execution")
    c.require("HARD_KILL_IDEAL_KL_EXCEEDS_PHYSICAL_RESERVOIR" in runner_text and
              "HARD_KILL_IDEAL_KL_EXCEEDS_PREREGISTERED_0P97_EXECUTION_LIMIT" in runner_text and
              "HARD_KILL_IDEAL_KL_EXCEEDS_FROZEN_RESERVOIR" not in runner_text and
              "HARD_KILL_FAVOURABLE_ORACLE_UCB_BELOW_EXACT_REQUIREMENT" in runner_text,
              "runner truthful distinct hard-kill statuses")
    c.require(all(field in runner_text for field in
                  ("fit_target_bits_by_expert", "execution_limit_bits_by_expert",
                   "physical_reservoir_bits_by_expert", "execution_limit_margin_bits_by_expert",
                   "physical_reservoir_margin_bits_by_expert")), "runner exact KL limit/margin ledger")
    c.require("qwen_stochastic_binary_channel_ptq_stage0_result_v1" in runner_text and
              "not independent held-out evidence" in runner_text, "v1 result schema/claim boundary")
    c.require(runner_text.index("ORACLE_SURVIVOR_REQUIRES_MATCHED_CONTROLS") <
              runner_text.index("controls = run_controls"), "controls are conditionally ordered")
    c.require("POLICY_REJECT_CONTROL_RATE_INCOMPARABLE" in runner_text and
              "POLICY_REJECT_SOURCE_NOT_ABOVE_MATCHED_CONTROLS" in runner_text and
              "POLICY_HOLD_FOR_OPERATIONAL_IMPLEMENTATION" in runner_text, "runner control decisions")
    c.require("astype(np.float32)" not in runner_text[runner_text.index("def matched_gaussian_panel"):],
              "matched controls retain FP64 moments")
    c.require("fresh_validation_files_opened\": 0" in runner_text and
              "compressed_outputs_created\": 0" in runner_text, "result access ledger")
    compile(runner_text, "stage0_screen.py", "exec")
    c.require(True, "runner compiles")

    paper = design["paper_inspiration"]
    c.require(paper["primary_url"] == "https://arxiv.org/abs/2606.29578v1" and
              "arXiv:2606.29578v1" in paper["citation"], "version-pinned primary inspiration")
    c.require("not the paper's implementation" in " ".join(paper["not_claimed"]) and
              "No theorem" in " ".join(paper["not_claimed"]), "paper claim boundary")
    c.require(receipt["design_lock_sha256"] == sha256(held["design_lock.json"]) and
              receipt["panel_bindings_sha256"] == sha256(held["panel_bindings.json"]) and
              receipt["stage0_script_sha256"] == sha256(held["stage0_screen.py"]), "receipt source bindings")
    c.require(receipt["payload_access"]["numeric_panel_files_opened"] == 0 and
              receipt["payload_access"]["fresh_validation_files_opened"] == 0 and
              receipt["execution"]["gpu_jobs_submitted"] == 0 and
              receipt["execution"]["stage0_runs"] == 0, "receipt no-access/no-execution ledger")
    c.require(receipt["lineage"]["source_manifest_sha256"] ==
              "e4dc43066165bc9e5f9344d477e63f35ff46ceb58bbb209559d717fe0740b04c" and
              receipt["lineage"]["v0_result_was_seen"] is True and
              receipt["lineage"]["v1_qwen_or_gpu_execution_before_seal"] == 0,
              "receipt adapted-v1 lineage and zero v1 execution")
    c.require(receipt["verifier_check_count"] == c.count + 1, "receipt frozen final check count")
    return c.count


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    try:
        count = verify(args.package)
    except Exception as error:
        print(json.dumps({"schema": "qwen_stochastic_binary_channel_ptq_stage0_verify_v1",
                          "verdict": "BLOCK", "error": f"{type(error).__name__}: {error}"},
                         sort_keys=True))
        return 1
    manifest_sha = sha256(held_regular_read(args.package.resolve() / MANIFEST_NAME))
    print(json.dumps({"schema": "qwen_stochastic_binary_channel_ptq_stage0_verify_v1",
                      "verdict": "PASS", "checks": count,
                      "manifest_sha256": manifest_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
