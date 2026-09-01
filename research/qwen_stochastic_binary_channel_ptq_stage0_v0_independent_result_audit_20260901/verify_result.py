#!/usr/bin/env python3
"""Independent standard-library verifier for the QSB-PTQ-v0 RunPod result audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath


MANIFEST = "RESULT_AUDIT_MANIFEST.json"
FILES = {
    "README.md", "audit_receipt.json", "design_lock_v0.json", "panel_bindings_v0.json",
    "recomputed_margins.json", "runpod_result.json", "stage0_screen_v0.py", "verify_result.py",
    MANIFEST,
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED = {
    "result": "954cf10187baca348ab1d134be8362efe7d79ca59d8592bb8e75be742ff1e6df",
    "design": "67d6a78d1b45d011ad9707f1484970346fb0070cbd7be8cbe0c7e6653c027aae",
    "bindings": "d3022a67a483ab84660cc67e9e141703e5d350e9c850a9e949f85e61224e6886",
    "runner": "ee671293ec6e608d7841e09406bdb30e02f23ae3917798f2fe65d435e5447b06",
    "plan": "8017582201468300dd07550a1a2f8d90dc704ffae7ae6d8801a560178e4a1868",
    "plan_lock": "99b17b18f74187b40aa7715260892491dc5f5f56baa0ef520509aa87d655df7d",
}
BLOCKS_PER_EXPERT = 73728


class Checks:
    def __init__(self): self.count = 0
    def require(self, condition, label):
        self.count += 1
        if not condition: raise AssertionError(f"check {self.count} failed: {label}")


def digest(raw): return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError("duplicate JSON key: " + key)
            out[key] = value
        return out
    def finite(text):
        value = float(text)
        if not math.isfinite(value): raise ValueError("nonfinite JSON")
        return value
    def constant(text): raise ValueError("nonfinite JSON: " + text)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=constant)


def held_read(path):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError("non-regular or link: " + str(path))
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise ValueError("identity changed: " + str(path))
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block: break
            chunks.append(block)
        after = os.fstat(fd)
        if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("changed during read: " + str(path))
        raw = b"".join(chunks)
        if len(raw) != opened.st_size: raise ValueError("short read")
        return raw
    finally:
        os.close(fd)


def close(actual, expected, tolerance=2e-13):
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def verify(root):
    c = Checks()
    root = root.resolve(strict=True)
    c.require(root.is_dir() and not root.is_symlink(), "audit directory")
    c.require({entry.name for entry in root.iterdir()} == FILES, "exact nine-file closure")
    c.require(all(entry.is_file() and not entry.is_symlink() for entry in root.iterdir()),
              "regular non-link closure")
    held = {name: held_read(root / name) for name in FILES}
    manifest = strict_json(held[MANIFEST])
    c.require(manifest.get("schema") == "qsb_ptq_v0_independent_result_audit_manifest_v0",
              "manifest schema")
    c.require(manifest.get("closed_world") is True, "closed-world manifest")
    entries = manifest.get("entries")
    c.require(isinstance(entries, list) and len(entries) == 8, "eight governed entries")
    c.require([row.get("path") for row in entries] == sorted(FILES - {MANIFEST}),
              "exact sorted manifest paths")
    for row in entries:
        name = row["path"]
        c.require(PurePosixPath(name).name == name, "safe manifest path: " + name)
        c.require(row.get("bytes") == len(held[name]), "manifest bytes: " + name)
        c.require(row.get("sha256") == digest(held[name]) and HEX64.fullmatch(row["sha256"]),
                  "manifest digest: " + name)

    c.require(digest(held["runpod_result.json"]) == EXPECTED["result"], "exact result pin")
    c.require(digest(held["design_lock_v0.json"]) == EXPECTED["design"], "exact v0 design evidence")
    c.require(digest(held["panel_bindings_v0.json"]) == EXPECTED["bindings"],
              "exact v0 binding evidence")
    c.require(digest(held["stage0_screen_v0.py"]) == EXPECTED["runner"], "exact v0 runner evidence")
    result = strict_json(held["runpod_result.json"])
    design = strict_json(held["design_lock_v0.json"])
    bindings = strict_json(held["panel_bindings_v0.json"])
    recomputed = strict_json(held["recomputed_margins.json"])
    receipt = strict_json(held["audit_receipt.json"])
    c.require(result.get("schema") == "qwen_stochastic_binary_channel_ptq_stage0_result_v0",
              "result schema")
    c.require(result.get("status") == "POLICY_REJECT_ALL_RATE_CELLS", "aggregate v0 policy status")
    c.require(result["bindings"]["design_lock_sha256"] == EXPECTED["design"] and
              result["bindings"]["panel_bindings_sha256"] == EXPECTED["bindings"] and
              result["bindings"]["stage0_script_sha256"] == EXPECTED["runner"],
              "result source-package hashes")
    c.require(result["bindings"]["plan_sha256"] == EXPECTED["plan"] and
              result["bindings"]["plan_internal_lock_sha256"] == EXPECTED["plan_lock"],
              "result plan hashes")
    c.require(bindings["plan"]["sha256"] == EXPECTED["plan"] and
              bindings["plan"]["internal_lock_sha256"] == EXPECTED["plan_lock"],
              "copied binding plan hashes")
    c.require(len(bindings["sources"]) == len(result["bindings"]["sources"]) == 18,
              "source binding counts")
    for expected_source, observed_source in zip(bindings["sources"], result["bindings"]["sources"]):
        c.require(observed_source == {"bytes": expected_source["bytes"],
                                     "matrix_ordinal": expected_source["matrix_ordinal"],
                                     "relative_path": expected_source["source_relpath"],
                                     "sha256": expected_source["sha256"]},
                  "result source identity: " + str(expected_source["matrix_ordinal"]))
    c.require(result["fixed_splits"] == {"fit_experts": [0, 2, 4],
                                          "calibration_experts": [1],
                                          "closed_score_experts": [3, 5]}, "fixed splits")
    c.require(result["access"] == {"authenticated_panel_sources_opened": 18,
                                    "compressed_outputs_created": 0,
                                    "fresh_validation_files_opened": 0,
                                    "network_operations": 0}, "result access ledger")
    c.require(result["execution"]["cuda_visible_devices"] == "0" and
              result["execution"]["device"] == "NVIDIA GeForce RTX 5090" and
              result["execution"]["elapsed_seconds"] > 0.0, "runtime ledger")
    c.require("source-leaking" in result["claim_boundary"] and
              "not a serialized channel simulator" in result["claim_boundary"], "result claim boundary")

    source = held["stage0_screen_v0.py"].decode("utf-8")
    c.require("KL_FILL = 0.97" in source, "v0 frozen 0.97 constant")
    c.require("value > KL_FILL * payload_bits" in source, "v0 gate compares against 0.97 payload")
    c.require("HARD_KILL_IDEAL_KL_EXCEEDS_FROZEN_RESERVOIR" in source,
              "v0 overstated status token reproduced")
    c.require("97% of that expert's fixed payload reservoir" in design["stage0_protocol"]["rate_gate"],
              "design establishes 0.97 policy ceiling")
    c.require(recomputed["result_sha256"] == EXPECTED["result"], "recomputation result binding")
    c.require(recomputed["verdict"] == {
        "v0_preregistered_policy_rejection": "PASS",
        "physical_reservoir_overflow_claim": "BLOCK",
        "reason": "some experts exceed the frozen 0.97 policy ceiling, but all 18 retain positive margins of approximately 2.98% to 3.02% of the true physical payload reservoir",
    }, "recomputation verdict")

    expected_cells = [("QSB215", 10092544, [2, 5]),
                      ("QSB230", 10813440, [2, 5]),
                      ("QSB250", 11730944, [1, 2, 5])]
    c.require(len(result["cells"]) == len(recomputed["cells"]) == 3, "three cells")
    for observed, frozen, (name, payload_bits, violators) in zip(
            result["cells"], recomputed["cells"], expected_cells):
        cell = observed["cell"]
        oracle = observed["qwen_oracle"]
        limit = 0.97 * payload_bits
        values = oracle["ideal_kl_bits_by_expert"]
        c.require(cell["cell"] == frozen["cell"] == name, "cell identity: " + name)
        c.require(cell["payload_bytes_per_expert"] * 8 == payload_bits and
                  oracle["reservoir_bits_by_expert"] == payload_bits, "physical reservoir: " + name)
        c.require(oracle["status"] == "HARD_KILL_IDEAL_KL_EXCEEDS_FROZEN_RESERVOIR",
                  "reported v0 rate status: " + name)
        c.require(observed["control_gate"]["status"] == "NOT_RUN_ORACLE_DID_NOT_SURVIVE" and
                  observed["matched_gaussian_controls"] == [], "controls correctly skipped: " + name)
        c.require(set(oracle) == {"ideal_kl_bits_by_expert", "reservoir_bits_by_expert", "status"},
                  "no distortion oracle after early kill: " + name)
        c.require(close(observed["fit_mean_ideal_kl_bits_per_block"] * BLOCKS_PER_EXPERT, limit),
                  "fit mean targeted 0.97: " + name)
        computed_violators = [index for index, value in enumerate(values) if value > limit + 2e-7]
        physical_violators = [index for index, value in enumerate(values) if value > payload_bits + 2e-7]
        c.require(computed_violators == frozen["experts_above_execution_limit"] == violators,
                  "0.97 violators: " + name)
        c.require(physical_violators == frozen["experts_above_physical_reservoir"] == [],
                  "no physical overflow: " + name)
        c.require(close(frozen["execution_limit_bits_per_expert"], limit) and
                  frozen["physical_reservoir_bits_per_expert"] == payload_bits,
                  "frozen thresholds: " + name)
        c.require(close(frozen["reported_fit_mean_total_bits"],
                        observed["fit_mean_ideal_kl_bits_per_block"] * BLOCKS_PER_EXPERT),
                  "frozen reported fit total: " + name)
        c.require(len(frozen["experts"]) == len(values) == 6, "expert rows: " + name)
        physical_margins = []
        limit_excesses = []
        for index, (value, row) in enumerate(zip(values, frozen["experts"])):
            limit_margin = limit - value
            physical_margin = payload_bits - value
            physical_percent = 100.0 * physical_margin / payload_bits
            c.require(row["expert"] == index and close(row["ideal_kl_bits"], value),
                      f"expert value: {name}/{index}")
            c.require(close(row["execution_limit_margin_bits"], limit_margin) and
                      close(row["physical_reservoir_margin_bits"], physical_margin) and
                      close(row["physical_reservoir_margin_percent"], physical_percent),
                      f"expert margins: {name}/{index}")
            c.require(physical_margin > 0.0, f"positive physical margin: {name}/{index}")
            physical_margins.append(physical_margin)
            limit_excesses.append(max(0.0, -limit_margin))
        c.require(close(frozen["minimum_physical_reservoir_margin_bits"], min(physical_margins)) and
                  close(frozen["minimum_physical_reservoir_margin_percent"],
                        100.0 * min(physical_margins) / payload_bits), "cell physical minimum: " + name)
        c.require(close(frozen["worst_execution_limit_excess_bits"], max(limit_excesses)),
                  "cell 0.97 excess maximum: " + name)

    c.require(receipt["schema"] == "qsb_ptq_v0_independent_result_audit_receipt_v0" and
              receipt["verdict"] == "BLOCK_PHYSICAL_OVERFLOW_CLAIM_PASS_V0_POLICY_REJECTION",
              "receipt schema/verdict")
    c.require(receipt["result_origin_path"] ==
              "research/qwen_stochastic_binary_channel_ptq_stage0_v0_runpod_result_20260901/result.json",
              "sibling result origin")
    body = dict(receipt); claim = body.pop("receipt_sha256", None)
    c.require(bool(HEX64.fullmatch(str(claim))) and digest(canonical(body)) == claim,
              "canonical receipt seal")
    c.require(receipt["result_sha256"] == EXPECTED["result"] and
              receipt["verifier_check_count"] == c.count + 1, "receipt result/count binding")
    return c.count


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    try:
        checks = verify(args.audit_dir)
    except Exception as error:
        print(json.dumps({"schema": "qsb_ptq_v0_independent_result_verify_v0", "verdict": "BLOCK",
                          "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
        return 1
    manifest_sha = digest(held_read(args.audit_dir.resolve() / MANIFEST))
    print(json.dumps({"schema": "qsb_ptq_v0_independent_result_verify_v0", "verdict": "PASS",
                      "checks": checks, "manifest_sha256": manifest_sha}, sort_keys=True))
    return 0


if __name__ == "__main__": sys.exit(main())
