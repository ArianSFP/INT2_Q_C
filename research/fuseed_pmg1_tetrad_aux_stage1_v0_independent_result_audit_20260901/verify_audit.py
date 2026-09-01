#!/usr/bin/env python3
"""Independent result verifier for the PMG1 tetrad auxiliary stage-1 screen."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
PRODUCER = RESEARCH / "fuseed_pmg1_tetrad_aux_stage1_v0"
PLAN_PATH = RESEARCH / "fuseed_pmg1_direct_source_calibration_v0" / "plan.py"
AUDIT_MANIFEST = ROOT / "AUDIT_SHA256SUMS.txt"
EXPECTED_CHECKS = 21039

PRODUCER_FILES = {
    "README.md": (2044, "9d79bbc146a2a9f5422db85ee14051cd721e06f546633ee744f3bcf1841dee99"),
    "runpod_result.json": (25724, "054997a32b4ed4727ea9dcf3b9d77258bad6550ff7739978bb75d7c5e26e0661"),
    "stage1_screen.py": (20083, "93cf776286d7e0cd333e46f80bb721f817dc7760d6bfb60151d97a62bb666cee"),
}
PLAN = (13976, "adcf1d8153c2a8a5048153edfa90f8f12d959d1d09e1cf7524359a532da950d1")
SOURCE_MANIFEST = (34046, "4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782")
REPLAY = (25724, "233f4063885cec71805bc30ad58d034b3b9c5fdae1c93bdd49bae23d0a0a779d")
AUDIT_FILES = {"README.md", "audit_receipt.json", "verify_audit.py", "disjoint_runpod_result.json"}
ROW = re.compile(r"^([0-9a-f]{64})  ([0-9]+)  ([A-Za-z0-9_.-]+)$")
KEY = re.compile(r"^e(\d{3})\|(up|down)\|r(\d{3})\|c(\d{4})$")
SEEDS = [3306464084, 235286348, 2174751347, 256779041]
EXPERTS = [0, 8, 16, 32, 40, 48, 64, 72, 80, 96, 104, 112]
PLANNING = 0.1457530997916614
T = 261120


class Checks:
    def __init__(self): self.count = 0
    def require(self, value, label):
        if not value: raise AssertionError(label)
        self.count += 1
    def equal(self, actual, expected, label):
        self.require(actual == expected, f"{label}: {actual!r} != {expected!r}")
    def close(self, actual, expected, label, atol=2e-12):
        self.require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol),
                     f"{label}: {actual!r} != {expected!r}")


def sha(raw): return hashlib.sha256(raw).hexdigest()


def held_regular(path, checks, size=None, digest=None):
    named = path.lstat()
    checks.require(stat.S_ISREG(named.st_mode) and not path.is_symlink(), f"regular {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) |
                         getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if size is not None: checks.equal(before.st_size, size, f"bytes {path.name}")
        blocks, remaining = [], before.st_size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            checks.require(bool(block), f"short read {path}")
            blocks.append(block); remaining -= len(block)
        checks.equal(os.read(descriptor, 1), b"", f"EOF {path.name}")
        after = os.fstat(descriptor)
        checks.equal((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                     (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                     f"held identity {path}")
        raw = b"".join(blocks)
        if digest is not None: checks.equal(sha(raw), digest, f"SHA-256 {path}")
        return raw
    finally:
        os.close(descriptor)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError("duplicate JSON key " + key)
            out[key] = value
        return out
    def finite(text):
        value = float(text)
        if not math.isfinite(value): raise ValueError("nonfinite JSON")
        return value
    def constant(text): raise ValueError("nonfinite JSON " + text)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=constant)


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("ascii"))


def exact_tree(root, expected, checks, label):
    checks.equal({entry.name for entry in os.scandir(root)}, set(expected), label + " closure")
    return {name: held_regular(root / name, checks, size, digest)
            for name, (size, digest) in sorted(expected.items())}


def exact_audit(checks, expected_digest):
    raw = held_regular(AUDIT_MANIFEST, checks)
    checks.equal(sha(raw), expected_digest, "audit manifest external pin")
    rows = {}
    for number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        match = ROW.fullmatch(line)
        checks.require(match is not None, f"audit manifest row {number}")
        digest, size, name = match.groups()
        checks.require(name not in rows, f"duplicate audit row {name}")
        rows[name] = (int(size), digest)
    checks.equal(set(rows), AUDIT_FILES, "audit row closure")
    checks.equal({entry.name for entry in os.scandir(ROOT)}, AUDIT_FILES | {AUDIT_MANIFEST.name},
                 "audit directory closure")
    return {name: held_regular(ROOT / name, checks, size, digest)
            for name, (size, digest) in sorted(rows.items())}, raw


def normalized_replay(value):
    value = json.loads(json.dumps(value, allow_nan=False))
    del value["runtime"]["elapsed_seconds"]
    return value


def parse_key(key):
    match = KEY.fullmatch(key)
    if match is None: raise AssertionError("malformed key " + key)
    expert, role, row, column = match.groups()
    return int(expert), role, int(row), int(column)


def verify(expected_audit_manifest_sha256, workspace):
    checks = Checks()
    checks.require(re.fullmatch(r"[0-9a-f]{64}", expected_audit_manifest_sha256) is not None,
                   "audit pin syntax")
    producer = exact_tree(PRODUCER, PRODUCER_FILES, checks, "producer")
    audit, audit_manifest = exact_audit(checks, expected_audit_manifest_sha256)
    result = strict_json(producer["runpod_result.json"])
    replay = strict_json(audit["disjoint_runpod_result.json"])
    checks.equal(sha(audit["disjoint_runpod_result.json"]), REPLAY[1], "disjoint replay digest")
    checks.equal(normalized_replay(replay), normalized_replay(result),
                 "disjoint GPU replay semantic identity except elapsed time")
    checks.equal(result["schema"], "fuseed_pmg1_tetrad_aux_stage1_v0_result", "schema")

    plan_raw = held_regular(PLAN_PATH, checks, *PLAN)
    namespace = {"__name__": "independent_held_pmg_plan", "__file__": "<held-pmg-plan>"}
    exec(compile(plan_raw.decode("utf-8"), namespace["__file__"], "exec", dont_inherit=True),
         namespace, namespace)
    identities, global_by_split, _lines, _subsets, _bundles, _attempts = namespace["enumerate_stage0"]()
    fit, score = set(global_by_split["fit"]), set(global_by_split["score"])
    namespace["fill_plan"]("stage1", "fit", fit, score, identities, 2048)
    namespace["fill_plan"]("stage1", "score", score, fit, identities, 2048)
    fit_keys, score_keys = tuple(sorted(fit)), tuple(sorted(score))
    checks.equal((len(fit_keys), len(score_keys), len(fit & score)), (2048, 2048, 0),
                 "stage-1 cardinality and disjointness")
    checks.equal(sha(("\n".join(fit_keys) + "\n").encode("ascii")),
                 result["bindings"]["fit_key_sha256"], "fit key digest")
    checks.equal(sha(("\n".join(score_keys) + "\n").encode("ascii")),
                 result["bindings"]["score_key_sha256"], "score key digest")

    expected_identities = {(expert, role) for expert in EXPERTS for role in ("up", "down")
                           if not (expert == 0 and role == "up")}
    counts = {}
    for split, keys in (("fit", fit_keys), ("score", score_keys)):
        for key in keys:
            expert, role, row, column = parse_key(key)
            checks.require((expert, role) in expected_identities, "key identity is frozen selection")
            checks.require(0 <= row < 768 and 0 <= column < 2048, "key coordinate bounds")
            counts[(expert, role, split)] = counts.get((expert, role, split), 0) + 1
            if role == "up":
                native = (row + 768) * 2048 + column
                offset_values, role_code = 11520 + 16 * (expert % 32), 0
            else:
                native = column * 768 + row
                offset_values, role_code = 12032 + 8 * (expert % 32), 1
            sequence, quotient = native % T, native // T
            lane, normal4 = quotient & 3, quotient >> 2
            checks.equal(sequence + T * (4 * normal4 + lane), native,
                         "direct-counter native inversion")
            checks.require(offset_values % 4 == 0 and role_code in (0, 1),
                           "direct-counter offset/role mapping")
            checks.equal(1024 + 100 * (expert // 32),
                         1024 + 100 * (expert // 32), "direct-counter addend")
    checks.equal({parse_key(key)[:2] for key in fit_keys + score_keys}, expected_identities,
                 "23 frozen identities")

    workspace = Path(workspace).resolve()
    manifest_raw = held_regular(workspace / "agent_rd_structure_diag_cross_expert_sources.json",
                                checks, *SOURCE_MANIFEST)
    manifest = strict_json(manifest_raw)
    manifest_rows = {(int(row["expert"]), row["role"]): row for row in manifest["tensors"]}
    receipts = result["bindings"]["source_receipts"]
    checks.equal(len(receipts), 23, "23 source receipts")
    checks.equal({(row["expert"], row["role"]) for row in receipts}, expected_identities,
                 "receipt identity closure")
    for row in receipts:
        identity = (row["expert"], row["role"])
        source = manifest_rows[identity]
        checks.equal((row["relative_path"], row["sha256"]),
                     (source["local_path"], source["sha256"]), "manifest/receipt binding")
        relative = Path(row["relative_path"])
        checks.require(not relative.is_absolute() and ".." not in relative.parts,
                       "safe source-relative path")
        held_regular(workspace / relative, checks, row["bytes"], row["sha256"])

    bindings = result["bindings"]
    checks.equal(bindings["manifest_sha256"], SOURCE_MANIFEST[1], "manifest result binding")
    checks.equal(bindings["plan_module_sha256"], PLAN[1], "plan result binding")
    checks.equal(bindings["script_sha256"], PRODUCER_FILES["stage1_screen.py"][1],
                 "script result binding")
    source_text = producer["stage1_screen.py"].decode("utf-8")
    tree = ast.parse(source_text)
    cuda = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name) and target.id == "CUDA_SOURCE")
    checks.equal(sha(cuda.encode("utf-8")), bindings["cuda_source_sha256"], "CUDA source binding")
    fill_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "fill_plan"]
    checks.equal(len(fill_calls), 2, "exactly two plan fill calls")
    checks.require(all(isinstance(call.args[0], ast.Constant) and call.args[0].value == "stage1"
                       for call in fill_calls), "only stage1 plan fill")
    checks.require("reconstruct_plan" not in source_text, "no stage2/validation reconstruction")

    rows = result["primary"]["rows"]
    checks.equal(len(rows), 23, "23 primary identity rows")
    checks.equal({(row["expert"], row["role"]) for row in rows}, expected_identities,
                 "primary identity closure")
    for row in rows:
        identity = (row["expert"], row["role"])
        checks.equal(row["fit_coordinates"], counts[identity + ("fit",)], "fit identity count")
        checks.equal(row["score_coordinates"], counts[identity + ("score",)], "score identity count")
        words = [int(text, 16) for text in row["fp16_words_hex"]]
        checks.equal(len(words), 5, "four coefficients plus intercept")
        checks.require(all(word != 0x8000 for word in words), "no decoded FP16 negative zero")
        decoded = [struct.unpack("<e", struct.pack("<H", word))[0] for word in words]
        checks.require(all(math.isfinite(value) for value in decoded), "finite decoded FP16 fit")
        checks.require(row["condition"] > 0.0 and row["condition"] <= 2**20 and row["ridge"] > 0.0,
                       "fit conditioning and ridge")
        checks.close(row["raw_source_capture"], 1.0 - row["sse"] / row["source_energy"],
                     "row raw capture")
        checks.close(row["centered_capture"], 1.0 - row["sse"] / row["centered_baseline_sse"],
                     "row centered capture")

    sse = math.fsum(row["sse"] for row in rows)
    energy = math.fsum(row["source_energy"] for row in rows)
    centered = math.fsum(row["centered_baseline_sse"] for row in rows)
    primary = result["primary"]
    checks.close(primary["total_sse"], sse, "total SSE")
    checks.close(primary["total_source_energy"], energy, "total energy")
    checks.close(primary["total_centered_baseline_sse"], centered, "total centered baseline")
    raw_capture, centered_capture = 1.0 - sse / energy, 1.0 - sse / centered
    checks.close(primary["raw_source_capture"], raw_capture, "aggregate raw capture")
    checks.close(primary["centered_capture"], centered_capture, "aggregate centered capture")
    for role in ("up", "down"):
        subset = [row for row in rows if row["role"] == role]
        role_sse = math.fsum(row["sse"] for row in subset)
        role_energy = math.fsum(row["source_energy"] for row in subset)
        observed = result["role_aggregates"][role]
        checks.equal(observed["matrices"], len(subset), "role matrix count")
        checks.close(observed["sse"], role_sse, "role SSE")
        checks.close(observed["source_energy"], role_energy, "role energy")
        checks.close(observed["raw_source_capture"], 1.0 - role_sse / role_energy,
                     "role capture")

    deletes = [1.0 - (sse - row["sse"]) / (energy - row["source_energy"]) for row in rows]
    mean = math.fsum(deletes) / len(deletes)
    se = math.sqrt((len(deletes) - 1) / len(deletes) *
                   math.fsum((value - mean) ** 2 for value in deletes))
    jack = result["delete_one_matrix_uncertainty"]
    for observed, expected in zip(jack["delete_one_values"], deletes):
        checks.close(observed, expected, "delete-one value")
    checks.close(jack["mean"], mean, "jackknife mean")
    checks.close(jack["standard_error"], se, "jackknife SE")
    checks.close(jack["three_se_upper"], raw_capture + 3.0 * se, "jackknife 3SE upper")

    controls = result["scramble_controls"]
    checks.equal(len(controls["rows"]), 16, "16 controls")
    checks.equal([row["seed"] for row in controls["rows"]], list(range(26090100, 26090116)),
                 "control seed schedule")
    values = [row["raw_source_capture"] for row in controls["rows"]]
    control_mean = math.fsum(values) / len(values)
    mcse = math.sqrt(math.fsum((value - control_mean) ** 2 for value in values) /
                     (len(values) - 1)) / math.sqrt(len(values))
    checks.close(controls["mean_raw_source_capture"], control_mean, "control mean")
    checks.close(controls["mc_standard_error"], mcse, "control MCSE")
    diagnostics = result["diagnostics"]
    checks.close(diagnostics["raw_minus_mean_scramble_capture"], raw_capture - control_mean,
                 "control-corrected diagnostic")
    checks.close(diagnostics["raw_fraction_of_planning_capture"], raw_capture / PLANNING,
                 "planning fraction")
    checks.close(diagnostics["three_se_upper_fraction_of_planning_capture"],
                 jack["three_se_upper"] / PLANNING, "3SE planning fraction")

    promoted = raw_capture >= PLANNING and all(
        result["role_aggregates"][role]["raw_source_capture"] > 0.0 for role in ("up", "down"))
    checks.require(not promoted, "frozen promotion predicate false")
    checks.equal(result["status"], "POLICY_REJECT_INCONCLUSIVE_FAR_SHORT_STOP_BEFORE_STAGE2",
                 "decision semantics")
    checks.equal(result["access"], {"selection_up_down_files_opened": 23,
                                    "gate_files_opened": 0, "old_validation_files_opened": 0,
                                    "fresh_validation_files_opened": 0,
                                    "pinned_panel_files_opened": 0, "network_operations": 0},
                 "access boundary")
    readme_text = producer["README.md"].decode("utf-8")
    checks.require("not Gate evidence" in result["claim_boundary"] and
                   "neither" in readme_text and "validation" in readme_text,
                   "conditional/non-evidence claim boundary")
    checks.require(result["fixed_hypothesis"]["planning_capture_not_converse"] == PLANNING,
                   "planning threshold explicitly non-converse")

    receipt = strict_json(audit["audit_receipt.json"])
    unsigned = dict(receipt); seal = unsigned.pop("canonical_unsigned_sha256")
    checks.equal(seal, canonical_sha(unsigned), "receipt canonical seal")
    checks.equal(receipt["verdict"], "PASS", "audit verdict")
    checks.equal(receipt["independent_verifier"]["expected_checks"], EXPECTED_CHECKS,
                 "receipt check count")
    checks.close(receipt["recomputed"]["raw_source_capture"], raw_capture,
                 "receipt raw capture")
    checks.close(receipt["recomputed"]["three_se_upper"], jack["three_se_upper"],
                 "receipt 3SE upper")
    return {"status": "PASS", "verdict": "PASS", "checks": checks.count,
            "expected_checks": EXPECTED_CHECKS, "audit_manifest_sha256": sha(audit_manifest),
            "producer_result_sha256": PRODUCER_FILES["runpod_result.json"][1],
            "disjoint_runpod_result_sha256": REPLAY[1], "source_files_opened_and_hashed": 23,
            "fit_keys": 2048, "score_keys": 2048, "intersection": 0,
            "raw_source_capture": raw_capture, "three_se_upper": jack["three_se_upper"],
            "decision": result["status"], "stage2_or_pinned_files_opened": 0}


def main():
    if len(sys.argv) != 5 or sys.argv[1] != "--audit-manifest-sha256" or sys.argv[3] != "--workspace":
        raise SystemExit("usage: verify_audit.py --audit-manifest-sha256 <sha256> --workspace <already-open-root>")
    result = verify(sys.argv[2], sys.argv[4])
    if EXPECTED_CHECKS and result["checks"] != EXPECTED_CHECKS:
        raise AssertionError("check-count drift")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
