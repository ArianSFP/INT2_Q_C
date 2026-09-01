#!/usr/bin/env python3
"""Independent source/result verifier for the MALT64 stage-0 screen."""

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
WORKSPACE = RESEARCH.parent.parent
PRODUCER = RESEARCH / "malt64_decoded_svd_tangent_stage0_v0"
AUDIT_MANIFEST = ROOT / "AUDIT_SHA256SUMS.txt"
EXPECTED_CHECKS = 524
PRODUCER_FILES = {
    "README.md": (2407, "7a86460c0dd473f1988fd34c26cfa7d8beff9b4ef1e17a49ea2eed3f5ae41e7b"),
    "runpod_result.json": (19122, "be374c052a556fdd67020593778fc4d99ce98ee61562e1a8efd16af51f989398"),
    "stage0_screen.py": (15449, "8cca40f82be8397a992e1b488379ece1537bbf330ebc98d646bbe95c17c7d609"),
}
PLAN = (24790, "8017582201468300dd07550a1a2f8d90dc704ffae7ae6d8801a560178e4a1868")
HEADER = (128, "3c16bcf308c0cfce2071be24bf612d202360510084540aa0b358938d8399a538")
POST_SHA = "af801b41a37774d3f0ea65a00d929ff0004122caf4a5632457dbbe232e3f84d0"
REPLAY = (19122, "9de04f91831c7da04f1b908d8cd6381aeaf263dfd0a7e1e7556934e214ade1a5")
AUDIT_FILES = {"README.md", "audit_receipt.json", "verify_audit.py",
               "independent_gpu_replay.py", "disjoint_runpod_result.json"}
MANIFEST_ROW = re.compile(r"^([0-9a-f]{64})  ([0-9]+)  ([A-Za-z0-9_.-]+)$")


class Checks:
    def __init__(self): self.count = 0
    def require(self, value, label):
        if not value: raise AssertionError(label)
        self.count += 1
    def equal(self, actual, expected, label):
        self.require(actual == expected, f"{label}: {actual!r} != {expected!r}")
    def close(self, actual, expected, label, atol=3e-12):
        self.require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol),
                     f"{label}: {actual!r} != {expected!r}")


def sha(raw): return hashlib.sha256(raw).hexdigest()


def held_regular(path, checks, size=None, digest=None):
    path = Path(path)
    named = path.lstat()
    checks.require(stat.S_ISREG(named.st_mode) and not path.is_symlink(), f"regular {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) |
                         getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if size is not None: checks.equal(before.st_size, size, f"bytes {path}")
        chunks, remaining = [], before.st_size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            checks.require(bool(block), f"short read {path}")
            chunks.append(block); remaining -= len(block)
        checks.equal(os.read(descriptor, 1), b"", f"EOF {path}")
        after = os.fstat(descriptor)
        checks.equal((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                     (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                     f"held identity {path}")
        raw = b"".join(chunks)
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


def exact_audit(checks, expected_pin):
    manifest_raw = held_regular(AUDIT_MANIFEST, checks)
    checks.equal(sha(manifest_raw), expected_pin, "external audit manifest pin")
    rows = {}
    for number, line in enumerate(manifest_raw.decode("ascii").splitlines(), 1):
        match = MANIFEST_ROW.fullmatch(line)
        checks.require(match is not None, f"manifest row {number}")
        digest, size, name = match.groups()
        checks.require(name not in rows, f"duplicate manifest row {name}")
        rows[name] = (int(size), digest)
    checks.equal(set(rows), AUDIT_FILES, "audit manifest row closure")
    checks.equal({entry.name for entry in os.scandir(ROOT)}, AUDIT_FILES | {AUDIT_MANIFEST.name},
                 "audit directory closure")
    return {name: held_regular(ROOT / name, checks, size, digest)
            for name, (size, digest) in sorted(rows.items())}, manifest_raw


def normalized_replay(value):
    value = json.loads(json.dumps(value, allow_nan=False))
    del value["execution"]["elapsed_seconds"]
    return value


def synthetic_tangent_checks(np, checks):
    # Independent algebra/numerics: explicit projection and the alternative
    # norm identity used by the supplied independent replay must agree.
    for seed in range(16):
        random = np.random.Generator(np.random.PCG64(91001 + seed))
        decoded = random.normal(size=(7, 7))
        error = random.normal(size=(7, 7))
        u, _s, vh = np.linalg.svd(decoded, full_matrices=False)
        u, v = u[:, :2], vh[:2, :]
        left = u @ (u.T @ error)
        right = (error @ v.T) @ v
        cross = (left @ v.T) @ v
        projection = left + right - cross
        ute, ev, utev = u.T @ error, error @ v.T, (u.T @ error) @ v.T
        alternate = float(np.sum(ute * ute) + np.sum(ev * ev) - np.sum(utev * utev))
        explicit = float(np.sum(projection * projection))
        residual = error - projection
        checks.close(alternate, explicit, "synthetic alternate projection energy", 2e-12)
        checks.close(float(np.sum(error * projection)), explicit,
                     "synthetic orthogonal projection inner product", 2e-12)
        checks.require(float(np.max(np.abs(u.T @ residual))) < 2e-12,
                       "synthetic residual orthogonal to left tangent")
        checks.require(float(np.max(np.abs(residual @ v.T))) < 2e-12,
                       "synthetic residual orthogonal to right tangent")
        checks.require(-1e-12 <= explicit <= float(np.sum(error * error)) + 1e-12,
                       "synthetic projection energy bounds")


def verify(expected_pin, plan_path, header_path):
    checks = Checks()
    checks.require(re.fullmatch(r"[0-9a-f]{64}", expected_pin) is not None, "pin syntax")
    producer = exact_tree(PRODUCER, PRODUCER_FILES, checks, "producer")
    audit, manifest_raw = exact_audit(checks, expected_pin)
    result = strict_json(producer["runpod_result.json"])
    replay = strict_json(audit["disjoint_runpod_result.json"])
    checks.equal(sha(audit["disjoint_runpod_result.json"]), REPLAY[1], "replay digest")
    checks.equal(normalized_replay(replay), normalized_replay(result),
                 "disjoint GPU replay exact except elapsed time")
    checks.equal(result["schema"], "malt64_decoded_svd_tangent_stage0_result_v0", "schema")

    plan_raw = held_regular(plan_path, checks, *PLAN)
    header_raw = held_regular(header_path, checks, *HEADER)
    plan = strict_json(plan_raw)
    clean = dict(plan); claimed_lock = clean.pop("lock_sha256")
    checks.equal(canonical_sha(clean), claimed_lock, "plan internal canonical seal")
    checks.equal(result["bindings"]["plan_sha256"], PLAN[1], "result plan binding")
    checks.equal(result["bindings"]["plan_internal_lock_sha256"], claimed_lock,
                 "result internal plan binding")
    checks.equal(result["bindings"]["header_sha256"], HEADER[1], "result header binding")
    checks.equal(result["bindings"]["post_klt_sha256"], POST_SHA, "result decoded binding")
    coefficients = struct.unpack_from("<12f", header_raw, 32)
    checks.require(all(math.isfinite(value) for value in coefficients), "finite header coefficients")
    for expert in range(6):
        cosine, sine = coefficients[2 * expert:2 * expert + 2]
        checks.require(cosine * cosine + sine * sine > 0.0, "invertible header rotation")

    plan_sources = plan["sources"]
    checks.equal(len(plan_sources), 18, "18 plan sources")
    result_sources = result["bindings"]["sources"]
    checks.equal(len(result_sources), 18, "18 result source receipts")
    for ordinal, (source, receipt) in enumerate(zip(plan_sources, result_sources)):
        role = ("gate", "up", "down")[ordinal % 3]
        checks.equal(source["matrix_ordinal"], ordinal, "plan matrix ordinal")
        checks.equal(source["role"], role, "plan role order")
        checks.equal(source["shape"], [2048, 768] if role == "down" else [768, 2048],
                     "plan source geometry")
        relative = Path(source["source_relpath"])
        checks.require(not relative.is_absolute() and ".." not in relative.parts,
                       "safe plan source path")
        checks.equal(receipt["matrix_ordinal"], ordinal, "receipt ordinal")
        checks.equal(receipt["bytes"], 3145728, "receipt byte count")
        checks.equal(receipt["sha256"], source["source_bf16_sha256"],
                     "receipt/plan source hash")
        checks.require("validation" not in source["source_relpath"].lower(),
                       "no fresh-validation path")

    architecture = result["architecture"]
    checks.equal(architecture["block_shape"], [64, 64], "block geometry")
    checks.equal(architecture["coarse_block_values"], 4096, "block values")
    checks.equal(architecture["decoded_svd_rank"], 3, "SVD rank")
    dimension = 3 * (64 + 64 - 3)
    checks.equal(dimension, 375, "tangent dimension derivation")
    checks.equal(architecture["continuous_tangent_dimension"], dimension,
                 "recorded tangent dimension")
    checks.close(architecture["tangent_rank_fraction"], dimension / 4096,
                 "tangent rank fraction")
    checks.close(architecture["null_isotropic_capture"], dimension / 4096,
                 "null isotropic share")
    checks.equal(architecture["coset_bits_per_block"], 384, "future target bits")

    ledger = result["physical_planning_ledger"]
    coarse, coset, metadata = 307 / 128, 384 / 4096, 1 / 128
    checks.close(ledger["coarse_bpw"], coarse, "coarse ledger")
    checks.close(ledger["coset_bpw"], coset, "coset ledger")
    checks.close(ledger["metadata_bpw"], metadata, "metadata ledger")
    checks.close(ledger["total_bpw"], coarse + coset + metadata, "total ledger")
    checks.close(ledger["total_bpw"], 2.5, "exact 2.5 rate")
    checks.close(ledger["cold_page_read_amplification"], 73 / 72,
                 "cold page read ledger")
    base_f = 0.9888693569009007
    assumed_coarse_mse = base_f * 2.0 ** (-2.0 * coarse)
    target_mse = 0.8 * 2.0 ** (-2.0 * 2.5)
    required = 1.0 - target_mse / assumed_coarse_mse
    checks.close(assumed_coarse_mse, 0.035574242296714034, "assumed coarse MSE")
    checks.close(target_mse, 0.025, "target relative MSE")
    checks.close(required, 0.2972443434920543, "required capture derivation")
    checks.close(ledger["required_coarse_error_capture"], required, "ledger threshold")
    checks.close(ledger["favourable_base_F_transfer"], base_f, "favourable F transfer")

    matrices = result["matrices"]
    checks.equal(len(matrices), 18, "18 matrix rows")
    checks.equal(sum(row["blocks"] for row in matrices), 6912, "all 6912 blocks")
    for ordinal, row in enumerate(matrices):
        checks.equal(row["matrix_ordinal"], ordinal, "matrix row ordinal")
        checks.equal(row["expert_ordinal"], ordinal // 3, "matrix expert ordinal")
        checks.equal(row["role"], ("gate", "up", "down")[ordinal % 3], "matrix role")
        checks.equal(row["blocks"], 384, "matrix block count")
        checks.require(0.0 <= row["tangent_projection_energy_fp64"] <= row["coarse_error_sse_fp64"],
                       "matrix projection energy bounds")
        checks.close(row["capture_fraction"],
                     row["tangent_projection_energy_fp64"] / row["coarse_error_sse_fp64"],
                     "matrix capture fraction")

    total_energy = math.fsum(row["source_energy_fp64"] for row in matrices)
    total_error = math.fsum(row["coarse_error_sse_fp64"] for row in matrices)
    total_capture = math.fsum(row["tangent_projection_energy_fp64"] for row in matrices)
    aggregate = result["aggregate"]
    checks.close(aggregate["source_energy_fp64"], total_energy, "aggregate source energy")
    checks.close(aggregate["coarse_error_sse_fp64"], total_error, "aggregate SSE")
    checks.close(aggregate["tangent_projection_energy_fp64"], total_capture,
                 "aggregate projection energy")
    checks.close(aggregate["coarse_relative_mse"], total_error / total_energy,
                 "coarse relative MSE")
    estimate = total_capture / total_error
    checks.close(aggregate["capture_fraction"], estimate, "aggregate capture")
    checks.close(total_error, 500.39553685426534, "frozen base SSE")
    checks.close(total_energy, 16192.89450885593, "frozen base energy", 5e-12)

    for expert, observed in enumerate(result["experts"]):
        subset = [row for row in matrices if row["expert_ordinal"] == expert]
        error = math.fsum(row["coarse_error_sse_fp64"] for row in subset)
        capture = math.fsum(row["tangent_projection_energy_fp64"] for row in subset)
        checks.equal(observed["expert_ordinal"], expert, "expert row ordinal")
        checks.close(observed["coarse_error_sse_fp64"], error, "expert SSE")
        checks.close(observed["tangent_projection_energy_fp64"], capture, "expert capture")
        checks.close(observed["capture_fraction"], capture / error, "expert ratio")
    for role, observed in zip(("gate", "up", "down"), result["roles"]):
        subset = [row for row in matrices if row["role"] == role]
        error = math.fsum(row["coarse_error_sse_fp64"] for row in subset)
        capture = math.fsum(row["tangent_projection_energy_fp64"] for row in subset)
        checks.equal(observed["role"], role, "role row identity")
        checks.close(observed["coarse_error_sse_fp64"], error, "role SSE")
        checks.close(observed["tangent_projection_energy_fp64"], capture, "role capture")
        checks.close(observed["capture_fraction"], capture / error, "role ratio")

    expert_errors = [row["coarse_error_sse_fp64"] for row in result["experts"]]
    expert_captures = [row["tangent_projection_energy_fp64"] for row in result["experts"]]
    deletes = [(total_capture - expert_captures[i]) / (total_error - expert_errors[i])
               for i in range(6)]
    center = math.fsum(deletes) / 6
    se = math.sqrt(5 / 6 * math.fsum((value - center) ** 2 for value in deletes))
    uncertainty = aggregate["uncertainty"]
    for observed, expected in zip(uncertainty["delete_one_expert"], deletes):
        checks.close(observed, expected, "delete-one expert ratio")
    checks.close(uncertainty["estimate"], estimate, "jackknife estimate")
    checks.close(uncertainty["jackknife_center"], center, "jackknife center")
    checks.close(uncertainty["jackknife_se"], se, "jackknife SE")
    upper = estimate + 3.0 * se
    checks.close(uncertainty["upper_three_se"], upper, "upper three SE")
    checks.close(aggregate["fraction_of_required_at_upper_three_se"], upper / required,
                 "fraction required at UCB")
    checks.require(upper < required, "UCB misses threshold")
    checks.equal(result["decision"], "POLICY_REJECT_MALT64_R3_FAR_SHORT_STOP_BEFORE_CONTROLS",
                 "hard-kill decision")
    checks.require("controls" not in result and "finite" not in result,
                   "post-kill stages absent")

    producer_source = producer["stage0_screen.py"].decode("utf-8")
    producer_ast = ast.parse(producer_source)
    checks.require(isinstance(producer_ast, ast.Module), "producer AST")
    for fragment in ("left + right - cross", "upper_three_se\"] < REQUIRED_CAPTURE",
                     "if output.exists()", "source_root / row[\"source_relpath\"]"):
        checks.require(fragment in producer_source, "producer algorithm/decision fragment " + fragment)
    checks.require("fresh_validation" not in producer_source and "validation/" not in producer_source,
                   "no fresh-validation code path")
    independent_source = audit["independent_gpu_replay.py"].decode("utf-8")
    independent_ast = ast.parse(independent_source)
    checks.require(isinstance(independent_ast, ast.Module), "independent replay AST")
    for fragment in ("cp.sum(ute * ute", "cp.sum(ev * ev", "cp.sum(utev * utev",
                     '"fresh_validation_files_opened": 0'):
        checks.require(fragment in independent_source, "independent alternate-method fragment " + fragment)
    checks.require("left + right - cross" not in independent_source,
                   "independent replay does not copy explicit projection")

    sys.path.insert(0, os.fspath(WORKSPACE / ".deps"))
    import numpy as np
    synthetic_tangent_checks(np, checks)
    checks.require("not a finite codec" in result["claim_boundary"] and
                   "universal converse" in producer["README.md"].decode("utf-8"),
                   "claim boundary")

    receipt = strict_json(audit["audit_receipt.json"])
    unsigned = dict(receipt); seal = unsigned.pop("canonical_unsigned_sha256")
    checks.equal(seal, canonical_sha(unsigned), "receipt canonical seal")
    checks.equal(receipt["verdict"], "PASS", "receipt verdict")
    checks.equal(receipt["independent_verifier"]["expected_checks"], EXPECTED_CHECKS,
                 "receipt check count")
    checks.close(receipt["recomputed"]["capture_fraction"], estimate, "receipt capture")
    checks.close(receipt["recomputed"]["upper_three_se"], upper, "receipt UCB")
    return {"status": "PASS", "verdict": "PASS", "checks": checks.count,
            "expected_checks": EXPECTED_CHECKS, "audit_manifest_sha256": sha(manifest_raw),
            "producer_result_sha256": PRODUCER_FILES["runpod_result.json"][1],
            "disjoint_runpod_result_sha256": REPLAY[1], "plan_sha256": PLAN[1],
            "post_klt_sha256": POST_SHA, "blocks_replayed_by_disjoint_run": 6912,
            "capture_fraction": estimate, "upper_three_se": upper,
            "required_capture": required, "decision": result["decision"],
            "fresh_validation_files_opened": 0,
            "independent_full_gpu_replay": "NOT_EXECUTED_UPLOAD_DENIED"}


def main():
    if (len(sys.argv) != 7 or sys.argv[1] != "--audit-manifest-sha256" or
            sys.argv[3] != "--plan" or sys.argv[5] != "--header"):
        raise SystemExit("usage: verify_audit.py --audit-manifest-sha256 <sha256> --plan <plan.lock.json> --header <header.bin>")
    result = verify(sys.argv[2], sys.argv[4], sys.argv[6])
    if EXPECTED_CHECKS and result["checks"] != EXPECTED_CHECKS:
        raise AssertionError("check-count drift")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__": main()
