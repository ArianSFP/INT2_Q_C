#!/usr/bin/env python3
"""Independent result verifier for the FOSP auxiliary gross screen."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import os
import re
import stat
import sys
from fractions import Fraction
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
PRODUCER = RESEARCH / "free_order_swiglu_path_aux_gross_v0"
ORACLE_PATH = RESEARCH / "free_order_swiglu_path_oracle_v4" / "scientific_oracle_v3.py"
BINDINGS_PATH = RESEARCH / "free_order_swiglu_path_oracle_v3" / "source_bindings.json"
AUDIT_MANIFEST = ROOT / "AUDIT_SHA256SUMS.txt"
EXPECTED_CHECKS = 6154

PRODUCER_FILES = {
    "README.md": (1468, "e9402388ec8c04072590747d333e9329ee328e11ef102fcbae3de1cabac05de6"),
    "gross_screen.py": (10472, "c8a08fa8dec5279380bc0d09fd2a6089b56a7a97faa16fac88010c3226b429ab"),
    "runpod_result.json": (6778, "52d9795004299e0e6bc055de7d900edaf8a89fc485a316e82c848682e73ecd7c"),
}
ORACLE_SHA256 = "9ca6f4bdd4150c8c0c68c0a298c00eb45c088a4af287895ebfdf9bf1e661a070"
BINDINGS_SHA256 = "cd12742910503f23d0d9224e277a030b923f8fc917c75a13a1aff8e9bcde090a"
REPLAY = (6778, "7e44d40926d578c88083d9a40a77334c55ca70710a4e203cef566e839156e611")
AUDIT_FILES = {"README.md", "audit_receipt.json", "verify_audit.py", "disjoint_runpod_result.json"}
AUDIT_ROW = re.compile(r"^([0-9a-f]{64})  ([0-9]+)  ([A-Za-z0-9_.-]+)$")


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
        chunks, remaining = [], before.st_size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            checks.require(bool(block), f"short read {path}")
            chunks.append(block); remaining -= len(block)
        checks.equal(os.read(descriptor, 1), b"", f"EOF {path.name}")
        after = os.fstat(descriptor)
        checks.equal((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                     (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                     f"identity {path.name}")
        raw = b"".join(chunks)
        if digest is not None: checks.equal(sha(raw), digest, f"SHA-256 {path.name}")
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


def exact_small_tree(root, expected, checks, label):
    checks.equal({entry.name for entry in os.scandir(root)}, set(expected), label + " closure")
    return {name: held_regular(root / name, checks, size, digest)
            for name, (size, digest) in sorted(expected.items())}


def exact_audit(checks, expected_digest):
    raw = held_regular(AUDIT_MANIFEST, checks)
    checks.equal(sha(raw), expected_digest, "audit manifest external pin")
    rows = {}
    for number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        match = AUDIT_ROW.fullmatch(line)
        checks.require(match is not None, f"audit manifest row {number}")
        digest, size, name = match.groups()
        checks.require(name not in rows, f"duplicate audit row {name}")
        rows[name] = (int(size), digest)
    checks.equal(set(rows), AUDIT_FILES, "audit row closure")
    checks.equal({entry.name for entry in os.scandir(ROOT)}, AUDIT_FILES | {AUDIT_MANIFEST.name},
                 "audit object closure")
    return {name: held_regular(ROOT / name, checks, size, digest)
            for name, (size, digest) in sorted(rows.items())}, raw


def load_held_source(raw, label):
    namespace = {"__name__": "independent_" + label, "__file__": "<held-" + label + ">"}
    exec(compile(raw.decode("utf-8"), namespace["__file__"], "exec", dont_inherit=True),
         namespace, namespace)
    return namespace


def verify_containment(checks):
    # Every non-anchor path target contributes one nonself row entry no larger
    # than that target's relaxed row maximum; regression captures are >= 0 and
    # the relaxed sum additionally includes the anchor target maximum.
    for n in range(2, 8):
        scores = [[Fraction(((target + 3) * 13 + (pred + 5) * 11) % 29 + 1, 29)
                   for pred in range(n)] for target in range(n)]
        relaxed = sum((max(scores[t][p] for p in range(n) if p != t)
                       for t in range(n)), Fraction(0))
        for path in itertools.permutations(range(n)):
            legal = sum((scores[target][pred] for pred, target in zip(path[:-1], path[1:])),
                        Fraction(0))
            checks.require(legal <= relaxed, f"relaxed contains path n={n}")


def normalized_replay(value):
    value = json.loads(json.dumps(value, allow_nan=False))
    del value["runtime"]["elapsed_seconds"]
    return value


def verify(expected_audit_manifest_sha256, workspace):
    checks = Checks()
    checks.require(re.fullmatch(r"[0-9a-f]{64}", expected_audit_manifest_sha256) is not None,
                   "audit pin syntax")
    producer = exact_small_tree(PRODUCER, PRODUCER_FILES, checks, "producer")
    audit, audit_manifest = exact_audit(checks, expected_audit_manifest_sha256)
    result = strict_json(producer["runpod_result.json"])
    replay = strict_json(audit["disjoint_runpod_result.json"])
    checks.equal(sha(audit["disjoint_runpod_result.json"]), REPLAY[1], "disjoint replay digest")
    checks.equal(normalized_replay(replay), normalized_replay(result),
                 "disjoint GPU replay semantic identity except elapsed time")
    checks.equal(result["schema"], "free_order_swiglu_path_aux_gross_v0_result", "result schema")

    oracle_raw = held_regular(ORACLE_PATH, checks, 21887, ORACLE_SHA256)
    bindings_raw = held_regular(BINDINGS_PATH, checks, 2683, BINDINGS_SHA256)
    oracle = load_held_source(oracle_raw, "fosp_oracle")
    bindings = strict_json(bindings_raw)
    checks.equal(bindings["schema"], "free_order_swiglu_path_auxiliary_bindings_v1",
                 "binding schema")
    checks.equal(result["science"]["oracle_sha256"], ORACLE_SHA256, "result oracle binding")
    checks.equal(result["science"]["source_bindings_sha256"], BINDINGS_SHA256,
                 "result source binding")
    checks.equal(result["bindings"]["script_sha256"], PRODUCER_FILES["gross_screen.py"][1],
                 "result script binding")

    workspace = Path(workspace).resolve()
    sys.path.insert(0, os.fspath(workspace / ".deps"))
    import numpy as np
    checks.equal(np.__version__, result["runtime"]["numpy"], "CPU replay NumPy version")
    rows_by_identity = {(row["layer"], row["expert"], row["role"]): row
                        for row in result["bindings"]["source_receipts"]}
    expected_identities = {(expert["layer"], expert["expert"], role["role"])
                           for expert in bindings["experts"] for role in expert["roles"]}
    checks.equal(set(rows_by_identity), expected_identities, "six source receipt identities")

    replay_rows = []
    for expert in bindings["experts"]:
        role_values = {}
        for role in expert["roles"]:
            identity = (expert["layer"], expert["expert"], role["role"])
            receipt = rows_by_identity[identity]
            checks.equal(receipt["sha256"], role["sha256"], f"receipt/binding hash {identity}")
            checks.equal(receipt["shape"], role["shape"], f"receipt/binding shape {identity}")
            checks.equal(receipt["bytes"], math.prod(role["shape"]) * 2,
                         f"receipt byte ledger {identity}")
            local = workspace / "qwen_weight_cache" / "tensors" / Path(role["relative_path"]).name
            raw = held_regular(local, checks, receipt["bytes"], receipt["sha256"])
            words = np.frombuffer(raw, dtype="<u2")
            values = (words.astype(np.uint32) << np.uint32(16)).view(np.float32)
            role_values[role["role"]] = values.reshape(tuple(role["shape"]))
        joined = np.stack((role_values["gate"], role_values["up"], role_values["down"].T), axis=1)
        checks.equal(joined.shape, (768, 3, 2048), "canonical expert geometry")
        source = np.asarray(joined, dtype=np.float64)
        energy = float(np.sum(source * source, dtype=np.float64))
        scores, _cross, _inverse = oracle["_pair_scores"](source, np)
        diagonal = np.arange(768)
        checks.equal(int(np.count_nonzero(np.isneginf(scores))), 768, "exact diagonal -inf count")
        checks.require(bool(np.isneginf(scores[diagonal, diagonal]).all()), "all self pairs excluded")
        best = np.max(scores, axis=1)
        predecessors = np.argmax(scores, axis=1).astype("<u2")
        checks.require(not bool(np.any(predecessors == np.arange(768, dtype=np.uint16))),
                       "CPU replay has no self predecessor")
        capture = float(np.sum(best, dtype=np.float64))
        replay_rows.append({"ordinal": expert["ordinal"], "layer": expert["layer"],
                            "expert": expert["expert"], "source_energy": energy,
                            "gross_relaxed_capture": capture,
                            "best_predecessor_sha256_u16le": sha(predecessors.tobytes())})

    result_rows = sorted(result["experts"], key=lambda row: row["ordinal"])
    for observed, expected in zip(replay_rows, result_rows):
        checks.equal((observed["ordinal"], observed["layer"], observed["expert"]),
                     (expected["ordinal"], expected["layer"], expected["expert"]),
                     "CPU replay expert identity")
        checks.close(observed["source_energy"], expected["source_energy"], "CPU source energy", 2e-10)
        checks.close(observed["gross_relaxed_capture"], expected["gross_relaxed_capture"],
                     "CPU gross capture", 2e-9)
        checks.equal(observed["best_predecessor_sha256_u16le"],
                     expected["best_predecessor_sha256_u16le"], "CPU predecessor digest")
        reduction = expected["gross_relaxed_capture"] / expected["source_energy"]
        checks.close(expected["energy_reduction"], reduction, "row energy reduction")
        checks.close(expected["gross_s_bpw"], -0.5 * math.log2(1.0 - reduction), "row gross s")
        checks.equal(expected["ordered_nonself_pairs"], 768 * 767, "row nonself pair count")

    total_energy = math.fsum(row["source_energy"] for row in result_rows)
    total_capture = math.fsum(row["gross_relaxed_capture"] for row in result_rows)
    ratio = (total_energy - total_capture) / total_energy
    gross_s = -0.5 * math.log2(ratio)
    aggregate = result["aggregate"]
    checks.close(aggregate["source_energy"], total_energy, "aggregate energy")
    checks.close(aggregate["gross_relaxed_capture"], total_capture, "aggregate capture")
    checks.close(aggregate["energy_reduction"], total_capture / total_energy, "aggregate reduction")
    checks.close(aggregate["residual_ratio"], ratio, "aggregate residual ratio")
    checks.close(aggregate["gross_s_bpw"], gross_s, "aggregate gross s")
    checks.close(aggregate["net_s_after_side_bpw"], gross_s - oracle["SIDE_BPW"], "aggregate net s")
    checks.close(aggregate["fraction_of_required_gross_s"], gross_s / oracle["REQUIRED_GROSS_S"],
                 "fraction of requirement")
    checks.close(aggregate["projected_optimistic_F_after_side"],
                 2.0 ** (-2.0 * (gross_s - oracle["SIDE_BPW"])), "projected F")

    for rate in oracle["RATES"]:
        expected = oracle["frame_ledger"](rate)
        observed = result["physical_ledgers"][format(rate, ".2f")]
        checks.equal(set(observed), set(expected), f"ledger fields {rate}")
        for key in expected:
            if isinstance(expected[key], float): checks.close(observed[key], expected[key], f"ledger {rate} {key}")
            else: checks.equal(observed[key], expected[key], f"ledger {rate} {key}")

    verify_containment(checks)
    tree = ast.parse(producer["gross_screen.py"].decode("utf-8"))
    source = producer["gross_screen.py"].decode("utf-8")
    for fragment in ("full[indices, indices] = -cp.inf", "cp.isneginf(scores[diagonal, diagonal])",
                     "oracle.ROWS * (oracle.ROWS - 1)", "if gross_s < oracle.REQUIRED_GROSS_S"):
        checks.require(fragment in (oracle_raw.decode("utf-8") if fragment.startswith("full[") else source),
                       "held self-mask/decision fragment " + fragment)
    checks.require(isinstance(tree, ast.Module), "runner AST")
    checks.require(gross_s < oracle["REQUIRED_GROSS_S"], "gross bound misses required threshold")
    checks.equal(result["status"], "HARD_KILL_GROSS_QWEN_RELAXED_NECESSARY_BOUND", "hard kill status")
    checks.require(result["early_stop"] is True, "early stop true")
    checks.require("controls" not in result and "legal_path" not in result, "post-kill stages absent")
    checks.require(result["science"]["gross_relaxed_contains_every_legal_path"] is True,
                   "result containment claim")
    checks.require(result["science"]["controls_required_after_gross_survival_only"] is True,
                   "controls skipped only after miss")
    checks.equal(result["access"], {"auxiliary_qwen_files_opened": 6,
                                    "fresh_validation_files_opened": 0,
                                    "network_operations": 0,
                                    "pinned_panel_files_opened": 0}, "access boundary")
    checks.require("not fresh validation" in result["claim_boundary"] and
                   "not a validation" in producer["README.md"].decode("utf-8"),
                   "auxiliary-only claim boundary")

    receipt = strict_json(audit["audit_receipt.json"])
    unsigned = dict(receipt); observed_seal = unsigned.pop("canonical_unsigned_sha256")
    checks.equal(observed_seal, canonical_sha(unsigned), "audit receipt seal")
    checks.equal(receipt["verdict"], "PASS", "audit verdict")
    checks.equal(receipt["independent_verifier"]["expected_checks"], EXPECTED_CHECKS,
                 "receipt check count")
    checks.equal(receipt["recomputed"]["aggregate_gross_s_bpw"], gross_s,
                 "receipt gross s")
    checks.equal(receipt["recomputed"]["cpu_predecessor_hashes"],
                 [row["best_predecessor_sha256_u16le"] for row in replay_rows],
                 "receipt CPU predecessor hashes")
    return {"status": "PASS", "verdict": "PASS", "checks": checks.count,
            "expected_checks": EXPECTED_CHECKS, "audit_manifest_sha256": sha(audit_manifest),
            "producer_result_sha256": PRODUCER_FILES["runpod_result.json"][1],
            "disjoint_runpod_result_sha256": REPLAY[1],
            "cpu_replay": "PASS_TWO_EXPERTS_ALL_NONSELF_PAIRS",
            "aggregate_gross_s_bpw": gross_s,
            "required_gross_s_bpw": oracle["REQUIRED_GROSS_S"],
            "decision": result["status"], "early_stop": True,
            "fresh_or_pinned_files_opened": 0}


def main():
    if len(sys.argv) != 5 or sys.argv[1] != "--audit-manifest-sha256" or sys.argv[3] != "--workspace":
        raise SystemExit("usage: verify_audit.py --audit-manifest-sha256 <sha256> --workspace <already-open-root>")
    result = verify(sys.argv[2], sys.argv[4])
    if EXPECTED_CHECKS and result["checks"] != EXPECTED_CHECKS:
        raise AssertionError("check-count drift")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
