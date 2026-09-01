#!/usr/bin/env python3
"""Hostile pure-stdlib verifier for the sealed source-only FOSP-v3 package."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import math
import re
import stat
import struct
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "ARTIFACT_SHA256SUMS.txt"
EXPECTED_FILES = {
    "README.md",
    "bootstrap_v3.py",
    "free_order_oracle_v3.py",
    "protocol_lock.json",
    "sealed_runtime_probe.py",
    "source_bindings.json",
    "source_only_receipt.json",
    "test_source_only.py",
    "verify_package.py",
}
RECEIPT_HASHED_FILES = EXPECTED_FILES - {"source_only_receipt.json"}
SHA_ROW = re.compile(r"^([0-9a-f]{64})  ([0-9]+)  ([A-Za-z0-9_.-]+)$")
HEAVY = {"cupy", "numpy", "scipy", "torch", "transformers", "cuda"}


class Checks:
    def __init__(self):
        self.count = 0

    def require(self, condition, message):
        if not condition:
            raise AssertionError(message)
        self.count += 1

    def equal(self, actual, expected, message):
        self.require(actual == expected, f"{message}: {actual!r} != {expected!r}")

    def close(self, actual, expected, message, tolerance=1e-14):
        self.require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
                     f"{message}: {actual!r} != {expected!r}")


def regular_bytes(path, checks, expected_size=None):
    info = path.lstat()
    checks.require(stat.S_ISREG(info.st_mode), f"nonregular object forbidden: {path.name}")
    checks.require(not path.is_symlink(), f"symlink forbidden: {path.name}")
    raw = path.read_bytes()
    if expected_size is not None:
        checks.equal(len(raw), expected_size, f"byte count {path.name}")
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reject_duplicate(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def finite_tree(value, checks, label, depth=0):
    checks.require(depth <= 64, f"excess JSON depth in {label}")
    if isinstance(value, float):
        checks.require(math.isfinite(value), f"nonfinite JSON number in {label}")
    elif isinstance(value, dict):
        for key, child in value.items():
            checks.require(isinstance(key, str), f"non-string JSON key in {label}")
            finite_tree(child, checks, label, depth + 1)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child, checks, label, depth + 1)
    else:
        checks.require(value is None or isinstance(value, (str, int, bool)),
                       f"non-JSON value in {label}")


def load_json(path, checks):
    value = json.loads(
        regular_bytes(path, checks).decode("utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    checks.require(isinstance(value, dict), f"JSON root is not object: {path.name}")
    finite_tree(value, checks, path.name)
    return value


def canonical_sha256(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("ascii"))


def parse_manifest(checks, expected_manifest_sha256):
    raw = regular_bytes(MANIFEST, checks)
    checks.equal(sha(raw), expected_manifest_sha256, "externally pinned manifest digest")
    rows = {}
    for number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        match = SHA_ROW.fullmatch(line)
        checks.require(match is not None, f"malformed manifest line {number}")
        digest, size_text, name = match.groups()
        checks.require(name not in rows, f"duplicate manifest member {name}")
        rows[name] = (digest, int(size_text))
    checks.equal(set(rows), EXPECTED_FILES, "manifest file closure")
    observed = {member.name for member in ROOT.iterdir()}
    checks.equal(observed, EXPECTED_FILES | {MANIFEST.name}, "exact package object closure")
    for name in sorted(observed):
        info = (ROOT / name).lstat()
        checks.require(stat.S_ISREG(info.st_mode), f"nonregular package member: {name}")
        checks.require(not (ROOT / name).is_symlink(), f"symlink package member: {name}")
    for name, (digest, size) in sorted(rows.items()):
        raw_member = regular_bytes(ROOT / name, checks, size)
        checks.equal(sha(raw_member), digest, f"artifact SHA-256 {name}")
    return rows, raw


def imported_oracle(checks):
    before = set(sys.modules)
    path = ROOT / "free_order_oracle_v3.py"
    spec = importlib.util.spec_from_file_location("fosp_v3_source_verifier", path)
    checks.require(spec is not None and spec.loader is not None, "oracle import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    added = {name.split(".")[0] for name in set(sys.modules) - before}
    checks.require(not (added & HEAVY), f"heavy import during source verification: {added & HEAVY}")
    return module


def top_imports(tree):
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def matmul(left, right):
    columns = transpose(right)
    return [[dot(row, column) for column in columns] for row in left]


def inverse(matrix):
    n = len(matrix)
    work = [list(row) + [Fraction(int(i == j)) for j in range(n)]
            for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = next((row for row in range(column, n) if work[row][column] != 0), None)
        if pivot is None:
            raise AssertionError("singular fraction fixture")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(n):
            if row == column:
                continue
            scale = work[row][column]
            work[row] = [a - scale * b for a, b in zip(work[row], work[column])]
    return [row[n:] for row in work]


def verify_science(checks, oracle, source, lock):
    f = Fraction
    x = [[f(1), f(0), f(1), f(2), f(0)],
         [f(0), f(2), f(1), f(1), f(1)],
         [f(1), f(1), f(0), f(1), f(2)]]
    y = [[f(2), f(1), f(0), f(1), f(0)],
         [f(0), f(1), f(3), f(0), f(2)],
         [f(1), f(0), f(1), f(1), f(3)]]
    gram = matmul(x, transpose(x))
    cross = matmul(y, transpose(x))
    coefficients = matmul(cross, inverse(gram))
    predicted = matmul(coefficients, x)
    residual = [[a - b for a, b in zip(yr, pr)] for yr, pr in zip(y, predicted)]
    energy = sum((v * v for row in y for v in row), f(0))
    residual_energy = sum((v * v for row in residual for v in row), f(0))
    capture = sum((cross[i][j] * coefficients[i][j] for i in range(3) for j in range(3)), f(0))
    trace_form = matmul(matmul(cross, inverse(gram)), transpose(cross))
    checks.equal(capture, sum(trace_form[i][i] for i in range(3)), "3x3 trace ordering")
    checks.equal(residual_energy, energy - capture, "3x3 regression identity")

    for fragment in (
        'gram = cp.einsum("nrd,nsd->nrs", expert, expert)',
        "cross[:, :, target_role, predecessor_role]",
        'full = cp.einsum("ijab,jbc,ijac->ij", cross, inverse, cross)',
        'exact_coefficients = cp.einsum("eab,ebc->eac", selected_cross, inverse[predecessors])',
        'predicted = cp.einsum("eab,ebd->ead", replay_coefficients, expert[predecessors])',
        'statistics["relaxed_reuse_exact"]["decision_eligible"] = False',
    ):
        checks.require(fragment in source, f"frozen science fragment {fragment}")
    checks.require("HARD_KILL_CONTROL_CORRECTED_RELAXED" not in source,
                   "invalid corrected-relaxed hard kill absent")

    for n in range(2, 8):
        scores = [[f(((target + 1) * 11 + (pred + 1) * 7) % 23 + 1, 23)
                   for pred in range(n)] for target in range(n)]
        relaxed = sum((max(scores[target][pred] for pred in range(n) if pred != target)
                       for target in range(n)), f(0))
        for path in itertools.permutations(range(n)):
            legal = sum((scores[target][pred] for pred, target in zip(path[:-1], path[1:])), f(0))
            checks.require(legal <= relaxed, f"gross relaxed containment n={n}")

    adversarial = oracle.adversarial_n8_statistics()
    checks.equal(struct.unpack("<e", struct.pack("<e", adversarial["r"]))[0], 0.875,
                 "n8 r exact binary16")
    checks.equal(struct.unpack("<e", struct.pack("<e", adversarial["rho"]))[0], 0.765625,
                 "n8 rho exact binary16")
    checks.close(adversarial["corrected_relaxed_s_bpw"], 0.0, "n8 corrected relaxed")
    checks.close(adversarial["qwen_legal_fp16_s_bpw"], 0.7995602818589078, "n8 q legal")
    checks.close(adversarial["control_legal_fp16_s_bpw"], 0.5885652320580218,
                 "n8 control legal")
    checks.close(adversarial["corrected_legal_fp16_s_bpw"], 0.21099504980088601,
                 "n8 corrected legal")
    checks.require(adversarial["corrected_legal_fp16_s_bpw"] > oracle.REQUIRED_GROSS_S,
                   "n8 corrected legal survives")
    corrected = lock["scientific_gate"]["corrected_relaxed_reuse"]
    checks.require(corrected["containing"] is False, "lock corrected relaxed noncontaining")
    checks.require(corrected["decision_eligible"] is False, "lock corrected relaxed diagnostic")
    checks.require(lock["scientific_gate"]["controls"]
                   ["corrected_legal_fp16_computed_before_any_control_corrected_decision"] is True,
                   "lock legal FP16 decision order")


def verify_bootstrap(checks, source, tree, lock):
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    checks.equal(len(imports), 1, "bootstrap import count")
    checks.require(isinstance(imports[0], ast.Import), "bootstrap only direct import")
    checks.equal(imports[0].names[0].name, "sys", "bootstrap only imports sys")
    for fragment in (
        "_sys.flags.isolated",
        "_sys.flags.no_site",
        "_sys.flags.safe_path",
        "_sys.path[:] = []",
        "nonregular package member forbidden",
        "externally pinned package-manifest hash mismatch",
        "snapshot = _snapshot_package(package, expected_package)",
        "runtime exact object closure mismatch",
        "runtime link/special object forbidden",
        "entrypoint is not a sealed package source",
        'source = snapshot[entrypoint].decode("utf-8")',
        'source = self.snapshot[relative].decode("utf-8")',
        '_sys.meta_path[:] = [frozen.BuiltinImporter, frozen.FrozenImporter, sealed_finder]',
    ):
        checks.require(fragment in source, f"bootstrap closure fragment {fragment}")
    checks.require(source.index("_sys.path[:] = []") < source.index("def _snapshot_package"),
                   "ambient import path closed before snapshot function")
    firewall = lock["entrypoint_firewall"]
    for key in (
        "ambient_import_path_cleared_before_package_scan",
        "reject_package_directories",
        "reject_package_symlinks_and_reparse_points",
        "reject_package_sockets_fifos_and_devices",
        "snapshot_all_member_bytes_before_entrypoint_exec",
    ):
        checks.require(firewall[key] is True, f"entrypoint firewall {key}")
    for key in (
        "python_executable_digest_externally_pinned",
        "runtime_manifest_digest_externally_pinned",
        "every_runtime_directory_explicitly_declared",
        "every_runtime_file_size_and_sha256_bound",
        "exact_recursive_object_closure",
        "runtime_symlinks_and_special_objects_forbidden",
        "only_authenticated_import_roots_installed",
        "normal_filesystem_import_finder_removed",
        "authenticated_python_sources_retained_in_memory",
    ):
        checks.require(firewall["runtime"][key] is True, f"runtime firewall {key}")


def verify(expected_manifest_sha256):
    checks = Checks()
    checks.require(re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is not None,
                   "external manifest digest syntax")
    rows, manifest_raw = parse_manifest(checks, expected_manifest_sha256)
    lock = load_json(ROOT / "protocol_lock.json", checks)
    bindings = load_json(ROOT / "source_bindings.json", checks)
    receipt = load_json(ROOT / "source_only_receipt.json", checks)

    sources = {}
    trees = {}
    for name in ("bootstrap_v3.py", "free_order_oracle_v3.py", "sealed_runtime_probe.py", "test_source_only.py",
                 "verify_package.py"):
        sources[name] = regular_bytes(ROOT / name, checks).decode("utf-8")
        trees[name] = ast.parse(sources[name], filename=name)
        checks.require(isinstance(trees[name], ast.Module), f"AST parse {name}")
    oracle = imported_oracle(checks)

    checks.equal(lock["schema"], "free_order_swiglu_path_protocol_v3", "protocol schema")
    checks.equal(lock["status"], "SOURCE_ONLY_DEPLOYMENT_BLOCKED_PENDING_NEW_INDEPENDENT_AUDITS",
                 "protocol status")
    checks.require(lock["lineage"]["v2_files_modified"] is False, "v2 immutability declaration")
    checks.equal(lock["geometry"]["weights_per_expert"], 3 * 768 * 2048, "expert geometry")
    checks.equal(lock["scientific_gate"]["pair_model"]["pair_count_per_expert"], 768 * 767,
                 "ordered nonself pairs")
    checks.equal(lock["eligible_physical_bridge"]["total_side_bits"], 117224, "side bits")
    checks.close(lock["objective"]["required_gross_s_after_side_bpw"], oracle.REQUIRED_GROSS_S,
                 "gross threshold")
    checks.require(lock["execution"]["source_access_authorized"] is False,
                   "source access blocked")
    checks.require(lock["execution"]["calibration_authorized"] is False,
                   "calibration blocked")
    checks.require(lock["execution"]["production_authorization_issued"] is False,
                   "authorization not issued")

    checks.equal(bindings["schema"], "free_order_swiglu_path_auxiliary_bindings_v1",
                 "bindings schema")
    checks.equal([(row["layer"], row["expert"]) for row in bindings["experts"]],
                 [(3, 57), (3, 121)], "fixed expert identities")
    for expert in bindings["experts"]:
        checks.equal([role["role"] for role in expert["roles"]], ["gate", "up", "down"],
                     "three frozen roles")
        for role in expert["roles"]:
            path = Path(role["relative_path"])
            checks.require(not path.is_absolute() and ".." not in path.parts, "safe dormant binding")
            checks.require(re.fullmatch(r"[0-9a-f]{64}", role["sha256"]) is not None,
                           "binding digest syntax")

    checks.require(not (top_imports(trees["free_order_oracle_v3.py"]) & HEAVY),
                   "oracle has no heavy top imports")
    verify_science(checks, oracle, sources["free_order_oracle_v3.py"], lock)
    verify_bootstrap(checks, sources["bootstrap_v3.py"], trees["bootstrap_v3.py"], lock)

    checks.equal(receipt["schema"], "free_order_swiglu_path_v3_source_only_receipt_v1",
                 "receipt schema")
    checks.equal(receipt["status"], "PASS_SOURCE_ONLY_PACKAGE_DEPLOYMENT_BLOCKED",
                 "receipt status")
    checks.equal(receipt["artifact_set_status"], "SEALED_SOURCE_ONLY_ARTIFACT_SET",
                 "receipt artifact status")
    checks.equal(set(receipt["package_files"]), RECEIPT_HASHED_FILES, "receipt file closure")
    for name in sorted(RECEIPT_HASHED_FILES):
        digest, size = rows[name]
        checks.equal(receipt["package_files"][name], {"bytes": size, "sha256": digest},
                     f"receipt package row {name}")
    unsigned = dict(receipt)
    observed_internal = unsigned.pop("canonical_unsigned_sha256", None)
    checks.equal(observed_internal, canonical_sha256(unsigned), "receipt canonical seal")
    zero = receipt["zero_access_ledger"]
    for key, value in zero.items():
        if key.endswith(("files_opened", "bytes_read", "paths_followed", "imports", "calls",
                         "fetches", "connections", "issued", "modified")):
            checks.equal(value, 0, f"zero access {key}")
    checks.require(receipt["deployment"]["authorized"] is False, "receipt deployment blocked")
    checks.require(not (set(sys.modules) & HEAVY), "verifier imported no heavy module")

    return {
        "status": "PASS",
        "checks": checks.count,
        "artifact_manifest_sha256": sha(manifest_raw),
        "source_only_receipt_sha256": rows["source_only_receipt.json"][0],
        "source_only_receipt_internal_sha256": observed_internal,
        "artifacts": {name: {"sha256": digest, "bytes": size}
                      for name, (digest, size) in sorted(rows.items())},
        "deployment_authorized": False,
        "corrected_relaxed_decision_eligible": False,
        "n8_corrected_relaxed_s_bpw": 0.0,
        "n8_corrected_legal_fp16_s_bpw": 0.21099504980088601,
        "zero_payload_access": True,
    }


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--manifest-sha256":
        raise SystemExit("usage: verify_package.py --manifest-sha256 <externally-pinned-sha256>")
    print(json.dumps(verify(sys.argv[2]), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
