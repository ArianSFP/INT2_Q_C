#!/usr/bin/env python3
"""Independent hostile, source-only verifier for FOSP-ARX-v3.

This program opens and hashes the complete v3 package, the immutable v2
package, the sealed v2 BLOCK audit, and its own audit artifacts.  It never
resolves a source binding and never imports a model, NumPy, CuPy, SciPy,
Torch, CUDA, or a payload library.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import os
import re
import stat
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
V3 = RESEARCH / "free_order_swiglu_path_oracle_v3"
V2 = RESEARCH / "free_order_swiglu_path_oracle_v2"
V2_AUDIT = RESEARCH / "free_order_swiglu_path_oracle_v2_independent_source_audit_20260901"
AUDIT_MANIFEST = ROOT / "AUDIT_SHA256SUMS.txt"

V3_MANIFEST_SHA256 = "8584bde5c09fb7df531884c2100d3892dd1e12fbe689cf5d23ce091918d96470"
V2_MANIFEST_SHA256 = "a7dc083b8fec762759100c23bb5e8f140642e1032fcbd4b4845be785e7e841f9"
V2_AUDIT_MANIFEST_SHA256 = "12fac1817e48fa396a65c0796601b2a36beee8141051e1ac5faa2466f33344cb"
EXPECTED_CHECKS = 0  # Updated only after the first complete sealed replay.

V3_FILES = {
    "ARTIFACT_SHA256SUMS.txt": (828, V3_MANIFEST_SHA256),
    "README.md": (6386, "d737a3d76b59e0cf7f984afb78d97da458a843f17268bcec96dc841b75326566"),
    "bootstrap_v3.py": (17922, "3f00b13bb3d9b913338bcf9c57cc0f6aa31f9d32d148d5ae1ebf046fbe91cadc"),
    "free_order_oracle_v3.py": (21887, "9ca6f4bdd4150c8c0c68c0a298c00eb45c088a4af287895ebfdf9bf1e661a070"),
    "protocol_lock.json": (7469, "f4660cb8876a749eb1635dbf010a8df6199e845b0517dd8b15039ac9cf1fd097"),
    "sealed_runtime_probe.py": (325, "5bad31a964f7049cf0f2d3a557153b38e4613f9c9ccee608c30d21d1bc5ad76d"),
    "source_bindings.json": (2683, "cd12742910503f23d0d9224e277a030b923f8fc917c75a13a1aff8e9bcde090a"),
    "source_only_receipt.json": (3721, "753d0db72cf104c40bbe1b8a6fea7a62055ba31f90d22520e31d59349c132883"),
    "test_source_only.py": (17086, "63b6bf1139ba50ad9b6bb87337b2dba8109a4558ec7dacf1f636d1d38b1e5110"),
    "verify_package.py": (17939, "197e1bd652a865813e1b9e0699d4f159ac58306cf0dc9b2f6b20a187661a44f5"),
}
V2_FILES = {
    "ARTIFACT_SHA256SUMS.txt": (776, V2_MANIFEST_SHA256),
    "README.md": (14049, "af56fa26e813d9c4f227c7b96ccdf45fc28367cbe1840ff1311b3973f29142e0"),
    "calibrate_runtime.py": (6709, "526c5d40304cba9eb6ca8449c0aea734aaeb1be299eab2c89bfea05abae6a883"),
    "create_authorization.py": (17377, "247a6de24279f4f1cacf6f2ba34abfdf6e9f4aef2fb35701c3ce4a931308c18e"),
    "free_order_oracle_v2.py": (54317, "98625fde2c0f1bd2510c5f07fbd9384953fcb8e9b7f7f98d284482c034a94eb2"),
    "protocol_lock.json": (11962, "57a7e4f36a13d12ff61c7966306af8135d8ba182de400106548d314d462380e6"),
    "source_bindings.json": (2878, "3454b718a65efc02c32463f955c10ff393f4218fac04f358107960ff3735990d"),
    "source_only_receipt.json": (4383, "0fe2e050f0e2ab345e16414d0f3cb64f460586ba4612a009417802ba5711abfe"),
    "test_source_only.py": (24617, "559c7660d81a2a13a88ef5c86c726764d3fb6337fff530c6c995270806294964"),
    "verify_package.py": (15176, "d18e12f65c5abc7a8dc7157e385c2aa856ac39f429463f7e9cb0c4911911f20a"),
}
V2_AUDIT_FILES = {
    "AUDIT_SHA256SUMS.txt": (329, V2_AUDIT_MANIFEST_SHA256),
    "README.md": (6444, "dbaa3034f74ed57dcd6d4581fb44d5f30815695ce5572fe2412fee5f9a783c8c"),
    "audit_receipt.json": (9995, "55587205a3f7576cb5d4fceb64be89d0e8c07279b17b260d5ef64b1e847b461c"),
    "replay_receipt.json": (2326, "b69c62a01a7d6df9c0bb6a3ca2b724e33ee1a7aafde7511b0b70696a5d3582bb"),
    "verify_audit.py": (33797, "51e2b8a972ccfd7d19a2ec65b2d142e126558fcb666b154a5ec2e6ba13e4abe0"),
}
AUDIT_FILES = {"README.md", "audit_receipt.json", "verify_audit.py"}
HEAVY = {"cupy", "numpy", "scipy", "torch", "transformers", "cuda"}
ROW_WITH_SIZE = re.compile(r"^([0-9a-f]{64})  ([0-9]+)  ([A-Za-z0-9_.-]+)$")
ROW_NO_SIZE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")


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


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def is_reparse(info):
    return bool(getattr(info, "st_file_attributes", 0) & 0x400)


def held_regular_bytes(path, checks, expected_size=None):
    before_name = path.lstat()
    checks.require(stat.S_ISREG(before_name.st_mode), f"nonregular object: {path}")
    checks.require(not path.is_symlink() and not is_reparse(before_name), f"link/reparse: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        checks.require(stat.S_ISREG(before.st_mode) and not is_reparse(before),
                       f"opened object not regular: {path}")
        if expected_size is not None:
            checks.equal(before.st_size, expected_size, f"byte count {path.name}")
        pieces = []
        remaining = before.st_size
        while remaining:
            piece = os.read(descriptor, min(1 << 20, remaining))
            checks.require(bool(piece), f"short read {path}")
            pieces.append(piece)
            remaining -= len(piece)
        checks.equal(os.read(descriptor, 1), b"", f"stable EOF {path.name}")
        after = os.fstat(descriptor)
        checks.equal((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                     (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                     f"held identity {path.name}")
        raw = b"".join(pieces)
        checks.equal(len(raw), before.st_size, f"held length {path.name}")
        return raw
    finally:
        os.close(descriptor)


def exact_directory(root, expected, checks, label):
    before = root.lstat()
    checks.require(stat.S_ISDIR(before.st_mode) and not root.is_symlink() and not is_reparse(before),
                   f"{label} root is not a real directory")
    names = {entry.name for entry in os.scandir(root)}
    checks.equal(names, set(expected), f"{label} exact object closure")
    held = {}
    for name, (size, digest) in sorted(expected.items()):
        raw = held_regular_bytes(root / name, checks, size)
        checks.equal(sha(raw), digest, f"{label} digest {name}")
        held[name] = raw
    after = root.lstat()
    checks.equal((before.st_dev, before.st_ino), (after.st_dev, after.st_ino),
                 f"{label} root identity")
    return held


def parse_manifest_with_sizes(raw, checks, label):
    rows = {}
    for number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        match = ROW_WITH_SIZE.fullmatch(line)
        checks.require(match is not None, f"{label} malformed row {number}")
        digest, size, name = match.groups()
        checks.require(name not in rows, f"{label} duplicate {name}")
        rows[name] = (int(size), digest)
    return rows


def parse_manifest_no_sizes(raw, checks, label):
    rows = {}
    for number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        match = ROW_NO_SIZE.fullmatch(line)
        checks.require(match is not None, f"{label} malformed row {number}")
        digest, name = match.groups()
        checks.require(name not in rows, f"{label} duplicate {name}")
        rows[name] = digest
    return rows


def reject_duplicates(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("duplicate JSON key: " + key)
        value[key] = child
    return value


def finite_json(value, checks, label, depth=0):
    checks.require(depth <= 64, f"excess JSON depth {label}")
    if isinstance(value, float):
        checks.require(math.isfinite(value), f"nonfinite JSON {label}")
    elif isinstance(value, dict):
        for key, child in value.items():
            checks.require(isinstance(key, str), f"non-string JSON key {label}")
            finite_json(child, checks, label, depth + 1)
    elif isinstance(value, list):
        for child in value:
            finite_json(child, checks, label, depth + 1)
    else:
        checks.require(value is None or isinstance(value, (str, int, bool)),
                       f"non-JSON value {label}")


def load_json_raw(raw, checks, label):
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    checks.require(isinstance(value, dict), f"JSON root {label}")
    finite_json(value, checks, label)
    return value


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                     allow_nan=False).encode("ascii")
    return sha(raw)


def open_all_evidence(checks, expected_audit_manifest_sha256):
    v3 = exact_directory(V3, V3_FILES, checks, "v3 package")
    v2 = exact_directory(V2, V2_FILES, checks, "v2 package")
    v2_audit = exact_directory(V2_AUDIT, V2_AUDIT_FILES, checks, "v2 audit")

    checks.equal(sha(v3["ARTIFACT_SHA256SUMS.txt"]), V3_MANIFEST_SHA256,
                 "v3 externally fixed manifest")
    v3_rows = parse_manifest_with_sizes(v3["ARTIFACT_SHA256SUMS.txt"], checks, "v3 manifest")
    checks.equal(v3_rows, {name: row for name, row in V3_FILES.items()
                           if name != "ARTIFACT_SHA256SUMS.txt"}, "v3 manifest rows")

    checks.equal(sha(v2["ARTIFACT_SHA256SUMS.txt"]), V2_MANIFEST_SHA256,
                 "v2 sealed manifest")
    v2_rows = parse_manifest_no_sizes(v2["ARTIFACT_SHA256SUMS.txt"], checks, "v2 manifest")
    checks.equal(v2_rows, {name: digest for name, (_, digest) in V2_FILES.items()
                           if name != "ARTIFACT_SHA256SUMS.txt"}, "v2 manifest rows")

    checks.equal(sha(v2_audit["AUDIT_SHA256SUMS.txt"]), V2_AUDIT_MANIFEST_SHA256,
                 "v2 audit sealed manifest")
    v2_audit_rows = parse_manifest_no_sizes(v2_audit["AUDIT_SHA256SUMS.txt"], checks,
                                             "v2 audit manifest")
    checks.equal(v2_audit_rows,
                 {name: digest for name, (_, digest) in V2_AUDIT_FILES.items()
                  if name != "AUDIT_SHA256SUMS.txt"}, "v2 audit manifest rows")

    manifest_raw = held_regular_bytes(AUDIT_MANIFEST, checks)
    checks.equal(sha(manifest_raw), expected_audit_manifest_sha256,
                 "externally pinned independent-audit manifest")
    audit_rows = parse_manifest_with_sizes(manifest_raw, checks, "independent audit manifest")
    checks.equal(set(audit_rows), AUDIT_FILES, "independent audit manifest closure")
    audit_expected = dict(audit_rows)
    audit_expected["AUDIT_SHA256SUMS.txt"] = (len(manifest_raw), sha(manifest_raw))
    audit = exact_directory(ROOT, audit_expected, checks, "independent audit")
    return v3, v2, v2_audit, audit, manifest_raw


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
        pivot = next(row for row in range(column, n) if work[row][column])
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(n):
            if row != column:
                scale = work[row][column]
                work[row] = [a - scale * b for a, b in zip(work[row], work[column])]
    return [row[n:] for row in work]


def load_oracle_from_held(raw, checks):
    tree = ast.parse(raw.decode("utf-8"), filename="<held-v3-oracle>")
    checks.require(isinstance(tree, ast.Module), "held oracle AST")
    before = set(sys.modules)
    namespace = {"__name__": "fosp_v3_independent_held", "__file__": "<held-v3-oracle>"}
    exec(compile(tree, "<held-v3-oracle>", "exec", dont_inherit=True), namespace, namespace)
    added = {name.split(".")[0] for name in set(sys.modules) - before}
    checks.require(not (added & HEAVY), "held oracle imported no heavy module")
    return namespace, tree


def verify_science(checks, oracle, tree, source):
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
    trace = matmul(matmul(cross, inverse(gram)), transpose(cross))
    checks.equal(capture, sum(trace[i][i] for i in range(3)), "3x3 trace direction")
    checks.equal(residual_energy, energy - capture, "3x3 regression residual identity")
    checks.require(f(0) <= capture <= energy, "3x3 capture PSD bounds")

    # Direct FP16 row-major coefficient serialization and replay on the exact fixture.
    rounded = [[f(struct.unpack("<e", struct.pack("<e", float(value)))[0])
                for value in row] for row in coefficients]
    fp16_bytes = b"".join(struct.pack("<e", float(coefficients[i][j]))
                           for i in range(3) for j in range(3))
    checks.equal(len(fp16_bytes), 18, "one edge has nine binary16 coefficients")
    replayed = matmul(rounded, x)
    replay_residual = [[a - b for a, b in zip(yr, pr)] for yr, pr in zip(y, replayed)]
    replay_energy = sum((v * v for row in replay_residual for v in row), f(0))
    checks.require(replay_energy >= residual_energy, "FP16 replay cannot beat exact least squares")

    fragments = (
        'gram = cp.einsum("nrd,nsd->nrs", expert, expert)',
        "cross[:, :, target_role, predecessor_role]",
        'full = cp.einsum("ijab,jbc,ijac->ij", cross, inverse, cross)',
        'exact_coefficients = cp.einsum("eab,ebc->eac", selected_cross, inverse[predecessors])',
        'predicted = cp.einsum("eab,ebd->ead", replay_coefficients, expert[predecessors])',
        'coefficient_host.astype("<f2", copy=False).tobytes()',
    )
    for fragment in fragments:
        checks.require(fragment in source, "held source 3x3/FP16 fragment " + fragment)

    # Formal containment: a legal path chooses one nonself edge for every
    # non-anchor target.  Each edge is bounded by its target row maximum; the
    # relaxed sum has the additional nonnegative anchor row maximum.
    for n in range(2, 8):
        scores = [[f(((target + 1) * 11 + (pred + 1) * 7) % 23 + 1, 23)
                   for pred in range(n)] for target in range(n)]
        maxima = [max(scores[target][pred] for pred in range(n) if pred != target)
                  for target in range(n)]
        checks.require(all(value >= 0 for value in maxima), f"nonnegative row maxima n={n}")
        relaxed = sum(maxima, f(0))
        for path in itertools.permutations(range(n)):
            legal = sum((scores[target][pred] for pred, target in zip(path[:-1], path[1:])), f(0))
            checks.require(legal <= relaxed, f"gross relaxed contains legal path n={n}")
    checks.equal(768 * 767, 589056, "ordered nonself pair count")

    class TinyMatrix:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows), len(rows))
        def __getitem__(self, key):
            return self.rows[key[0]][key[1]]
        def __neg__(self):
            return TinyMatrix([[-v for v in row] for row in self.rows])

    class TinyNP:
        float64 = float
        @staticmethod
        def asarray(value, dtype=None):
            del dtype
            return value
        @staticmethod
        def arange(n):
            return list(range(n))
        @staticmethod
        def array_equal(left, right):
            return list(left) == list(right)

    predecessor = [2, 0, 1, 4, 3]
    scores = [[0.0] * 5 for _ in range(5)]
    scores[0][2], scores[1][0], scores[2][1] = 1.0, 9.0, 8.0
    scores[3][4], scores[4][3], scores[4][2] = 7.0, 2.0, 6.0
    def assignment(_):
        return list(range(5)), predecessor
    cycles = oracle["_cycles_from_assignment"](predecessor)
    path = oracle["_legal_path_from_cycle_cover"](TinyMatrix(scores), TinyNP, assignment)
    checks.equal(cycles, [[0, 1, 2], [3, 4]], "cycle target/predecessor direction")
    checks.equal(path["path"], [0, 1, 2, 4, 3], "weakest cut and segment order")
    checks.close(path["cycle_cover_capture"], 27.0, "cycle-cover capture")
    checks.close(path["legal_path_capture"], 30.0, "explicit bridge capture")

    # Exact n=8 realizable construction from the sealed v2 BLOCK audit.
    n, r, rho = 8, f(7, 8), f(49, 64)
    checks.equal(r * r, rho, "n8 rho=r^2")
    checks.equal(struct.unpack("<e", struct.pack("<e", float(r)))[0], float(r), "n8 r binary16")
    checks.equal(struct.unpack("<e", struct.pack("<e", float(rho)))[0], float(rho),
                 "n8 rho binary16")
    checks.require(3 * n <= 2047, "three orthogonal n8 roles fit zero-mean coordinate subspace")
    # AR(1) is positive definite for |r|<1.  The star is explicitly realized
    # as leaf_i = r*hub + sqrt(1-r^2)*e_i, hence leaf-leaf inner product rho.
    checks.require(0 < r < 1 and 1 - r * r > 0, "n8 Q/star Gram realizability")
    total = f(3 * n)
    q_relaxed_capture = f(3 * n) * rho
    c_relaxed_capture = f(3 * n) * rho
    q_legal_capture = f(3 * (n - 1)) * rho
    c_legal_capture = f(3) * (2 * rho + (n - 3) * rho * rho)
    checks.equal(q_relaxed_capture, c_relaxed_capture, "n8 relaxed captures cancel")
    q_legal = -0.5 * math.log2(float((total - q_legal_capture) / total))
    c_legal = -0.5 * math.log2(float((total - c_legal_capture) / total))
    corrected = q_legal - c_legal
    checks.close(q_legal, 0.7995602818589078, "n8 direct FP16 Q replay")
    checks.close(c_legal, 0.5885652320580218, "n8 direct FP16 control replay")
    checks.close(corrected, 0.21099504980088601, "n8 corrected legal")
    row = oracle["adversarial_n8_statistics"]()
    checks.close(row["corrected_relaxed_s_bpw"], 0.0, "n8 oracle corrected relaxed")
    checks.close(row["corrected_legal_fp16_s_bpw"], corrected, "n8 oracle corrected legal")
    checks.require(corrected > oracle["REQUIRED_GROSS_S"], "n8 legal FP16 survivor")

    # Rate, physical bytes, and logical/cold-read ledgers.
    weights = 3 * 768 * 2048
    info_bits = (math.factorial(768) - 1).bit_length()
    factoradic_bytes = math.ceil(info_bits / 8)
    coefficient_count = 767 * 9
    coefficient_bits = coefficient_count * 16
    side_bits = 64 * 8 + factoradic_bytes * 8 + coefficient_bits
    checks.equal(weights, 4718592, "weights per expert")
    checks.equal(info_bits, 6260, "factoradic information bits")
    checks.equal(factoradic_bytes, 783, "factoradic physical bytes")
    checks.equal(coefficient_count, 6903, "coefficient count")
    checks.equal(coefficient_bits, 110448, "coefficient bits")
    checks.equal(side_bits, 117224, "total side bits")
    checks.close(side_bits / weights, 0.024843004014756944, "side bpw")
    maximum_cold = 0.0
    rate_rows = []
    for rate in (2.15, 2.30, 2.50):
        frame = math.floor(weights * rate / 8.0)
        actual = frame * 8 / weights
        payload = actual - side_bits / weights
        cold = (math.ceil(frame / 4096) + 1) * 4096
        amp = cold / frame
        maximum_cold = max(maximum_cold, amp)
        observed = oracle["frame_ledger"](rate)
        checks.equal(observed["frame_bytes"], frame, f"frame bytes {rate}")
        checks.close(observed["actual_rate_bpw"], actual, f"actual bpw {rate}")
        checks.close(observed["residual_payload_bpw"], payload, f"payload bpw {rate}")
        checks.equal(observed["cold_page_bytes_including_one_shared_page"], cold,
                     f"cold bytes {rate}")
        checks.close(observed["cold_page_amplification"], amp, f"cold amplification {rate}")
        checks.equal(observed["logical_byte_read_amplification"], 1.0, f"logical read {rate}")
        checks.require(amp < 2.0, f"cold bound {rate}")
        rate_rows.append((rate, frame, actual, payload, cold, amp))
    checks.close(maximum_cold, 1.0054349308378698, "maximum cold amplification")

    # Independent MC, delete-expert jackknife, and quadrature arithmetic.
    controls = [1.0, 2.0, 4.0, 5.0]
    mean = math.fsum(controls) / len(controls)
    mc = math.sqrt(math.fsum((value - mean) ** 2 for value in controls) /
                   (len(controls) * (len(controls) - 1)))
    deletes = [0.1, 0.2, 0.4]
    delete_mean = math.fsum(deletes) / len(deletes)
    jackknife = math.sqrt((len(deletes) - 1) / len(deletes) *
                          math.fsum((value - delete_mean) ** 2 for value in deletes))
    checks.close(mc, math.sqrt(f(10, 12)), "control MC SE")
    checks.close(jackknife, math.sqrt(7.0) / 15.0, "delete-expert jackknife SE")
    checks.close(math.hypot(mc, jackknife), math.sqrt(mc * mc + jackknife * jackknife),
                 "quadrature SE")

    return {
        "n8_q_legal_fp16_s_bpw": q_legal,
        "n8_control_legal_fp16_s_bpw": c_legal,
        "n8_corrected_legal_fp16_s_bpw": corrected,
        "maximum_cold_page_amplification": maximum_cold,
        "rate_rows": rate_rows,
        "replayed_fp16_fixture_residual": float(replay_energy),
    }


def function_node(tree, name):
    return next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name)


def verify_decision_order(checks, oracle, tree, source):
    decision_node = function_node(tree, "_decision_after_legal_statistics")
    decision_source = ast.get_source_segment(source, decision_node)
    checks.require("legal_path_fp16" in decision_source, "decision reads legal FP16")
    checks.require("relaxed_reuse_exact" not in decision_source,
                   "decision does not read corrected relaxed statistic")
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.startswith("HARD_KILL")}
    checks.equal(literals, {"HARD_KILL_GROSS_QWEN_RELAXED_NECESSARY_BOUND"},
                 "only gross Q hard-kill literal exists")
    direct_node = function_node(tree, "_direct_stage")
    direct_source = ast.get_source_segment(source, direct_node)
    checks.require(direct_source.index('"legal_path_fp16": _controlled_statistic') <
                   direct_source.index('"relaxed_reuse_exact": _controlled_statistic'),
                   "legal FP16 corrected statistic constructed before relaxed diagnostic")
    checks.require(direct_source.index("statistics = {") < direct_source.index("decision = "),
                   "all statistics constructed before decision")

    events = []
    saved = {name: oracle[name] for name in
             ("_pair_panel", "_matched_control", "_controlled_statistic")}
    class Pool:
        @staticmethod
        def free_all_blocks():
            events.append("free")
    class CP:
        @staticmethod
        def get_default_memory_pool():
            return Pool()
    def panel(experts, np, cp, assignment):
        del np, cp, assignment
        events.append("panel")
        rows = [{"source_energy": 1.0, "relaxed_reuse_exact_capture": 0.5,
                 "legal_exact": {"legal_path_capture": 0.4},
                 "legal_fp16": {"residual_energy": 0.6}} for _ in experts]
        return {"experts": rows, "total_source_energy": float(len(rows)),
                "relaxed_reuse_exact": {"s_bpw": 1.0},
                "legal_path_exact": {"s_bpw": 0.9},
                "legal_path_fp16": {"s_bpw": 0.8}}
    def matched(expert, seed, cp):
        del seed, cp
        events.append("matched")
        return expert, {}
    def controlled(qwen, controls, metric):
        del qwen, controls
        events.append("stat:" + metric)
        return {"upper_confidence_survives_target": True}
    oracle["_pair_panel"] = panel
    oracle["_matched_control"] = matched
    oracle["_controlled_statistic"] = controlled
    try:
        result = oracle["_direct_stage"]([object(), object()], None, CP, None)
    finally:
        oracle.update(saved)
    stat_events = [event for event in events if event.startswith("stat:")]
    checks.equal(stat_events,
                 ["stat:legal_path_fp16", "stat:legal_path_exact", "stat:relaxed_reuse_exact"],
                 "dynamic corrected-statistic evaluation order")
    checks.equal(result["decision"], "SURVIVE_SOURCE_ORACLE_FP16_PATH_RESIDUAL_CODEC_REQUIRED",
                 "dynamic legal FP16 decision")
    checks.require(result["statistics"]["relaxed_reuse_exact"]["decision_eligible"] is False,
                   "dynamic relaxed statistic diagnostic-only")
    base = {"legal_path_fp16": {"upper_confidence_survives_target": True},
            "relaxed_reuse_exact": {"upper_confidence_survives_target": False}}
    qwen = {"legal_path_fp16": {"s_bpw": 1.0}}
    first = oracle["_decision_after_legal_statistics"](qwen, base)
    base["relaxed_reuse_exact"]["upper_confidence_survives_target"] = True
    second = oracle["_decision_after_legal_statistics"](qwen, base)
    checks.equal(first, second, "corrected relaxed statistic cannot alter decision")
    try:
        oracle["_decision_after_legal_statistics"](qwen, {})
    except oracle["ProtocolError"]:
        checks.require(True, "missing legal FP16 fails closed")
    else:
        checks.require(False, "missing legal FP16 accepted")


def closed_environment():
    return {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ}


def run(command, cwd, timeout=60):
    return subprocess.run(command, cwd=cwd, env=closed_environment(), stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def write_package(destination, held):
    destination.mkdir()
    for name, raw in held.items():
        (destination / name).write_bytes(raw)


def bootstrap_command(package, manifest_digest, *extra):
    return [sys.executable, "-I", "-S", os.fspath(package / "bootstrap_v3.py"),
            "--package-manifest-sha256", manifest_digest, *extra]


def rewrite_manifest_member(package, name):
    target = package / name
    digest, size = sha(target.read_bytes()), target.stat().st_size
    lines = (package / "ARTIFACT_SHA256SUMS.txt").read_text(encoding="ascii").splitlines()
    replaced = []
    for line in lines:
        pieces = line.split("  ")
        replaced.append(f"{digest}  {size}  {name}" if pieces[-1] == name else line)
    (package / "ARTIFACT_SHA256SUMS.txt").write_text("\n".join(replaced) + "\n",
                                                      encoding="ascii", newline="\n")


def runtime_fixture(scratch):
    runtime = scratch / "runtime"
    library = runtime / "lib"
    library.mkdir(parents=True)
    dummy = library / "fosp_runtime_probe_dependency.py"
    dummy.write_bytes(b'VALUE = "SEALED_RUNTIME_SOURCE_PASS"\n')
    raw = dummy.read_bytes()
    manifest = scratch / "runtime_manifest.txt"
    manifest.write_text("FOSP_RUNTIME_CLOSURE_V1\nD  lib\nI  lib\n" +
                        f"F  {sha(raw)}  {len(raw)}  lib/fosp_runtime_probe_dependency.py\n",
                        encoding="ascii", newline="\n")
    return runtime, manifest


def runtime_args(runtime, manifest, entrypoint):
    python_raw = Path(sys.executable).read_bytes()
    return ("--runtime-root", os.fspath(runtime.resolve()),
            "--runtime-manifest", os.fspath(manifest.resolve()),
            "--runtime-manifest-sha256", sha(manifest.read_bytes()),
            "--python-sha256", sha(python_raw), "--entrypoint", entrypoint)


def verify_firewall_hostile(checks, v3_held):
    outcomes = {"package_symlink": "NOT_RUN", "package_fifo": "NOT_RUN",
                "package_socket": "NOT_RUN", "runtime_symlink": "NOT_RUN"}
    with tempfile.TemporaryDirectory(prefix="fosp_v3_independent_") as text:
        scratch = Path(text)
        exact = scratch / "exact"
        write_package(exact, v3_held)

        completed = run(bootstrap_command(exact, V3_MANIFEST_SHA256, "--verify-package"), exact)
        checks.equal(completed.returncode, 0, "genuine bootstrap exact package")
        checks.require(b"FOSP_V3_PACKAGE_SNAPSHOT_PASS files=9" in completed.stdout,
                       "genuine bootstrap snapshot count")

        bad_digest = run(bootstrap_command(exact, "0" * 64, "--verify-package"), exact)
        checks.require(bad_digest.returncode != 0, "manifest substitution rejected")
        checks.require(b"externally pinned package-manifest hash mismatch" in bad_digest.stderr,
                       "manifest substitution diagnostic")

        directory_case = scratch / "package_directory"
        write_package(directory_case, v3_held)
        injected = directory_case / "json"
        injected.mkdir()
        sentinel = scratch / "directory-sentinel"
        (injected / "__init__.py").write_text(
            "open(" + repr(os.fspath(sentinel)) + ", 'wb').write(b'EXECUTED')\n",
            encoding="utf-8")
        completed = run(bootstrap_command(directory_case, V3_MANIFEST_SHA256, "--verify-package"),
                        directory_case)
        checks.require(completed.returncode != 0, "package directory rejected")
        checks.require(b"nonregular package member forbidden: json" in completed.stderr,
                       "package directory rejection reason")
        checks.require(not sentinel.exists(), "package directory code did not execute")

        regular_case = scratch / "package_regular"
        write_package(regular_case, v3_held)
        (regular_case / "extra.py").write_bytes(b"raise RuntimeError('extra')\n")
        completed = run(bootstrap_command(regular_case, V3_MANIFEST_SHA256, "--verify-package"),
                        regular_case)
        checks.require(completed.returncode != 0, "unmanifested regular package member rejected")
        checks.require(b"package object closure mismatch" in completed.stderr,
                       "unmanifested regular rejection reason")

        # Symlinks/reparse points are attempted locally; lack of creation
        # privilege is an explicit platform gap, covered by sealed Linux replay.
        symlink_case = scratch / "package_symlink"
        write_package(symlink_case, v3_held)
        try:
            (symlink_case / "unsealed-link").symlink_to(symlink_case / "README.md")
        except (OSError, NotImplementedError):
            outcomes["package_symlink"] = "SKIP_CREATION_UNAVAILABLE"
            checks.require(True, "package symlink platform gap recorded")
        else:
            completed = run(bootstrap_command(symlink_case, V3_MANIFEST_SHA256, "--verify-package"),
                            symlink_case)
            checks.require(completed.returncode != 0, "package symlink rejected")
            checks.require(b"nonregular package member forbidden" in completed.stderr,
                           "package symlink rejection reason")
            outcomes["package_symlink"] = "PASS_REJECTED"

        if hasattr(os, "mkfifo"):
            fifo_case = scratch / "package_fifo"
            write_package(fifo_case, v3_held)
            os.mkfifo(fifo_case / "unsealed-fifo")
            completed = run(bootstrap_command(fifo_case, V3_MANIFEST_SHA256, "--verify-package"),
                            fifo_case)
            checks.require(completed.returncode != 0, "package FIFO rejected")
            checks.require(b"nonregular package member forbidden" in completed.stderr,
                           "package FIFO rejection reason")
            outcomes["package_fifo"] = "PASS_REJECTED"
        else:
            outcomes["package_fifo"] = "SKIP_NOT_SUPPORTED"
            checks.require(True, "package FIFO platform gap recorded")

        if os.name == "posix":
            import socket
            socket_case = scratch / "package_socket"
            write_package(socket_case, v3_held)
            sock = socket.socket(socket.AF_UNIX)
            try:
                sock.bind(os.fspath(socket_case / "unsealed-socket"))
                completed = run(bootstrap_command(socket_case, V3_MANIFEST_SHA256,
                                                  "--verify-package"), socket_case)
                checks.require(completed.returncode != 0, "package socket rejected")
                checks.require(b"nonregular package member forbidden" in completed.stderr,
                               "package socket rejection reason")
                outcomes["package_socket"] = "PASS_REJECTED"
            finally:
                sock.close()
        else:
            outcomes["package_socket"] = "SKIP_NOT_SUPPORTED"
            checks.require(True, "package socket platform gap recorded")

        runtime_case = scratch / "runtime_package"
        write_package(runtime_case, v3_held)
        runtime, runtime_manifest = runtime_fixture(scratch / "runtime_case")
        (runtime / "lib" / "json").mkdir()
        (runtime / "lib" / "json" / "__init__.py").write_bytes(b"raise RuntimeError('injected')\n")
        completed = run(bootstrap_command(runtime_case, V3_MANIFEST_SHA256,
                                          *runtime_args(runtime, runtime_manifest,
                                                        "sealed_runtime_probe.py")), runtime_case)
        checks.require(completed.returncode != 0, "runtime directory injection rejected")
        checks.require(b"runtime exact object closure mismatch" in completed.stderr,
                       "runtime directory injection reason")

        runtime_link_root = scratch / "runtime_link_package"
        write_package(runtime_link_root, v3_held)
        link_runtime, link_manifest = runtime_fixture(scratch / "runtime_link_case")
        try:
            (link_runtime / "unsealed-link").symlink_to(link_runtime / "lib", target_is_directory=True)
        except (OSError, NotImplementedError):
            outcomes["runtime_symlink"] = "SKIP_CREATION_UNAVAILABLE"
            checks.require(True, "runtime symlink platform gap recorded")
        else:
            completed = run(bootstrap_command(runtime_link_root, V3_MANIFEST_SHA256,
                                              *runtime_args(link_runtime, link_manifest,
                                                            "sealed_runtime_probe.py")),
                            runtime_link_root)
            checks.require(completed.returncode != 0, "runtime symlink rejected")
            checks.require(b"runtime link/special object forbidden" in completed.stderr,
                           "runtime symlink rejection reason")
            outcomes["runtime_symlink"] = "PASS_REJECTED"

        # BLOCK 1: bootstrap self-substitution.  The original externally pinned
        # manifest is supplied, yet replacement script bytes execute first and
        # can simply omit all checks.
        self_case = scratch / "bootstrap_self_binding"
        write_package(self_case, v3_held)
        self_sentinel = scratch / "bootstrap-self-sentinel"
        (self_case / "bootstrap_v3.py").write_text(
            "open(" + repr(os.fspath(self_sentinel)) + ", 'wb').write(b'EXECUTED')\n"
            "print('FOSP_V3_PACKAGE_SNAPSHOT_PASS files=9')\n",
            encoding="utf-8")
        completed = run(bootstrap_command(self_case, V3_MANIFEST_SHA256, "--verify-package"), self_case)
        checks.equal(completed.returncode, 0, "substituted bootstrap can counterfeit success")
        checks.require(self_sentinel.read_bytes() == b"EXECUTED",
                       "substituted bootstrap executes before self-authentication")

        # The producer verifier has the same circular self-binding property.
        verifier_case = scratch / "verifier_self_binding"
        write_package(verifier_case, v3_held)
        verifier_sentinel = scratch / "verifier-self-sentinel"
        (verifier_case / "verify_package.py").write_text(
            "open(" + repr(os.fspath(verifier_sentinel)) + ", 'wb').write(b'EXECUTED')\n"
            "print('{\"status\":\"PASS\",\"checks\":6928}')\n",
            encoding="utf-8")
        completed = run([sys.executable, "-B", "-I", os.fspath(verifier_case / "verify_package.py"),
                         "--manifest-sha256", V3_MANIFEST_SHA256], verifier_case)
        checks.equal(completed.returncode, 0, "substituted verifier can counterfeit success")
        checks.require(verifier_sentinel.read_bytes() == b"EXECUTED",
                       "verifier source runs before own manifest check")

        # BLOCK 2: startup-time filesystem modules remain inherited.  The
        # declared synthetic runtime intentionally contains no encodings file,
        # but a sealed entrypoint imports the already-loaded filesystem module.
        inherited_case = scratch / "inherited_runtime_package"
        write_package(inherited_case, v3_held)
        (inherited_case / "sealed_runtime_probe.py").write_text(
            "import encodings\n"
            "print('INHERITED_ENCODING_ORIGIN=' + str(encodings.__spec__.origin))\n",
            encoding="utf-8")
        rewrite_manifest_member(inherited_case, "sealed_runtime_probe.py")
        inherited_manifest = sha((inherited_case / "ARTIFACT_SHA256SUMS.txt").read_bytes())
        inherited_runtime, inherited_runtime_manifest = runtime_fixture(scratch / "inherited_case")
        completed = run(bootstrap_command(
            inherited_case, inherited_manifest,
            *runtime_args(inherited_runtime, inherited_runtime_manifest, "sealed_runtime_probe.py")),
            inherited_case)
        checks.equal(completed.returncode, 0, "undeclared inherited encodings import succeeds")
        output = completed.stdout.decode("utf-8", errors="replace").strip()
        checks.require(output.startswith("INHERITED_ENCODING_ORIGIN="),
                       "inherited encodings origin reported")
        origin = output.split("=", 1)[1]
        checks.require(origin not in ("built-in", "frozen", "None", ""),
                       "encodings came from filesystem before runtime closure")

        # Direct runner is deployment-blocked, but its guard occurs only after
        # a normal filesystem import.  Plain direct launch can execute a local
        # math.py before reaching the guard.
        direct_case = scratch / "direct_runner"
        write_package(direct_case, v3_held)
        exact_direct = run([sys.executable, "-B", "-I", "-S",
                            os.fspath(direct_case / "free_order_oracle_v3.py")], direct_case)
        checks.require(exact_direct.returncode != 0, "exact direct runner is blocked")
        checks.require(b"FOSP_V3_SOURCE_ONLY_DEPLOYMENT_BLOCKED" in exact_direct.stderr,
                       "exact direct runner block reason")
        direct_sentinel = scratch / "direct-import-sentinel"
        (direct_case / "math.py").write_text(
            "open(" + repr(os.fspath(direct_sentinel)) + ", 'wb').write(b'EXECUTED')\n"
            "raise SystemExit('DIRECT_IMPORT_EXECUTED')\n", encoding="utf-8")
        injected_direct = run([sys.executable, "-B", "-S",
                               os.fspath(direct_case / "free_order_oracle_v3.py")], direct_case)
        checks.require(injected_direct.returncode != 0, "injected direct runner exits")
        checks.require(direct_sentinel.read_bytes() == b"EXECUTED",
                       "direct runner imports package-local code before guard")

    return {"platform": sys.platform, "conditional_objects": outcomes,
            "inherited_encodings_origin": origin,
            "bootstrap_self_binding_exploit": "PASS_REPRODUCED",
            "verifier_self_binding_exploit": "PASS_REPRODUCED",
            "direct_runner_import_exploit": "PASS_REPRODUCED"}


def verify_source_boundaries(checks, v3_held, v2_held, v2_audit_held, oracle):
    lock = load_json_raw(v3_held["protocol_lock.json"], checks, "v3 protocol")
    bindings = load_json_raw(v3_held["source_bindings.json"], checks, "v3 dormant bindings")
    producer_receipt = load_json_raw(v3_held["source_only_receipt.json"], checks, "v3 receipt")
    v2_receipt = load_json_raw(v2_audit_held["audit_receipt.json"], checks, "sealed v2 audit receipt")
    audit_receipt = load_json_raw(held_regular_bytes(ROOT / "audit_receipt.json", checks), checks,
                                 "independent v3 audit receipt")

    checks.equal(v2_receipt["verdict"], "BLOCK", "sealed v2 audit verdict")
    checks.require({"FOSP2-SCI-001", "FOSP2-FW-001", "FOSP2-FW-002"} <=
                   set(v2_receipt["blocking_findings"]), "sealed v2 findings opened")
    checks.require(b"HARD_KILL_CONTROL_CORRECTED_RELAXED_UPPER_BOUND" in
                   v2_held["free_order_oracle_v2.py"], "v2 invalid decision opened as lineage evidence")
    checks.equal(lock["lineage"]["v2_verdict"],
                 "BLOCK_CONTROL_CORRECTED_NONCONTAINMENT_AND_RUNTIME_CLOSURE",
                 "v3 lineage names sealed v2 verdict")
    checks.require(lock["scientific_gate"]["corrected_relaxed_reuse"]["decision_eligible"] is False,
                   "v3 protocol corrected relaxed diagnostic")
    checks.require(lock["scientific_gate"]["controls"]
                   ["corrected_legal_fp16_computed_before_any_control_corrected_decision"] is True,
                   "v3 protocol legal FP16 order")
    checks.require(lock["execution"]["source_access_authorized"] is False,
                   "v3 source access unauthorized")
    checks.require(lock["execution"]["calibration_authorized"] is False,
                   "v3 calibration unauthorized")
    checks.require(lock["execution"]["production_authorization_issued"] is False,
                   "v3 production authorization absent")
    checks.equal(bindings["forbidden_runtime_inputs"], {
        "alternate_source_manifest": False,
        "individual_matrix_path_arguments": False,
        "pinned_panel_path_argument": False,
        "validation_path_argument": False,
    }, "dormant binding CLI closure")
    status = oracle["source_only_status"]()
    checks.equal(status["status"], "SOURCE_ONLY_DEPLOYMENT_BLOCKED", "source default blocked")
    checks.require(status["source_access_authorized"] is False, "source default no access")
    checks.require(status["calibration_authorized"] is False, "source default no calibration")

    unsigned = dict(producer_receipt)
    observed = unsigned.pop("canonical_unsigned_sha256")
    checks.equal(observed, canonical_sha(unsigned), "v3 producer receipt canonical seal")
    checks.require(producer_receipt["deployment"]["authorized"] is False,
                   "producer receipt deployment blocked")
    zero = producer_receipt["zero_access_ledger"]
    for key, value in zero.items():
        if key.endswith(("files_opened", "bytes_read", "paths_followed", "imports", "calls",
                         "fetches", "connections", "issued", "modified")):
            checks.equal(value, 0, "producer zero access " + key)

    unsigned = dict(audit_receipt)
    observed = unsigned.pop("canonical_unsigned_sha256")
    checks.equal(observed, canonical_sha(unsigned), "independent audit receipt canonical seal")
    checks.equal(audit_receipt["verdict"], "BLOCK", "independent receipt verdict")
    checks.equal(set(audit_receipt["blocking_findings"]),
                 {"FOSP3-FW-001", "FOSP3-FW-002", "FOSP3-DOC-001"},
                 "independent receipt findings")
    checks.equal(audit_receipt["independent_verifier"]["expected_checks"], EXPECTED_CHECKS,
                 "receipt expected check count")
    checks.require(not (set(sys.modules) & HEAVY), "independent verifier imported no heavy module")

    return lock, audit_receipt


def verify(expected_audit_manifest_sha256):
    checks = Checks()
    checks.require(re.fullmatch(r"[0-9a-f]{64}", expected_audit_manifest_sha256) is not None,
                   "external audit manifest digest syntax")
    v3, v2, v2_audit, audit, audit_manifest_raw = open_all_evidence(
        checks, expected_audit_manifest_sha256)
    source = v3["free_order_oracle_v3.py"].decode("utf-8")
    oracle, tree = load_oracle_from_held(v3["free_order_oracle_v3.py"], checks)
    science = verify_science(checks, oracle, tree, source)
    verify_decision_order(checks, oracle, tree, source)
    firewall = verify_firewall_hostile(checks, v3)
    lock, receipt = verify_source_boundaries(checks, v3, v2, v2_audit, oracle)

    # The generic README spelling is not executable identity: on Linux `python`
    # or `/usr/bin/python3` may be a symlink, while bootstrap rejects a symlinked
    # sys.executable.  Release instructions must name the externally pinned,
    # canonical regular interpreter and separately trusted bootstrap launcher.
    readme = v3["README.md"].decode("utf-8")
    checks.require("python -I -S bootstrap_v3.py" in readme, "generic launcher spelling observed")
    checks.require(lock["entrypoint_firewall"]["runtime"]
                   ["python_executable_digest_externally_pinned"] is True,
                   "protocol requires pinned interpreter")

    checks.equal(receipt["scientific_replay"]["n8_corrected_legal_fp16_s_bpw"],
                 science["n8_corrected_legal_fp16_s_bpw"], "receipt n8 corrected legal")
    checks.close(receipt["scientific_replay"]["maximum_cold_page_amplification"],
                 science["maximum_cold_page_amplification"], "receipt cold read maximum")
    checks.equal(receipt["hostile_replay"]["bootstrap_self_binding_exploit"],
                 firewall["bootstrap_self_binding_exploit"], "receipt bootstrap exploit")
    checks.equal(receipt["hostile_replay"]["inherited_encodings_origin"],
                 firewall["inherited_encodings_origin"], "receipt inherited origin")
    checks.require(receipt["zero_access_ledger"]["binding_paths_followed"] == 0,
                   "audit followed no binding")
    checks.require(receipt["zero_access_ledger"]["model_or_payload_files_opened"] == 0,
                   "audit opened no model or payload")

    return {
        "status": "BLOCK_CONFIRMED",
        "verdict": "BLOCK",
        "checks": checks.count,
        "expected_checks": EXPECTED_CHECKS,
        "audit_manifest_sha256": sha(audit_manifest_raw),
        "producer_manifest_sha256": V3_MANIFEST_SHA256,
        "v2_manifest_sha256": V2_MANIFEST_SHA256,
        "v2_audit_manifest_sha256": V2_AUDIT_MANIFEST_SHA256,
        "blocking_findings": ["FOSP3-FW-001", "FOSP3-FW-002", "FOSP3-DOC-001"],
        "science": science,
        "hostile_replay": firewall,
        "external_evidence": {
            "v3_files_opened_and_hashed": len(v3),
            "v2_files_opened_and_hashed": len(v2),
            "v2_audit_files_opened_and_hashed": len(v2_audit),
            "audit_files_opened_and_hashed": len(audit),
        },
        "zero_payload_access": True,
    }


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--audit-manifest-sha256":
        raise SystemExit("usage: verify_audit.py --audit-manifest-sha256 <externally-pinned-sha256>")
    result = verify(sys.argv[2])
    if EXPECTED_CHECKS and result["checks"] != EXPECTED_CHECKS:
        raise AssertionError(f"check-count drift: {result['checks']} != {EXPECTED_CHECKS}")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("AUDIT_FAIL: " + str(exc), file=sys.stderr)
        raise
