#!/usr/bin/env python3
"""Hostile, pure-stdlib verifier for the independent FOSP-ARX-v2 BLOCK audit.

This verifier deliberately opens only the frozen source package, the immutable
v1 audit, and its own four sealed audit artifacts.  It never follows a source
binding or imports NumPy, CuPy, SciPy, Torch, CUDA, or a model library.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PRODUCER = ROOT.parent / "free_order_swiglu_path_oracle_v2"
V1_AUDIT = ROOT.parent / "free_order_swiglu_path_oracle_v1_independent_source_audit_20260901"
AUDIT_MANIFEST = ROOT / "AUDIT_SHA256SUMS.txt"
AUDIT_FILES = {"README.md", "audit_receipt.json", "replay_receipt.json", "verify_audit.py"}
PRODUCER_FILES = {
    "ARTIFACT_SHA256SUMS.txt": (776, "a7dc083b8fec762759100c23bb5e8f140642e1032fcbd4b4845be785e7e841f9"),
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
V1_AUDIT_FILES = {
    "AUDIT_SHA256SUMS.txt": (329, "712c1b055a02b5c2957e1fbf5c1bfec9d134a4d7c8ca5f4edfe01614261585b8"),
    "README.md": (3422, "ac86a49ee21ae849664cc9be704b8bc6e9d81c2ab4ad51ac2bcffa0ae622ced2"),
    "audit_receipt.json": (7085, "dbe7e018cfb899b56fd46084b017fa345514578f6ff58eee157b57cb4b389de4"),
    "replay_receipt.json": (1748, "fbbfffed6f5d605bb1e3d4affa6d885ec9b651b19b3d82d71407ee61da6bc737"),
    "verify_audit.py": (19164, "607e29ab3dc1f8148782d244e66d28077fd9ef90244fc8f5fead4413af9cbf29"),
}
SHA_LINE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")
HEAVY = {"cupy", "numpy", "scipy", "torch", "transformers", "cuda"}


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)
        self.count += 1

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        self.require(actual == expected, f"{message}: {actual!r} != {expected!r}")

    def close(self, actual: float, expected: float, message: str, tolerance: float = 1e-14) -> None:
        self.require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
                     f"{message}: {actual!r} != {expected!r}")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def regular_bytes(path: Path, checks: Checks, expected_size: int | None = None) -> bytes:
    checks.require(path.exists(), f"missing {path}")
    checks.require(path.is_file(), f"not a file {path}")
    checks.require(not path.is_symlink(), f"symlink forbidden {path}")
    raw = path.read_bytes()
    if expected_size is not None:
        checks.equal(len(raw), expected_size, f"byte count {path.name}")
    return raw


def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_tree(value: Any, checks: Checks, label: str, depth: int = 0) -> None:
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
                       f"non-JSON node in {label}")


def load_json(path: Path, checks: Checks) -> dict[str, Any]:
    value = json.loads(regular_bytes(path, checks).decode("utf-8"), object_pairs_hook=reject_duplicate,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    checks.require(isinstance(value, dict), f"JSON root not object: {path.name}")
    finite_tree(value, checks, path.name)
    return value


def verify_seal(value: dict[str, Any], checks: Checks, label: str) -> str:
    unsigned = dict(value)
    observed = unsigned.pop("canonical_unsigned_sha256", None)
    checks.require(isinstance(observed, str) and re.fullmatch(r"[0-9a-f]{64}", observed) is not None,
                   f"missing canonical seal: {label}")
    expected = sha(canonical(unsigned))
    checks.equal(observed, expected, f"canonical seal: {label}")
    return expected


def parse_manifest(raw: bytes, checks: Checks, label: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        match = SHA_LINE.fullmatch(line)
        checks.require(match is not None, f"bad {label} line {line_number}")
        assert match is not None
        digest, name = match.groups()
        checks.require(name not in rows, f"duplicate {label} row {name}")
        rows[name] = digest
    return rows


def exact_closures(checks: Checks) -> dict[str, str]:
    audit_rows = parse_manifest(regular_bytes(AUDIT_MANIFEST, checks), checks, "audit manifest")
    checks.equal(set(audit_rows), AUDIT_FILES, "audit manifest closure")
    checks.equal({p.name for p in ROOT.iterdir()}, AUDIT_FILES | {AUDIT_MANIFEST.name},
                 "audit directory closure")
    for name, expected in sorted(audit_rows.items()):
        checks.equal(sha(regular_bytes(ROOT / name, checks)), expected, f"audit hash {name}")

    checks.equal({p.name for p in PRODUCER.iterdir()}, set(PRODUCER_FILES), "producer exact closure")
    for name, (size, expected) in sorted(PRODUCER_FILES.items()):
        checks.equal(sha(regular_bytes(PRODUCER / name, checks, size)), expected,
                     f"producer hash {name}")
    producer_rows = parse_manifest(regular_bytes(PRODUCER / "ARTIFACT_SHA256SUMS.txt", checks),
                                   checks, "producer manifest")
    expected_rows = {n: row[1] for n, row in PRODUCER_FILES.items()
                     if n != "ARTIFACT_SHA256SUMS.txt"}
    checks.equal(producer_rows, expected_rows, "producer manifest exact contents")

    checks.equal({p.name for p in V1_AUDIT.iterdir()}, set(V1_AUDIT_FILES), "v1 audit exact closure")
    for name, (size, expected) in sorted(V1_AUDIT_FILES.items()):
        checks.equal(sha(regular_bytes(V1_AUDIT / name, checks, size)), expected,
                     f"v1 audit hash {name}")
    v1_rows = parse_manifest(regular_bytes(V1_AUDIT / "AUDIT_SHA256SUMS.txt", checks),
                             checks, "v1 audit manifest")
    checks.equal(v1_rows, {n: row[1] for n, row in V1_AUDIT_FILES.items()
                           if n != "AUDIT_SHA256SUMS.txt"}, "v1 audit manifest contents")
    return audit_rows


def imported_module(checks: Checks) -> Any:
    before = set(sys.modules)
    path = PRODUCER / "free_order_oracle_v2.py"
    spec = importlib.util.spec_from_file_location("fosp_v2_hostile_audit_target", path)
    checks.require(spec is not None and spec.loader is not None, "producer import spec")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    added = {name.split(".")[0] for name in set(sys.modules) - before}
    checks.require(not (added & HEAVY), f"heavy module imported in source audit: {sorted(added & HEAVY)}")
    return module


def dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def transpose(matrix: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    return [list(column) for column in zip(*matrix)]


def matmul(left: Sequence[Sequence[Fraction]], right: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
    columns = transpose(right)
    return [[dot(row, column) for column in columns] for row in left]


def inverse(matrix: Sequence[Sequence[Fraction]]) -> list[list[Fraction]]:
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


def verify_pair_algebra(checks: Checks, source: str) -> None:
    F = Fraction
    x = [[F(1), F(0), F(1), F(2), F(0)],
         [F(0), F(2), F(1), F(1), F(1)],
         [F(1), F(1), F(0), F(1), F(2)]]
    y = [[F(2), F(1), F(0), F(1), F(0)],
         [F(0), F(1), F(3), F(0), F(2)],
         [F(1), F(0), F(1), F(1), F(3)]]
    gram = matmul(x, transpose(x))
    cross = matmul(y, transpose(x))
    coefficients = matmul(cross, inverse(gram))
    predicted = matmul(coefficients, x)
    residual = [[a - b for a, b in zip(yr, pr)] for yr, pr in zip(y, predicted)]
    energy = sum((v * v for row in y for v in row), F(0))
    residual_energy = sum((v * v for row in residual for v in row), F(0))
    capture = sum((cross[i][j] * coefficients[i][j] for i in range(3) for j in range(3)), F(0))
    trace_form = matmul(matmul(cross, inverse(gram)), transpose(cross))
    trace_capture = sum((trace_form[i][i] for i in range(3)), F(0))
    checks.equal(capture, trace_capture, "3x3 all-pairs trace ordering")
    checks.equal(residual_energy, energy - capture, "3x3 regression energy identity")
    checks.require(F(0) <= capture <= energy, "3x3 capture PSD bounds")
    for fragment in (
        'gram = cp.einsum("nrd,nsd->nrs", expert, expert)',
        'cross[:, :, target_role, predecessor_role]',
        'expert[:, target_role, :] @ expert[:, predecessor_role, :].T',
        'full = cp.einsum("ijab,jbc,ijac->ij", cross, inverse, cross)',
        'exact_coefficients = cp.einsum("eab,ebc->eac", selected_cross, inverse[predecessors])',
        'predicted = cp.einsum(',
        'replay_coefficients,',
        'expert[predecessors],',
    ):
        checks.require(fragment in source, f"pair/replay source fragment {fragment}")


def verify_relaxed_containment(checks: Checks) -> None:
    # Formal invariant: in a path, each non-anchor target occurs once and uses
    # one legal non-self predecessor.  Every term is at most that target's row
    # maximum; the omitted anchor maximum is nonnegative for regression scores.
    for n in range(2, 8):
        scores = [[Fraction(((target + 1) * 11 + (pred + 1) * 7) % 23 + 1, 23)
                   for pred in range(n)] for target in range(n)]
        relaxed = sum((max(scores[target][pred] for pred in range(n) if pred != target)
                       for target in range(n)), Fraction(0))
        for path in itertools.permutations(range(n)):
            legal = sum((scores[target][pred] for pred, target in zip(path[:-1], path[1:])),
                        Fraction(0))
            checks.require(legal <= relaxed, f"relaxed containment n={n} path={path}")
    checks.equal(768 * 767, 589056, "all directed nonself pair count")


class TinyMatrix:
    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __getitem__(self, key: tuple[int, int]) -> float:
        row, column = key
        return self.rows[row][column]

    def __neg__(self) -> "TinyMatrix":
        return TinyMatrix([[-value for value in row] for row in self.rows])


class TinyNP:
    float64 = float

    @staticmethod
    def asarray(value: Any, dtype: Any = None) -> TinyMatrix:
        del dtype
        return value if isinstance(value, TinyMatrix) else TinyMatrix(value)

    @staticmethod
    def arange(n: int) -> list[int]:
        return list(range(n))

    @staticmethod
    def array_equal(left: Sequence[int], right: Sequence[int]) -> bool:
        return list(left) == list(right)


def verify_cycle_direction(checks: Checks, module: Any) -> None:
    predecessor = [2, 0, 1, 4, 3]  # row/target -> column/predecessor
    scores = [[0.0] * 5 for _ in range(5)]
    scores[0][2] = 1.0
    scores[1][0] = 9.0
    scores[2][1] = 8.0
    scores[3][4] = 7.0
    scores[4][3] = 2.0
    scores[4][2] = 6.0  # explicit bridge from first segment to second
    matrix = TinyMatrix(scores)

    def assignment(_: TinyMatrix) -> tuple[list[int], list[int]]:
        return list(range(5)), predecessor

    cycles = module._cycles_from_assignment(predecessor)
    checks.equal(cycles, [[0, 1, 2], [3, 4]], "cycle traversal direction")
    result = module._legal_path_from_cycle_cover(matrix, TinyNP, assignment)
    checks.equal(result["path"], [0, 1, 2, 4, 3], "cycle cuts and segment direction")
    checks.close(result["cycle_cover_capture"], 27.0, "cycle-cover capture")
    checks.close(result["legal_path_capture"], 30.0, "path includes explicit bridge direction")
    checks.equal([(row["predecessor"], row["target"]) for row in result["dropped_edges"]],
                 [(2, 0), (3, 4)], "weakest incoming edges dropped")


def fp16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def verify_v1_and_control_counterexamples(checks: Checks, receipt: dict[str, Any]) -> tuple[float, float, float]:
    required_net = -0.5 * math.log2(0.8)
    side = Fraction(117224, 4718592)
    required_gross = required_net + float(side)
    neurons = 768
    checks.require(2048 - 1 >= neurons, "v1 zero-mean orthonormal basis fits")
    v1_capture = 2 * (neurons - 1)
    v1_energy = 3 * neurons
    v1_ratio = Fraction(v1_energy - v1_capture, v1_energy)
    v1_s = -0.5 * math.log2(float(v1_ratio))
    checks.equal(v1_capture, 1534, "v1 counterexample capture")
    checks.equal(v1_ratio, Fraction(770, 2304), "v1 counterexample residual ratio")
    checks.close(v1_s, 0.7906051829300244, "v1 counterexample s")
    checks.require(v1_s > required_gross, "v1 counterexample reaches direct gate and target")
    checks.equal(receipt["v1_counterexample_replay"]["status"],
                 "PASS_COUNTEREXAMPLE_REACHES_DIRECT_STAGE", "v1 replay receipt")

    # Noncontainment of the *control-corrected* relaxation.  Both Q and control
    # have zero role means and identity per-neuron 3x3 role Gram, so the frozen
    # matched-control moments agree.  Three roles occupy orthogonal zero-mean
    # subspaces.  Q correlations are AR(1), r^|i-j|.  The control is a star:
    # hub/leaf correlation r and leaf/leaf correlation rho=r^2.
    n = 8
    r = Fraction(7, 8)
    rho = r * r
    checks.equal(fp16(float(r)), float(r), "r exact binary16")
    checks.equal(fp16(float(rho)), float(rho), "rho exact binary16")
    total_energy = Fraction(3 * n)
    q_relaxed_capture = Fraction(3 * n) * rho
    c_relaxed_capture = Fraction(3 * n) * rho
    q_legal_capture = Fraction(3 * (n - 1)) * rho
    c_legal_capture = Fraction(3) * (2 * rho + (n - 3) * rho * rho)
    q_relaxed_s = -0.5 * math.log2(float((total_energy - q_relaxed_capture) / total_energy))
    c_relaxed_s = -0.5 * math.log2(float((total_energy - c_relaxed_capture) / total_energy))
    q_legal_s = -0.5 * math.log2(float((total_energy - q_legal_capture) / total_energy))
    c_legal_s = -0.5 * math.log2(float((total_energy - c_legal_capture) / total_energy))
    legal_excess = q_legal_s - c_legal_s
    checks.equal(q_relaxed_capture, c_relaxed_capture, "matched relaxed captures cancel")
    checks.close(q_relaxed_s - c_relaxed_s, 0.0, "corrected relaxed statistic")
    checks.close(q_legal_s, 0.7995602818589078, "AR legal FP16 score")
    checks.close(c_legal_s, 0.5885652320580218, "star legal FP16 score")
    checks.close(legal_excess, 0.21099504980088601, "legal corrected score")
    checks.require(q_relaxed_s > required_gross, "gross relaxed gate survives")
    checks.require(0.0 < required_gross < legal_excess, "relaxed hard-kill contradicts legal survivor")
    checks.equal(Fraction(3) + Fraction(3 * (n - 1)) * (1 - rho),
                 total_energy - q_legal_capture, "direct FP16 AR residual replay")
    row = receipt["blocking_findings"]["FOSP2-SCI-001"]
    checks.close(row["qwen_legal_fp16_s_bpw"], q_legal_s, "receipt q legal")
    checks.close(row["control_legal_fp16_s_bpw"], c_legal_s, "receipt control legal")
    checks.close(row["legal_control_corrected_s_bpw"], legal_excess, "receipt legal excess")
    checks.close(row["relaxed_control_corrected_plus_3se_s_bpw"], 0.0,
                 "receipt relaxed kill statistic")
    return v1_s, required_gross, legal_excess


def verify_statistics_and_controls(checks: Checks, source: str) -> None:
    fragments = (
        'source_mean = cp.mean(source, axis=2)',
        'source_gram = cp.einsum("nrd,nsd->nrs", source_centered, source_centered)',
        'transform = cp.einsum("nri,nis->nrs", source_root, raw_invroot)',
        '/ (len(control_s) * (len(control_s) - 1))',
        'control_delete = math.fsum(_metric_for_subset(row, metric, kept) for row in controls) / len(controls)',
        'jackknife_se = _jackknife_se(delete_estimates)',
        'combined_se = math.hypot(control_mc_se, jackknife_se)',
        'optimistic = excess + 3.0 * combined_se',
        'if not relaxed["upper_confidence_survives_target"]:',
        'decision = "HARD_KILL_CONTROL_CORRECTED_RELAXED_UPPER_BOUND"',
    )
    for fragment in fragments:
        checks.require(fragment in source, f"control/statistic source fragment {fragment}")
    controls = [1.0, 2.0, 4.0, 5.0]
    mean = math.fsum(controls) / len(controls)
    mc = math.sqrt(math.fsum((v - mean) ** 2 for v in controls) /
                   (len(controls) * (len(controls) - 1)))
    checks.close(mc, math.sqrt(10.0 / 12.0), "control MC standard error")
    deletes = [0.1, 0.2, 0.4]
    delete_mean = math.fsum(deletes) / len(deletes)
    jackknife = math.sqrt((len(deletes) - 1) / len(deletes) *
                          math.fsum((v - delete_mean) ** 2 for v in deletes))
    checks.close(jackknife, math.sqrt(7.0) / 15.0, "delete-expert jackknife")
    checks.close(math.hypot(mc, jackknife), math.sqrt(mc * mc + jackknife * jackknife),
                 "quadrature combination")


def verify_rate_read(checks: Checks, lock: dict[str, Any], receipt: dict[str, Any]) -> float:
    weights = 3 * 768 * 2048
    checks.equal(weights, 4718592, "weights per expert")
    checks.equal((math.factorial(768) - 1).bit_length(), 6260, "factoradic information bits")
    checks.equal(math.ceil(6260 / 8), 783, "factoradic physical bytes")
    coefficient_count = 767 * 9
    coefficient_bits = coefficient_count * 16
    side_bits = 64 * 8 + 783 * 8 + coefficient_bits
    checks.equal(coefficient_count, 6903, "FP16 coefficient count")
    checks.equal(coefficient_bits, 110448, "FP16 coefficient bits")
    checks.equal(side_bits, 117224, "all side bits")
    checks.close(side_bits / weights, 0.024843004014756944, "side bpw")
    maximum = 0.0
    for row in lock["rate_and_read"]["rows"]:
        rate = float(row["requested_rate_bpw"])
        frame = math.floor(weights * rate / 8)
        actual = 8 * frame / weights
        payload = actual - side_bits / weights
        cold = (math.ceil(frame / 4096) + 1) * 4096
        amplification = cold / frame
        maximum = max(maximum, amplification)
        checks.equal(int(row["frame_bytes"]), frame, f"frame bytes {rate}")
        checks.close(float(row["actual_rate_bpw"]), actual, f"actual rate {rate}")
        checks.close(float(row["residual_payload_bpw"]), payload, f"payload rate {rate}")
        checks.equal(int(row["cold_page_bytes"]), cold, f"cold bytes {rate}")
        checks.close(float(row["cold_page_amplification"]), amplification,
                     f"page read amplification {rate}")
        checks.require(amplification < 2.0, f"strict read bound {rate}")
    checks.close(maximum, 1.0054349308378698, "maximum cold-page read amplification")
    checks.close(receipt["independent_recomputations"]["maximum_cold_page_amplification"],
                 maximum, "receipt max read")
    checks.require(receipt["independent_recomputations"]["logical_read_amplification"] == 1.0,
                   "logical read is one frame")
    return maximum


def copy_producer(destination: Path) -> None:
    destination.mkdir()
    for name in PRODUCER_FILES:
        shutil.copy2(PRODUCER / name, destination / name)


def verify_runtime_closure_exploit(checks: Checks, module: Any, receipt: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="fosp_v2_hostile_stage_") as scratch_text:
        scratch = Path(scratch_text)
        package = scratch / "producer"
        copy_producer(package)
        injected = package / "json"
        injected.mkdir()
        sentinel = scratch / "preflight-code-executed"
        payload = (f"open({os.fspath(sentinel)!r}, 'wb').write(b'EXECUTED_BEFORE_PREFLIGHT')\n"
                   "raise RuntimeError('hostile import proof')\n")
        (injected / "__init__.py").write_text(payload, encoding="utf-8", newline="\n")

        rows, manifest_raw = module._artifact_rows(package)
        checks.equal(set(rows), set(PRODUCER_FILES) - {"ARTIFACT_SHA256SUMS.txt"},
                     "producer checker accepts unsealed directory")
        checks.equal(sha(manifest_raw), PRODUCER_FILES["ARTIFACT_SHA256SUMS.txt"][1],
                     "manifest unchanged in directory exploit")
        command = [sys.executable, "-B", os.fspath(package / "free_order_oracle_v2.py"), "--help"]
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(command, cwd=package, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
                                   check=False)
        checks.require(completed.returncode != 0, "hostile import interrupts runner")
        checks.equal(sentinel.read_bytes(), b"EXECUTED_BEFORE_PREFLIGHT",
                     "unsealed package code ran before package/auth closure")
        checks.equal({p.name for p in package.iterdir() if p.is_file()}, set(PRODUCER_FILES),
                     "all regular producer files remain exact during exploit")
        for name, (size, expected) in PRODUCER_FILES.items():
            raw = (package / name).read_bytes()
            checks.equal((len(raw), sha(raw)), (size, expected), f"exploit preserves {name}")

    finding = receipt["blocking_findings"]["FOSP2-FW-001"]
    checks.equal(finding["exploit_status"], "REPRODUCED_CODE_EXECUTION_BEFORE_PREFLIGHT",
                 "runtime exploit receipt")
    checks.require(finding["producer_regular_file_hashes_unchanged"] is True,
                   "exploit preserves producer hashes receipt")


def verify_evidence_closure_gap(checks: Checks, module: Any, receipt: dict[str, Any]) -> None:
    receipt_digest = "1" * 64
    raw = (f"{receipt_digest}  audit_receipt.json\n" +
           f"{'2' * 64}  verify_audit.py\n").encode("ascii")
    # No verifier file exists or is opened.  Acceptance proves this function
    # checks only a row name, contrary to the README's verifier-binding claim.
    module._audit_manifest_binds(raw, "audit_receipt.json", receipt_digest, "synthetic audit")
    checks.equal(receipt["blocking_findings"]["FOSP2-FW-002"]["synthetic_manifest_status"],
                 "ACCEPTED_WITHOUT_OPENING_OR_HASHING_VERIFIER", "evidence closure gap receipt")


def verify_firewalls(checks: Checks, source: str, trees: dict[str, ast.AST], lock: dict[str, Any]) -> None:
    oracle_tree = trees["free_order_oracle_v2.py"]
    top_imports: set[str] = set()
    for node in oracle_tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".")[0])
    checks.require(not (HEAVY & top_imports), "heavy imports deferred at AST top level")
    literals = {node.value for node in ast.walk(oracle_tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    checks.require({"--workspace-root", "--output", "--authorization", "--authorization-sha256"}
                   <= literals, "closed runner CLI required inputs")
    checks.require(not ({"--plan", "--manifest", "--source", "--target", "--pinned-panel",
                         "--validation"} & literals), "no alternate payload selectors")
    for fragment in (
        "os.O_NOFOLLOW", "dir_fd=cursor", "bytes decoded are the same descriptor bytes",
        "os.O_EXCL", "output parent is no longer a real directory",
        "package, source, output, authorization, and evidence roots must be disjoint",
        "authorization file hash mismatch", "authorization run-id seal mismatch",
        "source hash mismatch", "artifact hash mismatch",
    ):
        haystack = (
            source
            if fragment != "bytes decoded are the same descriptor bytes"
            else json.dumps(lock)
        )
        checks.require(fragment in haystack, f"firewall fragment {fragment}")
    checks.require(lock["execution_firewalls"]["runtime"]["package_authorizes_execution"] is False,
                   "package grants no runtime authority")
    checks.require(lock["promotion"]["pinned_panel_run_authorized"] is False,
                   "pinned panel remains unauthorized")


def verify() -> dict[str, Any]:
    checks = Checks()
    audit_rows = exact_closures(checks)
    receipt = load_json(ROOT / "audit_receipt.json", checks)
    replay = load_json(ROOT / "replay_receipt.json", checks)
    lock = load_json(PRODUCER / "protocol_lock.json", checks)
    bindings = load_json(PRODUCER / "source_bindings.json", checks)
    source_receipt = load_json(PRODUCER / "source_only_receipt.json", checks)
    verify_seal(receipt, checks, "audit receipt")
    verify_seal(replay, checks, "replay receipt")
    checks.equal(verify_seal(source_receipt, checks, "producer source receipt"),
                 "9e656b50e052fe037d376f87faca00bede9b61be08c200494aaefc6192cc4baa",
                 "producer internal receipt hash")

    checks.equal(receipt["schema"], "free-order-swiglu-path-v2-independent-source-audit-receipt-v1",
                 "audit schema")
    checks.equal(receipt["status"],
                 "BLOCK_CONTROL_CORRECTED_NONCONTAINMENT_AND_RUNTIME_CLOSURE",
                 "audit status")
    checks.equal(receipt["artifact_set_status"], "IMMUTABLE_BLOCK_AUDIT_ARTIFACT_SET",
                 "audit artifact status")
    checks.equal(receipt["verdict"], "BLOCK", "audit verdict")
    checks.require(receipt["producer_modified"] is False, "producer not modified")
    for name, (size, digest) in PRODUCER_FILES.items():
        row = receipt["audited_package"]["files"][name]
        checks.equal((row["bytes"], row["sha256"]), (size, digest), f"receipt package row {name}")
    checks.equal(receipt["audited_package"]["artifact_manifest_sha256"],
                 PRODUCER_FILES["ARTIFACT_SHA256SUMS.txt"][1], "receipt package manifest")

    checks.equal(lock["schema"], "free_order_swiglu_path_protocol_v2", "protocol schema")
    checks.equal(lock["status"],
                 "SOURCE_ONLY_DEPLOYMENT_BLOCKED_PENDING_INDEPENDENT_AUDITS_AND_ONE_SHOT_AUTHORIZATION",
                 "source-only protocol status")
    checks.equal(bindings["schema"], "free_order_swiglu_path_auxiliary_bindings_v1", "binding schema")
    checks.equal([(int(row["layer"]), int(row["expert"])) for row in bindings["experts"]],
                 [(3, 57), (3, 121)], "fixed auxiliary experts")
    checks.equal([[role["role"] for role in row["roles"]] for row in bindings["experts"]],
                 [["gate", "up", "down"], ["gate", "up", "down"]], "fixed role order")
    checks.require(bindings["forbidden_runtime_inputs"]["pinned_panel_path_argument"] is False,
                   "no pinned panel input")
    checks.require(bindings["forbidden_runtime_inputs"]["validation_path_argument"] is False,
                   "no validation input")

    sources: dict[str, str] = {}
    trees: dict[str, ast.AST] = {}
    for name in ("free_order_oracle_v2.py", "calibrate_runtime.py", "create_authorization.py",
                 "test_source_only.py", "verify_package.py"):
        sources[name] = regular_bytes(PRODUCER / name, checks).decode("utf-8")
        trees[name] = ast.parse(sources[name], filename=name)
        checks.require(isinstance(trees[name], ast.Module), f"AST parse {name}")

    module = imported_module(checks)
    verify_pair_algebra(checks, sources["free_order_oracle_v2.py"])
    verify_relaxed_containment(checks)
    verify_cycle_direction(checks, module)
    v1_s, required_gross, legal_excess = verify_v1_and_control_counterexamples(checks, receipt)
    verify_statistics_and_controls(checks, sources["free_order_oracle_v2.py"])
    maximum_read = verify_rate_read(checks, lock, receipt)
    verify_runtime_closure_exploit(checks, module, receipt)
    verify_evidence_closure_gap(checks, module, receipt)
    verify_firewalls(checks, sources["free_order_oracle_v2.py"], trees, lock)

    checks.equal(replay["status"], "SOURCE_ONLY_REPLAY_COMPLETE_BLOCK_CONFIRMED", "replay status")
    checks.equal(replay["local_windows"]["tests_run"], 17, "local tests")
    checks.equal(replay["local_windows"]["passes"], 14, "local passes")
    checks.equal(replay["local_windows"]["skips"], 3, "local skips")
    checks.equal(replay["linux_runpod"]["tests_run"], 17, "Linux tests")
    checks.equal(replay["linux_runpod"]["passes"], 17, "Linux passes")
    checks.equal(replay["linux_runpod"]["skips"], 0, "Linux skips")
    checks.equal(replay["local_windows"]["producer_verifier_checks"], 219,
                 "local producer verifier")
    checks.equal(replay["linux_runpod"]["producer_verifier_checks"], 219,
                 "Linux producer verifier")
    checks.require(replay["linux_runpod"]["all_three_windows_skipped_security_tests_passed"] is True,
                   "Linux security skips replayed")
    checks.require(replay["remote_hashes_match_local"] is True, "remote exact hash equality")
    checks.equal(replay["independent_verifier"]["expected_checks"], 7935,
                 "independent verifier check count")

    zero = receipt["zero_access_ledger"]
    for key in ("qwen_or_model_payload_files_opened", "qwen_or_model_payload_bytes_read",
                "binding_relative_paths_followed", "pinned_panel_files_opened",
                "validation_files_opened", "cupy_imports", "cuda_api_calls",
                "gpu_device_calls", "external_data_fetches", "runtime_authorizations_issued",
                "production_outputs_opened", "producer_files_modified"):
        checks.equal(zero[key], 0, f"zero access {key}")
    checks.equal(zero["runpod_source_only_replay_connections"], 1,
                 "one source-only RunPod connection recorded")
    checks.require(not (set(sys.modules) & HEAVY), "audit process imported no heavy modules")

    return {
        "status": "BLOCK_CONFIRMED",
        "verdict": "BLOCK",
        "checks": checks.count,
        "audit_manifest_sha256": sha(AUDIT_MANIFEST.read_bytes()),
        "audit_artifacts": audit_rows,
        "producer_manifest_sha256": PRODUCER_FILES["ARTIFACT_SHA256SUMS.txt"][1],
        "producer_verifier_checks": 219,
        "local_tests": {"run": 17, "passes": 14, "skips": 3},
        "linux_tests": {"run": 17, "passes": 17, "skips": 0},
        "v1_counterexample_s_bpw": v1_s,
        "required_gross_s_bpw": required_gross,
        "control_noncontainment_legal_excess_s_bpw": legal_excess,
        "maximum_cold_read_amplification": maximum_read,
        "blocking_findings": ["FOSP2-SCI-001", "FOSP2-FW-001", "FOSP2-FW-002"],
        "zero_payload_access": True,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2, sort_keys=True, allow_nan=False))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
