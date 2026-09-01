#!/usr/bin/env python3
"""Standard-library verifier for the sealed CCQ raw-MSE source package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PACKAGE_RELPATH = Path("research/ccq_raw_mse_stage0_v0")
LOCK_RELPATH = Path("blind_protocol_v2/unblinded/source_hashes.lock.json")
LOCK_BYTES = 46013
LOCK_SHA256 = "bf39877a4ac161f20b22fae9400f21cb604a0c5b69df666c54f00ec2e7e7cf23"
LOCK_INTERNAL = "5a82dac742110d4f48bbd73ae82081e1622b10b660b7850dadfe613ff475cc5b"
EXPECTED_FILES = {
    "MANIFEST.sha256",
    "PRIMARY_SOURCES.json",
    "README.md",
    "RESEARCH_FINDING.md",
    "SOURCE_RECEIPT.json",
    "ccq_stage0.py",
    "design_lock.json",
    "test_source_only.py",
    "verify_source.py",
}
MANIFEST_MEMBERS = EXPECTED_FILES - {"MANIFEST.sha256"}


class Failure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise Failure(message)

    def equal(self, observed: Any, expected: Any, message: str) -> None:
        self.require(observed == expected, f"{message}: {observed!r} != {expected!r}")

    def close(self, observed: float, expected: float, message: str, tolerance: float = 1.0e-12) -> None:
        self.require(math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance), f"{message}: {observed!r} != {expected!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def parse_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    text = path.read_text(encoding="ascii")
    if not text.endswith("\n"):
        raise Failure("manifest lacks final LF")
    for line in text.splitlines():
        pieces = line.split("  ")
        if len(pieces) != 2 or len(pieces[0]) != 64 or any(char not in "0123456789abcdef" for char in pieces[0]):
            raise Failure("malformed manifest row")
        name = pieces[1]
        if name in rows or "/" in name or "\\" in name or name in (".", ".."):
            raise Failure("unsafe or duplicate manifest member")
        rows[name] = pieces[0]
    return rows


def rate_ledger(expert_count: int) -> tuple[list[dict[str, float | int]], float, float]:
    values_per_expert = 4_718_592
    expert_fixed = 1_252_416
    fixed = 4096 + expert_count * expert_fixed
    values = expert_count * values_per_expert
    prefix = fixed * 8.0 / values
    required_q = 0.8 / 2.0 ** (2.0 * prefix)
    rows: list[dict[str, float | int]] = []
    for rate in (2.15, 2.30, 2.50):
        physical = math.ceil(rate * values / 8.0)
        local_total = physical - 4096
        local_min, remainder = divmod(local_total, expert_count)
        local_max = local_min + int(remainder != 0)
        cold = 4096 + 4096 * math.ceil(local_max / 4096.0)
        rows.append(
            {
                "rate": rate,
                "physical": physical,
                "residual": physical - fixed,
                "local_max": local_max,
                "cold": cold,
                "amp": cold / (physical / expert_count),
            }
        )
    return rows, prefix, required_q


def check_closure(package: Path, checks: Checks) -> None:
    checks.require(package.is_dir() and not package.is_symlink(), "package is not a regular directory")
    observed: set[str] = set()
    with os.scandir(package) as iterator:
        for entry in iterator:
            checks.require(not entry.is_symlink(), f"symlink rejected: {entry.name}")
            checks.require(entry.is_file(follow_symlinks=False), f"non-regular member rejected: {entry.name}")
            observed.add(entry.name)
    checks.equal(observed, EXPECTED_FILES, "exact closure")


def verify(root: Path, package_override: Path | None = None, verify_lock: bool = True) -> int:
    checks = Checks()
    root = root.resolve()
    package = package_override.resolve() if package_override is not None else (root / PACKAGE_RELPATH).resolve()
    check_closure(package, checks)

    manifest = parse_manifest(package / "MANIFEST.sha256")
    checks.equal(set(manifest), MANIFEST_MEMBERS, "manifest member set")
    for name in sorted(MANIFEST_MEMBERS):
        checks.equal(sha256_file(package / name), manifest[name], f"manifest hash {name}")

    primary = json.loads((package / "PRIMARY_SOURCES.json").read_text(encoding="utf-8"))
    checks.equal(primary.get("schema"), "ccq-primary-source-binding-v0", "primary schema")
    checks.equal(primary["paper"]["arxiv_id"], "2507.07145v1", "paper version")
    checks.equal(primary["official_code"]["commit"], "f5562df9fbd543a63dc28bb8e5709cb6d90e1707", "official code commit")
    for key in ("repository", "documentation", "loader", "dequant_kernel"):
        checks.require(primary["official_code"][key].startswith("https://github.com/PaddlePaddle/FastDeploy"), f"official URL {key}")

    design = json.loads((package / "design_lock.json").read_text(encoding="utf-8"))
    checks.equal(design.get("schema"), "ccq-raw-mse-stage0-design-lock-v0", "design schema")
    checks.equal(design["architecture"]["cell"], "CCQ Code Cluster (L=6,N=4,S=3)", "frozen cell")
    checks.equal(design["execution"]["fixed_split"]["fit_slots"], [0, 2, 3, 5], "fit split")
    checks.equal(design["execution"]["fixed_split"]["holdout_slots"], [1, 4], "holdout split")
    checks.equal(design["execution"]["continuous_scale_passes"], 3, "continuous passes")
    checks.equal(design["execution"]["cluster_refinement_passes"], 2, "cluster passes")
    checks.equal(design["execution"]["viterbi_vector_batch"], 65536, "Viterbi batch")
    checks.equal(design["execution"]["table_column_tile"], 64, "column tile")
    checks.equal(design["panel"]["source_lock_relpath"], str(LOCK_RELPATH).replace("\\", "/"), "lock relative path")
    checks.equal(design["panel"]["source_lock_bytes"], LOCK_BYTES, "lock bytes in design")
    checks.equal(design["panel"]["source_lock_file_sha256"], LOCK_SHA256, "lock hash in design")
    checks.equal(design["panel"]["source_lock_internal_sha256"], LOCK_INTERNAL, "lock internal hash in design")

    rate = design["rate"]
    checks.equal(rate["index_bytes_per_expert"], 1_179_648, "index bytes")
    checks.equal(rate["local_uint4_scale_bytes_per_expert"], 36_864, "local scale bytes")
    checks.equal(rate["code_float32_bytes_per_expert"], 28_672, "code float bytes")
    checks.equal(rate["super_scale_fp16_bytes_per_expert"], 7_168, "super scale bytes")
    checks.equal(rate["per_output_parameter_bytes_per_expert"], 35_840, "parameter bytes")
    expert_fixed = 64 + 1_179_648 + 36_864 + 35_840
    checks.equal(expert_fixed, 1_252_416, "expert fixed bytes")
    panel_fixed = 4096 + 6 * expert_fixed
    checks.equal(panel_fixed, 7_518_592, "panel fixed bytes")
    rows6, prefix6, required6 = rate_ledger(6)
    rows128, prefix128, required128 = rate_ledger(128)
    checks.close(prefix6, float(rate["six_expert_fixed_prefix_bpw"]), "six-expert prefix")
    checks.close(required6, float(rate["six_expert_required_q"]), "six-expert required q")
    expected6 = (
        (7_608_730, 90_138, 1_267_439, 1_273_856, 1.0045219110153665),
        (8_139_572, 620_980, 1_355_913, 1_363_968, 1.0054346837892678),
        (8_847_360, 1_328_768, 1_473_878, 1_478_656, 1.0027777777777778),
    )
    for row, expected in zip(rows6, expected6):
        for key, value in zip(("physical", "residual", "local_max", "cold"), expected[:4]):
            checks.equal(row[key], value, f"six-expert {row['rate']} {key}")
        checks.close(float(row["amp"]), expected[4], f"six-expert {row['rate']} amp")
        checks.require(float(row["amp"]) < 2.0, "six-expert read below 2x")
    checks.close(prefix128, 2.1234266493055554, "128-expert prefix")
    checks.close(required128, 0.04213662594769002, "128-expert required q")
    checks.require(all(float(row["amp"]) < 2.0 for row in rows128), "128-expert reads below 2x")

    hybrid = design["hybrid_rate_kill"]
    hybrid_fixed = 4096 + 6 * (1_474_560 + 7_168 + 64)
    hybrid_bpw = hybrid_fixed * 8.0 / (6 * 4_718_592)
    checks.close(hybrid_bpw, float(hybrid["six_expert_exact_prefix_bpw"]), "hybrid exact prefix")
    checks.require(hybrid_bpw > 2.5 and hybrid["status"] == "KILL_RATE_ABOVE_2P5", "hybrid rate kill")

    runner_path = package / "ccq_stage0.py"
    runner = runner_path.read_text(encoding="utf-8")
    ast.parse(runner, filename=str(runner_path))
    compile(runner, str(runner_path), "exec", dont_inherit=True)
    required_literals = (
        'SOURCE_LOCK_RELPATH = Path("blind_protocol_v2/unblinded/source_hashes.lock.json")',
        'AUTHORIZATION = "OPEN_AUTHENTICATED_18_MATRIX_PANEL_FOR_CCQ_RAW_MSE_STAGE0_V0"',
        "CONTINUOUS_PASSES = 3",
        "CLUSTER_PASSES = 2",
        "VITERBI_BATCH = 65536",
        "COLUMN_TILE = 64",
        "source_root = lock_path.parent.resolve()",
        "path.relative_to(source_root)",
        "decoded = __float2int_rd(fmaf((float)q, code_scale, code_zp + 0.5f));",
        "if float(source[\"holdout_F_oracle\"]) > TARGET_F:",
        'import cupy as cp',
    )
    for literal in required_literals:
        checks.require(literal in runner, f"runner literal: {literal}")
    checks.require(runner.index("validate_source_lock(args.root)") < runner.index("import cupy as cp"), "lock validated before CuPy import")
    payload_call = "source, source_receipts = load_sources(cp, lock_path, lock)"
    checks.equal(runner.count(payload_call), 1, "unique payload load call")
    checks.require(
        runner.index("args.authorization != AUTHORIZATION") < runner.index(payload_call),
        "authorization before payload load",
    )
    for token in ("requests", "urllib", "socket", "subprocess", "torch", "tensorflow"):
        checks.require(f"import {token}" not in runner and f"from {token}" not in runner, f"forbidden runner import {token}")

    receipt = json.loads((package / "SOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    checks.equal(receipt.get("schema"), "ccq-raw-mse-stage0-source-receipt-v0", "receipt schema")
    checks.equal(receipt.get("status"), "READY_SOURCE_ONLY_NOT_EXECUTED", "receipt status")
    checks.equal(receipt["package_file_count"], len(EXPECTED_FILES), "receipt closure count")
    checks.equal(receipt["manifest_member_count"], len(MANIFEST_MEMBERS), "receipt manifest count")
    checks.equal(receipt["runner_sha256"], sha256_file(runner_path), "receipt runner hash")
    checks.equal(receipt["design_lock_sha256"], sha256_file(package / "design_lock.json"), "receipt design hash")
    checks.equal(receipt["primary_sources_sha256"], sha256_file(package / "PRIMARY_SOURCES.json"), "receipt primary hash")
    checks.equal(receipt["source_lock_access"], "metadata bytes only; no matrix payload opened", "receipt source boundary")

    if verify_lock:
        lock_path = (root / LOCK_RELPATH).resolve()
        try:
            lock_path.relative_to(root)
        except ValueError as exc:
            raise Failure("lock escaped root") from exc
        checks.require(lock_path.is_file() and not lock_path.is_symlink(), "authenticated lock regular file")
        checks.equal(lock_path.stat().st_size, LOCK_BYTES, "authenticated lock bytes")
        checks.equal(sha256_file(lock_path), LOCK_SHA256, "authenticated lock file hash")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        checks.equal(lock.get("schema"), "int2-qwen-blind-source-finalization-v2", "authenticated lock schema")
        checks.equal(lock.get("lock_sha256"), LOCK_INTERNAL, "authenticated lock internal hash")
        checks.equal(lock.get("matrix_count"), 18, "authenticated matrix count")
        checks.equal(lock.get("source_values"), 28_311_552, "authenticated source values")

    return checks.count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        count = verify(args.root)
    except (Failure, OSError, ValueError, KeyError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: CCQ source-only closure, primary binding, containment, rate/read ledger, split, runner, and lock ({count} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
