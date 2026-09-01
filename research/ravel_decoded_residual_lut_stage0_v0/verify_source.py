#!/usr/bin/env python3
"""Standard-library verifier for the sealed RAVEL stage-0 source package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FILES = {
    "MANIFEST.sha256",
    "README.md",
    "SOURCE_RECEIPT.json",
    "design_lock.json",
    "ravel_stage0.py",
    "test_source_only.py",
    "verify_source.py",
}
MEMBERS = FILES - {"MANIFEST.sha256"}


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

    def close(self, observed: float, expected: float, message: str, tolerance: float = 1e-14) -> None:
        self.require(math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance), f"{message}: {observed} != {expected}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise Failure("manifest lacks final LF")
    result: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        pieces = line.split("  ")
        if len(pieces) != 2 or len(pieces[0]) != 64 or any(c not in "0123456789abcdef" for c in pieces[0]):
            raise Failure("malformed manifest row")
        name = pieces[1]
        if Path(name).name != name or name in result:
            raise Failure("unsafe or duplicate manifest name")
        result[name] = pieces[0]
    return result


def verify(package: Path, plan_dir: Path | None = None) -> int:
    checks = Checks()
    package = package.resolve(strict=True)
    checks.require(package.is_dir() and not package.is_symlink(), "package must be a real directory")
    observed = {row.name for row in package.iterdir()}
    checks.equal(observed, FILES, "exact source closure")
    for name in sorted(FILES):
        row = package / name
        checks.require(row.is_file() and not row.is_symlink(), f"regular non-link member {name}")
    manifest = parse_manifest(package / "MANIFEST.sha256")
    checks.equal(set(manifest), MEMBERS, "manifest member set")
    for name in sorted(MEMBERS):
        checks.equal(sha256_file(package / name), manifest[name], f"manifest hash {name}")

    design = json.loads((package / "design_lock.json").read_text(encoding="utf-8"))
    checks.equal(design["schema"], "ravel-decoded-residual-lut-stage0-design-lock-v0", "design schema")
    arch = design["architecture"]
    checks.equal(arch["table_entries"], 3 * 4 * 32 * 4 * 4, "table geometry")
    checks.equal(arch["packet_bytes"], 16_384, "packet bytes")
    panel = design["panel"]
    checks.equal(panel["weights"], 6 * 4_718_592, "panel values")
    side = 8.0 * arch["packet_bytes"] / panel["weights"]
    checks.close(side, design["rate_and_read"]["side_bpw"], "side rate")
    checks.close(2.5 - side, design["rate_and_read"]["base_payload_cap_bpw"], "base payload cap")
    cold = design["rate_and_read"]["published_worst_cold_page_read_amplification"] + 16_384 / (4_718_592 * 2.5 / 8.0)
    checks.close(cold, design["rate_and_read"]["conservative_cold_page_read_amplification"], "cold read")
    checks.require(cold < 2.0, "read gate")
    checks.equal(design["split"]["fit_expert_ordinals"], [0, 2, 3, 5], "fit split")
    checks.equal(design["split"]["holdout_expert_ordinals"], [1, 4], "holdout split")
    checks.equal(design["oracle"]["source_leaking"], True, "oracle disclosure")
    checks.equal(design["oracle"]["emitted"], False, "oracle not emitted")

    runner_path = package / "ravel_stage0.py"
    runner = runner_path.read_text(encoding="utf-8")
    ast.parse(runner, filename=str(runner_path))
    compile(runner, str(runner_path), "exec", dont_inherit=True)
    literals = (
        'AUTHORIZATION = "OPEN_AUTHENTICATED_DECODED_PANEL_FOR_RAVEL_STAGE0_V0"',
        'FIT_EXPERTS = (0, 2, 3, 5)',
        'HOLDOUT_EXPERTS = (1, 4)',
        'TABLE_ENTRIES = 3 * 4 * 32 * 4 * 4',
        'PACKET_BYTES = 16_384',
        'source_leaking_oracle',
        'output.mkdir(parents=True, exist_ok=False)',
        'import cupy as cp',
    )
    for literal in literals:
        checks.require(literal in runner, f"runner literal {literal}")
    authorization = 'if args.authorization != AUTHORIZATION:'
    import_cupy = '    import cupy as cp'
    payload_call = 'pairs, bindings = matrix_pairs(np, plan_dir)'
    checks.equal(runner.count(payload_call), 1, "unique payload call")
    checks.require(runner.index(authorization) < runner.index(import_cupy), "authorization before CuPy")
    checks.require(runner.index(authorization) < runner.index(payload_call), "authorization before payload")
    for token in ("requests", "urllib", "socket", "subprocess", "torch", "tensorflow"):
        checks.require(f"import {token}" not in runner and f"from {token}" not in runner, f"forbidden import {token}")

    receipt = json.loads((package / "SOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    checks.equal(receipt["schema"], "ravel-decoded-residual-lut-stage0-source-receipt-v0", "receipt schema")
    checks.equal(receipt["status"], "READY_SOURCE_ONLY_NOT_EXECUTED", "receipt status")
    checks.equal(receipt["package_file_count"], len(FILES), "receipt file count")
    checks.equal(receipt["manifest_member_count"], len(MEMBERS), "receipt member count")
    checks.equal(receipt["gpu_execution"], "not run", "GPU boundary")

    if plan_dir is not None:
        plan_dir = plan_dir.resolve(strict=True)
        bindings = {
            "plan_file_sha256": sha256_file(plan_dir / "plan.lock.json"),
            "header_sha256": sha256_file(plan_dir / "header.bin"),
            "decoded_reconstruction_sha256": sha256_file(plan_dir / "independent_audit/post_klt_canonical_groups.f64.bin"),
        }
        for key, value in bindings.items():
            checks.equal(value, panel[key], f"external binding {key}")
        external_plan = json.loads((plan_dir / "plan.lock.json").read_text(encoding="utf-8"))
        checks.equal(external_plan["lock_sha256"], panel["plan_internal_sha256"], "external plan internal hash")
        checks.equal(external_plan["schema"], "strata_expert_affine_n20n21_plan_v1", "external plan schema")
    return checks.count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--plan-dir", type=Path)
    args = parser.parse_args()
    try:
        count = verify(args.package, args.plan_dir)
    except (Failure, OSError, ValueError, KeyError, json.JSONDecodeError, SyntaxError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: RAVEL source closure, arithmetic, interlock, and bindings ({count} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

