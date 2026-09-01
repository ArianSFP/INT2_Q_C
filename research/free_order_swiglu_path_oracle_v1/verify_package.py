#!/usr/bin/env python3
"""Pure-stdlib integrity and contract verifier for FOSP-ARX v1."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "ARTIFACT_SHA256SUMS.txt"
EXPECTED_FILES = {
    "README.md",
    "free_order_oracle.py",
    "protocol_lock.json",
    "source_bindings.json",
    "test_source_only.py",
    "verify_package.py",
}
SHA_RE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$")


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)
        self.count += 1

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        self.require(actual == expected, f"{message}: {actual!r} != {expected!r}")

    def close(self, actual: float, expected: float, message: str, tolerance: float = 1e-15) -> None:
        self.require(abs(actual - expected) <= tolerance, f"{message}: {actual!r} != {expected!r}")


def regular_bytes(path: Path, checks: Checks) -> bytes:
    checks.require(path.exists(), f"missing {path.name}")
    checks.require(path.is_file(), f"not a regular file {path.name}")
    checks.require(not path.is_symlink(), f"symlink forbidden {path.name}")
    return path.read_bytes()


def sha256(path: Path, checks: Checks) -> str:
    return hashlib.sha256(regular_bytes(path, checks)).hexdigest()


def load_json(path: Path, checks: Checks) -> dict[str, Any]:
    value = json.loads(regular_bytes(path, checks).decode("utf-8"))
    checks.require(isinstance(value, dict), f"{path.name} root must be object")
    return value


def parse_manifest(checks: Checks) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_number, line in enumerate(regular_bytes(MANIFEST, checks).decode("ascii").splitlines(), 1):
        match = SHA_RE.fullmatch(line)
        checks.require(match is not None, f"malformed manifest line {line_number}")
        assert match is not None
        digest, name = match.groups()
        checks.require(name not in rows, f"duplicate manifest name {name}")
        rows[name] = digest
    checks.equal(set(rows), EXPECTED_FILES, "manifest file set")
    checks.equal(
        {path.name for path in ROOT.iterdir() if path.is_file() and path.name != MANIFEST.name},
        EXPECTED_FILES,
        "package file set",
    )
    for name, expected in sorted(rows.items()):
        checks.equal(sha256(ROOT / name, checks), expected, f"SHA-256 {name}")
    return rows


def verify() -> dict[str, Any]:
    checks = Checks()
    manifest = parse_manifest(checks)
    lock = load_json(ROOT / "protocol_lock.json", checks)
    bindings = load_json(ROOT / "source_bindings.json", checks)
    oracle_raw = regular_bytes(ROOT / "free_order_oracle.py", checks)
    oracle_tree = ast.parse(oracle_raw.decode("utf-8"), filename="free_order_oracle.py")

    checks.equal(lock["schema"], "free_order_swiglu_path_protocol_v1", "protocol schema")
    checks.equal(lock["status"], "SOURCE_ONLY_NOT_AUTHORIZED_FOR_QWEN_EXECUTION", "status")
    checks.close(float(lock["objective"]["required_s_bpw"]), -0.5 * math.log2(0.8), "target s")
    checks.equal(lock["objective"]["physical_rate_interval_bpw"], [2.15, 2.5], "rate interval")
    checks.equal(lock["geometry"]["weights_per_expert"], 3 * 768 * 2048, "expert geometry")

    metric = lock["metric_compatibility"]
    checks.equal(metric["zero_bit_deployed_gauge"], "INELIGIBLE_IMMEDIATE_KILL", "zero-bit verdict")
    checks.equal(metric["information_lower_bound_bits"], (math.factorial(768) - 1).bit_length(), "factorial bits")
    checks.equal(metric["information_lower_bound_bits"], 6260, "frozen factorial bits")
    checks.equal(metric["physical_factoradic_bytes"], math.ceil(6260 / 8), "factoradic bytes")
    checks.close(
        float(metric["physical_factoradic_bpw"]),
        783 * 8 / (3 * 768 * 2048),
        "factoradic bpw",
    )
    checks.require(lock["promotion"]["zero_bit_variant_can_never_be_promoted"] is True, "zero-bit promotion ban")
    checks.require(lock["eligible_codec"]["permutation_is_inline_side_information"] is True, "inline side admission")
    checks.require(lock["eligible_codec"]["additional_expert_payload_read"] is False, "single expert read")

    checks.equal(bindings["schema"], "free_order_swiglu_path_auxiliary_bindings_v1", "bindings schema")
    checks.equal(len(bindings["experts"]), 2, "bound expert count")
    identities = []
    for expert in bindings["experts"]:
        identities.append((int(expert["layer"]), int(expert["expert"])))
        checks.equal([row["role"] for row in expert["roles"]], ["gate", "up", "down"], "joint role order")
        for row in expert["roles"]:
            checks.require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "source digest syntax")
            checks.require(not Path(row["relative_path"]).is_absolute(), "relative source path")
            checks.require(".." not in Path(row["relative_path"]).parts, "source traversal forbidden")
    checks.equal(identities, [(3, 57), (3, 121)], "fixed auxiliary identities")
    bindings_sha = hashlib.sha256(regular_bytes(ROOT / "source_bindings.json", checks)).hexdigest()
    checks.equal(lock["execution_firewall"]["source_bindings_sha256"], bindings_sha, "lock-to-bindings hash")

    top_imports: set[str] = set()
    for node in oracle_tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".")[0])
    checks.require(not ({"cupy", "numpy", "scipy", "torch"} & top_imports), "heavy imports must be deferred")
    literals = {
        node.value
        for node in ast.walk(oracle_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    checks.require("--workspace-root" in literals, "workspace root CLI")
    checks.require("--output" in literals, "output CLI")
    checks.require("--authorization-sentinel" in literals, "authorization sentinel CLI")
    for forbidden in ("--plan", "--manifest", "--source", "--target"):
        checks.require(forbidden not in literals, f"forbidden CLI {forbidden}")

    n = 3 * 768 * 2048
    for row in lock["rate_and_read"]["rows"]:
        rate = float(row["requested_rate_bpw"])
        frame = math.floor(n * rate / 8)
        checks.equal(row["frame_bytes"], frame, f"frame bytes {rate}")
        checks.close(float(row["actual_rate_bpw"]), frame * 8 / n, f"actual rate {rate}")
        amplification = (math.ceil(frame / 4096) + 1) * 4096 / frame
        checks.close(float(row["cold_page_amplification"]), amplification, f"read amp {rate}")
        checks.require(amplification < 2.0, f"read cap {rate}")

    modes = {row["name"]: row for row in lock["eligible_codec"]["coefficient_modes"]}
    expected_bits = {
        "diag3_fp16_oracle_bridge": 767 * 3 * 16,
        "full3x3_fp16_oracle_bridge": 767 * 9 * 16,
        "diag3_fixed_nibble": math.ceil(767 * 3 * 4 / 8) * 8,
        "full3x3_fixed_nibble": math.ceil(767 * 9 * 4 / 8) * 8,
    }
    for name, bits in expected_bits.items():
        checks.equal(modes[name]["coefficient_bits"], bits, f"coefficient bits {name}")
        checks.close(modes[name]["coefficient_bpw"], bits / n, f"coefficient bpw {name}")

    controls = lock["oracle"]["controls"]
    checks.equal(controls["replicates"], 8, "control count")
    checks.equal(len(controls["seeds"]), 8, "control seed count")
    checks.equal(len(set(controls["seeds"])), 8, "distinct control seeds")
    checks.require(controls["identical_optimization"] is True, "identical control optimization")
    checks.require(lock["oracle"]["stage0_dense_envelope"]["all_three_roles_scored_together"] is True, "joint-role gate")
    checks.require(lock["oracle"]["stage1_pair_envelope_only_if_stage0_survives"]["matched_control_required"] is True, "pair controls")
    checks.require(lock["execution_firewall"]["package_authorizes_Qwen_run"] is False, "no execution authority")
    checks.require(lock["execution_firewall"]["package_requires_independent_source_audit_before_any_Qwen_run"] is True, "source audit gate")

    return {
        "status": "PASS",
        "checks": checks.count,
        "manifest_sha256": hashlib.sha256(regular_bytes(MANIFEST, checks)).hexdigest(),
        "artifacts": manifest,
        "zero_bit_verdict": metric["zero_bit_deployed_gauge"],
        "eligible_factoradic_bpw": metric["physical_factoradic_bpw"],
    }


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2, sort_keys=True))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
