#!/usr/bin/env python3
"""Verify the committed UWFA-SC v8 final source-review artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_FILES = {
    "AUDIT_REPORT.md",
    "INDEPENDENT_SOURCE_REVIEW.json",
    "SOURCE_INVENTORY.tsv",
    "independent_source_review.py",
    "run_independent_source_review.py",
    "verify_audit.py",
}
EXPECTED_MANIFEST = "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6"
EXPECTED_COMMIT = "d563c4ac1e78a6b6e7f0722291211d1209f775af"
EXPECTED_PARENT = "2315551e504b0c7c1e357793aa259b745ff4d717"
EXPECTED_REVIEWER = "a9d5a2c8ca046312e2dbce657ceadf11ed965468375ca6988f7f68f5de14e9c2"
EXPECTED_LAUNCHER = "1e0f77166c2724896c00387be13d74eead0fe3033094e9fa23d6bcb227f2b10e"


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"duplicate key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise AuditError(f"nonfinite JSON: {value}")

    parsed = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(parsed, dict), "JSON root")
    return parsed


def load_core(path: Path):
    source = path.read_bytes()
    require(sha256(source) == EXPECTED_REVIEWER, "reviewer digest")
    spec = importlib.util.spec_from_file_location("uwfa_v8_final_independent_core", path)
    require(spec is not None and spec.loader is not None, "reviewer import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    audit = Path(args.audit_dir).absolute()
    repo = Path(args.repo).absolute()

    actual = {entry.name for entry in audit.iterdir() if entry.is_file()}
    require(actual == EXPECTED_FILES, f"audit member set: {sorted(actual ^ EXPECTED_FILES)}")
    require(sha256((audit / "run_independent_source_review.py").read_bytes()) == EXPECTED_LAUNCHER, "launcher digest")
    core = load_core(audit / "independent_source_review.py")
    replay = core.review(repo)

    receipt = strict_json((audit / "INDEPENDENT_SOURCE_REVIEW.json").read_bytes())
    require(receipt.get("schema") == "unifilar-wfa-entropy-census-independent-source-review-v8", "receipt schema")
    require(receipt.get("status") == "PASS_INDEPENDENT_SOURCE_REVIEW", "receipt status")
    require(receipt.get("reviewed_public_commit") == replay["reviewed_public_commit"] == EXPECTED_COMMIT, "commit binding")
    require(receipt.get("reviewed_public_commit_parent") == replay["reviewed_public_commit_parent"] == EXPECTED_PARENT, "parent binding")
    require(receipt.get("reviewed_source_manifest_sha256") == replay["reviewed_source_manifest_sha256"] == EXPECTED_MANIFEST, "manifest binding")
    require(receipt.get("reviewed_member_count_excluding_manifest") == 17, "member count")
    require(receipt.get("reviewed_member_bytes_excluding_manifest") == 633319, "member bytes")
    require(receipt.get("reviewed_package_bytes_including_manifest") == 636837, "package bytes")

    transition = receipt["freeze_transition"]
    require(transition["changed_paths"] == replay["freeze_transition_changed_paths"], "transition paths")
    for key in (
        "only_lifecycle_readme_design_and_manifest",
        "non_lifecycle_members_byte_identical_to_prefreeze_parent",
        "python_members_byte_identical_to_prefreeze_parent",
        "design_lock_only_status_changed",
    ):
        require(transition[key] is True, f"transition gate: {key}")

    inventory_rows = (audit / "SOURCE_INVENTORY.tsv").read_text(encoding="utf-8").splitlines()
    require(inventory_rows[0] == "name\tbytes\tsha256\tbyte_identical_to_prefreeze_parent", "inventory header")
    parsed_inventory: dict[str, tuple[int, str, str]] = {}
    for line in inventory_rows[1:]:
        name, size, digest, state = line.split("\t")
        require(name not in parsed_inventory, f"duplicate inventory: {name}")
        parsed_inventory[name] = (int(size), digest, state)
    require(set(parsed_inventory) == {row["name"] for row in replay["manifest_members"]} | {"SOURCE_MANIFEST.json"}, "inventory members")
    for row in replay["manifest_members"]:
        size, digest, _ = parsed_inventory[row["name"]]
        require((size, digest) == (row["bytes"], row["sha256"]), f"inventory row: {row['name']}")
    require(parsed_inventory["SOURCE_MANIFEST.json"][:2] == (3518, EXPECTED_MANIFEST), "inventory manifest")

    verifier = receipt["sealed_source_verifier"]
    require(verifier["status"] == "PASS_SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "sealed verifier status")
    require(verifier["source_manifest_sha256"] == EXPECTED_MANIFEST, "sealed verifier manifest")
    require(verifier["return_code"] == 0 and verifier["stderr_bytes"] == 0, "sealed verifier execution")
    require(verifier["payload_authority_granted"] is False, "sealed verifier claim boundary")

    tests = receipt["source_only_tests"]
    for key, expected in (("return_code", 0), ("reported_tests", 68), ("explicit_ok_lines", 68), ("failures", 0), ("errors", 0), ("skips", 0)):
        require(tests[key] == expected, f"test receipt: {key}")
    require(tests["stderr_sha256"] == "08a25d0d5bc263e1b3c095b49e511ef29efd5a5df56a7ddf04c39831258497d2", "test transcript digest")

    scope = receipt["scope"]
    for key in (
        "producer_modules_imported_by_independent_reviewer",
        "qwen_opened_statted_hashed_or_enumerated",
        "current_artifact_opened_statted_hashed_or_enumerated",
        "gaussian_control_opened_statted_hashed_or_enumerated",
        "payload_authority_granted",
        "qwen_or_payload_performance_claim",
    ):
        require(scope[key] is False, f"scope boundary: {key}")

    result = {
        "schema": "unifilar-wfa-v8-final-source-audit-artifact-verification-v1",
        "status": "PASS_AUDIT_ARTIFACT_VERIFICATION",
        "reviewed_public_commit": EXPECTED_COMMIT,
        "reviewed_source_manifest_sha256": EXPECTED_MANIFEST,
        "source_tests_passed": 68,
        "source_tests_failed": 0,
        "payload_authority_granted": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_AUDIT_ARTIFACT_VERIFICATION: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
