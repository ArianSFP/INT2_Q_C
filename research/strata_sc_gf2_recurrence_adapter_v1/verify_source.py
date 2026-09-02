#!/usr/bin/env python3
"""Independent source-only manifest, invariant, and hostile-test verifier."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SOURCE_MANIFEST.json"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json(payload: bytes, label: str):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, f"{label} duplicate key")
            output[key] = value
        return output

    return json.loads(
        payload.decode("ascii"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            VerificationError(f"{label} nonfinite {token}")
        ),
    )


def main() -> None:
    manifest_payload = MANIFEST.read_bytes()
    manifest = strict_json(manifest_payload, "manifest")
    require(set(manifest) == {
        "schema", "status", "members", "aggregate_sha256",
        "payload_accessed", "publication_launch_disposition",
        "verification_history",
    }, "manifest schema")
    require(manifest["schema"] == "strata-sc-gf2-source-manifest-v1", "manifest identity")
    require(manifest["status"] == "FROZEN_SOURCE_ONLY_PENDING_FINAL_10_TEST_RERUN", "manifest status")
    require(manifest["payload_accessed"] is False, "payload access declaration")
    require(manifest["publication_launch_disposition"] == "HOLD_INDEPENDENT_Q016_REPLAY_IMPLEMENTATION_MISSING", "launch disposition")
    members = manifest["members"]
    require(isinstance(members, list) and members, "manifest members")
    names = [row["path"] for row in members]
    require(names == sorted(names) and len(names) == len(set(names)), "manifest member order")
    require(set(path.name for path in ROOT.iterdir()) == set(names) | {"SOURCE_MANIFEST.json"}, "package exact member set")
    aggregate = hashlib.sha256()
    for row in members:
        require(set(row) == {"path", "bytes", "sha256"}, "member schema")
        path = ROOT / row["path"]
        require(path.parent == ROOT and path.is_file() and not path.is_symlink(), "regular source member")
        payload = path.read_bytes()
        require(len(payload) == row["bytes"] and sha256(payload) == row["sha256"], f"member binding {row['path']}")
        aggregate.update(row["path"].encode("ascii") + b"\0")
        aggregate.update(len(payload).to_bytes(8, "little"))
        aggregate.update(bytes.fromhex(row["sha256"]))
    require(aggregate.hexdigest() == manifest["aggregate_sha256"], "manifest aggregate")

    design = strict_json((ROOT / "design_lock.json").read_bytes(), "design")
    require(design["status"] == "SOURCE_ONLY_9OF9_VERIFIED_ADDITIVE_BOUND_PENDING_FINAL_10_TEST_RERUN", "design hold")
    require(design["verification_state"]["current_additive_prepayload_bound_test_executed"] is False, "pending test is not preclaimed")
    require(design["launchable_when_v9_audit_alone_passes"] is False, "v9-alone launch forbidden")
    require(design["packet"]["chunk_decisions"] == 4096, "chunk freeze")
    require(design["bounded_negative_scope"]["recurrences_longer_than_4096_can_be_missed"] is True, "bounded negative")
    require(design["universality"]["packet_is_universal_swiglu_moe_format"] is False, "no universal claim")
    require(design["universality"]["owner_descriptor_bits"] == 128, "pilot descriptor")

    codec_text = (ROOT / "strata_recurrence_codec.py").read_text("utf-8")
    scorer_text = (ROOT / "independent_scorer.py").read_text("utf-8")
    gate_text = (ROOT / "publication_gate.py").read_text("utf-8")
    replay_text = (ROOT / "replay_gate.py").read_text("utf-8")
    prepayload_text = (ROOT / "prepayload_rate_gate.py").read_text("utf-8")
    combined = codec_text + scorer_text + gate_text + replay_text + prepayload_text
    for forbidden in (
        "/workspace/INT2__compression", "strata_expert_affine_n20n21.bin",
        "qwen_weight_cache", "torch.load(", "numpy.load(", "np.load(",
    ):
        require(forbidden not in combined, f"payload/path opener forbidden: {forbidden}")
    require("os.open(" not in codec_text + scorer_text + replay_text, "only publication gate may open")
    for token in (
        "def read_publication_member", "AuditAuthority",
        "expected_receipt_sha256", "publication exact set",
    ):
        require(token in gate_text, f"publication gate invariant {token}")
    for token in (
        "ReplayAuthority", "all_base_frequencies_regenerated",
        "all_candidate_payloads_q016_reencoded", "full_reconstruction_recomputed",
    ):
        require(token in replay_text, f"replay gate invariant {token}")
    for token in (
        "caller_supplied_mse_rate_ledger_or_controls\": False",
        "q016_recurrence_packet_replay_independently_authorized",
        "packet_rate_bounds(decoded, catalog_bytes=len(catalog))",
    ):
        require(token in scorer_text, f"scorer invariant {token}")
    for token in (
        "CHUNK_DECISIONS = 4096", "OWNER_DESCRIPTOR_BITS = 128",
        "six level-major ordering", "canonical expert chunk geometry/order",
        "recurrences_longer_than_4096_can_be_missed",
        "selected_decisions_are_arithmetic_coded_in_current_strata",
    ):
        require(token in codec_text, f"codec invariant {token}")
    for token in (
        "PINNED_UNIQUE_SELECTED_DECISIONS = 126_627_266",
        "raw_fallback_hard_kill_before_payload",
        "positive_recurrence_claim_permitted\": False",
    ):
        require(token in prepayload_text, f"prepayload invariant {token}")

    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / "test_source_only.py")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    require(completed.returncode == 0, "source-only tests failed\n" + completed.stdout + completed.stderr)
    require("Ran 10 tests" in completed.stderr and "OK" in completed.stderr, "test transcript closure")
    result = {
        "schema": "strata-sc-gf2-source-verification-v1",
        "status": "PASS_FINAL_10_TEST_RERUN_FROM_PENDING_SOURCE_FREEZE",
        "manifest_sha256": sha256(manifest_payload),
        "aggregate_sha256": manifest["aggregate_sha256"],
        "tests": 10,
        "qwen_or_publication_payload_accessed": False,
        "publication_launchable_after_v9_audit_alone": False,
        "blocking_requirement": "separate independently frozen authenticated publication extraction and Q0.16 replay implementation",
        "owner_descriptor_scope": "Qwen-shaped fixed-128-bit pilot; not universal",
        "negative_scope": "independent six-level BM chunks of at most 4096 decisions",
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
