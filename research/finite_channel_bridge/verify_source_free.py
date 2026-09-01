#!/usr/bin/env python3
"""Independent, payload-free verifier for the finite-bridge research bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import bridge_ledger


EXPECTED_SILWARP_V2 = {
    "PROTOCOL.md": "8438764d4e2d3f272afe40fe38860a42b47406d4dcd4db404107d14352738daf",
    "protocol_lock.json": "c3f896b9beaf12c30f49d05d791288f0b32a639385e225628211df06bd8e1dd9",
    "source_lock.json": "b0dc982e42a22fa960a1436ed5ebdcab1233d0bc2e463972e16dab4315e14042",
    "silwarp_common.py": "09c9091af31871d715e2270254413503157e0035efb217206cd480aacb057ade",
    "silwarp_gate.py": "c7720dd2557b1009b9947eb80f7e57921f8a77e2399f8848f6ce4ba218696be7",
    "test_source_free.py": "33c7cfb1c41b99bd06035687f14f4abadda0d2ce99ad7ee5791e31cb1d76096a",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--silwarp-v2",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "implicit_hyperdecoder_gate_v2",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    checks: dict[str, bool] = {}

    for name, expected in EXPECTED_SILWARP_V2.items():
        actual = sha256(args.silwarp_v2 / name)
        require(actual == expected, f"frozen SILWARP-v2 hash mismatch: {name}")
    checks["frozen_silwarp_v2_hashes"] = True

    stored_ledger = json.loads((bundle / "ledger.json").read_text(encoding="utf-8"))
    rebuilt_ledger = bridge_ledger.build_report(32)
    require(stored_ledger == rebuilt_ledger, "ledger.json is not the canonical rebuilt ledger")
    for system in rebuilt_ledger["systems"].values():
        for row in system.values():
            require(
                row["physical_bpw"] <= row["requested_cap_bpw"], "physical rate exceeds cap"
            )
            require(row["cold_read_amplification"] < 2.0, "cold read reaches 2x")
            require(row["unused_cap_bytes"] >= 0, "negative byte slack")
    checks["exact_integer_rate_and_read_ledgers"] = True

    synthetic_path = bundle / "synthetic_cupy_n19_p32.json"
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    require(synthetic["configuration"]["n"] == 1 << 19, "synthetic block is not N19")
    require(synthetic["configuration"]["device"]["backend"] == "cupy", "not a CuPy run")
    require(synthetic["configuration"]["phase_bins"] == 32, "wrong phase table")
    serial = synthetic["serialization"]
    require(serial["roundtrip_exact"] is True, "arithmetic round trip not exact")
    require(serial["escape_symbols"] == 0, "unexpected escape symbols")
    require(serial["logical_arithmetic_bits"] <= serial["payload_bytes"] * 8, "bad length")
    require(serial["arithmetic_redundancy_bits"] < 4.0, "unexpected coder redundancy")
    require(len(serial["payload_sha256"]) == 64, "bad payload digest")
    distortion = synthetic["distortion"]
    require(
        abs(distortion["empirical_over_represented"] - 1.0) < 0.01,
        "rotated dither distortion misses represented AWGN by >=1%",
    )
    system = synthetic["representative_128_expert_system"]
    require(system["physical_bpw"] < 2.5, "synthetic system exceeds 2.5 bpw")
    require(system["cold_read_amplification"] < 2.0, "synthetic system reaches 2x")
    # This is the decisive negative gate: a marginal ideal survivor at
    # s=0.1674 cannot absorb the scalar-lattice operational penalty.
    require(system["required_s_bpw"] > 0.40, "scalar bridge kill premise disappeared")
    require(
        system["required_s_bpw"] > bridge_ledger.IDEAL_GATE_REQUIRED_S + 0.20,
        "scalar bridge no longer decisively separated from ideal promotion",
    )
    checks["cupy_n19_real_serializer_and_negative_gate"] = True

    forbidden = ("qwen_aux_context_tensors", "unblinded", "pinned_panel_path")
    scanned = [
        bundle / "bridge_ledger.py",
        bundle / "synthetic_rotated_dither_gate.py",
        bundle / "test_source_free.py",
    ]
    for path in scanned:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            require(token not in source, f"forbidden payload token {token!r} in {path.name}")
    checks["executable_has_no_payload_path"] = True

    checksum_rows = (bundle / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    for row in checksum_rows:
        expected, name = row.split("  ", 1)
        require(sha256(bundle / name) == expected, f"bundle checksum mismatch: {name}")
    checks["bundle_checksums"] = True

    receipt = {
        "schema": "silwarp_finite_bridge_source_free_verification_v1",
        "status": "PASS_SOURCE_FREE_DESIGN_AND_SCALAR_NEGATIVE_GATE",
        "claim_boundary": (
            "Verifies source-free ledgers and synthetic bytes only; it does not prove a Qwen "
            "finite codec or a 20%-below-Gaussian result."
        ),
        "checks": checks,
        "synthetic_result_sha256": sha256(synthetic_path),
        "ledger_sha256": sha256(bundle / "ledger.json"),
        "frozen_silwarp_v2_hashes": EXPECTED_SILWARP_V2,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
