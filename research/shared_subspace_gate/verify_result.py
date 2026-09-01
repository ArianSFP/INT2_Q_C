#!/usr/bin/env python3
"""Independent source/ledger verifier for the shared-subspace early gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MANIFEST_SHA256 = "4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782"
FIT = (0, 8, 16, 32, 40, 48, 64, 72, 80, 96, 104, 112)
VALID = (24, 56, 88, 120)
ROLES = ("up", "down")
ROWS, COLS = 768, 2048
N = ROWS * COLS
FULL_EXPERTS = 128
FRAME_BITS = 512


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(a: float, b: float, name: str, *, rtol: float = 2e-11, atol: float = 2e-12) -> None:
    require(math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol), f"{name}: {a} != {b}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def waterfill(energies: list[float], dims: list[int], bits: float) -> tuple[float, float, list[float]]:
    variances = [e / d for e, d in zip(energies, dims)]
    lo = min(variances) * 2.0 ** -80
    hi = max(variances)
    for _ in range(180):
        level = math.sqrt(lo * hi)
        used = 0.5 * sum(d * max(math.log2(v / level), 0.0) for v, d in zip(variances, dims))
        if used > bits:
            lo = level
        else:
            hi = level
    rates = [0.5 * max(math.log2(v / hi), 0.0) for v in variances]
    distortion = sum(d * min(v, hi) for v, d in zip(variances, dims)) / sum(energies)
    return distortion, hi, rates


def component_arrays(candidate: dict) -> tuple[list[float], list[int]]:
    family = candidate["family"]
    rr = int(candidate["right_rank"])
    lr = int(candidate["left_rank"])
    energies = []
    dims = []
    for role in ROLES:
        e = candidate["validation_component_energies"][role]
        total, right, left, core = (float(e[x]) for x in ("total", "right", "left", "core"))
        require(total > 0, "non-positive energy")
        if family == "right":
            energies += [right, total - right]
            dims += [len(VALID) * ROWS * rr, len(VALID) * ROWS * (COLS - rr)]
        elif family == "left":
            energies += [left, total - left]
            dims += [len(VALID) * lr * COLS, len(VALID) * (ROWS - lr) * COLS]
        elif family == "two_sided":
            energies += [core, left - core, right - core, total - left - right + core]
            dims += [
                len(VALID) * lr * rr,
                len(VALID) * lr * (COLS - rr),
                len(VALID) * (ROWS - lr) * rr,
                len(VALID) * (ROWS - lr) * (COLS - rr),
            ]
        else:
            raise RuntimeError(f"unknown family {family}")
    require(min(energies) >= -1e-7 * sum(energies), "negative orthogonal component")
    energies = [max(0.0, x) for x in energies]
    require(sum(dims) == len(VALID) * len(ROLES) * N, "dimension closure")
    return energies, dims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(sha256_file(args.manifest) == MANIFEST_SHA256, "manifest hash")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    identities = set()
    for row in manifest["tensors"]:
        key = (int(row["expert"]), str(row["role"]))
        identities.add(key)
        path = args.root / row["local_path"]
        require(path.is_file() and not path.is_symlink(), f"bad source {path}")
        require(path.stat().st_size == int(row["bytes"]), f"source size {path}")
        require(sha256_file(path) == row["sha256"], f"source hash {path}")
    expected = {(e, role) for e in FIT + VALID for role in ROLES}
    require(identities == expected, "source identity set")

    result = json.loads(args.result.read_text(encoding="utf-8"))
    declared_seal = result.pop("result_sha256")
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    require(hashlib.sha256(canonical).hexdigest() == declared_seal, "internal result seal")
    candidates = result["candidates"]
    require(len(candidates) == 160, "candidate count")
    total_weights = len(VALID) * len(ROLES) * N
    checked = 0
    eligible = []
    for candidate in candidates:
        mode = candidate["mode"]
        copies = 2 if mode == "role_specific" else 1
        rr, lr = int(candidate["right_rank"]), int(candidate["left_rank"])
        basis_bits = copies * (COLS * rr + ROWS * lr) * 16
        side_bpw = (basis_bits + FULL_EXPERTS * FRAME_BITS) / (FULL_EXPERTS * len(ROLES) * N)
        energies, dims = component_arrays(candidate)
        for row in candidate["rates"]:
            rate = float(row["physical_rate_bpw"])
            bits = (rate - side_bpw) * total_weights
            distortion, level, rates = waterfill(energies, dims, bits)
            f_value = distortion * 2.0 ** (2.0 * rate)
            payload_per_expert = (rate - side_bpw) * len(ROLES) * N
            read = (payload_per_expert + basis_bits + FRAME_BITS) / (rate * len(ROLES) * N)
            close(row["basis_physical_side_bpw"], side_bpw, "side bpw")
            require(int(row["basis_bits_cold"]) == basis_bits, "basis bits")
            close(row["payload_bits_validation"], bits, "payload bits")
            close(row["relative_mse"], distortion, "distortion")
            close(row["F"], f_value, "F")
            close(row["s_bpw"], -0.5 * math.log2(f_value), "s")
            close(row["water_level"], level, "water level")
            close(row["cold_read_amplification"], read, "read amp")
            require(bool(row["passes_read_lt_2"]) == (read < 2.0), "read decision")
            require(bool(row["passes_target_F_0p8"]) == (read < 2.0 and f_value <= 0.8), "target decision")
            require(len(row["component_rates_bpw"]) == len(rates), "rate vector length")
            for got, want in zip(row["component_rates_bpw"], rates):
                close(got, want, "component rate")
            if read < 2.0:
                eligible.append((f_value, read, mode, candidate["family"], rr, lr, rate))
            checked += 1
    require(checked == 480, "rate-row count")
    best = min(eligible)
    declared = result["best_eligible"]
    close(declared["F"], best[0], "best F")
    close(declared["cold_read_amplification"], best[1], "best read")
    require((declared["mode"], declared["family"], int(declared["right_rank"]), int(declared["left_rank"]), float(declared["physical_rate_bpw"])) == best[2:], "best identity")
    expected_decision = "PROMOTE_TO_PINNED_FINITE_CODEC" if best[0] <= 0.8 else "RETAIN_AS_COMPOSITE_LEAD" if best[0] <= 0.90 else "HARD_KILL_SHARED_LINEAR_SUBSPACE"
    require(result["decision"] == expected_decision, "aggregate decision")
    receipt = {
        "schema": "qwen_aux_shared_subspace_verification_v1",
        "status": "PASS",
        "sources_verified": len(expected),
        "candidate_rows_verified": checked,
        "result_sha256": sha256_file(args.result),
        "internal_result_seal": declared_seal,
        "decision": expected_decision,
        "best_F": best[0],
        "best_cold_read_amplification": best[1],
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

