#!/usr/bin/env python3
"""Independent standard-library verifier for a completed RAVEL-6144-v1 result."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any


RESULT_FILES = {"COMPLETE.json", "fit_table_packet.bin", "result.json"}
SOURCE_FILES = {
    "MANIFEST.sha256", "README.md", "SOURCE_RECEIPT.json", "design_lock.json",
    "packet_codec.py", "ravel_stage0.py", "test_source_only.py", "verify_result.py",
    "verify_source.py",
}
ROLE_ORDER = ["gate", "up", "down"]
SEMANTICS = {
    "amplitude": "floor((decoded/row_scale + 4)*4), clipped to [0,31]; lower edges inclusive, upper edges exclusive before saturation",
    "boundary": "noncyclic self-clamp; a missing horizontal neighbor equals the center",
    "edge_state": "2*(neighbor >= 0) + 1*(abs(neighbor) > abs(center)); zero is nonnegative; magnitude ties are false",
    "flatten": "((((role*4 + row_class)*32 + amplitude)*4 + left_state)*4 + right_state); right_state fastest",
    "matrix_scale": "max(sqrt(mean(decoded_matrix^2) in FP64), 1e-30)",
    "role_order": ROLE_ORDER,
    "row_class": "count(log2(row_scale/matrix_scale) > threshold for threshold in [-0.25,0,0.25]); equality stays lower",
    "row_scale": "max(sqrt(mean(decoded_row^2) in FP64), 1e-30)",
}


class Failure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, label: str) -> None:
        self.count += 1
        if not condition:
            raise Failure(f"check {self.count} failed: {label}")

    def close(self, observed: float, expected: float, label: str, tolerance: float = 2e-12) -> None:
        self.require(math.isclose(float(observed), float(expected), rel_tol=tolerance, abs_tol=tolerance), label)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def strict_json(raw: bytes) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in rows:
            if key in out:
                raise Failure(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def finite(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise Failure("nonfinite JSON number")
        return result

    def bad(value: str) -> None:
        raise Failure(f"nonfinite JSON constant: {value}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=bad)


def held_read(path: Path) -> bytes:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise Failure(f"not regular/non-link: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise Failure(f"identity changed: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or len(raw) != opened.st_size:
            raise Failure(f"changed/short read: {path}")
        return raw
    finally:
        os.close(descriptor)


def parse_manifest(raw: bytes) -> dict[str, str]:
    if not raw.endswith(b"\n"):
        raise Failure("source manifest lacks LF")
    result: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        pieces = line.split("  ")
        if (len(pieces) != 2 or len(pieces[0]) != 64 or
                any(char not in "0123456789abcdef" for char in pieces[0])):
            raise Failure("malformed source manifest")
        name = pieces[1]
        if PurePosixPath(name).name != name or name in result:
            raise Failure("unsafe source manifest path")
        result[name] = pieces[0]
    return result


def parse_packet_independent(packet: bytes, checks: Checks) -> tuple[dict[str, Any], tuple[float, ...]]:
    checks.require(len(packet) == 16384, "packet length")
    newline = packet.find(b"\n", 0, 4096)
    checks.require(newline > 0, "packet header terminator")
    checks.require(packet[newline + 1:4096] == bytes(4096 - newline - 1), "zero header padding")
    header = strict_json(packet[:newline])
    expected_keys = {"dtype", "entries", "features", "format", "header_bytes", "packet_bytes",
                     "semantics", "semantics_sha256", "shared_table_count", "table_bytes",
                     "table_offset", "table_sha256", "version"}
    checks.require(set(header) == expected_keys, "packet header key set")
    expected = {"dtype": "<f2-finite", "entries": 6144, "features": [3, 4, 32, 4, 4],
                "format": "RAVEL6144-v1", "header_bytes": 4096, "packet_bytes": 16384,
                "semantics": SEMANTICS, "semantics_sha256": digest(canonical(SEMANTICS)),
                "shared_table_count": 1, "table_bytes": 12288, "table_offset": 4096,
                "version": 1}
    for key, value in expected.items():
        checks.require(header.get(key) == value, f"packet header {key}")
    table = packet[4096:]
    checks.require(digest(table) == header["table_sha256"], "packet table hash")
    values = struct.unpack("<" + "e" * 6144, table)
    checks.require(all(math.isfinite(value) for value in values), "finite packet table")
    return header, values


def verify(source_package: Path, result_dir: Path) -> int:
    checks = Checks()
    checks.require(source_package.is_dir() and not source_package.is_symlink(), "real source package")
    checks.require(result_dir.is_dir() and not result_dir.is_symlink(), "real result directory")
    source_package = source_package.resolve(strict=True)
    result_dir = result_dir.resolve(strict=True)
    checks.require({path.name for path in source_package.iterdir()} == SOURCE_FILES, "exact source closure")
    checks.require({path.name for path in result_dir.iterdir()} == RESULT_FILES, "exact completed result closure")
    source_raw = {name: held_read(source_package / name) for name in SOURCE_FILES}
    result_raw = {name: held_read(result_dir / name) for name in RESULT_FILES}
    manifest = parse_manifest(source_raw["MANIFEST.sha256"])
    checks.require(set(manifest) == SOURCE_FILES - {"MANIFEST.sha256"}, "source manifest member set")
    for name, expected in sorted(manifest.items()):
        checks.require(digest(source_raw[name]) == expected, f"source manifest hash {name}")
    manifest_sha = digest(source_raw["MANIFEST.sha256"])
    design = strict_json(source_raw["design_lock.json"])
    checks.require(design["schema"] == "ravel-decoded-residual-lut-stage0-design-lock-v1", "design schema")

    completion = strict_json(result_raw["COMPLETE.json"])
    checks.require(completion["schema"] == "ravel-decoded-residual-lut-stage0-completion-v1" and
                   completion["status"] == "COMPLETE", "completion receipt")
    completion_lock = completion.pop("completion_lock_sha256")
    checks.require(completion_lock == digest(canonical(completion)), "completion canonical lock")
    completion["completion_lock_sha256"] = completion_lock
    checks.require(set(completion["members"]) == {"fit_table_packet.bin", "result.json"}, "completion member set")
    for name in ("fit_table_packet.bin", "result.json"):
        row = completion["members"][name]
        checks.require(row["bytes"] == len(result_raw[name]), f"completion bytes {name}")
        checks.require(row["sha256"] == digest(result_raw[name]), f"completion hash {name}")
    checks.require(completion["source_package_manifest_sha256"] == manifest_sha, "completion source binding")
    header, values = parse_packet_independent(result_raw["fit_table_packet.bin"], checks)
    checks.require(len(values) == design["architecture"]["table_entries"], "packet/design entries")

    result = strict_json(result_raw["result.json"])
    checks.require(result["schema"] == "ravel-decoded-residual-lut-stage0-result-v1", "result schema")
    declared_lock = result.pop("result_lock_sha256")
    checks.require(declared_lock == digest(canonical(result)), "result canonical lock")
    result["result_lock_sha256"] = declared_lock
    checks.require(completion["result_lock_sha256"] == declared_lock, "completion result lock")
    bindings = result["bindings"]
    checks.require(bindings["source_package_manifest_sha256"] == manifest_sha, "result source manifest")
    checks.require(bindings["source_package_members"] == manifest, "result source member map")
    checks.require(bindings["script_sha256"] == manifest["ravel_stage0.py"], "result runner binding")
    checks.require(bindings["packet_sha256"] == digest(result_raw["fit_table_packet.bin"]), "result packet binding")
    panel = design["panel"]
    for key, expected in (("plan_sha256", panel["plan_file_sha256"]),
                          ("plan_internal_sha256", panel["plan_internal_sha256"]),
                          ("header_sha256", panel["header_sha256"]),
                          ("decoded_sha256", panel["decoded_reconstruction_sha256"]),
                          ("decoded_bytes", panel["decoded_reconstruction_bytes"])):
        checks.require(bindings[key] == expected, f"external binding {key}")
    receipts = bindings["sources"]
    checks.require(len(receipts) == 18, "18 source receipts")
    expected_identities = [(expert, role) for expert in range(6) for role in ROLE_ORDER]
    checks.require([(row["expert_ordinal"], row["role"]) for row in receipts] == expected_identities,
                   "source receipt order")
    checks.require(all(row["matrix_ordinal"] == index and row["bytes"] > 0 and
                       isinstance(row["sha256"], str) and len(row["sha256"]) == 64
                       for index, row in enumerate(receipts)), "source receipt structure")

    rows = result["matrix_rows"]
    checks.require(len(rows) == 18, "18 matrix ledgers")
    checks.require([(row["expert_ordinal"], row["role"]) for row in rows] == expected_identities,
                   "matrix ledger order")
    panel_sse = math.fsum(float(row["baseline_sse"]) for row in rows)
    panel_energy = math.fsum(float(row["source_energy"]) for row in rows)
    checks.close(panel_sse, result["baseline"]["panel_sse"], "panel SSE sum")
    checks.close(panel_energy, result["baseline"]["panel_energy"], "panel energy sum")
    checks.close(panel_sse, panel["baseline_sse"], "sealed panel SSE", 2e-9)
    checks.close(panel_energy, panel["baseline_energy"], "sealed panel energy", 2e-9)
    checks.close(panel_sse / panel_energy * 32.0, result["baseline"]["panel_F_at_2p5"], "panel F")
    holdout = [row for row in rows if row["expert_ordinal"] in (1, 4)]
    fit = [row for row in rows if row["expert_ordinal"] in (0, 2, 3, 5)]
    checks.require(len(holdout) == 6 and len(fit) == 12, "whole-expert split counts")
    checks.require(all(row["split"] == "holdout" and row["fit_fp16_table_sse"] is not None and
                       row["holdout_self_fit_fp64_oracle_sse"] is not None for row in holdout),
                   "holdout correction ledgers")
    checks.require(all(row["split"] == "fit" and row["fit_fp16_table_sse"] is None and
                       row["holdout_self_fit_fp64_oracle_sse"] is None for row in fit), "fit ledger isolation")
    holdout_sse = math.fsum(float(row["baseline_sse"]) for row in holdout)
    holdout_energy = math.fsum(float(row["source_energy"]) for row in holdout)
    finite_sse = math.fsum(float(row["fit_fp16_table_sse"]) for row in holdout)
    oracle_sse = math.fsum(float(row["holdout_self_fit_fp64_oracle_sse"]) for row in holdout)
    baseline = result["baseline"]
    checks.close(holdout_sse, baseline["holdout_sse"], "holdout SSE sum")
    checks.close(holdout_energy, baseline["holdout_energy"], "holdout energy sum")
    checks.close(holdout_sse / holdout_energy * 32.0, baseline["holdout_F_at_2p5"], "holdout F")
    finite = result["finite_fit_table"]
    oracle = result["source_leaking_oracle"]
    checks.close(finite_sse, finite["sse"], "finite SSE sum")
    checks.close(oracle_sse, oracle["sse"], "oracle SSE sum")
    checks.close(finite_sse / holdout_sse, finite["fraction_of_baseline_sse"], "finite ratio")
    checks.close(oracle_sse / holdout_sse, oracle["fraction_of_baseline_sse"], "oracle ratio")
    side = 8.0 * 16384 / 28311552
    checks.close(baseline["holdout_F_at_2p5"] * finite_sse / holdout_sse * 2.0 ** (2.0 * side),
                 finite["favorable_transfer_F"], "finite favorable F")
    oracle_f = baseline["holdout_F_at_2p5"] * oracle_sse / holdout_sse * 2.0 ** (2.0 * side)
    checks.close(oracle_f, oracle["favorable_transfer_F"], "oracle favorable F")
    tolerance = float(oracle["dominance_tolerance"])
    checks.require(oracle_sse <= holdout_sse + tolerance and oracle["dominates_zero_correction"] is True,
                   "oracle dominates zero correction")
    checks.require(oracle_sse <= finite_sse + tolerance and oracle["dominates_fit_fp16_table"] is True,
                   "oracle dominates fit table")
    checks.require(oracle["method"] == "numerical FP64 raw-SSE weighted least squares" and
                   oracle["emitted"] is False, "oracle wording/leakage")
    rate = result["rate_read"]
    checks.require(result["geometry"]["shared_table_count"] == rate["shared_table_count"] ==
                   design["architecture"]["shared_table_count"] == 1, "one-table contract")
    checks.close(rate["side_bpw"], side, "side rate")
    checks.close(rate["base_payload_cap_bpw"], 2.5 - side, "base rate")
    checks.close(rate["conservative_cold_amp"], 1.1805555555555556, "cold read")
    checks.require(rate["below_2x"] is True and rate["conservative_cold_amp"] < 2.0, "read gate")
    expected_status = "HARD_KILL_RAVEL6144_V1" if oracle_f > 0.8 else "PROMOTE_TO_CONTROLS_AND_REDUCED_RATE_REENCODE_ONLY"
    checks.require(result["status"] == expected_status, "decision")
    checks.require(result["controls"] == {"matched_controls_run": False,
                   "reason": "stage-0 source-leaking oracle gate decides promotion before controls"},
                   "control gate")
    return checks.count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        count = verify(args.source_package, args.result_dir)
    except Exception as exc:
        print(json.dumps({"schema": "ravel-v1-result-verifier-v0", "verdict": "BLOCK",
                          "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps({"schema": "ravel-v1-result-verifier-v0", "verdict": "PASS", "checks": count},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
