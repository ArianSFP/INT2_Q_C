#!/usr/bin/env python3
"""Independent standard-library result audit for RAVEL-6144-v1."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST = "AUDIT_MANIFEST.json"
SOURCE_NAMES = {"MANIFEST.sha256", "README.md", "SOURCE_RECEIPT.json", "design_lock.json",
                "packet_codec.py", "ravel_stage0.py", "test_source_only.py", "verify_result.py",
                "verify_source.py"}
RESULT_NAMES = {"COMPLETE.json", "fit_table_packet.bin", "result.json"}
EVIDENCE = {
    "evidence/source/MANIFEST.sha256": (661, "ffafd386ab4f3777fb6c9a70fa413f3bdf169658c64607f898a7969d0375c359"),
    "evidence/source/README.md": (6412, "55d18c4af473156da0a6481ead31e7ad5160957f1a2df14c4045f71e06f0be61"),
    "evidence/source/SOURCE_RECEIPT.json": (972, "8c2ca198b849932b66b5ec003116c378e0899da1702cdf8b3faf92a08a903694"),
    "evidence/source/design_lock.json": (4658, "67946e09306c1142cae3860fbedb3cb2ec05450cc7c9098de3ac97af0044a03b"),
    "evidence/source/packet_codec.py": (6920, "e06dde297e8eaa582f9b64994ce75a6281f025501bd9f5e8e03d152620c11ee2"),
    "evidence/source/ravel_stage0.py": (23459, "3417a5590c3f5ebbe798f23fa3aee0369fb39a5ce913ad5c3ad628a1c00f9a07"),
    "evidence/source/test_source_only.py": (4742, "2e58accb98e3ce3c21e7526aaddd0f5fab976f5df2c641868dc8bbfa6fcb3c0f"),
    "evidence/source/verify_result.py": (15605, "cb55763c5e4922cd16077cbf29017d567be62bfb7855d5dab9c4a8770223704c"),
    "evidence/source/verify_source.py": (15262, "db2d40114d80458a533034fcbf730e36f4bd880d2b4a2f1ce0814a99790440a0"),
    "evidence/result/COMPLETE.json": (676, "c61329ca4d1ec2d9be6a03733be772e253843769b79b57a88d620c34402cd185"),
    "evidence/result/fit_table_packet.bin": (16384, "5a8eea824c79e3421b0c1bcf00e04f609f19d59c892501761bf83a90d3b06c80"),
    "evidence/result/result.json": (13747, "ef67ee26246149472b5f3e4dc6f7e869d95c325a354abd7339bc8e4137dc0c47"),
}
FILES = {MANIFEST, "README.md", "audit_receipt.json", "test_audit.py", "verify_audit.py"} | set(EVIDENCE)
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
HEX = set("0123456789abcdef")


class Failure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, label: str) -> None:
        self.count += 1
        if not condition:
            raise Failure(f"check {self.count} failed: {label}")

    def equal(self, observed: Any, expected: Any, label: str) -> None:
        self.require(observed == expected, f"{label}: {observed!r} != {expected!r}")

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
            raise Failure(f"changed/short: {path}")
        return raw
    finally:
        os.close(descriptor)


def parse_source_manifest(raw: bytes) -> dict[str, str]:
    if not raw.endswith(b"\n"):
        raise Failure("source manifest lacks LF")
    rows: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        pieces = line.split("  ")
        if (len(pieces) != 2 or len(pieces[0]) != 64 or
                any(char not in HEX for char in pieces[0])):
            raise Failure("malformed source manifest")
        name = pieces[1]
        if PurePosixPath(name).name != name or name in rows:
            raise Failure("unsafe source manifest member")
        rows[name] = pieces[0]
    return rows


def parse_packet(packet: bytes, checks: Checks) -> dict[str, Any]:
    checks.require(len(packet) == 16384, "packet bytes")
    newline = packet.find(b"\n", 0, 4096)
    checks.require(newline > 0, "packet header LF")
    checks.require(packet[newline + 1:4096] == bytes(4096 - newline - 1), "packet zero padding")
    header = strict_json(packet[:newline])
    keys = {"dtype", "entries", "features", "format", "header_bytes", "packet_bytes",
            "semantics", "semantics_sha256", "shared_table_count", "table_bytes",
            "table_offset", "table_sha256", "version"}
    checks.equal(set(header), keys, "packet header keys")
    expected = {"dtype": "<f2-finite", "entries": 6144, "features": [3, 4, 32, 4, 4],
                "format": "RAVEL6144-v1", "header_bytes": 4096, "packet_bytes": 16384,
                "semantics": SEMANTICS, "semantics_sha256": digest(canonical(SEMANTICS)),
                "shared_table_count": 1, "table_bytes": 12288, "table_offset": 4096,
                "version": 1}
    for key, value in expected.items():
        checks.equal(header.get(key), value, f"packet header {key}")
    table = packet[4096:]
    checks.equal(digest(table), header["table_sha256"], "packet table hash")
    words = struct.unpack("<" + "H" * 6144, table)
    checks.require(all((word & 0x7C00) != 0x7C00 for word in words), "finite FP16 bit patterns")
    values = struct.unpack("<" + "e" * 6144, table)
    checks.require(all(math.isfinite(value) for value in values), "finite decoded FP16")
    checks.equal(sum((word & 0x7FFF) != 0 for word in words), 4427, "nonzero FP16 count")
    checks.close(min(values), -0.5126953125, "packet minimum", 0.0)
    checks.close(max(values), 0.46923828125, "packet maximum", 0.0)
    return header


def verify(root: Path) -> int:
    checks = Checks()
    checks.require(root.is_dir() and not root.is_symlink(), "real audit directory")
    root = root.resolve(strict=True)
    checks.equal({path.name for path in root.iterdir()},
                 {MANIFEST, "README.md", "audit_receipt.json", "test_audit.py", "verify_audit.py", "evidence"},
                 "exact audit top level")
    checks.equal({path.name for path in (root / "evidence").iterdir()}, {"source", "result"},
                 "exact evidence directories")
    checks.equal({path.name for path in (root / "evidence/source").iterdir()}, SOURCE_NAMES,
                 "exact copied source closure")
    checks.equal({path.name for path in (root / "evidence/result").iterdir()}, RESULT_NAMES,
                 "exact copied result closure")
    for name in sorted(FILES):
        checks.require((root / Path(name)).is_file() and not (root / Path(name)).is_symlink(),
                       f"regular audit member {name}")
    raw = {name: held_read(root / Path(name)) for name in FILES}
    audit_manifest = strict_json(raw[MANIFEST])
    checks.equal(audit_manifest.get("schema"), "ravel-v1-independent-result-audit-manifest-v0",
                 "audit manifest schema")
    checks.equal(audit_manifest.get("closed_world"), True, "audit closed world")
    entries = audit_manifest.get("entries")
    checks.require(isinstance(entries, list) and len(entries) == len(FILES) - 1, "audit manifest count")
    checks.equal([row.get("path") for row in entries], sorted(FILES - {MANIFEST}), "audit manifest paths")
    for row in entries:
        name = row["path"]
        checks.equal(row.get("bytes"), len(raw[name]), f"audit bytes {name}")
        checks.equal(row.get("sha256"), digest(raw[name]), f"audit hash {name}")
    for name, (size, sha256) in sorted(EVIDENCE.items()):
        checks.equal(len(raw[name]), size, f"evidence bytes {name}")
        checks.equal(digest(raw[name]), sha256, f"evidence hash {name}")

    source_manifest = parse_source_manifest(raw["evidence/source/MANIFEST.sha256"])
    checks.equal(set(source_manifest), SOURCE_NAMES - {"MANIFEST.sha256"}, "source manifest member set")
    for name, expected in sorted(source_manifest.items()):
        checks.equal(digest(raw["evidence/source/" + name]), expected, f"source manifest hash {name}")
    design = strict_json(raw["evidence/source/design_lock.json"])
    source_receipt = strict_json(raw["evidence/source/SOURCE_RECEIPT.json"])
    checks.equal(design["schema"], "ravel-decoded-residual-lut-stage0-design-lock-v1", "design schema")
    checks.equal(source_receipt["status"], "READY_SOURCE_ONLY_NOT_EXECUTED", "source receipt status")
    runner = raw["evidence/source/ravel_stage0.py"].decode("utf-8")
    ast.parse(runner, filename="ravel_stage0.py")
    checks.require("weights=scale * residual" in runner and "weights=scale * scale" in runner,
                   "weighted raw-SSE implementation")
    checks.require("oracle loses to legal zero-correction table" in runner and
                   "oracle loses to compared legal fit FP16 table" in runner, "dominance gates in source")
    checks.require("cp.roll" not in runner and "left[:, 0] = reconstruction[:, 0]" in runner,
                   "noncyclic source semantics")
    checks.require("read_regular_snapshot" in runner and "os.O_NOFOLLOW" in runner,
                   "held authenticated inputs")

    complete = strict_json(raw["evidence/result/COMPLETE.json"])
    completion_lock = complete.pop("completion_lock_sha256")
    checks.equal(completion_lock, digest(canonical(complete)), "completion canonical lock")
    complete["completion_lock_sha256"] = completion_lock
    checks.equal((complete["schema"], complete["status"]),
                 ("ravel-decoded-residual-lut-stage0-completion-v1", "COMPLETE"), "completion status")
    checks.equal(set(complete["members"]), {"fit_table_packet.bin", "result.json"}, "completion members")
    for name in ("fit_table_packet.bin", "result.json"):
        member = complete["members"][name]
        evidence_name = "evidence/result/" + name
        checks.equal(member["bytes"], len(raw[evidence_name]), f"completion bytes {name}")
        checks.equal(member["sha256"], digest(raw[evidence_name]), f"completion hash {name}")
    checks.equal(complete["source_package_manifest_sha256"], EVIDENCE["evidence/source/MANIFEST.sha256"][1],
                 "completion source binding")
    packet_header = parse_packet(raw["evidence/result/fit_table_packet.bin"], checks)
    checks.equal(packet_header["table_sha256"], "e4418f386d06562bf5500c7407a37ab8aea4bc022a8802e4e5c842f3e4f06734",
                 "table payload pin")

    result = strict_json(raw["evidence/result/result.json"])
    checks.equal(result["schema"], "ravel-decoded-residual-lut-stage0-result-v1", "result schema")
    result_lock = result.pop("result_lock_sha256")
    checks.equal(result_lock, digest(canonical(result)), "result canonical lock")
    result["result_lock_sha256"] = result_lock
    checks.equal(complete["result_lock_sha256"], result_lock, "completion/result lock")
    bindings = result["bindings"]
    checks.equal(bindings["source_package_manifest_sha256"], EVIDENCE["evidence/source/MANIFEST.sha256"][1],
                 "result source manifest")
    checks.equal(bindings["source_package_members"], source_manifest, "result source member map")
    checks.equal(bindings["script_sha256"], EVIDENCE["evidence/source/ravel_stage0.py"][1], "result runner")
    checks.equal(bindings["packet_sha256"], EVIDENCE["evidence/result/fit_table_packet.bin"][1], "result packet")
    panel = design["panel"]
    for key, expected in (("plan_sha256", panel["plan_file_sha256"]),
                          ("plan_internal_sha256", panel["plan_internal_sha256"]),
                          ("header_sha256", panel["header_sha256"]),
                          ("decoded_sha256", panel["decoded_reconstruction_sha256"]),
                          ("decoded_bytes", panel["decoded_reconstruction_bytes"])):
        checks.equal(bindings[key], expected, f"baseline binding {key}")
    source_rows = bindings["sources"]
    identities = [(expert, role) for expert in range(6) for role in ROLE_ORDER]
    checks.require(len(source_rows) == 18 and
                   [(row["expert_ordinal"], row["role"]) for row in source_rows] == identities,
                   "18 source receipt identities")
    for ordinal, row in enumerate(source_rows):
        checks.require(row["matrix_ordinal"] == ordinal and row["bytes"] == 3145728 and
                       isinstance(row["sha256"], str) and len(row["sha256"]) == 64,
                       f"source receipt structure {ordinal}")

    rows = result["matrix_rows"]
    checks.require(len(rows) == 18 and [(row["expert_ordinal"], row["role"]) for row in rows] == identities,
                   "18 matrix identities")
    for row in rows:
        checks.require(row["baseline_sse"] > 0.0 and row["source_energy"] > 0.0,
                       f"positive baseline row {row['expert_ordinal']}/{row['role']}")
    panel_sse = math.fsum(float(row["baseline_sse"]) for row in rows)
    panel_energy = math.fsum(float(row["source_energy"]) for row in rows)
    holdout = [row for row in rows if row["expert_ordinal"] in (1, 4)]
    fit = [row for row in rows if row["expert_ordinal"] in (0, 2, 3, 5)]
    checks.require(len(holdout) == 6 and len(fit) == 12, "whole-expert split")
    checks.require(all(row["split"] == "holdout" and row["fit_fp16_table_sse"] is not None and
                       row["holdout_self_fit_fp64_oracle_sse"] is not None for row in holdout),
                   "holdout correction rows")
    checks.require(all(row["split"] == "fit" and row["fit_fp16_table_sse"] is None and
                       row["holdout_self_fit_fp64_oracle_sse"] is None for row in fit), "fit rows unscored")
    holdout_sse = math.fsum(float(row["baseline_sse"]) for row in holdout)
    holdout_energy = math.fsum(float(row["source_energy"]) for row in holdout)
    finite_sse = math.fsum(float(row["fit_fp16_table_sse"]) for row in holdout)
    oracle_sse = math.fsum(float(row["holdout_self_fit_fp64_oracle_sse"]) for row in holdout)
    baseline = result["baseline"]
    checks.close(panel_sse, baseline["panel_sse"], "panel SSE")
    checks.close(panel_energy, baseline["panel_energy"], "panel energy")
    checks.close(panel_sse / panel_energy * 32.0, baseline["panel_F_at_2p5"], "panel F")
    checks.close(holdout_sse, baseline["holdout_sse"], "holdout SSE")
    checks.close(holdout_energy, baseline["holdout_energy"], "holdout energy")
    holdout_f = holdout_sse / holdout_energy * 32.0
    checks.close(holdout_f, baseline["holdout_F_at_2p5"], "holdout F")
    finite = result["finite_fit_table"]
    oracle = result["source_leaking_oracle"]
    checks.close(finite_sse, finite["sse"], "finite SSE")
    checks.close(oracle_sse, oracle["sse"], "oracle SSE")
    checks.close(finite_sse / holdout_sse, finite["fraction_of_baseline_sse"], "finite ratio")
    checks.close(oracle_sse / holdout_sse, oracle["fraction_of_baseline_sse"], "oracle ratio")
    checks.close(1.0 - oracle_sse / holdout_sse, oracle["capture"], "oracle capture")
    side = 8.0 * 16384 / 28311552
    finite_f = holdout_f * finite_sse / holdout_sse * 2.0 ** (2.0 * side)
    oracle_f = holdout_f * oracle_sse / holdout_sse * 2.0 ** (2.0 * side)
    checks.close(finite_f, finite["favorable_transfer_F"], "finite favorable F")
    checks.close(oracle_f, oracle["favorable_transfer_F"], "oracle favorable F")
    tolerance = float(oracle["dominance_tolerance"])
    checks.require(oracle_sse <= holdout_sse + tolerance and oracle["dominates_zero_correction"] is True,
                   "emitted oracle dominates zero correction")
    checks.require(oracle_sse <= finite_sse + tolerance and oracle["dominates_fit_fp16_table"] is True,
                   "emitted oracle dominates finite table")
    checks.equal((oracle["method"], oracle["emitted"]),
                 ("numerical FP64 raw-SSE weighted least squares", False), "oracle wording/leakage")
    rate = result["rate_read"]
    checks.equal(result["geometry"]["shared_table_count"], 1, "one-table geometry")
    checks.equal(rate["shared_table_count"], 1, "one-table ledger")
    checks.close(rate["side_bpw"], side, "side bpw")
    checks.close(rate["base_payload_cap_bpw"], 2.5 - side, "base payload cap")
    checks.close(rate["conservative_cold_amp"], 1.1805555555555556, "cold read")
    checks.require(rate["below_2x"] is True and rate["conservative_cold_amp"] < 2.0, "read gate")
    checks.require(oracle_f > 0.8 and result["status"] == "HARD_KILL_RAVEL6144_V1", "narrow hard kill")
    checks.close(oracle_f - 0.8, 0.21765194177793035, "kill margin")
    checks.equal(result["controls"], {"matched_controls_run": False,
                 "reason": "stage-0 source-leaking oracle gate decides promotion before controls"},
                 "control gate")
    checks.require("not a finite reduced-rate codec or universal converse" in result["claim_boundary"],
                   "narrow claim boundary")

    receipt = strict_json(raw["audit_receipt.json"])
    checks.equal(receipt["schema"], "ravel-v1-independent-result-audit-receipt-v0", "receipt schema")
    checks.equal(receipt["verdict"], "PASS_NARROW_HARD_KILL_WITH_ORACLE_REPLAY_LIMITATION", "receipt verdict")
    checks.equal(receipt["limitations"]["weighted_oracle_sse_replayable_without_payload"], False,
                 "receipt replay limitation")
    checks.equal(receipt["verifier_check_count"], checks.count + 2, "receipt check count")
    body = dict(receipt); claim = body.pop("receipt_sha256", None)
    checks.require(isinstance(claim, str) and len(claim) == 64 and all(char in HEX for char in claim) and
                   digest(canonical(body)) == claim, "receipt canonical seal")
    return checks.count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    try:
        count = verify(args.audit_dir)
    except Exception as exc:
        print(json.dumps({"schema": "ravel-v1-independent-result-audit-verify-v0", "verdict": "BLOCK",
                          "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps({"schema": "ravel-v1-independent-result-audit-verify-v0",
                      "verdict": "PASS_WITH_LIMITATION", "checks": count,
                      "manifest_sha256": digest(held_read(args.audit_dir.resolve() / MANIFEST))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
