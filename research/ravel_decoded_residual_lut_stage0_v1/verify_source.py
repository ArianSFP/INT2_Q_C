#!/usr/bin/env python3
"""Standard-library verifier for the sealed RAVEL-6144-v1 source package."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


FILES = {
    "MANIFEST.sha256", "README.md", "SOURCE_RECEIPT.json", "design_lock.json",
    "packet_codec.py", "ravel_stage0.py", "test_source_only.py", "verify_result.py",
    "verify_source.py",
}
MEMBERS = FILES - {"MANIFEST.sha256"}


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

    def close(self, observed: float, expected: float, label: str, tolerance: float = 1e-14) -> None:
        self.require(math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance), label)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
            raise Failure("nonfinite JSON")
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


def parse_manifest(raw: bytes) -> dict[str, str]:
    if not raw.endswith(b"\n"):
        raise Failure("manifest lacks final LF")
    result: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        pieces = line.split("  ")
        if (len(pieces) != 2 or len(pieces[0]) != 64 or
                any(char not in "0123456789abcdef" for char in pieces[0])):
            raise Failure("malformed manifest")
        name = pieces[1]
        if PurePosixPath(name).name != name or name in result:
            raise Failure("unsafe/duplicate manifest path")
        result[name] = pieces[0]
    return result


def assignments(tree: ast.AST) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return result


def load_packet_codec(package: Path) -> Any:
    spec = importlib.util.spec_from_file_location("ravel_v1_packet_codec", package / "packet_codec.py")
    if spec is None or spec.loader is None:
        raise Failure("cannot load packet codec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(package: Path) -> int:
    checks = Checks()
    checks.require(package.is_dir() and not package.is_symlink(), "real package directory")
    package = package.resolve(strict=True)
    checks.equal({path.name for path in package.iterdir()}, FILES, "exact nine-file closure")
    for name in sorted(FILES):
        checks.require((package / name).is_file() and not (package / name).is_symlink(), f"regular member {name}")
    raw = {name: held_read(package / name) for name in FILES}
    manifest = parse_manifest(raw["MANIFEST.sha256"])
    checks.equal(set(manifest), MEMBERS, "manifest member set")
    for name, expected in sorted(manifest.items()):
        checks.equal(digest(raw[name]), expected, f"manifest hash {name}")

    design = strict_json(raw["design_lock.json"])
    receipt = strict_json(raw["SOURCE_RECEIPT.json"])
    checks.equal(design["schema"], "ravel-decoded-residual-lut-stage0-design-lock-v1", "design schema")
    checks.equal(design["status"], "SOURCE_ONLY_NOT_EXECUTED", "design status")
    checks.equal(receipt["schema"], "ravel-decoded-residual-lut-stage0-source-receipt-v1", "receipt schema")
    checks.equal(receipt["status"], "READY_SOURCE_ONLY_NOT_EXECUTED", "receipt status")
    checks.equal(receipt["package_file_count"], len(FILES), "receipt file count")
    checks.equal(receipt["manifest_member_count"], len(MEMBERS), "receipt manifest count")
    checks.require(all(receipt["repairs"].values()), "all repair flags true")
    checks.require(all(value == 0 for value in receipt["access"].values()), "zero access ledger")

    arch = design["architecture"]
    checks.equal(arch["table_entries"], 3 * 4 * 32 * 4 * 4, "table geometry")
    checks.equal(arch["shared_table_count"], 1, "one shared table")
    packet_design = design["packet"]
    checks.equal((packet_design["packet_bytes"], packet_design["header_bytes"], packet_design["table_bytes"],
                  packet_design["table_offset"]), (16384, 4096, 12288, 4096), "aligned packet geometry")
    panel = design["panel"]
    checks.equal(panel["weights"], 6 * 4718592, "panel denominator")
    checks.equal(panel["decoded_reconstruction_bytes"], panel["weights"] * 8, "decoded bytes")
    side = 8.0 * 16384 / panel["weights"]
    checks.close(side, design["rate_and_read"]["side_bpw"], "side rate")
    checks.close(2.5 - side, design["rate_and_read"]["base_payload_cap_bpw"], "base rate")
    extra = 16384 / (4718592 * 2.5 / 8.0)
    checks.close(extra, design["rate_and_read"]["extra_table_read_amplification"], "extra read")
    checks.close(1.1694444444444445 + extra,
                 design["rate_and_read"]["conservative_cold_page_read_amplification"], "cold read")
    checks.require(design["rate_and_read"]["conservative_cold_page_read_amplification"] < 2.0,
                   "read gate")
    checks.close(panel["baseline_sse"] / panel["baseline_energy"] * 32.0, panel["baseline_F"], "baseline F")
    checks.equal(sorted(design["split"]["fit_expert_ordinals"] + design["split"]["holdout_expert_ordinals"]),
                 list(range(6)), "whole-expert split cover")
    checks.require(not set(design["split"]["fit_expert_ordinals"]).intersection(
        design["split"]["holdout_expert_ordinals"]), "split disjointness")
    checks.equal(design["projection"]["numerator"], "per-cell sum(row_scale * residual) in FP64",
                 "weighted numerator lock")
    checks.equal(design["projection"]["denominator"], "per-cell sum(row_scale^2) in FP64",
                 "weighted denominator lock")
    checks.require("one shared 16384-byte table" in design["rate_and_read"]["amortization_contract"],
                   "amortization contract")

    trees = {}
    for name in ("packet_codec.py", "ravel_stage0.py", "test_source_only.py", "verify_result.py", "verify_source.py"):
        text = raw[name].decode("utf-8")
        trees[name] = ast.parse(text, filename=name)
        compile(trees[name], name, "exec", dont_inherit=True)
    runner = raw["ravel_stage0.py"].decode("utf-8")
    constants = assignments(trees["ravel_stage0.py"])
    for key, expected in (("AUTHORIZATION", design["authorization"]),
                          ("PLAN_SHA256", panel["plan_file_sha256"]),
                          ("PLAN_INTERNAL_SHA256", panel["plan_internal_sha256"]),
                          ("HEADER_SHA256", panel["header_sha256"]),
                          ("DECODED_SHA256", panel["decoded_reconstruction_sha256"]),
                          ("FIT_EXPERTS", (0, 2, 3, 5)), ("HOLDOUT_EXPERTS", (1, 4)),
                          ("ROLE_NAMES", ("gate", "up", "down")), ("PACKET_BYTES", 16384),
                          ("BASELINE_SSE", panel["baseline_sse"]),
                          ("BASELINE_ENERGY", panel["baseline_energy"]),
                          ("BASELINE_F", panel["baseline_F"])):
        checks.equal(constants.get(key), expected, f"runner binding {key}")
    for token in ("requests", "urllib", "socket", "subprocess", "torch", "tensorflow"):
        checks.require(f"import {token}" not in runner and f"from {token}" not in runner,
                       f"forbidden import {token}")
    checks.require("DECODED_BYTES = PANEL_VALUES * 8" in runner, "decoded byte geometry binding")
    main_runner = runner[runner.index("def main() -> int:"):]
    order = [main_runner.index(token) for token in (
        "if args.authorization != AUTHORIZATION:",
        'if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":',
        "if os.path.lexists(output):",
        "package_bindings = self_authenticate()",
        "os.mkdir(output, 0o700)",
        "reject_link_components(plan_dir)",
        "    import cupy as cp",
        "pairs, bindings = matrix_pairs(np, plan_dir)",
    )]
    checks.equal(order, sorted(order), "authorization/reservation/input ordering")
    checks.require("os.O_NOFOLLOW" in runner and "os.lstat" in runner and "os.fstat" in runner,
                   "held regular-file authentication")
    checks.require("is_link_or_reparse" in runner and "reject_link_components" in runner,
                   "link/reparse rejection")
    checks.require("read_regular_snapshot(plan_dir / \"plan.lock.json\"" in runner and
                   "plan = strict_json(plan_raw)" in runner, "same plan snapshot parsed")
    checks.require("header_raw = read_regular_snapshot" in runner and "struct.unpack_from(\"<12f\", header_raw" in runner,
                   "same header snapshot parsed")
    checks.require("decoded_raw = read_regular_snapshot" in runner and "np.frombuffer(decoded_raw" in runner,
                   "same decoded snapshot parsed")
    checks.require("source_raw = read_regular_snapshot" in runner and "bf16_snapshot_to_f64(np, source_raw" in runner,
                   "same source snapshot parsed")
    checks.require("weights=scale * residual" in runner and "weights=scale * scale" in runner,
                   "weighted raw-SSE accumulation")
    checks.require("oracle loses to legal zero-correction table" in runner and
                   "oracle loses to compared legal fit FP16 table" in runner, "dominance gates")
    checks.require("numerical FP64 raw-SSE weighted least squares" in runner and "exact FP64" not in runner,
                   "numerical oracle wording")
    checks.require("cp.roll" not in runner and "left[:, 0] = reconstruction[:, 0]" in runner and
                   "right[:, -1] = reconstruction[:, -1]" in runner, "noncyclic boundary implementation")
    checks.require("cp.isfinite(fit_table)" in runner and "cp.isfinite(packet_table)" in runner,
                   "table finiteness gates")
    checks.require("parse_packet(packet)" in runner and "packet_table =" in runner,
                   "packet roundtrip before evaluation/output")
    checks.require("os.replace(stage / \"COMPLETE.json\"" in runner and
                   runner.index("os.replace(stage / \"COMPLETE.json\"") >
                   runner.index("os.replace(stage / \"result.json\""), "completion published last")
    checks.require("matrix_rows" in runner and "holdout_sse" in runner and "holdout_energy" in runner,
                   "matrix/holdout ledgers")

    codec = load_packet_codec(package)
    checks.equal(codec.SEMANTICS["boundary"],
                 "noncyclic self-clamp; a missing horizontal neighbor equals the center", "packet boundary lock")
    packet = codec.build_packet([0.0] * 6144)
    parsed = codec.parse_packet(packet)
    checks.equal(len(packet), 16384, "packet length")
    checks.equal(parsed["header"]["header_bytes"], 4096, "packet header alignment")
    checks.equal(len(parsed["values"]), 6144, "packet roundtrip entries")
    checks.require(all(value == 0.0 for value in parsed["values"]), "packet roundtrip values")
    checks.equal(packet[parsed["header_json_bytes"] + 1:4096],
                 bytes(4096 - parsed["header_json_bytes"] - 1), "packet zero padding")
    checks.equal(codec.reference_scalar_index(0, [1.0, 2.0, 3.0], 0, 2.0),
                 codec.reference_scalar_index(0, [1.0, 2.0, -3.0], 0, 2.0),
                 "left boundary ignores last coordinate")
    # The legal raw-SSE optimum strictly improves the v0 unweighted normalized mean.
    scales = (1.0, 2.0); residuals = (1.0, 1.0)
    exact = sum(scale * residual for scale, residual in zip(scales, residuals)) / sum(scale * scale for scale in scales)
    buggy = sum(residual / scale for scale, residual in zip(scales, residuals)) / len(scales)
    exact_sse = sum((residual - exact * scale) ** 2 for scale, residual in zip(scales, residuals))
    buggy_sse = sum((residual - buggy * scale) ** 2 for scale, residual in zip(scales, residuals))
    checks.close(exact, 0.6, "weighted counterexample entry")
    checks.close(exact_sse, 0.2, "weighted counterexample SSE")
    checks.require(exact_sse < buggy_sse, "weighted counterexample dominance")
    checks.require("parse_packet_independent" in raw["verify_result.py"].decode("utf-8"),
                   "independent result packet parser")
    return checks.count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        count = verify(args.package)
    except Exception as exc:
        print(json.dumps({"schema": "ravel-v1-source-verifier-v0", "verdict": "BLOCK",
                          "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    manifest_sha = digest(held_read(args.package.resolve() / "MANIFEST.sha256"))
    print(json.dumps({"schema": "ravel-v1-source-verifier-v0", "verdict": "PASS",
                      "checks": count, "manifest_sha256": manifest_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
