#!/usr/bin/env python3
"""Independent standard-library source audit for RAVEL-6144-v0."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST = "AUDIT_MANIFEST.json"
TOP = {MANIFEST, "README.md", "audit_receipt.json", "test_audit.py", "verify_audit.py", "evidence"}
EVIDENCE = {
    "MANIFEST.sha256": (496, "4859bad1347850a52bb3668a1015161ef9996ed9a649b6bff4f0bfb47c18ba5e"),
    "README.md": (3093, "9274c8311ec720d475ca8ad42d102ac8f87ec107d116dd48c788cf0829256985"),
    "SOURCE_RECEIPT.json": (386, "17e0799b3f5f8ff66132c4558d0afcbef3e5979c7e9a88685d197aca474503e4"),
    "design_lock.json": (2188, "f3aab79024aff2bbf09c1d955e3e07da08bceb435050c0f93612c5e7b18a243e"),
    "ravel_stage0.py": (13741, "a52eb36a5a9ed740fb0567c64cd87124d4739f7a8c991e53739dad4e93cb8fcd"),
    "test_source_only.py": (1921, "ca15604d9cbb331c4d08df67858fb6e4fe215586f52d82770ad554646aefbea7"),
    "verify_source.py": (7269, "f8e42e90f810b94bb8995d8c26c5c3e1692586ef4adcb7171694fe59b0245d14"),
}
FILES = {MANIFEST, "README.md", "audit_receipt.json", "test_audit.py", "verify_audit.py"} | {
    "evidence/" + name for name in EVIDENCE
}
HEX64 = set("0123456789abcdef")


class AuditFailure(RuntimeError):
    pass


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, label: str) -> None:
        self.count += 1
        if not condition:
            raise AuditFailure(f"check {self.count} failed: {label}")

    def equal(self, observed: Any, expected: Any, label: str) -> None:
        self.require(observed == expected, f"{label}: {observed!r} != {expected!r}")

    def close(self, observed: float, expected: float, label: str, tolerance: float = 1e-14) -> None:
        self.require(math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance), label)


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
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def finite(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result

    def bad_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_float=finite, parse_constant=bad_constant)


def held_read(path: Path) -> bytes:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise AuditFailure(f"not a regular non-link file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise AuditFailure(f"identity changed before open: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
        if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(raw) != opened.st_size:
            raise AuditFailure(f"file changed or short-read: {path}")
        return raw
    finally:
        os.close(descriptor)


def parse_producer_manifest(raw: bytes) -> dict[str, str]:
    if not raw.endswith(b"\n"):
        raise AuditFailure("producer manifest lacks final LF")
    rows: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        pieces = line.split("  ")
        if (len(pieces) != 2 or len(pieces[0]) != 64 or
                any(char not in HEX64 for char in pieces[0])):
            raise AuditFailure("malformed producer manifest")
        name = pieces[1]
        if PurePosixPath(name).name != name or name in rows:
            raise AuditFailure("unsafe or duplicate producer manifest path")
        rows[name] = pieces[0]
    return rows


def literal_assignments(tree: ast.AST) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def projection_counterexample() -> dict[str, float]:
    # Two legal coordinates in one cell, with different decoder-visible row RMS.
    scales = (1.0, 2.0)
    residuals = (1.0, 1.0)
    buggy = sum(r / s for r, s in zip(residuals, scales)) / len(scales)
    exact = sum(s * r for s, r in zip(scales, residuals)) / sum(s * s for s in scales)
    buggy_sse = sum((r - buggy * s) ** 2 for r, s in zip(residuals, scales))
    exact_sse = sum((r - exact * s) ** 2 for r, s in zip(residuals, scales))
    return {"unweighted_normalized_mean": buggy, "weighted_raw_sse_optimum": exact,
            "unweighted_raw_sse": buggy_sse, "optimal_raw_sse": exact_sse}


def packet_layout() -> dict[str, int]:
    header = canonical({"format": "RAVEL6144-v0", "entries": 6144, "dtype": "<f2",
                        "features": [3, 4, 32, 4, 4]}) + b"\n"
    table_bytes = 6144 * 2
    return {"header_bytes": len(header), "table_bytes": table_bytes,
            "padding_bytes": 16384 - len(header) - table_bytes,
            "table_offset_mod_2": len(header) % 2}


def verify(root: Path) -> int:
    checks = Checks()
    checks.require(root.is_dir() and not root.is_symlink(), "real audit directory")
    root = root.resolve(strict=True)
    checks.equal({path.name for path in root.iterdir()}, TOP, "exact top-level closure")
    evidence_dir = root / "evidence"
    checks.require(evidence_dir.is_dir() and not evidence_dir.is_symlink(), "real evidence directory")
    checks.equal({path.name for path in evidence_dir.iterdir()}, set(EVIDENCE), "exact evidence closure")
    for name in sorted(FILES):
        path = root / Path(name)
        checks.require(path.is_file() and not path.is_symlink(), f"regular non-link member {name}")
    held = {name: held_read(root / Path(name)) for name in FILES}

    audit_manifest = strict_json(held[MANIFEST])
    checks.equal(audit_manifest.get("schema"), "ravel-v0-independent-source-audit-manifest-v0", "audit manifest schema")
    checks.equal(audit_manifest.get("closed_world"), True, "closed-world flag")
    entries = audit_manifest.get("entries")
    checks.require(isinstance(entries, list) and len(entries) == len(FILES) - 1, "audit manifest entry count")
    checks.equal([row.get("path") for row in entries], sorted(FILES - {MANIFEST}), "audit manifest paths")
    for row in entries:
        name = row["path"]
        checks.equal(row.get("bytes"), len(held[name]), f"audit bytes {name}")
        checks.equal(row.get("sha256"), digest(held[name]), f"audit hash {name}")

    for name, (size, sha256) in sorted(EVIDENCE.items()):
        checks.equal(len(held["evidence/" + name]), size, f"evidence bytes {name}")
        checks.equal(digest(held["evidence/" + name]), sha256, f"evidence hash {name}")
    producer_manifest = parse_producer_manifest(held["evidence/MANIFEST.sha256"])
    checks.equal(set(producer_manifest), set(EVIDENCE) - {"MANIFEST.sha256"}, "producer manifest closure")
    for name, expected in sorted(producer_manifest.items()):
        checks.equal(digest(held["evidence/" + name]), expected, f"producer manifest hash {name}")

    design = strict_json(held["evidence/design_lock.json"])
    producer_receipt = strict_json(held["evidence/SOURCE_RECEIPT.json"])
    checks.equal(design["schema"], "ravel-decoded-residual-lut-stage0-design-lock-v0", "design schema")
    checks.equal(producer_receipt["status"], "READY_SOURCE_ONLY_NOT_EXECUTED", "producer not executed")
    arch = design["architecture"]
    checks.equal(arch["table_entries"], 3 * 4 * 32 * 4 * 4, "lookup geometry")
    checks.equal(arch["packet_bytes"], 16384, "packet bytes")
    panel = design["panel"]
    checks.equal(panel["weights"], 6 * 4718592, "panel denominator")
    side = 8.0 * 16384 / panel["weights"]
    checks.close(side, 0.004629629629629629, "independent side rate")
    checks.close(design["rate_and_read"]["side_bpw"], side, "sealed side rate")
    checks.close(design["rate_and_read"]["base_payload_cap_bpw"], 2.5 - side, "base rate after side")
    extra_read = 16384 / (4718592 * 2.5 / 8.0)
    checks.close(extra_read, 0.011111111111111112, "four-page read increment")
    cold = design["rate_and_read"]["published_worst_cold_page_read_amplification"] + extra_read
    checks.close(cold, 1.1805555555555556, "conservative cold read")
    checks.require(cold < design["rate_and_read"]["strict_read_limit"] == 2.0, "strict read gate")
    checks.close(panel["baseline_sse"] / panel["baseline_energy"] * 32.0,
                 panel["baseline_F"], "baseline F binding")
    checks.equal(design["split"]["fit_expert_ordinals"], [0, 2, 3, 5], "fit split")
    checks.equal(design["split"]["holdout_expert_ordinals"], [1, 4], "holdout split")
    checks.equal(sorted(design["split"]["fit_expert_ordinals"] + design["split"]["holdout_expert_ordinals"]),
                 list(range(6)), "split disjoint cover")

    source = held["evidence/ravel_stage0.py"].decode("utf-8")
    tree = ast.parse(source, filename="evidence/ravel_stage0.py")
    constants = literal_assignments(tree)
    for key, expected in (("PLAN_SHA256", panel["plan_file_sha256"]),
                          ("PLAN_INTERNAL_SHA256", panel["plan_internal_sha256"]),
                          ("HEADER_SHA256", panel["header_sha256"]),
                          ("DECODED_SHA256", panel["decoded_reconstruction_sha256"]),
                          ("BASELINE_SSE", panel["baseline_sse"]),
                          ("BASELINE_ENERGY", panel["baseline_energy"]),
                          ("BASELINE_F", panel["baseline_F"]),
                          ("PACKET_BYTES", 16384), ("FIT_EXPERTS", (0, 2, 3, 5)),
                          ("HOLDOUT_EXPERTS", (1, 4))):
        checks.equal(constants.get(key), expected, f"runner/design binding {key}")
    checks.equal(constants.get("ROLE_NAMES"), ("gate", "up", "down"), "role order")

    # Decoder visibility and feature definition.
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "features")
    checks.equal([arg.arg for arg in function.args.args], ["cp", "reconstruction", "role_index"], "feature inputs")
    feature_names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    checks.require("source" not in feature_names and "residual" not in feature_names, "features exclude source/residual")
    checks.require("cp.roll(reconstruction, 1, axis=1)" in source and
                   "cp.roll(reconstruction, -1, axis=1)" in source, "cyclic edge implementation")
    checks.require("(left >= 0.0)" in source and "cp.abs(left) > cp.abs(reconstruction)" in source,
                   "edge sign/magnitude state")
    checks.require("cp.floor(cp.clip((normalized + 4.0) * 4.0, 0.0, 31.0))" in source,
                   "amplitude bins")
    checks.require("(log_ratio > -0.25)" in source and "(log_ratio > 0.25)" in source,
                   "row-RMS classes")
    readme = held["evidence/README.md"].decode("utf-8").lower()
    checks.require("cyclic" not in readme and "wrap" not in readme,
                   "producer does not disclose cyclic boundary semantics")

    # Split is whole-expert and the finite table never consumes holdout residuals.
    checks.require("if expert in FIT_EXPERTS:\n            accumulate(cp, index, normalized_error, fit_sum, fit_count)\n        else:" in source,
                   "fit-only accumulation")
    checks.require("cached_holdout.append" in source and "for expert, role_index, role, index, scale, residual in cached_holdout:" in source,
                   "holdout-only evaluation")
    checks.require("fit_table_fp16[index] * scale" in source and "oracle_table[index] * scale" in source,
                   "frozen correction form")

    # Fatal projection defect: unweighted normalized mean is not raw-SSE LS when scale varies.
    checks.require("normalized_error = residual.reshape(-1) / scale" in source, "normalized residual")
    checks.require("counts += cp.bincount(index, minlength=TABLE_ENTRIES)" in source, "unweighted cell count")
    checks.require("oracle_sum / cp.maximum(oracle_count, 1.0)" in source, "unweighted oracle mean")
    example = projection_counterexample()
    checks.close(example["unweighted_normalized_mean"], 0.75, "counterexample implemented value")
    checks.close(example["weighted_raw_sse_optimum"], 0.6, "counterexample LS value")
    checks.close(example["unweighted_raw_sse"], 0.3125, "counterexample implemented SSE")
    checks.close(example["optimal_raw_sse"], 0.2, "counterexample optimal SSE")
    checks.require(example["unweighted_raw_sse"] > example["optimal_raw_sse"], "strict failure of dominance")
    checks.require("exact least-squares projection" in readme and design["oracle"]["definition"].startswith("exact FP64"),
                   "false exact-projection claim is material")

    # Favorable transfer arithmetic/wording itself is correctly scoped.
    checks.require("holdout_f0 * oracle_ratio * 2.0 ** (2.0 * SIDE_BPW)" in source,
                   "favorable transfer formula")
    checks.equal(design["oracle"]["source_leaking"], True, "source leakage disclosed")
    checks.equal(design["oracle"]["emitted"], False, "oracle not emitted")
    checks.require("not a finite reduced-rate codec" in source and "never a target result" in readme,
                   "conditional claim boundary")

    # Authorization/output checks occur before imports and payload calls.
    positions = [source.index(token) for token in (
        "if args.authorization != AUTHORIZATION:",
        'if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":',
        "if output.exists():",
        "plan_dir = args.plan_dir.resolve(strict=True)",
        "    import cupy as cp",
        "pairs, bindings = matrix_pairs(np, plan_dir)",
    )]
    checks.equal(positions, sorted(positions), "interlock ordering")
    checks.require("output.mkdir(parents=True, exist_ok=False)" in source and "os.O_EXCL" in source,
                   "collision-resistant output creation")
    checks.require("write_new(output / \"fit_table_packet.bin\"" in source and
                   "write_new(output / \"result.json\"" in source, "external output members")

    # Authentication is hash-before-reopen, not use of held authenticated bytes.
    for first, second, label in (
        ("sha256_file(plan_path) != PLAN_SHA256", "plan_path.read_text", "plan"),
        ("sha256_file(header_path) != HEADER_SHA256", "header_path.read_bytes", "header"),
        ("sha256_file(decoded_path) != DECODED_SHA256", "np.memmap(decoded_path", "decoded"),
        ("sha256_file(source_path) != str(row[\"source_bf16_sha256\"])", "bf16_to_f64(np, source_path", "source"),
    ):
        checks.require(first in source and second in source and source.index(first) < source.index(second),
                       f"hash-before-reopen defect {label}")

    checks.require("import cupy as cp" in source and "cp.bincount" in source and "dtype=cp.float64" in source,
                   "CuPy FP64 implementation")
    checks.require("cp.asnumpy(fit_table).astype(\"<f2\")" in source, "canonical little-endian FP16 table")
    checks.require("isfinite(fit_table_fp16" not in source and "isfinite(packet" not in source,
                   "packet FP16 finiteness is unchecked")
    layout = packet_layout()
    checks.equal(layout, {"header_bytes": 79, "table_bytes": 12288, "padding_bytes": 4017,
                          "table_offset_mod_2": 1}, "independent packet layout")
    checks.require("packet_header.find" not in source and "parse_packet" not in source,
                   "no packet parser or round-trip validation")

    receipt = strict_json(held["audit_receipt.json"])
    checks.equal(receipt["schema"], "ravel-v0-independent-source-audit-receipt-v0", "audit receipt schema")
    checks.equal(receipt["verdict"], "BLOCK_DO_NOT_LAUNCH", "audit verdict")
    checks.equal(receipt["producer_manifest_sha256"], EVIDENCE["MANIFEST.sha256"][1], "producer pin")
    checks.equal(receipt["findings"]["raw_sse_projection_dominance"], "FAIL", "projection finding")
    checks.equal(receipt["findings"]["external_evidence_use_binding"], "FAIL_HASH_THEN_REOPEN", "binding finding")
    checks.equal(receipt["verifier_check_count"], checks.count + 2, "receipt check count")
    body = dict(receipt)
    claim = body.pop("receipt_sha256", None)
    checks.require(isinstance(claim, str) and len(claim) == 64 and all(char in HEX64 for char in claim) and
                   digest(canonical(body)) == claim, "canonical receipt seal")
    return checks.count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    try:
        count = verify(args.audit_dir)
    except Exception as exc:
        print(json.dumps({"schema": "ravel-v0-independent-source-audit-verify-v0",
                          "verdict": "BLOCK", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    manifest_sha256 = digest(held_read(args.audit_dir.resolve() / MANIFEST))
    print(json.dumps({"schema": "ravel-v0-independent-source-audit-verify-v0",
                      "verdict": "PASS_AUDIT_BLOCKS_PRODUCER_LAUNCH", "checks": count,
                      "manifest_sha256": manifest_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
