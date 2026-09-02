#!/usr/bin/env python3
"""Independent fail-closed verifier for the untouched N18 v4 source snapshot."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parent
PRODUCER = ROOT.parent / "tactic_actual_coarse_n18_v4"
REPOSITORY = ROOT.parents[1]
EXPECTED_AUDIT_FILES = {
    "AUDIT_MANIFEST.json",
    "README.md",
    "audit_lock.json",
    "audit_receipt.json",
    "verify_audit.py",
}
EXPECTED_PRODUCER = {
    "IMPLEMENTATION_MAP.md": (3901, "262ce7936ae2e7632dbc8bb3d7ec7900f36ceb1cbc59dcde8832a2ec6afc6758"),
    "README.md": (6557, "090535bae370d0e4d636dbe804c1063795aff5b27085de00bf74e13ae2a23897"),
    "SYNTHETIC_SMOKE_RECEIPT.json": (2563, "f2106d279295b0ec28b4ca32c59d733077f09d89cb5e50dbf4b6b920028639b2"),
    "design_lock.json": (4497, "28f635337a797f4081f02588e590d780c1b55b9dc2d8284065edf431a67b45d8"),
    "independent_decoder.py": (18768, "1b8ddcf5d8199769252c2db5a5cc36127c3577e6d9b0ee256741f7b2791a3d20"),
    "numeric_encoder.py": (13953, "2508c69fa63fd7fbe8f9dfe66844542a6c916688d8c635bb04e941041344db8b"),
    "packet_format.py": (18925, "acf308843068399a436e88f32f76632ccea3cbc9ceeb9bcbe5faeb604f4c42d4"),
    "synthetic_cupy_smoke.py": (3506, "7d89f4be10bb06b3c1b9a8b2f008cdd43f70d7b76fec03cd8ebf677eff97cc8a"),
    "test_source_only.py": (11094, "4820f9523f58302fc2f84229caa1aac03024a20e674e276638c675d7b10f3147"),
    "verify_source.py": (7202, "09cfd7617afaa209c1c8e8dd9c0d9d85481f43225604aed79f2e278238258a12"),
}
EXPECTED_DEPENDENCIES = {
    "src/polaris_sc_v2_rht_encoder.py": (29633, "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0"),
    "strata_v2_klt_mixed_independent_auditor_v1.py": (116835, "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"),
}
EXPECTED_SOURCE_ROOT = "1f9f2c92df3796f5f23b7e3a6b0826d6d8a2ea53bc70014fb75e61e7bc8a9fbf"


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def strict_json(path: Path) -> Any:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    def reject(value: str):
        raise AuditError(f"nonfinite JSON constant in {path.name}: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )


def producer_root(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"TACTIC-N18-V4-SOURCE-ROOT-v1\0")
    for row in rows:
        name = row["name"].encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(int(row["bytes"]).to_bytes(8, "big"))
        digest.update(bytes.fromhex(row["sha256"]))
    return digest.hexdigest()


def verify_producer_closure() -> int:
    observed = {path.name for path in PRODUCER.iterdir()}
    require(observed == set(EXPECTED_PRODUCER), "producer exact closure")
    rows = []
    for name in sorted(EXPECTED_PRODUCER, key=lambda value: value.encode("utf-8")):
        path = PRODUCER / name
        expected_bytes, expected_hash = EXPECTED_PRODUCER[name]
        require(path.is_file() and not path.is_symlink(), f"producer regular member: {name}")
        require(path.stat().st_size == expected_bytes, f"producer bytes: {name}")
        require(sha256_file(path) == expected_hash, f"producer hash: {name}")
        rows.append({"name": name, "bytes": expected_bytes, "sha256": expected_hash})
    require(producer_root(rows) == EXPECTED_SOURCE_ROOT, "producer source root")
    for relative, (expected_bytes, expected_hash) in EXPECTED_DEPENDENCIES.items():
        path = REPOSITORY / relative
        require(path.is_file() and not path.is_symlink(), f"dependency regular file: {relative}")
        require(path.stat().st_size == expected_bytes, f"dependency bytes: {relative}")
        require(sha256_file(path) == expected_hash, f"dependency hash: {relative}")
    return len(rows) + len(EXPECTED_DEPENDENCIES)


def verify_audit_closure() -> int:
    observed = {path.name for path in ROOT.iterdir()}
    require(observed == EXPECTED_AUDIT_FILES, "audit exact closure")
    manifest = strict_json(ROOT / "AUDIT_MANIFEST.json")
    require(manifest["self_unlisted"] == "AUDIT_MANIFEST.json", "audit manifest self rule")
    rows = manifest["files"]
    require({row["path"] for row in rows} == EXPECTED_AUDIT_FILES - {"AUDIT_MANIFEST.json"}, "audit manifest closure")
    for row in rows:
        path = ROOT / row["path"]
        require(path.is_file() and not path.is_symlink(), f"audit regular member: {row['path']}")
        require(path.stat().st_size == int(row["bytes"]), f"audit bytes: {row['path']}")
        require(sha256_file(path) == row["sha256"], f"audit hash: {row['path']}")
    return len(rows)


def import_producer_modules():
    sys.path.insert(0, str(PRODUCER))
    packet_format = importlib.import_module("packet_format")
    numeric_encoder = importlib.import_module("numeric_encoder")
    independent_decoder = importlib.import_module("independent_decoder")
    return packet_format, numeric_encoder, independent_decoder


def verify_rate_tail_and_pages(packet_format: Any) -> int:
    require(8 * 78_592 * 128 == (1 << 18) * 307, "record 307/128 identity")
    require(packet_format.PAYLOAD_BITS == 627_712, "payload bits")
    require(packet_format.NOMINAL_BITS == 626_688, "nominal bits")
    require(packet_format.PAYLOAD_BITS - packet_format.NOMINAL_BITS == 1024, "reserve bits")
    qwen = packet_format.ExpertGeometry(768, 2048)
    require(qwen.target_eligible and qwen.records == 18, "Qwen no-tail geometry")
    require(qwen.frame_bytes == 1_414_656, "Qwen expert coarse bytes")
    require(8 * qwen.frame_bytes * 128 == qwen.values * 307, "Qwen exact rate")
    tail = packet_format.ExpertGeometry(1, (1 << 18) + 17)
    require(not tail.target_eligible and tail.streams_per_role == 2, "tail ineligibility")
    require(8.0 * tail.frame_bytes / tail.values > 307 / 128, "tail charged rate")
    ledgers = packet_format.qwen_frozen_ledgers()
    final = ledgers["frozen_final_planning_topology_not_implemented_here"]
    require(final["selected_pages"] == 365 and final["equal_share_pages"] == 360, "one-pass pages")
    require(final["cold_page_amplification"] == 73 / 72, "one-pass ratio")
    require(final["forbidden_second_private_frame_pass_pages"] == 724, "two-pass pages")
    require(final["forbidden_second_private_frame_pass_amplification"] == 181 / 90, "two-pass ratio")
    require(181 / 90 > 2.0, "two-pass strict failure")
    one = packet_format.frame_ledger(qwen, compressed_passes=1)
    require(one["repeated_compressed_bytes"] == qwen.frame_bytes, "current total-pass field semantics")
    return 14


def verify_terminal_overflow(packet_format: Any, numeric_encoder: Any) -> int:
    count = {"run_trial": 0}
    logical_bits = packet_format.PAYLOAD_BITS + 1
    payload = bytes((logical_bits + 7) // 8)

    class FakeEncoder:
        @staticmethod
        def run_trial(*_arguments):
            count["run_trial"] += 1
            legacy = struct.pack("<If", logical_bits, 1.0) + payload
            return {
                "_container_hex": legacy.hex(),
                "arithmetic_logical_bits": logical_bits,
                "arithmetic_payload_sha256": hashlib.sha256(payload).hexdigest(),
                "relative_mse": 0.0,
                "arithmetic_roundtrip_bits_match": True,
                "causal_decoder_frequencies_match": True,
                "reconstruction_indices_match": True,
            }

    original_flags = numeric_encoder._flags
    numeric_encoder._flags = lambda _runtime: (
        {"capacities": [], "test_channel_distortion": 0.1},
        [],
    )
    try:
        raw = struct.pack("<H", 0x3F80) + bytes(2 * packet_format.N - 2)
        runtime = SimpleNamespace(encoder=FakeEncoder())
        try:
            numeric_encoder.encode_tile(
                raw,
                packet_format.ExpertGeometry(packet_format.N, 1),
                0,
                0,
                runtime,
            )
        except packet_format.ContractError as exc:
            require("overflow" in str(exc), "terminal overflow error")
        else:
            raise AuditError("fake numerical overflow accepted")
    finally:
        numeric_encoder._flags = original_flags
    require(count["run_trial"] == 1, "exactly one numerical trial before overflow")
    try:
        packet_format.bits_to_payload(0 for _ in range(packet_format.PAYLOAD_BITS + 1))
    except packet_format.ContractError:
        pass
    else:
        raise AuditError("grammar overflow accepted")
    return 3


def verify_separation_and_findings(numeric_encoder: Any, independent_decoder: Any) -> int:
    encoder_source = (PRODUCER / "numeric_encoder.py").read_text(encoding="utf-8")
    decoder_source = (PRODUCER / "independent_decoder.py").read_text(encoding="utf-8")
    decoder_tree = ast.parse(decoder_source)
    imports = set()
    for node in ast.walk(decoder_tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    require("numeric_encoder" not in imports, "decoder imports encoder")
    require(numeric_encoder.DECODER_SHA256 == independent_decoder.DECODER_SHA256, "shared construction binding")
    require("all(row.canonical_packet for row in decoded_rows)" in decoder_source, "R1 aggregate receipt finding")
    for name in (
        "arithmetic_roundtrip_bits_match",
        "causal_decoder_frequencies_match",
        "reconstruction_indices_match",
    ):
        require(f'"{name}": row["{name}"] is True' in encoder_source, f"R2 reported flag: {name}")
    require("canonical I16 symbol overflow" in decoder_source, "R3 I16 terminal gate")
    encode_tree = ast.parse(encoder_source)
    run_trial_calls = [
        node
        for node in ast.walk(encode_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_trial"
    ]
    require(len(run_trial_calls) == 1, "one run_trial source call")
    require('require(logical_bits <= PAYLOAD_BITS, "fixed-reservoir overflow; retry forbidden")' in encoder_source, "terminal overflow source guard")
    return 8


def verify_locks() -> int:
    lock = strict_json(ROOT / "audit_lock.json")
    receipt = strict_json(ROOT / "audit_receipt.json")
    require(lock["verdict"] == "PASS_FIXED_SOURCE_GRAMMAR_BLOCK_PAYLOAD_PROMOTION_PENDING_REPAIRS", "audit verdict")
    require(lock["producer"]["source_root_sha256"] == EXPECTED_SOURCE_ROOT, "lock source root")
    require(len(lock["repair_before_payload"]) == 5, "repair finding count")
    require(all(value is False for value in lock["authority"].values()), "lock non-authority")
    require(receipt["status"] == "PASS_SOURCE_AUDIT_WITH_REPAIR_BLOCKERS_NO_PAYLOAD_AUTHORITY", "receipt status")
    require(receipt["source_free_cupy_replay"]["matches_checked_in_receipt"] is True, "CuPy replay")
    require(math.isclose(receipt["arithmetic"]["forbidden_second_private_pass_amplification"], 181 / 90), "receipt second pass")
    return 7


def main() -> int:
    audit_files = verify_audit_closure()
    source_files = verify_producer_closure()
    packet_format, numeric_encoder, independent_decoder = import_producer_modules()
    checks = 0
    checks += verify_rate_tail_and_pages(packet_format)
    checks += verify_terminal_overflow(packet_format, numeric_encoder)
    checks += verify_separation_and_findings(numeric_encoder, independent_decoder)
    checks += verify_locks()
    output = {
        "schema": "tactic-actual-coarse-n18-v4-independent-source-audit-verification-v1",
        "status": "PASS_SOURCE_AUDIT_WITH_REPAIR_BLOCKERS_NO_PAYLOAD_AUTHORITY",
        "producer_source_root_sha256": EXPECTED_SOURCE_ROOT,
        "producer_and_dependency_files": source_files,
        "audit_manifest_members": audit_files,
        "independent_checks": checks,
        "repair_findings": 5,
        "qwen_payload_opened": 0,
        "network_operations": 0,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"independent v4 audit verification failed: {exc}") from exc
