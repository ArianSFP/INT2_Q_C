#!/usr/bin/env python3
"""Standard-library source verifier; this script grants no execution authority."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

import independent_decoder
import numeric_encoder
import packet_format
from packet_format import ContractError, ExpertGeometry, require


EXPECTED_FILES = (
    "IMPLEMENTATION_MAP.md",
    "README.md",
    "SYNTHETIC_SMOKE_RECEIPT.json",
    "design_lock.json",
    "independent_decoder.py",
    "numeric_encoder.py",
    "packet_format.py",
    "synthetic_cupy_smoke.py",
    "test_source_only.py",
    "verify_source.py",
)
ALLOWED_TOP_LEVEL_IMPORTS = {
    "__future__",
    "ast",
    "argparse",
    "dataclasses",
    "hashlib",
    "inspect",
    "json",
    "math",
    "os",
    "packet_format",
    "independent_decoder",
    "numeric_encoder",
    "pathlib",
    "random",
    "stat",
    "struct",
    "sys",
    "tempfile",
    "types",
    "typing",
    "unittest",
    "zlib",
}


def _strict_json(path: Path) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid design JSON: {exc}") from exc


def _source_root(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"TACTIC-N18-V4-SOURCE-ROOT-v1\0")
    for row in rows:
        name = row["name"].encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(int(row["bytes"]).to_bytes(8, "big"))
        digest.update(bytes.fromhex(row["sha256"]))
    return digest.hexdigest()


def verify() -> dict[str, Any]:
    observed = tuple(
        sorted(
            (
                path.name
                for path in PACKAGE.iterdir()
                if path.is_file() and path.name != "__pycache__"
            ),
            key=lambda name: name.encode("utf-8"),
        )
    )
    require(observed == EXPECTED_FILES, "exact source file closure")
    rows = []
    for name in observed:
        path = PACKAGE / name
        require(not path.is_symlink(), "source member symlink")
        raw = path.read_bytes()
        require(0 < len(raw) <= 1 << 20, "bounded source member")
        rows.append(
            {
                "name": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        if name.endswith(".py"):
            tree = ast.parse(raw.decode("utf-8"), filename=name)
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
            require(imports <= ALLOWED_TOP_LEVEL_IMPORTS | {"cupy", "numpy"}, f"unexpected import root in {name}")

    design = _strict_json(PACKAGE / "design_lock.json")
    require(
        isinstance(design, dict)
        and design.get("schema") == "tactic_actual_coarse_n18_source_design_v4"
        and design.get("status")
        == "SOURCE_ONLY_IMPLEMENTATION_CANDIDATE_NO_PAYLOAD_NO_CUDA_NO_RESULT",
        "design schema/status",
    )
    record = design["record"]
    require(
        record["magic_ascii"] == packet_format.MAGIC.decode("ascii")
        and record["reservoir_bytes"] == packet_format.RESERVOIR_BYTES
        and record["payload_bits"] == packet_format.PAYLOAD_BITS
        and record["logical_reserve_bits"] == 1024,
        "design/implementation record constants",
    )
    numerical = design["numerical"]
    require(
        numerical["encoder_core"]["sha256"] == numeric_encoder.ENCODER_SHA256
        and numerical["independent_decoder_core"]["sha256"]
        == independent_decoder.DECODER_SHA256,
        "design dependency hashes",
    )
    qwen = packet_format.qwen_frozen_ledgers()
    geometry = ExpertGeometry(768, 2048)
    require(
        qwen["coarse"]["panel_bytes"] == 8_487_936
        and geometry.target_eligible
        and geometry.frame_bytes == 1_414_656,
        "Qwen coarse ledger",
    )
    final = qwen["frozen_final_planning_topology_not_implemented_here"]
    require(
        final["cold_page_amplification"] == 73 / 72
        and final["forbidden_second_private_frame_pass_amplification"] > 2.0,
        "one-pass routed-read boundary",
    )
    smoke = _strict_json(PACKAGE / "SYNTHETIC_SMOKE_RECEIPT.json")
    require(
        smoke.get("status")
        == "PASS_SOURCE_FREE_NUMERICAL_ENCODE_INDEPENDENT_DECODE_REENCODE"
        and smoke.get("packet_bytes") == packet_format.RESERVOIR_BYTES
        and smoke.get("physical_bpw") == 307 / 128
        and smoke.get("canonical_reencode_matches") is True
        and smoke.get("capacity_margin_bits", 0) > 0,
        "source-free numerical smoke receipt",
    )
    return {
        "schema": "tactic_actual_coarse_n18_v4_source_verification",
        "status": "PASS_SOURCE_INVARIANTS_NO_EXECUTION_AUTHORITY",
        "source_files": len(rows),
        "source_bytes": sum(row["bytes"] for row in rows),
        "source_root_sha256": _source_root(rows),
        "files": rows,
        "packet": {
            "magic": packet_format.MAGIC.decode("ascii"),
            "record_bytes": packet_format.RESERVOIR_BYTES,
            "payload_bits": packet_format.PAYLOAD_BITS,
            "nominal_bits": packet_format.NOMINAL_BITS,
            "reserve_bits": packet_format.PAYLOAD_BITS - packet_format.NOMINAL_BITS,
        },
        "qwen_coarse": qwen["coarse"],
        "source_free_cupy_smoke": {
            "packet_sha256": smoke["packet_sha256"],
            "logical_bits": smoke["logical_bits"],
            "capacity_margin_bits": smoke["capacity_margin_bits"],
            "relative_mse_original_coordinates": smoke[
                "relative_mse_original_coordinates"
            ],
            "canonical_reencode_matches": smoke["canonical_reencode_matches"],
        },
        "one_pass_final_planning": final,
        "claim_boundary": (
            "source-only grammar/implementation verification; no runtime, payload, CUDA, "
            "Qwen MSE, TACTIC or target-result authority"
        ),
    }


def main() -> int:
    print(json.dumps(verify(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        raise SystemExit(f"v4 source verification failed: {exc}") from exc
