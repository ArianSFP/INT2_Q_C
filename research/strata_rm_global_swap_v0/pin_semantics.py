#!/usr/bin/env python3
"""Authenticate the external codec lineage without touching weight payloads."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


PINS = {
    "agent_polaris_qwen_rht_encoder.py":
        "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0",
    "bg_codec_bec_encoder.py":
        "456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267",
    "strata_v2_klt_mixed_independent_auditor_v1.py":
        "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e",
}
OVERLAP_RELATIVE = Path("INT2_Q_C/research/rm_bec_overlap_probe_v0_all14_aggregate.json")
OVERLAP_SHA256 = "dc4eb2f4896a466226974ac98f0cab2d4f5e9640b49d7c1d7c63ae957f6b7db2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _functions(text: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(text)
    return {node.name: node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def authenticate(external_root: Path) -> dict[str, Any]:
    root = external_root.resolve()
    paths = {name: root / name for name in PINS}
    for name, expected in PINS.items():
        require(paths[name].is_file(), f"missing pinned external source: {name}")
        require(sha256_file(paths[name]) == expected, f"external pin mismatch: {name}")

    base_text = paths["agent_polaris_qwen_rht_encoder.py"].read_text(encoding="utf-8")
    bg_text = paths["bg_codec_bec_encoder.py"].read_text(encoding="utf-8")
    audit_text = paths["strata_v2_klt_mixed_independent_auditor_v1.py"].read_text(encoding="utf-8")
    base_functions, bg_functions, audit_functions = (
        _functions(base_text), _functions(bg_text), _functions(audit_text))

    require({"bit_reverse_indices", "polar_transform", "sc_encode_ratio",
             "arithmetic_encode_binary", "arithmetic_decode_binary", "run_trial"}
            <= base_functions.keys(), "pinned base API")
    require({"bec_synthesized_z", "bec_flags"} <= bg_functions.keys(), "pinned BEC API")
    require({"decode_sc_level", "decode_one_block", "polar_transform"}
            <= audit_functions.keys(), "frozen STRATA audit API")

    sc_text = ast.get_source_segment(base_text, base_functions["sc_encode_ratio"]) or ""
    trial_text = ast.get_source_segment(base_text, base_functions["run_trial"]) or ""
    bec_text = ast.get_source_segment(bg_text, bg_functions["bec_flags"]) or ""
    synth_text = ast.get_source_segment(bg_text, bg_functions["bec_synthesized_z"]) or ""
    decode_text = ast.get_source_segment(audit_text, audit_functions["decode_one_block"]) or ""

    require("external_u=u[reverse].copy()" in sc_text.replace(" ", ""),
            "internal/external SC orientation")
    require("x_bit = polar_transform(chosen.external_u)" in trial_text,
            "base polar reconstruction orientation")
    require("sc_seed + 1_000_003 * level" in decode_text,
            "frozen current-random coset")
    require("previous += (1 << level_index) * x_bit.astype(np.int16)" in decode_text,
            "six-level 0..63 reconstruction semantics")
    require("capacity_q31" in synth_text and "1 << 31" in synth_text and
            "left + right - product" in synth_text, "integer-Q31 BEC synthesis")
    require("int(np.ceil(n * float(capacity)))" in bec_text and
            "external[order[:keep]] = 0" in bec_text and
            "external[reverse].copy()" in bec_text, "current BEC selected-set semantics")
    require("arithmetic_encode_binary(all_selected_bits, all_freq1)" in trial_text and
            "arithmetic_decode_binary(payload, arithmetic_logical_bits, all_freq1)" in trial_text,
            "causal arithmetic round trip")

    overlap = root / OVERLAP_RELATIVE
    require(overlap.is_file() and sha256_file(overlap) == OVERLAP_SHA256,
            "committed overlap receipt pin")
    overlap_json = json.loads(overlap.read_text(encoding="utf-8"))
    require(overlap_json.get("claim_boundary") ==
            "row-set overlap only; no Qwen payload or RD claim", "overlap claim boundary")

    return {
        "schema": "strata-rm-global-swap-v0-pin-semantics-receipt",
        "external_root": str(root),
        "pins": PINS,
        "checks": {
            "internal_external_orientation": True,
            "six_level_completed_index": True,
            "current_random_coset": True,
            "integer_q31_bec": True,
            "current_selected_count_boundary": True,
            "causal_arithmetic_roundtrip": True,
            "overlap_evidence_non_rd_only": True,
        },
        "known_blocker": (
            "historical pinned base CLI accepts only 2**10..2**18; target global "
            "2**20/2**21 integration requires independent current encoder audit"
        ),
        "status": "PASS_SOURCE_PINS__HOLD_PAYLOAD",
    }

