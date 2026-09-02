#!/usr/bin/env python3
"""Read-only authentication of the current STRATA six-pass semantics."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from rm6_core import (generator_matrix, require, rm_dimension,
                      rm_information_positions)


AUDITOR_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"
AUDITOR_BYTES = 116835


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def authenticate_auditor(path: Path) -> dict[str, Any]:
    raw = path.resolve(strict=True).read_bytes()
    require(len(raw) == AUDITOR_BYTES and sha(raw) == AUDITOR_SHA256,
            "authenticated STRATA auditor")
    text = raw.decode("utf-8")
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    require({"decode_sc_level", "decode_one_block", "polar_transform"} <= functions.keys(),
            "STRATA function anchors")
    sc = ast.get_source_segment(text, functions["decode_sc_level"]) or ""
    block = ast.get_source_segment(text, functions["decode_one_block"]) or ""
    require("polar_transform(internal[reverse])" in sc, "bit-reverse polar output")
    require("for level_index, flag in enumerate(flags):" in block,
            "six level-major passes")
    require("previous += (1 << level_index) * x_bit.astype(np.int16)" in block,
            "0..63 reconstruction index")
    require("sc_seed + 1_000_003 * level" in block, "current random frozen coset")
    require("ETA = 0.25" in text and "ALPHABET_SIZE = 64" in text,
            "64-way alphabet constants")
    require("13 blocks of 2**21 values followed by one block of" in text and
            "2**20" in text, "current global polar geometry")
    return {"bytes": len(raw), "sha256": sha(raw),
            "level_major_sc_passes": 6, "indices": 64,
            "current_block_log2": [20, 21], "eta": 0.25,
            "random_frozen_coset_authenticated": True}


def verify_rm_orientation(variables: int = 12, order: int = 5) -> dict[str, Any]:
    positions = rm_information_positions(variables, order)
    matrix = generator_matrix(variables, order)
    observed = np.sum(matrix, axis=1, dtype=np.int64)
    expected = np.asarray([1 << int(position).bit_count() for position in positions],
                          dtype=np.int64)
    require(np.array_equal(observed, expected), "polar generator row orientation")
    threshold = 1 << (variables - order)
    require(np.all(observed >= threshold) and positions.size ==
            rm_dimension(order, variables), "exact RM row-weight set")
    return {"variables": variables, "order": order, "dimension": int(positions.size),
            "minimum_row_weight": int(observed.min()),
            "maximum_row_weight": int(observed.max()),
            "selection_rule": "popcount(internal_phase)>=m-r",
            "bit_reversal_preserves_popcount": True,
            "exact_rm_orientation": True}
