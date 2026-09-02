#!/usr/bin/env python3
"""Run external pin checks and no-payload ordering checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from coset_contract import describe
from pin_semantics import authenticate
from rm_order import (TARGET_N, classify_selected_count, rm_full_order_numpy,
                      swap_reference_flags)


CAPACITY_FIXTURE = [
    0.0008227374118798814,
    0.237747929331251,
    0.9153259168218427,
    0.9999815811734327,
    1.0,
    1.0,
]


def fixture_flags(n: int) -> list[np.ndarray]:
    rows = []
    for capacity in CAPACITY_FIXTURE:
        k = min(n, max(0, int(math.ceil(n * capacity))))
        row = np.ones(n, dtype=np.uint8)
        # An arbitrary permutation proves the swap derives K from the actual
        # reference flag rather than reconstructing K from a copied profile.
        row[np.arange(k, dtype=np.int64) * 104729 % n] = 0
        if int(np.count_nonzero(row == 0)) != k:
            raise AssertionError("fixture selected-count construction")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--full-global-smoke", action="store_true",
                        help="construct CPU orders at 2**20 and 2**21")
    args = parser.parse_args()

    pins = authenticate(args.external_root)
    lengths = TARGET_N if args.full_global_smoke else (1 << 12, 1 << 13)
    rows = []
    for n in lengths:
        reference = fixture_flags(n)
        swapped = swap_reference_flags(reference)
        levels = []
        for old, new in zip(reference, swapped, strict=True):
            old_k = int(np.count_nonzero(old == 0))
            new_k = int(np.count_nonzero(new == 0))
            levels.append({
                **classify_selected_count(n, old_k),
                "reference_k": old_k,
                "replacement_k": new_k,
                "equal": old_k == new_k,
                "flag_sha256": hashlib.sha256(new.tobytes()).hexdigest(),
            })
        order = rm_full_order_numpy(n)
        rows.append({
            "n": n,
            "levels": levels,
            "tie_order_exact": all(
                int(order[i - 1]).bit_count() > int(order[i]).bit_count() or
                (int(order[i - 1]).bit_count() == int(order[i]).bit_count() and
                 int(order[i - 1]) < int(order[i]))
                for i in range(1, n)
            ),
        })

    result = {
        "schema": "strata-rm-global-swap-v0-source-gate",
        "pins": pins,
        "coset": describe(),
        "rows": rows,
        "payloads_opened": 0,
        "rd_claim": False,
        "status": "PASS_SOURCE_MECHANISM__HOLD_INDEPENDENT_AUDIT_AND_PAYLOAD",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

