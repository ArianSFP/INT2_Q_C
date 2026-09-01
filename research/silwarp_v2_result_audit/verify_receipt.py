#!/usr/bin/env python3
"""Recompute the canonical and full-file hashes of an audit receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()

    raw = args.receipt.read_bytes()
    document = json.loads(raw)
    claimed = document["seal"].pop("canonical_unsigned_sha256")
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    computed = hashlib.sha256(canonical).hexdigest()
    full_file = hashlib.sha256(raw).hexdigest()
    output = {
        "canonical_bytes": len(canonical),
        "canonical_unsigned_sha256_claimed": claimed,
        "canonical_unsigned_sha256_computed": computed,
        "full_file_bytes": len(raw),
        "full_file_sha256": full_file,
        "seal_valid": claimed == computed,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["seal_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
