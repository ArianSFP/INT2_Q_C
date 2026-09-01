#!/usr/bin/env python3
"""Verify canonical self-seals and full-file hashes for audit receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
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
    return {
        "path": str(path),
        "canonical_bytes": len(canonical),
        "canonical_unsigned_sha256_claimed": claimed,
        "canonical_unsigned_sha256_computed": computed,
        "full_file_bytes": len(raw),
        "full_file_sha256": hashlib.sha256(raw).hexdigest(),
        "seal_valid": claimed == computed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", type=Path, nargs="+")
    args = parser.parse_args()
    reports = [verify(path) for path in args.receipts]
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0 if all(report["seal_valid"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
