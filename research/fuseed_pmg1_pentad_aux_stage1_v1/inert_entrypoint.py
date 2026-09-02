#!/usr/bin/env python3
"""Fail-closed proof that this source package cannot launch a payload run."""

from __future__ import annotations

import json


REFUSAL = {
    "schema": "fuseed-pmg1-fixed-pentad-aux-stage1-refusal-v1",
    "status": "REFUSED_NO_PAYLOAD_OR_RUN_AUTHORITY",
    "required_next": "independent source audit plus separate explicit auxiliary-only launch authority",
}


def main() -> int:
    print(json.dumps(REFUSAL, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
