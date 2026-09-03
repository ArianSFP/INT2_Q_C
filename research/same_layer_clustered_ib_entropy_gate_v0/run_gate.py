"""Fail-closed source-release entrypoint.

This package is a design aperture only.  It intentionally contains neither a
payload locator nor deployment authority.
"""

from __future__ import annotations

import sys


PAYLOAD_EXECUTION_ENABLED = False


def main(argv: list[str] | None = None) -> int:
    del argv
    if not PAYLOAD_EXECUTION_ENABLED:
        print("HOLD: source-only CBIB-1 design; no payload execution authority", file=sys.stderr)
        return 2
    # There is deliberately no enabled branch in this source-only package.
    raise RuntimeError("deployment implementation is absent")


if __name__ == "__main__":
    raise SystemExit(main())
