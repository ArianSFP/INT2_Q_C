#!/usr/bin/env python3
"""Inert entrypoint and explicit API surface for source-moment publication."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


PACKAGE = Path(__file__).resolve().parent


def _load_contract() -> Any:
    spec = importlib.util.spec_from_file_location(
        "uwfa_source_moment_contract_runtime", PACKAGE / "moment_contract.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned moment contract")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def publish_authenticated_contract(**kwargs: Any) -> dict[str, Any]:
    """Forward only explicit keyword arguments from a reviewed dispatcher."""
    return _load_contract().publish_authenticated_contract(**kwargs)


def direct_main() -> int:
    print(
        "BLOCKED_SOURCE_ONLY: this entrypoint is inert; use an externally pinned "
        "reviewed dispatcher with exact source and authorization digests",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(direct_main())
