#!/usr/bin/env python3
"""Positive probe for the bootstrap's authenticated in-memory source importer."""

import fosp_runtime_probe_dependency


if fosp_runtime_probe_dependency.VALUE != "SEALED_RUNTIME_SOURCE_PASS":
    raise SystemExit("sealed runtime dependency value mismatch")
print(fosp_runtime_probe_dependency.VALUE)
