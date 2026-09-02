#!/usr/bin/env python3
"""Import current encoder modules only from a preauthenticated private snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from authority_v2 import EXTERNAL_PINS, TARGET_N, real_directory, regular_bytes, require
from independent_rm_order import independent_cpu_order


def load_exact(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve(strict=True) == path.resolve(strict=True),
            f"module origin {name}")
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1,
            "worker requires python -I -B")
    require("PYTHONPATH" not in os.environ, "PYTHONPATH inherited")
    root = real_directory(args.external_snapshot, "external immutable snapshot")
    require({entry.name for entry in os.scandir(root)} == set(EXTERNAL_PINS) and
            all(entry.is_file(follow_symlinks=False) for entry in os.scandir(root)),
            "external snapshot exact closure")
    for name, expected in EXTERNAL_PINS.items():
        require(hashlib.sha256(regular_bytes(root / name, name)).hexdigest() == expected,
                f"external snapshot pin {name}")
    require("agent_polaris_qwen_rht_encoder" not in sys.modules and
            "bg_codec_bec_encoder" not in sys.modules,
            "current modules not preloaded")
    sys.path.insert(0, str(root))
    base = load_exact("agent_polaris_qwen_rht_encoder",
                      root / "agent_polaris_qwen_rht_encoder.py")
    bg = load_exact("bg_codec_bec_encoder", root / "bg_codec_bec_encoder.py")
    require(bg.base is base and base.reliability_freeze_flags is bg.bec_flags,
            "authenticated current reference hook identity")
    reference = bg.bec_flags

    def rm_ordered_truncated_polar_flags(repo, n, capacities):
        require(n in TARGET_N, "global production length")
        reference_rows = reference(repo, n, list(capacities))
        order = independent_cpu_order(n, base.np)
        output = []
        for row in reference_rows:
            row = base.np.asarray(row, dtype=base.np.uint8)
            selected = int(base.np.count_nonzero(row == 0))
            replacement = base.np.ones(n, dtype=base.np.uint8)
            replacement[order[:selected]] = 0
            require(int(base.np.count_nonzero(replacement == 0)) == selected,
                    "selected count invariant")
            output.append(replacement)
        return output

    base.reliability_freeze_flags = rm_ordered_truncated_polar_flags
    require(base.reliability_freeze_flags is rm_ordered_truncated_polar_flags and
            reference is bg.bec_flags, "final hook identity")
    capacities = [0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125]
    invocation_rows = []
    for n in TARGET_N:
        reference_rows = reference(None, n, capacities)
        installed_rows = base.reliability_freeze_flags(None, n, capacities)
        require(len(reference_rows) == len(installed_rows) == 6,
                "six invoked levels")
        selected = [int(base.np.count_nonzero(row == 0)) for row in reference_rows]
        replacement = [int(base.np.count_nonzero(row == 0)) for row in installed_rows]
        require(selected == replacement, "invoked selected counts preserved")
        invocation_rows.append({"n": n, "six_levels_invoked": True,
                                "selected_counts": selected,
                                "selected_counts_preserved": True})
    for name, expected in EXTERNAL_PINS.items():
        require(hashlib.sha256(regular_bytes(root / name,
                                             f"post-import {name}")).hexdigest() == expected,
                f"post-import immutable snapshot {name}")
    receipt = {
        "schema": "strata-rm-global-swap-v2-current-snapshot-receipt",
        "external_pins": EXTERNAL_PINS,
        "snapshot_path_received_not_external_root": True,
        "all_external_sources_imported_from_snapshot": True,
        "snapshot_hashes_rechecked_after_use": True,
        "reference_hook_identity": "bg_codec_bec_encoder.bec_flags",
        "installed_hook_identity":
            "agent_polaris_qwen_rht_encoder.reliability_freeze_flags",
        "global_hook_invocations": invocation_rows,
        "fresh_interpreter": True, "python_isolated_flag": True,
        "payloads_opened": 0, "rd_claim": False,
        "status": "PASS_IMMUTABLE_CURRENT_SNAPSHOT_AND_N20_N21_HOOK_INVOCATION",
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
