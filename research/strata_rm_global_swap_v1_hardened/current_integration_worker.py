#!/usr/bin/env python3
"""Fresh-process authentication and installation of the current RM hook.

This worker opens source code only.  It cannot receive a hook, module object,
payload path, packet path, source path, or model identifier from its caller.
"""

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
from authority import EXTERNAL_PINS, authenticate_current_external_root, require
from rm_order import TARGET_N, replacement_from_authenticated_reference


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
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sys.flags.isolated == 1 and sys.flags.dont_write_bytecode == 1,
            "worker requires python -I -B")
    require("PYTHONPATH" not in os.environ, "PYTHONPATH inherited")
    external = authenticate_current_external_root(args.external_root)
    root = Path(external["external_root"])

    # Base imports CuPy itself.  No module shell is accepted from the parent.
    require("agent_polaris_qwen_rht_encoder" not in sys.modules and
            "bg_codec_bec_encoder" not in sys.modules, "current modules preloaded")
    base = load_exact("agent_polaris_qwen_rht_encoder",
                      root / "agent_polaris_qwen_rht_encoder.py")
    bg = load_exact("bg_codec_bec_encoder", root / "bg_codec_bec_encoder.py")
    require(bg.base is base and base.reliability_freeze_flags is bg.bec_flags,
            "authenticated current reference hook identity")

    reference_hook = bg.bec_flags

    def rm_ordered_truncated_polar_flags(repo, n, capacities):
        require(n in TARGET_N, "global hook only N=2**20 or N=2**21")
        reference = reference_hook(repo, n, list(capacities))
        return replacement_from_authenticated_reference(reference, base.np)

    rm_ordered_truncated_polar_flags.__name__ = "rm_ordered_truncated_polar_flags"
    base.reliability_freeze_flags = rm_ordered_truncated_polar_flags
    require(base.reliability_freeze_flags is rm_ordered_truncated_polar_flags and
            reference_hook is bg.bec_flags, "final authenticated hook identity")

    record = {
        "schema": "strata-rm-global-swap-v1-current-integration-receipt",
        "external_pins": EXTERNAL_PINS,
        "base_module_file_sha256": hashlib.sha256(
            Path(base.__file__).read_bytes()).hexdigest(),
        "reference_module_file_sha256": hashlib.sha256(
            Path(bg.__file__).read_bytes()).hexdigest(),
        "reference_hook_identity": "bg_codec_bec_encoder.bec_flags",
        "installed_hook_identity":
            "agent_polaris_qwen_rht_encoder.reliability_freeze_flags",
        "final_hook_object_checked": True,
        "arbitrary_hook_parameter_accepted": False,
        "production_lengths": list(TARGET_N),
        "historical_base_cli_global_length_capable": False,
        "payload_launch_permitted": False,
        "independent_decoder_source_pinned": True,
        "fresh_interpreter": True,
        "python_isolated_flag": True,
        "pythonpath_inherited": False,
        "payloads_opened": 0,
        "rd_claim": False,
        "status": "PASS_AUTHENTICATED_CURRENT_HOOK_INSTALL__NO_PAYLOAD",
    }
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
