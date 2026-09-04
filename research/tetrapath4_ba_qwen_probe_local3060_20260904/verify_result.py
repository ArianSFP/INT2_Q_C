"""Fail-closed verifier for the final TETRAPATH-BA exploratory result."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULT = HERE / "RESULT_STAGE3_2PLUS2_ALL8.json"
EXPECTED_RESULT = "3b68c4ee7115bfb8d5f6b6e8027a2bb27c5c0f6d358647b4c4394182e7158353"
EXPECTED_RUNNER = "b9745f27ae3ec005e64be4287362cbafa328d885c5672404ee39d9c5145fa711"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


report = json.loads(RESULT.read_bytes())
need(digest(RESULT) == EXPECTED_RESULT, "result hash")
need(digest(HERE / "run_ba_probe.py") == EXPECTED_RUNNER, "runner hash")
need(report["status"] == "HARD_KILL_APERTURE_IRREDUCIBLE_FOURWAY_BELOW_0P045_BPW",
     "status")
need(report["runtime"]["device_uuid"] ==
     "GPU-458a424a-76e3-65e5-0470-803e0ed131ca", "local device")
need(len(report["pairs"]) == 8 and report["aperture"]["block_stride"] == 64,
     "aperture")
irreducible = [float(row["control_corrected_irreducible_gain_bpw"])
               for row in report["pairs"]]
corrected = [float(row["control_corrected_gain_bpw"]) for row in report["pairs"]]
need(all(math.isfinite(x) for x in irreducible + corrected), "finite metrics")
need(max(irreducible) < 0.045, "hard-kill threshold")
need(abs(max(irreducible) -
         report["aggregate"]["maximum_control_corrected_irreducible_gain_bpw"]) < 1e-15,
     "maximum recomputation")
need(abs(sum(irreducible) / len(irreducible) -
         report["aggregate"]["mean_control_corrected_irreducible_gain_bpw"]) < 1e-15,
     "mean recomputation")
print("PASS_TETRAPATH_BA_QWEN_APERTURE_HARD_KILL_IRREDUCIBLE_FOURWAY")
