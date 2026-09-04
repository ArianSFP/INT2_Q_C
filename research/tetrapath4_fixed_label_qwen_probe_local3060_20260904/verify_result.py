"""Verify the bounded fixed-label Qwen TETRAPATH result."""

import hashlib
import json
import math
from pathlib import Path

here = Path(__file__).resolve().parent
path = here / "RESULT.json"
expected = "2dab5c4175149d92f46409724bc2d204e05452e5072efcbb23cffad9a1f19418"
if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
    raise RuntimeError("result hash")
result = json.loads(path.read_bytes())
if result["status"] != "FIXED_LABEL_MEMORYLESS_FOURWAY_HARD_KILL_BELOW_0P045_BPW":
    raise RuntimeError("status")
if result["runtime"]["device_uuid"] != "GPU-458a424a-76e3-65e5-0470-803e0ed131ca":
    raise RuntimeError("device")
if len(result["pairs"]) != 8:
    raise RuntimeError("pair count")
gain = [float(row["source"]["fourway_gain_over_best_factorized_bpw"])
        for row in result["pairs"]]
if not all(math.isfinite(x) for x in gain) or max(gain) >= 0.045:
    raise RuntimeError("hard-kill metric")
print("PASS_FIXED_LABEL_TETRAPATH_QWEN_HARD_KILL")
