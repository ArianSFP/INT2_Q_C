"""Emit the deterministic, source-free COCHAIN-Q mechanism receipt to stdout."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cochain_q_oracle as c


def build_receipt() -> dict:
    out: dict = {
        "schema": "cochain_q.source_free_receipt.v0",
        "source_only": True,
        "qwen_or_model_data_accessed": False,
        "gpu_accessed": False,
        "fixtures": {},
    }
    for dimension in (2, 3):
        q = c.low_degree_even_ensemble(dimension)
        costs, energy = c.preference_costs(q)
        result = c.run_oracle(costs, energy, dimension)
        out["fixtures"][f"low_degree_d{dimension}"] = {
            "patterns": len(q),
            "baseline_relative_mse": result["baseline"]["relative_mse"],
            "ideal_gain_bpw": result["public_fiber"]["ideal_equivalent_gain_bpw"],
            "physical_gain_bpw": result["public_fiber"]["physical_equivalent_gain_bpw"],
            "status": result["status"],
            "fixed_reparameterization_saved_bits":
                result["fixed_label_reparameterization"]["saved_bits"],
            "packet_sha256": result["public_fiber"]["packet_sha256"],
            "max_pairwise_mutual_information_bits":
                max(c.pairwise_mutual_information(q).values()),
        }
    q = c.all_patterns(3)
    costs, energy = c.preference_costs(q, preferred_cost=1.0, alternate_cost=1001.0)
    result = c.run_oracle(costs, energy, 3)
    out["fixtures"]["balanced_iid_cube"] = {
        "patterns": len(q),
        "syndrome_counts": np.bincount(c.mixed_syndrome(q), minlength=2).tolist(),
        "physical_gain_bpw": result["public_fiber"]["physical_equivalent_gain_bpw"],
        "status": result["status"],
        "fixed_reparameterization_saved_bits":
            result["fixed_label_reparameterization"]["saved_bits"],
    }
    out["thresholds_bpw"] = {
        "hard_kill": c.HARD_KILL_BPW,
        "scientifically_real": c.SCIENTIFICALLY_REAL_BPW,
        "standalone_target": c.STANDALONE_TARGET_BPW,
        "engineering_margin": c.ENGINEERING_MARGIN_BPW,
    }
    out["conclusion"] = (
        "mechanism validation only; no Qwen result and no payload authority")
    return out


if __name__ == "__main__":
    print(json.dumps(build_receipt(), indent=2, sort_keys=True))
