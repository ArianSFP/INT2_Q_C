#!/usr/bin/env python3
"""Read-only diagnostic for cross-runtime polar-normal replay drift."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np


def main() -> None:
    repo = Path("/workspace/INT2__compression/INT2_Q_C")
    source_root = Path("/workspace/INT2__compression")
    parent_path = repo / "research/polar_normal_predictor/polar_normal_predictor.py"
    parent = runpy.run_path(str(parent_path))
    _, lock, _, matrices = parent["load_sources"](source_root)
    _, selections, _ = parent["load_base_selections"](
        repo / "research/composite_superoracle/result.json"
    )
    prior = json.loads(
        (repo / "research/polar_normal_predictor/result.json").read_text(encoding="utf-8")
    )["source_normal_records"]

    for ordinal, (matrix, metadata, selection, old) in enumerate(
        zip(matrices, lock["matrices"], selections, prior, strict=True)
    ):
        current = parent["normal_record"](matrix, metadata, int(selection["rank"]))
        normal_bytes = np.ascontiguousarray(current.normal, dtype="<f8").tobytes()
        normal_sha = hashlib.sha256(normal_bytes).hexdigest()
        fields = ("source_energy", "model_energy", "normal_energy", "common_scale")
        values = {name: float(getattr(current, name)) for name in fields}
        differences = {name: values[name] - float(old[name]) for name in fields}
        ulps = {
            name: differences[name] / float(np.spacing(float(old[name])))
            for name in fields
        }
        print(
            json.dumps(
                {
                    "matrix_ordinal": ordinal,
                    "current": values,
                    "prior": {name: float(old[name]) for name in fields},
                    "differences": differences,
                    "signed_ulps": ulps,
                    "normal_sha256_current": normal_sha,
                    "normal_sha256_prior": old["normal_sha256_f64"],
                    "normal_bytes_exact": normal_sha == old["normal_sha256_f64"],
                },
                sort_keys=True,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
