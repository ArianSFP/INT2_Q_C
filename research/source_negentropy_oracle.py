#!/usr/bin/env python3
"""Cross-fitted source-negentropy screen for the pinned expert-affine panel.

The signed RHT makes each scalar marginal nearly Gaussian, but an orthogonal
transform cannot erase joint negentropy.  This probe asks whether enough
non-Gaussianity exists *before* the RHT to justify a matched nonlinear/vector
source code.  It is deliberately an oracle screen, not a codec result.

Every BF16 staging byte and the sealed plan are authenticated.  Values are
first divided by their transmitted block RMS.  Known route role and STRATA
label may then select a probability table.  Shape gain is measured against a
Gaussian having the same class mean and variance, so scale allocation is not
double-counted.  Six leave-one-expert-out folds prevent a table from merely
memorising the expert it scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_expert_local_codec import common


INNER_EDGES = 1_023
EDGE_SIGMA = 8.0
PSEUDOCOUNT = 0.5
REQUIRED_GAIN_BPW = -0.5 * math.log2(0.8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_values(path: Path) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def class_ids(group_ordinals: np.ndarray, labels: np.ndarray, mode: str) -> np.ndarray:
    roles = (group_ordinals // 768) % 3
    strata = labels[group_ordinals].astype(np.int64)
    if mode == "global":
        return np.zeros(group_ordinals.size, dtype=np.int64)
    if mode == "role":
        return roles.astype(np.int64)
    if mode == "label":
        return strata
    if mode == "role_label":
        return 8 * roles.astype(np.int64) + strata
    raise ValueError(mode)


def class_count(mode: str) -> int:
    return {"global": 1, "role": 3, "label": 8, "role_label": 24}[mode]


def add_moments(
    counts: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    experts: np.ndarray,
    classes: np.ndarray,
    groups: np.ndarray,
) -> None:
    for expert in np.unique(experts):
        owner = experts == expert
        owner_classes = classes[owner]
        owner_values = groups[owner].reshape(-1)
        expanded = np.repeat(owner_classes, common.GROUP_VALUES)
        bins = int(expert) * counts.shape[1] + expanded
        shape = counts.size
        counts += np.bincount(bins, minlength=shape).reshape(counts.shape)
        sums += np.bincount(
            bins, weights=owner_values, minlength=shape
        ).reshape(sums.shape)
        sums2 += np.bincount(
            bins, weights=owner_values.astype(np.float64) ** 2, minlength=shape
        ).reshape(sums2.shape)


def table_crossfit(
    hist: np.ndarray,
    edges: np.ndarray,
    class_variances: np.ndarray,
    total_values: int,
) -> dict[str, Any]:
    experts, classes, bins = hist.shape
    folds = []
    weighted_gain = 0.0
    weighted_count = 0
    for heldout in range(experts):
        train = np.sum(np.delete(hist, heldout, axis=0), axis=0, dtype=np.int64)
        test = hist[heldout]
        fold_gain_bits = 0.0
        fold_count = int(test.sum())
        class_rows = []
        for cls in range(classes):
            test_row = test[cls]
            n = int(test_row.sum())
            if n == 0:
                continue
            train_row = train[cls]
            model = (train_row.astype(np.float64) + PSEUDOCOUNT) / (
                int(train_row.sum()) + PSEUDOCOUNT * bins
            )
            lower = edges[:-1]
            upper = edges[1:]
            gaussian = np.empty_like(lower)
            negative = upper <= 0.0
            positive = lower >= 0.0
            middle = ~(negative | positive)
            gaussian[negative] = ndtr(upper[negative]) - ndtr(lower[negative])
            # In the positive tail, subtract survival functions.  Direct CDF
            # subtraction loses all precision well before the final 8-sigma
            # bins used by this audit.
            gaussian[positive] = ndtr(-lower[positive]) - ndtr(-upper[positive])
            gaussian[middle] = ndtr(upper[middle]) - ndtr(lower[middle])
            if np.any(gaussian <= 0.0):
                raise AssertionError("non-positive Gaussian histogram probability")
            gaussian /= gaussian.sum()
            model_nll = float(-np.dot(test_row, np.log2(model)) / n)
            gaussian_nll = float(-np.dot(test_row, np.log2(gaussian)) / n)
            gain = gaussian_nll - model_nll
            fold_gain_bits += n * gain
            class_rows.append(
                {
                    "class": cls,
                    "test_values": n,
                    "train_values": int(train_row.sum()),
                    "gaussian_discrete_nll_bpw": gaussian_nll,
                    "crossfit_table_nll_bpw": model_nll,
                    "gain_bpw": gain,
                    "global_class_variance_before_standardisation": float(
                        class_variances[cls]
                    ),
                }
            )
        fold_gain = fold_gain_bits / fold_count
        folds.append(
            {
                "heldout_expert": heldout,
                "test_values": fold_count,
                "gross_shape_gain_bpw": fold_gain,
                "classes": class_rows,
            }
        )
        weighted_gain += fold_gain_bits
        weighted_count += fold_count
    gross = weighted_gain / weighted_count
    # One uint16 frequency per bin and class, charged once over the full panel.
    table_bits = classes * bins * 16
    table_cost = table_bits / total_values
    net = gross - table_cost
    return {
        "folds": folds,
        "gross_crossfit_shape_gain_bpw": gross,
        "probability_table_bits": table_bits,
        "probability_table_cost_bpw": table_cost,
        "net_shape_gain_bpw": net,
        "required_gain_bpw": REQUIRED_GAIN_BPW,
        "fraction_of_required_gain": net / REQUIRED_GAIN_BPW,
        "passes_early_screen": bool(net >= REQUIRED_GAIN_BPW),
    }


def evaluate_mode(
    mode: str,
    block_cache: list[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
) -> dict[str, Any]:
    classes = class_count(mode)
    counts = np.zeros((common.EXPERTS, classes), dtype=np.int64)
    sums = np.zeros_like(counts, dtype=np.float64)
    sums2 = np.zeros_like(counts, dtype=np.float64)

    for ordinals, values in block_cache:
        groups = values.reshape(-1, common.GROUP_VALUES)
        owners = ordinals // 2_304
        ids = class_ids(ordinals, labels, mode)
        add_moments(counts, sums, sums2, owners, ids, groups)

    aggregate_count = np.sum(counts, axis=0)
    means = np.sum(sums, axis=0) / aggregate_count
    second = np.sum(sums2, axis=0) / aggregate_count
    variances = second - means * means
    if np.any(variances <= 0.0):
        raise AssertionError(f"non-positive class variance in {mode}")

    edges = np.concatenate(
        (
            np.asarray([-np.inf]),
            np.linspace(-EDGE_SIGMA, EDGE_SIGMA, INNER_EDGES, dtype=np.float64),
            np.asarray([np.inf]),
        )
    )
    bins = edges.size - 1
    hist = np.zeros((common.EXPERTS, classes, bins), dtype=np.int64)
    fourth = np.zeros(classes, dtype=np.float64)

    for ordinals, values in block_cache:
        groups = values.reshape(-1, common.GROUP_VALUES)
        owners = ordinals // 2_304
        ids = class_ids(ordinals, labels, mode)
        for expert in np.unique(owners):
            owner = owners == expert
            owner_ids = ids[owner]
            owner_groups = groups[owner]
            for cls in np.unique(owner_ids):
                selected = owner_groups[owner_ids == cls].reshape(-1).astype(np.float64)
                z = (selected - means[cls]) / math.sqrt(variances[cls])
                hist[int(expert), int(cls)] += np.histogram(z, bins=edges)[0]
                fourth[int(cls)] += float(np.sum(z**4, dtype=np.float64))

    if int(hist.sum()) != common.WEIGHTS:
        raise AssertionError("histogram coverage mismatch")
    kurtosis = fourth / aggregate_count
    result = table_crossfit(hist, edges, variances, common.WEIGHTS)
    result.update(
        {
            "mode": mode,
            "class_count": classes,
            "bins": bins,
            "edge_sigma": EDGE_SIGMA,
            "pseudocount": PSEUDOCOUNT,
            "class_counts": aggregate_count.astype(int).tolist(),
            "class_means_after_block_rms": means.tolist(),
            "class_variances_after_block_rms": variances.tolist(),
            "class_excess_kurtosis": (kurtosis - 3.0).tolist(),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.plan_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace evidence: {output}")
    plan_path = root / "plan.lock.json"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if common.sealed({k: v for k, v in plan.items() if k != "lock_sha256"})[
        "lock_sha256"
    ] != plan.get("lock_sha256"):
        raise ValueError("plan self-seal mismatch")

    label_path = root / str(plan["assets"]["labels_3bit.bin"]["relpath"])
    if sha256_file(label_path) != plan["assets"]["labels_3bit.bin"]["sha256"]:
        raise ValueError("label hash mismatch")
    labels = common.unpack_labels(label_path.read_bytes()).astype(np.int64)
    expected_ordinals = common.expected_block_group_ordinals(labels)

    cache: list[tuple[np.ndarray, np.ndarray]] = []
    block_rows = []
    for ordinal, (row, ordinals) in enumerate(
        zip(plan["blocks"], expected_ordinals, strict=True)
    ):
        path = root / str(row["staging_relpath"])
        if sha256_file(path) != row["staging_sha256"]:
            raise ValueError(f"staging hash mismatch block {ordinal}")
        values = bf16_values(path)
        if values.size != int(row["values"]):
            raise ValueError(f"staging geometry mismatch block {ordinal}")
        rms = math.sqrt(float(np.mean(values.astype(np.float64) ** 2)))
        normalised = np.asarray(values / np.float32(rms), dtype=np.float32)
        cache.append((ordinals, normalised))
        block_rows.append(
            {
                "block_ordinal": ordinal,
                "staging_sha256": row["staging_sha256"],
                "values": int(values.size),
                "rms": rms,
                "normalised_mean": float(np.mean(normalised, dtype=np.float64)),
                "normalised_variance": float(np.var(normalised, dtype=np.float64)),
            }
        )

    modes = [
        evaluate_mode(mode, cache, labels)
        for mode in ("global", "role", "label", "role_label")
    ]
    best = max(modes, key=lambda row: row["net_shape_gain_bpw"])
    report = {
        "schema": "strata_source_negentropy_oracle_v1",
        "claim_boundary": (
            "Cross-fitted marginal-shape upper-bound screen only; not an achieved "
            "lossy-code rate or MSE result."
        ),
        "architecture_hypothesis": (
            "Replace RHT Gaussianisation with a class-matched nonlinear/vector "
            "source code so pre-RHT negentropy remains accessible."
        ),
        "plan": {
            "path": str(plan_path),
            "file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "lock_sha256": plan["lock_sha256"],
        },
        "block_rms_normalisation": block_rows,
        "modes": modes,
        "best_mode": best["mode"],
        "best_net_shape_gain_bpw": best["net_shape_gain_bpw"],
        "required_gain_bpw": REQUIRED_GAIN_BPW,
        "decision": "continue" if best["passes_early_screen"] else "hard_kill",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    common.write_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "best_mode": report["best_mode"],
                "best_net_shape_gain_bpw": report["best_net_shape_gain_bpw"],
                "required_gain_bpw": REQUIRED_GAIN_BPW,
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
