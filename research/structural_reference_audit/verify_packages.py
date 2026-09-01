#!/usr/bin/env python3
"""Source-free audit of the two structural auxiliary early-kill packages.

This verifier parses only copied code, JSON results/manifests, and existing
JSON provenance evidence.  It never imports the experiment modules, NumPy, or
CuPy and never opens a model tensor payload.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PERMUTATION_FILES = {
    "README.md": "5ef5a36b22fcfa106bb2e43ae2a1b61ca929863cdf04a0bfd3892b77eb2d7391",
    "pair_result.json": "ba22a5ac76a6cc697f63899787ab85396a5b00dc2d764299473ecb59e3a52a52",
    "permutation_pair_screen.py": "dc9cc00dfbfb9e378291bb0abf67ff9d5c21d62a0f6c1f3c85df1b947d3dca00",
}

DENSE_FILES = {
    "README.md": "eebc010a1661bec033416ba87e55e0d56579c6eec148015365708985b68fb694",
    "dense_reference_manifest.json": "f535a729750432dcc2cf5c95c78b32aaa32701f5b99d4ab4f3e63114cef294f6",
    "dense_upcycle_pair_result.json": "36e7eb51f3eef51f88e6b08c562905c0e2797949b18327c35e2033363cd5db71",
    "dense_upcycle_pair_screen.py": "416457ccbd1aeafa3017a9feb4c65f8c7a187041c5d6ae230754e5c74279a7e9",
    "fetch_locked_dense_ranges.py": "c771242918ebfc985c52eb1d5495e3390ff4ae97a58c84d63a1458f5f1dadc5e",
}

EVIDENCE_FILES = {
    "research/breakthrough_redteam/bisco_protocol_freeze.json": "28c2bd6656f31ce7315601d0048d0b43759a7f2859142f745465e8fa0fe83164",
    "research/bisco_raw_mse_oracle/run_1/bisco_raw_mse_result.json": "5904e3887e69cf47ee4a882aeaacceb27823504c1e23eeff6adb4b3360874d92",
    "research/NEURON_PERMUTATION_ORACLE.md": "535e7efa94d37ef0b1d7aa855971b0eb274bf3f39618933ff8073d169f5204a3",
    "research/neuron_permutation_oracle.py": "5d2b91c1fa42b4f8793eaeb69acd3a1b1dd2fb65f8f98197f96d01e470c705dc",
    "research/neuron_permutation_oracle_result.json": "3bdff037e24fdd853e569419df8cc769c53d4be04f90c043b35403adbd66bfbd",
}

PINNED_IDENTITIES = {
    (5, 18),
    (12, 7),
    (18, 20),
    (28, 83),
    (36, 76),
    (45, 41),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_unsigned_sha256(document: dict[str, Any]) -> tuple[str, str]:
    unsigned = dict(document)
    claimed = str(unsigned.pop("canonical_unsigned_sha256"))
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return claimed, hashlib.sha256(encoded).hexdigest()


def all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return True


def close(left: float, right: float, *, absolute: float = 2e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=2e-12, abs_tol=absolute)


def energy_row_consistent(row: dict[str, Any]) -> bool:
    source = float(row["source_energy"])
    residual = float(row["residual_energy"])
    roles = row["roles"]
    role_source = sum(float(role["source_energy"]) for role in roles.values())
    role_residual = sum(float(role["residual_energy"]) for role in roles.values())
    return all(
        (
            close(float(row["capture"]), 1.0 - residual / source),
            close(float(row["residual_fraction"]), residual / source),
            close(source, role_source, absolute=2e-9),
            close(residual, role_residual, absolute=2e-9),
            all(
                close(
                    float(role["capture"]),
                    1.0 - float(role["residual_energy"]) / float(role["source_energy"]),
                )
                for role in roles.values()
            ),
        )
    )


def declared_inventory(directory: Path, expected: dict[str, str]) -> tuple[bool, dict[str, Any]]:
    actual: dict[str, Any] = {}
    safe = directory.is_dir() and not directory.is_symlink()
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            actual[relative] = {"kind": "symlink"}
            continue
        if path.is_dir():
            actual[relative] = {"kind": "directory"}
            continue
        if not path.is_file():
            actual[relative] = {"kind": "nonregular"}
            continue
        actual[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    hashes_match = all(
        name in actual
        and actual[name].get("sha256") == expected_hash
        and (directory / name).is_file()
        and not (directory / name).is_symlink()
        for name, expected_hash in expected.items()
    )
    return safe and hashes_match, actual


def require_source_fragments(source: str, fragments: list[str]) -> bool:
    return all(fragment in source for fragment in fragments)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    permutation_dir = repo / "research/permutation_aligned_expert_template"
    dense_dir = repo / "research/dense_upcycle_reference"

    checks: dict[str, bool] = {}

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    permutation_inventory_ok, permutation_inventory = declared_inventory(
        permutation_dir, PERMUTATION_FILES
    )
    dense_inventory_ok, dense_inventory = declared_inventory(dense_dir, DENSE_FILES)
    check("permutation_declared_artifact_hashes", permutation_inventory_ok)
    check("dense_declared_artifact_hashes", dense_inventory_ok)
    evidence_hashes = {
        relative: sha256(repo / relative) if (repo / relative).is_file() else None
        for relative in EVIDENCE_FILES
    }
    check("provenance_evidence_hashes", evidence_hashes == EVIDENCE_FILES)

    permutation_result = load_json(permutation_dir / "pair_result.json")
    dense_result = load_json(dense_dir / "dense_upcycle_pair_result.json")
    dense_manifest = load_json(dense_dir / "dense_reference_manifest.json")
    freeze = load_json(repo / "research/breakthrough_redteam/bisco_protocol_freeze.json")
    bisco = load_json(repo / "research/bisco_raw_mse_oracle/run_1/bisco_raw_mse_result.json")

    for label, document in (
        ("permutation_result", permutation_result),
        ("dense_result", dense_result),
        ("dense_manifest", dense_manifest),
    ):
        claimed, computed = canonical_unsigned_sha256(document)
        check(f"{label}_canonical_seal", claimed == computed)
        check(f"{label}_all_finite", all_finite(document))

    declared_pinned = {
        tuple(map(int, item.replace("layer", "").replace("expert", "").split("_")))
        for item in freeze["data_firewall"]["pinned_panel_identities"]
    }
    check("pinned_identity_evidence", declared_pinned == PINNED_IDENTITIES)
    check(
        "auxiliary_identity_evidence",
        int(freeze["data_firewall"]["auxiliary_layer"]) == 15
        and {0, 8}.issubset(set(map(int, freeze["data_firewall"]["auxiliary_experts"]))),
    )
    authenticated_aux_hashes = bisco["data_firewall"]["source_sha256"]

    permutation_identity = permutation_result["source_identity"]
    permutation_pairs = {
        (int(permutation_identity["layer"]), int(permutation_identity["reference_expert"])),
        (int(permutation_identity["layer"]), int(permutation_identity["target_expert"])),
    }
    check("permutation_sources_non_pinned", permutation_pairs.isdisjoint(PINNED_IDENTITIES))
    check(
        "permutation_source_hashes_authenticated_auxiliary",
        all(authenticated_aux_hashes.get(name) == digest for name, digest in permutation_identity["hashes"].items()),
    )
    check("permutation_result_records_pinned_closed", permutation_identity["pinned_panel_opened"] is False)

    dense_identity = dense_result["target"]
    dense_pair = (int(dense_identity["layer"]), int(dense_identity["expert"]))
    check("dense_target_non_pinned", dense_pair not in PINNED_IDENTITIES)
    check(
        "dense_target_hashes_authenticated_auxiliary",
        all(authenticated_aux_hashes.get(name) == digest for name, digest in dense_identity["hashes"].items()),
    )
    check("dense_result_records_pinned_closed", dense_identity["pinned_panel_opened"] is False)

    required = 1.0 - 0.8 / 0.936397621
    permutation_rows = [permutation_result["source"], *permutation_result["controls"]]
    check("permutation_energy_arithmetic", all(energy_row_consistent(row) for row in permutation_rows))
    permutation_max_control = max(float(row["capture"]) for row in permutation_result["controls"])
    permutation_corrected = float(permutation_result["source"]["capture"]) - permutation_max_control
    check("permutation_control_max", close(permutation_result["max_control_capture"], permutation_max_control))
    check("permutation_corrected_capture", close(permutation_result["control_corrected_capture"], permutation_corrected))
    check("permutation_required_capture", close(permutation_result["required_incremental_capture_over_existing_composite"], required))
    check(
        "permutation_decision_arithmetic",
        permutation_corrected < required
        and permutation_result["decision"] == "EARLY_KILL_PERMUTATION_ALIGNED_SINGLE_TEMPLATE",
    )

    manifest_claimed, manifest_computed = canonical_unsigned_sha256(dense_manifest)
    check("dense_result_binds_manifest_file", dense_result["reference"]["manifest_sha256"] == sha256(dense_dir / "dense_reference_manifest.json"))
    check("dense_manifest_seal_redundant_replay", manifest_claimed == manifest_computed)
    data_start = 8 + int(dense_manifest["safetensors_header_bytes"])
    manifest_tensors_ok = True
    manifest_hashes: dict[str, str] = {}
    for tensor in dense_manifest["tensors"]:
        relative_start, relative_stop = map(int, tensor["relative_data_offsets"])
        absolute_start, absolute_stop = map(int, tensor["absolute_http_range"])
        manifest_tensors_ok &= int(tensor["bytes"]) == relative_stop - relative_start
        manifest_tensors_ok &= absolute_start == data_start + relative_start
        manifest_tensors_ok &= absolute_stop == data_start + relative_stop - 1
        manifest_tensors_ok &= tensor["content_range"] == f"bytes {absolute_start}-{absolute_stop}/{dense_manifest['checkpoint_total_bytes']}"
        manifest_hashes[str(tensor["output"])] = str(tensor["sha256"])
    check("dense_manifest_ranges_and_lengths", manifest_tensors_ok)
    check("dense_result_reference_hashes_match_manifest", dense_result["reference"]["tensor_hashes"] == manifest_hashes)
    dense_rows = [*dense_result["source_candidates"], *dense_result["scramble_controls"]]
    check("dense_energy_arithmetic", all(energy_row_consistent(row) for row in dense_rows))
    dense_best = max(dense_result["source_candidates"], key=lambda row: float(row["capture"]))
    dense_max_control = max(float(row["capture"]) for row in dense_result["scramble_controls"])
    dense_corrected = float(dense_best["capture"]) - dense_max_control
    check("dense_best_source", dense_result["best_source"] == dense_best)
    check("dense_control_max", close(dense_result["max_control_capture"], dense_max_control))
    check("dense_corrected_capture", close(dense_result["control_corrected_capture"], dense_corrected))
    check("dense_required_capture", close(dense_result["required_incremental_capture"], required))
    check(
        "dense_decision_arithmetic",
        dense_corrected < required
        and dense_result["decision"] == "EARLY_KILL_DENSE_UPCYCLE_REFERENCE",
    )

    permutation_source = (permutation_dir / "permutation_pair_screen.py").read_text(encoding="utf-8")
    dense_source = (dense_dir / "dense_upcycle_pair_screen.py").read_text(encoding="utf-8")
    fetch_source = (dense_dir / "fetch_locked_dense_ranges.py").read_text(encoding="utf-8")
    for label, source in (
        ("permutation", permutation_source),
        ("dense", dense_source),
        ("fetch", fetch_source),
    ):
        try:
            ast.parse(source)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False
        check(f"{label}_code_parses", syntax_ok)

    check(
        "dense_assignment_uses_explained_sse",
        require_source_fragments(
            dense_source,
            [
                "explained = dots.astype(cp.float64) ** 2 / reference_energy[:, None]",
                "score = explained if score is None else score + explained",
                "linear_sum_assignment(-cp.asnumpy(score.T))",
                "alpha = dots[ref_index_gpu, cp.arange(ROWS)].astype(cp.float64) / reference_energy[",
            ],
        ),
    )
    check(
        "dense_control_is_coordinate_scramble",
        require_source_fragments(
            dense_source,
            ["permutation = rng.permutation(COLS)", "signs = rng.integers(0, 2", "max_control = max("],
        ),
    )
    check(
        "permutation_assignment_is_unweighted_cosine_squared",
        require_source_fragments(
            permutation_source,
            [
                "normalized = dots.astype(cp.float64) / (ref_norm[:, None] * target_norm[None, :])",
                "score = cp.asnumpy(normalized * normalized)",
                "linear_sum_assignment(-score)",
            ],
        ),
    )
    check(
        "permutation_regression_is_one_joint_scalar",
        require_source_fragments(
            permutation_source,
            [
                "cp.concatenate((cp.asarray(target_up), cp.asarray(target_down)), axis=1)",
                "alpha = dot_selected / ref_energy",
                "reconstruction = alpha[:, None] * ref_selected",
            ],
        )
        and "reference_gate" not in permutation_source
        and "alpha_up" not in permutation_source
        and "alpha_down" not in permutation_source,
    )
    check(
        "permutation_controls_match_only_global_raw_energy",
        require_source_fragments(
            permutation_source,
            [
                "value -= np.mean(value, dtype=np.float64)",
                "float(np.sum(source.astype(np.float64) ** 2))",
                "return value.astype(np.float32)",
            ],
        ),
    )

    # A source-free algebraic counterexample to the permutation assignment.
    # score[reference][target] is cosine^2 and target energies are [100, 1].
    score = [[0.6, 0.9], [0.5, 0.0]]
    target_energy = [100.0, 1.0]
    identity_cosine = score[0][0] + score[1][1]
    swap_cosine = score[0][1] + score[1][0]
    identity_capture = score[0][0] * target_energy[0] + score[1][1] * target_energy[1]
    swap_capture = score[0][1] * target_energy[1] + score[1][0] * target_energy[0]
    assignment_counterexample = {
        "unweighted_cosine_prefers_swap": swap_cosine > identity_cosine,
        "raw_sse_capture_prefers_identity": identity_capture > swap_capture,
        "identity_unweighted_cosine": identity_cosine,
        "swap_unweighted_cosine": swap_cosine,
        "identity_explained_sse": identity_capture,
        "swap_explained_sse": swap_capture,
    }
    check("permutation_objective_mismatch_counterexample", all((assignment_counterexample["unweighted_cosine_prefers_swap"], assignment_counterexample["raw_sse_capture_prefers_identity"])))

    artifact_checks_all_pass = all(checks.values())
    output = {
        "schema": "structural_reference_package_audit_verifier_v1",
        "artifact_checks_all_pass": artifact_checks_all_pass,
        "checks": checks,
        "inventories": {
            "permutation_aligned_expert_template": permutation_inventory,
            "dense_upcycle_reference": dense_inventory,
            "provenance_evidence_hashes": evidence_hashes,
            "unsealed_package_extras": {
                "permutation_aligned_expert_template": sorted(
                    set(permutation_inventory) - set(PERMUTATION_FILES)
                ),
                "dense_upcycle_reference": sorted(
                    set(dense_inventory) - set(DENSE_FILES)
                ),
            },
        },
        "arithmetic": {
            "required_incremental_capture": required,
            "permutation": {
                "raw_capture": float(permutation_result["source"]["capture"]),
                "max_control_capture": permutation_max_control,
                "control_corrected_capture": permutation_corrected,
                "raw_shortfall_factor": required / float(permutation_result["source"]["capture"]),
                "corrected_shortfall_factor": required / permutation_corrected,
            },
            "dense": {
                "raw_capture": float(dense_best["capture"]),
                "max_control_capture": dense_max_control,
                "control_corrected_capture": dense_corrected,
                "raw_shortfall_factor": required / float(dense_best["capture"]),
                "corrected_shortfall_factor": required / dense_corrected,
            },
        },
        "formula_audit": {
            "dense": {
                "assignment": "Hungarian maximizes the sum of per-role explained SSE for the implemented centered affine regressions.",
                "numerical_caveat": "Dots are accumulated by FP32 CuPy GEMM before FP64 division; centered rows subtract FP32-rounded means, while the recorded beta hash uses FP64 means. This is not literally exact FP64 affine fitting, but the discrepancy is numerical and does not approach the 28.4x raw opportunity gap.",
                "control_caveat": "Up and Down receive independent signed coordinate scrambles, so cross-role target structure is not preserved. The raw, no-control result is already 28.4x short.",
            },
            "permutation": {
                "assignment_blocker": "Hungarian maximizes unweighted cosine^2. Raw-SSE reduction after LS is target_energy * cosine^2, so the recorded assignment is not generally capture-optimal.",
                "assignment_counterexample": assignment_counterexample,
                "regression_blocker": "Only one through-origin scalar is fitted jointly to concatenated Up/Down; separate role scalars/intercepts and Gate are absent.",
                "control_caveat": "Each Gaussian array is zero-mean and matched only to global raw energy before a final FP32 cast; source means, row-energy distribution, and uncertainty beyond two controls are not matched.",
            },
        },
        "verdicts": {
            "dense_upcycle_reference": {
                "artifact_integrity": "PASS",
                "scoped_early_kill": "JUSTIFIED_AS_A_FAVORABLE_TWO_LAYER_AFFINE_NEURON_ANCESTOR_GATE",
                "claim_boundary": "Does not reject unsearched dense layers, nonlinear/multi-neuron references, or other training lineages.",
            },
            "permutation_aligned_expert_template": {
                "artifact_integrity": "PASS",
                "empirical_signal": "STRONG_NEGATIVE_EVIDENCE_FOR_THIS_ONE_PAIR_JOINT_SCALAR_CELL",
                "certified_favorable_upper_screen": "BLOCKED_BY_ASSIGNMENT_OBJECTIVE_AND_MODEL_FAMILY",
                "broader_early_kill": "NOT_JUSTIFIED_BY_THIS_PACKAGE_ALONE",
            },
        },
        "access_boundary": {
            "model_payload_opened": False,
            "cuda_imported_or_initialized": False,
            "experiment_files_mutated": False,
            "parsed_inputs": "copied code, JSON results/manifests, and existing provenance result/manifest files only",
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0 if artifact_checks_all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
