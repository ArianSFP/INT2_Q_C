"""Independent fail-closed verifier for the Tier-A initialization-anchor gate.

The verifier never imports torch or CuPy and never opens the one excluded
weight payload.  With ``--aux-dir`` it independently hashes every eligible
auxiliary payload after checking the exact directory shape.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import common


def _reject_constant(value: str) -> None:
    raise common.ProtocolError(f"non-finite JSON numeric literal: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise common.ProtocolError("result must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise common.ProtocolError("result must contain one JSON object")
    return value


def _close(left: float, right: float, label: str, *, rtol: float = 2e-9, atol: float = 2e-11) -> None:
    if not (math.isfinite(float(left)) and math.isfinite(float(right))):
        raise common.ProtocolError(f"{label} is non-finite")
    if not math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol):
        raise common.ProtocolError(f"{label} mismatch: {left!r} != {right!r}")


def _equal_tree(observed: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            raise common.ProtocolError(f"{label} is not an object")
        common.strict_keys(observed, expected.keys(), label)
        for key in expected:
            _equal_tree(observed[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise common.ProtocolError(f"{label} list shape mismatch")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _equal_tree(left, right, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected:
            raise common.ProtocolError(f"{label} mismatch")
        return
    if isinstance(expected, int):
        if isinstance(observed, bool) or int(observed) != expected or float(observed) != expected:
            raise common.ProtocolError(f"{label} integer mismatch")
        return
    if isinstance(expected, float):
        _close(float(observed), expected, label)
        return
    raise common.ProtocolError(f"unsupported verifier tree type at {label}")


def _verify_metric(metric: Mapping[str, Any], label: str) -> None:
    common.strict_keys(metric, ("sse", "baseline_sse", "q", "capture"), label)
    expected = common.metric_from_sse(float(metric["sse"]), float(metric["baseline_sse"]))
    _close(metric["q"], expected["q"], f"{label}.q")
    _close(metric["capture"], expected["capture"], f"{label}.capture")


def _verify_detail_row(row: Mapping[str, Any], expected_source: common.SourceRow, label: str) -> None:
    common.strict_keys(row, ("tensor_name", "expert", "role", "fit", "score"), label)
    if row["tensor_name"] != expected_source.tensor_name:
        raise common.ProtocolError(f"{label}.tensor_name mismatch")
    if int(row["expert"]) != expected_source.expert or row["role"] != expected_source.role:
        raise common.ProtocolError(f"{label} identity mismatch")
    fit = row["fit"]
    score = row["score"]
    fit_keys = ("n", "sum_w", "sum_g", "sum_w2", "sum_g2", "sum_wg", "alpha", "mu", "fit_mean_w")
    score_keys = ("n", "sum_w", "sum_g", "sum_w2", "sum_g2", "sum_wg", "sse", "baseline_sse", "rho")
    common.strict_keys(fit, fit_keys, f"{label}.fit")
    common.strict_keys(score, score_keys, f"{label}.score")
    n_fit = int(fit["n"])
    n_score = int(score["n"])
    if n_fit <= 0 or n_score <= 0:
        raise common.ProtocolError(f"{label} has empty split")
    for section in (fit, score):
        for key, value in section.items():
            if key != "n" and not math.isfinite(float(value)):
                raise common.ProtocolError(f"{label}.{key} is non-finite")

    sum_w = float(fit["sum_w"])
    sum_g = float(fit["sum_g"])
    cgg = float(fit["sum_g2"]) - sum_g * sum_g / n_fit
    cwg = float(fit["sum_wg"]) - sum_w * sum_g / n_fit
    alpha = cwg / cgg if cgg > 0.0 else 0.0
    mean_w = sum_w / n_fit
    mu = mean_w - alpha * (sum_g / n_fit)
    _close(fit["alpha"], alpha, f"{label}.fit.alpha", rtol=5e-10, atol=5e-12)
    _close(fit["fit_mean_w"], mean_w, f"{label}.fit.fit_mean_w")
    _close(fit["mu"], mu, f"{label}.fit.mu")

    sw = float(score["sum_w"])
    sg = float(score["sum_g"])
    sw2 = float(score["sum_w2"])
    sg2 = float(score["sum_g2"])
    swg = float(score["sum_wg"])
    expected_sse = (
        sw2
        + n_score * mu * mu
        + alpha * alpha * sg2
        + 2.0 * mu * alpha * sg
        - 2.0 * mu * sw
        - 2.0 * alpha * swg
    )
    expected_baseline = sw2 - 2.0 * mean_w * sw + n_score * mean_w * mean_w
    _close(score["sse"], expected_sse, f"{label}.score.sse", rtol=2e-8, atol=2e-10)
    _close(score["baseline_sse"], expected_baseline, f"{label}.score.baseline_sse", rtol=2e-8, atol=2e-10)
    cww = sw2 - sw * sw / n_score
    cgg_score = sg2 - sg * sg / n_score
    cwg_score = swg - sw * sg / n_score
    rho = cwg_score / math.sqrt(cww * cgg_score) if cww > 0.0 and cgg_score > 0.0 else 0.0
    _close(score["rho"], rho, f"{label}.score.rho", rtol=2e-8, atol=2e-10)


def _verify_details(
    details: Mapping[str, Any], expected_rows: Sequence[common.SourceRow], label: str
) -> None:
    common.strict_keys(details, ("source", "gaussian", "permuted"), label)
    for domain in ("source", "gaussian", "permuted"):
        rows = details[domain]
        if not isinstance(rows, list) or len(rows) != len(expected_rows):
            raise common.ProtocolError(f"{label}.{domain} row count mismatch")
        for index, (row, source) in enumerate(zip(rows, expected_rows)):
            _verify_detail_row(row, source, f"{label}.{domain}[{index}]")


def _verify_parity(parity: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    required = (
        "all_required_checks_passed", "torch_version", "cupy_version", "cuda_runtime_version",
        "cuda_driver_version", "device_index", "device_name", "multi_processor_count",
        "max_threads_per_multi_processor", "dtype_paths",
    )
    common.strict_keys(parity, required, "backend.parity")
    if parity["all_required_checks_passed"] is not True or int(parity["device_index"]) != 0:
        raise common.ProtocolError("production CUDA parity did not pass on device zero")
    sm = int(parity["multi_processor_count"])
    threads = int(parity["max_threads_per_multi_processor"])
    if sm <= 0 or threads < 256:
        raise common.ProtocolError("invalid parity device geometry")
    dtype_rows = parity["dtype_paths"]
    if len(dtype_rows) != len(lock["dtype_paths_in_order"]):
        raise common.ProtocolError("parity dtype row count mismatch")
    parity_numels = [int(value) for value in lock["cuda_parity"]["parity_numels"]]
    for dtype_path, row in zip(lock["dtype_paths_in_order"], dtype_rows):
        common.strict_keys(
            row,
            ("dtype_path", "increment_checks", "jump_target_offset", "jump_value_sha256_f32le", "dlpack_value_sha256_f32le", "passed"),
            f"parity.{dtype_path}",
        )
        if row["dtype_path"] != dtype_path or row["passed"] is not True:
            raise common.ProtocolError(f"parity dtype mismatch/failure: {dtype_path}")
        if len(row["increment_checks"]) != len(parity_numels):
            raise common.ProtocolError(f"parity increment count mismatch: {dtype_path}")
        for expected_numel, check in zip(parity_numels, row["increment_checks"]):
            common.strict_keys(check, ("numel", "expected", "observed"), "parity.increment")
            numel = int(check["numel"])
            grid = min((numel + 255) // 256, sm * (threads // 256))
            formula = ((numel - 1) // (256 * grid * 4) + 1) * 4
            if numel != expected_numel or int(check["expected"]) != formula or int(check["observed"]) != formula:
                raise common.ProtocolError(f"Philox increment parity mismatch: {dtype_path}/{numel}")
        jump_hash = str(row["jump_value_sha256_f32le"])
        if len(jump_hash) != 64 or jump_hash != row["dlpack_value_sha256_f32le"]:
            raise common.ProtocolError(f"DLPack parity hash mismatch: {dtype_path}")


def _verify_candidate_selection(
    section: Mapping[str, Any], candidates: Sequence[common.Candidate], expected_rows: Sequence[common.SourceRow]
) -> None:
    common.strict_keys(
        section,
        ("candidate_count", "candidate_order_sha256", "winners", "summaries", "training_details", "training_folds"),
        "candidate_selection",
    )
    if int(section["candidate_count"]) != len(candidates):
        raise common.ProtocolError("candidate count mismatch")
    expected_order_hash = common.sha256_bytes(b"\n".join(candidate.id.encode() for candidate in candidates))
    if section["candidate_order_sha256"] != expected_order_hash:
        raise common.ProtocolError("candidate order hash mismatch")
    summaries = section["summaries"]
    if not isinstance(summaries, list) or len(summaries) != len(candidates):
        raise common.ProtocolError("candidate summary count mismatch")
    best: dict[str, tuple[float, int]] = {key: (math.inf, -1) for key in ("source", "gaussian", "permuted")}
    for candidate, summary in zip(candidates, summaries):
        common.strict_keys(summary, ("ordinal", "id", "source", "gaussian", "permuted"), "candidate_summary")
        if int(summary["ordinal"]) != candidate.ordinal or summary["id"] != candidate.id:
            raise common.ProtocolError("candidate summary order/identity mismatch")
        for domain in best:
            _verify_metric(summary[domain], f"candidate[{candidate.ordinal}].{domain}")
            q = float(summary[domain]["q"])
            if q < best[domain][0]:
                best[domain] = (q, candidate.ordinal)
    common.strict_keys(section["winners"], ("source", "gaussian", "permuted"), "candidate winners")
    for domain, (_, ordinal) in best.items():
        expected = candidates[ordinal].to_json()
        _equal_tree(section["winners"][domain], expected, f"candidate winner {domain}")

    _verify_details(section["training_details"], expected_rows, "training_details")
    common.strict_keys(section["training_folds"], ("source", "gaussian", "permuted"), "training_folds")
    for domain in ("source", "gaussian", "permuted"):
        expected_folds = common.fold_statistics(section["training_details"][domain])
        _equal_tree(section["training_folds"][domain], expected_folds, f"training_folds.{domain}")
        winner_summary = summaries[best[domain][1]][domain]
        _close(
            section["training_folds"][domain]["pooled"]["q"],
            winner_summary["q"],
            f"winner summary/detail q {domain}",
            rtol=2e-7,
            atol=2e-9,
        )


def verify_result(
    result_path: Path,
    workspace_root: Path | None = None,
    aux_dir: Path | None = None,
) -> dict[str, Any]:
    if common.environment_has_cuda_imports():
        raise common.ProtocolError("verifier must run in a CUDA-library-free process")
    result = _load_json(result_path)
    common.strict_keys(
        result,
        (
            "schema", "protocol", "strict_ptq", "pinned_panel", "backend", "bindings",
            "data_firewall", "sampler", "candidate_selection", "controls", "validation",
            "physical_ledger", "decision", "claim_boundary",
        ),
        "result",
    )
    if result["schema"] != common.SCHEMA or result["strict_ptq"] is not True:
        raise common.ProtocolError("schema/strict-PTQ mismatch")
    lock = common.load_candidate_lock()
    _equal_tree(
        result["protocol"],
        {"candidate_lock_status": lock["status"], "global_candidate_only": True, "scientific_cli_knobs": False},
        "protocol",
    )
    _equal_tree(result["pinned_panel"], {"opened": False, "access_permitted": False}, "pinned_panel")
    common.strict_keys(result["backend"], ("production", "name", "parity"), "backend")
    if result["backend"]["production"] is not True or result["backend"]["name"] != "cupy_with_pytorch_cuda_philox":
        raise common.ProtocolError("non-production backend")
    _verify_parity(result["backend"]["parity"], lock)

    expected_bindings = {
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "candidate_lock_internal_sha256": common.CANDIDATE_LOCK_INTERNAL_SHA256,
        "source_manifest_sha256": common.EXPECTED_SOURCE_MANIFEST_SHA256,
        "source_freeze_sha256": common.EXPECTED_SOURCE_FREEZE_SHA256,
        "exclusion_manifest_sha256": common.EXPECTED_EXCLUSION_MANIFEST_SHA256,
        "exclusion_intersection_lock_sha256": common.EXPECTED_EXCLUSION_INTERSECTION_SHA256,
        "runner_sha256": common.sha256_file(common.PACKAGE_DIR / "initialization_anchor_gate.py"),
        "common_sha256": common.sha256_file(common.PACKAGE_DIR / "common.py"),
        "revision": common.REVISION,
    }
    _equal_tree(result["bindings"], expected_bindings, "bindings")
    rows = common.load_frozen_source_rows(workspace_root)
    _, exclusion_binding = common.load_exclusion_binding(workspace_root)
    eligible = [row for row in rows if not row.excluded]
    train = [row for row in eligible if row.split == "candidate_selection"]
    validation_rows = [row for row in eligible if row.split == "validation"]
    plan = common.make_coordinate_plan(rows)

    firewall = result["data_firewall"]
    common.strict_keys(
        firewall,
        (
            "auxiliary_directory", "exact_file_count", "eligible_tensor_count",
            "candidate_selection_tensor_count", "validation_tensor_count", "excluded",
            "exclusion_binding", "eligible",
            "access_log", "candidate_selection_before_validation_payload_open",
        ),
        "data_firewall",
    )
    if any(
        (
            int(firewall["exact_file_count"]) != len(rows),
            int(firewall["eligible_tensor_count"]) != len(eligible),
            int(firewall["candidate_selection_tensor_count"]) != len(train),
            int(firewall["validation_tensor_count"]) != len(validation_rows),
            firewall["candidate_selection_before_validation_payload_open"] is not True,
        )
    ):
        raise common.ProtocolError("firewall counts/order flag mismatch")
    expected_excluded = [
        {
            "tensor_name": row.tensor_name,
            "basename": row.basename,
            "reason": "tensor identity occurs in bound heldout32 exclusion manifest",
            "payload_opened": False,
        }
        for row in rows if row.excluded
    ]
    expected_eligible = [
        {
            "tensor_name": row.tensor_name, "basename": row.basename, "expert": row.expert,
            "role": row.role, "split": row.split, "sha256": row.sha256, "bytes": row.bytes,
        }
        for row in eligible
    ]
    _equal_tree(firewall["excluded"], expected_excluded, "firewall.excluded")
    _equal_tree(firewall["exclusion_binding"], exclusion_binding, "firewall.exclusion_binding")
    _equal_tree(firewall["eligible"], expected_eligible, "firewall.eligible")
    expected_log: list[dict[str, Any]] = [{"sequence": 0, "event": "cuda_parity_passed_before_payload_access"}]
    for source in train:
        expected_log.append(
            {"sequence": len(expected_log), "event": "payload_opened_and_hash_verified", "tensor_name": source.tensor_name, "split": source.split, "sha256": source.sha256}
        )
    winners = result["candidate_selection"]["winners"]
    expected_log.append(
        {"sequence": len(expected_log), "event": "global_candidates_frozen_before_validation_payload_access", "winners": {domain: winners[domain]["id"] for domain in ("source", "gaussian", "permuted")}}
    )
    for source in validation_rows:
        expected_log.append(
            {"sequence": len(expected_log), "event": "payload_opened_and_hash_verified", "tensor_name": source.tensor_name, "split": source.split, "sha256": source.sha256}
        )
    _equal_tree(firewall["access_log"], expected_log, "firewall.access_log")

    sampler = result["sampler"]
    expected_sampler = {
        "total_coordinates": common.TOTAL_COORDINATES,
        "fit_coordinates": common.FIT_COORDINATES,
        "score_coordinates": common.SCORE_COORDINATES,
        "coordinate_plan_sha256": common.coordinate_plan_sha256(plan),
        "per_tensor": common.coordinate_plan_json(plan),
    }
    _equal_tree(sampler, expected_sampler, "sampler")

    candidates = common.enumerate_candidates(lock)
    _verify_candidate_selection(result["candidate_selection"], candidates, train)
    controls = result["controls"]
    expected_controls = {
        "matched_gaussian": {
            "same_candidate_count": len(candidates),
            "winner": winners["gaussian"]["id"],
            "generation": "fit-only moments plus stateless SHA256 Box-Muller",
        },
        "permuted_anchor": {
            "same_candidate_count": len(candidates),
            "winner": winners["permuted"]["id"],
            "generation": "SHA256 fixed permutation independently within matrix and split",
        },
    }
    _equal_tree(controls, expected_controls, "controls")

    validation = result["validation"]
    common.strict_keys(validation, ("source_winner", "details", "folds"), "validation")
    if validation["source_winner"] != winners["source"]["id"]:
        raise common.ProtocolError("validation source winner mismatch")
    _verify_details(validation["details"], validation_rows, "validation.details")
    common.strict_keys(validation["folds"], ("source", "gaussian", "permuted"), "validation.folds")
    for domain in ("source", "gaussian", "permuted"):
        expected_folds = common.fold_statistics(validation["details"][domain])
        _equal_tree(validation["folds"][domain], expected_folds, f"validation.folds.{domain}")

    ledger = common.physical_ledger()
    _equal_tree(result["physical_ledger"], ledger, "physical_ledger")
    decision = common.make_decision(
        validation["folds"]["source"],
        validation["folds"]["gaussian"]["pooled"]["capture"],
        validation["folds"]["permuted"]["pooled"]["capture"],
    )
    _equal_tree(result["decision"], decision, "decision")
    _equal_tree(result["claim_boundary"], lock["claim_boundary"], "claim_boundary")

    payload_receipt: dict[str, Any]
    if aux_dir is None:
        payload_receipt = {"rehash_performed": False, "eligible_hashed": 0, "excluded_payloads_opened": 0}
    else:
        paths = common.validate_aux_directory(aux_dir, rows)
        recorded_dir = Path(str(firewall["auxiliary_directory"])).resolve()
        if aux_dir.resolve() != recorded_dir:
            raise common.ProtocolError("verification aux directory differs from production record")
        hashes = []
        for source in eligible:
            digest = common.sha256_file(paths[source.tensor_name])
            if digest != source.sha256:
                raise common.ProtocolError(f"eligible payload SHA-256 mismatch: {source.basename}")
            hashes.append({"tensor_name": source.tensor_name, "sha256": digest})
        payload_receipt = {
            "rehash_performed": True,
            "eligible_hashed": len(hashes),
            "eligible_hashes": hashes,
            "excluded_payloads_opened": 0,
        }

    return {
        "schema": "qwen3_initialization_anchor_verification_receipt_v1",
        "status": "PASS",
        "result_path": str(result_path.resolve()),
        "result_sha256": common.sha256_file(result_path.resolve()),
        "verifier_sha256": common.sha256_file(Path(__file__).resolve()),
        "candidate_count": len(candidates),
        "decision_state": decision["state"],
        "payload_verification": payload_receipt,
        "pinned_panel_opened": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--aux-dir", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = verify_result(args.result, args.workspace_root, args.aux_dir)
    payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.receipt is not None:
        receipt_path = args.receipt.resolve()
        if receipt_path.exists():
            raise common.ProtocolError("verification receipt already exists")
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.chmod(receipt_path, 0o444)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
