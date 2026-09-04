#!/usr/bin/env python3
"""Independent, source-only hostile audit of the sealed PAIRPATH-P2 r2 package.

This program deliberately has no payload, model, GPU, CuPy, network, or
execution-authority path.  It verifies the exact target source closure and
replays adversarial source-free checks against caller-independent fixtures.
"""

from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import stat
import struct
import subprocess
import sys

import numpy as np


AUDIT = Path(__file__).resolve().parent
REPOSITORY = AUDIT.parent.parent
TARGET = REPOSITORY / "research" / "pairpath_fl_same_layer_microcodec_v0_20260903_r2"
TARGET_MANIFEST_SHA256 = "21983efff5ac5c0593a655cae4136d35ca24400fd807f9fe4be458a34b18e622"
TARGET_ROOT_SHA256 = "7ffb0b9c92861c7171a3b89f47d6fa03caac963322d772fb8c0b020ce501cf96"

sys.path.insert(0, str(TARGET))
import pairpath_r2_core as core  # noqa: E402
import run_gate  # noqa: E402
from source_free_fixtures import aligned_fixture, iid_fixture  # noqa: E402


class ProbeStop(RuntimeError):
    """Private control-flow exception for observing a computed bit weight."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def verify_target_closure() -> dict:
    manifest_path = TARGET / "SOURCE_MANIFEST.json"
    assert sha256(manifest_path) == TARGET_MANIFEST_SHA256
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    assert raw == canonical_json(manifest)
    rows = manifest["files"]
    assert [row["name"] for row in rows] == sorted(row["name"] for row in rows)
    assert sorted(path.name for path in TARGET.iterdir()) == sorted(
        [row["name"] for row in rows] + ["SOURCE_MANIFEST.json"])
    canonical_rows = []
    for row in rows:
        path = TARGET / row["name"]
        assert not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode)
        assert path.stat().st_size == row["bytes"]
        assert sha256(path) == row["sha256"]
        canonical_rows.append({"bytes": row["bytes"], "name": row["name"],
                               "sha256": row["sha256"]})
    root = hashlib.sha256(json.dumps(
        canonical_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert root == manifest["source_root_sha256"] == TARGET_ROOT_SHA256
    return {"manifest_sha256": sha256(manifest_path), "member_count": len(rows),
            "source_root_sha256": root}


def run_target_verifier_and_tests() -> dict:
    verifier = subprocess.run(
        [sys.executable, "-I", "-B", str(TARGET / "verify_source.py"),
         "--package", str(TARGET), "--manifest-sha256", TARGET_MANIFEST_SHA256,
         "--self-test"], cwd=str(REPOSITORY), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert verifier.returncode == 0, verifier.stdout
    row = json.loads(verifier.stdout.strip().splitlines()[-1])
    assert row["status"] == "PASS_SOURCE_ONLY_HOLD_PENDING_INDEPENDENT_HOSTILE_AUDIT"
    assert row["source_root_sha256"] == TARGET_ROOT_SHA256
    assert row["self_test_passed"] is True
    return {"returncode": verifier.returncode,
            "reported_status": row["status"],
            "self_test_passed": row["self_test_passed"]}


def check_execution_boundary() -> dict:
    assert run_gate.PAYLOAD_EXECUTION_ENABLED is False
    assert run_gate.LOCAL_GPU_EXECUTION_ENABLED is False
    assert run_gate.QWEN_APERTURE_AUTHORIZED is False
    try:
        run_gate.main()
    except SystemExit as error:
        message = str(error)
    else:
        raise AssertionError("run gate unexpectedly returned")
    assert message == "SOURCE_ONLY_HOLD_NO_PAYLOAD_OR_GPU_AUTHORITY"
    forbidden = ("paramiko", "requests.", "urllib.request", "socket.",
                 "subprocess.popen", ".safetensors", "huggingface", "cupy", "runpod")
    for name in ("pairpath_r2_core.py", "run_gate.py", "source_free_fixtures.py"):
        source = (TARGET / name).read_text(encoding="utf-8").lower()
        assert not [token for token in forbidden if token in source]
    return {"gpu_accessed": False, "network_accessed": False,
            "payload_opened": False, "run_gate_exit": message}


def check_multistarts_and_mi() -> dict:
    values = iid_fixture(16384)
    scales = core.estimate_scale_bits(values)
    levels = np.stack([core.levels_per_coordinate(scales[e, 1], values.shape[2])
                       for e in range(2)])
    starts = core._ideal_initializations(values[:, 1], levels)
    constants = {(int(q[0, 0]), int(q[1, 0])) for q in starts
                 if np.all(q[0] == q[0, 0]) and np.all(q[1] == q[1, 0])}
    assert constants == set(itertools.product(range(core.ALPHABET), repeat=2))
    measured = core.fixed_assignment_mi_ceiling(values)
    manual_rows = []
    for role in core.OPTIMIZED_ROLES:
        q0 = core.nearest_labels(values[0, role], scales[0, role])
        q1 = core.nearest_labels(values[1, role], scales[1, role])
        joint = np.zeros((4, 4), np.int64)
        np.add.at(joint, (q0, q1), 1)
        p = joint / joint.sum()
        p0, p1 = p.sum(axis=1), p.sum(axis=0)
        mi = 0.0
        for a, b in itertools.product(range(4), repeat=2):
            if p[a, b] > 0:
                mi += float(p[a, b] * math.log2(p[a, b] / (p0[a] * p1[b])))
        manual_rows.append(mi)
    observed = [row["mutual_information_bits_per_coordinate_pair"]
                for row in measured["role_rows"]]
    np.testing.assert_allclose(observed, manual_rows, rtol=0, atol=2e-15)
    assert measured["conditioning"] == "decoder-visible role"
    return {"multistart_count": len(starts), "constant_start_count": len(constants),
            "role_conditioned_mi": measured["mutual_information_bits_per_coordinate_pair"],
            "manual_role_mi": manual_rows}


def check_global_bit_weight_contract() -> dict:
    """Observe the actual finite role-local weights and compare to the sealed claim."""
    source = iid_fixture(16384, seed=0x42575431)
    source[:, 1] *= 0.25
    source[:, 2] *= 4.0
    scales = core.estimate_scale_bits(source)
    lagrange = Fraction(1, 64)
    observed = []
    original = core._fit_pair_fold

    def spy(values, levels, nearest, train, flexible, bit_weight):
        observed.append(float(bit_weight))
        raise ProbeStop

    core._fit_pair_fold = spy
    try:
        for role in core.OPTIMIZED_ROLES:
            try:
                core.choose_pair_labels(source[:, role], scales[:, role], lagrange, True)
            except ProbeStop:
                pass
    finally:
        core._fit_pair_fold = original
    assert len(observed) == 2
    expected_global = (float(lagrange) *
                       float(np.sum(source[:, core.OPTIMIZED_ROLES] ** 2)) /
                       source[:, core.OPTIMIZED_ROLES].size)
    expected_role_local = [float(lagrange) * float(np.sum(source[:, role] ** 2)) /
                           source[:, role].size for role in core.OPTIMIZED_ROLES]
    np.testing.assert_allclose(observed, expected_role_local, rtol=2e-15, atol=0)
    assert not math.isclose(observed[0], expected_global, rel_tol=1e-6)
    assert not math.isclose(observed[1], expected_global, rel_tol=1e-6)
    return {"contract_claim": "one global Up/Down rate-distortion multiplier",
            "expected_global_bit_weight": expected_global,
            "observed_up_bit_weight": observed[0],
            "observed_down_bit_weight": observed[1],
            "finding": "BLOCK_FINITE_ENCODER_USES_ROLE_LOCAL_MULTIPLIERS"}


def _empirical_joint_objective(values: np.ndarray, levels: np.ndarray,
                               labels: np.ndarray, bit_weight: float) -> float:
    reconstruction = np.take_along_axis(levels, labels[:, :, None], axis=2)[:, :, 0]
    sse = float(np.sum((values - reconstruction) ** 2, dtype=np.float64))
    index = labels[0].astype(np.int64) * core.ALPHABET + labels[1]
    rate = core._entropy_bits(np.bincount(index, minlength=16)) / 2.0
    return sse + bit_weight * rate * values.size


def check_joint_solver_dominance() -> dict:
    """Reproduce a legal-level case where the joint heuristic misses a valid seed."""
    rng = np.random.default_rng(16010)
    values = rng.normal(size=(2, 16))
    scales = np.asarray((0.7, 1.3), dtype=np.float64)
    levels = np.stack([np.tile(scales[e] * core.LEVELS_RMS, (16, 1))
                       for e in range(2)])
    bit_weight = 0.1
    independent_labels, _, _ = core._ideal_flexible_role(
        values, levels, bit_weight, False)
    joint_labels, _, _ = core._ideal_flexible_role(values, levels, bit_weight, True)
    valid_independent_seed_objective = _empirical_joint_objective(
        values, levels, independent_labels, bit_weight)
    returned_joint_objective = _empirical_joint_objective(
        values, levels, joint_labels, bit_weight)
    gap = returned_joint_objective - valid_independent_seed_objective
    assert gap > 1e-12
    return {"bit_weight": bit_weight,
            "valid_independent_labels_under_joint_objective": valid_independent_seed_objective,
            "returned_joint_objective": returned_joint_objective,
            "suboptimality_gap": gap,
            "finding": "BLOCK_KILL_ORACLE_HAS_NO_DOMINANCE_OR_GLOBAL_OPTIMALITY_CERTIFICATE"}


def check_oracle_controls_and_hulls() -> dict:
    grid = (core.LAMBDA_GRID[0], core.LAMBDA_GRID[-1])
    iid = core.optimistic_single_letter_joint_gate(iid_fixture(16384), grid)
    aligned = core.optimistic_single_letter_joint_gate(aligned_fixture(), grid)
    assert iid["status"] == "HARD_KILL_OPTIMISTIC_JOINT_GATE_BELOW_0P045"
    assert aligned["status"] == "SURVIVE_OPTIMISTIC_GATE_WITH_PHYSICAL_MARGIN"
    for result in (iid, aligned):
        for hull_name in ("independent_hull", "pair_hull"):
            hull = result[hull_name]
            assert all(b[0] > a[0] and b[1] < a[1] for a, b in zip(hull, hull[1:]))
            slopes = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(hull, hull[1:])]
            assert all(b > a for a, b in zip(slopes, slopes[1:]))
        for row in result["equal_rate"]:
            assert math.isclose(row["G_eq_bpw"],
                                0.5 * math.log2(row["D_ind"] / row["D_pair"]),
                                rel_tol=0, abs_tol=2e-15)
        for row in result["equal_mse"]:
            assert math.isclose(row["G_eq_bpw"], row["R_ind_bpw"] - row["R_pair_bpw"],
                                rel_tol=0, abs_tol=2e-15)
    return {"iid_status": iid["status"], "iid_best_gain_bpw": iid["best_G_eq_UD_bpw"],
            "aligned_status": aligned["status"],
            "aligned_best_gain_bpw": aligned["best_G_eq_UD_bpw"],
            "hull_formula_recomputed": True,
            "normalization": "rates per Up/Down weight; distortion per Up/Down energy"}


def _replace_header(packet: bytes, mutate) -> bytes:
    header, common_payload, _ = core._parse_packet(packet)
    common_size = int(header["common_pages"]) * core.PAGE_BYTES
    private_tail = packet[common_size:]
    changed = copy.deepcopy(header)
    mutate(changed)
    raw_header = core.canonical_json(changed)
    common_raw = core.MAGIC + struct.pack("<I", len(raw_header)) + raw_header + common_payload
    assert len(common_raw) <= common_size
    return common_raw + bytes(common_size - len(common_raw)) + private_tail


def check_literal_packet_binding_and_read() -> dict:
    source = iid_fixture(16384, seed=0x5041434B)
    result = core.run_micro_oracle(source, (core.LAMBDA_GRID[0],))
    packet = result["selected_packet"]
    decoded = core.decode_packet(packet)
    binding = core.make_binding(source, packet)
    receipt = core.validate_binding(binding, source, packet)
    assert receipt["status"] == "PASS_REAL_BYTE_BINDING"
    assert decoded["packet_sha256"] == hashlib.sha256(packet).hexdigest()
    changed_source = source.copy()
    changed_source[0, 0, 0] = np.nextafter(changed_source[0, 0, 0], math.inf)
    try:
        core.validate_binding(binding, changed_source, packet)
    except core.CodecError as error:
        source_tamper = str(error)
    else:
        raise AssertionError("source tamper accepted")
    ledger = core.packet_read_ledger(packet)
    header = decoded["header"]
    common = int(header["common_pages"]) * core.PAGE_BYTES
    privates = [int(v) * core.PAGE_BYTES for v in header["private_pages"]]
    common_raw = 12 + len(core.canonical_json(header)) + int(header["common_payload_bytes"])
    private_raw = [int(v) for v in header["private_payload_bytes"]]
    manual_physical = [Fraction(common + privates[e], 1) /
                       (Fraction(common, 2) + privates[e]) for e in range(2)]
    manual_conservative = [Fraction(common + privates[e], 1) /
                           (Fraction(common_raw, 2) + private_raw[e]) for e in range(2)]
    assert ledger["amplification_physical"] == [str(v) for v in manual_physical]
    assert ledger["amplification_conservative"] == [str(v) for v in manual_conservative]
    assert ledger["strictly_below_2x"] and max(manual_physical + manual_conservative) < 2

    def corrupt_tree(header_row: dict) -> None:
        header_row["tree_descriptor"] = {
            "packed": 1, "bits": 0, "pairs": [[0, 0]], "merge_ranks": [],
            "materialized": [9, 9]}

    malformed = _replace_header(packet, corrupt_tree)
    malformed_decoded = core.decode_packet(malformed)
    malformed_score = core.evaluate_packet(source, malformed)
    assert malformed_decoded["header"]["tree_descriptor"]["pairs"] == [[0, 0]]
    assert math.isfinite(malformed_score["F"])
    return {"selected_candidate": result["selected_candidate"],
            "selected_rate_fraction": result["selected_score"]["rate_fraction"],
            "selected_F": result["selected_score"]["F"],
            "source_tamper_rejection": source_tamper,
            "read_ledger": ledger,
            "finding": "BLOCK_DECODER_ACCEPTS_INVALID_UNREPLAYED_TREE_DESCRIPTOR",
            "malformed_packet_sha256": hashlib.sha256(malformed).hexdigest()}


def check_complete_controls() -> dict:
    # At the minimum 16,384-coordinate geometry some entropy realizations have
    # too much page/header overhead to satisfy the strict read gate.  The
    # package's own complete-control KAT consequently uses 32,768 coordinates.
    source = iid_fixture(32768, seed=0x434F4E54)
    result = core.run_complete_controls(
        source, control_seeds=(core.CONTROL_SEEDS[0],), include_gaussian=True,
        lambda_grid=(core.LAMBDA_GRID[0],))
    assert result["control_count"] == 2
    assert {row["kind"] for row in result["controls"]} == {"affine", "gaussian"}
    return {"control_count": result["control_count"],
            "source_gain_bpw": result["source_result"]["equivalent_gain_bpw"],
            "max_positive_control_gain_bpw": result["max_positive_control_gain_bpw"],
            "control_corrected_gain_bpw": result["control_corrected_gain_bpw"],
            "status": result["status"], "full_pipeline_refit": True}


def check_tree_descriptor() -> dict:
    counts = {}
    for experts in (2, 4, 6, 8):
        width = core.tree_descriptor_bits(experts)
        valid = 0
        for packed in range(1 << width if width else 1):
            try:
                row = core.decode_tree_descriptor(packed, experts)
            except core.CodecError:
                continue
            replay = core.encode_tree_descriptor(experts, row["pairs"], row["merge_ranks"])
            assert replay == (packed, width)
            assert core.flatten_tree(row["tree"]) == tuple(range(experts))
            valid += 1
        expected = core.odd_double_factorial(experts - 1)
        for active in range(experts // 2, 2, -1):
            expected *= math.comb(active, 2)
        assert valid == expected
        counts[str(experts)] = {"bits": width, "valid_codewords": valid}
    return counts


def main() -> None:
    report = {
        "schema": "pairpath_p2_r2_independent_hostile_audit_v1",
        "target": "research/pairpath_fl_same_layer_microcodec_v0_20260903_r2",
        "target_closure": verify_target_closure(),
        "target_verifier": run_target_verifier_and_tests(),
        "execution_boundary": check_execution_boundary(),
        "multistarts_and_mi": check_multistarts_and_mi(),
        "global_bit_weight": check_global_bit_weight_contract(),
        "joint_solver_dominance": check_joint_solver_dominance(),
        "oracle_controls_and_hulls": check_oracle_controls_and_hulls(),
        "literal_packet": check_literal_packet_binding_and_read(),
        "complete_controls": check_complete_controls(),
        "tree_descriptor": check_tree_descriptor(),
        "blockers": [
            "finite pair encoder uses separate role-energy-normalized bit weights despite sealed global-Up/Down claim",
            "heuristic joint oracle can return a worse joint objective than a valid label assignment found by its independent solver, so a low score is not a certified upper envelope and cannot authorize a hard kill",
            "literal decoder accepts an invalid, unreplayed tree descriptor and still evaluates the packet",
        ],
        "verdict": "BLOCK_R2_NO_PAYLOAD_CAPABILITY_OR_HARD_KILL_AUTHORITY",
        "qwen_payload_opened": False,
        "gpu_accessed": False,
        "network_accessed": False,
        "runpod_accessed": False,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
