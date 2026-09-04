"""Independent hostile audit of COCHAIN-Q v0; no payload or accelerator access."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TARGET = REPO / "research" / "cochain_q_plaquette_v0_20260904"
TARGET_MANIFEST_SHA256 = "ef12407301265d8e04da9f1ed5afaadff69f0d864c31ef1be4868279506a68b3"
TARGET_SOURCE_ROOT_SHA256 = "3e515bc146fde2dde734fb94eb11f5dc32397d227fd4dd3930037b2ce498190a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(REPO), text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)


def load_target():
    path = TARGET / "cochain_q_oracle.py"
    spec = importlib.util.spec_from_file_location("audited_cochain_q_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load target")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if sha256(TARGET / "SOURCE_MANIFEST.json") != TARGET_MANIFEST_SHA256:
        raise RuntimeError("target manifest drift")
    manifest = json.loads((TARGET / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("source_root_sha256") != TARGET_SOURCE_ROOT_SHA256:
        raise RuntimeError("target source root drift")
    for name, expected in manifest["files_sha256"].items():
        if sha256(TARGET / name) != expected:
            raise RuntimeError(f"target member drift: {name}")

    pinned = run([sys.executable, "-I", "-B", str(TARGET / "verify_source.py"),
                  "--package", str(TARGET), "--manifest-sha256",
                  TARGET_MANIFEST_SHA256, "--self-test"])
    if pinned.returncode != 0:
        raise RuntimeError("pinned target verifier failed")

    c = load_target()
    exact_cases = 0
    rng = np.random.default_rng(20260904)
    for dimension, cells in ((2, 1), (2, 2), (3, 1)):
        for _ in range(12):
            costs = rng.uniform(0.001, 5.0,
                                size=(cells, 1 << dimension, 2)).astype(np.float64)
            for syndrome in (None, 0, 1):
                fast_q, fast_sse = c.best_labels(costs, dimension, syndrome)
                brute_q, brute_sse = c.brute_force_global(costs, dimension, syndrome)
                if (abs(fast_sse - brute_sse) >
                        1e-12 * max(1.0, abs(fast_sse), abs(brute_sse)) or
                        not np.array_equal(fast_q, brute_q)):
                    raise RuntimeError("global optimum mismatch")
                exact_cases += 1

    bijection_patterns = 0
    public_packets = {}
    for dimension in (2, 3):
        patterns = c.all_patterns(dimension)
        boundary, syndrome = c.cochain_coordinates(patterns)
        if not np.array_equal(c.inverse_cochain_coordinates(boundary, syndrome), patterns):
            raise RuntimeError("cochain bijection")
        bijection_patterns += len(patterns)
        even = patterns[c.mixed_syndrome(patterns) == 0]
        packets = [c.encode_public_fiber(row[None, :], 0) for row in even]
        if len(set(packets)) != len(even):
            raise RuntimeError("public packet collision")
        for row, packet in zip(even, packets):
            if not np.array_equal(c.decode_public_fiber(packet, 1, dimension, 0)[0], row):
                raise RuntimeError("public packet decode")
        public_packets[str(dimension)] = len(packets)

    rate_identity_cases = 0
    for dimension in (2, 3):
        cells = 17
        costs = rng.uniform(0.05, 2.0,
                            size=(cells, 1 << dimension, 2)).astype(np.float64)
        result = c.run_oracle(costs, 1000.0, dimension, 0)
        baseline_q, baseline_sse = c.best_labels(costs, dimension)
        candidate_q, candidate_sse = c.best_labels(costs, dimension, 0)
        packet = c.encode_public_fiber(candidate_q, 0)
        physical_rate = 8 * len(packet) / (cells * (1 << dimension))
        manual_gain = ((1.0 - physical_rate) +
                       0.5 * np.log2(baseline_sse / candidate_sse))
        if abs(manual_gain - result["public_fiber"]["physical_equivalent_gain_bpw"]) > 1e-14:
            raise RuntimeError("equivalent-gain identity")
        rate_identity_cases += 1

    fixture = {}
    for dimension in (2, 3):
        q = c.low_degree_even_ensemble(dimension)
        mi = c.pairwise_mutual_information(q)
        if max(abs(x) for x in mi.values()) > 1e-15:
            raise RuntimeError("parity fixture pairwise MI")
        costs, energy = c.preference_costs(q)
        result = c.run_oracle(costs, energy, dimension)
        fixture[f"parity_d{dimension}"] = {
            "max_pairwise_mi": max(mi.values()),
            "physical_gain_bpw_per_affected_bitplane_site":
                result["public_fiber"]["physical_equivalent_gain_bpw"],
        }
    iid = c.all_patterns(3)
    if np.bincount(c.mixed_syndrome(iid), minlength=2).tolist() != [128, 128]:
        raise RuntimeError("IID syndrome balance")
    iid_costs, iid_energy = c.preference_costs(iid, 1.0, 1001.0)
    iid_result = c.run_oracle(iid_costs, iid_energy, 3)
    if not iid_result["status"].startswith("HARD_KILL"):
        raise RuntimeError("IID hard kill")
    fixture["iid_cube"] = {
        "syndrome_counts": [128, 128],
        "physical_gain_bpw_per_affected_bitplane_site":
            iid_result["public_fiber"]["physical_equivalent_gain_bpw"],
    }

    try:
        c.payload_execution_gate()
        gate_fail_closed = False
    except RuntimeError:
        gate_fail_closed = True
    if not gate_fail_closed:
        raise RuntimeError("payload gate open")

    # Hostile manifest test: an unlisted executable must invalidate a closed package.
    with tempfile.TemporaryDirectory(prefix="cochain_hostile_") as temp:
        copied = Path(temp) / "target"
        shutil.copytree(TARGET, copied)
        (copied / "UNLISTED_EXECUTABLE.py").write_text(
            "raise RuntimeError('unlisted member')\n", encoding="utf-8")
        extra = run([sys.executable, "-I", "-B", str(copied / "verify_source.py"),
                     "--package", str(copied), "--manifest-sha256",
                     TARGET_MANIFEST_SHA256])
        verifier_accepted_unlisted_member = extra.returncode == 0

    unpinned = run([sys.executable, "-I", "-B", str(TARGET / "verify_source.py"),
                    "--package", str(TARGET)])
    verifier_accepted_missing_external_pin = unpinned.returncode == 0

    malformed = np.asarray([[0.0, 0.0, 0.0, 0.9]], dtype=np.float64)
    try:
        malformed_packet = c.encode_public_fiber(malformed, 0)
        encoder_accepted_noninteger_labels = True
        malformed_packet_sha256 = hashlib.sha256(malformed_packet).hexdigest()
    except (ValueError, TypeError):
        encoder_accepted_noninteger_labels = False
        malformed_packet_sha256 = None

    small_ledger = c.physical_ledger(1, 2)
    production_ledger = c.physical_ledger(1 << 20, 2)
    report = {
        "schema": "cochain_q_plaquette_independent_hostile_audit_v0",
        "status": "PASS_MECHANISM__BLOCK_QWEN_CAPABILITY_PENDING_REPAIRS",
        "target_manifest_sha256": TARGET_MANIFEST_SHA256,
        "target_source_root_sha256": TARGET_SOURCE_ROOT_SHA256,
        "pinned_verifier_passed": True,
        "exact_global_random_cases": exact_cases,
        "bijection_patterns_checked": bijection_patterns,
        "public_packet_codewords_checked": public_packets,
        "rate_identity_cases": rate_identity_cases,
        "fixtures": fixture,
        "gate_fail_closed": gate_fail_closed,
        "logical_read": {
            "topology_expert_local": True,
            "logical_amplification": production_ledger["logical_routed_read_amplification"],
            "one_cell_page_rounding_over_payload": small_ledger["page_rounding_over_payload"],
            "production_fixture_page_rounding_over_payload":
                production_ledger["page_rounding_over_payload"],
            "verdict": "1x topology is valid; physical page layout remains unproven",
        },
        "blockers": {
            "manifest_verifier_accepted_unlisted_member": verifier_accepted_unlisted_member,
            "manifest_verifier_accepted_missing_external_pin":
                verifier_accepted_missing_external_pin,
            "encoder_accepted_noninteger_labels": encoder_accepted_noninteger_labels,
            "malformed_packet_sha256": malformed_packet_sha256,
            "gain_units": (
                "reported bpw is per affected bitplane site; role/plane subsets must be "
                "renormalized to all audited expert weights before global 0.045/0.10/"
                "0.15289 gates"),
            "distortion_scope": (
                "Qwen promotion requires one legal six-plane reconstruction and pooled "
                "original-source SSE; independent per-plane gains cannot be added"),
        },
        "qwen_accessed": False,
        "gpu_accessed": False,
        "network_accessed": False,
    }
    if not verifier_accepted_unlisted_member or not verifier_accepted_missing_external_pin:
        raise RuntimeError("hostile verifier reproduction unexpectedly repaired")
    if not encoder_accepted_noninteger_labels:
        raise RuntimeError("malformed-label reproduction unexpectedly repaired")
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
