#!/usr/bin/env python3
"""Frozen end-to-end confirmation harness for POLARIS-SC-v2.

The harness owns encode -> exact global-reservoir pack -> independent unpack ->
fresh-process causal decode.  Rate, absolute MSE, and sample-relative MSE use a
Bonferroni-corrected family-wise 99% one-sided Student-t gate.  All failures
are terminal for this frozen candidate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import scipy
from scipy.stats import t as student_t


PROTOCOL = "POLARIS-SC-v2-confirmation-2026-08-31"
SEEDS = (
    173205080, 173309809, 173414538, 173519267,
    223606797, 223711526, 223816255, 223920984,
    244948974, 245053703, 245158432, 245263161,
    264575131, 264679860, 264784589, 264889318,
    316227766, 316332495, 316437224, 316541953,
    331662479, 331767208, 331871937, 331976666,
    346410161, 346514890, 346619619, 346724348,
    360555127, 360659856, 360764585, 360869314,
)
V1_EXCLUDED_SEEDS = frozenset(
    (
        314159265, 314263994, 314368723, 314473452,
        271828182, 271932911, 272037640, 272142369,
        161803399, 161908128, 162012857, 162117586,
        141421356, 141526085, 141630814, 141735543,
    )
)
N = 262_144
SIGMA_SOURCE = 1.0
TEST_DISTORTION = 0.05110
ETA = 0.25
ALPHABET_SIZE = 64
PAYLOAD_CAPACITY_BITS_PER_BLOCK = 563_464
DIRECTORY_BITS_PER_BLOCK = 48
RANK2_BITS_PER_BLOCK = 563_512
RESERVOIR_HEADER_BYTES = 96
GAUSSIAN_LIMIT = 0.050765774772264724
FIVE_PERCENT_TARGET = 0.053304063510877964
FAMILY_WISE_CONFIDENCE = 0.99
BONFERRONI_METRICS = 3
PER_METRIC_CONFIDENCE = 1.0 - (1.0 - FAMILY_WISE_CONFIDENCE) / BONFERRONI_METRICS
MAX_BPW = 2.15
NUMERICAL_TOLERANCE = 1e-12
EXPECTED_GPU = "NVIDIA GeForce RTX 5090"
OPENED_LOCK_NAME = ".agent_polaris_sc_v2_confirmation_opened.lock"

EXPECTED_VERSIONS = {
    "python": "3.12.3",
    "numpy": "2.5.2",
    "cupy": "14.2.0",
    "scipy": "1.18.1",
}
EXPECTED_ARTIFACTS = {
    "agent_polaris_sc_v2_frozen_manifest.json": "06967a4e852c9d39c97fe39b45d50df558471e0f35912d330b6dc1e7493df5e0",
    "agent_root_polar_lattice_gate.py": "95cfd32e5d026f07ceffe90daa7f88ca5e62f9f90546dfe74fc37cf06854d9b8",
    "agent_polaris_sc_v1_decoder_map.npz": "a0e9895d5e30df71d51ee85ed8893c4983e4369748912fbdd61acbad0fed18ef",
    "agent_polaris_independent_decoder_v1.py": "0652521fae5c77567e67cc9434adfb84b1a3cab53e5a79f250a3093a467be072",
    "agent_polaris_reservoir_pack_v2.py": "c5fda34242153365dac07b5990bcd1fa19f0ac98d2512d47c3c8e1ec2a81dde8",
    "agent_polaris_reservoir_unpack_v2.py": "cf7113c3fbc6340f0870dadcf7608739aa651f5706befa163b5d13516dac7e07",
    "agent_polaris_reservoir_block_decode_v2.py": "efa7bd3fa527722538368ac0e39b7b8c585e7ea0efbff34ac5b96a0d7b4af801",
    "agent_novel_polaris_rate_ledger_v2.py": "bd5f608ed4ee1db4cbd9af17c36aa9027f4ccd2d5edb4d5e53ccae65c8ac457c",
    "agent_novel_polaris_rate_ledger_v2.json": "b3969bf5b0377f045dd2c08e97457603cf62d7db28d5607ff3dc87b36f33528e",
    "agent_polaris_reservoir_unpack_v2_selftest.py": "3da99dc8e706e799bd17c819214c52a27fd53c688a226e42b5fdcdb28e6fe229",
    "agent_polaris_v2_framing_selftest.py": "947a083d31336e912a9708413ce018b0e796b4a880e08590eaba9d98fc46084e",
}
POLAR_REPOSITORY_COMMIT = "458187b9b03db1768a4b72d617e591f7862f6fca"
POLAR_TABLE_HASHES = {
    "Pe_BIMod2AWGN_test_D_0.20_tSigma_0.4422_Lvl_1_n_18.mat": "cdc99245125ddcf07f6641df769e6f3e8b34e5c0a044b7824713a1e121c1ff2c",
    "Pe_BIMod2AWGN_test_D_0.20_tSigma_0.4422_Lvl_2_n_18.mat": "6f18f0967024961ed12d41938774c76e0bc7c3bb41e602e7a55e95435b7c69e4",
    "Pe_BIMod2AWGN_test_D_0.20_tSigma_0.4422_Lvl_3_n_18.mat": "0ee1332951c412f4182bc6ce7b30028930c3f6e53f624a148919a0f5c6181ebd",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_artifact_hashes(workspace: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name, expected in EXPECTED_ARTIFACTS.items():
        path = workspace / name
        require(path.is_file(), f"missing artifact: {path}")
        actual = sha256_file(path)
        require(actual == expected, f"hash mismatch for {name}: {actual}")
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": actual}
    return artifacts


def acquire_opened_lock(workspace: Path, run_dir: Path) -> dict[str, Any]:
    """Irreversibly mark this frozen confirmation as opened exactly once."""
    lock_path = workspace / OPENED_LOCK_NAME
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    record = {
        "protocol": PROTOCOL,
        "opened_utc": utc_now(),
        "run_directory": str(run_dir),
        "harness_path": str(Path(__file__).resolve()),
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "seed_count": len(SEEDS),
        "seed_list_sha256": hashlib.sha256(
            b"".join(struct.pack(">I", seed) for seed in SEEDS)
        ).hexdigest(),
        "warning": "This lock is intentionally never removed, including after failure.",
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {**record, "path": str(lock_path), "sha256": sha256_file(lock_path)}


def run_logged(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    process = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    require(
        process.returncode == 0,
        f"subprocess exited {process.returncode}: {' '.join(command[:3])}",
    )


def verify_provenance(workspace: Path, polar_repo: Path, run_dir: Path) -> dict[str, Any]:
    require(len(SEEDS) == 32, "confirmation requires exactly 32 seeds")
    require(len(set(SEEDS)) == len(SEEDS), "confirmation seeds are not unique")
    require(all(0 < seed <= 0xFFFFFFFF for seed in SEEDS), "confirmation seeds must fit positive u32")
    require(set(SEEDS).isdisjoint(V1_EXCLUDED_SEEDS), "v2 reuses a v1 seed")
    actual_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "scipy": scipy.__version__,
    }
    require(actual_versions == EXPECTED_VERSIONS, f"version mismatch: {actual_versions}")
    artifacts = verify_artifact_hashes(workspace)

    manifest = read_json(workspace / "agent_polaris_sc_v2_frozen_manifest.json")
    require(manifest["frozen_before_v2_confirmation"] is True, "manifest not frozen")
    require(manifest["confirmation_opened"] is False, "manifest says confirmation opened")
    require(manifest["no_post_freeze_tuning"] is True, "manifest allows post-freeze tuning")
    require(tuple(manifest["confirmation_seeds"]) == SEEDS, "manifest seeds mismatch")
    rule = manifest["confirmation_rule"]
    require(rule["blocks"] == len(SEEDS), "manifest block count mismatch")
    require(rule["family_wise_confidence"] == FAMILY_WISE_CONFIDENCE, "manifest FWER")
    require(rule["bonferroni_metrics"] == BONFERRONI_METRICS, "manifest Bonferroni count")
    require(rule["degrees_of_freedom"] == len(SEEDS) - 1, "manifest degrees of freedom")
    require(
        abs(rule["per_metric_one_sided_confidence"] - PER_METRIC_CONFIDENCE) <= 1e-15,
        "manifest confidence mismatch",
    )
    required_upper = rule["required_upper_bounds"]
    require(required_upper["mean_logical_arithmetic_bits"] == PAYLOAD_CAPACITY_BITS_PER_BLOCK, "manifest rate threshold")
    require(abs(required_upper["absolute_mse"] - FIVE_PERCENT_TARGET) <= 1e-18, "manifest absolute threshold")
    require(abs(required_upper["sample_relative_mse"] - FIVE_PERCENT_TARGET) <= 1e-18, "manifest relative threshold")
    require(abs(manifest["distortion_gate"]["gaussian_limit_at_2p15"] - GAUSSIAN_LIMIT) <= 1e-18, "manifest Gaussian limit")
    require(abs(manifest["distortion_gate"]["maximum_ucb"] - FIVE_PERCENT_TARGET) <= 1e-18, "manifest distortion gate")
    require(manifest["novel_framing"]["payload_capacity_bits_per_block_in_global_pool"] == PAYLOAD_CAPACITY_BITS_PER_BLOCK, "manifest payload capacity")
    require(manifest["novel_framing"]["total_rank2_bits_per_block"] == RANK2_BITS_PER_BLOCK, "manifest rank2 budget")

    ledger = read_json(workspace / "agent_novel_polaris_rate_ledger_v2.json")
    whole_rate = ledger["whole_checkpoint_rate"]
    require(whole_rate["fits"] is True, "ledger fails cap")
    require(float(whole_rate["bpw"]) <= MAX_BPW, "ledger exceeds 2.15 bpw")
    framing = ledger["reservoir_framing"]
    require(framing["logical_payload_pool_bits_per_block"] == PAYLOAD_CAPACITY_BITS_PER_BLOCK, "ledger pool")
    require(framing["total_rank2_budget_bits_per_block"] == RANK2_BITS_PER_BLOCK, "ledger rank2")

    commit_process = subprocess.run(
        ["git", "-C", str(polar_repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    require(commit_process.returncode == 0, commit_process.stderr.strip())
    commit = commit_process.stdout.strip()
    require(commit == POLAR_REPOSITORY_COMMIT, f"repository commit mismatch: {commit}")
    tables: dict[str, dict[str, Any]] = {}
    for name, expected in POLAR_TABLE_HASHES.items():
        path = polar_repo / name
        require(path.is_file(), f"missing table: {path}")
        actual = sha256_file(path)
        require(actual == expected, f"table hash mismatch: {name}")
        tables[name] = {"bytes": path.stat().st_size, "sha256": actual}

    selftests: dict[str, dict[str, Any]] = {}
    for script in (
        "agent_polaris_reservoir_unpack_v2_selftest.py",
        "agent_polaris_v2_framing_selftest.py",
    ):
        stdout_path = run_dir / f"{script}.stdout.log"
        stderr_path = run_dir / f"{script}.stderr.log"
        run_logged([sys.executable, str(workspace / script)], workspace, stdout_path, stderr_path)
        selftests[script] = {
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
        }
    gpu = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    require(gpu == EXPECTED_GPU, f"GPU mismatch: {gpu}")
    return {
        "versions": actual_versions,
        "gpu": gpu,
        "artifacts": artifacts,
        "polar_repository": {"path": str(polar_repo), "commit": commit, "tables": tables},
        "selftests": selftests,
        "whole_checkpoint_rate": whole_rate,
    }


def run_encoder(workspace: Path, polar_repo: Path, run_dir: Path, seed: int) -> dict[str, Any]:
    metadata_path = run_dir / f"seed_{seed}.encoder.json"
    legacy_path = metadata_path.with_suffix(".polar.bin")
    stdout_path = run_dir / f"seed_{seed}.encoder.stdout.log"
    stderr_path = run_dir / f"seed_{seed}.encoder.stderr.log"
    command = [
        sys.executable,
        str(workspace / "agent_root_polar_lattice_gate.py"),
        "--polar-repo", str(polar_repo),
        "--trials", "1",
        "--block-length", str(N),
        "--sigma-source", repr(SIGMA_SOURCE),
        "--test-distortion", repr(TEST_DISTORTION),
        "--eta", repr(ETA),
        "--alphabet-size", str(ALPHABET_SIZE),
        "--decision", "random",
        "--seed", str(seed),
        "--emit-container-hex",
        "--output", str(metadata_path),
    ]
    run_logged(command, workspace, stdout_path, stderr_path)
    require(metadata_path.is_file(), f"seed {seed}: no metadata")
    require(legacy_path.is_file(), f"seed {seed}: no payload")
    return {"seed": seed, "metadata": metadata_path, "legacy": legacy_path}


def validate_encoder_result(run_dir: Path, seed: int, capacity_schedule: list[float]) -> dict[str, Any]:
    metadata_path = run_dir / f"seed_{seed}.encoder.json"
    legacy_path = metadata_path.with_suffix(".polar.bin")
    metadata = read_json(metadata_path)
    parameters = metadata["parameters"]
    expected_parameters = {
        "block_length": N,
        "trials": 1,
        "sigma_source": SIGMA_SOURCE,
        "test_channel_distortion": TEST_DISTORTION,
        "eta": ETA,
        "alphabet_size": ALPHABET_SIZE,
        "decision": "random",
        "seed": seed,
    }
    for key, expected in expected_parameters.items():
        require(parameters[key] == expected, f"seed {seed}: parameter {key}")
    require(parameters["capacity_schedule"] == capacity_schedule, f"seed {seed}: schedule")
    require(metadata["cupy_version"] == EXPECTED_VERSIONS["cupy"], f"seed {seed}: CuPy")
    require(len(metadata["trials"]) == 1, f"seed {seed}: trial count")
    trial = metadata["trials"][0]
    require(trial["trial"] == 0, f"seed {seed}: trial index")
    require(trial["source"] == {"kind": "synthetic_gaussian", "trial_seed": seed}, f"seed {seed}: source")
    require(trial["arithmetic_roundtrip_bits_match"] is True, f"seed {seed}: arithmetic")
    require(trial["causal_decoder_frequencies_match"] is True, f"seed {seed}: causal")
    require(trial["reconstruction_indices_match"] is True, f"seed {seed}: reconstruction")
    legacy = legacy_path.read_bytes()
    require(len(legacy) >= 8, f"seed {seed}: truncated legacy")
    header_bits, scale = struct.unpack("<If", legacy[:8])
    payload = legacy[8:]
    logical_bits = int(trial["arithmetic_logical_bits"])
    require(0 < logical_bits <= 0xFFFFFFFF, f"seed {seed}: u32 length")
    require(header_bits == logical_bits, f"seed {seed}: header length")
    require(len(payload) == (logical_bits + 7) // 8, f"seed {seed}: payload bytes")
    require(len(payload) == trial["arithmetic_payload_bytes"], f"seed {seed}: metadata bytes")
    require(hashlib.sha256(payload).hexdigest() == trial["arithmetic_payload_sha256"], f"seed {seed}: payload hash")
    require(sha256_file(legacy_path) == trial["literal_container_sha256"], f"seed {seed}: container hash")
    require(math.isfinite(scale) and scale > 0.0, f"seed {seed}: scale")
    if logical_bits % 8:
        unused = 8 - logical_bits % 8
        require((payload[-1] & ((1 << unused) - 1)) == 0, f"seed {seed}: local tail")
    return {
        "seed": seed,
        "logical_arithmetic_bits": logical_bits,
        "encoder_absolute_mse": float(trial["absolute_mse"]),
        "encoder_sample_relative_mse": float(trial["relative_mse"]),
        "legacy_container_sha256": sha256_file(legacy_path),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "metadata_sha256": sha256_file(metadata_path),
    }


def statistical_summary(values: list[float], threshold: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size == len(SEEDS), "statistic does not contain all seeds")
    df = int(array.size - 1)
    critical = float(student_t.ppf(PER_METRIC_CONFIDENCE, df))
    mean = float(array.mean(dtype=np.float64))
    sample_std = float(array.std(ddof=1, dtype=np.float64))
    standard_error = sample_std / math.sqrt(array.size)
    upper = mean + critical * standard_error
    return {
        "n": int(array.size),
        "degrees_of_freedom": df,
        "family_wise_confidence": FAMILY_WISE_CONFIDENCE,
        "bonferroni_metrics": BONFERRONI_METRICS,
        "per_metric_one_sided_confidence": PER_METRIC_CONFIDENCE,
        "student_t_critical": critical,
        "mean": mean,
        "sample_standard_deviation": sample_std,
        "standard_error": standard_error,
        "one_sided_upper_confidence_bound": upper,
        "threshold": threshold,
        "margin_below_threshold": threshold - upper,
        "passed": bool(upper <= threshold),
    }


def pack_reservoir(workspace: Path, run_dir: Path) -> dict[str, Any]:
    reservoir = run_dir / "confirmation.plrsv2"
    audit = run_dir / "confirmation.pack.json"
    inputs = [run_dir / f"seed_{seed}.encoder.polar.bin" for seed in SEEDS]
    command = [
        sys.executable,
        str(workspace / "agent_polaris_reservoir_pack_v2.py"),
        "--inputs", *[str(path) for path in inputs],
        "--output", str(reservoir),
        "--audit", str(audit),
    ]
    run_logged(
        command,
        workspace,
        run_dir / "reservoir_packer.stdout.log",
        run_dir / "reservoir_packer.stderr.log",
    )
    result = read_json(audit)
    require(result["passed"] is True, "reservoir packer failed")
    require(result["format"]["block_count"] == len(SEEDS), "packer block count")
    rate = result["rate"]
    expected_physical_bytes = (
        RESERVOIR_HEADER_BYTES
        + len(SEEDS) * DIRECTORY_BITS_PER_BLOCK // 8
        + len(SEEDS) * PAYLOAD_CAPACITY_BITS_PER_BLOCK // 8
    )
    require(reservoir.stat().st_size == expected_physical_bytes, "reservoir physical size")
    require(rate["directory_plus_physical_payload_bits"] == len(SEEDS) * RANK2_BITS_PER_BLOCK, "rank2 exact physical bits")
    require(rate["total_logical_payload_bits"] <= rate["global_payload_budget_bits"], "realized global overflow")
    return {
        "reservoir_path": reservoir,
        "audit_path": audit,
        "audit": result,
        "reservoir_sha256": sha256_file(reservoir),
        "reservoir_bytes": reservoir.stat().st_size,
    }


def unpack_reservoir(workspace: Path, run_dir: Path, packed: dict[str, Any]) -> dict[str, Any]:
    output_dir = run_dir / "independently_extracted"
    audit_path = run_dir / "confirmation.unpack.json"
    command = [
        sys.executable,
        str(workspace / "agent_polaris_reservoir_unpack_v2.py"),
        "--input", str(packed["reservoir_path"]),
        "--output-dir", str(output_dir),
        "--audit", str(audit_path),
    ]
    run_logged(
        command,
        workspace,
        run_dir / "reservoir_unpacker.stdout.log",
        run_dir / "reservoir_unpacker.stderr.log",
    )
    audit = read_json(audit_path)
    require(audit["validation"] == "passed", "unpacker validation")
    require(audit["block_count"] == len(SEEDS), "unpacker block count")
    require(audit["reservoir_sha256"] == packed["reservoir_sha256"], "unpacker reservoir hash")
    require(audit["payload_capacity_bits"] == len(SEEDS) * PAYLOAD_CAPACITY_BITS_PER_BLOCK, "unpacker capacity")
    require(len(audit["blocks"]) == len(SEEDS), "unpacker block audits")
    return {
        "output_dir": output_dir,
        "audit_path": audit_path,
        "audit": audit,
        "audit_sha256": sha256_file(audit_path),
    }


def run_decoder(workspace: Path, run_dir: Path, extracted: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    block = extracted["audit"]["blocks"][index]
    require(block["index"] == index, f"seed {seed}: extracted index")
    container = extracted["output_dir"] / block["relative_path"]
    metadata = run_dir / f"seed_{seed}.encoder.json"
    output = run_dir / f"seed_{seed}.independent_decode.json"
    command = [
        sys.executable,
        str(workspace / "agent_polaris_reservoir_block_decode_v2.py"),
        "--container", str(container),
        "--metadata", str(metadata),
        "--map", str(workspace / "agent_polaris_sc_v1_decoder_map.npz"),
        "--output", str(output),
    ]
    run_logged(
        command,
        workspace,
        run_dir / f"seed_{seed}.decoder.stdout.log",
        run_dir / f"seed_{seed}.decoder.stderr.log",
    )
    result = read_json(output)
    require(result["passed"] is True, f"seed {seed}: decoder failed")
    require(result["seed"] == seed, f"seed {seed}: decoder seed")
    require(result["logical_arithmetic_bits"] == block["logical_arithmetic_bits"], f"seed {seed}: decoded length")
    require(result["container_sha256"] == block["variable_record_sha256"], f"seed {seed}: extracted hash")
    require(result["payload_sha256"] == block["variable_payload_physical_sha256"], f"seed {seed}: extracted payload")
    return {
        "seed": seed,
        "index": index,
        "decoder": result,
        "decoder_audit_sha256": sha256_file(output),
    }


def evaluate(workspace: Path, polar_repo: Path, run_dir: Path, workers: int) -> dict[str, Any]:
    require(workers >= 1, "workers must be positive")
    require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    provenance = verify_provenance(workspace, polar_repo, run_dir)
    manifest = read_json(workspace / "agent_polaris_sc_v2_frozen_manifest.json")
    schedule = [float(value) for value in manifest["parameters"]["capacity_schedule"]]
    opened_lock = acquire_opened_lock(workspace, run_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_encoder, workspace, polar_repo, run_dir, seed) for seed in SEEDS]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    encoded = [validate_encoder_result(run_dir, seed, schedule) for seed in SEEDS]
    packed = pack_reservoir(workspace, run_dir)
    lengths = [float(row["logical_arithmetic_bits"]) for row in encoded]
    rate_gate = statistical_summary(lengths, float(PAYLOAD_CAPACITY_BITS_PER_BLOCK))
    require(rate_gate["passed"] is True, "statistical logical-rate gate failed")
    extracted = unpack_reservoir(workspace, run_dir, packed)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_decoder, workspace, run_dir, extracted, index, seed)
            for index, seed in enumerate(SEEDS)
        ]
        decoded_rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    decoded_by_seed = {row["seed"]: row for row in decoded_rows}
    require(tuple(sorted(decoded_by_seed)) == tuple(sorted(SEEDS)), "decoded seed set")

    trials: list[dict[str, Any]] = []
    absolute_values: list[float] = []
    relative_values: list[float] = []
    for index, encoder_row in enumerate(encoded):
        seed = encoder_row["seed"]
        decoded_row = decoded_by_seed[seed]
        decoded = decoded_row["decoder"]
        absolute = float(decoded["decoded_absolute_mse"])
        relative = float(decoded["decoded_sample_relative_mse"])
        require(abs(absolute - encoder_row["encoder_absolute_mse"]) <= NUMERICAL_TOLERANCE, f"seed {seed}: absolute mismatch")
        require(abs(relative - encoder_row["encoder_sample_relative_mse"]) <= NUMERICAL_TOLERANCE, f"seed {seed}: relative mismatch")
        absolute_values.append(absolute)
        relative_values.append(relative)
        block = extracted["audit"]["blocks"][index]
        trials.append(
            {
                **encoder_row,
                "reservoir_logical_start_bit": block["logical_start_bit"],
                "variable_record_sha256": block["variable_record_sha256"],
                "decoded_absolute_mse": absolute,
                "decoded_sample_relative_mse": relative,
                "reconstruction_indices_sha256": decoded["reconstruction_indices_sha256"],
                "reconstruction_fp64_sha256": decoded["reconstruction_fp64_sha256"],
                "causal_frequency_u16_sha256": decoded["causal_frequency_u16_sha256"],
                "decoder_audit_sha256": decoded_row["decoder_audit_sha256"],
            }
        )

    absolute_gate = statistical_summary(absolute_values, FIVE_PERCENT_TARGET)
    relative_gate = statistical_summary(relative_values, FIVE_PERCENT_TARGET)
    realized_rate_pass = packed["audit"]["rate"]["global_payload_headroom_bits"] >= 0
    whole_rate_pass = provenance["whole_checkpoint_rate"]["bpw"] <= MAX_BPW
    passed = bool(
        rate_gate["passed"]
        and absolute_gate["passed"]
        and relative_gate["passed"]
        and realized_rate_pass
        and whole_rate_pass
    )
    final_artifact_hashes = verify_artifact_hashes(workspace)
    absolute_gate["gaussian_limit"] = GAUSSIAN_LIMIT
    absolute_gate["ucb_excess_over_gaussian_limit_percent"] = 100.0 * (
        absolute_gate["one_sided_upper_confidence_bound"] / GAUSSIAN_LIMIT - 1.0
    )
    relative_gate["gaussian_limit"] = GAUSSIAN_LIMIT
    relative_gate["ucb_excess_over_gaussian_limit_percent"] = 100.0 * (
        relative_gate["one_sided_upper_confidence_bound"] / GAUSSIAN_LIMIT - 1.0
    )
    return {
        "protocol": PROTOCOL,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "decision_rule": (
            "Pass iff exact global serialization fits, all fresh independent decodes "
            "match, whole-checkpoint rate <=2.15 bpw, and Bonferroni-corrected "
            "family-wise 99% one-sided Student-t UCBs pass for logical rate, absolute "
            "MSE, and sample-relative MSE."
        ),
        "seeds": list(SEEDS),
        "provenance": provenance,
        "confirmation_opened_lock": opened_lock,
        "final_artifact_hashes": final_artifact_hashes,
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "reservoir": {
            "sha256": packed["reservoir_sha256"],
            "physical_bytes": packed["reservoir_bytes"],
            "pack_audit_sha256": sha256_file(packed["audit_path"]),
            "unpack_audit_sha256": extracted["audit_sha256"],
            "logical_payload_bits": packed["audit"]["rate"]["total_logical_payload_bits"],
            "fixed_capacity_bits": packed["audit"]["rate"]["global_payload_budget_bits"],
            "zero_reserve_bits": packed["audit"]["rate"]["global_zero_reserve_bits"],
            "rank2_physical_bits_excluding_global_header": packed["audit"]["rate"]["directory_plus_physical_payload_bits"],
            "realized_rate_pass": bool(realized_rate_pass),
        },
        "whole_checkpoint_rate_pass": bool(whole_rate_pass),
        "logical_arithmetic_rate_gate": rate_gate,
        "absolute_mse_gate": absolute_gate,
        "sample_relative_mse_gate": relative_gate,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--polar-repo", type=Path, default=Path("/root/PolarLatticeQuantization"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = utc_now()
    try:
        result = evaluate(
            args.workspace.resolve(),
            args.polar_repo.resolve(),
            args.run_dir.resolve(),
            args.workers,
        )
    except BaseException as error:
        result = {
            "protocol": PROTOCOL,
            "status": "failed",
            "passed": False,
            "hard_failure": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "seeds": list(SEEDS),
        }
        lock_path = args.workspace.resolve() / OPENED_LOCK_NAME
        if lock_path.is_file():
            result["confirmation_opened_lock"] = {
                "path": str(lock_path),
                "sha256": sha256_file(lock_path),
            }
        try:
            result["final_artifact_hashes"] = verify_artifact_hashes(
                args.workspace.resolve()
            )
        except BaseException as hash_error:
            result["final_artifact_recheck_failure"] = (
                f"{type(hash_error).__name__}: {hash_error}"
            )
    result["started_utc"] = started
    result["finished_utc"] = utc_now()
    write_json(args.summary, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
