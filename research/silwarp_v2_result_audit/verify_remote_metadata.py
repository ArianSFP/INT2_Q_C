#!/usr/bin/env python3
"""Metadata-only verifier for the frozen SILWARP-v2 auxiliary result.

This script is intentionally limited to the frozen candidate, the completed
run directory, and its JSONL log.  It hashes binary result/checkpoint artifacts
but never opens a Qwen source tensor and never imports CuPy or initializes CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


FROZEN_CANDIDATE_SHA256 = {
    "PROTOCOL.md": "8438764d4e2d3f272afe40fe38860a42b47406d4dcd4db404107d14352738daf",
    "protocol_lock.json": "c3f896b9beaf12c30f49d05d791288f0b32a639385e225628211df06bd8e1dd9",
    "source_lock.json": "b0dc982e42a22fa960a1436ed5ebdcab1233d0bc2e463972e16dab4315e14042",
    "silwarp_common.py": "09c9091af31871d715e2270254413503157e0035efb217206cd480aacb057ade",
    "silwarp_gate.py": "c7720dd2557b1009b9947eb80f7e57921f8a77e2399f8848f6ce4ba218696be7",
    "test_source_free.py": "33c7cfb1c41b99bd06035687f14f4abadda0d2ce99ad7ee5791e31cb1d76096a",
    "gpu_preflight_rtx5090_v2.json": "4c76d310d1f9580e6ce28703e593de7fca92123dd9921384196d0897117cb1c7",
    "launch_sentinel_v2.json": "75a6a19c7e57cc754e82984700584330ae86af8ea82d99941b7348898a872e8e",
}

EXPECTED_RUN_FILES = {
    "checkpoint_000256/checkpoint.json": (53232, "0037de00c4c9faa4f2a8e57ebabb6dcb5c094c860ac8f44824ce64277333b8ce"),
    "checkpoint_000256/state.npz": (17036014, "686cbfcc37976272e510aeadf2f421246a94e85bd2af847ce5f3a02eb62132b7"),
    "checkpoint_000512/checkpoint.json": (105249, "4c849972bc63e66ad08601c1bc0b65209836705c51b086258c6687ef5f96c01e"),
    "checkpoint_000512/state.npz": (17036014, "6e65c3ac5b34a17fadf547dbea22aa8a9f40835042a1bff6b3c98ed187a8d109"),
    "model_seed_26090131_null_a.fp16.bin": (475654, "2e78e5847f0322033948845a2feeecbb90f117af72c59c4033f49526db631cf3"),
    "model_seed_26090131_null_b.fp16.bin": (475654, "5f28e871436b1ca10ae302492b5db1008abcffb03c0964967d0c98fb408cb4c1"),
    "model_seed_26090131_qwen.fp16.bin": (475654, "4381807cc593c8391ffc83d69b89de713c5660e839cbacf656895a6455e8fa7b"),
    "model_seed_26090179_null_a.fp16.bin": (475654, "18c4097bbc5108fd41f45bd172d0d8ea724b00828e323e27c445566acff9ceeb"),
    "model_seed_26090179_null_b.fp16.bin": (475654, "31e3bdda21a1c0de5de063d14b516d3f7b901238c5ba92c63e78e7fc63a7dbd3"),
    "model_seed_26090179_qwen.fp16.bin": (475654, "93926d164655bcd78283d1679efb4f06017c3de01f86f155f2805d4b50cd7a58"),
    "result.json": (120093, "c21567d337f6349f31c8c8f0eaa2a544f54a772afceb3cf607bcc10119a84592"),
}

EXPECTED_LOG = (20094, "0fd2599a009ac6c62e783738711e4d3ec494512f9d40fb1fb700b345fe73484c")
TRAINING_SEEDS = (26090131, 26090179)
CORPORA = ("qwen", "null_a", "null_b")
PARAMETERS = (
    "Wy", "Wc", "b0", "A", "ba", "C", "bc", "B", "bb", "Wo", "bo", "role_gain"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_finite(child) for child in value.values())
    if isinstance(value, list):
        return all(all_finite(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    run = args.run.resolve()
    log_path = args.log.resolve()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    def check(name: str, condition: bool) -> None:
        checks[name] = bool(condition)

    check("candidate_is_directory", candidate.is_dir() and not candidate.is_symlink())
    check("run_is_directory", run.is_dir() and not run.is_symlink())
    check("log_is_regular", log_path.is_file() and not log_path.is_symlink())
    check("candidate_tree_has_no_symlinks", not any(path.is_symlink() for path in candidate.rglob("*")))
    check("run_tree_has_no_symlinks", not any(path.is_symlink() for path in run.rglob("*")))

    candidate_hashes = {
        name: sha256_file(candidate / name) for name in FROZEN_CANDIDATE_SHA256
    }
    check("frozen_candidate_hashes", candidate_hashes == FROZEN_CANDIDATE_SHA256)
    details["candidate_hashes"] = candidate_hashes

    actual_run_files = sorted(
        path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file()
    )
    check("run_inventory_exact", set(actual_run_files) == set(EXPECTED_RUN_FILES))
    run_inventory = {}
    for relative, (expected_bytes, expected_hash) in sorted(EXPECTED_RUN_FILES.items()):
        path = run / relative
        actual_bytes = path.stat().st_size
        actual_hash = sha256_file(path)
        run_inventory[relative] = {"bytes": actual_bytes, "sha256": actual_hash}
        check(f"run_file:{relative}", actual_bytes == expected_bytes and actual_hash == expected_hash)
    details["run_inventory"] = run_inventory

    log_raw = log_path.read_bytes()
    check("log_size_and_hash", len(log_raw) == EXPECTED_LOG[0] and hashlib.sha256(log_raw).hexdigest() == EXPECTED_LOG[1])
    log_rows = [json.loads(line) for line in log_raw.decode("utf-8").splitlines()]
    event_counts = Counter(row.get("event", "terminal") for row in log_rows)
    split_counts = Counter(
        row["split"] for row in log_rows if row.get("event") == "source_loaded"
    )
    checkpoint_updates = [
        row["update"] for row in log_rows if row.get("event") == "checkpoint"
    ]
    terminal_rows = [row for row in log_rows if "decision" in row]
    check("log_event_inventory", event_counts == {"source_loaded": 98, "checkpoint": 2, "terminal": 1})
    check("log_source_splits", split_counts == {"fit": 82, "calibration": 16})
    check("log_checkpoint_updates", checkpoint_updates == [256, 512])
    check("log_terminal_decision", len(terminal_rows) == 1 and terminal_rows[0]["decision"] == "HARD_KILL_FROZEN_SILWARP_CELL_AT_UPDATE_512")
    check("log_has_no_confirmation_or_pinned_text", b"confirmation" not in log_raw.lower() and b"pinned" not in log_raw.lower())
    details["log"] = {
        "bytes": len(log_raw),
        "sha256": hashlib.sha256(log_raw).hexdigest(),
        "line_count": len(log_rows),
        "event_counts": dict(event_counts),
        "source_split_counts": dict(split_counts),
        "checkpoint_updates": checkpoint_updates,
    }

    protocol = load_json(candidate / "protocol_lock.json")
    source_lock = load_json(candidate / "source_lock.json")
    result = load_json(run / "result.json")
    check("result_all_finite", all_finite(result))
    unsigned_result = dict(result)
    claimed_result_seal = unsigned_result.pop("canonical_unsigned_sha256")
    computed_result_seal = canonical_sha256(unsigned_result)
    check("result_internal_seal", claimed_result_seal == computed_result_seal)
    check("result_schema", result.get("schema") == "silwarp_auxiliary_result_v2")
    check("result_frozen_bindings", result.get("protocol_sha256") == candidate_hashes["protocol_lock.json"] and result.get("source_lock_sha256") == candidate_hashes["source_lock.json"] and result.get("common_sha256") == candidate_hashes["silwarp_common.py"] and result.get("runner_sha256") == candidate_hashes["silwarp_gate.py"])
    check("result_protocol_sections", result.get("architecture") == protocol.get("architecture") and result.get("information_channel") == protocol.get("information_channel"))
    details["result"] = {
        "bytes": (run / "result.json").stat().st_size,
        "sha256": sha256_file(run / "result.json"),
        "claimed_unsigned_sha256": claimed_result_seal,
        "computed_unsigned_sha256": computed_result_seal,
        "schema": result.get("schema"),
        "decision": result.get("decision"),
    }

    sentinel = load_json(candidate / "launch_sentinel_v2.json")
    unsigned_sentinel = dict(sentinel)
    claimed_sentinel_seal = unsigned_sentinel.pop("internal_seal_sha256")
    check("sentinel_internal_seal", claimed_sentinel_seal == canonical_sha256(unsigned_sentinel))
    check("sentinel_flags", sentinel.get("pinned_panel_authorized") is False and sentinel.get("confirmation_numeric_access_before_promotion") is False)
    check("sentinel_result_binding", result.get("launch_sentinel", {}).get("sha256") == candidate_hashes["launch_sentinel_v2.json"] and result.get("launch_sentinel", {}).get("internal_seal_sha256") == claimed_sentinel_seal)
    check("sentinel_code_bindings", sentinel.get("protocol_sha256") == candidate_hashes["protocol_lock.json"] and sentinel.get("source_lock_sha256") == candidate_hashes["source_lock.json"] and sentinel.get("common_sha256") == candidate_hashes["silwarp_common.py"] and sentinel.get("runner_sha256") == candidate_hashes["silwarp_gate.py"])

    gpu_receipt = load_json(candidate / "gpu_preflight_rtx5090_v2.json")
    check("gpu_preflight_pass", gpu_receipt.get("status") == "PASS_GPU_SOURCE_FREE_PREFLIGHT" and gpu_receipt.get("payload_opened") is False and gpu_receipt.get("cuda_imported") is True)
    check("gpu_preflight_code_bindings", gpu_receipt.get("protocol_sha256") == candidate_hashes["protocol_lock.json"] and gpu_receipt.get("source_lock_sha256") == candidate_hashes["source_lock.json"] and gpu_receipt.get("common_sha256") == candidate_hashes["silwarp_common.py"] and gpu_receipt.get("runner_sha256") == candidate_hashes["silwarp_gate.py"])
    check("gpu_preflight_runtime_binding", gpu_receipt.get("runtime_identity") == result.get("backend", {}).get("runtime_identity"))

    expected_bindings = {
        "protocol_sha256": result["protocol_sha256"],
        "source_lock_sha256": result["source_lock_sha256"],
        "runner_sha256": result["runner_sha256"],
        "common_sha256": result["common_sha256"],
        "launch_sentinel_sha256": result["launch_sentinel"]["sha256"],
        "runtime_identity": result["backend"]["runtime_identity"],
    }
    expected_state_members = {
        f"{prefix}__{seed}__{corpus}__{parameter}.npy"
        for prefix in ("p", "m", "v")
        for seed in TRAINING_SEEDS
        for corpus in CORPORA
        for parameter in PARAMETERS
    }
    checkpoint_rows = []
    predecessor = None
    for update in (256, 512):
        directory = run / f"checkpoint_{update:06d}"
        metadata_path = directory / "checkpoint.json"
        state_path = directory / "state.npz"
        metadata_raw = metadata_path.read_bytes()
        metadata_hash = hashlib.sha256(metadata_raw).hexdigest()
        metadata = json.loads(metadata_raw)
        unsigned_metadata = dict(metadata)
        claimed_seal = unsigned_metadata.pop("internal_seal_sha256")
        state_hash = sha256_file(state_path)
        with zipfile.ZipFile(state_path) as archive:
            members = archive.namelist()
        history_prefix = {
            str(seed): {
                point: value
                for point, value in result["history"][str(seed)].items()
                if int(point) <= update
            }
            for seed in TRAINING_SEEDS
        }
        check(f"checkpoint_{update}:seal", claimed_seal == canonical_sha256(unsigned_metadata))
        check(f"checkpoint_{update}:schema", metadata.get("schema") == "silwarp_training_checkpoint_v2" and metadata.get("update") == update and metadata.get("adam_step") == update)
        check(f"checkpoint_{update}:bindings", metadata.get("bindings") == expected_bindings)
        check(f"checkpoint_{update}:state_hash", metadata.get("state_file") == "state.npz" and metadata.get("state_sha256") == state_hash)
        check(f"checkpoint_{update}:state_inventory", metadata.get("state_arrays") == 216 and len(members) == len(set(members)) == 216 and set(members) == expected_state_members)
        check(f"checkpoint_{update}:predecessor", metadata.get("predecessor") == predecessor)
        check(f"checkpoint_{update}:history", metadata.get("history") == history_prefix)
        check(f"checkpoint_{update}:normalizer", metadata.get("log_rms_center_fp16") == result["log_rms_normalizer"]["center_fp16"] and metadata.get("log_rms_scale_fp16") == result["log_rms_normalizer"]["scale_fp16"])
        check(f"checkpoint_{update}:counter_randomness", metadata.get("counter_randomness") is True)
        checkpoint_rows.append({
            "update": update,
            "metadata_sha256": metadata_hash,
            "state_sha256": state_hash,
            "internal_seal_sha256": claimed_seal,
            "state_arrays": len(members),
        })
        predecessor = {
            "update": update,
            "metadata_sha256": metadata_hash,
            "state_sha256": state_hash,
        }
    details["checkpoints"] = checkpoint_rows

    final_models = {
        f"model_seed_{seed}_{corpus}.fp16.bin": metadata
        for seed, seed_history in result["history"].items()
        for corpus, metadata in seed_history["512"]["models"].items()
    }
    check("final_model_inventory_matches_result", set(final_models) == set(result["artifacts"]))
    model_rows = []
    for name, history_metadata in sorted(final_models.items()):
        raw = (run / name).read_bytes()
        header_region = raw[:4096]
        json_region, separator, padding = header_region.partition(b"\n")
        header = json.loads(json_region.decode("ascii"))
        parameter_payload = raw[4096:]
        file_hash = hashlib.sha256(raw).hexdigest()
        check(f"model:{name}:result_hash", result["artifacts"][name] == {"bytes": len(raw), "sha256": file_hash} and history_metadata["bytes"] == len(raw) and history_metadata["sha256"] == file_hash)
        check(f"model:{name}:header", bool(separator) and not any(padding) and header.get("magic") == "SILWARP_MODEL_V2" and header.get("schema") == "silwarp_auxiliary_protocol_v2")
        check(f"model:{name}:bindings", header.get("protocol_sha256") == result["protocol_sha256"] and header.get("source_lock_sha256") == result["source_lock_sha256"])
        check(f"model:{name}:payload", len(parameter_payload) == 471558 and header.get("parameter_payload_bytes") == len(parameter_payload) and header.get("parameter_payload_sha256") == hashlib.sha256(parameter_payload).hexdigest() == history_metadata["parameter_payload_sha256"])
        model_rows.append({
            "name": name,
            "bytes": len(raw),
            "sha256": file_hash,
            "parameter_payload_sha256": hashlib.sha256(parameter_payload).hexdigest(),
        })
    details["models"] = model_rows

    tensor_pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight\.bf16\.bin"
    )
    locked_hashes = {"fit": {}, "calibration": {}, "confirmation": {}}
    for row in source_lock["files"]:
        match = tensor_pattern.fullmatch(row["filename"])
        if match is None:
            raise ValueError(f"unexpected source-lock filename: {row['filename']}")
        key = str((int(match.group(1)), int(match.group(2)), match.group(3)))
        locked_hashes[row["split"]][key] = row["sha256"]
    check("fit_source_hashes_match_lock", result["source_hashes"]["fit"] == locked_hashes["fit"])
    check("calibration_source_hashes_match_lock", result["source_hashes"]["calibration"] == locked_hashes["calibration"])
    check("confirmation_source_hashes_empty", result["source_hashes"]["confirmation"] == {})
    check("confirmation_and_pinned_closed", result.get("confirmation_opened") is False and result.get("confirmation") is None and result.get("pinned_panel") == {"opened": False, "permitted": False})

    hard_kill_rows = []
    hard_kill = True
    all_bypassed = True
    for seed in TRAINING_SEEDS:
        history = result["history"][str(seed)]
        u256 = history["256"]["matched"]["s_match_worst"] + 2.0 * history["256"]["matched"]["cluster_se"]
        u512 = history["512"]["matched"]["s_match_worst"] + 2.0 * history["512"]["matched"]["cluster_se"]
        delta = u512 - u256
        seed_passes = u512 < 0.10 and delta < 0.012
        hard_kill &= seed_passes
        hard_kill_rows.append({
            "seed": seed,
            "U256": u256,
            "U512": u512,
            "delta": delta,
            "U512_lt_0.10": u512 < 0.10,
            "delta_lt_0.012": delta < 0.012,
        })
        for update in ("256", "512"):
            for corpus in CORPORA:
                corpus_result = history[update]["corpora"][corpus]
                all_bypassed &= corpus_result["aggregate"]["relative_to_identity_q"] == 1.0
                all_bypassed &= len(corpus_result["matrices"]) == 16
                all_bypassed &= all(
                    row["bypass_identity"] is True and row["selected_sse"] == row["identity_sse"]
                    for row in corpus_result["matrices"]
                )
    check("hard_kill_replay", hard_kill)
    check("hard_kill_decision", hard_kill and result.get("decision") == "HARD_KILL_FROZEN_SILWARP_CELL_AT_UPDATE_512" and result.get("stopped_early") is True and result.get("stop_update") == 512 and result.get("final_update") == 512)
    check("all_calibration_rows_bypassed", all_bypassed)
    details["hard_kill"] = {
        "rule": "for both seeds: U512 < 0.10 and U512-U256 < 0.012",
        "rows": hard_kill_rows,
        "all_192_matrix_evaluations_selected_identity": all_bypassed,
    }

    all_pass = all(checks.values())
    report = {
        "schema": "silwarp_v2_metadata_verification_v1",
        "all_pass": all_pass,
        "checks": checks,
        "details": details,
        "access_boundary": {
            "qwen_tensor_payload_opened": False,
            "cuda_imported_or_initialized": False,
            "checkpoint_numeric_arrays_deserialized": False,
            "binary_artifacts_streamed_for_hash_only": True,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"all_pass": False, "fatal_error": repr(error)}, sort_keys=True))
        raise
