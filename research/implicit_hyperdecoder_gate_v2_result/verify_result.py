#!/usr/bin/env python3
"""Source-free verifier for the completed SILWARP-v2 negative result."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent / "implicit_hyperdecoder_gate_v2"
sys.path.insert(0, str(CANDIDATE))

import silwarp_common as sw  # noqa: E402


EXPECTED = {
    "PROTOCOL.md": "8438764d4e2d3f272afe40fe38860a42b47406d4dcd4db404107d14352738daf",
    "protocol_lock.json": "c3f896b9beaf12c30f49d05d791288f0b32a639385e225628211df06bd8e1dd9",
    "source_lock.json": "b0dc982e42a22fa960a1436ed5ebdcab1233d0bc2e463972e16dab4315e14042",
    "silwarp_common.py": "09c9091af31871d715e2270254413503157e0035efb217206cd480aacb057ade",
    "silwarp_gate.py": "c7720dd2557b1009b9947eb80f7e57921f8a77e2399f8848f6ce4ba218696be7",
    "test_source_free.py": "33c7cfb1c41b99bd06035687f14f4abadda0d2ce99ad7ee5791e31cb1d76096a",
    "gpu_preflight_rtx5090_v2.json": "4c76d310d1f9580e6ce28703e593de7fca92123dd9921384196d0897117cb1c7",
    "launch_sentinel_v2.json": "75a6a19c7e57cc754e82984700584330ae86af8ea82d99941b7348898a872e8e",
    "checkpoint_000256.json": "0037de00c4c9faa4f2a8e57ebabb6dcb5c094c860ac8f44824ce64277333b8ce",
    "checkpoint_000512.json": "4c849972bc63e66ad08601c1bc0b65209836705c51b086258c6687ef5f96c01e",
    "result.json": "c21567d337f6349f31c8c8f0eaa2a544f54a772afceb3cf607bcc10119a84592",
    "run.log": "0fd2599a009ac6c62e783738711e4d3ec494512f9d40fb1fb700b345fe73484c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path.name} must contain one JSON object")
    return value


def verify_seal(value: dict, field: str) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop(field)
    actual = hashlib.sha256(sw.canonical_json_bytes(unsigned)).hexdigest()
    assert claimed == actual, (field, claimed, actual)


def assert_finite_json(value: object) -> None:
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            assert_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            assert_finite_json(child)


def verify_candidate_and_evidence_hashes() -> None:
    for name in (
        "PROTOCOL.md",
        "protocol_lock.json",
        "source_lock.json",
        "silwarp_common.py",
        "silwarp_gate.py",
        "test_source_free.py",
    ):
        assert sha256(CANDIDATE / name) == EXPECTED[name]
    for name in (
        "gpu_preflight_rtx5090_v2.json",
        "launch_sentinel_v2.json",
        "checkpoint_000256.json",
        "checkpoint_000512.json",
        "result.json",
        "run.log",
    ):
        assert sha256(HERE / name) == EXPECTED[name]


def verify_gpu_and_sentinel() -> None:
    gpu = load_json(HERE / "gpu_preflight_rtx5090_v2.json")
    assert_finite_json(gpu)
    assert gpu["schema"] == "silwarp_gpu_source_free_preflight_v2"
    assert gpu["status"] == "PASS_GPU_SOURCE_FREE_PREFLIGHT"
    assert gpu["payload_opened"] is False
    assert gpu["cuda_imported"] is True
    assert gpu["identity_exact"] is True
    assert gpu["counter_replay_exact"] is True
    assert gpu["production_feasibility"]["resident_model_adam_cells"] == 6
    assert gpu["protocol_sha256"] == EXPECTED["protocol_lock.json"]
    assert gpu["source_lock_sha256"] == EXPECTED["source_lock.json"]
    assert gpu["runner_sha256"] == EXPECTED["silwarp_gate.py"]
    assert gpu["common_sha256"] == EXPECTED["silwarp_common.py"]

    sentinel_path = HERE / "launch_sentinel_v2.json"
    sentinel = sw.validate_launch_sentinel(
        sentinel_path,
        CANDIDATE / "silwarp_gate.py",
        CANDIDATE / "silwarp_common.py",
    )
    assert sentinel["pinned_panel_authorized"] is False
    assert sentinel["confirmation_numeric_access_before_promotion"] is False


def verify_checkpoints(result: dict) -> None:
    previous = None
    for update in (256, 512):
        path = HERE / f"checkpoint_{update:06d}.json"
        checkpoint = load_json(path)
        assert_finite_json(checkpoint)
        verify_seal(checkpoint, "internal_seal_sha256")
        assert checkpoint["schema"] == "silwarp_training_checkpoint_v2"
        assert checkpoint["update"] == update
        assert checkpoint["adam_step"] == update
        assert checkpoint["counter_randomness"] is True
        assert checkpoint["state_file"] == "state.npz"
        assert checkpoint["state_arrays"] == 216
        assert len(checkpoint["state_sha256"]) == 64
        assert checkpoint["bindings"]["protocol_sha256"] == EXPECTED["protocol_lock.json"]
        assert checkpoint["bindings"]["source_lock_sha256"] == EXPECTED["source_lock.json"]
        assert checkpoint["bindings"]["runner_sha256"] == EXPECTED["silwarp_gate.py"]
        assert checkpoint["bindings"]["common_sha256"] == EXPECTED["silwarp_common.py"]
        assert checkpoint["bindings"]["launch_sentinel_sha256"] == EXPECTED["launch_sentinel_v2.json"]
        assert set(checkpoint["history"]) == {str(seed) for seed in sw.TRAINING_SEEDS}
        expected_points = {"256"} if update == 256 else {"256", "512"}
        for seed in sw.TRAINING_SEEDS:
            assert set(checkpoint["history"][str(seed)]) == expected_points
            for point in expected_points:
                assert checkpoint["history"][str(seed)][point] == result["history"][str(seed)][point]
        if previous is None:
            assert checkpoint["predecessor"] is None
        else:
            assert checkpoint["predecessor"] == {
                "update": 256,
                "metadata_sha256": EXPECTED["checkpoint_000256.json"],
                "state_sha256": previous["state_sha256"],
            }
        previous = checkpoint


def verify_result() -> dict:
    result = load_json(HERE / "result.json")
    assert_finite_json(result)
    verify_seal(result, "canonical_unsigned_sha256")
    assert result["schema"] == "silwarp_auxiliary_result_v2"
    assert result["decision"] == "HARD_KILL_FROZEN_SILWARP_CELL_AT_UPDATE_512"
    assert result["stopped_early"] is True
    assert result["stop_update"] == result["final_update"] == 512
    assert result["promotion_evidence"]["checks"] == {"update_512_hard_kill": True}
    assert result["protocol_sha256"] == EXPECTED["protocol_lock.json"]
    assert result["source_lock_sha256"] == EXPECTED["source_lock.json"]
    assert result["runner_sha256"] == EXPECTED["silwarp_gate.py"]
    assert result["common_sha256"] == EXPECTED["silwarp_common.py"]
    assert result["launch_sentinel"]["sha256"] == EXPECTED["launch_sentinel_v2.json"]
    assert result["confirmation_opened"] is False
    assert result["confirmation"] is None
    assert result["source_hashes"]["confirmation"] == {}
    assert result["pinned_panel"] == {"opened": False, "permitted": False}
    assert 2.15 <= result["ledger_128"]["production_physical_bpw"] <= 2.5
    assert result["ledger_128"]["production_cold_read_amplification"] < 2.0

    assert set(result["history"]) == {str(seed) for seed in sw.TRAINING_SEEDS}
    stop_input = {}
    for seed in sw.TRAINING_SEEDS:
        points = result["history"][str(seed)]
        assert set(points) == {"256", "512"}
        stop_input[seed] = {}
        for update in (256, 512):
            evaluation = points[str(update)]
            assert evaluation["update"] == update
            assert set(evaluation["corpora"]) == {"qwen", "null_a", "null_b"}
            assert evaluation["matched"]["s_match_worst"] == 0.0
            assert evaluation["matched"]["cluster_se"] == 0.0
            for corpus in ("qwen", "null_a", "null_b"):
                report = evaluation["corpora"][corpus]
                aggregate = report["aggregate"]
                assert aggregate["relative_to_identity_q"] == 1.0
                assert aggregate["s_absolute_from_identity"] == 0.0
                assert report["selected_sse"] == report["identity_sse"]
                assert all(matrix["bypass_identity"] is True for matrix in report["matrices"])
                assert all(matrix["selected_sse"] == matrix["identity_sse"] for matrix in report["matrices"])
            stop_input[seed][update] = {
                "s_match_worst": evaluation["matched"]["s_match_worst"],
                "cluster_se": evaluation["matched"]["cluster_se"],
            }
    assert sw.hard_kill_at_512(stop_input) is True
    return result


def verify_log(result: dict) -> None:
    rows = [json.loads(line) for line in (HERE / "run.log").read_text(encoding="utf-8").splitlines() if line]
    source_rows = [row for row in rows if row.get("event") == "source_loaded"]
    checkpoints = [row for row in rows if row.get("event") == "checkpoint"]
    decisions = [row for row in rows if "decision" in row]
    assert len(source_rows) == 98
    assert len(checkpoints) == 2
    assert [row["update"] for row in checkpoints] == [256, 512]
    assert len(decisions) == 1
    assert decisions[0]["decision"] == result["decision"]
    assert not any(row.get("split") == "confirmation" for row in source_rows)
    seen = {"fit": set(), "calibration": set()}
    for row in source_rows:
        split = row["split"]
        assert split in seen
        key = tuple(row["key"])
        key_text = repr(key)
        assert row["sha256"] == result["source_hashes"][split][key_text]
        seen[split].add(key_text)
        assert 0.0 < row["normalized_second_moment"] <= 1.0
    assert seen["fit"] == set(result["source_hashes"]["fit"])
    assert seen["calibration"] == set(result["source_hashes"]["calibration"])


def main() -> None:
    verify_candidate_and_evidence_hashes()
    verify_gpu_and_sentinel()
    result = verify_result()
    verify_checkpoints(result)
    verify_log(result)
    print(
        json.dumps(
            {
                "status": "PASS_SILWARP_V2_NEGATIVE_RESULT",
                "decision": result["decision"],
                "result_sha256": EXPECTED["result.json"],
                "confirmation_opened": False,
                "pinned_panel_opened": False,
                "verified_checkpoints": [256, 512],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
