"""Read-only validator for the sole authorized r3 source-free preflight receipt."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat


FRESH_PARENT = Path(
    "/tmp/codex_cbib1_r3_source_free_rtx5090_preflight_20260903_5bac3594"
)
REVIEW_ROOT = FRESH_PARENT / "review"
RECEIPT_PATH = FRESH_PARENT / "receipt.json"
CLAIM_PATH = FRESH_PARENT / "ONE_USE_PREFLIGHT_CLAIM.json"
CHILD_STDERR_PATH = FRESH_PARENT / "child_stderr.txt"
WRAPPER_STATUS_PATH = FRESH_PARENT / "wrapper_status.json"
DEPLOYMENT_MANIFEST_SHA256 = (
    "5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f"
)
AUTHORITY_SHA256 = "ff38022186f39d610a9108e0926be0263b4fad065146725e9311a637219371a2"
WRAPPER_SHA256 = "91d55282c1d84b3e108e54c9772267a234cba7aeb8784504183badc90b25d71e"
FIXTURE_SHA256 = "33f7ba9d4ae0589d06abcfab06bac46d06ef75188d714350ba993df0ca9bbab5"


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def real_file(path: Path) -> None:
    need(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
         f"regular nonsymlink file required: {path.name}")


def main() -> int:
    need(Path(__file__).resolve(strict=True).parent == REVIEW_ROOT, "fixed review root")
    need(sha(REVIEW_ROOT / "AUTHORIZED_PREFLIGHT.json") == AUTHORITY_SHA256,
         "authority digest")
    need(sha(REVIEW_ROOT / "run_authorized_preflight_once.py") == WRAPPER_SHA256,
         "wrapper digest")
    for path in (RECEIPT_PATH, CLAIM_PATH, CHILD_STDERR_PATH, WRAPPER_STATUS_PATH):
        real_file(path)
    claim = json.loads(CLAIM_PATH.read_bytes())
    need(claim.get("schema") ==
         "same-layer-clustered-ib-r3-source-free-preflight-claim-v0" and
         claim.get("status") == "ATTEMPT_CONSUMED_BEFORE_VALIDATION_AND_CHILD_SPAWN" and
         claim.get("deployment_manifest_sha256") == DEPLOYMENT_MANIFEST_SHA256,
         "persistent claim")
    wrapper = json.loads(WRAPPER_STATUS_PATH.read_bytes())
    need(wrapper.get("schema") ==
         "same-layer-clustered-ib-r3-source-free-preflight-wrapper-v0" and
         wrapper.get("status") == "PASS_SINGLE_ATTEMPT_CAPTURED_AND_VALIDATED" and
         wrapper.get("deployment_manifest_sha256") == DEPLOYMENT_MANIFEST_SHA256 and
         wrapper.get("receipt_sha256") == sha(RECEIPT_PATH), "wrapper status")
    raw = RECEIPT_PATH.read_bytes()
    lines = raw.splitlines()
    need(len(lines) == 1 and bool(lines[0]), "exactly one JSON receipt line")
    result = json.loads(lines[0])
    expected_top = {
        "all_controls_executed", "control_count", "cuda_driver", "cuda_runtime",
        "cupy_version", "deployment_manifest_sha256", "detailed_assignments_counts_partitions",
        "device_name", "fixture_labels_sha256", "full_gate_exact_and_float_parity",
        "full_gate_status", "numpy_file", "numpy_version", "pairwise_mi_max_absolute_delta",
        "payload_or_qwen_accessed", "production_geometry", "quantizer_scales_labels_exact",
        "schema", "source_read_survivor_endpoints", "source_read_survivor_group_sizes",
        "status",
    }
    need(set(result) == expected_top, "receipt fields")
    need(result["schema"] ==
         "same-layer-clustered-ib-qwen-deployment-source-free-cupy-v0-r3" and
         result["status"] == "PASS_PRODUCTION_GEOMETRY_FULL_CPU_CUPY_PARITY" and
         result["deployment_manifest_sha256"] == DEPLOYMENT_MANIFEST_SHA256 and
         result["payload_or_qwen_accessed"] is False, "receipt identity")
    geometry = result["production_geometry"]
    need(set(geometry) == {"coordinates_per_role", "experts", "fold_coordinate_counts",
                           "fold_count", "group_sizes", "roles", "scale_bytes_per_expert",
                           "superblock_values"} and
         geometry["experts"] == 16 and geometry["roles"] == 2 and
         geometry["coordinates_per_role"] == 131072 and geometry["fold_count"] == 8 and
         geometry["superblock_values"] == 2048 and geometry["scale_bytes_per_expert"] == 256 and
         geometry["group_sizes"] == [2, 4, 8, 16] and
         geometry["fold_coordinate_counts"] == [16384] * 8, "production geometry")
    need(result["quantizer_scales_labels_exact"] is True and
         math.isfinite(float(result["pairwise_mi_max_absolute_delta"])) and
         float(result["pairwise_mi_max_absolute_delta"]) <= 1e-12,
         "quantizer/pairwise parity")
    detailed = result["detailed_assignments_counts_partitions"]
    need(set(detailed) == {"conditional_count_arrays_checked", "groups_checked",
                           "heldout_assignments_checked", "latent_count_arrays_checked",
                           "max_model_float_absolute_delta", "models_checked",
                           "training_assignments_checked"} and
         detailed["groups_checked"] == 120 and detailed["models_checked"] == 240 and
         detailed["training_assignments_checked"] == 27525120 and
         detailed["heldout_assignments_checked"] == 3932160 and
         detailed["latent_count_arrays_checked"] == 480 and
         detailed["conditional_count_arrays_checked"] == 480 and
         math.isfinite(float(detailed["max_model_float_absolute_delta"])) and
         float(detailed["max_model_float_absolute_delta"]) <= 1e-8,
         "detailed parity coverage")
    gate = result["full_gate_exact_and_float_parity"]
    need(set(gate) == {"exact_field_count", "float_field_count",
                       "max_float_absolute_delta", "max_float_path"} and
         isinstance(gate["exact_field_count"], int) and gate["exact_field_count"] > 0 and
         isinstance(gate["float_field_count"], int) and gate["float_field_count"] > 0 and
         math.isfinite(float(gate["max_float_absolute_delta"])) and
         float(gate["max_float_absolute_delta"]) <= 1e-8 and
         isinstance(gate["max_float_path"], str), "full gate parity")
    survivors = result["source_read_survivor_group_sizes"]
    endpoints = result["source_read_survivor_endpoints"]
    need(isinstance(survivors, list) and 2 in survivors and
         all(item in (2, 4, 8, 16) for item in survivors) and
         set(endpoints) == {str(item) for item in survivors} and
         "5/2" in endpoints["2"], "source/read survivor")
    need(result["full_gate_status"] == "HARD_KILL_CHARGED_OR_CONTROLS_BELOW_TARGET" and
         result["all_controls_executed"] is True and result["control_count"] == 8,
         "control execution/disposition")
    need(result["fixture_labels_sha256"] == FIXTURE_SHA256 and
         result["numpy_version"] == "2.5.2" and
         result["numpy_file"] ==
         "/workspace/int2-cupy-venv/lib/python3.12/site-packages/numpy/__init__.py" and
         result["cupy_version"] == "14.2.0" and
         result["device_name"] == "NVIDIA GeForce RTX 5090" and
         result["cuda_runtime"] == 12090 and result["cuda_driver"] == 13000,
         "runtime/fixture identity")
    print(json.dumps({
        "authorized_attempts_remaining": 0,
        "child_stderr_bytes": CHILD_STDERR_PATH.stat().st_size,
        "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
        "payload_or_qwen_accessed": False,
        "receipt_sha256": sha(RECEIPT_PATH),
        "schema": "same-layer-clustered-ib-r3-source-free-preflight-receipt-verification-v0",
        "status": "PASS_AUTHORIZED_SINGLE_PREFLIGHT_RECEIPT",
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
