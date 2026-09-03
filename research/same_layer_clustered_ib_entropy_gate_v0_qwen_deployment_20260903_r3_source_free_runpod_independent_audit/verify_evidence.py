"""Detached, read-only audit of the copied CBIB-1 r3 RunPod preflight evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import stat


DEPLOYMENT_MANIFEST_SHA256 = (
    "5bac35949d374961f280d12680e3755c8c0d45f58d1937d6fb19120547b3649f"
)
DEPLOYMENT_ROOT_SHA256 = (
    "ae1ae10c0a39b8167739498db13b69aaaf738a17b247ca7861c5d76a57d6c1ee"
)
REVIEW_MANIFEST_SHA256 = (
    "9465a1d6c9ffdb7553721e1a470b2ab485dcb66d8d32e7b21b605e9ff99421e2"
)
REVIEW_ROOT_SHA256 = (
    "00bb72454630baf53d1b8f0d89a4438aedae3618b6c2bc465fef5839213cc3d9"
)
AUTHORITY_SHA256 = (
    "ff38022186f39d610a9108e0926be0263b4fad065146725e9311a637219371a2"
)
WRAPPER_SHA256 = (
    "91d55282c1d84b3e108e54c9772267a234cba7aeb8784504183badc90b25d71e"
)
FIXTURE_SHA256 = (
    "33f7ba9d4ae0589d06abcfab06bac46d06ef75188d714350ba993df0ca9bbab5"
)
EVIDENCE = {
    "ONE_USE_PREFLIGHT_CLAIM.json": (
        355, "b79b7a39013230c64f9b9504781bc07b0cc879b2ffe5798191118d0a1b5a44b0"
    ),
    "child_stderr.txt": (
        0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "receipt.json": (
        1574, "38a4eb497983aa8b5a559fa96fcbcbb11dc77cdd78dabfff9d2d4d06c5bf1913"
    ),
    "wrapper_status.json": (
        375, "1c5a7c859f6b3b3e061974dfd9dbb0e431b8066c74b511620631a805a0ce242a"
    ),
}


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def regular_bytes(path: Path) -> bytes:
    need(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(),
         f"regular nonsymlink required: {path.name}")
    return path.read_bytes()


def strict_json_line(path: Path) -> tuple[dict, bytes]:
    raw = regular_bytes(path)
    need(len(raw.splitlines()) == 1 and raw.endswith(b"\n"),
         f"one newline-terminated JSON line required: {path.name}")
    value = json.loads(raw)
    need(raw == (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(),
         f"canonical JSON required: {path.name}")
    return value, raw


def authenticate(root: Path, manifest_sha256: str, source_root_sha256: str,
                 schema: str) -> dict:
    root = root.resolve(strict=True)
    need(root.is_dir() and not root.is_symlink(), "real package directory")
    manifest, raw = strict_json_line(root / "SOURCE_MANIFEST.json")
    need(sha_bytes(raw) == manifest_sha256, "externally pinned manifest")
    need(manifest.get("schema") == schema, "manifest schema")
    rows = manifest.get("files")
    need(isinstance(rows, list) and rows, "manifest rows")
    names = [row.get("name") for row in rows]
    need(names == sorted(names) and len(names) == len(set(names)), "manifest ordering")
    need(sorted(path.name for path in root.iterdir()) ==
         sorted(names + ["SOURCE_MANIFEST.json"]), "exact package membership")
    normalized = []
    for row in rows:
        need(set(row) == {"bytes", "name", "sha256"}, "manifest row fields")
        member = root / row["name"]
        payload = regular_bytes(member)
        need(len(payload) == int(row["bytes"]) and sha_bytes(payload) == row["sha256"],
             f"package member mismatch: {row['name']}")
        normalized.append({"bytes": len(payload), "name": row["name"],
                           "sha256": row["sha256"]})
    observed_root = sha_bytes(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    )
    need(observed_root == source_root_sha256 == manifest.get("source_root_sha256"),
         "source-root digest")
    return manifest


def verify_wrapper_source(review: Path) -> None:
    authority = regular_bytes(review / "AUTHORIZED_PREFLIGHT.json")
    wrapper = regular_bytes(review / "run_authorized_preflight_once.py")
    need(sha_bytes(authority) == AUTHORITY_SHA256, "preflight authority digest")
    need(sha_bytes(wrapper) == WRAPPER_SHA256, "preflight wrapper digest")
    text = wrapper.decode("utf-8")
    main = text[text.index("def main()") :]
    ordered = (
        "_exclusive_write(CLAIM_PATH, claim)",
        "_authenticate_package()",
        "receipt_fd = os.open(",
        "subprocess.run(",
        "_validate_receipt(RECEIPT_PATH.read_bytes())",
        "_exclusive_write(\n        WRAPPER_STATUS_PATH",
    )
    cursor = -1
    for token in ordered:
        cursor = main.index(token, cursor + 1)
    need("os.O_WRONLY | os.O_CREAT | os.O_EXCL" in text, "exclusive-create primitive")
    need("shell=False" in text and "run_source_free_cupy.py" in text and
         "run_gate.py" not in text and "CLAIM_PATH.unlink" not in text,
         "source-free one-use wrapper scope")


def verify_evidence(evidence: Path) -> dict:
    evidence = evidence.resolve(strict=True)
    need(evidence.is_dir() and not evidence.is_symlink(), "real evidence directory")
    need(sorted(path.name for path in evidence.iterdir()) == sorted(EVIDENCE),
         "exact four-member evidence closure")
    for name, (size, digest) in EVIDENCE.items():
        payload = regular_bytes(evidence / name)
        need(len(payload) == size and sha_bytes(payload) == digest,
             f"evidence member mismatch: {name}")

    need(regular_bytes(evidence / "child_stderr.txt") == b"", "child stderr is not empty")
    claim, _ = strict_json_line(evidence / "ONE_USE_PREFLIGHT_CLAIM.json")
    need(set(claim) == {"authorization_id", "claimed_at_utc",
                        "deployment_manifest_sha256", "schema", "status"},
         "claim fields")
    need(claim["authorization_id"] ==
         "CBIB1_R3_SOURCE_FREE_RTX5090_PREFLIGHT_ONCE_20260903" and
         claim["deployment_manifest_sha256"] == DEPLOYMENT_MANIFEST_SHA256 and
         claim["schema"] == "same-layer-clustered-ib-r3-source-free-preflight-claim-v0" and
         claim["status"] == "ATTEMPT_CONSUMED_BEFORE_VALIDATION_AND_CHILD_SPAWN",
         "claim identity")
    claimed = datetime.fromisoformat(claim["claimed_at_utc"])
    need(claimed.tzinfo is not None and claimed.utcoffset() == timezone.utc.utcoffset(claimed),
         "UTC claim timestamp")

    wrapper, _ = strict_json_line(evidence / "wrapper_status.json")
    need(set(wrapper) == {"deployment_manifest_sha256", "receipt_sha256", "schema",
                          "status", "underlying_status"}, "wrapper-status fields")
    need(wrapper == {
        "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
        "receipt_sha256": EVIDENCE["receipt.json"][1],
        "schema": "same-layer-clustered-ib-r3-source-free-preflight-wrapper-v0",
        "status": "PASS_SINGLE_ATTEMPT_CAPTURED_AND_VALIDATED",
        "underlying_status": "PASS_PRODUCTION_GEOMETRY_FULL_CPU_CUPY_PARITY",
    }, "wrapper-status identity")

    result, raw = strict_json_line(evidence / "receipt.json")
    expected_top = {
        "all_controls_executed", "control_count", "cuda_driver", "cuda_runtime",
        "cupy_version", "deployment_manifest_sha256",
        "detailed_assignments_counts_partitions", "device_name", "fixture_labels_sha256",
        "full_gate_exact_and_float_parity", "full_gate_status", "numpy_file",
        "numpy_version", "pairwise_mi_max_absolute_delta", "payload_or_qwen_accessed",
        "production_geometry", "quantizer_scales_labels_exact", "schema",
        "source_read_survivor_endpoints", "source_read_survivor_group_sizes", "status",
    }
    need(set(result) == expected_top, "receipt fields")
    need(result["schema"] ==
         "same-layer-clustered-ib-qwen-deployment-source-free-cupy-v0-r3" and
         result["status"] == "PASS_PRODUCTION_GEOMETRY_FULL_CPU_CUPY_PARITY" and
         result["deployment_manifest_sha256"] == DEPLOYMENT_MANIFEST_SHA256 and
         result["payload_or_qwen_accessed"] is False, "receipt identity")
    need(result["production_geometry"] == {
        "coordinates_per_role": 131072,
        "experts": 16,
        "fold_coordinate_counts": [16384] * 8,
        "fold_count": 8,
        "group_sizes": [2, 4, 8, 16],
        "roles": 2,
        "scale_bytes_per_expert": 256,
        "superblock_values": 2048,
    }, "exact production geometry")
    need(result["quantizer_scales_labels_exact"] is True and
         result["pairwise_mi_max_absolute_delta"] == 0.0 and
         math.isfinite(result["pairwise_mi_max_absolute_delta"]),
         "exact quantizer/MI parity")
    need(result["detailed_assignments_counts_partitions"] == {
        "conditional_count_arrays_checked": 480,
        "groups_checked": 120,
        "heldout_assignments_checked": 3932160,
        "latent_count_arrays_checked": 480,
        "max_model_float_absolute_delta": 4.656612873077393e-10,
        "models_checked": 240,
        "training_assignments_checked": 27525120,
    }, "exact assignment/count/partition coverage")
    need(result["full_gate_exact_and_float_parity"] == {
        "exact_field_count": 5061,
        "float_field_count": 1127,
        "max_float_absolute_delta": 1.862645149230957e-09,
        "max_float_path": "gate.source_scores[0].baseline_data_bits",
    }, "exact full-gate parity receipt")
    need(result["source_read_survivor_group_sizes"] == [2] and
         result["source_read_survivor_endpoints"] == {"2": ["5/2"]},
         "exact source/read survivor")
    need(result["full_gate_status"] == "HARD_KILL_CHARGED_OR_CONTROLS_BELOW_TARGET" and
         result["all_controls_executed"] is True and result["control_count"] == 8,
         "exact control execution and disposition")
    need(result["fixture_labels_sha256"] == FIXTURE_SHA256 and
         result["numpy_version"] == "2.5.2" and
         result["numpy_file"] ==
         "/workspace/int2-cupy-venv/lib/python3.12/site-packages/numpy/__init__.py" and
         result["cupy_version"] == "14.2.0" and
         result["device_name"] == "NVIDIA GeForce RTX 5090" and
         result["cuda_runtime"] == 12090 and result["cuda_driver"] == 13000,
         "exact runtime and fixture identity")
    need(sha_bytes(raw) == EVIDENCE["receipt.json"][1], "receipt digest")
    return {"claimed_at_utc": claim["claimed_at_utc"],
            "receipt_sha256": sha_bytes(raw)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-directory", required=True, type=Path)
    parser.add_argument("--deployment-package", required=True, type=Path)
    parser.add_argument("--review-package", required=True, type=Path)
    parser.add_argument("--deployment-manifest-sha256", required=True)
    parser.add_argument("--review-manifest-sha256", required=True)
    args = parser.parse_args()
    need(args.deployment_manifest_sha256 == DEPLOYMENT_MANIFEST_SHA256,
         "external deployment manifest pin")
    need(args.review_manifest_sha256 == REVIEW_MANIFEST_SHA256,
         "external review manifest pin")
    deployment = authenticate(
        args.deployment_package, DEPLOYMENT_MANIFEST_SHA256, DEPLOYMENT_ROOT_SHA256,
        "same-layer-clustered-ib-qwen-deployment-manifest-v0-r3",
    )
    review = authenticate(
        args.review_package, REVIEW_MANIFEST_SHA256, REVIEW_ROOT_SHA256,
        "same-layer-clustered-ib-qwen-deployment-r3-independent-review-manifest-v0",
    )
    need(len(deployment["files"]) == 15, "deployment member count")
    need(review["audited_deployment_manifest_sha256"] == DEPLOYMENT_MANIFEST_SHA256 and
         review["audited_deployment_root_sha256"] == DEPLOYMENT_ROOT_SHA256 and
         review["status"] == "PASS_R3_AUTHORIZE_ONE_SOURCE_FREE_RTX5090_PREFLIGHT_ONLY",
         "review-to-deployment binding")
    verify_wrapper_source(args.review_package.resolve(strict=True))
    observed = verify_evidence(args.evidence_directory)
    print(json.dumps({
        "authorized_qwen_invocations": 0,
        "authorized_source_free_preflight_attempts_remaining": 0,
        "child_stderr_bytes": 0,
        "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
        "deployment_source_root_sha256": DEPLOYMENT_ROOT_SHA256,
        "evidence_member_count": 4,
        "payload_or_qwen_accessed": False,
        "receipt_sha256": observed["receipt_sha256"],
        "review_manifest_sha256": REVIEW_MANIFEST_SHA256,
        "review_source_root_sha256": REVIEW_ROOT_SHA256,
        "schema": "same-layer-clustered-ib-r3-source-free-runpod-independent-audit-v0",
        "status": "PASS_AUTHORIZED_SINGLE_PREFLIGHT_RECEIPT_INDEPENDENTLY_AUDITED",
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
