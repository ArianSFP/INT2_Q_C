"""Post-bootstrap named grouped-v4 state authentication helper.

This is a library member, not a standalone entrypoint.  The clean v5
``verify_prelaunch.py`` authenticates the closed package before importing it.
"""

from __future__ import annotations

from pathlib import Path


def verify_named_state(common, overlay, v4_run_root: Path, v4_result_audit: Path) -> dict:
    if common.environment_has_cuda_imports():
        raise common.ProtocolError("v4-reuse verifier must remain CPU-only")
    lock = common.load_candidate_lock()
    authenticated = overlay.authenticate_v4_topk(v4_run_root, v4_result_audit, lock)
    return {
        "schema": "qwen3_tier_c_grouped_v5_v4_reuse_verification_v5",
        "status": "PASS_AUTHENTICATED_AFTER_CLEAN_BOOTSTRAP_NO_QWEN_OR_CUDA_ACCESS",
        "candidate_lock_file_sha256": common.CANDIDATE_LOCK_FILE_SHA256,
        "translated_topk_shape": list(authenticated.translated_ordinals.shape),
        "metric_shape": list(authenticated.metrics.shape),
        "authentication": authenticated.receipt,
        "cuda_or_qwen_access": False,
    }


if __name__ == "__main__":
    raise SystemExit(
        "standalone execution is forbidden; use `python -B -I "
        "/workspace/INT2__compression/INT2_Q_C/research/"
        "initialization_anchor_oracle_tier_c_grouped_v5_layout_overlay_v5/verify_prelaunch.py "
        "--v4-run-root PATH --v4-result-audit FILE`"
    )
