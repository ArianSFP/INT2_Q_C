#!/usr/bin/env python3
"""Fail-closed validator for a future independent physical result receipt."""

from __future__ import annotations

from typing import Any

from rm_order import TARGET_N, classify_selected_count


EXPECTED_PINS = {
    "agent_polaris_qwen_rht_encoder.py":
        "062f74ca3e44ae2df1abea7762967f9f7c14188d1e963a06c4a07bed56f478a0",
    "bg_codec_bec_encoder.py":
        "456a3ae5fe00c578456dc9430bf7ae059ed9dbb8dcf04a6bafad3a88cc5cb267",
    "strata_v2_klt_mixed_independent_auditor_v1.py":
        "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e",
}
REQUIRED_PACKET_FIELDS = {
    "expert_local_header", "profile_and_order_selector", "fp_scale", "seeds",
    "arithmetic_payload", "termination", "integrity_trailer", "padding",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_independent_result(receipt: dict[str, Any], *, require_target_rate: bool = True) -> dict[str, Any]:
    require(receipt.get("schema") ==
            "strata-rm-global-swap-v0-independent-physical-result", "result schema")
    require(receipt.get("external_pins") == EXPECTED_PINS, "external pins")
    require(receipt.get("candidate") == "RM-ordered truncated polar", "candidate name")
    require(receipt.get("coset") == "current_random", "current-random coset only")
    require(receipt.get("rate_basis") ==
            "literal_full_packet_bytes_plus_charged_shared_bytes", "physical rate basis")
    require(receipt.get("independent_decoder_source_sha256") not in (None, ""),
            "independent decoder pin")
    require(receipt.get("independent_decode_complete") is True, "independent decode")
    require(receipt.get("causal_probabilities_regenerated") is True,
            "causal probabilities")
    require(receipt.get("packet_consumed_exactly") is True, "exact packet consumption")
    require(receipt.get("canonical_reencode_byte_identical") is True,
            "canonical byte replay")
    require(receipt.get("source_domain_score_from_decoded_packet") is True,
            "decoded source-domain score")
    require(receipt.get("overlap_receipt_used_for_rd") is False,
            "overlap receipt is non-RD evidence only")
    fields = set(receipt.get("charged_packet_fields", []))
    require(REQUIRED_PACKET_FIELDS <= fields, "full packet fields")

    blocks = receipt.get("blocks")
    require(isinstance(blocks, list) and blocks, "block records")
    total_weights = total_packet_bytes = 0
    for block in blocks:
        n = block.get("n")
        require(n in TARGET_N, "production block length")
        levels = block.get("levels")
        require(isinstance(levels, list) and len(levels) == 6, "six levels")
        for level in levels:
            old_k, new_k = level.get("reference_bec_k"), level.get("rm_ordered_k")
            require(isinstance(old_k, int) and old_k == new_k and 0 <= old_k <= n,
                    "selected-count equality")
            expected = classify_selected_count(n, old_k)
            require(level.get("set_name") == expected["name"], "exact RM naming")
        packet_bytes = block.get("literal_packet_bytes")
        require(isinstance(packet_bytes, int) and packet_bytes > 0, "literal packet bytes")
        require(isinstance(block.get("literal_packet_sha256"), str) and
                len(block["literal_packet_sha256"]) == 64, "packet hash")
        require(block.get("canonical_reencode_sha256") == block["literal_packet_sha256"],
                "block canonical hash")
        total_weights += n
        total_packet_bytes += packet_bytes

    shared = receipt.get("charged_shared_bytes")
    require(isinstance(shared, int) and shared >= 0, "charged shared bytes")
    require(receipt.get("total_original_weights") == total_weights, "weight total")
    require(receipt.get("total_physical_bytes") == total_packet_bytes + shared,
            "physical byte total")
    actual_bpw = 8.0 * (total_packet_bytes + shared) / total_weights
    try:
        declared_bpw = float(receipt.get("actual_physical_bpw"))
    except (TypeError, ValueError) as error:
        raise ValueError("actual physical bpw") from error
    require(abs(declared_bpw - actual_bpw) <= 1e-15, "actual physical bpw")
    if require_target_rate:
        require(2.15 <= actual_bpw <= 2.5, "target rate interval")
    require(receipt.get("selected_count_used_as_rate") is False,
            "selected count is not a rate")
    return {
        "passed": True,
        "actual_physical_bpw": actual_bpw,
        "blocks": len(blocks),
        "status": "RESULT_CONTRACT_PASS__STILL_REQUIRES_AUDITOR_AUTHENTICITY",
    }

