#!/usr/bin/env python3
"""Independent source-only authentication and hostile checks for N18 v3.

This harness never imports numerical dependencies and never opens model data.
It authenticates a flat producer tree against the audit-owned inventory, then
reports independent arithmetic/static counterexamples.  The producer's own
authenticated verifier/tests are launched separately through its held-procfd
bootstrap so this file is not part of that trust path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


ROOT_DOMAIN = b"TACTIC-N18-V3-AUTHENTICATED-SOURCE-ROOT-v1\0"
EXPECTED_ROOT = "1db2a1fd9da07743e556a02ce58d97424672129683a09c5b805714b7ce6709f5"
PAGE = 4096
MICRO = 4096
COARSE_PER_MICRO = 1228
FINE_PER_MICRO = 48
METADATA_PER_MICRO = 4
MIN_EXPLICIT = 8191


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_inventory(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"), object_pairs_hook=_pairs)
    require(
        isinstance(value, dict)
        and set(value) == {"schema", "status", "files", "authority_boundary"},
        "inventory exact keys",
    )
    rows = value["files"]
    require(isinstance(rows, list) and len(rows) == 14, "inventory exact row count")
    names = [row["name"] for row in rows]
    require(names == sorted(set(names), key=lambda item: item.encode("utf-8")), "bytewise inventory order")
    return value


def authenticate(producer: Path, inventory: dict[str, Any]) -> tuple[dict[str, bytes], str]:
    require(producer.is_absolute(), "absolute producer path")
    entries = list(os.scandir(producer))
    require(
        all(entry.is_file(follow_symlinks=False) and not entry.is_symlink() for entry in entries),
        "producer must be a flat regular-file closure",
    )
    expected = {row["name"]: row for row in inventory["files"]}
    require({entry.name for entry in entries} == set(expected), "producer exact file closure")
    packets: dict[str, bytes] = {}
    root = hashlib.sha256(ROOT_DOMAIN)
    for name in sorted(expected, key=lambda item: item.encode("utf-8")):
        path = producer / name
        before = path.stat(follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode), f"regular producer member: {name}")
        packet = path.read_bytes()
        after = path.stat(follow_symlinks=False)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            f"identity-stable producer member: {name}",
        )
        row = expected[name]
        digest = hashlib.sha256(packet).hexdigest()
        require(len(packet) == row["bytes"] and digest == row["sha256"], f"inventory binding: {name}")
        encoded = name.encode("utf-8")
        root.update(len(encoded).to_bytes(4, "big"))
        root.update(encoded)
        root.update(len(packet).to_bytes(8, "big"))
        root.update(bytes.fromhex(digest))
        packets[name] = packet
    observed = root.hexdigest()
    require(observed == EXPECTED_ROOT, "independently pinned producer root")
    return packets, observed


def partition(intermediate: int, hidden: int) -> dict[str, Any]:
    require(type(intermediate) is int and type(hidden) is int and intermediate > 0 and hidden > 0, "shape")
    role = intermediate * hidden
    weights = 3 * role
    full_per_role, tail_per_role = divmod(role, MICRO)
    full = 3 * full_per_role
    tail = 3 * tail_per_role
    coarse = full * COARSE_PER_MICRO + (307 * tail) // 1024
    fine = full * FINE_PER_MICRO
    metadata = full * METADATA_PER_MICRO
    tail_total = (5 * tail) // 16
    tail_extra = tail_total - (307 * tail) // 1024
    budget = (5 * weights) // 16
    total = coarse + fine + metadata + tail_extra
    require(total == budget, "upper-budget arithmetic")
    explicit = budget >= MIN_EXPLICIT
    physical = budget if explicit else 0
    return {
        "intermediate": intermediate,
        "hidden": hidden,
        "role_values": role,
        "weights": weights,
        "full_microblocks": full,
        "residual_microblocks_per_role": full_per_role % 64,
        "tail_values": tail,
        "coarse_bytes": coarse,
        "fine_bytes": fine,
        "metadata_bytes": metadata,
        "tail_extra_bytes": tail_extra,
        "coded_budget_bytes": budget,
        "physical_bytes": physical,
        "physical_bpw": 8 * physical / weights,
        "explicit": explicit,
    }


def page_ledger(length: int, offset: int, passes: int = 1) -> dict[str, Any]:
    if length == 0:
        return {"unique_pages": 0, "unique_page_bytes": 0, "unique_amplification": 0.0, "pass_amplification": 0.0}
    first = offset // PAGE
    last = (offset + length - 1) // PAGE
    page_bytes = (last - first + 1) * PAGE
    return {
        "unique_pages": last - first + 1,
        "unique_page_bytes": page_bytes,
        "unique_amplification": page_bytes / length,
        "pass_amplification": passes * page_bytes / length,
    }


def inspect_semantics(packets: dict[str, bytes]) -> dict[str, Any]:
    for name, packet in packets.items():
        if name.endswith(".py"):
            ast.parse(packet.decode("utf-8"), filename=name)
    all_text = b"\n".join(packets.values()).decode("utf-8", errors="ignore")
    layout_text = packets["universal_layout.py"].decode("utf-8")
    runtime_text = packets["runtime_auth.py"].decode("utf-8")
    design = json.loads(packets["design_lock.json"])

    # The prior frozen physical packet had these executable fields.  Absence
    # here proves only that v3 is a layout scaffold, not a packet-equivalent
    # implementation.
    frozen_fields = {
        "packet_magic_TACN18C2": "TACN18C2" in all_text,
        "header_bytes_128": "HEADER_BYTES" in all_text and "128" in all_text,
        "logical_hard_eof": "hard EOF" in all_text or "hard_eof" in all_text,
        "canonical_reencode": "canonical_bit_reencode" in all_text,
        "payload_digest_field": "payload_sha256" in all_text,
        "decoder_scale_field": "decoder_scale" in all_text,
        "packet_parse_function": "parse_reservoir" in all_text,
        "packet_pack_function": "pack_reservoir" in all_text,
    }
    require(not any(frozen_fields.values()), "unexpected implemented frozen N18 packet semantics")

    runtime_tree_holds_files = "held: tuple[HeldRegularFile" in runtime_text
    runtime_uses_normal_import_metadata = "importlib.metadata.distribution" in runtime_text
    runtime_closes_each_tree_file = "source.close()" in runtime_text
    runtime_has_distribution_byte_loader = "DistributionBytesLoader" in runtime_text
    runtime_rejects_preloaded_distributions = "preloaded distribution" in runtime_text

    return {
        "frozen_packet_fields_present": frozen_fields,
        "codec_semantics_conclusion": "RATE_PARTITION_ONLY_NO_PACK_PARSE_DECODE_REENCODE_OR_TAIL_LANGUAGE",
        "residual_group_rule_present": "residual_microblocks" in layout_text,
        "residual_group_packet_semantics_present": "residual_microblocks" in design["coarse_contract"].get("canonical_tail_rule", ""),
        "runtime_tree_authentication": {
            "holds_distribution_files_for_runtime": runtime_tree_holds_files,
            "uses_importlib_metadata_file_inventory": runtime_uses_normal_import_metadata,
            "closes_each_file_immediately_after_hash": runtime_closes_each_tree_file,
            "authenticated_distribution_byte_loader": runtime_has_distribution_byte_loader,
            "preloaded_distribution_rejection": runtime_rejects_preloaded_distributions,
            "conclusion": "HASH_TO_LATER_NORMAL_IMPORT_TOCTOU_REMAINS",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", required=True)
    parser.add_argument("--inventory", required=True)
    args = parser.parse_args()
    producer = Path(args.producer).resolve(strict=True)
    inventory_path = Path(args.inventory).resolve(strict=True)
    inventory = load_inventory(inventory_path)
    packets, root = authenticate(producer, inventory)

    qwen = partition(768, 2048)
    odd = partition(769, 2051)
    tiny = partition(1, 1)
    small_full = partition(1, 4096)
    threshold = partition(1, 8738)
    qwen_page = page_ledger(qwen["physical_bytes"], 0)
    odd_page = page_ledger(odd["physical_bytes"], 0)
    threshold_offsets = [page_ledger(threshold["physical_bytes"], offset) for offset in range(PAGE)]

    receipt = {
        "schema": "tactic_actual_coarse_n18_v3_independent_hostile_audit_v1",
        "scope": "source-only; no model/payload, numerical producer, numeric dependency import, CUDA, runtime freeze, or result",
        "producer_root": root,
        "producer_files": len(packets),
        "producer_bytes": sum(len(packet) for packet in packets.values()),
        "layout": {
            "qwen": qwen,
            "qwen_owner_pages": qwen_page,
            "odd_769x2051": odd,
            "odd_owner_pages": odd_page,
            "tiny_1x1": tiny,
            "small_complete_micro_1x4096": small_full,
            "minimum_explicit_1x8738": threshold,
            "minimum_explicit_worst_unique_page_amplification": max(row["unique_amplification"] for row in threshold_offsets),
            "minimum_explicit_worst_two_pass_amplification": max(row["pass_amplification"] for row in threshold_offsets) * 2,
        },
        "target_contract_counterexamples": {
            "tiny_artifact_bpw": tiny["physical_bpw"],
            "tiny_artifact_meets_2p15_floor": tiny["physical_bpw"] >= 2.15,
            "tiny_zero_reconstruction_relative_mse_for_nonzero_source": 1.0,
            "tiny_zero_reconstruction_F_at_actual_R": 1.0,
            "small_complete_micro_nominal_partition_is_discarded_by_zero_fallback": small_full["physical_bytes"] == 0 and small_full["coarse_bytes"] > 0,
            "repeated_read_ledger_declared": "compressed_frame_reads" in packets["design_lock.json"].decode("utf-8"),
        },
        "semantics": inspect_semantics(packets),
        "verdict": "BLOCK_SOURCE_CLOSURE",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
