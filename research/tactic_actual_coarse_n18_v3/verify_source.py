#!/usr/bin/env python3
"""Authenticated-byte, source-only verifier for the N18 v3 review candidate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from dependency_auth import validate_dependency_graph
from immutable_bootstrap import EXPECTED_SOURCE_FILES, ROOT_DOMAIN
from runtime_auth import REQUIRED_DISTRIBUTIONS
from universal_layout import ExpertGeometry, layout_panel, partition_expert, qwen_evaluation_ledger
from v3_common import (
    COARSE_BYTES_PER_MICRO,
    DESIGN_SCHEMA,
    MAX_SOURCE_AGGREGATE_BYTES,
    MAX_SOURCE_MEMBER_BYTES,
    MICRO,
    N18_COARSE_RESERVOIR_BYTES,
    canonical_json,
    exact_keys,
    require,
    valid_sha256,
)


def source_root(packets: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(ROOT_DOMAIN)
    total = 0
    for name in sorted(packets, key=lambda item: item.encode("utf-8")):
        packet = packets[name]
        require(type(packet) is bytes and 0 < len(packet) <= MAX_SOURCE_MEMBER_BYTES, "bounded source packet")
        total += len(packet)
        require(total <= MAX_SOURCE_AGGREGATE_BYTES, "source aggregate byte cap")
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(packet).to_bytes(8, "big"))
        digest.update(hashlib.sha256(packet).digest())
    return digest.hexdigest()


def _verify_design(raw: bytes) -> dict[str, Any]:
    from v3_common import strict_json_loads

    value = exact_keys(
        strict_json_loads(raw),
        {
            "schema",
            "status",
            "scope",
            "coarse_contract",
            "handoff_contract",
            "shape_contract",
            "physical_contract",
            "source_execution_contract",
            "runtime_contract",
            "publication_contract",
            "authority_contract",
            "telemetry_contract",
        },
        "design lock",
    )
    require(value["schema"] == DESIGN_SCHEMA, "design schema")
    require(
        value["status"] == "POSTIMPLEMENTATION_REVIEW_CANDIDATE_SOURCE_ONLY_NOT_EXTERNALLY_SEALED",
        "unsealed source-only status",
    )
    coarse = value["coarse_contract"]
    require(
        coarse["n18_values"] == 262144
        and coarse["microblock_values"] == MICRO
        and coarse["coarse_bytes_per_full_microblock"] == COARSE_BYTES_PER_MICRO
        and coarse["coarse_bytes_per_full_n18"] == N18_COARSE_RESERVOIR_BYTES
        and coarse["coarse_logical_bpw_on_full_microblocks"] == "307/128",
        "frozen N18-307 coarse design",
    )
    handoff = value["handoff_contract"]
    require(
        handoff["fine_bits_per_full_microblock"] == 384
        and handoff["metadata_bytes_per_full_microblock"] == 4
        and handoff["total_bytes_per_full_microblock"] == 1280
        and "cannot kill" in handoff["branch_rule"],
        "frozen DH384 handoff and branch boundary",
    )
    require(value["authority_contract"]["producer_authority"] == "NONE", "no producer review authority")
    require(
        tuple(value["runtime_contract"]["required_distributions"]) == REQUIRED_DISTRIBUTIONS,
        "runtime distribution contract",
    )
    return value


def verify_packets(context: Mapping[str, Any]) -> dict[str, Any]:
    require(
        isinstance(context, Mapping)
        and set(context) == {"packets", "source_root", "inventory_sha256", "loader_kind"},
        "authenticated context exact keys",
    )
    packets = context["packets"]
    require(isinstance(packets, Mapping), "immutable source packet mapping")
    require(
        tuple(sorted(packets, key=lambda name: name.encode("utf-8")))
        == tuple(sorted(EXPECTED_SOURCE_FILES, key=lambda name: name.encode("utf-8"))),
        "exact source packet closure",
    )
    observed_root = source_root(packets)
    require(observed_root == context["source_root"], "authenticated context source root")
    require(valid_sha256(context["inventory_sha256"], nonzero=True), "authenticated inventory digest")
    require(
        context["loader_kind"] == "immutable_authenticated_bytes_no_pycache_no_live_path",
        "authenticated loader kind",
    )
    for name, packet in packets.items():
        if name.endswith(".py"):
            ast.parse(packet.decode("utf-8"), filename=name)
    design = _verify_design(packets["design_lock.json"])
    dependencies = validate_dependency_graph(packets["dependency_graph.json"])

    qwen = qwen_evaluation_ledger()
    odd = layout_panel([ExpertGeometry(769, 2051)])
    tiny = layout_panel([ExpertGeometry(1, 1)])
    odd_partition = partition_expert(ExpertGeometry(769, 2051))
    tiny_partition = partition_expert(ExpertGeometry(1, 1))
    require(
        odd.physical_bpw <= 2.5 and odd.maximum_read_amplification < 2.0,
        "odd-tail physical closure",
    )
    require(
        odd_partition.tail_values > 0 and odd_partition.tail_coarse_bytes < N18_COARSE_RESERVOIR_BYTES,
        "odd-tail no reservoir expansion",
    )
    require(
        tiny.physical_bytes == 0
        and tiny.maximum_read_amplification == 0.0
        and tiny_partition.fallback == "IMPLICIT_ZERO_EXPERT_NO_BYTES_NO_READ_V1",
        "tiny-owner fallback closure",
    )
    return {
        "schema": "tactic_actual_coarse_n18_source_verification_v3",
        "status": "SOURCE_ONLY_VERIFIED_NOT_EXTERNALLY_SEALED",
        "source_root": observed_root,
        "source_files": len(packets),
        "source_bytes": sum(len(packet) for packet in packets.values()),
        "dependency_sources": len(dependencies),
        "qwen_reference_ledger": qwen,
        "odd_769x2051": {
            "physical_bpw": odd.physical_bpw,
            "unique_page_read_amplification": odd.maximum_read_amplification,
            "tail_values": odd_partition.tail_values,
        },
        "tiny_1x1": {
            "physical_bytes": tiny.physical_bytes,
            "unique_page_read_amplification": tiny.maximum_read_amplification,
            "fallback": tiny_partition.fallback,
        },
        "authority": design["authority_contract"]["review_authority"],
        "claims": [
            "source structure and arithmetic invariants only",
            "no payload/model read",
            "no numerical producer or CUDA run",
            "no runtime freeze or external review seal",
        ],
    }


def authenticated_main(context: Mapping[str, Any], argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args(list(argv))
    receipt = verify_packets(context)
    if arguments.compact:
        print(canonical_json(receipt).decode("utf-8"))
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit("verify_source.py must execute through immutable_bootstrap.py")
