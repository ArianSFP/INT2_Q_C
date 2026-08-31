#!/usr/bin/env python3
"""Build a sealed, metadata-only proposal for a second blind Qwen panel.

The selector is deliberately offline.  It reads the already cached
``model.safetensors.index.json`` and safetensors JSON headers, but contains no
HTTP client and no tensor-payload reader.  It also constructs a conservative
contamination ledger from prior source manifests and rejects every layer and
every expert index that has appeared in any prior opened/reserved expert pair.

This is a selection/route *proposal*, not a codec freeze and not authority to
materialize any selected tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PINNED_REPO = "Qwen/Qwen3-30B-A3B"
PINNED_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
V1_AUDIT_SUMMARY_SHA256 = "6aa4d8538389179cdbdb2edaf332d707eef39aa4ef7cd395f8e82d755ca1bb37"
EXPECTED_EVIDENCE_HASHES = {
    "qwen_polaris_heldout32/heldout32_manifest.json": "3b882c74870c1e27bcddf7427e4c6ffea816d4f9847447eb218729ca69426a55",
    "agent_rd_structure_diag_cross_expert_sources.json": "4194ff0aa13e71e2c9631f6f2cfd145c5146edf9c6d287084197499872dff782",
    "blind_protocol/selection.lock.json": "78a00016a0f62eca3f3c9b451b226d6821b38b8c596eb7b0f691bbe465174095",
    "blind_protocol/unblinded/source_hashes.lock.json": "b520461fa71783ad597e6f71211cd777c0af93baef14177edc9181f54cf918d5",
    "agent_l46e13_final_sota_protocol_freeze_v2.json": "ec34b13ddff7f31a3a37d9b81eadcf349d5669eaf44afb6437ce28584fa54e40",
}
BLOCK_VALUES = 1 << 18
GROUP_VALUES = 2048
ROLES = ("gate_proj", "up_proj", "down_proj")
ROLE_SHAPES = {
    "gate_proj": (768, 2048),
    "up_proj": (768, 2048),
    "down_proj": (2048, 768),
}
STRATA = tuple((8 * index, 8 * index + 7) for index in range(6))
ROUTE_RECORD = struct.Struct(">HHBBH")
ROLE_ENUM = {"gate": 0, "up": 1, "down": 2}
AXIS_ENUM = {"row": 0, "column": 1}
TENSOR_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)
SHARD_RE = re.compile(r"^model-\d{5}-of-00016\.safetensors$")
TEXT_EXTENSIONS = {".json", ".md", ".txt", ".log", ".py", ".csv", ".yaml", ".yml"}

# Exact conservative union found before v2 selection.  Pair L46:E13 is a
# reservation from a prior source-free protocol, not a claim that its payload
# was opened.  Reserving it makes the new panel disjoint from both actual
# accesses and named prior protocols.
EXPECTED_EXCLUDED_PAIRS = {
    (0, 0), (0, 31), (0, 63), (0, 127),
    (3, 57), (3, 121), (3, 125),
    (11, 104),
    (15, 0), (15, 8), (15, 16), (15, 24), (15, 31), (15, 32),
    (15, 40), (15, 48), (15, 56), (15, 64), (15, 72), (15, 80),
    (15, 88), (15, 96), (15, 104), (15, 112), (15, 120), (15, 127),
    (16, 108),
    (22, 15), (22, 87),
    (25, 75),
    (31, 0), (31, 31), (31, 63),
    (32, 5),
    (44, 111),
    (46, 13),
    (47, 0), (47, 31), (47, 127),
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["lock_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def verify_seal(value: dict[str, Any]) -> str:
    declared = value.get("lock_sha256")
    clean = dict(value)
    clean.pop("lock_sha256", None)
    actual = sha256_bytes(canonical_bytes(clean))
    if declared != actual:
        raise AssertionError(f"invalid internal seal: {declared} != {actual}")
    return actual


def tensor_name(layer: int, expert: int, role: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{role}.weight"


def parse_pair(tensor: str) -> tuple[int, int] | None:
    match = TENSOR_RE.fullmatch(tensor)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def pair_records(values: Iterable[str]) -> set[tuple[int, int]]:
    return {pair for value in values if (pair := parse_pair(str(value))) is not None}


def render_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_new_or_identical(path: Path, raw: bytes, *, replace_generated: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != raw and not replace_generated:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        if existing == raw:
            return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def build_contamination_ledger(workspace: Path) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    actual_hashes = {
        rel: sha256_file(workspace / rel) for rel in EXPECTED_EVIDENCE_HASHES
    }
    if actual_hashes != EXPECTED_EVIDENCE_HASHES:
        raise AssertionError(f"contamination evidence hash drift: {actual_hashes}")

    categories: dict[tuple[int, int], set[str]] = defaultdict(set)
    evidence_details: dict[str, Any] = {}

    heldout = load_object(workspace / "qwen_polaris_heldout32/heldout32_manifest.json")
    heldout_tensors = [str(row["tensor"]) for row in heldout["blocks"]]
    for pair in pair_records(heldout_tensors):
        categories[pair].add("heldout32_block_payload_opened")
    evidence_details["heldout32"] = {
        "unique_expert_projection_tensors": sorted(
            tensor for tensor in set(heldout_tensors) if parse_pair(tensor) is not None
        ),
        "expert_pairs": [list(pair) for pair in sorted(pair_records(heldout_tensors))],
    }

    cross = load_object(workspace / "agent_rd_structure_diag_cross_expert_sources.json")
    if cross.get("status") != "COMPLETE_32_OF_32_PINNED_RANGE_FETCHES":
        raise AssertionError("cross-expert source record is not a completed access receipt")
    cross_tensors = [str(row["tensor_name"]) for row in cross["tensors"]]
    for pair in pair_records(cross_tensors):
        categories[pair].add("cross_expert_full_payload_opened")
    evidence_details["cross_expert"] = {
        "status": cross["status"],
        "tensor_count": len(cross_tensors),
        "expert_pairs": [list(pair) for pair in sorted(pair_records(cross_tensors))],
    }

    source_lock = load_object(workspace / "blind_protocol/unblinded/source_hashes.lock.json")
    verify_seal(source_lock)
    if source_lock.get("status") != "all_locked_sources_materialized_and_hash_finalized":
        raise AssertionError("v1 source lock is not finalized")
    v1_tensors = [str(row["tensor"]) for row in source_lock["matrices"]]
    for pair in pair_records(v1_tensors):
        categories[pair].add("blind_v1_full_payload_opened")
    evidence_details["blind_v1"] = {
        "source_lock_internal_sha256": source_lock["lock_sha256"],
        "tensor_count": len(v1_tensors),
        "expert_pairs": [list(pair) for pair in sorted(pair_records(v1_tensors))],
    }

    direct_payload_paths: list[dict[str, Any]] = []
    direct_roots = [workspace / "qwen_weight_cache/tensors", workspace]
    seen_paths: set[Path] = set()
    for root in direct_roots:
        if not root.exists():
            continue
        iterator = root.iterdir() if root == workspace else root.glob("*.bf16.bin")
        for path in iterator:
            if not path.is_file() or path in seen_paths:
                continue
            seen_paths.add(path)
            match = TENSOR_RE.search(path.name)
            if not match:
                continue
            pair = int(match.group(1)), int(match.group(2))
            categories[pair].add("direct_expert_payload_file_present")
            direct_payload_paths.append(
                {
                    "relpath": path.relative_to(workspace).as_posix(),
                    "bytes": path.stat().st_size,
                    "pair": list(pair),
                    "inspection": "path and file size only; payload bytes not opened by selector",
                }
            )
    evidence_details["direct_payload_files"] = sorted(
        direct_payload_paths, key=lambda row: row["relpath"]
    )

    # The pair is named by an older source-free one-open protocol.  Its status
    # does not prove payload access, so it is kept in a distinct category.
    categories[(46, 13)].add("conservative_prior_protocol_reservation_payload_not_asserted_open")
    evidence_details["prior_protocol_reservation"] = {
        "pair": [46, 13],
        "evidence": "agent_l46e13_final_sota_protocol_freeze_v2.json",
        "interpretation": "reserved for disjointness; not classified as an observed payload fetch",
    }

    pairs = set(categories)
    if pairs != EXPECTED_EXCLUDED_PAIRS:
        missing = sorted(EXPECTED_EXCLUDED_PAIRS - pairs)
        extra = sorted(pairs - EXPECTED_EXCLUDED_PAIRS)
        raise AssertionError(f"contamination union drift: missing={missing}, extra={extra}")

    opened_layers = sorted({layer for layer, _ in pairs})
    opened_experts = sorted({expert for _, expert in pairs})
    records = [
        {
            "layer": layer,
            "expert": expert,
            "evidence_categories": sorted(categories[(layer, expert)]),
            "payload_opened_asserted": any(
                category != "conservative_prior_protocol_reservation_payload_not_asserted_open"
                for category in categories[(layer, expert)]
            ),
        }
        for layer, expert in sorted(pairs)
    ]
    ledger = {
        "policy": "reject a candidate if its layer OR its expert index appeared in any prior opened/reserved pair; this is stricter than pair-only disjointness",
        "evidence_file_sha256s": actual_hashes,
        "pair_count": len(pairs),
        "opened_or_reserved_pairs": records,
        "excluded_layer_indices": opened_layers,
        "excluded_expert_indices": opened_experts,
        "evidence_details": evidence_details,
    }
    return ledger, pairs


def scan_nonmetadata_mentions(
    workspace: Path, proposal_dir: Path
) -> tuple[set[tuple[int, int]], dict[str, list[str]]]:
    pair_paths: dict[tuple[int, int], set[str]] = defaultdict(set)
    for directory, dirnames, filenames in os.walk(workspace):
        current = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".git", "__pycache__", "headers"}
            and (current / name).resolve() != proposal_dir.resolve()
        ]
        if proposal_dir.resolve() in current.resolve().parents or current.resolve() == proposal_dir.resolve():
            dirnames[:] = []
            continue
        for filename in filenames:
            path = current / filename
            rel = path.relative_to(workspace).as_posix()
            if rel == "qwen_weight_cache/model.safetensors.index.json":
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > 20_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for match in TENSOR_RE.finditer(text):
                pair_paths[(int(match.group(1)), int(match.group(2)))].add(rel)
    return set(pair_paths), {
        f"L{layer}:E{expert}": sorted(paths)
        for (layer, expert), paths in sorted(pair_paths.items())
    }


def load_metadata(cache: Path) -> tuple[bytes, dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    index_path = cache / "model.safetensors.index.json"
    index_raw = index_path.read_bytes()
    index = json.loads(index_raw.decode("utf-8"))
    if not isinstance(index.get("weight_map"), dict):
        raise AssertionError("invalid safetensors index")
    shards = sorted(set(str(value) for value in index["weight_map"].values()))
    headers: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for shard in shards:
        path = cache / "headers" / f"{shard}.header.json"
        wrapped_raw = path.read_bytes()
        wrapped = json.loads(wrapped_raw.decode("utf-8"))
        header = wrapped["header"]
        if not isinstance(header, dict):
            raise AssertionError(f"invalid cached header: {shard}")
        headers[shard] = {
            "header_length": int(wrapped["header_length"]),
            "header": header,
        }
        receipts.append(
            {
                "shard": shard,
                "header_length": int(wrapped["header_length"]),
                "wrapped_header_file_sha256": sha256_bytes(wrapped_raw),
                "header_canonical_sha256": sha256_bytes(canonical_bytes(header)),
            }
        )
    if len(shards) != 16:
        raise AssertionError(f"expected 16 shards, found {len(shards)}")
    return index_raw, index, headers, receipts


def validate_population(index: dict[str, Any], headers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    count = 0
    for layer in range(48):
        for expert in range(128):
            for role in ROLES:
                name = tensor_name(layer, expert, role)
                shard = str(index["weight_map"].get(name, ""))
                if shard not in headers or name not in headers[shard]["header"]:
                    raise AssertionError(f"index/header population gap: {name}")
                metadata = headers[shard]["header"][name]
                shape = tuple(int(value) for value in metadata["shape"])
                offsets = tuple(int(value) for value in metadata["data_offsets"])
                if metadata["dtype"] != "BF16" or shape != ROLE_SHAPES[role]:
                    raise AssertionError(f"unexpected metadata: {name}")
                if offsets[1] - offsets[0] != math.prod(shape) * 2:
                    raise AssertionError(f"offset span mismatch: {name}")
                count += 1
    return {
        "layers": 48,
        "experts_per_layer": 128,
        "roles_per_pair": 3,
        "validated_tensor_count": count,
        "dtype": "BF16",
        "validation_inputs": "cached safetensors index and JSON headers only",
    }


def candidate_score(seed: bytes, stratum: int, layer: int, expert: int) -> str:
    return sha256_bytes(seed + struct.pack(">BBB", stratum, layer, expert))


def block_seeds(tensor: str, block_index: int) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"{PINNED_REVISION}:{tensor}:{block_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[4:12], "big")


def choose_pairs(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    excluded_layers = set(int(value) for value in ledger["excluded_layer_indices"])
    excluded_experts = set(int(value) for value in ledger["excluded_expert_indices"])
    seed = bytes.fromhex(V1_AUDIT_SUMMARY_SHA256)
    selected: list[dict[str, Any]] = []
    for stratum, (low, high) in enumerate(STRATA):
        candidates: list[tuple[str, int, int]] = []
        for layer in range(low, high + 1):
            for expert in range(128):
                if layer in excluded_layers or expert in excluded_experts:
                    continue
                candidates.append((candidate_score(seed, stratum, layer, expert), layer, expert))
        candidates.sort()
        if not candidates:
            raise AssertionError(f"no eligible candidates in stratum {stratum}")
        score, layer, expert = candidates[0]
        selected.append(
            {
                "stratum": stratum,
                "layer_interval_inclusive": [low, high],
                "raw_candidate_count": 8 * 128,
                "eligible_candidate_count": len(candidates),
                "selected_layer": layer,
                "selected_expert": expert,
                "winning_score_sha256": score,
                "runner_up_scores": [
                    {"score_sha256": row[0], "layer": row[1], "expert": row[2]}
                    for row in candidates[1:4]
                ],
            }
        )
    return selected


def build_matrices(
    selected: list[dict[str, Any]], index: dict[str, Any], headers: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    matrices: list[dict[str, Any]] = []
    for choice in selected:
        layer = int(choice["selected_layer"])
        expert = int(choice["selected_expert"])
        for role in ROLES:
            name = tensor_name(layer, expert, role)
            shard = str(index["weight_map"][name])
            header_length = int(headers[shard]["header_length"])
            metadata = headers[shard]["header"][name]
            shape = [int(value) for value in metadata["shape"]]
            offsets = [int(value) for value in metadata["data_offsets"]]
            values = math.prod(shape)
            blocks = []
            for block_index in range(values // BLOCK_VALUES):
                sc_seed, rht_seed = block_seeds(name, block_index)
                blocks.append(
                    {
                        "canonical_block_index": block_index,
                        "flat_value_start": block_index * BLOCK_VALUES,
                        "flat_value_end_exclusive": (block_index + 1) * BLOCK_VALUES,
                        "sc_seed_u32": sc_seed,
                        "rht_seed_u64": rht_seed,
                        "source_bf16_sha256": None,
                    }
                )
            matrices.append(
                {
                    "matrix_ordinal": len(matrices),
                    "stratum": int(choice["stratum"]),
                    "layer": layer,
                    "expert": expert,
                    "role": role.removesuffix("_proj"),
                    "tensor": name,
                    "shard": shard,
                    "dtype": metadata["dtype"],
                    "shape": shape,
                    "nvalues": values,
                    "nbytes": offsets[1] - offsets[0],
                    "block_values": BLOCK_VALUES,
                    "block_count": len(blocks),
                    "shard_data_offsets": offsets,
                    "absolute_http_byte_range_inclusive": [
                        8 + header_length + offsets[0],
                        8 + header_length + offsets[1] - 1,
                    ],
                    "future_output_relpath": f"sources/{name}.bf16.bin",
                    "source_bf16_sha256": None,
                    "blocks": blocks,
                }
            )
    return matrices


def build_route(matrices: list[dict[str, Any]], selection_file_sha256: str, selection_internal_sha256: str) -> tuple[bytes, dict[str, Any]]:
    raw = bytearray()
    rows = []
    for ordinal, matrix in enumerate(matrices):
        role = str(matrix["role"])
        axis = "column" if role == "down" else "row"
        groups = int(matrix["shape"][1] if axis == "column" else matrix["shape"][0])
        if groups != 768:
            raise AssertionError("unexpected natural group count")
        record = ROUTE_RECORD.pack(
            int(matrix["layer"]), int(matrix["expert"]), ROLE_ENUM[role], AXIS_ENUM[axis], groups
        )
        raw.extend(record)
        rows.append(
            {
                "matrix_ordinal": ordinal,
                "layer": int(matrix["layer"]),
                "expert": int(matrix["expert"]),
                "role": role,
                "role_enum_u8": ROLE_ENUM[role],
                "natural_group_axis": axis,
                "axis_enum_u8": AXIS_ENUM[axis],
                "natural_group_count_u16": groups,
                "record_hex": record.hex(),
                "tensor": matrix["tensor"],
            }
        )
    route_raw = bytes(raw)
    audit = seal(
        {
            "schema": "polaris_strata_blind_route_table_proposal_v2",
            "status": "sealed_metadata_only_route_proposal_not_codec_frozen",
            "passed": True,
            "tensor_payload_bytes_read": 0,
            "selection_proposal_internal_sha256": selection_internal_sha256,
            "selection_proposal_file_sha256": selection_file_sha256,
            "format": {
                "framing": "headerless fixed records",
                "record_struct": ">HHBBH",
                "record_bytes": ROUTE_RECORD.size,
                "byte_order": "big-endian",
                "fields": ["layer_u16", "expert_u16", "role_enum_u8", "axis_enum_u8", "natural_group_count_u16"],
                "role_enum": ROLE_ENUM,
                "axis_enum": AXIS_ENUM,
            },
            "matrix_count": len(matrices),
            "physical_bytes_if_adopted": len(route_raw),
            "physical_bits_if_adopted": len(route_raw) * 8,
            "route_table_sha256": sha256_bytes(route_raw),
            "rows": rows,
            "execution_authorized": False,
        }
    )
    return route_raw, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--metadata-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--replace-existing-proposal",
        action="store_true",
        help="replace only this program's unfrozen proposal outputs after validation",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    metadata_cache = (args.metadata_cache or workspace / "qwen_weight_cache").resolve(strict=True)
    output_dir = args.output_dir.resolve()

    v1_summary = workspace / "frozen_strata_blind_independent_audit_v1/summary.json"
    if sha256_file(v1_summary) != V1_AUDIT_SUMMARY_SHA256:
        raise AssertionError("v1 audit summary seed artifact drift")
    v1 = load_object(v1_summary)
    if v1.get("claim_passed") or v1.get("mse_passed") or v1.get("status") != "complete_primary_claim_failed":
        raise AssertionError("v1 seed artifact does not represent the recorded failed claim")

    ledger, excluded_pairs = build_contamination_ledger(workspace)
    artifact_pairs, artifact_pair_paths = scan_nonmetadata_mentions(workspace, output_dir)
    if not artifact_pairs <= excluded_pairs:
        raise AssertionError(
            f"unledgered concrete expert pairs in prior artifacts: {sorted(artifact_pairs - excluded_pairs)}"
        )

    index_raw, index, headers, header_receipts = load_metadata(metadata_cache)
    population = validate_population(index, headers)
    selected = choose_pairs(ledger)
    selected_pairs = {(int(row["selected_layer"]), int(row["selected_expert"])) for row in selected}
    selected_layers = {layer for layer, _ in selected_pairs}
    selected_experts = {expert for _, expert in selected_pairs}
    if selected_pairs & excluded_pairs:
        raise AssertionError("selected pair intersects contamination ledger")
    if selected_layers & set(ledger["excluded_layer_indices"]):
        raise AssertionError("selected layer was previously opened/reserved")
    if selected_experts & set(ledger["excluded_expert_indices"]):
        raise AssertionError("selected expert index was previously opened/reserved")
    if selected_pairs & artifact_pairs:
        raise AssertionError("selected pair appears in a nonmetadata prior artifact")

    matrices = build_matrices(selected, index, headers)
    candidate_tensors = {str(row["tensor"]) for row in matrices}
    if len(matrices) != 18 or len(candidate_tensors) != 18:
        raise AssertionError("panel matrix cardinality failure")
    if sum(int(row["block_count"]) for row in matrices) != 108:
        raise AssertionError("panel block cardinality failure")

    full_shards = []
    candidate_path_hits = []
    alias_patterns = {
        (layer, expert): re.compile(
            rf"(?:^|[^0-9])l(?:ayer)?[._-]?0*{layer}[._-]?e(?:xpert)?[._-]?0*{expert}(?:[^0-9]|$)",
            re.IGNORECASE,
        )
        for layer, expert in selected_pairs
    }
    alias_path_hits: dict[str, list[str]] = defaultdict(list)
    for directory, dirnames, filenames in os.walk(workspace):
        current = Path(directory)
        dirnames[:] = [name for name in dirnames if name not in {".git", "__pycache__"}]
        if output_dir.resolve() in current.resolve().parents or current.resolve() == output_dir.resolve():
            dirnames[:] = []
            continue
        for filename in filenames:
            path = current / filename
            rel = path.relative_to(workspace).as_posix()
            if SHARD_RE.fullmatch(filename):
                full_shards.append(rel)
            if any(tensor in rel for tensor in candidate_tensors):
                candidate_path_hits.append(rel)
            for pair, pattern in alias_patterns.items():
                if pattern.search(rel):
                    alias_path_hits[f"L{pair[0]}:E{pair[1]}"].append(rel)
    if full_shards or candidate_path_hits or alias_path_hits:
        raise AssertionError(
            f"candidate unopened preflight failed: full_shards={full_shards}, candidate_paths={candidate_path_hits}, aliases={dict(alias_path_hits)}"
        )

    # Confirm the selected exact byte ranges do not overlap any finalized v1
    # full-matrix range in the same shard.  Other opened sources are excluded
    # by identity/layer/expert and recorded separately in the ledger.
    v1_source = load_object(workspace / "blind_protocol/unblinded/source_hashes.lock.json")
    range_overlap_rows = []
    for candidate in matrices:
        c0, c1 = map(int, candidate["absolute_http_byte_range_inclusive"])
        for old in v1_source["matrices"]:
            if old["shard"] != candidate["shard"]:
                continue
            o0, o1 = map(int, old["http_range_inclusive"])
            if max(c0, o0) <= min(c1, o1):
                range_overlap_rows.append(
                    {"candidate": candidate["tensor"], "v1_tensor": old["tensor"], "overlap": [max(c0, o0), min(c1, o1)]}
                )
    if range_overlap_rows:
        raise AssertionError(f"selected ranges overlap v1 payload ranges: {range_overlap_rows}")

    v1_reaudit_path = output_dir / "v1_failure_independent_audit.json"
    if not v1_reaudit_path.exists():
        raise FileNotFoundError("run audit_v1_result.py before preparing the panel proposal")
    v1_reaudit = load_object(v1_reaudit_path)
    verify_seal(v1_reaudit)
    if v1_reaudit.get("status") != "complete_primary_claim_reconfirmed_failed":
        raise AssertionError("v1 independent re-audit receipt is not the expected failure receipt")

    try:
        proposal_relpath = output_dir.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise ValueError("proposal output directory must be inside the workspace") from exc
    unopened_snapshot = seal(
        {
            "schema": "int2-qwen-second-panel-unopened-snapshot-v1",
            "status": "metadata_only_snapshot_candidates_have_no_workspace_payload_evidence",
            "passed": True,
            "scope_caveat": "proves absence in the audited workspace and access manifests, not absence from every external machine or network log",
            "workspace_logical_root": ".",
            "proposal_output_relpath_excluded_from_prior_artifact_scan": proposal_relpath,
            "nonmetadata_concrete_pair_mentions": artifact_pair_paths,
            "nonmetadata_pair_count": len(artifact_pairs),
            "all_nonmetadata_pairs_covered_by_contamination_ledger": True,
            "candidate_pairs": [list(pair) for pair in sorted(selected_pairs)],
            "candidate_tensors": sorted(candidate_tensors),
            "candidate_tensor_path_hits_before_proposal": [],
            "candidate_alias_path_hits_before_proposal": {},
            "full_safetensors_shards_present_in_workspace": [],
            "selected_range_overlap_with_finalized_v1_full_matrix_ranges": [],
            "metadata_inputs": {
                "index": "qwen_weight_cache/model.safetensors.index.json",
                "index_file_sha256": sha256_bytes(index_raw),
                "wrapped_headers": header_receipts,
            },
            "selector_network_calls": 0,
            "selector_tensor_payload_files_opened": 0,
            "selector_tensor_payload_bytes_read": 0,
        }
    )

    selection = seal(
        {
            "schema": "int2-qwen-blind-selection-proposal-v2",
            "status": "sealed_metadata_only_proposal_payload_unopened_not_codec_frozen",
            "checkpoint": {"repo": PINNED_REPO, "revision": PINNED_REVISION},
            "proposal_semantics": {
                "is_codec_freeze": False,
                "authorizes_payload_access": False,
                "authorizes_encode": False,
                "purpose": "precommit a rigorously disjoint second panel while codec development remains separate",
            },
            "seed": {
                "source": "raw SHA-256 of the immutable v1 independent-audit summary file",
                "v1_audit_summary_file_sha256": V1_AUDIT_SUMMARY_SHA256,
                "score_preimage": "32 seed bytes || uint8(stratum) || uint8(layer) || uint8(expert)",
                "score": "SHA-256",
                "why_not_free_form": "the failed v1 outcome fixes all seed bytes; there is no user-selected salt, timestamp, or retry counter",
            },
            "selection_algorithm": {
                "strata": [list(value) for value in STRATA],
                "sampling_unit": "one (layer, expert) triplet containing gate/up/down",
                "eligibility": "layer index absent from all prior opened/reserved pairs AND expert index absent from all prior opened/reserved pairs",
                "choice": "lexicographically smallest 32-byte score in each stratum",
                "role_order": list(ROLES),
                "flattening": "safetensors row-major contiguous order",
                "block_values": BLOCK_VALUES,
                "uniformity_model": "uniform over the declared eligible set in each stratum under the SHA-256 random-oracle model",
            },
            "contamination_ledger": ledger,
            "unopened_snapshot": {
                "lock_sha256": unopened_snapshot["lock_sha256"],
                "status": unopened_snapshot["status"],
                "payload_bytes_read": 0,
            },
            "metadata_provenance": {
                "mode": "strictly offline cached index/header validation",
                "allowed_inputs": ["model.safetensors.index.json", "safetensors JSON headers"],
                "index_raw_sha256": sha256_bytes(index_raw),
                "index_canonical_sha256": sha256_bytes(canonical_bytes(index)),
                "shard_headers": header_receipts,
                "population_validation": population,
                "tensor_payload_bytes_read": 0,
            },
            "selected_pairs": selected,
            "matrices": matrices,
            "panel_totals": {
                "matrix_count": len(matrices),
                "block_count": sum(int(row["block_count"]) for row in matrices),
                "source_values": sum(int(row["nvalues"]) for row in matrices),
                "source_bytes": sum(int(row["nbytes"]) for row in matrices),
                "roles": {"gate": 6, "up": 6, "down": 6},
            },
            "seed_derivation": {
                "block_key_utf8": "{revision}:{tensor}:{canonical_block_index}",
                "digest": "SHA-256",
                "sc_seed_u32": "big-endian digest bytes [0:4]",
                "rht_seed_u64": "big-endian digest bytes [4:12]",
            },
            "future_source_finalization": {
                "selection_proposal_is_immutable_if_adopted": True,
                "source_hashes_remain_null_here": True,
                "matrix_hash": "SHA-256 of exact contiguous BF16 payload bytes",
                "block_hash": f"SHA-256 of each consecutive {BLOCK_VALUES * 2}-byte slice",
                "required_order": "first freeze codec and blind allocator against this exact proposal; only then create a separate source-finalization lock",
            },
            "gates_if_adopted": {
                "physical_rate_bpw_max": 2.15,
                "gaussian_relative_mse_limit": 0.050765774772264724,
                "primary_pass": "integrity && exact route-inclusive physical bpw <= 2.15 && pooled original-BF16 FP64 SSE/energy < 2^-4.3",
                "reporting": "v2 is a new confirmatory panel; never pool/select it with v1 after observing either result",
            },
            "payload_interlock": {
                "authorized": False,
                "materializer_present": False,
                "codec_freeze_present_for_v2": False,
                "next_permitted_step": "continue source-free codec work or create an audited codec freeze; do not fetch selected ranges",
            },
            "v1_failure_reaudit": {
                "receipt_status_verified": v1_reaudit["status"],
                "energy_weighted_relative_mse_cupy": v1_reaudit["distortion"]["energy_weighted_relative_mse_cupy"],
                "gaussian_limit_at_2p15": v1_reaudit["distortion"]["gaussian_limit_at_2p15"],
                "v1_claim_passed": False,
                "note": "the separate machine-specific re-audit receipt is evidence, not selection randomness or a path-dependent selection binding",
            },
        }
    )
    selection_raw = render_json(selection)
    selection_file_sha256 = sha256_bytes(selection_raw)
    route_raw, route_audit = build_route(
        matrices, selection_file_sha256, selection["lock_sha256"]
    )

    replacement_targets = [
        output_dir / "unopened_snapshot.audit.json",
        output_dir / "selection.proposal.lock.json",
        output_dir / "route_table.proposal.bin",
        output_dir / "route_table.proposal.audit.json",
    ]
    if args.replace_existing_proposal:
        selection_path = output_dir / "selection.proposal.lock.json"
        if selection_path.exists():
            old = load_object(selection_path)
            verify_seal(old)
            if old.get("schema") != "int2-qwen-blind-selection-proposal-v2" or old.get("proposal_semantics", {}).get("is_codec_freeze") is not False:
                raise AssertionError("replacement target is not this program's unfrozen proposal")
        forbidden = [output_dir / "codec_freeze.lock.json", output_dir / "unblinded"]
        if any(path.exists() for path in forbidden):
            raise AssertionError("refusing proposal replacement after freeze/unblind state exists")
    write_new_or_identical(replacement_targets[0], render_json(unopened_snapshot), replace_generated=args.replace_existing_proposal)
    write_new_or_identical(replacement_targets[1], selection_raw, replace_generated=args.replace_existing_proposal)
    write_new_or_identical(replacement_targets[2], route_raw, replace_generated=args.replace_existing_proposal)
    write_new_or_identical(replacement_targets[3], render_json(route_audit), replace_generated=args.replace_existing_proposal)
    print(
        json.dumps(
            {
                "passed": True,
                "status": selection["status"],
                "selection_internal_sha256": selection["lock_sha256"],
                "selection_file_sha256": selection_file_sha256,
                "route_table_sha256": sha256_bytes(route_raw),
                "pairs": [[row["selected_layer"], row["selected_expert"]] for row in selected],
                "matrix_count": len(matrices),
                "payload_bytes_read": 0,
                "codec_freeze_created": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
