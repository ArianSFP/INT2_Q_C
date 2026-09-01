#!/usr/bin/env python3
"""Pure-stdlib reconstruction of the frozen FUSEED-v1 plan and PMG ABI1 subset."""

from __future__ import annotations

import hashlib
import json


T = 261120
ROWS = 768
COLUMNS = 2048
ABI_IDS = (
    "PAI6BA_LGP_GATE_UP_DIRECT_BF16",
    "CURRENT_PMG_GATE_UP_DIRECT_BF16",
    "CURRENT_PMG_UP_GATE_DIRECT_BF16",
)
UP_NUMELS = (100663296, 3145728, 3145728)
DOWN_NUMELS = (50331648, 1572864, 1572864)
EXPECTED_V1_STAGE0_SHA256 = "97ac7933a1d3735960bed34977279d45212c553e1717d03bd7c87fe4ff7e9981"
EXPECTED_V1_ALL_PLAN_SHA256 = "f19492e5ed1cc93949f1c9ca8038576a7fe17fe7f519b473773d721d17d6f260"
EXPECTED_V1_BUNDLE_ATTEMPTS = 17151
EXPECTED_ABI1_STAGE0_SUBSET_SHA256 = "d4cb9d33f09779ed10c1993ccc08bb4aa8a864f0f975f2ec86a2922a8729749f"
EXPECTED_ABI1_CATEGORY_BUNDLE_SHA256 = "0d2f6cdeebe8134d463e147d331ef671d4c10d9b0e2968e75dd31ff61cf39245"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def u32_le(digest: bytes, offset: int) -> int:
    return int.from_bytes(digest[offset : offset + 4], "little")


def hash_text(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def canonical_key(expert: int, role: str, row: int, column: int) -> str:
    return f"e{expert:03d}|{role}|r{row:03d}|c{column:04d}"


def invert_native(abi: int, wanted_expert: int, role: str, native: int):
    if abi == 0:
        if role == "up":
            block = 2048 * 1536
            local_expert, within = divmod(native, block)
            column, fused_row = divmod(within, 1536)
            if local_expert != wanted_expert % 32 or fused_row < 768:
                return None
            return fused_row - 768, column
        block = 768 * 2048
        local_expert, within = divmod(native, block)
        row, column = divmod(within, 2048)
        if local_expert != wanted_expert % 32:
            return None
        return row, column
    if role == "up":
        fused_row, column = divmod(native, 2048)
        if abi == 1:
            if not 768 <= fused_row < 1536:
                return None
            return fused_row - 768, column
        if not 0 <= fused_row < 768:
            return None
        return fused_row, column
    column, row = divmod(native, 768)
    if not 0 <= column < 2048:
        return None
    return row, column


def selection_identities():
    return [
        (expert, role)
        for expert in (0, 8, 16, 32, 40, 48, 64, 72, 80, 96, 104, 112)
        for role in ("up", "down")
        if not (expert == 0 and role == "up")
    ]


def validation_identities():
    return [
        (expert, role)
        for expert in (24, 56, 88, 120)
        for role in ("up", "down")
    ]


def high_count_identities(identities):
    result = set()
    for role, count in (("up", 6), ("down", 7)):
        ranked = sorted(
            (
                sha256_text(f"FUSEED-U32-v1|quota|{expert}|{candidate_role}"),
                canonical_key(expert, candidate_role, 0, 0),
                expert,
                candidate_role,
            )
            for expert, candidate_role in identities
            if candidate_role == role
        )
        result.update((row[2], row[3]) for row in ranked[:count])
    if len(result) != 13:
        raise RuntimeError("high-count identity cardinality mismatch")
    return result


def enumerate_stage0():
    identities = selection_identities()
    high_count = high_count_identities(identities)
    global_by_split = {"fit": set(), "score": set()}
    all_lines: list[str] = []
    subset_lines: dict[int, list[str]] = {abi: [] for abi in range(3)}
    bundles: dict[int, list[dict]] = {abi: [] for abi in range(3)}
    attempts = 0
    for abi, abi_id in enumerate(ABI_IDS):
        used_by_identity: dict[tuple[int, str], set[str]] = {}
        for expert, role in identities:
            identity = (expert, role)
            used_by_identity[identity] = set()
            coordinate_count = 24 if identity in high_count else 20
            for split in ("fit", "score"):
                bundle_target = coordinate_count // 4
                accepted = 0
                counter = 0
                numel = UP_NUMELS[abi] if role == "up" else DOWN_NUMELS[abi]
                valid_bundle_count = (numel - 1 - 3 * T) // (4 * T) + 1
                if valid_bundle_count <= 0:
                    raise RuntimeError("invalid full-bundle count")
                while accepted < bundle_target:
                    if counter >= 1_000_000:
                        raise RuntimeError("bundle search did not terminate")
                    digest = hash_text(
                        f"FUSEED-U32-v1|{abi_id}|{expert}|{role}|{split}|{counter}"
                    )
                    sequence = u32_le(digest, 0) % T
                    normal4_index = u32_le(digest, 4) % valid_bundle_count
                    coords = []
                    local_keys = set()
                    valid = True
                    for lane in range(4):
                        native = sequence + T * (4 * normal4_index + lane)
                        inverted = invert_native(abi, expert, role, native)
                        if inverted is None:
                            valid = False
                            break
                        row, column = inverted
                        key = canonical_key(expert, role, row, column)
                        if (
                            key in local_keys
                            or key in used_by_identity[identity]
                            or key in global_by_split["score" if split == "fit" else "fit"]
                        ):
                            valid = False
                            break
                        local_keys.add(key)
                        coords.append(
                            {
                                "row": row,
                                "column": column,
                                "native": native,
                                "key": key,
                                "lane": lane,
                            }
                        )
                    counter += 1
                    attempts += 1
                    if not valid:
                        continue
                    for coord in coords:
                        used_by_identity[identity].add(coord["key"])
                        global_by_split[split].add(coord["key"])
                        line = (
                            f"stage0|abi={abi}|{split}|e={expert}|role={role}|"
                            f"r={coord['row']}|c={coord['column']}|seq={sequence}|"
                            f"j={normal4_index}|lane={coord['lane']}|native={coord['native']}"
                        )
                        all_lines.append(line)
                        subset_lines[abi].append(line)
                    bundles[abi].append(
                        {
                            "abi": abi,
                            "expert": expert,
                            "role": role,
                            "split": split,
                            "sequence": sequence,
                            "normal4_index": normal4_index,
                            "coordinates": coords,
                        }
                    )
                    accepted += 1
    if len(all_lines) != 3 * 1024:
        raise RuntimeError("stage0 record cardinality mismatch")
    return identities, global_by_split, all_lines, subset_lines, bundles, attempts


def fill_plan(namespace: str, split: str, values: set[str], opposite: set[str], identities, target: int):
    counter = 0
    while len(values) < target:
        if counter >= 10_000_000:
            raise RuntimeError("plan fill did not terminate")
        digest = hash_text(f"FUSEED-U32-v1|{namespace}|{split}|{counter}")
        expert, role = identities[u32_le(digest, 0) % len(identities)]
        row = u32_le(digest, 4) % ROWS
        column = u32_le(digest, 8) % COLUMNS
        key = canonical_key(expert, role, row, column)
        if key not in opposite:
            values.add(key)
        counter += 1


def reconstruct_plan() -> dict:
    identities, global_by_split, stage0_lines, subset_lines, bundles, attempts = enumerate_stage0()
    all_lines = list(stage0_lines)
    stage1_fit = set(global_by_split["fit"])
    stage1_score = set(global_by_split["score"])
    if stage1_fit & stage1_score:
        raise RuntimeError("stage0 fit/score overlap")
    fill_plan("stage1", "fit", stage1_fit, stage1_score, identities, 2048)
    fill_plan("stage1", "score", stage1_score, stage1_fit, identities, 2048)
    all_lines.extend(f"stage1|fit|{key}" for key in sorted(stage1_fit))
    all_lines.extend(f"stage1|score|{key}" for key in sorted(stage1_score))

    full_fit = set(stage1_fit)
    full_score = set(stage1_score)
    fill_plan("stage2", "fit", full_fit, full_score, identities, 24312)
    fill_plan("stage2", "score", full_score, full_fit, identities, 24312)
    if full_fit & full_score:
        raise RuntimeError("full fit/score overlap")
    all_lines.extend(f"stage2|fit|{key}" for key in sorted(full_fit))
    all_lines.extend(f"stage2|score|{key}" for key in sorted(full_score))

    validation = validation_identities()
    validation_fit: set[str] = set()
    validation_score: set[str] = set()
    fill_plan("validation", "fit", validation_fit, validation_score, validation, 8456)
    fill_plan("validation", "score", validation_score, validation_fit, validation, 8456)
    if validation_fit & validation_score:
        raise RuntimeError("validation fit/score overlap")
    all_lines.extend(f"validation|fit|{key}" for key in sorted(validation_fit))
    all_lines.extend(f"validation|score|{key}" for key in sorted(validation_score))

    stage0_payload = "\n".join(stage0_lines) + "\n"
    all_payload = "\n".join(all_lines) + "\n"
    stage0_digest = sha256_text(stage0_payload)
    all_digest = sha256_text(all_payload)
    if stage0_digest != EXPECTED_V1_STAGE0_SHA256:
        raise RuntimeError(f"v1 stage0 digest mismatch: {stage0_digest}")
    if all_digest != EXPECTED_V1_ALL_PLAN_SHA256:
        raise RuntimeError(f"v1 all-plan digest mismatch: {all_digest}")
    if attempts != EXPECTED_V1_BUNDLE_ATTEMPTS:
        raise RuntimeError(f"v1 attempt count mismatch: {attempts}")

    abi1_categories = []
    for split, role in (("fit", "up"), ("fit", "down"), ("score", "up"), ("score", "down")):
        abi1_categories.extend(
            bundle
            for bundle in bundles[1]
            if bundle["split"] == split and bundle["role"] == role
        )
    if len(abi1_categories) != 256:
        raise RuntimeError("ABI1 category-ordered bundle count mismatch")
    category_counts = {
        "up_fit": sum(row["split"] == "fit" and row["role"] == "up" for row in abi1_categories),
        "down_fit": sum(row["split"] == "fit" and row["role"] == "down" for row in abi1_categories),
        "up_score": sum(row["split"] == "score" and row["role"] == "up" for row in abi1_categories),
        "down_score": sum(row["split"] == "score" and row["role"] == "down" for row in abi1_categories),
    }
    if category_counts != {"up_fit": 61, "down_fit": 67, "up_score": 61, "down_score": 67}:
        raise RuntimeError(f"ABI1 category bundle counts mismatch: {category_counts}")
    wire = []
    for bundle in abi1_categories:
        expert = bundle["expert"]
        local_expert = expert % 32
        role = bundle["role"]
        offset_values = (
            11520 + 16 * local_expert if role == "up" else 12032 + 8 * local_expert
        )
        wire.append(
            {
                "expert": expert,
                "role": role,
                "split": bundle["split"],
                "seed_addend": 1024 + 100 * (expert // 32),
                "sequence": bundle["sequence"],
                "offset_values": offset_values,
                "offset_quads": offset_values // 4,
                "normal4_index": bundle["normal4_index"],
                "coordinates": [
                    {
                        "expert": expert,
                        "role": role,
                        "row": coord["row"],
                        "column": coord["column"],
                        "lane": coord["lane"],
                        "native": coord["native"],
                    }
                    for coord in bundle["coordinates"]
                ],
            }
        )
    wire_bytes = json.dumps(wire, sort_keys=True, separators=(",", ":")).encode("utf-8")
    subset_payload = "\n".join(subset_lines[1]) + "\n"
    subset_digest = sha256_text(subset_payload)
    bundle_digest = hashlib.sha256(wire_bytes).hexdigest()
    if subset_digest != EXPECTED_ABI1_STAGE0_SUBSET_SHA256:
        raise RuntimeError(f"ABI1 stage0 subset digest mismatch: {subset_digest}")
    if bundle_digest != EXPECTED_ABI1_CATEGORY_BUNDLE_SHA256:
        raise RuntimeError(f"ABI1 category bundle digest mismatch: {bundle_digest}")
    return {
        "facts": {
            "v1_stage0_sha256": stage0_digest,
            "v1_all_plan_sha256": all_digest,
            "v1_bundle_attempts": attempts,
            "abi1_stage0_subset_sha256": subset_digest,
            "abi1_stage0_record_count": len(subset_lines[1]),
            "abi1_category_ordered_bundle_sha256": bundle_digest,
            "abi1_category_ordered_bundle_count": len(wire),
            "category_bundle_counts": category_counts,
            "stage1_fit_count": len(stage1_fit),
            "stage1_score_count": len(stage1_score),
            "full_fit_count": len(full_fit),
            "full_score_count": len(full_score),
            "validation_fit_count": len(validation_fit),
            "validation_score_count": len(validation_score),
        },
        "wire": wire,
    }


if __name__ == "__main__":
    result = reconstruct_plan()
    print(json.dumps(result["facts"], indent=2, sort_keys=True))
