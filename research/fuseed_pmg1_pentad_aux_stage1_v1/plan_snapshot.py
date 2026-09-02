#!/usr/bin/env python3
"""Exact standard-library reconstruction of the frozen 2048/2048 PMG plan.

This is a narrow snapshot of the already-frozen PMG ABI1 plan.  It generates
only the stage-1 Up/Down coordinate keys needed by the fixed-pentad screen and
checks their historical digests.  Importing it has no external effects.
"""

from __future__ import annotations

import hashlib
import re


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
FIT_KEY_SHA256 = "42a2cb8170a1de43f23ee399ef93687cf738fce42d9f314b9273839855961f9e"
SCORE_KEY_SHA256 = "c112da528fcedfbcf62a0f71ea3f63150cc2b68a0bc7b0d34d63290580f0d7bb"
KEY_RE = re.compile(r"^e(\d{3})\|(up|down)\|r(\d{3})\|c(\d{4})$")


class PlanError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_text(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def u32_le(digest: bytes, offset: int) -> int:
    return int.from_bytes(digest[offset : offset + 4], "little")


def canonical_key(expert: int, role: str, row: int, column: int) -> str:
    return f"e{expert:03d}|{role}|r{row:03d}|c{column:04d}"


def parse_key(key: str) -> tuple[int, str, int, int]:
    match = KEY_RE.fullmatch(key)
    require(match is not None, f"malformed key: {key}")
    expert, role, row, column = match.groups()
    result = (int(expert), role, int(row), int(column))
    require(result[0] in selection_experts(), "key expert")
    require(0 <= result[2] < ROWS and 0 <= result[3] < COLUMNS, "key coordinate")
    return result


def selection_experts() -> tuple[int, ...]:
    return (0, 8, 16, 32, 40, 48, 64, 72, 80, 96, 104, 112)


def selection_identities() -> list[tuple[int, str]]:
    rows = [
        (expert, role)
        for expert in selection_experts()
        for role in ("up", "down")
        if not (expert == 0 and role == "up")
    ]
    require(len(rows) == 23, "selection identity count")
    return rows


def high_count_identities(identities: list[tuple[int, str]]) -> set[tuple[int, str]]:
    result: set[tuple[int, str]] = set()
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
    require(len(result) == 13, "high-count identity count")
    return result


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


def _enumerate_stage0_sets() -> tuple[list[tuple[int, str]], dict[str, set[str]]]:
    identities = selection_identities()
    high_count = high_count_identities(identities)
    global_by_split: dict[str, set[str]] = {"fit": set(), "score": set()}
    records = 0
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
                require(valid_bundle_count > 0, "valid bundle count")
                while accepted < bundle_target:
                    require(counter < 1_000_000, "bundle search bound")
                    digest = hash_text(
                        f"FUSEED-U32-v1|{abi_id}|{expert}|{role}|{split}|{counter}"
                    )
                    sequence = u32_le(digest, 0) % T
                    normal4_index = u32_le(digest, 4) % valid_bundle_count
                    keys: list[str] = []
                    local_keys: set[str] = set()
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
                        keys.append(key)
                    counter += 1
                    if not valid:
                        continue
                    for key in keys:
                        used_by_identity[identity].add(key)
                        global_by_split[split].add(key)
                        records += 1
                    accepted += 1
    require(records == 3 * 1024, "stage0 record count")
    require(not (global_by_split["fit"] & global_by_split["score"]), "stage0 overlap")
    return identities, global_by_split


def _fill_plan(
    namespace: str,
    split: str,
    values: set[str],
    opposite: set[str],
    identities: list[tuple[int, str]],
    target: int,
) -> None:
    counter = 0
    while len(values) < target:
        require(counter < 10_000_000, "plan fill bound")
        digest = hash_text(f"FUSEED-U32-v1|{namespace}|{split}|{counter}")
        expert, role = identities[u32_le(digest, 0) % len(identities)]
        row = u32_le(digest, 4) % ROWS
        column = u32_le(digest, 8) % COLUMNS
        key = canonical_key(expert, role, row, column)
        if key not in opposite:
            values.add(key)
        counter += 1


def stage1_keys() -> tuple[tuple[str, ...], tuple[str, ...]]:
    identities, rows = _enumerate_stage0_sets()
    fit = set(rows["fit"])
    score = set(rows["score"])
    _fill_plan("stage1", "fit", fit, score, identities, 2048)
    _fill_plan("stage1", "score", score, fit, identities, 2048)
    require(len(fit) == 2048 and len(score) == 2048, "stage1 sizes")
    require(not (fit & score), "stage1 overlap")
    fit_keys = tuple(sorted(fit))
    score_keys = tuple(sorted(score))
    fit_hash = sha256_text("\n".join(fit_keys) + "\n")
    score_hash = sha256_text("\n".join(score_keys) + "\n")
    require(fit_hash == FIT_KEY_SHA256, f"fit key hash: {fit_hash}")
    require(score_hash == SCORE_KEY_SHA256, f"score key hash: {score_hash}")
    observed = {parse_key(key)[:2] for key in (*fit_keys, *score_keys)}
    require(observed == set(identities), "stage1 identity set")
    return fit_keys, score_keys


if __name__ == "__main__":
    fit_keys, score_keys = stage1_keys()
    print(
        {
            "fit": len(fit_keys),
            "score": len(score_keys),
            "fit_sha256": FIT_KEY_SHA256,
            "score_sha256": SCORE_KEY_SHA256,
        }
    )

