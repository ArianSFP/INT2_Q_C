#!/usr/bin/env python3
"""Independent, standard-library-only audit of the PMG1 binary64 calibrations.

This program deliberately opens only the exact, source-only producer members and
the exact runtime/header paths named by those members.  It never imports Torch,
CuPy, NumPy, or any producer module which imports them.
"""

from __future__ import annotations

import ast
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import stat
import statistics
import struct


AUDIT_SCHEMA = "fuseed_pmg1_binary64_fma_v2_independent_source_audit_v1"
DESIGN_MANIFEST_SHA = "ea7086c401cd6981d097ecc9b52196d3d01cda123d0cb8ab28c001cf008b27ff"
V1_SCRIPT_SHA = "9376720ec812b93e070ccb93433e83ff243213d6d244c7a18afa84b3d8690c24"
V1_RESULT_SHA = "2e2e6dc73a16921221cfd309a243fce6794464c201f80714ca3e2dcc94078c4d"
V2_SCRIPT_SHA = "0e2f354415d2d8cfebfceda58b6ade77eddc2b4e025488baba329beba09d0a87"
V2_RESULT_SHA = "82e29cbfc8ec1ac23761c37712a3fda3d2745b04c9a71ae296ce864796ddc75e"
PRIOR_V1_RESULT_SHA = "6ef38ff14f69ab02caf0e48ad37e5dbc3dfa9ebe7ba5e663f85080114e0f828d"
PRIOR_TIER_B_RESULT_SHA = "e450c10767b54c190f901df8460c6ac57fe86cfaca7719c3db48475d4196fb92"
ABI1_BUNDLE_SHA = "16aacb6f5fa6a1ed12fe0c01506410ad69585894077a4a6af627674b6b90adda"
COMPLETE_PLAN_SHA = "86639758eda1835b9ea9e883372bb55ec13ec3487705a91d892878972db74760"
TOP_K = 8192
SHARDS = 256
SHARD_SIZE = 1 << 24

EXPECTED_DESIGN_MEMBERS = {
    "DESIGN_RECEIPT.json": "5b5ff763b7eacb9d2d498a3f0fd930585391415ac3cd02d7835538a10c1f2e28",
    "README.md": "f2357858089c61051c7459fcafb5bbbf080ae49b4af9da186e91fc63447ec214",
    "RED_TEAM.md": "b432b0c08d1530276d542a8ab34862a6dbb763074cf29aa3a996617181a251bc",
    "design_lock.json": "29d0c149f3120fb4d74cf39c26bd8ee538f01895f4c1e6bd29feb1078c1d118f",
    "source_bindings.json": "eda7e8581f98caceb6d3ffb0ae95ef43b013596961b3e218b98446c1a50edbea",
    "verify_design.ps1": "8125bd8cfc9f21662cbd6724125ddede785419e40713b8c02388546b3697e540",
}

EXPECTED_JOURNALS = {
    "v1": (
        ("binary64_shard_replay_0.bin", "ce1f38ad8702d099bfcb951c021f63a69dead14afa544cefd52e3ab8d5e387a4", 99006),
        ("binary64_shard_replay_1.bin", "85e4e1315d0d8e3cd8e5f6f7f82c4350ca27c76fc7ffc7c186ed10b894fc41d1", 99006),
        ("binary64_shard_replay_2.bin", "4194b051c88247c7f5d17f20b1ad825f6c1b19aa588e1a236807f466ed140154", 99006),
    ),
    "v2": (
        ("binary64_shard_replay_0.bin", "b5d640d451bd39c1d18e2d319d1e1c74a0060c5b255df5edb73adf7da555f997", 99019),
        ("binary64_shard_replay_1.bin", "19ea5c141ab0be9fe74d3585b47c3f25855b4335fc07bf9ca6af2b7a901b1ee5", 99019),
        ("binary64_shard_replay_2.bin", "2737b32808dd2b3698c670ea73be235ff1742950507fab6e63320b51996ad32b", 99019),
    ),
}


class AuditFailure(RuntimeError):
    pass


checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        raise AuditFailure(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def regular_nonsymlink(path: Path) -> None:
    mode = path.lstat().st_mode
    check(stat.S_ISREG(mode), f"not regular: {path}")
    check(not path.is_symlink(), f"symlink rejected: {path}")


def strict_json_bytes(data: bytes, label: str):
    def pairs(values):
        out = {}
        for key, value in values:
            if key in out:
                raise AuditFailure(f"duplicate JSON key in {label}: {key}")
            out[key] = value
        return out

    def bad_constant(value):
        raise AuditFailure(f"nonfinite JSON constant in {label}: {value}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
    except UnicodeDecodeError as exc:
        raise AuditFailure(f"non-UTF8 JSON in {label}") from exc


def read_json(path: Path):
    regular_nonsymlink(path)
    return strict_json_bytes(path.read_bytes(), str(path))


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                found.append(ast.literal_eval(node.value))
    check(len(found) == 1, f"literal assignment cardinality {name}: {len(found)}")
    return found[0]


def isolated_function(path: Path, name: str, globals_dict=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    check(len(nodes) == 1, f"function cardinality {name}: {len(nodes)}")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), nodes[0]],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    env = {} if globals_dict is None else dict(globals_dict)
    exec(compile(module, str(path), "exec"), env)
    return env[name]


def manifest_map(path: Path) -> dict[str, str]:
    regular_nonsymlink(path)
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        check(len(parts) == 2, f"bad manifest row: {line!r}")
        digest, member = parts
        check(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "bad manifest digest")
        check(member not in rows, f"duplicate manifest member: {member}")
        check("/" not in member and "\\" not in member and member not in (".", ".."), f"unsafe manifest member: {member}")
        rows[member] = digest
    return rows


def reconstruct_sources(repo: Path, v1_result: dict, v2_result: dict) -> dict:
    shape = repo / "research/fuseed_u32_33domain_calibration_v0/calibrate.py"
    direct = repo / "research/fuseed_u32_direct_counter_calibration_v0/calibrate_direct.py"
    domain = repo / "research/fuseed_u32_direct_domain_collapse_probe_v0/probe.py"
    v1 = repo / "research/fuseed_pmg1_binary64_calibration_v1/calibrate_binary64.py"
    v2 = repo / "research/fuseed_pmg1_binary64_fma_calibration_v2/calibrate_fma.py"
    source = literal_assignment(shape, "CUDA_SOURCE")
    derive_direct = isolated_function(direct, "derive_direct_source")
    domain_counts = literal_assignment(domain, "DOMAIN_COUNTS")
    derive_domain = isolated_function(domain, "derive_active_domain_source", {"DOMAIN_COUNTS": domain_counts})
    derive_v1 = isolated_function(v1, "derive_binary64_capture_source")
    explicit_wrapper = isolated_function(v2, "explicit_fma_deriver")
    direct_source, direct_counts = derive_direct(source)
    active_source, active_counts = derive_domain(direct_source, 1)
    binary64_source, binary64_counts = derive_v1(active_source)
    explicit_source, explicit_counts = explicit_wrapper(derive_v1)(active_source)
    for result, derived, dc, ac, bc in (
        (v1_result, binary64_source, direct_counts, active_counts, binary64_counts),
        (v2_result, explicit_source, direct_counts, active_counts, explicit_counts),
    ):
        check(result["derivation"]["performance_cuda_sha256"] == sha256_bytes(derived.encode()), "derived CUDA hash mismatch")
        check(result["derivation"]["direct_replacement_counts"] == dc, "direct replacement counts mismatch")
        check(result["derivation"]["active_domain_replacement_counts"] == ac, "domain replacement counts mismatch")
        check(result["derivation"]["binary64_capture_replacement_counts"] == bc, "binary64 replacement counts mismatch")
    check(sha256_bytes(binary64_source.encode()) == "d0af6e0e93706b69c523d0bfded4ffb57717b827fa9c5bc69f0a37d7bb1c7d43", "v1 CUDA source digest")
    check(sha256_bytes(explicit_source.encode()) == "c2de564889b34f00ca9e2f937dd70d9804845957c8bae3e847c37b8ddeae3ab8", "v2 CUDA source digest")

    # The ten explicit contraction sites must be the complete expected set.
    expected_fma_lhs = {
        "sse": 5,
        "baseline": 2,
        "local_sum_x2[category]": 1,
        "sum_xw0[category]": 1,
        "sum_xw1[category]": 1,
    }
    observed_fma_lhs = {key: 0 for key in expected_fma_lhs}
    statements = [piece.strip() for piece in explicit_source.replace("\r", "").split(";")]
    fma_statements = [statement for statement in statements if "__fma_rn(" in statement]
    check(len(fma_statements) == 10, f"explicit FMA site count: {len(fma_statements)}")
    for statement in fma_statements:
        matches = re.findall(r"(?:double\s+)?([A-Za-z_]\w*(?:\[[^\]]+\])?)\s*=\s*__fma_rn\(", statement)
        check(len(matches) == 1, f"cannot isolate explicit FMA lhs: {statement!r}")
        lhs = matches[0]
        check(lhs in observed_fma_lhs, f"unexpected explicit FMA lhs: {lhs!r}")
        observed_fma_lhs[lhs] += 1
    check(observed_fma_lhs == expected_fma_lhs, f"explicit FMA inventory mismatch: {observed_fma_lhs}")
    check(explicit_source.count("__fma_rn(") == 10, "explicit FMA token count")
    check(sum(explicit_source.count(token) for token in ("__dadd_rn(", "__dsub_rn(", "__dmul_rn(", "__ddiv_rn(")) == 19, "rounding intrinsic token count")
    for forbidden in (
        "local_sum_x2[category] += x * x",
        "sum_xw0[category] += x *",
        "sum_xw1[category] += x *",
        "sum_x2[fit_cat] - sum_x[fit_cat] * sum_x[fit_cat]",
        "sum_xw[fit_cat] - sum_x[fit_cat] * sw_fit",
        "mean_w - alpha_raw * sum_x[fit_cat]",
        "sw2_score + (double)score_n * mu * mu",
        "sw2_score - 2.0 * mean_w * sw_score",
        "1.0 - (sse / baseline)",
    ):
        check(forbidden not in explicit_source, f"implicit FP64 contraction/reassociation site survived: {forbidden}")
    # All three multiply-accumulate classes in the FP64 moment loop must either
    # be explicit FMA or a multiplication-free shuffle reduction.  The metric's
    # centered/affine/SSE/baseline/capture expressions were checked above by
    # their exact pre-rewrite strings, so an ordinary eligible FP64 a*b+c site
    # cannot survive those cardinality-one rewrites.
    check(not re.search(r"local_sum_x2\[category\]\s*\+=\s*[^;]*\*", explicit_source), "implicit x2 accumulation")
    check(not re.search(r"sum_xw[01]\[category\]\s*\+=\s*[^;]*\*", explicit_source), "implicit cross-moment accumulation")
    check(explicit_source.count("local_sum_x2[category] += __shfl_down_sync") == 1, "x2 shuffle reduction is addition-only")
    check(explicit_source.count("local_sum_x[category] += __shfl_down_sync") == 1, "x shuffle reduction is addition-only")
    return {
        "v1_cuda_sha256": sha256_bytes(binary64_source.encode()),
        "v2_cuda_sha256": sha256_bytes(explicit_source.encode()),
        "explicit_fma_sites": observed_fma_lhs,
        "rounding_intrinsic_occurrences": 19,
    }


def plan_audit(repo: Path, v1_result: dict, v2_result: dict) -> dict:
    plan_path = repo / "research/fuseed_pmg1_direct_source_calibration_v0/plan.py"
    reconstruct = isolated_function(plan_path, "reconstruct_plan", {
        "hashlib": hashlib,
        "json": json,
        **{
            name: literal_assignment(plan_path, name)
            for name in (
                "T", "ROWS", "COLUMNS", "ABI_IDS", "UP_NUMELS", "DOWN_NUMELS",
                "EXPECTED_V1_STAGE0_SHA256", "EXPECTED_V1_ALL_PLAN_SHA256",
                "EXPECTED_V1_BUNDLE_ATTEMPTS", "EXPECTED_ABI1_STAGE0_SUBSET_SHA256",
                "EXPECTED_ABI1_CATEGORY_BUNDLE_SHA256",
            )
        },
    })
    # reconstruct_plan calls sibling helpers; execute all pure function definitions,
    # while executing no imports or module main.
    tree = ast.parse(plan_path.read_text(encoding="utf-8"), filename=str(plan_path))
    function_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + function_nodes,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    env = {"hashlib": hashlib, "json": json}
    for name in (
        "T", "ROWS", "COLUMNS", "ABI_IDS", "UP_NUMELS", "DOWN_NUMELS",
        "EXPECTED_V1_STAGE0_SHA256", "EXPECTED_V1_ALL_PLAN_SHA256",
        "EXPECTED_V1_BUNDLE_ATTEMPTS", "EXPECTED_ABI1_STAGE0_SUBSET_SHA256",
        "EXPECTED_ABI1_CATEGORY_BUNDLE_SHA256",
    ):
        env[name] = literal_assignment(plan_path, name)
    exec(compile(module, str(plan_path), "exec"), env)
    result = env["reconstruct_plan"]()
    # Independently expose the per-ABI attempt counter which the inherited plan
    # reports only in aggregate.  Cross-ABI opposite-split collision history is
    # intentionally retained between ABI loops.
    identities = env["selection_identities"]()
    high_count = env["high_count_identities"](identities)
    global_by_split = {"fit": set(), "score": set()}
    attempts_by_abi = []
    for abi, abi_id in enumerate(env["ABI_IDS"]):
        used_by_identity = {}
        abi_attempts = 0
        for expert, role in identities:
            identity = (expert, role)
            used_by_identity[identity] = set()
            coordinate_count = 24 if identity in high_count else 20
            for split in ("fit", "score"):
                accepted = 0
                counter = 0
                numel = env["UP_NUMELS"][abi] if role == "up" else env["DOWN_NUMELS"][abi]
                valid_bundle_count = (numel - 1 - 3 * env["T"]) // (4 * env["T"]) + 1
                while accepted < coordinate_count // 4:
                    digest_bytes = env["hash_text"](f"FUSEED-U32-v1|{abi_id}|{expert}|{role}|{split}|{counter}")
                    sequence = env["u32_le"](digest_bytes, 0) % env["T"]
                    normal4_index = env["u32_le"](digest_bytes, 4) % valid_bundle_count
                    local_keys = set()
                    valid = True
                    for lane in range(4):
                        native = sequence + env["T"] * (4 * normal4_index + lane)
                        inverted = env["invert_native"](abi, expert, role, native)
                        if inverted is None:
                            valid = False
                            break
                        row, column = inverted
                        key = env["canonical_key"](expert, role, row, column)
                        if key in local_keys or key in used_by_identity[identity] or key in global_by_split["score" if split == "fit" else "fit"]:
                            valid = False
                            break
                        local_keys.add(key)
                    counter += 1
                    abi_attempts += 1
                    if not valid:
                        continue
                    used_by_identity[identity].update(local_keys)
                    global_by_split[split].update(local_keys)
                    accepted += 1
        attempts_by_abi.append(abi_attempts)
    counters = {}
    lines = []
    for index, bundle in enumerate(result["wire"]):
        key = (bundle["expert"], bundle["role"], bundle["split"])
        accepted = counters.get(key, 0)
        counters[key] = accepted + 1
        category = f"{bundle['role']}_{bundle['split']}"
        scale = "3c03126f" if bundle["role"] == "up" else "3a560a28"
        line = (
            f"bundle|index={index:03d}|category={category}|e={bundle['expert']:03d}|"
            f"role={bundle['role']}|split={bundle['split']}|accepted={accepted:02d}|"
            f"seed_delta={bundle['seed_addend']}|offset={bundle['offset_values']}|"
            f"scale={scale}|seq={bundle['sequence']}|j={bundle['normal4_index']}"
        )
        for coordinate in bundle["coordinates"]:
            line += (
                f"|lane{coordinate['lane']}=r{coordinate['row']:03d},"
                f"c{coordinate['column']:04d},native{coordinate['native']}"
            )
        lines.append(line)
    digest = sha256_bytes(("\n".join(lines) + "\n").encode())
    check(digest == ABI1_BUNDLE_SHA, f"ABI1 bundle digest: {digest}")
    check(len(result["wire"]) == 256, "ABI1 bundle count")
    check(sum(len(bundle["coordinates"]) for bundle in result["wire"]) == 1024, "ABI1 coordinate count")
    check(result["facts"]["v1_bundle_attempts"] == sum(attempts_by_abi) == 17151, "all-ABI attempt count")
    check(attempts_by_abi[1] == 531, f"ABI1 attempt count: {attempts_by_abi}")
    identities2, global_by_split2, _, _, _, _ = env["enumerate_stage0"]()
    stage1_fit = set(global_by_split2["fit"])
    stage1_score = set(global_by_split2["score"])
    env["fill_plan"]("stage1", "fit", stage1_fit, stage1_score, identities2, 2048)
    env["fill_plan"]("stage1", "score", stage1_score, stage1_fit, identities2, 2048)
    full_fit = set(stage1_fit)
    full_score = set(stage1_score)
    env["fill_plan"]("stage2", "fit", full_fit, full_score, identities2, 24312)
    env["fill_plan"]("stage2", "score", full_score, full_fit, identities2, 24312)
    validation_ids = env["validation_identities"]()
    validation_fit = set()
    validation_score = set()
    env["fill_plan"]("validation", "fit", validation_fit, validation_score, validation_ids, 8456)
    env["fill_plan"]("validation", "score", validation_score, validation_fit, validation_ids, 8456)
    set_specs = (
        ("stage1", "fit", stage1_fit, "568990c4fc1e9fb2c259a7d1a70a7aea9a062ddbaadb778392c6eea12f0aaadd"),
        ("stage1", "score", stage1_score, "41711a4ff0d0198b21b4732cb2716aee68d8a8b4d87e51a601a54bad5faa090b"),
        ("stage2", "fit", full_fit, "21181a92fb1252865475193cb7d34bd578f21cd82df027026d05520bffac1f69"),
        ("stage2", "score", full_score, "d5b646fd653c173b18e9cfce65b62833f6122f3a90a2c0f0fe7e2e6d5bd5ff42"),
        ("validation", "fit", validation_fit, "35061888d4458fd0e1b0db4f457ebc4197f55c7fcbaead17279f0e89cb0dde37"),
        ("validation", "score", validation_score, "37ad183ce83f4f964bb246ff267e528951457c4bac436b2639d386645d9969c0"),
    )
    for namespace, split, values, expected in set_specs:
        payload = "".join(f"{namespace}|{split}|{key}\n" for key in sorted(values))
        check(sha256_bytes(payload.encode()) == expected, f"{namespace}/{split} set digest")
    complete_lines = ["family|FUSEED-PMG1-v2", "abi|CURRENT_PMG_GATE_UP_DIRECT_BF16", *lines]
    for namespace, split, values, _ in set_specs:
        complete_lines.extend(f"{namespace}|{split}|{key}" for key in sorted(values))
    complete_digest = sha256_bytes(("\n".join(complete_lines) + "\n").encode())
    check(complete_digest == COMPLETE_PLAN_SHA, f"complete PMG1 plan digest: {complete_digest}")
    for producer in (v1_result, v2_result):
        check(producer["bindings"]["design_bundle_sha256"] == digest, "producer ABI1 bundle binding")
        check(producer["bindings"]["design_complete_plan_sha256"] == COMPLETE_PLAN_SHA, "producer complete-plan binding")
    return {"bundle_sha256": digest, "complete_plan_sha256": complete_digest, "bundles": 256, "coordinates": 1024, "attempts": 531, "attempts_by_abi": attempts_by_abi}


def wrapper_audit(repo: Path, v1_result: dict, v2_result: dict) -> dict:
    v1_path = repo / "research/fuseed_pmg1_binary64_calibration_v1/calibrate_binary64.py"
    v2_path = repo / "research/fuseed_pmg1_binary64_fma_calibration_v2/calibrate_fma.py"
    check(sha256_file(v1_path) == V1_SCRIPT_SHA, "wrapper template file binding")
    source = v1_path.read_text(encoding="utf-8")
    replacements = (
        ('"fuseed_pmg1_binary64_source_free_stage0_calibration_v1"', '"fuseed_pmg1_binary64_explicit_fma_stage0_calibration_v2"'),
        ('"BINARY64_STAGE0_MARGIN_PASS_PENDING_FULL_PIPELINE_AND_INDEPENDENT_AUDIT"', '"EXPLICIT_FMA_BINARY64_STAGE0_MARGIN_PASS_PENDING_FULL_PIPELINE_AND_INDEPENDENT_AUDIT"'),
        ('"EARLY_KILL_BINARY64_STAGE0_NO_QWEN"', '"EARLY_KILL_EXPLICIT_FMA_BINARY64_STAGE0_NO_QWEN"'),
        ('"fuseed_pmg1_binary64_shard_journal_v1"', '"fuseed_pmg1_binary64_explicit_fma_shard_journal_v2"'),
        ('"fuseed_pmg1_binary64_stage0.cu"', '"fuseed_pmg1_binary64_explicit_fma_stage0.cu"'),
        ('"fuseed_pmg1_binary64_parity.cu"', '"fuseed_pmg1_binary64_explicit_fma_parity.cu"'),
    )
    counts = {}
    for old, new in replacements:
        counts[old.strip('"')] = source.count(old)
        check(source.count(old) == 1, f"wrapper label cardinality: {old}")
        source = source.replace(old, new)
    check(v2_result["schema"] == "fuseed_pmg1_binary64_explicit_fma_stage0_calibration_v2", "derived result schema")
    check(v1_result["schema"] == "fuseed_pmg1_binary64_source_free_stage0_calibration_v1", "template result schema")
    return {
        "template_sha256": V1_SCRIPT_SHA,
        "derived_python_sha256": sha256_bytes(source.encode()),
        "label_replacement_counts": counts,
    }


def parse_journal(path: Path, expected_sha: str, expected_size: int, result: dict, row: dict, version: str):
    regular_nonsymlink(path)
    data = path.read_bytes()
    check(len(data) == expected_size == row["journal_bytes"], f"journal size: {path.name}")
    check(sha256_bytes(data) == expected_sha == row["journal_sha256"], f"journal digest: {path.name}")
    check(len(data) >= 4, "journal header length present")
    header_len = struct.unpack_from("<I", data, 0)[0]
    check(4 + header_len + TOP_K * 12 == len(data), f"journal exact framing: {path.name}")
    header = strict_json_bytes(data[4:4 + header_len], f"{path.name}:header")
    expected_schema = (
        "fuseed_pmg1_binary64_shard_journal_v1"
        if version == "v1" else "fuseed_pmg1_binary64_explicit_fma_shard_journal_v2"
    )
    check(set(header) == {
        "candidate_count", "capture_sha256", "design_bundle_sha256",
        "design_complete_plan_sha256", "metric_order", "performance_cubin_sha256",
        "record_wire", "repetition", "schema", "seed_sha256", "shard_base_u32", "top_k",
    }, f"journal exact header keys: {path.name}")
    check(header["schema"] == expected_schema, "journal schema")
    check(header["repetition"] == row["repetition"], "journal repetition")
    check(header["shard_base_u32"] == 0 and header["candidate_count"] == SHARD_SIZE, "journal shard shape")
    check(header["top_k"] == TOP_K, "journal K")
    check(header["record_wire"] == "packed little-endian u32 seed then binary64 capture", "journal wire")
    check(header["metric_order"] == "capture descending then seed_u32 ascending", "journal order label")
    check(header["design_bundle_sha256"] == ABI1_BUNDLE_SHA, "journal ABI1 binding")
    check(header["design_complete_plan_sha256"] == COMPLETE_PLAN_SHA, "journal plan binding")
    check(header["performance_cubin_sha256"] == result["compiled_kernels"]["performance"]["cubin_sha256"], "journal cubin binding")
    records = data[4 + header_len:]
    seeds = []
    captures = []
    for index in range(TOP_K):
        seed, capture = struct.unpack_from("<Id", records, index * 12)
        check(math.isfinite(capture), f"nonfinite journal capture {index}")
        check(not (capture == 0.0 and math.copysign(1.0, capture) < 0), f"negative zero {index}")
        seeds.append(seed)
        captures.append(capture)
        if index:
            check(
                captures[index - 1] > capture
                or (captures[index - 1] == capture and seeds[index - 1] < seed),
                f"journal total order violation {index}",
            )
    check(len(set(seeds)) == TOP_K, "journal unique seeds")
    seed_bytes = b"".join(struct.pack("<I", seed) for seed in seeds)
    capture_bytes = b"".join(struct.pack("<d", capture) for capture in captures)
    seed_sha = sha256_bytes(seed_bytes)
    capture_sha = sha256_bytes(capture_bytes)
    check(seed_sha == header["seed_sha256"] == row["topk_seed_sha256"], "journal seed vector digest")
    check(capture_sha == header["capture_sha256"] == row["topk_capture_sha256"], "journal capture vector digest")
    check(seeds[0] == row["best_seed_u32"] and captures[0] == row["best_capture"], "journal best row")
    check(captures[-1] == row["threshold_capture"], "journal threshold row")
    check(row["boundary_tie_cardinality"] == sum(c == captures[-1] for c in captures), "journal boundary tie cardinality")
    check(row["packed_record_bytes"] == TOP_K * 12, "packed record byte ledger")
    return header, records, seeds, captures


def global_merge_probe(seeds: list[int], captures: list[float]):
    heap = []
    for shard in range(SHARDS):
        heapq.heappush(heap, (-captures[0], seeds[0] + shard * SHARD_SIZE, shard, 0))
    out_seeds = []
    out_captures = []
    for _ in range(TOP_K):
        negative, seed, shard, index = heapq.heappop(heap)
        out_seeds.append(seed)
        out_captures.append(-negative)
        index += 1
        if index < TOP_K:
            heapq.heappush(heap, (-captures[index], seeds[index] + shard * SHARD_SIZE, shard, index))
    seed_bytes = b"".join(struct.pack("<I", seed) for seed in out_seeds)
    capture_bytes = b"".join(struct.pack("<d", capture) for capture in out_captures)
    return sha256_bytes(seed_bytes), sha256_bytes(capture_bytes)


def audit_parity(result: dict, version: str) -> dict:
    parity = result["parity"]
    direct = parity["direct_shifted_reference_three_replays"]
    sequential = parity["direct_shifted_and_original_offset_sequential_three_replays"]
    torch_state = parity["torch_initial_terminal_and_bf16_three_replays"]
    check(direct["repetitions"] == 3 and direct["identical"] is True, f"{version} direct 3x receipt")
    dr = direct["receipt"]
    check(dr["rows"] == 132 and dr["raw_bitwise_equal"] is True and dr["scaled_bf16_bitwise_equal"] is True, "direct parity rows/values")
    check(dr["terminal_counter_equal"] is True, "direct parity terminal counter")
    check(dr["max_normal4_indices_by_call_size"] == [95, 47, 2, 0], "direct parity max-j")
    check(dr["zero_kat_words"] == ["6627e8d5", "e169c58d", "bc57ac4c", "9b00dbd8"], "direct KAT")
    check(sequential["repetitions"] == 3 and sequential["identical"] is True, "sequential 3x receipt")
    sr = sequential["receipt"]
    check(sr["rows"] == 132, "sequential row count")
    check(sr["direct_equals_shifted"] is True and sr["direct_equals_original_offset_sequential_j_plus_1"] is True, "sequential dual reference")
    check(sr["terminal_counters_equal"] is True, "sequential terminal state")
    check(sr["scaled_bf16_sha256"] == dr["scaled_widened_bf16_sha256"], "direct/sequential role-scaled BF16 cross-receipt parity")
    check(torch_state["repetitions"] == 3 and torch_state["identical"] is True, "Torch-state 3x receipt")
    check(torch_state["case_count"] == 8 and torch_state["coordinate_count"] == 56 and len(torch_state["rows"]) == 8, "Torch-state shape")
    check(torch_state["stride"] == 261120, "Torch-state stride")
    for case in torch_state["rows"]:
        check(case["effective_seed"] == case["base_seed"] + 1024 + 100 * (case["expert"] // 32), "expert seed addend")
        expected_increment = ((case["numel"] - 1) // (261120 * 4) + 1) * 4
        check(case["expected_increment"] == expected_increment, "Torch expected increment")
        check(case["terminal_offset"] == case["initial_offset"] + expected_increment, "Torch terminal offset")
        check(case["initial_seed"] == case["effective_seed"] and case["offset"] == case["initial_offset"], "Torch initial state")
        for field in ("initial_state_sha256", "terminal_state_sha256", "bf16_widened_sha256"):
            check(len(case[field]) == 64, f"Torch state digest {field}")
    return {
        "direct_rows": 132,
        "sequential_rows": 132,
        "torch_cases": 8,
        "torch_coordinates": 56,
        "sequential_raw_sha256": sr["raw_float32_sha256"],
        "scaled_bf16_sha256": sr["scaled_bf16_sha256"],
    }


def audit_runtime(repo: Path, v1_result: dict, v2_result: dict) -> dict:
    base = repo / "research/fuseed_pmg1_direct_source_calibration_v0/calibrate.py"
    direct = repo / "research/fuseed_u32_direct_counter_calibration_v0/calibrate_direct.py"
    expected_runtime = literal_assignment(base, "EXPECTED_RUNTIME_FILES")
    expected_headers = literal_assignment(direct, "CUDA_HEADERS")
    observed_runtime = {}
    for raw, digest in expected_runtime.items():
        path = Path(raw)
        regular_nonsymlink(path)
        actual = sha256_file(path)
        check(actual == digest, f"runtime file hash mismatch: {raw}")
        observed_runtime[raw] = actual
    observed_headers = {}
    for raw, digest in expected_headers.items():
        path = Path(raw)
        regular_nonsymlink(path)
        actual = sha256_file(path)
        check(actual == digest, f"CUDA header hash mismatch: {raw}")
        observed_headers[raw] = actual
    for producer in (v1_result, v2_result):
        check(set(producer["runtime"]["file_bindings"]) == set(expected_runtime), "runtime exact member set")
        for raw, digest in expected_runtime.items():
            check(producer["runtime"]["file_bindings"][raw]["sha256"] == digest, "result runtime hash")
            check(producer["runtime"]["file_bindings"][raw]["resolved"] == raw, "result runtime resolved path")
        check(producer["bindings"]["cuda_headers"] == expected_headers, "result header closure")
        check(producer["runtime"]["python"] == "3.12.3", "Python version")
        check(producer["runtime"]["numpy"] == "2.5.2", "NumPy version")
        check(producer["runtime"]["cupy"] == "14.2.0", "CuPy version")
        check(producer["runtime"]["torch"] == "2.8.0+cu128", "Torch version")
        check(producer["runtime"]["torch_cuda"] == "12.8", "Torch CUDA version")
        check(producer["runtime"]["cuda_runtime"] == 12090 and producer["runtime"]["cuda_driver_api"] == 13000, "CUDA versions")
        check(producer["runtime"]["nvrtc"] == [12, 8], "NVRTC version")
        check(producer["runtime"]["device"] == "NVIDIA GeForce RTX 5090" and producer["runtime"]["compute_capability"] == "120", "device binding")
        check(producer["runtime"]["loaded_cuda_libraries"] == [
            "/usr/lib/x86_64-linux-gnu/libcuda.so.580.126.09",
            "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
            "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12",
        ], "loaded CUDA library closure")
        for kernel in ("performance", "parity"):
            receipt = producer["compiled_kernels"][kernel]
            check(receipt["arch"] == "120", "cubin arch")
            check(receipt["cubin_magic_hex"] == "7f454c4602010141", "cubin ELF magic")
            check(receipt["cubin_bytes"] > 0 and len(receipt["cubin_sha256"]) == 64, "cubin receipt")
    v1_options = ["--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", "-I/usr/local/cuda/include"]
    v2_options = ["--std=c++17", "--fmad=true", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", "-I/usr/local/cuda/include"]
    for kernel in ("performance", "parity"):
        check(v1_result["compiled_kernels"][kernel]["options"] == v1_options, "v1 compile options")
        check(v2_result["compiled_kernels"][kernel]["options"] == v2_options, "v2 compile options")
    check(v1_result["compiled_kernels"]["parity_source_sha256"] == v2_result["compiled_kernels"]["parity_source_sha256"] == "63cc615c7f9b0a5f07920cc6c9b04516160e3784d3dabf5427f170efe3bd46b1", "parity source binding")
    return {"runtime_files_rehashed": len(observed_runtime), "headers_rehashed": len(observed_headers), "cubin_receipts_checked": 4}


def audit_one(repo: Path, version: str, result: dict, journal_directory: Path) -> tuple[dict, list[int], list[float], bytes]:
    expected_status = (
        "EARLY_KILL_BINARY64_STAGE0_NO_QWEN" if version == "v1"
        else "EXPLICIT_FMA_BINARY64_STAGE0_MARGIN_PASS_PENDING_FULL_PIPELINE_AND_INDEPENDENT_AUDIT"
    )
    check(result["status"] == expected_status, f"{version} status")
    check(result["claim_boundary"].startswith("Source-free binary64 stage0 calibration only."), f"{version} claim boundary")
    shape = result["shape"]
    check(shape == {
        "abi_count": 1, "active_domains": 1, "all_shard_topk_record_bytes": 25165824,
        "block_threads": 256, "candidates_per_shard": SHARD_SIZE,
        "complete_candidate_count": 1 << 32, "grid_blocks": 65535,
        "journal_record_bytes": 12, "normal4_bundles_per_candidate": 256,
        "normal_values_per_candidate": 1024, "q_bytes": 134217728,
        "q_dtype": "binary64", "repetitions": 3, "shards": SHARDS,
        "top_k": TOP_K, "warps_per_block": 8,
    }, f"{version} exact shape ledger")
    check(len(result["rows"]) == 3, f"{version} repetitions")
    parsed = []
    for (filename, digest, size), row in zip(EXPECTED_JOURNALS[version], result["rows"], strict=True):
        parsed.append(parse_journal(journal_directory / filename, digest, size, result, row, version))
    record_payloads = [item[1] for item in parsed]
    check(record_payloads[0] == record_payloads[1] == record_payloads[2], f"{version} exact journal replay")
    seeds, captures = parsed[0][2], parsed[0][3]
    global_seed_sha, global_capture_sha = global_merge_probe(seeds, captures)
    probe = result["global_merge_shape_probe"]
    check(probe["input_records"] == SHARDS * TOP_K and probe["output_records"] == TOP_K, "global merge shape")
    check(probe["uses_synthetic_repeated_shard_metrics"] is True, "global merge synthetic scope")
    check(probe["seed_sha256"] == global_seed_sha and probe["capture_sha256"] == global_capture_sha, "global merge values")
    row_times = [row["shard_end_to_end_seconds"] for row in result["rows"]]
    for row in result["rows"]:
        computed = row["kernel_seconds"] + row["finite_and_zero_validation_seconds"] + row["topk_seconds"] + row["journal_fsync_seconds"]
        check(computed == row["shard_end_to_end_seconds"], "row wall arithmetic")
        check(row["negative_zero_count"] == 0, "negative zero receipt")
    median = statistics.median(row_times)
    cold = max(0.0, row_times[0] - median)
    projection = median * SHARDS + cold + probe["seconds"]
    aggregate = result["aggregate"]
    check(aggregate["median_complete_stage0_shard_seconds"] == median, "median shard time")
    check(aggregate["one_time_cold_excess_seconds"] == cold, "cold excess")
    check(aggregate["projected_complete_u32_stage0_seconds_including_finite_topk_journal_and_global_merge"] == projection, "projection arithmetic")
    check(aggregate["prospective_stage0_margin_gate_seconds"] == 650.0, "stage0 gate")
    check(aggregate["reserved_seconds_for_unmeasured_stage1_stage2_validation_and_final_journal"] == 250.0, "unmeasured reserve")
    check(aggregate["full_pipeline_projection_claimed"] is False, "no full projection claim")
    below = projection < 650.0
    check(aggregate["stage0_projection_below_margin_gate"] is below, "stage0 decision")
    check((version == "v1" and not below) or (version == "v2" and below), "version decision boundary")
    access = result["access"]
    check(access == {"model_or_qwen_path_arguments": 0, "network_operations": 0, "payload_files_opened": 0}, "source-free access receipt")
    return ({
        "projection_seconds": projection,
        "below_650": below,
        "best_seed_u32": seeds[0],
        "best_capture": captures[0],
        "topk_seed_sha256": result["rows"][0]["topk_seed_sha256"],
        "topk_capture_sha256": result["rows"][0]["topk_capture_sha256"],
        "global_seed_sha256": global_seed_sha,
        "global_capture_sha256": global_capture_sha,
    }, seeds, captures, record_payloads[0])


def main() -> None:
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    research = repo / "research"
    design_dir = research / "fuseed_pmg1_v2_design_draft"
    v1_dir = research / "fuseed_pmg1_binary64_calibration_v1"
    v2_dir = research / "fuseed_pmg1_binary64_fma_calibration_v2"
    v1_result_path = v1_dir / "result.json"
    v2_result_path = v2_dir / "result.json"
    v1_journal_dir = v1_dir
    v2_journal_dir = v2_dir
    # The RunPod source tree intentionally retains only producer source.  An
    # auditor may place byte-identical, hash-addressed result snapshots in this
    # exact temporary input root; they are never treated as audit members.
    snapshot_root = Path("/tmp/fuseed_pmg1_binary64_fma_v2_audit_inputs")
    if not v1_result_path.exists():
        v1_result_path = snapshot_root / "v1/result.json"
        v1_journal_dir = snapshot_root / "v1"
    if not v2_result_path.exists():
        v2_result_path = snapshot_root / "v2/result.json"
        v2_journal_dir = snapshot_root / "v2"
    exact_inputs = {
        design_dir / "ARTIFACT_SHA256SUMS.txt": DESIGN_MANIFEST_SHA,
        v1_dir / "calibrate_binary64.py": V1_SCRIPT_SHA,
        v1_result_path: V1_RESULT_SHA,
        v2_dir / "calibrate_fma.py": V2_SCRIPT_SHA,
        v2_result_path: V2_RESULT_SHA,
    }
    for path, digest in exact_inputs.items():
        regular_nonsymlink(path)
        check(sha256_file(path) == digest, f"input digest: {path}")
    manifest_path = design_dir / "ARTIFACT_SHA256SUMS.txt"
    rows = manifest_map(manifest_path)
    check(rows == EXPECTED_DESIGN_MEMBERS, "design manifest exact closure")
    for member, digest in rows.items():
        path = design_dir / member
        regular_nonsymlink(path)
        check(sha256_file(path) == digest, f"design member digest: {member}")
    design = read_json(design_dir / "design_lock.json")
    design_receipt = read_json(design_dir / "DESIGN_RECEIPT.json")
    check(design["sealed"] is False, "design remains an unsealed draft")
    check(design["authorization"] == {
        "implementation_or_execution_authorized": False,
        "payload_access_authorized": False,
        "production_result_claimed": False,
        "required_next_action": "independent source-only audit, chronology/firewall attestation, then a separately authorized hardened source-free calibration; this draft itself grants none of those actions",
        "runtime_calibration_authorized": False,
    }, "design authorization is exactly NONE")
    check(design_receipt["scientific_verdict"]["chronology_and_validation_independence_authenticated"] is False, "chronology explicitly unauthenticated")
    check(design_receipt["authorization"] == "NONE", "design receipt authorization")
    # Control/validation semantics are coherent but not executed by either stage-0 calibration.
    claim = design["scientific_claim"]
    check(claim["control_searches_seed_family"] is False, "controls do not search seeds")
    check(claim["randomization_or_familywise_p_value_claimed"] is False, "no invalid control p-value")
    check(claim["exactly_one_validation_descriptor"] is True, "one validation descriptor")
    check(claim["control_failure_permits_retry"] is False and claim["validation_failure_permits_retry"] is False, "terminal control/validation failure")
    check(design["post_selection_controls"]["control_seed_search_count"] == 0, "control seed search count")
    check(design["cascade"]["maximum_generated_normal_values_total"] == 4398092530192, "complete cascade arithmetic")
    check(design["cascade"]["stage1"]["generated_normal_values_max"] == 33554432, "stage1 arithmetic")
    check(design["cascade"]["stage2"]["generated_normal_values_max"] == 12447744, "stage2 arithmetic")
    check(design["cascade"]["validation"]["generated_normal_values"] == 16912, "validation arithmetic")

    # The proposed validation identities and values are not untouched.  Two
    # earlier, published result receipts contain fit/score moments for exactly
    # the same four experts.  These are metadata receipts, not model files.
    prior_v1_path = research / "initialization_anchor_oracle/result.json"
    prior_tier_b_path = research / "initialization_anchor_oracle_tier_b/result.json"
    if not prior_v1_path.exists():
        prior_v1_path = snapshot_root / "prior/initialization_anchor_result.json"
    if not prior_tier_b_path.exists():
        prior_tier_b_path = snapshot_root / "prior/tier_b_result.json"
    for path, digest in ((prior_v1_path, PRIOR_V1_RESULT_SHA), (prior_tier_b_path, PRIOR_TIER_B_RESULT_SHA)):
        regular_nonsymlink(path)
        check(sha256_file(path) == digest, f"prior result receipt digest: {path.name}")
    prior_v1 = read_json(prior_v1_path)
    prior_tier_b = read_json(prior_tier_b_path)
    expected_validation_experts = [24, 56, 88, 120]
    check(design["coordinate_protocol"]["validation_experts"] == expected_validation_experts, "PMG validation expert identities")
    for label, prior in (("prior_v1", prior_v1), ("prior_tier_b", prior_tier_b)):
        source_rows = prior["validation"]["details"]["source"]
        check(len(source_rows) == 8, f"{label} prior source validation matrix count")
        observed = sorted({int(row["expert"]) for row in source_rows})
        check(observed == expected_validation_experts, f"{label} prior validation experts")
        check({row["role"] for row in source_rows} == {"up", "down"}, f"{label} prior validation roles")
        for row in source_rows:
            # Nonempty target-derived sufficient statistics prove that these
            # validation values were opened and scored, not merely named.
            for split in ("fit", "score"):
                stats = row[split]
                check(stats["n"] > 0, f"{label} opened validation count")
                check(all(math.isfinite(float(stats[key])) for key in ("sum_w", "sum_w2")), f"{label} finite opened validation moments")

    v1_result = read_json(v1_result_path)
    v2_result = read_json(v2_result_path)
    check(v1_result["script_sha256"] == V1_SCRIPT_SHA, "v1 self binding")
    check(v2_result["script_sha256"] == V2_SCRIPT_SHA, "v2 self binding")
    # Wrapper/template chain and exact label cardinalities are independently checked.
    v2_source = (v2_dir / "calibrate_fma.py").read_text(encoding="utf-8")
    check(literal_assignment(v2_dir / "calibrate_fma.py", "EXPECTED_TEMPLATE_SHA256") == V1_SCRIPT_SHA, "v2 template binding")
    check(v2_source.count("EXPECTED_TEMPLATE_SHA256") == 2, "template binding use cardinality")
    check("exec(compile(source, str(template_path), \"exec\"), namespace)" in v2_source, "wrapper executes derived template bytes")
    source_facts = reconstruct_sources(repo, v1_result, v2_result)
    plan_facts = plan_audit(repo, v1_result, v2_result)
    wrapper_facts = wrapper_audit(repo, v1_result, v2_result)
    runtime_facts = audit_runtime(repo, v1_result, v2_result)
    v1_parity = audit_parity(v1_result, "v1")
    v2_parity = audit_parity(v2_result, "v2")
    v1_facts, v1_seeds, v1_captures, _ = audit_one(repo, "v1", v1_result, v1_journal_dir)
    v2_facts, v2_seeds, v2_captures, _ = audit_one(repo, "v2", v2_result, v2_journal_dir)
    overlap = set(v1_seeds).intersection(v2_seeds)
    common_changed = sum(
        struct.pack("<d", v1_captures[v1_seeds.index(seed)])
        != struct.pack("<d", v2_captures[v2_seeds.index(seed)])
        for seed in overlap
    )
    check(v1_facts["projection_seconds"] == 696.8922319519334 and v1_facts["projection_seconds"] > 650.0, "v1 696.892 kill")
    check(v2_facts["projection_seconds"] == 520.8358833260136 and v2_facts["projection_seconds"] < 650.0, "v2 520.835 pass")
    check(v1_facts["topk_seed_sha256"] != v2_facts["topk_seed_sha256"], "FMA changes TopK seed bytes")
    check(v1_facts["topk_capture_sha256"] != v2_facts["topk_capture_sha256"], "FMA changes TopK capture bytes")
    check(common_changed > 0, "explicit FMA is result-changing on common retained candidates")
    check(v1_facts["best_seed_u32"] == v2_facts["best_seed_u32"] == 10309563, "best seed remains stable")
    check(v1_facts["best_capture"] == v2_facts["best_capture"] == 0.03529210911201497, "best capture remains stable")
    # The raw sequential normal receipt differs solely with the compile-policy
    # successor, while the role-scaled BF16 ABI and Torch-state receipts agree.
    check(v1_parity["sequential_raw_sha256"] != v2_parity["sequential_raw_sha256"], "compile policy changes raw parity receipt")
    check(v1_parity["scaled_bf16_sha256"] == v2_parity["scaled_bf16_sha256"], "scaled BF16 ABI parity remains stable")
    check(v1_result["parity"]["torch_initial_terminal_and_bf16_three_replays"] == v2_result["parity"]["torch_initial_terminal_and_bf16_three_replays"], "Torch-state receipt remains stable")

    # Source-only audit verdict: the v2 stage-0 survivor is authentic, but the
    # design itself requires prerequisites not present in this input set.
    missing = [
        "a new genuinely untouched validation panel: the frozen [24,56,88,120] panel was already opened and scored in two prior published branches",
        "independent prospective chronology and untouched-validation firewall attestation",
        "256-cell source-free retention PASS",
        "complete stage1/stage2/one-descriptor-validation timing and semantics",
        "full future compiler/runtime closure including OS/kernel, raw argv/environment, compiled-intermediate and serialized loaded-kernel bytes",
        "crash-safe 256-shard journal/resume audit and two-tree byte-identical global merge",
        "durable no-retry selection commit before validation visibility",
    ]
    receipt = {
        "schema": AUDIT_SCHEMA,
        "verdict": "BLOCK_PAYLOAD_AUTHORIZATION_STAGE0_V2_SURVIVOR_AUTHENTICATED",
        "authorization": "NONE",
        "checks": checks + 1,
        "inputs": {
            "design_manifest_sha256": DESIGN_MANIFEST_SHA,
            "v1_script_sha256": V1_SCRIPT_SHA,
            "v1_result_sha256": V1_RESULT_SHA,
            "v2_script_sha256": V2_SCRIPT_SHA,
            "v2_result_sha256": V2_RESULT_SHA,
        },
        "authenticated": {
            "hash_dependency_graph": "design manifest -> v1 script and v1 result; v1 script -> v2 wrapper; v2 result self-binds wrapper. No artifact authenticates wall-clock execution order or prospective choice.",
            "result_chronology_authenticated": False,
            "scientific_prospective_chronology": False,
            "untouched_validation_claim": False,
            "prior_open_validation_receipts": {
                "initialization_anchor_result_sha256": PRIOR_V1_RESULT_SHA,
                "tier_b_result_sha256": PRIOR_TIER_B_RESULT_SHA,
                "experts": [24, 56, 88, 120],
                "matrix_rows_per_receipt": 8,
            },
            "design_source_semantics": True,
            "abi1_plan": plan_facts,
            "wrapper_template_derivation": wrapper_facts,
            "derived_sources": source_facts,
            "runtime_receipt": runtime_facts,
            "parity_v1": v1_parity,
            "parity_v2": v2_parity,
            "journals": "three exact packed 12-byte TopK replays per version; strict header/finite/order/hash checks PASS",
            "topk_and_synthetic_global_merge": True,
            "projection_v1": v1_facts,
            "projection_v2": v2_facts,
            "explicit_fma_is_result_changing": True,
            "topk_seed_overlap": len(overlap),
            "common_seed_capture_bytes_changed": common_changed,
        },
        "limitations": {
            "cubin_bytes_archived": False,
            "cubin_note": "Results bind cubin size/hash/magic and code loads the exact just-hashed blob, but cubin bytes are not serialized; this source-only audit cannot independently rehash them.",
            "global_merge_note": "The calibration merge is a synthetic repeated-shard shape probe, not a 256-distinct-shard/two-tree production proof.",
            "journal_note": "Calibration files are fsynced create-new probes, not the design's temp+atomic-rename+parent-fsync crash-safe protocol.",
            "full_pipeline_projection_claimed": False,
        },
        "missing_prerequisites": missing,
        "repair_assessment": {
            "distinct_v3_with_new_unopened_experts": "DIRECTIONALLY_VALID_IF_RESEALED",
            "requirements": [
                "give the repair a new protocol/version; v2's untouched-validation claim cannot be repaired retrospectively",
                "derive and precommit new expert identities and coordinate plan using source-only rules with no target statistic",
                "prove the new identities are outside every prior opened cache/result/log, not only outside one current cache",
                "keep the validation root absent and inaccessible until one descriptor, all selection states, thresholds, controls and a no-retry sentinel are durably sealed",
                "permit exactly one open and no retry, ABI/arithmetic/K/threshold/identity change after visibility",
                "retain the same fixed-descriptor controls and do not interpret them as a p-value or matched search",
            ],
        },
        "access": {
            "producer_modules_imported": False,
            "third_party_packages_imported": False,
            "accelerator_or_framework_execution": False,
            "network_operations": 0,
            "payload_paths_opened": 0,
            "workspace_enumeration": False,
        },
    }
    receipt["receipt_internal_sha256"] = "0" * 64
    normalized = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["receipt_internal_sha256"] = sha256_bytes(normalized)
    sealed_path = package / "audit_receipt.json"
    if sealed_path.exists():
        sealed = read_json(sealed_path)
        if sealed != receipt:
            raise AuditFailure("sealed audit receipt differs from independent recomputation")
        sealed_normalized = dict(sealed)
        sealed_internal = sealed_normalized["receipt_internal_sha256"]
        sealed_normalized["receipt_internal_sha256"] = "0" * 64
        if sha256_bytes(json.dumps(sealed_normalized, sort_keys=True, separators=(",", ":")).encode()) != sealed_internal:
            raise AuditFailure("sealed audit receipt internal digest mismatch")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
