"""Payload-blind independent audit of the Qwen same-layer common-latent result.

This auditor deliberately does not import producer code and does not open any
Qwen weight file.  It authenticates the frozen source/review closures, binds the
reported input ledger to the frozen panel, and recomputes the count/entropy/MDL
and page-read arithmetic from the evidence embedded in result.json.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Iterable


SOURCE_MANIFEST_SHA256 = "b92d4b5f307ba1d2b6bc6370d0b7cd118c4ab138dc6c8943402efe632a2a5d8f"
SOURCE_ROOT_SHA256 = "f9fe8b64b31edc7599e8e9c302b7e283b2aed9cc24c165916ae3447a9f78311c"
SOURCE_AUDIT_MANIFEST_SHA256 = "9273cbc0c503bf6703bfb71e3d5ef6c390690cdc98ebe2a2ae0ef3be9df6ec00"
SOURCE_AUDIT_ROOT_SHA256 = "de5fce23649530fef56505a4b04e04832f1eac53bb915754bcaae9cb5f13c57c"
DEPLOYMENT_MANIFEST_SHA256 = "a969382640ad69ee71b6029d901d7eade7b88112d582059d83b947e33d1767c3"
DEPLOYMENT_ROOT_SHA256 = "edea8361c0c6d990b9875e0e016e5d31c9cfe525d8803ce2f4d406a2077adae6"
DEPLOYMENT_REVIEW_MANIFEST_SHA256 = "c3357054a7e2bd674b527b37e5ec28c31c925eefb1a09357725a22b9f62e79c9"
DEPLOYMENT_REVIEW_ROOT_SHA256 = "08af8927571b2dd257139d3d93b39110c747bd2ab389b97937395ce1891e1936"
RESULT_SHA256 = "21642374e5de79dc8014aeb6bda751d16eacbe3afcc1365b039e97231df7f1f0"
PANEL_SHA256 = "1da2d993aee033b6dc9d165dc8d5482eecfb276d30e5e398edc388a83b8f5af5"

ALPHABET = 4
PAGE_BYTES = 4096
GLOBAL_HEADER_BYTES = 4096
PRIVATE_HEADER_BYTES = 256
BLOCK_VALUES = 2048
TARGET = 0.22933495044437175
TRIAGE = 0.045
RATES = {"2.15": Fraction(43, 20), "2.5": Fraction(5, 2)}


def req(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def audit_root(rows: list[dict[str, Any]]) -> str:
    canon = [
        {"bytes": int(row["bytes"]), "name": row["name"], "sha256": row["sha256"]}
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_closure(
    root: Path,
    expected_manifest: str,
    expected_schema: str,
    root_field: str,
    expected_root: str,
) -> dict[str, Any]:
    req(root.is_absolute() and root.is_dir() and not root.is_symlink(), f"bad root: {root}")
    manifest_path = root / "SOURCE_MANIFEST.json"
    req(sha256(manifest_path) == expected_manifest, f"manifest hash: {root.name}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    req(raw == canonical_json_bytes(manifest), f"noncanonical manifest: {root.name}")
    req(manifest.get("schema") == expected_schema, f"manifest schema: {root.name}")
    rows = manifest.get("files")
    req(isinstance(rows, list) and rows, f"manifest rows: {root.name}")
    names = [row.get("name") for row in rows]
    req(names == sorted(names) and len(names) == len(set(names)), f"member order: {root.name}")
    req(
        sorted(item.name for item in root.iterdir()) == sorted(names + ["SOURCE_MANIFEST.json"]),
        f"closure mismatch: {root.name}",
    )
    for row in rows:
        path = root / row["name"]
        mode = path.lstat().st_mode
        req(stat.S_ISREG(mode) and not path.is_symlink(), f"unsafe member: {path}")
        req(path.stat().st_size == int(row["bytes"]), f"member bytes: {path}")
        req(sha256(path) == row["sha256"], f"member hash: {path}")
    computed_root = audit_root(rows)
    req(computed_root == expected_root == manifest.get(root_field), f"root hash: {root.name}")
    return manifest


def close_float(actual: Any, expected: float, name: str) -> None:
    req(isinstance(actual, (int, float)) and not isinstance(actual, bool), f"not number: {name}")
    req(math.isfinite(float(actual)), f"nonfinite: {name}")
    req(math.isclose(float(actual), float(expected), rel_tol=2e-14, abs_tol=2e-6), name)


def compare(actual: Any, expected: Any, name: str) -> None:
    if isinstance(expected, float):
        close_float(actual, expected, name)
    elif isinstance(expected, dict):
        req(isinstance(actual, dict) and set(actual) == set(expected), f"keys: {name}")
        for key in expected:
            compare(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, list):
        req(isinstance(actual, list) and len(actual) == len(expected), f"length: {name}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{name}[{index}]")
    else:
        req(type(actual) is type(expected) and actual == expected, f"value: {name}")


def entropy_bits(counts: Iterable[int]) -> float:
    row = tuple(int(x) for x in counts)
    req(all(x >= 0 for x in row), "negative entropy count")
    total = sum(row)
    if total == 0:
        return 0.0
    return total * math.log2(total) - math.fsum(x * math.log2(x) for x in row if x)


def descriptor_bits(counts: Iterable[int]) -> int:
    row = tuple(int(x) for x in counts)
    req(len(row) >= 2 and all(x >= 0 for x in row), "descriptor counts")
    states = sum(row) + 1
    width = 0 if states == 1 else (states - 1).bit_length()
    return (len(row) - 1) * width


def score_from_evidence(evidence: dict[str, Any], cardinality: int, scale_bits: int) -> dict[str, Any]:
    marginal = evidence["marginal_counts"]
    latent = evidence["latent_counts"]
    conditional = evidence["conditional_counts"]
    experts, roles = 16, 2
    coordinates = 768 * 2048
    req(len(marginal) == experts and len(latent) == roles and len(conditional) == experts, "count outer shapes")
    marginal_rows: list[list[int]] = []
    latent_rows: list[list[int]] = []
    conditional_rows: list[list[int]] = []
    for expert in range(experts):
        req(len(marginal[expert]) == roles and len(conditional[expert]) == roles, "count role shapes")
        for role in range(roles):
            mrow = marginal[expert][role]
            req(len(mrow) == ALPHABET and all(type(x) is int and x >= 0 for x in mrow), "marginal row")
            req(sum(mrow) == coordinates, "marginal total")
            marginal_rows.append(mrow)
            req(len(conditional[expert][role]) == cardinality, "conditional state shape")
            state_rows = conditional[expert][role]
            for row in state_rows:
                req(len(row) == ALPHABET and all(type(x) is int and x >= 0 for x in row), "conditional row")
                conditional_rows.append(row)
            for state in range(cardinality):
                req(sum(state_rows[state]) == latent[role][state], "conditional latent projection")
            for symbol in range(ALPHABET):
                req(sum(state_rows[state][symbol] for state in range(cardinality)) == mrow[symbol], "conditional marginal projection")
    for role in range(roles):
        row = latent[role]
        req(len(row) == cardinality and all(type(x) is int and x >= 0 for x in row), "latent row")
        req(sum(row) == coordinates, "latent total")
        latent_rows.append(row)

    marginal_data = math.fsum(entropy_bits(row) for row in marginal_rows)
    latent_data = math.fsum(entropy_bits(row) for row in latent_rows)
    conditional_data = math.fsum(entropy_bits(row) for row in conditional_rows)
    marginal_model = sum(descriptor_bits(row) for row in marginal_rows)
    latent_model = sum(descriptor_bits(row) for row in latent_rows)
    conditional_model = sum(descriptor_bits(row) for row in conditional_rows)
    plane_selector = roles if cardinality == 2 else 0
    family_selector = 1
    selector = plane_selector + family_selector
    weights = experts * roles * coordinates
    marginal_two_part = marginal_data + marginal_model + scale_bits
    common_two_part = latent_data + conditional_data + latent_model + conditional_model + selector + scale_bits
    per_expert_data = []
    per_expert_model = []
    for expert in range(experts):
        rows = [conditional[expert][role][state] for role in range(roles) for state in range(cardinality)]
        per_expert_data.append(math.fsum(entropy_bits(row) for row in rows))
        per_expert_model.append(sum(descriptor_bits(row) for row in rows))
    return {
        "cardinality": cardinality,
        "source_weights": weights,
        "marginal_data_bits": marginal_data,
        "conditional_data_bits": conditional_data,
        "latent_data_bits": latent_data,
        "favorable_gross_gain_bpw": (marginal_data - conditional_data) / weights,
        "net_ideal_gain_bpw": (marginal_data - conditional_data - latent_data) / weights,
        "marginal_model_bits": marginal_model,
        "latent_model_bits": latent_model,
        "conditional_model_bits": conditional_model,
        "plane_selector_bits": plane_selector,
        "family_selector_bits": family_selector,
        "selector_bits": selector,
        "scale_bits_identical_each_scheme": scale_bits,
        "marginal_two_part_bits": marginal_two_part,
        "common_two_part_bits": common_two_part,
        "two_part_gain_bpw": (marginal_two_part - common_two_part) / weights,
        "per_expert_conditional_data_bits": per_expert_data,
        "per_expert_conditional_model_bits": per_expert_model,
        "count_evidence": evidence,
    }


def verify_score(score: dict[str, Any], cardinality: int, scale_bits: int, name: str) -> dict[str, Any]:
    expected = score_from_evidence(score["count_evidence"], cardinality, scale_bits)
    expected["planes"] = score["planes"]
    for key, value in expected.items():
        compare(score[key], value, f"{name}.{key}")
    return expected


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def private_requirements(score: dict[str, Any], scale_bytes_per_expert: int) -> list[int]:
    return [
        PRIVATE_HEADER_BYTES + scale_bytes_per_expert + ceil_div(math.ceil(data) + int(model), 8)
        for data, model in zip(
            score["per_expert_conditional_data_bits"],
            score["per_expert_conditional_model_bits"],
        )
    ]


def physical_envelope(
    *, experts: int, coordinates: int, latent_width: int, rate: Fraction,
    common_model_bits: int, private_required: list[int],
) -> dict[str, Any]:
    weights = experts * 2 * coordinates
    total_pages = ceil_div(weights * rate.numerator, rate.denominator * 8 * PAGE_BYTES)
    actual_rate = Fraction(total_pages * PAGE_BYTES * 8, weights)
    common_unpadded = GLOBAL_HEADER_BYTES + ceil_div(2 * coordinates * latent_width, 8) + ceil_div(common_model_bits, 8)
    common_pages = ceil_div(common_unpadded, PAGE_BYTES)
    req(actual_rate <= Fraction(5, 2), "unexpected page rate failure")
    req(common_pages + experts <= total_pages, "unexpected private-page failure")
    private_total = total_pages - common_pages
    base, extra = divmod(private_total, experts)
    private_pages = [base + (1 if index < extra else 0) for index in range(experts)]
    capacity = all(required <= pages * PAGE_BYTES for required, pages in zip(private_required, private_pages))
    common_padded = common_pages * PAGE_BYTES
    route_bytes: list[int] = []
    denominators: list[str] = []
    physical: list[str] = []
    nonpadding: list[str] = []
    fractions: list[Fraction] = []
    strict = capacity
    for required, pages in zip(private_required, private_pages):
        private_padded = pages * PAGE_BYTES
        routed = common_padded + private_padded
        numerator = experts * routed
        denominator_physical = common_padded + experts * private_padded
        denominator_nonpadding = common_unpadded + experts * required
        pfrac = Fraction(numerator, denominator_physical)
        nfrac = Fraction(numerator, denominator_nonpadding)
        route_bytes.append(routed)
        denominators.append(str(Fraction(denominator_physical, experts)))
        physical.append(str(pfrac))
        nonpadding.append(str(nfrac))
        fractions.extend((pfrac, nfrac))
        strict = strict and numerator < 2 * denominator_physical and numerator < 2 * denominator_nonpadding
    return {
        "status": "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC" if strict else "FAIL_CAPACITY_OR_STRICT_READ_AMPLIFICATION",
        "requested_rate": str(rate),
        "actual_rate_fraction": str(actual_rate),
        "actual_rate_bpw": float(actual_rate),
        "total_pages": total_pages,
        "common_bytes_unpadded": common_unpadded,
        "common_pages": common_pages,
        "private_pages": private_pages,
        "route_bytes": route_bytes,
        "amortized_denominator_bytes": denominators,
        "amplification_physical_fraction": physical,
        "amplification_nonpadding_fraction": nonpadding,
        "max_amplification": float(max(fractions)),
        "capacity_ok": capacity,
        "strictly_below_2x": strict,
    }


def audit(research_root: Path) -> dict[str, Any]:
    research_root = research_root.resolve()
    source = research_root / "same_layer_common_latent_entropy_gate_v0"
    source_audit = research_root / "same_layer_common_latent_entropy_gate_v0_independent_source_audit_r2_20260903"
    deployment = research_root / "same_layer_common_latent_entropy_gate_v0_deployment_20260903"
    deployment_review = research_root / "same_layer_common_latent_entropy_gate_v0_deployment_20260903_independent_review"
    result_path = research_root / "same_layer_common_latent_entropy_gate_v0_runpod_result_20260903" / "result.json"

    verify_closure(source, SOURCE_MANIFEST_SHA256, "same_layer_common_latent_source_manifest_v0", "source_root_sha256", SOURCE_ROOT_SHA256)
    verify_closure(source_audit, SOURCE_AUDIT_MANIFEST_SHA256, "same_layer_common_latent_independent_audit_r2_manifest_v0", "audit_root_sha256", SOURCE_AUDIT_ROOT_SHA256)
    verify_closure(deployment, DEPLOYMENT_MANIFEST_SHA256, "same_layer_common_latent_deployment_manifest_v0", "source_root_sha256", DEPLOYMENT_ROOT_SHA256)
    verify_closure(deployment_review, DEPLOYMENT_REVIEW_MANIFEST_SHA256, "same_layer_common_latent_deployment_review_manifest_v0", "audit_root_sha256", DEPLOYMENT_REVIEW_ROOT_SHA256)

    source_receipt = json.loads((source_audit / "AUDIT_RECEIPT.json").read_bytes())
    req(source_receipt["status"] == "PASS_REPAIRED_SOURCE_ELIGIBLE_FOR_SEPARATE_DEPLOYMENT_REVIEW", "source review verdict")
    req(source_receipt["payload_accessed"] is False and source_receipt["panel"]["payload_files_opened"] == 0, "source review boundary")
    review_receipt = json.loads((deployment_review / "AUDIT_RECEIPT.json").read_bytes())
    req(review_receipt["status"] == "PASS_AUTHORIZE_EXACTLY_ONE_QWEN_RUN", "deployment review verdict")
    req(review_receipt["deployment_manifest_sha256"] == DEPLOYMENT_MANIFEST_SHA256, "deployment review pin")
    req(review_receipt["authorization"]["authorized_use_count"] == 1, "deployment use count")
    req(review_receipt["payload_accessed"] is False, "deployment review boundary")

    # Recheck the sole activation delta independently.
    source_names = sorted(path.name for path in source.iterdir() if path.name != "SOURCE_MANIFEST.json")
    deployment_names = sorted(path.name for path in deployment.iterdir() if path.name != "SOURCE_MANIFEST.json")
    req(source_names == deployment_names, "source/deployment member names")
    for name in source_names:
        left = (source / name).read_bytes()
        right = (deployment / name).read_bytes()
        if name == "run_gate.py":
            req(left.count(b"PAYLOAD_EXECUTION_ENABLED = False") == 1, "source activation literal")
            req(right.count(b"PAYLOAD_EXECUTION_ENABLED = True") == 1, "deployment activation literal")
            req(left.replace(b"PAYLOAD_EXECUTION_ENABLED = False", b"PAYLOAD_EXECUTION_ENABLED = True") == right, "sole activation bytes")
        else:
            req(left == right, f"unexpected deployment delta: {name}")

    req(sha256(result_path) == RESULT_SHA256, "result hash")
    result_raw = result_path.read_bytes()
    result = json.loads(result_raw)
    req(result_raw == (json.dumps(result, sort_keys=True, indent=2) + "\n").encode(), "result canonical formatting")
    panel_path = source / "panel_lock.json"
    req(sha256(panel_path) == PANEL_SHA256 == result["panel_lock_sha256"], "panel binding")
    panel = json.loads(panel_path.read_bytes())
    req(panel["schema"] == "same_layer_common_latent_panel_lock_v0", "panel schema")
    req(panel["model"] == "Qwen/Qwen3-30B-A3B" and panel["layer"] == 15, "panel identity")
    req(panel["d_ff"] == 768 and panel["d_model"] == 2048 and len(panel["experts"]) == 16, "panel geometry")

    ledger = result["input_read_ledger"]
    expected_inputs = [
        {key: row[key] for key in ("bytes", "expert", "relative_path", "role", "sha256")}
        for row in panel["files"]
    ]
    compare(ledger["authenticated_inputs"], expected_inputs, "input_read_ledger.authenticated_inputs")
    source_bytes = sum(row["bytes"] for row in panel["files"])
    weights = len(panel["experts"]) * 2 * panel["d_ff"] * panel["d_model"]
    scale_bits = weights // BLOCK_VALUES * 16
    req(ledger["source_files_read_once"] == 32 and ledger["source_bytes_read_once"] == source_bytes, "source read totals")
    req(ledger["source_logical_host_scan_amplification"] == 1.0, "source scan amplification")
    req(ledger["scale_bits"] == scale_bits == 393216, "scale bits")
    req(ledger["scale_bytes_per_expert"] == scale_bits // 8 // 16 == 3072, "scale bytes/expert")

    variants = result["variants"]
    req(set(variants) == {"binary_favorable_oracle", "binary_charged_mdl", "quaternary"}, "variant set")
    for variant_name in ("binary_favorable_oracle", "binary_charged_mdl"):
        score = variants[variant_name]
        req(len(score["binary_plane_candidate_scores"]) == 4, f"candidate count: {variant_name}")
        candidates = []
        expected_planes = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for index, candidate in enumerate(score["binary_plane_candidate_scores"]):
            req(candidate["planes"] == expected_planes[index], f"candidate planes: {variant_name}")
            recomputed = score_from_evidence(candidate["count_evidence"], 2, scale_bits)
            recomputed["planes"] = candidate["planes"]
            for key in ("conditional_data_bits", "latent_data_bits", "common_two_part_bits", "favorable_gross_gain_bpw", "two_part_gain_bpw", "count_evidence", "planes"):
                compare(candidate[key], recomputed[key], f"{variant_name}.candidate[{index}].{key}")
            candidates.append(recomputed)
        objective = "conditional_data_bits" if variant_name == "binary_favorable_oracle" else "common_two_part_bits"
        chosen = min(candidates, key=lambda item: (item[objective], item["planes"]))
        req(score["planes"] == chosen["planes"], f"plane selection: {variant_name}")
        req(score["binary_plane_selection_objective"] == ("favorable" if variant_name.endswith("oracle") else "charged"), f"objective label: {variant_name}")
        verify_score(score, 2, scale_bits, variant_name)
        compare(score["count_evidence"], chosen["count_evidence"], f"chosen evidence: {variant_name}")
    verify_score(variants["quaternary"], 4, scale_bits, "quaternary")
    req(variants["quaternary"]["planes"] == [None, None], "quaternary planes")
    for name in variants:
        compare(variants[name]["count_evidence"]["marginal_counts"], variants["quaternary"]["count_evidence"]["marginal_counts"], f"shared marginals: {name}")

    envelopes_expected: dict[str, dict[str, Any]] = {}
    for name, latent_width in (("binary_charged_mdl", 1), ("quaternary", 2)):
        score = variants[name]
        private = private_requirements(score, ledger["scale_bytes_per_expert"])
        common_model_bits = int(score["latent_model_bits"]) + int(score["selector_bits"])
        envelopes_expected[name] = {
            label: physical_envelope(
                experts=16,
                coordinates=768 * 2048,
                latent_width=latent_width,
                rate=rate,
                common_model_bits=common_model_bits,
                private_required=private,
            )
            for label, rate in RATES.items()
        }
    compare(result["physical_page_envelopes"], envelopes_expected, "physical_page_envelopes")
    eligible = {
        name: [label for label in ("2.15", "2.5") if rows[label]["status"] == "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC" and rows[label]["capacity_ok"] is True and rows[label]["strictly_below_2x"] is True]
        for name, rows in envelopes_expected.items()
    }
    compare(result["read_eligible_rate_endpoints"], eligible, "read_eligible_rate_endpoints")

    favorable = {
        "binary_favorable_oracle": variants["binary_favorable_oracle"],
        "quaternary": variants["quaternary"],
    }
    best_favorable_name, best_favorable = max(favorable.items(), key=lambda item: item[1]["favorable_gross_gain_bpw"])
    charged = {name: variants[name] for name in ("binary_charged_mdl", "quaternary")}
    best_charged_name, best_charged = max(charged.items(), key=lambda item: item[1]["two_part_gain_bpw"])
    read_candidates = {name: charged[name] for name, endpoints in eligible.items() if endpoints}
    best_read_name, best_read = max(read_candidates.items(), key=lambda item: item[1]["two_part_gain_bpw"])
    hard_kill = best_favorable["favorable_gross_gain_bpw"] < TARGET
    req(best_favorable_name == "quaternary" and best_charged_name == "binary_charged_mdl" and best_read_name == "binary_charged_mdl", "best variants")
    req(hard_kill, "hard-kill predicate")
    req(result["schema"] == "same_layer_common_latent_entropy_gate_result_v0", "result schema")
    req(result["claim_boundary"] == "IDEAL_LABEL_MDL_APERTURE_ONLY_NOT_A_FINITE_CODEC", "claim boundary")
    req(result["status"] == "HARD_KILL_FAVORABLE_IDEAL_BELOW_TARGET", "status")
    req(result["target_gain_bpw_up_down"] == TARGET and result["triage_gain_bpw_nonpromoting"] == TRIAGE, "thresholds")
    req(result["best_favorable_variant"] == best_favorable_name, "best favorable name")
    close_float(result["best_favorable_gross_gain_bpw"], best_favorable["favorable_gross_gain_bpw"], "best favorable gain")
    req(result["best_charged_variant"] == best_charged_name, "best charged name")
    close_float(result["best_charged_two_part_gain_bpw"], best_charged["two_part_gain_bpw"], "best charged gain")
    req(result["best_read_eligible_charged_variant"] == best_read_name, "best read name")
    close_float(result["best_read_eligible_charged_two_part_gain_bpw"], best_read["two_part_gain_bpw"], "best read gain")
    req(result["triage_threshold_reached_nonpromoting"] is True, "triage flag")
    req(result["hard_kill_before_controls_and_finite_coder"] is True, "hard-kill flag")
    req(result["eligible_for_finite_coder_research"] is False, "finite eligibility")
    req(result["controls_run"] == 0 and result["controls"] == [] and result["control_best_gain_summary"] is None, "early stop")
    req(result["cuda"] == {"cupy_version": "14.2.0", "device_id": 0, "device_name": "NVIDIA GeForce RTX 5090"}, "CUDA receipt")

    gross = float(best_favorable["favorable_gross_gain_bpw"])
    return {
        "schema": "same_layer_common_latent_qwen_result_independent_audit_receipt_v0",
        "status": "PASS_INTERNAL_MATH_CONFIRMS_HARD_KILL",
        "auditor_id": "same_layer_qwen_result_audit",
        "claim_boundary": "PAYLOAD_BLIND_RESULT_ARITHMETIC_AUDIT_NOT_EXECUTION_ATTESTATION_OR_FINITE_CODEC_EVIDENCE",
        "payload_accessed": False,
        "payload_files_opened": 0,
        "dependencies": {
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_root_sha256": SOURCE_ROOT_SHA256,
            "source_audit_manifest_sha256": SOURCE_AUDIT_MANIFEST_SHA256,
            "source_audit_root_sha256": SOURCE_AUDIT_ROOT_SHA256,
            "deployment_manifest_sha256": DEPLOYMENT_MANIFEST_SHA256,
            "deployment_root_sha256": DEPLOYMENT_ROOT_SHA256,
            "deployment_review_manifest_sha256": DEPLOYMENT_REVIEW_MANIFEST_SHA256,
            "deployment_review_root_sha256": DEPLOYMENT_REVIEW_ROOT_SHA256,
            "panel_lock_sha256": PANEL_SHA256,
            "result_sha256": RESULT_SHA256,
            "result_bytes": result_path.stat().st_size,
        },
        "execution_provenance": {
            "deployment_authority_was_one_run": True,
            "result_embeds_panel_hash": True,
            "result_embeds_deployment_manifest_or_executable_hash": False,
            "limitation": "The result is exactly reproducible from its embedded count evidence and matches the reviewed worker's output schema, but result.json alone does not cryptographically attest which executable produced it.",
        },
        "panel": {
            "model": panel["model"],
            "revision": panel["revision"],
            "layer": panel["layer"],
            "experts": panel["experts"],
            "roles": ["Up", "Down.T"],
            "source_files": 32,
            "source_bytes_reported_read_once": source_bytes,
            "source_weights": weights,
            "scale_bits": scale_bits,
        },
        "math": {
            "best_favorable_variant": best_favorable_name,
            "best_favorable_gross_gain_bpw": gross,
            "target_gain_bpw_up_down": TARGET,
            "gross_fraction_of_target": gross / TARGET,
            "gross_shortfall_bpw": TARGET - gross,
            "best_charged_variant": best_charged_name,
            "best_charged_two_part_gain_bpw": float(best_charged["two_part_gain_bpw"]),
            "best_read_eligible_charged_variant": best_read_name,
            "best_read_eligible_charged_two_part_gain_bpw": float(best_read["two_part_gain_bpw"]),
            "triage_threshold_reached_nonpromoting": True,
            "all_count_entropy_mdl_fields_recomputed": True,
        },
        "read_envelopes": {
            "binary_read_eligible_endpoints": eligible["binary_charged_mdl"],
            "binary_max_amplification_2_15": envelopes_expected["binary_charged_mdl"]["2.15"]["max_amplification"],
            "binary_max_amplification_2_5": envelopes_expected["binary_charged_mdl"]["2.5"]["max_amplification"],
            "quaternary_read_eligible_endpoints": eligible["quaternary"],
            "quaternary_max_amplification_2_15": envelopes_expected["quaternary"]["2.15"]["max_amplification"],
            "quaternary_max_amplification_2_5": envelopes_expected["quaternary"]["2.5"]["max_amplification"],
            "all_capacity_and_fraction_fields_recomputed": True,
        },
        "disposition": {
            "status": result["status"],
            "controls_run": 0,
            "early_stop_correct": True,
            "eligible_for_finite_coder_research": False,
            "finite_codec_emitted": False,
            "conclusion": "Identity-aligned same-layer common labels show a 0.04704 bpw favorable conditional-entropy signal, but this is only 20.51% of the gate and becomes negative after the common latent/model cost. The fixed branch is correctly hard-killed.",
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.research_root)), sort_keys=True, separators=(",", ":")))
