"""Independent payload-blind hostile audit of the frozen v0 source package."""

from __future__ import annotations

import argparse
import ast
import contextlib
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import stat
import sys

import numpy as np


EXPECTED_SOURCE_MANIFEST_SHA256 = "238ce67f3670566c277d7baac019a69227ccff05ae944f1561ff8a0d32b1bce9"
EXPECTED_SOURCE_ROOT_SHA256 = "16cceafcfe06e1c2683c0e89048700edd47fda395a2a6d06a70cef19d8eb858b"
EXPECTED_PRIOR_RESULT_SHA256 = "a1ef6fb136027525b6312635cdcca320f05f51c1340c3875b32192454aac1bb3"
MODEL = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
EXPERTS = list(range(0, 128, 8))
TARGET_GLOBAL_RATE_EQUIVALENT_BPW = 0.1528899669629145
TARGET_UP_DOWN_BPW = 0.22933495044437175


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticate_source(root: Path) -> dict:
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "source package path")
    manifest_path = root / "SOURCE_MANIFEST.json"
    require(sha(manifest_path) == EXPECTED_SOURCE_MANIFEST_SHA256, "source manifest digest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    canonical = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    require(raw == canonical, "source manifest canonical encoding")
    require(manifest.get("schema") == "same_layer_common_latent_source_manifest_v0", "source schema")
    rows = manifest.get("files")
    require(isinstance(rows, list) and rows, "source rows")
    names = [row["name"] for row in rows]
    require(names == sorted(names) and len(names) == len(set(names)), "source member order")
    require(sorted(path.name for path in root.iterdir()) == sorted(names + ["SOURCE_MANIFEST.json"]),
            "source exact closure")
    canonical_rows = []
    for row in rows:
        require(set(row) == {"bytes", "name", "sha256"}, "source row fields")
        path = root / row["name"]
        require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), "source regular member")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                f"source member mismatch: {path.name}")
        canonical_rows.append({"bytes": row["bytes"], "name": row["name"], "sha256": row["sha256"]})
    source_root = hashlib.sha256(
        json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(source_root == EXPECTED_SOURCE_ROOT_SHA256 == manifest["source_root_sha256"],
            "source root")
    return {"member_count_excluding_manifest": len(rows), "source_root_sha256": source_root}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module spec: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def literal_assignments(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return values


def audit_hold(root: Path) -> dict:
    gate_path = root / "run_gate.py"
    values = literal_assignments(gate_path)
    require(values.get("PAYLOAD_EXECUTION_ENABLED") is False, "payload HOLD literal")
    require(values.get("AUTHORIZATION_PHRASE") == "EXECUTE_AUTHENTICATED_QWEN_L15_COMMON_LATENT_V0",
            "authorization phrase")
    require(values.get("PANEL_LOCK_SHA256") == sha(root / "panel_lock.json"), "panel source pin")
    require(values.get("CORE_SHA256") == sha(root / "common_latent_core.py"), "core source pin")
    require(values.get("WORKER_SHA256") == sha(root / "cupy_worker.py"), "worker source pin")
    text = gate_path.read_text(encoding="utf-8")
    require(text.index("if not PAYLOAD_EXECUTION_ENABLED:") < text.index("Path(__file__)"),
            "HOLD ordering")
    gate = load_module("same_layer_common_latent_gate_audit_hold", gate_path)
    class ForbiddenPath:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Path touched before HOLD")
    gate.Path = ForbiddenPath
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = gate.main([
            "--authorization", gate.AUTHORIZATION_PHRASE,
            "--payload-root", "FORBIDDEN_PAYLOAD",
            "--output", "FORBIDDEN_OUTPUT.json",
        ])
    result = json.loads(stdout.getvalue())
    require(code == 2 and result["status"] == "HOLD_NO_PAYLOAD_ACCESS", "dynamic HOLD")
    require("same_layer_common_latent_cupy_worker_v0" not in sys.modules, "worker imported before HOLD")
    return {
        "payload_execution_enabled": False,
        "dynamic_status": result["status"],
        "path_access_before_hold": False,
        "worker_import_before_hold": False,
    }


def audit_panel(root: Path, prior_result: Path) -> dict:
    panel = json.loads((root / "panel_lock.json").read_text(encoding="utf-8"))
    require(panel["schema"] == "same_layer_common_latent_panel_lock_v0", "panel schema")
    require(panel["model"] == MODEL and panel["revision"] == REVISION, "panel model revision")
    require(panel["layer"] == 15 and panel["dtype"] == "BF16_LE", "panel layer dtype")
    require(panel["d_ff"] == 768 and panel["d_model"] == 2048, "panel shape")
    require(panel["experts"] == EXPERTS and len(panel["files"]) == 32, "panel expert closure")
    bindings = []
    for index, row in enumerate(panel["files"]):
        expert = EXPERTS[index // 2]
        role = ("up", "down")[index % 2]
        require(row["expert"] == expert and row["role"] == role, "panel binding sequence")
        require(row["relative_path"] ==
                f"qwen_weight_cache/rd_structure_diag_cross_expert/l15e{expert}_{role}.bf16.bin",
                "panel explicit relative path")
        require(".." not in Path(row["relative_path"]).parts and not Path(row["relative_path"]).is_absolute(),
                "panel safe path")
        require(row["bytes"] == 768 * 2048 * 2, "panel byte count")
        require(len(row["sha256"]) == 64 and all(c in "0123456789abcdef" for c in row["sha256"]),
                "panel digest")
        if role == "up":
            require(row["raw_shape"] == [768, 2048] and row["canonical_shape"] == [768, 2048]
                    and row["down_transposed"] is False, "Up canonicalization lock")
        else:
            require(row["raw_shape"] == [2048, 768] and row["canonical_shape"] == [768, 2048]
                    and row["down_transposed"] is True, "Down.T canonicalization lock")
        bindings.append((expert, role, row["relative_path"], row["bytes"], row["sha256"]))
    require(prior_result.is_absolute() and prior_result.is_file() and not prior_result.is_symlink(),
            "prior evidence path")
    require(sha(prior_result) == EXPECTED_PRIOR_RESULT_SHA256, "prior evidence digest")
    prior = json.loads(prior_result.read_text(encoding="utf-8"))
    prior_bindings = [
        (row["expert"], row["role"], row["local_path"], row["bytes"], row["sha256"])
        for row in prior["bindings"]["sources"]
    ]
    require(bindings == prior_bindings, "panel differs from independently recorded Qwen bindings")
    return {
        "model": MODEL,
        "revision": REVISION,
        "layer": 15,
        "experts": EXPERTS,
        "roles": ["Up", "Down.T"],
        "file_count": 32,
        "bytes_per_file": 3145728,
        "panel_lock_sha256": sha(root / "panel_lock.json"),
        "prior_binding_evidence_sha256": EXPECTED_PRIOR_RESULT_SHA256,
        "exact_prior_binding_sequence_match": True,
        "payload_files_opened": 0,
    }


def entropy_bits(counts) -> float:
    counts = [int(x) for x in counts]
    total = sum(counts)
    return 0.0 if total == 0 else total * math.log2(total) - sum(x * math.log2(x) for x in counts if x)


def descriptor_bits(counts) -> int:
    counts = [int(x) for x in counts]
    states = sum(counts) + 1
    width = 0 if states == 1 else (states - 1).bit_length()
    return (len(counts) - 1) * width


def independent_summary(labels: np.ndarray, cardinality: int, planes=None) -> dict:
    e, roles, n = labels.shape
    gray = np.asarray(((0, 0), (0, 1), (1, 1), (1, 0)), dtype=np.uint8)
    marginal = np.zeros((e, roles, 4), dtype=np.int64)
    latent = np.zeros((roles, cardinality), dtype=np.int64)
    conditional = np.zeros((e, roles, cardinality, 4), dtype=np.int64)
    for expert in range(e):
        for role in range(roles):
            for symbol in range(4):
                marginal[expert, role, symbol] = int(np.sum(labels[expert, role] == symbol))
    for role in range(roles):
        source = labels[:, role] if cardinality == 4 else gray[labels[:, role], planes[role]]
        counts = np.asarray([[int(np.sum(source[:, i] == state)) for i in range(n)]
                             for state in range(cardinality)])
        u = np.argmax(counts, axis=0).astype(np.uint8)
        for state in range(cardinality):
            latent[role, state] = int(np.sum(u == state))
            for expert in range(e):
                for symbol in range(4):
                    conditional[expert, role, state, symbol] = int(
                        np.sum((u == state) & (labels[expert, role] == symbol))
                    )
    return {"marginal": marginal, "latent": latent, "conditional": conditional}


def audit_math(root: Path) -> tuple[dict, object]:
    core = load_module("common_latent_core_audit_snapshot", root / "common_latent_core.py")
    require(core.TARGET_GAIN_BPW == TARGET_UP_DOWN_BPW, "core threshold literal")
    baseline_f = 0.9888693569009007
    target_f = 0.8
    global_gap = -0.5 * math.log2(target_f / baseline_f)
    role_scaled = global_gap * Fraction(3, 2)
    require(abs(global_gap - TARGET_GLOBAL_RATE_EQUIVALENT_BPW) < 2e-15, "global gap derivation")
    require(abs(float(role_scaled) - TARGET_UP_DOWN_BPW) < 2e-15, "Up/Down threshold derivation")

    labels = np.asarray([
        [[0, 1, 2, 3, 0, 2, 1, 3], [3, 2, 1, 0, 1, 1, 2, 2]],
        [[0, 1, 3, 3, 2, 2, 1, 0], [3, 0, 1, 0, 1, 2, 2, 3]],
        [[1, 1, 2, 0, 0, 3, 1, 3], [2, 2, 1, 0, 3, 1, 2, 2]],
        [[0, 2, 2, 3, 0, 2, 0, 3], [3, 2, 0, 0, 1, 1, 3, 2]],
    ], dtype=np.uint8)
    checked = {}
    for cardinality, planes in ((2, (0, 1)), (4, None)):
        independent = independent_summary(labels, cardinality, planes)
        source = core.summarize_counts_cpu(labels, cardinality, planes)
        require(np.array_equal(independent["marginal"], source["marginal_counts"]), "marginal counts")
        require(np.array_equal(independent["latent"], source["latent_counts"]), "latent counts")
        require(np.array_equal(independent["conditional"], source["conditional_counts"]),
                "conditional counts")
        scored = core.score_count_summary(source, scale_bits=123)
        marginal_bits = sum(entropy_bits(row) for row in independent["marginal"].reshape(-1, 4))
        latent_bits = sum(entropy_bits(row) for row in independent["latent"])
        conditional_bits = sum(entropy_bits(row) for row in independent["conditional"].reshape(-1, 4))
        marginal_model = sum(descriptor_bits(row) for row in independent["marginal"].reshape(-1, 4))
        latent_model = sum(descriptor_bits(row) for row in independent["latent"])
        conditional_model = sum(descriptor_bits(row) for row in independent["conditional"].reshape(-1, 4))
        selector = 3 if cardinality == 2 else 1
        require(scored["marginal_data_bits"] == marginal_bits, "marginal entropy")
        require(scored["latent_data_bits"] == latent_bits, "latent entropy")
        require(scored["conditional_data_bits"] == conditional_bits, "conditional entropy")
        require(scored["marginal_model_bits"] == marginal_model, "marginal descriptor")
        require(scored["latent_model_bits"] == latent_model, "latent descriptor")
        require(scored["conditional_model_bits"] == conditional_model, "conditional descriptor")
        require(scored["selector_bits"] == selector, "selector charge")
        require(scored["marginal_two_part_bits"] == marginal_bits + marginal_model + 123,
                "marginal two-part identity")
        require(scored["common_two_part_bits"] ==
                latent_bits + conditional_bits + latent_model + conditional_model + selector + 123,
                "common two-part identity")
        checked[f"k{cardinality}"] = {
            "counts_exact": True,
            "entropy_identity_exact": True,
            "descriptor_identity_exact": True,
            "selector_bits": selector,
        }

    values = np.asarray([
        [-2.0, -1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 2.0],
        [-1.5, -1.0, -0.5, -0.125, 0.125, 0.5, 1.0, 1.5],
    ], dtype=np.float64)
    got = core.quantize_canonical_cpu(values, 8)
    flat = values.reshape(-1)
    independent_scale_bits = []
    independent_labels = []
    independent_recon = []
    for lo in range(0, flat.size, 8):
        block = flat[lo:lo + 8]
        rms = math.sqrt(math.fsum(float(x) * float(x) for x in block) / len(block))
        scale16 = np.float16(rms)
        independent_scale_bits.append(int(scale16.view(np.uint16)))
        scale = float(scale16)
        threshold = 0.981598821873 * scale
        for value in block:
            label = 0 if value < -threshold else 1 if value < 0 else 2 if value <= threshold else 3
            independent_labels.append(label)
            independent_recon.append((-1.510417608, -0.452780039, 0.452780039, 1.510417608)[label] * scale)
    require(got.scale_u16.tolist() == independent_scale_bits, "binary16 scale bits")
    require(got.labels.reshape(-1).tolist() == independent_labels, "quantizer labels")
    require(np.array_equal(got.reconstruction.reshape(-1), np.asarray(independent_recon)),
            "quantizer reconstruction")
    tie = np.asarray([[0, 3], [1, 2]], dtype=np.uint8)
    require(core.modal_common_latent_cpu(tie, 4).tolist() == [0, 2], "lower-symbol modal ties")
    return ({
        "global_rate_equivalent_gap_bpw": global_gap,
        "up_down_fraction_of_full_swiglu": "2/3",
        "derived_up_down_threshold_bpw": float(role_scaled),
        "frozen_threshold_bpw": core.TARGET_GAIN_BPW,
        "threshold_exact_within_2e_15": True,
        "count_entropy_descriptor_checks": checked,
        "quantizer": {
            "fp64_rms_to_binary16_bits_match": True,
            "decoded_binary16_threshold_labels_match": True,
            "decoded_binary16_reconstruction_match": True,
            "down_transpose_contract_inspected": True,
        },
    }, core)


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    z = (value + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def independent_affine(n: int, seed: int, expert: int, role: int) -> tuple[int, int]:
    material = seed ^ (expert << 24) ^ (role << 56)
    a = splitmix64(material) % n or 1
    while math.gcd(a, n) != 1:
        a = (a + 1) % n or 1
    b = splitmix64(material ^ 0xD1B54A32D192ED03) % n
    return a, b


def audit_controls(core) -> dict:
    labels = np.arange(5 * 2 * 257, dtype=np.uint32).reshape(5, 2, 257).astype(np.uint8) % 4
    for seed in core.CONTROL_SEEDS:
        scrambled = core.coordinate_scramble_cpu(labels, seed)
        for expert in range(5):
            for role in range(2):
                expected = independent_affine(257, seed, expert, role)
                require(core.affine_permutation_parameters(257, seed, expert, role) == expected,
                        "affine parameter derivation")
                a, b = expected
                indices = (a * np.arange(257, dtype=np.int64) + b) % 257
                require(np.unique(indices).size == 257, "affine map not bijective")
                require(np.array_equal(scrambled[expert, role], labels[expert, role, indices]),
                        "scramble implementation")
                require(np.array_equal(np.bincount(scrambled[expert, role], minlength=4),
                                       np.bincount(labels[expert, role], minlength=4)),
                        "scramble changes marginal")
    return {
        "seed_count": len(core.CONTROL_SEEDS),
        "seeds": list(core.CONTROL_SEEDS),
        "independent_expert_role_affine_parameters_match": True,
        "bijection_for_all_checked_streams": True,
        "marginal_histograms_exactly_preserved": True,
    }


def independent_envelope(e, n, latent_width, rate, common_model_bits, private, page=4096):
    total_pages = math.ceil((e * 2 * n * rate.numerator) / (rate.denominator * 8 * page))
    actual_rate = Fraction(total_pages * page * 8, e * 2 * n)
    common_unpadded = page + math.ceil((2 * n * latent_width) / 8) + math.ceil(common_model_bits / 8)
    common_pages = math.ceil(common_unpadded / page)
    private_total = total_pages - common_pages
    base, extra = divmod(private_total, e)
    pages = [base + (i < extra) for i in range(e)]
    cpage = common_pages * page
    physical = []
    nonpadding = []
    for required, count in zip(private, pages):
        routed = cpage + count * page
        physical.append(Fraction(e * routed, cpage + e * count * page))
        nonpadding.append(Fraction(e * routed, common_unpadded + e * required))
    return total_pages, actual_rate, common_unpadded, common_pages, pages, physical, nonpadding


def audit_read_ledgers(core) -> dict:
    e, n = 16, 768 * 2048
    cases = {}
    for name, width, model, rate in (
        ("binary_2_15", 1, 45, Fraction(43, 20)),
        ("binary_2_5", 1, 45, Fraction(5, 2)),
        ("quaternary_2_15", 2, 127, Fraction(43, 20)),
        ("quaternary_2_5", 2, 127, Fraction(5, 2)),
    ):
        private = [780000] * e
        got = core.physical_page_envelope(
            expert_count=e, coordinates_per_role=n, latent_bits_per_coordinate=width,
            requested_rate=rate, common_model_bits=model, private_required_bytes=private,
        )
        independent = independent_envelope(e, n, width, rate, model, private)
        total, actual, common_unpadded, common_pages, pages, physical, nonpadding = independent
        require(got["total_pages"] == total and got["actual_rate_fraction"] == str(actual),
                "page/rate ledger")
        require(got["common_bytes_unpadded"] == common_unpadded and got["common_pages"] == common_pages,
                "common ledger")
        require(got["private_pages"] == pages, "private page allocation")
        require(got["amplification_physical_fraction"] == [str(x) for x in physical],
                "physical amplification ledger")
        require(got["amplification_nonpadding_fraction"] == [str(x) for x in nonpadding],
                "nonpadding amplification ledger")
        require(got["strictly_below_2x"] ==
                all(x < 2 for x in physical + nonpadding), "strict read predicate")
        cases[name] = {
            "actual_rate_fraction": str(actual),
            "total_pages": total,
            "common_pages": common_pages,
            "max_physical": float(max(physical)),
            "max_nonpadding": float(max(nonpadding)),
            "strictly_below_2x": got["strictly_below_2x"],
        }
    equality = core.physical_page_envelope(
        expert_count=2, coordinates_per_role=16384, latent_bits_per_coordinate=1,
        requested_rate=Fraction(5, 2), common_model_bits=0,
        private_required_bytes=[4096, 2048],
    )
    padding = core.physical_page_envelope(
        expert_count=2, coordinates_per_role=16384, latent_bits_per_coordinate=1,
        requested_rate=Fraction(5, 2), common_model_bits=0,
        private_required_bytes=[1, 1],
    )
    require(not equality["strictly_below_2x"] and equality["max_amplification"] == 2.0,
            "exact 2x must fail")
    require(max(Fraction(x) for x in padding["amplification_physical_fraction"]) < 2 and
            max(Fraction(x) for x in padding["amplification_nonpadding_fraction"]) > 2 and
            not padding["strictly_below_2x"], "padding attack")
    return {
        "formula": "max((Cpage+Ppage)/(Cpage/E+Ppage),(Cpage+Ppage)/(Cunpadded/E+Prequired))",
        "qwen_anchor_cases": cases,
        "exact_2x_rejected": True,
        "padding_attack_rejected_by_nonpadding_ledger": True,
        "ledger_math": "PASS",
    }


class FakeRuntime:
    @staticmethod
    def getDeviceProperties(device):
        return {"name": b"source-free-fake-device"}

    @staticmethod
    def getDevice():
        return 0


class FakeCuda:
    runtime = FakeRuntime()


class FakeCupy:
    __version__ = "source-free-fake"
    cuda = FakeCuda()


class ControlMarker:
    pass


def audit_read_gate_regression(root: Path, core) -> dict:
    # Load the exact worker snapshot, but replace every payload/GPU operation.
    # This executes only result-selection logic and the real page-envelope math.
    sys.modules["common_latent_core"] = core
    worker = load_module("same_layer_common_latent_worker_read_regression", root / "cupy_worker.py")
    source_marker = object()
    worker._cupy = lambda: FakeCupy
    worker.load_quantized_panel_gpu = lambda panel, payload_root: (
        source_marker,
        {"scale_bits": 0, "scale_bytes_per_expert": 0, "synthetic": True},
    )
    worker.coordinate_scramble_gpu = lambda labels, seed: ControlMarker()

    def fake_score(labels, cardinality, scale_bits=0, selection_objective="charged"):
        source = labels is source_marker
        return {
            "cardinality": cardinality,
            "planes": [0, 0] if cardinality == 2 else [None, None],
            "source_weights": 16 * 2 * 768 * 2048,
            "favorable_gross_gain_bpw": 0.5 if source else 0.0,
            "two_part_gain_bpw": 0.5 if source else 0.0,
            "latent_model_bits": 0,
            "selector_bits": 3 if cardinality == 2 else 1,
            "per_expert_conditional_data_bits": [0.0] * 16,
            "per_expert_conditional_model_bits": [0] * 16,
        }

    worker.score_labels_gpu = fake_score
    result = worker.run_authorized_panel(root / "panel_lock.json", Path("FORBIDDEN_PAYLOAD_NOT_OPENED"))
    failures = {
        family: {rate: envelope["status"] for rate, envelope in rates.items()}
        for family, rates in result["physical_page_envelopes"].items()
    }
    all_failed = all(
        not envelope["strictly_below_2x"]
        for rates in result["physical_page_envelopes"].values()
        for envelope in rates.values()
    )
    require(result["status"] == "SURVIVE_IDEAL_APERTURE_REQUIRES_FINITE_CODER", "regression survivor")
    require(result["eligible_for_finite_coder_research"] is True, "regression eligibility")
    require(all_failed, "regression envelopes unexpectedly pass")
    return {
        "status_returned": result["status"],
        "eligible_for_finite_coder_research_returned": True,
        "all_four_read_envelopes_failed": True,
        "envelope_statuses": failures,
        "payload_files_opened": 0,
        "defect": (
            "run_authorized_panel computes physical and nonpadding envelopes but does not use "
            "their status, capacity_ok, or strictly_below_2x values when setting status and eligibility"
        ),
        "required_fix": (
            "After constructing both family/rate envelopes, make promotion/eligibility conditional "
            "on an explicitly frozen feasible-rate rule whose selected envelope has capacity_ok=true "
            "and strictly_below_2x=true; add this regression as a mandatory source test"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", required=True)
    parser.add_argument("--prior-binding-result", required=True)
    args = parser.parse_args()
    source = Path(args.source_package).resolve(strict=True)
    prior = Path(args.prior_binding_result).resolve(strict=True)
    closure = authenticate_source(source)
    hold = audit_hold(source)
    panel = audit_panel(source, prior)
    math_receipt, core = audit_math(source)
    controls = audit_controls(core)
    ledgers = audit_read_ledgers(core)
    regression = audit_read_gate_regression(source, core)
    receipt = {
        "schema": "same_layer_common_latent_independent_source_audit_receipt_v0",
        "status": "BLOCKED_MATERIAL_READ_GATE_DEFECT",
        "auditor_id": "common_latent_source_audit",
        "independence": "separate agent; no producer-source edits",
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "source_root_sha256": closure["source_root_sha256"],
        "source_member_count_excluding_manifest": closure["member_count_excluding_manifest"],
        "claim_boundary": "PAYLOAD_BLIND_SOURCE_REVIEW_ONLY_NO_QWEN_EXECUTION_AUTHORITY",
        "payload_accessed": False,
        "source_files_modified": False,
        "hold": hold,
        "panel": panel,
        "math_and_quantizer": math_receipt,
        "entropy_mdl_accounting": {
            "marginal_plugin_entropy": "PASS",
            "conditional_plugin_entropy_given_raw_U": "PASS",
            "latent_entropy": "PASS",
            "fixed_width_count_descriptors_final_count_derived": "PASS_CONSERVATIVE",
            "scale_bits_cancel_identically": "PASS",
            "binary_plane_selector_bits": 2,
            "family_selector_bits": 1,
            "finite_coder_claimed": False,
        },
        "controls": controls,
        "read_ledgers": ledgers,
        "material_defect": regression,
        "authorization": "DENY_PAYLOAD_DEPLOYMENT_FROM_THIS_SOURCE_SNAPSHOT",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
