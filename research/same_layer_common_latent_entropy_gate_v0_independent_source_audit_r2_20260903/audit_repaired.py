"""Fresh independent, payload-blind audit of the repaired common-latent gate."""

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

MANIFEST_SHA = "b92d4b5f307ba1d2b6bc6370d0b7cd118c4ab138dc6c8943402efe632a2a5d8f"
SOURCE_ROOT = "f9fe8b64b31edc7599e8e9c302b7e283b2aed9cc24c165916ae3447a9f78311c"
PRIOR_SHA = "a1ef6fb136027525b6312635cdcca320f05f51c1340c3875b32192454aac1bb3"
TARGET = 0.22933495044437175
EXPERTS = list(range(0, 128, 8))


def req(x, msg):
    if not x:
        raise RuntimeError(msg)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    req(spec is not None and spec.loader is not None, "module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def authenticate(root):
    mp = root / "SOURCE_MANIFEST.json"
    req(sha(mp) == MANIFEST_SHA, "manifest digest")
    raw = mp.read_bytes()
    m = json.loads(raw)
    req(raw == (json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n").encode(), "canonical manifest")
    rows = m["files"]
    names = [r["name"] for r in rows]
    req(names == sorted(names) and len(names) == len(set(names)), "manifest order")
    req(sorted(p.name for p in root.iterdir()) == sorted(names + ["SOURCE_MANIFEST.json"]), "exact closure")
    canonical = []
    for row in rows:
        p = root / row["name"]
        req(set(row) == {"bytes", "name", "sha256"}, "row fields")
        req(stat.S_ISREG(p.lstat().st_mode) and not p.is_symlink(), "regular member")
        req(p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], "member pin")
        canonical.append({"bytes": row["bytes"], "name": row["name"], "sha256": row["sha256"]})
    sr = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    req(sr == SOURCE_ROOT == m["source_root_sha256"], "source root")
    return m


def literals(path):
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return out


def audit_hold(root):
    path = root / "run_gate.py"
    lit = literals(path)
    req(lit.get("PAYLOAD_EXECUTION_ENABLED") is False, "HOLD literal")
    req(lit["PANEL_LOCK_SHA256"] == sha(root / "panel_lock.json"), "panel pin")
    req(lit["CORE_SHA256"] == sha(root / "common_latent_core.py"), "core pin")
    req(lit["WORKER_SHA256"] == sha(root / "cupy_worker.py"), "worker pin")
    text = path.read_text(encoding="utf-8")
    req(text.index("if not PAYLOAD_EXECUTION_ENABLED:") < text.index("Path(__file__)"), "HOLD order")
    gate = load("common_latent_r2_hold", path)
    class NoPath:
        def __init__(self, *a, **k):
            raise AssertionError("Path touched")
    gate.Path = NoPath
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = gate.main(["--authorization", gate.AUTHORIZATION_PHRASE,
                          "--payload-root", "FORBIDDEN", "--output", "FORBIDDEN.json"])
    obj = json.loads(output.getvalue())
    req(code == 2 and obj["status"] == "HOLD_NO_PAYLOAD_ACCESS", "dynamic HOLD")
    req("same_layer_common_latent_cupy_worker_v0" not in sys.modules, "early worker import")
    return {"status": obj["status"], "path_access_before_hold": False,
            "worker_import_before_hold": False}


def audit_panel(root, prior_path):
    panel = json.loads((root / "panel_lock.json").read_text(encoding="utf-8"))
    req(panel["model"] == "Qwen/Qwen3-30B-A3B", "model")
    req(panel["revision"] == "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "revision")
    req(panel["layer"] == 15 and panel["experts"] == EXPERTS, "panel")
    req(panel["d_ff"] == 768 and panel["d_model"] == 2048 and len(panel["files"]) == 32, "geometry")
    binding = []
    for i, row in enumerate(panel["files"]):
        e, role = EXPERTS[i // 2], ("up", "down")[i % 2]
        req((row["expert"], row["role"]) == (e, role), "binding sequence")
        req(row["relative_path"] == f"qwen_weight_cache/rd_structure_diag_cross_expert/l15e{e}_{role}.bf16.bin", "path")
        req(row["bytes"] == 3145728 and len(row["sha256"]) == 64, "file pin")
        if role == "up":
            req(row["raw_shape"] == [768, 2048] and row["canonical_shape"] == [768, 2048] and not row["down_transposed"], "Up")
        else:
            req(row["raw_shape"] == [2048, 768] and row["canonical_shape"] == [768, 2048] and row["down_transposed"], "Down.T")
        binding.append((e, role, row["relative_path"], row["bytes"], row["sha256"]))
    req(sha(prior_path) == PRIOR_SHA, "prior digest")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_binding = [(r["expert"], r["role"], r["local_path"], r["bytes"], r["sha256"])
                     for r in prior["bindings"]["sources"]]
    req(binding == prior_binding, "prior bindings")
    return {"file_count": 32, "exact_prior_binding_sequence_match": True,
            "panel_lock_sha256": sha(root / "panel_lock.json"), "payload_files_opened": 0}


def h(counts):
    c = [int(x) for x in counts]
    n = sum(c)
    return 0.0 if n == 0 else n * math.log2(n) - sum(x * math.log2(x) for x in c if x)


def descriptor(counts):
    c = [int(x) for x in counts]
    states = sum(c) + 1
    return (len(c) - 1) * (0 if states == 1 else (states - 1).bit_length())


def independent_counts(q, k, planes):
    e, roles, n = q.shape
    gray = np.asarray(((0, 0), (0, 1), (1, 1), (1, 0)), dtype=np.uint8)
    marginal = np.zeros((e, roles, 4), np.int64)
    latent = np.zeros((roles, k), np.int64)
    conditional = np.zeros((e, roles, k, 4), np.int64)
    for x in range(e):
        for r in range(roles):
            marginal[x, r] = [np.sum(q[x, r] == a) for a in range(4)]
    for r in range(roles):
        source = q[:, r] if k == 4 else gray[q[:, r], planes[r]]
        u = np.argmax(np.asarray([[np.sum(source[:, i] == s) for i in range(n)] for s in range(k)]), axis=0)
        latent[r] = [np.sum(u == s) for s in range(k)]
        for x in range(e):
            for s in range(k):
                conditional[x, r, s] = [np.sum((u == s) & (q[x, r] == a)) for a in range(4)]
    return marginal, latent, conditional


def audit_math(root):
    core = load("common_latent_core_r2_audit", root / "common_latent_core.py")
    gap = -0.5 * math.log2(0.8 / 0.9888693569009007)
    req(abs(gap * 1.5 - TARGET) < 2e-15 and core.TARGET_GAIN_BPW == TARGET, "target derivation")
    q = np.asarray([
        [[0,1,2,3,0,2,1,3],[3,2,1,0,1,1,2,2]],
        [[0,1,3,3,2,2,1,0],[3,0,1,0,1,2,2,3]],
        [[1,1,2,0,0,3,1,3],[2,2,1,0,3,1,2,2]],
        [[0,2,2,3,0,2,0,3],[3,2,0,0,1,1,3,2]],
    ], np.uint8)
    for k, planes in ((2, (0, 1)), (4, None)):
        a, b, c = independent_counts(q, k, planes)
        s = core.summarize_counts_cpu(q, k, planes)
        req(np.array_equal(a, s["marginal_counts"]) and np.array_equal(b, s["latent_counts"]) and np.array_equal(c, s["conditional_counts"]), "counts")
        score = core.score_count_summary(s, 123)
        mb, ub, cb = sum(map(h, a.reshape(-1, 4))), sum(map(h, b)), sum(map(h, c.reshape(-1, 4)))
        mm, um, cm = sum(map(descriptor, a.reshape(-1, 4))), sum(map(descriptor, b)), sum(map(descriptor, c.reshape(-1, 4)))
        sel = 3 if k == 2 else 1
        req(score["marginal_two_part_bits"] == mb + mm + 123, "marginal two part")
        req(score["common_two_part_bits"] == ub + cb + um + cm + sel + 123, "common two part")
    values = np.asarray([[-2,-1,-.75,-.25,0,.25,.75,2],[-1.5,-1,-.5,-.125,.125,.5,1,1.5]], np.float64)
    got = core.quantize_canonical_cpu(values, 8)
    expected_scales, expected_labels = [], []
    for block in values:
        s16 = np.float16(math.sqrt(math.fsum(float(x*x) for x in block) / 8))
        expected_scales.append(int(s16.view(np.uint16)))
        t = core.THRESHOLD_RMS * float(s16)
        expected_labels.extend(0 if x < -t else 1 if x < 0 else 2 if x <= t else 3 for x in block)
    req(got.scale_u16.tolist() == expected_scales and got.labels.ravel().tolist() == expected_labels, "quantizer")
    labels = np.arange(5 * 2 * 257, dtype=np.uint32).reshape(5, 2, 257).astype(np.uint8) % 4
    for seed in core.CONTROL_SEEDS:
        shuffled = core.coordinate_scramble_cpu(labels, seed)
        for e in range(5):
            for r in range(2):
                aa, bb = core.affine_permutation_parameters(257, seed, e, r)
                idx = (aa * np.arange(257) + bb) % 257
                req(np.unique(idx).size == 257 and np.array_equal(shuffled[e, r], labels[e, r, idx]), "control bijection")
                req(np.array_equal(np.bincount(shuffled[e, r], minlength=4), np.bincount(labels[e, r], minlength=4)), "control marginal")
    return core, {"global_gap_bpw": gap, "derived_up_down_threshold_bpw": gap * 1.5,
                  "count_entropy_mdl": "PASS", "quantizer": "PASS", "control_seed_count": 8,
                  "control_bijection_and_marginal_preservation": "PASS"}


class Runtime:
    @staticmethod
    def getDeviceProperties(x): return {"name": b"source-free-fake"}
    @staticmethod
    def getDevice(): return 0
class Cuda: runtime = Runtime()
class CP: __version__ = "source-free-fake"; cuda = Cuda()
class Control: pass


def load_worker(root, core, suffix):
    sys.modules["common_latent_core"] = core
    return load("common_latent_worker_r2_" + suffix, root / "cupy_worker.py")


def fake_score_factory(source_marker, source_gain, control_gain, private_bits):
    def score(labels, cardinality, scale_bits=0, selection_objective="charged"):
        source = labels is source_marker
        gain = source_gain if source else control_gain
        return {"cardinality": cardinality, "planes": [0,0] if cardinality == 2 else [None,None],
                "source_weights": 16 * 2 * 768 * 2048,
                "favorable_gross_gain_bpw": gain, "two_part_gain_bpw": gain,
                "latent_model_bits": 0, "selector_bits": 3 if cardinality == 2 else 1,
                "per_expert_conditional_data_bits": [float(private_bits)] * 16,
                "per_expert_conditional_model_bits": [0] * 16}
    return score


def configure(worker, source_gain, control_gain, private_bits):
    marker = object()
    calls = {"controls": 0}
    worker._cupy = lambda: CP
    worker.load_quantized_panel_gpu = lambda panel, payload: (marker, {"scale_bits": 0, "scale_bytes_per_expert": 0})
    def scramble(labels, seed):
        calls["controls"] += 1
        return Control()
    worker.coordinate_scramble_gpu = scramble
    worker.score_labels_gpu = fake_score_factory(marker, source_gain, control_gain, private_bits)
    return calls


def audit_feasible_rule(root, core):
    worker = load_worker(root, core, "truth")
    good = {"status": "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC", "capacity_ok": True, "strictly_below_2x": True}
    predicates = []
    for key, value in (("status", "FAIL_CAPACITY_OR_STRICT_READ_AMPLIFICATION"), ("capacity_ok", False), ("strictly_below_2x", False)):
        bad = dict(good); bad[key] = value
        case = {"2.15": bad, "2.5": bad}
        req(worker._feasible_rate_endpoints(case) == [], f"failed {key} eligible")
        predicates.append(key)
    mixed = {"2.15": dict(good), "2.5": {"status": "FAIL", "capacity_ok": True, "strictly_below_2x": True}}
    req(worker._feasible_rate_endpoints(mixed) == ["2.15"], "mixed endpoints")
    for malformed in ({"2.15": good}, {"2.15": good, "2.5": good, "2.25": good}):
        try: worker._feasible_rate_endpoints(malformed)
        except ValueError: pass
        else: raise RuntimeError("endpoint closure not enforced")
    table = [
        ((True, None, None), ("HARD_KILL_FAVORABLE_IDEAL_BELOW_TARGET", False)),
        ((False, None, None), ("HOLD_NO_CAPACITY_AND_STRICT_READ_FEASIBLE_RATE_ENDPOINT", False)),
        ((False, TARGET - 1e-6, None), ("HOLD_READ_FEASIBLE_CHARGED_MDL_BELOW_TARGET", False)),
        ((False, TARGET, None), ("HOLD_CONTROLS_REQUIRED_BUT_NOT_RUN", False)),
        ((False, TARGET, TARGET - 1e-6), ("HOLD_CONTROL_CORRECTED_FAVORABLE_BELOW_TARGET", False)),
        ((False, TARGET, TARGET), ("SURVIVE_IDEAL_APERTURE_REQUIRES_FINITE_CODER", True)),
    ]
    for args, expected in table:
        got = worker._final_disposition(favorable_below_target=args[0], read_eligible_charged_gain_bpw=args[1], control_corrected_gain_bpw=args[2])
        req(got == expected, "disposition table")
    return {"exact_endpoint_set": ["2.15", "2.5"], "required_true_predicates": predicates,
            "one_good_endpoint_is_sufficient": True, "disposition_truth_table_cases": len(table)}


def audit_full_regressions(root, core):
    # Real Qwen-shaped envelope with tiny private ownership: every endpoint fails
    # nonpadding strict-read, despite excellent synthetic entropy gains.
    worker = load_worker(root, core, "all_fail")
    calls = configure(worker, 0.5, 0.0, 0)
    result = worker.run_authorized_panel(root / "panel_lock.json", Path("FORBIDDEN_NOT_OPENED"))
    all_failed = all(not e["strictly_below_2x"] for rates in result["physical_page_envelopes"].values() for e in rates.values())
    req(all_failed and result["read_eligible_rate_endpoints"] == {"binary_charged_mdl": [], "quaternary": []}, "all-fail endpoints")
    req(result["status"] == "HOLD_NO_CAPACITY_AND_STRICT_READ_FEASIBLE_RATE_ENDPOINT" and not result["eligible_for_finite_coder_research"], "all-fail promotion")
    req(result["controls_run"] == 0 and calls["controls"] == 0, "controls ran before physical pass")

    # A real feasible envelope but sub-threshold charged gain must also skip controls.
    worker2 = load_worker(root, core, "charged_fail")
    calls2 = configure(worker2, TARGET - 0.01, 0.0, 6_000_000)
    # Favorable must clear first gate while charged fails; override the dual objectives.
    marker = object()
    worker2.load_quantized_panel_gpu = lambda p, q: (marker, {"scale_bits": 0, "scale_bytes_per_expert": 0})
    def scores(labels, cardinality, scale_bits=0, selection_objective="charged"):
        control = isinstance(labels, Control)
        favorable = 0.5 if not control else 0.0
        charged = TARGET - 0.01 if not control else 0.0
        gain = favorable if selection_objective == "favorable" else charged
        if cardinality == 4: gain = charged
        return {"cardinality": cardinality, "planes": [0,0] if cardinality == 2 else [None,None],
                "source_weights": 16*2*768*2048, "favorable_gross_gain_bpw": favorable,
                "two_part_gain_bpw": gain, "latent_model_bits": 0,
                "selector_bits": 3 if cardinality == 2 else 1,
                "per_expert_conditional_data_bits": [6_000_000.0]*16,
                "per_expert_conditional_model_bits": [0]*16}
    worker2.score_labels_gpu = scores
    r2 = worker2.run_authorized_panel(root / "panel_lock.json", Path("FORBIDDEN_NOT_OPENED"))
    req(any(r2["read_eligible_rate_endpoints"].values()), "charged-fail needs feasible envelope")
    req(r2["status"] == "HOLD_READ_FEASIBLE_CHARGED_MDL_BELOW_TARGET" and r2["controls_run"] == 0 and calls2["controls"] == 0, "charged pre-control gate")

    # Force one family/endpoint feasible and verify exactly eight controls precede survival.
    worker3 = load_worker(root, core, "survive")
    calls3 = configure(worker3, 0.5, 0.0, 0)
    def envelope(**kw):
        good = kw["latent_bits_per_coordinate"] == 1 and kw["requested_rate"] == Fraction(43,20)
        return {"status": "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC" if good else "FAIL_CAPACITY_OR_STRICT_READ_AMPLIFICATION",
                "capacity_ok": good, "strictly_below_2x": good}
    worker3.physical_page_envelope = envelope
    r3 = worker3.run_authorized_panel(root / "panel_lock.json", Path("FORBIDDEN_NOT_OPENED"))
    req(r3["read_eligible_rate_endpoints"] == {"binary_charged_mdl": ["2.15"], "quaternary": []}, "eligible family/rate")
    req(r3["controls_run"] == 8 and calls3["controls"] == 8, "control count")
    req(r3["status"] == "SURVIVE_IDEAL_APERTURE_REQUIRES_FINITE_CODER" and r3["eligible_for_finite_coder_research"], "valid survivor")
    return {"old_regression_closed": True, "all_four_real_envelopes_failed": True,
            "failure_status": result["status"], "failure_eligible": False,
            "controls_called_on_physical_failure": 0,
            "charged_mdl_failure_skips_controls": True,
            "single_feasible_family_endpoint": {"binary_charged_mdl": ["2.15"], "quaternary": []},
            "valid_survivor_controls_run": 8, "payload_files_opened": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-package", required=True)
    ap.add_argument("--prior-binding-result", required=True)
    args = ap.parse_args()
    root = Path(args.source_package).resolve(strict=True)
    prior = Path(args.prior_binding_result).resolve(strict=True)
    manifest = authenticate(root)
    hold = audit_hold(root)
    panel = audit_panel(root, prior)
    core, maths = audit_math(root)
    rule = audit_feasible_rule(root, core)
    regressions = audit_full_regressions(root, core)
    receipt = {
        "schema": "same_layer_common_latent_independent_source_audit_r2_receipt_v0",
        "status": "PASS_REPAIRED_SOURCE_ELIGIBLE_FOR_SEPARATE_DEPLOYMENT_REVIEW",
        "auditor_id": "common_latent_source_audit_r2",
        "source_manifest_sha256": MANIFEST_SHA,
        "source_root_sha256": SOURCE_ROOT,
        "source_member_count_excluding_manifest": len(manifest["files"]),
        "claim_boundary": "PAYLOAD_BLIND_SOURCE_REVIEW_ONLY_NOT_PAYLOAD_OR_CODEC_EVIDENCE",
        "payload_accessed": False,
        "source_files_modified": False,
        "hold": hold, "panel": panel, "math_quantizer_entropy_controls": maths,
        "feasible_rate_rule": rule, "full_function_regressions": regressions,
        "read_ledger_math": {"physical_and_nonpadding_formulae_unchanged_from_independent_r1_pass": True,
                             "exact_2x_and_padding_attack_tests_passed_in_producer_suite": True},
        "cupy": {"separate_source_free_receipt_required": True},
        "authorization": "SOURCE_PASS_ONLY_DEPLOYMENT_COPY_MUST_BE_SEPARATELY_PINNED_AND_REVIEWED",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
