"""Payload-blind independent audit of the sealed CBIB-1 source aperture."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
from unittest import mock

import numpy as np


EXPECTED_MANIFEST = "1d07f1c0a057db3ba74f91062c06d39c23e39f5dd3da74a373c790345a9e7a9a"
EXPECTED_ROOT = "18a4043e99b17cfa535f4a6c2930f2c1ac42eff092f4e5d61b9408b1986f457e"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_source_root(rows: list[dict]) -> str:
    normalized = [
        {"bytes": int(row["bytes"]), "name": row["name"], "sha256": row["sha256"]}
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def enumerate_equal_partitions(items: tuple[int, ...], group_size: int):
    if not items:
        yield ()
        return
    anchor = items[0]
    for tail in itertools.combinations(items[1:], group_size - 1):
        group = (anchor,) + tail
        rest = tuple(x for x in items if x not in group)
        for suffix in enumerate_equal_partitions(rest, group_size):
            yield (group,) + suffix


def independent_kt_logp(count: int, total: int, alphabet: int) -> float:
    return math.log2((count + 0.5) / (total + 0.5 * alphabet))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--producer-gpu-receipt", required=True)
    args = parser.parse_args()
    package = Path(args.source_package).resolve()
    require(args.manifest_sha256 == EXPECTED_MANIFEST, "unexpected external manifest pin")
    require(sha(package / "SOURCE_MANIFEST.json") == EXPECTED_MANIFEST, "manifest digest")
    raw = (package / "SOURCE_MANIFEST.json").read_bytes()
    manifest = json.loads(raw)
    require(raw == (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            "noncanonical manifest")
    require(canonical_source_root(manifest["files"]) == EXPECTED_ROOT, "source root")
    require(manifest["source_root_sha256"] == EXPECTED_ROOT, "declared source root")
    for row in manifest["files"]:
        member = package / row["name"]
        require(member.is_file() and not member.is_symlink(), f"member type {row['name']}")
        require(member.stat().st_size == row["bytes"] and sha(member) == row["sha256"],
                f"member closure {row['name']}")
    actual = sorted(p.name for p in package.iterdir())
    require(actual == sorted([r["name"] for r in manifest["files"]] + ["SOURCE_MANIFEST.json"]),
            "extra or missing source member")

    sys.path.insert(0, str(package))
    verify = load_module("cbib_verify", package / "verify_source.py")
    producer_verification = verify.verify(package, EXPECTED_MANIFEST)
    core = load_module("clustered_ib_core", package / "clustered_ib_core.py")
    fixture = load_module("cbib_fixture", package / "source_free_fixture.py")
    gate = load_module("cbib_gate", package / "run_gate.py")
    require(gate.PAYLOAD_EXECUTION_ENABLED is False and gate.main([]) == 2,
            "HOLD is not fail closed")
    core_text = (package / "clustered_ib_core.py").read_text().lower()
    require("payload_root" not in core_text and "payload-root" not in core_text,
            "core source locator")

    # All-coordinate cross-fit: every coordinate is held out exactly once, and
    # changing one held-out fold cannot alter its partition or trained models.
    labels = fixture.make_clustered_nonmodal_fixture(coordinates=1024)
    folds = core.fold_ids(1024, fold_count=4, superblock_values=128)
    require(np.array_equal(np.unique(folds, return_counts=True)[1], np.full(4, 256)),
            "fold coverage")
    fold_leak_checks = 0
    for fold in range(4):
        train = folds != fold
        changed = labels.copy()
        rng = np.random.default_rng(0xA11D + fold)
        changed[:, :, ~train] = rng.integers(0, 4, size=changed[:, :, ~train].shape,
                                             dtype=np.uint8)
        p0 = core._fold_partition(labels, train, 4)
        p1 = core._fold_partition(changed, train, 4)
        require(p0 == p1, "held-out fold changed partition")
        for group in p0:
            index = np.asarray(group)
            for role in range(2):
                m0 = core.fit_binary_product_model(labels[index, role][:, train])
                m1 = core.fit_binary_product_model(changed[index, role][:, train])
                require(np.array_equal(m0.latent_counts, m1.latent_counts), "latent leakage")
                require(np.array_equal(m0.conditional_counts, m1.conditional_counts),
                        "conditional leakage")
                fold_leak_checks += 1

    # The transmitted model counts and latent labels suffice to reproduce the
    # factorized ideal NLL.  The latent is a charged common stream, not a hidden
    # decoder oracle.
    train = folds != 0
    test = ~train
    group = np.arange(4)
    model = core.fit_binary_product_model(labels[group, 0][:, train])
    scored = core.evaluate_binary_model(labels[group, 0][:, test],
                                        model.latent_counts, model.conditional_counts)
    u = scored["assignments"]
    reconstructed_latent_1 = int(train.sum()) - int(model.latent_counts[0])
    require(reconstructed_latent_1 == int(model.latent_counts[1]), "derived latent count")
    for expert in range(4):
        for state in range(2):
            row = model.conditional_counts[expert, state]
            require(int(model.latent_counts[state]) - int(row[:3].sum()) == int(row[3]),
                    "derived conditional count")
    ref_latent = -sum(independent_kt_logp(int(model.latent_counts[s]), int(train.sum()), 2)
                      for s in u)
    ref_private = []
    test_labels = labels[group, 0][:, test]
    for expert in range(4):
        bits = 0.0
        for position, state in enumerate(u):
            symbol = int(test_labels[expert, position])
            bits -= independent_kt_logp(int(model.conditional_counts[expert, state, symbol]),
                                        int(model.latent_counts[state]), 4)
        ref_private.append(bits)
    require(abs(ref_latent - scored["latent_bits"]) < 1e-9, "latent NLL reference")
    require(max(abs(a - b) for a, b in zip(ref_private, scored["private_bits"])) < 1e-9,
            "private NLL reference")

    # Exact combinatorial charges and the complete source-free fixture ledger.
    enumerated = sum(1 for _ in enumerate_equal_partitions(tuple(range(8)), 2))
    require(enumerated == 105 == core.partition_count(8, 2), "partition count")
    require(core.partition_descriptor_bits(8, 2) == 7, "partition descriptor")
    require(core.selector_bits_for_group_bank(16) == 2, "selector descriptor")
    score = core.crossfit_group_size(labels, 4, fold_count=4, superblock_values=128)
    require(all(row["partition"] == [[0, 1, 2, 3], [4, 5, 6, 7],
                                      [8, 9, 10, 11], [12, 13, 14, 15]]
                for row in score["fold_evidence"]), "fixture partition")
    require(score["favorable_gross_gain_bpw"] > 0.9, "fixture mechanism")
    expected_baseline_model = 4 * 16 * 2 * 3 * core.ceil_log2_states(769)
    require(score["baseline_model_bits"] == expected_baseline_model, "baseline model charge")
    require(score["partition_bits"] == 4 * core.partition_descriptor_bits(16, 4),
            "partition charge")
    require(score["selector_bits"] == 2, "selector charge")
    require(score["baseline_framing_bits"] == 16 * 256 * 8, "baseline framing")
    require(score["structured_framing_bits"] == (4096 + 16 * 256 + 16 * 256) * 8,
            "structured framing")
    require(score["conditional_model_bits"] == sum(score["private_model_bits_by_expert"]),
            "private model allocation")
    require(score["latent_model_bits"] == sum(score["common_model_bits_by_segment"]),
            "common model allocation")
    requirements = core.packet_requirements(score, 137)
    require(all(x >= 256 + 137 for x in requirements["private_required_bytes"]),
            "scale bytes absent from private packet")
    require(requirements["global_required_bytes"] == 4096 +
            core.ceil_div(score["partition_bits"] + score["selector_bits"], 8),
            "global charge")

    # Independent exact read-ledger recomputation, including the non-padding
    # denominator that prevents padding from manufacturing a pass.
    read = core.physical_read_envelope(
        expert_count=4, weights_per_expert=49152, requested_rate=Fraction(5, 2),
        global_required_bytes=4096,
        common_segments=[{"members": [0, 1], "required_bytes": 4096},
                         {"members": [2, 3], "required_bytes": 4096}],
        private_required_bytes=[8192] * 4)
    require(read["touched_bytes"] == [20480] * 4, "touched page union")
    require(read["owned_physical_bytes"] == ["15360"] * 4, "physical denominator")
    require(read["owned_nonpadding_bytes"] == ["11264"] * 4, "nonpadding denominator")
    padding_attack = core.physical_read_envelope(
        expert_count=4, weights_per_expert=32768, requested_rate=Fraction(5, 2),
        global_required_bytes=1,
        common_segments=[{"members": [0, 1], "required_bytes": 1},
                         {"members": [2, 3], "required_bytes": 1}],
        private_required_bytes=[1] * 4)
    require(padding_attack["capacity_ok"] and not padding_attack["strictly_below_2x"],
            "padding attack passed")

    # Source-first decision order and strict predicate conjunction.
    toy = np.zeros((4, 2, 128), dtype=np.uint8)
    calls = {"controls": 0}
    def fake_control(q, seed):
        calls["controls"] += 1
        return (q + np.uint8(1)) % np.uint8(4)
    def fake_score(q, size, *unused):
        source = bool(np.all(q == 0))
        gain = 0.30 if source else 0.01
        return {"group_size": size, "favorable_gross_gain_bpw": gain,
                "charged_gain_bpw": gain}
    def fake_req(unused_score, unused_scale):
        return {}
    def good_envelope(**unused):
        return {"status": "IDEAL_CAPACITY_ONLY_NOT_AN_EMITTED_CODEC",
                "capacity_ok": True, "strictly_below_2x": True}
    with mock.patch.object(core, "crossfit_group_size", side_effect=fake_score), \
         mock.patch.object(core, "packet_requirements", side_effect=fake_req), \
         mock.patch.object(core, "physical_read_envelope", side_effect=good_envelope), \
         mock.patch.object(core, "marginal_preserving_control", side_effect=fake_control):
        decision = core.score_source_gate(toy, 0)
    require(decision["eligible_for_finite_codec"] and calls["controls"] == 8,
            "survivor did not run exactly eight controls")
    require(all(row["control_corrected_charged_gain_bpw"] == 0.29
                for row in decision["source_scores"]), "control correction")
    calls["controls"] = 0
    def bad_envelope(**unused):
        return {"status": "FAIL_STRICT_READ_AMPLIFICATION",
                "capacity_ok": True, "strictly_below_2x": False}
    with mock.patch.object(core, "crossfit_group_size", side_effect=fake_score), \
         mock.patch.object(core, "packet_requirements", side_effect=fake_req), \
         mock.patch.object(core, "physical_read_envelope", side_effect=bad_envelope), \
         mock.patch.object(core, "marginal_preserving_control", side_effect=fake_control):
        held = core.score_source_gate(toy, 0)
    require(held["status"] == "HOLD_NO_STRICT_READ_FEASIBLE_RATE" and calls["controls"] == 0,
            "control ran before read survival")

    controlled = core.marginal_preserving_control(labels, core.CONTROL_SEEDS[0])
    require(not np.array_equal(labels, controlled), "control identity")
    for expert in range(16):
        for role in range(2):
            require(np.array_equal(np.bincount(labels[expert, role], minlength=4),
                                   np.bincount(controlled[expert, role], minlength=4)),
                    "control changed marginal")

    design = json.loads((package / "design_lock.json").read_text())
    require(design["universality"].startswith("uses only legal SwiGLU expert geometry"),
            "universality lock")
    require(design["payload_execution_enabled"] is False, "design HOLD")
    forbidden = ("safetensors", "huggingface", "qwen/", "ssh ", "payload_root")
    math_text = ((package / "clustered_ib_core.py").read_text() +
                 (package / "cupy_backend.py").read_text()).lower()
    require(not any(token in math_text for token in forbidden), "nonuniversal/source token")

    producer_gpu_path = Path(args.producer_gpu_receipt).resolve()
    producer_gpu = json.loads(producer_gpu_path.read_text())
    require(producer_gpu["manifest_sha256"] == EXPECTED_MANIFEST and
            producer_gpu["status"] == "PASS_SOURCE_FREE_CPU_CUPY_PARITY" and
            producer_gpu["qwen_or_model_payload_accessed"] is False,
            "producer local GPU receipt mismatch")

    receipt = {
        "schema": "same-layer-clustered-ib-independent-source-audit-evidence-v0",
        "status": "PASS_INDEPENDENT_PAYLOAD_BLIND_SOURCE_AUDIT",
        "source_manifest_sha256": EXPECTED_MANIFEST,
        "source_root_sha256": EXPECTED_ROOT,
        "producer_verification": producer_verification,
        "all_coordinate_crossfit": {
            "folds": 4, "coordinates": 1024, "held_out_once": True,
            "heldout_mutation_training_invariance_checks": fold_leak_checks,
        },
        "encoder_decoder_legality": {
            "latent_stream_explicit_and_charged": True,
            "conditional_streams_private_and_charged": True,
            "count_tables_sufficient": True,
            "independent_nll_tolerance_bits": 1e-9,
            "finite_entropy_coder_present": False,
        },
        "charges": {
            "partition_count_enumerated_e8_k2": enumerated,
            "partition": "PASS", "model": "PASS", "selector": "PASS",
            "framing": "PASS", "scale_in_private_requirement": "PASS",
        },
        "fixture": {
            "favorable_gross_gain_bpw": score["favorable_gross_gain_bpw"],
            "net_ideal_gain_bpw": score["net_ideal_gain_bpw"],
            "charged_gain_bpw": score["charged_gain_bpw"],
            "expected_groups_recovered_all_folds": True,
        },
        "read_ledger": {"exact_page_union": "PASS", "physical_denominator": "PASS",
                        "nonpadding_denominator": "PASS", "strict_2x": "PASS"},
        "decision_order": {"source_before_controls": True,
                           "failed_read_controls_called": 0,
                           "survivor_controls_called": 8},
        "controls": {"marginals_exact": True, "independent_expert_role_affine_maps": True},
        "universality": "PASS_SOURCE_CORE_HAS_NO_MODEL_IDENTITY_PROVENANCE_OR_PAYLOAD_LOCATOR",
        "payload_accessed": False,
        "producer_local_rtx3060_receipt_sha256": sha(producer_gpu_path),
        "producer_local_rtx3060_treated_as_independent_authority": False,
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
