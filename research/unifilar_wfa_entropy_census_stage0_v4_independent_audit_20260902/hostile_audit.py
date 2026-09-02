#!/usr/bin/env python3
"""Independent source-only hostile audit for the pre-review UWFA-SC v4 tree.

The exact producer inventory is authenticated into immutable byte snapshots
before any producer module is compiled.  This tool never opens a model/Qwen,
current-codec, extracted-stream, or matched-control payload and never imports
CuPy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PACKAGE_NAME = "unifilar_wfa_entropy_census_stage0_v4"
ROOT_PREFIX = b"UWFA-SC-V4-INDEPENDENT-PRE-REVIEW-ROOT-v1\n"
EXPECTED_ROOT = "57d7c99e616da8d56dabb7fedab75fb8a9dbf940008762d1e43ae452ad4356c6"
EXPECTED_INVENTORY = (
    ("container_codec.py", 81120, "2a4a1bb61d9c62ce54942a132b4a1b0442fb72bcebca2bec59f2059c4355840c"),
    ("cupy_backend.py", 40964, "7904a5e122686487d89fb684b70052507089bfe3bbfe4f1f02520df6ce3fb1ba"),
    ("design_lock.json", 8666, "ac71007e2903ee29d66c7af6f7cb730a25193c26335b02591ef34439f8866481"),
    ("dispatcher_contract.py", 9205, "3092f2e4f6272cca3d3e627cdf1165f4f99a965d2e6bbbfe10755bacc72e1961"),
    ("fixture_long_memory.py", 4307, "3fa05fe68b4e05a9933d45a846949f1a9e48690d19e77c3a232f0d46a082084b"),
    ("fixture_portability.py", 16350, "d8eaa1453413f967a4a0ec9ed3cdc881296e1b2bf8eb709b96909ec59f0ccded"),
    ("INDEPENDENT_BOOTSTRAP_ABI.md", 7553, "5444db5b944f34023c47c2742db3c846b7ea135e1acfec009a59fa63548c05d8"),
    ("protocol.py", 19389, "58f30c52fccdc206e97fa5a34cc6e5c4921898b3a45e0043d434ddcdd6d2c7b2"),
    ("README.md", 10939, "dda417127ac605e8b7f045099d09d7da2359c7789fb498716cf502ca65112686"),
    ("result_envelope.py", 2688, "b12fa69fd22b3e6b06b20415b3e43b689d7fd4eaf0bcd88dc64328280362ceca"),
    ("run_source_free_gpu_dev.py", 7125, "d7ff362289553633d34c2a2dde025bc497e994220f86b25e9668bd1766379c18"),
    ("stage0_census.py", 78161, "8c4e4c5d1f0c0c13d606178e68a21c6ab904dad9fc09d87209281d4653d605a7"),
    ("strata_sc_adapter.py", 36184, "e021a33bd21cbb0256decdebf046ded4dbf5e28df82817221b44d0b8d324f4cc"),
    ("test_source_only.py", 79809, "a7adadcbc843d01b89139bb95e76f3ba56fbdb162faa1f6130c4cb76cbf1504f"),
    ("universal_adapter.py", 11577, "b55bdf1eab31ab3ef270560efa769f3ae8c951c09ec564c165296186ce327b33"),
    ("uwfa_common.py", 38771, "ab445e55c763654f46ad588289770b9b132081e7df30fc056927f21cec091295"),
    ("verify_source.py", 11140, "11fd66a22f4dfd0549690c5cdec625bcdbbe7dcb2afb488d420aea92cd95a686"),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_package() -> Path:
    configured = os.environ.get("UWFA_V4_AUDIT_PACKAGE")
    if configured:
        return Path(configured).absolute()
    here = Path(__file__).absolute()
    for parent in here.parents:
        candidate = parent / "research" / PACKAGE_NAME
        if candidate.is_dir():
            return candidate
    raise RuntimeError("producer package not found")


def authenticate() -> tuple[Path, dict[str, bytes], str]:
    package = find_package()
    info = os.lstat(package)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError("producer path must be a no-follow directory")
    expected = {name: (size, digest) for name, size, digest in EXPECTED_INVENTORY}
    actual = sorted(entry.name for entry in os.scandir(package))
    if actual != sorted(expected):
        raise RuntimeError(f"producer inventory drift: {sorted(set(actual) ^ set(expected))}")
    snapshots: dict[str, bytes] = {}
    records: list[str] = []
    for name in sorted(expected, key=str.lower):
        size, digest = expected[name]
        path = package / name
        meta = os.lstat(path)
        if stat.S_ISLNK(meta.st_mode) or not stat.S_ISREG(meta.st_mode):
            raise RuntimeError(f"nonregular producer member: {name}")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            before = os.fstat(fd)
            chunks = []
            while chunk := os.read(fd, 1 << 20):
                chunks.append(chunk)
            data = b"".join(chunks)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns)
        if identity(before) != identity(after) or len(data) != size or sha(data) != digest:
            raise RuntimeError(f"producer member drift: {name}")
        snapshots[name] = data
        records.append(f"{name}\t{size}\t{digest}\n")
    root = sha(ROOT_PREFIX + "".join(records).encode("utf-8"))
    if root != EXPECTED_ROOT:
        raise RuntimeError("producer root mismatch")
    return package, snapshots, root


PACKAGE, SNAPSHOTS, AUTHENTICATED_ROOT = authenticate()
sys.dont_write_bytecode = True


def load_snapshot(label: str, filename: str) -> types.ModuleType:
    data = SNAPSHOTS[filename]
    expected = {row[0]: row[2] for row in EXPECTED_INVENTORY}[filename]
    if sha(data) != expected:
        raise RuntimeError(f"snapshot drift before compile: {filename}")
    module = types.ModuleType(label)
    module.__file__ = f"<authenticated-uwfa-v4:{filename}>"
    module.__package__ = ""
    sys.modules[label] = module
    exec(compile(data, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


common = load_snapshot("uwfa_v4_audit_common", "uwfa_common.py")
protocol = load_snapshot("uwfa_v4_audit_protocol", "protocol.py")
semantic = load_snapshot("uwfa_v4_audit_semantic", "universal_adapter.py")
codec = load_snapshot("uwfa_v4_audit_codec", "container_codec.py")
fixture = load_snapshot("uwfa_v4_audit_fixture", "fixture_portability.py")
stage = load_snapshot("uwfa_v4_audit_stage", "stage0_census.py")
cuda_source = load_snapshot("uwfa_v4_audit_cuda_source", "cupy_backend.py")
dispatcher = load_snapshot("uwfa_v4_audit_dispatcher", "dispatcher_contract.py")
envelope = load_snapshot("uwfa_v4_audit_envelope", "result_envelope.py")


BLOCKERS: dict[str, str] = {}
OBSERVATIONS: dict[str, object] = {}


def binding_hashes() -> dict[str, str]:
    return {name: sha(("binding:" + name).encode("ascii")) for name in codec._HEADER_BINDINGS}


def evidence(*, full: str, structural: str, pipeline: str, score_bytes: bytes,
             decoder: str = "12" * 32, root: str = "18" * 32,
             preflight: str = "19" * 32, salt: str = "") -> object:
    def d(name: str) -> str:
        return sha((salt + name).encode("ascii"))
    return stage.BoundEvidence(
        baseline_plan_sha256=d("plan"), baseline_score_sha256=sha(score_bytes),
        universal_decoder_sha256=decoder, producer_manifest_sha256=d("manifest"),
        audit_bootstrap_sha256=d("bootstrap"), source_full_geometry_sha256=full,
        source_structural_geometry_sha256=structural, extraction_program_sha256=d("extract"),
        universal_adapter_sha256=d("adapter"), pipeline_sha256=pipeline,
        source_snapshot_root_sha256=root, source_preflight_receipt_sha256=preflight,
    )


def scientific_panel() -> dict[str, object]:
    rows = []
    for expert in range(6):
        for role_index, role in enumerate(("gate", "up", "down")):
            ordinal = 3 * expert + role_index
            owner = protocol.owner_set_from_ordinals(6, [expert])
            length = 7 + ordinal
            bits = bytes((i + ordinal) & 1 for i in range(length))
            levels = bytes((i + ordinal) % common.LEVELS for i in range(length))
            base = [32768] * length
            rows.append({
                "stream_ordinal": ordinal, "owner_set_hex": owner.hex(), "owner_set": owner,
                "owner_contributions": ({"expert": expert, "role": role, "source_offset": 0, "weight_count": 1},),
                "owner_expert_ordinals": [expert], "owner_identity_indices": [expert],
                "owner_weight_contributions": {expert: 1}, "weight_charge": 1,
                "shape_rows": 1, "shape_cols": 1, "role": role, "symbols": length,
                "bits": list(bits), "levels": list(levels), "base": base,
                "bits_bytes": bits, "levels_bytes": levels,
                "base_bytes": struct.pack(f"<{length}H", *base),
                "baseline_payload_bytes": (length + 7) // 8 + 2,
                "baseline_logical_bits": length, "profile_q": 0, "decoder_scale": 1.0, "logn": 1,
            })
    return {
        "streams": rows, "weights": 18, "experts": 6,
        "artifact": {"raw_bytes": 100, "raw_sha256": "ab" * 32},
        "immutable_state": b"", "semantic_identities": [(0, i) for i in range(6)],
        "expert_shapes": [{"expert": i, "hidden": 1, "intermediate": 1} for i in range(6)],
        "reconstruction": {"full_reconstruction_f64_sha256": "44" * 32},
    }


def preflight_with_duplicate_cells() -> tuple[object, object]:
    root = "18" * 32
    uuid = "GPU-c06e0fe0-9836-2f98-8f10-0514d085f722"
    pci = "00000000:16:00.0"
    environment = {
        "cupy_version": "13.6.0", "cuda_runtime_version": 12090,
        "cuda_driver_version": 12090, "python_version": "3.12", "platform": "Linux",
        "device_id": 0, "device_name": "NVIDIA GeForce RTX 5090", "device_uuid": uuid,
        "pci_bus_id": pci, "compute_capability": [12, 0], "current_free_vram_bytes": 1,
        "total_vram_bytes": 32 * (1 << 30),
        "statistics": {"telemetry_samples": 1, "peak_process_tree_rss_bytes": 1,
                       "total_vram_bytes": 32 * (1 << 30)},
        "telemetry_samples": [{"phase": "counterfeit"}], "host_byteorder": "little",
        "explicit_device_synchronization_at_phase_boundaries_and_after_every_kernel": True,
        "fatal_telemetry_sampling": True, "transfer_formula": {},
    }
    identity = {
        "schema": "uwfa-sc-v4-independent-gpu-identity",
        "status": "PASS_INDEPENDENT_GPU_IDENTITY", "device_uuid": uuid,
        "pci_bus_id": pci, "device_name": "NVIDIA GeForce RTX 5090", "provider": "nvidia-smi",
    }
    identity["identity_receipt_sha256"] = sha(common.canonical_json(identity))
    # Deliberately 150 copies of one selector and no arithmetic/candidate receipt.
    all150 = {
        "schema": "uwfa-sc-v4-all150-source-free-preflight",
        "status": "PASS_ALL_150_CPU_CUPY_EXACT_REPEATED", "source_snapshot_root_sha256": root,
        "cell_count": 150,
        "cells": [{"selector_ordinal": 0, "repeated_gpu_run_exact": True} for _ in range(150)],
        "environment": copy.deepcopy(environment),
    }
    representative = {
        "schema": "uwfa-sc-v4-representative-source-free-preflight",
        "status": "PASS_REPRESENTATIVE_SOURCE_FREE_OUTER_FOLD", "source_snapshot_root_sha256": root,
        "fixture": {"streams": 15, "semantic_owners": 6, "private_streams": 12, "shared_tail_streams": 3},
        "outer_fold": {"all_150_candidates_fit_and_scored": True,
                       "literal_container_parse_decode_reencode_rebuild": True},
        "runtime_projection": {"passes": True}, "telemetry": copy.deepcopy(environment),
        "model_h2d_bytes_nonzero": True, "d2h_bytes_nonzero": True,
        "peak_host_ram_recorded": True, "peak_vram_recorded": True,
    }
    record = {"schema": "uwfa-sc-v4-bound-source-preflight", "source_snapshot_root_sha256": root,
              "all150": all150, "representative": representative,
              "independent_gpu_identity": identity}
    receipt = sha(common.canonical_json(record))
    bound = evidence(full="aa" * 32, structural="bb" * 32, pipeline="cc" * 32,
                     score_bytes=b"{}", root=root, preflight=receipt)
    return stage.SourcePreflightEvidence(all150, representative, identity, receipt), bound


class BoundaryAudit(unittest.TestCase):
    def test_exact_inventory_and_inert_unsealed_boundary(self) -> None:
        self.assertEqual(AUTHENTICATED_ROOT, EXPECTED_ROOT)
        self.assertNotIn("SOURCE_MANIFEST.json", SNAPSHOTS)
        self.assertEqual(stage.direct_main(), 2)
        OBSERVATIONS["exact_inventory_root"] = AUTHENTICATED_ROOT

    @unittest.skipUnless(os.name == "posix", "POSIX descriptor hostile test")
    def test_snapshot_nofollow_and_held_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = root / "real"
            real.mkdir()
            target = real / "source.py"
            target.write_bytes(b"old")
            (root / "alias").symlink_to(real, target_is_directory=True)
            with self.assertRaises(Exception):
                dispatcher.open_snapshot(root / "alias" / "source.py")
            held = dispatcher.open_snapshot(target)
            try:
                target.rename(real / "moved.py")
                target.write_bytes(b"replacement")
                self.assertEqual(held.data, b"old")
                held.verify_stable()
            finally:
                held.close()


class PublicationAudit(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "POSIX publication test")
    def test_staging_mutation_and_extra_member_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = common.RetainedOutputParent.open_path_source_only(Path(temp), "10" * 32)
            try:
                with common.CompletionLastOutput(parent, "result", "11" * 16) as transaction:
                    transaction.write_new("RESULT.json", b"original")
                    fd = os.open("RESULT.json", os.O_WRONLY | os.O_TRUNC, dir_fd=transaction.dir_fd)
                    try:
                        os.write(fd, b"changed")
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    evil = os.open("UNDECLARED.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                                   dir_fd=transaction.dir_fd)
                    os.close(evil)
                    transaction.complete(list(transaction.members), "22" * 32)
                published = Path(temp) / "result"
                self.assertTrue((published / "COMPLETE.json").is_file())
                with self.assertRaises(Exception):
                    envelope.verify_completed_directory(common, published.absolute())
            finally:
                parent.close()
        BLOCKERS["B1_COMPLETION_STAGING_NOT_REHASHED"] = (
            "complete() published mutated and undeclared staging bytes; only the later envelope verifier rejected them"
        )

    @unittest.skipUnless(os.name == "posix", "POSIX publication test")
    def test_retained_parent_noreplace_and_postrename_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent_path = root / "parent"
            parent_path.mkdir()
            parent = common.RetainedOutputParent.open_path_source_only(parent_path, "33" * 32)
            parent_path.rename(root / "held-parent")
            parent_path.mkdir()
            try:
                with common.CompletionLastOutput(parent, "kept", "44" * 16) as transaction:
                    transaction.write_new("A", b"a")
                    transaction.complete(list(transaction.members), "55" * 32)
                self.assertTrue((root / "held-parent" / "kept").is_dir())
                self.assertFalse((root / "parent" / "kept").exists())
            finally:
                parent.close()


class ScientificBindingAudit(unittest.TestCase):
    def test_duplicate_all150_and_weak_representative_receipt_are_accepted(self) -> None:
        typed, bound = preflight_with_duplicate_cells()
        result = stage.validate_source_preflight(common, protocol, typed, bound)
        self.assertEqual(result["receipt_sha256"], bound.source_preflight_receipt_sha256)
        BLOCKERS["B3_PREFLIGHT_RECEIPT_UNDERSPECIFIED"] = (
            "validator accepted 150 duplicate selector-0 rows and a representative receipt without candidate/winner/hash coverage"
        )

    def test_controls_accept_foreign_source_closure_and_wrong_source_artifact_argument(self) -> None:
        panel = scientific_panel()
        source_bindings = evidence(full="aa" * 32, structural="dd" * 32, pipeline="bb" * 32,
                                   score_bytes=b"source-score", decoder="12" * 32, salt="source-")
        source_result = {
            "controls_may_be_opened": True, "source_full_geometry_sha256": "aa" * 32,
            "source_structural_geometry_sha256": "dd" * 32, "source_pipeline_sha256": "bb" * 32,
            "source_final": {"absolute_saving_vs_bound_current_artifact_bpw": 1.0},
            "_panel": panel, "_bindings": source_bindings,
        }
        controls = []
        foreign_decoder = "99" * 32
        for seed, artifact in zip(common.CONTROL_SEEDS, (bytes([i + 1]) for i in range(8)), strict=True):
            score = {
                "schema": "uwfa-bound-baseline-score-v4", "status": "PASS_INDEPENDENT_BASELINE_SCORE",
                "artifact_sha256": sha(artifact), "artifact_bytes": len(artifact), "weights": 18,
                "relative_mse": 0.025, "sse_fp64": 0.025, "source_energy_fp64": 1.0,
                "normalization": "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM",
                "reconstruction_f64_sha256": "44" * 32, "original_source_panel_sha256": "ee" * 32,
                "independent_decoder_source_sha256": foreign_decoder,
            }
            score["score_receipt_sha256"] = sha(common.canonical_json(score))
            score_bytes = common.canonical_json(score)
            foreign = evidence(full="ee" * 32, structural="dd" * 32, pipeline="bb" * 32,
                               score_bytes=score_bytes, decoder=foreign_decoder, salt=f"foreign-{seed}-")
            record = {
                "schema": "uwfa-matched-gaussian-control-binding-v4", "seed": int(seed),
                "source_artifact_sha256": "cc" * 32,
                "source_full_geometry_sha256": "aa" * 32,
                "source_structural_geometry_sha256": "dd" * 32,
                "pipeline_sha256": "bb" * 32, "generator_source_sha256": sha(b"generator"),
                "moment_match_receipt_sha256": sha(b"moment"),
                "control_artifact_sha256": sha(artifact), "control_full_geometry_sha256": "ee" * 32,
                "control_structural_geometry_sha256": "dd" * 32,
            }
            record["binding_sha256"] = sha(common.canonical_json(record))
            controls.append({"artifact_bytes": artifact, "score_receipt_bytes": score_bytes,
                             "binding_record": record, "bindings": foreign,
                             "moment_match_receipt_bytes": b"moment", "generator_source_bytes": b"generator"})
        nested = {"estimable": True, "final_topology_selected_from_nested_fold_votes": {"selector_ordinal": 0}}
        with (
            mock.patch.object(stage, "prepare_panel", return_value=panel),
            mock.patch.object(protocol, "geometry_sha256", return_value="ee" * 32),
            mock.patch.object(protocol, "structural_geometry_sha256", return_value="dd" * 32),
            mock.patch.object(stage, "projected_updates", return_value={
                "primary_exact_identity_estimable": True, "passes_pre_fit_resource_budget": True,
                "passes_pre_fit_runtime_budget": True}),
            mock.patch.object(stage, "prepare_backend_cache", return_value={}),
            mock.patch.object(stage, "nested_holdout", return_value=nested),
            mock.patch.object(stage, "final_container", return_value={
                "absolute_saving_vs_bound_current_artifact_bpw": 0.5}),
        ):
            result = stage.controls_phase(
                common=common, protocol=protocol, container_codec=codec, semantic_codec=semantic,
                adapter_factory=object, backend_factory=object, source_result=source_result,
                source_artifact_sha256="cc" * 32, controls=controls,
                authenticated_descriptor_source_builder=lambda _raw: None,
                moment_match_replayer=lambda **kwargs: {
                    "status": "PASS_RECOMPUTED_MOMENT_MATCH", "seed": kwargs["seed"],
                    "source_moments_sha256": "55" * 32, "control_moments_sha256": "66" * 32,
                    "moment_match_receipt_sha256": sha(b"moment")},
            )
        self.assertEqual(len(result["controls"]), 8)
        self.assertNotEqual(panel["artifact"]["raw_sha256"], "cc" * 32)
        self.assertNotEqual(source_bindings.universal_decoder_sha256, foreign_decoder)
        BLOCKERS["B2_CONTROL_CLOSURE_NOT_BOUND_TO_SOURCE"] = (
            "controls_phase accepted a caller artifact digest not matching source panel and controls with foreign decoder/manifest/bootstrap/snapshot/preflight closures"
        )

    def test_v3_fold_geometry_and_structural_geometry_repairs(self) -> None:
        panel = scientific_panel()
        exact = stage._fold_plan(common, protocol, panel, policy="exact_identity")
        coordinate = stage._fold_plan(common, protocol, panel, policy="coordinate_disjoint")
        self.assertTrue(all(row["estimable"] and len(row["development_indices"]) == 15 for row in exact))
        self.assertFalse(any(row["estimable"] for row in coordinate))
        changed = copy.deepcopy(panel)
        changed["streams"][0]["baseline_payload_bytes"] += 1
        self.assertNotEqual(protocol.geometry_sha256(common, panel), protocol.geometry_sha256(common, changed))
        self.assertEqual(protocol.structural_geometry_sha256(common, panel),
                         protocol.structural_geometry_sha256(common, changed))


class CodecAndResourceAudit(unittest.TestCase):
    def test_candidate_q16_model_and_triplet_canonicality(self) -> None:
        bank = common.candidate_bank()
        self.assertEqual([row.selector_ordinal for row in bank], list(range(150)))
        self.assertEqual(common.q16_frequencies_from_counts([0, 0, 10**12, 0, 0, 10**12]),
                         [32768, 1, 65535])
        for candidate in bank:
            freq = [32768] * common.model_frequency_count(candidate)
            packet = common.serialize_model(candidate, freq)
            self.assertEqual(common.serialize_model(*common.deserialize_model(packet)), packet)
        bits, levels, base = b"\x00\x01", b"\x01\x02", struct.pack("<2H", 1, 65535)
        committed = common.selected_decision_triplet_sha256(bits, levels, base)
        self.assertNotEqual(committed, common.selected_decision_triplet_sha256(bits, levels, struct.pack("<2H", 2, 65535)))

    def test_static_resource_ceiling_and_preallocation_order(self) -> None:
        self.assertEqual(stage.MAX_VRAM_BYTES, cuda_source.MAX_VRAM_BYTES)
        self.assertEqual(stage.MAX_PACKED_SYMBOLS, cuda_source.MAX_PACKED_SYMBOLS)
        required = (4 * cuda_source.MAX_PACKED_SYMBOLS + cuda_source.MAX_AUXILIARY_DEVICE_BYTES
                    + cuda_source.VRAM_ALLOCATION_RESERVE_BYTES)
        self.assertLessEqual(required, cuda_source.MAX_VRAM_BYTES)
        source = SNAPSHOTS["cupy_backend.py"].decode("utf-8")
        plan = source.index("resource_plan = self.pack_resource_plan(offset, len(streams))")
        join = source.index('bit_blob = b"".join(bit_parts)', plan)
        cupy = source.index("device_bits = cp.frombuffer", join)
        self.assertLess(plan, join)
        self.assertLess(join, cupy)

    def test_literal_rate_owner_pages_and_repeated_range_visibility(self) -> None:
        source = fixture.make_fixture(common, codec, semantic, experts=2, hidden=32, intermediate=32)
        raw, _ = codec.build_container(
            common, semantic, model_packet=source["model_packet"], semantic_packet=source["semantic_packet"],
            immutable_state=b"audit", regions=source["regions"], weights=source["weights"], experts=2,
            baseline_object_bytes=10_000_000, audited_relative_mse=0.025,
            baseline_artifact_sha256=sha(b"baseline"), reconstruction_sha256=source["reconstruction_sha256"],
            audit_binding_sha256=sha(b"audit"), binding_hashes=binding_hashes())
        parsed = codec.parse_container(common, semantic, raw)
        metrics = codec.physical_metrics(common, semantic, parsed)
        self.assertEqual(metrics["actual_physical_rate_rational"]["numerator"] * metrics["actual_physical_rate_rational"]["denominator"],
                         metrics["actual_physical_rate_rational"]["numerator"] * metrics["actual_physical_rate_rational"]["denominator"])
        self.assertAlmostEqual(metrics["actual_physical_rate_bpw"], 8 * len(raw) / source["weights"])
        symbols = sum(int(row["symbols"]) for row in parsed["directory"])
        repeated = []
        for row in metrics["experts"]:
            ranges = row["instrumented_routed_read_ranges"]
            requested = sum(end - begin for begin, end in ranges)
            repeated.append({"expert": row["expert_ordinal"], "symbols_per_weight": symbols / source["weights"],
                             "requested_range_bytes": requested,
                             "unique_page_bytes": row["touched_page_bytes"]})
        OBSERVATIONS["literal_rate_and_repeated_ranges"] = repeated
        self.assertTrue(all(row["requested_range_bytes"] > 0 for row in repeated))

    @unittest.skipUnless(os.name == "posix", "larger portability replay kept on POSIX")
    def test_e250_unequal_shape_and_unselected_frame_nonread(self) -> None:
        source = fixture.make_unequal_shape_e250_fixture(common, codec, semantic)
        raw, _ = codec.build_container(
            common, semantic, model_packet=source["model_packet"], semantic_packet=source["semantic_packet"],
            immutable_state=b"e250-audit", regions=source["regions"], weights=source["weights"], experts=250,
            baseline_object_bytes=20_000_000, audited_relative_mse=0.025,
            baseline_artifact_sha256=sha(b"baseline"), reconstruction_sha256=source["reconstruction_sha256"],
            audit_binding_sha256=sha(b"audit"), binding_hashes=binding_hashes())
        parsed = codec.parse_container(common, semantic, raw)
        self.assertEqual((len(parsed["directory"]), len(parsed["regions"])), (751, 251))
        shared = next(row for row in parsed["directory"] if tuple(row["owners"]) == (0, 249))
        self.assertEqual([(r["expert"], r["weight_count"]) for r in shared["owner_contributions"]], [(0, 1), (249, 2)])


class RecordingResult(unittest.TextTestResult):
    def addSuccess(self, test: unittest.case.TestCase) -> None:
        super().addSuccess(test)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)
    receipt = {
        "schema": "uwfa-v4-independent-hostile-source-audit-v1",
        "status": "BLOCK_SOURCE_FREEZE" if BLOCKERS else "PASS_SOURCE_FREEZE",
        "producer_root_sha256": AUTHENTICATED_ROOT,
        "tests_run": result.testsRun,
        "failures": [str(test) for test, _ in result.failures],
        "errors": [str(test) for test, _ in result.errors],
        "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
        "blockers": BLOCKERS,
        "observations": OBSERVATIONS,
        "payload_or_qwen_access": False,
        "cupy_imported_or_cuda_context_created": False,
    }
    receipt["receipt_sha256"] = sha(common.canonical_json(receipt))
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise RuntimeError("refusing to overwrite audit receipt")
        output.write_bytes(encoded)
    print(encoded.decode("utf-8"), end="")
    return 0 if result.wasSuccessful() and BLOCKERS else 1


if __name__ == "__main__":
    raise SystemExit(main())
