#!/usr/bin/env python3
"""Hostile source-only tests for the v2 authority boundary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
import authority_v2 as auth
from independent_rm_order import fixed_weight_integers, independent_cpu_order


ZERO = "0" * 64
ONE = "1" * 64


def canonical_file(path: Path, value) -> str:
    payload = auth.canonical_json(value) + b"\n"
    path.write_bytes(payload)
    return auth.sha256(payload)


def source_rows(prefix: str):
    return [
        {"ordinal": 0, "role": "gate", "layer": 0, "expert": 0,
         "shape": [1, 2], "relative_path": f"{prefix}-g.bf16",
         "bytes": 4, "sha256": ZERO},
        {"ordinal": 1, "role": "up", "layer": 0, "expert": 0,
         "shape": [1, 2], "relative_path": f"{prefix}-u.bf16",
         "bytes": 4, "sha256": ZERO},
        {"ordinal": 2, "role": "down", "layer": 0, "expert": 0,
         "shape": [2, 1], "relative_path": f"{prefix}-d.bf16",
         "bytes": 4, "sha256": ZERO},
    ]


def scientific_record():
    pipeline = "2" * 64
    models = []
    controls = []
    for index, (family, kind) in enumerate(
            (("Qwen-SwiGLU-MoE", "qwen_bf16"),
             ("Other-SwiGLU-MoE", "swiglu_moe_bf16"))):
        model_id = f"model-{index}"
        control_id = f"control-{index}"
        models.append({
            "capability_id": model_id, "kind": kind,
            "architecture_family": family, "pipeline_sha256": pipeline,
            "checkpoint_manifest_sha256": "3" * 64,
            "tensor_manifest_sha256": "4" * 64,
            "control_family": None, "paired_model_capability_id": None,
            "generator_sha256": None, "seed_commitment_sha256": None,
            "moments_sha256": None,
            "required_control_capability_ids": [control_id],
            "sources": source_rows(model_id),
        })
        controls.append({
            "capability_id": control_id, "kind": "matched_gaussian_bf16",
            "architecture_family": family, "pipeline_sha256": pipeline,
            "checkpoint_manifest_sha256": None, "tensor_manifest_sha256": None,
            "control_family": "moment_matched_gaussian",
            "paired_model_capability_id": model_id,
            "generator_sha256": "5" * 64,
            "seed_commitment_sha256": "6" * 64,
            "moments_sha256": "7" * 64,
            "required_control_capability_ids": [],
            "sources": source_rows(control_id),
        })
    return {
        "schema": "strata-rm-global-swap-v2-scientific-capability",
        "owner": "independent_auditor",
        "audit_execution": {"receipt_sha256": "8" * 64,
                            "auditor_source_root_sha256": "9" * 64,
                            "executed": True,
                            "status": "PASS_INDEPENDENT_PROVENANCE_AUDIT"},
        "selection": {"frozen_before_test": True, "test_bytes_opened": 0,
                      "search_replayed_on_controls": True,
                      "pipeline_sha256": pipeline},
        "architecture_families": ["Qwen-SwiGLU-MoE", "Other-SwiGLU-MoE"],
        "cases": models + controls,
        "status": "PASS_AUDITOR_OWNED_PROVENANCE_CAPABILITIES",
    }


def result_row(capability_id: str, kind: str, family: str,
               target_f: float, control_family=None, paired=None):
    weights = 10
    packet_bytes = 3
    rate = 8 * packet_bytes / weights
    relative = target_f / (2.0 ** (2.0 * rate))
    return {"case_id": capability_id, "capability_id": capability_id,
            "kind": kind, "architecture_family": family,
            "control_family": control_family,
            "paired_model_capability_id": paired,
            "weights": weights, "literal_packet_bytes": packet_bytes,
            "sse_fp64_hex": relative.hex(), "energy_fp64_hex": (1.0).hex(),
            "physical_rate_bpw": rate, "relative_mse": relative, "F": target_f,
            "saving_bpw": -0.5 * math.log2(target_f),
            "read": {"read_amplification": 1.0}}


class AuthorityTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        with self.assertRaises(auth.AuthorityError):
            auth.strict_json(b'{"x":1,"x":2}', "duplicate")
        with self.assertRaises(auth.AuthorityError):
            auth.strict_json(b'{"x":NaN}', "nonfinite")

    def test_root_symlink_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlink unavailable")
            with self.assertRaises(auth.AuthorityError):
                auth.real_directory(link, "linked root")

    def test_external_file_snapshot_is_byte_pinned_and_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            payload = b"immutable external module\n"
            (source / "module.py").write_bytes(payload)
            destination = root / "snapshot"
            receipt = auth.snapshot_pinned_files(
                source, {"module.py": auth.sha256(payload)}, destination)
            self.assertTrue(receipt["immutable"])
            self.assertEqual((destination / "module.py").read_bytes(), payload)
            with self.assertRaises(auth.AuthorityError):
                auth.snapshot_pinned_files(
                    source, {"module.py": auth.sha256(payload)}, destination)

    def test_independent_cpu_order_semantics(self):
        self.assertEqual(list(fixed_weight_integers(5, 2)),
                         sorted(value for value in range(32)
                                if value.bit_count() == 2))
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable")
        for width in range(1, 11):
            n = 1 << width
            actual = independent_cpu_order(n, np).tolist()
            expected = sorted(range(n), key=lambda value:
                              (-value.bit_count(), value))
            self.assertEqual(actual, expected)

    def test_scientific_capability_is_out_of_band_and_auditor_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capability.json"
            record = scientific_record()
            pin = canonical_file(path, record)
            result = auth.authenticate_scientific_capability(path, pin)
            self.assertEqual(len(result["cases"]), 4)
            record["owner"] = "encoder"
            bad_pin = canonical_file(path, record)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_scientific_capability(path, bad_pin)

    def test_scientific_control_pairing_is_not_declarative_in_commitment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capability.json"
            record = scientific_record()
            record["cases"][2]["paired_model_capability_id"] = "wrong-model"
            pin = canonical_file(path, record)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_scientific_capability(path, pin)

    def test_successful_decoder_audit_receipt_is_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker_sha = "a" * 64
            launcher_sha = "b" * 64
            receipt = {
                "schema": "strata-rm-global-swap-v2-decoder-independent-audit-receipt",
                "executed": True, "status": "PASS_INDEPENDENT_DECODER_AUDIT_V2",
                "producer_worker_sha256": worker_sha,
                "instrumented_launcher_sha256": launcher_sha,
                "audit_source_root_sha256": "PLACEHOLDER",
                "protocol": "strata-rm-v2-decoder-worker-protocol",
                "filesystem_bypass_absent": True,
                "source_payload_access_absent": True,
                "packet_only_read_instrumentation_verified": True,
                "canonical_replay_verified": True,
                "fixed_universal_decoder": True,
                "qwen_specific_tables_absent": True,
                "hostile_tests": 12, "payloads_opened": 0,
            }
            audit_source = b"independent audit source\n"
            (root / "audit.py").write_bytes(audit_source)
            source_rows_manifest = [{"name": "audit.py", "bytes": len(audit_source),
                                     "sha256": auth.sha256(audit_source)}]
            source_root = auth._manifest_root(source_rows_manifest)
            receipt["audit_source_root_sha256"] = source_root
            receipt_payload = auth.canonical_json(receipt) + b"\n"
            (root / "AUDIT_RECEIPT.json").write_bytes(receipt_payload)
            manifest = {"schema":
                        "strata-rm-global-swap-v2-decoder-independent-audit-manifest",
                        "producer_worker_sha256": worker_sha,
                        "instrumented_launcher_sha256": launcher_sha,
                        "source_root_sha256": source_root,
                        "receipt_name": "AUDIT_RECEIPT.json",
                        "members": source_rows_manifest}
            manifest_payload = auth.canonical_json(manifest) + b"\n"
            (root / "source_manifest.json").write_bytes(manifest_payload)
            result = auth.authenticate_decoder_audit_capability(
                root, expected_manifest_sha256=auth.sha256(manifest_payload),
                expected_source_root_sha256=source_root,
                expected_receipt_sha256=auth.sha256(receipt_payload),
                expected_decoder_worker_sha256=worker_sha,
                expected_launcher_sha256=launcher_sha)
            self.assertEqual(result["status"],
                             "PASS_SEPARATELY_PINNED_SUCCESSFUL_DECODER_AUDIT")
            receipt["executed"] = False
            failed_payload = auth.canonical_json(receipt) + b"\n"
            (root / "AUDIT_RECEIPT.json").write_bytes(failed_payload)
            with self.assertRaises(auth.AuthorityError):
                auth.authenticate_decoder_audit_capability(
                    root, expected_manifest_sha256=auth.sha256(manifest_payload),
                    expected_source_root_sha256=source_root,
                    expected_receipt_sha256=auth.sha256(failed_payload),
                    expected_decoder_worker_sha256=worker_sha,
                    expected_launcher_sha256=launcher_sha)

    def test_family_target_and_strongest_control_are_both_enforced(self):
        scientific = scientific_record()
        results = []
        for index, family in enumerate(scientific["architecture_families"]):
            results.append(result_row(f"model-{index}",
                                      "qwen_bf16" if index == 0 else
                                      "swiglu_moe_bf16", family, 0.70))
            results.append(result_row(f"control-{index}",
                                      "matched_gaussian_bf16", family, 1.0,
                                      "moment_matched_gaussian", f"model-{index}"))
        accepted = auth.evaluate_family_acceptance(results, scientific)
        self.assertTrue(accepted["all_families_passed"])
        bad_family = [dict(row) for row in results]
        bad_family[2] = result_row("model-1", "swiglu_moe_bf16",
                                   scientific["architecture_families"][1], 0.81)
        with self.assertRaises(auth.AuthorityError):
            auth.evaluate_family_acceptance(bad_family, scientific)
        control_artifact = [dict(row) for row in results]
        control_artifact[1] = result_row("control-0", "matched_gaussian_bf16",
                                         scientific["architecture_families"][0],
                                         0.69, "moment_matched_gaussian", "model-0")
        with self.assertRaises(auth.AuthorityError):
            auth.evaluate_family_acceptance(control_artifact, scientific)

    def test_instrumented_launcher_measures_literal_packet_not_decoder_report(self):
        decoder_text = r'''#!/usr/bin/env python3
import argparse, array, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--request",type=Path,required=True); p.add_argument("--packet",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
request=json.loads(a.request.read_text(encoding="utf-8")); packet=a.packet.read_bytes(); a.output_dir.mkdir(exist_ok=True)
(a.output_dir/"canonical_packet.bin").write_bytes(packet)
files=[]
for row in request["sources"]:
 name=f"reconstruction-{row['ordinal']:04d}.f64"; files.append(name); count=row["shape"][0]*row["shape"][1]; values=array.array("d",[0.0]*count); (a.output_dir/name).write_bytes(values.tobytes())
receipt={"schema":"strata-rm-v2-independent-decoder-receipt","case_id":request["case_id"],"packet_sha256":hashlib.sha256(packet).hexdigest(),"packet_bytes":len(packet),"canonical_packet_sha256":hashlib.sha256(packet).hexdigest(),"canonical_packet_bytes":len(packet),"independent_decode_complete":True,"canonical_reencode_complete":True,"causal_probabilities_regenerated":True,"packet_consumed_exactly":True,"encoder_decisions_read":False,"encoder_probabilities_read":False,"source_payloads_opened":False,"reconstruction_files":files,"status":"PASS_INDEPENDENT_DECODE_V2"}
(a.output_dir/"receipt.json").write_text(json.dumps(receipt,sort_keys=True),encoding="utf-8")
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decoder = root / "decoder.py"
            request = root / "request.json"
            packet = root / "packet.bin"
            output = root / "output"
            instrumentation = root / "instrumentation.json"
            output.mkdir()
            decoder.write_text(decoder_text, encoding="utf-8")
            request.write_text(json.dumps({"case_id": "fixture", "sources": [
                {"ordinal": 0, "shape": [1, 2]}]}), encoding="utf-8")
            packet.write_bytes(bytes(range(251)) * 3)
            command = [sys.executable, "-I", "-B",
                       str(PACKAGE / "instrumented_decoder_worker.py"),
                       "--decoder", str(decoder), "--request", str(request),
                       "--packet", str(packet), "--output-dir", str(output),
                       "--instrumentation-output", str(instrumentation),
                       "--case-id", "fixture"]
            completed = subprocess.run(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       check=False, timeout=60)
            self.assertEqual(completed.returncode, 0,
                             completed.stderr.decode(errors="replace"))
            record = json.loads(instrumentation.read_text(encoding="utf-8"))
            self.assertEqual(record["unique_packet_bytes_read"], packet.stat().st_size)
            self.assertEqual(record["source_paths_supplied"], 0)
            self.assertTrue(record["denied_process_escape"])

    def test_commitment_cannot_declare_source_family_or_control_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"schema": "strata-rm-global-swap-v2-physical-commitment",
                      "mode": "production_global_rm_swap",
                      "v1_source_root_sha256": auth.V1_SOURCE_ROOT_SHA256,
                      "v1_review_source_root_sha256":
                      auth.V1_REVIEW_SOURCE_ROOT_SHA256,
                      "decoder_worker": {"relative_path": "decoder.py", "bytes": 1,
                                         "sha256": ZERO,
                                         "protocol":
                                         "strata-rm-v2-decoder-worker-protocol"},
                      "cases": [{"case_id": "c", "capability_id": "model-0",
                                 "packet": {"relative_path": "packet.bin",
                                            "bytes": 1, "sha256": ZERO}}]}
            path = root / "commitment.json"
            pin = canonical_file(path, record)
            auth._strict_commitment(path, pin)
            record["cases"][0]["architecture_family"] = "forged"
            bad_pin = canonical_file(path, record)
            with self.assertRaises(auth.AuthorityError):
                auth._strict_commitment(path, bad_pin)

    def test_physical_entry_requires_explicit_authorization(self):
        with self.assertRaises(auth.AuthorityError):
            auth.validate_physical_bundle(
                v2_package=PACKAGE, expected_v2_manifest_sha256=ZERO,
                evidence_root=PACKAGE, commitment_path=PACKAGE / "README.md",
                expected_commitment_sha256=ZERO,
                scientific_capability_path=PACKAGE / "README.md",
                expected_scientific_capability_sha256=ZERO,
                decoder_audit_root=PACKAGE,
                expected_decoder_audit_manifest_sha256=ZERO,
                expected_decoder_audit_source_root_sha256=ZERO,
                expected_decoder_audit_receipt_sha256=ZERO,
                authorization="WRONG")


if __name__ == "__main__":
    unittest.main(verbosity=2)
