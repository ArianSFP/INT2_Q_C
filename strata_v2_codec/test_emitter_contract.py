#!/usr/bin/env python3
"""Source-free adversarial tests for STRATA-v2 blind control bindings."""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
import tempfile
import unittest
from pathlib import Path

from strata_v2_codec import common, emit_and_lock


class EmitterContractTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> argparse.Namespace:
        source_root = root / "sources"
        source_root.mkdir()
        checkpoint = {"repo": "fixture/Qwen", "revision": "0" * 40}
        selection_rows = []
        source_rows = []
        route_records = []
        for ordinal in range(18):
            triplet, role_id = divmod(ordinal, 3)
            role = ("gate_proj", "up_proj", "down_proj")[role_id]
            shape = [2048, 768] if role_id == 2 else [768, 2048]
            relpath = f"matrix_{ordinal:02d}.bf16.bin"
            shard = "model-00001-of-00001.safetensors"
            range_begin = ordinal * 3_145_728
            range_end = range_begin + 3_145_728 - 1
            request_url = (
                f"https://huggingface.co/{checkpoint['repo']}/resolve/"
                f"{checkpoint['revision']}/{shard}"
            )
            (source_root / relpath).touch()
            (source_root / relpath).write_bytes(b"")
            with (source_root / relpath).open("r+b") as stream:
                stream.truncate(2 * 1_572_864)
            selection_rows.append(
                {
                    "matrix_ordinal": ordinal,
                    "tensor": (
                        f"model.layers.{triplet}.mlp.experts.{triplet}.{role}.weight"
                    ),
                    "role": role,
                    "layer": triplet,
                    "expert": triplet,
                    "shape": shape,
                    "dtype": "BF16",
                    "nvalues": 1_572_864,
                    "nbytes": 3_145_728,
                    "shard": shard,
                    "absolute_http_byte_range_inclusive": [range_begin, range_end],
                    "future_output_relpath": relpath,
                    "source_bf16_sha256": None,
                    "blocks": [
                        {
                            "canonical_block_index": index,
                            "source_bf16_sha256": None,
                        }
                        for index in range(6)
                    ],
                }
            )
            source_rows.append(
                {
                    "matrix_ordinal": ordinal,
                    "tensor": selection_rows[-1]["tensor"],
                    "role": role,
                    "layer": triplet,
                    "expert": triplet,
                    "shape": shape,
                    "dtype": "BF16",
                    "nvalues": 1_572_864,
                    "nbytes": 3_145_728,
                    "block_count": 6,
                    "shard": shard,
                    "http_range_inclusive": [range_begin, range_end],
                    "http_response": {
                        "status": 206,
                        "request_url": request_url,
                        "requested_range": f"bytes={range_begin}-{range_end}",
                        "content_range": f"bytes {range_begin}-{range_end}/{18 * 3_145_728 + 1}",
                        "content_length": 3_145_728,
                        "content_encoding": "identity",
                        "body_bytes": 3_145_728,
                        "body_sha256": "0" * 64,
                    },
                    "output_relpath": relpath,
                    "source_bf16_sha256": "0" * 64,
                    "blocks": [
                        {
                            "canonical_block_index": index,
                            "nvalues": 1 << 18,
                            "nbytes": 1 << 19,
                            "source_bf16_sha256": "1" * 64,
                        }
                        for index in range(6)
                    ],
                }
            )
            route_records.append(
                struct.pack(">HHBBH", triplet, triplet, role_id, role_id == 2, 768)
            )

        selection = common.sealed(
            {
                "schema": emit_and_lock.BLIND_SELECTION_CONTRACT[0],
                "status": emit_and_lock.BLIND_SELECTION_CONTRACT[1],
                "checkpoint": checkpoint,
                "matrices": selection_rows,
            }
        )
        selection_path = root / "selection.lock.json"
        common.write_json(selection_path, selection)
        route_path = root / "route.bin"
        route_path.write_bytes(b"".join(route_records))
        format_path = Path(emit_and_lock.__file__).with_name("FORMAT.md").resolve()
        artifacts = {
            "emitter": common.sha256_file(Path(emit_and_lock.__file__).resolve()),
            "common": common.sha256_file(Path(common.__file__).resolve()),
            "format": common.sha256_file(format_path),
            "freeze_validator": "2" * 64,
        }
        freeze = common.sealed(
            {
                "schema": "strata_xklt_sc_v2_codec_freeze_v1",
                "status": "frozen_before_blind_source_access",
                "selection_lock_sha256": selection["lock_sha256"],
                "selection_lock_file_sha256": common.sha256_file(selection_path),
                "route_file_sha256": common.sha256_file(route_path),
                "preaccess_state": {},
                "frozen_artifact_sha256s": artifacts,
            }
        )
        freeze_path = root / "codec_freeze.lock.json"
        common.write_json(freeze_path, freeze)
        validation = common.sealed(
            {
                "schema": "strata_xklt_sc_v2_codec_freeze_validation_v1",
                "status": "validated_before_blind_source_access",
                "passed": True,
                "freeze_path": "blind_protocol_v2/codec_freeze.lock.json",
                "freeze_file_sha256": common.sha256_file(freeze_path),
                "freeze_internal_lock_sha256": freeze["lock_sha256"],
                "executing_validator_sha256": artifacts["freeze_validator"],
                "frozen_artifact_count": len(artifacts),
                "development_pooled_relative_mse": 0.049,
                "gaussian_mse_reference": math.exp2(-4.3),
                "physical_bits": common.PHYSICAL_BITS,
                "physical_bpw": common.PHYSICAL_BITS / common.WEIGHTS,
                "preaccess_state": {},
            }
        )
        validation_path = root / "codec_freeze.validation.json"
        common.write_json(validation_path, validation)
        source = common.sealed(
            {
                "schema": emit_and_lock.BLIND_SOURCE_CONTRACT[0],
                "status": emit_and_lock.BLIND_SOURCE_CONTRACT[1],
                "dtype": "BF16",
                "selection_lock_sha256": selection["lock_sha256"],
                "codec_freeze": {
                    "file_sha256": common.sha256_file(freeze_path),
                    "internal_lock_sha256": freeze["lock_sha256"],
                },
                "codec_freeze_validation": {
                    "file_sha256": common.sha256_file(validation_path),
                    "internal_lock_sha256": validation["lock_sha256"],
                },
                "checkpoint": checkpoint,
                "source_root": "sources",
                "matrix_count": 18,
                "block_count": 108,
                "source_values": common.WEIGHTS,
                "source_bytes": 2 * common.WEIGHTS,
                "matrices": source_rows,
            }
        )
        source_path = root / "source_hashes.lock.json"
        common.write_json(source_path, source)
        return argparse.Namespace(
            selection_lock=selection_path,
            source_lock=source_path,
            route=route_path,
            format=format_path,
            codec_freeze=freeze_path,
            source_root=None,
            protocol_mode="blind",
        )

    def test_valid_blind_control_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_fixture(Path(temporary))
            selection, source, route, root, _ = emit_and_lock.load_documents(args)
            rows = emit_and_lock.validate_matrix_bindings(
                selection, source, common.parse_route(route), root
            )
            self.assertEqual(len(rows), 18)

    def test_schema_tamper_is_rejected_by_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_fixture(Path(temporary))
            value = __import__("json").loads(args.selection_lock.read_text())
            value["schema"] = "wrong"
            common.write_json(args.selection_lock, value)
            with self.assertRaisesRegex(ValueError, "invalid internal seal"):
                emit_and_lock.load_documents(args)

    def test_forged_failed_validation_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_fixture(Path(temporary))
            validation_path = args.codec_freeze.with_name("codec_freeze.validation.json")
            validation = __import__("json").loads(validation_path.read_text())
            validation.pop("lock_sha256")
            validation["passed"] = False
            validation = common.sealed(validation)
            common.write_json(validation_path, validation)
            source = __import__("json").loads(args.source_lock.read_text())
            source.pop("lock_sha256")
            source["codec_freeze_validation"] = {
                "file_sha256": common.sha256_file(validation_path),
                "internal_lock_sha256": validation["lock_sha256"],
            }
            common.write_json(args.source_lock, common.sealed(source))
            with self.assertRaisesRegex(ValueError, "validation receipt contract"):
                emit_and_lock.load_documents(args)

    def test_route_rebinding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_fixture(Path(temporary))
            route = bytearray(args.route.read_bytes())
            # Preserve route structure while changing triplet 0's expert.
            for record in range(3):
                route[8 * record + 3] ^= 1
            args.route.write_bytes(route)
            with self.assertRaisesRegex(ValueError, "exact selection/route"):
                emit_and_lock.load_documents(args)

    def test_role_shape_and_path_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_fixture(Path(temporary))
            selection, source, route, root, _ = emit_and_lock.load_documents(args)
            selection["matrices"][0]["shape"] = [2048, 768]
            with self.assertRaisesRegex(ValueError, "matrix binding mismatch"):
                emit_and_lock.validate_matrix_bindings(
                    selection, source, common.parse_route(route), root
                )

    def test_tensor_identity_metadata_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_fixture(Path(temporary))
            selection, source, route, root, _ = emit_and_lock.load_documents(args)
            source["matrices"][0]["role"] = "up_proj"
            with self.assertRaisesRegex(ValueError, "matrix binding mismatch"):
                emit_and_lock.validate_matrix_bindings(
                    selection, source, common.parse_route(route), root
                )

    def test_finalized_matrix_identity_fields_are_mandatory(self) -> None:
        for field in ("role", "layer", "expert", "block_count"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                args = self.make_fixture(Path(temporary))
                selection, source, route, root, _ = emit_and_lock.load_documents(args)
                del source["matrices"][0][field]
                with self.assertRaisesRegex(ValueError, "omits finalized fields"):
                    emit_and_lock.validate_matrix_bindings(
                        selection, source, common.parse_route(route), root
                    )

    def test_finalized_http_provenance_fields_are_exact(self) -> None:
        mutations = {
            "shard": lambda row: row.__setitem__("shard", "wrong.safetensors"),
            "range": lambda row: row.__setitem__(
                "http_range_inclusive", [1, 3_145_728]
            ),
            "status": lambda row: row["http_response"].__setitem__("status", 200),
            "body_hash": lambda row: row["http_response"].__setitem__(
                "body_sha256", "f" * 64
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                args = self.make_fixture(Path(temporary))
                selection, source, route, root, _ = emit_and_lock.load_documents(args)
                mutate(source["matrices"][0])
                with self.assertRaisesRegex(ValueError, "matrix binding mismatch"):
                    emit_and_lock.validate_matrix_bindings(
                        selection, source, common.parse_route(route), root
                    )

    def test_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.make_fixture(Path(temporary))
            selection, source, route, root, _ = emit_and_lock.load_documents(args)
            source["matrices"][0]["output_relpath"] = "../escape.bf16.bin"
            selection["matrices"][0]["future_output_relpath"] = "../escape.bf16.bin"
            with self.assertRaisesRegex(ValueError, "escapes source root"):
                emit_and_lock.validate_matrix_bindings(
                    selection, source, common.parse_route(route), root
                )

    def test_exact_matrix_and_block_hashes_are_both_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = bytes(3_145_728)
            path = root / "matrix.bin"
            path.write_bytes(payload)
            chunk = 1 << 19
            meta = {
                "matrix_ordinal": 0,
                "source_path": path,
                "shape": [768, 2048],
                "source_bf16_sha256": hashlib.sha256(payload).hexdigest(),
                "source_block_sha256s": [
                    hashlib.sha256(payload[offset : offset + chunk]).hexdigest()
                    for offset in range(0, len(payload), chunk)
                ],
            }
            self.assertEqual(emit_and_lock.load_source_words(meta).size, 1_572_864)
            meta["source_block_sha256s"][3] = "f" * 64
            with self.assertRaisesRegex(ValueError, "block hashes mismatch"):
                emit_and_lock.load_source_words(meta)


if __name__ == "__main__":
    unittest.main()
