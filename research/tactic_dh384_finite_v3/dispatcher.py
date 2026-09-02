#!/usr/bin/env python3
"""Reviewed, source-authenticated finite TACTIC-DH384 v3 pilot dispatcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any


AUTHORIZATION = "RUN_REVIEWED_TACTIC_DH384_FINITE_V3_ONE_BOUND_EXPERT"
PACKAGE_SCHEMA = "tactic-dh384-finite-v3-source-manifest-v1"
MAX_SOURCE_BYTES = 4 * (1 << 20)
TARGET_D_AT_2P5 = 0.025


class DispatchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True,
        allow_nan=False) + "\n").encode("ascii")


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"{label}: duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                DispatchError(f"{label}: nonfinite {item}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DispatchError(f"{label}: JSON: {error}") from error
    require(isinstance(value, dict), f"{label}: object")
    return value


def _reject_symlink_chain(path: Path, label: str) -> None:
    cursor = path
    while True:
        metadata = os.lstat(cursor)
        require(not stat.S_ISLNK(metadata.st_mode),
                f"{label}: symlink component {cursor}")
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def read_regular(path: Path, *, expected_sha256: str | None,
                 maximum_bytes: int, label: str) -> bytes:
    require(path.is_absolute(), f"{label}: absolute path")
    _reject_symlink_chain(path, label)
    descriptor = os.open(
        os.fspath(path), os.O_RDONLY | getattr(os, "O_BINARY", 0) |
        getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 < before.st_size <= maximum_bytes,
                f"{label}: regular sole-link byte bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(bool(chunk), f"{label}: short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        require(os.read(descriptor, 1) == b"", f"{label}: trailing bytes")
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_mode, before.st_size,
                 before.st_mtime_ns, before.st_ctime_ns, before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_mode, after.st_size,
                 after.st_mtime_ns, after.st_ctime_ns, after.st_nlink),
                f"{label}: identity drift")
        payload = b"".join(chunks)
        if expected_sha256 is not None:
            require(sha256(payload) == expected_sha256,
                    f"{label}: externally pinned digest")
        return payload
    finally:
        os.close(descriptor)


def load_module(name: str, source: bytes) -> Any:
    require(name not in sys.modules, f"authenticated module collision: {name}")
    digest = sha256(source)
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated:{name}:{digest}>"
    module.__package__ = ""
    module.__authenticated_source_sha256__ = digest
    sys.modules[name] = module
    try:
        exec(compile(source, module.__file__, "exec", dont_inherit=True,
                     optimize=0), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def authenticate_package(package_dir: Path,
                         expected_manifest_sha256: str) -> Any:
    manifest_payload = read_regular(
        package_dir / "SOURCE_MANIFEST.json",
        expected_sha256=expected_manifest_sha256,
        maximum_bytes=MAX_SOURCE_BYTES, label="finite source manifest")
    manifest = strict_json(manifest_payload, "finite source manifest")
    require(manifest.get("schema") == PACKAGE_SCHEMA,
            "finite source manifest schema")
    rows = manifest.get("members")
    require(isinstance(rows, list), "finite source rows")
    candidates = [row for row in rows if isinstance(row, dict) and
                  row.get("name") == "source_auth.py"]
    require(len(candidates) == 1, "source_auth bootstrap row")
    row = candidates[0]
    source = read_regular(
        package_dir / "source_auth.py", expected_sha256=row["sha256"],
        maximum_bytes=MAX_SOURCE_BYTES, label="finite source_auth")
    require(len(source) == row["bytes"], "finite source_auth exact bytes")
    auth = load_module("tactic_dh384_finite_v3_source_auth", source)
    return auth.HeldSourcePackage(
        package_dir, expected_manifest_sha256,
        executing_path=Path(__file__).resolve(strict=True))


def _source_f64(np: Any, payload: bytes, expected_values: int) -> Any:
    require(type(payload) is bytes and len(payload) == 2 * expected_values,
            "canonical source BF16 bytes")
    words = np.frombuffer(payload, dtype="<u2")
    values = ((words.astype(np.uint32) << np.uint32(16))
              .view(np.float32).astype(np.float64))
    require(values.size == expected_values and bool(np.all(np.isfinite(values))),
            "finite source BF16 decode")
    return values


def _decoded_tile(coarse: bytes, ordinal: int, spec: Any,
                  runtime: Any, v6_codec: Any) -> Any:
    begin = ordinal * spec.COARSE_RECORD_BYTES
    packet = coarse[begin:begin + spec.COARSE_RECORD_BYTES]
    require(len(packet) == spec.COARSE_RECORD_BYTES,
            "coarse tile packet bytes")
    decoded = v6_codec.decode_tile_v6(packet, runtime)
    require(decoded.canonical_packet == packet and
            decoded.report["canonical_reencode_matches"] is True and
            decoded.canonical_symbols_i32.dtype.str == "<i4",
            "exact v6 tile decoder/reencoder")
    return decoded


def _continuous_gate(cp: Any, np: Any, coarse: bytes,
                     role_bytes: dict[str, bytes], runtime: Any,
                     v6_codec: Any, spec: Any,
                     encoder: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    source_energy = 0.0
    coarse_sse = 0.0
    parent_projected = 0.0
    active_projected = 0.0
    rows: list[dict[str, Any]] = []
    sources = {
        role: _source_f64(np, role_bytes[role], spec.ROLE_VALUES)
        for role in spec.ROLES
    }
    for role_ordinal, role in enumerate(spec.ROLES):
        for tile_ordinal in range(spec.COARSE_TILES_PER_ROLE):
            coarse_ordinal = role_ordinal * spec.COARSE_TILES_PER_ROLE + tile_ordinal
            decoded = _decoded_tile(
                coarse, coarse_ordinal, spec, runtime, v6_codec)
            begin = tile_ordinal * spec.COARSE_TILE_VALUES
            end = begin + spec.COARSE_TILE_VALUES
            source = sources[role][begin:end]
            reconstruction = decoded.reconstruction_f32.astype(np.float64)
            residual = source - reconstruction
            tile_source_energy = float(np.dot(source, source))
            gate = encoder.continuous_tile(
                cp, decoded.canonical_symbols_i32, residual,
                role_ordinal, spec)
            source_energy += tile_source_energy
            coarse_sse += gate["error_energy_fp64"]
            parent_projected += gate[
                "audited_parent_rank384_projected_energy_fp64"]
            active_projected += gate[
                "active_rank376_projected_energy_fp64"]
            rows.append({
                "role": role,
                "role_ordinal": role_ordinal,
                "tile_ordinal": tile_ordinal,
                "source_energy_fp64": tile_source_energy,
                **gate,
            })
    require(source_energy > 0.0 and coarse_sse > 0.0,
            "positive pilot energy")
    d0 = coarse_sse / source_energy
    required = max(0.0, 1.0 - TARGET_D_AT_2P5 / d0)
    parent_capture = parent_projected / coarse_sse
    active_capture = active_projected / coarse_sse
    parent_survives = parent_capture + 2e-12 >= required
    active_survives = active_capture + 2e-12 >= required
    if not parent_survives:
        decision = "HARD_REJECT_PARENT_RANK384_CONTINUOUS_SPAN_BELOW_NESTED_THRESHOLD"
    elif not active_survives:
        decision = "HARD_REJECT_ACTIVE_RANK376_CONTINUOUS_SUBSPAN_BELOW_NESTED_THRESHOLD"
    else:
        decision = "PROMOTE_TO_LITERAL_FINITE_384_BIT_RECORDS"
    gaussian_parent_one_bit = (2.0 / math.pi) * (384.0 / 4096.0)
    gaussian_active_one_bit = (2.0 / math.pi) * (376.0 / 4096.0)
    ideal_all_remaining_rate_d = d0 * 2.0 ** (-2.0 * (13.0 / 128.0))
    ideal_fine_payload_only_d = d0 * 2.0 ** (-2.0 * (12.0 / 128.0))
    record = {
        "schema": "tactic-dh384-finite-v3-continuous-gate-v1",
        "decision": decision,
        "target_D_at_literal_2p5_bpw": TARGET_D_AT_2P5,
        "source_energy_fp64": source_energy,
        "coarse_sse_fp64": coarse_sse,
        "coarse_relative_mse": d0,
        "exact_nested_required_error_capture": required,
        "old_19p1_percent_threshold_reused": False,
        "gaussian_successive_refinement_diagnostic": {
            "all_remaining_rate_bpw": 13.0 / 128.0,
            "all_remaining_rate_ideal_D": ideal_all_remaining_rate_d,
            "all_remaining_rate_ideal_F_at_2p5":
                ideal_all_remaining_rate_d * 32.0,
            "literal_fine_payload_rate_bpw": 12.0 / 128.0,
            "literal_fine_payload_ideal_D": ideal_fine_payload_only_d,
            "literal_fine_payload_ideal_F_at_2p5":
                ideal_fine_payload_only_d * 32.0,
            "interpretation": (
                "Gaussian successive-refinement hierarchy preserves the "
                "normalized coarse F and cannot produce a below-Gaussian gain."),
        },
        "audited_parent_rank384": {
            "projected_energy_fp64": parent_projected,
            "measured_capture": parent_capture,
            "survives_exact_nested_threshold": parent_survives,
            "isotropic_dimension_fraction_ceiling": 384.0 / 4096.0,
            "isotropic_dimension_fraction_percent": 100.0 * 384.0 / 4096.0,
            "gaussian_one_bit_per_dimension_expected_total_capture":
                gaussian_parent_one_bit,
            "gaussian_one_bit_per_dimension_expected_total_capture_percent":
                100.0 * gaussian_parent_one_bit,
        },
        "implemented_active_rank376": {
            "projected_energy_fp64": active_projected,
            "measured_capture": active_capture,
            "survives_exact_nested_threshold": active_survives,
            "isotropic_dimension_fraction_ceiling": 376.0 / 4096.0,
            "gaussian_sign_capture_expected_total": gaussian_active_one_bit,
            "charged_scale_bits_per_record": 8,
            "charged_sign_bits_per_record": 376,
        },
        "dominance_rule": (
            "Every finite v3 correction lies in B[:,0:376], a subset of the "
            "audited B[:,0:384] span. Failure of either applicable continuous "
            "capture threshold hard-stops finite engineering for this branch."),
        "rows": rows,
    }
    return record, sources


def _encode_fine(cp: Any, np: Any, coarse: bytes,
                 sources: dict[str, Any], runtime: Any, v6_codec: Any,
                 spec: Any, encoder: Any) -> tuple[bytes, dict[str, Any]]:
    fine_parts: list[bytes] = []
    rows: list[dict[str, Any]] = []
    coarse_sse = 0.0
    finite_sse = 0.0
    for role_ordinal, role in enumerate(spec.ROLES):
        for tile_ordinal in range(spec.COARSE_TILES_PER_ROLE):
            coarse_ordinal = role_ordinal * spec.COARSE_TILES_PER_ROLE + tile_ordinal
            decoded = _decoded_tile(
                coarse, coarse_ordinal, spec, runtime, v6_codec)
            begin = tile_ordinal * spec.COARSE_TILE_VALUES
            end = begin + spec.COARSE_TILE_VALUES
            source = sources[role][begin:end]
            reconstruction = decoded.reconstruction_f32.astype(np.float64)
            residual = source - reconstruction
            records, _correction, receipt = encoder.encode_tile(
                cp, np, decoded.canonical_symbols_i32, residual,
                reconstruction, role_ordinal, spec)
            fine_parts.append(records)
            coarse_sse += receipt["coarse_sse_fp64"]
            finite_sse += receipt["finite_sse_fp64"]
            rows.append({
                "role": role,
                "role_ordinal": role_ordinal,
                "tile_ordinal": tile_ordinal,
                **receipt,
            })
    fine = b"".join(fine_parts)
    require(len(fine) == spec.FINE_BYTES and
            len(spec.split_fine_stream(fine)) == spec.FINE_RECORDS,
            "complete canonical fine stream")
    require(finite_sse <= coarse_sse * (1.0 + 5e-13),
            "finite full expert never worsens")
    return fine, {
        "schema": "tactic-dh384-finite-v3-encoder-receipt-v1",
        "status": "PASS_ALL_LITERAL_48_BYTE_RECORDS_CANONICAL",
        "fine_sha256": sha256(fine),
        "fine_bytes": len(fine),
        "fine_records": spec.FINE_RECORDS,
        "fine_record_bytes": spec.FINE_RECORD_BYTES,
        "bits_per_record": spec.FINE_RECORD_BITS,
        "charged_scale_bits_per_record": 8,
        "charged_sign_bits_per_record": spec.ACTIVE_RANK,
        "unallocated_or_free_bits_per_record": 0,
        "all_records_canonical_decode_reencode": True,
        "all_corrections_in_active_rank376_subset_parent_rank384": True,
        "coarse_sse_fp64": coarse_sse,
        "finite_sse_fp64": finite_sse,
        "finite_error_capture": 1.0 - finite_sse / coarse_sse,
        "rows": rows,
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    require(arguments.authorization == AUTHORIZATION,
            "explicit finite-pilot authorization token")
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "invoke dispatcher with CPython -I -B")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
            "CUDA_VISIBLE_DEVICES must be exactly 0")
    for value, label in (
        (arguments.repo_root, "repo root"),
        (arguments.v6_package_dir, "v6 package"),
        (arguments.v6_result_dir, "v6 result"),
        (arguments.input_manifest, "input manifest"),
        (arguments.launch_review, "launch review"),
        (arguments.output_dir, "output"),
    ):
        require(Path(value).is_absolute(), f"{label} absolute")
    package_dir = Path(__file__).resolve().parent
    require(arguments.repo_root.resolve(strict=True) ==
            package_dir.parents[1], "canonical repository root")
    require(not os.path.lexists(arguments.output_dir),
            "output namespace must be absent")
    require(package_dir not in arguments.output_dir.parent.resolve(strict=True).parents
            and arguments.output_dir.parent.resolve(strict=True) != package_dir,
            "output outside immutable package")

    with authenticate_package(
        package_dir, arguments.package_manifest_sha256) as package:
        sources = package.sources
        external = load_module(
            "tactic_dh384_finite_v3_external_contract",
            sources["external_contract.py"])
        bridge_module = load_module(
            "tactic_dh384_finite_v3_runtime_bridge",
            sources["runtime_bridge.py"])
        v6_lock = strict_json(sources["V6_LOCK.json"], "frozen v6 lock")

        # The external review is the first non-source input opened. It binds
        # both the immutable finite source and the already completed/audited
        # v6 object. No v6 result or BF16 path is touched before this passes.
        review_payload = external.read_held_regular(
            arguments.launch_review,
            expected_sha256=arguments.launch_review_sha256,
            maximum_bytes=1 << 20, label="external launch review")
        review = external.validate_launch_review(
            review_payload,
            package_manifest_sha256=package.manifest_sha256,
            package_source_root_sha256=package.source_root_sha256,
            v6_source_manifest_sha256=
                v6_lock["source_manifest"]["sha256"],
            v6_source_root_sha256=v6_lock["source_root_sha256"])

        with bridge_module.HeldV6Package(
            arguments.repo_root, arguments.v6_package_dir,
            sources["V6_LOCK.json"]) as v6_package:
            # Exact v6 runtime/dependencies are authenticated before payload.
            runtime, v6_codec = v6_package.load_runtime(arguments.repo_root)
            cp, np = runtime.cupy, runtime.numpy
            spec = load_module(
                "tactic_dh384_finite_v3_format_spec",
                sources["format_spec.py"])
            encoder = load_module(
                "tactic_dh384_finite_v3_encoder",
                sources["finite_encoder.py"])
            independent = load_module(
                "tactic_dh384_finite_v3_independent_decoder",
                sources["independent_decoder.py"])
            publisher = load_module(
                "tactic_dh384_finite_v3_atomic_publish",
                sources["atomic_publish.py"])
            require(spec.sha256(spec.universal_selector_packet()) ==
                    spec.SELECTOR_PACKET_SHA256,
                    "audited parent selector at runtime")

            with external.HeldCompletedV6Result(
                arguments.v6_result_dir,
                expected_complete_sha256=review["v6_complete_sha256"],
                expected_v6_source_root_sha256=
                    v6_lock["source_root_sha256"],
                expected_input_manifest_sha256=
                    review["input_manifest_sha256"]) as v6_result:
                inputs = external.authenticate_inputs(
                    arguments.input_manifest,
                    expected_manifest_sha256=review["input_manifest_sha256"],
                    expected_v6_binding=v6_result.input_binding)
                coarse = v6_result.coarse
                gate, sources_f64 = _continuous_gate(
                    cp, np, coarse, inputs["role_bytes"], runtime,
                    v6_codec, spec, encoder)
                source_binding = package.receipt()
                base_bindings = {
                    "source_closure": source_binding,
                    "v6_source_closure": v6_package.receipt(),
                    "v6_result_binding": v6_result.receipt(),
                    "input_manifest_sha256": inputs["manifest_sha256"],
                    "input_roles": inputs["bindings"],
                    "launch_review_file_sha256": sha256(review_payload),
                    "launch_review_claim_sha256":
                        review["review_claim_sha256"],
                    "runtime_closure": runtime.receipt,
                }
                common_boundary = {
                    "one_qwen_geometry_expert_pilot_only": True,
                    "qwen_or_model_identity_available_to_codec": False,
                    "six_expert_amortized_global_packet_emitted_or_parsed": False,
                    "seventy_three_over_seventy_two_read_claim": False,
                    "universal_tail_result": False,
                    "non_qwen_portability_result": False,
                    "accelerator_inference_hbm_below_2x_claim": False,
                }
                if gate["decision"] != "PROMOTE_TO_LITERAL_FINITE_384_BIT_RECORDS":
                    result = {
                        "schema": "tactic-dh384-finite-v3-result-v1",
                        "status": gate["decision"],
                        "positive_claim_authority": False,
                        "finite_composite_emitted": False,
                        "continuous_gate": gate,
                        "bindings": base_bindings,
                        "claim_boundary": common_boundary,
                    }
                    members = {
                        "CONTINUOUS_GATE.json": pretty_json(gate),
                        "RESULT.json": pretty_json(result),
                    }
                    completion = {
                        "schema": "tactic-dh384-finite-v3-completion-v1",
                        "status": result["status"],
                        "finite_composite_emitted": False,
                        "positive_claim_authority": False,
                        "source_root_sha256": package.source_root_sha256,
                        "v6_complete_sha256": review["v6_complete_sha256"],
                    }
                else:
                    fine, encoder_receipt = _encode_fine(
                        cp, np, coarse, sources_f64, runtime, v6_codec,
                        spec, encoder)
                    header = spec.make_header({
                        "coarse_sha256": sha256(coarse),
                        "fine_sha256": sha256(fine),
                        "input_manifest_sha256": inputs["manifest_sha256"],
                        "v6_complete_sha256": review["v6_complete_sha256"],
                        "producer_source_manifest_sha256":
                            package.manifest_sha256,
                        "producer_source_root_sha256": package.source_root_sha256,
                    })
                    composite = header + coarse + fine
                    require(len(composite) == spec.COMPOSITE_BYTES and
                            spec.split_composite(composite)[1:] == (coarse, fine),
                            "literal composite construction")
                    decoder_receipt = independent.decode_composite(
                        cp, np, composite, inputs["role_bytes"], runtime,
                        v6_codec, spec)
                    score = decoder_receipt["original_domain_score"]
                    require(math.isclose(
                        score["pooled_sse_fp64"],
                        encoder_receipt["finite_sse_fp64"],
                        rel_tol=2e-12, abs_tol=2e-9),
                        "encoder/independent-decoder FP64 SSE")
                    target_pass = score["F_at_literal_2p5_bpw"] <= 0.8
                    status = (
                        "PASS_FINITE_F_LE_0P8_ONE_EXPERT_NONPROMOTING_AUDIT_REQUIRED"
                        if target_pass else
                        "HARD_REJECT_FINITE_F_ABOVE_0P8_ONE_EXPERT"
                    )
                    result = {
                        "schema": "tactic-dh384-finite-v3-result-v1",
                        "status": status,
                        "positive_claim_authority": False,
                        "finite_composite_emitted": True,
                        "continuous_gate": gate,
                        "encoder": encoder_receipt,
                        "independent_decoder": decoder_receipt,
                        "literal_single_expert_physical_ledger": {
                            "header_bytes": spec.PILOT_HEADER_BYTES,
                            "coarse_bytes": spec.COARSE_BYTES,
                            "fine_bytes": spec.FINE_BYTES,
                            "composite_bytes": spec.COMPOSITE_BYTES,
                            "physical_bpw_exact": "320/128",
                            "physical_bpw": 2.5,
                            "fine_bits_per_4096": 384,
                            "scale_bits_inside_each_record": 8,
                            "sign_bits_inside_each_record": 376,
                            "unallocated_metadata_bits": 0,
                            "global_packet_bytes": 0,
                        },
                        "traffic": spec.single_expert_traffic(
                            start_offset_mod_page=0, external_passes=1),
                        "bindings": base_bindings,
                        "claim_boundary": common_boundary,
                    }
                    members = {
                        "COMPOSITE.bin": composite,
                        "CONTINUOUS_GATE.json": pretty_json(gate),
                        "ENCODER_RECEIPT.json": pretty_json(encoder_receipt),
                        "INDEPENDENT_DECODER_RECEIPT.json":
                            pretty_json(decoder_receipt),
                        "RESULT.json": pretty_json(result),
                    }
                    completion = {
                        "schema": "tactic-dh384-finite-v3-completion-v1",
                        "status": status,
                        "finite_composite_emitted": True,
                        "positive_claim_authority": False,
                        "source_root_sha256": package.source_root_sha256,
                        "v6_complete_sha256": review["v6_complete_sha256"],
                        "composite_sha256": sha256(composite),
                    }

                package.verify_final()
                v6_package.verify_final()
                v6_result.verify_final()
                publication = publisher.publish_atomic(
                    arguments.output_dir, members, completion)
                package.verify_final()
                v6_package.verify_final()
                v6_result.verify_final()
                return {"result": result, "publication": publication}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--package-manifest-sha256", required=True)
    result.add_argument("--repo-root", type=Path, required=True)
    result.add_argument("--v6-package-dir", type=Path, required=True)
    result.add_argument("--v6-result-dir", type=Path, required=True)
    result.add_argument("--input-manifest", type=Path, required=True)
    result.add_argument("--launch-review", type=Path, required=True)
    result.add_argument("--launch-review-sha256", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> int:
    record = run(parser().parse_args())
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
