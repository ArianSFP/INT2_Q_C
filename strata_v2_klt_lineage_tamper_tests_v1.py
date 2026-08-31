#!/usr/bin/env python3
"""Negative tests for the standalone STRATA-v2 independent auditor.

The test consumes an explicit completed development or blind rehearsal.  It
never discovers source paths.  Every mutation is written under a temporary
directory, and success means the independent auditor rejects every tamper.
"""

from __future__ import annotations

import argparse
import copy
import json
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

import strata_v2_klt_mixed_independent_auditor_v1 as audit


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def reseal(value: dict[str, Any]) -> None:
    value.pop("lock_sha256", None)
    value["lock_sha256"] = audit.sha256_bytes(audit.canonical_json_bytes(value))


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", type=Path, required=True)
    ap.add_argument("--protocol-mode", choices=("blind", "development"), required=True)
    ap.add_argument("--selection-lock", type=Path, required=True)
    ap.add_argument("--source-lock", type=Path, required=True)
    ap.add_argument("--codec-freeze", type=Path, required=True)
    ap.add_argument("--format-freeze", type=Path, required=True)
    ap.add_argument("--preencoding-manifest", type=Path, required=True)
    ap.add_argument("--allocation-lock", type=Path, required=True)
    ap.add_argument("--one-shot-intent", type=Path, required=True)
    ap.add_argument("--one-shot-summary", type=Path, required=True)
    ap.add_argument("--source-root", type=Path)
    ap.add_argument("--device", choices=("cupy", "numpy"), default="cupy")
    ap.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args()
    run_root = args.run_root.resolve(strict=True)
    if args.one_shot_summary.resolve(strict=True).parent != run_root:
        raise AssertionError("one-shot summary is not directly under the explicit run root")
    output = run_root / "independent_lineage_tamper_tests.json"
    if output.exists():
        raise FileExistsError(output)
    executing_harness_hash = audit.sha256_file(Path(__file__).resolve())

    original_paths = {
        "selection": args.selection_lock.resolve(strict=True),
        "source": args.source_lock.resolve(strict=True),
        "codec": args.codec_freeze.resolve(strict=True),
        "format": args.format_freeze.resolve(strict=True),
        "manifest": args.preencoding_manifest.resolve(strict=True),
        "allocation": args.allocation_lock.resolve(strict=True),
        "intent": args.one_shot_intent.resolve(strict=True),
        "summary": args.one_shot_summary.resolve(strict=True),
    }
    original_source_document = load(original_paths["source"])
    effective_source_root = (
        args.source_root.resolve(strict=True)
        if args.source_root is not None
        else (
            original_paths["source"].parent
            / str(original_source_document.get("source_root", "."))
        ).resolve(strict=True)
    )
    parsed = audit.parse_container(args.container)

    def validate(paths: dict[str, Path], candidate: audit.ParsedContainer = parsed) -> dict[str, Any]:
        return audit.validate_source_lineage(
            candidate,
            args.protocol_mode,
            paths["selection"],
            paths["source"],
            paths["codec"],
            paths["format"],
            paths["manifest"],
            paths["allocation"],
            paths["intent"],
            paths["summary"],
            effective_source_root,
        )

    baseline = validate(original_paths)
    if args.protocol_mode == "blind":
        frozen = baseline["codec_freeze"].get("frozen_artifact_sha256s", {})
        if frozen.get("lineage_tamper_test") != executing_harness_hash:
            raise AssertionError(
                "executing tamper harness is not the exact frozen lineage_tamper_test"
            )
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="strata_v2_tamper_") as temporary:
        temp = Path(temporary)

        def expect_json_rejection(
            name: str,
            key: str,
            mutate: Any,
            seal: bool,
        ) -> None:
            document = copy.deepcopy(load(original_paths[key]))
            mutate(document)
            if seal:
                reseal(document)
            candidate_path = temp / f"{name}.json"
            write_json(candidate_path, document)
            candidate_paths = dict(original_paths)
            candidate_paths[key] = candidate_path
            try:
                validate(candidate_paths)
            except Exception as exc:  # Every mutation must fail closed.
                rows.append({"tamper": name, "rejected": True, "error": f"{type(exc).__name__}: {exc}"})
                return
            raise AssertionError(f"lineage tamper was accepted: {name}")

        expect_json_rejection(
            "selection_role_resealed",
            "selection",
            lambda doc: doc["matrices"][0].__setitem__("role", "up"),
            True,
        )
        expect_json_rejection(
            "source_nested_hash_resealed",
            "source",
            lambda doc: doc["matrices"][0]["blocks"][0].__setitem__("source_bf16_sha256", "0" * 64),
            True,
        )
        expect_json_rejection(
            "codec_threshold_resealed",
            "codec",
            lambda doc: doc.__setitem__("primary_mse_threshold", 0.5),
            True,
        )
        expect_json_rejection(
            "manifest_source_binding",
            "manifest",
            lambda doc: doc["bindings"]["sources"][0].__setitem__("source_bf16_sha256", "1" * 64),
            False,
        )
        expect_json_rejection(
            "allocation_profile_resealed",
            "allocation",
            lambda doc: doc["blocks"][0].__setitem__("profile_id", (int(doc["blocks"][0]["profile_id"]) + 1) & 255),
            True,
        )
        expect_json_rejection(
            "intent_retry_permission",
            "intent",
            lambda doc: doc.__setitem__("retry_resume_or_adaptive_rate_change_allowed", True),
            False,
        )
        expect_json_rejection(
            "summary_artifact_hash",
            "summary",
            lambda doc: doc["physical"].__setitem__("artifact_sha256", "2" * 64),
            False,
        )

        raw = bytearray(args.container.read_bytes())
        raw[audit.HEADER_OFFSET_COEFFICIENTS] ^= 1
        struct.pack_into(
            "<I",
            raw,
            audit.HEADER_OFFSET_CRC32,
            zlib.crc32(raw[: audit.HEADER_OFFSET_CRC32]) & 0xFFFFFFFF,
        )
        coefficient_path = temp / "coefficient_tamper.bin"
        coefficient_path.write_bytes(raw)
        try:
            audit.parse_container(coefficient_path)
        except Exception as exc:
            rows.append({"tamper": "coefficient_regeneration", "rejected": True, "error": f"{type(exc).__name__}: {exc}"})
        else:
            raise AssertionError("coefficient tamper was accepted")

        raw = bytearray(args.container.read_bytes())
        byte_cursor = 0
        for directory in parsed.directory:
            remainder = directory.logical_bits & 7
            payload_bytes = (directory.logical_bits + 7) // 8
            if remainder:
                final_byte = audit.RESERVOIR_OFFSET + byte_cursor + payload_bytes - 1
                raw[final_byte] |= 1 << (7 - remainder)
                break
            byte_cursor += payload_bytes
        else:
            raise AssertionError("rehearsal unexpectedly has no per-stream padding bit")
        padding_path = temp / "padding_tamper.bin"
        padding_path.write_bytes(raw)
        try:
            audit.parse_container(padding_path)
        except Exception as exc:
            rows.append({"tamper": "nonzero_stream_padding", "rejected": True, "error": f"{type(exc).__name__}: {exc}"})
        else:
            raise AssertionError("stream-padding tamper was accepted")

        raw = bytearray(args.container.read_bytes())
        raw[audit.DIRECTORY_OFFSET + 1] ^= 1
        scale_path = temp / "scale_tamper.bin"
        scale_path.write_bytes(raw)
        scale_parsed = audit.parse_container(scale_path)
        try:
            audit.audit_source_staging_and_scales(scale_parsed, baseline, args.device)
        except Exception as exc:
            rows.append({"tamper": "directory_scale", "rejected": True, "error": f"{type(exc).__name__}: {exc}"})
        else:
            raise AssertionError("directory-scale tamper was accepted")

    expected_names = {
        "selection_role_resealed",
        "source_nested_hash_resealed",
        "codec_threshold_resealed",
        "manifest_source_binding",
        "allocation_profile_resealed",
        "intent_retry_permission",
        "summary_artifact_hash",
        "coefficient_regeneration",
        "nonzero_stream_padding",
        "directory_scale",
    }
    observed_names = [str(row["tamper"]) for row in rows]
    exact_unique_name_set = (
        len(observed_names) == len(expected_names)
        and len(observed_names) == len(set(observed_names))
        and set(observed_names) == expected_names
    )
    result = {
        "schema": "strata_v2_klt_independent_lineage_tamper_tests_v1",
        "passed": all(row["rejected"] for row in rows) and exact_unique_name_set,
        "protocol_mode": args.protocol_mode,
        "container_sha256": audit.sha256_file(args.container),
        "auditor_sha256": audit.sha256_file(Path(audit.__file__).resolve()),
        "executing_tamper_harness_sha256": executing_harness_hash,
        "tamper_count": len(rows),
        "exact_unique_tamper_name_set": exact_unique_name_set,
        "expected_tamper_names": sorted(expected_names),
        "tamper_rows": rows,
        "claim_boundary": "explicit completed rehearsal only; no source discovery",
    }
    if not result["passed"]:
        raise AssertionError(result)
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
