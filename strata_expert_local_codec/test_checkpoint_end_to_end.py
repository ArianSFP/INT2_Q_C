from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any

from strata_expert_local_codec import checkpoint_tamper_tests as tamper
from strata_expert_local_codec import common
from strata_expert_local_codec import verify_checkpoint as verify
from strata_expert_local_codec.test_verify_checkpoint import synthetic_container


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def release_file_row(root: Path, path: Path, role: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
        "role": role,
        "classification": "synthetic_test_evidence",
    }


def read_ledger(parsed: dict[str, Any]) -> dict[str, Any]:
    route = parsed["route_rows"]
    rows = []
    for ordinal, expected in enumerate(parsed["experts"]):
        rows.append(
            {
                "expert_ordinal": ordinal,
                "layer": route[3 * ordinal]["layer"],
                "expert": route[3 * ordinal]["expert"],
                "required_blocks": expected["required_blocks"],
                "payload_bytes": expected["payload_bytes"],
                "cold_bytes": expected["cold_bytes"],
                "cold_amplification_vs_equal_physical_share": expected[
                    "cold_amplification"
                ],
                "page_4k_union_bytes": expected["page_4k_union_bytes"],
                "page_4k_amplification_vs_equal_physical_share": expected[
                    "page_4k_amplification"
                ],
            }
        )
    return {
        "definition": "cold bytes fetched divided by one-sixth of physical container bytes",
        "equal_physical_share_bytes": verify.PHYSICAL_BYTES / verify.EXPERTS,
        "experts": rows,
        "max_cold": parsed["max_cold"],
        "max_4k": parsed["max_4k"],
        "passes_below_2x": True,
    }


def build_release(root: Path) -> None:
    container = root / "strata_expert_affine_n20n21.bin"
    container.write_bytes(synthetic_container(first_logical_bits=1, first_payload=0))
    parsed = verify.parse_container(container)
    header = parsed["asset_payloads"]["header.bin"]
    route = parsed["asset_payloads"]["route.bin"]
    labels = parsed["asset_payloads"]["labels_3bit.bin"]
    profiles = parsed["asset_payloads"]["profiles.bin"]

    asset_rows: dict[str, dict[str, Any]] = {}
    asset_paths: dict[str, Path] = {}
    for name, payload in parsed["asset_payloads"].items():
        path = root / "assets" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        asset_paths[name] = path
        asset_rows[name] = {
            "relpath": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    blocks = []
    for ordinal, logn in enumerate(verify.BLOCK_LOG2):
        sc_seed, rht_seed, digest = verify.derive_seeds(
            header, route, labels, profiles, ordinal
        )
        blocks.append(
            {
                "block_ordinal": ordinal,
                "block_log2": logn,
                "values": 1 << logn,
                "groups": verify.BLOCK_GROUPS[ordinal],
                "owner_experts": verify.expected_owner_experts(ordinal),
                "segment": "private" if ordinal < verify.PRIVATE_BLOCKS else "paired_tail",
                "source_energy_fp64": float(1 << logn),
                "selected_group_ordinals_sha256": "1" * 64,
                "staging_relpath": f"staging/block_{ordinal:02d}.bin",
                "staging_bytes": 2 * (1 << logn),
                "staging_sha256": hashlib.sha256(f"staging-{ordinal}".encode()).hexdigest(),
                "profile_id": 0,
                "nominal_rate_bpw": 1.75,
                "test_distortion": math.exp2(-3.5),
                "sc_seed_u32": sc_seed,
                "rht_seed_u64": rht_seed,
                "seed_digest_sha256": digest,
            }
        )

    sources = []
    for ordinal, route_row in enumerate(parsed["route_rows"]):
        role = route_row["role"]
        shape = [2048, 768] if role == "down" else [768, 2048]
        tensor = (
            f"model.layers.{route_row['layer']}.mlp.experts.{route_row['expert']}."
            f"{role}_proj.weight"
        )
        sources.append(
            {
                "matrix_ordinal": ordinal,
                "tensor": tensor,
                "role": role,
                "axis": route_row["axis"],
                "shape": shape,
                "source_relpath": f"sources/{tensor}.bf16.bin",
                "source_bf16_sha256": hashlib.sha256(f"source-{ordinal}".encode()).hexdigest(),
                "bytes": 2 * verify.GROUPS_PER_MATRIX * verify.GROUP_VALUES,
            }
        )

    plan = common.sealed(
        {
            "schema": verify.PLAN_SCHEMA,
            "status": "sealed_before_arithmetic_encoding",
            "sources": sources,
            "assets": asset_rows,
            "blocks": blocks,
            "allocation": {"profile_ids": [0] * verify.BLOCKS},
            "physical_ledger": {
                "header_bytes": verify.HEADER_BYTES,
                "route_bytes": verify.ROUTE_BYTES,
                "label_bytes": verify.LABEL_BYTES,
                "directory_bytes": verify.DIRECTORY_BYTES,
                "reservoir_bytes": verify.PHYSICAL_BYTES - verify.PREFIX_BYTES,
                "physical_bytes": verify.PHYSICAL_BYTES,
                "physical_bits": verify.PHYSICAL_BITS,
                "physical_bpw": verify.PHYSICAL_BPW,
                "reserve_bits": 65_536,
            },
            "coverage": {
                "experts": verify.EXPERTS,
                "matrices": 18,
                "groups": verify.GROUPS,
                "weights": verify.WEIGHTS,
                "blocks": verify.BLOCKS,
                "every_group_once": True,
                "cupy_energy_sum_fp64": float(verify.WEIGHTS),
            },
        }
    )
    plan_path = root / "plan.lock.json"
    write_json(plan_path, plan)

    encoded_rows = []
    metadata_paths = []
    for ordinal, block in enumerate(blocks):
        directory = parsed["directory"][ordinal]
        payload = parsed["_raw"][
            directory["file_byte_begin"] : directory["file_byte_end_exclusive"]
        ]
        logical_bits = directory["logical_bits"]
        payload_hash = hashlib.sha256(payload).hexdigest()
        literal_hash = hashlib.sha256(
            struct.pack("<If", logical_bits, 1.0) + payload
        ).hexdigest()
        metadata = {
            "schema": "strata_xklt_sc_v2_single_block_encoder_v1",
            "parameters": {
                "block_length": block["values"],
                "trials": 1,
                "sigma_source": 1.0,
                "test_channel_distortion": block["test_distortion"],
                "eta": 0.25,
                "alphabet_size": 64,
                "decision": "map",
                "seed": block["sc_seed_u32"],
            },
            "trials": [
                {
                    "arithmetic_logical_bits": logical_bits,
                    "arithmetic_payload_bytes": len(payload),
                    "arithmetic_payload_sha256": payload_hash,
                    "literal_container_bytes": len(payload) + 8,
                    "literal_container_sha256": literal_hash,
                    "arithmetic_roundtrip_bits_match": True,
                    "causal_decoder_frequencies_match": True,
                    "reconstruction_indices_match": True,
                    "relative_mse": 0.04,
                    "source": {
                        "block_bf16_sha256": block["staging_sha256"],
                        "block_rms_fp64": 1.0,
                        "decoder_scale_fp32": 1.0,
                        "values": block["values"],
                        "rht": {"enabled": True, "seed_u64": block["rht_seed_u64"]},
                    },
                }
            ],
        }
        path = root / "encoder_metadata" / f"block_{ordinal:02d}.json"
        write_json(path, metadata)
        metadata_paths.append(path)
        encoded_rows.append(
            {
                "block_ordinal": ordinal,
                "metadata_sha256": common.sha256_file(path),
                "container_sha256": literal_hash,
                "logical_bits": logical_bits,
                "normalized_relative_mse": 0.04,
                "block_rms_fp64": 1.0,
                "checks": {"synthetic_contract": True},
            }
        )

    ledger = read_ledger(parsed)
    summary_directory = []
    for ordinal, row in enumerate(parsed["directory"]):
        summary_directory.append(
            {
                "block_ordinal": ordinal,
                "owner_experts": verify.expected_owner_experts(ordinal),
                "logical_bits": row["logical_bits"],
                "payload_bytes": row["payload_bytes"],
                "file_byte_begin": row["file_byte_begin"],
                "file_byte_end_exclusive": row["file_byte_end_exclusive"],
                "scale_fp16_hex": row["scale_fp16_hex"],
            }
        )
    summary = {
        "schema": verify.SUMMARY_SCHEMA,
        "status": "encoded_once_and_packed",
        "plan_lock_sha256": plan["lock_sha256"],
        "artifact": {
            "relpath": container.name,
            "sha256": parsed["sha256"],
            "physical_bytes": verify.PHYSICAL_BYTES,
            "physical_bits": verify.PHYSICAL_BITS,
            "physical_bpw": verify.PHYSICAL_BPW,
            "logical_payload_bits": parsed["logical_payload_bits"],
            "payload_bytes": parsed["used_payload_bytes"],
            "zero_reservoir_tail_bytes": parsed["zero_tail_bytes"],
        },
        "encoded_blocks": encoded_rows,
        "directory": summary_directory,
        "read_amplification": ledger,
        "encoder_side_staging_mse": 0.04,
        "encoder_side_gaussian_gain_at_physical_rate": 1.0 - 0.04 / 0.03125,
    }
    summary_path = root / "summary.json"
    write_json(summary_path, summary)

    matrix_rows = []
    for source in sources:
        matrix_rows.append(
            {
                **{key: source[key] for key in (
                    "matrix_ordinal", "tensor", "role", "axis", "shape",
                    "source_relpath", "source_bf16_sha256"
                )},
                "sse_fp64": 0.04,
                "source_energy_fp64": 1.0,
                "relative_mse": 0.04,
            }
        )
    expert_rows = []
    for ordinal in range(verify.EXPERTS):
        expert_rows.append(
            {
                "expert_ordinal": ordinal,
                "layer": parsed["route_rows"][3 * ordinal]["layer"],
                "expert": parsed["route_rows"][3 * ordinal]["expert"],
                "sse_fp64": sum(row["sse_fp64"] for row in matrix_rows[3 * ordinal:3 * ordinal + 3]),
                "source_energy_fp64": 3.0,
                "relative_mse": 0.04,
            }
        )
    total_sse = sum(row["sse_fp64"] for row in matrix_rows)
    total_energy = sum(row["source_energy_fp64"] for row in matrix_rows)
    gaussian = math.exp2(-2.0 * verify.PHYSICAL_BPW)
    target = (1.0 - verify.TARGET_FRACTION) * gaussian
    decoded = []
    for ordinal, directory in enumerate(parsed["directory"]):
        decoded.append(
            {
                "block_ordinal": ordinal,
                "values": 1 << verify.BLOCK_LOG2[ordinal],
                "profile_q": 0,
                "logical_bits": directory["logical_bits"],
                "payload_sha256": directory["payload_sha256"],
                "canonical_reencode_matches": True,
            }
        )
    audit = {
        "schema": verify.AUDIT_SCHEMA,
        "status": "passed",
        "bindings": {
            "plan_lock_sha256": plan["lock_sha256"],
            "sources_canonical_sha256": hashlib.sha256(
                verify.canonical_json_bytes(sources)
            ).hexdigest(),
        },
        "container": {
            "sha256": parsed["sha256"],
            "physical_bytes": verify.PHYSICAL_BYTES,
            "physical_bpw": verify.PHYSICAL_BPW,
            "logical_payload_bits": parsed["logical_payload_bits"],
            "used_payload_bytes": parsed["used_payload_bytes"],
            "zero_tail_bytes": parsed["zero_tail_bytes"],
        },
        "decode": {
            "blocks": decoded,
            "decoded_blocks": verify.BLOCKS,
            "canonical_reencode_all_match": True,
            "every_group_once": True,
        },
        "source_score": {
            "sse_sum_fp64": total_sse,
            "source_energy_sum_fp64": total_energy,
            "energy_weighted_relative_mse": total_sse / total_energy,
            "matrices": matrix_rows,
            "experts": expert_rows,
        },
        "rate_relative": {
            "physical_bpw": verify.PHYSICAL_BPW,
            "gaussian_assumed_mse": gaussian,
            "mse_below_gaussian_fraction": 1.0 - 0.04 / gaussian,
            "target_fraction": verify.TARGET_FRACTION,
            "target_mse_at_same_rate": target,
            "passes_20_percent_below_same_rate_gaussian": False,
        },
        "read_amplification": ledger,
        "milestone_gate": {
            "current_mse_ceiling": verify.CURRENT_MSE,
            "source_mse_passed": True,
            "max_4k_read_amplification": parsed["max_4k"],
            "read_below_2x_passed": True,
            "rate_at_or_below_2p5_passed": True,
            "passed": True,
        },
    }
    audit_path = root / "independent_audit.json"
    write_json(audit_path, audit)

    files = [
        release_file_row(root, plan_path, "plan"),
        release_file_row(root, summary_path, "summary"),
        release_file_row(root, container, "container"),
        release_file_row(root, audit_path, "independent_audit"),
    ]
    for name, path in asset_paths.items():
        files.append(release_file_row(root, path, f"asset_{name.replace('.', '_')}"))
    for ordinal, path in enumerate(metadata_paths):
        files.append(release_file_row(root, path, f"encoder_block_{ordinal:02d}"))
    manifest = {
        "schema": verify.MANIFEST_SCHEMA,
        "artifact": {
            "format_magic": "PLRLOC3\\0",
            "model": "Qwen/Qwen3-30B-A3B",
            "weights": verify.WEIGHTS,
            "physical_bytes": verify.PHYSICAL_BYTES,
            "physical_bits": verify.PHYSICAL_BITS,
            "physical_bpw": verify.PHYSICAL_BPW,
            "container_sha256": parsed["sha256"],
        },
        "claim": {
            "checkpoint_passed": True,
            "final_rate_relative_gate_passed": False,
            "energy_weighted_relative_mse": total_sse / total_energy,
            "physical_bpw": verify.PHYSICAL_BPW,
            "max_4k_read_amplification": parsed["max_4k"],
            "gaussian_assumed_mse": gaussian,
            "target_mse_at_same_rate": target,
        },
        "files": files,
    }
    write_json(root / "checkpoint_manifest.json", manifest)


def run_release_verifier(root: Path) -> dict[str, Any]:
    manifest_path = root / "checkpoint_manifest.json"
    manifest = verify.load_json(manifest_path, "manifest")
    roles = verify.verify_manifest(root, manifest_path, manifest)
    parsed = verify.parse_container(roles["container"])
    return verify.verify_evidence(roles, parsed, manifest)


class FullCheckpointVerifierTests(unittest.TestCase):
    def test_synthetic_release_and_deep_tamper_cases(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            release = Path(text) / "release"
            release.mkdir()
            build_release(release)
            result = run_release_verifier(release)
            self.assertTrue(result["checkpoint_passed"])
            self.assertFalse(result["final_rate_relative_gate_passed"])

            for name, mutation, expected in tamper.CASES:
                trial = Path(text) / f"trial_{name}"
                shutil.copytree(release, trial)
                mutation(trial)
                with self.assertRaisesRegex(Exception, expected):
                    run_release_verifier(trial)


if __name__ == "__main__":
    unittest.main()
