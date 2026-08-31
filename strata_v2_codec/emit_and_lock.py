#!/usr/bin/env python3
"""Materialize and seal a STRATA-XKLT-SC v2 pre-encoding stream.

The emitter accepts any finalized eighteen-matrix gate/up/down selection that
matches the public route geometry.  It contains no tensor identities or paths.
It writes the header, literal route, raw labels, profiles, fourteen staging
blocks, a manifest, and an allocation lock before any encoder is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_v2_codec import common


DEVELOPMENT_SELECTION_CONTRACT = (
    "int2-qwen-blind-selection-v1",
    "selected_and_header_validated_tensor_payload_unopened",
)
BLIND_SELECTION_CONTRACT = (
    "int2-qwen-blind-selection-proposal-v2",
    "sealed_metadata_only_proposal_payload_unopened_not_codec_frozen",
)
DEVELOPMENT_SOURCE_CONTRACT = (
    "int2-qwen-blind-source-finalization-v1",
    "all_locked_sources_materialized_and_hash_finalized",
)
BLIND_SOURCE_CONTRACT = (
    "int2-qwen-blind-source-finalization-v2",
    "all_locked_sources_materialized_and_hash_finalized",
)


def role_name(value: str) -> str:
    value = value.lower()
    return value[:-5] if value.endswith("_proj") else value


_QWEN_EXPERT_TENSOR = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)


def parse_tensor_identity(value: object) -> tuple[int, int, str]:
    """Recover the normative route identity from a canonical Qwen tensor id."""
    if not isinstance(value, str):
        raise ValueError(f"tensor identity is not a string: {value!r}")
    match = _QWEN_EXPERT_TENSOR.fullmatch(value)
    if match is None:
        raise ValueError(f"noncanonical Qwen expert tensor identity: {value!r}")
    return int(match.group(1)), int(match.group(2)), match.group(3)


def natural_words(raw: np.ndarray, axis: str) -> np.ndarray:
    if axis == "row":
        return np.ascontiguousarray(raw.reshape(-1, common.GROUP_VALUES))
    if axis == "column":
        return np.ascontiguousarray(raw.T.reshape(-1, common.GROUP_VALUES))
    raise ValueError(f"unknown natural axis {axis!r}")


def bf16_words_to_gpu(words: np.ndarray) -> cp.ndarray:
    words_gpu = cp.asarray(words, dtype=cp.uint16)
    return (words_gpu.astype(cp.uint32) << cp.uint32(16)).view(cp.float32)


def fp32_gpu_to_bf16_rne(values: cp.ndarray) -> cp.ndarray:
    words = values.astype(cp.float32, copy=False).view(cp.uint32)
    rounded = words + cp.uint32(0x7FFF) + ((words >> cp.uint32(16)) & cp.uint32(1))
    return (rounded >> cp.uint32(16)).astype(cp.uint16)


def fp32_klt_gpu(
    up: cp.ndarray, down: cp.ndarray, cosine: np.float32, sine: np.float32
) -> tuple[cp.ndarray, cp.ndarray]:
    # Separate multiply kernels make the FP32 rounding points normative and
    # prevent backend-dependent fused multiply-add contraction.
    cu, si = cp.float32(cosine), cp.float32(sine)
    c_up = (cu * up).astype(cp.float32)
    s_down = (si * down).astype(cp.float32)
    neg_s_up = (-si * up).astype(cp.float32)
    c_down = (cu * down).astype(cp.float32)
    return (
        (c_up + s_down).astype(cp.float32),
        (neg_s_up + c_down).astype(cp.float32),
    )


def fp32_klt_cpu(
    up: np.ndarray, down: np.ndarray, cosine: np.float32, sine: np.float32
) -> tuple[np.ndarray, np.ndarray]:
    c_up = np.asarray(cosine * up, dtype=np.float32)
    s_down = np.asarray(sine * down, dtype=np.float32)
    neg_s_up = np.asarray(-sine * up, dtype=np.float32)
    c_down = np.asarray(cosine * down, dtype=np.float32)
    return (
        np.asarray(c_up + s_down, dtype=np.float32),
        np.asarray(neg_s_up + c_down, dtype=np.float32),
    )


def load_json_bytes(path: Path, description: str) -> tuple[dict, bytes, str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object: {path}")
    return value, payload, hashlib.sha256(payload).hexdigest()


def load_documents(
    args: argparse.Namespace,
) -> tuple[dict, dict, bytes, Path, dict[str, Any]]:
    selection_path = args.selection_lock.resolve()
    source_lock_path = args.source_lock.resolve()
    route_path = args.route.resolve()
    format_path = args.format.resolve()
    codec_freeze_path = args.codec_freeze.resolve()
    selection, selection_payload, selection_file_hash = load_json_bytes(
        selection_path, "selection lock"
    )
    source_lock, source_lock_payload, source_lock_file_hash = load_json_bytes(
        source_lock_path, "source lock"
    )
    codec_freeze, codec_freeze_payload, codec_freeze_file_hash = load_json_bytes(
        codec_freeze_path, "codec freeze"
    )
    route = route_path.read_bytes()
    format_payload = format_path.read_bytes()
    common.parse_route(route)
    if not common.verify_internal_seal(selection):
        raise ValueError("selection lock has an invalid internal seal")
    if not common.verify_internal_seal(source_lock):
        raise ValueError("source lock has an invalid internal seal")
    if not common.verify_internal_seal(codec_freeze):
        raise ValueError("codec freeze has an invalid internal seal")
    expected_freeze_schema = (
        "strata_xklt_sc_v2_codec_freeze_v1"
        if args.protocol_mode == "blind"
        else "polaris_strata_blind_codec_freeze_v1"
    )
    if (codec_freeze.get("schema"), codec_freeze.get("status")) != (
        expected_freeze_schema,
        "frozen_before_blind_source_access",
    ):
        raise ValueError("codec freeze schema/status contract mismatch")
    expected_selection_contract = (
        BLIND_SELECTION_CONTRACT
        if args.protocol_mode == "blind"
        else DEVELOPMENT_SELECTION_CONTRACT
    )
    expected_source_contract = (
        BLIND_SOURCE_CONTRACT
        if args.protocol_mode == "blind"
        else DEVELOPMENT_SOURCE_CONTRACT
    )
    if (selection.get("schema"), selection.get("status")) != expected_selection_contract:
        raise ValueError(
            f"{args.protocol_mode} selection schema/status mismatch: "
            f"{(selection.get('schema'), selection.get('status'))!r}"
        )
    if (source_lock.get("schema"), source_lock.get("status")) != expected_source_contract:
        raise ValueError(
            f"{args.protocol_mode} source schema/status mismatch: "
            f"{(source_lock.get('schema'), source_lock.get('status'))!r}"
        )
    if args.protocol_mode == "blind" and source_lock.get("dtype") != "BF16":
        raise ValueError("blind source-finalization top-level dtype must be BF16")
    selection_internal = selection.get("lock_sha256")
    if source_lock.get("selection_lock_sha256") != selection_internal:
        raise ValueError("source lock is not bound to the supplied selection lock")
    if codec_freeze.get("selection_lock_sha256") != selection_internal:
        raise ValueError("codec freeze is not bound to the supplied selection lock")
    source_freeze = source_lock.get("codec_freeze")
    expected_source_freeze = {
        "file_sha256": codec_freeze_file_hash,
        "internal_lock_sha256": codec_freeze.get("lock_sha256"),
    }
    if not isinstance(source_freeze, dict) or any(
        source_freeze.get(key) != value for key, value in expected_source_freeze.items()
    ):
        raise ValueError(
            "source lock is not bound to the exact supplied codec-freeze file and seal"
        )
    validation_path: Path | None = None
    validation_payload: bytes | None = None
    validation_file_hash: str | None = None
    validation_internal: str | None = None
    if args.protocol_mode == "blind":
        validation_path = codec_freeze_path.with_name("codec_freeze.validation.json")
        validation, validation_payload, validation_file_hash = load_json_bytes(
            validation_path, "codec-freeze validation receipt"
        )
        if not common.verify_internal_seal(validation):
            raise ValueError("codec-freeze validation receipt has an invalid internal seal")
        validation_internal = str(validation.get("lock_sha256"))
        expected_validation_keys = {
            "schema",
            "status",
            "passed",
            "freeze_path",
            "freeze_file_sha256",
            "freeze_internal_lock_sha256",
            "executing_validator_sha256",
            "frozen_artifact_count",
            "development_pooled_relative_mse",
            "gaussian_mse_reference",
            "physical_bits",
            "physical_bpw",
            "preaccess_state",
            "lock_sha256",
        }
        frozen_artifacts = codec_freeze.get("frozen_artifact_sha256s")
        development_mse = float(
            validation.get("development_pooled_relative_mse", float("nan"))
        )
        if (
            set(validation) != expected_validation_keys
            or validation.get("schema")
            != "strata_xklt_sc_v2_codec_freeze_validation_v1"
            or validation.get("status") != "validated_before_blind_source_access"
            or validation.get("passed") is not True
            or validation.get("freeze_path")
            != "blind_protocol_v2/codec_freeze.lock.json"
            or validation.get("freeze_file_sha256") != codec_freeze_file_hash
            or validation.get("freeze_internal_lock_sha256")
            != codec_freeze.get("lock_sha256")
            or not isinstance(frozen_artifacts, dict)
            or validation.get("executing_validator_sha256")
            != frozen_artifacts.get("freeze_validator")
            or int(validation.get("frozen_artifact_count", -1))
            != len(frozen_artifacts)
            or not math.isfinite(development_mse)
            or development_mse >= math.exp2(-4.3)
            or validation.get("gaussian_mse_reference") != math.exp2(-4.3)
            or int(validation.get("physical_bits", -1)) != common.PHYSICAL_BITS
            or validation.get("physical_bpw")
            != common.PHYSICAL_BITS / common.WEIGHTS
            or validation.get("preaccess_state") != codec_freeze.get("preaccess_state")
        ):
            raise ValueError("codec-freeze validation receipt contract mismatch")
        expected_validation_binding = {
            "file_sha256": validation_file_hash,
            "internal_lock_sha256": validation_internal,
        }
        source_validation = source_lock.get("codec_freeze_validation")
        if (
            not isinstance(source_validation, dict)
            or set(source_validation) != set(expected_validation_binding)
            or source_validation != expected_validation_binding
        ):
            raise ValueError(
                "source lock is not bound to the exact codec-freeze validation receipt"
            )
    if len(selection.get("matrices", [])) != 18 or len(source_lock.get("matrices", [])) != 18:
        raise ValueError("selection and source lock must each contain eighteen matrices")
    for ordinal, matrix in enumerate(selection["matrices"]):
        blocks = matrix.get("blocks", [])
        block_hashes = [block.get("source_bf16_sha256") for block in blocks]
        if (
            len(blocks) != 6
            or matrix.get("source_bf16_sha256") is not None
            or any(digest is not None for digest in block_hashes)
        ):
            raise ValueError(
                "selection proposal must contain six null-hash metadata blocks at "
                f"ordinal {ordinal}"
            )
    if (
        int(source_lock.get("matrix_count", -1)) != 18
        or int(source_lock.get("block_count", -1)) != 108
        or int(source_lock.get("source_values", -1)) != common.WEIGHTS
        or int(source_lock.get("source_bytes", -1)) != 2 * common.WEIGHTS
    ):
        raise ValueError("source-finalization lock has wrong 18/108/value/byte totals")
    for ordinal, matrix in enumerate(source_lock["matrices"]):
        blocks = matrix.get("blocks", [])
        hashes = [matrix.get("source_bf16_sha256")] + [
            block.get("source_bf16_sha256") for block in blocks
        ]
        block_geometry_ok = all(
            int(block.get("canonical_block_index", -1)) == index
            and int(block.get("nvalues", -1)) == 1 << 18
            and int(block.get("nbytes", -1)) == 1 << 19
            for index, block in enumerate(blocks)
        )
        if len(blocks) != 6 or not block_geometry_ok or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in hashes
        ):
            raise ValueError(
                f"source-finalization hashes are incomplete at ordinal {ordinal}"
            )
    if selection.get("checkpoint") != source_lock.get("checkpoint"):
        raise ValueError("selection and source lock checkpoint metadata differ")
    source_root = (
        args.source_root.resolve()
        if args.source_root is not None
        else (source_lock_path.parent / source_lock.get("source_root", ".")).resolve()
    )
    emitter_path = Path(__file__).resolve()
    common_path = Path(common.__file__).resolve()
    bindings = {
        "protocol_mode": args.protocol_mode,
        "selection_lock": {
            "path": str(selection_path),
            "file_sha256": selection_file_hash,
            "internal_lock_sha256": selection_internal,
        },
        "route": {
            "path": str(route_path),
            "sha256": hashlib.sha256(route).hexdigest(),
            "bytes": len(route),
        },
        "source_lock": {
            "path": str(source_lock_path),
            "file_sha256": source_lock_file_hash,
            "internal_lock_sha256": source_lock.get("lock_sha256"),
            "selection_lock_sha256": source_lock.get("selection_lock_sha256"),
        },
        "codec_freeze": {
            "path": str(codec_freeze_path),
            "file_sha256": codec_freeze_file_hash,
            "internal_lock_sha256": codec_freeze.get("lock_sha256"),
        },
        "format_freeze": {
            "path": str(format_path),
            "sha256": hashlib.sha256(format_payload).hexdigest(),
        },
        "emitter": {
            "path": str(emitter_path),
            "sha256": common.sha256_file(emitter_path),
        },
        "common": {
            "path": str(common_path),
            "sha256": common.sha256_file(common_path),
        },
    }
    if args.protocol_mode == "blind":
        assert validation_path is not None
        assert validation_file_hash is not None
        assert validation_internal is not None
        bindings["codec_freeze_validation"] = {
            "path": str(validation_path),
            "file_sha256": validation_file_hash,
            "internal_lock_sha256": validation_internal,
        }
    if args.protocol_mode == "blind":
        frozen_artifacts = codec_freeze.get("frozen_artifact_sha256s")
        if not isinstance(frozen_artifacts, dict):
            raise ValueError("blind codec freeze has no frozen_artifact_sha256s map")
        exact_blind_inputs = {
            "selection_lock_file_sha256": selection_file_hash,
            "route_file_sha256": bindings["route"]["sha256"],
        }
        unbound_inputs = {
            field: digest
            for field, digest in exact_blind_inputs.items()
            if codec_freeze.get(field) != digest
        }
        if unbound_inputs:
            raise ValueError(
                "blind codec freeze does not bind exact selection/route bytes: "
                f"{unbound_inputs}"
            )
        required_current_digests = {
            "emitter": bindings["emitter"]["sha256"],
            "common": bindings["common"]["sha256"],
            "format": bindings["format_freeze"]["sha256"],
        }
        mismatched = {
            name: {"actual": digest, "frozen": frozen_artifacts.get(name)}
            for name, digest in required_current_digests.items()
            if digest != frozen_artifacts.get(name)
        }
        if mismatched:
            raise ValueError(
                "blind codec freeze named emitter bindings differ: "
                f"{mismatched}"
            )
    bindings["_immutability_checks"] = [
        (selection_path, hashlib.sha256(selection_payload).hexdigest()),
        (source_lock_path, hashlib.sha256(source_lock_payload).hexdigest()),
        (codec_freeze_path, hashlib.sha256(codec_freeze_payload).hexdigest()),
        (route_path, hashlib.sha256(route).hexdigest()),
        (format_path, hashlib.sha256(format_payload).hexdigest()),
        (emitter_path, bindings["emitter"]["sha256"]),
        (common_path, bindings["common"]["sha256"]),
    ]
    if args.protocol_mode == "blind":
        assert validation_path is not None and validation_payload is not None
        bindings["_immutability_checks"].append(
            (validation_path, hashlib.sha256(validation_payload).hexdigest())
        )
    return selection, source_lock, route, source_root, bindings


def validate_matrix_bindings(
    selection: dict, source_lock: dict, route_rows: list[dict], source_root: Path
) -> list[dict[str, Any]]:
    expected_shapes = {
        "gate": [768, common.GROUP_VALUES],
        "up": [768, common.GROUP_VALUES],
        "down": [common.GROUP_VALUES, 768],
    }
    source_by_ordinal = {int(row["matrix_ordinal"]): row for row in source_lock["matrices"]}
    if len(source_by_ordinal) != 18:
        raise ValueError("source lock matrix ordinals are duplicated")
    result = []
    for ordinal, (selected, route) in enumerate(zip(selection["matrices"], route_rows)):
        source = source_by_ordinal.get(ordinal)
        if source is None:
            raise ValueError(f"source lock omits matrix ordinal {ordinal}")
        tensor = selected.get("tensor", selected.get("canonical_tensor_id"))
        source_tensor = source.get("tensor", source.get("canonical_tensor_id"))
        selected_role = role_name(str(selected["role"]))
        selected_tensor_layer, selected_tensor_expert, selected_tensor_role = (
            parse_tensor_identity(tensor)
        )
        source_tensor_layer, source_tensor_expert, source_tensor_role = (
            parse_tensor_identity(source_tensor)
        )
        required_source_identity = {"role", "layer", "expert", "block_count"}
        blind_source = source_lock.get("schema") == BLIND_SOURCE_CONTRACT[0]
        if blind_source and not required_source_identity.issubset(source):
            missing = sorted(required_source_identity - set(source))
            raise ValueError(
                f"source lock matrix ordinal {ordinal} omits finalized fields: {missing}"
            )
        # The historical development-v1 receipt predates redundant identity
        # fields. Blind-v2 is fail-closed; development alone may derive them
        # from the canonical tensor id already sealed in that source receipt.
        source_role = role_name(str(source.get("role", source_tensor_role)))
        source_layer = int(source.get("layer", source_tensor_layer))
        source_expert = int(source.get("expert", source_tensor_expert))
        selected_shape = [int(value) for value in selected["shape"]]
        source_shape = [int(value) for value in source["shape"]]
        expected_shape = expected_shapes.get(selected_role)
        expected_values = 768 * common.GROUP_VALUES
        if selection.get("schema") == BLIND_SELECTION_CONTRACT[0]:
            selected_relpath = selected.get("future_output_relpath")
            if "output_relpath" in selected:
                raise ValueError(
                    f"blind proposal uses non-future output path key ordinal {ordinal}"
                )
        else:
            selected_relpath = selected.get("output_relpath")
        selected_shard = selected.get("shard")
        selected_range = selected.get("absolute_http_byte_range_inclusive")
        source_range = source.get("http_range_inclusive")
        range_ok = (
            isinstance(selected_range, list)
            and len(selected_range) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in selected_range)
            and int(selected_range[0]) >= 0
            and int(selected_range[1]) >= int(selected_range[0])
            and source_range == selected_range
            and int(selected_range[1]) - int(selected_range[0]) + 1
            == 2 * expected_values
        )
        http_response = source.get("http_response")
        blind_http_response_ok = not blind_source
        if blind_source and range_ok and isinstance(http_response, dict):
            begin, end = (int(selected_range[0]), int(selected_range[1]))
            content_range_match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)",
                str(http_response.get("content_range", "")),
            )
            checkpoint = selection.get("checkpoint", {})
            expected_url = (
                f"https://huggingface.co/{checkpoint.get('repo')}/resolve/"
                f"{checkpoint.get('revision')}/{selected_shard}"
            )
            blind_http_response_ok = (
                set(http_response)
                == {
                    "status",
                    "request_url",
                    "requested_range",
                    "content_range",
                    "content_length",
                    "content_encoding",
                    "body_bytes",
                    "body_sha256",
                }
                and int(http_response.get("status", -1)) == 206
                and http_response.get("request_url") == expected_url
                and http_response.get("requested_range") == f"bytes={begin}-{end}"
                and int(http_response.get("content_length", -1)) == end - begin + 1
                and http_response.get("content_encoding") == "identity"
                and int(http_response.get("body_bytes", -1)) == end - begin + 1
                and http_response.get("body_sha256")
                == source.get("source_bf16_sha256")
                and content_range_match is not None
                and int(content_range_match.group(1)) == begin
                and int(content_range_match.group(2)) == end
                and int(content_range_match.group(3)) > end
            )
        checks = {
            "selection_ordinal": int(selected["matrix_ordinal"]) == ordinal,
            "source_ordinal": int(source["matrix_ordinal"]) == ordinal,
            "tensor": tensor == source_tensor,
            "selection_tensor_identity": (
                selected_tensor_layer == int(selected["layer"])
                and selected_tensor_expert == int(selected["expert"])
                and selected_tensor_role == selected_role
            ),
            "source_tensor_identity": (
                source_tensor_layer == source_layer
                and source_tensor_expert == source_expert
                and source_tensor_role == source_role
            ),
            "source_selection_identity": (
                source_layer == int(selected["layer"])
                and source_expert == int(selected["expert"])
                and source_role == selected_role
            ),
            "shape_binding": selected_shape == source_shape,
            "canonical_role_shape": selected_shape == expected_shape,
            "route_role": selected_role == route["role"],
            "route_layer": int(selected["layer"]) == int(route["layer"]),
            "route_expert": int(selected["expert"]) == int(route["expert"]),
            "dtype": str(selected.get("dtype", "BF16")).upper() == "BF16"
            and str(source.get("dtype", "")).upper() == "BF16",
            "weights": int(np.prod(selected_shape)) == expected_values,
            "selection_nvalues": int(selected.get("nvalues", expected_values))
            == expected_values,
            "source_nvalues": int(source.get("nvalues", -1))
            == expected_values,
            "selection_nbytes": int(selected.get("nbytes", 2 * expected_values))
            == 2 * expected_values,
            "source_nbytes": int(source.get("nbytes", -1))
            == 2 * expected_values,
            "source_block_count": int(source.get("block_count", len(source.get("blocks", []))))
            == 6,
            "source_shard": isinstance(selected_shard, str)
            and bool(selected_shard)
            and source.get("shard") == selected_shard,
            "source_http_range": range_ok,
            "blind_http_206_receipt": blind_http_response_ok,
            "output_relpath": selected_relpath == source.get("output_relpath"),
        }
        if not all(checks.values()):
            raise ValueError(f"matrix binding mismatch ordinal {ordinal}: {checks}")
        path = (source_root / source["output_relpath"]).resolve()
        if path == source_root or source_root not in path.parents:
            raise ValueError(f"source path escapes source root ordinal {ordinal}: {path}")
        if not path.is_file():
            raise ValueError(f"source file is absent ordinal {ordinal}: {path}")
        if path.stat().st_size != 2 * expected_values:
            raise ValueError(f"source byte size mismatch ordinal {ordinal}")
        digest = source.get("source_bf16_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid locked source digest ordinal {ordinal}")
        result.append(
            {
                "matrix_ordinal": ordinal,
                "tensor": tensor,
                "role": selected_role,
                "layer": int(route["layer"]),
                "expert": int(route["expert"]),
                "axis": route["axis"],
                "groups": int(route["groups"]),
                "shape": source_shape,
                "source_path": path,
                "source_relpath": source["output_relpath"],
                "source_bf16_sha256": digest,
                "source_block_sha256s": [
                    block["source_bf16_sha256"] for block in source["blocks"]
                ],
                "shard": selected_shard,
                "http_range_inclusive": source_range,
                **(
                    {"http_response": http_response}
                    if blind_source
                    else {}
                ),
            }
        )
    return result


def load_source_words(meta: dict[str, Any]) -> np.ndarray:
    """Read once, then authenticate the exact bytes returned to the staging path."""
    payload = meta["source_path"].read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != meta["source_bf16_sha256"]:
        raise ValueError(
            f"source hash mismatch ordinal {meta['matrix_ordinal']}: "
            f"{meta['source_path']}"
        )
    expected_bytes = 2 * int(np.prod(meta["shape"]))
    if len(payload) != expected_bytes:
        raise ValueError(f"source byte size changed ordinal {meta['matrix_ordinal']}")
    chunk_bytes = (1 << 18) * 2
    actual_blocks = [
        hashlib.sha256(payload[offset : offset + chunk_bytes]).hexdigest()
        for offset in range(0, len(payload), chunk_bytes)
    ]
    if actual_blocks != meta["source_block_sha256s"]:
        raise ValueError(
            f"source block hashes mismatch ordinal {meta['matrix_ordinal']}"
        )
    return np.frombuffer(payload, dtype="<u2").reshape(meta["shape"])


def build_staging(
    matrices: list[dict[str, Any]], output_dir: Path
) -> tuple[list[tuple[np.float32, np.float32]], list[int], bytes, list[dict], np.ndarray, list[dict]]:
    canonical_groups: list[cp.ndarray] = []
    coefficient_pairs: list[tuple[np.float32, np.float32]] = []
    angle_codes: list[int] = []
    klt_rows: list[dict] = []
    source_energy = 0.0
    staged_energy = 0.0
    for triplet in range(6):
        gate_meta, up_meta, down_meta = matrices[3 * triplet : 3 * triplet + 3]
        loaded = []
        for meta in (gate_meta, up_meta, down_meta):
            raw = load_source_words(meta)
            loaded.append(natural_words(raw, meta["axis"]))
        gate_words, up_words, down_words = loaded
        gate = bf16_words_to_gpu(gate_words)
        up = bf16_words_to_gpu(up_words)
        down = bf16_words_to_gpu(down_words)
        if gate.shape != (768, common.GROUP_VALUES) or up.shape != gate.shape or down.shape != gate.shape:
            raise ValueError(f"natural triplet geometry mismatch {triplet}")
        up64, down64 = up.astype(cp.float64), down.astype(cp.float64)
        a = float(cp.sum(up64 * up64, dtype=cp.float64).get())
        b = float(cp.sum(down64 * down64, dtype=cp.float64).get())
        cross = float(cp.sum(up64 * down64, dtype=cp.float64).get())
        code, cosine, sine = common.derive_klt(a, b, cross)
        component0, component1 = fp32_klt_gpu(up, down, cosine, sine)
        component0_words = fp32_gpu_to_bf16_rne(component0)
        component1_words = fp32_gpu_to_bf16_rne(component1)
        component0_bf = bf16_words_to_gpu(component0_words).astype(cp.float64)
        component1_bf = bf16_words_to_gpu(component1_words).astype(cp.float64)
        gate_words_gpu = cp.asarray(gate_words, dtype=cp.uint16)
        canonical_groups.extend((gate_words_gpu, component0_words, component1_words))

        # Independent CPU parity at every specified rounding point.  This is
        # deliberately audited during emission; it is not used to choose the
        # result or repair a mismatch.
        up_cpu = common.bf16_to_fp32(up_words)
        down_cpu = common.bf16_to_fp32(down_words)
        cpu0, cpu1 = fp32_klt_cpu(up_cpu, down_cpu, cosine, sine)
        cpu0_words = common.fp32_to_bf16_rne(cpu0)
        cpu1_words = common.fp32_to_bf16_rne(cpu1)
        gpu0_words = cp.asnumpy(component0_words)
        gpu1_words = cp.asnumpy(component1_words)
        if not np.array_equal(cpu0_words, gpu0_words) or not np.array_equal(cpu1_words, gpu1_words):
            raise AssertionError(f"CPU/CuPy FP32+BF16 KLT parity failed triplet {triplet}")
        up_cpu64, down_cpu64 = up_cpu.astype(np.float64), down_cpu.astype(np.float64)
        a_cpu = float(np.sum(up_cpu64 * up_cpu64, dtype=np.float64))
        b_cpu = float(np.sum(down_cpu64 * down_cpu64, dtype=np.float64))
        cross_cpu = float(np.sum(up_cpu64 * down_cpu64, dtype=np.float64))
        code_cpu, cosine_cpu, sine_cpu = common.derive_klt(a_cpu, b_cpu, cross_cpu)
        if (code_cpu, cosine_cpu.tobytes(), sine_cpu.tobytes()) != (
            code, cosine.tobytes(), sine.tobytes()
        ):
            raise AssertionError(f"CPU/CuPy reduction changes Q15 KLT triplet {triplet}")
        coefficient_pairs.append((cosine, sine))
        angle_codes.append(code)
        gate64 = gate.astype(cp.float64)
        gate_energy = float(cp.sum(gate64 * gate64, dtype=cp.float64).get())
        source_energy += gate_energy + a + b
        staged_triplet_energy = gate_energy + float(
            cp.sum(component0_bf * component0_bf, dtype=cp.float64).get()
            + cp.sum(component1_bf * component1_bf, dtype=cp.float64).get()
        )
        staged_energy += staged_triplet_energy
        klt_rows.append(
            {
                "triplet": triplet,
                "gate_tensor": gate_meta["tensor"],
                "up_tensor": up_meta["tensor"],
                "down_tensor": down_meta["tensor"],
                "energy_up_fp64": a,
                "energy_down_fp64": b,
                "cross_inner_product_fp64": cross,
                "rho_fp64": cross / math.sqrt(a * b),
                "angle_code_q15_pi": code,
                "cosine_fp32": float(cosine),
                "sine_fp32": float(sine),
                "cosine_squared_plus_sine_squared": float(cosine) ** 2
                + float(sine) ** 2,
                "staged_triplet_energy_fp64": staged_triplet_energy,
                "cupy_cpu_bf16_staging_parity": True,
                "cupy_cpu_q15_code_parity": True,
                "fp64_reduction_cpu_minus_gpu": {
                    "energy_up": a_cpu - a,
                    "energy_down": b_cpu - b,
                    "cross": cross_cpu - cross,
                },
                "component0_bf16_sha256": hashlib.sha256(
                    gpu0_words.astype("<u2", copy=False).tobytes()
                ).hexdigest(),
                "component1_bf16_sha256": hashlib.sha256(
                    gpu1_words.astype("<u2", copy=False).tobytes()
                ).hexdigest(),
            }
        )

    canonical = cp.concatenate(canonical_groups, axis=0)
    if canonical.shape != (common.GROUPS, common.GROUP_VALUES):
        raise AssertionError(f"canonical staging geometry changed: {canonical.shape}")
    values = bf16_words_to_gpu(canonical).astype(cp.float64)
    group_energy = cp.sum(values * values, axis=1, dtype=cp.float64)
    ordinal = cp.arange(common.GROUPS, dtype=cp.int64)
    rank = cp.empty_like(ordinal)
    energy_order = cp.lexsort(
        cp.stack((ordinal.astype(cp.float64), group_energy), axis=0)
    )
    rank[energy_order] = ordinal
    labels_gpu = cp.minimum(7, rank * 8 // common.GROUPS).astype(cp.uint8)
    labels = cp.asnumpy(labels_gpu)
    if np.bincount(labels, minlength=8).tolist() != [1728] * 8:
        raise AssertionError("stratum labels are not exactly equipopulous")
    label_bytes = common.pack_labels(labels)
    permutation_gpu = cp.lexsort(
        cp.stack((ordinal, labels_gpu.astype(cp.int64)), axis=0)
    )
    permutation = cp.asnumpy(permutation_gpu)
    # Explicit ordinal secondary keys make GPU ordering deterministic even
    # when energies or labels tie.  Audit parity against NumPy lexsort.
    group_energy_cpu = cp.asnumpy(group_energy)
    ordinal_cpu = np.arange(common.GROUPS, dtype=np.int64)
    energy_order_cpu = np.lexsort((ordinal_cpu, group_energy_cpu))
    if not np.array_equal(cp.asnumpy(energy_order), energy_order_cpu):
        raise AssertionError("CuPy/NumPy energy+ordinal ordering mismatch")
    if not np.array_equal(permutation, np.lexsort((ordinal_cpu, labels))):
        raise AssertionError("CuPy/NumPy label+ordinal ordering mismatch")
    staging_dir = output_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=False)
    block_rows = []
    cursor = 0
    block_energy = np.empty(14, dtype=np.float64)
    for block, (logn, groups) in enumerate(zip(common.BLOCK_LOG2, common.BLOCK_GROUPS)):
        selected_ordinals = permutation[cursor : cursor + groups]
        selected = np.ascontiguousarray(
            cp.asnumpy(canonical[cp.asarray(selected_ordinals)]), dtype="<u2"
        )
        path = staging_dir / f"block_{block:02d}_n{logn}.bf16.bin"
        selected.tofile(path)
        block_energy[block] = float(
            cp.sum(group_energy[cp.asarray(selected_ordinals)], dtype=cp.float64).get()
        )
        block_rows.append(
            {
                "block_ordinal": block,
                "block_log2": logn,
                "values": 1 << logn,
                "groups": groups,
                "sorted_group_begin": cursor,
                "sorted_group_end_exclusive": cursor + groups,
                "source_energy_fp64": float(block_energy[block]),
                "staging_relpath": str(path.relative_to(output_dir)).replace("\\", "/"),
                "staging_bytes": path.stat().st_size,
                "staging_sha256": common.sha256_file(path),
                "selected_group_ordinals_sha256": hashlib.sha256(
                    selected_ordinals.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
            }
        )
        cursor += groups
    if cursor != common.GROUPS:
        raise AssertionError("mixed geometry did not consume every group")
    energy_audit = [
        {
            "original_source_energy_fp64": source_energy,
            "staged_klt_bf16_energy_fp64": staged_energy,
            "relative_staging_energy_drift": (staged_energy - source_energy) / source_energy,
            "sum_block_energy_fp64": float(block_energy.sum(dtype=np.float64)),
        }
    ]
    return coefficient_pairs, angle_codes, label_bytes, klt_rows, block_energy, block_rows + energy_audit


def emit_candidate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    selection, source_lock, route, source_root, bindings = load_documents(args)
    route_rows = common.parse_route(route)
    matrices = validate_matrix_bindings(selection, source_lock, route_rows, source_root)
    coefficients, angle_codes, labels, klt_rows, block_energy, combined_rows = build_staging(
        matrices, output_dir
    )
    block_rows = combined_rows[:14]
    energy_audit = combined_rows[14]
    profiles, allocation = common.allocate_profiles(block_energy)
    profile_bytes = profiles.tobytes()
    header = common.build_header(coefficients, angle_codes, route, labels)
    seeds = [
        common.derive_seeds(header, route, labels, profile_bytes, ordinal)
        for ordinal in range(14)
    ]
    for row, profile, seed_row in zip(block_rows, profiles, seeds):
        sc_seed, rht_seed, digest = seed_row
        q = int(profile)
        row.update(
            {
                "profile_id": q,
                "nominal_rate_bpw": common.PROFILE_BASE + q / 256.0,
                "test_distortion": math.pow(
                    2.0, -2.0 * (common.PROFILE_BASE + q / 256.0)
                ),
                "sc_seed_u32": sc_seed,
                "rht_seed_u64": rht_seed,
                "seed_digest_sha256": digest,
            }
        )

    assets = {
        "header.bin": header,
        "route.bin": route,
        "labels_3bit.bin": labels,
        "profiles.bin": profile_bytes,
    }
    asset_rows = {}
    for filename, payload in assets.items():
        path = output_dir / filename
        path.write_bytes(payload)
        asset_rows[filename] = {
            "bytes": len(payload),
            "sha256": common.sha256_file(path),
        }

    immutability_checks = bindings.pop("_immutability_checks")
    for path, expected_digest in immutability_checks:
        actual_digest = common.sha256_file(path)
        if actual_digest != expected_digest:
            raise RuntimeError(f"frozen input changed during emission: {path}")
    source_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"source_path"}
        }
        for row in matrices
    ]
    bindings["sources"] = source_rows
    manifest = {
        "schema": "strata_xklt_sc_v2_preencoding_manifest_v1",
        "status": "complete_and_allocation_sealed_before_encoding",
        "claim_boundary": "pre-encoding staging only; no arithmetic encoder has run",
        "protocol_mode": args.protocol_mode,
        "bindings": bindings,
        "physical_format": {
            "physical_bytes": common.PHYSICAL_BYTES,
            "physical_bits": common.PHYSICAL_BITS,
            "physical_bpw": common.PHYSICAL_BITS / common.WEIGHTS,
            "integer_2p15_cap_bits": common.INTEGER_CAP_BITS,
            "headroom_bits": common.INTEGER_CAP_BITS - common.PHYSICAL_BITS,
            "reservoir_bits": common.RESERVOIR_BYTES * 8,
            "global_blind_reserve_bits": common.GLOBAL_RESERVE_BITS,
        },
        "assets": asset_rows,
        "klt": {
            "derivation": "Q15-over-pi angle; stored FP32 cos/sin; FP32 transform; BF16-RNE staging",
            "rows": klt_rows,
            "energy_audit": energy_audit,
        },
        "strata": {
            "label_histogram": [1728] * 8,
            "stable_order": "(label, canonical_group_ordinal)",
        },
        "allocation": allocation,
        "blocks": block_rows,
        "coverage": {
            "matrices": 18,
            "groups": common.GROUPS,
            "weights": common.WEIGHTS,
            "every_weight_staged_once": True,
        },
    }
    manifest_path = output_dir / "preencoding_manifest.json"
    common.write_json(manifest_path, manifest)
    lock = common.sealed(
        {
            "schema": "strata_xklt_sc_v2_allocation_lock_v1",
            "status": "allocation_sealed_before_first_encoder_invocation",
            "protocol_mode": args.protocol_mode,
            "manifest_sha256": common.sha256_file(manifest_path),
            "bindings": bindings,
            "assets": asset_rows,
            "physical_format": manifest["physical_format"],
            "allocation": allocation,
            "blocks": block_rows,
            "nondeterministic_or_postencoding_fields_excluded": [
                "runtime", "GPU", "logs", "logical lengths", "decoder scales",
                "encoder outputs", "distortion measurements",
            ],
        }
    )
    lock_path = output_dir / "allocation.lock.json"
    common.write_json(lock_path, lock)
    receipt = {
        "status": "STRATA-v2 staging and allocation lock complete",
        "output_dir": str(output_dir),
        "preencoding_manifest_sha256": common.sha256_file(manifest_path),
        "allocation_lock_file_sha256": common.sha256_file(lock_path),
        "allocation_lock_internal_sha256": lock["lock_sha256"],
        "profile_ids": profiles.astype(int).tolist(),
        "projected_relative_mse": allocation["projected_relative_mse"],
        "encoder_invocations": 0,
        "protocol_mode": args.protocol_mode,
    }
    common.write_json(output_dir / "EMISSION_RECEIPT.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-lock", type=Path, required=True)
    ap.add_argument("--route", type=Path, required=True)
    ap.add_argument("--source-lock", type=Path, required=True)
    ap.add_argument("--source-root", type=Path)
    ap.add_argument(
        "--protocol-mode", choices=("development", "blind"), required=True
    )
    ap.add_argument("--codec-freeze", type=Path, required=True)
    ap.add_argument("--format", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    return ap


def main() -> None:
    receipt = emit_candidate(parser().parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
