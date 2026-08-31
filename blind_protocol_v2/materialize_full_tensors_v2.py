#!/usr/bin/env python3
"""One-shot materializer for the sealed Qwen v2 evaluation panel.

This administrative program is intentionally created only after the codec
freeze and its fresh-process validation receipt exist.  It has no tensor
selection logic and accepts no URL, tensor, range, or output-path override.
Exactly the eighteen ranges already sealed in the proposal are requested from
the pinned checkpoint revision.  A request is accepted only as an HTTP 206
response with the exact Content-Range/Content-Length and identity encoding.

All payloads and the finalized source lock are first written beneath one
private temporary directory.  The complete directory is atomically renamed to
``blind_protocol_v2/unblinded`` only after every matrix and nested block hash
has been computed and the source lock has been internally sealed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SELECTION_FILE_SHA256 = "528250d8c6bac52dfdf64958d7f4929a115ff68d907a47880cab85d532aade14"
SELECTION_INTERNAL_SHA256 = "cd8cb70ca7509d2ddd4899df8a7047b7b8f47d381b637e2eb497db9ecd4eb9f8"
AUTHORIZATION_PHRASE = "AUTHORIZE SEALED QWEN V2 ONE-SHOT MATERIALIZATION"
WEIGHTS = 28_311_552
PHYSICAL_BITS = 60_869_832
MATRICES = 18
BLOCKS_PER_MATRIX = 6
MATRIX_VALUES = 1_572_864
MATRIX_BYTES = 3_145_728
BLOCK_VALUES = 262_144
BLOCK_BYTES = 524_288
CANONICAL_SCRIPT = "blind_protocol_v2/materialize_full_tensors_v2.py"
CANONICAL_SELECTION = "blind_protocol_v2/selection.proposal.lock.json"
CANONICAL_FREEZE = "blind_protocol_v2/codec_freeze.lock.json"
CANONICAL_VALIDATION = "blind_protocol_v2/codec_freeze.validation.json"
CANONICAL_FINAL_ROOT = "blind_protocol_v2/unblinded"
EXPECTED_SOURCE_LOCK = {
    "schema": "int2-qwen-blind-source-finalization-v2",
    "status": "all_locked_sources_materialized_and_hash_finalized",
    "matrix_count": 18,
    "block_count": 108,
    "source_values": WEIGHTS,
    "source_bytes": 2 * WEIGHTS,
    "dtype": "BF16",
    "exact_codec_freeze_validation_binding_required": True,
    "required_matrix_fields": [
        "matrix_ordinal",
        "tensor",
        "role",
        "layer",
        "expert",
        "dtype",
        "shape",
        "nvalues",
        "nbytes",
        "block_count",
        "shard",
        "http_range_inclusive",
        "http_response",
        "output_relpath",
        "source_bf16_sha256",
        "blocks",
    ],
    "required_http_response_fields": [
        "status",
        "request_url",
        "requested_range",
        "content_range",
        "content_length",
        "content_encoding",
        "body_bytes",
        "body_sha256",
    ],
    "required_block_fields": [
        "canonical_block_index",
        "nvalues",
        "nbytes",
        "source_bf16_sha256",
    ],
}


class MaterializationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_exact(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value, payload, sha256_bytes(payload)


def verify_seal(value: dict[str, Any], label: str) -> str:
    clean = dict(value)
    declared = clean.pop("lock_sha256", None)
    actual = sha256_bytes(canonical_bytes(clean))
    require(declared == actual, f"{label} internal seal mismatch")
    return actual


def seal(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    require("lock_sha256" not in result, "attempted to seal an already sealed object")
    result["lock_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def create_only_json(path: Path, value: Any) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        payload = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def safe_output(root: Path, relative: str) -> Path:
    relpath = Path(relative)
    require(
        relative not in ("", ".")
        and not relpath.is_absolute()
        and ".." not in relpath.parts,
        f"unsafe sealed output path: {relative!r}",
    )
    path = (root / relpath).resolve()
    path.relative_to(root.resolve())
    return path


def path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any destination entry."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 is required for no-replace publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), os.fspath(destination))


def safe_cleanup_directory(path: Path, parent: Path, identity: tuple[int, int]) -> None:
    """Best-effort cleanup that never resolves or follows a replaced root entry."""
    try:
        require(path.parent == parent, "cleanup path parent changed")
        require(path.name.startswith(".materialize_v2_"), "unsafe cleanup path name")
        current = os.lstat(path)
        require(stat.S_ISDIR(current.st_mode), "cleanup root is no longer a directory")
        require(not stat.S_ISLNK(current.st_mode), "cleanup root became a symlink")
        require((current.st_dev, current.st_ino) == identity, "cleanup root identity changed")
        shutil.rmtree(path)
    except BaseException:
        # Preserve the originating materialization exception.  An untrusted
        # replacement entry is deliberately left untouched for manual audit.
        return


def safe_remove_lock(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = os.lstat(path)
        if (current.st_dev, current.st_ino) == identity and stat.S_ISREG(
            current.st_mode
        ):
            os.unlink(path)
    except BaseException:
        return


def revalidate_staged_tree(
    temporary: Path,
    temporary_identity: tuple[int, int],
    source_lock: dict[str, Any],
    source_lock_file_hash: str,
) -> None:
    root_stat = os.lstat(temporary)
    require(
        stat.S_ISDIR(root_stat.st_mode)
        and not stat.S_ISLNK(root_stat.st_mode)
        and (root_stat.st_dev, root_stat.st_ino) == temporary_identity,
        "staging root identity changed",
    )
    source_lock_path = temporary / "source_hashes.lock.json"
    require(not source_lock_path.is_symlink(), "staged source lock became a symlink")
    loaded, _, loaded_hash = load_exact(source_lock_path, "staged source lock")
    verify_seal(loaded, "staged source lock")
    require(
        loaded == source_lock and loaded_hash == source_lock_file_hash,
        "staged source lock bytes changed",
    )
    expected_files = {Path("source_hashes.lock.json")}
    for matrix in source_lock["matrices"]:
        relpath = Path(str(matrix["output_relpath"]))
        expected_files.add(relpath)
        path = safe_output(temporary, str(relpath))
        require(not path.is_symlink() and path.is_file(), f"staged source type: {relpath}")
        require(path.stat().st_size == int(matrix["nbytes"]), f"staged size: {relpath}")
        payload = path.read_bytes()
        require(
            sha256_bytes(payload) == matrix["source_bf16_sha256"]
            == matrix["http_response"]["body_sha256"],
            f"staged matrix hash: {relpath}",
        )
        require(
            len(payload) == matrix["http_response"]["body_bytes"],
            f"staged HTTP-body length: {relpath}",
        )
        blocks = matrix["blocks"]
        require(len(blocks) == BLOCKS_PER_MATRIX, f"staged block count: {relpath}")
        for block_ordinal, block in enumerate(blocks):
            begin = block_ordinal * BLOCK_BYTES
            end = begin + BLOCK_BYTES
            require(
                int(block["canonical_block_index"]) == block_ordinal
                and int(block["nvalues"]) == BLOCK_VALUES
                and int(block["nbytes"]) == BLOCK_BYTES
                and sha256_bytes(payload[begin:end])
                == block["source_bf16_sha256"],
                f"staged nested block hash: {relpath}:{block_ordinal}",
            )
    observed_files: set[Path] = set()
    for current_root, directory_names, filenames in os.walk(
        temporary, topdown=True, followlinks=False
    ):
        current_path = Path(current_root)
        for name in directory_names:
            child = current_path / name
            require(not child.is_symlink(), f"staged directory symlink: {child}")
        for name in filenames:
            child = current_path / name
            require(not child.is_symlink(), f"staged file symlink: {child}")
            observed_files.add(child.relative_to(temporary))
    require(observed_files == expected_files, "unexpected or missing staged files")


def validate_control_chain(
    workspace: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    selection_path = (workspace / CANONICAL_SELECTION).resolve(strict=True)
    freeze_path = (workspace / CANONICAL_FREEZE).resolve(strict=True)
    validation_path = (workspace / CANONICAL_VALIDATION).resolve(strict=True)
    selection, _, selection_file_hash = load_exact(selection_path, "selection proposal")
    freeze, _, freeze_file_hash = load_exact(freeze_path, "codec freeze")
    validation, _, validation_file_hash = load_exact(
        validation_path, "codec-freeze validation receipt"
    )
    selection_internal = verify_seal(selection, "selection proposal")
    freeze_internal = verify_seal(freeze, "codec freeze")
    validation_internal = verify_seal(validation, "codec-freeze validation receipt")
    require(
        selection_file_hash == SELECTION_FILE_SHA256
        and selection_internal == SELECTION_INTERNAL_SHA256,
        "selection proposal bytes/seal differ from the precommitted panel",
    )
    require(
        (selection.get("schema"), selection.get("status"))
        == (
            "int2-qwen-blind-selection-proposal-v2",
            "sealed_metadata_only_proposal_payload_unopened_not_codec_frozen",
        ),
        "selection proposal schema/status mismatch",
    )
    require(
        (freeze.get("schema"), freeze.get("status"))
        == ("strata_xklt_sc_v2_codec_freeze_v1", "frozen_before_blind_source_access")
        and freeze.get("selection_lock_file_sha256") == selection_file_hash
        and freeze.get("selection_lock_sha256") == selection_internal
        and freeze.get("architecture_frozen") is True
        and freeze.get("allocator_frozen") is True
        and freeze.get("no_retry_resume_or_postaccess_tuning") is True
        and freeze.get("blind_materializer_authorization_phrase") == AUTHORIZATION_PHRASE,
        "codec freeze does not authorize the exact sealed panel",
    )
    require(
        freeze.get("physical_rate_limit_bpw") == 2.15
        and freeze.get("gaussian_mse_reference") == math.exp2(-4.3)
        and freeze.get("primary_mse_threshold") == math.exp2(-4.3),
        "codec freeze scientific/rate thresholds mismatch",
    )
    require(
        freeze.get("expected_source_lock") == EXPECTED_SOURCE_LOCK,
        "codec freeze expected-source contract mismatch",
    )
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
    frozen_artifacts = freeze.get("frozen_artifact_sha256s")
    development = freeze.get("development_evidence")
    development_mse = float(
        validation.get("development_pooled_relative_mse", float("nan"))
    )
    require(
        set(validation) == expected_validation_keys
        and validation.get("schema")
        == "strata_xklt_sc_v2_codec_freeze_validation_v1"
        and validation.get("status") == "validated_before_blind_source_access"
        and validation.get("passed") is True
        and validation.get("freeze_path") == CANONICAL_FREEZE
        and validation.get("freeze_file_sha256") == freeze_file_hash
        and validation.get("freeze_internal_lock_sha256") == freeze_internal
        and isinstance(frozen_artifacts, dict)
        and validation.get("executing_validator_sha256")
        == frozen_artifacts.get("freeze_validator")
        and int(validation.get("frozen_artifact_count", -1)) == len(frozen_artifacts)
        and validation.get("preaccess_state") == freeze.get("preaccess_state"),
        "codec-freeze validation receipt mismatch",
    )
    require(
        isinstance(development, dict)
        and math.isfinite(development_mse)
        and development_mse < math.exp2(-4.3)
        and development_mse == development.get("pooled_relative_mse")
        and validation.get("gaussian_mse_reference") == math.exp2(-4.3)
        and int(validation.get("physical_bits", -1)) == PHYSICAL_BITS
        and validation.get("physical_bpw") == PHYSICAL_BITS / WEIGHTS
        and validation.get("physical_bits")
        == freeze.get("physical_ledger", {}).get("physical_bits")
        and validation.get("physical_bpw")
        == freeze.get("physical_ledger", {}).get("physical_bpw"),
        "codec-freeze validation scientific/rate result mismatch",
    )
    require(
        selection.get("checkpoint") == freeze.get("checkpoint"),
        "selection/freeze checkpoint mismatch",
    )
    return selection, freeze, validation, {
        "selection_file_sha256": selection_file_hash,
        "selection_internal_sha256": selection_internal,
        "freeze_file_sha256": freeze_file_hash,
        "freeze_internal_sha256": freeze_internal,
        "validation_file_sha256": validation_file_hash,
        "validation_internal_sha256": validation_internal,
    }


def validate_pre_materialization_state(
    workspace: Path, selection: dict[str, Any], final_root: Path
) -> None:
    proposal_dir = (workspace / "blind_protocol_v2").resolve(strict=True)
    require(
        final_root.parent == proposal_dir and final_root.name == "unblinded",
        "final publication path is noncanonical",
    )
    require(
        not path_lexists(final_root),
        f"final unblinded root entry already exists: {final_root}",
    )
    forbidden = [
        workspace / "blind_protocol_v2/source_hashes.lock.json",
        workspace / "blind_protocol_v2/source_materialization.receipt.json",
    ]
    require(not any(path.exists() for path in forbidden), "legacy source output exists")
    matrices = selection.get("matrices")
    require(isinstance(matrices, list) and len(matrices) == MATRICES, "matrix count")
    require(
        [int(row.get("matrix_ordinal", -1)) for row in matrices]
        == list(range(MATRICES)),
        "matrix ordinals are not canonical",
    )
    expected_paths: set[Path] = set()
    for row in matrices:
        require(row.get("source_bf16_sha256") is None, "proposal leaked matrix hash")
        blocks = row.get("blocks")
        require(
            isinstance(blocks, list)
            and len(blocks) == BLOCKS_PER_MATRIX
            and all(block.get("source_bf16_sha256") is None for block in blocks),
            "proposal leaked nested block hash or has wrong block count",
        )
        relative = Path(str(row.get("future_output_relpath", "")))
        require(
            str(relative) not in ("", ".")
            and not relative.is_absolute()
            and ".." not in relative.parts,
            "unsafe future output path",
        )
        require(relative not in expected_paths, "duplicate future output path")
        expected_paths.add(relative)
    full_shards = [
        path
        for path in workspace.rglob("model-*-of-00016.safetensors")
        if path.is_file()
    ]
    require(not full_shards, f"full checkpoint shard already exists: {full_shards}")


def fetch_exact_range(
    request_url: str, begin: int, end: int, expected_bytes: int
) -> tuple[bytes, dict[str, Any]]:
    request_range = f"bytes={begin}-{end}"
    request = urllib.request.Request(
        request_url,
        headers={
            "Range": request_range,
            "Accept-Encoding": "identity",
            "User-Agent": "INT2-Q-C-sealed-v2-materializer/1",
        },
        method="GET",
    )
    try:
        response = urllib.request.urlopen(request, timeout=180)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise MaterializationError(f"range request failed: {request_url}: {exc}") from exc
    with response:
        status = int(getattr(response, "status", response.getcode()))
        require(status == 206, f"server did not honor Range (HTTP {status})")
        content_range = str(response.headers.get("Content-Range", ""))
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        require(match is not None, f"invalid Content-Range: {content_range!r}")
        assert match is not None
        require(
            int(match.group(1)) == begin
            and int(match.group(2)) == end
            and int(match.group(3)) > end,
            f"wrong Content-Range: {content_range!r}",
        )
        content_length = response.headers.get("Content-Length")
        require(
            content_length is not None and int(content_length) == expected_bytes,
            f"wrong Content-Length: {content_length!r}",
        )
        raw_encoding = response.headers.get("Content-Encoding")
        require(
            raw_encoding is None or raw_encoding.lower() == "identity",
            f"non-identity Content-Encoding: {raw_encoding!r}",
        )
        body = response.read(expected_bytes + 1)
    require(len(body) == expected_bytes, f"wrong HTTP body length: {len(body)}")
    body_hash = sha256_bytes(body)
    return body, {
        "status": 206,
        "request_url": request_url,
        "requested_range": request_range,
        "content_range": content_range,
        "content_length": expected_bytes,
        "content_encoding": "identity",
        "body_bytes": expected_bytes,
        "body_sha256": body_hash,
    }


def materialize(workspace: Path, authorization_phrase: str) -> dict[str, Any]:
    require(
        authorization_phrase == AUTHORIZATION_PHRASE,
        "exact post-freeze materialization authorization phrase required",
    )
    script_path = Path(__file__).resolve(strict=True)
    require(
        script_path == (workspace / CANONICAL_SCRIPT).resolve(strict=True),
        "noncanonical materializer was executed",
    )
    executing_materializer_hash = sha256_file(script_path)
    selection, freeze, _, control = validate_control_chain(workspace)
    proposal_dir = (workspace / "blind_protocol_v2").resolve(strict=True)
    final_root = proposal_dir / "unblinded"
    validate_pre_materialization_state(workspace, selection, final_root)
    checkpoint = selection["checkpoint"]
    repo = str(checkpoint.get("repo", ""))
    revision = str(checkpoint.get("revision", ""))
    require(repo == "Qwen/Qwen3-30B-A3B", "unexpected checkpoint repository")
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "unpinned revision")
    require(
        sha256_file(script_path) == executing_materializer_hash,
        "materializer bytes changed before source access",
    )
    publication_lock = proposal_dir / ".materialize_v2.publish.lock"
    lock_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_descriptor = os.open(publication_lock, lock_flags, 0o400)
    lock_stat = os.fstat(lock_descriptor)
    lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
    temporary: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=".materialize_v2_", dir=proposal_dir))
        temporary_stat = os.lstat(temporary)
        require(
            stat.S_ISDIR(temporary_stat.st_mode)
            and not stat.S_ISLNK(temporary_stat.st_mode),
            "temporary staging root is not a real directory",
        )
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        matrix_rows: list[dict[str, Any]] = []
        total_bytes = 0
        for ordinal, selected in enumerate(selection["matrices"]):
            tensor = str(selected.get("tensor", ""))
            role = str(selected.get("role", ""))
            shape = [int(value) for value in selected.get("shape", [])]
            expected_shape = [2048, 768] if role == "down" else [768, 2048]
            byte_range = selected.get("absolute_http_byte_range_inclusive")
            blocks = selected.get("blocks")
            require(
                int(selected.get("matrix_ordinal", -1)) == ordinal
                and role in {"gate", "up", "down"}
                and shape == expected_shape
                and str(selected.get("dtype", "")).upper() == "BF16"
                and int(selected.get("nvalues", -1)) == MATRIX_VALUES
                and int(selected.get("nbytes", -1)) == MATRIX_BYTES
                and int(selected.get("block_count", -1)) == BLOCKS_PER_MATRIX
                and isinstance(byte_range, list)
                and len(byte_range) == 2
                and isinstance(blocks, list)
                and len(blocks) == BLOCKS_PER_MATRIX,
                f"sealed matrix geometry mismatch ordinal {ordinal}",
            )
            begin, end = int(byte_range[0]), int(byte_range[1])
            require(end - begin + 1 == MATRIX_BYTES, f"sealed range length {ordinal}")
            shard = str(selected.get("shard", ""))
            require(
                re.fullmatch(r"model-\d{5}-of-00016\.safetensors", shard) is not None,
                f"noncanonical shard name ordinal {ordinal}",
            )
            request_url = f"https://huggingface.co/{repo}/resolve/{revision}/{shard}"
            body, http_response = fetch_exact_range(
                request_url, begin, end, MATRIX_BYTES
            )
            relpath = str(selected.get("future_output_relpath", ""))
            output_path = safe_output(temporary, relpath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor = os.open(output_path, flags, 0o444)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    descriptor = -1
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            matrix_hash = sha256_file(output_path)
            require(
                matrix_hash == http_response["body_sha256"],
                f"saved body hash mismatch ordinal {ordinal}",
            )
            block_rows = []
            for block_ordinal in range(BLOCKS_PER_MATRIX):
                offset = block_ordinal * BLOCK_BYTES
                block_payload = body[offset : offset + BLOCK_BYTES]
                require(len(block_payload) == BLOCK_BYTES, "nested block length")
                block_rows.append(
                    {
                        "canonical_block_index": block_ordinal,
                        "nvalues": BLOCK_VALUES,
                        "nbytes": BLOCK_BYTES,
                        "source_bf16_sha256": sha256_bytes(block_payload),
                    }
                )
            matrix_rows.append(
                {
                    "matrix_ordinal": ordinal,
                    "tensor": tensor,
                    "role": role,
                    "layer": int(selected["layer"]),
                    "expert": int(selected["expert"]),
                    "dtype": "BF16",
                    "shape": shape,
                    "nvalues": MATRIX_VALUES,
                    "nbytes": MATRIX_BYTES,
                    "block_count": BLOCKS_PER_MATRIX,
                    "shard": shard,
                    "http_range_inclusive": [begin, end],
                    "http_response": http_response,
                    "output_relpath": relpath,
                    "source_bf16_sha256": matrix_hash,
                    "blocks": block_rows,
                }
            )
            total_bytes += len(body)
            print(
                json.dumps(
                    {
                        "materialized": ordinal + 1,
                        "of": MATRICES,
                        "matrix_ordinal": ordinal,
                        "tensor": tensor,
                        "sha256": matrix_hash,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        require(total_bytes == 2 * WEIGHTS, "aggregate downloaded byte count")
        source_lock = seal(
            {
                "schema": "int2-qwen-blind-source-finalization-v2",
                "status": "all_locked_sources_materialized_and_hash_finalized",
                "dtype": "BF16",
                "checkpoint": checkpoint,
                "selection_lock_sha256": control["selection_internal_sha256"],
                "codec_freeze": {
                    "file_sha256": control["freeze_file_sha256"],
                    "internal_lock_sha256": control["freeze_internal_sha256"],
                },
                "codec_freeze_validation": {
                    "file_sha256": control["validation_file_sha256"],
                    "internal_lock_sha256": control["validation_internal_sha256"],
                },
                "source_root": ".",
                "matrix_count": MATRICES,
                "block_count": MATRICES * BLOCKS_PER_MATRIX,
                "source_values": WEIGHTS,
                "source_bytes": total_bytes,
                "materialization": {
                    "executing_materializer_sha256": executing_materializer_hash,
                    "authorization_phrase_sha256": sha256_bytes(
                        AUTHORIZATION_PHRASE.encode("utf-8")
                    ),
                    "http_range_request_count": MATRICES,
                    "http_206_response_count": MATRICES,
                    "downloaded_body_bytes": total_bytes,
                    "full_checkpoint_shards_written": False,
                    "selection_was_metadata_only": True,
                },
                "matrices": matrix_rows,
            }
        )
        source_lock_path = temporary / "source_hashes.lock.json"
        create_only_json(source_lock_path, source_lock)
        source_lock_file_hash = sha256_file(source_lock_path)
        # Recheck all controls immediately before the atomic publication.
        selection2, freeze2, validation2, control2 = validate_control_chain(workspace)
        require(selection2 == selection and freeze2 == freeze, "control document drift")
        require(control2 == control, "control receipt drift")
        require(
            validation2.get("lock_sha256") == control["validation_internal_sha256"],
            "validation drift",
        )
        require(
            sha256_file(script_path) == executing_materializer_hash,
            "materializer bytes changed during source access",
        )
        assert temporary_identity is not None
        revalidate_staged_tree(
            temporary, temporary_identity, source_lock, source_lock_file_hash
        )
        require(
            not path_lexists(final_root),
            "final unblinded root entry appeared concurrently",
        )
        rename_directory_noreplace(temporary, final_root)
        published_stat = os.lstat(final_root)
        require(
            stat.S_ISDIR(published_stat.st_mode)
            and not stat.S_ISLNK(published_stat.st_mode)
            and (published_stat.st_dev, published_stat.st_ino)
            == temporary_identity,
            "published source-root identity mismatch",
        )
        return {
            "schema": "int2-qwen-blind-materialization-receipt-v2",
            "passed": True,
            "matrix_count": MATRICES,
            "source_bytes": total_bytes,
            "source_lock": str(final_root / "source_hashes.lock.json"),
            "source_lock_file_sha256": source_lock_file_hash,
            "source_lock_internal_sha256": source_lock["lock_sha256"],
            "codec_freeze_file_sha256": control["freeze_file_sha256"],
            "codec_freeze_validation_file_sha256": control[
                "validation_file_sha256"
            ],
            "executing_materializer_sha256": executing_materializer_hash,
        }
    except BaseException:
        if temporary is not None and temporary_identity is not None:
            safe_cleanup_directory(temporary, proposal_dir, temporary_identity)
        raise
    finally:
        try:
            os.close(lock_descriptor)
        except BaseException:
            pass
        safe_remove_lock(publication_lock, lock_identity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--authorization-phrase", required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    receipt = materialize(workspace, args.authorization_phrase)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
