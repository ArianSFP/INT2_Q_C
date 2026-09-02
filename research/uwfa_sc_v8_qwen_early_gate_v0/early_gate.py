#!/usr/bin/env python3
"""Source-auditable Qwen early-kill runner for sealed UWFA-SC v8.

This is deliberately not a claim-authority dispatcher.  It authenticates and
executes the exact sealed v8 source snapshots, runs the exact source-free
all-150 and representative CuPy gates before touching the Qwen artifact, then
uses the fixed STRATA adapter and the unchanged v8 ``source_phase``.  It never
runs controls and can emit only an explicitly nonpromoting diagnostic.

Importing this module performs no path access, NumPy/CuPy import, CUDA
initialization, subprocess launch, or payload discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile
import types
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "uwfa-sc-v8-qwen-early-gate-v0"
AUTHORIZATION = "RUN_EXACT_QWEN_EARLY_KILL_NO_CONTROLS_NO_CLAIM_V0"
SEALED_V8_MANIFEST_SHA256 = "a54593c13a864a28d2797faf360321cf3cce5b834292aff013ca8eff175c68b6"
STRATA_COMMON_SHA256 = "3f085c9531b714d0d7877388f54ae50495dc3ea631491563abceb4db55608fd1"
FROZEN_AUDITOR_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"
CURRENT_ARTIFACT_SHA256 = "4842d0754156d8ad1e174199dd211396346ffa9b5472f7278c41f2f30691405b"
CURRENT_ARTIFACT_BYTES = 8_847_360
SOURCE_WEIGHTS = 28_311_552
BASELINE_PLAN_SHA256 = "8017582201468300dd07550a1a2f8d90dc704ffae7ae6d8801a560178e4a1868"
AUDITED_RELATIVE_MSE = 0.030902167403153148
AUDITED_SSE_FP64 = 500.39553685426534
AUDITED_SOURCE_ENERGY_FP64 = 16192.89450885593
MAX_SOURCE_MEMBER_BYTES = 2 * (1 << 20)
MAX_RESULT_JSON_BYTES = 256 * (1 << 20)

V8_REQUIRED_MEMBERS = (
    "INDEPENDENT_BOOTSTRAP_ABI.md",
    "README.md",
    "container_codec.py",
    "cupy_backend.py",
    "design_lock.json",
    "dispatcher_contract.py",
    "fixture_long_memory.py",
    "fixture_portability.py",
    "protocol.py",
    "result_envelope.py",
    "run_source_free_gpu_dev.py",
    "stage0_census.py",
    "strata_sc_adapter.py",
    "test_source_only.py",
    "universal_adapter.py",
    "uwfa_common.py",
    "verify_source.py",
)


class EarlyGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EarlyGateError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def pretty_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def strict_json(data: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise EarlyGateError(f"nonfinite JSON constant: {value}")

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    except EarlyGateError:
        raise
    except Exception as exc:
        raise EarlyGateError(f"invalid JSON: {exc}") from exc
    require(isinstance(value, dict), "JSON root must be object")
    return value


def require_digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64, f"{label} digest geometry")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise EarlyGateError(f"{label} digest encoding") from exc
    require(len(raw) == 32, f"{label} digest width")
    return value.lower()


def require_absolute_lexical(path: Path, label: str) -> Path:
    text = os.fspath(path)
    require(os.path.isabs(text), f"{label} path must be absolute")
    drive, tail = os.path.splitdrive(text)
    parts = tuple(part for part in tail.replace("\\", "/").split("/") if part)
    require(parts and all(part not in {".", ".."} for part in parts), f"{label} path noncanonical")
    return Path(text)


def reject_symlink_chain(path: Path, label: str) -> None:
    path = require_absolute_lexical(path, label)
    cursor = Path(path.anchor)
    for component in path.parts[1:]:
        cursor = cursor / component
        require(os.path.lexists(cursor), f"{label} component absent: {cursor}")
        info = os.lstat(cursor)
        require(not stat.S_ISLNK(info.st_mode), f"{label} symlink component: {cursor}")


class HeldRegularInput:
    """Bounded no-follow regular input retained through its consumer lifetime."""

    def __init__(
        self,
        path: Path,
        *,
        label: str,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
        maximum_bytes: int,
    ) -> None:
        self.path = require_absolute_lexical(path, label)
        reject_symlink_chain(self.path, label)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(os.fspath(self.path), flags)
        try:
            info = os.fstat(self.fd)
            require(stat.S_ISREG(info.st_mode), f"{label} is not regular")
            require(0 <= info.st_size <= maximum_bytes, f"{label} byte bound")
            if expected_bytes is not None:
                require(info.st_size == expected_bytes, f"{label} exact bytes")
            self.identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            chunks: list[bytes] = []
            while chunk := os.read(self.fd, 1 << 20):
                chunks.append(chunk)
            self.data = b"".join(chunks)
            require(len(self.data) == info.st_size, f"{label} short read")
            self.sha256 = sha256(self.data)
            if expected_sha256 is not None:
                require(self.sha256 == require_digest(expected_sha256, label), f"{label} digest")
            self.verify_stable()
        except Exception:
            os.close(self.fd)
            self.fd = -1
            raise

    def verify_stable(self) -> None:
        require(self.fd >= 0, "closed held input")
        info = os.fstat(self.fd)
        require(
            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) == self.identity,
            f"held input changed: {self.path}",
        )

    def close(self) -> None:
        if self.fd >= 0:
            self.verify_stable()
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "HeldRegularInput":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def load_snapshot_module(name: str, source: bytes, expected_sha256: str) -> types.ModuleType:
    require(sha256(source) == require_digest(expected_sha256, name), f"{name} snapshot digest")
    require(name not in sys.modules, f"preloaded snapshot module: {name}")
    module = types.ModuleType(name)
    module.__file__ = f"<authenticated-early-gate:{name}>"
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(source, module.__file__, "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def normalize_strata_group_ordinal_abi(strata_common: Any) -> dict[str, Any]:
    """Bridge NumPy integer scalars to the adapter's frozen Python-int ABI.

    ``expected_block_group_ordinals`` in the pinned STRATA format source
    returns NumPy arrays, while the sealed v8 adapter deliberately accepts
    exact built-in ``int`` ordinals only.  Converting each already-validated
    ordinal changes neither its value nor its ordering.  This bridge is part
    of this nonpromoting runner and is committed into the decoder-bundle hash;
    it is not represented as execution of the sealed producer unchanged.
    """
    original = getattr(strata_common, "expected_block_group_ordinals", None)
    require(callable(original), "STRATA expected-block-group entrypoint")

    def normalized(labels: Any) -> list[tuple[int, ...]]:
        rows = original(labels)
        require(isinstance(rows, list), "STRATA group rows")
        converted: list[tuple[int, ...]] = []
        for row in rows:
            require(hasattr(row, "__iter__"), "STRATA group row iterable")
            values = tuple(int(value) for value in row)
            require(all(type(value) is int for value in values), "STRATA group Python-int ABI")
            converted.append(values)
        return converted

    strata_common.expected_block_group_ordinals = normalized
    receipt = {
        "schema": "uwfa-sc-v8-qwen-early-gate-strata-group-ordinal-abi-v0",
        "status": "EXPLORATORY_VALUE_PRESERVING_NUMPY_INTEGER_TO_PYTHON_INT",
        "operation": "for every group ordinal emitted by the pinned STRATA helper, apply built-in int without reordering",
        "positive_claim_authority": False,
    }
    receipt["receipt_sha256"] = sha256(canonical_json(receipt))
    return receipt


def authenticate_v8_package(package: Path) -> dict[str, Any]:
    package = require_absolute_lexical(package, "v8 package")
    reject_symlink_chain(package, "v8 package")
    require(stat.S_ISDIR(os.lstat(package).st_mode), "v8 package must be directory")
    manifest_path = package / "SOURCE_MANIFEST.json"
    with HeldRegularInput(
        manifest_path,
        label="v8 manifest",
        expected_sha256=SEALED_V8_MANIFEST_SHA256,
        maximum_bytes=1 << 20,
    ) as held_manifest:
        manifest_bytes = held_manifest.data
    manifest = strict_json(manifest_bytes)
    require(
        set(manifest) == {"schema", "status", "members", "access_attestation", "post_freeze_requirements"},
        "v8 manifest fields",
    )
    require(manifest["schema"] == "unifilar-wfa-source-manifest-v8", "v8 manifest schema")
    require(manifest["status"] == "SEALED_SOURCE_ONLY_NO_PAYLOAD_AUTHORITY", "v8 manifest status")
    rows = manifest["members"]
    require(isinstance(rows, list) and len(rows) == len(V8_REQUIRED_MEMBERS), "v8 member rows")
    names = [row.get("name") if isinstance(row, dict) else None for row in rows]
    require(names == list(V8_REQUIRED_MEMBERS), "v8 members must use exact UTF-8 order")
    actual = {entry.name for entry in os.scandir(package)}
    require(actual == set(V8_REQUIRED_MEMBERS) | {"SOURCE_MANIFEST.json"}, "v8 undeclared/missing members")
    snapshots: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"name", "bytes", "sha256"}, "v8 manifest row")
        name = str(row["name"])
        require(type(row["bytes"]) is int and 0 < row["bytes"] <= MAX_SOURCE_MEMBER_BYTES, f"v8 member bytes: {name}")
        expected = require_digest(row["sha256"], f"v8 member {name}")
        with HeldRegularInput(
            package / name,
            label=f"v8 member {name}",
            expected_sha256=expected,
            expected_bytes=int(row["bytes"]),
            maximum_bytes=MAX_SOURCE_MEMBER_BYTES,
        ) as held:
            snapshots[name] = held.data
            hashes[name] = held.sha256
    return {
        "package": package,
        "manifest_bytes": manifest_bytes,
        "manifest_sha256": sha256(manifest_bytes),
        "manifest": manifest,
        "snapshots": snapshots,
        "member_hashes": hashes,
        "source_snapshot_root_sha256": sha256(canonical_json(rows)),
    }


def load_v8_modules(closure: Mapping[str, Any]) -> dict[str, Any]:
    source = closure["snapshots"]
    hashes = closure["member_hashes"]
    return {
        "common": load_snapshot_module("uwfa_sc_v8_eg_common", source["uwfa_common.py"], hashes["uwfa_common.py"]),
        "protocol": load_snapshot_module("uwfa_sc_v8_eg_protocol", source["protocol.py"], hashes["protocol.py"]),
        "semantic": load_snapshot_module("uwfa_sc_v8_eg_semantic", source["universal_adapter.py"], hashes["universal_adapter.py"]),
        "codec": load_snapshot_module("uwfa_sc_v8_eg_codec", source["container_codec.py"], hashes["container_codec.py"]),
        "stage": load_snapshot_module("uwfa_sc_v8_eg_stage", source["stage0_census.py"], hashes["stage0_census.py"]),
        "backend_source": load_snapshot_module("uwfa_sc_v8_eg_backend", source["cupy_backend.py"], hashes["cupy_backend.py"]),
        "adapter_source": load_snapshot_module("uwfa_sc_v8_eg_adapter", source["strata_sc_adapter.py"], hashes["strata_sc_adapter.py"]),
    }


def load_external_decoder_sources(strata_common_path: Path, frozen_auditor_path: Path) -> dict[str, Any]:
    with HeldRegularInput(
        strata_common_path,
        label="STRATA common source",
        expected_sha256=STRATA_COMMON_SHA256,
        maximum_bytes=MAX_SOURCE_MEMBER_BYTES,
    ) as held_common:
        common_bytes = held_common.data
    with HeldRegularInput(
        frozen_auditor_path,
        label="frozen independent auditor source",
        expected_sha256=FROZEN_AUDITOR_SHA256,
        maximum_bytes=MAX_SOURCE_MEMBER_BYTES,
    ) as held_auditor:
        auditor_bytes = held_auditor.data
    strata_common = load_snapshot_module("uwfa_sc_v8_eg_strata_common", common_bytes, STRATA_COMMON_SHA256)
    frozen_auditor = load_snapshot_module("uwfa_sc_v8_eg_frozen_auditor", auditor_bytes, FROZEN_AUDITOR_SHA256)
    group_ordinal_abi = normalize_strata_group_ordinal_abi(strata_common)
    return {
        "strata_common": strata_common,
        "frozen_auditor": frozen_auditor,
        "group_ordinal_abi": group_ordinal_abi,
        "source_hashes": {
            "strata_expert_local_codec_common_sha256": sha256(common_bytes),
            "strata_v2_klt_mixed_independent_auditor_sha256": sha256(auditor_bytes),
        },
    }


def independent_gpu_identity(common: Any, protocol: Any) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    require(len(rows) == 1, "independent GPU identity row count")
    columns = [item.strip() for item in rows[0].split(",")]
    require(len(columns) == 3, "independent GPU identity column count")
    name, uuid, pci = columns
    record = {
        "schema": "uwfa-sc-v8-independent-gpu-identity",
        "status": "PASS_INDEPENDENT_GPU_IDENTITY",
        "device_uuid": protocol.canonical_gpu_uuid(uuid, "independent GPU UUID"),
        "pci_bus_id": protocol.canonical_pci_bus_id(pci, "independent GPU PCI bus id"),
        "device_name": name,
        "provider": "nvidia-smi",
    }
    record["identity_receipt_sha256"] = sha256(common.canonical_json(record))
    return record


def run_source_free_preflight(modules: Mapping[str, Any], backend: Any, source_root: str) -> tuple[Any, dict[str, Any]]:
    common = modules["common"]
    protocol = modules["protocol"]
    stage = modules["stage"]
    identity = independent_gpu_identity(common, protocol)
    all150 = stage.gpu_preflight_all_150(common, backend, source_root)
    representative = stage.representative_outer_fold_benchmark(
        common,
        protocol,
        modules["codec"],
        modules["semantic"],
        backend,
        source_root,
    )
    bound = {
        "schema": "uwfa-sc-v8-bound-source-preflight",
        "source_snapshot_root_sha256": source_root,
        "all150": all150,
        "representative": representative,
        "independent_gpu_identity": identity,
    }
    receipt_sha = sha256(common.canonical_json(bound))
    typed = stage.SourcePreflightEvidence(all150, representative, identity, receipt_sha)
    return typed, {**bound, "receipt_sha256": receipt_sha}


def build_score_receipt(
    common: Any,
    *,
    artifact_sha256: str,
    reconstruction_sha256: str,
    full_geometry_sha256: str,
    decoder_bundle_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    expected = AUDITED_SSE_FP64 / AUDITED_SOURCE_ENERGY_FP64
    require(abs(expected - AUDITED_RELATIVE_MSE) <= 4.0 * math.ulp(expected), "audited score constants disagree")
    clean = {
        "schema": "uwfa-bound-baseline-score-v8",
        "status": "PASS_INDEPENDENT_BASELINE_SCORE",
        "artifact_sha256": require_digest(artifact_sha256, "artifact"),
        "artifact_bytes": CURRENT_ARTIFACT_BYTES,
        "weights": SOURCE_WEIGHTS,
        "relative_mse": AUDITED_RELATIVE_MSE,
        "sse_fp64": AUDITED_SSE_FP64,
        "source_energy_fp64": AUDITED_SOURCE_ENERGY_FP64,
        "normalization": "FP64_SSE_SUM_DIVIDED_BY_FP64_SOURCE_ENERGY_SUM",
        "reconstruction_f64_sha256": require_digest(reconstruction_sha256, "reconstruction"),
        "original_source_panel_sha256": require_digest(full_geometry_sha256, "full geometry"),
        "independent_decoder_source_sha256": require_digest(decoder_bundle_sha256, "decoder bundle"),
    }
    receipt = dict(clean)
    receipt["score_receipt_sha256"] = sha256(canonical_json(clean))
    encoded = common.pretty_json(receipt)
    return receipt, encoded


def descriptor_source_builder(codec: Any):
    class OwnedAuthenticatedDescriptorSource(codec.AuthenticatedDescriptorSource):
        """Keep the temporary-file owner alive for the authenticated source."""

        def __init__(self, owner: Any, expected_sha256: str) -> None:
            self._early_gate_owner = owner
            try:
                super().__init__(owner.fileno(), expected_sha256)
            except Exception:
                owner.close()
                self._early_gate_owner = None
                raise

        def close(self) -> None:
            owner = self._early_gate_owner
            self._early_gate_owner = None
            try:
                super().close()
            finally:
                if owner is not None:
                    owner.close()

    def build(raw: bytes) -> Any:
        handle = tempfile.TemporaryFile(prefix="uwfa-v8-early-gate-")
        try:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            handle.seek(0)
            return OwnedAuthenticatedDescriptorSource(handle, sha256(raw))
        except Exception:
            handle.close()
            raise

    return build


def fraction_from_record(record: Mapping[str, Any]) -> Fraction:
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": f"{value.numerator}/{value.denominator}",
        "float": float(value),
    }


def bandwidth_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    maximum_page = Fraction(0, 1)
    maximum_repeated = Fraction(0, 1)
    maximum_coalesced = Fraction(0, 1)
    for row in metrics["experts"]:
        total = fraction_from_record(row["attributable_total_physical_bytes"])
        nonpadding = fraction_from_record(row["attributable_nonpadding_decodable_bytes"])
        page = max(Fraction(int(row["touched_page_bytes"]), 1) / total, Fraction(int(row["touched_page_bytes"]), 1) / nonpadding)
        repeated_bytes = int(row["instrumented_routed_requested_bytes_with_repetition"])
        unique_bytes = int(row["instrumented_routed_unique_requested_bytes"])
        repeated = max(Fraction(repeated_bytes, 1) / total, Fraction(repeated_bytes, 1) / nonpadding)
        coalesced = max(Fraction(unique_bytes, 1) / total, Fraction(unique_bytes, 1) / nonpadding)
        maximum_page = max(maximum_page, page)
        maximum_repeated = max(maximum_repeated, repeated)
        maximum_coalesced = max(maximum_coalesced, coalesced)
        rows.append({
            "expert_ordinal": int(row["expert_ordinal"]),
            "descriptor_backed_unique_page_ratio_strict": fraction_record(page),
            "requested_with_repetition_ratio_strict": fraction_record(repeated),
            "ideal_coalesced_unique_requested_ratio_strict": fraction_record(coalesced),
            "touched_page_bytes": int(row["touched_page_bytes"]),
            "requested_bytes_with_repetition": repeated_bytes,
            "unique_requested_bytes": unique_bytes,
            "overlap_bytes_requested_again": int(row["instrumented_routed_overlap_bytes_requested_again"]),
            "read_request_count": int(row["instrumented_routed_read_request_count"]),
            "causal_decode_reencode_reconstruction": row["causal_decode_reencode_reconstruction"],
            "passes_descriptor_backed_unique_page_below_2x": page < 2,
            "passes_requested_with_repetition_below_2x": repeated < 2,
            "passes_ideal_coalesced_unique_requested_below_2x": coalesced < 2,
            "passes_all_reported_bandwidth_ratios_below_2x": page < 2 and repeated < 2 and coalesced < 2,
        })
    return {
        "definition": {
            "unique_page": "union of descriptor-backed 4096-byte pages touched divided by the stricter owner-local denominator",
            "requested_with_repetition": "sum of literal read-call lengths including overlap divided by the stricter owner-local denominator",
            "ideal_coalesced_unique_requested": "union of requested byte intervals divided by the stricter owner-local denominator; diagnostic, not the frozen cold gate",
        },
        "experts": rows,
        "maximum_descriptor_backed_unique_page_ratio_strict": fraction_record(maximum_page),
        "maximum_requested_with_repetition_ratio_strict": fraction_record(maximum_repeated),
        "maximum_ideal_coalesced_unique_requested_ratio_strict": fraction_record(maximum_coalesced),
        "passes_frozen_unique_page_below_2x": maximum_page < 2,
        "passes_strict_requested_with_repetition_below_2x": maximum_repeated < 2,
        "passes_strict_ideal_coalesced_unique_requested_below_2x": maximum_coalesced < 2,
        "passes_all_reported_bandwidth_ratios_below_2x": (
            maximum_page < 2 and maximum_repeated < 2 and maximum_coalesced < 2
        ),
    }


def compact_component_rows(scientific: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in scientific.get("folds", []):
        result.append({
            "component_ordinal": row.get("outer_dependence_component_ordinal"),
            "identity_indices": row.get("outer_identity_indices"),
            "identities": row.get("outer_identities_from_artifact"),
            "test_stream_ordinals": row.get("test_stream_ordinals"),
            "allocated_test_weights": row.get("allocated_test_weights"),
            "selected": row.get("selected_by_inner_validation_only"),
            "literal_baseline_bits": row.get("literal_authenticated_current_baseline_container_bits"),
            "literal_candidate_bits": row.get("literal_candidate_container_bits"),
            "literal_model_aligned_increment_bits": row.get("literal_selected_model_aligned_increment_bits"),
            "literal_saved_bits": row.get("literal_test_saving_after_exact_container_delta_bits"),
            "exact_saving_bpw": row.get("exact_test_saving_bpw"),
        })
    return result


def write_exclusive(dir_fd: int, name: str, payload: bytes) -> dict[str, Any]:
    require(name == Path(name).name and name not in {"", ".", ".."}, "output member name")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
    try:
        cursor = 0
        while cursor < len(payload):
            written = os.write(fd, payload[cursor:])
            require(written > 0, "short output write")
            cursor += written
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"name": name, "bytes": len(payload), "sha256": sha256(payload)}


def publish(output_dir: Path, members: Mapping[str, bytes], *, source_root: str, status: str) -> dict[str, Any]:
    output_dir = require_absolute_lexical(output_dir, "output directory")
    parent = output_dir.parent
    reject_symlink_chain(parent, "output parent")
    require(stat.S_ISDIR(os.lstat(parent).st_mode), "output parent must be directory")
    os.mkdir(output_dir, 0o700)
    dir_fd = os.open(os.fspath(output_dir), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        rows = []
        for name in sorted(members, key=lambda value: value.encode("utf-8")):
            rows.append(write_exclusive(dir_fd, name, members[name]))
        complete = {
            "schema": "uwfa-sc-v8-qwen-early-gate-completion-v0",
            "status": status,
            "positive_claim_authority": False,
            "controls_run": False,
            "source_snapshot_root_sha256": source_root,
            "members": rows,
        }
        complete["completion_sha256"] = sha256(canonical_json(complete))
        completion_row = write_exclusive(dir_fd, "COMPLETE.json", pretty_json(complete))
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return {"output_dir": os.fspath(output_dir), "members": rows + [completion_row], "completion": complete}


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    require(arguments.authorization == AUTHORIZATION, "explicit early-gate authorization token")
    # Path strings are checked lexically here without touching the Qwen leaf.
    artifact_path = require_absolute_lexical(Path(arguments.artifact), "artifact")
    output_dir = require_absolute_lexical(Path(arguments.output_dir), "output directory")

    closure = authenticate_v8_package(Path(arguments.v8_package))
    modules = load_v8_modules(closure)

    # Numerical libraries and CUDA are imported only after the sealed v8 source
    # authenticates.  The Qwen artifact has still not been statted or opened.
    import numpy as np
    import cupy as cp

    external = load_external_decoder_sources(Path(arguments.strata_common), Path(arguments.frozen_auditor))
    backend = modules["backend_source"].build_backend(cp)
    typed_preflight, preflight_record = run_source_free_preflight(
        modules, backend, closure["source_snapshot_root_sha256"]
    )

    # This is the sole first Qwen access point.  It occurs after both exact
    # source-free CuPy gates and independent GPU identity validation.
    with HeldRegularInput(
        artifact_path,
        label="authenticated current Qwen artifact",
        expected_sha256=CURRENT_ARTIFACT_SHA256,
        expected_bytes=CURRENT_ARTIFACT_BYTES,
        maximum_bytes=CURRENT_ARTIFACT_BYTES,
    ) as held_artifact:
        artifact_bytes = held_artifact.data
        artifact_identity = {
            "st_dev": held_artifact.identity[0],
            "st_ino": held_artifact.identity[1],
            "bytes": held_artifact.identity[2],
            "mtime_ns": held_artifact.identity[3],
            "sha256": held_artifact.sha256,
        }

    adapter = modules["adapter_source"].StrataSCAdapter(
        common=modules["common"],
        semantic_codec=modules["semantic"],
        np=np,
        frozen_auditor=external["frozen_auditor"],
        strata_common=external["strata_common"],
        device="cupy",
    )
    panel = modules["stage"].prepare_panel(modules["protocol"], adapter, artifact_bytes)
    full_geometry = modules["protocol"].geometry_sha256(modules["common"], panel)
    structural_geometry = modules["protocol"].structural_geometry_sha256(modules["common"], panel)
    reconstruction_sha = str(panel["reconstruction"]["full_reconstruction_f64_sha256"])

    decoder_bundle = {
        "schema": "uwfa-sc-v8-qwen-early-gate-decoder-bundle-v0",
        "members": [
            {"name": "strata_expert_local_codec/common.py", "sha256": STRATA_COMMON_SHA256},
            {"name": "strata_v2_klt_mixed_independent_auditor_v1.py", "sha256": FROZEN_AUDITOR_SHA256},
            {"name": "strata_sc_adapter.py", "sha256": closure["member_hashes"]["strata_sc_adapter.py"]},
            {"name": "universal_adapter.py", "sha256": closure["member_hashes"]["universal_adapter.py"]},
            {"name": "container_codec.py", "sha256": closure["member_hashes"]["container_codec.py"]},
        ],
        "exploratory_semantic_bridge": external["group_ordinal_abi"],
    }
    decoder_bundle_sha = sha256(canonical_json(decoder_bundle))
    score, score_bytes = build_score_receipt(
        modules["common"],
        artifact_sha256=artifact_identity["sha256"],
        reconstruction_sha256=reconstruction_sha,
        full_geometry_sha256=full_geometry,
        decoder_bundle_sha256=decoder_bundle_sha,
    )
    modules["protocol"].validate_score_receipt(
        score,
        artifact_sha256=artifact_identity["sha256"],
        artifact_bytes=CURRENT_ARTIFACT_BYTES,
        weights=SOURCE_WEIGHTS,
        reconstruction_sha256=reconstruction_sha,
        original_source_panel_sha256=full_geometry,
        independent_decoder_source_sha256=decoder_bundle_sha,
    )

    runner_sha = sha256(Path(__file__).read_bytes())
    pipeline_record = {
        "schema": "uwfa-sc-v8-qwen-early-gate-pipeline-v0",
        "sealed_v8_manifest_sha256": closure["manifest_sha256"],
        "source_snapshot_root_sha256": closure["source_snapshot_root_sha256"],
        "decoder_bundle_sha256": decoder_bundle_sha,
        "runner_sha256": runner_sha,
        "baseline_plan_sha256": BASELINE_PLAN_SHA256,
    }
    pipeline_sha = sha256(canonical_json(pipeline_record))
    bindings = modules["stage"].BoundEvidence(
        baseline_plan_sha256=BASELINE_PLAN_SHA256,
        baseline_score_sha256=sha256(score_bytes),
        universal_decoder_sha256=decoder_bundle_sha,
        producer_manifest_sha256=closure["manifest_sha256"],
        audit_bootstrap_sha256=runner_sha,
        source_full_geometry_sha256=full_geometry,
        source_structural_geometry_sha256=structural_geometry,
        extraction_program_sha256=closure["member_hashes"]["strata_sc_adapter.py"],
        universal_adapter_sha256=closure["member_hashes"]["universal_adapter.py"],
        pipeline_sha256=pipeline_sha,
        source_snapshot_root_sha256=closure["source_snapshot_root_sha256"],
        source_preflight_receipt_sha256=preflight_record["receipt_sha256"],
    )
    source_result = modules["stage"].source_phase(
        common=modules["common"],
        protocol=modules["protocol"],
        container_codec=modules["codec"],
        semantic_codec=modules["semantic"],
        adapter=adapter,
        backend=backend,
        artifact_bytes=artifact_bytes,
        score_receipt_bytes=score_bytes,
        bindings=bindings,
        source_preflight=typed_preflight,
        authenticated_descriptor_source_builder=descriptor_source_builder(modules["codec"]),
    )
    source_status = str(source_result["status"])
    public_source = {key: value for key, value in source_result.items() if not key.startswith("_")}
    scientific = public_source.get("scientific_nested_holdout", {})
    physical = public_source.get("source_final")
    metrics = physical.get("parsed_metrics") if isinstance(physical, dict) else None
    compact_physical = None
    bandwidth = None
    canonical = None
    if isinstance(metrics, dict):
        bandwidth = bandwidth_summary(metrics)
        standalone = physical["standalone_decode"]
        compact_physical = {
            "container_bytes": metrics["actual_container_bytes"],
            "physical_rate_bpw": metrics["actual_physical_rate_bpw"],
            "physical_rate_rational": metrics["actual_physical_rate_rational"],
            "relative_mse": metrics["audited_identical_reconstruction_relative_mse"],
            "F": metrics["F_from_actual_bytes_and_identical_reconstruction"],
            "net_physical_saving_bpw": metrics["net_physical_saving_bpw"],
            "passes_rate_interval": metrics["passes_rate_interval"],
            "passes_F_target": metrics["passes_F_target"],
            "passes_descriptor_backed_cold_below_2x": metrics["passes_cold_read_below_2x"],
            "container_sha256": physical["container_sha256"],
            "identity_framing_container_sha256": physical["identity_framing_container_sha256"],
            "model_packet_sha256": physical["model_packet_sha256"],
        }
        canonical = {
            "standalone_all_payloads_canonically_reencoded": standalone["all_payloads_canonically_reencoded"],
            "standalone_all_three_roles_reconstructed": standalone["all_three_roles_reconstructed"],
            "full_reconstruction_f64_sha256": standalone["reconstruction"]["full_reconstruction_f64_sha256"],
            "matches_recomputed_panel_reconstruction": standalone["reconstruction"]["full_reconstruction_f64_sha256"] == reconstruction_sha,
            "routed_full_reconstruction": metrics["routed_full_reconstruction"],
            "literal_container_canonical_rebuild_was_enforced_by_exact_v8_final_container": True,
        }

    wrapper_status = (
        "EARLY_DIAGNOSTIC_SOURCE_SURVIVOR_REQUIRES_CONTROLS_AND_INDEPENDENT_AUDIT"
        if source_status == "SOURCE_SURVIVOR_CONTROLS_AUTHORIZED_NOT_YET_OPENED"
        else f"EARLY_DIAGNOSTIC_{source_status}"
    )
    environment = backend.environment_receipt()
    result = {
        "schema": SCHEMA,
        "status": wrapper_status,
        "underlying_exact_v8_source_status": source_status,
        "positive_claim_authority": False,
        "controls_run": False,
        "controls_may_not_be_inferred_or_added": True,
        "claim_boundary": "early-kill Qwen diagnostic using exact sealed v8 source; never a positive compression claim",
        "binding_authority_disclosure": {
            "status": "EARLY_DIAGNOSTIC_LOCAL_BINDINGS_NOT_PRODUCTION_DISPATCHER_AUTHORITY",
            "externally_pinned_dispatcher_receipt_present": False,
            "baseline_score_receipt": "constructed locally from the fixed audited D/SSE/energy and recomputed artifact identities",
            "decoder_bundle_sha256": "canonical aggregate of the exact hash-pinned decoder source members",
            "audit_bootstrap_sha256": "self-reported hash of this unsealed exploratory runner",
            "pipeline_sha256": "canonical aggregate constructed by this exploratory runner",
            "positive_claim_use_permitted": False,
        },
        "artifact_identity": artifact_identity,
        "baseline_score": score,
        "source_full_geometry_sha256": full_geometry,
        "source_structural_geometry_sha256": structural_geometry,
        "recomputed_panel_reconstruction_f64_sha256": reconstruction_sha,
        "winner": scientific.get("final_topology_selected_from_nested_fold_votes") if isinstance(scientific, dict) else None,
        "pooled_exact_heldout_saving_bpw": scientific.get("pooled_exact_heldout_saving_bpw") if isinstance(scientific, dict) else None,
        "per_dependence_component_saving": compact_component_rows(scientific) if isinstance(scientific, dict) else [],
        "physical": compact_physical,
        "bandwidth": bandwidth,
        "canonical_decode_reencode": canonical,
        "source_preflight_receipt_sha256": preflight_record["receipt_sha256"],
        "telemetry": environment,
        "bindings": bindings.container_hashes(),
        "decoder_bundle": decoder_bundle,
        "decoder_bundle_sha256": decoder_bundle_sha,
        "pipeline_record": pipeline_record,
        "pipeline_sha256": pipeline_sha,
        "source_hashes": {
            "sealed_v8_manifest_sha256": closure["manifest_sha256"],
            "sealed_v8_source_snapshot_root_sha256": closure["source_snapshot_root_sha256"],
            "sealed_v8_members": closure["member_hashes"],
            **external["source_hashes"],
            "early_gate_runner_sha256": runner_sha,
        },
        "exact_v8_source_result": public_source,
    }
    result_bytes = pretty_json(result)
    require(len(result_bytes) <= MAX_RESULT_JSON_BYTES, "result JSON bound")
    output_members: dict[str, bytes] = {
        "RESULT.json": result_bytes,
        "BOUND_BASELINE_SCORE.json": score_bytes,
        "SOURCE_PREFLIGHT.json": pretty_json(preflight_record),
        "DECODER_BUNDLE.json": pretty_json(decoder_bundle),
    }
    if "_container" in source_result:
        output_members["UWFCV8.bin"] = bytes(source_result["_container"])
    if "_identity_framing_container" in source_result:
        output_members["IDENTITY_FRAMING.bin"] = bytes(source_result["_identity_framing_container"])
    publication = publish(
        output_dir,
        output_members,
        source_root=closure["source_snapshot_root_sha256"],
        status=wrapper_status,
    )
    return {
        "schema": "uwfa-sc-v8-qwen-early-gate-launch-summary-v0",
        "status": wrapper_status,
        "positive_claim_authority": False,
        "controls_run": False,
        "output_dir": publication["output_dir"],
        "result_sha256": sha256(result_bytes),
        "container_sha256": compact_physical["container_sha256"] if compact_physical else None,
        "physical": compact_physical,
        "winner": result["winner"],
        "pooled_exact_heldout_saving_bpw": result["pooled_exact_heldout_saving_bpw"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--authorization", required=True)
    result.add_argument("--v8-package", required=True)
    result.add_argument("--strata-common", required=True)
    result.add_argument("--frozen-auditor", required=True)
    result.add_argument("--artifact", required=True)
    result.add_argument("--output-dir", required=True)
    return result


def main() -> int:
    arguments = parser().parse_args()
    summary = run(arguments)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL_UWFA_SC_V8_QWEN_EARLY_GATE: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
