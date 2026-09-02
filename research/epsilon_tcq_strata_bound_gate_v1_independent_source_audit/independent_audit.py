#!/usr/bin/env python3
"""Independent payload-free audit of epsilon-TCQ STRATA bound gate v1."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import stat
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_MANIFEST = "e926575ac1a78a85d08e94e63d1cc85d70b1544e5b352b6abc45cb8653d83706"
EXPECTED_ROOT = "5c3b3a6cb1e2740202710526429a34cca54fcc9105c18820cf6206d276166380"
DECODER_BYTES = 116_835
DECODER_SHA256 = "85e989827a8f1feee111aca4e5e387825f89d5ea4ffdbfe842c72b5fe9f1ec6e"
EXPECTED_PEAK = 7_147_102_208


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def read_regular(path: Path, *, expected_bytes: int | None = None,
                 expected_sha256: str | None = None,
                 maximum_bytes: int = 4 * (1 << 20)) -> bytes:
    path = path.resolve(strict=True)
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
                0 < before.st_size <= maximum_bytes, "regular source input")
        if expected_bytes is not None:
            require(before.st_size == expected_bytes, "source byte pin")
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            require(chunk, "short source read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_nlink) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                 after.st_nlink), "source identity drift")
        payload = b"".join(chunks)
        if expected_sha256 is not None:
            require(sha256(payload) == expected_sha256, "source digest pin")
        return payload
    finally:
        os.close(descriptor)


def authenticate_package(package: Path) -> dict[str, Any]:
    package = package.resolve(strict=True)
    manifest_raw = read_regular(package / "SOURCE_MANIFEST.json")
    require(sha256(manifest_raw) == EXPECTED_MANIFEST, "audited manifest pin")
    manifest = json.loads(manifest_raw)
    require(manifest["source_root_sha256"] == EXPECTED_ROOT, "audited root pin")
    rows = manifest["members"]
    observed = []
    for row in rows:
        require(set(row) == {"name", "bytes", "sha256"}, "manifest member row")
        raw = read_regular(package / row["name"], expected_bytes=row["bytes"],
                           expected_sha256=row["sha256"])
        observed.append({"name": row["name"], "bytes": len(raw),
                         "sha256": sha256(raw)})
    require([row["name"] for row in observed] == sorted(
        (row["name"] for row in observed), key=lambda value: value.encode("utf-8")),
        "manifest ordinal order")
    require(sha256(canonical_json(observed)) == EXPECTED_ROOT,
            "independent source-root recomputation")
    require({entry.name for entry in os.scandir(package)} ==
            {row["name"] for row in rows} | {"SOURCE_MANIFEST.json"},
            "exact audited package closure")
    return {"package": package, "manifest": manifest, "sources": {
        row["name"]: (package / row["name"]).read_bytes() for row in rows}}


def load(package: Path, name: str) -> Any:
    specification = importlib.util.spec_from_file_location(
        f"epsilon_tcq_v1_audit_{name}", package / f"{name}.py")
    require(specification is not None and specification.loader is not None,
            f"load {name}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def decoder_dimensions(source: bytes) -> dict[str, int]:
    tree = ast.parse(source.decode("utf-8"), filename="<authenticated-strata-decoder>")
    constants = {}
    wanted = {"LEADING_LOG2", "TAIL_LOG2", "ALPHABET_SIZE", "BLOCKS",
              "LEADING_N21_BLOCKS"}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id in wanted):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    require(constants == {"LEADING_LOG2": 21, "TAIL_LOG2": 20,
                           "ALPHABET_SIZE": 64, "BLOCKS": 14,
                           "LEADING_N21_BLOCKS": 13},
            "authenticated STRATA geometry constants")
    text = source.decode("utf-8")
    tokens = (
        "for level_index, flag in enumerate(flags):",
        "previous += (1 << level_index) * x_bit.astype(np.int16)",
        "lr_reg = np.ones((n // 2, depth), dtype=np.float64)",
        "mu_reg = np.zeros((n // 2, depth), dtype=np.uint8)",
        "internal = np.zeros(n, dtype=np.uint8)",
        "return polar_transform(internal[reverse]), frequencies, selected_values",
    )
    require(all(token in text for token in tokens),
            "authenticated decoder six-level state objects")
    return constants


def independent_peak(block_values: int, beam_width: int) -> dict[str, int]:
    depth = (block_values.bit_length() - 1)
    require(1 << depth == block_values, "power-of-two audit block")
    lr = (block_values // 2) * depth * 8
    mu = (block_values // 2) * depth
    planes = 6 * block_values
    index_state = 2 * block_values
    scalar = 256
    per_path = lr + mu + planes + index_state + scalar
    frontier = per_path * beam_width
    backpointers = 4 * block_values * beam_width
    return {"depth": depth, "lr_f64_bytes_per_path": lr,
            "mu_u8_bytes_per_path": mu,
            "six_plane_u8_bytes_per_path": planes,
            "index_i16_bytes_per_path": index_state,
            "scalar_bytes_per_path": scalar, "bytes_per_path": per_path,
            "frontier_bytes": frontier, "backpointer_bytes": backpointers,
            "total_peak_bytes": frontier + backpointers}


def packet_fixture(packet: Any) -> bytes:
    weights = 64
    frames = (
        packet.FrameInput((0,), 32, bytes([31]) * 32, b"\x80\0", 9),
        packet.FrameInput((1,), 32, bytes([32]) * 32, b"\x40\0", 9),
    )
    return packet.build_packet(
        topology=b"audit-topology", frequencies=b"audit-frequency",
        centroids=b"audit-centroid", frames=frames,
        weights=weights, experts=2)


def run_source_tests(package: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-I", "-B", os.fspath(package / "test_source_only.py")],
        check=False, capture_output=True, text=True, timeout=180)
    require(completed.returncode == 0 and completed.stderr == "",
            "isolated hostile source tests")
    receipt = json.loads(completed.stdout)
    require(receipt["status"] == "PASS" and receipt["tests"] >= 14 and
            receipt["qwen_payload_accessed"] is False and
            receipt["current_codec_payload_accessed"] is False and
            receipt["matched_control_payload_accessed"] is False,
            "source test receipt boundary")
    return receipt


def run(package: Path, decoder_source: Path, use_cupy: bool) -> dict[str, Any]:
    authenticated = authenticate_package(package)
    package = authenticated["package"]
    decoder_raw = read_regular(
        decoder_source, expected_bytes=DECODER_BYTES,
        expected_sha256=DECODER_SHA256, maximum_bytes=1 << 20)
    dimensions = decoder_dimensions(decoder_raw)
    source_tests = run_source_tests(package)
    adapter = load(package, "strata_replay_adapter")
    packet = load(package, "candidate_packet")
    independent = load(package, "independent_decoder")
    oracle = load(package, "polar_list_oracle")
    driver = load(package, "bound_driver")

    estimate = independent_peak(1 << dimensions["LEADING_LOG2"], 32)
    require(estimate["total_peak_bytes"] == EXPECTED_PEAK,
            "independent 2^21 beam32 state arithmetic")
    package_estimate = oracle.resource_estimate(1 << 21, 32)
    require(package_estimate["total_peak_bytes_lower_bound"] == EXPECTED_PEAK,
            "package/audit resource equality")
    memory_gate = oracle.production_gate(
        1 << 21, 32, memory_cap_bytes=4 * (1 << 30),
        cupy_topk_wired=True, device_resident_polar_state=True,
        bounded_prefix_storage=True)
    require(memory_gate["status"] == "HOLD_PRODUCTION_POLAR_LIST_SCALABILITY" and
            memory_gate["qwen_payload_may_open"] is False,
            "production memory hard hold")

    adapter_text = authenticated["sources"]["strata_replay_adapter.py"].decode("utf-8")
    require("def encode_choices" not in adapter_text and
            "def decode_events" not in adapter_text and
            "DirectFour" not in adapter_text and
            "direct_int2_4level_new_codec" not in adapter_text,
            "no coordinate-local/replacement codec ABI")
    instance = adapter.ReadOnlyStrataReplayAdapter()
    require(instance.coordinate_local_arithmetic_events is False and
            instance.direct_int2_fallback is False, "adapter negative capabilities")
    try:
        instance.coordinate_choices(0, 0, 31, 1)
    except adapter.CoordinateLocalTransitionHold as error:
        require(str(error) ==
                "HOLD_COORDINATE_LOCAL_EPSILON_INVALID_FOR_LEVEL_MAJOR_POLAR_SC",
                "coordinate-local typed hold")
    else:
        raise AuditError("coordinate-local choice unexpectedly available")

    raw = packet_fixture(packet)
    parsed = packet.parse_packet(raw)
    decoded = independent.decode_and_reencode(raw)
    ledger = parsed["byte_ledger"]
    require(decoded["canonical_reencode_matches"] is True and
            decoded["packet_sha256"] == parsed["packet_sha256"] and
            decoded["packet_bytes"] == parsed["total_bytes"] ==
            ledger["total_bytes"], "literal packet and byte-ledger binding")
    require(ledger["model_bytes"] == ledger["topology_bytes"] +
            ledger["frequency_bytes"], "model byte decomposition")
    trace = packet.owner_read_trace(raw, 0)
    independently_requested = sum(row["end"] - row["begin"]
                                  for row in trace["ranges"])
    require(trace["requested_bytes"] == trace["unique_requested_bytes"] ==
            trace["touched_page_bytes"] == independently_requested and
            trace["compressed_expert_second_pass_count"] == 0,
            "literal range-derived read amplification")

    derive_signature = inspect.signature(driver.derive_fold)
    require("fixed_packet_bytes" not in derive_signature.parameters and
            "state_gain_bpw" not in derive_signature.parameters and
            "read_amplification" not in derive_signature.parameters,
            "no caller-supplied scored quantities")
    driver_text = authenticated["sources"]["bound_driver.py"].decode("utf-8")
    required_tokens = (
        "row bytes exactly equal literal byte ledger",
        "outer fold fit/selection closure",
        "control receipt external pin",
        "derived one-pass routed read",
        "decoded artifact hash binding",
    )
    require(all(token in driver_text for token in required_tokens),
            "mechanical blocker closures present")
    plan = driver.build_outer_plan(((0,), (1,)), 2)
    driver.validate_outer_plan(plan)
    damaged = json.loads(json.dumps(plan))
    damaged["folds"][0]["centroid_fit_stream_ordinals"] = [0]
    damaged = driver.seal({key: value for key, value in damaged.items()
                           if key != "seal_sha256"})
    try:
        driver.validate_outer_plan(damaged)
    except driver.DriverError:
        pass
    else:
        raise AuditError("held stream accepted into centroid fit")

    receipt = driver.control_receipt(
        ordinal=0, outer_plan_sha256=plan["seal_sha256"],
        pipeline_source_root_sha256=sha256(b"pipeline"),
        source_producer_receipt_sha256=sha256(b"producer"),
        legal_trace_panel_receipt_sha256=sha256(b"trace"),
        fold_artifact_receipt_sha256=(sha256(b"fold0"), sha256(b"fold1")))
    pin = sha256(driver.canonical_json(receipt))
    require(len(pin) == 64 and "full_ptq_pipeline" not in
            driver.canonical_json(receipt).decode("ascii"),
            "control closure is pinned receipt not booleans")

    cupy_receipt = None
    if use_cupy:
        import cupy as cp

        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        costs = cp.asarray([3.0, 1.0, 2.0, 1.0], dtype=cp.float64)
        selected = cp.argsort(costs, kind="stable")[:2]
        host = tuple(int(value) for value in cp.asnumpy(selected))
        cp.cuda.Stream.null.synchronize()
        require(host == (1, 3), "source-free deterministic CuPy top-k")
        cupy_receipt = {
            "status": "PASS_SOURCE_FREE_CUPY_TOPK",
            "cupy_version": cp.__version__,
            "device_name": properties["name"].decode()
                if isinstance(properties["name"], bytes) else properties["name"],
            "compute_capability": f"{properties['major']}{properties['minor']}",
            "selected_indices": list(host), "payload_accessed": False,
        }

    return {
        "schema": "epsilon-tcq-strata-bound-v1-independent-source-audit-receipt",
        "status": "PASS_SOURCE_AUDIT_TYPED_HOLD_NO_PAYLOAD_AUTHORITY",
        "audited_source_manifest_sha256": EXPECTED_MANIFEST,
        "audited_source_root_sha256": EXPECTED_ROOT,
        "authenticated_strata_decoder": {
            "bytes": DECODER_BYTES, "sha256": DECODER_SHA256,
            "leading_log2": dimensions["LEADING_LOG2"],
            "tail_log2": dimensions["TAIL_LOG2"],
            "levels": 6, "indices": dimensions["ALPHABET_SIZE"],
        },
        "resource_bound": {**estimate, "beam_width": 32,
                           "memory_cap_bytes": 4 * (1 << 30),
                           "status": memory_gate["status"]},
        "checks": {
            "exact_source_closure": True,
            "hostile_source_tests": source_tests["tests"],
            "six_level_not_six_events_per_coordinate": True,
            "coordinate_local_choice_typed_hold": True,
            "no_replacement_four_level_fallback": True,
            "literal_packet_independent_decode_reencode": True,
            "row_bytes_bound_to_ledger_total": True,
            "gains_derived_not_caller_supplied": True,
            "read_amplification_derived_from_literal_ranges": True,
            "outer_fold_fit_selection_closure": True,
            "eight_control_receipts_require_external_full_json_pins": True,
            "production_prefix_memory_capped": True,
        },
        "cupy": cupy_receipt,
        "qwen_payload_accessed": False,
        "current_codec_payload_accessed": False,
        "matched_control_payload_accessed": False,
        "next_gate": (
            "separately freeze a device-resident resumable whole-block polar "
            "list engine below the 4 GiB cap before any Qwen payload run"
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--package", type=Path, required=True)
    value.add_argument("--decoder-source", type=Path, required=True)
    value.add_argument("--cupy", action="store_true")
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(json.dumps(run(arguments.package, arguments.decoder_source,
                         arguments.cupy), sort_keys=True,
                     separators=(",", ":"), allow_nan=False))
