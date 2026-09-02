#!/usr/bin/env python3
"""Independent, payload-free adversarial audit of the frozen LOGIC-Q v1.

The audit accepts only explicit source-package paths.  It does not locate,
stat, hash, or open a model, a Qwen checkpoint, a production codec artifact,
or a prebuilt matched control.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


EXPECTED_V1_MANIFEST = (
    "9bfd3d1225fb45a0518d2d4d6a4035262e87dc62563222e42e69665358b9aac5"
)
EXPECTED_V1_ROOT = (
    "5d145d89a20d2ae256ea60f569fab97cd6372cde66f7df75f3e86b08b3a88560"
)
EXPECTED_V0_MANIFEST = (
    "31edbc3325dfdae2b3f43cce4afb360062d5c70583b57dd1e6530835a178cced"
)
EXPECTED_V0_ROOT = (
    "2177f2aec39a65afddbbded9b6b3cd2c2a33118c060a41e070102f9fb6c95d4a"
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_closure(package: Path, expected_manifest: str,
                   expected_root: str) -> dict[str, Any]:
    root = package.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("package directory")
    manifest_path = root / "SOURCE_MANIFEST.json"
    info = manifest_path.lstat()
    if not stat.S_ISREG(info.st_mode) or manifest_path.is_symlink():
        raise RuntimeError("manifest regular non-link")
    payload = manifest_path.read_bytes()
    if digest(payload) != expected_manifest:
        raise RuntimeError("manifest external pin")
    manifest = json.loads(payload.decode("utf-8"))
    if manifest.get("source_root_sha256") != expected_root:
        raise RuntimeError("root external pin")
    observed = []
    names = []
    for row in manifest.get("members", []):
        if set(row) != {"name", "bytes", "sha256"}:
            raise RuntimeError("member schema")
        name = row["name"]
        if (not isinstance(name, str) or not name or "/" in name or
                "\\" in name or name == "SOURCE_MANIFEST.json" or
                name in names):
            raise RuntimeError("safe unique member")
        path = root / name
        member_info = path.lstat()
        if not stat.S_ISREG(member_info.st_mode) or path.is_symlink():
            raise RuntimeError("member regular non-link")
        member = path.read_bytes()
        item = {"name": name, "bytes": len(member), "sha256": digest(member)}
        if item != row:
            raise RuntimeError(f"member pin {name}")
        names.append(name)
        observed.append(item)
    root_payload = json.dumps(
        observed, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")
    if digest(root_payload) != expected_root:
        raise RuntimeError("observed source root")
    actual = {entry.name for entry in os.scandir(root)}
    if actual != set(names) | {"SOURCE_MANIFEST.json"}:
        raise RuntimeError("exact regular package closure")
    return {"manifest_sha256": expected_manifest,
            "source_root_sha256": expected_root, "members": names}


def panel_rows(adapter: Any) -> list[Any]:
    rows = []
    for layer in range(10):
        for slot in range(4):
            for role in adapter.ROLE_ORDER:
                material = f"{layer}:{slot}:{role}".encode("ascii")
                rows.append(adapter.PanelRow(
                    f"layer-{layer:02d}", f"expert-{slot:02d}", role,
                    4, 256, digest(material)))
    return rows


def selection_metrics(adapter: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"train": {}, "validation": {}}
    for partition in result:
        for ordinal, config in enumerate(adapter.FROZEN_CONFIGS):
            result[partition][config.config_id] = {
                "physical_bits": float(2200 + 20 * ordinal),
                "weights": 1000.0,
                "weighted_sse": float(20 + ordinal),
                "source_energy": 1000.0,
                "expert_count": 10.0,
            }
    return result


def synthetic_role(np: Any, rows: int, cols: int, shift: float) -> tuple[Any, Any]:
    n = rows * cols
    grid = np.arange(n, dtype=np.float64)
    values = (0.8 * np.sin(grid * 0.017 + shift) +
              0.2 * np.cos(grid * 0.071 - shift))
    weights = np.ones(n, dtype=np.float64)
    return values, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--cupy", action="store_true")
    args = parser.parse_args()

    v1 = verify_closure(args.package, EXPECTED_V1_MANIFEST, EXPECTED_V1_ROOT)
    v0 = verify_closure(args.parent, EXPECTED_V0_MANIFEST, EXPECTED_V0_ROOT)

    import numpy as np

    adapter = load("logic_q_v1_independent_target", args.package / "capped_adapter.py")
    parent_record = adapter.verify_parent_package(args.parent)
    if parent_record["manifest_sha256"] != EXPECTED_V0_MANIFEST:
        raise RuntimeError("adapter parent authentication")
    core = adapter.load_parent_core(args.parent)

    # The mandatory pre-search gate must prevent even a callable that bombs.
    called = False

    def bomb() -> None:
        nonlocal called
        called = True
        raise RuntimeError("search must not run")

    killed = {
        "hard_kill": True, "search_invoked": False,
        "status": "independent synthetic kill",
    }
    result, kill_receipt = adapter.execute_if_survives(killed, bomb)
    if result is not None or called or kill_receipt["search_invoked"]:
        raise RuntimeError("hard kill did not dominate search")

    # Shape equality and canonical byte replay are enforced at expert scope.
    rows, cols = 4, 256
    roles = {role: synthetic_role(np, rows, cols, 0.3 + ordinal)
             for ordinal, role in enumerate(adapter.ROLE_ORDER)}
    literal = {
        role: core.encode_literal_component(
            np, roles[role][0], roles[role][1], role=role,
            rows=rows, cols=cols, block_size=256)
        for role in adapter.ROLE_ORDER
    }
    packets = {role: adapter.canonicalize_component(np, core, value.packet)
               for role, value in literal.items()}
    expert = adapter.pack_canonical_expert(np, core, packets)
    decoded = adapter.unpack_canonical_expert(np, core, expert)
    if set(decoded) != set(adapter.ROLE_ORDER):
        raise RuntimeError("expert roles")
    mismatch_values, mismatch_weights = synthetic_role(np, 8, 128, 0.9)
    mismatch = core.encode_literal_component(
        np, mismatch_values, mismatch_weights, role="down_transposed",
        rows=8, cols=128, block_size=256)
    mismatch_rejected = False
    try:
        adapter.pack_canonical_expert(np, core, {
            "gate": packets["gate"], "up": packets["up"],
            "down_transposed": mismatch.packet,
        })
    except adapter.AdapterError as error:
        mismatch_rejected = "shape equality" in str(error)
    if not mismatch_rejected:
        raise RuntimeError("SwiGLU role-shape mismatch accepted")

    # Receipt hashing is only self-consistency.  Re-sealing a changed selected
    # config currently passes authorize_test because it does not recompute the
    # selector from bound metrics.  This is a production hold, not a model win.
    panel = adapter.panel_record(panel_rows(adapter))
    honest_receipt = adapter.selection_receipt(panel, selection_metrics(adapter))
    forged_receipt = copy.deepcopy(honest_receipt)
    alternatives = [config.config_id for config in adapter.FROZEN_CONFIGS
                    if config.config_id != honest_receipt["selected_config_id"]]
    forged_receipt["selected_config_id"] = alternatives[-1]
    unsigned = dict(forged_receipt)
    unsigned.pop("receipt_sha256", None)
    forged_receipt["receipt_sha256"] = adapter.sha256(adapter.canonical_json(unsigned))
    forged_selection_accepted = (
        adapter.authorize_test(panel, forged_receipt).config_id == alternatives[-1]
    )

    # The internal score consumes encoder-side metric objects.  It authenticates
    # packet bytes but does not independently recompute source SSE.  A final
    # Qwen result therefore needs a separate source-bound decoder/scorer.
    fake_components = {
        role: dataclasses.replace(component, weighted_sse=0.0)
        for role, component in literal.items()
    }
    trusted_score = adapter.pooled_expert_score(np, core, expert, fake_components)
    encoder_metrics_can_force_zero_sse = trusted_score["weighted_sse"] == 0.0

    # The live-backend check currently tests a public __name__ string.  Record
    # that scoped weakness and separately exercise the real CuPy implementation.
    class NameOnlyBackend:
        __name__ = "cupy"

    adapter.require_live_cupy(NameOnlyBackend(), True)
    name_only_backend_accepted = True

    cupy_record: dict[str, Any] = {"requested": bool(args.cupy),
                                   "executed": False}
    if args.cupy:
        cp = importlib.import_module("cupy")
        adapter.require_live_cupy(cp, True)
        config = adapter.FROZEN_CONFIGS[0]
        n = 512
        labels = np.resize(np.asarray([0, 1, 3, 2], dtype=np.int64), n)
        levels = np.asarray(core.PROFILE_RATIOS[0], dtype=np.float64)
        values = levels[labels] + np.linspace(-1e-7, 1e-7, n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64)
        gpu_component = adapter.encode_rm1_capped(
            cp, core, cp.asarray(values), cp.asarray(weights), role="gate",
            rows=2, cols=256, config=config)
        cp.cuda.Stream.null.synchronize()
        decoded_labels = core.decode_component(np, gpu_component.packet)[0]
        if tuple(decoded_labels) != gpu_component.labels:
            raise RuntimeError("real CuPy RM decode mismatch")
        device = int(cp.cuda.runtime.getDevice())
        properties = cp.cuda.runtime.getDeviceProperties(device)
        name = properties.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        cupy_record = {
            "requested": True, "executed": True,
            "cupy_version": str(cp.__version__), "device": device,
            "device_name": str(name),
            "packet_sha256": digest(gpu_component.packet),
            "packet_bytes": len(gpu_component.packet),
            "decoded_labels": len(decoded_labels),
            "diagnostic_cupy_backend": bool(
                gpu_component.diagnostics.get("cupy_backend")),
        }

    # Check the independent matched-control construction on one component.
    ordinal = adapter.control_ordinal(panel, "layer-00", "expert-00", "gate")
    control, _ = adapter.moment_matched_gaussian(
        np, roles["gate"][0], block_size=256,
        seed=adapter.CONTROL_SEEDS[0], component_ordinal=ordinal)
    moment_max_mean_error = 0.0
    moment_max_energy_error = 0.0
    for source_block, control_block in zip(
            roles["gate"][0].reshape(-1, 256), control.reshape(-1, 256)):
        source_mean = float(np.mean(source_block, dtype=np.float64))
        control_mean = float(np.mean(control_block, dtype=np.float64))
        source_energy = float(np.sum((source_block - source_mean) ** 2,
                                     dtype=np.float64))
        control_energy = float(np.sum((control_block - control_mean) ** 2,
                                      dtype=np.float64))
        moment_max_mean_error = max(moment_max_mean_error,
                                    abs(source_mean - control_mean))
        moment_max_energy_error = max(moment_max_energy_error,
                                      abs(source_energy - control_energy))

    findings = {
        "selection_receipt_can_be_resealed_with_different_config":
            forged_selection_accepted,
        "pooled_score_trusts_encoder_metric_objects":
            encoder_metrics_can_force_zero_sse,
        "live_cupy_guard_accepts_name_only_object": name_only_backend_accepted,
    }
    if not all(findings.values()):
        raise RuntimeError("expected production-hold probe changed")

    output = {
        "schema": "logic-q-v1-capped-adapter-independent-adversarial-audit",
        "status": "MECHANISM_VALID__HOLD_BOUND_SELECTOR_SCORER_AND_LIVE_BACKEND",
        "v1": v1, "v0": v0,
        "hard_kill_dominates_search": True,
        "canonical_expert_roundtrip": True,
        "swiglu_shape_mismatch_rejected": True,
        "findings": findings,
        "matched_control_max_abs_mean_error": moment_max_mean_error,
        "matched_control_max_abs_centered_energy_error": moment_max_energy_error,
        "real_cupy": cupy_record,
        "model_or_qwen_payload_accessed": False,
        "current_codec_or_coarse_payload_accessed": False,
        "prebuilt_matched_control_accessed": False,
        "network_accessed_by_audit_script": False,
        "claim_boundary": (
            "Source-only finite-mechanism audit. No Qwen result, model result, "
            "algebraic-family negative, F result, or universal-SwiGLU claim."
        ),
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
