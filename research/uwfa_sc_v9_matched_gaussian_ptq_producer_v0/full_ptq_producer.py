#!/usr/bin/env python3
"""Authenticated-runtime core for eight full-PTQ Gaussian controls.

This file is intentionally not a direct launcher.  A future dispatcher must
authenticate and inject the exact production modules named by the source
manifest.  The functions here contain the mechanical bridge from generated
BF16 matrices to the current 15-stream STRATA artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


DIRECT_STATUS = "BLOCK_DIRECT_EXECUTION_REQUIRES_AUTHENTICATED_RUNTIME_DISPATCHER"


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_hashes(payload: bytes) -> list[str]:
    chunk = (1 << 18) * 2
    if len(payload) != 6 * chunk:
        raise ValueError("matrix is not six N18 BF16 chunks")
    return [hashlib.sha256(payload[start : start + chunk]).hexdigest() for start in range(0, len(payload), chunk)]


def generate_sources(
    *,
    np: Any,
    contract: Any,
    moments: Sequence[Any],
    seed: int,
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Generate and retain all eighteen BF16 matrices for one seed."""
    source_dir = output / "bf16_sources"
    source_dir.mkdir(parents=True, exist_ok=False)
    matrix_meta: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    manifest_rows = []
    for moment in moments:
        words, receipt = contract.generate_matrix_bf16(np, moment, seed)
        key = contract.matrix_key(moment.slot, moment.role)
        relpath = f"bf16_sources/slot_{moment.slot:02d}_{moment.role}.bf16.bin"
        path = output / relpath
        payload = np.ascontiguousarray(words, dtype="<u2").tobytes(order="C")
        _write_new(path, payload)
        observed_sha = _sha_file(path)
        if observed_sha != receipt["control_bf16_sha256"]:
            raise RuntimeError("generated source changed while materializing")
        blocks = _block_hashes(payload)
        matrix_meta.append(
            {
                "matrix_ordinal": moment.ordinal,
                "tensor": f"universal-swiglu:{key}",
                "role": moment.role,
                "slot": moment.slot,
                "axis": "column" if moment.role == "down" else "row",
                "shape": list(moment.shape),
                "source_path": path,
                "source_relpath": relpath,
                "source_bf16_sha256": observed_sha,
                "source_block_sha256s": blocks,
            }
        )
        manifest_rows.append(
            {
                "matrix_ordinal": moment.ordinal,
                "slot": moment.slot,
                "role": moment.role,
                "shape": list(moment.shape),
                "relpath": relpath,
                "bytes": len(payload),
                "sha256": observed_sha,
                "n18_chunk_sha256s": blocks,
            }
        )
        receipts.append(receipt)
    source_manifest = {
        "schema": "uwfa-sc-v9-control-bf16-source-panel-v1",
        "status": "COMPLETE_RETAINED_ENCODER_INPUT",
        "seed": seed,
        "identity_semantics": "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY",
        "matrices": manifest_rows,
        "source_bytes": sum(row["bytes"] for row in manifest_rows),
        "source_weights": sum(row["bytes"] // 2 for row in manifest_rows),
    }
    source_manifest["source_panel_manifest_sha256"] = hashlib.sha256(
        contract.canonical_json(source_manifest)
    ).hexdigest()
    _write_new(output / "SOURCE_PANEL.json", contract.pretty_json(source_manifest))
    return matrix_meta, source_manifest, receipts


def _gpu_energy(cp: Any, words: Any) -> float:
    device_words = cp.asarray(words, dtype=cp.uint16)
    values = (device_words.astype(cp.uint32) << cp.uint32(16)).view(cp.float32).astype(cp.float64)
    result = float(cp.sum(values * values, dtype=cp.float64).get())
    del device_words, values
    return result


def build_current_plan(
    *,
    np: Any,
    cp: Any,
    contract: Any,
    v2_emitter: Any,
    v2_common: Any,
    expert_common: Any,
    matrix_meta: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    control_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Run the exact XKLT/STRATA transform and make the current 15-block plan."""
    transform_root = control_root / "exact_strata_transform"
    plan_root = control_root / "current_plan"
    transform_root.mkdir(parents=True, exist_ok=False)
    plan_root.mkdir(parents=True, exist_ok=False)
    coefficients, angle_codes, label_bytes, klt_rows, _, combined = v2_emitter.build_staging(
        list(matrix_meta), transform_root
    )
    v2_rows = combined[: len(v2_common.BLOCK_LOG2)]
    labels = expert_common.unpack_labels(label_bytes)
    sorted_parts = []
    for row in v2_rows:
        path = transform_root / row["staging_relpath"]
        words = np.fromfile(path, dtype="<u2")
        if words.size != int(row["values"]):
            raise RuntimeError("exact transform staging geometry")
        sorted_parts.append(words.reshape(-1, expert_common.GROUP_VALUES))
    globally_sorted = np.concatenate(sorted_parts, axis=0)
    order = np.lexsort((np.arange(expert_common.GROUPS, dtype=np.int64), labels))
    canonical = np.empty_like(globally_sorted)
    canonical[order] = globally_sorted
    current_ordinals = expert_common.expected_block_group_ordinals(labels)
    staging = plan_root / "staging"
    staging.mkdir()
    energies = np.empty(expert_common.BLOCKS, dtype=np.float64)
    block_rows = []
    for ordinal, (logn, selected_ordinals) in enumerate(
        zip(expert_common.BLOCK_LOG2, current_ordinals, strict=True)
    ):
        selected = np.ascontiguousarray(canonical[selected_ordinals], dtype="<u2")
        path = staging / f"block_{ordinal:02d}_n{logn}.bf16.bin"
        _write_new(path, selected.tobytes(order="C"))
        energies[ordinal] = _gpu_energy(cp, selected)
        block_rows.append(
            {
                "block_ordinal": ordinal,
                "block_log2": logn,
                "values": 1 << logn,
                "groups": len(selected_ordinals),
                "owner_experts": expert_common.block_owner_experts(ordinal),
                "segment": "private" if ordinal < expert_common.PRIVATE_BLOCKS else "paired_tail",
                "source_energy_fp64": float(energies[ordinal]),
                "selected_group_ordinals_sha256": hashlib.sha256(
                    np.asarray(selected_ordinals, dtype="<i8").tobytes()
                ).hexdigest(),
                "staging_relpath": str(path.relative_to(plan_root)).replace("\\", "/"),
                "staging_bytes": path.stat().st_size,
                "staging_sha256": _sha_file(path),
            }
        )
    profiles, allocation = expert_common.allocate_profiles(energies)
    profile_bytes = profiles.tobytes()
    route = contract.build_universal_route()
    header = expert_common.build_header(coefficients, angle_codes, route, label_bytes)
    for row in block_rows:
        ordinal = int(row["block_ordinal"])
        sc_seed, rht_seed, seed_digest = expert_common.derive_seeds(
            header, route, label_bytes, profile_bytes, ordinal
        )
        q = int(profiles[ordinal])
        row.update(
            {
                "profile_id": q,
                "nominal_rate_bpw": expert_common.PROFILE_BASE + q / 256.0,
                "test_distortion": 2.0 ** (-2.0 * (expert_common.PROFILE_BASE + q / 256.0)),
                "sc_seed_u32": sc_seed,
                "rht_seed_u64": rht_seed,
                "seed_digest_sha256": seed_digest,
            }
        )
    asset_payloads = {
        "header.bin": header,
        "route.bin": route,
        "labels_3bit.bin": label_bytes,
        "profiles.bin": profile_bytes,
    }
    assets = {}
    for name, payload in asset_payloads.items():
        path = plan_root / name
        _write_new(path, payload)
        assets[name] = {"relpath": name, "bytes": len(payload), "sha256": _sha_file(path)}
    sources = []
    for row in matrix_meta:
        sources.append(
            {
                "matrix_ordinal": int(row["matrix_ordinal"]),
                "tensor": str(row["tensor"]),
                "slot": int(row["slot"]),
                "role": str(row["role"]),
                "axis": str(row["axis"]),
                "shape": list(row["shape"]),
                "source_relpath": str(row["source_relpath"]),
                "source_bf16_sha256": str(row["source_bf16_sha256"]),
                "bytes": int((control_root / row["source_relpath"]).stat().st_size),
            }
        )
    plan = expert_common.sealed(
        {
            "schema": "strata_expert_affine_n20n21_plan_v1",
            "status": "sealed_before_arithmetic_encoding",
            "architecture": "universal-slot full-PTQ matched-Gaussian current STRATA15",
            "identity_semantics": "CANONICAL_SLOT_AND_SWIGLU_ROLE_ONLY",
            "source_run": {
                "kind": "exact production v2 build_staging primitive followed by exact current regrouping",
                "source_panel_manifest_sha256": source_manifest["source_panel_manifest_sha256"],
                "klt_rows": klt_rows,
            },
            "source_root": "..",
            "sources": sources,
            "assets": assets,
            "blocks": block_rows,
            "allocation": allocation,
            "physical_ledger": {
                "header_bytes": expert_common.HEADER_BYTES,
                "route_bytes": expert_common.ROUTE_BYTES,
                "label_bytes": expert_common.LABEL_BYTES,
                "directory_bytes": expert_common.DIRECTORY_BYTES,
                "reservoir_bytes": expert_common.RESERVOIR_BYTES,
                "physical_bytes": expert_common.PHYSICAL_BYTES,
                "physical_bits": expert_common.PHYSICAL_BITS,
                "physical_bpw": expert_common.PHYSICAL_BITS / expert_common.WEIGHTS,
                "reserve_bits": expert_common.GLOBAL_RESERVE_BITS,
            },
            "coverage": {
                "experts": expert_common.EXPERTS,
                "matrices": expert_common.MATRICES,
                "groups": expert_common.GROUPS,
                "weights": expert_common.WEIGHTS,
                "blocks": expert_common.BLOCKS,
                "every_group_once": True,
                "cupy_energy_sum_fp64": float(energies.sum(dtype=np.float64)),
            },
        }
    )
    _write_new(plan_root / "plan.lock.json", contract.pretty_json(plan))
    return plan_root, plan


def run_exact_current_encoder(
    *,
    run_and_pack: Any,
    workspace: Path,
    python: Path,
    encoder: Path,
    polar_repo: Path,
    plan_root: Path,
) -> dict[str, Any]:
    """Invoke the unchanged production polar encoder and current packer."""
    plan = run_and_pack.verify_plan(plan_root)
    encoded = [
        run_and_pack.run_block(workspace, python, encoder, polar_repo, plan_root, block)
        for block in plan["blocks"]
    ]
    return run_and_pack.pack(plan_root, plan, encoded)


def decode_and_score_universal(
    *,
    np: Any,
    contract: Any,
    expert_common: Any,
    independent_auditor: Any,
    adapter: Any,
    protocol: Any,
    plan_root: Path,
    plan: Mapping[str, Any],
    score_root: Path,
) -> dict[str, Any]:
    """Decode independently, score retained BF16 inputs, and bind v8 geometry."""
    score_root.mkdir(parents=True, exist_ok=False)
    decoded_root = score_root / "decoded"
    decoded_root.mkdir()
    summary = json.loads((plan_root / "summary.json").read_text(encoding="utf-8"))
    artifact_path = plan_root / summary["artifact"]["relpath"]
    parsed = independent_auditor.parse_container(artifact_path, dict(plan))
    decoded = []
    for ordinal, row in enumerate(parsed["directory"]):
        output = decoded_root / f"block_{ordinal:02d}.f64.bin"
        decoded.append(
            independent_auditor.decode_block_worker(
                (str(artifact_path), str(output), row, expert_common.BLOCK_LOG2[ordinal])
            )
        )
    post = np.empty((expert_common.GROUPS, expert_common.GROUP_VALUES), dtype=np.float64)
    coverage = np.zeros(expert_common.GROUPS, dtype=np.uint8)
    for row, ordinals in zip(decoded, expert_common.expected_block_group_ordinals(parsed["labels"]), strict=True):
        values = np.fromfile(row["output_path"], dtype="<f8").reshape(-1, expert_common.GROUP_VALUES)
        post[ordinals] = values
        coverage[ordinals] += 1
    if not bool(np.all(coverage == 1)):
        raise RuntimeError("independent reconstruction coverage")
    coefficients = struct.unpack_from("<12f", parsed["header"], 32)
    total_sse = 0.0
    total_energy = 0.0
    reconstruction_digest = hashlib.sha256()
    matrix_scores = []
    for slot in range(expert_common.EXPERTS):
        base = slot * expert_common.GROUPS_PER_EXPERT
        gate_hat = np.asarray(post[base : base + 768], dtype=np.float64)
        z0 = np.asarray(post[base + 768 : base + 1536], dtype=np.float64)
        z1 = np.asarray(post[base + 1536 : base + 2304], dtype=np.float64)
        cosine = float(coefficients[2 * slot])
        sine = float(coefficients[2 * slot + 1])
        norm2 = cosine * cosine + sine * sine
        reconstructions = (
            gate_hat,
            (cosine * z0 - sine * z1) / norm2,
            (sine * z0 + cosine * z1) / norm2,
        )
        for role, reconstruction in zip(contract.ROLES, reconstructions, strict=True):
            ordinal = 3 * slot + contract.ROLES.index(role)
            source_row = plan["sources"][ordinal]
            source_root = Path(plan["source_root"])
            if not source_root.is_absolute():
                source_root = (plan_root / source_root).resolve()
            source_path = source_root / source_row["source_relpath"]
            if _sha_file(source_path) != source_row["source_bf16_sha256"]:
                raise RuntimeError("retained control source digest")
            words = np.fromfile(source_path, dtype="<u2").reshape(tuple(source_row["shape"]))
            source = contract.bf16_to_f64(np, words)
            natural = source.T if role == "down" else source
            error = np.asarray(reconstruction, dtype=np.float64) - natural
            sse = float(np.sum(error * error, dtype=np.float64))
            energy = float(np.sum(natural * natural, dtype=np.float64))
            total_sse += sse
            total_energy += energy
            payload = np.asarray(reconstruction, dtype="<f8").tobytes(order="C")
            reconstruction_digest.update(payload)
            matrix_scores.append(
                {
                    "matrix_ordinal": ordinal,
                    "slot": slot,
                    "role": role,
                    "sse_fp64": sse,
                    "source_energy_fp64": energy,
                    "relative_mse": sse / energy,
                    "reconstruction_f64_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    artifact = artifact_path.read_bytes()
    panel = adapter.extract_from_current(artifact)
    observed_reconstruction = reconstruction_digest.hexdigest()
    if panel["reconstruction"]["full_reconstruction_f64_sha256"] != observed_reconstruction:
        raise RuntimeError("generic scorer differs from exact v8 adapter reconstruction")
    full_geometry_record = protocol.panel_geometry(panel)
    structural_geometry_record = protocol.structural_panel_geometry(panel)
    full_geometry = protocol.geometry_sha256(adapter.common, panel)
    structural_geometry = protocol.structural_geometry_sha256(adapter.common, panel)
    return {
        "schema": "uwfa-sc-v9-universal-control-independent-score-v1",
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "artifact_bytes": len(artifact),
        "reconstruction_f64_sha256": observed_reconstruction,
        "control_full_geometry_sha256": full_geometry,
        "control_structural_geometry_sha256": structural_geometry,
        "control_full_geometry": full_geometry_record,
        "control_structural_geometry": structural_geometry_record,
        "universal_format_geometry_sha256": contract.universal_format_geometry_sha256(),
        "sse_fp64": total_sse,
        "source_energy_fp64": total_energy,
        "relative_mse": total_sse / total_energy,
        "matrix_scores": matrix_scores,
        "all_payloads_canonically_reencoded": all(row["canonical_reencode_matches"] for row in decoded),
        "same_reconstruction_as_exact_v8_adapter": True,
    }


def _tree_members(root: Path, *, excluded: set[str]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root)).replace("\\", "/")):
        if not path.is_file():
            continue
        name = str(path.relative_to(root)).replace("\\", "/")
        if name in excluded:
            continue
        rows.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _members_root(contract: Any, rows: Sequence[Mapping[str, Any]], domain: bytes) -> str:
    digest = hashlib.sha256(domain)
    for row in rows:
        name = str(row["name"]).encode("utf-8")
        digest.update(struct.pack("<Q", len(name)))
        digest.update(name)
        digest.update(struct.pack("<Q", int(row["bytes"])))
        digest.update(bytes.fromhex(str(row["sha256"])))
    return digest.hexdigest()


def produce_all_controls(
    *,
    contract: Any,
    moment_contract_record: Mapping[str, Any],
    output_root: Path,
    generator_capsule_bytes: bytes,
    symmetric_codec_closure: Mapping[str, str],
    independent_decoder_source_sha256: str,
    np: Any,
    cp: Any,
    v2_emitter: Any,
    v2_common: Any,
    expert_common: Any,
    run_and_pack: Any,
    independent_auditor: Any,
    adapter_factory: Any,
    protocol: Any,
    workspace: Path,
    python: Path,
    encoder: Path,
    polar_repo: Path,
) -> dict[str, Any]:
    """Produce all eight bundles after an external dispatcher authenticates inputs.

    This function never performs source-result selection or WFA fitting.  It
    creates the symmetric full-PTQ null inputs.  A separately audited v9
    controls bridge must authenticate every returned bundle before launching
    the unchanged all-150 search independently for each seed.
    """
    moment_record, moments = contract.validate_moment_contract(moment_contract_record)
    if output_root.exists():
        raise FileExistsError(f"control output must not exist: {output_root}")
    output_root.mkdir(parents=True)
    incomplete = output_root / "INCOMPLETE"
    _write_new(incomplete, b"UWFA-SC-v9 Gaussian controls incomplete\n")
    contract.validate_generator_capsule(generator_capsule_bytes)
    generator_sha = hashlib.sha256(generator_capsule_bytes).hexdigest()
    _write_new(output_root / "GENERATOR_CAPSULE.json", generator_capsule_bytes)
    _write_new(output_root / "MOMENT_CONTRACT.json", contract.pretty_json(moment_record))
    universal_geometry = contract.universal_format_geometry()
    if hashlib.sha256(contract.canonical_json(universal_geometry)).hexdigest() != contract.universal_format_geometry_sha256():
        raise RuntimeError("universal format geometry closure")
    _write_new(
        output_root / "UNIVERSAL_FORMAT_GEOMETRY.json",
        contract.pretty_json(universal_geometry),
    )
    control_rows = []
    for index, seed in enumerate(contract.CONTROL_SEEDS):
        control_root = output_root / f"control_{index:02d}_{seed}"
        control_root.mkdir()
        matrix_meta, source_manifest, matrix_receipts = generate_sources(
            np=np,
            contract=contract,
            moments=moments,
            seed=seed,
            output=control_root,
        )
        plan_root, plan = build_current_plan(
            np=np,
            cp=cp,
            contract=contract,
            v2_emitter=v2_emitter,
            v2_common=v2_common,
            expert_common=expert_common,
            matrix_meta=matrix_meta,
            source_manifest=source_manifest,
            control_root=control_root,
        )
        summary = run_exact_current_encoder(
            run_and_pack=run_and_pack,
            workspace=workspace,
            python=python,
            encoder=encoder,
            polar_repo=polar_repo,
            plan_root=plan_root,
        )
        adapter = adapter_factory()
        score = decode_and_score_universal(
            np=np,
            contract=contract,
            expert_common=expert_common,
            independent_auditor=independent_auditor,
            adapter=adapter,
            protocol=protocol,
            plan_root=plan_root,
            plan=plan,
            score_root=control_root / "independent_score",
        )
        if (
            score["artifact_sha256"] != summary["artifact"]["sha256"]
            or score["artifact_bytes"] != summary["artifact"]["physical_bytes"]
        ):
            raise RuntimeError("packed artifact changed before independent scoring")
        moment_receipt = contract.build_moment_receipt(
            seed=seed,
            moment_contract_sha256=moment_record["moment_contract_sha256"],
            generator_capsule_sha256=generator_sha,
            matrix_receipts=matrix_receipts,
            source_panel_manifest_sha256=source_manifest["source_panel_manifest_sha256"],
        )
        score_receipt = contract.build_score_receipt(
            artifact_sha256=score["artifact_sha256"],
            artifact_bytes=score["artifact_bytes"],
            reconstruction_sha256=score["reconstruction_f64_sha256"],
            control_full_geometry_sha256=score["control_full_geometry_sha256"],
            independent_decoder_source_sha256=independent_decoder_source_sha256,
            sse=score["sse_fp64"],
            energy=score["source_energy_fp64"],
        )
        binding = contract.build_control_binding_v9(
            seed=seed,
            source_closure=moment_record["source_closure"],
            generator_capsule_sha256=generator_sha,
            moment_match_receipt_sha256=moment_receipt["moment_match_receipt_sha256"],
            source_panel_manifest_sha256=source_manifest["source_panel_manifest_sha256"],
            control_artifact_sha256=score["artifact_sha256"],
            control_full_geometry_sha256=score["control_full_geometry_sha256"],
            control_structural_geometry_sha256=score["control_structural_geometry_sha256"],
            symmetric_codec_closure=symmetric_codec_closure,
        )
        _write_new(control_root / "MOMENT_MATCH_RECEIPT.json", contract.pretty_json(moment_receipt))
        _write_new(control_root / "INDEPENDENT_SCORE.json", contract.pretty_json(score))
        _write_new(control_root / "SCORE_RECEIPT.json", contract.pretty_json(score_receipt))
        _write_new(control_root / "CONTROL_BINDING.json", contract.pretty_json(binding))
        members = _tree_members(control_root, excluded={"COMPLETE.json"})
        complete = {
            "schema": "uwfa-sc-v9-full-ptq-control-complete-v1",
            "status": "COMPLETE_FULL_BF16_TO_CURRENT_STRATA_ARTIFACT_NONPROMOTING",
            "seed": seed,
            "members": members,
            "members_root_sha256": _members_root(
                contract, members, b"UWFA-SC-V9-CONTROL-MEMBERS-v1\x00"
            ),
            "artifact": summary["artifact"],
            "source_panel_manifest_sha256": source_manifest["source_panel_manifest_sha256"],
            "moment_match_receipt_sha256": moment_receipt["moment_match_receipt_sha256"],
            "score_receipt_sha256": score_receipt["score_receipt_sha256"],
            "binding_sha256": binding["binding_sha256"],
            "all_150_wfa_search_run": False,
            "requires_separately_audited_v9_controls_bridge": True,
        }
        _write_new(control_root / "COMPLETE.json", contract.pretty_json(complete))
        control_rows.append(
            {
                "index": index,
                "seed": seed,
                "relpath": control_root.name,
                "complete_sha256": _sha_file(control_root / "COMPLETE.json"),
                "artifact_sha256": score["artifact_sha256"],
                "source_panel_manifest_sha256": source_manifest["source_panel_manifest_sha256"],
                "control_full_geometry_sha256": score["control_full_geometry_sha256"],
                "control_structural_geometry_sha256": score["control_structural_geometry_sha256"],
            }
        )
    all_members = _tree_members(
        output_root,
        excluded={"INCOMPLETE", "COMPLETE.json"},
    )
    root = {
        "schema": "uwfa-sc-v9-full-ptq-eight-control-root-v1",
        "status": "COMPLETE_EIGHT_FULL_PTQ_CONTROLS_AWAITING_V9_ALL150_CONSUMER",
        "controls": control_rows,
        "generator_capsule_sha256": generator_sha,
        "moment_contract_sha256": moment_record["moment_contract_sha256"],
        "universal_format_geometry_sha256": contract.universal_format_geometry_sha256(),
        "members": all_members,
        "members_root_sha256": _members_root(
            contract, all_members, b"UWFA-SC-V9-EIGHT-CONTROL-ROOT-v1\x00"
        ),
        "all_control_sources_retained": True,
        "all_control_artifacts_literal_current_format": True,
        "all_150_wfa_search_run": False,
        "positive_claim_authority": False,
    }
    incomplete.unlink()
    _write_new(output_root / "COMPLETE.json", contract.pretty_json(root))
    return root


def direct_main() -> int:
    print(DIRECT_STATUS, file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(direct_main())
