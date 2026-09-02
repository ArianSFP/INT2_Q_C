#!/usr/bin/env python3
"""Authenticated v9 matched-Gaussian consumer and result-audit core.

The module is import-inert and standard-library-only. Numeric modules, the
sealed v8 primitives, adapters and payload readers are injected only through
``consume_controls`` after external primary/run authorizations validate.
"""

from __future__ import annotations

import array
import gc
import hashlib
import json
import math
import os
import statistics
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


DIRECT_STATUS = "BLOCK_DIRECT_EXECUTION_REQUIRES_EXTERNAL_AUDITED_RUN_AUTHORIZATION"


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha_file(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            state.update(chunk)
    return state.hexdigest()


def _u16_sha256(values: Sequence[int]) -> str:
    packed = array.array("H", (int(value) for value in values))
    if sys.byteorder != "little":
        packed.byteswap()
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _u64_sha256(values: Sequence[int]) -> str:
    state = hashlib.sha256()
    for value in values:
        state.update(struct.pack("<Q", int(value)))
    return state.hexdigest()


def _recorded_nested_holdout(
    *,
    contract: Any,
    stage0: Any,
    common: Any,
    protocol: Any,
    container_codec: Any,
    backend: Any,
    cache: Any,
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    """Exact v8 nested search with every one of 150 cell scores retained."""
    streams = panel["streams"]
    identities = panel["semantic_identities"]
    plans = stage0._component_fold_plan(common, protocol, panel)
    if len(plans) < 2:
        return {
            "kind": "disjoint_stream_owner_dependence_component_holdout",
            "status": "HOLD_NONESTIMABLE_DEPENDENCE_COMPONENTS",
            "folds": [],
            "estimable": False,
            "source_winner_reused": False,
            "complete_150_cell_search_recorded_every_fold": False,
            "positive_promotion": False,
        }
    skipped = [
        {
            "outer_unit": {
                "component_ordinal": int(plan["component_ordinal"]),
                "identity_indices": list(plan["identity_indices"]),
            },
            "reason": plan["reason"],
        }
        for plan in plans if not plan["estimable"]
    ]
    if skipped:
        return {
            "kind": "disjoint_stream_owner_dependence_component_holdout",
            "status": "NOT_ESTIMABLE_EXACT_IDENTITY_HOLDOUT",
            "folds": [],
            "skipped_folds": skipped,
            "estimable": False,
            "source_winner_reused": False,
            "complete_150_cell_search_recorded_every_fold": False,
            "positive_promotion": False,
        }
    candidates = list(common.candidate_bank())
    if len(candidates) != contract.ALL150 or [int(row.selector_ordinal) for row in candidates] != list(range(contract.ALL150)):
        raise RuntimeError("canonical all-150 candidate bank changed")
    fold_rows = []
    for plan in plans:
        identity_indices = [int(value) for value in plan["identity_indices"]]
        test_indices = list(plan["test_indices"])
        development_indices = list(plan["development_indices"])
        validation_indices = list(plan["validation_indices"])
        train_indices = list(plan["train_indices"])
        choices = []
        cells = []
        for candidate in candidates:
            frequencies = stage0.fit_candidate(common, backend, cache, train_indices, candidate)
            lengths = stage0.exact_lengths(
                common, backend, cache, validation_indices, candidate, frequencies
            )
            charged_raw = stage0.literal_validation_score(
                common,
                protocol,
                container_codec,
                panel,
                validation_indices,
                lengths,
                candidate,
                frequencies,
            )
            charged = int(charged_raw)
            if charged != charged_raw:
                raise RuntimeError("nonintegral exact validation charge")
            cell = contract.sealed(
                {
                    **candidate.as_dict(),
                    "validation_charged_bits": charged,
                    "fitted_frequency_u16_sha256": _u16_sha256(frequencies),
                    "validation_lengths_u64_sha256": _u64_sha256(lengths),
                    "validation_stream_ordinals": [
                        int(streams[index]["stream_ordinal"]) for index in validation_indices
                    ],
                    "trained_only_on_inner_train_streams": True,
                    "source_winner_reused": False,
                },
                "cell_result_sha256",
            )
            cells.append(cell)
            choices.append((charged, int(candidate.selector_ordinal), candidate))
        validation_bits, _selector, selected = min(choices)
        frequencies = stage0.fit_candidate(
            common, backend, cache, development_indices, selected
        )
        test_lengths = stage0.exact_lengths(
            common, backend, cache, test_indices, selected, frequencies
        )
        literal_baseline_bits = stage0.literal_current_baseline_score(
            protocol, container_codec, panel
        )
        literal_model_only_bits = stage0.literal_validation_score(
            common, protocol, container_codec, panel, [], [], selected, frequencies
        )
        literal_candidate_bits = stage0.literal_validation_score(
            common,
            protocol,
            container_codec,
            panel,
            test_indices,
            test_lengths,
            selected,
            frequencies,
        )
        baseline_allocated_bits = 0.0
        candidate_allocated_bits = 0.0
        allocated_weights = 0.0
        for index, logical in zip(test_indices, test_lengths, strict=True):
            row = streams[index]
            baseline_allocated_bits += 8.0 * int(row["baseline_payload_bytes"])
            candidate_allocated_bits += 8.0 * ((int(logical) + 7) // 8)
            allocated_weights += int(row["weight_charge"])
        literal_model_increment = literal_model_only_bits - literal_baseline_bits
        literal_saved_bits = literal_baseline_bits - literal_candidate_bits
        fold_rows.append(
            {
                "outer_dependence_component_ordinal": int(plan["component_ordinal"]),
                "outer_identity_indices": identity_indices,
                "outer_identities_from_artifact": [
                    list(identities[index]) for index in identity_indices
                ],
                "development_exclusion_policy": "exact_identity",
                "test_stream_ordinals": [
                    int(streams[index]["stream_ordinal"]) for index in test_indices
                ],
                "development_stream_ordinals": [
                    int(streams[index]["stream_ordinal"]) for index in development_indices
                ],
                "inner_train_stream_ordinals": [
                    int(streams[index]["stream_ordinal"]) for index in train_indices
                ],
                "inner_validation_stream_ordinals": [
                    int(streams[index]["stream_ordinal"]) for index in validation_indices
                ],
                "all_150_inner_validation_cells": cells,
                "all_150_cell_results_sha256": hashlib.sha256(
                    contract.canonical_json(cells)
                ).hexdigest(),
                "selected_by_inner_validation_only": selected.as_dict(),
                "inner_validation_exact_charged_bits": validation_bits,
                "literal_authenticated_current_baseline_container_bits": literal_baseline_bits,
                "literal_candidate_container_bits": literal_candidate_bits,
                "literal_selected_model_aligned_increment_bits": literal_model_increment,
                "literal_test_saving_after_exact_container_delta_bits": literal_saved_bits,
                "allocated_test_weights": allocated_weights,
                "allocated_baseline_bits": baseline_allocated_bits,
                "allocated_candidate_bits": candidate_allocated_bits,
                "exact_test_saving_bpw": literal_saved_bits / allocated_weights,
                "source_winner_reused": False,
            }
        )
    allocated_total = sum(float(row["allocated_test_weights"]) for row in fold_rows)
    if abs(allocated_total - int(panel["weights"])) > 1e-6:
        raise RuntimeError("owner-attributed outer folds do not partition weights")
    pooled_saved_bits = sum(
        float(row["literal_test_saving_after_exact_container_delta_bits"])
        for row in fold_rows
    )
    pooled = pooled_saved_bits / allocated_total
    values = [float(row["exact_test_saving_bpw"]) for row in fold_rows]
    leave_one_component_out = []
    for omitted, row in enumerate(fold_rows):
        kept_weights = allocated_total - float(row["allocated_test_weights"])
        kept_bits = pooled_saved_bits - float(
            row["literal_test_saving_after_exact_container_delta_bits"]
        )
        leave_one_component_out.append(
            {
                "omitted_component_ordinal": omitted,
                "pooled_saving_bpw": kept_bits / kept_weights,
            }
        )
    votes: dict[int, int] = {}
    for row in fold_rows:
        ordinal = int(row["selected_by_inner_validation_only"]["selector_ordinal"])
        votes[ordinal] = votes.get(ordinal, 0) + 1
    selected_ordinal = min(votes, key=lambda ordinal: (-votes[ordinal], ordinal))
    selected = candidates[selected_ordinal]
    component_positive = all(value > 0.0 for value in values)
    result = {
        "kind": "disjoint_stream_owner_dependence_component_holdout",
        "primary_policy": "exact_identity",
        "status": "PASS_DISJOINT_DEPENDENCE_COMPONENT_HOLDOUT",
        "folds": fold_rows,
        "skipped_folds": [],
        "estimable": True,
        "pooled_exact_heldout_saving_bpw": pooled,
        "minimum_fold_exact_saving_bpw": min(values),
        "dependence_component_mean_saving_bpw_diagnostic_only": statistics.fmean(values),
        "confidence_rule": "disjoint component conjunction; no iid interval",
        "independent_component_count": len(fold_rows),
        "all_dependence_components_positive": component_positive,
        "leave_one_component_out_pooled_saving_bpw_diagnostic_only": leave_one_component_out,
        "candidate_vote_counts": votes,
        "final_topology_selected_from_nested_fold_votes": selected.as_dict(),
        "passes_pooled_standalone_threshold": pooled
        >= common.STANDALONE_REQUIRED_SAVING_BPW,
        "passes_every_disjoint_component_positive": component_positive,
        "passes_heldout_gate": pooled >= common.STANDALONE_REQUIRED_SAVING_BPW
        and component_positive,
        "complete_150_cell_search_recorded_every_fold": True,
        "source_winner_reused": False,
        "positive_promotion": False,
    }
    contract.validate_all150_scientific(result, common)
    return result


def _validate_moment_replay(
    *,
    contract: Any,
    value: Any,
    control: Mapping[str, Any],
    run_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema", "status", "seed", "source_panel_manifest_sha256",
        "moment_contract_sha256", "generator_capsule_sha256",
        "moment_receipt_file_sha256", "matrices_replayed",
        "all_retained_bf16_bytes_reproduced", "all_source_and_control_moments_recomputed",
        "moment_replayer_source_sha256", "replay_receipt_sha256",
    }
    contract.require(isinstance(value, dict) and set(value) == required, "moment replay fields")
    contract.require(value["schema"] == "uwfa-sc-v9-independent-full-bf16-moment-replay-v1", "moment replay schema")
    contract.require(value["status"] == "PASS_REGENERATED_ALL_CONTROL_BF16_AND_RECOMPUTED_MOMENTS", "moment replay status")
    contract.require(value["seed"] == control["seed"] and value["matrices_replayed"] == 18, "moment replay coverage")
    contract.require(value["source_panel_manifest_sha256"] == control["source_panel"]["source_panel_manifest_sha256"], "moment replay source panel")
    contract.require(value["moment_contract_sha256"] == control["moment_contract_sha256"], "moment replay contract")
    contract.require(value["generator_capsule_sha256"] == control["generator_capsule_sha256"], "moment replay generator")
    contract.require(value["moment_receipt_file_sha256"] == control["moment_receipt_file_sha256"], "moment replay receipt bytes")
    contract.require(value["all_retained_bf16_bytes_reproduced"] is True and value["all_source_and_control_moments_recomputed"] is True, "moment replay flags")
    contract.require(value["moment_replayer_source_sha256"] == run_authorization["moment_replayer_source_sha256"], "moment replayer source pin")
    contract.validate_seal(value, "replay_receipt_sha256")
    return dict(value)


def _prepare_all_controls(
    *,
    contract: Any,
    stage0: Any,
    common: Any,
    protocol: Any,
    adapter_factory: Callable[[], Any],
    member_loader: Callable[[str], bytes],
    authentication: Mapping[str, Any],
    primary: Mapping[str, Any],
    run_authorization: Mapping[str, Any],
    moment_replayer: Callable[..., Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prepared = []
    for control in authentication["controls"]:
        artifact = member_loader(control["artifact_member"])
        contract.require(
            isinstance(artifact, bytes)
            and len(artifact) == contract.ARTIFACT_BYTES
            and hashlib.sha256(artifact).hexdigest()
            == control["binding"]["control_artifact_sha256"],
            "control artifact changed after global authentication",
        )
        adapter = adapter_factory()
        panel = stage0.prepare_panel(protocol, adapter, artifact)
        full_record = protocol.panel_geometry(panel)
        structural_record = protocol.structural_panel_geometry(panel)
        full_sha = protocol.geometry_sha256(common, panel)
        structural_sha = protocol.structural_geometry_sha256(common, panel)
        contract.require(full_record == control["score"]["control_full_geometry"], "adapter/control full geometry record")
        contract.require(structural_record == control["score"]["control_structural_geometry"], "adapter/control structural geometry record")
        contract.require(full_sha == control["binding"]["control_full_geometry_sha256"], "adapter/control full geometry digest")
        contract.require(structural_sha == control["binding"]["control_structural_geometry_sha256"], "adapter/control structural geometry digest")
        # Deliberately no equality comparison to source full or structural
        # geometry. Only the pre-frozen universal format geometry is shared.
        score_receipt_bytes = member_loader(control["score_receipt_member"])
        contract.require(
            isinstance(score_receipt_bytes, bytes)
            and hashlib.sha256(score_receipt_bytes).hexdigest()
            == control["score_receipt_sha256"],
            "score receipt changed after global authentication",
        )
        score = protocol.validate_score_receipt(
            common.strict_json_loads(score_receipt_bytes),
            artifact_sha256=control["binding"]["control_artifact_sha256"],
            artifact_bytes=len(artifact),
            weights=int(panel["weights"]),
            reconstruction_sha256=str(
                panel["reconstruction"]["full_reconstruction_f64_sha256"]
            ),
            original_source_panel_sha256=full_sha,
            independent_decoder_source_sha256=control["binding"][
                "symmetric_codec_closure"
            ]["independent_auditor_sha256"],
        )
        moment_contract_bytes = member_loader("MOMENT_CONTRACT.json")
        generator_capsule_bytes = member_loader("GENERATOR_CAPSULE.json")
        receipt_member = f"{control['prefix']}/MOMENT_MATCH_RECEIPT.json"
        receipt_bytes = member_loader(receipt_member)
        contract.require(
            isinstance(moment_contract_bytes, bytes)
            and hashlib.sha256(moment_contract_bytes).hexdigest()
            == authentication["moment_contract_sha256"],
            "moment contract changed after global authentication",
        )
        contract.require(
            isinstance(generator_capsule_bytes, bytes)
            and hashlib.sha256(generator_capsule_bytes).hexdigest()
            == authentication["generator_capsule_sha256"],
            "generator capsule changed after global authentication",
        )
        contract.require(
            isinstance(receipt_bytes, bytes)
            and hashlib.sha256(receipt_bytes).hexdigest()
            == control["moment_receipt_file_sha256"],
            "moment receipt changed after global authentication",
        )
        replay = moment_replayer(
            seed=control["seed"],
            prefix=control["prefix"],
            source_panel=control["source_panel"],
            source_member_loader=member_loader,
            moment_contract_bytes=moment_contract_bytes,
            generator_capsule_bytes=generator_capsule_bytes,
            moment_match_receipt_bytes=receipt_bytes,
        )
        enriched = dict(control)
        enriched.update(
            {
                "moment_contract_sha256": authentication["moment_contract_sha256"],
                "generator_capsule_sha256": authentication["generator_capsule_sha256"],
                "moment_receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            }
        )
        replay = _validate_moment_replay(
            contract=contract,
            value=replay,
            control=enriched,
            run_authorization=run_authorization,
        )
        prepared.append(
            {
                **enriched,
                "decoded_full_geometry_sha256": full_sha,
                "decoded_structural_geometry_sha256": structural_sha,
                "decoded_reconstruction_sha256": str(
                    panel["reconstruction"]["full_reconstruction_f64_sha256"]
                ),
                "validated_score_receipt_sha256": score[
                    "score_receipt_sha256"
                ],
                "moment_replay": replay,
            }
        )
        del panel, adapter, artifact, score
        gc.collect()
    contract.require(
        len(prepared) == 8
        and [row["seed"] for row in prepared] == list(contract.CONTROL_SEEDS),
        "all eight controls fully prepared before fit",
    )
    return prepared


def _materialize_control_for_fit(
    *,
    contract: Any,
    stage0: Any,
    common: Any,
    protocol: Any,
    adapter_factory: Callable[[], Any],
    member_loader: Callable[[str], bytes],
    control: Mapping[str, Any],
    primary: Mapping[str, Any],
    run_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen one compactly authenticated control immediately before fitting."""
    artifact = member_loader(control["artifact_member"])
    contract.require(
        isinstance(artifact, bytes)
        and len(artifact) == contract.ARTIFACT_BYTES
        and hashlib.sha256(artifact).hexdigest()
        == control["binding"]["control_artifact_sha256"],
        "fit control artifact changed after preauthentication",
    )
    adapter = adapter_factory()
    panel = stage0.prepare_panel(protocol, adapter, artifact)
    full_sha = protocol.geometry_sha256(common, panel)
    structural_sha = protocol.structural_geometry_sha256(common, panel)
    reconstruction_sha = str(
        panel["reconstruction"]["full_reconstruction_f64_sha256"]
    )
    contract.require(
        full_sha == control["decoded_full_geometry_sha256"]
        and structural_sha == control["decoded_structural_geometry_sha256"]
        and reconstruction_sha == control["decoded_reconstruction_sha256"],
        "rematerialized panel commitment",
    )
    score_receipt_bytes = member_loader(control["score_receipt_member"])
    contract.require(
        isinstance(score_receipt_bytes, bytes)
        and hashlib.sha256(score_receipt_bytes).hexdigest()
        == control["score_receipt_sha256"],
        "fit score receipt changed after preauthentication",
    )
    score = protocol.validate_score_receipt(
        common.strict_json_loads(score_receipt_bytes),
        artifact_sha256=control["binding"]["control_artifact_sha256"],
        artifact_bytes=len(artifact),
        weights=int(panel["weights"]),
        reconstruction_sha256=reconstruction_sha,
        original_source_panel_sha256=full_sha,
        independent_decoder_source_sha256=control["binding"][
            "symmetric_codec_closure"
        ]["independent_auditor_sha256"],
    )
    contract.require(
        score["score_receipt_sha256"]
        == control["validated_score_receipt_sha256"],
        "rematerialized score receipt commitment",
    )
    runtime = run_authorization["v8_runtime_closure"]
    evidence = stage0.BoundEvidence(
        baseline_plan_sha256=control["plan_sha256"],
        baseline_score_sha256=control["score_receipt_sha256"],
        universal_decoder_sha256=control["binding"]["symmetric_codec_closure"][
            "independent_auditor_sha256"
        ],
        producer_manifest_sha256=run_authorization[
            "producer_source_manifest_sha256"
        ],
        audit_bootstrap_sha256=run_authorization["audit_bootstrap_sha256"],
        source_full_geometry_sha256=full_sha,
        source_structural_geometry_sha256=structural_sha,
        extraction_program_sha256=runtime["v8_strata_adapter_sha256"],
        universal_adapter_sha256=runtime["v8_universal_adapter_sha256"],
        pipeline_sha256=primary["source_pipeline_sha256"],
        source_snapshot_root_sha256=run_authorization[
            "source_snapshot_root_sha256"
        ],
        source_preflight_receipt_sha256=run_authorization[
            "v8_all150_preflight_receipt_sha256"
        ],
    )
    return {
        **control,
        "adapter": adapter,
        "panel": panel,
        "score_validated": score,
        "evidence": evidence,
    }


def _public_input_authentication(
    authentication: Mapping[str, Any], prepared: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        key: value for key, value in authentication.items() if key != "controls"
    } | {
        "controls": [
            {
                "index": index,
                "seed": row["seed"],
                "complete_sha256": row["complete_sha256"],
                "artifact_sha256": row["binding"]["control_artifact_sha256"],
                "source_panel_manifest_sha256": row["source_panel"][
                    "source_panel_manifest_sha256"
                ],
                "control_full_geometry_sha256": row["binding"][
                    "control_full_geometry_sha256"
                ],
                "control_structural_geometry_sha256": row["binding"][
                    "control_structural_geometry_sha256"
                ],
                "moment_replay_receipt_sha256": row["moment_replay"][
                    "replay_receipt_sha256"
                ],
                "ready_for_independent_all150": True,
            }
            for index, row in enumerate(prepared)
        ],
        "all_adapter_geometries_and_moments_authenticated_before_fit": True,
        "candidate_fit_calls": 0,
    }


def _output_members(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root)).replace("\\", "/")):
        if path.is_file() and path.name not in {"INCOMPLETE", "COMPLETE.json"}:
            rows.append(
                {
                    "name": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": _sha_file(path),
                }
            )
    return rows


def consume_controls(
    *,
    contract: Any,
    primary_authorization_bytes: bytes,
    run_authorization_bytes: bytes,
    expected_primary_auditor_manifest_sha256: str,
    expected_primary_audit_receipt_sha256: str,
    expected_consumer_source_manifest_sha256: str,
    expected_consumer_auditor_manifest_sha256: str,
    expected_consumer_audit_receipt_sha256: str,
    expected_producer_auditor_manifest_sha256: str,
    expected_producer_audit_receipt_sha256: str,
    expected_v8_all150_preflight_receipt_sha256: str,
    expected_gpu_identity_receipt_sha256: str,
    expected_source_snapshot_root_sha256: str,
    expected_descriptor_source_builder_sha256: str,
    expected_moment_replayer_source_sha256: str,
    expected_audit_bootstrap_sha256: str,
    root_complete_bytes: bytes,
    observed_member_names: Sequence[str],
    member_loader: Callable[[str], bytes],
    moment_replayer: Callable[..., Mapping[str, Any]],
    stage0: Any,
    common: Any,
    protocol: Any,
    container_codec: Any,
    semantic_codec: Any,
    adapter_factory: Callable[[], Any],
    backend_factory: Callable[[], Any],
    authenticated_descriptor_source_builder: Callable[[bytes], Any],
    output_root: Path,
) -> dict[str, Any]:
    """Run the matched nulls after all external and bundle gates close."""
    primary_record = contract.strict_json(
        primary_authorization_bytes, "primary authorization"
    )
    primary = contract.validate_primary_authorization(
        primary_record,
        expected_auditor_manifest_sha256=expected_primary_auditor_manifest_sha256,
        expected_audit_receipt_sha256=expected_primary_audit_receipt_sha256,
    )
    primary_sha = hashlib.sha256(primary_authorization_bytes).hexdigest()
    run_record = contract.strict_json(run_authorization_bytes, "run authorization")
    run_authorization = contract.validate_run_authorization(
        run_record,
        expected_consumer_source_manifest_sha256=expected_consumer_source_manifest_sha256,
        expected_consumer_auditor_manifest_sha256=expected_consumer_auditor_manifest_sha256,
        expected_consumer_audit_receipt_sha256=expected_consumer_audit_receipt_sha256,
        expected_producer_auditor_manifest_sha256=expected_producer_auditor_manifest_sha256,
        expected_producer_audit_receipt_sha256=expected_producer_audit_receipt_sha256,
        expected_v8_all150_preflight_receipt_sha256=expected_v8_all150_preflight_receipt_sha256,
        expected_gpu_identity_receipt_sha256=expected_gpu_identity_receipt_sha256,
        expected_source_snapshot_root_sha256=expected_source_snapshot_root_sha256,
        expected_descriptor_source_builder_sha256=expected_descriptor_source_builder_sha256,
        expected_moment_replayer_source_sha256=expected_moment_replayer_source_sha256,
        expected_audit_bootstrap_sha256=expected_audit_bootstrap_sha256,
        expected_root_complete_sha256=hashlib.sha256(root_complete_bytes).hexdigest(),
        expected_primary_authorization_sha256=primary_sha,
    )
    authentication = contract.authenticate_eight_control_root(
        root_complete_bytes=root_complete_bytes,
        expected_root_complete_sha256=run_authorization[
            "eight_control_root_complete_sha256"
        ],
        observed_member_names=observed_member_names,
        member_loader=member_loader,
        primary_authorization=primary,
    )
    prepared = _prepare_all_controls(
        contract=contract,
        stage0=stage0,
        common=common,
        protocol=protocol,
        adapter_factory=adapter_factory,
        member_loader=member_loader,
        authentication=authentication,
        primary=primary,
        run_authorization=run_authorization,
        moment_replayer=moment_replayer,
    )

    if output_root.exists():
        raise FileExistsError(f"output root exists: {output_root}")
    output_root.mkdir(parents=True)
    incomplete = output_root / "INCOMPLETE"
    _write_new(incomplete, b"UWFA-SC-v9 matched controls incomplete\n")
    input_receipt = _public_input_authentication(authentication, prepared)
    _write_new(
        output_root / "INPUT_AUTHENTICATION.json", contract.pretty_json(input_receipt)
    )
    executed = []
    terminal_override = None
    source_saving = float(primary["source_absolute_saving_bpw"])
    for index, compact_control in enumerate(prepared):
        control = _materialize_control_for_fit(
            contract=contract,
            stage0=stage0,
            common=common,
            protocol=protocol,
            adapter_factory=adapter_factory,
            member_loader=member_loader,
            control=compact_control,
            primary=primary,
            run_authorization=run_authorization,
        )
        backend = backend_factory()
        projection = stage0.projected_updates(common, protocol, control["panel"])
        if (
            projection.get("primary_exact_identity_estimable") is not True
            or projection.get("passes_pre_fit_resource_budget") is not True
            or projection.get("passes_pre_fit_runtime_budget") is not True
        ):
            terminal_override = {
                "status": "BLOCK_CONTROL_RESOURCE_OR_ESTIMABILITY_BEFORE_FIT",
                "seed": control["seed"],
                "projection": projection,
                "specificity_pass": False,
                "positive_claim_authority": False,
            }
            break
        cache = stage0.prepare_backend_cache(backend, control["panel"])
        scientific = _recorded_nested_holdout(
            contract=contract,
            stage0=stage0,
            common=common,
            protocol=protocol,
            container_codec=container_codec,
            backend=backend,
            cache=cache,
            panel=control["panel"],
        )
        if scientific.get("estimable") is not True:
            terminal_override = {
                "status": "BLOCK_CONTROL_HOLDOUT_UNESTIMABLE_AFTER_ADMISSION",
                "seed": control["seed"],
                "specificity_pass": False,
                "positive_claim_authority": False,
            }
            break
        contract.validate_all150_scientific(scientific, common)
        selected = common.candidate_bank()[
            int(
                scientific["final_topology_selected_from_nested_fold_votes"][
                    "selector_ordinal"
                ]
            )
        ]
        physical = stage0.final_container(
            common,
            container_codec,
            semantic_codec,
            control["adapter"],
            backend,
            cache,
            control["panel"],
            selected,
            control["score_validated"],
            control["evidence"],
            authenticated_descriptor_source_builder,
        )
        public_physical = stage0._result_without_payload(physical)
        gain = float(
            public_physical["absolute_saving_vs_bound_current_artifact_bpw"]
        )
        row = {
            "schema": "uwfa-sc-v9-matched-control-all150-result-v1",
            "index": index,
            "seed": control["seed"],
            "scientific_nested_holdout": scientific,
            "final": public_physical,
            "absolute_saving_bpw": gain,
            "repeated_complete_150_cell_selection_fit_pack_decode": True,
            "source_winner_reused": False,
            "moment_replay": control["moment_replay"],
            "positive_claim_authority": False,
        }
        row["control_result_sha256"] = hashlib.sha256(
            contract.canonical_json(row)
        ).hexdigest()
        _write_new(
            output_root / f"CONTROL_{index:02d}.json", contract.pretty_json(row)
        )
        executed.append(row)
        del control, backend, cache, physical
        gc.collect()
        decision = contract.early_null_decision(
            source_saving_bpw=source_saving,
            executed_controls=executed,
            authenticated_controls=8,
        )
        if decision["status"] == "HARD_KILL_MATCHED_GAUSSIAN_NOT_SPECIFIC":
            break
    if terminal_override is not None:
        decision = terminal_override
    else:
        decision = contract.early_null_decision(
            source_saving_bpw=source_saving,
            executed_controls=executed,
            authenticated_controls=8,
        )
    result = {
        "schema": "uwfa-sc-v9-matched-gaussian-controls-result-v1",
        "status": decision["status"],
        "input_authentication_sha256": _sha_file(
            output_root / "INPUT_AUTHENTICATION.json"
        ),
        "primary_authorization_sha256": primary_sha,
        "run_authorization_sha256": hashlib.sha256(
            run_authorization_bytes
        ).hexdigest(),
        "controls_authenticated": 8,
        "controls_executed": len(executed),
        "executed_seeds": [row["seed"] for row in executed],
        "unexecuted_seeds": list(contract.CONTROL_SEEDS[len(executed):]),
        "decision": decision,
        "every_executed_control_recorded_all_150_per_fold": all(
            row["repeated_complete_150_cell_selection_fit_pack_decode"] is True
            for row in executed
        ),
        "source_winner_reuse_forbidden_and_not_observed": all(
            row["source_winner_reused"] is False for row in executed
        ),
        "specificity_pass": decision.get("specificity_pass") is True,
        "positive_claim_authority": False,
        "fresh_independent_result_audit_required": True,
    }
    result["result_sha256"] = hashlib.sha256(contract.canonical_json(result)).hexdigest()
    contract.audit_terminal_result(result=result, control_rows=executed, common=common)
    _write_new(output_root / "RESULT.json", contract.pretty_json(result))
    members = _output_members(output_root)
    complete = {
        "schema": "uwfa-sc-v9-matched-controls-complete-v1",
        "status": result["status"],
        "result_sha256": result["result_sha256"],
        "members": members,
        "members_root_sha256": contract.members_root(
            members, b"UWFA-SC-V9-MATCHED-CONTROLS-OUTPUT-v1\x00"
        ),
        "positive_claim_authority": False,
        "fresh_independent_result_audit_required": True,
    }
    incomplete.unlink()
    _write_new(output_root / "COMPLETE.json", contract.pretty_json(complete))
    return result


def direct_main() -> int:
    print(DIRECT_STATUS, file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(direct_main())
