#!/usr/bin/env python3
"""Independent hostile checks for the exact UWFA-SC v3 pre-freeze snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_ROOT = "a1fc85ffdfaa5e7fde25deea98b33d186c915868a8a546333a7fefb64fa9b035"
EXPECTED = {
    "INDEPENDENT_BOOTSTRAP_ABI.md": (7165, "5a43ff712395fb8c7d8354edd0a83960bba96940b99aaf2b2175c94672c77513"),
    "README.md": (9776, "57a114c12121d98fc52ba5de5d44790713c19d83800400ed1e87a6c2e900bae2"),
    "container_codec.py": (80409, "3c81ea3e67a7908a0e28ff05c2fd3f17d7404a24ca44710d0acfdd97d8f4d8d5"),
    "cupy_backend.py": (35251, "e717d602086457a5e3b5fb0746d7a67e9a3584090a22ad675b6e0206e4212424"),
    "design_lock.json": (8203, "f57cc432dc39ac72a83ae3781bff380cf6d8a96b55e26278fd9059e47799d630"),
    "dispatcher_contract.py": (9205, "2a231ef7e7b37f296387bc7825567e372736ced1769917ba7b97839924e83bbf"),
    "fixture_long_memory.py": (4307, "1d425b56ea0923e74996b488ea7c12ef0b70569df19c5468c36812648bb3f6ff"),
    "fixture_portability.py": (11265, "71a8c1eb2c5dad9f6b8e66f106547f09b0bfff449be1aa785b363d3055e2318d"),
    "protocol.py": (17596, "2b5ea430bb73a715c2eda08de359d874fa8a5a823d825f8256d2dff230f6b4f0"),
    "result_envelope.py": (2688, "9ada6c9b6a5fcb57fb8972e05e519e8aada68aeabb740ce3b67bd318cf2b7993"),
    "stage0_census.py": (59116, "fd71686644e13253293644d3793adaacdc1e9977b792771c75f56e08288d89c7"),
    "strata_sc_adapter.py": (36184, "cfdb1f887fc1473f67aa758cd45570d9fd58b33765443e6c87581a43f1435bc7"),
    "test_source_only.py": (59352, "e9025bf1ee1702c5778b6527657fdcaf4edc38229d2c8422284944819c1835ef"),
    "universal_adapter.py": (11577, "dae13363c23e3a59a071b16b36ad282fc71b5e08be158539690e82eefbcbc899"),
    "uwfa_common.py": (33639, "dea23efa7211715ff6fa654cbf98452dc08a318d54874869db2321649d511397"),
    "verify_source.py": (10277, "72926daba955af6bdb3a701bd0192a9fa8e5271e33ab3b9a85920e2103408edf"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def authenticate(package: Path) -> dict[str, bytes]:
    package = package.absolute()
    cursor = Path(package.anchor)
    for component in package.parts[1:]:
        cursor = cursor / component
        info = os.lstat(cursor)
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"symlink package ancestor: {cursor}")
    if {entry.name for entry in os.scandir(package)} != set(EXPECTED):
        raise RuntimeError("exact 16-member inventory mismatch")
    snapshots: dict[str, bytes] = {}
    canonical = bytearray()
    for name in sorted(EXPECTED, key=str.lower):
        path = package / name
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"non-regular member: {name}")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            chunks = []
            while chunk := os.read(fd, 1 << 20):
                chunks.append(chunk)
            data = b"".join(chunks)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError(f"member changed while held: {name}")
        expected_bytes, expected_sha = EXPECTED[name]
        if len(data) != expected_bytes or sha(data) != expected_sha:
            raise RuntimeError(f"member authentication failed: {name}")
        snapshots[name] = data
        canonical.extend(name.encode("utf-8") + b"\0" + str(len(data)).encode("ascii") + b"\0" + expected_sha.encode("ascii") + b"\n")
    if sha(bytes(canonical)) != EXPECTED_ROOT:
        raise RuntimeError("canonical inventory root mismatch")
    return snapshots


def load_snapshot(name: str, source: bytes) -> Any:
    module = types.ModuleType(name)
    module.__file__ = f"<independently-authenticated:{name}>"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec", dont_inherit=True, optimize=0), module.__dict__)
    return module


def panel(common: Any, protocol: Any, identities: list[tuple[int, int]]) -> dict[str, Any]:
    rows = []
    for expert in range(6):
        for role_index, role in enumerate(("gate", "up", "down")):
            ordinal = 3 * expert + role_index
            owner = protocol.owner_set_from_ordinals(6, [expert])
            rows.append({
                "stream_ordinal": ordinal,
                "owner_set_hex": owner.hex(),
                "owner_set": owner,
                "owner_contributions": ({"expert": expert, "role": role, "source_offset": 0, "weight_count": 1},),
                "owner_expert_ordinals": [expert],
                "owner_identity_indices": [expert],
                "owner_weight_contributions": {expert: 1},
                "weight_charge": 1,
                "shape_rows": 1,
                "shape_cols": 1,
                "role": role,
                "symbols": 31 + ordinal,
                "bits": [ordinal & 1],
                "levels": [ordinal % common.LEVELS],
                "base": [32768],
                "bits_bytes": bytes((ordinal & 1,)),
                "levels_bytes": bytes((ordinal % common.LEVELS,)),
                "base_bytes": (32768).to_bytes(2, "little"),
                "baseline_payload_bytes": 5,
                "baseline_logical_bits": 31 + ordinal,
                "profile_q": 0,
                "decoder_scale": 1.0,
                "logn": 1,
            })
    return {
        "streams": rows,
        "weights": 18,
        "experts": 6,
        "artifact": {"raw_bytes": 1},
        "immutable_state": b"",
        "semantic_identities": identities,
        "expert_shapes": [{"expert": index, "hidden": 1, "intermediate": 1} for index in range(6)],
        "reconstruction": {"full_reconstruction_f64_sha256": "44" * 32},
    }


def source_phase_binding_probe(common: Any, protocol: Any, stage: Any) -> dict[str, Any]:
    artifact = b"x"
    score_bytes = b"{}"
    actual_geometry = "aa" * 32
    forged_geometry = "bb" * 32
    source_panel = {
        "artifact": {"raw_sha256": sha(artifact), "raw_bytes": len(artifact)},
        "weights": 1,
        "experts": 1,
        "streams": [],
        "reconstruction": {"full_reconstruction_f64_sha256": "44" * 32},
    }
    evidence = stage.BoundEvidence(
        baseline_plan_sha256="10" * 32,
        baseline_score_sha256=sha(score_bytes),
        universal_decoder_sha256="12" * 32,
        producer_manifest_sha256="13" * 32,
        audit_bootstrap_sha256="14" * 32,
        source_panel_sha256=forged_geometry,
        extraction_program_sha256="16" * 32,
        universal_adapter_sha256="17" * 32,
        pipeline_sha256="18" * 32,
    )
    replacements = {
        (stage, "prepare_panel"): lambda *_args: source_panel,
        (common, "strict_json_loads"): lambda _raw: {},
        (protocol, "validate_score_receipt"): lambda *_args, **_kwargs: {"relative_mse": 0.025},
        (stage, "projected_updates"): lambda *_args: {"passes_pre_fit_runtime_budget": True},
        (stage, "prepare_backend_cache"): lambda *_args: object(),
        (stage, "nested_holdout"): lambda *_args: {
            "final_topology_selected_from_nested_fold_votes": {"selector_ordinal": 0},
            "passes_heldout_gate": False,
        },
        (stage, "final_container"): lambda *_args, **_kwargs: {
            "container": b"c",
            "identity_framing_container": b"i",
            "parsed_metrics": {"passes_rate_interval": True, "passes_F_target": True, "passes_cold_read_below_2x": True},
            "standalone_decode": {"all_payloads_canonically_reencoded": True},
            "identical_reconstruction_proved_by_full_f64_digest": True,
            "all_adapted_values_deserialized_from_transmitted_model": True,
            "absolute_saving_vs_bound_current_artifact_bpw": 0.0,
        },
        (protocol, "geometry_sha256"): lambda *_args: actual_geometry,
    }
    originals = {(id(module), name): getattr(module, name) for module, name in replacements}
    try:
        for (module, name), value in replacements.items():
            setattr(module, name, value)
        result = stage.source_phase(
            common=common,
            protocol=protocol,
            container_codec=object(),
            semantic_codec=object(),
            adapter=object(),
            backend=object(),
            artifact_bytes=artifact,
            score_receipt_bytes=score_bytes,
            bindings=evidence,
            gpu_preflight={"status": "FORGED_PRETEND_PASS"},
            authenticated_descriptor_source_builder=lambda _raw: None,
        )
    finally:
        for module, name in replacements:
            setattr(module, name, originals[(id(module), name)])
    return {
        "actual_geometry": actual_geometry,
        "bound_geometry": forged_geometry,
        "source_phase_returned": True,
        "returned_status": result["status"],
        "forged_gpu_preflight_preserved_without_gate": result["gpu_preflight"]["status"],
    }


def unequal_shape_portability(common: Any, semantic: Any, codec: Any, fixture: Any) -> dict[str, Any]:
    experts = 250
    shapes = tuple(
        semantic.ExpertShape(index, 1 + index % 3, 2 + index % 4)
        for index in range(experts)
    )
    semantic_packet = semantic.build_semantic_packet(shapes, b"independent-unequal-shape-e250")
    candidate = common.Candidate("suffix", 2, 32)
    frequencies = [32768] * common.model_frequency_count(candidate)
    model_packet = common.serialize_model(candidate, frequencies)
    tail = {(0, "gate"): 1, (experts - 1, "gate"): 2}
    raw_specs: list[tuple[str, tuple[tuple[int, int, int], ...]]] = []
    for expert, shape in enumerate(shapes):
        matrix = shape.matrix_weights
        for role in semantic.ROLES:
            count = matrix - tail.get((expert, role), 0)
            raw_specs.append((role, ((expert, 0, count),)))
    raw_specs.append((
        "gate",
        (
            (0, shapes[0].matrix_weights - 1, 1),
            (experts - 1, shapes[-1].matrix_weights - 2, 2),
        ),
    ))
    stream_specs = []
    references = []
    for ordinal, (role, contribution_rows) in enumerate(raw_specs):
        symbols = 17 + ordinal % 5
        levels, base = fixture.public_context_rows(symbols)
        bits = [((position * 29) ^ (ordinal * 17) ^ (position >> 2)) & 1 for position in range(symbols)]
        payload, logical_bits = common.encode_unifilar_stream(bits, levels, base, candidate, frequencies)
        contributions = tuple(
            codec.OwnerContribution(expert, role, begin, count)
            for expert, begin, count in contribution_rows
        )
        owner_set = codec.owner_set_from_ordinals(experts, [row.expert for row in contributions])
        source_weights = sum(row.weight_count for row in contributions)
        spec = codec.StreamSpec(
            ordinal=ordinal,
            symbols=symbols,
            logical_bits=logical_bits,
            payload=payload,
            source_digest=sha(bytes(bits)),
            profile_q=0,
            decoder_scale=1.0,
            role=role,
            group_rows=1,
            group_cols=source_weights,
            owner_contributions=contributions,
        )
        stream_specs.append((owner_set, spec))
        references.append(bits)
    grouped: dict[bytes, list[Any]] = defaultdict(list)
    for owner_set, spec in stream_specs:
        grouped[owner_set].append(spec)
    owner_sets = sorted(
        grouped,
        key=lambda value: (len(codec.owner_ordinals(value, experts)) != 1, codec.owner_ordinals(value, experts)),
    )
    regions = tuple(
        codec.RegionSpec(owner_set, tuple(sorted(grouped[owner_set], key=lambda row: row.ordinal)))
        for owner_set in owner_sets
    )
    matrices = {
        (expert, role): [None] * shapes[expert].matrix_weights
        for expert in range(experts)
        for role in semantic.ROLES
    }
    for (_owner_set, spec), bits in zip(stream_specs, references, strict=True):
        for contribution in spec.owner_contributions:
            target = matrices[(contribution.expert, contribution.role)]
            for local in range(contribution.weight_count):
                position = contribution.source_offset + local
                if target[position] is not None:
                    raise RuntimeError("unequal-shape reconstruction overlap")
                target[position] = fixture._scalar_bytes(bits[local % len(bits)], spec.ordinal, position)
    digest = hashlib.sha256()
    for expert in range(experts):
        for role in semantic.ROLES:
            values = matrices[(expert, role)]
            if any(value is None for value in values):
                raise RuntimeError("unequal-shape reconstruction hole")
            digest.update(b"".join(values))
    reconstruction = digest.hexdigest()
    weights = sum(shape.expert_weights for shape in shapes)
    binding_hashes = {
        name: sha(("independent-e250:" + name).encode("ascii"))
        for name in codec._HEADER_BINDINGS
    }
    raw, _diagnostic_metrics = codec.build_container(
        common,
        semantic,
        model_packet=model_packet,
        semantic_packet=semantic_packet,
        immutable_state=b"",
        regions=regions,
        weights=weights,
        experts=experts,
        baseline_object_bytes=10_000_000,
        audited_relative_mse=0.025,
        baseline_artifact_sha256="11" * 32,
        reconstruction_sha256=reconstruction,
        audit_binding_sha256="33" * 32,
        binding_hashes=binding_hashes,
    )
    parsed = codec.parse_container(common, semantic, raw)
    if codec.canonical_rebuild(common, semantic, parsed) != raw:
        raise RuntimeError("unequal-shape canonical rebuild")
    with tempfile.TemporaryFile() as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        source = codec.AuthenticatedDescriptorSource(handle.fileno(), sha(raw))
        try:
            metrics = codec.physical_metrics(
                common,
                semantic,
                parsed,
                routed_descriptor_source=source,
                externally_authenticated_container_sha256=sha(raw),
                routed_decoder=fixture.FixtureRoutedDecoder(common),
            )
        finally:
            source.close()
    return {
        "experts": experts,
        "streams": len(stream_specs),
        "regions": len(regions),
        "weights": weights,
        "shape_pairs": len({(shape.hidden, shape.intermediate) for shape in shapes}),
        "shared_unequal_tail_owners": [0, experts - 1],
        "shared_unequal_tail_counts": [1, 2],
        "canonical_rebuild": True,
        "all_routed_payloads_canonically_reencoded": all(
            row["causal_decode_reencode_reconstruction"]["all_payloads_canonically_reencoded"]
            for row in metrics["experts"]
        ),
        "routed_full_reconstruction_matches": metrics["routed_full_reconstruction"]["matches_container_reconstruction"],
        "container_sha256": sha(raw),
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: hostile_audit.py ABSOLUTE_PACKAGE")
    snapshots = authenticate(Path(sys.argv[1]))
    common = load_snapshot("uwfa_v3_hostile_common", snapshots["uwfa_common.py"])
    protocol = load_snapshot("uwfa_v3_hostile_protocol", snapshots["protocol.py"])
    semantic = load_snapshot("uwfa_v3_hostile_semantic", snapshots["universal_adapter.py"])
    codec = load_snapshot("uwfa_v3_hostile_codec", snapshots["container_codec.py"])
    stage = load_snapshot("uwfa_v3_hostile_stage", snapshots["stage0_census.py"])
    cupy_backend = load_snapshot("uwfa_v3_hostile_cupy", snapshots["cupy_backend.py"])
    strata = load_snapshot("uwfa_v3_hostile_strata", snapshots["strata_sc_adapter.py"])
    fixture = load_snapshot("uwfa_v3_hostile_fixture", snapshots["fixture_portability.py"])

    same_layer = panel(common, protocol, [(0, index) for index in range(6)])
    split_exception = None
    try:
        stage.projected_updates(common, protocol, same_layer)
    except Exception as exc:
        split_exception = f"{type(exc).__name__}: {exc}"

    source_panel = panel(common, protocol, [(index, index) for index in range(6)])
    altered_control = json.loads(json.dumps(protocol.panel_geometry(source_panel)))
    source_geometry = protocol.geometry_sha256(common, source_panel)
    altered_panel = panel(common, protocol, [(index, index) for index in range(6)])
    altered_panel["streams"][0]["baseline_payload_bytes"] += 1
    altered_geometry = protocol.geometry_sha256(common, altered_panel)

    symlink_output_accepted = False
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        real = root / "real"
        alias = root / "alias"
        real.mkdir()
        alias.symlink_to(real, target_is_directory=True)
        final = alias / "result"
        try:
            with common.CompletionLastOutput(final) as transaction:
                transaction.complete(list(transaction.members), "ab" * 32)
            symlink_output_accepted = (real / "result" / "COMPLETE.json").is_file()
        except Exception:
            symlink_output_accepted = False

    bit_rows = bytes((0, 1, 1, 0))
    level_rows = bytes((0, 1, 2, 3))
    base_rows = b"\x00\x80" * 4
    strata_source_digest = strata._stream_digest(bit_rows, level_rows, base_rows)
    bit_only_digest = sha(bit_rows)

    report = {
        "schema": "uwfa-sc-v3-independent-hostile-probes",
        "authenticated_root": EXPECTED_ROOT,
        "inventory_members": len(snapshots),
        "probes": {
            "universal_same_layer_semantic_identities": {
                "unique_identity_pairs": True,
                "expected_exact_pair_exclusion_development_streams_per_fold": 15,
                "observed_exception": split_exception,
            },
            "source_binding_and_preflight": source_phase_binding_probe(common, protocol, stage),
            "control_geometry_payload_length_coupling": {
                "source_geometry": source_geometry,
                "altered_only_baseline_payload_bytes_geometry": altered_geometry,
                "digests_differ": source_geometry != altered_geometry,
                "validated_source_geometry_snapshot": altered_control["weights"] == 18,
            },
            "completion_output_symlink_ancestor": {
                "accepted_and_completed_through_symlink_ancestor": symlink_output_accepted,
            },
            "gpu_memory_bound": {
                "max_packed_symbols": cupy_backend.MAX_PACKED_SYMBOLS,
                "payload_device_bytes_at_declared_max": 4 * cupy_backend.MAX_PACKED_SYMBOLS,
                "stage_vram_budget_bytes": stage.MAX_VRAM_BYTES,
                "declared_max_exceeds_stage_vram_budget": 4 * cupy_backend.MAX_PACKED_SYMBOLS > stage.MAX_VRAM_BYTES,
            },
            "gpu_device_mapping_receipt": {
                "environment_source_mentions_uuid": "uuid" in snapshots["cupy_backend.py"].decode("utf-8").lower(),
                "environment_source_mentions_pci": "pci" in snapshots["cupy_backend.py"].decode("utf-8").lower(),
            },
            "posterior_handoff_decision_digest_label": {
                "strata_source_digest": strata_source_digest,
                "bit_only_sha256": bit_only_digest,
                "source_digest_is_not_bit_only_digest": strata_source_digest != bit_only_digest,
                "handoff_field_name": "decoded_symbol_bits_sha256",
            },
            "unequal_shape_e250_descriptor_portability": unequal_shape_portability(
                common, semantic, codec, fixture
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
