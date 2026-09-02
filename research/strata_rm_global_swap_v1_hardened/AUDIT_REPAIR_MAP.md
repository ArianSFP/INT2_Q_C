# Independent-v0 audit repair map

This map is normative for the v1 source boundary.  “Repaired” means that the
former exploit is rejected by source/authority code; it does not mean a Qwen
payload result exists.

| Independent-v0 finding | v1 repair | Regression evidence | Remaining hold |
|---|---|---|---|
| The producer verifier ignored unmanifested directories such as `cupy/`. | `authority.authenticate_flat_package`, `authority.authenticate_v1_package`, and `verify_source.py` enumerate the exact entry set and require every entry to be a regular non-link file. Package-root links and member-path links are rejected. | `test_unmanifested_directory_is_rejected`, `test_symlink_member_is_rejected_when_supported`, and `test_exact_closure_enumerates_all_entry_kinds`. | Python tests remain unexecuted until the RunPod returns. |
| `install()` accepted an arbitrary object and arbitrary callable, and did not bind the final hook after import. | The injection API was removed. `current_integration_worker.py` hashes and imports the exact base/BEC modules itself, requires `bg.base is base`, authenticates the live `bg.bec_flags`, installs the local RM closure, and checks the final base hook object. The parent runs a manifest-authenticated source snapshot as `python -I -B`. | `test_worker_command_is_fixed_isolated_and_has_no_hook_argument` and `test_hook_api_has_no_injected_module_or_reference_callable`. | The pinned historical base CLI still gates public runs below global N20/N21; no payload launch is authorized. |
| A fabricated schema receipt with no packet, no decode, and fake hash strings passed. | `physical_authority.validate_physical_bundle` opens a committed nonempty packet, starts a separately hash-pinned decoder worker, reads literal reconstruction files, and compares the independently re-encoded packet to the input bytes. It accepts no receipt or packet object from the caller. Production additionally requires an exact independent-audit closure for that decoder worker. | `test_literal_fixture_decode_and_byte_reencode`, `test_fabricated_no_packet_receipt_is_impossible`, `test_packet_mutation_even_when_recommitted_fails_decoder`, and `test_no_declared_boolean_can_replace_canonical_packet_bytes`. | The bundled decoder is a synthetic mechanism fixture only; a production decoder worker and its independent audit do not exist yet. |
| MSE, F, source identity, Gaussian controls, routed reads, and universal SwiGLU-MoE evidence were absent. | Exact source BF16 files are externally hashed; FP64 source energy/SSE, relative MSE, literal physical bpw, and F are recomputed. Production mode requires matched Gaussian cases with identical geometry/pipeline, expert-local packet-only read traces below 2x, zero untraced shared bytes, exact Gate/Up/Down shape triplets, no Qwen-specific tables, and at least two model architecture families. | `test_exact_bf16_source_metric`, `test_source_mutation_fails_external_pin`, `test_read_trace_requires_literal_packet_object_only`, and `test_production_commitment_requires_matched_and_universal_fields`. | No Qwen, Gaussian-control, portability, or routed-read payload has been produced. |
| The CuPy smoke accepted an injected NumPy facade. | The source package is snapshotted after external manifest authentication. `real_cupy_worker.py` starts in a fresh `python -I -B` child under an allowlisted environment, rejects preloaded backends, rejects backend origins under controlled roots, requires live runtime/driver/device values, synchronizes a GPU arithmetic probe, and checks complete N20/N21 orders against NumPy. | `test_python_import_facade_environment_is_removed`, `test_fake_backend_inside_controlled_root_is_rejected`, plus the pending full real-CuPy worker run. | Real-CuPy execution is pending because the RunPod endpoint refused SSH. |

The only current disposition is:

```text
FROZEN_SOURCE_ONLY_UNEXECUTED__HOLD_RUNPOD_AND_PAYLOAD
```

