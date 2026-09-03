# Independent source review: global STRATA RM swap v4 Wasmtime authority

Date: 2026-09-03

Producer pins supplied out of band:

- manifest SHA-256: `62bf04cd413317e2e8b98635713419c84394db7b7d2bd4567afddf56957a5e2f`
- source root SHA-256: `f535699c4828a02e5769b916b1207309768f7381db5f92a0fb58e10915ae8a25`

This is a benign source-only review. It did not connect to a network, open or
enumerate model/checkpoint/packet/control payloads, execute Wasmtime, edit the
producer, or make an RD claim.

## Result

The v3 repair mechanisms are materially present.

- The producer is an exact nine-member flat closure. Every byte count and
  digest matches, and the independently recomputed canonical row root equals
  the supplied source root.
- The Wasmtime distribution authority authenticates an exact recursive tree,
  Python modules, distribution metadata, native-library inventory, capability,
  executed audit receipt, and five distinct external pins. The sandbox imports
  `wasmtime` from the copied snapshot and rehashes the tree and mapped snapshot
  libraries after execution.
- `Config.consume_fuel` is enabled before compilation. A fresh decoder Store
  and encoder Store each receive `set_limits(...)` and `set_fuel(...)` before
  their respective instantiation. The externally audited runtime must report
  successful limiter and fuel-exhaustion probes.
- The routed packet is read into an immutable host `bytes` object. The decoder
  receives only `authority.read_packet`, which range-checks source and guest
  destination ranges, rejects overlapping source intervals, and must cover the
  complete literal packet exactly once. No path, descriptor, WASI interface,
  mutable host buffer, or native read capability is imported by the guest.
- The physical decoder and canonical encoder are distinct pinned Wasm objects.
  The encoder has zero imports and is instantiated in a separate Store with
  only the decoder-produced semantic-state bytes copied into its memory. It
  has no direct packet callback or packet object.
- V4 delegates to the exactly pinned v3 scientific authority and acceptance
  code. The per-family physical-rate, absolute-F, strongest-control advantage,
  architecture-family, distinct-packet, one-routed-expert, and page-rounded
  read gates are therefore preserved without accepting caller-provided scores.

## Conditional boundaries and gaps

### R1 — external audit trust is essential

The authority validates exact bytes against caller-supplied expected hashes,
but it has no built-in approved audit roots, signer identity, or signature.
The public authorization string is an explicit mode gate, not authentication.
The producer's own fixtures demonstrate that a caller can create dummy runtime
and invalid Wasm objects, write PASS receipts, supply the resulting hashes, and
have the individual audit-package authenticators accept them.

This does not defeat a deployment in which an independent reviewer publishes
and the launcher separately pins the five runtime and seven semantic authority
hashes. It means `validate_physical_bundle(...)` is not a standalone trust
oracle: the provenance of those expected pins is part of the trusted computing
base and must be recorded outside the evidence producer.

### R2 — semantic-state purity is audit-enforced, not runtime-enforced

The canonical encoder has no direct packet capability. However,
`semantic_state` is treated as opaque bytes by the sandbox. The host checks its
maximum length and copies it to the encoder, but does not parse it against a
field-level format or prove that it excludes a verbatim packet. A colluding
decoder and encoder could therefore relay the packet through that buffer.

The separately pinned semantic audit explicitly has to rule this out and prove
complete decisions, causal regeneration, alias rejection, and uniqueness. The
repair is sound only under that external audit trust; the declarative
`raw_packet_bytes_permitted: false` field alone is not enforcement.

### R3 — executing-host provenance is not fully closed

The capability records `python_abi`, `platform_tag`, and compilation `target`,
and an earlier runtime-audit receipt asserts its observed target. The production
sandbox does not compare those values with its current Python ABI/platform/
machine before execution. It also proves that every native image copied inside
the runtime snapshot is mapped and unchanged, but it deliberately permits
mapped native dependencies outside that snapshot. Consequently this pins the
Wasmtime distribution and bundled native tree, not the full executing Python,
OS, loader, or transitive native-image closure.

For reproducible production authority, either bind the execution host image
and interpreter separately or add current-host ABI/platform/target checks plus
an explicit allowlisted/pinned external-native dependency policy.

### R4 — the read metric remains narrow

The inherited acceptance gate measures one page-rounded supply of the literal
expert packet through the callback. It does not count the parent evidence read,
temporary snapshot write/read, callback copy, decoder linear-memory loads, or
downstream GEMM reads. The `<2x` result is therefore a routed expert-packet
cold-supply metric, not total process or inference memory bandwidth.

### R5 — runtime remains pending

The frozen producer says all source tests, Wasmtime probes, runtime audit,
semantic audit, payload runs, controls, and RD results are unexecuted. Python
and Wasmtime were unavailable in this local review environment, so the
independent Python suite is frozen but not represented as passed.

## Disposition

```text
PASS_V3_IMMUTABILITY_LIMITS_SEMANTIC_AND_DELEGATED_GATE_REPAIRS__
CONDITIONAL_ON_EXTERNALLY_TRUSTED_EXECUTED_RUNTIME_AND_SEMANTIC_AUDITS__
HOLD_EXECUTING_HOST_AND_TRANSITIVE_NATIVE_PROVENANCE__
HOLD_TOTAL_READ_BANDWIDTH_RUNTIME_PAYLOAD_AND_RD
```

## Replay

From this review directory:

```bash
python -I -B run_review.py \
  --producer ../strata_rm_global_swap_v4_wasm_authority \
  --output /tmp/strata-rm-v4-independent-source-review.json
```

Exact review closure:

```bash
python -I -B verify_review_source.py \
  --package . \
  --expected-manifest-sha256 REVIEW_MANIFEST_SHA256
```

