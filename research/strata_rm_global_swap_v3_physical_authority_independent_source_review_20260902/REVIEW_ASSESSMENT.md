# V3 source assessment

## Authenticated scope

- Producer manifest: `9105dd69a2a82d1eaf14e176e4334189a4c31be840dafee467d243c231788e83`
- Producer source root: `83d79990515fca16387723cdea544d41fac76413fe80f919c30517d14551d6ad`
- Pinned v2 root: `e9ce4c24017831fab50696c2c5d81739d1f24d8121075c3aa56612b9a77013c9`
- Pinned v2-review root: `d642889efcf8c54173eb7659602181cb9e71e122ce11ff05da6b24e45c47a113`

Every member of the flat producer package was read and independently hashed.
No payload-like file was accessed.

## Scientific-audit authentication

The v2 provenance gap is repaired in source:

- manifest, audit-source root, receipt, and capability each have distinct
  out-of-band SHA-256 arguments;
- the manifest and receipt must be canonical JSON;
- the audit-source closure is exact, flat, regular, and independently rooted;
- the capability and executed PASS receipt are separately pinned and bound to
  the source root;
- the receipt must attest checkpoint/tensor inspection, source rehashing,
  generator and moment replay, family identity, alias rejection, selection
  replay, and at least twelve hostile tests.

As with any audit capability, independence ultimately rests on who controls
the external pins. Within that trust model, v3 now authenticates scientific
provenance as strongly as its decoder audit.

## Family and source aliases

Exact cross-family reuse is rejected for checkpoint-manifest, tensor-manifest,
checkpoint-identity, and architecture-schema hashes. Every source path is
globally unique, a source-byte hash cannot cross family boundaries, and paired
model/control hashes must be disjoint. Model/control routes are paired by
family, route ID, layer, expert, and tensor shapes.

This establishes exact-byte non-aliasing. It cannot prevent semantically
equivalent evidence being repackaged under different hashes; the mandatory
independent scientific audit's `family_identity_verified` attestation is the
appropriate remaining trust boundary.

## Routed-expert and page ledger

The physical unit is now correctly constrained to one ordered Gate/Up/Down
triplet sharing one `(layer, expert)` and compatible shapes. Each audited route
has one unique packet path and packet hash, shared/common streams are excluded,
and `_run_route` launches a fresh decode for every route.

The ledger is arithmetically correct for its declared cold-supply metric:

```text
pages = ceil(literal_packet_bytes / 4096)
physical_page_bytes = pages * 4096
cold_read_amplification = physical_page_bytes / literal_packet_bytes
```

It validates exact page indices, literal coverage, 4 KiB supplied bytes per
page, and final-page zero padding. Acceptance takes the maximum routed-expert
ratio within each family and requires it to be strictly below 2.

This measures one cold page-padded transfer into the decoder. It does not count
repeated linear-memory loads performed by the Wasm program or downstream GEMM
reads. The result should therefore be named and interpreted as expert-packet
cold-supply amplification, not total decoder/runtime memory bandwidth.

## Wasm isolation and packet buffer

The source establishes the intended capability boundary:

- `module.imports` must be exactly empty;
- the module is instantiated directly with `[]`, without a linker or WASI;
- packet bytes are opened only by the trusted Python host and copied into the
  module's private linear memory;
- reconstruction and canonical-output ranges are disjoint and 64-byte aligned;
- the complete padded packet region is compared before and after decoding;
- total linear memory is required to be at most 1 GiB before and after calls;
- canonical output length and bytes must equal the literal packet;
- the module, host, packet, and request are copied into fresh snapshots and the
  executable snapshots are rehashed after execution.

Zero imports genuinely prevent the Wasm module from obtaining paths, file
descriptors, WASI, sockets, clocks, randomness, callbacks, or native I/O from
the host. This closes the Python-handle bypass in v2.

## Residual runtime and semantic limits

1. **Input mutation is end-state, not write protection.** The host compares the
   packet region only before `decode_route` and after `canonical_reencode`.
   A module can modify packet or padding bytes and restore them before the final
   comparison. Thus `packet_input_unchanged` is valid as a final-state claim,
   while `input_immutability_verified` requires independent module analysis or
   per-call checks. At minimum, compare immediately after each exported call;
   true write prevention requires a separate read-only memory mechanism or
   verified Wasm instrumentation.
2. **The memory cap is post hoc.** The host checks initial and final linear
   memory sizes, but no Wasmtime `Store` limiter or module maximum is required.
   A module can attempt excessive `memory.grow` or consume unbounded CPU before
   the post-call check. The subprocess timeout fails closed, but a pinned store
   limit and fuel/epoch budget are needed for robust bounded execution.
3. **Replay equality is not semantic canonicality by itself.** A
   `canonical_reencode` export can simply copy the input. Byte equality proves
   stable replay, not unique encoding, complete packet consumption, rejection
   of malleable aliases, or regeneration of all decoder decisions. The required
   decoder audit must test alternate/noncanonical packets and bind semantic
   decode followed by independent re-encode.
4. **Wasmtime provenance is not pinned.** The host imports whichever system
   `wasmtime` package is installed and records neither its version nor its
   module/native-library hashes. The producer honestly holds Wasmtime runtime
   authority; reproducible physical execution should pin and report the engine,
   Python binding, configuration, target, and relevant native library hashes.

These limits do not reopen filesystem or source leakage. They prevent this
source-only package from independently proving the stronger mutation, resource,
and canonicality statements until its mandated runtime and decoder audits run.

## Rate-distortion acceptance

Literal packet bytes determine rate. BF16 sources and FP64 reconstructions
determine SSE, energy, relative MSE, F, and saving. All families separately
face the rate, F, maximum routed cold-supply, and strongest-control gates;
absolute pooled Qwen F must also be at most 0.8. No caller metrics are accepted.

## Disposition

V3 fixes the two blocking v2 authority errors: scientific provenance is now an
authenticated audit package, and read accounting is one packet per independently
routed expert. The zero-import Wasm boundary is a sound isolation design.
Promotion still requires a pinned Wasmtime execution and an independent decoder
audit that substantively tests transient mutation, resource limits, and semantic
canonicality.

```text
PASS_V3_STATIC_AUTHORITY_REPAIRS__CONDITIONAL_ON_PINNED_WASMTIME_AND_DECODER_AUDIT_OF_TRANSIENT_MUTATION_MEMORY_AND_CANONICALITY__HOLD_PAYLOAD_AND_RD
```

