# Independent assessment

## Scope and integrity

The reviewed `SOURCE_MANIFEST.json` matches the externally supplied SHA-256
`7901e78eaf7c6b854d7bfaa2afbb4eb7be337449a72ef66e66d00adb87f64ab4`.
All eight members match their byte lengths and hashes, the canonical member
root independently recomputes to
`14ec1fdc19435f4f3655b4f3458ef774a6503d9c88c2d62c510815499c14aecd`,
and there are no surplus entries or linked members.

No producer file was changed. This assessment inspected source and its frozen
metadata only.

## What is genuinely repaired

1. **The production hold is real.** `TRUSTED_LAUNCH_MANIFEST_SHA256` is `None`,
   and `authorize_production()` rejects before resolving evidence or payload
   paths.
2. **Capability containers are exact-closed.** Each launch pin names a kind,
   directory, manifest hash, member root, execution-receipt hash and independent
   audit-receipt hash. Members are rehashed and surplus entries fail.
3. **Fixture flags fail in the production branch.** Literal `test_fixture`,
   `dummy`, or `self_authored` true values are rejected. Producer, executor and
   auditor identifier strings must differ within each capability.
4. **Literal source aliases are strongly checked.** BF16 role files are opened
   and rejected on resolved path, `(device,inode)`, or SHA-256 reuse across all
   routes and roles.
5. **Some physical bindings are independently recomputed.** Packet byte count
   and hash are reopened; receipt score ratios/rates/F are recomputed; read
   event offsets and page hashes are checked against the literal packet; and
   event bytes are included in the reported amplification.

Those are useful mechanics. They do not, however, make the semantic contents
of an execution receipt true.

## Blocking findings

### B1 — “Literal current STRATA” is still an attestation

`_validate_adapter_details()` never receives or opens packet bytes. It accepts
`scale_payload_inside_packet=true`, a positive `scale_bytes`, arbitrary
well-formed transform hashes, framing counts that sum, and
`canonical_reencode_equal=true`.

It does not:

- parse header/payload/trailer offsets;
- extract scale bytes and reproduce `scale_payload_sha256`;
- bind forward/inverse hashes to authenticated current-STRATA implementation
  files;
- decode the six planes or independently prove the 0..63 index range;
- run a canonical encoder or compare its literal output to the packet.

The capability audit receipt's `semantic_replay_verified=true` is itself data
inside the same closure. A successor needs an independently frozen executable
decoder/encoder audit and a literal replay artifact.

### B2 — the “independent BF16 scorer” does not score bytes

`_validate_scorer_details()` divides caller-supplied `sse_fp64` by
`source_energy_fp64` and correctly recomputes `R` and `F`. It never parses BF16,
opens reconstruction bytes, or recomputes either sum from samples.

The source hashes and reconstruction hash are joined across receipts, but no
reconstruction artifact is required or hash-checked. Therefore invented SSE
and energy totals can be internally consistent and pass.

### B3 — counts can manufacture rate and distortion

The following equalities are absent:

```text
scorer.weight_count == sum(literal BF16 bytes)/2
scorer.weight_count == adapter.decoded_weight_count
reconstruction FP64 count == adapter.decoded_weight_count
```

An arbitrary large scorer count lowers `physical_rate_bpw`; an arbitrary score
total lowers relative MSE. This directly compromises the claimed target.

### B4 — packet aliases are accepted

Source BF16 files have path/inode/content alias maps, but STRATA packets do
not. Two routed experts may reference the same packet path, hard link or byte
hash. This violates a literal one-private-packet-per-routed-expert claim and can
mask shared-read accounting.

### B5 — the trace need not cover the decoder's packet

The read validator checks only events present in the receipt. It never requires
the observed page set to equal the packet's required page set. For a multi-page
packet, one correctly hashed page can be reported and produce amplification
below one. `one_routed_expert_only=true` and `instrumented_reads=true` are also
receipt booleans; no isolated decoder is run by the authority.

A physical successor should expose packet data only through an instrumented
host capability and derive the exact page ledger from those calls.

### B6 — capability output hashes do not bind output artifacts

Common execution validation requires `output_sha256` to look like a hash, but
does not require a corresponding output member or bind that digest to
`details`. Invocation and input-manifest hashes are likewise not joined to the
launch object. Exact-closing a receipt authenticates its bytes, not the event it
describes.

### B7 — independence is nominal and incomplete across capabilities

Authority IDs are unsigned strings. They must differ only within one
capability. The independent-launch-audit producer/executor/auditor may reuse
identities from capabilities it purportedly audits, and the launch issuer may
reuse them as well. A false `self_authored=false` value is structurally valid.

This can be acceptable only when an external controller separately authenticates
and pins genuinely independent packages. It is not established by this source
authority itself.

### B8 — launch self-binding and the lower-level result are unsafe to quote

The launch's `v3_source_manifest_sha256` and `v3_source_root_sha256` need only be
well-formed hashes; they are not compared with this reviewed producer. Also,
`verify_precommitted_evidence(..., allow_source_test_fixture=False)` returns
`production_authorized=true` from a caller-supplied digest even though the
docstring says only `authorize_production()` is authoritative.

The compiled hold prevents exploitation in this freeze. A successor should
bind the exact producer lineage and make the lower-level verifier always return
non-authoritative status.

### B9 — scientific control acceptance is absent

Eight control routes are required, but the authority never computes a
Qwen-minus-strongest-control advantage. It does not authenticate Qwen identity
or require multiple non-aliased SwiGLU-MoE families, and model `R/F` is pooled
globally rather than enforced per claimed family. This is insufficient for a
universal source-specific below-Gaussian claim.

## Additional hardening

- Reject the launch path object as a symlink before calling `resolve()`.
- Reopen the literal pinned v2 and audit packages in the production entry
  rather than accepting only a predecessor capability's booleans.
- Bind control generator outputs, selected reconstructions and moment-matching
  records to literal artifacts, not only hashes written in receipts.
- Require unique `(architecture_family, layer, expert)` route identities and
  unique packet path/inode/hash ownership.

## Disposition

The exact source closure and fail-closed production hold pass. Runtime and RD
approval do not. The package should remain source-only until B1–B9 are repaired
and independently reviewed.

```text
PASS_EXACT_SOURCE_AND_EFFECTIVE_COMPILED_HOLD__BLOCK_LITERAL_REPLAY_SCORING_COUNT_PACKET_ALIAS_AND_TRACE_CLAIMS__HOLD_PAYLOAD_AND_RD
```
