# Independent source audit: UNIPOLAR-N18-307 v4

Verdict: **the fixed packet grammar and the Qwen exact-rate arithmetic pass;
payload promotion remains blocked pending narrow source repairs and a separate
runtime/source authorization.**

This sibling audit binds the untouched producer directory
`../tactic_actual_coarse_n18_v4` at source root
`1f9f2c92df3796f5f23b7e3a6b0826d6d8a2ea53bc70014fb75e61e7bc8a9fbf`.
It does not modify v4 and grants no payload, CUDA, SSH, output, or result
authority.

## What passed

- The record is exactly 78,592 bytes for 262,144 values:
  `8 × 78,592 × 128 = 262,144 × 307`, hence exactly `307/128 =
  2.3984375` bpw.
- Its 78,464-byte payload has 627,712 physical bits. The frozen
  `153/64` test channel uses 626,688 nominal bits, leaving exactly 1,024
  logical reserve bits after separately charging the 128-byte header.
- The Qwen `768 × 2048` role geometry has six complete N18 records per role,
  18 per expert, no tail, 1,414,656 coarse bytes per expert, and exact
  `307/128` bpw.
- Logical overflow is rejected before packet packing, and the numerical
  producer calls `run_trial` once per tile without an overflow retry loop.
- Tail records literally pad with BF16 `+0`, charge a complete record, expose
  only the valid prefix, and are marked target-ineligible whenever
  `intermediate × hidden` is not divisible by 262,144.
- The decoder does not import the numerical encoder, independently decodes
  causal arithmetic symbols, re-encodes their bytes, and enforces equality at
  every individual record.
- The producer's 17 source-only hostile tests pass. The checked-in RTX 5090
  source-free CuPy smoke was independently rerun and reproduced all decisive
  fields exactly: packet SHA-256
  `7a15e96389cab06f9de16ca36ff3c9a74580d8d13d5c0a351af02ab1733cfce1`,
  626,926 logical bits, 786 reserve bits, exact canonical re-encode, and
  `0.03693951352239193` original-coordinate relative MSE.
- The frozen planning ledger is arithmetically correct: one private fetch plus
  six common pages is `365/360 = 73/72`; fetching the private frame twice is
  `724/360 = 181/90 = 2.011111...`, outside the strict `<2×` bound.

## Repair-before-payload findings

1. `independent_decoder.py` reports `canonical_reencode_all_match` via
   `all(row.canonical_packet for row in decoded_rows)`. That checks nonempty
   byte strings, not equality. Individual `decode_reservoir` calls already
   require byte-identical re-encoding, so mechanics are protected, but the
   aggregate receipt expression must be made literal.
2. `numeric_encoder.py` records `arithmetic_roundtrip_bits_match`,
   `causal_decoder_frequencies_match`, and `reconstruction_indices_match` but
   does not require them to be true before returning a `PASS_FINITE...`
   receipt. The later independent decoder closes much of this gap, but a
   producer pass receipt must fail closed itself.
3. The decoder has an additional terminal condition not listed in the design
   overflow contract: `_integer_inverse_symbols` rejects an absolute inverse
   symbol above 32,767. Either widen the transient buffer to I32 or freeze and
   audit this as a second no-retry terminal gate before any Qwen pilot.
4. `frame_ledger.repeated_compressed_bytes` currently means total bytes read
   (`passes × frame_bytes`), whereas the decoder's `compressed_frame_reread_bytes`
   means extra bytes and is zero for one pass. Rename or add an explicit
   `(passes-1) × frame_bytes` field before consuming the ledger automatically.
5. Model source inputs and scorer inputs are not independently manifest-bound
   by v4, and their parent path components are not all rejected when
   symlinked. A future owner must bind exact input hashes/file descriptors,
   runtime versions, the output namespace, and fail-closed atomic publication.

## Read-bandwidth and universality boundary

The `73/72` result is a correct **compressed-file cold-page planning** number
for the unimplemented final layout. It is not yet an inference-kernel traffic
measurement. The Python decoder buffers full FP32 reconstructions and full
I16 symbol arrays; their writes and downstream reads are excluded. A fused or
bounded-scratch implementation must demonstrate that these transient HBM
movements do not erase the read benefit.

V4 is model-, layer-, and expert-identity agnostic, but its target-rate cell is
not universal over all positive SwiGLU shapes. Non-divisible shapes have a
valid compatibility packet at a higher bpw, and the implementation also has
explicit dimension/stream caps. It can therefore support Qwen and other
N18-divisible BF16 geometries, but it cannot yet satisfy the project's final
universal `<=2.5`-bpw contract without a bounded tail codec. Down-transposition
and canonical BF16 normalization also remain upstream responsibilities.

## Verification

From this audit directory:

```bash
python3 -B verify_audit.py
```

The verifier checks exact producer and dependency closure, recomputes every
rate/page identity, runs independent overflow/tail/separation probes, verifies
the frozen audit files, and emits a non-authorizing receipt.
