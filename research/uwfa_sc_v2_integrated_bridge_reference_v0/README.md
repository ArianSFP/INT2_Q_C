# UWFA-SC v2 integrated bridge — independent source-only reference

This package implements the reusable byte-level bridge required by
`docs/UWFA_SC_V2_INTEGRATED_CONTAINER_SPEC.md`.  It was developed without
opening Qwen weights, the held current container, selected-bit/frequency dumps,
Gaussian controls, or the main v2 producer source.

It is **not payload authority** and is not an independent audit of a future
producer.  It proves only that the finite ABI and verification mechanics can be
implemented and attacked with source-free fixtures.

## Implemented contract

- fixed 4,096-byte header with CRC, topology, exact byte ranges, evidence
  hashes, semantic reconstruction hash, and a root over all nonpadding bytes;
- literal inherited-metadata region, page-rounded serialized UWFA model,
  page-rounded fifteen-record directory, 64-byte-aligned frames, and the unique
  minimal floor/page padding;
- explicit dense model rows keyed in canonical
  `(level, prior_bin, position_in_reset, state)` order, with no model/expert,
  layer, or stream identity probability key;
- a deterministic 32-bit binary arithmetic ABI and required
  `decode(original_freq1)` adapter; the synthetic mini-STRATA loop regenerates
  every `original_freq1` from a literal seed and earlier decoded decisions;
- independent decode followed by literal arithmetic byte and logical-bit
  re-encode equality;
- exact `R = 8*container_bytes/source_weights` as a rational and
  `F = relative_MSE*2**(2R)` from literal physical bytes;
- owner-aware storage shares and exact union-of-4-KiB cold pages;
- completion-last output primitive using exclusive file creation and disabled
  post-completion API writes;
- hostile tests for CRC/root-bound fields, reserved bytes, padding, extra
  pages, frame hashes/tails, directory overlap, model row order, semantic
  decision hashes, path traversal, and late writes.

## Deliberate reference choices / ABI boundary

The normative document leaves several bit-level choices open.  This reference
freezes them locally so tests are reproducible:

1. arithmetic total `65536` and a canonical 32-bit E1/E2/E3 binary coder;
2. topology id 1 with the frozen transition in `UWFAModel.transition`;
3. uniform prior bins `floor(original_freq1*K/65536)`;
4. 256-byte directory records and 64-byte frame headers;
5. model/metadata/directory start on page boundaries, with frame records
   contiguous in ordinal order;
6. root computation zeros the root and header-CRC fields and concatenates the
   header plus only semantic (nonpadding) region/frame bytes;
7. energy convention id 1 means original-BF16 source and reconstruction,
   FP64 SSE/energy accumulation, and relative MSE `SSE/energy`.

A producer choosing another arithmetic ABI or transition must use another
minor version/topology id.  The real inherited STRATA metadata parser and
inverse RHT/XKLT reconstruction are intentionally callbacks outside this
source-free package; the fixture's decision hash stands in for that semantic
reconstruction callback.

The `CompletionLastCapsule` is also not the normative external bootstrap.  It
enforces absent-output/exclusive-leaf/completion-last behavior for the producer
API, but it does not authenticate symlink ancestors, retain input descriptors,
or execute an immutable source snapshot.  Those checks must remain in a
separately pinned and independently audited launcher.

## Run

```bash
python3 -m unittest -v test_uwfa_bridge.py
```

The tests use only Python's standard library.  CuPy is not imported: GPU
preflight is intentionally downstream of the byte-integrity gates.
