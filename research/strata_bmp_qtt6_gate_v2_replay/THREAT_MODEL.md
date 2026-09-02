# Threat model: STRATA-BMP/OBDD/QTT6 replay v2

This package is adversarially scoped as source-only. It must not convert a
mechanism fixture, estimate, plausible digest string or GPU import into payload
authority.

## Closed v1 audit findings

- **Manifest ordering alias.** Members are sorted by their UTF-8 bytes and the
  verifier enforces exactly that list.
- **Documented CLI drift.** The README command uses the verifier's literal
  `--package` and required `--expected-manifest-sha256` options. A hostile test
  executes that exact shape.
- **Geometry-independent packet slot.** Serialized candidate maxima are
  derived from row count, column count, active features and family caps. Skew
  geometries larger than the former 2,048-byte slot are frozen in tests.
- **int32/intp confusion.** The actual stable `np.argsort` result is required
  to be `numpy.intp`; its own `nbytes` enters the runtime event ledger.
- **Capacity/measurement conflation.** Logical capacities, actual retained
  packet bytes, instrumented host object lifetimes and synchronized CuPy pool
  samples are separate fields. No cross-allocator process peak is claimed.
- **Digest-shaped capability.** Production authorization opens stable regular
  non-link objects, hashes the bytes, and semantically validates control,
  routed-read and independent-audit receipts. Hex syntax alone cannot pass.

## Retained codec attacks

- **Four-level substitution.** Only complete `D[N,64]` semantics and six
  decoded planes are accepted.
- **GF(2) gauge aliases.** Minimum-rank canonical BMP factors and canonical
  QTT cut gauges reject rank inflation, state padding and basis changes.
- **OBDD aliases.** Fixed variable order, reduction, topological references and
  canonical re-encode close unused nodes and alternate graph serializations.
- **Header overflow.** Every uint16 geometry field is checked before packing.
- **Free selector/model/tail.** Every byte through the CRC is physical and the
  complete integer 2.15--2.5 bpw ledger charges outer/reserved bits.
- **Fake CuPy facade.** A fresh `-I -B` process authenticates module origin,
  distribution ownership/version, compiled execution, device identity, nonce
  and a distinct PID.

## Explicit holds

Source self-replay does not establish numerical runtime, GPU runtime, Qwen
gain, matched-control excess, production compatibility, `F<=0.8`, or routed
reads below `2x`. ROBDD/QTT GPU search and all outer-codec objects remain held.
Independent audit must authenticate the frozen source and emit a receipt bound
to its manifest and source root before any production hook can authorize.
