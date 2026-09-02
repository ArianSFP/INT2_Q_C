# Threat model: hardened source-only gate v1

Date: 2026-09-02

The RunPod endpoint refused SSH while this package was frozen. The 22 hostile
tests, N=4096 fixture, source verifier and real CuPy worker are unexecuted.
Source presence is not a PASS receipt. No model or codec payload was accessed.

## Protected properties

1. Six completed planes define one literal 0..63 index per weight.
2. Family, order, arbitrary uint16 shape, role, tile, model, exceptions and CRC
   are physical packet bytes.
3. One decoded BMP or QTT function has one accepted semantic representation;
   gauge aliases cannot carry a hidden channel.
4. A search cannot select a packet outside an explicit complete 2.15--2.5 bpw
   ledger after outer, prior and reserved bits.
5. Payload authorization cannot occur while any scale, transform, scorer,
   controls, framing, audit or read-ledger hook is absent.

## Attacks and defenses

- **Rank inflation / zero-state aliases.** BMP derives exact rank and a fixed
  earliest-column/earliest-row factor. QTT derives exact cut ranks and fixed
  GF(2) factors; zero uses rank code zero and no payload.
- **Factor or bond gauge.** BMP column swaps and GL(r,2) changes fail the exact
  canonical comparison. QTT cores are re-derived from the full truth table and
  compared bit-for-bit.
- **Unused rank-mask or packed-tail bits.** Rank codes must use only d-1 bits;
  all byte tails are zero.
- **Header overflow.** All six uint16 geometry values are range-checked before
  `struct.pack`. Values at 65,536 fail with `CodecError`.
- **Coordinate ambiguity.** The public domain is role(base 3) times arbitrary
  row and column ranges. The active tile map must be a bijection over exactly
  its aligned Boolean cube.
- **Packet corruption and splice.** Exact extents, CRC32, sorted nonredundant
  exceptions, topological canonical ROBDDs and canonical re-encode fail
  closed. CRC32 is not cryptographic authentication.
- **Rate omission.** `CompleteRateCap` has no implicit default. It charges
  outer, prior and future-reserved bits using integer bounds. A source fixture
  below 2.15 is explicitly incomplete, not a favourable compression result.
- **Workspace euphemism.** CPU output names and counts every owned logical
  buffer and disclaims a Python allocator peak. The fresh CuPy worker uses a
  dedicated memory pool and records synchronized exact used/reserved samples.
- **Fake CuPy facade.** A module object, installed distribution ownership,
  version agreement, file hash and compiled-kernel probe are mandatory. The
  active rather than assumed device is recorded.
- **Same-process contamination.** A nonce-authenticated distinct worker runs
  with `-I -B`; stdout must be exactly one JSON record.
- **Partial accelerator overclaim.** The actual GPU search is identified as
  rank-0/rank-1 BMP only. ROBDD and QTT GPU paths remain holds.
- **Metric or control leakage.** There is no payload adapter. Future launch
  must bind original-BF16 scoring and at least eight controls that repeat the
  complete selection process.
- **Read-amplification omission.** No runtime claim exists until expert-local
  framing and the routed cold-page ledger are hashed and authorized.

## Residual risks

Python, NumPy, CuPy, CUDA, filesystem and host compromise are outside this
source gate. SHA-256 collision resistance is assumed. The manifest is an
inventory, not a signature; an external pin is required. Exact dedicated-pool
reserved bytes are not a whole-process RSS measurement. The mathematical
families remain bounded heuristics and an unfavourable gate result would not
prove that a larger family has no useful structure.

## Claim boundary

This package cannot establish Qwen gain, F<=0.8, a physical 2.15--2.5 bpw
container or routed reads below 2x. Independent source audit and an explicitly
separate production-launch review are both mandatory.
