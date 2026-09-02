# Threat model and unexecuted freeze record

Date: 2026-09-02

This is a source-only research mechanism. The configured RunPod endpoint
refused TCP/SSH connections while this freeze was prepared. The hostile test
suite, N=4096 fixture, source verifier and explicit CuPy smoke are therefore
**unexecuted in this freeze**. Their presence is not a pass receipt. No model,
Qwen, current-STRATA, coarse-code or matched-control payload was opened,
listed, statted or hashed.

## Protected properties

- A packet decodes exactly six completed binary planes and assembles a 0..63
  index with level 0 as the least-significant bit.
- Packet bytes bind family, variable order, role, shape, tile, model payload,
  exceptions and a CRC32.
- Accepted models obey hard dimension, rank, node, exception, workspace and
  search-operation caps.
- Canonical decode/re-encode, zero padding and canonical ROBDD reduction deny
  multiple byte encodings of the same accepted object.
- A future experiment must bind raw-source scoring and every complete matched
  control to an externally pinned source and selection record.

## Adversary capabilities and defenses

1. **Corrupt or splice packet bytes.** CRC32, exact extents, geometry checks,
   sorted exceptions, topological OBDD checks and canonical re-encoding reject
   accidental corruption and many malformed packets. CRC32 is not a MAC and
   provides no protection against a deliberate attacker who recomputes it.
2. **Rewrite and self-reseal the source package.** `SOURCE_MANIFEST.json` is a
   deterministic inventory, not a signature. Only an external manifest
   SHA-256 pin can distinguish a publicly rewritten package. Independent
   audit must pin that hash before any payload authorization.
3. **Exploit unused padding or aliases.** All packed-bit tails must be zero;
   OBDDs must be canonical reductions; exceptions must be sorted, unique and
   nonredundant; decode must byte-identically re-encode.
4. **Smuggle four-level or SC-local semantics.** The public distortion ABI
   requires exactly 64 columns. The decoder exposes six completed planes and
   a literal 0..63 assembly. Four columns and internal polar/SC decisions are
   rejected.
5. **Exploit coordinate ambiguity.** Shape validation requires rows `3*2^k`,
   power-of-two hidden width, a role trit and aligned power-of-two tiles. The
   active coordinate map must enumerate every tile position exactly once.
6. **Denial through model size or search explosion.** Ranks, OBDD nodes,
   exceptions, candidates, workspaces and scored operations are hard-capped.
   A cap hit fails closed and is not evidence against a larger family.
7. **Forge a favourable encoder metric.** This package emits no payload score
   receipt. A production adapter must independently decode its literal packet
   and recompute FP64 SSE and source energy from authenticated raw BF16 bytes;
   encoder-reported MSE or rate must be non-authoritative.
8. **Overfit family/order/lambda to test weights.** A future protocol must
   freeze its bank and train/validation selection before test access, rerun the
   entire selection on at least eight matched Gaussian controls, and derive
   uncertainty by whole layer and expert.
9. **GPU/backend drift.** The optional CuPy smoke compares generated GPU and
   CPU arrays and synchronizes the active stream, but it does not yet pin a
   CuPy build, CUDA device or kernel receipt. Production must freeze and record
   all three and repeat controls on the same backend.
10. **Claim a complete runtime packet from a tile descriptor.** This v0 packet
    omits STRATA scale, KLT/RHT/profile, component/expert headers, page padding
    and runtime layout. Its byte rate cannot establish a 2.15--2.5 bpw codec or
    a routed read-amplification result.

## Exact mechanism packet bounds for N=4096

These bounds include the 30-byte header, all physically byte-sized selectors,
the four-byte CRC and byte padding. They omit the production fields listed
above and therefore are not complete-codec bounds.

| Family/configuration | Bytes | Bits/weight |
|---|---:|---:|
| GF(2) matrix factor, six rank-0 planes, no exceptions | 40 | 0.078125 |
| ROBDD, six terminal roots, no nodes/exceptions | 58 | 0.11328125 |
| BMP/QTT, six rank-1 depth-12 cores, no exceptions | 58 | 0.11328125 |
| GF(2) factor at rank 4 on every 16x256 plane plus 64 exceptions | 1,048 | 2.046875 |
| ROBDD at the aggregate 240-node cap plus 64 exceptions | 1,450 | 2.83203125 |
| BMP/QTT at rank 2 on every depth-12 plane plus 64 exceptions | 298 | 0.58203125 |

The low minima encode extremely restricted functions and say nothing about
distortion. The OBDD cap can exceed 2.5 bpw even before production fields, so
the eventual physical-rate gate must reject such a candidate. Conversely,
sub-2.15 mechanism packets require legitimately charged production or
refinement bytes; filler cannot be portrayed as a compression gain.

## Residual risks

The implementation has not been executed here, so syntax, runtime,
performance and GPU parity remain unverified. SHA-256 collision resistance is
assumed. Side channels, malicious Python/NumPy/CuPy runtimes, host compromise,
and cryptographic authenticity are outside this gate. No Qwen conclusion is
permitted until independent replay and a separately authorized production
adapter pass.

