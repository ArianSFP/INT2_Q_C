# Independent hostile audit: COCHAIN-Q plaquette/cube v0

## Verdict

**The mathematical source mechanism passes, but Qwen execution must remain
blocked pending three closure/API repairs and a separately reviewed six-plane
capability.**

Target pins:

- manifest SHA-256: `ef12407301265d8e04da9f1ed5afaadff69f0d864c31ef1be4868279506a68b3`
- source root SHA-256: `3e515bc146fde2dde734fb94eb11f5dc32397d227fd4dd3930037b2ce498190a`

## What passed

- The fixed public-syndrome optimizer is globally exact for disjoint cells.
  The audit compared the vectorized solver with literal global enumeration on
  108 random dimension/cell/syndrome cases.
- Boundary plus syndrome is bijective over every one of the 16 plaquette and
  256 cube patterns.
- The fixed-even-fiber packet is one-to-one and independently decodes for all
  8 plaquette and 128 cube codewords. Packet length and zero padding are
  checked by the target decoder.
- The physical equivalent-gain implementation equals
  `(R0-R1)+0.5*log2(D0/D1)` in independent recomputation.
- Low-degree parity fixtures have zero pairwise mutual information and exact
  ideal savings of 0.25/0.125 bits per affected bitplane site. The balanced
  IID cube has syndrome counts `[128,128]` and hard-kills.
- The explicit payload gate raises. No Qwen/model data, GPU, or network was
  accessed.

## Blocking findings

1. **The source manifest is not a closed package.** A copied package with an
   additional unlisted executable still passes `verify_source.py`. The
   verifier checks listed hashes but neither rejects extra members nor rejects
   symlinks/non-regular files.

2. **The external manifest pin is optional.** Running the verifier without
   `--manifest-sha256` succeeds, and the README's example omits the pin. A
   self-consistent replacement manifest therefore authenticates itself.

3. **The encoder silently coerces invalid labels.** `encode_public_fiber`
   casts FP64 labels to `uint8` before validation; a label `0.9` is accepted as
   zero and serialized. This does not break canonical decoding of valid
   packets, but it breaks a fail-closed canonical encoder contract.

4. **The reported rate unit is local.** `run_oracle` reports bits per affected
   bitplane site and compares that number directly with project-wide bpw
   thresholds. If a future screen applies to one role, plane, or subset, the
   rate delta must be divided by the total audited expert weight count. The
   API currently has no global coverage denominator.

5. **The distortion metric is not yet the deployed codec metric.** The v0
   result is internally correct for caller-supplied binary distortion fields,
   but Qwen promotion requires one legal six-plane STRATA reconstruction and
   pooled original-BF16 source SSE. Per-plane equivalent gains cannot be
   summed, and a favorable binary-plane oracle is not a finite codec result.

## Read-bandwidth finding

The `1x` statement is valid only as a logical topology statement: the packet
is expert-local and has no common or neighbour stream. It is not yet a
physical cold-read result. One tiny plaquette packet occupies one byte but a
standalone 4096-byte page, while the production-size synthetic ledger happens
to be page-aligned. A future packet must provide an integrated expert-page
layout and measure actual rounded reads. No evidence suggests intrinsic
cross-expert amplification in the fixed public-fiber design.

## Required repair before a Qwen capability

- require an external manifest digest and enforce exact member closure,
  regular files, sizes, canonical JSON, and no symlinks;
- validate label dtype/range before conversion;
- add total expert weight count and a literal whole-codec source SSE/rate
  ledger, then bind the actual legal six-plane reconstruction;
- independently audit the repaired source closure;
- only then build a one-use local RTX 3060/CuPy capability with fixed
  layer/expert splits and complete matched-Gaussian controls.
