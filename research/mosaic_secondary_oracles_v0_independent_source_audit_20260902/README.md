# Independent source audit: MOSAIC secondary oracles v0

Date: 2026-09-02

Disposition: **mechanisms valid; production launch held**.

This directory audits the frozen source-only package whose manifest SHA-256 is
`4259e8e8dc87b4c25301ca89ade7dbd63c1e0c9e3415fdaa4d7881d7d10ccc06`
and whose source-root SHA-256 is
`60bf8cb7575c165c1e8e648360b9d81f39c092070a9489684904bcf06d0bd820`.
The audit does not accept or open a Qwen checkpoint, coarse packet, completed
result, or matched-control payload.

## Findings

### GF(2) packet mechanics pass

The audit independently brute-forces the minimum GF(2) recurrence for every
binary sequence of lengths 1 through 10 and compares all 2,046 cases with the
frozen Berlekamp-Massey result. It also verifies exact replay. CRC-resealed raw
aliases, deliberately non-minimal LFSR aliases, nonzero reserved fields,
nonzero terminal padding, and a false expert weight count are rejected.

`LRB0/LRC0/LRE0` is therefore a strict, exact serializer for the four-level
labels it is given. The expert decoder derives coverage from literal decoded
blocks, and its physical rate is independently recoverable as
`8 * packet_bytes / decoded_weights`.

The scale field is only an opaque two-byte field. The serializer accepts bit
patterns that represent NaN under IEEE binary16 or bfloat16. This is not a
packet-canonicality defect, but a production adapter must freeze the numeric
format and reject illegal scale values before any reconstruction or MSE claim.

### Promotion gates are not result scorers

`recurrence_codec_gate` trusts its caller's `relative_mse`, all ledger fields,
the literal-four reference rate, and the matched-control saving. A fabricated
dictionary is sufficient to obtain `ELIGIBLE_FOR_PORTABILITY_AND_LITERAL_NESTING`.
Likewise, `residual_source_gate` trusts caller-supplied source and control SSEs.

This is acceptable for a source-only algebra helper, but neither function can
authorize a payload result. A production scorer must parse the literal expert
packet, derive weight count and byte rate, reconstruct weights, and recompute
FP64 source SSE and complete matched controls itself.

### Physical bytes pass; cold-read evidence does not yet exist

For zero shared-model bytes, the abstract layout helper agrees with a literal
`LRE0` packet. However, the helper also accepts `shared_model_bytes` even though
`LRE0 v1` has no corresponding shared-model field. The recurrence gate accepts
an arbitrary caller-created ledger rather than parsing a packet.

The reported cold-read amplification is identically
`cold_storage_bytes / physical_bytes = 1`, while `external_storage_reads=1` and
`external_storage_refetches=0` are constants. This describes the intended
one-packet topology; it does not observe file/page reads or establish runtime
traffic. Production needs a frozen denominator and an independent page/read
trace or a decoder whose exact reads are derived from literal offsets.

### Ramanujan and AR arithmetic is internally coherent but remains oracle-only

The Ramanujan basis is orthonormal to the stated tolerance. FP16 coefficient
and combinatorial-support budgets are per block and therefore must be
multiplied by the number of blocks. The ideal waterfill correctly remains
marked as having no finite backend.

The audit independently constructs the finite lower-triangular inverse matrix
and confirms that `inverse_noise_gain` equals `trace(H H^T)/n`. Every AR row
conserves its 384-bit descriptor-plus-innovation budget. The reported AR value
is still an ideal iid-Gaussian innovation diagnostic; it is not a finite
innovation packet.

NumPy and CuPy use different QR/RNG implementations: the source-free RTX 5090
audit confirms that basis bytes and matched-Gaussian bytes are not bit-identical
across those backends, despite numerically close oracle metrics. A production
run must freeze an actual CuPy build/device arithmetic and run source plus every
control through that identical authenticated backend.

## Required STRATA/POLARIS adapter

The packet accepts exactly four labels represented by two Gray bitplanes.
Current STRATA is not that ABI: each reconstruction index is in `0..63` and is
assembled from six complete level-major polar passes.

A legitimate production branch must choose one of two explicit formats:

1. A new direct four-level codec, with frozen legal reconstruction levels and
   scale semantics, label selection, literal `LRC0/LRE0` emission, independent
   decode, inverse transform, and original-source MSE scoring.
2. A current-STRATA recoder generalized to six output bitplanes, or a model of
   selected SC events that preserves all six polar levels, reconstruction
   contexts, arithmetic semantics, and canonical re-encode.

It is invalid to reinterpret current 64-way STRATA indices as the existing
four-way packet. A Qwen pilot would remain pilot evidence; universal
SwiGLU-MoE authority requires a sealed transfer family.

## Reproduction

```bash
python -I -B audit_source.py \
  --upstream-source ../mosaic_secondary_oracles_v0

python -I -B run_cupy_backend_audit.py \
  --upstream-source ../mosaic_secondary_oracles_v0
```

The first command is CPU/source-only. The second uses synthetic arrays on a
real CuPy CUDA device and still exposes no payload path.
