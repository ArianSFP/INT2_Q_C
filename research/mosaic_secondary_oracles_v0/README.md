# MOSAIC secondary oracles v0

Date: 2026-09-02

Status: source-only, source-frozen mechanics. This package has no production
payload adapter and no authority to open Qwen, `COARSE.bin`, a completed model
result, or matched-control data.

## Why this ladder exists

Two nearby branches are already closed at their declared scope:

- CYCLO-FRI4 tested periods 1, 2 and 4 and hard-killed at audited best
  `F=0.9379899307967997` before controls.
- The coarse-programmed graph/Krylov oracle reached a superficially favorable
  ideal waterfill, but identical controls reproduced it; the source-specific
  excess was only `0.00015723251757482348 bpw`.

This package therefore does not rerun dyadic recurrence, another graph basis,
or another arithmetic coder. It freezes three inexpensive, distinct tests:

1. exact finite-field recurrence on legal four-level label bitplanes;
2. exact-period non-dyadic Ramanujan residual subspaces; and
3. low-order Hankel/annihilating-filter structure with quantization-noise
   pullback through the inverse recurrence.

BM3D-style grouping is deferred. The current coarse-derived graph result gives
no evidence that coarse similarity predicts residual similarity strongly
enough to justify the much larger matching/aggregation search.

## A. Exact GF(2) recurrence packet

Each legal four-level block is Gray-mapped to two bitplanes. Exact
Berlekamp-Massey returns the shortest error-free binary recurrence for each
plane. The canonical `LRB0` packet chooses independently between:

- raw: `n` bits; or
- LFSR: `L` connection bits plus `L` initial-state bits.

Ties choose raw. The physical block packet includes an eight-byte header, two
four-byte plane records, byte padding and CRC32. Literal `LRC0` component and
`LRE0` expert serializers charge a 64-byte role header, four-byte block
offsets including the terminal offset, one BF16 scale per block, role
alignment, the expert header, the 2.15-bpw lower-bound padding and final 4-KiB
padding. Both layers have CRC32, zero-padding checks, role-order checks and
canonical decode/re-encode. Their literal byte length must equal the
independent physical ledger.

This is an exact finite codec for the labels it receives. It does not solve
label-flexible recurrence or robust recurrence with sparse exceptions. A miss
therefore closes only exact LFSR recoding.

## B. Non-dyadic Ramanujan oracle

The period bank excludes powers of two and starts at 3, 5, 6 and 7. For each
period it emits the real cosine/sine atoms associated with primitive
frequencies, then performs one source-independent QR. The same public basis is
used for source and controls.

Four measurements remain separate:

- fixed-prefix continuous projection with free amplitudes;
- fixed-prefix FP16 amplitudes fitting literally inside 384 bits;
- source-selected support with its combinatorial rank and FP16 amplitudes
  charged inside 384 bits; and
- ideal Gaussian waterfill over the public basis, explicitly marked as lacking
  a finite backend.

Every score is measured after reconstruction in source coordinates. Per-block
support and period selection are never free.

## C. Capped AR/Hankel oracle

Orders are `{1,2,4,8,12}`. Each block pays four selector bits and 16 bits for
every fitted coefficient, leaving the balance of the 384-bit field for an
ideal innovation code. Coefficients are rounded to IEEE binary16 before the
innovation sequence is formed.

Innovation variance alone is not a valid MSE result: a nearly unstable inverse
filter can amplify quantization noise catastrophically. The gate therefore
computes the exact finite-length impulse response and charges
`trace(H H^T)/n`. The reported distortion is still an optimistic iid-Gaussian
innovation diagnostic; no finite innovation codec is emitted.

## Source-first controls and stop rules

For residual branches, controls remain unopened unless the source alone reaches
`D<=0.025`. A survivor then must beat both a frozen odd-affine
phase-destruction control and all eight complete moment-matched Gaussian
pipelines by at least `0.03 bpw`. Search, period/rank/order selection and all
finite rounding repeat inside every control. Control subtraction cannot turn
an absolute miss into a pass.

The exact coarse ledger is:

```text
307/128 bpw coarse + 12/128 bpw fine + 1/128 bpw metadata = 2.5 bpw
```

Every descriptor displaces fine bits. The required coarse-residual capture is
`32.3870222053737%`, not the 19.10% finite-baseline number. The compressed
expert may be read once and buffered; a second fetch is forbidden. Storage,
host scratch and HBM traffic are separate ledgers.

## Source-only verification

```bash
python -I -B research/mosaic_secondary_oracles_v0/test_source_only.py
python -I -B research/mosaic_secondary_oracles_v0/run_source_free_fixture.py
python -I -B research/mosaic_secondary_oracles_v0/verify_source.py \
  --package research/mosaic_secondary_oracles_v0 \
  --manifest-sha256 <expected-manifest-sha256>
```

The GPU smoke accepts no payload path:

```bash
python -I -B research/mosaic_secondary_oracles_v0/run_source_free_cupy_smoke.py \
  --authorization RUN_SOURCE_FREE_MOSAIC_SECONDARY_CUPY_SMOKE_V0 \
  --manifest-sha256 <expected-manifest-sha256>
```

No production run is authorized until a separate source audit freezes an
authenticated universal adapter and a one-run launch review. A Qwen pilot
would remain pilot evidence; a universal claim requires sealed transfer to a
disjoint SwiGLU-MoE family.
