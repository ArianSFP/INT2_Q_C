# UNIPOLAR-N18-307 v4 — executable coarse source candidate

Status: **source-only implementation candidate; no Qwen payload has been
opened and no numerical result is claimed**.

V4 takes the shortest technically honest route out of the v3 audit block.  It
does not call `1,228 bytes / 4,096 weights` a codec.  It preserves the one
finite language that was already numerically connected to the POLARIS-SC
primitive: one self-describing `N=2^18` arithmetic reservoir per source tile.

## What is implemented

Each record is exactly:

```text
128-byte TACN18C4 header
78,464-byte causal arithmetic reservoir, including charged zero fill
---------------------------------------------------------------------
78,592 bytes = 628,736 physical bits = 307/128 bpw for 262,144 values
```

The header binds:

- format/version and algorithm identifier;
- canonical role, intermediate/hidden shape and tile ordinal;
- exact valid source values and zero-padding count;
- shape/role/tile-derived SC and RHT seeds;
- one transmitted FP32 decoder scale;
- exact logical arithmetic EOF;
- payload SHA-256, payload CRC32 and header CRC32; and
- zero-tile, padded-tail, source-order and canonical-re-encode flags.

The nominal test-channel rate is `153/64 = 2.390625 bpw`, leaving exactly
1,024 bits of finite arithmetic reserve.  Overflow is a terminal architecture
failure.  There is no seed, profile, transform or retry search after overflow.

An expert frame is the literal concatenation of records in this order:

```text
Gate tiles 0..T-1
Up tiles 0..T-1
transposed-Down tiles 0..T-1
```

Every record repeats and authenticates the shape and coordinate, so the frame
is self-describing without a hidden expert/model lookup.  Missing, extra,
reordered or cross-shape records fail.

`numeric_encoder.py` authenticates the exact existing CuPy producer and
procedural Q31-BEC construction source bytes, pads a tail with BF16 `+0`, and
emits literal records.  `independent_decoder.py` imports no encoder.  It
regenerates all causal probabilities, re-encodes the arithmetic symbols,
requires byte-identical records, performs the inverse signed RHT, and can emit
the original-coordinate FP64 residual and pooled relative MSE against exact
canonical BF16 inputs.

## Tail and eligibility semantics

For a role whose value count is not divisible by `2^18`, its last record is a
full physical reservoir.  The source prefix is followed by implicit BF16
`+0` values before the RHT; the decoder reconstructs the complete internal
tile but exposes only the shape-bound source prefix.

This is an executable compatibility language, not a rate sleight of hand.
Such a frame has more than `307/128` bpw and is marked **not target eligible**.
V4 may support and decode it, but may not use it as evidence for the
`2.15–2.5 bpw, F<=0.8` cell.  The exact-rate eligibility condition is:

```text
intermediate * hidden is divisible by 262,144.
```

This includes the pinned Qwen `768 x 2048` geometry and many other shapes; it
does not pretend that a three-weight `1 x 1` expert can carry a standalone
self-describing packet inside 2.5 bpw.  A future bounded tail codec is a
separate format revision.

An exactly zero source tile has a canonical zero-tile record: no logical
payload, stored scale `1.0`, and a zero reconstruction.  The physical
reservoir remains fully charged.

## Exact Qwen coarse and frozen-final ledgers

For one `768 x 2048` expert:

```text
6 records / role
18 records / expert
1,414,656 coarse bytes / expert
8,487,936 coarse bytes / six-expert panel
67,903,488 / 28,311,552 = 307/128 physical bpw
```

The coarse codec has no trained/shared model packet.  Public decoder code is
not source-fitted state.  This does not make downstream selector, QC, fine or
schema packets free.

The frozen TACTIC planning topology remains:

```text
coarse reservoirs       8,487,936 bytes
384-bit fine fields       331,776 bytes
six 512-byte headers        3,072 bytes
one global packet          24,576 bytes
----------------------------------------
final                    8,847,360 bytes = 2.5 bpw
```

V4 implements only the coarse prefix.  It does not claim that the frozen fine
packet is implemented or that TACTIC improves MSE.

## One-pass routed-read schedule

The file decoder reads every compressed reservoir once, in canonical order,
and buffers decoded coarse symbols/reconstruction for all later refinement or
graph work.  It records exactly one compressed pass and zero reread bytes.

Under the frozen final layout, the selected private expert frame is 359 pages
and the common packet is six pages.  One pass reads `365/360 = 73/72 =
1.013888...x`.  A second private-frame fetch would read
`6 + 2*359 = 724` pages, or `724/360 = 2.011111...x`, and is explicitly
invalid.  Unique-page union and repeated compressed traffic are reported
separately.

## Source-only verification

These commands import no NumPy, CuPy or model data:

```bash
/usr/bin/python3.12 -I -B verify_source.py
/usr/bin/python3.12 -I -B test_source_only.py -v
```

The suite covers packet mutation, hard logical EOF, terminal fill, shape/seed
binding, zero tiles, literal padded tails, frame ordering, exact Qwen rate,
target eligibility, page/repeated-read accounting, pinned dependency hashes,
and the independent-decoder/encoder separation.

After an independent source review, the bounded source-free numerical smoke is:

```bash
/workspace/int2-cupy-venv/bin/python -I -B synthetic_cupy_smoke.py \
  --repo-root /absolute/INT2_Q_C \
  --authorization SYNTHETIC_ONLY_TACN18_V4_CUPY_SMOKE
```

## Remaining execution gates

1. independent source audit and source-root freeze;
2. numerical runtime freeze for Python, NumPy, CuPy, SciPy and CUDA;
3. source-free synthetic CuPy encode -> independent decode -> canonical
   re-encode, including forced overflow and zero/tail fixtures;
4. three authenticated Qwen pilot records (Gate, Up, DownT block zero);
5. stop immediately if any record overflows; otherwise independently score
   the actual original-coordinate residual;
6. only then build all 108 Qwen records and hand the real residual to the
   separately repaired DH384/CAGE experiment.

No file in this directory authorizes payload access, CUDA execution, a source
seal, or a Qwen/Tactic performance claim.

The checked-in source-free smoke receipt records one RTX 5090 execution:
`626,926` logical bits, `786` reserve bits, exact canonical re-encode and
`0.03693951352239193` original-coordinate relative MSE on the deterministic
Gaussian BF16 fixture.  It is mechanics evidence only.
