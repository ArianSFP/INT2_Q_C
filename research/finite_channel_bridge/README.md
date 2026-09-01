# Finite channel bridge for SILWARP

## Verdict

The most credible finite successor is **not** scalar rotated-dithered
quantization.  It is an expert-local finite POLARIS/polar-lattice base codec,
trained and evaluated end-to-end with the SILWARP decoder on the exact decoded
base bytes.  A small SC-list analysis-by-synthesis encoder can then choose the
valid base codeword that minimizes post-warp distortion without transmitting a
list index.

Scalar rotated dither remains valuable as an AWGN-transfer control.  This
bundle implements it through a real canonical arithmetic bitstream and tests
it at the production `N=2^19` block size on the RTX 5090.  It is decisively
too rate-expensive to preserve a merely threshold-passing SILWARP result:

```text
actual scalar-RDQ payload                 2.4075832367 bpw
system rate incl. FP16 SILWARP + table    2.4141676956 bpw
cold expert read                          1.3344648962 x
identity MSE                              0.05070058749
same-rate target MSE                      0.02815893369
required reduction                        44.46034043 %
required s                                0.4242048804 bpw
ideal SILWARP promotion threshold s       0.1673974075 bpw
```

Therefore an ideal result near its promotion boundary cannot simply be turned
into scalar-dither bytes.  Calling it a finite success would be wrong.  Direct
finite polar reconstruction has no scalar shaping penalty and is the route
with enough margin to be credible.

No Qwen tensor, pinned tensor, confirmation value, or auxiliary payload is
opened by any executable in this directory.  The only data experiment is
i.i.d. synthetic Gaussian input.  Both frozen SILWARP directories remain
unchanged.

## Primary architecture: POLARWARP-F

### Decoder object

Keep the frozen SILWARP map conceptually unchanged:

```text
X_hat = Y_finite + f_theta(Y_finite, public coordinates, charged moments).
```

Unlike the auxiliary gate, `Y_finite` is reconstructed solely from a physical
POLARIS stream.  The deployed decoder never receives a source feature.  The
same global model is shared by all experts and counted in both rate and every
cold-read calculation.

### Expert-local base layout

Each canonical role has exactly `3 * 2^19` weights, so an expert is exactly
nine private `N=2^19` polar-lattice blocks.  A conservative 128-byte expert
header contains nine `(u32 arithmetic_bits, u16 escape_bytes)` records, three
FP16 means, three FP16 RMS values, flags, a checksum and reserved bytes.  All
seeds and construction tables are global/public.

This nine-block layout has exactly `1x` local coefficient ownership.  Counting
the entire 475,654-byte FP16 SILWARP object on every cold expert read gives:

| Total cap | Bytes/N19 stream | Actual bpw | Cold read |
|---:|---:|---:|---:|
| 2.15 | 140,475 | 2.1499956714 | 1.3721558747x |
| 2.30 | 150,305 | 2.2999895679 | 1.3478857169x |
| 2.50 | 163,412 | 2.4999865161 | 1.3200551341x |

These are exact integer ledgers with the FP16 model and without an RDQ CDF
table.  At 2.5 bpw, using the already validated N20/N21 ownership first is an
even lower-risk implementation path.  If its current worst page amplification
`1.1694444444x` is conservatively retained after the small rate reallocation,
counting the complete FP16 decoder yields a planning upper estimate of
`1.489071x`, still comfortably below 2x.  The emitted successor must replace
that estimate with literal byte and page unions.

### Why 2.5 bpw should be tried first

The published expert-affine base has measured MSE `0.030902167403153148` at
2.5 bpw.  Removing roughly 0.00653 payload bpw to make room for the FP16
decoder and 128-byte finite header gives the transparent Gaussian-slope
planning estimate `0.03118321228`; it is not a measurement.  The exact-ledger
target is `0.02500046732`, so the finite decoder needs approximately a 19.83%
correction, or `s=0.15941`.  This is slightly less
than the ideal gate's preregistered `s=0.1673974` threshold.  The cushion is
only about 0.008 bpw, so actual finite training—not an inference from the ideal
result—is mandatory.

At 2.15 bpw the corresponding slope-only requirement is about a 19.28%
correction (`s=0.15449`).  It is similar, but 2.5 has an already published
finite local artifact and more robust absolute distortion, so it is the first
operational cell.

### Encoder: list analysis by synthesis

The first finite gate uses the existing deterministic SC codeword (`L=1`).
If it is close but does not pass, expose the two and four best legal SC paths
at predeclared ambiguous decisions.  For each full candidate bitstream:

1. decode the exact finite `Y`;
2. reload the serialized FP16 SILWARP model;
3. compute `g_theta(Y)`; and
4. select the candidate minimizing source-domain SSE.

The selected candidate is itself the transmitted polar stream.  No candidate
number or source-derived feature is sent.  `L={1,2,4}` is fixed before scores;
stop if `L=2` adds less than `0.003` matched `s` or if `L=4` adds less than
`0.0015` over `L=2`.

### Optional decoder PTQ

A deterministic symmetric per-output-channel INT8 successor has an exact
planning size of 243,462 bytes:

```text
six int8 weight matrices                    234,752 B
1,280 FP16 output-channel scales              2,560 B
1,027 FP16 biases and role gains               2,054 B
closed header                                 4,096 B
total                                        243,462 B
```

At 2.5 bpw with nine private N19 streams this would reduce the cold read to
`1.163819x` before any optional RDQ table.  It is not yet an accuracy result.
Test it only after the FP16 finite codec passes; retain FP16 unless the exact
INT8-reloaded result remains under `F=0.8` with the frozen uncertainty rule.

## Secondary architecture: dithered polar/RHT

If the ideal result is unusually strong, add source-independent subtractive
dither to the finite polar lattice before inverse expert-local RHT.  Polar
lattices have constructive `O(N log^2 N)` lossy coding and are known to
approach the Gaussian rate-distortion bound; a later proof establishes
quantization-good normalized second moment.  Published simulations report a
gap below 0.2 dB at `N=2^18`, making N19 the right scale for an empirical gate:

- [Polar Lattices for Lossy Compression](https://arxiv.org/abs/1501.05683)
- [On the Quantization Goodness of Polar Lattices](https://arxiv.org/abs/2405.04051)

The operational 0.2-dB planning bound is a `0.0332193` bpw penalty.  At the
2.5-bpw FP16 ledger this raises required correction to roughly 24.3%,
`s=0.2007`.  Promote this branch only if the ideal result has enough surplus
over 0.20 bpw and the literal polar+dither bytes fit.  The current POLARIS
implementation is not assumed to inherit a theorem automatically.

Rotated dithered quantization is supported by a direct channel-simulation
result: its divergence from a Gaussian channel decreases as the rotation
dimension grows, while scalar, E8 and Leech shaping costs are progressively
smaller.  See [Gaussian Channel Simulation with Rotated Dithered
Quantization](https://arxiv.org/abs/2407.12970).  Our synthetic N19 result
confirms tile projections are already within the Monte Carlo separation of
two independent Gaussian samples, but rate—not channel shape—is the scalar
variant's fatal bottleneck.

## Measured source-free scalar gate

The experiment applies a signed FP32 Walsh-Hadamard rotation, subtractive
counter-derived dither, and scalar quantization with

```text
a       = 0.9492341876029968
sigma   = 0.21951906383037567
Delta   = sqrt(12) * sigma = 0.7604363560676575
```

It entropy-codes a 65-symbol alphabet using a stored 32-phase, 16-bit binary
probability tree and the repository's canonical 32-bit arithmetic-coder
semantics.  The N19 payload was actually encoded and decoded, not estimated:

| Metric | RTX 5090 / CuPy result |
|---|---:|
| Logical bits | 1,262,267 |
| Logical / byte-padded bpw | 2.4075832367 / 2.4075927734 |
| Payload bytes | 157,784 |
| Arithmetic excess over quantized model | 1.038 bits |
| Escape symbols | 0 |
| Exact round trip | yes |
| Payload SHA-256 | `0448ce9ec00322d98a58ee304dea0602b0bee6517bd5e1d43ba0566b74e5a56d` |
| Empirical channel MSE | 0.05070058749 |
| RHT Parseval relative error | `3.42e-8` |
| RHT/quantizer time | 0.211 s |
| Arithmetic encode + decode | 6.543 s |

For 2,048 independent 256-value views, normalized noise energy was
`255.9634` with variance `515.5244`, versus Gaussian values 256 and 512.  Mean
sliced Wasserstein distance to Gaussian was `0.05888`; the independent
Gaussian-vs-Gaussian Monte Carlo floor was `0.05448`.  Maximum sliced KS was
`0.04199`, below the independent Gaussian comparison's `0.04639`.  These
numbers support channel-shape transfer but do not rescue the shaping-rate
penalty.

## Frozen finite promotion protocol

After an ideal survivor, the finite experiment should be sealed before any new
payload score:

1. Select the 2.5-bpw FP16 direct-polar cell and exact expert ownership.
2. Freeze N19 or reused N20/N21 profiles, seeds, stream capacities, model
   serialization, `L=1/2/4` rules, auxiliary split and stopping thresholds.
3. Encode every fit/calibration base stream once and retain literal bytes.
4. Train SILWARP on exact decoded finite `Y`, weighted by raw source energy.
   Gaussian controls pass through the identical finite codec and optimizer.
5. At updates 256 and 512, kill if both seeds have
   `s_match+2SE < 0.12` and gain from 256 to 512 is below 0.012.  If the
   measured finite identity makes the exact required `s` larger, replace 0.12
   by `required_s-0.04` before launch.
6. Promote only an FP16-reloaded model with actual total `F<=0.8`, matched
   lower confidence bound at or above its exact required correction, positive
   layer aggregates, rate at most 2.5 and worst literal cold/page read below
   2x.
7. Open confirmation only after that finite calibration promotion.  The pinned
   panel remains closed until the complete finite protocol authorizes it.
8. If FP16 passes, run the preregistered INT8 decoder PTQ gate.  It is an
   optional bandwidth optimization, never a replacement for a failed FP16
   result.

The exact required statistic is recomputed from the emitted base identity:

```text
s_required = -0.5 * log2((0.8 * 2^(-2 Rphysical)) / MSE_identity_finite)
```

This avoids assuming the finite base has the ideal AWGN distortion.

## Reproduction

Local source-free checks:

```bash
python -m unittest -v test_source_free.py
python bridge_ledger.py --output ledger.json
python verify_source_free.py --output verification_receipt.json
```

Run the production-shape synthetic serializer with CuPy:

```bash
PYTHONPATH=/usr/local/lib/python3.12/dist-packages \
  /workspace/int2-cupy-venv/bin/python synthetic_rotated_dither_gate.py \
  --backend cupy --log2-n 19 --phase-bins 32 \
  --output synthetic_cupy_n19_p32.json
```

The verifier binds the frozen SILWARP-v2 hashes, rebuilds every ledger, checks
all rate/read caps, checks the N19 CUDA receipt and confirms the decisive
scalar-bridge negative gate.  Passing it is not a Qwen compression claim.
