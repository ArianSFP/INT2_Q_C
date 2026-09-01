# BiSCo-LLM / two-stage BSQ red-team assessment

## Decision

Two-stage nonlinear binary spherical coding is **not strictly covered** by the
completed additive-VQ or nonlinear-flow screens.  It is therefore a legitimate
untested member of the one remaining broad loophole: a shared learned weight
prior/decoder that generalizes to whole held-out Qwen experts.

It is not an evidence-backed survivor.  The primary
[BiSCo-LLM paper](https://arxiv.org/abs/2607.08643) reports perplexity and task
accuracy after protected channels and recovery distillation, not codec-only
pooled source-relative MSE.  It gives the abstract storage equation but does
not report the decoder widths, decoder parameter bytes, per-component physical
byte ledger, or a raw-MSE rate-distortion table needed for this target.  Its
experimental section also describes planned/missing evaluations, and its
main comparison omits the bit-width column.  Those functional results cannot
establish

```text
D <= 0.8 * 2^(-2R),  2.15 <= R <= 2.5.
```

No GPU experiment was run for this assessment while the sealed run was active.
The protocol below is the cheapest favorable experiment worth running after
that exclusion ends.  `bisco_bsq_ledger.py` recomputes both exact ledger
tables using only the Python standard library.  The no-peeking rules and stop
thresholds are serialized in `bisco_protocol_freeze.json` before any result.

## What the codec is in source-MSE terms

For a chunk `x in R^d`, each stage stores a binary vector and reconstructs it
with a category-shared nonlinear decoder:

```text
q1 = sign(E1(x)) / sqrt(b1)
r1 = D1(q1)
q2 = sign(E2(x-r1)) / sqrt(b2)
x_hat = D1(q1) + D2(q2).
```

Normalizing the encoder output before taking its sign does not change the sign
pattern.  Thus the spherical step is an optimization regularizer; it adds no
representational capacity to the stored code.  For fixed decoders, the
reconstruction set is the sum of two implicit codebooks,

```text
{D1(q1)} + {D2(q2)},  qs in {-1,+1}^bs.
```

This is an additive VQ with exponentially large *implicit* stage codebooks.
The decoder parameters replace explicit centroid tables.

## Why existing negative probes do not constitute a bound

- The additive-VQ screen learned real binary/ternary/quaternary component
  codebooks at `d=8/16/32`.  Its reconstruction is additive across many tiny
  alphabets.  It does not contain an arbitrary nonlinear map of all `b` bits,
  so it is not a superset of a `2^b` implicit codebook.
- The nonlinear affine-flow MLP predicted conditional scalar moments from
  neighboring weights.  It was not a discrete analysis/synthesis transform
  and did not optimize a same-dimensional binary latent.
- The 256-to-64 hyperdecoder tested a low-dimensional whole-expert manifold,
  not a local `d -> b -> d` implicit vector codebook.

Those failures are strong prior evidence against a large gain, especially
when combined with the near-zero nonparametric, ICA, bitplane, hyperprior, and
shared-expert results, but none is a theorem for this architecture.

## Favorable matched-Gaussian oracle

### Data firewall

1. Freeze whole `(layer, expert)` folds before fitting anything.
2. Fit one pair of stage decoders per MLP role (`gate`, `up`, `down`) on the
   remaining Qwen experts.  No test layer or test expert may update a decoder,
   normalization table, stopping rule, or hyperparameter.
3. Fit an identical codec, with identical seeds, batches, updates, widths, and
   stopping rule, to iid Gaussian chunks matched to the training role moments.
4. The compressor may optimize the held-out binary codes because those bits
   are stored.  It may not update the decoder.  After encoder inference, grant
   both domains the same favorable greedy bit-flip refinement.
5. Score complete held-out matrices in original coordinates.  Store an FP16
   mean and RMS per matrix (72 bytes for the 18-matrix panel) rather than
   treating target moments as free.

### Minimal decoder

For each role and stage use a stored FP16 decoder

```text
Linear(bs, h) -> SiLU -> Linear(h, d)
```

and a mirrored training-only encoder.  Encoder bytes are not deployed.  With
`B=b1+b2`, the six deployed decoders contain exactly

```text
P = 3*h*(B + 2*d + 2) + 6*d
```

FP16 parameters, including biases.  A 256-byte fixed header records shapes,
offsets, hashes, and the packing convention.

### Exact favorable production ledger

The following ledger amortizes a role-shared decoder over only 128 experts,
even though BiSCo proposes category sharing across layers as well.  Each stage
gets half the latent bits and the code stream uses 2.25 bpw.  Every requested
expert cold-loads the complete decoder plus its own packed codes and three
FP16 `(mean,RMS)` pairs.

| d | h | b1+b2 | Decoder params | Decoder bytes | Physical R | Cold expert bytes | Cold read amp | Minimum matched `s` even with ideal Gaussian coding |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 18+18 | 13,536 | 27,072 | 2.2503823 | 1,354,444 | 1.02043x | 0.1613464 |
| 32 | 128 | 36+36 | 53,184 | 106,368 | 2.2514326 | 1,433,740 | 1.07966x | 0.1623967 |
| 64 | 256 | 72+72 | 210,816 | 421,632 | 2.2556085 | 1,749,004 | 1.31463x | 0.1665725 |

Here one expert has 4,718,592 weights and a 2.25-bpw code stream is exactly
1,327,104 bytes.  The attributed physical bytes in the read-amplification
denominator include `1/128` of the shared decoder/header and the expert's 12
scale bytes.  The numerator includes the full decoder/header once.  All rows
are below the strict `2x` external compressed-read limit.

The last column is stronger than the generic `0.1609640474` requirement.  If
`r_side` physical bpw is spent on a decoder rather than held-out per-chunk
codes, then even an ideal Gaussian code at `R_code` incurs

```text
s_absolute <= s_Qwen-vs-Gaussian - r_side.
```

Thus the matched source advantage must be at least
`0.1609640474 + r_side`; any finite-dimensional Gaussian implementation gap
raises the bar further.

For a self-contained six-expert artifact rather than a production category,
the decoder amortization is much harsher.  Closest-to-2.25 shallow rows have:

| d | h | b1+b2 | Physical R | Cold read amp | Required matched `s` |
|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 18+18 | 2.2577424 | 1.01710x | 0.1687065 |
| 32 | 128 | 36+35 | 2.2486821 | 1.06651x | 0.1908961 |
| 64 | 256 | 69+68 | 2.2568201 | 1.25739x | 0.2771592 |

This second table charges the whole decoder and header over the six test
experts, and is the ledger to use for a serialized six-expert oracle artifact.

### Early-kill sequence

1. Run `d=16`, no residual MLP block, on auxiliary folds and its matched
   Gaussian control.  This has the smallest decoder tax and the closest
   required matched advantage to the target.
2. At 25% of the fixed update budget, stop the family if the best
   held-out, control-adjusted `s_match + 2 SE` is below `0.08` and has improved
   by less than `0.01` over the latter half of training.  Do not open the
   pinned panel after that gate.
3. If it survives, finish `d=16` and run shallow `d=32` and `d=64`.  Promote
   only if every whole-expert fold is positive and pooled
   `s_absolute >= 0.1609640474` from serialized physical bytes.
4. Test one hidden residual block only after a shallow row reaches at least
   `s_match=0.12`.  At `d=64,h=256`, one block would push a six-expert artifact
   to about 2.70 bpw and 2.04x reads, so that cell is analytically illegal.

The result JSON must report `D_Qwen`, `D_Gaussian`,
`s_match=-0.5*log2(D_Qwen/D_Gaussian)`, the Gaussian operational gap, physical
`R`, and `s_absolute=-0.5*log2(D_Qwen*2^(2R))` separately.  Perplexity,
activation reconstruction, protected-channel, or distillation improvements
must not be substituted for these quantities.

## Evidence-weighted forecast

The only reason to run this bounded oracle is that it tests a genuinely
different implicit codebook.  The evidence-weighted expectation is still an
early kill:

- BiSCo reports no codec-only raw-MSE number.
- Its own 64-bit occupancy diagnostic assigns 524,286 distinct codes to
  524,288 chunks.  The observation that this finite sample occupies a tiny
  fraction of `2^64` is arithmetic, not evidence that the stored 64-bit stream
  is compressible: the map is almost injective.  Without a low-cost model that
  predicts unseen codes or decoded vectors, held-out decoder generalization,
  not nominal code-space cardinality, remains the bottleneck.
- NWC, the closest published shared nonlinear weight codec, reports that its
  learned transform slightly worsens raw MSE relative to its scalar baseline
  while helping perplexity.
- The strongest source-specific opportunity found locally is the favorable
  NanoQuant cut-factor screen at only `s=0.04957`, and the strongest earlier
  whole-matrix structural oracle is `s=0.03663`.

Accordingly, BiSCo is a **bounded test candidate**, not a GPU survivor and not
evidence that the 20%-below-Gaussian goal is reachable.

## Serving caveat

The read ledger above counts compressed external bytes.  BiSCo's paper states
that its present inference path materializes decoded weights and that direct
binary-code computation or fused load-time decoding remains future work.
Consequently, the architecture can satisfy the external `<2x` byte rule while
still causing large internal HBM traffic and decoder compute.  A positive
source-MSE result would need a second, separate fused decoder-GEMM audit.
