# Same-layer conditional entropy on Qwen experts

Date: 2026-09-03

Panel: Qwen layer 15, experts `0,8,...,120`, Up and transposed Down

Method: CBIB-1 r3, fixed nearest-quantizer labels

Boundary: ideal label-entropy census; not an emitted codec and not an MSE result

## Answer

Yes.  Same-layer conditional entropy was tested on authenticated Qwen weights.
The fixed-label CBIB model is a hard kill.  It met the routed-read constraint,
but it found essentially no net joint source advantage after transmitting the
common labels and charging all model and framing fields.

The preserved producer result has SHA-256:

```text
e24d8795c655704732a42b2fb6e39ca323c2cc09d0d0e5cf34a070de9ef5b916
```

The child completed and emitted the result; the outer wrapper reported failure
only because a harmless CuPy `CUDA path could not be detected` warning made
stderr nonempty.  A detached source-only audit authenticated the six execution
files and independently recomputed the scientific result.  It sealed verdict
`PASS_COMPLETED_CHILD_RESULT_WITH_HARMLESS_STDERR_WARNING__HARD_KILL_CBIB_FIXED_LABEL`
at manifest
`c200effe602dcb4fb87a84787ebed0acc35cc5e62c44de7e890da67107822ec5`
and root
`6a23ea1d1a8760bb25eb3633d4943fe27f5e0bad632e1044c194b7bf553c9ada`.

## Exact result

The aperture contains 16 experts, two matrix roles, and 50,331,648 weights.
It authenticated and read 32 BF16 files once (100,663,296 bytes), with logical
host-scan amplification `1.0`.  Eight marginal-preserving controls were run for
the only group size that passed the gross pre-control gate.

| Experts per common group | Gross private saving (bpw) | Ideal net gain (bpw) | Charged gain (bpw) | Max read at 2.15 bpw | Max read at 2.5 bpw |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.45979510004777663 | 0.000010730760043135371 | -0.00357806490188306 | 1.4193741857450828x | 1.622032524456343x |
| 4 | 0.22695975151121317 | 0.000019602605729492534 | -0.002238658439090694 | 1.6639946411283693x | 1.9154427202322117x |
| 8 | 0.1133477942723428 | 0.00002533066769651778 | -0.0015668683399736594 | 1.6636642602150957x | 1.888126898498085x |
| 16 | 0.056627428608557885 | 0.00003083509623766356 | -0.0012271010635321507 | 1.7047980859964635x | 1.9294613996549124x |

The target for an Up-plus-Down-only entropy route is
`0.22933495044437174 bpw`.  Group 2 exceeded that threshold only in the invalid
gross-private view.  Its common-label stream restored almost the entire saving,
leaving approximately `1.07e-5 bpw` ideally and a loss after physical charges.
The eight control charged gains range from approximately `-0.00359832` to
`-0.00359526 bpw`; the reported control-corrected source gain remains
`-0.00357806490188306 bpw`.

All read figures above are capacity/page-layout envelopes, not an emitted byte
stream or measured accelerator-HBM trace.  They nevertheless show that this
branch failed for lack of source advantage, not because common pages necessarily
violate the `<2x` cold-read requirement.

## What the result means

Let `Z` be a common label stream and `Q_e` expert-private labels.  A gross report
of only `sum_e H(Q_e) - sum_e H(Q_e | Z)` counts the conditional saving but omits
the cost of communicating `Z`.  A valid joint description must charge

```text
H(Z) + sum_e H(Q_e | Z).
```

The Qwen result shows that the apparent pairwise redundancy is almost exactly
paid back by `H(Z)`.  Larger clusters reduce the gross saving and remain
net-negative when finite fields are charged.  Extending this same frozen-label
model to more experts is therefore not justified.

## What remains open

This result does **not** close same-layer *rate-distortion* coding.  CBIB encoded
the labels already selected by the current nearest quantizer.  A flexible-label
encoder can instead choose among nearby legal levels jointly:

```text
argmin_(q_1,q_2)  D(x_1,q_1) + D(x_2,q_2)
                 + lambda * L(q_1,q_2).
```

That mechanism may deliberately coordinate cheap near-boundary rounding choices
into a shorter pair description.  It must be tested as one nested physical
rate-distortion calculation, with the identical label search refitted on
moment-matched Gaussian controls.  A negative fixed-label entropy census alone
cannot decide it.

The next bounded experiment is therefore PAIRPATH r2: a literal two-expert
encoder/decoder with flexible legal labels, exact source-domain distortion,
finite model bytes, complete control refits, and a `<2x` expert-local read
ledger.  It will be hard-killed before a large payload run unless a small
dominant oracle shows material source-specific advantage.

## Flexible-pair early-kill bound

For any fixed pair-label assignment, lossless joint coding can save at most

```text
I(A; B) / 2 bpw
```

over separate marginal coding because each coordinate pair represents two
weights.  The standalone Up-plus-Down target of
`0.22933495044437174 bpw` therefore requires at least
`0.4586699008887435 bits` of mutual information per coordinate pair before
overhead.  A distortion penalty raises that requirement further.

The first local-RTX-3060 payload gate will intentionally grant the joint model
free empirical probabilities and ignore locality, while giving the independent
baseline the same flexible-label optimization.  Both rate-distortion frontiers
must be convexified and compared at equal rate or equal distortion.  Define

```text
G_eq,UD = R_ind - R_pair + 0.5 * log2(D_ind / D_pair).
```

Promotion bands are:

| `G_eq,UD` | Decision |
|---:|---|
| `< 0.045 bpw` | hard kill memoryless flexible-pair coding |
| `0.045--0.170340945 bpw` | measurable but insufficient |
| `0.170340945--0.22933495044437174 bpw` | eligible only for one nested composite |
| `>= 0.22933495044437174 bpw` | ideal standalone target |
| `>= 0.27 bpw` | enough margin to justify a finite packet |

Only a source survivor earns iid-Gaussian, covariance-matched bivariate
Gaussian, and within-stratum expert-permutation refits.  A full-joint oracle is
optimistic about reads: if it fails, every owner-local `<2x` common/private
realization fails; if it passes, locality and physical bytes remain separate
requirements.
